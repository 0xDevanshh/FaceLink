"""Search-provider resilience and lifecycle.

The property under test is isolation: whatever one provider does — succeed,
find nothing, get CAPTCHA'd, get throttled, raise, hang forever, or not be
configured at all — the search stage must still terminate and still report the
other providers' real outcomes. A scan that hangs because one engine hung is
the failure mode this file exists to prevent.

Adapters are stubbed here so the tests are deterministic and touch no network;
the live behaviour is exercised by the real end-to-end run.
"""

from __future__ import annotations

import time

import pytest

from facechain.config import settings
from facechain.models import ProviderStatus
from facechain.search import orchestrator
from facechain.search.base import EngineResult, build_candidates, classify_block
from facechain.search.orchestrator import classify_error, count_by_platform, run_reverse_search


def _result(engine: str, hrefs: list[str] | None = None, **kw) -> EngineResult:
    cands = build_candidates(engine, [{"href": h, "text": ""} for h in (hrefs or [])])
    return EngineResult(engine, candidates=cands, ok=bool(cands), **kw)


@pytest.fixture
def stub_adapters(monkeypatch):
    """Replace the adapter registries with callables we control."""
    behaviours: dict[str, callable] = {}

    def runner(name, image_path, public_url):
        return behaviours[name]()

    monkeypatch.setattr(orchestrator, "BROWSER_ADAPTERS", {})
    monkeypatch.setattr(orchestrator, "API_ADAPTERS", behaviours)
    monkeypatch.setattr(orchestrator, "_run_api_engine", runner)
    return behaviours


# ---- error/status classification ----------------------------------------

@pytest.mark.parametrize(
    "error,expected",
    [
        ("bot challenge interstitial (/sorry/)", ProviderStatus.CHALLENGED),
        ("CAPTCHA / bot challenge", ProviderStatus.CHALLENGED),
        ("engine flagged unusual traffic", ProviderStatus.CHALLENGED),
        ("access denied", ProviderStatus.CHALLENGED),
        ("too many requests", ProviderStatus.RATE_LIMITED),
        ("HTTP 429 rate limit exceeded", ProviderStatus.RATE_LIMITED),
        ("TimeoutError: navigation timed out", ProviderStatus.TIMEOUT),
        ("SERPAPI_KEY not set", ProviderStatus.NOT_CONFIGURED),
        ("no outbound result links found", ProviderStatus.NO_RESULTS),
    ],
)
def test_error_text_maps_to_the_right_status(error, expected):
    assert classify_error(error) == expected


def test_unrecognised_error_is_failed_not_no_results():
    """An error we cannot read means we do not know the search ran.

    Reporting NO_RESULTS would assert "searched, found nothing", which is a
    stronger claim than the evidence supports.
    """
    assert classify_error("KaboomError: something we have never seen") == ProviderStatus.FAILED
    assert classify_error("") == ProviderStatus.FAILED


def test_rate_limit_is_distinguished_from_a_bot_challenge():
    assert classify_block("rate limited") == ProviderStatus.RATE_LIMITED
    assert classify_block("CAPTCHA / bot challenge") == ProviderStatus.CHALLENGED


# ---- one provider's outcome never decides another's ----------------------

def test_mixed_success_and_failure_still_completes(stub_adapters):
    stub_adapters.update({
        "good": lambda: _result("good", ["https://instagram.com/p/A/"], status=ProviderStatus.COMPLETED),
        "challenged": lambda: EngineResult("challenged", ok=False, error="CAPTCHA",
                                           status=ProviderStatus.CHALLENGED),
        "boom": lambda: (_ for _ in ()).throw(RuntimeError("adapter exploded")),
        "empty": lambda: EngineResult("empty", ok=False, error="no outbound result links found",
                                      status=ProviderStatus.NO_RESULTS),
    })
    report, _ = run_reverse_search("x.jpg", engines=["good", "challenged", "boom", "empty"])

    statuses = {p.engine: p.status for p in report.providers}
    assert statuses["good"] == ProviderStatus.COMPLETED
    assert statuses["challenged"] == ProviderStatus.CHALLENGED
    assert statuses["boom"] == ProviderStatus.FAILED
    assert statuses["empty"] == ProviderStatus.NO_RESULTS
    # The successful provider's candidates survived the others' failures.
    assert report.total_candidates == 1


def test_a_raising_adapter_cannot_abort_the_stage(stub_adapters):
    stub_adapters.update({
        "boom": lambda: (_ for _ in ()).throw(ValueError("bad parse")),
        "good": lambda: _result("good", ["https://github.com/someone"]),
    })
    report, _ = run_reverse_search("x.jpg", engines=["boom", "good"])
    assert report.total_candidates == 1
    assert report.provider("boom").status == ProviderStatus.FAILED
    assert "ValueError" in report.provider("boom").error


def test_every_provider_reaches_a_terminal_state(stub_adapters):
    stub_adapters.update({
        "a": lambda: _result("a", ["https://example.com/x"]),
        "b": lambda: EngineResult("b", ok=False, error="CAPTCHA", status=ProviderStatus.CHALLENGED),
    })
    report, _ = run_reverse_search("x.jpg", engines=["a", "b"])
    assert len(report.providers) == 2
    assert all(p.status.terminal for p in report.providers)


def test_all_providers_unavailable_yields_an_honest_empty_report(stub_adapters):
    stub_adapters.update({
        "a": lambda: EngineResult("a", ok=False, error="SERPAPI_KEY not set",
                                  status=ProviderStatus.NOT_CONFIGURED),
        "b": lambda: EngineResult("b", ok=False, error="CAPTCHA", status=ProviderStatus.CHALLENGED),
    })
    report, _ = run_reverse_search("x.jpg", engines=["a", "b"])
    assert report.total_candidates == 0
    assert report.candidates == []
    # No provider succeeded, and the report says so rather than inventing hits.
    assert report.engines_succeeded == []
    assert {p.status for p in report.providers} == {
        ProviderStatus.NOT_CONFIGURED, ProviderStatus.CHALLENGED}


def test_unknown_engine_is_reported_not_ignored(stub_adapters):
    stub_adapters["a"] = lambda: _result("a", ["https://example.com/x"])
    report, _ = run_reverse_search("x.jpg", engines=["a", "no_such_engine"])
    assert report.provider("no_such_engine").status == ProviderStatus.FAILED
    assert report.provider("no_such_engine").error == "unknown engine"
    assert report.total_candidates == 1


# ---- hangs -------------------------------------------------------------

def test_a_stuck_provider_cannot_stall_the_stage(stub_adapters, monkeypatch):
    """The whole point: a provider that never returns must be abandoned.

    The stage budget is squeezed to a second so the test is fast; the mechanism
    is the same one that bounds a real wedged browser.
    """
    monkeypatch.setattr(settings, "search_total_timeout_s", 1)

    stub_adapters.update({
        "fast": lambda: _result("fast", ["https://linkedin.com/in/someone"]),
        "stuck": lambda: (time.sleep(30), _result("stuck"))[1],
    })

    started = time.monotonic()
    report, _ = run_reverse_search("x.jpg", engines=["fast", "stuck"])
    elapsed = time.monotonic() - started

    assert elapsed < 15, f"stage waited {elapsed:.1f}s on a stuck provider"
    assert report.timed_out
    assert report.provider("stuck").status == ProviderStatus.TIMEOUT
    # And the healthy provider's work was still collected.
    assert report.provider("fast").status == ProviderStatus.COMPLETED
    assert report.total_candidates == 1


def test_falsy_engine_list_falls_back_to_the_configured_engines(stub_adapters):
    """`engines=[]`/`None` means "use ENGINES from config", not "search nothing".

    The stubs registered here replace the adapter tables, so the configured
    names resolve to nothing and every one is reported as unknown — which is
    the point: the stage still terminates and still says what it tried.
    """
    report, _ = run_reverse_search("x.jpg", engines=[])
    assert report.engines_attempted == settings.engine_list
    assert report.total_candidates == 0
    assert all(p.status.terminal for p in report.providers)


# ---- events ------------------------------------------------------------

def test_events_name_the_terminal_status(stub_adapters):
    stub_adapters.update({
        "good": lambda: _result("good", ["https://x.com/u/status/1"]),
        "bad": lambda: EngineResult("bad", ok=False, error="CAPTCHA", status=ProviderStatus.CHALLENGED),
    })
    events: list[tuple[str, str, str]] = []
    run_reverse_search("x.jpg", engines=["good", "bad"],
                       on_event=lambda *a: events.append(a))

    assert ("good", "start", "") in events
    assert ("bad", "start", "") in events
    # A UI must be able to tell CHALLENGED from a generic failure.
    assert any(e[0] == "bad" and "CHALLENGED" in e[2] for e in events)
    assert any(e[0] == "good" and "COMPLETED" in e[2] for e in events)


# ---- platform tallies --------------------------------------------------

def test_platform_counts_include_explicit_zeros():
    cands = build_candidates("yandex", [
        {"href": "https://instagram.com/p/A/", "text": ""},
        {"href": "https://example.com/page", "text": ""},
    ])
    counts = count_by_platform(cands)
    assert counts["Instagram"] == 1
    assert counts["Other Web"] == 1
    # Looked for and not found — reported as 0, never omitted.
    for absent in ("LinkedIn", "X/Twitter", "GitHub", "YouTube"):
        assert counts[absent] == 0


def test_recognised_but_unprioritised_platform_counts_as_other_web():
    """Facebook is recognised but not a priority target, so it is not given a
    column of its own in the priority tally."""
    cands = build_candidates("yandex", [{"href": "https://facebook.com/story.php?id=1", "text": ""}])
    counts = count_by_platform(cands)
    assert counts.get("Facebook", 0) == 1
    assert counts["LinkedIn"] == 0
