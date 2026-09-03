"""Lazy, single, capability-aware temporary publication (search/orchestrator.py).

The property under test: the central temporary-hosting step (Litterbox by
default) is only ever invoked when at least one selected engine genuinely
benefits from it, invoked at most once per scan regardless of how many
engines need the result, and — critically — a real publication failure must
be reported as FAILED with the real cause, never conflated with "never
configured".

All offline. `API_ADAPTERS`/`BROWSER_ADAPTERS` are replaced with small fake
adapters carrying real capability attributes (`requires_public_url`,
`has_reliable_upload_alternative`), not raw result-producing stubs — unlike
`test_providers.py`'s fixtures, these must behave like real adapter objects
because the orchestrator now reads capability flags off them directly.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from facechain.config import settings
from facechain.models import ProviderStatus
from facechain.search import orchestrator
from facechain.search.base import EngineResult
from facechain.search.orchestrator import run_reverse_search
from facechain.search.uploader import UploadError


class FakeAdapter:
    def __init__(self, name, requires_public_url=False, has_reliable_upload_alternative=False,
                result=None):
        self.name = name
        self.requires_public_url = requires_public_url
        self.has_reliable_upload_alternative = has_reliable_upload_alternative
        self._result = result
        self.calls: list[tuple[str, str | None]] = []

    def search(self, image_path, image_url=None):
        self.calls.append((image_path, image_url))
        return self._result or EngineResult(self.name, ok=True, status=ProviderStatus.COMPLETED)


@pytest.fixture
def fake_registry(monkeypatch):
    """Replace API_ADAPTERS with fakes we construct per-test, BROWSER_ADAPTERS
    emptied so only the fakes under test run."""
    registry: dict[str, FakeAdapter] = {}
    monkeypatch.setattr(orchestrator, "BROWSER_ADAPTERS", {})
    monkeypatch.setattr(orchestrator, "API_ADAPTERS", {
        name: (lambda n=name: registry[n]) for name in ()
    })
    return registry


def _register(monkeypatch, registry: dict[str, FakeAdapter], adapter: FakeAdapter):
    registry[adapter.name] = adapter
    monkeypatch.setattr(
        orchestrator, "API_ADAPTERS",
        {**orchestrator.API_ADAPTERS, adapter.name: (lambda n=adapter.name: registry[n])},
    )


# ---- Requirement 4/16: FAILED, not NOT_CONFIGURED, on a real upload failure -

def test_a_url_only_adapter_reports_failed_with_the_real_reason_when_upload_fails(
    fake_registry, monkeypatch,
):
    monkeypatch.setattr(settings, "allow_upload_host", True)
    _register(monkeypatch, fake_registry, FakeAdapter("serpapi_yandex", requires_public_url=True))
    monkeypatch.setattr(orchestrator, "publish_temporarily",
                       lambda path: (_ for _ in ()).throw(UploadError("host returned 403: Forbidden")))

    report, _ = run_reverse_search("x.jpg", engines=["serpapi_yandex"])

    provider = report.provider("serpapi_yandex")
    assert provider.status == ProviderStatus.FAILED
    assert "temporary image publication failed" in provider.error
    assert "403" in provider.error
    # The adapter's own (misleading, in this case) search() must never even
    # run — the failure is already known before it would be called.
    assert fake_registry["serpapi_yandex"].calls == []


def test_the_url_only_adapter_still_runs_normally_once_upload_succeeds(fake_registry, monkeypatch):
    monkeypatch.setattr(settings, "allow_upload_host", True)
    _register(monkeypatch, fake_registry, FakeAdapter("serpapi_yandex", requires_public_url=True))
    monkeypatch.setattr(orchestrator, "publish_temporarily", lambda path: "https://host.example/x.jpg")

    report, public_url = run_reverse_search("x.jpg", engines=["serpapi_yandex"])

    assert public_url == "https://host.example/x.jpg"
    assert fake_registry["serpapi_yandex"].calls == [("x.jpg", "https://host.example/x.jpg")]
    assert report.provider("serpapi_yandex").status == ProviderStatus.COMPLETED


# ---- Requirement 7: lazy publication ---------------------------------------

def test_publication_is_skipped_when_the_only_engine_has_a_reliable_alternative(
    fake_registry, monkeypatch,
):
    """Google Lens via SerpAPI has its own first-party upload path — central
    hosting must never be attempted just for it."""
    monkeypatch.setattr(settings, "allow_upload_host", True)
    _register(monkeypatch, fake_registry, FakeAdapter(
        "serpapi_google_lens", requires_public_url=False, has_reliable_upload_alternative=True))
    upload_mock = MagicMock(side_effect=AssertionError("publish_temporarily must not be called"))
    monkeypatch.setattr(orchestrator, "publish_temporarily", upload_mock)

    events = []
    report, public_url = run_reverse_search(
        "x.jpg", engines=["serpapi_google_lens"], on_event=lambda *a: events.append(a))

    upload_mock.assert_not_called()
    assert public_url is None
    assert any(e[0] == "host" and e[1] == "skip" for e in events)


def test_publication_still_happens_when_a_genuinely_url_only_engine_is_selected(
    fake_registry, monkeypatch,
):
    monkeypatch.setattr(settings, "allow_upload_host", True)
    _register(monkeypatch, fake_registry, FakeAdapter("serpapi_yandex", requires_public_url=True))
    upload_mock = MagicMock(return_value="https://host.example/x.jpg")
    monkeypatch.setattr(orchestrator, "publish_temporarily", upload_mock)

    run_reverse_search("x.jpg", engines=["serpapi_yandex"])

    upload_mock.assert_called_once()


# ---- Requirement 6: at most one publication, reused across every provider --

def test_publication_happens_once_and_is_reused_across_two_url_only_providers(
    fake_registry, monkeypatch,
):
    monkeypatch.setattr(settings, "allow_upload_host", True)
    _register(monkeypatch, fake_registry, FakeAdapter("serpapi_yandex", requires_public_url=True))
    _register(monkeypatch, fake_registry, FakeAdapter("serpapi_bing", requires_public_url=True))
    upload_mock = MagicMock(return_value="https://host.example/shared.jpg")
    monkeypatch.setattr(orchestrator, "publish_temporarily", upload_mock)

    run_reverse_search("x.jpg", engines=["serpapi_yandex", "serpapi_bing"])

    upload_mock.assert_called_once()
    assert fake_registry["serpapi_yandex"].calls == [("x.jpg", "https://host.example/shared.jpg")]
    assert fake_registry["serpapi_bing"].calls == [("x.jpg", "https://host.example/shared.jpg")]


# ---- Requirement 8: an existing image_url is never re-published -----------

def test_an_existing_image_url_is_used_directly_without_publishing(fake_registry, monkeypatch):
    monkeypatch.setattr(settings, "allow_upload_host", True)
    _register(monkeypatch, fake_registry, FakeAdapter("serpapi_yandex", requires_public_url=True))
    upload_mock = MagicMock(side_effect=AssertionError("must not publish when image_url is given"))
    monkeypatch.setattr(orchestrator, "publish_temporarily", upload_mock)

    report, public_url = run_reverse_search(
        "x.jpg", engines=["serpapi_yandex"], image_url="https://already-hosted.example/x.jpg")

    upload_mock.assert_not_called()
    assert public_url == "https://already-hosted.example/x.jpg"
    assert fake_registry["serpapi_yandex"].calls == [("x.jpg", "https://already-hosted.example/x.jpg")]
