"""TinEyeAdapter regression tests.

RCA this file pins down: TinEye computes a search server-side and then
client-navigates to a content-addressed results permalink
(``/search/<hash>?sort=...``) instead of rendering results in place on
``/search?url=...``. A fixed sleep alone races that navigation, and under any
extra latency the marker check used to run before the redirect landed,
misreporting a real-but-slow search as "results view not reached". This
mostly went unnoticed while a public `image_url` was rarely available (so
TinEye almost always ran in upload mode); once central image hosting became
reliable, by-url mode — and this race — started firing far more often.

None of the browser adapters have a Playwright-page-level test harness
(see `tests/test_providers.py`'s note that adapter internals are exercised
live), so this file adds a minimal, deterministic fake `Page`/`session`
covering exactly what `TinEyeAdapter.search()` and the shared
`browser.settle`/`base.collect_results`/`base.harvest_anchors` helpers it
calls actually touch. No real browser, no network.
"""

from __future__ import annotations

import contextlib
from urllib.parse import quote

from facechain.models import ProviderStatus
from facechain.search.tineye import BY_URL, RESULTS_URL_PATTERN, TinEyeAdapter


class _FakeMouse:
    def wheel(self, *a, **kw):
        pass


class _FakePage:
    """Stand-in for a Playwright Page.

    `states` maps a URL to `{"body": str, "anchors": list[dict]}`.
    `redirect_to`, when set, is where the page "asynchronously" navigates to
    on its own — modelling TinEye's real client-side redirect — independent
    of whether `wait_for_url`'s pattern happens to match it, exactly like a
    real page navigating regardless of what a caller is waiting for.
    """

    def __init__(self, *, states: dict[str, dict], start_url: str,
                 redirect_to: str | None = None):
        self.url = start_url
        self._states = states
        self._redirect_to = redirect_to
        self.mouse = _FakeMouse()

    def goto(self, url, wait_until=None):
        self.url = url

    def wait_for_load_state(self, state=None, timeout=None):
        pass

    def wait_for_timeout(self, ms):
        pass

    def wait_for_url(self, pattern, timeout=None):
        if self._redirect_to is None:
            raise TimeoutError("no navigation happened")
        self.url = self._redirect_to
        if not pattern.search(self._redirect_to):
            raise TimeoutError("redirect target does not match the awaited pattern")

    def inner_text(self, selector):
        return self._states.get(self.url, {}).get("body", "")

    def query_selector(self, selector):
        return None  # force the whole-page harvest fallback, same as a real
        # page whenever none of the known result-container selectors matches.

    def evaluate(self, js, node):
        return self._states.get(self.url, {}).get("anchors", [])


class _FakeSession:
    def __init__(self, page: _FakePage) -> None:
        self._page = page

    @contextlib.contextmanager
    def page(self):
        yield self._page


IMAGE_URL = "https://cdn.example.com/tmp/abc123.jpg"
START_URL = BY_URL.format(url=quote(IMAGE_URL, safe=""))
RESULTS_URL = "https://tineye.com/search/9f8e7d6c5b4a3210?sort=score&order=desc&page=1"


def test_results_url_pattern_matches_a_real_tineye_permalink():
    assert RESULTS_URL_PATTERN.search(RESULTS_URL)
    assert not RESULTS_URL_PATTERN.search(START_URL)


# ---- the redirect race this file exists to fix -----------------------------

def test_by_url_search_waits_for_the_results_redirect_then_parses_candidates():
    states = {
        # The initial `/search?url=...` page: TinEye is still computing the
        # search server-side, so there is nothing to see here yet.
        START_URL: {"body": "", "anchors": []},
        # The real results permalink it redirects to once ready.
        RESULTS_URL: {
            "body": "20 results\nTinEye searched 85.6 billion images for: photo.jpg",
            "anchors": [
                {"href": "https://en.wikipedia.org/wiki/Someone", "text": "Someone",
                 "thumb": "https://upload.wikimedia.org/full-res.jpg"},
                # A duplicate of the row above (trailing slash) must not
                # produce a second candidate.
                {"href": "https://en.wikipedia.org/wiki/Someone/", "text": "dup",
                 "thumb": "https://upload.wikimedia.org/should-not-win.jpg"},
                {"href": "https://news.example.com/article", "text": "An article",
                 "thumb": "https://news.example.com/photo.jpg"},
            ],
        },
    }
    page = _FakePage(states=states, start_url=START_URL, redirect_to=RESULTS_URL)
    adapter = TinEyeAdapter(_FakeSession(page))

    result = adapter.search("unused/local/path.jpg", image_url=IMAGE_URL)

    assert result.ok is True
    assert result.status == ProviderStatus.COMPLETED
    assert result.query_mode == "by-url"
    urls = [c.url for c in result.candidates]
    assert urls == ["https://en.wikipedia.org/wiki/Someone", "https://news.example.com/article"]
    # First-seen row wins on a duplicate; original/full-res image preserved.
    assert result.candidates[0].thumbnail == "https://upload.wikimedia.org/full-res.jpg"


def test_a_slow_redirect_still_succeeds_within_the_bounded_wait():
    """Same as above, just confirming the fix is the wait itself, not a
    coincidence of timing — the fake page never "arrives" until
    `wait_for_url` is awaited, unlike the old fixed-sleep-only code path."""
    states = {
        START_URL: {"body": "", "anchors": []},
        RESULTS_URL: {
            "body": "1 result\nmatches found",
            "anchors": [{"href": "https://example.com/x", "text": "x", "thumb": ""}],
        },
    }
    page = _FakePage(states=states, start_url=START_URL, redirect_to=RESULTS_URL)
    adapter = TinEyeAdapter(_FakeSession(page))

    result = adapter.search("unused.jpg", image_url=IMAGE_URL)

    assert result.ok is True
    assert len(result.candidates) == 1


# ---- TinEye's own "could not fetch that URL" error -------------------------

def test_url_fetch_failure_is_reported_precisely_not_as_a_layout_change():
    """A real, specific TinEye failure (our temp-hosted URL expired/was
    unreachable by the time TinEye's server tried it) must be named for what
    it is, not folded into the generic 'results view not reached' message."""
    states = {
        START_URL: {
            "body": (
                "Upload an image\nOops, something didn't work!\n\n"
                "TinEye could not read that image url. This may be due to an "
                "unsupported file format."
            ),
            "anchors": [],
        },
    }
    # No redirect: TinEye never got far enough to compute a search.
    page = _FakePage(states=states, start_url=START_URL, redirect_to=None)
    adapter = TinEyeAdapter(_FakeSession(page))

    result = adapter.search("unused.jpg", image_url=IMAGE_URL)

    assert result.ok is False
    assert "could not fetch the supplied image url" in result.error.lower()
    assert "layout" not in result.error.lower()
    assert "results view not reached" not in result.error.lower()


# ---- a genuine block must never be reported as success ---------------------

def test_a_bot_challenge_is_reported_as_challenged_not_as_a_successful_search():
    block_url = "https://tineye.com/sorry/index?continue=abc"
    states = {
        START_URL: {"body": "", "anchors": []},
        block_url: {"body": "please verify you are human", "anchors": []},
    }
    # The page navigates to TinEye's block page on its own, same as it would
    # navigate to the results permalink when not blocked — our specific
    # RESULTS_URL_PATTERN wait times out either way, which must not be
    # mistaken for "no results".
    page = _FakePage(states=states, start_url=START_URL, redirect_to=block_url)
    adapter = TinEyeAdapter(_FakeSession(page))

    result = adapter.search("unused.jpg", image_url=IMAGE_URL)

    assert result.ok is False
    assert result.status == ProviderStatus.CHALLENGED
    assert result.candidates == []


# ---- upload mode is unaffected ---------------------------------------------

def test_upload_mode_still_attaches_the_file_and_parses_results(monkeypatch):
    home_url = "https://tineye.com/"
    states = {
        home_url: {"body": "", "anchors": []},
        RESULTS_URL: {
            "body": "5 results\nmatches",
            "anchors": [{"href": "https://example.com/found", "text": "t", "thumb": "https://example.com/full.jpg"}],
        },
    }
    page = _FakePage(states=states, start_url=home_url, redirect_to=RESULTS_URL)
    monkeypatch.setattr("facechain.search.tineye.attach_file", lambda *a, **kw: True)

    adapter = TinEyeAdapter(_FakeSession(page))
    result = adapter.search("some/local/path.jpg", image_url=None)

    assert result.ok is True
    assert result.query_mode == "upload"
    assert result.candidates and result.candidates[0].url == "https://example.com/found"


def test_upload_mode_reports_a_precise_error_when_the_file_cannot_be_attached(monkeypatch):
    home_url = "https://tineye.com/"
    page = _FakePage(states={home_url: {"body": "", "anchors": []}}, start_url=home_url)
    monkeypatch.setattr("facechain.search.tineye.attach_file", lambda *a, **kw: False)

    adapter = TinEyeAdapter(_FakeSession(page))
    result = adapter.search("some/local/path.jpg", image_url=None)

    assert result.ok is False
    assert result.query_mode == "upload"
    assert "could not attach file" in result.error


# ---- an empty-but-genuine search is NO_RESULTS, not a failure --------------

def test_zero_matches_is_reported_as_no_results_not_a_failure():
    states = {
        START_URL: {"body": "", "anchors": []},
        RESULTS_URL: {"body": "0 results\nTinEye searched 85.6 billion images", "anchors": []},
    }
    page = _FakePage(states=states, start_url=START_URL, redirect_to=RESULTS_URL)
    adapter = TinEyeAdapter(_FakeSession(page))

    result = adapter.search("unused.jpg", image_url=IMAGE_URL)

    assert result.status == ProviderStatus.NO_RESULTS
    assert result.candidates == []
