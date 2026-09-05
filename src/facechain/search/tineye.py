"""TinEye adapter.

TinEye rarely indexes social platforms, so it is not enabled by default. It is
still valuable as *corroborating* provenance evidence (earliest known
appearance of the exact image), which is why the adapter exists.

RCA (regression this file guards against): TinEye's search is computed
server-side, and the page then *client-navigates* to a content-addressed
results permalink (``/search/<hash>?sort=...``) rather than rendering results
in place on ``/search?url=...``. A fixed sleep alone races that navigation —
under any extra latency (a slow temporary image host, a busy TinEye backend)
the marker check can run before the redirect ever happens, misreporting a
real-but-slow search as "results view not reached". This mostly went
unnoticed while `image_url` was rarely available (central image hosting used
to fail most of the time), so TinEye almost always ran in upload mode; once
hosting became reliable, by-url mode — and this race — started firing far
more often. `search()` below now waits for that redirect explicitly before
falling back to the same fixed-beat settle as before.
"""

from __future__ import annotations

import contextlib
import re
from urllib.parse import quote

from .base import EngineResult, SearchEngineAdapter, collect_results
from .browser import attach_file, settle

HOME = "https://tineye.com/"
BY_URL = "https://tineye.com/search?url={url}"
FILE_INPUTS = ("input[type=file]", "input#upload_box")
UPLOAD_TRIGGERS = ("button:has-text('Upload')", "label[for='upload_box']")
RESULT_CONTAINERS = (".matches", "#matches", "main")
MARKERS = ("searched over", "results", "matches")
VIEW_LABELS = ()

# A stable ID scheme, not a CSS selector that breaks on the next redesign —
# robust to layout changes the way `base.py`'s outbound-link harvesting is.
RESULTS_URL_PATTERN = re.compile(r"tineye\.com/search/[0-9a-f]+", re.I)
# Generous but bounded: well inside both `search_timeout_s` (per-operation)
# and `engine_timeout_s` (whole-adapter) budgets, so a page that never
# redirects (a genuine block, or a future TinEye that renders in place) is
# never waited on for long — it simply falls through to the checks below.
REDIRECT_WAIT_MS = 15000

# TinEye's own wording when it could not fetch the image at a supplied URL —
# a real, specific failure (the temp-hosted URL expired, was unreachable, or
# wasn't a supported format by the time TinEye's server tried it), distinct
# from a bot challenge or a genuine layout change and worth naming precisely
# rather than folding into the generic "results view not reached".
_URL_FETCH_FAILURE_MARKER = "could not read that image url"


class TinEyeAdapter(SearchEngineAdapter):
    name = "tineye"
    supports_by_url = True

    def __init__(self, session) -> None:
        self._session = session

    def search(self, image_path: str, image_url: str | None = None) -> EngineResult:
        mode = "by-url" if image_url else "upload"
        with self._session.page() as page:
            try:
                if image_url:
                    page.goto(BY_URL.format(url=quote(image_url, safe="")), wait_until="domcontentloaded")
                else:
                    page.goto(HOME, wait_until="domcontentloaded")
                    settle(page, 1000)
                    if not attach_file(page, image_path, FILE_INPUTS, UPLOAD_TRIGGERS):
                        return EngineResult(self.name, ok=False, query_mode=mode,
                                            error="could not attach file to TinEye")

                # Give the server-side search + client redirect a real chance
                # to land before the fixed settle() beat below. Tolerated
                # failing outright: a genuine block/CAPTCHA page (no redirect
                # ever happens) or a results page that renders in place both
                # fall through to the checks that follow, unaffected.
                with contextlib.suppress(Exception):
                    page.wait_for_url(RESULTS_URL_PATTERN, timeout=REDIRECT_WAIT_MS)

                settle(page, 3000)

                try:
                    body = (page.inner_text("body") or "").lower()
                except Exception:  # noqa: BLE001
                    body = ""
                if _URL_FETCH_FAILURE_MARKER in body:
                    return EngineResult(
                        self.name, ok=False, query_mode=mode,
                        error="TinEye could not fetch the supplied image URL "
                              "(expired, unreachable, or not a supported image format)",
                    )

                result = collect_results(
                    page,
                    engine=self.name,
                    containers=RESULT_CONTAINERS,
                    markers=MARKERS,
                    view_labels=VIEW_LABELS,
                )
                result.query_mode = mode
                return result

            except Exception as exc:  # noqa: BLE001
                return EngineResult(self.name, ok=False, query_mode=mode,
                                    error=f"{type(exc).__name__}: {str(exc)[:200]}")
