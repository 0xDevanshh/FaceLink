"""Bing Visual Search adapter.

Bing's by-URL entry point lands on a visual-search SERP whose
"Pages with this image" cards live in `.b_cit_row .cit_cards`; those cards are
the high-precision source of real page URLs.
"""

from __future__ import annotations

import logging
from urllib.parse import quote

from .base import EngineResult, SearchEngineAdapter, collect_results
from .browser import attach_file, settle

log = logging.getLogger(__name__)

HOME = "https://www.bing.com/images?FORM=Z9LH"
BY_URL = (
    "https://www.bing.com/images/search?view=detailv2&iss=sbi&form=SBIVSP"
    "&q=imgurl:{url}&mediaurl={url}"
)

FILE_INPUTS = ("input#sb_fileinput", "input[type=file].fileinput", "input[type=file]")
# `#sb_sbi` ("Search using an image") opens the visual-search pane. Bing binds
# the change handler for `#sb_fileinput` when that pane opens, so setting the
# file *before* clicking it attaches the image to an input nobody is listening
# to: the page stays on the images homepage and a whole-page link harvest would
# return trending-topic links. Hence PANE_TRIGGERS, applied first, not as a
# fallback.
PANE_TRIGGERS = (
    "#sb_sbi",
    "[aria-label='Search using an image']",
    "#sbi_b",
)
UPLOAD_TRIGGERS = (
    "div#sbiarea_camera",
    "div.camera",
    "a#sbi_b",
    "[aria-label*='Search using an image' i]",
    "[title*='Search using an image' i]",
)
RESULT_CONTAINERS = (".b_cit_row", ".cit_cards", "#insights_results", ".insightsOverlay")
MARKERS = ("pages with this image", "pages including this image", "visual matches",
           "related searches", "image results")
VIEW_LABELS = (r"pages with this image", r"pages including")


def _open_visual_search_pane(page) -> bool:
    """Click Bing's camera affordance so the file input becomes live.

    Best-effort: if none of the triggers are present the adapter still tries the
    plain input, and the marker guard in `collect_results` catches the case where
    that silently fails to submit.
    """
    for sel in PANE_TRIGGERS:
        try:
            el = page.query_selector(sel)
            if el:
                el.click(timeout=3000)
                settle(page, 1200)
                return True
        except Exception as exc:  # noqa: BLE001
            log.debug("bing pane trigger %s failed: %s", sel, exc)
    return False


class BingVisualAdapter(SearchEngineAdapter):
    name = "bing"
    supports_by_url = True

    def __init__(self, session) -> None:
        self._session = session

    def search(self, image_path: str, image_url: str | None = None) -> EngineResult:
        mode = "by-url" if image_url else "upload"
        with self._session.page() as page:
            try:
                if image_url:
                    enc = quote(image_url, safe="")
                    page.goto(BY_URL.format(url=enc), wait_until="domcontentloaded")
                else:
                    page.goto(HOME, wait_until="domcontentloaded")
                    settle(page, 1200)
                    _open_visual_search_pane(page)
                    if not attach_file(page, image_path, FILE_INPUTS, UPLOAD_TRIGGERS):
                        return EngineResult(self.name, ok=False, query_mode=mode,
                                            error="could not attach file to Bing visual search")
                settle(page, 4500)

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
