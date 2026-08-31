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

FILE_INPUTS = ("input#sb_fileinput", "input[type=file]")
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
                    if not attach_file(page, image_path, FILE_INPUTS, UPLOAD_TRIGGERS):
                        return EngineResult(self.name, ok=False, query_mode=mode,
                                            error="could not attach file to Bing visual search")
                settle(page, 3500)

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
