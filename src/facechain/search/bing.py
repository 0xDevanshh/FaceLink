"""Bing Visual Search adapter (incl. its 'Pages Including This Image' view)."""

from __future__ import annotations

import logging
from urllib.parse import quote

from .base import EngineResult, SearchEngineAdapter, build_candidates
from .browser import attach_file, click_text, detect_block, scroll_through, settle

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
RESULT_CONTAINERS = (
    "div.pageIncludes",
    "div#insights_results",
    "div.insights",
    "main",
)


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
                settle(page, 3000)

                blocked = detect_block(page)
                if blocked:
                    return EngineResult(self.name, ok=False, query_mode=mode, error=blocked)

                for label in (r"pages including", r"pages with this image", r"related searches"):
                    if click_text(page, label, timeout_ms=3000):
                        settle(page, 2000)
                        break

                scroll_through(page, rounds=3)
                rows = self.harvest_anchors(page, RESULT_CONTAINERS)
                cands = build_candidates(self.name, rows)
                if not cands:
                    return EngineResult(self.name, ok=False, query_mode=mode,
                                        error="Bing returned no outbound result links")
                return EngineResult(self.name, candidates=cands, query_mode=mode)

            except Exception as exc:  # noqa: BLE001
                return EngineResult(self.name, ok=False, query_mode=mode,
                                    error=f"{type(exc).__name__}: {str(exc)[:200]}")
