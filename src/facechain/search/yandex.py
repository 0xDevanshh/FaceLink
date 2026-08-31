"""Yandex Images adapter.

Yandex is historically the strongest engine for faces and for surfacing social
media pages, so it is worth driving carefully: after the reverse search we
explicitly open the "Sites containing information about the image" view.
"""

from __future__ import annotations

import logging
from urllib.parse import quote

from .base import EngineResult, SearchEngineAdapter, build_candidates
from .browser import attach_file, click_text, detect_block, scroll_through, settle

log = logging.getLogger(__name__)

HOME = "https://yandex.com/images/"
BY_URL = "https://yandex.com/images/search?rpt=imageview&url={url}&cbir_page=sites"

FILE_INPUTS = ("input[type=file]", "input.cbir-panel__file-input")
UPLOAD_TRIGGERS = (
    "button[aria-label*='Search by image' i]",
    "div.input__cbir-button",
    "button.input__button_type_camera",
    "[class*='cbir'] button",
)
RESULT_CONTAINERS = (
    "div.CbirSites",
    "div.CbirSites-Items",
    "section[data-state*='sites']",
    "div.main__content",
)


class YandexAdapter(SearchEngineAdapter):
    name = "yandex"
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
                    settle(page, 1200)
                    if not attach_file(page, image_path, FILE_INPUTS, UPLOAD_TRIGGERS):
                        return EngineResult(self.name, ok=False, query_mode=mode,
                                            error="could not attach file to Yandex CBIR input")
                settle(page, 3000)

                blocked = detect_block(page)
                if blocked:
                    return EngineResult(self.name, ok=False, query_mode=mode, error=blocked)

                # The pages-list view is where real URLs live.
                for label in (r"sites containing", r"similar images.*sites", r"image sizes"):
                    if click_text(page, label, timeout_ms=3000):
                        settle(page, 2000)
                        break

                scroll_through(page, rounds=4)
                rows = self.harvest_anchors(page, RESULT_CONTAINERS)
                cands = build_candidates(self.name, rows)
                if not cands:
                    return EngineResult(self.name, ok=False, query_mode=mode,
                                        error="Yandex returned no outbound result links")
                return EngineResult(self.name, candidates=cands, query_mode=mode)

            except Exception as exc:  # noqa: BLE001
                return EngineResult(self.name, ok=False, query_mode=mode,
                                    error=f"{type(exc).__name__}: {str(exc)[:200]}")
