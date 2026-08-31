"""Yandex Images adapter.

Yandex is historically the strongest engine for faces and for surfacing social
media pages, so it gets the most careful handling: `cbir_page=sites` opens the
"Sites containing information about the image" view directly, and the results
list (`.CbirSites-Items`) is scraped in preference to the whole page.
"""

from __future__ import annotations

import logging
from urllib.parse import quote

from .base import EngineResult, SearchEngineAdapter, collect_results
from .browser import attach_file, settle

log = logging.getLogger(__name__)

HOME = "https://yandex.com/images/"
BY_URL = "https://yandex.com/images/search?rpt=imageview&url={url}&cbir_page=sites"

FILE_INPUTS = ("input.CbirCore-FileInput", "input[type=file]")
UPLOAD_TRIGGERS = (
    "button[aria-label*='Search by image' i]",
    "[class*='CbirCore'] button",
    "div.input__cbir-button",
    "button.input__button_type_camera",
)
# Class names carry no tag prefix on purpose: these nodes are not all <div>s,
# and `div.CbirSites-Items` silently matches nothing.
RESULT_CONTAINERS = (".CbirSites-Items", ".CbirSitesPage", ".CbirSites")
MARKERS = ("sites containing", "sites with this image", "cbirsites")
VIEW_LABELS = (r"sites containing", r"^sites$")


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
                    page.goto(BY_URL.format(url=quote(image_url, safe="")),
                              wait_until="domcontentloaded")
                else:
                    page.goto(HOME, wait_until="domcontentloaded")
                    settle(page, 1200)
                    if not attach_file(page, image_path, FILE_INPUTS, UPLOAD_TRIGGERS):
                        return EngineResult(self.name, ok=False, query_mode=mode,
                                            error="could not attach file to Yandex CBIR input")
                settle(page, 3000)

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
