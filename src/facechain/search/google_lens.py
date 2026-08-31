"""Google Lens adapter — drives the real lens.google.com UI."""

from __future__ import annotations

import logging
from urllib.parse import quote

from ..config import settings
from .base import EngineResult, SearchEngineAdapter, collect_results
from .browser import attach_file, settle

log = logging.getLogger(__name__)

UPLOAD_URL = "https://lens.google.com/upload?ep=ccm&s=&st=1"
BY_URL = "https://lens.google.com/uploadbyurl?url={url}"

FILE_INPUTS = (
    "input[type=file][name=encoded_image]",
    "input[type=file]",
)
UPLOAD_TRIGGERS = (
    "div[aria-label*='Search by image' i]",
    "div[jsname='ZtOxCb']",
    "span:has-text('upload a file')",
)
RESULT_CONTAINERS = (
    "div[data-async-context] div[role='list']",
    "div[jsname='Cpkphb']",
    "#search",
    "div[role='main']",
)
MARKERS = ("exact matches", "visual matches", "pages with matching images",
           "find image source", "about this image", "results for")
VIEW_LABELS = (r"find image source", r"exact matches", r"pages with matching images")


class GoogleLensAdapter(SearchEngineAdapter):
    name = "google_lens"
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
                    page.goto(UPLOAD_URL, wait_until="domcontentloaded")
                    settle(page, 1200)
                    if not attach_file(page, image_path, FILE_INPUTS, UPLOAD_TRIGGERS):
                        return EngineResult(self.name, ok=False, query_mode=mode,
                                            error="could not attach file to Lens upload input")
                settle(page, 2500)

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
