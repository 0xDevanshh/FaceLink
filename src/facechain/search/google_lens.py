"""Google Lens adapter — drives the real lens.google.com UI."""

from __future__ import annotations

import logging
from urllib.parse import quote

from ..config import settings
from .base import EngineResult, SearchEngineAdapter, build_candidates
from .browser import attach_file, click_text, detect_block, scroll_through, settle

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

                blocked = detect_block(page)
                if blocked:
                    return EngineResult(self.name, ok=False, query_mode=mode, error=blocked)

                # Lens defaults to "visual matches"; the source-pages view is the
                # one that yields real page URLs (incl. social posts).
                for label in (r"find image source", r"exact matches", r"pages with matching images"):
                    if click_text(page, label, timeout_ms=3500):
                        log.debug("clicked Lens view: %s", label)
                        settle(page, 2500)
                        break

                scroll_through(page, rounds=4)
                rows = self.harvest_anchors(page, RESULT_CONTAINERS)
                cands = build_candidates(self.name, rows)
                if not cands:
                    return EngineResult(self.name, ok=False, query_mode=mode,
                                        error="Lens returned no outbound result links")
                return EngineResult(self.name, candidates=cands, query_mode=mode)

            except Exception as exc:  # noqa: BLE001
                return EngineResult(self.name, ok=False, query_mode=mode,
                                    error=f"{type(exc).__name__}: {str(exc)[:200]}")
