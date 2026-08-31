"""TinEye adapter.

TinEye rarely indexes social platforms, so it is not enabled by default. It is
still valuable as *corroborating* provenance evidence (earliest known
appearance of the exact image), which is why the adapter exists.
"""

from __future__ import annotations

from urllib.parse import quote

from .base import EngineResult, SearchEngineAdapter, build_candidates
from .browser import attach_file, detect_block, scroll_through, settle

HOME = "https://tineye.com/"
BY_URL = "https://tineye.com/search?url={url}"
FILE_INPUTS = ("input[type=file]", "input#upload_box")
UPLOAD_TRIGGERS = ("button:has-text('Upload')", "label[for='upload_box']")
RESULT_CONTAINERS = ("div.matches", "div#matches", "main")


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
                settle(page, 3000)

                blocked = detect_block(page)
                if blocked:
                    return EngineResult(self.name, ok=False, query_mode=mode, error=blocked)

                scroll_through(page, rounds=2)
                cands = build_candidates(self.name, self.harvest_anchors(page, RESULT_CONTAINERS))
                if not cands:
                    return EngineResult(self.name, ok=False, query_mode=mode,
                                        error="TinEye returned no matches")
                return EngineResult(self.name, candidates=cands, query_mode=mode)
            except Exception as exc:  # noqa: BLE001
                return EngineResult(self.name, ok=False, query_mode=mode,
                                    error=f"{type(exc).__name__}: {str(exc)[:200]}")
