"""SerpAPI adapter — a genuine reverse-image-search API, not a mock.

This is an optional reliability net for the demo: browser automation against
Google/Yandex can hit a CAPTCHA at any moment, and SerpAPI performs the same
real Google Lens / Yandex Images reverse search server-side. It is only enabled
when `SERPAPI_KEY` is set, and it needs a publicly reachable image URL (the API
fetches the image itself), so it pairs with `--image-url` / `--allow-upload-host`.

Free tier: 100 searches/month. Everything here is still a real external
reverse-image lookup — no hardcoded results.
"""

from __future__ import annotations

import json
import logging
import urllib.parse
import urllib.request

from ..config import settings
from ..models import ProviderStatus
from .base import EngineResult, SearchEngineAdapter, build_candidates

log = logging.getLogger(__name__)

ENDPOINT = "https://serpapi.com/search.json"

# SerpAPI result sections that contain page URLs, in order of usefulness.
# Listed generously on purpose: SerpAPI renames and reshuffles these as the
# upstream engines change, and a section we do not read is a real result
# silently discarded.
SECTIONS = (
    "image_results",
    "visual_matches",
    "exact_matches",
    "pages_with_matching_images",
    "related_content",
    "knowledge_graph",
    "inline_images",
    "image_sources",
    "organic_results",
)


class SerpApiAdapter(SearchEngineAdapter):
    """`engine` is the SerpAPI engine id, e.g. google_lens / yandex_images."""

    supports_upload = False
    supports_by_url = True

    def __init__(self, serp_engine: str = "google_lens") -> None:
        self.serp_engine = serp_engine
        self.name = f"serpapi_{serp_engine}"

    def search(self, image_path: str, image_url: str | None = None) -> EngineResult:
        if not settings.serpapi_key:
            return EngineResult(self.name, ok=False, query_mode="api",
                                status=ProviderStatus.NOT_CONFIGURED,
                                error="SERPAPI_KEY not set")
        if not image_url:
            return EngineResult(
                self.name, ok=False, query_mode="api",
                status=ProviderStatus.NOT_CONFIGURED,
                error="SerpAPI needs a public image URL (use --image-url or --allow-upload-host)",
            )

        # Deliberately unrestricted. Pinning `type=exact_matches` made Lens
        # return "hasn't returned any results" for images that do have visual
        # matches, so the narrower query was costing real candidates. Exact
        # versus same-face is decided here anyway — by measuring the image —
        # not by asking the provider to pre-filter.
        params = {"engine": self.serp_engine, "api_key": settings.serpapi_key, "url": image_url}
        query = f"{ENDPOINT}?{urllib.parse.urlencode(params)}"

        try:
            req = urllib.request.Request(query, headers={"User-Agent": settings.user_agent})
            with urllib.request.urlopen(req, timeout=settings.search_timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8", "replace"))
        except Exception as exc:  # noqa: BLE001
            return EngineResult(self.name, ok=False, query_mode="api",
                                error=f"{type(exc).__name__}: {str(exc)[:200]}")

        if payload.get("error"):
            return EngineResult(self.name, ok=False, query_mode="api", error=str(payload["error"])[:200])

        rows: list[dict] = []
        for section in SECTIONS:
            for item in payload.get(section) or []:
                if not isinstance(item, dict):
                    continue
                link = item.get("link") or item.get("source_url") or item.get("url")
                if not link:
                    continue
                rows.append({
                    "href": link,
                    "text": item.get("title") or item.get("source") or "",
                    "thumb": item.get("thumbnail") or item.get("original") or "",
                })

        cands = build_candidates(self.name, rows)
        if not cands:
            return EngineResult(self.name, ok=False, query_mode="api",
                                status=ProviderStatus.NO_RESULTS,
                                error="API returned no usable page links")
        return EngineResult(self.name, candidates=cands, query_mode="api",
                            status=ProviderStatus.COMPLETED)
