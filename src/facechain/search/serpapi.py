"""SerpAPI adapter — a genuine reverse-image-search API, not a mock.

This is an optional reliability net for the demo: browser automation against
Google/Yandex can hit a CAPTCHA at any moment, and SerpAPI performs the same
real Google Lens / Yandex Images reverse search server-side. It is only enabled
when `SERPAPI_KEY` is set.

Whether it needs a publicly reachable image URL depends on which upstream
engine is being queried — verified against SerpAPI's own documentation, not
assumed:

  * Google Lens: SerpAPI offers its own direct-upload endpoint
    (`POST https://serpapi.com/image`), which returns a short-lived
    `image_id` that substitutes for the `url` parameter. No public hosting
    needed at all for this one.
  * Yandex Images / Bing reverse-image: `url` is a required parameter with no
    upload alternative (file upload is an open, unimplemented feature request
    on SerpAPI's own roadmap as of this writing) — these genuinely require a
    public URL, which is what `requires_public_url` communicates to the
    orchestrator.

Free tier: 100 searches/month. Everything here is still a real external
reverse-image lookup — no hardcoded results.
"""

from __future__ import annotations

import io
import json
import logging
import urllib.parse
import urllib.request

from ..config import settings
from ..models import ProviderStatus
from .base import EngineResult, SearchEngineAdapter, build_candidates

log = logging.getLogger(__name__)

ENDPOINT = "https://serpapi.com/search.json"
IMAGE_UPLOAD_ENDPOINT = "https://serpapi.com/image"
# SerpAPI's own direct-upload limit for Google Lens.
MAX_UPLOAD_BYTES = 500 * 1024
# SerpAPI engine ids that have no upload alternative — `url` is mandatory.
URL_ONLY_ENGINES = frozenset({"yandex_images", "bing_reverse_image"})

# SerpAPI result sections that contain page URLs, in order of usefulness.
# Listed generously on purpose: SerpAPI renames and reshuffles these as the
# upstream engines change, and a section we do not read is a real result
# silently discarded.
SECTIONS = (
    "image_results",
    "images_results",
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

    supports_by_url = True

    def __init__(self, serp_engine: str = "google_lens") -> None:
        self.serp_engine = serp_engine
        self.name = f"serpapi_{serp_engine}"
        # Only Google Lens has a direct-upload alternative on SerpAPI's side;
        # Yandex/Bing reverse-image are url-only (see module docstring).
        self.supports_upload = serp_engine not in URL_ONLY_ENGINES
        self.requires_public_url = serp_engine in URL_ONLY_ENGINES
        # SerpAPI's own /image upload endpoint is first-party and as
        # reliable as a public URL — no central hosting needed to benefit.
        self.has_reliable_upload_alternative = self.supports_upload

    def _upload_for_image_id(self, image_path: str) -> str:
        """POST to SerpAPI's own upload endpoint, return the `image_id`.

        This is SerpAPI's first-party upload target, not a third-party
        anonymous host — using it needs no separate opt-in beyond the
        `SERPAPI_KEY` the caller already configured.
        """
        with open(image_path, "rb") as fh:
            data = fh.read()
        if len(data) > MAX_UPLOAD_BYTES:
            from PIL import Image
            img = Image.open(io.BytesIO(data)).convert("RGB")
            quality = 85
            while True:
                buf = io.BytesIO()
                img.save(buf, "JPEG", quality=quality)
                data = buf.getvalue()
                if len(data) <= MAX_UPLOAD_BYTES or quality <= 30:
                    break
                quality -= 15
            if len(data) > MAX_UPLOAD_BYTES:
                raise ValueError(
                    f"image could not be reduced under SerpAPI's {MAX_UPLOAD_BYTES}-byte "
                    "upload limit"
                )

        boundary = "----facechainSerpApiUpload"
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="api_key"\r\n\r\n{settings.serpapi_key}\r\n'
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="image.jpg"\r\n'
            f"Content-Type: image/jpeg\r\n\r\n"
        ).encode("utf-8") + data + f"\r\n--{boundary}--\r\n".encode("utf-8")

        req = urllib.request.Request(
            IMAGE_UPLOAD_ENDPOINT, data=body, method="POST",
            headers={
                "User-Agent": settings.user_agent,
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
        )
        with urllib.request.urlopen(req, timeout=settings.search_timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8", "replace"))
        image_id = payload.get("image_id")
        if not image_id:
            raise ValueError(f"upload response had no image_id: {str(payload)[:200]}")
        return image_id

    def search(self, image_path: str, image_url: str | None = None) -> EngineResult:
        if not settings.serpapi_key:
            return EngineResult(self.name, ok=False, query_mode="api",
                                status=ProviderStatus.NOT_CONFIGURED,
                                error="SERPAPI_KEY not set")

        params = {"engine": self.serp_engine, "api_key": settings.serpapi_key}
        if image_url:
            params["url"] = image_url
        elif self.supports_upload:
            try:
                params["image_id"] = self._upload_for_image_id(image_path)
            except Exception as exc:  # noqa: BLE001
                return EngineResult(
                    self.name, ok=False, query_mode="api",
                    error=f"direct image upload to SerpAPI failed: {type(exc).__name__}: {exc}",
                )
        else:
            return EngineResult(
                self.name, ok=False, query_mode="api",
                status=ProviderStatus.NOT_CONFIGURED,
                error=(
                    f"SerpAPI's {self.serp_engine} requires a public image URL "
                    "(no upload alternative) — use --image-url or --allow-upload-host"
                ),
            )

        # Deliberately unrestricted. Pinning `type=exact_matches` made Lens
        # return "hasn't returned any results" for images that do have visual
        # matches, so the narrower query was costing real candidates. Exact
        # versus same-face is decided here anyway — by measuring the image —
        # not by asking the provider to pre-filter.
        query = f"{ENDPOINT}?{urllib.parse.urlencode(params)}"

        try:
            req = urllib.request.Request(query, headers={"User-Agent": settings.user_agent})
            with urllib.request.urlopen(req, timeout=settings.search_timeout_s) as resp:
                payload = json.loads(resp.read().decode("utf-8", "replace"))
        except Exception as exc:  # noqa: BLE001
            return EngineResult(self.name, ok=False, query_mode="api",
                                error=f"{type(exc).__name__}: {str(exc)[:200]}")

        if not isinstance(payload, dict):
            return EngineResult(
                self.name, ok=False, query_mode="api",
                error=f"malformed API response: expected object, got {type(payload).__name__}",
            )
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
                    # Prefer `original` (full-resolution source image, e.g.
                    # media.licdn.com/dms/image/...) over `thumbnail` (Google's
                    # compressed ~50 px cache copy, encrypted-tbn*.gstatic.com).
                    # The old order (thumbnail or original) silently discarded the
                    # full-res URL whenever a thumbnail was present — which is
                    # always — causing ArcFace to compare against a 50 px cache
                    # copy instead of the actual profile photo, producing
                    # face_similarity ≈ 0.15–0.19 even for the correct person.
                    "thumb": item.get("original") or item.get("thumbnail") or "",
                })

        cands = build_candidates(self.name, rows)
        if not cands:
            return EngineResult(self.name, ok=False, query_mode="api",
                                status=ProviderStatus.NO_RESULTS,
                                error="API returned no usable page links")
        return EngineResult(self.name, candidates=cands, query_mode="api",
                            status=ProviderStatus.COMPLETED)
