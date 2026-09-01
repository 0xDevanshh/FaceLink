"""Independently verify a search hit instead of trusting the engine.

For each candidate URL we fetch the page ourselves, pull out the image the post
actually displays, download it, and re-run both similarity tests locally. A
search engine saying "this page matches" is a lead, not evidence.

Reality check that shaped this module: social platforms often refuse anonymous
page fetches (login walls, 403s). When that happens we fall back to the
thumbnail the engine itself stored for that result — still a real image tied to
that result, never a fabricated one — and we record which source was used in
`candidate_image_source` so the evidence stays honest about its provenance.
"""

from __future__ import annotations

import json
import logging
import re
from urllib.parse import urljoin

import httpx
import numpy as np
from bs4 import BeautifulSoup

from ..config import settings
from ..evidence.hashing import sha256_bytes
from ..face.encoder import decode_image
from ..face.detector import load_backend
from ..face.similarity import best_cosine
from ..models import SearchCandidate, VerifiedCandidate
from ..security.ssrf import SSRFViolation, safe_url_or_none
from .image_similarity import compare, perceptual_hashes
from .social import metadata_consistency

log = logging.getLogger(__name__)

MIN_IMAGE_BYTES = 3000  # skip tracking pixels, spacers, icons
MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_PAGE_BYTES = 10 * 1024 * 1024
SKIP_IMAGE_HINTS = ("sprite", "logo", "icon", "avatar_default", "favicon",
                    "placeholder", "blank", "1x1", "spacer")
CONTENT_TYPE_ALLOWLIST = ("image/jpeg", "image/png", "image/webp", "image/avif",
                          "image/gif", "image/bmp")


def _client() -> httpx.Client:
    return httpx.Client(
        follow_redirects=False,  # we handle redirects manually for SSRF re-validation
        timeout=httpx.Timeout(connect=5.0, read=15.0, write=5.0, pool=5.0),
        headers={
            "User-Agent": settings.user_agent,
            "Accept": "text/html,application/xhtml+xml,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
        max_redirects=0,
    )


def _safe_get(client: httpx.Client, url: str, max_redirects: int = 3) -> tuple[httpx.Response | None, str]:
    """SSRF-safe GET with per-hop IP re-validation.

    Returns (response, final_url). If any hop fails SSRF checks or exhausts
    redirects, returns (None, url) with the reason logged.
    """
    current = url
    for hop in range(max_redirects + 1):
        safe = safe_url_or_none(current)
        if safe is None:
            return None, current
        try:
            resp = client.get(safe, follow_redirects=False)
        except Exception as exc:
            log.debug("fetch error %s: %s", current, exc)
            return None, current
        if resp.is_redirect:
            location = resp.headers.get("location", "")
            if not location:
                return None, current
            if not location.startswith(("http://", "https://")):
                location = urljoin(current, location)
            current = location
            continue
        return resp, current
    log.debug("too many redirects for %s", url)
    return None, current


def _jsonld_images(soup: BeautifulSoup) -> list[str]:
    out: list[str] = []
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads(tag.string or "{}")
        except (json.JSONDecodeError, TypeError):
            continue
        stack = [data]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                img = node.get("image") or node.get("thumbnailUrl")
                if isinstance(img, str):
                    out.append(img)
                elif isinstance(img, list):
                    out.extend(i for i in img if isinstance(i, str))
                elif isinstance(img, dict) and isinstance(img.get("url"), str):
                    out.append(img["url"])
                stack.extend(v for v in node.values() if isinstance(v, (dict, list)))
            elif isinstance(node, list):
                stack.extend(node)
    return out


def extract_image_urls(html: str, page_url: str) -> list[tuple[str, str]]:
    """Return `(image_url, source_label)` in descending order of trust."""
    soup = BeautifulSoup(html, "lxml")
    found: list[tuple[str, str]] = []

    def add(url: str | None, label: str) -> None:
        if url and url.strip():
            found.append((urljoin(page_url, url.strip()), label))

    for prop, label in (
        ("og:image:secure_url", "og:image"),
        ("og:image", "og:image"),
        ("twitter:image", "twitter:image"),
        ("twitter:image:src", "twitter:image"),
    ):
        for tag in soup.find_all("meta", attrs={"property": prop}) + soup.find_all(
            "meta", attrs={"name": prop}
        ):
            add(tag.get("content"), label)

    for tag in soup.find_all("link", attrs={"rel": "image_src"}):
        add(tag.get("href"), "link:image_src")

    for url in _jsonld_images(soup):
        add(url, "json-ld")

    # Plain <img> as a last resort, biggest-looking first.
    imgs: list[tuple[int, str]] = []
    for tag in soup.find_all("img"):
        src = tag.get("src") or tag.get("data-src") or tag.get("data-original")
        if not src or any(h in src.lower() for h in SKIP_IMAGE_HINTS):
            continue
        try:
            area = int(tag.get("width") or 0) * int(tag.get("height") or 0)
        except (TypeError, ValueError):
            area = 0
        imgs.append((area, src))
    for _, src in sorted(imgs, key=lambda t: -t[0])[:6]:
        add(src, "img")

    # De-duplicate, keep first (highest-trust) label per URL.
    seen: set[str] = set()
    ordered: list[tuple[str, str]] = []
    for url, label in found:
        if url not in seen and url.startswith(("http://", "https://")):
            seen.add(url)
            ordered.append((url, label))
    return ordered


def _download_image(client: httpx.Client, url: str, referer: str | None = None) -> bytes | None:
    """SSRF-safe image download with content-type and size validation."""
    safe = safe_url_or_none(url)
    if safe is None:
        return None
    try:
        resp, _ = _safe_get(client, safe)
        if resp is None or resp.status_code != 200:
            return None
        ctype = resp.headers.get("content-type", "").split(";")[0].strip().lower()
        # Accept if content-type is image OR URL looks like image file.
        if ctype not in CONTENT_TYPE_ALLOWLIST and not re.search(
            r"\.(jpe?g|png|webp|avif|gif|bmp)(\?|$)", url, re.I
        ):
            return None
        data = resp.content
        if not (MIN_IMAGE_BYTES <= len(data) <= MAX_IMAGE_BYTES):
            return None
        return data
    except Exception as exc:  # noqa: BLE001
        log.debug("image download failed %s: %s", url, exc)
        return None


def verify_candidate(
    candidate: SearchCandidate,
    input_hashes: dict[str, str],
    input_embedding: np.ndarray,
) -> VerifiedCandidate:
    """Fetch, download, and locally re-measure one candidate."""
    vc = VerifiedCandidate(
        engine=candidate.engine,
        url=candidate.url,
        domain=candidate.domain,
        platform=candidate.platform,
        is_social=candidate.is_social,
    )

    # SSRF check the candidate URL before fetching.
    if safe_url_or_none(candidate.url) is None:
        vc.fetch_note = "SSRF: rejected (private/loopback/invalid address)"
        vc.rejection_reason = "URL failed SSRF safety check"
        return vc

    image_targets: list[tuple[str, str]] = []
    with _client() as client:
        # 1. the page itself — SSRF-safe with per-hop re-validation
        try:
            resp, final_url = _safe_get(client, candidate.url)
            if resp is not None and resp.status_code == 200:
                ctype = resp.headers.get("content-type", "")
                if "html" in ctype:
                    # Size cap on page HTML.
                    raw = resp.content[:MAX_PAGE_BYTES].decode("utf-8", "replace")
                    vc.fetched = True
                    image_targets = extract_image_urls(raw, final_url or candidate.url)
                    vc.fetch_note = f"HTTP 200, {len(image_targets)} image refs"
                else:
                    vc.fetch_note = f"HTTP 200 but content-type={ctype!r}"
            elif resp is not None:
                vc.fetch_note = f"HTTP {resp.status_code} (login wall or bot block)"
            else:
                vc.fetch_note = "fetch failed (SSRF block or network error)"
        except Exception as exc:  # noqa: BLE001
            vc.fetch_note = f"fetch failed: {type(exc).__name__}"

        # 2. engine thumbnail as a documented fallback
        if candidate.thumbnail:
            image_targets.append((candidate.thumbnail, "engine-thumbnail"))

        if not image_targets:
            vc.fetch_note += " | no comparable image found"
            return vc

        best: tuple[float, bytes, str, str] | None = None
        for url, label in image_targets[:8]:
            data = _download_image(client, url, referer=candidate.url)
            if data is None:
                continue
            try:
                cand_hashes = perceptual_hashes(data)
            except Exception:  # noqa: BLE001
                continue
            sim = compare(input_hashes, cand_hashes)
            if best is None or sim > best[0]:
                best = (sim, data, url, label)
            if sim >= 0.95:  # near-identical; no need to keep looking
                break

    if best is None:
        vc.fetch_note += " | no downloadable image"
        return vc

    sim, data, img_url, label = best
    vc.image_similarity = sim
    vc.candidate_image_url = img_url
    vc.candidate_image_source = label
    vc.candidate_image_sha256 = sha256_bytes(data)
    vc.candidate_image_phash = perceptual_hashes(data)["phash"]

    # ---- face comparison on the retrieved image --------------------------
    img = decode_image(data)
    if img is not None:
        try:
            faces = load_backend().detect(img)
            vc.candidate_faces_found = len(faces)
            vc.face_detected = bool(faces)
            if faces:
                vc.face_similarity = best_cosine(input_embedding, [f.embedding for f in faces])
        except Exception as exc:  # noqa: BLE001
            log.warning("candidate face pass failed for %s: %s", candidate.url, exc)

    vc.metadata_consistency = metadata_consistency(candidate, vc)
    return vc
