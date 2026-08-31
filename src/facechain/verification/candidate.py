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
from .image_similarity import compare, perceptual_hashes
from .social import metadata_consistency

log = logging.getLogger(__name__)

MIN_IMAGE_BYTES = 3000  # skip tracking pixels, spacers, icons
MAX_IMAGE_BYTES = 25 * 1024 * 1024
SKIP_IMAGE_HINTS = ("sprite", "logo", "icon", "avatar_default", "favicon",
                    "placeholder", "blank", "1x1", "spacer")


def _client() -> httpx.Client:
    return httpx.Client(
        follow_redirects=True,
        timeout=settings.http_timeout_s,
        headers={
            "User-Agent": settings.user_agent,
            "Accept": "text/html,application/xhtml+xml,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )


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
    try:
        headers = {"Referer": referer} if referer else {}
        resp = client.get(url, headers=headers)
        if resp.status_code != 200:
            return None
        ctype = resp.headers.get("content-type", "")
        data = resp.content
        if "image" not in ctype and not re.search(r"\.(jpe?g|png|webp|avif)", url, re.I):
            return None
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

    image_targets: list[tuple[str, str]] = []
    with _client() as client:
        # 1. the page itself
        try:
            resp = client.get(candidate.url)
            if resp.status_code == 200 and "html" in resp.headers.get("content-type", ""):
                vc.fetched = True
                image_targets = extract_image_urls(resp.text, str(resp.url))
                vc.fetch_note = f"HTTP 200, {len(image_targets)} image refs"
            else:
                vc.fetch_note = f"HTTP {resp.status_code} (login wall or bot block)"
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
