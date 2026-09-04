"""Independently verify a search hit instead of trusting the engine.

For each candidate URL we fetch the page ourselves, pull out the image the post
actually displays, download it, and re-run both similarity tests locally. A
search engine saying "this page matches" is a lead, not evidence.

Image-source acquisition priority (highest trust first):
  1. Direct image URL  — candidate URL itself is an image (GitHub avatar, CDN asset)
  2. github:avatar     — real avatar from avatars.githubusercontent.com markup
  3. og:image          — Open Graph meta tag  (most social platforms set this)
  4. twitter:image     — Twitter card meta tag
  5. link:image_src    — <link rel="image_src">
  6. json-ld           — JSON-LD image / thumbnailUrl
  7. img               — largest <img> tag by declared dimensions
  8. engine-thumbnail  — LAST RESORT only; a compressed, typically ~50 px
                         Google/Yandex cache copy.  Never used when a better
                         source was already found.

Why thumbnails are last: every engine-thumbnail is a re-encoded, heavily
down-sampled version of the original image.  ArcFace cosine similarity between
a 512-D embedding from the original photo and one from a 50 px Google thumbnail
reliably scores below the 0.38 verification threshold even for the identical
person — which was the root cause of false-rejection for all LinkedIn candidates
(HTTP 999 → page not fetched → thumbnail used → face_similarity ≈ 0.19).

The thumbnail remains in the fallback chain so a discovered URL is never
silently abandoned, but its retrieval is logged distinctly and the evidence
record shows ``candidate_image_source == "engine-thumbnail"`` so a reader can
see exactly which source was used.
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
from ..face.quality import score_face_quality
from ..face.similarity import best_match_index
from ..models import SearchCandidate, VerifiedCandidate
from ..search.base import canonicalise_url, classify_platform
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

# Source labels that come from the page itself (not an engine cache).
# Used to decide whether an engine thumbnail fallback is warranted.
_TRUSTED_SOURCES = frozenset({
    "direct-image", "github:avatar",
    "og:image", "twitter:image", "link:image_src", "json-ld", "img",
})


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


def _github_images(soup: BeautifulSoup) -> list[str]:
    """Public avatar/profile images on a GitHub page.

    Worth handling specially: GitHub's `og:image` for a profile is a *generated
    summary card* — the avatar shrunk into a banner with text around it — which
    is a poor input for face comparison. The real avatar is served full-size
    from `avatars.githubusercontent.com` and referenced directly in the markup,
    so preferring it measures the actual photograph rather than a thumbnail
    composited into a card.
    """
    out: list[str] = []
    selectors = (
        "img.avatar-user",
        "a[itemprop='image'] img",
        "img.avatar",
        "img[data-testid='github-avatar']",
    )
    for sel in selectors:
        for tag in soup.select(sel):
            src = tag.get("src") or tag.get("data-src")
            if src:
                out.append(src)
    return out


def extract_image_urls(html: str, page_url: str) -> list[tuple[str, str]]:
    """Return `(image_url, source_label)` in descending order of trust."""
    soup = BeautifulSoup(html, "lxml")
    found: list[tuple[str, str]] = []

    def add(url: str | None, label: str) -> None:
        if url and url.strip():
            found.append((urljoin(page_url, url.strip()), label))

    # Platform-specific, highest-trust sources first.
    _, platform, _ = classify_platform(page_url)
    if platform == "GitHub":
        for src in _github_images(soup):
            add(src, "github:avatar")

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


class MediaCache:
    """Per-run cache of downloaded candidate images, keyed by URL.

    Engines routinely return several pages that all display the same CDN image,
    and the same avatar is referenced from a profile, an org page and a
    contributor list. Downloading those bytes once per run is both faster and
    politer to the hosts involved. Bounded so a large scan cannot grow it
    without limit.
    """

    def __init__(self, max_entries: int = 64, max_bytes: int = 64 * 1024 * 1024) -> None:
        self._store: dict[str, bytes | None] = {}
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._bytes = 0  # bytes actually held in the cache store (bounded)
        self.hits = 0
        self.downloads = 0
        # Real cumulative bytes pulled over the wire this run — unlike `_bytes`,
        # this keeps counting once the cache store is full, so a caller
        # enforcing a total download budget (e.g. `runner.MAX_DOWNLOAD_BYTES`)
        # sees what was actually downloaded rather than an estimate that goes
        # stale the moment the bounded cache stops growing.
        self.total_bytes = 0

    def get_or_fetch(self, client: httpx.Client, url: str) -> bytes | None:
        if url in self._store:
            self.hits += 1
            return self._store[url]
        self.downloads += 1
        data = _download_image(client, url)
        self.total_bytes += len(data or b"")
        # Negative results are cached too: a URL that failed SSRF or returned a
        # non-image will fail identically for every candidate that references it.
        if len(self._store) < self._max_entries and self._bytes < self._max_bytes:
            self._store[url] = data
            self._bytes += len(data or b"")
        return data

    def record_page_bytes(self, n: int) -> None:
        """Count bytes fetched outside `get_or_fetch` (candidate page HTML)."""
        self.total_bytes += n


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
    cache: MediaCache | None = None,
) -> VerifiedCandidate:
    """Fetch, download, and locally re-measure one candidate.

    Acquisition priority (see module docstring):
      direct-image > github:avatar > og:image > twitter:image >
      link:image_src > json-ld > img > engine-thumbnail (last resort)

    The engine thumbnail is only used when no trusted source succeeded — i.e.
    the page was unreachable (login wall, bot block, network error) *and* no
    page-extracted image could be downloaded.  When a trusted source is found
    the thumbnail is skipped entirely so a low-quality cache copy never
    displaces a genuine full-resolution image.
    """
    vc = VerifiedCandidate(
        engine=candidate.engine,
        url=candidate.url,
        domain=candidate.domain,
        platform=candidate.platform,
        is_social=candidate.is_social,
        canonical_url=canonicalise_url(candidate.url),
        platform_priority=candidate.platform_priority,
        candidate_type=candidate.candidate_type,
    )
    cache = cache if cache is not None else MediaCache()

    # SSRF check the candidate URL before fetching.
    if safe_url_or_none(candidate.url) is None:
        vc.fetch_note = "SSRF: rejected (private/loopback/invalid address)"
        vc.rejection_reason = "URL failed SSRF safety check"
        return vc

    # trusted_targets: sources from the page itself (priority 1-7)
    # thumbnail_target: engine cache (priority 8, last resort only)
    trusted_targets: list[tuple[str, str]] = []
    thumbnail_target: tuple[str, str] | None = (
        (candidate.thumbnail, "engine-thumbnail") if candidate.thumbnail else None
    )

    with _client() as client:
        # ---- Step 1: fetch the candidate page --------------------------
        try:
            resp, final_url = _safe_get(client, candidate.url)
            if resp is not None and resp.status_code == 200:
                ctype = resp.headers.get("content-type", "")
                cache.record_page_bytes(len(resp.content))
                if "html" in ctype:
                    raw = resp.content[:MAX_PAGE_BYTES].decode("utf-8", "replace")
                    vc.fetched = True
                    trusted_targets = extract_image_urls(raw, final_url or candidate.url)
                    vc.fetch_note = f"HTTP 200, {len(trusted_targets)} image refs"
                elif ctype.split(";")[0].strip().lower() in CONTENT_TYPE_ALLOWLIST:
                    # The candidate URL *is* the image (GitHub avatar, CDN asset).
                    vc.fetched = True
                    trusted_targets = [(final_url or candidate.url, "direct-image")]
                    vc.fetch_note = f"HTTP 200, direct image ({ctype})"
                else:
                    vc.fetch_note = f"HTTP 200 but content-type={ctype!r}"
            elif resp is not None:
                vc.fetch_note = f"HTTP {resp.status_code} (login wall or bot block)"
            else:
                vc.fetch_note = "fetch failed (SSRF block or network error)"
        except Exception as exc:  # noqa: BLE001
            vc.fetch_note = f"fetch failed: {type(exc).__name__}"

        # ---- Step 2: find the best downloadable image ------------------
        # Try trusted sources first (page-extracted), then thumbnail as
        # last resort only if nothing better was found.
        #
        # "best" means highest perceptual similarity to the input — we want
        # to pick the image that is most likely to be the original, which is
        # the one that looks most like what we searched for.  Within trusted
        # sources, the first downloadable one often wins because extraction
        # order is already trust-ranked; the similarity tiebreak handles
        # cases where og:image returns a banner and a lower-ranked img tag
        # is actually the face photo.

        best_trusted: tuple[float, bytes, str, str] | None = None
        for url, label in trusted_targets[:8]:
            data = cache.get_or_fetch(client, url)
            if data is None:
                continue
            try:
                cand_hashes = perceptual_hashes(data)
            except Exception:  # noqa: BLE001
                continue
            sim = compare(input_hashes, cand_hashes)
            if best_trusted is None or sim > best_trusted[0]:
                best_trusted = (sim, data, url, label)
            if sim >= 0.95:  # near-identical — no need to keep looking
                break

        # Use thumbnail only when no trusted source yielded an image.
        best = best_trusted
        if best is None and thumbnail_target is not None:
            t_url, t_label = thumbnail_target
            data = cache.get_or_fetch(client, t_url)
            if data is not None:
                try:
                    cand_hashes = perceptual_hashes(data)
                    sim = compare(input_hashes, cand_hashes)
                    best = (sim, data, t_url, t_label)
                    log.debug(
                        "candidate %s: using engine-thumbnail as last resort "
                        "(page fetch: %s)",
                        candidate.url, vc.fetch_note,
                    )
                except Exception:  # noqa: BLE001
                    pass

    if best is None:
        vc.fetch_note += " | no downloadable image"
        return vc

    sim, data, img_url, label = best
    vc.image_similarity = sim
    vc.candidate_image_url = img_url
    vc.candidate_image_source = label
    vc.candidate_image_sha256 = sha256_bytes(data)
    vc.candidate_image_phash = perceptual_hashes(data)["phash"]

    # ---- Step 3: face comparison on the retrieved image ----------------
    img = decode_image(data)
    if img is not None:
        try:
            faces = load_backend().detect(img)
            vc.candidate_faces_found = len(faces)
            vc.face_detected = bool(faces)
            if faces:
                vc.face_similarity, idx = best_match_index(
                    input_embedding, [f.embedding for f in faces]
                )
                vc.candidate_face_index = idx
                if idx >= 0:
                    bands, overall = score_face_quality(img, faces[idx])
                    vc.candidate_face_quality = overall
                    vc.candidate_face_bands = bands
        except Exception as exc:  # noqa: BLE001
            log.warning("candidate face pass failed for %s: %s", candidate.url, exc)

    vc.metadata_consistency = metadata_consistency(candidate, vc)
    return vc
