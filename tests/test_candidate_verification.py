"""Candidate verification: what gets fetched, what gets measured, what is refused.

`verify_candidate` is the only place a claim about a candidate becomes a
measurement, so these tests pin the boundaries around it: a URL that fails SSRF
is never fetched, a non-image response is never decoded as one, a corrupt or
oversized body is refused, and the exact-image / same-face distinction reflects
what was actually measured rather than what a URL looked like.

The network is stubbed at the httpx-transport boundary, so the real fetch,
redirect-revalidation and content-type code all still run.
"""

from __future__ import annotations

import io

import httpx
import numpy as np
import pytest
from PIL import Image

from facechain.models import CandidateType, SearchCandidate, Stage, VerifiedCandidate
from facechain.verification import candidate as candmod
from facechain.verification.candidate import (
    MediaCache,
    _github_images,
    extract_image_urls,
    verify_candidate,
)
from facechain.verification.image_similarity import perceptual_hashes
from facechain.verification.scorer import score_candidate


def png_bytes(colour=(120, 90, 60), size=(200, 200)) -> bytes:
    """A textured PNG large enough to clear MIN_IMAGE_BYTES."""
    rng = np.random.default_rng(7)
    arr = np.clip(np.array(colour, dtype=np.int16) + rng.integers(-70, 70, (*size, 3)), 0, 255)
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, "PNG")
    return buf.getvalue()


def embedding(seed: int = 0, dim: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vec = rng.normal(size=dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


def candidate(url: str = "https://example.com/page", **kw) -> SearchCandidate:
    from facechain.search.base import classify_platform, initial_candidate_type

    recognised, platform, priority = classify_platform(url)
    base = dict(
        engine="yandex", url=url,
        domain=url.split("/")[2],
        is_social=recognised, platform=platform, platform_priority=priority,
        candidate_type=initial_candidate_type(url, platform),
    )
    base.update(kw)
    return SearchCandidate(**base)


@pytest.fixture
def stub_http(monkeypatch):
    """Route every httpx request through a routing table we control."""
    routes: dict[str, httpx.Response] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        resp = routes.get(str(request.url))
        if resp is None:
            raise httpx.ConnectError("no route", request=request)
        return httpx.Response(resp.status_code, headers=resp.headers, content=resp.content,
                              request=request)

    real_client = candmod._client

    def patched() -> httpx.Client:
        client = real_client()
        client._transport = httpx.MockTransport(handler)
        return client

    monkeypatch.setattr(candmod, "_client", patched)
    # Every host in these tests resolves fine; SSRF-specific behaviour has its
    # own tests below that do NOT stub this out.
    monkeypatch.setattr(candmod, "safe_url_or_none", lambda u: u)
    return routes


def html_page(body: str) -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "text/html; charset=utf-8"},
                          content=body.encode())


def image_response(data: bytes, ctype: str = "image/png") -> httpx.Response:
    return httpx.Response(200, headers={"content-type": ctype}, content=data)


# ---- SSRF: refused before any connection --------------------------------

@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/admin",
        "http://localhost/admin",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.5/internal",
        "http://192.168.1.1/router",
        "http://[::1]/local",
        "file:///etc/passwd",
        "ftp://example.com/x",
        "gopher://example.com/x",
    ],
)
def test_unsafe_candidate_urls_are_refused_without_fetching(url):
    vc = verify_candidate(candidate(url, domain="unsafe"), {"phash": "0" * 16}, embedding())
    assert not vc.fetched
    assert vc.rejection_reason == "URL failed SSRF safety check"
    assert vc.candidate_image_sha256 is None


# ---- content handling ---------------------------------------------------

def test_a_non_image_response_is_not_treated_as_an_image(stub_http):
    stub_http["https://example.com/page"] = httpx.Response(
        200, headers={"content-type": "application/pdf"}, content=b"%PDF-1.7 not an image")
    vc = verify_candidate(candidate(), {"phash": "0" * 16}, embedding())
    assert vc.candidate_image_sha256 is None
    assert "content-type" in vc.fetch_note


def test_a_login_wall_is_reported_as_such(stub_http):
    stub_http["https://instagram.com/p/ABC/"] = httpx.Response(
        403, headers={"content-type": "text/html"}, content=b"login required")
    vc = verify_candidate(candidate("https://instagram.com/p/ABC/"), {"phash": "0" * 16}, embedding())
    assert "403" in vc.fetch_note
    assert "login wall" in vc.fetch_note


def test_a_corrupt_image_body_does_not_crash_verification(stub_http):
    stub_http["https://example.com/page"] = html_page(
        '<meta property="og:image" content="https://cdn.example.com/broken.png">')
    stub_http["https://cdn.example.com/broken.png"] = image_response(
        b"\x89PNG\r\n\x1a\n" + b"\xff" * 8000)
    vc = verify_candidate(candidate(), {"phash": "0" * 16}, embedding())
    # Refused cleanly rather than raising or claiming a measurement.
    assert not vc.face_detected
    assert vc.face_similarity == 0.0


def test_an_oversized_image_is_refused(stub_http, monkeypatch):
    monkeypatch.setattr(candmod, "MAX_IMAGE_BYTES", 5000)
    stub_http["https://example.com/page"] = html_page(
        '<meta property="og:image" content="https://cdn.example.com/big.png">')
    stub_http["https://cdn.example.com/big.png"] = image_response(png_bytes(size=(400, 400)))
    vc = verify_candidate(candidate(), {"phash": "0" * 16}, embedding())
    assert vc.candidate_image_sha256 is None


def test_a_tracking_pixel_is_refused_as_too_small(stub_http):
    stub_http["https://example.com/page"] = html_page(
        '<meta property="og:image" content="https://cdn.example.com/px.png">')
    stub_http["https://cdn.example.com/px.png"] = image_response(png_bytes(size=(1, 1)))
    vc = verify_candidate(candidate(), {"phash": "0" * 16}, embedding())
    assert vc.candidate_image_sha256 is None


def test_a_direct_image_url_is_measured_rather_than_discarded(stub_http):
    """A candidate that *is* an image (a GitHub avatar, a CDN asset) is a
    perfectly good measurement and used to be thrown away."""
    data = png_bytes()
    stub_http["https://avatars.githubusercontent.com/u/1"] = image_response(data)
    vc = verify_candidate(
        candidate("https://avatars.githubusercontent.com/u/1"),
        perceptual_hashes(data), embedding())
    assert vc.fetched
    assert vc.candidate_image_source == "direct-image"
    assert vc.image_similarity == pytest.approx(1.0)


def test_the_engine_thumbnail_is_a_documented_fallback(stub_http):
    data = png_bytes()
    stub_http["https://instagram.com/p/ABC/"] = httpx.Response(
        403, headers={"content-type": "text/html"}, content=b"nope")
    stub_http["https://thumb.cdn/x.png"] = image_response(data)
    vc = verify_candidate(
        candidate("https://instagram.com/p/ABC/", thumbnail="https://thumb.cdn/x.png"),
        perceptual_hashes(data), embedding())
    # Provenance is recorded honestly rather than being passed off as the page's
    # own image.
    assert vc.candidate_image_source == "engine-thumbnail"
    assert vc.candidate_image_sha256


# ---- image extraction ---------------------------------------------------

def test_metadata_images_outrank_plain_img_tags():
    html = """
      <meta property="og:image" content="https://cdn/og.jpg">
      <meta name="twitter:image" content="https://cdn/tw.jpg">
      <img src="https://cdn/body.jpg" width="800" height="600">
    """
    order = [label for _, label in extract_image_urls(html, "https://example.com/p")]
    assert order[0] == "og:image"
    assert "img" in order
    assert order.index("og:image") < order.index("img")


def test_github_avatar_is_preferred_over_the_generated_og_card():
    """GitHub's og:image for a profile is a banner with the avatar shrunk into
    it — a poor input for face comparison next to the real avatar file."""
    html = """
      <meta property="og:image" content="https://opengraph.githubassets.com/card.png">
      <img class="avatar-user" src="https://avatars.githubusercontent.com/u/1?v=4">
    """
    pairs = extract_image_urls(html, "https://github.com/someone")
    assert pairs[0][1] == "github:avatar"
    assert "avatars.githubusercontent.com" in pairs[0][0]


def test_github_avatar_extraction_is_not_applied_to_other_hosts():
    html = '<img class="avatar-user" src="https://elsewhere.example/a.png">'
    labels = [label for _, label in extract_image_urls(html, "https://example.com/p")]
    assert "github:avatar" not in labels


def test_github_image_selectors_tolerate_a_page_without_them():
    from bs4 import BeautifulSoup
    assert _github_images(BeautifulSoup("<html><body></body></html>", "lxml")) == []


def test_decorative_images_are_skipped():
    html = """
      <img src="https://cdn/sprite.png"><img src="https://cdn/favicon.ico">
      <img src="https://cdn/real-photo.jpg" width="600" height="400">
    """
    urls = [u for u, _ in extract_image_urls(html, "https://example.com/p")]
    assert any("real-photo" in u for u in urls)
    assert not any("sprite" in u or "favicon" in u for u in urls)


# ---- media cache -------------------------------------------------------

def test_the_same_image_url_is_downloaded_once_per_run(stub_http):
    data = png_bytes()
    stub_http["https://a.example/page"] = html_page(
        '<meta property="og:image" content="https://cdn.example.com/shared.png">')
    stub_http["https://b.example/page"] = html_page(
        '<meta property="og:image" content="https://cdn.example.com/shared.png">')
    stub_http["https://cdn.example.com/shared.png"] = image_response(data)

    cache = MediaCache()
    hashes = {}
    for host in ("a", "b"):
        vc = verify_candidate(candidate(f"https://{host}.example/page"),
                              perceptual_hashes(data), embedding(), cache)
        hashes[host] = vc.candidate_image_sha256

    assert hashes["a"] == hashes["b"]
    assert cache.downloads == 1
    assert cache.hits == 1


def test_the_cache_remembers_failures_too(stub_http):
    stub_http["https://a.example/page"] = html_page(
        '<meta property="og:image" content="https://cdn.example.com/gone.png">')
    stub_http["https://cdn.example.com/gone.png"] = httpx.Response(
        404, headers={"content-type": "text/html"}, content=b"nope")
    cache = MediaCache()
    for _ in range(3):
        verify_candidate(candidate("https://a.example/page"), {"phash": "0" * 16},
                         embedding(), cache)
    assert cache.downloads == 1
    assert cache.hits == 2


def test_the_cache_is_bounded():
    cache = MediaCache(max_entries=2)
    assert cache._max_entries == 2


def test_total_bytes_counts_the_page_and_the_image(stub_http):
    """Regression: the download-budget guard in `runner.py` used to estimate
    bytes as `downloads * 500_000` rather than summing what was actually
    fetched. `total_bytes` must reflect real page + image bytes."""
    data = png_bytes()
    page_html = html_page(
        '<meta property="og:image" content="https://cdn.example.com/shared.png">')
    stub_http["https://a.example/page"] = page_html
    stub_http["https://cdn.example.com/shared.png"] = image_response(data)

    cache = MediaCache()
    verify_candidate(candidate("https://a.example/page"), perceptual_hashes(data),
                     embedding(), cache)

    assert cache.total_bytes == len(page_html.content) + len(data)


def test_total_bytes_keeps_counting_once_the_cache_store_is_full(stub_http):
    """`_bytes` (the bounded in-memory store) stops growing once the cache is
    full; `total_bytes` must not silently freeze with it — a scan that keeps
    downloading past the cache's capacity is still spending real bandwidth."""
    cache = MediaCache(max_entries=1, max_bytes=1)
    images = [png_bytes(colour=c) for c in ((10, 10, 10), (120, 120, 120), (230, 230, 230))]
    for i, data in enumerate(images):
        url = f"https://cdn.example.com/img{i}.png"
        stub_http[f"https://site{i}.example/page"] = html_page(
            f'<meta property="og:image" content="{url}">')
        stub_http[url] = image_response(data)

    for i, data in enumerate(images):
        verify_candidate(candidate(f"https://site{i}.example/page"),
                         perceptual_hashes(data), embedding(), cache)

    assert cache.downloads == 3
    # The bounded store can only hold the first entry...
    assert len(cache._store) == 1
    # ...but total_bytes reflects every download, not just what was retained.
    total_image_bytes = sum(len(d) for d in images)
    assert cache.total_bytes >= total_image_bytes


# ---- exact-image vs same-face ------------------------------------------

def _measured(**kw) -> VerifiedCandidate:
    base = dict(
        engine="yandex", url="https://example.com/p", domain="example.com",
        candidate_image_sha256="ab" * 32, candidate_image_source="og:image",
        face_detected=True, metadata_consistency=0.6,
        candidate_type=CandidateType.PUBLIC_WEB_PAGE,
    )
    base.update(kw)
    return VerifiedCandidate(**base)


def test_the_same_picture_is_typed_exact_image():
    vc = score_candidate(_measured(image_similarity=0.97, face_similarity=0.91))
    assert vc.candidate_type == CandidateType.EXACT_IMAGE
    assert vc.match_type == "exact-image"
    assert Stage.IMAGE_MATCH in vc.stages


def test_a_different_picture_of_the_same_face_is_typed_same_face():
    vc = score_candidate(_measured(image_similarity=0.31, face_similarity=0.88))
    assert vc.candidate_type == CandidateType.SAME_FACE
    assert vc.match_type == "face-only"
    assert Stage.IMAGE_MATCH not in vc.stages
    assert Stage.FACE_MATCH in vc.stages


def test_a_different_face_keeps_its_url_derived_type():
    """EXACT_IMAGE and SAME_FACE are claims about measurements, so an unmeasured
    or non-matching candidate must not acquire one."""
    vc = score_candidate(_measured(image_similarity=0.20, face_similarity=0.05,
                                   candidate_type=CandidateType.SOCIAL_PROFILE))
    assert vc.candidate_type == CandidateType.SOCIAL_PROFILE
    assert not vc.verified


def test_an_image_with_no_face_cannot_be_typed_same_face():
    vc = score_candidate(_measured(face_detected=False, face_similarity=0.0,
                                   image_similarity=0.4,
                                   candidate_type=CandidateType.PUBLIC_ARTICLE))
    assert vc.candidate_type == CandidateType.PUBLIC_ARTICLE
    assert not vc.verified
    assert "no face detectable" in vc.rejection_reason


def test_multiple_candidate_faces_use_the_best_valid_comparison():
    """A group photo's subject need not be its largest face."""
    from facechain.face.similarity import best_cosine

    reference = embedding(1)
    others = [embedding(2), reference.copy(), embedding(3)]
    assert best_cosine(reference, others) == pytest.approx(1.0, abs=1e-5)
    # And a set with nobody matching stays low.
    assert best_cosine(reference, [embedding(4), embedding(5)]) < 0.5
