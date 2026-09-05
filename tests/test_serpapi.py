"""SerpAPI adapter: capability flags, and the Google Lens direct-upload path.

Verified against SerpAPI's own documentation (not assumed): Google Lens has a
first-party upload endpoint that returns an `image_id` substituting for `url`;
Yandex Images and Bing reverse-image have no upload alternative at all — `url`
is mandatory. These tests pin that distinction so the wrong one can never
silently regress back to depending on external temp-hosting for Lens, or
silently claim an upload path exists for Yandex/Bing that SerpAPI does not
actually offer.

All offline — `urllib.request.urlopen` is monkeypatched.
"""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

from facechain.config import settings
from facechain.models import ProviderStatus
from facechain.search.serpapi import SerpApiAdapter


# ---- capability flags -----------------------------------------------------

def test_google_lens_has_no_public_url_requirement_and_a_reliable_alternative():
    a = SerpApiAdapter("google_lens")
    assert a.supports_upload is True
    assert a.requires_public_url is False
    assert a.has_reliable_upload_alternative is True


@pytest.mark.parametrize("serp_engine", ["yandex_images", "bing_reverse_image"])
def test_yandex_and_bing_require_a_public_url_with_no_alternative(serp_engine):
    a = SerpApiAdapter(serp_engine)
    assert a.supports_upload is False
    assert a.requires_public_url is True
    assert a.has_reliable_upload_alternative is False


# ---- search() behaviour ----------------------------------------------------

class _FakeResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_google_lens_uploads_directly_when_no_url_is_given(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "serpapi_key", "test-key")
    img = tmp_path / "face.jpg"
    img.write_bytes(b"\xff\xd8\xff" + b"fake-jpeg-body" * 50)  # well under 500KB

    calls = []

    def fake_urlopen(req, timeout=None):
        calls.append(req.full_url)
        if req.full_url.startswith("https://serpapi.com/image"):
            return _FakeResponse({"image_id": "abc123"})
        # The actual search call — confirm image_id made it into the query.
        assert "image_id=abc123" in req.full_url
        assert "url=" not in req.full_url
        return _FakeResponse({"visual_matches": [
            {"link": "https://example.com/found", "title": "a match"},
        ]})

    monkeypatch.setattr("facechain.search.serpapi.urllib.request.urlopen", fake_urlopen)

    adapter = SerpApiAdapter("google_lens")
    result = adapter.search(str(img), image_url=None)

    assert result.status == ProviderStatus.COMPLETED
    assert len(result.candidates) == 1
    assert any(c.startswith("https://serpapi.com/image") for c in calls)


def test_google_lens_still_uses_url_when_one_is_available(tmp_path, monkeypatch):
    """An explicit image_url must still take priority over uploading —
    uploading is a fallback for when no URL exists, not the default path."""
    monkeypatch.setattr(settings, "serpapi_key", "test-key")
    img = tmp_path / "face.jpg"
    img.write_bytes(b"fake")

    upload_called = []

    def fake_urlopen(req, timeout=None):
        if req.full_url.startswith("https://serpapi.com/image"):
            upload_called.append(True)
        assert "url=https" in req.full_url or "url=http" in req.full_url
        return _FakeResponse({"visual_matches": [{"link": "https://example.com/x", "title": ""}]})

    monkeypatch.setattr("facechain.search.serpapi.urllib.request.urlopen", fake_urlopen)

    adapter = SerpApiAdapter("google_lens")
    result = adapter.search(str(img), image_url="https://cdn.example/photo.jpg")

    assert result.status == ProviderStatus.COMPLETED
    assert upload_called == []


def test_yandex_reports_not_configured_without_a_url_not_a_generic_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "serpapi_key", "test-key")
    img = tmp_path / "face.jpg"
    img.write_bytes(b"fake")

    adapter = SerpApiAdapter("yandex_images")
    result = adapter.search(str(img), image_url=None)

    assert result.status == ProviderStatus.NOT_CONFIGURED
    assert "no upload alternative" in result.error


def test_upload_failure_is_reported_distinctly_not_as_a_result(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "serpapi_key", "test-key")
    img = tmp_path / "face.jpg"
    img.write_bytes(b"fake")

    def fake_urlopen(req, timeout=None):
        raise ConnectionError("upload endpoint unreachable")

    monkeypatch.setattr("facechain.search.serpapi.urllib.request.urlopen", fake_urlopen)

    adapter = SerpApiAdapter("google_lens")
    result = adapter.search(str(img), image_url=None)

    assert not result.ok
    assert "direct image upload to SerpAPI failed" in result.error


def test_an_oversized_image_is_recompressed_under_the_upload_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "serpapi_key", "test-key")
    from PIL import Image
    import numpy as np

    # A smooth gradient, not noise: JPEG needs a low-entropy image to have
    # room to shrink under the limit once recompressed at a lower quality.
    x = np.linspace(0, 255, 2600, dtype="uint8")
    gradient = np.tile(x, (2600, 1)).astype("uint8")
    noisy = np.clip(gradient.astype("int16") + np.random.default_rng(0).integers(-15, 15, gradient.shape),
                    0, 255).astype("uint8")
    big = Image.fromarray(np.stack([noisy, noisy.T, gradient], axis=-1))
    img = tmp_path / "big.jpg"
    big.save(img, "JPEG", quality=100, subsampling=0)
    assert img.stat().st_size > 500 * 1024, "test setup needs a file over the upload limit"

    seen_sizes = []

    def fake_urlopen(req, timeout=None):
        if req.full_url.startswith("https://serpapi.com/image"):
            seen_sizes.append(len(req.data))
            return _FakeResponse({"image_id": "abc123"})
        return _FakeResponse({"visual_matches": []})

    monkeypatch.setattr("facechain.search.serpapi.urllib.request.urlopen", fake_urlopen)

    adapter = SerpApiAdapter("google_lens")
    adapter.search(str(img), image_url=None)

    assert seen_sizes, "upload should have been attempted"
    assert seen_sizes[0] < img.stat().st_size


# ---- original-over-thumbnail priority (RCA fix) ---------------------------

def test_original_image_url_takes_priority_over_thumbnail_in_result_rows(tmp_path, monkeypatch):
    """Regression: serpapi.py used `thumbnail or original`, meaning the full-res
    `original` field (media.licdn.com, pbs.twimg.com, etc.) was silently
    discarded whenever a thumbnail was also present — which is always.

    ArcFace comparing against a 50 px Google cache thumbnail of a LinkedIn
    profile reliably scores ~0.15–0.19 even for the exact same person, which
    is below the 0.38 verification threshold.  The fix reverses the priority
    to `original or thumbnail` so the full-resolution source image is stored
    in `candidate.thumbnail` and used for face comparison instead.
    """
    monkeypatch.setattr(settings, "serpapi_key", "test-key")
    img = tmp_path / "face.jpg"
    img.write_bytes(b"x" * 1000)

    full_res_url = "https://media.licdn.com/dms/image/v2/profile.jpg"
    compressed_thumb = "https://encrypted-tbn1.gstatic.com/images?q=tbn:abc"

    def fake_urlopen(req, timeout=None):
        if "serpapi.com/image" in req.full_url:
            return _FakeResponse({"image_id": "id1"})
        # Simulate a SerpAPI visual_matches item with BOTH fields present.
        return _FakeResponse({
            "visual_matches": [{
                "link": "https://linkedin.com/in/someone",
                "title": "Someone",
                "original": full_res_url,      # full-res CDN image
                "thumbnail": compressed_thumb,  # 50px Google cache — must NOT win
            }]
        })

    monkeypatch.setattr("facechain.search.serpapi.urllib.request.urlopen", fake_urlopen)

    result = SerpApiAdapter("google_lens").search(str(img))
    assert result.candidates, "should have found a candidate"
    cand = result.candidates[0]
    assert cand.thumbnail == full_res_url, (
        f"Expected full-res original URL in thumbnail, got: {cand.thumbnail}"
    )
    assert compressed_thumb not in cand.thumbnail


def test_yandex_images_nested_link_object_does_not_crash(tmp_path, monkeypatch):
    """Regression for a real production incident: SerpAPI's Yandex Images
    engine (unlike Google Lens) represents `original_image`/`thumbnail` as
    ``{"link": "...", "width":.., "height":..}`` objects, not plain strings.
    The old code did `item.get("original") or item.get("thumbnail") or ""`
    then sliced the result — `dict[:500]` raises `KeyError: slice(...)`,
    which silently took down every Yandex candidate row. This is the actual
    real response shape captured from SerpAPI, field names included.
    """
    monkeypatch.setattr(settings, "serpapi_key", "test-key")
    img = tmp_path / "face.jpg"
    img.write_bytes(b"x" * 1000)

    def fake_urlopen(req, timeout=None):
        assert "url=https" in req.full_url or "url=http" in req.full_url
        return _FakeResponse({
            "image_results": [{
                "title": "Someone",
                "link": "https://ur.m.wikipedia.org/wiki/Someone",
                "source": "ur.m.wikipedia.org",
                "thumbnail": {
                    "link": "https://avatars.mds.yandex.net/i?id=abc",
                    "height": 90, "width": 148,
                },
                "original_image": {
                    "link": "https://upload.wikimedia.org/wikipedia/commons/photo.jpg",
                    "height": 640, "width": 480,
                },
            }]
        })

    monkeypatch.setattr("facechain.search.serpapi.urllib.request.urlopen", fake_urlopen)

    result = SerpApiAdapter("yandex_images").search(str(img), image_url="https://cdn.example/photo.jpg")

    assert result.status == ProviderStatus.COMPLETED
    assert result.candidates
    cand = result.candidates[0]
    assert cand.url == "https://ur.m.wikipedia.org/wiki/Someone"
    # The genuine full-resolution original wins over the cache thumbnail,
    # with both correctly unwrapped from their nested {"link": ...} shape.
    assert cand.thumbnail == "https://upload.wikimedia.org/wikipedia/commons/photo.jpg"


def test_thumbnail_used_when_original_is_absent(tmp_path, monkeypatch):
    """When `original` is absent the existing `thumbnail` must still be used."""
    monkeypatch.setattr(settings, "serpapi_key", "test-key")
    img = tmp_path / "face.jpg"
    img.write_bytes(b"x" * 1000)

    compressed_thumb = "https://encrypted-tbn1.gstatic.com/images?q=tbn:abc"

    def fake_urlopen(req, timeout=None):
        if "serpapi.com/image" in req.full_url:
            return _FakeResponse({"image_id": "id1"})
        return _FakeResponse({
            "visual_matches": [{
                "link": "https://linkedin.com/in/someone",
                "title": "Someone",
                "thumbnail": compressed_thumb,
                # no "original" key
            }]
        })

    monkeypatch.setattr("facechain.search.serpapi.urllib.request.urlopen", fake_urlopen)

    result = SerpApiAdapter("google_lens").search(str(img))
    assert result.candidates
    assert result.candidates[0].thumbnail == compressed_thumb


# ---- bing_reverse_image: real production incident regressions -------------

def test_bing_reverse_image_uses_the_image_url_param_not_url(tmp_path, monkeypatch):
    """Regression: SerpAPI's bing_reverse_image engine rejects the generic
    `url` parameter every other engine accepts, with HTTP 400 "Missing query
    `image_url` parameter." — it needs `image_url` instead."""
    monkeypatch.setattr(settings, "serpapi_key", "test-key")
    img = tmp_path / "face.jpg"
    img.write_bytes(b"fake")

    seen_urls = []

    def fake_urlopen(req, timeout=None):
        seen_urls.append(req.full_url)
        return _FakeResponse({"pages_with_this_image": []})

    monkeypatch.setattr("facechain.search.serpapi.urllib.request.urlopen", fake_urlopen)
    SerpApiAdapter("bing_reverse_image").search(str(img), image_url="https://cdn.example/photo.jpg")

    assert seen_urls
    query = seen_urls[0].split("?", 1)[1]
    assert "image_url=https" in query or "image_url=http" in query
    assert "&url=" not in f"&{query}"


def test_bings_own_link_is_a_viewer_permalink_source_field_is_the_real_page(tmp_path, monkeypatch):
    """Regression: bing_reverse_image's "link" field is always a bing.com
    image-viewer permalink (bing.com/images/search?view=detailv2&...), never
    the actual page the image was found on — using it produced zero usable
    candidates on every single search, even a fully successful API call,
    because bing.com is filtered as junk-domain chrome downstream. "source"
    is the genuine external page URL and must win for this engine only.
    """
    monkeypatch.setattr(settings, "serpapi_key", "test-key")
    img = tmp_path / "face.jpg"
    img.write_bytes(b"fake")

    def fake_urlopen(req, timeout=None):
        return _FakeResponse({
            "pages_with_this_image": [{
                "title": "A real article",
                "link": "https://www.bing.com/images/search?view=detailv2&id=ABC123",
                "source": "https://www.example-news.com/article-about-someone",
                "original": "https://cdn.example-news.com/full-res.jpg",
                "thumbnail": "https://tse1.mm.bing.net/th/id/tiny-cached-thumb",
            }],
        })

    monkeypatch.setattr("facechain.search.serpapi.urllib.request.urlopen", fake_urlopen)
    result = SerpApiAdapter("bing_reverse_image").search(str(img), image_url="https://cdn.example/photo.jpg")

    assert result.status == ProviderStatus.COMPLETED
    assert result.candidates
    cand = result.candidates[0]
    assert cand.url == "https://www.example-news.com/article-about-someone"
    assert "bing.com" not in cand.url
    # Original full-res image still wins over the compressed cache thumbnail.
    assert cand.thumbnail == "https://cdn.example-news.com/full-res.jpg"


def test_other_engines_still_prefer_link_over_source(tmp_path, monkeypatch):
    """The source/link swap above must stay scoped to bing_reverse_image —
    Google Lens's "source" is a plain site-name label, not a URL, and must
    never be preferred over its real "link" field."""
    monkeypatch.setattr(settings, "serpapi_key", "test-key")
    img = tmp_path / "face.jpg"
    img.write_bytes(b"fake")

    def fake_urlopen(req, timeout=None):
        if "serpapi.com/image" in req.full_url:
            return _FakeResponse({"image_id": "id1"})
        return _FakeResponse({
            "visual_matches": [{
                "link": "https://instagram.com/p/real-post/",
                "source": "Instagram",  # a label, not a URL
                "title": "A post",
            }],
        })

    monkeypatch.setattr("facechain.search.serpapi.urllib.request.urlopen", fake_urlopen)
    result = SerpApiAdapter("google_lens").search(str(img))
    assert result.candidates
    assert result.candidates[0].url == "https://instagram.com/p/real-post/"


def test_pages_with_this_image_section_is_parsed(tmp_path, monkeypatch):
    """Regression: `pages_with_this_image` — bing_reverse_image's actual
    section key for this concept — was previously entirely absent from the
    parsed SECTIONS list, silently discarding every Bing result even on a
    successful, well-formed API response."""
    monkeypatch.setattr(settings, "serpapi_key", "test-key")
    img = tmp_path / "face.jpg"
    img.write_bytes(b"fake")

    def fake_urlopen(req, timeout=None):
        return _FakeResponse({
            "pages_with_this_image": [{
                "title": "t",
                "link": "https://www.bing.com/images/search?x=1",
                "source": "https://example.com/found-here",
            }],
        })

    monkeypatch.setattr("facechain.search.serpapi.urllib.request.urlopen", fake_urlopen)
    result = SerpApiAdapter("bing_reverse_image").search(str(img), image_url="https://cdn.example/photo.jpg")
    assert result.candidates and result.candidates[0].url == "https://example.com/found-here"


def test_http_error_body_is_surfaced_instead_of_a_bare_status_message(tmp_path, monkeypatch):
    """Regression: `urllib` discards the response body on a non-2xx status by
    default, so a real, specific SerpAPI error ("Missing query `image_url`
    parameter.") was reported as an opaque "HTTPError: HTTP Error 400: Bad
    Request" — undiagnosable without a manual repro against the live API.
    """
    monkeypatch.setattr(settings, "serpapi_key", "test-key")
    img = tmp_path / "face.jpg"
    img.write_bytes(b"fake")

    def fake_urlopen(req, timeout=None):
        body = b'{"error": "Missing query `image_url` parameter."}'
        raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", None, io.BytesIO(body))

    monkeypatch.setattr("facechain.search.serpapi.urllib.request.urlopen", fake_urlopen)
    result = SerpApiAdapter("bing_reverse_image").search(str(img), image_url="https://cdn.example/photo.jpg")

    assert not result.ok
    assert "Missing query" in result.error
    assert "image_url" in result.error
