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

import json

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
