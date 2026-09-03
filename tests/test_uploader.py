"""Temporary-hosting gate: off by default, and never silently bypassed.

All offline — the HTTP client is stubbed at the httpx-transport boundary.
"""

from __future__ import annotations

import httpx
import pytest

from facechain.config import settings
from facechain.search.uploader import UploadError, publish_temporarily


def test_disabled_by_default_regardless_of_the_developers_local_env(tmp_path):
    """Regression: `Settings` reads the developer's local `.env`
    unconditionally, so a real dev setup with uploads enabled for actual scans
    must not leak into what a test observes without opting in explicitly.
    `tests/conftest.py::_never_upload_for_real` is what guarantees this."""
    assert settings.allow_upload_host is False
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake-jpeg-bytes")
    with pytest.raises(UploadError, match="disabled"):
        publish_temporarily(img)


def test_enabling_the_gate_allows_a_real_upload_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "allow_upload_host", True)

    def fake_transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="https://litter.catbox.moe/abc123.jpg")

    monkeypatch.setattr(
        httpx, "post",
        lambda url, **kw: httpx.Client(transport=httpx.MockTransport(fake_transport)).post(url, **kw),
    )

    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake-jpeg-bytes")
    url = publish_temporarily(img)
    assert url == "https://litter.catbox.moe/abc123.jpg"


def test_a_non_200_response_is_reported_as_an_upload_error(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "allow_upload_host", True)

    def fake_transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="upstream unavailable")

    monkeypatch.setattr(
        httpx, "post",
        lambda url, **kw: httpx.Client(transport=httpx.MockTransport(fake_transport)).post(url, **kw),
    )

    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake-jpeg-bytes")
    with pytest.raises(UploadError, match="503"):
        publish_temporarily(img)


def test_a_response_body_that_is_not_a_url_is_reported_as_an_upload_error(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "allow_upload_host", True)

    def fake_transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="Error: file too large")

    monkeypatch.setattr(
        httpx, "post",
        lambda url, **kw: httpx.Client(transport=httpx.MockTransport(fake_transport)).post(url, **kw),
    )

    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake-jpeg-bytes")
    with pytest.raises(UploadError):
        publish_temporarily(img)


def test_a_network_error_is_wrapped_as_an_upload_error(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "allow_upload_host", True)
    monkeypatch.setattr(
        httpx, "post",
        lambda *a, **kw: (_ for _ in ()).throw(httpx.ConnectError("connection refused")),
    )

    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake-jpeg-bytes")
    with pytest.raises(UploadError, match="ConnectError"):
        publish_temporarily(img)
