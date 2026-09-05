"""Temporary-hosting gate: off by default, and never silently bypassed.

All offline — the HTTP client is stubbed at the httpx-transport boundary.
"""

from __future__ import annotations

import socket

import httpx
import pytest

from facechain.config import settings
from facechain.search import uploader as uploader_mod
from facechain.search.uploader import (
    UploadError,
    _host_is_externally_routable,
    hosting_health,
    publish_image,
    publish_local,
    publish_temporarily,
)


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
        if request.method == "POST":
            return httpx.Response(200, text="https://litter.catbox.moe/abc123.jpg")
        # The post-publish remote validation HEAD/GET.
        return httpx.Response(200, headers={"content-type": "image/jpeg"})

    mock_client = httpx.Client(transport=httpx.MockTransport(fake_transport))
    monkeypatch.setattr(httpx, "post", lambda url, **kw: mock_client.post(url, **kw))
    monkeypatch.setattr(httpx, "head", lambda url, **kw: mock_client.head(url, **kw))

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


# ---- UploadError.reason — machine-checkable failure classification --------

def test_the_disabled_gate_reports_reason_host_disabled(tmp_path):
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake-jpeg-bytes")
    with pytest.raises(UploadError) as exc_info:
        publish_temporarily(img)
    assert exc_info.value.reason == "host_disabled"


def test_a_network_error_that_exhausts_every_host_reports_all_hosts_exhausted(tmp_path, monkeypatch):
    # publish_temporarily tries every configured third-party host (Litterbox,
    # then its fallback) before giving up — a network error that breaks every
    # one of them is a distinct, more precise fact ("no working hosting path
    # exists right now") than any single host's own failure reason, so the
    # terminal reason here is "all_hosts_exhausted", not "network_error".
    monkeypatch.setattr(settings, "allow_upload_host", True)
    monkeypatch.setattr(
        httpx, "post",
        lambda *a, **kw: (_ for _ in ()).throw(httpx.ConnectError("connection refused")),
    )
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake-jpeg-bytes")
    with pytest.raises(UploadError, match="ConnectError") as exc_info:
        publish_temporarily(img)
    assert exc_info.value.reason == "all_hosts_exhausted"


# ---- fallback host selection -----------------------------------------------
#
# Regression for a real production incident: Litterbox intermittently answers
# an automated upload with a bot-challenge 403 page (Cloudflare-style CSP
# nonces, no real error body) rather than a genuine error — not something to
# bypass, just a provider that is unavailable right now. publish_temporarily
# must move on to the next configured host rather than surfacing that 403 as
# the final failure.

def test_falls_back_to_the_next_host_when_litterbox_is_waf_challenged(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "allow_upload_host", True)

    waf_challenge_body = (
        '<!doctype html><html lang="en"><meta charset="UTF-8">'
        '<title>403 | Forbidden</title></html>'
    )

    def fake_transport(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and "litterbox" in str(request.url):
            return httpx.Response(403, text=waf_challenge_body,
                                  headers={"content-type": "text/html; charset=utf-8"})
        if request.method == "POST" and "uguu" in str(request.url):
            return httpx.Response(200, json={
                "success": True,
                "files": [{"url": "https://h.uguu.se/abc123.jpg", "mimetype": "image/jpeg"}],
            })
        # The post-publish remote validation HEAD/GET against uguu's URL.
        return httpx.Response(200, headers={"content-type": "image/jpeg"})

    mock_client = httpx.Client(transport=httpx.MockTransport(fake_transport))
    monkeypatch.setattr(httpx, "post", lambda url, **kw: mock_client.post(url, **kw))
    monkeypatch.setattr(httpx, "head", lambda url, **kw: mock_client.head(url, **kw))

    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake-jpeg-bytes")
    url = publish_temporarily(img)
    assert url == "https://h.uguu.se/abc123.jpg"


def test_uguu_response_with_no_usable_url_is_reported_as_an_upload_error(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "allow_upload_host", True)

    def fake_transport(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and "litterbox" in str(request.url):
            return httpx.Response(403, text="forbidden")
        if request.method == "POST" and "uguu" in str(request.url):
            return httpx.Response(200, json={"success": False, "files": [], "errors": ["nope"]})
        return httpx.Response(200, headers={"content-type": "image/jpeg"})

    monkeypatch.setattr(
        httpx, "post",
        lambda url, **kw: httpx.Client(transport=httpx.MockTransport(fake_transport)).post(url, **kw),
    )
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake-jpeg-bytes")
    with pytest.raises(UploadError) as exc_info:
        publish_temporarily(img)
    assert exc_info.value.reason == "all_hosts_exhausted"


# ---- external reachability: don't pretend a private address is public -----
#
# A host that answers an HTTP request made from this same machine "works"
# when curled locally while being genuinely unreachable from a reverse-image
# engine's own infrastructure — this is precisely the gap that let
# `LOCAL_IMAGE_BASE_URL=http://localhost:8000` silently masquerade as a public
# URL. See `_host_is_externally_routable` and mission Section 1.

def test_loopback_ip_is_not_externally_routable():
    routable, why = _host_is_externally_routable("http://127.0.0.1:8000")
    assert routable is False
    assert why


def test_localhost_hostname_is_not_externally_routable():
    routable, why = _host_is_externally_routable("http://localhost:8000")
    assert routable is False


def test_private_lan_address_is_not_externally_routable():
    routable, why = _host_is_externally_routable("http://192.168.1.5:9000")
    assert routable is False


def test_link_local_address_is_not_externally_routable():
    routable, why = _host_is_externally_routable("http://169.254.1.1:8000")
    assert routable is False


def test_an_unresolvable_hostname_is_not_externally_routable(monkeypatch):
    monkeypatch.setattr(
        socket, "getaddrinfo",
        lambda *a, **kw: (_ for _ in ()).throw(socket.gaierror("name or service not known")),
    )
    routable, why = _host_is_externally_routable("https://does-not-exist.invalid")
    assert routable is False
    assert "DNS" in why


def test_a_resolvable_public_hostname_is_externally_routable(monkeypatch):
    monkeypatch.setattr(
        socket, "getaddrinfo",
        lambda *a, **kw: [(socket.AF_INET, None, None, "", ("93.184.216.34", 0))],
    )
    routable, why = _host_is_externally_routable("https://public.example.com")
    assert routable is True
    assert why == ""


def test_publish_local_rejects_an_unreachable_loopback_base_url(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "local_image_base_url", "http://127.0.0.1:8000")
    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake-jpeg-bytes")
    with pytest.raises(UploadError, match="not externally reachable") as exc_info:
        publish_local(img)
    assert exc_info.value.reason == "unreachable_host"


def test_publish_image_falls_back_past_an_unreachable_local_host(tmp_path, monkeypatch):
    # LOCAL_IMAGE_BASE_URL is "configured" but points at localhost — must not
    # be trusted as public. With ALLOW_UPLOAD_HOST also set, publish_image
    # should fall through to the third-party path exactly as it already does
    # for any other UploadError.
    monkeypatch.setattr(settings, "local_image_base_url", "http://localhost:8000")
    monkeypatch.setattr(settings, "allow_upload_host", True)

    def fake_transport(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(200, text="https://litter.catbox.moe/abc123.jpg")
        return httpx.Response(200, headers={"content-type": "image/jpeg"})

    mock_client = httpx.Client(transport=httpx.MockTransport(fake_transport))
    monkeypatch.setattr(httpx, "post", lambda url, **kw: mock_client.post(url, **kw))
    monkeypatch.setattr(httpx, "head", lambda url, **kw: mock_client.head(url, **kw))

    img = tmp_path / "photo.jpg"
    img.write_bytes(b"fake-jpeg-bytes")
    url = publish_image(img)
    assert url == "https://litter.catbox.moe/abc123.jpg"


# ---- hosting_health() — cheap, no-network-call readiness snapshot ---------

def test_hosting_health_reports_none_when_nothing_is_configured(monkeypatch):
    monkeypatch.setattr(settings, "local_image_base_url", "")
    monkeypatch.setattr(settings, "allow_upload_host", False)
    health = hosting_health()
    assert health == {
        "mode": "none",
        "configured": False,
        "reachable": False,
        "reason": "neither LOCAL_IMAGE_BASE_URL nor ALLOW_UPLOAD_HOST is set",
        "fallback_configured": False,
    }


def test_hosting_health_flags_a_loopback_local_url_as_unreachable(monkeypatch):
    monkeypatch.setattr(settings, "local_image_base_url", "http://127.0.0.1:8000")
    monkeypatch.setattr(settings, "allow_upload_host", False)
    health = hosting_health()
    assert health["mode"] == "local"
    assert health["configured"] is True
    assert health["reachable"] is False
    assert health["reason"]


def test_hosting_health_reports_third_party_mode_when_enabled(monkeypatch):
    monkeypatch.setattr(settings, "local_image_base_url", "")
    monkeypatch.setattr(settings, "allow_upload_host", True)
    health = hosting_health()
    assert health["mode"] == "third_party"
    assert health["configured"] is True
    assert health["reachable"] is None  # not probed until a scan needs it


def test_hosting_health_reports_a_reachable_local_host(monkeypatch):
    monkeypatch.setattr(settings, "local_image_base_url", "https://public.example.com")
    monkeypatch.setattr(uploader_mod, "_host_is_externally_routable", lambda url: (True, ""))
    health = hosting_health()
    assert health["mode"] == "local"
    assert health["reachable"] is True
    assert health["reason"] == ""
