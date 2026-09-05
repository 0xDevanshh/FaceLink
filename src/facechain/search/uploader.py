"""Expose the input image at a temporary public URL.

Why this exists: every engine's *by-URL* reverse-search endpoint is far more
reliable than its drag-and-drop upload flow, and SerpAPI requires a URL. To use
those paths with a local file, the file must be reachable from the internet.

Two publication paths:

1. **Local server** (``LOCAL_IMAGE_BASE_URL`` is set): the FastAPI server
   already has a ``/api/v1/tmp-image/{token}`` route that serves any file
   registered via ``register_local_image`` / ``unregister_local_image``.
   When the caller knows the server's public base URL it can set
   ``LOCAL_IMAGE_BASE_URL`` and images are served directly — no third-party
   upload, no privacy concern, instant TTL on scan completion.

2. **Third-party host** (``allow_upload_host=True``): uploads to Litterbox
   (1h TTL) or any other host configured via ``UPLOAD_HOST``. Off by default
   because it sends the photo to an external service.

If you already host the image somewhere, pass ``--image-url`` instead and
nothing is uploaded or registered anywhere.
"""

from __future__ import annotations

import ipaddress
import logging
import secrets
import socket
import threading
import time
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse

import httpx

from ..config import settings
from ..security.ssrf import is_blocked_ip

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Local image registry — thread-safe token → (path, expiry_monotonic)
# ---------------------------------------------------------------------------
_local_registry: dict[str, tuple[Path, float]] = {}
_local_registry_lock = threading.Lock()

# How long a locally-registered image stays available after registration.
# Sized well above the search stage's total budget so it is never evicted mid-scan.
_LOCAL_TOKEN_TTL_S: float = 7200.0  # 2 hours


def register_local_image(path: Path, ttl_s: float = _LOCAL_TOKEN_TTL_S) -> str:
    """Register *path* for local serving and return the URL token.

    The token is a 32-hex-character random string.  Call
    ``unregister_local_image(token)`` when the scan is done; stale entries
    are also pruned automatically by ``local_image_for_token``.
    """
    token = secrets.token_hex(16)
    expiry = time.monotonic() + ttl_s
    with _local_registry_lock:
        _local_registry[token] = (path, expiry)
    log.debug("local_image.registered token=%s path=%s ttl_s=%.0f", token, path, ttl_s)
    return token


def unregister_local_image(token: str) -> None:
    """Remove *token* from the registry (best-effort, never raises)."""
    with _local_registry_lock:
        _local_registry.pop(token, None)
    log.debug("local_image.unregistered token=%s", token)


def local_image_for_token(token: str) -> Optional[Path]:
    """Return the path for *token* if it exists and has not expired."""
    now = time.monotonic()
    with _local_registry_lock:
        entry = _local_registry.get(token)
        if entry is None:
            return None
        path, expiry = entry
        if now > expiry:
            del _local_registry[token]
            log.debug("local_image.expired token=%s", token)
            return None
        return path


def build_local_url(token: str) -> str:
    """Construct the public URL for a locally-registered image token."""
    base = (settings.local_image_base_url or "").rstrip("/")
    return f"{base}/api/v1/tmp-image/{token}"


class UploadError(RuntimeError):
    """Raised when an image cannot be published to a fetchable public URL.

    `reason` is a short machine-checkable code so a caller (the orchestrator,
    the health endpoint) can classify the failure without parsing prose:
    ``"not_configured"`` | ``"unreachable_host"`` | ``"host_disabled"`` |
    ``"validation_failed"`` | ``"network_error"``. Left ``""`` only for
    call sites that predate this classification (none, currently).
    """

    def __init__(self, message: str, reason: str = "") -> None:
        super().__init__(message)
        self.reason = reason


def _redact(url: str) -> str:
    """Drop a query string before it ever reaches a log line — a signed URL's
    query parameters can themselves be a bearer credential."""
    return url.split("?", 1)[0] + ("?<REDACTED>" if "?" in url else "")


def _host_is_externally_routable(url: str) -> tuple[bool, str]:
    """Best-effort check that *url*'s host is not loopback/private/link-local.

    A host in one of those ranges can answer an HTTP request made from this
    same machine — which is all a fetch-based check proves — while being
    genuinely unreachable from a reverse-image engine's own infrastructure.
    ``http://localhost:8000`` and ``http://192.168.1.5:8000`` both "work" when
    curled from the server itself and are exactly the case this guards
    against: pretending a URL is public when it is not.

    Returns ``(True, "")`` when the host looks publicly reachable, or
    ``(False, reason)`` naming why not. Unparseable/unresolvable hosts are
    treated as NOT externally routable — the same conservative default used by
    `security.ssrf`.
    """
    try:
        host = urlparse(url).hostname or ""
    except ValueError:
        return False, "URL has no parseable host"
    if not host:
        return False, "URL has no host"
    if host.lower() in ("localhost", "localhost.localdomain"):
        return False, f"{host!r} is this machine, not a publicly reachable address"

    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        addr = None
    if addr is not None:
        if is_blocked_ip(str(addr)):
            return False, f"{host} is a private/loopback/link-local address"
        return True, ""

    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        return False, f"DNS resolution failed for {host!r}: {exc}"
    if not infos:
        return False, f"no DNS records for {host!r}"
    for info in infos:
        ip = info[4][0]
        if is_blocked_ip(ip):
            return False, f"{host!r} resolves to a private/loopback/link-local address ({ip})"
    return True, ""


def hosting_health() -> dict:
    """Cheap, no-network-call snapshot of the image-hosting path's readiness.

    Reachability is classified from the URL's host alone (IP-literal check or
    DNS resolution) — never a live HTTP fetch — so this is safe to call on
    every ``/health`` poll. The real fetch-and-check (`_validate_remote`)
    still runs once, at actual publish time.
    """
    base = (settings.local_image_base_url or "").strip()
    if base:
        routable, why = _host_is_externally_routable(base)
        return {
            "mode": "local",
            "configured": True,
            "reachable": routable,
            "reason": "" if routable else why,
            "fallback_configured": settings.allow_upload_host,
        }
    if settings.allow_upload_host:
        return {
            "mode": "third_party",
            "configured": True,
            "reachable": None,  # not probed until a scan actually needs it
            "reason": "",
            "fallback_configured": False,
        }
    return {
        "mode": "none",
        "configured": False,
        "reachable": False,
        "reason": "neither LOCAL_IMAGE_BASE_URL nor ALLOW_UPLOAD_HOST is set",
        "fallback_configured": False,
    }


def _validate_remote(url: str) -> None:
    """Confirm the published URL is actually fetchable before trusting it.

    A host that returns 200 with a URL that doesn't resolve, or resolves to
    something that isn't an image, is a failure a caller should see now — not
    several engine calls later as a string of confusing per-provider errors.
    Uses HEAD first (cheap); falls back to a streamed, immediately-closed GET
    for hosts that reject HEAD, so the image body is never fully downloaded
    just to validate it.
    """
    ctype = ""
    resp_status = 0
    try:
        resp = httpx.head(url, timeout=settings.http_timeout_s, follow_redirects=True,
                          headers={"User-Agent": settings.user_agent})
        if resp.status_code >= 400:
            raise httpx.HTTPStatusError("HEAD rejected", request=resp.request, response=resp)
        ctype, resp_status = resp.headers.get("content-type", ""), resp.status_code
    except Exception:  # noqa: BLE001 — HEAD unsupported/rejected; fall back to GET
        try:
            with httpx.stream("GET", url, timeout=settings.http_timeout_s, follow_redirects=True,
                              headers={"User-Agent": settings.user_agent}) as resp:
                ctype, resp_status = resp.headers.get("content-type", ""), resp.status_code
        except Exception as exc:  # noqa: BLE001
            raise UploadError(
                f"published URL is not fetchable: {type(exc).__name__}: {exc}",
                reason="network_error",
            ) from exc
        if resp_status >= 400:
            raise UploadError(
                f"published URL returned HTTP {resp_status} on validation fetch",
                reason="validation_failed",
            )

    if not ctype.lower().startswith("image/"):
        raise UploadError(
            f"published URL did not return image content (HTTP {resp_status}, "
            f"content-type={ctype or 'none'})",
            reason="validation_failed",
        )


def publish_local(image_path: str | Path, ttl_s: float = _LOCAL_TOKEN_TTL_S) -> str:
    """Register *image_path* with the local token registry and return its URL.

    Requires ``settings.local_image_base_url`` to be set to the server's own
    public base URL (e.g. ``https://myserver.example.com:8000``).  When set,
    this is the preferred path: it never contacts a third-party service, the
    image stays on the host machine, and the token expires automatically.

    Raises ``UploadError`` if ``local_image_base_url`` is not configured.
    """
    base = (settings.local_image_base_url or "").strip()
    if not base:
        raise UploadError(
            "LOCAL_IMAGE_BASE_URL is not set — cannot publish via local server",
            reason="not_configured",
        )
    routable, why = _host_is_externally_routable(base)
    if not routable:
        raise UploadError(
            f"LOCAL_IMAGE_BASE_URL is not externally reachable — {why}. Point it at a "
            "public tunnel/host, or enable ALLOW_UPLOAD_HOST as a fallback.",
            reason="unreachable_host",
        )
    path = Path(image_path)
    token = register_local_image(path, ttl_s=ttl_s)
    url = build_local_url(token)
    log.info("local_image.published token=%s url=%s", token, _redact(url))
    return url


def _upload_to_litterbox(path: Path, expiry: str) -> str:
    """POST to Litterbox (catbox.moe), return the raw URL string it reports.

    Does not validate the URL is actually fetchable — that is the shared
    `_validate_remote` step every host in the fallback chain goes through
    identically, in `publish_temporarily`.
    """
    with open(path, "rb") as fh:
        resp = httpx.post(
            settings.upload_host,
            data={"reqtype": "fileupload", "time": expiry},
            files={"fileToUpload": (path.name, fh, "application/octet-stream")},
            timeout=settings.http_timeout_s * 2,
            headers={"User-Agent": settings.user_agent},
        )
    body = (resp.text or "").strip()
    if resp.status_code != 200 or not body.startswith("http"):
        raise UploadError(
            f"host returned {resp.status_code}: {body[:200]}", reason="validation_failed"
        )
    return body


def _upload_to_uguu(path: Path) -> str:
    """POST to uguu.se, return the raw image URL from its JSON response.

    A second, independent anonymous host — used only as a fallback when
    `upload_host` (Litterbox) fails. Chosen after directly verifying (not
    assuming) that its returned links serve raw image bytes with no HTML
    wrapper, no cookies, and no redirect, unlike some superficially similar
    free hosts (tmpfiles.org's links always resolve to an HTML preview page;
    file.io's public upload endpoint now redirects to a marketing site).
    """
    with open(path, "rb") as fh:
        resp = httpx.post(
            settings.upload_fallback_host,
            files={"files[]": (path.name, fh, "application/octet-stream")},
            timeout=settings.http_timeout_s * 2,
            headers={"User-Agent": settings.user_agent},
        )
    if resp.status_code != 200:
        raise UploadError(
            f"host returned {resp.status_code}: {resp.text[:200]}", reason="validation_failed"
        )
    try:
        payload = resp.json()
        url = payload["files"][0]["url"]
    except Exception as exc:  # noqa: BLE001 — malformed/unexpected JSON shape
        raise UploadError(
            f"unexpected response shape: {type(exc).__name__}: {resp.text[:200]}",
            reason="validation_failed",
        ) from exc
    if not isinstance(url, str) or not url.startswith("http"):
        raise UploadError(f"response did not contain a usable URL: {str(payload)[:200]}",
                          reason="validation_failed")
    return url


# Tried in order. Each entry is (display name, upload function). A host that
# raises `UploadError` — including one rejected by `_validate_remote` after
# upload — is skipped in favour of the next; `publish_temporarily` only
# raises once every entry here has been tried and failed.
_THIRD_PARTY_HOSTS: tuple[tuple[str, "Callable[[Path], str]"], ...] = (
    ("litterbox", lambda path: _upload_to_litterbox(path, "1h")),
    ("uguu", _upload_to_uguu),
)


def publish_temporarily(image_path: str | Path, expiry: str = "1h") -> str:
    """Try each configured third-party host in order, validate, return the URL.

    Raises ``UploadError`` only after every host has been tried and failed —
    including a host that "succeeds" with a 200 but returns something that
    isn't a working image URL (an HTML wrapper, a login/redirect page, a
    bot-challenge response). The failure carries the most recent host's
    reason unless every host was tried, in which case `reason` is
    ``"all_hosts_exhausted"`` so a caller can tell "one host had a problem"
    from "there is no working hosting path at all right now".
    """
    if not settings.allow_upload_host:
        raise UploadError(
            "temporary hosting disabled (pass --allow-upload-host to enable)",
            reason="host_disabled",
        )

    path = Path(image_path)
    size = path.stat().st_size if path.exists() else 0
    last_exc: UploadError | None = None

    for name, upload_fn in _THIRD_PARTY_HOSTS:
        started = time.monotonic()
        log.info("temporary_image_publish.start provider=%s bytes=%d", name, size)
        try:
            url = upload_fn(path)
            _validate_remote(url)
        except UploadError as exc:
            elapsed_ms = int((time.monotonic() - started) * 1000)
            log.warning("temporary_image_publish.failure provider=%s elapsed_ms=%d reason=%s error=%s",
                       name, elapsed_ms, exc.reason, exc)
            last_exc = exc
            continue
        except Exception as exc:  # noqa: BLE001 — network/transport failure
            elapsed_ms = int((time.monotonic() - started) * 1000)
            log.warning("temporary_image_publish.failure provider=%s elapsed_ms=%d error=%s",
                       name, elapsed_ms, type(exc).__name__)
            last_exc = UploadError(
                f"upload failed: {type(exc).__name__}: {exc}", reason="network_error"
            )
            continue

        elapsed_ms = int((time.monotonic() - started) * 1000)
        log.info("temporary_image_publish.success provider=%s elapsed_ms=%d url=%s",
                 name, elapsed_ms, _redact(url))
        return url

    tried = ", ".join(name for name, _ in _THIRD_PARTY_HOSTS)
    raise UploadError(
        f"every configured temporary-hosting provider failed ({tried}); "
        f"most recent error: {last_exc}",
        reason="all_hosts_exhausted",
    ) from last_exc


def publish_image(image_path: str | Path, *, validate: bool = False) -> str:
    """Publish *image_path* using the best available method and return a URL.

    Priority:
      1. Local server (``LOCAL_IMAGE_BASE_URL`` set) — no external service,
         no privacy concern, instant availability.
      2. Third-party host (``allow_upload_host=True``) — Litterbox or
         whatever ``UPLOAD_HOST`` is configured to.

    Raises ``UploadError`` if neither path is available.
    """
    # Try local server first — preferred when available.
    if (settings.local_image_base_url or "").strip():
        try:
            url = publish_local(image_path)
            if validate:
                try:
                    _validate_remote(url)
                except UploadError:
                    unregister_local_image(url.rsplit("/", 1)[-1])
                    raise
            return url
        except UploadError as exc:
            if not settings.allow_upload_host:
                raise
            log.warning("local image publication failed; using configured fallback: %s", exc)
    if settings.allow_upload_host:
        return publish_temporarily(image_path)
    raise UploadError(
        "no image publication method available — set LOCAL_IMAGE_BASE_URL "
        "(recommended) or ALLOW_UPLOAD_HOST=true",
        reason="not_configured",
    )


def _diagnose(image_path: str) -> int:
    """`python -m facechain.search.uploader <path>` — safe-only diagnostics.

    Prints only non-secret information: never the upload host's response
    body verbatim (it could echo request details) and never any credential.
    """
    path = Path(image_path)
    local_base = (settings.local_image_base_url or "").strip()
    print(f"local_image_base_url: {local_base or '(not set)'}")
    print(f"allow_upload_host: {settings.allow_upload_host}")
    hosts = ", ".join(name for name, _ in _THIRD_PARTY_HOSTS)
    print(f"third-party hosts (tried in order): {hosts}")
    if not path.exists():
        print("input readable: no (file not found)")
        return 1
    size = path.stat().st_size
    print("input readable: yes")
    print(f"size: {size / 1024:.1f} KB")
    try:
        from PIL import Image
        with Image.open(path) as img:
            print(f"image type: {img.format}")
    except Exception as exc:  # noqa: BLE001
        print(f"image type: could not decode ({type(exc).__name__})")
        return 1

    if local_base:
        print("publication: local server (no upload)")
        token = register_local_image(path)
        url = build_local_url(token)
        print(f"local URL: {url}")
        unregister_local_image(token)
        return 0

    if not settings.allow_upload_host:
        print("publication: skipped (LOCAL_IMAGE_BASE_URL not set and ALLOW_UPLOAD_HOST is false)")
        return 0

    try:
        publish_temporarily(path)
        print("publication: success")
        print("remote fetch validation: success")
    except UploadError as exc:
        print(f"publication: FAILED — {exc}")
        return 1
    return 0


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("usage: python -m facechain.search.uploader <image_path>")
        raise SystemExit(2)
    raise SystemExit(_diagnose(sys.argv[1]))
