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

import logging
import secrets
import threading
import time
from pathlib import Path
from typing import Optional

import httpx

from ..config import settings

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
    pass


def _redact(url: str) -> str:
    """Drop a query string before it ever reaches a log line — a signed URL's
    query parameters can themselves be a bearer credential."""
    return url.split("?", 1)[0] + ("?<REDACTED>" if "?" in url else "")


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
                f"published URL is not fetchable: {type(exc).__name__}: {exc}"
            ) from exc
        if resp_status >= 400:
            raise UploadError(f"published URL returned HTTP {resp_status} on validation fetch")

    if not ctype.lower().startswith("image/"):
        raise UploadError(
            f"published URL did not return image content (HTTP {resp_status}, "
            f"content-type={ctype or 'none'})"
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
            "LOCAL_IMAGE_BASE_URL is not set — cannot publish via local server"
        )
    path = Path(image_path)
    token = register_local_image(path, ttl_s=ttl_s)
    url = build_local_url(token)
    log.info("local_image.published token=%s url=%s", token, _redact(url))
    return url


def publish_temporarily(image_path: str | Path, expiry: str = "1h") -> str:
    """Upload to the configured third-party host, validate, and return the URL.

    Raises ``UploadError`` on any failure — including a host that "succeeds"
    with a 200 but returns something that isn't a working image URL.
    """
    if not settings.allow_upload_host:
        raise UploadError("temporary hosting disabled (pass --allow-upload-host to enable)")

    path = Path(image_path)
    size = path.stat().st_size if path.exists() else 0
    started = time.monotonic()
    log.info("temporary_image_publish.start provider=%s bytes=%d", settings.upload_host, size)
    try:
        with open(path, "rb") as fh:
            resp = httpx.post(
                settings.upload_host,
                data={"reqtype": "fileupload", "time": expiry},
                files={"fileToUpload": (path.name, fh, "application/octet-stream")},
                timeout=settings.http_timeout_s * 2,
                headers={"User-Agent": settings.user_agent},
            )
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = int((time.monotonic() - started) * 1000)
        log.warning("temporary_image_publish.failure provider=%s elapsed_ms=%d error=%s",
                   settings.upload_host, elapsed_ms, type(exc).__name__)
        raise UploadError(f"upload failed: {type(exc).__name__}: {exc}") from exc

    elapsed_ms = int((time.monotonic() - started) * 1000)
    body = (resp.text or "").strip()
    if resp.status_code != 200 or not body.startswith("http"):
        log.warning("temporary_image_publish.failure provider=%s status=%d elapsed_ms=%d",
                   settings.upload_host, resp.status_code, elapsed_ms)
        raise UploadError(f"host returned {resp.status_code}: {body[:200]}")

    try:
        _validate_remote(body)
    except UploadError:
        log.warning("temporary_image_publish.validation_failure provider=%s url=%s",
                   settings.upload_host, _redact(body))
        raise

    log.info("temporary_image_publish.success provider=%s elapsed_ms=%d url=%s",
             settings.upload_host, elapsed_ms, _redact(body))
    return body


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
        "(recommended) or ALLOW_UPLOAD_HOST=true"
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
    print(f"third-party backend: {settings.upload_host.split('/')[2] if '/' in settings.upload_host else settings.upload_host}")
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
