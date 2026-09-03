"""Optional: expose the input image at a temporary public URL.

Why this exists: every engine's *by-URL* reverse-search endpoint is far more
reliable than its drag-and-drop upload flow, and SerpAPI requires a URL. To use
those paths with a local file, the file must be reachable from the internet.

This is OFF by default and must be enabled explicitly (`--allow-upload-host`),
because it uploads the photo to a third-party host. Default host is Litterbox,
which auto-deletes after 1 hour. If you already host the image somewhere, pass
`--image-url` instead and nothing is uploaded anywhere.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import httpx

from ..config import settings

log = logging.getLogger(__name__)


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


def publish_temporarily(image_path: str | Path, expiry: str = "1h") -> str:
    """Upload, verify the result is actually fetchable, and return the URL.

    Raises `UploadError` on any failure — including a host that "succeeds"
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


def _diagnose(image_path: str) -> int:
    """`python -m facechain.search.uploader <path>` — safe-only diagnostics.

    Prints only non-secret information: never the upload host's response
    body verbatim (it could echo request details) and never any credential.
    """
    path = Path(image_path)
    print(f"backend: {settings.upload_host.split('/')[2] if '/' in settings.upload_host else settings.upload_host}")
    print(f"allow_upload_host: {settings.allow_upload_host}")
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

    if not settings.allow_upload_host:
        print("publication: skipped (ALLOW_UPLOAD_HOST is false)")
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
