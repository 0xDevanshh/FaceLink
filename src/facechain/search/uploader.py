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
from pathlib import Path

import httpx

from ..config import settings

log = logging.getLogger(__name__)


class UploadError(RuntimeError):
    pass


def publish_temporarily(image_path: str | Path, expiry: str = "1h") -> str:
    """Upload and return a public URL. Raises `UploadError` on failure."""
    if not settings.allow_upload_host:
        raise UploadError("temporary hosting disabled (pass --allow-upload-host to enable)")

    path = Path(image_path)
    log.info("uploading %s to %s (expires in %s)", path.name, settings.upload_host, expiry)
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
        raise UploadError(f"upload failed: {type(exc).__name__}: {exc}") from exc

    body = (resp.text or "").strip()
    if resp.status_code != 200 or not body.startswith("http"):
        raise UploadError(f"host returned {resp.status_code}: {body[:200]}")
    log.info("temporary public URL: %s", body)
    return body
