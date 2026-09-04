"""Luxand.cloud face-recognition adapter.

Uses the ``/photo/search/v2`` endpoint to cross-check ArcFace cosine similarity
with an independent cloud recognition service.  Results are purely additive to
the existing verification ladder — they never lower a score, and the adapter is
skipped entirely when ``LUXAND_API_KEY`` is not set.

Endpoint: POST https://api.luxand.cloud/photo/search/v2
Docs:     https://luxand.cloud/
Auth:     ``token`` request header (bearer-style API key)

The adapter returns a ``LuxandResult`` with:
  - ``matched`` (bool): Luxand independently found a face match.
  - ``confidence`` (float 0–1): highest match confidence returned by Luxand.
  - ``note`` (str): reason if skipped or failed.

This result is attached to ``VerifiedCandidate.luxand`` in a future model
extension; for now it is logged and used only to emit a pipeline event.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import numpy as np

from ..config import settings

log = logging.getLogger(__name__)

ENDPOINT = "https://api.luxand.cloud/photo/search/v2"
# Luxand confidence threshold for calling a result a match.
# The API returns values in [0, 1]; ≥ 0.85 is a confident recognition.
LUXAND_MATCH_THRESHOLD = 0.85
# Hard timeout for one Luxand call — kept well below the engine timeout so it
# can never stall a scan that is already close to its wall-clock budget.
LUXAND_TIMEOUT_S = 15


@dataclass
class LuxandResult:
    matched: bool = False
    confidence: float = 0.0
    faces_found: int = 0
    note: str = ""


def _not_configured() -> LuxandResult:
    return LuxandResult(note="LUXAND_API_KEY not set — skipped")


def search_face(image_path: str | Path | bytes) -> LuxandResult:
    """POST *image_path* to Luxand ``/photo/search/v2`` and return the result.

    Accepts a filesystem path, a ``Path`` object, or raw image bytes.
    Returns a ``LuxandResult`` with ``matched=False`` and an explanatory
    ``note`` on any failure rather than raising — the caller treats Luxand as
    an optional second opinion, not a hard dependency.
    """
    if not settings.luxand_api_key:
        return _not_configured()

    try:
        if isinstance(image_path, bytes):
            raw = image_path
            fname = "image.jpg"
        else:
            p = Path(image_path)
            raw = p.read_bytes()
            fname = p.name

        resp = httpx.post(
            ENDPOINT,
            headers={"token": settings.luxand_api_key},
            files={"photo": (fname, io.BytesIO(raw), "application/octet-stream")},
            timeout=LUXAND_TIMEOUT_S,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("luxand.search_face network error: %s", exc)
        return LuxandResult(note=f"network error: {type(exc).__name__}: {str(exc)[:120]}")

    if resp.status_code == 401:
        return LuxandResult(note="LUXAND_API_KEY invalid or expired")
    if resp.status_code == 402:
        return LuxandResult(note="Luxand quota exhausted — upgrade plan")
    if resp.status_code != 200:
        return LuxandResult(note=f"HTTP {resp.status_code}")

    try:
        payload = resp.json()
    except Exception:  # noqa: BLE001
        return LuxandResult(note="could not parse Luxand JSON response")

    # Response schema (array of face objects):
    # [{"name": "...", "probability": 0.97, "uuid": "...", ...}, ...]
    # An empty array means no face was found or no match in the collection.
    if not isinstance(payload, list) or not payload:
        return LuxandResult(faces_found=0, note="no faces or matches returned")

    best_confidence = max(
        (float(item.get("probability", 0.0)) for item in payload if isinstance(item, dict)),
        default=0.0,
    )
    matched = best_confidence >= LUXAND_MATCH_THRESHOLD
    return LuxandResult(
        matched=matched,
        confidence=best_confidence,
        faces_found=len(payload),
        note="ok" if matched else f"best confidence {best_confidence:.3f} < {LUXAND_MATCH_THRESHOLD}",
    )
