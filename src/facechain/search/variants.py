"""Generate a bounded set of search variants from the query image.

Submitting only the original upload to reverse-image engines misses hits where
the subject is small in the frame (a group photo, a full-body shot) — cropping
tighter around the face is often the difference between zero hits and several.
But every extra variant means another full pass across every configured
engine, so this module exists to keep that fan-out *deliberate and bounded*
rather than unlimited.

Two rules:

1. **Budgeted, not exhaustive.** `VARIANT_BUDGETS` caps the total variants
   (including the original) per scan depth. Nobody asked for ten searches per
   scan; a handful of well-chosen crops beats many redundant ones.

2. **Deduplicated by content, not intent.** A "tight crop" of a photo that is
   already a tight headshot is pixel-for-pixel close to the original — running
   both would burn budget on a second, near-identical search. Perceptual-hash
   distance decides whether a candidate variant earns its slot.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import imagehash

from ..evidence.hashing import sha256_bytes
from ..face.detector import DetectedFace
from ..face.encoder import crop_face
from ..verification.image_similarity import phash_hex

# Total variants per scan depth, the original upload/crop always included.
VARIANT_BUDGETS: dict[str, int] = {
    "fast": 1,
    "standard": 2,
    "deep": 3,
}

# Two variants whose pHash Hamming distance is at or below this are considered
# the same search from an engine's point of view — not worth a second pass.
DEDUP_HAMMING_THRESHOLD = 6


@dataclass
class SearchVariant:
    variant_id: str
    variant_type: str  # original | tight_crop | loose_crop
    image_path: str
    sha256: str
    width: int
    height: int


def _write_temp_png(image_bgr: np.ndarray, prefix: str) -> tuple[str, str]:
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        raise ValueError("could not encode search variant as PNG")
    data = buf.tobytes()
    fd, path = tempfile.mkstemp(prefix=f"{prefix}_", suffix=".png")
    with open(fd, "wb") as fh:
        fh.write(data)
    return path, sha256_bytes(data)


def _hamming(a: str, b: str) -> int:
    return imagehash.hex_to_hash(a) - imagehash.hex_to_hash(b)


def generate_variants(
    image_bgr: np.ndarray,
    original_path: str,
    face: DetectedFace | None,
    scan_depth: str,
) -> list[SearchVariant]:
    """Build up to `VARIANT_BUDGETS[scan_depth]` distinct search variants.

    The first variant is always the image already chosen upstream to search
    (the operator's crop, or the full upload) — its bytes are reused as-is
    rather than re-encoded, so its hash matches what the rest of the pipeline
    already recorded for it.
    """
    budget = VARIANT_BUDGETS.get(scan_depth, VARIANT_BUDGETS["standard"])
    h, w = image_bgr.shape[:2]

    with open(original_path, "rb") as fh:
        original_bytes = fh.read()
    variants = [
        SearchVariant(
            variant_id="v0-original",
            variant_type="original",
            image_path=original_path,
            sha256=sha256_bytes(original_bytes),
            width=w,
            height=h,
        )
    ]
    if budget <= 1 or face is None:
        return variants

    seen_hashes = [phash_hex(image_bgr)]
    candidates: list[tuple[str, np.ndarray]] = []
    if budget >= 2:
        candidates.append(("tight_crop", crop_face(image_bgr, face, margin=0.10)))
    if budget >= 3:
        candidates.append(("loose_crop", crop_face(image_bgr, face, margin=0.60)))

    for i, (variant_type, crop_bgr) in enumerate(candidates, start=1):
        if crop_bgr.size == 0:
            continue
        crop_hash = phash_hex(crop_bgr)
        if any(_hamming(crop_hash, h_) <= DEDUP_HAMMING_THRESHOLD for h_ in seen_hashes):
            continue  # near-identical to a variant already queued
        path, digest = _write_temp_png(crop_bgr, f"variant_{variant_type}")
        ch, cw = crop_bgr.shape[:2]
        variants.append(
            SearchVariant(
                variant_id=f"v{i}-{variant_type}",
                variant_type=variant_type,
                image_path=path,
                sha256=digest,
                width=cw,
                height=ch,
            )
        )
        seen_hashes.append(crop_hash)
        if len(variants) >= budget:
            break

    return variants


def cleanup_variants(variants: list[SearchVariant], keep_path: str) -> None:
    """Remove temp files this module wrote, except the caller-owned original."""
    for v in variants:
        if v.image_path == keep_path:
            continue
        try:
            Path(v.image_path).unlink(missing_ok=True)
        except OSError:
            pass
