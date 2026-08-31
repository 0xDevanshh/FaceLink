"""Perceptual image similarity (robust to re-encoding, resizing, mild crops)."""

from __future__ import annotations

import io

import imagehash
import numpy as np
from PIL import Image

HASH_BITS = 64  # 8x8 hashes


def _pil(data: bytes | np.ndarray) -> Image.Image:
    if isinstance(data, bytes):
        return Image.open(io.BytesIO(data)).convert("RGB")
    # BGR ndarray (OpenCV) -> RGB PIL
    return Image.fromarray(data[:, :, ::-1]).convert("RGB")


def perceptual_hashes(data: bytes | np.ndarray) -> dict[str, str]:
    img = _pil(data)
    return {
        "phash": str(imagehash.phash(img)),
        "dhash": str(imagehash.dhash(img)),
        "ahash": str(imagehash.average_hash(img)),
    }


def phash_hex(data: bytes | np.ndarray) -> str:
    return str(imagehash.phash(_pil(data)))


def _sim(a: str, b: str) -> float:
    ha, hb = imagehash.hex_to_hash(a), imagehash.hex_to_hash(b)
    return 1.0 - (ha - hb) / HASH_BITS


def compare(a: dict[str, str], b: dict[str, str]) -> float:
    """Weighted perceptual similarity in [0, 1].

    pHash carries the most weight (DCT-based, survives compression and scaling);
    dHash adds gradient structure. aHash is kept in the evidence bundle for
    transparency but is too crude to weight heavily.
    """
    phash_sim = _sim(a["phash"], b["phash"])
    dhash_sim = _sim(a["dhash"], b["dhash"])
    return float(np.clip(0.7 * phash_sim + 0.3 * dhash_sim, 0.0, 1.0))
