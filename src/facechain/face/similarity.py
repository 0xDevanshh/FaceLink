"""Face similarity on L2-normalised embeddings."""

from __future__ import annotations

import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity, clamped to [-1, 1].

    Both operands are re-normalised defensively so a caller passing a raw
    embedding cannot silently inflate the score.
    """
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    if a.size != b.size or a.size == 0:
        return 0.0
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.clip(np.dot(a / na, b / nb), -1.0, 1.0))


def best_cosine(reference: np.ndarray, candidates: list[np.ndarray]) -> float:
    """Best score against any face found in a candidate image.

    A social post is often a group photo, so the subject need not be the
    largest face there.
    """
    return max((cosine(reference, c) for c in candidates), default=0.0)
