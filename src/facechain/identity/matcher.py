"""Brute-force top-K cosine search over the loaded known-person index.

Why brute-force numpy instead of FAISS: this repo has no FAISS dependency
today, and an MVP index (tens to a few hundred people, a few thousand
reference embeddings at most) makes a single `(N, D) @ (D,)` matmul
sub-millisecond — FAISS's approximate-nearest-neighbour speed only starts to
matter past ~100k vectors. `top_k()` is the one function to swap for a
FAISS-backed implementation if the index ever grows that far; nothing
outside this module would need to change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import settings
from .index import KnownPersonIndex
from .models import KnownPerson


@dataclass
class PersonMatch:
    """One known person's aggregate similarity to a query face."""

    person: KnownPerson
    best_similarity: float
    # One cosine score per `person.references`, same order — lets the scorer
    # measure agreement across *multiple* reference images (mission section
    # 3), not just the single best-matching one.
    reference_similarities: list[float]


def _normalise_rows(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def match_all(index: KnownPersonIndex, query_embedding: np.ndarray) -> list[PersonMatch]:
    """Compare `query_embedding` against every reference in `index`.

    One `PersonMatch` per person with at least one reference embedding.
    Unordered — sort by `best_similarity` yourself, or use `top_k` below.
    Returns `[]` for an empty index or an all-zero query embedding, never
    raises.
    """
    if index.is_empty:
        return []

    query = np.asarray(query_embedding, dtype=np.float32).ravel()
    qnorm = float(np.linalg.norm(query))
    if qnorm == 0.0 or query.size != index.embeddings.shape[1]:
        return []
    query = query / qnorm

    refs = _normalise_rows(index.embeddings)
    scores = np.clip(refs @ query, -1.0, 1.0)  # (N,) cosine per reference embedding

    results: list[PersonMatch] = []
    for person in index.people:
        if not person.references:
            continue
        sims = [float(scores[ref.embedding_index]) for ref in person.references]
        results.append(PersonMatch(person=person, best_similarity=max(sims), reference_similarities=sims))
    return results


def top_k(
    index: KnownPersonIndex, query_embedding: np.ndarray, k: int | None = None
) -> list[PersonMatch]:
    """`match_all`, sorted best-first, limited to the top `k` people."""
    matches = match_all(index, query_embedding)
    matches.sort(key=lambda m: m.best_similarity, reverse=True)
    return matches[: (k or settings.identity_top_k)]
