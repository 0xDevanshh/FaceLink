"""Brute-force top-K identity matching — offline, synthetic fixtures only.

Never touches the real (offline-built, gitignored) identity_index/ artifact:
every test builds its own tiny in-memory KnownPersonIndex.
"""

from __future__ import annotations

import numpy as np

from facechain.identity.index import KnownPersonIndex
from facechain.identity.matcher import match_all, top_k
from facechain.identity.models import KnownPerson, KnownPersonIndexManifest, KnownPersonReference


def _unit(vec: list[float]) -> np.ndarray:
    a = np.array(vec, dtype=np.float32)
    return a / np.linalg.norm(a)


def make_index(people_vectors: dict[str, list[list[float]]]) -> KnownPersonIndex:
    """Build a KnownPersonIndex where each person has one or more reference
    vectors, given as raw (non-unit) lists — normalised here."""
    rows: list[np.ndarray] = []
    people: list[KnownPerson] = []
    for person_id, vectors in people_vectors.items():
        refs = []
        for v in vectors:
            refs.append(KnownPersonReference(embedding_index=len(rows), source_url=f"https://example.org/{person_id}"))
            rows.append(_unit(v))
        people.append(KnownPerson(
            person_id=person_id, canonical_name=person_id.replace("_", " ").title(),
            references=refs,
        ))
    dim = len(next(iter(people_vectors.values()))[0]) if people_vectors else 0
    manifest = KnownPersonIndexManifest(version="test", built_at="", embedding_dim=dim, people=people)
    embeddings = np.stack(rows) if rows else np.zeros((0, dim), dtype=np.float32)
    return KnownPersonIndex(manifest, embeddings.astype(np.float32))


def test_empty_index_returns_no_matches():
    idx = KnownPersonIndex.empty()
    assert match_all(idx, np.array([1.0, 0.0, 0.0])) == []
    assert top_k(idx, np.array([1.0, 0.0, 0.0])) == []


def test_top_match_is_the_closest_vector():
    idx = make_index({
        "alice": [[1.0, 0.0, 0.0]],
        "bob": [[0.0, 1.0, 0.0]],
    })
    query = [0.95, 0.05, 0.0]  # close to alice
    results = top_k(idx, query)
    assert results[0].person.person_id == "alice"
    assert results[0].best_similarity > results[1].best_similarity


def test_multiple_references_per_person_all_considered():
    idx = make_index({
        "alice": [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 0.0, 1.0]],  # one bad ref
    })
    query = [1.0, 0.0, 0.0]
    results = match_all(idx, query)
    assert len(results) == 1
    assert len(results[0].reference_similarities) == 3
    # best_similarity uses the single best reference, not an average.
    assert results[0].best_similarity > max(results[0].reference_similarities[1:])


def test_top_k_limits_and_orders_results():
    idx = make_index({f"person_{i}": [[float(i), 1.0, 0.0]] for i in range(10)})
    results = top_k(idx, [0.0, 1.0, 0.0], k=3)
    assert len(results) == 3
    sims = [r.best_similarity for r in results]
    assert sims == sorted(sims, reverse=True)


def test_zero_query_embedding_yields_no_matches():
    idx = make_index({"alice": [[1.0, 0.0, 0.0]]})
    assert match_all(idx, np.zeros(3)) == []


def test_query_dimension_mismatch_yields_no_matches_not_a_crash():
    idx = make_index({"alice": [[1.0, 0.0, 0.0]]})
    assert match_all(idx, np.array([1.0, 0.0])) == []
