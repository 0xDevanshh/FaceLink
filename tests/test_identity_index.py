"""Known-person index loading — never crashes, never trusts a corrupt artifact."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from facechain.config import settings
from facechain.identity import index as index_mod
from facechain.identity.index import KnownPersonIndex, _load_from_disk, get_index, reset_cache


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    reset_cache()
    yield
    reset_cache()


def _write_index(tmp_path, people: list[dict], embeddings: np.ndarray, bad_checksum: bool = False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest = {"version": "1", "built_at": "now", "embedding_dim": embeddings.shape[1] if embeddings.size else 0,
               "people": people}
    manifest_bytes = json.dumps(manifest).encode("utf-8")
    (tmp_path / "manifest.json").write_bytes(manifest_bytes)
    np.save(tmp_path / "embeddings.npy", embeddings)
    checksum = hashlib.sha256(manifest_bytes).hexdigest()
    if bad_checksum:
        checksum = "0" * 64
    (tmp_path / "MANIFEST_SHA256").write_text(checksum, encoding="utf-8")


def test_missing_artifact_yields_empty_index(tmp_path):
    idx = _load_from_disk(tmp_path / "does_not_exist")
    assert idx.is_empty


def test_valid_artifact_loads_correctly(tmp_path):
    people = [{"person_id": "p1", "canonical_name": "Jane Doe", "references": [
        {"embedding_index": 0, "source_url": "https://example.org/p1.jpg"},
    ]}]
    embeddings = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    _write_index(tmp_path, people, embeddings)
    idx = _load_from_disk(tmp_path)
    assert not idx.is_empty
    assert idx.people[0].person_id == "p1"
    assert idx.embeddings.shape == (1, 3)


def test_checksum_mismatch_is_treated_as_unavailable_not_trusted(tmp_path):
    people = [{"person_id": "p1", "canonical_name": "Jane Doe", "references": []}]
    embeddings = np.zeros((0, 3), dtype=np.float32)
    _write_index(tmp_path, people, embeddings, bad_checksum=True)
    idx = _load_from_disk(tmp_path)
    assert idx.is_empty  # corrupt/stale artifact -> safe empty, not a crash


def test_shape_mismatch_between_manifest_and_embeddings_is_rejected(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest = {"version": "1", "built_at": "now", "embedding_dim": 512, "people": []}
    manifest_bytes = json.dumps(manifest).encode("utf-8")
    (tmp_path / "manifest.json").write_bytes(manifest_bytes)
    np.save(tmp_path / "embeddings.npy", np.zeros((0, 3), dtype=np.float32))  # wrong dim
    idx = _load_from_disk(tmp_path)
    assert idx.is_empty


def test_malformed_json_does_not_crash(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "manifest.json").write_bytes(b"{not valid json")
    np.save(tmp_path / "embeddings.npy", np.zeros((0, 3), dtype=np.float32))
    idx = _load_from_disk(tmp_path)
    assert idx.is_empty


def test_get_index_is_a_process_wide_cache(tmp_path, monkeypatch):
    people = [{"person_id": "p1", "canonical_name": "Jane Doe", "references": [
        {"embedding_index": 0, "source_url": "x"},
    ]}]
    _write_index(tmp_path, people, np.array([[1.0, 0.0]], dtype=np.float32))
    monkeypatch.setattr(settings, "identity_index_dir", tmp_path)
    monkeypatch.setattr(settings, "identity_enabled", True)

    first = get_index()
    # Mutate the file on disk — a cached process must NOT re-read it (mission
    # section 13: "do not regenerate on every request").
    (tmp_path / "manifest.json").write_bytes(b"{not valid json")
    second = get_index()
    assert first is second
    assert not second.is_empty


def test_identity_disabled_always_returns_empty_regardless_of_a_valid_index(tmp_path, monkeypatch):
    people = [{"person_id": "p1", "canonical_name": "Jane Doe", "references": [
        {"embedding_index": 0, "source_url": "x"},
    ]}]
    _write_index(tmp_path, people, np.array([[1.0, 0.0]], dtype=np.float32))
    monkeypatch.setattr(settings, "identity_index_dir", tmp_path)
    monkeypatch.setattr(settings, "identity_enabled", False)
    assert get_index().is_empty
