"""End-to-end identity resolution (resolve_preliminary + enrich_with_evidence)
against a real on-disk index — the exact two call sites `runner.py` uses.
"""

from __future__ import annotations

import hashlib
import json

import httpx
import numpy as np
import pytest

from facechain.config import settings
from facechain.identity import api_ninjas, resolve
from facechain.identity.index import reset_cache
from facechain.models import Case, IdentityLevel, SearchCandidate, SearchReport


@pytest.fixture(autouse=True)
def _reset():
    api_ninjas.reset_cache()
    reset_cache()
    yield
    reset_cache()
    api_ninjas.reset_cache()


def _write_index(tmp_path, people: list[dict], embeddings: np.ndarray):
    manifest = {"version": "1", "built_at": "now", "embedding_dim": embeddings.shape[1], "people": people}
    manifest_bytes = json.dumps(manifest).encode("utf-8")
    (tmp_path / "manifest.json").write_bytes(manifest_bytes)
    np.save(tmp_path / "embeddings.npy", embeddings)
    (tmp_path / "MANIFEST_SHA256").write_text(hashlib.sha256(manifest_bytes).hexdigest(), encoding="utf-8")


def _setup_index(tmp_path, monkeypatch):
    people = [
        {
            "person_id": "jane_doe", "canonical_name": "Jane Doe",
            "occupation": "Engineer", "category": "technology",
            "official_website": "https://janedoe.com",
            "official_social_accounts": {"GitHub": "https://github.com/janedoe"},
            "references": [
                {"embedding_index": 0, "source_url": "https://commons.wikimedia.org/a"},
                {"embedding_index": 1, "source_url": "https://commons.wikimedia.org/b"},
            ],
        },
        {
            "person_id": "someone_else", "canonical_name": "Someone Else",
            "references": [{"embedding_index": 2, "source_url": "https://commons.wikimedia.org/c"}],
        },
    ]
    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.98, 0.02, 0.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float32)
    _write_index(tmp_path, people, embeddings)
    monkeypatch.setattr(settings, "identity_index_dir", tmp_path)
    monkeypatch.setattr(settings, "identity_enabled", True)


def test_no_embedding_returns_none():
    assert resolve.resolve_preliminary(None) is None


def test_positive_strong_match_resolves_preliminary(tmp_path, monkeypatch):
    _setup_index(tmp_path, monkeypatch)
    result = resolve.resolve_preliminary(np.array([1.0, 0.0, 0.0]))
    assert result is not None
    assert result.name == "Jane Doe"
    assert result.level in (IdentityLevel.MEDIUM, IdentityLevel.HIGH)


def test_unknown_person_not_in_index_resolves_to_no_match(tmp_path, monkeypatch):
    _setup_index(tmp_path, monkeypatch)
    result = resolve.resolve_preliminary(np.array([0.0, 0.0, 1.0]))  # orthogonal to everyone
    assert result is not None
    assert result.name is None
    assert result.level == IdentityLevel.UNKNOWN


def test_enrich_with_evidence_upgrades_and_populates_official_profiles(tmp_path, monkeypatch):
    _setup_index(tmp_path, monkeypatch)
    embedding = np.array([1.0, 0.0, 0.0])
    preliminary = resolve.resolve_preliminary(embedding)
    case = Case(
        case_id="case_test", created_at="now", observed_at=0,
        reverse_search=SearchReport(candidates=[
            SearchCandidate(engine="yandex", url="https://github.com/janedoe",
                            domain="github.com", title="Jane Doe on GitHub"),
        ]),
    )
    final = resolve.enrich_with_evidence(preliminary, embedding, case)
    assert final.level == IdentityLevel.HIGH
    assert case.official_profiles  # populated as a side effect
    assert any(p.platform == "GitHub" for p in case.official_profiles)


def test_enrich_with_evidence_leaves_profiles_empty_when_no_name(tmp_path, monkeypatch):
    _setup_index(tmp_path, monkeypatch)
    embedding = np.array([0.0, 0.0, 1.0])  # matches no one
    preliminary = resolve.resolve_preliminary(embedding)
    case = Case(case_id="case_test", created_at="now", observed_at=0)
    final = resolve.enrich_with_evidence(preliminary, embedding, case)
    assert final.name is None
    assert case.official_profiles == []


def test_disabled_flag_is_a_full_no_op_even_with_a_valid_index_present(tmp_path, monkeypatch):
    _setup_index(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "identity_enabled", False)
    assert resolve.resolve_preliminary(np.array([1.0, 0.0, 0.0])) is None


def test_a_broken_index_never_raises_out_of_resolve_preliminary(monkeypatch):
    def _boom():
        raise RuntimeError("disk exploded")
    monkeypatch.setattr("facechain.identity.resolve.get_index", _boom)
    assert resolve.resolve_preliminary(np.array([1.0, 0.0, 0.0])) is None


def test_a_broken_index_never_raises_out_of_enrich_and_returns_preliminary(monkeypatch):
    def _boom():
        raise RuntimeError("disk exploded")
    monkeypatch.setattr("facechain.identity.resolve.get_index", _boom)
    case = Case(case_id="case_test", created_at="now", observed_at=0)
    from facechain.models import IdentityResult
    preliminary = IdentityResult(name="Whatever", level=IdentityLevel.MEDIUM)
    result = resolve.enrich_with_evidence(preliminary, np.array([1.0, 0.0, 0.0]), case)
    assert result is preliminary


# ---- API Ninjas celebrity enrichment wired into resolve_with_evidence -----

def test_celebrity_metadata_is_populated_when_a_name_is_resolved(tmp_path, monkeypatch):
    _setup_index(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "api_ninjas_api_key", "test-key")

    def _get(url, params=None, headers=None, timeout=None):
        return httpx.Response(200, json=[{"name": "jane doe", "age": 40, "nationality": "us"}],
                              request=httpx.Request("GET", url))
    monkeypatch.setattr(httpx, "get", _get)

    embedding = np.array([1.0, 0.0, 0.0])
    preliminary = resolve.resolve_preliminary(embedding)
    case = Case(case_id="case_test", created_at="now", observed_at=0)
    resolve.enrich_with_evidence(preliminary, embedding, case)

    assert case.celebrity is not None
    assert case.celebrity.available is True
    assert case.celebrity.name == "Jane Doe"  # the resolved identity name, not API Ninjas' casing
    assert case.celebrity.age == 40


def test_no_api_ninjas_key_leaves_celebrity_unavailable_but_identity_intact(tmp_path, monkeypatch):
    _setup_index(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "api_ninjas_api_key", "")

    embedding = np.array([1.0, 0.0, 0.0])
    preliminary = resolve.resolve_preliminary(embedding)
    case = Case(case_id="case_test", created_at="now", observed_at=0)
    final = resolve.enrich_with_evidence(preliminary, embedding, case)

    assert final.name == "Jane Doe"  # identity resolution is completely unaffected
    assert case.celebrity is not None
    assert case.celebrity.available is False


def test_api_ninjas_network_failure_does_not_affect_identity_or_profiles(tmp_path, monkeypatch):
    _setup_index(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "api_ninjas_api_key", "test-key")
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: (_ for _ in ()).throw(httpx.ConnectError("boom")))

    embedding = np.array([1.0, 0.0, 0.0])
    preliminary = resolve.resolve_preliminary(embedding)
    case = Case(
        case_id="case_test", created_at="now", observed_at=0,
        reverse_search=SearchReport(candidates=[
            SearchCandidate(engine="yandex", url="https://github.com/janedoe",
                            domain="github.com", title="Jane Doe on GitHub"),
        ]),
    )
    final = resolve.enrich_with_evidence(preliminary, embedding, case)

    # Identity + official-profile resolution (computed before the celebrity
    # call) are completely unaffected by API Ninjas failing.
    assert final.level == IdentityLevel.HIGH
    assert case.official_profiles
    assert case.celebrity.available is False


def test_celebrity_metadata_can_never_override_the_resolved_identity_name(tmp_path, monkeypatch):
    """Even if API Ninjas returned a different-looking name, the identity
    the face-evidence layer resolved must be what the caller already has —
    celebrity enrichment never writes back into `IdentityResult`."""
    _setup_index(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "api_ninjas_api_key", "test-key")

    def _get(url, params=None, headers=None, timeout=None):
        return httpx.Response(200, json=[{"name": "Someone Completely Different"}],
                              request=httpx.Request("GET", url))
    monkeypatch.setattr(httpx, "get", _get)

    embedding = np.array([1.0, 0.0, 0.0])
    preliminary = resolve.resolve_preliminary(embedding)
    case = Case(case_id="case_test", created_at="now", observed_at=0)
    final = resolve.enrich_with_evidence(preliminary, embedding, case)

    assert final.name == "Jane Doe"  # IdentityResult itself: completely untouched
    # API Ninjas' name didn't even plausibly match -> no celebrity data attached.
    assert case.celebrity.available is False
