"""Composite identity scoring and the false-positive margin guard.

Pins the mission's own worked examples exactly:
  - top=0.82, second=0.81 (margin 0.01)  -> NOT a reliable identification
  - top=0.86, second=0.63 + web evidence -> can become HIGH confidence
"""

from __future__ import annotations

from facechain.config import settings
from facechain.identity.matcher import PersonMatch
from facechain.identity.models import KnownPerson
from facechain.identity.scorer import enrich, score_preliminary
from facechain.models import IdentityLevel


def _match(person_id: str, name: str, best: float, refs: list[float] | None = None) -> PersonMatch:
    return PersonMatch(
        person=KnownPerson(person_id=person_id, canonical_name=name),
        best_similarity=best,
        reference_similarities=refs or [best],
    )


def test_no_matches_returns_none():
    assert score_preliminary([]) is None
    assert enrich([], 0.0, 0.0) is None


def test_thin_margin_is_not_a_reliable_identification():
    """The mission's own example: 0.82 vs 0.81 must never assert a name."""
    matches = [_match("p1", "Person One", 0.82), _match("p2", "Person Two", 0.81)]
    result = score_preliminary(matches)
    assert result is not None
    assert result.name is None
    assert result.person_id is None
    assert result.level in (IdentityLevel.LOW, IdentityLevel.UNKNOWN)
    # Still transparent about who was considered.
    assert {c.person_id for c in result.candidate_identities} == {"p1", "p2"}


def test_strong_margin_alone_without_web_evidence_is_capped_at_medium():
    """Face similarity + margin alone (no web/reverse-image evidence yet) —
    the mission's "DO NOT claim identity solely from an image embedding":
    a name IS reported (the face evidence is genuinely strong), but the
    level stays MEDIUM until independent corroboration exists."""
    matches = [_match("p1", "Person One", 0.90), _match("p2", "Person Two", 0.50)]
    result = score_preliminary(matches)
    assert result is not None
    assert result.name == "Person One"
    assert result.level == IdentityLevel.MEDIUM
    assert result.evidence_count == 0


def test_strong_margin_plus_web_evidence_reaches_high():
    """The mission's own example: 0.86 vs 0.63 + reverse-image/web evidence
    can become HIGH confidence."""
    matches = [_match("p1", "Sundar Pichai", 0.86), _match("p2", "Someone Else", 0.63)]
    result = enrich(matches, reverse_image_score=1.0, web_profile_score=1.0)
    assert result is not None
    assert result.name == "Sundar Pichai"
    assert result.level == IdentityLevel.HIGH
    assert result.evidence_count == 2
    assert result.confidence > 0.85


def test_below_the_floor_threshold_is_unknown_not_a_weak_guess():
    matches = [_match("p1", "Person One", 0.20)]
    result = score_preliminary(matches)
    assert result is not None
    assert result.level == IdentityLevel.UNKNOWN
    assert result.name is None


def test_single_candidate_index_has_no_margin_penalty():
    """A database with only one person has nothing to be confused with —
    margin should not artificially block a strong, unambiguous match."""
    matches = [_match("p1", "Only Person", 0.90)]
    result = score_preliminary(matches)
    assert result is not None
    assert result.margin == 1.0
    assert result.name == "Only Person"


def test_reference_consistency_reflects_all_references_not_just_the_best():
    weak_consistency = _match("p1", "Person One", 0.90, refs=[0.90, 0.20, 0.20])
    strong_consistency = _match("p1", "Person One", 0.90, refs=[0.90, 0.88, 0.85])
    other = _match("p2", "Person Two", 0.40)

    weak_result = score_preliminary([weak_consistency, other])
    strong_result = score_preliminary([strong_consistency, other])
    assert strong_result.confidence > weak_result.confidence


def test_weights_are_configurable_and_sum_to_one_by_default():
    total = (
        settings.identity_weight_face
        + settings.identity_weight_reference_consistency
        + settings.identity_weight_reverse_image
        + settings.identity_weight_web_profile
    )
    assert abs(total - 1.0) < 1e-9


def test_evidence_can_only_raise_never_lower_the_preliminary_result(monkeypatch):
    """enrich() re-scores the same candidates; adding evidence must not make
    a genuinely strong preliminary match look weaker."""
    matches = [_match("p1", "Person One", 0.90), _match("p2", "Person Two", 0.50)]
    preliminary = score_preliminary(matches)
    enriched = enrich(matches, reverse_image_score=1.0, web_profile_score=1.0)
    assert enriched.confidence >= preliminary.confidence
    assert enriched.level in (IdentityLevel.HIGH, IdentityLevel.MEDIUM)


def test_supporting_evidence_is_never_empty_for_a_named_result():
    matches = [_match("p1", "Person One", 0.90), _match("p2", "Person Two", 0.30)]
    result = score_preliminary(matches)
    assert result.name is not None
    assert len(result.supporting_evidence) >= 2  # face_similarity + margin_check at least
    kinds = {e.kind for e in result.supporting_evidence}
    assert "face_similarity" in kinds
    assert "margin_check" in kinds
