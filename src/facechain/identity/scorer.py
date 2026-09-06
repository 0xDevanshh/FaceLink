"""Composite identity scoring and the false-positive guard.

The one non-negotiable rule from the mission (section 6): a margin-over-
second-best-candidate check gates *before* the weighted composite score even
runs. A top candidate that barely beats the runner-up is capped at
LOW/UNKNOWN regardless of how high its raw face similarity is — "highest
similarity = definitely that person" is exactly what this file exists to
prevent.

Two entry points, both pure functions (no I/O, easy to unit test):

  score_preliminary(matches)  — face similarity + reference consistency only,
                                 called before reverse-image search has run.
  enrich(matches, ...)        — the same candidates, re-scored once this
                                 scan's web/reverse-image evidence is known.
                                 Monotonic: evidence can only raise the
                                 result, never lower it below what face-only
                                 scoring already established.

Neither ever invents a name: `IdentityResult.name`/`person_id` are set only
when `level` is HIGH or MEDIUM. LOW and UNKNOWN report the same candidate
shortlist for transparency (`candidate_identities`) but leave `name` unset —
"never force a name" (mission section 5).
"""

from __future__ import annotations

from ..config import settings
from ..models import IdentityCandidate, IdentityEvidence, IdentityLevel, IdentityResult
from .matcher import PersonMatch


def _reference_consistency(match: PersonMatch) -> float:
    """Mean similarity across ALL of this person's reference embeddings —
    not just the single best match (mission section 3: multiple references,
    not one image)."""
    sims = match.reference_similarities
    return sum(sims) / len(sims) if sims else 0.0


def _margin(matches: list[PersonMatch]) -> float:
    """Top-vs-second-best gap. A single-candidate index has nothing to be
    confused with, so it is treated as fully separated (margin = 1.0) rather
    than penalised for the database only containing one person."""
    if len(matches) < 2:
        return 1.0
    return matches[0].best_similarity - matches[1].best_similarity


def _face_tier(face_similarity: float, margin: float) -> str:
    """"strong" / "moderate" / "insufficient" — the hard gate. Margin is
    checked here, before any weighted sum, so it can never be diluted by a
    high reverse-image/web-profile score."""
    if margin < settings.identity_margin_min:
        return "insufficient"
    if face_similarity >= settings.identity_face_threshold_high:
        return "strong"
    if face_similarity >= settings.identity_face_threshold_medium:
        return "moderate"
    return "insufficient"


def _candidate_list(matches: list[PersonMatch]) -> list[IdentityCandidate]:
    return [
        IdentityCandidate(
            person_id=m.person.person_id,
            name=m.person.canonical_name,
            face_similarity=m.best_similarity,
            category=m.person.category,
            occupation=m.person.occupation,
        )
        for m in matches
    ]


def _score(
    matches: list[PersonMatch],
    reverse_image_score: float,
    web_profile_score: float,
    extra_evidence: list[IdentityEvidence],
) -> IdentityResult:
    top = matches[0]
    face_similarity = top.best_similarity
    margin = _margin(matches)
    consistency = _reference_consistency(top)
    tier = _face_tier(face_similarity, margin)

    evidence: list[IdentityEvidence] = [
        IdentityEvidence(
            kind="face_similarity",
            description=f"face similarity {face_similarity:.3f} against best reference image",
            weight=settings.identity_weight_face,
            value=face_similarity,
        ),
        IdentityEvidence(
            kind="margin_check",
            description=(
                f"top candidate leads the second-best by {margin:.3f} "
                f"(minimum required: {settings.identity_margin_min})"
            ),
            weight=0.0,  # gate, not a weighted component
            value=margin,
        ),
        IdentityEvidence(
            kind="reference_consistency",
            description=(
                f"mean similarity {consistency:.3f} across all "
                f"{len(top.reference_similarities)} reference image(s) for this person"
            ),
            weight=settings.identity_weight_reference_consistency,
            value=consistency,
        ),
    ]
    if reverse_image_score > 0.0:
        evidence.append(IdentityEvidence(
            kind="reverse_image",
            description="this scan's own reverse-image search independently found "
                        "this person's name/profile among the discovered candidates",
            weight=settings.identity_weight_reverse_image,
            value=reverse_image_score,
        ))
    if web_profile_score > 0.0:
        evidence.append(IdentityEvidence(
            kind="web_profile",
            description="this scan's evidence matches one of this person's known "
                        "official public profile domains",
            weight=settings.identity_weight_web_profile,
            value=web_profile_score,
        ))
    evidence.extend(extra_evidence)

    confidence = (
        settings.identity_weight_face * face_similarity
        + settings.identity_weight_reference_consistency * consistency
        + settings.identity_weight_reverse_image * reverse_image_score
        + settings.identity_weight_web_profile * web_profile_score
    )
    confidence = max(0.0, min(1.0, confidence))

    # "Evidence" here means independent corroboration beyond the face match
    # itself (mission section 0: never claim identity from an embedding
    # alone) — reverse-image and web-profile signals, not the face/margin/
    # consistency checks every candidate already has.
    corroboration_count = (1 if reverse_image_score > 0.0 else 0) + (1 if web_profile_score > 0.0 else 0)

    if tier == "strong" and corroboration_count > 0:
        level, name, person_id = IdentityLevel.HIGH, top.person.canonical_name, top.person.person_id
    elif tier == "strong" or (tier == "moderate" and corroboration_count > 0):
        level, name, person_id = IdentityLevel.MEDIUM, top.person.canonical_name, top.person.person_id
    elif tier == "moderate":
        # Real similarity, no corroboration yet — reported as a candidate,
        # not asserted as an identity ("never force a name").
        level, name, person_id = IdentityLevel.LOW, None, None
    else:
        level, name, person_id = IdentityLevel.UNKNOWN, None, None

    return IdentityResult(
        person_id=person_id,
        name=name,
        aliases=list(top.person.aliases) if name else [],
        occupation=top.person.occupation if name else "",
        category=top.person.category if name else "",
        confidence=confidence,
        level=level,
        face_similarity=face_similarity,
        margin=margin,
        evidence_count=corroboration_count,
        supporting_evidence=evidence,
        candidate_identities=_candidate_list(matches),
    ).rounded()


def score_preliminary(matches: list[PersonMatch]) -> IdentityResult | None:
    """Face similarity + reference consistency only — no web evidence yet."""
    if not matches:
        return None
    return _score(matches, reverse_image_score=0.0, web_profile_score=0.0, extra_evidence=[])


def enrich(
    matches: list[PersonMatch],
    reverse_image_score: float,
    web_profile_score: float,
    extra_evidence: list[IdentityEvidence] | None = None,
) -> IdentityResult | None:
    """Re-score the same candidates with this scan's web evidence folded in."""
    if not matches:
        return None
    return _score(matches, reverse_image_score, web_profile_score, extra_evidence or [])
