"""Fuse the independent signals into one score and walk the verification ladder.

    SEARCH_FOUND -> SOCIAL_MATCH -> IMAGE_MATCH -> FACE_MATCH -> VERIFIED

Each rung is a separate, checkable claim. `VERIFIED` requires all of them, so
"verified" can never mean "a search engine returned a link".

What this score does NOT claim: it does not establish a person's real-world
identity. It states that the supplied image and its primary face match the
retrieved public image under the thresholds recorded in the evidence bundle.
"""

from __future__ import annotations

from ..config import settings
from ..models import LADDER, Stage, VerifiedCandidate


def score_candidate(vc: VerifiedCandidate) -> VerifiedCandidate:
    """Populate `stages`, `final_score` and `verified` on a measured candidate."""
    stages: list[Stage] = [Stage.SEARCH_FOUND]

    if vc.is_social:
        stages.append(Stage.SOCIAL_MATCH)
    if vc.image_similarity >= settings.image_match_threshold:
        stages.append(Stage.IMAGE_MATCH)
    if vc.face_detected and vc.face_similarity >= settings.face_match_threshold:
        stages.append(Stage.FACE_MATCH)

    vc.final_score = (
        settings.weight_face * max(0.0, vc.face_similarity)
        + settings.weight_image * vc.image_similarity
        + settings.weight_meta * vc.metadata_consistency
    )

    required = {Stage.SEARCH_FOUND, Stage.SOCIAL_MATCH, Stage.IMAGE_MATCH, Stage.FACE_MATCH}
    if required.issubset(set(stages)) and vc.final_score >= settings.verify_min_score:
        stages.append(Stage.VERIFIED)
        vc.verified = True

    vc.stages = [s for s in LADDER if s in set(stages)]
    return vc


def rank(candidates: list[VerifiedCandidate]) -> list[VerifiedCandidate]:
    """Best first: verified, then more rungs climbed, then higher score."""
    return sorted(
        candidates,
        key=lambda c: (c.verified, len(c.stages), c.final_score, c.face_similarity),
        reverse=True,
    )


def highest_stage_reached(candidates: list[VerifiedCandidate]) -> list[Stage]:
    """Union of rungs any candidate reached — used for the run's diagnostics."""
    reached = {s for c in candidates for s in c.stages}
    return [s for s in LADDER if s in reached]


def explain_failure(candidates: list[VerifiedCandidate]) -> str:
    """Say precisely which rung the run died on, for the CLI and the case file."""
    if not candidates:
        return "no candidates survived reverse image search"

    best = rank(candidates)[0]
    if not any(c.is_social for c in candidates):
        return "reverse search found matches, but none on a supported social platform"
    if best.image_similarity < settings.image_match_threshold:
        return (
            f"best image similarity {best.image_similarity:.3f} < "
            f"threshold {settings.image_match_threshold}"
        )
    if not best.face_detected:
        return "no face detectable in the retrieved candidate image"
    if best.face_similarity < settings.face_match_threshold:
        return (
            f"best face similarity {best.face_similarity:.3f} < "
            f"threshold {settings.face_match_threshold}"
        )
    return (
        f"final score {best.final_score:.3f} < threshold {settings.verify_min_score}"
    )
