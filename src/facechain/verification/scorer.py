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
    """Populate `stages`, `match_type`, `final_score` and `verified`.

    What VERIFIED requires, and why:

      SEARCH_FOUND  the URL came from a real reverse-image engine
      SOCIAL_MATCH  it is a post on a supported social platform
      FACE_MATCH    the face in the retrieved image matches the input face
      final_score   >= `verify_min_score`, where image similarity carries 40%

    `IMAGE_MATCH` is recorded but deliberately *not* mandatory. Social reposts
    routinely crop, pad and overlay text, which drops a perceptual hash well
    below the exact-image bar while the face stays unmistakable — and this is a
    face-identification task, so the face is the primary signal and the image
    hash is corroboration. Which of the two held is preserved in `match_type`:

      "exact-image"  the retrieved image is perceptually the same picture
      "face-only"    same face, visibly different/edited picture

    A wrong person cannot sneak through: with face similarity near zero the
    composite score cannot reach the threshold on image similarity alone.
    """
    stages: list[Stage] = [Stage.SEARCH_FOUND]

    if vc.is_social:
        stages.append(Stage.SOCIAL_MATCH)
    exact_image = vc.image_similarity >= settings.image_match_threshold
    if exact_image:
        stages.append(Stage.IMAGE_MATCH)
    face_ok = vc.face_detected and vc.face_similarity >= settings.face_match_threshold
    if face_ok:
        stages.append(Stage.FACE_MATCH)

    vc.match_type = "exact-image" if exact_image else ("face-only" if face_ok else "none")

    vc.final_score = (
        settings.weight_face * max(0.0, vc.face_similarity)
        + settings.weight_image * vc.image_similarity
        + settings.weight_meta * vc.metadata_consistency
    )

    required = {Stage.SEARCH_FOUND, Stage.SOCIAL_MATCH, Stage.FACE_MATCH}
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
    """Say precisely which rung blocked verification.

    The reason must describe the *social* candidates, since only those can ever
    be verified — reporting the best overall candidate's numbers would claim a
    threshold failure that never happened (e.g. a non-social page scoring 0.89
    while the real blocker was that nothing social was found).
    """
    if not candidates:
        return "no candidates survived reverse image search"

    social = [c for c in candidates if c.is_social]
    if not social:
        best = rank(candidates)[0]
        return (
            f"{len(candidates)} candidate(s) checked but none on a supported social "
            f"platform (best non-social: {best.domain} at score {best.final_score:.3f})"
        )

    best = rank(social)[0]
    if not best.candidate_image_sha256:
        return (
            f"social candidates found ({', '.join(sorted({c.domain for c in social}))}) but no "
            "comparable image could be retrieved — login walls or bot blocks on every one"
        )
    if not best.face_detected:
        return f"no face detectable in the image retrieved from {best.domain}"
    if best.face_similarity < settings.face_match_threshold:
        return (
            f"best social face similarity {best.face_similarity:.3f} < threshold "
            f"{settings.face_match_threshold} ({best.domain})"
        )
    return (
        f"best social candidate {best.domain} scored {best.final_score:.3f} < threshold "
        f"{settings.verify_min_score} (face {best.face_similarity:.3f}, "
        f"image {best.image_similarity:.3f})"
    )
