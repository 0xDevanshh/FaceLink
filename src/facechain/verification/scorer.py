"""Fuse the independent signals into one score and walk the verification ladder.

    SEARCH_FOUND -> SOCIAL_MATCH -> IMAGE_MATCH -> FACE_MATCH -> VERIFIED

Each rung is a separate, checkable claim. `VERIFIED` requires all of them, so
"verified" can never mean "a search engine returned a link".

What this score does NOT claim: it does not establish a person's real-world
identity. It states that the supplied image and its primary face match the
retrieved public image under the thresholds recorded in the evidence bundle.
"""

from __future__ import annotations

from ..config import confidence_band, settings
from ..models import LADDER, CandidateType, Stage, VerifiedCandidate


def score_candidate(vc: VerifiedCandidate) -> VerifiedCandidate:
    """Populate `stages`, `match_type`, `final_score`, `verified`, and `confidence_band`.

    What VERIFIED requires, and why:

      SEARCH_FOUND  the URL came from a real reverse-image engine
      FACE_MATCH    the face in the retrieved image matches the input face
      final_score   >= `verify_min_score`, where image similarity carries 40%

    `SOCIAL_MATCH` is recorded and drives discovery priority, but it is *not*
    required. It used to be, and that was wrong for a face-matching tool: it
    made a strongly face-verified match on a personal site, a university page
    or a conference programme permanently unverifiable, purely because its
    domain was absent from a list. Search priority is not search exclusivity —
    the platform a match was found on is metadata about provenance, not
    evidence about the face. What the face evidence says is decided by the face
    measurement alone.

    `IMAGE_MATCH` is likewise recorded and not mandatory. Reposts routinely
    crop, pad and overlay text, which drops a perceptual hash well below the
    exact-image bar while the face stays unmistakable. Which of the two held is
    preserved in `match_type` and `candidate_type`:

      "exact-image" / EXACT_IMAGE  the retrieved image is the same picture
      "face-only"   / SAME_FACE    same face, visibly different picture

    A wrong person still cannot sneak through, and the arithmetic guarantees it
    rather than a policy doing so: with face similarity at zero the remaining
    weights total 0.5 (image 0.4 + metadata 0.1), below the 0.70 minimum. No
    combination of a perfect image hash and perfect metadata can verify a face
    that does not match.
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
    vc.confidence_band = confidence_band(vc.face_similarity)

    required = {Stage.SEARCH_FOUND, Stage.FACE_MATCH}
    combined_score_ok = required.issubset(set(stages)) and vc.final_score >= settings.verify_min_score
    # Opt-in, disabled by default (see config.py) — a second, independent path
    # to VERIFIED based on face similarity alone, for a deployment that has
    # explicitly decided the combined score is too strict. Still requires
    # FACE_MATCH itself (face_ok), so this can never verify a candidate whose
    # face didn't match at all.
    face_only_ok = (
        settings.face_only_verify_enabled
        and face_ok
        and vc.face_similarity >= settings.face_only_verify_threshold
    )
    if combined_score_ok or face_only_ok:
        stages.append(Stage.VERIFIED)
        vc.verified = True

    vc.stages = [s for s in LADDER if s in set(stages)]
    vc.candidate_type = _measured_candidate_type(vc, exact_image, face_ok)

    if not vc.verified:
        vc.rejection_reason = _rejection_reason(vc, stages)

    return vc


def _measured_candidate_type(
    vc: VerifiedCandidate, exact_image: bool, face_ok: bool
) -> CandidateType:
    """Upgrade the URL-derived type with what the measurement actually showed.

    EXACT_IMAGE and SAME_FACE are claims about pixels and embeddings, so they
    are only ever assigned here, after both comparisons have run. A candidate we
    could not measure keeps the provisional type it got from its URL, which
    stays honest about the fact that nothing was measured.
    """
    if exact_image:
        return CandidateType.EXACT_IMAGE
    if face_ok:
        return CandidateType.SAME_FACE
    return vc.candidate_type


def highest_rung(vc: VerifiedCandidate) -> str:
    """The furthest ladder rung this candidate reached, as a plain string."""
    return vc.stages[-1].value if vc.stages else ""


def _rejection_reason(vc: VerifiedCandidate, stages: list[Stage]) -> str:
    """Precise reason why this specific candidate was not verified."""
    if not vc.candidate_image_sha256:
        return "no comparable image could be retrieved (login wall or bot block)"
    if not vc.face_detected:
        return "no face detectable in the retrieved image"
    if vc.face_similarity < settings.face_match_threshold:
        return (
            f"face similarity {vc.face_similarity:.3f} below threshold "
            f"{settings.face_match_threshold}"
        )
    reason = (
        f"composite score {vc.final_score:.3f} below minimum "
        f"{settings.verify_min_score} (face {vc.face_similarity:.3f}, "
        f"image {vc.image_similarity:.3f})"
    )
    if settings.face_only_verify_enabled:
        reason += (
            f"; face-only path also did not qualify (face {vc.face_similarity:.3f} "
            f"< {settings.face_only_verify_threshold})"
        )
    return reason


# Rungs that are claims about the *evidence* rather than about provenance.
# SOCIAL_MATCH is deliberately absent: which platform a page lives on says
# nothing about whether the face matches.
EVIDENTIAL_STAGES = frozenset({Stage.IMAGE_MATCH, Stage.FACE_MATCH, Stage.VERIFIED})


def evidential_strength(vc: VerifiedCandidate) -> int:
    """How many *measured* rungs this candidate climbed."""
    return sum(1 for s in vc.stages if s in EVIDENTIAL_STAGES)


def rank(candidates: list[VerifiedCandidate]) -> list[VerifiedCandidate]:
    """Best first.

    Verification strength dominates; platform priority only ever breaks ties
    between candidates of equal evidential strength and equal score.

    Counting *evidential* rungs rather than all rungs matters, and a real run
    showed why: a YouTube channel scoring 0.873 was being ranked above a page
    scoring 0.940 with a higher face similarity, purely because the YouTube hit
    also carried SOCIAL_MATCH and so appeared to have climbed one rung further.
    That is a platform name outranking a measurement, which is precisely what
    this ordering exists to prevent.

    A candidate at or above `high_face_similarity_priority` is additionally
    promoted ahead of every candidate below it, ranked among themselves by
    face similarity first — a strong-face, weak-image-similarity hit (a
    different photo of the same person) must not be outranked by a
    weak-face, strong-image-similarity one (a near-identical copy of a
    different person's picture) purely because the latter scores better on
    `final_score`. This is a ranking rule only: it does not touch
    `final_score`, `face_match_threshold`, or whether a candidate verifies at
    all — `verified` still dominates every candidate that never qualified.
    """
    threshold = settings.high_face_similarity_priority
    return sorted(
        candidates,
        key=lambda c: (
            c.verified,
            c.face_similarity >= threshold,
            c.face_similarity,
            evidential_strength(c),
            c.final_score,
            # Negated so that a *lower* priority number sorts earlier under the
            # surrounding reverse=True.
            -c.platform_priority,
        ),
        reverse=True,
    )


def highest_stage_reached(candidates: list[VerifiedCandidate]) -> list[Stage]:
    """Union of rungs any candidate reached — used for the run's diagnostics."""
    reached = {s for c in candidates for s in c.stages}
    return [s for s in LADDER if s in reached]


def explain_failure(candidates: list[VerifiedCandidate]) -> str:
    """Say precisely which rung blocked verification.

    The reason must be arithmetically true of the candidate it names. A regression
    this guards against: reporting "final score 0.891 < threshold 0.7", which is
    false on its face. So the explanation is always derived from the single best
    candidate we actually measured, and it reports the first gate that candidate
    genuinely failed.
    """
    if not candidates:
        return "no candidates survived reverse image search"

    # Only unverified candidates can explain a failure to verify. Reporting a
    # verified candidate's numbers here would produce exactly the falsehood this
    # function exists to avoid — "score 0.935 < threshold 0.70".
    unverified = [c for c in candidates if not c.verified]
    if not unverified:
        return ""

    candidates = unverified
    measured = [c for c in candidates if c.candidate_image_sha256]

    if not measured:
        domains = ", ".join(sorted({c.domain for c in candidates})[:5])
        return (
            f"{len(candidates)} candidate(s) checked but no comparable image could be "
            f"retrieved from any of them — login walls or bot blocks ({domains})"
        )

    best = rank(measured)[0]
    if not best.face_detected:
        return f"no face detectable in the image retrieved from {best.domain}"
    if best.face_similarity < settings.face_match_threshold:
        return (
            f"best face similarity {best.face_similarity:.3f} < threshold "
            f"{settings.face_match_threshold} ({best.domain})"
        )
    return (
        f"best candidate {best.domain} scored {best.final_score:.3f} < threshold "
        f"{settings.verify_min_score} (face {best.face_similarity:.3f}, "
        f"image {best.image_similarity:.3f})"
    )
