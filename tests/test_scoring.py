"""The verification ladder is the pipeline's honesty guarantee — test it hard."""

import pytest

from facechain.config import settings
from facechain.models import Stage, VerifiedCandidate
from facechain.verification.clustering import CorroborationSummary
from facechain.verification.evidence_graph import EvidenceGraph
from facechain.verification.scorer import (
    explain_failure,
    explain_verified,
    highest_stage_reached,
    rank,
    score_candidate,
)
from facechain.verification.social import metadata_consistency
from facechain.models import SearchCandidate


def make(**kw) -> VerifiedCandidate:
    base = dict(
        engine="yandex",
        url="https://instagram.com/p/ABC123/",
        domain="instagram.com",
        platform="Instagram",
        is_social=True,
        fetched=True,
        candidate_image_sha256="ab" * 32,
        candidate_image_source="og:image",
        image_similarity=0.95,
        face_detected=True,
        face_similarity=0.90,
        metadata_consistency=0.7,
    )
    base.update(kw)
    return VerifiedCandidate(**base)


def test_strong_social_match_verifies():
    vc = score_candidate(make())
    assert vc.verified
    assert vc.match_type == "exact-image"
    assert Stage.VERIFIED in vc.stages
    assert vc.stages == [
        Stage.SEARCH_FOUND, Stage.SOCIAL_MATCH, Stage.IMAGE_MATCH,
        Stage.FACE_MATCH, Stage.VERIFIED,
    ]


def test_strong_non_social_match_verifies_on_face_evidence():
    """Search priority must not become search exclusivity.

    A face-verified match on a personal site, a university page or a conference
    programme is real evidence. It used to be unverifiable purely because its
    domain was absent from the platform table, which let a platform name stand
    in for a measurement.
    """
    vc = score_candidate(make(is_social=False, platform=None, domain="news.example.com",
                              platform_priority=90,
                              image_similarity=1.0, face_similarity=1.0))
    assert vc.verified
    assert Stage.SOCIAL_MATCH not in vc.stages   # honest about where it came from
    assert Stage.FACE_MATCH in vc.stages          # and about what was measured


def test_non_social_with_weak_face_still_fails():
    """Dropping the social requirement must not lower the face bar."""
    vc = score_candidate(make(is_social=False, platform=None, domain="news.example.com",
                              platform_priority=90,
                              image_similarity=1.0, face_similarity=0.10,
                              metadata_consistency=1.0))
    assert not vc.verified
    assert Stage.FACE_MATCH not in vc.stages


def test_no_candidate_can_verify_without_a_face_match():
    """The arithmetic, not a policy, is what blocks a non-matching face.

    Face weight is 0.5, so image (0.4) plus metadata (0.1) top out at 0.5 —
    below the 0.70 minimum. This asserts the weights still make that true.
    """
    ceiling = settings.weight_image + settings.weight_meta
    assert ceiling < settings.verify_min_score


def test_wrong_person_cannot_pass_on_image_similarity_alone():
    """The central anti-false-positive property."""
    vc = score_candidate(make(face_similarity=0.02, image_similarity=1.0,
                              metadata_consistency=1.0))
    assert not vc.verified
    assert Stage.FACE_MATCH not in vc.stages
    assert vc.final_score < settings.verify_min_score


def test_no_face_in_candidate_cannot_verify():
    vc = score_candidate(make(face_detected=False, face_similarity=0.0))
    assert not vc.verified
    assert Stage.FACE_MATCH not in vc.stages


def test_edited_repost_verifies_as_face_only():
    """A cropped/overlaid social repost: face is unmistakable, phash is not."""
    vc = score_candidate(make(image_similarity=0.75, face_similarity=0.97))
    assert vc.verified
    assert vc.match_type == "face-only"
    assert Stage.IMAGE_MATCH not in vc.stages
    assert Stage.FACE_MATCH in vc.stages


# ---- face_only_verify_enabled: opt-in face-similarity-alone acceptance ----

def test_disabled_by_default_a_strong_face_with_weak_image_still_fails():
    """Regression guard: this path must be off unless explicitly enabled."""
    assert not settings.face_only_verify_enabled
    vc = score_candidate(make(face_similarity=0.75, image_similarity=0.05,
                              metadata_consistency=0.0))
    assert not vc.verified


def test_enabled_a_strong_face_alone_verifies_despite_weak_image_and_metadata(monkeypatch):
    monkeypatch.setattr(settings, "face_only_verify_enabled", True)
    monkeypatch.setattr(settings, "face_only_verify_threshold", 0.50)
    vc = score_candidate(make(face_similarity=0.75, image_similarity=0.05,
                              metadata_consistency=0.0))
    assert vc.verified
    assert Stage.VERIFIED in vc.stages


def test_enabled_but_below_the_face_only_threshold_still_fails(monkeypatch):
    monkeypatch.setattr(settings, "face_only_verify_enabled", True)
    monkeypatch.setattr(settings, "face_only_verify_threshold", 0.50)
    vc = score_candidate(make(face_similarity=0.49, image_similarity=0.05,
                              metadata_consistency=0.0))
    assert not vc.verified


def test_enabled_still_requires_a_real_face_match_not_just_a_face_similarity_number(monkeypatch):
    """face_only_ok requires `face_ok` (face_detected AND >= face_match_threshold)
    — this path can never verify a candidate whose face didn't match at all,
    regardless of the face_only threshold."""
    monkeypatch.setattr(settings, "face_only_verify_enabled", True)
    monkeypatch.setattr(settings, "face_only_verify_threshold", 0.50)
    vc = score_candidate(make(face_similarity=0.60, face_detected=False,
                              image_similarity=0.05, metadata_consistency=0.0))
    assert not vc.verified


def test_enabled_does_not_change_the_final_score_formula(monkeypatch):
    """Additive only — the score itself, and the combined-score path, are
    both unaffected by this flag."""
    monkeypatch.setattr(settings, "face_only_verify_enabled", True)
    vc = score_candidate(make(face_similarity=0.91, image_similarity=0.50, metadata_consistency=0.5))
    assert vc.final_score == pytest.approx(0.5 * 0.91 + 0.4 * 0.50 + 0.1 * 0.5)


def test_face_just_below_threshold_fails():
    vc = score_candidate(make(face_similarity=settings.face_match_threshold - 0.01,
                              image_similarity=0.99, metadata_consistency=1.0))
    assert Stage.FACE_MATCH not in vc.stages
    assert not vc.verified


def test_weights_sum_to_one():
    assert settings.weight_face + settings.weight_image + settings.weight_meta == pytest.approx(1.0)


def test_score_is_weighted_sum():
    vc = score_candidate(make(face_similarity=0.8, image_similarity=0.6, metadata_consistency=0.5))
    expected = 0.5 * 0.8 + 0.4 * 0.6 + 0.1 * 0.5
    assert vc.final_score == pytest.approx(expected)


def test_negative_cosine_does_not_reduce_below_zero_contribution():
    vc = score_candidate(make(face_similarity=-0.5, image_similarity=0.5,
                              metadata_consistency=0.0))
    assert vc.final_score == pytest.approx(0.4 * 0.5)


def test_rank_puts_verified_first():
    weak = score_candidate(make(url="https://instagram.com/p/weak/", face_similarity=0.1,
                                image_similarity=0.2))
    strong = score_candidate(make(url="https://instagram.com/p/strong/"))
    assert rank([weak, strong])[0].url.endswith("strong/")


# ---- face_similarity >= high_face_similarity_priority ranking rule --------
#
# A ranking rule only: it must never affect score_candidate()'s final_score,
# face_match_threshold, or verified/stages — only the order rank() returns
# verified candidates in.

def _ranked(url: str, face_similarity: float, final_score: float,
           platform_priority: int = 90) -> VerifiedCandidate:
    return VerifiedCandidate(
        engine="yandex", url=url, domain=url.split("/")[2],
        platform_priority=platform_priority,
        face_similarity=face_similarity, final_score=final_score,
        verified=True, stages=[Stage.SEARCH_FOUND, Stage.FACE_MATCH, Stage.VERIFIED],
    )


def test_face_similarity_of_exactly_the_threshold_qualifies():
    assert settings.high_face_similarity_priority == 0.75
    at_threshold = _ranked("https://a.com/1", face_similarity=0.75, final_score=0.60)
    # A much higher final_score must not be enough to outrank the priority group.
    just_below = _ranked("https://b.com/1", face_similarity=0.749, final_score=0.99)
    ranked = rank([just_below, at_threshold])
    assert ranked[0].url == "https://a.com/1"


def test_face_similarity_of_0_749_does_not_qualify():
    just_below = _ranked("https://b.com/1", face_similarity=0.749, final_score=0.99)
    much_lower = _ranked("https://c.com/1", face_similarity=0.30, final_score=0.10)
    # Outside the priority group, existing signals (here: final_score) still
    # decide — 0.749 does not get special treatment.
    ranked = rank([much_lower, just_below])
    assert ranked[0].url == "https://b.com/1"


def test_priority_group_members_rank_by_face_similarity_not_final_score():
    """A strong-face/weak-image hit must not be outranked by a weak-face/
    strong-image hit purely because the latter scores better overall."""
    strong_face_weak_image = _ranked("https://a.com/1", face_similarity=0.91, final_score=0.60)
    weak_face_strong_image = _ranked("https://b.com/1", face_similarity=0.76, final_score=0.95)
    ranked = rank([weak_face_strong_image, strong_face_weak_image])
    assert ranked[0].url == "https://a.com/1"


def test_the_documented_worked_example_ranks_exactly_as_specified():
    """LinkedIn 0.91, Instagram 0.78, X 0.72, YouTube 0.61 — the two
    qualifying (>=0.75) candidates lead regardless of image similarity."""
    linkedin = _ranked("https://linkedin.com/in/x", face_similarity=0.91, final_score=0.853)
    instagram = _ranked("https://instagram.com/p/x", face_similarity=0.78, final_score=0.84)
    x_com = _ranked("https://x.com/x", face_similarity=0.72, final_score=0.826)
    youtube = _ranked("https://youtube.com/x", face_similarity=0.61, final_score=0.767)

    ranked = rank([youtube, x_com, instagram, linkedin])
    assert [c.url for c in ranked] == [
        "https://linkedin.com/in/x", "https://instagram.com/p/x",
        "https://x.com/x", "https://youtube.com/x",
    ]


def test_priority_ranking_never_promotes_an_unverified_candidate():
    """Verification status still dominates everything — a high face
    similarity on a candidate that never verified must not outrank a
    verified one, priority group or not."""
    unverified_high_face = VerifiedCandidate(
        engine="yandex", url="https://a.com/1", domain="a.com",
        face_similarity=0.95, final_score=0.50, verified=False,
        stages=[Stage.SEARCH_FOUND],
    )
    verified_lower_face = _ranked("https://b.com/1", face_similarity=0.40, final_score=0.71)
    ranked = rank([unverified_high_face, verified_lower_face])
    assert ranked[0].url == "https://b.com/1"


def test_priority_ranking_does_not_alter_the_final_score_formula():
    """The new ranking rule must be additive — score_candidate()'s output is
    unaffected by high_face_similarity_priority."""
    vc = score_candidate(make(face_similarity=0.91, image_similarity=0.50, metadata_consistency=0.5))
    assert vc.final_score == pytest.approx(0.5 * 0.91 + 0.4 * 0.50 + 0.1 * 0.5)


def test_highest_stage_reached_is_union_in_ladder_order():
    a = score_candidate(make(is_social=False, platform=None, face_similarity=0.9))
    b = score_candidate(make(face_similarity=0.0, face_detected=False, image_similarity=0.1))
    reached = highest_stage_reached([a, b])
    assert reached == [s for s in reached]  # ladder-ordered
    assert Stage.SOCIAL_MATCH in reached and Stage.IMAGE_MATCH in reached


# ---- failure explanation: must never state a falsehood --------------------

def test_explain_failure_never_claims_a_threshold_failure_that_did_not_happen():
    """Regression: a 0.89-scoring hit was reported as
    'final score 0.891 < threshold 0.7', which is arithmetically false.

    A candidate that cleared every gate cannot be cited as the reason nothing
    verified, so it must not appear in the explanation at all.
    """
    verified = score_candidate(
        make(is_social=False, platform=None, domain="archived.example",
             image_similarity=1.0, face_similarity=0.93)
    )
    assert verified.verified
    assert explain_failure([verified]) == ""


def test_explain_failure_describes_the_best_unverified_candidate():
    weak_social = score_candidate(make(face_similarity=0.05, image_similarity=0.3))
    weaker_web = score_candidate(make(is_social=False, platform=None,
                                      domain="x.example", face_similarity=0.01,
                                      image_similarity=0.1))
    reason = explain_failure([weak_social, weaker_web])
    # The strongest *unverified* candidate is the social one, so it is the one
    # named, and the stated similarity is really its own.
    assert "instagram.com" in reason
    assert "0.050" in reason


def test_explain_failure_on_empty():
    assert "no candidates" in explain_failure([]).lower()


def test_explain_failure_when_images_unreachable():
    blocked = score_candidate(make(candidate_image_sha256=None, image_similarity=0.0,
                                   face_detected=False, face_similarity=0.0))
    assert "login wall" in explain_failure([blocked]) or "comparable image" in explain_failure([blocked])


# ---- verified-rank explanation: the positive counterpart to explain_failure

def test_explain_verified_is_empty_for_an_unverified_candidate():
    rejected = score_candidate(make(face_similarity=0.05, image_similarity=0.1))
    assert not rejected.verified
    assert explain_verified(rejected) == ""


def test_explain_verified_cites_the_candidates_own_measured_numbers():
    vc = score_candidate(make(face_similarity=0.92, image_similarity=0.95))
    assert vc.verified
    explanation = explain_verified(vc)
    assert f"{vc.face_similarity:.3f}" in explanation
    assert vc.confidence_band.lower() in explanation.lower()


def test_explain_verified_distinguishes_exact_image_from_face_only():
    exact = score_candidate(make(face_similarity=0.92, image_similarity=0.95))
    assert "exact retrieved image" in explain_verified(exact)

    # Below the exact-image threshold but a high enough combined score to
    # still verify — a real repost/edit scenario, not the exact same picture.
    face_only = score_candidate(make(face_similarity=0.99, image_similarity=0.50,
                                     metadata_consistency=1.0))
    assert face_only.verified and face_only.match_type == "face-only"
    assert "different picture of the same face" in explain_verified(face_only)


def test_explain_verified_names_the_platform_and_its_tier():
    priority = score_candidate(make(platform="Instagram", platform_priority=2,
                                    face_similarity=0.92, image_similarity=0.95))
    assert "Instagram" in explain_verified(priority)
    assert "priority platform" in explain_verified(priority)

    wider_web = score_candidate(make(is_social=False, platform=None, platform_priority=90,
                                     domain="news.example.com",
                                     face_similarity=0.92, image_similarity=0.95))
    # No platform name at all — nothing false is claimed about provenance.
    assert "found on" not in explain_verified(wider_web)


def test_explain_verified_cites_independent_evidence_when_present():
    vc = score_candidate(make(face_similarity=0.92, image_similarity=0.95))
    graph = EvidenceGraph(independent_evidence_count=3)
    explanation = explain_verified(vc, graph=graph)
    assert "3 independent sources" in explanation


def test_explain_verified_falls_back_to_corroboration_domains_without_a_graph():
    vc = score_candidate(make(face_similarity=0.92, image_similarity=0.95))
    corr = CorroborationSummary(independent_domains=2)
    explanation = explain_verified(vc, corr=corr)
    assert "2 distinct domains" in explanation


def test_explain_verified_never_claims_corroboration_that_was_not_found():
    vc = score_candidate(make(face_similarity=0.92, image_similarity=0.95))
    assert explain_verified(vc) != ""
    assert "independent source" not in explain_verified(vc)
    assert "distinct domains" not in explain_verified(vc)


# ---- metadata consistency -------------------------------------------------

def test_metadata_rewards_multi_engine_corroboration():
    cand = SearchCandidate(engine="yandex", url="https://instagram.com/p/ABC/",
                           domain="instagram.com", is_social=True, platform="Instagram")
    both = SearchCandidate(**{**cand.model_dump(), "engine": "yandex+bing"})
    vc = make()
    assert metadata_consistency(both, vc) > metadata_consistency(cand, vc)


def test_metadata_prefers_page_image_over_engine_thumbnail():
    cand = SearchCandidate(engine="yandex", url="https://instagram.com/p/ABC/",
                           domain="instagram.com", is_social=True, platform="Instagram")
    from_page = make(candidate_image_source="og:image")
    from_engine = make(candidate_image_source="engine-thumbnail")
    assert metadata_consistency(cand, from_page) > metadata_consistency(cand, from_engine)


def test_metadata_is_bounded():
    cand = SearchCandidate(engine="a+b+c", url="https://instagram.com/p/ABC/xyz/",
                           domain="instagram.com", is_social=True, platform="Instagram",
                           title="a fairly long caption here")
    assert 0.0 <= metadata_consistency(cand, make()) <= 1.0
