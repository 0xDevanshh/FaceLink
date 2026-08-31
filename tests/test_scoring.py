"""The verification ladder is the pipeline's honesty guarantee — test it hard."""

import pytest

from facechain.config import settings
from facechain.models import Stage, VerifiedCandidate
from facechain.verification.scorer import (
    explain_failure,
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


def test_non_social_never_verifies_however_good():
    vc = score_candidate(make(is_social=False, platform=None, domain="news.example.com",
                              image_similarity=1.0, face_similarity=1.0))
    assert not vc.verified
    assert Stage.SOCIAL_MATCH not in vc.stages
    assert vc.final_score > settings.verify_min_score  # high score, still unverified


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


def test_highest_stage_reached_is_union_in_ladder_order():
    a = score_candidate(make(is_social=False, platform=None, face_similarity=0.9))
    b = score_candidate(make(face_similarity=0.0, face_detected=False, image_similarity=0.1))
    reached = highest_stage_reached([a, b])
    assert reached == [s for s in reached]  # ladder-ordered
    assert Stage.SOCIAL_MATCH in reached and Stage.IMAGE_MATCH in reached


# ---- failure explanation: must never state a falsehood --------------------

def test_explain_failure_reports_missing_social_not_a_bogus_threshold():
    """Regression: a 0.89-scoring non-social hit was reported as
    'final score 0.891 < threshold 0.7', which is arithmetically false."""
    good_but_not_social = score_candidate(
        make(is_social=False, platform=None, domain="archived.example",
             image_similarity=1.0, face_similarity=0.93)
    )
    reason = explain_failure([good_but_not_social])
    assert "social" in reason.lower()
    assert "< threshold" not in reason


def test_explain_failure_describes_social_candidates_only():
    social_weak = score_candidate(make(face_similarity=0.05, image_similarity=0.3))
    nonsocial_strong = score_candidate(make(is_social=False, platform=None,
                                            domain="x.example", face_similarity=0.99))
    reason = explain_failure([social_weak, nonsocial_strong])
    assert "instagram.com" in reason


def test_explain_failure_on_empty():
    assert "no candidates" in explain_failure([]).lower()


def test_explain_failure_when_images_unreachable():
    blocked = score_candidate(make(candidate_image_sha256=None, image_similarity=0.0,
                                   face_detected=False, face_similarity=0.0))
    assert "login wall" in explain_failure([blocked]) or "comparable image" in explain_failure([blocked])


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
