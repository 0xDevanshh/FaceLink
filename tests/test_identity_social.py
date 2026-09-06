"""Official social account discovery — never OFFICIAL from index data alone.

Mission section 9: "Never label an account 'official' merely because its
username matches." Index-sourced accounts (Wikidata) start as
LIKELY_OFFICIAL and only reach OFFICIAL when THIS scan's own reverse-image
evidence independently corroborates the domain.
"""

from __future__ import annotations

from facechain.identity.models import KnownPerson
from facechain.identity.social import build_official_profiles, discover_from_evidence, merge_profiles
from facechain.models import Case, SearchCandidate, SearchReport, SocialAccount, SocialAccountStatus, VerifiedCandidate


def _case_with_candidates(candidates: list[SearchCandidate]) -> Case:
    return Case(
        case_id="case_test", created_at="now", observed_at=0,
        reverse_search=SearchReport(candidates=candidates),
    )


def _verified(**kw) -> VerifiedCandidate:
    base = dict(
        engine="yandex", url="https://instagram.com/realjanedoe", domain="instagram.com",
        canonical_url="https://instagram.com/realjanedoe", platform="Instagram",
        verified=True, final_score=0.9, face_similarity=0.92,
    )
    base.update(kw)
    return VerifiedCandidate(**base)


def test_index_only_accounts_are_likely_official_not_official():
    person = KnownPerson(
        person_id="p1", canonical_name="Jane Doe",
        official_social_accounts={"GitHub": "https://github.com/janedoe"},
    )
    case = Case(case_id="case_test", created_at="now", observed_at=0)  # no search results at all
    profiles = build_official_profiles(person, case)
    assert len(profiles) == 1
    assert profiles[0].status == SocialAccountStatus.LIKELY_OFFICIAL
    assert profiles[0].confidence < 0.9


def test_corroborated_account_is_upgraded_to_official():
    person = KnownPerson(
        person_id="p1", canonical_name="Jane Doe",
        official_social_accounts={"GitHub": "https://github.com/janedoe"},
    )
    case = _case_with_candidates([
        SearchCandidate(engine="yandex", url="https://github.com/janedoe", domain="github.com", title=""),
    ])
    profiles = build_official_profiles(person, case)
    assert profiles[0].status == SocialAccountStatus.OFFICIAL
    assert profiles[0].confidence >= 0.9
    assert profiles[0].source == "wikidata+this-scan-evidence"


def test_a_fan_or_impersonator_domain_does_not_corroborate():
    """A candidate on a completely different domain must not upgrade an
    unrelated official account to OFFICIAL."""
    person = KnownPerson(
        person_id="p1", canonical_name="Jane Doe",
        official_social_accounts={"GitHub": "https://github.com/janedoe"},
    )
    case = _case_with_candidates([
        SearchCandidate(engine="yandex", url="https://fanpages.example.com/jane-doe-fan-club",
                        domain="fanpages.example.com", title="Jane Doe Fan Club"),
    ])
    profiles = build_official_profiles(person, case)
    assert profiles[0].status == SocialAccountStatus.LIKELY_OFFICIAL


def test_username_is_extracted_via_the_existing_enrichment_patterns():
    person = KnownPerson(
        person_id="p1", canonical_name="Jane Doe",
        official_social_accounts={"GitHub": "https://github.com/janedoe"},
    )
    case = Case(case_id="case_test", created_at="now", observed_at=0)
    profiles = build_official_profiles(person, case)
    assert profiles[0].username == "janedoe"
    assert profiles[0].platform == "GitHub"


def test_official_website_is_included_alongside_social_accounts():
    person = KnownPerson(
        person_id="p1", canonical_name="Jane Doe",
        official_website="https://janedoe.com",
        official_social_accounts={"GitHub": "https://github.com/janedoe"},
    )
    case = Case(case_id="case_test", created_at="now", observed_at=0)
    profiles = build_official_profiles(person, case)
    platforms = {p.platform for p in profiles}
    assert "Website" in platforms
    assert "GitHub" in platforms


def test_no_accounts_configured_yields_an_empty_list():
    person = KnownPerson(person_id="p1", canonical_name="Jane Doe")
    case = Case(case_id="case_test", created_at="now", observed_at=0)
    assert build_official_profiles(person, case) == []


# ---- discover_from_evidence: purely from this scan's own results ---------
# No URL is ever constructed from a name here — every returned `url` is a
# literal value taken straight from a SearchCandidate/VerifiedCandidate the
# (unmodified) reverse-image pipeline already produced.

def test_face_verified_candidate_is_official_the_strongest_tier():
    case = Case(
        case_id="case_test", created_at="now", observed_at=0,
        verification=[_verified()],
    )
    profiles = discover_from_evidence("Jane Doe", [], case)
    assert len(profiles) == 1
    p = profiles[0]
    assert p.platform == "Instagram"
    assert p.url == "https://instagram.com/realjanedoe"  # literal, not constructed
    assert p.status == SocialAccountStatus.OFFICIAL
    assert p.source == "reverse-image-face-verified"


def test_unverified_candidate_is_never_promoted_to_official():
    case = Case(
        case_id="case_test", created_at="now", observed_at=0,
        verification=[_verified(verified=False)],
    )
    assert discover_from_evidence("Jane Doe", [], case) == []


def test_multi_engine_candidate_without_face_verification_is_likely_official():
    case = _case_with_candidates([
        SearchCandidate(engine="yandex+tineye", url="https://x.com/realjanedoe",
                        domain="x.com", platform="X/Twitter", title=""),
    ])
    profiles = discover_from_evidence("Jane Doe", [], case)
    assert len(profiles) == 1
    assert profiles[0].status == SocialAccountStatus.LIKELY_OFFICIAL
    assert profiles[0].url == "https://x.com/realjanedoe"


def test_single_engine_with_name_in_title_is_unverified_not_ignored():
    case = _case_with_candidates([
        SearchCandidate(engine="yandex", url="https://github.com/realjanedoe",
                        domain="github.com", platform="GitHub", title="Jane Doe's projects"),
    ])
    profiles = discover_from_evidence("Jane Doe", [], case)
    assert len(profiles) == 1
    assert profiles[0].status == SocialAccountStatus.UNVERIFIED


def test_single_engine_without_any_name_match_is_not_surfaced_at_all():
    """The weakest possible case (one engine, no name mention) must not
    produce a guessed/unsupported entry — nothing shown beats something
    invented."""
    case = _case_with_candidates([
        SearchCandidate(engine="yandex", url="https://github.com/someoneelse",
                        domain="github.com", platform="GitHub", title="Unrelated repo"),
    ])
    assert discover_from_evidence("Jane Doe", [], case) == []


def test_an_alias_also_counts_for_the_weakest_tier():
    case = _case_with_candidates([
        SearchCandidate(engine="yandex", url="https://github.com/jd",
                        domain="github.com", platform="GitHub", title="J. Doe's repos"),
    ])
    profiles = discover_from_evidence("Jane Doe", ["J. Doe"], case)
    assert len(profiles) == 1


def test_non_target_platform_is_ignored():
    case = _case_with_candidates([
        SearchCandidate(engine="yandex+tineye", url="https://reddit.com/u/janedoe",
                        domain="reddit.com", platform="Reddit", title=""),
    ])
    assert discover_from_evidence("Jane Doe", [], case) == []


def test_no_reverse_search_or_verification_yields_empty_list():
    case = Case(case_id="case_test", created_at="now", observed_at=0)
    assert discover_from_evidence("Jane Doe", [], case) == []


def test_a_face_verified_post_is_not_treated_as_the_persons_own_official_profile():
    """Regression: a face match on a SPECIFIC POST (a news account's tweet
    about the person, a fan repost) only proves 'their photo is here' — not
    that the posting account belongs to them. Only a profile-shaped URL
    (not looks_like_post) may be labelled OFFICIAL."""
    case = Case(
        case_id="case_test", created_at="now", observed_at=0,
        verification=[_verified(
            url="https://x.com/newsaccount/status/1234567890123456789",
            canonical_url="https://x.com/newsaccount/status/1234567890123456789",
            platform="X/Twitter",
        )],
    )
    assert discover_from_evidence("Jane Doe", [], case) == []


def test_a_face_verified_profile_still_wins_when_a_post_for_the_same_platform_also_exists():
    case = Case(
        case_id="case_test", created_at="now", observed_at=0,
        verification=[
            _verified(url="https://x.com/newsaccount/status/1234567890123456789",
                     canonical_url="https://x.com/newsaccount/status/1234567890123456789",
                     platform="X/Twitter"),
            _verified(url="https://x.com/realjanedoe", canonical_url="https://x.com/realjanedoe",
                     platform="X/Twitter"),
        ],
    )
    profiles = discover_from_evidence("Jane Doe", [], case)
    assert len(profiles) == 1
    assert profiles[0].url == "https://x.com/realjanedoe"
    assert profiles[0].status == SocialAccountStatus.OFFICIAL


def test_multi_engine_post_is_not_promoted_to_likely_official():
    case = _case_with_candidates([
        SearchCandidate(engine="yandex+tineye", url="https://instagram.com/p/someposta/",
                        domain="instagram.com", platform="Instagram", title=""),
    ])
    assert discover_from_evidence("Jane Doe", [], case) == []


def test_single_engine_post_with_name_in_title_is_not_shown_even_as_unverified():
    case = _case_with_candidates([
        SearchCandidate(engine="yandex", url="https://facebook.com/someone/posts/12345",
                        domain="facebook.com", platform="Facebook", title="Jane Doe spotted"),
    ])
    assert discover_from_evidence("Jane Doe", [], case) == []


def test_face_verified_wins_over_a_weaker_signal_on_the_same_platform():
    case = Case(
        case_id="case_test", created_at="now", observed_at=0,
        verification=[_verified(platform="GitHub", url="https://github.com/real",
                                canonical_url="https://github.com/real")],
        reverse_search=SearchReport(candidates=[
            SearchCandidate(engine="yandex", url="https://github.com/fake",
                            domain="github.com", platform="GitHub", title="Jane Doe"),
        ]),
    )
    profiles = discover_from_evidence("Jane Doe", [], case)
    assert len(profiles) == 1
    assert profiles[0].url == "https://github.com/real"
    assert profiles[0].status == SocialAccountStatus.OFFICIAL


# ---- merge_profiles --------------------------------------------------------

def test_merge_profiles_prefers_the_higher_confidence_source():
    likely = SocialAccount(platform="Instagram", url="https://instagram.com/a",
                           status=SocialAccountStatus.LIKELY_OFFICIAL, confidence=0.7)
    official = SocialAccount(platform="Instagram", url="https://instagram.com/b",
                             status=SocialAccountStatus.OFFICIAL, confidence=0.95)
    merged = merge_profiles([likely], [official])
    assert len(merged) == 1
    assert merged[0].url == "https://instagram.com/b"


def test_merge_profiles_prefers_the_later_source_on_a_tie():
    index_based = SocialAccount(platform="GitHub", url="https://github.com/index-source",
                                status=SocialAccountStatus.LIKELY_OFFICIAL, confidence=0.7)
    evidence_based = SocialAccount(platform="GitHub", url="https://github.com/evidence-source",
                                   status=SocialAccountStatus.LIKELY_OFFICIAL, confidence=0.65)
    merged = merge_profiles([index_based], [evidence_based])
    assert len(merged) == 1
    assert merged[0].url == "https://github.com/evidence-source"


def test_merge_profiles_keeps_platforms_unique_to_either_source():
    index_only = SocialAccount(platform="Website", url="https://janedoe.com",
                               status=SocialAccountStatus.LIKELY_OFFICIAL, confidence=0.7)
    evidence_only = SocialAccount(platform="YouTube", url="https://youtube.com/channel/x",
                                  status=SocialAccountStatus.OFFICIAL, confidence=0.9)
    merged = merge_profiles([index_only], [evidence_only])
    assert {p.platform for p in merged} == {"Website", "YouTube"}


def test_merge_profiles_with_no_sources_is_empty():
    assert merge_profiles([], []) == []
