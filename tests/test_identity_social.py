"""Official social account discovery — never OFFICIAL from index data alone.

Mission section 9: "Never label an account 'official' merely because its
username matches." Index-sourced accounts (Wikidata) start as
LIKELY_OFFICIAL and only reach OFFICIAL when THIS scan's own reverse-image
evidence independently corroborates the domain.
"""

from __future__ import annotations

from facechain.identity.models import KnownPerson
from facechain.identity.social import build_official_profiles
from facechain.models import Case, SearchCandidate, SearchReport, SocialAccountStatus


def _case_with_candidates(candidates: list[SearchCandidate]) -> Case:
    return Case(
        case_id="case_test", created_at="now", observed_at=0,
        reverse_search=SearchReport(candidates=candidates),
    )


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
