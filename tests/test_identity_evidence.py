"""Cross-checking a known-person candidate against a scan's OWN evidence.

Uses the real `SearchCandidate`/`SearchReport`/`Case` models directly —
these are exactly what `runner.py` already has in hand by the time
`identity/evidence.py::cross_check` runs; no second search is performed.
"""

from __future__ import annotations

from facechain.identity.evidence import cross_check
from facechain.identity.models import KnownPerson
from facechain.models import Case, SearchCandidate, SearchReport


def _case_with_candidates(candidates: list[SearchCandidate]) -> Case:
    return Case(
        case_id="case_test", created_at="now", observed_at=0,
        reverse_search=SearchReport(candidates=candidates),
    )


def test_no_search_results_yields_no_evidence():
    person = KnownPerson(person_id="p1", canonical_name="Jane Doe")
    case = Case(case_id="case_test", created_at="now", observed_at=0)  # reverse_search=None
    reverse_score, web_score, extra = cross_check(person, case)
    assert reverse_score == 0.0
    assert web_score == 0.0
    assert extra == []


def test_name_in_a_result_title_is_reverse_image_evidence():
    person = KnownPerson(person_id="p1", canonical_name="Jane Doe")
    case = _case_with_candidates([
        SearchCandidate(engine="yandex", url="https://news.example.com/a", domain="news.example.com",
                        title="Jane Doe wins award"),
    ])
    reverse_score, web_score, extra = cross_check(person, case)
    assert reverse_score == 1.0
    assert web_score == 0.0
    assert any(e.kind == "reverse_image" for e in extra)


def test_an_alias_also_counts_as_reverse_image_evidence():
    person = KnownPerson(person_id="p1", canonical_name="Jane Doe", aliases=["J. Doe"])
    case = _case_with_candidates([
        SearchCandidate(engine="yandex", url="https://news.example.com/a", domain="news.example.com",
                        title="Meet J. Doe, the new CEO"),
    ])
    reverse_score, _, _ = cross_check(person, case)
    assert reverse_score == 1.0


def test_unrelated_titles_yield_no_reverse_image_evidence():
    person = KnownPerson(person_id="p1", canonical_name="Jane Doe")
    case = _case_with_candidates([
        SearchCandidate(engine="yandex", url="https://news.example.com/a", domain="news.example.com",
                        title="Completely unrelated headline"),
    ])
    reverse_score, web_score, extra = cross_check(person, case)
    assert reverse_score == 0.0
    assert web_score == 0.0
    assert extra == []


def test_official_website_seen_among_candidates_is_web_profile_evidence():
    person = KnownPerson(person_id="p1", canonical_name="Jane Doe",
                         official_website="https://janedoe.com")
    case = _case_with_candidates([
        SearchCandidate(engine="yandex", url="https://janedoe.com/about", domain="janedoe.com", title=""),
    ])
    reverse_score, web_score, extra = cross_check(person, case)
    assert web_score == 1.0
    assert any(e.kind == "web_profile" for e in extra)


def test_official_social_account_domain_seen_is_web_profile_evidence():
    person = KnownPerson(person_id="p1", canonical_name="Jane Doe",
                         official_social_accounts={"GitHub": "https://github.com/janedoe"})
    case = _case_with_candidates([
        SearchCandidate(engine="yandex", url="https://github.com/janedoe", domain="github.com", title=""),
    ])
    _, web_score, _ = cross_check(person, case)
    assert web_score == 1.0


def test_lookalike_domain_does_not_count_as_web_profile_evidence():
    """github.com must not be confused with notgithub.com or
    github.com.attacker.com — reuses search.base.host_matches's dot-anchored
    check, same protection the reverse-image layer already has."""
    person = KnownPerson(person_id="p1", canonical_name="Jane Doe",
                         official_social_accounts={"GitHub": "https://github.com/janedoe"})
    case = _case_with_candidates([
        SearchCandidate(engine="yandex", url="https://notgithub.com/janedoe", domain="notgithub.com", title=""),
    ])
    _, web_score, _ = cross_check(person, case)
    assert web_score == 0.0
