"""Turns a matched KnownPerson's pre-vetted accounts into the API's
`official_profiles` list.

The account data itself (platform -> full profile URL) is sourced offline
from Wikidata's structured properties (see `scripts/build_identity_index.py`),
never scraped live per request. Whether an entry is labelled OFFICIAL or only
LIKELY_OFFICIAL depends on whether THIS scan's own reverse-image evidence
independently corroborates the URL — the index data alone is never enough to
call something OFFICIAL (mission section 9: "Never label an account
'official' merely because its username matches").
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..enrichment.extractor import extract_username_from_url
from ..models import SocialAccount, SocialAccountStatus
from ..search.base import host_matches, normalise_domain
from .models import KnownPerson

if TYPE_CHECKING:
    from ..models import Case

_OFFICIAL_SOURCE = "wikidata"
_CORROBORATED_SOURCE = "wikidata+this-scan-evidence"


def _corroborated(url: str, candidates: list) -> bool:
    domain = normalise_domain(url)
    if not domain:
        return False
    return any(host_matches(normalise_domain(c.url), domain) for c in candidates)


def build_official_profiles(person: KnownPerson, case: "Case") -> list[SocialAccount]:
    """One `SocialAccount` per entry in `person.official_website` /
    `person.official_social_accounts`. Never raises: a `case` with no search
    results yet just means nothing gets corroborated (every entry stays
    LIKELY_OFFICIAL rather than OFFICIAL).
    """
    search = getattr(case, "reverse_search", None)
    candidates = list(search.candidates) if search is not None else []

    profiles: list[SocialAccount] = []

    if person.official_website:
        corroborated = _corroborated(person.official_website, candidates)
        profiles.append(SocialAccount(
            platform="Website",
            url=person.official_website,
            status=SocialAccountStatus.OFFICIAL if corroborated else SocialAccountStatus.LIKELY_OFFICIAL,
            confidence=0.95 if corroborated else 0.70,
            source=_CORROBORATED_SOURCE if corroborated else _OFFICIAL_SOURCE,
            evidence=(
                ["matched among this scan's own reverse-image search candidates"]
                if corroborated else
                ["sourced from the known-person index (Wikidata official-website property)"]
            ),
        ).rounded())

    for platform, url in person.official_social_accounts.items():
        corroborated = _corroborated(url, candidates)
        profiles.append(SocialAccount(
            platform=platform,
            username=extract_username_from_url(url, platform) or "",
            url=url,
            status=SocialAccountStatus.OFFICIAL if corroborated else SocialAccountStatus.LIKELY_OFFICIAL,
            confidence=0.95 if corroborated else 0.70,
            source=_CORROBORATED_SOURCE if corroborated else _OFFICIAL_SOURCE,
            evidence=(
                ["matched among this scan's own reverse-image search candidates"]
                if corroborated else
                ["sourced from the known-person index (Wikidata structured account property)"]
            ),
        ).rounded())

    return profiles
