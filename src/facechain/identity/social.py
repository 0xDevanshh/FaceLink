"""Social profile discovery for a resolved identity.

Two independent, complementary evidence sources feed `case.official_profiles`
— never a username/URL guessed from the person's name:

1. `build_official_profiles` — the matched `KnownPerson`'s pre-vetted
   accounts, sourced offline from Wikidata's structured properties (see
   `scripts/build_identity_index.py`), corroborated (OFFICIAL) or not
   (LIKELY_OFFICIAL) against THIS scan's own reverse-image evidence.

2. `discover_from_evidence` — profiles found directly among THIS scan's own
   reverse-image search results (`case.reverse_search`/`case.verification`),
   independent of whether the index has anything for this platform at all.
   This is what lets a genuinely new platform account surface (Wikidata
   coverage of social handles is always partial) and is what makes a person
   who isn't even in the known-person index's curated account list — only
   face-matched via `KnownPersonIndex` — still get real, evidence-backed
   profiles. Every returned URL is a literal value some reverse-image engine
   actually returned for this exact scan; nothing here is constructed.

`merge_profiles` combines both into one entry per platform for the UI,
without inventing anything neither source actually found.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..enrichment.extractor import extract_username_from_url
from ..models import SocialAccount, SocialAccountStatus
from ..search.base import host_matches, looks_like_post, normalise_domain
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


# Platforms this discovery path surfaces. Any other named platform
# (`config.py::SOCIAL_DOMAINS` recognises more) is still real evidence, but
# the "Social Profiles" UI section is scoped to these per the mission.
_TARGET_PLATFORMS = frozenset({
    "Instagram", "X/Twitter", "LinkedIn", "Facebook", "YouTube", "GitHub",
})

_STATUS_RANK: dict[SocialAccountStatus, int] = {
    SocialAccountStatus.OFFICIAL: 3,
    SocialAccountStatus.LIKELY_OFFICIAL: 2,
    SocialAccountStatus.UNVERIFIED: 1,
    SocialAccountStatus.REJECTED: 0,
}


def _name_variants(name: str, aliases: list[str]) -> list[str]:
    return [n.lower() for n in [name, *aliases] if n]


def discover_from_evidence(name: str, aliases: list[str], case: "Case") -> list[SocialAccount]:
    """Discover social profiles purely from THIS scan's own reverse-image
    search results — never a username/URL constructed from `name`. Every
    returned `url` is a literal value some reverse-image engine actually
    returned this scan; `name`/`aliases` are used only to judge the weakest
    tier below, never to build a URL.

    Every tier requires a *profile-shaped* URL (`not looks_like_post`), not
    merely a hit on the right platform. A face match on a specific post only
    proves "this person's photo appears here" — a news account's tweet about
    someone, or a fan repost, face-matches just as well as their own account
    would, and reusing `search/base.py::looks_like_post` (the same
    profile-vs-post distinction the reverse-image ranking already relies on)
    is what tells "their own profile page" apart from "a post that happens
    to feature them." Getting this wrong would have labelled other people's
    posts about someone as that person's own "OFFICIAL" account.

    Three tiers, each backed by a specific, already-computed signal — never
    invented here:

      OFFICIAL        `VerifiedCandidate.verified` on a profile-shaped URL —
                      InsightFace independently matched the query face to
                      the face found on that exact page, and the URL itself
                      is the account's own page, not a specific post.
      LIKELY_OFFICIAL a profile-shaped URL found independently by 2+
                      reverse-image engines (`SearchCandidate.engine`
                      already records this via the orchestrator's
                      cross-engine merge — see
                      `search/orchestrator.py::_merge_candidates`).
      UNVERIFIED      a profile-shaped URL found by exactly one engine, with
                      the resolved name/alias appearing in the result's own
                      title — the weakest tier shown, explicitly labelled.

    One entry per platform, best tier wins. Never raises: absent
    `reverse_search`/`verification` on `case` just yields fewer/no entries.
    """
    names = _name_variants(name, aliases)
    verification = list(getattr(case, "verification", None) or [])
    search = getattr(case, "reverse_search", None)
    raw_candidates = list(search.candidates) if search is not None else []

    by_platform: dict[str, SocialAccount] = {}

    def _is_profile(url: str) -> bool:
        return bool(url) and not looks_like_post(url)

    # Tier 1 — face-verified: the strongest possible evidence, restricted to
    # the account's own profile page (see docstring above).
    for vc in verification:
        if not vc.verified or not vc.platform or vc.platform not in _TARGET_PLATFORMS:
            continue
        if vc.platform in by_platform:
            continue
        url = vc.canonical_url or vc.url
        if not _is_profile(url):
            continue
        by_platform[vc.platform] = SocialAccount(
            platform=vc.platform,
            username=extract_username_from_url(url, vc.platform) or "",
            url=url,
            status=SocialAccountStatus.OFFICIAL,
            confidence=max(0.9, min(vc.final_score, 1.0)),
            source="reverse-image-face-verified",
            evidence=[
                f"InsightFace independently matched the query face to the face found "
                f"on this exact {vc.platform} profile page (face similarity {vc.face_similarity:.2f})",
            ],
        ).rounded()

    # Tier 2 — corroborated by 2+ independent reverse-image engines.
    for cand in raw_candidates:
        if not cand.platform or cand.platform not in _TARGET_PLATFORMS or cand.platform in by_platform:
            continue
        if "+" not in cand.engine or not _is_profile(cand.url):
            continue
        by_platform[cand.platform] = SocialAccount(
            platform=cand.platform,
            username=extract_username_from_url(cand.url, cand.platform) or "",
            url=cand.url,
            status=SocialAccountStatus.LIKELY_OFFICIAL,
            confidence=0.65,
            source="reverse-image-multi-engine",
            evidence=[f"independently found by multiple reverse-image engines ({cand.engine})"],
        ).rounded()

    # Tier 3 — single engine, but the resolved name/alias appears in the
    # result's own title. Weakest tier: shown, but explicitly UNVERIFIED.
    for cand in raw_candidates:
        if not cand.platform or cand.platform not in _TARGET_PLATFORMS or cand.platform in by_platform:
            continue
        if not _is_profile(cand.url):
            continue
        title = (cand.title or "").lower()
        if names and any(n in title for n in names):
            by_platform[cand.platform] = SocialAccount(
                platform=cand.platform,
                username=extract_username_from_url(cand.url, cand.platform) or "",
                url=cand.url,
                status=SocialAccountStatus.UNVERIFIED,
                confidence=0.35,
                source="reverse-image-name-match",
                evidence=[
                    f"'{name}' appears in this {cand.platform} result's title "
                    f"(found by {cand.engine}) — not independently corroborated",
                ],
            ).rounded()

    return list(by_platform.values())


def merge_profiles(*sources: list[SocialAccount]) -> list[SocialAccount]:
    """Merge multiple `SocialAccount` lists into one entry per platform.

    Never invents an account neither source found — purely picks, per
    platform, whichever input already found the higher-confidence one. On a
    tie, the source passed *later* wins: callers should pass evidence-based
    discovery (a literal URL seen in this exact scan) after index-based
    discovery (a static Wikidata record), since a live, this-scan match is
    the more directly proven of the two.
    """
    best: dict[str, SocialAccount] = {}
    for source in sources:
        for account in source:
            existing = best.get(account.platform)
            if existing is None or _STATUS_RANK[account.status] >= _STATUS_RANK[existing.status]:
                best[account.platform] = account
    return list(best.values())
