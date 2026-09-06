"""Cross-checks a known-person candidate against THIS scan's own evidence.

Not a second live search: `case.reverse_search.candidates` and
`case.verification` were already gathered by the existing reverse-image
pipeline for this exact image, before the identity layer ever runs (see the
two insertion points in `runner.py`). This module only asks whether that
already-collected evidence happens to corroborate one specific known-person
candidate — reusing `search.base.normalise_domain`/`host_matches` rather than
re-implementing domain matching.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..models import IdentityEvidence
from ..search.base import host_matches, normalise_domain
from .models import KnownPerson

if TYPE_CHECKING:
    from ..models import Case


def _name_variants(person: KnownPerson) -> list[str]:
    return [n.lower() for n in [person.canonical_name, *person.aliases] if n]


def _official_domains(person: KnownPerson) -> set[str]:
    domains = set()
    if person.official_website:
        d = normalise_domain(person.official_website)
        if d:
            domains.add(d)
    for handle in person.official_social_accounts.values():
        if handle.startswith("http"):
            d = normalise_domain(handle)
            if d:
                domains.add(d)
    return domains


def cross_check(person: KnownPerson, case: "Case") -> tuple[float, float, list[IdentityEvidence]]:
    """Returns `(reverse_image_score, web_profile_score, extra_evidence)`.

    Both scores are in `{0.0, 1.0}` — this is a yes/no corroboration signal
    per source, not a graded similarity; `identity/scorer.py` is what turns
    it into a weighted contribution. Never raises: a `case` with no search
    results yet (or none at all) just yields `(0.0, 0.0, [])`.
    """
    extra: list[IdentityEvidence] = []
    reverse_image_score = 0.0
    web_profile_score = 0.0

    search = getattr(case, "reverse_search", None)
    candidates = list(search.candidates) if search is not None else []
    names = _name_variants(person)

    for cand in candidates:
        title = (cand.title or "").lower()
        if any(n in title for n in names):
            reverse_image_score = 1.0
            extra.append(IdentityEvidence(
                kind="reverse_image",
                description=(
                    f"'{person.canonical_name}' appears in a reverse-image search "
                    f"result title from {cand.domain}"
                ),
                weight=0.0,  # already folded into scorer._score's reverse_image_score
                value=1.0,
            ))
            break

    official_domains = _official_domains(person)
    if official_domains:
        for cand in candidates:
            host = normalise_domain(cand.url)
            if any(host_matches(host, d) for d in official_domains):
                web_profile_score = 1.0
                extra.append(IdentityEvidence(
                    kind="web_profile",
                    description=(
                        f"a known official domain for {person.canonical_name} "
                        f"({cand.domain}) appears among this scan's candidates"
                    ),
                    weight=0.0,
                    value=1.0,
                ))
                break

    return reverse_image_score, web_profile_score, extra
