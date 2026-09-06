"""The single entry point `runner.py` calls for identity recognition.

Two functions, matching the two insertion points in the pipeline: a
face-embedding-only preliminary pass right after face encoding (covers a
photo where reverse-image search later finds nothing at all), and a
web-evidence enrichment pass once this scan's reverse-image search has run.

Both are safe to call unconditionally. Every failure mode — identity
disabled, index artifact absent, corrupt index, no match, an unexpected
exception anywhere in the identity stack — is swallowed here and returns
`None`/the untouched previous result rather than raising, so a broken or
missing identity layer can never break the existing scan (mission's
explicit hard rule).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from ..models import IdentityResult
from .evidence import cross_check
from .index import get_index
from .matcher import top_k
from .scorer import enrich as _enrich_score
from .scorer import score_preliminary

if TYPE_CHECKING:
    from ..models import Case

log = logging.getLogger(__name__)


def resolve_preliminary(embedding: np.ndarray | None) -> IdentityResult | None:
    """Face-embedding-only identity match. Call right after face encoding,
    before reverse-image search — never raises."""
    if embedding is None:
        return None
    try:
        index = get_index()
        if index.is_empty:
            return None
        matches = top_k(index, embedding)
        return score_preliminary(matches)
    except Exception as exc:  # noqa: BLE001 — identity layer must never break a scan
        log.warning("identity resolve_preliminary failed: %s: %s", type(exc).__name__, exc)
        return None


def enrich_with_evidence(
    preliminary: IdentityResult | None,
    embedding: np.ndarray | None,
    case: "Case",
) -> IdentityResult | None:
    """Re-score with this scan's own reverse-image/web evidence folded in.

    Also populates `case.official_profiles` (in place) from the matched
    person's pre-vetted accounts, once the result reaches at least MEDIUM.
    Returns `preliminary` unchanged if anything here fails or there is
    nothing to enrich.
    """
    if preliminary is None or embedding is None:
        return preliminary
    try:
        index = get_index()
        if index.is_empty:
            return preliminary
        matches = top_k(index, embedding)
        if not matches:
            return preliminary

        top_person = matches[0].person
        reverse_image_score, web_profile_score, extra = cross_check(top_person, case)
        result = _enrich_score(matches, reverse_image_score, web_profile_score, extra)

        if result is not None and result.name:
            from .social import build_official_profiles
            case.official_profiles = build_official_profiles(top_person, case)

        return result
    except Exception as exc:  # noqa: BLE001
        log.warning("identity enrich_with_evidence failed: %s: %s", type(exc).__name__, exc)
        return preliminary
