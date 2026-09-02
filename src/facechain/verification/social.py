"""Social-media classification and metadata-consistency scoring."""

from __future__ import annotations

from ..models import SearchCandidate, VerifiedCandidate
from ..search.base import classify_social, looks_like_post

__all__ = ["classify_social", "looks_like_post", "metadata_consistency"]


def metadata_consistency(candidate: SearchCandidate, verified: VerifiedCandidate) -> float:
    """A small corroboration score in [0, 1] — the 10% tiebreaker term.

    It deliberately rewards things that are hard to fake accidentally:
    multiple independent engines returning the same URL, a URL shaped like a
    real post, and an image we pulled from the page's own metadata rather than
    from the search engine's cache.
    """
    score = 0.0

    # Corroborated by more than one engine.
    if "+" in candidate.engine:
        score += 0.35

    # A specific post, not a profile root or homepage.
    if looks_like_post(candidate.url):
        score += 0.25

    # We reached the page ourselves.
    if verified.fetched:
        score += 0.2

    # The compared image came from the page (or is the page), not the engine's
    # thumbnail cache. `github:avatar` and `direct-image` rank with the metadata
    # sources because both are the platform's own canonical asset.
    if verified.candidate_image_source in (
        "og:image", "twitter:image", "json-ld", "link:image_src",
        "github:avatar", "direct-image",
    ):
        score += 0.15
    elif verified.candidate_image_source == "img":
        score += 0.1

    # A real title/caption was present.
    if len(candidate.title.strip()) >= 12:
        score += 0.05

    return min(1.0, score)
