"""Group near-identical candidate images so duplicates don't masquerade as
independent evidence.

The same photograph turns up under multiple URLs constantly: the original post,
a CDN copy, a Pinterest re-pin, a mirror site, a resized thumbnail. All of
these contain the same face evidence exactly once. Counting them as separate
confirmations would inflate confidence in a way that is arithmetically correct
but epistemically wrong — it would let one widely-reposted photo score as if
several independent people had posted it.

Each cluster has one *canonical member* (the one with the best candidate
quality score), and all others are *duplicates*. The evidence summary reports
cluster count rather than raw candidate count so a reader knows how many
distinct visual sources actually exist.

Duplicate detection uses the perceptual hash already computed during
candidate verification (no extra downloads). Two images whose pHash differs by
≤ MAX_HAMMING bits are considered the same picture for evidence purposes. The
threshold is conservative: minor re-encoding and resizing stay together, while
a genuine different photograph of the same face does not.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import imagehash

from ..models import VerifiedCandidate

log = logging.getLogger(__name__)

# Hamming distance ≤ this → same picture.
MAX_HAMMING = 6


@dataclass
class ImageCluster:
    """One group of near-identical images."""

    canonical: VerifiedCandidate
    duplicates: list[VerifiedCandidate] = field(default_factory=list)

    @property
    def size(self) -> int:
        return 1 + len(self.duplicates)

    @property
    def domains(self) -> set[str]:
        return {self.canonical.domain} | {d.domain for d in self.duplicates}

    @property
    def platforms(self) -> set[str]:
        platforms = set()
        for vc in [self.canonical, *self.duplicates]:
            if vc.platform:
                platforms.add(vc.platform)
        return platforms

    @property
    def best_face_similarity(self) -> float:
        return max(
            (c.face_similarity for c in [self.canonical, *self.duplicates]),
            default=0.0,
        )


def _phash_or_none(candidate: VerifiedCandidate) -> imagehash.ImageHash | None:
    try:
        if not candidate.candidate_image_phash:
            return None
        return imagehash.hex_to_hash(candidate.candidate_image_phash)
    except Exception:  # noqa: BLE001
        return None


def cluster_candidates(candidates: list[VerifiedCandidate]) -> list[ImageCluster]:
    """Group candidates by perceptual image similarity.

    Candidates without a measured image (login-wall misses, etc.) each get
    their own singleton cluster so they still appear in the summary without
    inflating evidence counts.

    Algorithm: greedy union-find. O(n²) on phash comparisons, acceptable for
    the candidate counts this pipeline produces (≤ ~100 per scan).
    """
    if not candidates:
        return []

    clusters: list[ImageCluster] = []
    assigned: set[int] = set()

    for i, cand in enumerate(candidates):
        if i in assigned:
            continue
        ph_i = _phash_or_none(cand)
        cluster = ImageCluster(canonical=cand)

        for j, other in enumerate(candidates):
            if j <= i or j in assigned:
                continue
            ph_j = _phash_or_none(other)
            if ph_i is not None and ph_j is not None:
                try:
                    if ph_i - ph_j <= MAX_HAMMING:
                        cluster.duplicates.append(other)
                        assigned.add(j)
                        log.debug(
                            "cluster: %s and %s share the same image (Hamming %d)",
                            cand.domain, other.domain, ph_i - ph_j,
                        )
                except Exception:  # noqa: BLE001
                    pass

        assigned.add(i)
        # Keep the strongest member as canonical.
        all_members = [cluster.canonical, *cluster.duplicates]
        all_members.sort(key=lambda c: (c.face_similarity, c.image_similarity), reverse=True)
        cluster.canonical = all_members[0]
        cluster.duplicates = all_members[1:]
        clusters.append(cluster)

    return clusters


@dataclass
class CorroborationSummary:
    """Evidence independence metrics for a set of candidates.

    These are recorded in the manifest so a reader can tell apart:
      * five URLs of the same picture → one image cluster, one source
      * five different pictures of the same face → five clusters, stronger evidence
    """

    total_candidates: int = 0
    image_clusters: int = 0          # distinct visual sources (post-dedup)
    duplicate_count: int = 0         # candidates that are image-duplicates
    independent_domains: int = 0     # distinct eTLD+1 hostnames
    independent_platforms: int = 0   # distinct named platforms
    verified_clusters: int = 0       # clusters whose canonical candidate verified
    best_face_similarity: float = 0.0


def corroboration_summary(
    clusters: list[ImageCluster],
    verified_only: bool = False,
) -> CorroborationSummary:
    """Aggregate the evidence-independence picture."""
    subject = [c for c in clusters if c.canonical.verified] if verified_only else clusters

    domains: set[str] = set()
    platforms: set[str] = set()
    best_sim = 0.0

    for cl in clusters:
        domains.update(cl.domains)
        platforms.update(pl for pl in cl.platforms if pl)
        best_sim = max(best_sim, cl.best_face_similarity)

    return CorroborationSummary(
        total_candidates=sum(cl.size for cl in clusters),
        image_clusters=len(clusters),
        duplicate_count=sum(len(cl.duplicates) for cl in clusters),
        independent_domains=len(domains),
        independent_platforms=len(platforms),
        verified_clusters=sum(1 for cl in clusters if cl.canonical.verified),
        best_face_similarity=best_sim,
    )
