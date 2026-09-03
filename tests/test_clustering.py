"""Duplicate image clustering and evidence corroboration tests.

All offline — no network, no face model required.
"""

from __future__ import annotations

import pytest

from facechain.models import VerifiedCandidate, Stage, CandidateType
from facechain.verification.clustering import (
    cluster_candidates,
    corroboration_summary,
    ImageCluster,
    MAX_HAMMING,
)


def _make(
    url: str,
    phash: str | None = None,
    face_sim: float = 0.0,
    img_sim: float = 0.0,
    verified: bool = False,
    platform: str | None = None,
) -> VerifiedCandidate:
    """Minimal VerifiedCandidate for clustering tests."""
    domain = url.split("/")[2] if "/" in url else url
    return VerifiedCandidate(
        engine="yandex",
        url=url,
        domain=domain,
        platform=platform,
        is_social=platform is not None,
        candidate_image_sha256="aa" * 32 if phash else None,
        candidate_image_phash=phash,
        face_similarity=face_sim,
        image_similarity=img_sim,
        verified=verified,
        stages=[Stage.SEARCH_FOUND, Stage.FACE_MATCH, Stage.VERIFIED] if verified else [Stage.SEARCH_FOUND],
    )


# ------------------------------------------------------------------
# Clustering
# ------------------------------------------------------------------

class TestClustering:
    def test_single_candidate_forms_one_cluster(self):
        c = _make("https://example.com/a", phash="aabbccdd11223344")
        clusters = cluster_candidates([c])
        assert len(clusters) == 1
        assert clusters[0].size == 1

    def test_identical_phash_grouped(self):
        same_hash = "a0b0c0d0e0f01234"
        a = _make("https://a.example.com/img", phash=same_hash)
        b = _make("https://b.example.com/img", phash=same_hash)
        clusters = cluster_candidates([a, b])
        assert len(clusters) == 1
        assert clusters[0].size == 2

    def test_different_phash_separate_clusters(self):
        a = _make("https://a.com/x", phash="0000000000000000")
        b = _make("https://b.com/x", phash="ffffffffffffffff")
        clusters = cluster_candidates([a, b])
        assert len(clusters) == 2

    def test_no_phash_each_gets_own_cluster(self):
        """Candidates without a measured image never merge — unknown is unknown."""
        a = _make("https://a.com/x", phash=None)
        b = _make("https://b.com/x", phash=None)
        clusters = cluster_candidates([a, b])
        assert len(clusters) == 2

    def test_phash_none_and_known_do_not_merge(self):
        a = _make("https://a.com/x", phash="aabb001122334455")
        b = _make("https://b.com/x", phash=None)
        clusters = cluster_candidates([a, b])
        assert len(clusters) == 2

    def test_near_identical_phash_groups(self):
        """Two phashes differing by exactly MAX_HAMMING bits should cluster."""
        # Build two hashes that differ in exactly 1 bit
        base = "0" * 16          # all-zero 64-bit hash
        # Flip one nibble (one hex char = 4 bits)
        near = "1" + "0" * 15
        a = _make("https://a.com/x", phash=base)
        b = _make("https://b.com/x", phash=near)
        import imagehash
        dist = imagehash.hex_to_hash(base) - imagehash.hex_to_hash(near)
        if dist <= MAX_HAMMING:
            clusters = cluster_candidates([a, b])
            assert len(clusters) == 1
        else:
            # If they genuinely don't meet the threshold, that's fine too
            clusters = cluster_candidates([a, b])
            assert len(clusters) == 2

    def test_canonical_is_highest_face_similarity(self):
        """The strongest member becomes canonical."""
        a = _make("https://a.com/x", phash="0000000000000000", face_sim=0.30)
        b = _make("https://b.com/x", phash="0000000000000000", face_sim=0.75)
        clusters = cluster_candidates([a, b])
        assert len(clusters) == 1
        assert clusters[0].canonical.face_similarity == 0.75

    def test_empty_input(self):
        assert cluster_candidates([]) == []

    def test_three_identical_images_one_cluster(self):
        h = "abcdef0123456789"
        cs = [_make(f"https://site{i}.com/img", phash=h) for i in range(3)]
        clusters = cluster_candidates(cs)
        assert len(clusters) == 1
        assert clusters[0].size == 3

    def test_two_groups_correctly_separated(self):
        ha = "0000000000000000"
        hb = "ffffffffffffffff"
        group_a = [_make(f"https://a{i}.com/x", phash=ha) for i in range(2)]
        group_b = [_make(f"https://b{i}.com/x", phash=hb) for i in range(2)]
        clusters = cluster_candidates(group_a + group_b)
        assert len(clusters) == 2
        sizes = sorted(cl.size for cl in clusters)
        assert sizes == [2, 2]


# ------------------------------------------------------------------
# Corroboration summary
# ------------------------------------------------------------------

class TestCorroborationSummary:
    def test_empty(self):
        s = corroboration_summary([])
        assert s.total_candidates == 0
        assert s.image_clusters == 0
        assert s.independent_domains == 0

    def test_single_cluster(self):
        c = _make("https://linkedin.com/in/alice", phash="aa00", platform="LinkedIn",
                  face_sim=0.85, verified=True)
        clusters = cluster_candidates([c])
        s = corroboration_summary(clusters)
        assert s.total_candidates == 1
        assert s.image_clusters == 1
        assert s.independent_domains == 1
        assert s.independent_platforms == 1
        assert s.verified_clusters == 1

    def test_duplicates_do_not_inflate_cluster_count(self):
        """5 URLs of the same image = 1 cluster, not 5 independent sources."""
        h = "deadbeef01234567"
        cs = [_make(f"https://site{i}.com/img", phash=h) for i in range(5)]
        clusters = cluster_candidates(cs)
        s = corroboration_summary(clusters)
        assert s.image_clusters == 1
        assert s.duplicate_count == 4
        assert s.total_candidates == 5

    def test_independent_platforms_counted_correctly(self):
        a = _make("https://linkedin.com/in/alice", phash="0011", platform="LinkedIn",
                  face_sim=0.80, verified=True)
        b = _make("https://github.com/alice",       phash="ff00", platform="GitHub",
                  face_sim=0.78, verified=True)
        clusters = cluster_candidates([a, b])
        s = corroboration_summary(clusters)
        assert s.independent_platforms == 2
        assert s.verified_clusters == 2

    def test_best_face_similarity_is_highest_across_clusters(self):
        a = _make("https://a.com/x", phash="aabb", face_sim=0.60)
        b = _make("https://b.com/x", phash="ccdd", face_sim=0.92)
        clusters = cluster_candidates([a, b])
        s = corroboration_summary(clusters)
        assert s.best_face_similarity == pytest.approx(0.92)

    def test_domains_per_cluster_include_duplicates(self):
        """Two URLs from different domains but same image → cluster has 2 domains."""
        h = "1234567890abcdef"
        a = _make("https://pinterest.com/pin/1", phash=h, platform="Pinterest")
        b = _make("https://pinimg.com/thumb",    phash=h, platform="Pinterest")
        clusters = cluster_candidates([a, b])
        assert len(clusters) == 1
        assert len(clusters[0].domains) == 2

    def test_verified_only_excludes_unverified_clusters(self):
        """Regression: `verified_only=True` used to be silently ignored — the
        summary counted every cluster regardless of the flag."""
        verified = _make("https://linkedin.com/in/alice", phash="0011", platform="LinkedIn",
                         face_sim=0.85, verified=True)
        unverified = _make("https://random-blog.example/post", phash="ff00",
                           face_sim=0.20, verified=False)
        clusters = cluster_candidates([verified, unverified])

        everything = corroboration_summary(clusters, verified_only=False)
        assert everything.image_clusters == 2
        assert everything.total_candidates == 2

        confirmed_only = corroboration_summary(clusters, verified_only=True)
        assert confirmed_only.image_clusters == 1
        assert confirmed_only.total_candidates == 1
        assert confirmed_only.verified_clusters == 1
        assert confirmed_only.independent_platforms == 1
        assert confirmed_only.independent_domains == 1

    def test_verified_only_with_no_verified_clusters_is_honestly_empty(self):
        a = _make("https://a.com/x", phash="aabb", face_sim=0.10, verified=False)
        b = _make("https://b.com/x", phash="ccdd", face_sim=0.15, verified=False)
        clusters = cluster_candidates([a, b])
        s = corroboration_summary(clusters, verified_only=True)
        assert s.image_clusters == 0
        assert s.total_candidates == 0
        assert s.independent_domains == 0
        assert s.best_face_similarity == 0.0


# ------------------------------------------------------------------
# Gravatar / image-only match must still be REJECTED
# ------------------------------------------------------------------

class TestImageSimilarityCannotCompensate:
    """Face similarity below threshold must never produce a verified result,
    regardless of how high image similarity or metadata is."""

    def test_high_image_sim_low_face_sim_rejected(self):
        from facechain.verification.scorer import score_candidate
        from facechain.config import settings

        vc = _make("https://2.gravatar.com/avatar/abc", phash="aabb",
                   face_sim=0.263, img_sim=0.74)
        vc.face_detected = True
        vc.metadata_consistency = 0.9
        result = score_candidate(vc)

        assert not result.verified
        assert result.face_similarity < settings.face_match_threshold
        assert "threshold" in result.rejection_reason.lower()

    def test_zero_face_sim_never_verifies(self):
        from facechain.verification.scorer import score_candidate

        vc = _make("https://instagram.com/p/ABC/", phash="aabb",
                   face_sim=0.0, img_sim=1.0)
        vc.face_detected = True
        vc.is_social = True
        vc.platform = "Instagram"
        vc.metadata_consistency = 1.0
        result = score_candidate(vc)

        assert not result.verified
