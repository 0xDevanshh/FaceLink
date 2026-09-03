"""Evidence graph: nodes/edges over image clusters, and independent-source counting.

All offline — builds `ImageCluster`s directly, the same objects
`verification/clustering.py` produces, without any network or model.
"""

from __future__ import annotations

from facechain.models import Stage, VerifiedCandidate
from facechain.verification.clustering import cluster_candidates
from facechain.verification.evidence_graph import build_evidence_graph


def _vc(url: str, phash: str, platform: str | None = None, face_sim: float = 0.0,
       verified: bool = False) -> VerifiedCandidate:
    domain = url.split("/")[2]
    return VerifiedCandidate(
        engine="yandex", url=url, domain=domain, platform=platform,
        is_social=platform is not None, candidate_image_phash=phash,
        face_similarity=face_sim, verified=verified,
        stages=[Stage.SEARCH_FOUND, Stage.FACE_MATCH, Stage.VERIFIED] if verified else [Stage.SEARCH_FOUND],
    )


def test_a_single_candidate_produces_one_image_node_and_a_same_image_edge():
    clusters = cluster_candidates([_vc("https://a.com/x", "0000000000000000")])
    graph = build_evidence_graph(clusters)

    image_nodes = [n for n in graph.nodes if n.type == "image"]
    candidate_nodes = [n for n in graph.nodes if n.type == "candidate"]
    assert len(image_nodes) == 1
    assert len(candidate_nodes) == 1
    same_image_edges = [e for e in graph.edges if e.type == "same_image"]
    assert len(same_image_edges) == 1
    assert same_image_edges[0].source == candidate_nodes[0].id
    assert same_image_edges[0].target == image_nodes[0].id


def test_reposts_of_the_same_image_share_one_image_node_not_many():
    """Five URLs of the same photo -> one image node, five same_image edges
    into it, never edges between the candidates themselves."""
    h = "deadbeef01234567"
    cs = [_vc(f"https://site{i}.com/img", h) for i in range(5)]
    clusters = cluster_candidates(cs)
    graph = build_evidence_graph(clusters)

    assert len([n for n in graph.nodes if n.type == "image"]) == 1
    assert len([n for n in graph.nodes if n.type == "candidate"]) == 5
    assert len([e for e in graph.edges if e.type == "same_image"]) == 5
    # No candidate-to-candidate edges of any kind.
    candidate_ids = {n.id for n in graph.nodes if n.type == "candidate"}
    assert not any(e.source in candidate_ids and e.target in candidate_ids for e in graph.edges)


def test_two_verified_clusters_on_different_domains_are_independent_sources():
    a = _vc("https://linkedin.com/in/alice", "0011", platform="LinkedIn",
           face_sim=0.85, verified=True)
    b = _vc("https://github.com/alice", "ff00", platform="GitHub",
           face_sim=0.80, verified=True)
    clusters = cluster_candidates([a, b])
    graph = build_evidence_graph(clusters)

    independent_edges = [e for e in graph.edges if e.type == "independent_source"]
    assert len(independent_edges) == 1
    assert graph.independent_evidence_count == 2


def test_two_verified_clusters_on_the_same_domain_are_not_independent():
    """Two different photos posted by the same account support one claim,
    not two — no independent_source edge, and it doesn't inflate the count."""
    a = _vc("https://linkedin.com/in/alice/photo1", "0011", platform="LinkedIn",
           face_sim=0.85, verified=True)
    b = _vc("https://linkedin.com/in/alice/photo2", "ff00", platform="LinkedIn",
           face_sim=0.80, verified=True)
    clusters = cluster_candidates([a, b])
    graph = build_evidence_graph(clusters)

    independent_edges = [e for e in graph.edges if e.type == "independent_source"]
    assert independent_edges == []
    assert graph.independent_evidence_count == 1


def test_unverified_clusters_never_produce_independent_source_or_same_face_edges():
    a = _vc("https://a.com/x", "0011", face_sim=0.20, verified=False)
    b = _vc("https://b.com/x", "ff00", face_sim=0.15, verified=False)
    clusters = cluster_candidates([a, b])
    graph = build_evidence_graph(clusters)

    assert not any(e.type in ("independent_source", "same_face") for e in graph.edges)
    assert graph.independent_evidence_count == 0


def test_same_face_edge_between_verified_clusters_is_labelled_as_an_approximation():
    a = _vc("https://linkedin.com/in/alice", "0011", platform="LinkedIn",
           face_sim=0.85, verified=True)
    b = _vc("https://github.com/alice", "ff00", platform="GitHub",
           face_sim=0.80, verified=True)
    clusters = cluster_candidates([a, b])
    graph = build_evidence_graph(clusters)

    same_face_edges = [e for e in graph.edges if e.type == "same_face"]
    assert len(same_face_edges) == 1
    assert "not a direct candidate-to-candidate comparison" in same_face_edges[0].note


def test_domain_and_platform_nodes_are_deduplicated_across_clusters():
    """Two different images, both on github.com, must not create two
    'domain:github.com' nodes."""
    a = _vc("https://github.com/alice", "0011", platform="GitHub", verified=True, face_sim=0.9)
    b = _vc("https://github.com/bob", "ff00", platform="GitHub", verified=True, face_sim=0.9)
    clusters = cluster_candidates([a, b])
    graph = build_evidence_graph(clusters)

    domain_nodes = [n for n in graph.nodes if n.type == "domain"]
    platform_nodes = [n for n in graph.nodes if n.type == "platform"]
    assert len(domain_nodes) == 1
    assert len(platform_nodes) == 1


def test_empty_clusters_produce_an_empty_graph():
    graph = build_evidence_graph([])
    assert graph.nodes == []
    assert graph.edges == []
    assert graph.independent_evidence_count == 0
