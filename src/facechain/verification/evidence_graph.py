"""An explicit evidence graph: what corroborates what, and why.

`clustering.py` already answers "how many distinct images are there" as
aggregate counts. This module answers the forensic question underneath that —
*which* candidates, images, domains and platforms relate to each other, and by
which specific relationship — so a reader (or the frontend) can show the
actual graph a verdict rests on instead of a handful of summary numbers.

Two rules carried over from `clustering.py`, restated here because they matter
even more once the relationships are explicit:

1. **Same image is not independent evidence.** Every candidate in an image
   cluster shares one `same_image` edge to that cluster's node — never to each
   other pairwise — so a widely-reposted photo cannot look like many
   corroborating links just because it has many URLs.

2. **Same domain is not independent evidence either.** Two different photos
   posted by the same account, or two pages on the same site, support the same
   underlying claim rather than two separate ones. `independent_source` edges
   only connect verified clusters that share *no* domain.

`same_face` is a named exception worth being honest about: it does not mean
two candidate images were compared face-to-face (their raw embeddings are
never retained — see `evidence/writer.py`'s hash-not-vector design). It means
both candidates *independently* matched the query face above threshold. That
is real corroboration, just not a literal pairwise face comparison, and the
edge's `note` field says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..models import VerifiedCandidate
from .clustering import ImageCluster, corroboration_summary

NodeType = str   # "candidate" | "image" | "domain" | "platform"
EdgeType = str   # "same_image" | "same_domain" | "same_platform" | "same_face" | "independent_source"


@dataclass
class GraphNode:
    id: str
    type: NodeType
    label: str


@dataclass
class GraphEdge:
    source: str
    target: str
    type: EdgeType
    note: str = ""


@dataclass
class EvidenceGraph:
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    # Verified image clusters that share no domain with any other verified
    # cluster — see the module docstring. This is the number that should back
    # a "N independent sources" claim, not the raw candidate or cluster count.
    independent_evidence_count: int = 0


def _domain_id(domain: str) -> str:
    return f"domain:{domain}"


def _platform_id(platform: str) -> str:
    return f"platform:{platform}"


def _image_id(index: int) -> str:
    return f"image:{index}"


def _candidate_id(vc: VerifiedCandidate) -> str:
    return f"candidate:{vc.canonical_url or vc.url}"


def build_evidence_graph(clusters: list[ImageCluster]) -> EvidenceGraph:
    """Build the node/edge graph from already-computed image clusters."""
    graph = EvidenceGraph()
    seen_domains: set[str] = set()
    seen_platforms: set[str] = set()

    for i, cluster in enumerate(clusters):
        image_node = _image_id(i)
        graph.nodes.append(GraphNode(
            id=image_node, type="image",
            label=f"image cluster ({cluster.size} URL(s))",
        ))

        members = [cluster.canonical, *cluster.duplicates]
        for vc in members:
            cand_node = _candidate_id(vc)
            graph.nodes.append(GraphNode(id=cand_node, type="candidate", label=vc.url))
            graph.edges.append(GraphEdge(source=cand_node, target=image_node, type="same_image"))

            if vc.domain and vc.domain not in seen_domains:
                seen_domains.add(vc.domain)
                graph.nodes.append(GraphNode(id=_domain_id(vc.domain), type="domain", label=vc.domain))
            if vc.domain:
                graph.edges.append(GraphEdge(
                    source=cand_node, target=_domain_id(vc.domain), type="same_domain"))

            if vc.platform and vc.platform not in seen_platforms:
                seen_platforms.add(vc.platform)
                graph.nodes.append(GraphNode(
                    id=_platform_id(vc.platform), type="platform", label=vc.platform))
            if vc.platform:
                graph.edges.append(GraphEdge(
                    source=cand_node, target=_platform_id(vc.platform), type="same_platform"))

    # "Independent source" and "same face" only make sense among *verified*
    # clusters — an unverified hit is not evidence of anything yet.
    verified_clusters = [(i, cl) for i, cl in enumerate(clusters) if cl.canonical.verified]
    for a_idx, (i, cl_a) in enumerate(verified_clusters):
        for j, cl_b in verified_clusters[a_idx + 1:]:
            domains_a, domains_b = cl_a.domains, cl_b.domains
            graph.edges.append(GraphEdge(
                source=_image_id(i), target=_image_id(j), type="same_face",
                note="both independently matched the query face above threshold "
                     "(not a direct candidate-to-candidate comparison)",
            ))
            if domains_a.isdisjoint(domains_b):
                graph.edges.append(GraphEdge(
                    source=_image_id(i), target=_image_id(j), type="independent_source",
                ))

    graph.independent_evidence_count = corroboration_summary(
        clusters, verified_only=True
    ).independent_domains

    return graph
