"""Index-time data model for the known-person database.

Distinct from `facechain.models` (the API/evidence-facing `IdentityResult`,
`SocialAccount`, etc.): these types describe what `scripts/build_identity_index.py`
writes to `identity_index/manifest.json` and what `index.py` loads back — the
offline artifact, not the per-scan result.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

# Categories the mission asks the index to support (section 2). Advisory,
# not enforced by an Enum — a builder run against a growing dataset should
# never fail because a real person's Wikidata occupation maps to a category
# spelling this list didn't anticipate.
KNOWN_CATEGORIES: tuple[str, ...] = (
    "technology", "business", "politics", "science", "sports",
    "entertainment", "actors", "musicians", "creators", "influencers",
    "academics", "authors", "founders", "public_figures",
)


class KnownPersonReference(BaseModel):
    """One reference image for a known person, already embedded.

    The embedding vector itself lives at `embedding_index` in the sibling
    `embeddings.npy` array — kept out of this JSON manifest so the manifest
    stays small and human-diffable, and the (N, D) float32 array stays one
    contiguous numpy load (see `index.py::KnownPersonIndex.load`).
    """

    embedding_index: int
    source_url: str  # provenance — e.g. a Wikimedia Commons file page
    license: str = ""
    width: int = 0
    height: int = 0
    det_score: float = 0.0


class KnownPerson(BaseModel):
    """One entry in the known-person database (mission section 2)."""

    person_id: str
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    occupation: str = ""
    category: str = ""
    country: str = ""
    description: str = ""
    official_website: str = ""
    # platform name -> username/handle (or a URL when the platform has no
    # simple handle form) — sourced from structured, licensed data
    # (Wikidata properties) at build time, not scraped per-request.
    official_social_accounts: dict[str, str] = Field(default_factory=dict)
    # Provenance for the record itself (e.g. the Wikidata QID / Wikipedia URL
    # this entry was built from) — distinct from each reference image's own
    # `source_url`.
    source_urls: list[str] = Field(default_factory=list)
    references: list[KnownPersonReference] = Field(default_factory=list)


class KnownPersonIndexManifest(BaseModel):
    """The full offline-built artifact `identity_index/manifest.json` holds."""

    version: str
    built_at: str
    embedding_dim: int
    embedding_model: str = ""  # e.g. "buffalo_l/SCRFD+ArcFace" — must match
                               # the runtime face backend or scores are meaningless
    people: list[KnownPerson] = Field(default_factory=list)
