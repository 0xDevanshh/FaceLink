"""Data models for enriched profile graphs.

Deliberately separate from models.py so the blockchain/evidence contracts stay
frozen.  Everything here is purely additive.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class EvidenceLevel(str, Enum):
    """How confident are we that this profile belongs to the same person?

    Multiple independent signals are required for CONFIRMED; a single signal
    (even a face match) only reaches HIGH_CONFIDENCE without corroboration.
    Name alone never reaches above POSSIBLE.
    """
    CONFIRMED = "CONFIRMED"           # face + ≥1 independent corroborating signal
    HIGH_CONFIDENCE = "HIGH_CONFIDENCE"  # face match, no independent corroboration
    POSSIBLE = "POSSIBLE"             # non-face signals only (name, handle, link)
    REJECTED = "REJECTED"             # explicitly does not match


class ProfileField(BaseModel):
    """One piece of extracted profile data with full provenance."""
    value: str
    source_url: str           # page it was extracted from
    extraction_method: str    # og:image | twitter:image | json-ld | html-meta | text | link
    confidence: float = 1.0   # 0-1, lower for inferred/heuristic values


class DiscoveredProfile(BaseModel):
    """A single public profile discovered for the subject.

    Every field carries its own provenance so a human reviewer can audit
    exactly which page provided which claim.
    """
    profile_id: str                       # stable slug: platform:handle
    platform: str
    canonical_url: str

    # ---- identity fields (with provenance) --------------------------------
    username: Optional[ProfileField] = None
    display_name: Optional[ProfileField] = None
    bio: Optional[ProfileField] = None
    avatar_url: Optional[ProfileField] = None  # full-res, not thumbnail

    # ---- linked profiles found on this page --------------------------------
    linked_profiles: list[ProfileField] = Field(default_factory=list)
    # Recognised platform names of linked profiles
    linked_platforms: list[str] = Field(default_factory=list)

    # ---- fetch / extraction state -----------------------------------------
    fetched: bool = False
    fetch_note: str = ""
    fetch_status: int = 0        # HTTP status code (0 = never attempted)

    # ---- verification signals ---------------------------------------------
    face_similarity: float = 0.0
    image_similarity: float = 0.0
    face_detected: bool = False
    candidate_image_url: Optional[str] = None
    candidate_image_source: str = ""

    # ---- provenance -------------------------------------------------------
    discovery_method: str = ""   # reverse-image | cross-profile-link | direct
    discovered_from: str = ""    # URL that led us here (empty for seed profiles)
    evidence_signals: list[str] = Field(default_factory=list)

    # ---- verdict ----------------------------------------------------------
    evidence_level: EvidenceLevel = EvidenceLevel.POSSIBLE
    rejection_reason: str = ""

    # Raw structured data extracted from the page (JSON-LD, OpenGraph, etc.)
    # Stored for auditing; never used as authoritative without measurement.
    raw_metadata: dict[str, Any] = Field(default_factory=dict)


class ProfileGraphEdge(BaseModel):
    source_id: str
    target_id: str
    relationship: str   # linked-from | same-avatar | same-face | shared-username
    note: str = ""


class ProfileGraph(BaseModel):
    """The full enrichment result: all discovered profiles + their relationships."""

    profiles: list[DiscoveredProfile] = Field(default_factory=list)
    edges: list[ProfileGraphEdge] = Field(default_factory=list)

    # Summary statistics
    confirmed_count: int = 0
    high_confidence_count: int = 0
    possible_count: int = 0
    rejected_count: int = 0

    def profile(self, profile_id: str) -> Optional[DiscoveredProfile]:
        return next((p for p in self.profiles if p.profile_id == profile_id), None)

    def update_counts(self) -> None:
        self.confirmed_count = sum(
            1 for p in self.profiles if p.evidence_level == EvidenceLevel.CONFIRMED)
        self.high_confidence_count = sum(
            1 for p in self.profiles if p.evidence_level == EvidenceLevel.HIGH_CONFIDENCE)
        self.possible_count = sum(
            1 for p in self.profiles if p.evidence_level == EvidenceLevel.POSSIBLE)
        self.rejected_count = sum(
            1 for p in self.profiles if p.evidence_level == EvidenceLevel.REJECTED)
