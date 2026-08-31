"""Typed data model for the whole pipeline (and for the evidence bundle)."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field

from . import PIPELINE_VERSION
from .evidence.hashing import q3


class Stage(str, Enum):
    """The verification ladder. Each rung must be earned by real evidence."""

    SEARCH_FOUND = "SEARCH_FOUND"
    SOCIAL_MATCH = "SOCIAL_MATCH"
    IMAGE_MATCH = "IMAGE_MATCH"
    FACE_MATCH = "FACE_MATCH"
    VERIFIED = "VERIFIED"


LADDER: list[Stage] = [
    Stage.SEARCH_FOUND,
    Stage.SOCIAL_MATCH,
    Stage.IMAGE_MATCH,
    Stage.FACE_MATCH,
    Stage.VERIFIED,
]


class InputImage(BaseModel):
    path: str
    filename: str
    bytes_len: int
    width: int
    height: int
    sha256: str
    phash: str


class FaceRecord(BaseModel):
    detected: bool
    backend: str
    model: str
    faces_found: int = 0
    bbox: Optional[list[int]] = None
    det_score: Optional[float] = None
    embedding_dimension: Optional[int] = None
    embedding_sha256: Optional[str] = None


class SearchCandidate(BaseModel):
    """One raw hit from a reverse-image engine, before any verification."""

    engine: str
    url: str
    domain: str
    title: str = ""
    thumbnail: str = ""
    is_social: bool = False
    platform: Optional[str] = None


class SearchReport(BaseModel):
    engines_attempted: list[str] = Field(default_factory=list)
    engines_succeeded: list[str] = Field(default_factory=list)
    engine_errors: dict[str, str] = Field(default_factory=dict)
    query_mode: dict[str, str] = Field(default_factory=dict)
    total_candidates: int = 0
    social_candidates: int = 0
    candidates: list[SearchCandidate] = Field(default_factory=list)


class VerifiedCandidate(BaseModel):
    """A candidate after we fetched it and compared it to the input ourselves."""

    engine: str
    url: str
    domain: str
    platform: Optional[str] = None
    is_social: bool = False

    fetched: bool = False
    fetch_note: str = ""
    candidate_image_url: Optional[str] = None
    candidate_image_source: str = ""  # og:image | twitter:image | json-ld | img | engine-thumbnail
    candidate_image_sha256: Optional[str] = None
    candidate_image_phash: Optional[str] = None
    candidate_faces_found: int = 0

    image_similarity: float = 0.0
    face_detected: bool = False
    face_similarity: float = 0.0
    metadata_consistency: float = 0.0

    stages: list[Stage] = Field(default_factory=list)
    final_score: float = 0.0
    verified: bool = False

    def rounded(self) -> "VerifiedCandidate":
        c = self.model_copy()
        c.image_similarity = q3(c.image_similarity)
        c.face_similarity = q3(c.face_similarity)
        c.metadata_consistency = q3(c.metadata_consistency)
        c.final_score = q3(c.final_score)
        return c


class ChainRecord(BaseModel):
    network: str = "Base Sepolia"
    chain_id: int = 84532
    eas_contract: str = ""
    schema_uid: str = ""
    schema_definition: str = ""
    attester: str = ""
    recipient: str = ""
    tx_hash: Optional[str] = None
    block_number: Optional[int] = None
    gas_used: Optional[int] = None
    attestation_uid: Optional[str] = None
    explorer_tx: Optional[str] = None
    explorer_attestation: Optional[str] = None
    mode: str = "onchain"  # onchain | simulate | skipped
    readback_verified: bool = False
    readback_mismatches: list[str] = Field(default_factory=list)
    onchain_decoded: dict[str, Any] = Field(default_factory=dict)
    note: str = ""


class AttestedPayload(BaseModel):
    """Exactly the fields that get hashed into `evidenceHash` and attested.

    This object *is* the tamper-evident record: its canonical JSON hashes to
    `evidenceHash`, and its individual hashes are written into the attestation
    field by field, so a verifier can catch tampering either way.
    """

    case_id: str
    pipeline_version: str = PIPELINE_VERSION
    observed_at: int
    input_image_sha256: str
    input_image_phash: str
    face_embedding_sha256: str
    face_bbox: list[int]
    matched_url: str
    matched_url_sha256: str
    matched_image_sha256: str
    matched_image_phash: str
    search_engine: str
    social_platform: str
    image_similarity: float
    face_similarity: float
    match_score: float
    stages_passed: list[str]


class Case(BaseModel):
    case_id: str
    pipeline_version: str = PIPELINE_VERSION
    created_at: str
    observed_at: int
    input: Optional[InputImage] = None
    face: Optional[FaceRecord] = None
    reverse_search: Optional[SearchReport] = None
    verification: list[VerifiedCandidate] = Field(default_factory=list)
    best_match: Optional[VerifiedCandidate] = None
    stages_passed: list[Stage] = Field(default_factory=list)
    evidence_sha256: Optional[str] = None
    blockchain: Optional[ChainRecord] = None
    verdict: str = "INCOMPLETE"
    failure_reason: Optional[str] = None
