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


class ProviderStatus(str, Enum):
    """Explicit lifecycle for one search provider.

    Every provider ends in exactly one terminal state, and the state is
    recorded in the evidence bundle. "Failed" is never collapsed into "no
    results": a CAPTCHA, a timeout and a genuinely empty result set are
    different facts about the search, and a reader has to be able to tell them
    apart.
    """

    NOT_CONFIGURED = "NOT_CONFIGURED"   # no API key / prerequisite missing
    READY = "READY"                     # selected, not started yet
    STARTED = "STARTED"                 # running
    COMPLETED = "COMPLETED"             # returned candidates
    NO_RESULTS = "NO_RESULTS"           # ran fine, found nothing
    TIMEOUT = "TIMEOUT"                 # exceeded its wall-clock budget
    CHALLENGED = "CHALLENGED"           # CAPTCHA / bot interstitial
    RATE_LIMITED = "RATE_LIMITED"       # throttled by the provider
    FAILED = "FAILED"                   # error, layout change, parse failure
    CANCELLED = "CANCELLED"             # search budget exhausted before it ran

    @property
    def terminal(self) -> bool:
        return self not in (ProviderStatus.READY, ProviderStatus.STARTED)

    @property
    def produced_results(self) -> bool:
        return self is ProviderStatus.COMPLETED


class CandidateType(str, Enum):
    """What kind of evidence a candidate is, once we have measured it.

    This is orthogonal to *whether* it verified: it describes the shape of the
    match so a reader can tell "the same picture turned up on X" from "a
    different photo of the same face is on a LinkedIn profile".
    """

    EXACT_IMAGE = "EXACT_IMAGE"              # perceptually the same picture
    SAME_FACE = "SAME_FACE"                  # different picture, same face
    SOCIAL_PROFILE = "SOCIAL_PROFILE"        # a profile page on a social platform
    SOCIAL_POST = "SOCIAL_POST"              # a specific post
    DEVELOPER_PROFILE = "DEVELOPER_PROFILE"  # GitHub-style developer identity page
    PUBLIC_ARTICLE = "PUBLIC_ARTICLE"        # news / blog article
    PUBLIC_WEB_PAGE = "PUBLIC_WEB_PAGE"      # any other indexed public page
    OTHER = "OTHER"


class ProviderReport(BaseModel):
    """One provider's outcome — the auditable record of the search stage."""

    engine: str
    status: ProviderStatus = ProviderStatus.READY
    candidates: int = 0
    duration_s: float = 0.0
    query_mode: str = ""  # upload | by-url | api
    error: str = ""

    def rounded(self) -> "ProviderReport":
        p = self.model_copy()
        p.duration_s = q3(p.duration_s)
        return p


class InputImage(BaseModel):
    path: str
    filename: str
    bytes_len: int
    width: int
    height: int
    sha256: str
    phash: str


class DetectedFaceInfo(BaseModel):
    """One detected face, as offered to the operator for selection.

    Deliberately carries no embedding: the UI needs geometry and quality to let
    someone pick a face, and the embedding never leaves the backend.
    """

    index: int
    bbox: list[int]  # x1, y1, x2, y2 in the coordinates of the returned image
    det_score: float
    face_px: int          # shorter side of the box — the size the gate checks
    area: int
    usable: bool = True   # passes the per-face size/confidence floor
    note: str = ""


class FaceQuality(BaseModel):
    """The quality gate's verdict, recorded whether it passed or failed."""

    passed: bool
    error: Optional[str] = None   # QualityError code when it failed
    detail: str = ""
    blur_score: float = 0.0
    face_px: int = 0
    face_count: int = 0

    def rounded(self) -> "FaceQuality":
        q = self.model_copy()
        q.blur_score = q3(q.blur_score)
        return q


class FaceSelection(BaseModel):
    """Which face was chosen, how, and what the original looked like.

    This is evidence, not UI state. A manual crop must never be able to
    masquerade as the original image, so the original's hash and dimensions are
    recorded alongside the crop's, and both survive into the bundle.
    """

    mode: str = "auto"  # auto | manual-face | manual-crop
    face_index: Optional[int] = None
    faces_offered: int = 0
    bbox: Optional[list[int]] = None          # the selected face box
    crop_rect: Optional[list[int]] = None     # x, y, w, h actually cropped, if any
    crop_sha256: Optional[str] = None         # hash of the cropped PNG bytes
    original_sha256: str = ""                 # hash of the untouched upload
    original_width: int = 0
    original_height: int = 0
    selected_at: str = ""                     # ISO-8601 UTC


class FaceRecord(BaseModel):
    detected: bool
    backend: str
    model: str
    faces_found: int = 0
    bbox: Optional[list[int]] = None
    det_score: Optional[float] = None
    embedding_dimension: Optional[int] = None
    embedding_sha256: Optional[str] = None
    faces: list[DetectedFaceInfo] = Field(default_factory=list)
    quality: Optional[FaceQuality] = None


class SearchCandidate(BaseModel):
    """One raw hit from a reverse-image engine, before any verification."""

    engine: str
    url: str
    domain: str
    title: str = ""
    thumbnail: str = ""
    is_social: bool = False
    platform: Optional[str] = None
    # Discovery rank of `platform` — ordering only, never truth.
    platform_priority: int = 90
    # Provisional type from the URL alone; upgraded after measurement.
    candidate_type: CandidateType = CandidateType.PUBLIC_WEB_PAGE


class SearchReport(BaseModel):
    engines_attempted: list[str] = Field(default_factory=list)
    engines_succeeded: list[str] = Field(default_factory=list)
    engine_errors: dict[str, str] = Field(default_factory=dict)
    query_mode: dict[str, str] = Field(default_factory=dict)
    # Explicit per-provider lifecycle. `engines_succeeded`/`engine_errors` are
    # kept for compatibility with existing evidence bundles and the CLI.
    providers: list[ProviderReport] = Field(default_factory=list)
    total_candidates: int = 0
    social_candidates: int = 0
    # Candidate count per platform name, plus "Other Web" for the unrecognised
    # public web. Reported as-is, including zeros for platforms we looked for.
    platform_counts: dict[str, int] = Field(default_factory=dict)
    timed_out: bool = False
    candidates: list[SearchCandidate] = Field(default_factory=list)

    def provider(self, engine: str) -> Optional[ProviderReport]:
        return next((p for p in self.providers if p.engine == engine), None)


class VerifiedCandidate(BaseModel):
    """A candidate after we fetched it and compared it to the input ourselves."""

    engine: str
    url: str
    domain: str
    platform: Optional[str] = None
    is_social: bool = False
    # The tracking-stripped form whose hash goes on-chain. Equal to `url` for
    # candidates that arrived already canonical.
    canonical_url: str = ""
    platform_priority: int = 90
    candidate_type: CandidateType = CandidateType.PUBLIC_WEB_PAGE

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
    confidence_band: str = "INSUFFICIENT"  # STRONG | MODERATE | WEAK | INSUFFICIENT

    stages: list[Stage] = Field(default_factory=list)
    match_type: str = "none"  # exact-image | face-only | none
    final_score: float = 0.0
    verified: bool = False
    rejection_reason: str = ""  # populated when not verified

    def rounded(self) -> "VerifiedCandidate":
        c = self.model_copy()
        c.image_similarity = q3(c.image_similarity)
        c.face_similarity = q3(c.face_similarity)
        c.metadata_consistency = q3(c.metadata_consistency)
        c.final_score = q3(c.final_score)
        return c


class ChainRecord(BaseModel):
    # Always set from the live client, never assumed: a record must not
    # claim a chain it was not written to.
    network: str = ""
    chain_id: int = 0
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
    match_type: str
    stages_passed: list[str]

    # ---- fields anchored off-chain -------------------------------------
    # The on-chain schema is unchanged; these live in the evidence document and
    # are covered by `evidenceHash`, which IS on-chain. That is the documented
    # way to extend the record without re-registering a schema.
    candidate_type: str = CandidateType.OTHER.value
    confidence_band: str = "INSUFFICIENT"
    verification_rung: str = ""            # highest ladder rung this match reached
    face_selection_mode: str = "auto"
    face_crop_sha256: str = ""             # "" when no crop was taken
    provider_summary: dict[str, str] = Field(default_factory=dict)  # engine -> status


class Case(BaseModel):
    case_id: str
    pipeline_version: str = PIPELINE_VERSION
    created_at: str
    observed_at: int
    input: Optional[InputImage] = None
    face: Optional[FaceRecord] = None
    face_selection: Optional[FaceSelection] = None
    reverse_search: Optional[SearchReport] = None
    verification: list[VerifiedCandidate] = Field(default_factory=list)
    best_match: Optional[VerifiedCandidate] = None
    stages_passed: list[Stage] = Field(default_factory=list)
    evidence_sha256: Optional[str] = None
    blockchain: Optional[ChainRecord] = None
    verdict: str = "INCOMPLETE"
    failure_reason: Optional[str] = None
