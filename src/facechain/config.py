"""Central configuration. Everything overridable by env / .env / CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Ethereum Sepolia + EAS constants
# ---------------------------------------------------------------------------
SEPOLIA_CHAIN_ID = 11155111
EAS_CONTRACT = "0x0000000000000000000000000000000000000021"
SCHEMA_REGISTRY_CONTRACT = "0x0000000000000000000000000000000000000020"
EXPLORER_TX = "https://sepolia.etherscan.io/tx/{tx}"
EASSCAN_ATTESTATION = "https://sepolia.easscan.org/attestation/view/{uid}"
EASSCAN_SCHEMA = "https://sepolia.easscan.org/schema/view/{uid}"

FALLBACK_RPCS = (
    "https://ethereum-sepolia-rpc.publicnode.com",
    "https://rpc.sepolia.org",
    "https://sepolia.gateway.tenderly.co",
)

# The on-chain schema. Order matters: it is part of the schema UID.
EAS_SCHEMA_DEFINITION = (
    "bytes32 caseId,"
    "bytes32 inputImageHash,"
    "bytes32 faceEmbeddingHash,"
    "bytes32 matchedImageHash,"
    "bytes32 matchedUrlHash,"
    "bytes32 evidenceHash,"
    "string searchEngine,"
    "string socialPlatform,"
    "uint16 matchScoreBps,"
    "uint64 observedAt,"
    "string pipelineVersion"
)

# Social platforms we accept as a "social media post".
SOCIAL_DOMAINS: dict[str, str] = {
    "instagram.com": "Instagram",
    "facebook.com": "Facebook",
    "fb.com": "Facebook",
    "twitter.com": "X/Twitter",
    "x.com": "X/Twitter",
    "tiktok.com": "TikTok",
    "threads.net": "Threads",
    "threads.com": "Threads",
    "linkedin.com": "LinkedIn",
    "reddit.com": "Reddit",
    "vk.com": "VK",
    "weibo.com": "Weibo",
    "pinterest.com": "Pinterest",
    "tumblr.com": "Tumblr",
    "mastodon.social": "Mastodon",
    "bsky.app": "Bluesky",
    "youtube.com": "YouTube",
    "flickr.com": "Flickr",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- blockchain -------------------------------------------------------
    base_sepolia_rpc_url: str = "https://ethereum-sepolia-rpc.publicnode.com"
    private_key: str = ""
    eas_schema_uid: str = ""
    attestation_recipient: str = "0x0000000000000000000000000000000000000000"
    tx_confirm_timeout_s: int = 180

    # ---- face -------------------------------------------------------------
    face_backend: Literal["auto", "insightface", "opencv"] = "auto"
    insightface_model: str = "buffalo_l"
    face_det_size: int = 640
    face_det_threshold: float = 0.5
    # Cosine similarity on L2-normalised ArcFace embeddings.
    face_match_threshold: float = 0.38

    # ---- reverse image search --------------------------------------------
    engines: str = "yandex,bing,google_lens"
    headless: bool = True
    search_timeout_s: int = 60
    max_candidates_per_engine: int = 60
    # Optional: turn a local file into a public URL so engines' by-URL search
    # endpoints can be used (far more reliable than their upload flows).
    # OFF by default: it uploads your photo to a third-party host.
    allow_upload_host: bool = False
    upload_host: str = "https://litterbox.catbox.moe/resources/internals/api.php"
    # Optional genuine reverse-image-search APIs (used only if key present).
    serpapi_key: str = ""

    # ---- verification ----------------------------------------------------
    image_match_threshold: float = 0.80
    weight_face: float = 0.50
    weight_image: float = 0.40
    weight_meta: float = 0.10
    verify_min_score: float = 0.70
    max_candidates_to_verify: int = 12
    http_timeout_s: int = 25
    user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    )

    # ---- output ----------------------------------------------------------
    evidence_dir: Path = Field(default=REPO_ROOT / "evidence")

    @property
    def engine_list(self) -> list[str]:
        return [e.strip() for e in self.engines.split(",") if e.strip()]


settings = Settings()
