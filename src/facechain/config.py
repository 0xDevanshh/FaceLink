"""Central configuration. Everything overridable by env / .env / CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Confidence bands for face similarity scores (plain-language labels).
CONFIDENCE_BANDS = [
    (0.85, "STRONG"),
    (0.70, "MODERATE"),
    (0.50, "WEAK"),
    (0.0, "INSUFFICIENT"),
]


def confidence_band(score: float) -> str:
    for threshold, label in CONFIDENCE_BANDS:
        if score >= threshold:
            return label
    return "INSUFFICIENT"

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Supported testnets.
#
# EAS lives at DIFFERENT addresses per chain: on Base it is an OP-Stack
# predeploy (0x4200…0021), on Ethereum Sepolia it is a normal deployment
# (0xC2679f…). Getting these wrong is silent and expensive — an attestation
# sent to an address with no code burns gas and records nothing — so
# `scripts/check_network.py` verifies the bytecode is actually there.
#
# Mainnets are deliberately absent: this project is testnet-only by design.
# ---------------------------------------------------------------------------
NETWORKS: dict[str, dict] = {
    "ethereum-sepolia": {
        "name": "Ethereum Sepolia",
        "chain_id": 11155111,
        "eas": "0xC2679fBD37d54388Ce493F1DB75320D236e1815e",
        "schema_registry": "0x0a7E2Ff54e76B8E6659aedc9103FB21c038050D0",
        "explorer_tx": "https://sepolia.etherscan.io/tx/{tx}",
        "easscan_attestation": "https://sepolia.easscan.org/attestation/view/{uid}",
        "easscan_schema": "https://sepolia.easscan.org/schema/view/{uid}",
        "rpcs": (
            "https://ethereum-sepolia-rpc.publicnode.com",
            "https://rpc.sepolia.org",
            "https://sepolia.gateway.tenderly.co",
        ),
        "faucets": (
            "https://cloud.google.com/application/web3/faucet/ethereum/sepolia",
            "https://www.alchemy.com/faucets/ethereum-sepolia",
        ),
    },
    "base-sepolia": {
        "name": "Base Sepolia",
        "chain_id": 84532,
        "eas": "0x4200000000000000000000000000000000000021",
        "schema_registry": "0x4200000000000000000000000000000000000020",
        "explorer_tx": "https://sepolia.basescan.org/tx/{tx}",
        "easscan_attestation": "https://base-sepolia.easscan.org/attestation/view/{uid}",
        "easscan_schema": "https://base-sepolia.easscan.org/schema/view/{uid}",
        "rpcs": (
            "https://sepolia.base.org",
            "https://base-sepolia-rpc.publicnode.com",
            "https://base-sepolia.drpc.org",
        ),
        "faucets": (
            "https://portal.cdp.coinbase.com/products/faucet",
            "https://www.alchemy.com/faucets/base-sepolia",
        ),
    },
}

DEFAULT_NETWORK = "ethereum-sepolia"

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

# Platforms we recognise by exact hostname (or a genuine subdomain of one).
#
# "social" here means "a platform we can name", which is why GitHub is in the
# table: it is a first-class discovery target for this pipeline, and naming the
# platform is what lets the ranker prioritise it and the evidence record where a
# match came from. Recognition is never evidence of identity — every candidate
# still has to pass the face-verification ladder.
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
    "youtu.be": "YouTube",
    "flickr.com": "Flickr",
    # GitHub is a primary discovery target: profiles, avatars and org/contributor
    # pages are public and frequently indexed by reverse-image engines. The
    # user-content hosts are where the actual avatar/image bytes live, so they
    # must classify to the same platform as the page that referenced them.
    "github.com": "GitHub",
    "githubusercontent.com": "GitHub",
    "github.io": "GitHub",
    # Professional/developer platforms `enrichment/extractor.py` already
    # recognises for profile enrichment. Listed here too so a reverse-image
    # hit that lands directly on one of these domains is attributed to its
    # platform at discovery time rather than filed as unrecognised "Other Web"
    # — the two layers must agree on what these hosts are.
    "medium.com": "Medium",
    "devfolio.co": "Devfolio",
    "leetcode.com": "LeetCode",
    "hackerrank.com": "HackerRank",
    "kaggle.com": "Kaggle",
    "stackoverflow.com": "StackOverflow",
}

# ---------------------------------------------------------------------------
# Platform-owned media CDNs.
#
# These are a *separate registrable domain* from the platform's site, and that
# distinction was quietly costing real discoveries: a reverse-image engine that
# finds a LinkedIn profile photo returns `media.licdn.com/...`, not
# `linkedin.com/in/...`, because the CDN is where the pixels live. Matching only
# the site domain filed every such hit as unrecognised "Other Web" at the lowest
# discovery priority — so the platform a match came from was both mis-reported
# and de-prioritised for verification.
#
# Only CDNs that belong to exactly one platform are listed. `fbcdn.net` is
# deliberately absent: it serves both Facebook and Instagram, and guessing which
# would put a claim in the evidence that the URL does not support.
# ---------------------------------------------------------------------------
PLATFORM_MEDIA_DOMAINS: dict[str, str] = {
    "licdn.com": "LinkedIn",             # media.licdn.com, static.licdn.com
    "cdninstagram.com": "Instagram",
    "twimg.com": "X/Twitter",            # pbs.twimg.com, abs.twimg.com
    "githubusercontent.com": "GitHub",
    "ggpht.com": "YouTube",              # yt3.ggpht.com — channel avatars
    "ytimg.com": "YouTube",              # i.ytimg.com — video thumbnails
    "pinimg.com": "Pinterest",
    "tiktokcdn.com": "TikTok",
    "tiktokcdn-us.com": "TikTok",
    "redditmedia.com": "Reddit",
    "redd.it": "Reddit",
}

# GitHub hosts that serve image bytes rather than pages. Kept explicit so
# candidate typing can tell "an avatar file" from "a profile page".
GITHUB_MEDIA_HOSTS = frozenset({
    "avatars.githubusercontent.com",
    "user-images.githubusercontent.com",
    "raw.githubusercontent.com",
    "camo.githubusercontent.com",
    "media.githubusercontent.com",
})

# ---------------------------------------------------------------------------
# Discovery priority.
#
# Priority orders *where we look first and which leads we spend fetch budget
# on*. It never orders truth: a candidate's verification strength always
# outranks its platform (see verification/scorer.rank), so a strongly verified
# personal website beats an unverified LinkedIn profile. Lower number = looked
# at earlier.
# ---------------------------------------------------------------------------
PLATFORM_PRIORITY: dict[str, int] = {
    "LinkedIn": 1,
    "Instagram": 2,
    "X/Twitter": 3,
    "GitHub": 4,
    "YouTube": 5,
}
# A platform we can name but that is not in the priority list (Facebook,
# Reddit, TikTok…). Still ahead of the unrecognised web.
RECOGNISED_PLATFORM_PRIORITY = 20
# Anything else on the public web. Searched and verified like everything else.
OTHER_WEB_PRIORITY = 90


def platform_priority(platform: str | None) -> int:
    """Discovery rank for a platform name. Unknown/None → the wider web."""
    if not platform:
        return OTHER_WEB_PRIORITY
    return PLATFORM_PRIORITY.get(platform, RECOGNISED_PLATFORM_PRIORITY)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- blockchain -------------------------------------------------------
    # Which testnet to attest on. Everything else (EAS addresses, chain id,
    # explorers, RPCs) is derived from this, so a chain switch cannot leave a
    # half-updated set of constants behind.
    network: str = DEFAULT_NETWORK
    # Optional explicit RPC. Leave blank to use the network's public RPCs.
    rpc_url: str = ""
    # Back-compat alias: older .env files set BASE_SEPOLIA_RPC_URL.
    base_sepolia_rpc_url: str = ""
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
    # Hard wall-clock ceiling for one provider. `search_timeout_s` bounds
    # individual browser operations; this bounds the whole adapter, so an engine
    # that hangs between operations still cannot stall the scan.
    engine_timeout_s: int = 120
    # Hard ceiling for the entire search stage across all providers.
    search_total_timeout_s: int = 300
    # How many providers may run at once. Each browser provider owns its own
    # Chromium, so this is also a cap on concurrent browser processes.
    search_concurrency: int = 3
    max_candidates_per_engine: int = 60
    # Optional: turn a local file into a public URL so engines' by-URL search
    # endpoints can be used (far more reliable than their upload flows).
    # Two publication paths (tried in order):
    #
    # 1. Local server — set LOCAL_IMAGE_BASE_URL to the server's own public
    #    base URL (e.g. https://myhost:8000).  The FastAPI server exposes a
    #    ``/api/v1/tmp-image/{token}`` route that serves the file for the
    #    duration of the scan.  Nothing is uploaded to a third party.
    #    This is the recommended path when the server is internet-reachable.
    #
    # 2. Third-party host — set ALLOW_UPLOAD_HOST=true to upload to Litterbox
    #    (1h TTL) or UPLOAD_HOST.  Only used when LOCAL_IMAGE_BASE_URL is not set.
    local_image_base_url: str = ""
    allow_upload_host: bool = False
    upload_host: str = "https://litterbox.catbox.moe/resources/internals/api.php"
    # Tried automatically, in order, only if `upload_host` fails or a WAF
    # blocks it (observed: Litterbox intermittently 403s automated requests
    # from some networks — a bot-challenge page, not a real error). uguu.se
    # is a second anonymous, no-signup temporary image host whose links serve
    # raw image bytes directly (verified: no HTML wrapper, no cookies, no
    # redirect — see `search/uploader.py::_upload_to_uguu`). Never the only
    # option tried: `publish_temporarily` still prefers `upload_host` first.
    upload_fallback_host: str = "https://uguu.se/upload"
    # Optional genuine reverse-image-search APIs (used only if key present).
    serpapi_key: str = ""

    # ---- verification ----------------------------------------------------
    image_match_threshold: float = 0.80
    weight_face: float = 0.50
    weight_image: float = 0.40
    weight_meta: float = 0.10
    verify_min_score: float = 0.70
    # Candidates at or above this face similarity are promoted to a priority
    # ranking group ahead of everything else verified — a ranking rule only,
    # independent of `face_match_threshold` (0.38, which decides whether a
    # candidate can verify at all) and `final_score` (which stays the
    # deciding number for ties, not for group membership).
    high_face_similarity_priority: float = 0.75
    # Off by default: normally a candidate needs `final_score >= verify_min_score`
    # (face + image + metadata combined) to verify. Enabling this adds a second,
    # independent path — face similarity alone, ignoring image/metadata — for a
    # deployment that has decided the combined-score gate is rejecting genuine
    # face matches whose retrieved image just happens to differ a lot (a
    # different photo, heavy cropping/filters, a stale/low-quality repost).
    # This is a real accuracy tradeoff: it raises the false-accept rate,
    # since a strong face match on an unrelated/miscropped image now verifies
    # on its own. Not a "fix" for a formula bug — a deliberate policy choice.
    face_only_verify_enabled: bool = False
    face_only_verify_threshold: float = 0.50
    max_candidates_to_verify: int = 12
    # Optional: path to a `CalibrationResult` JSON written by
    # `facechain.benchmark`. When set and its status is "CALIBRATED", every
    # case's `threshold_snapshot` records that instead of the honest default
    # "DEFAULT / not calibrated" state. Never changes `face_match_threshold`
    # itself — calibration only annotates the evidence, it does not silently
    # rewrite the operating point a human configured.
    calibration_file: str = ""
    http_timeout_s: int = 25
    user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    )

    # ---- quality gating --------------------------------------------------
    # Minimum face dimension in pixels (shorter side of bounding box).
    min_face_px: int = 80
    # Laplacian variance below this → BLURRY rejection.
    quality_blur_threshold: float = 40.0
    # Mean luminance bounds.
    quality_min_brightness: float = 15.0
    quality_max_brightness: float = 240.0
    # Maximum input image edge (pixels) before hard rejection (IMAGE_TOO_LARGE).
    max_image_edge: int = 8000
    # What to do when multiple faces are detected: reject | largest | all
    multi_face_policy: Literal["reject", "largest", "all"] = "largest"

    # ---- graded quality bands (informational, not hard gates) ------------
    # These score the passing image on a continuum so downstream stages
    # (ranking, confidence, "insufficient evidence" messaging) can react to
    # *how* good a passing face is, not just pass/fail. They never reject on
    # their own — `gate()`'s hard checks above still own that.
    quality_good_face_px: int = 160
    quality_good_blur: float = 80.0
    quality_good_det_score: float = 0.80
    quality_acceptable_det_score: float = 0.60
    quality_good_pose_deg: float = 15.0
    quality_acceptable_pose_deg: float = 30.0
    quality_brightness_margin: float = 25.0  # distance from min/max edge counted "GOOD"

    # ---- API keys for Tier-1 / Tier-2 search ----------------------------
    facecheck_api_key: str = ""
    search4faces_api_key: str = ""
    # Luxand.cloud face recognition API (https://luxand.cloud).
    # When set, every verified candidate is also cross-checked against the
    # Luxand recognition engine as an independent second opinion.
    luxand_api_key: str = ""

    # ---- web server (FastAPI) -------------------------------------------
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    api_upload_max_mb: int = 10
    api_max_concurrent_scans: int = 4
    api_job_ttl_s: int = 3600  # 1 hour

    # ---- output ----------------------------------------------------------
    evidence_dir: Path = Field(default=REPO_ROOT / "evidence")

    @property
    def engine_list(self) -> list[str]:
        return [e.strip() for e in self.engines.split(",") if e.strip()]

    # ---- derived network properties --------------------------------------

    @property
    def chain(self) -> dict:
        try:
            return NETWORKS[self.network]
        except KeyError:
            raise ValueError(
                f"unknown NETWORK {self.network!r}; choose one of {sorted(NETWORKS)}"
            ) from None

    @property
    def chain_id(self) -> int:
        return self.chain["chain_id"]

    @property
    def chain_name(self) -> str:
        return self.chain["name"]

    @property
    def eas_contract(self) -> str:
        return self.chain["eas"]

    @property
    def schema_registry_contract(self) -> str:
        return self.chain["schema_registry"]

    @property
    def rpc_candidates(self) -> tuple[str, ...]:
        """Configured RPC first (if any), then the network's public ones."""
        explicit = (self.rpc_url or self.base_sepolia_rpc_url or "").strip()
        public = tuple(self.chain["rpcs"])
        if not explicit:
            return public
        return (explicit, *(u for u in public if u != explicit))

    @property
    def faucet_hint(self) -> str:
        lines = "\n".join(f"  • {u}" for u in self.chain["faucets"])
        return f"Get free {self.chain_name} ETH:\n{lines}"

    def explorer_tx(self, tx: str) -> str:
        return self.chain["explorer_tx"].format(tx=tx)

    def easscan_attestation(self, uid: str) -> str:
        return self.chain["easscan_attestation"].format(uid=uid)

    def easscan_schema(self, uid: str) -> str:
        return self.chain["easscan_schema"].format(uid=uid)
    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.api_cors_origins.split(",") if o.strip()]


settings = Settings()
