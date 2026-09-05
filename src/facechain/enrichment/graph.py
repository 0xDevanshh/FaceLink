"""Cross-platform profile graph builder.

Takes the verified candidates from runner.run() and:
  1. Extracts structured profile data from each verified profile page.
  2. Discovers linked profiles (GitHub → LinkedIn, etc.) from page content.
  3. Face-verifies each newly discovered profile using the original embedding.
  4. Deduplicates profiles by canonical URL and platform:handle.
  5. Builds edges between profiles based on direct links and shared signals.
  6. Assigns evidence levels: CONFIRMED / HIGH_CONFIDENCE / POSSIBLE / REJECTED.

Evidence-level rules (conservative by design):
  CONFIRMED         face_similarity ≥ threshold AND ≥1 independent corroborating signal
                    (cross-link, shared username, verified from different platform)
  HIGH_CONFIDENCE   face_similarity ≥ threshold, no independent corroboration
  POSSIBLE          no face match; signals are name / handle / link only
  REJECTED          face detected, face_similarity < threshold (wrong person)

Name alone never reaches above POSSIBLE.
Face alone never reaches above HIGH_CONFIDENCE without corroboration.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional
from urllib.parse import urlparse

import numpy as np

from ..config import settings
from ..models import VerifiedCandidate
from ..verification.candidate import MediaCache, _client, _download_image, _safe_get
from ..verification.image_similarity import compare, perceptual_hashes
from ..face.encoder import decode_image
from ..face.detector import load_backend
from ..face.similarity import best_match_index
from .extractor import (
    ENRICHMENT_PLATFORMS,
    _classify_enrichment_platform,
    _extract_username,
    _profile_id,
    extract_profile,
)
from .profile import (
    DiscoveredProfile,
    EvidenceLevel,
    ProfileField,
    ProfileGraph,
    ProfileGraphEdge,
)

log = logging.getLogger(__name__)

Reporter = Callable[[str, str, str], None]

# Maximum number of cross-platform profiles to discover per seed profile.
MAX_CROSS_PLATFORM_PER_SEED = 8
# Maximum total profiles in one enrichment run.
MAX_TOTAL_PROFILES = 30


def _face_verify_profile(
    profile: DiscoveredProfile,
    input_embedding: np.ndarray,
    input_hashes: dict[str, str],
    cache: MediaCache,
) -> None:
    """Fetch the profile's avatar and run ArcFace against the input embedding.

    Mutates *profile* in place: sets face_similarity, image_similarity,
    face_detected, candidate_image_url, candidate_image_source.
    """
    if not profile.avatar_url:
        return

    img_url = profile.avatar_url.value
    with _client() as client:
        data = cache.get_or_fetch(client, img_url)

    if data is None:
        log.debug("enrichment: could not download avatar for %s", profile.profile_id)
        return

    try:
        cand_hashes = perceptual_hashes(data)
        profile.image_similarity = compare(input_hashes, cand_hashes)
        profile.candidate_image_url = img_url
        profile.candidate_image_source = profile.avatar_url.extraction_method
    except Exception as exc:  # noqa: BLE001
        log.debug("enrichment: phash failed for %s: %s", profile.profile_id, exc)
        return

    img = decode_image(data)
    if img is None:
        return
    try:
        faces = load_backend().detect(img)
        profile.face_detected = bool(faces)
        if faces:
            sim, _ = best_match_index(input_embedding, [f.embedding for f in faces])
            profile.face_similarity = sim
    except Exception as exc:  # noqa: BLE001
        log.debug("enrichment: face detection failed for %s: %s", profile.profile_id, exc)


def _assign_evidence_level(
    profile: DiscoveredProfile,
    seed_usernames: set[str],
    seed_handles: set[str],
    confirmed_profile_ids: set[str],
) -> None:
    """Assign evidence_level based on all available signals.

    Mutates *profile* in place.  Rules:
    - REJECTED:          face detected but similarity below threshold
    - HIGH_CONFIDENCE:   face similarity ≥ threshold
    - CONFIRMED:         face similarity ≥ threshold + ≥1 corroborating signal
    - POSSIBLE:          name/handle/link match only (no face evidence)
    """
    threshold = settings.face_match_threshold
    face_ok = profile.face_detected and profile.face_similarity >= threshold

    if profile.face_detected and not face_ok:
        profile.evidence_level = EvidenceLevel.REJECTED
        profile.rejection_reason = (
            f"face detected but similarity {profile.face_similarity:.3f} < "
            f"threshold {threshold}"
        )
        return

    signals: list[str] = []
    if face_ok:
        signals.append("face_match")

    # Username/handle consistency
    if profile.username and profile.username.value.lower() in seed_usernames:
        signals.append("username_match")
    if profile.username and profile.username.value.lower() in seed_handles:
        signals.append("handle_match")

    # Cross-link from an already-confirmed profile
    if profile.discovered_from:
        src_id = profile.discovered_from
        if any(pid in confirmed_profile_ids for pid in [src_id]):
            signals.append("cross_link_from_confirmed")

    profile.evidence_signals = signals

    if face_ok and len(signals) >= 2:
        profile.evidence_level = EvidenceLevel.CONFIRMED
    elif face_ok:
        profile.evidence_level = EvidenceLevel.HIGH_CONFIDENCE
    elif signals:
        profile.evidence_level = EvidenceLevel.POSSIBLE
    else:
        profile.evidence_level = EvidenceLevel.POSSIBLE


def _canonical_key(url: str) -> str:
    """Stable deduplication key for a profile URL."""
    parsed = urlparse(url.lower().rstrip("/"))
    return f"{parsed.netloc}{parsed.path}"


def enrich_case(
    verified_candidates: list[VerifiedCandidate],
    input_embedding: np.ndarray,
    input_hashes: dict[str, str],
    emit: Reporter | None = None,
) -> ProfileGraph:
    """Build a ProfileGraph from the verified candidates of one scan.

    Steps:
      1. Seed: extract profile data from each verified candidate.
      2. Discover: follow cross-platform links found on seed pages.
      3. Verify: face-verify each discovered profile.
      4. Deduplicate: merge profiles with the same canonical URL.
      5. Assign evidence levels.
      6. Build edges.
    """
    _emit = emit or (lambda *_: None)
    graph = ProfileGraph()
    cache = MediaCache()
    seen_keys: set[str] = set()   # canonical URL keys already added
    profile_map: dict[str, DiscoveredProfile] = {}  # profile_id → profile

    # ---- collect seed usernames from verified candidates ------------------
    seed_usernames: set[str] = set()
    seed_handles: set[str] = set()

    verified_profiles = [c for c in verified_candidates if c.verified]
    if not verified_profiles:
        _emit("enrich", "info", "no verified candidates — skipping enrichment")
        return graph

    _emit("enrich", "start",
          f"enriching {len(verified_profiles)} verified candidate(s)")

    # ---- Step 1: seed profiles -------------------------------------------
    for vc in verified_profiles:
        platform = vc.platform or "Other"
        if platform not in ENRICHMENT_PLATFORMS:
            continue
        url = vc.canonical_url or vc.url
        key = _canonical_key(url)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        _emit("enrich:seed", "info", f"{platform} {url[:60]}")
        profile = extract_profile(url, platform)
        profile.face_similarity = vc.face_similarity
        profile.image_similarity = vc.image_similarity
        profile.face_detected = vc.face_detected
        profile.candidate_image_url = vc.candidate_image_url
        profile.candidate_image_source = vc.candidate_image_source
        profile.discovery_method = "reverse-image"

        if profile.username:
            seed_usernames.add(profile.username.value.lower())
            seed_handles.add(profile.username.value.lower().lstrip("@"))

        profile_map[profile.profile_id] = profile

        if len(profile_map) >= MAX_TOTAL_PROFILES:
            break

    # ---- Step 2: cross-platform discovery --------------------------------
    seeds_for_cross = list(profile_map.values())
    for seed in seeds_for_cross:
        if len(profile_map) >= MAX_TOTAL_PROFILES:
            break
        cross_count = 0
        for link_field in seed.linked_profiles:
            if cross_count >= MAX_CROSS_PLATFORM_PER_SEED:
                break
            if len(profile_map) >= MAX_TOTAL_PROFILES:
                break

            link_url = link_field.value
            link_platform = _classify_enrichment_platform(link_url)
            if not link_platform or link_platform not in ENRICHMENT_PLATFORMS:
                continue
            key = _canonical_key(link_url)
            if key in seen_keys:
                continue
            seen_keys.add(key)

            _emit("enrich:discover", "info",
                  f"{link_platform} (from {seed.platform}) {link_url[:60]}")
            cross_profile = extract_profile(link_url, link_platform)
            cross_profile.discovery_method = "cross-profile-link"
            cross_profile.discovered_from = seed.profile_id

            # Face-verify the newly discovered profile
            _face_verify_profile(cross_profile, input_embedding, input_hashes, cache)

            if cross_profile.username:
                seed_usernames.add(cross_profile.username.value.lower())

            profile_map[cross_profile.profile_id] = cross_profile
            cross_count += 1

    # ---- Step 3: assign evidence levels ----------------------------------
    confirmed_ids: set[str] = set()
    # First pass: face-verified seeds are at least HIGH_CONFIDENCE
    for p in profile_map.values():
        _assign_evidence_level(p, seed_usernames, seed_handles, confirmed_ids)
        if p.evidence_level == EvidenceLevel.CONFIRMED:
            confirmed_ids.add(p.profile_id)

    # Second pass: cross-platform profiles discovered from confirmed seeds
    # may now upgrade to CONFIRMED if the confirming seed was just resolved.
    for p in profile_map.values():
        if p.evidence_level == EvidenceLevel.HIGH_CONFIDENCE and p.discovered_from:
            if p.discovered_from in confirmed_ids:
                p.evidence_signals.append("cross_link_from_confirmed")
                if len(p.evidence_signals) >= 2:
                    p.evidence_level = EvidenceLevel.CONFIRMED
                    confirmed_ids.add(p.profile_id)

    # ---- Step 4: build edges ---------------------------------------------
    for p in profile_map.values():
        graph.profiles.append(p)

    edge_seen: set[tuple[str, str, str]] = set()

    def add_edge(src: str, tgt: str, rel: str, note: str = "") -> None:
        key = (src, tgt, rel)
        if key not in edge_seen:
            edge_seen.add(key)
            graph.edges.append(ProfileGraphEdge(
                source_id=src, target_id=tgt, relationship=rel, note=note))

    for p in graph.profiles:
        if p.discovered_from and p.discovered_from in profile_map:
            add_edge(p.discovered_from, p.profile_id, "linked-from")

        # Same-face edges between confirmed/high-confidence profiles
        for other in graph.profiles:
            if other.profile_id == p.profile_id:
                continue
            if (p.face_similarity >= settings.face_match_threshold and
                    other.face_similarity >= settings.face_match_threshold):
                add_edge(p.profile_id, other.profile_id, "same-face",
                         f"face_sim={p.face_similarity:.2f},{other.face_similarity:.2f}")

        # Same-username edge
        if p.username:
            for other in graph.profiles:
                if other.profile_id == p.profile_id:
                    continue
                if (other.username and
                        other.username.value.lower() == p.username.value.lower()):
                    add_edge(p.profile_id, other.profile_id, "shared-username",
                             p.username.value)

    graph.update_counts()
    _emit(
        "enrich", "ok",
        f"{len(graph.profiles)} profiles: "
        f"confirmed={graph.confirmed_count} "
        f"high={graph.high_confidence_count} "
        f"possible={graph.possible_count} "
        f"rejected={graph.rejected_count}",
    )
    return graph
