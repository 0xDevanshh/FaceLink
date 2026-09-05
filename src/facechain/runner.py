"""The pipeline itself: image -> face -> reverse search -> verify -> chain -> read back.

Stage order is deliberate. Nothing touches the blockchain until a candidate has
actually passed the local verification ladder, so we can never attest to a
match we did not independently measure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import cv2

from . import PIPELINE_VERSION
from .benchmark import load_calibration_status
from .chain.eas import ChainError, EasClient, InsufficientFunds
from .chain.schema import schema_uid as predict_schema_uid
from .config import EAS_SCHEMA_DEFINITION, settings
from .evidence.hashing import sha256_bytes, sha256_file
from .evidence.writer import EvidenceWriter, new_case_id
from .face import selection as face_selection
from .face.encoder import crop_face, encode_detected, read_image
from .face.detector import load_backend
from .models import (
    Case, ChainRecord, EvidenceGraphEdge, EvidenceGraphNode, EvidenceGraphReport,
    FaceRecord, FaceSelection, InputImage, ThresholdSnapshot, VerifiedCandidate,
)
from .search.orchestrator import run_reverse_search
from .search.variants import generate_variants, cleanup_variants
from .verification.candidate import MediaCache, verify_candidate
from .verification.evidence_graph import build_evidence_graph
from .verification.image_similarity import perceptual_hashes
from .verification.scorer import explain_failure, highest_stage_reached, rank, score_candidate
from .enrichment.graph import enrich_case
from .face.luxand import search_face as luxand_search_face

log = logging.getLogger(__name__)

# on_event(stage, status, detail) — the CLI renders these; core logic stays UI-free.
Reporter = Callable[[str, str, str], None]


@dataclass
class RunOptions:
    image: str
    image_url: str | None = None
    engines: list[str] | None = None
    chain_mode: str = "onchain"  # onchain | simulate | skip
    face_backend: str | None = None
    max_verify: int | None = None
    case_id: str | None = None
    # ---- face selection -------------------------------------------------
    face_index: int | None = None
    crop_rect: list[int] | None = None
    selection_mode: str | None = None
    # ---- scan depth -----------------------------------------------------
    # fast: small candidate set (10), standard: normal (12), deep: maximum (30+)
    scan_depth: str = "standard"  # fast | standard | deep
    # ---- enrichment -----------------------------------------------------
    enrich: bool = False  # Run cross-platform profile enrichment after verification


class PipelineError(RuntimeError):
    pass


from .verification.clustering import cluster_candidates, corroboration_summary

# ---- scan depth budgets -----------------------------------------------
DEPTH_BUDGETS: dict[str, int] = {
    "fast": 5,
    "standard": 12,
    "deep": 30,
}
# Maximum bytes to download across all candidate images in one scan.
MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024  # 50 MB

# Share of the verification budget reserved for candidates outside the priority
# platforms. Without a reservation, a run that happens to surface 12 LinkedIn
# pages would never look at the wider web at all — which is how search priority
# quietly turns into search exclusivity.
WIDER_WEB_BUDGET_SHARE = 0.25

# At most this many candidates from any single *registrable* (eTLD+1) domain.
# Previously this cap applied per subdomain, which meant ca.linkedin.com,
# in.linkedin.com, nl.linkedin.com etc. each consumed separate slots — so 53
# LinkedIn candidates spread across country subdomains could fill 26 slots
# without ever reaching the target profile.  eTLD+1 normalisation ensures the
# cap is per brand, not per regional subdomain.
MAX_PER_DOMAIN = 2


def _root_domain(domain: str) -> str:
    """Best-effort registrable (eTLD+1) domain from a raw hostname.

    Splits on '.' and returns the last two labels, which covers the vast
    majority of cases (linkedin.com, github.com, x.com, …) without requiring
    a full public-suffix list dependency.  Three-part ccTLDs
    (e.g. co.uk, com.au) are handled by taking the last three labels when the
    penultimate label is 2-3 chars — an approximation that is accurate enough
    for the deduplication purpose here.
    """
    parts = domain.lower().split(".")
    if len(parts) <= 2:
        return domain
    # Heuristic: if the second-to-last label is very short it is likely a
    # second-level registry indicator (co, com, net, org, …) — take 3 parts.
    if len(parts) >= 3 and len(parts[-2]) <= 3:
        return ".".join(parts[-3:])
    return ".".join(parts[-2:])


def _spread_domains(candidates: list, cap: int = MAX_PER_DOMAIN) -> list:
    """Order-preserving pass that limits how many hits one root domain contributes.

    The cap applies to the registrable (eTLD+1) domain rather than the full
    subdomain, so ca.linkedin.com and in.linkedin.com both count against the
    same "linkedin.com" slot.  This prevents a single platform's 50+ country-
    subdomain variants from filling the entire verification budget before the
    target profile is reached.

    Overflow is not discarded but appended after everything else, so a domain
    with many hits still gets looked at if budget remains.
    """
    kept: list = []
    overflow: list = []
    seen: dict[str, int] = {}
    for cand in candidates:
        root = _root_domain(cand.domain)
        seen[root] = seen.get(root, 0) + 1
        (kept if seen[root] <= cap else overflow).append(cand)
    return kept + overflow


def _verification_queue(candidates: list, limit: int) -> list:
    """Choose which candidates get the (finite) fetch-and-measure budget.

    Priority platforms are drained first, but a slice of the budget is held back
    for everything else, so a strong match on a personal site or a conference
    page is still reachable in a run dominated by social hits.

    Skipped candidates (beyond the budget) are logged at DEBUG level so the
    absence of a known target can be diagnosed without re-running the scan.
    """
    if limit <= 0:
        return []
    from .config import OTHER_WEB_PRIORITY

    candidates = _spread_domains(candidates)
    priority = [c for c in candidates if c.platform_priority < OTHER_WEB_PRIORITY]
    wider = [c for c in candidates if c.platform_priority >= OTHER_WEB_PRIORITY]

    reserved = min(len(wider), max(1, int(limit * WIDER_WEB_BUDGET_SHARE))) if wider else 0
    queue = priority[: limit - reserved]
    queue += wider[: limit - len(queue)]
    # Any budget the priority tier did not use goes back to the wider web, and
    # vice versa, so the reservation never wastes capacity.
    if len(queue) < limit:
        seen = {id(c) for c in queue}
        queue += [c for c in candidates if id(c) not in seen][: limit - len(queue)]

    # Log skipped candidates so a missing ground-truth hit is diagnosable.
    queued_ids = {id(c) for c in queue}
    skipped = [c for c in candidates if id(c) not in queued_ids]
    if skipped:
        log.debug(
            "verification_queue: %d/%d candidates selected (budget=%d); "
            "skipped: %s",
            len(queue), len(candidates), limit,
            ", ".join(f"{c.domain}[{c.platform or 'web'}]" for c in skipped[:20]),
        )
    return queue


def run(opts: RunOptions, report: Reporter | None = None) -> Case:
    emit = report or (lambda *_: None)
    now = datetime.now(timezone.utc)
    case_id = opts.case_id or new_case_id(now)
    writer = EvidenceWriter(case_id)

    calibration_status, calibration_note = load_calibration_status(settings.calibration_file)
    case = Case(
        case_id=case_id,
        pipeline_version=PIPELINE_VERSION,
        created_at=now.isoformat(),
        observed_at=int(now.timestamp()),
        threshold_snapshot=ThresholdSnapshot(
            face_match_threshold=settings.face_match_threshold,
            image_match_threshold=settings.image_match_threshold,
            verify_min_score=settings.verify_min_score,
            weight_face=settings.weight_face,
            weight_image=settings.weight_image,
            weight_meta=settings.weight_meta,
            insightface_model=settings.insightface_model,
            face_backend=opts.face_backend or settings.face_backend,
            high_face_similarity_priority=settings.high_face_similarity_priority,
            face_only_verify_enabled=settings.face_only_verify_enabled,
            face_only_verify_threshold=settings.face_only_verify_threshold,
            calibration_status=calibration_status,
            calibration_note=calibration_note,
        ),
    )

    # ---- 1. input --------------------------------------------------------
    emit("input", "start", opts.image)
    path = Path(opts.image)
    if not path.exists():
        raise PipelineError(f"input image not found: {path}")
    img = read_image(path)
    input_hashes = perceptual_hashes(img)
    case.input = InputImage(
        path=str(path.resolve()),
        filename=path.name,
        bytes_len=path.stat().st_size,
        height=int(img.shape[0]),
        width=int(img.shape[1]),
        sha256=sha256_file(path),
        phash=input_hashes["phash"],
    )
    emit("input", "ok", f"{case.input.width}x{case.input.height}, sha256 {case.input.sha256[:16]}…")

    # ---- 2. face detection, selection, quality gate, embedding ----------
    #
    # Order matters here. Detection runs once on the original image; an
    # operator-supplied crop is applied and re-detected inside it, so the face
    # that gets embedded is always a face that was detected in the pixels the
    # gate measured. The original image and its hash are never replaced.
    emit("face", "start", "")
    original_sha256 = case.input.sha256
    working = img
    crop_applied: tuple[int, int, int, int] | None = None
    crop_sha256: str | None = None
    crop_path: Path | None = None
    mode = "auto"

    if opts.crop_rect:
        try:
            working, crop_applied = face_selection.apply_crop(img, opts.crop_rect)
        except face_selection.CropError as exc:
            case.verdict = "INVALID_CROP"
            case.failure_reason = f"crop rejected: {exc}"
            emit("face", "fail", case.failure_reason)
            writer.write_bundle(case, path)
            return case
        crop_bytes = face_selection.encode_png(working)
        crop_sha256 = sha256_bytes(crop_bytes)
        crop_path = writer.save_bytes("selected_crop.png", crop_bytes)
        mode = "manual-crop"
        emit("face:crop", "ok",
             f"{crop_applied[2]}x{crop_applied[3]} at {crop_applied[0]},{crop_applied[1]}")

    backend = load_backend(opts.face_backend)
    all_faces = backend.detect(working)
    offer = face_selection.offer(working, all_faces)

    if not all_faces:
        case.face = FaceRecord(detected=False, backend=backend.name,
                               model=backend.model_name, faces_found=0)
        case.verdict = "NO_FACE"
        case.failure_reason = (
            "No usable face detected. Please upload a clearer image or manually "
            "select a face."
        )
        emit("face", "fail", case.failure_reason)
        writer.write_bundle(case, path)
        return case

    # Resolve which face this scan is about.
    face_index = opts.face_index
    if face_index is not None:
        if not 0 <= face_index < len(all_faces):
            case.verdict = "INVALID_FACE_SELECTION"
            case.failure_reason = (
                f"face_index {face_index} does not exist; {len(all_faces)} face(s) detected"
            )
            emit("face", "fail", case.failure_reason)
            writer.write_bundle(case, path)
            return case
        # Only *infer* a manual selection. When the caller told us how the face
        # was chosen we believe it — the UI passes `auto` along with the index
        # it was itself given, and recording that as "manual-face" would put a
        # claim in the evidence that a human made a choice they never made.
        if mode == "auto" and opts.selection_mode is None:
            mode = "manual-face"
    elif offer.auto_index is not None:
        face_index = offer.auto_index
    elif crop_applied is not None:
        # An operator drew a crop around the face they meant; the largest face
        # inside that crop is that face, so no further prompt is warranted.
        face_index = max(range(len(all_faces)), key=lambda i: all_faces[i].area)
    else:
        # Ambiguous, and nobody has chosen. Refusing beats guessing: a wrong
        # automatic pick yields a confident scan of the wrong person.
        case.face = FaceRecord(
            detected=True, backend=backend.name, model=backend.model_name,
            faces_found=len(all_faces),
            faces=offer.faces,
        )
        case.face_selection = FaceSelection(
            mode="pending", faces_offered=len(offer.faces),
            original_sha256=original_sha256,
            original_width=case.input.width, original_height=case.input.height,
            selected_at=datetime.now(timezone.utc).isoformat(),
        )
        case.verdict = "FACE_SELECTION_REQUIRED"
        case.failure_reason = offer.reason
        emit("face:selection_required", "fail", offer.reason)
        writer.write_bundle(case, path)
        return case

    # A caller-supplied mode wins over the inferred one, except that an applied
    # crop is a fact about the pixels and cannot be relabelled away.
    if mode != "manual-crop" and opts.selection_mode in ("auto", "manual-face", "manual-crop"):
        mode = opts.selection_mode

    # Quality gate on the face we are actually about to embed.
    quality = face_selection.gate_selected(working, all_faces, face_index)
    if not quality.passed:
        case.face = FaceRecord(
            detected=True, backend=backend.name, model=backend.model_name,
            faces_found=len(all_faces), faces=offer.faces, quality=quality.rounded(),
        )
        case.verdict = "FACE_QUALITY_INSUFFICIENT"
        case.failure_reason = (
            "Face quality is insufficient for reliable matching. Please select a "
            f"clearer or larger face. ({quality.error}: {quality.detail})"
        )
        emit("face:quality", "fail", f"{quality.error}: {quality.detail}")
        writer.write_bundle(case, path)
        return case
    emit("face:quality", "ok",
         f"blur {quality.blur_score:.1f}, face {quality.face_px}px, {quality.face_count} face(s)")

    face_record, embedding, primary = encode_detected(backend, all_faces, face_index)
    face_record.faces = offer.faces
    face_record.quality = quality.rounded()
    case.face = face_record
    if not face_record.detected or embedding is None:
        case.verdict = "NO_FACE"
        case.failure_reason = "no face could be embedded from the selected region"
        emit("face", "fail", case.failure_reason)
        writer.write_bundle(case, path)
        return case

    if primary is not None:
        ok, buf = cv2.imencode(".png", crop_face(working, primary))
        if ok:
            writer.save_bytes("face_crop.png", buf.tobytes())

    case.face_selection = FaceSelection(
        mode=mode,
        face_index=face_index,
        faces_offered=len(offer.faces),
        bbox=face_record.bbox,
        crop_rect=list(crop_applied) if crop_applied else None,
        crop_sha256=crop_sha256,
        original_sha256=original_sha256,
        original_width=case.input.width,
        original_height=case.input.height,
        selected_at=datetime.now(timezone.utc).isoformat(),
    )
    emit(
        "face",
        "ok",
        f"{face_record.faces_found} face(s), selected #{face_index} ({mode}), "
        f"{face_record.embedding_dimension}-D {face_record.model}, "
        f"det {face_record.det_score:.3f}",
    )

    # ---- 3. reverse image search ----------------------------------------
    #
    # The *query* is the region the operator selected, not always the whole
    # upload. When someone crops one person out of a group shot, searching the
    # full frame searches for the group shot: a real run cropped one face out of
    # a two-person photo and every engine hit came back matching the composite,
    # leaving the best face similarity at 0.079 while the uncropped run on the
    # same person scored 0.96. The crop is what they asked to look for.
    #
    # A bare face *selection* (no crop) still searches the whole image on
    # purpose: the other people in it are legitimately part of the photograph,
    # the full frame can find the original as an exact-image match, and face
    # verification against the selected face is what decides whether a hit is
    # about the right person.
    #
    # The original is untouched throughout — its bytes, hash and phash are the
    # evidence anchor regardless of what was searched.
    search_path = crop_path if crop_path is not None else path
    query_hashes = perceptual_hashes(working) if crop_path is not None else input_hashes
    if crop_path is not None:
        emit("search:query", "info", "searching the selected crop, not the full upload")

    # Search-variant generation (see `search/variants.py`): beyond `fast` depth,
    # also search one or two crops around the selected face, budgeted and
    # deduplicated so a headshot-sized upload does not burn extra search passes
    # on crops indistinguishable from the original.
    variants = generate_variants(working, str(search_path), primary, opts.scan_depth)
    if len(variants) > 1:
        emit("search:query", "info",
             f"generated {len(variants)} search variant(s) for scan_depth={opts.scan_depth}")

    emit("search", "start", "")
    try:
        search_report, public_url = run_reverse_search(
            str(search_path),
            engines=opts.engines,
            image_url=opts.image_url,
            on_event=lambda eng, status, detail: emit(f"search:{eng}", status, detail),
            variants=variants,
        )
    finally:
        cleanup_variants(variants, keep_path=str(search_path))
    case.reverse_search = search_report
    # Report every provider's terminal state, including the ones that found
    # nothing — "we asked Bing and it challenged us" is part of the record.
    for provider in search_report.providers:
        emit(f"search:{provider.engine}", "info",
             f"{provider.status.value} · {provider.candidates} candidates "
             f"· {provider.duration_s:.1f}s")
    for platform, count in search_report.platform_counts.items():
        emit("search:platform", "info", f"{platform}: {count}")

    if not search_report.candidates:
        statuses = ", ".join(
            f"{p.engine}={p.status.value}" for p in search_report.providers
        ) or "no provider ran"
        case.verdict = "NO_SEARCH_RESULTS"
        case.failure_reason = f"no reverse-image provider returned usable results ({statuses})"
        emit("search", "fail", case.failure_reason)
        writer.write_bundle(case, path)
        return case
    emit(
        "search",
        "ok",
        f"{search_report.total_candidates} candidates "
        f"({search_report.social_candidates} on named platforms) from "
        f"{', '.join(search_report.engines_succeeded) or 'no engine'}",
    )

    # ---- 4. candidate verification --------------------------------------
    emit("verify", "start", "")
    depth_limit = DEPTH_BUDGETS.get(opts.scan_depth, DEPTH_BUDGETS["standard"])
    limit = opts.max_verify or depth_limit
    if opts.scan_depth == "deep":
        emit("verify:depth", "info", f"deep scan mode — budget {limit} candidates")
    queue = _verification_queue(search_report.candidates, limit)
    emit(
        "verify:queue", "info",
        f"{len(queue)}/{search_report.total_candidates} candidates selected for verification "
        f"(budget={limit}, depth={opts.scan_depth})",
    )

    verified: list[VerifiedCandidate] = []
    media_cache = MediaCache()
    download_bytes = 0
    for cand in queue:
        if download_bytes >= MAX_DOWNLOAD_BYTES:
            emit("verify:limit", "info",
                 f"download budget reached ({download_bytes // (1024*1024)}MB) — stopping early")
            break
        try:
            vc = score_candidate(verify_candidate(cand, query_hashes, embedding, media_cache))
        except Exception as exc:  # noqa: BLE001
            log.warning("candidate %s raised during verification: %s", cand.url, exc)
            vc = VerifiedCandidate(
                engine=cand.engine, url=cand.url, domain=cand.domain,
                platform=cand.platform, is_social=cand.is_social,
                canonical_url=cand.url, platform_priority=cand.platform_priority,
                candidate_type=cand.candidate_type,
                rejection_reason=f"verification error: {type(exc).__name__}",
            )
        verified.append(vc)
        download_bytes = media_cache.total_bytes  # actual bytes downloaded this run
        emit(
            "verify:candidate",
            "ok" if vc.verified else "info",
            f"{(vc.platform or vc.domain)[:24]:<24} img {vc.image_similarity:.2f} "
            f"face {vc.face_similarity:.2f} score {vc.final_score:.2f} "
            f"{'VERIFIED' if vc.verified else vc.match_type}",
        )
        # Deep mode: early-stop only when we have 3+ independent verified clusters
        if opts.scan_depth != "deep" and vc.verified:
            # Standard/fast: stop at first verified match to save time
            pass  # still process remainder so full ranked list is available
    emit("verify:media", "info",
         f"{media_cache.downloads} image download(s), {media_cache.hits} reused from cache, "
         f"{media_cache.total_bytes / (1024*1024):.1f}MB total")

    # ---- 4a. optional Luxand cross-check ---------------------------------
    # Independent cloud face recognition on the input image.  Additive only —
    # never changes scores or thresholds, just adds an SSE event to the
    # evidence trail so operators can see the second-opinion result.
    if settings.luxand_api_key:
        try:
            luxand = luxand_search_face(str(path))
            emit(
                "verify:luxand", "ok" if luxand.matched else "info",
                f"matched={luxand.matched} confidence={luxand.confidence:.3f} "
                f"faces={luxand.faces_found} note={luxand.note}",
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("luxand cross-check raised: %s", exc)
            emit("verify:luxand", "info", f"skipped ({type(exc).__name__})")

    # ---- 4b. image deduplication + corroboration -------------------------
    clusters = cluster_candidates(verified)
    corr = corroboration_summary(clusters)
    emit(
        "verify:corroboration", "info",
        f"{corr.image_clusters} image clusters (deduped from {corr.total_candidates}), "
        f"{corr.independent_domains} domains, {corr.independent_platforms} platforms, "
        f"{corr.verified_clusters} verified",
    )
    if corr.duplicate_count > 0:
        emit("verify:duplicates", "info",
             f"{corr.duplicate_count} near-duplicate image(s) grouped — count as 1 evidence cluster each")

    graph = build_evidence_graph(clusters)
    case.evidence_graph = EvidenceGraphReport(
        nodes=[EvidenceGraphNode(id=n.id, type=n.type, label=n.label) for n in graph.nodes],
        edges=[EvidenceGraphEdge(source=e.source, target=e.target, type=e.type, note=e.note)
               for e in graph.edges],
        independent_evidence_count=graph.independent_evidence_count,
    )
    emit("verify:evidence_graph", "info",
         f"{graph.independent_evidence_count} independent evidence source(s) "
         f"({len(graph.nodes)} nodes, {len(graph.edges)} relationships)")

    ranked = rank(verified)
    case.verification = ranked
    case.stages_passed = highest_stage_reached(ranked)
    case.best_match = ranked[0] if ranked else None

    confirmed = next((c for c in ranked if c.verified), None)
    if confirmed is None:
        case.verdict = "UNVERIFIED"
        case.failure_reason = explain_failure(ranked)
        emit("verify", "fail", case.failure_reason)
        writer.write_bundle(case, path)
        return case

    case.best_match = confirmed
    emit(
        "verify",
        "ok",
        f"{confirmed.platform} {confirmed.url} score {confirmed.final_score:.3f}",
    )

    # ---- 5. evidence payload + hash -------------------------------------
    emit("evidence", "start", "")
    payload = writer.build_payload(case)
    _, evidence_hash = writer.write_payload(payload)
    case.evidence_sha256 = evidence_hash
    emit("evidence", "ok", f"evidenceHash sha256:{evidence_hash[:24]}…")

    # ---- 6. blockchain attestation --------------------------------------
    if opts.chain_mode == "skip":
        case.blockchain = ChainRecord(network=settings.chain_name, chain_id=settings.chain_id,
                                      mode="skipped", note="--no-chain requested")
        case.verdict = "VERIFIED_OFFCHAIN"
        emit("chain", "info", "skipped by request")
        writer.write_bundle(case, path)
        return case

    emit("chain", "start", "")
    try:
        client = EasClient()
        fields = EasClient.build_fields(payload, evidence_hash)

        if opts.chain_mode == "simulate":
            # Simulation needs a *registered* schema to call against; fall back
            # to the predicted UID so the encoding is still exercised.
            schema = settings.eas_schema_uid or predict_schema_uid()
            uid = client.simulate_attest(schema, fields)
            case.blockchain = ChainRecord(
                network=client.chain_name,
                chain_id=client.chain_id,
                mode="simulate",
                eas_contract=client.eas.address,
                schema_uid=schema,
                schema_definition=EAS_SCHEMA_DEFINITION,
                attester=client.account.address if client.account else "",
                attestation_uid=uid,
                note="eth_call simulation: encoding and contract call validated, nothing written",
            )
            emit("chain", "ok", f"simulated attestation uid {uid[:18]}… (no gas spent)")
            case.verdict = "VERIFIED_SIMULATED"
            writer.write_bundle(case, path)
            return case

        balance = client.preflight()
        emit("chain:wallet", "ok", f"{client.address} — {balance:.6f} ETH on {client.chain_name}")

        schema = client.resolve_schema()
        emit("chain:schema", "ok", schema)

        record = client.attest(schema, fields)
        case.blockchain = record
        emit("chain:tx", "ok", f"{record.tx_hash} in block {record.block_number}")

        # ---- 7. independent read-back ----------------------------------
        emit("readback", "start", "")
        if not record.attestation_uid:
            raise ChainError("transaction mined but no attestation UID was emitted")
        ok, mismatches, onchain = client.verify_readback(
            record.attestation_uid, fields, expected_attester=client.address
        )
        record.readback_verified = ok
        record.readback_mismatches = mismatches
        record.onchain_decoded = onchain["decoded"]
        if ok:
            case.verdict = "VERIFIED"
            emit("readback", "ok", f"all {len(fields)} on-chain fields match local evidence")
        else:
            case.verdict = "CHAIN_MISMATCH"
            case.failure_reason = "; ".join(mismatches)
            emit("readback", "fail", case.failure_reason)

    except Exception as exc:  # noqa: BLE001
        # Local verification already succeeded and is fully recorded on disk;
        # a chain-side failure must degrade the verdict, not lose the evidence.
        if not isinstance(exc, (InsufficientFunds, ChainError)):
            log.exception("unexpected error in blockchain stage")
        case.blockchain = ChainRecord(network=settings.chain_name, chain_id=settings.chain_id,
                                      mode="failed", note=str(exc)[:500])
        case.verdict = "VERIFIED_OFFCHAIN"
        case.failure_reason = f"blockchain stage failed: {str(exc)[:300]}"
        emit("chain", "fail", str(exc)[:300])

    # ---- 8. optional profile enrichment ----------------------------------
    # Runs only when requested (opts.enrich=True) and a verified candidate
    # exists.  Entirely additive — never modifies the verdict or evidence hash.
    if opts.enrich and case.verification:
        try:
            case.profile_graph = enrich_case(
                case.verification,
                embedding,
                query_hashes,
                emit=lambda stage, status, detail: emit(f"enrich:{stage}", status, detail),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("enrichment raised, skipping: %s", exc)
            emit("enrich", "fail", f"skipped: {type(exc).__name__}")

    writer.write_bundle(case, path)
    return case


def bundle_dir(case_id: str) -> Path:
    return Path(settings.evidence_dir) / case_id
