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
from .chain.eas import ChainError, EasClient, InsufficientFunds
from .chain.schema import schema_uid as predict_schema_uid
from .config import EAS_SCHEMA_DEFINITION, settings
from .evidence.hashing import sha256_file
from .evidence.writer import EvidenceWriter, new_case_id
from .face.encoder import crop_face, encode_face, read_image
from .face.detector import select_primary
from .models import Case, ChainRecord, InputImage, Stage, VerifiedCandidate
from .search.orchestrator import run_reverse_search
from .verification.candidate import verify_candidate
from .verification.image_similarity import perceptual_hashes
from .verification.scorer import explain_failure, highest_stage_reached, rank, score_candidate

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


class PipelineError(RuntimeError):
    pass


def run(opts: RunOptions, report: Reporter | None = None) -> Case:
    emit = report or (lambda *_: None)
    now = datetime.now(timezone.utc)
    case_id = opts.case_id or new_case_id(now)
    writer = EvidenceWriter(case_id)

    case = Case(
        case_id=case_id,
        pipeline_version=PIPELINE_VERSION,
        created_at=now.isoformat(),
        observed_at=int(now.timestamp()),
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

    # ---- 2. face detection + embedding ----------------------------------
    emit("face", "start", "")
    face_record, embedding, all_faces = encode_face(img, opts.face_backend)
    case.face = face_record
    if not face_record.detected or embedding is None:
        case.verdict = "NO_FACE"
        case.failure_reason = "no face detected in the input image"
        emit("face", "fail", case.failure_reason)
        writer.write_bundle(case, path)
        return case

    primary = select_primary(all_faces)
    if primary is not None:
        ok, buf = cv2.imencode(".png", crop_face(img, primary))
        if ok:
            writer.save_bytes("face_crop.png", buf.tobytes())
    emit(
        "face",
        "ok",
        f"{face_record.faces_found} face(s), {face_record.embedding_dimension}-D "
        f"{face_record.model}, det {face_record.det_score:.3f}",
    )

    # ---- 3. reverse image search ----------------------------------------
    emit("search", "start", "")
    search_report, public_url = run_reverse_search(
        str(path),
        engines=opts.engines,
        image_url=opts.image_url,
        on_event=lambda eng, status, detail: emit(f"search:{eng}", status, detail),
    )
    case.reverse_search = search_report
    if not search_report.candidates:
        case.verdict = "NO_SEARCH_RESULTS"
        case.failure_reason = (
            "no reverse-image engine returned usable results: "
            + "; ".join(f"{k}={v}" for k, v in search_report.engine_errors.items())
        )
        emit("search", "fail", case.failure_reason)
        writer.write_bundle(case, path)
        return case
    emit(
        "search",
        "ok",
        f"{search_report.total_candidates} candidates "
        f"({search_report.social_candidates} social) from "
        f"{', '.join(search_report.engines_succeeded) or 'no engine'}",
    )

    # ---- 4. candidate verification --------------------------------------
    emit("verify", "start", "")
    limit = opts.max_verify or settings.max_candidates_to_verify
    # Social candidates first — the task asks specifically for a social post.
    queue = [c for c in search_report.candidates if c.is_social][:limit]
    remaining = limit - len(queue)
    if remaining > 0:
        queue += [c for c in search_report.candidates if not c.is_social][:remaining]

    verified: list[VerifiedCandidate] = []
    for cand in queue:
        vc = score_candidate(verify_candidate(cand, input_hashes, embedding))
        verified.append(vc)
        emit(
            "verify:candidate",
            "ok" if vc.verified else "info",
            f"{vc.domain[:24]:<24} img {vc.image_similarity:.2f} face {vc.face_similarity:.2f} "
            f"score {vc.final_score:.2f} {'VERIFIED' if vc.verified else vc.match_type}",
        )
        if vc.verified:
            break  # one confirmed social match is what the task requires

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
        case.blockchain = ChainRecord(mode="skipped", note="--no-chain requested")
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
        emit("chain:wallet", "ok", f"{client.address} — {balance:.6f} ETH on Base Sepolia")

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
        case.blockchain = ChainRecord(mode="failed", note=str(exc)[:500])
        case.verdict = "VERIFIED_OFFCHAIN"
        case.failure_reason = f"blockchain stage failed: {str(exc)[:300]}"
        emit("chain", "fail", str(exc)[:300])

    writer.write_bundle(case, path)
    return case


def bundle_dir(case_id: str) -> Path:
    return Path(settings.evidence_dir) / case_id
