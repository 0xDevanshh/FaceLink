"""Write the evidence bundle — the local half of the tamper-evident record.

The on-chain attestation stores hashes. This bundle stores the preimages. Held
together they are checkable: anyone can re-hash these files and compare against
the attestation, and any later edit to the bundle breaks the comparison.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from ..config import settings
from ..models import AttestedPayload, Case
from .hashing import q3, sha256_canonical, sha256_file


def new_case_id(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return f"case_{now.strftime('%Y%m%d_%H%M%S')}"


class EvidenceWriter:
    def __init__(self, case_id: str, root: Path | None = None) -> None:
        self.case_id = case_id
        self.dir = Path(root or settings.evidence_dir) / case_id
        self.artifacts = self.dir / "artifacts"
        self.artifacts.mkdir(parents=True, exist_ok=True)

    # ---- primitives ------------------------------------------------------

    def path(self, name: str) -> Path:
        return self.dir / name

    def save_json(self, name: str, obj) -> Path:
        dest = self.dir / name
        payload = obj.model_dump(mode="json") if hasattr(obj, "model_dump") else obj
        dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return dest

    def save_bytes(self, name: str, data: bytes) -> Path:
        dest = self.artifacts / name
        dest.write_bytes(data)
        return dest

    def save_text(self, name: str, text: str) -> Path:
        dest = self.dir / name
        dest.write_text(text, encoding="utf-8")
        return dest

    def save_digest(self, name: str, digest: str, subject: str) -> Path:
        """A `sha256sum`-compatible line, so `shasum -c` can check it."""
        return self.save_text(name, f"{digest}  {subject}\n")

    def copy_input(self, image_path: str | Path) -> Path:
        src = Path(image_path)
        dest = self.artifacts / f"input{src.suffix.lower() or '.jpg'}"
        # Re-writing a bundle can pass the bundle's own copy back in; copying a
        # file onto itself raises, and would truncate the evidence if it didn't.
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
        return dest

    # ---- the attested payload -------------------------------------------

    @staticmethod
    def build_payload(case: Case) -> AttestedPayload:
        """Assemble exactly what gets hashed into `evidenceHash`.

        Floats are quantised so the canonical JSON — and therefore the hash —
        is reproducible on any machine.
        """
        if case.input is None or case.face is None or case.best_match is None:
            raise ValueError("cannot build attested payload from an incomplete case")

        best = case.best_match
        from .hashing import sha256_text

        selection = case.face_selection
        providers = {}
        if case.reverse_search:
            providers = {p.engine: p.status.value for p in case.reverse_search.providers}

        return AttestedPayload(
            case_id=case.case_id,
            observed_at=case.observed_at,
            input_image_sha256=case.input.sha256,
            input_image_phash=case.input.phash,
            face_embedding_sha256=case.face.embedding_sha256 or "",
            face_bbox=case.face.bbox or [],
            matched_url=best.url,
            matched_url_sha256=sha256_text(best.url),
            matched_image_sha256=best.candidate_image_sha256 or "",
            matched_image_phash=best.candidate_image_phash or "",
            search_engine=best.engine,
            social_platform=best.platform or "unknown",
            image_similarity=q3(best.image_similarity),
            face_similarity=q3(best.face_similarity),
            match_score=q3(best.final_score),
            match_type=best.match_type,
            stages_passed=[s.value for s in best.stages],
            # Anchored on-chain through `evidenceHash` rather than through the
            # schema, which stays exactly as registered.
            candidate_type=best.candidate_type.value,
            confidence_band=best.confidence_band,
            verification_rung=best.stages[-1].value if best.stages else "",
            face_selection_mode=selection.mode if selection else "auto",
            face_crop_sha256=(selection.crop_sha256 or "") if selection else "",
            provider_summary=providers,
        )

    def write_payload(self, payload: AttestedPayload) -> tuple[Path, str]:
        """Persist the payload and return its canonical SHA-256."""
        data = payload.model_dump(mode="json")
        digest = sha256_canonical(data)
        self.save_json("attested_payload.json", data)
        self.save_digest("attested_payload.sha256", digest, "attested_payload.json (canonical JSON)")
        return self.dir / "attested_payload.json", digest

    # ---- full bundle -----------------------------------------------------

    def write_bundle(self, case: Case, input_path: str | Path) -> Path:
        self.copy_input(input_path)
        self.save_json("case.json", case)

        if case.input:
            self.save_digest("input.sha256", case.input.sha256, Path(input_path).name)
        if case.face and case.face.embedding_sha256:
            self.save_digest(
                "face_embedding.sha256",
                case.face.embedding_sha256,
                f"{case.face.model} embedding ({case.face.embedding_dimension}-D, little-endian float32)",
            )
        if case.face_selection:
            # Which face, chosen how, and what the untouched original hashed to.
            self.save_json("face_selection.json", case.face_selection)
            if case.face_selection.crop_sha256:
                self.save_digest(
                    "selected_crop.sha256",
                    case.face_selection.crop_sha256,
                    "artifacts/selected_crop.png (operator-selected region)",
                )
        if case.reverse_search:
            self.save_json("reverse_search.json", case.reverse_search)
            self.save_json(
                "search_providers.json",
                {"providers": [p.rounded().model_dump(mode="json")
                               for p in case.reverse_search.providers],
                 "platform_counts": case.reverse_search.platform_counts,
                 "timed_out": case.reverse_search.timed_out},
            )
        if case.verification:
            self.save_json(
                "verification.json",
                {"candidates": [c.rounded().model_dump(mode="json") for c in case.verification]},
            )
        if case.best_match and case.best_match.candidate_image_sha256:
            self.save_digest(
                "matched_image.sha256",
                case.best_match.candidate_image_sha256,
                case.best_match.candidate_image_url or "matched image",
            )
        if case.blockchain:
            self.save_json("blockchain.json", case.blockchain)
            self.save_text("attestation.txt", self._receipt(case))
        return self.dir

    @staticmethod
    def _receipt(case: Case) -> str:
        chain = case.blockchain
        best = case.best_match
        lines = [
            "FACECHAIN VERIFICATION RECEIPT",
            "=" * 62,
            f"Case ID          : {case.case_id}",
            f"Pipeline version : {case.pipeline_version}",
            f"Verdict          : {case.verdict}",
            f"Observed at      : {case.observed_at} (unix)",
        ]

        if case.face_selection:
            sel = case.face_selection
            lines += [
                "",
                "FACE SELECTION",
                "-" * 62,
                f"Mode             : {sel.mode}"
                + (f" (face #{sel.face_index})" if sel.face_index is not None else ""),
                f"Faces offered    : {sel.faces_offered}",
                f"Original sha256  : {sel.original_sha256 or 'n/a'}",
                f"Original size    : {sel.original_width}x{sel.original_height}",
                f"Crop rect        : {sel.crop_rect or 'none — original used unmodified'}",
                f"Crop sha256      : {sel.crop_sha256 or 'n/a'}",
            ]

        if case.reverse_search and case.reverse_search.providers:
            lines += ["", "SEARCH PROVIDERS", "-" * 62]
            for p in case.reverse_search.providers:
                detail = f"{p.candidates} candidates" if p.candidates else (p.error or "—")
                lines.append(f"{p.engine:<17}: {p.status.value:<15} {p.duration_s:>6.1f}s  {detail}")
            counts = case.reverse_search.platform_counts
            if counts:
                lines.append("Platform counts  : "
                             + ", ".join(f"{k}={v}" for k, v in counts.items()))

        lines += ["", "MATCH", "-" * 62]
        if best:
            lines += [
                f"Matched URL      : {best.url}",
                f"Platform         : {best.platform or 'n/a'}",
                f"Candidate type   : {best.candidate_type.value}",
                f"Confidence band  : {best.confidence_band}",
                f"Found via        : {best.engine}",
                f"Image similarity : {best.image_similarity:.3f}",
                f"Face similarity  : {best.face_similarity:.3f}",
                f"Final score      : {best.final_score:.3f}",
                f"Stages passed    : {' -> '.join(s.value for s in best.stages)}",
            ]
        if chain:
            lines += [
                "",
                "BLOCKCHAIN",
                "-" * 62,
                f"Network          : {chain.network} (chain id {chain.chain_id})",
                f"Mode             : {chain.mode}",
                f"EAS contract     : {chain.eas_contract}",
                f"Schema UID       : {chain.schema_uid}",
                f"Attester         : {chain.attester}",
                f"Tx hash          : {chain.tx_hash or 'n/a'}",
                f"Block            : {chain.block_number or 'n/a'}",
                f"Gas used         : {chain.gas_used or 'n/a'}",
                f"Attestation UID  : {chain.attestation_uid or 'n/a'}",
                f"Read-back verify : {'PASS' if chain.readback_verified else 'FAIL/NOT RUN'}",
                f"Explorer (tx)    : {chain.explorer_tx or 'n/a'}",
                f"Explorer (EAS)   : {chain.explorer_attestation or 'n/a'}",
            ]
            if chain.readback_mismatches:
                lines += ["", "MISMATCHES", *[f"  - {m}" for m in chain.readback_mismatches]]
        lines += [
            "",
            "HOW TO VERIFY THIS RECEIPT INDEPENDENTLY",
            "-" * 62,
            "1. shasum -a 256 -c input.sha256            # input image unchanged",
            "2. python scripts/verify_attestation.py --case <this directory>",
            "   (re-hashes attested_payload.json, reads the attestation from",
            "    the chain it was written to, and compares every field)",
            "3. Or open the EAS explorer link above and compare the hashes by eye.",
            "",
            "SCOPE: this record attests that the supplied image and its primary",
            "face match the retrieved public image under the thresholds recorded",
            "in case.json. It does NOT establish a person's real-world identity.",
        ]
        return "\n".join(lines) + "\n"


def verify_bundle_integrity(case_dir: Path) -> tuple[bool, list[str]]:
    """Re-hash a bundle's own files and check them against its recorded digests."""
    problems: list[str] = []
    case_path = case_dir / "case.json"
    if not case_path.exists():
        return False, [f"missing {case_path}"]

    case = json.loads(case_path.read_text())

    payload_path = case_dir / "attested_payload.json"
    if payload_path.exists():
        recomputed = sha256_canonical(json.loads(payload_path.read_text()))
        recorded = case.get("evidence_sha256")
        if recorded and recomputed != recorded:
            problems.append(
                f"evidence hash mismatch: case.json says {recorded}, payload hashes to {recomputed}"
            )
    else:
        problems.append("missing attested_payload.json")

    inputs = list((case_dir / "artifacts").glob("input.*"))
    recorded_input = (case.get("input") or {}).get("sha256")
    if inputs and recorded_input:
        actual = sha256_file(inputs[0])
        if actual != recorded_input:
            problems.append(f"input image hash mismatch: {actual} != {recorded_input}")

    return (not problems), problems
