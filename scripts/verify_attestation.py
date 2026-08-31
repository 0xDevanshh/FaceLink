#!/usr/bin/env python3
"""Independently verify a FaceChain record against Base Sepolia.

This script deliberately shares as little as possible with the pipeline that
produced the record: it re-reads the evidence bundle from disk, re-computes
every hash, reads the attestation straight from the EAS contract, and compares.
It never trusts `blockchain.json`'s own claims about what was verified.

    # full check of a bundle (hashes + on-chain fields + bundle integrity)
    python scripts/verify_attestation.py --case evidence/case_20260901_004512

    # just dump any attestation on Base Sepolia
    python scripts/verify_attestation.py --uid 0xabc…
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from facechain.chain.eas import ChainError, EasClient  # noqa: E402
from facechain.evidence.hashing import sha256_canonical, sha256_file, sha256_text  # noqa: E402
from facechain.evidence.writer import verify_bundle_integrity  # noqa: E402
from facechain.models import AttestedPayload  # noqa: E402

OK, BAD = "PASS", "FAIL"


def check(label: str, passed: bool, detail: str = "") -> bool:
    mark = OK if passed else BAD
    print(f"  [{mark}] {label}" + (f"  — {detail}" if detail else ""))
    return passed


def verify_case(case_dir: Path) -> int:
    case_path = case_dir / "case.json"
    payload_path = case_dir / "attested_payload.json"
    if not case_path.exists():
        print(f"no case.json in {case_dir}", file=sys.stderr)
        return 2

    case = json.loads(case_path.read_text())
    print(f"Case            : {case.get('case_id')}")
    print(f"Verdict claimed : {case.get('verdict')}")
    print(f"Pipeline        : {case.get('pipeline_version')}\n")

    results: list[bool] = []

    # ---- 1. local bundle integrity ---------------------------------------
    print("1. Local evidence bundle")
    intact, problems = verify_bundle_integrity(case_dir)
    results.append(check("bundle hashes self-consistent", intact, "; ".join(problems)))

    inputs = sorted((case_dir / "artifacts").glob("input.*"))
    if inputs:
        recomputed = sha256_file(inputs[0])
        recorded = (case.get("input") or {}).get("sha256")
        results.append(
            check("input image re-hashes to case.json value", recomputed == recorded,
                  f"{recomputed[:20]}…")
        )

    if not payload_path.exists():
        print("\nno attested_payload.json — nothing was attested for this case")
        return 0 if all(results) else 1

    payload_raw = json.loads(payload_path.read_text())
    evidence_hash = sha256_canonical(payload_raw)
    results.append(
        check("evidenceHash recomputed from payload", evidence_hash == case.get("evidence_sha256"),
              f"sha256:{evidence_hash[:24]}…")
    )
    results.append(
        check("matched URL hash matches its plaintext",
              sha256_text(payload_raw["matched_url"]) == payload_raw["matched_url_sha256"])
    )

    # ---- 2. on-chain comparison ------------------------------------------
    chain = case.get("blockchain") or {}
    uid = chain.get("attestation_uid")
    if chain.get("mode") != "onchain" or not uid:
        print(f"\n2. Blockchain: nothing to check (mode={chain.get('mode')!r})")
        return 0 if all(results) else 1

    print(f"\n2. On-chain attestation {uid}")
    try:
        client = EasClient()
        onchain = client.read_attestation(uid)
    except ChainError as exc:
        print(f"  [{BAD}] could not read attestation — {exc}")
        return 1

    print(f"  RPC: {client.rpc_url}")
    payload = AttestedPayload.model_validate(payload_raw)
    expected = EasClient.build_fields(payload, evidence_hash)
    decoded = onchain["decoded"]

    for key, want in expected.items():
        got = decoded.get(key)
        same = (
            want.lower() == got.lower()
            if isinstance(want, str) and isinstance(got, str)
            else want == got
        )
        results.append(check(f"field {key}", same, "" if same else f"want {want!r}, chain {got!r}"))

    results.append(check("attestation not revoked", onchain["revocationTime"] == 0))
    if chain.get("attester"):
        results.append(
            check("attester matches recorded signer",
                  onchain["attester"].lower() == chain["attester"].lower(),
                  onchain["attester"])
        )
    results.append(
        check("schema matches recorded schema UID",
              onchain["schema"].lower() == (chain.get("schema_uid") or "").lower(),
              onchain["schema"])
    )

    print(f"\n  block timestamp of attestation: {onchain['time']}")
    print(f"  explorer: https://base-sepolia.easscan.org/attestation/view/{uid}")

    passed = all(results)
    print(f"\n{'=' * 62}\nRESULT: {'VERIFIED — local evidence and chain agree' if passed else 'MISMATCH — see FAIL lines above'}")
    return 0 if passed else 1


def dump_uid(uid: str) -> int:
    try:
        client = EasClient()
        att = client.read_attestation(uid)
    except ChainError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    att_out = dict(att)
    print(json.dumps(att_out, indent=2, default=str))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--case", type=Path, help="path to an evidence/case_* directory")
    group.add_argument("--uid", help="attestation UID to read from Base Sepolia")
    args = ap.parse_args()
    return verify_case(args.case) if args.case else dump_uid(args.uid)


if __name__ == "__main__":
    raise SystemExit(main())
