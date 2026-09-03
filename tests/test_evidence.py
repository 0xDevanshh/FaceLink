"""The evidence bundle must be self-verifying and tamper-evident."""

import json

import pytest

from facechain.evidence.hashing import sha256_canonical
from facechain.evidence.writer import EvidenceWriter, new_case_id, verify_bundle_integrity
from facechain.models import (
    Case, EvidenceGraphReport, FaceRecord, InputImage, Stage, ThresholdSnapshot,
    VerifiedCandidate,
)


@pytest.fixture
def bundle(tmp_path):
    """A complete, verified case written to a temp evidence root."""
    img = tmp_path / "input.jpg"
    img.write_bytes(b"\xff\xd8\xff" + b"pretend-jpeg" * 500)

    from facechain.evidence.hashing import sha256_file

    case = Case(
        case_id=new_case_id(),
        created_at="2026-09-01T00:00:00+00:00",
        observed_at=1788202747,
        input=InputImage(path=str(img), filename="input.jpg", bytes_len=img.stat().st_size,
                         width=800, height=1200, sha256=sha256_file(img),
                         phash="a8a01be7a10fbed1"),
        face=FaceRecord(detected=True, backend="insightface", model="buffalo_l/SCRFD+ArcFace",
                        faces_found=1, bbox=[10, 20, 110, 140], det_score=0.89,
                        embedding_dimension=512, embedding_sha256="bb" * 32),
        best_match=VerifiedCandidate(
            engine="yandex", url="https://instagram.com/p/ABC/", domain="instagram.com",
            platform="Instagram", is_social=True, fetched=True,
            candidate_image_sha256="dd" * 32, candidate_image_phash="fafc13e1a083a6d8",
            candidate_image_url="https://cdn.example/a.jpg", candidate_image_source="og:image",
            image_similarity=0.9512, face_similarity=0.9701, metadata_consistency=0.7,
            match_type="exact-image", final_score=0.8512, verified=True,
            stages=[Stage.SEARCH_FOUND, Stage.SOCIAL_MATCH, Stage.IMAGE_MATCH,
                    Stage.FACE_MATCH, Stage.VERIFIED],
        ),
        verdict="VERIFIED",
    )
    case.verification = [case.best_match]

    writer = EvidenceWriter(case.case_id, root=tmp_path / "evidence")
    payload = writer.build_payload(case)
    _, evidence_hash = writer.write_payload(payload)
    case.evidence_sha256 = evidence_hash
    writer.write_bundle(case, img)
    return writer.dir, case, payload, evidence_hash


def test_bundle_contains_expected_files(bundle):
    d, *_ = bundle
    for name in ["case.json", "attested_payload.json", "attested_payload.sha256",
                 "input.sha256", "face_embedding.sha256", "matched_image.sha256",
                 "verification.json"]:
        assert (d / name).exists(), name
    assert (d / "artifacts" / "input.jpg").exists()


def test_bundle_integrity_passes_on_untouched_bundle(bundle):
    d, *_ = bundle
    ok, problems = verify_bundle_integrity(d)
    assert ok, problems


def test_tampering_with_the_payload_is_detected(bundle):
    """The whole point: editing the evidence must break the recorded hash."""
    d, *_ = bundle
    payload_path = d / "attested_payload.json"
    data = json.loads(payload_path.read_text())
    data["matched_url"] = "https://instagram.com/p/SOMETHING-ELSE/"
    payload_path.write_text(json.dumps(data, indent=2))

    ok, problems = verify_bundle_integrity(d)
    assert not ok
    assert any("evidence hash mismatch" in p for p in problems)


def test_tampering_with_the_input_image_is_detected(bundle):
    d, *_ = bundle
    (d / "artifacts" / "input.jpg").write_bytes(b"different bytes entirely")
    ok, problems = verify_bundle_integrity(d)
    assert not ok
    assert any("input image hash mismatch" in p for p in problems)


def test_evidence_hash_is_reproducible_from_the_written_file(bundle):
    d, _, _, evidence_hash = bundle
    reread = json.loads((d / "attested_payload.json").read_text())
    assert sha256_canonical(reread) == evidence_hash


def test_payload_quantises_floats_for_hash_stability(bundle):
    _, _, payload, _ = bundle
    assert payload.image_similarity == 0.951
    assert payload.face_similarity == 0.97
    assert payload.match_score == 0.851


def test_payload_url_hash_matches_its_plaintext(bundle):
    _, _, payload, _ = bundle
    from facechain.evidence.hashing import sha256_text

    assert payload.matched_url_sha256 == sha256_text(payload.matched_url)


def test_receipt_states_scope_and_is_human_readable(bundle):
    d, case, *_ = bundle
    from facechain.models import ChainRecord

    case.blockchain = ChainRecord(tx_hash="0x" + "ab" * 32, attestation_uid="0x" + "cd" * 32,
                                  readback_verified=True, attester="0x" + "11" * 20)
    writer = EvidenceWriter(case.case_id, root=d.parent)
    writer.write_bundle(case, d / "artifacts" / "input.jpg")

    receipt = (d / "attestation.txt").read_text()
    assert "FACECHAIN VERIFICATION RECEIPT" in receipt
    assert "does NOT establish a person's real-world identity" in receipt
    assert "Read-back verify : PASS" in receipt
    assert "https://instagram.com/p/ABC/" in receipt


def test_payload_defaults_corroboration_and_threshold_fields_when_absent(bundle):
    """The fixture's case never sets evidence_graph/threshold_snapshot — the
    payload must still build, defaulting honestly rather than crashing."""
    _, _, payload, _ = bundle
    assert payload.independent_evidence_count == 0
    assert payload.face_match_threshold == 0.0
    assert payload.calibration_status == "DEFAULT"


def test_payload_carries_evidence_graph_and_threshold_data_into_the_hash(tmp_path):
    """Phase 11: independent-evidence count and the governing thresholds are
    bound into evidenceHash, so a dispute over 'what threshold decided this'
    is answered by the attestation itself, not the current config file."""
    img = tmp_path / "input.jpg"
    img.write_bytes(b"\xff\xd8\xff" + b"pretend-jpeg" * 500)
    from facechain.evidence.hashing import sha256_file

    base_kwargs = dict(
        case_id=new_case_id(),
        created_at="2026-09-01T00:00:00+00:00",
        observed_at=1788202747,
        input=InputImage(path=str(img), filename="input.jpg", bytes_len=img.stat().st_size,
                         width=800, height=1200, sha256=sha256_file(img), phash="a8a01be7a10fbed1"),
        face=FaceRecord(detected=True, backend="insightface", model="buffalo_l/SCRFD+ArcFace",
                        faces_found=1, bbox=[10, 20, 110, 140], det_score=0.89,
                        embedding_dimension=512, embedding_sha256="bb" * 32),
        best_match=VerifiedCandidate(
            engine="yandex", url="https://instagram.com/p/ABC/", domain="instagram.com",
            platform="Instagram", is_social=True, fetched=True,
            candidate_image_sha256="dd" * 32, candidate_image_phash="fafc13e1a083a6d8",
            image_similarity=0.95, face_similarity=0.97, final_score=0.85, verified=True,
            stages=[Stage.SEARCH_FOUND, Stage.FACE_MATCH, Stage.VERIFIED],
        ),
        verdict="VERIFIED",
    )

    case_a = Case(**base_kwargs, evidence_graph=EvidenceGraphReport(independent_evidence_count=1),
                 threshold_snapshot=ThresholdSnapshot(
                     face_match_threshold=0.38, image_match_threshold=0.80,
                     verify_min_score=0.70, weight_face=0.5, weight_image=0.4, weight_meta=0.1,
                     insightface_model="buffalo_l", face_backend="insightface",
                 ))
    case_b = Case(**{**base_kwargs, "case_id": base_kwargs["case_id"]},
                 evidence_graph=EvidenceGraphReport(independent_evidence_count=3),  # different!
                 threshold_snapshot=case_a.threshold_snapshot)

    writer = EvidenceWriter(case_a.case_id, root=tmp_path / "evidence")
    payload_a = writer.build_payload(case_a)
    payload_b = writer.build_payload(case_b)

    assert payload_a.independent_evidence_count == 1
    assert payload_a.face_match_threshold == 0.38
    assert payload_a.calibration_status == "DEFAULT"

    # A different corroboration count must change the hashed payload.
    _, hash_a = writer.write_payload(payload_a)
    _, hash_b = writer.write_payload(payload_b)
    assert hash_a != hash_b


def test_build_fields_ignores_the_new_payload_fields_without_error():
    """The on-chain schema is fixed and must not gain fields just because the
    off-chain payload did — `build_fields` should map exactly the same set of
    schema fields regardless of independent_evidence_count/threshold data."""
    from facechain.chain.eas import EasClient
    from facechain.chain.schema import parse_schema
    from facechain.models import AttestedPayload

    payload = AttestedPayload(
        case_id="case_20260901_000000", observed_at=1788202747,
        input_image_sha256="aa" * 32, input_image_phash="a8a01be7a10fbed1",
        face_embedding_sha256="bb" * 32, face_bbox=[1, 2, 3, 4],
        matched_url="https://instagram.com/p/ABC/", matched_url_sha256="cc" * 32,
        matched_image_sha256="dd" * 32, matched_image_phash="fafc13e1a083a6d8",
        search_engine="yandex", social_platform="Instagram",
        image_similarity=0.75, face_similarity=0.97, match_score=0.85,
        match_type="face-only", stages_passed=["SEARCH_FOUND", "FACE_MATCH", "VERIFIED"],
        independent_evidence_count=3, face_match_threshold=0.38,
        image_match_threshold=0.80, verify_min_score=0.70, calibration_status="CALIBRATED",
    )
    fields = EasClient.build_fields(payload, "ee" * 32)
    assert set(fields) == {name for _, name in parse_schema()}


def test_build_payload_refuses_incomplete_case():
    case = Case(case_id="x", created_at="now", observed_at=1)
    with pytest.raises(ValueError, match="incomplete case"):
        EvidenceWriter.build_payload(case)


def test_verify_bundle_integrity_on_missing_dir(tmp_path):
    ok, problems = verify_bundle_integrity(tmp_path / "nope")
    assert not ok and problems
