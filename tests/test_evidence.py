"""The evidence bundle must be self-verifying and tamper-evident."""

import json

import pytest

from facechain.evidence.hashing import sha256_canonical
from facechain.evidence.writer import EvidenceWriter, new_case_id, verify_bundle_integrity
from facechain.models import Case, FaceRecord, InputImage, Stage, VerifiedCandidate


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


def test_build_payload_refuses_incomplete_case():
    case = Case(case_id="x", created_at="now", observed_at=1)
    with pytest.raises(ValueError, match="incomplete case"):
        EvidenceWriter.build_payload(case)


def test_verify_bundle_integrity_on_missing_dir(tmp_path):
    ok, problems = verify_bundle_integrity(tmp_path / "nope")
    assert not ok and problems
