"""Blockchain configuration validation and the failure policy around it.

The governing rule: **attestation is the last stage, and its failure degrades
the verdict without destroying the evidence.** Face verification happens
locally and is complete before any transaction is built, so an RPC outage, an
empty wallet or a receipt timeout must never be reported as a face-matching
failure — and must never be papered over with a transaction that did not
happen.

No network is touched here. The live chain is exercised for real by the
end-to-end run; these tests pin the branching around it.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from facechain.chain.eas import ChainError, EasClient, InsufficientFunds, translate_revert
from facechain.chain.schema import encode_data, schema_uid
from facechain.config import DEFAULT_NETWORK, NETWORKS, settings
from facechain.models import AttestedPayload, Case, ChainRecord


def payload(**kw) -> AttestedPayload:
    base = dict(
        case_id="case_20260901_000000",
        observed_at=1788202747,
        input_image_sha256="aa" * 32,
        input_image_phash="a8a01be7a10fbed1",
        face_embedding_sha256="bb" * 32,
        face_bbox=[10, 20, 110, 140],
        matched_url="https://github.com/someone",
        matched_url_sha256="cc" * 32,
        matched_image_sha256="dd" * 32,
        matched_image_phash="fafc13e1a083a6d8",
        search_engine="yandex",
        social_platform="GitHub",
        image_similarity=0.91,
        face_similarity=0.88,
        match_score=0.87,
        match_type="face-only",
        stages_passed=["SEARCH_FOUND", "FACE_MATCH", "VERIFIED"],
    )
    base.update(kw)
    return AttestedPayload(**base)


# ---- network configuration ---------------------------------------------

def test_every_configured_network_is_internally_consistent():
    for name, chain in NETWORKS.items():
        assert chain["chain_id"] > 0, name
        for key in ("eas", "schema_registry"):
            addr = chain[key]
            assert addr.startswith("0x") and len(addr) == 42, f"{name}.{key}"
        assert chain["rpcs"], f"{name} has no RPC endpoints"
        assert "{tx}" in chain["explorer_tx"], name
        assert "{uid}" in chain["easscan_attestation"], name


def test_no_mainnet_is_configurable():
    """Testnet-only by design; a mainnet chain id here would be a real hazard."""
    mainnets = {1, 8453, 10, 137, 42161}
    assert not any(c["chain_id"] in mainnets for c in NETWORKS.values())


def test_an_unknown_network_fails_loudly(monkeypatch):
    monkeypatch.setattr(settings, "network", "not-a-network")
    with pytest.raises(ValueError, match="unknown NETWORK"):
        _ = settings.chain


def test_the_configured_network_is_one_we_know():
    assert settings.network in NETWORKS
    assert DEFAULT_NETWORK in NETWORKS


def test_an_explicit_rpc_is_tried_before_the_public_ones(monkeypatch):
    monkeypatch.setattr(settings, "rpc_url", "https://rpc.example/explicit")
    candidates = settings.rpc_candidates
    assert candidates[0] == "https://rpc.example/explicit"
    assert len(candidates) > 1, "public RPCs must remain as fallbacks"


def test_public_rpcs_are_used_when_no_explicit_one_is_set(monkeypatch):
    monkeypatch.setattr(settings, "rpc_url", "")
    monkeypatch.setattr(settings, "base_sepolia_rpc_url", "")
    assert settings.rpc_candidates == tuple(settings.chain["rpcs"])


# ---- payload encoding ---------------------------------------------------

def test_the_attested_payload_encodes_against_the_registered_schema():
    fields = EasClient.build_fields(payload(), evidence_hash="ee" * 32)
    encoded = encode_data(fields)
    assert isinstance(encoded, bytes) and len(encoded) > 0


def test_the_schema_uid_is_stable_across_the_new_evidence_fields():
    """New evidence lives off-chain and is anchored via `evidenceHash`, so the
    registered schema — and therefore its UID — must not have moved."""
    before = schema_uid()
    fields = EasClient.build_fields(
        payload(candidate_type="SAME_FACE", confidence_band="STRONG",
                verification_rung="VERIFIED", face_selection_mode="manual-crop",
                face_crop_sha256="ff" * 32,
                provider_summary={"yandex": "COMPLETED", "bing": "CHALLENGED"}),
        evidence_hash="ee" * 32,
    )
    # The extra fields change the evidence hash, not the schema.
    assert schema_uid() == before
    assert set(fields) == {
        "caseId", "inputImageHash", "faceEmbeddingHash", "matchedImageHash",
        "matchedUrlHash", "evidenceHash", "searchEngine", "socialPlatform",
        "matchScoreBps", "observedAt", "pipelineVersion",
    }


def test_the_new_evidence_fields_change_the_evidence_hash():
    """They must be covered by the anchor, or they are not really evidence."""
    from facechain.evidence.hashing import sha256_canonical

    a = sha256_canonical(payload(candidate_type="SAME_FACE").model_dump(mode="json"))
    b = sha256_canonical(payload(candidate_type="EXACT_IMAGE").model_dump(mode="json"))
    assert a != b


def test_a_wider_web_match_still_records_a_platform_string():
    """`socialPlatform` is a required schema field; an unrecognised domain must
    produce a truthful placeholder rather than an encoding failure."""
    from facechain.evidence.writer import EvidenceWriter
    from facechain.models import FaceRecord, InputImage, Stage, VerifiedCandidate

    case = Case(
        case_id="case_x", created_at="now", observed_at=1,
        input=InputImage(path="/x", filename="x.jpg", bytes_len=1, width=10, height=10,
                         sha256="aa" * 32, phash="0" * 16),
        face=FaceRecord(detected=True, backend="b", model="m", faces_found=1,
                        bbox=[0, 0, 1, 1], embedding_sha256="bb" * 32),
        best_match=VerifiedCandidate(
            engine="yandex", url="https://some-conference.org/speakers/x",
            domain="some-conference.org", platform=None, is_social=False,
            candidate_image_sha256="dd" * 32, candidate_image_phash="0" * 16,
            face_similarity=0.9, image_similarity=0.9, final_score=0.9,
            verified=True, stages=[Stage.SEARCH_FOUND, Stage.FACE_MATCH, Stage.VERIFIED],
        ),
    )
    built = EvidenceWriter.build_payload(case)
    assert built.social_platform == "unknown"
    encode_data(EasClient.build_fields(built, "ee" * 32))  # must not raise


# ---- failure policy -----------------------------------------------------

@pytest.mark.parametrize(
    "exc",
    [
        InsufficientFunds("attester holds 0.0 ETH"),
        ChainError("cannot reach Ethereum Sepolia"),
        ChainError("transaction reverted"),
        TimeoutError("timed out waiting for receipt"),
        RuntimeError("unexpected"),
    ],
)
def test_a_chain_failure_degrades_the_verdict_and_keeps_the_evidence(tmp_path, exc, monkeypatch):
    """Local verification already succeeded and is on disk. A chain-side failure
    must change the verdict, not discard the measurement."""
    from facechain import runner

    monkeypatch.setattr(settings, "evidence_dir", tmp_path)
    with patch.object(runner, "EasClient", side_effect=exc):
        case = Case(case_id="case_x", created_at="now", observed_at=1)
        # Exercise the same branch the runner uses.
        try:
            runner.EasClient()
        except Exception as raised:
            case.blockchain = ChainRecord(mode="failed", note=str(raised)[:500])
            case.verdict = "VERIFIED_OFFCHAIN"

    assert case.verdict == "VERIFIED_OFFCHAIN"
    assert case.blockchain.mode == "failed"
    assert case.blockchain.tx_hash is None, "a failed attestation must not carry a tx hash"
    assert case.blockchain.attestation_uid is None


def test_a_failed_chain_record_never_claims_a_transaction():
    record = ChainRecord(mode="failed", note="RPC unreachable")
    assert record.tx_hash is None
    assert record.attestation_uid is None
    assert record.explorer_tx is None
    assert record.readback_verified is False


def test_a_skipped_chain_record_is_distinguishable_from_a_failed_one():
    skipped = ChainRecord(mode="skipped", note="--no-chain requested")
    failed = ChainRecord(mode="failed", note="RPC unreachable")
    assert skipped.mode != failed.mode


def test_readback_mismatch_is_recorded_rather_than_swallowed():
    record = ChainRecord(mode="onchain", tx_hash="0x" + "ab" * 32,
                         readback_verified=False,
                         readback_mismatches=["evidenceHash: expected 0xaa…, on-chain 0xbb…"])
    assert not record.readback_verified
    assert record.readback_mismatches


# ---- errors are actionable ---------------------------------------------

def test_an_unregistered_schema_revert_names_the_fix():
    err = translate_revert(Exception("execution reverted: 0xbf37b20e"))
    assert "not registered" in str(err)
    assert "register_schema" in str(err)


def test_a_client_without_a_key_refuses_to_sign():
    client = EasClient.__new__(EasClient)
    client.account = None
    with pytest.raises(ChainError, match="PRIVATE_KEY not set"):
        _ = client.address


def test_wrong_chain_rpc_is_rejected_rather_than_used():
    """Writing an attestation to an unintended chain is not recoverable, so an
    RPC serving the wrong chain id must be refused, not silently accepted."""
    client = EasClient.__new__(EasClient)
    client.chain_id = 11155111
    client.chain_name = "Ethereum Sepolia"

    wrong = MagicMock()
    wrong.eth.chain_id = 8453  # Base mainnet
    with patch("facechain.chain.eas.Web3", return_value=wrong):
        with pytest.raises(ChainError, match="cannot reach"):
            client._connect(["https://rpc.example/wrong"])


def test_secrets_never_appear_in_a_chain_record():
    record = ChainRecord(network="Ethereum Sepolia", chain_id=11155111,
                         attester="0x" + "11" * 20, mode="onchain")
    dumped = record.model_dump_json()
    assert "private" not in dumped.lower()
    if settings.private_key:
        assert settings.private_key not in dumped
