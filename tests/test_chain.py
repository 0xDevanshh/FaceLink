"""EAS schema encoding and UID derivation.

These run offline. The encoding must be byte-identical to what the EAS SDK
produces, or the on-chain record decodes to garbage.
"""

import pytest
from eth_utils import keccak, to_bytes

from facechain.chain.eas import EAS_ERRORS, EasClient, translate_revert, ChainError
from facechain.chain.schema import decode_data, encode_data, parse_schema, schema_uid
from facechain.config import EAS_SCHEMA_DEFINITION
from facechain.models import AttestedPayload

FIELDS = {
    "caseId": "0x" + "11" * 32,
    "inputImageHash": "0x" + "22" * 32,
    "faceEmbeddingHash": "0x" + "33" * 32,
    "matchedImageHash": "0x" + "44" * 32,
    "matchedUrlHash": "0x" + "55" * 32,
    "evidenceHash": "0x" + "66" * 32,
    "searchEngine": "yandex",
    "socialPlatform": "Instagram",
    "matchScoreBps": 9123,
    "observedAt": 1788202747,
    "pipelineVersion": "1.0.0",
}


def test_parse_schema():
    fields = parse_schema()
    assert len(fields) == 11
    assert fields[0] == ("bytes32", "caseId")
    assert ("uint16", "matchScoreBps") in fields


def test_parse_schema_rejects_malformed():
    with pytest.raises(ValueError):
        parse_schema("bytes32")


def test_encode_decode_roundtrip():
    decoded = decode_data(encode_data(FIELDS))
    for key, value in FIELDS.items():
        if isinstance(value, str) and value.startswith("0x"):
            assert decoded[key].lower() == value.lower()
        else:
            assert decoded[key] == value


def test_encode_is_abi_aligned():
    """ABI encoding is 32-byte aligned, like the EAS SDK's SchemaEncoder."""
    assert len(encode_data(FIELDS)) % 32 == 0


def test_encode_rejects_missing_fields():
    partial = {k: v for k, v in FIELDS.items() if k != "evidenceHash"}
    with pytest.raises(ValueError, match="missing schema values"):
        encode_data(partial)


def test_encode_rejects_wrong_length_bytes32():
    bad = dict(FIELDS, caseId="0xdeadbeef")
    with pytest.raises(ValueError, match="bytes32"):
        encode_data(bad)


def test_schema_uid_matches_eas_formula():
    """EAS: keccak(abi.encodePacked(schema, resolver, revocable))."""
    resolver = "0x0000000000000000000000000000000000000000"
    expected = "0x" + keccak(
        EAS_SCHEMA_DEFINITION.encode() + to_bytes(hexstr=resolver) + b"\x01"
    ).hex()
    assert schema_uid() == expected


def test_schema_uid_is_deterministic_and_definition_sensitive():
    assert schema_uid() == schema_uid()
    assert schema_uid("bytes32 a") != schema_uid("bytes32 b")
    assert schema_uid(revocable=True) != schema_uid(revocable=False)


# ---- payload -> on-chain field mapping -----------------------------------

def payload() -> AttestedPayload:
    return AttestedPayload(
        case_id="case_20260901_000000",
        observed_at=1788202747,
        input_image_sha256="aa" * 32,
        input_image_phash="a8a01be7a10fbed1",
        face_embedding_sha256="bb" * 32,
        face_bbox=[1, 2, 3, 4],
        matched_url="https://instagram.com/p/ABC/",
        matched_url_sha256="cc" * 32,
        matched_image_sha256="dd" * 32,
        matched_image_phash="fafc13e1a083a6d8",
        search_engine="yandex",
        social_platform="Instagram",
        image_similarity=0.752,
        face_similarity=0.97,
        match_score=0.851,
        match_type="face-only",
        stages_passed=["SEARCH_FOUND", "SOCIAL_MATCH", "FACE_MATCH", "VERIFIED"],
    )


def test_build_fields_is_encodable_and_covers_the_schema():
    fields = EasClient.build_fields(payload(), "ee" * 32)
    assert set(fields) == {name for _, name in parse_schema()}
    encode_data(fields)  # must not raise


def test_build_fields_converts_score_to_basis_points():
    fields = EasClient.build_fields(payload(), "ee" * 32)
    assert fields["matchScoreBps"] == 8510  # 0.851 -> bps
    assert 0 <= fields["matchScoreBps"] <= 10_000


def test_build_fields_hashes_the_case_id_into_bytes32():
    fields = EasClient.build_fields(payload(), "ee" * 32)
    assert fields["caseId"].startswith("0x") and len(fields["caseId"]) == 66


def test_build_fields_never_puts_the_raw_url_or_embedding_on_chain():
    """Privacy requirement: only hashes leave the machine."""
    fields = EasClient.build_fields(payload(), "ee" * 32)
    blob = " ".join(str(v) for v in fields.values())
    assert "instagram.com" not in blob
    assert fields["faceEmbeddingHash"] == "0x" + "bb" * 32


def test_revert_translation_is_actionable():
    err = translate_revert(Exception("execution reverted: ('0xbf37b20e', '0xbf37b20e')"))
    assert isinstance(err, ChainError)
    assert "not registered" in str(err)
    assert "register_schema" in str(err)


def test_unknown_revert_still_surfaces_detail():
    assert "0xdeadbeef" in str(translate_revert(Exception("reverted 0xdeadbeef")))


def test_known_error_selectors_are_well_formed():
    for selector in EAS_ERRORS:
        assert selector.startswith("0x") and len(selector) == 10


def test_build_fields_preserves_the_payloads_own_pipeline_version():
    """Regression: re-deriving fields from an old bundle must reproduce what was
    attested then, not stamp the currently-running version."""
    old = payload()
    old.pipeline_version = "0.9.0-old"
    assert EasClient.build_fields(old, "ee" * 32)["pipelineVersion"] == "0.9.0-old"
