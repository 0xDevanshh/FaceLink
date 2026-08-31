"""Hashing must be deterministic and platform-independent: the on-chain record
is worthless if a verifier on another machine computes a different digest."""

import hashlib

import numpy as np
import pytest

from facechain.evidence.hashing import (
    canonical_json,
    embedding_hash,
    hex0x,
    q3,
    sha256_bytes,
    sha256_canonical,
    sha256_file,
    sha256_text,
)


def test_sha256_matches_stdlib():
    assert sha256_bytes(b"facechain") == hashlib.sha256(b"facechain").hexdigest()
    assert sha256_text("facechain") == hashlib.sha256(b"facechain").hexdigest()


def test_sha256_file(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"\x00\x01\x02" * 1000)
    assert sha256_file(p) == hashlib.sha256(b"\x00\x01\x02" * 1000).hexdigest()


def test_canonical_json_is_key_order_independent():
    a = {"b": 1, "a": [1, 2], "c": {"z": 1, "y": 2}}
    b = {"c": {"y": 2, "z": 1}, "a": [1, 2], "b": 1}
    assert canonical_json(a) == canonical_json(b)
    assert sha256_canonical(a) == sha256_canonical(b)


def test_canonical_json_has_no_incidental_whitespace():
    assert canonical_json({"a": 1, "b": 2}) == '{"a":1,"b":2}'


def test_canonical_json_list_order_matters():
    assert sha256_canonical([1, 2]) != sha256_canonical([2, 1])


def test_embedding_hash_is_byte_order_stable():
    """Big-endian and little-endian views of the same vector must agree."""
    vec = np.linspace(-1, 1, 512).astype(np.float32)
    assert embedding_hash(vec) == embedding_hash(vec.astype(">f4"))
    assert embedding_hash(vec) == embedding_hash(vec.astype("<f4"))


def test_embedding_hash_detects_change():
    vec = np.ones(512, dtype=np.float32)
    other = vec.copy()
    other[7] = 0.999
    assert embedding_hash(vec) != embedding_hash(other)


def test_embedding_hash_shape_independent():
    vec = np.ones((1, 512), dtype=np.float32)
    assert embedding_hash(vec) == embedding_hash(vec.ravel())


@pytest.mark.parametrize(
    "value,expected", [(0.8512345, 0.851), (0.9999, 1.0), (0.0004, 0.0), (1 / 3, 0.333)]
)
def test_q3_quantisation(value, expected):
    assert q3(value) == expected


def test_q3_makes_float_hashes_reproducible():
    """0.1+0.2 != 0.3 in binary floats; quantisation must erase that."""
    assert sha256_canonical({"s": q3(0.1 + 0.2)}) == sha256_canonical({"s": q3(0.3)})


def test_hex0x_is_idempotent():
    assert hex0x("ab" * 32) == "0x" + "ab" * 32
    assert hex0x("0x" + "ab" * 32) == "0x" + "ab" * 32
