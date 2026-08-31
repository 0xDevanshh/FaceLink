"""Deterministic hashing helpers.

Every hash that ends up on-chain is produced here, so that an independent
verifier can reproduce it from the evidence bundle with no ambiguity.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def canonical_json(obj: Any) -> str:
    """RFC-8785-ish canonical form: sorted keys, no insignificant whitespace.

    Floats are *not* allowed to reach this function unrounded — the caller must
    quantise them (see `q3`) so that re-serialisation on another machine cannot
    drift the hash.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_canonical(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


def q3(value: float) -> float:
    """Quantise a float to 3 decimals for hash-stable serialisation."""
    return float(f"{float(value):.3f}")


def embedding_hash(embedding: np.ndarray) -> str:
    """Hash an embedding in a byte-exact, platform-stable way.

    The vector is cast to little-endian float32 before hashing, so the digest
    does not depend on the host's native byte order or on numpy's default
    dtype. The raw embedding itself never leaves the machine.
    """
    vec = np.asarray(embedding, dtype="<f4").ravel()
    return sha256_bytes(vec.tobytes())


def hex0x(digest: str) -> str:
    """`deadbeef...` -> `0xdeadbeef...` (EAS bytes32 wants the 0x form)."""
    return digest if digest.startswith("0x") else "0x" + digest
