"""EAS schema handling: parse the definition, encode field values, predict UIDs."""

from __future__ import annotations

from eth_abi import encode as abi_encode
from eth_abi import decode as abi_decode
from eth_utils import keccak, to_bytes

from ..config import EAS_SCHEMA_DEFINITION


def parse_schema(definition: str = EAS_SCHEMA_DEFINITION) -> list[tuple[str, str]]:
    """`"bytes32 caseId,string url"` -> `[("bytes32","caseId"), ("string","url")]`."""
    fields: list[tuple[str, str]] = []
    for part in definition.split(","):
        tokens = part.strip().split()
        if len(tokens) != 2:
            raise ValueError(f"malformed schema field: {part!r}")
        fields.append((tokens[0], tokens[1]))
    return fields


def schema_uid(
    definition: str = EAS_SCHEMA_DEFINITION,
    resolver: str = "0x0000000000000000000000000000000000000000",
    revocable: bool = True,
) -> str:
    """Reproduce EAS's `keccak(abi.encodePacked(schema, resolver, revocable))`.

    Being able to compute this locally makes schema registration idempotent:
    we can check whether our schema already exists before spending gas.
    """
    packed = (
        definition.encode("utf-8")
        + to_bytes(hexstr=resolver)
        + (b"\x01" if revocable else b"\x00")
    )
    return "0x" + keccak(packed).hex()


def encode_data(values: dict[str, object], definition: str = EAS_SCHEMA_DEFINITION) -> bytes:
    """ABI-encode schema values in declaration order (same as EAS SchemaEncoder)."""
    fields = parse_schema(definition)
    missing = [name for _, name in fields if name not in values]
    if missing:
        raise ValueError(f"missing schema values: {missing}")

    types = [t for t, _ in fields]
    args = []
    for typ, name in fields:
        val = values[name]
        if typ == "bytes32" and isinstance(val, str):
            val = to_bytes(hexstr=val)
            if len(val) != 32:
                raise ValueError(f"{name}: bytes32 needs 32 bytes, got {len(val)}")
        args.append(val)
    return abi_encode(types, args)


def decode_data(data: bytes, definition: str = EAS_SCHEMA_DEFINITION) -> dict[str, object]:
    """Inverse of `encode_data` — used to verify what the chain actually stores."""
    fields = parse_schema(definition)
    decoded = abi_decode([t for t, _ in fields], data)
    out: dict[str, object] = {}
    for (typ, name), val in zip(fields, decoded):
        if typ == "bytes32" and isinstance(val, bytes):
            val = "0x" + val.hex()
        out[name] = val
    return out
