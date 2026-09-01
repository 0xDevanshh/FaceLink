#!/usr/bin/env python3
"""Register (once) the FaceChain EAS schema on the configured testnet.

The schema UID is derived from the schema string itself, so this script is
idempotent: if the schema already exists on chain — registered by you or by
anyone else — it reports the UID and spends nothing.

    python scripts/register_schema.py            # register if needed
    python scripts/register_schema.py --check    # read-only status
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from facechain.chain.eas import ChainError, EasClient  # noqa: E402
from facechain.chain.schema import parse_schema, schema_uid  # noqa: E402
from facechain.config import EAS_SCHEMA_DEFINITION, settings  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="do not register, just report")
    args = ap.parse_args()

    uid = schema_uid()
    print("Schema definition:")
    for typ, name in parse_schema():
        print(f"  {typ:<10} {name}")
    print(f"\nDeterministic UID : {uid}")
    print(f"Network           : {settings.chain_name} (chain id {settings.chain_id})")
    print(f"Explorer          : {settings.easscan_schema(uid)}")

    try:
        client = EasClient()
    except ChainError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    print(f"RPC               : {client.rpc_url}")
    exists = client._schema_exists(uid)
    print(f"Registered        : {'yes' if exists else 'no'}")

    if exists or args.check:
        if not exists:
            print("\nRun without --check to register it.")
        else:
            print(f"\nAdd to your .env:\n  EAS_SCHEMA_UID={uid}")
        return 0

    try:
        client.preflight()
        registered = client.resolve_schema(EAS_SCHEMA_DEFINITION)
    except ChainError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

    print(f"\nRegistered schema : {registered}")
    print(f"Add to your .env:\n  EAS_SCHEMA_UID={registered}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
