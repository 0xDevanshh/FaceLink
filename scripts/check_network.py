#!/usr/bin/env python3
"""Preflight the chain configuration before spending any gas.

Motivated by a real incident during development: the EAS addresses in config
were edited to `0x0000…0021` / `0x0000…0020` (the Base predeploys with the
`42` prefix zeroed). Those addresses hold no code on any chain — and a call to
an address with no code does not revert, so the attestation would have burned
gas and recorded nothing while looking like it worked.

    python scripts/check_network.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from facechain.chain.eas import ChainError, EasClient  # noqa: E402
from facechain.chain.schema import schema_uid  # noqa: E402
from facechain.config import NETWORKS, settings  # noqa: E402

OK, BAD, WARN = "PASS", "FAIL", "WARN"


def line(mark: str, label: str, detail: str = "") -> None:
    print(f"  [{mark}] {label}" + (f"  — {detail}" if detail else ""))


def main() -> int:
    print(f"Configured network : {settings.network}")
    if settings.network not in NETWORKS:
        line(BAD, "unknown network", f"choose one of {sorted(NETWORKS)}")
        return 1

    chain = settings.chain
    print(f"Name               : {chain['name']} (chain id {chain['chain_id']})")
    print(f"EAS                : {chain['eas']}")
    print(f"SchemaRegistry     : {chain['schema_registry']}")
    print(f"RPC candidates     : {', '.join(settings.rpc_candidates)}\n")

    print("Connectivity and contracts")
    try:
        # The client asserts chain id and contract bytecode on construction.
        client = EasClient()
    except ChainError as exc:
        line(BAD, "could not initialise chain client", str(exc))
        return 1

    line(OK, "RPC serves the expected chain", f"{client.rpc_url} → {client.chain_id}")
    line(OK, "EAS has contract code", client.eas_address)
    line(OK, "SchemaRegistry has contract code", client.registry_address)

    print("\nSchema")
    uid = settings.eas_schema_uid or schema_uid()
    registered = client._schema_exists(uid)
    line(OK if registered else WARN, f"schema {uid[:18]}…",
         "registered" if registered else "not registered yet — run scripts/register_schema.py")
    print(f"       {settings.easscan_schema(uid)}")

    print("\nAttester")
    if client.account is None:
        line(BAD, "PRIVATE_KEY not set in .env")
        return 1
    balance = client.balance_eth()
    funded = balance >= 0.00002
    line(OK if funded else BAD, f"{client.address}", f"{balance:.6f} ETH")
    if not funded:
        print(f"\n{settings.faucet_hint}")
        return 1

    ready = registered and funded
    print(f"\n{'=' * 62}")
    print("READY — a full run can attest on chain." if ready
          else "NOT READY — register the schema first (see above).")
    return 0 if ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
