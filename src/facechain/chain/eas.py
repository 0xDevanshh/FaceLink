"""Base Sepolia + Ethereum Attestation Service client.

Why Base Sepolia: a real, public, EVM L2 testnet (chain id 84532) with free
faucet ETH, ~2s blocks, and a working block explorer — so the record is
genuinely on a blockchain while the demo costs nothing.

Why EAS instead of a bespoke Solidity contract: EAS is a permissionless,
already-audited attestation registry deployed at the canonical predeploy
addresses on Base Sepolia. It gives exactly what the task asks for — a
tamper-evident, independently readable record — with no contract of our own to
deploy, verify or trust.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from eth_account import Account
from web3 import Web3
from web3.logs import DISCARD

from ..config import (
    BASE_SEPOLIA_CHAIN_ID,
    EAS_CONTRACT,
    EAS_SCHEMA_DEFINITION,
    EASSCAN_ATTESTATION,
    EASSCAN_SCHEMA,
    EXPLORER_TX,
    FALLBACK_RPCS,
    SCHEMA_REGISTRY_CONTRACT,
    settings,
)
from ..evidence.hashing import hex0x, sha256_text
from ..models import AttestedPayload, ChainRecord
from .abi import EAS_ABI, SCHEMA_REGISTRY_ABI
from .schema import decode_data, encode_data, schema_uid

log = logging.getLogger(__name__)

ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"
ZERO_BYTES32 = "0x" + "00" * 32
FAUCET_HINT = (
    "Fund the attester with free Base Sepolia ETH:\n"
    "  • https://portal.cdp.coinbase.com/products/faucet  (Base Sepolia, 0.1 ETH/day)\n"
    "  • https://www.alchemy.com/faucets/base-sepolia\n"
    "  • bridge Sepolia ETH via https://superbridge.app/base-sepolia"
)


# EAS reverts with 4-byte custom errors; a bare selector is useless in a demo.
EAS_ERRORS = {
    "0xbf37b20e": (
        "InvalidSchema() — the schema UID is not registered on this chain. "
        "Run `python scripts/register_schema.py` first (needs a little faucet ETH)."
    ),
    "0x08e8b937": "InvalidExpirationTime() — expiration must be 0 or in the future.",
    "0xc5723b51": "NotFound() — no such attestation/schema.",
    "0x947d5a84": "InvalidLength() — encoded schema data does not match the schema.",
    "0xbd8ba84d": "InvalidAttestation() — attestation rejected by EAS.",
    "0x4ca88867": "AccessDenied() — caller not permitted (resolver or revocation rules).",
    "0x11011294": "InsufficientValue() — the schema's resolver requires ETH with the attestation.",
    "0x157bd4c3": "Irrevocable() — schema is not revocable.",
}


class ChainError(RuntimeError):
    pass


def translate_revert(exc: Exception) -> ChainError:
    """Turn a web3 revert into something a human can act on."""
    text = str(exc)
    for selector, meaning in EAS_ERRORS.items():
        if selector in text:
            return ChainError(f"EAS reverted: {meaning}")
    return ChainError(f"{type(exc).__name__}: {text[:300]}")


class InsufficientFunds(ChainError):
    pass


class EasClient:
    def __init__(self, rpc_url: str | None = None, private_key: str | None = None) -> None:
        self.rpc_url = rpc_url or settings.base_sepolia_rpc_url
        self.w3 = self._connect(self.rpc_url)
        self.eas = self.w3.eth.contract(
            address=Web3.to_checksum_address(EAS_CONTRACT), abi=EAS_ABI
        )
        self.registry = self.w3.eth.contract(
            address=Web3.to_checksum_address(SCHEMA_REGISTRY_CONTRACT), abi=SCHEMA_REGISTRY_ABI
        )
        key = (private_key or settings.private_key or "").strip()
        self.account = Account.from_key(key) if key else None

    # ---- connection ------------------------------------------------------

    def _connect(self, primary: str) -> Web3:
        """Try the configured RPC, then public fallbacks; assert the chain id."""
        errors: list[str] = []
        for url in [primary, *[u for u in FALLBACK_RPCS if u != primary]]:
            try:
                w3 = Web3(Web3.HTTPProvider(url, request_kwargs={"timeout": 30}))
                chain_id = w3.eth.chain_id
                if chain_id != BASE_SEPOLIA_CHAIN_ID:
                    errors.append(f"{url}: wrong chain id {chain_id}")
                    continue
                if url != primary:
                    log.warning("primary RPC unusable, using fallback %s", url)
                self.rpc_url = url
                return w3
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{url}: {type(exc).__name__}: {str(exc)[:120]}")
        raise ChainError("cannot reach Base Sepolia. Tried:\n  " + "\n  ".join(errors))

    @property
    def address(self) -> str:
        if self.account is None:
            raise ChainError("PRIVATE_KEY not set — cannot sign attestations")
        return self.account.address

    def balance_eth(self) -> float:
        return float(self.w3.from_wei(self.w3.eth.get_balance(self.address), "ether"))

    def preflight(self, min_eth: float = 0.00002) -> float:
        bal = self.balance_eth()
        if bal < min_eth:
            raise InsufficientFunds(
                f"attester {self.address} holds {bal:.6f} ETH on Base Sepolia; "
                f"need ~{min_eth} ETH.\n{FAUCET_HINT}"
            )
        return bal

    # ---- schema ----------------------------------------------------------

    def resolve_schema(self, definition: str = EAS_SCHEMA_DEFINITION, register: bool = True) -> str:
        """Return our schema's UID, registering it once if it does not exist yet.

        Idempotent: the UID is derived from the definition itself, so a rerun
        finds the existing schema instead of paying to register a duplicate.
        """
        if settings.eas_schema_uid:
            uid = settings.eas_schema_uid
            if self._schema_exists(uid):
                return uid
            log.warning("configured EAS_SCHEMA_UID %s not found on chain", uid)

        uid = schema_uid(definition)
        if self._schema_exists(uid):
            log.info("schema already registered: %s", uid)
            return uid
        if not register:
            raise ChainError(f"schema {uid} is not registered (run scripts/register_schema.py)")

        log.info("registering schema %s", uid)
        receipt = self._send(
            self.registry.functions.register(definition, ZERO_ADDRESS, True),
            gas_floor=300_000,
        )
        if receipt["status"] != 1:
            raise ChainError(f"schema registration reverted (tx {receipt['transactionHash'].hex()})")
        if not self._schema_exists(uid):
            raise ChainError("schema registration mined but schema still not readable")
        return uid

    def _schema_exists(self, uid: str) -> bool:
        try:
            record = self.registry.functions.getSchema(uid).call()
        except Exception as exc:  # noqa: BLE001
            log.debug("getSchema failed: %s", exc)
            return False
        return bool(record[3])  # non-empty schema string

    # ---- attestation -----------------------------------------------------

    @staticmethod
    def build_fields(payload: AttestedPayload, evidence_hash: str) -> dict[str, Any]:
        """Map the evidence payload onto the on-chain schema fields.

        Privacy choices baked in here:
          * the face embedding is represented only by its SHA-256 — the raw
            512-D vector never leaves the machine;
          * the matched URL is stored as a hash, not plaintext, so the on-chain
            record is tamper-evident without republishing someone's profile
            link. The plaintext URL stays in the local evidence bundle, which
            is what makes the hash checkable.
        """
        case_id_hash = (
            payload.case_id if payload.case_id.startswith("0x") else sha256_text(payload.case_id)
        )
        return {
            "caseId": hex0x(case_id_hash),
            "inputImageHash": hex0x(payload.input_image_sha256),
            "faceEmbeddingHash": hex0x(payload.face_embedding_sha256),
            "matchedImageHash": hex0x(payload.matched_image_sha256),
            "matchedUrlHash": hex0x(payload.matched_url_sha256),
            "evidenceHash": hex0x(evidence_hash),
            "searchEngine": payload.search_engine,
            "socialPlatform": payload.social_platform,
            "matchScoreBps": int(round(payload.match_score * 10_000)),
            "observedAt": int(payload.observed_at),
            # The payload's own version, never the running one: re-deriving the
            # fields from an older bundle must reproduce what was attested then,
            # or the independent verifier reports a mismatch that never happened.
            "pipelineVersion": payload.pipeline_version,
        }

    def _request(self, schema: str, data: bytes, recipient: str | None) -> tuple:
        return (
            schema,
            (
                Web3.to_checksum_address(recipient or settings.attestation_recipient or ZERO_ADDRESS),
                0,              # expirationTime: never
                True,           # revocable
                ZERO_BYTES32,   # refUID
                data,
                0,              # value
            ),
        )

    def simulate_attest(self, schema: str, fields: dict[str, Any], recipient: str | None = None) -> str:
        """`eth_call` the attestation to prove the encoding is valid.

        Costs nothing and needs no funded wallet, so the full pipeline can be
        exercised end to end before the faucet ETH lands.
        """
        if not self._schema_exists(schema):
            raise ChainError(
                f"schema {schema} is not registered on Base Sepolia yet — "
                "run `python scripts/register_schema.py` (one-off, needs faucet ETH). "
                "Simulation cannot proceed without a registered schema."
            )
        req = self._request(schema, encode_data(fields), recipient)
        sender = self.account.address if self.account else ZERO_ADDRESS
        try:
            uid = self.eas.functions.attest(req).call({"from": sender})
        except ChainError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise translate_revert(exc) from exc
        return "0x" + uid.hex() if isinstance(uid, bytes) else str(uid)

    def attest(
        self, schema: str, fields: dict[str, Any], recipient: str | None = None
    ) -> ChainRecord:
        """Write the attestation on-chain and wait for its receipt."""
        data = encode_data(fields)
        req = self._request(schema, data, recipient)
        record = ChainRecord(
            eas_contract=EAS_CONTRACT,
            schema_uid=schema,
            schema_definition=EAS_SCHEMA_DEFINITION,
            attester=self.address,
            recipient=req[1][0],
            mode="onchain",
        )

        try:
            receipt = self._send(self.eas.functions.attest(req), gas_floor=400_000)
        except ChainError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise translate_revert(exc) from exc
        tx_hash = receipt["transactionHash"].hex()
        tx_hash = tx_hash if tx_hash.startswith("0x") else "0x" + tx_hash
        record.tx_hash = tx_hash
        record.block_number = int(receipt["blockNumber"])
        record.gas_used = int(receipt["gasUsed"])
        record.explorer_tx = EXPLORER_TX.format(tx=tx_hash)

        if receipt["status"] != 1:
            raise ChainError(f"attestation transaction reverted: {record.explorer_tx}")

        record.attestation_uid = self._uid_from_receipt(receipt)
        if record.attestation_uid:
            record.explorer_attestation = EASSCAN_ATTESTATION.format(uid=record.attestation_uid)
        return record

    def _uid_from_receipt(self, receipt) -> str | None:
        """Pull the attestation UID out of the EAS `Attested` event."""
        try:
            events = self.eas.events.Attested().process_receipt(receipt, errors=DISCARD)
        except Exception as exc:  # noqa: BLE001
            log.warning("could not decode Attested event: %s", exc)
            return None
        for ev in events:
            uid = ev["args"]["uid"]
            return "0x" + uid.hex() if isinstance(uid, bytes) else str(uid)
        log.warning("no Attested event in receipt %s", receipt["transactionHash"].hex())
        return None

    # ---- transaction plumbing -------------------------------------------

    def _send(self, fn, gas_floor: int = 300_000):
        """Estimate, sign (EIP-1559) and broadcast, then wait for the receipt."""
        if self.account is None:
            raise ChainError("PRIVATE_KEY not set — cannot sign transactions")

        sender = self.account.address
        try:
            gas = int(fn.estimate_gas({"from": sender}) * 1.25)
        except Exception as exc:  # noqa: BLE001
            log.warning("gas estimation failed (%s); using floor %d", exc, gas_floor)
            gas = gas_floor

        latest = self.w3.eth.get_block("latest")
        base_fee = latest.get("baseFeePerGas", self.w3.to_wei(0.001, "gwei"))
        try:
            tip = self.w3.eth.max_priority_fee
        except Exception:  # noqa: BLE001
            tip = self.w3.to_wei(0.001, "gwei")
        tip = max(int(tip), self.w3.to_wei(0.0001, "gwei"))

        tx = fn.build_transaction(
            {
                "from": sender,
                "nonce": self.w3.eth.get_transaction_count(sender),
                "chainId": BASE_SEPOLIA_CHAIN_ID,
                "gas": max(gas, 60_000),
                "maxPriorityFeePerGas": tip,
                "maxFeePerGas": int(base_fee * 2) + tip,
            }
        )

        signed = self.account.sign_transaction(tx)
        raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
        tx_hash = self.w3.eth.send_raw_transaction(raw)
        log.info("tx submitted: %s", tx_hash.hex())
        return self.w3.eth.wait_for_transaction_receipt(
            tx_hash, timeout=settings.tx_confirm_timeout_s, poll_latency=1.5
        )

    # ---- read-back verification ------------------------------------------

    def read_attestation(self, uid: str) -> dict[str, Any]:
        """Read an attestation back from the chain and decode its schema data.

        This is the step that turns "we sent a transaction" into "we created a
        record and independently verified it": the values compared afterwards
        come from a fresh contract read, not from local memory.
        """
        att = self.eas.functions.getAttestation(uid).call()
        if int.from_bytes(att[0], "big") == 0:
            raise ChainError(f"attestation {uid} not found on chain")

        # Decode against the schema the attestation itself points at, read from
        # the on-chain registry — not against an assumption baked into this code.
        schema_hex = "0x" + att[1].hex()
        definition = EAS_SCHEMA_DEFINITION
        try:
            registered = self.registry.functions.getSchema(schema_hex).call()
            if registered[3]:
                definition = registered[3]
        except Exception as exc:  # noqa: BLE001
            log.debug("could not fetch schema %s: %s", schema_hex, exc)

        return {
            "uid": "0x" + att[0].hex(),
            "schema": schema_hex,
            "schema_definition": definition,
            "time": int(att[2]),
            "expirationTime": int(att[3]),
            "revocationTime": int(att[4]),
            "refUID": "0x" + att[5].hex(),
            "recipient": att[6],
            "attester": att[7],
            "revocable": bool(att[8]),
            "decoded": decode_data(att[9], definition),
        }

    def verify_readback(
        self, uid: str, expected_fields: dict[str, Any], expected_attester: str | None = None
    ) -> tuple[bool, list[str], dict[str, Any]]:
        """Compare every on-chain field against what we intended to write."""
        onchain = self.read_attestation(uid)
        decoded = onchain["decoded"]
        mismatches: list[str] = []

        for key, expected in expected_fields.items():
            actual = decoded.get(key)
            if isinstance(expected, str) and isinstance(actual, str):
                same = expected.lower() == actual.lower()
            else:
                same = expected == actual
            if not same:
                mismatches.append(f"{key}: expected {expected!r}, on-chain {actual!r}")

        if expected_attester and onchain["attester"].lower() != expected_attester.lower():
            mismatches.append(
                f"attester: expected {expected_attester}, on-chain {onchain['attester']}"
            )
        if onchain["revocationTime"] != 0:
            mismatches.append(f"attestation was revoked at {onchain['revocationTime']}")

        return (not mismatches), mismatches, onchain

    def schema_explorer_url(self, uid: str) -> str:
        return EASSCAN_SCHEMA.format(uid=uid)


def wait_for_funding(client: EasClient, min_eth: float = 0.00002, timeout_s: int = 0) -> float:
    """Optionally block until the faucet lands, for a smoother live demo."""
    deadline = time.time() + timeout_s
    while True:
        try:
            return client.preflight(min_eth)
        except InsufficientFunds:
            if timeout_s <= 0 or time.time() > deadline:
                raise
            time.sleep(5)
