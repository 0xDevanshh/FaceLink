"""Minimal ABIs for the EAS contracts we actually call.

We talk to EAS directly with web3.py rather than shelling out to the Node EAS
SDK: one runtime, one dependency tree, and nothing hidden behind an SDK when a
judge asks what exactly was written to chain.
"""

from __future__ import annotations

ATTESTATION_REQUEST_DATA = {
    "components": [
        {"name": "recipient", "type": "address"},
        {"name": "expirationTime", "type": "uint64"},
        {"name": "revocable", "type": "bool"},
        {"name": "refUID", "type": "bytes32"},
        {"name": "data", "type": "bytes"},
        {"name": "value", "type": "uint256"},
    ],
    "name": "data",
    "type": "tuple",
}

EAS_ABI = [
    {
        "inputs": [
            {
                "components": [
                    {"name": "schema", "type": "bytes32"},
                    ATTESTATION_REQUEST_DATA,
                ],
                "name": "request",
                "type": "tuple",
            }
        ],
        "name": "attest",
        "outputs": [{"name": "", "type": "bytes32"}],
        "stateMutability": "payable",
        "type": "function",
    },
    {
        "inputs": [{"name": "uid", "type": "bytes32"}],
        "name": "getAttestation",
        "outputs": [
            {
                "components": [
                    {"name": "uid", "type": "bytes32"},
                    {"name": "schema", "type": "bytes32"},
                    {"name": "time", "type": "uint64"},
                    {"name": "expirationTime", "type": "uint64"},
                    {"name": "revocationTime", "type": "uint64"},
                    {"name": "refUID", "type": "bytes32"},
                    {"name": "recipient", "type": "address"},
                    {"name": "attester", "type": "address"},
                    {"name": "revocable", "type": "bool"},
                    {"name": "data", "type": "bytes"},
                ],
                "name": "",
                "type": "tuple",
            }
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "recipient", "type": "address"},
            {"indexed": True, "name": "attester", "type": "address"},
            {"indexed": False, "name": "uid", "type": "bytes32"},
            {"indexed": True, "name": "schemaUID", "type": "bytes32"},
        ],
        "name": "Attested",
        "type": "event",
    },
]

SCHEMA_REGISTRY_ABI = [
    {
        "inputs": [
            {"name": "schema", "type": "string"},
            {"name": "resolver", "type": "address"},
            {"name": "revocable", "type": "bool"},
        ],
        "name": "register",
        "outputs": [{"name": "", "type": "bytes32"}],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [{"name": "uid", "type": "bytes32"}],
        "name": "getSchema",
        "outputs": [
            {
                "components": [
                    {"name": "uid", "type": "bytes32"},
                    {"name": "resolver", "type": "address"},
                    {"name": "revocable", "type": "bool"},
                    {"name": "schema", "type": "string"},
                ],
                "name": "",
                "type": "tuple",
            }
        ],
        "stateMutability": "view",
        "type": "function",
    },
]
