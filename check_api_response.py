#!/usr/bin/env python3
"""Check API response for evidence_sha256 field."""

import json
import httpx

response = httpx.get("http://127.0.0.1:8000/api/v1/scan/case_20260901_172554_243c1082/result")
data = response.json()

print("[Top-level keys in API response]")
print(json.dumps({k: (v if not isinstance(v, (dict, list)) else f'{type(v).__name__}') for k, v in data.items()}, indent=2))

print(f"\n[evidence_sha256 field]")
print(f"Value: {data.get('evidence_sha256', 'NOT_FOUND')}")

print(f"\n[evidence_hash field (alternative name)]")
print(f"Value: {data.get('evidence_hash', 'NOT_FOUND')}")
