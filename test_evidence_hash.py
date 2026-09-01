#!/usr/bin/env python3
"""Test evidence hash in API response."""

import asyncio
import json
import time
from pathlib import Path

import httpx


async def test_evidence():
    """Test evidence hash in result."""
    image_path = Path("samples/sundar_pichai.jpg")
    
    async with httpx.AsyncClient(timeout=600) as client:
        print("Uploading second image (sundar_pichai.jpg)...")
        with open(image_path, "rb") as f:
            files = {"image": f}
            data = {
                "engines": "yandex",
                "user_declaration": "true",
                "no_chain": "true",
            }
            resp = await client.post(
                "http://127.0.0.1:8000/api/v1/scan",
                files=files,
                data=data,
            )
        
        result = resp.json()
        case_id = result["case_id"]
        print(f"Case ID: {case_id}")
        
        # Wait for completion
        result_url = f"http://127.0.0.1:8000/api/v1/scan/{case_id}/result"
        for attempt in range(30):
            resp = await client.get(result_url)
            if resp.status_code == 200:
                data = resp.json()
                
                print(f"\n[Full result.json keys]")
                print(json.dumps({k: type(v).__name__ for k, v in data.items()}, indent=2))
                
                print(f"\n[Evidence-related fields]")
                print(f"  evidence_hash: {data.get('evidence_hash', 'KEY_NOT_FOUND')}")
                print(f"  case_id: {data.get('case_id')}")
                print(f"  verdict: {data.get('verdict')}")
                
                if 'best_match' in data:
                    print(f"  best_match keys: {list(data['best_match'].keys())}")
                
                # Check the actual case.json file
                case_dir = Path(f"evidence/{case_id}")
                if case_dir.exists():
                    case_json_path = case_dir / "case.json"
                    if case_json_path.exists():
                        with open(case_json_path, encoding="utf-8") as f:
                            case_data = json.load(f)
                        print(f"\n[From case.json file]")
                        print(f"  evidence_hash (file): {case_data.get('evidence_hash', 'NOT_FOUND')}")
                        
                        # Check attested_payload.sha256
                        attested_file = case_dir / "attested_payload.sha256"
                        if attested_file.exists():
                            attested_hash = attested_file.read_text(encoding="utf-8").split()[0]
                            print(f"  attested_payload.sha256: {attested_hash}")
                
                break
            elif resp.status_code == 202:
                print(f"Still running ({attempt + 1}/30)...")
                await asyncio.sleep(2)
            else:
                print(f"ERROR: {resp.status_code}")
                break


if __name__ == "__main__":
    asyncio.run(test_evidence())
