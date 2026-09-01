#!/usr/bin/env python3
"""Quick test of the /api/v1/scan endpoint."""

import asyncio
import json
import time
from pathlib import Path

import httpx


async def test_scan():
    """Test the full scan API."""
    image_path = Path("samples/satya_nadella.jpg")
    if not image_path.exists():
        print(f"ERROR: {image_path} not found")
        return

    async with httpx.AsyncClient(timeout=600) as client:
        # Step 1: Upload and start scan
        print("\n[TEST] POST /api/v1/scan with real image...")
        with open(image_path, "rb") as f:
            files = {"image": f}
            data = {
                "engines": "yandex,bing,google_lens",
                "chain_mode": "skip",
                "user_declaration": "true",
                "no_chain": "true",
            }
            resp = await client.post(
                "http://127.0.0.1:8000/api/v1/scan",
                files=files,
                data=data,
            )
        
        if resp.status_code != 200:
            print(f"ERROR: {resp.status_code}")
            print(resp.text)
            return
        
        result = resp.json()
        case_id = result["case_id"]
        events_url = result["events_url"]
        status_url = result["status_url"]
        result_url = result["result_url"]
        
        print(f"✓ Case ID: {case_id}")
        print(f"  Events URL: {events_url}")
        print(f"  Status URL: {status_url}")
        print(f"  Result URL: {result_url}")
        
        # Step 2: Stream events via SSE
        print("\n[TEST] GET /api/v1/scan/{case_id}/events (SSE)...")
        async with client.stream("GET", f"http://127.0.0.1:8000{events_url}") as resp:
            if resp.status_code != 200:
                print(f"ERROR: {resp.status_code}")
                return
            
            event_count = 0
            async for line in resp.aiter_lines():
                if line.startswith("data:"):
                    event_count += 1
                    try:
                        evt = json.loads(line[5:].strip())
                        stage = evt.get("stage", "?")
                        status = evt.get("status", "?")
                        detail = evt.get("detail", "")[:60]
                        print(f"  [{event_count:02d}] {stage:15s} {status:6s}  {detail}")
                        
                        if stage in ("done", "error"):
                            break
                    except json.JSONDecodeError:
                        pass
        
        # Step 3: Poll status
        print("\n[TEST] GET /api/v1/scan/{case_id}/status...")
        resp = await client.get(f"http://127.0.0.1:8000{status_url}")
        if resp.status_code == 200:
            status_data = resp.json()
            print(f"✓ Status: {status_data['status']}")
            print(f"  Events received: {status_data['event_count']}")
            if status_data.get('error'):
                print(f"  Error: {status_data['error']}")
        
        # Step 4: Get result
        print("\n[TEST] GET /api/v1/scan/{case_id}/result...")
        for attempt in range(10):
            resp = await client.get(f"http://127.0.0.1:8000{result_url}")
            if resp.status_code == 200:
                result_data = resp.json()
                print(f"✓ Verdict: {result_data.get('verdict', 'N/A')}")
                print(f"  Best match found: {bool(result_data.get('best_match'))}")
                if result_data.get('best_match'):
                    bm = result_data['best_match']
                    print(f"    - URL: {bm.get('raw_url', 'N/A')[:70]}")
                    print(f"    - Platform: {bm.get('platform', 'N/A')}")
                    print(f"    - Face similarity: {bm.get('face_similarity', 0):.3f}")
                    print(f"    - Verification rung: {bm.get('verification_rung', 'N/A')}")
                
                # Check evidence
                evidence_hash = result_data.get('evidence_hash', '')
                print(f"  Evidence hash: {evidence_hash[:32]}..." if evidence_hash else "  Evidence hash: (none)")
                break
            elif resp.status_code == 202:
                print(f"  Still processing ({attempt + 1}/10)...")
                await asyncio.sleep(2)
            else:
                print(f"ERROR: {resp.status_code}")
                print(resp.text)
                break
        
        print("\n[RESULT] End-to-end scan completed!")


if __name__ == "__main__":
    asyncio.run(test_scan())
