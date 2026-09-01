#!/usr/bin/env python3
"""Test SSE streaming with real progress updates."""

import asyncio
import json
import time
from pathlib import Path

import httpx


async def test_sse_streaming():
    """Test that SSE events stream properly."""
    print("\n" + "="*70)
    print("TESTING SSE STREAMING FIX")
    print("="*70)
    
    image_path = Path("samples/satya_nadella.jpg")
    async with httpx.AsyncClient(timeout=600) as client:
        # Start scan
        print("\n[1] Uploading image and starting scan...")
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
        print(f"✓ Scan started: {case_id}")
        
        # Stream SSE events
        print(f"\n[2] Connecting to SSE stream...")
        print(f"   URL: http://127.0.0.1:8000/api/v1/scan/{case_id}/events")
        
        event_count = 0
        start_time = time.time()
        last_event_time = start_time
        stages_seen = set()
        
        try:
            async with client.stream(
                "GET", 
                f"http://127.0.0.1:8000/api/v1/scan/{case_id}/events"
            ) as resp:
                print(f"   Status: {resp.status_code}")
                
                if resp.status_code != 200:
                    print(f"✗ SSE failed: {resp.status_code}")
                    return
                
                print("   ✓ Connected!")
                print("\n[3] Receiving events:\n")
                
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    
                    event_count += 1
                    now = time.time()
                    time_since_last = now - last_event_time
                    last_event_time = now
                    
                    try:
                        evt = json.loads(line[5:].strip())
                        stage = evt.get("stage", "?")
                        status = evt.get("status", "?")
                        detail = evt.get("detail", "")[:50]
                        
                        stages_seen.add(stage)
                        
                        # Print event with timing info
                        if event_count % 5 == 0 or stage in ("input", "face", "search", "verify", "evidence", "done", "error"):
                            print(f"   [{event_count:3d}] {stage:15s} {status:8s} (+{time_since_last:.1f}s) {detail}")
                        
                        if stage in ("done", "error"):
                            print(f"\n✓ Stream complete after {event_count} events in {now - start_time:.1f}s")
                            break
                    except json.JSONDecodeError:
                        pass
        except httpx.ReadTimeout:
            print(f"\n✗ SSE stream timeout after {event_count} events")
            return
        except Exception as e:
            print(f"\n✗ SSE stream error: {e}")
            return
        
        # Get final result
        print(f"\n[4] Fetching final result...")
        resp = await client.get(f"http://127.0.0.1:8000/api/v1/scan/{case_id}/result")
        if resp.status_code == 200:
            data = resp.json()
            verdict = data.get("verdict", "UNKNOWN")
            print(f"✓ Verdict: {verdict}")
            
            best = data.get("best_match")
            if best:
                print(f"✓ Best match: {best.get('platform')} (face: {best.get('face_similarity', 0):.3f})")
        else:
            print(f"✗ Result endpoint returned {resp.status_code}")
        
        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Events received: {event_count}")
        print(f"Stages seen: {sorted(stages_seen)}")
        print(f"Total time: {time.time() - start_time:.1f}s")
        
        if event_count > 50 and "done" in stages_seen:
            print("\n✓✓✓ SSE STREAMING WORKING CORRECTLY ✓✓✓")
        else:
            print("\n✗ SSE streaming may not be working properly")
        
        print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(test_sse_streaming())
