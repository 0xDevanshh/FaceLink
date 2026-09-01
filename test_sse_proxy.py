#!/usr/bin/env python3
"""Test SSE through Vite proxy (as frontend would)."""

import asyncio
import json
import time
from pathlib import Path

import httpx


async def test_sse_through_proxy():
    """Test that SSE events flow through Vite proxy."""
    print("\n" + "="*70)
    print("TESTING SSE THROUGH VITE PROXY")
    print("="*70)
    
    # First, upload through proxy
    image_path = Path("samples/satya_nadella.jpg")
    async with httpx.AsyncClient(timeout=600) as client:
        print("\n[1] Uploading through Vite proxy (http://localhost:5173/api)...")
        with open(image_path, "rb") as f:
            files = {"image": f}
            data = {
                "engines": "yandex",
                "user_declaration": "true",
                "no_chain": "true",
            }
            try:
                resp = await client.post(
                    "http://localhost:5173/api/v1/scan",  # Via Vite proxy
                    files=files,
                    data=data,
                    timeout=30,
                )
                result = resp.json()
                case_id = result["case_id"]
                print(f"✓ Scan started via proxy: {case_id}")
            except Exception as e:
                print(f"✗ Failed to start scan through proxy: {e}")
                print("  Make sure Vite dev server is running on http://localhost:5173")
                return
        
        # Now stream SSE through proxy
        print(f"\n[2] Connecting to SSE through Vite proxy...")
        print(f"   URL: http://localhost:5173/api/v1/scan/{case_id}/events")
        
        event_count = 0
        start_time = time.time()
        stages_seen = set()
        
        try:
            async with client.stream(
                "GET", 
                f"http://localhost:5173/api/v1/scan/{case_id}/events",  # Via Vite proxy
                timeout=600,
            ) as resp:
                print(f"   Status: {resp.status_code}")
                print(f"   Headers: Content-Type={resp.headers.get('content-type')}")
                
                if resp.status_code != 200:
                    text = await resp.aread()
                    print(f"✗ SSE failed: {resp.status_code}")
                    print(f"   Response: {text[:200]}")
                    return
                
                print("   ✓ Connected!")
                print("\n[3] Receiving events through proxy:\n")
                
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        if event_count == 0:
                            print(f"   [???] Got line (no 'data:' prefix): {line[:60]}")
                        continue
                    
                    event_count += 1
                    
                    try:
                        evt = json.loads(line[5:].strip())
                        stage = evt.get("stage", "?")
                        status = evt.get("status", "?")
                        stages_seen.add(stage)
                        
                        # Print key events
                        if stage in ("input", "face", "search", "verify", "evidence", "done", "error", "ping"):
                            print(f"   [{event_count:3d}] {stage:15s} {status:8s}")
                        
                        if stage in ("done", "error"):
                            print(f"\n✓ Stream complete after {event_count} events in {time.time() - start_time:.1f}s")
                            break
                    except json.JSONDecodeError:
                        print(f"   [???] Could not parse: {line[:60]}")
        except Exception as e:
            print(f"\n✗ SSE stream error: {e}")
            print(f"   Type: {type(e).__name__}")
            return
        
        # Summary
        print("\n" + "="*70)
        print("RESULT")
        print("="*70)
        if event_count > 20:
            print(f"✓✓✓ PROXY SSE WORKING! {event_count} events received ✓✓✓")
        else:
            print(f"✗ Not enough events ({event_count}) - proxy may not be streaming properly")
        print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(test_sse_through_proxy())
