#!/usr/bin/env python3
"""Test evidence download and verification."""

import json
import httpx

case_id = "case_20260901_172554_243c1082"

# Test evidence download
print("[TEST] GET /api/v1/scan/{case_id}/evidence (ZIP download)...")
resp = httpx.get(f"http://127.0.0.1:8000/api/v1/scan/{case_id}/evidence")

if resp.status_code == 200:
    # Check if it's a ZIP file
    if resp.content[:2] == b"PK":  # ZIP magic bytes
        print(f"✓ Evidence ZIP downloaded ({len(resp.content)} bytes)")
        
        # Try to extract it
        import io
        import zipfile
        try:
            zf = zipfile.ZipFile(io.BytesIO(resp.content))
            files = zf.namelist()
            print(f"✓ ZIP contains {len(files)} files:")
            for f in sorted(files)[:10]:
                print(f"    - {f}")
        except Exception as e:
            print(f"✗ Failed to read ZIP: {e}")
    else:
        print(f"✗ Response is not a ZIP file (first bytes: {resp.content[:20]})")
else:
    print(f"✗ Error: {resp.status_code}")
    print(resp.text)

# Test evidence verification
print("\n[TEST] POST /api/v1/verify (upload evidence ZIP)...")
import io
import zipfile

# Create a minimal ZIP to test the endpoint
zip_buffer = io.BytesIO()
with zipfile.ZipFile(zip_buffer, 'w') as zf:
    zf.writestr("dummy.txt", "test")
zip_buffer.seek(0)

files = {
    'evidence_zip': ('evidence.zip', zip_buffer.getvalue(), 'application/zip')
}

resp = httpx.post(
    "http://127.0.0.1:8000/api/v1/verify",
    files=files
)

if resp.status_code in (200, 400, 422):  # Expected responses
    print(f"✓ Verification endpoint responded: {resp.status_code}")
    try:
        result = resp.json()
        print(f"  Response: {json.dumps(result, indent=2)[:500]}")
    except:
        print(f"  Response: {resp.text[:500]}")
else:
    print(f"✗ Unexpected status: {resp.status_code}")
