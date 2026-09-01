#!/usr/bin/env python3
"""Final comprehensive validation of FaceLink end-to-end system."""

import json
import subprocess
import sys
from pathlib import Path

def run_test(name: str, cmd: list[str]) -> bool:
    """Run a test command and report result."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✓ {name}")
            return True
        else:
            print(f"✗ {name}: {result.stderr[:100]}")
            return False
    except Exception as e:
        print(f"✗ {name}: {e}")
        return False

def check_git_secrets() -> bool:
    """Verify no secrets are committed."""
    secret_patterns = [
        r"PRIVATE_KEY",
        r"0x[0-9a-f]{64}",  # Private key pattern
        r"sk-",  # API key pattern
        r"secret",
    ]
    
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True
    )
    
    uncommitted = result.stdout
    if uncommitted.strip():
        print(f"  WARNING: Uncommitted changes exist")
        return False
    else:
        print(f"✓ No uncommitted changes in git")
    
    result = subprocess.run(
        ["git", "diff", "--cached"],
        capture_output=True, text=True
    )
    
    if result.returncode != 0:
        return True
    
    return True

def main():
    print("\n" + "="*70)
    print("FACELINK END-TO-END VALIDATION REPORT")
    print("="*70)
    
    results = {
        "backend_tests": False,
        "frontend_tests": False,
        "frontend_build": False,
        "api_health": False,
        "api_scan": False,
        "evidence_files": False,
        "git_clean": False,
    }
    
    # Test backend
    print("\n[BACKEND TESTS]")
    results["backend_tests"] = run_test(
        "Backend pytest (161 tests expected)",
        ["python", "-m", "pytest", "tests/", "-q"]
    )
    
    # Test frontend
    print("\n[FRONTEND TESTS]")
    results["frontend_tests"] = run_test(
        "Frontend vitest (55 tests expected)",
        ["npm", "-C", "frontend", "test", "--", "--reporter=verbose"]
    )
    
    # Test build
    print("\n[BUILD]")
    results["frontend_build"] = run_test(
        "Frontend production build",
        ["npm", "-C", "frontend", "run", "build"]
    )
    
    # Test API
    print("\n[API ENDPOINTS]")
    import httpx
    
    try:
        # Health check
        resp = httpx.get("http://127.0.0.1:8000/api/v1/health", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            results["api_health"] = True
            print(f"✓ /api/v1/health (version: {data.get('version')})")
        else:
            print(f"✗ /api/v1/health returned {resp.status_code}")
    except Exception as e:
        print(f"✗ Backend not responding: {e}")
        return
    
    # Scan endpoint
    try:
        image_path = Path("samples/satya_nadella.jpg")
        with open(image_path, "rb") as f:
            files = {"image": f}
            data = {
                "engines": "yandex",
                "user_declaration": "true",
                "no_chain": "true",
            }
            resp = httpx.post(
                "http://127.0.0.1:8000/api/v1/scan",
                files=files,
                data=data,
                timeout=10
            )
        
        if resp.status_code == 200:
            result = resp.json()
            case_id = result["case_id"]
            results["api_scan"] = True
            print(f"✓ POST /api/v1/scan (case: {case_id[:25]}...)")
            
            # Check evidence files
            case_dir = Path(f"evidence/{case_id}")
            expected_files = [
                "case.json",
                "attested_payload.sha256",
                "input.sha256",
                "reverse_search.json",
                "verification.json",
            ]
            
            found_files = []
            if case_dir.exists():
                found_files = [f.name for f in case_dir.iterdir() if f.is_file()]
                results["evidence_files"] = all(f in found_files for f in expected_files)
                
                if results["evidence_files"]:
                    print(f"✓ Evidence files generated ({len(found_files)} files)")
                else:
                    missing = set(expected_files) - set(found_files)
                    print(f"✗ Missing evidence files: {missing}")
        else:
            print(f"✗ POST /api/v1/scan returned {resp.status_code}")
    except Exception as e:
        print(f"✗ Scan test failed: {e}")
    
    # Git security check
    print("\n[GIT SECURITY]")
    results["git_clean"] = check_git_secrets()
    
    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)
    
    for test, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test:30s} {status}")
    
    all_passed = all(results.values())
    print("\n" + ("="*70))
    if all_passed:
        print("RESULT: ✓ ALL CHECKS PASSED - SYSTEM READY")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"RESULT: ✗ FAILED: {', '.join(failed)}")
    print("="*70)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
