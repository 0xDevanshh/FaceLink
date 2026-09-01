#!/usr/bin/env python3
"""Inspect full result JSON."""

import json
from pathlib import Path

result_file = Path("evidence/case_20260901_172554_243c1082/case.json")
if result_file.exists():
    with open(result_file, encoding="utf-8") as f:
        case = json.load(f)
    
    print(f"Case ID: {case.get('case_id')}")
    print(f"Verdict: {case.get('verdict')}")
    print(f"Evidence hash: {case.get('evidence_hash', '(none)')}")
    print(f"Input image SHA256: {case.get('input', {}).get('sha256', '(none)')[:32]}...")
    print(f"Face encoding count: {len(case.get('faces', []))}")
    
    best = case.get('best_match')
    if best:
        print(f"\nBest match:")
        print(f"  Platform: {best.get('platform')}")
        print(f"  Raw URL: {best.get('raw_url', 'N/A')[:70]}")
        print(f"  Face similarity: {best.get('face_similarity', 0):.3f}")
        print(f"  Image similarity: {best.get('image_similarity', 0):.3f}")
        print(f"  Verification rung: {best.get('verification_rung')}")
        print(f"  Confidence band: {best.get('confidence_band')}")
    
    search_result = case.get('search', {})
    print(f"\nReverse search:")
    print(f"  Total candidates: {search_result.get('candidate_count', 0)}")
    print(f"  Social candidates: {sum(1 for c in search_result.get('results', []) if c.get('platform'))}")
    
    engines = {}
    for result in search_result.get('results', []):
        e = result.get('engine', 'unknown')
        engines[e] = engines.get(e, 0) + 1
    print(f"  Engines: {engines}")
    
    print(f"\n[Full case.json]")
    print(json.dumps(case, indent=2)[:2000])
else:
    print(f"Not found: {result_file}")
    # Try to list what evidence dirs exist
    evid_dir = Path("evidence")
    if evid_dir.exists():
        print(f"\nAvailable evidence:")
        for d in sorted(evid_dir.iterdir()):
            if d.is_dir():
                print(f"  {d.name}")
