#!/usr/bin/env python3
"""Fetch a real, freely-licensed portrait to test the pipeline with.

Pulls the lead image of an English Wikipedia article via the public REST API.
Wikipedia lead images are freely licensed (CC/PD — check the file page for the
exact terms) and, for public figures, are widely reposted, which gives the
reverse-image stage a realistic chance of finding social-media copies.

    python scripts/fetch_sample.py "Sundar Pichai"
    python scripts/fetch_sample.py "Lionel Messi" --out samples/messi.jpg

Use your own photo instead if you prefer — nothing here depends on this script.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_API = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
UA = "facechain-sample-fetcher/1.0 (educational OSINT pipeline demo)"


def fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=45) as resp:
        return resp.read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("subject", help="Wikipedia article title, e.g. 'Sundar Pichai'")
    ap.add_argument("--out", default=None, help="output path (default: samples/<slug>.jpg)")
    args = ap.parse_args()

    title = urllib.parse.quote(args.subject.replace(" ", "_"), safe="")
    try:
        meta = json.loads(fetch(SUMMARY_API.format(title=title)).decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"could not reach Wikipedia: {exc}", file=sys.stderr)
        return 1

    src = (meta.get("originalimage") or meta.get("thumbnail") or {}).get("source")
    if not src:
        print(f"no lead image for {args.subject!r}", file=sys.stderr)
        return 1

    slug = args.subject.lower().replace(" ", "_")
    out = Path(args.out) if args.out else REPO_ROOT / "samples" / f"{slug}.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(fetch(src))

    print(f"subject     : {meta.get('title')}")
    print(f"source      : {src}")
    print(f"page        : {(meta.get('content_urls') or {}).get('desktop', {}).get('page', '')}")
    print(f"saved       : {out}  ({out.stat().st_size / 1024:.0f} KB)")
    print("\nLicensing: check the file page on Wikimedia Commons before redistributing.")
    print(f"Next: python pipeline.py --image {out} --simulate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
