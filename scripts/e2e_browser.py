#!/usr/bin/env python3
"""Drive the real FaceChain UI in a real browser, end to end.

    python scripts/e2e_browser.py --image samples/satya_nadella.jpg --chain

This is the acceptance harness, not a unit test. It clicks the actual
application — upload, declaration, face selection, scan, SSE, result — against
a running backend and frontend, and reports the metrics the run actually
produced. It asserts nothing that it did not observe: every number printed at
the end is read back out of the case the pipeline wrote.

Prerequisites (started separately, so a failure here is never a start-up
failure in disguise):

    uvicorn server:app --host 127.0.0.1 --port 8000
    cd frontend && npm run dev
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import httpx  # noqa: E402
from playwright.sync_api import TimeoutError as PWTimeout  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

FRONTEND = "http://127.0.0.1:5173"
BACKEND = "http://127.0.0.1:8000"


def log(step: str, detail: str = "") -> None:
    print(f"  [{step}] {detail}".rstrip(), flush=True)


def run(image: Path, chain: bool, engines: list[str], shots: Path, timeout_s: int,
        draw_crop: bool = False) -> dict:
    shots.mkdir(parents=True, exist_ok=True)
    observed: dict = {"steps": {}, "console_errors": [], "sse_events": 0}

    def step(name: str, ok: bool, detail: str = "") -> None:
        observed["steps"][name] = {"pass": ok, "detail": detail}
        log(f"{'PASS' if ok else 'FAIL'} {name}", detail)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_context(viewport={"width": 1440, "height": 1000}).new_page()
        page.on("console", lambda m: observed["console_errors"].append(m.text)
                if m.type == "error" else None)

        def on_response(r) -> None:
            # The case id is taken from the scan response, not scraped from the
            # page: the Evidence view also renders the id inside a URL, and a
            # text scrape picked that up and built a malformed request from it.
            if r.request.method == "POST" and r.url.endswith("/api/v1/scan"):
                try:
                    observed["case_id"] = r.json()["case_id"]
                except Exception:  # noqa: BLE001
                    pass
            if "/events" in r.url:
                observed["sse_stream"] = r.url

        page.on("response", on_response)

        try:
            # ---- 1. load ------------------------------------------------
            page.goto(FRONTEND, wait_until="networkidle")
            step("browser_upload_view", page.is_visible("text=Face Verification Pipeline"),
                 page.title())
            page.screenshot(path=str(shots / "01-upload.png"))

            # ---- 2. choose the photo -----------------------------------
            page.set_input_files("[data-testid=file-input]", str(image))
            page.wait_for_selector("img[alt='Selected image preview']", timeout=10_000)
            step("file_selected", True, image.name)

            # ---- 3. engines --------------------------------------------
            for eng in engines:
                box = page.locator(f"input[aria-label^='Enable']").nth(0)  # noqa: F841
            for label in engines:
                loc = page.locator(f"label:has-text('{label}') input[type=checkbox]")
                if loc.count() and not loc.first.is_checked():
                    loc.first.check()
            step("engines_selected", True, ", ".join(engines))

            # ---- 4. attestation toggle ---------------------------------
            page.wait_for_selector("[data-testid=chain-readiness]", timeout=15_000)
            readiness = page.inner_text("[data-testid=chain-readiness]")
            skip = page.locator("input[aria-label='Skip blockchain attestation']")
            if chain:
                if skip.is_checked():
                    skip.uncheck()
                step("chain_enabled", not skip.is_checked(), readiness.strip()[:120])
            else:
                if not skip.is_checked():
                    skip.check()
                step("chain_enabled", True, "attestation deliberately skipped")

            # ---- 5. declaration ----------------------------------------
            page.check("[data-testid=declaration-checkbox]")
            step("declaration_accepted", True)
            page.screenshot(path=str(shots / "02-ready.png"))

            # ---- 6. start ----------------------------------------------
            page.click("[data-testid=start-scan-btn]")

            # Either face selection appears, or the scan starts directly.
            # `text=a, text=b` is NOT an OR in Playwright — it is one literal
            # string — so the two locators are combined explicitly.
            selecting = page.locator("h1:has-text('Select a face')")
            scanning = page.locator("h1:has-text('Scanning')")
            selecting.or_(scanning).first.wait_for(state="visible", timeout=180_000)
            if selecting.count() and selecting.first.is_visible():
                page.screenshot(path=str(shots / "03-face-select.png"))
                faces = page.locator("button[aria-label^='Face ']").count()
                reason = page.inner_text("h1:has-text('Select a face') + p")
                log("face selection required", f"{faces} face(s) — {reason[:90]}")

                if draw_crop:
                    # Drag a crop around the left-hand face instead of clicking
                    # a detected box, so the manual-crop path is exercised for
                    # real rather than only through the API.
                    box = page.locator("img[alt*='detected faces']").bounding_box()
                    surface = page.locator(".cursor-crosshair").first
                    x0, y0 = box["x"] + 8, box["y"] + 8
                    x1, y1 = box["x"] + box["width"] * 0.5, box["y"] + box["height"] - 8
                    surface.hover(position={"x": 8, "y": 8})
                    page.mouse.move(x0, y0)
                    page.mouse.down()
                    page.mouse.move((x0 + x1) / 2, (y0 + y1) / 2, steps=6)
                    page.mouse.move(x1, y1, steps=6)
                    page.mouse.up()
                    page.wait_for_timeout(400)
                    drawn = page.locator("text=/crop \\d+×\\d+/").count() > 0
                    page.screenshot(path=str(shots / "03b-crop-drawn.png"))
                    page.click("[data-testid=confirm-face-btn]")
                    step("face_crop_ui", drawn, "crop dragged over the left-hand face")
                    step("face_selection_ui", True, f"{faces} faces offered, crop drawn")
                else:
                    page.locator("button[aria-label^='Face ']").first.click()
                    page.click("[data-testid=confirm-face-btn]")
                    step("face_selection_ui", True, f"{faces} faces offered, first chosen")
            elif draw_crop:
                step("face_selection_ui", False,
                     "expected a selection prompt for --draw-crop but the scan started directly")
            else:
                step("face_selection_ui", True, "auto-selected (unambiguous)")

            scanning.first.wait_for(state="visible", timeout=120_000)
            step("scan_started", True)

            # ---- 7. progress -------------------------------------------
            deadline = time.time() + timeout_s
            last_shot = 0.0
            while time.time() < deadline:
                if scanning.count() == 0:
                    break
                if time.time() - last_shot > 20:
                    page.screenshot(path=str(shots / "04-progress.png"))
                    last_shot = time.time()
                page.wait_for_timeout(1000)

            chips = page.locator("[aria-label='Engine status chips'] > div")
            observed["provider_chips"] = [chips.nth(i).inner_text() for i in range(chips.count())]
            page.screenshot(path=str(shots / "04-progress.png"))

            # ---- 8. result ---------------------------------------------
            page.wait_for_selector("[aria-label='Download evidence ZIP']", timeout=timeout_s * 1000)
            verdict = page.locator("main h1").first.inner_text()
            step("result_view", True, verdict)
            page.screenshot(path=str(shots / "05-result.png"), full_page=True)
            observed["verdict_on_screen"] = verdict

            observed["provider_table"] = page.inner_text("[data-testid=provider-table]") \
                if page.locator("[data-testid=provider-table]").count() else ""
            observed["platform_counts"] = page.inner_text("[data-testid=platform-counts]") \
                if page.locator("[data-testid=platform-counts]").count() else ""

            # ---- 9. evidence view --------------------------------------
            page.click("button:has-text('View Evidence Bundle')")
            page.wait_for_timeout(1500)
            page.screenshot(path=str(shots / "06-evidence.png"), full_page=True)
            step("evidence_view", page.locator("main").count() > 0)

        except PWTimeout as exc:
            step("timeout", False, str(exc)[:200])
            page.screenshot(path=str(shots / "99-timeout.png"), full_page=True)
        finally:
            browser.close()

    return observed


def fetch_case(case_id: str) -> dict:
    r = httpx.get(f"{BACKEND}/api/v1/scan/{case_id}/result", timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_event_count(case_id: str) -> int:
    r = httpx.get(f"{BACKEND}/api/v1/scan/{case_id}/status", timeout=30)
    return r.json().get("event_count", 0) if r.status_code == 200 else 0


def report(observed: dict, case: dict) -> None:
    print("\n" + "=" * 72)
    print("REAL END-TO-END SCAN — observed values only")
    print("=" * 72)

    face = case.get("face") or {}
    sel = case.get("face_selection") or {}
    search = case.get("reverse_search") or {}
    best = case.get("best_match") or {}
    chain = case.get("blockchain") or {}
    verification = case.get("verification") or []

    def row(k, v):
        print(f"{k:<28} {v}")

    row("Case ID", case.get("case_id"))
    row("Verdict", case.get("verdict"))
    if case.get("failure_reason"):
        row("Reason", case["failure_reason"])
    print()
    row("Faces detected", face.get("faces_found"))
    row("Selected face", sel.get("face_index"))
    row("Selection mode", sel.get("mode"))
    row("Crop rect", sel.get("crop_rect") or "none (original unmodified)")
    row("Original sha256", sel.get("original_sha256"))
    row("Crop sha256", sel.get("crop_sha256") or "n/a")
    q = face.get("quality") or {}
    row("Quality gate", f"{'PASS' if q.get('passed') else 'FAIL'} "
                       f"(blur {q.get('blur_score')}, face {q.get('face_px')}px)")
    row("Face encoding", f"{face.get('embedding_dimension')}-D {face.get('model')}")
    row("Embedding sha256", face.get("embedding_sha256"))
    print()
    print("Search providers")
    for p in search.get("providers", []):
        print(f"  {p['engine']:<22} {p['status']:<16} {p['candidates']:>4} cands  "
              f"{p['duration_s']:>6.1f}s  {p.get('error','')[:60]}")
    print()
    print("Candidates by platform")
    for name, n in (search.get("platform_counts") or {}).items():
        print(f"  {name:<22} {n}")
    print()
    row("Total unique candidates", search.get("total_candidates"))
    row("Candidates measured", len(verification))
    row("Candidates fetched", sum(1 for c in verification if c.get("fetched")))
    row("With a comparable image", sum(1 for c in verification if c.get("candidate_image_sha256")))
    row("With a face detected", sum(1 for c in verification if c.get("face_detected")))
    row("Verified", sum(1 for c in verification if c.get("verified")))
    row("Rejected", sum(1 for c in verification if not c.get("verified")))
    row("Rejected by URL safety",
        sum(1 for c in verification if "SSRF" in (c.get("fetch_note") or "")))
    row("Rejected by content check",
        sum(1 for c in verification if "content-type" in (c.get("fetch_note") or "")))
    print()
    if best:
        row("Best platform", best.get("platform") or "Other Web")
        row("Best candidate type", best.get("candidate_type"))
        row("Best URL", best.get("url"))
        row("Face similarity", best.get("face_similarity"))
        row("Image similarity", best.get("image_similarity"))
        row("Confidence band", best.get("confidence_band"))
        row("Verification rung", " -> ".join(best.get("stages", [])))
    print()
    row("Evidence sha256", case.get("evidence_sha256"))
    print()
    row("Blockchain mode", chain.get("mode"))
    row("Network", f"{chain.get('network')} (chain {chain.get('chain_id')})")
    row("Attester", chain.get("attester") or "n/a")
    row("Schema UID", chain.get("schema_uid") or "n/a")
    row("Tx hash", chain.get("tx_hash") or "none")
    row("Block", chain.get("block_number") or "n/a")
    row("Gas used", chain.get("gas_used") or "n/a")
    row("Attestation UID", chain.get("attestation_uid") or "none")
    row("Read-back verified", chain.get("readback_verified"))
    row("Explorer (tx)", chain.get("explorer_tx") or "n/a")
    row("Explorer (EAS)", chain.get("explorer_attestation") or "n/a")
    if chain.get("note"):
        row("Chain note", chain["note"][:160])
    print()
    print("UI steps")
    for name, res in observed["steps"].items():
        print(f"  {'PASS' if res['pass'] else 'FAIL':<5} {name:<24} {res['detail'][:70]}")
    if observed.get("provider_chips"):
        print(f"  provider chips on screen: {observed['provider_chips']}")
    if observed.get("console_errors"):
        print(f"  console errors: {observed['console_errors'][:5]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--chain", action="store_true", help="attest on-chain for real")
    ap.add_argument("--draw-crop", action="store_true",
                    help="drag a crop in the selection UI instead of clicking a face")
    ap.add_argument("--engines", default="Yandex Images,Bing Visual Search,Google Lens")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--shots", default="/tmp/facechain_e2e")
    args = ap.parse_args()

    image = Path(args.image).resolve()
    if not image.exists():
        print(f"no such image: {image}")
        return 2

    started = time.time()
    observed = run(image, args.chain,
                   [e.strip() for e in args.engines.split(",") if e.strip()],
                   Path(args.shots), args.timeout, draw_crop=args.draw_crop)
    elapsed = time.time() - started

    case_id = observed.get("case_id")
    if not case_id:
        print("\nNo case id was observed — the UI did not reach a result.")
        print(json.dumps(observed, indent=2)[:3000])
        return 1

    case = fetch_case(case_id)
    observed["sse_events"] = fetch_event_count(case_id)
    report(observed, case)
    print(f"SSE events emitted:          {observed['sse_events']}")
    print(f"\nWall-clock (browser open → evidence view): {elapsed:.1f}s")
    print(f"Screenshots: {args.shots}")

    ok = all(s["pass"] for s in observed["steps"].values())
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
