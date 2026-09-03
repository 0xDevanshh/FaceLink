# FaceChain — Pipeline Technical Reference

## Overview

`src/facechain/runner.py::run()` is the pipeline. It is the single entry point both the CLI (`pipeline.py`) and the API (`server.py`) call — there is no separate "web" logic path. It takes one photo and produces a `Case` (see `models.py`): a face record, a search report, a ranked list of independently-verified candidates, a corroboration graph, a hashed evidence bundle, and — unless skipped — an on-chain attestation of that bundle's hash. It never terminates a run without writing an evidence bundle, even on early failure (`NO_FACE`, `FACE_QUALITY_INSUFFICIENT`, etc.) — every path through `run()` ends in `writer.write_bundle(case, path)`.

## End-to-End Flow

```
Input Photo
    │
    ▼
[01] Input Validation & Hashing
     - File read (downscaled to MAX_EDGE=2000px for detection; hard reject
       above max_image_edge=8000px in the quality gate)
     - SHA-256 of raw bytes, pHash/dHash/aHash (imagehash)
     - Recorded: InputImage.sha256, .phash
    │
    ▼
[02] Face Detection + Selection
     - Primary backend: InsightFace (SCRFD detector + ArcFace, buffalo_l bundle)
     - Fallback backend: OpenCV YuNet + SFace (128-D), used if InsightFace
       cannot be loaded
     - All faces detected once; an operator-supplied crop (--crop-rect) is
       applied and re-detected inside it before anything else happens
     - Auto-selected only when unambiguous: one dominant face, detector
       confidence >= AUTO_SELECT_MIN_DET_SCORE (0.60), no runner-up face
       >= 55% of the primary's area (face/selection.py)
     - Otherwise: FACE_SELECTION_REQUIRED — the pipeline stops and asks,
       rather than guessing which face the scan is about
    │
    ▼
[03] Face Quality Gate (HARD GATE — blocks the pipeline on failure)
     Hard checks (face/quality.py::gate), each a distinct QualityError:
       - IMAGE_TOO_LARGE   max edge > max_image_edge (8000px)
       - LOW_EXPOSURE / HIGH_EXPOSURE   mean luminance outside
         [quality_min_brightness=15, quality_max_brightness=240]
       - BLURRY   Laplacian variance < quality_blur_threshold (40.0)
       - NO_FACE / MULTI_FACE (per multi_face_policy: reject|largest|all)
       - FACE_TOO_SMALL   shortest face dimension < min_face_px (80px)
     Graded metrics (informational only, never gate on their own):
       overall_quality (0..1), per-metric bands (GOOD/ACCEPTABLE/POOR) for
       resolution/blur/exposure/pose/detection, plus yaw/roll pose estimate
       from 5-point landmarks. See FaceQuality in models.py.
    │
    ▼
[04] Face Embedding
     - ArcFace 512-D (or SFace 128-D on the fallback backend),
       L2-normalised float32
     - SHA-256 of the little-endian float32 bytes recorded as
       embedding_sha256 — the raw vector is never written to disk or logged
    │
    ▼
[05] Search Variant Generation (search/variants.py)
     - Query image = the operator's crop (if one was applied) or the full
       working image — cropping one person out of a group photo and then
       searching the full frame searches for the group, not the person
     - Beyond `fast` scan depth, additional crops around the selected face
       are generated and searched too: VARIANT_BUDGETS = fast:1 (original
       only), standard:2 (+tight crop), deep:3 (+loose crop)
     - Near-duplicate variants are skipped (pHash Hamming distance <= 6 to
       an already-queued variant) so a headshot-sized upload doesn't burn
       budget on a crop indistinguishable from the original
     - The original image's bytes/hash are never replaced by a variant
    │
    ▼
[06] Multi-Engine Reverse Image Search (search/orchestrator.py)
     - Engines run concurrently, ThreadPoolExecutor, search_concurrency=3
     - Each engine: hard wall-clock budget engine_timeout_s=120s
     - Whole stage: search_total_timeout_s=300s, split across the primary
       pass and any variant passes
     - Per-engine terminal ProviderStatus: COMPLETED / NO_RESULTS /
       CHALLENGED / RATE_LIMITED / TIMEOUT / FAILED / NOT_CONFIGURED —
       CHALLENGED is never silently folded into "no results"
     - Temporary public-URL publication (for engines that benefit or
       genuinely require one) is lazy — only attempted if allow_upload_host
       is on and no selected engine already has an equally reliable
       alternative — and happens at most once per scan, reused by every
       URL-based engine and every variant pass

     Browser engines (Playwright Chromium, headless):
       yandex        cbir_page=sites, .CbirSites-Items extraction
       bing          visual-search pane, .b_cit_row extraction
       google_lens   lens.google.com upload, outbound link harvest
       tineye        by-URL primary when a public URL exists, upload fallback

     API engines (no browser, require SERPAPI_KEY):
       serpapi_google_lens   SerpAPI's own direct-upload endpoint
                             (POST /image -> image_id) — never needs a
                             public URL at all
       serpapi_yandex        requires a public URL; SerpAPI has no upload
                             alternative for this engine (verified against
                             their own docs — an open, unimplemented
                             feature request on their roadmap)
       serpapi_bing          same URL-only constraint as serpapi_yandex
    │
    ▼
[07] Candidate Collection & Deduplication (search/base.py)
     - All engine results merged by canonical URL (tracking params stripped,
       redirect wrappers unwrapped)
     - Platform classified against SOCIAL_DOMAINS / PLATFORM_MEDIA_DOMAINS
       (config.py) — CDN-hosted images (media.licdn.com, pbs.twimg.com, ...)
       map back to their owning platform
     - Sorted by platform priority, then post-vs-profile, then domain
    │
    ▼
[08] Candidate Verification Queue (runner.py::_verification_queue)
     - Budget by scan_depth: fast=5, standard=12, deep=30
       (DEPTH_BUDGETS, or --max-verify to override)
     - Domain cap: MAX_PER_DOMAIN=2 — a domain with more hits gets the
       overflow appended after everything else, not discarded
     - Wider-web reservation: WIDER_WEB_BUDGET_SHARE=0.25 of the budget held
       back for non-priority platforms, so a run dominated by one priority
       platform still looks at the wider web
     - Platform priority (config.py): LinkedIn=1, Instagram=2, X/Twitter=3,
       GitHub=4, YouTube=5; other named platforms=20; unrecognised web=90
     - Total download budget: MAX_DOWNLOAD_BYTES=50MB across all candidate
       images in the scan (tracked as real bytes downloaded, not an estimate)
    │
    ▼
[09] Per-Candidate Local Verification (verification/candidate.py)
     For each queued candidate URL:
       a. SSRF check (private/loopback/link-local IPs and file:// rejected)
       b. Fetch the page itself, SSRF-safe with per-hop re-validation on
          every redirect
       c. Extract a comparable image: og:image -> twitter:image -> JSON-LD
          -> <img> tags -> engine's own thumbnail as a last resort
       d. GitHub gets a special path: the real avatar
          (avatars.githubusercontent.com), not the generated og:image
          summary card
       e. Download (3KB-25MB per image, content-type allowlisted)
       f. MediaCache: the same URL is downloaded at most once per scan
       g. Image similarity: 0.7×pHash + 0.3×dHash vs the query image
       h. Face detection on every face in the candidate image (not just the
          first/largest — group photos are handled)
       i. Best-matching face's cosine similarity recorded, along with which
          face index matched and that face's own graded quality
          (candidate_face_quality, candidate_face_bands)
       j. Metadata consistency score (0..1, cross-engine corroboration +
          URL shape + fetch success + image-source trust)
    │
    ▼
[10] Image Duplicate Clustering (verification/clustering.py)
     - pHash Hamming distance <= 6 -> same image cluster (MAX_HAMMING)
     - Every cluster counts as exactly one evidence unit regardless of how
       many URLs contain the same photo — five reposts of one picture are
       one confirmation, not five
     - Canonical member = highest (face_similarity, image_similarity) in
       the cluster
     - CorroborationSummary: image_clusters, duplicate_count,
       independent_domains, independent_platforms, verified_clusters
    │
    ▼
[11] Evidence Graph (verification/evidence_graph.py)
     - Explicit nodes (candidate/image/domain/platform) and typed edges
       (same_image, same_domain, same_platform, same_face,
       independent_source) built from the clusters above
     - independent_source only connects verified clusters sharing no
       domain — the number backing an "N independent sources" claim
     - same_face is explicitly labelled as an approximation: it means two
       candidates each independently matched the query face above
       threshold, never a literal candidate-to-candidate face comparison
       (raw embeddings are never retained to make that comparison possible)
    │
    ▼
[12] Verification Scoring & Ranking (verification/scorer.py)
     final_score = 0.50 × face_similarity
                 + 0.40 × image_similarity
                 + 0.10 × metadata_consistency

     VERIFIED requires:
       - SEARCH_FOUND (the URL came from a real engine) and
       - FACE_MATCH (face_similarity >= face_match_threshold=0.38) and
       - final_score >= verify_min_score (0.70)
       OR, only if face_only_verify_enabled (off by default):
       - FACE_MATCH and face_similarity >= face_only_verify_threshold (0.50)
         — a deliberate, opt-in accuracy tradeoff: image/metadata support is
         no longer required for a candidate accepted this way
     SOCIAL_MATCH and IMAGE_MATCH are recorded but not required — a face
     match on a personal site is not less true than one on LinkedIn, and a
     repost that crops/filters the image can still be the same, unmistakable
     face.

     Mathematical guarantee: at face_similarity=0, the remaining weights
     total 0.5 (image 0.4 + metadata 0.1) — below the 0.70 minimum. No
     amount of image/metadata support can verify a face that does not match
     (with the combined-score path; the face_only path is explicitly opt-in
     and separately gated).

     Confidence bands (on face_similarity): STRONG >=0.85, MODERATE >=0.70,
     WEAK >=0.50, else INSUFFICIENT.

     Ranking (rank()): verified first; then, at/above
     high_face_similarity_priority (0.75, ranking-only — never gates
     acceptance), face_similarity dominates; then evidential_strength
     (measured rungs only — SOCIAL_MATCH excluded); then final_score; then
     platform priority as the last tiebreaker.
    │
    ▼
[13] Evidence Bundle (evidence/writer.py)
     Written to evidence/{case_id}/:
       case.json                 full Case model
       attested_payload.json     exactly the fields hashed into evidenceHash
       attested_payload.sha256   SHA-256 of the canonical JSON
       input.sha256              input image hash (shasum -c compatible)
       face_embedding.sha256     embedding hash (vector never stored)
       face_selection.json       which face, how chosen, crop rectangle
       reverse_search.json       every candidate, every provider, every status
       search_providers.json     per-provider timing/candidate counts
       verification.json         per-candidate measurements (rounded)
       evidence_graph.json       corroboration nodes/edges/independent count
       threshold_snapshot.json   the exact thresholds/weights that governed
                                 this scan (including calibration status)
       matched_image.sha256      hash of the retrieved candidate image
       blockchain.json           on-chain record, if attested
       attestation.txt           human-readable receipt
       artifacts/input.jpg       copy of the input image
       artifacts/face_crop.png  detected face crop
       artifacts/selected_crop.png  operator crop, if one was applied

     Canonical JSON: sorted keys, no whitespace, floats quantised to 3
     decimal places for cross-machine reproducibility.
     evidenceHash = SHA-256(canonical_json(attested_payload))
     Tamper detection: re-hashing attested_payload.json must reproduce the
     recorded evidenceHash (verify_bundle_integrity(), scripts/verify_attestation.py).
    │
    ▼
[14] Blockchain Attestation (optional — --no-chain to skip, --simulate to
     validate encoding without spending gas)
     Network: Ethereum Sepolia (chain id 11155111) by default; Base Sepolia
     also supported (config.py NETWORKS). Mainnets are absent by design.
     Protocol: EAS (Ethereum Attestation Service).

     On-chain fields (11, config.py EAS_SCHEMA_DEFINITION):
       caseId, inputImageHash, faceEmbeddingHash, matchedImageHash,
       matchedUrlHash, evidenceHash, searchEngine, socialPlatform,
       matchScoreBps, observedAt, pipelineVersion

     NOT on-chain (privacy by design — only hashes go up):
       Raw face embedding (only its SHA-256)
       Plaintext matched URL (only its SHA-256)
       Input image bytes (only their SHA-256)
       independent_evidence_count / threshold values are anchored via
       evidenceHash (they're part of attested_payload.json, whose hash IS
       on-chain) rather than as separate on-chain fields — extending the
       payload never requires re-registering the schema

     Read-back verification: after the transaction confirms, the pipeline
     independently re-reads the attestation from the chain and compares
     every field against the local record before ever calling the result
     VERIFIED.

     Verdicts:
       VERIFIED            all gates passed, attested on-chain, read-back OK
       VERIFIED_OFFCHAIN   all gates passed, chain skipped or a chain-side
                           failure occurred (local evidence is never lost
                           because of a chain failure)
       VERIFIED_SIMULATED  all gates passed, --simulate used (eth_call only,
                           no gas spent)
       CHAIN_MISMATCH      attested, but the independent read-back found a
                           field that doesn't match
       UNVERIFIED          no candidate met the verification thresholds
       NO_FACE             no usable face detected
       NO_SEARCH_RESULTS   every search provider failed/challenged/returned
                           nothing
       FACE_QUALITY_INSUFFICIENT  the quality gate rejected the image
       FACE_SELECTION_REQUIRED    ambiguous — an operator must choose a face
       INVALID_CROP / INVALID_FACE_SELECTION  a supplied crop/index was
                           invalid for this image
```

## Security Controls

- **SSRF**: every candidate/image URL checked against private/loopback/link-local/CGNAT ranges before fetching, re-validated at every redirect hop (`security/ssrf.py`, `verification/candidate.py::_safe_get`)
- **Path traversal**: `case_id`/`upload_id` validated by regex before touching the filesystem (`security/paths.py`)
- **Log scrubbing**: `PRIVATE_KEY`, API keys, and raw embedding arrays redacted from every log line and SSE event (`security/scrubber.py`)
- **CORS**: explicit origin allowlist (`API_CORS_ORIGINS`), never a wildcard
- **Upload validation**: MIME allowlist plus magic-byte sniffing on every uploaded file (`server.py`)
- **Download limits**: 3KB–25MB per candidate image, 10MB per candidate page, 50MB total per scan
- **Temporary public hosting**: off by default (`ALLOW_UPLOAD_HOST=false`); when enabled, the published URL is validated (HEAD/GET, confirms real image content) before being trusted, and query strings are redacted from logs

## Verdict Reference

| Verdict | Meaning | Next step |
|---|---|---|
| `VERIFIED` | Passed every gate, attested on-chain, read-back confirmed | Nothing needed — fully attested |
| `VERIFIED_OFFCHAIN` | Passed every gate; chain skipped or failed | Evidence is complete; attest later if desired |
| `VERIFIED_SIMULATED` | Passed every gate; `--simulate` used | Re-run without `--simulate` to actually attest |
| `CHAIN_MISMATCH` | Attested, but on-chain read-back disagrees with local evidence | Investigate — do not trust this record |
| `UNVERIFIED` | No candidate met the thresholds | See `failure_reason` for the specific gate that blocked the best candidate |
| `NO_FACE` | No usable face detected | Upload a clearer image or select a face manually |
| `NO_SEARCH_RESULTS` | Every provider failed/challenged | Check provider statuses; try different engines or `--allow-upload-host` |
| `FACE_QUALITY_INSUFFICIENT` | Quality gate rejected the image | See the specific `QualityError` in `failure_reason` |
| `FACE_SELECTION_REQUIRED` | Ambiguous face selection | Choose a face index or supply a crop |
| `INVALID_CROP` / `INVALID_FACE_SELECTION` | Bad crop rectangle or face index | Correct the request parameters |

## Threshold Reference

| Setting | Default | Description | Effect of change |
|---|---|---|---|
| `face_match_threshold` | 0.38 | Minimum cosine similarity to count as `FACE_MATCH` at all | Lower = more matches reach the ladder but with less confidence; never lower this to force a specific result |
| `image_match_threshold` | 0.80 | Minimum image similarity to count as `IMAGE_MATCH`/exact-image | Not required for `VERIFIED` — informational/typing only |
| `verify_min_score` | 0.70 | Minimum combined `final_score` to reach `VERIFIED` | Lower = more candidates verify on weaker combined evidence |
| `high_face_similarity_priority` | 0.75 | Ranking-only cutoff | Never affects acceptance — only display order |
| `face_only_verify_enabled` | `false` | Enables face-similarity-alone acceptance | A real accuracy tradeoff — raises false-accept rate when on |
| `face_only_verify_threshold` | 0.50 | Face-similarity-alone acceptance bar (only used if enabled above) | Lower = more permissive face-only acceptance |
| `min_face_px` | 80 | Minimum face dimension before hard rejection | Lower = accepts smaller/lower-resolution faces |
| `quality_blur_threshold` | 40.0 | Laplacian variance floor before `BLURRY` rejection | Lower = accepts blurrier images |
