# FaceChain — Forensic Face Verification Pipeline

Upload a photo, and FaceChain finds real public appearances of that face through genuine reverse-image search, independently re-measures every candidate locally, and produces a tamper-evident evidence bundle — optionally attested on-chain.

## What It Does

Given one photo, the pipeline detects and embeds the primary face (InsightFace/ArcFace, 512-D), searches for it across real reverse-image engines (Yandex, Bing, Google Lens, TinEye, optionally SerpAPI), and independently re-fetches and re-measures every candidate it gets back — comparing both the image itself (perceptual hash) and the face (cosine similarity on the embedding) rather than trusting what a search engine claims. The result is a ranked, scored candidate list plus a hashed, self-verifying evidence bundle, with an optional attestation of that bundle's hash on Ethereum Sepolia via EAS.

**What it does not do:** it is not an identity database and does not claim to recognize *who* someone is. Its output is always framed as "this image and its primary face match this retrieved public image under these recorded thresholds" — never a real-world identity claim.

## Quick Start

### Requirements
- Python 3.11 or 3.12 (3.13/3.14 currently lack `onnxruntime` wheels InsightFace needs)
- Node 18+ (for the frontend)
- ~500 MB disk (InsightFace models auto-download on first run)

### Install

```bash
git clone <this-repo>
cd HH-3
pip install -r requirements.txt
playwright install chromium
cp .env.example .env
```

### Run (CLI)

```bash
python pipeline.py --image samples/sundar_pichai.jpg --no-chain
python pipeline.py --image samples/sundar_pichai.jpg --scan-depth deep --no-chain
```

### Run (Web UI)

```bash
# Terminal 1
uvicorn server:app --host 127.0.0.1 --port 8000

# Terminal 2
cd frontend && npm install && npm run dev
# Open http://localhost:5173
```

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `SERPAPI_KEY` | *(empty)* | Optional real reverse-image API (Google Lens / Yandex / Bing), used alongside the browser engines. Free tier ~100 searches/month. |
| `PRIVATE_KEY` | *(empty)* | Testnet-only signer key for EAS attestation. Never a mainnet key. |
| `ENGINES` | `yandex,bing,google_lens` | Comma-separated engines to run when none are given per-scan. |
| `FACE_MATCH_THRESHOLD` | `0.38` | Minimum cosine similarity to count as a face match at all (`FACE_MATCH` stage). |
| `VERIFY_MIN_SCORE` | `0.70` | Minimum combined `final_score` to reach `VERIFIED`. |
| `HIGH_FACE_SIMILARITY_PRIORITY` | `0.75` | Ranking-only: promotes candidates at/above this face similarity ahead of everything else verified. Does not affect acceptance. |
| `FACE_ONLY_VERIFY_ENABLED` | `false` | Opt-in: lets a candidate verify on face similarity alone (ignoring image/metadata) once it's above `FACE_ONLY_VERIFY_THRESHOLD`. A real accuracy tradeoff — off by default. |
| `ALLOW_UPLOAD_HOST` | `false` | Uploads the photo to a temporary public host (Litterbox, 1h TTL) so engines' by-URL search can be used. Off by default — this is an explicit privacy tradeoff. |
| `--scan-depth` (CLI flag, not env) | `standard` | `fast` (5 candidates) / `standard` (12) / `deep` (30, max discovery). |

See `.env.example` for the complete list, including quality-gate thresholds and blockchain network settings.

## Search Engines

| Engine | Type | Requires | Notes |
|---|---|---|---|
| `yandex` | Browser (Playwright) | — | By-URL when a public URL is available, upload flow otherwise. Historically the strongest engine for faces/social pages. |
| `bing` | Browser | — | Same by-URL/upload fallback pattern. |
| `google_lens` | Browser | — | Frequently CAPTCHA-challenged (`/sorry/` interstitial) in headless mode. |
| `tineye` | Browser | — | By-URL primary when available; exact-image provenance rather than a social-discovery engine. |
| `serpapi_google_lens` | API | `SERPAPI_KEY` | Uses SerpAPI's own direct-upload endpoint (`image_id`) — never needs a public URL. |
| `serpapi_yandex` | API | `SERPAPI_KEY` + a public URL | SerpAPI has no upload alternative for this engine (confirmed against their docs) — genuinely requires `ALLOW_UPLOAD_HOST=true` or `--image-url`. |
| `serpapi_bing` | API | `SERPAPI_KEY` + a public URL | Same URL-only constraint as `serpapi_yandex`. |

A provider's outcome is always one of `COMPLETED / NO_RESULTS / CHALLENGED / RATE_LIMITED / TIMEOUT / FAILED / NOT_CONFIGURED` — `CHALLENGED` (bot-blocked) is never silently reported as "no results", and `NOT_CONFIGURED` only means configuration is genuinely absent, never a runtime failure.

## Pipeline Stages

Input validation → face detection/selection/quality gate → face embedding → search-variant generation → multi-engine reverse search → candidate verification (image + face) → duplicate clustering → evidence-graph corroboration → scoring/ranking → evidence bundle → optional blockchain attestation → independent on-chain read-back.

Full technical detail, including the exact scoring formula and every evidence file, is in [`PIPELINE.md`](PIPELINE.md).

## Blockchain Attestation

Network: **Ethereum Sepolia** (chain id 11155111) via [EAS](https://attest.org), a permissionless, audited attestation protocol. Only hashes are ever written on-chain — the raw face embedding, the plaintext matched URL, and the input image bytes all stay local; only their SHA-256 hashes (plus `evidenceHash`, the hash of the full local evidence payload) go on-chain. After a transaction confirms, the pipeline independently reads the attestation back from the chain and compares every field against the local record before calling anything `VERIFIED`.

Configure via `.env`: `PRIVATE_KEY` (testnet-only signer), `NETWORK` (`ethereum-sepolia` by default). Skip entirely with `--no-chain`, or validate the encoding without spending gas with `--simulate`.

## Project Structure

```
.
├── pipeline.py              CLI entrypoint
├── server.py                FastAPI backend (same pipeline, for the UI)
├── src/facechain/
│   ├── runner.py             the pipeline itself (image → ... → attestation)
│   ├── config.py             central Settings, network/platform tables
│   ├── models.py             Case/evidence Pydantic schema
│   ├── face/                 detection, quality, selection, embedding
│   ├── search/                per-engine adapters + orchestrator + variants
│   ├── verification/          candidate scoring, clustering, evidence graph
│   ├── evidence/               hashing + bundle writer
│   └── chain/                  EAS attestation client
├── frontend/                React/Vite UI
├── scripts/                  register_schema.py, verify_attestation.py, fetch_sample.py, check_network.py
├── samples/                  sample photos used by tests
├── tests/                    pytest suite
└── evidence/                 generated case bundles (gitignored)
```

## Tests

```bash
python -m pytest tests/ -q
```

## License

See [`LICENSE`](LICENSE).
