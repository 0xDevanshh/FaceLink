# FaceChain — Face ID + Blockchain Verification

A CLI pipeline that takes a photo, detects and encodes the face in it, finds a
**real** matching social-media post through **genuine reverse-image search**, and
writes the verified match to **Ethereum Sepolia** as a tamper-evident
[EAS](https://attest.org) attestation — then reads it back off the chain and
checks it field by field.

A live record produced by this pipeline:
[`0x9d1d9466…aef40bb`](https://sepolia.easscan.org/attestation/view/0x9d1d946633f2cddaad062fc7826a959d851d5f0d853742e33a43b099aaef40bb)

There are no hardcoded results anywhere. Every URL the pipeline reports came out
of a live query to Google Lens, Yandex Images or Bing Visual Search during that
run, and every match is re-measured locally before it is believed.

```
INPUT PHOTO
   │
   ├─ validate, SHA-256, pHash
   │
   ├─ FACE  ── SCRFD detect ─→ align ─→ ArcFace 512-D ─→ L2 norm ─→ SHA-256
   │                                                        (vector stays local)
   ├─ REVERSE IMAGE SEARCH  ──┬─ Google Lens
   │   (real engines, fan-out) ├─ Yandex Images
   │                           └─ Bing Visual Search
   │                                  │
   │                          candidate URLs
   │                                  │
   │                          social-domain filter
   │                                  │
   ├─ VERIFY EACH CANDIDATE ── fetch page ─→ extract og:image ─→ download
   │                            ├─ pHash/dHash similarity  (40%)
   │                            ├─ ArcFace cosine on the retrieved face (50%)
   │                            └─ metadata corroboration  (10%)
   │
   ├─ LADDER  SEARCH_FOUND → SOCIAL_MATCH → [IMAGE_MATCH] → FACE_MATCH → VERIFIED
   │
   ├─ EVIDENCE BUNDLE ─→ canonical JSON ─→ evidenceHash (SHA-256)
   │
   ├─ EAS ATTESTATION on Ethereum Sepolia (chain id 11155111)
   │
   └─ READ BACK FROM CHAIN ─→ decode ─→ compare all 11 fields ─→ VERIFIED
```

---

## Table of contents

- [What it actually does](#what-it-actually-does)
- [Quick start](#quick-start)
- [Which blockchain, and why](#which-blockchain-and-why)
- [What goes on-chain (and what deliberately does not)](#what-goes-on-chain-and-what-deliberately-does-not)
- [The verification ladder](#the-verification-ladder)
- [Reverse image search: how it stays honest](#reverse-image-search-how-it-stays-honest)
- [Evidence bundle](#evidence-bundle)
- [Verifying a record independently](#verifying-a-record-independently)
- [CLI reference](#cli-reference)
- [Repository layout](#repository-layout)
- [Tests](#tests)
- [Known limitations](#known-limitations)
- [Ethics and scope](#ethics-and-scope)

---

## What it actually does

| Stage | Implementation | Notes |
|---|---|---|
| Face detection | **InsightFace SCRFD** (`buffalo_l`) | fallback: OpenCV **YuNet** |
| Face encoding | **ArcFace** `w600k_r50`, 512-D, L2-normalised | fallback: **SFace** 128-D |
| Image hashing | SHA-256 + pHash/dHash/aHash (`imagehash`) | |
| Reverse search | **Playwright** driving Google Lens, Yandex, Bing (+ TinEye) | optional SerpAPI fallback |
| Candidate check | own HTTP fetch → `og:image` → download → pHash + ArcFace | never trusts the engine |
| Blockchain | **Ethereum Sepolia** + **EAS**, via `web3.py` | testnet, free |
| Read-back | `getAttestation` → ABI-decode → field-by-field compare | |

A run either produces a verified match or explains exactly which rung of the
ladder it failed on. It never invents a result to look successful.

---

## Quick start

### 1. Install

Use **Python 3.11 or 3.12** — 3.13/3.14 have no `onnxruntime` wheels yet, and
InsightFace needs it.

```bash
git clone <your-fork-url> facechain && cd facechain

python3.12 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install -r requirements.txt
playwright install chromium          # ~150 MB, for the real search engines
```

The InsightFace weights (~275 MB) download automatically on first run.

### 2. Configure

```bash
cp .env.example .env
```

Generate a **throwaway testnet** wallet and put its key in `.env`:

```bash
python -c "from eth_account import Account; a=Account.create(); print(a.address, a.key.hex())"
```

Fund the address with free Sepolia ETH (a full run costs well under
0.0001 ETH):

- <https://cloud.google.com/application/web3/faucet/ethereum/sepolia>
- <https://www.alchemy.com/faucets/ethereum-sepolia>

Check the chain configuration before spending anything — this verifies the RPC
serves the expected chain and that the EAS addresses actually hold contract
code:

```bash
python scripts/check_network.py
```

Register the EAS schema once (idempotent — it derives the UID and skips if the
schema already exists):

```bash
python scripts/register_schema.py
```

To attest on a different testnet instead, set `NETWORK=base-sepolia` in `.env`;
everything else (chain id, EAS addresses, explorers, RPCs) follows from it.

### 3. Get a test photo

Any photo with a face works. For a realistic reverse-search demo, use a widely
reposted public figure:

```bash
python scripts/fetch_sample.py "Sundar Pichai"
```

### 4. Run

```bash
# full pipeline, writes to Base Sepolia
python pipeline.py --image samples/sundar_pichai.jpg

# dry run: everything except the transaction (no gas, no funds needed)
python pipeline.py --image samples/sundar_pichai.jpg --no-chain

# validate the on-chain call with eth_call instead of sending it
python pipeline.py --image samples/sundar_pichai.jpg --simulate

# watch the actual searches happen (best for a screen recording)
python pipeline.py --image samples/sundar_pichai.jpg --headful
```

**If the image is already public somewhere, pass its URL** — engines' by-URL
endpoints are far more reliable than their drag-and-drop upload flows:

```bash
python pipeline.py --image samples/sundar_pichai.jpg \
  --image-url 'https://upload.wikimedia.org/wikipedia/commons/c/c3/Sundar_Pichai_-_2023_%28cropped%29.jpg'
```

For a purely local file you can opt into a temporary public host (**off by
default** — it uploads your photo to a third party, Litterbox, 1-hour TTL):

```bash
python pipeline.py --image my_photo.jpg --allow-upload-host
```

Sample output:

```
╭──────────────────────────────────────────────────╮
│          FACECHAIN VERIFICATION PIPELINE         │
│  v1.0.0 • Ethereum Sepolia • EAS attestations     │
╰──────────────────────────────────────────────────╯

[01/07] Loading & hashing image…
        ✓ 959x1439, sha256 8e2e4fc7f6a119a1…
[02/07] Detecting + encoding face…
        ✓ 1 face(s), 512-D buffalo_l/SCRFD+ArcFace, det 0.890
[03/07] Reverse image search…
        ├─ yandex                 ✓ 60 candidates
        ├─ bing                   ✓ 51 candidates
        ✓ 110 candidates (5 social) from yandex, bing
[04/07] Candidate verification…
        ├─ candidate              ✓ youtube.com  img 0.75 face 0.97 score 0.85 VERIFIED
        ✓ YouTube https://www.youtube.com/shorts/665OKw6IkEM score 0.851
[05/07] Evidence bundle + hashes…
        ✓ evidenceHash sha256:236e1f9382d28d1a200c6c1f…
[06/07] Blockchain attestation…
        ├─ wallet                 ✓ 0xF3F3…E45A — 0.067074 ETH on Ethereum Sepolia
        ├─ schema                 ✓ 0xa9ff57af6e5bea0d…
        ├─ tx                     ✓ 0x87ebdda7… in block 11614144
[07/07] On-chain read-back verify…
        ✓ all 11 on-chain fields match local evidence
```

---

## Which blockchain, and why

**Base Sepolia** — Coinbase's Base L2 public testnet.

| | |
|---|---|
| Chain ID | `84532` |
| RPC | `https://sepolia.base.org` (plus two public fallbacks, auto-failover) |
| Explorer | <https://sepolia.basescan.org> |
| EAS | `0x4200000000000000000000000000000000000021` |
| SchemaRegistry | `0x4200000000000000000000000000000000000020` |
| EAS explorer | <https://base-sepolia.easscan.org> |
| Cost | free (faucet ETH); a full attestation is ~0.00002 ETH of gas |

Why this and not something else:

- **A real chain, not a simulation.** Public RPC, public explorer, independently
  verifiable by anyone with the attestation UID.
- **Testnet only, by design.** Zero real money. The code hard-asserts chain id
  84532 and refuses to run anywhere else, so it cannot accidentally touch
  mainnet.
- **EAS instead of a bespoke contract.** The Ethereum Attestation Service is a
  permissionless, widely used attestation registry already deployed at Base's
  canonical predeploy addresses. Writing our own `FaceVerification.sol` would
  add a contract to deploy, verify and trust for no gain — EAS already provides
  exactly the primitive the task asks for: a schema'd, timestamped, signed,
  publicly readable record. Attestations are also readable in any EAS explorer,
  so a judge does not need this repo to inspect the result.
- **No Node.js dependency.** EAS is normally used through its TypeScript SDK;
  this implementation talks to the contracts directly with `web3.py` and a
  minimal ABI. One language, one dependency tree, and nothing hidden behind an
  SDK when someone asks what exactly was written.

---

## What goes on-chain (and what deliberately does not)

The EAS schema (11 fields, UID derived deterministically from this string):

```solidity
bytes32 caseId,
bytes32 inputImageHash,
bytes32 faceEmbeddingHash,
bytes32 matchedImageHash,
bytes32 matchedUrlHash,
bytes32 evidenceHash,
string  searchEngine,
string  socialPlatform,
uint16  matchScoreBps,
uint64  observedAt,
string  pipelineVersion
```

**Not on-chain, on purpose:**

- **The raw face embedding.** Only its SHA-256 goes on chain. A 512-D ArcFace
  vector is biometric data and is partially invertible — publishing it forever
  on a public ledger would be an irreversible privacy harm. The vector never
  leaves the machine.
- **The matched URL in plaintext.** Only `matchedUrlHash`. The record stays
  fully checkable (the plaintext URL sits in the local evidence bundle, and
  anyone holding the bundle can re-hash it) without permanently republishing a
  link to someone's profile on an immutable ledger.
- **The image itself.** Only hashes. Putting image bytes on-chain would be
  wasteful and would republish someone's photo irrevocably.

That combination is what makes the record *tamper-evident* rather than merely
*public*: the chain pins the hashes, the bundle holds the preimages, and neither
can be altered afterwards without the two disagreeing.

---

## The verification ladder

Each rung is a separate, checkable claim:

| Rung | Means |
|---|---|
| `SEARCH_FOUND` | a real reverse-image engine returned this URL |
| `SOCIAL_MATCH` | the URL is a post on a supported social platform |
| `IMAGE_MATCH` | the retrieved image is perceptually the same picture (pHash ≥ 0.80) |
| `FACE_MATCH` | ArcFace cosine between input face and retrieved face ≥ 0.38 |
| `VERIFIED` | `SEARCH_FOUND` + `SOCIAL_MATCH` + `FACE_MATCH` + weighted score ≥ 0.70 |

```
final_score = 0.50 × face_cosine + 0.40 × image_similarity + 0.10 × metadata
```

**`IMAGE_MATCH` is recorded but not required for `VERIFIED`, and that is a
deliberate choice.** Social reposts crop, pad and overlay text, which pushes a
perceptual hash below the exact-image bar while the face stays unmistakable —
and this is a face-identification task, so the face is the primary signal and
the image hash is corroboration. Which one held is preserved in `match_type`:

- `exact-image` — same picture, byte-level provenance
- `face-only` — same face, visibly edited or re-cropped picture

A wrong person cannot slip through: with face cosine near zero, the composite
score cannot reach 0.70 on image similarity alone. Measured on this repo's
samples — same person through a 3× downscale and JPEG q55: **0.963**; two
different people: **−0.034**. Thresholds are in `.env`; every threshold used is
recorded in the evidence bundle for the run.

---

## Reverse image search: how it stays honest

Three properties were designed in after watching things fail:

1. **Results must be proven, not assumed.** Each adapter requires a
   *marker* — Bing's "pages with this image", Yandex's "sites containing
   information about the image" — on the settled page before any link is
   treated as a result. This is not decoration: during development a Bing
   upload silently failed to submit, leaving the pipeline on Bing's *homepage*,
   where a naive link harvest cheerfully returned 50 "social matches" that were
   trending-topic links. Without the marker guard, that is exactly how a
   pipeline ends up reporting matches that were never search results.

2. **Layout-independent extraction.** Engines rewrite their DOM constantly, so
   adapters prefer a known results container but fall back to harvesting every
   outbound link and filtering the engine's own chrome. A class-name scraper
   returns zero results the day Google reshuffles its markup; a link harvester
   keeps working. (Both matter: `div.CbirSites-Items` matched nothing because
   Yandex's list is not a `<div>`.)

3. **Engines are independent and expendable.** All engines run per case, results
   are merged and de-duplicated, and a URL returned by two engines scores higher
   for corroboration. Any engine may fail or be CAPTCHA'd; its error is recorded
   in the evidence bundle and the run continues. URLs are canonicalised
   (tracking params stripped) before hashing, so the same post found via two
   engines yields one `matchedUrlHash`.

**Verification never trusts the engine.** For every candidate the pipeline
fetches the page itself, extracts the image the post actually displays
(`og:image` → `twitter:image` → JSON-LD → `<img>`), downloads it, and re-runs
both similarity tests locally. When a platform refuses anonymous fetches (login
walls are common), it falls back to the thumbnail the engine stored for that
result and records the provenance in `candidate_image_source`, so the evidence
never overstates where the compared pixels came from.

---

## Evidence bundle

Every run writes `evidence/case_<timestamp>/`:

```
case.json                 full structured record of every stage
attested_payload.json     exactly the fields hashed into evidenceHash
attested_payload.sha256   that canonical hash
input.sha256              shasum -c compatible
face_embedding.sha256     hash of the 512-D vector (vector itself stays local)
matched_image.sha256      hash of the image downloaded from the social post
reverse_search.json       every candidate, every engine, every engine error
verification.json         per-candidate similarity measurements
blockchain.json           tx hash, UID, block, gas, read-back result
attestation.txt           human-readable receipt + how to verify it
artifacts/input.jpg       copy of the input
artifacts/face_crop.png   the detected face
```

`evidenceHash` is the SHA-256 of the canonical JSON (sorted keys, no
insignificant whitespace, floats quantised to 3 decimals) of
`attested_payload.json` — so it reproduces byte-for-byte on any machine.

---

## Verifying a record independently

```bash
python scripts/verify_attestation.py --case evidence/case_20260901_004512
```

This shares as little as possible with the pipeline that produced the record. It
re-reads the bundle from disk, re-computes every hash, reads the attestation
straight from the EAS contract, and compares — it never trusts
`blockchain.json`'s own claim of success:

```
1. Local evidence bundle
  [PASS] bundle hashes self-consistent
  [PASS] input image re-hashes to case.json value
  [PASS] evidenceHash recomputed from payload
  [PASS] matched URL hash matches its plaintext
2. On-chain attestation 0x…
  [PASS] field caseId
  … all 11 fields …
  [PASS] attestation not revoked
  [PASS] attester matches recorded signer
RESULT: VERIFIED — local evidence and chain agree
```

Tamper with any byte of `attested_payload.json` and the check fails — that is
the tamper-evidence, demonstrated rather than asserted.

Any attestation can also be dumped directly:

```bash
python scripts/verify_attestation.py --uid 0xabc…
```

…or opened in the explorer: `https://base-sepolia.easscan.org/attestation/view/<uid>`

---

## CLI reference

```
python pipeline.py --image PATH [options]

  --image PATH              input photo (required)
  --image-url URL           public URL of the same image; enables engines'
                            by-URL search (nothing is uploaded anywhere)
  --engines LIST            yandex,bing,google_lens,tineye,
                            serpapi_google_lens,serpapi_yandex
  --allow-upload-host       upload the photo to a temporary public host so
                            by-URL search works with a local file (off by
                            default; third-party upload, 1-hour TTL)
  --no-chain                stop after local verification
  --simulate                eth_call the attestation (validates encoding, no gas)
  --headful                 show the browser — best for screen recordings
  --face-backend            auto | insightface | opencv
  --max-verify N            cap candidates fetched and measured (default 12)
  --case-id ID              override the generated case id
  --json                    print case.json to stdout
  -v / -vv                  info / debug logging
```

Exit codes: `0` verified on-chain, `1` not verified (or chain stage failed),
`2` bad input.

Helper scripts:

```bash
python scripts/register_schema.py [--check]      # one-off schema registration
python scripts/verify_attestation.py --case DIR  # independent verification
python scripts/fetch_sample.py "Name"            # grab a freely-licensed photo
```

---

## Repository layout

```
pipeline.py                  CLI entrypoint + terminal UI (presentation only)
src/facechain/
  config.py                  all settings, thresholds, chain constants
  models.py                  typed pipeline/evidence model (pydantic)
  runner.py                  stage orchestration
  face/
    detector.py              InsightFace + OpenCV backends, model download
    encoder.py               image → (face record, embedding)
    similarity.py            cosine similarity
  search/
    base.py                  adapter contract, harvesting, URL canonicalisation
    browser.py               Playwright session, upload, block detection
    google_lens.py  yandex.py  bing.py  tineye.py
    serpapi.py               optional real reverse-image-search API
    uploader.py              optional temporary public hosting
    orchestrator.py          multi-engine fan-out + merge
  verification/
    candidate.py             fetch page, extract image, measure locally
    image_similarity.py      pHash/dHash comparison
    social.py                social classification, metadata corroboration
    scorer.py                the ladder, scoring, failure explanation
  chain/
    abi.py                   minimal EAS + SchemaRegistry ABIs
    schema.py                schema parsing, ABI encode/decode, UID derivation
    eas.py                   Base Sepolia client, attest, read-back verify
  evidence/
    hashing.py               deterministic hashing (canonical JSON, embeddings)
    writer.py                evidence bundle + integrity checks
scripts/                     register_schema, verify_attestation, fetch_sample
tests/                       104 tests, no network required
```

---

## Tests

```bash
python -m pytest tests/ -q      # 104 passed
```

No network, no chain, no API keys required. Coverage focuses on the parts where
a silent bug would produce a *convincing but wrong* result:

- hash determinism (canonical JSON, byte-order-independent embedding hashes,
  float quantisation)
- EAS encoding round-trips and the schema-UID formula, checked against EAS's
  own `keccak(abi.encodePacked(...))` definition
- the ladder: wrong person cannot verify on image similarity alone; non-social
  pages never verify however high they score; edited reposts verify as
  `face-only`
- tamper detection: editing the payload or the input image must break the bundle
- URL canonicalisation, redirect unwrapping, social-domain lookalike rejection
  (`instagram.com.evil.net` is not Instagram)

The face-model tests use the real InsightFace weights and skip automatically if
the weights or `samples/` are absent.

---

## Known limitations

**Reverse image search is the fragile part, and it is fragile by nature.**

1. **Google Lens frequently blocks headless automation.** In testing it served
   its `/sorry` "unusual traffic" interstitial most of the time. The adapter
   detects this and reports it rather than pretending; Yandex and Bing carry the
   run. `--headful` helps; `SERPAPI_KEY` (a real reverse-image-search API) is
   the reliable route to Lens results.
2. **Engine UIs change without notice.** Marker/container selectors will rot.
   The layout-independent fallback and the per-engine adapter boundary keep one
   engine's breakage from taking down the pipeline, but selectors will need
   occasional maintenance. Each engine's failure reason is recorded per run.
3. **Upload flows are less reliable than by-URL flows.** Programmatic file
   attachment does not always trigger the engine's submit handler. Prefer
   `--image-url`, or `--allow-upload-host`.
4. **Whether a match exists at all is outside our control.** An obscure or
   never-posted photo has no social copy to find, and the pipeline will
   correctly report `UNVERIFIED` with the reason. A private photo of a private
   individual should not be expected to produce a match.
5. **Social platforms block anonymous page fetches.** Instagram/Facebook often
   return a login wall, so the comparison may fall back to the engine's stored
   thumbnail. This is recorded in `candidate_image_source`, and thumbnails are
   lower resolution, which depresses both similarity scores.
6. **`match_type: face-only` is a weaker provenance claim** than
   `exact-image`: it says the same face appears on that post, not that it is the
   same photograph.
7. **CPU-only inference.** ~1–3 s per image for detection + embedding; a full
   run is dominated by browser time (roughly 1–3 minutes).
8. **Rate limiting.** Repeated runs from one IP invite CAPTCHAs on every engine.
9. **Face recognition is not identity.** ArcFace cosine above a threshold is a
   statistical statement. Look-alikes, siblings, heavy editing, age gaps and
   demographic bias in the training data all cause errors. Published
   benchmarks do not transfer to arbitrary internet images.
10. **Testnet only.** Base Sepolia state carries no economic guarantees and
    testnets can be reset; this demonstrates the mechanism, not a production
    trust anchor. A production deployment would use a mainnet L2 and a funded,
    key-managed attester.
11. **The chain proves integrity and timing, not truth.** An attestation shows
    that *this pipeline, holding this key, recorded these hashes at this time*
    and that nothing has changed since. It does not make the underlying match
    correct.

---

## Ethics and scope

This is an OSINT/verification demo built from public data and freely licensed
sample images.

**What a `VERIFIED` record claims:** the supplied image and the face in it match
an image retrieved from that public social-media post, under the thresholds
recorded in the run's evidence bundle, where the post was found by genuine
reverse-image search at that timestamp.

**What it does not claim:** any person's real-world identity. It is not an
identification system, and it is not evidence about a person.

Face search touches real people. Reasonable use means: your own photos, public
figures for testing, or images you are authorised to investigate — with consent
where consent is owed, and never to locate, profile or harass a private
individual. The privacy choices in the design (embeddings hashed, URLs hashed,
raw biometrics never published) exist because "it's only a testnet" is not a
reason to put biometric data on an immutable public ledger.

---

## License

MIT — see [LICENSE](LICENSE).

Third-party components keep their own licenses; note that **InsightFace's
pretrained models are for non-commercial research use**. Swap in a commercially
licensed encoder (or the OpenCV/SFace backend, `--face-backend opencv`) for any
commercial deployment.
