# Recording the end-to-end demo

The submission asks for an **unedited screen recording of the full pipeline
running end to end**. This is a script for that take, ordered so a viewer can
see each requirement being satisfied without any cuts.

## Before you hit record

Do these first — they are slow, boring, and not part of the pipeline:

```bash
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium

# warm the InsightFace weights (~275 MB) so the run isn't a download progress bar
python -c "
import sys; sys.path.insert(0,'src')
from facechain.face.detector import load_backend; load_backend()"

# fetch a sample photo
python scripts/fetch_sample.py 'Sundar Pichai'

# confirm the attester is funded
python scripts/register_schema.py --check
```

Have ready:

- a funded Base Sepolia attester (`PRIVATE_KEY` in `.env`)
- a browser tab on <https://base-sepolia.easscan.org>
- a terminal at least 100 columns wide

> Rate limiting is the main thing that ruins a take. Do not run the pipeline
> repeatedly right before recording — engines start serving CAPTCHAs. Leave a
> few minutes between rehearsal and the real take.

## The take (~5 minutes)

**1. Show the repo and the tests — no network, no chain, nothing mocked.**

```bash
python -m pytest tests/ -q
```

**2. Show the schema that will be written, and that it is already registered.**

```bash
python scripts/register_schema.py --check
```

**3. Run the pipeline.** `--headful` puts the actual reverse-image searches on
screen, which is the single most convincing part of the recording.

```bash
python pipeline.py --image samples/sundar_pichai.jpg --headful \
  --image-url 'https://upload.wikimedia.org/wikipedia/commons/c/c3/Sundar_Pichai_-_2023_%28cropped%29.jpg'
```

Narrate as the stages land:

- face detected, 512-D ArcFace embedding, hashed (the vector never leaves the machine)
- engines queried live — the browser windows are visible
- each candidate fetched and re-measured locally, not trusted from the engine
- the ladder: `SEARCH_FOUND → SOCIAL_MATCH → FACE_MATCH → VERIFIED`
- the transaction, then the read-back comparing all 11 on-chain fields

**4. Show the evidence bundle.**

```bash
ls evidence/case_*/
cat evidence/case_*/attestation.txt
```

**5. Verify independently, in the same take.**

```bash
python scripts/verify_attestation.py --case evidence/case_<id>
```

**6. Open the attestation in a block explorer** — a third party displaying the
same hashes closes the loop:

```
https://base-sepolia.easscan.org/attestation/view/<uid>
```

**7. Prove tamper-evidence.** Work on a *copy* so the real bundle stays intact:

```bash
cp -r evidence/case_<id> /tmp/tampered
python - <<'EOF'
import json, pathlib
p = pathlib.Path('/tmp/tampered/attested_payload.json')
d = json.loads(p.read_text())
d['matched_url'] = 'https://instagram.com/p/FAKE-EVIDENCE/'
p.write_text(json.dumps(d, indent=2))
EOF
python scripts/verify_attestation.py --case /tmp/tampered
```

Three independent `[FAIL]` lines appear. That is the tamper-evidence
demonstrated rather than asserted.

## If an engine gets blocked mid-take

Keep rolling. A CAPTCHA on one engine is a real property of the system, the
pipeline reports it honestly and continues on the others, and the README
documents it as a known limitation. Recovering gracefully on camera reads far
better than a suspiciously perfect run.

If *every* engine is blocked, stop, wait a few minutes, and consider:

- switching networks / IP
- `--engines yandex,bing`
- setting `SERPAPI_KEY` and adding `serpapi_google_lens` to `--engines`

## Two honest notes for the voiceover

- The run verifies that the input face matches an image on that public social
  post. It is **not** an identity claim about a person.
- Everything is on **Base Sepolia testnet**. No real money, and testnet state
  carries no economic guarantee — it demonstrates the mechanism.
