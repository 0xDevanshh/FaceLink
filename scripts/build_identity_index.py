#!/usr/bin/env python3
"""Offline builder for the known-person identity index.

    python scripts/build_identity_index.py [--out identity_index] [--limit N]

Pipeline (mission section 2/3/17):

    seed names (resolved to Wikidata QIDs by search — nothing here hardcodes
    an ID from memory; a wrong guess would silently mismatch a person)
      -> Wikidata entity data (occupation/country/website/social handles)
      -> Wikimedia Commons reference images (multiple per person, licensed)
      -> face/encoder.py::encode_face()  <- the SAME embedding pipeline the
                                             live scan uses, never a second model
      -> identity_index/manifest.json + embeddings.npy

Run manually, offline from the API server, whenever the seed list grows —
not imported by src/facechain (tooling, like scripts/fetch_sample.py).

Sources: Wikidata (CC0) and Wikimedia Commons (per-file licenses recorded in
each reference's `license`/`source_url`). Both are public, documented,
unauthenticated REST APIs — no scraping, no CAPTCHA/WAF bypass, no login.
A descriptive User-Agent is sent on every request, per Wikimedia's API
etiquette policy, and requests are paced (REQUEST_DELAY_S) rather than fired
as fast as possible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

log = logging.getLogger("build_identity_index")

USER_AGENT = (
    "FaceLinkIdentityIndexBuilder/1.0 "
    "(offline known-person index builder for a personal reverse-image-search "
    "project; contact: repository owner)"
)
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
WIKIDATA_ENTITY_URL = "https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"

# Wikidata properties consumed per person — verified against real entity
# data before use (see PIPELINE.md-style commentary in the plan this
# implements), not assumed from documentation alone.
P_WEBSITE = "P856"
P_IMAGE = "P18"
P_COMMONS_CATEGORY = "P373"
P_TWITTER = "P2002"
P_INSTAGRAM = "P2003"
P_FACEBOOK = "P2013"
P_YOUTUBE_CHANNEL = "P2397"
P_GITHUB = "P2037"

# platform -> (wikidata property, URL template). LinkedIn deliberately
# absent: Wikidata has no reliable personal-profile property for it — the
# existing enrichment pipeline's live discovery covers LinkedIn instead
# (see the approved plan's "known gap, disclosed not hidden").
SOCIAL_PROPERTIES: dict[str, tuple[str, str]] = {
    "X/Twitter": (P_TWITTER, "https://x.com/{value}"),
    "Instagram": (P_INSTAGRAM, "https://instagram.com/{value}"),
    "Facebook": (P_FACEBOOK, "https://facebook.com/{value}"),
    "YouTube": (P_YOUTUBE_CHANNEL, "https://www.youtube.com/channel/{value}"),
    "GitHub": (P_GITHUB, "https://github.com/{value}"),
}

MAX_REFERENCES_PER_PERSON = 5
REQUEST_DELAY_S = 0.4
HTTP_TIMEOUT_S = 20
MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# Seed set (mission section 17: "start with a high-quality seed set" — a
# literal, curated list is expected for the MVP; nothing about the runtime
# *matching* logic is hardcoded — matcher.py/scorer.py have no idea how many
# or which people are in the index). Resolved to QIDs by name search below,
# never a hand-typed ID, so a misremembered ID can't silently mismatch a
# person's face embeddings to the wrong name.
# ---------------------------------------------------------------------------
SEED_PEOPLE: list[dict] = [
    {"name": "Sundar Pichai", "category": "technology", "occupation": "CEO, Google and Alphabet"},
    {"name": "Satya Nadella", "category": "technology", "occupation": "CEO, Microsoft"},
    {"name": "Bill Gates", "category": "technology", "occupation": "Co-founder, Microsoft"},
    {"name": "Mark Zuckerberg", "category": "technology", "occupation": "CEO, Meta"},
    {"name": "Tim Cook", "category": "technology", "occupation": "CEO, Apple"},
    {"name": "Jensen Huang", "category": "technology", "occupation": "CEO, Nvidia"},
    {"name": "Sam Altman", "category": "technology", "occupation": "CEO, OpenAI"},
    {"name": "Jeff Bezos", "category": "business", "occupation": "Founder, Amazon"},
    {"name": "Warren Buffett", "category": "business", "occupation": "Investor, Berkshire Hathaway"},
    {"name": "Barack Obama", "category": "politics", "occupation": "44th President of the United States"},
    {"name": "Narendra Modi", "category": "politics", "occupation": "Prime Minister of India"},
    {"name": "Angela Merkel", "category": "politics", "occupation": "Former Chancellor of Germany"},
    {"name": "Marie Curie", "category": "science", "occupation": "Physicist and chemist"},
    {"name": "Stephen Hawking", "category": "science", "occupation": "Theoretical physicist"},
    {"name": "Neil deGrasse Tyson", "category": "science", "occupation": "Astrophysicist"},
    {"name": "Serena Williams", "category": "sports", "occupation": "Tennis player"},
    {"name": "Cristiano Ronaldo", "category": "sports", "occupation": "Footballer"},
    {"name": "Lionel Messi", "category": "sports", "occupation": "Footballer"},
    {"name": "Leonardo DiCaprio", "category": "actors", "occupation": "Actor"},
    {"name": "Meryl Streep", "category": "actors", "occupation": "Actor"},
    {"name": "Dwayne Johnson", "category": "actors", "occupation": "Actor"},
    {"name": "Taylor Swift", "category": "musicians", "occupation": "Musician"},
    {"name": "Beyoncé", "category": "musicians", "occupation": "Musician"},
    {"name": "Ed Sheeran", "category": "musicians", "occupation": "Musician"},
    {"name": "Linus Torvalds", "category": "founders", "occupation": "Creator of Linux and Git"},
    {"name": "Guido van Rossum", "category": "founders", "occupation": "Creator of Python"},
    {"name": "Jack Dorsey", "category": "founders", "occupation": "Co-founder, Twitter/Block"},
    {"name": "J.K. Rowling", "category": "authors", "occupation": "Author"},
    {"name": "Malala Yousafzai", "category": "public_figures", "occupation": "Education activist"},
    {"name": "Oprah Winfrey", "category": "entertainment", "occupation": "Media executive and host"},
    {"name": "Chris Evans", "category": "actors", "occupation": "Actor"},
    {"name": "Zendaya", "category": "actors", "occupation": "Actor"},
    {"name": "Elon Musk", "category": "technology", "occupation": "CEO, Tesla and SpaceX"},
]


def _get_with_retries(url: str, params: Optional[dict] = None) -> Optional[bytes]:
    """GET with retries, returning the response body or `None` on failure.

    Uses `urllib.request` (matching `search/serpapi.py`'s existing HTTP
    style in this codebase) rather than `httpx` — verified during
    development that `httpx`'s TLS/HTTP client shape gets a 403 from
    Wikimedia's edge in some network environments where plain `urllib` (and
    `curl`) succeed against the exact same URL; this is not a bypass of
    anything, just a different, equally standard HTTP client.
    """
    full_url = f"{url}?{urllib.parse.urlencode(params)}" if params else url
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(full_url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            log.warning("GET %s -> HTTP %d (attempt %d/%d)", full_url, exc.code, attempt + 1, MAX_RETRIES)
        except Exception as exc:  # noqa: BLE001 — one bad request must not kill the build
            log.warning("GET %s failed: %s: %s (attempt %d/%d)",
                       full_url, type(exc).__name__, exc, attempt + 1, MAX_RETRIES)
        time.sleep(REQUEST_DELAY_S * (attempt + 1))
    return None


# Category -> plausible words in a Wikidata search result's own `description`
# field, used only to disambiguate WHICH same-named entity is meant (e.g.
# "Chris Evans" matches an author, a British politician, AND the American
# actor — Wikidata's default relevance ranking is not fame-aware). This maps
# generic categories to generic keywords; it names no specific person and
# applies identically regardless of which name is being resolved.
_CATEGORY_HINTS: dict[str, tuple[str, ...]] = {
    "actors": ("actor", "actress"),
    "musicians": ("singer", "musician", "rapper", "songwriter"),
    "technology": ("ceo", "businessperson", "entrepreneur", "engineer", "computer scientist", "executive"),
    "business": ("businessman", "businesswoman", "businessperson", "investor", "entrepreneur", "executive"),
    "politics": ("politician", "president", "prime minister", "senator"),
    "science": ("physicist", "scientist", "chemist", "astrophysicist", "mathematician"),
    "sports": ("tennis", "footballer", "athlete", "player", "basketball"),
    "founders": ("engineer", "computer scientist", "entrepreneur", "programmer", "founder"),
    "authors": ("author", "writer", "novelist"),
    "creators": ("youtuber", "content creator", "influencer"),
    "influencers": ("influencer", "content creator", "personality"),
    "academics": ("professor", "academic", "researcher"),
    "public_figures": ("activist", "public figure"),
    "entertainment": ("host", "media", "actress", "actor", "presenter"),
}


def search_wikidata_qid(name: str, category: str = "") -> Optional[str]:
    """Resolve a person's name to a Wikidata QID via the public search API.

    Common names are genuinely ambiguous — Wikidata's own relevance ranking
    is not fame-aware, so the top result for "Chris Evans" is an author, not
    the actor. When `category` is given, the first candidate whose own
    Wikidata `description` contains one of that category's hint words (see
    `_CATEGORY_HINTS` — generic, never person-specific) wins; otherwise this
    falls back to the plain top result, unchanged from before.
    """
    body = _get_with_retries(WIKIDATA_API, params={
        "action": "wbsearchentities", "search": name, "language": "en", "format": "json", "limit": 5,
    })
    if body is None:
        return None
    results = json.loads(body).get("search") or []
    if not results:
        log.warning("no Wikidata entity found for %r", name)
        return None

    hints = _CATEGORY_HINTS.get(category, ())
    if hints:
        for hit in results:
            description = (hit.get("description") or "").lower()
            if any(h in description for h in hints):
                if hit is not results[0]:
                    log.info("disambiguated %r to %s (%r) over the plain top search result",
                            name, hit["id"], hit.get("description"))
                return hit["id"]

    hit = results[0]
    if hit.get("label", "").lower() != name.lower():
        log.warning("Wikidata's best match for %r is labelled %r (%s) — using it, but verify",
                   name, hit.get("label"), hit.get("id"))
    return hit["id"]


def fetch_entity(qid: str) -> Optional[dict]:
    body = _get_with_retries(WIKIDATA_ENTITY_URL.format(qid=qid))
    if body is None:
        return None
    return json.loads(body).get("entities", {}).get(qid)


def _claim_value(entity: dict, prop: str) -> Optional[object]:
    claims = entity.get("claims", {}).get(prop) or []
    for claim in claims:
        value = claim.get("mainsnak", {}).get("datavalue", {}).get("value")
        if value is not None:
            return value
    return None


def commons_file_url(filename: str) -> str:
    """Convert a Wikimedia Commons filename (as stored in a P18 claim) to its
    real upload.wikimedia.org URL — the documented MD5-hash-bucket
    convention, not guessed: verified directly against a real file during
    development of this script."""
    name = filename.replace(" ", "_")
    digest = hashlib.md5(name.encode("utf-8")).hexdigest()
    return f"https://upload.wikimedia.org/wikipedia/commons/{digest[0]}/{digest[0:2]}/{urllib.parse.quote(name)}"


def fetch_commons_category_files(category: str, limit: int) -> list[str]:
    """Extra reference images beyond the single P18 portrait. Best-effort:
    returns `[]` (never raises) if the category API is unreachable — the
    single P18 image is still enough to build a working (if thinner)
    reference set for that person."""
    body = _get_with_retries(COMMONS_API, params={
        "action": "query", "list": "categorymembers", "cmtitle": f"Category:{category}",
        "cmtype": "file", "cmlimit": limit, "format": "json",
    })
    if body is None:
        return []
    members = json.loads(body).get("query", {}).get("categorymembers", [])
    return [m["title"].removeprefix("File:") for m in members]


def build_person_record(seed: dict) -> Optional[dict]:
    """Resolve one seed entry to raw fields ready for embedding. Returns
    `None` (logs why) rather than raising if the person can't be resolved."""
    qid = search_wikidata_qid(seed["name"], seed.get("category", ""))
    if qid is None:
        return None
    time.sleep(REQUEST_DELAY_S)
    entity = fetch_entity(qid)
    if entity is None:
        log.warning("could not fetch Wikidata entity %s for %r", qid, seed["name"])
        return None

    label = entity.get("labels", {}).get("en", {}).get("value", seed["name"])
    aliases = [a["value"] for a in entity.get("aliases", {}).get("en", [])][:5]
    website = _claim_value(entity, P_WEBSITE)
    social: dict[str, str] = {}
    for platform, (prop, template) in SOCIAL_PROPERTIES.items():
        value = _claim_value(entity, prop)
        if isinstance(value, str) and value:
            social[platform] = template.format(value=value)

    image_files: list[str] = []
    primary_image = _claim_value(entity, P_IMAGE)
    if isinstance(primary_image, str):
        image_files.append(primary_image)

    commons_category = _claim_value(entity, P_COMMONS_CATEGORY)
    if isinstance(commons_category, str):
        time.sleep(REQUEST_DELAY_S)
        extra = fetch_commons_category_files(commons_category, MAX_REFERENCES_PER_PERSON)
        for f in extra:
            if f not in image_files and f.lower().endswith((".jpg", ".jpeg", ".png")):
                image_files.append(f)

    if not image_files:
        log.warning("no usable reference images found for %r (%s) — skipping", label, qid)
        return None

    return {
        "person_id": qid,
        "canonical_name": label,
        "aliases": [a for a in aliases if a.lower() != label.lower()],
        "occupation": seed.get("occupation", ""),
        "category": seed.get("category", ""),
        "official_website": website if isinstance(website, str) else "",
        "official_social_accounts": social,
        "source_urls": [f"https://www.wikidata.org/wiki/{qid}"],
        "image_files": image_files[:MAX_REFERENCES_PER_PERSON],
    }


def embed_reference_images(record: dict) -> tuple[list[dict], list[np.ndarray]]:
    """Download each candidate image and run it through the SAME face
    embedding pipeline the live scan uses — never a second model. Skips
    (logs, doesn't raise) any image that fails to download or has no
    detectable face; a person with zero usable images after this is dropped
    entirely by the caller."""
    from facechain.face.detector import load_backend
    from facechain.face.encoder import decode_image, encode_face

    backend = load_backend()
    references: list[dict] = []
    embeddings: list[np.ndarray] = []

    for filename in record["image_files"]:
        url = commons_file_url(filename)
        body = _get_with_retries(url)
        if body is None:
            log.warning("could not download reference image for %s: %s", record["canonical_name"], url)
            continue
        img = decode_image(body)
        if img is None:
            log.warning("could not decode reference image for %s: %s", record["canonical_name"], url)
            continue
        face_record, embedding, _ = encode_face(img, backend_name=backend.name)
        if embedding is None:
            log.warning("no detectable face in reference image for %s: %s", record["canonical_name"], url)
            continue
        references.append({
            "source_url": url,
            "license": "see Wikimedia Commons file page for license/attribution",
            "width": img.shape[1], "height": img.shape[0],
            "det_score": face_record.det_score or 0.0,
        })
        embeddings.append(embedding.astype(np.float32))
        time.sleep(REQUEST_DELAY_S)

    return references, embeddings


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(REPO_ROOT / "identity_index"))
    parser.add_argument("--limit", type=int, default=None, help="only build the first N seed people")
    args = parser.parse_args()

    seeds = SEED_PEOPLE[: args.limit] if args.limit else SEED_PEOPLE
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    people_json: list[dict] = []
    all_embeddings: list[np.ndarray] = []
    embedding_dim = 0
    embedding_model = ""

    for seed in seeds:
        log.info("building %s ...", seed["name"])
        record = build_person_record(seed)
        time.sleep(REQUEST_DELAY_S)
        if record is None:
            continue
        references, embeddings = embed_reference_images(record)
        if not embeddings:
            log.warning("skipping %s: no reference embeddings produced", record["canonical_name"])
            continue
        for i, ref in enumerate(references):
            ref["embedding_index"] = len(all_embeddings) + i
        all_embeddings.extend(embeddings)
        if embedding_dim == 0:
            embedding_dim = embeddings[0].size
            from facechain.face.detector import load_backend
            backend = load_backend()
            embedding_model = f"{backend.model_name}/{backend.name}"
        people_json.append({
            "person_id": record["person_id"],
            "canonical_name": record["canonical_name"],
            "aliases": record["aliases"],
            "occupation": record["occupation"],
            "category": record["category"],
            "official_website": record["official_website"],
            "official_social_accounts": record["official_social_accounts"],
            "source_urls": record["source_urls"],
            "references": references,
        })
        log.info("  -> %d reference embedding(s)", len(embeddings))

    if not people_json:
        log.error("no people were successfully built — nothing written")
        return 1

    manifest = {
        "version": time.strftime("%Y%m%d-%H%M%S"),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "embedding_dim": embedding_dim,
        "embedding_model": embedding_model,
        "people": people_json,
    }
    manifest_bytes = json.dumps(manifest, indent=2).encode("utf-8")
    (out_dir / "manifest.json").write_bytes(manifest_bytes)
    np.save(out_dir / "embeddings.npy", np.stack(all_embeddings).astype(np.float32))
    (out_dir / "MANIFEST_SHA256").write_text(hashlib.sha256(manifest_bytes).hexdigest(), encoding="utf-8")

    log.info("wrote %d people, %d reference embeddings to %s",
             len(people_json), len(all_embeddings), out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
