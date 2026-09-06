"""API Ninjas Celebrity API — optional, additive celebrity-metadata enrichment.

    GET https://api.api-ninjas.com/v1/celebrity?name=<resolved name>
    X-Api-Key: ${API_NINJAS_API_KEY}

Docs: https://api-ninjas.com/api/celebrity

This module is metadata enrichment ONLY, called strictly after the
Known-Person Identity layer (InsightFace + KnownPersonIndex + evidence) has
already resolved a name from face evidence — never the other way around, and
never used for face recognition itself. `fetch_celebrity_info` always
returns a `CelebrityInfo`, never raises: `available=False` uniformly covers
every failure mode (no key configured, network error, timeout, rate limit,
auth error, malformed JSON, no plausible name match in the results), the
same shape `face/luxand.py` already uses for its own optional third-party
cross-check.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Optional

import httpx

from ..config import settings
from ..models import CelebrityInfo

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lightweight process-wide TTL cache — no generic TTL cache exists elsewhere
# in this codebase to reuse; the closest precedent is
# `search/uploader.py`'s local-image registry (a dict + lock + monotonic
# expiry), the same shape reused here rather than adding a new dependency.
# ---------------------------------------------------------------------------
_cache: dict[str, tuple[CelebrityInfo, float]] = {}
_cache_lock = threading.Lock()


def _cache_key(resolved_name: str) -> str:
    return f"celebrity:{_normalise_name(resolved_name)}"


def _cache_get(key: str) -> Optional[CelebrityInfo]:
    with _cache_lock:
        entry = _cache.get(key)
        if entry is None:
            return None
        info, expiry = entry
        if time.monotonic() > expiry:
            del _cache[key]
            return None
        return info


def _cache_set(key: str, info: CelebrityInfo) -> None:
    with _cache_lock:
        _cache[key] = (info, time.monotonic() + settings.api_ninjas_cache_ttl_s)


def reset_cache() -> None:
    """Clear the cache. Test-only hook (mirrors `identity/index.py::reset_cache`)."""
    with _cache_lock:
        _cache.clear()


def _normalise_name(name: str) -> str:
    """lowercase, trim, collapse whitespace, drop punctuation.

    Used both as the cache key and to match API Ninjas' returned candidates
    back to the already-resolved identity name — API Ninjas' own casing/
    punctuation is never trusted as canonical.
    """
    name = name.lower().strip()
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name


def _unavailable() -> CelebrityInfo:
    return CelebrityInfo(available=False, source="api_ninjas")


def _select_best_match(results: list, resolved_name: str) -> Optional[dict]:
    """API Ninjas can return multiple candidates for an ambiguous name query.
    Never blindly take the first — pick the one whose normalised `name`
    field best matches the already-resolved identity name (exact match
    preferred; otherwise token overlap). Returns `None` if nothing is a
    plausible match, rather than guessing.
    """
    target = _normalise_name(resolved_name)
    target_tokens = set(target.split())
    best: Optional[dict] = None
    best_score = -1.0

    for item in results:
        if not isinstance(item, dict):
            continue
        candidate_name = item.get("name")
        if not isinstance(candidate_name, str) or not candidate_name:
            continue
        cand_norm = _normalise_name(candidate_name)
        if cand_norm == target:
            return item
        cand_tokens = set(cand_norm.split())
        if not cand_tokens:
            continue
        overlap = len(target_tokens & cand_tokens) / max(len(target_tokens | cand_tokens), 1)
        if overlap > best_score:
            best_score, best = overlap, item

    # Require at least half the tokens to overlap, or this is plausibly a
    # different person API Ninjas happened to also match on a partial name.
    return best if best is not None and best_score >= 0.5 else None


def _opt_str(item: dict, key: str) -> Optional[str]:
    value = item.get(key)
    return value if isinstance(value, str) and value else None


def _opt_float(item: dict, key: str) -> Optional[float]:
    value = item.get(key)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _opt_int(item: dict, key: str) -> Optional[int]:
    value = item.get(key)
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _opt_bool(item: dict, key: str) -> Optional[bool]:
    value = item.get(key)
    return value if isinstance(value, bool) else None


def _map_result(item: dict, resolved_name: str) -> CelebrityInfo:
    """API Ninjas -> FaceLink schema mapping (mission section 3)."""
    occupations = item.get("occupation")
    occupations = [o for o in occupations if isinstance(o, str)] if isinstance(occupations, list) else []
    return CelebrityInfo(
        available=True,
        # The already-resolved identity name, never API Ninjas' own
        # lowercase/differently-punctuated "name" field — celebrity
        # metadata must never relabel who was identified.
        name=resolved_name,
        nationality=_opt_str(item, "nationality"),
        occupations=occupations,
        birthday=_opt_str(item, "birthday"),
        age=_opt_int(item, "age"),
        gender=_opt_str(item, "gender"),
        height=_opt_float(item, "height"),
        net_worth=_opt_int(item, "net_worth"),
        is_alive=_opt_bool(item, "is_alive"),
        source="api_ninjas",
    )


def fetch_celebrity_info(resolved_name: str) -> CelebrityInfo:
    """Look up celebrity metadata for `resolved_name`.

    `resolved_name` must already be the Known-Person Identity system's own
    resolved name (face evidence + margin + corroboration) — this function
    only ever enriches that name with metadata, never decides or changes it.

    Always returns a `CelebrityInfo`; never raises. `available=False` for:
    no name, no API key configured, any network/timeout error, any non-200
    status (401/403/429/5xx/...), malformed JSON, an empty result array, or
    no result that plausibly matches `resolved_name`.
    """
    if not resolved_name:
        return _unavailable()
    if not settings.api_ninjas_api_key:
        return _unavailable()

    key = _cache_key(resolved_name)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    try:
        resp = httpx.get(
            settings.api_ninjas_base_url,
            params={"name": resolved_name},
            headers={"X-Api-Key": settings.api_ninjas_api_key},
            timeout=httpx.Timeout(
                connect=settings.api_ninjas_connect_timeout_s,
                read=settings.api_ninjas_read_timeout_s,
                write=settings.api_ninjas_connect_timeout_s,
                pool=settings.api_ninjas_connect_timeout_s,
            ),
        )
    except Exception as exc:  # noqa: BLE001 — a network error must never reach the scan
        log.warning("api_ninjas: network error for %r: %s: %s",
                   resolved_name, type(exc).__name__, str(exc)[:160])
        return _unavailable()

    if resp.status_code in (401, 403):
        log.warning("api_ninjas: auth error (HTTP %d) — check API_NINJAS_API_KEY", resp.status_code)
        return _unavailable()
    if resp.status_code == 429:
        log.warning("api_ninjas: rate limited (HTTP 429) for %r", resolved_name)
        return _unavailable()
    if resp.status_code != 200:
        log.warning("api_ninjas: HTTP %d for %r", resp.status_code, resolved_name)
        return _unavailable()

    try:
        payload = resp.json()
    except Exception as exc:  # noqa: BLE001 — malformed response, not our problem to crash over
        log.warning("api_ninjas: malformed JSON for %r: %s", resolved_name, type(exc).__name__)
        return _unavailable()

    if not isinstance(payload, list) or not payload:
        result = _unavailable()
        _cache_set(key, result)
        return result

    best = _select_best_match(payload, resolved_name)
    result = _map_result(best, resolved_name) if best is not None else _unavailable()
    _cache_set(key, result)
    return result
