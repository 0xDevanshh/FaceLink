"""Loads the offline-built known-person index once per process and caches it.

Mirrors `face/detector.py::load_backend()`'s lazy-singleton pattern: the
first call that needs the index loads it from disk and keeps it in memory;
every later call reuses the same object. Never rebuilt per request (mission
section 13).

Absence of the artifact is not an error — it is the expected state for any
checkout that hasn't run `scripts/build_identity_index.py`, and the whole
identity layer must degrade to "no reliable match" rather than raise.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import settings
from .models import KnownPerson, KnownPersonIndexManifest

log = logging.getLogger(__name__)

MANIFEST_FILENAME = "manifest.json"
EMBEDDINGS_FILENAME = "embeddings.npy"
CHECKSUM_FILENAME = "MANIFEST_SHA256"


class KnownPersonIndex:
    """In-memory view of the built known-person index.

    `embeddings` is a single `(N, D)` float32, L2-normalised array; each
    `KnownPersonReference.embedding_index` in `manifest.people[i].references`
    is a row index into it. Empty (`len(people) == 0`) when no artifact is
    present or `identity_enabled` is off — every method on an empty index
    returns "nothing found" rather than raising.
    """

    def __init__(self, manifest: KnownPersonIndexManifest, embeddings: np.ndarray) -> None:
        self.manifest = manifest
        self.embeddings = embeddings

    @property
    def people(self) -> list[KnownPerson]:
        return self.manifest.people

    @property
    def is_empty(self) -> bool:
        return len(self.manifest.people) == 0

    @classmethod
    def empty(cls) -> "KnownPersonIndex":
        return cls(
            KnownPersonIndexManifest(version="none", built_at="", embedding_dim=0),
            np.zeros((0, 0), dtype=np.float32),
        )


_lock = threading.Lock()
_cached: Optional[KnownPersonIndex] = None
_cached_dir: Optional[Path] = None


def _load_from_disk(index_dir: Path) -> KnownPersonIndex:
    manifest_path = index_dir / MANIFEST_FILENAME
    embeddings_path = index_dir / EMBEDDINGS_FILENAME
    checksum_path = index_dir / CHECKSUM_FILENAME

    if not manifest_path.exists() or not embeddings_path.exists():
        log.info("identity index not found at %s — identity recognition disabled", index_dir)
        return KnownPersonIndex.empty()

    manifest_bytes = manifest_path.read_bytes()
    if checksum_path.exists():
        expected = checksum_path.read_text(encoding="utf-8").strip()
        actual = hashlib.sha256(manifest_bytes).hexdigest()
        if expected and expected != actual:
            log.warning(
                "identity index manifest checksum mismatch (expected %s, got %s) — "
                "treating index as unavailable rather than trusting a possibly "
                "corrupt/stale artifact", expected[:12], actual[:12],
            )
            return KnownPersonIndex.empty()

    try:
        manifest = KnownPersonIndexManifest.model_validate(json.loads(manifest_bytes))
        embeddings = np.load(embeddings_path)
    except Exception as exc:  # noqa: BLE001 — a broken artifact must never crash a scan
        log.warning("failed to load identity index from %s: %s: %s",
                   index_dir, type(exc).__name__, exc)
        return KnownPersonIndex.empty()

    if embeddings.ndim != 2 or embeddings.shape[1] != manifest.embedding_dim:
        log.warning(
            "identity index embeddings shape %s does not match manifest embedding_dim=%d — "
            "treating index as unavailable", embeddings.shape, manifest.embedding_dim,
        )
        return KnownPersonIndex.empty()

    total_refs = sum(len(p.references) for p in manifest.people)
    log.info(
        "identity index loaded: %d people, %d reference embeddings (version=%s)",
        len(manifest.people), total_refs, manifest.version,
    )
    return KnownPersonIndex(manifest, embeddings.astype(np.float32, copy=False))


def get_index() -> KnownPersonIndex:
    """Return the process-wide `KnownPersonIndex`, loading it on first use.

    Returns an empty index (never raises) when `identity_enabled` is off or
    the artifact is absent/corrupt.
    """
    global _cached, _cached_dir
    index_dir = Path(settings.identity_index_dir)

    if not settings.identity_enabled:
        return KnownPersonIndex.empty()

    with _lock:
        if _cached is not None and _cached_dir == index_dir:
            return _cached
        _cached = _load_from_disk(index_dir)
        _cached_dir = index_dir
        return _cached


def reset_cache() -> None:
    """Force the next `get_index()` call to reload from disk. Test-only hook."""
    global _cached, _cached_dir
    with _lock:
        _cached = None
        _cached_dir = None
