"""Path traversal protection for evidence file operations.

All evidence filenames are generated server-side from a fixed case_id format.
This module provides the sanitisation layer that rejects any client-supplied
path that would escape the evidence root.
"""

from __future__ import annotations

import re
from pathlib import Path

# Only allow case IDs of the form case_YYYYMMDD_HHMMSS or similar safe slugs.
_SAFE_CASE_ID = re.compile(r"^case_[0-9A-Za-z_\-]{4,64}$")


class PathTraversalError(ValueError):
    """Raised when a path escapes the evidence root or contains unsafe segments."""


def safe_case_id(candidate: str) -> str:
    """Validate and return a case ID, raising PathTraversalError if unsafe."""
    if not _SAFE_CASE_ID.match(candidate):
        raise PathTraversalError(
            f"case_id {candidate!r} contains invalid characters or is outside expected format"
        )
    # Extra: reject any path separators embedded in the value.
    if "/" in candidate or "\\" in candidate or ".." in candidate:
        raise PathTraversalError(f"case_id {candidate!r} contains path separator")
    return candidate


def safe_evidence_path(root: Path, case_id: str, filename: str) -> Path:
    """Construct an evidence path and verify it stays inside `root`.

    Raises PathTraversalError if the resolved path would escape the root.
    """
    safe_case_id(case_id)
    # Filename must not contain separators or traversal sequences.
    if any(c in filename for c in ("/", "\\", "\x00")) or ".." in filename:
        raise PathTraversalError(f"filename {filename!r} contains path traversal sequence")
    resolved = (root / case_id / filename).resolve()
    root_resolved = root.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError:
        raise PathTraversalError(
            f"path {resolved} escapes evidence root {root_resolved}"
        )
    return resolved
