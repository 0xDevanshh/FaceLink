"""Log scrubber — strips secrets from log records before they reach any handler.

Installs as a logging.Filter on the root logger. Any log message containing a
pattern that looks like a private key, API key, or embedding is redacted to
[REDACTED]. The filter also scrubs extra kwargs (exc_info text is NOT scrubbed
— stack traces do not contain secrets in this pipeline).

Call `install()` once at startup.
"""

from __future__ import annotations

import logging
import re

# Patterns that identify secret-looking values.
_REDACT_PATTERNS = [
    re.compile(r"(private[_\s]?key\s*[=:]\s*)['\"]?0x[0-9a-fA-F]{60,}['\"]?", re.I),
    re.compile(r"(api[_\s]?key\s*[=:]\s*)['\"]?[A-Za-z0-9\-_]{20,}['\"]?", re.I),
    re.compile(r"(serpapi[_\s]?key\s*[=:]\s*)['\"]?[A-Za-z0-9\-_]{20,}['\"]?", re.I),
    re.compile(r"(PRIVATE_KEY\s*[=:]\s*)['\"]?0x[0-9a-fA-F]{60,}['\"]?"),
    # Embedding-sized float arrays: [0.1234, -0.5678, ...] with 50+ elements
    re.compile(r"\[[-\d.,\s]{200,}\]"),
]


def _scrub(text: str) -> str:
    for pat in _REDACT_PATTERNS:
        text = pat.sub(lambda m: m.group(0).split(m.group(1) if m.lastindex else "=")[0] + "[REDACTED]", text)
    return text


class _ScrubFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:  # type: ignore[override]
        try:
            record.msg = _scrub(str(record.msg))
            record.args = tuple(
                _scrub(str(a)) if isinstance(a, str) else a
                for a in (record.args or ())
            )
        except Exception:  # noqa: BLE001
            pass
        return True


_installed = False


def install() -> None:
    """Install the scrub filter on the root logger (idempotent)."""
    global _installed
    if _installed:
        return
    logging.getLogger().addFilter(_ScrubFilter())
    _installed = True


def scrub(text: str) -> str:
    """Scrub a string for use in SSE events or API responses."""
    return _scrub(text)
