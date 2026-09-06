"""Known-person identity recognition — additive, optional layer.

Consumes the face embedding the existing pipeline already computes
(`face/encoder.py::encode_face`) and checks it against a small, offline-built
index of known public figures. Never asserts identity from face similarity
alone (see `scorer.py`), and never blocks or alters the existing
reverse-image-search / verification / ranking pipeline: every public
function here is safe to call with an empty or absent index and returns a
"no reliable match" result rather than raising.
"""

from __future__ import annotations
