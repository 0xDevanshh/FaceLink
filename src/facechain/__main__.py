"""Entry point for ``python -m facechain <command>``.

Commands
--------
benchmark-face   Calibrate face-similarity thresholds on your own image set.
verify-evidence  Independently verify a local evidence bundle.

Usage
-----
    python -m facechain benchmark-face \\
        --genuine alice_a.jpg alice_b.jpg \\
        --impostor alice_a.jpg bob_b.jpg

    python -m facechain verify-evidence \\
        --case evidence/case_20260901_123456
"""

from __future__ import annotations

import sys
from pathlib import Path

# When the package is installed (pip install -e .) or the src/ directory is on
# sys.path (e.g. via PYTHONPATH=src), __file__ is already inside the importable
# package and nothing extra is needed.
#
# When run as ``python -m facechain`` from the repository root WITHOUT installation,
# Python adds the *parent of the directory containing __main__.py* to sys.path —
# which is src/facechain, not src/. The outer src/ directory must be on the path
# so that ``import facechain.benchmark`` resolves. We add it here if needed.
_src_dir = Path(__file__).resolve().parent.parent   # .../src/
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 0

    cmd = sys.argv[1]

    if cmd == "benchmark-face":
        from facechain.benchmark import main as bench_main
        return bench_main(sys.argv[2:])

    if cmd == "verify-evidence":
        script = Path(__file__).resolve().parents[2] / "scripts" / "verify_attestation.py"
        if script.exists():
            import runpy
            sys.argv = [str(script), *sys.argv[2:]]
            runpy.run_path(str(script), run_name="__main__")
            return 0
        print(f"ERROR: script not found at {script}", file=sys.stderr)
        return 1

    print(f"Unknown command: {cmd!r}", file=sys.stderr)
    print("Available commands: benchmark-face, verify-evidence")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
