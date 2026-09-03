"""Face-recognition calibration benchmark.

    python -m facechain.benchmark --genuine path/a.jpg path/b.jpg ...
                                  --impostor path/x.jpg path/y.jpg ...

Computes pairwise cosine similarity for every genuine pair (same person,
different photos you authorise) and every impostor pair (different people),
then reports the distributions so you can choose a threshold that is
*defensible for your deployment* rather than copied from a paper.

Why this matters
----------------
Published thresholds (e.g. "0.38 on LFW") are measured on a specific dataset
with a specific resolution and JPEG quality distribution. Your photos will
differ. A threshold calibrated on your own authorised sample set is always
more honest than one borrowed from a benchmark you did not run.

Usage
-----
Prepare at least two genuine pairs and two impostor pairs.

    # Two shots of the same person from different angles
    python -m facechain.benchmark \\
        --genuine alice_morning.jpg alice_passport.jpg \\
        --impostor alice_morning.jpg bob_linkedin.jpg

The tool prints:
  - per-pair scores
  - genuine/impostor distributions (mean, median, std)
  - suggested conservative threshold
  - false-accept / false-reject counts at that threshold
  - a reminder that calibration on ~10 pairs is illustrative, not rigorous

Output format
-------------
Results are printed to stdout. Exit 0 if calibration ran without errors, 1 if
fewer than 2 pairs of either class were provided (not enough to characterise
even a toy distribution).
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

log = logging.getLogger(__name__)

# Below this many pairs of either class, a threshold recommendation is not a
# statistically meaningful calibration — it is at best a sanity check. The
# spec this pipeline follows is explicit: show "CALIBRATION INSUFFICIENT"
# rather than dressing up a small sample as a validated operating point.
MIN_PAIRS_FOR_CALIBRATION = 50


@dataclass
class ThresholdSweepRow:
    threshold: float
    far_pct: float
    frr_pct: float
    fa_count: int
    fr_count: int


@dataclass
class CalibrationResult:
    """Structured, machine-usable calibration output.

    `run_benchmark` prints this for a human; the same object is what a caller
    (an API endpoint, a test, a future calibration-config writer) should use
    instead of scraping stdout.
    """

    n_genuine_pairs: int
    n_impostor_pairs: int
    status: str  # "CALIBRATED" | "CALIBRATION_INSUFFICIENT"
    sweep: list[ThresholdSweepRow] = field(default_factory=list)
    suggested_threshold: float = 0.38
    far_at_suggested_pct: float = 0.0
    frr_at_suggested_pct: float = 0.0
    note: str = ""


def calibrate(
    genuine_scores: list[float],
    impostor_scores: list[float],
    min_pairs: int = MIN_PAIRS_FOR_CALIBRATION,
    default_threshold: float = 0.38,
) -> CalibrationResult:
    """Sweep thresholds against labelled score distributions.

    `impostor_scores` should be cross-class (genuine-image × impostor-image)
    or true impostor-pair scores — whichever the caller has. Below
    `min_pairs` for either class, the result is explicitly marked
    CALIBRATION_INSUFFICIENT and `suggested_threshold` falls back to
    `default_threshold` rather than an unreliable estimate from too few pairs.
    """
    n_g, n_i = len(genuine_scores), len(impostor_scores)
    insufficient = n_g < min_pairs or n_i < min_pairs

    labelled = [(s, True) for s in genuine_scores] + [(s, False) for s in impostor_scores]
    sweep: list[ThresholdSweepRow] = []
    best_threshold = default_threshold
    best_equal_error = float("inf")
    best_far = best_frr = 0.0

    for t in [round(x * 0.05, 2) for x in range(4, 20)]:  # 0.20 .. 0.95
        fa = sum(1 for s, genuine in labelled if not genuine and s >= t)
        fr = sum(1 for s, genuine in labelled if genuine and s < t)
        far = fa / n_i * 100 if n_i else 0.0
        frr = fr / n_g * 100 if n_g else 0.0
        sweep.append(ThresholdSweepRow(threshold=t, far_pct=far, frr_pct=frr,
                                       fa_count=fa, fr_count=fr))
        eer_dist = abs(far - frr)
        if eer_dist < best_equal_error:
            best_equal_error = eer_dist
            best_threshold = t
            best_far, best_frr = far, frr

    if insufficient:
        note = (
            f"Only {n_g} genuine and {n_i} impostor pair(s) supplied; "
            f"{min_pairs}+ of each are needed for a defensible calibration. "
            "This sweep is illustrative only — the configured default "
            f"threshold ({default_threshold}) is reported, not the sweep's estimate."
        )
        return CalibrationResult(
            n_genuine_pairs=n_g, n_impostor_pairs=n_i, status="CALIBRATION_INSUFFICIENT",
            sweep=sweep, suggested_threshold=default_threshold,
            far_at_suggested_pct=0.0, frr_at_suggested_pct=0.0, note=note,
        )

    return CalibrationResult(
        n_genuine_pairs=n_g, n_impostor_pairs=n_i, status="CALIBRATED",
        sweep=sweep, suggested_threshold=best_threshold,
        far_at_suggested_pct=best_far, frr_at_suggested_pct=best_frr,
        note=f"Equal-error operating point across {n_g} genuine / {n_i} impostor pairs.",
    )


def _embed(path: str | Path, backend_name: str | None = None) -> np.ndarray | None:
    """Read an image, detect the primary face, return its L2-normalised embedding."""
    p = Path(path)
    if not p.exists():
        log.error("file not found: %s", p)
        return None

    from facechain.face.encoder import read_image, encode_face
    try:
        img = read_image(p)
        record, embedding, _ = encode_face(img, backend_name)
        if not record.detected or embedding is None:
            log.warning("no face detected in %s", p)
            return None
        return embedding
    except Exception as exc:  # noqa: BLE001
        log.error("could not embed %s: %s", p, exc)
        return None


def _pairwise(embeddings: list[np.ndarray]) -> list[float]:
    from facechain.face.similarity import cosine
    scores = []
    for a, b in itertools.combinations(embeddings, 2):
        scores.append(cosine(a, b))
    return scores


def _stats(scores: list[float]) -> dict:
    if not scores:
        return {"n": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}
    a = np.array(scores, dtype=np.float32)
    return {
        "n": len(scores),
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "std": float(np.std(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


def run_benchmark(
    genuine_paths: list[str],
    impostor_paths: list[str],
    backend: str | None = None,
    verbose: bool = False,
    out_path: str | None = None,
) -> int:
    """Core benchmark logic. Returns exit code."""
    print()
    print("FaceChain — face similarity calibration benchmark")
    print("=" * 60)
    print()

    if len(genuine_paths) < 2:
        print("ERROR: need at least 2 genuine images to form one pair.", file=sys.stderr)
        return 1
    if len(impostor_paths) < 2:
        print("ERROR: need at least 2 impostor images to form one pair.", file=sys.stderr)
        return 1

    # ---- embed all images ------------------------------------------------
    print("Embedding genuine images…")
    genuine_embs: list[tuple[str, np.ndarray]] = []
    for p in genuine_paths:
        emb = _embed(p, backend)
        if emb is not None:
            genuine_embs.append((p, emb))
            print(f"  ✓ {Path(p).name}")
        else:
            print(f"  ✗ {Path(p).name}  (skipped — no usable face)")

    print()
    print("Embedding impostor images…")
    impostor_embs: list[tuple[str, np.ndarray]] = []
    for p in impostor_paths:
        emb = _embed(p, backend)
        if emb is not None:
            impostor_embs.append((p, emb))
            print(f"  ✓ {Path(p).name}")
        else:
            print(f"  ✗ {Path(p).name}  (skipped — no usable face)")

    if len(genuine_embs) < 2:
        print("\nERROR: fewer than 2 genuine images could be embedded.", file=sys.stderr)
        return 1
    if len(impostor_embs) < 2:
        print("\nERROR: fewer than 2 impostor images could be embedded.", file=sys.stderr)
        return 1

    # ---- pairwise scores -------------------------------------------------
    from facechain.face.similarity import cosine

    genuine_scores: list[tuple[str, str, float]] = []
    for (pa, ea), (pb, eb) in itertools.combinations(genuine_embs, 2):
        score = cosine(ea, eb)
        genuine_scores.append((pa, pb, score))

    impostor_scores: list[tuple[str, str, float]] = []
    for (pa, ea), (pb, eb) in itertools.combinations(impostor_embs, 2):
        score = cosine(ea, eb)
        impostor_scores.append((pa, pb, score))

    # Cross-class: genuine image vs impostor image (classic FNMR/FMR pairs)
    cross_scores: list[tuple[str, str, float]] = []
    for (pg, eg) in genuine_embs:
        for (pi, ei) in impostor_embs:
            cross_scores.append((pg, pi, cosine(eg, ei)))

    try:
        from facechain.config import settings
        default_threshold = settings.face_match_threshold
    except Exception:  # noqa: BLE001
        default_threshold = 0.38

    # ---- print results ---------------------------------------------------
    print()
    print("GENUINE PAIRS  (same person, expect HIGH similarity)")
    print("-" * 60)
    for pa, pb, score in genuine_scores:
        flag = f"  ← BELOW {default_threshold}" if score < default_threshold else ""
        print(f"  {Path(pa).name:<28} × {Path(pb).name:<28}  {score:.4f}{flag}")

    print()
    print("IMPOSTOR PAIRS  (different people, expect LOW similarity)")
    print("-" * 60)
    for pa, pb, score in impostor_scores:
        flag = f"  ← ABOVE {default_threshold} (false accept risk)" if score >= default_threshold else ""
        print(f"  {Path(pa).name:<28} × {Path(pb).name:<28}  {score:.4f}{flag}")

    if verbose:
        print()
        print("CROSS-CLASS PAIRS  (genuine × impostor)")
        print("-" * 60)
        for pg, pi, score in cross_scores:
            print(f"  {Path(pg).name:<28} × {Path(pi).name:<28}  {score:.4f}")

    # ---- distributions ---------------------------------------------------
    gs = _stats([s for _, _, s in genuine_scores])
    ims = _stats([s for _, _, s in cross_scores])

    print()
    print("GENUINE distribution")
    print(f"  n={gs['n']}  mean={gs['mean']:.4f}  median={gs['median']:.4f}  "
          f"std={gs['std']:.4f}  min={gs['min']:.4f}  max={gs['max']:.4f}")
    print()
    print("IMPOSTOR distribution  (genuine × impostor cross-pairs)")
    print(f"  n={ims['n']}  mean={ims['mean']:.4f}  median={ims['median']:.4f}  "
          f"std={ims['std']:.4f}  min={ims['min']:.4f}  max={ims['max']:.4f}")

    # ---- threshold sweep + calibration verdict ----------------------------
    result = calibrate(
        [s for _, _, s in genuine_scores],
        [s for _, _, s in cross_scores],
        default_threshold=default_threshold,
    )

    print()
    print("THRESHOLD SWEEP  (false accept rate / false reject rate)")
    print(f"  {'Threshold':>10}  {'FAR %':>8}  {'FRR %':>8}  {'FA count':>9}  {'FR count':>9}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*9}")
    for row in result.sweep:
        print(f"  {row.threshold:>10.2f}  {row.far_pct:>8.1f}  {row.frr_pct:>8.1f}  "
              f"{row.fa_count:>9d}  {row.fr_count:>9d}")

    print()
    print(f"  CALIBRATION STATUS: {result.status}")
    if result.status == "CALIBRATED":
        print(f"  Approximate equal-error threshold: {result.suggested_threshold:.2f} "
              f"(FAR {result.far_at_suggested_pct:.1f}%, FRR {result.frr_at_suggested_pct:.1f}%)")
    print(f"  {result.note}")

    if out_path:
        import dataclasses
        import json
        Path(out_path).write_text(json.dumps(dataclasses.asdict(result), indent=2))
        print(f"\n  Wrote calibration result to {out_path}")
        print(f"  Set CALIBRATION_FILE={out_path} to record it on every scan's evidence.")

    print()
    print("CURRENT CONFIG")
    try:
        from facechain.config import settings
        print(f"  face_match_threshold = {settings.face_match_threshold}")
        print(f"  insightface_model    = {settings.insightface_model}")
        print(f"  face_backend         = {settings.face_backend}")
    except Exception:  # noqa: BLE001
        print("  (could not load settings)")

    print()
    print("NOTE")
    print(f"  This benchmark uses the pairs you supplied. Below {MIN_PAIRS_FOR_CALIBRATION}+ genuine")
    print(f"  and {MIN_PAIRS_FOR_CALIBRATION}+ impostor pairs, the sweep is illustrative only and the")
    print("  status above reads CALIBRATION_INSUFFICIENT rather than recommending a threshold change.")
    print("  Do NOT lower the threshold below the configured default without a rigorous evaluation.")
    print()

    return 0


def load_calibration_status(path: str) -> tuple[str, str]:
    """Read a `CalibrationResult` JSON file's status/note for `ThresholdSnapshot`.

    Never raises: a missing, unreadable, or malformed file is indistinguishable
    from "no calibration was run" — the case's threshold snapshot then honestly
    reports DEFAULT rather than crashing the scan over an optional file.
    """
    if not path:
        return "DEFAULT", "thresholds are hand-set defaults, not calibrated on authorised pairs"
    try:
        import json
        data = json.loads(Path(path).read_text())
        status = data.get("status", "DEFAULT")
        note = data.get("note", "")
        return status, note
    except Exception as exc:  # noqa: BLE001
        log.warning("could not read calibration file %s: %s", path, exc)
        return "DEFAULT", f"calibration_file set but unreadable ({type(exc).__name__})"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Calibrate face-similarity thresholds on your own image set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--genuine", nargs="+", metavar="IMG", required=True,
                    help="Two or more photos of the SAME person (same person, different shots)")
    ap.add_argument("--impostor", nargs="+", metavar="IMG", required=True,
                    help="Two or more photos of DIFFERENT people")
    ap.add_argument("--backend", choices=["auto", "insightface", "opencv"], default=None)
    ap.add_argument("-v", "--verbose", action="store_true", help="Show cross-class pairs")
    ap.add_argument("--out", metavar="PATH", default=None,
                    help="Write the CalibrationResult as JSON (point CALIBRATION_FILE at it)")
    ap.add_argument("--log-level", default="WARNING")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.WARNING),
                        format="%(levelname)s %(name)s: %(message)s")

    return run_benchmark(args.genuine, args.impostor, args.backend, args.verbose, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
