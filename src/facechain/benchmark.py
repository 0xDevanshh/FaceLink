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
from pathlib import Path
from typing import Sequence

import numpy as np

log = logging.getLogger(__name__)


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

    # ---- print results ---------------------------------------------------
    print()
    print("GENUINE PAIRS  (same person, expect HIGH similarity)")
    print("-" * 60)
    for pa, pb, score in genuine_scores:
        flag = "  ← BELOW 0.38" if score < 0.38 else ""
        print(f"  {Path(pa).name:<28} × {Path(pb).name:<28}  {score:.4f}{flag}")

    print()
    print("IMPOSTOR PAIRS  (different people, expect LOW similarity)")
    print("-" * 60)
    for pa, pb, score in impostor_scores:
        flag = "  ← ABOVE 0.38 (false accept risk)" if score >= 0.38 else ""
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

    # ---- threshold sweep -------------------------------------------------
    all_scores_with_label = (
        [(s, True) for _, _, s in genuine_scores]
        + [(s, False) for _, _, s in cross_scores]
    )

    print()
    print("THRESHOLD SWEEP  (false accept rate / false reject rate)")
    print(f"  {'Threshold':>10}  {'FAR %':>8}  {'FRR %':>8}  {'FA count':>9}  {'FR count':>9}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*9}")

    best_threshold = 0.38
    best_equal_error = float("inf")

    for t in [round(x * 0.05, 2) for x in range(4, 20)]:  # 0.20 … 0.95
        fa = sum(1 for s, genuine in all_scores_with_label if not genuine and s >= t)
        fr = sum(1 for s, genuine in all_scores_with_label if genuine and s < t)
        n_impostor = sum(1 for _, genuine in all_scores_with_label if not genuine)
        n_genuine = sum(1 for _, genuine in all_scores_with_label if genuine)
        far = fa / n_impostor * 100 if n_impostor else 0.0
        frr = fr / n_genuine * 100 if n_genuine else 0.0
        print(f"  {t:>10.2f}  {far:>8.1f}  {frr:>8.1f}  {fa:>9d}  {fr:>9d}")
        eer_dist = abs(far - frr)
        if eer_dist < best_equal_error:
            best_equal_error = eer_dist
            best_threshold = t

    print()
    print(f"  Approximate equal-error threshold: {best_threshold:.2f}")
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
    print("  This benchmark uses the pairs you supplied. The equal-error threshold")
    print("  above is illustrative. For a production deployment, use ≥50 genuine")
    print("  pairs and ≥50 impostor pairs drawn from your actual use-case images.")
    print("  Do NOT lower the threshold below 0.38 without a rigorous evaluation.")
    print()

    return 0


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
    ap.add_argument("--log-level", default="WARNING")
    args = ap.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.WARNING),
                        format="%(levelname)s %(name)s: %(message)s")

    return run_benchmark(args.genuine, args.impostor, args.backend, args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
