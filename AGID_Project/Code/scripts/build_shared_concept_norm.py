"""Build a shared concept-normalization across multiple generators (Route B).

The default `precompute_concept_labels.py` normalizes each subset using its own
P2/P98 percentiles. That makes train_20k, train_100k, and any future BigGAN/ADM
subsets live in DIFFERENT concept scales, breaking cross-generator consistency
and saturating concepts under OOD distribution shift.

This script consumes the already-saved `concept_raw.npy` arrays (produced by
`precompute_concept_labels.py`, no need to recompute heuristics) and either:

  - `compute`: aggregates raw values from N generators, computes shared P2/P98
    across the union, writes a single norm-stats JSON
  - `apply`:   re-normalizes one generator's raw values using a shared stats JSON,
    writes a new `concept_labels_*.npy`

Usage:

    # Step 1: aggregate stats across 3 generators
    python scripts/build_shared_concept_norm.py compute \
        --raw Data/GenImage/Stable_Diffusion_v1.4/train/concept_raw.npy \
        --raw Data/GenImage/BigGAN/train/concept_raw.npy \
        --raw Data/GenImage/ADM/train/concept_raw.npy \
        --out Data/GenImage/shared_concept_norm.json

    # Step 2: re-normalize each generator's labels using shared stats
    python scripts/build_shared_concept_norm.py apply \
        --raw   Data/GenImage/Stable_Diffusion_v1.4/train/concept_raw.npy \
        --stats Data/GenImage/shared_concept_norm.json \
        --out   Data/GenImage/Stable_Diffusion_v1.4/train/concept_labels_shared.npy
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Allow running as a script from the scripts/ directory
CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from cbnet_agid.concepts.heuristics import CONCEPT_NAMES, normalize_concept_labels


def cmd_compute(args: argparse.Namespace) -> None:
    if not args.raw:
        raise SystemExit("--raw is required (at least one .npy)")
    arrays = []
    for raw_path in args.raw:
        rp = Path(raw_path)
        if not rp.exists():
            raise SystemExit(f"raw file not found: {rp}")
        arr = np.load(rp)
        if arr.ndim != 2 or arr.shape[1] != len(CONCEPT_NAMES):
            raise SystemExit(
                f"{rp} shape {arr.shape} incompatible with K={len(CONCEPT_NAMES)}"
            )
        print(f"[INFO] +{arr.shape[0]:>7d} rows from {rp}")
        arrays.append(arr)

    pooled = np.concatenate(arrays, axis=0)
    print(f"[INFO] pooled raw shape: {pooled.shape}")

    stats = {}
    for k, name in enumerate(CONCEPT_NAMES):
        vals = pooled[:, k]
        lo = float(np.percentile(vals, args.p_low))
        hi = float(np.percentile(vals, args.p_high))
        stats[name] = {"low": lo, "high": hi}
        print(f"  {name:<20s}  P{args.p_low:.0f}={lo:>11.4f}  P{args.p_high:.0f}={hi:>11.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"\n[SAVED] shared norm stats → {out_path}")


def cmd_apply(args: argparse.Namespace) -> None:
    raw = np.load(args.raw)
    if raw.ndim != 2 or raw.shape[1] != len(CONCEPT_NAMES):
        raise SystemExit(f"{args.raw} shape {raw.shape} incompatible with K={len(CONCEPT_NAMES)}")
    with open(args.stats) as fh:
        stats = json.load(fh)
    missing = [n for n in CONCEPT_NAMES if n not in stats]
    if missing:
        raise SystemExit(f"stats file missing concepts: {missing}")

    normalized = np.zeros_like(raw)
    for k, name in enumerate(CONCEPT_NAMES):
        normalized[:, k] = normalize_concept_labels(
            raw[:, k], low=stats[name]["low"], high=stats[name]["high"], method="percentile"
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, normalized)
    print(f"[SAVED] re-normalized labels → {out_path}  shape={normalized.shape}")
    # quick sanity print: per-concept saturation fraction (clipped to 0 or 1)
    for k, name in enumerate(CONCEPT_NAMES):
        col = normalized[:, k]
        sat_lo = float((col <= 0.0).mean())
        sat_hi = float((col >= 1.0).mean())
        print(f"  {name:<20s}  sat_lo={sat_lo*100:5.2f}%  sat_hi={sat_hi*100:5.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_compute = sub.add_parser("compute", help="Pool raw values from N generators, compute shared P2/P98 → JSON")
    p_compute.add_argument("--raw", action="append", required=True,
                           help="Path to a concept_raw.npy (repeat for each generator).")
    p_compute.add_argument("--out", required=True, help="Output JSON path for shared stats.")
    p_compute.add_argument("--p_low", type=float, default=2.0, help="Low percentile (default 2.0).")
    p_compute.add_argument("--p_high", type=float, default=98.0, help="High percentile (default 98.0).")
    p_compute.set_defaults(func=cmd_compute)

    p_apply = sub.add_parser("apply", help="Re-normalize one generator's raw values using shared stats → new labels .npy")
    p_apply.add_argument("--raw", required=True, help="Input concept_raw.npy.")
    p_apply.add_argument("--stats", required=True, help="Shared stats JSON from `compute`.")
    p_apply.add_argument("--out", required=True, help="Output normalized concept_labels .npy.")
    p_apply.set_defaults(func=cmd_apply)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
