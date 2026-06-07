"""Pre-compute heuristic concept labels for a dataset.

For each image in the dataset, computes the 6 heuristic concept values, normalizes them
to [0, 1] using min-max from a training subset, and saves the result as a .npy file.

This avoids recomputing heuristics every epoch (saving ~30-60s per epoch on typical
dataset sizes).

Usage:
    python -m cbnet_agid.precompute_concept_labels --root <GenImage-root> --generator Stable_Diffusion_v1.4 --split train
    # Output: <root>/<generator>/<split>/concept_labels.npy + concept_norm.json (min/max per concept)
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

from .concepts.heuristics import (
    CONCEPT_NAMES,
    compute_concept_labels,
    get_real_color_prior,
    load_real_color_prior,
    normalize_concept_labels,
    set_real_color_prior,
)
from .data.genimage import GenImageDataset


def _load_image_array(path, center_crop: Optional[int]):
    """Load an image as uint8 RGB ndarray, optionally resize+center-crop first.

    The crop, when enabled, mirrors get_eval_transform: Resize((size+32, size+32))
    then CenterCrop(size). This aligns the heuristic input with the actual region
    the model sees at inference and approximates the center of the train-time RandomCrop.
    """
    img = Image.open(path).convert("RGB")
    if center_crop is not None:
        from torchvision import transforms as T
        img = T.Resize((center_crop + 32, center_crop + 32))(img)
        img = T.CenterCrop(center_crop)(img)
    return np.array(img)


def _gather_extra_real_paths(extra_dirs, n_each: int) -> list:
    """Gather up to n_each images from each of extra_dirs (used to widen color prior)."""
    if not extra_dirs:
        return []
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    out: list = []
    for d in extra_dirs:
        dp = Path(d)
        if not dp.exists():
            print(f"[WARN] extra real dir not found, skipping: {dp}")
            continue
        paths = sorted([p for p in dp.rglob("*") if p.suffix.lower() in exts])[:n_each]
        out.extend(paths)
        print(f"[INFO] +{len(paths)} images from {dp}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="GenImage root directory")
    parser.add_argument("--generator", default="Stable_Diffusion_v1.4")
    parser.add_argument("--split", default="train")
    parser.add_argument("--norm_stats_path", default=None,
                        help="Path to JSON with min/max per concept. If provided, use these "
                             "instead of computing from the data (e.g., for val using train stats).")
    parser.add_argument("--n_prior_images", type=int, default=500,
                        help="How many real images to use for the color prior (per source dir).")
    parser.add_argument("--n_workers", type=int, default=1,
                        help="Sequential = 1 (heuristics are NumPy, CPU-bound).")
    # --- Fix #1: align heuristic input with what the model sees ---
    parser.add_argument("--center_crop", type=int, default=None,
                        help="If set, resize to (N+32,N+32) and center-crop N before computing "
                             "heuristics. Mirrors get_eval_transform and approximates train-time "
                             "RandomCrop center. Recommended for Route B; default None preserves "
                             "Route A precompute behavior (heuristic on full-resolution image).")
    # --- Fix #3: persistent / multi-source color prior + decouple from norm_stats path ---
    parser.add_argument("--color_prior_in", default=None,
                        help="Path to a precomputed color prior .npy. If provided, load it instead "
                             "of building from real images. Recommended for Route B so all "
                             "generators share one prior.")
    parser.add_argument("--color_prior_out", default=None,
                        help="If set, save the (built or loaded) color prior to this .npy path.")
    parser.add_argument("--extra_real_dirs", nargs="+", default=None,
                        help="Extra directories of real images to include when building the color "
                             "prior. Useful to widen the prior across multiple generators' nature "
                             "subsets so it is not SD-1.4-locked.")
    args = parser.parse_args()

    ds = GenImageDataset(
        root=args.root, generator=args.generator, split=args.split,
        transform=None, return_path=True,
    )

    # ----- Color prior: ALWAYS built or loaded (bug fix: previously skipped when --norm_stats_path was set) -----
    if args.color_prior_in is not None:
        print(f"[INFO] Loading color prior from {args.color_prior_in}")
        prior_array = np.load(args.color_prior_in)
        load_real_color_prior(prior_array)
    else:
        print(f"[INFO] Building real-color prior from up to {args.n_prior_images} real images "
              f"(+ extras from {len(args.extra_real_dirs) if args.extra_real_dirs else 0} dirs)...")
        real_paths = [p for (p, lbl) in ds.samples if lbl == 0][: args.n_prior_images]
        real_paths.extend(_gather_extra_real_paths(args.extra_real_dirs, args.n_prior_images))
        real_imgs = []
        for p in tqdm(real_paths, desc="prior"):
            try:
                real_imgs.append(_load_image_array(p, args.center_crop))
            except Exception as e:
                print(f"[WARN] {p}: {e}")
        set_real_color_prior(real_imgs)
        print(f"[INFO] Color prior built from {len(real_imgs)} images")

    if args.color_prior_out is not None:
        prior_array = get_real_color_prior()
        if prior_array is None:
            print("[WARN] No prior available to save (set_real_color_prior produced None).")
        else:
            Path(args.color_prior_out).parent.mkdir(parents=True, exist_ok=True)
            np.save(args.color_prior_out, prior_array)
            print(f"[SAVED] color prior → {args.color_prior_out}")

    if args.norm_stats_path is not None:
        print(f"[INFO] Will normalize using existing norm stats from {args.norm_stats_path}")

    # ----- Compute raw heuristic values for every image -----
    n = len(ds)
    raw = np.zeros((n, len(CONCEPT_NAMES)), dtype=np.float32)
    pbar = tqdm(range(n), desc=f"{args.generator}/{args.split}")
    for i in pbar:
        item = ds[i]
        path = item["path"]
        try:
            img = _load_image_array(path, args.center_crop)
        except Exception as e:
            pbar.write(f"[WARN] {path}: {e}")
            continue
        labels = compute_concept_labels(img, CONCEPT_NAMES)
        for k, name in enumerate(CONCEPT_NAMES):
            raw[i, k] = labels[name]
        del img, labels
        if (i + 1) % 100 == 0:
            gc.collect()

    # ----- Normalize each concept independently -----
    if args.norm_stats_path is None:
        stats = {}
        normalized = np.zeros_like(raw)
        for k, name in enumerate(CONCEPT_NAMES):
            vals = raw[:, k]
            lo = float(np.percentile(vals, 2))
            hi = float(np.percentile(vals, 98))
            stats[name] = {"low": lo, "high": hi}
            normalized[:, k] = normalize_concept_labels(vals, low=lo, high=hi, method="percentile")
    else:
        with open(args.norm_stats_path) as fh:
            stats = json.load(fh)
        normalized = np.zeros_like(raw)
        for k, name in enumerate(CONCEPT_NAMES):
            normalized[:, k] = normalize_concept_labels(
                raw[:, k], low=stats[name]["low"], high=stats[name]["high"], method="percentile"
            )

    # ----- Save outputs -----
    out_dir = Path(args.root) / args.generator / args.split
    out_labels = out_dir / "concept_labels.npy"
    out_raw = out_dir / "concept_raw.npy"
    out_stats = out_dir / "concept_norm.json"
    np.save(out_labels, normalized)
    np.save(out_raw, raw)
    with open(out_stats, "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"\n[SAVED] {out_labels}  shape={normalized.shape}")
    print(f"[SAVED] {out_raw}")
    print(f"[SAVED] {out_stats}")


if __name__ == "__main__":
    main()
