"""E-Delta GATE 2 — concept label preparation for debiased smoke dataset.

Step 1a: Build multi-source color prior from nature images of all 4 debiased gens.
Step 1b: For each gen, precompute raw concept labels (heuristics on debiased images).
Step 1c: Build shared P2/P98 norm across the 4 raw arrays.
Step 1d: Apply shared norm → concept_labels_shared.npy per gen.

The output files go alongside the debiased images:
  Data/GenImage_debiased_smoke/
    color_prior.npy
    shared_concept_norm.json
    <gen>/train/
      concept_raw.npy
      concept_labels_shared.npy   ← consumed by train_multigen

Usage (from Code/):
    python scripts/edelta_gate2_prep_labels.py \
        --root ../Data/GenImage_debiased_smoke \
        --generators Stable_Diffusion_v1.4 BigGAN ADM Midjourney \
        --n_prior_images 200
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from cbnet_agid.concepts.heuristics import (
    CONCEPT_NAMES,
    compute_concept_labels,
    get_real_color_prior,
    load_real_color_prior,
    normalize_concept_labels,
    set_real_color_prior,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(d: Path):
    if not d.exists():
        return []
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def center_crop_arr(img: Image.Image, n: int) -> np.ndarray:
    from torchvision import transforms as T
    img = T.Resize((n + 32, n + 32))(img)
    img = T.CenterCrop(n)(img)
    return np.array(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root of debiased dataset")
    ap.add_argument("--generators", nargs="+", default=["Stable_Diffusion_v1.4", "BigGAN", "ADM", "Midjourney"])
    ap.add_argument("--n_prior_images", type=int, default=200)
    ap.add_argument("--center_crop", type=int, default=256)
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    root = Path(args.root); gens = args.generators

    # ---- 1a: Build color prior from all gens' nature images ----
    prior_path = root / "color_prior.npy"
    if prior_path.exists():
        print(f"[1a] Loading existing prior from {prior_path}")
        load_real_color_prior(np.load(prior_path))
    else:
        print(f"[1a] Building color prior from {args.n_prior_images} nature images per gen × {len(gens)} gens")
        real_imgs = []
        for gen in gens:
            na_dir = root / gen / args.split / "nature"
            paths = list_images(na_dir)[:args.n_prior_images]
            for p in tqdm(paths, desc=f"prior/{gen}"):
                try:
                    real_imgs.append(center_crop_arr(Image.open(p).convert("RGB"), args.center_crop))
                except Exception as e:
                    print(f"[WARN] {p}: {e}")
        set_real_color_prior(real_imgs)
        print(f"[1a] Prior built from {len(real_imgs)} images")
        np.save(prior_path, get_real_color_prior())
        print(f"[1a] Saved → {prior_path}")

    # ---- 1b: Precompute raw concept labels per gen ----
    for gen in gens:
        raw_out = root / gen / args.split / "concept_raw.npy"
        if raw_out.exists():
            print(f"[1b] {gen}: raw already exists ({raw_out}), skipping")
            continue
        ai_dir = root / gen / args.split / "ai"
        na_dir = root / gen / args.split / "nature"
        ai_paths = list_images(ai_dir)
        na_paths = list_images(na_dir)
        all_paths = ai_paths + na_paths
        n = len(all_paths)
        raw = np.zeros((n, len(CONCEPT_NAMES)), dtype=np.float32)
        for i, p in enumerate(tqdm(all_paths, desc=f"labels/{gen}")):
            try:
                arr = center_crop_arr(Image.open(p).convert("RGB"), args.center_crop)
                labs = compute_concept_labels(arr, CONCEPT_NAMES)
            except Exception as e:
                print(f"[WARN] {p}: {e}")
                labs = {k: 0.0 for k in CONCEPT_NAMES}
            for k, name in enumerate(CONCEPT_NAMES):
                raw[i, k] = labs[name]
        raw_out.parent.mkdir(parents=True, exist_ok=True)
        np.save(raw_out, raw)
        print(f"[1b] {gen}: saved raw → {raw_out}  shape={raw.shape}")

    # ---- 1c: Build shared P2/P98 norm ----
    norm_path = root / "shared_concept_norm.json"
    if norm_path.exists():
        print(f"[1c] Shared norm already exists ({norm_path}), skipping")
    else:
        print("[1c] Building shared norm from raw arrays")
        arrays = []
        for gen in gens:
            rp = root / gen / args.split / "concept_raw.npy"
            a = np.load(rp)
            print(f"  +{a.shape[0]:>6d} rows from {gen}")
            arrays.append(a)
        pooled = np.concatenate(arrays, axis=0)
        stats = {}
        for k, name in enumerate(CONCEPT_NAMES):
            vals = pooled[:, k]
            lo = float(np.percentile(vals, 2))
            hi = float(np.percentile(vals, 98))
            stats[name] = {"low": lo, "high": hi}
            print(f"  {name:<20s}  P2={lo:>11.4f}  P98={hi:>11.4f}")
        norm_path.parent.mkdir(parents=True, exist_ok=True)
        with open(norm_path, "w") as fh:
            json.dump(stats, fh, indent=2)
        print(f"[1c] Saved → {norm_path}")

    # ---- 1d: Apply shared norm → concept_labels_shared.npy per gen ----
    with open(norm_path) as fh:
        stats = json.load(fh)
    for gen in gens:
        shared_out = root / gen / args.split / "concept_labels_shared.npy"
        if shared_out.exists():
            print(f"[1d] {gen}: shared labels already exist, skipping")
            continue
        raw = np.load(root / gen / args.split / "concept_raw.npy")
        normed = np.zeros_like(raw)
        for k, name in enumerate(CONCEPT_NAMES):
            normed[:, k] = normalize_concept_labels(
                raw[:, k], low=stats[name]["low"], high=stats[name]["high"], method="percentile"
            )
        np.save(shared_out, normed)
        sat_lo = float((normed <= 0.0).mean()) * 100
        sat_hi = float((normed >= 1.0).mean()) * 100
        print(f"[1d] {gen}: saved → {shared_out}  (sat_lo={sat_lo:.1f}% sat_hi={sat_hi:.1f}%)")

    print("\n[DONE] GATE 2 label preparation complete.")


if __name__ == "__main__":
    main()
