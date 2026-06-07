"""E-Delta GATE 2 — build debiased smoke dataset (JPEG-Q96, 8k/class × 4 gens).

Samples a balanced subset from each training generator, applies JPEG-Q96 re-encode,
and saves to disk under Data/GenImage_debiased_smoke/. This materialized dataset
is then used for concept label precomputation + training.

Design: JPEG-Q96 is applied BEFORE concept label computation and training —
concept labels are computed on the same pixels the model trains on (Fix #1 alignment).

Usage (from Code/):
    python scripts/edelta_build_debiased_dataset.py \
        --root ../Data/GenImage \
        --out  ../Data/GenImage_debiased_smoke \
        --n_per_class 8000 --q 96 --seed 42
"""
from __future__ import annotations

import argparse, random, sys
from io import BytesIO
from pathlib import Path

from PIL import Image
from tqdm import tqdm

GENS = ["Stable_Diffusion_v1.4", "BigGAN", "ADM", "Midjourney"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(d: Path):
    if not d.exists():
        return []
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def jpeg_recompress(img: Image.Image, q: int) -> Image.Image:
    buf = BytesIO(); img.save(buf, format="JPEG", quality=q); buf.seek(0)
    return Image.open(buf).convert("RGB")


def build_one(src_root: Path, out_root: Path, gen: str, split: str, n_per_class: int,
              q: int, rng: random.Random):
    ai_dir = src_root / gen / "train" / "ai"
    na_dir = src_root / gen / "train" / "nature"
    if not ai_dir.exists() or not na_dir.exists():
        print(f"[WARN] {gen}: ai or nature dir missing — SKIP"); return
    ai_pool = list_images(ai_dir)
    na_pool = list_images(na_dir)
    # Exclude images already used in train_25k (held-out for eval)
    subset = src_root / gen / "train_25k"
    if subset.exists():
        used_ai   = {p.name for p in list_images(subset / "ai")}
        used_real = {p.name for p in list_images(subset / "nature")}
        ai_pool   = [p for p in ai_pool   if p.name not in used_ai]
        na_pool   = [p for p in na_pool   if p.name not in used_real]
    ai_samples   = rng.sample(ai_pool, min(n_per_class, len(ai_pool)))
    na_samples   = rng.sample(na_pool, min(n_per_class, len(na_pool)))
    for cls, paths in [("ai", ai_samples), ("nature", na_samples)]:
        out_dir = out_root / gen / "train" / cls; out_dir.mkdir(parents=True, exist_ok=True)
        for p in tqdm(paths, desc=f"{gen}/{cls}"):
            try:
                img = Image.open(p).convert("RGB")
                img_q = jpeg_recompress(img, q)
                out_name = p.stem + ".jpg"
                img_q.save(out_dir / out_name, format="JPEG", quality=q)
            except Exception as e:
                print(f"[WARN] {p}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_per_class", type=int, default=8000)
    ap.add_argument("--q", type=int, default=96)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.root); out = Path(args.out)
    rng = random.Random(args.seed)
    for gen in GENS:
        build_one(src, out, gen, "train", args.n_per_class, args.q, rng)
    print(f"\n[DONE] debiased smoke dataset → {out}")


if __name__ == "__main__":
    main()
