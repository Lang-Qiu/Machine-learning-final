"""E-Delta GATE 3 — build full debiased dataset (JPEG-Q96, 25k/class × 4 gens + val).

Reproduces Route B's training set identically: takes the EXACT same images from
GenImage/<gen>/train_25k/{ai,nature}/, applies JPEG-Q96, saves to a new root.
Also creates held-out val splits (1k/class per training gen) from images NOT in
train_25k. OOD gens are NOT debiased — they serve as raw generalization test.

Usage (from Code/):
    python scripts/edelta_gate3_build_dataset.py \
        --src  ../Data/GenImage \
        --out  ../Data/GenImage_debiased_full \
        --n_val_per_class 1000 --q 96 --seed 42
"""
from __future__ import annotations

import argparse, random, sys
from io import BytesIO
from pathlib import Path

from PIL import Image
from tqdm import tqdm

GENS = ["Stable_Diffusion_v1.4", "BigGAN", "ADM", "Midjourney"]
OOD_GENS = ["GLIDE", "Wukong", "VQDM"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(d: Path):
    if not d.exists():
        return []
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def jpeg_recompress(img: Image.Image, q: int) -> Image.Image:
    buf = BytesIO(); img.save(buf, format="JPEG", quality=q); buf.seek(0)
    return Image.open(buf).convert("RGB")


def build_dir(src_paths, out_dir, q, desc=""):
    """Q96-encode src_paths → out_dir as JPEG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in tqdm(src_paths, desc=desc):
        try:
            img = Image.open(p).convert("RGB")
            img_q = jpeg_recompress(img, q)
            img_q.save(out_dir / (p.stem + ".jpg"), format="JPEG", quality=q)
        except Exception as e:
            print(f"[WARN] {p}: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Original GenImage root")
    ap.add_argument("--out", required=True, help="Output root for debiased dataset")
    ap.add_argument("--q", type=int, default=96)
    ap.add_argument("--n_val_per_class", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    src = Path(args.src); out = Path(args.out)
    rng = random.Random(args.seed)

    for gen in GENS:
        # ---- train_25k: exact same images as Route B ----
        src_train = src / gen / "train_25k"
        if not src_train.exists():
            print(f"[ERROR] {gen}/train_25k not found — Route B prerequisite missing"); continue
        for cls in ["ai", "nature"]:
            src_cls = src_train / cls
            if not src_cls.exists(): continue
            paths = list_images(src_cls)
            print(f"[{gen}] train_25k/{cls}: {len(paths)} images")
            out_cls = out / gen / "train_25k" / cls
            build_dir(paths, out_cls, args.q, f"Q96/{gen}/train_25k/{cls}")

        # ---- val: held-out images NOT in train_25k ----
        src_train_full = src / gen / "train"
        train25k_names = set()
        for cls in ["ai", "nature"]:
            for p in list_images(src_train / cls):
                train25k_names.add(p.name)

        for cls in ["ai", "nature"]:
            src_cls = src_train_full / cls
            if not src_cls.exists(): continue
            all_paths = list_images(src_cls)
            held_out = [p for p in all_paths if p.name not in train25k_names]
            sample = rng.sample(held_out, min(args.n_val_per_class, len(held_out)))
            print(f"[{gen}] val/{cls}: {len(sample)} held-out images")
            out_cls = out / gen / "val" / cls
            build_dir(sample, out_cls, args.q, f"Q96/{gen}/val/{cls}")

    # ---- OOD: RAW (not debiased) ---- symlink or copy val dirs
    for ood in OOD_GENS:
        ood_val = src / ood / "val"
        if not ood_val.exists():
            print(f"[WARN] {ood}/val not found — skip OOD"); continue
        # Just symlink the raw OOD val for evaluation
        import shutil
        for cls in ["ai", "nature"]:
            src_cls = ood_val / cls
            if not src_cls.exists(): continue
            out_cls = out / ood / "val" / cls
            out_cls.mkdir(parents=True, exist_ok=True)
            paths = list_images(src_cls)
            for p in tqdm(paths[:args.n_val_per_class], desc=f"copy/{ood}/val/{cls}"):
                try:
                    import shutil
                    shutil.copy2(p, out_cls / p.name)
                except Exception:
                    pass
        print(f"[{ood}] val: {args.n_val_per_class} raw images copied")

    print(f"\n[DONE] debiased full dataset → {out}")


if __name__ == "__main__":
    main()
