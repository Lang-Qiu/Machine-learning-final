"""Create a reproducible subsample of SD-1.4 train set for debug runs.

Usage:
    python scripts/subset_sd14_train.py --source Data/GenImage/Stable_Diffusion_v1.4/train --dest Data/GenImage/Stable_Diffusion_v1.4/train_20k --per-class 10000 --seed 42
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--per-class", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action="store_true", default=False,
                        help="Copy files instead of symlinks (default: symlink on Linux, copy on Windows)")
    args = parser.parse_args()

    source = Path(args.source)
    dest = Path(args.dest)
    random.seed(args.seed)

    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    for cls_name, cls_dir in [("ai", source / "ai"), ("nature", source / "nature")]:
        if not cls_dir.exists():
            print(f"[SKIP] {cls_dir} not found")
            continue

        all_files = sorted([p for p in cls_dir.rglob("*") if p.suffix.lower() in exts])
        n = min(args.per_class, len(all_files))
        selected = random.sample(all_files, n)

        out_dir = dest / cls_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{cls_name}] copying {n}/{len(all_files)} images → {out_dir}")
        for i, src in enumerate(selected):
            dst = out_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{n}")

    print(f"\n[DONE] subset created at {dest}")
    print(f"  ai:     {len(list((dest / 'ai').rglob('*')))} files")
    print(f"  nature: {len(list((dest / 'nature').rglob('*')))} files")


if __name__ == "__main__":
    main()
