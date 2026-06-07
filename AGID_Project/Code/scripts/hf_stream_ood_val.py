"""Stream OOD val images from bitmind/GenImage_* via hf-mirror.com.

Downloads N images per class (ai + real) for each OOD generator into:
  <root>/<generator_dir>/val/ai/      (fake/AI images, label=1)
  <root>/<generator_dir>/val/nature/  (real images, label=0)

OOD generators (NOT in Route B training): GLIDE, Wukong, VQDM, SD-1.5

Strategy:
  - Use requests to download parquet shards directly from hf-mirror.com
  - Decode image bytes from each row and save to disk
  - Stop after collecting n_per_class images per class

Usage:
    python scripts/hf_stream_ood_val.py \
        --root E:/LQiu/lab_folder/Machine_learning/AGID_Project/Data/GenImage \
        --n_per_class 1000

Set HF_TOKEN or HUGGINGFACE_TOKEN for gated datasets; public shards work without it.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

import requests
import pyarrow.parquet as pq
import pandas as pd
from PIL import Image

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
HF_MIRROR = "https://hf-mirror.com"

# Map: (bitmind_repo_suffix, num_shards) -> local directory name
# num_shards = how many parquet shards in the repo's data/ folder
GENERATORS = {
    "glide":                 ("GLIDE",                  21),
    "wukong":                ("Wukong",                157),
    "VQDM":                  ("VQDM",                   32),
    "Stable_Diffusion_V1_5": ("Stable_Diffusion_v1.5",  30),  # gated, may need auth
}

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def count_images(directory: Path) -> int:
    if not directory.exists():
        return 0
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sum(1 for p in directory.rglob("*") if p.suffix.lower() in exts)


def shard_url(repo_suffix: str, shard_idx: int, n_shards: int) -> str:
    fname = f"data/train-{shard_idx:05d}-of-{n_shards:05d}.parquet"
    return f"{HF_MIRROR}/datasets/bitmind/GenImage_{repo_suffix}/resolve/main/{fname}"


def download_shard(url: str, cache_path: Path, token: str) -> Path:
    """Download parquet shard to cache_path, skip if already exists."""
    if cache_path.exists() and cache_path.stat().st_size > 100_000:
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    r = requests.get(url, headers=headers, stream=True, timeout=120)
    r.raise_for_status()
    size = int(r.headers.get("content-length", 0))
    print(f"    GET {url.split('/')[-1]}  ({size/1e6:.1f} MB)")
    with open(cache_path, "wb") as f:
        for chunk in r.iter_content(65536):
            f.write(chunk)
    return cache_path


def save_image_bytes(img_data, out_path: Path):
    """Save image from dict/bytes/PIL to out_path."""
    if isinstance(img_data, dict) and "bytes" in img_data:
        raw = img_data["bytes"]
    elif isinstance(img_data, bytes):
        raw = img_data
    else:
        img = Image.fromarray(img_data)
        img.save(out_path, "JPEG", quality=95)
        return
    # Verify + save (preserve original format)
    out_path.write_bytes(raw)


# --------------------------------------------------------------------------- #
# Per-generator download
# --------------------------------------------------------------------------- #

def download_generator(
    repo_suffix: str,
    local_name: str,
    n_shards: int,
    root: Path,
    n_per_class: int,
    cache_dir: Path,
):
    ai_dir     = root / local_name / "val" / "ai"
    nature_dir = root / local_name / "val" / "nature"
    ai_dir.mkdir(parents=True, exist_ok=True)
    nature_dir.mkdir(parents=True, exist_ok=True)

    n_ai  = count_images(ai_dir)
    n_nat = count_images(nature_dir)

    if n_ai >= n_per_class and n_nat >= n_per_class:
        print(f"  [SKIP] already have ai={n_ai}  nature={n_nat}")
        return

    gen_cache = cache_dir / f"GenImage_{repo_suffix}"
    gen_cache.mkdir(parents=True, exist_ok=True)

    global_idx = 0  # unique file name counter

    for shard_i in range(n_shards):
        if n_ai >= n_per_class and n_nat >= n_per_class:
            break

        url        = shard_url(repo_suffix, shard_i, n_shards)
        cache_path = gen_cache / f"shard_{shard_i:05d}.parquet"

        try:
            local = download_shard(url, cache_path, HF_TOKEN)
        except requests.HTTPError as e:
            print(f"    [WARN] HTTP error shard {shard_i}: {e}")
            continue
        except Exception as e:
            print(f"    [WARN] shard {shard_i}: {e}")
            continue

        try:
            df = pd.read_parquet(local)
        except Exception as e:
            print(f"    [WARN] parquet read failed shard {shard_i}: {e}")
            continue

        if "image" not in df.columns:
            print(f"    [WARN] no 'image' column in shard {shard_i}: {list(df.columns)}")
            continue
        if "label" not in df.columns:
            print(f"    [WARN] no 'label' column in shard {shard_i}: {list(df.columns)}")
            continue

        for _, row in df.iterrows():
            if n_ai >= n_per_class and n_nat >= n_per_class:
                break
            label = int(row["label"])
            if label == 1 and n_ai < n_per_class:
                out = ai_dir / f"ood_{global_idx:07d}.jpg"
                try:
                    save_image_bytes(row["image"], out)
                    n_ai += 1
                except Exception as e:
                    print(f"      [WARN] save failed: {e}")
            elif label == 0 and n_nat < n_per_class:
                out = nature_dir / f"ood_{global_idx:07d}.jpg"
                try:
                    save_image_bytes(row["image"], out)
                    n_nat += 1
                except Exception as e:
                    print(f"      [WARN] save failed: {e}")
            global_idx += 1

        # Clean up cache shard after extraction (save disk space)
        cache_path.unlink(missing_ok=True)

        print(f"    shard {shard_i:2d}/{n_shards-1}  ai={n_ai}/{n_per_class}  nature={n_nat}/{n_per_class}")

    print(f"  [DONE] ai={n_ai}  nature={n_nat}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="GenImage root directory")
    parser.add_argument("--n_per_class", type=int, default=1000)
    parser.add_argument("--cache_dir",
                        default="E:/LQiu/lab_folder/Machine_learning/AGID_Project/Code/tmp_hf_cache",
                        help="Temporary directory for downloaded parquet shards")
    parser.add_argument("--generators", nargs="+",
                        default=list(GENERATORS.keys()),
                        help=f"Subset of: {list(GENERATORS.keys())}")
    args = parser.parse_args()

    root      = Path(args.root)
    cache_dir = Path(args.cache_dir)

    print(f"[INFO] GenImage root  : {root}")
    print(f"[INFO] n_per_class    : {args.n_per_class}")
    print(f"[INFO] cache_dir      : {cache_dir}")
    print(f"[INFO] mirror         : {HF_MIRROR}")
    print()

    for key in args.generators:
        if key not in GENERATORS:
            print(f"[WARN] Unknown generator '{key}', skip")
            continue
        local_name, n_shards = GENERATORS[key]
        print(f"{'='*60}")
        print(f"Generator : {key}  ->  {local_name}  ({n_shards} shards)")
        download_generator(key, local_name, n_shards, root, args.n_per_class, cache_dir)
        print()

    print("[ALL DONE]")
