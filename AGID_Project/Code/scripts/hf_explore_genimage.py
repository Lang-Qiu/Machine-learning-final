"""Explore GenImage on HuggingFace: list available configs / splits / files.

Run this first to understand the repo layout before downloading.

Usage:
    python scripts/hf_explore_genimage.py [--use_mirror]
"""
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--use_mirror", action="store_true", default=True,
                    help="Use hf-mirror.com (recommended for China, default True)")
parser.add_argument("--repo_id", default="xingjunm/GenImage",
                    help="HuggingFace repo ID for GenImage dataset")
args = parser.parse_args()

if args.use_mirror:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    print(f"[INFO] HF_ENDPOINT = https://hf-mirror.com")

print(f"[INFO] Exploring repo: {args.repo_id}\n")

# --- Method 1: huggingface_hub list repo files ---
try:
    from huggingface_hub import list_repo_files, repo_info
    print("=" * 60)
    print("REPO INFO")
    print("=" * 60)
    try:
        info = repo_info(args.repo_id, repo_type="dataset")
        print(f"  id       : {info.id}")
        print(f"  sha      : {info.sha[:12]}")
        print(f"  private  : {info.private}")
    except Exception as e:
        print(f"  [WARN] repo_info failed: {e}")

    print("\n" + "=" * 60)
    print("FILES IN REPO (first 200, sorted)")
    print("=" * 60)
    files = sorted(list_repo_files(args.repo_id, repo_type="dataset"))
    for f in files[:200]:
        print(f"  {f}")
    if len(files) > 200:
        print(f"  ... ({len(files)} total)")

except ImportError:
    print("[WARN] huggingface_hub not installed: pip install huggingface_hub")
except Exception as e:
    print(f"[ERROR] huggingface_hub list failed: {e}")

# --- Method 2: datasets library info ---
try:
    from datasets import get_dataset_config_names, get_dataset_split_names
    print("\n" + "=" * 60)
    print("DATASET CONFIGS")
    print("=" * 60)
    try:
        configs = get_dataset_config_names(args.repo_id, trust_remote_code=True)
        print(f"  configs: {configs}")
        for cfg in configs[:10]:
            try:
                splits = get_dataset_split_names(args.repo_id, config_name=cfg, trust_remote_code=True)
                print(f"    {cfg}: splits={splits}")
            except Exception as e:
                print(f"    {cfg}: [split check failed] {e}")
    except Exception as e:
        print(f"  [ERROR] get_dataset_config_names: {e}")
except ImportError:
    print("[WARN] datasets not installed")

print("\n[DONE] exploration complete.")
