"""
Download a small AGID test subset from HuggingFace for Day 2-3 sanity check.

Sources:
- AI-generated:  InfImagine/FakeImageDataset (val splits, streamed)
- Real:          A reliable small natural-image dataset (zh-plus/tiny-imagenet or similar)

Saves to:
    Data/test_samples/0_real/   (50 images)
    Data/test_samples/1_fake/   (50 images)

Usage:
    python download_test_samples.py --n_per_class 50 [--use_hf_mirror]
"""

import argparse
import io
import os
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parents[2] / "Data" / "test_samples"


def setup_hf_mirror():
    """Use HF mirror (hf-mirror.com) for faster access in China."""
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    print("[INFO] Using hf-mirror.com endpoint")


def download_fake_images(n: int, out_dir: Path):
    """Stream n AI-generated images from InfImagine FakeImageDataset val split."""
    from datasets import load_dataset
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Streaming {n} fake images from InfImagine/FakeImageDataset...")

    # Try several known val configs; some may fail due to archive layout
    candidates = [
        # name, split (or None)
        ("MPBench/Midjourneyv5-5K", None),
        ("MPBench/SDv15-CC30K", None),
        ("MPBench/IF-CC95K", None),
    ]

    saved = 0
    for cfg_name, split in candidates:
        if saved >= n:
            break
        try:
            print(f"[INFO]   trying config={cfg_name} ...")
            ds = load_dataset(
                "InfImagine/FakeImageDataset",
                name=cfg_name,
                split=split or "train",
                streaming=True,
                trust_remote_code=True,
            )
            for i, item in enumerate(ds):
                if saved >= n:
                    break
                # item structure unknown; try common keys
                img_data = item.get("image") or item.get("img") or item.get("png") or item.get("jpg")
                if img_data is None:
                    # Fall back: dump raw bytes if present
                    if "bytes" in item:
                        img_data = Image.open(io.BytesIO(item["bytes"]))
                    else:
                        continue
                if not isinstance(img_data, Image.Image):
                    img_data = Image.open(io.BytesIO(img_data))
                img_data = img_data.convert("RGB")
                img_data.save(out_dir / f"fake_{saved:04d}.jpg", quality=95)
                saved += 1
                if saved % 10 == 0:
                    print(f"[INFO]   saved {saved}/{n}")
        except Exception as e:
            print(f"[WARN] config {cfg_name} failed: {e}")
            continue

    return saved


def download_real_images(n: int, out_dir: Path):
    """Get n real natural-image samples from a HF tiny ImageNet-style dataset."""
    from datasets import load_dataset

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Streaming {n} real images...")

    candidates = [
        # (repo, name, split)
        ("zh-plus/tiny-imagenet", None, "valid"),
        ("Maysee/tiny-imagenet", None, "train"),
        ("evanarlian/imagenet_1k_resized_256", None, "val"),
    ]

    saved = 0
    last_err = None
    for repo, name, split in candidates:
        if saved >= n:
            break
        try:
            print(f"[INFO]   trying {repo} (split={split})...")
            ds = load_dataset(
                repo, name=name, split=split, streaming=True, trust_remote_code=True
            )
            for i, item in enumerate(ds):
                if saved >= n:
                    break
                img = item.get("image") or item.get("img")
                if img is None:
                    continue
                if hasattr(img, "convert"):
                    img = img.convert("RGB")
                else:
                    from PIL import Image
                    img = Image.open(io.BytesIO(img["bytes"]) if isinstance(img, dict) else io.BytesIO(img)).convert("RGB")
                img.save(out_dir / f"real_{saved:04d}.jpg", quality=95)
                saved += 1
                if saved % 10 == 0:
                    print(f"[INFO]   saved {saved}/{n}")
            if saved >= n:
                break
        except Exception as e:
            last_err = e
            print(f"[WARN] {repo} failed: {e}")
            continue

    if saved == 0:
        raise RuntimeError(f"Could not download any real images. Last error: {last_err}")
    return saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_per_class", type=int, default=50)
    parser.add_argument("--use_hf_mirror", action="store_true", default=True,
                        help="Use https://hf-mirror.com (recommended for China).")
    parser.add_argument("--skip_real", action="store_true")
    parser.add_argument("--skip_fake", action="store_true")
    args = parser.parse_args()

    if args.use_hf_mirror:
        setup_hf_mirror()

    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    fake_dir = DATA_ROOT / "1_fake"
    real_dir = DATA_ROOT / "0_real"

    n_fake_saved = 0
    n_real_saved = 0
    if not args.skip_fake:
        n_fake_saved = download_fake_images(args.n_per_class, fake_dir)
        print(f"[OK] fake: saved {n_fake_saved} → {fake_dir}")
    if not args.skip_real:
        n_real_saved = download_real_images(args.n_per_class, real_dir)
        print(f"[OK] real: saved {n_real_saved} → {real_dir}")

    print(f"\n[SUMMARY] real={n_real_saved}  fake={n_fake_saved}  destination={DATA_ROOT}")


if __name__ == "__main__":
    main()
