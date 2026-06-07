"""B1-style confound test for BaselinePlain: JPEG-q95 re-encoding.

Applies JPEG compression at quality 95 to all images (both real and fake),
then evaluates detection accuracy.  If the model relies on compression artifacts,
fake_acc should drop (compression erases the signal).  If it relies on genuine
generative features, accuracy should be preserved.

Compares BaselinePlain (no bottleneck) with the CBNet result from Results/confound/jpeg-q95.json.
"""
from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import sys
_PKG_ROOT = Path(__file__).resolve().parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from cbnet_agid.data.transforms import get_eval_transform
from cbnet_agid.models.baseline_plain import BaselinePlain


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(d):
    if not d.exists(): return []
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def jpeg_reencode(img: Image.Image, quality: int = 95) -> Image.Image:
    """Re-encode image through JPEG at given quality."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


class ReencodedDataset(Dataset):
    def __init__(self, samples, transform, jpeg_quality=95):
        self.samples = samples
        self.transform = transform
        self.q = jpeg_quality
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (256, 256))
        img = jpeg_reencode(img, self.q)
        if self.transform:
            img = self.transform(img)
        return {"image": img, "label": torch.tensor(label, dtype=torch.long)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n_per_class", type=int, default=1000)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generators", nargs="+",
                        default=["Stable_Diffusion_v1.4", "BigGAN", "ADM", "Midjourney"])
    args = parser.parse_args()

    root = Path(args.root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)

    print(f"[INFO] loading BaselinePlain: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = BaselinePlain(pretrained=False, signal_channels=512).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    transform = get_eval_transform(256)

    results = {}
    for gen in args.generators:
        ai_dir = root / gen / "val" / "ai"
        na_dir = root / gen / "val" / "nature"
        if gen in ("BigGAN", "ADM", "Midjourney"):
            # Use held-out train (same as analyze_all.py)
            ai_pool = list_images(root / gen / "train" / "ai")
            na_pool = list_images(root / gen / "train" / "nature")
            subset = root / gen / "train_25k"
            if subset.exists():
                used = {p.name for p in list_images(subset / "ai")}
                ai_pool = [p for p in ai_pool if p.name not in used]
                used_r = {p.name for p in list_images(subset / "nature")}
                na_pool = [p for p in na_pool if p.name not in used_r]
        else:
            ai_pool = list_images(ai_dir)
            na_pool = list_images(na_dir)

        ai_sel = rng.sample(ai_pool, min(args.n_per_class, len(ai_pool)))
        na_sel = rng.sample(na_pool, min(args.n_per_class, len(na_pool)))
        samples = [(p, 1) for p in ai_sel] + [(p, 0) for p in na_sel]
        print(f"  {gen:<30s}  ai={len(ai_sel)}  real={len(na_sel)}  [JPEG q{args.quality}]")

        ds = ReencodedDataset(samples, transform, args.quality)
        dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2,
                        pin_memory=(device.type == "cuda"))
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(dl, desc=f"  {gen}", leave=False):
                x = batch["image"].to(device, non_blocking=True)
                out = model(x)
                all_probs.append(out["prob"].cpu().numpy())
                all_labels.append(batch["label"].numpy())
        probs = np.concatenate(all_probs)
        labels = np.concatenate(all_labels)
        preds = (probs > 0.5).astype(int)
        from sklearn.metrics import roc_auc_score
        acc = float((preds == labels).mean())
        try: auc = float(roc_auc_score(labels, probs))
        except ValueError: auc = float("nan")
        rm, fm = labels == 0, labels == 1
        real_acc = float((preds[rm] == 0).mean()) if rm.any() else float("nan")
        fake_acc = float((preds[fm] == 1).mean()) if fm.any() else float("nan")
        results[gen] = {"n": len(labels), "acc": acc, "auc": auc,
                        "real_acc": real_acc, "fake_acc": fake_acc}
        print(f"  {gen:<30s}  acc={acc*100:5.2f}%  real={real_acc*100:5.2f}%  fake={fake_acc*100:5.2f}%")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump({
            "variant": f"jpeg-q{args.quality}",
            "model_type": "BaselinePlain",
            "ckpt": args.ckpt,
            "n_per_class": args.n_per_class,
            "seed": args.seed,
            "results": results,
        }, fh, indent=2)
    print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    main()
