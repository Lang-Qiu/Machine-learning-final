"""Held-out evaluation for Route B.

For BigGAN / ADM / Midjourney (no val split), this script finds images that were
NOT selected into train_25k and uses them as a held-out test set (1k ai + 1k nature
per generator).  For SD-1.4 it uses the official val split.

Usage:
    python -m cbnet_agid.eval_heldout \
        --root    <GenImage-root> \
        --ckpt    <path-to-ckpt.pth> \
        --out     <results.json> \
        --n_held  1000
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

from .data.transforms import get_eval_transform
from .models import CBNetAGID


# --------------------------------------------------------------------------- #
# Dataset that takes an explicit file list
# --------------------------------------------------------------------------- #

class FileListDataset(Dataset):
    """Simple dataset backed by an explicit (path, label) list."""

    def __init__(self, samples: List[Tuple[Path, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (256, 256))
        if self.transform:
            img = self.transform(img)
        return {"image": img, "label": torch.tensor(label, dtype=torch.long)}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def list_images(directory: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in exts)


def heldout_samples(
    root: Path,
    generator: str,
    n_per_class: int,
    seed: int = 0,
) -> List[Tuple[Path, int]]:
    """Return samples from the held-out portion of each generator's train split.

    The held-out set = (all train images) - (images already in train_25k).
    If no train_25k exists (generator not in Route B), use the full train set.
    """
    train_dir  = root / generator / "train"
    subset_dir = root / generator / "train_25k"

    all_ai   = list_images(train_dir / "ai")
    all_real = list_images(train_dir / "nature")

    if subset_dir.exists():
        subset_ai_names   = {p.name for p in list_images(subset_dir / "ai")}
        subset_real_names = {p.name for p in list_images(subset_dir / "nature")}
        heldout_ai   = [p for p in all_ai   if p.name not in subset_ai_names]
        heldout_real = [p for p in all_real if p.name not in subset_real_names]
    else:
        heldout_ai   = all_ai
        heldout_real = all_real

    rng = random.Random(seed)
    sel_ai   = rng.sample(heldout_ai,   min(n_per_class, len(heldout_ai)))
    sel_real = rng.sample(heldout_real, min(n_per_class, len(heldout_real)))

    print(f"  {generator:<35s}  held-out ai={len(heldout_ai):>6d} → sample {len(sel_ai)}"
          f"  nature={len(heldout_real):>6d} → sample {len(sel_real)}")

    return [(p, 1) for p in sel_ai] + [(p, 0) for p in sel_real]


def val_samples(root: Path, generator: str, max_n: Optional[int] = None) -> List[Tuple[Path, int]]:
    """Return all (or up to max_n) images from the official val split."""
    val_dir = root / generator / "val"
    if not val_dir.exists():
        raise FileNotFoundError(f"val split not found: {val_dir}")
    ai   = list_images(val_dir / "ai")
    real = list_images(val_dir / "nature")
    if max_n:
        rng = random.Random(0)
        ai   = rng.sample(ai,   min(max_n, len(ai)))
        real = rng.sample(real, min(max_n, len(real)))
    print(f"  {generator:<35s}  official val  ai={len(ai)}  nature={len(real)}")
    return [(p, 1) for p in ai] + [(p, 0) for p in real]


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def evaluate(model, samples, transform, batch_size, num_workers, device, name) -> dict:
    from sklearn.metrics import accuracy_score, roc_auc_score
    ds = FileListDataset(samples, transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True)
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(dl, desc=f"  eval {name}", leave=False):
            x = batch["image"].to(device, non_blocking=True)
            out = model(x)
            all_probs.append(out["prob"].cpu().numpy())
            all_labels.append(batch["label"].numpy())
    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds  = (probs > 0.5).astype(int)
    acc    = float(accuracy_score(labels, preds))
    try:
        auc = float(roc_auc_score(labels, probs))
    except ValueError:
        auc = float("nan")
    real_mask = labels == 0
    fake_mask = labels == 1
    real_acc = float((preds[real_mask] == 0).mean()) if real_mask.any() else float("nan")
    fake_acc = float((preds[fake_mask] == 1).mean()) if fake_mask.any() else float("nan")
    r = {"name": name, "n": len(labels),
         "acc": acc, "auc": auc, "real_acc": real_acc, "fake_acc": fake_acc}
    print(f"  {name:<35s}  acc={acc*100:5.2f}  auc={auc:.3f}  "
          f"real={real_acc*100:5.2f}  fake={fake_acc*100:5.2f}  n={len(labels)}")
    return r


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",      required=True, help="GenImage root directory")
    parser.add_argument("--ckpt",      required=True, help="Path to checkpoint .pth")
    parser.add_argument("--out",       required=True, help="Output JSON path for results")
    parser.add_argument("--n_held",    type=int, default=1000,
                        help="Held-out images per class per generator (default 1000).")
    parser.add_argument("--batch_size",type=int, default=64)
    parser.add_argument("--num_workers",type=int, default=2)
    parser.add_argument("--image_size",type=int, default=256)
    parser.add_argument("--n_concepts",type=int, default=6)
    parser.add_argument("--signal_channels", type=int, default=512)
    parser.add_argument("--seed",      type=int, default=99,
                        help="RNG seed for held-out sampling (different from train seed=42).")
    args = parser.parse_args()

    root = Path(args.root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    # Load model
    print(f"\n[STEP 1] loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)
    model = CBNetAGID(
        n_concepts=args.n_concepts, pretrained=False,
        signal_channels=args.signal_channels,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"  loaded epoch {ckpt['epoch']}")

    transform = get_eval_transform(args.image_size)

    # Build eval sets
    print(f"\n[STEP 2] building eval sets (n_held={args.n_held}/class):")

    eval_sets = {}

    # SD-1.4: use official val split
    try:
        eval_sets["SD-1.4_val"] = val_samples(root, "Stable_Diffusion_v1.4")
    except FileNotFoundError as e:
        print(f"  [WARN] {e}")

    # BigGAN, ADM, Midjourney: held-out from train (exclude train_25k)
    for gen, tag in [
        ("BigGAN",    "BigGAN_heldout"),
        ("ADM",       "ADM_heldout"),
        ("Midjourney","MJ_heldout"),
    ]:
        try:
            eval_sets[tag] = heldout_samples(root, gen, args.n_held, seed=args.seed)
        except Exception as e:
            print(f"  [WARN] {gen}: {e}")

    # Evaluate
    print(f"\n[STEP 3] evaluating:")
    results = {}
    for name, samples in eval_sets.items():
        r = evaluate(model, samples, transform,
                     args.batch_size, args.num_workers, device, name)
        results[name] = r

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    accs = [r["acc"] for r in results.values()]
    aucs = [r["auc"] for r in results.values() if not np.isnan(r["auc"])]
    print(f"  mean acc : {np.mean(accs)*100:.2f}%")
    print(f"  mean AUC : {np.mean(aucs):.3f}")

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump({
            "ckpt": args.ckpt,
            "n_held": args.n_held,
            "results": results,
            "mean_acc": float(np.mean(accs)),
            "mean_auc": float(np.mean(aucs)),
        }, fh, indent=2)
    print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    main()
