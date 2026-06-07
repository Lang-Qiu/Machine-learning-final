"""OOD evaluation on unseen generators (GLIDE, Wukong, VQDM, SD-1.5).

Uses images downloaded into <root>/<generator>/val/{ai,nature}/ by
scripts/hf_stream_ood_val.py.  Evaluates the Route B checkpoint
(trained on SD-1.4 + BigGAN + ADM + Midjourney) on these held-out generators.

Usage:
    python -m cbnet_agid.eval_ood \
        --root    <GenImage-root> \
        --ckpt    <path-to-ckpt.pth> \
        --out     Results/ood_eval.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .data.transforms import get_eval_transform
from .models import CBNetAGID


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #

class FileListDataset(Dataset):
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


def val_samples(
    root: Path,
    generator_dir: str,
    n_per_class: Optional[int] = None,
    seed: int = 0,
    real_dir: Optional[Path] = None,
) -> List[Tuple[Path, int]]:
    """Build (path, label) list for a generator's val split.

    If real_dir is provided, use it instead of <root>/<generator_dir>/val/nature/.
    This allows sharing a common real-image pool across OOD generators
    (all GenImage generators share the same ImageNet real images).
    """
    val_dir = root / generator_dir / "val"
    ai_dir  = val_dir / "ai"
    if not ai_dir.exists() or not list_images(ai_dir):
        raise FileNotFoundError(f"No AI images in {ai_dir}")

    # Real images: use shared pool if provided, else generator-local
    nat_dir = real_dir if real_dir is not None else (val_dir / "nature")
    if not nat_dir.exists():
        raise FileNotFoundError(f"real dir not found: {nat_dir}")

    ai   = list_images(ai_dir)
    real = list_images(nat_dir)
    if n_per_class:
        rng = random.Random(seed)
        ai   = rng.sample(ai,   min(n_per_class, len(ai)))
        real = rng.sample(real, min(n_per_class, len(real)))
    print(f"  {generator_dir:<35s}  val ai={len(ai)}  nature={len(real)}"
          + ("  [shared real]" if real_dir else ""))
    return [(p, 1) for p in ai] + [(p, 0) for p in real]


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

def evaluate(model, samples, transform, batch_size, num_workers, device, name) -> dict:
    from sklearn.metrics import accuracy_score, roc_auc_score
    ds = FileListDataset(samples, transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=(device.type == "cuda"))
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
    acc    = float((preds == labels).mean())
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(labels, probs))
    except ValueError:
        auc = float("nan")
    real_mask = labels == 0
    fake_mask = labels == 1
    real_acc = float((preds[real_mask] == 0).mean()) if real_mask.any() else float("nan")
    fake_acc = float((preds[fake_mask] == 1).mean()) if fake_mask.any() else float("nan")
    r = {
        "name": name, "n": int(len(labels)),
        "acc": acc, "auc": auc, "real_acc": real_acc, "fake_acc": fake_acc,
    }
    print(f"  {name:<35s}  acc={acc*100:5.2f}  auc={auc:.3f}  "
          f"real={real_acc*100:5.2f}  fake={fake_acc*100:5.2f}  n={len(labels)}")
    return r


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",         required=True, help="GenImage root directory")
    parser.add_argument("--ckpt",         required=True, help="Path to checkpoint .pth")
    parser.add_argument("--out",          required=True, help="Output JSON path")
    parser.add_argument("--n_per_class",  type=int, default=1000,
                        help="Max images per class per generator (default 1000)")
    parser.add_argument("--batch_size",   type=int, default=64)
    parser.add_argument("--num_workers",  type=int, default=2)
    parser.add_argument("--image_size",   type=int, default=256)
    parser.add_argument("--n_concepts",   type=int, default=6)
    parser.add_argument("--signal_channels", type=int, default=512)
    # OOD generators to evaluate
    parser.add_argument("--generators", nargs="+",
                        default=["GLIDE", "Wukong", "VQDM", "Stable_Diffusion_v1.5"],
                        help="Generator local directory names under root/")
    parser.add_argument("--real_dir", default=None,
                        help="Shared real-image directory (e.g. SD-1.4 val/nature). "
                             "If omitted, uses each generator's own val/nature.")
    args = parser.parse_args()

    root   = Path(args.root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    # Load model
    print(f"\n[STEP 1] loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = CBNetAGID(
        n_concepts=args.n_concepts, pretrained=False,
        signal_channels=args.signal_channels,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    print(f"  loaded epoch {ckpt['epoch']}")

    transform = get_eval_transform(args.image_size)

    real_dir = Path(args.real_dir) if args.real_dir else None
    if real_dir:
        print(f"[INFO] shared real_dir: {real_dir}")

    # Build eval sets
    print(f"\n[STEP 2] building eval sets (n_per_class={args.n_per_class}):")
    eval_sets = {}
    for gen_dir in args.generators:
        try:
            samples = val_samples(root, gen_dir, args.n_per_class, real_dir=real_dir)
            eval_sets[gen_dir] = samples
        except FileNotFoundError as e:
            print(f"  [WARN] {gen_dir}: {e}")
        except Exception as e:
            print(f"  [ERROR] {gen_dir}: {e}")

    if not eval_sets:
        print("\n[ERROR] No eval sets found. Run hf_stream_ood_val.py first.")
        return

    # Evaluate
    print(f"\n[STEP 3] evaluating {len(eval_sets)} OOD generators:")
    results = {}
    for name, samples in eval_sets.items():
        r = evaluate(model, samples, transform,
                     args.batch_size, args.num_workers, device, name)
        results[name] = r

    # Summary
    accs = [r["acc"] for r in results.values()]
    aucs = [r["auc"] for r in results.values() if not np.isnan(r["auc"])]
    print(f"\n{'='*60}")
    print("OOD SUMMARY (generators not seen during training)")
    print(f"{'='*60}")
    for name, r in results.items():
        print(f"  {name:<35s}  acc={r['acc']*100:5.2f}%  AUC={r['auc']:.3f}")
    print(f"  {'MEAN':<35s}  acc={np.mean(accs)*100:5.2f}%  AUC={np.mean(aucs):.3f}")

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump({
            "ckpt": args.ckpt,
            "n_per_class": args.n_per_class,
            "generators": args.generators,
            "results": results,
            "mean_acc":  float(np.mean(accs)),
            "mean_auc":  float(np.mean(aucs)),
        }, fh, indent=2)
    print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    main()
