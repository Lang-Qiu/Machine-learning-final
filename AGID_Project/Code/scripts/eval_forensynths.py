"""Evaluate CBNet/NPR/LOTA on ForenSynths OOD generators.

ForenSynths layout: test/<generator>/0_real/  and  test/<generator>/1_fake/
This is cross-architecture OOD (ProGAN, StyleGAN, BigGAN, etc. vs SD-1.4 training).

Usage:
    python scripts/eval_forensynths.py --method cbnet --ckpt Logs/debug_run_20k_s42/ckpt_epoch4.pth --root Data/ForenSynths
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

# Add code root to path
CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

from cbnet_agid.data.transforms import get_eval_transform

FORENSYNTHS_GENERATORS = ["biggan", "deepfake", "gaugan", "stargan"]


class ForenSynthsDataset(Dataset):
    """Loads ForenSynths test/<gen>/0_real/ and test/<gen>/1_fake/."""

    def __init__(self, root: str, generator: str, transform=None, max_samples=None):
        self.root = Path(root)
        self.generator = generator
        self.transform = transform

        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        real_dir = self.root / generator / "0_real"
        fake_dir = self.root / generator / "1_fake"

        real_paths = sorted([p for p in real_dir.rglob("*") if p.suffix.lower() in exts]) if real_dir.exists() else []
        fake_paths = sorted([p for p in fake_dir.rglob("*") if p.suffix.lower() in exts]) if fake_dir.exists() else []

        if max_samples:
            random.seed(42)
            real_paths = random.sample(real_paths, min(max_samples, len(real_paths)))
            fake_paths = random.sample(fake_paths, min(max_samples, len(fake_paths)))

        self.samples = [(p, 0) for p in real_paths] + [(p, 1) for p in fake_paths]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return {"image": img, "label": torch.tensor(label, dtype=torch.long)}


def load_model(method: str, ckpt_path: str, device: torch.device):
    from cbnet_agid.evaluate import load_cbnet, load_npr, load_lota
    return {"cbnet": load_cbnet, "npr": load_npr, "lota": load_lota}[method](ckpt_path, device)


def evaluate_fs(method: str, model, loader, device) -> dict:
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", leave=False):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].cpu().numpy()
            if method == "cbnet":
                prob = model(x)["prob"].cpu().numpy()
            elif method == "npr":
                prob = torch.sigmoid(model(x).flatten()).cpu().numpy()
            elif method == "lota":
                prob = 1.0 - torch.sigmoid(model(x).flatten()).cpu().numpy()
            else:
                raise ValueError(method)
            all_probs.append(prob)
            all_labels.append(y)
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds = (probs > 0.5).astype(int)
    return {
        "n": int(len(labels)),
        "acc": float(accuracy_score(labels, preds)),
        "auc": float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else float("nan"),
        "ap": float(average_precision_score(labels, probs)) if len(np.unique(labels)) > 1 else float("nan"),
        "real_acc": float((preds[labels == 0] == 0).mean()) if (labels == 0).any() else float("nan"),
        "fake_acc": float((preds[labels == 1] == 1).mean()) if (labels == 1).any() else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["cbnet", "npr", "lota"])
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--root", required=True, help="ForenSynths root (contains test/)")
    parser.add_argument("--generators", default="biggan,deepfake,gaugan,stargan")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--out", default="Results/forensynths_eval.json")
    parser.add_argument("--lota_preprocess", action="store_true", default=False,
                        help="Use LOTA bit-plane preprocessing")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    print(f"[INFO] loading {args.method} model from {args.ckpt}")
    model = load_model(args.method, args.ckpt, device)

    generators = args.generators.split(",")
    eval_tf = None if args.lota_preprocess else get_eval_transform(args.image_size)

    print("\n" + "=" * 86)
    print(f"{'Generator':<20s} {'n':>6s} {'Acc':>7s} {'AUC':>7s} {'AP':>7s} {'RealAcc':>9s} {'FakeAcc':>9s}")
    print("-" * 86)

    results = {}
    accs, aucs = [], []
    for gen in generators:
        ds = ForenSynthsDataset(
            root=str(Path(args.root) / "test"),
            generator=gen, transform=eval_tf, max_samples=args.max_samples,
        )
        if len(ds) == 0:
            print(f"  [skip {gen}] no images")
            continue
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
        res = evaluate_fs(args.method, model, loader, device)
        results[gen] = res
        accs.append(res["acc"])
        aucs.append(res["auc"])
        print(f"{gen:<20s} {res['n']:>6d} {res['acc']*100:>6.2f}% {res['auc']:>7.3f} {res['ap']:>7.3f} {res['real_acc']*100:>8.2f}% {res['fake_acc']*100:>8.2f}%")

    if accs:
        mean_acc = float(np.mean(accs))
        valid_aucs = [a for a in aucs if not np.isnan(a)]
        mean_auc = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
        print("-" * 86)
        print(f"{'MEAN':<20s} {'':>6s} {mean_acc*100:>6.2f}% {mean_auc:>7.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"method": args.method, "ckpt": args.ckpt, "results": results}, fh, indent=2)
    print(f"\n[SAVED] {args.out}")


if __name__ == "__main__":
    main()
