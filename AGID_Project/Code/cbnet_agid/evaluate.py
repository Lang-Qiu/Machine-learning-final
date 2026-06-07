"""Evaluation script — unified across CBNet-AGID, NPR, LOTA.

Computes per-generator accuracy, AUC, real-acc, fake-acc on GenImage val sets.
Reports a comparison table.

Usage:
    # Evaluate CBNet-AGID checkpoint
    python -m cbnet_agid.evaluate --method cbnet --ckpt <path> --root <GenImage-root>

    # Evaluate NPR (uses NPR's pretrained weights)
    python -m cbnet_agid.evaluate --method npr --ckpt <NPR.pth> --root <GenImage-root>

    # Evaluate LOTA (uses LOTA's pretrained weights)
    python -m cbnet_agid.evaluate --method lota --ckpt <lota_sdv14.pth> --root <GenImage-root>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from .data.genimage import GENIMAGE_GENERATORS, GenImageDataset
from .data.transforms import get_eval_transform
from .models import CBNetAGID

# Repo root (for importing NPR / LOTA from external/)
CODE_ROOT = Path(__file__).resolve().parents[1]
NPR_ROOT = CODE_ROOT / "external" / "NPR-DeepfakeDetection"
LOTA_ROOT = CODE_ROOT / "external" / "LOTA"


def load_cbnet(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    model = CBNetAGID(n_concepts=6, pretrained=False).to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"] if "model" in state else state, strict=True)
    model.eval()
    return model


def load_npr(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    sys.path.insert(0, str(NPR_ROOT))
    from networks.resnet import resnet50  # noqa: E402
    model = resnet50(num_classes=1).to(device)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    cleaned = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=True)
    model.eval()
    return model


def load_lota(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    sys.path.insert(0, str(LOTA_ROOT))
    from model import model as LOTAModel  # noqa: E402
    model = LOTAModel(pretrain=False).to(device)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    cleaned = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=True)
    model.eval()
    return model


def lota_preprocess(img: Image.Image, img_height: int = 256, patch_size: int = 32):
    """LOTA preprocessing: bit-plane composite + max-gradient patch."""
    from bit_patch import bit_patch
    from torchvision import transforms as T
    patch_np = bit_patch(img, img_height, "scaling", patch_size, "max")
    pil = Image.fromarray(patch_np)
    tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return tf(pil)


def _lota_collate(batch):
    """Custom collate: apply lota_preprocess per-image, then stack."""
    import torch
    from torchvision import transforms as T
    images, labels = [], []
    for item in batch:
        img_pil = item["image"]  # PIL Image (transform=None on dataset)
        # LOTA bit-plane preprocessing
        from bit_patch import bit_patch
        patch_np = bit_patch(img_pil, 256, "scaling", 32, "max")
        patch_pil = Image.fromarray(patch_np)
        tf = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        images.append(tf(patch_pil))
        labels.append(item["label"])
    return {"image": torch.stack(images), "label": torch.tensor(labels)}


def evaluate_single(method: str, model, loader, device,
                    lota_inverted_labels: bool = False) -> dict:
    """Compute acc/auc/AP for a single generator's val split."""
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].cpu().numpy()
            if method == "cbnet":
                out = model(x)
                prob = out["prob"].cpu().numpy()
            elif method == "npr":
                logit = model(x).flatten()
                prob = torch.sigmoid(logit).cpu().numpy()
            elif method == "lota":
                logit = model(x).flatten()
                # LOTA's convention: sigmoid > 0.5 = REAL; we want P(AI)
                prob = 1.0 - torch.sigmoid(logit).cpu().numpy()
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
        "ap":  float(average_precision_score(labels, probs)) if len(np.unique(labels)) > 1 else float("nan"),
        "real_acc": float((preds[labels == 0] == 0).mean()) if (labels == 0).any() else float("nan"),
        "fake_acc": float((preds[labels == 1] == 1).mean()) if (labels == 1).any() else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=["cbnet", "npr", "lota"])
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--root", required=True, help="GenImage root")
    parser.add_argument("--val_generators", default="all")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0,
                        help="DataLoader workers (0 = main process only, avoids shared-memory issues)")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--out", default="Results/eval_results.json")
    parser.add_argument("--max_val_samples", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    print(f"[INFO] loading model: {args.method} ← {args.ckpt}")
    if args.method == "cbnet":
        model = load_cbnet(args.ckpt, device)
    elif args.method == "npr":
        model = load_npr(args.ckpt, device)
    elif args.method == "lota":
        model = load_lota(args.ckpt, device)

    val_gens = GENIMAGE_GENERATORS if args.val_generators == "all" else args.val_generators.split(",")

    print("\n" + "=" * 86)
    print(f"{'Generator':<28s} {'n':>6s} {'Acc':>7s} {'AUC':>7s} {'AP':>7s} {'RealAcc':>9s} {'FakeAcc':>9s}")
    print("-" * 86)
    results = {}
    accs, aucs = [], []
    for g in val_gens:
        try:
            if args.method == "lota":
                # LOTA needs raw PIL images for bit-plane preprocessing
                ds = GenImageDataset(
                    root=args.root, generator=g, split="val",
                    transform=None, max_samples=args.max_val_samples,
                )
                loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True,
                                    collate_fn=_lota_collate)
            else:
                eval_tf = get_eval_transform(args.image_size)
                ds = GenImageDataset(
                    root=args.root, generator=g, split="val",
                    transform=eval_tf, max_samples=args.max_val_samples,
                )
                loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                    num_workers=args.num_workers, pin_memory=True)
        except FileNotFoundError:
            print(f"  [skip {g}] val split not found")
            continue
        res = evaluate_single(args.method, model, loader, device)
        results[g] = res
        accs.append(res["acc"])
        aucs.append(res["auc"])
        print(f"{g:<28s} {res['n']:>6d} {res['acc']*100:>6.2f}% {res['auc']:>7.3f} {res['ap']:>7.3f} {res['real_acc']*100:>8.2f}% {res['fake_acc']*100:>8.2f}%")

    if accs:
        mean_acc = float(np.mean(accs))
        valid_aucs = [a for a in aucs if not np.isnan(a)]
        mean_auc = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
        print("-" * 86)
        print(f"{'MEAN':<28s} {'':>6s} {mean_acc*100:>6.2f}% {mean_auc:>7.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"method": args.method, "ckpt": args.ckpt, "results": results}, fh, indent=2)
    print(f"\n[SAVED] {args.out}")


if __name__ == "__main__":
    main()
