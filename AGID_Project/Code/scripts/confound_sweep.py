"""Confound quantification sweep (B1/B2/B3/B5).

Re-runs eval on the 7-generator setup under different preprocessing variants
applied BEFORE the standard eval transform. All variants share the same Route B
epoch-20 checkpoint and the same sample lists (same paths/seed as analyze_all.py)
so deltas are attributable to preprocessing alone.

Variants:
    baseline    — no preprocessing (matches Batch 1 numbers, sanity)
    jpeg-q95    — re-encode raw image as JPEG quality 95 in memory
    jpeg-q75    — quality 75
    jpeg-q50    — quality 50
    jpeg-q30    — quality 30
    png         — re-encode raw image as PNG (lossless, but normalizes encoder)
    res128      — bilinear downscale to 128² then upscale to 288² (kills high-freq)
    independent_real — for OOD generators only: each gen gets disjoint 1k real images
                       from SD-1.4 val/nature (3000 total instead of shared 1000)

Usage:
    python -m scripts.confound_sweep \
        --root  AGID_Project/Data/GenImage \
        --ckpt  AGID_Project/Logs/cbnet_multigen_main_25k_s42/ckpt_epoch20.pth \
        --out   AGID_Project/Code/Results/confound/<variant>.json \
        --variant jpeg-q95
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from io import BytesIO
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from cbnet_agid.data.transforms import get_eval_transform
from cbnet_agid.models import CBNetAGID


CONCEPT_NAMES = [
    "bitplane_lsb", "freq_radial", "color_manifold",
    "hf_noise", "jpeg_quant", "texture_geometry",
]
TRAIN_GENS = ["Stable_Diffusion_v1.4", "BigGAN", "ADM", "Midjourney"]
OOD_GENS   = ["GLIDE", "Wukong", "VQDM"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(d: Path) -> List[Path]:
    if not d.exists():
        return []
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


# ---------------------------------------------------------------------------- #
# Preprocessing variants — applied to PIL Image before eval transform
# Top-level classes (must be picklable for Windows DataLoader spawn).
# ---------------------------------------------------------------------------- #

class _Identity:
    def __call__(self, img):
        return img

class _PngRoundtrip:
    def __call__(self, img):
        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        return Image.open(buf).convert("RGB")

class _JpegQuality:
    def __init__(self, q): self.q = q
    def __call__(self, img):
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=self.q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

class _ResN:
    def __init__(self, n, up=288):
        self.n = n
        self.up = up
    def __call__(self, img):
        img = img.resize((self.n, self.n), Image.BILINEAR)
        return img.resize((self.up, self.up), Image.BILINEAR)

def make_preprocess(variant: str):
    if variant == "baseline":
        return _Identity()
    if variant == "png":
        return _PngRoundtrip()
    if variant.startswith("jpeg-q"):
        return _JpegQuality(int(variant.split("q")[1]))
    if variant.startswith("res"):
        n = int(variant.replace("res", ""))
        return _ResN(n=n, up=288)
    raise ValueError(f"unknown variant: {variant}")


# ---------------------------------------------------------------------------- #
# Sample building
# ---------------------------------------------------------------------------- #

def shared_real_pool(root: Path) -> List[Path]:
    return list_images(root / "Stable_Diffusion_v1.4" / "val" / "nature")


def build_samples_one_gen(root: Path, gen: str, n_per_class: int, seed: int,
                          shared_real: List[Path]
                          ) -> List[Tuple[Path, int]]:
    rng = random.Random(seed)
    if gen == "Stable_Diffusion_v1.4":
        ai_pool   = list_images(root / gen / "val" / "ai")
        real_pool = list_images(root / gen / "val" / "nature")
    elif gen in ("BigGAN", "ADM", "Midjourney"):
        train_ai   = list_images(root / gen / "train" / "ai")
        train_real = list_images(root / gen / "train" / "nature")
        subset = root / gen / "train_25k"
        if subset.exists():
            used_ai   = {p.name for p in list_images(subset / "ai")}
            used_real = {p.name for p in list_images(subset / "nature")}
            ai_pool   = [p for p in train_ai   if p.name not in used_ai]
            real_pool = [p for p in train_real if p.name not in used_real]
        else:
            ai_pool, real_pool = train_ai, train_real
    elif gen in OOD_GENS:
        ai_pool   = list_images(root / gen / "val" / "ai")
        real_pool = shared_real
    else:
        raise ValueError(gen)

    ai   = rng.sample(ai_pool,   min(n_per_class, len(ai_pool)))
    real = rng.sample(real_pool, min(n_per_class, len(real_pool)))
    return [(p, 1) for p in ai] + [(p, 0) for p in real]


def build_independent_real_for_ood(root: Path, n_per_class: int
                                   ) -> dict[str, List[Tuple[Path, int]]]:
    """Each OOD generator gets a disjoint 1k real subset from SD-1.4 val/nature."""
    pool = shared_real_pool(root)
    rng  = random.Random(123)
    rng.shuffle(pool)
    chunks = [pool[i*n_per_class:(i+1)*n_per_class] for i in range(len(OOD_GENS))]
    result = {}
    for gen, chunk in zip(OOD_GENS, chunks):
        ai = list_images(root / gen / "val" / "ai")
        rng2 = random.Random(0)
        ai = rng2.sample(ai, min(n_per_class, len(ai)))
        result[gen] = [(p, 1) for p in ai] + [(p, 0) for p in chunk]
    return result


# ---------------------------------------------------------------------------- #
# Dataset
# ---------------------------------------------------------------------------- #

class ConfoundDataset(Dataset):
    def __init__(self, samples, preprocess, transform):
        self.samples = samples
        self.preprocess = preprocess
        self.transform  = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            img = self.preprocess(img)
        except Exception:
            img = Image.new("RGB", (256, 256))
        return {
            "image": self.transform(img),
            "label": torch.tensor(label, dtype=torch.long),
            "idx":   torch.tensor(idx, dtype=torch.long),
        }


# ---------------------------------------------------------------------------- #
# Eval
# ---------------------------------------------------------------------------- #

def eval_one(model, samples, preprocess, transform, batch_size, num_workers,
             device, name) -> dict:
    from sklearn.metrics import roc_auc_score
    ds = ConfoundDataset(samples, preprocess, transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers,
                    pin_memory=(device.type == "cuda"))
    n = len(samples)
    K = 6
    probs    = np.zeros(n, dtype=np.float32)
    labels   = np.zeros(n, dtype=np.int64)
    concepts = np.zeros((n, K), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dl, desc=f"  {name}", leave=False):
            x = batch["image"].to(device, non_blocking=True)
            i = batch["idx"].numpy()
            out = model(x)
            probs[i]    = out["prob"].cpu().numpy()
            concepts[i] = out["concepts"].cpu().numpy()
            labels[i]   = batch["label"].numpy()
    preds = (probs > 0.5).astype(int)
    acc   = float((preds == labels).mean())
    try:
        auc = float(roc_auc_score(labels, probs))
    except ValueError:
        auc = float("nan")
    real_m = labels == 0
    fake_m = labels == 1
    real_acc = float((preds[real_m] == 0).mean()) if real_m.any() else float("nan")
    fake_acc = float((preds[fake_m] == 1).mean()) if fake_m.any() else float("nan")
    concept_means = {n: float(concepts[:, k].mean()) for k, n in enumerate(CONCEPT_NAMES)}
    print(f"  {name:<28s}  acc={acc*100:5.2f}  auc={auc:.4f}  "
          f"real={real_acc*100:5.2f}  fake={fake_acc*100:5.2f}")
    return {
        "name": name, "n": int(n),
        "acc": acc, "auc": auc, "real_acc": real_acc, "fake_acc": fake_acc,
        "concept_means": concept_means,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out",  required=True)
    ap.add_argument("--variant", required=True,
                    choices=["baseline", "jpeg-q95", "jpeg-q75", "jpeg-q50",
                             "jpeg-q30", "png",
                             "res64", "res128", "res192", "res384", "res512",
                             "independent_real"])
    ap.add_argument("--n_per_class", type=int, default=1000)
    ap.add_argument("--seed",        type=int, default=0)
    ap.add_argument("--batch_size",  type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--image_size",  type=int, default=256)
    args = ap.parse_args()

    root = Path(args.root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}  variant={args.variant}")

    print(f"\n[STEP 1] loading ckpt: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = CBNetAGID(n_concepts=6, pretrained=False, signal_channels=512).to(device)
    model.load_state_dict(ckpt["model"])

    transform = get_eval_transform(args.image_size)
    shared_real = shared_real_pool(root)

    if args.variant == "independent_real":
        preprocess = make_preprocess("baseline")
        eval_sets = build_independent_real_for_ood(root, args.n_per_class)
        # tag: which 1k subset went where
        subset_info = {gen: [str(p) for p, _ in s if _ == 0][:3] for gen, s in eval_sets.items()}
    else:
        preprocess = make_preprocess(args.variant)
        eval_sets = {}
        for gen in TRAIN_GENS + OOD_GENS:
            try:
                eval_sets[gen] = build_samples_one_gen(
                    root, gen, args.n_per_class, args.seed, shared_real)
            except Exception as e:
                print(f"  [WARN] skip {gen}: {e}")
        subset_info = None

    print(f"\n[STEP 2] eval {len(eval_sets)} generators under variant={args.variant}")
    results = {}
    for gen, samples in eval_sets.items():
        r = eval_one(model, samples, preprocess, transform,
                     args.batch_size, args.num_workers, device, gen)
        results[gen] = r

    accs = [r["acc"] for r in results.values()]
    aucs = [r["auc"] for r in results.values() if not np.isnan(r["auc"])]
    print(f"\n  MEAN  acc={np.mean(accs)*100:.2f}%  AUC={np.mean(aucs):.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "variant": args.variant,
        "ckpt": args.ckpt,
        "n_per_class": args.n_per_class,
        "seed": args.seed,
        "results": results,
        "mean_acc": float(np.mean(accs)),
        "mean_auc": float(np.mean(aucs)),
    }
    if subset_info:
        payload["independent_subset_samples"] = subset_info
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[SAVED] {out_path}")


if __name__ == "__main__":
    main()
