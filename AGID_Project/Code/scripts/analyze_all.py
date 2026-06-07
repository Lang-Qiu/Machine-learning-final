"""Single-pass inference dump for Batch 1 analysis (Stage 1.5 pre-Stage 2).

Runs the Route B epoch20 checkpoint over 7 generators (4 train + 3 OOD) and
saves all per-image artifacts (concepts, logit, prob, label, path) to a single
NPZ file. All A/C/D/F derived analyses (A1/A5/A6/C1/C2/C4/C5/...) read from
this dump rather than re-running inference.

Sample sources (all 1000 ai + 1000 nature per generator, deterministic seed):
    SD-1.4         : val/ai            + val/nature
    BigGAN/ADM/MJ  : held-out train/ai + held-out train/nature
    GLIDE/Wukong/VQDM : val/ai (bitmind) + SD-1.4 val/nature (shared real)

Usage:
    python -m scripts.analyze_all \
        --root  AGID_Project/Data/GenImage \
        --ckpt  AGID_Project/Logs/cbnet_multigen_main_25k_s42/ckpt_epoch20.pth \
        --out   AGID_Project/Code/Results/full_inference_dump.npz
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import sys
_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from cbnet_agid.data.transforms import get_eval_transform
from cbnet_agid.models import CBNetAGID


CONCEPT_NAMES = [
    "bitplane_lsb", "freq_radial", "color_manifold",
    "hf_noise", "jpeg_quant", "texture_geometry",
]

# Sample source mode per generator
TRAIN_GENS = ["Stable_Diffusion_v1.4", "BigGAN", "ADM", "Midjourney"]
OOD_GENS   = ["GLIDE", "Wukong", "VQDM"]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def build_samples(root: Path, gen: str, n_per_class: int, seed: int,
                  shared_real: List[Path]) -> List[Tuple[Path, int, str]]:
    """Return [(path, label, generator)] for one generator's 2*n_per_class images."""
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
    elif gen in ("GLIDE", "Wukong", "VQDM"):
        ai_pool   = list_images(root / gen / "val" / "ai")
        real_pool = shared_real
    else:
        raise ValueError(f"unknown generator: {gen}")

    if not ai_pool:
        raise FileNotFoundError(f"no AI images for {gen}")
    if not real_pool:
        raise FileNotFoundError(f"no real images for {gen}")

    ai_sel   = rng.sample(ai_pool,   min(n_per_class, len(ai_pool)))
    real_sel = rng.sample(real_pool, min(n_per_class, len(real_pool)))

    print(f"  {gen:<25s}  ai={len(ai_sel):>4d}  real={len(real_sel):>4d}")
    return [(p, 1, gen) for p in ai_sel] + [(p, 0, gen) for p in real_sel]


class InferenceDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, gen = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (256, 256))
        return {
            "image": self.transform(img),
            "label": torch.tensor(label, dtype=torch.long),
            "idx":   torch.tensor(idx, dtype=torch.long),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out",  required=True)
    ap.add_argument("--n_per_class", type=int, default=1000)
    ap.add_argument("--seed",        type=int, default=0)
    ap.add_argument("--batch_size",  type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--image_size",  type=int, default=256)
    ap.add_argument("--n_concepts",  type=int, default=6)
    ap.add_argument("--signal_channels", type=int, default=512)
    args = ap.parse_args()

    root = Path(args.root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    print(f"\n[STEP 1] loading ckpt: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = CBNetAGID(
        n_concepts=args.n_concepts, pretrained=False,
        signal_channels=args.signal_channels,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"  loaded epoch={ckpt.get('epoch','?')}")

    classifier_w = model.classifier.weight.detach().cpu().numpy().squeeze()  # [K]
    classifier_b = float(model.classifier.bias.detach().cpu().item())
    print(f"  classifier weights: {classifier_w}")
    print(f"  classifier bias:    {classifier_b:.4f}")

    transform = get_eval_transform(args.image_size)

    print(f"\n[STEP 2] shared real pool (SD-1.4 val/nature)")
    shared_real = list_images(root / "Stable_Diffusion_v1.4" / "val" / "nature")
    print(f"  pool size: {len(shared_real)}")

    print(f"\n[STEP 3] building per-generator sample lists")
    all_generators = TRAIN_GENS + OOD_GENS
    all_samples: List[Tuple[Path, int, str]] = []
    for gen in all_generators:
        try:
            all_samples.extend(build_samples(root, gen, args.n_per_class,
                                             args.seed, shared_real))
        except (FileNotFoundError, ValueError) as e:
            print(f"  [WARN] skip {gen}: {e}")

    print(f"\n[STEP 4] inference on {len(all_samples)} images")
    ds = InferenceDataset(all_samples, transform)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=(device.type == "cuda"))

    all_concepts = np.zeros((len(all_samples), args.n_concepts), dtype=np.float32)
    all_logits   = np.zeros(len(all_samples), dtype=np.float32)
    all_probs    = np.zeros(len(all_samples), dtype=np.float32)
    all_labels   = np.zeros(len(all_samples), dtype=np.int64)

    with torch.no_grad():
        for batch in tqdm(dl, desc="  inference"):
            x   = batch["image"].to(device, non_blocking=True)
            idx = batch["idx"].numpy()
            out = model(x)
            all_concepts[idx] = out["concepts"].cpu().numpy()
            all_logits[idx]   = out["logit"].cpu().numpy()
            all_probs[idx]    = out["prob"].cpu().numpy()
            all_labels[idx]   = batch["label"].numpy()

    paths       = np.array([str(s[0]) for s in all_samples])
    generators  = np.array([s[2]      for s in all_samples])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        concepts=all_concepts, logits=all_logits, probs=all_probs,
        labels=all_labels, paths=paths, generators=generators,
        concept_names=np.array(CONCEPT_NAMES),
        classifier_weight=classifier_w.astype(np.float32),
        classifier_bias=np.float32(classifier_b),
        ckpt_path=str(args.ckpt),
        ckpt_epoch=int(ckpt.get("epoch", -1)),
    )
    print(f"\n[SAVED] {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")

    print(f"\n[STEP 5] per-generator quick sanity (acc/auc)")
    from sklearn.metrics import roc_auc_score
    for gen in all_generators:
        m = generators == gen
        if not m.any():
            continue
        p = all_probs[m]
        l = all_labels[m]
        acc = float(((p > 0.5).astype(int) == l).mean())
        try:
            auc = float(roc_auc_score(l, p))
        except ValueError:
            auc = float("nan")
        print(f"  {gen:<25s}  n={m.sum():>4d}  acc={acc*100:5.2f}%  auc={auc:.3f}")


if __name__ == "__main__":
    main()
