"""F5: per-concept spatial heatmaps for representative samples.

CBNet-AGID's ConceptBottleneckLayer emits both a scalar concept vector AND
per-concept spatial heatmaps of shape [B, K, H', W']. We can visualize these
directly — no Grad-CAM gradient computation needed — because they ARE the
intrinsic concept-activation maps the architecture computes.

For each selected sample we render:
    [original | heatmap_concept_1 | ... | heatmap_concept_6]
with each heatmap upscaled from 7×7 to 256×256 and α-blended on the image.

Sample selection: from full_inference_dump.npz pick a balanced set:
    - 2× SD-1.4 / 2× BigGAN / 2× ADM / 2× Midjourney correct-classified
      (one real + one fake each)
    - 2× GLIDE / 2× Wukong / 2× VQDM correct-classified OOD
    - Plus all misclassified samples (≤35 total, all are interesting)

Usage:
    python -m scripts.concept_heatmaps \
        --dump  Code/Results/full_inference_dump.npz \
        --ckpt  AGID_Project/Logs/cbnet_multigen_main_25k_s42/ckpt_epoch20.pth \
        --out_dir AGID_Project/Code/Results/batch1_analysis/F5_heatmaps
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

_PKG_ROOT = Path(__file__).resolve().parent.parent
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from cbnet_agid.data.transforms import get_eval_transform
from cbnet_agid.models import CBNetAGID

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONCEPT_NAMES = ["bitplane_lsb", "freq_radial", "color_manifold",
                 "hf_noise", "jpeg_quant", "texture_geometry"]


def pick_samples(dump_data) -> List[dict]:
    """Build sample list: 2 per training gen (1 real + 1 fake), 2 per OOD gen,
    plus all misclassified."""
    paths      = dump_data["paths"]
    labels     = dump_data["labels"]
    probs      = dump_data["probs"]
    generators = dump_data["generators"]
    preds = (probs > 0.5).astype(int)

    rng = np.random.default_rng(42)
    picks = []

    all_gens = ["Stable_Diffusion_v1.4", "BigGAN", "ADM", "Midjourney",
                "GLIDE", "Wukong", "VQDM"]
    for g in all_gens:
        m = generators == g
        correct = m & (preds == labels)
        # 1 correct real
        idx_real = np.where(correct & (labels == 0))[0]
        if len(idx_real):
            i = int(rng.choice(idx_real))
            picks.append({"path": str(paths[i]), "label": 0, "pred": int(preds[i]),
                          "prob": float(probs[i]), "generator": g,
                          "tag": "correct_real"})
        # 1 correct fake
        idx_fake = np.where(correct & (labels == 1))[0]
        if len(idx_fake):
            i = int(rng.choice(idx_fake))
            picks.append({"path": str(paths[i]), "label": 1, "pred": int(preds[i]),
                          "prob": float(probs[i]), "generator": g,
                          "tag": "correct_fake"})

    # Misclassified (all)
    miscls_idx = np.where(preds != labels)[0]
    for i in miscls_idx:
        picks.append({"path": str(paths[i]), "label": int(labels[i]),
                      "pred": int(preds[i]), "prob": float(probs[i]),
                      "generator": str(generators[i]),
                      "tag": "FP_predfake" if labels[i] == 0 else "FN_predreal"})

    return picks


def render_one_sample(model, transform, sample, device, w_vec, b_scalar
                      ) -> Image.Image:
    """Return a PIL composite image [original | h1 | ... | h6] with annotations."""
    img = Image.open(sample["path"]).convert("RGB")
    img_disp = img.resize((256, 256), Image.BILINEAR)
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)
    concepts = out["concepts"][0].cpu().numpy()        # [K]
    heatmaps = out["concept_maps"][0].cpu().numpy()    # [K, H', W']

    fig, axes = plt.subplots(1, 7, figsize=(22, 3.3))
    axes[0].imshow(img_disp)
    axes[0].set_title(
        f"{sample['generator']}\n"
        f"label={'real' if sample['label']==0 else 'fake'} "
        f"pred={'real' if sample['pred']==0 else 'fake'} "
        f"prob={sample['prob']:.3f}",
        fontsize=8,
    )
    axes[0].axis("off")

    img_np = np.array(img_disp) / 255.0  # H, W, 3 in [0,1]
    for k in range(6):
        hm = heatmaps[k]
        hm_norm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
        # Upsample 7×7 -> 256×256
        hm_pil = Image.fromarray((hm_norm * 255).astype(np.uint8))
        hm_up = np.array(hm_pil.resize((256, 256), Image.BILINEAR)) / 255.0
        # Composite: red overlay weighted by hm_up
        overlay = img_np.copy()
        cmap = plt.get_cmap("jet")
        rgba = cmap(hm_up)[..., :3]    # H, W, 3
        alpha = 0.5
        blended = (1 - alpha) * img_np + alpha * rgba
        axes[k + 1].imshow(np.clip(blended, 0, 1))
        contrib = w_vec[k] * concepts[k]
        sign = "+" if contrib >= 0 else ""
        axes[k + 1].set_title(
            f"{CONCEPT_NAMES[k]}\n"
            f"c={concepts[k]:.3f}  w·c={sign}{contrib:.2f}",
            fontsize=8,
        )
        axes[k + 1].axis("off")

    plt.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump",     required=True)
    ap.add_argument("--ckpt",     required=True)
    ap.add_argument("--out_dir",  required=True)
    ap.add_argument("--max_samples", type=int, default=60,
                    help="Cap on total figures rendered (to bound runtime).")
    ap.add_argument("--image_size", type=int, default=256)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")

    print(f"[STEP 1] loading checkpoint")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = CBNetAGID(n_concepts=6, pretrained=False, signal_channels=512).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    w_vec = model.classifier.weight.detach().cpu().numpy().squeeze()
    b_scalar = float(model.classifier.bias.detach().cpu().item())
    transform = get_eval_transform(args.image_size)

    print(f"[STEP 2] loading dump and selecting samples")
    d = np.load(args.dump, allow_pickle=True)
    samples = pick_samples(d)
    print(f"  selected {len(samples)} candidates (cap={args.max_samples})")
    samples = samples[:args.max_samples]

    print(f"[STEP 3] rendering heatmaps")
    index_rows = []
    for i, s in enumerate(samples):
        fig = render_one_sample(model, transform, s, device, w_vec, b_scalar)
        tag = s["tag"]
        out_name = f"{i:03d}_{s['generator']}_{tag}.png"
        fig.savefig(out_dir / out_name, dpi=120, bbox_inches="tight")
        plt.close(fig)
        index_rows.append({
            "idx": i, "file": out_name,
            "generator": s["generator"], "tag": tag,
            "label": s["label"], "pred": s["pred"], "prob": s["prob"],
            "path": s["path"],
        })
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(samples)}")

    # Write index
    import pandas as pd
    pd.DataFrame(index_rows).to_csv(out_dir / "_index.csv", index=False)

    # Composite "best-of" sheet: pick 9 samples (3 generators × 3 tag types)
    chosen = []
    for g in ["Stable_Diffusion_v1.4", "ADM", "GLIDE"]:
        for tag in ["correct_real", "correct_fake"]:
            row = next((r for r in index_rows if r["generator"] == g and r["tag"] == tag), None)
            if row:
                chosen.append(row)
    if chosen:
        chosen = chosen[:9]
        print(f"[STEP 4] composing best-of sheet ({len(chosen)} samples)")
        figs = []
        from PIL import Image as PILImage
        cols = 1
        rows = len(chosen)
        widths = []
        heights = []
        imgs = [PILImage.open(out_dir / r["file"]) for r in chosen]
        max_w = max(im.width for im in imgs)
        sum_h = sum(im.height for im in imgs)
        composite = PILImage.new("RGB", (max_w, sum_h), color="white")
        y = 0
        for im in imgs:
            composite.paste(im, ((max_w - im.width) // 2, y))
            y += im.height
        composite.save(out_dir / "_bestof_sheet.png")
        print(f"[SAVED] {out_dir / '_bestof_sheet.png'}")

    print(f"\n[DONE] {len(samples)} heatmaps in {out_dir}")


if __name__ == "__main__":
    main()
