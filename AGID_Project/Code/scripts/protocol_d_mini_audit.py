"""Audit Protocol D mini-training checkpoints with frozen concept interventions.

Protocol D is a small-scale design-sensitivity check, not a full retraining
matrix. Each checkpoint is evaluated on the same mini-val sample; the audit then
uses the trained linear head to run closed-form single-channel zero-outs.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np


CONCEPT_NAMES = [
    "bitplane_lsb",
    "freq_radial",
    "color_manifold",
    "hf_noise",
    "jpeg_quant",
    "texture_geometry",
]
COMPRESSION_PAIR = {"jpeg_quant", "freq_radial"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

FIELDNAMES = [
    "variant",
    "checkpoint",
    "generator",
    "n",
    "channel",
    "baseline_acc",
    "baseline_auc",
    "acc",
    "auc",
    "real_acc",
    "fake_acc",
    "delta_pp",
    "drop_pp",
    "protocol",
    "claim_scope",
]


def _round(value):
    if value is None:
        return ""
    try:
        if np.isnan(value):
            return ""
    except TypeError:
        pass
    return round(float(value), 6)


def logit_to_prob(logits):
    logits = np.asarray(logits, dtype=np.float64)
    return np.where(
        logits >= 0,
        1.0 / (1.0 + np.exp(-logits)),
        np.exp(logits) / (1.0 + np.exp(logits)),
    )


def safe_auc(labels, probs):
    labels = np.asarray(labels)
    if len(set(labels.tolist())) < 2:
        return None
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(labels, probs))
    except Exception:
        return None


def metrics_for(probs, labels):
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs)
    preds = (probs > 0.5).astype(int)
    real_mask = labels == 0
    fake_mask = labels == 1
    return {
        "n": int(len(labels)),
        "acc": float((preds == labels).mean()),
        "auc": safe_auc(labels, probs),
        "real_acc": float((preds[real_mask] == 0).mean()) if real_mask.any() else None,
        "fake_acc": float((preds[fake_mask] == 1).mean()) if fake_mask.any() else None,
    }


def closed_form_zero_out(concepts, labels, weights, bias, names, *, variant, generator, checkpoint=""):
    """Return single-channel zero-out rows for a frozen bottleneck dump."""
    concepts = np.asarray(concepts, dtype=np.float32)
    labels = np.asarray(labels).astype(int)
    weights = np.asarray(weights, dtype=np.float32)
    bias = float(bias)

    baseline_probs = logit_to_prob(concepts @ weights + bias)
    baseline = metrics_for(baseline_probs, labels)
    rows = []
    for k, channel in enumerate(names):
        edited = concepts.copy()
        edited[:, k] = 0.0
        vals = metrics_for(logit_to_prob(edited @ weights + bias), labels)
        delta_pp = (vals["acc"] - baseline["acc"]) * 100.0
        rows.append(
            {
                "variant": variant,
                "checkpoint": str(checkpoint),
                "generator": generator,
                "n": vals["n"],
                "channel": channel,
                "baseline_acc": _round(baseline["acc"]),
                "baseline_auc": _round(baseline["auc"]),
                "acc": _round(vals["acc"]),
                "auc": _round(vals["auc"]),
                "real_acc": _round(vals["real_acc"]),
                "fake_acc": _round(vals["fake_acc"]),
                "delta_pp": _round(delta_pp),
                "drop_pp": _round(max(0.0, -delta_pp)),
                "protocol": "frozen_bottleneck_closed_form",
                "claim_scope": "mini_training_design_sensitivity_not_full_retraining",
            }
        )
    return rows


def summarize_reliance(rows):
    """Group rows by variant and summarize compression-axis reliance."""
    grouped = {}
    for row in rows:
        grouped.setdefault(row["variant"], []).append(row)

    summary = {}
    for variant, variant_rows in grouped.items():
        drops = {
            row["channel"]: float(row["drop_pp"])
            for row in variant_rows
            if row.get("channel")
        }
        total_drop = sum(v for v in drops.values() if v > 0)
        compression_drop = sum(drops.get(name, 0.0) for name in COMPRESSION_PAIR)
        top_channel = ""
        top_drop = 0.0
        if drops:
            top_channel, top_drop = max(drops.items(), key=lambda item: item[1])
        baseline_acc = variant_rows[0].get("baseline_acc", "")
        baseline_auc = variant_rows[0].get("baseline_auc", "")
        summary[variant] = {
            "baseline_acc": baseline_acc,
            "baseline_auc": baseline_auc,
            "top_channel": top_channel,
            "top_drop_pp": _round(top_drop),
            "compression_drop_pp": _round(compression_drop),
            "total_positive_drop_pp": _round(total_drop),
            "compression_share": _round(compression_drop / total_drop) if total_drop > 0 else None,
            "channels": {name: _round(drops[name]) for name in sorted(drops)},
        }
    return summary


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_md(path, summary, *, generator, n_per_class):
    lines = [
        "# Protocol D Mini-Training Design Sensitivity",
        "",
        f"- Eval set: `{generator}` val, {n_per_class}/class sampled with a fixed seed",
        "- Claim scope: mini-training design sensitivity, not a full retraining matrix",
        "",
        "| Variant | Baseline acc | Baseline AUC | Top zero-out | Top drop (pp) | Compression drop (pp) | Compression share |",
        "|---|---:|---:|---|---:|---:|---:|",
    ]
    for variant in sorted(summary):
        row = summary[variant]
        lines.append(
            f"| {variant} | {float(row['baseline_acc']) * 100:.2f}% | "
            f"{float(row['baseline_auc']):.6f} | {row['top_channel']} | "
            f"{float(row['top_drop_pp']):.2f} | {float(row['compression_drop_pp']):.2f} | "
            f"{float(row['compression_share']):.3f} |"
        )
    lines.extend(
        [
            "",
            "The rows above are computed by freezing the trained checkpoint, exporting the",
            "six bottleneck activations, and zeroing one linear-head input at a time in",
            "closed form. They are mechanism checks for a small training subset.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def list_images(directory):
    directory = Path(directory)
    if not directory.exists():
        return []
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def sample_eval_set(root, generator, split, n_per_class, seed):
    root = Path(root)
    rng = random.Random(seed)
    ai = list_images(root / generator / split / "ai")
    nature = list_images(root / generator / split / "nature")
    if not ai or not nature:
        raise FileNotFoundError(f"Missing eval images under {root / generator / split}")
    ai_sample = rng.sample(ai, min(n_per_class, len(ai)))
    nature_sample = rng.sample(nature, min(n_per_class, len(nature)))
    return [(p, 1) for p in ai_sample] + [(p, 0) for p in nature_sample]


def export_concepts(checkpoint, samples, *, batch_size, device):
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset

    pkg_root = Path(__file__).resolve().parent.parent
    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))

    from cbnet_agid.data.transforms import get_eval_transform
    from cbnet_agid.models import CBNetAGID

    class EvalDataset(Dataset):
        def __init__(self, items):
            self.items = items
            self.transform = get_eval_transform(256)

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            path, label = self.items[idx]
            try:
                image = Image.open(path).convert("RGB")
            except Exception:
                image = Image.new("RGB", (256, 256))
            return self.transform(image), int(label)

    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = CBNetAGID(n_concepts=6, pretrained=False, signal_channels=512).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dataset = EvalDataset(samples)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    concepts = np.zeros((len(samples), 6), dtype=np.float32)
    labels = np.zeros(len(samples), dtype=np.int64)
    offset = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            out = model(x)
            batch = len(y)
            concepts[offset : offset + batch] = out["concepts"].detach().cpu().numpy()
            labels[offset : offset + batch] = y.numpy()
            offset += batch

    weights = model.classifier.weight.detach().cpu().numpy().reshape(-1)
    bias = float(model.classifier.bias.detach().cpu().numpy().reshape(-1)[0])
    return concepts, labels, weights, bias


def parse_checkpoints(items):
    parsed = []
    for item in items:
        if len(item) != 2:
            raise ValueError("--checkpoint expects VARIANT PATH")
        variant, checkpoint = item
        parsed.append((variant, Path(checkpoint)))
    return parsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="../Data/GenImage")
    parser.add_argument("--generator", default="Stable_Diffusion_v1.4")
    parser.add_argument("--split", default="val")
    parser.add_argument("--n-per-class", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out-dir", default="Results/protocol_d_mini_audit")
    parser.add_argument(
        "--checkpoint",
        nargs=2,
        action="append",
        metavar=("VARIANT", "PATH"),
        required=True,
        help="Variant name and checkpoint path. Repeat for each variant.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    samples = sample_eval_set(
        args.root,
        args.generator,
        args.split,
        args.n_per_class,
        args.seed,
    )

    rows = []
    for variant, checkpoint in parse_checkpoints(args.checkpoint):
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        print(f"[AUDIT] {variant}: {checkpoint} on {device}")
        concepts, labels, weights, bias = export_concepts(
            checkpoint,
            samples,
            batch_size=args.batch_size,
            device=device,
        )
        rows.extend(
            closed_form_zero_out(
                concepts,
                labels,
                weights,
                bias,
                CONCEPT_NAMES,
                variant=variant,
                generator=args.generator,
                checkpoint=checkpoint,
            )
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    summary = summarize_reliance(rows)
    csv_path = out_dir / "protocol_d_mini_design_sensitivity.csv"
    json_path = out_dir / "protocol_d_mini_summary.json"
    md_path = out_dir / "protocol_d_mini_summary.md"
    _write_csv(csv_path, rows)
    json_path.write_text(
        json.dumps(
            {
                "n_rows": len(rows),
                "generator": args.generator,
                "n_per_class": args.n_per_class,
                "claim_scope": "mini_training_design_sensitivity_not_full_retraining",
                "summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_summary_md(md_path, summary, generator=args.generator, n_per_class=args.n_per_class)

    print(f"[DONE] wrote {len(rows)} rows")
    print(f"[DONE] {csv_path}")
    print(f"[DONE] {json_path}")
    print(f"[DONE] {md_path}")


if __name__ == "__main__":
    main()
