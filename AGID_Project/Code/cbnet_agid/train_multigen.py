"""Route B training entry point: 4-generator interleaved training with content-pairing loss.

Uses InterleavedMultiGenDataset (8+8+8+8 per batch) and CBNetMultiGenLoss, which adds
a cross-generator content-pairing consistency term on top of the standard Route A losses.

Default config is calibrated for RTX 4060 Laptop GPU (8GB VRAM):
  - batch_size=32  (8 per generator)
  - image_size=256
  - bfloat16 mixed precision
  - gradient accumulation steps=2 (effective batch = 64)

Usage:
    python -m cbnet_agid.train_multigen \\
        --root E:/LQiu/lab_folder/Machine_learning/AGID_Project/Data/GenImage \\
        --generators Stable_Diffusion_v1.4 BigGAN ADM Midjourney \\
        --split train_25k \\
        --epochs 20
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from .data.genimage import GENIMAGE_GENERATORS
from .data.multigen import build_multigen_loaders
from .data.transforms import get_eval_transform, get_train_transform
from .losses.losses import CBNetMultiGenLoss
from .models import CBNetAGID


def evaluate_loader(model: torch.nn.Module, loader, device, name: str) -> dict:
    """Compute accuracy + AUC on a single eval loader."""
    from sklearn.metrics import accuracy_score, roc_auc_score
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            out = model(x)
            all_probs.append(out["prob"].cpu().numpy())
            all_labels.append(y.cpu().numpy())
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds = (probs > 0.5).astype(int)
    acc = float(accuracy_score(labels, preds))
    try:
        auc = float(roc_auc_score(labels, probs))
    except ValueError:
        auc = float("nan")
    real_mask = labels == 0
    fake_mask = labels == 1
    real_acc = float((preds[real_mask] == 0).mean()) if real_mask.any() else float("nan")
    fake_acc = float((preds[fake_mask] == 1).mean()) if fake_mask.any() else float("nan")
    return {
        "name": name, "n": int(len(labels)), "acc": acc, "auc": auc,
        "real_acc": real_acc, "fake_acc": fake_acc,
    }


def main():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("--root", required=True, help="GenImage root directory")
    parser.add_argument("--generators", nargs="+",
                        default=["Stable_Diffusion_v1.4", "BigGAN", "ADM", "Midjourney"],
                        help="Generators to include in interleaved training.")
    parser.add_argument("--split", default="train_25k",
                        help="Training split name (e.g. 'train_25k', 'train').")
    parser.add_argument("--use_shared_concept_labels", action="store_true", default=True,
                        help="Load concept_labels_shared.npy instead of concept_labels.npy.")
    parser.add_argument("--val_generators", default="all",
                        help="Comma-separated list or 'all' for OOD val generators.")
    parser.add_argument("--max_train_samples", type=int, default=None,
                        help="Per-class subsample per generator (for smoke tests). "
                             "Concept labels are correctly re-indexed when set.")
    parser.add_argument("--max_val_samples", type=int, default=None)
    # Model
    parser.add_argument("--n_concepts", type=int, default=6)
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--signal_channels", type=int, default=512)
    # Optimization
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Must be divisible by len(generators). Default 32 = 8×4.")
    parser.add_argument("--accum_steps", type=int, default=2)
    parser.add_argument("--lr_backbone", type=float, default=1e-4)
    parser.add_argument("--lr_head", type=float, default=5e-4)
    parser.add_argument("--lr_classifier", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lambda_concept", type=float, default=0.5)
    parser.add_argument("--lambda_pair", type=float, default=0.2,
                        help="Content-pairing consistency loss weight.")
    parser.add_argument("--lambda_sparsity", type=float, default=0.05)
    parser.add_argument("--amp", action="store_true", default=True)
    # Image
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--disable_destructive_aug", action="store_true", default=True,
                        help="Disable JPEG+blur augmentation (default True for Route B concept supervision).")
    # IO
    parser.add_argument("--out_dir", default="Logs/cbnet_multigen_run",
                        help="Output directory for checkpoints + logs.")
    parser.add_argument("--ckpt_every", type=int, default=1)
    parser.add_argument("--eval_every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as fh:
        json.dump(vars(args), fh, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")
    if device.type == "cuda":
        print(f"[INFO] gpu: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")

    val_generators = (GENIMAGE_GENERATORS if args.val_generators == "all"
                      else args.val_generators.split(","))

    # Build loaders
    print("\n[STEP 1] building data loaders...")
    train_tf = get_train_transform(args.image_size,
                                   disable_destructive=args.disable_destructive_aug)
    eval_tf = get_eval_transform(args.image_size)
    loaders = build_multigen_loaders(
        root=args.root,
        generators=args.generators,
        split=args.split,
        use_shared_concept_labels=args.use_shared_concept_labels,
        train_transform=train_tf,
        eval_transform=eval_tf,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_generators=val_generators,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
    )
    train_ds = loaders["train"].dataset
    print(f"\n  train: {len(train_ds)} samples across {train_ds.n_gens} generators")

    # Model
    print("\n[STEP 2] building model...")
    model = CBNetAGID(
        n_concepts=args.n_concepts, pretrained=args.pretrained,
        signal_channels=args.signal_channels,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  params: {n_params:.2f}M")

    backbone_params = list(model.backbone.parameters())
    head_params = list(model.cbl.parameters())
    cls_params = list(model.classifier.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": head_params,     "lr": args.lr_head},
        {"params": cls_params,      "lr": args.lr_classifier},
    ], weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = CBNetMultiGenLoss(
        lambda_concept=args.lambda_concept,
        lambda_pair=args.lambda_pair,
        lambda_sparsity=args.lambda_sparsity,
    )
    scaler = GradScaler(enabled=args.amp)

    # Training loop
    print("\n[STEP 3] training...")
    history = []
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        running = {k: 0.0 for k in ["total", "task", "concept", "content_pair", "sparsity"]}
        n_seen = 0
        pbar = tqdm(loaders["train"], desc=f"epoch {epoch+1}/{args.epochs}")
        optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            gen_ids = batch["generator_id"].to(device, non_blocking=True)
            cls_ids = batch["class_idx"].to(device, non_blocking=True)

            target_concepts = batch.get("concept_labels")
            if target_concepts is not None:
                target_concepts = target_concepts.to(device, non_blocking=True)

            with autocast(enabled=args.amp, dtype=torch.bfloat16):
                out = model(x)
                losses = loss_fn(
                    out, y.float(), gen_ids, cls_ids,
                    target_concepts=target_concepts,
                )
                loss = losses["total"] / args.accum_steps

            scaler.scale(loss).backward()
            if (step + 1) % args.accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            bs = x.size(0)
            n_seen += bs
            for k in running:
                running[k] += float(losses[k]) * bs

            if step % 10 == 0:
                pbar.set_postfix({
                    "loss":   f"{losses['total'].item():.3f}",
                    "task":   f"{losses['task'].item():.3f}",
                    "pair":   f"{losses['content_pair'].item():.4f}",
                    "sparse": f"{losses['sparsity'].item():.3f}",
                })

        avg = {k: v / max(n_seen, 1) for k, v in running.items()}
        epoch_time = time.time() - t0
        log = {"epoch": epoch + 1, "time_sec": epoch_time,
               **avg, "lr": optimizer.param_groups[0]["lr"]}
        scheduler.step()

        # Eval
        if (epoch + 1) % args.eval_every == 0:
            print("\n[EVAL]")
            results = []
            for k, dl in loaders.items():
                if not k.startswith("val_"):
                    continue
                res = evaluate_loader(model, dl, device, k.replace("val_", ""))
                results.append(res)
                print(f"  {res['name']:<28s} acc={res['acc']*100:5.2f}  auc={res['auc']:.3f}  "
                      f"real={res['real_acc']*100:5.2f}  fake={res['fake_acc']*100:5.2f}")
            if results:
                mean_acc = float(np.mean([r["acc"] for r in results]))
                mean_auc = float(np.mean([r["auc"] for r in results if not np.isnan(r["auc"])]))
                log["mean_acc"] = mean_acc
                log["mean_auc"] = mean_auc
                log["eval"] = results
                print(f"  MEAN ACROSS GENERATORS:    acc={mean_acc*100:5.2f}  auc={mean_auc:.3f}")

        history.append(log)
        with open(out_dir / "history.json", "w") as fh:
            json.dump(history, fh, indent=2)
        print(f"\n[EPOCH {epoch+1}] time={epoch_time:.1f}s "
              f"total={avg['total']:.4f} task={avg['task']:.4f} "
              f"pair={avg['content_pair']:.4f} sparse={avg['sparsity']:.4f}")

        if (epoch + 1) % args.ckpt_every == 0:
            ckpt_path = out_dir / f"ckpt_epoch{epoch+1}.pth"
            torch.save({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
            }, ckpt_path)
            print(f"  [CKPT] saved {ckpt_path}")

    # Save final results summary
    final_results = {g: None for g in val_generators}
    if history and "eval" in history[-1]:
        for r in history[-1]["eval"]:
            final_results[r["name"]] = r
    results_path = Path("Results") / f"cbnet_multigen_{Path(args.out_dir).name}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as fh:
        json.dump({"config": vars(args), "final_eval": final_results, "history": history}, fh, indent=2)
    print(f"\n[SAVED] results → {results_path}")
    print("\n✅ TRAINING DONE")


if __name__ == "__main__":
    main()
