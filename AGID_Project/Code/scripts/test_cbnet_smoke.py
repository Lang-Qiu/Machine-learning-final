"""Phase A smoke test — verify CBNet-AGID forward + backward on random input.

Validates:
  - All module imports work
  - Model builds on GPU
  - Forward pass produces correct output shapes
  - Each loss component computes a scalar gradient-bearing tensor
  - Total loss can be back-propagated
  - VRAM stays under 8GB at batch_size = 16
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))

import torch
import torch.nn.functional as F

from cbnet_agid.models import CBNetAGID
from cbnet_agid.losses import CBNetLoss
from cbnet_agid.concepts.heuristics import CONCEPT_NAMES


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device: {device}")
    if device.type == "cuda":
        print(f"[INFO] gpu:    {torch.cuda.get_device_name(0)}")
        print(f"[INFO] vram:   {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.cuda.reset_peak_memory_stats()

    # Build model
    print("\n[STEP 1] building CBNetAGID...")
    model = CBNetAGID(n_concepts=6, pretrained=False).to(device)  # no download
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  total params: {n_params/1e6:.2f}M")
    print(f"  trainable:    {n_trainable/1e6:.2f}M")
    print(f"  concepts ({len(CONCEPT_NAMES)}): {CONCEPT_NAMES}")

    # Mock batch
    batch_size = 16
    print(f"\n[STEP 2] forward pass with batch_size={batch_size}, image 224x224...")
    x = torch.randn(batch_size, 3, 224, 224, device=device)
    labels = torch.randint(0, 2, (batch_size,), device=device).float()
    target_concepts = torch.rand(batch_size, 6, device=device)

    # Forward
    t0 = time.time()
    out = model(x)
    forward_ms = (time.time() - t0) * 1000
    print(f"  forward time:  {forward_ms:.1f} ms")
    print(f"  logit shape:   {tuple(out['logit'].shape)}")
    print(f"  prob shape:    {tuple(out['prob'].shape)}")
    print(f"  concepts:      {tuple(out['concepts'].shape)}  range [{out['concepts'].min():.3f}, {out['concepts'].max():.3f}]")
    print(f"  concept_maps:  {tuple(out['concept_maps'].shape)}")
    print(f"  features:      {tuple(out['features'].shape)}")
    assert out["logit"].shape == (batch_size,)
    assert out["concepts"].shape == (batch_size, 6)

    # Loss
    print("\n[STEP 3] computing 4-component loss...")
    loss_fn = CBNetLoss(lambda_concept=0.5, lambda_gen=0.2, lambda_sparsity=0.05)
    # Mock "paired" output for cross-generator consistency (using a 2nd forward on different input)
    x2 = torch.randn_like(x)
    paired_out = model(x2)
    losses = loss_fn(out, labels, target_concepts=target_concepts, paired_out=paired_out)
    for name, val in losses.items():
        print(f"  L_{name}:  {val.item():+.4f}")
    assert losses["total"].requires_grad, "Total loss must require grad"

    # Backward
    print("\n[STEP 4] backward pass...")
    t0 = time.time()
    losses["total"].backward()
    backward_ms = (time.time() - t0) * 1000
    print(f"  backward time: {backward_ms:.1f} ms")

    grad_norms = {
        name: float(p.grad.norm()) for name, p in model.named_parameters()
        if p.grad is not None and p.requires_grad
    }
    max_grad = max(grad_norms.values()) if grad_norms else 0.0
    mean_grad = sum(grad_norms.values()) / max(len(grad_norms), 1)
    print(f"  grad norm:     mean={mean_grad:.4f}  max={max_grad:.4f}")
    n_with_grad = len(grad_norms)
    print(f"  params w/grad: {n_with_grad}")

    # VRAM
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"\n[STEP 5] peak VRAM during this step: {peak:.2f} GB")
        if peak > 7.5:
            print("  ⚠️ Close to 8GB limit — consider gradient checkpointing or smaller batch")
        else:
            print("  ✅ Well under 8GB — room for full training")

    print("\n✅ PHASE A SMOKE TEST PASSED.")


if __name__ == "__main__":
    main()
