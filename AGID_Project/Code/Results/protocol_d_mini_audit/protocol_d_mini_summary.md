# Protocol D Mini-Training Design Sensitivity

- Eval set: `Stable_Diffusion_v1.4` val, 500/class sampled with a fixed seed
- Claim scope: mini-training design sensitivity, not a full retraining matrix

| Variant | Baseline acc | Baseline AUC | Top zero-out | Top drop (pp) | Compression drop (pp) | Compression share |
|---|---:|---:|---|---:|---:|---:|
| full | 98.70% | 0.999356 | jpeg_quant | 2.20 | 2.40 | 0.706 |
| no_pair | 98.50% | 0.999264 | color_manifold | 0.50 | 0.40 | 0.250 |
| no_sparsity | 98.20% | 0.999348 | jpeg_quant | 0.20 | 0.20 | 1.000 |

The rows above are computed by freezing the trained checkpoint, exporting the
six bottleneck activations, and zeroing one linear-head input at a time in
closed form. They are mechanism checks for a small training subset.
