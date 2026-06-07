# CBNet-AGID — Concept Bottleneck Network for AI-Generated Image Detection

**Project:** 2026 Spring ML Course Final Project
**Topic:** AI-Generated Image Detection (AGID) with structural interpretability and cross-generator generalization
**Author:** [user] (langqiudmz@gmail.com)
**Status:** Pre-Run (Stage 0.5) — methodology validation in progress

---

## Project Layout

```
AGID_Project/
├── README.md                       <- this file
├── Stage1_Research/                <- Stage 1 deliverables (locked)
│   ├── 01_RQ_Brief.md
│   ├── 02_Methodology_Blueprint.md
│   ├── 03_Bibliography.md (90 refs)
│   └── 04_Synthesis.md
├── Code/                           <- all source code
│   ├── external/                   <- cloned baseline repos
│   │   ├── LOTA/                   <- https://github.com/hongsong-wang/LOTA
│   │   └── NPR-DeepfakeDetection/  <- https://github.com/chuangchuangtan/NPR-DeepfakeDetection
│   ├── cbnet_agid/                 <- our method implementation
│   │   ├── models/                 <- model architectures (TBD)
│   │   ├── data/                   <- data loaders (TBD)
│   │   ├── concepts/               <- concept signal extractors (TBD)
│   │   ├── losses/                 <- loss functions (TBD)
│   │   ├── train.py                <- training entry point (TBD)
│   │   └── evaluate.py             <- evaluation entry point (TBD)
│   └── scripts/                    <- one-off scripts (download, preprocess, etc.)
├── Data/                           <- datasets (GenImage subset, ForenSynths, etc.)
├── Checkpoints/                    <- model weights (ours + downloaded baselines)
├── Logs/                           <- training logs, tensorboard
└── Results/                        <- experiment outputs (tables, plots)
```

## Environment

- **OS:** Windows 11 (PowerShell + Bash via Claude Code tooling)
- **GPU:** NVIDIA RTX 4060 Laptop GPU, 8GB VRAM
- **Conda env:** `agid` at `E:\LQiu\conda_envs\agid` (Python 3.11)
- **PyTorch:** 2.5+ with CUDA 12.1

## Hardware-driven defaults (must follow)

Due to 8GB VRAM constraint:
- `batch_size=16` (not 64 as in original LOTA spec)
- `grad_accum_steps=4` (effective batch = 64)
- `mixed_precision=bfloat16` (AMP)
- `image_size=256` (LOTA-style)

## Pre-Run (Stage 0.5) Goals

| Day | Task | Go/No-Go Criterion |
|---|---|---|
| 0-1 | Env setup, clone repos | PyTorch sees GPU; repos cloned |
| 2-3 | Reproduce LOTA + NPR inference | Accuracy ≥ 95% on their sample data |
| 4 | Get GenImage SD-1.4 subset (~30-50GB) | Data loader works |
| 5 | Sanity check 3 concepts | ≥ 2 concepts show t-test p < 0.01 between real/AI |
| 6-7 | Scaffold CBNet-AGID + 1-epoch training | Loss decreases; time-per-epoch measured |

## Method Summary (from Stage 1.2)

**CBNet-AGID** = dual-stream backbone (ResNet50 pixel stream + bit-plane/NPR signal stream) → Concept Bottleneck Layer (6 weakly-supervised concepts: HF-Noise, BitPlane-LSB, Freq-Subband, EdgeSharpness, Color-Manifold, JPEG-QuantTrace) → linear classifier.

Loss = task BCE + concept supervision (MSE on heuristic labels) + cross-generator concept consistency + sparsity entropy regularizer.

**Differentiation from AIGI-Holmes (ICCV 2025):**
- We: structural / ante-hoc interpretability, single GPU
- They: post-hoc text rationalization via LLaVA, cluster scale

## References

90 references in `Stage1_Research/03_Bibliography.md`, covering 7 AGID method families + Concept Bottleneck literature + generative model background + benchmarks.

## License

To be determined before public release. Default: research-use-only until paper acceptance.
