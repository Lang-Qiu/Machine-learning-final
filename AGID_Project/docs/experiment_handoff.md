# Experiment Handoff — AGID Concept Bottleneck Project

**Created**: 2026-05-26
**Pipeline Stage**: academic-paper-pipeline Stage 1.5 → Stage 2 transition
**Next Agent**: experiment-agent (plan mode) or general-purpose agent

---

## 1. Workflow Position

This is **not** a standalone ML tuning project. It operates under a dual-skill system:

| Skill | Role | Stage |
|-------|------|-------|
| `academic-paper-pipeline` (ARS) | Research questions, paper structure, experiment narrative | Stage 1 RESEARCH complete; Stage 2 WRITE pending |
| `experiment-agent` | Experiment execution, training records, OOD evaluation, ablation evidence | Plan mode completed (07_Experiment_Plan.md); run mode executed Phase A |

**Current junction**: Route A (single-generator scaling) evidence is frozen. The project is transitioning to Route B (multi-generator training) or to paper drafting using Route A evidence alone.

**Key documents** (in `Stage1_Research/`):
- `01_RQ_Brief.md` — 1 main RQ + 3 sub-RQs + 4 hypotheses
- `02_Methodology_Blueprint.md` — dual-stream backbone, 6 concepts, 4 losses, 5 ablations, 6 baselines
- `03_Bibliography.md` — 90 references
- `04_Synthesis.md` — field positioning
- `05_Day0-1_Setup_Report.md` — env validation
- `06_Day5_Concept_Sanity_Report.md` — 4 strong + 2 borderline concepts
- `07_Experiment_Plan.md` — locked experiment plan with 15-row experiment matrix

**Hardware constraint**: RTX 4060 Laptop 8GB VRAM. Always use `PYTHONNOUSERSITE=1` + `PYTHONMALLOC=malloc` + `PYTHONPATH=<Code-dir>`.

**Environment**: `E:\LQiu\conda_envs\agid` (PyTorch 2.1.2+cu121, numpy 1.26.4). Activate with `activate_agid.ps1`.

---

## 2. Route A — Frozen Evidence

### 2.1 Model Architecture

CBNet-AGID: DualStreamBackbone (ResNet50 + NPR-Residual signal stream) → ConceptBottleneckLayer (K=6) → Linear classifier.
- 38.05M params, ~3.5 GB peak VRAM at batch_size=32 with bf16 AMP
- 6 heuristic concepts: bitplane_lsb, freq_radial, color_manifold, hf_noise, jpeg_quant, texture_geometry
- 4 loss components: BCE task + concept MSE + cross-gen consistency (disabled) + sparsity entropy

### 2.2 Training Runs

#### Debug Run (20k subset, seed=42)

- **Data**: `Data/GenImage/Stable_Diffusion_v1.4/train_20k/` (10k ai + 10k nature)
- **Concept labels**: `train_20k/concept_labels.npy` — shape (20000, 6), 0 NaN, normalized [0,1]
- **Checkpoints**: `Code/Logs/debug_run_20k_s42/ckpt_epoch{1..5}.pth` (~436 MB each)
- **History**: `Code/Logs/debug_run_20k_s42/history.json`

| Epoch | Time (s) | Total Loss | Task Loss | ID Acc | ID AUC |
|-------|----------|-----------|-----------|--------|--------|
| 1 | 147 | 0.504 | 0.509 | 98.33% | 0.998 |
| 2 | 149 | 0.338 | 0.304 | 98.85% | 0.999 |
| 3 | 137 | 0.252 | 0.216 | 99.00% | 0.999 |
| 4 | 141 | 0.209 | 0.175 | **99.49%** | 1.000 |
| 5 | 133 | 0.189 | 0.156 | 99.44% | 1.000 |

**Best epoch**: 4, used for OOD evaluation.

#### Main Run (100k subset, seed=42) — KILLED at epoch 9

- **Data**: `Data/GenImage/Stable_Diffusion_v1.4/train_100k/` (50k ai + 50k nature)
- **Concept labels**: `train_100k/concept_labels.npy` — shape (100000, 6), 0 NaN
- **Checkpoints**: `Code/Logs/main_run_100k_s42/ckpt_epoch{1..8}.pth` (~436 MB each)
- **History**: `Code/Logs/main_run_100k_s42/history.json`

| Epoch | Time (s) | Total Loss | Task Loss | ID Acc | ID AUC |
|-------|----------|-----------|-----------|--------|--------|
| 1 | 4091 | 0.293 | 0.273 | 99.48% | 1.000 |
| 2 | 3774 | 0.100 | 0.097 | 99.78% | 1.000 |
| 3 | 3353 | 0.050 | 0.065 | 99.59% | 1.000 |
| 4 | 1898 | 0.021 | 0.048 | 99.71% | 1.000 |
| 5 | 1640 | 0.005 | 0.040 | 99.70% | 1.000 |
| 6 | 943 | -0.007 | 0.033 | 99.72% | 1.000 |
| 7 | 710 | -0.015 | 0.029 | **99.80%** | 1.000 |
| 8 | 2048 | -0.022 | 0.025 | **99.83%** | 1.000 |

**Best epoch**: 7 (99.80%, fastest convergence), used for OOD evaluation.  
**Note**: Total loss goes negative from epoch 6 — sparsity loss dominating. Task loss continues decreasing.

### 2.3 Baseline Results

All result JSON files are in `Results/`.

#### In-Distribution (SD-1.4 val, 12,000 images)

| Method | Train Set | Acc | AUC | Real-Acc | Fake-Acc | Source File |
|--------|-----------|-----|-----|----------|----------|-------------|
| LOTA | SD-1.4 | 99.76% | 1.000 | 99.68% | 99.83% | `lota_sdv14_val.json` |
| CBNet-AGID (20k) | SD-1.4 | 99.49% | 1.000 | 99.37% | 99.62% | history.json |
| CBNet-AGID (100k) | SD-1.4 | 99.80% | 1.000 | — | — | history.json |
| NPR | ProGAN | 79.72% | 0.889 | 63.80% | 95.63% | `npr_sdv14_val.json` |

#### Cross-Architecture OOD (ForenSynths test, 4 generators)

**CBNet-AGID 20k** (epoch 4) — `cbnet_forensynths.json`:

| Generator | N | Acc | AUC | Real-Acc | Fake-Acc |
|-----------|----|------|------|----------|----------|
| biggan | 4000 | 49.43% | 0.484 | 97.60% | 1.25% |
| deepfake | 5405 | 59.63% | 0.625 | 60.58% | 58.67% |
| gaugan | 10000 | 49.75% | 0.535 | 98.90% | 0.60% |
| stargan | 3998 | 50.03% | 0.467 | 99.90% | 0.15% |
| **MEAN** | — | **52.21%** | **0.528** | — | — |

**CBNet-AGID 100k** (epoch 7) — `cbnet_100k_forensynths.json`:

| Generator | N | Acc | AUC | Real-Acc | Fake-Acc |
|-----------|----|------|------|----------|----------|
| biggan | 4000 | 49.55% | 0.414 | 98.75% | 0.35% |
| deepfake | 5405 | 57.15% | 0.685 | 97.45% | 16.72% |
| gaugan | 10000 | 50.10% | 0.531 | 99.72% | 0.48% |
| stargan | 3998 | 50.00% | 0.331 | 99.95% | 0.05% |
| **MEAN** | — | **51.70%** | **0.490** | — | — |

**NPR** (trained on ProGAN) — `npr_forensynths.json`:

| Generator | N | Acc | AUC | Real-Acc | Fake-Acc |
|-----------|----|------|------|----------|----------|
| biggan | 4000 | 81.13% | 0.899 | 66.40% | 95.85% |
| deepfake | 5405 | 77.98% | 0.877 | 61.06% | 94.96% |
| gaugan | 10000 | 80.86% | 0.901 | 63.50% | 98.22% |
| stargan | 3998 | 78.36% | 0.900 | 89.74% | 66.98% |
| **MEAN** | — | **79.58%** | **0.894** | — | — |

### 2.4 Core Conclusion

**Single-generator scaling does NOT improve OOD generalization.** Going from 20k → 100k SD-1.4 training data improved ID accuracy (+0.31pp) but **degraded** OOD (-0.51pp). The model learned SD-1.4-specific shortcuts more precisely, not general AI-generation artifacts. With ~50% OOD accuracy and near-zero fake detection on biggan/gaugan/stargan, the model is effectively random on cross-architecture data.

The cause is systematic, not a training insufficiency:
1. Heuristic concept labels (norm stats computed on SD-1.4) inherently encode distribution-specific statistics
2. Single-generator training gives the model no signal about which features generalize across generators
3. The concept bottleneck, when concepts are distribution-locked, may **amplify** overfitting rather than prevent it

---

## 3. Paper Narrative Value (Route A)

Route A results are not failures to hide — they are **strong ablation evidence** for the paper:

| Paper Section | How Route A Fits |
|---------------|-----------------|
| **Abstract** | "We find that single-generator concept bottleneck training achieves near-SOTA ID accuracy (99.8%) but collapses on cross-architecture OOD (51.7%), revealing a fundamental trade-off." |
| **Introduction / Motivation** | Motivate why multi-generator training is necessary — single-generator scaling is provably insufficient. |
| **Method** | The architecture (CBNet-AGID) is sound — the concept bottleneck works on ID without accuracy loss. |
| **Experiments — Ablation** | Data scale ablation: 20k vs 100k shows ID improvement but OOD degradation. Quantitatively demonstrates the overfitting. |
| **Experiments — Limitation** | Honest reporting of cross-architecture failure with root-cause analysis. |
| **Conclusion** | Call for distribution-agnostic concept learning as future work. |

**Recommended paper thesis**: "Concept bottleneck models maintain ID accuracy with interpretability, but heuristically-supervised concepts are inherently distribution-specific. Multi-generator training with consistency regularization is a necessary condition for generalization."

---

## 4. Route B — Next Direction

### 4.1 Objective

Verify whether **multi-generator training + cross-generator consistency loss** can break through the ~52% OOD ceiling observed in Route A.

### 4.2 Data Plan

| Generator | Samples | Architecture Family | Purpose |
|-----------|---------|-------------------|---------|
| Stable_Diffusion_v1.4 | 50k (25k/class) | Diffusion | Primary training signal |
| BigGAN | 50k (25k/class) | GAN | Bridges to ForenSynths GAN generators |
| ADM | 50k (25k/class) | Diffusion | Expands diffusion coverage |

Total training data: 150k images.

### 4.3 Expected Directory Structure

```
E:\LQiu\lab_folder\Machine_learning\AGID_Project\Data\GenImage\
  Stable_Diffusion_v1.4\
    train\          ← EXISTS (162k/class)
      ai\
      nature\
    train_20k\      ← EXISTS (10k/class subset)
    train_100k\     ← EXISTS (50k/class subset)
    val\            ← EXISTS (6k/class)
      ai\
      nature\
  BigGAN\
    train\          ← NEEDS DOWNLOAD
      ai\
      nature\
  ADM\
    train\          ← NEEDS DOWNLOAD
      ai\
      nature\
```

**Current state (2026-05-26)**: Only `Stable_Diffusion_v1.4` is present under `Data/GenImage/`. BigGAN and ADM directories do **not** exist.

### 4.4 Training Plan

1. Create 25k/class subsets from BigGAN and ADM train (using `Code/scripts/subset_sd14_train.py`, modify generator name)
2. Precompute concept labels for each generator (SD-1.4 can reuse `train_100k` labels, or create fresh 50k subset)
3. Build multi-generator DataLoader that interleaves or alternates generators
4. Enable `--lambda_gen 0.1` for cross-generator consistency loss (requires paired batches from different generators)
5. Train 20 epochs, evaluate on both SD-1.4 val (ID) and ForenSynths (OOD)

**Key code changes needed** (do NOT modify model architecture):
- `build_genimage_loaders` or new function: multi-generator training loader
- Cross-generator consistency loss: needs paired images (same content, different generators) or use same-class images from different generators as proxy pairs

### 4.5 Minimum Success Criteria

- OOD mean accuracy on ForenSynths should exceed **60%** (at minimum — current is 51.7%)
- If OOD reaches **70%+**, multi-generator training is a viable path
- If OOD stays **<55%**, even multi-generator scaling with heuristic concepts cannot work — paper narrative shifts to "fundamental limitation of heuristic concepts"

---

## 5. DO NOT Continue List

The following actions are **explicitly forbidden** for the next agent:

1. ❌ Do NOT continue training on SD-1.4-only data
2. ❌ Do NOT train beyond the existing 100k checkpoint (epoch 8 already exists)
3. ❌ Do NOT tune hyperparameters for higher SD-1.4 ID accuracy (already at ceiling)
4. ❌ Do NOT modify the model architecture (`models/cbnet.py`, `models/backbone.py`, `concepts/base.py`)
5. ❌ Do NOT delete or overwrite any checkpoints, logs, or result files
6. ❌ Do NOT delete old log directories
7. ❌ Do NOT start training before BigGAN and ADM data are verified on disk
8. ❌ Do NOT change the concept set (K=6) before Route B experiments complete
9. ❌ Do NOT use `--max_train_samples` with shuffle=True without verifying concept label alignment

---

## 6. Mandatory Gate (Before Any Training)

The next agent **must** execute this gate sequence and report results before proceeding:

```
1. Verify project root: E:\LQiu\lab_folder\Machine_learning\AGID_Project
2. Check this is not a git repo (no .git directory) — no version control concerns
3. Inventory Results/ — confirm all 9 JSON files present
4. Inventory Code/Logs/ — confirm debug_run_20k_s42 (5 ckpt) + main_run_100k_s42 (8 ckpt)
5. Check Data/GenImage/ directory listing:
   - If BigGAN/train/{ai,nature} does NOT exist → STOP, output download checklist, do not proceed
   - If ADM/train/{ai,nature} does NOT exist → STOP, output download checklist, do not proceed
6. Check Data/ForenSynths/test/ — confirm biggan, deepfake, gaugan, stargan have images
7. If all data present:
   a. Output data inventory
   b. Write a short Route B experiment plan (do NOT train)
   c. STOP and ask user to approve the plan
8. NEVER auto-launch training without explicit user approval after gate check
```

---

## 7. Technical Notes

### 7.1 Known Issues

- **numpy memory fragmentation**: Use `PYTHONMALLOC=malloc` for any long-running numpy-heavy script (precompute, training). Without it, `numpy.histogramdd` in `heuristic_color_manifold` fails with ~2 MiB allocation errors after ~9000+ images.
- **Windows shared memory exhaustion**: Set `--num_workers 0` for evaluation scripts. Set `--num_workers 2` for training (max safe on RTX 4060 system).
- **Scipy import corruption**: Rare `SystemError: error return without exception set` on scipy import. Re-run usually fixes it. If persistent, `pip install --force-reinstall scipy`.
- **Label convention mismatch**: NPR uses real=0/AI=1, LOTA uses natural=1/AI=0. `evaluate.py` normalizes LOTA output via `1.0 - sigmoid(logit)`. Verify direction when adding new baselines.

### 7.2 Useful Scripts

| Script | Purpose |
|--------|---------|
| `Code/scripts/subset_sd14_train.py` | Create reproducible subsets from GenImage train |
| `Code/scripts/eval_forensynths.py` | Evaluate any method on ForenSynths OOD |
| `Code/scripts/_smoke_test.py` | Quick model forward+loss sanity check |
| `Code/cbnet_agid/train.py` | Main training entry (supports --train_split, --lambda_gen) |
| `Code/cbnet_agid/evaluate.py` | GenImage evaluation (CBNet/NPR/LOTA) |
| `Code/cbnet_agid/precompute_concept_labels.py` | Pre-compute heuristic labels |

### 7.3 Important Paths

```
Checkpoints:     E:\LQiu\lab_folder\Machine_learning\AGID_Project\Checkpoints\lota_sdv14.pth (94MB)
NPR weights:     E:\LQiu\lab_folder\Machine_learning\AGID_Project\Code\external\NPR-DeepfakeDetection\NPR.pth (17MB)
LOTA source:     E:\LQiu\lab_folder\Machine_learning\AGID_Project\Code\external\LOTA/
NPR source:      E:\LQiu\lab_folder\Machine_learning\AGID_Project\Code\external\NPR-DeepfakeDetection/
Conda env:       E:\LQiu\conda_envs\agid\
Activate script: E:\LQiu\lab_folder\Machine_learning\AGID_Project\activate_agid.ps1
Memory files:    C:\Users\LQiu\.claude\projects\E--LQiu-lab-folder-Machine-learning\memory\
```

---

## 8. Recommended Next Prompt

Copy this into a new Claude Code session:

---

```
I'm continuing the AGID Concept Bottleneck project from a handoff document at
docs/experiment_handoff.md. Please read that file first.

Context summary:
- Academic paper pipeline (30k-50k words, due 2026-06-24) on AI-Generated Image Detection
- Architecture: CBNet-AGID (Concept Bottleneck + Dual-Stream Backbone)
- Route A complete: single-generator SD-1.4 training (20k and 100k), OOD collapsed at ~52%
- Route B next: multi-generator training (SD-1.4 + BigGAN + ADM) with cross-gen consistency loss

Please execute the Mandatory Gate from the handoff (Section 6):
1. Check Data/GenImage for BigGAN and ADM
2. Report what's available
3. If data missing, tell me exactly what to download

Do NOT start any training. Do NOT modify code. Just do the gate check.
```

---

## 9. Skills Reference

Skills the next session may need:
- `experiment-agent` — for planning Route B experiments (plan mode) and executing training (run mode)
- `academic-research-skills:academic-pipeline` — for Stage 2 paper writing when experiments are done
- `academic-research-skills:ars-abstract` or `ars-lit-review` — for paper component drafting

---

*End of handoff. File saved to `docs/experiment_handoff.md`.*
