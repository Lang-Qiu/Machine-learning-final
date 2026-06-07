# Methodology Blueprint — Concept Bottleneck Framework for AGID

**Project:** 2026 Spring ML Course Final — AGID Paper
**Version:** v1.0 (Stage 1.2 Full deliverable)
**Date:** 2026-05-24

---

## 1. Paradigm and Method Family

**Paradigm:** Supervised binary classification with interpretable bottleneck. *Family:* Concept Bottleneck Models (CBM, Koh et al. ICML 2020) adapted to AGID. *Inspiration:* E-BotCL (April 2025) — dual-path contrastive concept learning on CUB200/ImageNet; we adapt this to AGID.

**Working name for the method:** **CBNet-AGID** (Concept-Bottleneck Network for AGID). Final name TBD during writing.

---

## 2. Notation

Let $I \in \mathbb{R}^{3 \times H \times W}$ be an RGB input image, $y \in \{0, 1\}$ the binary label (0 = real, 1 = AI-generated), and $g \in \mathcal{G}$ the generator that produced $I$ (real images have $g = \emptyset$).

We define $K$ concepts $\{C_k\}_{k=1}^{K}$, with concept activation vector $\mathbf{c}(I) = [c_1(I), \dots, c_K(I)] \in [0,1]^K$ and per-concept spatial heatmap $\mathbf{h}_k(I) \in [0,1]^{H' \times W'}$.

---

## 3. Architecture (Three-Stage Pipeline)

### Stage A — Backbone feature extractor $\boldsymbol{f_\theta}$

We propose using a **hybrid backbone** that fuses two complementary signal pathways:

1. **Pixel-domain ResNet-50 stream** (operates on raw image).
2. **Bit-plane + neighborhood-pixel-relationship stream** (operates on LOTA-style bit-plane composite + NPR-style 1st-order pixel difference map).

The two streams are fused via concatenation at the final feature map, yielding $f_\theta(I) \in \mathbb{R}^{2048 \times H' \times W'}$ where $H' = W' = 7$ for a 224×224 input (typical ResNet stride).

*Rationale.* A single-stream backbone makes the "novelty boundary" of the bottleneck weaker (it would look like A+B = NPR+CBM). The dual-stream feature input is itself a (small) novelty because **no prior CBM work has used dual-stream low-level + high-level signals for AGID**. This is part of "C".

### Stage B — Concept Bottleneck Layer (CBL) $\boldsymbol{C_\phi}$

For each of the $K$ concepts, a small concept head $h_{\phi_k}: \mathbb{R}^{2048 \times 7 \times 7} \rightarrow \mathbb{R}^{1 \times 7 \times 7}$ produces a spatial heatmap. The scalar concept activation is:

$$c_k(I) = \sigma\left(\text{GAP}(h_{\phi_k}(f_\theta(I)))\right)$$

where GAP is global average pooling and $\sigma$ is sigmoid. The heatmap $\mathbf{h}_k(I)$ is the upsampled $h_{\phi_k}$ output for visualization.

The concept vector $\mathbf{c}(I) = [c_1, \dots, c_K]$ is the *only* signal passed to the classifier.

### Stage C — Linear classifier $\boldsymbol{w}$

$$\hat{y}(I) = \sigma\left(\mathbf{w}^\top \mathbf{c}(I) + b\right)$$

The bottleneck is enforced by:
1. **No skip connections** from $f_\theta(I)$ to the classifier (this is the key architectural constraint that prevents post-hoc explanation drift).
2. **Linear** (not non-linear) classifier on $\mathbf{c}$, so the contribution of each concept to the prediction is $w_k \cdot c_k$, fully decomposable.

---

## 4. Concept Set (revised post Day 5 validation — 6 concepts)

### 4.1 Final concept set (after Day 5+ sanity check on 20+20 sample)

| Rank | Concept name | Cohen's d | Status | Heuristic signal definition |
|---|---|---|---|---|
| 🥇 1 | **BitPlane-LSB-Pattern** | -2.77 | ✅ ANCHOR | \|1-lag horizontal autocorrelation of LSB plane of grayscale\| |
| 🥈 2 | **Freq-Radial-Profile** | -1.83 | ✅ CONFIRMED | \|slope of log-log radial PSD of full-image FFT\| |
| 🥉 3 | **Color-Manifold-Deviation** | -1.63 | ✅ CONFIRMED | KL(image-RGB-hist \|\| real-image-prior-hist), 16³ bins |
| 4 | **HF-Noise-Anomaly** | -0.99 | ✅ CONFIRMED (direction adaptive) | variance(Laplacian(grayscale)) — direction is dataset-dependent; the linear classifier absorbs sign |
| 5 | **JPEG-QuantTrace-Absence** | +0.64 | 🟡 BORDERLINE — retest on GenImage | variance of DCT-coefficient residuals after re-quantization (grid=10) |
| 6 | **Texture-Geometry-Drift** | +0.58 | 🟡 BORDERLINE — retest on GenImage | 1 − Pearson(per-patch edge density, per-patch grayscale variance) |

### 4.2 Concepts considered but rejected on Day 5

- **Freq-Subband-Energy (8×8 block DCT mid+high ratio)** — d=-0.31 not significant. Replaced by Freq-Radial-Profile, which captures full-image spectral shape rather than JPEG-grid-aligned block energies.
- **EdgeSharpness-Inconsistency** — d=+0.05 effectively zero. Possible reason: modern T2I generators produce uniformly sharp output without spatial sharpness variance, while real photos do have spatial variation BUT the per-patch variance metric is dominated by overall scene composition. May reformulate using a more local sharpness measure if reintroduced.

### 4.3 Day 5 methodological discovery

**Frequency formulation choice matters more than concept-family choice.** Freq-Radial-Profile (d=-1.83) vs. Freq-Subband-Energy (d=-0.31) shows that within a single concept family ("frequency-domain analysis"), the specific operationalization can change the effect size by 6×. We adopt the radial-profile formulation for our final concept set; the failed block-DCT formulation is reported as ablation evidence in the paper.

### 4.4 Supervision strategy

Weakly-supervised — soft labels for each concept come from cheap heuristic detectors (the numpy code in `Code/scripts/day5_concept_extended.py`, lifted to PyTorch nn.Modules in `Code/cbnet_agid/concepts/`). During Phase B training, concept activations are matched against these soft labels via MSE. No human concept labels are required.

### 4.5 Pruning policy

We commit to 6 concepts ex ante (no further pruning). If a borderline concept (#5, #6) fails GenImage retest, it is replaced by **Recon-Residual-Magnitude** (DIRE-inspired) — a 9th candidate held in reserve and computed via a small frozen VAE.

### 4.6 Rationale

Each concept corresponds to a known artifact family in the AGID literature; together they span the catalog of AGID cues at multiple physical/signal levels (pixel-bit, frequency, color, edge, compression, texture). The bottleneck *forces* the model to express its decision through these 6 named concepts, making the explanation *complete* (the prediction is mathematically a linear combination of the 6 concept activations) and *faithful* (no skip connection allows the classifier to bypass the concepts).

---

## 5. Loss Function

The total loss is a weighted sum of four terms:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{task}} + \lambda_c \mathcal{L}_{\text{concept}} + \lambda_g \mathcal{L}_{\text{gen-consistency}} + \lambda_s \mathcal{L}_{\text{sparsity}}$$

### 5.1 Task loss
$$\mathcal{L}_{\text{task}} = -[y \log \hat{y} + (1-y)\log(1-\hat{y})]$$

### 5.2 Concept supervision loss (weakly-supervised)
$$\mathcal{L}_{\text{concept}} = \sum_{k=1}^{K} (c_k - \tilde{c}_k)^2$$
where $\tilde{c}_k$ is the heuristic soft label for concept $k$, normalized to $[0,1]$.

### 5.3 **Cross-generator concept consistency loss** (key novelty)
For mini-batches containing the *same* real image $I^{\text{real}}$ and its AI-generated counterparts $I^{g_1}, I^{g_2}$ (different generators, similar content via shared prompt or semantic match):

$$\mathcal{L}_{\text{gen-consistency}} = \sum_{k=1}^{K} \mathbb{E}_{(g_1, g_2)} \left[ \left| c_k(I^{g_1}) - c_k(I^{g_2}) \right| \right]$$

This forces each concept's activation to depend on *intrinsic AI-generated-ness* rather than *generator-specific signatures*. Critically, this is what makes the model generalize to unseen generators.

### 5.4 Sparsity regularizer (prevent degenerate single-concept shortcut)
$$\mathcal{L}_{\text{sparsity}} = -\sum_{k=1}^{K} c_k \log c_k$$
(entropy of concept activation distribution; high entropy means concepts are diverse, not collapsed)

### Loss weights (initial; tune via val set)
$\lambda_c = 0.5$, $\lambda_g = 0.2$, $\lambda_s = 0.05$. *Schedule:* warm up $\mathcal{L}_{\text{task}}$ alone for 1 epoch, then activate $\mathcal{L}_{\text{concept}}$, then $\mathcal{L}_{\text{gen-consistency}}$, then $\mathcal{L}_{\text{sparsity}}$.

---

## 6. Training Protocol

### 6.1 Two-phase training

**Phase 1 (Backbone warm-up, ~3 epochs):** Train the dual-stream backbone with a temporary FC classification head and standard BCE loss on a subset of GenImage. Discard the temporary head.

**Phase 2 (Bottleneck end-to-end, ~15–20 epochs):** Insert CBL between backbone and linear classifier. Train with $\mathcal{L}_{\text{total}}$.

### 6.2 Optimizer
- Optimizer: AdamW
- LR: 1e-4 (backbone with 0.1x decay), 5e-4 (CBL), 1e-3 (linear classifier)
- Weight decay: 1e-4
- Batch size: 32 (single 16GB GPU) or 64 (single 24GB GPU)
- Image size: 224×224 (standard) and 256×256 (LOTA-style)
- Total wall-clock budget: 5–7 days on single RTX 30/40-class GPU

### 6.3 Data augmentation
- Random horizontal flip
- Random JPEG compression (Q ∈ [70, 100]) — robustness
- Random Gaussian blur (σ ∈ [0, 2]) — robustness
- Random cropping
- **NOT** color jitter (would interfere with Color-Manifold-Deviation concept)

---

## 7. Datasets

### 7.1 Training and in-distribution evaluation
- **GenImage** (NeurIPS 2023): primary dataset; 8 generators × ~330k images each. Train on full split.
- **GenImage subsplit for cross-generator OOD experiments:** Train on SD-1.4 only, test on remaining 7 generators (BigGAN, Midjourney, Wukong, SD-1.5, ADM, GLIDE, VQDM) as OOD.

### 7.2 Out-of-distribution evaluation
- **ForenSynths** (Wang et al. CVPR 2020): 11 generators including ProGAN, StyleGAN, BigGAN, CycleGAN, GauGAN, StarGAN. *Used for cross-architecture and cross-domain OOD.*
- **DiffusionForensics** (Wang et al. ICCV 2023, comes with DIRE): subset of diffusion generators not in GenImage.
- **AIGCDetection** (Zhong et al. 2024, optional if time permits).

### 7.3 Real-image source
ImageNet validation set (subsampled to match AI-image counts in each split, avoiding class imbalance).

### 7.4 Compute estimate
- Single training run on full GenImage: ~24–36 hours on RTX 4090; ~48–60 hours on RTX 3090
- Inference on full test set (~500k images): ~3 hours on RTX 4090
- Total budget for all experiments (10 ablation rows): ~14 days of GPU time. **Tight but feasible.**

---

## 8. Evaluation Metrics

### 8.1 Detection performance (Standard)
- **Accuracy (ACC)** per generator + average
- **Average Precision (AP)** per generator + average
- **AUC** per generator + average

Report Acc/AP/AUC for:
- In-distribution test set (same generators as training)
- Cross-generator OOD (held-out generators in GenImage)
- Cross-architecture OOD (ForenSynths generators)

### 8.2 Interpretability metrics
- **Deletion / Insertion AUC** (Petsiuk et al. 2018): faithfulness of attribution maps
- **Pointing Game** (Zhang et al. 2018): localization accuracy when ground-truth artifact regions exist (sample subset with manual annotation, ~200 images)
- **TCAV-style concept sensitivity** (Kim et al. 2018): does each concept correlate with prediction?
- **Concept Discovery Rate (CDR)** and **Concept Consistency (CC)** as in E-BotCL (2025)

### 8.3 Compute and efficiency
- Inference latency (ms/image) on consumer GPU
- Trainable parameter count
- Training wall-clock time

### 8.4 Robustness
- Accuracy under JPEG compression (Q = 75, 85, 95)
- Accuracy under Gaussian blur (σ = 1, 2, 3)
- Accuracy after re-sizing (0.5×, 2×)

---

## 9. Baseline Methods (6 comparisons)

| # | Baseline | Year/Venue | Why included |
|---|---|---|---|
| 1 | **CNNDetection** (Wang et al.) | CVPR 2020 | Classic baseline, mandatory reference for AGID |
| 2 | **UniversalFakeDetect (UnivFD)** (Ojha et al.) | CVPR 2023 | CLIP-based universal detector, the "generalization" benchmark |
| 3 | **NPR** (Tan et al.) | CVPR 2024 | Recent SOTA, neighborhood-pixel relationship; closest to our backbone family |
| 4 | **LOTA** (Wang et al.) | ICCV 2025 | Latest bit-plane SOTA; shares bit-plane intuition with our backbone |
| 5 | **DIRE** (Wang et al.) | ICCV 2023 | Diffusion-specific baseline; reconstruction-based |
| 6 | **AIGI-Holmes** (Zhou et al., ICCV 2025) | ICCV 2025 | **Direct competitor** in explainability+generalization space; *may be infeasible to run locally* — if so, cite reported numbers from their paper |

If AIGI-Holmes cannot be run on our single-GPU setup, we will follow standard practice and cite their reported numbers on shared benchmarks (GenImage), and add a *fairness caveat* in the Experiments section.

---

## 10. Ablation Studies (planned)

1. **Per-concept ablation:** drop each of the 8 candidate concepts in turn; report Δ accuracy.
2. **Loss component ablation:** drop $\mathcal{L}_{\text{gen-consistency}}$, drop $\mathcal{L}_{\text{sparsity}}$, drop $\mathcal{L}_{\text{concept}}$; report effects.
3. **Backbone ablation:** dual-stream vs. only-pixel vs. only-bit-plane-NPR streams; report Δ accuracy.
4. **Number of concepts $K$:** $K \in \{4, 6, 8, 12\}$; identify sweet spot.
5. **Bottleneck vs. no-bottleneck:** baseline = same backbone, no CBL (post-hoc Grad-CAM only); compare both accuracy and faithfulness.

---

## 11. Expected Outputs (for paper)

| Section | Content | Approximate word count |
|---|---|---|
| Introduction | Background + Motivation + Differentiation from AIGI-Holmes + Contributions | 2,500 |
| Related Work | 7 method families + Concept Bottleneck literature + Explainability literature | 4,500 |
| Method | Architecture + Concepts + Losses + Training (with equations) | 6,000 |
| Experiments | Setup + In-distribution + OOD + Interpretability + Ablation + Robustness | 9,000 |
| Discussion (optional) | Failure cases + Limitations + Future work | 2,500 |
| Conclusion | Summary + Implications | 1,000 |
| Abstract | 250 words |
| **Total body** | | **~25,750** |
| **+ References + appendix** | | **~30,000 with refs and tables** |

This fits the 30,000-word target.

---

## 12. Timeline (1 month, single author)

| Week | Tasks | Deliverables |
|---|---|---|
| Week 1 (May 25 – May 31) | Setup environment; clone LOTA, NPR, GenImage; reproduce baselines on subset | Working baselines, dataset pipeline |
| Week 2 (Jun 1 – Jun 7) | Implement dual-stream backbone + CBL; train Phase 1 + Phase 2 on small subset | Initial Concept Bottleneck model, sanity-check results |
| Week 3 (Jun 8 – Jun 14) | Full training; in-distribution + OOD experiments; faithfulness metrics; ablations | Complete result tables |
| Week 4 (Jun 15 – Jun 22) | Paper writing (Stage 2 of pipeline); figures; tables; LaTeX polish | Final manuscript |
| Buffer (Jun 23 – Jun 24) | Stage 3 review / Stage 4 revision / Stage 5 format conversion / Stage 6 process summary | Submission package |

**Risk-aware buffer:** if Week 2 implementation slips, drop concepts 7 and 8 from the initial set (use 6 instead of 8), saving ~3 days.

---

**Next deliverable:** `03_Bibliography.md` — 80–90 references organized by topic.
