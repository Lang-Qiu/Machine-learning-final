# Synthesis — Field Positioning and Gap Analysis

**Project:** 2026 Spring ML Course Final — AGID Paper
**Version:** v1.0 (Stage 1.2 Full deliverable)
**Date:** 2026-05-24

---

## 1. Field Snapshot (where AGID stands as of mid-2026)

AI-Generated Image Detection has moved through three eras in five years.

**Era 1 (2019–2021): "Are GAN images detectable?"** Pioneered by Wang et al.'s 2020 CNNDetection finding (a simple ResNet50 trained on ProGAN generalizes surprisingly well across GAN families), and Zhang et al.'s 2019 spectral-domain analysis. The field was largely concerned with proving feasibility on GAN outputs.

**Era 2 (2022–2024): "Can we generalize?"** Diffusion models broke first-generation detectors. Three response families emerged:
- *Reconstruction-based*: DIRE (Wang et al. ICCV 2023), AEROBLADE (CVPR 2024), FakeInversion (CVPR 2024) — use diffusion or auto-encoder inversion residuals.
- *Foundation-model-feature*: UniversalFakeDetect (Ojha et al. CVPR 2023), CLIP-based methods, MoLE (2024) — leverage CLIP-ViT's frozen features.
- *Signal-domain refinement*: NPR (Tan et al. CVPR 2024), LGrad (Tan et al. CVPR 2023), PatchCraft (Zhong et al. 2023), AIDE (Yan et al. 2024) — engineer better low-level signals.

**Era 3 (2025–2026, current): "Can we explain — and still generalize?"** The latest wave attempts to add interpretability without sacrificing detection quality:
- *LOTA* (ICCV 2025) makes bit-plane signal interpretable by construction (the bit-plane decomposition itself is semantically intelligible).
- *AIGI-Holmes* (ICCV 2025) attempts text-based explanations via LLaVA + DPO + Multi-Expert Jury annotation.
- *ThinkFake* (2025), *ForenX* (2025), *Ji et al. Grounded Reasoning* (2025) — all variations of "MLLM rationalizes the detection".

Two competing philosophies have crystallized:
- **Heavy MLLM + post-hoc text rationalization** (AIGI-Holmes, ThinkFake, ForenX) — requires cluster-scale compute, produces fluent text explanations whose *faithfulness* to the actual decision is unverified.
- **Lightweight signal-domain + visual interpretability** (LOTA, NPR's saliency maps, PatchCraft's patch attention) — runs on single GPU, produces structurally-grounded but harder-to-narrate explanations.

The field has *not yet* reconciled these two — and that is the gap we target.

---

## 2. Method Family Comparison

| Family | Representative | Cross-gen acc. (Avg on GenImage) | Explainability mode | Compute (training) | OOD reported |
|---|---|---|---|---|---|
| Classic CNN | CNNDetection (Wang 2020) | ~85% (GAN), ~50% (diffusion) | None | 1 GPU | Weak |
| Frequency-domain | FreqNet (Tan 2024), AIDE (Yan 2024) | ~88-92% | Spectral plots (post-hoc) | 1-2 GPU | Mid |
| Reconstruction-based | DIRE (Wang 2023), AEROBLADE (Ricker 2024) | ~85-93% | Reconstruction-error heatmap | 1-2 GPU + diffusion inference | Mid |
| Gradient/NPR | NPR (Tan 2024), LGrad (Tan 2023) | ~96-98% | None / Grad-CAM | 1 GPU | Strong |
| Bit-plane | **LOTA (Wang 2025)** | **98.9%** | Bit-plane signal is intrinsically interpretable | 1 GPU | **Strong (>98% cross-family)** |
| CLIP / foundation | UnivFD (Ojha 2023), MoLE (Liu 2024) | ~89-93% | None | 1-4 GPU | Strong |
| **MLLM / reasoning** | **AIGI-Holmes (Zhou 2025), ForenX (Tan 2025)** | 97.8-99.2% | **Text rationalization** | **8+ A100s for training** | **Strong** |

**Observations:**
1. **LOTA is the highest-accuracy single-GPU method as of mid-2026.** Strong contender for our backbone.
2. **NPR (CVPR 2024) is also a strong, code-available, ResNet-based baseline.** Easier to slot a Concept Bottleneck on top.
3. **AIGI-Holmes claims 99.2% accuracy + textual explanations** — but at cluster-scale training cost and post-hoc explanation faithfulness uncertainty.
4. **No prior method provides structurally-enforced, ablatable, multi-concept interpretability for AGID.** This is our claim space.

---

## 3. Gap Analysis (4 gaps identified, our work targets §3.1)

### §3.1 The "structural interpretability + single-GPU generalization" gap (our target)

**Statement of gap:** As surveyed (Lin et al. 2025, *Methods and Trends in Detecting AI-Generated Images* arXiv:2502.15176; and *Recent Advances on Generalizable Diffusion-generated Image Detection* arXiv:2502.19716), the AGID field has produced (i) strong single-GPU detectors *without* interpretability (NPR, LOTA), and (ii) interpretable detectors at cluster-scale (AIGI-Holmes, ForenX). **The combination of (a) structural / ante-hoc interpretability, (b) single-GPU feasibility, (c) cross-generator generalization, has not been demonstrated in a single architecture.**

**Why this matters:**
- Post-hoc explanations from MLLMs can be unfaithful ("rationalization") — well-documented limitation in interpretable ML (Lipton 2018, Rudin 2019).
- Cluster-scale methods are not deployable in many practical settings (academic labs, course projects, edge devices, smaller institutions).
- The current ICCV 2025 ecosystem ironically rewards explainability *and* heavy compute simultaneously, which excludes most practitioners.

### §3.2 The "concept-level supervision without expensive labels" gap (we partially address)

Most CBM literature (Koh 2020, Margeloiu 2021, Mahinpei 2021) assumes per-image human-annotated concept labels. In AGID, no such concept-annotated dataset exists. Two recent works partially solve this:
- *E-BotCL* (April 2025): dual-path contrastive self-supervised concept discovery on CUB200/ImageNet.
- *LFCBM* (Oikarinen et al. ICLR 2023): label-free CBM using CLIP-based concept names.

**Our contribution:** weak supervision via cheap heuristic detectors (DCT for frequency concepts, bit-plane entropy for LSB concepts, etc.) — neither requires human annotation nor relies on language priors. This is methodologically novel for AGID.

### §3.3 The "cross-generator concept consistency" gap (we propose first solution)

No existing CBM literature addresses the question: *how do we ensure concepts maintain the same meaning across different domains/generators?* This is central to AGID because we want concepts to capture "AI-generated-ness" rather than "this specific generator's signature".

**Our contribution:** explicit cross-generator concept consistency loss $\mathcal{L}_{\text{gen-consistency}}$ (§5.3 of Methodology Blueprint). To our knowledge, this is the first such formulation.

### §3.4 The "interpretability faithfulness metric" gap (we adopt established metrics)

The AGID community uses ad-hoc qualitative figures for "interpretability" claims. We import the deletion/insertion AUC (Petsiuk et al. 2018) and TCAV (Kim et al. 2018) metrics from interpretable ML to *quantify* faithfulness. This is not novel by itself but is novel in *transferring established faithfulness metrics into AGID*.

---

## 4. Differentiation From AIGI-Holmes (paragraph-by-paragraph)

AIGI-Holmes (Zhou et al. ICCV 2025) is the most direct competitor and *must* be addressed head-on in Introduction §1.3 of the paper.

**Their approach:** A Visual Expert (NPR-based small model) supplies low-level detection cues; a tuned LLaVA-13B with multi-expert-jury supervision and DPO produces natural-language explanations describing "anatomical errors, physical law violations, common sense violations" — i.e., high-level semantic anomalies. Collaborative decoding fuses Visual Expert + MLLM at inference. Trained on Holmes-SFTSet (65K) + Holmes-DPOSet (4K) using 4-MLLM-jury annotation (Qwen2VL-72B, InternVL2-76B, InternVL2.5-78B, Pixtral-124B).

**Our approach:** A dual-stream lightweight backbone (ResNet-50 + bit-plane/NPR signal) feeds an *architecturally-enforced* Concept Bottleneck Layer of 6 concepts (HF-Noise, BitPlane-LSB, Freq-Subband, EdgeSharpness, Color-Manifold, JPEG-QuantTrace). Predictions are *mathematically* a linear combination of concept activations — no MLLM, no post-hoc explanation.

**Five concrete differentiations:**

| Axis | AIGI-Holmes | CBNet-AGID (ours) |
|---|---|---|
| Interpretability *kind* | Textual narrative explanations of high-level anomalies | Concept-vector + per-concept spatial heatmap |
| Interpretability *guarantee* | None (text is generated by separate LLM head) | Structural: prediction *passes through* concept layer |
| Compute *training* | 8+ A100-class GPUs (for MLLM SFT + DPO) | 1 RTX 30/40-class GPU |
| Compute *inference* | LLaVA-13B forward pass per image | ResNet-50 + linear classifier per image |
| Ablation possibility | Hard (MLLM is monolithic) | Easy (drop each of the 6 concepts) |

**Honest acknowledgment** (must appear in our Discussion §6.1): AIGI-Holmes provides natural-language explanations that are richer and more accessible to non-experts than our concept-vector visualizations. We do not claim superiority on explanation *richness*; we claim superiority on explanation *faithfulness* and *compute accessibility*.

---

## 5. Positioning Statement (for Introduction § "Contributions" subsection)

> This paper proposes **CBNet-AGID** (Concept-Bottleneck Network for AI-Generated Image Detection), the first AGID method that combines (i) *architecturally-enforced* structural interpretability via a Concept Bottleneck Layer of 6 weakly-supervised concepts grounded in established image-signal-domain artifacts; (ii) cross-generator generalization via a novel concept-consistency loss that forces concept activations to be generator-invariant; and (iii) single-GPU computational feasibility throughout training and inference. We demonstrate that CBNet-AGID achieves cross-generator generalization competitive with the heavyweight MLLM-based AIGI-Holmes (ICCV 2025) on the GenImage benchmark, while providing concept-level explanations whose faithfulness is *mathematically guaranteed* by the bottleneck architecture and *quantified* via insertion/deletion AUC and TCAV sensitivity. Our contributions are: (a) the first Concept Bottleneck framework adapted to AGID; (b) the first cross-generator concept consistency loss; (c) an ablation-rich evaluation that decomposes detection performance into individual concept contributions; and (d) public code release for reproducibility.

This is the message that will appear in the paper's Abstract, Introduction §1.3, and Conclusion. The four-bullet contribution structure is what the Novelty (5/5) score rests on:
- (a) is **A+C**: existing CBM technology (A) + adapted to AGID (C-adaptation)
- (b) is closer to pure **C**: cross-generator concept consistency loss is genuinely new
- (c) is methodological rigor
- (d) is good open science

---

## 6. Risk and Honest Limitations

To stand up to a Stage 3 reviewer audit, we list the most likely critical attacks:

1. **"You are subsumed by AIGI-Holmes."** Response: differentiated on faithfulness guarantee and compute. AIGI-Holmes's text explanations *may be unfaithful*, ours *cannot be*. AIGI-Holmes requires cluster compute, ours does not.

2. **"Your concepts are designed-not-learned, so what's novel?"** Response: concept *labels* are heuristic-derived; concept *encoders* are learned; the *bottleneck architecture* + *consistency loss* are novel. Compare to E-BotCL (April 2025) which learns concepts but on general images; we adapt to AGID with domain-specific concept design.

3. **"6 concepts may not span all AI artifacts."** Response: candidly acknowledge. Our concept set is *AGID-knowledge-curated*, not exhaustive. We invite future work to extend. The bottleneck architecture itself is extensible.

4. **"Cross-generator consistency may hurt in-distribution accuracy."** Response: H1 hypothesis falsifiable; reported. We tolerate ≤ 2pp ID drop for OOD gain.

5. **"Why not just use AIGI-Holmes's NPR Visual Expert with our CBL on top?"** Response: that's actually our backbone consideration — NPR + dual-stream is our backbone. We will compare CBL-on-NPR vs. CBL-on-dual-stream as one of our ablation rows.

---

## 7. Expected Contribution to Each Score Dimension

| Dimension | Weight | How this work earns the score |
|---|---|---|
| **Informativeness** (10) | Aim 8-9 | 90-reference comprehensive literature; 7-family taxonomy; differential analysis of AIGI-Holmes; 6-concept design with physical motivation per concept; ablation tables; faithfulness metrics imported from interp ML. |
| **Novelty** (5) | Aim 4-5 | A+C = NPR/LOTA backbone (A) + Concept Bottleneck for AGID (C) + cross-generator concept consistency loss (C') + faithfulness benchmarking transfer (C''). Multi-component C, ablation-able. |
| **Attractiveness** (10) | Aim 8-9 | Concept heatmap visualizations (6 concepts × multiple images); contribution decomposition bar charts; insertion/deletion curves; method-family pareto plot (accuracy vs. interpretability); risk-honest discussion. |
| **Writing** (15) | Aim 13-14 | Tight 30k-word, no padding; equations precisely defined; clear differentiation language vs. AIGI-Holmes; honest limitations section; structured Related Work covering all 7 families. |
| **TOTAL** | 40 | Aim 33-37 (~85-92%) |

---

## 8. Stage 1.2 Full — Delivery Audit

| Required Deliverable | File | Status |
|---|---|---|
| RQ Brief | `01_RQ_Brief.md` | ✅ Delivered |
| Methodology Blueprint | `02_Methodology_Blueprint.md` | ✅ Delivered |
| Bibliography (40–80 → 90 refs) | `03_Bibliography.md` | ✅ Delivered (90 refs) |
| Synthesis | `04_Synthesis.md` | ✅ Delivered (this file) |

**Stage 1 status:** Complete pending user FULL Checkpoint confirmation.

---

## 9. Decisions Locked / Decisions Still Open

| Item | Status | Notes |
|---|---|---|
| Direction | Locked | Concept Bottleneck Framework for AGID |
| Novelty target | Locked | A+C = 4-5/5 |
| Paper length | Locked | 30,000 words |
| Reference count | Locked | 90 (target was 80-90) |
| Compute | Locked | Single GPU, RTX 30/40 class |
| Team | Locked | Solo |
| Communication language | Locked | Chinese (user-AI) / English (paper) |
| **Backbone choice** | **Open** | Dual-stream proposed; final choice may shift in Stage 2 Phase 1 (sanity-check experiments) |
| **Final concept set (6 of 8)** | **Open** | Determined in Stage 2 via initial ablation |
| **Whether to actually run AIGI-Holmes locally** | **Open** | Depends on whether their checkpoint runs on single GPU |

---

**End of Stage 1.2 Full deliverables.**

**Pipeline next step:** FULL Checkpoint — user reviews and confirms transition to Stage 2 WRITE.
