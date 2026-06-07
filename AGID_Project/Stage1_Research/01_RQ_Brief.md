# Research Question Brief — AGID with Concept Bottleneck

**Project:** 2026 Spring ML Course Final — AGID Paper
**Version:** v1.0 (Stage 1.2 Full deliverable)
**Date:** 2026-05-24

---

## 1. Main Research Question (RQ)

> **How can a Concept-Bottleneck-style architectural interpretability mechanism be integrated into AI-Generated Image Detection so as to deliver both (a) human-faithful structured explanations and (b) cross-generator generalization, under a single-GPU compute budget?**

This RQ is *generative*, not merely descriptive — it asks "how can X be designed" rather than "does X work" — to ground a method-paper rather than a survey.

---

## 2. Sub-Research Questions (3)

### SubRQ-1 (Concept design) — *What concepts?*

> What set of 4–8 visual / signal-domain concepts can be defined that are (i) semantically meaningful as cues for "AI-generated-ness", (ii) computable from image inputs without requiring per-image human annotation, and (iii) cross-generator stable (i.e., the same concept retains its meaning whether the AI image was made by a GAN or a diffusion model)?

This is the conceptual-design question. We will propose a candidate set of 6–8 concepts grounded in the AGID literature's known artifact families (high-frequency noise, sub-band frequency anomaly, bit-plane LSB pattern, edge sharpness inconsistency, color manifold deviation, JPEG/sensor trace absence, texture–geometry drift, optional: reconstruction-residual concept), and prune to a final set via ablation.

### SubRQ-2 (Architecture and training) — *How to bottle?*

> How should a Concept Bottleneck Layer (CBL) be inserted into an existing strong AGID backbone such that (i) the final prediction is *architecturally forced* to pass through the concept vector (interpretability-by-design), (ii) each concept activation has both a scalar score and a spatial heatmap, and (iii) the training procedure does not require dense per-pixel human labels?

This is the architectural and supervision question. We will propose a two-stage training scheme: (Phase A) backbone feature extraction with optional supervised concept signals derived from cheap heuristics (e.g., DCT for frequency concepts, bit-plane decomposition for LSB concepts); (Phase B) end-to-end fine-tuning of the bottleneck and the linear classifier with concept-task joint loss.

### SubRQ-3 (Generalization mechanism) — *Why does this generalize?*

> What loss formulation and training-time augmentation strategy enforces *concept invariance across generators* while preserving *predictive power*? Specifically, can a cross-generator concept consistency loss + domain-mixed batch sampling improve cross-generator OOD accuracy at a minimal in-distribution cost?

This is the generalization-mechanism question. The hypothesis is: if concepts capture *generator-agnostic* image-physics properties (e.g., "no JPEG quantization trace") rather than *generator-specific* signatures (e.g., "checkerboard from this StyleGAN"), then by *forcing* the model to predict through these concepts via the bottleneck, generalization to unseen generators improves *because the model cannot use generator-specific shortcuts*.

---

## 3. Hypotheses (4, falsifiable, with planned experiments)

### H1 — *Structural interpretability is not a free lunch but is small-cost.*

The Concept-Bottleneck variant of an AGID detector is hypothesized to achieve **in-distribution accuracy within 2 percentage points** of the same backbone without the bottleneck (the "interpretability tax" is small).
**Experiment:** Train backbone + linear head (baseline) vs. backbone + CBL + linear head (ours) on the same GenImage train split; compare GenImage in-distribution test accuracy.

### H2 — *Bottleneck improves OOD generalization.*

The Concept-Bottleneck variant achieves **at least +3 percentage points** higher accuracy on out-of-distribution generators (OOD := generators unseen in training) than the baseline.
**Experiment:** Train on a subset of GenImage generators (e.g., SD-1.4 only), test on held-out generators (ADM, Midjourney, BigGAN, etc.) and on ForenSynths.

### H3 — *Concept explanations are more faithful than Grad-CAM.*

Concept activation maps are more faithful (per the standard deletion/insertion metric) than Grad-CAM applied to the baseline classifier on the same predictions.
**Experiment:** Compute deletion-AUC and insertion-AUC over a held-out set of 1000 images for both Grad-CAM-on-baseline and concept-heatmap-from-our-method.

### H4 — *Concept ablation reveals individual concept contribution.*

Removing any single concept from the bottleneck causes a *measurable* accuracy drop, with the drop differing across concepts — empirically demonstrating that the bottleneck is *not* using a degenerate single-concept shortcut.
**Experiment:** For each of the K concepts, retrain (or ablate at inference time) with that concept zeroed out, report per-concept accuracy delta.

---

## 4. Boundary Conditions and Scope

### In scope
- Detection on **still images** (RGB, 256×256 to 1024×1024).
- Generators of interest: GAN family + Diffusion family (LDM-based and pixel-based) + Auto-regressive (e.g., DALL-E 3 grade if in dataset).
- Compute: single GPU, RTX 30/40 class. All experiments must complete within wall-clock budget of ~3 weeks.
- Length: paper at 30,000 words, 80–90 references.

### Out of scope
- Video deepfake detection (separate field, distinct artifacts).
- Face-swap deepfake detection (distinct, narrow domain).
- Detection of LLM-generated text or text-conditioned hybrid media.
- Detection robustness against adaptive *adversarial* attacks (a separate paper-length problem; we acknowledge robustness to *non-adaptive* post-processing like JPEG/blur only).
- Real-time low-latency deployment optimization (we report inference time but do not optimize against ms-scale targets).

---

## 5. Success Criteria for the Paper

| Criterion | Threshold |
|---|---|
| In-distribution accuracy on GenImage | ≥ 97% (matching strong baselines like NPR / LOTA) |
| Cross-generator (OOD) accuracy on held-out generators | ≥ 90% averaged across ≥ 5 unseen generators |
| Concept-explanation faithfulness | Deletion-AUC ≤ Grad-CAM's deletion-AUC by ≥ 0.05 (lower-is-better for deletion) |
| Ablation table | Each of 6 concepts contributes ≥ 0.5pp to final accuracy when included |
| Code release | Public GitHub repo with training + evaluation scripts + pre-trained weights |

If we miss in-distribution by > 2pp OR fail OOD criterion, the paper pivots into a "honest limitations" narrative — but Method/Concepts/Visualization sections remain valuable contributions and the paper can still defensibly hit A+C grade (4/5) on novelty.

---

## 6. Risk Register (top 5)

| # | Risk | Mitigation |
|---|---|---|
| 1 | **Concepts fail to learn (degenerate solution).** Bottleneck collapses to using only 1–2 concepts as classification shortcut. | Sparsity-encouraging entropy regularizer + supervised concept signals from heuristics + curriculum: train concepts independently first, then jointly. |
| 2 | **OOD accuracy gap larger than hoped.** Cross-generator concept consistency loss is insufficient. | Add domain-mixed batch sampling + augmentation-induced consistency (predictions invariant across small JPEG/blur perturbations). |
| 3 | **AIGI-Holmes already covers explainability+generalization.** Reviewer says we are subsumed. | Sharpen differentiation: ours is structural / single-GPU / falsifiable concepts vs. theirs is MLLM-text / cluster-scale / post-hoc rationalization. Section 1 (Intro) must lead with this. |
| 4 | **Backbone choice trap.** If we pick LOTA as backbone, our "novelty" looks like LOTA + concepts on top. | Pick NPR or a hybrid backbone — see Methodology §4 for the backbone decision rule. |
| 5 | **One month is too short.** Implementation + training + ablation + writing within 30 days. | Frontload code reuse: LOTA repo + NPR repo + GenImage dataset all exist. Week 1: setup + reproduce. Week 2: implement CBL. Week 3: experiments. Week 4: paper polish. |

---

## 7. RQ Decision Log

| Decision | Choice | Date |
|---|---|---|
| Topic | AGID (AI-Generated Image Detection) | 2026-05-24 |
| Emphasis | Explainability primary, Generalization secondary | 2026-05-24 |
| Novelty grade target | A+C (4–5/5) | 2026-05-24 |
| Direction | Concept Bottleneck Framework for AGID | 2026-05-24 |
| Backbone choice | Deferred (NPR vs. LOTA vs. hybrid) — see Methodology Blueprint | 2026-05-24 |

---

**Next deliverable:** `02_Methodology_Blueprint.md` — concrete architecture, losses, datasets, experimental plan.
