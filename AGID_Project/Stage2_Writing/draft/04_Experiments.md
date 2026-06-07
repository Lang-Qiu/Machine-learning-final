<!-- DRAFT — Experiments. Written 2026-06-03. ALL numbers verified from artifact files (no memory/handoff):
     F3_main_results.md, A_intervention_summary.md, B_confound_summary.md, gate3_full_audit.json,
     cbnet_multigen_forensynths.json, cbnet_forensynths.json (Route A, re-run), npr_forensynths.json (re-run),
     E1_sota_baselines.md (literature, labelled). Structure D-26: §1 Setup / §2 Detection / §3 Audit / §4 Ablations.
     Guards: no hard SOTA, no pure confound, no guaranteed interpretability, no jpeg_quant=generative-semantics.
     freq_radial is a SECONDARY signal on Route B (−4..−30pp), inert only on E-Delta — stated precisely. natbib keys. -->

## 4. Experiments

### 4.1 Setup

**Two models.** We evaluate two instances of CBNet-AGID that are identical in architecture and seed and differ only in training data: **Route B**, trained on four raw GenImage generators (20 epochs), and **E-Delta**, retrained on the Grommelt-style Q96-debiased corpus (10 epochs; §3.6). Unless stated otherwise, "the detector" denotes Route B.

**Datasets and evaluation tiers.** Training uses four GenImage \citep{zhu2023genimage} generators—Stable Diffusion v1.4, BigGAN, ADM, Midjourney—at 25k images per class. We evaluate at four tiers of increasing distribution shift: (i) **in-distribution** (SD-1.4 validation, 12k); (ii) **held-out, same generators** (BigGAN/ADM/Midjourney validation, 2k each); (iii) **cross-generator OOD within GenImage** (GLIDE, Wukong, VQDM—never seen in training, 2k each); and (iv) **cross-architecture OOD** on ForenSynths \citep{wang2020cnngenerated} (BigGAN, DeepFake, GauGAN, StarGAN—entirely different real and synthetic sources). The Q96-debiased variant re-encodes every image to uniform JPEG quality 96.

**Metrics.** We report accuracy, AUC, average precision, and—uncommonly for this literature—per-class real- and fake-accuracy, which proves diagnostic below. Calibration is strong (expected calibration error 0.0034).

**Baselines and the comparability problem.** Direct numerical comparison in this field is fraught: at least three training protocols are in active use—**P1** (train on SD-1.4 only, test on all generators; the original GenImage protocol used by most recent work), **P2** (per-generator training averaged across all eight, used by LOTA, which inflates the mean by including in-distribution cells), and **P3** (train on ProGAN). CBNet uses a fourth, **P4** (joint training on four generators), so for CBNet only GLIDE/Wukong/VQDM/SD-1.5 are genuinely held-out, whereas for P1 methods all non-SD generators are. We therefore tabulate baselines (Table 1) with an explicit **protocol label** and **comparability flag**, drawing reported numbers from the most recent uniform re-implementations \citep{zhu2023genimage, tan2024npr, wang2025lota, ojha2023universal}, and we are explicit that we claim **competitive, not state-of-the-art**, detection. Where a direct, same-protocol comparison is possible we make it: we re-evaluate NPR \citep{tan2024npr} on ForenSynths under our exact pipeline (§4.4).

*Table 1 (schematic): baselines on GenImage cross-generator, each row tagged P1/P2/P3/P4 and "directly comparable / re-implemented / protocol-divergent." Representative reported means: CNNSpot 64.2 (P1), UnivFD 79.5 (P1), AIDE 86.9 (P1), NPR 88.6 (P1), DRCT 89.5 (P1), C2P-CLIP 95.8 (P1), LOTA 98.9 (P2, not directly comparable); CBNet-AGID Route B reports two sub-rows—in-distribution (four training generators, ≥99.85) and held-out OOD (99.67)—under P4. Numbers from \citep{zhu2023genimage} Table 3 and recent re-implementations; see supplementary for full table and sources.*

### 4.2 Detection

**GenImage.** Route B is strong across all GenImage tiers (Table 2): 99.88% in-distribution, ≥99.85% on held-out training-family generators, and a **99.67% mean on the three unseen GenImage generators** (GLIDE 99.80, Wukong 99.45, VQDM 99.75), with per-class accuracies all ≥99.3%.

| Generator | Tier | n | Acc (%) | AUC | Real-acc | Fake-acc |
|---|---|---|---|---|---|---|
| SD-1.4 (val) | in-dist | 12000 | 99.88 | 0.9998 | 99.80 | 99.95 |
| BigGAN | held-out | 2000 | 99.90 | 1.0000 | 99.80 | 100.0 |
| ADM | held-out | 2000 | 99.95 | 0.9997 | 99.90 | 100.0 |
| Midjourney | held-out | 2000 | 99.85 | 1.0000 | 99.90 | 99.80 |
| GLIDE | OOD | 2000 | 99.80 | 1.0000 | 99.60 | 100.0 |
| Wukong | OOD | 2000 | 99.45 | 0.9999 | 99.60 | 99.30 |
| VQDM | OOD | 2000 | 99.75 | 1.0000 | 99.60 | 99.90 |

**Cross-architecture (ForenSynths).** On a benchmark whose real and synthetic images do not share GenImage's acquisition statistics, the same model attains **73.65% mean (AUC 0.820)**—StarGAN 96.42, DeepFake 82.24, BigGAN 61.10, GauGAN 54.82—with markedly low real-accuracy on BigGAN (22.2%) and GauGAN (9.7%). We report this here as a measurement and interpret it in §4.3c.

**Double dissociation (numbers).** Comparing Route B and E-Delta on matched versus mismatched evaluation distributions (from the GATE-3 audit) yields a clean dissociation, reported here without interpretation (see §4.3c): on a Q96-debiased validation set Route B scores **89.75%** mean and E-Delta **99.78%**; on the raw cross-generator OOD set Route B scores **99.73%** and E-Delta **95.0%**.

### 4.3 Audit

This section is the paper's core. We use the bottleneck's no-bypass guarantee (§3.3) to measure, causally, what the detector's accuracy rests on.

**4.3a Channel reliance.** The learned classifier is $\hat{y}=\sigma(\mathbf{w}^\top\mathbf{c}+b)$ with $\mathbf{w}=[+6.45,+9.84,+6.21,+3.73,\mathbf{-13.53},+7.35]$ (order: bitplane-LSB, freq-radial, color-manifold, hf-noise, **jpeg-quant**, texture-geometry) and $b=-5.06$. Three interventions converge on a sharp picture.

*Single-channel ablation.* Zeroing `jpeg-quant` collapses accuracy to chance on **every** generator (−49.2 to −49.8pp; 99.7%→50.2% cumulative); zeroing `freq-radial` costs a substantial but smaller −4 to −30pp; the remaining four channels each cost about a point or less on most generators (texture-geometry the largest exception, up to −5.8pp on Wukong). So reliance is concentrated, but not on a single channel alone.

*Keep-only-one.* Restricting the classifier to one channel disentangles their roles: `freq-radial` alone reaches **81.6–99.7%** across generators—it is the continuous discriminator—whereas `jpeg-quant` alone yields exactly 50% (its negative weight plus the bias label everything "real"). `jpeg-quant` thus acts as a **decision-flipper** that, in concert with `freq-radial`, sets the operating point, rather than as a standalone classifier.

*Counterfactual swap.* Replacing a single channel's value with the opposite class's mean confirms the causal lever: swapping `jpeg-quant` flips **23.9–72.5%** of real predictions and 9.8–44.9% of fake predictions, while every other channel—`freq-radial` included—flips under ~2% (at most 9% on one generator). 

Finally, `jpeg-quant` and `freq-radial` are **−0.80 correlated**: the two load-bearing channels encode the *same* low-level cue—a compression/container signal—in scalar and spectral form, and the other four channels are essentially inert despite being trained by the concept-MSE loss. Two lessons follow. First, **weight magnitude is an unreliable proxy for functional role**: `jpeg-quant` has the largest $|w|$ yet cannot classify alone, while `freq-radial` carries the standalone signal—only intervention reveals this. Second, the detector is effectively a two-channel, single-cue model; the bottleneck's six "concepts" are not six independent pieces of evidence.

**4.3b Confound battery.** If the load-bearing cue is a compression/container artifact, perturbing that artifact should degrade the detector. It does (Table 3). Re-encoding every image to a uniform JPEG quality of 95 (closing the PNG-vs-JPEG container gap) drops mean accuracy **−10.16pp** (worst on BigGAN −19.4 and ADM −14.0); re-saving as PNG with no other change costs **0.00pp** (a control); downsampling to 128² (destroying high-frequency content) collapses accuracy **−44.64pp**, driven by a real-accuracy collapse; and using disjoint real-image pools across OOD generators changes results by **+0.12pp**, ruling out a shared-real-pool artifact.

| Perturbation | mean Δacc | train-gen | OOD |
|---|---|---|---|
| JPEG-q95 (close container gap) | −10.16 | −10.83 | −9.28 |
| PNG re-save (control) | +0.00 | +0.00 | +0.00 |
| res128 (destroy high-freq) | −44.64 | −43.91 | −45.62 |
| independent real pool | — | — | +0.12 |

**4.3c Persistence under debiasing, and the inflation/generalization split.** We then ask the question Grommelt's black-box analysis could not: does the reliance survive removing the confound at the data level? We retrain on the Q96-debiased corpus (E-Delta) and re-audit. The `jpeg-quant` weight shrinks—from −13.53 to −9.11 (a 1.49× reduction)—which, taken alone, would suggest reduced reliance. **Intervention says otherwise:** on the debiased model, zeroing `jpeg-quant` still costs **−48.18pp** (it remains rank-1 by ablation), while `freq-radial`'s contribution falls to −0.90pp (it becomes inert once the container signal is debiased), and `jpeg-quant`'s per-generator real-vs-fake separation stays large (Cohen's $d$ −3.84 to −5.47). We label this **Outcome C (persistence)**: the channel is *not reducible* to the trivial PNG-vs-JPEG container label, since the model rebuilds its dependence on the same channel from a corpus in which that label has been neutralized. Whether the surviving signal reflects generative-pipeline traces or residual compression-history statistics that outlast uniform re-encoding is **unresolved**, and we do not claim the former. This also reads the double-dissociation of §4.2: Route B (trained on the confounded corpus) wins on raw-OOD that shares the confound, E-Delta wins on Q96-matched data, and **both** route their decision through `jpeg-quant`—debiasing re-tunes the channel rather than removing reliance on it.

The cross-architecture result bounds how much of the headline number is confound-inflated. Route B's accuracy falls from **99.67%** on GenImage-OOD to **73.65%** on ForenSynths—a 26-point drop on a container-mismatched benchmark, consistent with the confound battery. We are careful not to attribute the entire gap to the confound: ForenSynths is also a genuine GAN-versus-diffusion distribution shift, so the clean confound isolation remains B1/B2 and §4.3a, with ForenSynths as corroborating external evidence. The honest reading is **partial confound inflation alongside genuine partial generalization** (quantified next).

### 4.4 Ablations (what the data support)

**Generator coverage (single- vs. multi-generator), on a common benchmark.** Evaluated identically on ForenSynths, an otherwise-comparable single-generator (SD-1.4-only) model reaches only **52.21% (AUC 0.528)**—essentially chance, classifying almost every GAN image as real—whereas the four-generator Route B reaches **73.65% (AUC 0.820)** (Table 4). Multi-generator training therefore buys *genuine* partial cross-architecture generalization (+21.4pp). For reference, NPR \citep{tan2024npr}, trained on ProGAN and thus domain-matched to ForenSynths' GANs, scores **79.58% (AUC 0.894)** on the same benchmark—above Route B, underscoring that Route B's generalization is partial rather than complete. We flag that the single-vs-multi contrast is suggestive rather than perfectly controlled: the two models also differ in training code, loss, and data scale, so we read it as directional evidence, not a clean one-factor ablation.

| Model (ForenSynths) | Train protocol | Mean Acc (%) | Mean AUC |
|---|---|---|---|
| CBNet single-gen (SD-1.4) | P1 | 52.21 | 0.528 |
| CBNet Route B (4 gens) | P4 | 73.65 | 0.820 |
| NPR (ProGAN) | P3 | 79.58 | 0.894 |

**Ablations not run.** Our concept set ($K{=}6$), the dual-stream backbone, the no-bottleneck alternative, and the loss components were fixed *a priori*; we did not run controlled ablations over $K$, backbone variants, the presence of the bottleneck, or the individual loss terms ($L_{\text{pair}}$, $L_{\text{sparsity}}$). We therefore make no empirical claim about the optimality of $K{=}6$, the necessity of the bottleneck for accuracy, or the contribution of the auxiliary losses; we discuss these as limitations (§6) and future work rather than asserting unsupported results.
