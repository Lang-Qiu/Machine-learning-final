# Batch 3: Concept intervention summary (A3 + A4)

All results computed in closed form from `full_inference_dump.npz` using
`logit = w·c + b` (no GPU inference). Linear head means concept interventions
have exact algebraic effects.

Classifier head: `logit = w·concepts + b` where
- `w = [+6.45, +9.84, +6.21, +3.73, -13.53, +7.35]`
- `b = -5.06`
- Concept order: `[bitplane_lsb, freq_radial, color_manifold, hf_noise, jpeg_quant, texture_geometry]`

---

## A3 — Zero-out ablation: which concepts carry information?

### Per-concept Δaccuracy (% point change when concept_k is zero'd)

| Concept           | ADM    | BigGAN | GLIDE  | Midjourney | SD-1.4 | VQDM   | Wukong |
|-------------------|--------|--------|--------|------------|--------|--------|--------|
| bitplane_lsb      |  +0.05 |  +0.15 |  +0.10 |   -0.05    |  +0.00 |  +0.05 |  -0.35 |
| freq_radial       | -21.50 | -30.25 |  -4.20 |  -15.40    | -24.60 |  -5.05 | -19.70 |
| color_manifold    |  +0.05 |  +0.15 |  +0.15 |   -0.15    |  +0.05 |  -0.10 |  -0.35 |
| hf_noise          |  +0.05 |  +0.10 |  +0.05 |   +0.00    |  -0.05 |  +0.05 |  -0.45 |
| **jpeg_quant**    | **-49.80** | **-49.70** | **-49.55** | **-49.45** | **-49.60** | **-49.50** | **-49.20** |
| texture_geometry  |  -0.90 |  +0.15 |  +0.15 |   -0.80    |  -2.60 |  -1.30 |  -5.75 |

**Read**: Zero-out of `jpeg_quant` collapses accuracy by ~50 pp on EVERY generator
(99.7% → ~50%, random guessing). `freq_radial` is secondary (5-30 pp drops).
The remaining **4 concepts are essentially dead weight** (|Δ| < 1 pp except
texture_geometry on Wukong = -5.75 pp).

### Cumulative ablation (zero concepts in descending |w| order)

| step | removed                                                                   | acc (%) |
|------|---------------------------------------------------------------------------|---------|
| 0    | (baseline)                                                                | 99.75   |
| 1    | jpeg_quant                                                                | **50.21** |
| 2    | + freq_radial                                                             | 66.32   |
| 3    | + texture_geometry                                                        | 81.41   |
| 4    | + bitplane_lsb                                                            | 60.39   |
| 5    | + color_manifold                                                          | 50.00   |
| 6    | + hf_noise (all zero'd)                                                   | 50.00   |

**Read**: Single removal of `jpeg_quant` already gives random guessing. Non-monotonic
behavior at steps 2-4 reflects the **bias term `b = -5.06`** dominating once the
strongest features are gone — the model is effectively a one-feature logistic regressor.

### Keep-only-one: what can each concept achieve alone?

| only concept     | ADM   | BigGAN | GLIDE  | Midjourney | SD-1.4 | VQDM   | Wukong |
|------------------|-------|--------|--------|------------|--------|--------|--------|
| bitplane_lsb     | 51.45 |  51.00 |  71.00 |   54.25    | 49.80  |  50.0  |  49.80 |
| **freq_radial**  | **95.50** | **99.65** | **94.05** | **89.90** | **91.30** | **82.20** | **81.65** |
| color_manifold   | 51.25 |  52.65 |  78.25 |   50.40    | 50.00  |  66.60 |  49.95 |
| hf_noise         | 50.00 |  50.00 |  50.00 |   50.00    | 50.00  |  50.00 |  50.00 |
| jpeg_quant       | 50.00 |  50.00 |  50.00 |   50.00    | 50.00  |  50.00 |  50.00 |
| texture_geometry | 52.15 |  49.55 |  49.10 |   50.95    | 55.85  |  51.70 |  57.40 |

**Read**: **`freq_radial` alone achieves 82-99.65% per generator** — it is the truly
discriminative signal. `jpeg_quant` alone gives 50% because `w_jpeg<0` combined with
`b=-5.06` pushes every prediction to "real" — it works as a **decision-flipper** in
concert with `freq_radial`, not as a standalone signal.

---

## A4 — Counterfactual swap: which concepts CAUSE the prediction?

Per-image intervention: replace concept_k with the per-generator mean of the
opposite class. Measure: of correctly-classified images, what fraction flip to
the opposite prediction?

### Real → Fake swap (flip rate %): real images get fake's concept-k mean

| concept          | ADM  | BigGAN | GLIDE  | Midjourney | SD-1.4 | VQDM | Wukong |
|------------------|------|--------|--------|------------|--------|------|--------|
| bitplane_lsb     |  0.1 |   0.0  |   0.2  |    0.1     |   0.0  |  0.0 |   0.0  |
| freq_radial      |  1.0 |   9.0  |   0.9  |    0.4     |   0.5  |  0.4 |   0.4  |
| color_manifold   |  0.1 |   0.0  |   0.1  |    0.1     |   0.0  |  0.1 |   0.0  |
| hf_noise         |  0.0 |   0.0  |   0.0  |    0.0     |   0.0  |  0.0 |   0.0  |
| **jpeg_quant**   | **47.4** | **53.9** | **72.5** | **42.8** | **23.9** | **53.7** | **24.2** |
| texture_geometry |  0.2 |   0.0  |   0.0  |    0.0     |   0.1  |  0.1 |   0.3  |

### Fake → Real swap (flip rate %): fake images get real's concept-k mean

| concept          | ADM  | BigGAN | GLIDE | Midjourney | SD-1.4 | VQDM | Wukong |
|------------------|------|--------|-------|------------|--------|------|--------|
| bitplane_lsb     |  0.0 |   0.0  |   0.0 |    0.1     |   0.1  |  0.0 |   0.2  |
| freq_radial      |  0.0 |   0.2  |   0.1 |    0.2     |   0.0  |  0.3 |   1.6  |
| color_manifold   |  0.0 |   0.0  |   0.0 |    0.0     |   0.1  |  0.1 |   0.2  |
| hf_noise         |  0.0 |   0.0  |   0.0 |    0.0     |   0.1  |  0.0 |   0.1  |
| **jpeg_quant**   | **36.6** | **29.9** | **9.8** | **33.6** | **32.5** | **44.9** | **42.5** |
| texture_geometry |  0.0 |   0.0  |   0.0 |    0.0     |   0.1  |  0.1 |   0.8  |

**Read**: **`jpeg_quant` is the only concept causally tied to predictions.**
Swapping it alone flips 24-72% of real predictions and 10-45% of fake predictions.
All other 5 concepts have <2% flip rate — they correlate with class but are not
load-bearing for the decision.

---

## Mechanistic story (combine with Batch 2)

| Evidence | Finding |
|---|---|
| A2 (weight) | `jpeg_quant` |w|=13.53, 1.4× the next concept |
| A3 (zero-out) | Remove `jpeg_quant` → 99.7% → 50% on every generator |
| A3 (keep-only) | `jpeg_quant` alone = 50% (works as flipper, not as classifier) |
| A4 (causal swap) | `jpeg_quant` only concept with non-trivial flip rate |
| B1 (jpeg-q95) | Unified JPEG re-encoding drops fake_acc 100% → 60-90% |
| B2 (res128) | Resolution destruction collapses real_acc to 8% |
| A5 (correlation) | `jpeg_quant ↔ freq_radial` = -0.80; high redundancy |

**Synthesis**: CBNet-AGID is effectively a 2-feature classifier:
1. `jpeg_quant` (PNG vs JPEG container difference) — provides the **bias-flipper** that pushes predictions toward "real" when present
2. `freq_radial` (low-frequency spectral profile) — provides the **continuous discriminator** capable of 82-99% accuracy alone, but is **strongly anti-correlated with `jpeg_quant`** (-0.80), meaning it's largely encoding the same compression signal in spectral form

The remaining 4 concepts (`bitplane_lsb`, `color_manifold`, `hf_noise`, `texture_geometry`)
contribute <1 pp each. They are trained by the concept-MSE loss but the linear head
**learns to ignore them**.

---

## Implications for paper (writing Stage 2)

### What CAN be claimed honestly

- ✓ **Concept bottleneck enables mechanistic audit**: the 6-concept structure made it
  *possible* to find this leak; a blackbox CNN would still give 99.67% with no signal
  about why.
- ✓ **Cross-generator generalization holds** (Route B held-out + OOD all ≥99.4%) at the
  benchmark level — useful baseline number.
- ✓ **Calibration is excellent** (ECE 0.0034).
- ✓ **OOD result is not a real-pool artifact** (B5 confirmed).

### What MUST be qualified

- ⚠️ The 99.67% number relies ~50% on PNG/JPEG format leakage (B1).
- ⚠️ The "6 interpretable concepts" claim is **misleading** — effectively only `jpeg_quant`
  and (redundantly) `freq_radial` are used by the classifier head.
- ⚠️ Reframing recommendation: Title the contribution as
  **"Concept bottleneck as a diagnostic instrument for confound detection"**
  rather than "Concept bottleneck for AGID accuracy."

### Pivot opportunity

This finding *itself* is publishable novelty: a controlled audit revealing that
SOTA-grade AGID on GenImage is partly format-detection, with concept bottleneck
as the audit tool. Frame as:
> "We trained a 6-concept CBNet-AGID achieving 99.67% OOD acc on GenImage. We
> then used the model's own structure to mechanically audit the contribution of
> each concept and showed that ~half the signal traces to image-container artifacts
> (PNG vs JPEG), not generative-pipeline traces. This positions concept bottleneck
> as a *diagnostic instrument* for benchmark confound auditing in addition to a
> classifier architecture."

This is **higher-value C-grade novelty** than "we got 99% on GenImage."

---

## Files produced (this batch)

- `A3_concept_zeroout.csv` — per-(concept, generator) Δacc, real/fake acc, AUC
- `A3_concept_zeroout.png` — heatmap of Δacc
- `A3_cumulative_ablation.csv` — sequential removal trace
- `A3_keep_only.csv` — per-(concept-alone, generator) acc/auc
- `A4_real_to_fake_swap.csv` — counterfactual real→fake flip rate
- `A4_fake_to_real_swap.csv` — counterfactual fake→real flip rate
- `A4_concept_swap.png` — flip-rate heatmaps
- `A_intervention_summary.md` — this file
