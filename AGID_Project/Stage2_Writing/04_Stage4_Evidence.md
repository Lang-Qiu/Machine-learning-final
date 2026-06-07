# Stage 4 (REVISE) — New Experimental Evidence

Append-only log of the peer-review-response experiments (Stage 3 → Stage 4). All numbers
computed from disk artifacts by the **identical** code path across seeds (`scripts/analyze_all.py`
dump → `scripts/concept_intervention.py` closed-form zero-out). Protected files untouched.

Status: R-2 multi-seed ✓ · R-1 baseline ✓ · P0-#1 attribution head-to-head ✓ · P0-#2 multi-seed E-Delta ✓ (seed-1 ep10 complete, seed-2 ep6 partial). All Stage-4 + peer-review-response experiments complete.

---

## R-2 — Multi-seed replication (resolves Stage-3 MAJOR issue M1: "single seed n=1")

Route B, identical config (4 generators, 25k/class, 20 ep, batch 32, same LRs/AMP/aug),
only `--seed` differs. seed-42 = original Route B (`Logs/cbnet_multigen_main_25k_s42`);
seed-1 = `Logs/cbnet_multigen_seed1`; seed-2 = `Logs/cbnet_multigen_seed2` (running).

### Detection accuracy — STABLE across seeds ✓

Per-generator acc (%), from the canonical 7-generator dump (1000 ai + 1000 real each,
deterministic seed-0 sampling; OOD reals = shared SD-1.4 val/nature):

| seed | SD-1.4 | BigGAN | ADM | MJ | **in-dist mean** | GLIDE | Wukong | VQDM | **OOD mean** |
|------|--------|--------|-----|------|------------------|-------|--------|------|--------------|
| 42   | 99.85  | 99.95  | 99.95 | 99.80 | ~99.89 | 99.80 | 99.45 | 99.75 | **99.67** |
| 1    | 99.85  | 99.80  | 99.95 | 99.45 | **99.76** | 99.90 | 99.45 | 99.85 | **99.73** |
| 2    | 99.85  | 99.90  | 99.85 | 99.60 | **99.80** | 99.90 | 99.65 | 99.90 | **99.82** |

(seed-42 dump per-gen recomputed via analyze_all = SD 99.85 / BigGAN 99.95 / ADM 99.95 /
MJ 99.80; published OOD headline 99.67 from `Results/ood_eval_full.json`.)
→ **Headline detection numbers are seed-stable** (OOD mean 99.67 / 99.73 / 99.82; in-dist all ≥99.45).
The "competitive single-GPU detector" claim survives multi-seed.

### Concept attribution — NOT stable across seeds (this is the important finding)

**Closed-form per-concept zero-out, mean Δacc over the 7 generators (pp):**

| concept | seed-42 | seed-1 | seed-2 | stable? |
|---------|--------:|-------:|-------:|---------|
| bitplane_lsb     |  −0.01 |  −0.64 |  −0.14 | all ≈0 ✓ |
| **freq_radial**  | **−17.24** | **−38.66** | **−24.73** | top-2 pair; magnitude swings |
| color_manifold   |  −0.03 |  −0.97 |  −0.12 | all ≈0 ✓ |
| hf_noise         |  −0.04 |  −1.04 |  −0.19 | all ≈0 ✓ |
| **jpeg_quant**   | **−49.54** | **−24.49** | **−25.73** | top-2 pair; magnitude swings |
| texture_geometry |  −1.58 |  −3.23 |  −7.81 | small–moderate |

**Classifier weights w (6-vector) + bias b:**

| seed | bitplane | freq_radial | color | hf_noise | jpeg_quant | texture | bias | rank-1 \|w\| |
|------|---------:|------------:|------:|---------:|-----------:|--------:|-----:|-------------|
| 42 | +6.45 | +9.84 | +6.21 | +3.73 | **−13.53** | +7.35 | −5.06 | jpeg_quant |
| 1  | +9.07 | **+11.58** | −6.89 | +7.29 | −9.63 | −9.36 | +1.07 | freq_radial |
| 2  | +9.43 | **+10.77** | +6.56 | +5.60 | −9.94 | **−10.51** | −1.97 | freq_radial (texture #2) |

### Reading (honest, locked-thesis-consistent)

1. **The compression axis dominates in ALL THREE seeds.** In each seed the top-2 zero-outs are
   exactly the correlated pair `{jpeg_quant, freq_radial}` (A5: r = −0.80), and they dwarf
   the other four concepts (all |Δ| < 1 pp in all seeds, texture_geometry ≤7.8 pp). The
   detector's *functional reliance on a compression signal* is robust and reproducible. This
   is the stable finding.

2. **Which named concept "carries" that reliance is seed-dependent.** seed-42 loads it on
   `jpeg_quant` (zero-out −49.5 pp, |w| rank-1); seed-1 loads it on `freq_radial` (zero-out
   −38.7 pp, |w| rank-1) with `jpeg_quant` secondary (−24.5 pp); seed-2 splits nearly evenly
   (freq_radial −24.7 pp, jpeg_quant −25.7 pp, texture_geometry now −7.8 pp with |w| rank-2).
   The two −0.80-correlated
   channels trade off; the linear head is free to distribute weight between them.

3. **This STRENGTHENS the locked thesis, it does not threaten it:**
   - **"intervention > weights" (D-31):** weight rank AND single-concept zero-out rank are
     both seed-unstable; only the *pair-level* compression reliance is stable. A single
     concept's number is not a faithful readout.
   - **R-5 "functional reliance ≠ semantic faithfulness":** the instrument reliably reveals
     *that* the model leans on compression artifacts; it does NOT reliably assign that
     reliance to one fixed human-named concept. Direct empirical support for the prohibited
     claim "jpeg_quant = generative semantics" staying prohibited.
   - keep-only-one (secondary; bias-calibration-sensitive, treat as illustrative): seed-42
     freq_radial-alone ≈ 90%, jpeg_quant-alone ≈ chance; seed-1 NO concept alone clears ~53%;
     seed-2 bitplane_lsb-alone ≈ 95% (!), freq_radial ≈ 60%, jpeg_quant ≈ chance.
     Standalone separability is wildly seed-dependent → reinforces (2).

4. **Manuscript impact (for the Stage-4 revision, NOT yet applied):**
   - The headline that currently names `jpeg_quant` specifically as THE relied-on concept
     must be reframed to **the compression pair `{jpeg_quant, freq_radial}`**, with an explicit
     seed-robustness sentence: the *axis* is stable, the *channel label* is not.
   - **E-Delta Outcome C** ("jpeg_quant survives Q96 debiasing, zero-out −48 pp") is **single-seed
     (seed-42 lineage)**; it is consistent with seed-42 where jpeg_quant dominates, but must be
     stated as seed-42-specific, cross-referencing this multi-seed caveat. Do NOT generalize
     E-Delta's specific-concept result across seeds.
   - Correct the stale "freq_radial zero-out ≈ −2 pp" approximation (hardcoded in
     `edelta_gate3_full_audit.py` chart) → real identically-computed value is −17.2 pp (seed-42).

Artifacts: `Code/Results/full_inference_dump_seed1.npz`, `Code/Results/seed1_analysis/`,
`Code/Results/seed42_analysis/` (recomputed), `Logs/cbnet_multigen_seed1/`.

---

## R-1 — No-bottleneck baseline (resolves M2) — DONE ✓

`BaselinePlain` (backbone→GAP→Linear(2560→1), 28.2M, no concept bottleneck) via
`cbnet_agid/train_baseline.py` (seed 42, 20 ep, same loaders/LRs/AMP/aug as Route B).

### Detection accuracy — matches CBNet ✓

| Model | SD-1.4 | GLIDE | Wukong | VQDM | OOD mean |
|-------|--------|-------|--------|------|----------|
| CBNet-AGID (3 seeds) | 99.85 | 99.80–99.90 | 99.45–99.65 | 99.75–99.90 | **99.67–99.82** |
| BaselinePlain | 99.95 | 99.90 | 99.00 | 99.90 | **99.60** |

→ **The bottleneck does not cost detection accuracy.** Baseline OOD 99.60% is within
~0.1 pp of the CBNet range. The concept bottleneck is accuracy-neutral.

### B1 confound (JPEG-q95 re-encoding) — both models equally vulnerable

Under JPEG-q95 re-encoding (the compression confound from Batch 2), fake_acc collapses
on harder generators for **both** models:

| Generator | CBNet fake_acc | Baseline fake_acc | CBNet acc | Baseline acc |
|-----------|---------------:|------------------:|----------:|-------------:|
| SD-1.4 | 89.1% | 86.5% | 94.35 | 93.20 |
| BigGAN | 61.5% | 55.4% | 80.45 | 77.65 |
| ADM | 72.1% | 68.9% | 85.90 | 84.25 |
| Midjourney | 90.7% | 88.3% | 95.25 | 93.90 |
| **mean** | **78.4%** | **74.8%** | **88.99** | **87.25** |

→ **Both models rely on the same compression shortcut.** Removing the bottleneck does
NOT remove the confound reliance — the plain model is *slightly more* vulnerable
(−12.6 pp mean acc vs CBNet's −10.9 pp). The bottleneck is not the source of the
shortcut; it's in the backbone + training data.

### What the bottleneck adds (summary)

The bottleneck provides a structured 6-dim concept vector for **closed-form audit**
(zero-out, swap, weight reading) at **zero accuracy cost** and **no additional confound
vulnerability**. Without it, the same backbone exploits the same shortcut, but you can
only audit it via opaque gradient-based attribution (post-hoc, harder to reproduce).
This resolves M2 ("what does the bottleneck add?") and the Devil's Advocate's
"single-feature ablation in a six-concept costume" critique: the costume isn't the
problem — the backbone is — and the costume is how you can see it.

Artifacts: `Logs/cbnet_baseline_nobottleneck_s42/`, `Results/ood_eval_baseline_nobottleneck.json`,
`Results/confound/baseline_nobottleneck_jpeg-q95.json`, `Code/tmp_eval_baseline_ood.py`,
`Code/tmp_conf_baseline_b1.py`, `Code/cbnet_agid/models/baseline_plain.py`,
`Code/cbnet_agid/train_baseline.py`.

---

## P0-#1 — Post-hoc attribution head-to-head (resolves Stage-3' MAJOR "audit value asserted not demonstrated")

Peer-review (2026-06-07, full 5-reviewer) asked: does the bottleneck reveal compression reliance that
post-hoc attribution on the plain backbone would miss, or does attribution recover it just as well?
Direct test (`scripts/p0_attribution_headtohead.py`, CPU): on 1680 eval images (240/gen × 7 gens,
70/30 train/test split) fit a ridge probe from the **no-bottleneck baseline's** 2560-d pooled features
to each of the 6 dataset-level concept heuristics, then ablate the probe direction `v̂` and re-run the
classifier (`logit' = logit − (f·v̂)(w·v̂)`).

| concept (probe target) | held-out probe R² | cos(v̂, ŵ) | Δacc (pp) |
|------------------------|------------------:|-----------:|----------:|
| bitplane_lsb     | 0.81 | −0.009 | 0.0 |
| freq_radial      | 0.61 | −0.020 | 0.0 |
| color_manifold   | (degenerate 1.0 — prior not loaded; excluded) | 0.000 | 0.0 |
| hf_noise         | 0.76 | +0.019 | 0.0 |
| **jpeg_quant**   | **0.88** | +0.003 | 0.0 |
| texture_geometry | 0.35 | −0.003 | 0.0 |
| random direction (control) | — | ≈0.02 | 0.0 ± 0.0 |

Baseline test-split acc = 100% (overall 99.88%). **Reading:** the forensic statistics ARE linearly
decodable from the plain backbone (jpeg_quant R²=0.88) → the evidence the bottleneck names is *latent*
in the backbone, consistent with R-1 (baseline equally confound-susceptible). BUT the decoded
compression direction is ~orthogonal to the decision axis (|cos|≈0.003 vs ≈0.02 random) and the
baseline saturates → no single-direction ablation (compression or random) moves accuracy. The
bottleneck's causal `jpeg_quant` zero-out (−49.5 pp) has no apples-to-apples analog → **reframe
(reviewer-endorsed): the bottleneck supplies a built-in, named, causally-ablatable LOCUS, not a cue the
backbone lacks.** This is the *interpretive, not predictive* value claimed. Written into §4.5 + App G
(Table 17).

Artifacts: `Results/p0_attribution_headtohead.json`, `Code/scripts/p0_attribution_headtohead.py`.

---

## P0-#2 — Multi-seed Outcome C (resolves Stage-3' MAJOR "Outcome C single-seed n=1")

Retrained the Q96-debiased model (E-Delta) on 2 further seeds, identical recipe to `edelta_full`
(`Logs/edelta_full/config.json`) except `--seed`. **Training interrupted by force majeure 2026-06-07/08:**
seed-1 reached **epoch 10 (complete)**; seed-2 reached **epoch 6 (partial; plateaued ~0.97 mean-acc)**.
Launched detached via `Code/run_edelta_seeds.ps1` (Start-Process, survives session interrupts);
harness-tracked bg jobs were observed to die on session interrupt/model-switch. Audited both with the
**identical** `scripts/edelta_gate3_full_audit.py` that produced the seed-42 number, on Q96-val.

| seed (epochs) | jpeg_quant w | freq_radial w | jpeg_quant zero-out | freq_radial zero-out | rank-1 zero-out | Q96-val mean | raw-OOD mean |
|---------------|-------------:|--------------:|--------------------:|---------------------:|-----------------|-------------:|-------------:|
| 42 (10) | −9.11 | +6.37 | **−48.18** | −0.90  | jpeg_quant  | 99.78 | 95.00 |
| 1  (10) | −6.74 | +7.34 | −8.10      | **−14.15** | freq_radial | 99.70 | 93.20 |
| 2  (6)  | −6.36 | +6.00 | **−7.70**  | −1.62  | jpeg_quant  | 99.67 | 96.27 |

**Reading (honest, moderates the strong single-seed claim):** in EVERY seed a compression-pair channel
`{jpeg_quant, freq_radial}` is the rank-1 single-channel zero-out on the debiased model → data-level
debiasing does **not eliminate** compression-axis reliance (qualitative Outcome C holds). BUT the
magnitude is seed-dependent and seed-42 (−48 pp) is the **strong end**: seeds 1–2 distribute the cue
and the dominant single-channel cost falls to 8–14 pp. So the robust claim is qualitative (axis persists,
not reducible to container label), NOT the specific −48 pp. Mirrors the carrier seed-dependence of the
raw model (R-2 / §4.3a). Detection stays competitive on all 3 debiased seeds. Updated 6 manuscript
spots single-seed → multi-seed + new App-F Table 13.

Artifacts: `Logs/edelta_seed1/ckpt_epoch10.pth`, `Logs/edelta_seed2/ckpt_epoch6.pth` (+ config/history/
train.log each), `Results/edelta_seed1_audit/gate3_full_audit.json`,
`Results/edelta_seed2_audit/gate3_full_audit.json`, `Code/run_edelta_seeds.ps1`.
(Intermediate epoch checkpoints purged 2026-06-08 to free ~6 GB; audited finals retained.)

---

## P0-#3 — Protocol C/D ablation expansion (matrix breadth without misreporting)

User asked to enlarge the ablation matrix while keeping the actual experiment budget small. Integrity
boundary locked: **do not write this as a full retraining matrix**. Final design:

### Protocol C — frozen causal/interventional matrix (main added ablation evidence)

Generated `scripts/protocol_c_matrix.py` from existing frozen bottleneck dumps and audit outputs.
Rows = **2812**:

| family | rows |
|--------|-----:|
| drop | 984 |
| keep_only | 984 |
| impute_single | 432 |
| counterfactual_swap | 252 |
| edelta_single_drop | 90 |
| jpeg_quality_sweep | 35 |
| resolution_sweep | 35 |

Single-drop reliance summary: 18 rows, compression-pair share
`{jpeg_quant, freq_radial}` = **91.95%** of positive single-drop reliance. Claim scope =
`frozen_bottleneck_closed_form` / `audit_intervention_not_retraining`.

Artifacts: `Code/scripts/protocol_c_matrix.py`, `Code/tests/test_protocol_c_matrix.py`,
`Code/Results/protocol_c_matrix/protocol_c_frozen_audit_matrix.csv`,
`Code/Results/protocol_c_matrix/protocol_c_summary.json`,
`Code/Results/protocol_c_matrix/protocol_c_summary.md`.

### Protocol D — mini training design-sensitivity check (fallback/sanity only)

Ran 3 actual mini-trainings on `train_25k`, 2000 images/class/generator, 5 epochs, seed 42:
full loss, `no_pair`, `no_sparsity`. Detection sanity on SD-1.4 mini-val passes:

| variant | SD acc | AUC | top zero-out | compression share |
|---------|-------:|----:|--------------|------------------:|
| full | 98.70% | 0.999356 | jpeg_quant (2.20 pp) | 0.706 |
| no_pair | 98.50% | 0.999264 | color_manifold (0.50 pp) | 0.250 |
| no_sparsity | 98.20% | 0.999348 | jpeg_quant (0.20 pp) | 1.000 |

Reading: D is **not strong enough** to support a full-scale loss-term claim. Accuracy is high, but
zero-out effects are tiny and mechanism ranking is unstable. Use D only as a truthful feasibility /
design-sensitivity sanity check; main expanded ablation evidence remains Protocol C.

Artifacts: `Code/Results/cbnet_multigen_protocol_d_mini_full_s42.json`,
`Code/Results/cbnet_multigen_protocol_d_mini_no_pair_s42.json`,
`Code/Results/cbnet_multigen_protocol_d_mini_no_sparsity_s42.json`,
`Code/scripts/protocol_d_mini_audit.py`, `Code/tests/test_protocol_d_mini_audit.py`,
`Code/Results/protocol_d_mini_audit/protocol_d_mini_design_sensitivity.csv`,
`Code/Results/protocol_d_mini_audit/protocol_d_mini_summary.json`,
`Code/Results/protocol_d_mini_audit/protocol_d_mini_summary.md`.
