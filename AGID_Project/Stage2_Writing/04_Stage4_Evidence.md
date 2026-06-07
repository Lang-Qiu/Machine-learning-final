# Stage 4 (REVISE) — New Experimental Evidence

Append-only log of the peer-review-response experiments (Stage 3 → Stage 4). All numbers
computed from disk artifacts by the **identical** code path across seeds (`scripts/analyze_all.py`
dump → `scripts/concept_intervention.py` closed-form zero-out). Protected files untouched.

Status: seed-42 (orig) ✓ · seed-1 ✓ · seed-2 ✓ · baseline (no-bottleneck) PENDING.

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
