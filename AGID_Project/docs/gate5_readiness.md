# Gate #5 Readiness — Pre-Stage 2 Sign-Off Package

**Prepared**: 2026-05-28
**Pipeline state**: **Stage 2 ACTIVE** — Gate #5 unlocked 2026-06-01 (user: 「解锁 Gate #5」). Stage 1.5 fully saturated; Batch 1-4 complete.
**Stage 2 mode (locked)**: academic-paper skill, plan mode (Socratic chapter guidance). In ARS v3.9.3 the trigger command is `/ars-plan`.

---

## 1. Evidence inventory (what is now in hand)

### Training-stage evidence
- **Route A (single-gen baseline)**: SD-1.4 train_100k, 8 epochs available, final SD-1.4 val acc 99.83%
- **Route B (multi-gen)**: SD-1.4 + BigGAN + ADM + Midjourney @ 25k/class, 20 epochs, ckpt_epoch20.pth used for all eval
  - Methodological fixes applied pre-training: Fix #1 (--disable_destructive_aug), Fix #2 (shared concept norm), Fix #3 (color prior bug + multi-source)
- **Held-out eval (training generators)**: 99.85–99.95% per generator
- **OOD eval (3 unseen generators)**: GLIDE 99.80% / Wukong 99.45% / VQDM 99.75% → mean 99.67%, AUC=1.000

### Audit-stage evidence (Stage 1.5, Batches 1-3)

| Audit dimension | Headline finding |
|---|---|
| Concept weight \|w\| (A2) | `jpeg_quant` w=-13.53 dominant (1.4× next) |
| Concept activation distribution (A1) | Real images: jpeg_quant ≈ 0.72; fakes: ≈ 0.23 (clean separation) |
| Concept correlation (A5) | `jpeg_quant ↔ freq_radial` Pearson = -0.80 (high redundancy) |
| Zero-out ablation (A3) | Removing `jpeg_quant` collapses 99.75% → 50.21% on every generator |
| Keep-only-one (A3) | `freq_radial` alone 82-99.65%; `jpeg_quant` alone 50% (bias-locked) |
| Counterfactual swap (A4) | `jpeg_quant` causes 24-72% prediction flips; other concepts <2% |
| Format leakage (B1) | jpeg-q95 unified re-encoding → mean Δacc -10.16 pp (fake_acc collapses) |
| Resolution leakage (B2) | res128 down-up sample → real_acc 99.7% → 8.1% |
| OOD pool confound (B5) | Independent real samples per OOD gen → +0.12 pp (CONFOUND CLEARED) |
| Calibration (C5) | Overall ECE 0.0034, Brier 0.00189 — well-calibrated |
| Failure rate | 35 errors out of 14,000 (0.25%); Wukong worst with 7 FN |

---

## 2. Paper narrative decision (must confirm with user before Stage 2)

### Recommended pivot: "Concept bottleneck as confound-audit instrument"

The audit evidence makes the original "6 interpretable concepts" claim
indefensible. Recommended Stage 2 framing:

> *We train a 6-concept bottleneck network (CBNet-AGID) on the GenImage benchmark
> and achieve 99.67% mean OOD accuracy across 3 unseen generators. Using the
> model's bottleneck structure as a probe, we then mechanically audit which
> concepts carry the signal and find that ~50% of the discriminative power
> traces to a single concept (`jpeg_quant`) capturing the PNG vs JPEG container
> difference between fake and real images on this benchmark. We argue this
> repositions the concept bottleneck as a **diagnostic instrument for benchmark
> confound auditing** in AGI detection, beyond its role as a classifier
> architecture.*

**Why this pivot is stronger than the original**:
- Same data, higher methodological novelty (C-grade defensible)
- Honest about limitations (aligns with `feedback-novelty-honesty` rule)
- The audit *itself* is the contribution; the 99.67% becomes the **setup** rather than the punchline
- Reviewers cannot pull the "you're just learning JPEG format" criticism — we get there first

### Alternative (NOT recommended): "Cross-generator generalization via concepts"
Original framing. Weak under reviewer scrutiny once B1/B2/A3 are published.

---

## 3. Stage 2 plan-mode entry parameters

When Gate #5 unlocks, the next session should invoke:

```
/ars-plan        (= academic-paper skill, plan mode; ARS v3.9.3)
  paper_type:      empirical research (CS / ML domain)
  citation_format: NeurIPS 2022 natbib  (template at 课程大作业latex模板/)
  word_target:     30000 (compact tier per existing lock)
  novelty_grade:   A+C (per existing lock)
  narrative:       Path A — concept bottleneck as confound-audit instrument (user-confirmed 2026-06-01)
```

The Socratic planning should cover at minimum:
1. Title + abstract draft (pivot-aware)
2. Section outline with evidence map → each claim → which Batch artifact backs it
3. Limitations strategy (B1/B2/A3 framed as **discovered findings**, not weaknesses)
4. Related-work positioning (concept-bottleneck literature + AGID literature + benchmark-audit literature)

---

## 4. Pre-Stage-2 checklist (sign-off items)

- [x] User has read this document
- [x] User has reviewed `Code/Results/batch1_analysis/A_intervention_summary.md`
- [x] User has reviewed `Code/Results/batch1_analysis/B_confound_summary.md`
- [x] User confirms paper narrative pivot to **audit-instrument framing** (Path A, confirmed 2026-06-01)
- [x] Citation format: NeurIPS 2022 natbib (existing course template)
- [x] Target venue: Spring 2026 ML course final paper (graduate-thesis tier)
- [x] User unlocks Gate #5 explicitly — 2026-06-01: 「解锁 Gate #5」

---

## 5. What will NOT happen during Stage 2 (per CLAUDE.md DO NOT)

- No further training runs
- No model architecture changes
- No K≠6 experiments
- No deleting/overwriting `Logs/`, `Results/`, `Checkpoints/`
- No auto-launching academic-pipeline substages without user batch approval

---

## 6. Open paths if user declines the pivot

- Path B: Keep original "6 concepts" framing → must add explicit ablation defending the 4 dormant concepts (no such experiment is currently possible without retraining); novelty ceiling drops to "B grade" honest assessment.
- Path C: Fallback to Route A only — leaves Batch 1-3 audit evidence as future work. Wastes the strongest finding.

Recommendation: **Path A (audit pivot)** preserves all current evidence and is the only path that uses Batch 3 findings as a contribution rather than as Limitations.
