# Stage 2 Chapter Plan & Status — AGID Paper

**Updated:** 2026-06-02 · IMRaD · 30k words · NeurIPS 2022 natbib
**Companion files:** `00_Evidence_Spine_Lock.md` (🔒 evidence contract), `01_Decision_Log.md` (decisions)

---

## Chapter status tracker

| # | Chapter | Status | Notes |
|---|---|---|---|
| 1 | Abstract | **DRAFTED ✓ (EN only)** | `draft/08_Abstract.md` (~260w) — leads with 99.67% held-out + immediate audit finding; competitive-not-SOTA; intervention>weights; partial+partial |
| 2 | Introduction | **DRAFTED ✓** | `draft/01_Introduction.md` (D-01′…D-07; RQ wording confirmed; numbers trimmed to 99.67/73.65/−48pp) |
| 3 | Related Work | **DRAFTED ✓** | `draft/02_RelatedWork.md` (3-line β arc; AIGI-Holmes foil; Grommelt closest prior) |
| 4 | Method | **DRAFTED ✓** | `draft/03_Method.md` — as-built (D-17…D-24); 7 subsections incl. audit-protocol; standard backbone, no-skip guarantee, channels-not-concepts, losses-not-claimed, single-phase |
| 5 | Experiments | **DRAFTED ✓** | `draft/04_Experiments.md` — §1-4, all numbers artifact-verified; 3-way ForenSynths (A 52.21 / B 73.65 / NPR 79.58) re-run + populated; freq_radial = secondary-on-RouteB / inert-on-EDelta stated precisely |
| 6 | Discussion | **DRAFTED ✓** | `draft/05_Discussion.md` — 8 paras per 口径; interpretation-only, no new claims; intervention>weights, two-channel, Outcome C, ForenSynths(+NPR honest), regularizers-didn't-prevent-collapse, extend-Grommelt, testability-surface |
| 7 | Limitations | **DRAFTED ✓** | `draft/06_Limitations.md` — 8 items honesty-ledger: protocol comparability, unrun ablations, ForenSynths shift-entanglement, Q96 ceiling, supervision-induced identity, GenImage/ImageNet scope + resolution/container, smoke-vs-full, crop/label |
| 8 | Conclusion | **DRAFTED ✓** | `draft/07_Conclusion.md` — 3 paras; detector=subject, contribution=audit surface+intervention, partial+partial, intervention>weights, extend-Grommelt, future work; no new claims/numbers |

---

## Related Work — reconstructed plan (D-06) — to advance next

**Progressive 3-"line" structure** (not a flat survey). Each line escalates toward the gap:

- **Line 1 — Detection SOTA / artifact detectors [1b-restrained · LOCKED 2026-06-02]:** present SOTA normally —
  NPR (~96–98%), LOTA (98.9%), AIGI-Holmes (99.2%), our Route B (99.67%) → near-saturation on GenImage. Then
  **one** restrained forward-pointer (no critique expansion, no weakening of the flex): *"these near-saturated
  GenImage numbers may reflect not only generative-trace detection but also dataset shortcuts — a concern raised
  by Grommelt and tested model-internally in our experiments."* Rhythm: saturate first → flag that the
  *measurement validity* of saturation is unresolved at the model level.
- **Line 2 — Concept-bottleneck / interpretable models [β audit-instrument · LOCKED 2026-06-02]:**
  `koh2020concept` (CBM seed) → faithfulness-critique trio `margeloiu2021concept` / `mahinpei2021promises` /
  `havasi2022addressing` (**scaffolding**) → **AIGI-Holmes (`zhou2025aigiholmes`) = SOLE headline foil** at the
  un-falsifiable extreme. Claim = the bottleneck is an *interventionable, falsifiable audit surface*; we do NOT
  claim native faithfulness (α rejected).
- **Line 3 — Dataset-confound work [LOCKED 2026-06-02]:** **Grommelt 2024 (`grommelt2024fakejpeg`)** = closest
  prior — credit **fully**. Delta fixed at claim level: *black-box/dataset-level analysis → model-internal
  localization + causal intervention + Q96-persistence (Outcome C)*. Numbers → Experiments. Do **not** raise the
  double-dissociation / "+11pp" tension here (reserved for Experiments/Discussion). RW positions only.

→ converges on **D-05 gap**: "lacks an architecture-level audit mechanism that can intervene on
separately-testable evidence channels."

**RW dialogue status: ✅ COMPLETE — all 3 lines LOCKED** (D-13 / D-15 / D-16). Arc: SOTA + forward-pointer →
β audit-instrument (AIGI-Holmes foil) → Grommelt closest-prior → D-05 gap. **Next chapter to plan: Method.**

---

## Evidence → chapter mapping (which artifact feeds which chapter)

| Evidence artifact | Feeds |
|---|---|
| `Results/edelta/gate3_audit/gate3_full_audit.json` | Experiments (double dissociation + audit), Discussion |
| `Results/batch1_analysis/A_intervention_summary.md` (A3/A4/A5) | Experiments (audit battery), Method |
| `Results/batch1_analysis/B_confound_summary.md` (B1/B2/B5) | Experiments, Limitations |
| `Results/batch1_analysis/E1_sota_baselines.md` | Related Work Line 1, Experiments |
| `Results/batch1_analysis/F3_main_results.md` | Experiments (headline detection) |
| `Stage1_Research/03_Bibliography.md` (91 refs) | Related Work, citations throughout |
| `docs/EDelta_experiment_plan.md` | Method (Q96 retrain protocol) — read when drafting Method |

---

## Experiments — gating TODO

- **A-3′ ✅ RESOLVED (D-25):** loss-component ablation was **NOT run** (6-run inventory). Consistency-loss
  contribution **DISABLED**; `content_pair` = as-implemented detail only.
- **A-4 ✅ RESOLVED (D-26):** 6-run inventory → only Route A/B/E-Delta, all K=6 + full dual-stream + bottleneck.
  EXIST: per-concept A3, confound sweeps B1/B2/B5, data ablations (Route A vs B; 20k vs 100k), E-Delta debiasing.
  **NOT run** (→ Limitations, not claims): backbone, K∈{4,8,12}, no-bottleneck, loss-component.
  ⚠️ verify Route A↔B OOD benchmark cleanliness (`D3_route_a_vs_b.md`).

---

## Back-matter chapters — PLAN LOCKED (2026-06-02, user 口径)

### Discussion (§6)
Main line = **partial confound inflation + genuine partial generalization** (NOT pure confound; NOT
"robust generalization solved").
- Core audit finding: functional reliance concentrates on `jpeg_quant` (zero-out −48pp); **weights ≠ reliance**
  (intervention is the authority).
- Confound: clean = B1/B2/A3/E-Delta; corroborating = ForenSynths −26pp (with dist-shift caveat);
  Q96-persistence (Outcome C) ⇒ `jpeg_quant` not reducible to the final container label.
- Generalization: ForenSynths 73.65% > single-gen 52.21% ⇒ multi-gen buys genuine partial cross-arch
  generalization; the model is NOT a pure container detector.
- Interpretation (marked as such): anti-shortcut regularizers didn't prevent collapse — sparsity-entropy targets
  activation *diversity* not functional *reliance*; a generator-invariant confound trivially satisfies the
  consistency loss.
- Grommelt: we **extend** (model-internal + intervention + Q96-persistence).
- Methodological: supervision-induced concept identity ⇒ names aren't auto-faithful ⇒ causal audit required;
  CBM = testability surface.

### Limitations (§7)
- **Protocol comparability** (E1): P4 vs P1; CBNet's OOD set ≠ baselines'; no clean SOTA comparison; Route A↔B
  confounded by code/loss/scale.
- **Unrun ablations:** loss-component, backbone (dual/pixel/signal), K∈{4,8,12}, bottleneck-vs-no-bottleneck ⇒
  cannot empirically justify K=6, the bottleneck's necessity, or `content_pair`'s effect → future work.
- **ForenSynths −26pp** entangles confound + GAN-vs-diffusion distribution shift (cannot cleanly attribute).
- **Q96 debiasing** removes only the final-container label; cannot isolate generative-trace vs
  residual-compression-history (spine §3 guard / L-a).
- spine **L-b** (smoke→full = overfit artifact; only full GATE 3 stands), **L-c** (8× resolution span +
  PNG/JPEG leakage = dataset-level, unfixable in-model).
- Concept identity supervision-induced → we do NOT claim 6 individually-faithful concepts.
- Single training dataset family (GenImage); real = ImageNet only. crop↔label-center approximation (Method footnote, D-24).

### Conclusion (§8)
- The **competitive single-GPU detector is the audit SUBJECT**, not the headline contribution.
- True contribution = **architecture-level audit surface + intervention evidence** (model-internal localization
  of the GenImage confound; Q96-persistence) — closes the D-05 gap.
- Honest scope: competitive (not SOTA); partial generalization + quantified confound reliance.
- Future work: the unrun ablations; trace-vs-history disentanglement; audit applied to other detectors/datasets.

### Abstract (write LAST, after drafting)
- **NO hard SOTA.** Lead with "**competitive / near-saturated** GenImage performance" (labelled OOD/P4) →
  **immediately** the audit finding (`jpeg_quant` carries ~50pp; persists under Q96; partial confound inflation).
  D-02′ / D-31.
- Structure: context (GenImage saturation) → CBM-as-audit-instrument on a competitive single-GPU detector →
  key finding → contribution (audit surface + intervention).
- Bilingual EN/ZH per ARS abstract mode (confirm at Abstract time).

---

*Update the status tracker as each chapter advances.*
