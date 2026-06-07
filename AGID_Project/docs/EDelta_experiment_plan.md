# E-Delta Experiment Plan — Concept Reliance Migration under Grommelt Debiasing

## Material Passport

```yaml
artifact_id:        CBNet-AGID-EDELTA-PLAN-v1
artifact_type:      code_experiment_plan
producer:           academic-paper (plan mode) Stage-2 excursion
producer_session:   2026-06-01
status:             APPROVED 2026-06-01 (GATE 0 ✓). GATE 1+2 authorized; GATE 3 (full retrain) pending smoke result.
provenance:
  derived_from:
    - memory/project-agid-grommelt-scoop
    - Stage1_Research/07_Experiment_Plan.md   (format + Route B config)
    - Grommelt et al. 2024 "Fake or JPEG?" (arXiv:2403.17608)
  user_decisions:
    - debias route 3a (local reproduction; no unbiased-genimage.org download on critical path)
    - fallback date 2026-06-11; smoke gate 2026-06-04
    - GO granularity: spec-first → smoke → full (full NOT pre-authorized)
governs:
  - reopening of experiment phase (Stage 2 writing PAUSED during this excursion)
```

---

## 0. Why this experiment exists (the one-paragraph version)

Grommelt et al. (2024) already showed, at the **dataset level** with black-box detectors (ResNet50/Swin-T), that GenImage has JPEG-compression + image-size biases and that debiasing changes cross-generator performance. That **scoops our audit "discovery."** What remains genuinely un-scooped: **nobody has looked *inside* a model to watch where its competence goes once the container shortcut is removed.** E-Delta retrains our concept-bottleneck detector on Grommelt-style debiased data and uses the bottleneck's linear decomposability to measure **concept reliance migration** — does the model relocate to other (genuinely generative) concept channels, or does accuracy collapse? This is the novel knowledge that re-secures the A+C target (RE-POSITION 3).

---

## 1. Scope

**In scope.** One retrain of CBNet-AGID on locally-reproduced Grommelt-style debiased GenImage (4 training generators), plus the audit battery (A2 weights, A3 ablation, A5 correlation, concept Cohen's d) on the debiased model, compared side-by-side against the existing Route B (biased-data) model.

**Out of scope.** Downloading `unbiased-genimage.org` (3b — optional spot-check only, NOT critical path). New architectures. New concepts (K=6 locked). Any retrain on SD-1.4 single-gen (CLAUDE.md DO NOT #1).

**Hardware.** RTX 4060 Laptop 8GB (same as Route B; proven feasible). bs≤32, bf16 AMP.

---

## 2. Relationship to Grommelt — what is reproduction vs novel (state this explicitly in the paper)

| Element | Grommelt 2024 | E-Delta (ours) | Status |
|---|---|---|---|
| JPEG/size confound exists in GenImage | ✅ first | reproduce locally (debias build) | **reproduction (cite Grommelt)** |
| Debias → cross-gen perf change | ✅ +11pp | will re-observe on CBNet | reproduction |
| **Per-concept reliance INSIDE a model, before vs after debias** | ❌ none | ✅ A2/A3 migration on bottleneck | **NOVEL** |
| **Bottleneck as instrument that tracks reliance across data regimes** | ❌ | ✅ | **NOVEL** |

> Writing rule: E-Delta's contribution sentence must read "*we do not claim to discover the confound (Grommelt 2024 did); we provide the first model-internal account of how a detector's concept reliance migrates when the confound is removed.*"

---

## 3. Hypotheses (falsifiable) and Outcome Map

- **HΔ1 — debias neutralizes the concept at the label level.** On Q=96-debiased data, the `jpeg_quant` heuristic's real-vs-fake separation collapses to **|Cohen's d| < 0.3** (from its strongly-separating value on raw data). *Precondition; if it fails, the debias build is wrong — fix before training.*
- **HΔ2 — reliance migrates or collapses.** A debiased-trained CBNet's classifier weight on `jpeg_quant` loses top rank (|w| no longer largest), and exactly one of:
  - **HΔ2a (migration → Outcome B):** another concept (freq_radial / bitplane_lsb / hf_noise) becomes the new ablation-dominant channel **and** mean OOD acc stays **≥ 75%**.
  - **HΔ2b (collapse → Outcome A):** no concept compensates; mean OOD acc falls **≤ 60%**.
- **HΔ3 — instrument validity.** Per-concept zero-out on the debiased model localizes the new reliance (zeroing the new top concept reproduces the largest Δacc), showing the bottleneck tracks reliance across both data regimes.

| Outcome | Signature | Paper narrative |
|---|---|---|
| **A — collapse** | OOD ≤ 60%, no concept compensates | "Detector competence was substantially the container shortcut; quantifies the hollow-SOTA component from the inside." |
| **B — migration** | OOD ≥ 75%, reliance moves to a generative-trace concept | **richest** — "the instrument watches competence relocate from container artifact to genuine trace." |
| **C — persistence** | OOD high, jpeg_quant still dominant | "our concept captured signal beyond pure format — debias incomplete or concept broader than container." |

All three are publishable → experiment is **novelty-safe regardless of result.**

---

## 4. Debias protocol (3a — local reproduction)

**Primary axis — JPEG Q=96 (the dominant container confound).**
- Re-encode **every** image (real + fake, train + val + OOD) to JPEG **quality 96**, reusing `scripts/confound_sweep.py::_JpegQuality(96)`.
- Mechanism note (state in paper): GenImage reals are already JPEG (modal Q≈96); fakes are PNG (lossless). So this recompression is ~no-op on reals and **adds the JPEG quantization the fakes lacked** → neutralizes the "fakes lack JPEG trace" shortcut. This is exactly the B1/B3 effect, now applied at **train** time.

**Secondary axis — size normalization (optional, to fully match Grommelt; VERIFY protocol).**
- Resize all images to a common resolution before JPEG (candidate: shorter-side→256, bilinear, via `_ResN`-style op). **Grommelt's exact size protocol must be verified against their paper/repo before this axis is included.** Primary smoke + full runs use JPEG-Q96 only; size-norm is a v2 refinement if time permits.
- **Writing caption (LOCKED, user 2026-06-01):** report this as *"Grommelt-style JPEG-axis debias"* — do NOT claim full reproduction of Grommelt's protocol. size-norm stays a v2 optional axis / robustness check.

**Critical: materialize to disk, do NOT debias on-the-fly.**
- Write debiased copies to `Data/GenImage_debiased/<gen>/{train,val}/{ai,nature}/` via a new `scripts/build_debiased_dataset.py` (reuses `_JpegQuality`).
- Reason: concept labels must be computed on the **same pixels the model trains on**. On-the-fly debias would reintroduce the image↔label misalignment that Fix #1 was created to prevent.

---

## 5. Pipeline & script reuse map

| Step | Script | New/Reuse | Notes |
|---|---|---|---|
| 5.1 build debiased data | `scripts/build_debiased_dataset.py` | **NEW (small)** | reuse `_JpegQuality(96)`; staged (smoke subset first) |
| 5.2 shared concept norm | `scripts/build_shared_concept_norm.py` | reuse | across debiased 4 gens (Fix #2) |
| 5.3 precompute concept labels | `cbnet_agid.precompute_concept_labels` | reuse | on debiased data; `--center_crop 256` + shared norm + color prior (Fix #1/#3) |
| 5.4 train | `cbnet_agid.train` | reuse | `--disable_destructive_aug`, seed 42, Route B config; out = `cbnet_debiased` |
| 5.5 inference dump | `scripts/analyze_all.py` | reuse | → `Results/edelta/full_inference_dump_debiased.npz` |
| 5.6 weights/corr (A2/A5) | `scripts/derived_analyses.py` | reuse | on debiased dump |
| 5.7 ablation (A3) | `scripts/concept_intervention.py` | reuse | zero-out on debiased dump |
| 5.8 side-by-side | `scripts/edelta_compare.py` | **NEW (small)** | concept reliance Route-B vs debiased = headline figure |

> **Pre-flight (verify before any run):** confirm exact CLI flag names/signatures of `precompute_concept_labels.py`, `train.py`, `analyze_all.py` against the actual current code (this plan reconstructs them from project notes). Adjust commands if signatures differ.

---

## 6. Staged gating (the GO ladder)

```
GATE 0 — spec approval (USER)                          ← you are here
   │  nothing below runs until you approve this doc
   ▼
GATE 1 — pre-flight label sanity  (~hours, NO training)
   │  build SMALL debiased sample → precompute labels →
   │  check HΔ1: jpeg_quant Cohen's d collapses (|d|<0.3)?
   │  PASS → smoke;  FAIL → debias build bug, fix protocol
   ▼
GATE 2 — SMOKE retrain  (target signal by 2026-06-04)
   │  train cbnet_debiased_smoke on small subset
   │  (8k/class × 4 gen, 6 ep [→8 ep ONLY if signal ambiguous], seed 42)
   │  read A2 weights + quick A3 ablation
   │  GO  = jpeg_quant loses top-|w| OR its ablation Δacc
   │         drops from ~50pp to <15pp (regime CHANGED)
   │  NO-GO (per user gate) = signal unclear/unstable by 6/04
   │         → do NOT blindly full-train; debug or fall back
   ▼
GATE 3 — FULL retrain + full audit  (target by 2026-06-08)
   │  full debiased data, Route B config, seed 42
   │  full A2/A3/A5 + Cohen's d + side-by-side figure
   ▼
FALLBACK — if model + audit NOT done by 2026-06-11 23:59
           → FREEZE E-Delta, revert to RE-POSITION 1
             (mechanism-first, zero new compute, still honest)
```

Each gate requires explicit user GO (DO NOT #8). Smoke is authorized only after GATE 0; full only after GATE 2 passes.

---

## 7. Measurements & analysis

Compare **cbnet_debiased** vs **cbnet_RouteB** (existing, biased-data control — its audit already in `batch1_analysis/`):

1. **Detection:** mean acc/AUC, per-gen, in-dist + 3 OOD (on debiased test). Δ vs Route B's 99.67%.
2. **Concept weight migration:** classifier `w_k` per concept, both models, side-by-side bar. Does `jpeg_quant` |w| (Route B −13.53) shrink? who rises?
3. **Ablation reliance (new A3):** per-concept zero-out Δacc on debiased model. Who now carries the ~discrimination?
4. **Concept separation:** per-concept Cohen's d real-vs-fake on debiased data (esp. jpeg_quant → expect collapse).
5. **Headline figure:** `edelta_compare.py` → concept reliance Route-B vs debiased, the visual story of migration/collapse.

---

## 8. Decision rules / contingencies

| Scenario | Rule |
|---|---|
| HΔ1 fails (jpeg_quant d does NOT collapse on Q=96 data) | Debias build bug OR concept broader than format. Inspect `build_debiased_dataset` output; verify fakes actually gained JPEG trace. Do not train until resolved. |
| Smoke unclear by 2026-06-04 | Per user gate: do NOT full-train. One debug iteration (more smoke epochs / larger subset); if still unclear → fall back RE-POSITION 1. |
| Training diverges / OOM | Route B remedies (07 §7): lr 5× down; bs→16; grad checkpointing. |
| Full not done by 2026-06-11 | **FALLBACK to RE-POSITION 1** (locked). |
| Outcome C (jpeg_quant persists) | Still publishable; reframe as "concept captures beyond-format signal" + check debias completeness as limitation. |

---

## 9. Comparability controls

- **Only the training data changes.** Same architecture (CBNetAGID, K=6, dual-stream), same seed (42), same eval protocol/sample lists as Route B / `analyze_all.py`, same audit scripts. This isolates the debias as the sole cause of any reliance shift.
- Route B model + its `full_inference_dump.npz` are the frozen control (CLAUDE.md DO NOT #3 — not overwritten; E-Delta writes to new `Results/edelta/` + `Logs/cbnet_debiased_*`).

---

## 10. Risk register

| # | Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|---|
| 1 | Debias build disk cost (~40GB+ for full) | Low | Low | E: ~1TB free; materialize smoke subset first |
| 2 | Smoke too short to read weights | Med | Med | weights readable even at high loss; if not, +epochs once |
| 3 | jpeg_quant d doesn't collapse (HΔ1 fail) | Low | High | pre-flight catches it before training; verify fakes gain JPEG trace |
| 4 | Schedule slip past 6/11 | Med | Med | hard fallback to RE-POSITION 1 (no compute) |
| 5 | Grommelt size-protocol mismatch | Med | Low | JPEG-Q96 primary; size-norm deferred + verify-before-use |
| 6 | Script signature drift vs this plan | Med | Low | pre-flight signature check (§5) |

---

## 11. Schedule (today 2026-06-01, deadline 2026-06-24)

| Phase | Target | Compute |
|---|---|---|
| GATE 1 pre-flight label sanity | on GO, ~hours | none (label calc) |
| GATE 2 smoke | by **2026-06-04** | ~few hours train |
| GATE 3 full + audit | by **2026-06-08** | ~1–2 days train + fast audit |
| FALLBACK trigger | **2026-06-11 23:59** | — |
| Resume Stage 2 writing | post-experiment | — |

~13 days remain for writing 30k after a 6/08 finish — feasible.

---

## 12. Governance & GO gates

1. **No compute runs without explicit user GO.** This doc is GATE 0.
2. Stage 2 writing is **PAUSED** during the excursion (CLAUDE.md state lock updated 2026-06-01).
3. E-Delta is forensic+training on **debiased multi-gen** data — NOT the SD-1.4 single-gen retrain barred by DO NOT #1; clear information gain (un-scooped finding).
4. Writes only to new paths (`Data/GenImage_debiased/`, `Results/edelta/`, `Logs/cbnet_debiased_*`); never overwrites Route A/B artifacts (DO NOT #3).

---

## 13. Outputs (handoff to writing)

```
Data/GenImage_debiased/<gen>/{train,val}/{ai,nature}/   — debiased images
Results/edelta/
├── full_inference_dump_debiased.npz
├── A2_weights_debiased.json
├── A3_zeroout_debiased.csv
├── A5_correlation_debiased.csv
├── cohens_d_debiased.csv
└── edelta_compare_reliance.png         — headline migration figure
Logs/cbnet_debiased_smoke/ , Logs/cbnet_debiased_full/
```

Feeds: Related Work (Grommelt integration), a new Results subsection ("Concept reliance under debiasing"), and Discussion (what the migration/collapse means for benchmark validity).

---

## 14. Open items — verify before running

- [ ] Confirm CLI signatures of `precompute_concept_labels.py` / `train.py` / `analyze_all.py` vs current code (§5 pre-flight).
- [ ] Decide size-norm inclusion after checking Grommelt's exact protocol (§4 secondary).
- [ ] Confirm Route B exact train config (epochs/lr) to mirror (read `Logs/cbnet_multigen_main_25k_s42/` config).
- [ ] Smoke subset size + epoch count final pick at GATE 1.

---

**END OF E-DELTA PLAN (DRAFT).** No compute authorized. Awaiting GATE 0 approval.
```
