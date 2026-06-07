# Code Experiment Plan — CBNet-AGID

## Material Passport

```yaml
artifact_id:        CBNet-AGID-EXP-PLAN-v1
artifact_type:      code_experiment_plan
producer:           experiment-agent (plan mode)
producer_version:   v1.1.0
producer_session:   2026-05-25
status:             FINAL
content_hash:       (computed at file save)
provenance:
  derived_from:
    - Stage1_Research/01_RQ_Brief.md
    - Stage1_Research/02_Methodology_Blueprint.md (revised)
    - Stage1_Research/06_Day5_Concept_Sanity_Report.md
  user_decisions: (logged below in §8)
reusable_by:
  - cbnet_agid pipeline (executable as written)
  - ARS Stage 2 WRITE (provides Experiments § structure)
target_skill_chain: experiment-agent → ARS Stage 2 academic-paper (after results in)
```

---

## 1. Title and Scope

**Working title.** CBNet-AGID experiments — Phase A (implementation validation) + Phase B (full ablation matrix + baseline comparison + OOD generalization).

**Scope.** All training and evaluation experiments needed to produce the result tables, figures, and ablations for the 30,000-word paper. Excluded: human evaluation, ethics-board review (no human subjects), real-world deployment timing.

**Wall-clock budget.** 14 days (Phase A 6 days + Phase B 8 days), targeting completion by 2026-06-07. Subsequent ~17 days reserved for Stage 2 WRITE through Stage 6.

**Hardware budget.** RTX 4060 Laptop GPU, 8GB VRAM (fixed; no cloud GPU allowance).

---

## 2. Research Questions and Hypotheses (from Stage 1, locked)

**Main RQ.** How can Concept-Bottleneck-style architectural interpretability be integrated into AGID such that it delivers structural explanations AND cross-generator generalization AND remains single-GPU feasible?

**3 sub-RQs.**
- SubRQ-1: What concept set is semantic-meaningful + heuristic-supervisable + cross-generator-stable?
- SubRQ-2: How to insert CBL in backbone with what training procedure?
- SubRQ-3: What loss formulation enforces concept invariance while preserving discriminative power?

**4 hypotheses (falsifiable).**
- **H1** — Bottleneck-induced ID accuracy drop ≤ 2pp vs baseline.
- **H2** — Bottleneck improves OOD accuracy by ≥ +3pp vs baseline. **Failure response:** §7 decision rule.
- **H3** — Concept activation maps are more faithful (lower deletion-AUC, higher insertion-AUC) than Grad-CAM on baseline.
- **H4** — Each concept contributes ≥ 0.5pp via single-drop ablation.

---

## 3. Variables

| Type | Variable | Levels |
|---|---|---|
| IV | training generator | SD-1.4 (single generator, per AGID convention) |
| IV | model | CBNet-AGID (ours) + 6 baselines |
| IV (ablation) | CBL on/off | binary |
| IV (ablation) | concept K | {4, 6, 8} (4 = anchor only; 6 = full set; 8 = +2 reserve) |
| IV (ablation) | each concept inclusion | drop 1 of 6 → 6 levels |
| IV (ablation) | loss component | drop L_concept, L_gen, L_sparsity individually |
| IV (ablation) | backbone | pixel-only / signal-only / dual |
| DV | ID accuracy on SD-1.4 val | scalar % |
| DV | per-generator OOD accuracy on 7 GenImage + 11 ForenSynths gens | scalar % per gen |
| DV | macro avg + size-weighted avg OOD acc | scalar |
| DV | AUC, AP | scalar |
| DV | faithfulness (Del-AUC, Ins-AUC, TCAV, Pointing Game) | scalar |
| Control | bs=32, image=256, AdamW, AMP-bf16, epochs=20, weight decay 1e-4 | fixed |
| Confound | dataset compression (JPEG/WebP), generator era, ImageNet pretraining | flagged in Discussion |

---

## 4. Design

**Type.** Controlled comparative experimental design + factorial ablation.

**Training protocol.** Multi-stage:
- **Stage 1 (debug)** — 20k SD-1.4 subset × 10 epochs × 1 seed × 1 config to verify training loop. ~2.5 hours.
- **Stage 2 (full)** — 100k SD-1.4 subset × 20 epochs × **3 seeds** × all ablation rows. Per user's decision: rigor over speed.
- (Optional stage 3) — Full 324k × 20 epochs × 1 seed on chosen best config, as appendix "scale validation".

**Evaluation protocol.** All trained models evaluated on:
- 8 GenImage val sets (SD-1.4 in-distribution + 7 OOD)
- 11 ForenSynths val sets (cross-architecture OOD)
- Subset of 200 manually-annotated images for Pointing Game (manual ground truth heatmap, deferred to Phase B end)

---

## 5. Datasets and Sample

| Set | Source | Size | Purpose |
|---|---|---|---|
| Train | GenImage / Stable_Diffusion_v1.4 / train / {ai, nature} | 100k subsample (50k AI + 50k real) of ~324k available | Main training |
| ID val | GenImage / Stable_Diffusion_v1.4 / val | ~6k AI + ~6k real | In-distribution |
| OOD val × 7 | GenImage / {SD-1.5, BigGAN, Midjourney, Wukong, ADM, GLIDE, VQDM} / val | varies, ~6k each | Cross-generator |
| OOD val × 11 | ForenSynths / test | ~140k total | Cross-architecture |
| Faithfulness sample | 200 images held-out from GenImage val | 200 | Pointing Game ground truth |

**Subsampling.** Random with seed 42; documented in `Code/scripts/subset_sd14_train.py` (to be written before Phase A-4 begins).

**Real-image source for OOD coverage.** GenImage uses ImageNet val as the "nature" source across all generators, providing a clean control.

---

## 6. Analysis Strategy

| Output | Method |
|---|---|
| Main result table (CBNet vs 6 baselines × 19 generators) | Mean + Bootstrap **95% CI** over 3 seeds (1000 resamples) |
| OOD aggregate (across 19 generators) | Report **both macro-avg and sample-size-weighted-avg** |
| Ablation table (each row vs full CBNet) | Δacc only (1pp granularity); 3 seeds run but std not reported per user decision |
| Significance vs strongest baseline | Paired t-test on per-generator accuracy across 19 generators (CBNet vs each baseline) |
| Faithfulness (Del/Ins AUC, TCAV) | Mean ± Bootstrap 95% CI |
| Per-generator breakdown | Full appendix table |
| Hypothesis test | H1 = ID-CBNet within 2pp of NPR baseline; H2 = OOD-CBNet ≥ NPR + 3pp; H3 = Del-AUC-CBNet < Del-AUC-GradCAM-on-NPR; H4 = each concept Δacc ≥ 0.5pp |

**Multiple comparison correction.** Bonferroni for the paired t-test family across 6 baselines (α/6 = 0.0083). Not applied to per-generator p-values (descriptive use only).

---

## 7. Decision Rules and Contingencies

| Scenario | Decision rule (locked) |
|---|---|
| Training loss NaN or diverges | Reduce lr by 5×, retry. If still diverges, reduce grad accumulation steps. Document. |
| Training crashes (OOM) | Reduce bs to 16. If still OOM, gradient checkpointing on. |
| ID accuracy < 90% | Train +10 more epochs. If still <90%, flag in Discussion as "ID accuracy ceiling on this train budget" — not a project-killer. |
| **H2 fails (OOD acc ≤ NPR + 3pp)** | **Pivot contribution narrative: emphasize structural interpretability + ID accuracy preservation; demote generalization to "secondary contribution / future work" in Discussion.** Honest research path. Paper still defensible at Novelty grade 4-5. |
| H1 fails (ID drop > 2pp) | Try lambda_concept reduction (0.5 → 0.2). If still fails, document as "interpretability-accuracy tradeoff" honestly. |
| **Borderline concept fail GenImage retest (JPEG-QuantTrace or Texture-Geometry remain p > 0.05)** | **Switch to K=4 (anchor concepts only): BitPlane-LSB + Freq-Radial + Color-Manifold + HF-Noise. Re-run main experiment with K=4. Report "8 candidates tested, 4 retained" as a methodologically honest narrative.** |
| AIGI-Holmes baseline cannot run locally (>8GB VRAM, MLLM size) | Cite their reported numbers from the ICCV 2025 paper with a "fairness caveat" footnote. Standard practice when re-running infeasible. |
| Phase B 2/3 elapsed, behind schedule | Cut K-sweep (4 rows) + backbone-ablation (3 rows). Keep concept ablation (6 rows) + loss ablation (3 rows) + main table. |
| Result clearly negative (CBNet acc < strongest baseline by >5pp on ID) | Stop. Re-examine methodology. May need to revisit Stage 1 RESEARCH. Do not write up. |

---

## 8. User-locked Decisions Log

| Decision | User choice | Date |
|---|---|---|
| Training set size | Multi-stage (Stage 1 debug 20k → Stage 2 full 100k → optional Stage 3 324k) | 2026-05-25 |
| Seed count | **3 seeds for all experiments** (including ablations) | 2026-05-25 |
| Val coverage | All GenImage val + ForenSynths OOD (19 generators total) | 2026-05-25 |
| Main table statistics | Mean + Bootstrap 95% CI (1000 resamples) | 2026-05-25 |
| Cross-generator aggregation | Both macro avg + sample-size-weighted avg reported | 2026-05-25 |
| Ablation reporting | 3 seeds run but only Δacc reported (no ±std) | 2026-05-25 |
| H2 failure response | Pivot to "interpretability + ID-acc" framing | 2026-05-25 |
| Borderline concept failure response | K=4 anchor-only set | 2026-05-25 |

**Disagreement on record:** experiment-agent's time-budget estimate for "3 seeds for everything" was 51-76 days (exceeds 14-day Phase B budget). User assessed this estimate as too pessimistic and elected to proceed with the original plan. Risk note: revisit Day 8 of Phase B if elapsed time/run is significantly higher than user's expectation.

---

## 9. Experiment Matrix (executable)

### 9.1 Stage 1: Debug (1 day, 1 seed)

| ID | Config | Purpose |
|---|---|---|
| D-1 | Full CBNet, 20k train, 10 epochs, bs=32, lr default | Verify loss decreases, OOM, time/epoch measurement |

### 9.2 Stage 2: Main + Ablations (Phase B core, 8-12 days budget)

| ID | Config | Seeds | Notes |
|---|---|---|---|
| **M-CBNet** | Full CBNet-AGID (6 concepts, dual stream, all losses) | **3** | Main result |
| M-CNNDetection | Run their published checkpoint, inference only | 1 | Baseline (no training) |
| M-UnivFD | Run their published checkpoint, inference only | 1 | Baseline |
| **M-NPR** | NPR.pth, inference only | 1 | Strongest baseline; published reproducible |
| **M-LOTA** | lota_sdv14.pth, inference only | 1 | Strongest baseline (latest SOTA) |
| M-DIRE | If feasible — diffusion inversion may be too slow | 1 (or cite paper) | Diffusion-specific |
| M-AIGI-Holmes | Cite reported numbers (likely cluster-only model) | — | Direct competitor differentiation |
| **A-concept-drop × 6** | Drop each concept k ∈ {1..6}, otherwise identical to M-CBNet | **3 each** | Ablation 1 |
| **A-loss-drop × 3** | Drop {L_concept, L_gen, L_sparsity} independently | **3 each** | Ablation 2 |
| **A-backbone × 2** | pixel-only / signal-only | **3 each** | Ablation 3 (dual is M-CBNet) |
| **A-K-sweep × 2** | K ∈ {4, 8} (K=6 is M-CBNet) | **3 each** | Ablation 4 |
| **A-no-bottleneck** | Same backbone but FC classifier instead of CBL | **3** | Ablation 5 (the "what does bottleneck cost" experiment) |

**Total Phase B trained rows (with 3 seeds):** 1 + 6 + 3 + 2 + 2 + 1 = **15 rows × 3 seeds = 45 training runs.**

**Schedule policy.** Run all in batched script (e.g., `Code/scripts/run_phase_B.ps1`). Save intermediate checkpoints. Log to wandb or tensorboard.

### 9.3 Stage 3: Optional Scale Validation (1-2 days)

| ID | Config | Purpose |
|---|---|---|
| S-FullTrain | Full 324k × 20 epochs × 1 seed, best M-CBNet config | "Scale" appendix table to show our method works at full scale |

### 9.4 Stage 4: Inference-only Evaluations

Run for each trained checkpoint:
- E-genimage-val: 8 GenImage val sets (~1 hour)
- E-forensynths: 11 ForenSynths val sets (~2 hours)
- E-faithfulness: Del-AUC + Ins-AUC on 200-image faithfulness subset (~30 min per model)
- E-tcav: TCAV concept-sensitivity test (~30 min per model)

---

## 10. Reproducibility

| Item | Spec |
|---|---|
| Random seed scheme | `--seed {42, 1337, 8675309}` — fixed values for the 3 seeds, deterministic Python/numpy/torch seeding |
| Code git commit | TBD at experiment start. Tag the commit on the day of M-CBNet run as `exp/v1`. |
| Environment | `agid` conda env at `E:\LQiu\conda_envs\agid`. Pinned: torch 2.1.2+cu121, torchvision 0.16.2+cu121, numpy 1.26.4, opencv 4.10.0.84. Full `pip freeze` saved to `Logs/<run_id>/env.txt`. |
| Data version | GenImage SD-1.4 official Baidu Pan release downloaded 2026-05-25; ForenSynths official release. SHA256 of `Stable_Diffusion_v1.4/val/ai/` and `nature/` directories logged. |
| Concept-label artifact | `Data/GenImage/Stable_Diffusion_v1.4/train/concept_labels.npy` + `concept_norm.json` (min-max stats). Hash logged. |
| Checkpoints | All saved in `Logs/<run_id>/ckpt_epoch{N}.pth`. Final checkpoint per run is the artifact for evaluation. |
| Eval result artifacts | `Results/eval_<method>_<config>.json` per evaluation, with config + git commit + env hash embedded. |

---

## 11. Risk Register (operational risks for Phase B)

| # | Risk | Likelihood | Severity | Mitigation |
|---|---|---|---|---|
| 1 | Wall-clock per training run exceeds estimate → schedule slip | Med | High | Day 8 check-in; cut ablations per §7 rule if slipping |
| 2 | OOM at bs=32, 256x256, AMP-bf16 | Low | Med | Drop to bs=16 + grad accum; gradient checkpointing |
| 3 | Loss NaN due to bf16 underflow in sparsity entropy | Med | Low | Loss component already clamps `concepts.clamp(min=1e-9)`; switch to fp16 if persists |
| 4 | Borderline concepts fail GenImage retest | High | Low | K=4 fallback already locked (§7) |
| 5 | H2 fails (OOD gain) | Med | Med | Narrative pivot to interpretability framing (§7) |
| 6 | AIGI-Holmes unreplicable | High | Low | Cite their numbers (§7) |
| 7 | ForenSynths download blocked by network | Low | Med | Skip ForenSynths, use only GenImage 8 generators; reduce statistical power on OOD |
| 8 | Disk full (E: has 1020GB free, but checkpoints add up) | Low | Med | Keep only final + best epoch per run; delete intermediate after evaluation |
| 9 | Concept activations collapse to single concept (sparsity loss fails) | Med | Med | Increase lambda_sparsity (0.05 → 0.2); if still collapses, supervise concepts heavier (lambda_concept = 1.0) |

---

## 12. Phase B Operational Plan

### Pre-flight checklist (before any training run)

- [ ] GenImage SD-1.4 train + val downloaded and verified
- [ ] ForenSynths test downloaded (or skip noted)
- [ ] LOTA checkpoint downloaded (`Checkpoints/lota_sdv14.pth`)
- [ ] `python -m cbnet_agid.precompute_concept_labels` completed on train set, output `concept_labels.npy` saved
- [ ] Sanity training run D-1 completed (verifies pipeline + measures actual sec/step)
- [ ] Git commit tagged `exp/v1`

### Daily monitoring (during Phase B)

- Every 12 hours: check `Logs/<run_id>/history.json` for loss trajectory + eval metrics
- Day 8 check-in: actual elapsed vs estimated. If >60% behind, invoke §7 cut rule
- Day 10 check-in: borderline-concept retest results. Decide K=4 vs K=6 for main M-CBNet run

### Stop conditions

- All 15 trained rows × 3 seeds done → proceed to Stage 4 inference-only evaluations
- 14 days elapsed regardless of completion → freeze experiments, proceed to Stage 2 WRITE with what's done
- Catastrophic failure (e.g., all runs diverge) → escalate, reconsider methodology before continuing

---

## 13. Handoff to ARS Stage 2 WRITE

After Phase B completes, the following artifacts will be available for Stage 2 academic-paper writing:

```
Results/
├── main_table.csv           — 7 methods × 19 generators × 4 metrics × 3 seeds
├── ablation_concept.csv      — 6 ablations × 4 metrics × 3 seeds
├── ablation_loss.csv         — 3 ablations × ... × 3 seeds
├── ablation_backbone.csv     — 2 ablations × ... × 3 seeds
├── ablation_K.csv            — 2 ablations × ... × 3 seeds
├── ablation_no_cbl.csv       — 1 row × 3 seeds
├── faithfulness.csv          — Del-AUC + Ins-AUC + TCAV per method
└── pointing_game.csv         — Localization accuracy on 200 annotated images

Logs/
├── exp/v1.txt                — Git tag info
├── env_freeze.txt            — pip freeze
├── system_info.json          — GPU model, driver, VRAM, OS
└── <run_id>/                 — Per-run logs + checkpoints + config
```

These map directly to the Methodology Blueprint §11 paper-section structure.

---

## 14. Open Items / Future Decisions

- [ ] Subsampling script for SD-1.4 train (to be written before Phase A-4 begins). 100k samples, seed 42, balanced AI/real.
- [ ] AIGI-Holmes resource check (memory budget test on local hardware — formality, expected to fail).
- [ ] Pointing Game manual annotation: 200 images, decide whether to do once after main runs or in parallel. **Recommendation:** parallel — annotate during week 1 of Phase B, ~3 hours of manual work.
- [ ] Whether to add ECE (Expected Calibration Error) as 5th metric: low-cost to compute, useful for explainable models. **Recommendation:** add.

---

**END OF EXPERIMENT PLAN.**

This plan is the authoritative reference for Phase B execution. Update with strikethrough + amendment notes if locked decisions change. Final result tables must reference this plan version.
