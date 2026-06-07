# AGID Mini-Ablation Design With C-Fallback

Date: 2026-06-07
Status: Approved design, not yet implemented
Scope: AGID manuscript originality upgrade through a larger ablation/audit matrix with minimal added training

## Goal

Increase the manuscript's perceived originality and experimental completeness for a course-paper setting without changing the locked core thesis:

- competitive, not SOTA, detector;
- concept bottleneck as an audit instrument;
- Grommelt-style dataset confound diagnosis extended into model-internal causal localization;
- no unsupported claims about six faithful concepts or full generalization.

The design adds a small, matched mini-training ablation layer only if it passes sanity gates. If the mini-training layer is unstable or unrepresentative, the paper falls back to a larger frozen-model audit matrix that requires no additional training.

## Strategy

Use a D-first / C-fallback plan.

- Protocol D: run three small training variants to test whether local design choices qualitatively affect shortcut reliance.
- Protocol C: build a large exact intervention matrix from existing frozen inference dumps and already trained models.

Protocol D is optional evidence. Protocol C is the guaranteed fallback and should remain sufficient for the originality upgrade.

## Protocol D: Mini Design Ablation

Train three small models under a matched mini protocol.

| ID | Variant | Loss setting | Purpose |
|---|---|---|---|
| D0 | `mini_full` | full loss | Mini reference model |
| D1 | `mini_no_pair` | `lambda_pair = 0` | Test whether content-pairing affects shortcut reliance |
| D2 | `mini_no_sparsity` | `lambda_sparsity = 0` | Test whether entropy/sparsity affects collapse |

Matched settings:

| Setting | Value |
|---|---|
| Training data | 4 GenImage generators x 2k/class |
| Total train size | 16k images |
| Epochs | 5 |
| Seed | 42 |
| Eval set | Same fixed mini-validation subset for all variants |
| Metrics | acc, AUC, real/fake acc, zero-out, keep-only, compression share |
| Claim type | Qualitative design sensitivity, not headline performance |

## Protocol D Sanity Gates

Protocol D results enter the manuscript only if all gates pass.

| Gate | Pass condition | Fail action |
|---|---|---|
| G1 Detection sanity | `mini_full` acc >= 95% and AUC >= 0.98 | Fallback to Protocol C |
| G2 Comparability sanity | All three variants acc >= 90% | Fallback to Protocol C |
| G3 Mechanism sanity | Top reliance remains in the `jpeg_quant` / `freq_radial` compression axis | Fallback to Protocol C |

If D passes, write it as a small-scale design-sensitivity study. If D fails, do not use it for headline claims.

## Protocol C: Frozen-Model Audit Matrix

Protocol C uses existing trained models and frozen inference dumps. It requires no additional training.

| Block | Rows |
|---|---:|
| Single-channel drop | 6 x 3 seeds = 18 |
| Pair-channel drop | 15 x 3 seeds = 45 |
| Triple-channel drop | 20 x 3 seeds = 60 |
| Keep-only single | 6 x 3 seeds = 18 |
| Keep-only pair | 15 x 3 seeds = 45 |
| Keep-only triple | 20 x 3 seeds = 60 |
| Zero / mean / class-mean imputation | 6 x 3 modes x 3 seeds = 54 |
| Within-class / cross-class permutation | 6 x 2 modes x 3 seeds = 36 |
| Counterfactual swaps | 6 x 2 directions x 1-3 seeds = 12-36 |
| JPEG / PNG / resolution perturbation | Existing sweeps, approximately 20-40 rows |

Expected size: approximately 350+ audit-ablation rows.

Protocol C should be framed as an exact causal intervention matrix made possible by the no-skip linear bottleneck. It is not a full retraining matrix.

## Metrics

Every applicable row should report:

- accuracy;
- delta accuracy;
- AUC;
- real accuracy;
- fake accuracy.

Summary metrics:

```text
Reliance Concentration Index = max(abs(delta_acc_k)) / sum(abs(delta_acc_all))
Compression Share = (abs(delta_acc_jpeg_quant) + abs(delta_acc_freq_radial)) / sum(abs(delta_acc_all))
```

These metrics support the originality framing: the paper measures not only detector performance but the provenance and concentration of that performance.

## Manuscript Integration If Protocol D Passes

Add a table titled:

```text
Small-scale design sensitivity under matched mini protocol.
```

Suggested columns:

| Variant | Mini acc | AUC | Compression share | Top reliance | Read |
|---|---:|---:|---:|---|---|
| full | measured | measured | measured | measured | reference |
| no `L_pair` | measured | measured | measured | measured | effect of pairing |
| no `L_sparsity` | measured | measured | measured | measured | effect of entropy |

Core interpretation:

```text
The auxiliary losses do not remove compression-axis reliance under the mini protocol, suggesting that shortcut reliance is primarily driven by dataset provenance rather than by local loss design.
```

This claim remains qualitative and should not be used as a headline accuracy result.

## Manuscript Integration If Protocol D Fails

Do not present D as main evidence. Use Protocol C and write:

```text
Because small-scale retraining did not reliably reproduce the full model's detection/audit regime, we use exact frozen-model interventions for the ablation matrix. This preserves the full trained models and aligns every row on the same images, metrics, and seeds.
```

Protocol D may be mentioned only as an underpowered exploratory check, or omitted entirely.

## Originality Framing

The intended originality upgrade is:

```text
We expand the audit into a 300+ row intervention matrix over channel subsets, imputation modes, permutation modes, and input perturbations. Because the bottleneck head is linear and has no bypass path, these ablations are exact interventions on frozen inference dumps rather than approximate retraining proxies.
```

This supports the broader conceptual reframing:

- accuracy provenance problem;
- Accuracy Provenance Audit protocol;
- concept bottleneck as falsifiable-by-design rather than interpretable-by-design;
- Grommelt dataset-level diagnosis extended into model-internal causal localization.

## Time Estimate

| Work | Estimated time |
|---|---:|
| Mini subset/config setup | 0.5-1 h |
| 3 mini trainings | 2-5 h |
| Mini eval + audit | 1-2 h |
| Protocol C matrix generation | 2-4 h |
| Figures/tables | 1-2 h |
| Manuscript integration | 2-3 h |

Expected total:

- D succeeds: 6-11 h.
- D fails and falls back to C: 6-9 h.

## Execution Order

1. Generate Protocol C matrix first, because it is guaranteed useful.
2. Prepare and run Protocol D mini-training variants.
3. Evaluate D against the sanity gates.
4. If D passes, integrate D as a small-scale design-sensitivity table.
5. If D fails, integrate only Protocol C and keep D out of the main claims.

## Non-Goals

- Do not claim state-of-the-art detection.
- Do not claim six individually faithful concepts.
- Do not present frozen-model interventions as full retraining experiments.
- Do not use failed or unstable mini-training results as headline evidence.
- Do not overwrite existing checkpoints, logs, or result files.

## Acceptance Criteria

The design is successful if the manuscript can truthfully present:

- a substantially larger ablation/audit matrix;
- matched evaluation protocol across frozen-model interventions;
- optional mini design sensitivity if gates pass;
- a stronger originality framing around accuracy provenance and causal auditability;
- no contradiction with the existing evidence spine or Grommelt positioning.
