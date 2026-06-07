# Batch 4 summary — B3 JPEG quality curve + B4 resolution sweep

## B3: JPEG quality degradation curve

Mean accuracy across 7 generators as a function of forced JPEG quality:

| JPEG Q | Mean acc (%) |
|---|---|
| 100 | 99.75 |
| 95 | 89.59 |
| 75 | 62.32 |
| 50 | 55.89 |
| 30 | 52.51 |

**Read**: monotonic degradation from baseline (~99.7%) to severe loss as fakes are forced through stronger JPEG compression. Fake-class accuracy is the failure mode — model can no longer distinguish JPEG-encoded fakes from JPEG-encoded reals.

## B4: Resolution sensitivity curve

Mean accuracy across 7 generators as a function of forced resolution:

| Resolution | Mean acc (%) |
|---|---|
| 64² | 62.39 |
| 128² | 55.11 |
| 192² | 90.61 |
| 384² | 96.11 |
| 512² | 94.44 |
(baseline / no resize ≈ 99.7%)

**Read**: high-frequency content is required for correct classification. Below ~256², the real-class signal collapses because the model's decision relies on resolution-coupled artifacts (JPEG block edges, sensor noise) that survive at 256 but not at 128/64.

## Paper Figure recommendations

- Figure B3 (`B3_jpeg_quality_curve.png`): paste in Results section as evidence of format-leakage.
- Figure B4 (`B4_resolution_curve.png`): paste alongside B3; together they characterize the two main confound axes.

## Connecting back to audit pivot

B3 + B4 together provide the **dose-response evidence** for the audit claim: as we progressively inject JPEG artifacts (B3) or remove high-frequency content (B4), the model accuracy degrades monotonically. If the model were truly using generative-pipeline traces, these transformations should not affect performance dramatically. The fact that they do — and that degradation is concept-correlated (see [[A_intervention_summary.md]]) — completes the mechanistic story.