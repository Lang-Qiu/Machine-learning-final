# Batch 2: Confound quantification — summary

## Δacc vs baseline (negative = degradation under perturbation)

| Variant | mean Δacc all | mean Δacc train-gen | mean Δacc OOD |
|---|---|---|---|
| jpeg-q95 | -10.16 pp | -10.83 pp | -9.28 pp |
| png | +0.00 pp | +0.00 pp | +0.00 pp |
| res128 | -44.64 pp | -43.91 pp | -45.62 pp |
| independent_real | — | — | +0.12 pp |

## Per-generator detail (acc % under each variant)

| generator             | baseline | jpeg-q95 | png   | res128 | independent_real |
| --------------------- | -------- | -------- | ----- | ------ | ---------------- |
| ADM                   | 99.90    | 85.90    | 99.90 | 55.90  | —                |
| BigGAN                | 99.80    | 80.45    | 99.80 | 57.00  | —                |
| GLIDE                 | 99.80    | 87.45    | 99.80 | 54.05  | 99.90            |
| Midjourney            | 99.70    | 95.25    | 99.70 | 56.45  | —                |
| Stable_Diffusion_v1.4 | 99.85    | 94.35    | 99.85 | 54.25  | —                |
| VQDM                  | 99.75    | 89.35    | 99.75 | 54.05  | 99.85            |
| Wukong                | 99.45    | 94.35    | 99.45 | 54.05  | 99.60            |

## Interpretation hints (for paper Limitations)

**B1 (jpeg-q95 / png):** If Δacc ≈ 0, the model is robust to image format. If Δacc < -5pp, the model leans heavily on encoder artifacts (a confound). Recall A2 found |w| for jpeg_quant = 13.5 (largest); the format leakage hypothesis predicts a measurable drop under unified re-encoding.

**B2 (res128):** Downsampling to 128² wipes high-frequency components. If Δacc < -5pp, the bitplane-LSB / hf-noise / freq-radial concepts are the load-bearing ones — resolution-coupled. Expected, given native resolutions span 128² (BigGAN) → 1024² (MJ).

**B5 (independent_real):** OOD eval previously shared the same 1000 real images across GLIDE/Wukong/VQDM. Under disjoint 1k samples per OOD generator, real_acc remaining ~99.6% confirms the result is not a spurious artifact of which 1000 real images were sampled.