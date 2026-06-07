# E1. SOTA Baselines on GenImage — Related-Work Comparison Table

**Compiled:** 2026-05-28 (research-agent web search, ~40 min budget)
**Purpose:** related-work / comparison table for CBNet-AGID paper.
**GenImage reference:** Zhu, M. et al. (2023). *GenImage: A Million-Scale Benchmark for Detecting AI-Generated Image.* NeurIPS 2023 D&B Track.

---

## TL;DR

- **The "standard" GenImage cross-generator protocol** (used in GenImage, NPR, AIDE, FreqNet's third-party re-eval, FatFormer's third-party re-eval, IAPL, and most CVPR 2024-2025 papers) = **train on SD-v1.4 subset only, test on each of the 8 generators' val sets**. Report per-generator accuracy + mean.
- **LOTA (ICCV 2025) uses a DIFFERENT protocol** = 8 separate single-generator models, averaged across all 8 train/test pairs. Its 98.9 % mean is NOT directly comparable to the SD-v1.4-only protocol numbers.
- Several headline methods (**NPR's own paper**, **FreqNet**, **FatFormer**) do **not evaluate on GenImage in their own publications** — they train on ProGAN / ForenSynths. Their GenImage numbers in the literature are **third-party re-implementations** by AIDE / IAPL / LOTA / Sanity-Check. This is a real source of inconsistency that the CBNet paper must flag.
- For CBNet's 4-generator (SD-v1.4 + BigGAN + ADM + MJ) joint-training protocol, **no existing baseline uses the same protocol** — closest analog is the GenImage paper's joint-trained ResNet-50 baseline. All comparison numbers below are SD-v1.4-only single-generator training unless flagged.

---

## Master Comparison Table — GenImage, SD-v1.4-trained, cross-generator test

Numbers are per-generator val accuracy (%). Cells marked "—" = not reported in that paper / not in this protocol.

Sources are listed in the "Src" column (legend below table). Where multiple sources report different numbers for the same method, I cite the **most recent re-implementation** (typically IAPL 2025 or LOTA 2025), and flag inconsistencies in the Notes section.

| Method | Year / Venue | Protocol | Midjourney | SD-v1.4 | SD-v1.5 | ADM | GLIDE | Wukong | VQDM | BigGAN | **Mean** | Src |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CNNSpot (Wang) | CVPR 2020 | SD-v1.4 train | 52.8 | 96.3 | 95.9 | 50.1 | 39.8 | 78.6 | 53.4 | 46.8 | **64.2** | A,B |
| F3-Net (Qian) | ECCV 2020 | SD-v1.4 train | 50.1 | 99.9 | 99.9 | 49.9 | 50.0 | 99.9 | 49.9 | 49.9 | **68.7** | A,B |
| GramNet (Liu) | CVPR 2020 | SD-v1.4 train | 54.2 | 99.2 | 99.1 | 50.3 | 54.6 | 98.9 | 50.8 | 51.7 | **69.9** | A,B |
| Spec (Frank) | ICML 2020 | SD-v1.4 train | 52.0 | 99.4 | 99.2 | 49.7 | 49.8 | 94.8 | 55.6 | 49.8 | **68.8** | A,B |
| ResNet-50 (baseline) | — | SD-v1.4 train | 54.9 | 99.9 | 99.7 | 53.5 | 61.9 | 98.2 | 56.6 | 52.0 | **72.1** | B |
| DeiT-S (baseline) | — | SD-v1.4 train | 55.6 | 99.9 | 99.8 | 49.8 | 58.1 | 98.9 | 56.9 | 53.5 | **71.6** | B |
| Swin-T (baseline) | — | SD-v1.4 train | 62.1 | 99.9 | 99.8 | 49.8 | 67.6 | 99.1 | 62.3 | 57.6 | **74.8** | B |
| DIRE (Wang) | ICCV 2023 | SD-v1.4 train | 60.2 | 99.9 | 99.8 | 50.9 | 55.0 | 99.2 | 50.1 | 50.2 | **70.7** | B |
| UnivFD / Ojha (CLIP-LR) | CVPR 2023 | SD-v1.4 train | 73.2 / 91.5\* | 84.2 / 96.4\* | 84.0 / 96.1\* | 55.2 / 58.1\* | 76.9 / 73.4\* | 75.6 / 94.5\* | 56.9 / 67.8\* | 80.3 / 57.7\* | **73.3 / 79.5\*** | B / C |
| GenDet | arXiv 2023 | SD-v1.4 train | 89.6 | 96.1 | 96.1 | 58.0 | 78.4 | 92.8 | 66.5 | 75.0 | **81.6** | B |
| PatchCraft (Zhong) | arXiv 2024 | SD-v1.4 train | 79.0 | 89.5 | 89.3 | 77.3 | 78.4 | 89.3 | 83.7 | 72.4 | **82.3** | B |
| **NPR (Tan)** | **CVPR 2024** | **SD-v1.4 train** | **81.0** | **98.2** | **97.9** | **76.9** | **89.8** | **96.9** | **84.1** | **84.2** | **88.6** | C, D |
| **FreqNet (Tan)** | **AAAI 2024** | SD-v1.4 train (re-impl) | 89.6 | 98.8 | 98.6 | 66.8 | 86.5 | 97.3 | 75.8 | 81.4 | **86.8** | C |
| **AIDE (Yan)** | **arXiv 2024 (NeurIPS-W?)** | **SD-v1.4 train** | 79.38 | 99.74 | 99.76 | 78.54 | 91.82 | 98.65 | 80.26 | 66.89 | **86.88** | E |
| FatFormer (Liu) | CVPR 2024 | SD-v1.4 train (re-impl) | 92.7 | 100.0 | 99.9 | 75.9 | 88.0 | 99.9 | 98.8 | 55.8 | **88.9** | C |
| DRCT | arXiv 2024 | SD-v1.4 train | 91.5 | 95.0 | 94.4 | 79.4 | 89.2 | 94.7 | 90.0 | 81.7 | **89.5** | C |
| SAFE | arXiv 2024 | SD-v1.4 train | 95.7 | 99.9 | 99.8 | 59.5 | 91.7 | 95.7 | 98.0 | 77.4 | **90.3** | C |
| C2P-CLIP | arXiv 2024 | SD-v1.4 train | 88.2 | 90.9 | 97.9 | 96.4 | 99.0 | 98.8 | 96.5 | 98.7 | **95.8** | C |
| **LOTA (Wang)** | **ICCV 2025** | **per-gen train, cross-gen test, avg over all 8** | 93.2 | 99.8 | 99.8 | 99.5 | 99.2 | 99.8 | 99.5 | 99.9 | **98.9** ⚠ | F |

\*UnivFD: two sets of numbers reported — the AIDE-paper re-implementation gives 73.3 % mean, the more recent IAPL-paper re-impl gives 79.5 % mean. Discrepancy attributable to fine-tuning protocol differences.

⚠ LOTA mean is **NOT directly comparable** — see protocol notes below.

### Source legend

| Code | Source | Used for |
|---|---|---|
| A | Zhu et al., GenImage paper (NeurIPS 2023 D&B), Table 3 | baseline CNNSpot/F3Net/GramNet/Spec |
| B | Yan et al., AIDE paper (arXiv:2406.19435), Table 3 | CNNSpot, Spec, F3Net, GramNet, ResNet-50, DeiT-S, Swin-T, DIRE, UnivFD, GenDet, PatchCraft, AIDE |
| C | IAPL paper (arXiv:2508.01603), Table 3 | UnivFD, NPR, FreqNet, FatFormer, DRCT, AIDE, C2P-CLIP, SAFE — most recent re-eval |
| D | NPR GitHub README (chuangchuangtan/NPR-DeepfakeDetection) | NPR self-reported GenImage cross-gen |
| E | AIDE paper (arXiv:2406.19435), Table 3, "Ours" col | AIDE self-reported |
| F | LOTA paper (arXiv:2510.14230, ICCV 2025), Table 2 | LOTA, ESSP, LaRE2, DIRE (LOTA's protocol) |

---

## Per-method details

### 1. NPR — Tan, Liu, et al. CVPR 2024

- **Title:** Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection
- **Authors:** Chuangchuang Tan, Huan Liu, Yao Zhao, Shikui Wei, Guanghua Gu, Ping Liu, Yunchao Wei
- **arXiv:** 2312.10461 (CVPR 2024)
- **Code:** https://github.com/chuangchuangtan/NPR-DeepfakeDetection
- **In-paper evaluation:** trains on ProGAN-4class (ForenSynths protocol), evaluates on 28 GAN+diffusion sources. **The published paper itself does NOT contain a GenImage table.**
- **GenImage numbers** in the table above come from:
  - the NPR GitHub README (which provides an SD-v1.4-trained checkpoint with seed=70 and reports the cross-gen numbers used in IAPL Table 3, basically matching)
  - the IAPL 2025 re-implementation, which is consistent (mean 88.6 %).
- **Metric:** overall accuracy (real+fake combined per generator). No per-class real/fake breakdown in the open results.
- **Label convention:** **1 = fake, 0 = real** (verified from train.py in the NPR repo; matches the project memory note that NPR uses opposite convention vs. LOTA).
- **Caveat:** Because NPR generalizes well from ProGAN to diffusion via up-sampling artifacts, the SD-v1.4-trained number is technically "best case" — it was *not* the primary protocol the authors evaluated.

### 2. LOTA — Wang et al. ICCV 2025

- **Title:** LOTA: Bit-Planes Guided AI-Generated Image Detection
- **arXiv:** 2510.14230
- **Code:** https://github.com/hongsong-wang/LOTA
- **In-paper protocol (very important):** "For each classifier, training was conducted on the training subsets, followed by comprehensive evaluation across all eight testing subsets." → **8 separate models, one per generator, then average all 8×8 = 64 cells (or the diagonal-included full matrix)**. This means LOTA's mean 98.9 % includes in-distribution diagonal cells (always ≥99 %), pushing the average up artificially relative to a strict cross-gen protocol.
- **Metric:** overall accuracy.
- **Label convention:** **0 = fake, 1 = real** (opposite of NPR — confirmed by the project-memory note).
- **Caveat for the CBNet paper:** LOTA's 98.9 % cannot be put in the same column as NPR's 88.6 % without a footnote. If you want a strictly comparable LOTA number on the SD-v1.4-only protocol, you must either re-run their code or cite their per-row results when trained on SD-v1.4. From LOTA's Table 1 (per-train-source breakdown — not all extracted, see paper), the SD-v1.4-trained row is the most comparable single row but I did not extract those individual cells in this search budget.

### 3. AIDE — Yan, Wang, et al. 2024

- **Title:** A Sanity Check for AI-generated Image Detection (introduces AIDE = "AI-generated Image DEtector with Hybrid features")
- **Authors:** Shilin Yan, Ouxiang Li, Jiayin Cai, Yanbin Hao, …
- **arXiv:** 2406.19435 (NeurIPS 2024 workshop / preprint; openreview ID ODRHZrkOQM)
- **Protocol:** SD-v1.4 train, cross-gen test on all 8 generators. **Same protocol as GenImage paper's Table 3.**
- **Per-generator (Table 3, "Ours" column):** MJ 79.38 / SDv1.4 99.74 / SDv1.5 99.76 / ADM 78.54 / GLIDE 91.82 / Wukong 98.65 / VQDM 80.26 / BigGAN 66.89 / **mean 86.88 %**.
  - ⚠ Note: there are TWO sets of AIDE numbers in the literature. The number compiled in IAPL Table 3 says AIDE mean = **86.9 %** with the same per-gen breakdown. Earlier blog/secondary sources sometimes quote AIDE as "+4.6 % over SOTA → 90 %ish" which refers to a different ablation. Use 86.88 % as the canonical AIDE number.
- **Metric:** overall acc; AIDE separately reports fake-acc / real-acc on its own "Chameleon" testset (Table 4) but on GenImage uses single overall acc.
- **Label convention:** **1 = fake, 0 = real** (binary classifier, but the convention is not made explicit in the AIDE paper — verify if needed).

### 4. FreqNet — Tan et al. AAAI 2024

- **Title:** Frequency-Aware Deepfake Detection: Improving Generalizability through Frequency Space Domain Learning
- **Authors:** Chuangchuang Tan, Yao Zhao, Shikui Wei, Guanghua Gu, Ping Liu, Yunchao Wei
- **arXiv:** 2403.07240 (AAAI 2024)
- **In-paper protocol:** **trains on ProGAN (ForenSynths 20-class)**, evaluates on 17 GAN models + DIRE-diffusions + Ojha-diffusions + 5 self-synth diffusions (DDPM/IDDPM/ADM/MJ/DALLE). **The published paper does NOT report on GenImage.**
- **GenImage numbers** in the master table above (mean 86.8 %) come from the **IAPL 2025 re-implementation** where the authors retrained FreqNet on GenImage's SD-v1.4 subset using the public code.
- **Caveat:** A different secondary source (an aicoin.com article and one of my early searches) quoted FreqNet with per-gen numbers MJ 97.25 / SDv1.4 87.95 / SDv1.5 84.40 / ADM 86.55 / GLIDE 67.25 / Wukong 97.75 / VQDM 97.40 / BigGAN 97.20 / mean **89.47 %**. This is likely an *all-generator-joint-training* re-evaluation by a different group and **does not match IAPL's SD-v1.4-only re-impl**. Use IAPL's 86.8 % for consistency with the SD-v1.4 column.
- **Metric:** overall acc.
- **Label convention:** **1 = fake, 0 = real** (same Tan-lab repo conventions as NPR).

### 5. PatchCraft — Zhong et al. arXiv 2024

- **arXiv:** 2311.12397
- **Protocol:** SD-v1.4 train, cross-gen test.
- **AIDE-paper Table 3 row:** MJ 79.0 / SDv1.4 89.5 / SDv1.5 89.3 / ADM 77.3 / GLIDE 78.4 / Wukong 89.3 / VQDM 83.7 / BigGAN 72.4 / mean **82.3 %**.
- Notable because it's the strongest of the "traditional" feature-engineering baselines.

### 6. FatFormer — Liu et al. CVPR 2024

- **Title:** Forgery-aware Adaptive Transformer for Generalizable Synthetic Image Detection
- **arXiv:** 2312.16649
- **In-paper evaluation:** trains on 4-class ProGAN. Evaluates on ForenSynths-GANs + a 6-diffusion suite. **Does NOT evaluate on GenImage in the paper.**
- **GenImage numbers** above (mean 88.9 %) come from IAPL 2025 re-implementation.
- **Caveat:** FatFormer's BigGAN drop to 55.8 % is suspicious — likely a re-implementation artifact rather than a fundamental weakness, because the paper itself reports strong GAN generalization. Treat with skepticism.

### 7. Universal Fake Detector (UnivFD / Ojha) — CVPR 2023

- **Title:** Towards Universal Fake Image Detectors that Generalize Across Generative Models
- **Authors:** Utkarsh Ojha, Yuheng Li, Yong Jae Lee
- **CVPR 2023**, pp. 24480-24489
- **Method:** linear probe on frozen CLIP-ViT-L/14 features
- **In-paper protocol:** trains on ProGAN, tests on diffusion suites (LDM, GLIDE, Guided, DALL-E). **No GenImage table in original paper.**
- **GenImage numbers:** AIDE-paper Table 3 gives mean **73.3 %**; IAPL paper gives mean **79.5 %**. Pick the more recent IAPL number for citation consistency. The discrepancy reflects different LR / regularization choices in the linear probe.

### 8. CNNDetection / CNNSpot — Wang et al. CVPR 2020

- **Title:** CNN-Generated Images Are Surprisingly Easy to Spot… for Now
- **In-paper:** trained on ProGAN-LSUN, tested on 11 GAN sources.
- **GenImage numbers** from GenImage paper Table 3 (the canonical baseline): mean **64.2 %**. This is the "lower bound" reference for the comparison table.
- **Label convention:** **1 = fake, 0 = real**.

### 9. F3-Net — Qian et al. ECCV 2020

- **Title:** Thinking in Frequency: Face Forgery Detection by Mining Frequency-aware Clues
- **In-paper:** trained for face forgery (FaceForensics++).
- **GenImage numbers** from GenImage paper Table 3: mean **68.7 %**. Heavily collapses to ~50 % on out-of-domain generators — a useful weak baseline.

---

## Critical comparability notes (must appear in CBNet paper)

1. **Protocol heterogeneity is endemic.** At least three protocols are in active use:
   - **(P1) SD-v1.4-only train, 8-gen cross-gen test** — the original GenImage paper protocol. Used by AIDE, NPR (re-impl), FreqNet (re-impl), PatchCraft, FatFormer (re-impl), DRCT, SAFE, C2P-CLIP, IAPL, and the published CNNDetection/F3Net/GramNet/Spec baselines.
   - **(P2) Per-generator single-train, cross-gen test, averaged over all 8 training sources** — used by LOTA. Mean values look ~10 pp higher than under P1 because the diagonal cells are included.
   - **(P3) ProGAN-only train, GenImage cross-test** — used by NPR's original paper, FatFormer, FreqNet originals, UnivFD. Numbers in this regime are usually a few points worse than P1.
2. **CBNet trained on 4 generators (SD-v1.4 + BigGAN + ADM + MJ)** is its own protocol, call it **P4: joint-multi-gen train**. The closest analog in the literature is the GenImage paper's joint-trained ResNet-50 baseline (~98+ % when trained on multiple generators). For a fair comparison, CBNet should either:
   - retrain a baseline (CNNSpot / NPR / AIDE) under the same P4 4-generator protocol and report it, or
   - acknowledge that CBNet's per-gen accuracy is "in-distribution" on the 4 training gens and "OOD" on GLIDE/Wukong/VQDM/SDv1.5, then compare CBNet-on-OOD against P1-baselines-on-cross-gen (apples-to-oranges with a footnote).
3. **Label conventions diverge.** Confirmed:
   - **NPR / FreqNet / CNNSpot:** `1 = fake, 0 = real` (Tan-lab convention).
   - **LOTA:** `0 = fake, 1 = real` (opposite — verified per project-memory note).
   - **AIDE:** not explicitly stated; treat as `1 = fake` unless code says otherwise.
   - **CBNet (own code):** check `concepts/base.py` for what convention CBNet uses before importing any pretrained NPR/LOTA checkpoint or comparing logits directly.
4. **"Mean" definitions vary.** Some papers report unweighted arithmetic mean of 8 per-gen accuracies (most common). Others report a sample-size-weighted mean (rare). The numbers above are unweighted means.
5. **Real-vs-fake breakdown is rare.** Of the methods surveyed, only AIDE reports `fake-acc / real-acc` separately, and only on its proprietary Chameleon set — not on GenImage. CBNet's real_acc / fake_acc reporting would be a useful contribution beyond the SOTA convention.

---

## BibTeX-ready citations

```bibtex
@inproceedings{zhu2023genimage,
  title     = {{GenImage}: A Million-Scale Benchmark for Detecting {AI}-Generated Image},
  author    = {Zhu, Mingjian and Chen, Hanting and Yan, Qiangyu and Huang, Xudong and Lin, Guanyu and Li, Wei and Tu, Zhijun and Hu, Hailin and Hu, Jie and Wang, Yunhe},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track},
  year      = {2023}
}

@inproceedings{tan2024npr,
  title     = {Rethinking the Up-Sampling Operations in {CNN}-based Generative Network for Generalizable Deepfake Detection},
  author    = {Tan, Chuangchuang and Liu, Huan and Zhao, Yao and Wei, Shikui and Gu, Guanghua and Liu, Ping and Wei, Yunchao},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2024}
}

@inproceedings{tan2024freqnet,
  title     = {Frequency-Aware Deepfake Detection: Improving Generalizability through Frequency Space Domain Learning},
  author    = {Tan, Chuangchuang and Zhao, Yao and Wei, Shikui and Gu, Guanghua and Liu, Ping and Wei, Yunchao},
  booktitle = {AAAI Conference on Artificial Intelligence (AAAI)},
  year      = {2024}
}

@article{yan2024aide,
  title   = {A Sanity Check for {AI}-generated Image Detection},
  author  = {Yan, Shilin and Li, Ouxiang and Cai, Jiayin and Hao, Yanbin and Jiang, Xiaolong and Hu, Yao and Xie, Weidi},
  journal = {arXiv preprint arXiv:2406.19435},
  year    = {2024}
}

@inproceedings{wang2025lota,
  title     = {{LOTA}: Bit-Planes Guided {AI}-Generated Image Detection},
  author    = {Wang, Hongsong and others},
  booktitle = {IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025}
}

@inproceedings{liu2024fatformer,
  title     = {Forgery-aware Adaptive Transformer for Generalizable Synthetic Image Detection},
  author    = {Liu, Huan and Tan, Zichang and Tan, Chuangchuang and Wei, Yunchao and Wang, Jingdong and Zhao, Yao},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2024}
}

@inproceedings{ojha2023univfd,
  title     = {Towards Universal Fake Image Detectors that Generalize Across Generative Models},
  author    = {Ojha, Utkarsh and Li, Yuheng and Lee, Yong Jae},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2023}
}

@inproceedings{wang2020cnnspot,
  title     = {{CNN}-Generated Images Are Surprisingly Easy to Spot… for Now},
  author    = {Wang, Sheng-Yu and Wang, Oliver and Zhang, Richard and Owens, Andrew and Efros, Alexei A.},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2020}
}

@inproceedings{qian2020f3net,
  title     = {Thinking in Frequency: Face Forgery Detection by Mining Frequency-aware Clues},
  author    = {Qian, Yuyang and Yin, Guojun and Sheng, Lu and Chen, Zixuan and Shao, Jing},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2020}
}

@article{zhong2024patchcraft,
  title   = {{PatchCraft}: Exploring Texture Patch for Efficient {AI}-generated Image Detection},
  author  = {Zhong, Nan and Xu, Yiran and Qian, Zhenxing and Zhang, Xinpeng},
  journal = {arXiv preprint arXiv:2311.12397},
  year    = {2024}
}
```

---

## Recommendation for the CBNet paper related-work section

1. **Show the master table above** (or a subset of 6–8 most-cited methods) in the related-work or experiments section.
2. **Add a footnote on each row** stating training protocol (P1 / P2 / P3 / P4) explicitly. A reader scanning the table without footnotes will treat the numbers as comparable when they are not.
3. **Pick ONE comparable mean column** as the headline number. Recommend using the SD-v1.4-only protocol (P1) numbers from IAPL Table 3, since IAPL re-implemented the most methods uniformly and the table is the most recent (2025).
4. **For CBNet's own row, report TWO sub-rows:**
   - In-distribution (SD-v1.4, BigGAN, ADM, MJ) — already at 99+ % per the May-27 results in handoff.
   - OOD (GLIDE, Wukong, VQDM, SD-v1.5) — these are the cells that determine whether CBNet beats the SOTA on cross-gen generalization. From May-27 OOD eval: GLIDE 99.80 / Wukong 99.45 / VQDM 99.75, mean 99.67 %. That mean OOD number — if it holds up under fair re-comparison — would clearly beat AIDE (86.88), NPR (88.6), FatFormer (88.9), DRCT (89.5), and approach C2P-CLIP (95.8) and LOTA-P2 (98.9).
5. **Add a "Comparability" column** that flags each baseline as "directly comparable / re-implemented / protocol-divergent". This earns reviewer goodwill and pre-empts the standard "are these numbers apples-to-apples?" Reviewer-2 question.
6. **Do NOT cite LOTA's 98.9 % as a direct upper bound** without the P2 footnote — Reviewer-2 will catch it. Better: cite LOTA's per-source breakdown if you can extract it from their Table 1, or state explicitly that LOTA's protocol differs.

---

*End of E1 SOTA Baselines report.*
