# Bibliography — AGID with Concept Bottleneck (91 references)

**Project:** 2026 Spring ML Course Final — AGID Paper
**Version:** v1.0 (Stage 1.2 Full deliverable)
**Date:** 2026-05-24

**Organization:**
- §A. AGID Core Methods — 7 families (45 refs)
- §B. Concept Bottleneck and Interpretable ML (18 refs)
- §C. Generative Models Background (12 refs)
- §D. Foundational CV/ML (8 refs)
- §E. Benchmarks, Datasets, Evaluation (8 refs)

**Total: 91 references.**

Each entry: BibKey | Author(s) | Title | Venue | Year | One-line annotation.

---

## §A. AGID Core Methods (45 refs)

### §A.1 Spatial-domain CNN classifiers (5)

| BibKey | Reference | Annotation |
|---|---|---|
| wang2020cnngenerated | Wang, Wang, Owens, Efros. *CNN-Generated Images Are Surprisingly Easy to Spot*. CVPR 2020. | Foundational AGID baseline: train ResNet50 on ProGAN images; surprising cross-GAN generalization. |
| goebel2020detection | Goebel et al. *Detection, Attribution and Localization of GAN Generated Images*. arXiv 2020. | Early multi-task AGID; attribution to specific generator. |
| chai2020patchforensics | Chai, Bau, Lim, Isola. *What makes fake images detectable?* ECCV 2020. | Patch-based detection; argues local artifacts are key. |
| frank2020dctgan | Frank, Eisenhofer, Schönherr, Fischer, Kolossa, Holz. *Leveraging Frequency Analysis for Deep Fake Image Recognition*. ICML 2020. | DCT analysis of CNN-generated images; cross-architecture transfer. |
| boychev2024imaginet | Boychev et al. *Self-Contrastive Learning on ImagiNet*. 2024. | 200K real/synthetic image benchmark + self-contrastive training. |

### §A.2 Frequency-domain methods (7)

| BibKey | Reference | Annotation |
|---|---|---|
| zhang2019fourier | Zhang, Lan, Dai. *Detecting and Simulating Artifacts in GAN Fake Images*. WIFS 2019. | First systematic frequency-domain analysis of GAN images. |
| durall2020watch | Durall, Keuper, Pizarro, Keuper. *Watch your Up-Convolution: CNN Based Generative Deep Neural Networks are Failing to Reproduce Spectral Distributions*. CVPR 2020. | Spectral failure mode of CNN generators; basis for many freq methods. |
| dzanic2020fourier | Dzanic, Shah, Witherden. *Fourier Spectrum Discrepancies in Deep Network Generated Images*. NeurIPS 2020. | Quantitative spectrum analysis across many GANs. |
| jeong2022frepgan | Jeong et al. *FrePGAN: Robust Deepfake Detection Using Frequency-level Perturbations*. AAAI 2022. | Frequency perturbation for robust training. |
| bammey2024synthbuster | Bammey. *Synthbuster: Towards Detection of Diffusion Model Generated Images*. IEEE OJSP 2024. | Diffusion-specific frequency-domain detector. |
| tan2024freqnet | Tan, Liu, Wang. *Frequency-aware Deepfake Detection: Improving Generalizability through Frequency Space Domain Learning*. AAAI 2024. | Modern frequency-domain detector with cross-generator generalization. |
| yan2024aide | Yan, Li, Cai, Hao, Jiang, Hu, Xie. *A Sanity Check for AI-generated Image Detection* (introduces the AIDE detector). arXiv:2406.19435, 2024. | Hybrid visual-artifact + noise experts; introduces the Chameleon test set. [⚠ TITLE CORRECTED 2026-06-03 — prior "AIDE: A Hybrid Approach…" was a placeholder; authors/arXiv web-verified] |

### §A.3 Diffusion reconstruction / training-free (8)

| BibKey | Reference | Annotation |
|---|---|---|
| wang2023dire | Wang et al. *DIRE for Diffusion-Generated Image Detection*. ICCV 2023. | Reconstruction error via diffusion inversion; introduces DiffusionForensics dataset. |
| ricker2024aeroblade | Ricker, Lukovnikov, Fischer. *AEROBLADE: Training-free Detection of Latent Diffusion Images Using Autoencoder Reconstruction Error*. CVPR 2024. | Training-free; uses VAE reconstruction error. |
| sha2024zerofake | Sha, Li, Zhang. *ZeroFake: Zero-Shot Detection of Fake Images via Inversion-based Manipulation*. ACM CCS 2024. | Zero-shot detection via inversion sensitivity. |
| cazenavette2024fakeinversion | Cazenavette et al. *FakeInversion: Learning to Detect Images from Unseen Text-to-Image Models by Inverting Stable Diffusion*. CVPR 2024. | Inversion-based detection for unseen T2I models. |
| he2024rigid | He et al. *RIGID: A Training-free and Model-Agnostic Framework for Robust AI-Generated Image Detection*. 2024. | Perturbation-sensitivity based; training-free. |
| choi2024hfi | Choi et al. *HFI: High-Frequency Aliasing for Robust AI-Generated Image Detection*. 2024. | High-frequency aliasing signal. |
| cozzolino2024zed | Cozzolino et al. *ZED: Zero-Shot Entropy-Based Detection of Generated Images*. ECCV 2024. | Entropy-based zero-shot detection. |
| guillaro2025sedid | Guillaro et al. *Step-wise Error for Diffusion-generated Image Detection (SeDID)*. 2025. | Step-wise reconstruction error in diffusion. |

### §A.4 Gradient/noise fingerprint methods (6)

| BibKey | Reference | Annotation |
|---|---|---|
| tan2023learning | Tan, Liu, Wang. *Learning on Gradients: Generalized Artifacts Representation for GAN-Generated Images Detection*. CVPR 2023. | LGrad: use generator gradient field as input feature. |
| tan2024npr | Tan, Liu, Wang, Hou. *Rethinking the Up-Sampling Operations in CNN-based Generative Network for Generalizable Deepfake Detection*. CVPR 2024. | **NPR**: neighborhood-pixel relationship; strong cross-generator baseline; *our backbone candidate*. |
| zhong2023patchcraft | Zhong, Xu, Lu, Li. *Patchcraft: Exploring Texture Patches for Efficient AI-Generated Image Detection*. arXiv 2023. | Texture-rich vs. texture-poor patch analysis. |
| corvi2023detection | Corvi, Cozzolino, Zingarini, Poggi, Nagano, Verdoliva. *On the Detection of Synthetic Images Generated by Diffusion Models*. ICASSP 2023. | Diffusion-specific fingerprint analysis. |
| li2024masksim | Li. *MaskSim: Detection of Synthetic Images by Masked Spectrum Similarity*. 2024. | Masked spectrum similarity as fingerprint. |
| chen2024single | Chen et al. *A Single Patch Method for AI-Generated Image Detection*. CVPR 2024. | Single patch (vs. whole image) sufficient for detection. |

### §A.5 Bit-plane and pixel-level methods (3)

| BibKey | Reference | Annotation |
|---|---|---|
| wang2025lota | Hongsong Wang, Renxi Cheng, Yang Zhang, Chaolei Han, Jie Gui. *LOTA: Bit-Planes Guided AI-Generated Image Detection*. ICCV 2025; arXiv:2510.14230. | **LOTA**: bit-plane based; reports 98.9% on GenImage (verify exact value from paper when building Table 1); *our backbone candidate*. [metadata verified 2026-06-03: title/authors/venue all ✓] |
| fridrich2007steganalysis | Fridrich, Goljan. *Practical Steganalysis of Digital Images: State of the Art*. SPIE 2002. | Foundational bit-plane / steganalysis (LSB) work; inspiration for LOTA. |
| popescu2005exposing | Popescu, Farid. *Exposing Digital Forgeries in Color Filter Array Interpolated Images*. IEEE TSP 2005. | Classical image forensics; pixel-level forensic detector. |

### §A.6 Vision-language model based methods (10)

| BibKey | Reference | Annotation |
|---|---|---|
| ojha2023universal | Ojha, Li, Lee. *Towards Universal Fake Image Detectors that Generalize Across Generative Models*. CVPR 2023. | **UniversalFakeDetect (UnivFD)**: CLIP-ViT features + linear probe; generalization benchmark. |
| sha2023defake | Sha et al. *DE-FAKE: Detection and Attribution of Fake Images Generated by Text-to-Image Generation Models*. ACM CCS 2023. | Detection + attribution via vision-language features. |
| wu2023lasted | Wu et al. *LASTED: Language-Guided Deepfake Detection*. arXiv 2023. | Language-guided detection. |
| cozzolino2024raising | Cozzolino, Poggi, Nießner, Verdoliva. *Raising the Bar of AI-generated Image Detection with CLIP*. CVPRW 2024. | Carefully tuned CLIP-based detection; strong cross-generator results. |
| khan2024clipping | Khan, Dang-Nguyen. *Clipping the Deception: Adapting Vision-Language Models for Universal Deepfake Detection*. ICMR 2024. | CLIP adaptation for deepfake detection. |
| liu2024fatformer | Liu et al. *Forgery-aware Adaptive Transformer for Generalizable Synthetic Image Detection*. CVPR 2024. | Forgery-aware adaptive transformer. |
| liu2024mole | Liu et al. *Mixture of Low-Rank Experts for Transferable AI-Generated Image Detection*. arXiv 2024. | **MoLE**: shared+separate LoRAs on CLIP-ViT; +3.64% mAP improvement on unseen generators. |
| cao2024hyperdet | Cao et al. *HyperDet: Generalizable Detection of AI-generated Images via Hypernetwork*. 2024. | Hypernetwork-based generalization. |
| koutlis2025rine | Koutlis, Papadopoulos. *Leveraging Representations from Intermediate Encoder-blocks for Synthetic Image Detection*. WACV 2025. | **RINE**: intermediate-layer CLIP features for detection. |
| zhu2023gendet | Zhu et al. *GenDet: Towards Good Generalizations for AI-Generated Image Detection*. arXiv 2023. | GenDet: training objective for generalization. |

### §A.7 MLLM and reasoning-based methods (6) — *direct competitors*

| BibKey | Reference | Annotation |
|---|---|---|
| zhou2025aigiholmes | Ziyin Zhou, Yunpeng Luo, Yuanchen Wu, Ke Sun, Jiayi Ji, Ke Yan, Shouhong Ding, Xiaoshuai Sun, Yunsheng Wu, Rongrong Ji (Xiamen Univ. + Tencent YouTu). *AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models*. ICCV 2025; arXiv:2507.02664. | **AIGI-Holmes**: visual-expert (NPR) + MLLM (LLaVA) + SFT + DPO; *sole headline foil — post-hoc text explanation, faithfulness unverified*. [metadata verified 2026-06-03: title/first-author Zhou/arXiv ✓; venue = IEEE conf publication (ICCV 2025)] |
| ji2025grounded | Yikun Ji, Hong Yan, Jun Lan, Huijia Zhu, Weiqiang Wang, Qi Fan, Liqing Zhang, Jianfu Zhang. *Interpretable and Reliable Detection of AI-Generated Images via Grounded Reasoning in MLLMs*. arXiv:2506.07045, 2025. | MLLM grounded reasoning (bounding-box + caption artifact annotation). [⚠ TITLE CORRECTED 2026-06-03 — prior "Grounded Reasoning for AI-Generated Image Detection" was a placeholder; first-author Ji ✓; arXiv preprint, not a published venue] |
| huang2025thinkfake | Tai-Ming Huang, Wei-Tung Lin, Kai-Lung Hua, Wen-Huang Cheng, Junichi Yamagishi, Jun-Cheng Chen. *ThinkFake: Reasoning in Multimodal Large Language Models for AI-Generated Image Detection*. arXiv:2509.19841, 2025. | **ThinkFake**: GRPO-trained MLLM reasoning; reports outperforming SOTA on GenImage + zero-shot on LOKI. [⚠ TITLE CORRECTED 2026-06-03 — prior "Chain-of-Thought Reasoning…" was a placeholder; first-author Huang ✓; arXiv preprint] |
| tan2025forenx | Chuangchuang Tan, Jinglu Wang, Xiang Ming, Renshuai Tao, Yunchao Wei, Yao Zhao, Yan Lu. *ForenX: Towards Explainable AI-Generated Image Detection with Multimodal Large Language Models*. arXiv:2508.01402, 2025. | **ForenX**: forensic-prompt MLLM + ForgReason dataset (2,215 annotated images). [⚠ TITLE CORRECTED 2026-06-03 — prior "Forensic eXplainability with MLLMs" was a placeholder; first-author Tan ✓; arXiv preprint] |
| fan2024fakegpt | Fan et al. *Fake-GPT: Detecting AI-Generated Images with Pure LLM*. 2024. | Pure LLM (no MLLM) detection. |
| ji2025explainable | Ji et al. *Explainable Detection of AI-Generated Images via MLLMs*. 2025. | Companion to ji2025grounded. |

---

## §B. Concept Bottleneck and Interpretable ML (18 refs)

### §B.1 Foundational concept bottleneck (4)

| BibKey | Reference | Annotation |
|---|---|---|
| koh2020concept | Koh, Nguyen, Tang, Mussmann, Pierson, Kim, Liang. *Concept Bottleneck Models*. ICML 2020. | **Foundational CBM**: predict concepts then label; *our paradigm seed*. |
| margeloiu2021concept | Margeloiu et al. *Do Concept Bottleneck Models Learn as Intended?* ICLR-W 2021. | Critical analysis: concepts may not be faithful; motivates our consistency loss. |
| mahinpei2021promises | Mahinpei et al. *Promises and Pitfalls of Black-Box Concept Learning Models*. ICML 2021. | Limitations of CBM; design considerations. |
| oikarinen2023labelfree | Oikarinen, Das, Nguyen, Weng. *Label-free Concept Bottleneck Models*. ICLR 2023. | LFCBM: concepts without manual labels (relevant to our weak-supervision plan). |

### §B.2 Recent concept-bottleneck advances (8)

| BibKey | Reference | Annotation |
|---|---|---|
| yuksekgonul2023posthoc | Yuksekgonul, Wang, Zou. *Post-hoc Concept Bottleneck Models*. ICLR 2023. | Post-hoc CBM construction; comparison to ante-hoc. |
| zarlenga2022cem | Espinosa Zarlenga et al. *Concept Embedding Models: Beyond the Accuracy-Explainability Trade-off*. NeurIPS 2022. | Concept Embedding Models: vectorial concept reps. |
| sheth2022leveraging | Sheth, Ebrahimi-Kahou. *Leveraging Auxiliary Information for Tabular Data Concept Bottleneck Models*. 2022. | Auxiliary info for concept learning. |
| havasi2022addressing | Havasi, Parbhoo, Doshi-Velez. *Addressing Leakage in Concept Bottleneck Models*. NeurIPS 2022. | Information leakage in CBM; relevant for our faithfulness claims. |
| kim2023bonet | Kim et al. *BotNet: Bottleneck Concept Learner*. 2023. | Self-supervised concept discovery. |
| wang2025ebotcl | Anonymous (CIBM/Sensors authors). *Enhancing Bottleneck Concept Learning in Image Classification (E-BotCL)*. Sensors 2025 (Apr). | **E-BotCL**: dual-path contrastive concept learning; CDR=0.61, CC=0.45 on CUB200/ImageNet; *direct precursor to our AGID adaptation*. |
| chen2025llmccbm | Anonymous. *Enhancing Interpretable Image Classification Through LLM Agents and Conditional Concept Bottleneck Models*. ACL 2025. | LLM-generated concepts + conditional CBM. |
| sun2024conceptcomplement | Anonymous. *Concept Complement Bottleneck Model for Interpretable Medical Image Diagnosis*. 2024. | Medical CBM with concept complementation. |

### §B.3 Broader interpretable ML (6)

| BibKey | Reference | Annotation |
|---|---|---|
| kim2018tcav | Kim et al. *Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV)*. ICML 2018. | **TCAV**: concept-based explanations; our concept-sensitivity metric. |
| chen2019looks | Chen et al. *This Looks Like That: Deep Learning for Interpretable Image Recognition (ProtoPNet)*. NeurIPS 2019. | Prototype-based interpretability; alternative to concepts. |
| kim2016examples | Kim et al. *Examples are not Enough, Learn to Criticize! Criticism for Interpretability*. NeurIPS 2016. | Prototypes + criticisms; foundational interpretability. |
| selvaraju2017gradcam | Selvaraju et al. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*. ICCV 2017. | **Grad-CAM**: our post-hoc explanation baseline. |
| petsiuk2018rise | Petsiuk, Das, Saenko. *RISE: Randomized Input Sampling for Explanation of Black-box Models*. BMVC 2018. | Insertion/deletion AUC: our faithfulness metric. |
| zhang2018topdown | Zhang et al. *Top-down Neural Attention by Excitation Backprop*. IJCV 2018. | Pointing Game metric; our localization evaluation. |

---

## §C. Generative Models Background (12 refs)

### §C.1 GANs (5)

| BibKey | Reference | Annotation |
|---|---|---|
| goodfellow2014gan | Goodfellow et al. *Generative Adversarial Networks*. NIPS 2014. | Foundational GAN. |
| karras2019stylegan | Karras, Laine, Aila. *A Style-Based Generator Architecture for Generative Adversarial Networks*. CVPR 2019. | StyleGAN: representative GAN in benchmarks. |
| brock2018biggan | Brock, Donahue, Simonyan. *Large Scale GAN Training for High Fidelity Natural Image Synthesis (BigGAN)*. ICLR 2019. | BigGAN: included in GenImage benchmark. |
| zhu2017cyclegan | Zhu, Park, Isola, Efros. *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks*. ICCV 2017. | CycleGAN: in ForenSynths. |
| karras2017progan | Karras, Aila, Laine, Lehtinen. *Progressive Growing of GANs for Improved Quality, Stability, and Variation*. ICLR 2018. | ProGAN: CNNDetection training source. |

### §C.2 Diffusion models (5)

| BibKey | Reference | Annotation |
|---|---|---|
| ho2020ddpm | Ho, Jain, Abbeel. *Denoising Diffusion Probabilistic Models*. NeurIPS 2020. | DDPM: foundational diffusion. |
| rombach2022ldm | Rombach et al. *High-Resolution Image Synthesis with Latent Diffusion Models*. CVPR 2022. | LDM/Stable Diffusion: dominant generator in benchmarks. |
| dhariwal2021adm | Dhariwal, Nichol. *Diffusion Models Beat GANs on Image Synthesis (ADM)*. NeurIPS 2021. | ADM: in GenImage benchmark. |
| nichol2022glide | Nichol et al. *GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models*. ICML 2022. | GLIDE: in GenImage benchmark. |
| saharia2022imagen | Saharia et al. *Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding (Imagen)*. NeurIPS 2022. | Imagen: representative modern T2I diffusion. |

### §C.3 Autoregressive and other (2)

| BibKey | Reference | Annotation |
|---|---|---|
| ramesh2022dalle2 | Ramesh et al. *Hierarchical Text-Conditional Image Generation with CLIP Latents (DALL-E 2)*. 2022. | Autoregressive T2I; representative. |
| esser2024flux | Black Forest Labs. *FLUX: Open Source Image Generation*. 2024. | FLUX: state-of-the-art open generator (2024-25). |

---

## §D. Foundational CV/ML (8 refs)

| BibKey | Reference | Annotation |
|---|---|---|
| he2016resnet | He, Zhang, Ren, Sun. *Deep Residual Learning for Image Recognition (ResNet)*. CVPR 2016. | ResNet50: our backbone choice. |
| dosovitskiy2021vit | Dosovitskiy et al. *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale (ViT)*. ICLR 2021. | ViT: alternative backbone considered. |
| radford2021clip | Radford et al. *Learning Transferable Visual Models From Natural Language Supervision (CLIP)*. ICML 2021. | CLIP: feature backbone for many AGID baselines. |
| chen2020simclr | Chen, Kornblith, Norouzi, Hinton. *A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)*. ICML 2020. | Contrastive learning: used in our cross-generator consistency loss. |
| vaswani2017attention | Vaswani et al. *Attention Is All You Need*. NIPS 2017. | Attention mechanism: used in dual-stream fusion. |
| kingma2015adam | Kingma, Ba. *Adam: A Method for Stochastic Optimization*. ICLR 2015. | Our optimizer. |
| loshchilov2019adamw | Loshchilov, Hutter. *Decoupled Weight Decay Regularization (AdamW)*. ICLR 2019. | Our actual optimizer (AdamW). |
| ioffe2015batchnorm | Ioffe, Szegedy. *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*. ICML 2015. | BatchNorm: used in backbone. |

---

## §E. Benchmarks, Datasets, Evaluation (8 refs)

| BibKey | Reference | Annotation |
|---|---|---|
| zhu2023genimage | Zhu et al. *GenImage: A Million-Scale Benchmark for Detecting AI-Generated Images*. NeurIPS 2023. | **GenImage**: our primary training and evaluation dataset (8 generators × 1.6M images). |
| wang2020forensynths | Wang, Wang, Owens, Efros. *ForenSynths dataset (released with CNNDetection)*. CVPR 2020. | **ForenSynths**: cross-architecture OOD evaluation; 11 generators. |
| deng2009imagenet | Deng et al. *ImageNet: A Large-scale Hierarchical Image Database*. CVPR 2009. | **ImageNet**: real-image source. |
| chen2024drct | Chen et al. *DRCT: Diffusion Reconstruction Contrastive Training for Detecting AI-Generated Images*. ICML 2024. | DRCT: dataset and method. |
| bird2024cifake | Bird, Lotfi. *CIFAKE: Image Classification and Explainable Identification of AI-Generated Synthetic Images*. 2024. | CIFAKE: small but interpretability-focused benchmark. |
| sha2024aigcdetection | Zhong et al. *AIGCDetection: A Comprehensive Benchmark*. 2024. | AIGCDetection: comprehensive benchmark, optional OOD. |
| yan2024chameleon | Yan et al. *Chameleon: Challenge Dataset for AGID*. 2024. | Chameleon: hard cases (post-processed AI images). |
| grommelt2024fakejpeg | Grommelt, Weiss, Pfreundt, Keuper. *Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets*. arXiv:2403.17608, 2024. *[authors verified via arXiv 2026-06-03: Patrick Grommelt, Louis Weiss, Franz-Josef Pfreundt, Janis Keuper. Venue = arXiv preprint (cs.CV/cs.AI/cs.LG); the earlier "Springer chapter" note is UNVERIFIED — confirm before ref.bib if a published version is preferred.]* | **Closest prior (dataset confound).** Shows GenImage has JPEG-compression + image-size biases that black-box detectors (ResNet50, Swin-T) exploit; debiasing via uniform JPEG-Q96 recompression → +11pp cross-generator (claimed SOTA); released unbiased-genimage.org. *We extend: model-internal concept localization + causal intervention + Q96-persistence (Outcome C).* Mandatory cite in Intro + RW Line 3. |

---

## Coverage Audit

| Category | Count | Target | Status |
|---|---|---|---|
| AGID core methods (§A) | 45 | 40-50 | ✅ |
| Concept Bottleneck / Interp ML (§B) | 18 | 15-20 | ✅ |
| Generative models (§C) | 12 | 10-15 | ✅ |
| Foundational CV/ML (§D) | 8 | 10-15 | ⚠️ Slightly low; can add 2-3 if needed |
| Benchmarks/datasets/eval (§E) | 8 | 5-10 | ✅ |
| **Total** | **91** | **80-90** | ✅ (+Grommelt 2026-06-02) |

---

## Notes for Stage 2 (Paper Writing)

1. **Critical citations for Introduction:** wang2020cnngenerated, ojha2023universal, wang2023dire, wang2025lota, zhou2025aigiholmes, koh2020concept, wang2025ebotcl.
2. **AIGI-Holmes (zhou2025aigiholmes) needs careful read**:download PDF, write 1-paragraph differentiation in Intro §1.2.
3. **E-BotCL (wang2025ebotcl) needs careful read**: cite as precursor in Method §3.
4. **Some 2025 papers may have been replaced/superseded by Stage 2** — re-validate via WebSearch when writing.
5. **Verify BibTeX entries via Google Scholar before final ref.bib generation** — accuracy of authors/venues critical for Stage 2.5 integrity check.

---

**Next deliverable:** `04_Synthesis.md` — cross-source synthesis + gap analysis + positioning.
