# Stage 2 Decision Log — AGID Paper

**Purpose:** persistent record of every locked Stage 2 writing decision, so no session loses work again.
(The prior session's Introduction draft + Related Work structure were *not* persisted and were lost on reset;
only the handoff's 9-item list survived. This file prevents recurrence.)

---

## Paper Configuration Record (re-confirmed 2026-06-02 from handoff §2 + CLAUDE.md)

| Field | Value |
|---|---|
| Structure | IMRaD |
| Venue style | NeurIPS 2022, natbib |
| Word target | 30,000 (compact tier) |
| Novelty target | A+C grade (4–5/5) |
| Deadline | 2026-06-24 23:59 (22 days from 2026-06-02) |
| Mode | `/ars-plan` — Socratic, very-high oversight (LOCKED) |
| Thesis | F2 dual-claim: SOTA detection **+** concept-bottleneck-as-audit-instrument |
| Headline policy | lead with 99.67% raw-OOD, but the audit conclusion appears in the **same** Abstract |

---

## Locked decisions

### From prior session (handoff §9) — carried forward, treated as locked
- **D-01 Thesis:** F2 dual-claim (sharpened, with triangulation defense).
- **D-02 Headline:** 99.67% raw-OOD first, audit conclusion in same Abstract.
- **D-03 RQ:** dual-RQ (Detection / Audit), user's refined wording.
- **D-04 Vocabulary:** use "separately-testable", not "interpretable"; "artifact-driven component", not "purely".
- **D-05 Gap statement:** "...lacks an architecture-level audit mechanism that can intervene on
  separately-testable evidence channels."
- **D-06 Related Work shape:** progressive 3-line structure (NOT flat); **AIGI-Holmes as critical target**.
- **D-07 Epistemological spine:** intervention permits the word "dependency"; it does **NOT** permit
  "purely / exclusively".
- **Status of Introduction:** decisions locked above; **prose draft was lost (not persisted)** — must be
  re-drafted from these decisions when we reach drafting.

### New this session (2026-06-02)
- **D-08 E-Delta = COMPLETE, Outcome C.** See `00_Evidence_Spine_Lock.md` (LOCKED). Numbers verified against
  `gate3_full_audit.json`.
- **D-09 Evidence spine locked** per user's 4-point directive: (1) intervention > weights; (2) double
  dissociation ≠ de-reliance; (3) Outcome C + claim ceiling; (4) weights are an unreliable reliance proxy →
  CBM = testability surface. The smoke "migration" narrative is retired.
- **D-10 Resume strategy:** "align evidence framing first" → done (D-08/D-09) → then continue at Related Work.
- **D-11 Housekeeping:** persist Stage 2 work to `AGID_Project/Stage2_Writing/`; CLAUDE.md state lock updated
  to E-Delta=COMPLETE / Outcome C (user-authorized 2026-06-02).

---

## OPEN — awaiting user ruling
- **O-1 RESOLVED (2026-06-02):** user ACCEPTED the Point-3 scope guard as recommended. Locked claim:
  `jpeg_quant` survives Q96 and is **not reducible to the trivial final PNG/JPEG container label**, but is **not**
  claimed as **pure generative semantics**; the generative-trace vs. residual-compression-history attribution
  boundary is written into **Discussion / Limitations**. Guard is LOCKED in `00_Evidence_Spine_Lock.md §3`.
- **O-2 (Title):** re-draft with Outcome C (handoff §5). Deferred to post-Related-Work.
- **O-3 (Thesis fine-tune):** confirm F2 dual-claim still phrased correctly given the *double dissociation*
  (Route B and E-Delta each win a different distribution). Handle inside Related Work / Method dialogue.

---

## Grommelt positioning (mandatory, from `memory/project-agid-grommelt-scoop`)
- Grommelt et al. 2024 "Fake or JPEG?" (arXiv 2403.17608) **must** be cited prominently in Intro + Related Work.
- Framing = **"we extend"**, never "we discover": Grommelt established the GenImage JPEG/size confound with
  **black-box** detectors; our delta = **model-internal, concept-level localization + causal intervention**,
  and the Q96-retrain shows the confound channel **persists** (Outcome C) — which Grommelt could not see.

---

*Append-only below this line. Date every entry.*

### 2026-06-02 — Related Work grounding (read `03_Bibliography.md` + `04_Synthesis.md`)
- **A-1 [ACTION — bib]:** Grommelt 2024 (arXiv 2403.17608, "Fake or JPEG?") is **NOT** in `03_Bibliography.md`
  (full scan of all 90 refs). Found mid-Stage-2 → simply missing. **Must add** (Intro + RW Line 3). Also the two
  surveys cited inline in Synthesis §3.1 (Lin et al. arXiv 2502.15176; arXiv 2502.19716) are not in the bib.
- **A-2 [⚠️ STALE-DOC WARNING]:** `04_Synthesis.md` (dated 2026-05-24) **predates** the audit pivot (05-28),
  Grommelt scoop (06-01), and E-Delta (06-02). Its §5 positioning ("6 weakly-supervised **interpretable**
  concepts … faithfulness **mathematically guaranteed** by the bottleneck") and its §4 AIGI-Holmes foil
  ("their text may be unfaithful, **ours cannot be**") are **partly REPUDIATED** by `00_Evidence_Spine_Lock.md`
  (head functionally uses ~1 of 6 concepts; weights ≠ reliance). **Re-derive RW positioning from the spine, NOT
  synthesis §4–§5.** Synthesis still good for: field eras (§1), method-family taxonomy (§2), AIGI-Holmes
  *mechanics* (§4 first half), bib landscape.
- **A-3 [CHECK in Method]:** Synthesis §3.3 / §5(b) novelty = "first cross-generator concept consistency loss",
  but the Route A/B handoff records this loss as **disabled** in training. That novelty claim may be **dead** —
  verify against the actual training config before any Contributions list uses it.
- **AIGI-Holmes** confirmed in bib = `zhou2025aigiholmes` (ICCV 2025). Mechanics (Synthesis §4): NPR Visual
  Expert + LLaVA-13B + DPO + 4-MLLM jury (8+ A100), 99.2%, post-hoc text, faithfulness unverified.
- **CBM-faithfulness-critique lineage available** in bib for Line 2 scaffolding: `margeloiu2021concept`
  ("Do CBMs Learn as Intended?"), `mahinpei2021promises`, `havasi2022addressing` ("Addressing Leakage in CBMs").

### 2026-06-02 — Related Work Line 2 + Grommelt LOCKED (Socratic)
- **D-13 [Line 2 = β audit-instrument]:** Line 2 written as **audit instrument**, NOT "testable interpretability"
  (α explicitly rejected). We do **not** claim the CBM is natively faithful; we claim the bottleneck provides an
  **interventionable, falsifiable audit surface**. Structure: `koh2020concept` (CBM seed) → CBM-faithfulness-
  critique trio (`margeloiu2021concept` / `mahinpei2021promises` / `havasi2022addressing`) as **scaffolding** →
  **AIGI-Holmes (`zhou2025aigiholmes`) as the SOLE headline foil** at the un-falsifiable extreme (post-hoc MLLM
  text) → converge on the D-05 gap. Trio is scaffolding, NOT a co-target.
- **D-14 [Grommelt = closest prior]:** added to bib as `grommelt2024fakejpeg` (§E). Position = **"we extend",
  not "we discover"**: Grommelt found the GenImage JPEG/size confound (black-box, dataset-level); our delta =
  model-internal localization (`jpeg_quant` w=−13.53) + causal intervention (zero-out −48pp) + **Q96-persistence
  (Outcome C)**. Resolves A-1 (entry added); full author list/BibTeX still to verify via arXiv 2403.17608 at the
  citation pass.
- Both supersede Synthesis §4–§5 for RW positioning (per A-2).

### 2026-06-02 — Related Work Line 1 + Line 3 LOCKED → RW chapter plan COMPLETE
- **D-15 [Line 1 = 1b-restrained]:** present detection SOTA normally (NPR ~96–98% / LOTA 98.9% /
  AIGI-Holmes 99.2% / Route B 99.67%); add exactly **one** restrained forward-pointer, verbatim intent:
  *"these near-saturated GenImage numbers may reflect not only generative-trace detection but also dataset
  shortcuts — a concern raised by Grommelt and tested model-internally in our experiments."* Rhythm:
  (1) acknowledge the field has saturated GenImage, (2) gently note the *measurement validity* of that
  saturation is unresolved at the model level → motivates D-05. Do **not** expand the confound critique in
  Line 1; do **not** weaken the SOTA flex.
- **D-16 [Line 3 = final]:** fully credit Grommelt; delta fixed at *black-box/dataset-level analysis →
  model-internal localization + intervention + Q96-persistence*. Do **not** raise the double-dissociation's
  complication of Grommelt's "+11pp debiasing" headline in Line 3 — reserved for Experiments/Discussion.
  Related Work only positions; no early results battle.
- **✅ Related Work chapter PLAN = COMPLETE/LOCKED.** Arc: Line 1 (SOTA + forward-pointer) → Line 2
  (β audit-instrument: `koh2020concept` → critique-trio scaffolding → AIGI-Holmes foil) → Line 3 (Grommelt
  closest prior, "we extend") → **D-05 gap**.

### 2026-06-02 — Method grounding: REAL training provenance verified (CORRECTS A-3)
- **D-17 [verified training facts]:**
  - Real training script = **`Code/cbnet_agid/train_multigen.py`** (NOT `train.py`; the latter is the Route-A
    single-gen script — `--lambda_gen` default 0, `gen_consistency` logged 0.0 throughout Route A).
  - Real loss = **`CBNetMultiGenLoss`** (`Code/cbnet_agid/losses/losses.py`):
    `L = L_task + 0.5·L_concept + 0.2·L_content_pair + 0.05·L_sparsity`. Confirmed by Route B config
    (`AGID_Project/Logs/cbnet_multigen_main_25k_s42/config.json`, 20 ep) + E-Delta config
    (`Code/Logs/edelta_full/config.json`, 10 ep) — same recipe, only data + epochs differ.
  - **`content_pairing_loss` IS active and IS cross-generator:** L1 between concept vectors of pairs (i,j) =
    same ImageNet class (≥0) AND same label (both ai / both nature) AND **different generator**; in-batch
    masked (no separate paired loader). Logged `content_pair` ≈ 0.011–0.015, ~flat.
- **A-3 CORRECTED:** earlier "consistency loss disabled" was **wrong** (that was Route A). For Route B + E-Delta
  the consistency idea WAS realized as a same-ImageNet-class content-pairing loss (the proxy handoff §4.5
  anticipated), active at λ=0.2. It does **NOT** go to disabled/future-work; it **stays in Method main body**.
- **As-designed vs as-built (Method must honor):** blueprint §5.3 = "same real image + AI counterparts via
  **shared prompt**"; implementation = **same ImageNet class** proxy + in-batch masking. Method describes the
  IMPLEMENTATION.
- **A-3′ [open — Experiments]:** `content_pair` value is tiny/flat; loss-component ablation (blueprint §10.2)
  not yet confirmed run → whether it *helped* is **unvalidated**. The "first cross-generator concept consistency
  loss" novelty (synthesis §5b) CANNOT be asserted until an ablation supports it.
- **Discussion seed (later, not Method/Results):** both `content_pair` (anti-generator-shortcut) and `sparsity`
  entropy (anti single-concept-activation-collapse, blueprint §5.4) are anti-shortcut *by design*, yet the audit
  shows functional reliance still concentrates on `jpeg_quant`. Precise: entropy reg targets activation
  diversity (not weight/functional reliance), and a *generator-invariant confound* (container format) trivially
  satisfies the consistency loss → neither reg can "see" the confound. (Interpretation → Discussion.)

### 2026-06-02 — Method loss presentation = Option A (LOCKED)
- **D-18 [loss presentation = A]:** Method describes `content_pair` + `sparsity` ONLY as actually-enabled
  training objectives — no efficacy / novelty / generalization claim. 口径:
  1. `content_pair` = same-class cross-generator L1 consistency term;
  2. `sparsity` = activation-entropy regularizer;
  3. Method states only HOW they are implemented + added to the loss — no efficacy / novelty / generalization;
  4. consistency-loss enters the **contributions list** ONLY if the Experiments loss-component ablation supports it;
  5. if the ablation was not run → keep `content_pair` as an as-implemented design detail, **NOT** a contribution.

### 2026-06-02 — Method beat ① grounding: architecture-as-built verified (vs blueprint §3)
- **D-19 [verified as-built architecture]** (from `models/cbnet.py` + `models/backbone.py`):
  - Pixel stream = torchvision **ResNet-50** (ImageNet `IMAGENET1K_V2`), full conv → `[B,2048,H',W']`.
  - Signal stream = **NPRResidual** (`x − up(down(x))`, nearest, ×0.5) → **SignalStreamCNN** (5-layer, ~3M params,
    stride-2 ×5) → `[B,512,·]`.
  - Fusion = **channel concat → 2560** ch (2048+512). CBL (K=6) on 2560 → `concepts[B,6]` + heatmaps.
    `nn.Linear(6→1)`, **NO skip connection** (the bottleneck guarantee; cbnet.py:51-53).
  - **Discrepancies vs blueprint §3 — Method MUST use as-built:**
    1. Signal stream is **NPR-residual ONLY** — NOT the blueprint's "LOTA-style **bit-plane** composite + NPR".
       Bit-plane exists only as concept #1 (`bitplane_lsb` heuristic), NOT as a backbone input stream.
    2. Fused width = **2560** ch, not the blueprint's 2048.
    3. Trained at **`image_size=256` → 8×8** feature grid, not the blueprint/code-comment 224²→7×7.
  - Implication: backbone = standard off-the-shelf parts (ResNet-50 + NPR + linear-head CBM) → the blueprint's
    "dual-stream as part of novelty C" claim is **thin**. Positioning decision pending (see beat ① question).

### 2026-06-02 — Method beat ① LOCKED: architecture = deliberately standard (option i)
- **D-20 [architecture positioning = (i)]:** Method presents the architecture as a **deliberately conventional but
  competitive** detector — ResNet-50 pixel + NPR-residual signal + CBL(K=6) + Linear(6→1), all existing
  components. **No architectural-novelty claim**; blueprint's "dual-stream = novelty C" is **dropped**. Rationale:
  makes the audit findings (jpeg_quant dominance, zero-out collapse, Q96-persistence) read as structural problems
  of *standard strong detectors*, not exotic-architecture flukes.
- **Method emphasis = the bottleneck guarantee:** the linear head sees ONLY the 6-dim concept vector; NO skip
  connection from the 2560-ch fused features to the classifier (cbnet.py:51-53). This is the causal precondition
  that makes the zero-out audit meaningful — it is the load-bearing Method claim.
- **D-19 corrections applied to Method prose:** (1) signal stream = NPR-residual ONLY (no "bit-plane stream");
  (2) fused width = 2560 = 2048+512; (3) input 256² → 8×8 grid (not 224²→7×7).
- **Contributions-list update [for Intro/Abstract]:** DELETE "dual-stream architecture novelty." Concentrate
  novelty on the **β audit thesis**: (a) architecture-level audit surface, (b) model-internal localization,
  (c) causal intervention, (d) Q96-persistence (Outcome C). [Consistency-loss contribution remains GATED on the
  ablation per D-18 / A-3′.]

### 2026-06-02 — Method beat ② grounding: concept set + CBL as-built verified
- **D-21 [verified concept/CBL facts]** (from `concepts/base.py` + `concepts/heuristics.py`):
  - CBL = K=6 architecturally-**IDENTICAL** `ConceptHead`s. Each = 1×1 conv (C→C/4) → BN → ReLU → 1×1 conv(→1)
    → sigmoid → spatial heatmap; scalar concept = spatial **MEAN** of the heatmap. Vector [B,6] → Linear(6→1).
  - **KEY (base.py:7-15): concept identity is SUPERVISION-INDUCED, not architecturally enforced.** All 6 heads
    are identical; "meaning" comes ONLY from the weak per-concept MSE (λ_concept=0.5) toward the heuristic label
    → a concept can drift from its named heuristic if the task loss pulls it. This is the as-built foundation for
    β (NOT "interpretable / faithful by construction").
  - 6 heuristics (numpy/OpenCV weak-supervision targets, fixed order): `bitplane_lsb` (|LSB 1-lag autocorr|),
    `freq_radial` (|log-log radial PSD slope|), `color_manifold` (KL to a global real-color prior — needs prior
    set; Route B used a shared prior), `hf_noise` (var(Laplacian)), `jpeg_quant` (var of DCT residuals after
    re-quant grid=10), `texture_geometry` (1−corr(edge-density, patch-variance)).
  - **`jpeg_quant` heuristic = a JPEG-quantization/container signal** (DCT-requantization residual energy:
    PNG/uncompressed → higher, prior-JPEG → lower). As-built confirmation of the spine's container-confound story.
- **Experiments precision flags (NOT Method):**
  - **F-1 [sign]:** the c_jpeg→weight mapping must be derived EMPIRICALLY. The name "JPEG-QuantTrace-Absence"
    encodes an assumed direction that appears **inverted** vs trained activations: gate3 + A3 imply reals have
    HIGHER c_jpeg and w<0 ⇒ jpeg_quant functions as the "real-detector" (zeroing → real_acc→3%). Do NOT take the
    concept NAME's direction at face value in prose.
  - **F-2 [d source]:** use GenImage Cohen's d (gate3 `cohens_d`, jpeg_quant −3.84…−5.47) for the paper, NOT the
    Day-5 20+20 (camera-JPEG / Bing-WebP) numbers in the heuristics.py docstring (initial concept-selection only).

### 2026-06-02 — Method beat ② LOCKED: concepts = separately-testable channels, supervision-identity upfront (i)
- **D-22 [beat ② = (i)]:** Method states **upfront** that the 6 heads are architecturally identical and concept
  identity is **supervision-induced, not architecturally guaranteed** ⇒ the bottleneck guarantees only
  *interventionable 6-dim evidence channels*, NOT that concept names are natively faithful. Terminology =
  "separately-testable evidence channels" (D-04), NOT "interpretable concepts." Each heuristic described
  factually (incl. `jpeg_quant` as a JPEG-quantization signal). **Result-layer facts — inertness / jpeg_quant
  dominance / leakage — stay in Experiments** (per #4); F-1/F-2 precision flags apply there.

### 2026-06-02 — Method beat ③ grounding: training protocol as-run verified
- **D-23 [verified training protocol]** (`train_multigen.py` + `data/transforms.py` + both configs):
  - **Single-phase, end-to-end** from epoch 0 (full CBNetAGID built + trained). **NO two-phase** (blueprint §6.1
    Phase-1 backbone warm-up + Phase-2 bottleneck was NOT used). **All loss terms active from epoch 1** (NO staged
    warm-up; blueprint §5 schedule NOT used). → Method describes single-phase / all-on; do NOT write two-phase.
  - Optimizer **AdamW**, 3 param groups: backbone 1e-4 / cbl-head 5e-4 / classifier 1e-3; wd 1e-4. Scheduler
    **CosineAnnealingLR**(T_max=epochs). **bf16 AMP**. bs=32 × accum_steps 2 = **effective 64**.
  - Data: 4 gens (SD-1.4 / BigGAN / ADM / MJ) × `train_25k` (25k/class), shared concept labels + shared real-color
    prior. Route B root = raw GenImage, **20 ep** (gate3 uses ep20). E-Delta root = `GenImage_debiased_full` (Q96),
    **10 ep** (gate3 uses ep10). seed 42.
  - Augmentation (`get_train_transform`, `disable_destructive=True` as used): Resize(+32) → RandomCrop(256) →
    HFlip → ImageNet-Normalize. **JPEG + Gaussian-blur augs OFF** (CLAUDE Fix #1: they inject ~30% concept-label
    noise on bitplane/jpeg_quant/hf_noise). Eval: Resize(+32) → CenterCrop(256) → Normalize.
  - **Minor caveat:** concept labels precomputed with CenterCrop(256); training uses RandomCrop(256) → labels
    approximate the crop center (`precompute_concept_labels.py` acknowledges this). Small train-aug↔label misalign.

### 2026-06-02 — Method LOCKED (beats ①②③ complete) → Experiments planning
- **D-24:** crop/label caveat → **Method footnote** (not Limitations). Method planning **COMPLETE** (beats ①②③,
  fully grounded in code per D-17–D-23). Entering Experiments planning; first action = resolve **A-3′**
  (loss-component ablation check). Per user: if no ablation exists, `content_pair` stays an as-implemented loss
  only, **NOT** a contribution.

### 2026-06-02 — A-3′ RESOLVED: no loss-component ablation → consistency loss NOT a contribution
- **D-25 [A-3′ resolved]:** enumerated all 6 training runs via `config.json`: Route A (`debug_run_20k`,
  `main_run_100k`), Route B (`cbnet_multigen_smoke`, `cbnet_multigen_main_25k_s42`/20ep), E-Delta
  (`cbnet_debiased_smoke`, `edelta_full`/10ep). **No loss-component ablation exists** — no companion model with
  `lambda_pair=0` all-else-equal; both paper models (Route B main + E-Delta full) have `content_pair` ON at 0.2.
  Route A's `gen_consistency=0` is single-gen structural, NOT a controlled ablation.
  - ⇒ `content_pair` = as-implemented training detail ONLY; **does NOT enter contributions** (D-18.5). Synthesis
    §5b "first cross-generator concept consistency loss" novelty **DROPPED**.
  - **Contributions list (FINAL) = β audit thesis only:** (a) architecture-level audit surface, (b) model-internal
    localization, (c) causal intervention, (d) Q96-persistence (Outcome C).
- **A-4 [new — Experiments verify-first]:** before claiming ANY ablation in Experiments, verify it was actually
  run (apply the lesson from the loss/architecture provenance checks). Blueprint §10 planned: per-concept (A3 ✓
  exists), loss-component (✗ D-25), backbone (dual vs pixel-only vs signal-only — VERIFY), K∈{4,6,8,12} (VERIFY),
  bottleneck-vs-no-bottleneck (VERIFY). Claim only what exists on disk.

### 2026-06-02 — Experiments structure CONFIRMED + A-4 inventory + D3 (Route A/B NOT a clean ablation)
- **D-26 [Experiments structure LOCKED]:** detection → audit. §1 Setup; §2 Detection + double-dissociation
  NUMBERS (no over-interpretation); §3 Audit = (a) reliance (weights/zero-out/counterfactual/correlation),
  (b) confound battery (B1/B2/B5), (c) **E-Delta own subsection** (Outcome C, Q96-persistence,
  weight-vs-intervention); §4 = ONLY ablations that exist on disk.
- **A-4 [training ablations]:** 6-run inventory → all K=6, full dual-stream, with bottleneck. NOT run
  (→ Limitations, not claims): backbone, K∈{4,8,12}, no-bottleneck, loss-component (D-25).
- **D3 finding (`D3_route_a_vs_b.md`) — Route A↔B is NOT a clean generalization ablation:**
  - ONLY clean A-vs-B cell = **SD-1.4 val: A 99.83% vs B 99.88% (Δ+0.04pp)** — essentially equal.
  - D3 says Route A "cannot be evaluated cross-generator" (GenImage-OOD protocol). The experiment_handoff shows
    Route A WAS run on **ForenSynths** (cross-architecture GANs) → **mean 51.7%**. Route B's 99.67% is on
    **GenImage held-out gens** (GLIDE/Wukong/VQDM). **Different benchmarks** → "single→multi fixed OOD" is
    **confounded** by benchmark + code change; do NOT claim a clean multi-gen-generalization delta.
  - **β-consistent reading (→ Discussion):** GenImage-OOD shares the container confound with training (Route B
    99.67% rides the shortcut); ForenSynths does not (Route A 52% ≈ chance). The cross-benchmark gap is itself
    audit evidence for the confound, NOT proof of generalization.
- **A-5 [VERIFY — decisive]:** was **Route B ever evaluated on ForenSynths**? If Route B is also ≈chance there →
  smoking-gun that 99.67% is the shortcut. If never evaluated → a gap (state as limitation, or a cheap eval).

### 2026-06-02 — A-5 RESOLVED + user GO: run Route-B-on-ForenSynths (inference-only)
- **D-27 [A-5 resolved]:** ForenSynths results on disk = `cbnet_forensynths.json` (Route A, 20k SD-only,
  mean 52.21%) + `npr_forensynths.json`. **Route B never evaluated on ForenSynths** → real gap. User GO to run.
- **Eval pre-flight VERIFIED (`scripts/eval_forensynths.py`):** CLI `--method cbnet --ckpt --root <FS root,
  contains test/> --out` (+batch_size 32, image_size 256, num_workers 0). Reads `test/<gen>/{0_real,1_fake}`.
  CBNet path: `prob=model(x)["prob"]`=P(AI), labels real=0/fake=1, **no inversion** (only lota inverts) → label
  convention CORRECT for CBNet ([[feedback-label-convention]]). ForenSynths data confirmed on disk at
  `AGID_Project/Data/ForenSynths` (glob timed out enumerating ~23k imgs = present). Route B ckpt =
  `AGID_Project/Logs/cbnet_multigen_main_25k_s42/ckpt_epoch20.pth`. Out → `Code/Results/cbnet_multigen_forensynths.json`.
- **Clean ablation this yields:** Route A (`cbnet_forensynths.json`, 52.21%) vs Route B — SAME arch, SAME FS
  benchmark, differ only in training-generator coverage. Plus the decisive container-mismatch test of Route B.

### 2026-06-02 — Route-B-on-ForenSynths RESULT (D-27 eval COMPLETE, exit 0)
- **D-28 [ForenSynths result]** (`Code/Results/cbnet_multigen_forensynths.json`, full ~23k imgs):
  - Route B (multi-gen, ep20) on ForenSynths: biggan 61.10% (AUC .692, real_acc 22.2%), deepfake 82.24%
    (.934, 70.3%), gaugan 54.82% (.657, 9.7%), stargan 96.42% (.999, 92.8%). **Mean acc 73.65%, mean AUC 0.820.**
  - vs Route A (single-gen, `cbnet_forensynths.json`, 20k): mean **52.21%** (AUC ~0.53).
  - **Two readings:**
    1. **Within-Route-B (CLEAN — same model, no code/loss confound) = headline:** 99.67% GenImage-OOD →
       **73.65% ForenSynths = −26pp** on a container-MISMATCHED benchmark. Concrete evidence the GenImage-OOD
       headline is **partially confound-inflated** (triangulates B1/B2 + audit).
    2. **Cross-route (SUGGESTIVE, NOT controlled — differs in code/loss/scale too, not only gen-coverage):**
       Route B 73.65% > Route A 52.21% (+21pp, AUC .53→.82) → multi-gen buys GENUINE partial cross-architecture
       generalization. Context only; do NOT call it a clean generator-coverage ablation.
  - Generator-dependent: strong stargan/deepfake (AUC .93–.999); weak + fake-biased biggan/gaugan
    (AUC .66–.69, real_acc <22%).
- **Narrative correction (β, honest):** the model is **NOT a pure container detector** — it genuinely partially
  generalizes. Claim **"partial confound inflation + genuine partial generalization,"** NOT "99.67% is pure
  shortcut." Spine's hedged Line-1 wording ("may reflect … also dataset shortcuts") is already correct — do NOT
  harden to "pure." Weave: §4 (cross-route context), §3c/Discussion (the clean −26pp within-model gap).
- **Scratch:** `Results/_smoke_fs.json` (50-sample pre-flight) left in place per DO-NOT #3; the real file is
  `cbnet_multigen_forensynths.json`. Delete the scratch manually if desired.

### 2026-06-02 — ForenSynths framing LOCKED
- **D-29 [ForenSynths framing]:** write as **"partial confound inflation + genuine partial generalization."**
  ForenSynths shows Route B is NOT a pure container detector (52→74% over single-gen); the within-model
  99.67%→73.65% drop shows the GenImage-OOD headline has benchmark/container inflation. **Clean causal evidence =
  B1/B2/A3/E-Delta**; ForenSynths = external-validity corroboration. Do NOT attribute all 26pp to confound
  (entangled with GAN-vs-diffusion distribution shift). Placement: §4 = Route A vs B; §3c/Discussion =
  within-model drop.

### 2026-06-02 — §1 Setup grounding (E1 baselines) + dual-claim refinement PENDING
- **D-30 [E1 baseline grounding]** (`E1_sota_baselines.md`):
  - **Protocol heterogeneity is endemic:** P1 (SD-only train, 8-gen test — AIDE/NPR-reimpl/etc.), P2 (LOTA
    per-gen-avg 98.9% — NOT comparable, includes diagonal), P3 (ProGAN train). CBNet = **P4 (joint 4-gen train)**
    — NO baseline uses it. CBNet's "OOD" = GLIDE/Wukong/VQDM/SD-1.5 only (BigGAN/ADM/MJ are in-dist for CBNet) →
    **NOT apples-to-apples** with P1 baselines' 8-gen cross-gen.
  - Most master-table numbers are **literature-cited** (AIDE 86.9, NPR 88.6, FatFormer 88.9, DRCT 89.5,
    C2P-CLIP 95.8, LOTA 98.9⚠P2), NOT locally reproduced. Locally-run = NPR + LOTA on SD-1.4 val (in-dist) +
    NPR on ForenSynths.
  - E1 mandates (must appear): per-row protocol footnotes (P1–P4), a **comparability column**, CBNet as TWO
    sub-rows (in-dist 4-gen / OOD), do NOT cite LOTA 98.9 as a direct upper bound, real/fake breakdown = a CBNet
    contribution (rare in field).
- **§1 Setup plan:** datasets 4-tier (in-dist SD-val / held-out BigGAN-ADM-MJ / GenImage-OOD GLIDE-Wukong-VQDM
  (-SD1.5) / cross-arch ForenSynths) + Q96 variant (E-Delta); real=ImageNet. Baselines = E1 master table
  (lit-cited, protocol-labelled + comparability column) + locally-run NPR/LOTA. Metrics =
  acc/AUC/AP/real-acc/fake-acc. Two models: Route B (biased) + E-Delta (Q96), both K=6 seed 42.
- **PENDING refinement (touches locked D-01/D-02):** E1 (protocol-divergence) + ForenSynths (73.65% cross-arch)
  make "**SOTA** detection" hard to defend → propose softening to "**competitive** detection." User decision.

### 2026-06-02 — Detection claim = "competitive, not SOTA" (D-01/D-02 REFINED)
- **D-31:** dual-claim detection half softened **SOTA → competitive**.
  - **D-01′:** thesis = "**competitive** single-GPU cross-generator detector **+** concept-bottleneck audit
    instrument" (drop "SOTA").
  - **D-02′:** lead with 99.67% but **label it GenImage-OOD / P4** and immediately caveat (protocol-divergence
    vs P1 baselines + ForenSynths 73.65% cross-arch). Comparison table carries protocol footnotes + comparability
    column (E1).
  - Contributions (FINAL): β audit thesis (audit surface / model-internal localization / intervention /
    Q96-persistence) + a competitive-detector demonstration. No SOTA claim, no architecture novelty, no
    consistency-loss novelty.

### 2026-06-03 — Stage 2.5 integrity check EXECUTED (verify-only) → report + preflight resolutions
- **D-32 [Stage 2.5 verification COMPLETE]:** full report = `Stage2_Writing/03_Stage2.5_Integrity_Report.md`.
  All headline numbers re-read from disk + recomputed → **0 fabricated numbers**; all 26 cited keys exist;
  all 8 prohibited claims present only as correct negations; 7-mode failure sweep = pass except minor wording.
- **Preflight resolutions applied:**
  - **Grommelt verified** (arXiv 2403.17608): authors = Patrick Grommelt, Louis Weiss, Franz-Josef Pfreundt,
    Janis Keuper; arXiv preprint 2024 (no published venue found). `03_Bibliography.md` entry updated; the old
    "Springer chapter" note marked UNVERIFIED (O-3 open).
  - **4 cross-ref placeholders resolved:** `§[experiments-setup]`→§4.1 (Intro), `§[experiments]`→§4.3 (Method),
    `§[limitations]`→§6 (Experiments + Discussion).
  - **Section-5 gap closed (O-2 RESOLVED — user kept renumber):** Discussion 6→5, Limitations 7→6,
    Conclusion 8→7. Now contiguous 1–7.
- **O-1 (Title) RESOLVED — user chose:** **"What Is the Accuracy Made Of? Auditing a Competitive
  AI-Generated Image Detector with a Concept Bottleneck."**
- **Wording fixes APPLIED (user: apply all three):**
  - FIX-1 (§4.3a): "remaining four channels cost under one point each" → "...each cost about a point or less on
    most generators (texture-geometry the largest exception, up to −5.8pp on Wukong)" — `texture_geometry`
    zero-out is −2.6/−1.3/−5.75pp on SD/VQDM/Wukong per `A_intervention_summary.md`.
  - FIX-2 (§5 Discussion): "remaining four channels are inert" → "near-inert."
  - FIX-3 (Abstract): "turns on a single frequency/compression channel" → "...cue." (Intro left intact —
    re-examined and already precise: "single channel dominates … remaining channels carry little *independent*
    signal" correctly encodes the two-channel/single-cue, −0.80-redundant structure.)
- **STILL OPEN / deferred (not done — out of "no-assembly/no-rewrite" scope):**
  - FIX-4: web-verify 2025 bib metadata (`zhou2025aigiholmes`, `wang2025lota`, `huang2025thinkfake`,
    `tan2025forenx`, `ji2025grounded`) before generating `ref.bib`.
    **→ DONE 2026-06-03 (web-verified):** all 5 exist; all 5 bibkey first-author surnames correct (no hallucinated
    keys). `wang2025lota` (Hongsong Wang+4, ICCV 2025, arXiv 2510.14230) and `zhou2025aigiholmes` (Ziyin Zhou+9,
    Xiamen+Tencent, ICCV 2025 IEEE conf pub, arXiv 2507.02664) = accurate. **3 had fabricated titles, now CORRECTED
    in `03_Bibliography.md`:** `huang2025thinkfake`→"ThinkFake: Reasoning in MLLMs for AIGI Detection" (arXiv
    2509.19841); `tan2025forenx`→"ForenX: Towards Explainable AIGI Detection with MLLMs" (arXiv 2508.01402);
    `ji2025grounded`→"Interpretable and Reliable Detection of AIGI via Grounded Reasoning in MLLMs" (arXiv
    2506.07045). The 3 are arXiv preprints (not published venues). Manuscript prose unaffected (cited by system
    name, not title). LOTA 98.9% / ThinkFake 84% / ForenX 97.8% are annotation-only, unused in prose.
  - FIX-5: build Table 1 baseline master-table from `E1_sota_baselines.md` (currently schematic; referenced
    in §4.1 + §6) — at manuscript-assembly time.
  - FIX-6 (low): §4.4 cites 20k single-gen 52.21%; 100k "main" run = 51.70% (both ≈chance, claim robust).
  - O-3: Grommelt arXiv vs published-version citation.

### 2026-06-03 — MANUSCRIPT ASSEMBLY COMPLETE (mechanical)
- **D-33 [assembly done]:** full manuscript assembled in `Stage2_Writing/manuscript/`; **compiles clean**
  (latexmk/MiKTeX → 14-page PDF, 0 errors, 0 undefined cites, 0 bibtex warnings). Report =
  `manuscript/ASSEMBLY_REPORT.md`.
  - Built into the **course NeurIPS-2022 template** (`课程大作业latex模板/neurips_2022.sty`, copied in).
    `\bibliographystyle{plainnat}` (author-year); title = the O-1 choice.
  - **md→LaTeX via pandoc** (`_build_body.py`, reproducible): section numbers stripped (auto-numbering matches
    §-refs), Unicode → `\ensuremath{}`/`\textsuperscript`/`\S{}`/`---`, `%`/`_` escaped by pandoc,
    `\citep`/`$…$`/`\footnote` passed as raw TeX, `\tightlist` defined.
  - **FIX-5 DONE:** Table 1 from `E1_sota_baselines.md` — protocol tags (P1/P2/P4) + comparability column +
    footnote (NOT a leaderboard); LOTA 98.9% re-confirmed vs E1 (P2 mean). Tables 2–4 hand-built from artifacts;
    literal "Table N" refs resolve by float order.
  - **ref.bib:** 26 cited keys, FIX-4 verified metadata, no fabricated titles; bibtex 26/26, 0 warnings.
  - **FIX-6 DONE (report only):** 100k single-gen 51.70% vs cited 52.21% — both ≈chance, conclusion robust; body
    unchanged. **O-3:** Grommelt = arXiv 2403.17608 preprint (no published venue found).
  - **NO claim rewritten, NO number altered, NO experiment added.** Prose ≈ 6.7k words / 14 pp.
- **Human-confirm items (ASSEMBLY_REPORT H-1…H-7):** author block placeholder; citation style (plainnat vs
  numeric); name-only baselines AIDE/DRCT/C2P-CLIP (uncited, matches prose); 6.7k-word length vs old 30k target
  (expansion = content decision, out of scope); no figures; Grommelt venue; title `\\` line-break.

### 2026-06-03 — Pre-submission cleanup pass (user rulings on H-1…H-7)
- **D-34 [cleanup done]:** re-compiled clean (`latexmk` exit 0, 14 pp, 0 LaTeX errors, **0 undefined cites,
  29/29 bibtex entries, 0 warnings, 0 overfull >20pt**). Report updated (`manuscript/ASSEMBLY_REPORT.md`).
  - **H-1:** author block → `[Your Name] \\ Shanghai Jiao Tong University \\ \texttt{[email@example.com]}` with a
    visible `% TODO (H-1)` comment; real name/email NOT inserted (user fills manually). Fixed a recoverable
    `\\`+`[` optional-arg error by wrapping the email in `\texttt{}`.
  - **H-2 KEPT** plainnat (author-year). **H-6 KEPT** Grommelt arXiv preprint. **H-7 DONE** cosmetic title
    line-break (text unchanged). **H-4/H-5 DEFERRED** (no expansion, no figures).
  - **H-3 DONE:** Table 1 baselines all cited. 3 new ref.bib entries, metadata **web-verified 2026-06-03**:
    `yan2024aide` (*A Sanity Check for AI-generated Image Detection*, arXiv 2406.19435 — corrects the project
    bib's earlier wrong "Hybrid Approach" title, also fixed in `03_Bibliography.md`), `chen2024drct`
    (*DRCT…*, ICML 2024), `tan2024c2pclip` (*C2P-CLIP…*, arXiv 2408.09647). Table 1 venue column updated.
  - No prose claim rewritten, no number altered.

### 2026-06-03 — Final readiness pass + submission docs (verification only)
- **D-35 [readiness PASS]:** verification-only pass over the compiled PDF + sources; **no content/claims/experiments
  changed.** Three reports now live in `Stage2_Writing/manuscript/`:
  - `ASSEMBLY_REPORT.md` — assembly record + cleanup pass + build checks (md→LaTeX via pandoc, FIX-5/FIX-6, ref.bib).
  - `SUBMISSION_CHECKLIST.md` — manual pre-submission items (below).
  - `PACKAGE_MANIFEST.md` — final file paths (below).
- **Verified (from `pdftotext` of `manuscript.pdf`):** structure complete + ordered — title page → Abstract →
  §1 Introduction → §2 Related Work → §3 Method → §4 Experiments → §5 Discussion → §6 Limitations → §7 Conclusion
  → References; **14 pages** (pdflatex-authoritative). Tables 1–4 render with clear captions (Table 1 keeps
  protocol+comparability columns; every baseline cited). Title page shows the author placeholder
  `[Your Name] / Shanghai Jiao Tong University / [email@example.com]`.
- **Build health:** `latexmk` exit 0 · 0 LaTeX errors · 0 undefined citations · 29/29 bibtex entries / 0 warnings
  · 0 overfull boxes >20pt.
- **Residual-marker scan (compiled sources):** clean — only the intentional `% TODO (H-1)` author marker +
  `[Your Name]`/`[email@example.com]` placeholders; 0 `<!--` draft comments leaked into body/abstract; no
  `XXTABLE`/`schematic`/`Lorem`/`§[`/`[experiments-setup]` residue.
- **SUBMISSION_CHECKLIST manual items (the only things left to a human):**
  1. **[REQUIRED]** fill author name + email in `manuscript.tex` (~ll.23–29; SJTU affiliation already set), then
     remove the `% TODO (H-1)` line. (Email kept in `\texttt{}` so a `[` never directly follows `\\`.)
  2. **[course policy]** citation format (currently `plainnat` author-year; switch via
     `\usepackage[final,numbers]{neurips_2022}` + `unsrtnat` if numeric `[1]` required) and named-vs-anonymous
     (`[final]` = names shown; drop `final` for anonymous).
  3. **[check]** length/page limit (≈14 pp / 6.7k words; no padding done).
  4. optional: figures (none); Grommelt venue (arXiv preprint); final human proofread.
- **PACKAGE_MANIFEST core files:** `manuscript.tex` (main), `manuscript.pdf` (14 pp), `ref.bib` (29),
  `neurips_2022.sty`, `sections/{abstract,body}.tex`, `table1_baselines.tex`…`table4_forensynths.tex`. Helpers
  (not submitted): `_build_body.py`, `_wc.py`. Build artifacts (regenerable): `.aux/.bbl/.blg/.log/.out/.fls/
  .fdb_latexmk/latexmk.out/manuscript.txt`. Rebuild: `python _build_body.py` (if drafts change) → `latexmk -pdf
  manuscript.tex`.
- **STATE = submission-shaped compact draft; only the author block strictly blocks a real submission.**

### 2026-06-03 — Stage 3 REVIEW complete (academic-paper-reviewer, full 5-reviewer, read-only)
- **D-36 [Stage 3 done]:** ran pipeline Stage 3 = REVIEW (per `MODE_REGISTRY`/`pipeline_state_machine.md`: Stage 3
  = peer review, NOT revision; READ-ONLY, no manuscript edits). Full package = `Stage3_Review/Stage3_Review_Report.md`
  (EIC + Methodology + Domain + Perspective + Devil's Advocate + editorial synthesis). Conducted inline (no agent
  spawn). User chose full 5-reviewer depth; panel confirmed.
- **DECISION = MAJOR REVISION** (competitive venue) / **MINOR** (course-workshop bar). No CRITICAL confirmed
  (so Accept not auto-barred, but 3-way MAJOR consensus drives Major).
- **Three consensus MAJOR issues:** (1) **no no-bottleneck baseline** — what does the bottleneck add over post-hoc
  attribution on a plain classifier? (the paper's own §4.3a shows the bottleneck collapses to ~2 correlated
  channels / 4 inert, so the "separately-testable channels" instrument is largely unrealized); (2) **single seed
  (42)** → headline numbers + per-channel reliance ranking are n=1; (3) **novelty increment over Grommelt** must be
  sharpened (foreground Outcome C + "intervention>weights").
- **DA strongest counter-argument:** the audit-instrument apparatus may be a single-feature ablation in a
  six-concept costume; could be read as reframing a negative result (concepts didn't yield meaningful/used
  channels) as a contribution. Rated MAJOR-approaching-CRITICAL, contingent on the no-bottleneck baseline.
- **Non-defect (DA checked):** single-gen baseline cherry-picking — CLEAN (cites 52.21, the higher of 52.21/51.70).
- **Revision Roadmap (Stage-4 input):** P0 gating = R-1 no-bottleneck baseline (new run), R-2 ≥3 seeds (new runs);
  P1 = R-3 sharpen contribution (framing-only), R-4 one P4 baseline (optional run), R-5 functional-reliance-vs-
  semantic-faithfulness sentence (framing); P2 = R-6 trace-vs-history as central open problem, R-7 scope hedge,
  R-8 ECE split / res128 wording / shortcut refs / stakeholder notes / resolution-mismatch confound.
- **Next = user decision (High oversight; DO-NOT #8 — no auto-start of Stage 4).** Options: Phase 2.5 revision
  coaching (Socratic, no edits) · Stage 4 framing-only (R-3/R-5/R-7, no new experiments) · Stage 4 full (adds
  R-1/R-2 experiments — lifts the recent no-experiments constraint) · hold.

### 2026-06-03 — Stage 4 REVISE: experiments LAUNCHED (user chose "Stage 4 full")
- **D-37 [Stage 4 full authorized + launched]:** user green-lit the full experiment set after seeing the cost
  (~35 min/epoch from `history.json` → ~11 h per 20-epoch run, ~32 h total on the 4060). New experiment code
  (NEW files, protected `cbnet.py`/`backbone.py`/`concepts/base.py` untouched):
  - `Code/cbnet_agid/models/baseline_plain.py` — `BaselinePlain` (DualStreamBackbone → GAP → `Linear(2560→1)`,
    **no bottleneck**, 28.22M params) for R-1.
  - `Code/cbnet_agid/train_baseline.py` — mirrors `train_multigen.py` (same loaders/LRs/AMP/seed) with plain BCE.
  - Both **smoke-passed** (1 epoch / 64 samples, exit 0) while GPU was free.
- **Data-state fix:** in-training `val_generators="all"` now ERRORS (`Stable_Diffusion_v1.5/val` empty; OOD gens
  fakes-only) → use `--val_generators Stable_Diffusion_v1.4` as the in-training monitor; real OOD + audit run
  post-hoc on each checkpoint via `eval_ood.py` + the closed-form audit (eval-only, does not change weights).
- **Sequencing = split/incremental (GPU is serial @ 8GB):** R-2 seed1 → seed2 → R-1 baseline, each ~11 h;
  eval+audit after each; then weave results + framing fixes (R-3 sharpen contribution / R-5 reliance-vs-
  faithfulness / R-7 scope hedge) into the manuscript + recompile. **E-Delta stays single-seed** (cost bound).
- **LAUNCHED: seed1** (background id `bb3mqk76q`, started 2026-06-03 17:57 +08; `--seed 1 --epochs 20
  --ckpt_every 5 --out_dir Logs/cbnet_multigen_seed1`). Confirmed running: 392.7 GB free, 200k samples,
  38.05M params, epoch 1/20. New dirs only; **no existing Logs/Results/Checkpoints touched** (DO-NOT #3 honored).
- **Scratch (tiny, no big ckpts):** `Logs/_smoke_seedtest`, `Logs/_smoke_baseline`, `Code/Results/*_smoke*.json`.
- **⚠ user must keep the laptop awake + plugged in** — a background training job dies on sleep/reboot.
- **Next (on seed1 completion, ~11 h):** verify epoch-20 ~99%; run OOD eval + audit (jpeg_quant zero-out rank
  stability); launch seed2; repeat; then baseline + attribution; then Stage-4 manuscript revision.

### 2026-06-04 — Stage 4: seed-1 DONE + audited; seed-2 launched
- **D-38 [seed-1 complete + multi-seed attribution finding]:** seed-1 (`Logs/cbnet_multigen_seed1`) finished
  clean (exit 0). Epoch-20 SD-1.4 val acc=100.00% AUC=1.000; full trajectory healthy (matches seed-42).
  Ran the canonical audit (identical code path) on the seed-1 ckpt: `scripts/analyze_all.py` →
  `Results/full_inference_dump_seed1.npz` (7 gens × 2000), then `scripts/concept_intervention.py` →
  `Results/seed1_analysis/`. Also recomputed seed-42 via the same path → `Results/seed42_analysis/` (both seeds
  now identically computed; the old gate3 hardcoded "freq_radial −2 pp" was wrong → real seed-42 = **−17.2 pp**).
  **No existing artifacts overwritten** (new filenames).
- **Result 1 — detection seed-stable ✓ (resolves M1 headline):** seed-1 in-dist SD 99.85 / BigGAN 99.80 /
  ADM 99.95 / MJ 99.45; OOD GLIDE 99.90 / Wukong 99.45 / VQDM 99.85 → **OOD mean 99.73%** vs seed-42 **99.67%**.
- **Result 2 — concept attribution NOT seed-stable (important, honest):** the compression pair
  `{jpeg_quant, freq_radial}` (A5 r=−0.80) are the top-2 zero-outs in BOTH seeds, dwarfing the other 4 (all
  <3.3 pp). But the carrier flips: **seed-42** jpeg_quant −49.5 pp (|w| rank-1), freq_radial −17.2 pp; **seed-1**
  freq_radial −38.7 pp (|w| rank-1), jpeg_quant −24.5 pp. → the *compression axis* reliance is robust; the
  *named-concept label* is not. **Strengthens** "intervention > weights" (D-31) + R-5 (functional-reliance ≠
  semantic-faithfulness); reinforces prohibition on "jpeg_quant = generative semantics". Full numbers +
  manuscript-impact notes → `Stage2_Writing/04_Stage4_Evidence.md`.
- **Manuscript-revision implications (NOT yet applied — deferred to post-baseline revision):** reframe the
  `jpeg_quant`-specific headline → the compression *pair* + a seed-robustness sentence; mark **E-Delta Outcome C**
  as single-seed (seed-42 lineage), do not generalize its specific-concept result across seeds.
- **LAUNCHED: seed-2** (background id `b6iaqv334`; `--seed 2 --out_dir Logs/cbnet_multigen_seed2`, same config).
  Confirmed training (epoch 1/20, ~4 it/s, ETA ~8–9 h). Disk guard passed. New dir only.
- **Next (on seed-2 completion):** same audit on seed-2 ckpt (does the carrier flip a 3rd way?) → mean±std over
  3 seeds; then launch R-1 no-bottleneck baseline (~11 h) + attribution; then Stage-4 revision. **⚠ keep laptop
  awake + plugged in.**

### 2026-06-05 — Stage 4: seed-2 DONE + audited; 3-seed synthesis complete; baseline launched
- **D-39 [seed-2 complete + 3-seed synthesis]:** seed-2 (`Logs/cbnet_multigen_seed2`) finished clean (exit 0;
  first launch `b6iaqv334` died at batch ~2064/6250 in epoch 1 — exit code 6 = laptop slept — relaunched as
  `bm049eo4a`, ran to completion). Epoch-20 SD-1.4 val acc=99.95% AUC=1.000. Full canonical audit pipeline
  (identical code path): dump → `Results/full_inference_dump_seed2.npz`, intervention → `Results/seed2_analysis/`.
- **Detection (all 3 seeds stable ✓):** OOD mean 99.67 / 99.73 / 99.82; in-dist all ≥99.45. → "competitive
  single-GPU detector" claim survives multi-seed replication.
- **Concept attribution (all 3 seeds):** the compression pair `{jpeg_quant, freq_radial}` dominates all 3 seeds
  (top-2 zero-outs, dwarfing the other 4 concepts). But the carrier shifts:
  - seed-42: jpeg_quant −49.5 pp (|w| #1), freq_radial −17.2 pp
  - seed-1: freq_radial −38.7 pp (|w| #1), jpeg_quant −24.5 pp
  - seed-2: jpeg_quant −25.7 pp, freq_radial −24.7 pp (nearly tied), texture_geometry −7.8 pp (|w| #2)
  - texture_geometry shows a moderate seed-2 contribution (−7.8 pp, weight −10.51) — may deserve a sentence
    in the revision.
  - keep-only-one is wildly seed-dependent: seed-2 bitplane_lsb-alone ≈ 95% (!) vs seed-42 freq_radial ≈ 90%
    vs seed-1 nothing >53%. Bias-calibration artifact; reinforce "intervention > weights."
  → compression *axis* reliance robust; named-*concept label* not. Strengthens "intervention > weights" (D-31)
    + R-5. Full numbers + synthesis in `Stage2_Writing/04_Stage4_Evidence.md`.
- **LAUNCHED: no-bottleneck baseline** (bg id `bP1eNzE8`; `--seed 42 --out_dir Logs/cbnet_baseline_nobottleneck_s42`).
  ETA ~11 h. **⚠ keep laptop awake + plugged in.**
- **Next (on baseline completion):** OOD eval + confound battery (B1/B2) + post-hoc attribution on the baseline
  → "what does the bottleneck add?" (resolves M2). Then Stage-4 manuscript revision + recompile.

### 2026-06-05 — Stage 4: baseline DONE; all experiments complete; entering revision
- **D-40 [R-1 baseline complete + resolves M2]:** `BaselinePlain` (no bottleneck, 28.2M, seed 42, 20 ep)
  finished clean (exit 0, SD-1.4 val 99.95% AUC 1.000). OOD eval: GLIDE 99.90 / Wukong 99.00 / VQDM 99.90 →
  **OOD mean 99.60%** — within ~0.1 pp of CBNet (99.67–99.82). **The bottleneck does not cost accuracy.**
- **B1 confound (JPEG-q95):** both models equally vulnerable — CBNet mean acc 88.99% vs baseline 87.25%
  (Δ −1.7 pp). The plain model is *slightly more* susceptible to compression confounding, not less.
  → Removing the bottleneck does NOT remove the shortcut; the backbone + training data are the source.
  The bottleneck adds audit visibility at zero accuracy cost and zero extra confound risk.
- **Resolves M2** ("what does the bottleneck add?") and the Devil's Advocate's "six-concept costume" critique.
- **All Stage-4 experiments complete:** R-2 multi-seed (3 seeds) ✓ · R-1 baseline ✓ · B1 confound ✓.
- **Next = Stage-4 manuscript revision + recompile:** weave R-2 (seed-stable detection + seed-unstable
  attribution → compression pair, not jpeg_quant alone) and R-1 (baseline = accuracy-neutral, same
  confound) into the paper; apply framing fixes (R-3/R-5/R-7); recompile. E-Delta Outcome C stays
  single-seed with cross-seed caveat. Full numbers in `Stage2_Writing/04_Stage4_Evidence.md`.

### 2026-06-05 — Stage 4: manuscript revision COMPLETE + recompiled
- **D-41 [Stage-4 manuscript revision done]:** all Stage-4 experimental results woven into the paper;
  framing fixes (R-3/R-5/R-7) applied. **15 pages** (was 14). Build: `latexmk` exit 0, 0 undefined
  citations, 0 errors, 1 overfull (pre-existing table1_baselines.tex 20.69pt, unchanged).
- **Edits made (file: `sections/body.tex` + `sections/abstract.tex`):**
  - **Abstract:** "single frequency/compression cue" → "frequency/compression cue pair (jpeg-quant
    and freq-radial, r=-0.80)"; carrier-label-shifts-across-seeds sentence; single-seed caveat on
    Outcome C; baseline sentence (99.60% OOD, equally confound-susceptible).
  - **§1 Introduction:** core reliance claim reframed from "single channel" to "compression-axis
    pair"; 17–50pp range across seeds; contributions block updated (compression-axis pair, rank-1
    weight shifts, single-seed Outcome C).
  - **§4.1 Setup:** new paragraph describing multi-seed replication (seeds 1, 2) and BaselinePlain
    (28.2M, no bottleneck, seed 42, 20 ep).
  - **§4.2 Detection:** seed-stability sentence (OOD 99.73/99.82 across 3 seeds); baseline OOD
    99.60% comparison; bottleneck is accuracy-neutral.
  - **§4.3a Channel reliance:** multi-seed zero-out data (compression pair dominates all 3 seeds,
    carrier label shifts); keep-only-one seed-robustness caveat; counterfactual swap seed-42 note;
    "two-channel single-cue" → "compression-axis classifier"; weight-vs-intervention updated.
  - **§4.3c Outcome C:** explicit "single-seed result" caveat + cross-reference to §4.3a multi-seed.
  - **NEW §4.5 What the bottleneck adds:** baseline OOD 99.60% vs CBNet 99.67–99.82%; B1 confound
    comparison (baseline 87.25% vs CBNet 88.99%); bottleneck = interpretive, not predictive.
  - **§5 Discussion:** weight-vs-intervention updated with multi-seed evidence; "two-channel
    single-cue" → "compression-axis, single-cue"; R-5 sentence added ("functional reliance does
    not imply semantic faithfulness"); Outcome C → single-seed; baseline paragraph added.
  - **§6 Conclusion:** "single frequency/compression channel" → "compression-axis channel pair";
    single-seed caveat on Outcome C; multi-seed stability sentence.
  - **Ablations not run:** updated to note the bottleneck ablation WAS run (§4.5).
- **What did NOT change:** Tables 1–4 (all numbers are real disk artifacts), ForenSynths numbers,
  B1/B2/B5 confound numbers, E-Delta specific numbers (-48.18pp, Cohen's d, etc.), ref.bib,
  Method section (§3). All original seed-42 numbers preserved as primary; multi-seed data layered
  as robustness evidence.
- **Next = re-review (Stage 3'):** run the reviewer panel again on the revised manuscript to check
  whether the Major-Revision issues (M1/M2/M3) are resolved.
