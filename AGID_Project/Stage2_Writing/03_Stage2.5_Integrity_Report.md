# Stage 2.5 — Integrity Report (AGID Paper)

**Date:** 2026-06-03 · **Mode:** ARS academic-pipeline Stage 2.5 (verify-only; no manuscript assembly, no new experiments, no major rewrite)
**Scope:** all 8 drafted chapters in `Stage2_Writing/draft/` (Abstract, Intro, RW, Method, Experiments, Discussion, Limitations, Conclusion)
**Verification basis:** every number re-read from disk artifacts this session — `gate3_full_audit.json`, `A_intervention_summary.md`, `B_confound_summary.md`, `F3_main_results.md`, `cbnet_multigen_forensynths.json`, `cbnet_forensynths.json`, `npr_forensynths.json`, `cbnet_100k_forensynths.json`; citations cross-checked against `Stage1_Research/03_Bibliography.md`; claims cross-checked against `00_Evidence_Spine_Lock.md`.

---

## 0. Preflight actions COMPLETED this pass (mechanical, in-scope)

| # | Action | Result |
|---|---|---|
| P-1 | Grommelt metadata via arXiv 2403.17608 | Authors = **Patrick Grommelt, Louis Weiss, Franz-Josef Pfreundt, Janis Keuper**; title "Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets"; 2024; venue = arXiv preprint (no published venue found). Bib entry updated; "Springer chapter" note marked UNVERIFIED. |
| P-2 | Resolve 4 cross-ref placeholders | `§[experiments-setup]`→`§4.1` (Intro); `§[experiments]`→`§4.3` (Method); `§[limitations]`→`§6` (Experiments); `§[limitations]`→`§6` (Discussion). All resolved. |
| P-3 | Section-numbering gap (5 was skipped) | Renumbered headers: Discussion 6→**5**, Limitations 7→**6**, Conclusion 8→**7**. Numbering now contiguous 1–7. (Confirm acceptable — see OPEN DECISION O-2.) |
| P-4 | Title candidates | Listed below as OPEN DECISION O-1 — NOT finalized. |
| P-5 | To-verify bibliography entries | Flagged below as FIX-4. |

---

## 1. PASS — verified clean

**1a. Numbers (all trace exactly to disk artifacts):**

| Claim (location) | Value in draft | Artifact | Status |
|---|---|---|---|
| GenImage held-out OOD mean (Abs/Intro/Exp/Disc) | 99.67% (GLIDE 99.80 / Wukong 99.45 / VQDM 99.75) | `F3_main_results.md` | ✅ exact (mean=99.667) |
| Detection table 7 rows incl. AUC/real/fake (Exp §4.2) | all values | `F3_main_results.md` | ✅ exact |
| ForenSynths Route B (Exp/Disc) | 73.65% / AUC 0.820; StarGAN 96.42, DeepFake 82.24, BigGAN 61.10, GauGAN 54.82; real-acc 22.2/9.7 | `cbnet_multigen_forensynths.json` | ✅ exact |
| ForenSynths single-gen Route A (Exp §4.4) | 52.21% / AUC 0.528 | `cbnet_forensynths.json` (debug_run_20k ep4) | ✅ exact (see FIX-5) |
| ForenSynths NPR (Exp §4.4) | 79.58% / AUC 0.894 | `npr_forensynths.json` | ✅ exact |
| Linear-head weights / bias (Exp §4.3a) | [+6.45,+9.84,+6.21,+3.73,−13.53,+7.35], b=−5.06 | `A_intervention_summary.md` + `gate3` concept_weights.routeb | ✅ exact |
| jpeg_quant zero-out, Route B (Exp §4.3a) | −49.2…−49.8pp; 99.7→50.2 | `A_intervention_summary.md` | ✅ exact |
| keep-only freq_radial / jpeg_quant (Exp §4.3a) | 81.6–99.7% / exactly 50% | `A_intervention_summary.md` | ✅ exact |
| counterfactual jpeg_quant flips (Exp §4.3a) | real 23.9–72.5%, fake 9.8–44.9%; others <2% (freq_radial max 9%) | `A_intervention_summary.md` | ✅ exact |
| jpeg_quant↔freq_radial correlation | −0.80 | `A_intervention_summary.md` (A5) | ✅ exact |
| Confound battery (Exp §4.3b) | jpeg-q95 −10.16 (BigGAN −19.4, ADM −14.0); png +0.00; res128 −44.64; indep-real +0.12 | `B_confound_summary.md` | ✅ exact |
| E-Delta weight shrink (Exp §4.3c/Disc) | −13.53→−9.11 = 1.49× | `gate3` (1.4856) | ✅ exact |
| E-Delta zero-out jpeg_quant (Exp §4.3c) | −48.18pp, rank-1; freq_radial −0.90pp | `gate3` a3_ablation_debiased | ✅ exact |
| E-Delta Cohen's d jpeg_quant (Exp §4.3c) | −3.84…−5.47 | `gate3` cohens_d_debiased | ✅ exact |
| Double dissociation (Exp §4.2) | Q96-val: RouteB 89.75 / EDelta 99.78; raw-OOD: RouteB 99.73 / EDelta 95.0 | `gate3` detection (recomputed) | ✅ exact |
| Calibration ECE (Exp §4.1) | 0.0034 | `A_intervention_summary.md` | ✅ exact |
| Cross-benchmark drop / multi-gen gain | −26pp (99.67→73.65); +21.4pp (52.21→73.65) | arithmetic | ✅ exact |
| Architecture/training facts (Method §3) | 2560-ch fuse, K=6, ~38M, AdamW lrs, bf16, bs32×2, 256², seed42, 20/10 ep, λ=(0.5,0.2,0.05) | `01_Decision_Log` D-17/19/21/23 (code-verified) | ✅ consistent |

**1b. Citations:** all **26** cited keys (`brock2018biggan, rombach2022ldm, dhariwal2021adm, nichol2022glide, zhu2023genimage, wang2020cnngenerated, tan2024npr, wang2025lota, zhou2025aigiholmes, koh2020concept, margeloiu2021concept, mahinpei2021promises, havasi2022addressing, grommelt2024fakejpeg, he2016resnet, frank2020dctgan, durall2020watch, tan2024freqnet, wang2023dire, ojha2023universal, huang2025thinkfake, tan2025forenx, ji2025grounded, kim2018tcav, petsiuk2018rise, loshchilov2019adamw`) exist in `03_Bibliography.md`. No dangling `\citep{}`/`\citet{}`. ✅

**1c. Claims vs evidence spine:** consistent with all of spine §1–§4. Lead-with-intervention (not weights) ✅; double-dissociation-not-de-reliance ✅; Outcome-C claim ceiling (not generative semantics) ✅; CBM = testability surface ✅.

**1d. Prohibited-claims audit (spine §5, all 8):** every occurrence of a prohibited term is a **correct negation**, not an assertion. ✅
- "competitive, **not** state-of-the-art" (Abstract, Intro §31, Exp §4.1, Limitations) — #6 OK
- "**neither** 'merely exploits a confound' **nor** robustly generalizes" / "partial … partial" (Intro/Disc/Concl) — #7/#8 OK
- "we do **not** claim the channel captures generative semantics" (Disc §5, Intro §22) — #3 OK
- "'interpretable by construction' is the wrong claim" / "we do not claim six … faithful concepts" (Disc, RW, Limitations) — #4 OK
- weight-shrink always introduced then overridden by zero-out (Exp §4.3c, Disc) — #5 OK
- migration narrative retired; only Outcome-C/persistence stated (Limitations §"Scale of the audit evidence") — #1/#2 OK

---

## 2. FIX REQUIRED (before final submission; NOT yet applied — awaiting your go)

**FIX-1 (numerical accuracy — Experiments §4.3a).** The sentence *"the remaining four channels cost under one point each"* is **inaccurate**. Zero-out of `texture_geometry` costs −2.6pp (SD-1.4), −1.3pp (VQDM), and −5.75pp (Wukong) — the source (`A_intervention_summary.md`) itself notes "essentially dead weight **except texture_geometry on Wukong = −5.75pp**." Three of the four are truly <0.5pp; texture_geometry is the exception.
→ *Suggested reword:* "the remaining four channels each cost about a point or less on most generators (texture-geometry the largest exception, up to −5.8pp on Wukong)." (This is the same class of fix already applied to the Introduction; §4.3a is the residual instance.)

**FIX-2 (consistency follow-on — Discussion §5).** "the remaining four channels are inert despite being trained" should be softened to "**near-inert**" (or add the texture_geometry caveat) to stay consistent with FIX-1.

**FIX-3 (Abstract/Intro vs Experiments — "single channel" vs "two-channel").** Abstract + Intro say the decision "turns on a **single** frequency/compression channel"; Experiments §4.3a concludes the model is "effectively a **two-channel**, single-cue model" (jpeg_quant + freq_radial, −0.80 correlated). Not a hard contradiction (jpeg_quant is the sole *causal flipper*; freq_radial is a redundant standalone discriminator encoding the *same cue*), but a reviewer could read tension.
→ *Suggested:* in Abstract/Intro write "a single compression/frequency **cue**" (cue, not channel), or add "(carried by two correlated channels)". Low effort, removes ambiguity.

**FIX-4 (bibliography — 2025 entries to verify before `ref.bib`).** The bib's own note (§Notes #4–5) flags that 2025 entries may be placeholders. High-risk keys actually cited in the paper, needing author/venue/title confirmation:
- `zhou2025aigiholmes` (AIGI-Holmes, ICCV 2025) — **sole headline foil**, must be correct.
- `wang2025lota` (LOTA, ICCV 2025) — its 98.9% is tabulated.
- `huang2025thinkfake`, `tan2025forenx`, `ji2025grounded` — cited in RW Line 2; most likely to have placeholder metadata.
- `grommelt2024fakejpeg` — authors now verified; still confirm whether to cite arXiv vs a published version (see O-3).

**FIX-5 (Table 1 not finalized).** §4.1 "Table 1 (schematic)" + "see supplementary" is a prose placeholder, and §6 (Limitations) references "Table 1" as if it exists. The baseline master-table must be built from `E1_sota_baselines.md` (with protocol P1–P4 labels + comparability column + CBNet two sub-rows) at assembly time. *(Deferred — you asked not to assemble the manuscript yet; logging it so it isn't forgotten.)*

**FIX-6 (low priority — §4.4 baseline checkpoint).** The cited single-gen number 52.21% is from the **20k debug** checkpoint (epoch 4); the **100k "main"** Route-A run is **51.70%** (`cbnet_100k_forensynths.json`). Both are ≈chance, so the +21pp conclusion is **robust either way** and the cited value is not cherry-picked high (it is the higher of the two by 0.5pp). Optional: cite the 100k main run, or add "(both single-gen checkpoints ≈ 52%)" for defensibility.

---

## 3. OPEN DECISION (your ruling needed)

**O-1 (Title — choose one; do not want me to finalize).** Candidates, all honoring competitive-not-SOTA + audit-instrument + "what is the accuracy made of":
1. *What Is the Accuracy Made Of? Auditing a Competitive AI-Generated Image Detector with a Concept Bottleneck*
2. *Concept Bottlenecks as Audit Instruments: Localizing the GenImage Confound Inside a Competitive Detector*
3. *From Dataset Bias to Model Reliance: Causally Auditing What an AI-Generated-Image Detector's Accuracy Rests On*
4. *Not Interpretable, but Testable: A Concept-Bottleneck Audit of Confound Reliance in AGID*
5. *Auditing AI-Generated Image Detectors from the Inside: Causal Intervention on Separately-Testable Evidence Channels*

**O-2 (Section numbering).** I closed the gap by renumbering (Discussion 5 / Limitations 6 / Conclusion 7). Alternative: promote the Audit (§4.3) to a standalone "5. Audit" section (fills the gap differently, foregrounds the contribution). Confirm renumber, or request the split.

**O-3 (Grommelt venue).** Cite as arXiv:2403.17608 (2024) — current choice — or track down a published version if you prefer a non-preprint citation. The "Springer chapter" note is unverified.

**O-4 (settled — no action).** Abstract language = **English only** per your instruction (chapter plan's "bilingual EN/ZH" option is overridden). Logged so it isn't reopened.

---

## 4. Seven-mode AI-research-failure sweep

| # | Failure mode | Verdict | Note |
|---|---|---|---|
| 1 | Fabricated / unverifiable numbers | ✅ PASS | every quantitative claim traced to a disk artifact (§1a) |
| 2 | Citation integrity (exist + supported) | ⚠️ MOSTLY | all keys exist; 2025 metadata unverified (FIX-4) |
| 3 | Overclaiming (SOTA / pure-confound / faithful / generative-semantics) | ✅ PASS | all 4 prohibited families absent; present only as negations (§1d) |
| 4 | Internal contradiction across chapters | ⚠️ MINOR | "single channel" vs "two-channel" wording (FIX-3); "under one point" vs source (FIX-1) |
| 5 | Unfair comparison / cherry-picking | ✅ PASS | protocol P1–P4 disclosed; baseline checkpoint robust (FIX-6); NPR (79.58>73.65) reported *against* own favor |
| 6 | Causal / circular reasoning | ✅ PASS | intervention>weights applied consistently; "dependency" licensed by ablation, never "purely/exclusively" |
| 7 | Undisclosed confounds / leakage | ✅ PASS | protocol divergence, ForenSynths shift-entanglement, Q96 ceiling, smoke-overfit, crop/label all in Limitations §6 |

---

## 5. Bottom line

The draft is **numerically clean and claim-disciplined**: 0 fabricated numbers, 0 prohibited claims asserted, 0 dangling citations. Blocking items before a final manuscript are small and mostly mechanical: **FIX-1/2/3** (three wording precisions), **FIX-4** (verify 5 citation entries), **FIX-5** (build Table 1). Decisions **O-1 (title)** and **O-2 (numbering)** are yours. No new experiments are required and no major rewrite is implied.

*Per your instruction, I stopped here: manuscript not assembled, FIX items listed but not applied (except the in-scope preflight resolutions in §0).*

---

## 6. Resolution addendum (2026-06-03, after user rulings)

- **O-1 Title RESOLVED:** **"What Is the Accuracy Made Of? Auditing a Competitive AI-Generated Image Detector with a Concept Bottleneck."** (Recorded in Intro draft header + decision log D-32. Title page itself is created at assembly.)
- **O-2 Numbering RESOLVED:** keep renumber (Discussion 5 / Limitations 6 / Conclusion 7).
- **FIX-1 APPLIED** (Experiments §4.3a) — texture-geometry exception now stated.
- **FIX-2 APPLIED** (Discussion §5) — "inert" → "near-inert."
- **FIX-3 APPLIED** (Abstract) — "single … channel" → "single … cue." Intro deliberately left intact (already precise; see D-32).
- **FIX-4 RESOLVED (2026-06-03, web-verified):** all 5 cited 2025 papers EXIST and all 5 bibkey first-author surnames are CORRECT (no hallucinated keys). But **3 of 5 had fabricated titles** in the bib — corrected:

| bibkey | exists | 1st author ✓ | title | venue | arXiv |
|---|---|---|---|---|---|
| `wang2025lota` | ✓ | Hongsong Wang | ✓ already correct | ICCV 2025 ✓ | 2510.14230 |
| `zhou2025aigiholmes` | ✓ | Ziyin Zhou | ✓ (expanded "LLM") | ICCV 2025 (IEEE conf pub) | 2507.02664 |
| `huang2025thinkfake` | ✓ | Tai-Ming Huang | ❌→✓ FIXED ("Reasoning in MLLMs…", not "Chain-of-Thought…") | arXiv preprint | 2509.19841 |
| `tan2025forenx` | ✓ | Chuangchuang Tan | ❌→✓ FIXED ("Towards Explainable AIGI Detection w/ MLLMs", not "Forensic eXplainability…") | arXiv preprint | 2508.01402 |
| `ji2025grounded` | ✓ | Yikun Ji | ❌→✓ FIXED ("Interpretable and Reliable Detection… via Grounded Reasoning in MLLMs", not "Grounded Reasoning for AIGI Detection") | arXiv preprint | 2506.07045 |

  Manuscript prose is **unaffected** — the wrong titles lived only in the bib annotation; the drafts cite these by system name (`ThinkFake`/`ForenX`) or group them generically, never asserting a title. Two residual notes: (a) `huang`/`tan`/`ji` are arXiv **preprints**, not the "2025" published-venue the bib implied — fine to cite as preprints; (b) the bib's per-paper accuracy figures (ThinkFake 84%, ForenX 97.8%, LOTA 98.9%) are annotation-only and NOT used in the manuscript prose — re-confirm LOTA 98.9% from source when building Table 1 (FIX-5).
- **Still open / deferred:** FIX-5 (build Table 1 at assembly), FIX-6 (optional 100k-checkpoint note), O-3 (Grommelt arXiv vs published venue).
