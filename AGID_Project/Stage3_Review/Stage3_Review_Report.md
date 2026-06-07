# Stage 3 — Peer Review Package (CBNet-AGID audit paper)

**Date:** 2026-06-03 · **Skill:** academic-paper-reviewer (full mode, 5 reviewers) · **Mode A** (pipeline Stage 3)
**Manuscript under review:** `Stage2_Writing/manuscript/manuscript.pdf` (14 pp) — *"What Is the Accuracy Made Of? Auditing a Competitive AI-Generated Image Detector with a Concept Bottleneck."*
**READ-ONLY:** no manuscript file was modified. All output below is critique only (IRON RULE #6).
**Prerequisites:** Stage 2.5 integrity report ✓ (`03_Stage2.5_Integrity_Report.md`), verified draft ✓.

> Scores are 0–100 per dimension. Calibration note (anti-pattern #5): a paper with unrun core ablations and a single training seed cannot score >60 on Methodological Rigor regardless of polish.

---

## Phase 0 — Field analysis & panel configuration (confirmed by user)

- **Primary field:** AI-generated image detection / media forensics (computer vision).
- **Secondary:** interpretable ML (concept-bottleneck models) + benchmark measurement-validity / dataset confounds.
- **Paradigm:** empirical deep learning with a diagnostic/audit emphasis (causal intervention on model internals).
- **Target tier (as positioned):** NeurIPS-style single-column; honest "competitive, not SOTA" + audit-instrument framing. Realistic venue = trustworthy-ML / forensics workshop or a strong course capstone.
- **Maturity:** complete compact draft; numbers artifact-verified; explicit limitations ledger.

Panel: EIC + R1 Methodology + R2 Domain(AGID) + R3 Perspective(interpretability/benchmarking) + Devil's Advocate.

---

# Phase 1 — Independent reviews

*(Each reviewer wrote without seeing the others; overlaps are addressed from different angles.)*

## Review 1 — Editor-in-Chief (journal fit, originality, significance)

**Summary.** The paper trains a deliberately conventional dual-stream + 6-channel concept-bottleneck detector on four GenImage generators, reaches 99.67% mean held-out-OOD accuracy, and then uses the bottleneck as a causal "audit instrument" to show the decision is carried by a single compression/frequency cue that survives Grommelt-style Q96 debiasing (Outcome C) and partly fails to transfer to ForenSynths (73.65%). The thesis is "competitive-not-SOTA detector + audit instrument," read as "partial confound inflation + genuine partial generalization."

**Assessment.** This is an unusually *honest* paper, and that is its strongest asset: the claims are hedged exactly where the evidence is thin, the prohibited over-claims (hard SOTA, pure confound, faithful-by-construction, generative-semantics) are conspicuously avoided, and the limitations ledger is exemplary. The audit framing is a legitimate and timely reframe of concept bottlenecks for measurement-validity.

The concern for any competitive venue is **originality vs. increment**. The headline phenomenon (GenImage conflates "generated-ness" with a JPEG/container confound, and detectors exploit it) is Grommelt 2024's, which the paper correctly credits. The paper's increment is "model-internal localization + causal intervention + Q96-persistence." Two of the three legs are at risk: (a) the model-internal apparatus (the bottleneck) is shown by the paper's own §4.3a to collapse to ~2 correlated channels with 4 inert — so the "separately-testable evidence channels" selling point is largely unrealized; and (b) the confound battery (§4.3b: JPEG-q95, res128) is itself essentially black-box perturbation, not unique to the bottleneck. What genuinely survives as novel is the **Q96-persistence result (Outcome C) + the methodological lesson "intervention > weights."** That is real but narrow.

**Recommendation:** **Major Revision.** The contribution is honest and partially novel but needs (i) a no-bottleneck baseline establishing what the bottleneck adds, and (ii) multi-seed evidence, before the "audit instrument" claim is defensible at a competitive venue. (For a course/workshop bar, this is closer to Minor.)

**Scores:** Originality 52 · Significance 58 · Rigor 55 · Clarity 88 · Positioning 78 · Reproducibility 80 · Soundness-of-claims 82.

---

## Review 2 — Reviewer 1: Methodology (design, statistics, reproducibility)

**Summary.** Empirical audit built on intervention probes (zero-out, counterfactual, weight/Cohen's-*d*) over a linear-head concept bottleneck, plus a confound-perturbation battery and a Q96-debiased retrain.

**Strengths.** The no-skip bottleneck genuinely licenses clean causal ablation (the logit is exactly $w^\top c + b$), and the audit battery is internally coherent. Reporting per-class real/fake accuracy (Table 2) is good practice and is diagnostically used. The double dissociation (§4.2/§4.3c) is a thoughtful design.

**Major weaknesses.**
- **M1 — Single seed (location: §3.6 "fixed seed of 42"; every result table).** All headline numbers are single-run point estimates with no variance/CI: 99.67, 73.65, the −48.18pp zero-out, the double-dissociation means. The central qualitative claim ("the decision turns on a *single* channel") is a statement about a learned solution that could be seed-dependent — which channel dominates may vary across initializations. *Fix:* report ≥3 seeds with mean±std on the headline metrics and on the per-channel zero-out ranking; show the jpeg_quant dominance is stable.
- **M2 — Core ablations unrun (location: §4.4 "Ablations not run"; Limitations §6).** No K-sweep, no backbone ablation, and critically **no no-bottleneck baseline** (same backbone + post-hoc attribution). The paper's contribution *is* the bottleneck-as-instrument, yet there is no experiment isolating what the bottleneck buys over post-hoc attribution on a plain classifier. *Fix:* at minimum, a same-backbone linear-probe / Grad-CAM comparison showing the bottleneck localizes the confound where post-hoc attribution does not (or admit it does not).
- **M3 — No same-protocol baseline (location: §4.1, Table 1).** Every baseline is P1/P2; CBNet is P4 on a *different* held-out set. The comparability column helps, but a reader still sees 99.67 stacked above 88.6 (NPR). *Fix:* retrain one baseline (CNNSpot or NPR) under P4 on the same four generators and report it — the only way to make Table 1 a real comparison rather than a labeled non-comparison.
- **M4 — Route B vs E-Delta is not one-factor (location: §4.2, §4.3c).** The two models differ in epochs (20 vs 10) as well as data debiasing; "attributable to the debiasing alone" (§3.6) is too strong. *Fix:* match epochs or show insensitivity.

**Minor.**
- **m1 — ECE underspecified (§4.1 "ECE 0.0034").** On which split? OOD calibration is what matters; report it or scope the claim.
- **m2 — "balanced accuracy" (§3.7) vs plain accuracy (tables)** — confirm which is used where; the zero-out "−49.x to chance" reads as balanced acc but tables look like raw acc.
- **m3 — res128 (§4.3b)** conflates resolution with high-frequency content; it is evidence for *frequency* reliance, not specifically the *container* confound — tighten the wording.

**Recommendation:** **Major Revision.** M1 and M2 are the gating issues; the audit thesis is not yet empirically isolated.

**Scores:** Rigor 50 · Reproducibility 78 · Soundness 72 · Clarity 86.

---

## Review 3 — Reviewer 2: Domain / AGID (literature, framework, contribution)

**Summary.** Positions a concept-bottleneck detector as a model-internal complement to dataset-level confound work, extending Grommelt.

**Strengths.** Literature is current and the framing is fair: NPR, LOTA, UnivFD, AIGI-Holmes, and Grommelt are all correctly characterized, and "we extend, not discover" (Intro, §2 Line 3, Conclusion) is exactly the right posture. The CBM-faithfulness-critique scaffolding (margeloiu/mahinpei/havasi) is apt. Crediting Grommelt prominently pre-empts the obvious scoop objection.

**Major weaknesses.**
- **D1 — Increment over Grommelt may be thin (location: §2 Line 3; §4.3c).** Grommelt already shows the JPEG/size confound and that uniform re-encoding shifts performance. The model-internal localization is welcome, but because the bottleneck degenerates to one cue (§4.3a), the "localization" reduces to "the one channel that works is the compression channel" — which a single-feature probe would also show. The durable new fact is **Q96-persistence (Outcome C)**; the paper should foreground it as *the* contribution rather than the bottleneck apparatus. *Fix:* sharpen the contribution statement around Outcome C + "intervention > weights," and explicitly state what Grommelt could not show.
- **D2 — Generative-trace vs compression-history is the question readers care about, and it is unresolved (§4.3c, Limitations).** The honest hedge is correct, but it leaves the headline ("not reducible to container label") as a modest positive. *Fix:* even a small controlled experiment (e.g., re-encode reals through the same generative VAE decode, or matched-pipeline reals) would meaningfully raise the payoff; if infeasible, frame the limitation as the central open problem, not a footnote.
- **D3 — Scope (Limitations).** Single dataset family (GenImage) + ImageNet reals; ForenSynths is the only external test and is confounded with GAN-vs-diffusion shift. The audit-generality claim is asserted. *Fix:* soften "the audit applies broadly" language, or add one more corpus.

**Minor.**
- **d1 — Missing references** on shortcut/spurious-feature learning (e.g., Geirhos et al., shortcut learning) and on frequency-artifact detectors beyond those cited; would strengthen the measurement-validity framing.
- **d2 — LOTA P2 caveat (Table 1)** is handled well; consider also extracting LOTA's SD-v1.4-only row if available, for a fairer cell.

**Recommendation:** **Major Revision** (borderline Minor if D1 reframing + D3 hedging are done and D2 is acknowledged as the central open problem).

**Scores:** Positioning 80 · Originality 55 · Significance 60 · Soundness 80.

---

## Review 4 — Reviewer 3: Perspective (interpretability / benchmarking / cross-disc.)

**Summary.** Reframes the concept bottleneck from an interpretability device to a "testability surface."

**Strengths.** The reframe is genuinely interesting and travels beyond AGID: "design intent is a hypothesis about behavior; behavior must be audited" (§5) is a clean, quotable lesson that the CBM and trustworthy-ML communities will appreciate. The "intervention > weights" demonstration (a large-|w| channel that cannot classify alone; a weight that shrinks under debiasing while functional reliance does not) is a crisp, transferable result.

**Major/Moderate weaknesses.**
- **P1 — The paper critiques others' unverified faithfulness but does not evaluate its own concepts' faithfulness to *human* meaning (location: Intro obstacle 1; §3.4; §5).** It tests *functional reliance* (which channel the decision uses), not whether jpeg_quant *means* "JPEG quantization" to a human. The paper is careful to say names aren't guaranteed — but then the "concept" vocabulary is purely nominal, and the contrast with AIGI-Holmes is partly rhetorical (both end with unverified human-faithfulness; this paper verifies *use*, not *meaning*). *Fix:* state explicitly that the audit establishes functional reliance, not semantic faithfulness, and that these are different claims; this strengthens rather than weakens the paper.
- **P2 — Practical payoff for benchmark design is implied but not operationalized.** The paper shows GenImage is confounded model-internally; what should a benchmark builder *do*? *Fix:* one paragraph of concrete recommendations (or future work) would raise impact.

**Minor.**
- **p1** — the "testability surface" idea would benefit from one sentence relating it to TCAV/insertion-deletion (already cited) as prior testability tooling, to position the novelty precisely (it is the *no-skip architectural guarantee*, not the testing per se).

**Recommendation:** **Minor Revision.** The cross-disciplinary contribution is the paper's most exportable asset; P1 is a framing fix, not new work.

**Scores:** Originality 62 · Significance 64 · Clarity 88 · Soundness 80.

---

## Review 5 — Devil's Advocate (core-argument challenge) — *special format*

**Strongest counter-argument (the case for rejecting the central claim).**
The paper's thesis is that a concept bottleneck functions as an *audit instrument* that localizes the GenImage confound inside a competitive detector. But the paper's own central result refutes the instrument's premise: §4.3a shows the bottleneck collapses to **two −0.80-correlated channels encoding one cue, with four channels inert.** The advertised capability — *separately*-testable evidence channels — is therefore unrealized; there is effectively **one** testable cue. Meanwhile, the confound *evidence* comes from (a) zeroing that one channel and (b) the §4.3b perturbation battery (JPEG-q95, res128) — and (b) is a black-box, model-agnostic manipulation that needs no bottleneck at all. Grommelt already established the confound with black-box detectors. So the genuinely bottleneck-dependent increment narrows to: "a linear probe on one frequency channel, zeroed, drops accuracy to chance, and still does so after Q96 debiasing." That is a real observation, but it is a **single-feature ablation dressed in a six-concept apparatus**, and the paper can be read as **reframing a negative result (concept supervision did not yield meaningful, used concepts) as a positive contribution.** The "competitive detector" is also not a contribution (the paper says so), and its 99.67% is on a non-comparable protocol. Strip the framing and ask "what is established that Grommelt + a standard saturation/ablation study did not establish?" — the honest answer is "Q96-persistence and the weights-vs-intervention dissociation," which is a workshop-sized increment.

**Issue list.**
- **[MAJOR → approaches CRITICAL] Instrument degeneracy undercuts the headline contribution.** (§4.3a vs Abstract/Intro/§5.) If unaddressed, the "audit instrument" claim is not supported by a working instrument. *Required:* either a no-bottleneck baseline showing the bottleneck adds localization value (ties to M2), or an explicit reframing to "Q96-persistence + intervention-over-weights" as the contribution, demoting the bottleneck to a convenient vehicle.
- **[MAJOR] Novelty increment over Grommelt is small once the apparatus is discounted.** (§2 Line 3, §4.3c.) Ties to D1.
- **[MAJOR] Single seed** (M1) makes "the decision turns on a single channel" an n=1 claim about a learned solution.
- **[MINOR] "Cherry-picking" check — passed.** The single-gen baseline uses the *higher* of two ≈chance checkpoints (52.21 vs 51.70), so the +21pp claim is not inflated; good. (Non-defect, noted for the record.)
- **[MINOR] "So what?" partial pass.** The benchmark-validity message matters, but the actionable consequence is left implicit (ties to P2).

**Ignored alternative explanations.**
- The 99.67→73.65 drop is attributed to "confound inflation + distribution shift," but a third factor — **resolution/center-crop mismatch** between GenImage (256² crops) and ForenSynths native sizes — is not isolated and could account for part of the drop independent of either named cause.
- jpeg_quant "surviving Q96" could reflect that **Q96 itself imprints a detectable, class-correlated re-encoding signature** (reals and AIs entered Q96 from different prior compression states), i.e., the debiasing does not neutralize what the paper assumes it neutralizes. The paper gestures at this (compression-history) but does not treat it as a threat to the Outcome-C interpretation.

**Missing stakeholder perspectives.**
- A **benchmark maintainer** (what to fix in GenImage) and a **deployment/forensics practitioner** (does any of this transfer to in-the-wild images, which are almost all JPEG?) are absent. The latter is pointed: if real-world images are uniformly JPEG, the container confound may be *less* exploitable in deployment than the paper's alarm implies — or more. Unaddressed.

**Observations (non-defects).** Exceptional claim hygiene; the limitations ledger genuinely anticipates most reviewer objections; reproducibility is strong.

---

# Phase 2 — Editorial synthesis & decision

## Cross-reviewer consensus / disagreement matrix

| Issue | EIC | R1 Meth | R2 Domain | R3 Persp | DA | Status |
|---|---|---|---|---|---|---|
| Single seed → n=1 claims (M1) | (implied) | **MAJOR** | — | — | **MAJOR** | **Consensus MAJOR** |
| No no-bottleneck baseline; what does the bottleneck add? (M2) | **MAJOR** | **MAJOR** | (D1) | (p1) | **MAJOR→crit** | **Consensus MAJOR** (gating) |
| Novelty increment over Grommelt thin | **MAJOR** | — | **MAJOR (D1)** | — | **MAJOR** | **Consensus MAJOR** |
| No same-protocol baseline (Table 1) | (noted) | **MAJOR (M3)** | (d2) | — | (noted) | Majority MAJOR |
| Generative-trace vs compression-history unresolved | — | — | **MAJOR (D2)** | — | alt-explanation | Domain-flagged; central open problem |
| Own concepts' semantic faithfulness untested | — | — | — | **MAJOR (P1)** | (degeneracy) | Perspective-flagged; framing fix |
| Scope = GenImage/ImageNet only | — | — | MINOR (D3) | — | — | Minor |
| Calibration/ECE underspecified | — | MINOR (m1) | — | — | — | Minor |
| Cherry-picking of single-gen ckpt | — | — | — | — | **checked: clean** | No defect |

**No CRITICAL issue is confirmed** (DA's degeneracy concern is rated MAJOR-approaching-CRITICAL, contingent on M2). Per IRON RULE #4, Accept would be barred only by a confirmed CRITICAL; here the decision is driven by the **three-way MAJOR consensus** (single seed · no-bottleneck baseline · novelty framing).

## Editorial Decision Letter

**Decision: MAJOR REVISION.**

The panel finds a well-executed, exceptionally honest paper whose contribution is real but currently under-isolated. The reframing of a concept bottleneck as a *testability surface*, the **Q96-persistence result (Outcome C)**, and the **"intervention > weights"** dissociation are genuine and exportable. However, three issues prevent acceptance at a competitive venue:

1. **The audit-instrument claim is not yet empirically isolated** from a simpler alternative. Because the bottleneck collapses to one cue (the paper's own §4.3a), the panel needs evidence that the bottleneck *adds* localization value over post-hoc attribution on a plain same-backbone classifier (R1-M2, EIC, DA). Absent that, the contribution should be reframed around Outcome C + intervention-over-weights, with the bottleneck presented as a convenient vehicle rather than the contribution.
2. **Single-seed evidence** for claims about a learned solution (R1-M1, DA). Multi-seed mean±std is required for the headline metrics and the per-channel reliance ranking.
3. **Novelty increment over Grommelt** must be stated more precisely and defended (EIC, R2-D1, DA).

None of these requires new claims or rewriting the main line; (1) and (2) are added experiments, (3) is a sharpened contribution statement. For a *course/workshop* bar, the panel notes this is closer to **Minor Revision**, since the honesty and execution are strong and the gaps are openly disclosed.

*This decision letter was synthesized only from the five Phase-1 reports above; no critique was introduced at synthesis that is not traceable to a reviewer (anti-pattern #1).*

## Revision Roadmap (prioritized — directly usable as Stage-4 input)

**P0 — gating (do first):**
- **R-1 [Methodology, new experiment]:** add a **no-bottleneck baseline** — same dual-stream backbone + linear classifier with post-hoc attribution (Grad-CAM / linear probe), and show whether it localizes the compression confound as well as / worse than the bottleneck. Resolves M2 + DA-degeneracy + part of D1. *(Compute: one training run + attribution; single-GPU feasible.)*
- **R-2 [Methodology, new runs]:** re-run Route B (and ideally E-Delta) on **≥3 seeds**; report mean±std on 99.67/73.65/double-dissociation and confirm the jpeg_quant zero-out rank is seed-stable. Resolves M1.

**P1 — important (framing + one comparison):**
- **R-3 [Positioning, no new data]:** sharpen the **contribution statement** to foreground Outcome C + "intervention > weights"; explicitly enumerate what Grommelt's black-box analysis could not establish. Resolves D1, EIC-originality. *(No main-line rewrite — a tightened contributions paragraph.)*
- **R-4 [Methodology, optional new run]:** train **one baseline under P4** (same 4 generators) for a genuine Table-1 cell. Resolves M3.
- **R-5 [Framing]:** add one sentence distinguishing **functional reliance (tested) vs semantic faithfulness (not tested)**; relate "testability surface" to TCAV/insertion-deletion. Resolves P1, p1.

**P2 — recommended (hedges + small adds):**
- **R-6:** treat generative-trace-vs-compression-history as the **central open problem** (promote from footnote); if feasible, a matched-pipeline real-image mini-experiment (D2).
- **R-7:** soften audit-generality language; acknowledge GenImage/ImageNet scope (D3).
- **R-8:** specify ECE split (OOD if possible) (m1); tighten res128 wording to "frequency" not "container" (m3); add shortcut-learning references (d1); add benchmark-maintainer + deployment-practitioner notes (DA stakeholders); isolate/disclose the resolution-mismatch confound in the ForenSynths drop (DA alt-explanation).

**Unresolved-if-deferred → Acknowledged Limitations** (already partly present): if R-1/R-2 are not run, the audit-instrument claim must be explicitly downgraded to a hypothesis and the single-seed caveat stated in the main text, not only in Limitations.

---

*End of Stage 3 review package. Decision = Major Revision (competitive venue) / Minor (course-workshop). No manuscript file was modified.*
