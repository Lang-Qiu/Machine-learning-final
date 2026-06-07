# Evidence Spine Lock — AGID Paper (Stage 2)

**Status:** 🔒 LOCKED 2026-06-02 · authoritative · **all downstream chapters MUST NOT contradict this**
**Authority:** user resume-point directive (2026-06-02) + independently verified against
`Code/Results/edelta/gate3_audit/gate3_full_audit.json`
**Retires:** the smoke-level *"reliance migrates / distributes"* narrative — confirmed overfit artifact
(`verdict.smoke_pattern_confirmed = false`; smoke eval was on the training set).

---

## 0. One-sentence spine

> Grommelt-style **JPEG-Q96 debiasing** produces a clean **detection double-dissociation** between the
> biased (Route B) and debiased (E-Delta) models, **without** producing functional **de-reliance**: both
> models still route essentially all discriminative signal through the single `jpeg_quant` concept slot,
> and the **causal zero-out — not the linear-head weight magnitude — is the authority** that establishes this.

This is **Outcome C (PERSISTENCE)**, not Outcome B (migration).

---

## 1. Lead with intervention, not weights  [LOCKED]

The linear-head weight on `jpeg_quant` shrinks under debiasing (Route B −13.53 → E-Delta −9.11, only
**1.49×**), and the auto-`verdict` block in the JSON therefore flags `distributed_reliance = true`. **That flag
is descriptive only and must NOT drive the narrative.** The causal zero-out on the *debiased* model shows
functional reliance is undiminished:

| Concept zeroed (E-Delta / debiased) | mean acc | Δ vs full |
|---|---|---|
| **jpeg_quant** | **51.8%** | **−48.18 pp** |
| freq_radial | 99.1% | −0.90 pp |
| texture_geometry | 99.65% | −0.35 pp |
| color_manifold | 99.73% | −0.28 pp |
| hf_noise | 99.75% | −0.25 pp |
| bitplane_lsb | 99.78% | −0.22 pp |

Concrete characterization: zeroing `jpeg_quant` collapses **real_acc to 2.6–4.2%** while fake_acc stays 100%
— i.e. `jpeg_quant` is functionally the slot that lets the model recognize a **real** image; without it
everything reads as fake. **Rule: weight magnitude is descriptive; the zero-out (causal intervention) is the
evidence of record.**

---

## 2. Double dissociation, NOT de-reliance  [LOCKED]

| Detection (mean acc) | Route B (biased) | E-Delta (Q96-debiased) | Reading |
|---|---|---|---|
| **Q96-val** (4 gens) | 89.75% | **99.78%** | each model wins on its *matched* distribution |
| **raw-OOD** (GLIDE/Wukong/VQDM) | **99.73%** | 95.0% | → clean **double dissociation** |
| `jpeg_quant` zero-out Δacc | −49.5 pp | **−48.18 pp** | **both remain functionally dependent on `jpeg_quant`** |

Narrative: Q96 debiasing **re-tunes which compression regime the `jpeg_quant` slot is matched to** — it does
**not** wean the model off the slot. Route B (trained on raw PNG-ai/JPEG-nature) excels on raw/OOD-like data
that still carries the container gap; E-Delta (trained on uniform Q96) excels on Q96-val; **neither de-relies.**

**Third axis — cross-architecture (ForenSynths, D-28, added 2026-06-02):** Route B = **73.65%** (AUC .82) on
ForenSynths vs **99.67%** on GenImage-OOD = **−26pp** on a container-MISMATCHED benchmark; and 73.65% vs
single-gen Route A's **52.21%** = multi-gen buys GENUINE partial cross-architecture generalization. Reading =
**partial confound inflation + genuine partial generalization** — NOT pure confound (the −26pp is entangled with
GAN-vs-diffusion distribution shift). Clean confound isolation stays B1/B2/A3/E-Delta; ForenSynths = external-
validity corroboration. Detection claim therefore = **"competitive," not "SOTA"** (P4-vs-P1 protocol divergence).

---

## 3. Outcome-C label + the claim ceiling  [LOCKED · with scope guard]

**Correct claim:** `jpeg_quant` is **not** a trivial file-format cue. E-Delta was trained on data where the
PNG-vs-JPEG container gap is removed (uniform Q96), yet zero-out still costs −48 pp ⇒ the slot encodes a
**frequency/compression-domain channel that is not reducible to the container label.**

**❌ Do NOT claim** "the model moved away from `jpeg_quant`" (false — Outcome C).

**⚠️ SCOPE GUARD (🔒 LOCKED 2026-06-02, user-confirmed):** do **not** escalate the
correct claim to *"`jpeg_quant` captures generative-pipeline semantics."* Uniform Q96 re-encoding standardizes
the **final** compression step but does **not** erase the images' **cumulative upstream history** (generative
decoder frequency signature + any pre-existing compression for AI; camera ISP + original JPEG for real). So the
surviving channel could be (a) genuine generative traces, (b) residual cumulative-compression/frequency
statistics that differ real-vs-AI even after uniform Q96, or a mix — and **we ran no experiment that separates
(a) from (b).** The user's own Point-3 wording ("compressed container/**generation** evidence channel") already
hedges this; the guard just makes the ceiling explicit so chapter prose does not drift.

**Safe published form:** *"After Q96 debiasing removes the container-label confound, the `jpeg_quant` slot still
carries essentially all of the model's functional discriminative load. The channel it encodes is therefore not
reducible to the PNG-vs-JPEG container artifact; whether it reflects generative-pipeline traces proper or
residual compression-history statistics is unresolved and stated as a limitation."*

---

## 4. Methodological takeaway (the contribution)  [LOCKED]

Concept-bottleneck weights are an **unreliable proxy for functional reliance**; intervention / zero-out is the
authority. This is exactly why the CBM is valuable as an **audit instrument**: not because its concepts are
*automatically interpretable*, but because it exposes a **surface on which we can causally test whether an
apparent explanation is functionally real.** (Consistent with `memory/project-agid-audit-pivot`.)

---

## 5. Prohibited claims (prose drift watchlist)

1. ❌ "reliance migrates / distributes after debiasing" (smoke artifact — retired).
2. ❌ "the model stopped using `jpeg_quant`" (Outcome C says the opposite).
3. ❌ "`jpeg_quant` proves the model reads generative semantics" (exceeds the claim ceiling, §3 guard).
4. ❌ "the 6 concepts are interpretable / all used" (head functionally uses ~1; `project-agid-audit-pivot`).
5. ❌ leading with weight magnitudes instead of the zero-out.
6. ❌ "SOTA detection" — CBNet is protocol P4, baselines are P1; not apples-to-apples (E1). Use "competitive."
7. ❌ "pure confound / no generalization" — ForenSynths 73.65% > single-gen 52.21% refutes it.
8. ❌ attributing the full −26pp (GenImage-OOD → ForenSynths) to confound — entangled with GAN-vs-diffusion shift.

## 6. Mandatory Limitations-section entries

- L-a: Q96 debiasing homogenizes only the final compression step; generative-trace vs. compression-history
  attribution of the surviving `jpeg_quant` channel is unresolved (§3).
- L-b: smoke→full discrepancy = training-set-eval overfit artifact; only the full-scale GATE 3 result stands.
- L-c: raw cross-generator resolution span (8×) + PNG/JPEG container leakage are dataset-level and not
  removable in-model (carry forward from CLAUDE.md Route-B risk notes).

---

*Source numbers: `Code/Results/edelta/gate3_audit/gate3_full_audit.json`
(detection, concept_weights, a3_ablation_debiased, cohens_d_debiased, verdict). Verified 2026-06-02.*
