# Manuscript Assembly Report — AGID Paper

**Date:** 2026-06-03 · **Scope:** mechanical assembly only (no prose claims rewritten, no new experiments, no new claims).
**Build:** `latexmk -pdf` (MiKTeX pdflatex + bibtex) → **14-page PDF, 0 errors, 0 undefined citations, 0 bibtex warnings, 0 overfull boxes >20pt.**

---

## Deliverable paths

| Artifact | Path |
|---|---|
| **Assembled manuscript (main)** | `AGID_Project/Stage2_Writing/manuscript/manuscript.tex` |
| **Compiled PDF** | `AGID_Project/Stage2_Writing/manuscript/manuscript.pdf` (14 pp) |
| **ref.bib** | `AGID_Project/Stage2_Writing/manuscript/ref.bib` (29 entries) |
| **Table 1 (baselines, FIX-5)** | `AGID_Project/Stage2_Writing/manuscript/table1_baselines.tex` (`\input` into §4.1) |
| Tables 2–4 | `table2_detection.tex`, `table3_confound.tex`, `table4_forensynths.tex` (`\input` in §4.2/§4.3b/§4.4) |
| Body (pandoc-generated) | `manuscript/sections/body.tex`; abstract `manuscript/sections/abstract.tex` |
| Style file | `manuscript/neurips_2022.sty` (copied from course template) |
| Reproducible converter | `manuscript/_build_body.py` (md→LaTeX driver; re-run if drafts change) |

Structure: title page → English abstract → §1 Introduction … §7 Conclusion → references. Section auto-numbering
(Method = 3 → 3.1–3.7; Experiments = 4 → 4.1–4.4) **matches** the literal cross-references resolved in Stage 2.5
(§4.1, §4.3, §6, §3.3, §3.7, etc.). Prose word count ≈ **6,680 words** (Intro 1329 / RW 672 / Method 1096 /
Experiments 1404 / Discussion 952 / Limitations 686 / Conclusion 298 / Abstract 241).

---

## Auto-processed (mechanical) — done

1. **Markdown → LaTeX** via pandoc per the full body + abstract. pandoc handled `%`, `_`, `&`, quote and dash
   escaping and passed `\citep`/`\citet`/`\footnote`/`$…$` through as raw TeX. Draft HTML comments stripped.
2. **Section numbers stripped** (`## 1. Introduction` → `\section{Introduction}`); headings shifted so `##`→
   `\section`, `###`→`\subsection` → LaTeX auto-numbers, matching the prose's literal "§3.3 / §4.1 / Table 2" refs.
3. **Unicode → LaTeX** (so pdflatex+T1 compiles): `− × ≥ ≤ → ² Δ § — –` and the "approximately" tilde mapped to
   `\ensuremath{…}` / `\textsuperscript{2}` / `\S{}` / `---` (raw-TeX forms, avoiding the `$…$`-before-digit
   pandoc pitfall). Verified: 0 stray `\$`, 0 residual `§` in output.
4. **FIX-5 — Table 1 built** from `E1_sota_baselines.md` as a numbered float, **not a leaderboard**: every row
   carries a **protocol tag (P1/P2/P4)** and an explicit **comparability flag** (re-impl. / reported / not
   comparable / ours), with a footnote spelling out P1 vs P2 vs P4 and why LOTA's P2 mean is not comparable.
   Rows = CNNSpot 64.2, UnivFD 79.5, AIDE 86.9, NPR 88.6, DRCT 89.5, C2P-CLIP 95.8, LOTA 98.9 (P2), + CBNet two
   sub-rows (in-dist ≥99.85 / held-out OOD 99.67, P4) — exactly matching the prose's representative list.
   **LOTA 98.9% re-confirmed** against `E1_sota_baselines.md` §2/row-F (P2 mean, arXiv 2510.14230 Table 2).
5. **Tables 2–4** hand-built as numbered floats from artifact files (`F3_main_results.md`,
   `B_confound_summary.md`, the three ForenSynths JSONs) and placed in order so literal "Table 2/3/4" prose refs
   resolve correctly.
6. **ref.bib generated** with the 26 cited keys. The 5 FIX-4 entries use web-verified metadata (corrected titles
   for ThinkFake/ForenX/Grounded; full author lists for AIGI-Holmes/LOTA/Grommelt). **No fabricated titles
   remain.** bibtex: 26/26 entries used, 0 warnings. `plainnat` style → author-year citations.
7. **O-3 — Grommelt** cited as `@article … arXiv preprint arXiv:2403.17608, 2024`. No published (non-preprint)
   version surfaced during assembly; left as preprint per your instruction (not changed to an unconfirmed venue).
8. **`\tightlist` defined** in the preamble (pandoc emits it for the RQ list); compile is clean.

## FIX-6 — single-gen checkpoint robustness (integrity note, not a body change)

The manuscript's §4.4 cites the single-generator ForenSynths number **52.21%** (the 20k checkpoint,
`cbnet_forensynths.json`). The alternative **100k "main" single-gen run is 51.70%**
(`cbnet_100k_forensynths.json`). **Both are ≈ chance**, so the "single-gen near chance → multi-gen +21pp"
conclusion does **not** depend on a cherry-picked checkpoint (the cited 52.21% is in fact the *higher* of the two
by 0.5pp). No change made to the body; logged here per your instruction.

---

## Pre-submission cleanup pass (2026-06-03, after assembly accepted)

User rulings applied; manuscript re-compiled clean (`latexmk` exit 0).

| # | Item | Status |
|---|---|---|
| H-1 | Author block | **DONE** — set to `\author{[Your Name] \\ Shanghai Jiao Tong University \\ \texttt{[email@example.com]}}` with a `% TODO (H-1)` comment directly above. Name/email left as bracketed placeholders for **you** to fill manually (real name/email not inserted). *(Email wrapped in `\texttt{}` so the preceding `\\` does not mis-parse `[...]` as an optional length argument — the one recoverable error caught and fixed during this pass.)* |
| H-2 | Citation style | **KEPT** `\bibliographystyle{plainnat}` (author-year). Template does not require numeric; no change. |
| H-3 | Table 1 baseline citations | **DONE** — every baseline row now cited. Added 3 verified bib entries: `yan2024aide` (Yan et al., *A Sanity Check for AI-generated Image Detection*, arXiv 2406.19435), `chen2024drct` (Chen et al., *DRCT…*, **ICML 2024**), `tan2024c2pclip` (Tan et al., *C2P-CLIP…*, arXiv 2408.09647). Metadata **web-verified 2026-06-03** (authors + title + venue); the project bib's prior wrong AIDE title ("A Hybrid Approach…") corrected in `03_Bibliography.md` too. Table 1 venue column updated (DRCT → ICML 2024; AIDE/C2P-CLIP → arXiv 2024). |
| H-4 | Length | **DEFERRED (your ruling)** — kept at ≈6.7k words / 14 pp as the compact draft. No padding. Expansion pass only if the course later requires it. |
| H-5 | Figures | **DEFERRED (your ruling)** — none added (not a blocker). Optional evidence-spine / audit-pipeline figure later. |
| H-6 | Grommelt venue | **KEPT** as arXiv 2403.17608 preprint; no hunt for a published version. |
| H-7 | Title line break | **DONE** — cosmetic two-line break kept (`…Competitive\\ AI-Generated…`); **title text unchanged**. |

**Post-cleanup build check (your 5 confirmation points):**
- ✅ 0 undefined citations
- ✅ 0 bibtex warnings (29/29 entries used)
- ✅ Table 1 baseline citations complete (all 7 baselines + 2 CBNet sub-rows; AIDE/DRCT/C2P-CLIP now cited)
- ✅ Author placeholder clearly marked (`% TODO (H-1)` + bracketed `[Your Name]` / `[email@example.com]`)
- ✅ PDF still compiles cleanly (`latexmk` exit 0, 14 pp, 0 LaTeX errors, 0 overfull boxes >20pt)

## Still open (deferred by your ruling, not blockers)

- **H-4 length** (expansion is a content pass), **H-5 figures** (optional), **H-6 Grommelt** (preprint unless a published version is confirmed). Nothing here blocks compilation or submission of the compact draft.

---

## How to rebuild

```
cd AGID_Project/Stage2_Writing/manuscript
python _build_body.py        # regenerate sections/body.tex + sections/abstract.tex from draft/*.md
latexmk -pdf manuscript.tex  # pdflatex + bibtex passes -> manuscript.pdf
```

*All numbers in the assembled manuscript remain the Stage-2.5-verified values; assembly changed only markup,
tables, citations, and cross-references — no claim was rewritten and no number was altered.*
