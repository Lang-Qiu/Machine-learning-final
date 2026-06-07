# Submission Checklist — AGID Paper (CBNet-AGID audit)

**Status as of 2026-06-03:** submission-shaped compact draft. Final readiness pass PASSED — see "Verified" below.
This checklist is **only the items that still require your manual action** before you submit. Nothing here blocks
compilation; the PDF builds clean today.

---

## ✅ Already verified (no action needed)

- Document structure complete and in order: title page → Abstract → §1 Introduction → §2 Related Work →
  §3 Method → §4 Experiments → §5 Discussion → §6 Limitations → §7 Conclusion → References.
- **14-page PDF** compiles clean (`latexmk` exit 0; 0 LaTeX errors; 0 undefined citations; 29/29 bibtex entries,
  0 warnings; 0 overfull boxes > 20 pt).
- Tables 1–4 all render with clear captions; Table 1 carries protocol tags + comparability column; every baseline
  row is cited.
- No stray draft comments, `TODO`s, or placeholder tokens leaked into the manuscript (the only `TODO` is the
  intentional author marker below).
- All numbers remain the Stage-2.5-verified values; all citation metadata web-verified.

---

## ☐ Manual items before submission

### 1. Author block — REQUIRED  (`manuscript.tex`, lines ~23–29)
Replace the bracketed placeholders with your real details, then delete the `% TODO (H-1)` line:
```latex
\author{%
  [Your Name] \\              <-- replace with your name
  Shanghai Jiao Tong University \\
  \texttt{[email@example.com]} <-- replace with your email
}
```
*(The email is wrapped in `\texttt{}` for a technical reason — if you remove the wrap, make sure no `[` directly
follows a `\\`, or LaTeX will mis-read it as a length argument.)*

### 2. Confirm submission style — REQUIRED (course-dependent)
- **Citation format:** currently `\bibliographystyle{plainnat}` → author-year, e.g. "(Koh et al., 2020)".
  If the course requires numbered `[1]` citations, change to `\usepackage[final,numbers]{neurips_2022}` (top of
  `manuscript.tex`) and `\bibliographystyle{unsrtnat}` (or `plainnat` still works with the `numbers` option).
- **Named vs anonymous:** the file uses `\usepackage[final]{neurips_2022}` → author names **shown** (correct for a
  named course submission). If anonymous review is required, drop `final` → `\usepackage{neurips_2022}`.

### 3. Confirm length / page limit — CHECK
Current = ~14 pages / ~6,680 words (compact). If the course sets a different page or word target, that is a
content-expansion pass (deliberately not done here). No padding was added.

### 4. Optional — figures
No figures are included. If the rubric expects one, an architecture / audit-pipeline diagram could be added later
(a content task, out of scope for this readiness pass).

### 5. Optional — Grommelt citation venue
`grommelt2024fakejpeg` is cited as the arXiv 2403.17608 preprint (2024). If you prefer a published version and
confirm one exists, update the entry in `ref.bib`.

### 6. Final human proofread — RECOMMENDED
A read-through of `manuscript.pdf` is advised. Structure, numbers, and citations were machine-verified, but a
human eye on phrasing/flow is good practice before submission.

---

## How to rebuild after any edit
```
cd AGID_Project/Stage2_Writing/manuscript
# if you edited the draft .md prose (not needed for the items above):
python _build_body.py
# always, to regenerate the PDF (runs pdflatex + bibtex):
latexmk -pdf manuscript.tex
```
Requires a TeX distribution (MiKTeX/TeX Live) with `natbib`, `booktabs`, `amsmath`, `amssymb`; `neurips_2022.sty`
is bundled in this folder.

---

*Editing the author block (item 1) is the only strictly required step; items 2–3 are course-policy confirmations;
items 4–6 are optional/recommended.*
