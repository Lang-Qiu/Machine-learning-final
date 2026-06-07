# ⚠️ These draft/*.md files are NO LONGER the source of truth (since 2026-06-03)

The canonical manuscript is now the **hand-maintained LaTeX** at:

    AGID_Project/Stage2_Writing/manuscript/sections/body.tex   (+ abstract.tex)

## Why these markdown drafts are stale

After these drafts were written (2026-06-03), substantial work was applied
**directly to `manuscript/sections/body.tex`**, and was *never* back-ported here:

1. **Stage-4 revision (resolving peer-review MAJOR issues M1/M2/M3):**
   - multi-seed replication (seeds 1 & 2) in §4.2 / §4.3a
   - no-bottleneck baseline (`BaselinePlain`) + new §4.5 "What the bottleneck adds"
   - compression-*pair* reframing + seed-robustness caveats; Outcome C marked single-seed
2. **Six re-review minor fixes (2026-06-07):** Appendix-A ref removed, Table-1 seed
   footnote, §5 scoping, §4.5 "and/or", ECE scoped to SD-1.4 val, res128 clause.
3. **Three figures (2026-06-07):** `\input{fig_architecture}` (TikZ), `\input{fig_reliance}`,
   `\input{fig_generalization}` — none of which this markdown or the generator knows about.
4. **A conservative typography pass** (appositive em-dash → parentheses/colon).

So `draft/04_Experiments.md` etc. are roughly **two revisions behind** the real paper.

## Do NOT regenerate body.tex from these drafts

`manuscript/_build_body.py` (pandoc md→LaTeX) used to overwrite `sections/body.tex`
from these files. It is now **disabled** (it `sys.exit`s unless `AGID_ALLOW_REBUILD=1`)
precisely because running it would destroy all of the above. Edit `sections/body.tex`
directly instead.

To rebuild the PDF: `latexmk -pdf manuscript.tex` (figures: `python figs/make_figures.py`).
