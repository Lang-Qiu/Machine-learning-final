#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Mechanical md -> latex assembly for the AGID manuscript body + abstract.
Strips draft HTML comments, removes section numbers, maps problematic Unicode to
raw-TeX (\\ensuremath{...} forms, so no $-delimiters that pandoc would mangle when
followed by a digit), replaces the 4 markdown tables with \\input{} placeholders,
then pandoc with --shift-heading-level-by=-1 (## -> \\section). No prose is rewritten."""
import re, subprocess, sys, pathlib, os

DRAFT = pathlib.Path(r"E:\LQiu\lab_folder\Machine_learning\AGID_Project\Stage2_Writing\draft")
OUT   = pathlib.Path(r"E:\LQiu\lab_folder\Machine_learning\AGID_Project\Stage2_Writing\manuscript")

BODY_FILES = ["01_Introduction.md","02_RelatedWork.md","03_Method.md","04_Experiments.md",
              "05_Discussion.md","06_Limitations.md","07_Conclusion.md"]

# raw-TeX replacements (no $...$, so a following digit cannot break pandoc math parsing)
UNI = {
    "−":"-",
    "×":r"\ensuremath{\times}",
    "≥":r"\ensuremath{\geq}",
    "≤":r"\ensuremath{\leq}",
    "→":r"\ensuremath{\rightarrow}",
    "↔":r"\ensuremath{\leftrightarrow}",
    "≈":r"\ensuremath{\approx}",
    "Δ":r"\ensuremath{\Delta}",
    "²":r"\textsuperscript{2}",
    "α":r"\ensuremath{\alpha}",
    "β":r"\ensuremath{\beta}",
    "λ":r"\ensuremath{\lambda}",
    "…":r"\ldots{}",
    "§":r"\S{}",
    "—":"---",
    "–":"--",
    " ":" ",
}

def strip_comments(t):
    return re.sub(r"<!--.*?-->", "", t, flags=re.DOTALL)

def strip_header_numbers(t):
    return re.sub(r"^(#{2,3})\s+\d+(?:\.\d+)*[a-z]?\.?\s+", r"\1 ", t, flags=re.M)

def fix_unicode(t):
    # pre-existing literal inline math "$\sim$N" mangles in pandoc (closing $ before digit) -> raw tex
    t = t.replace(r"$\sim$", r"\ensuremath{\sim}")
    for k, v in UNI.items():
        t = t.replace(k, v)
    t = t.replace("~", r"\ensuremath{\sim}")  # ascii tilde used as "approximately"
    return t

def replace_tables(t):
    t = re.sub(r"^\*Table 1 \(schematic\).*$", "XXTABLEONEXX", t, flags=re.M)
    lines = t.split("\n"); out = []; i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith("|"):
            block = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                block.append(lines[i]); i += 1
            head = block[0]
            tok = ("XXTABLETWOXX"   if ("Generator" in head and "Fake-acc" in head) else
                   "XXTABLETHREEXX" if "Perturbation" in head else
                   "XXTABLEFOURXX"  if "Model (ForenSynths)" in head else None)
            if tok:
                out += ["", tok, ""]
            else:
                out += block
            continue
        out.append(lines[i]); i += 1
    return "\n".join(out)

def preprocess(t, is_experiments):
    t = strip_comments(t)
    t = strip_header_numbers(t)
    if is_experiments:
        t = replace_tables(t)
    return fix_unicode(t).strip() + "\n"

def pandoc(md_text):
    p = subprocess.run(
        ["pandoc","-f","markdown-auto_identifiers","-t","latex","--wrap=none",
         "--shift-heading-level-by=-1"],
        input=md_text.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        sys.stderr.write(p.stderr.decode("utf-8","replace")); sys.exit(2)
    return p.stdout.decode("utf-8")

def sub_tokens(tex):
    return (tex.replace("XXTABLEONEXX",  r"\input{table1_baselines}")
               .replace("XXTABLETWOXX",  r"\input{table2_detection}")
               .replace("XXTABLETHREEXX",r"\input{table3_confound}")
               .replace("XXTABLEFOURXX", r"\input{table4_forensynths}"))

# ---------------------------------------------------------------------------
# RETIRED 2026-06-07. sections/body.tex and sections/abstract.tex are now the
# HAND-MAINTAINED source of truth. Since 2026-06-03 they have accumulated work
# that is NOT in draft/*.md: the Stage-4 revision (multi-seed, no-bottleneck
# baseline, §4.5), the 6 re-review minor fixes, three figures (\input{fig_*}),
# and a typography pass. This generator regenerates body.tex from the *stale*
# drafts and would OVERWRITE and destroy all of that. It is intentionally
# disabled. See draft/_CANONICAL_SOURCE_MOVED.md. Override only after re-syncing
# the drafts AND re-adding the figure \input lines by hand.
if os.environ.get("AGID_ALLOW_REBUILD") != "1":
    sys.exit("_build_body.py is RETIRED: sections/body.tex is now hand-maintained "
             "and diverged from draft/*.md (Stage-4 fixes + figures). Refusing to "
             "overwrite. Set AGID_ALLOW_REBUILD=1 only if you understand this will "
             "destroy the hand-edits. See draft/_CANONICAL_SOURCE_MOVED.md.")

parts = [preprocess((DRAFT/fn).read_text(encoding="utf-8"), fn.startswith("04")) for fn in BODY_FILES]
body_tex = sub_tokens(pandoc("\n\n".join(parts)))
(OUT/"sections"/"body.tex").write_text(body_tex, encoding="utf-8")

abs_raw = strip_comments((DRAFT/"08_Abstract.md").read_text(encoding="utf-8"))
abs_raw = re.sub(r"^##\s+Abstract\s*$", "", abs_raw, flags=re.M)
abs_tex = pandoc(fix_unicode(abs_raw.strip())+"\n")
(OUT/"sections"/"abstract.tex").write_text(abs_tex, encoding="utf-8")

# sanity checks
import collections
checks = {
    "\\section{": body_tex.count(r"\section{"),
    "\\subsection{": body_tex.count(r"\subsection{"),
    "stray \\$": body_tex.count(r"\$") + abs_tex.count(r"\$"),
    "residual §": body_tex.count("§") + abs_tex.count("§"),
    "\\input{table": body_tex.count(r"\input{table"),
    "\\tightlist": body_tex.count(r"\tightlist"),
}
print("body chars", len(body_tex), "| abstract chars", len(abs_tex))
for k,v in checks.items(): print(f"  {k}: {v}")
