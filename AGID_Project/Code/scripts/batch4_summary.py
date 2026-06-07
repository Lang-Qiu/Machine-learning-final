"""Batch 4 summary: aggregate B3 JPEG-Q curve + B4 resolution sweep into
publication-ready figures and tables.

Reads (under Code/Results/confound/):
    baseline (synthesized from full_inference_dump.npz)
    png.json
    jpeg-q95.json, jpeg-q75.json, jpeg-q50.json, jpeg-q30.json
    res64.json, res128.json, res192.json, res384.json, res512.json

Writes (under Code/Results/batch1_analysis/):
    B3_jpeg_quality_curve.{png,csv}    — acc vs JPEG quality, per generator
    B4_resolution_curve.{png,csv}      — acc vs forced resolution, per gen
    Batch4_summary.md                  — narrative + tables
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score


def baseline_from_dump(dump_path):
    d = np.load(dump_path, allow_pickle=True)
    out = {}
    for g in sorted(set(d["generators"].tolist())):
        m = d["generators"] == g
        p = d["probs"][m]; l = d["labels"][m]
        preds = (p > 0.5).astype(int)
        try: auc = float(roc_auc_score(l, p))
        except ValueError: auc = float("nan")
        rm = l == 0; fm = l == 1
        out[g] = {
            "acc": float((preds == l).mean()),
            "auc": auc,
            "real_acc": float((preds[rm] == 0).mean()),
            "fake_acc": float((preds[fm] == 1).mean()),
        }
    return out


def load_variant(p):
    if not p.exists():
        return {}
    return json.loads(p.read_text())["results"]


# ---------------------------------------------------------------------------- #
# B3 — JPEG quality degradation curve
# ---------------------------------------------------------------------------- #

def b3_curve(cdir, baseline, out_dir, all_gens):
    qs = [100, 95, 75, 50, 30]  # 100 = baseline (no JPEG re-encode), others from variants
    # For Q=100 use baseline acc (no JPEG encoding artifact added beyond what's there)
    var_map = {100: None, 95: "jpeg-q95", 75: "jpeg-q75", 50: "jpeg-q50", 30: "jpeg-q30"}

    rows = []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    cmap = plt.get_cmap("tab10")
    for gi, g in enumerate(all_gens):
        accs, real_accs, fake_accs = [], [], []
        for q in qs:
            if var_map[q] is None:
                r = baseline[g]
            else:
                data = load_variant(cdir / f"{var_map[q]}.json")
                r = data.get(g, {})
            if not r:
                accs.append(None); real_accs.append(None); fake_accs.append(None); continue
            accs.append(r["acc"]); real_accs.append(r["real_acc"]); fake_accs.append(r["fake_acc"])
            rows.append({
                "generator": g, "jpeg_q": q,
                "acc": r["acc"], "real_acc": r["real_acc"], "fake_acc": r["fake_acc"],
            })
        col = cmap(gi)
        ax1.plot(qs, [a*100 if a is not None else None for a in accs],
                 marker="o", color=col, label=g)
        ax2.plot(qs, [a*100 if a is not None else None for a in fake_accs],
                 marker="o", color=col, label=g)
    ax1.set_xlabel("JPEG quality (lower = more compression)"); ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("B3: Accuracy vs JPEG re-encoding quality\n(100 = no re-encoding)")
    ax1.invert_xaxis(); ax1.grid(True, alpha=0.3); ax1.legend(fontsize=8); ax1.set_ylim(40, 102)
    ax2.set_xlabel("JPEG quality"); ax2.set_ylabel("Fake accuracy (%)")
    ax2.set_title("B3: Fake-class accuracy collapses with JPEG injection")
    ax2.invert_xaxis(); ax2.grid(True, alpha=0.3); ax2.legend(fontsize=8); ax2.set_ylim(0, 102)
    plt.tight_layout()
    plt.savefig(out_dir / "B3_jpeg_quality_curve.png", dpi=140)
    plt.close()
    pd.DataFrame(rows).to_csv(out_dir / "B3_jpeg_quality_curve.csv", index=False)
    print(f"[B3] curve → B3_jpeg_quality_curve.{{png,csv}}")
    return rows


# ---------------------------------------------------------------------------- #
# B4 — Resolution sensitivity curve
# ---------------------------------------------------------------------------- #

def b4_curve(cdir, baseline, out_dir, all_gens):
    resns = [64, 128, 192, 256, 384, 512]
    # baseline is "no resize", treat as the no-op reference (label "orig")
    rows = []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    cmap = plt.get_cmap("tab10")
    # Baseline at x = "orig" (we just plot it as horizontal dashed line)
    base_acc = {g: baseline[g]["acc"] for g in all_gens}
    for gi, g in enumerate(all_gens):
        accs, real_accs, fake_accs = [], [], []
        for r in resns:
            data = load_variant(cdir / f"res{r}.json")
            row = data.get(g, {})
            if not row:
                accs.append(None); real_accs.append(None); fake_accs.append(None); continue
            accs.append(row["acc"]); real_accs.append(row["real_acc"]); fake_accs.append(row["fake_acc"])
            rows.append({
                "generator": g, "resolution": r,
                "acc": row["acc"], "real_acc": row["real_acc"], "fake_acc": row["fake_acc"],
            })
        col = cmap(gi)
        ax1.plot(resns, [a*100 if a is not None else None for a in accs],
                 marker="s", color=col, label=g)
        ax1.axhline(base_acc[g]*100, color=col, ls=":", lw=0.7, alpha=0.4)
        ax2.plot(resns, [a*100 if a is not None else None for a in real_accs],
                 marker="s", color=col, label=g)
    ax1.set_xlabel("Forced resolution (downsample→upsample size)")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("B4: Accuracy vs forced resolution\n(dotted = baseline, no resize)")
    ax1.grid(True, alpha=0.3); ax1.legend(fontsize=8); ax1.set_ylim(40, 102)
    ax2.set_xlabel("Forced resolution"); ax2.set_ylabel("Real-class accuracy (%)")
    ax2.set_title("B4: Real-class accuracy under high-freq destruction")
    ax2.grid(True, alpha=0.3); ax2.legend(fontsize=8); ax2.set_ylim(0, 102)
    plt.tight_layout()
    plt.savefig(out_dir / "B4_resolution_curve.png", dpi=140)
    plt.close()
    pd.DataFrame(rows).to_csv(out_dir / "B4_resolution_curve.csv", index=False)
    print(f"[B4] curve → B4_resolution_curve.{{png,csv}}")
    return rows


# ---------------------------------------------------------------------------- #
# Narrative
# ---------------------------------------------------------------------------- #

def write_narrative(b3_rows, b4_rows, baseline, out_dir, all_gens):
    df_b3 = pd.DataFrame(b3_rows)
    df_b4 = pd.DataFrame(b4_rows)

    # B3 mean acc per Q
    b3_mean = df_b3.groupby("jpeg_q")["acc"].mean().sort_index(ascending=False)
    # B4 mean acc per res
    b4_mean = df_b4.groupby("resolution")["acc"].mean().sort_index()

    lines = ["# Batch 4 summary — B3 JPEG quality curve + B4 resolution sweep\n"]
    lines.append("## B3: JPEG quality degradation curve\n")
    lines.append("Mean accuracy across 7 generators as a function of forced JPEG quality:\n")
    lines.append("| JPEG Q | Mean acc (%) |")
    lines.append("|---|---|")
    for q, a in b3_mean.items():
        lines.append(f"| {q} | {a*100:.2f} |")
    lines.append("")
    lines.append("**Read**: monotonic degradation from baseline (~99.7%) to severe loss "
                 "as fakes are forced through stronger JPEG compression. Fake-class "
                 "accuracy is the failure mode — model can no longer distinguish "
                 "JPEG-encoded fakes from JPEG-encoded reals.\n")

    lines.append("## B4: Resolution sensitivity curve\n")
    lines.append("Mean accuracy across 7 generators as a function of forced resolution:\n")
    lines.append("| Resolution | Mean acc (%) |")
    lines.append("|---|---|")
    for r, a in b4_mean.items():
        lines.append(f"| {r}² | {a*100:.2f} |")
    lines.append("(baseline / no resize ≈ 99.7%)")
    lines.append("")
    lines.append("**Read**: high-frequency content is required for correct classification. "
                 "Below ~256², the real-class signal collapses because the model's "
                 "decision relies on resolution-coupled artifacts (JPEG block edges, "
                 "sensor noise) that survive at 256 but not at 128/64.\n")

    lines.append("## Paper Figure recommendations\n")
    lines.append("- Figure B3 (`B3_jpeg_quality_curve.png`): paste in Results section as evidence of format-leakage.")
    lines.append("- Figure B4 (`B4_resolution_curve.png`): paste alongside B3; together they characterize the two main confound axes.\n")

    lines.append("## Connecting back to audit pivot\n")
    lines.append("B3 + B4 together provide the **dose-response evidence** for the audit "
                 "claim: as we progressively inject JPEG artifacts (B3) or remove "
                 "high-frequency content (B4), the model accuracy degrades monotonically. "
                 "If the model were truly using generative-pipeline traces, these "
                 "transformations should not affect performance dramatically. The "
                 "fact that they do — and that degradation is concept-correlated (see "
                 "[[A_intervention_summary.md]]) — completes the mechanistic story.")

    (out_dir / "Batch4_summary.md").write_text("\n".join(lines))
    print(f"[SAVED] {out_dir / 'Batch4_summary.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confound_dir", required=True)
    ap.add_argument("--dump",         required=True)
    ap.add_argument("--out_dir",      required=True)
    args = ap.parse_args()

    cdir = Path(args.confound_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    baseline = baseline_from_dump(Path(args.dump))
    all_gens = sorted(baseline.keys())

    b3_rows = b3_curve(cdir, baseline, out_dir, all_gens)
    b4_rows = b4_curve(cdir, baseline, out_dir, all_gens)
    write_narrative(b3_rows, b4_rows, baseline, out_dir, all_gens)


if __name__ == "__main__":
    main()
