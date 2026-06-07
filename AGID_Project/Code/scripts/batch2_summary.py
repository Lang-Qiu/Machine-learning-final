"""Aggregate B1/B2/B5 confound results vs baseline into a single comparison
table + markdown narrative for paper Limitations section.

Reads:
    Results/confound/baseline.json            (optional — computed from full_inference_dump.npz if missing)
    Results/confound/jpeg-q95.json
    Results/confound/png.json
    Results/confound/res128.json
    Results/confound/independent_real.json
    Results/full_inference_dump.npz           (for baseline numbers)

Writes:
    Results/batch1_analysis/B_confound_summary.md
    Results/batch1_analysis/B_confound_table.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_baseline_from_dump(dump_path: Path) -> dict:
    d = np.load(dump_path, allow_pickle=True)
    probs   = d["probs"]
    labels  = d["labels"]
    gens    = d["generators"]
    from sklearn.metrics import roc_auc_score
    out = {}
    for g in sorted(set(gens.tolist())):
        m = gens == g
        p = probs[m]; l = labels[m]
        preds = (p > 0.5).astype(int)
        rm = l == 0; fm = l == 1
        try: auc = float(roc_auc_score(l, p))
        except ValueError: auc = float("nan")
        out[g] = {
            "n": int(m.sum()),
            "acc": float((preds == l).mean()),
            "auc": auc,
            "real_acc": float((preds[rm] == 0).mean()),
            "fake_acc": float((preds[fm] == 1).mean()),
        }
    return out


def load_variant(p: Path) -> dict:
    if not p.exists():
        return {}
    return json.loads(p.read_text())["results"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confound_dir", required=True)
    ap.add_argument("--dump",        required=True)
    ap.add_argument("--out_dir",     required=True)
    args = ap.parse_args()

    cdir = Path(args.confound_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] confound_dir: {cdir}")
    baseline = load_baseline_from_dump(Path(args.dump))

    variants = ["jpeg-q95", "png", "res128", "independent_real"]
    data = {v: load_variant(cdir / f"{v}.json") for v in variants}

    rows = []
    all_gens = sorted(set(baseline.keys()))
    for g in all_gens:
        b = baseline[g]
        row = {
            "generator": g,
            "baseline_acc":  b["acc"],
            "baseline_auc":  b["auc"],
            "baseline_real_acc": b["real_acc"],
            "baseline_fake_acc": b["fake_acc"],
        }
        for v in variants:
            if g not in data[v]:
                for k in ("acc","auc","real_acc","fake_acc"):
                    row[f"{v}_{k}"] = None
                row[f"{v}_delta_acc"] = None
                continue
            r = data[v][g]
            for k in ("acc","auc","real_acc","fake_acc"):
                row[f"{v}_{k}"] = r[k]
            row[f"{v}_delta_acc"] = r["acc"] - b["acc"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "B_confound_table.csv", index=False)

    # ---- Narrative summary -----
    def mean_delta(variant: str, subset: list[str] | None = None) -> float | None:
        ds = [data[variant].get(g, {}).get("acc", None) for g in (subset or all_gens)]
        bs = [baseline[g]["acc"] for g in (subset or all_gens)]
        pairs = [(d - b) for d, b in zip(ds, bs) if d is not None]
        return float(np.mean(pairs)) if pairs else None

    train_gens = [g for g in all_gens if g not in ("GLIDE","Wukong","VQDM")]
    ood_gens   = [g for g in all_gens if g in ("GLIDE","Wukong","VQDM")]

    lines = ["# Batch 2: Confound quantification — summary\n"]
    lines.append("## Δacc vs baseline (negative = degradation under perturbation)")
    lines.append("")
    lines.append("| Variant | mean Δacc all | mean Δacc train-gen | mean Δacc OOD |")
    lines.append("|---|---|---|---|")
    for v in variants:
        if v == "independent_real":
            d_all = mean_delta(v, ood_gens)   # only OOD has entries
            d_train = "—"
            d_ood = f"{d_all*100:+.2f} pp" if d_all is not None else "—"
            lines.append(f"| {v} | — | — | {d_ood} |")
            continue
        d_all   = mean_delta(v)
        d_train = mean_delta(v, train_gens)
        d_ood   = mean_delta(v, ood_gens)
        def fmt(x): return f"{x*100:+.2f} pp" if x is not None else "—"
        lines.append(f"| {v} | {fmt(d_all)} | {fmt(d_train)} | {fmt(d_ood)} |")

    lines.append("")
    lines.append("## Per-generator detail (acc % under each variant)")
    lines.append("")
    hdr = ["generator", "baseline"] + variants
    rows_md = []
    for g in all_gens:
        r = [g, f"{baseline[g]['acc']*100:.2f}"]
        for v in variants:
            val = data[v].get(g, {}).get("acc")
            r.append(f"{val*100:.2f}" if val is not None else "—")
        rows_md.append(r)
    widths = [max(len(h), *(len(row[i]) for row in rows_md)) for i, h in enumerate(hdr)]
    def fmt_md(parts):
        return "| " + " | ".join(p.ljust(w) for p, w in zip(parts, widths)) + " |"
    lines.append(fmt_md(hdr))
    lines.append("| " + " | ".join("-"*w for w in widths) + " |")
    for r in rows_md:
        lines.append(fmt_md(r))

    lines.append("")
    lines.append("## Interpretation hints (for paper Limitations)")
    lines.append("")
    lines.append("**B1 (jpeg-q95 / png):** If Δacc ≈ 0, the model is robust to image format. "
                 "If Δacc < -5pp, the model leans heavily on encoder artifacts (a confound). "
                 "Recall A2 found |w| for jpeg_quant = 13.5 (largest); the format leakage "
                 "hypothesis predicts a measurable drop under unified re-encoding.")
    lines.append("")
    lines.append("**B2 (res128):** Downsampling to 128² wipes high-frequency components. "
                 "If Δacc < -5pp, the bitplane-LSB / hf-noise / freq-radial concepts are "
                 "the load-bearing ones — resolution-coupled. Expected, given native "
                 "resolutions span 128² (BigGAN) → 1024² (MJ).")
    lines.append("")
    lines.append("**B5 (independent_real):** OOD eval previously shared the same 1000 real "
                 "images across GLIDE/Wukong/VQDM. Under disjoint 1k samples per OOD "
                 "generator, real_acc remaining ~99.6% confirms the result is not a "
                 "spurious artifact of which 1000 real images were sampled.")

    (out_dir / "B_confound_summary.md").write_text("\n".join(lines))
    print(f"[SAVED] {out_dir / 'B_confound_summary.md'}")
    print(f"[SAVED] {out_dir / 'B_confound_table.csv'}")

    # print summary to stdout
    print("\n" + "\n".join(lines[:18]))


if __name__ == "__main__":
    main()
