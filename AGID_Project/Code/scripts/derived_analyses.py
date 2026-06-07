"""Compute all Batch-1 derived analyses from full_inference_dump.npz.

Outputs (under <out_dir>):
    A2_classifier_weights.json    — concept importance from linear head
    A1_concept_stats.csv          — K×7 mean/std/median
    A1_concept_stats.md           — same as Markdown table
    A5_concept_correlation.csv    — Pearson + Spearman matrices
    A5_concept_correlation.png    — heatmap
    A6_concept_tsne.png           — 2D t-SNE colored by generator
    A6_concept_pca.png            — 2D PCA colored by generator
    C1_misclassified_samples.csv  — FP/FN list with concept values
    C2_misclassified_concept_means.csv  — mean concept value of FP/FN vs correct
    C4_confidence_histograms.png  — 7-panel P(fake) histogram
    C5_calibration.json           — ECE / Brier / NLL per gen + overall
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


def safe_seterr():
    np.seterr(all="warn")


def df_to_md(df, index=True):
    """Lightweight Markdown table — avoids tabulate dependency."""
    cols = list(df.columns)
    if index:
        header = ["", *map(str, cols)]
        rows   = [[str(idx), *(str(v) for v in row)] for idx, row in
                  zip(df.index, df.itertuples(index=False))]
    else:
        header = list(map(str, cols))
        rows   = [[str(v) for v in row] for row in df.itertuples(index=False)]
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
              for i, h in enumerate(header)]
    def fmt(parts):
        return "| " + " | ".join(p.ljust(w) for p, w in zip(parts, widths)) + " |"
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    return "\n".join([fmt(header), sep, *(fmt(r) for r in rows)])


# ---------------------------------------------------------------------------- #
# A2 — Linear classifier weight inspection
# ---------------------------------------------------------------------------- #

def a2_classifier_weights(dump, out_dir):
    w = dump["classifier_weight"].astype(float)
    b = float(dump["classifier_bias"])
    names = dump["concept_names"].tolist()
    order = np.argsort(-np.abs(w))

    rows = [{
        "rank": i + 1,
        "concept": names[idx],
        "weight": float(w[idx]),
        "abs_weight": float(abs(w[idx])),
        "sign": "+" if w[idx] > 0 else "-",
    } for i, idx in enumerate(order)]

    payload = {
        "bias": b,
        "weights": rows,
        "interpretation": (
            "concepts ranked by |w|; sign + → activation raises P(fake), "
            "sign - → raises P(real)."
        ),
    }
    (out_dir / "A2_classifier_weights.json").write_text(
        json.dumps(payload, indent=2))

    print("\n[A2] Linear classifier weights (ranked by |w|):")
    for r in rows:
        print(f"  {r['rank']}. {r['concept']:<18s}  w = {r['weight']:+.4f}")


# ---------------------------------------------------------------------------- #
# A1 — Concept activation statistics
# ---------------------------------------------------------------------------- #

def a1_concept_stats(dump, out_dir):
    concepts   = dump["concepts"]                     # [N, K]
    labels     = dump["labels"]
    generators = dump["generators"]
    names      = dump["concept_names"].tolist()
    K = concepts.shape[1]

    gens = sorted(set(generators.tolist()))

    rows = []
    # Per generator × per concept × (all / ai / real)
    for g in gens:
        mask = generators == g
        for split, sm in (("all",  np.ones_like(mask)),
                          ("ai",   labels == 1),
                          ("real", labels == 0)):
            sub_mask = mask & sm
            if not sub_mask.any():
                continue
            c = concepts[sub_mask]
            for k in range(K):
                rows.append({
                    "generator": g, "split": split, "concept": names[k],
                    "n": int(sub_mask.sum()),
                    "mean":   float(c[:, k].mean()),
                    "std":    float(c[:, k].std()),
                    "median": float(np.median(c[:, k])),
                    "q25":    float(np.percentile(c[:, k], 25)),
                    "q75":    float(np.percentile(c[:, k], 75)),
                })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "A1_concept_stats.csv", index=False)

    # Compact markdown pivot: mean for each (generator, concept) for split=all
    pivot = df[df["split"] == "all"].pivot(
        index="generator", columns="concept", values="mean")
    pivot = pivot.reindex(columns=names)  # fix column order
    md = df_to_md(pivot.round(4), index=True)
    (out_dir / "A1_concept_stats.md").write_text(
        "# A1: Concept activation mean (split=all) per generator × concept\n\n"
        + md + "\n\n"
        + "Full per-split stats (mean/std/median/q25/q75) in A1_concept_stats.csv\n"
    )
    print(f"\n[A1] Concept activation mean per generator (split=all):")
    print(pivot.round(3).to_string())


# ---------------------------------------------------------------------------- #
# A5 — Concept correlation matrix
# ---------------------------------------------------------------------------- #

def a5_concept_correlation(dump, out_dir):
    from scipy.stats import spearmanr
    concepts = dump["concepts"]
    names = dump["concept_names"].tolist()
    K = concepts.shape[1]

    pearson  = np.corrcoef(concepts.T)                     # [K, K]
    spear, _ = spearmanr(concepts, axis=0)                 # [K, K]
    if np.ndim(spear) == 0:  # only 1 concept edge case
        spear = np.array([[spear]])

    pdf = pd.DataFrame(pearson,  index=names, columns=names)
    sdf = pd.DataFrame(spear,    index=names, columns=names)
    pdf.to_csv(out_dir / "A5_pearson.csv")
    sdf.to_csv(out_dir / "A5_spearman.csv")

    # Heatmap (Pearson)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for a, m, title in zip(ax, [pearson, spear], ["Pearson", "Spearman"]):
        im = a.imshow(m, vmin=-1, vmax=1, cmap="RdBu_r")
        a.set_xticks(range(K)); a.set_yticks(range(K))
        a.set_xticklabels(names, rotation=45, ha="right")
        a.set_yticklabels(names)
        a.set_title(f"{title} correlation")
        for i in range(K):
            for j in range(K):
                a.text(j, i, f"{m[i,j]:.2f}", ha="center", va="center",
                       color="white" if abs(m[i,j]) > 0.5 else "black", fontsize=8)
        plt.colorbar(im, ax=a, fraction=0.046)
    plt.tight_layout()
    plt.savefig(out_dir / "A5_concept_correlation.png", dpi=140)
    plt.close()

    print(f"\n[A5] Concept correlation matrix (Pearson):")
    print(pdf.round(2).to_string())


# ---------------------------------------------------------------------------- #
# A6 — t-SNE / PCA on concept space
# ---------------------------------------------------------------------------- #

def a6_tsne_pca(dump, out_dir, seed=0):
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    concepts   = dump["concepts"]
    labels     = dump["labels"]
    generators = dump["generators"]
    gens = sorted(set(generators.tolist()))
    cmap = plt.get_cmap("tab10")

    # PCA
    pca = PCA(n_components=2, random_state=seed).fit_transform(concepts)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for i, g in enumerate(gens):
        m = generators == g
        axes[0].scatter(pca[m & (labels==1), 0], pca[m & (labels==1), 1],
                        s=4, alpha=0.5, color=cmap(i), label=f"{g}/ai", marker="x")
        axes[0].scatter(pca[m & (labels==0), 0], pca[m & (labels==0), 1],
                        s=4, alpha=0.3, color=cmap(i), label=f"{g}/real", marker="o")
    axes[0].set_title("PCA on 6D concept space"); axes[0].legend(fontsize=6, ncol=2)
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")

    # t-SNE (subsample if too large)
    rng = np.random.default_rng(seed)
    if concepts.shape[0] > 5000:
        sel = rng.choice(concepts.shape[0], 5000, replace=False)
    else:
        sel = np.arange(concepts.shape[0])
    tsne = TSNE(n_components=2, perplexity=30, random_state=seed,
                init="pca", learning_rate="auto"
                ).fit_transform(concepts[sel])
    sub_gen = generators[sel]; sub_lab = labels[sel]
    for i, g in enumerate(gens):
        m = sub_gen == g
        axes[1].scatter(tsne[m & (sub_lab==1), 0], tsne[m & (sub_lab==1), 1],
                        s=4, alpha=0.5, color=cmap(i), label=f"{g}/ai", marker="x")
        axes[1].scatter(tsne[m & (sub_lab==0), 0], tsne[m & (sub_lab==0), 1],
                        s=4, alpha=0.3, color=cmap(i), label=f"{g}/real", marker="o")
    axes[1].set_title("t-SNE on 6D concept space (subsample 5000)")
    axes[1].legend(fontsize=6, ncol=2)
    plt.tight_layout()
    plt.savefig(out_dir / "A6_concept_pca_tsne.png", dpi=140)
    plt.close()
    print(f"\n[A6] PCA + t-SNE saved → A6_concept_pca_tsne.png")


# ---------------------------------------------------------------------------- #
# C1 + C2 — Misclassified samples
# ---------------------------------------------------------------------------- #

def c1_c2_misclassified(dump, out_dir):
    probs      = dump["probs"]
    labels     = dump["labels"]
    concepts   = dump["concepts"]
    paths      = dump["paths"]
    generators = dump["generators"]
    names      = dump["concept_names"].tolist()
    preds      = (probs > 0.5).astype(int)
    miscls     = preds != labels

    # C1: per-sample misclassified list
    rows = []
    for i in np.where(miscls)[0]:
        d = {
            "generator": str(generators[i]),
            "label": int(labels[i]),
            "pred":  int(preds[i]),
            "prob_fake": float(probs[i]),
            "path": str(paths[i]),
        }
        for k, n in enumerate(names):
            d[f"c_{n}"] = float(concepts[i, k])
        rows.append(d)
    pd.DataFrame(rows).to_csv(out_dir / "C1_misclassified_samples.csv", index=False)

    # C2: per-generator, per-concept mean for (correct_real, correct_fake, FP, FN)
    summary = []
    for g in sorted(set(generators.tolist())):
        m = generators == g
        masks = {
            "correct_real": m & (labels == 0) & ~miscls,
            "FP_predfake":  m & (labels == 0) &  miscls,
            "correct_fake": m & (labels == 1) & ~miscls,
            "FN_predreal":  m & (labels == 1) &  miscls,
        }
        for split, sm in masks.items():
            if not sm.any():
                continue
            row = {"generator": g, "split": split, "n": int(sm.sum())}
            for k, n in enumerate(names):
                row[f"mean_{n}"] = float(concepts[sm, k].mean())
            summary.append(row)
    pd.DataFrame(summary).to_csv(out_dir / "C2_misclassified_concept_means.csv",
                                 index=False)

    print(f"\n[C1] {len(rows)} misclassified samples → C1_misclassified_samples.csv")
    print(f"[C2] concept-mean comparison → C2_misclassified_concept_means.csv")


# ---------------------------------------------------------------------------- #
# C4 — Softmax confidence histograms
# ---------------------------------------------------------------------------- #

def c4_confidence_histograms(dump, out_dir):
    probs      = dump["probs"]
    labels     = dump["labels"]
    generators = dump["generators"]
    gens = sorted(set(generators.tolist()))

    n = len(gens)
    rows = (n + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows))
    axes = axes.flatten()

    for i, g in enumerate(gens):
        m = generators == g
        ax = axes[i]
        ax.hist(probs[m & (labels == 0)], bins=50, range=(0, 1),
                alpha=0.6, label="real", color="C0")
        ax.hist(probs[m & (labels == 1)], bins=50, range=(0, 1),
                alpha=0.6, label="ai", color="C3")
        ax.axvline(0.5, color="k", linestyle="--", lw=0.8)
        ax.set_yscale("log")
        ax.set_title(g)
        ax.set_xlabel("P(fake)")
        ax.legend(fontsize=8)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.savefig(out_dir / "C4_confidence_histograms.png", dpi=140)
    plt.close()
    print(f"\n[C4] confidence histograms → C4_confidence_histograms.png")


# ---------------------------------------------------------------------------- #
# C5 — Calibration metrics
# ---------------------------------------------------------------------------- #

def _ece(probs, labels, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (probs >= lo) & (probs < hi)
        if not m.any():
            continue
        acc  = (labels[m] == (probs[m] > 0.5).astype(int)).mean()
        conf = np.where(probs[m] > 0.5, probs[m], 1 - probs[m]).mean()
        ece += (m.sum() / len(probs)) * abs(acc - conf)
    return float(ece)


def c5_calibration(dump, out_dir):
    probs      = dump["probs"].astype(np.float64)
    labels     = dump["labels"].astype(np.float64)
    generators = dump["generators"]

    eps = 1e-12
    res = {}
    for g in sorted(set(generators.tolist())) + ["__ALL__"]:
        m = (generators == g) if g != "__ALL__" else np.ones_like(generators, dtype=bool)
        p = probs[m]
        l = labels[m]
        if not m.any():
            continue
        # Brier
        brier = float(((p - l) ** 2).mean())
        # NLL (binary cross-entropy)
        nll = float(-(l * np.log(p + eps) + (1 - l) * np.log(1 - p + eps)).mean())
        # ECE
        ece = _ece(p, l.astype(int))
        res[g] = {"n": int(m.sum()), "brier": brier, "nll": nll, "ece": ece}

    (out_dir / "C5_calibration.json").write_text(json.dumps(res, indent=2))
    print(f"\n[C5] Calibration metrics:")
    for g, r in res.items():
        print(f"  {g:<25s}  n={r['n']:>5d}  Brier={r['brier']:.5f}  "
              f"NLL={r['nll']:.4f}  ECE={r['ece']:.4f}")


# ---------------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump",    required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    safe_seterr()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump = np.load(args.dump, allow_pickle=True)

    print(f"[INFO] loaded dump: {args.dump}")
    print(f"  N={len(dump['labels'])}  K={dump['concepts'].shape[1]}  "
          f"generators={sorted(set(dump['generators'].tolist()))}")

    a2_classifier_weights(dump, out_dir)
    a1_concept_stats(dump, out_dir)
    a5_concept_correlation(dump, out_dir)
    a6_tsne_pca(dump, out_dir)
    c1_c2_misclassified(dump, out_dir)
    c4_confidence_histograms(dump, out_dir)
    c5_calibration(dump, out_dir)

    print(f"\n[DONE] all outputs in {out_dir}")


if __name__ == "__main__":
    main()
