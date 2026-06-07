"""A3 + A4: Concept intervention experiments on the existing inference dump.

Because the head is a single Linear layer (logit = w·c + b), all interventions
on the concept vector c can be done in closed form on the dump — no GPU needed.

A3 (zero-out ablation):
    For each concept k, replace c_k with 0 → new prob → per-generator Δacc.
    Outputs:
      - A3_concept_zeroout.csv (6×7 matrix of Δacc, plus baseline acc row)
      - A3_concept_zeroout.png (heatmap)
      - A3_cumulative_ablation.csv (zero concepts in order of |w|, see acc decay)
      - A3_keep_only.csv (zero ALL concepts except k, see what each concept alone can do)

A4 (concept swap counterfactual):
    For each real-image (label=0) of generator g, replace c_k with the mean of
    that generator's correctly-classified fake samples' c_k → new prob.
    Measure: what fraction of REALs flip from "pred=real" to "pred=fake"?
    This is causal evidence: does the concept *cause* the prediction?
    Outputs:
      - A4_real_to_fake_swap.csv (per generator × concept flip rate)
      - A4_fake_to_real_swap.csv (mirror direction)

Usage:
    python -m scripts.concept_intervention \
        --dump  Code/Results/full_inference_dump.npz \
        --out_dir Code/Results/batch1_analysis
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


# ---------------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------------- #

def per_gen_metrics(probs, labels, gens, all_gens):
    out = {}
    for g in all_gens:
        m = gens == g
        if not m.any():
            continue
        p = probs[m]; l = labels[m]
        preds = (p > 0.5).astype(int)
        try: auc = float(roc_auc_score(l, p))
        except ValueError: auc = float("nan")
        rm = l == 0; fm = l == 1
        out[g] = {
            "n": int(m.sum()),
            "acc": float((preds == l).mean()),
            "auc": auc,
            "real_acc": float((preds[rm] == 0).mean()) if rm.any() else float("nan"),
            "fake_acc": float((preds[fm] == 1).mean()) if fm.any() else float("nan"),
        }
    return out


def logit_to_prob(z):
    # numerically stable sigmoid
    return np.where(z >= 0,
                    1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))


# ---------------------------------------------------------------------------- #
# A3 — zero-out ablation
# ---------------------------------------------------------------------------- #

def a3_zero_out(concepts, labels, gens, w, b, names, all_gens, out_dir):
    K = concepts.shape[1]

    # Baseline (no intervention)
    baseline_logits = concepts @ w + b
    baseline_probs  = logit_to_prob(baseline_logits)
    base = per_gen_metrics(baseline_probs, labels, gens, all_gens)

    # Per-concept zero-out
    rows = []
    deltas = np.zeros((K, len(all_gens)))
    for k in range(K):
        c2 = concepts.copy()
        c2[:, k] = 0.0
        logits = c2 @ w + b
        probs  = logit_to_prob(logits)
        gm = per_gen_metrics(probs, labels, gens, all_gens)
        for gi, g in enumerate(all_gens):
            r = {
                "zeroed_concept": names[k],
                "generator": g,
                "baseline_acc": base[g]["acc"],
                "new_acc":     gm[g]["acc"],
                "delta_acc":   gm[g]["acc"] - base[g]["acc"],
                "new_real_acc": gm[g]["real_acc"],
                "new_fake_acc": gm[g]["fake_acc"],
                "new_auc":     gm[g]["auc"],
            }
            deltas[k, gi] = r["delta_acc"]
            rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "A3_concept_zeroout.csv", index=False)

    # Heatmap
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(deltas * 100, cmap="RdBu", vmin=-50, vmax=50, aspect="auto")
    ax.set_xticks(range(len(all_gens)))
    ax.set_xticklabels(all_gens, rotation=30, ha="right")
    ax.set_yticks(range(K))
    ax.set_yticklabels(names)
    ax.set_title("A3: Δacc (%) when concept_k is zero'd at inference\n"
                 "(red = drop in accuracy = concept matters)")
    for k in range(K):
        for gi in range(len(all_gens)):
            v = deltas[k, gi] * 100
            ax.text(gi, k, f"{v:+.1f}", ha="center", va="center",
                    color="white" if abs(v) > 25 else "black", fontsize=9)
    plt.colorbar(im, ax=ax, label="Δacc (%)")
    plt.tight_layout()
    plt.savefig(out_dir / "A3_concept_zeroout.png", dpi=140)
    plt.close()

    print("\n[A3] Per-concept zero-out (Δacc % per generator):")
    print(pd.DataFrame(deltas * 100, index=names, columns=all_gens).round(2).to_string())

    # Cumulative ablation: zero concepts in order of |w|
    order = np.argsort(-np.abs(w))
    cum_rows = []
    c_kept = concepts.copy()
    cum_probs = logit_to_prob(c_kept @ w + b)
    cum_acc_all = float(((cum_probs > 0.5).astype(int) == labels).mean())
    cum_rows.append({
        "step": 0, "zeroed_so_far": "none", "remaining_concepts": K,
        "overall_acc": cum_acc_all,
    })
    for step, k in enumerate(order, start=1):
        c_kept[:, k] = 0.0
        cum_probs = logit_to_prob(c_kept @ w + b)
        cum_acc_all = float(((cum_probs > 0.5).astype(int) == labels).mean())
        cum_rows.append({
            "step": step,
            "zeroed_so_far": ", ".join(names[i] for i in order[:step]),
            "remaining_concepts": K - step,
            "overall_acc": cum_acc_all,
        })
    pd.DataFrame(cum_rows).to_csv(out_dir / "A3_cumulative_ablation.csv", index=False)

    print("\n[A3] Cumulative ablation (zeroing concepts in order of |w|):")
    for r in cum_rows:
        print(f"  step {r['step']}: removed [{r['zeroed_so_far']}]  acc={r['overall_acc']*100:.2f}%")

    # Keep-only-one: zero ALL except concept_k
    keep_rows = []
    for k in range(K):
        c_only = np.zeros_like(concepts)
        c_only[:, k] = concepts[:, k]
        probs = logit_to_prob(c_only @ w + b)
        gm = per_gen_metrics(probs, labels, gens, all_gens)
        for g in all_gens:
            keep_rows.append({
                "only_concept": names[k],
                "generator": g,
                "acc": gm[g]["acc"],
                "auc": gm[g]["auc"],
            })
    pd.DataFrame(keep_rows).to_csv(out_dir / "A3_keep_only.csv", index=False)
    print("\n[A3] Keep-only-one acc per (concept, generator):")
    df_ko = pd.DataFrame(keep_rows).pivot(
        index="only_concept", columns="generator", values="acc")
    print((df_ko * 100).round(2).to_string())

    return deltas, order


# ---------------------------------------------------------------------------- #
# A4 — concept swap (counterfactual)
# ---------------------------------------------------------------------------- #

def a4_concept_swap(concepts, labels, gens, w, b, names, all_gens, out_dir):
    """For each generator g and concept k:
        - real→fake swap: take real images of g; replace c_k with mean of g's fakes' c_k
        - measure new pred; flip rate = fraction that move from real→fake prediction
    """
    K = concepts.shape[1]

    # Per-generator concept means by class
    means_real = {}
    means_fake = {}
    for g in all_gens:
        mg = gens == g
        means_real[g] = concepts[mg & (labels == 0)].mean(axis=0)
        means_fake[g] = concepts[mg & (labels == 1)].mean(axis=0)

    baseline_logits = concepts @ w + b
    baseline_probs  = logit_to_prob(baseline_logits)
    baseline_preds  = (baseline_probs > 0.5).astype(int)

    rows_r2f = []  # real → fake swap
    rows_f2r = []  # fake → real swap
    for g in all_gens:
        mg = gens == g
        real_idx = np.where(mg & (labels == 0))[0]
        fake_idx = np.where(mg & (labels == 1))[0]
        # Real → fake intervention
        for k in range(K):
            c2 = concepts.copy()
            c2[real_idx, k] = means_fake[g][k]    # swap real's c_k to fake's mean
            new_probs = logit_to_prob(c2 @ w + b)
            new_preds = (new_probs > 0.5).astype(int)
            n_real = len(real_idx)
            # Only count flips among real images that were CORRECTLY classified initially
            was_real = baseline_preds[real_idx] == 0
            now_fake = new_preds[real_idx] == 1
            flips = int((was_real & now_fake).sum())
            denom = int(was_real.sum())
            rows_r2f.append({
                "generator": g, "concept": names[k],
                "n_correct_real": denom,
                "n_flipped_to_fake": flips,
                "flip_rate": (flips / denom) if denom else float("nan"),
                "mean_prob_before": float(baseline_probs[real_idx].mean()),
                "mean_prob_after":  float(new_probs[real_idx].mean()),
            })
            # Mirror: fake → real
            c2 = concepts.copy()
            c2[fake_idx, k] = means_real[g][k]
            new_probs2 = logit_to_prob(c2 @ w + b)
            new_preds2 = (new_probs2 > 0.5).astype(int)
            was_fake = baseline_preds[fake_idx] == 1
            now_real = new_preds2[fake_idx] == 0
            flips2 = int((was_fake & now_real).sum())
            denom2 = int(was_fake.sum())
            rows_f2r.append({
                "generator": g, "concept": names[k],
                "n_correct_fake": denom2,
                "n_flipped_to_real": flips2,
                "flip_rate": (flips2 / denom2) if denom2 else float("nan"),
                "mean_prob_before": float(baseline_probs[fake_idx].mean()),
                "mean_prob_after":  float(new_probs2[fake_idx].mean()),
            })

    df_r2f = pd.DataFrame(rows_r2f)
    df_f2r = pd.DataFrame(rows_f2r)
    df_r2f.to_csv(out_dir / "A4_real_to_fake_swap.csv", index=False)
    df_f2r.to_csv(out_dir / "A4_fake_to_real_swap.csv", index=False)

    # Heatmap
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, df, title in [
        (axes[0], df_r2f, "Real → fake swap (per concept replaced with fake mean)"),
        (axes[1], df_f2r, "Fake → real swap (per concept replaced with real mean)"),
    ]:
        pivot = df.pivot(index="concept", columns="generator", values="flip_rate")
        pivot = pivot.reindex(index=names, columns=all_gens)
        im = ax.imshow(pivot.values * 100, vmin=0, vmax=100, cmap="Reds", aspect="auto")
        ax.set_xticks(range(len(all_gens)))
        ax.set_xticklabels(all_gens, rotation=30, ha="right")
        ax.set_yticks(range(K))
        ax.set_yticklabels(names)
        ax.set_title(title)
        for ki in range(K):
            for gi in range(len(all_gens)):
                v = pivot.values[ki, gi] * 100
                ax.text(gi, ki, f"{v:.0f}", ha="center", va="center",
                        color="white" if v > 50 else "black", fontsize=9)
        plt.colorbar(im, ax=ax, label="flip rate (%)")
    plt.tight_layout()
    plt.savefig(out_dir / "A4_concept_swap.png", dpi=140)
    plt.close()

    print("\n[A4] Real → fake swap, flip rate (%) per generator × concept:")
    pivot_r2f = df_r2f.pivot(index="concept", columns="generator", values="flip_rate")
    print((pivot_r2f.reindex(index=names, columns=all_gens) * 100).round(1).to_string())

    print("\n[A4] Fake → real swap, flip rate (%) per generator × concept:")
    pivot_f2r = df_f2r.pivot(index="concept", columns="generator", values="flip_rate")
    print((pivot_f2r.reindex(index=names, columns=all_gens) * 100).round(1).to_string())


# ---------------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump",     required=True)
    ap.add_argument("--out_dir",  required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    d = np.load(args.dump, allow_pickle=True)
    concepts = d["concepts"]                    # [N, K]
    labels   = d["labels"]                      # [N]
    gens     = d["generators"]                  # [N]
    w        = d["classifier_weight"].astype(np.float32)  # [K]
    b        = float(d["classifier_bias"])
    names    = d["concept_names"].tolist()
    all_gens = sorted(set(gens.tolist()))

    print(f"[INFO] N={len(labels)}  K={concepts.shape[1]}  generators={all_gens}")
    print(f"[INFO] classifier w = {w}")
    print(f"[INFO] classifier b = {b:.4f}")

    a3_zero_out(concepts, labels, gens, w, b, names, all_gens, out_dir)
    a4_concept_swap(concepts, labels, gens, w, b, names, all_gens, out_dir)

    print(f"\n[DONE] outputs in {out_dir}")


if __name__ == "__main__":
    main()
