#!/usr/bin/env python
"""Generate the two result figures for the CBNet-AGID manuscript.

All numbers are transcribed verbatim from AGID_Project/Stage2_Writing/04_Stage4_Evidence.md
(multi-seed zero-out table) and sections/body.tex (cross-benchmark generalization).
NO number is computed or invented here -- this script only lays them out.

Outputs (vector PDF, NeurIPS text width ~5.5in, serif to match the Times body):
  fig_channel_reliance.pdf   -> placed in section 4.3a
  fig_generalization.pdf     -> placed near section 4.3c / 4.4
"""
import os, csv, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# Verified result artifacts (read at runtime so figures cannot drift from disk).
RESULTS = r"E:\LQiu\lab_folder\Machine_learning\AGID_Project\Code\Results"

# Times-like serif to blend with the NeurIPS body; falls back cleanly if absent.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#444444",
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 200,
})

# Okabe-Ito colour-blind-safe palette
OI_BLUE   = "#0072B2"
OI_ORANGE = "#E69F00"
OI_GREEN  = "#009E73"
OI_GREY   = "#BBBBBB"
OI_VERM   = "#D55E00"
OI_SKY    = "#56B4E9"


# ---------------------------------------------------------------------------
# Figure 1: multi-seed channel reliance (zero-out cost per channel, 3 seeds)
# Source: 04_Stage4_Evidence.md, "Closed-form per-concept zero-out, mean dAcc"
# Values are negative (accuracy DROP); we plot magnitude (reliance) upward.
# ---------------------------------------------------------------------------
concepts = ["bitplane-\nLSB", "freq-\nradial", "color-\nmanifold",
            "hf-\nnoise", "jpeg-\nquant", "texture-\ngeometry"]
seed42 = [-0.01, -17.24, -0.03, -0.04, -49.54, -1.58]
seed1  = [-0.64, -38.66, -0.97, -1.04, -24.49, -3.23]
seed2  = [-0.14, -24.73, -0.12, -0.19, -25.73, -7.81]

mag42 = [abs(v) for v in seed42]
mag1  = [abs(v) for v in seed1]
mag2  = [abs(v) for v in seed2]

x = np.arange(len(concepts))
w = 0.26

fig, ax = plt.subplots(figsize=(5.5, 3.05))
b1 = ax.bar(x - w, mag42, w, label="seed 42", color=OI_BLUE)
b2 = ax.bar(x,     mag1,  w, label="seed 1",  color=OI_ORANGE)
b3 = ax.bar(x + w, mag2,  w, label="seed 2",  color=OI_GREEN)

# Shade the correlated compression pair {freq-radial, jpeg-quant}.
for idx in (1, 4):
    ax.axvspan(idx - 0.5, idx + 0.5, color="#FFF3D6", zorder=0)

# Value label on every bar so the inert channels read as "measured ~0", not "missing".
for cont in (b1, b2, b3):
    ax.bar_label(cont, fmt="%.1f", fontsize=5.3, padding=1.5, rotation=90)

ax.set_ylabel("Zero-out cost  |$\\Delta$acc|  (pp)")
ax.set_xticks(x)
ax.set_xticklabels(concepts)
ax.set_ylim(0, 62)
ax.legend(frameon=False, loc="upper left", ncol=3, columnspacing=1.2,
          handlelength=1.2, bbox_to_anchor=(0.0, 1.02))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(length=2)

# annotate the compression-pair bracket
ax.annotate("compression axis\n(r = $-$0.80)", xy=(2.5, 50), xytext=(2.5, 50),
            ha="center", va="center", fontsize=8, color=OI_VERM, style="italic")

fig.tight_layout(pad=0.4)
out1 = os.path.join(HERE, "fig_channel_reliance.pdf")
fig.savefig(out1, bbox_inches="tight")
plt.close(fig)
print("wrote", out1)


# ---------------------------------------------------------------------------
# Figure 2: cross-benchmark generalization (inflation vs genuine transfer)
# Sources: body.tex 4.2/4.3c/4.4 -- GenImage-OOD 99.67, ForenSynths Route B
# 73.65, single-gen 52.21, NPR(ProGAN) 79.58; chance = 50.
# ---------------------------------------------------------------------------
labels = ["GenImage OOD\n(Route B, P4)",
          "ForenSynths\n(Route B, P4)",
          "ForenSynths\nsingle-gen (P1)",
          "ForenSynths\nNPR (P3)"]
vals   = [99.67, 73.65, 52.21, 79.58]
colors = [OI_BLUE, OI_SKY, OI_GREY, OI_GREEN]

fig, ax = plt.subplots(figsize=(5.5, 2.9))
xb = np.arange(len(labels))
bars = ax.bar(xb, vals, 0.62, color=colors, edgecolor="#333333", linewidth=0.5)

ax.axhline(50, ls="--", lw=0.8, color=OI_VERM)
ax.text(len(labels) - 0.5, 50.8, "chance", color=OI_VERM, fontsize=7.5,
        ha="right", va="bottom")

for xi, v in zip(xb, vals):
    ax.text(xi, v + 0.8, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

ax.set_ylabel("Mean accuracy (%)")
ax.set_xticks(xb)
ax.set_xticklabels(labels)
ax.set_ylim(45, 107)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(length=2)

# 26-pt drop (confound inflation) between the two Route B bars
ax.annotate("", xy=(1, 73.65 + 4), xytext=(0, 99.67 + 4),
            arrowprops=dict(arrowstyle="<->", color="#555555", lw=0.8))
ax.text(0.5, 90, "$-$26 pp\n(partial confound\ninflation)", ha="center",
        va="center", fontsize=7.5, color="#555555")

# +21-pt genuine gain (multi vs single generator)
ax.annotate("", xy=(1, 73.65 - 3), xytext=(2, 52.21 - 3),
            arrowprops=dict(arrowstyle="<->", color="#555555", lw=0.8))
ax.text(1.5, 60.5, "+21 pp\n(genuine partial\ngeneralization)", ha="center",
        va="center", fontsize=7.5, color="#555555")

fig.tight_layout(pad=0.4)
out2 = os.path.join(HERE, "fig_generalization.pdf")
fig.savefig(out2, bbox_inches="tight")
plt.close(fig)
print("wrote", out2)


# ===========================================================================
#  APPENDIX/EXPANSION FIGURES (read numbers straight from Code/Results)
# ===========================================================================
GEN_ORDER = ["Stable_Diffusion_v1.4", "BigGAN", "ADM", "Midjourney",
             "GLIDE", "Wukong", "VQDM"]
GEN_LABEL = {"Stable_Diffusion_v1.4": "SD-1.4", "BigGAN": "BigGAN", "ADM": "ADM",
             "Midjourney": "MJ", "GLIDE": "GLIDE", "Wukong": "Wukong", "VQDM": "VQDM"}
GEN_COLOR = {"Stable_Diffusion_v1.4": "#0072B2", "BigGAN": "#E69F00", "ADM": "#009E73",
             "Midjourney": "#D55E00", "GLIDE": "#56B4E9", "Wukong": "#CC79A7",
             "VQDM": "#000000"}
TRAIN_GENS = {"Stable_Diffusion_v1.4", "BigGAN", "ADM", "Midjourney"}


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Figure 3: confound battery as full sweeps (JPEG quality + resolution)
# Source: batch1_analysis/B3_jpeg_quality_curve.csv, B4_resolution_curve.csv
# ---------------------------------------------------------------------------
b3 = read_csv(os.path.join(RESULTS, "batch1_analysis", "B3_jpeg_quality_curve.csv"))
b4 = read_csv(os.path.join(RESULTS, "batch1_analysis", "B4_resolution_curve.csv"))

def series(rows, xkey):
    d = {}
    for r in rows:
        d.setdefault(r["generator"], []).append((float(r[xkey]), float(r["acc"])))
    for g in d:
        d[g].sort()
    return d

s_jpeg = series(b3, "jpeg_q")
s_res  = series(b4, "resolution")

fig, (axL, axR) = plt.subplots(1, 2, figsize=(5.5, 2.7))
for g in GEN_ORDER:
    ls = "-" if g in TRAIN_GENS else "--"
    mk = "o" if g in TRAIN_GENS else "^"
    xs, ys = zip(*s_jpeg[g])
    axL.plot(xs, [100*y for y in ys], ls, marker=mk, ms=3, lw=1.1,
             color=GEN_COLOR[g], label=GEN_LABEL[g])
    xs, ys = zip(*s_res[g])
    axR.plot(xs, [100*y for y in ys], ls, marker=mk, ms=3, lw=1.1, color=GEN_COLOR[g])

for ax in (axL, axR):
    ax.axhline(50, ls=":", lw=0.8, color="#888888")
    ax.set_ylim(45, 103)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(length=2)
axL.set_xlabel("JPEG quality (re-encode)"); axL.set_ylabel("Accuracy (%)")
axL.set_title("(a) Close the container gap", fontsize=9)
axL.invert_xaxis()  # native (~95+) on the left, harsher compression to the right
axR.set_xlabel("Resolution (px)"); axR.set_title("(b) Destroy high frequencies", fontsize=9)
axR.set_xticks([64, 128, 192, 384, 512])
axL.legend(frameon=False, fontsize=6.5, ncol=2, handlelength=1.6,
           columnspacing=1.0, loc="lower center")
fig.tight_layout(pad=0.4)
out3 = os.path.join(HERE, "fig_confound.pdf")
fig.savefig(out3, bbox_inches="tight"); plt.close(fig)
print("wrote", out3)


# ---------------------------------------------------------------------------
# Figure 4: per-generator real- vs fake-accuracy, GenImage vs ForenSynths
# Source: batch1_analysis/B_confound_table.csv (baseline cols, GenImage),
#         cbnet_multigen_forensynths.json (ForenSynths)
# ---------------------------------------------------------------------------
conf = {r["generator"]: r for r in read_csv(
    os.path.join(RESULTS, "batch1_analysis", "B_confound_table.csv"))}
gi_labels = [GEN_LABEL[g] for g in GEN_ORDER]
gi_real = [100*float(conf[g]["baseline_real_acc"]) for g in GEN_ORDER]
gi_fake = [100*float(conf[g]["baseline_fake_acc"]) for g in GEN_ORDER]

with open(os.path.join(RESULTS, "cbnet_multigen_forensynths.json")) as f:
    fs = json.load(f)["results"]
fs_order = ["stargan", "deepfake", "biggan", "gaugan"]
fs_labels = ["StarGAN", "DeepFake", "BigGAN", "GauGAN"]
fs_real = [100*fs[k]["real_acc"] for k in fs_order]
fs_fake = [100*fs[k]["fake_acc"] for k in fs_order]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(5.5, 2.7),
                               gridspec_kw={"width_ratios": [7, 4]})
def paired(ax, labels, real, fake, title):
    x = np.arange(len(labels)); w = 0.38
    ax.bar(x - w/2, real, w, label="real-acc", color="#0072B2")
    ax.bar(x + w/2, fake, w, label="fake-acc", color="#D55E00")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
    ax.set_ylim(0, 108); ax.set_title(title, fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(length=2)
paired(axL, gi_labels, gi_real, gi_fake, "(a) GenImage: balanced")
axL.set_ylabel("Accuracy (%)")
axL.legend(frameon=False, fontsize=7, loc="lower left", ncol=2)
paired(axR, fs_labels, fs_real, fs_fake, "(b) ForenSynths: real-acc collapses")
fig.tight_layout(pad=0.4)
out4 = os.path.join(HERE, "fig_pergen.pdf")
fig.savefig(out4, bbox_inches="tight"); plt.close(fig)
print("wrote", out4)


# ---------------------------------------------------------------------------
# Figure 5: Route B vs E-Delta double dissociation (per-model means)
# Source: edelta/gate3_audit/gate3_full_audit.json
# ---------------------------------------------------------------------------
with open(os.path.join(RESULTS, "edelta", "gate3_audit", "gate3_full_audit.json")) as f:
    g3 = json.load(f)["detection"]
def mean_acc(block):
    return 100*np.mean([v["acc"] for v in block.values()])
rb_q96 = mean_acc(g3["routeb"]["q96_val"]);  rb_raw = mean_acc(g3["routeb"]["raw_ood"])
ed_q96 = mean_acc(g3["debiased"]["q96_val"]); ed_raw = mean_acc(g3["debiased"]["raw_ood"])

fig, ax = plt.subplots(figsize=(4.2, 2.8))
groups = ["Q96-debiased val\n(matched to E-Delta)", "Raw OOD\n(matched to Route B)"]
xg = np.arange(2); w = 0.34
rb = [rb_q96, rb_raw]; ed = [ed_q96, ed_raw]
ax.bar(xg - w/2, rb, w, label="Route B (raw-trained)", color="#0072B2")
ax.bar(xg + w/2, ed, w, label="E-Delta (Q96-debiased)", color="#009E73")
for xi, (a, b) in enumerate(zip(rb, ed)):
    ax.text(xi - w/2, a + 0.6, f"{a:.1f}", ha="center", va="bottom", fontsize=7.5)
    ax.text(xi + w/2, b + 0.6, f"{b:.1f}", ha="center", va="bottom", fontsize=7.5)
ax.set_xticks(xg); ax.set_xticklabels(groups, fontsize=7.5)
ax.set_ylabel("Mean accuracy (%)"); ax.set_ylim(80, 104)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.tick_params(length=2)
ax.legend(frameon=False, fontsize=7, loc="lower center")
fig.tight_layout(pad=0.4)
out5 = os.path.join(HERE, "fig_dissociation.pdf")
fig.savefig(out5, bbox_inches="tight"); plt.close(fig)
print("wrote", out5)
