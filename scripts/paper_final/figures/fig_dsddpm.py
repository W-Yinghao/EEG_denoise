#!/usr/bin/env python3
"""fig-dsddpm — DS-DDPM reproduction findings (appendix, Sec 5.7).

Panel A: per-training-subject column means M (9 small dots) + grand mean
(large marker) for all 12 protocol variants (3 arms x 4 recipes) from
table1_*.json; reference lines = chance, published ICA / DS-DDPM Table I
M-means, and the on-disk EEGNet-8,2 within-participant RAW mean from
results/paper_final/bci2a/route1.json (0.593; the prose "~66%" figure has
no machine-readable source and is NOT printed).
Panel A inset (right of dots): representative 9x9 accuracy matrices
(raw / ica / dsddpm_paper), shared color scale, diagonal outlined.
Panel B: real_real vs gen_real 9x9 PSD-signature correlation heatmaps
(table2_correlations.json) on one shared 0-1 scale; subject 6 flagged.
Panel C: training-loss components (per-epoch means over 81 steps/epoch,
faint raw 8100-step traces) from train_metrics.csv, log y.

House rules: NO confidence intervals / error bars / whiskers / bands;
per-participant dots instead. Every number is read from files at runtime.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/infres/yinwang/denoiseNet/scripts/paper_final/figures")
import figstyle

figstyle.setup()

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path("/home/infres/yinwang/denoiseNet")
DSD = ROOT / "results/paper_final/dsddpm"
C = figstyle.C

ARMS = ["raw", "ica", "dsddpm"]
ARM_COL = {"raw": C["RAW"], "ica": C["ICA"], "dsddpm": C["DSDDPM"]}
ARM_DISP = {"raw": "raw", "ica": "ICA", "dsddpm": "DS-DDPM"}
RECIPES = ["paper", "adam", "adam_l2", "adam_zscore"]
REC_DISP = {"paper": "paper", "adam": "Adam", "adam_l2": "Adam+L2",
            "adam_zscore": "Adam+z"}


def t1(arm, recipe):
    suff = "" if recipe == "paper" else f"_{recipe}"
    if arm == "dsddpm":                       # dsddpm has explicit _paper file
        suff = "_paper" if recipe == "paper" else f"_{recipe}"
    return json.loads((DSD / f"table1_{arm}{suff}.json").read_text())


tables = {(a, r): t1(a, r) for a in ARMS for r in RECIPES}
pub = tables[("raw", "paper")]["published_reference"]  # 50.58 / 52.87 (%)
route1 = json.loads(
    (ROOT / "results/paper_final/bci2a/route1.json").read_text())
eegnet_within = route1["accuracy_means"]["RAW"]        # 0.5926, official T->E
chance = route1["chance"]                              # 0.25
corr = json.loads((DSD / "table2_correlations.json").read_text())

# train_metrics.csv is headerless; column order is fixed by the writerow in
# scripts/paper_final/dsddpm_repro.py:
#   [total, total-orth-sub_arc (base), time_diff, noise_kl, sub_arc, orth]
M = np.loadtxt(DSD / "train_metrics.csv", delimiter=",")   # (8100, 6)
STEPS_PER_EPOCH = 81                                       # 8100 = 100 x 81
n_ep = M.shape[0] // STEPS_PER_EPOCH
Mep = M[: n_ep * STEPS_PER_EPOCH].reshape(n_ep, STEPS_PER_EPOCH, 6).mean(1)
COMP = [("total", 0), ("time-diff", 2), ("noise-KL", 3),
        ("sub-arc", 4), ("orth", 5)]
LC = {"total": "#2b2b2b", "time-diff": "#004488", "noise-KL": "#DDAA33",
      "sub-arc": "#BB5566", "orth": "#8855BB"}

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(figstyle.FULL, 3.35))
gs = fig.add_gridspec(1, 4, width_ratios=[1.75, 0.62, 1.02, 1.55],
                      wspace=0.42, left=0.085, right=0.965,
                      top=0.90, bottom=0.13)

# -- Panel A: 12-variant dot summary -------------------------------------
axA = fig.add_subplot(gs[0])
rng = np.random.default_rng(3)
yrows, ylabels = [], []
y = 0
for a in ARMS[::-1]:                       # raw bottom group ... dsddpm top
    for r in RECIPES[::-1]:
        d = tables[(a, r)]
        col_means = np.asarray(d["column_means_M"])   # 9 per-subject means
        gm = d["grand_mean"]
        jit = rng.uniform(-0.18, 0.18, size=col_means.size)
        axA.scatter(col_means, y + jit, s=5, color=ARM_COL[a], alpha=0.45,
                    lw=0, zorder=2)
        axA.scatter([gm], [y], s=26, color=ARM_COL[a], edgecolor="white",
                    lw=0.5, zorder=3)
        yrows.append(y)
        ylabels.append(REC_DISP[r])
        y += 1
    y += 0.7                               # gap between arm groups
axA.set_yticks(yrows)
axA.set_yticklabels(ylabels, fontsize=5)
# arm group labels in matching color, left of the tick labels
for i, a in enumerate(ARMS[::-1]):
    yc = np.mean(yrows[4 * i: 4 * i + 4])
    axA.text(-0.185, yc, ARM_DISP[a], transform=axA.get_yaxis_transform(),
             color=ARM_COL[a], fontsize=6.5, fontweight="bold",
             ha="center", va="center", rotation=90)
# reference lines (all values read from files above)
# labels hang downward from the top, body just left of their line; the
# DS-DDPM body goes just RIGHT of its line instead (the two published
# lines are only 0.023 apart) into the clear gap before the EEGNet line
for x, col, lab, xanch in (
        (chance, "#9a9a9a", f"chance {chance:.2f}", -0.006),
        (pub["ICA_M_mean"] / 100, C["ICA"],
         f"pub. ICA {pub['ICA_M_mean']:.1f}%", -0.006),
        (pub["DSDDPM_M_mean"] / 100, C["DSDDPM"],
         f"pub. DS-DDPM {pub['DSDDPM_M_mean']:.1f}%", 0.023),
        (eegnet_within, "#2b2b2b",
         f"EEGNet-8,2 within {100 * eegnet_within:.1f}%", -0.006)):
    axA.axvline(x, color=col, lw=0.7, ls=(0, (4, 2)), zorder=1)
    axA.text(x + xanch, y - 0.45, lab, color=col, fontsize=5,
             rotation=90, ha="right", va="top")
axA.set_xlim(0.19, 0.66)
axA.set_ylim(-0.8, y - 0.3)
axA.set_xlabel("cross-subject accuracy (M mean)")
figstyle.panel(axA, "A")

# -- Panel A inset: representative 9x9 accuracy matrices ------------------
gsT = gs[1].subgridspec(3, 1, hspace=0.28)
rep = {a: np.asarray(tables[(a, "paper")]["matrix_rows_test_cols_train"])
       for a in ARMS}
vmin = min(m.min() for m in rep.values())
vmax = max(m.max() for m in rep.values())
for i, a in enumerate(ARMS):
    ax = fig.add_subplot(gsT[i])
    ax.imshow(rep[a], cmap="Blues", vmin=vmin, vmax=vmax,
              interpolation="nearest")
    for k in range(9):                     # outline within-subject diagonal
        ax.add_patch(Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False,
                               edgecolor="#2b2b2b", lw=0.35))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(1.06, 0.5, ARM_DISP[a], transform=ax.transAxes,
            color=ARM_COL[a], fontsize=5.5, fontweight="bold",
            va="center", rotation=90)
    if i == 0:
        ax.text(0.5, 1.14, f"acc {vmin:.2f}–{vmax:.2f}",
                transform=ax.transAxes, fontsize=5, color="#555555",
                ha="center")
    if i == 2:
        ax.set_xlabel("train subj", fontsize=5, labelpad=1)
        ax.set_ylabel("test subj", fontsize=5, labelpad=1)

# -- Panel B: PSD-signature correlation heatmaps --------------------------
gsB = gs[2].subgridspec(2, 2, width_ratios=[1, 0.07],
                        hspace=0.30, wspace=0.08)
rr = np.asarray(corr["real_real"])
gr = np.asarray(corr["gen_real"])
n = rr.shape[0]
offmask = ~np.eye(n, dtype=bool)
im = None
for i, (mat, name) in enumerate(((rr, "real × real"),
                                 (gr, "gen × real"))):
    ax = fig.add_subplot(gsB[i, 0])
    im = ax.imshow(mat, cmap="Blues", vmin=0.0, vmax=1.0,
                   interpolation="nearest")
    ax.set_xticks([5])                     # flag subject 6 (0-indexed 5)
    ax.set_xticklabels(["S6"], fontsize=5)
    ax.set_yticks([5])
    ax.set_yticklabels(["S6"], fontsize=5)
    ax.tick_params(length=1.5, pad=1)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_ylabel(name, fontsize=5.5, color="#2b2b2b", labelpad=2)
    dg, od = mat[np.eye(n, dtype=bool)].mean(), mat[offmask].mean()
    ax.text(0.5, -0.10, f"diag {dg:.2f} | off {od:.2f}",
            transform=ax.transAxes, fontsize=5, color="#555555",
            ha="center", va="top")
    if i == 0:
        figstyle.panel(ax, "B")
cbax = fig.add_subplot(gsB[1, 1])
bb = cbax.get_position()
cbax.set_position([bb.x0, bb.y0, bb.width * 0.35, bb.height * 0.85])
cb = fig.colorbar(im, cax=cbax)
cb.set_ticks([0, 0.5, 1])
cb.ax.tick_params(labelsize=5, length=1.5, pad=1)
cb.outline.set_visible(False)

# -- Panel C: training-loss components ------------------------------------
axC = fig.add_subplot(gs[3])
step_ep = (np.arange(M.shape[0]) + 0.5) / STEPS_PER_EPOCH
for name, j in COMP:
    axC.plot(step_ep, M[:, j], color=LC[name], lw=0.3, alpha=0.15, zorder=1)
    axC.plot(np.arange(n_ep) + 0.5, Mep[:, j], color=LC[name], lw=1.0,
             zorder=2)
    end = Mep[-1, j]
    lab = f"{name} {end:.2f}" if end >= 0.01 else f"{name} {end:.0e}"
    axC.text(n_ep + 1.5, end, lab, color=LC[name], fontsize=5.5,
             va="center")
axC.set_yscale("log")
axC.set_xlim(0, n_ep + 33)
axC.set_xticks([0, 25, 50, 75, 100])
axC.set_xlabel("epoch")
axC.set_ylabel("loss (log)")
figstyle.panel(axC, "C")

figstyle.save(fig, "fig-dsddpm")
