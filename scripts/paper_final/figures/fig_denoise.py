#!/usr/bin/env python3
"""fig-denoise — Paired denoising and participant effects (v3 main figure).

Panel A: waveform small-multiples for sub-02/ses-02/SSVEP/ep1, channels
Fp1/Fp2/AFz; rows = EOG drive, contaminated, reference, linear regression,
unguided, matched (+80% predictive band, the subject of the paper).
Panel B: per-participant paired rRMSE slope lines, unguided -> matched,
15 dev participants x (5 folds x 3 seeds) rows.
Panel C: scalp topomap of per-channel median improvement (NO_A0 - MATCH);
around-the-ear channels shown as a dot strip beside the head.

All numbers are read from the banked arrays/JSONs at runtime.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/infres/yinwang/denoiseNet/scripts/paper_final/figures")
import figstyle

figstyle.setup()

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import mne

REPO = Path("/home/infres/yinwang/denoiseNet")
ARR = REPO / "paper_final_arrays"
UNITS = REPO / "results/paper_final/s0_units"
SFREQ = 100.0  # configs/calib_saddpm_cond_v42r/data.yaml sampling_rate

# ---------------------------------------------------------------- data
wave = np.load(ARR / "t6_waveform_exemplar_dev.npz", allow_pickle=True)
names = [str(n) for n in wave["eeg_names"]]
fixed = [str(n) for n in wave["fixed_channels"]]          # Fp1 Fp2 AFz
ch_idx = {c: names.index(c) for c in fixed}
t = np.arange(wave["contaminated"].shape[1]) / SFREQ

rows = []
for fold in range(5):
    for seed in (20261201, 20261202, 20261203):
        rows += json.loads((UNITS / f"fold_{fold}_seed_{seed}.json").read_text())["rows"]

def per_participant(cond):
    acc = {}
    for r in rows:
        if r["condition"] == cond:
            acc.setdefault(r["participant"], []).append(float(r["rrmse_temporal"]))
    return {p: float(np.mean(v)) for p, v in sorted(acc.items())}

pp_no = per_participant("NO_A0")
pp_ma = per_participant("MATCH_gated")
parts = sorted(pp_no)
assert set(parts) == set(pp_ma) and len(parts) == 15
mean_no = float(np.mean([pp_no[p] for p in parts]))
mean_ma = float(np.mean([pp_ma[p] for p in parts]))
n_pos = sum(pp_no[p] > pp_ma[p] for p in parts)

scalp = np.load(ARR / "t6_scalp_improvement.npz", allow_pickle=True)
imp = np.asarray(scalp["improvement_noa0_minus_match"], float)
imp_names = [str(n) for n in scalp["eeg_names"]]
is_ear = [bool(re.fullmatch(r"[LR]\d+", n)) for n in imp_names]
scalp_names = [n for n, e in zip(imp_names, is_ear) if not e]
scalp_vals = np.array([v for v, e in zip(imp, is_ear) if not e])
ear_names = [n for n, e in zip(imp_names, is_ear) if e]
ear_vals = np.array([v for v, e in zip(imp, is_ear) if e])

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(figstyle.FULL, 2.55))
outer = fig.add_gridspec(1, 3, width_ratios=[3.05, 1.15, 2.05],
                         wspace=0.34, left=0.075, right=0.985,
                         top=0.90, bottom=0.10)

# ============================ Panel A ================================
gsA = outer[0].subgridspec(2, 3, height_ratios=[1.0, 4.9],
                           hspace=0.14, wspace=0.10)
axE = fig.add_subplot(gsA[0, :])
figstyle.panel(axE, "A")
eog = wave["eog_drive"]
for k, (lab, off) in enumerate((("VEOG", 55.0), ("HEOG", -55.0))):
    axE.plot(t, eog[k] + off, color=figstyle.C["eog"], lw=0.55,
             alpha=1.0 if k == 0 else 0.55)
    axE.text(-0.005, off, lab, color=figstyle.C["eog"], fontsize=5,
             ha="right", va="center",
             transform=mtransforms.blended_transform_factory(
                 axE.transAxes, axE.transData),
             alpha=1.0 if k == 0 else 0.65)
axE.set_xlim(t[0], t[-1])
axE.set_ylim(-125, 125)
axE.set_axis_off()
# 1 s scale bar for the strip
axE.plot([t[-1] - 1.0, t[-1]], [100, 100], color="0.25", lw=0.8,
         solid_capstyle="butt")
axE.text(t[-1] - 1.15, 100, "1 s", fontsize=5, color="0.25",
         ha="right", va="center")

ROWS = [("contam.", "contaminated", figstyle.C["RAW"]),
        ("reference", "reference", figstyle.C["reference"]),
        ("linear", "linear_regression", figstyle.C["LINEAR"]),
        ("unguided", "unguided", figstyle.C["NO_A0"]),
        ("matched", "matched", figstyle.C["MATCH"])]
STEP = 13.0
OFFS = [4.55 * STEP, 3.45 * STEP, 2.35 * STEP, 1.25 * STEP, 0.0]
band_lo = wave["band_mean"] - wave["band_halfwidth_80"]
band_hi = wave["band_mean"] + wave["band_halfwidth_80"]

for col, ch in enumerate(fixed):
    ax = fig.add_subplot(gsA[1, col])
    i = ch_idx[ch]
    for (lab, key, color), off in zip(ROWS, OFFS):
        y = wave[key][i]
        if key == "matched":  # 80% predictive band (subject of the paper)
            ax.fill_between(t, band_lo[i] + off, band_hi[i] + off,
                            color=figstyle.C["MATCH"], alpha=0.22, lw=0)
        if key in ("linear_regression", "unguided", "matched"):
            ax.plot(t, wave["reference"][i] + off, color="0.55", lw=0.35,
                    alpha=0.8, zorder=2)
        ax.plot(t, y + off, color=color, lw=0.55, zorder=3)
        if col == 0:
            ax.text(-0.03, off, lab, color=color, fontsize=5.5, ha="right",
                    va="center",
                    transform=mtransforms.blended_transform_factory(
                        ax.transAxes, ax.transData))
    ax.text(0.5, 1.005, ch, fontsize=6, color="0.15", ha="center",
            va="bottom", transform=ax.transAxes)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-19, OFFS[0] + 13)
    ax.set_axis_off()
    if col == 2:  # time scale bar
        ax.plot([t[-1] - 1.0, t[-1]], [-17, -17], color="0.25", lw=0.8,
                solid_capstyle="butt")
        ax.text(t[-1] - 0.5, -18.5, "1 s", fontsize=5, color="0.25",
                ha="center", va="top")
    if col == 0:  # amplitude scale bar
        ax.plot([0.06, 0.06], [-17, -12], color="0.25", lw=0.8,
                solid_capstyle="butt")
        ax.text(0.16, -14.5, "5 a.u.", fontsize=5, color="0.25",
                ha="left", va="center")

# ============================ Panel B ================================
axB = fig.add_subplot(outer[1])
figstyle.panel(axB, "B")
for p in parts:
    axB.plot([0, 1], [pp_no[p], pp_ma[p]], color="0.75", lw=0.6, zorder=1)
axB.scatter([0] * len(parts), [pp_no[p] for p in parts], s=7,
            color=figstyle.C["NO_A0"], zorder=3, lw=0)
axB.scatter([1] * len(parts), [pp_ma[p] for p in parts], s=7,
            color=figstyle.C["MATCH"], zorder=3, lw=0)
# pooled means as short thick ticks with source numbers
axB.plot([-0.09, 0.09], [mean_no] * 2, color=figstyle.C["NO_A0"], lw=1.6,
         solid_capstyle="butt", zorder=4)
axB.plot([0.91, 1.09], [mean_ma] * 2, color=figstyle.C["MATCH"], lw=1.6,
         solid_capstyle="butt", zorder=4)
axB.text(-0.15, mean_no, f"{mean_no:.3f}", color=figstyle.C["NO_A0"],
         fontsize=5.5, ha="right", va="center")
axB.text(1.15, mean_ma, f"{mean_ma:.3f}", color=figstyle.C["MATCH"],
         fontsize=5.5, ha="left", va="center")
axB.text(0.5, 1.75, f"{n_pos}/{len(parts)}" + r"$\downarrow$",
         fontsize=5.5, color="0.25", ha="center", va="center")
axB.set_yscale("log")
axB.set_yticks([0.1, 0.2, 0.5, 1.0, 2.0])
axB.set_yticklabels(["0.1", "0.2", "0.5", "1", "2"])
axB.tick_params(axis="y", which="minor", length=0)
axB.set_xlim(-0.55, 1.55)
axB.set_xticks([0, 1])
axB.set_xticklabels(["unguided", "matched"])
for tick, color in zip(axB.get_xticklabels(),
                       (figstyle.C["NO_A0"], figstyle.C["MATCH"])):
    tick.set_color(color)
axB.set_ylabel("rRMSE")
axB.spines["bottom"].set_visible(False)
axB.tick_params(axis="x", length=0)

# ============================ Panel C ================================
gsC = outer[2].subgridspec(2, 2, height_ratios=[1.0, 0.055],
                           width_ratios=[1.0, 0.30], hspace=0.02,
                           wspace=0.02)
axC = fig.add_subplot(gsC[0, 0])
figstyle.panel(axC, "C")
vmax = float(imp.max())
info = mne.create_info(scalp_names, SFREQ, "eeg")
info.set_montage(mne.channels.make_standard_montage("standard_1020"))
mask = np.array([n in fixed for n in scalp_names])
im, _ = mne.viz.plot_topomap(
    scalp_vals, info, axes=axC, show=False, cmap="Oranges",
    vlim=(0.0, vmax), contours=3, sensors=True, outlines="head",
    mask=mask,
    mask_params=dict(marker="o", markerfacecolor="none",
                     markeredgecolor=figstyle.C["MATCH"], markersize=3.2,
                     markeredgewidth=0.7))
i_fp1 = scalp_names.index("Fp1")
axC.text(0.02, 0.97, f"Fp1 +{scalp_vals[i_fp1]:.2f}", fontsize=5.5,
         color=figstyle.C["MATCH"], ha="left", va="top",
         transform=axC.transAxes)

# around-the-ear channels: dot strip (montage has no positions for them)
axEar = fig.add_subplot(gsC[0, 1])
cmap = plt.get_cmap("Oranges")
norm = plt.Normalize(0.0, vmax)
for x, side in ((0.0, "L"), (1.0, "R")):
    col_names = [n for n in ear_names if n.startswith(side)]
    for j, n in enumerate(col_names):
        v = ear_vals[ear_names.index(n)]
        axEar.scatter([x], [-j], s=14, color=cmap(norm(v)),
                      edgecolor="0.4", lw=0.3)
        axEar.text(x + 0.28, -j, n, fontsize=4.5, color="0.35",
                   ha="left", va="center")
axEar.set_xlim(-0.4, 2.1)
axEar.set_ylim(-7.8, 0.8)
axEar.set_axis_off()

axCb = fig.add_subplot(gsC[1, 0])
cb = fig.colorbar(im, cax=axCb, orientation="horizontal")
cb.set_ticks([0.0, 0.1, 0.2, round(vmax, 2)])
cb.ax.tick_params(labelsize=5, length=1.5, width=0.5)
cb.outline.set_linewidth(0.5)
cb.set_label(r"$\Delta$ rRMSE", fontsize=6, labelpad=1)

figstyle.save(fig, "fig-denoise")
