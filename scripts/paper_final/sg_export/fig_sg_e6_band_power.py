#!/usr/bin/env python3
"""fig-sg-e6-band-power — band power after / before on the low-EOG samples.

Reads results/paper_final/paper_final_arrays/natural_band_power.npz (export_natural_post.py).
Compact dot matrix: one small panel per condition (raw omitted: ratio 1 by
construction), rows = bands (delta / theta / alpha), columns = the twelve
electrodes (sg_common.NINE + EXTRA_ROW); dot colour = participant-mean ratio in dB
(diverging, symmetric, one shared colour bar).  No CI; minimal text.

Usage: fig_sg_e6_band_power.py [npz] [out_dir]   (overrides = smoke test only)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/infres/yinwang/denoiseNet/scripts/paper_final/sg_export")
sys.path.insert(0, "/home/infres/yinwang/denoiseNet/scripts/paper_final/figures")
import figstyle  # noqa: E402
import sg_common as sg  # noqa: E402

figstyle.setup()

import matplotlib.pyplot as plt  # noqa: E402

SOURCE = Path(sys.argv[1]) if len(sys.argv) > 1 else sg.ARRAYS_OUT / "natural_band_power.npz"
SMOKE = len(sys.argv) > 1
if len(sys.argv) > 2:
    figstyle.OUT = Path(sys.argv[2])
NAME = "fig-sg-e6-band-power" + ("-smoke" if SMOKE else "")
SHORT = {"MATCH_gated": "subject", "POP": "population", "NO_A0": "unguided", "LINEAR": "linear",
         "WRONG_gated": "mismatched", "SHUFFLED": "shuffled", "ICA": "ICA", "ASR": "ASR",
         "SGEYESUB": "eye-subspace"}
COLOUR = {"MATCH_gated": "MATCH", "WRONG_gated": "WRONG_gated"}
CMAP = "RdBu_r"

npz = np.load(SOURCE, allow_pickle=False)
conditions = [str(c) for c in npz["condition"]]
bands = [str(b) for b in npz["bands"]]
electrodes = [str(e) for e in npz["electrodes"]]
ratio_db = np.asarray(npz["ratio_db"], float)                   # (cond, rec, band, electrode)
participants = [str(p) for p in npz["participants"]]
rec_pidx = np.asarray(npz["recording_participant_index"], int)
drawn = [c for c in conditions if c != "RAW"]

per_p = np.full((len(conditions), len(participants), len(bands), len(electrodes)), np.nan)
for p in range(len(participants)):
    with np.errstate(invalid="ignore"):
        per_p[:, p] = np.nanmean(ratio_db[:, rec_pidx == p], axis=1)
with np.errstate(invalid="ignore"):
    mean = np.nanmean(per_p, axis=1)                             # (cond, band, electrode)
vmax = float(np.nanmax(np.abs(mean[[conditions.index(c) for c in drawn]])))
vmax = float(np.ceil(min(max(vmax, 1.0), 12.0)))

n_col = 3
n_row = int(np.ceil(len(drawn) / n_col))
fig, axes = plt.subplots(n_row, n_col, figsize=(figstyle.FULL, 0.95 * n_row + 0.5), squeeze=False)
fig.subplots_adjust(left=0.06, right=0.91, top=0.92, bottom=0.12, hspace=0.55, wspace=0.12)
xs, ys = np.meshgrid(np.arange(len(electrodes)), np.arange(len(bands)))
sc = None
for k, cond in enumerate(drawn):
    ax = axes[k // n_col, k % n_col]
    c = conditions.index(cond)
    sc = ax.scatter(xs.ravel(), ys.ravel(), c=mean[c].ravel(), cmap=CMAP, vmin=-vmax, vmax=vmax,
                    s=34, edgecolor="0.3", lw=0.3)
    ax.set_title(SHORT[cond], fontsize=6.5, color=figstyle.C[COLOUR.get(cond, cond)], pad=2)
    ax.set_xlim(-0.6, len(electrodes) - 0.4)
    ax.set_ylim(-0.6, len(bands) - 0.4)
    ax.set_xticks(range(len(electrodes)))
    ax.set_yticks(range(len(bands)))
    ax.set_xticklabels(electrodes if k // n_col == n_row - 1 else [], fontsize=5, rotation=90)
    ax.set_yticklabels(bands if k % n_col == 0 else [], fontsize=5.5)
    ax.tick_params(length=0, pad=1.5)
    ax.invert_yaxis()
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
for k in range(len(drawn), n_row * n_col):
    axes[k // n_col, k % n_col].set_axis_off()
cax = fig.add_axes([0.925, 0.2, 0.012, 0.6])
cb = fig.colorbar(sc, cax=cax)
cb.set_ticks([-vmax, 0, vmax])
cb.set_label("band power after / before (dB)", fontsize=6, labelpad=2)
cb.ax.tick_params(labelsize=5.5, length=1.5, width=0.5)
cb.outline.set_linewidth(0.5)
figstyle.panel(axes[0, 0], "")
figstyle.save(fig, NAME)
print(json.dumps({"figure": NAME, "vmax_db": vmax,
                  "mean_db_by_band": {c: [round(float(np.nanmean(mean[conditions.index(c), b])), 2)
                                          for b in range(len(bands))] for c in drawn}}))
