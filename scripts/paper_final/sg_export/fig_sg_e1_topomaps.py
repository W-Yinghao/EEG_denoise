#!/usr/bin/env python3
"""fig-sg-e1-topomaps — the calibrated eye-artifact propagation patterns.

Reads results/paper_final/paper_final_arrays/e1_operators.npz (export_e1.py).
Rows    = own calibration (A_tilde) / population (A_pop) / later coupling (A_later)
Columns = vertical / horizontal EOG column of the 46x2 operator
Block a = mean over accepted development cells with full coverage
Block b = one exemplar cell (sub-02 | ses-02 | ERP)
Topomaps on the 32 scalp electrodes (standard_1020 montage); the 14 around-the-
ear channels are a dot strip beside each head on the same colour scale.  One
diverging colour scale per column within a block (symmetric about zero).
House rules: no CI / error bars / bands; minimal text; runtime-loaded data.

Usage: fig_sg_e1_topomaps.py [path/to/e1_operators.npz]   (override = smoke test only)
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
import mne  # noqa: E402

mne.set_log_level("ERROR")

EXEMPLAR = ("sub-02", "ses-02", "ERP")
ROWS = (("A_tilde", "own calibration", figstyle.C["MATCH"]),
        ("A_pop", "population", figstyle.C["POP"]),
        ("A_later", "later coupling", figstyle.C["reference"]))
COLS = ("vertical", "horizontal")
CMAP = "RdBu_r"
SOURCE = Path(sys.argv[1]) if len(sys.argv) > 1 else sg.ARRAYS_OUT / "e1_operators.npz"
NAME = "fig-sg-e1-topomaps" + ("-smoke" if len(sys.argv) > 1 else "")

# ---------------------------------------------------------------- data
npz = np.load(SOURCE, allow_pickle=False)
names = [str(n) for n in npz["eeg_names"]]
assert names == sg.EEG_NAMES
cov = np.asarray(npz["coverage"], bool)
cov_names = [str(c) for c in npz["coverage_names"]]
accepted = np.asarray(npz["accepted"], bool) & cov.all(axis=0)
mats = {k: np.asarray(npz[k], float) for k, _, _ in ROWS}
keys = [str(k) for k in npz["cell_keys"]]
i_ex = keys.index("|".join(EXEMPLAR))
assert accepted[i_ex], "exemplar cell is not an accepted, fully covered cell"

blocks = [("mean of %d accepted cells" % int(accepted.sum()),
           {k: m[accepted].mean(axis=0) for k, m in mats.items()}),
          ("%s %s %s, %s" % (EXEMPLAR + (sg.SESSION_CONDITION[EXEMPLAR[1]].replace("_", " "),)),
           {k: m[i_ex] for k, m in mats.items()})]

scalp_idx = [sg.IDX[n] for n in sg.SCALP]
info = mne.create_info(sg.SCALP, sg.RATE, "eeg")
info.set_montage(mne.channels.make_standard_montage("standard_1020"))
cmap = plt.get_cmap(CMAP)

# ---------------------------------------------------------------- layout
# per block: 2 head columns (V, H), each = head + ear strip; 3 rows + a colourbar row
fig = plt.figure(figsize=(figstyle.FULL, 4.0))
outer = fig.add_gridspec(1, 2, wspace=0.12, left=0.075, right=0.995, top=0.90, bottom=0.04)
vlims = {}
for b, (title, block) in enumerate(blocks):
    gs = outer[b].subgridspec(4, 4, height_ratios=[1, 1, 1, 0.06],
                              width_ratios=[1, 0.26, 1, 0.26], hspace=0.06, wspace=0.02)
    first_head = None
    for c, col in enumerate(COLS):
        vmax = max(float(np.abs(block[k][:, c]).max()) for k, _, _ in ROWS)
        vlims[(b, col)] = vmax
        norm = plt.Normalize(-vmax, vmax)
        for r, (k, label, colour) in enumerate(ROWS):
            values = block[k][:, c]
            ax = fig.add_subplot(gs[r, 2 * c])
            first_head = first_head or ax
            im, _ = mne.viz.plot_topomap(values[scalp_idx], info, axes=ax, show=False,
                                         cmap=CMAP, vlim=(-vmax, vmax), contours=4,
                                         sensors=True, outlines="head")
            if r == 0:
                ax.set_title(col, fontsize=7, pad=2)
            if b == 0 and c == 0:
                ax.text(-0.18, 0.5, label, transform=ax.transAxes, fontsize=6.5,
                        color=colour, rotation=90, ha="center", va="center")
            # around-the-ear channels: dot strip (montage has no positions for them)
            axE = fig.add_subplot(gs[r, 2 * c + 1])
            for x, side in ((0.0, "L"), (1.0, "R")):
                for j, n in enumerate([n for n in sg.EAR if n.startswith(side)]):
                    axE.scatter([x], [-j], s=9, color=cmap(norm(values[sg.IDX[n]])),
                                edgecolor="0.4", lw=0.25)
            if r == 0:
                for x, side in ((0.0, "L"), (1.0, "R")):
                    axE.text(x, 1.0, side, fontsize=4.5, color="0.35", ha="center", va="bottom")
            axE.set_xlim(-0.6, 1.6)
            axE.set_ylim(-7.8, 1.8)
            axE.set_axis_off()
        axCb = fig.add_subplot(gs[3, 2 * c])
        cb = fig.colorbar(im, cax=axCb, orientation="horizontal")
        cb.set_ticks([-vmax, 0.0, vmax])
        cb.set_ticklabels(["%.2g" % -vmax, "0", "%.2g" % vmax])
        cb.ax.tick_params(labelsize=5, length=1.5, width=0.5)
        cb.outline.set_linewidth(0.5)
    # block header centred over the block's two head columns
    left = gs[0, 0].get_position(fig).x0
    right = gs[0, 3].get_position(fig).x1
    fig.text((left + right) / 2, 0.955, title, fontsize=7, ha="center", va="bottom")
    figstyle.panel(first_head, "ab"[b])

figstyle.save(fig, NAME)
print(json.dumps({"figure": NAME, "accepted_cells": int(accepted.sum()),
                  "coverage": dict(zip(cov_names, cov.sum(axis=1).tolist())),
                  "exemplar": "|".join(EXEMPLAR),
                  "vlim": {f"block{b}_{c}": round(v, 4) for (b, c), v in vlims.items()}}))
