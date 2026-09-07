#!/usr/bin/env python3
"""fig-sg-e5-exemplars — one natural 5.12-s window per movement condition.

Reads results/paper_final/paper_final_arrays/natural_exemplars.npz (export_e5.py).
One column per condition (order and labels from the file, which carries
sg_common.SESSION_CONDITION + CONDITION_NOTE), the raw bipolar EOG (V above H)
on top, then the 12-channel stack sg_common.EXEMPLAR_STACK with the raw EEG in
black and the subject-calibrated output (figstyle.C['MATCH']) overlaid, all in
microvolt (tile units x eeg_scale).  Scale bars in the last column, axes off,
minimal text.  House rules: no CI / error bars / bands; runtime-loaded data.
SGEYESUB Fig. 1d / EEGDfus Fig. 7 layout.

Usage: fig_sg_e5_exemplars.py [path/to/natural_exemplars.npz]   (override = smoke only)
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

SOURCE = Path(sys.argv[1]) if len(sys.argv) > 1 else sg.ARRAYS_OUT / "natural_exemplars.npz"
NAME = "fig-sg-e5-exemplars" + ("-smoke" if len(sys.argv) > 1 else "")
RAW_COLOUR, MATCH_COLOUR, EOG_COLOUR = "black", figstyle.C["MATCH"], figstyle.C["eog"]
NICE = np.array([5, 10, 20, 50, 100, 200, 500, 1000], float)


def nice_below(value: float) -> float:
    ok = NICE[NICE <= value]
    return float(ok[-1]) if len(ok) else float(NICE[0])


# ---------------------------------------------------------------- data
npz = np.load(SOURCE, allow_pickle=False)
names = [str(n) for n in npz["eeg_names"]]
assert names == sg.EEG_NAMES
stack = [str(n) for n in npz["stack_channels"]] if "stack_channels" in npz.files else sg.EXEMPLAR_STACK
assert stack == sg.EXEMPLAR_STACK
conditions = [str(c) for c in npz["condition"]]
sessions = [str(s) for s in npz["session"]]
starts = [int(s) for s in npz["start"]]
cov = np.asarray(npz["coverage"], bool)
cov_arms = [str(a) for a in npz["coverage_arms"]]
assert cov[:, cov_arms.index("RAW")].all() and cov[:, cov_arms.index("MATCH")].all()
scale = np.asarray(npz["eeg_scale"], float)[:, None]
rows = [sg.IDX[n] for n in stack]
raw = np.asarray(npz["contaminated"], float) * scale            # (cond, 46, T) microvolt
match = np.asarray(npz["matched"], float) * scale
eog = np.asarray(npz["eog_raw_bipolar"], float)                 # (cond, 2, T) microvolt
n_cond, _, T = raw.shape
t = np.arange(T) / sg.RATE

# one vertical spacing for every column: the 99.5th percentile of |raw| on the stack
spacing = 2.0 * float(np.percentile(np.abs(raw[:, rows]), 99.5))
eog_spacing = 1.1 * float(np.abs(eog).max())
eeg_bar = nice_below(0.8 * spacing)
eog_bar = nice_below(0.8 * eog_spacing)

# ---------------------------------------------------------------- layout
fig = plt.figure(figsize=(figstyle.FULL, 5.2))
gs = fig.add_gridspec(2, n_cond, height_ratios=[1.0, 5.0], hspace=0.04, wspace=0.05,
                      left=0.06, right=0.995, top=0.94, bottom=0.03)
for c in range(n_cond):
    # EOG: V above H, raw bipolar, one colour
    axE = fig.add_subplot(gs[0, c])
    for k, label in enumerate(("V", "H")):
        offset = -k * eog_spacing
        axE.plot(t, eog[c, k] + offset, color=EOG_COLOUR, lw=0.6)
        if c == 0:
            axE.text(-0.05, offset, label, fontsize=6, ha="right", va="center", color=EOG_COLOUR)
    axE.set_xlim(-0.02, t[-1] + 0.02)
    axE.set_ylim(-eog_spacing * 1.7, eog_spacing * 0.8)
    axE.set_axis_off()
    axE.set_title(conditions[c].replace("_", " "), fontsize=7, pad=2)
    figstyle.panel(axE, "abc"[c])
    # EEG stack: raw black, subject-calibrated MATCH colour
    ax = fig.add_subplot(gs[1, c])
    for r, i in enumerate(rows):
        offset = -r * spacing
        ax.plot(t, raw[c, i] + offset, color=RAW_COLOUR, lw=0.45)
        ax.plot(t, match[c, i] + offset, color=MATCH_COLOUR, lw=0.55)
        if c == 0:
            ax.text(-0.05, offset, stack[r], fontsize=5.5, ha="right", va="center")
    ax.set_xlim(-0.02, t[-1] + 0.02)
    ax.set_ylim(-(len(rows) - 1) * spacing - 0.9 * spacing, 0.9 * spacing)
    ax.set_axis_off()
    if c == n_cond - 1:
        # scale bars: 1 s and eeg_bar microvolt (stack), eog_bar microvolt (EOG)
        x0, y0 = t[-1] - 1.0, -(len(rows) - 1) * spacing - 0.75 * spacing
        ax.plot([x0, x0 + 1.0], [y0, y0], color="0.25", lw=0.8, solid_capstyle="butt")
        ax.plot([x0, x0], [y0, y0 + eeg_bar], color="0.25", lw=0.8, solid_capstyle="butt")
        ax.text(x0 + 0.5, y0 - 0.06 * spacing, "1 s", fontsize=5.5, ha="center", va="top")
        ax.text(x0 - 0.04, y0 + eeg_bar / 2, "%g µV" % eeg_bar, fontsize=5.5, ha="right", va="center")
        xe, ye = t[-1] - 1.0, -eog_spacing * 1.55
        axE.plot([xe, xe], [ye, ye + eog_bar], color="0.25", lw=0.8, solid_capstyle="butt")
        axE.text(xe - 0.04, ye + eog_bar / 2, "%g µV" % eog_bar, fontsize=5.5, ha="right", va="center")
    if c == 0:
        # direct line labelling instead of a legend box
        ax.text(0.0, 0.75 * spacing, "raw", fontsize=6, color=RAW_COLOUR, ha="left", va="bottom")
        ax.text(0.45, 0.75 * spacing, "subject-calibrated", fontsize=6, color=MATCH_COLOUR,
                ha="left", va="bottom")

figstyle.save(fig, NAME)
print(json.dumps({"figure": NAME, "participant": str(npz["participant"]), "sessions": sessions,
                  "conditions": conditions, "starts": starts,
                  "coverage": {a: cov[:, i].tolist() for i, a in enumerate(cov_arms)},
                  "spacing_uv": round(spacing, 2), "eeg_bar_uv": eeg_bar, "eog_bar_uv": eog_bar,
                  "condition_note": str(npz["condition_note"])}))
