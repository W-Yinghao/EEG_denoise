#!/usr/bin/env python3
"""fig-sg-e3-psd-ratio — spectral cost on the low-EOG samples (SGEYESUB Fig. 3).

Reads results/paper_final/paper_final_arrays/natural_psd_ratio.npz (export_natural_post.py).
3x3 grid = sg_common.NINE (F3 Fz F4 / C3 Cz C4 / P3 Pz P4); y = PSD ratio
corrected / uncorrected in dB against frequency; per-participant thin lines
(mean over the participant's covered recordings) plus a thick participant-mean
line per condition, one arm = one colour; 0 dB reference line; raw is the
reference (identically 0 dB) and is not drawn.  No CI / bands; direct line
labels in one panel; minimal text.

Usage: fig_sg_e3_psd_ratio.py [npz] [out_dir]   (overrides = smoke test only)
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

SOURCE = Path(sys.argv[1]) if len(sys.argv) > 1 else sg.ARRAYS_OUT / "natural_psd_ratio.npz"
SMOKE = len(sys.argv) > 1
if len(sys.argv) > 2:
    figstyle.OUT = Path(sys.argv[2])
NAME = "fig-sg-e3-psd-ratio" + ("-smoke" if SMOKE else "")
SHORT = {"MATCH_gated": "subject", "POP": "population", "NO_A0": "unguided", "LINEAR": "linear",
         "WRONG_gated": "mismatched", "SHUFFLED": "shuffled", "ICA": "ICA", "ASR": "ASR",
         "SGEYESUB": "eye-subspace"}
COLOUR = {"MATCH_gated": "MATCH", "WRONG_gated": "WRONG_gated"}

# ---------------------------------------------------------------- data
npz = np.load(SOURCE, allow_pickle=False)
conditions = [str(c) for c in npz["condition"]]
electrodes = [str(e) for e in npz["electrodes"]]
freqs = np.asarray(npz["freqs"], float)
ratio = np.asarray(npz["ratio_db"], float)                      # (cond, rec, electrode, freq)
participants = [str(p) for p in npz["participants"]]
rec_pidx = np.asarray(npz["recording_participant_index"], int)
drawn = [c for c in conditions if c != "RAW"]

per_p = np.full((len(conditions), len(participants), len(electrodes), len(freqs)), np.nan)
for p in range(len(participants)):
    with np.errstate(invalid="ignore"):
        per_p[:, p] = np.nanmean(ratio[:, rec_pidx == p], axis=1)
with np.errstate(invalid="ignore"):
    mean = np.nanmean(per_p, axis=1)                             # (cond, electrode, freq)

# common y-limits from the participant lines of the nine electrodes
nine_idx = [electrodes.index(e) for e in sg.NINE]
lines = per_p[[conditions.index(c) for c in drawn]][:, :, nine_idx]
lo, hi = np.nanpercentile(lines, [1, 99])
ylim = (min(lo, -1.0) - 0.5, max(hi, 1.0) + 0.5)

fig, axes = plt.subplots(3, 3, figsize=(figstyle.FULL, 4.6), sharex=True, sharey=True)
fig.subplots_adjust(left=0.07, right=0.99, top=0.92, bottom=0.08, hspace=0.22, wspace=0.08)
for k, name in enumerate(sg.NINE):
    ax = axes[k // 3, k % 3]
    e = electrodes.index(name)
    ax.axhline(0.0, color=figstyle.C["RAW"], lw=0.8, ls="--", zorder=1)
    for cond in drawn:
        c = conditions.index(cond)
        colour = figstyle.C[COLOUR.get(cond, cond)]
        for p in range(len(participants)):
            if np.isfinite(per_p[c, p, e]).any():
                ax.plot(freqs, per_p[c, p, e], color=colour, lw=0.35, alpha=0.22, zorder=2)
        ax.plot(freqs, mean[c, e], color=colour, lw=1.4, zorder=3)
    ax.text(0.03, 0.96, name, transform=ax.transAxes, fontsize=7, va="top", ha="left")
    ax.set_xlim(sg.FMIN, sg.FMAX)
    ax.set_ylim(*ylim)
    if k // 3 == 2:
        ax.set_xlabel("frequency (Hz)")
    if k % 3 == 0:
        ax.set_ylabel("PSD ratio (dB)")
# direct labels: one coloured row above the grid (no legend box), ordered by the mean level at 15 Hz
order = sorted(drawn, key=lambda cond: -np.nanmean(mean[conditions.index(cond), nine_idx, -1]))
x = 0.07
for cond in order:
    t = fig.text(x, 0.975, SHORT[cond], fontsize=6, color=figstyle.C[COLOUR.get(cond, cond)], ha="left", va="top")
    x += 0.012 + 0.0095 * len(SHORT[cond])
figstyle.panel(axes[0, 0], "")
figstyle.save(fig, NAME)
print(json.dumps({"figure": NAME, "nperseg": int(npz["nperseg"]), "ylim": [round(v, 2) for v in ylim],
                  "mean_db_at_Fz_last_freq": {c: round(float(mean[conditions.index(c), electrodes.index("Fz"), -1]), 2)
                                              for c in drawn},
                  "mean_db_at_Fz_first_freq": {c: round(float(mean[conditions.index(c), electrodes.index("Fz"), 0]), 2)
                                               for c in drawn}}))
