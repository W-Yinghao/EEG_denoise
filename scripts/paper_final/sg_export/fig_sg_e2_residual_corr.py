#!/usr/bin/env python3
"""fig-sg-e2-residual-corr — residual EOG correlation per channel on the natural windows.

Reads results/paper_final/paper_final_arrays/natural_residual_corr.npz (export_natural_post.py).
Rows = conditions (raw first), split into two side-by-side blocks of five;
columns per block = vertical / horizontal topographies (32 scalp electrodes,
standard_1020) of the participant-averaged |r| (window -> recording ->
participant -> mean), ONE shared 0-1 colour bar, plus a dot column with the
channel-averaged |r| (all 46 channels) per participant — V lane above H lane,
condition colour, mean as a vertical tick.  No CI / error bars; minimal text.
Second figure (-high): the same on the high-EOG samples (corr_high).
SGEYESUB Fig. 2a/e layout.

Usage: fig_sg_e2_residual_corr.py [npz] [out_dir]   (overrides = smoke test only)
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

SOURCE = Path(sys.argv[1]) if len(sys.argv) > 1 else sg.ARRAYS_OUT / "natural_residual_corr.npz"
SMOKE = len(sys.argv) > 1
if len(sys.argv) > 2:
    figstyle.OUT = Path(sys.argv[2])
NAME = "fig-sg-e2-residual-corr" + ("-smoke" if SMOKE else "")
SHORT = {"RAW": "raw", "MATCH_gated": "subject", "POP": "population", "NO_A0": "unguided",
         "LINEAR": "linear", "WRONG_gated": "mismatched", "SHUFFLED": "shuffled",
         "ICA": "ICA", "ASR": "ASR", "SGEYESUB": "eye-subspace"}
COLOUR = {"MATCH_gated": "MATCH", "WRONG_gated": "WRONG_gated"}
CMAP = "Reds"

# ---------------------------------------------------------------- data
npz = np.load(SOURCE, allow_pickle=False)
names = [str(n) for n in npz["eeg_names"]]
assert names == sg.EEG_NAMES
conditions = [str(c) for c in npz["condition"]]
assert conditions[0] == "RAW"
participants = [str(p) for p in npz["participants"]]
rec_pidx = np.asarray(npz["recording_participant_index"], int)
cov_rec = np.asarray(npz["coverage_recording"], bool)
scalp_idx = [sg.IDX[n] for n in sg.SCALP]
info = mne.create_info(sg.SCALP, sg.RATE, "eeg")
info.set_montage(mne.channels.make_standard_montage("standard_1020"))


def participant_level(values):
    """(cond, rec, 46, 2) -> (cond, participant, 46, 2) nanmean over the participant's covered recordings."""
    out = np.full((len(conditions), len(participants), 46, 2), np.nan)
    for p in range(len(participants)):
        sel = rec_pidx == p
        with np.errstate(invalid="ignore"):
            out[:, p] = np.nanmean(values[:, sel], axis=1)
    return out


def draw(key: str, suffix: str):
    per_p = participant_level(np.asarray(npz[key], float))         # (cond, part, 46, 2)
    with np.errstate(invalid="ignore"):
        mean_map = np.nanmean(per_p, axis=1)                        # (cond, 46, 2)
        chan_mean = np.nanmean(per_p, axis=2)                       # (cond, part, 2)
    blocks = [conditions[:5], conditions[5:]]
    n_rows = max(len(b) for b in blocks)
    fig = plt.figure(figsize=(figstyle.FULL, 0.78 * n_rows + 0.55))
    outer = fig.add_gridspec(1, 2, wspace=0.10, left=0.06, right=0.995, top=0.93, bottom=0.11)
    im = None
    first_ax = None
    for b, block in enumerate(blocks):
        gs = outer[b].subgridspec(n_rows, 3, width_ratios=[1, 1, 1.35], hspace=0.05, wspace=0.05)
        for r, cond in enumerate(block):
            c = conditions.index(cond)
            colour = figstyle.C[COLOUR.get(cond, cond)]
            for a in (0, 1):
                ax = fig.add_subplot(gs[r, a])
                first_ax = first_ax or ax
                values = mean_map[c, :, a]
                if np.isfinite(values[scalp_idx]).all():
                    im, _ = mne.viz.plot_topomap(values[scalp_idx], info, axes=ax, show=False,
                                                 cmap=CMAP, vlim=(0.0, 1.0), contours=4,
                                                 sensors=True, outlines="head")
                else:
                    ax.set_axis_off()
                if r == 0:
                    ax.set_title(("vertical", "horizontal")[a], fontsize=7, pad=2)
                if a == 0:
                    ax.text(-0.12, 0.5, SHORT[cond], transform=ax.transAxes, fontsize=6.5,
                            color=colour, rotation=90, ha="center", va="center")
            # channel-averaged |r| per participant: V lane (y=1) above H lane (y=0)
            axd = fig.add_subplot(gs[r, 2])
            for a, y in ((0, 1.0), (1, 0.0)):
                v = chan_mean[c, :, a]
                axd.scatter(v, np.full(len(v), y), s=7, color=colour, alpha=0.55, lw=0,
                            marker="o" if a == 0 else "D", zorder=2)
                if np.isfinite(v).any():
                    axd.plot([np.nanmean(v)] * 2, [y - 0.32, y + 0.32], color=colour, lw=1.4, zorder=3)
            axd.set_xlim(0, 1)
            axd.set_ylim(-0.6, 1.6)
            axd.set_yticks([1.0, 0.0])
            axd.set_yticklabels(["V", "H"], fontsize=5.5)
            axd.tick_params(axis="y", length=0, pad=1)
            axd.spines["left"].set_visible(False)
            axd.axvline(0, color="0.8", lw=0.5, zorder=1)
            if r == len(block) - 1:
                axd.set_xticks([0, 0.5, 1])
                axd.set_xticklabels(["0", "0.5", "1"], fontsize=5.5)
                axd.set_xlabel("channel-mean |r|", fontsize=6, labelpad=1)
            else:
                axd.set_xticks([0, 0.5, 1])
                axd.set_xticklabels([])
            axd.tick_params(axis="x", length=1.5)
            if r == 0:
                axd.set_title("per participant", fontsize=7, pad=2)
    if im is not None:
        cax = fig.add_axes([0.09, 0.035, 0.24, 0.014])   # under the left head columns
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_ticks([0, 0.5, 1])
        cb.set_label("|r| with EOG", fontsize=6, labelpad=1)
        cb.ax.tick_params(labelsize=5.5, length=1.5, width=0.5)
        cb.outline.set_linewidth(0.5)
    figstyle.panel(first_ax, "a" if suffix == "" else "b")
    figstyle.save(fig, NAME + suffix)
    plt.close(fig)
    return {cond: [round(float(np.nanmean(chan_mean[c, :, a])), 3) for a in (0, 1)]
            for c, cond in enumerate(conditions)}


summary = {"all_samples": draw("corr_recording", ""), "high_eog_samples": draw("corr_high_recording", "-high"),
           "participants": len(participants), "recordings_covered": cov_rec.sum(axis=1).tolist()}
print(json.dumps({"figure": NAME, **summary}))
