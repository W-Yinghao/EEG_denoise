#!/usr/bin/env python3
"""fig-sg-e4-erp — ERP target-minus-nontarget time courses and P300 peak topographies.

Reads results/paper_final/paper_final_arrays/erp_topographies.npz (export_e4.py).

Layout (one column per arm, raw first; one arm = one colour, raw in black):
  (a) rows 1-4: target-minus-nontarget at Pz and Fz, dev cohort (rows 1-2) then the
      held-out cohort (rows 3-4).  Per-participant thin lines + the participant-first
      mean (bold); the raw mean repeated as a thin black dotted reference in every
      other column; dotted vertical line = the raw P300 peak latency of that cohort.
  (b) rows 5-6: topography of the grand-average target-minus-nontarget at the raw P300
      peak latency, dev / held-out, on the 32 scalp electrodes (standard_1020), one
      symmetric colour scale for every map.  Ear channels omitted.

House rules: no CIs / error bars / bands; minimal text; runtime-loaded data only.

Usage:  fig_sg_e4_erp.py               render from the exported npz
        fig_sg_e4_erp.py --synthetic   shapes-only render to the scratchpad (smoke)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/infres/yinwang/denoiseNet/scripts/paper_final/sg_export")
import sg_common as sg  # noqa: E402

sys.path.insert(0, "/home/infres/yinwang/denoiseNet/scripts/paper_final/figures")
import figstyle  # noqa: E402

figstyle.setup()

import matplotlib.pyplot as plt  # noqa: E402
import mne  # noqa: E402

mne.set_log_level("ERROR")

NAME = "fig-sg-e4-erp"
ELECTRODES = ("Pz", "Fz")
COHORT_LABEL = {"dev": "dev", "heldout": "held-out"}
ARM_LABEL = {"RAW": "raw", "MATCH": "subject cal.", "POP": "population cal.",
             "NO_A0": "unguided", "LINEAR": "linear", "ICA": "ICA", "ASR": "ASR",
             "SGEYESUB": "eye-subspace"}
RAW_COLOUR = "#000000"      # the export doc asks for raw in black in this figure


def colour(arm: str) -> str:
    return RAW_COLOUR if arm == "RAW" else figstyle.C[arm]


def load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def synthetic(rng: np.random.Generator) -> dict:
    """Random arrays with the exported shapes (smoke only; never saved to artifacts)."""
    arms = list(sg.DWAVE_ARMS)
    t = (np.arange(100) - 20) / 100
    bump = np.exp(-((t - 0.38) / 0.08) ** 2)
    pt = rng.normal(0, 1, (2, len(arms), 15, 46, 100)).astype(np.float32)
    pn = rng.normal(0, 1, (2, len(arms), 15, 46, 100)).astype(np.float32)
    pt += 3 * bump[None, None, None, None, :]
    cov = np.ones((2, len(arms), 15), bool)
    cov[1, :, 8:] = False
    cov[1, arms.index("ASR"), :] = False       # exercise the "n/a" path
    pt[~cov] = np.nan
    pn[~cov] = np.nan
    grand = np.stack([np.nanmean(pn, axis=2), np.nanmean(pt, axis=2)], axis=2)
    return {"grand_average": grand, "participant_target": pt, "participant_nontarget": pn,
            "coverage": cov, "arms": np.asarray(arms), "cohorts": np.asarray(["dev", "heldout"]),
            "time_s": t, "p300_peak_latency_s": np.asarray([0.38, 0.40]),
            "p300_peak_index": np.asarray([58, 60]), "eeg_names": np.asarray(sg.EEG_NAMES),
            "participants": np.asarray([["p"] * 15, ["p"] * 8 + [""] * 7])}


def draw(d: dict, out_name: str | None) -> Path:
    arms = [str(a) for a in d["arms"]]
    cohorts = [str(c) for c in d["cohorts"]]
    names = [str(n) for n in d["eeg_names"]]
    idx = {n: i for i, n in enumerate(names)}
    scalp = names[:32]
    t = d["time_s"]
    grand, pt, pn, cov = d["grand_average"], d["participant_target"], d["participant_nontarget"], d["coverage"]
    peak_idx = d["p300_peak_index"].astype(int)
    peak_lat = d["p300_peak_latency_s"]
    raw_i = arms.index("RAW")
    n_arm = len(arms)

    # rows: (cohort, electrode) x 4 time-course rows, then 2 topo rows
    rows = [(ci, e) for ci in range(len(cohorts)) for e in ELECTRODES]
    fig = plt.figure(figsize=(figstyle.FULL, 6.4))
    gs = fig.add_gridspec(len(rows) + len(cohorts), n_arm,
                          height_ratios=[1.0] * len(rows) + [1.25] * len(cohorts),
                          left=0.085, right=0.985, top=0.94, bottom=0.085,
                          hspace=0.35, wspace=0.12)

    # ---- (a) time courses
    # y-limits per row from the per-participant difference curves (1st-99th percentile)
    ylim = {}
    for ri, (ci, e) in enumerate(rows):
        diffs = (pt[ci, :, :, idx[e]] - pn[ci, :, :, idx[e]])[cov[ci]]
        if diffs.size and np.isfinite(diffs).any():
            lo, hi = np.nanpercentile(diffs, [1, 99])
            pad = 0.08 * (hi - lo)
            ylim[ri] = (lo - pad, hi + pad)
        else:
            ylim[ri] = (-1, 1)
    axes_a = np.empty((len(rows), n_arm), dtype=object)
    for ri, (ci, e) in enumerate(rows):
        raw_mean = grand[ci, raw_i, 1, idx[e]] - grand[ci, raw_i, 0, idx[e]]
        for ai, arm in enumerate(arms):
            ax = fig.add_subplot(gs[ri, ai])
            axes_a[ri, ai] = ax
            ax.axhline(0, color="0.85", lw=0.5, zorder=0)
            ax.axvline(peak_lat[ci], color="0.6", lw=0.5, ls=":", zorder=0)
            if cov[ci, ai].any():
                for pi in np.flatnonzero(cov[ci, ai]):
                    ax.plot(t, pt[ci, ai, pi, idx[e]] - pn[ci, ai, pi, idx[e]],
                            color=colour(arm), lw=0.4, alpha=0.3, zorder=1)
                if ai != raw_i:
                    ax.plot(t, raw_mean, color=RAW_COLOUR, lw=0.6, ls=":", alpha=0.8, zorder=2)
                ax.plot(t, grand[ci, ai, 1, idx[e]] - grand[ci, ai, 0, idx[e]],
                        color=colour(arm), lw=1.2, zorder=3)
            else:
                ax.text(0.5, 0.5, "n/a", transform=ax.transAxes, ha="center", va="center",
                        fontsize=6, color="0.5")
            ax.set_xlim(t[0], t[-1])
            ax.set_ylim(*ylim[ri])
            ax.set_xticks([0, 0.4, 0.8])
            if ri != len(rows) - 1:
                ax.set_xticklabels([])
            if ai == 0:
                ax.set_ylabel(f"{COHORT_LABEL[cohorts[ci]]}\n{e}  (µV)", fontsize=6)
            else:
                ax.set_yticklabels([])
                ax.tick_params(axis="y", length=0)
            if ri == 0:
                ax.set_title(ARM_LABEL.get(arm, arm), fontsize=6.5, color=colour(arm), pad=3)
            ax.tick_params(labelsize=5, length=1.5)
    axes_a[-1, 0].set_xlabel("time (s)", fontsize=6)
    fig.text(0.012, axes_a[0, 0].get_position().y1 + 0.012, "a", fontsize=9,
             fontweight="bold", va="bottom", ha="left")

    # ---- (b) topographies at the raw P300 peak latency, one symmetric colour scale
    info = mne.create_info(scalp, sg.RATE, "eeg")
    info.set_montage(mne.channels.make_standard_montage("standard_1020"))
    maps = np.full((len(cohorts), n_arm, 32), np.nan)
    for ci in range(len(cohorts)):
        for ai in range(n_arm):
            if cov[ci, ai].any():
                maps[ci, ai] = (grand[ci, ai, 1, :32, peak_idx[ci]]
                                - grand[ci, ai, 0, :32, peak_idx[ci]])
    vmax = float(np.nanmax(np.abs(maps))) if np.isfinite(maps).any() else 1.0
    vmax = float(np.ceil(vmax * 2) / 2)  # round up to 0.5 µV
    im = None
    first_topo = None
    for ci in range(len(cohorts)):
        for ai, arm in enumerate(arms):
            ax = fig.add_subplot(gs[len(rows) + ci, ai])
            if first_topo is None:
                first_topo = ax
            if not np.isfinite(maps[ci, ai]).all():
                ax.set_axis_off()
                ax.text(0.5, 0.5, "n/a", transform=ax.transAxes, ha="center", va="center",
                        fontsize=6, color="0.5")
                continue
            im, _ = mne.viz.plot_topomap(maps[ci, ai], info, axes=ax, show=False,
                                         cmap="RdBu_r", vlim=(-vmax, vmax), contours=0,
                                         sensors=False, outlines="head")
            for spine in ax.spines.values():
                spine.set_visible(False)
            if ai == 0:
                box = ax.get_position()
                fig.text(0.012, 0.5 * (box.y0 + box.y1),
                         f"{COHORT_LABEL[cohorts[ci]]}\n{peak_lat[ci]:.2f} s",
                         ha="left", va="center", fontsize=6)
    fig.text(0.012, first_topo.get_position().y1 + 0.012, "b", fontsize=9,
             fontweight="bold", va="bottom", ha="left")
    if im is not None:
        cax = fig.add_axes([0.86, 0.028, 0.12, 0.010])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_ticks([-vmax, 0, vmax])
        cb.ax.tick_params(labelsize=5, length=1.5, width=0.5)
        cb.outline.set_linewidth(0.5)
        cb.set_label("target − nontarget (µV)", fontsize=5.5, labelpad=1)

    if out_name is None:
        out = Path(sg.HERE.parent.parent.parent / "artifacts/figures/v3") / f"{NAME}.pdf"
        figstyle.save(fig, NAME)
    else:
        out = Path(out_name)
        fig.savefig(out, dpi=110)
        print(f"saved {out}")
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", default=None,
                        help="render random arrays of the exported shapes to this PNG path (smoke)")
    args = parser.parse_args()
    if args.synthetic:
        draw(synthetic(np.random.default_rng(0)), args.synthetic)
    else:
        draw(load(sg.ARRAYS_OUT / "erp_topographies.npz"), None)


if __name__ == "__main__":
    main()
