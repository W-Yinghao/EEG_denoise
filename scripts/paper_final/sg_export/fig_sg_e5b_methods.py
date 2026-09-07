#!/usr/bin/env python3
"""fig-sg-e5b-methods — held-out paired exemplar (sub-04): every method on the same input.

Reads results/paper_final/paper_final_arrays/t6_heldout_methods_sub-04.npz (export_e5b.py).

Layout (EEGDfus Fig. 7 style): one column per window, two rows (Fp1, AFz); in each
panel five stacked traces, top to bottom
    contaminated / linear / unguided / population / subject-calibrated
each in its arm colour with the reference overlaid as a thin black line.  A second
file (fig-sg-e5b-methods-ref) shows contaminated / ICA / ASR / eye-subspace /
subject-calibrated in the same layout.  Windows are selected by their 0-BASED index
(default 1, 4, 6 = the doc's "windows 1, 4, 6" read 0-based; the 1-based reading
would include window 3, a zero-artifact control); pass --one-based to draw the
other convention.  Every panel has its own vertical scale (the windows differ in
amplitude by an order of magnitude), so each carries its own amplitude bar in the
right margin (fold-scaled EEG units); time is on the shared x axis.

House rules: no CIs / error bars / bands; minimal text; one arm = one colour;
runtime-loaded data only.

Usage:  fig_sg_e5b_methods.py [--windows 1,4,6] [--one-based]
        fig_sg_e5b_methods.py --synthetic      shapes-only render to the scratchpad
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/infres/yinwang/denoiseNet/scripts/paper_final/sg_export")
import sg_common as sg  # noqa: E402

sys.path.insert(0, "/home/infres/yinwang/denoiseNet/scripts/paper_final/figures")
import figstyle  # noqa: E402

figstyle.setup()

import matplotlib.pyplot as plt  # noqa: E402

SOURCE = sg.ARRAYS_OUT / "t6_heldout_methods_sub-04.npz"
CHANNELS = ("Fp1", "AFz")
# (npz key, figstyle.C key, in-figure label)
MAIN = (("contaminated", "RAW", "contaminated"), ("linear_regression", "LINEAR", "linear"),
        ("unguided", "NO_A0", "unguided"), ("population", "POP", "population"),
        ("matched_mean", "MATCH", "subject cal."))
REF = (("contaminated", "RAW", "contaminated"), ("ica", "ICA", "ICA"), ("asr", "ASR", "ASR"),
       ("eye_subspace", "SGEYESUB", "eye-subspace"), ("matched_mean", "MATCH", "subject cal."))
NAMES = {"main": "fig-sg-e5b-methods", "ref": "fig-sg-e5b-methods-ref"}


def load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def synthetic(rng: np.random.Generator) -> dict:
    """Random arrays with the exported shapes (smoke only; never saved to artifacts)."""
    t = np.arange(512) / sg.RATE
    ref = rng.normal(0, 1, (8, 46, 512)).astype(np.float32)
    blink = 6 * np.exp(-((t - 2.5) / 0.15) ** 2)[None, None, :]
    d = {"reference": ref, "contaminated": ref + blink * rng.uniform(0.2, 1.5, (8, 46, 1)),
         "window_index_0based": np.arange(8), "window_index_1based": np.arange(1, 9),
         "eeg_names": np.asarray(sg.EEG_NAMES), "zero_artifact": np.zeros(8, int)}
    arms = [k for k, _, _ in MAIN + REF if k != "contaminated"]
    cov = np.ones((len(arms), 8), bool)
    for a, arm in enumerate(arms):
        d[arm] = (ref + 0.3 * rng.normal(0, 1, ref.shape)).astype(np.float32)
        if arm == "asr":
            cov[a, 4] = False
            d[arm][4] = np.nan
    d["coverage"] = cov
    d["arm_names"] = np.asarray(arms)
    return d


def _covered(d: dict, key: str, pos: int) -> bool:
    if key == "contaminated":
        return True
    arms = [str(a) for a in d["arm_names"]]
    return bool(d["coverage"][arms.index(key), pos])


def _nice(value: float) -> float:
    exp = 10 ** np.floor(np.log10(max(value, 1e-12)))
    return float(exp * max(m for m in (1, 2, 5) if m * exp <= value) if value >= exp else exp)


def draw(d: dict, arms, windows0, name: str, out_dir: Path | None) -> None:
    names = [str(n) for n in d["eeg_names"]]
    idx = {n: i for i, n in enumerate(names)}
    win0 = [int(w) for w in d["window_index_0based"]]
    win1 = [int(w) for w in d["window_index_1based"]]
    positions = [win0.index(w) for w in windows0]
    t = np.arange(d["reference"].shape[-1]) / sg.RATE
    ncol = len(windows0)
    fig, axes = plt.subplots(len(CHANNELS), ncol, figsize=(figstyle.FULL, 3.6), squeeze=False,
                             gridspec_kw={"left": 0.11, "right": 0.985, "top": 0.94, "bottom": 0.09,
                                          "hspace": 0.18, "wspace": 0.08})
    ref_c = figstyle.C["reference"]
    for r, ch in enumerate(CHANNELS):
        ci = idx[ch]
        for c, (w0, pos) in enumerate(zip(windows0, positions)):
            ax = axes[r, c]
            ref = d["reference"][pos, ci].astype(np.float64)
            traces = []
            for key, colour_key, label in arms:
                ok = _covered(d, key, pos)
                x = d[key][pos, ci].astype(np.float64) if ok else None
                traces.append((key, figstyle.C[colour_key], label, x, ok))
            ptp = max([np.ptp(ref)] + [np.ptp(x) for _, _, _, x, ok in traces if ok])
            step = 1.15 * ptp
            lo, hi = np.inf, -np.inf
            for k, (key, colour, label, x, ok) in enumerate(traces):
                off = -k * step
                if ok:
                    ax.plot(t, x + off, color=colour, lw=0.7, zorder=2)
                    lo, hi = min(lo, (x + off).min()), max(hi, (x + off).max())
                else:
                    ax.text(t[-1] / 2, off, "n/a", color="0.45", fontsize=5.5, ha="center", va="center",
                            zorder=5, bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
                ax.plot(t, ref + off, color=ref_c, lw=0.4, zorder=3)
                lo, hi = min(lo, (ref + off).min()), max(hi, (ref + off).max())
                if c == 0:
                    ax.text(-0.02, off, label, color=colour, fontsize=5.5, ha="right", va="center",
                            transform=ax.get_yaxis_transform())
            pad = 0.06 * (hi - lo)
            ax.set_ylim(lo - pad, hi + pad)
            ax.set_xlim(t[0], t[-1] + 0.9)          # right margin holds the per-panel amplitude bar
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
            # per-panel amplitude scale bar (each panel has its own vertical scale), centred
            # on the bottom trace's baseline, in the reserved right margin
            amp = _nice(0.5 * ptp)
            xb, yb = t[-1] + 0.3, -(len(traces) - 1) * step - amp / 2
            ax.plot([xb, xb], [yb, yb + amp], color="0.15", lw=0.8, solid_capstyle="butt")
            ax.text(xb + 0.08, yb + amp / 2, f"{amp:g}", color="0.15", fontsize=5, ha="left", va="center")
            if r == 0:
                ax.text(0.01, 1.0, f"w{w0} (1-based {win1[pos]})", color="0.3", fontsize=5.5,
                        ha="left", va="bottom", transform=ax.transAxes)
            if r < len(CHANNELS) - 1:
                ax.set_xticks([])
                ax.spines["bottom"].set_visible(False)
            else:
                ax.set_xticks([0, 2.5, 5])
                ax.tick_params(axis="x", pad=1.5)
                if c == ncol // 2:
                    ax.set_xlabel("time (s)", labelpad=1)
            if c == 0:
                figstyle.panel(ax, "AB"[r])
                ax.text(0.99, 1.0, ch, color="0.15", fontsize=6.5, fontweight="bold", ha="right",
                        va="bottom", transform=ax.transAxes)
    if out_dir is None:
        figstyle.save(fig, name)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{name}.png", dpi=110, bbox_inches="tight")
        print(f"saved {out_dir}/{name}.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", default="1,4,6", help="window indices (0-based unless --one-based)")
    parser.add_argument("--one-based", action="store_true")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    windows = [int(w) for w in args.windows.split(",")]
    windows0 = [w - 1 for w in windows] if args.one_based else windows
    print(json.dumps({"windows_drawn_0based": windows0, "windows_drawn_1based": [w + 1 for w in windows0],
                      "convention": "1-based input" if args.one_based else "0-based input"}))
    if args.synthetic:
        d = synthetic(np.random.default_rng(0))
        out_dir = Path("/tmp/claude-34987/-home-infres-yinwang-denoiseNet/11cbfe34-6c44-4498-bfb8-deafdf46e607/scratchpad")
    else:
        d = load(SOURCE)
        out_dir = None
    for kind, arms in (("main", MAIN), ("ref", REF)):
        draw(d, arms, windows0, NAMES[kind], out_dir)


if __name__ == "__main__":
    main()
