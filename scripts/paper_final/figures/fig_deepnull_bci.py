#!/usr/bin/env python3
"""fig-deepnull-bci — Deep-decoder SSVEP and BCI IV-2a null results (appendix).

Panel A: EEGNet-8,2 SSVEP accuracy, per-participant dots + arm-mean ticks,
dev (n=15) | held-out (n=8) sub-axes; chance (1/len(frozen freqs)) and the
shallow-CCA RAW reference show the deep endpoint is underpowered.
Panel B: BCI IV-2a route 1 (calibrated linear operator, EEGNet, official T/E
split) per-subject paired Delta accuracy vs RAW, subjects connected across arms.
Panel C: route 2 (22-ch population prior + calibrated guide, participant-held-
out priors) same layout.
Panel D: per-cell anchored vs unguided training-episode RRMSE (18 cells, T/E),
mean cross from the banked gate file — the guide works mechanically on 2a
even though downstream accuracy does not move.

House rules: NO confidence intervals / error bars / whiskers / bootstrap
bands anywhere — per-participant dots and paired lines instead. All numbers
are read from the banked npz/json at runtime.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/infres/yinwang/denoiseNet/scripts/paper_final/figures")
import figstyle

figstyle.setup()

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path("/home/infres/yinwang/denoiseNet")
ARR = ROOT / "paper_final_arrays"
C = figstyle.C


def load(name):
    z = np.load(ARR / name, allow_pickle=True)
    return json.loads(str(z["decision"][()]))


deep_dev = load("dwave_deep_ssvep_dev.npz")
deep_held = load("dwave_deep_ssvep_heldout.npz")
route1 = load("bci2a_route1.npz")
route2 = load("bci2a_route2.npz")
dwave_dev = load("dwave_dev.npz")      # shallow-CCA RAW reference (dev)
dwave_held = load("dwave_heldout.npz")  # shallow-CCA RAW reference (held-out)
episodes = json.loads(
    (ROOT / "results/paper_final/bci2a/route2_episodes.json").read_text())
gate = json.loads(
    (ROOT / "results/paper_final/bci2a/route2_gate.json").read_text())

CHANCE = 1.0 / len(dwave_dev["frozen_frequencies_hz"])  # 3 frozen freqs

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(figstyle.FULL, 2.55))
gs = fig.add_gridspec(1, 5, width_ratios=[1.06, 1.06, 0.92, 0.92, 1.30],
                      wspace=0.34, left=0.075, right=0.985,
                      top=0.86, bottom=0.19)

# -- Panel A: deep SSVEP absolute accuracy, dev | held-out ---------------
ARMS_A = ["RAW", "LINEAR", "NO_A0", "MATCH", "POP", "ICA", "ASR", "SGEYESUB"]
DISP = {"NO_A0": "NO-A0", "LINEAR_MATCH": "MATCH", "LINEAR_POP": "POP",
        "LINEAR_WRONG": "WRONG"}
COL = {"LINEAR_MATCH": "MATCH", "LINEAR_POP": "POP", "LINEAR_WRONG": "WRONG"}

aL = fig.add_subplot(gs[0])
aR = fig.add_subplot(gs[1], sharey=aL, sharex=aL)
ys = np.arange(len(ARMS_A))[::-1]
rng = np.random.default_rng(7)
for ax, d, cca in ((aL, deep_dev, dwave_dev["d1_ssvep_accuracy"]["RAW"]),
                   (aR, deep_held, dwave_held["d1_ssvep_accuracy"]["RAW"])):
    per = {}
    for r in d["rows"]:
        per.setdefault(r["arm"], []).append(r["metric"])
    for arm, y in zip(ARMS_A, ys):
        v = np.asarray(per[arm])
        assert np.isclose(v.mean(), d["means"][arm], atol=1e-9)
        jit = rng.uniform(-0.24, 0.24, size=len(v))
        ax.plot(v, y + jit, "o", ms=1.8, color=C[arm], mec="none",
                alpha=0.55, zorder=3)
        ax.plot([v.mean()] * 2, [y - 0.32, y + 0.32], color=C[arm],
                lw=1.4, zorder=4)
        if len(v) != d["participants"]:  # disclose smaller cohort (ASR
            # per-cell calibration drop in the banked data)
            ax.text(v.max() + 0.045, y, f"n={len(v)}", fontsize=4.8,
                    color=C[arm], ha="left", va="center")
    ax.axvline(CHANCE, color="0.55", lw=0.7, ls=(0, (2, 2)), zorder=1)
    ax.axvline(cca, color=C["reference"], lw=0.9, zorder=1)
    ax.text(cca - 0.02, len(ARMS_A) - 0.55, f"CCA\n{cca:.2f}", ha="right",
            va="top", fontsize=5.0, color=C["reference"])
    ax.set_ylim(-0.6, len(ARMS_A) - 0.4)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
aL.text(CHANCE + 0.015, 1.5, "chance", fontsize=4.8, color="0.55",
        ha="left", va="center")
aL.set_xlim(0.22, 0.90)
aL.set_yticks(ys)
aL.set_yticklabels([DISP.get(a, a) for a in ARMS_A], fontsize=5.6)
for tick, arm in zip(aL.get_yticklabels(), ARMS_A):
    tick.set_color(C[arm])
plt.setp(aR.get_yticklabels(), visible=False)
aR.tick_params(axis="y", length=0)
aL.set_title(f"dev · n={deep_dev['participants']}", fontsize=5.2,
             color="0.4", pad=2)
aR.set_title(f"held-out · n={deep_held['participants']}", fontsize=5.2,
             color="0.4", pad=2)
aL.set_xlabel("SSVEP accuracy (EEGNet-8,2)")
aL.xaxis.set_label_coords(1.10, -0.135)
figstyle.panel(aL, "A")

# -- Panels B, C: BCI IV-2a paired Delta accuracy vs RAW -----------------
def route_panel(ax, d, arms, tag):
    per = {}
    for r in d["rows"]:
        per.setdefault(r["subject"], {})[r["arm"]] = r["accuracy"]
    subs = sorted(per)
    off = np.linspace(-0.14, 0.14, len(subs))  # same subject = same offset
    xs = np.arange(len(arms))
    deltas = {a: np.asarray([per[s][a] - per[s]["RAW"] for s in subs])
              for a in arms}
    for a in arms:
        banked = d["contrasts_vs_RAW"][f"{a}_minus_RAW"]
        assert len(deltas[a]) == banked["participants"]
        assert np.isclose(deltas[a].mean(), banked["mean"], atol=1e-9)
    for i, s in enumerate(subs):  # paired subject lines across arms
        ax.plot(xs + off[i], [deltas[a][i] for a in arms], color="0.6",
                lw=0.5, alpha=0.55, zorder=2)
    for x, a in zip(xs, arms):
        col = C[COL.get(a, a)]
        ax.plot(x + off, deltas[a], "o", ms=2.2, color=col, mec="none",
                alpha=0.8, zorder=3)
        ax.plot([x - 0.24, x + 0.24], [deltas[a].mean()] * 2, color=col,
                lw=1.5, zorder=4)
    ax.axhline(0, color=C["RAW"], lw=0.7, ls=(0, (2, 2)), zorder=1)
    ax.set_xlim(-0.55, len(arms) - 0.45)
    ax.set_xticks(xs)
    ax.set_xticklabels([DISP.get(a, a) for a in arms], fontsize=5.4)
    for tick, a in zip(ax.get_xticklabels(), arms):
        tick.set_color(C[COL.get(a, a)])
    ax.set_title(f"route {tag} · RAW {d['accuracy_means']['RAW']:.3f}",
                 fontsize=5.2, color="0.4", pad=2)

bx = fig.add_subplot(gs[2])
cx = fig.add_subplot(gs[3], sharey=bx)
route_panel(bx, route1, ["LINEAR_MATCH", "LINEAR_POP", "LINEAR_WRONG"], "1")
route_panel(cx, route2, ["MATCH", "NO_A0", "POP"], "2")
bx.set_ylim(-0.125, 0.115)
bx.set_ylabel("$\\Delta$ accuracy vs RAW")
bx.yaxis.set_label_coords(-0.36, 0.5)
plt.setp(cx.get_yticklabels(), visible=False)
cx.tick_params(axis="y", length=0)
figstyle.panel(bx, "B")
figstyle.panel(cx, "C")

# -- Panel D: anchored vs unguided episode RRMSE, 18 cells ---------------
dx = fig.add_subplot(gs[4])
anch = np.asarray([e["rrmse_anchored"] for e in episodes])
ungd = np.asarray([e["rrmse_unguided"] for e in episodes])
is_T = np.asarray([e["cell"].endswith("T") for e in episodes])
assert len(episodes) == gate["cells"] == 18 and gate["gate_pass"]
assert np.isclose(anch.mean(), gate["mean_anchored"], atol=1e-9)
assert np.isclose(ungd.mean(), gate["mean_unguided"], atol=1e-9)

lim = 1.03 * max(ungd.max(), anch.max())
dx.plot([0, lim], [0, lim], color="0.6", lw=0.7, ls=(0, (2, 2)), zorder=1)
dx.text(0.965 * lim, 0.90 * lim, "y=x", fontsize=5.0, color="0.5",
        ha="right", va="bottom", rotation=45)
dx.plot(ungd[is_T], anch[is_T], "o", ms=3.0, color=C["MATCH"], mec="none",
        alpha=0.85, zorder=3, label="T")
dx.plot(ungd[~is_T], anch[~is_T], "o", ms=3.0, mfc="none", mew=0.9,
        color=C["MATCH"], zorder=3, label="E")
# banked gate means as a reference cross (source: route2_gate.json)
dx.plot(gate["mean_unguided"], gate["mean_anchored"], "+", ms=6, mew=1.1,
        color=C["reference"], zorder=4)
dx.text(0.05, 0.66, f"mean {gate['mean_unguided']:.2f}"
        f"$\\to${gate['mean_anchored']:.2f}", transform=dx.transAxes,
        fontsize=5.0, color=C["reference"], ha="left", va="top")
dx.set_xlim(0, lim)
dx.set_ylim(0, lim)
dx.set_aspect("equal")
dx.xaxis.set_major_locator(MaxNLocator(nbins=4))
dx.yaxis.set_major_locator(MaxNLocator(nbins=4))
dx.set_xlabel("unguided RRMSE")
dx.set_ylabel("anchored RRMSE")
dx.legend(loc="upper left", fontsize=5.5, handletextpad=0.3,
          borderaxespad=0.2, labelspacing=0.25)
figstyle.panel(dx, "D")

figstyle.save(fig, "fig-deepnull-bci")
