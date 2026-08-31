#!/usr/bin/env python3
"""fig-downstream — Downstream utility in three currencies (v3 main figure).

Panel A: SSVEP CCA accuracy delta vs RAW (D1), dev | held-out forests, plus
MATCH-by-EOG-contamination-tertile ticks (L/M/H).
Panel B: ERP shrinkage-LDA AUC delta vs RAW (D2), dev | held-out forests.
Panel C: deep-decoder (EEGNet-8,2) ERP AUC delta vs RAW with per-participant
paired dots (dev n=15, held-out n=8) and the arm mean as a short tick.
Panel D: ERP preservation vs ERP-AUC delta per arm, both cohorts — the
"helps the classifier, destroys the morphology" trade-off (held-out ASR).

House rules: NO confidence intervals / error bars / bootstrap bands anywhere.
D1/D2 files bank only per-arm summaries (mean, median, k/n participants
improved) — those are plotted; the deep-ERP files bank per-participant rows,
plotted as dots. All numbers are read from the banked npz at runtime.
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
from matplotlib.transforms import blended_transform_factory

ARR = Path("/home/infres/yinwang/denoiseNet/paper_final_arrays")
C = figstyle.C

def load(name):
    z = np.load(ARR / name, allow_pickle=True)
    return json.loads(str(z["decision"][()]))

dev = load("dwave_dev.npz")
held = load("dwave_heldout.npz")
deep_dev = load("dwave_deep_erp_dev.npz")
deep_held = load("dwave_deep_erp_heldout.npz")

ARMS = ["LINEAR", "NO_A0", "MATCH", "POP", "ICA", "ASR", "SGEYESUB"]
DISP = {"LINEAR": "LINEAR", "NO_A0": "NO-A0", "MATCH": "MATCH", "POP": "POP",
        "ICA": "ICA", "ASR": "ASR", "SGEYESUB": "SGEYESUB"}

def deep_deltas(d):
    """Per-participant paired delta (arm - RAW) from the banked rows."""
    per = {}
    for r in d["rows"]:
        per.setdefault(r["participant"], {})[r["arm"]] = r["metric"]
    out = {}
    for arm in ARMS:
        vals = [per[p][arm] - per[p]["RAW"] for p in sorted(per)
                if arm in per[p] and "RAW" in per[p]]
        out[arm] = np.asarray(vals)
        banked = d["contrasts_vs_RAW"][f"{arm}_minus_RAW"]
        assert len(vals) == banked["participants"]
        assert np.isclose(np.mean(vals), banked["mean"], atol=1e-9)
    return out

dd_dev = deep_deltas(deep_dev)
dd_held = deep_deltas(deep_held)

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(figstyle.FULL, 5.4))
outer = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.30,
                         left=0.09, right=0.985, top=0.94, bottom=0.075)

def forest_pair(cell, key_contrast, tertile_key=None):
    """dev | held-out twin forest axes inside one outer cell."""
    gs = cell.subgridspec(1, 2, wspace=0.08)
    axL = fig.add_subplot(gs[0])
    axR = fig.add_subplot(gs[1], sharey=axL)
    n_rows = len(ARMS) + (1 if tertile_key else 0)
    ys = np.arange(n_rows)[::-1]
    vals = []
    for ax, d in ((axL, dev), (axR, held)):
        con = d[key_contrast]["vs_RAW"]
        for arm, y in zip(ARMS, ys):
            s = con[f"{arm}_minus_RAW"]
            ax.plot(s["mean"], y, "o", ms=3.4, color=C[arm], zorder=4)
            ax.plot(s["median"], y, "o", ms=3.4, mfc="none", mew=0.7,
                    color=C[arm], zorder=3)
            tr = blended_transform_factory(ax.transAxes, ax.transData)
            ax.text(0.99, y + 0.03, f"{s['positive_count']}/{s['participants']}",
                    transform=tr, ha="right", va="center", fontsize=4.6,
                    color="0.45", zorder=2)
            vals += [s["mean"], s["median"]]
        if tertile_key:  # tertile row sits at y=0 (bottom)
            for (tk, letter, dy) in (("low", "L", 0.33), ("mid", "M", -0.42),
                                     ("high", "H", 0.33)):
                m = d[tertile_key][tk]["mean"]
                ax.plot(m, 0, "^", ms=2.8, color=C["MATCH"], zorder=4)
                ax.text(m, 0 + dy, letter, ha="center",
                        va="bottom" if dy > 0 else "top",
                        fontsize=4.6, color="0.45")
                vals.append(m)
        ax.axvline(0, color=C["RAW"], lw=0.7, ls=(0, (2, 2)), zorder=1)
        ax.set_ylim(-0.7, n_rows - 0.3)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    lo, hi = min(vals), max(vals)
    rng_ = hi - lo
    for ax in (axL, axR):  # extra right margin keeps the k/n text off markers
        ax.set_xlim(min(lo - 0.12 * rng_, -0.12 * rng_),
                    max(hi + 0.34 * rng_, 0.34 * rng_))
    labels = [DISP[a] for a in ARMS] + (["MATCH tertile"] if tertile_key else [])
    axL.set_yticks(ys)
    axL.set_yticklabels(labels, fontsize=5.6)
    for tick, arm in zip(axL.get_yticklabels(), ARMS + (["MATCH"] if tertile_key else [])):
        tick.set_color(C[arm])
    plt.setp(axR.get_yticklabels(), visible=False)
    axR.tick_params(axis="y", length=0)
    return axL, axR

def headers(axL, axR, dL, dR, key, fmt=".3f"):
    axL.set_title(f"dev · RAW {dL[key]['RAW']:{fmt}}", fontsize=5.2,
                  color="0.4", pad=2)
    axR.set_title(f"held-out · RAW {dR[key]['RAW']:{fmt}}", fontsize=5.2,
                  color="0.4", pad=2)

# -- Panel A: SSVEP CCA accuracy (D1)
aL, aR = forest_pair(outer[0, 0], "d1_contrasts",
                     tertile_key="d1_match_minus_raw_by_contamination_tertile")
headers(aL, aR, dev, held, "d1_ssvep_accuracy")
aL.set_xlabel("$\\Delta$ SSVEP accuracy vs RAW")
aL.xaxis.set_label_coords(1.04, -0.14)
figstyle.panel(aL, "A")

# -- Panel B: ERP shrinkage-LDA AUC (D2)
bL, bR = forest_pair(outer[0, 1], "d2_contrasts")
headers(bL, bR, dev, held, "d2_erp_auc")
bL.set_xlabel("$\\Delta$ ERP AUC vs RAW (shrinkage LDA)")
bL.xaxis.set_label_coords(1.04, -0.14)
figstyle.panel(bL, "B")

# -- Panel C: deep decoder EEGNet-8,2 ERP AUC, per-participant dots
gs = outer[1, 0].subgridspec(1, 2, wspace=0.08)
cL = fig.add_subplot(gs[0])
cR = fig.add_subplot(gs[1], sharey=cL)
ys = np.arange(len(ARMS))[::-1]
rng = np.random.default_rng(7)
vals = []
for ax, dd in ((cL, dd_dev), (cR, dd_held)):
    for arm, y in zip(ARMS, ys):
        v = dd[arm]
        jit = rng.uniform(-0.22, 0.22, size=len(v))
        ax.plot(v, y + jit, "o", ms=1.9, color=C[arm], mec="none",
                alpha=0.55, zorder=3)
        ax.plot([v.mean()] * 2, [y - 0.30, y + 0.30], color=C[arm],
                lw=1.4, zorder=4)
        vals += [v.min(), v.max()]
    ax.axvline(0, color=C["RAW"], lw=0.7, ls=(0, (2, 2)), zorder=1)
    ax.set_ylim(-0.7, len(ARMS) - 0.3)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
lo, hi = min(vals), max(vals)
pad = 0.08 * (hi - lo)
for ax in (cL, cR):
    ax.set_xlim(lo - pad, hi + pad)
cL.set_yticks(ys)
cL.set_yticklabels([DISP[a] for a in ARMS], fontsize=5.6)
for tick, arm in zip(cL.get_yticklabels(), ARMS):
    tick.set_color(C[arm])
plt.setp(cR.get_yticklabels(), visible=False)
cR.tick_params(axis="y", length=0)
headers(cL, cR, {"m": deep_dev["means"]}, {"m": deep_held["means"]}, "m")
cL.set_xlabel("$\\Delta$ ERP AUC vs RAW (EEGNet-8,2)")
cL.xaxis.set_label_coords(1.04, -0.14)
figstyle.panel(cL, "C")

# -- Panel D: ERP preservation vs ERP-AUC delta (trade-off plane)
dx = fig.add_subplot(outer[1, 1])
dx.axhline(0, color=C["RAW"], lw=0.7, ls=(0, (2, 2)), zorder=1)
pts = {}
for arm in ARMS:
    p = (dev["erp_preservation"][arm],
         dev["d2_contrasts"]["vs_RAW"][f"{arm}_minus_RAW"]["mean"])
    q = (held["erp_preservation"][arm],
         held["d2_contrasts"]["vs_RAW"][f"{arm}_minus_RAW"]["mean"])
    pts[arm] = (p, q)
    dx.plot([p[0], q[0]], [p[1], q[1]], color=C[arm], lw=0.7, alpha=0.35,
            zorder=2)
    dx.plot(*p, "o", ms=3.6, mfc="none", mew=0.9, color=C[arm], zorder=4)
    dx.plot(*q, "o", ms=3.6, color=C[arm], mec="none", zorder=4)
dx.plot(dev["erp_preservation"]["RAW"], 0, "s", ms=3.2, color=C["RAW"],
        zorder=4)
dx.text(dev["erp_preservation"]["RAW"] - 0.006, 0.0035, "RAW", fontsize=5.5,
        color=C["RAW"], ha="right")

# direct labels (thin connectors into the dense cluster)
def lab(arm, xy_text, anchor, ha="left"):
    p, q = pts[arm]
    a = p if anchor == "dev" else q
    dx.annotate(DISP[arm], xy=a, xytext=xy_text, fontsize=5.5,
                color=C[arm], ha=ha, va="center",
                arrowprops=dict(arrowstyle="-", lw=0.5, color=C[arm],
                                alpha=0.6, shrinkA=1, shrinkB=2))
lab("ASR", (0.652, 0.0425), "held")
lab("SGEYESUB", (0.716, 0.0245), "held")
lab("ICA", (0.910, -0.0215), "held")
lab("NO_A0", (0.955, -0.0125), "dev", ha="right")
lab("MATCH", (0.885, 0.0065), "dev", ha="right")
lab("LINEAR", (0.905, 0.0140), "dev", ha="right")
lab("POP", (0.952, 0.0155), "dev", ha="left")

# open = dev, filled = held-out (tiny frameless key)
dx.plot([], [], "o", ms=3.6, mfc="none", mew=0.9, color="0.4", label="dev")
dx.plot([], [], "o", ms=3.6, color="0.4", mec="none", label="held-out")
dx.legend(loc="lower left", fontsize=5.5, handletextpad=0.3,
          borderaxespad=0.2, labelspacing=0.25)
dx.set_ylim(-0.027, 0.047)
dx.set_xlabel("ERP preservation (r)")
dx.set_ylabel("$\\Delta$ ERP AUC vs RAW")
figstyle.panel(dx, "D")

figstyle.save(fig, "fig-downstream")
