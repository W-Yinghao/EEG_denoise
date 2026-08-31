#!/usr/bin/env python3
"""fig-natural — Natural behavior and owner specificity (v3 main figure).

Panel A: operating-point plane, low-EOG retention (x) vs EOG attenuation dB
(y); six diffusion-family conditions as filled markers (t5_natural_plane.npz),
literature anchors ICA / ASR / SGEYESUB as open markers (cpu_reference_rows
report JSON); vertical 0.75 retention validity bar. No error bars (house
rule 2) — points only.
Panel B: paired contrasts MATCH_gated minus each control, three metric
columns (Δattenuation, Δretention, Δcoherence); filled dot = mean, open dot
= median, gray k/15 = per-participant positive count (15 dev participants).
Replaces the planned bootstrap whiskers per house rule 2.
Panel C: coherence reduction, horizontal bars for the six conditions plus
the three CPU anchors (open bars). No CI arrays exist for coherence; none
are drawn anywhere per house rule 2.

All numbers are read from the banked arrays/JSONs at runtime.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/infres/yinwang/denoiseNet/scripts/paper_final/figures")
import figstyle

figstyle.setup()

import matplotlib.pyplot as plt
from matplotlib import gridspec

REPO = Path("/home/infres/yinwang/denoiseNet")

# ---------------------------------------------------------------- data
npz = np.load(REPO / "paper_final_arrays/t5_natural_plane.npz", allow_pickle=True)
conds = [str(c) for c in npz["conditions"]]
plane = {c: dict(att=float(npz["attenuation_db"][i]),
                 ret=float(npz["retention"][i]),
                 coh=float(npz["coherence_reduction"][i]))
         for i, c in enumerate(conds)}

ref_npz = np.load(REPO / "paper_final_arrays/cpu_reference_rows.npz",
                  allow_pickle=True)
report = json.loads(str(ref_npz["report"][()]))
anchors = {}
for name in ("ica", "asr", "sgeyesub"):
    nat = report[name]["natural"]
    anchors[name.upper()] = dict(
        att=float(nat["attenuation_db"]["mean"]),
        ret=float(nat["low_eog_observation_retention"]["mean"]),
        coh=float(nat["coherence_reduction"]["mean"]))

tj = json.loads((REPO / "results/paper_final/t5_natural_plane.json").read_text())
contrasts = tj["contrasts"]

RET_BAR = 0.75  # validity bar from the pre-registered plan

# ---------------------------------------------------------------- layout
fig = plt.figure(figsize=(figstyle.FULL, 2.05))
gs = gridspec.GridSpec(1, 5, width_ratios=[2.35, 0.85, 0.85, 0.85, 1.30],
                       wspace=0.55, left=0.065, right=0.985,
                       bottom=0.21, top=0.90)
axA = fig.add_subplot(gs[0])
axB = [fig.add_subplot(gs[i]) for i in (1, 2, 3)]
axC = fig.add_subplot(gs[4])

# ---------------------------------------------------------------- panel A
MARK = {"MATCH_gated": "o", "NO_A0": "s", "POP": "D",
        "WRONG_gated": "v", "WRONG": "^", "SHUFFLED": "P"}
AMARK = {"ICA": "o", "ASR": "s", "SGEYESUB": "D"}

axA.axhline(0.0, color="0.75", lw=0.5, zorder=0)
axA.axvline(RET_BAR, color="0.45", lw=0.7, ls=(0, (3, 2)), zorder=0)
axA.text(RET_BAR, -1.95, f"{RET_BAR:.2f}", fontsize=5, color="0.45",
         ha="center", va="bottom")
# healthy region: right of the validity bar, above 0 dB
axA.fill_between([RET_BAR, 1.0], 0.0, 5.6, color="#D55E00", alpha=0.045,
                 lw=0, zorder=0)

for c in conds:
    axA.plot(plane[c]["ret"], plane[c]["att"], MARK[c], ms=4.5,
             color=figstyle.C[c], mec="white", mew=0.4, zorder=3)
for a, v in anchors.items():
    axA.plot(v["ret"], v["att"], AMARK[a], ms=4.5, mfc="none",
             mec=figstyle.C[a], mew=1.0, zorder=3)

LBL = {  # x, y, ha, text
    "MATCH_gated": (0.856, 2.30, "left", "MATCH gated"),
    "NO_A0": (0.918, 0.62, "center", "NO_A0"),
    "POP": (0.757, 1.28, "right", "POP"),
    "WRONG_gated": (0.758, 0.57, "right", "WRONG gated"),
    "WRONG": (0.352, -1.55, "left", "WRONG"),
    "SHUFFLED": (0.383, -0.42, "left", "SHUFFLED"),
    "ICA": (0.712, 3.22, "center", "ICA"),
    "ASR": (0.610, 2.40, "center", "ASR"),
    "SGEYESUB": (0.513, 4.80, "left", "SGEYESUB"),
}
for key, (x, y, ha, txt) in LBL.items():
    axA.text(x, y, txt, fontsize=5.5, color=figstyle.C[key],
             ha=ha, va="center")

axA.set_xlim(0.28, 1.0)
axA.set_ylim(-2.1, 5.6)
axA.set_xlabel("low-EOG retention")
axA.set_ylabel("EOG attenuation (dB)")
figstyle.panel(axA, "A")

# ---------------------------------------------------------------- panel B
ORDER = ["NO_A0", "POP", "WRONG_gated", "WRONG", "SHUFFLED"]
ROWTXT = {"NO_A0": "NO_A0", "POP": "POP", "WRONG_gated": "WRONG g.",
          "WRONG": "WRONG", "SHUFFLED": "SHUFFLED"}
METRICS = [("attenuation_db", "$\\Delta$atten. (dB)"),
           ("low_eog_observation_retention", "$\\Delta$retention"),
           ("coherence_reduction", "$\\Delta$coherence")]
ys = np.arange(len(ORDER))[::-1]

for j, (met, xlab) in enumerate(METRICS):
    ax = axB[j]
    ax.axvline(0.0, color="0.75", lw=0.5, zorder=0)
    for y, cond in zip(ys, ORDER):
        s = contrasts[f"MATCH_gated_minus_{cond}"][met]
        col = figstyle.C[cond]
        ax.plot([0, s["mean"]], [y, y], color=col, lw=0.6, alpha=0.45,
                zorder=2)
        ax.plot(s["median"], y, "o", ms=3.2, mfc="none", mec=col, mew=0.8,
                zorder=3)
        ax.plot(s["mean"], y, "o", ms=3.6, color=col, mec="white", mew=0.3,
                zorder=4)
        ax.text(1.0, y, f"{s['positive_count']}/{s['participants']}",
                transform=ax.get_yaxis_transform(), fontsize=4.6,
                color="0.35", ha="right", va="center")
    ax.set_ylim(-0.6, len(ORDER) - 0.4)
    ax.set_yticks(ys)
    if j == 0:
        ax.set_yticklabels([ROWTXT[c] for c in ORDER], fontsize=5.5)
        for tick, cond in zip(ax.get_yticklabels(), ORDER):
            tick.set_color(figstyle.C[cond])
        ax.text(-0.02, 1.06, "MATCH gated $-$ control", fontsize=5.5,
                color="0.35", transform=ax.transAxes, ha="left")
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
    ax.set_xlabel(xlab)

# metric-specific x ranges, padded right so the k/15 column stays clear
axB[0].set_xlim(-0.40, 6.2)
axB[0].set_xticks([0, 2, 4])
axB[1].set_xlim(-0.22, 1.00)
axB[1].set_xticks([0, 0.5])
axB[2].set_xlim(-0.02, 0.33)
axB[2].set_xticks([0, 0.1, 0.2])
figstyle.panel(axB[0], "B")

# ---------------------------------------------------------------- panel C
C_ORDER = ["MATCH_gated", "NO_A0", "POP", "WRONG_gated", "WRONG", "SHUFFLED"]
CTXT = {"MATCH_gated": "MATCH g.", "NO_A0": "NO_A0", "POP": "POP",
        "WRONG_gated": "WRONG g.", "WRONG": "WRONG", "SHUFFLED": "SHUFFLED"}
A_ORDER = ["ICA", "ASR", "SGEYESUB"]

yc, labels, lab_cols = [], [], []
for i, c in enumerate(C_ORDER):
    y = len(C_ORDER) + len(A_ORDER) - i + 0.6  # top block
    axC.barh(y, plane[c]["coh"], height=0.72, color=figstyle.C[c], lw=0)
    yc.append(y); labels.append(CTXT[c]); lab_cols.append(figstyle.C[c])
for i, a in enumerate(A_ORDER):
    y = len(A_ORDER) - i
    axC.barh(y, anchors[a]["coh"], height=0.72, facecolor="none",
             edgecolor=figstyle.C[a], lw=0.9)
    yc.append(y); labels.append(a); lab_cols.append(figstyle.C[a])

axC.axvline(0.0, color="0.45", lw=0.6)
axC.set_yticks(yc)
axC.set_yticklabels(labels, fontsize=5.5)
for tick, col in zip(axC.get_yticklabels(), lab_cols):
    tick.set_color(col)
axC.tick_params(axis="y", length=0)
axC.set_xlim(0, 0.245)
axC.set_ylim(0.3, len(C_ORDER) + len(A_ORDER) + 1.4)
axC.set_xlabel("coherence reduction")
axC.spines["left"].set_visible(False)
figstyle.panel(axC, "C")

figstyle.save(fig, "fig-natural")
