#!/usr/bin/env python3
"""fig-natural — Natural behavior and owner specificity (v3 main figure).

Panel A: operating-point plane, low-EOG retention (x) vs EOG attenuation dB
(y); six diffusion-family conditions as filled markers (t5_natural_plane.npz),
literature anchors ICA / ASR / SGEYESUB as open markers (cpu_reference_rows
report JSON); vertical 0.75 retention validity bar. No error bars (house
rule 2) — points only.
Panel B: paired contrasts MATCH_gated minus each control, three metric
columns (Δattenuation, Δretention, Δcoherence). Per-participant dots (15 dev
participants, g3_natural_units.json rows, jitter fixed per participant across
subpanels), filled dot = grand mean, gray k/15 = per-participant positive
count. The per-participant rows are asserted at runtime to reproduce the
banked pooled stats (t5_natural_plane.json) before anything is drawn.
Panel C: coherence reduction, horizontal bars for the six conditions plus
the three CPU anchors (open bars). No CI arrays exist for coherence; none
are drawn anywhere per house rule 2.
Panel D: warm-RLS on-line adaptation (g2_rls_curve.json, V44-S2 OR-1):
per-participant thin lines + grand mean of rrmse_temporal over adaptation
time 10/30/60/120/240 s (log-log); dashed reference = static calibrated
diffusion MATCH_gated. The 240 s upturn and the heavy-tail participant
sub-09 are shown as-is.
Panel E: ownership vs estimation quality (g6_ownership.json, V44-S1):
per-participant rrmse_temporal of MATCH_gated and gate-passing WRONG_gated
against the POP anchor (log-log, y=x diagonal); vertical connectors pair the
two conditions within a participant. WRONG_gated sits above the diagonal for
9/15 participants (pooled +0.038) although it passes the reliability gate —
the motivation for the V44-S2 ownership guard.

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
GAP = REPO / "results/paper_final/gapfill"

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

# per-participant natural-endpoint contrasts (g3) — verified against banked
g3 = json.loads((GAP / "g3_natural_units.json").read_text())
g3_rows = sorted(g3["rows"], key=lambda r: r["participant"])
G3_PARTS = [r["participant"] for r in g3_rows]
for _c in ("NO_A0", "POP", "WRONG_gated", "WRONG", "SHUFFLED"):
    for _m in ("attenuation_db", "low_eog_observation_retention",
               "coherence_reduction"):
        _v = np.array([r["match_gated_minus"][_c][_m] for r in g3_rows])
        _b = contrasts[f"MATCH_gated_minus_{_c}"][_m]
        assert abs(_v.mean() - _b["mean"]) < 1e-9, (_c, _m)
        assert int((_v > 0).sum()) == _b["positive_count"], (_c, _m)

# warm-RLS adaptation curve (g2) — verified against pooled_check
g2 = json.loads((GAP / "g2_rls_curve.json").read_text())
RLS_T = [10, 30, 60, 120, 240]
rls = {}
for r in g2["rows"]:
    rls.setdefault(r["participant"], {})[r["adaptation_time_s"]] = \
        r["mean_rrmse_temporal"]
RLS_PARTS = sorted(rls)
rls_mean = np.array([np.mean([rls[p][t] for p in RLS_PARTS]) for t in RLS_T])
for t, m in zip(RLS_T, rls_mean):
    q = g2["pooled_check"][f"RLS_warm_{t}s"]["quoted_full_precision"]
    assert abs(m - q) < 1e-9, (t, m, q)
RLS_STATIC = float(g2["static_reference"]["static_calibrated_MATCH_gated"])

# ownership vs estimation quality (g6) — verified against pooled_check
g6 = json.loads((GAP / "g6_ownership.json").read_text())
g6_rows = sorted(g6["rows"], key=lambda r: r["participant"])
own_pop = np.array([r["rrmse_temporal_POP"] for r in g6_rows])
own_mg = np.array([r["rrmse_temporal_MATCH_gated"] for r in g6_rows])
own_wg = np.array([r["rrmse_temporal_WRONG_gated"] for r in g6_rows])
_q = g6["pooled_check"]["quoted"]
assert abs((own_wg - own_pop).mean() - _q["mean"]) < 1e-12
assert int(((own_wg - own_pop) > 0).sum()) == _q["positive_count"]
n_wg_above = int((own_wg > own_pop).sum())
n_mg_above = int((own_mg > own_pop).sum())

RET_BAR = 0.75  # validity bar from the pre-registered plan

# ---------------------------------------------------------------- layout
fig = plt.figure(figsize=(figstyle.FULL, 3.95))
gs = gridspec.GridSpec(2, 5, width_ratios=[2.35, 0.85, 0.85, 0.85, 1.30],
                       height_ratios=[1.0, 0.92],
                       wspace=0.55, hspace=0.62, left=0.065, right=0.985,
                       bottom=0.105, top=0.945)
axA = fig.add_subplot(gs[0, 0])
axB = [fig.add_subplot(gs[0, i]) for i in (1, 2, 3)]
axC = fig.add_subplot(gs[0, 4])
axD = fig.add_subplot(gs[1, 0:3])
axE = fig.add_subplot(gs[1, 3:5])

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
# fixed vertical jitter per participant, identical across rows and metrics
JIT = np.linspace(-0.24, 0.24, len(G3_PARTS))

for j, (met, xlab) in enumerate(METRICS):
    ax = axB[j]
    ax.axvline(0.0, color="0.75", lw=0.5, zorder=0)
    for y, cond in zip(ys, ORDER):
        s = contrasts[f"MATCH_gated_minus_{cond}"][met]
        col = figstyle.C[cond]
        pts = np.array([r["match_gated_minus"][cond][met] for r in g3_rows])
        ax.plot([0, s["mean"]], [y, y], color=col, lw=0.6, alpha=0.45,
                zorder=2)
        ax.plot(pts, y + JIT, "o", ms=1.7, color=col, alpha=0.45,
                mec="none", zorder=3)
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
axB[0].set_xlim(-0.7, 9.4)
axB[0].set_xticks([0, 4, 8])
axB[1].set_xlim(-0.26, 1.14)
axB[1].set_xticks([0, 0.5, 1])
axB[2].set_xlim(-0.05, 0.42)
axB[2].set_xticks([0, 0.2, 0.4])
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

# ---------------------------------------------------------------- panel D
axD.set_xscale("log")
axD.set_yscale("log")
for p in RLS_PARTS:
    v = [rls[p][t] for t in RLS_T]
    axD.plot(RLS_T, v, color="0.72", lw=0.5, alpha=0.9, zorder=2)
# heavy-tail participant, named honestly at the end of its line
axD.text(RLS_T[-1] * 1.06, rls["sub-09"][RLS_T[-1]], "sub-09", fontsize=4.8,
         color="0.45", ha="left", va="center")
axD.plot(RLS_T, rls_mean, "o-", color=figstyle.C["LINEAR"], lw=1.4, ms=2.8,
         mec="white", mew=0.3, zorder=4)
axD.text(RLS_T[-1] * 1.06, rls_mean[-1], "RLS warm\nmean", fontsize=5,
         color=figstyle.C["LINEAR"], ha="left", va="center", linespacing=1.1)
axD.axhline(RLS_STATIC, color=figstyle.C["MATCH_gated"], lw=0.8,
            ls=(0, (3, 2)), zorder=3)
axD.text(RLS_T[0] * 0.97, RLS_STATIC * 0.86,
         f"static matched {RLS_STATIC:.3f}", fontsize=5,
         color=figstyle.C["MATCH_gated"], ha="left", va="top")
axD.set_xlim(8.6, 430)
axD.set_ylim(0.045, 13)
axD.set_xticks(RLS_T)
axD.set_xticklabels([str(t) for t in RLS_T])
axD.set_yticks([0.1, 0.3, 1, 3, 10])
axD.set_yticklabels(["0.1", "0.3", "1", "3", "10"])
axD.minorticks_off()
axD.set_xlabel("adaptation time (s)")
axD.set_ylabel("rRMSE$_t$")
figstyle.panel(axD, "D")

# ---------------------------------------------------------------- panel E
LIMS = (0.045, 6.5)
axE.set_xscale("log")
axE.set_yscale("log")
axE.plot(LIMS, LIMS, color="0.75", lw=0.6, ls=(0, (3, 2)), zorder=1)
for x, ym, yw in zip(own_pop, own_mg, own_wg):
    axE.plot([x, x], [ym, yw], color="0.82", lw=0.5, zorder=2)
axE.plot(own_pop, own_wg, "v", ms=2.8, color=figstyle.C["WRONG_gated"],
         mec="white", mew=0.25, zorder=4)
axE.plot(own_pop, own_mg, "o", ms=2.8, color=figstyle.C["MATCH_gated"],
         mec="white", mew=0.25, zorder=4)
axE.text(0.04, 0.96, "above diag. = worse than POP", fontsize=4.8,
         color="0.45", transform=axE.transAxes, ha="left", va="top")
axE.text(0.04, 0.86, f"WRONG g. {n_wg_above}/{len(own_pop)} above",
         fontsize=5, color=figstyle.C["WRONG_gated"],
         transform=axE.transAxes, ha="left", va="top")
axE.text(0.04, 0.76, f"MATCH g. {n_mg_above}/{len(own_pop)} above",
         fontsize=5, color=figstyle.C["MATCH_gated"],
         transform=axE.transAxes, ha="left", va="top")
axE.set_xlim(0.10, LIMS[1])
axE.set_ylim(LIMS)
axE.set_xticks([0.1, 0.3, 1, 3])
axE.set_xticklabels(["0.1", "0.3", "1", "3"])
axE.set_yticks([0.1, 0.3, 1, 3])
axE.set_yticklabels(["0.1", "0.3", "1", "3"])
axE.minorticks_off()
axE.set_xlabel("POP rRMSE$_t$")
axE.set_ylabel("operator rRMSE$_t$")
figstyle.panel(axE, "E")

figstyle.save(fig, "fig-natural")
