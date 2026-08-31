#!/usr/bin/env python3
"""fig-loop — Calibration economics and operator staleness (v3 main figure).

Panel A: matched-over-unguided rRMSE gain vs calibration duration, system
(reliability rule on) vs rule-off, gate-fire fractions annotated.
Panel B: operator displacement vs window start (top) and natural attenuation
gain by elapsed position (bottom) on a shared elapsed-time axis — two stacked
axes instead of a twin y-axis.
Panel C: paired gain by record third, y-scale shared with Panel A.

House rules: no CI/error bands anywhere — means + medians only (medians from
the banked JSON add the second per-group summary the bands would have hidden).
All numbers are read from the banked arrays/JSON at runtime.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/infres/yinwang/denoiseNet/scripts/paper_final/figures")
import figstyle

figstyle.setup()

import matplotlib.pyplot as plt

REPO = Path("/home/infres/yinwang/denoiseNet")
ARR = REPO / "paper_final_arrays"

# ---------------------------------------------------------------- data
t2 = np.load(ARR / "t2_duration_curve.npz", allow_pickle=True)
dur = np.asarray(t2["durations"], float)                  # [30 60 90 120]
g_sys = np.asarray(t2["gain_system"], float)
g_off = np.asarray(t2["gain_rule_off"], float)
gate = np.asarray(t2["hard_gate_fraction"], float)

t4 = np.load(ARR / "t4_staleness.npz", allow_pickle=True)
d_start = np.asarray(t4["displacement_start_s"], float)   # [0 .. 480]
d_mean = np.asarray(t4["displacement_mean"], float)
g_third = np.asarray(t4["gain_third"], float)

t4j = json.loads((REPO / "results/paper_final/t4_staleness.json").read_text())
dj = t4j["displacement_curve_by_window_start_s"]
d_med = np.array([dj[str(int(s))]["median"] for s in d_start])
assert np.allclose([dj[str(int(s))]["mean"] for s in d_start], d_mean)

tj = t4j["gain_by_record_third"]
third_med = np.array([tj[f"third_{i}"]["median"] for i in range(3)])
assert np.allclose([tj[f"third_{i}"]["mean"] for i in range(3)], g_third)
third_pos = [tj[f"third_{i}"]["positive_count"] for i in range(3)]
third_n = [tj[f"third_{i}"]["participants"] for i in range(3)]

nj = t4j["natural_gain_by_elapsed_time"]
pos_keys = sorted(nj, key=lambda k: int(k.split("_")[1]))
n_x = np.array([nj[k]["mean_start_s"] for k in pos_keys])
n_mean = np.array([nj[k]["attenuation_gain_db"]["mean"] for k in pos_keys])
n_med = np.array([nj[k]["attenuation_gain_db"]["median"] for k in pos_keys])

C = figstyle.C
GRAY = "#7f7f7f"
INK = "#444444"

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(figstyle.FULL, 1.95))
outer = fig.add_gridspec(1, 3, width_ratios=[1.30, 1.35, 0.88],
                         wspace=0.42, left=0.065, right=0.99,
                         top=0.88, bottom=0.225)

# ============================ Panel A ================================
axA = fig.add_subplot(outer[0])
figstyle.panel(axA, "A")
axA.axhline(0.0, color=C["NO_A0"], lw=0.6, ls=":")
axA.text(dur[-1] + 3, 0.0, "unguided", color=C["NO_A0"], fontsize=5.5,
         va="center", ha="left")
axA.plot(dur, g_off, color=C["MATCH"], ls="--", lw=0.9, marker="o",
         ms=2.6, mfc="white", mew=0.8, alpha=0.75)
axA.plot(dur, g_sys, color=C["MATCH"], ls="-", marker="o", ms=3.0)
axA.text(dur[-1] + 3, g_sys[-1], "rule on", color=C["MATCH"], fontsize=5.5,
         va="center", ha="left")
axA.text(dur[-1] + 3, g_off[-1] - 0.012, "rule off", color=C["MATCH"],
         fontsize=5.5, va="top", ha="left", alpha=0.75)
# gate-fire fractions (designed refusal at 30 s)
axA.annotate(f"gate {gate[0]*100:.0f}%\ndesigned refusal",
             xy=(dur[0], g_sys[0]), xytext=(dur[0] + 6, g_sys[0] + 0.052),
             fontsize=5.5, color=INK, ha="left", va="bottom",
             arrowprops=dict(arrowstyle="-", lw=0.5, color=INK))
axA.text(np.mean(dur[1:]), -0.033,
         f"gate {gate[1:].min()*100:.1f}–{gate[1:].max()*100:.1f}%",
         fontsize=5.5, color=GRAY, ha="center", va="bottom")
# 60 s buys ~97% of the 120-s gain
axA.text(dur[1] + 3, g_sys[1] - 0.042, f"{g_sys[1]/g_sys[3]*100:.0f}% of 120 s",
         fontsize=5.5, color=C["MATCH"], ha="left", va="top")
axA.set_xticks(dur)
axA.set_xlim(20, 152)
axA.set_ylim(-0.048, 0.185)
axA.set_xlabel("calibration duration (s)")
axA.set_ylabel("rRMSE gain")

# ============================ Panel B ================================
gsB = outer[1].subgridspec(2, 1, hspace=0.16)
axB1 = fig.add_subplot(gsB[0])
axB2 = fig.add_subplot(gsB[1], sharex=axB1)
figstyle.panel(axB1, "B")

# top: operator displacement vs window start
axB1.axhline(1.0, color=GRAY, lw=0.6, ls=":")
axB1.text(-8, 1.06, r"$\|A_0\|$", color=GRAY, fontsize=5.5, va="bottom",
          ha="left")
axB1.plot(d_start, d_med, color=C["MATCH"], ls="--", lw=0.9, marker="o",
          ms=2.4, mfc="white", mew=0.8, alpha=0.75)
axB1.plot(d_start, d_mean, color=C["MATCH"], ls="-", marker="o", ms=2.8)
axB1.text(d_start[-1] + 12, d_mean[-1], "mean", color=C["MATCH"],
          fontsize=5.5, va="center", ha="left")
axB1.text(d_start[-1] + 12, d_med[-1], "median", color=C["MATCH"],
          fontsize=5.5, va="center", ha="left", alpha=0.75)
axB1.annotate(f"{d_mean[1]*100:.0f}%", xy=(d_start[1], d_mean[1]),
              xytext=(d_start[1] - 8, d_mean[1] + 0.28), fontsize=5.5,
              color=INK, ha="center")
axB1.annotate(f"{d_mean[-1]*100:.0f}%", xy=(d_start[-1], d_mean[-1]),
              xytext=(d_start[-1] - 52, d_mean[-1] + 0.30), fontsize=5.5,
              color=INK, ha="center")
axB1.set_ylabel(r"drift$\,/\,\|A_0\|$")
axB1.set_ylim(-0.12, 1.95)
plt.setp(axB1.get_xticklabels(), visible=False)

# bottom: natural attenuation gain by elapsed position
axB2.plot(n_x, n_med, color=GRAY, ls="--", lw=0.9, marker="s", ms=2.4,
          mfc="white", mew=0.8, alpha=0.85)
axB2.plot(n_x, n_mean, color=GRAY, ls="-", marker="s", ms=2.8)
axB2.text(n_x[0] - 18, n_mean[0], "natural attn.", color=GRAY, fontsize=5.5,
          va="center", ha="right")
axB2.set_ylabel("dB")
axB2.set_ylim(0.9, 3.1)
axB2.set_xlim(-25, 660)
axB2.set_xlabel("elapsed time (s)")

# ============================ Panel C ================================
axC = fig.add_subplot(outer[2])
figstyle.panel(axC, "C")
xs3 = np.arange(3)
axC.axhline(0.0, color=C["NO_A0"], lw=0.6, ls=":")
axC.plot(xs3, third_med, color=C["MATCH"], ls="--", lw=0.9, marker="o",
         ms=2.4, mfc="white", mew=0.8, alpha=0.75)
axC.plot(xs3, g_third, color=C["MATCH"], ls="-", marker="o", ms=2.8)
axC.text(xs3[-1] + 0.12, g_third[-1], "mean", color=C["MATCH"], fontsize=5.5,
         va="bottom", ha="left")
axC.text(xs3[-1] + 0.12, third_med[-1], "median", color=C["MATCH"],
         fontsize=5.5, va="top", ha="left", alpha=0.75)
assert len(set(third_pos)) == 1 and len(set(third_n)) == 1
axC.text(1.0, 0.028, f"{third_pos[0]}/{third_n[0]} subj. > 0",
         fontsize=5.5, color=GRAY, ha="center", va="bottom")
axC.set_xticks(xs3)
axC.set_xticklabels(["1st", "2nd", "3rd"])
axC.set_xlim(-0.35, 2.85)
axC.set_ylim(-0.048, 0.185)  # shared with Panel A: flat-to-rising, not decay
axC.set_xlabel("record third")
axC.set_ylabel("rRMSE gain")

figstyle.save(fig, "fig-loop")
