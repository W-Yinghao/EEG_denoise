#!/usr/bin/env python3
"""fig-loop — Calibration economics and operator staleness (v3 main figure).

Panel A: matched-over-unguided rRMSE gain vs calibration duration, system
(reliability rule on) vs rule-off, gate-fire fractions annotated.
Panel B: operator displacement vs window start (top) and natural attenuation
gain by elapsed position (bottom) on a shared elapsed-time axis — two stacked
axes instead of a twin y-axis.
Panel C: paired gain by record third, y-scale shared with Panel A.
Panel D: reliability gate on the 4 abstained cells (g5 per-cell rows) — top:
per-cell subtraction-harm avoided (Δ rRMSE, subtract − fallback) paired across
the linear and diffusion classes; bottom: retention/attenuation trade as
per-cell dumbbells (subtract → fallback), flat means bold (0.917→0.984,
0.41→0.13 dB). n = 4 cells is the honest content.

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

# g5 reliability-gate per-cell rows (4 abstained cells)
g5 = json.loads(
    (REPO / "results/paper_final/gapfill/g5_gate_cells.json").read_text())
pc = g5["pooled_check"]
lin = {r["cell"]: r["diff_BINARY_minus_BINARY_NOA0FB"]
       for r in g5["rows"] if r["panel"] == "linear_class_fallback"}
dif = {r["cell"]: r["diff_MATCH_gated_minus_MATCH_NOA0FB"]
       for r in g5["rows"] if r["panel"] == "diffusion_class_fallback"}
cells = sorted(lin)
assert cells == sorted(dif) and len(cells) == 4
d_lin = np.array([lin[c] for c in cells])
d_dif = np.array([dif[c] for c in cells])
assert np.allclose(np.round(d_lin, 4), pc["p1r_per_cell_diffs_4dp"]["quoted"],
                   atol=5e-5)
assert np.allclose(np.round(d_dif, 4),
                   pc["f1_primary_per_cell_diffs_4dp"]["quoted"], atol=5e-5)
assert abs(d_lin.mean()
           - pc["p1r_paired_mean_BINARY_minus_BINARY_NOA0FB"]["quoted"]) < 1e-12
assert abs(d_dif.mean()
           - pc["f1_primary_paired_mean_MATCHgated_minus_MATCHNOA0FB"]["quoted"]
           ) < 1e-12

nat = {(r["cell"], r["condition"]): r
       for r in g5["rows"] if r["panel"] == "natural_hard_cells"}
ret_g = np.array([nat[(c, "MATCH_gated")]["low_eog_observation_retention"]
                  for c in cells])
ret_f = np.array([nat[(c, "MATCH_NOA0FB")]["low_eog_observation_retention"]
                  for c in cells])
att_g = np.array([nat[(c, "MATCH_gated")]["attenuation_db"] for c in cells])
att_f = np.array([nat[(c, "MATCH_NOA0FB")]["attenuation_db"] for c in cells])
assert abs(ret_g.mean() - pc["natural_retention_MATCH_gated_flat"]["quoted"]) \
    < 1e-12
assert abs(ret_f.mean() - pc["natural_retention_MATCH_NOA0FB_flat"]["quoted"]) \
    < 1e-12
assert abs(att_g.mean()
           - pc["natural_attenuation_MATCH_gated_flat"]["quoted"]) < 1e-12
assert abs(att_f.mean()
           - pc["natural_attenuation_MATCH_NOA0FB_flat"]["quoted"]) < 1e-12
assert [round(ret_g.mean(), 3), round(ret_f.mean(), 3)] \
    == pc["headline_retention_0.917_to_0.984"]["quoted"]
assert [round(att_g.mean(), 2), round(att_f.mean(), 2)] \
    == pc["headline_attenuation_0.41_to_0.13_dB"]["quoted"]

C = figstyle.C
GRAY = "#7f7f7f"
INK = "#444444"

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(figstyle.FULL, 1.95))
outer = fig.add_gridspec(1, 4, width_ratios=[1.30, 1.35, 0.88, 0.98],
                         wspace=0.50, left=0.058, right=0.995,
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

# ============================ Panel D ================================
# Reliability gate on the 4 abstained cells: harm avoided + the price paid.
gsD = outer[3].subgridspec(2, 1, hspace=0.85)
axD1 = fig.add_subplot(gsD[0])
axD2 = fig.add_subplot(gsD[1])
figstyle.panel(axD1, "D")

# top: per-cell subtraction-harm avoided, paired across the two model classes
xsD = np.array([0.0, 1.0])
axD1.axhline(0.0, color=GRAY, lw=0.6, ls=":")
for dl, dd in zip(d_lin, d_dif):
    axD1.plot(xsD, [dl, dd], color=GRAY, lw=0.5, alpha=0.55)
axD1.plot(np.zeros(4), d_lin, "o", ls="none", color=C["LINEAR"], ms=2.8)
axD1.plot(np.ones(4), d_dif, "o", ls="none", color=C["MATCH"], ms=2.8)
axD1.plot([0], [d_lin.mean()], marker="_", ms=8, mew=1.2, color=C["LINEAR"])
axD1.plot([1], [d_dif.mean()], marker="_", ms=8, mew=1.2, color=C["MATCH"])
axD1.text(0.13, d_lin.mean(), f"+{d_lin.mean():.2f}", color=C["LINEAR"],
          fontsize=5.5, va="center", ha="left")
axD1.text(1.13, d_dif.mean(), f"+{d_dif.mean():.2f}", color=C["MATCH"],
          fontsize=5.5, va="center", ha="left")
axD1.text(0.5, 0.56, "harm avoided, 4 cells", color=GRAY, fontsize=5.5,
          ha="center", va="bottom")
axD1.set_xticks(xsD)
axD1.set_xticklabels(["linear", "diffusion"])
for tick, col in zip(axD1.get_xticklabels(), [C["LINEAR"], C["MATCH"]]):
    tick.set_color(col)
axD1.set_xlim(-0.42, 1.68)
axD1.set_ylim(-0.115, 0.62)
axD1.set_yticks([0.0, 0.25, 0.5])
axD1.set_ylabel("$\\Delta$rRMSE")

# bottom: retention/attenuation trade, subtract (filled) -> fallback (open)
for i in range(4):
    axD2.plot([ret_g[i], ret_f[i]], [att_g[i], att_f[i]], color=GRAY,
              lw=0.5, alpha=0.55)
axD2.plot(ret_g, att_g, "o", ls="none", color=C["MATCH"], ms=2.8)
axD2.plot(ret_f, att_f, "o", ls="none", color=C["MATCH"], ms=2.8,
          mfc="white", mew=0.8)
axD2.plot([ret_g.mean(), ret_f.mean()], [att_g.mean(), att_f.mean()],
          color=C["MATCH"], lw=1.3)
axD2.plot([ret_g.mean()], [att_g.mean()], "o", ms=4.2, color=C["MATCH"])
axD2.plot([ret_f.mean()], [att_f.mean()], "o", ms=4.2, color=C["MATCH"],
          mfc="white", mew=1.1)
axD2.text(0.748, 0.98, "● subtract", color=C["MATCH"], fontsize=5.5,
          ha="left", va="center")
axD2.text(0.748, 0.78, "○ fallback", color=C["MATCH"], fontsize=5.5,
          ha="left", va="center", alpha=0.75)
axD2.text(0.902, att_g.mean() - 0.01, f"{att_g.mean():.2f} dB",
          color=C["MATCH"], fontsize=5.5, ha="right", va="center")
axD2.annotate(f"{att_f.mean():.2f} dB", xy=(0.979, 0.105),
              xytext=(0.952, 0.005), fontsize=5.5, color=C["MATCH"],
              ha="right", va="center", alpha=0.75,
              arrowprops=dict(arrowstyle="-", lw=0.5, color=C["MATCH"],
                              alpha=0.5))
axD2.set_xlim(0.735, 1.015)
axD2.set_xticks([0.8, 0.9, 1.0])
axD2.set_ylim(-0.12, 1.12)
axD2.set_yticks([0.0, 0.5, 1.0])
axD2.set_xlabel("low-EOG retention")
axD2.set_ylabel("attn. (dB)")

figstyle.save(fig, "fig-loop")
