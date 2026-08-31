"""fig-overview: Method overview (three-stage pipeline), Sec 3, figure* 46mm.

Left-to-right schematic. Stage A calibration (timeline, ridge+EB shrinkage,
reliability branch to fallback), Stage B guided diffusion restoration
(guide a_i, shared U-Net in DDIM loop, Eq. 4 decomposition), Stage C
uncertainty (K sampled propagation matrices -> 80% predictive band).
All waveform snippets are real exemplar arrays (sub-02 ses-02 SSVEP, Fp1)
loaded at runtime from t6_waveform_exemplar_dev.npz. No performance numbers.
"""
import sys

sys.path.insert(0, "/home/infres/yinwang/denoiseNet/scripts/paper_final/figures")
import figstyle

figstyle.setup()

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

NPZ = "/home/infres/yinwang/denoiseNet/paper_final_arrays/t6_waveform_exemplar_dev.npz"

d = np.load(NPZ, allow_pickle=True)
ch = list(d["eeg_names"]).index("Fp1")
eog = np.asarray(d["eog_drive"], float)            # (2, 512) real EOG drive
y_obs = np.asarray(d["contaminated"], float)[ch]   # contaminated EEG, Fp1
x_hat = np.asarray(d["matched"], float)[ch]        # restored (matched arm)
b_mean = np.asarray(d["band_mean"], float)[ch]     # trajectory mean
b_hw = np.asarray(d["band_halfwidth_80"], float)[ch]  # 80% PI halfwidth
t = np.arange(y_obs.size)

ORANGE = figstyle.C["matched"]      # calibration-derived quantities / ours
BLUE = figstyle.C["population"]     # shared population components
GRAY = figstyle.C["RAW"]            # fallback path / raw observation
PINK = figstyle.C["eog"]            # EOG traces
DARK = figstyle.C["reference"]

fig = plt.figure(figsize=(figstyle.FULL, 1.85))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_axis_off()
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# ---------- helpers -------------------------------------------------------
def wax(rect):
    a = fig.add_axes(rect)
    a.set_axis_off()
    return a

def box(x0, y0, w, h, fc, ec, lw=0.7, ls="-"):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, fc=fc, ec=ec, lw=lw, ls=ls,
                                boxstyle="round,pad=0.002,rounding_size=0.006",
                                mutation_aspect=7.0 / 1.85))

def arrow(p0, p1, color, lw=0.8, ls="-", rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", color=color, lw=lw,
                                 linestyle=ls, shrinkA=0, shrinkB=0,
                                 mutation_scale=6,
                                 connectionstyle=f"arc3,rad={rad}"))

def txt(x, y, s, color, fs=6, ha="center", va="center", w=None):
    ax.text(x, y, s, color=color, fontsize=fs, ha=ha, va=va,
            fontweight=w if w else "normal")

# ---------- stage separators + panel tags ---------------------------------
for xs in (0.330, 0.685):
    ax.plot([xs, xs], [0.05, 0.95], ls=":", color="#cccccc", lw=0.7)
    txt(xs, 0.50, "›", "#bbbbbb", fs=9)

for x0, wdt, tag in ((0.038, 0.28, "A"), (0.368, 0.29, "B"), (0.718, 0.26, "C")):
    a = fig.add_axes([x0, 0.04, wdt, 0.84])
    a.set_axis_off()
    figstyle.panel(a, tag)

# ---------- Stage A: calibration ------------------------------------------
# timeline: 120-s calibration prefix, break, later evaluation windows
ax.plot([0.035, 0.305], [0.845, 0.845], color=DARK, lw=0.7)
ax.add_patch(Rectangle((0.035, 0.845), 0.085, 0.055, fc=ORANGE, ec="none", alpha=0.9))
for xe in (0.150, 0.205, 0.260):
    ax.add_patch(Rectangle((xe, 0.845), 0.042, 0.055, fc="#dddddd", ec=GRAY, lw=0.4))
for xb in (0.128, 0.136):  # break marks: disjoint segments
    ax.plot([xb, xb + 0.006], [0.835, 0.91], color=DARK, lw=0.7)
txt(0.0775, 0.805, "calib 120 s", ORANGE, fs=5.5)
txt(0.2275, 0.805, "evaluation", "#666666", fs=5.5)

# calibration EEG + EOG snippets (real exemplar waveforms)
axE = wax([0.035, 0.545, 0.078, 0.135])
axE.plot(t, y_obs, color=GRAY, lw=0.55)
txt(0.038, 0.715, "EEG $y$", "#555555", fs=6, ha="left")
axO = wax([0.035, 0.335, 0.078, 0.135])
axO.plot(t, eog[0], color=PINK, lw=0.55)
axO.plot(t, eog[1] - 3.5 * eog[1].std(), color=PINK, lw=0.55, alpha=0.55)
txt(0.038, 0.505, "EOG $e$", PINK, fs=6, ha="left")

# ridge + empirical-Bayes shrinkage
box(0.132, 0.40, 0.082, 0.24, "white", "#444444")
txt(0.173, 0.585, "ridge (Eq. 2)", "#222222", fs=5.3)
txt(0.173, 0.505, "EB shrink", "#222222", fs=5.3)
txt(0.173, 0.428, "(Eq. 3)", "#222222", fs=5.3)
arrow((0.114, 0.61), (0.132, 0.565), GRAY)
arrow((0.114, 0.41), (0.132, 0.46), PINK)

# reliability criterion branch
dx, dy = 0.028, 0.105
ax.add_patch(Polygon([(0.248 - dx, 0.52), (0.248, 0.52 + dy),
                      (0.248 + dx, 0.52), (0.248, 0.52 - dy)],
                     closed=True, fc="white", ec="#444444", lw=0.7))
txt(0.248, 0.52, "reliable?", "#222222", fs=4.8)
arrow((0.214, 0.52), (0.2185, 0.52), "#444444")
# pass -> calibrated propagation matrix
box(0.262, 0.645, 0.052, 0.13, "#fbe4d8", ORANGE)
txt(0.288, 0.712, r"$\widetilde{A}_i$", ORANGE, fs=8)
arrow((0.252, 0.612), (0.271, 0.648), ORANGE)
txt(0.252, 0.66, "✓", ORANGE, fs=5.5)
# fail -> fallback: guide withheld
box(0.247, 0.115, 0.078, 0.155, "#f0f0f0", GRAY)
txt(0.286, 0.215, "withhold guide", "#555555", fs=5.3)
txt(0.286, 0.150, r"$a_i\,{=}\,0$", "#555555", fs=6)
arrow((0.252, 0.428), (0.268, 0.272), GRAY)
txt(0.251, 0.35, "✗", "#777777", fs=5.5)

# ---------- Stage B: guided diffusion restoration -------------------------
# evaluation EOG through calibrated matrix -> guide a_i
axO2 = wax([0.343, 0.60, 0.068, 0.14])
axO2.plot(t, eog[0], color=PINK, lw=0.55)
axO2.plot(t, eog[1] - 3.5 * eog[1].std(), color=PINK, lw=0.55, alpha=0.55)
txt(0.346, 0.775, "$e$ (eval)", PINK, fs=6, ha="left")
txt(0.452, 0.67, r"$\otimes$", "#222222", fs=9)
arrow((0.411, 0.67), (0.438, 0.67), PINK)
arrow((0.314, 0.72), (0.4395, 0.685), ORANGE, rad=-0.18)
arrow((0.464, 0.664), (0.512, 0.545), ORANGE, rad=-0.1)
txt(0.487, 0.66, "$a_i$", ORANGE, fs=6.5)
# calibration features into the network
arrow((0.302, 0.641), (0.512, 0.45), ORANGE, lw=0.6, rad=0.2)
txt(0.44, 0.435, "$c_i$", ORANGE, fs=6)
# fallback joins the same guide port (gray, dashed)
arrow((0.327, 0.19), (0.512, 0.40), GRAY, ls=(0, (2, 1.4)), rad=-0.1)
# contaminated y snippet
axY = wax([0.352, 0.075, 0.068, 0.13])
axY.plot(t, y_obs, color=GRAY, lw=0.55)
txt(0.341, 0.13, "$y$", "#555555", fs=6, ha="center")
arrow((0.420, 0.15), (0.512, 0.30), GRAY, rad=0.12)

# DDIM loop around shared U-Net
box(0.508, 0.185, 0.098, 0.46, "none", "#999999", lw=0.7, ls=(0, (2.5, 1.5)))
txt(0.557, 0.605, "DDIM", "#777777", fs=5.3)
ax.add_patch(FancyArrowPatch((0.578, 0.185), (0.528, 0.185), arrowstyle="-|>",
                             color="#999999", lw=0.7, mutation_scale=5,
                             shrinkA=0, shrinkB=0,
                             connectionstyle="arc3,rad=0.55"))
box(0.516, 0.285, 0.078, 0.27, BLUE, BLUE, lw=0.8)
txt(0.555, 0.465, "shared", "#1a1a1a", fs=6)
txt(0.555, 0.38, "U-Net", "#1a1a1a", fs=6)

# Eq. 4 output decomposition: subtraction + two learned corrections
arrow((0.606, 0.42), (0.618, 0.42), DARK)
txt(0.612, 0.52, r"$\hat{x}_0{=}(y{-}a_i)$", "#222222", fs=5.8, ha="left")
txt(0.638, 0.42, r"$+\,\Delta_{\mathrm{pop}}$", BLUE, fs=6.2, ha="left")
txt(0.638, 0.32, r"$+\,\Delta_{\mathrm{cal}}$", ORANGE, fs=6.2, ha="left")
txt(0.612, 0.60, "Eq. 4", "#888888", fs=4.8, ha="left")
# restored waveform snippet (real matched output)
arrow((0.648, 0.27), (0.648, 0.225), ORANGE, lw=0.7)
axR = wax([0.612, 0.075, 0.07, 0.135])
axR.plot(t, x_hat, color=ORANGE, lw=0.6)

# ---------- Stage C: uncertainty ------------------------------------------
# K sampled propagation matrices (Eq. 5), stacked chips
for i, (cx, cy) in enumerate(((0.700, 0.44), (0.707, 0.485), (0.714, 0.53))):
    box(cx, cy, 0.046, 0.115, "white" if i < 2 else "#fbe4d8", ORANGE, lw=0.6)
txt(0.737, 0.588, r"$A_i^{(k)}$", ORANGE, fs=6.5)
txt(0.737, 0.40, "Eq. 5", "#888888", fs=4.8)
txt(0.775, 0.63, r"$\times K$", "#555555", fs=6)
# K trajectories fan into the band
for y0, y1, al in ((0.55, 0.60, 0.45), (0.50, 0.50, 0.75), (0.45, 0.40, 0.45)):
    arrow((0.762, y0), (0.797, y1), ORANGE, lw=0.6)
# 80% predictive band around the restored waveform (real band arrays)
axB = fig.add_axes([0.800, 0.20, 0.188, 0.56])
axB.set_axis_off()
axB.fill_between(t, b_mean - b_hw, b_mean + b_hw, color=ORANGE, alpha=0.22, lw=0)
axB.plot(t, b_mean, color=ORANGE, lw=0.85)
txt(0.894, 0.83, "80% PI (Eq. 6)", ORANGE, fs=6)
txt(0.894, 0.10, "mean of $K$ trajectories", "#555555", fs=5.3)

# ---------- compact color key ---------------------------------------------
for xk, ck, sk in ((0.40, BLUE, "shared"), (0.475, ORANGE, "calibrated"),
                   (0.565, GRAY, "fallback")):
    ax.add_patch(Rectangle((xk, 0.038), 0.010, 0.038, fc=ck, ec="none"))
    txt(xk + 0.014, 0.057, sk, "#555555", fs=5.3, ha="left")

figstyle.save(fig, "fig-overview")
