"""fig-ablation: guide-feature ablation matrix + contrast bars (appendix).

Panel A: guide (rows) x calibration-features (cols) matrix of
participant-first temporal RRMSE, darker = worse; sparse cells shown as
not-run.  Panel B: delta-RRMSE vs the full system (matched|matched),
per-cell horizontal bars, direct-labeled.  All numbers read at runtime
from t3_ablation_matrix.npz (single seed, shared episodes/noise — stated
in the caption, not here).
"""
import json
import sys

import numpy as np

sys.path.insert(0, "/home/infres/yinwang/denoiseNet/scripts/paper_final/figures")
import figstyle

figstyle.setup()

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize

NPZ = "/home/infres/yinwang/denoiseNet/paper_final_arrays/t3_ablation_matrix.npz"
f = np.load(NPZ, allow_pickle=True)
cells = json.loads(str(f["cells"][()]))
contrasts = json.loads(str(f["contrasts"][()]))


def key(guide, feat):
    return f"guide={guide}|features={feat}"


GUIDES = ["matched", "matched_unshrunk", "none", "population"]
FEATS = ["matched", "population"]
GUIDE_LBL = {
    "matched": "matched",
    "matched_unshrunk": "matched, $\\lambda$=1",
    "none": "none",
    "population": "population",
}
# arm color per guide level (one arm = one color, from figstyle.C)
GUIDE_COL = {
    "matched": figstyle.C["matched"],
    "matched_unshrunk": figstyle.C["matched"],
    "none": figstyle.C["unguided"],
    "population": figstyle.C["population"],
}

fig, (axA, axB) = plt.subplots(
    2, 1, figsize=(figstyle.HALF, 3.5),
    gridspec_kw={"height_ratios": [1.0, 0.95], "hspace": 0.52},
)

# ---------------------------------------------------------------- Panel A
M = np.full((len(GUIDES), len(FEATS)), np.nan)
for i, g in enumerate(GUIDES):
    for j, ft in enumerate(FEATS):
        if key(g, ft) in cells:
            M[i, j] = cells[key(g, ft)]

vals = M[np.isfinite(M)]
norm = Normalize(vmin=vals.min(), vmax=vals.max())
cmap = mpl.colormaps["YlOrRd"]  # darker = worse
axA.imshow(np.ma.masked_invalid(M), cmap=cmap, norm=norm, aspect="auto")
for i in range(len(GUIDES)):
    for j in range(len(FEATS)):
        if np.isfinite(M[i, j]):
            dark = norm(M[i, j]) > 0.6
            axA.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center",
                     fontsize=7, fontweight="bold",
                     color="white" if dark else "#2b2b2b")
        else:
            axA.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                        facecolor="#f2f2f2",
                                        edgecolor="#cccccc", lw=0.4,
                                        hatch="///", zorder=0))
            axA.text(j, i, "not run", ha="center", va="center",
                     fontsize=5.5, color="#8a8a8a", style="italic")
# full-system reference cell (baseline of panel B) outlined in arm color
axA.add_patch(plt.Rectangle((-0.5 + 0.02, -0.5 + 0.04), 0.96, 0.92,
                            facecolor="none", lw=1.4,
                            edgecolor=figstyle.C["matched"], zorder=3))
# thin white grid between cells
for j in range(len(FEATS) - 1):
    axA.axvline(j + 0.5, color="white", lw=1.2)
for i in range(len(GUIDES) - 1):
    axA.axhline(i + 0.5, color="white", lw=1.2)
axA.set_xticks(range(len(FEATS)), FEATS)
axA.set_yticks(range(len(GUIDES)),
               [GUIDE_LBL[g] for g in GUIDES])
for tick, g in zip(axA.get_yticklabels(), GUIDES):
    tick.set_color(GUIDE_COL[g])
axA.set_xlabel("calibration features")
axA.set_ylabel("guide")
axA.tick_params(length=0)
for s in axA.spines.values():
    s.set_visible(False)
figstyle.panel(axA, "A")

# ---------------------------------------------------------------- Panel B
order = sorted(contrasts, key=contrasts.get)  # smallest delta on top
BAR_LBL = {
    key("matched", "population"): "pop. features",
    key("none", "matched"): "no guide",
    key("none", "population"): "no guide + pop. feat.",
    key("population", "population"): "pop. guide + feat.",
    key("matched_unshrunk", "matched"): "no shrinkage ($\\lambda$=1)",
}
ypos = np.arange(len(order))[::-1]
for y, k in zip(ypos, order):
    g = k.split("|")[0].split("=")[1]
    col = GUIDE_COL[g]
    unshrunk = g == "matched_unshrunk"
    axB.barh(y, contrasts[k], height=0.62, color="none" if unshrunk else col,
             edgecolor=col, lw=1.1 if unshrunk else 0.0,
             hatch="////" if unshrunk else None)
    axB.text(contrasts[k] + 0.006, y, f"+{contrasts[k]:.3f}",
             va="center", ha="left", fontsize=6.5, color=col,
             fontweight="bold" if unshrunk else "normal")
axB.set_yticks(ypos, [BAR_LBL[k] for k in order])
for tick, k in zip(axB.get_yticklabels(), order):
    tick.set_color(GUIDE_COL[k.split("|")[0].split("=")[1]])
axB.axvline(0, color="#2b2b2b", lw=0.8)
full = cells[key("matched", "matched")]
axB.set_xlabel(f"$\\Delta$RRMSE vs full system ({full:.3f})")
axB.set_xlim(0, max(contrasts.values()) * 1.22)
axB.spines["left"].set_visible(False)
axB.tick_params(axis="y", length=0)
figstyle.panel(axB, "B")

figstyle.save(fig, "fig-ablation")
