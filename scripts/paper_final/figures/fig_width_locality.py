"""fig-width-locality: where interval width comes from, per calibration cell.

Scatter of the operator-propagation share of squared interval width
Var_op/(sigma^2 + Var_op) against calibration within-variance (log x) for all
270 subject|session|task cells, colored by EB shrinkage lambda, hard-gated
cells ring-marked, with a marginal histogram of the share (mean ~0.30 tick).
All numbers are read from t6_width_locality.npz at runtime.
"""
import sys

import numpy as np

sys.path.insert(0, "/home/infres/yinwang/denoiseNet/scripts/paper_final/figures")
import figstyle  # noqa: E402

figstyle.setup()

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import gridspec  # noqa: E402

DATA = "/home/infres/yinwang/denoiseNet/paper_final_arrays/t6_width_locality.npz"
d = np.load(DATA, allow_pickle=True)
wv = d["within_v"]
share = d["propagation_width_share"]
lam = d["lam"]
gate = d["hard_gate"].astype(bool)
n = wv.size
mean_share = share.mean()

fig = plt.figure(figsize=(figstyle.HALF, 2.35))
gs = gridspec.GridSpec(1, 2, width_ratios=[1.0, 0.24], wspace=0.06)
axS = fig.add_subplot(gs[0, 0])
axH = fig.add_subplot(gs[0, 1], sharey=axS)

# --- A: scatter, 270 cells ---
sc = axS.scatter(wv, share, c=lam, cmap="viridis", vmin=0.0, vmax=1.0,
                 s=11, alpha=0.9, lw=0, zorder=3)
axS.scatter(wv[gate], share[gate], facecolors="none", edgecolors="k",
            s=30, lw=0.7, zorder=4)
axS.set_xscale("log")

# binned median of the share (deciles of log within-variance)
edges = np.quantile(np.log10(wv), np.linspace(0, 1, 11))
mids, meds = [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (np.log10(wv) >= lo) & (np.log10(wv) <= hi)
    mids.append(10 ** (0.5 * (lo + hi)))
    meds.append(np.median(share[m]))
axS.plot(mids, meds, color="#2b2b2b", lw=1.0, zorder=5)

# mean-share reference tick (value from data)
for ax in (axS, axH):
    ax.axhline(mean_share, color="#2b2b2b", lw=0.6, ls=(0, (4, 3)), zorder=2)
axS.set_xlim(wv.min() * 0.55, wv.max() * 1.8)
axS.text(wv.min() * 0.62, mean_share - 0.014, f"mean {mean_share:.2f}",
         fontsize=6, color="#2b2b2b", ha="left", va="top", zorder=6)

# direct label for the ring marker, next to the gated cluster
gx = np.exp(np.median(np.log(wv[gate])))
axS.text(gx, share[gate].max() + 0.045, f"hard-gated ({gate.sum()})",
         fontsize=6, color="k", ha="center", va="bottom")

axS.set_xlabel("calibration within-variance")
axS.set_ylabel(r"operator share $\mathrm{Var_{op}}/(\sigma^2{+}\mathrm{Var_{op}})$")
axS.set_ylim(-0.02, 0.60)
axS.set_yticks([0.0, 0.15, 0.30, 0.45, 0.60])
figstyle.panel(axS, "A")

# --- colorbar for lambda, inset in the empty top-left strip ---
axC = axS.inset_axes([0.03, 0.94, 0.40, 0.045])
cb = fig.colorbar(sc, cax=axC, orientation="horizontal")
cb.set_ticks([0.0, 0.5, 1.0])
axC.tick_params(labelsize=5.5, length=1.5, pad=1)
cb.outline.set_linewidth(0.5)
axS.text(0.45, 0.965, r"EB shrinkage $\lambda$", transform=axS.transAxes,
         fontsize=6, ha="left", va="center")

# --- B: marginal histogram of the width share ---
axH.hist(share, bins=26, orientation="horizontal", color="#bdbdbd", lw=0)
axH.hist(share[gate], bins=26, range=(share.min(), share.max()),
         orientation="horizontal", color="k", lw=0)
axH.set_xlabel("cells")
axH.set_xticks([25])
axH.text(0.95, 0.99, f"n={n}", transform=axH.transAxes, fontsize=6,
         ha="right", va="top")
axH.tick_params(labelleft=False)
axH.spines["left"].set_visible(False)
axH.tick_params(left=False)
figstyle.panel(axH, "B")

figstyle.save(fig, "fig-width-locality")
