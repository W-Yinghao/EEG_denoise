"""fig-sealed55 — Sealed 55-participant confirmation (Sec 5.6).

A: per-subject paired rRMSE own/zero/wrong at n=30 and n=259 (slope columns),
   blind reference tick.
B: per-subject contrast columns (gain zero-own at n=30/n=259, independent
   probe cohort, own-wrong at n=30/n=259, trend n259-n30). All 55 dots shown,
   no CIs.
C: per-subject 80% coverage strips for the three UQ policies (raw T=1.0,
   temperature-only T=3.25, propagation+temperature T=2.45), nominal line.
D: per-subject gain vs query injection ratio (log x), n=30 and n=259.

Every number is read at runtime from
  results/iris/sealed_confirm/sealed_rows.json
  results/iris/sealed_confirm/sealed_confirm_decision.json
  results/iris/sealed_confirm/probe/probe.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.transforms as mtransforms

sys.path.insert(0, "/home/infres/yinwang/denoiseNet/scripts/paper_final/figures")
import figstyle  # noqa: E402

figstyle.setup()
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path("/home/infres/yinwang/denoiseNet")
SEALED = ROOT / "results/iris/sealed_confirm"

rows = json.loads((SEALED / "sealed_rows.json").read_text())["rows"]
dec = json.loads((SEALED / "sealed_confirm_decision.json").read_text())
probe = json.loads((SEALED / "probe" / "probe.json").read_text())

# ------------------------------------------------------------------ data
def by_subj(rws, key):
    return {r["subject"]: r[key] for r in rws}

cond30 = [r for r in rows if r["checkpoint"] == "model_n30_COND.pt"]
cond259 = [r for r in rows if r["checkpoint"] == "model_n259_COND.pt"]
blind = [r for r in rows if r["checkpoint"] == "model_n259_BLIND.pt"]
subs = sorted(r["subject"] for r in cond259)
assert len(cond30) == len(cond259) == len(blind) == 55

own30 = by_subj(cond30, "rrmse_own"); zero30 = by_subj(cond30, "rrmse_zero")
wrong30 = by_subj(cond30, "rrmse_wrong_mean")
own259 = by_subj(cond259, "rrmse_own"); zero259 = by_subj(cond259, "rrmse_zero")
wrong259 = by_subj(cond259, "rrmse_wrong_mean")
qratio = by_subj(cond259, "query_ratio")
blind_med = float(np.median([r["rrmse_blind"] for r in blind]))

gain30 = {s: zero30[s] - own30[s] for s in subs}
gain259 = {s: zero259[s] - own259[s] for s in subs}
dwrong30 = {s: wrong30[s] - own30[s] for s in subs}
dwrong259 = {s: wrong259[s] - own259[s] for s in subs}
trend = {s: gain259[s] - gain30[s] for s in subs}
probe_gain = {r["subject"]: r["rrmse_zero"] - r["rrmse_own"]
              for r in probe["rows"] if "rrmse_zero" in r}

pol = dec["option_b"]["policies"]
POLICIES = [("raw", "raw_samples", figstyle.C["RAW"]),
            ("temp.", "temperature_only", figstyle.C["POP"]),
            ("prop.+temp.", "propagation_plus_temperature",
             figstyle.C["MATCH"])]
cov = {k: pol[k]["per_subject_coverage_80"] for _, k, _ in POLICIES}
pooled = {k: pol[k]["coverage"]["0.8"] for _, k, _ in POLICIES}
temps = {k: pol[k]["temperature"] for _, k, _ in POLICIES}

rng = np.random.default_rng(7)

C_OWN = figstyle.C["MATCH"]
C_ZERO = figstyle.C["NO_A0"]
C_WRONG = figstyle.C["WRONG"]
C_REF = figstyle.C["reference"]

# ------------------------------------------------------------------ figure
fig = plt.figure(figsize=(figstyle.FULL, 4.5))
outer = fig.add_gridspec(2, 2, width_ratios=[1.18, 1.0],
                         height_ratios=[1.0, 1.0],
                         wspace=0.24, hspace=0.42,
                         left=0.065, right=0.995, top=0.95, bottom=0.075)

# ========================== Panel A: paired columns =====================
axA = fig.add_subplot(outer[0, 0])
figstyle.panel(axA, "A")
GROUPS = [(0.0, own30, zero30, wrong30, "n=30"),
          (3.4, own259, zero259, wrong259, "n=259")]
ARMS = [("own", C_OWN), ("zero", C_ZERO), ("wrong", C_WRONG)]
for x0, o, z, w, lab in GROUPS:
    vals = [o, z, w]
    for s in subs:
        axA.plot([x0, x0 + 1, x0 + 2], [o[s], z[s], w[s]],
                 color="0.82", lw=0.4, zorder=1)
    for j, ((arm, col), v) in enumerate(zip(ARMS, vals)):
        xj = x0 + j + rng.uniform(-0.10, 0.10, len(subs))
        axA.scatter(xj, [v[s] for s in subs], s=5, color=col, lw=0,
                    alpha=0.85, zorder=3)
        med = float(np.median([v[s] for s in subs]))
        axA.plot([x0 + j - 0.28, x0 + j + 0.28], [med] * 2, color=col,
                 lw=1.5, solid_capstyle="butt", zorder=4)
        if x0 > 0:  # source medians on the n=259 group
            axA.text(x0 + j, 1.01, f"{med:.3f}", color=col, fontsize=5,
                     ha="center", va="bottom",
                     transform=mtransforms.blended_transform_factory(
                         axA.transData, axA.transAxes))
# blind reference tick on the n=259 group
axA.plot([3.0, 5.8], [blind_med] * 2, color=C_REF, lw=0.7, ls=(0, (3, 2)),
         zorder=2)
axA.text(5.88, blind_med, f"blind {blind_med:.3f}", color=C_REF,
         fontsize=5, ha="left", va="center")
axA.set_yscale("log")
yt = [0.2, 0.3, 0.5, 1.0, 2.0]
axA.set_yticks(yt)
axA.set_yticklabels(["0.2", "0.3", "0.5", "1", "2"])
axA.tick_params(axis="y", which="minor", length=0)
axA.set_ylabel("rRMSE")
axA.set_xticks([0, 1, 2, 3.4, 4.4, 5.4])
axA.set_xticklabels(["own", "zero", "wrong"] * 2)
for tick, col in zip(axA.get_xticklabels(), [c for _, c in ARMS] * 2):
    tick.set_color(col)
for xc, lab in ((1.0, "n=30"), (4.4, "n=259")):
    axA.text(xc, -0.17, lab, transform=axA.get_xaxis_transform(),
             fontsize=6, color="0.15", ha="center", va="top")
axA.set_xlim(-0.55, 7.0)
axA.spines["bottom"].set_visible(False)
axA.tick_params(axis="x", length=0)

# ==================== Panel B: contrast dot columns =====================
axB = fig.add_subplot(outer[0, 1])
figstyle.panel(axB, "B")
ga = dec["gain_by_n"]; ow = dec["own_minus_wrong_by_n"]
tr = dec["trend_nmax_minus_nmin"]; pg = probe["gain"]
COLS = [
    (0.0, [gain30[s] for s in subs], C_ZERO, ga["30"]["mean"],
     f'{ga["30"]["positive_count"]}/{ga["30"]["subjects"]}', "gain\nn=30", False),
    (1.0, [gain259[s] for s in subs], C_ZERO, ga["259"]["mean"],
     f'{ga["259"]["positive_count"]}/{ga["259"]["subjects"]}', "gain\nn=259", False),
    (2.0, [probe_gain[s] for s in sorted(probe_gain)], C_ZERO, pg["mean"],
     f'{pg["positive_count"]}/{pg["subjects"]}', "probe\nn=8", True),
    (3.2, [dwrong30[s] for s in subs], C_WRONG, ow["30"]["mean"],
     f'{ow["30"]["positive_count"]}/{ow["30"]["subjects"]}', "own-wr.\nn=30", False),
    (4.2, [dwrong259[s] for s in subs], C_WRONG, ow["259"]["mean"],
     f'{ow["259"]["positive_count"]}/{ow["259"]["subjects"]}', "own-wr.\nn=259", False),
    (5.4, [trend[s] for s in subs], figstyle.C["RAW"], tr["mean"], "flat",
     "trend\n259-30", False),
]
axB.axhline(0.0, color="0.55", lw=0.6, zorder=1)
for x0, vals, col, mean_src, count, lab, hollow in COLS:
    xj = x0 + rng.uniform(-0.16, 0.16, len(vals))
    if hollow:
        axB.scatter(xj, vals, s=8, facecolor="none", edgecolor=col,
                    lw=0.6, zorder=3)
    else:
        axB.scatter(xj, vals, s=5, color=col, lw=0, alpha=0.8, zorder=3)
    axB.plot([x0 - 0.26, x0 + 0.26], [mean_src] * 2, color=col, lw=1.5,
             solid_capstyle="butt", zorder=4)
    axB.text(x0, 1.01, count, transform=axB.get_xaxis_transform(),
             fontsize=5, color="0.25", ha="center", va="bottom")
    axB.text(x0, -0.055, lab, transform=axB.get_xaxis_transform(),
             fontsize=5.2, color=col,
             ha="center", va="top", linespacing=1.1)
axB.text(5.4 + 0.32, tr["mean"], f'{tr["mean"]:+.3f}', color="0.3",
         fontsize=5, ha="left", va="center")
axB.set_ylabel(r"$\Delta$rRMSE")
axB.set_xticks([])
axB.set_xlim(-0.55, 6.35)
axB.spines["bottom"].set_visible(False)

# ================ Panel C: 80% coverage slope strips ====================
axC = fig.add_subplot(outer[1, 0])
figstyle.panel(axC, "C")
xs = [0.0, 1.0, 2.0]
for s in subs:
    axC.plot(xs, [cov[k][s] for _, k, _ in POLICIES], color="0.85",
             lw=0.35, zorder=1)
for x0, (lab, k, col) in zip(xs, POLICIES):
    v = [cov[k][s] for s in subs]
    xj = x0 + rng.uniform(-0.08, 0.08, len(v))
    axC.scatter(xj, v, s=5, color=col, lw=0, alpha=0.85, zorder=3)
    axC.plot([x0 - 0.22, x0 + 0.22], [pooled[k]] * 2, color=col, lw=1.6,
             solid_capstyle="butt", zorder=4)
    axC.text(x0 + 0.26, pooled[k], f"{pooled[k]:.3f}", color=col,
             fontsize=5.5, ha="left", va="center")
    axC.text(x0, -0.065, f"{lab}\nT={temps[k]:g}",
             transform=axC.get_xaxis_transform(), fontsize=5.2, color=col,
             ha="center", va="top", linespacing=1.1)
axC.axhline(0.80, color="0.3", lw=0.6, ls=(0, (3, 2)), zorder=2)
axC.text(2.62, 0.815, "nominal 0.80", color="0.3", fontsize=5,
         ha="left", va="bottom")
axC.set_ylim(0.0, 1.02)
axC.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
axC.set_ylabel("per-subject 80% coverage")
axC.set_xticks([])
axC.set_xlim(-0.4, 3.35)
axC.spines["bottom"].set_visible(False)

# ============ Panel D: gain vs query injection ratio (log x) ============
axD = fig.add_subplot(outer[1, 1])
figstyle.panel(axD, "D")
axD.axhline(0.0, color="0.55", lw=0.6, zorder=1)
q = np.array([qratio[s] for s in subs])
g259 = np.array([gain259[s] for s in subs])
g30 = np.array([gain30[s] for s in subs])
axD.scatter(q, g30, s=8, facecolor="none", edgecolor=C_ZERO, lw=0.5,
            zorder=2)
axD.scatter(q, g259, s=7, color=C_ZERO, lw=0, alpha=0.85, zorder=3)
axD.set_xscale("log")
xt = [0.6, 1, 2, 4, 6]
axD.set_xticks(xt); axD.set_xticklabels([str(v) for v in xt])
axD.tick_params(axis="x", which="minor", length=0)
axD.set_xlabel("query injection ratio")
axD.set_ylabel("gain (zero - own)")
# direct labels for the two checkpoints
axD.scatter([0.98], [0.945], s=7, color=C_ZERO, lw=0,
            transform=axD.transAxes, clip_on=False)
axD.text(0.96, 0.945, "n=259", color=C_ZERO, fontsize=5.5, ha="right",
         va="center", transform=axD.transAxes)
axD.scatter([0.98], [0.87], s=8, facecolor="none", edgecolor=C_ZERO,
            lw=0.5, transform=axD.transAxes, clip_on=False)
axD.text(0.96, 0.87, "n=30", color=C_ZERO, fontsize=5.5, ha="right",
         va="center", transform=axD.transAxes)

figstyle.save(fig, "fig-sealed55")
