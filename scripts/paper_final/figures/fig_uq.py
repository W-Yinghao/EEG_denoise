"""fig-uq — Predictive intervals and calibration (Sec 5.4/5.6, figure*, 2 rows).

A: nominal vs empirical coverage (0.5/0.8/0.9) for the three UQ policies on
   held-out MobileBCI sealed-8, identity line, dev-frozen reference points
   at 0.80/0.90 (open diamonds).
B: per-participant held-out 80% coverage, one slope line per subject across
   the three policies, nominal 0.8 line, sub-01 under-coverage labeled.
C: interval exemplars for two held-out participants (the only banked
   exemplar arrays): sub-01 (worst-covered, 0.46) and sub-04 (0.77, also
   below nominal) — reference vs predictive mean with the 80% predictive
   interval (the SUBJECT of the paper, not a statistical CI), an
   operator-propagation share strip var_op/(sigma^2+var_op), and a
   zero-artifact episode appended as negative control.
D: CRPS per policy (held-out bars) with the dev-corpus prop+temp value as a
   dashed reference tick (different corpus — scale context only).
E: dev (IRIS F4) temperature sweep from G9 — coverage vs width scale s for
   both width policies (INFL = prop.+temp., TEMP = temp.-only) at nominal
   0.8 (solid) and 0.9 (dotted), joint-15-cell rule curves with the five
   leave-one-fold-out curves as thin lines; frozen scalars (2.40 / 3.00)
   marked at their 0.80 crossings (open diamonds; small diamond = the
   nominal-0.9 coverage reached at the same frozen s), per-fold LOFO picks
   as ticks on the nominal line.
F: diffusion vs deterministic seed-ensemble UQ from G8 per-episode rows
   (IRIS W4 held-out folds, different corpus from A-D): per-fold paired
   points (thin pair lines, NOT bars) for CRPS (banked headline recipe:
   participant mean of episode means, per fold) and for conformal 80%
   hold-out coverage (banked recipe: even-index rows calibrate one width
   scale, odd-index rows held out; per-fold means of the hold-out rows at
   that scale). Pooled values drawn as thick ticks.

Every number is read at runtime from
  paper_final_arrays/t1_heldout_uq_summary.npz   (key `policies`, JSON string)
  results/paper_final/t1_heldout_uq.json         (key `dev_reference`)
  paper_final_arrays/t6_heldout_intervals_sub-01.npz
  paper_final_arrays/t6_heldout_intervals_sub-04.npz
  results/paper_final/gapfill/g9_temp_sweep.json (panel E)
  results/paper_final/gapfill/g8_ensemble.json   (panel F)
Panel E asserts the frozen scalars against BOTH the g9 pooled_check and the
banked t1 policy temperatures, and the curve values at the frozen s against
pooled_check. Panel F rebuilds per-fold values with the banked recipes and
asserts that their means reproduce the pooled_check quoted scalars exactly
(tol 1e-12) before plotting.
Panel C bands are rebuilt as T * sqrt(sigma^2 + var_op) with the banked
policy temperature and CHECKED against the banked per-participant coverage
before plotting (assert, tol 2e-3).

Policy colors follow fig-sealed55: raw=C[RAW], temperature-only=C[POP],
propagation+temperature=C[MATCH]. No CIs/error bars anywhere; the only band
is the 80% predictive interval itself.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import norm

sys.path.insert(0, "/home/infres/yinwang/denoiseNet/scripts/paper_final/figures")
import figstyle  # noqa: E402

figstyle.setup()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.gridspec import GridSpecFromSubplotSpec  # noqa: E402

ROOT = Path("/home/infres/yinwang/denoiseNet")
ARR = ROOT / "paper_final_arrays"

pol = json.loads(str(np.load(ARR / "t1_heldout_uq_summary.npz",
                             allow_pickle=True)["policies"][()]))
dev = json.loads((ROOT / "results/paper_final/t1_heldout_uq.json")
                 .read_text())["dev_reference"]

POLICIES = [("raw", "raw_samples", figstyle.C["RAW"]),
            ("temp.", "temperature_only", figstyle.C["POP"]),
            ("prop.+temp.", "propagation_plus_temperature",
             figstyle.C["MATCH"])]
C_REF = figstyle.C["reference"]
C_PT = figstyle.C["MATCH"]

NOMS = [0.5, 0.8, 0.9]
subs = sorted(pol["raw_samples"]["per_participant_coverage_80"])  # 8 subjects
z80 = norm.ppf(0.9)  # two-sided 80%

rng = np.random.default_rng(3)

fig = plt.figure(figsize=(figstyle.FULL, 3.62))
grid = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.74], hspace=0.50,
                        left=0.055, right=0.985, top=0.945, bottom=0.090)
outer = GridSpecFromSubplotSpec(1, 4, subplot_spec=grid[0],
                                width_ratios=[0.92, 0.88, 2.00, 0.78],
                                wspace=0.52)
bot = GridSpecFromSubplotSpec(1, 3, subplot_spec=grid[1],
                              width_ratios=[2.10, 0.80, 0.80], wspace=0.45)

# ===================== A: nominal vs empirical coverage ==================
axA = fig.add_subplot(outer[0, 0])
figstyle.panel(axA, "A")
axA.plot([0.2, 1.0], [0.2, 1.0], color="0.75", lw=0.6, ls=(0, (3, 2)),
         zorder=1)
for lab, k, col in POLICIES:
    emp = [pol[k]["coverage"][str(n)] for n in NOMS]
    mk = {"raw_samples": "s", "temperature_only": "^",
          "propagation_plus_temperature": "o"}[k]
    open_mk = k == "temperature_only"
    axA.plot(NOMS, emp, color=col, lw=0.9, marker=mk,
             ls=(0, (4, 3)) if open_mk else "-",
             ms=4.0 if open_mk else 3.2, mfc="white" if open_mk else col,
             mew=0.8, zorder=5 if open_mk else 3)
# dev-frozen reference (dev corpus): open diamonds at 0.80 / 0.90
axA.scatter([0.8, 0.9], dev["coverage_80_90"], s=14, marker="D",
            facecolor="none", edgecolor=C_PT, lw=0.7, zorder=4)
axA.text(0.807, dev["coverage_80_90"][0] - 0.028, "dev", color=C_PT,
         fontsize=5, ha="left", va="top")
# direct labels along the (overlapping) lines
axA.text(0.915, pol["raw_samples"]["coverage"]["0.9"], "raw",
         color=figstyle.C["RAW"], fontsize=5.5, ha="left", va="center")
axA.text(0.60, 0.755, "temp.", color=figstyle.C["POP"], fontsize=5.5,
         ha="center", va="bottom")
axA.text(0.665, 0.655, "prop.+temp.", color=C_PT, fontsize=5.5,
         ha="center", va="top")
axA.set_xlim(0.44, 1.01)
axA.set_ylim(0.22, 1.0)
axA.set_xticks(NOMS)
axA.set_xticklabels(["0.5", "0.8", "0.9"])
axA.set_yticks([0.3, 0.5, 0.8, 0.9])
axA.set_xlabel("nominal coverage", labelpad=1.5)
axA.set_ylabel("empirical coverage", labelpad=1.0)

# ============== B: per-participant 80% coverage slope strip ==============
axB = fig.add_subplot(outer[0, 1])
figstyle.panel(axB, "B")
xs = [0.0, 1.0, 2.0]
cov = {k: pol[k]["per_participant_coverage_80"] for _, k, _ in POLICIES}
for s in subs:
    axB.plot(xs, [cov[k][s] for _, k, _ in POLICIES], color="0.85",
             lw=0.4, zorder=1)
for x0, (lab, k, col) in zip(xs, POLICIES):
    v = [cov[k][s] for s in subs]
    xj = x0 + rng.uniform(-0.09, 0.09, len(v))
    axB.scatter(xj, v, s=6, color=col, lw=0, alpha=0.9, zorder=3)
    pooled = pol[k]["coverage"]["0.8"]
    axB.plot([x0 - 0.24, x0 + 0.24], [pooled] * 2, color=col, lw=1.5,
             solid_capstyle="butt", zorder=4)
    sh_lab = {"raw": "raw", "temp.": "temp.",
              "prop.+temp.": "p.+t."}[lab]
    axB.text(x0, -0.07, f"{sh_lab}\nT={pol[k]['temperature']:g}",
             transform=axB.get_xaxis_transform(), fontsize=5.2, color=col,
             ha="center", va="top", linespacing=1.1)
axB.axhline(0.80, color="0.3", lw=0.6, ls=(0, (3, 2)), zorder=2)
axB.text(2.42, 0.815, "nom.", color="0.3", fontsize=5, ha="left",
         va="bottom")
s01 = cov["propagation_plus_temperature"]["sub-01"]
axB.text(2.14, s01, f"sub-01 {s01:.2f}", color="0.15", fontsize=5,
         ha="left", va="center")
axB.set_ylim(0.16, 1.0)
axB.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
axB.set_ylabel("per-part. 80% coverage", labelpad=1.0)
axB.set_xticks([])
axB.set_xlim(-0.42, 3.1)
axB.spines["bottom"].set_visible(False)

# =================== C: two held-out interval exemplars ==================
# The only banked exemplar arrays are sub-01 and sub-04. sub-01 is the
# worst-covered subject (0.46); sub-04 (0.77) is also below nominal —
# shown as two held-out participants, no best/worst framing.
gsC = GridSpecFromSubplotSpec(4, 1, subplot_spec=outer[0, 2],
                              height_ratios=[1.0, 0.16, 1.0, 0.16],
                              hspace=0.22)
EXEMPLARS = [("sub-01", 3, 13, 7), ("sub-04", 4, 32, 7)]  # (sub, ep, ch, zero-ep)
T_pt = pol["propagation_plus_temperature"]["temperature"]
cmap_share = LinearSegmentedColormap.from_list(
    "share", ["#ffffff", figstyle.C["NO_A0"]])
NZ = 256  # zero-artifact control samples appended

for row, (sub, ep, ch, zep) in enumerate(EXEMPLARS):
    z = np.load(ARR / f"t6_heldout_intervals_{sub}.npz")
    m, s, v, r, za = (z["mean"], z["sigma"], z["var_op"], z["reference"],
                      z["zero_artifact"])
    assert za[zep] == 1 and za[ep] == 0
    tot = T_pt * np.sqrt(s ** 2 + v)  # policy scale: temperature * (sigma^2+var_op)^0.5
    # honesty check: this reconstruction must reproduce the banked
    # per-participant coverage before anything is plotted
    cov_re = float((np.abs(r - m) <= z80 * tot).mean())
    assert abs(cov_re - cov["propagation_plus_temperature"][sub]) < 2e-3, \
        (sub, cov_re)
    share = v / (s ** 2 + v + 1e-12)
    cat = lambda a: np.concatenate([a[ep, ch], a[zep, ch][:NZ]])
    mm, rr, tt, sh = cat(m), cat(r), cat(tot), cat(share)
    t = np.arange(mm.size)

    axW = fig.add_subplot(gsC[2 * row])
    if row == 0:
        figstyle.panel(axW, "C")
    # 80% predictive interval (the subject of the paper, not a CI)
    axW.fill_between(t, mm - z80 * tt, mm + z80 * tt, color=C_PT,
                     alpha=0.30, lw=0, zorder=2)
    axW.plot(t, rr, color=C_REF, lw=0.55, zorder=3)
    axW.plot(t, mm, color=C_PT, lw=0.55, zorder=4)
    axW.axvline(512, color="0.5", lw=0.5, ls=":", zorder=5)
    lo = np.minimum(rr, mm - z80 * tt).min()
    hi = np.maximum(rr, mm + z80 * tt).max()
    pad = 0.10 * (hi - lo)
    axW.set_ylim(lo - pad, hi + 2.4 * pad)
    axW.set_xlim(0, t.size)
    axW.set_xticks([])
    axW.tick_params(axis="y", pad=1)
    c80 = cov["propagation_plus_temperature"][sub]
    axW.text(0.012, 0.97, sub, color="0.15", fontsize=5.5, ha="left",
             va="top", transform=axW.transAxes)
    axW.text(0.155, 0.97, f"{c80:.2f}", color=C_PT, fontsize=5.5,
             ha="left", va="top", transform=axW.transAxes)
    if row == 0:
        axW.text(0.345, 0.97, "ref", color=C_REF, fontsize=5, ha="left",
                 va="top", transform=axW.transAxes)
        axW.text(0.41, 0.97, "mean+80% PI", color=C_PT, fontsize=5,
                 ha="left", va="top", transform=axW.transAxes)
        axW.text(0.835, 0.97, "zero-artifact", color="0.4", fontsize=5,
                 ha="center", va="top", transform=axW.transAxes)

    axS = fig.add_subplot(gsC[2 * row + 1])
    axS.imshow(sh[None, :], aspect="auto", cmap=cmap_share, vmin=0, vmax=1,
               extent=(0, t.size, 0, 1), interpolation="nearest")
    axS.axvline(512, color="0.5", lw=0.5, ls=":")
    axS.set_yticks([])
    axS.set_xlim(0, t.size)
    for sp in axS.spines.values():
        sp.set_visible(True); sp.set_linewidth(0.4); sp.set_color("0.6")
    axS.text(-0.012, 0.5, "op. share", color=figstyle.C["NO_A0"],
             fontsize=5, ha="right", va="center", transform=axS.transAxes)
    if row == 1:
        axS.set_xticks([0, 256, 512, 768])
        axS.tick_params(axis="x", pad=1.5)
        axS.set_xlabel("sample", labelpad=1.0)
        axS.text(1.012, 0.5, "0→1", color=figstyle.C["NO_A0"],
                 fontsize=5, ha="left", va="center",
                 transform=axS.transAxes)
    else:
        axS.set_xticks([])

# ========================= D: CRPS per policy ============================
axD = fig.add_subplot(outer[0, 3])
figstyle.panel(axD, "D")
lift = [0.006, 0.006, 0.030]  # stagger the two near-equal labels
for i, (lab, k, col) in enumerate(POLICIES):
    c = pol[k]["crps_gaussian"]
    axD.bar(i, c, width=0.64, color=col, lw=0)
    axD.text(i, c + lift[i], f"{c:.3f}", color="0.15", fontsize=4.8,
             ha="center", va="bottom")
short = ["raw", "temp.", "p.+t."]
for i, ((lab, k, col), sh_lab) in enumerate(zip(POLICIES, short)):
    axD.text(i, -0.045, sh_lab, transform=axD.get_xaxis_transform(),
             fontsize=5.0, color=col, ha="center", va="top")
# dev-corpus prop+temp CRPS: scale context only (different corpus)
axD.axhline(dev["crps"], color="0.4", lw=0.6, ls=(0, (3, 2)))
axD.text(2.62, dev["crps"], f"dev\n{dev['crps']:.3f}", color="0.4",
         fontsize=4.8, ha="left", va="center", linespacing=1.1)
axD.set_ylim(0, 0.44)
axD.set_yticks([0, 0.1, 0.2, 0.3, 0.4])
axD.set_ylabel("CRPS", labelpad=1.0)
axD.set_xticks([])
axD.set_xlim(-0.6, 3.35)
axD.spines["bottom"].set_visible(False)

# ============ E: dev temperature sweep, both policies (G9) ===============
GAP = ROOT / "results/paper_final/gapfill"
g9 = json.loads((GAP / "g9_temp_sweep.json").read_text())
g9row = {(r["curve"], r["policy"], r["nominal_coverage"]): r
         for r in g9["rows"]}
g9chk = g9["pooled_check"]

axE = fig.add_subplot(bot[0, 0])
figstyle.panel(axE, "E")
E_YLO = 0.27
SWEEP = [("INFL", "propagation_plus_temperature", C_PT, "prop.+temp."),
         ("TEMP", "temperature_only", figstyle.C["POP"], "temp.")]
for nom in (0.8, 0.9):
    axE.axhline(nom, color="0.3", lw=0.5, ls=(0, (3, 2)), zorder=1)
    axE.text(0.56, nom + 0.008, f"nom. {nom:.2f}", color="0.3", fontsize=5,
             ha="left", va="bottom")
for pname, tkey, col, lab in SWEEP:
    # five leave-one-fold-out curves (nominal 0.8), thin
    picks = []
    for f in range(5):
        r = g9row[(f"dev_lofo_holdfold_{f}", pname, 0.8)]
        axE.plot(r["s_grid"], r["coverage"], color=col, lw=0.35, alpha=0.35,
                 zorder=2)
        picks.append(r["first_s_reaching_080"])
    bank = g9chk[f"dev_lofo_{pname}_picks"]["banked_f4_decision_per_hold_fold"]
    assert picks == [bank[str(f)] for f in range(5)], (pname, picks)
    axE.scatter(picks, [0.8] * 5, marker="|", s=12, color=col, lw=0.7,
                zorder=5)
    # joint 15-cell rule curves: nominal 0.8 solid, 0.9 dotted
    r8 = g9row[("dev_joint_15cell", pname, 0.8)]
    r9 = g9row[("dev_joint_15cell", pname, 0.9)]
    axE.plot(r8["s_grid"], r8["coverage"], color=col, lw=1.0, zorder=4)
    axE.plot(r9["s_grid"], r9["coverage"], color=col, lw=0.7,
             ls=(0, (1.2, 1.4)), zorder=3)
    # frozen scalar: must equal both the g9 pooled_check crossing and the
    # banked t1 policy temperature; curve value must equal pooled_check
    s_star = r8["first_s_reaching_080"]
    assert s_star == g9chk[f"dev_joint_{pname}_first_s_ge_080"]["recomputed"]
    assert s_star == pol[tkey]["temperature"], (pname, s_star)
    i = r8["s_grid"].index(s_star)
    cov8, cov9 = r8["coverage"][i], r9["coverage"][i]
    ck = g9chk[f"dev_joint_{pname}_cov80_at_s{s_star:.2f}"]["recomputed"]
    assert abs(cov8 - ck) < 1e-12, (pname, cov8, ck)
    axE.vlines(s_star, E_YLO, cov9, color=col, lw=0.5, ls=":", zorder=2)
    axE.scatter([s_star], [cov8], s=14, marker="D", facecolor="white",
                edgecolor=col, lw=0.8, zorder=6)
    axE.scatter([s_star], [cov9], s=7, marker="D", facecolor="white",
                edgecolor=col, lw=0.6, zorder=6)
    axE.text(s_star + 0.05, E_YLO + 0.012, f"{s_star:.2f}", color=col,
             fontsize=5, ha="left", va="bottom")
axE.text(0.98, 0.855, "prop.+temp.", color=C_PT, fontsize=5.5, ha="left",
         va="center")
axE.text(1.80, 0.575, "temp.", color=figstyle.C["POP"], fontsize=5.5,
         ha="left", va="top")
axE.text(5.92, 0.335, "solid: nom. 0.80\ndotted: nom. 0.90", color="0.4",
         fontsize=5, ha="right", va="bottom", linespacing=1.25)
axE.text(0.33, 0.985, "dev (IRIS F4)", color="0.35", fontsize=5,
         ha="center", va="top", transform=axE.transAxes)
axE.set_xlim(0.5, 6.0)
axE.set_ylim(E_YLO, 0.975)
axE.set_xticks([1, 2, 3, 4, 5, 6])
axE.set_yticks([0.3, 0.5, 0.8, 0.9])
axE.set_xlabel("width scale s", labelpad=1.5)
axE.set_ylabel("dev coverage", labelpad=1.0)

# ===== F: diffusion vs deterministic seed-ensemble, per fold (G8) ========
g8 = json.loads((GAP / "g8_ensemble.json").read_text())
g8chk = g8["pooled_check"]
w4 = [r for r in g8["rows"] if r["block"] == "w4_heldout_uq"]
FOLDS = sorted({r["fold"] for r in w4})
ARMS = [("diffusion", "diff", figstyle.C["MATCH"], "diff."),
        ("det_ensemble", "det", figstyle.C["DET_ENS"], "det. ens.")]
crps_f, conf_f = {}, {}
for arm, tag, col, lab in ARMS:
    ar = [r for r in w4 if r["arm"] == arm]
    # banked headline CRPS recipe: participant mean of episode means
    per_part = {}
    for r in ar:
        per_part.setdefault((r["fold"], r["participant"]), []).append(r["crps"])
    pm = {k: float(np.mean(v)) for k, v in per_part.items()}
    crps_f[tag] = [float(np.mean([v for (f, _), v in pm.items() if f == fold]))
                   for fold in FOLDS]
    assert abs(np.mean(crps_f[tag]) - g8chk[f"{tag}_crps"]["quoted"]) < 1e-12
    # banked conformal recipe: even-index rows calibrate one width scale
    # (argmin |mean cov80 - 0.80| over the 26-point grid), odd rows hold out
    cal = np.array([r["cov80_scale_grid"]
                    for i, r in enumerate(ar) if i % 2 == 0])
    hold = [r for i, r in enumerate(ar) if i % 2 == 1]
    j = int(np.argmin(np.abs(cal.mean(axis=0) - 0.80)))
    conf_f[tag] = [float(np.mean([r["cov80_scale_grid"][j] for r in hold
                                  if r["fold"] == fold])) for fold in FOLDS]
    assert abs(np.mean(conf_f[tag]) -
               g8chk[f"{tag}_conformal_holdout_coverage_80"]["quoted"]) < 1e-12

axF1 = fig.add_subplot(bot[0, 1])
figstyle.panel(axF1, "F")
axF2 = fig.add_subplot(bot[0, 2])
dxs = np.linspace(-0.06, 0.06, len(FOLDS))  # fixed per-fold offsets
for ax, data in ((axF1, crps_f), (axF2, conf_f)):
    for fi in range(len(FOLDS)):
        ax.plot([dxs[fi], 1 + dxs[fi]],
                [data["diff"][fi], data["det"][fi]], color="0.85", lw=0.5,
                zorder=1)
    for x0, (arm, tag, col, lab) in zip((0, 1), ARMS):
        v = data[tag]
        ax.scatter(x0 + dxs, v, s=7, color=col, lw=0, zorder=3)
        pooled = float(np.mean(v))
        ax.plot([x0 - 0.24, x0 + 0.24], [pooled] * 2, color=col, lw=1.5,
                solid_capstyle="butt", zorder=4)
        ax.text(x0, -0.055, lab, transform=ax.get_xaxis_transform(),
                fontsize=5.2, color=col, ha="center", va="top")
    ax.set_xticks([])
    ax.set_xlim(-0.62, 1.78)
    ax.spines["bottom"].set_visible(False)
axF1.set_ylim(0, 0.36)
axF1.set_yticks([0, 0.1, 0.2, 0.3])
axF1.set_ylabel("CRPS (per fold)", labelpad=1.0)
axF1.text(-0.30, float(np.mean(crps_f["diff"])),
          f"{np.mean(crps_f['diff']):.3f}", color=figstyle.C["MATCH"],
          fontsize=4.8, ha="right", va="center")
axF1.text(1.30, float(np.mean(crps_f["det"])),
          f"{np.mean(crps_f['det']):.3f}", color=figstyle.C["DET_ENS"],
          fontsize=4.8, ha="left", va="center")
axF1.text(0.5, max(crps_f["diff"][1], crps_f["det"][1]) + 0.008, "fold 1",
          color="0.5", fontsize=4.8, ha="center", va="bottom")
axF2.axhline(0.80, color="0.3", lw=0.6, ls=(0, (3, 2)), zorder=2)
axF2.text(1.74, 0.807, "nom.", color="0.3", fontsize=5, ha="right",
          va="bottom")
axF2.set_ylim(0.55, 0.88)
axF2.set_yticks([0.6, 0.7, 0.8])
axF2.set_ylabel("conformal 80% cov.", labelpad=1.0)
for x0, tag, col in ((0, "diff", figstyle.C["MATCH"]),
                     (1, "det", figstyle.C["DET_ENS"])):
    axF2.text(x0 + 0.28, float(np.mean(conf_f[tag])),
              f"{np.mean(conf_f[tag]):.3f}", color=col, fontsize=4.8,
              ha="left", va="center")
axF2.text(0.5, 1.055, "IRIS held-out", color="0.35", fontsize=5,
          ha="center", va="bottom", transform=axF2.transAxes)

figstyle.save(fig, "fig-uq")
