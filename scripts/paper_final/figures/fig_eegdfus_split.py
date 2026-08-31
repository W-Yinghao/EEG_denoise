#!/usr/bin/env python3
"""fig-eegdfus-split — EEGDfus released-vs-strict split effect (v3 appendix).

Panel A: CC dumbbells released -> strict per arm on SSED, identity baseline
(CC(noisy, clean), recomputed here from the upstream-prepared rows) as a
reference line, published EEGDfus 0.992 as a tick.
Panel B: same dumbbells for RRMSE_t (EEGDfus 0.085 -> 0.373 is the headline).
Panel C: PLV in the discriminative delta/theta bands, released (open) vs
strict (filled) per arm; alpha/beta/gamma omitted (>= 0.99 for every arm).
Panel D: leak mechanism — per-row EEGDfus CC on the released test split,
grouped by whether the row also sits in the training split (1089/1334 do),
vs the strict test split (no overlap). The released number is driven by
memorized rows; fresh released rows match the strict distribution.
Panel E: representative strict-test segment (the row closest to both the
EEGDfus and ours-SSED strict means), clean vs noisy/EEGDfus/ours overlays
with per-row CC. Released-vs-strict outputs for the same row are near-
identical (max per-row CC gap 0.04 over the 132 shared rows) — the split
effect is compositional, so one split is shown.

House rules: NO confidence intervals / error bars / bands; per-row dots
instead. All plotted numbers are read from the banked JSONs / npz or
recomputed from the upstream-prepared dataset at runtime.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/infres/yinwang/denoiseNet/scripts/paper_final/figures")
import figstyle

figstyle.setup()

import matplotlib.pyplot as plt

sys.path.insert(0, "/home/infres/yinwang/denoiseNet/scripts/paper_final")
import eegdfus_ssed as ES

REPO = Path("/home/infres/yinwang/denoiseNet")
DFUS = Path("/projects/EEG-foundation-model/derived/denoiseNet/eegdfus_ssed")
OURS = Path("/projects/EEG-foundation-model/derived/denoiseNet/e34_ours")
C = figstyle.C

# ---------------------------------------------------------------- banked JSONs
ssed = json.loads((REPO / "results/paper_final/eegdfus_ssed/ssed_eval.json").read_text())
e3b = json.loads((REPO / "results/paper_final/e34/e3b_ours_ssed.json").read_text())
e4 = json.loads((REPO / "results/paper_final/e34/e4_generalization.json").read_text())
plv = json.loads((REPO / "results/paper_final/e5/plv.json").read_text())

# one arm = one color (first figure to introduce these arms)
ARMS = [  # (display, color, {split: metrics})
    ("EEGDfus", C["EEGDfus"],
     {"rel": ssed["released_test"], "st": ssed["strict_test"]}),
    ("ours zero-shot", C["POP"],
     {"rel": e4["zero_shot"]["released_test"], "st": e4["zero_shot"]["strict_test"]}),
    ("ours SSED-trained", C["ours"],
     {"rel": e3b["released_test"], "st": e3b["strict_test"]}),
    ("ours fine-tuned", C["NO_A0"],
     {"rel": e4["finetune_10pct"]["released_test"], "st": e4["finetune_10pct"]["strict_test"]}),
]
PUB = ssed["published_reference"]

# ------------------------------------------- recomputed rows (mandatory fix)
x, y = ES._dataset()          # upstream prepare: standardize + re-mix + unroll
x = np.asarray(x, np.float64)
y = np.asarray(y, np.float64)
splits = ES._splits(len(x))


def cc_rows(a, b):
    a = a - a.mean(-1, keepdims=True)
    b = b - b.mean(-1, keepdims=True)
    return (a * b).sum(-1) / np.sqrt((a * a).sum(-1) * (b * b).sum(-1))


rel_idx = np.asarray(splits["released_test"])
st_idx = np.asarray(splits["strict_test"])
id_cc = {"rel": float(cc_rows(x[rel_idx], y[rel_idx]).mean()),
         "st": float(cc_rows(x[st_idx], y[st_idx]).mean())}

d_rel = np.load(DFUS / "denoised_released_test.npz")
d_st = np.load(DFUS / "denoised_strict_test.npz")
cc_dfus_rel = cc_rows(d_rel["denoised"].astype(np.float64), y[d_rel["idx"]])
cc_dfus_st = cc_rows(d_st["denoised"].astype(np.float64), y[d_st["idx"]])
# recomputation must reproduce the banked means exactly
assert np.isclose(cc_dfus_rel.mean(), ssed["released_test"]["cc"], atol=1e-6)
assert np.isclose(cc_dfus_st.mean(), ssed["strict_test"]["cc"], atol=1e-6)

in_train = np.zeros(len(x), bool)
in_train[np.asarray(splits["train"])] = True
leak_mask = in_train[d_rel["idx"]]          # 1089 of 1334 released-test rows
assert not in_train[d_st["idx"]].any()

e3b_st = np.load(OURS / "denoised_e3b_strict_test.npz")
cc_e3b_st = cc_rows(np.asarray(e3b_st["denoised"], np.float64), y[e3b_st["idx"]])

# representative strict row: closest to both arms' strict per-row CC means
pos_dfus = {v: i for i, v in enumerate(d_st["idx"])}
dist = np.array([abs(cc_dfus_st[pos_dfus[v]] - cc_dfus_st.mean())
                 + abs(cc_e3b_st[i] - cc_e3b_st.mean())
                 for i, v in enumerate(e3b_st["idx"])])
k = int(np.argmin(dist))
row = int(e3b_st["idx"][k])
seg = {"clean": y[row], "noisy": x[row],
       "EEGDfus": d_st["denoised"][pos_dfus[row]].astype(np.float64),
       "ours": np.asarray(e3b_st["denoised"][k], np.float64)}
seg_cc = {a: float(cc_rows(seg[a][None], seg["clean"][None])[0])
          for a in ("noisy", "EEGDfus", "ours")}

# ---------------------------------------------------------------- figure
fig = plt.figure(figsize=(figstyle.FULL, 4.9))
gs = fig.add_gridspec(2, 6, hspace=0.55, wspace=1.5,
                      left=0.105, right=0.99, top=0.955, bottom=0.09,
                      height_ratios=[1.0, 1.12])
axA = fig.add_subplot(gs[0, 0:2])
axB = fig.add_subplot(gs[0, 2:4])
axC = fig.add_subplot(gs[0, 4:6])
axD = fig.add_subplot(gs[1, 0:2])
axE = fig.add_subplot(gs[1, 2:6])

# ---------------- A/B: dumbbells released (open) -> strict (filled)
def dumbbell(ax, metric, xlim, pub_val, headline_fmt):
    ys = np.arange(len(ARMS))[::-1]
    for (name, col, d), yy in zip(ARMS, ys):
        r, s = d["rel"][metric], d["st"][metric]
        ax.plot([r, s], [yy, yy], color=col, lw=1.1, zorder=2)
        ax.scatter([r], [yy], s=22, facecolors="white", edgecolors=col,
                   lw=1.0, zorder=3)
        ax.scatter([s], [yy], s=22, facecolors=col, edgecolors=col, zorder=3)
    # headline numbers on the EEGDfus endpoints only
    r, s = ARMS[0][2]["rel"][metric], ARMS[0][2]["st"][metric]
    ax.text(r, ys[0] + 0.30, headline_fmt.format(r), ha="center", fontsize=5.5,
            color=ARMS[0][1])
    ax.text(s, ys[0] + 0.30, headline_fmt.format(s), ha="center", fontsize=5.5,
            color=ARMS[0][1])
    # published reference tick (from ssed_eval.json)
    ax.plot([pub_val], [ys[0] + 0.62], marker="v", ms=3.5, color=ARMS[0][1],
            clip_on=False)
    ax.text(pub_val, ys[0] + 0.95, f"pub {pub_val:g}", ha="center",
            fontsize=5.5, color=ARMS[0][1])
    ax.set_yticks(ys)
    ax.set_yticklabels([a[0] for a in ARMS])
    for tick, (_, col, _) in zip(ax.get_yticklabels(), ARMS):
        tick.set_color(col)
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.55, len(ARMS) - 0.30)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)


dumbbell(axA, "cc", (0.42, 1.01), PUB["cc"], "{:.3f}")
axA.set_xlabel("CC")
# identity baseline: CC(noisy, clean), recomputed on both test splits
for v in (id_cc["rel"], id_cc["st"]):
    axA.axvline(v, color=C["identity"], lw=0.7, ls=":", zorder=1)
axA.text(id_cc["st"] - 0.012, -0.50,
         f"identity {id_cc['rel']:.2f}/{id_cc['st']:.2f}",
         fontsize=5.5, color=C["identity"], ha="right", va="bottom")
axA.text(0.44, 3.62, "$\\circ$ released $\\rightarrow$ $\\bullet$ strict",
         fontsize=6, color="#2b2b2b")
figstyle.panel(axA, "A")

dumbbell(axB, "rrmse_t", (0.0, 1.12), PUB["rrmse_t"], "{:.3f}")
axB.set_xlabel("RRMSE$_t$")
axB.set_yticklabels([])
figstyle.panel(axB, "B")

# ---------------- C: delta/theta PLV, released (open) vs strict (filled)
plv_arms = [("EEGDfus", C["EEGDfus"], "eegdfus"),
            ("identity", C["identity"], "identity"),
            ("zero-shot", C["POP"], "e4zero"),
            ("SSED-tr.", C["ours"], "e3b"),
            ("fine-t.", C["NO_A0"], "e4ft")]
bw = 0.38
for bi, band in enumerate(("delta", "theta")):
    for ai, (name, col, key) in enumerate(plv_arms):
        xc = bi * (len(plv_arms) + 1.2) + ai
        r = plv[f"{key}/released_test"][band]
        s = plv[f"{key}/strict_test"][band]
        axC.bar(xc - bw / 2, r, bw, facecolor="white", edgecolor=col, lw=0.8)
        axC.bar(xc + bw / 2, s, bw, facecolor=col, edgecolor=col, lw=0.5)
centers = [1.0 * (len(plv_arms) - 1) / 2, (len(plv_arms) + 1.2) + (len(plv_arms) - 1) / 2]
axC.set_xticks(list(centers))
axC.set_xticklabels(["$\\delta$ 1–4 Hz", "$\\theta$ 4–8 Hz"])
axC.set_ylim(0, 1.02)
axC.set_ylabel("PLV")
axC.text(0.99, 1.005, "$\\alpha\\beta\\gamma\\geq$0.98 all arms (omitted)",
         transform=axC.transAxes, ha="right", va="bottom", fontsize=5,
         color="#7f7f7f")
axC.tick_params(axis="x", length=0)
figstyle.panel(axC, "C")

# ---------------- D: leak mechanism, per-row EEGDfus CC
rng = np.random.default_rng(0)
groups = [("in train", cc_dfus_rel[leak_mask], "open"),
          ("fresh", cc_dfus_rel[~leak_mask], "open"),
          ("strict", cc_dfus_st, "filled")]
col = C["EEGDfus"]
for gi, (name, vals, style) in enumerate(groups):
    xs = gi + rng.uniform(-0.22, 0.22, len(vals))
    if style == "open":
        axD.scatter(xs, vals, s=2.5, facecolors="none", edgecolors=col,
                    lw=0.35, alpha=0.35, zorder=2)
    else:
        axD.scatter(xs, vals, s=2.5, color=col, lw=0, alpha=0.25, zorder=2)
    m = vals.mean()
    axD.plot([gi - 0.30, gi + 0.30], [m, m], color="#2b2b2b", lw=1.1, zorder=3)
    axD.text(gi + 0.36, m, f"{m:.3f}", fontsize=5.5, va="center",
             color="#2b2b2b", zorder=4,
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.75,
                       pad=0.4))
axD.set_xticks(range(3))
axD.set_xticklabels([f"{n}\nn={len(v)}" for n, v, _ in groups])
axD.set_ylabel("per-row CC (EEGDfus)")
lo = min(v.min() for _, v, _ in groups)
axD.set_ylim(lo - 0.03, 1.06)
axD.set_xlim(-0.5, 2.75)
# bracket: the first two groups are the released test split
ylo = lo - 0.015
axD.plot([-0.35, 1.35], [ylo, ylo], color="#7f7f7f", lw=0.6, clip_on=False)
axD.text(0.5, ylo + 0.015, "released", ha="center", fontsize=5.5,
         color="#7f7f7f")
figstyle.panel(axD, "D")

# ---------------- E: representative strict segment, clean vs each arm
t = np.arange(seg["clean"].shape[-1]) / 200.0
amp = 1.4 * max(np.abs(seg[a]).max() for a in seg)
order = [("noisy", C["identity"]), ("EEGDfus", C["EEGDfus"]),
         ("ours", C["ours"])]
for si, (arm, col) in enumerate(order):
    off = -si * amp
    axE.plot(t, seg["clean"] + off, color=C["clean"], lw=0.6, zorder=2)
    axE.plot(t, seg[arm] + off, color=col, lw=0.8, alpha=0.9, zorder=3)
    axE.text(t[-1] + 0.02, off + 0.25 * amp, f"{arm}\nCC {seg_cc[arm]:.2f}",
             fontsize=5.5, color=col, va="center")
axE.text(t[-1] + 0.02, 0.68 * amp, "clean", fontsize=5.5, color=C["clean"],
         va="center")
axE.set_xlim(0, t[-1] + 0.32)
axE.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
axE.set_xlabel("time (s)")
axE.set_yticks([])
axE.spines["left"].set_visible(False)
figstyle.panel(axE, "E")

figstyle.save(fig, "fig-eegdfus-split")
