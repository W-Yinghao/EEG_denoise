#!/usr/bin/env python3
"""fig-estimators — point-estimator family per participant (v3, single column).

Slope-graph over the five point-estimator arms (G4 gap-fill bank), ordered by
pooled participant-first mean rrmse_temporal:
  matched diffusion (MATCH_gated)      0.431
  calibrated linear (C_gated)          0.435
  deterministic twin (DET_MATCH_gated) 0.497
  det. population-retrained (DET_POP)  0.504
  diffusion population-retr. (POP)     0.526
One thin gray line per dev participant (15), colored dots at each arm,
pooled participant-first means as large colored markers on a connecting line
with their values printed — the near-parity of matched diffusion and
calibrated linear (0.431 vs 0.435, linear better for 9/15 participants) is
the honesty headline and is labeled from runtime-computed numbers only.
The first three arms share the V44 episode banks and noise seeds (exactly
paired); the two population-retrained arms follow the V43-S2 EB120 protocol
(same protocol class, not episode-identical) — those cross-campaign segments
are drawn dotted and separated by a light vertical rule. Heavy-tail
participant sub-09 shown as-is and labeled. No CIs/error bars anywhere
(house rule); log y.

Every number is read at runtime from
  results/paper_final/gapfill/g4_estimators.json
and the per-participant means are rebuilt from the per-episode rows
(participant-first, zero_artifact included, per the bank's note) and asserted
against BOTH the banked per_participant_means and the pooled_check scalars
before anything is drawn.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/infres/yinwang/denoiseNet/scripts/paper_final/figures")
import figstyle
from figstyle import C

figstyle.setup()

import matplotlib.pyplot as plt

REPO = Path("/home/infres/yinwang/denoiseNet")
G4 = json.loads((REPO / "results/paper_final/gapfill/g4_estimators.json").read_text())

# ---------------------------------------------------------------- arms
ARMS = [  # (key, short label, color, filled?)  ordered by pooled mean
    ("diffusion_matched",       "matched\ndiff.",  C["MATCH"],   True),
    ("linear_calibrated",       "calibr.\nlinear", C["LINEAR"],  True),
    ("det_twin_matched",        "det.\ntwin",      C["DET_ENS"], True),
    ("det_pop_retrained",       "det.\npop",       C["DET_ENS"], False),
    ("diffusion_pop_retrained", "diff.\npop",      C["POP"],     True),
]
N_V44 = 3  # first three arms share the V44 episode banks (exact pairing)

# ------------------------------------------------- rebuild + assert
rows = G4["rows"]
parts = sorted({r["participant"] for r in rows})
assert len(parts) == 15, parts

per_part = {}   # arm -> np.array over parts (participant-first episode mean)
pooled = {}     # arm -> participant-first pooled mean
for key, *_ in ARMS:
    arm_rows = [r for r in rows if r["arm"] == key]
    chk = G4["pooled_check"][key]
    assert chk["match"] is True, key
    assert len(arm_rows) == chk["n_rows"], (key, len(arm_rows))
    v = np.array([np.mean([r["rrmse_temporal"] for r in arm_rows
                           if r["participant"] == p]) for p in parts])
    banked = np.array([G4["per_participant_means"][key][p] for p in parts])
    assert np.allclose(v, banked, atol=1e-12), key
    pooled[key] = float(v.mean())
    assert abs(pooled[key] - chk["reproduced_participant_first_mean"]) < 1e-12
    assert abs(pooled[key] - chk["quoted_in_figure_plan"]) < 5e-4, key
    per_part[key] = v

k_lin = int(np.sum(per_part["linear_calibrated"]
                   < per_part["diffusion_matched"]))

# ---------------------------------------------------------------- draw
fig, ax = plt.subplots(figsize=(figstyle.HALF, 2.75))
X = np.arange(len(ARMS))
mat = np.column_stack([per_part[k] for k, *_ in ARMS])  # (15, 5)

for i, p in enumerate(parts):  # per-participant profiles
    y = mat[i]
    ax.plot(X[:N_V44], y[:N_V44], color="0.45", lw=0.6, alpha=0.6, zorder=2)
    ax.plot(X[N_V44 - 1:], y[N_V44 - 1:], color="0.45", lw=0.6, alpha=0.6,
            ls=(0, (1.5, 1.5)), zorder=2)
for j, (key, _lab, col, filled) in enumerate(ARMS):
    ax.plot(np.full(15, X[j]), mat[:, j], ls="none", marker="o", ms=2.0,
            mfc=col if filled else "white", mec=col, mew=0.5, zorder=3)

pm = np.array([pooled[k] for k, *_ in ARMS])  # pooled means, on top
ax.plot(X[:N_V44], pm[:N_V44], color="0.15", lw=1.4, zorder=4)
ax.plot(X[N_V44 - 1:], pm[N_V44 - 1:], color="0.15", lw=1.4,
        ls=(0, (1.5, 1.5)), zorder=4)
for j, (key, _lab, col, filled) in enumerate(ARMS):
    ax.plot(X[j], pm[j], marker="o", ms=5.5, mfc=col if filled else "white",
            mec=col, mew=1.1, zorder=5)
    ax.annotate(f"{pm[j]:.3f}", (X[j], pm[j]), xytext=(0, -8),
                textcoords="offset points", ha="center", va="top",
                fontsize=6, color=col, fontweight="bold", zorder=6)

# honesty headline: diffusion ~ linear (runtime numbers only)
ax.text(0.5, 1.7, f"$\\Delta$={pm[1] - pm[0]:+.3f}\nlinear better {k_lin}/15",
        ha="center", va="center", fontsize=6, color="0.35", zorder=6)

# campaign split: V44 paired arms | V43-S2 protocol arms
ax.axvline(N_V44 - 0.5, color="0.8", lw=0.6, ls=":", zorder=1)
ax.text((N_V44 - 1) / 2, 1.02, "V44, paired episodes",
        transform=ax.get_xaxis_transform(), ha="center", va="bottom",
        fontsize=5.5, color="0.45")
ax.text((N_V44 - 0.5 + len(ARMS) - 1) / 2, 1.02, "V43-S2 protocol",
        transform=ax.get_xaxis_transform(), ha="center", va="bottom",
        fontsize=5.5, color="0.45")

i09 = parts.index("sub-09")  # heavy tail, shown as-is
ax.annotate("sub-09", (X[-1], mat[i09, -1]), xytext=(3, 3),
            textcoords="offset points", fontsize=5.5, color="0.45")

ax.set_yscale("log")
ax.set_yticks([0.05, 0.1, 0.2, 0.5, 1, 2, 4])
ax.set_yticklabels(["0.05", "0.1", "0.2", "0.5", "1", "2", "4"])
ax.set_ylabel("rRMSE (temporal), participant mean")
ax.set_xticks(X)
ax.set_xticklabels([lab for _k, lab, *_ in ARMS])
for tick, (_k, _lab, col, _f) in zip(ax.get_xticklabels(), ARMS):
    tick.set_color(col)
ax.set_xlim(-0.4, len(ARMS) - 0.55)
ax.tick_params(axis="x", length=0)

fig.tight_layout()
figstyle.save(fig, "fig-estimators")
