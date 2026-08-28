#!/usr/bin/env python3
"""PAPER-FINAL T1 (part A) — freeze ONE scalar temperature from the development set.

Grid rule as in IRIS F4 (0.50-6.00 step 0.05, smallest value reaching 80% mean
coverage), but applied to ALL 15 dev fold-seed cells JOINTLY (no leave-one-fold-out),
per SERVER_INSTRUCTIONS_PAPER_FINAL_RUNS.md T1.  Computed from the banked IRIS-F4
K=32 chain arrays; committed BEFORE any held-out inference.  Temperatures are frozen
for both scoring policies used in T1: INFL (operator-posterior inflation,
width = s*sqrt(sigma^2 + var_op)) and TEMP (width = s*sigma).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DERIVED = Path("/projects/EEG-foundation-model/derived/denoiseNet/iris_f4")
OUT = REPO / "results/paper_final"
SEEDS = (20261201, 20261202, 20261203)
Z80 = 1.2815515655446004
TEMP_GRID = np.arange(0.5, 6.0 + 1e-9, 0.05)


def main() -> None:
    units = []
    for fold in range(5):
        for seed in SEEDS:
            d = np.load(DERIVED / f"fold_{fold}_seed_{seed}.npz", allow_pickle=False)
            units.append({"errors": d["errors"].astype(np.float64),
                          "sigma": d["sigma"].astype(np.float64),
                          "var_op": d["var_op"].astype(np.float64)})
    temps = {}
    for policy in ("INFL", "TEMP"):
        for s in TEMP_GRID:
            covs = []
            for u in units:
                base = (np.sqrt(u["sigma"] ** 2 + u["var_op"])
                        if policy == "INFL" else u["sigma"])
                covs.append(float(np.mean(u["errors"] <= Z80 * s * base)))
            if np.mean(covs) >= 0.80:
                temps[policy] = float(round(s, 2))
                break
        else:
            temps[policy] = float(TEMP_GRID[-1])
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {"rule": "smallest s in arange(0.50,6.00,0.05) with mean 80% coverage "
                       ">= 0.80 over all 15 dev fold-seed cells jointly",
               "source": "banked IRIS-F4 K=32 arrays (derived/denoiseNet/iris_f4)",
               "z80": Z80, "temperatures": temps,
               "frozen_before_heldout_inference": True}
    (OUT / "t1_temperature.json").write_text(json.dumps(payload, indent=2,
                                                        sort_keys=True) + "\n")
    print(json.dumps(payload["temperatures"]))


if __name__ == "__main__":
    main()
