#!/usr/bin/env python3
"""IRIS K1 — abstention-cause taxonomy of the banked gate-shrinkage row.

Preregistered in reports/iris_prereg_k.md (frozen before execution). Re-derives the
gate state of the same 93 cells WAVE3-T4 priced (fold-0 registry, 120-s EB states) and
attributes each cell's shrinkage mass to a cause class. The banked T4 row
(mean 0.13707) is the denominator and is never edited; the banked slope is reused.

Cause classes (prereg):
  R  reliability-recoverable  = within-outlier hard-gate (support >= minimum) OR
                                active lambda-shrinkage — what covariance inflation
                                could convert IN PRINCIPLE
  H  support-floor hard gate  = effective support below the hard minimum (structural;
                                the prereg's "10-s" phrasing echoed the digest's
                                duration-cell language — the deployed constant on this
                                cohort is HARD_GATE_MIN_SECONDS = 60 s; the class is
                                defined by cause, not by the constant)
  N  no-reference             = reference channel absent (vacuous on MobileBCI)
  D  identity-hazard          = wrong-context cells (vacuous: all 93 are own-context)

f_conv = mass(R) / total; P1's reclamation bar = min(0.30, 0.75 * f_conv) per charter.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
V44_SRC = Path("/home/infres/yinwang/denoiseNet_rgcc_eog_v44/src")
WAVE3_T4 = Path("/home/infres/yinwang/denoiseNet_wave3/results/wave3/t4_gate_shrinkage.json")
CROSS_PANEL = {
    "w3_transport": Path("/home/infres/yinwang/denoiseNet_flagship_m0/results/"
                         "flagship_m13/w3_transport/decision.json"),
    "m35_u1": Path("/home/infres/yinwang/denoiseNet_flagship_m0/results/"
                   "flagship_m35/u1_factorial/decision.json"),
    "v43_s3c": Path("/home/infres/yinwang/denoiseNet_rgcc_v43/results/rgcc_v43/"
                    "stage3c_crosspanel/crosspanel_floor.json"),
}
OUT = REPO / "results/iris/k/k1_abstention_taxonomy.json"
sys.path.insert(0, str(V44_SRC))


def main() -> None:
    from eeg_scad.cli.run_v43 import configs
    from eeg_scad.data.artifact_transfer_v41r import TransferRegistry
    from eeg_scad.data.eb_transfer_v43 import EBTransferRegistry, HARD_GATE_MIN_SECONDS

    banked = json.loads(WAVE3_T4.read_text())
    slope = banked["operator_to_rrmse_slope"]

    data, folds, _ = configs()
    registry = TransferRegistry(data, folds[0], 30, 0.05)
    keys = sorted(registry.cells)
    eb = EBTransferRegistry(data, folds[0], registry, 120)
    rms = lambda m: float(np.sqrt(np.mean(np.square(m))))  # noqa: E731 - T4 verbatim

    cells = []
    for key in keys:
        cell = eb.cells[key]
        pop = registry.population_transfer[key[1:]]
        gated = pop + cell.lam * (cell.transfer - pop)
        mass = slope * rms(gated - cell.transfer)
        if cell.hard_gate and cell.effective_seconds < HARD_GATE_MIN_SECONDS:
            cause = "H_support_floor"
        elif cell.hard_gate:
            cause = "R_hard_within_outlier"
        else:
            cause = "R_active_shrinkage"
        cells.append({"cell": "|".join(key), "participant": key[0],
                      "lambda": float(cell.lam), "hard_gate": bool(cell.hard_gate),
                      "effective_seconds": float(cell.effective_seconds),
                      "within": float(cell.within), "tau2": float(cell.tau2),
                      "mass_rrmse": mass, "cause": cause})

    total = float(np.mean([c["mass_rrmse"] for c in cells]))
    banked_total = banked["shrink_to_pop_deployed"]["mean"]
    discrepancy = abs(total - banked_total) / banked_total
    by_cause: dict[str, dict] = {}
    for cause in ("R_hard_within_outlier", "R_active_shrinkage", "H_support_floor",
                  "N_no_reference", "D_identity_hazard"):
        sub = [c for c in cells if c["cause"] == cause]
        by_cause[cause] = {"cells": len(sub),
                           "mass_mean_contribution":
                               float(np.sum([c["mass_rrmse"] for c in sub]) / len(cells)),
                           "mass_share": (float(np.sum([c["mass_rrmse"] for c in sub])
                                                / (total * len(cells)))
                                          if total > 0 else 0.0)}
    denominator = banked_total if discrepancy > 0.10 else total
    r_mass = sum(c["mass_rrmse"] for c in cells
                 if c["cause"].startswith("R_")) / len(cells)
    f_conv = float(r_mass / denominator)
    abstained = [c for c in cells if c["hard_gate"]]

    cross = {}
    for name, path in CROSS_PANEL.items():
        try:
            payload = json.loads(path.read_text())
            text = json.dumps(payload)
            cross[name] = {"source": str(path),
                           "abstention_mentions":
                               {k: payload[k] for k in payload
                                if "abstain" in k.lower() or "abstention" in k.lower()},
                           "has_abstention_language": ("abstain" in text.lower())}
        except Exception as error:                      # noqa: BLE001 - reason-coded
            cross[name] = {"source": str(path), "error": str(error)}

    payload = {
        "prereg": "reports/iris_prereg_k.md (K1)",
        "banked_row": {"total_mean": banked_total,
                       "abstained_fraction": banked["shrink_to_pop_split"]["abstained_fraction"],
                       "abstained_cells_mean": banked["shrink_to_pop_split"]["abstained_cells_mean"],
                       "active_cells_mean": banked["shrink_to_pop_split"]["active_cells_mean"]},
        "reconstruction": {"total_mean": total, "n_cells": len(cells),
                           "discrepancy_vs_banked": discrepancy,
                           "validity_pass": bool(discrepancy <= 0.10),
                           "denominator_used": denominator},
        "slope_reused_from_banked": slope,
        "hard_gate_min_seconds_deployed": HARD_GATE_MIN_SECONDS,
        "taxonomy": by_cause,
        "abstained_cells": [{k: c[k] for k in
                             ("cell", "lambda", "effective_seconds", "within",
                              "mass_rrmse", "cause")} for c in abstained],
        "f_conv": f_conv,
        "p1_reclamation_bar": float(min(0.30, 0.75 * f_conv)),
        "cross_panel_companion_descriptive": cross,
        "cells": cells,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"f_conv": round(f_conv, 4),
                      "p1_bar": round(payload["p1_reclamation_bar"], 4),
                      "abstained": len(abstained),
                      "validity": payload["reconstruction"]["validity_pass"]}))


if __name__ == "__main__":
    main()
