#!/usr/bin/env python3
"""IRIS K2 — Σ_drift prior-predictive recheck under corrected accounting.

Preregistered in reports/iris_prereg_k.md (frozen before execution). WAVE2's G0 failed
at coverage 0.9267 (> 0.90: prior too wide) using Σ_drift = raw cross-cell variance of
(C_query − C_support), which CONTAINS both operators' estimation-noise floors. The
corrected object subtracts the per-coefficient floors (within/4 each side, the same
estimator class the banked debiased reading D_deb = D_raw − W/4 − W/4 used) and re-runs
the identical coverage instrument. Gate (frozen): corrected mean coverage in
[0.70, 0.90] → drift term candidate-ON inside the IRIS inflation gate; outside → OFF.

The banked gibbs_g0.json (0.9267) is reported alongside and never edited.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
V44_SRC = Path("/home/infres/yinwang/denoiseNet_rgcc_eog_v44/src")
BANKED_G0 = Path("/home/infres/yinwang/denoiseNet_wave2/results/wave2/gibbs_g0.json")
BANKED_SIGMA = Path("/home/infres/yinwang/denoiseNet_wave2/results/wave2/sigma_drift.npz")
OUT = REPO / "results/iris/k/k2_sigma_drift.json"
NOMINAL, BAND = 0.80, (0.70, 0.90)
sys.path.insert(0, str(V44_SRC))


def main() -> None:
    from scipy import stats
    from eeg_scad.cli.run_v43 import configs
    from eeg_scad.data.artifact_transfer_v41r import TransferRegistry
    from eeg_scad.data.eb_transfer_v43 import EBTransferRegistry
    from eeg_scad.cli.run_v44_s2 import _posterior_variance

    data, folds, _ = configs()

    # Pass 1: rebuild the drift vectors exactly as the banked shared layer did,
    # additionally collecting the per-coefficient estimation floors.
    drift_vectors, floors = [], []
    registries = []
    for fold in folds:
        registry30 = TransferRegistry(data, fold, 30, 0.05)
        eb120 = EBTransferRegistry(data, fold, registry30, 120)
        registries.append((fold, registry30, eb120))
        dev = set(fold["train"] + fold["validation"] + fold["test"])
        for key in sorted(eb120.cells):
            if key[0] not in dev:
                continue
            cell = eb120.cells[key]
            c_query = registry30.cells[key].query_transfer
            drift_vectors.append((c_query - cell.transfer).reshape(-1))
            within_rows = np.asarray(cell.within_rows, float)
            per_coeff = np.broadcast_to(
                within_rows.reshape(cell.transfer.shape[0], -1),
                cell.transfer.shape).reshape(-1)
            # support floor within/4 + query floor within/4 (same estimator class,
            # matching the banked debiased reading D_deb = D_raw − W/4 − W/4)
            floors.append(per_coeff / 4.0 + per_coeff / 4.0)

    raw_var = np.var(np.stack(drift_vectors), axis=0, ddof=1)
    banked_sigma = np.load(BANKED_SIGMA)["sigma_drift"].reshape(-1)
    reproduction_max_abs = float(np.max(np.abs(raw_var - banked_sigma)))
    mean_floor = np.mean(np.stack(floors), axis=0)
    sigma_deb = np.clip(raw_var - mean_floor, 0.0, None)
    shape = (46, 2)
    removed_fraction = float(1.0 - sigma_deb.sum() / max(raw_var.sum(), 1e-18))
    zeroed = float((sigma_deb == 0.0).mean())

    # Pass 2: the identical G0 coverage instrument, corrected Σ_drift.
    def coverage(sigma_flat: np.ndarray) -> tuple[float, dict[str, float]]:
        sigma = sigma_flat.reshape(shape)
        z = stats.norm.ppf(0.5 + NOMINAL / 2.0)
        cov: dict[str, list] = {}
        for fold, registry30, eb120 in registries:
            post_var = _posterior_variance(registry30, eb120, fold)
            for key in sorted(eb120.cells):
                if key not in post_var:
                    continue
                c_query = registry30.cells[key].query_transfer
                center = eb120.cells[key].transfer
                sd = np.sqrt(post_var[key] + sigma)
                inside = np.abs(c_query - center) <= z * sd
                cov.setdefault(key[0], []).append(float(inside.mean()))
        per = {p: float(np.mean(v)) for p, v in cov.items()}
        return float(np.mean(list(per.values()))), per

    corrected_mean, corrected_per = coverage(sigma_deb)
    raw_mean, _ = coverage(raw_var)
    banked = json.loads(BANKED_G0.read_text())
    gate_pass = bool(BAND[0] <= corrected_mean <= BAND[1])

    payload = {
        "prereg": "reports/iris_prereg_k.md (K2)",
        "banked_g0": {"coverage_mean": banked["coverage_mean"], "pass": banked["pass"]},
        "raw_sigma_reproduction": {
            "max_abs_diff_vs_banked_npz": reproduction_max_abs,
            "raw_coverage_recomputed": raw_mean},
        "correction": {
            "rule": "sigma_deb = clip(var_cells(C_query − C_support) − within/4 − within/4, 0)",
            "removed_variance_fraction": removed_fraction,
            "coefficients_zeroed_fraction": zeroed,
            "n_cells": len(drift_vectors)},
        "corrected": {"coverage_mean": corrected_mean,
                      "per_participant": corrected_per},
        "gate": {"band": list(BAND), "nominal": NOMINAL, "pass": gate_pass,
                 "verdict": ("drift term candidate-ON in the IRIS inflation gate"
                             if gate_pass else
                             "drift term OFF — fallback-ladder trigger only")},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"corrected_coverage": round(corrected_mean, 4),
                      "raw_coverage": round(raw_mean, 4),
                      "removed_fraction": round(removed_fraction, 4),
                      "gate_pass": gate_pass}))


if __name__ == "__main__":
    main()
