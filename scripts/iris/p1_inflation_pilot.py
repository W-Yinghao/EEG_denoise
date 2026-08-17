#!/usr/bin/env python3
"""IRIS P1 — covariance-inflation gate pilot (linear path, MobileBCI dev-15).

Preregistered in reports/iris_prereg_p1.md (frozen before execution; bar 0.30 priced
by K1 through the charter-frozen formula; drift term OFF per K2). Reuses the V44-S1
machinery verbatim: registries, episode sampler, drive convention, paired metrics.
CPU only — the linear subtraction path adjudicates the gate; no diffusion.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
V44_SRC = Path("/home/infres/yinwang/denoiseNet_rgcc_eog_v44/src")
OUT_DIR = REPO / "results/iris/p1"
SEEDS = (20261201, 20261202, 20261203)
EPISODES = 8
NEVER_WORSE_EPS = 0.005
HARM_MARGIN = 0.005
RECLAMATION_BAR = 0.30
VACUOUS_DENOM = 0.005
BOOT_SEED, BOOT_DRAWS = 420, 5000
sys.path.insert(0, str(V44_SRC))


def soft_lambda(cell) -> float:
    return float(np.clip(cell.tau2 / max(cell.tau2 + cell.within / 4.0, 1e-12),
                         0.0, 1.0))


def wiener_estimate(c_soft: np.ndarray, variance: np.ndarray,
                    drive: np.ndarray) -> np.ndarray:
    a_hat = c_soft @ drive
    var_t = variance @ (drive ** 2)
    num = np.mean(a_hat ** 2, axis=1)
    den = num + np.mean(var_t, axis=1)
    weights = num / np.maximum(den, 1e-18)
    return weights[:, None] * a_hat


def main() -> None:
    from eeg_scad.cli.run_v43 import configs
    from eeg_scad.data.artifact_transfer_v41r import TransferRegistry, TransferEpisodeSampler
    from eeg_scad.data.eb_transfer_v43 import EBTransferRegistry
    from eeg_scad.cli.run_v44_s2 import _posterior_variance
    from eeg_scad.evaluation.paired_metrics import paired_metrics

    data, folds, _ = configs()
    rows = []
    for fold in folds:
        fold_id = fold["fold"]
        registry30 = TransferRegistry(data, fold, 30, 0.05)
        eb120 = EBTransferRegistry(data, fold, registry30, 120)
        post_var = _posterior_variance(registry30, eb120, fold)
        for seed in SEEDS:
            sampler = TransferEpisodeSampler(data, fold, "test", seed + 3, registry30)
            bank = sampler.sample_balanced(EPISODES)
            wrongs = [sampler.condition_signature(meta, "WRONG")[1]
                      for meta in bank["meta"]]
            for clean, observed, artifact, meta, wrong in zip(
                    bank["x"], bank["y"], bank["artifact"], bank["meta"], wrongs):
                key = (meta["participant"], meta["session"], meta["task"])
                wrong_key = (wrong, meta["session"], meta["task"])
                cell = eb120.cells[key]
                wrong_cell = eb120.cells[wrong_key]
                pop = registry30.population_transfer[key[1:]]
                drive = np.linalg.pinv(registry30.cells[key].query_transfer) \
                    @ np.asarray(artifact, np.float64)

                lam = soft_lambda(cell)
                c_soft = pop + lam * (cell.transfer - pop)
                wrong_lam = soft_lambda(wrong_cell)
                wrong_soft = pop + wrong_lam * (wrong_cell.transfer - pop)
                arms = {
                    "BINARY": eb120.operator(*key, "EB") @ drive,
                    "INFLATION": wiener_estimate(c_soft, post_var[key], drive),
                    "INFLATION_SOFT": c_soft @ drive,
                    "POP": pop @ drive,
                    "NO_A0": np.zeros_like(np.asarray(artifact, np.float64)),
                    "ORACLE": registry30.cells[key].query_transfer @ drive,
                    "WRONG_binary": eb120.operator(*wrong_key, "EB") @ drive,
                    "WRONG_inflation": wiener_estimate(wrong_soft,
                                                       post_var[wrong_key], drive),
                }
                for arm, estimate in arms.items():
                    rows.append({
                        "fold": fold_id, "seed": seed, "participant": key[0],
                        "cell": "|".join(key), "condition": arm,
                        "abstained_binary": bool(cell.hard_gate),
                        "lambda_binary": float(cell.lam),
                        "lambda_soft": lam,
                        **paired_metrics(clean, observed, artifact, estimate)})
        print(json.dumps({"fold": fold_id, "rows": len(rows)}), flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "p1_rows.json").write_text(json.dumps(rows, indent=1) + "\n")

    # ---- frozen gates (prereg iris_prereg_p1.md)
    import pandas as pd
    frame = pd.DataFrame(rows)
    cell_means = frame.groupby(["cell", "condition"]).rrmse_temporal.mean().unstack()
    abstained_cells = sorted(frame[frame.abstained_binary].cell.unique())
    ab = cell_means.loc[abstained_cells]

    denom = float(ab["BINARY"].mean() - ab["ORACLE"].mean())
    numer = float(ab["BINARY"].mean() - ab["INFLATION"].mean())
    reclamation = numer / denom if abs(denom) > 1e-12 else float("nan")
    rng = np.random.default_rng(BOOT_SEED)
    draws = []
    per_cell = ab[["BINARY", "INFLATION", "ORACLE"]].to_numpy()
    for _ in range(BOOT_DRAWS):
        pick = per_cell[rng.integers(0, len(per_cell), len(per_cell))]
        d = pick[:, 0].mean() - pick[:, 2].mean()
        draws.append((pick[:, 0].mean() - pick[:, 1].mean()) / d
                     if abs(d) > 1e-12 else np.nan)
    draws = np.asarray([d for d in draws if np.isfinite(d)])
    vacuous = denom < VACUOUS_DENOM
    p1a_pass = bool((not vacuous) and reclamation >= RECLAMATION_BAR)

    violations = cell_means[cell_means["INFLATION"]
                            > cell_means["NO_A0"] + NEVER_WORSE_EPS]
    p1b_pass = bool(len(violations) == 0)

    harm = float((cell_means["WRONG_inflation"] - cell_means["POP"]).mean())
    p1c_pass = bool(harm <= HARM_MARGIN)

    ladder = ("ADOPTED" if (p1a_pass and p1b_pass and p1c_pass) else
              "HYBRID" if (p1b_pass and p1c_pass) else
              "DEAD_POINT_ESTIMATES")
    decision = {
        "prereg": "reports/iris_prereg_p1.md",
        "GATE_P1a_reclamation": {
            "abstained_cells": abstained_cells,
            "binary_mean": float(ab["BINARY"].mean()),
            "inflation_mean": float(ab["INFLATION"].mean()),
            "inflation_soft_mean": float(ab["INFLATION_SOFT"].mean()),
            "oracle_mean": float(ab["ORACLE"].mean()),
            "pop_mean": float(ab["POP"].mean()),
            "no_a0_mean": float(ab["NO_A0"].mean()),
            "denominator": denom, "vacuous": bool(vacuous),
            "reclamation": reclamation, "bar": RECLAMATION_BAR,
            "bootstrap_low": float(np.quantile(draws, .025)) if len(draws) else None,
            "bootstrap_high": float(np.quantile(draws, .975)) if len(draws) else None,
            "pass": p1a_pass},
        "GATE_P1b_never_worse": {
            "epsilon": NEVER_WORSE_EPS, "violations": int(len(violations)),
            "violating_cells": violations.index.tolist(), "pass": p1b_pass,
            "worst_excess": float((cell_means["INFLATION"] - cell_means["NO_A0"]).max())},
        "GATE_P1c_wrong_donor_harm": {
            "contrast": "mean(WRONG_inflation - POP)", "value": harm,
            "margin": HARM_MARGIN, "pass": p1c_pass,
            "binary_reference_banked": -0.000223,
            "wrong_binary_mean_minus_pop":
                float((cell_means["WRONG_binary"] - cell_means["POP"]).mean())},
        "ladder": ladder,
        "abstained_cell_detail": [
            {"cell": c,
             "lambda_soft": float(frame[frame.cell == c].lambda_soft.iloc[0]),
             "binary": float(ab.loc[c, "BINARY"]),
             "inflation": float(ab.loc[c, "INFLATION"]),
             "inflation_soft": float(ab.loc[c, "INFLATION_SOFT"]),
             "oracle": float(ab.loc[c, "ORACLE"]),
             "no_a0": float(ab.loc[c, "NO_A0"])} for c in abstained_cells],
    }
    (OUT_DIR / "p1_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"reclamation": round(reclamation, 4) if np.isfinite(reclamation)
                      else None, "P1a": p1a_pass, "P1b": p1b_pass, "P1c": p1c_pass,
                      "ladder": ladder}))


if __name__ == "__main__":
    main()
