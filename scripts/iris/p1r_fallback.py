#!/usr/bin/env python3
"""IRIS P1R — fallback repair + vs-incumbent gate comparison (MobileBCI dev-15).

Preregistered in reports/iris_prereg_p1r_trepair.md (frozen before execution).
Machinery identical to P1 (V44-S1 verbatim); new arms only. The banked P1 verdict
is not revisited. CPU only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/iris"))
from p1_inflation_pilot import (  # noqa: E402
    BOOT_DRAWS, BOOT_SEED, EPISODES, HARM_MARGIN, NEVER_WORSE_EPS, SEEDS, V44_SRC,
    soft_lambda, wiener_estimate)

OUT_DIR = REPO / "results/iris/p1r"
sys.path.insert(0, str(V44_SRC))


def _boot(values: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(BOOT_SEED)
    draws = np.asarray([rng.choice(values, len(values), replace=True).mean()
                        for _ in range(BOOT_DRAWS)])
    return float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


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
                artifact64 = np.asarray(artifact, np.float64)
                drive = np.linalg.pinv(registry30.cells[key].query_transfer) @ artifact64

                lam = soft_lambda(cell)
                c_soft = pop + lam * (cell.transfer - pop)
                wrong_lam = soft_lambda(wrong_cell)
                wrong_soft = pop + wrong_lam * (wrong_cell.transfer - pop)
                zero = np.zeros_like(artifact64)
                binary = eb120.operator(*key, "EB") @ drive
                inflation = wiener_estimate(c_soft, post_var[key], drive)
                wrong_binary = eb120.operator(*wrong_key, "EB") @ drive
                arms = {
                    "BINARY": binary,
                    "INFLATION": inflation,
                    "BINARY_NOA0FB": zero if cell.hard_gate else binary,
                    "INFLATION_NOA0FB": zero if cell.hard_gate else inflation,
                    "POP": pop @ drive,
                    "NO_A0": zero,
                    "ORACLE": registry30.cells[key].query_transfer @ drive,
                    "WRONG_binary": wrong_binary,
                    "WRONG_binary_NOA0FB":
                        zero if wrong_cell.hard_gate else wrong_binary,
                    "WRONG_inflation": wiener_estimate(wrong_soft,
                                                       post_var[wrong_key], drive),
                }
                for arm, estimate in arms.items():
                    rows.append({"fold": fold_id, "seed": seed,
                                 "participant": key[0], "cell": "|".join(key),
                                 "condition": arm,
                                 "abstained_binary": bool(cell.hard_gate),
                                 **paired_metrics(clean, observed, artifact, estimate)})
        print(json.dumps({"fold": fold_id, "rows": len(rows)}), flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "p1r_rows.json").write_text(json.dumps(rows, indent=1) + "\n")

    import pandas as pd
    frame = pd.DataFrame(rows)
    cm = frame.groupby(["cell", "condition"]).rrmse_temporal.mean().unstack()
    abstained = sorted(frame[frame.abstained_binary].cell.unique())
    ab = cm.loc[abstained]

    paired_a = (ab["BINARY"] - ab["BINARY_NOA0FB"]).to_numpy()
    lo_a, hi_a = _boot(paired_a)
    a_pass = bool(paired_a.mean() > 0 and lo_a > 0)

    viol = cm[cm["INFLATION_NOA0FB"] > cm["BINARY_NOA0FB"] + NEVER_WORSE_EPS]
    b_pass = bool(len(viol) == 0)

    paired_c = (cm["BINARY_NOA0FB"] - cm["INFLATION_NOA0FB"]).to_numpy()
    lo_c, hi_c = _boot(paired_c)
    c_pass = bool(lo_c > -NEVER_WORSE_EPS)

    harm_inf = float((cm["WRONG_inflation"] - cm["POP"]).mean())
    harm_bin = float((cm["WRONG_binary_NOA0FB"] - cm["POP"]).mean())
    d_pass = bool(harm_inf <= HARM_MARGIN and harm_bin <= HARM_MARGIN)

    adopted = ("INFLATION_NOA0FB" if (a_pass and b_pass and c_pass and d_pass) else
               "BINARY_NOA0FB" if (a_pass and d_pass) else
               "BINARY_incumbent_unchanged")
    decision = {
        "prereg": "reports/iris_prereg_p1r_trepair.md (P1R)",
        "G_P1R_a_fallback_repair": {
            "abstained_cells": abstained, "n": len(abstained),
            "paired_mean": float(paired_a.mean()), "ci_low": lo_a, "ci_high": hi_a,
            "binary_mean": float(ab["BINARY"].mean()),
            "binary_noa0fb_mean": float(ab["BINARY_NOA0FB"].mean()),
            "pass": a_pass},
        "G_P1R_b_relative_never_worse": {
            "epsilon": NEVER_WORSE_EPS, "violations": int(len(viol)),
            "violating_cells": viol.index.tolist(), "pass": b_pass},
        "G_P1R_c_pooled": {
            "contrast": "BINARY_NOA0FB - INFLATION_NOA0FB (positive favors inflation)",
            "mean": float(paired_c.mean()), "ci_low": lo_c, "ci_high": hi_c,
            "pass": c_pass,
            "gain_sized": bool(lo_c > 0)},
        "G_P1R_d_harm": {"wrong_inflation_minus_pop": harm_inf,
                         "wrong_binary_noa0fb_minus_pop": harm_bin,
                         "margin": HARM_MARGIN, "pass": d_pass},
        "adopted_gate_for_fights": adopted,
        "pooled_means": {a: float(cm[a].mean()) for a in
                         ("BINARY", "BINARY_NOA0FB", "INFLATION",
                          "INFLATION_NOA0FB", "POP", "NO_A0")},
    }
    (OUT_DIR / "p1r_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"a": a_pass, "b": b_pass, "c": c_pass, "d": d_pass,
                      "adopted": adopted,
                      "fallback_gain": round(float(paired_a.mean()), 4),
                      "pooled_gain": round(float(paired_c.mean()), 4)}))


if __name__ == "__main__":
    main()
