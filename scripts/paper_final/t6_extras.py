#!/usr/bin/env python3
"""PAPER-FINAL T6.3 — width-locality scatter (CPU).

Per-cell propagation-width share (Var_op / (sigma^2 + Var_op), averaged over the
cell's episodes, channels and samples) against the calibration within-variance v_i,
from the banked IRIS-F4 K=32 arrays.  Episode-to-cell mapping is recovered by
re-instantiating the frozen episode sampler (deterministic seeds).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pf_common import ARRAYS, OUT, S1_SEEDS

DERIVED_F4 = Path("/projects/EEG-foundation-model/derived/denoiseNet/iris_f4")


def main() -> None:
    from eeg_scad.cli.run_v43 import configs
    from eeg_scad.data.artifact_transfer_v41r import (TransferEpisodeSampler,
                                                      TransferRegistry)
    from eeg_scad.data.eb_transfer_v43 import EBTransferRegistry

    data, folds, _ = configs()
    points = []
    for fold in folds:
        fold_id = fold["fold"]
        registry30 = TransferRegistry(data, fold, 30, .05)
        eb120 = EBTransferRegistry(data, fold, registry30, 120)
        for seed in S1_SEEDS:
            sampler = TransferEpisodeSampler(data, fold, "test", seed + 3, registry30)
            bank = sampler.sample_balanced(8)
            keys = [(m["participant"], m["session"], m["task"]) for m in bank["meta"]]
            d = np.load(DERIVED_F4 / f"fold_{fold_id}_seed_{seed}.npz",
                        allow_pickle=False)
            stored = [str(p) for p in d["participants"]]
            if stored != [k[0] for k in keys]:
                raise SystemExit(f"participant order mismatch fold {fold_id} "
                                 f"seed {seed} — sampler reconstruction failed")
            sigma = d["sigma"].astype(np.float64)
            var_op = d["var_op"].astype(np.float64)
            share = var_op / (sigma ** 2 + var_op)
            per_cell: dict[tuple, list[float]] = {}
            for episode, key in enumerate(keys):
                per_cell.setdefault(key, []).append(float(share[episode].mean()))
            for key, values in per_cell.items():
                points.append({"fold": fold_id, "seed": seed, "cell": "|".join(key),
                               "participant": key[0],
                               "within_v": float(eb120.cells[key].within),
                               "hard_gate": int(eb120.cells[key].hard_gate),
                               "lambda": float(eb120.cells[key].lam),
                               "propagation_width_share": float(np.mean(values))})
        print(json.dumps({"fold": fold_id, "cells": len(points)}), flush=True)
    (OUT / "t6_width_locality.json").write_text(json.dumps(
        {"points": points}, indent=1, sort_keys=True) + "\n")
    np.savez_compressed(
        ARRAYS / "t6_width_locality.npz",
        within_v=np.asarray([p["within_v"] for p in points]),
        propagation_width_share=np.asarray([p["propagation_width_share"]
                                            for p in points]),
        hard_gate=np.asarray([p["hard_gate"] for p in points]),
        lam=np.asarray([p["lambda"] for p in points]),
        cell=np.asarray([p["cell"] for p in points]))
    corr = np.corrcoef([p["within_v"] for p in points],
                       [p["propagation_width_share"] for p in points])[0, 1]
    print(json.dumps({"points": len(points), "pearson_r": float(corr)}))


if __name__ == "__main__":
    main()
