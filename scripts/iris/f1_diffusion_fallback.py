#!/usr/bin/env python3
"""IRIS F1 — adopted-fallback fight on the deployed diffusion class (MobileBCI).

Preregistered in reports/iris_prereg_f.md (frozen before execution). V44-S1
evaluation machinery verbatim; four arms; bit-identity guard on non-hard-gated cells.
Incremental per-fold-seed outputs (resume = skip existing). `aggregate` mode applies
the frozen gates.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
V44_ROOT = Path("/home/infres/yinwang/denoiseNet_rgcc_eog_v44")
sys.path.insert(0, str(V44_ROOT / "src"))
OUT_DIR = REPO / "results/iris/f1"
SEEDS = (20261201, 20261202, 20261203)
BOOT_SEED, BOOT_DRAWS = 420, 5000
BANKED_DEV_ANCHOR = 0.1428
ARMS = ("MATCH_gated", "MATCH_NOA0FB", "NO_A0", "POP")


def run() -> None:
    import torch
    from eeg_scad.cli.run_v43 import configs
    from eeg_scad.cli.run_v44 import (_bank_drives, _gated_assets, _natural_metrics,
                                      _natural_windows, natural_noise_seed,
                                      noise_seed, sample_bank_eog)
    from eeg_scad.data.artifact_transfer_v41r import (TransferEpisodeSampler,
                                                      TransferRegistry)
    from eeg_scad.data.eb_transfer_v43 import EBTransferRegistry
    from eeg_scad.evaluation.paired_metrics import paired_metrics
    from eeg_scad.models.calib_saddpm_cond_v42r import LinearX0Schedule
    from eeg_scad.models.calib_saddpm_eog_v44 import CalibSADDPMEOG

    data, folds, _ = configs()
    device = torch.device("cuda")
    schedule = LinearX0Schedule().to(device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        fold_id = fold["fold"]
        registry30 = TransferRegistry(data, fold, 30, 0.05)
        eb120 = EBTransferRegistry(data, fold, registry30, 120)
        assets = _gated_assets(registry30, eb120)
        for seed in SEEDS:
            out_path = OUT_DIR / f"fold_{fold_id}_seed_{seed}.json"
            if out_path.is_file():
                print(json.dumps({"fold": fold_id, "seed": seed, "skipped": True}),
                      flush=True)
                continue
            source = json.loads((V44_ROOT / "results/rgcc_eog_v44/stage1" /
                                 f"fold_{fold_id}_seed_{seed}" /
                                 "train_curve.json").read_text())
            model = CalibSADDPMEOG().to(device)
            model.load_state_dict(torch.load(source["checkpoint"],
                                             map_location=device,
                                             weights_only=False)["ema"])
            sampler = TransferEpisodeSampler(data, fold, "test", seed + 3, registry30)
            bank = sampler.sample_balanced(8)
            drives = _bank_drives(assets, bank)
            keys = [(m["participant"], m["session"], m["task"])
                    for m in bank["meta"]]
            hard = [bool(eb120.cells[k].hard_gate) for k in keys]

            outputs = {}
            for arm in ARMS:
                a0_stack, sig_stack = [], []
                for key, drive, is_hard in zip(keys, drives, hard):
                    asset = assets[key]
                    if arm == "MATCH_gated":
                        a0, sig = asset["C_gated"] @ drive, asset["sig_gated"]
                    elif arm == "MATCH_NOA0FB":
                        a0 = (np.zeros((len(asset["C0"]), drive.shape[1]))
                              if is_hard else asset["C_gated"] @ drive)
                        sig = asset["sig_gated"]
                    elif arm == "NO_A0":
                        a0, sig = np.zeros((len(asset["C0"]), drive.shape[1])), \
                            asset["sig_gated"]
                    else:
                        a0, sig = asset["C0"] @ drive, asset["sig_pop"]
                    a0_stack.append(a0)
                    sig_stack.append(sig)
                outputs[arm] = sample_bank_eog(model, schedule, bank["y"],
                                               np.stack(a0_stack),
                                               np.stack(sig_stack), device,
                                               noise_seed(fold_id, seed))
            # frozen correctness guard: bit-identity off the hard-gated cells
            for episode, is_hard in enumerate(hard):
                if not is_hard and not np.array_equal(
                        outputs["MATCH_gated"][episode],
                        outputs["MATCH_NOA0FB"][episode]):
                    raise AssertionError(
                        f"bit-identity violated on non-hard cell episode {episode} "
                        f"fold {fold_id} seed {seed} — instrument defect, stopping")

            rows = []
            for arm in ARMS:
                for episode, (clean, observed, artifact, meta, key, is_hard) in \
                        enumerate(zip(bank["x"], bank["y"], bank["artifact"],
                                      bank["meta"], keys, hard)):
                    prediction = outputs[arm][episode]
                    if not np.isfinite(prediction).all():
                        raise FloatingPointError("nonfinite F1 output")
                    rows.append({"fold": fold_id, "seed": seed,
                                 "participant": key[0], "cell": "|".join(key),
                                 "condition": arm, "hard_gated": is_hard,
                                 **paired_metrics(clean, observed, artifact,
                                                  observed - prediction)})

            natural_rows = []
            hard_test_keys = sorted({k for k, h in zip(keys, hard) if h})
            for key in hard_test_keys:
                windows = list(_natural_windows(registry30, data, key))
                y_stack = np.stack([w[1] for w in windows])
                for arm in ("MATCH_gated", "MATCH_NOA0FB"):
                    asset = assets[key]
                    a0_stack = []
                    for _, y, drive in windows:
                        a0 = (np.zeros((len(asset["C0"]), drive.shape[1]))
                              if arm == "MATCH_NOA0FB"
                              else asset["C_gated"] @ drive)
                        a0_stack.append(a0)
                    sig_stack = [asset["sig_gated"]] * len(windows)
                    out = sample_bank_eog(model, schedule, y_stack,
                                          np.stack(a0_stack), np.stack(sig_stack),
                                          device, natural_noise_seed(fold_id, seed))
                    for (start, y, drive), o in zip(windows, out):
                        natural_rows.append(
                            {"fold": fold_id, "seed": seed, "cell": "|".join(key),
                             "condition": arm, "start": start,
                             **_natural_metrics(y, drive,
                                                y - np.asarray(o, np.float64))})
            out_path.write_text(json.dumps(
                {"fold": fold_id, "seed": seed, "checkpoint": source["checkpoint"],
                 "rows": rows, "natural_rows": natural_rows,
                 "sealed_reads": 0}, indent=1, sort_keys=True) + "\n")
            print(json.dumps({"fold": fold_id, "seed": seed, "rows": len(rows),
                              "natural_rows": len(natural_rows),
                              "hard_cells": len(hard_test_keys)}), flush=True)


def aggregate() -> None:
    import pandas as pd
    files = sorted(OUT_DIR.glob("fold_*_seed_*.json"))
    if len(files) != 15:
        raise SystemExit(f"expected 15 unit files, found {len(files)} — not aggregating")
    rows, natural = [], []
    for path in files:
        payload = json.loads(path.read_text())
        rows += payload["rows"]
        natural += payload["natural_rows"]
    frame = pd.DataFrame(rows)
    cm = frame.groupby(["cell", "condition"]).rrmse_temporal.mean().unstack()
    hard_cells = sorted(frame[frame.hard_gated].cell.unique())
    hb = cm.loc[hard_cells]
    paired = (hb["MATCH_gated"] - hb["MATCH_NOA0FB"]).to_numpy()
    rng = np.random.default_rng(BOOT_SEED)
    draws = np.asarray([rng.choice(paired, len(paired), replace=True).mean()
                        for _ in range(BOOT_DRAWS)])
    lo, hi = float(np.quantile(draws, .025)), float(np.quantile(draws, .975))
    verdict = "WIN" if lo > 0 else ("LOSS" if hi < 0 else "TIE")

    per = frame.groupby(["participant", "condition"]).rrmse_temporal.mean().unstack()
    anchor = (per["NO_A0"] - per["MATCH_NOA0FB"]).to_numpy()
    adraws = np.asarray([rng.choice(anchor, len(anchor), replace=True).mean()
                         for _ in range(BOOT_DRAWS)])
    alo, ahi = float(np.quantile(adraws, .025)), float(np.quantile(adraws, .975))
    nat = pd.DataFrame(natural)
    nat_summary = (nat.groupby("condition")[["attenuation_db",
                                             "low_eog_observation_retention"]]
                   .mean().to_dict() if len(nat) else {})
    decision = {
        "prereg": "reports/iris_prereg_f.md (F1)",
        "F1_primary": {"contrast": "MATCH_gated - MATCH_NOA0FB, hard-gated cells",
                       "cells": hard_cells, "n": len(hard_cells),
                       "paired_mean": float(paired.mean()),
                       "ci_low": lo, "ci_high": hi, "verdict": verdict},
        "F1_anchor": {"contrast": "MATCH_NOA0FB - NO_A0 (participant-first)",
                      "mean": float(anchor.mean()), "ci_low": alo, "ci_high": ahi,
                      "n": int(len(anchor)),
                      "positive_count": int((anchor > 0).sum()),
                      "banked_incumbent_dev": BANKED_DEV_ANCHOR,
                      "preserved": bool(alo > 0)},
        "F1_natural_hard_cells": nat_summary,
        "incumbent_anchor_this_run": {
            "mean": float((per["NO_A0"] - per["MATCH_gated"]).mean())},
        "bit_identity_guard": "enforced in run(); any violation raises",
    }
    (OUT_DIR / "f1_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"verdict": verdict, "paired_mean": round(float(paired.mean()), 4),
                      "anchor": round(float(anchor.mean()), 4),
                      "anchor_preserved": bool(alo > 0)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["run", "aggregate"])
    args = parser.parse_args()
    (run if args.mode == "run" else aggregate)()


if __name__ == "__main__":
    main()
