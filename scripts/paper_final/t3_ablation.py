#!/usr/bin/env python3
"""PAPER-FINAL T3 — ablation-matrix completion (guide x calibration features) + shrinkage arm.

Missing cells run here on the standard dev episodes (seed 20261201 models/noise only):
  NONE_POPFEAT   — (guide none,    features population)
  MATCH_POPFEAT  — (guide matched, features population)
  MATCH_UNSHRUNK — guide from the unshrunk 120-s ridge estimate (lambda=1),
                   matched features (quantifies what shrinkage buys)
Existing cells come from stored V44-S1 outputs (seed-20261201 rows):
  (matched, matched) = MATCH_gated;  (none, matched) = NO_A0;
  plus the POP row (population guide, population features) for context.
Resume-safe per fold.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from pf_common import (ARRAYS, OUT, SEED, load_model, participant_means, stat,
                       stored_stage1_rows)

UNIT_DIR = OUT / "t3_units"
NEW_ARMS = ("NONE_POPFEAT", "MATCH_POPFEAT", "MATCH_UNSHRUNK")


def run(only_fold: int | None = None) -> None:
    import torch
    from eeg_scad.cli.run_v43 import configs
    from eeg_scad.cli.run_v44 import (_bank_drives, _gated_assets, noise_seed,
                                      sample_bank_eog)
    from eeg_scad.data.artifact_transfer_v41r import (TransferEpisodeSampler,
                                                      TransferRegistry)
    from eeg_scad.data.eb_transfer_v43 import EBTransferRegistry
    from eeg_scad.evaluation.paired_metrics import paired_metrics
    from eeg_scad.models.calib_saddpm_cond_v42r import LinearX0Schedule

    data, folds, _ = configs()
    if only_fold is not None:
        folds = [fold for fold in folds if fold["fold"] == only_fold]
    device = torch.device("cuda")
    schedule = LinearX0Schedule().to(device)
    UNIT_DIR.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        fold_id = fold["fold"]
        out_path = UNIT_DIR / f"fold_{fold_id}.json"
        if out_path.is_file():
            print(json.dumps({"fold": fold_id, "skipped": True}), flush=True)
            continue
        registry30 = TransferRegistry(data, fold, 30, .05)
        eb120 = EBTransferRegistry(data, fold, registry30, 120)
        assets = _gated_assets(registry30, eb120)
        model = load_model(fold_id, device, SEED)
        sampler = TransferEpisodeSampler(data, fold, "test", SEED + 3, registry30)
        bank = sampler.sample_balanced(8)
        drives = _bank_drives(assets, bank)
        keys = [(m["participant"], m["session"], m["task"]) for m in bank["meta"]]

        rows = []
        for arm in NEW_ARMS:
            a0_stack, sig_stack = [], []
            for key, drive in zip(keys, drives):
                asset = assets[key]
                if arm == "NONE_POPFEAT":
                    a0, sig = np.zeros((46, drive.shape[1])), asset["sig_pop"]
                elif arm == "MATCH_POPFEAT":
                    a0, sig = asset["C_gated"] @ drive, asset["sig_pop"]
                else:  # MATCH_UNSHRUNK: raw full-prefix ridge, lambda = 1
                    a0, sig = asset["C_raw"] @ drive, asset["sig_gated"]
                a0_stack.append(a0)
                sig_stack.append(sig)
            output = sample_bank_eog(model, schedule, bank["y"], np.stack(a0_stack),
                                     np.stack(sig_stack), device,
                                     noise_seed(fold_id, SEED))
            for episode, (clean, observed, artifact, meta, key) in enumerate(
                    zip(bank["x"], bank["y"], bank["artifact"], bank["meta"], keys)):
                prediction = output[episode]
                if not np.isfinite(prediction).all():
                    raise FloatingPointError("nonfinite T3 output")
                rows.append({"fold": fold_id, "participant": key[0], "condition": arm,
                             **paired_metrics(clean, observed, artifact,
                                              observed - prediction)})
        out_path.write_text(json.dumps({"fold": fold_id, "rows": rows},
                                       indent=1, sort_keys=True) + "\n")
        print(json.dumps({"fold": fold_id, "rows": len(rows)}), flush=True)


def aggregate() -> None:
    files = sorted(UNIT_DIR.glob("fold_*.json"))
    if len(files) != 5:
        raise SystemExit(f"expected 5 fold files, found {len(files)}")
    rows = []
    for path in files:
        rows += json.loads(path.read_text())["rows"]
    stored = stored_stage1_rows(seeds=(SEED,))
    cells = {
        ("matched", "matched"): participant_means(stored, "MATCH_gated"),
        ("none", "matched"): participant_means(stored, "NO_A0"),
        ("population", "population"): participant_means(stored, "POP"),
        ("none", "population"): participant_means(rows, "NONE_POPFEAT"),
        ("matched", "population"): participant_means(rows, "MATCH_POPFEAT"),
        ("matched_unshrunk", "matched"): participant_means(rows, "MATCH_UNSHRUNK"),
    }
    reference = cells[("matched", "matched")]
    matrix, contrasts = {}, {}
    for (guide, features), per in cells.items():
        name = f"guide={guide}|features={features}"
        matrix[name] = float(np.mean(list(per.values())))
        if (guide, features) != ("matched", "matched"):
            contrasts[name] = stat([per[p] - reference[p] for p in reference])
    decision = {
        "note": "seed-20261201 cells only (stored + new arms share episodes and noise); "
                "matrix entries are participant-first temporal RRMSE; contrasts are "
                "cell minus (matched,matched), positive = matched-matched better",
        "matrix": matrix, "contrasts_vs_matched_matched": contrasts,
        "shrinkage_effect_unshrunk_minus_shrunk":
            stat([cells[("matched_unshrunk", "matched")][p] - reference[p]
                  for p in reference]),
    }
    (OUT / "t3_ablation.json").write_text(json.dumps(decision, indent=2,
                                                     sort_keys=True) + "\n")
    np.savez_compressed(ARRAYS / "t3_ablation_matrix.npz",
                        cells=np.asarray(json.dumps(matrix)),
                        contrasts=np.asarray(json.dumps(
                            {k: v["mean"] for k, v in contrasts.items()})))
    print(json.dumps(matrix, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["run", "aggregate"])
    parser.add_argument("--fold", type=int, default=None)
    args = parser.parse_args()
    if args.mode == "run":
        run(args.fold)
    else:
        aggregate()


if __name__ == "__main__":
    main()
