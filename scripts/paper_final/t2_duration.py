#!/usr/bin/env python3
"""PAPER-FINAL T2 — calibration-duration curve, 30/60/90/120 s.

For each duration d the calibration prefix is truncated to d seconds and scaling,
ridge and EB shrinkage are recomputed closed-form (the frozen registry class
whitelists {10,30,60,120}; the 90-s point replays the identical constructor with the
whitelist relaxed — pf_common.make_eb_registry).  Two readings per duration:
  (a) system   — reliability rule active; a rejected calibration WITHHOLDS the guide
                 (a0=0) and its features short-circuit to the population values
                 (the adopted BINARY_NOA0FB rule).  30 s always rejects by design
                 (hard gate: effective support < 60 s) — that is the rule firing.
  (b) rule_off — the raw shrinkage curve with the rejection floor bypassed
                 (lambda = tau2/(tau2 + within/4) even where the gate would fire).
Matched vs unguided on the standard dev paired episodes, seed 20261201 models only.
Resume-safe per fold.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from pf_common import (ARRAYS, OUT, SEED, lambda_rule_off, load_model,
                       make_eb_registry, participant_means, signature_with_lambda,
                       stat, stored_stage1_rows)

DURATIONS = (30, 60, 90, 120)
UNIT_DIR = OUT / "t2_units"


def run() -> None:
    import torch
    from eeg_scad.cli.run_v43 import configs
    from eeg_scad.cli.run_v44 import (_bank_drives, _gated_assets, noise_seed,
                                      sample_bank_eog)
    from eeg_scad.data.artifact_transfer_v41r import (TransferEpisodeSampler,
                                                      TransferRegistry)
    from eeg_scad.evaluation.paired_metrics import paired_metrics
    from eeg_scad.models.calib_saddpm_cond_v42r import LinearX0Schedule

    data, folds, _ = configs()
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
        eb_by_duration = {d: make_eb_registry(data, fold, registry30, d)
                          for d in DURATIONS}
        assets120 = _gated_assets(registry30, eb_by_duration[120])
        model = load_model(fold_id, device, SEED)
        sampler = TransferEpisodeSampler(data, fold, "test", SEED + 3, registry30)
        bank = sampler.sample_balanced(8)
        drives = _bank_drives(assets120, bank)
        keys = [(m["participant"], m["session"], m["task"]) for m in bank["meta"]]
        pop = {key: registry30.population_transfer[key[1:]] for key in set(keys)}

        rows, gate_rows = [], []
        for duration in DURATIONS:
            eb = eb_by_duration[duration]
            for key in sorted(set(keys)):
                cell = eb.cells[key]
                gate_rows.append({"duration": duration, "cell": "|".join(key),
                                  "participant": key[0], "lambda": float(cell.lam),
                                  "lambda_rule_off": lambda_rule_off(cell),
                                  "hard_gate": int(cell.hard_gate),
                                  "within": float(cell.within),
                                  "tau2": float(cell.tau2)})
            for reading in ("system", "rule_off"):
                a0_stack, sig_stack = [], []
                for key, drive in zip(keys, drives):
                    cell = eb.cells[key]
                    if reading == "system":
                        if cell.hard_gate:
                            a0 = np.zeros((46, drive.shape[1]))
                        else:
                            operator = pop[key] + cell.lam * (cell.transfer - pop[key])
                            a0 = operator @ drive
                        sig = eb.signature(*key, "EB")
                    else:
                        lam = lambda_rule_off(cell)
                        operator = pop[key] + lam * (cell.transfer - pop[key])
                        a0 = operator @ drive
                        sig = signature_with_lambda(eb, key, lam)
                    a0_stack.append(a0)
                    sig_stack.append(sig)
                output = sample_bank_eog(model, schedule, bank["y"],
                                         np.stack(a0_stack), np.stack(sig_stack),
                                         device, noise_seed(fold_id, SEED))
                for episode, (clean, observed, artifact, meta, key) in enumerate(
                        zip(bank["x"], bank["y"], bank["artifact"], bank["meta"], keys)):
                    prediction = output[episode]
                    if not np.isfinite(prediction).all():
                        raise FloatingPointError("nonfinite T2 output")
                    rows.append({"fold": fold_id, "participant": key[0],
                                 "condition": f"MATCH_{reading}_{duration}s",
                                 "duration": duration, "reading": reading,
                                 **paired_metrics(clean, observed, artifact,
                                                  observed - prediction)})
        out_path.write_text(json.dumps({"fold": fold_id, "rows": rows,
                                        "gate_rows": gate_rows},
                                       indent=1, sort_keys=True) + "\n")
        print(json.dumps({"fold": fold_id, "rows": len(rows)}), flush=True)


def aggregate() -> None:
    files = sorted(UNIT_DIR.glob("fold_*.json"))
    if len(files) != 5:
        raise SystemExit(f"expected 5 fold files, found {len(files)}")
    rows, gate_rows = [], []
    for path in files:
        payload = json.loads(path.read_text())
        rows += payload["rows"]
        gate_rows += payload["gate_rows"]
    # unguided reference: stored stage1 NO_A0, seed 20261201 only (same episodes/noise)
    stored = [r for r in stored_stage1_rows(seeds=(SEED,))]
    noa0 = participant_means(stored, "NO_A0")
    curve = {}
    for duration in DURATIONS:
        for reading in ("system", "rule_off"):
            arm = f"MATCH_{reading}_{duration}s"
            per = participant_means(rows, arm)
            gain = [noa0[p] - per[p] for p in per]
            fired = [g for g in gate_rows if g["duration"] == duration]
            curve[arm] = {
                "duration_s": duration, "reading": reading,
                "rrmse": float(np.mean(list(per.values()))),
                "gain_vs_unguided": stat(gain),
                "hard_gate_fraction": float(np.mean([g["hard_gate"] for g in fired])),
                "lambda_mean": float(np.mean([g["lambda"] for g in fired])),
                "lambda_rule_off_mean": float(np.mean([g["lambda_rule_off"]
                                                       for g in fired])),
            }
    decision = {"unguided_reference_rrmse":
                float(np.mean(list(noa0.values()))),
                "note": "unguided reference = stored V44-S1 NO_A0 rows, seed 20261201 "
                        "only (identical episodes and noise seeds)",
                "curve": curve}
    (OUT / "t2_duration.json").write_text(json.dumps(decision, indent=2,
                                                     sort_keys=True) + "\n")
    np.savez_compressed(
        ARRAYS / "t2_duration_curve.npz",
        durations=np.asarray(DURATIONS),
        gain_system=np.asarray([curve[f"MATCH_system_{d}s"]["gain_vs_unguided"]["mean"]
                                for d in DURATIONS]),
        gain_system_ci=np.asarray(
            [[curve[f"MATCH_system_{d}s"]["gain_vs_unguided"]["bootstrap_low"],
              curve[f"MATCH_system_{d}s"]["gain_vs_unguided"]["bootstrap_high"]]
             for d in DURATIONS]),
        gain_rule_off=np.asarray(
            [curve[f"MATCH_rule_off_{d}s"]["gain_vs_unguided"]["mean"]
             for d in DURATIONS]),
        gain_rule_off_ci=np.asarray(
            [[curve[f"MATCH_rule_off_{d}s"]["gain_vs_unguided"]["bootstrap_low"],
              curve[f"MATCH_rule_off_{d}s"]["gain_vs_unguided"]["bootstrap_high"]]
             for d in DURATIONS]),
        hard_gate_fraction=np.asarray(
            [curve[f"MATCH_system_{d}s"]["hard_gate_fraction"] for d in DURATIONS]))
    print(json.dumps({arm: {"gain": v["gain_vs_unguided"]["mean"],
                            "gate_fired": v["hard_gate_fraction"]}
                      for arm, v in curve.items()}, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["run", "aggregate"])
    args = parser.parse_args()
    (run if args.mode == "run" else aggregate)()


if __name__ == "__main__":
    main()
