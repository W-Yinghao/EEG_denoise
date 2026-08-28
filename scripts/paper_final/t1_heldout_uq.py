#!/usr/bin/env python3
"""PAPER-FINAL T1 (parts B+C) — held-out predictive-interval run (MobileBCI sealed-8).

Same frozen seed-20261201 fold checkpoints, episodes and protocol as the M35 sealed
point-estimate pass (fold-99 construction, sampler seed 20269001, 5-fold output
ensemble).  K=32 stochastic trajectories jointly sample DDIM initial noise and an
entrywise EB-posterior operator draw (IRIS-F4 construction transplanted to fold 99).
Ensemble convention: each chain is the mean of the 5 fold-model outputs under a
shared noise draw and a shared operator draw (mirrors the M35 point-estimate
convention).  The scalar temperature was frozen from the development set and
committed before this run (results/paper_final/t1_temperature.json).

`run` (GPU) stores per-sample arrays; `aggregate` (CPU) scores the three policies
(raw samples / temperature-only / operator-posterior inflation + temperature) at
50/80/90% coverage, Gaussian CRPS, risk-coverage area, per-participant spread.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pf_common import (ARRAYS, OUT, SEED, Z, gauss_crps, load_model, stat)

DERIVED_T1 = OUT / "t1_arrays"
SEALED = ("sub-01", "sub-04", "sub-08", "sub-10", "sub-13", "sub-16", "sub-20", "sub-22")
EXEMPLAR_SUBJECTS = ("sub-01", "sub-04")
MERGED_ROOT = Path("/projects/EEG-foundation-model/derived/denoiseNet/flagship_m35/sealed_root")
SAMPLER_SEED = 20269001
PAIRED_SEED_BASE = 421000
K_CHAINS = 32
FOLD99 = 99


def _fold99_context():
    from eeg_scad.cli.run_v43 import configs
    from eeg_scad.cli.run_v44 import _gated_assets
    from eeg_scad.data.artifact_transfer_v41r import (TransferEpisodeSampler,
                                                      TransferRegistry)
    from eeg_scad.data.eb_transfer_v43 import EBTransferRegistry

    data, _, _ = configs()
    data = dict(data)
    data["v19_derived_root"] = str(MERGED_ROOT)
    fold = {"fold": FOLD99, "train": list(data["participants"]), "validation": [],
            "test": list(SEALED)}
    registry30 = TransferRegistry(data, fold, 30, .05)
    eb120 = EBTransferRegistry(data, fold, registry30, 120)
    assets = _gated_assets(registry30, eb120)
    sampler = TransferEpisodeSampler(data, fold, "test", SAMPLER_SEED, registry30)
    return data, fold, registry30, eb120, assets, sampler


def run() -> None:
    import torch
    from eeg_scad.cli.run_v44 import _bank_drives, sample_bank_eog
    from eeg_scad.cli.run_v44_s2 import _posterior_variance
    from eeg_scad.models.calib_saddpm_cond_v42r import LinearX0Schedule

    temp_path = OUT / "t1_temperature.json"
    if not temp_path.is_file():
        raise SystemExit("t1_temperature.json missing — freeze the dev temperature first")
    npz_path = DERIVED_T1 / "heldout_k32.npz"
    if npz_path.is_file():
        print(json.dumps({"skipped": "T1 arrays already stored"}))
        return
    data, fold, registry30, eb120, assets, sampler = _fold99_context()
    post_var = _posterior_variance(registry30, eb120, fold)
    bank = sampler.sample_balanced(8)
    drives = _bank_drives(assets, bank)
    keys = [(m["participant"], m["session"], m["task"]) for m in bank["meta"]]

    device = torch.device("cuda")
    schedule = LinearX0Schedule().to(device)
    models = [load_model(fold_id, device, SEED) for fold_id in range(5)]

    subject_order = [s for s in SEALED
                     if any(m["participant"] == s for m in bank["meta"])]
    chain_outputs = np.zeros((K_CHAINS,) + np.asarray(bank["x"]).shape, np.float32)
    for chain in range(K_CHAINS):
        rng = np.random.default_rng(910000 + FOLD99 * 1000 + SEED % 100 + chain * 17)
        a0_all = np.zeros_like(np.asarray(bank["y"], np.float64))
        for i, (key, drive) in enumerate(zip(keys, drives)):
            operator = assets[key]["C_gated"] + rng.standard_normal(
                assets[key]["C_gated"].shape) * np.sqrt(post_var[key])
            a0_all[i] = operator @ drive
        for subject_index, subject in enumerate(SEALED):
            indices = [i for i, m in enumerate(bank["meta"])
                       if m["participant"] == subject]
            if not indices:
                continue
            sub_y = np.stack([bank["y"][i] for i in indices])
            sub_a0 = np.stack([a0_all[i] for i in indices])
            sub_sig = np.stack([assets[keys[i]]["sig_gated"] for i in indices])
            ensemble = np.mean([sample_bank_eog(model, schedule, sub_y, sub_a0, sub_sig,
                                                device,
                                                PAIRED_SEED_BASE + subject_index
                                                + 31 * (chain + 1))
                                for model in models], axis=0)
            if not np.isfinite(ensemble).all():
                raise FloatingPointError("nonfinite T1 chain output")
            for local, i in enumerate(indices):
                chain_outputs[chain, i] = ensemble[local]
        print(json.dumps({"chain": chain, "done": True}), flush=True)

    clean = np.asarray(bank["x"], np.float64)
    mean = chain_outputs.mean(axis=0)
    sigma = chain_outputs.std(axis=0, ddof=1).clip(1e-9)
    errors = np.abs(clean - mean)
    var_op = np.stack([post_var[key] @ (np.asarray(d, np.float64) ** 2)
                       for key, d in zip(keys, drives)])
    DERIVED_T1.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        errors=errors.astype(np.float16), sigma=sigma.astype(np.float16),
        var_op=var_op.astype(np.float16),
        participants=np.asarray([m["participant"] for m in bank["meta"]]),
        zero_artifact=np.asarray([m["zero_artifact"] for m in bank["meta"]]))
    # per-window mean + width arrays for two held-out participants (feeds T6)
    for subject in EXEMPLAR_SUBJECTS:
        indices = [i for i, m in enumerate(bank["meta"]) if m["participant"] == subject]
        np.savez_compressed(
            ARRAYS / f"t6_heldout_intervals_{subject}.npz",
            mean=mean[indices].astype(np.float32),
            sigma=sigma[indices].astype(np.float32),
            var_op=var_op[indices].astype(np.float32),
            contaminated=np.stack([bank["y"][i] for i in indices]).astype(np.float32),
            reference=np.stack([bank["x"][i] for i in indices]).astype(np.float32),
            eog_drive=np.stack([drives[i] for i in indices]).astype(np.float32),
            zero_artifact=np.asarray([bank["meta"][i]["zero_artifact"]
                                      for i in indices]))
    print(json.dumps({"episodes": int(len(clean)), "subjects": subject_order}))


def aggregate() -> None:
    temps = json.loads((OUT / "t1_temperature.json").read_text())["temperatures"]
    d = np.load(DERIVED_T1 / "heldout_k32.npz", allow_pickle=False)
    errors = d["errors"].astype(np.float64)
    sigma = d["sigma"].astype(np.float64)
    var_op = d["var_op"].astype(np.float64)
    participants = [str(p) for p in d["participants"]]

    policies = {
        "raw_samples": (1.0, sigma),
        "temperature_only": (temps["TEMP"], sigma),
        "propagation_plus_temperature": (temps["INFL"], np.sqrt(sigma ** 2 + var_op)),
    }
    report = {}
    for name, (s, base) in policies.items():
        width = s * base
        coverage = {str(level): float(np.mean(errors <= Z[level] * width))
                    for level in (0.50, 0.80, 0.90)}
        ep_rows = [{"crps": gauss_crps(errors[i], width[i]),
                    "spread": float(width[i].mean()),
                    "err": float(errors[i].mean())} for i in range(len(errors))]
        order = np.argsort([r["spread"] for r in ep_rows])
        errs = np.asarray([ep_rows[i]["err"] for i in order])
        per_part = {}
        for p in sorted(set(participants)):
            idx = [i for i, q in enumerate(participants) if q == p]
            per_part[p] = float(np.mean(errors[idx] <= Z[0.80] * width[idx]))
        report[name] = {
            "temperature": s, "coverage": coverage,
            "crps_gaussian": float(np.mean([r["crps"] for r in ep_rows])),
            "risk_coverage_auc": float(np.mean(np.cumsum(errs)
                                               / np.arange(1, len(errs) + 1))),
            "per_participant_coverage_80": per_part,
            "per_participant_coverage_80_range": [float(min(per_part.values())),
                                                  float(max(per_part.values()))],
        }
    decision = {
        "protocol": "M35 sealed episodes/protocol; K=32 chains; 5-model-mean per chain "
                    "with shared noise and operator draw; temperatures dev-frozen "
                    "jointly on all 15 dev cells before this run",
        "temperatures": temps, "policies": report,
        "dev_reference": {"coverage_80_90": [0.80240, 0.85299], "crps": 0.15031},
    }
    (OUT / "t1_heldout_uq.json").write_text(json.dumps(decision, indent=2,
                                                       sort_keys=True) + "\n")
    np.savez_compressed(ARRAYS / "t1_heldout_uq_summary.npz",
                        policies=np.asarray(json.dumps(report)))
    print(json.dumps({name: {"coverage": r["coverage"], "crps": r["crps_gaussian"]}
                      for name, r in report.items()}, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["run", "aggregate"])
    args = parser.parse_args()
    (run if args.mode == "run" else aggregate)()


if __name__ == "__main__":
    main()
