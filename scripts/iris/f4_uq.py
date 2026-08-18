#!/usr/bin/env python3
"""IRIS F4 — UQ width-policy adjudication on bit-identically regenerated W4 chains.

Preregistered in reports/iris_prereg_f4.md (frozen before execution). `run` (GPU)
regenerates the M13-W4 K-chain ensembles with the frozen seeds and stores per-sample
(|error|, sigma_chain, Var_op) to the derived root; `aggregate` (CPU) evaluates the
four width policies with leave-one-fold-out temperature and applies the frozen gates.
DET reference numbers are consumed from the banked W4 decision verbatim.
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
DERIVED = Path("/projects/EEG-foundation-model/derived/denoiseNet/iris_f4")
OUT_DIR = REPO / "results/iris/f4"
BANKED_W4 = Path("/home/infres/yinwang/denoiseNet_flagship_m0/results/flagship_m13/w4_uq")
SEEDS = (20261201, 20261202, 20261203)
K_CHAINS = 32
Z = {0.50: 0.6744897501960817, 0.80: 1.2815515655446004, 0.90: 1.6448536269514722}
TEMP_GRID = np.arange(0.5, 6.0 + 1e-9, 0.05)
CAL_TOL = 0.05
COST_FACTOR = 3.0
DET_RC_AUC_BANKED = 0.11286150412569007
REPRO_TOL = 0.01


def run() -> None:
    import torch
    from eeg_scad.cli.run_v43 import configs
    from eeg_scad.cli.run_v44 import (_bank_drives, _gated_assets, noise_seed,
                                      sample_bank_eog)
    from eeg_scad.cli.run_v44_s2 import _posterior_variance
    from eeg_scad.data.artifact_transfer_v41r import (TransferEpisodeSampler,
                                                      TransferRegistry)
    from eeg_scad.data.eb_transfer_v43 import EBTransferRegistry
    from eeg_scad.models.calib_saddpm_cond_v42r import LinearX0Schedule
    from eeg_scad.models.calib_saddpm_eog_v44 import CalibSADDPMEOG

    data, folds, _ = configs()
    device = torch.device("cuda")
    schedule = LinearX0Schedule().to(device)
    DERIVED.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        fold_id = fold["fold"]
        registry30 = TransferRegistry(data, fold, 30, .05)
        eb120 = EBTransferRegistry(data, fold, registry30, 120)
        assets = _gated_assets(registry30, eb120)
        post_var = _posterior_variance(registry30, eb120, fold)
        for seed in SEEDS:
            npz_path = DERIVED / f"fold_{fold_id}_seed_{seed}.npz"
            if npz_path.is_file():
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
            sig = np.stack([assets[(m["participant"], m["session"], m["task"])]
                            ["sig_gated"] for m in bank["meta"]])
            base_seed = noise_seed(fold_id, seed)
            chain_outputs = []
            for chain in range(K_CHAINS):
                rng = np.random.default_rng(
                    910000 + fold_id * 1000 + seed % 100 + chain * 17)
                a0 = []
                for meta, drive in zip(bank["meta"], drives):
                    key = (meta["participant"], meta["session"], meta["task"])
                    operator = assets[key]["C_gated"] + rng.standard_normal(
                        assets[key]["C_gated"].shape) * np.sqrt(post_var[key])
                    a0.append(operator @ drive)
                output = sample_bank_eog(model, schedule, bank["y"], np.stack(a0),
                                         sig, device, base_seed + 31 * (chain + 1))
                chain_outputs.append(output.astype(np.float32))
            ensemble = np.stack(chain_outputs)
            mean = ensemble.mean(axis=0)
            std = ensemble.std(axis=0, ddof=1).clip(1e-9)
            clean = np.asarray(bank["x"], np.float64)
            errors = np.abs(clean - mean)
            var_op = np.stack([post_var[(m["participant"], m["session"], m["task"])]
                               @ (np.asarray(d, np.float64) ** 2)
                               for m, d in zip(bank["meta"], drives)])
            # empirical CRPS at s=1 (reproduction guard vs the banked W4 rows)
            emp_crps = []
            for index in range(len(clean)):
                samples = ensemble[:, index].astype(np.float64)
                emp_crps.append(float(
                    np.mean(np.abs(samples - clean[index][None]).mean(axis=0)
                            - 0.5 * np.abs(samples[:, None]
                                           - samples[None]).mean(axis=(0, 1)))))
            np.savez_compressed(
                npz_path, errors=errors.astype(np.float16),
                sigma=std.astype(np.float16), var_op=var_op.astype(np.float16),
                participants=np.asarray([m["participant"] for m in bank["meta"]]),
                emp_crps=np.asarray(emp_crps))
            print(json.dumps({"fold": fold_id, "seed": seed,
                              "emp_crps_mean": float(np.mean(emp_crps))}), flush=True)


def _gauss_crps(error: np.ndarray, sigma: np.ndarray) -> float:
    from scipy.stats import norm
    z = error / sigma
    return float(np.mean(sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z)
                                  - 1 / np.sqrt(np.pi))))


def aggregate() -> None:
    units = {}
    for fold in range(5):
        for seed in SEEDS:
            d = np.load(DERIVED / f"fold_{fold}_seed_{seed}.npz",
                        allow_pickle=False)
            units[(fold, seed)] = {k: (d[k].astype(np.float64)
                                       if d[k].dtype.kind in "fiu" else d[k])
                                   for k in d.files}

    banked_rows = []
    for fold in range(5):
        for seed in SEEDS:
            payload = json.loads(
                (BANKED_W4 / f"fold_{fold}_seed_{seed}.json").read_text())
            banked_rows += [r["crps"] for r in payload["diff_rows"]]
    regen = np.concatenate([units[k]["emp_crps"] for k in sorted(units)])
    repro_gap = float(abs(np.mean(banked_rows) - regen.mean()))

    def sigma_policy(u, policy, s=1.0):
        base = (np.sqrt(u["sigma"] ** 2 + u["var_op"])
                if "INFL" in policy else u["sigma"])
        return s * base

    def coverage(u, sig_p, level):
        return float(np.mean(u["errors"] <= Z[level] * sig_p))

    def episode_stats(u, sig_p):
        rows = []
        for i in range(u["errors"].shape[0]):
            rows.append({"crps": _gauss_crps(u["errors"][i], sig_p[i]),
                         "spread": float(sig_p[i].mean()),
                         "rrmse_err": float(np.mean(u["errors"][i]))})
        return rows

    policies = ("W-SHARP", "W-TEMP", "W-INFL", "W-INFL-TEMP")
    results = {}
    for policy in policies:
        cov = {0.5: [], 0.8: [], 0.9: []}
        ep_rows = []
        temps = {}
        for hold_fold in range(5):
            if "TEMP" in policy:
                grid_cov = []
                for s in TEMP_GRID:
                    cal = [coverage(units[(f, sd)],
                                    sigma_policy(units[(f, sd)], policy, s), 0.80)
                           for f in range(5) if f != hold_fold for sd in SEEDS]
                    grid_cov.append(np.mean(cal))
                pick = next((s for s, c in zip(TEMP_GRID, grid_cov) if c >= 0.80),
                            TEMP_GRID[-1])
                temps[hold_fold] = float(pick)
            else:
                temps[hold_fold] = 1.0
            for sd in SEEDS:
                u = units[(hold_fold, sd)]
                sig_p = sigma_policy(u, policy, temps[hold_fold])
                for level in cov:
                    cov[level].append(coverage(u, sig_p, level))
                ep_rows += episode_stats(u, sig_p)
        crps = float(np.mean([r["crps"] for r in ep_rows]))
        order = np.argsort([r["spread"] for r in ep_rows])
        errs = np.asarray([ep_rows[i]["rrmse_err"] for i in order])
        rc_auc = float(np.mean(np.cumsum(errs) / np.arange(1, len(errs) + 1)))
        results[policy] = {
            "coverage": {str(k): float(np.mean(v)) for k, v in cov.items()},
            "crps_gaussian": crps, "risk_coverage_auc": rc_auc,
            "temperatures": temps}

    sharp_crps = results["W-SHARP"]["crps_gaussian"]
    verdicts = {}
    for policy, r in results.items():
        cal = (abs(r["coverage"]["0.8"] - 0.80) <= CAL_TOL
               and abs(r["coverage"]["0.9"] - 0.90) <= CAL_TOL)
        cost = r["crps_gaussian"] <= COST_FACTOR * sharp_crps
        rank = r["risk_coverage_auc"] <= DET_RC_AUC_BANKED
        verdicts[policy] = {"G_F4_cal": bool(cal), "G_F4_cost": bool(cost),
                            "G_F4_rank": bool(rank),
                            "pass_all": bool(cal and cost and rank),
                            "crps_ratio_vs_sharp":
                                float(r["crps_gaussian"] / sharp_crps)}
    passing = [p for p in policies if verdicts[p]["pass_all"]]
    head = min(passing, key=lambda p: results[p]["crps_gaussian"]) if passing else None
    physics_wording = bool(head and "INFL" in head and "W-TEMP" in passing
                           and results[head]["crps_gaussian"]
                           < results["W-TEMP"]["crps_gaussian"]) \
        if head else False
    decision = {
        "prereg": "reports/iris_prereg_f4.md",
        "reproduction_guard": {"banked_emp_crps_mean": float(np.mean(banked_rows)),
                               "regen_emp_crps_mean": float(regen.mean()),
                               "gap": repro_gap, "tol": REPRO_TOL,
                               "pass": bool(repro_gap <= REPRO_TOL)},
        "det_reference_banked": {"crps": 0.1547910834032748,
                                 "risk_coverage_auc": DET_RC_AUC_BANKED},
        "policies": results, "verdicts": verdicts,
        "uq_head": head,
        "physics_informed_wording_permitted": physics_wording,
        "wording": ("operator-posterior width calibrates the bands" if physics_wording
                    else "calibration, not physics, fixes the bands"
                    if head else "no policy passes all gates"),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "f4_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"head": head, "physics_wording": physics_wording,
                      **{p: verdicts[p]["pass_all"] for p in policies}}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["run", "aggregate"])
    args = parser.parse_args()
    (run if args.mode == "run" else aggregate)()


if __name__ == "__main__":
    main()
