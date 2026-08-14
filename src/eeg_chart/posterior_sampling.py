"""M13-W4: operator-posterior K-chain UQ on the frozen V44-S1 checkpoints.

Each chain draws C ~ N(C_gated, Sigma_post) from the U0-b entrywise EB
posterior, computes a0 = C·e, and runs the registered DDIM trajectory with a
fresh diffusion noise seed (total predictive variability = operator posterior
+ diffusion sampling).  Chains are EQUAL-WEIGHT (the registered particle
scheme).  Intervals are Gaussian (mean ± z·std of the chain ensemble) for
method/reference comparability; the empirical-quantile version is reported as
a secondary for K=8.  Runs with the V44 worktree's eeg_scad on PYTHONPATH
(read-only consumption).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


Z = {0.50: 0.6744897501960817, 0.80: 1.2815515655446004, 0.90: 1.6448536269514722}
V44_ROOT = Path("/home/infres/yinwang/denoiseNet_rgcc_eog_v44")
V44_RESULT = V44_ROOT / "results/rgcc_eog_v44"
COVERAGE_BANDS = {0.50: (0.35, 0.65), 0.80: (0.65, 0.90), 0.90: (0.80, 0.97)}


def run_cell(fold_id: int, seed: int, k_chains: int, out_dir: Path) -> None:
    import torch
    from eeg_scad.cli.run_v43 import configs
    from eeg_scad.cli.run_v44 import _bank_drives, _gated_assets, noise_seed, sample_bank_eog
    from eeg_scad.cli.run_v44_s2 import _posterior_variance
    from eeg_scad.data.artifact_transfer_v41r import TransferEpisodeSampler, TransferRegistry
    from eeg_scad.data.eb_transfer_v43 import EBTransferRegistry
    from eeg_scad.models.calib_saddpm_cond_v42r import LinearX0Schedule
    from eeg_scad.models.calib_saddpm_eog_v44 import CalibSADDPMEOG

    out_path = out_dir / f"fold_{fold_id}_seed_{seed}.json"
    if out_path.is_file():
        print(json.dumps({"fold": fold_id, "seed": seed, "skipped": "complete"}))
        return
    source = json.loads((V44_RESULT / "stage1" / f"fold_{fold_id}_seed_{seed}"
                         / "train_curve.json").read_text())
    data, folds, _ = configs()
    fold = folds[fold_id]
    device = torch.device("cuda")
    registry30 = TransferRegistry(data, fold, 30, .05)
    eb120 = EBTransferRegistry(data, fold, registry30, 120)
    assets = _gated_assets(registry30, eb120)
    post_var = _posterior_variance(registry30, eb120, fold)
    sampler = TransferEpisodeSampler(data, fold, "test", seed + 3, registry30)
    bank = sampler.sample_balanced(8)
    drives = _bank_drives(assets, bank)
    model = CalibSADDPMEOG().to(device)
    model.load_state_dict(torch.load(source["checkpoint"], map_location=device,
                                     weights_only=False)["ema"])
    schedule = LinearX0Schedule().to(device)
    base_seed = noise_seed(fold_id, seed)

    sig = np.stack([assets[(m["participant"], m["session"], m["task"])]["sig_gated"]
                    for m in bank["meta"]])
    chain_outputs = []
    for chain in range(k_chains):
        rng = np.random.default_rng(910000 + fold_id * 1000 + seed % 100 + chain * 17)
        a0 = []
        for meta, drive in zip(bank["meta"], drives):
            key = (meta["participant"], meta["session"], meta["task"])
            operator = assets[key]["C_gated"] + rng.standard_normal(
                assets[key]["C_gated"].shape) * np.sqrt(post_var[key])
            a0.append(operator @ drive)
        output = sample_bank_eog(model, schedule, bank["y"], np.stack(a0), sig, device,
                                 base_seed + 31 * (chain + 1))
        chain_outputs.append(output.astype(np.float32))
    ensemble = np.stack(chain_outputs)          # K x episodes x 46 x 512

    # DET-ensemble reference (2 seeds per fold, one-step; registered deviation)
    det_outputs = []
    for det_seed in (20261201, 20261202):
        det_source = json.loads((V44_RESULT / "stage1" / f"det_fold_{fold_id}_seed_{det_seed}"
                                 / "det_result.json").read_text())
        det_model = CalibSADDPMEOG().to(device)
        det_model.load_state_dict(torch.load(det_source["checkpoint"], map_location=device,
                                             weights_only=False)["ema"])
        det_model.eval()
        pieces = []
        with torch.no_grad():
            for start in range(0, len(bank["y"]), 8):
                stop = min(len(bank["y"]), start + 8)
                y = torch.from_numpy(np.asarray(bank["y"][start:stop], np.float32)).to(device)
                a0 = np.stack([assets[(m["participant"], m["session"], m["task"])]["C_gated"]
                               @ d for m, d in zip(bank["meta"][start:stop],
                                                   drives[start:stop])])
                anchor = torch.from_numpy(np.asarray(a0, np.float32)).to(device)
                signature = torch.from_numpy(np.asarray(sig[start:stop], np.float32)).to(device)
                timestep = torch.zeros(len(y), dtype=torch.long, device=device)
                pieces.append(det_model(y - anchor, y, anchor, timestep,
                                        signature).cpu().numpy())
        det_outputs.append(np.concatenate(pieces).astype(np.float32))
    det_ensemble = np.stack(det_outputs)

    def metrics(stack: np.ndarray) -> list[dict]:
        mean = stack.mean(axis=0)
        std = stack.std(axis=0, ddof=1).clip(1e-9)
        rows = []
        scales = np.linspace(0.5, 3.0, 26)
        for index, (clean, meta) in enumerate(zip(bank["x"], bank["meta"])):
            clean = np.asarray(clean, np.float64)
            inside = {}
            for level, z in Z.items():
                inside[str(level)] = float(np.mean(
                    np.abs(clean - mean[index]) <= z * std[index]))
            normalized = np.abs(clean - mean[index]) / std[index]
            cov_grid = [float(np.mean(normalized <= Z[0.80] * s)) for s in scales]
            samples = stack[:, index].astype(np.float64)
            crps = float(np.mean(np.abs(samples - clean[None]).mean(axis=0)
                                 - 0.5 * np.abs(samples[:, None] - samples[None]).mean(axis=(0, 1))))
            rrmse = float(np.linalg.norm(mean[index] - clean)
                          / max(np.linalg.norm(clean), 1e-12))
            rows.append({"participant": meta["participant"], "episode": index,
                         "coverage": inside, "crps": crps, "rrmse": rrmse,
                         "spread": float(std[index].mean()),
                         "cov80_scale_grid": cov_grid})
        return rows

    payload = {"fold": fold_id, "seed": seed, "k_chains": k_chains,
               "diff_rows": metrics(ensemble), "det_rows": metrics(det_ensemble),
               "det_members": det_ensemble.shape[0], "sealed_reads": 0}
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"fold": fold_id, "seed": seed, "chains": k_chains}))


def risk_coverage_auc(rows: list[dict]) -> float:
    order = np.argsort([row["spread"] for row in rows])
    errors = np.asarray([rows[i]["rrmse"] for i in order])
    cumulative = np.cumsum(errors) / np.arange(1, len(errors) + 1)
    return float(np.trapz(cumulative, dx=1.0 / len(errors)))


def aggregate(out_dir: Path, seeds) -> dict:
    import pandas as pd

    diff_rows, det_rows = [], []
    for fold_id in range(5):
        for seed in seeds:
            path = out_dir / f"fold_{fold_id}_seed_{seed}.json"
            if not path.is_file():
                continue
            payload = json.loads(path.read_text())
            diff_rows += payload["diff_rows"]
            det_rows += payload["det_rows"]

    def summarize(rows):
        frame = pd.DataFrame([{"participant": r["participant"], "crps": r["crps"],
                               **{f"cov{level}": r["coverage"][level]
                                  for level in ("0.5", "0.8", "0.9")}} for r in rows])
        per = frame.groupby("participant").mean(numeric_only=True)
        return {"coverage_50": float(per["cov0.5"].mean()),
                "coverage_80": float(per["cov0.8"].mean()),
                "coverage_90": float(per["cov0.9"].mean()),
                "crps": float(per.crps.mean()),
                "risk_coverage_auc": risk_coverage_auc(rows),
                "participants": len(per)}

    diff = summarize(diff_rows)
    det = summarize(det_rows)
    uq1_pass = all(COVERAGE_BANDS[level][0] <= diff[f"coverage_{int(level * 100)}"]
                   <= COVERAGE_BANDS[level][1] for level in COVERAGE_BANDS)

    def conformal(rows):
        """UQ-3: episode-parity split conformal at the 80% level (a DOWNGRADE:
        reported as 'conformalized', never as native posterior calibration)."""
        scales = np.linspace(0.5, 3.0, 26)
        calibration = [r for i, r in enumerate(rows) if i % 2 == 0]
        holdout = [r for i, r in enumerate(rows) if i % 2 == 1]
        grid = np.mean([r["cov80_scale_grid"] for r in calibration], axis=0)
        pick = int(np.argmin(np.abs(grid - 0.80)))
        achieved = float(np.mean([r["cov80_scale_grid"][pick] for r in holdout]))
        return {"scale": float(scales[pick]), "holdout_coverage_80": achieved}

    return {"diff": diff, "det_reference": det,
            "UQ-1": {"bands": {str(k): v for k, v in COVERAGE_BANDS.items()},
                     "pass": bool(uq1_pass), "v37t_reference": 0.0029},
            "UQ-2": {"crps_win": bool(diff["crps"] < det["crps"]),
                     "risk_coverage_win": bool(diff["risk_coverage_auc"]
                                               < det["risk_coverage_auc"])},
            "UQ-3_conformalized": {"diff": conformal(diff_rows),
                                   "det_reference": conformal(det_rows)}}


__all__ = ["aggregate", "risk_coverage_auc", "run_cell"]
