#!/usr/bin/env python3
"""Shared helpers for the paper-final runs (SERVER_INSTRUCTIONS_PAPER_FINAL_RUNS.md).

Light-harness paper-finishing science: frozen V44-S1 checkpoints, seed-20261201
conventions, participant-first aggregation, 5000-resample participant bootstrap.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
V44_ROOT = Path("/home/infres/yinwang/denoiseNet_rgcc_eog_v44")
sys.path.insert(0, str(V44_ROOT / "src"))
FLAGSHIP = Path("/home/infres/yinwang/denoiseNet_flagship_m0")
OUT = REPO / "results/paper_final"
ARRAYS = REPO / "paper_final_arrays"
SEED = 20261201
S1_SEEDS = (20261201, 20261202, 20261203)
BOOT_SEED, BOOT_DRAWS = 420, 5000
Z = {0.50: 0.6744897501960817, 0.80: 1.2815515655446004, 0.90: 1.6448536269514722}
SEALED = ("sub-01", "sub-04", "sub-08", "sub-10", "sub-13", "sub-16", "sub-20", "sub-22")
STAGE1 = V44_ROOT / "results/rgcc_eog_v44/stage1"


def stat(values) -> dict:
    """Participant-first mean with 5000-draw participant bootstrap (seed 420)."""
    value = np.asarray(values, float)
    rng = np.random.default_rng(BOOT_SEED)
    draws = np.asarray([rng.choice(value, len(value), replace=True).mean()
                        for _ in range(BOOT_DRAWS)])
    return {"mean": float(value.mean()), "median": float(np.median(value)),
            "positive_count": int((value > 0).sum()), "participants": int(len(value)),
            "bootstrap_low": float(np.quantile(draws, .025)),
            "bootstrap_high": float(np.quantile(draws, .975))}


def participant_means(rows, condition, metric="rrmse_temporal"):
    """rows -> {participant: mean metric} for one condition."""
    values: dict[str, list[float]] = {}
    for row in rows:
        if row["condition"] == condition:
            values.setdefault(row["participant"], []).append(float(row[metric]))
    return {p: float(np.mean(v)) for p, v in sorted(values.items())}


def load_model(fold_id: int, device, seed: int = SEED):
    import torch
    from eeg_scad.models.calib_saddpm_eog_v44 import CalibSADDPMEOG

    source = json.loads((STAGE1 / f"fold_{fold_id}_seed_{seed}" /
                         "train_curve.json").read_text())
    model = CalibSADDPMEOG().to(device)
    model.load_state_dict(torch.load(source["checkpoint"], map_location=device,
                                     weights_only=False)["ema"])
    model.eval()
    return model


def stored_stage1_rows(seeds=S1_SEEDS):
    rows = []
    for fold_id in range(5):
        for seed in seeds:
            payload = json.loads((STAGE1 / f"fold_{fold_id}_seed_{seed}" /
                                  "stage1_result.json").read_text())
            rows += payload["rows"]
    return rows


def stored_stage1_natural_rows(seeds=S1_SEEDS):
    rows = []
    for fold_id in range(5):
        for seed in seeds:
            payload = json.loads((STAGE1 / f"fold_{fold_id}_seed_{seed}" /
                                  "stage1_result.json").read_text())
            rows += payload["natural_rows"]
    return rows


def make_eb_registry(data, fold, registry30, seconds):
    """EBTransferRegistry for any duration whose prefix splits into 4 blocks.

    The frozen class whitelists {10,30,60,120}; T2 needs 90 s.  This replays the
    frozen __init__ verbatim with the whitelist relaxed — no other change."""
    from eeg_scad.data import eb_transfer_v43 as ebm

    if seconds in (120, 60, 30, 10):
        return ebm.EBTransferRegistry(data, fold, registry30, seconds)

    self = object.__new__(ebm.EBTransferRegistry)
    self.data, self.fold, self.registry30, self.seconds = data, fold, registry30, seconds
    rate = int(data.get("sampling_rate", 100))
    prefix = seconds * rate
    if prefix % ebm.SUB_BLOCKS:
        raise ValueError("prefix must split into four equal sub-blocks")
    train_quality = np.stack([cell.quality for key, cell in registry30.cells.items()
                              if key[0] in fold["train"]])
    self.quality_min = train_quality.min(axis=0)
    self.quality_max = train_quality.max(axis=0)
    fits = {key: self._fit(key, prefix, rate) for key in registry30.cells}
    self.tau2 = {}
    self.tau2_rows = {}
    for group in registry30.population_transfer:
        pop = registry30.population_transfer[group]
        train_full = np.stack([fits[key][0] for key in sorted(fits)
                               if key[0] in fold["train"] and key[1:] == group])
        deviation = np.square(train_full - pop[None])
        self.tau2[group] = float(deviation.mean())
        self.tau2_rows[group] = deviation.mean(axis=(0, 2))
    within_values = {}
    for key, (full, blocks, _, _) in fits.items():
        deviation = np.square(blocks - full[None])
        within_values[key] = (float(deviation.mean()), deviation.mean(axis=(0, 2)))
    self.within_threshold = float(np.percentile(
        [within_values[key][0] for key in sorted(within_values)
         if key[0] in fold["train"]], ebm.WITHIN_PERCENTILE))
    self.cells = {}
    for key, (full, _, quality, effective) in fits.items():
        group = key[1:]
        if group not in self.tau2:
            continue
        within, within_rows = within_values[key]
        lam, hard_gate = ebm.eb_lambda(self.tau2[group], within, effective,
                                       self.within_threshold)
        lam_rows = ebm.eb_lambda_rows(self.tau2_rows[group], within_rows, hard_gate)
        self.cells[key] = ebm.EBCell(key[0], key[1], key[2], effective, full, quality,
                                     self.tau2[group], within, lam, hard_gate,
                                     self.tau2_rows[group], within_rows, lam_rows)
    return self


def lambda_rule_off(cell) -> float:
    """The shrinkage weight WITHOUT the hard rejection gate (T2 rule-off reading)."""
    return float(np.clip(cell.tau2 / max(cell.tau2 + cell.within / 4.0, 1e-12), 0.0, 1.0))


def signature_with_lambda(eb, key, lam) -> np.ndarray:
    """Gated-signature construction with an arbitrary blend weight (rule-off arm)."""
    cell = eb.cells[key]
    registry = eb.registry30
    pop_transfer = registry.population_transfer[key[1:]]
    pop_quality = registry.population_quality[key[1:]]
    quality_clamped = np.clip(cell.quality, eb.quality_min, eb.quality_max)
    transfer = pop_transfer + lam * (cell.transfer - pop_transfer)
    quality = pop_quality + lam * (quality_clamped - pop_quality)
    continuous = ((registry._continuous(transfer, quality) - registry.continuous_center)
                  / registry.continuous_scale)
    return np.concatenate((continuous, np.eye(len(transfer))), axis=1).astype(np.float32)


def per_channel_rrmse(clean: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    """Temporal RRMSE per channel (46-vector)."""
    num = np.linalg.norm(estimate - clean, axis=-1)
    den = np.clip(np.linalg.norm(clean, axis=-1), 1e-12, None)
    return num / den


def gauss_crps(error: np.ndarray, sigma: np.ndarray) -> float:
    from scipy.stats import norm
    z = error / sigma
    return float(np.mean(sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z)
                                  - 1 / np.sqrt(np.pi))))
