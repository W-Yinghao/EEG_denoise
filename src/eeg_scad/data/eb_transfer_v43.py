"""V43 RGCC empirical-Bayes gated transfer states.

Implements the preregistered gate (reports/v43_preregistration.md, frozen):

    lambda = clip(tau2 / (tau2 + within/4), 0, 1)        scalar, primary
    h_gated = h_pop + lambda * (h_full - h_pop)
    HARD GATE: effective support < 60 s OR within > 95th percentile of
               fold-train within values  ->  lambda := 0 exactly.

lambda = 0 short-circuits to the registry30 POP signature object so the gated
state is bit-identical to the frozen population state.  Population transfer and
quality are reused from registry30 unchanged (never refit); the registry30
continuous center/scale normalizes every emitted signature, so POP
comparability is exact.  Quality features of the full-prefix fit are clamped to
the fold-train 30-s range before blending.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from eeg_scad.data.artifact_transfer_v41r import TransferRegistry, bipolar_eog, ridge_transfer
from eeg_scad.data.v24_coordinate_contract import robust_center_scale


HARD_GATE_MIN_SECONDS = 60
WITHIN_PERCENTILE = 95.0
SUB_BLOCKS = 4
VARIANTS = ("EB", "RAW", "PERROW")


def eb_lambda(tau2: float, within: float, effective_seconds: float,
              within_threshold: float) -> tuple[float, bool]:
    """Preregistered scalar gate.  Returns (lambda, hard_gate_fired)."""
    if effective_seconds < HARD_GATE_MIN_SECONDS or within > within_threshold:
        return 0.0, True
    return float(np.clip(tau2 / max(tau2 + within / 4.0, 1e-12), 0.0, 1.0)), False


def eb_lambda_rows(tau2_rows: np.ndarray, within_rows: np.ndarray, hard_gate: bool) -> np.ndarray:
    """Secondary per-row gate; the scalar hard gate zeroes every row."""
    if hard_gate:
        return np.zeros_like(np.asarray(tau2_rows, np.float64))
    tau2 = np.asarray(tau2_rows, np.float64)
    within = np.asarray(within_rows, np.float64)
    return np.clip(tau2 / np.maximum(tau2 + within / 4.0, 1e-12), 0.0, 1.0)


@dataclass(frozen=True)
class EBCell:
    owner: str
    session: str
    task: str
    effective_seconds: float
    transfer: np.ndarray          # full-prefix 46x2 fit
    quality: np.ndarray           # unclamped V41R quality of the full fit
    tau2: float
    within: float
    lam: float
    hard_gate: bool
    tau2_rows: np.ndarray
    within_rows: np.ndarray
    lam_rows: np.ndarray


class EBTransferRegistry:
    """Fold-local gated transfer states on top of registry30.

    S1 used 120 and 10 s; the S2 duration set adds 30 and 60 s (S2 addendum).
    Durations below HARD_GATE_MIN_SECONDS (10, 30) always hard-gate to POP."""

    def __init__(self, data: Mapping[str, Any], fold: Mapping[str, Any],
                 registry30: TransferRegistry, seconds: int) -> None:
        if seconds not in (120, 60, 30, 10):
            raise ValueError("V43 EB registries are defined for 10/30/60/120 seconds")
        self.data, self.fold, self.registry30, self.seconds = data, fold, registry30, seconds
        rate = int(data.get("sampling_rate", 100))
        prefix = seconds * rate
        if prefix % SUB_BLOCKS:
            raise ValueError("prefix must split into four equal sub-blocks")
        train_quality = np.stack([cell.quality for key, cell in registry30.cells.items()
                                  if key[0] in fold["train"]])
        self.quality_min = train_quality.min(axis=0)
        self.quality_max = train_quality.max(axis=0)
        fits = {key: self._fit(key, prefix, rate) for key in registry30.cells}
        self.tau2: dict[tuple[str, str], float] = {}
        self.tau2_rows: dict[tuple[str, str], np.ndarray] = {}
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
            [within_values[key][0] for key in sorted(within_values) if key[0] in fold["train"]],
            WITHIN_PERCENTILE))
        self.cells: dict[tuple[str, str, str], EBCell] = {}
        for key, (full, _, quality, effective) in fits.items():
            group = key[1:]
            if group not in self.tau2:
                continue
            within, within_rows = within_values[key]
            lam, hard_gate = eb_lambda(self.tau2[group], within, effective, self.within_threshold)
            lam_rows = eb_lambda_rows(self.tau2_rows[group], within_rows, hard_gate)
            self.cells[key] = EBCell(key[0], key[1], key[2], effective, full, quality,
                                     self.tau2[group], within, lam, hard_gate,
                                     self.tau2_rows[group], within_rows, lam_rows)

    def _fit(self, key: tuple[str, str, str], prefix: int, rate: int):
        eeg, eye, names = self.registry30._load(*key)
        eog = bipolar_eog(eye, names)
        available = min(eeg.shape[1], eog.shape[1])
        effective = min(self.seconds, available / rate)
        center, scale = robust_center_scale(eog[:, :prefix])
        latent = (eog[:, :prefix] - center[:, None]) / scale[:, None]
        scaled_eeg = eeg[:, :prefix] / self.registry30.eeg_scale[:, None]
        full, diagnostics = ridge_transfer(scaled_eeg, latent, self.registry30.ridge_ratio)
        block = prefix // SUB_BLOCKS
        blocks = np.stack([ridge_transfer(scaled_eeg[:, index * block:(index + 1) * block],
                                          latent[:, index * block:(index + 1) * block],
                                          self.registry30.ridge_ratio)[0]
                           for index in range(SUB_BLOCKS)])
        rms = np.sqrt(np.mean((eog[:, :prefix] - center[:, None]) ** 2, axis=1)).clip(1e-8)
        quality = np.array([np.log(rms[0]), np.log(rms[1]), diagnostics["fit_r2"],
                            np.log1p(diagnostics["condition_number"])], dtype=np.float64)
        return full, blocks, quality, effective

    def signature(self, owner: str, session: str, task: str, variant: str = "EB") -> np.ndarray:
        if variant not in VARIANTS:
            raise ValueError(variant)
        cell = self.cells[(owner, session, task)]
        if variant == "EB" and cell.lam == 0.0:
            return self.registry30.signature(owner, session, task, "POP")
        if variant == "PERROW" and cell.hard_gate:
            return self.registry30.signature(owner, session, task, "POP")
        pop_transfer = self.registry30.population_transfer[(session, task)]
        pop_quality = self.registry30.population_quality[(session, task)]
        quality_clamped = np.clip(cell.quality, self.quality_min, self.quality_max)
        if variant == "PERROW":
            transfer = pop_transfer + cell.lam_rows[:, None] * (cell.transfer - pop_transfer)
            quality = pop_quality + cell.lam * (quality_clamped - pop_quality)
        else:
            lam = 1.0 if variant == "RAW" else cell.lam
            transfer = pop_transfer + lam * (cell.transfer - pop_transfer)
            quality = pop_quality + lam * (quality_clamped - pop_quality)
        registry = self.registry30
        continuous = (registry._continuous(transfer, quality) - registry.continuous_center) / registry.continuous_scale
        return np.concatenate((continuous, np.eye(len(transfer))), axis=1).astype(np.float32)

    def operator(self, owner: str, session: str, task: str, variant: str = "EB") -> np.ndarray:
        """Raw 46x2 operator in V41R latent -> scaled-EEG coordinates.

        This is the physical operator for direct subtraction in the V44
        EOG-guided class (query EOG is a declared runtime input there), not the
        normalized signature.  The gate itself is unchanged from V43."""
        if variant not in VARIANTS:
            raise ValueError(variant)
        cell = self.cells[(owner, session, task)]
        pop = self.registry30.population_transfer[(session, task)]
        if variant == "RAW":
            return cell.transfer.copy()
        if variant == "PERROW":
            return pop + cell.lam_rows[:, None] * (cell.transfer - pop)
        return pop + cell.lam * (cell.transfer - pop)

    def manifest_rows(self) -> list[dict[str, Any]]:
        auxiliary = str(self.data["auxiliary_support_owner"])
        rows = []
        for key, cell in sorted(self.cells.items()):
            role = "auxiliary" if cell.owner == auxiliary else next(
                role for role in ("train", "validation", "test") if cell.owner in self.fold[role])
            rows.append({
                "fold": self.fold["fold"], "seconds": self.seconds, "participant": cell.owner,
                "role": role, "session": cell.session, "task": cell.task,
                "effective_seconds": cell.effective_seconds, "tau2": cell.tau2,
                "within": cell.within, "within_threshold": self.within_threshold,
                "lambda": cell.lam, "hard_gate": int(cell.hard_gate),
                "lambda_row_min": float(cell.lam_rows.min()),
                "lambda_row_mean": float(cell.lam_rows.mean()),
                "lambda_row_max": float(cell.lam_rows.max()),
            })
        return rows


__all__ = ["EBCell", "EBTransferRegistry", "HARD_GATE_MIN_SECONDS", "SUB_BLOCKS",
           "VARIANTS", "WITHIN_PERCENTILE", "eb_lambda", "eb_lambda_rows"]
