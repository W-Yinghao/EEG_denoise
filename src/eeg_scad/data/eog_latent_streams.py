"""Correct-coordinate paired and natural-reference streams for V24."""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from eeg_scad.data.counterfactual_pairs import _load_raw, _load_signal, _query_operator, fold_eeg_scale
from eeg_scad.data.v24_coordinate_contract import canonical_operator, robust_center_scale


def ridge_projection(operator: np.ndarray, value: np.ndarray, ratio: float = 0.01) -> np.ndarray:
    c = np.asarray(operator, dtype=np.float64)
    gram = c.T @ c
    ridge = ratio * max(float(np.trace(gram) / max(1, gram.shape[0])), 1e-9)
    return np.linalg.solve(gram + ridge * np.eye(gram.shape[0]), c.T @ np.asarray(value, dtype=np.float64))


class EOGStreamSampler:
    """Online sampler whose targets obey ``A~=C_query~ Z_e`` exactly."""

    def __init__(self, data: Mapping[str, Any], fold: Mapping[str, Any], split: str, seed: int) -> None:
        self.data = data
        self.fold = fold
        self.split = split
        self.training = list(fold["train"])
        self.participants = list(fold[split])
        self.root = Path(data["v19_derived_root"])
        self.eeg_scale = fold_eeg_scale(data, self.training)
        offset = {"train": 0, "validation": 1, "test": 2}[split]
        self.rng = np.random.Generator(np.random.PCG64DXSM(seed + int(fold["fold"]) * 1000 + offset))
        self.operators: dict[tuple[str, str, str], np.ndarray] = {}
        self.coordinates: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
        self.query_operators: dict[tuple[str, str, str], np.ndarray] = {}
        owners = set(self.training + self.participants + [str(data["auxiliary_support_owner"])])
        for owner, session, task in itertools.product(owners, data["sessions"], data["tasks"]):
            try:
                raw, fallback = _load_raw(self.root, owner, session, task, data["tasks"])
                actual_task = task if not fallback else next(value for value in data["tasks"] if value != task)
                _, eog = _load_signal(self.root, owner, session, actual_task)
            except FileNotFoundError:
                continue
            center, scale = robust_center_scale(eog[:, : int(data["support_samples"])])
            self.coordinates[(owner, session, task)] = (center, scale)
            self.operators[(owner, session, task)] = canonical_operator(raw["C_raw"], self.eeg_scale, scale)
        self.population: dict[tuple[str, str], np.ndarray] = {}
        for session, task in itertools.product(data["sessions"], data["tasks"]):
            values = [self.operators[(owner, session, task)] for owner in self.training if (owner, session, task) in self.operators]
            if values:
                self.population[(session, task)] = np.mean(values, axis=0)
        self.records: list[tuple[str, str, str]] = []
        for participant, session, task in itertools.product(self.participants, data["sessions"], data["tasks"]):
            path = _query_operator(self.root, participant, session, task)
            key = (participant, session, task)
            if not path.is_file() or key not in self.operators or (session, task) not in self.population:
                continue
            with np.load(path, allow_pickle=False) as archive:
                raw_query = np.asarray(archive["C_query"], dtype=np.float64)
            _, scale = self.coordinates[key]
            self.query_operators[key] = canonical_operator(raw_query, self.eeg_scale, scale)
            self.records.append(key)
        self.sources: list[tuple[str, str, str, np.ndarray, np.ndarray]] = []
        for owner, session, task in itertools.product(self.training, data["sessions"], data["tasks"]):
            try:
                eeg, eog = _load_signal(self.root, owner, session, task)
            except FileNotFoundError:
                continue
            if eeg.shape[-1] > int(data["qnatural_start"]) + int(data["window_samples"]):
                self.sources.append((owner, session, task, eeg, eog))
        if not self.records or not self.sources:
            raise RuntimeError(f"incomplete V24 stream for fold {fold['fold']} split {split}")

    def state(self) -> dict[str, Any]:
        return self.rng.bit_generator.state

    def set_state(self, state: dict[str, Any]) -> None:
        self.rng.bit_generator.state = state

    def _source(self, excluded: set[str], session: str, task: str) -> tuple[str, np.ndarray, np.ndarray]:
        choices = [row for row in self.sources if row[0] not in excluded and row[1] == session and row[2] == task]
        if not choices:
            choices = [row for row in self.sources if row[0] not in excluded]
        row = choices[int(self.rng.integers(len(choices)))]
        return row[0], row[3], row[4]

    def _window(self, value: np.ndarray) -> np.ndarray:
        length = int(self.data["window_samples"])
        low = int(self.data["qnatural_start"])
        start = int(self.rng.integers(low, max(low + 1, value.shape[-1] - length + 1)))
        return np.asarray(value[:, start : start + length], dtype=np.float64)

    def sample_paired(self, batch_size: int, zero_proportion: float = 0.10) -> dict[str, Any]:
        keys = ("x", "y", "artifact", "latent", "c0", "cs", "cw", "cquery", "ds", "dw", "q0")
        values: dict[str, list[np.ndarray]] = {key: [] for key in keys}
        meta = []
        for _ in range(batch_size):
            recipient, session, task = self.records[int(self.rng.integers(len(self.records)))]
            key = (recipient, session, task)
            c0 = self.population[(session, task)]
            cs = self.operators[key]
            cq = self.query_operators[key]
            wrong_pool = [owner for owner in self.training + self.participants + [str(self.data["auxiliary_support_owner"])] if owner != recipient and (owner, session, task) in self.operators]
            wrong = wrong_pool[int(self.rng.integers(len(wrong_pool)))]
            cw = self.operators[(wrong, session, task)]
            xowner, xeeg, _ = self._source({recipient}, session, task)
            eowner, _, eeog = self._source({recipient, xowner}, session, task)
            x = self._window(xeeg) / self.eeg_scale[:, None]
            physical_e = self._window(eeog)
            center, scale = self.coordinates[key]
            latent = (physical_e - center[:, None]) / scale[:, None]
            gain = float(self.rng.choice([0.35, 0.7, 1.15]) * self.rng.uniform(0.85, 1.15))
            latent = gain * latent
            artifact = cq @ latent
            zero = bool(self.rng.random() < zero_proportion)
            if zero:
                latent = np.zeros_like(latent)
                artifact = np.zeros_like(artifact)
            y = x + artifact
            q0 = ridge_projection(c0, y)
            for name, value in (("x", x), ("y", y), ("artifact", artifact), ("latent", latent), ("c0", c0), ("cs", cs), ("cw", cw), ("cquery", cq), ("ds", cs - c0), ("dw", cw - c0), ("q0", q0)):
                values[name].append(value)
            meta.append({"participant": recipient, "session": session, "task": task, "clean_owner": xowner, "eog_owner": eowner, "operator_recipient": recipient, "wrong_owner": wrong, "strict_three_way": int(len({recipient, xowner, eowner}) == 3), "gain": gain, "zero_artifact": int(zero)})
        return {**{key: np.asarray(value, dtype=np.float32) for key, value in values.items()}, "meta": meta, "stream": "paired"}

    def sample_natural(self, batch_size: int, evaluator: bool = True) -> dict[str, Any]:
        keys = ("y", "latent", "teacher_artifact", "c0", "cs", "cw", "ds", "dw", "q0")
        values: dict[str, list[np.ndarray]] = {key: [] for key in keys}
        meta = []
        for _ in range(batch_size):
            recipient, session, task = self.records[int(self.rng.integers(len(self.records)))]
            key = (recipient, session, task)
            eeg, eog = _load_signal(self.root, recipient, session, task)
            y = self._window(eeg) / self.eeg_scale[:, None]
            # A separately sampled synchronized start is required. Recreate it
            # deterministically by drawing once and slicing both arrays.
            length = int(self.data["window_samples"]); low = int(self.data["qnatural_start"])
            start = int(self.rng.integers(low, max(low + 1, eeg.shape[-1] - length + 1)))
            y = np.asarray(eeg[:, start : start + length], dtype=np.float64) / self.eeg_scale[:, None]
            physical_e = np.asarray(eog[:, start : start + length], dtype=np.float64)
            center, scale = self.coordinates[key]
            latent = (physical_e - center[:, None]) / scale[:, None]
            c0 = self.population[(session, task)]; cs = self.operators[key]; cq = self.query_operators[key]
            wrong_pool = [owner for owner in self.training + self.participants + [str(self.data["auxiliary_support_owner"])] if owner != recipient and (owner, session, task) in self.operators]
            wrong = wrong_pool[int(self.rng.integers(len(wrong_pool)))]; cw = self.operators[(wrong, session, task)]
            teacher = cq @ latent
            q0 = ridge_projection(c0, y)
            for name, value in (("y", y), ("latent", latent), ("teacher_artifact", teacher), ("c0", c0), ("cs", cs), ("cw", cw), ("ds", cs-c0), ("dw", cw-c0), ("q0", q0)):
                values[name].append(value)
            meta.append({"participant": recipient, "session": session, "task": task, "wrong_owner": wrong, "start": start})
        output = {key: np.asarray(value, dtype=np.float32) for key, value in values.items()}
        if not evaluator:
            for forbidden in ("latent", "teacher_artifact"):
                output.pop(forbidden, None)
        return {**output, "meta": meta, "stream": "natural"}


def generate_bank(sampler: EOGStreamSampler, samples: int, seed: int, natural_fraction: float = 0.30) -> dict[str, Any]:
    state = sampler.state()
    sampler.rng = np.random.Generator(np.random.PCG64DXSM(seed))
    paired_count = int(round(samples * (1.0 - natural_fraction)))
    paired = sampler.sample_paired(paired_count)
    natural = sampler.sample_natural(samples - paired_count)
    sampler.set_state(state)
    return {"paired": paired, "natural": natural}


__all__ = ["EOGStreamSampler", "generate_bank", "ridge_projection"]

