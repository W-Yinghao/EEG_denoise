"""Participant-first aggregation helpers for CalibSDEdit V26."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping

import numpy as np


LOWER_IS_BETTER = {
    "rrmse_temporal",
    "rrmse_spectral",
    "artifact_rrmse",
    "remaining_ratio",
    "blink_residual_ratio",
    "frontal_topography_residual_proxy",
    "psd_distortion",
    "covariance_distortion",
}


def participant_first(rows: Iterable[Mapping[str, object]], metrics: list[str]) -> list[dict[str, object]]:
    cells: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        cells[(str(row["panel"]), str(row["participant"]), str(row["method"]))].append(row)
    output = []
    for (panel, participant, method), values in sorted(cells.items()):
        item: dict[str, object] = {"panel": panel, "participant": participant, "method": method}
        for metric in metrics:
            vector = np.asarray([float(v[metric]) for v in values], dtype=float)
            item[metric] = float(np.nanmean(vector))
        output.append(item)
    return output


def contrast(rows: Iterable[Mapping[str, object]], first: str, second: str, metric: str) -> list[dict[str, object]]:
    values = {(str(r["participant"]), str(r["method"])): float(r[metric]) for r in rows}
    sign = -1.0 if metric in LOWER_IS_BETTER else 1.0
    participants = sorted({participant for participant, method in values if method == first} & {participant for participant, method in values if method == second})
    return [{"participant": p, "first": first, "second": second, "metric": metric, "effect": sign * (values[p, first] - values[p, second])} for p in participants]


def bootstrap(vector: np.ndarray, seed: int = 20260831, replicates: int = 20000) -> dict[str, object]:
    vector = np.asarray(vector, dtype=float)
    generator = np.random.Generator(np.random.PCG64DXSM(seed))
    draws = vector[generator.integers(0, len(vector), size=(replicates, len(vector)))].mean(1)
    return {"mean": float(vector.mean()), "median": float(np.median(vector)), "positive": int((vector > 0).sum()), "participants": int(len(vector)), "bootstrap_low": float(np.quantile(draws, .025)), "bootstrap_high": float(np.quantile(draws, .975))}


__all__ = ["participant_first", "contrast", "bootstrap", "LOWER_IS_BETTER"]
