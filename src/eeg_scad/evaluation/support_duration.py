"""Support-duration contracts and summaries."""
from __future__ import annotations

from typing import Iterable, Mapping, Any

import numpy as np

from eeg_scad.evaluation.common_panel_v30 import support_starts


def validate_duration_contract(durations: Iterable[int]) -> dict[str, list[int]]:
    result = {str(value): support_starts(int(value)) for value in durations}
    for duration, starts in result.items():
        seconds = int(duration)
        if seconds == 0:
            if starts:
                raise RuntimeError("zero support must have no windows")
            continue
        if len(starts) != len(set(starts)) or any(start < 0 or start + 200 > seconds * 100 for start in starts):
            raise RuntimeError(f"invalid support prefix: {duration}")
    return result


def aggregate_duration(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = list(rows); result = []
    keys = sorted({(str(row["panel"]), str(row["method"]), int(row["duration_seconds"])) for row in rows})
    for panel, method, duration in keys:
        chosen = [row for row in rows if row["panel"] == panel and row["method"] == method and int(row["duration_seconds"]) == duration]
        for metric in ("risk", "artifact_remaining", "retention", "context_stability", "projector_stability", "encoding_ms"):
            values = np.asarray([float(row[metric]) for row in chosen if str(row.get(metric, "")) not in ("", "nan")])
            if len(values):
                result.append({"panel": panel, "method": method, "duration_seconds": duration, "metric": metric, "mean": float(values.mean()), "median": float(np.median(values)), "rows": len(values)})
    return result


__all__ = ["aggregate_duration", "validate_duration_contract"]

