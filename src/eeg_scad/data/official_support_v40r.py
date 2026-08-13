"""Corrected V31 support-duration contract for V40R."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import numpy as np

from eeg_scad.data.counterfactual_pairs import _load_signal
from eeg_scad.data.v24_coordinate_contract import robust_center_scale


def exact_support(data: Mapping[str, Any], fold: Mapping[str, Any], owner: str, session: str, task: str, seconds: int) -> dict[str, Any] | None:
    if seconds == 0:
        return None
    if seconds not in (10, 30):
        raise ValueError("V40R admits only 0/10/30 second support")
    root = Path(data["v19_derived_root"])
    actual = task
    try:
        eeg, eog = _load_signal(root, owner, session, task)
    except FileNotFoundError:
        actual = next(candidate for candidate in data["tasks"] if candidate != task)
        eeg, eog = _load_signal(root, owner, session, actual)
    prefix, length = seconds * 100, 200
    starts = list(range(0, prefix - length + 1, length))
    scale_path = Path(data["v24_derived_root"]) / f"fold_{fold['fold']}" / "eeg_scale.npy"
    eeg_scale = np.load(scale_path)
    center, scale = robust_center_scale(eog[:, :prefix])
    seeg = np.stack([eeg[:, start:start + length] / eeg_scale[:, None] for start in starts]).astype(np.float32)
    seog = np.stack([(eog[:, start:start + length] - center[:, None]) / scale[:, None] for start in starts]).astype(np.float32)
    _load_signal.cache_clear()
    return {"eeg": seeg, "eog": seog, "starts": starts, "seconds": seconds, "actual_task": actual, "normalization_samples": prefix}


def validate_support_episode(episode: dict[str, Any] | None, seconds: int) -> bool:
    if seconds == 0:
        return episode is None
    assert episode is not None
    starts = episode["starts"]
    return starts == list(range(0, seconds * 100 - 199, 200)) and len(set(starts)) == len(starts) and episode["normalization_samples"] == seconds * 100


__all__ = ["exact_support", "validate_support_episode"]
