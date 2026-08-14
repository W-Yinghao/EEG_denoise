"""Unit-sphere electrode positions per montage (hardware descriptors only).

Standard-label positions come from mne's standard_1005 montage, re-centered
and L2-normalized to the unit sphere.  The 14 MobileBCI cEEGrid around-ear
electrodes have no standard coordinates; they are laid out on a deterministic
C-shaped ring around each ear (documented approximation — a hardware
descriptor with zero subject content, like the lift itself).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np


MOBILEBCI_SCALP = ("Fp1", "Fp2", "AFz", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2",
                   "FC6", "C3", "Cz", "C4", "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz",
                   "P4", "P8", "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2")
MOBILEBCI_EAR = ("L1", "L2", "L4", "L5", "L6", "L7", "L9", "L10",
                 "R1", "R2", "R4", "R5", "R7", "R8")
MOBILEBCI_CHANNELS = MOBILEBCI_SCALP + MOBILEBCI_EAR
KLADOS_CHANNELS = ("FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
                   "F7", "F8", "T3", "T4", "T5", "T6", "Fz", "Cz", "Pz")
BCI2B_CHANNELS = ("C3", "Cz", "C4")
_CASE_FIX = {"FP1": "Fp1", "FP2": "Fp2", "FPZ": "Fpz"}


@lru_cache(maxsize=1)
def _standard_positions() -> dict[str, np.ndarray]:
    import mne

    montage = mne.channels.make_standard_montage("standard_1005")
    raw = montage.get_positions()["ch_pos"]
    stack = np.stack([np.asarray(value, np.float64) for value in raw.values()])
    center = stack.mean(axis=0)
    return {name: np.asarray(value, np.float64) - center for name, value in raw.items()}


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def _resolve(labels, allow_missing: bool = False):
    table = _standard_positions()
    positions, kept, missing = [], [], []
    for label in labels:
        name = _CASE_FIX.get(str(label).upper(), str(label))
        if name not in table:
            name = _CASE_FIX.get(str(label), str(label))
        if name in table:
            positions.append(_unit(table[name]))
            kept.append(str(label))
        elif allow_missing:
            missing.append(str(label))
        else:
            raise KeyError(f"no standard position for electrode {label}")
    return np.stack(positions), kept, missing


def _ceegrid_ring(side: str, count: int = 10) -> np.ndarray:
    """Deterministic C-shaped ring around the ear (approximate cEEGrid)."""
    azimuth = np.deg2rad(100.0) * (1.0 if side == "R" else -1.0)
    center = _unit(np.array([np.sin(azimuth), np.cos(azimuth) * 0.15 - 0.05, -0.15]))
    up = _unit(np.array([0.0, 0.0, 1.0]) - center * center[2])
    front = _unit(np.cross(up, center) * (1.0 if side == "R" else -1.0))
    ring = np.deg2rad(16.0)
    angles = np.deg2rad(np.linspace(-135.0, 135.0, count))
    return np.stack([_unit(np.cos(ring) * center
                           + np.sin(ring) * (np.cos(a) * front + np.sin(a) * up))
                     for a in angles])


def mobilebci_positions() -> np.ndarray:
    scalp, _, _ = _resolve(MOBILEBCI_SCALP)
    left = _ceegrid_ring("L")
    right = _ceegrid_ring("R")
    ear_lookup = {f"L{i + 1}": left[i] for i in range(10)}
    ear_lookup.update({f"R{i + 1}": right[i] for i in range(10)})
    ears = np.stack([ear_lookup[name] for name in MOBILEBCI_EAR])
    return np.concatenate((scalp, ears), axis=0)


def klados_positions() -> np.ndarray:
    positions, _, _ = _resolve(KLADOS_CHANNELS)
    return positions


def bci2b_positions() -> np.ndarray:
    positions, _, _ = _resolve(BCI2B_CHANNELS)
    return positions


def sgeyesub_positions(labels) -> tuple[np.ndarray, list[str], list[str]]:
    """Resolve an SGEYESUB layout's ordered labels; unresolvable labels (named
    EOG electrodes) are excluded from the lift and reported."""
    return _resolve(labels, allow_missing=True)


__all__ = ["BCI2B_CHANNELS", "KLADOS_CHANNELS", "MOBILEBCI_CHANNELS", "bci2b_positions",
           "klados_positions", "mobilebci_positions", "sgeyesub_positions"]
