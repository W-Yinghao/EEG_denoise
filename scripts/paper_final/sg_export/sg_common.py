"""Shared contract for the SGEYESUB-style figure arrays (SERVER_EXPORT_FIGURE_ARRAYS_v4.md).

Every exporter and figure script imports from here so that the session->condition
mapping, channel lists, EOG masks, spectral settings and output locations are
defined once. CPU-only helpers; GPU sampling lives in the exporters that need it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PF = HERE.parent                                   # scripts/paper_final
sys.path.insert(0, str(PF))
from pf_common import ARRAYS, OUT, REPO, SEALED, SEED, V44_ROOT, load_model, stat  # noqa: E402,F401

sys.path.insert(0, str(V44_ROOT / "src"))

ARRAYS_OUT = OUT / "paper_final_arrays"            # the path the export doc asks for
FIG_OUT = REPO / "artifacts/figures/v3"            # figstyle.OUT on this branch
RATE = 100
WINDOW = 512
CALIB = 12000                                      # first 120 s = calibration prefix (D-wave convention)

# --- movement condition: MobileBCI README lists standing / slow walking / fast walking /
# slight running at 0 / 0.8 / 1.6 / 2.0 m/s; ses-01 is ERP-only, ses-02..05 carry both
# tasks. The project uses ses-02/03/04 only. The mapping below follows the README order;
# no sidecar in the BIDS tree states it, so every array carries CONDITION_NOTE.
SESSION_CONDITION = {"ses-02": "standing", "ses-03": "slow_walking", "ses-04": "fast_walking"}
SESSION_SPEED_MPS = {"ses-02": 0.0, "ses-03": 0.8, "ses-04": 1.6}
CONDITION_NOTE = ("session->speed mapping inferred from the dataset README order "
                  "(ses-02..05 = 0/0.8/1.6/2.0 m/s); ses-05 (running) is not part of "
                  "this project's protocol; confirm against Lee et al. 2021 before labelling")

# --- channels (order of the prepared 46-channel records)
EEG_NAMES = [str(n) for n in np.load(ARRAYS / "t6_scalp_improvement.npz", allow_pickle=False)["eeg_names"]]
SCALP = EEG_NAMES[:32]
EAR = EEG_NAMES[32:]
NINE = ["F3", "Fz", "F4", "C3", "Cz", "C4", "P3", "Pz", "P4"]
EXTRA_ROW = ["AFz", "Fp1", "Fp2"]
EXEMPLAR_STACK = ["Fp1", "Fp2", "AFz", "F3", "Fz", "F4", "C3", "Cz", "C4", "P3", "Pz", "P4"]
IDX = {n: i for i, n in enumerate(EEG_NAMES)}

# --- arm names as stored by the D-wave pass and their export labels
DWAVE_ARMS = ("RAW", "MATCH", "POP", "NO_A0", "LINEAR", "ICA", "ASR", "SGEYESUB")
EXPORT_LABEL = {"RAW": "raw", "MATCH": "subject_calibrated", "POP": "population_calibrated",
                "NO_A0": "unguided", "LINEAR": "calibrated_linear", "WRONG_gated": "mismatched_rule_applied",
                "SHUFFLED": "temporally_shuffled", "ICA": "ica", "ASR": "asr", "SGEYESUB": "eye_subspace_projection"}

# --- spectral settings (export doc E3): Welch 2-s window, 1-s overlap, 0.5-15 Hz at 100 Hz
WELCH_NPERSEG, WELCH_NOVERLAP, FMIN, FMAX = 200, 100, 0.5, 15.0
BANDS = {"delta": (0.5, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 13.0)}


def fold_of(participant: str):
    """(fold_id, fold) whose TEST set holds `participant` (dev cohort), else None."""
    from eeg_scad.cli.run_v43 import configs
    _, folds, _ = configs()
    for i, fold in enumerate(folds):
        if participant in fold["test"]:
            return i, fold
    return None


def registry(fold):
    from eeg_scad.cli.run_v43 import configs
    from eeg_scad.data.artifact_transfer_v41r import TransferRegistry
    from eeg_scad.data.eb_transfer_v43 import EBTransferRegistry
    data, _, _ = configs()
    r30 = TransferRegistry(data, fold, 30, .05)
    return data, r30, EBTransferRegistry(data, fold, r30, 120)


def record(r30, key):
    """(eeg 46xT scaled by the fold eeg_scale, bipolar eog 2xT latent-scaled per cell, raw eog 2xT)."""
    from eeg_scad.data.artifact_transfer_v41r import bipolar_eog
    eeg, eye, names = r30._load(*key)
    eog = bipolar_eog(eye, names)
    cell = r30.cells[key]
    drive = (eog - cell.eog_center[:, None]) / cell.eog_scale[:, None]
    return eeg / r30.eeg_scale[:, None], drive, eog


def eog_masks(drive: np.ndarray):
    """Per-sample masks exactly as run_v44._natural_metrics: low = bottom 30% of EOG
    energy, high = top 30%."""
    energy = np.sqrt(np.mean(drive * drive, axis=0))
    return energy <= np.quantile(energy, .3), energy >= np.quantile(energy, .7)


def welch(x: np.ndarray, nperseg: int | None = None):
    """Welch PSD along the last axis at RATE; returns (freqs, psd) restricted to FMIN..FMAX."""
    from scipy.signal import welch as _welch
    n = x.shape[-1]
    nperseg = min(nperseg or WELCH_NPERSEG, n)
    f, p = _welch(x, fs=RATE, nperseg=nperseg, noverlap=min(WELCH_NOVERLAP, nperseg // 2), axis=-1)
    keep = (f >= FMIN) & (f <= FMAX)
    return f[keep], p[..., keep]


def dwave_tiles(participant: str, session: str, task: str):
    """D-wave natural tiles for one cell: dict arm->(n,46,512), plus starts, veog_rms, hard_gate.
    d1 = SSVEP, d2 = ERP. Reference methods come from the cpuarms_ file."""
    prefix = "d1" if task == "SSVEP" else "d2"
    cell = f"{participant}|{session}|{task}"
    den = OUT / "dwave/denoised"
    out = {}
    with np.load(den / f"{prefix}_{cell}.npz", allow_pickle=False) as d:
        for k in d.files:
            out[k] = d[k]
    cpu = den / f"cpuarms_{cell}.npz"
    if cpu.is_file():
        with np.load(cpu, allow_pickle=False) as d:
            for k in ("ICA", "ASR", "SGEYESUB"):
                if k in d.files:
                    out[k] = d[k]
    return out


def save_npz(name: str, **arrays):
    """Write to ARRAYS_OUT with the condition note and a JSON manifest of shapes."""
    ARRAYS_OUT.mkdir(parents=True, exist_ok=True)
    arrays.setdefault("condition_note", np.asarray(CONDITION_NOTE))
    arrays.setdefault("eeg_names", np.asarray(EEG_NAMES))
    path = ARRAYS_OUT / name
    np.savez_compressed(path, **arrays)
    manifest = {k: {"shape": list(np.asarray(v).shape), "dtype": str(np.asarray(v).dtype)}
                for k, v in arrays.items()}
    (ARRAYS_OUT / (name.replace(".npz", "") + ".manifest.json")).write_text(
        json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({"wrote": str(path), "keys": list(manifest)}))
    return path
