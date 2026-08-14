"""W2 data pipeline: clean-carrier caches, EOG drive bank, montage registry.

Caches store CHANNEL-space windows plus a montage tag; lifting to canonical
space happens in the training loop (cheap matmul), which also applies the
montage-mask augmentation.  Sealed subsets are excluded by construction
(MobileBCI dev-16; SHU sessions 1-3; PhysioMotion development subjects;
SGEYESUB development studies).  Truth-free corpora contribute LOW-ARTIFACT
window selections; paired corpora contribute clean carriers.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from eeg_scad.data.v24_coordinate_contract import robust_center_scale
from eeg_chart.positions import (bci2b_positions, klados_positions, mobilebci_positions,
                                 sgeyesub_positions)


CACHE = Path("/projects/EEG-foundation-model/derived/denoiseNet/flagship_m13/cache")
WINDOW = 512
PHYSIOMOTION_SEALED = {3, 4, 7, 12, 15, 17, 19, 24, 25, 28}
SHU_ROOT = Path("/projects/EEG-foundation-model/SHU-MI-cbramod/mat")
SHU_CHANNELS = ("FP1", "FP2", "FZ", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5", "FC6",
                "CZ", "C3", "C4", "T3", "T4", "A1", "A2", "CP1", "CP2", "CP5", "CP6",
                "PZ", "P3", "P4", "T5", "T6", "PO3", "PO4", "OZ", "O1", "O2")
EEGDN_ROOT = Path("/projects/EEG-foundation-model/derived/denoiseNet/scad_v22/baseline/eegdus_public")


def _resample_100(data: np.ndarray, fs: float) -> np.ndarray:
    """Registered rate unification: every carrier is resampled to 100 Hz before
    windowing, so canonical windows share temporal statistics across corpora."""
    from fractions import Fraction

    from scipy.signal import resample_poly

    if abs(fs - 100.0) < 1e-6:
        return data
    ratio = Fraction(100, int(round(fs))).limit_denominator(64)
    return resample_poly(data, ratio.numerator, ratio.denominator, axis=1)


def _windows_lowest_energy(data: np.ndarray, count: int, energy_of=None) -> np.ndarray:
    """Non-overlapping 512-sample windows with the lowest energy statistic."""
    starts = np.arange(0, data.shape[1] - WINDOW + 1, WINDOW)
    if len(starts) == 0:
        return np.zeros((0, data.shape[0], WINDOW), np.float32)
    stat = np.asarray([float(np.sqrt(np.mean((energy_of if energy_of is not None else data)
                                             [:, s:s + WINDOW] ** 2))) for s in starts])
    keep = starts[np.argsort(stat)[:count]]
    return np.stack([data[:, s:s + WINDOW] for s in keep]).astype(np.float32)


def _scale(windows_source: np.ndarray) -> np.ndarray:
    _, scale = robust_center_scale(windows_source)
    return scale


def harvest_mobilebci() -> dict:
    from eeg_scad.cli.run_v43 import configs
    from eeg_scad.data.artifact_transfer_v41r import bipolar_eog

    data, folds, _ = configs()
    root = Path(data["v19_derived_root"])
    windows, subjects, drives = [], [], []
    dev = sorted({p for fold in folds for p in fold["train"] + fold["validation"] + fold["test"]})
    for participant in dev:
        for path in sorted((root / "prepared" / participant).glob("*.npz")):
            with np.load(path, allow_pickle=False) as archive:
                eeg = np.asarray(archive["eeg"], np.float64)
                eye = np.asarray(archive["eog"], np.float64)
                names = [str(v) for v in archive["eog_names"]]
            eog = bipolar_eog(eye, names)
            q = int(data["qnatural_start"])
            scale = _scale(eeg[:, q:])
            scaled = eeg[:, q:] / scale[:, None]
            center, escale = robust_center_scale(eog[:, :12000])
            latent = (eog[:, q:] - center[:, None]) / escale[:, None]
            picked = _windows_lowest_energy(scaled, 12, energy_of=np.broadcast_to(
                np.sqrt(np.mean(latent ** 2, axis=0, keepdims=True)), latent.shape))
            windows.append(picked)
            subjects += [participant] * len(picked)
            starts = np.arange(0, latent.shape[1] - WINDOW + 1, WINDOW)
            energy = np.asarray([float(np.mean(latent[:, s:s + WINDOW] ** 2)) for s in starts])
            for s in starts[np.argsort(energy)[-6:]]:
                drives.append(latent[:, s:s + WINDOW].astype(np.float32))
    return {"windows": np.concatenate(windows), "subjects": np.asarray(subjects),
            "drives": np.stack(drives), "montage": "mobilebci"}


def harvest_klados() -> dict:
    import yaml
    from eeg_cgdr.data.klados import load_klados_records

    root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load((root / "configs/cgdr/klados_v4.yaml").read_text())
    windows, subjects, drives = [], [], []
    for record in load_klados_records(config):
        clean = np.asarray(record.clean, np.float64)
        eog = np.stack((np.asarray(record.veog, np.float64).reshape(-1),
                        np.asarray(record.heog, np.float64).reshape(-1)))
        length = min(clean.shape[1], eog.shape[1])
        clean, eog = _resample_100(clean[:, :length], 200.0), _resample_100(eog[:, :length], 200.0)
        scale = _scale(clean)
        scaled = clean / scale[:, None]
        center, escale = robust_center_scale(eog)
        latent = (eog - center[:, None]) / escale[:, None]
        starts = np.arange(0, scaled.shape[1] - WINDOW + 1, WINDOW)
        for s in starts:
            windows.append(scaled[:, s:s + WINDOW].astype(np.float32)[None])
            subjects.append(f"P{(record.record_id - 1) % 27 + 1:02d}")
        energy = np.asarray([float(np.mean(latent[:, s:s + WINDOW] ** 2)) for s in starts])
        for s in starts[np.argsort(energy)[-3:]]:
            drives.append(latent[:, s:s + WINDOW].astype(np.float32))
    return {"windows": np.concatenate(windows), "subjects": np.asarray(subjects),
            "drives": np.stack(drives), "montage": "klados"}


def harvest_bci2b() -> dict:
    import mne

    root = Path("/projects/EEG-foundation-model/BCI-IV")
    windows, subjects, drives = [], [], []
    for subject in range(1, 10):
        for run in (1, 2, 3):
            matches = sorted(root.glob(f"B{subject:02d}{run:02d}T.gdf"))
            if not matches:
                continue
            raw = mne.io.read_raw_gdf(matches[0], preload=True, verbose="error")
            picks = np.asarray(raw.get_data(), np.float64) * 1e6
            fs = float(raw.info["sfreq"])
            eeg = _resample_100(np.nan_to_num(picks[:3]), fs)
            eog = _resample_100(np.nan_to_num(picks[3:6]), fs)
            scale = _scale(eeg)
            scaled = eeg / scale[:, None]
            center, escale = robust_center_scale(eog)
            standardized = (eog - center[:, None]) / escale[:, None]
            pca = np.linalg.svd(np.cov(standardized), full_matrices=False)[0][:, :2].T
            latent = pca @ standardized
            picked = _windows_lowest_energy(scaled, 20, energy_of=np.broadcast_to(
                np.sqrt(np.mean(latent ** 2, axis=0, keepdims=True)), latent.shape))
            windows.append(picked)
            subjects += [f"B{subject:02d}"] * len(picked)
            starts = np.arange(0, latent.shape[1] - WINDOW + 1, WINDOW)
            energy = np.asarray([float(np.mean(latent[:, s:s + WINDOW] ** 2)) for s in starts])
            for s in starts[np.argsort(energy)[-4:]]:
                drives.append(latent[:, s:s + WINDOW].astype(np.float32))
    return {"windows": np.concatenate(windows), "subjects": np.asarray(subjects),
            "drives": np.stack(drives), "montage": "bci2b"}


def harvest_shu() -> dict:
    import scipy.io as sio

    windows, subjects = [], []
    for subject in range(1, 26):
        for session in (1, 2, 3):        # sessions 4/5 sealed
            path = SHU_ROOT / f"sub-{subject:03d}_ses-{session:02d}_task_motorimagery_eeg.mat"
            if not path.is_file():
                continue
            data = np.asarray(sio.loadmat(path)["data"], np.float64)  # trials x 32 x 1000
            joined = _resample_100(data.transpose(1, 0, 2).reshape(data.shape[1], -1), 250.0)
            scale = _scale(joined)
            picked = _windows_lowest_energy(joined / scale[:, None], 20)
            windows.append(picked)
            subjects += [f"sub-{subject:03d}"] * len(picked)
    return {"windows": np.concatenate(windows), "subjects": np.asarray(subjects),
            "montage": "shu"}


def harvest_physiomotion() -> dict:
    import mne

    root = Path("/projects/EEG-foundation-model/PhysioMotion_Artifact")
    windows, subjects, kept_labels = [], [], None
    for subject_dir in sorted(root.glob("sub-*")):
        try:
            number = int(subject_dir.name.split("-")[1])
        except ValueError:
            continue
        if number in PHYSIOMOTION_SEALED:
            continue
        runs = sorted((subject_dir / "eeg").glob("*_eeg.edf"))
        if not runs:
            continue
        try:
            raw = mne.io.read_raw_edf(runs[0], preload=True, verbose="error")
            picks = mne.pick_types(raw.info, eeg=True)
            labels = [raw.ch_names[i] for i in picks]
            from eeg_chart.positions import _resolve
            _, kept, missing = _resolve(labels, allow_missing=True)
            index = [labels.index(name) for name in kept]
            data = _resample_100(raw.get_data(picks=picks)[index] * 1e6,
                                 float(raw.info["sfreq"]))
            scale = _scale(data)
            picked = _windows_lowest_energy(data / scale[:, None], 15)
            if kept_labels is None:
                kept_labels = kept
            if kept != kept_labels:
                continue  # keep one consistent montage; reason-coded skip
            windows.append(picked)
            subjects += [subject_dir.name] * len(picked)
        except Exception:
            continue
    return {"windows": np.concatenate(windows), "subjects": np.asarray(subjects),
            "montage": "physiomotion", "labels": kept_labels}


def harvest_sgeyesub() -> list[dict]:
    """Per-record caches from the DEVELOPMENT studies (study01/study03), read
    directly via mne's EEGLAB reader (training carriers only; the SGEYESUB
    query-governance loader is for its operator experiments, not needed here)."""
    import mne

    data_root = Path("/projects/EEG-foundation-model/sgeyesub/osf-2qgrd")
    caches = []
    for study in ("study01", "study03"):
        for set_path in sorted((data_root / study).glob("*.set")):
            try:
                raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose="error")
                labels = list(raw.ch_names)
                eog_index = [i for i, name in enumerate(labels)
                             if "EOG" in name.upper()]
                from eeg_chart.positions import _resolve
                eeg_candidates = [name for i, name in enumerate(labels) if i not in eog_index]
                _, kept, _ = _resolve(eeg_candidates, allow_missing=True)
                index = [labels.index(name) for name in kept]
                data = _resample_100(np.asarray(raw.get_data()[index], np.float64) * 1e6,
                                     float(raw.info["sfreq"]))
                eog = _resample_100(np.asarray(raw.get_data()[eog_index], np.float64) * 1e6,
                                    float(raw.info["sfreq"])) if eog_index else None
                scale = _scale(data)
                energy = None
                if eog is not None:
                    length = min(data.shape[1], eog.shape[1])
                    data = data[:, :length]
                    energy = np.broadcast_to(np.sqrt(np.mean(eog[:, :length] ** 2, axis=0,
                                                             keepdims=True)), (1, length))
                picked = _windows_lowest_energy(data / scale[:, None], 10, energy_of=energy)
                if len(picked) == 0:
                    continue
                caches.append({"windows": picked,
                               "subjects": np.asarray([set_path.stem] * len(picked)),
                               "montage": f"sgeyesub_{study}_{set_path.stem}",
                               "labels": kept})
            except Exception:
                continue
    return caches


def harvest_eegdenoisenet() -> dict:
    with np.load(EEGDN_ROOT / "train.npz", allow_pickle=False) as archive:
        clean = np.asarray(archive["clean"], np.float64)
    clean = _resample_100(clean.reshape(len(clean), -1), 256.0)   # 512@256Hz -> 200@100Hz
    usable = (len(clean) // 3) * 3
    stitched = clean[:usable].reshape(usable // 3, -1)[:, :WINDOW]  # 3 segments -> 600 -> 512
    stitched = stitched[:, None, :]
    scale = np.maximum(np.std(stitched, axis=(1, 2), keepdims=True), 1e-6)
    return {"windows": (stitched / scale).astype(np.float32),
            "subjects": np.asarray(["eegdn"] * len(stitched)), "montage": "eegdn_1ch"}


def montage_positions(tag: str, labels=None) -> np.ndarray:
    from eeg_chart.positions import _resolve

    if tag == "mobilebci":
        return mobilebci_positions()
    if tag == "klados":
        return klados_positions()
    if tag == "bci2b":
        return bci2b_positions()
    if tag == "shu":
        positions, _, _ = _resolve(SHU_CHANNELS, allow_missing=True)
        return positions
    if labels is not None:
        positions, _, _ = _resolve(labels, allow_missing=True)
        return positions
    raise KeyError(tag)


__all__ = ["CACHE", "WINDOW", "harvest_bci2b", "harvest_eegdenoisenet", "harvest_klados",
           "harvest_mobilebci", "harvest_physiomotion", "harvest_sgeyesub", "harvest_shu",
           "montage_positions"]
