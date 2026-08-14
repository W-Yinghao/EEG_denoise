"""U0-c cross-corpus operator & covariance atlas.

Operator rows (between-/within-subject variances = the τ² inputs for the ρ and
λ EB rules) come from the three panel builders.  SHU (sessions 1-3 only) and
PhysioMotion (development subjects only) contribute covariance-only rows.
Sealed subsets are never opened: SHU sessions 4/5 and the PhysioMotion sealed
ten are excluded by construction below.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


SHU_ROOT = Path("/projects/EEG-foundation-model/SHU-MI-cbramod/mat")
SHU_DEVELOPMENT_SESSIONS = (1, 2, 3)          # sessions 4/5 sealed
PHYSIOMOTION_ROOT = Path("/projects/EEG-foundation-model/PhysioMotion_Artifact")
PHYSIOMOTION_SEALED = {3, 4, 7, 12, 15, 17, 19, 24, 25, 28}


def operator_rows(cells) -> list[dict]:
    rows = []
    by_group: dict[str, list[np.ndarray]] = {}
    for cell in cells:
        by_group.setdefault(cell.panel, []).append(cell.a_support)
    pop = {panel: np.mean(np.stack(values), axis=0) for panel, values in by_group.items()}
    for cell in cells:
        within = float(np.mean(np.square(cell.a_halves[0] - cell.a_halves[1]))) / 2.0
        between = float(np.mean(np.square(cell.a_support - pop[cell.panel])))
        severity = [episode["severity"] for episode in cell.episodes]
        rows.append({
            "dataset": cell.panel, "subject": cell.subject, "record": cell.cell,
            "channels": cell.a_support.shape[0], "regressors": cell.a_support.shape[1],
            "operator_between_sq": between, "operator_within_sq": within,
            "operator_norm": float(np.linalg.norm(cell.a_support)),
            "episodes": len(cell.episodes),
            "severity_median": float(np.median(severity)) if severity else np.nan,
            "severity_p90": float(np.quantile(severity, .9)) if severity else np.nan,
            "kind": "operator+covariance",
        })
    return rows


def shu_covariance_rows() -> list[dict]:
    import scipy.io as sio

    rows = []
    for subject in range(1, 26):
        stacks = []
        for session in SHU_DEVELOPMENT_SESSIONS:
            path = SHU_ROOT / f"sub-{subject:03d}_ses-{session:02d}_task_motorimagery_eeg.mat"
            if not path.is_file():
                continue
            data = np.asarray(sio.loadmat(path)["data"], np.float64)   # trials x 32 x 1000
            stacks.append(data.transpose(1, 0, 2).reshape(data.shape[1], -1))
        if not stacks:
            rows.append({"dataset": "shu", "subject": f"sub-{subject:03d}",
                         "kind": "covariance_only", "status": "no_development_sessions"})
            continue
        joined = np.concatenate(stacks, axis=1)
        covariance = np.cov(joined)
        rows.append({"dataset": "shu", "subject": f"sub-{subject:03d}", "record": "ses1-3",
                     "channels": joined.shape[0], "kind": "covariance_only", "status": "ok",
                     "samples": joined.shape[1],
                     "cov_trace": float(np.trace(covariance)),
                     "cov_logdet": float(np.linalg.slogdet(covariance + 1e-9 * np.eye(len(covariance)))[1]),
                     "cov_condition": float(np.linalg.cond(covariance + 1e-9 * np.eye(len(covariance))))})
    return rows


def physiomotion_covariance_rows() -> list[dict]:
    import mne

    rows = []
    for subject_dir in sorted(PHYSIOMOTION_ROOT.glob("sub-*")):
        try:
            number = int(subject_dir.name.split("-")[1])
        except ValueError:
            continue
        if number in PHYSIOMOTION_SEALED:
            continue  # sealed cohort: never read
        runs = sorted((subject_dir / "eeg").glob("*_eeg.edf"))
        if not runs:
            rows.append({"dataset": "physiomotion", "subject": subject_dir.name,
                         "kind": "covariance_only", "status": "no_edf"})
            continue
        try:
            raw = mne.io.read_raw_edf(runs[0], preload=True, verbose="error")
            picks = mne.pick_types(raw.info, eeg=True)
            data = raw.get_data(picks=picks)
            covariance = np.cov(data)
            rows.append({"dataset": "physiomotion", "subject": subject_dir.name,
                         "record": runs[0].stem, "channels": data.shape[0],
                         "kind": "covariance_only", "status": "ok", "samples": data.shape[1],
                         "cov_trace": float(np.trace(covariance)),
                         "cov_logdet": float(np.linalg.slogdet(covariance + 1e-18 * np.eye(len(covariance)))[1]),
                         "cov_condition": float(np.linalg.cond(covariance + 1e-18 * np.eye(len(covariance))))})
        except Exception as error:  # reason-coded, fail loud in the report
            rows.append({"dataset": "physiomotion", "subject": subject_dir.name,
                         "kind": "covariance_only", "status": f"read_error:{type(error).__name__}"})
    return rows


def sgeyesub_layout_rows(repo_root: Path) -> list[dict]:
    import json

    from eeg_chart.positions import sgeyesub_positions
    from eeg_chart.transport import sh_lift

    contracts = json.loads((repo_root / "results/cgdr/sgeyesub_operator_specificity/metadata/"
                            "layout_contracts.json").read_text())
    entries = contracts.values() if isinstance(contracts, dict) else contracts
    rows = []
    for entry in entries:
        labels = entry["ordered_channel_labels"]
        positions, kept, missing = sgeyesub_positions(labels)
        lift = sh_lift(positions)
        rows.append({"dataset": "sgeyesub", "record": entry.get("layout_id"),
                     "channels": len(labels), "resolved": len(kept), "unresolved": len(missing),
                     "kind": "montage_stress", "lift_condition": float(np.linalg.cond(lift)),
                     "lift_roundtrip_error": float(np.max(np.abs(np.linalg.pinv(lift) @ lift
                                                                 - np.eye(len(kept)))))})
    return rows


def write_atlas(cells, repo_root: Path, target: Path) -> pd.DataFrame:
    target.mkdir(parents=True, exist_ok=True)
    frames = {
        "panel_operators": pd.DataFrame(operator_rows(cells)),
        "shu": pd.DataFrame(shu_covariance_rows()),
        "physiomotion": pd.DataFrame(physiomotion_covariance_rows()),
        "sgeyesub_layouts": pd.DataFrame(sgeyesub_layout_rows(repo_root)),
    }
    for name, frame in frames.items():
        frame.to_csv(target / f"{name}.csv", index=False)
    return frames["panel_operators"]


__all__ = ["operator_rows", "physiomotion_covariance_rows", "sgeyesub_layout_rows",
           "shu_covariance_rows", "write_atlas"]
