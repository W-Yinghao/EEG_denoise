"""Frozen OpenBMI/Lee2019-MI data contract for V36P.

The loader consumes the existing project datalake, never downloads data, and
returns exactly one 0--4 s, 200 Hz, 62-channel EEG trial per registered event.
Filtering/resampling and EEG-only channel selection were performed by the
audited datalake producer; this module only epochs and per-trial standardizes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


OPENBMI_ROOT = Path("/projects/EEG-foundation-model/datalake/processed/4704743c/Lee2019_MI")
N_SUBJECTS = 54
N_CHANNELS = 62
N_SAMPLES = 800
SAMPLING_RATE = 200


@dataclass(frozen=True)
class OpenBMITrials:
    eeg: np.ndarray
    task: np.ndarray
    subject: np.ndarray
    session: np.ndarray
    trial: np.ndarray


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def outer_folds() -> list[dict[str, object]]:
    """Six fixed outer folds; each participant appears in test exactly once."""
    groups = [list(range(start, start + 9)) for start in range(0, N_SUBJECTS, 9)]
    result = []
    for fold, test in enumerate(groups):
        validation = groups[(fold + 1) % len(groups)]
        train = sorted(set(range(N_SUBJECTS)) - set(test) - set(validation))
        result.append({
            "fold": fold,
            "test_subjects": test,
            "validation_subjects": validation,
            "train_subjects": train,
            "model_train_session": "ses_0",
            "model_validation_session": "ses_1 (participant-disjoint)",
            "final_refit_session": "ses_0",
            "privacy_gallery_session": "ses_0",
            "privacy_query_session": "ses_1",
        })
    return result


def validate_folds(folds: list[dict[str, object]] | None = None) -> None:
    folds = outer_folds() if folds is None else folds
    observed: list[int] = []
    for item in folds:
        train = set(item["train_subjects"])
        validation = set(item["validation_subjects"])
        test = set(item["test_subjects"])
        if train & validation or train & test or validation & test:
            raise ValueError(f"fold {item['fold']} participant leakage")
        if len(train) != 36 or len(validation) != 9 or len(test) != 9:
            raise ValueError(f"fold {item['fold']} has invalid 36/9/9 counts")
        observed.extend(sorted(test))
    if sorted(observed) != list(range(N_SUBJECTS)):
        raise ValueError("all 54 participants must occur exactly once in outer test")


def _tables(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_parquet(root / "metadata.parquet")
    events = pd.read_parquet(root / "events.parquet")
    return metadata, events


def load_openbmi(root: Path, subjects: list[int], session: str) -> OpenBMITrials:
    if session not in {"ses_0", "ses_1"}:
        raise ValueError("session must be ses_0 or ses_1")
    metadata, events = _tables(root)
    rows: list[np.ndarray] = []
    tasks: list[int] = []
    owners: list[int] = []
    sessions: list[int] = []
    trials: list[int] = []
    for subject in subjects:
        recording = metadata[(metadata.subject == subject) & (metadata.session == session)]
        if len(recording) != 1:
            raise ValueError(f"subject={subject} session={session}: expected one recording")
        record = recording.iloc[0]
        continuous = np.load(root / str(record.filepath), mmap_mode="r")
        if continuous.ndim != 2 or continuous.shape[1] != N_CHANNELS:
            raise ValueError(f"recording {record.recording_id}: unexpected shape {continuous.shape}")
        registered = events[events.recording_id == record.recording_id].sort_values("onset_sample")
        if len(registered) != 100:
            raise ValueError(f"recording {record.recording_id}: expected 100 trials, got {len(registered)}")
        for local_trial, event in enumerate(registered.itertuples(index=False)):
            start = int(event.onset_sample)
            stop = start + N_SAMPLES
            if start < 0 or stop > len(continuous):
                raise ValueError(f"recording {record.recording_id}: out-of-bounds trial {local_trial}")
            epoch = np.asarray(continuous[start:stop], dtype=np.float32).T
            mean = epoch.mean(axis=1, keepdims=True)
            std = epoch.std(axis=1, keepdims=True)
            epoch = (epoch - mean) / np.maximum(std, 1e-6)
            rows.append(epoch.astype(np.float32, copy=False))
            if int(event.event_code) not in (1, 2):
                raise ValueError(f"unexpected event code {event.event_code}")
            tasks.append(int(event.event_code) - 1)
            owners.append(subject)
            sessions.append(0 if session == "ses_0" else 1)
            trials.append(local_trial)
    return OpenBMITrials(
        eeg=np.stack(rows),
        task=np.asarray(tasks, dtype=np.int64),
        subject=np.asarray(owners, dtype=np.int64),
        session=np.asarray(sessions, dtype=np.int64),
        trial=np.asarray(trials, dtype=np.int64),
    )


def concatenate(parts: list[OpenBMITrials]) -> OpenBMITrials:
    return OpenBMITrials(**{
        name: np.concatenate([getattr(part, name) for part in parts])
        for name in OpenBMITrials.__dataclass_fields__
    })


def build_dataset_inventory(root: Path = OPENBMI_ROOT) -> tuple[list[dict[str, object]], dict[str, object]]:
    metadata, events = _tables(root)
    infos = json.loads((root / "infos.json").read_text(encoding="utf-8"))
    if len(metadata) != 108 or metadata.subject.nunique() != 54 or metadata.session.nunique() != 2:
        raise ValueError("OpenBMI cache is incomplete")
    if len(events) != 10800:
        raise ValueError("OpenBMI event table is incomplete")
    channels = json.loads(str(metadata.iloc[0].channels))
    rows = []
    for record in metadata.sort_values(["subject", "session"]).itertuples(index=False):
        registered = events[events.recording_id == record.recording_id]
        rows.append({
            "dataset": "MOABB Lee2019_MI (OpenBMI)",
            "processed_cache_id": "4704743c",
            "subject": int(record.subject),
            "original_subject": int(infos["original_subjects"][f"sub_{int(record.subject)}"]),
            "session": str(record.session),
            "original_session": int(infos["original_sessions"][str(record.session)]),
            "run": str(record.run),
            "recording_id": int(record.recording_id),
            "channels": int(record.n_channels),
            "samples": int(record.n_timepoints),
            "trials": int(len(registered)),
            "left_trials": int((registered.event_code == 2).sum()),
            "right_trials": int((registered.event_code == 1).sum()),
            "filepath": str((root / record.filepath).resolve()),
        })
    manifest = {
        "dataset": "OpenBMI / Lee2019_MI",
        "source": "existing project datalake produced through MOABB Lee2019_MI",
        "moabb_version_audited": "1.5.0",
        "processed_root": str(root.resolve()),
        "processed_cache_id": "4704743c",
        "metadata_sha256": sha256(root / "metadata.parquet"),
        "events_sha256": sha256(root / "events.parquet"),
        "infos_sha256": sha256(root / "infos.json"),
        "participants": 54,
        "sessions": ["ses_0", "ses_1"],
        "recordings": 108,
        "trials": 10800,
        "channels": channels,
        "sampling_rate_hz": SAMPLING_RATE,
        "producer_preprocessing": infos["preprocessing"],
        "event_id": infos["event_id"],
        "trial_interval_seconds": infos["interval"],
        "active_epoch_contract": "0--4 s from event onset, 800 samples, per-trial per-channel z-score",
        "license_provenance": "Lee et al. 2019 OpenBMI dataset; local MOABB cache and producer metadata",
        "download_performed": False,
        "waveform_sealed_reads": 0,
    }
    validate_folds()
    return rows, manifest


__all__ = [
    "N_CHANNELS", "N_SAMPLES", "N_SUBJECTS", "OPENBMI_ROOT", "OpenBMITrials",
    "build_dataset_inventory", "concatenate", "load_openbmi", "outer_folds", "validate_folds",
]
