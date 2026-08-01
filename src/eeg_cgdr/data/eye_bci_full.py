"""Explicit-file Eye-BCI loader for one frozen natural-EEG outer fold."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.signal import resample_poly

from .eye_bci import EYE_BCI_SCALP_CHANNELS, EyeBciTarget, _infer_sampling_rate, _safe_target_path


@dataclass(frozen=True)
class EyeBciRecord:
    participant: str
    session: str
    eeg: np.ndarray
    heo: np.ndarray
    triggers: np.ndarray
    cues: np.ndarray
    blinks: np.ndarray
    sampling_rate: float
    time_units: str
    signal_units: str = "unknown_not_encoded_in_csv"


@dataclass(frozen=True)
class EyeBciNormalization:
    mean: float
    standard_deviation: float
    participants: tuple[str, ...]
    samples: int


@dataclass(frozen=True)
class EyeBciQuery:
    eeg_windows: np.ndarray
    heo_windows: np.ndarray
    trigger_windows: np.ndarray
    cue_windows: np.ndarray
    blink_windows: np.ndarray
    valid_samples: np.ndarray
    sampling_rate: int


def target_for(participant: str, session: str = "Sess01") -> EyeBciTarget:
    if len(participant) != 3 or not participant.startswith("S") or not participant[1:].isdigit():
        raise ValueError(f"invalid Eye-BCI participant: {participant}")
    number = participant[1:]
    return EyeBciTarget(
        participant,
        session,
        Path(participant) / session / "Neuroscan" / f"ME{number}1.csv",
    )


def _read_with_pandas(path: Path, *, nrows: int | None = None) -> dict[str, np.ndarray]:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - exercised by scheduled environment
        raise RuntimeError("Eye-BCI full-fold loading requires registered pandas") from exc
    columns = [
        "Time",
        *EYE_BCI_SCALP_CHANNELS,
        "HEO",
        "Trig",
        "Cues",
        "Blinks",
    ]
    numeric_columns = ["Time", *EYE_BCI_SCALP_CHANNELS, "HEO"]
    dtypes = {column: np.float32 for column in numeric_columns}
    dtypes["Time"] = np.float64
    frame = pd.read_csv(
        path,
        usecols=columns,
        dtype=dtypes,
        nrows=nrows,
        engine="c",
        memory_map=True,
        low_memory=False,
    )
    if list(frame.columns) != columns:
        frame = frame.loc[:, columns]
    # Some task files begin with UI/instruction rows (for example a Cues value
    # of "Welcome") before the amplifier stream starts.  Such rows have no
    # Time, EEG, or HEO value at all and are not EEG samples.  Drop only rows
    # where every numeric signal field is missing; partially missing samples
    # remain below and fail the strict finite-value check.
    instruction_rows = frame[numeric_columns].isna().all(axis=1)
    if bool(instruction_rows.any()):
        frame = frame.loc[~instruction_rows].reset_index(drop=True)
    output = {
        column: frame[column].to_numpy(copy=True) for column in numeric_columns
    }
    # Eye-BCI event columns are heterogeneous: some records contain numeric
    # codes while others contain labels such as "Welcome".  Signal columns
    # remain strictly typed above.  Events are encoded deterministically within
    # each record, reserving zero for blank/no-event values and preserving
    # numeric codes where present.
    for column in ("Trig", "Cues", "Blinks"):
        series = frame[column]
        text = series.fillna("").astype(str).str.strip()
        numeric = pd.to_numeric(text, errors="coerce")
        values = np.zeros(len(series), dtype=np.float32)
        numeric_mask = numeric.notna().to_numpy()
        if np.any(numeric_mask):
            values[numeric_mask] = numeric[numeric_mask].to_numpy(dtype=np.float32)
        label_mask = (~numeric_mask) & (~text.isin(("", "nan", "None"))).to_numpy()
        if np.any(label_mask):
            labels = sorted(set(text[label_mask].tolist()))
            label_codes = {label: float(100000 + index) for index, label in enumerate(labels)}
            values[label_mask] = np.asarray(
                [label_codes[label] for label in text[label_mask]], dtype=np.float32
            )
        output[column] = values
    invalid: dict[str, dict[str, int]] = {}
    for column, value in output.items():
        bad = np.flatnonzero(~np.isfinite(value))
        if value.ndim != 1 or bad.size:
            invalid[column] = {
                "count": int(bad.size),
                "first_row": int(bad[0]) if bad.size else -1,
            }
    if invalid:
        raise ValueError(
            f"Eye-BCI selected columns contain invalid values: {path.name}; "
            f"invalid_fields={invalid}"
        )
    return output


def read_eye_bci_record(
    root: Path,
    target: EyeBciTarget,
    *,
    seconds: float | None = None,
) -> EyeBciRecord:
    path = _safe_target_path(root, target)
    prefix = _read_with_pandas(path, nrows=4096)
    sampling_rate, time_units, _ = _infer_sampling_rate(prefix["Time"])
    desired_samples = None
    if seconds is not None:
        if seconds <= 0:
            raise ValueError("prefix seconds must be positive")
        desired_samples = int(np.ceil(seconds * sampling_rate))
    if desired_samples is None:
        values = _read_with_pandas(path, nrows=None)
    elif desired_samples <= prefix["Time"].size:
        values = prefix
    else:
        # Account for leading instruction rows without reading every large CSV
        # merely to obtain a short calibration prefix.  One bounded retry also
        # covers additional all-empty instruction rows later in the prefix.
        omitted_in_prefix = max(0, 4096 - int(prefix["Time"].size))
        physical_rows = desired_samples + omitted_in_prefix
        values = _read_with_pandas(path, nrows=physical_rows)
        if values["Time"].size < desired_samples:
            physical_rows += desired_samples - int(values["Time"].size) + 4096
            values = _read_with_pandas(path, nrows=physical_rows)
        if values["Time"].size < desired_samples:
            raise ValueError(
                f"Eye-BCI record has only {values['Time'].size} valid samples; "
                f"requested {desired_samples}: {path.name}"
            )
    if desired_samples is not None:
        values = {
            name: value[:desired_samples] for name, value in values.items()
        }
    eeg = np.stack([values[channel] for channel in EYE_BCI_SCALP_CHANNELS], axis=0)
    return EyeBciRecord(
        participant=target.participant_id,
        session=target.session_id,
        eeg=np.asarray(eeg, dtype=np.float32),
        heo=np.asarray(values["HEO"], dtype=np.float32),
        triggers=np.asarray(values["Trig"], dtype=np.float32),
        cues=np.asarray(values["Cues"], dtype=np.float32),
        blinks=np.asarray(values["Blinks"], dtype=np.float32),
        sampling_rate=sampling_rate,
        time_units=time_units,
    )


def fit_outer_training_normalization(
    root: Path,
    participants: Iterable[str],
    *,
    session: str,
    seconds_per_participant: float,
) -> EyeBciNormalization:
    frozen = tuple(participants)
    if not frozen or len(frozen) != len(set(frozen)):
        raise ValueError("outer-training participants must be non-empty and unique")
    total = 0.0
    total_square = 0.0
    count = 0
    for participant in frozen:
        record = read_eye_bci_record(
            root,
            target_for(participant, session),
            seconds=seconds_per_participant,
        )
        values = np.asarray(record.eeg, dtype=np.float64)
        total += float(values.sum())
        total_square += float(np.square(values).sum())
        count += values.size
    mean = total / count
    variance = max(total_square / count - mean * mean, 0.0)
    if variance <= np.finfo(np.float64).eps:
        raise ValueError("Eye-BCI outer-training EEG has no usable scale")
    return EyeBciNormalization(mean, float(np.sqrt(variance)), tuple(sorted(frozen)), count)


def _resample_continuous(array: np.ndarray, source: int, target: int) -> np.ndarray:
    divisor = int(np.gcd(source, target))
    return resample_poly(array, target // divisor, source // divisor, axis=-1)


def _resample_events(array: np.ndarray, source: int, target: int, output_samples: int) -> np.ndarray:
    native_index = np.minimum(
        np.rint(np.arange(output_samples) * source / target).astype(np.int64),
        array.size - 1,
    )
    return np.asarray(array)[native_index]


def _windows(array: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    samples = array.shape[-1]
    count = int(np.ceil(samples / length))
    output = np.zeros((count, *array.shape[:-1], length), dtype=array.dtype)
    valid = np.zeros(count, dtype=np.int64)
    for index in range(count):
        start = index * length
        stop = min(start + length, samples)
        valid[index] = stop - start
        output[index, ..., : valid[index]] = array[..., start:stop]
    return output, valid


def prepare_eye_bci_query(
    record: EyeBciRecord,
    *,
    query_start_seconds: float,
    target_sampling_rate: int,
    window_samples: int,
    normalization: EyeBciNormalization,
) -> EyeBciQuery:
    source_rate = int(round(record.sampling_rate))
    start = int(round(query_start_seconds * source_rate))
    if start >= record.eeg.shape[1]:
        raise ValueError("Eye-BCI query starts after record end")
    eeg = _resample_continuous(record.eeg[:, start:], source_rate, target_sampling_rate)
    heo = _resample_continuous(record.heo[None, start:], source_rate, target_sampling_rate)
    eeg = ((eeg - normalization.mean) / normalization.standard_deviation).astype(np.float32)
    heo = heo.astype(np.float32)
    output_samples = eeg.shape[1]
    triggers = _resample_events(record.triggers[start:], source_rate, target_sampling_rate, output_samples)
    cues = _resample_events(record.cues[start:], source_rate, target_sampling_rate, output_samples)
    blinks = _resample_events(record.blinks[start:], source_rate, target_sampling_rate, output_samples)
    eeg_windows, valid = _windows(eeg, window_samples)
    heo_windows, heo_valid = _windows(heo, window_samples)
    trigger_windows, trigger_valid = _windows(triggers[None], window_samples)
    cue_windows, cue_valid = _windows(cues[None], window_samples)
    blink_windows, blink_valid = _windows(blinks[None], window_samples)
    for other in (heo_valid, trigger_valid, cue_valid, blink_valid):
        if not np.array_equal(valid, other):
            raise AssertionError("Eye-BCI streams lost temporal alignment")
    return EyeBciQuery(
        eeg_windows=eeg_windows,
        heo_windows=heo_windows,
        trigger_windows=trigger_windows[:, 0],
        cue_windows=cue_windows[:, 0],
        blink_windows=blink_windows[:, 0],
        valid_samples=valid,
        sampling_rate=target_sampling_rate,
    )


def write_eye_bci_split_manifest(
    path: Path,
    *,
    config: dict[str, Any],
    sampling_rate: float,
) -> None:
    eye = config["eye_bci"]
    train = set(eye["training_participants"])
    validation = set(eye["validation_participants"])
    test = set(eye["test_participants"])
    if train & validation or train & test or validation & test:
        raise ValueError("Eye-BCI participant outer splits overlap")
    fields = [
        "dataset_version",
        "outer_fold",
        "split",
        "participant",
        "session",
        "record",
        "calibration_start",
        "calibration_end",
        "query_start",
        "query_end",
        "sampling_rate",
        "status",
    ]
    rows: list[dict[str, Any]] = []
    for split_name, participants in (
        ("train", sorted(train)),
        ("validation", sorted(validation)),
    ):
        for participant in participants:
            target = target_for(participant, eye["session"])
            rows.append(
                {
                    "dataset_version": "eye_bci_syn64005218_neuroscan",
                    "outer_fold": eye["outer_fold"],
                    "split": split_name,
                    "participant": participant,
                    "session": eye["session"],
                    "record": str(target.relative_path),
                    "calibration_start": "",
                    "calibration_end": "",
                    "query_start": "",
                    "query_end": "",
                    "sampling_rate": sampling_rate,
                    "status": "population_source",
                }
            )
    for participant in sorted(test):
        target = target_for(participant, eye["session"])
        base = {
            "dataset_version": "eye_bci_syn64005218_neuroscan",
            "outer_fold": eye["outer_fold"],
            "split": "test",
            "participant": participant,
            "session": eye["session"],
            "record": str(target.relative_path),
            "sampling_rate": sampling_rate,
        }
        rows.append(
            {
                **base,
                "calibration_start": eye["calibration_start_seconds"],
                "calibration_end": eye["calibration_end_seconds"],
                "query_start": "",
                "query_end": "",
                "status": "held_out_calibration",
            }
        )
        rows.append(
            {
                **base,
                "calibration_start": "",
                "calibration_end": "",
                "query_start": eye["query_start_seconds"],
                "query_end": "EOF",
                "status": "held_out_query",
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
