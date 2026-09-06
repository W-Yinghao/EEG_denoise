"""Leakage-safe MobileBCI metadata split and development-only loader.

The metadata stage parses text headers/channel tables only.  Signal and event
loading requires an explicit development allowlist; sealed participants are
rejected before any binary or marker path is opened.
"""

from __future__ import annotations

import csv
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PARTICIPANT_RE = re.compile(r"sub-(\d{2})")
HEADER_RE = re.compile(r"(sub-\d{2})_(ses-\d{2})_task-(ERP|SSVEP)_eeg\.vhdr$")
PRIMARY_SESSIONS = ("ses-02", "ses-03", "ses-04")
TASKS = ("ERP", "SSVEP")


class SealedParticipantAccessError(PermissionError):
    """Raised before opening any signal/outcome belonging to a sealed unit."""


@dataclass(frozen=True)
class MobileSplit:
    development: tuple[str, ...]
    sealed: tuple[str, ...]
    folds: Mapping[int, tuple[str, ...]]


def parse_brainvision_header(path: Path) -> dict[str, Any]:
    """Parse only harmless BrainVision text pointers and sampling interval."""
    values: dict[str, str] = {}
    for line in path.read_text(encoding="latin-1").splitlines():
        if "=" in line and not line.lstrip().startswith(";"):
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    interval = float(values["SamplingInterval"])
    return {
        "data_file": values.get("DataFile", ""),
        "marker_file": values.get("MarkerFile", ""),
        "sampling_rate_hz": 1_000_000.0 / interval,
    }


def channel_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            kind = str(row.get("type", "unknown")).upper()
            counts[kind] = counts.get(kind, 0) + 1
    return counts


def metadata_inventory(root: Path) -> list[dict[str, Any]]:
    """Inventory text metadata without reading binary, markers, or events."""
    rows: list[dict[str, Any]] = []
    for header in sorted(root.glob("sub-[0-9][0-9]/ses-[0-9][0-9]/eeg/*_eeg.vhdr")):
        match = HEADER_RE.match(header.name)
        if match is None:
            continue
        participant, session, task = match.groups()
        channels = header.with_name(header.name.replace("_eeg.vhdr", "_channels.tsv"))
        parsed = parse_brainvision_header(header)
        data_path = header.parent / str(parsed["data_file"])
        marker_path = header.parent / str(parsed["marker_file"])
        counts = channel_counts(channels) if channels.is_file() else {}
        source_channels = (
            root / "sourcedata" / participant / session / "eeg"
            / f"{participant}_{session}_task-{task}_channels.tsv"
        )
        source_counts = channel_counts(source_channels) if source_channels.is_file() else {}
        rows.append({
            "participant": participant,
            "session": session,
            "task": task,
            "header": str(header),
            "header_exists": header.is_file(),
            "data_file": str(data_path),
            "data_exists": data_path.is_file(),
            "marker_file": str(marker_path),
            "marker_exists": marker_path.is_file(),
            "channels_tsv": str(channels),
            "channels_exists": channels.is_file(),
            "sampling_rate_hz": parsed["sampling_rate_hz"],
            "processed_eeg_channels": counts.get("EEG", 0),
            "processed_eog_channels": counts.get("EOG", 0),
            "processed_imu_channels": counts.get("IMU", 0),
            "source_eeg_channels": source_counts.get("EEG", 0),
            "source_eog_channels": source_counts.get("EOG", 0),
            "binary_signal_opened": False,
            "marker_content_opened": False,
            "outcome_opened": False,
        })
    return rows


def freeze_metadata_split(rows: Sequence[Mapping[str, Any]], *, seed: int = 20260806) -> MobileSplit:
    participants = tuple(f"sub-{index:02d}" for index in range(1, 25))
    by_participant: dict[str, list[Mapping[str, Any]]] = {participant: [] for participant in participants}
    for row in rows:
        by_participant[str(row["participant"])].append(row)

    def facts(participant: str) -> tuple[int, int, int, int]:
        records = by_participant[participant]
        keys = {(str(row["session"]), str(row["task"])) for row in records
                if bool(row["header_exists"]) and bool(row["data_exists"])
                and bool(row["marker_exists"]) and bool(row["channels_exists"])}
        primary = sum((session, task) in keys for session in PRIMARY_SESSIONS for task in TASKS)
        secondary = sum(("ses-05", task) in keys for task in TASKS)
        eog_metadata = sum(int(row["source_eog_channels"]) >= 4 for row in records)
        imu_metadata = sum(int(row["processed_imu_channels"]) > 0 for row in records)
        return primary, secondary, eog_metadata, imu_metadata

    rng = random.Random(seed)
    sealed: list[str] = []
    # Exactly one sealed participant per consecutive group.  The member whose
    # metadata completeness is closest to its group median is selected; seeded
    # random order resolves exact ties without inspecting signal outcomes.
    for start in range(1, 25, 3):
        group = [f"sub-{index:02d}" for index in range(start, start + 3)]
        shuffled = list(group); rng.shuffle(shuffled)
        group_facts = {participant: facts(participant) for participant in group}
        medians = tuple(sorted(group_facts[p][axis] for p in group)[1] for axis in range(4))
        sealed.append(min(shuffled, key=lambda p: sum(abs(group_facts[p][axis] - medians[axis]) for axis in range(4))))
    development = [participant for participant in participants if participant not in set(sealed)]
    shuffled_development = list(development); rng.shuffle(shuffled_development)
    folds = {fold: tuple(sorted(shuffled_development[fold::4])) for fold in range(4)}
    if len(development) != 16 or len(sealed) != 8 or any(len(value) != 4 for value in folds.values()):
        raise AssertionError("MobileBCI metadata split cardinality is invalid")
    return MobileSplit(tuple(sorted(development)), tuple(sorted(sealed)), folds)


def assert_development_access(participant: str, allowlist: Iterable[str]) -> None:
    allowed = frozenset(str(value) for value in allowlist)
    if participant not in allowed:
        raise SealedParticipantAccessError(
            f"signal/outcome access denied before open for non-development participant {participant}"
        )


def development_record_paths(
    root: Path, participant: str, session: str, task: str, *, allowlist: Iterable[str]
) -> tuple[Path, Path, Path]:
    assert_development_access(participant, allowlist)
    stem = f"{participant}_{session}_task-{task}_eeg"
    directory = root / participant / session / "eeg"
    return directory / f"{stem}.vhdr", directory / f"{stem}.vmrk", directory / f"{stem}.eeg"


def read_development_record(
    root: Path,
    participant: str,
    session: str,
    task: str,
    *,
    allowlist: Iterable[str],
) -> dict[str, Any]:
    """Read one processed EEG+IMU record after allowlist enforcement."""
    assert_development_access(participant, allowlist)
    import mne

    header, _, _ = development_record_paths(
        root, participant, session, task, allowlist=allowlist
    )
    if not header.is_file():
        raise FileNotFoundError(header)
    channels_path = header.with_name(header.name.replace("_eeg.vhdr", "_channels.tsv"))
    with channels_path.open(encoding="utf-8", newline="") as stream:
        channel_rows = [dict(row) for row in csv.DictReader(stream, delimiter="\t")]
    raw = mne.io.read_raw_brainvision(header, preload=True, verbose="ERROR")
    names = tuple(raw.ch_names)
    if len(channel_rows) != len(names):
        raise ValueError(f"channel table/raw count mismatch for {header}")
    eeg_indices = [index for index, row in enumerate(channel_rows) if str(row.get("type", "")).upper() == "EEG"]
    imu_indices = [index for index, row in enumerate(channel_rows) if str(row.get("type", "")).upper() == "IMU"]
    data = raw.get_data()
    return {
        "eeg": np.asarray(data[eeg_indices], dtype=np.float64) * 1e6,
        "imu": np.asarray(data[imu_indices], dtype=np.float64),
        "eeg_names": tuple(names[index] for index in eeg_indices),
        "imu_names": tuple(names[index] for index in imu_indices),
        "sampling_rate_hz": float(raw.info["sfreq"]),
        "duration_seconds": float(data.shape[1] / raw.info["sfreq"]),
        "header": str(header),
    }


def read_source_eeg_eog(
    root: Path, participant: str, session: str, task: str, *, allowlist: Iterable[str]
) -> dict[str, Any]:
    assert_development_access(participant, allowlist)
    import mne

    directory = root / "sourcedata" / participant / session / "eeg"
    header = directory / f"{participant}_{session}_task-{task}_eeg.vhdr"
    channels_path = directory / f"{participant}_{session}_task-{task}_channels.tsv"
    with channels_path.open(encoding="utf-8", newline="") as stream:
        channel_rows = [dict(row) for row in csv.DictReader(stream, delimiter="\t")]
    raw = mne.io.read_raw_brainvision(header, preload=True, verbose="ERROR")
    names = tuple(raw.ch_names); data = raw.get_data()
    eeg_indices = [index for index, row in enumerate(channel_rows) if str(row.get("type", "")).upper() == "EEG"]
    eog_indices = [index for index, row in enumerate(channel_rows) if str(row.get("type", "")).upper() == "EOG"]
    return {
        "eeg": np.asarray(data[eeg_indices], dtype=np.float64) * 1e6,
        "eog": np.asarray(data[eog_indices], dtype=np.float64) * 1e6,
        "eeg_names": tuple(names[index] for index in eeg_indices),
        "eog_names": tuple(names[index] for index in eog_indices),
        "sampling_rate_hz": float(raw.info["sfreq"]), "header": str(header),
    }


def read_events(
    root: Path, participant: str, session: str, task: str, *, allowlist: Iterable[str]
) -> list[dict[str, str]]:
    assert_development_access(participant, allowlist)
    path = root / participant / session / "eeg" / f"{participant}_{session}_task-{task}_events.tsv"
    with path.open(encoding="utf-8", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream, delimiter="\t")]
