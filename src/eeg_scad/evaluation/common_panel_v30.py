"""Immutable common-panel and support-bank helpers for V30."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from eeg_scad.data.counterfactual_pairs import _load_signal, fold_eeg_scale
from eeg_scad.data.v24_coordinate_contract import robust_center_scale


SESSIONS = ("ses-02", "ses-03", "ses-04")
TASKS = ("ERP", "SSVEP")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_digest(values: Iterable[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = np.asarray(value)
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    value = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(value[:8], "little")


def select_balanced_indices(rows: list[dict[str, str]], per_cell: int) -> list[int]:
    """Choose a deterministic participant/session/task panel without outcomes."""
    grouped: dict[tuple[str, str, str], list[int]] = {}
    for index, row in enumerate(rows):
        grouped.setdefault((row["participant"], row["session"], row["task"]), []).append(index)
    chosen: list[int] = []
    for key in sorted(grouped):
        if len(grouped[key]) < per_cell:
            raise RuntimeError(f"common-panel cell too short: {key}")
        chosen.extend(grouped[key][:per_cell])
    return sorted(chosen)


def read_role_rows(role_manifest: Path, fold: int, stream: str) -> list[dict[str, str]]:
    return [
        row for row in csv.DictReader(role_manifest.open())
        if int(row["fold"]) == fold and row["split"] == "test" and row["stream"] == stream
    ]


def support_starts(duration_seconds: int, window_samples: int = 200, rate: int = 100) -> list[int]:
    if duration_seconds <= 0:
        return []
    stop = duration_seconds * rate
    if stop < window_samples:
        raise ValueError("support duration shorter than one window")
    count = 16 if duration_seconds >= 120 else max(1, int(np.ceil(duration_seconds / 2)))
    return np.rint(np.linspace(0, stop - window_samples, count)).astype(int).tolist()


def support_episode(
    root: Path,
    owner: str,
    session: str,
    task: str,
    eeg_scale: np.ndarray,
    duration_seconds: int = 120,
) -> tuple[np.ndarray, np.ndarray, list[int], str]:
    try:
        eeg, eog = _load_signal(root, owner, session, task)
        actual_task = task
    except FileNotFoundError:
        actual_task = next(value for value in TASKS if value != task)
        eeg, eog = _load_signal(root, owner, session, actual_task)
    starts = support_starts(duration_seconds)
    center, scale = robust_center_scale(eog[:, :12000])
    support_eeg = np.stack([eeg[:, start:start + 200] / eeg_scale[:, None] for start in starts])
    support_eog = np.stack([(eog[:, start:start + 200] - center[:, None]) / scale[:, None] for start in starts])
    return support_eeg.astype(np.float32), support_eog.astype(np.float32), starts, actual_task


def build_support_bank(
    data: Mapping[str, Any],
    fold: Mapping[str, Any],
    owners: list[str],
    destination: Path,
) -> list[dict[str, Any]]:
    root = Path(data["v19_derived_root"])
    eeg_scale = fold_eeg_scale(data, list(fold["train"]))
    arrays: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for duration in (5, 10, 30, 120):
        eeg_values, eog_values = [], []
        for owner in owners:
            for session in SESSIONS:
                for task in TASKS:
                    eeg, eog, starts, actual = support_episode(root, owner, session, task, eeg_scale, duration)
                    eeg_values.append(eeg); eog_values.append(eog)
                    rows.append({
                        "fold": fold["fold"], "owner": owner, "session": session, "task": task,
                        "actual_task": actual, "duration_seconds": duration,
                        "windows": len(starts), "starts": ";".join(map(str, starts)),
                        "digest": content_digest((eeg, eog, np.asarray(starts, dtype=np.int32))),
                    })
        arrays[f"eeg_{duration}"] = np.asarray(eeg_values, dtype=np.float32)
        arrays[f"eog_{duration}"] = np.asarray(eog_values, dtype=np.float32)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **arrays)
    return rows


def support_bank_index(owner: str, session: str, task: str, owners: list[str]) -> int:
    return owners.index(owner) * len(SESSIONS) * len(TASKS) + SESSIONS.index(session) * len(TASKS) + TASKS.index(task)


def attach_support(
    batch: dict[str, Any],
    bank: Mapping[str, np.ndarray],
    owners: list[str],
    duration: int = 120,
    donor: str | None = None,
    wrong: bool = False,
) -> dict[str, Any]:
    eeg, eog = [], []
    for meta in batch["meta"]:
        owner = donor or (meta["wrong_owner"] if wrong else meta["participant"])
        index = support_bank_index(owner, meta["session"], meta["task"], owners)
        eeg.append(bank[f"eeg_{duration}"][index]); eog.append(bank[f"eog_{duration}"][index])
    result = dict(batch)
    key = "wrong_" if wrong else ""
    result[key + "support_eeg"] = np.asarray(eeg)
    result[key + "support_eog"] = np.asarray(eog)
    return result


def load_panel(
    v24_root: Path,
    role_manifest: Path,
    fold: int,
    stream: str,
    indices: list[int],
    evaluator: bool = False,
) -> dict[str, Any]:
    source = v24_root / f"fold_{fold}" / f"{stream}_test_{'evaluator' if evaluator else 'inference'}.npz"
    with np.load(source, allow_pickle=False) as archive:
        values = {key: np.asarray(archive[key])[indices] for key in archive.files}
    rows = read_role_rows(role_manifest, fold, stream)
    return {**values, "meta": [rows[index] for index in indices], "stream": stream}


def save_panel_index(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")


__all__ = [
    "SESSIONS", "TASKS", "attach_support", "build_support_bank", "content_digest",
    "load_panel", "read_role_rows", "save_panel_index", "select_balanced_indices", "sha256",
    "stable_seed", "support_bank_index", "support_episode", "support_starts",
]
