"""Fail-closed reader for the existing SHU multi-session MI datalake copy.

The available datalake asset is a 256 Hz derivative of the nominal 250 Hz
processed-trial release.  Trials are deterministically resampled to the frozen
250 Hz protocol on load.  Sessions 04/05 are session-sealed until an explicit
final-evaluation capability is supplied.
"""
from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import lmdb
import numpy as np
from scipy.signal import resample_poly


DEFAULT_LMDB = Path("/projects/EEG-foundation-model/tdoan-24/SHUMI_256hz")
KEY_RE = re.compile(
    r"^sub-(?P<subject>\d{3})_ses-(?P<session>\d{2})_task_motorimagery_eeg-(?P<trial>\d+)$"
)


@dataclass(frozen=True, order=True)
class ShuTrialKey:
    subject: int
    session: int
    trial: int

    @property
    def lmdb_key(self) -> bytes:
        return (
            f"sub-{self.subject:03d}_ses-{self.session:02d}_"
            f"task_motorimagery_eeg-{self.trial}"
        ).encode()


@dataclass(frozen=True)
class ShuTrial:
    key: ShuTrialKey
    eeg_uv: np.ndarray
    label: int
    source_sfreq: float = 256.0
    protocol_sfreq: float = 250.0


class ShuMiStore:
    """Read SHU trials while enforcing the Day-4/5 session seal."""

    def __init__(self, path: Path = DEFAULT_LMDB) -> None:
        self.path = Path(path)
        if not (self.path / "data.mdb").is_file():
            raise FileNotFoundError(f"SHU datalake LMDB is missing: {self.path}")
        self._env = lmdb.open(
            str(self.path), subdir=True, readonly=True, lock=False,
            readahead=False, max_dbs=32,
        )

    def inventory(self) -> list[ShuTrialKey]:
        """List keys only; this does not deserialize any trial payload."""
        output: list[ShuTrialKey] = []
        with self._env.begin() as txn:
            for raw_key, _ in txn.cursor():
                key = raw_key.decode("utf-8")
                if key == "__keys__":
                    continue
                match = KEY_RE.fullmatch(key)
                if match is None:
                    raise ValueError(f"unexpected SHU key: {key}")
                output.append(
                    ShuTrialKey(
                        int(match["subject"]), int(match["session"]), int(match["trial"])
                    )
                )
        return output

    @staticmethod
    def _check_access(key: ShuTrialKey, *, final_evaluation: bool) -> None:
        if key.session >= 4 and not final_evaluation:
            raise PermissionError(
                f"SHU session {key.session:02d} is sealed; payload access refused"
            )

    def load(self, key: ShuTrialKey, *, final_evaluation: bool = False) -> ShuTrial:
        self._check_access(key, final_evaluation=final_evaluation)
        with self._env.begin() as txn:
            raw = txn.get(key.lmdb_key)
        if raw is None:
            raise KeyError(key)
        record = pickle.loads(raw)
        source = np.asarray(record["sample"], dtype=np.float32)
        if source.shape != (32, 1024) or not np.isfinite(source).all():
            raise ValueError((key, source.shape, bool(np.isfinite(source).all())))
        # 4 seconds: 1024 at 256 Hz -> 1000 at the frozen nominal 250 Hz.
        eeg = resample_poly(source, 125, 128, axis=-1).astype(np.float32)
        if eeg.shape != (32, 1000):
            raise RuntimeError((key, eeg.shape))
        label = int(record["label"])
        if label not in (0, 1, 2):
            raise ValueError((key, label))
        return ShuTrial(key=key, eeg_uv=eeg, label=label)

    def iter_trials(
        self, subject: int, sessions: tuple[int, ...], *, final_evaluation: bool = False
    ) -> Iterator[ShuTrial]:
        for key in self.inventory():
            if key.subject == subject and key.session in sessions:
                yield self.load(key, final_evaluation=final_evaluation)


__all__ = ["DEFAULT_LMDB", "ShuMiStore", "ShuTrial", "ShuTrialKey"]
