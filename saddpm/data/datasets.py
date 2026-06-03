"""Torch datasets over preprocessed EEG windows (handoff §6, §8)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .bcic2a import SubjectWindows
from .cache import load_subject_cached
from .config import DataConfig


class WindowDataset(Dataset):
    """Dataset of ``(window, subject_id, mi_label)`` triples.

    Args:
        windows: ``(N, 22, 512)`` float32 array.
        subject_ids: ``(N,)`` 0-based subject ids (for embeddings / ArcFace).
        mi_labels: ``(N,)`` 4-class motor-imagery labels (for downstream eval).
    """

    def __init__(self, windows: np.ndarray, subject_ids: np.ndarray, mi_labels: np.ndarray) -> None:
        assert len(windows) == len(subject_ids) == len(mi_labels)
        self.windows = torch.from_numpy(np.ascontiguousarray(windows)).float()
        self.subject_ids = torch.from_numpy(np.ascontiguousarray(subject_ids)).long()
        self.mi_labels = torch.from_numpy(np.ascontiguousarray(mi_labels)).long()

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.windows[idx], self.subject_ids[idx], self.mi_labels[idx]


@dataclass
class PooledWindows:
    """Concatenated windows across subjects for one session role."""

    windows: np.ndarray
    subject_ids: np.ndarray
    mi_labels: np.ndarray

    def dataset(self) -> WindowDataset:
        return WindowDataset(self.windows, self.subject_ids, self.mi_labels)


def _pick_session(per_session: dict[str, SubjectWindows], session_role: str) -> SubjectWindows:
    return next(v for v in per_session.values() if v.session_role == session_role)


def load_session_windows(subject: int, cfg: DataConfig, session_role: str) -> SubjectWindows:
    """Load one subject's windows for ``session_role`` ('T' or 'E') from the cache."""
    return _pick_session(load_subject_cached(subject, cfg), session_role)


def pool_subjects(
    subjects: Sequence[int],
    cfg: DataConfig,
    session_role: str = "T",
) -> PooledWindows:
    """Pool windows of several subjects for one session role into a single array set.

    Args:
        subjects: 1-based subject ids to include.
        cfg: data config.
        session_role: 'T' (train) or 'E' (evaluation).

    Returns:
        A :class:`PooledWindows` with concatenated windows / 0-based subject ids / MI labels.
    """
    win_list: List[np.ndarray] = []
    subj_list: List[np.ndarray] = []
    mi_list: List[np.ndarray] = []
    for s in subjects:
        sw = load_session_windows(s, cfg, session_role)
        win_list.append(sw.windows)
        subj_list.append(sw.subject_id)
        mi_list.append(sw.mi_labels)
    return PooledWindows(
        windows=np.concatenate(win_list, axis=0),
        subject_ids=np.concatenate(subj_list, axis=0),
        mi_labels=np.concatenate(mi_list, axis=0),
    )
