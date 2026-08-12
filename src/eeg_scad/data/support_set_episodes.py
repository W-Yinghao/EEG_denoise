"""Replayable query-disjoint raw support-set episodes for V25."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from eeg_scad.data.counterfactual_pairs import _load_signal
from eeg_scad.data.eog_latent_streams import EOGStreamSampler


class SupportSetEpisodeSampler(EOGStreamSampler):
    """V24-correct query streams augmented only with S120 support windows."""

    def _support_signal(self, owner: str, session: str, task: str) -> tuple[np.ndarray, np.ndarray, str]:
        try:
            eeg, eog = _load_signal(self.root, owner, session, task)
            return eeg, eog, task
        except FileNotFoundError:
            fallback = next(value for value in self.data["tasks"] if value != task)
            eeg, eog = _load_signal(self.root, owner, session, fallback)
            return eeg, eog, fallback

    def _support_set(self, owner: str, session: str, task: str) -> tuple[np.ndarray, np.ndarray, list[int], str]:
        eeg, eog, actual_task = self._support_signal(owner, session, task)
        length = int(self.data["support_window_samples"])
        count = int(self.data["support_windows"])
        stop = min(int(self.data["support_samples"]), eeg.shape[-1], eog.shape[-1])
        if stop < length:
            raise RuntimeError(f"support shorter than one window: {owner}/{session}/{actual_task}")
        starts = self.rng.integers(0, stop - length + 1, size=count).tolist()
        key = (owner, session, task)
        center, scale = self.coordinates[key]
        seeg = np.stack([eeg[:, start:start+length] / self.eeg_scale[:, None] for start in starts])
        seog = np.stack([(eog[:, start:start+length] - center[:, None]) / scale[:, None] for start in starts])
        return seeg.astype(np.float32), seog.astype(np.float32), starts, actual_task

    def _augment(self, batch: dict[str, Any]) -> dict[str, Any]:
        correct_eeg=[];correct_eog=[];wrong_eeg=[];wrong_eog=[];query_operators=[]
        for meta in batch["meta"]:
            owner=meta["participant"];wrong=meta["wrong_owner"];session=meta["session"];task=meta["task"]
            ce,co,starts,actual=self._support_set(owner,session,task)
            we,wo,wstarts,wactual=self._support_set(wrong,session,task)
            correct_eeg.append(ce);correct_eog.append(co);wrong_eeg.append(we);wrong_eog.append(wo)
            query_operators.append(self.query_operators[(owner,session,task)])
            meta.update({"support_owner":owner,"support_starts":starts,"support_actual_task":actual,"wrong_support_starts":wstarts,"wrong_support_actual_task":wactual})
        return {**batch,"cquery":np.asarray(query_operators,dtype=np.float32),"support_eeg":np.asarray(correct_eeg),"support_eog":np.asarray(correct_eog),"wrong_support_eeg":np.asarray(wrong_eeg),"wrong_support_eog":np.asarray(wrong_eog)}

    def sample_paired(self, batch_size: int, zero_proportion: float = 0.10) -> dict[str, Any]:
        return self._augment(super().sample_paired(batch_size, zero_proportion))

    def sample_natural(self, batch_size: int, evaluator: bool = True) -> dict[str, Any]:
        return self._augment(super().sample_natural(batch_size, evaluator))


def episode_digest(eeg: np.ndarray, eog: np.ndarray, starts: list[int]) -> str:
    digest=hashlib.sha256();digest.update(np.asarray(eeg).tobytes());digest.update(np.asarray(eog).tobytes());digest.update(np.asarray(starts,dtype=np.int32).tobytes());return digest.hexdigest()


__all__=["SupportSetEpisodeSampler","episode_digest"]
