"""Development-only context/projector participant-linkage diagnostic."""
from __future__ import annotations

from typing import Mapping

import numpy as np
from sklearn.metrics import roc_auc_score


def _projector_vector(value: np.ndarray) -> np.ndarray:
    u, _, _ = np.linalg.svd(value, full_matrices=False)
    rank = min(value.shape[-1], u.shape[-1])
    return (u[:, :rank] @ u[:, :rank].T).reshape(-1)


def linkage(features: Mapping[str, tuple[np.ndarray, np.ndarray]]) -> list[dict[str, float | int | str]]:
    """``features[participant] == (gallery/query context or basis rows)``."""
    owners = sorted(features); galleries = np.stack([features[owner][0] for owner in owners]); queries = np.stack([features[owner][1] for owner in owners])
    distances = np.linalg.norm(queries[:, None] - galleries[None], axis=-1)
    # Treat exact ties fractionally.  This is essential for the population-token
    # negative control: a set of identical vectors carries chance-level, not
    # perfect, linkage information merely because gallery/query ordering agrees.
    top1_values, top3_values = [], []
    for index in range(len(owners)):
        row = distances[index]
        below = int(np.sum(row < row[index] - 1e-12))
        tied = int(np.sum(np.isclose(row, row[index], rtol=0.0, atol=1e-12)))
        top1_values.append(1.0 / tied if below == 0 else 0.0)
        available = max(0, 3 - below)
        top3_values.append(min(available, tied) / tied)
    top1 = np.mean(top1_values)
    top3 = np.mean(top3_values)
    labels, scores = [], []
    for left in range(len(owners)):
        for right in range(len(owners)):
            labels.append(int(left == right)); scores.append(-distances[left, right])
    within = np.diag(distances); between = distances[~np.eye(len(owners), dtype=bool)]
    return [{"top1_accuracy": float(top1), "top3_accuracy": float(top3), "same_different_auroc": float(roc_auc_score(labels, scores)), "within_distance": float(within.mean()), "between_distance": float(between.mean()), "participants": len(owners)}]


def projector_features(bases: Mapping[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {owner: (_projector_vector(pair[0]), _projector_vector(pair[1])) for owner, pair in bases.items()}


__all__ = ["linkage", "projector_features"]
