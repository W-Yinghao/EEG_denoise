"""Model-only Gaussian channel and registered exemplar-exposure diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import pairwise_distances
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _class_confidence(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(h, dtype=np.float32)
    return values.argmax(1), np.linalg.norm(values, axis=1)


@dataclass(frozen=True)
class FiberGaussian:
    """Regularized class/H-conditional Gaussian without exemplar state."""

    coefficients: dict[int, np.ndarray]
    intercepts: dict[int, np.ndarray]
    cholesky: dict[tuple[int, int], np.ndarray]
    tertiles: dict[int, tuple[float, float]]
    fiber_dim: int

    @classmethod
    def fit(cls, training_u: np.ndarray, training_h: np.ndarray) -> "FiberGaussian":
        u = np.asarray(training_u, dtype=np.float64)
        h = np.asarray(training_h, dtype=np.float64)
        predicted, confidence = _class_confidence(h)
        coefficients: dict[int, np.ndarray] = {}
        intercepts: dict[int, np.ndarray] = {}
        cholesky: dict[tuple[int, int], np.ndarray] = {}
        tertiles: dict[int, tuple[float, float]] = {}
        for task in sorted(np.unique(predicted)):
            mask = predicted == task
            regression = Ridge(alpha=1e-2).fit(h[mask], u[mask])
            coefficients[int(task)] = regression.coef_.T.astype(np.float32)
            intercepts[int(task)] = np.asarray(regression.intercept_, dtype=np.float32)
            residual = u[mask] - regression.predict(h[mask])
            cuts = np.quantile(confidence[mask], [1 / 3, 2 / 3])
            tertiles[int(task)] = (float(cuts[0]), float(cuts[1]))
            levels = np.digitize(confidence[mask], cuts)
            for level in range(3):
                selected = residual[levels == level]
                if len(selected) < 4:
                    selected = residual
                covariance = LedoitWolf().fit(selected).covariance_
                scale = max(float(np.trace(covariance)) / len(covariance), 1e-8)
                covariance = covariance + np.eye(len(covariance)) * scale * 1e-5
                cholesky[(int(task), level)] = np.linalg.cholesky(covariance).astype(np.float32)
        return cls(coefficients, intercepts, cholesky, tertiles, int(u.shape[1]))

    def sample(self, query_h: np.ndarray, *, seed: int) -> tuple[np.ndarray, list[dict[str, object]]]:
        """Sample using H and frozen parameters only; no source U or subject."""
        h = np.asarray(query_h, dtype=np.float32)
        predicted, confidence = _class_confidence(h)
        rng = np.random.default_rng(seed)
        output = np.empty((len(h), self.fiber_dim), dtype=np.float32)
        coverage: list[dict[str, object]] = []
        for index, (condition, task, value) in enumerate(zip(h, predicted, confidence)):
            task = int(task)
            level = int(np.digitize(value, self.tertiles[task]))
            mean = condition @ self.coefficients[task] + self.intercepts[task]
            output[index] = mean + rng.standard_normal(self.fiber_dim).astype(np.float32) @ self.cholesky[(task, level)].T
            coverage.append({"query_index": index, "predicted_class": task, "confidence_tertile": level})
        return output, coverage

    def sample_many(self, query_h: np.ndarray, *, releases: int, seed: int) -> np.ndarray:
        return np.stack([self.sample(query_h, seed=seed + index)[0] for index in range(releases)])

    def save(self, path: Path) -> None:
        payload: dict[str, np.ndarray] = {"fiber_dim": np.asarray(self.fiber_dim, dtype=np.int64)}
        for task, value in self.coefficients.items():
            payload[f"coef_{task}"] = value
            payload[f"intercept_{task}"] = self.intercepts[task]
            payload[f"tertiles_{task}"] = np.asarray(self.tertiles[task], dtype=np.float64)
            for level in range(3):
                payload[f"chol_{task}_{level}"] = self.cholesky[(task, level)]
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **payload)


def _nearest(query: np.ndarray, bank: np.ndarray, *, exclude_self: bool = False) -> tuple[np.ndarray, np.ndarray]:
    distances: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    for start in range(0, len(query), 64):
        chunk = pairwise_distances(query[start:start + 64], bank)
        if exclude_self:
            rows = np.arange(len(chunk))
            cols = np.arange(start, start + len(chunk))
            chunk[rows, cols] = np.inf
        indices.append(chunk.argmin(axis=1))
        distances.append(chunk.min(axis=1))
    return np.concatenate(distances), np.concatenate(indices)


def fit_membership_attack(training_u: np.ndarray, nontraining_u: np.ndarray, seed: int):
    """Registered distance attack fit without outer-test data."""
    positive, _ = _nearest(training_u, training_u)
    negative, _ = _nearest(nontraining_u, training_u)
    features = np.concatenate([positive, negative])[:, None]
    target = np.concatenate([np.ones(len(positive)), np.zeros(len(negative))])
    return make_pipeline(StandardScaler(), LogisticRegression(random_state=seed)).fit(features, target)


def training_exposure(
    method: str,
    releases: np.ndarray,
    training_u: np.ndarray,
    training_subject: np.ndarray,
    heldout_gallery_u: np.ndarray,
    membership_attack,
    *,
    fold: int,
    seed: int,
    query_subject: np.ndarray,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Training-exemplar exposure for all registered releases (R,N,D)."""
    values = np.asarray(releases, dtype=np.float32)
    if values.ndim == 2:
        values = values[None]
    flat = values.reshape(-1, values.shape[-1])
    nearest_train, donor_index = _nearest(flat, training_u)
    nearest_heldout, _ = _nearest(flat, heldout_gallery_u)
    nonself, _ = _nearest(training_u, training_u, exclude_self=True)
    near_threshold = float(0.1 * np.median(nonself))
    membership_probability = membership_attack.predict_proba(nearest_train[:, None])[:, 1]
    donor_subject = training_subject[donor_index]
    counts = np.bincount(donor_subject, minlength=int(training_subject.max()) + 1)
    shares = counts[counts > 0] / max(counts.sum(), 1)
    summary = {
        "fold": fold,
        "seed": seed,
        "method": method,
        "release_count": int(values.shape[0]),
        "samples": int(len(flat)),
        "exact_copy_rate": float(np.mean(nearest_train <= 1e-7)),
        "near_copy_rate": float(np.mean(nearest_train <= near_threshold)),
        "near_copy_threshold_training_only": near_threshold,
        "nearest_training_fiber_distance": float(nearest_train.mean()),
        "nearest_heldout_fiber_distance": float(nearest_heldout.mean()),
        "membership_attack_probability": float(membership_probability.mean()),
        "membership_attack_positive_rate_0_5": float(np.mean(membership_probability >= 0.5)),
        "nearest_training_donor_max_share": float(shares.max(initial=0.0)),
        "nearest_training_donor_entropy": float(-(shares * np.log(np.maximum(shares, 1e-12))).sum()),
        "interpretation": "registered exposure diagnostic; not formal membership privacy",
    }
    participant_rows: list[dict[str, object]] = []
    tiled_subject = np.tile(query_subject, values.shape[0])
    for subject in sorted(np.unique(query_subject)):
        mask = tiled_subject == subject
        participant_rows.append({
            "fold": fold,
            "seed": seed,
            "method": method,
            "participant": int(subject + 1),
            "exact_copy_rate": float(np.mean(nearest_train[mask] <= 1e-7)),
            "near_copy_rate": float(np.mean(nearest_train[mask] <= near_threshold)),
            "nearest_training_fiber_distance": float(nearest_train[mask].mean()),
            "membership_attack_probability": float(membership_probability[mask].mean()),
        })
    return summary, participant_rows


__all__ = ["FiberGaussian", "fit_membership_attack", "training_exposure"]
