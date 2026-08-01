"""Context-level CGDR metrics and participant-clustered paired inference.

Signals use ``(channels, samples)`` order.  Paired metrics require a verified
clean target; natural-EEG metrics deliberately fall back to observable
subspace attenuation and clean-interval preservation instead of pretending
that an unobserved clean signal exists.

The bootstrap first averages paired context effects within each participant
and source, then resamples those participant/source units.  It never treats a
window or epoch as an independent statistical unit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np


Scalar = str | int | float | bool | None
ContextStatus = Literal[
    "success",
    "failed",
    "abstained",
    "rolled_back",
    "inconclusive",
    "stale_input",
]

_EPS = np.finfo(np.float64).eps
_VALID_CONTEXT_STATUSES = {
    "success",
    "failed",
    "abstained",
    "rolled_back",
    "inconclusive",
    "stale_input",
}

METRIC_FIELDS = (
    "e_parallel",
    "e_perp",
    "d_perp_y",
    "overlap_fraction",
    "time_rrmse",
    "frequency_rrmse",
    "correlation",
    "delta_snr_db",
    "projector_distance",
    "projector_mean_angle_deg",
    "projector_max_angle_deg",
    "oracle_projector_rank",
    "estimated_projector_rank",
    "artifact_attenuation_db",
    "clean_interval_preservation",
    "clean_interval_relative_change",
    "clean_interval_correlation",
    "artifact_mask_overlap_fraction",
    "artifact_mask_iou",
)


@dataclass(frozen=True)
class ContextIdentity:
    """Stable identifiers for one complete evaluation context."""

    dataset_id: str
    source_id: str
    participant_id: str
    context_id: str
    method_id: str
    outer_fold: str = ""
    session_id: str = ""
    operator_source: str = ""
    seed: int | None = None


@dataclass(frozen=True)
class RuntimeEvaluation:
    """Optional measured computational cost for the same context."""

    latency_seconds: float | None = None
    peak_memory_bytes: int | None = None
    function_evaluations: int | None = None
    score_evaluations: int | None = None
    energy_evaluations: int | None = None
    model_forward_evaluations: int | None = None

    def as_fields(self) -> dict[str, Scalar]:
        values: dict[str, Scalar] = {
            "latency_seconds": self.latency_seconds,
            "peak_memory_bytes": self.peak_memory_bytes,
            "function_evaluations": self.function_evaluations,
            "score_evaluations": self.score_evaluations,
            "energy_evaluations": self.energy_evaluations,
            "model_forward_evaluations": self.model_forward_evaluations,
        }
        for name, value in values.items():
            if value is not None and float(value) < 0:
                raise ValueError(f"{name} must be non-negative")
        return values


@dataclass(frozen=True)
class PairedBootstrapCI:
    """One method-minus-reference contrast at participant/source level."""

    source_id: str
    metric: str
    method_id: str
    reference_method_id: str
    status: Literal["estimated", "inconclusive"]
    estimate: float | None
    ci_low: float | None
    ci_high: float | None
    confidence: float
    bootstrap_replicates: int
    participants_total: int
    participants_with_metric: int
    participants_without_metric: int
    context_keys_total: int
    paired_contexts: int
    numeric_context_pairs: int
    unpaired_contexts: int
    method_non_success: int
    reference_non_success: int
    method_pop_fallbacks: int
    reference_pop_fallbacks: int
    aggregation: str
    effect_definition: str = "method_minus_reference"
    statistical_unit: str = "source_participant"

    def as_row(self) -> dict[str, Scalar]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


def _as_signal(value: np.ndarray, name: str) -> np.ndarray:
    signal = np.asarray(value, dtype=np.float64)
    if signal.ndim == 1:
        signal = signal[None, :]
    if signal.ndim != 2 or signal.shape[0] < 1 or signal.shape[1] < 2:
        raise ValueError(f"{name} must have shape (channels, samples); got {signal.shape}")
    if not np.isfinite(signal).all():
        raise ValueError(f"{name} contains non-finite values")
    return signal


def _aligned_signals(**signals: np.ndarray) -> dict[str, np.ndarray]:
    converted = {name: _as_signal(value, name) for name, value in signals.items()}
    shapes = {value.shape for value in converted.values()}
    if len(shapes) != 1:
        detail = ", ".join(f"{name}={value.shape}" for name, value in converted.items())
        raise ValueError(f"signals are not aligned: {detail}")
    return converted


def _as_mask(
    mask: np.ndarray | None,
    samples: int,
    name: str,
    *,
    require_nonempty: bool = True,
) -> np.ndarray:
    if mask is None:
        return np.ones(samples, dtype=bool)
    value = np.asarray(mask)
    if value.ndim != 1 or value.size != samples:
        raise ValueError(f"{name} must have shape ({samples},); got {value.shape}")
    if value.dtype != np.bool_:
        if not np.isin(value, (0, 1)).all():
            raise ValueError(f"{name} must contain only boolean or 0/1 values")
        value = value.astype(bool)
    if require_nonempty and not np.any(value):
        raise ValueError(f"{name} selects no samples")
    return value


def _selected(signal: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    selection = _as_mask(mask, signal.shape[1], "mask")
    return signal[:, selection]


def _relative_norm(numerator: np.ndarray, denominator: np.ndarray) -> float | None:
    denominator_norm = float(np.linalg.norm(denominator))
    if denominator_norm <= _EPS * np.sqrt(max(denominator.size, 1)):
        return None
    return float(np.linalg.norm(numerator) / denominator_norm)


def _attenuation_db(reference: np.ndarray, residual: np.ndarray) -> float | None:
    reference_energy = float(np.sum(np.square(reference)))
    if reference_energy <= _EPS * max(reference.size, 1):
        return None
    residual_energy = float(np.sum(np.square(residual)))
    floor = _EPS * reference_energy
    return float(10.0 * np.log10(reference_energy / max(residual_energy, floor)))


def _orthoprojector(
    value: np.ndarray,
    channels: int,
    name: str,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    projector = np.asarray(value, dtype=np.float64)
    if projector.shape != (channels, channels):
        raise ValueError(f"{name} must have shape {(channels, channels)}; got {projector.shape}")
    if not np.isfinite(projector).all():
        raise ValueError(f"{name} contains non-finite values")
    scale = max(float(np.linalg.norm(projector, ord="fro")), 1.0)
    symmetry_error = float(np.linalg.norm(projector - projector.T, ord="fro")) / scale
    idempotence_error = float(np.linalg.norm(projector @ projector - projector, ord="fro")) / scale
    if symmetry_error > tolerance or idempotence_error > tolerance:
        raise ValueError(
            f"{name} is not an orthogonal projector: "
            f"symmetry={symmetry_error:.3g}, idempotence={idempotence_error:.3g}"
        )
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (projector + projector.T))
    basis = eigenvectors[:, eigenvalues > 0.5]
    canonical = basis @ basis.T
    return canonical, basis


def time_rrmse(
    restored: np.ndarray,
    clean: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> float | None:
    """Relative temporal RMS error over a complete context or selected interval."""

    values = _aligned_signals(restored=restored, clean=clean)
    estimate = _selected(values["restored"], mask)
    target = _selected(values["clean"], mask)
    return _relative_norm(estimate - target, target)


def frequency_rrmse(
    restored: np.ndarray,
    clean: np.ndarray,
    *,
    sampling_rate: float,
    frequency_band: tuple[float, float] | None = None,
    mask: np.ndarray | None = None,
) -> float | None:
    """Relative RMS difference between rectangular-periodogram powers."""

    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive")
    values = _aligned_signals(restored=restored, clean=clean)
    estimate = _selected(values["restored"], mask)
    target = _selected(values["clean"], mask)
    if estimate.shape[1] < 2:
        return None
    frequencies = np.fft.rfftfreq(estimate.shape[1], d=1.0 / sampling_rate)
    if frequency_band is None:
        selected_frequencies = np.ones(frequencies.size, dtype=bool)
    else:
        low, high = frequency_band
        if low < 0 or high <= low or high > sampling_rate / 2.0 + 1e-12:
            raise ValueError("frequency_band must lie within [0, Nyquist]")
        selected_frequencies = (frequencies >= low) & (frequencies <= high)
    if not np.any(selected_frequencies):
        return None
    estimate_power = np.abs(np.fft.rfft(estimate, axis=1)) ** 2 / estimate.shape[1]
    target_power = np.abs(np.fft.rfft(target, axis=1)) ** 2 / target.shape[1]
    estimate_power = estimate_power[:, selected_frequencies]
    target_power = target_power[:, selected_frequencies]
    return _relative_norm(estimate_power - target_power, target_power)


def correlation(
    restored: np.ndarray,
    clean: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> float | None:
    """Mean channel-wise Pearson correlation, excluding constant channels."""

    values = _aligned_signals(restored=restored, clean=clean)
    estimate = _selected(values["restored"], mask)
    target = _selected(values["clean"], mask)
    estimate = estimate - estimate.mean(axis=1, keepdims=True)
    target = target - target.mean(axis=1, keepdims=True)
    numerator = np.sum(estimate * target, axis=1)
    denominator = np.linalg.norm(estimate, axis=1) * np.linalg.norm(target, axis=1)
    valid = denominator > _EPS * max(estimate.shape[1], 1)
    if not np.any(valid):
        return None
    values_per_channel = numerator[valid] / denominator[valid]
    return float(np.mean(np.clip(values_per_channel, -1.0, 1.0)))


def delta_snr_db(
    restored: np.ndarray,
    observed: np.ndarray,
    clean: np.ndarray,
    *,
    mask: np.ndarray | None = None,
) -> float | None:
    """SNR(restored, clean) minus SNR(observed, clean), in decibels."""

    values = _aligned_signals(restored=restored, observed=observed, clean=clean)
    restored_value = _selected(values["restored"], mask)
    observed_value = _selected(values["observed"], mask)
    clean_value = _selected(values["clean"], mask)
    return _attenuation_db(observed_value - clean_value, restored_value - clean_value)


def subspace_error_metrics(
    restored: np.ndarray,
    observed: np.ndarray,
    projector: np.ndarray,
    *,
    clean: np.ndarray | None = None,
    projector_tolerance: float = 1e-6,
) -> dict[str, float | None]:
    """Compute contamination-span error and orthogonal-complement damage.

    ``e_parallel`` is residual paired error in the registered contamination
    span relative to the original paired artifact. ``e_perp`` is collateral
    paired error outside that span relative to the clean complement.
    ``d_perp_y`` is the output's change from the observation outside the span
    relative to the observed complement and is available without clean truth.
    """

    values = _aligned_signals(restored=restored, observed=observed)
    restored_value = values["restored"]
    observed_value = values["observed"]
    projection, _ = _orthoprojector(
        projector, observed_value.shape[0], "projector", projector_tolerance
    )
    complement = np.eye(observed_value.shape[0], dtype=np.float64) - projection
    output: dict[str, float | None] = {
        "e_parallel": None,
        "e_perp": None,
        "d_perp_y": _relative_norm(
            complement @ (restored_value - observed_value),
            complement @ observed_value,
        ),
    }
    if clean is None:
        return output
    clean_value = _as_signal(clean, "clean")
    if clean_value.shape != observed_value.shape:
        raise ValueError("clean is not aligned with restored and observed")
    original_artifact = observed_value - clean_value
    restoration_error = restored_value - clean_value
    output["e_parallel"] = _relative_norm(
        projection @ restoration_error,
        projection @ original_artifact,
    )
    output["e_perp"] = _relative_norm(
        complement @ restoration_error,
        complement @ clean_value,
    )
    return output


def projector_metrics(
    estimated_projector: np.ndarray,
    oracle_projector: np.ndarray,
    *,
    projector_tolerance: float = 1e-6,
) -> dict[str, float | int | None]:
    """Captured oracle fraction, normalized chordal distance, and angles."""

    oracle_raw = np.asarray(oracle_projector)
    if oracle_raw.ndim != 2 or oracle_raw.shape[0] != oracle_raw.shape[1]:
        raise ValueError("oracle_projector must be square")
    channels = int(oracle_raw.shape[0])
    estimated, estimated_basis = _orthoprojector(
        estimated_projector, channels, "estimated_projector", projector_tolerance
    )
    oracle, oracle_basis = _orthoprojector(
        oracle_projector, channels, "oracle_projector", projector_tolerance
    )
    estimated_rank = int(estimated_basis.shape[1])
    oracle_rank = int(oracle_basis.shape[1])
    rank_sum = estimated_rank + oracle_rank
    distance = 0.0 if rank_sum == 0 else float(
        np.linalg.norm(estimated - oracle, ord="fro") / np.sqrt(rank_sum)
    )
    overlap = None
    if oracle_rank > 0:
        overlap = float(np.clip(np.trace(oracle @ estimated) / oracle_rank, 0.0, 1.0))
    angles: list[float] = []
    if estimated_rank > 0 and oracle_rank > 0:
        singular_values = np.linalg.svd(
            oracle_basis.T @ estimated_basis, compute_uv=False
        )
        angles.extend(np.arccos(np.clip(singular_values, 0.0, 1.0)).tolist())
    angles.extend([np.pi / 2.0] * abs(estimated_rank - oracle_rank))
    angles_degrees = np.rad2deg(np.asarray(angles, dtype=np.float64))
    return {
        "overlap_fraction": overlap,
        "projector_distance": distance,
        "projector_mean_angle_deg": (
            float(np.mean(angles_degrees)) if angles_degrees.size else None
        ),
        "projector_max_angle_deg": (
            float(np.max(angles_degrees)) if angles_degrees.size else None
        ),
        "oracle_projector_rank": oracle_rank,
        "estimated_projector_rank": estimated_rank,
    }


def artifact_attenuation(
    restored: np.ndarray,
    observed: np.ndarray,
    projector: np.ndarray,
    *,
    clean: np.ndarray | None = None,
    artifact_mask: np.ndarray | None = None,
    projector_tolerance: float = 1e-6,
) -> tuple[float | None, str]:
    """Artifact-span attenuation in dB and its paired/proxy interpretation."""

    values = _aligned_signals(restored=restored, observed=observed)
    restored_value = values["restored"]
    observed_value = values["observed"]
    selection = _as_mask(
        artifact_mask,
        observed_value.shape[1],
        "artifact_mask",
        require_nonempty=False,
    )
    projection, _ = _orthoprojector(
        projector, observed_value.shape[0], "projector", projector_tolerance
    )
    if clean is not None:
        clean_value = _as_signal(clean, "clean")
        if clean_value.shape != observed_value.shape:
            raise ValueError("clean is not aligned with restored and observed")
        if not np.any(selection):
            return None, "paired_artifact_residual"
        reference = projection @ (observed_value[:, selection] - clean_value[:, selection])
        residual = projection @ (restored_value[:, selection] - clean_value[:, selection])
        return _attenuation_db(reference, residual), "paired_artifact_residual"
    if not np.any(selection):
        return None, "projected_observation_proxy"
    reference = projection @ observed_value[:, selection]
    residual = projection @ restored_value[:, selection]
    return _attenuation_db(reference, residual), "projected_observation_proxy"


def clean_interval_preservation(
    restored: np.ndarray,
    observed: np.ndarray,
    *,
    clean_mask: np.ndarray,
) -> dict[str, float | None]:
    """Preservation against the deployable observation on clean intervals.

    The score is ``1 - relative_change`` and is intentionally not clipped:
    one means unchanged, zero means a change as large as the observed signal,
    and negative values expose destructive failures.
    """

    values = _aligned_signals(restored=restored, observed=observed)
    selection = _as_mask(clean_mask, values["observed"].shape[1], "clean_mask")
    restored_value = values["restored"][:, selection]
    observed_value = values["observed"][:, selection]
    relative_change = _relative_norm(restored_value - observed_value, observed_value)
    return {
        "clean_interval_preservation": (
            None if relative_change is None else 1.0 - relative_change
        ),
        "clean_interval_relative_change": relative_change,
        "clean_interval_correlation": correlation(restored_value, observed_value),
    }


def mask_overlap_metrics(
    predicted_artifact_mask: np.ndarray,
    artifact_mask: np.ndarray,
) -> dict[str, float | None]:
    """Temporal artifact-mask recall (overlap fraction) and intersection/union."""

    truth = np.asarray(artifact_mask)
    predicted = np.asarray(predicted_artifact_mask)
    if truth.ndim != 1 or predicted.shape != truth.shape:
        raise ValueError("artifact masks must be aligned one-dimensional arrays")
    truth = _as_mask(
        truth,
        truth.size,
        "artifact_mask",
        require_nonempty=False,
    )
    if predicted.dtype != np.bool_:
        if not np.isin(predicted, (0, 1)).all():
            raise ValueError("predicted_artifact_mask must contain boolean or 0/1 values")
        predicted = predicted.astype(bool)
    intersection = int(np.count_nonzero(truth & predicted))
    union = int(np.count_nonzero(truth | predicted))
    truth_count = int(np.count_nonzero(truth))
    return {
        "artifact_mask_overlap_fraction": (
            float(intersection / truth_count) if truth_count else None
        ),
        "artifact_mask_iou": float(intersection / union) if union else None,
    }


def _blank_metric_fields() -> dict[str, Scalar]:
    return {name: None for name in METRIC_FIELDS}


def evaluate_context(
    identity: ContextIdentity,
    *,
    status: ContextStatus,
    observed: np.ndarray | None = None,
    restored: np.ndarray | None = None,
    sampling_rate: float | None = None,
    clean: np.ndarray | None = None,
    oracle_projector: np.ndarray | None = None,
    estimated_projector: np.ndarray | None = None,
    artifact_mask: np.ndarray | None = None,
    predicted_artifact_mask: np.ndarray | None = None,
    clean_mask: np.ndarray | None = None,
    frequency_band: tuple[float, float] | None = None,
    fallback_method_id: str | None = None,
    failure_reason: str | None = None,
    runtime: RuntimeEvaluation | None = None,
    extra_fields: Mapping[str, Scalar] | None = None,
    projector_tolerance: float = 1e-6,
) -> dict[str, Scalar]:
    """Build one flat context row without dropping failures or POP fallbacks."""

    if status not in _VALID_CONTEXT_STATUSES:
        raise ValueError(f"unsupported context status: {status}")
    row: dict[str, Scalar] = {
        "dataset_id": identity.dataset_id,
        "source_id": identity.source_id,
        "participant_id": identity.participant_id,
        "outer_fold": identity.outer_fold,
        "session_id": identity.session_id,
        "context_id": identity.context_id,
        "method_id": identity.method_id,
        "operator_source": identity.operator_source,
        "seed": identity.seed,
        "status": status,
        "failure_reason": failure_reason,
        "fallback_method_id": fallback_method_id,
        "pop_fallback_used": fallback_method_id == "POP",
        "paired_target_available": clean is not None,
        "subspace_metric_basis": None,
        "artifact_attenuation_mode": None,
        **_blank_metric_fields(),
        **(runtime or RuntimeEvaluation()).as_fields(),
    }
    if extra_fields:
        overlap = set(row).intersection(extra_fields)
        if overlap:
            raise ValueError(f"extra_fields may not overwrite reserved fields: {sorted(overlap)}")
        row.update(extra_fields)
    if observed is None or restored is None:
        if status == "success":
            raise ValueError("a successful context requires observed and restored signals")
        return row
    values = _aligned_signals(observed=observed, restored=restored)
    observed_value = values["observed"]
    restored_value = values["restored"]
    clean_value = None
    if clean is not None:
        clean_value = _as_signal(clean, "clean")
        if clean_value.shape != observed_value.shape:
            raise ValueError("clean is not aligned with observed and restored")
        if sampling_rate is None:
            raise ValueError("paired frequency metrics require sampling_rate")
        row["time_rrmse"] = time_rrmse(restored_value, clean_value)
        row["frequency_rrmse"] = frequency_rrmse(
            restored_value,
            clean_value,
            sampling_rate=sampling_rate,
            frequency_band=frequency_band,
        )
        row["correlation"] = correlation(restored_value, clean_value)
        row["delta_snr_db"] = delta_snr_db(restored_value, observed_value, clean_value)
    metric_projector = oracle_projector if oracle_projector is not None else estimated_projector
    if metric_projector is not None:
        row["subspace_metric_basis"] = (
            "oracle_projector" if oracle_projector is not None else "estimated_projector"
        )
        row.update(
            subspace_error_metrics(
                restored_value,
                observed_value,
                metric_projector,
                clean=clean_value,
                projector_tolerance=projector_tolerance,
            )
        )
        attenuation, attenuation_mode = artifact_attenuation(
            restored_value,
            observed_value,
            metric_projector,
            clean=clean_value,
            artifact_mask=artifact_mask,
            projector_tolerance=projector_tolerance,
        )
        row["artifact_attenuation_db"] = attenuation
        row["artifact_attenuation_mode"] = attenuation_mode
    if oracle_projector is not None and estimated_projector is not None:
        row.update(
            projector_metrics(
                estimated_projector,
                oracle_projector,
                projector_tolerance=projector_tolerance,
            )
        )
    if clean_mask is None and artifact_mask is not None:
        artifact_selection = _as_mask(
            artifact_mask,
            observed_value.shape[1],
            "artifact_mask",
            require_nonempty=False,
        )
        complement = ~artifact_selection
        if np.any(complement):
            clean_mask = complement
    if clean_mask is not None:
        row.update(
            clean_interval_preservation(
                restored_value,
                observed_value,
                clean_mask=clean_mask,
            )
        )
    if artifact_mask is not None and predicted_artifact_mask is not None:
        row.update(mask_overlap_metrics(predicted_artifact_mask, artifact_mask))
    return row


def _numeric_metric(row: Mapping[str, Any] | None, metric: str) -> float | None:
    if row is None:
        return None
    value = row.get(metric)
    if value is None or isinstance(value, (bool, np.bool_)):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _non_success(row: Mapping[str, Any] | None) -> bool:
    return row is None or row.get("status") != "success"


def _uses_pop(row: Mapping[str, Any] | None) -> bool:
    if row is None:
        return False
    return bool(row.get("pop_fallback_used")) or row.get("fallback_method_id") == "POP"


def _aggregate(values: Sequence[float], aggregation: str) -> float:
    array = np.asarray(values, dtype=np.float64)
    if aggregation == "mean":
        return float(np.mean(array))
    if aggregation == "median":
        return float(np.median(array))
    raise ValueError("aggregation must be 'mean' or 'median'")


def _bootstrap_result(
    *,
    source_id: str,
    participant_effects: Mapping[str, Sequence[float]],
    counters: Mapping[str, int],
    metric: str,
    method_id: str,
    reference_method_id: str,
    confidence: float,
    replicates: int,
    minimum_participants: int,
    aggregation: str,
    rng: np.random.Generator,
) -> PairedBootstrapCI:
    unit_effects = np.asarray(
        [
            _aggregate(participant_effects[participant], aggregation)
            for participant in sorted(participant_effects)
            if participant_effects[participant]
        ],
        dtype=np.float64,
    )
    participants_total = int(counters["participants_total"])
    participants_with_metric = int(unit_effects.size)
    estimate = float(np.mean(unit_effects)) if unit_effects.size else None
    ci_low = None
    ci_high = None
    status: Literal["estimated", "inconclusive"] = "inconclusive"
    if unit_effects.size >= minimum_participants:
        bootstrap = np.empty(replicates, dtype=np.float64)
        for replicate in range(replicates):
            sample = rng.choice(unit_effects, size=unit_effects.size, replace=True)
            bootstrap[replicate] = np.mean(sample)
        alpha = (1.0 - confidence) / 2.0
        ci_low, ci_high = [
            float(value)
            for value in np.quantile(bootstrap, (alpha, 1.0 - alpha))
        ]
        status = "estimated"
    return PairedBootstrapCI(
        source_id=source_id,
        metric=metric,
        method_id=method_id,
        reference_method_id=reference_method_id,
        status=status,
        estimate=estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        confidence=confidence,
        bootstrap_replicates=replicates,
        participants_total=participants_total,
        participants_with_metric=participants_with_metric,
        participants_without_metric=participants_total - participants_with_metric,
        context_keys_total=int(counters["context_keys_total"]),
        paired_contexts=int(counters["paired_contexts"]),
        numeric_context_pairs=int(counters["numeric_context_pairs"]),
        unpaired_contexts=int(counters["unpaired_contexts"]),
        method_non_success=int(counters["method_non_success"]),
        reference_non_success=int(counters["reference_non_success"]),
        method_pop_fallbacks=int(counters["method_pop_fallbacks"]),
        reference_pop_fallbacks=int(counters["reference_pop_fallbacks"]),
        aggregation=aggregation,
    )


def paired_bootstrap_ci(
    rows: Iterable[Mapping[str, Any]],
    *,
    metric: str,
    method_id: str,
    reference_method_id: str,
    source_field: str = "source_id",
    participant_field: str = "participant_id",
    method_field: str = "method_id",
    pair_fields: Sequence[str] = (
        "dataset_id",
        "outer_fold",
        "session_id",
        "context_id",
        "seed",
    ),
    confidence: float = 0.95,
    bootstrap_replicates: int = 2000,
    minimum_participants: int = 2,
    aggregation: Literal["mean", "median"] = "mean",
    seed: int = 20260801,
    include_overall: bool = True,
) -> list[PairedBootstrapCI]:
    """Participant-clustered paired bootstrap, reported separately by source.

    All context keys remain in the denominator. Missing, failed, abstained, and
    POP-fallback rows are counted in each returned object. Only finite paired
    metric values contribute to the numerical effect, and participants with no
    such pair are explicitly reported rather than silently disappearing.
    """

    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    if minimum_participants < 2:
        raise ValueError("minimum_participants must be at least two")
    if method_id == reference_method_id:
        raise ValueError("method_id and reference_method_id must differ")
    if not participant_field:
        raise ValueError("participant_field is required; windows cannot be bootstrap units")
    selected: dict[tuple[str, str, tuple[Any, ...], str], Mapping[str, Any]] = {}
    for row in rows:
        row_method = row.get(method_field)
        if row_method not in (method_id, reference_method_id):
            continue
        if source_field not in row or participant_field not in row:
            raise ValueError("every selected row requires source and participant identifiers")
        source = str(row[source_field])
        participant = str(row[participant_field])
        pair_key = tuple(row.get(field) for field in pair_fields)
        key = (source, participant, pair_key, str(row_method))
        if key in selected:
            raise ValueError(
                "duplicate method row for source/participant/context; add distinguishing pair_fields"
            )
        selected[key] = row
    contexts = sorted(
        {(source, participant, pair_key) for source, participant, pair_key, _ in selected},
        key=lambda item: (item[0], item[1], repr(item[2])),
    )
    if not contexts:
        raise ValueError("no rows match the requested method contrast")

    def empty_counters() -> dict[str, int]:
        return {
            "participants_total": 0,
            "context_keys_total": 0,
            "paired_contexts": 0,
            "numeric_context_pairs": 0,
            "unpaired_contexts": 0,
            "method_non_success": 0,
            "reference_non_success": 0,
            "method_pop_fallbacks": 0,
            "reference_pop_fallbacks": 0,
        }

    source_effects: dict[str, dict[str, list[float]]] = {}
    source_participants: dict[str, set[str]] = {}
    source_counters: dict[str, dict[str, int]] = {}
    for source, participant, pair_key in contexts:
        effects = source_effects.setdefault(source, {})
        effects.setdefault(participant, [])
        source_participants.setdefault(source, set()).add(participant)
        counters = source_counters.setdefault(source, empty_counters())
        counters["context_keys_total"] += 1
        method_row = selected.get((source, participant, pair_key, method_id))
        reference_row = selected.get((source, participant, pair_key, reference_method_id))
        if method_row is None or reference_row is None:
            counters["unpaired_contexts"] += 1
        else:
            counters["paired_contexts"] += 1
        counters["method_non_success"] += int(_non_success(method_row))
        counters["reference_non_success"] += int(_non_success(reference_row))
        counters["method_pop_fallbacks"] += int(_uses_pop(method_row))
        counters["reference_pop_fallbacks"] += int(_uses_pop(reference_row))
        method_value = _numeric_metric(method_row, metric)
        reference_value = _numeric_metric(reference_row, metric)
        if method_value is not None and reference_value is not None:
            effects[participant].append(method_value - reference_value)
            counters["numeric_context_pairs"] += 1

    for source, participants in source_participants.items():
        source_counters[source]["participants_total"] = len(participants)

    rng = np.random.default_rng(seed)
    results = [
        _bootstrap_result(
            source_id=source,
            participant_effects=source_effects[source],
            counters=source_counters[source],
            metric=metric,
            method_id=method_id,
            reference_method_id=reference_method_id,
            confidence=confidence,
            replicates=bootstrap_replicates,
            minimum_participants=minimum_participants,
            aggregation=aggregation,
            rng=rng,
        )
        for source in sorted(source_effects)
    ]
    if include_overall:
        overall_effects: dict[str, list[float]] = {}
        overall_counters = empty_counters()
        for source in sorted(source_effects):
            for participant, effects in source_effects[source].items():
                overall_effects[f"{source}\x1f{participant}"] = effects
            for name in overall_counters:
                if name != "participants_total":
                    overall_counters[name] += source_counters[source][name]
        overall_counters["participants_total"] = len(overall_effects)
        results.append(
            _bootstrap_result(
                source_id="__all__",
                participant_effects=overall_effects,
                counters=overall_counters,
                metric=metric,
                method_id=method_id,
                reference_method_id=reference_method_id,
                confidence=confidence,
                replicates=bootstrap_replicates,
                minimum_participants=minimum_participants,
                aggregation=aggregation,
                rng=rng,
            )
        )
    return results
