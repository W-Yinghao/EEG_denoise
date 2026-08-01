"""Auditable release-internal Python port of native SGEYESUB.

The algorithm is transcribed from the official reference repository at commit
``2c95b4f46f37670d25399ac0fdd705ae18248b25``:

* ``algorithms/sgeyesub.m`` (LGPL-3.0-or-later),
* ``utility/equalizeLabels.m`` (LGPL-3.0-or-later), and
* ``external/cov_shrink.m`` (GPL-2.0-or-later).

The API deliberately encodes the release-internal protocol: sample-wise
``artifactclasses`` from block 1 are the only fitting labels, and the fitted
operator may only be applied to block 2 with the identical channel layout.
Trial labels are not an input to this implementation.  Only channels whose
channel-location type is ``EEG`` are fitted and changed.

This is a source-faithful FP64 port, not a claim of bitwise MATLAB equivalence:
the optimizer and covariance implementation have not been numerically
cross-validated against MATLAB/EEGLAB.  Every fit therefore reports
``reference_equivalence_status`` explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np


__all__ = [
    "OFFICIAL_ALPHA",
    "OFFICIAL_BETA",
    "OFFICIAL_SOURCE_COMMIT",
    "REFERENCE_EQUIVALENCE_STATUS",
    "NativeSGEyeSubConfig",
    "NativeSGEyeSubFitOutcome",
    "NativeSGEyeSubModel",
    "fit_native_sgeyesub",
]


OFFICIAL_SOURCE_COMMIT = "2c95b4f46f37670d25399ac0fdd705ae18248b25"
OFFICIAL_ALPHA = 1.0
OFFICIAL_BETA = 0.01
REFERENCE_EQUIVALENCE_STATUS = (
    "source_faithful_python_port_not_numerically_cross_validated_with_matlab"
)


class _Ineligible(RuntimeError):
    """Internal control flow for an auditable numerical rejection."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class NativeSGEyeSubConfig:
    """Frozen parameters from ``sgeyesub.m`` at the registered commit."""

    alpha: float = OFFICIAL_ALPHA
    beta: float = OFFICIAL_BETA
    tolerance: float = 1.0e-3
    maximum_iterations: int = 10_000
    numerical_tolerance: float = 1.0e-12

    def __post_init__(self) -> None:
        if float(self.alpha) != OFFICIAL_ALPHA:
            raise ValueError("native SGEYESUB alpha/plr_lambda_l2 is fixed at 1")
        if float(self.beta) != OFFICIAL_BETA:
            raise ValueError("native SGEYESUB beta/plr_lambda_l1 is fixed at 0.01")
        if float(self.tolerance) != 1.0e-3:
            raise ValueError("native SGEYESUB PLR tolerance is fixed at 1e-3")
        if isinstance(self.maximum_iterations, bool) or self.maximum_iterations != 10_000:
            raise ValueError("native SGEYESUB PLR maximum_iterations is fixed at 10000")
        numerical_tolerance = float(self.numerical_tolerance)
        if not np.isfinite(numerical_tolerance) or numerical_tolerance <= 0.0:
            raise ValueError("numerical_tolerance must be finite and positive")

    @property
    def plr_lambda_l2(self) -> float:
        return self.alpha

    @property
    def plr_lambda_l1(self) -> float:
        return self.beta


@dataclass(frozen=True)
class NativeSGEyeSubModel:
    """Fitted two-stage correction with a fixed block-1 channel layout."""

    correction_matrix: np.ndarray
    unmixing_matrix: np.ndarray
    mixing_matrix: np.ndarray
    eeg_channel_indices: tuple[int, ...]
    channel_types: tuple[str, ...]
    diagnostics: dict[str, object]
    fit_block: Literal[1] = 1

    def apply(
        self,
        data: np.ndarray,
        *,
        channel_types: Sequence[str],
        block_id: Literal[2],
    ) -> np.ndarray:
        """Apply the frozen correction to block 2 without reading labels.

        The full channel layout is checked before any correction.  Non-EEG
        channels are copied and never enter the matrix multiplication.
        """

        if block_id != 2:
            raise ValueError("native SGEYESUB may only be applied to block 2")
        query = _as_channel_data(data, name="block2_data")
        normalized_types = _normalize_channel_types(
            channel_types, expected_channels=query.shape[0]
        )
        if normalized_types != self.channel_types:
            raise ValueError("block 2 channel layout differs from fitted block 1")
        if not np.isfinite(query).all():
            raise ValueError("block 2 data contains NaN or Inf")

        flat = _flatten_channels(query)
        corrected = np.array(flat, dtype=np.float64, copy=True)
        eeg_indices = np.asarray(self.eeg_channel_indices, dtype=int)
        corrected[eeg_indices] = self.correction_matrix @ flat[eeg_indices]
        return np.reshape(corrected, query.shape, order="F")


@dataclass(frozen=True)
class NativeSGEyeSubFitOutcome:
    """Eligible model or an explicit, non-silent fitting rejection."""

    status: Literal["eligible", "ineligible"]
    model: NativeSGEyeSubModel | None
    reasons: tuple[str, ...]
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class _PLRResult:
    weights: np.ndarray
    intercept: float
    iterations: int
    relative_weight_change: float


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _as_channel_data(data: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(data, dtype=np.float64)
    if value.ndim < 2 or value.shape[0] < 1 or int(np.prod(value.shape[1:])) < 1:
        raise ValueError(f"{name} must have shape channels x samples [...]")
    return value


def _flatten_channels(value: np.ndarray) -> np.ndarray:
    """Match MATLAB ``data(channel_indices, :)`` trailing-axis flattening."""

    return np.reshape(value, (value.shape[0], -1), order="F")


def _normalize_channel_types(
    channel_types: Sequence[str], *, expected_channels: int
) -> tuple[str, ...]:
    if isinstance(channel_types, (str, bytes)):
        raise TypeError("channel_types must contain one entry per channel")
    normalized = tuple(str(item).strip().upper() for item in channel_types)
    if len(normalized) != expected_channels:
        raise ValueError("channel_types length does not match the data channel axis")
    if any(not item for item in normalized):
        raise ValueError("channel_types entries must be non-empty")
    return normalized


def _base_diagnostics(
    *,
    config: NativeSGEyeSubConfig,
    class_counts: dict[int, int] | None = None,
) -> dict[str, object]:
    return {
        "method_id": "native_sgeyesub_python_release_internal",
        "official_source_commit": OFFICIAL_SOURCE_COMMIT,
        "reference_equivalence_status": REFERENCE_EQUIVALENCE_STATUS,
        "protocol_claim": "release_internal_block1_to_block2_not_paper_reproduction",
        "numeric_precision": "float64",
        "fit_block": 1,
        "apply_block": 2,
        "label_source": "sample_wise_artifactclasses_1_through_6",
        "alpha_plr_lambda_l2": config.alpha,
        "beta_plr_lambda_l1": config.beta,
        "plr_tolerance": config.tolerance,
        "plr_maximum_iterations": config.maximum_iterations,
        "class_counts": class_counts or {},
    }


def _reject(
    reason: str,
    *,
    diagnostics: dict[str, object],
) -> NativeSGEyeSubFitOutcome:
    rejected = dict(diagnostics)
    rejected["fit_status"] = "ineligible"
    rejected["failure_reason"] = reason
    return NativeSGEyeSubFitOutcome(
        status="ineligible",
        model=None,
        reasons=(reason,),
        diagnostics=rejected,
    )


def _validate_artifactclasses(
    artifactclasses: np.ndarray,
    *,
    expected_samples: int,
) -> tuple[np.ndarray | None, dict[int, int], str | None]:
    labels = np.asarray(artifactclasses)
    if labels.size != expected_samples:
        return None, {}, "artifactclasses_shape"
    if not np.issubdtype(labels.dtype, np.number):
        return None, {}, "artifactclasses_dtype"
    numeric = np.reshape(labels, (-1,), order="F").astype(np.float64, copy=False)
    if not np.isfinite(numeric).all():
        return None, {}, "artifactclasses_nonfinite"
    rounded = np.rint(numeric)
    if not np.array_equal(numeric, rounded):
        return None, {}, "artifactclasses_noninteger"
    integer = rounded.astype(np.int64)
    if np.any((integer < 1) | (integer > 6)):
        return None, {}, "artifactclasses_outside_1_through_6"
    counts = {label: int(np.sum(integer == label)) for label in range(1, 7)}
    missing = [label for label, count in counts.items() if count == 0]
    if missing:
        return None, counts, "missing_artifactclasses_" + "_".join(map(str, missing))
    return integer, counts, None


def _equalize_labels(
    x: np.ndarray,
    labels: np.ndarray,
    *,
    mode: Literal["max", "min"],
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic ``equalizeLabels(..., do_shuffle=false)`` translation."""

    unique = np.unique(labels)
    if unique.size < 2:
        raise _Ineligible("equalize_labels_requires_multiple_classes")
    counts = np.asarray([np.sum(labels == label) for label in unique], dtype=int)
    if np.any(counts == 0):
        raise _Ineligible("equalize_labels_empty_class")
    target = int(np.max(counts) if mode == "max" else np.min(counts))
    data_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    for label in unique:
        indices = np.flatnonzero(labels == label)
        selected = indices[np.arange(target, dtype=int) % indices.size]
        data_parts.append(x[:, selected])
        label_parts.append(np.full(target, int(label), dtype=np.int64))
    return np.concatenate(data_parts, axis=1), np.concatenate(label_parts)


def _covariance_shrink(
    observations: np.ndarray,
    *,
    stage: str,
    tolerance: float,
) -> tuple[np.ndarray, float]:
    """Schaefer--Strimmer analytical shrinkage used by ``cov_shrink.m``.

    Rows are observations and columns are variables, exactly as in the
    official helper.  The unweighted path is the only path SGEYESUB invokes.
    """

    x = np.asarray(observations, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2 or x.shape[1] < 1:
        raise _Ineligible(f"insufficient_covariance_samples_{stage}")
    if not np.isfinite(x).all():
        raise _Ineligible(f"nonfinite_covariance_input_{stage}")
    n, variables = x.shape
    weights = np.full(n, 1.0 / n, dtype=np.float64)
    mean = np.sum(x * weights[:, None], axis=0)
    centered = x - mean
    population_variance = np.sum(weights[:, None] * centered**2, axis=0)
    variance = population_variance * n / (n - 1)
    scale_floor = tolerance * max(
        float(np.max(np.abs(variance))), np.finfo(np.float64).tiny
    )
    if np.any(~np.isfinite(variance)) or np.any(variance <= scale_floor):
        raise _Ineligible(f"singular_covariance_{stage}")
    if variables == 1:
        return np.asarray([[variance[0]]], dtype=np.float64), 0.0

    standard_deviation = np.sqrt(variance)
    standardized = centered / standard_deviation
    # cov_shrink.m constructs the sample-wise outer-product tensor in blocks.
    # The two matrix products below are the same first and second moments but
    # avoid an observations x channels x channels allocation on full records.
    product_mean = standardized.T @ standardized / n
    squared = standardized**2
    product_second_moment = squared.T @ squared / n
    product_population_variance = np.maximum(
        product_second_moment - product_mean**2,
        0.0,
    )
    variance_of_correlation = product_population_variance * n**2 / (n - 1) ** 3
    correlation = product_mean * n / (n - 1)
    np.fill_diagonal(correlation, 1.0)
    lower = np.tril_indices(variables, k=-1)
    denominator = float(np.sum(correlation[lower] ** 2))
    numerator = float(np.sum(variance_of_correlation[lower]))
    if not np.isfinite(denominator) or denominator <= tolerance:
        raise _Ineligible(f"singular_covariance_{stage}")
    shrinkage = float(np.clip(numerator / denominator, 0.0, 1.0))
    correlation = (1.0 - shrinkage) * correlation + shrinkage * np.eye(variables)
    covariance = standard_deviation[:, None] * correlation * standard_deviation[None, :]
    covariance = (covariance + covariance.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(covariance)
    eigenvalue_floor = tolerance * max(
        float(np.max(np.abs(eigenvalues))), np.finfo(np.float64).tiny
    )
    if (
        not np.isfinite(covariance).all()
        or not np.isfinite(eigenvalues).all()
        or float(np.min(eigenvalues)) <= eigenvalue_floor
    ):
        raise _Ineligible(f"singular_covariance_{stage}")
    return covariance, shrinkage


def _sigmoid(value: np.ndarray) -> np.ndarray:
    result = np.empty_like(value, dtype=np.float64)
    positive = value >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    result[~positive] = exp_value / (1.0 + exp_value)
    return result


def _soft_threshold(value: np.ndarray, threshold: float) -> np.ndarray:
    return np.maximum(0.0, np.abs(value) - threshold) * np.sign(value)


def _fit_plr(
    x: np.ndarray,
    labels: np.ndarray,
    noise_covariance: np.ndarray,
    *,
    config: NativeSGEyeSubConfig,
    stage: str,
) -> _PLRResult:
    """Elastic-net PLR translated from the official proximal iteration."""

    features = np.asarray(x, dtype=np.float64)
    target = np.asarray(labels, dtype=np.float64).reshape(-1)
    dimensions, samples = features.shape
    if target.shape != (samples,) or set(np.unique(target)) != {0.0, 1.0}:
        raise _Ineligible(f"invalid_binary_labels_{stage}")
    penalty = config.alpha * np.asarray(noise_covariance, dtype=np.float64)
    if penalty.shape != (dimensions, dimensions):
        raise _Ineligible(f"noise_covariance_shape_{stage}")

    weights = np.zeros(dimensions, dtype=np.float64)
    intercept = 0.0
    old_weights = np.array(weights, copy=True)
    old_intercept = 0.0
    lipschitz: float | None = None
    intercept_curvature: float | None = None
    relative_change = float("inf")

    for iteration in range(1, config.maximum_iterations + 1):
        momentum = (iteration - 1.0) / (iteration + 2.0)
        accelerated_weights = weights + momentum * (weights - old_weights)
        accelerated_intercept = intercept + momentum * (intercept - old_intercept)
        old_weights = np.array(weights, copy=True)
        old_intercept = float(intercept)

        # sgeyesub.m evaluates the sigmoid with the current (not accelerated)
        # intercept; this preserves that source behavior deliberately.
        probability = _sigmoid(accelerated_weights @ features + intercept)
        gradient = (
            features @ (target - probability) / samples
            - penalty @ accelerated_weights
        )
        intercept_gradient = float(np.mean(target - probability))
        if iteration % 10 == 1:
            curvature = probability * (1.0 - probability)
            hessian = (features * curvature[None, :]) @ features.T / samples + penalty
            singular_values = np.linalg.svd(hessian, compute_uv=False)
            lipschitz = float(np.max(singular_values))
            intercept_curvature = float(np.mean(curvature))
        if (
            lipschitz is None
            or intercept_curvature is None
            or not np.isfinite(lipschitz)
            or not np.isfinite(intercept_curvature)
            or lipschitz <= config.numerical_tolerance
            or intercept_curvature <= config.numerical_tolerance
        ):
            raise _Ineligible(f"singular_plr_hessian_{stage}")

        weights = _soft_threshold(
            accelerated_weights + gradient / lipschitz,
            config.beta / lipschitz,
        )
        intercept = accelerated_intercept + intercept_gradient / intercept_curvature
        if not np.isfinite(weights).all() or not np.isfinite(intercept):
            raise _Ineligible(f"nonfinite_plr_{stage}")
        change = float(np.linalg.norm(weights - old_weights))
        norm = float(np.linalg.norm(weights))
        if norm <= config.numerical_tolerance:
            relative_change = 0.0 if change <= config.numerical_tolerance else float("inf")
        else:
            relative_change = change / norm
        if relative_change <= config.tolerance:
            return _PLRResult(
                weights=_readonly(weights[:, None]),
                intercept=float(intercept),
                iterations=iteration,
                relative_weight_change=float(relative_change),
            )
    raise _Ineligible(f"plr_nonconvergence_{stage}")


def _right_solve(
    numerator: np.ndarray,
    denominator: np.ndarray,
    *,
    stage: str,
    tolerance: float,
) -> np.ndarray:
    singular_values = np.linalg.svd(denominator, compute_uv=False)
    if (
        singular_values.size == 0
        or not np.isfinite(singular_values).all()
        or singular_values[-1]
        <= tolerance
        * max(float(singular_values[0]), np.finfo(np.float64).tiny)
    ):
        raise _Ineligible(f"singular_covariance_{stage}")
    try:
        return np.linalg.solve(denominator.T, numerator.T).T
    except np.linalg.LinAlgError as error:
        raise _Ineligible(f"singular_covariance_{stage}") from error


def fit_native_sgeyesub(
    block1_data: np.ndarray,
    artifactclasses: np.ndarray,
    *,
    channel_types: Sequence[str],
    block_id: Literal[1],
    config: NativeSGEyeSubConfig | None = None,
) -> NativeSGEyeSubFitOutcome:
    """Fit native SGEYESUB on release-internal block 1 only.

    No block-2/query value, query label, trial label, or evaluation outcome is
    accepted by this interface.  Missing artifact classes, optimization
    failures, and singular covariance geometry return ``ineligible`` rather
    than silently changing the algorithm.
    """

    if block_id != 1:
        raise ValueError("native SGEYESUB must be fitted on block 1")
    frozen = config or NativeSGEyeSubConfig()
    calibration = _as_channel_data(block1_data, name="block1_data")
    normalized_types = _normalize_channel_types(
        channel_types, expected_channels=calibration.shape[0]
    )
    flat = _flatten_channels(calibration)
    labels, class_counts, label_problem = _validate_artifactclasses(
        artifactclasses,
        expected_samples=flat.shape[1],
    )
    diagnostics = _base_diagnostics(config=frozen, class_counts=class_counts)
    diagnostics["input_channels"] = int(calibration.shape[0])
    diagnostics["input_samples"] = int(flat.shape[1])
    if not np.isfinite(flat).all():
        return _reject("nonfinite_block1_data", diagnostics=diagnostics)
    if label_problem is not None or labels is None:
        return _reject(label_problem or "invalid_artifactclasses", diagnostics=diagnostics)

    eeg_indices = tuple(
        index for index, channel_type in enumerate(normalized_types) if channel_type == "EEG"
    )
    if not eeg_indices:
        return _reject("missing_EEG_channel_type", diagnostics=diagnostics)
    diagnostics["eeg_channel_indices"] = list(eeg_indices)
    x = flat[np.asarray(eeg_indices, dtype=int)]
    channels = x.shape[0]

    try:
        rest = x[:, labels == 6]
        noise_covariance, shrinkage_rest = _covariance_shrink(
            rest.T,
            stage="rest",
            tolerance=frozen.numerical_tolerance,
        )

        eye, eye_labels = _equalize_labels(
            x[:, (labels >= 1) & (labels <= 4)],
            labels[(labels >= 1) & (labels <= 4)],
            mode="max",
        )
        eye_covariance, shrinkage_eye = _covariance_shrink(
            eye.T,
            stage="eye",
            tolerance=frozen.numerical_tolerance,
        )

        blink = x[:, labels == 5]
        blink_labels = labels[labels == 5]
        rest_labels = labels[labels == 6]
        rest_blink, rest_blink_labels = _equalize_labels(
            np.concatenate([rest, blink, blink], axis=1),
            np.concatenate([rest_labels, blink_labels, blink_labels]),
            mode="min",
        )

        horizontal_mask = (eye_labels == 1) | (eye_labels == 2)
        horizontal = _fit_plr(
            eye[:, horizontal_mask],
            (eye_labels[horizontal_mask] == 2).astype(np.float64),
            noise_covariance,
            config=frozen,
            stage="horizontal",
        )
        vertical_mask = (eye_labels == 3) | (eye_labels == 4)
        vertical = _fit_plr(
            eye[:, vertical_mask],
            (eye_labels[vertical_mask] == 4).astype(np.float64),
            noise_covariance,
            config=frozen,
            stage="vertical",
        )

        # The reference orders vertical then horizontal.
        movement_unmixing = np.concatenate(
            [vertical.weights, horizontal.weights], axis=1
        )
        movement_latent_covariance = (
            movement_unmixing.T @ eye_covariance @ movement_unmixing
        )
        movement_mixing = _right_solve(
            eye_covariance @ movement_unmixing,
            movement_latent_covariance,
            stage="eye_latent",
            tolerance=frozen.numerical_tolerance,
        )
        movement_correction = (
            np.eye(channels, dtype=np.float64)
            - movement_mixing @ movement_unmixing.T
        )

        corrected_rest_blink = movement_correction @ rest_blink
        rest_blink_covariance, shrinkage_rest_blink = _covariance_shrink(
            corrected_rest_blink.T,
            stage="corrected_rest_blink",
            tolerance=frozen.numerical_tolerance,
        )
        corrected_rest = movement_correction @ rest
        corrected_noise_covariance, shrinkage_corrected_rest = _covariance_shrink(
            corrected_rest.T,
            stage="corrected_rest",
            tolerance=frozen.numerical_tolerance,
        )
        blink_fit = _fit_plr(
            corrected_rest_blink,
            (rest_blink_labels == 5).astype(np.float64),
            corrected_noise_covariance,
            config=frozen,
            stage="blink",
        )
        blink_latent_variance = float(
            (blink_fit.weights.T @ rest_blink_covariance @ blink_fit.weights).item()
        )
        if (
            not np.isfinite(blink_latent_variance)
            or blink_latent_variance
            <= frozen.numerical_tolerance
            * max(
                float(np.linalg.norm(rest_blink_covariance, ord=2)),
                np.finfo(np.float64).tiny,
            )
        ):
            raise _Ineligible("singular_covariance_blink_latent")
        blink_mixing = (
            rest_blink_covariance @ blink_fit.weights / blink_latent_variance
        )
        correction = (
            np.eye(channels, dtype=np.float64)
            - blink_mixing @ blink_fit.weights.T
        ) @ movement_correction
        if not np.isfinite(correction).all():
            raise _Ineligible("nonfinite_correction_matrix")
    except _Ineligible as error:
        return _reject(error.reason, diagnostics=diagnostics)
    except np.linalg.LinAlgError:
        return _reject("linear_algebra_failure", diagnostics=diagnostics)

    unmixing = np.concatenate(
        [movement_unmixing, blink_fit.weights], axis=1
    )
    mixing = np.concatenate([movement_mixing, blink_mixing], axis=1)
    diagnostics.update(
        {
            "fit_status": "eligible",
            "eeg_channels": channels,
            "covariance_shrinkage_rest": shrinkage_rest,
            "covariance_shrinkage_eye": shrinkage_eye,
            "covariance_shrinkage_corrected_rest_blink": shrinkage_rest_blink,
            "covariance_shrinkage_corrected_rest": shrinkage_corrected_rest,
            "plr_horizontal_iterations": horizontal.iterations,
            "plr_vertical_iterations": vertical.iterations,
            "plr_blink_iterations": blink_fit.iterations,
            "plr_horizontal_relative_weight_change": horizontal.relative_weight_change,
            "plr_vertical_relative_weight_change": vertical.relative_weight_change,
            "plr_blink_relative_weight_change": blink_fit.relative_weight_change,
            "movement_latent_covariance_condition": float(
                np.linalg.cond(movement_latent_covariance)
            ),
            "blink_latent_variance": blink_latent_variance,
            "two_stage_order": "horizontal_vertical_then_blink",
        }
    )
    model = NativeSGEyeSubModel(
        correction_matrix=_readonly(correction),
        unmixing_matrix=_readonly(unmixing),
        mixing_matrix=_readonly(mixing),
        eeg_channel_indices=eeg_indices,
        channel_types=normalized_types,
        diagnostics=dict(diagnostics),
    )
    return NativeSGEyeSubFitOutcome(
        status="eligible",
        model=model,
        reasons=(),
        diagnostics=dict(diagnostics),
    )
