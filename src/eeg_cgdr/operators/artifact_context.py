"""Leakage-safe subject-calibrated artifact-transfer utilities.

The routines in this module are deliberately independent of model and runner
code.  They establish the small piece of auditable algebra needed by a later
subject-conditioned inference path:

* EOG coordinates are frozen by explicit order, polarity, and fit-scope
  metadata;
* the FP64 ridge transfer is fitted on calibration/support samples only;
* population and subject predictions are mixed only by a support-frozen
  ``rho``; and
* stochastic outputs are reduced by an eight-sample posterior waveform mean,
  never by target-dependent best-of-K selection.

Arrays use channel-major ``(channels, samples)`` layout throughout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Sequence

import numpy as np


FitScope = Literal["outer_training_only", "support_only"]
RhoSelectionSource = Literal[
    "support_only_diagnostics",
    "pre_frozen_development",
]

POSTERIOR_SAMPLE_COUNT = 8

__all__ = [
    "ArtifactContextCorrection",
    "ArtifactTransfer",
    "EOGStandardizationMetadata",
    "POSTERIOR_SAMPLE_COUNT",
    "SupportOnlyRho",
    "fit_artifact_transfer",
    "fit_eog_standardization",
    "freeze_support_only_rho",
    "population_subject_mixing_correction",
    "posterior_mean_k8",
]


def _nonempty(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _channel_order(value: Sequence[str], *, name: str) -> tuple[str, ...]:
    result = tuple(value)
    if not result:
        raise ValueError(f"{name} must be non-empty")
    if any(not isinstance(channel, str) or not channel.strip() for channel in result):
        raise ValueError(f"{name} entries must be non-empty strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} entries must be unique")
    return result


def _fit_scope(value: str) -> FitScope:
    if value not in ("outer_training_only", "support_only"):
        raise ValueError("fit_scope must be outer_training_only or support_only")
    return value  # type: ignore[return-value]


def _fp64_matrix(value: object, *, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric matrix") from exc
    if result.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional channel-major matrix")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains non-finite values")
    return result


def _readonly_fp64(value: object, *, name: str) -> np.ndarray:
    result = np.array(_fp64_matrix(value, name=name), dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _readonly_fp64_vector(value: object, *, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric vector") from exc
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite one-dimensional vector")
    result = np.array(result, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _canonical_eog(
    eog: object,
    *,
    input_order: Sequence[str],
    canonical_order: Sequence[str],
    polarity: Sequence[float],
) -> np.ndarray:
    value = _fp64_matrix(eog, name="EOG")
    input_channels = _channel_order(input_order, name="input_order")
    canonical_channels = _channel_order(canonical_order, name="canonical_order")
    if value.shape[0] != len(input_channels):
        raise ValueError("input_order does not match the EOG channel dimension")
    if set(input_channels) != set(canonical_channels):
        raise ValueError("input_order and canonical_order must name the same channels")
    signs = np.asarray(tuple(polarity), dtype=np.float64)
    if signs.shape != (len(canonical_channels),):
        raise ValueError("polarity must have one entry per canonical EOG channel")
    if not np.all(np.isin(signs, (-1.0, 1.0))):
        raise ValueError("EOG polarity entries must be exactly -1 or +1")
    indices = [input_channels.index(channel) for channel in canonical_channels]
    return value[indices] * signs[:, None]


@dataclass(frozen=True)
class EOGStandardizationMetadata:
    """Frozen EOG coordinate convention and calibration-only statistics.

    ``polarity`` is aligned with ``canonical_order``.  Means and scales are
    computed after reordering and polarity application.  They are never
    recomputed from an inference/query interval.
    """

    canonical_order: tuple[str, ...]
    polarity: tuple[float, ...]
    mean: tuple[float, ...]
    standard_deviation: tuple[float, ...]
    standardization_scale: tuple[float, ...]
    fit_scope: FitScope
    source_id: str
    standardization: Literal["per_channel_population_std_ddof0"] = (
        "per_channel_population_std_ddof0"
    )

    def __post_init__(self) -> None:
        channels = _channel_order(self.canonical_order, name="canonical_order")
        if len(channels) not in (2, 3):
            raise ValueError("artifact context requires exactly two or three EOG channels")
        object.__setattr__(self, "canonical_order", channels)
        signs = tuple(float(value) for value in self.polarity)
        means = tuple(float(value) for value in self.mean)
        deviations = tuple(float(value) for value in self.standard_deviation)
        scales = tuple(float(value) for value in self.standardization_scale)
        if not (
            len(signs)
            == len(means)
            == len(deviations)
            == len(scales)
            == len(channels)
        ):
            raise ValueError("EOG metadata fields must have the same channel count")
        if any(value not in (-1.0, 1.0) for value in signs):
            raise ValueError("EOG polarity entries must be exactly -1 or +1")
        if not np.isfinite(np.asarray(means, dtype=np.float64)).all():
            raise ValueError("EOG means must be finite")
        if not np.isfinite(np.asarray(deviations, dtype=np.float64)).all() or any(
            value <= 0.0 for value in deviations
        ):
            raise ValueError("EOG standard deviations must be finite and positive")
        if not np.isfinite(np.asarray(scales, dtype=np.float64)).all() or any(
            value <= 0.0 for value in scales
        ):
            raise ValueError("EOG standardization scales must be finite and positive")
        if not np.allclose(
            np.asarray(scales),
            1.0 / np.asarray(deviations),
            rtol=2.0e-13,
            atol=2.0e-13,
        ):
            raise ValueError("EOG scale must equal inverse standard deviation")
        object.__setattr__(self, "polarity", signs)
        object.__setattr__(self, "mean", means)
        object.__setattr__(self, "standard_deviation", deviations)
        object.__setattr__(self, "standardization_scale", scales)
        object.__setattr__(self, "fit_scope", _fit_scope(self.fit_scope))
        _nonempty(self.source_id, name="source_id")

    def canonicalize(
        self,
        eog: object,
        *,
        input_order: Sequence[str],
    ) -> np.ndarray:
        """Apply the registered order and polarity without refitting stats."""

        return _canonical_eog(
            eog,
            input_order=input_order,
            canonical_order=self.canonical_order,
            polarity=self.polarity,
        )

    def standardize(
        self,
        eog: object,
        *,
        input_order: Sequence[str],
    ) -> np.ndarray:
        """Apply support/outer-training statistics to any compatible EOG block."""

        canonical = self.canonicalize(eog, input_order=input_order)
        mean = np.asarray(self.mean, dtype=np.float64)[:, None]
        scale = np.asarray(self.standardization_scale, dtype=np.float64)[:, None]
        return np.asarray((canonical - mean) * scale, dtype=np.float64)


def fit_eog_standardization(
    support_eog: object,
    *,
    input_order: Sequence[str],
    canonical_order: Sequence[str],
    polarity: Sequence[float],
    source_id: str,
    fit_scope: FitScope,
    minimum_scale: float = 1.0e-12,
) -> EOGStandardizationMetadata:
    """Fit registered EOG statistics from one authorized calibration scope."""

    scope = _fit_scope(fit_scope)
    source = _nonempty(source_id, name="source_id")
    scale_floor = float(minimum_scale)
    if not np.isfinite(scale_floor) or scale_floor <= 0.0:
        raise ValueError("minimum_scale must be finite and positive")
    canonical = _canonical_eog(
        support_eog,
        input_order=input_order,
        canonical_order=canonical_order,
        polarity=polarity,
    )
    if canonical.shape[1] < 2:
        raise ValueError("EOG standardization requires at least two samples")
    mean = canonical.mean(axis=1, dtype=np.float64)
    standard_deviation = canonical.std(axis=1, ddof=0, dtype=np.float64)
    if np.any(standard_deviation <= scale_floor):
        raise ValueError("EOG support contains a constant or degenerate channel")
    scale = 1.0 / standard_deviation
    return EOGStandardizationMetadata(
        canonical_order=tuple(canonical_order),
        polarity=tuple(float(value) for value in polarity),
        mean=tuple(float(value) for value in mean),
        standard_deviation=tuple(float(value) for value in standard_deviation),
        standardization_scale=tuple(float(value) for value in scale),
        fit_scope=scope,
        source_id=source,
    )


@dataclass(frozen=True)
class ArtifactTransfer:
    """Raw FP64 ridge diagnostics plus a full-column retained-rank transfer.

    ``raw_transfer_matrix`` is the exact ridge solution in standardized EOG
    coordinates. ``transfer_matrix`` is its rank-``rank`` SVD approximation
    with the *same* ``(EEG,EOG)`` shape; selecting rank never deletes an EOG
    coordinate. Its columns are normalized explicitly, so with
    ``Z=diag(transfer_scale) E`` the full representation obeys
    ``C=C_normalized@diag(transfer_scale)`` and
    ``Delta=C_s E=C_normalized Z``. A third external reference therefore keeps
    a three-dimensional latent even when the development-selected retained
    rank is two. EOG standardization scale is separate and kept
    only in :class:`EOGStandardizationMetadata`.
    """

    raw_transfer_matrix: np.ndarray
    transfer_matrix: np.ndarray
    transfer_normalized: np.ndarray
    transfer_scale: np.ndarray
    eeg_subspace_basis: np.ndarray
    projector: np.ndarray
    singular_values: np.ndarray
    rank: int
    numerical_rank: int
    rank_tolerance: float
    eeg_mean: np.ndarray
    eeg_channel_order: tuple[str, ...]
    eog_metadata: EOGStandardizationMetadata
    ridge_lambda: float
    fit_scope: FitScope
    fit_id: str

    def __post_init__(self) -> None:
        raw_transfer = _readonly_fp64(
            self.raw_transfer_matrix,
            name="raw_transfer_matrix",
        )
        transfer = _readonly_fp64(self.transfer_matrix, name="transfer_matrix")
        normalized = _readonly_fp64(
            self.transfer_normalized,
            name="transfer_normalized",
        )
        transfer_scale = _readonly_fp64_vector(
            self.transfer_scale,
            name="transfer_scale",
        )
        basis = _readonly_fp64(self.eeg_subspace_basis, name="eeg_subspace_basis")
        projector = _readonly_fp64(self.projector, name="projector")
        singular = _readonly_fp64_vector(self.singular_values, name="singular_values")
        eeg_mean = _readonly_fp64_vector(self.eeg_mean, name="eeg_mean")
        channels = _channel_order(self.eeg_channel_order, name="eeg_channel_order")
        if not isinstance(self.eog_metadata, EOGStandardizationMetadata):
            raise TypeError("eog_metadata must be EOGStandardizationMetadata")
        scope = _fit_scope(self.fit_scope)
        if self.eog_metadata.fit_scope != scope:
            raise ValueError("transfer and EOG metadata fit scopes differ")
        rank = int(self.rank)
        if isinstance(self.rank, bool) or rank != self.rank or rank < 1:
            raise ValueError("rank must be a positive integer")
        if raw_transfer.shape != transfer.shape or transfer.shape != normalized.shape:
            raise ValueError("transfer representations must have identical shapes")
        if transfer_scale.shape != (transfer.shape[1],):
            raise ValueError("transfer scales must have one entry per EOG coordinate")
        if np.any(transfer_scale <= 0.0):
            raise ValueError("transfer scales must be strictly positive")
        if transfer.shape != (len(channels), len(self.eog_metadata.canonical_order)):
            raise ValueError("transfer shape does not match registered channel orders")
        if basis.shape != (len(channels), rank):
            raise ValueError("basis shape does not match effective rank")
        if projector.shape != (len(channels), len(channels)):
            raise ValueError("projector shape does not match EEG channels")
        if eeg_mean.shape != (len(channels),):
            raise ValueError("eeg_mean shape does not match EEG channels")
        if rank > min(transfer.shape) or singular.shape != (min(transfer.shape),):
            raise ValueError("rank or singular values do not match transfer dimensions")
        rank_tolerance = float(self.rank_tolerance)
        if not np.isfinite(rank_tolerance) or rank_tolerance < 0.0:
            raise ValueError("rank_tolerance must be finite and non-negative")
        expected_rank = int(np.sum(singular > rank_tolerance))
        numerical_rank = int(self.numerical_rank)
        if expected_rank != numerical_rank or not rank <= numerical_rank:
            raise ValueError("retained/effective ranks are inconsistent")
        ridge = float(self.ridge_lambda)
        if not np.isfinite(ridge) or ridge < 0.0:
            raise ValueError("ridge_lambda must be finite and non-negative")
        if not np.allclose(basis.T @ basis, np.eye(rank), rtol=0.0, atol=1.0e-10):
            raise ValueError("eeg_subspace_basis must be orthonormal")
        if not np.allclose(projector, basis @ basis.T, rtol=0.0, atol=1.0e-10):
            raise ValueError("projector must equal U U.T")
        if np.any(singular < 0.0):
            raise ValueError("singular values must be non-negative")
        if not np.allclose(
            transfer,
            normalized * transfer_scale[None, :],
            rtol=2.0e-12,
            atol=2.0e-12,
        ):
            raise ValueError("full C must equal C_normalized @ diag(scale)")
        if not np.allclose(
            (np.eye(projector.shape[0]) - projector) @ transfer,
            0.0,
            rtol=0.0,
            atol=2.0e-10,
        ):
            raise ValueError("retained C must lie in its registered projector span")
        for name, value in (
            ("raw_transfer_matrix", raw_transfer),
            ("transfer_matrix", transfer),
            ("transfer_normalized", normalized),
            ("transfer_scale", transfer_scale),
            ("eeg_subspace_basis", basis),
            ("projector", projector),
            ("singular_values", singular),
            ("eeg_mean", eeg_mean),
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "eeg_channel_order", channels)
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "numerical_rank", numerical_rank)
        object.__setattr__(self, "rank_tolerance", rank_tolerance)
        object.__setattr__(self, "ridge_lambda", ridge)
        object.__setattr__(self, "fit_scope", scope)
        _nonempty(self.fit_id, name="fit_id")

    def standardized_eog(
        self,
        eog: object,
        *,
        input_order: Sequence[str],
    ) -> np.ndarray:
        return self.eog_metadata.standardize(eog, input_order=input_order)

    def standardized_artifact_latent(
        self,
        eog: object,
        *,
        input_order: Sequence[str],
    ) -> np.ndarray:
        """Return ``Z=diag(scale)E`` in all registered EOG coordinates."""

        standardized = self.standardized_eog(eog, input_order=input_order)
        return np.asarray(
            self.transfer_scale[:, None] * standardized,
            dtype=np.float64,
        )

    def predict_contamination(
        self,
        eog: object,
        *,
        input_order: Sequence[str],
    ) -> np.ndarray:
        """Predict full-transfer contamination through normalized ``C``."""

        latent = self.standardized_artifact_latent(eog, input_order=input_order)
        return np.asarray(
            self.transfer_normalized @ latent,
            dtype=np.float64,
        )

    def predict_contamination_from_full_transfer(
        self,
        eog: object,
        *,
        input_order: Sequence[str],
    ) -> np.ndarray:
        """Equivalent prediction using ``C E`` directly."""

        standardized = self.standardized_eog(eog, input_order=input_order)
        return np.asarray(
            self.transfer_matrix @ standardized,
            dtype=np.float64,
        )

    def reconstruct(
        self,
        observed_eeg: object,
        eog: object,
        *,
        input_order: Sequence[str],
    ) -> np.ndarray:
        """Return ``observed_eeg - C E`` without refitting query statistics."""

        observed = _fp64_matrix(observed_eeg, name="observed_eeg")
        predicted = self.predict_contamination(eog, input_order=input_order)
        if observed.shape != predicted.shape:
            raise ValueError("observed EEG and EOG prediction shapes differ")
        return np.asarray(observed - predicted, dtype=np.float64)


def _column_normalize_transfer(
    transfer: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``C_normalized, scale`` with positive scales and exact recovery."""

    scale = np.linalg.norm(transfer, axis=0)
    safe_scale = np.where(scale > 0.0, scale, 1.0)
    normalized = transfer / safe_scale[None, :]
    return (
        np.asarray(normalized, dtype=np.float64),
        np.asarray(safe_scale, dtype=np.float64),
    )


def fit_artifact_transfer(
    support_eeg: object,
    support_eog: object,
    *,
    eeg_channel_order: Sequence[str],
    eog_input_order: Sequence[str],
    eog_canonical_order: Sequence[str],
    eog_polarity: Sequence[float],
    ridge_lambda: float,
    retained_rank: int,
    fit_scope: FitScope,
    fit_id: str,
) -> ArtifactTransfer:
    """Fit ``C = Y E.T solve(E E.T + lambda I)`` in FP64.

    ``Y`` is centered support EEG and ``E`` is support-standardized canonical
    EOG.  After the solve, column norms define ``scale`` and
    ``C_normalized=C@diag(1/scale)``.  Thus ``Z=diag(scale)E`` and both paths
    give ``Delta=C E=C_normalized Z``.  No query or clean-target argument
    exists in this API.
    """

    eeg = _fp64_matrix(support_eeg, name="support_eeg")
    eog = _fp64_matrix(support_eog, name="support_eog")
    eeg_channels = _channel_order(eeg_channel_order, name="eeg_channel_order")
    if eeg.shape[0] != len(eeg_channels):
        raise ValueError("eeg_channel_order does not match support_eeg")
    if eeg.shape[1] != eog.shape[1] or eeg.shape[1] < 2:
        raise ValueError("support EEG/EOG must be aligned and contain at least two samples")
    scope = _fit_scope(fit_scope)
    identifier = _nonempty(fit_id, name="fit_id")
    ridge = float(ridge_lambda)
    if not np.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge_lambda must be finite and non-negative")
    rank = int(retained_rank)
    if isinstance(retained_rank, bool) or rank != retained_rank or rank < 1:
        raise ValueError("retained_rank must be a positive integer")
    eog_metadata = fit_eog_standardization(
        eog,
        input_order=eog_input_order,
        canonical_order=eog_canonical_order,
        polarity=eog_polarity,
        source_id=identifier,
        fit_scope=scope,
    )
    standardized_eog = eog_metadata.standardize(eog, input_order=eog_input_order)
    eeg_mean = eeg.mean(axis=1, dtype=np.float64)
    y = np.asarray(eeg - eeg_mean[:, None], dtype=np.float64)
    e = np.asarray(standardized_eog, dtype=np.float64)
    gram = e @ e.T + ridge * np.eye(e.shape[0], dtype=np.float64)
    cross = y @ e.T
    try:
        raw_transfer = np.linalg.solve(gram, cross.T).T
    except np.linalg.LinAlgError as exc:
        raise ValueError("regularized EOG Gram matrix is singular") from exc
    basis_full, singular_values, right_full = np.linalg.svd(
        raw_transfer,
        full_matrices=False,
    )
    rank_tolerance = (
        max(raw_transfer.shape)
        * np.finfo(np.float64).eps
        * float(singular_values[0])
    )
    numerical_rank = int(np.sum(singular_values > rank_tolerance))
    if numerical_rank < rank:
        raise ValueError("retained_rank exceeds the raw ridge numerical rank")
    basis = basis_full[:, :rank]
    transfer = (
        basis
        @ np.diag(singular_values[:rank])
        @ right_full[:rank, :]
    )
    projector = basis @ basis.T
    transfer_normalized, transfer_scale = _column_normalize_transfer(transfer)
    return ArtifactTransfer(
        raw_transfer_matrix=raw_transfer,
        transfer_matrix=transfer,
        transfer_normalized=transfer_normalized,
        transfer_scale=transfer_scale,
        eeg_subspace_basis=basis,
        projector=projector,
        singular_values=singular_values,
        rank=rank,
        numerical_rank=numerical_rank,
        rank_tolerance=rank_tolerance,
        eeg_mean=eeg_mean,
        eeg_channel_order=eeg_channels,
        eog_metadata=eog_metadata,
        ridge_lambda=ridge,
        fit_scope=scope,
        fit_id=identifier,
    )


@dataclass(frozen=True)
class SupportOnlyRho:
    """A calibration strength frozen without query outcomes or targets."""

    value: float
    support_id: str
    selection_source: RhoSelectionSource
    fit_scope: Literal["support_only"] = "support_only"

    def __post_init__(self) -> None:
        value = float(self.value)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("rho must be finite and lie in [0, 1]")
        _nonempty(self.support_id, name="support_id")
        if self.selection_source not in (
            "support_only_diagnostics",
            "pre_frozen_development",
        ):
            raise ValueError("rho selection_source is not leakage-safe")
        if self.fit_scope != "support_only":
            raise ValueError("rho fit_scope must be support_only")
        object.__setattr__(self, "value", value)


def freeze_support_only_rho(
    value: float,
    *,
    support_id: str,
    selection_source: RhoSelectionSource,
) -> SupportOnlyRho:
    """Create a support-only rho token; no query outcome can be supplied."""

    return SupportOnlyRho(
        value=value,
        support_id=support_id,
        selection_source=selection_source,
    )


@dataclass(frozen=True)
class ArtifactContextCorrection:
    """Population/subject contamination correction in registered geometry."""

    restored_eeg: np.ndarray
    correction: np.ndarray
    mixed_contamination: np.ndarray
    population_contamination: np.ndarray
    subject_contamination: np.ndarray | None
    union_basis: np.ndarray
    union_projector: np.ndarray
    rho: float
    branch: Literal["population", "mixed", "subject"]
    subject_context_constructed: bool


SubjectTransferFactory = Callable[[], ArtifactTransfer]


def _span_basis(*matrices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    concatenated = np.concatenate(matrices, axis=1)
    basis_full, singular_values, _ = np.linalg.svd(concatenated, full_matrices=False)
    if singular_values.size == 0 or singular_values[0] <= 0.0:
        raise ValueError("artifact transfer union has zero rank")
    tolerance = max(concatenated.shape) * np.finfo(np.float64).eps * singular_values[0]
    rank = int(np.sum(singular_values > tolerance))
    if rank < 1:
        raise ValueError("artifact transfer union has zero numerical rank")
    basis = np.asarray(basis_full[:, :rank], dtype=np.float64)
    return basis, np.asarray(basis @ basis.T, dtype=np.float64)


def _compatible_transfers(
    population: ArtifactTransfer,
    subject: ArtifactTransfer,
    *,
    support_id: str,
) -> None:
    if population.fit_scope != "outer_training_only":
        raise ValueError("population transfer must be fitted on outer_training_only")
    if subject.fit_scope != "support_only":
        raise ValueError("subject transfer must be fitted on support_only")
    if subject.fit_id != support_id:
        raise ValueError("rho support_id and subject transfer fit_id differ")
    if population.eeg_channel_order != subject.eeg_channel_order:
        raise ValueError("population and subject EEG channel orders differ")
    if (
        population.eog_metadata.canonical_order
        != subject.eog_metadata.canonical_order
    ):
        raise ValueError("population and subject EOG channel orders differ")
    if population.eog_metadata.polarity != subject.eog_metadata.polarity:
        raise ValueError("population and subject EOG polarity conventions differ")


def population_subject_mixing_correction(
    observed_eeg: object,
    eog: object,
    *,
    eog_input_order: Sequence[str],
    population_transfer: ArtifactTransfer,
    rho: SupportOnlyRho,
    subject_transfer_factory: SubjectTransferFactory | None,
) -> ArtifactContextCorrection:
    """Subtract a support-frozen population/subject transfer prediction.

    At ``rho=0`` this function returns before inspecting or invoking the
    subject factory.  At either pure endpoint the change from the observation
    obeys ``Q_i (x_out-y)=0``.  A mixed correction lies in the joint column
    space ``span([C0, Cs])``; the convex mixture itself is not mislabeled as a
    projector.
    """

    if not isinstance(population_transfer, ArtifactTransfer):
        raise TypeError("population_transfer must be ArtifactTransfer")
    if population_transfer.fit_scope != "outer_training_only":
        raise ValueError("population transfer must be fitted on outer_training_only")
    if not isinstance(rho, SupportOnlyRho):
        raise TypeError("rho must be SupportOnlyRho")
    observed = _fp64_matrix(observed_eeg, name="observed_eeg")
    if observed.shape[0] != len(population_transfer.eeg_channel_order):
        raise ValueError("observed_eeg does not match population EEG channels")
    population_prediction = population_transfer.predict_contamination(
        eog,
        input_order=eog_input_order,
    )
    if population_prediction.shape != observed.shape:
        raise ValueError("EOG and observed EEG sample dimensions differ")

    # This exact endpoint is deliberately before any access to the subject
    # factory, its fit metadata, or a subject-specific residual/prediction.
    if rho.value == 0.0:
        correction = -population_prediction
        restored = observed + correction
        basis = np.array(population_transfer.eeg_subspace_basis, copy=True)
        projector = np.array(population_transfer.projector, copy=True)
        return ArtifactContextCorrection(
            restored_eeg=np.asarray(restored, dtype=np.float64),
            correction=np.asarray(correction, dtype=np.float64),
            mixed_contamination=np.asarray(population_prediction, dtype=np.float64),
            population_contamination=np.asarray(
                population_prediction,
                dtype=np.float64,
            ),
            subject_contamination=None,
            union_basis=basis,
            union_projector=projector,
            rho=0.0,
            branch="population",
            subject_context_constructed=False,
        )

    if subject_transfer_factory is None or not callable(subject_transfer_factory):
        raise ValueError("non-zero rho requires a subject transfer factory")
    subject = subject_transfer_factory()
    if not isinstance(subject, ArtifactTransfer):
        raise TypeError("subject transfer factory must return ArtifactTransfer")
    _compatible_transfers(
        population_transfer,
        subject,
        support_id=rho.support_id,
    )
    subject_prediction = subject.predict_contamination(
        eog,
        input_order=eog_input_order,
    )
    if subject_prediction.shape != observed.shape:
        raise ValueError("subject EOG and observed EEG sample dimensions differ")
    mixed = (
        (1.0 - rho.value) * population_prediction
        + rho.value * subject_prediction
    )
    correction = -mixed
    restored = observed + correction
    union_basis, union_projector = _span_basis(
        population_transfer.transfer_matrix,
        subject.transfer_matrix,
    )
    branch: Literal["mixed", "subject"] = (
        "subject" if rho.value == 1.0 else "mixed"
    )
    return ArtifactContextCorrection(
        restored_eeg=np.asarray(restored, dtype=np.float64),
        correction=np.asarray(correction, dtype=np.float64),
        mixed_contamination=np.asarray(mixed, dtype=np.float64),
        population_contamination=np.asarray(population_prediction, dtype=np.float64),
        subject_contamination=np.asarray(subject_prediction, dtype=np.float64),
        union_basis=union_basis,
        union_projector=union_projector,
        rho=rho.value,
        branch=branch,
        subject_context_constructed=True,
    )


def posterior_mean_k8(
    samples: Sequence[object],
    *,
    output_rule: Literal["posterior_mean_waveform"] = "posterior_mean_waveform",
) -> np.ndarray:
    """Return the waveform mean of exactly eight posterior samples.

    The API intentionally accepts no clean target, outcome, metric, or score.
    ``best_of_k`` and every other sample-selection rule are rejected.
    """

    if output_rule != "posterior_mean_waveform":
        raise ValueError("best-of-K/sample selection is forbidden; use posterior mean")
    values = tuple(samples)
    if len(values) != POSTERIOR_SAMPLE_COUNT:
        raise ValueError(
            f"posterior mean requires exactly K={POSTERIOR_SAMPLE_COUNT} samples"
        )
    arrays = [
        _fp64_matrix(value, name=f"posterior_sample_{index}")
        for index, value in enumerate(values)
    ]
    shape = arrays[0].shape
    if any(value.shape != shape for value in arrays[1:]):
        raise ValueError("posterior samples must have identical waveform shapes")
    return np.mean(np.stack(arrays, axis=0), axis=0, dtype=np.float64)
