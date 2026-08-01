"""Deferred B6 population shrinkage for calibration projectors.

This module deliberately operates only on two already fitted EEG-space
projectors.  It cannot inspect an observed query, a clean target, EOG, events,
or any other evaluation-time field.  B6 remains a development-only diagnostic
until a separate protocol explicitly enables it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


__all__ = [
    "ProjectorCompatibilityKey",
    "PopShrinkOutcome",
    "spectral_projector_shrink",
]


@dataclass(frozen=True)
class ProjectorCompatibilityKey:
    """Acquisition identity required before two projectors may be combined."""

    dataset_id: str
    montage_id: str
    reference_id: str
    preprocessing_id: str
    channel_order: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in (
            "dataset_id",
            "montage_id",
            "reference_id",
            "preprocessing_id",
        ):
            _nonempty(getattr(self, field_name), name=field_name)
        if not isinstance(self.channel_order, tuple) or not self.channel_order:
            raise ValueError("channel_order must be a non-empty tuple")
        if any(
            not isinstance(channel, str) or not channel.strip()
            for channel in self.channel_order
        ):
            raise ValueError("channel_order entries must be non-empty strings")
        if len(set(self.channel_order)) != len(self.channel_order):
            raise ValueError("channel_order entries must be unique")


@dataclass(frozen=True)
class PopShrinkOutcome:
    """Result of one frozen spectral-projector shrinkage decision.

    A fallback contains no context projector.  This makes it impossible for a
    caller to accidentally run a context path after an invalid P0 calibration;
    the caller must invoke POP instead.
    """

    status: Literal["eligible", "fallback_POP"]
    projector: np.ndarray | None
    reasons: tuple[str, ...]
    diagnostics: dict[str, float | int | str]
    fallback: Literal["POP"] = "POP"
    context_projector_constructed: bool = False


def _positive_rank(rank: int) -> int:
    if isinstance(rank, bool) or int(rank) != rank or int(rank) < 1:
        raise ValueError("rank must be a positive integer")
    return int(rank)


def _unit_interval(value: float, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1]")
    return result


def _nonnegative_finite(value: float, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _nonempty(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _projector_problem(
    projector: object,
    *,
    name: str,
    expected_shape: tuple[int, int] | None,
    rank: int,
    symmetry_tolerance: float,
    idempotence_tolerance: float,
) -> tuple[np.ndarray | None, str | None]:
    """Validate without repairing, symmetrizing, or changing projector rank."""

    try:
        value = np.asarray(projector)
    except (TypeError, ValueError):
        return None, f"{name}_unreadable"
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        return None, f"{name}_shape"
    if expected_shape is not None and value.shape != expected_shape:
        return None, f"{name}_shape"
    if rank > value.shape[0]:
        return None, f"{name}_rank"
    if not np.issubdtype(value.dtype, np.number) or np.issubdtype(
        value.dtype, np.complexfloating
    ):
        return None, f"{name}_dtype"
    numeric = np.asarray(value, dtype=np.float64)
    if not np.isfinite(numeric).all():
        return None, f"{name}_nonfinite"
    if not np.allclose(
        numeric,
        numeric.T,
        rtol=0.0,
        atol=symmetry_tolerance,
    ):
        return None, f"{name}_not_symmetric"
    if not np.allclose(
        numeric @ numeric,
        numeric,
        rtol=0.0,
        atol=idempotence_tolerance,
    ):
        return None, f"{name}_not_idempotent"
    numerical_rank = int(np.linalg.matrix_rank(numeric, tol=idempotence_tolerance))
    if numerical_rank != rank:
        return None, f"{name}_rank"
    return value, None


def _fallback(
    reason: str,
    *,
    gamma: float,
    rank: int,
    extra_diagnostics: dict[str, float | int | str] | None = None,
) -> PopShrinkOutcome:
    diagnostics: dict[str, float | int | str] = {
        "operator": "B6_POP_SHRINK",
        "gamma": gamma,
        "rank": rank,
        "fallback": "POP",
    }
    if extra_diagnostics:
        diagnostics.update(extra_diagnostics)
    return PopShrinkOutcome(
        status="fallback_POP",
        projector=None,
        reasons=(reason,),
        diagnostics=diagnostics,
        context_projector_constructed=False,
    )


def _endpoint_diagnostics(
    *,
    gamma: float,
    rank: int,
    channels: int,
) -> dict[str, float | int | str]:
    return {
        "operator": "B6_POP_SHRINK",
        "construction": "rank_r_spectral_projector",
        "gamma": gamma,
        "rank": rank,
        "channels": channels,
        "spectral_eigengap": 1.0,
        "endpoint": "Pi0" if gamma == 0.0 else "PiC",
    }


def spectral_projector_shrink(
    population_projector: np.ndarray,
    context_projector: np.ndarray | None,
    *,
    rank: int,
    gamma: float,
    context_eligible: bool,
    population_compatibility: ProjectorCompatibilityKey,
    population_fit_scope: Literal["outer_training_only"],
    context_compatibility: ProjectorCompatibilityKey | None,
    context_fit_scope: Literal["support_only"] | None,
    minimum_eigengap: float = 1.0e-6,
    symmetry_tolerance: float = 1.0e-10,
    idempotence_tolerance: float = 1.0e-8,
) -> PopShrinkOutcome:
    """Construct the deferred B6 rank-``r`` spectral projector.

    For a development-frozen ``gamma``, the pre-projection matrix is

    ``S = (1-gamma) Pi0 + gamma PiC``.

    The returned projector is formed from the top ``rank`` eigenvectors of
    ``S``.  The raw convex combination is never mislabeled as a projector.
    ``gamma=0`` and ``gamma=1`` return exact copies of their respective input
    projectors, rather than numerically reconstructing an endpoint.

    Population-projector corruption is fatal because POP would itself be
    unavailable.  Every context/P0 problem instead returns ``fallback_POP``
    with ``projector=None`` and does not construct a context projector.
    """

    retained_rank = _positive_rank(rank)
    shrink_weight = _unit_interval(gamma, name="gamma")
    eigengap_floor = _nonnegative_finite(
        minimum_eigengap, name="minimum_eigengap"
    )
    symmetry_atol = _nonnegative_finite(
        symmetry_tolerance, name="symmetry_tolerance"
    )
    idempotence_atol = _nonnegative_finite(
        idempotence_tolerance, name="idempotence_tolerance"
    )
    if not isinstance(population_compatibility, ProjectorCompatibilityKey):
        raise TypeError("population_compatibility must be ProjectorCompatibilityKey")
    if population_fit_scope != "outer_training_only":
        raise ValueError("population projector must be fitted on outer_training_only")

    population_value, population_problem = _projector_problem(
        population_projector,
        name="population_projector",
        expected_shape=None,
        rank=retained_rank,
        symmetry_tolerance=symmetry_atol,
        idempotence_tolerance=idempotence_atol,
    )
    if population_problem is not None or population_value is None:
        raise ValueError(
            "POP population projector is invalid: "
            f"{population_problem or 'unknown_population_projector_error'}"
        )
    if len(population_compatibility.channel_order) != population_value.shape[0]:
        raise ValueError(
            "population compatibility channel_order does not match projector"
        )

    # gamma=0 is the exact POP endpoint. It must not inspect, validate, or
    # construct anything derived from calibration/context data.
    if shrink_weight == 0.0:
        return PopShrinkOutcome(
            status="eligible",
            projector=np.array(population_value, copy=True),
            reasons=(),
            diagnostics=_endpoint_diagnostics(
                gamma=shrink_weight,
                rank=retained_rank,
                channels=population_value.shape[0],
            ),
            context_projector_constructed=False,
        )

    # This branch deliberately precedes every access to context_projector or
    # its metadata.  An ineligible P0 therefore cannot construct a context.
    if not bool(context_eligible):
        return _fallback(
            "context_ineligible", gamma=shrink_weight, rank=retained_rank
        )
    if context_projector is None:
        return _fallback(
            "missing_context_projector", gamma=shrink_weight, rank=retained_rank
        )
    if context_fit_scope != "support_only":
        return _fallback(
            "context_fit_scope", gamma=shrink_weight, rank=retained_rank
        )
    if not isinstance(context_compatibility, ProjectorCompatibilityKey):
        return _fallback(
            "missing_context_compatibility",
            gamma=shrink_weight,
            rank=retained_rank,
        )
    if context_compatibility != population_compatibility:
        return _fallback(
            "compatibility_mismatch", gamma=shrink_weight, rank=retained_rank
        )

    context_value, context_problem = _projector_problem(
        context_projector,
        name="context_projector",
        expected_shape=population_value.shape,
        rank=retained_rank,
        symmetry_tolerance=symmetry_atol,
        idempotence_tolerance=idempotence_atol,
    )
    if context_problem is not None or context_value is None:
        return _fallback(
            context_problem or "invalid_context_projector",
            gamma=shrink_weight,
            rank=retained_rank,
        )

    if shrink_weight == 1.0:
        return PopShrinkOutcome(
            status="eligible",
            projector=np.array(context_value, copy=True),
            reasons=(),
            diagnostics=_endpoint_diagnostics(
                gamma=shrink_weight,
                rank=retained_rank,
                channels=context_value.shape[0],
            ),
            context_projector_constructed=True,
        )

    population_numeric = np.asarray(population_value, dtype=np.float64)
    context_numeric = np.asarray(context_value, dtype=np.float64)
    mixture = (
        (1.0 - shrink_weight) * population_numeric
        + shrink_weight * context_numeric
    )
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(mixture)
    except np.linalg.LinAlgError:
        return _fallback(
            "spectral_eigendecomposition",
            gamma=shrink_weight,
            rank=retained_rank,
        )
    descending = np.argsort(eigenvalues, kind="stable")[::-1]
    ordered_eigenvalues = eigenvalues[descending]
    if retained_rank < ordered_eigenvalues.size:
        eigengap = float(
            ordered_eigenvalues[retained_rank - 1]
            - ordered_eigenvalues[retained_rank]
        )
    else:
        eigengap = float("inf")
    if not np.isfinite(eigengap) and retained_rank < ordered_eigenvalues.size:
        return _fallback(
            "spectral_eigengap_nonfinite",
            gamma=shrink_weight,
            rank=retained_rank,
        )
    if eigengap < eigengap_floor:
        return _fallback(
            "spectral_eigengap",
            gamma=shrink_weight,
            rank=retained_rank,
            extra_diagnostics={
                "spectral_eigengap": eigengap,
                "minimum_spectral_eigengap": eigengap_floor,
                "largest_eigenvalue": float(ordered_eigenvalues[0]),
                "retained_eigenvalue": float(
                    ordered_eigenvalues[retained_rank - 1]
                ),
                "first_discarded_eigenvalue": float(
                    ordered_eigenvalues[retained_rank]
                ),
            },
        )

    basis = eigenvectors[:, descending[:retained_rank]]
    shrunken = basis @ basis.T
    shrunken_problem = _projector_problem(
        shrunken,
        name="shrunken_projector",
        expected_shape=population_value.shape,
        rank=retained_rank,
        symmetry_tolerance=symmetry_atol,
        idempotence_tolerance=idempotence_atol,
    )[1]
    if shrunken_problem is not None:
        return _fallback(
            shrunken_problem,
            gamma=shrink_weight,
            rank=retained_rank,
        )
    diagnostics: dict[str, float | int | str] = {
        "operator": "B6_POP_SHRINK",
        "construction": "rank_r_spectral_projector",
        "gamma": shrink_weight,
        "rank": retained_rank,
        "channels": int(shrunken.shape[0]),
        "spectral_eigengap": eigengap,
        "largest_eigenvalue": float(ordered_eigenvalues[0]),
        "retained_eigenvalue": float(ordered_eigenvalues[retained_rank - 1]),
        "first_discarded_eigenvalue": (
            float(ordered_eigenvalues[retained_rank])
            if retained_rank < ordered_eigenvalues.size
            else 0.0
        ),
        "distance_to_population": float(
            np.linalg.norm(shrunken - population_numeric, ord="fro")
        ),
        "distance_to_context": float(
            np.linalg.norm(shrunken - context_numeric, ord="fro")
        ),
    }
    return PopShrinkOutcome(
        status="eligible",
        projector=shrunken,
        reasons=(),
        diagnostics=diagnostics,
        context_projector_constructed=True,
    )
