"""Construct matched POP and P0 observation states from one observed query."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
from torch import Tensor

from .population import PopulationObservationState


@dataclass(frozen=True)
class DatasetPopulationProjector:
    """Frozen outer-training population subspace for one dataset/montage."""

    dataset_id: str
    montage_id: str
    projector: np.ndarray | Tensor
    source: str

    def __post_init__(self) -> None:
        for field_name in ("dataset_id", "montage_id", "source"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")


@dataclass(frozen=True)
class CalibrationContextProjector:
    """Calibration-derived subspace tagged with its dataset and montage."""

    dataset_id: str
    montage_id: str
    projector: np.ndarray | Tensor
    calibration_id: str

    def __post_init__(self) -> None:
        for field_name in ("dataset_id", "montage_id", "calibration_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")


ContextStateFactory = Callable[[], PopulationObservationState]


def _validated_attenuation(observation: Tensor, attenuation: Tensor) -> Tensor:
    """Validate the legacy window-level ``(B,)`` attenuation."""
    batch = observation.shape[0]
    value = torch.as_tensor(
        attenuation,
        device=observation.device,
        dtype=observation.dtype,
    ).detach()
    if value.shape != (batch,):
        raise ValueError(f"attenuation must have shape ({batch},)")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("attenuation contains non-finite values")
    if bool(((value < 0.0) | (value > 1.0)).any()):
        raise ValueError("attenuation must lie in [0, 1]")
    return value


def _validated_frame_field(
    observation: Tensor,
    value: Tensor,
    *,
    name: str,
) -> Tensor:
    """Validate a formal per-frame field with shape ``(B,L)`` and range [0,1]."""
    batch, _, length = observation.shape
    field = torch.as_tensor(
        value,
        device=observation.device,
        dtype=observation.dtype,
    ).detach()
    if field.shape != (batch, length):
        raise ValueError(f"{name} must have shape ({batch}, {length})")
    if not bool(torch.isfinite(field).all()):
        raise ValueError(f"{name} contains non-finite values")
    if bool(((field < 0.0) | (field > 1.0)).any()):
        raise ValueError(f"{name} must lie in [0, 1]")
    return field


def _validated_base_precision(base_precision: float) -> float:
    value = float(base_precision)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("base_precision must be finite and non-negative")
    return value


def _validated_projector(
    observation: Tensor,
    projector: np.ndarray | Tensor,
    *,
    name: str,
) -> Tensor:
    channels = observation.shape[1]
    value = torch.as_tensor(
        projector,
        device=observation.device,
        dtype=observation.dtype,
    ).detach()
    if value.shape != (channels, channels):
        raise ValueError(f"{name} does not match {channels} EEG channels")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")
    if not torch.allclose(value, value.T, atol=1.0e-6, rtol=1.0e-5):
        raise ValueError(f"{name} must be symmetric")
    if not torch.allclose(value @ value, value, atol=2.0e-5, rtol=2.0e-5):
        raise ValueError(f"{name} must be an orthogonal projector")
    return value


def _validated_observation_weight(
    observation: Tensor,
    observation_weight: Tensor | float,
) -> Tensor:
    """Return the non-negative population observation weight as ``(B,1,1)``."""
    batch = observation.shape[0]
    value = torch.as_tensor(
        observation_weight,
        device=observation.device,
        dtype=observation.dtype,
    ).detach()
    if value.ndim == 0:
        value = value.expand(batch)
    if value.shape != (batch,):
        raise ValueError(f"observation_weight must be scalar or have shape ({batch},)")
    if not bool(torch.isfinite(value).all()) or bool((value < 0.0).any()):
        raise ValueError("observation_weight must be finite and non-negative")
    return value.reshape(batch, 1, 1)


def attenuation_from_external_reference(
    eog: Tensor,
    *,
    scale: float,
    floor: float,
) -> Tensor:
    """Map window-level external EOG magnitude to a precision reliability.

    EOG is standardized within each complete calibration/query context before
    this function.  The mapping follows ``a=sqrt(1/(1+kappa^2))`` and returns
    one scalar per window.  It is frozen, deterministic and uses no clean EEG.
    """
    if eog.ndim != 3 or not eog.dtype.is_floating_point:
        raise ValueError("EOG must be a floating (B,R,L) tensor")
    if not 0.0 <= floor <= 1.0 or scale <= 0.0:
        raise ValueError("invalid attenuation mapping")
    magnitude = torch.sqrt(torch.mean(eog.square(), dim=(1, 2)))
    kappa = magnitude / float(scale)
    attenuation = torch.rsqrt(1.0 + kappa.square())
    return attenuation.clamp(min=float(floor), max=1.0)


def frame_attenuation_from_external_reference(
    eog: Tensor,
    *,
    scale: float,
    floor: float,
) -> Tensor:
    """Map external-reference magnitude to frozen per-frame ``(B,L)`` attenuation."""
    if eog.ndim != 3 or not eog.dtype.is_floating_point:
        raise ValueError("EOG must be a floating (B,R,L) tensor")
    if not 0.0 <= floor <= 1.0 or not math.isfinite(float(scale)) or scale <= 0.0:
        raise ValueError("invalid attenuation mapping")
    magnitude = torch.sqrt(torch.mean(eog.square(), dim=1))
    kappa = magnitude / float(scale)
    attenuation = torch.rsqrt(1.0 + kappa.square())
    return attenuation.clamp(min=float(floor), max=1.0)


def _frame_precision(
    observation: Tensor,
    *,
    attenuation: Tensor,
    valid_weight: Tensor,
    projector: np.ndarray | Tensor,
    base_precision: float,
    projector_name: str,
) -> tuple[Tensor, Tensor]:
    """Construct ``base*v*[I-(1-a^2)Pi]`` and its canonical valid mask."""
    if observation.ndim != 3:
        raise ValueError("observation must be (B,C,L)")
    attenuation_value = _validated_frame_field(
        observation, attenuation, name="attenuation"
    )
    weight_value = _validated_frame_field(
        observation, valid_weight, name="valid_weight"
    )
    projection = _validated_projector(
        observation, projector, name=projector_name
    )
    base = _validated_base_precision(base_precision)
    batch, channels, length = observation.shape
    identity = torch.eye(
        channels, device=observation.device, dtype=observation.dtype
    ).reshape(1, 1, channels, channels)
    spatial = identity - (
        1.0 - attenuation_value.square()
    ).reshape(batch, length, 1, 1) * projection.reshape(
        1, 1, channels, channels
    )
    precision = base * weight_value.reshape(batch, length, 1, 1) * spatial
    return precision, weight_value > 0.0


def dataset_population_state(
    observation: Tensor,
    *,
    attenuation: Tensor,
    valid_weight: Tensor,
    population_projector: DatasetPopulationProjector,
    base_precision: float,
    energy_scale: float = 1.0,
) -> PopulationObservationState:
    """Construct formal dataset-specific ``E0`` using mandatory ``Pi0``.

    Unlike the legacy isotropic ablation, this constructor has no default for
    ``population_projector``.  The per-frame attenuation source is therefore
    shared by POP and calibrated inference without consulting calibration.
    """
    if not isinstance(population_projector, DatasetPopulationProjector):
        raise TypeError("population_projector must be DatasetPopulationProjector")
    precision, valid_time_mask = _frame_precision(
        observation,
        attenuation=attenuation,
        valid_weight=valid_weight,
        projector=population_projector.projector,
        base_precision=base_precision,
        projector_name="dataset population projector Pi0",
    )
    return PopulationObservationState(
        observation=observation,
        precision=precision,
        energy_scale=energy_scale,
        name="population_E0",
        valid_time_mask=valid_time_mask,
        dataset_id=population_projector.dataset_id,
        montage_id=population_projector.montage_id,
        precision_semantics="dataset_population_and_context_precision",
    )


def dataset_population_and_context_states(
    observation: Tensor,
    *,
    attenuation: Tensor,
    valid_weight: Tensor,
    population_projector: DatasetPopulationProjector,
    context_projector: CalibrationContextProjector,
    base_precision: float,
    energy_scale: float = 1.0,
) -> tuple[PopulationObservationState, PopulationObservationState]:
    """Construct matched formal ``W0`` and ``WC`` from ``Pi0`` and ``PiC``."""
    if not isinstance(population_projector, DatasetPopulationProjector):
        raise TypeError("population_projector must be DatasetPopulationProjector")
    if not isinstance(context_projector, CalibrationContextProjector):
        raise TypeError("context_projector must be CalibrationContextProjector")
    if population_projector.dataset_id != context_projector.dataset_id:
        raise ValueError("Pi0 and PiC dataset IDs differ")
    if population_projector.montage_id != context_projector.montage_id:
        raise ValueError("Pi0 and PiC montage IDs differ")
    population = dataset_population_state(
        observation,
        attenuation=attenuation,
        valid_weight=valid_weight,
        population_projector=population_projector,
        base_precision=base_precision,
        energy_scale=energy_scale,
    )
    context_precision, valid_time_mask = _frame_precision(
        observation,
        attenuation=attenuation,
        valid_weight=valid_weight,
        projector=context_projector.projector,
        base_precision=base_precision,
        projector_name="calibration context projector PiC",
    )
    context = PopulationObservationState(
        observation=observation,
        precision=context_precision,
        energy_scale=energy_scale,
        name="calibrated_EC",
        valid_time_mask=valid_time_mask,
        dataset_id=context_projector.dataset_id,
        montage_id=context_projector.montage_id,
        precision_semantics="dataset_population_and_context_precision",
    )
    return population, context


def rho_interpolated_precision_state(
    population_state: PopulationObservationState,
    *,
    rho: float,
    calibration_accepted: bool,
    context_state_factory: Optional[ContextStateFactory],
) -> PopulationObservationState:
    """Lazily construct ``W_rho=(1-rho)W0+rho WC``.

    The POP branch returns the already constructed population state before a
    context factory is inspected or invoked.  Consequently ``rho=0`` and a
    rejected calibration cannot construct a context operator, precision, or
    residual; shared per-frame population attenuation remains available in
    ``population_state``.
    """
    rho_value = float(rho)
    if not math.isfinite(rho_value) or not 0.0 <= rho_value <= 1.0:
        raise ValueError("rho must be finite and lie in [0, 1]")
    if rho_value == 0.0 or not bool(calibration_accepted):
        return population_state
    if context_state_factory is None:
        raise ValueError("accepted non-zero rho requires a context state factory")
    context_state = context_state_factory()
    if not isinstance(context_state, PopulationObservationState):
        raise TypeError("context factory must return PopulationObservationState")
    if (
        context_state.observation is not population_state.observation
        and not torch.equal(context_state.observation, population_state.observation)
    ):
        raise ValueError("W0 and WC must use the same observed query")
    for field_name in ("dataset_id", "montage_id", "precision_semantics"):
        if getattr(context_state, field_name) != getattr(population_state, field_name):
            raise ValueError(f"W0 and WC {field_name} differ")
    if context_state.energy_scale != population_state.energy_scale:
        raise ValueError("W0 and WC energy scales differ")
    if not torch.equal(
        context_state.valid_time_mask, population_state.valid_time_mask
    ):
        raise ValueError("W0 and WC valid-time masks differ")
    if context_state.precision.shape != population_state.precision.shape:
        raise ValueError("W0 and WC precision shapes differ")
    precision = (
        (1.0 - rho_value) * population_state.precision
        + rho_value * context_state.precision
    )
    return PopulationObservationState(
        observation=population_state.observation,
        precision=precision,
        energy_scale=population_state.energy_scale,
        name=f"rho_interpolated_W_{rho_value:g}",
        valid_time_mask=population_state.valid_time_mask,
        dataset_id=population_state.dataset_id,
        montage_id=population_state.montage_id,
        precision_semantics=population_state.precision_semantics,
    )


def matched_population_and_context_states(
    observation: Tensor,
    *,
    attenuation: Tensor,
    projector: np.ndarray | Tensor,
    base_precision: float,
    observation_weight: Tensor | float = 1.0,
    energy_scale: float = 1.0,
    valid_time_mask: Optional[Tensor] = None,
) -> tuple[PopulationObservationState, PopulationObservationState]:
    """Create the backward-compatible ``legacy_isotropic_ablation`` pair.

    ``observation_weight`` is the common population observation weight ``w``.
    ``energy_scale`` is passed to both energy states without branch-specific
    rescaling.
    POP uses ``base*w*I`` and the context energy uses
    ``base*w*[Q + a^2 Pi]``.  The external attenuation source is therefore
    received and validated by both paths, but it can only act on a calibrated
    subspace in the context path.  In particular, ``a=0`` removes precision
    only in ``span(Pi)`` and leaves the orthogonal complement finite.
    """
    if observation.ndim != 3:
        raise ValueError("observation must be (B,C,L)")
    batch, channels, _ = observation.shape
    attenuation = _validated_attenuation(observation, attenuation)
    weight = _validated_observation_weight(observation, observation_weight)
    base = _validated_base_precision(base_precision)
    projection = _validated_projector(
        observation,
        projector,
        name="legacy calibration projector",
    )
    identity = torch.eye(channels, device=observation.device, dtype=observation.dtype)
    population_precision = base * weight * identity.reshape(
        1, channels, channels
    )
    context_precision = base * weight * (
        identity.reshape(1, channels, channels)
        - (1.0 - attenuation.square()).reshape(batch, 1, 1)
        * projection.reshape(1, channels, channels)
    )
    population = PopulationObservationState(
        observation=observation,
        precision=population_precision,
        energy_scale=energy_scale,
        name="population_E0_legacy_isotropic_ablation",
        valid_time_mask=valid_time_mask,
        dataset_id="legacy_unspecified_dataset",
        montage_id="legacy_unspecified_montage",
        precision_semantics="legacy_isotropic_ablation",
    )
    context = PopulationObservationState(
        observation=observation,
        precision=context_precision,
        energy_scale=energy_scale,
        name="calibrated_EC_legacy_isotropic_ablation",
        valid_time_mask=population.valid_time_mask,
        dataset_id=population.dataset_id,
        montage_id=population.montage_id,
        precision_semantics=population.precision_semantics,
    )
    return population, context


def population_state_only(
    observation: Tensor,
    *,
    attenuation: Tensor,
    base_precision: float,
    observation_weight: Tensor | float = 1.0,
    energy_scale: float = 1.0,
    valid_time_mask: Optional[Tensor] = None,
) -> PopulationObservationState:
    """Construct the explicitly named legacy isotropic POP ablation.

    ``attenuation`` is deliberately received and validated so POP and P0 use
    the same legal attenuation source.  With no calibration-derived subspace,
    it does not alter the frozen isotropic population observation energy.
    """
    if observation.ndim != 3:
        raise ValueError("observation must be (B,C,L)")
    _validated_attenuation(observation, attenuation)
    weight = _validated_observation_weight(observation, observation_weight)
    base = _validated_base_precision(base_precision)
    channels = observation.shape[1]
    identity = torch.eye(channels, device=observation.device, dtype=observation.dtype)
    precision = base * weight * identity.reshape(
        1, channels, channels
    )
    return PopulationObservationState(
        observation=observation,
        precision=precision,
        energy_scale=energy_scale,
        name="population_E0_legacy_isotropic_ablation",
        valid_time_mask=valid_time_mask,
        dataset_id="legacy_unspecified_dataset",
        montage_id="legacy_unspecified_montage",
        precision_semantics="legacy_isotropic_ablation",
    )
