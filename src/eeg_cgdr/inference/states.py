"""Construct matched POP and P0 observation states from one observed query."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from .population import PopulationObservationState


def _validated_attenuation(observation: Tensor, attenuation: Tensor) -> Tensor:
    """Return a frozen batch attenuation with the registered precision semantics."""
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


def matched_population_and_context_states(
    observation: Tensor,
    *,
    attenuation: Tensor,
    projector: np.ndarray | Tensor,
    base_precision: float,
    observation_weight: Tensor | float = 1.0,
    guidance_scale: float = 1.0,
) -> tuple[PopulationObservationState, PopulationObservationState]:
    """Create energies differing only by the calibrated EEG-space projector.

    ``observation_weight`` is the common population observation weight ``w``.
    ``guidance_scale`` is passed to both energy states without branch-specific
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
    if not np.isfinite(base_precision) or base_precision < 0.0:
        raise ValueError("base_precision must be finite and non-negative")
    projection = torch.as_tensor(
        projector, device=observation.device, dtype=observation.dtype
    )
    if projection.shape != (channels, channels):
        raise ValueError("projector does not match EEG channels")
    identity = torch.eye(channels, device=observation.device, dtype=observation.dtype)
    population_precision = float(base_precision) * weight * identity.reshape(
        1, channels, channels
    )
    context_precision = float(base_precision) * weight * (
        identity.reshape(1, channels, channels)
        - (1.0 - attenuation.square()).reshape(batch, 1, 1)
        * projection.reshape(1, channels, channels)
    )
    population = PopulationObservationState(
        observation=observation,
        precision=population_precision,
        scale=guidance_scale,
        name="population_E0",
    )
    context = PopulationObservationState(
        observation=observation,
        precision=context_precision,
        scale=guidance_scale,
        name="calibrated_EC",
    )
    return population, context


def population_state_only(
    observation: Tensor,
    *,
    attenuation: Tensor,
    base_precision: float,
    observation_weight: Tensor | float = 1.0,
    guidance_scale: float = 1.0,
) -> PopulationObservationState:
    """Construct isotropic POP without accepting a calibration operator.

    ``attenuation`` is deliberately received and validated so POP and P0 use
    the same legal attenuation source.  With no calibration-derived subspace,
    it does not alter the frozen isotropic population observation energy.
    """
    if observation.ndim != 3:
        raise ValueError("observation must be (B,C,L)")
    _validated_attenuation(observation, attenuation)
    weight = _validated_observation_weight(observation, observation_weight)
    if not np.isfinite(base_precision) or base_precision < 0.0:
        raise ValueError("base_precision must be finite and non-negative")
    channels = observation.shape[1]
    identity = torch.eye(channels, device=observation.device, dtype=observation.dtype)
    precision = float(base_precision) * weight * identity.reshape(
        1, channels, channels
    )
    return PopulationObservationState(
        observation=observation,
        precision=precision,
        scale=guidance_scale,
        name="population_E0",
    )
