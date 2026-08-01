"""Construct matched POP and P0 observation states from one observed query."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from .population import PopulationObservationState


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
) -> tuple[PopulationObservationState, PopulationObservationState]:
    """Create energies differing only by the calibrated EEG-space projector.

    A common temporal reliability ``w=a`` multiplies both energies.  POP uses
    ``w I`` and the context energy uses ``w [Q + a^2 Pi]``.  Thus the same
    query, population observation weight and attenuation source are preserved;
    only the calibration-derived subspace changes the spatial precision.
    """
    if observation.ndim != 3:
        raise ValueError("observation must be (B,C,L)")
    batch, channels, _ = observation.shape
    if attenuation.shape != (batch,):
        raise ValueError(f"attenuation must have shape ({batch},)")
    if base_precision < 0.0:
        raise ValueError("base_precision must be non-negative")
    projection = torch.as_tensor(
        projector, device=observation.device, dtype=observation.dtype
    )
    if projection.shape != (channels, channels):
        raise ValueError("projector does not match EEG channels")
    identity = torch.eye(channels, device=observation.device, dtype=observation.dtype)
    population_precision = (
        float(base_precision) * attenuation.reshape(batch, 1, 1)
    )
    context_precision = float(base_precision) * attenuation.reshape(batch, 1, 1) * (
        identity.reshape(1, channels, channels)
        - (1.0 - attenuation.square()).reshape(batch, 1, 1)
        * projection.reshape(1, channels, channels)
    )
    population = PopulationObservationState(
        observation=observation,
        precision=population_precision,
        name="population_E0",
    )
    context = PopulationObservationState(
        observation=observation,
        precision=context_precision,
        name="calibrated_EC",
    )
    return population, context


def population_state_only(
    observation: Tensor,
    *,
    attenuation: Tensor,
    base_precision: float,
) -> PopulationObservationState:
    """Construct POP without accepting or constructing a calibration operator."""
    if observation.ndim != 3 or attenuation.shape != (observation.shape[0],):
        raise ValueError("observation/attenuation shape mismatch")
    precision = float(base_precision) * attenuation.reshape(-1, 1, 1)
    return PopulationObservationState(
        observation=observation,
        precision=precision,
        name="population_E0",
    )
