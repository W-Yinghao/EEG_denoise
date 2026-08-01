"""Explicit population-observation inference for CGDR.

The clean prior never consumes the query observation.  Instead, an explicit
quadratic observation state contributes a VJP to the prior score.  Optional
calibration uses ``E_rho = E_0 + rho * (E_C - E_0)`` and is reached only after
the POP short circuit has been ruled out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

import torch
from torch import Tensor

from ..models.clean_prior import CleanEEGDiffusionPrior

PrecisionKind = Literal[
    "scalar",
    "channel_diagonal",
    "elementwise",
    "matrix",
    "batch_matrix",
    "time_matrix",
    "batch_time_matrix",
]
EnergyFunction = Callable[[Tensor], Tensor]
ObservationStateFactory = Callable[[], "PopulationObservationState"]


@dataclass(frozen=True)
class PopulationObservationState:
    """One explicit quadratic observation energy.

    ``observation`` has shape ``(B,C,L)``.  ``precision`` may be a non-negative
    scalar, channel diagonal ``(C,)``, any non-negative tensor broadcastable to
    ``(B,C,L)``, a channel matrix ``(C,C)``, batched channel matrices
    ``(B,C,C)``, time-varying matrices ``(L,C,C)``, or fully batched
    time-varying matrices ``(B,L,C,C)``.  The latter two forms support P0
    projector precision modulated by a time-varying attenuation.

    The population state represents ``E_0``.  A lazily constructed context
    state of the same type may represent ``E_C``; both must reference the same
    query observation.
    """

    observation: Tensor
    precision: Tensor | float
    scale: float = 1.0
    name: str = "population"
    _precision_kind: PrecisionKind = field(init=False, repr=False)

    def __post_init__(self) -> None:
        observation = self.observation
        if observation.ndim != 3 or not observation.dtype.is_floating_point:
            raise ValueError("observation must be a floating (B,C,L) tensor")
        if any(size < 1 for size in observation.shape):
            raise ValueError("observation cannot contain an empty dimension")
        if not bool(torch.isfinite(observation).all()):
            raise ValueError("observation contains non-finite values")
        if not math.isfinite(float(self.scale)) or float(self.scale) < 0.0:
            raise ValueError("observation energy scale must be finite and non-negative")

        precision = torch.as_tensor(
            self.precision,
            device=observation.device,
            dtype=observation.dtype,
        ).detach()
        batch, channels, length = observation.shape
        if precision.ndim == 0:
            kind: PrecisionKind = "scalar"
        elif precision.ndim == 1 and precision.shape == (channels,):
            kind = "channel_diagonal"
        elif precision.ndim == 2 and precision.shape == (channels, channels):
            kind = "matrix"
        elif precision.ndim == 3 and precision.shape == (batch, channels, channels):
            kind = "batch_matrix"
        elif precision.ndim == 3 and precision.shape == (length, channels, channels):
            kind = "time_matrix"
        elif precision.ndim == 4 and precision.shape == (
            batch,
            length,
            channels,
            channels,
        ):
            kind = "batch_time_matrix"
        else:
            try:
                broadcast_shape = torch.broadcast_shapes(
                    tuple(precision.shape), tuple(observation.shape)
                )
            except RuntimeError as exc:
                raise ValueError(
                    f"precision shape {tuple(precision.shape)} is not valid for "
                    f"observation {tuple(observation.shape)}"
                ) from exc
            if broadcast_shape != tuple(observation.shape):
                raise ValueError(
                    "elementwise precision must broadcast exactly to observation shape"
                )
            kind = "elementwise"

        if not bool(torch.isfinite(precision).all()):
            raise ValueError("precision contains non-finite values")
        if kind in ("scalar", "channel_diagonal", "elementwise"):
            if bool((precision < 0).any()):
                raise ValueError("diagonal precision must be non-negative")
        else:
            transpose = precision.transpose(-1, -2)
            if not torch.allclose(precision, transpose, atol=1.0e-6, rtol=1.0e-5):
                raise ValueError("channel precision must be symmetric")
            eigenvalues = torch.linalg.eigvalsh(precision.float())
            tolerance = 1.0e-5 * max(1.0, float(precision.abs().max()))
            if float(eigenvalues.min()) < -tolerance:
                raise ValueError("channel precision must be positive semidefinite")

        object.__setattr__(self, "precision", precision)
        object.__setattr__(self, "_precision_kind", kind)

    def energy_per_sample(self, clean_estimate: Tensor) -> Tensor:
        """Evaluate the quadratic energy and return one value per batch item."""

        if clean_estimate.shape != self.observation.shape:
            raise ValueError("clean estimate and observation must have identical shapes")
        if (
            clean_estimate.device != self.observation.device
            or clean_estimate.dtype != self.observation.dtype
        ):
            raise ValueError("clean estimate and observation must share device and dtype")
        residual = clean_estimate - self.observation
        precision = self.precision
        if self._precision_kind == "matrix":
            quadratic = torch.einsum("bcl,cd,bdl->b", residual, precision, residual)
        elif self._precision_kind == "batch_matrix":
            quadratic = torch.einsum("bcl,bcd,bdl->b", residual, precision, residual)
        elif self._precision_kind == "time_matrix":
            quadratic = torch.einsum("bcl,lcd,bdl->b", residual, precision, residual)
        elif self._precision_kind == "batch_time_matrix":
            quadratic = torch.einsum("bcl,blcd,bdl->b", residual, precision, residual)
        else:
            if self._precision_kind == "channel_diagonal":
                precision = precision.reshape(1, residual.shape[1], 1)
            quadratic = (residual.square() * precision).flatten(start_dim=1).sum(dim=1)
        return 0.5 * float(self.scale) * quadratic

    def energy(self, clean_estimate: Tensor) -> Tensor:
        """Return the batch-summed scalar energy used for the VJP."""

        return self.energy_per_sample(clean_estimate).sum()


class PopulationOnlyInference:
    """Observation-conditioned POP sampler with an explicit clean prior and ``E_0``."""

    def __init__(self, prior: CleanEEGDiffusionPrior) -> None:
        self.prior = prior

    @staticmethod
    def _validate_steps(prior: CleanEEGDiffusionPrior, ddim_steps: int) -> None:
        steps = int(ddim_steps)
        if steps != ddim_steps or not 1 <= steps <= prior.diffusion.num_timesteps - 1:
            raise ValueError(
                "ddim_steps must lie in [1, num_timesteps-1] to avoid duplicate timesteps"
            )

    def make_initial_noise(self, state: PopulationObservationState, *, seed: int) -> Tensor:
        """Create explicit, local-RNG initial noise for paired POP/CGDR calls."""

        seed_value = int(seed)
        if seed_value != seed or seed_value < 0:
            raise ValueError("seed must be non-negative")
        generator = torch.Generator(device=state.observation.device)
        generator.manual_seed(seed_value)
        return torch.randn(
            state.observation.shape,
            device=state.observation.device,
            dtype=state.observation.dtype,
            generator=generator,
        )

    def _resolve_initial_noise(
        self,
        state: PopulationObservationState,
        *,
        seed: Optional[int],
        initial_noise: Optional[Tensor],
    ) -> Tensor:
        if initial_noise is None:
            if seed is None:
                raise ValueError("provide seed or explicit initial_noise")
            return self.make_initial_noise(state, seed=seed)
        if (
            initial_noise.shape != state.observation.shape
            or initial_noise.device != state.observation.device
            or initial_noise.dtype != state.observation.dtype
        ):
            raise ValueError("initial_noise must match the observation shape, device and dtype")
        if not bool(torch.isfinite(initial_noise).all()):
            raise ValueError("initial_noise contains non-finite values")
        return initial_noise.detach()

    def _guided_noise_function(self, energy: EnergyFunction) -> Callable[[Tensor, Tensor], Tensor]:
        prior = self.prior

        def guided_noise(x_t: Tensor, timesteps: Tensor) -> Tensor:
            # GaussianDiffusion.ddim_sample_loop is no-grad.  The nested context
            # deliberately re-enables only the input VJP needed by the explicit
            # observation energy.
            with torch.enable_grad():
                differentiable_x = x_t.detach().requires_grad_(True)
                prior_noise = prior.predict_noise(differentiable_x, timesteps)
                clean_estimate = prior.predict_clean(
                    differentiable_x, timesteps, prior_noise
                )
                value = energy(clean_estimate).sum()
                gradient = torch.autograd.grad(
                    value,
                    differentiable_x,
                    create_graph=False,
                    retain_graph=False,
                    only_inputs=True,
                )[0]
                sigma = prior.noise_standard_deviation(timesteps, x_t.ndim)
                return (prior_noise + sigma * gradient).detach()

        return guided_noise

    def _sample_energy(
        self,
        state: PopulationObservationState,
        energy: EnergyFunction,
        *,
        seed: Optional[int],
        initial_noise: Optional[Tensor],
        ddim_steps: int,
    ) -> Tensor:
        self._validate_steps(self.prior, ddim_steps)
        x_t = self._resolve_initial_noise(
            state,
            seed=seed,
            initial_noise=initial_noise,
        )
        self.prior.eval()
        return self.prior.diffusion.ddim_sample_loop(
            self._guided_noise_function(energy),
            tuple(state.observation.shape),
            state.observation.device,
            ddim_steps=int(ddim_steps),
            eta=0.0,
            x_t=x_t,
            clip_denoised=False,
        )

    def sample(
        self,
        population_state: PopulationObservationState,
        *,
        seed: Optional[int] = None,
        initial_noise: Optional[Tensor] = None,
        ddim_steps: int = 50,
    ) -> Tensor:
        """Sample the explicit population posterior using only ``E_0``."""

        return self._sample_energy(
            population_state,
            population_state.energy_per_sample,
            seed=seed,
            initial_noise=initial_noise,
            ddim_steps=ddim_steps,
        )

    @staticmethod
    def _same_query(
        population_state: PopulationObservationState,
        context_state: PopulationObservationState,
    ) -> None:
        population_query = population_state.observation
        context_query = context_state.observation
        if context_query is population_query:
            return
        if (
            context_query.shape != population_query.shape
            or context_query.device != population_query.device
            or context_query.dtype != population_query.dtype
            or not torch.equal(context_query, population_query)
        ):
            raise ValueError("E_0 and E_C must use the same query observation")

    def sample_cgdr(
        self,
        population_state: PopulationObservationState,
        *,
        rho: float,
        calibration_accepted: bool,
        context_state_factory: Optional[ObservationStateFactory],
        seed: Optional[int] = None,
        initial_noise: Optional[Tensor] = None,
        ddim_steps: int = 50,
    ) -> Tensor:
        """Sample CGDR or directly short-circuit to POP.

        The first branch is intentionally before context-state construction,
        context VJP construction and local RNG use.  A rejected calibration or
        ``rho=0`` therefore calls the exact POP method and cannot touch an
        individualized precision/operator factory.
        """

        rho_value = float(rho)
        if not math.isfinite(rho_value) or not 0.0 <= rho_value <= 1.0:
            raise ValueError("rho must be finite and lie in [0,1]")
        if rho_value == 0.0 or not bool(calibration_accepted):
            return self.sample(
                population_state,
                seed=seed,
                initial_noise=initial_noise,
                ddim_steps=ddim_steps,
            )

        if context_state_factory is None:
            raise ValueError("accepted non-zero calibration requires a lazy context state factory")
        context_state = context_state_factory()
        if not isinstance(context_state, PopulationObservationState):
            raise TypeError("context_state_factory must return PopulationObservationState")
        self._same_query(population_state, context_state)

        def interpolated_energy(clean_estimate: Tensor) -> Tensor:
            population_energy = population_state.energy_per_sample(clean_estimate)
            context_energy = context_state.energy_per_sample(clean_estimate)
            return population_energy + rho_value * (context_energy - population_energy)

        return self._sample_energy(
            population_state,
            interpolated_energy,
            seed=seed,
            initial_noise=initial_noise,
            ddim_steps=ddim_steps,
        )

    def sample_with_calibration(
        self,
        population_state: PopulationObservationState,
        *,
        rho: float,
        calibration_accepted: bool,
        context_state_factory: Optional[ObservationStateFactory],
        seed: Optional[int] = None,
        initial_noise: Optional[Tensor] = None,
        ddim_steps: int = 50,
    ) -> Tensor:
        """Readable alias for :meth:`sample_cgdr`."""

        return self.sample_cgdr(
            population_state,
            rho=rho,
            calibration_accepted=calibration_accepted,
            context_state_factory=context_state_factory,
            seed=seed,
            initial_noise=initial_noise,
            ddim_steps=ddim_steps,
        )
