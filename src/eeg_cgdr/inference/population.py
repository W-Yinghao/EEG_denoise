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

from ..models.clean_prior import CleanEEGDiffusionPrior, canonical_valid_time_mask

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
class FullVJPResult:
    """One exact VJP through ``x0_hat(x_t, epsilon_theta(x_t,t))``."""

    prior_epsilon: Tensor
    clean_estimate: Tensor
    energy_per_sample: Tensor
    energy_vjp: Tensor


@dataclass(frozen=True)
class GuidanceStepTrace:
    """Auditable values for exactly one reverse-network evaluation."""

    timestep: int
    checkpoint_label: str
    raw_energy_mean: float
    normalized_energy_mean: float
    prior_score_l2: float
    prior_epsilon_l2: float
    clean_estimate_l2: float
    raw_energy_vjp_l2: float
    normalized_energy_vjp_l2: float
    epsilon_guidance_l2: float
    guided_epsilon_l2: float
    guided_score_l2: float
    valid_fraction: float
    finite_fraction: float
    clipping_fraction: float
    state_norm_before_ddim: Optional[float] = None
    state_norm_after_ddim: Optional[float] = None
    p_residual_before: Optional[float] = None
    q_residual_before: Optional[float] = None
    p_residual_after: Optional[float] = None
    q_residual_after: Optional[float] = None
    sample_norm_before_consistency: Optional[float] = None
    sample_norm_after_consistency: Optional[float] = None
    consistency_update_l2: Optional[float] = None
    network_evaluations: int = 1
    mechanism_id: str = "M0"
    gradient_semantics: str = "full_VJP_through_epsilon_network"
    sign_convention: str = "epsilon_guided=epsilon_prior+sigma*VJP(E)"
    consistency_semantics: str = "none"
    precision_residual_before: Optional[float] = None
    precision_residual_after: Optional[float] = None


@dataclass(frozen=True)
class GuidanceStabilityConfig:
    """Frozen dimension normalization and relative trust-radius clipping."""

    normalize_by_residual_dimension: bool = True
    trust_radius_ratio: float = 1.0
    minimum_reference_norm: float = 1.0e-8

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.trust_radius_ratio)) or self.trust_radius_ratio <= 0:
            raise ValueError("trust_radius_ratio must be finite and positive")
        if (
            not math.isfinite(float(self.minimum_reference_norm))
            or self.minimum_reference_norm <= 0
        ):
            raise ValueError("minimum_reference_norm must be finite and positive")


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
    energy_scale: float = 1.0
    name: str = "population"
    valid_time_mask: Optional[Tensor] = None
    dataset_id: str = "unspecified_dataset"
    montage_id: str = "unspecified_montage"
    precision_semantics: str = "unspecified_precision"
    consistency_rho: Optional[float] = None
    population_consistency_projector: Optional[Tensor] = None
    context_consistency_projector: Optional[Tensor] = None
    _precision_kind: PrecisionKind = field(init=False, repr=False)

    def __post_init__(self) -> None:
        observation = self.observation
        if observation.ndim != 3 or not observation.dtype.is_floating_point:
            raise ValueError("observation must be a floating (B,C,L) tensor")
        if any(size < 1 for size in observation.shape):
            raise ValueError("observation cannot contain an empty dimension")
        if not bool(torch.isfinite(observation).all()):
            raise ValueError("observation contains non-finite values")
        if not math.isfinite(float(self.energy_scale)) or float(self.energy_scale) < 0.0:
            raise ValueError("observation energy scale must be finite and non-negative")
        for field_name in ("name", "dataset_id", "montage_id", "precision_semantics"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        valid_time_mask = canonical_valid_time_mask(
            observation, self.valid_time_mask
        )[:, 0, :]
        masked_observation = observation * valid_time_mask[:, None, :].to(
            dtype=observation.dtype
        )

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

        rho = self.consistency_rho
        if rho is not None:
            rho = float(rho)
            if not math.isfinite(rho) or not 0.0 <= rho <= 1.0:
                raise ValueError("consistency_rho must be finite and lie in [0,1]")

        def consistency_projector(value: Optional[Tensor], name: str) -> Optional[Tensor]:
            if value is None:
                return None
            projector = torch.as_tensor(
                value,
                device=observation.device,
                dtype=observation.dtype,
            ).detach()
            if projector.shape != (channels, channels):
                raise ValueError(f"{name} must have shape (C,C)")
            if not bool(torch.isfinite(projector).all()):
                raise ValueError(f"{name} contains non-finite values")
            if not torch.allclose(projector, projector.T, atol=1.0e-6, rtol=1.0e-5):
                raise ValueError(f"{name} must be symmetric")
            if not torch.allclose(
                projector @ projector,
                projector,
                atol=2.0e-5,
                rtol=2.0e-5,
            ):
                raise ValueError(f"{name} must be idempotent")
            return projector

        population_projector = consistency_projector(
            self.population_consistency_projector,
            "population_consistency_projector",
        )
        context_projector = consistency_projector(
            self.context_consistency_projector,
            "context_consistency_projector",
        )
        if rho == 0.0 and population_projector is None:
            raise ValueError("rho=0 consistency requires the population projector")
        if rho == 1.0 and context_projector is None:
            raise ValueError("rho=1 consistency requires the context projector")
        if rho is not None and 0.0 < rho < 1.0:
            if population_projector is None or context_projector is None:
                raise ValueError(
                    "interpolated consistency requires population and context projectors"
                )

        object.__setattr__(self, "observation", masked_observation)
        object.__setattr__(self, "precision", precision)
        object.__setattr__(self, "valid_time_mask", valid_time_mask)
        object.__setattr__(self, "consistency_rho", rho)
        object.__setattr__(
            self, "population_consistency_projector", population_projector
        )
        object.__setattr__(
            self, "context_consistency_projector", context_projector
        )
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
        mask = self.valid_time_mask[:, None, :].to(dtype=clean_estimate.dtype)
        residual = (clean_estimate - self.observation) * mask
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
        return 0.5 * float(self.energy_scale) * quadratic

    def energy(self, clean_estimate: Tensor) -> Tensor:
        """Return the batch-summed scalar energy used for the VJP."""

        return self.energy_per_sample(clean_estimate).sum()

    def residual_dimensions(self) -> Tensor:
        """Return effective observed residual dimensions for each batch item."""

        batch, channels, length = self.observation.shape
        valid = self.valid_time_mask
        precision = self.precision
        if self._precision_kind == "scalar":
            active_channels = channels if float(precision) > 0.0 else 0
            dimensions = valid.sum(dim=1) * active_channels
        elif self._precision_kind == "channel_diagonal":
            active_channels = int((precision > 0.0).sum())
            dimensions = valid.sum(dim=1) * active_channels
        elif self._precision_kind == "elementwise":
            expanded = torch.broadcast_to(precision, self.observation.shape)
            dimensions = (
                (expanded > 0.0) & valid[:, None, :]
            ).sum(dim=(1, 2))
        elif self._precision_kind == "matrix":
            rank = int(torch.linalg.matrix_rank(precision.float()))
            dimensions = valid.sum(dim=1) * rank
        elif self._precision_kind == "batch_matrix":
            rank = torch.linalg.matrix_rank(precision.float())
            dimensions = valid.sum(dim=1) * rank
        elif self._precision_kind == "time_matrix":
            rank = torch.linalg.matrix_rank(precision.float())
            dimensions = (valid * rank.reshape(1, length)).sum(dim=1)
        elif self._precision_kind == "batch_time_matrix":
            rank = torch.linalg.matrix_rank(precision.float())
            dimensions = (valid * rank).sum(dim=1)
        else:  # pragma: no cover - all kinds are exhaustive above
            raise AssertionError(self._precision_kind)
        return dimensions.to(device=self.observation.device, dtype=self.observation.dtype).clamp_min(1.0)


class PopulationOnlyInference:
    """Observation-conditioned POP sampler with an explicit clean prior and ``E_0``."""

    def __init__(
        self,
        prior: CleanEEGDiffusionPrior,
        *,
        stability: Optional[GuidanceStabilityConfig] = None,
    ) -> None:
        self.prior = prior
        self.stability = stability or GuidanceStabilityConfig()

    @staticmethod
    def _validate_steps(
        prior: CleanEEGDiffusionPrior,
        ddim_steps: int,
        *,
        t_start: Optional[int] = None,
    ) -> None:
        prior.diffusion.ddim_timesteps(ddim_steps, t_start=t_start)

    @staticmethod
    def _mask_tensor(value: Tensor, state: PopulationObservationState) -> Tensor:
        return value * state.valid_time_mask[:, None, :].to(dtype=value.dtype)

    def make_initial_noise(self, state: PopulationObservationState, *, seed: int) -> Tensor:
        """Create explicit, local-RNG initial noise for paired POP/CGDR calls."""

        seed_value = int(seed)
        if seed_value != seed or seed_value < 0:
            raise ValueError("seed must be non-negative")
        generator = torch.Generator(device=state.observation.device)
        generator.manual_seed(seed_value)
        noise = torch.randn(
            state.observation.shape,
            device=state.observation.device,
            dtype=state.observation.dtype,
            generator=generator,
        )
        return self._mask_tensor(noise, state)

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
        return self._mask_tensor(initial_noise.detach(), state)

    def full_energy_vjp(
        self,
        x_t: Tensor,
        timesteps: Tensor,
        *,
        energy: EnergyFunction,
        valid_time_mask: Tensor,
    ) -> FullVJPResult:
        """Compute the full, non-stop-gradient VJP of ``E(x0_hat(x_t))``.

        The derivative includes the dependence of epsilon-theta on ``x_t``.
        This public helper is the target of the directional finite-difference
        acceptance test; no simplified Jacobian or detached denoiser path is
        hidden in the sampler.
        """

        mask = canonical_valid_time_mask(x_t, valid_time_mask)
        with torch.enable_grad():
            differentiable_x = (
                x_t.detach() * mask.to(dtype=x_t.dtype)
            ).requires_grad_(True)
            prior_epsilon = self.prior.predict_noise(
                differentiable_x,
                timesteps,
                valid_time_mask=mask,
            )
            clean_estimate = self.prior.predict_clean(
                differentiable_x,
                timesteps,
                prior_epsilon,
                valid_time_mask=mask,
            )
            energy_per_sample = energy(clean_estimate)
            if energy_per_sample.shape != (x_t.shape[0],):
                raise ValueError("energy function must return exactly one value per sample")
            if not bool(torch.isfinite(energy_per_sample).all()):
                raise FloatingPointError("observation energy is non-finite")
            gradient = torch.autograd.grad(
                energy_per_sample.sum(),
                differentiable_x,
                create_graph=False,
                retain_graph=False,
                only_inputs=True,
            )[0]
        gradient = gradient * mask.to(dtype=gradient.dtype)
        if not bool(torch.isfinite(gradient).all()):
            raise FloatingPointError("full observation-energy VJP is non-finite")
        return FullVJPResult(
            prior_epsilon=prior_epsilon.detach(),
            clean_estimate=clean_estimate.detach(),
            energy_per_sample=energy_per_sample.detach(),
            energy_vjp=gradient.detach(),
        )

    def guided_epsilon(
        self,
        x_t: Tensor,
        timesteps: Tensor,
        *,
        energy: EnergyFunction,
        valid_time_mask: Tensor,
        guidance_trace: Optional[list[GuidanceStepTrace]] = None,
        residual_dimensions: Optional[Tensor] = None,
        checkpoint_label: str = "intermediate",
    ) -> Tensor:
        """Return epsilon for posterior score ``s_prior - VJP(E)``.

        Since ``s_prior=-epsilon_prior/sigma``, the correct epsilon-domain
        update is ``epsilon_prior + sigma*VJP(E)``. The plus sign is explicit
        here and is covered by the score/epsilon unit test.
        """

        result = self.full_energy_vjp(
            x_t,
            timesteps,
            energy=energy,
            valid_time_mask=valid_time_mask,
        )
        mask = canonical_valid_time_mask(x_t, valid_time_mask)
        if residual_dimensions is None:
            residual_dimensions = (
                mask[:, 0, :].sum(dim=1).to(dtype=x_t.dtype) * x_t.shape[1]
            ).clamp_min(1.0)
        else:
            residual_dimensions = torch.as_tensor(
                residual_dimensions, device=x_t.device, dtype=x_t.dtype
            )
            if residual_dimensions.shape != (x_t.shape[0],):
                raise ValueError("residual_dimensions must have shape (B,)")
            if not bool(torch.isfinite(residual_dimensions).all()) or bool(
                (residual_dimensions <= 0).any()
            ):
                raise ValueError("residual_dimensions must be finite and positive")
        divisor = (
            residual_dimensions.reshape(-1, 1, 1)
            if self.stability.normalize_by_residual_dimension
            else torch.ones_like(residual_dimensions).reshape(-1, 1, 1)
        )
        normalized_vjp = result.energy_vjp / divisor
        sigma = self.prior.noise_standard_deviation(timesteps, x_t.ndim)
        raw_epsilon_delta = sigma * normalized_vjp
        batch_delta_norm = torch.linalg.vector_norm(
            raw_epsilon_delta.flatten(start_dim=1), dim=1
        )
        batch_prior_norm = torch.linalg.vector_norm(
            result.prior_epsilon.flatten(start_dim=1), dim=1
        )
        trust_limit = float(self.stability.trust_radius_ratio) * batch_prior_norm.clamp_min(
            float(self.stability.minimum_reference_norm)
        )
        clip_factor = torch.minimum(
            torch.ones_like(batch_delta_norm),
            trust_limit / batch_delta_norm.clamp_min(
                float(self.stability.minimum_reference_norm)
            ),
        )
        epsilon_delta = raw_epsilon_delta * clip_factor.reshape(-1, 1, 1)
        guided = result.prior_epsilon + epsilon_delta
        guided = guided * mask.to(dtype=guided.dtype)
        finite_values = torch.cat(
            [
                result.prior_epsilon.flatten(),
                result.clean_estimate.flatten(),
                result.energy_vjp.flatten(),
                guided.flatten(),
            ]
        )
        finite_fraction = float(torch.isfinite(finite_values).float().mean())
        if finite_fraction != 1.0:
            raise FloatingPointError("guided epsilon trace contains non-finite values")
        if guidance_trace is not None:
            unique_timesteps = torch.unique(timesteps.detach())
            if unique_timesteps.numel() != 1:
                raise ValueError("guidance trace requires one shared timestep per batch")
            guidance_trace.append(
                GuidanceStepTrace(
                    timestep=int(unique_timesteps.item()),
                    checkpoint_label=checkpoint_label,
                    raw_energy_mean=float(result.energy_per_sample.mean()),
                    normalized_energy_mean=float(
                        (result.energy_per_sample / residual_dimensions).mean()
                    ),
                    prior_score_l2=float(
                        torch.linalg.vector_norm(
                            self.prior.score_from_epsilon(
                                result.prior_epsilon, timesteps
                            )
                        )
                    ),
                    prior_epsilon_l2=float(torch.linalg.vector_norm(result.prior_epsilon)),
                    clean_estimate_l2=float(
                        torch.linalg.vector_norm(result.clean_estimate)
                    ),
                    raw_energy_vjp_l2=float(
                        torch.linalg.vector_norm(result.energy_vjp)
                    ),
                    normalized_energy_vjp_l2=float(
                        torch.linalg.vector_norm(normalized_vjp)
                    ),
                    epsilon_guidance_l2=float(torch.linalg.vector_norm(epsilon_delta)),
                    guided_epsilon_l2=float(torch.linalg.vector_norm(guided)),
                    guided_score_l2=float(
                        torch.linalg.vector_norm(
                            self.prior.score_from_epsilon(guided, timesteps)
                        )
                    ),
                    valid_fraction=float(mask.float().mean()),
                    finite_fraction=finite_fraction,
                    clipping_fraction=float((clip_factor < 1.0).float().mean()),
                    state_norm_before_ddim=float(
                        torch.linalg.vector_norm(
                            x_t * mask.to(dtype=x_t.dtype)
                        )
                    ),
                )
            )
        return guided.detach()

    def _guided_noise_function(
        self,
        energy: EnergyFunction,
        *,
        valid_time_mask: Tensor,
        guidance_trace: Optional[list[GuidanceStepTrace]],
        residual_dimensions: Tensor,
        trace_labels: dict[int, str],
    ) -> Callable[[Tensor, Tensor], Tensor]:

        def guided_noise(x_t: Tensor, timesteps: Tensor) -> Tensor:
            return self.guided_epsilon(
                x_t,
                timesteps,
                energy=energy,
                valid_time_mask=valid_time_mask,
                guidance_trace=guidance_trace,
                residual_dimensions=residual_dimensions,
                checkpoint_label=trace_labels.get(
                    int(timesteps[0].detach()), "intermediate"
                ),
            )

        return guided_noise

    def _sample_energy(
        self,
        state: PopulationObservationState,
        energy: EnergyFunction,
        *,
        seed: Optional[int],
        initial_noise: Optional[Tensor],
        ddim_steps: int,
        guidance_trace: Optional[list[GuidanceStepTrace]],
        t_start: Optional[int] = None,
        step_transform: Optional[Callable[[Tensor, int, bool], Tensor]] = None,
    ) -> Tensor:
        self._validate_steps(self.prior, ddim_steps, t_start=t_start)
        x_t = self._resolve_initial_noise(
            state,
            seed=seed,
            initial_noise=initial_noise,
        )
        self.prior.eval()
        sequence = self.prior.diffusion.ddim_timesteps(
            int(ddim_steps), t_start=t_start
        )
        checkpoint_indices = {
            0: "first",
            len(sequence) // 2: "middle",
            len(sequence) - 1: "last",
        }
        trace_labels: dict[int, str] = {}
        for index, timestep in enumerate(sequence):
            label = checkpoint_indices.get(index, "intermediate")
            if len(sequence) == 1:
                label = "first_middle_last"
            trace_labels[timestep] = label
        trace_start = len(guidance_trace) if guidance_trace is not None else 0
        output = self.prior.diffusion.ddim_sample_loop(
            self._guided_noise_function(
                energy,
                valid_time_mask=state.valid_time_mask,
                guidance_trace=guidance_trace,
                residual_dimensions=state.residual_dimensions(),
                trace_labels=trace_labels,
            ),
            tuple(state.observation.shape),
            state.observation.device,
            ddim_steps=int(ddim_steps),
            eta=0.0,
            x_t=x_t,
            clip_denoised=False,
            valid_time_mask=state.valid_time_mask,
            t_start=t_start,
            step_transform=step_transform,
        )
        if guidance_trace is not None:
            calls = len(guidance_trace) - trace_start
            if calls != int(ddim_steps):
                raise AssertionError(
                    f"DDIM requested {ddim_steps} steps but recorded {calls} network calls"
                )
        return output

    def sample(
        self,
        population_state: PopulationObservationState,
        *,
        seed: Optional[int] = None,
        initial_noise: Optional[Tensor] = None,
        ddim_steps: int = 50,
        guidance_trace: Optional[list[GuidanceStepTrace]] = None,
    ) -> Tensor:
        """Sample the explicit population posterior using only ``E_0``."""

        return self._sample_energy(
            population_state,
            population_state.energy_per_sample,
            seed=seed,
            initial_noise=initial_noise,
            ddim_steps=ddim_steps,
            guidance_trace=guidance_trace,
        )

    @staticmethod
    def _same_query(
        population_state: PopulationObservationState,
        context_state: PopulationObservationState,
    ) -> None:
        population_query = population_state.observation
        context_query = context_state.observation
        if (
            context_query is not population_query
            and (
                context_query.shape != population_query.shape
                or context_query.device != population_query.device
                or context_query.dtype != population_query.dtype
                or not torch.equal(context_query, population_query)
            )
        ):
            raise ValueError("E_0 and E_C must use the same query observation")
        if not torch.equal(
            population_state.valid_time_mask, context_state.valid_time_mask
        ):
            raise ValueError("E_0 and E_C must use the same valid-time mask")
        if population_state.energy_scale != context_state.energy_scale:
            raise ValueError("E_0 and E_C must use the same energy_scale")
        for field_name in ("dataset_id", "montage_id", "precision_semantics"):
            if getattr(population_state, field_name) != getattr(context_state, field_name):
                raise ValueError(f"E_0 and E_C {field_name} differ")

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
        guidance_trace: Optional[list[GuidanceStepTrace]] = None,
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
                guidance_trace=guidance_trace,
            )

        if context_state_factory is None:
            raise ValueError("accepted non-zero calibration requires a lazy context state factory")
        context_state = context_state_factory()
        if not isinstance(context_state, PopulationObservationState):
            raise TypeError("context_state_factory must return PopulationObservationState")
        self._same_query(population_state, context_state)
        # Local import avoids the states -> population construction dependency.
        # Precision interpolation is algebraically identical to E0+rho(EC-E0)
        # and supplies the correct effective residual dimension to stabilization.
        from .states import rho_interpolated_precision_state

        interpolated_state = rho_interpolated_precision_state(
            population_state,
            rho=rho_value,
            calibration_accepted=True,
            context_state_factory=lambda: context_state,
        )

        return self._sample_energy(
            interpolated_state,
            interpolated_state.energy_per_sample,
            seed=seed,
            initial_noise=initial_noise,
            ddim_steps=ddim_steps,
            guidance_trace=guidance_trace,
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
        guidance_trace: Optional[list[GuidanceStepTrace]] = None,
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
            guidance_trace=guidance_trace,
        )
