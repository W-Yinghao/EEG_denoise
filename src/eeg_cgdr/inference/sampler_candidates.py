"""Frozen M0--M5 sampler-mechanism names, independent of operator source.

These IDs describe how the population prior and observation consistency are
combined. They must never be reused for matching/population/wrong/shuffled/
oracle operator provenance, which is a separate experimental axis.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Literal, Optional

import torch
from torch import Tensor

from .one_step import InformationMatchedOneStep
from .population import (
    GuidanceStepTrace,
    PopulationObservationState,
    PopulationOnlyInference,
)


class SamplerMechanism(str, Enum):
    M0 = "full_generation_guided_ddim"
    M1 = "observation_warm_start_sdedit"
    M2 = "final_hard_q_consistency"
    M3 = "per_step_hard_q_consistency"
    M4 = "per_step_quadratic_proximal_q_consistency"
    M5 = "single_prior_eval_proximal"


ImplementationStatus = Literal["implemented"]


@dataclass(frozen=True)
class SamplerCandidateSpec:
    candidate_id: str
    mechanism: SamplerMechanism
    consistency_timing: str
    prior_evaluations: str
    implementation_status: ImplementationStatus
    implementation_entrypoint: str

    @property
    def registered_name(self) -> str:
        """Return the exact configuration-facing candidate name."""

        return f"{self.candidate_id}_{self.mechanism.value}"


SAMPLER_CANDIDATES = (
    SamplerCandidateSpec(
        "M0",
        SamplerMechanism.M0,
        "observation-energy full VJP at every guided DDIM step",
        "exactly ddim_steps",
        "implemented",
        "RepairedSamplerRunner.run(M0)",
    ),
    SamplerCandidateSpec(
        "M1",
        SamplerMechanism.M1,
        "observation q-sample warm start followed by guided DDIM",
        "exactly ddim_steps after warm start",
        "implemented",
        "RepairedSamplerRunner.run(M1)",
    ),
    SamplerCandidateSpec(
        "M2",
        SamplerMechanism.M2,
        "hard Q-consistency once after the final reverse step",
        "exactly ddim_steps",
        "implemented",
        "RepairedSamplerRunner.run(M2)",
    ),
    SamplerCandidateSpec(
        "M3",
        SamplerMechanism.M3,
        "hard Q-consistency after every reverse step",
        "exactly ddim_steps",
        "implemented",
        "RepairedSamplerRunner.run(M3)",
    ),
    SamplerCandidateSpec(
        "M4",
        SamplerMechanism.M4,
        "quadratic proximal Q-consistency after every reverse step",
        "exactly ddim_steps",
        "implemented",
        "RepairedSamplerRunner.run(M4)",
    ),
    SamplerCandidateSpec(
        "M5",
        SamplerMechanism.M5,
        "one quadratic proximal update after one prior evaluation",
        "exactly one",
        "implemented",
        "RepairedSamplerRunner.run(M5)",
    ),
)


def sampler_candidate(candidate_name: str) -> SamplerCandidateSpec:
    """Resolve a short ID or its exact long registered name.

    Operator-source names and approximate spellings are deliberately rejected.
    """

    matches = [
        item
        for item in SAMPLER_CANDIDATES
        if candidate_name in (item.candidate_id, item.registered_name)
    ]
    if len(matches) != 1:
        raise ValueError(f"unknown sampler candidate name: {candidate_name!r}")
    return matches[0]


@dataclass(frozen=True)
class SamplerRunResult:
    mechanism: SamplerMechanism
    candidate_id: str
    mechanism_name: str
    restored: Tensor
    trace: tuple[GuidanceStepTrace, ...]
    network_evaluations: int
    residual_dimension_normalization: bool
    trust_radius_ratio: float


class RepairedSamplerRunner:
    """Callable M0--M5 mechanisms with one shared stability contract."""

    def __init__(self, inference: PopulationOnlyInference) -> None:
        self.inference = inference

    @staticmethod
    def _projector(
        projector: Optional[Tensor], state: PopulationObservationState
    ) -> Optional[Tensor]:
        if projector is None:
            return None
        value = torch.as_tensor(
            projector,
            device=state.observation.device,
            dtype=state.observation.dtype,
        )
        channels = state.observation.shape[1]
        if value.shape != (channels, channels):
            raise ValueError("Q-consistency projector must have shape (C,C)")
        if not torch.allclose(value, value.T, atol=1.0e-6, rtol=1.0e-5):
            raise ValueError("Q-consistency projector must be symmetric")
        if not torch.allclose(value @ value, value, atol=2.0e-5, rtol=2.0e-5):
            raise ValueError("Q-consistency projector must be idempotent")
        return value

    @staticmethod
    def _pq_residuals(
        value: Tensor,
        state: PopulationObservationState,
        projector: Optional[Tensor],
    ) -> tuple[Optional[float], Optional[float]]:
        if projector is None:
            return None, None
        mask = state.valid_time_mask[:, None, :].to(dtype=value.dtype)
        residual = (value - state.observation) * mask
        parallel = torch.einsum("cd,bdl->bcl", projector, residual)
        perpendicular = residual - parallel
        dimensions = state.residual_dimensions().sum().clamp_min(1.0)
        return (
            float(torch.linalg.vector_norm(parallel) / torch.sqrt(dimensions)),
            float(torch.linalg.vector_norm(perpendicular) / torch.sqrt(dimensions)),
        )

    @staticmethod
    def _hard_q(
        value: Tensor,
        state: PopulationObservationState,
        projector: Tensor,
    ) -> Tensor:
        residual = state.observation - value
        parallel = torch.einsum("cd,bdl->bcl", projector, residual)
        q_correction = residual - parallel
        mask = state.valid_time_mask[:, None, :].to(dtype=value.dtype)
        return (value + q_correction) * mask

    @staticmethod
    def _proximal_q(
        value: Tensor,
        state: PopulationObservationState,
        projector: Tensor,
        strength: float,
    ) -> Tensor:
        if not 0.0 < strength < float("inf"):
            raise ValueError("Q proximal strength must be finite and positive")
        residual = state.observation - value
        parallel = torch.einsum("cd,bdl->bcl", projector, residual)
        q_correction = residual - parallel
        mask = state.valid_time_mask[:, None, :].to(dtype=value.dtype)
        return (value + (strength / (1.0 + strength)) * q_correction) * mask

    def _effective_precision(self, state: PopulationObservationState) -> Tensor:
        dimensions = (
            state.residual_dimensions()
            if self.inference.stability.normalize_by_residual_dimension
            else torch.ones(
                state.observation.shape[0],
                device=state.observation.device,
                dtype=state.observation.dtype,
            )
        )
        precision = state.precision * float(state.energy_scale)
        batch, channels, length = state.observation.shape
        kind = state._precision_kind
        if kind == "scalar":
            identity = torch.eye(
                channels, device=precision.device, dtype=precision.dtype
            )
            precision = precision * identity.unsqueeze(0).expand(batch, -1, -1)
        elif kind == "channel_diagonal":
            precision = torch.diag(precision).unsqueeze(0).expand(batch, -1, -1)
        elif kind == "elementwise":
            elementwise = torch.broadcast_to(precision, state.observation.shape)
            precision = torch.diag_embed(elementwise.transpose(1, 2))
        elif kind == "matrix":
            precision = precision.unsqueeze(0).expand(batch, -1, -1)
        elif kind == "time_matrix":
            precision = precision.unsqueeze(0).expand(batch, length, -1, -1)
        if precision.ndim == 3:
            return precision / dimensions.reshape(-1, 1, 1)
        if precision.ndim == 4:
            return precision / dimensions.reshape(-1, 1, 1, 1)
        raise AssertionError(f"unsupported one-step precision kind: {kind}")

    def _trace_transform(
        self,
        *,
        mechanism: SamplerMechanism,
        state: PopulationObservationState,
        projector: Optional[Tensor],
        proximal_strength: float,
        trace: list[GuidanceStepTrace],
    ):
        def transform(value: Tensor, _timestep: int, is_final: bool) -> Tensor:
            before = value
            p_before, q_before = self._pq_residuals(before, state, projector)
            apply_hard = (
                mechanism == SamplerMechanism.M3
                or (mechanism == SamplerMechanism.M2 and is_final)
            )
            if apply_hard:
                if projector is None:
                    raise ValueError(f"{mechanism.name} requires a Q projector")
                after = self._hard_q(before, state, projector)
            elif mechanism == SamplerMechanism.M4:
                if projector is None:
                    raise ValueError("M4 requires a Q projector")
                after = self._proximal_q(
                    before, state, projector, proximal_strength
                )
            else:
                after = before
            p_after, q_after = self._pq_residuals(after, state, projector)
            if not trace:
                raise AssertionError("consistency transform ran before guidance trace")
            trace[-1] = replace(
                trace[-1],
                mechanism_id=mechanism.name,
                state_norm_after_ddim=float(torch.linalg.vector_norm(before)),
                p_residual_before=p_before,
                q_residual_before=q_before,
                p_residual_after=p_after,
                q_residual_after=q_after,
                sample_norm_before_consistency=float(torch.linalg.vector_norm(before)),
                sample_norm_after_consistency=float(torch.linalg.vector_norm(after)),
                consistency_update_l2=float(
                    torch.linalg.vector_norm(after - before)
                ),
            )
            return after

        return transform

    def _one_step_trace(
        self,
        *,
        state: PopulationObservationState,
        prior_epsilon: Tensor,
        prior_clean: Tensor,
        restored: Tensor,
        timestep: int,
        projector: Optional[Tensor],
        clipped_fraction: float,
    ) -> GuidanceStepTrace:
        with torch.enable_grad():
            raw_clean = prior_clean.detach().requires_grad_(True)
            energy = state.energy_per_sample(raw_clean)
            raw_gradient = torch.autograd.grad(energy.sum(), raw_clean)[0].detach()
        dimensions = state.residual_dimensions().reshape(-1, 1, 1)
        normalized_gradient = raw_gradient / dimensions
        p_before, q_before = self._pq_residuals(prior_clean, state, projector)
        p_after, q_after = self._pq_residuals(restored, state, projector)
        score = self.inference.prior.score_from_epsilon(
            prior_epsilon,
            torch.full(
                (state.observation.shape[0],),
                int(timestep),
                device=state.observation.device,
                dtype=torch.long,
            ),
        )
        return GuidanceStepTrace(
            timestep=int(timestep),
            checkpoint_label="first_middle_last",
            raw_energy_mean=float(energy.mean()),
            normalized_energy_mean=float(
                (energy / state.residual_dimensions()).mean()
            ),
            prior_score_l2=float(torch.linalg.vector_norm(score)),
            prior_epsilon_l2=float(torch.linalg.vector_norm(prior_epsilon)),
            clean_estimate_l2=float(torch.linalg.vector_norm(prior_clean)),
            raw_energy_vjp_l2=float(torch.linalg.vector_norm(raw_gradient)),
            normalized_energy_vjp_l2=float(
                torch.linalg.vector_norm(normalized_gradient)
            ),
            epsilon_guidance_l2=float(
                torch.linalg.vector_norm(restored - prior_clean)
            ),
            guided_epsilon_l2=float(torch.linalg.vector_norm(prior_epsilon)),
            valid_fraction=float(state.valid_time_mask.float().mean()),
            finite_fraction=float(torch.isfinite(restored).float().mean()),
            clipping_fraction=clipped_fraction,
            state_norm_before_ddim=float(torch.linalg.vector_norm(prior_clean)),
            state_norm_after_ddim=float(torch.linalg.vector_norm(restored)),
            p_residual_before=p_before,
            q_residual_before=q_before,
            p_residual_after=p_after,
            q_residual_after=q_after,
            sample_norm_before_consistency=float(torch.linalg.vector_norm(prior_clean)),
            sample_norm_after_consistency=float(torch.linalg.vector_norm(restored)),
            consistency_update_l2=float(
                torch.linalg.vector_norm(restored - prior_clean)
            ),
            mechanism_id="M5",
            gradient_semantics="direct_x0_energy_gradient_single_prior_eval",
            sign_convention="single_prior_eval_then_quadratic_proximal",
        )

    def run(
        self,
        mechanism: SamplerMechanism | str,
        state: PopulationObservationState,
        *,
        seed: int,
        ddim_steps: int,
        projector: Optional[Tensor] = None,
        warm_start_timestep: Optional[int] = None,
        one_step_timestep: Optional[int] = None,
        proximal_strength: float = 1.0,
    ) -> SamplerRunResult:
        """Run one exact mechanism; operator provenance is supplied separately."""

        if isinstance(mechanism, str) and not isinstance(
            mechanism, SamplerMechanism
        ):
            mechanism = sampler_candidate(mechanism).mechanism
        if not isinstance(mechanism, SamplerMechanism):
            raise TypeError("mechanism must be a SamplerMechanism or registered name")
        projection = self._projector(projector, state)
        trace: list[GuidanceStepTrace] = []
        if mechanism == SamplerMechanism.M5:
            if int(ddim_steps) != 1 or ddim_steps != 1:
                raise ValueError("M5 has no DDIM loop and requires ddim_steps=1")
            if one_step_timestep is None:
                raise ValueError("M5 requires one_step_timestep")
            if int(one_step_timestep) != one_step_timestep:
                raise ValueError("one_step_timestep must be an integer")
            detailed = InformationMatchedOneStep(
                self.inference.prior
            ).restore_detailed(
                observation=state.observation,
                channel_precision=self._effective_precision(state),
                seed=seed,
                timestep=one_step_timestep,
                proximal_strength=proximal_strength,
                valid_time_mask=state.valid_time_mask,
            )
            delta = detailed.restored - detailed.prior_clean
            delta_norm = torch.linalg.vector_norm(delta.flatten(start_dim=1), dim=1)
            reference = torch.linalg.vector_norm(
                detailed.prior_clean.flatten(start_dim=1), dim=1
            ).clamp_min(self.inference.stability.minimum_reference_norm)
            limit = self.inference.stability.trust_radius_ratio * reference
            factor = torch.minimum(
                torch.ones_like(delta_norm),
                limit
                / delta_norm.clamp_min(
                    self.inference.stability.minimum_reference_norm
                ),
            )
            restored = detailed.prior_clean + delta * factor.reshape(-1, 1, 1)
            restored = restored * state.valid_time_mask[:, None, :].to(
                dtype=restored.dtype
            )
            trace.append(
                self._one_step_trace(
                    state=state,
                    prior_epsilon=detailed.prior_epsilon,
                    prior_clean=detailed.prior_clean,
                    restored=restored,
                    timestep=one_step_timestep,
                    projector=projection,
                    clipped_fraction=float((factor < 1.0).float().mean()),
                )
            )
        else:
            if mechanism in (SamplerMechanism.M2, SamplerMechanism.M3, SamplerMechanism.M4):
                if projection is None:
                    raise ValueError(f"{mechanism.name} requires projector")
            initial = self.inference.make_initial_noise(state, seed=seed)
            t_start = None
            if mechanism == SamplerMechanism.M1:
                if warm_start_timestep is None:
                    raise ValueError("M1 requires warm_start_timestep")
                if int(warm_start_timestep) != warm_start_timestep:
                    raise ValueError("warm_start_timestep must be an integer")
                t_start = int(warm_start_timestep)
                if not 0 < t_start < self.inference.prior.diffusion.num_timesteps:
                    raise ValueError("warm_start_timestep must lie in (0,T)")
                self.inference.prior.diffusion.ddim_timesteps(
                    ddim_steps, t_start=t_start
                )
                timestep = torch.full(
                    (state.observation.shape[0],),
                    t_start,
                    device=state.observation.device,
                    dtype=torch.long,
                )
                initial = self.inference.prior.diffusion.q_sample(
                    state.observation, timestep, initial
                )
            transform = self._trace_transform(
                mechanism=mechanism,
                state=state,
                projector=projection,
                proximal_strength=proximal_strength,
                trace=trace,
            )
            restored = self.inference._sample_energy(
                state,
                state.energy_per_sample,
                seed=None,
                initial_noise=initial,
                ddim_steps=ddim_steps,
                guidance_trace=trace,
                t_start=t_start,
                step_transform=transform,
            )
        return SamplerRunResult(
            mechanism=mechanism,
            candidate_id=mechanism.name,
            mechanism_name=mechanism.value,
            restored=restored,
            trace=tuple(trace),
            network_evaluations=len(trace),
            residual_dimension_normalization=self.inference.stability.normalize_by_residual_dimension,
            trust_radius_ratio=float(self.inference.stability.trust_radius_ratio),
        )
