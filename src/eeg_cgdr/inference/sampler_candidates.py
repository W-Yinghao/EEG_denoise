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
    consistency_semantics: str = "none"


@dataclass(frozen=True)
class _ResolvedConsistency:
    """Endpoint projector or non-projector PSD geometry for ``W_rho``."""

    semantics: str
    projector: Optional[Tensor]
    use_psd_precision: bool


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

    def _resolve_consistency(
        self,
        supplied_projector: Optional[Tensor],
        state: PopulationObservationState,
    ) -> _ResolvedConsistency:
        """Resolve endpoint Q geometry or the actual interpolated PSD precision.

        For ``rho=0`` and ``rho=1`` the respective Pi0/PiC is an orthogonal
        projector and the historical hard-Q endpoint operation is well
        defined.  For ``0<rho<1``, ``W_rho`` is generally not a projector;
        accepting PiC there would silently apply the wrong geometry.  Such a
        state therefore uses an explicitly labelled PSD precision update.

        States without formal rho metadata retain the legacy explicit-
        projector path for isolated ablations and backwards-compatible tests.
        """

        supplied = self._projector(supplied_projector, state)
        rho = state.consistency_rho
        if rho is None:
            return _ResolvedConsistency(
                semantics=(
                    "legacy_explicit_orthogonal_complement_projector"
                    if supplied is not None
                    else "none"
                ),
                projector=supplied,
                use_psd_precision=False,
            )
        if 0.0 < rho < 1.0:
            if supplied is not None:
                raise ValueError(
                    "0<rho<1 uses PSD precision consistency from W_rho; "
                    "a fixed Q projector is not valid"
                )
            return _ResolvedConsistency(
                semantics="psd_precision_consistency_Wrho_not_a_projector",
                projector=None,
                use_psd_precision=True,
            )
        expected = (
            state.population_consistency_projector
            if rho == 0.0
            else state.context_consistency_projector
        )
        if expected is None:  # guarded by PopulationObservationState
            raise AssertionError("endpoint consistency projector is missing")
        expected = self._projector(expected, state)
        assert expected is not None
        if supplied is not None and not torch.allclose(
            supplied, expected, atol=1.0e-6, rtol=1.0e-5
        ):
            raise ValueError("supplied endpoint projector disagrees with observation state")
        return _ResolvedConsistency(
            semantics=(
                "rho0_population_orthogonal_complement_projector"
                if rho == 0.0
                else "rho1_context_orthogonal_complement_projector"
            ),
            projector=expected,
            use_psd_precision=False,
        )

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

    def _precision_matrices(
        self,
        state: PopulationObservationState,
        *,
        normalize_by_residual_dimension: bool,
    ) -> Tensor:
        """Materialize the state's PSD channel precision as ``(B,C,C)`` or
        ``(B,L,C,C)`` without treating it as an orthogonal projector.
        """

        dimensions = (
            state.residual_dimensions()
            if normalize_by_residual_dimension
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
        raise AssertionError(f"unsupported precision kind: {kind}")

    def _effective_precision(self, state: PopulationObservationState) -> Tensor:
        return self._precision_matrices(
            state,
            normalize_by_residual_dimension=(
                self.inference.stability.normalize_by_residual_dimension
            ),
        )

    def _time_precision(self, state: PopulationObservationState) -> Tensor:
        precision = self._precision_matrices(
            state,
            normalize_by_residual_dimension=False,
        )
        if precision.ndim == 3:
            precision = precision[:, None, :, :].expand(
                -1, state.observation.shape[-1], -1, -1
            )
        return precision

    def _precision_residual(
        self,
        value: Tensor,
        state: PopulationObservationState,
    ) -> float:
        residual = (
            value - state.observation
        ) * state.valid_time_mask[:, None, :].to(dtype=value.dtype)
        precision = self._time_precision(state)
        quadratic = torch.einsum(
            "bcl,blcd,bdl->", residual, precision, residual
        ).clamp_min(0.0)
        dimensions = state.residual_dimensions().sum().clamp_min(1.0)
        return float(torch.sqrt(quadratic / dimensions))

    def _psd_precision_consistency(
        self,
        value: Tensor,
        state: PopulationObservationState,
    ) -> Tensor:
        """Apply one normalized PSD ``W_rho`` residual update.

        This is intentionally not called a projection.  Dividing each frame's
        PSD matrix by its largest eigenvalue gives a non-expansive correction
        with eigenvalues in ``[0,1]`` while preserving the eigengeometry of the
        actual interpolated precision.
        """

        precision = self._time_precision(state)
        largest = torch.linalg.eigvalsh(precision).amax(dim=-1)
        scale = torch.where(
            largest > 0.0,
            largest.clamp_min(torch.finfo(largest.dtype).eps).reciprocal(),
            torch.zeros_like(largest),
        )
        gain = precision * scale[:, :, None, None]
        residual = (state.observation - value).transpose(1, 2)
        update = torch.einsum("blcd,bld->blc", gain, residual).transpose(1, 2)
        mask = state.valid_time_mask[:, None, :].to(dtype=value.dtype)
        return (value + update) * mask

    def _psd_precision_proximal(
        self,
        value: Tensor,
        state: PopulationObservationState,
        strength: float,
    ) -> Tensor:
        """Solve the quadratic proximal step induced by PSD ``W_rho``."""

        if not 0.0 < strength < float("inf"):
            raise ValueError("PSD proximal strength must be finite and positive")
        precision = self._time_precision(state)
        batch, length, channels, _ = precision.shape
        identity = torch.eye(
            channels, device=value.device, dtype=value.dtype
        ).reshape(1, 1, channels, channels)
        residual = (state.observation - value).transpose(1, 2)
        rhs = strength * torch.einsum("blcd,bld->blc", precision, residual)
        update = torch.linalg.solve(
            identity.expand(batch, length, -1, -1) + strength * precision,
            rhs.unsqueeze(-1),
        ).squeeze(-1).transpose(1, 2)
        mask = state.valid_time_mask[:, None, :].to(dtype=value.dtype)
        return (value + update) * mask

    def _trace_transform(
        self,
        *,
        mechanism: SamplerMechanism,
        state: PopulationObservationState,
        consistency: _ResolvedConsistency,
        proximal_strength: float,
        trace: list[GuidanceStepTrace],
    ):
        def transform(value: Tensor, _timestep: int, is_final: bool) -> Tensor:
            before = value
            projector = consistency.projector
            p_before, q_before = self._pq_residuals(before, state, projector)
            precision_before = self._precision_residual(before, state)
            apply_hard = (
                mechanism == SamplerMechanism.M3
                or (mechanism == SamplerMechanism.M2 and is_final)
            )
            if apply_hard:
                if consistency.use_psd_precision:
                    after = self._psd_precision_consistency(before, state)
                elif projector is not None:
                    after = self._hard_q(before, state, projector)
                else:
                    raise ValueError(f"{mechanism.name} requires consistency geometry")
            elif mechanism == SamplerMechanism.M4:
                if consistency.use_psd_precision:
                    after = self._psd_precision_proximal(
                        before, state, proximal_strength
                    )
                elif projector is not None:
                    after = self._proximal_q(
                        before, state, projector, proximal_strength
                    )
                else:
                    raise ValueError("M4 requires consistency geometry")
            else:
                after = before
            p_after, q_after = self._pq_residuals(after, state, projector)
            precision_after = self._precision_residual(after, state)
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
                consistency_semantics=consistency.semantics,
                precision_residual_before=precision_before,
                precision_residual_after=precision_after,
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
        consistency_semantics: str,
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
        precision_before = self._precision_residual(prior_clean, state)
        precision_after = self._precision_residual(restored, state)
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
            guided_score_l2=float(torch.linalg.vector_norm(score)),
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
            consistency_semantics=consistency_semantics,
            precision_residual_before=precision_before,
            precision_residual_after=precision_after,
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
        consistency = self._resolve_consistency(projector, state)
        projection = consistency.projector
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
                    consistency_semantics="psd_quadratic_proximal_Wrho",
                    clipped_fraction=float((factor < 1.0).float().mean()),
                )
            )
        else:
            if mechanism in (SamplerMechanism.M2, SamplerMechanism.M3, SamplerMechanism.M4):
                if projection is None and not consistency.use_psd_precision:
                    raise ValueError(f"{mechanism.name} requires consistency geometry")
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
                consistency=consistency,
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
            consistency_semantics=(
                "psd_quadratic_proximal_Wrho"
                if mechanism == SamplerMechanism.M5
                else consistency.semantics
            ),
        )
