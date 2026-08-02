"""Population/subject inference for artifact-latent estimators.

This module mixes mapped artifact predictions, never projectors.  It exposes no
query EOG, eye-tracking, label, outcome, or clean-target argument.  A zero-rho
call executes the population branch and returns before constructing a subject
context or making a subject sampler call.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import torch
from torch import Tensor

from .clean_prior import canonical_valid_time_mask


ContextRole = Literal["population", "subject"]
InferenceKind = Literal["deterministic", "diffusion"]


class DeterministicArtifactModel(Protocol):
    def __call__(
        self,
        observed: Tensor,
        *,
        full_transfer: Tensor,
        normalized_transfer: Tensor,
        transfer_scale: Tensor,
        singular_values: Tensor,
        rank: int,
        rho: float,
        calibration_duration_seconds: float,
        channel_mask: Tensor,
        valid_time_mask: Tensor | None,
    ) -> Tensor: ...


class DiffusionArtifactModel(Protocol):
    def posterior_mean(self, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class ArtifactInferenceContext:
    """One immutable tensor context with outer-training latent normalization."""

    context_id: str
    role: ContextRole
    full_transfer: Tensor
    normalized_transfer: Tensor
    transfer_scale: Tensor
    singular_values: Tensor
    rank: int
    calibration_duration_seconds: float
    latent_mean: Tensor
    latent_standard_deviation: Tensor
    subspace_basis: Tensor

    def __post_init__(self) -> None:
        if not self.context_id.strip():
            raise ValueError("artifact inference context_id is empty")
        if self.role not in {"population", "subject"}:
            raise ValueError("artifact inference context role is invalid")
        tensors = {
            "full_transfer": self.full_transfer,
            "normalized_transfer": self.normalized_transfer,
            "transfer_scale": self.transfer_scale,
            "singular_values": self.singular_values,
            "latent_mean": self.latent_mean,
            "latent_standard_deviation": self.latent_standard_deviation,
            "subspace_basis": self.subspace_basis,
        }
        for name, value in tensors.items():
            if not isinstance(value, Tensor) or not value.dtype.is_floating_point:
                raise TypeError(f"{name} must be a floating-point tensor")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"{name} contains NaN/Inf")
        full = self.full_transfer
        normalized = self.normalized_transfer
        if full.ndim != 2 or normalized.shape != full.shape:
            raise ValueError("transfer tensors must share shape (C,E)")
        channels, latent_channels = full.shape
        if self.transfer_scale.shape != (latent_channels,):
            raise ValueError("transfer_scale must have one value per latent coordinate")
        if self.singular_values.shape != (min(full.shape),):
            raise ValueError("singular_values do not match the transfer matrix")
        if self.latent_mean.shape != (latent_channels,) or (
            self.latent_standard_deviation.shape != (latent_channels,)
        ):
            raise ValueError("latent normalization must have shape (E,)")
        if bool((self.transfer_scale <= 0.0).any()) or bool(
            (self.latent_standard_deviation <= 0.0).any()
        ):
            raise ValueError("transfer and latent scales must be positive")
        rank = int(self.rank)
        if isinstance(self.rank, bool) or rank != self.rank:
            raise ValueError("rank must be an integer")
        if not 1 <= rank <= min(full.shape):
            raise ValueError("rank is outside the transfer dimensions")
        if self.subspace_basis.shape != (channels, rank):
            raise ValueError("subspace_basis must have shape (C,rank)")
        basis = self.subspace_basis.double()
        if not torch.allclose(
            basis.T @ basis,
            torch.eye(rank, dtype=basis.dtype, device=basis.device),
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise ValueError("subspace_basis must be orthonormal")
        expected_full = normalized * self.transfer_scale[None, :]
        if not torch.allclose(
            full.double(), expected_full.double(), rtol=1.0e-7, atol=1.0e-9
        ):
            raise ValueError("full C must equal normalized C times transfer_scale")
        singular = self.singular_values
        if bool((singular < 0.0).any()) or (
            singular.numel() > 1
            and bool((singular[1:] > singular[:-1]).any())
        ):
            raise ValueError("singular values must be nonnegative and nonincreasing")
        duration = float(self.calibration_duration_seconds)
        if not math.isfinite(duration) or duration < 0.0:
            raise ValueError("calibration duration must be finite and nonnegative")

    @property
    def projector(self) -> Tensor:
        return self.subspace_basis @ self.subspace_basis.T


@dataclass(frozen=True)
class ArtifactBranchInference:
    context_id: str
    standardized_latent: Tensor
    mapped_delta: Tensor
    stochastic_standard_deviation: Tensor | None
    sampler_calls: int
    network_calls: int


@dataclass(frozen=True)
class PopulationSubjectRestoration:
    inference_kind: InferenceKind
    branch: Literal["population", "mixed", "subject"]
    rho: float
    restored: Tensor
    mixed_delta: Tensor
    population: ArtifactBranchInference
    subject: ArtifactBranchInference | None
    geometry_basis: Tensor
    geometry_projector: Tensor
    complement_relative_error: float
    subject_context_constructed: bool
    shared_observation: bool
    shared_model_weights: bool
    shared_latent_normalization: bool
    shared_diffusion_seeds: bool | None


SubjectContextFactory = Callable[[], ArtifactInferenceContext]


def _rho(value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("rho must be finite and lie in [0,1]")
    return result


def _output_mask(
    observed: Tensor,
    channel_mask: Tensor,
    valid_time_mask: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    if observed.ndim != 3 or not observed.dtype.is_floating_point:
        raise ValueError("observed must have floating shape (B,C,T)")
    if not bool(torch.isfinite(observed).all()):
        raise ValueError("observed contains NaN/Inf")
    channels = torch.as_tensor(channel_mask, device=observed.device)
    if channels.shape == (observed.shape[1],):
        channels = channels.unsqueeze(0).expand(observed.shape[0], -1)
    if channels.shape != (observed.shape[0], observed.shape[1]):
        raise ValueError("channel_mask must have shape (C,) or (B,C)")
    if channels.dtype != torch.bool:
        if not bool(((channels == 0) | (channels == 1)).all()):
            raise ValueError("numeric channel_mask must contain 0/1")
        channels = channels.bool()
    time = canonical_valid_time_mask(observed, valid_time_mask)
    output = channels[:, :, None].to(dtype=observed.dtype) * time.to(
        dtype=observed.dtype
    )
    return channels, time, output


def _context_on_observation(
    context: ArtifactInferenceContext,
    observed: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    if context.full_transfer.shape[0] != observed.shape[1]:
        raise ValueError("context EEG channels differ from observed")
    normalized = context.normalized_transfer.to(
        device=observed.device, dtype=observed.dtype
    )
    mean = context.latent_mean.to(device=observed.device, dtype=observed.dtype)
    standard_deviation = context.latent_standard_deviation.to(
        device=observed.device, dtype=observed.dtype
    )
    return normalized, mean, standard_deviation


def _same_latent_normalization(
    population: ArtifactInferenceContext,
    subject: ArtifactInferenceContext,
) -> bool:
    return bool(
        population.latent_mean.shape == subject.latent_mean.shape
        and population.latent_standard_deviation.shape
        == subject.latent_standard_deviation.shape
        and torch.equal(
            population.latent_mean.cpu(), subject.latent_mean.cpu()
        )
        and torch.equal(
            population.latent_standard_deviation.cpu(),
            subject.latent_standard_deviation.cpu(),
        )
    )


def _validate_subject_contexts(
    population: ArtifactInferenceContext,
    subject: ArtifactInferenceContext,
) -> None:
    if population.role != "population" or subject.role != "subject":
        raise ValueError("population/subject context roles are incorrect")
    if population.full_transfer.shape != subject.full_transfer.shape:
        raise ValueError("population and subject transfer shapes differ")
    if not _same_latent_normalization(population, subject):
        raise ValueError("population and subject latent normalization differs")


def _latent_to_delta(
    standardized_latent: Tensor,
    context: ArtifactInferenceContext,
    observed: Tensor,
    output_mask: Tensor,
) -> Tensor:
    normalized, mean, standard_deviation = _context_on_observation(
        context, observed
    )
    expected = (
        observed.shape[0],
        normalized.shape[1],
        observed.shape[-1],
    )
    if standardized_latent.shape != expected or not bool(
        torch.isfinite(standardized_latent).all()
    ):
        raise ValueError("standardized artifact latent has invalid shape or values")
    latent = (
        standardized_latent * standard_deviation[None, :, None]
        + mean[None, :, None]
    )
    return torch.einsum("ce,bet->bct", normalized, latent) * output_mask


def _union_basis(
    population: ArtifactInferenceContext,
    subject: ArtifactInferenceContext | None,
    observed: Tensor,
) -> Tensor:
    first = population.subspace_basis.to(
        device=observed.device, dtype=observed.dtype
    )
    if subject is None:
        return first
    second = subject.subspace_basis.to(
        device=observed.device, dtype=observed.dtype
    )
    joined = torch.cat((first, second), dim=1)
    basis, singular, _ = torch.linalg.svd(joined, full_matrices=False)
    tolerance = (
        max(joined.shape)
        * torch.finfo(joined.dtype).eps
        * singular[0]
    )
    rank = int((singular > tolerance).sum().item())
    if rank < 1:
        raise ValueError("population/subject union span has zero rank")
    return basis[:, :rank]


def _geometry_error(delta: Tensor, basis: Tensor) -> tuple[Tensor, Tensor, float]:
    projector = basis @ basis.T
    complement = torch.eye(
        projector.shape[0], device=projector.device, dtype=projector.dtype
    ) - projector
    residual = torch.einsum("cd,bdt->bct", complement, delta)
    denominator = torch.linalg.vector_norm(delta)
    numerator = torch.linalg.vector_norm(residual)
    if float(denominator) == 0.0:
        relative = 0.0 if float(numerator) == 0.0 else float("inf")
    else:
        relative = float((numerator / denominator).detach().cpu())
    return basis, projector, relative


def _finish(
    *,
    kind: InferenceKind,
    observed: Tensor,
    output_mask: Tensor,
    rho: float,
    population_context: ArtifactInferenceContext,
    population: ArtifactBranchInference,
    subject_context: ArtifactInferenceContext | None,
    subject: ArtifactBranchInference | None,
    shared_diffusion_seeds: bool | None,
) -> PopulationSubjectRestoration:
    if rho == 0.0:
        mixed_delta = population.mapped_delta
        branch: Literal["population", "mixed", "subject"] = "population"
    else:
        if subject is None or subject_context is None:
            raise AssertionError("nonzero rho completed without a subject branch")
        mixed_delta = (
            (1.0 - rho) * population.mapped_delta
            + rho * subject.mapped_delta
        )
        branch = "subject" if rho == 1.0 else "mixed"
    restored = (observed * output_mask - mixed_delta) * output_mask
    basis = _union_basis(population_context, subject_context, observed)
    basis, projector, error = _geometry_error(mixed_delta, basis)
    return PopulationSubjectRestoration(
        inference_kind=kind,
        branch=branch,
        rho=rho,
        restored=restored,
        mixed_delta=mixed_delta,
        population=population,
        subject=subject,
        geometry_basis=basis,
        geometry_projector=projector,
        complement_relative_error=error,
        subject_context_constructed=subject_context is not None,
        shared_observation=True,
        shared_model_weights=True,
        shared_latent_normalization=(
            True
            if subject_context is None
            else _same_latent_normalization(population_context, subject_context)
        ),
        shared_diffusion_seeds=shared_diffusion_seeds,
    )


def _deterministic_branch(
    model: DeterministicArtifactModel,
    observed: Tensor,
    context: ArtifactInferenceContext,
    *,
    rho: float,
    channel_mask: Tensor,
    valid_time_mask: Tensor | None,
    output_mask: Tensor,
) -> ArtifactBranchInference:
    standardized = model(
        observed,
        full_transfer=context.full_transfer,
        normalized_transfer=context.normalized_transfer,
        transfer_scale=context.transfer_scale,
        singular_values=context.singular_values,
        rank=context.rank,
        rho=rho,
        calibration_duration_seconds=context.calibration_duration_seconds,
        channel_mask=channel_mask,
        valid_time_mask=valid_time_mask,
    )
    delta = _latent_to_delta(standardized, context, observed, output_mask)
    return ArtifactBranchInference(
        context_id=context.context_id,
        standardized_latent=standardized,
        mapped_delta=delta,
        stochastic_standard_deviation=None,
        sampler_calls=1,
        network_calls=1,
    )


def deterministic_population_subject_restore(
    model: DeterministicArtifactModel,
    observed: Tensor,
    *,
    population_context: ArtifactInferenceContext,
    rho: float,
    subject_context_factory: SubjectContextFactory | None,
    channel_mask: Tensor,
    valid_time_mask: Tensor | None,
) -> PopulationSubjectRestoration:
    """Restore with one shared deterministic model and explicit delta mixing."""

    rho_value = _rho(rho)
    if population_context.role != "population":
        raise ValueError("population_context has the wrong role")
    _, _, output_mask = _output_mask(observed, channel_mask, valid_time_mask)
    population = _deterministic_branch(
        model,
        observed,
        population_context,
        rho=rho_value,
        channel_mask=channel_mask,
        valid_time_mask=valid_time_mask,
        output_mask=output_mask,
    )
    if rho_value == 0.0:
        return _finish(
            kind="deterministic",
            observed=observed,
            output_mask=output_mask,
            rho=0.0,
            population_context=population_context,
            population=population,
            subject_context=None,
            subject=None,
            shared_diffusion_seeds=None,
        )
    if subject_context_factory is None or not callable(subject_context_factory):
        raise ValueError("nonzero rho requires a lazy subject_context_factory")
    subject_context = subject_context_factory()
    _validate_subject_contexts(population_context, subject_context)
    subject = _deterministic_branch(
        model,
        observed,
        subject_context,
        rho=rho_value,
        channel_mask=channel_mask,
        valid_time_mask=valid_time_mask,
        output_mask=output_mask,
    )
    return _finish(
        kind="deterministic",
        observed=observed,
        output_mask=output_mask,
        rho=rho_value,
        population_context=population_context,
        population=population,
        subject_context=subject_context,
        subject=subject,
        shared_diffusion_seeds=None,
    )


def _diffusion_branch(
    model: DiffusionArtifactModel,
    observed: Tensor,
    context: ArtifactInferenceContext,
    *,
    rho: float,
    channel_mask: Tensor,
    valid_time_mask: Tensor | None,
    output_mask: Tensor,
    sample_seeds: tuple[int, ...],
    ddim_steps: int,
    record_trajectory: bool,
) -> ArtifactBranchInference:
    posterior = model.posterior_mean(
        observed=observed,
        full_transfer=context.full_transfer,
        normalized_transfer=context.normalized_transfer,
        transfer_scale=context.transfer_scale,
        singular_values=context.singular_values,
        rank=context.rank,
        rho=rho,
        calibration_duration_seconds=context.calibration_duration_seconds,
        channel_mask=channel_mask,
        latent_mean=context.latent_mean,
        latent_standard_deviation=context.latent_standard_deviation,
        valid_time_mask=valid_time_mask,
        sample_seeds=sample_seeds,
        ddim_steps=ddim_steps,
        record_trajectory=record_trajectory,
    )
    standardized = posterior.standardized_latent_mean
    delta = _latent_to_delta(standardized, context, observed, output_mask)
    returned_delta = posterior.correction
    expected_delta = (observed.shape[0], observed.shape[1], observed.shape[-1])
    if returned_delta.shape != expected_delta or not bool(
        torch.isfinite(returned_delta).all()
    ):
        raise ValueError("diffusion branch returned an invalid mapped delta")
    returned_delta = returned_delta * output_mask
    if not torch.allclose(
        delta,
        returned_delta,
        rtol=1.0e-5,
        atol=1.0e-6,
    ):
        raise ValueError(
            "diffusion correction differs from the shared latent normalization"
        )
    if getattr(posterior, "sample_count", None) != 8:
        raise ValueError("diffusion branch did not return the frozen K=8 posterior")
    stochastic_standard_deviation = (
        posterior.standardized_latent_standard_deviation
    )
    if stochastic_standard_deviation.shape != standardized.shape or not bool(
        torch.isfinite(stochastic_standard_deviation).all()
    ):
        raise ValueError("diffusion branch returned an invalid posterior deviation")
    return ArtifactBranchInference(
        context_id=context.context_id,
        standardized_latent=standardized,
        mapped_delta=delta,
        stochastic_standard_deviation=stochastic_standard_deviation,
        sampler_calls=1,
        network_calls=int(posterior.network_calls),
    )


def diffusion_population_subject_restore(
    model: DiffusionArtifactModel,
    observed: Tensor,
    *,
    population_context: ArtifactInferenceContext,
    rho: float,
    subject_context_factory: SubjectContextFactory | None,
    channel_mask: Tensor,
    valid_time_mask: Tensor | None,
    sample_seeds: Sequence[int],
    ddim_steps: int,
    record_trajectory: bool = False,
) -> PopulationSubjectRestoration:
    """Restore with same-seed K=8 population/subject diffusion branches."""

    rho_value = _rho(rho)
    if population_context.role != "population":
        raise ValueError("population_context has the wrong role")
    raw_seeds = tuple(sample_seeds)
    if any(
        isinstance(value, bool) or int(value) != value
        for value in raw_seeds
    ):
        raise ValueError("diffusion sample seeds must be integers")
    seeds = tuple(int(value) for value in raw_seeds)
    if len(seeds) != 8 or len(set(seeds)) != 8:
        raise ValueError("diffusion population/subject inference requires 8 unique seeds")
    if isinstance(ddim_steps, bool) or int(ddim_steps) != ddim_steps or ddim_steps < 1:
        raise ValueError("ddim_steps must be a positive integer")
    _, _, output_mask = _output_mask(observed, channel_mask, valid_time_mask)
    population = _diffusion_branch(
        model,
        observed,
        population_context,
        rho=rho_value,
        channel_mask=channel_mask,
        valid_time_mask=valid_time_mask,
        output_mask=output_mask,
        sample_seeds=seeds,
        ddim_steps=ddim_steps,
        record_trajectory=record_trajectory,
    )
    if rho_value == 0.0:
        return _finish(
            kind="diffusion",
            observed=observed,
            output_mask=output_mask,
            rho=0.0,
            population_context=population_context,
            population=population,
            subject_context=None,
            subject=None,
            shared_diffusion_seeds=True,
        )
    if subject_context_factory is None or not callable(subject_context_factory):
        raise ValueError("nonzero rho requires a lazy subject_context_factory")
    subject_context = subject_context_factory()
    _validate_subject_contexts(population_context, subject_context)
    subject = _diffusion_branch(
        model,
        observed,
        subject_context,
        rho=rho_value,
        channel_mask=channel_mask,
        valid_time_mask=valid_time_mask,
        output_mask=output_mask,
        sample_seeds=seeds,
        ddim_steps=ddim_steps,
        record_trajectory=record_trajectory,
    )
    return _finish(
        kind="diffusion",
        observed=observed,
        output_mask=output_mask,
        rho=rho_value,
        population_context=population_context,
        population=population,
        subject_context=subject_context,
        subject=subject,
        shared_diffusion_seeds=True,
    )


__all__ = [
    "ArtifactBranchInference",
    "ArtifactInferenceContext",
    "PopulationSubjectRestoration",
    "deterministic_population_subject_restore",
    "diffusion_population_subject_restore",
]
