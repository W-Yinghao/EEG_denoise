"""Support-calibrated artifact-subspace diffusion.

The support-derived orthonormal basis is not a soft subject feature: it
defines the two-dimensional query coordinates, every reverse-step condition,
and the final EEG correction.  Query EOG, labels, outcomes, and participant
identifiers are deliberately absent from the public inference API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Callable
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn

from saddpm.models.config import ModelConfig
from saddpm.models.unet1d import UNet1D

from .artifact_latent_diffusion import cosine_alpha_bar
from .clean_prior import canonical_valid_time_mask


def aligned_artifact_basis(
    transfer: np.ndarray,
    population_basis: np.ndarray | None = None,
    *,
    rank: int = 2,
    relative_rank_tolerance: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a thin support basis aligned to the training-only population basis."""

    value = np.asarray(transfer, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] < rank or value.shape[1] < 1:
        raise ValueError("transfer must be a finite C-by-E matrix")
    if rank != 2 or not np.isfinite(value).all():
        raise ValueError("the frozen artifact-subspace rank is two")
    left, singular, _ = np.linalg.svd(value, full_matrices=False)
    available = min(rank, left.shape[1])
    basis = np.zeros((value.shape[0], rank), dtype=np.float64)
    basis[:, :available] = left[:, :available]
    # Deterministically complete a rank-deficient thin basis.  The added
    # coordinate is masked and therefore cannot create a correction.
    if available < rank:
        for column in np.eye(value.shape[0], dtype=np.float64).T:
            residual = column - basis[:, :available] @ (basis[:, :available].T @ column)
            norm = np.linalg.norm(residual)
            if norm > 1.0e-8:
                basis[:, available] = residual / norm
                available += 1
                if available == rank:
                    break
    singular_values = np.zeros(rank, dtype=np.float64)
    singular_values[: min(rank, singular.size)] = singular[:rank]
    threshold = max(float(singular_values[0]) * relative_rank_tolerance, np.finfo(np.float64).eps)
    rank_mask = singular_values > threshold
    if population_basis is not None:
        reference = np.asarray(population_basis, dtype=np.float64)
        if reference.shape != basis.shape or not np.isfinite(reference).all():
            raise ValueError("population basis differs from the exact compatibility cell")
        if bool(rank_mask.all()):
            p, _, qt = np.linalg.svd(basis.T @ reference, full_matrices=False)
            basis = basis @ (p @ qt)
        else:
            # Do not rotate a masked coordinate into an active coordinate.
            # Rank-one support receives only a deterministic sign alignment.
            if float(basis[:, 0] @ reference[:, 0]) < 0.0:
                basis[:, 0] *= -1.0
    if not np.allclose(basis.T @ basis, np.eye(rank), atol=1.0e-8):
        raise AssertionError("aligned artifact basis is not orthonormal")
    return basis.astype(np.float32), singular_values.astype(np.float32), rank_mask.astype(bool)


def batched_aligned_bases(
    transfers: np.ndarray,
    population_transfer: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit one training-only population basis and align every support basis."""

    population, _, _ = aligned_artifact_basis(population_transfer)
    bases, singular, masks = [], [], []
    for transfer in np.asarray(transfers):
        basis, values, mask = aligned_artifact_basis(transfer, population)
        bases.append(basis)
        singular.append(values)
        masks.append(mask)
    return population, np.stack(bases), np.stack(singular), np.stack(masks)


def bounded_subspace_target(
    contamination: np.ndarray,
    bases: np.ndarray,
    tau: np.ndarray,
) -> np.ndarray:
    """Project a training artifact into the frozen bounded subspace target."""

    artifact = np.asarray(contamination, dtype=np.float64)
    operator = np.asarray(bases, dtype=np.float64)
    scale = np.asarray(tau, dtype=np.float64)
    if artifact.ndim != 3 or operator.shape != (artifact.shape[0], artifact.shape[1], 2):
        raise ValueError("contamination/basis batch shapes differ")
    if scale.shape != (2,) or np.any(scale <= 0) or not np.isfinite(scale).all():
        raise ValueError("tau must contain two positive training-only scales")
    coefficient = np.einsum("ncr,nct->nrt", operator, artifact)
    return np.tanh(coefficient / scale[None, :, None]).astype(np.float32)


def training_tau(coefficients: np.ndarray, *, quantile: float = 0.995) -> np.ndarray:
    values = np.asarray(coefficients, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != 2 or not 0.5 < quantile < 1.0:
        raise ValueError("training coefficients/tau quantile are invalid")
    result = np.quantile(np.abs(values), quantile, axis=(0, 2))
    return np.maximum(result, 1.0e-3).astype(np.float32)


def artifact_coordinates(observed: Tensor, basis: Tensor, rank_mask: Tensor) -> Tensor:
    if observed.ndim != 3 or basis.shape != (observed.shape[0], observed.shape[1], 2):
        raise ValueError("observed/basis shapes differ")
    if rank_mask.shape != (observed.shape[0], 2):
        raise ValueError("rank mask must have shape (B,2)")
    return torch.einsum("bcr,bct->brt", basis, observed) * rank_mask[:, :, None].to(observed.dtype)


def reconstruct_from_subspace(
    observed: Tensor,
    basis: Tensor,
    bounded_u: Tensor,
    tau: Tensor,
    rank_mask: Tensor,
    valid_time_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    """Map bounded coefficients through A and subtract from the observation."""

    mask = canonical_valid_time_mask(observed, valid_time_mask).to(observed.dtype)
    if bounded_u.shape != (observed.shape[0], 2, observed.shape[2]):
        raise ValueError("bounded coefficient shape differs from EEG")
    if basis.shape != (observed.shape[0], observed.shape[1], 2):
        raise ValueError("basis shape differs from EEG")
    scale = torch.as_tensor(tau, device=observed.device, dtype=observed.dtype)
    if scale.shape == (2,):
        scale = scale[None].expand(observed.shape[0], -1)
    if scale.shape != (observed.shape[0], 2) or bool((scale <= 0).any()):
        raise ValueError("tau must have shape (2,) or (B,2)")
    coefficient = bounded_u.clamp(-1.0, 1.0) * scale[:, :, None]
    coefficient = coefficient * rank_mask[:, :, None].to(coefficient.dtype) * mask
    correction = torch.einsum("bcr,brt->bct", basis, coefficient) * mask
    return (observed - correction) * mask, correction


def complement_consistency_error(observed: Tensor, restored: Tensor, basis: Tensor, valid: Tensor) -> Tensor:
    mask = canonical_valid_time_mask(observed, valid).to(observed.dtype)
    residual = restored - observed
    parallel = torch.einsum("bcr,brt->bct", basis, torch.einsum("bcr,bct->brt", basis, residual))
    complement = (residual - parallel) * mask
    return torch.linalg.vector_norm(complement) / torch.linalg.vector_norm(observed * mask).clamp_min(1.0e-12)


def union_span_consistency_error(
    observed: Tensor,
    restored: Tensor,
    population_basis: Tensor,
    subject_basis: Tensor,
    valid: Tensor,
) -> Tensor:
    """Measure correction leakage outside span([A0,As])."""

    union = torch.cat((population_basis, subject_basis), dim=2)
    orthogonal = torch.linalg.qr(union, mode="reduced").Q
    return complement_consistency_error(observed, restored, orthogonal, valid)


def population_fallback_correction(
    population_correction: Tensor,
    rho: float,
    subject_correction_factory: Callable[[], Tensor],
) -> tuple[Tensor, bool]:
    """Lazily blend correction while guaranteeing the rho=0 short circuit."""

    reliability = float(rho)
    if not math.isfinite(reliability) or not 0.0 <= reliability <= 1.0:
        raise ValueError("support reliability must lie in [0,1]")
    if reliability == 0.0:
        return population_correction, False
    subject = subject_correction_factory()
    if subject.shape != population_correction.shape:
        raise ValueError("population/subject corrections differ in shape")
    return (1.0 - reliability) * population_correction + reliability * subject, True


@dataclass(frozen=True)
class ArtifactSubspaceConfig:
    eeg_channels: int
    signal_length: int
    latent_channels: int = 2
    base_channels: int = 32
    num_timesteps: int = 1000
    min_snr_gamma: float = 5.0
    ddim_steps: int = 25
    posterior_samples: int = 8

    def __post_init__(self) -> None:
        if self.latent_channels != 2 or self.eeg_channels < 2:
            raise ValueError("artifact-subspace diffusion freezes rank two")
        if self.signal_length < 8 or self.signal_length % 8:
            raise ValueError("signal length must be a positive multiple of eight")
        if self.ddim_steps != 25 or self.posterior_samples != 8:
            raise ValueError("main inference freezes DDIM25 and K=8")
        _, alpha_bar = cosine_alpha_bar(self.num_timesteps)
        if float(alpha_bar[-1]) > 1.0e-4:
            raise ValueError("terminal alpha_bar is too large for noise initialization")


class _Condition(nn.Module):
    def __init__(self, config: ArtifactSubspaceConfig, *, diffusion: bool) -> None:
        super().__init__()
        self.config = config
        self.diffusion = diffusion
        # state(2, diffusion only), A^T y(2), rho(1), rank mask(2)
        inputs = (2 if diffusion else 0) + 5
        self.unet = UNet1D(
            ModelConfig(
                in_channels=inputs,
                out_channels=2,
                signal_length=config.signal_length,
                base_channels=config.base_channels,
                channel_mults=[1, 2, 4],
                num_res_blocks=2,
                groupnorm_groups=8,
                dropout=0.05,
                time_sinusoidal_dim=64,
                time_embed_dim=256,
                attention_length=64,
                attention_heads=4,
            ),
            subject_conditioned=False,
        )

    def forward(
        self,
        state: Tensor | None,
        timestep: Tensor,
        *,
        observed: Tensor,
        basis: Tensor,
        reliability: Tensor,
        rank_mask: Tensor,
        valid_time_mask: Tensor,
    ) -> Tensor:
        mask = canonical_valid_time_mask(observed, valid_time_mask)
        coordinates = artifact_coordinates(observed, basis, rank_mask)
        rho = reliability[:, None, None].to(observed).expand(-1, 1, observed.shape[2])
        rank = rank_mask[:, :, None].to(observed).expand(-1, -1, observed.shape[2])
        fields = [coordinates, rho, rank] if state is None else [state, coordinates, rho, rank]
        features = torch.cat(fields, dim=1) * mask.to(observed.dtype)
        return self.unet(features, timestep, valid_time_mask=mask) * mask.to(observed.dtype)


class DeterministicSubspaceEstimator(nn.Module):
    visible_input_fields = ("observed_EEG", "support_basis", "support_reliability", "rank_mask")
    forbidden_input_fields = ("query_EOG", "query_eye_tracking", "query_labels", "query_outcomes", "participant_ID")

    def __init__(self, config: ArtifactSubspaceConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = _Condition(config, diffusion=False)

    def forward(self, **condition: Tensor) -> Tensor:
        observed = condition["observed"]
        timestep = torch.zeros(observed.shape[0], device=observed.device, dtype=torch.long)
        return torch.tanh(self.backbone(None, timestep, **condition))


def _extract(values: Tensor, timestep: Tensor, ndim: int) -> Tensor:
    return values.gather(0, timestep).reshape(timestep.shape[0], *((1,) * (ndim - 1)))


class ArtifactSubspaceDiffusion(nn.Module):
    visible_input_fields = DeterministicSubspaceEstimator.visible_input_fields
    forbidden_input_fields = DeterministicSubspaceEstimator.forbidden_input_fields

    def __init__(self, config: ArtifactSubspaceConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = _Condition(config, diffusion=True)
        _, alpha_bar = cosine_alpha_bar(config.num_timesteps)
        self.register_buffer("alpha_bar", alpha_bar.float())

    def training_loss(
        self,
        target_u: Tensor,
        *,
        generator: torch.Generator,
        timestep: Tensor | None = None,
        noise: Tensor | None = None,
        **condition: Tensor,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        time_mask = canonical_valid_time_mask(
            condition["observed"], condition["valid_time_mask"]
        ).to(target_u.dtype)
        rank_mask = condition["rank_mask"][:, :, None].to(target_u.dtype)
        mask = time_mask * rank_mask
        batch = target_u.shape[0]
        if timestep is None:
            timestep = torch.randint(0, self.config.num_timesteps, (batch,), device=target_u.device, generator=generator)
        if noise is None:
            noise = torch.randn(target_u.shape, device=target_u.device, dtype=target_u.dtype, generator=generator)
        alpha = _extract(self.alpha_bar, timestep, target_u.ndim)
        x_t = (alpha.sqrt() * target_u + (1.0 - alpha).sqrt() * noise) * mask
        v = (alpha.sqrt() * noise - (1.0 - alpha).sqrt() * target_u) * mask
        predicted = self.backbone(x_t, timestep, **condition)
        squared = ((predicted - v) * mask).square().sum((1, 2)) / mask.sum(
            (1, 2)
        ).clamp_min(1.0)
        snr = alpha.flatten(1)[:, 0] / (1.0 - alpha.flatten(1)[:, 0]).clamp_min(1.0e-8)
        weight = torch.minimum(snr, torch.full_like(snr, self.config.min_snr_gamma)) / (snr + 1.0)
        predicted_x0 = alpha.sqrt() * x_t - (1.0 - alpha).sqrt() * predicted
        return (squared * weight).mean(), {
            "u_mse": (
                ((predicted_x0.clamp(-1, 1) - target_u) * mask).square().sum()
                / mask.sum().clamp_min(1.0)
            ).detach(),
            "mean_timestep": timestep.float().mean().detach(),
        }

    def _sequence(self) -> tuple[int, ...]:
        values = torch.linspace(self.config.num_timesteps - 1, 0, self.config.ddim_steps, dtype=torch.float64).round().long().tolist()
        if len(set(values)) != self.config.ddim_steps:
            raise AssertionError("DDIM25 call count changed")
        return tuple(int(value) for value in values)

    @torch.no_grad()
    def sample(
        self,
        *,
        sample_seeds: Sequence[int] | None = None,
        initial_noise_bank: Tensor | None = None,
        record_trajectory: bool = False,
        **condition: Tensor,
    ) -> tuple[Tensor, Tensor, int, list[dict[str, float]]]:
        observed = condition["observed"]
        mask = canonical_valid_time_mask(observed, condition["valid_time_mask"]).to(observed.dtype)
        seeds = () if sample_seeds is None else tuple(int(value) for value in sample_seeds)
        if initial_noise_bank is None:
            if len(seeds) not in (1, 8, 32) or len(set(seeds)) != len(seeds):
                raise ValueError("inference permits primary K=8 and diagnostic K=1/K=32 unique seeds")
        else:
            if seeds:
                raise ValueError("provide sample_seeds or initial_noise_bank, not both")
            expected_tail = (observed.shape[0], 2, observed.shape[2])
            if initial_noise_bank.ndim != 4 or tuple(initial_noise_bank.shape[1:]) != expected_tail:
                raise ValueError("initial_noise_bank must have shape (K,B,2,T)")
            if initial_noise_bank.shape[0] not in (1, 8, 32) or not torch.isfinite(initial_noise_bank).all():
                raise ValueError("initial_noise_bank must contain finite K=1/8/32 states")
        sequence = self._sequence()
        samples, trace = [], []
        calls = 0
        count = len(seeds) if initial_noise_bank is None else int(initial_noise_bank.shape[0])
        for sample_index in range(count):
            if initial_noise_bank is None:
                generator = torch.Generator(device=observed.device).manual_seed(seeds[sample_index])
                state = torch.randn((observed.shape[0], 2, observed.shape[2]), device=observed.device, dtype=observed.dtype, generator=generator) * mask
            else:
                state = initial_noise_bank[sample_index].to(device=observed.device, dtype=observed.dtype) * mask
            previous = None
            for reverse_index, value in enumerate(sequence):
                timestep = torch.full((observed.shape[0],), value, device=observed.device, dtype=torch.long)
                predicted_v = self.backbone(state, timestep, **condition)
                alpha = self.alpha_bar[value]
                x0 = (alpha.sqrt() * state - (1.0 - alpha).sqrt() * predicted_v).clamp(-1.0, 1.0) * mask
                epsilon = (1.0 - alpha).sqrt() * state + alpha.sqrt() * predicted_v
                if reverse_index + 1 == len(sequence):
                    next_state = x0
                else:
                    next_alpha = self.alpha_bar[sequence[reverse_index + 1]]
                    next_state = (next_alpha.sqrt() * x0 + (1.0 - next_alpha).sqrt() * epsilon) * mask
                calls += 1
                if record_trajectory:
                    rms = float(torch.sqrt((next_state.square() * mask).mean()).cpu())
                    trace.append({"sample": float(sample_index), "reverse_index": float(reverse_index), "timestep": float(value), "u_rms": rms, "ratio": 1.0 if previous is None else rms / max(previous, 1.0e-12)})
                    previous = rms
                state = next_state
            samples.append(state)
        stacked = torch.stack(samples)
        return stacked.mean(0), stacked.std(0, unbiased=False), calls, trace


def window_noise_bank(
    participant_key: str,
    training_seed: int,
    absolute_window_indices: Sequence[int],
    *,
    posterior_samples: int,
    signal_length: int,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Create batch-size-invariant common noise indexed by window and sample."""

    if posterior_samples not in (1, 8, 32) or signal_length < 1:
        raise ValueError("noise bank requires K=1/8/32 and a positive signal length")
    indices = tuple(int(value) for value in absolute_window_indices)
    if len(indices) != len(set(indices)) or any(value < 0 for value in indices):
        raise ValueError("absolute window indices must be unique and non-negative")
    participant_base = participant_sample_seeds(participant_key, training_seed, count=posterior_samples)
    samples = []
    for sample_index, base in enumerate(participant_base):
        windows = []
        for window_index in indices:
            seed = (base + 104729 * window_index + 1009 * sample_index) % (2**63 - 1)
            generator = torch.Generator(device=device).manual_seed(seed)
            windows.append(torch.randn((2, signal_length), generator=generator, device=device, dtype=dtype))
        samples.append(torch.stack(windows))
    return torch.stack(samples)


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def participant_sample_seeds(participant_key: str, training_seed: int, *, count: int = 8) -> tuple[int, ...]:
    """Stable non-cryptographic participant-specific RNG streams."""

    accumulator = 2166136261
    for byte in participant_key.encode("utf-8"):
        accumulator = ((accumulator ^ byte) * 16777619) & 0xFFFFFFFF
    base = (int(training_seed) * 1000003 + accumulator) % (2**63 - 1000)
    return tuple(base + 97 * index for index in range(count))


__all__ = [
    "ArtifactSubspaceConfig", "ArtifactSubspaceDiffusion",
    "DeterministicSubspaceEstimator", "aligned_artifact_basis",
    "artifact_coordinates", "batched_aligned_bases", "bounded_subspace_target",
    "complement_consistency_error", "parameter_count", "participant_sample_seeds",
    "population_fallback_correction", "reconstruct_from_subspace", "training_tau",
    "union_span_consistency_error", "window_noise_bank",
]
