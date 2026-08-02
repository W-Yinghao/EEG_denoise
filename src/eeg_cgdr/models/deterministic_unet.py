"""Task-matched deterministic multichannel restoration baseline.

The baseline receives exactly the deployment-time fields shared with the
repaired sampler comparison: the observed EEG, one frozen operator projector,
the framewise external-reference attenuation, and the valid-time mask.  It has
no participant identifier, query clean target, or query outcome input.  The
complete projector is exposed to the network (not only its action on one
observation), so this is an information-matched operator-conditioned baseline.

The backbone is the same masked three-level multichannel U-Net used by the
diffusion prior.  A constant time embedding is used as a learned global bias;
there is no diffusion timestep or random input, so inference is deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from saddpm.models.config import ModelConfig
from saddpm.models.unet1d import UNet1D

from .clean_prior import canonical_valid_time_mask


@dataclass(frozen=True)
class DeterministicUNetConfig:
    eeg_channels: int = 19
    signal_length: int = 512
    base_channels: int = 64
    channel_mults: tuple[int, int, int] = (1, 2, 4)
    num_res_blocks: int = 2
    groupnorm_groups: int = 8
    dropout: float = 0.05
    time_sinusoidal_dim: int = 128
    time_embed_dim: int = 512
    attention_length: int = 64
    attention_heads: int = 4
    residual_output: bool = True

    def __post_init__(self) -> None:
        if self.eeg_channels < 2:
            raise ValueError("deterministic scientific baseline must be multichannel")
        if self.signal_length < 8 or self.signal_length % 8:
            raise ValueError("signal_length must be a positive multiple of eight")
        if tuple(self.channel_mults) != (1, 2, 4):
            raise ValueError("the matched backbone requires channel_mults [1,2,4]")
        if not self.residual_output:
            raise ValueError("the frozen task-matched model uses residual output")


def _projector_batch(observed: Tensor, projector: Tensor) -> Tensor:
    value = torch.as_tensor(
        projector,
        device=observed.device,
        dtype=observed.dtype,
    ).detach()
    batch, channels, _ = observed.shape
    if value.shape == (channels, channels):
        value = value.unsqueeze(0).expand(batch, -1, -1)
    if value.shape != (batch, channels, channels):
        raise ValueError("projector must have shape (C,C) or (B,C,C)")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("projector contains non-finite values")
    # Scientific projectors are fitted in FP64 and stored as FP32 model inputs.
    # Keep their semantic checks outside CUDA autocast; FP16 matrix products can
    # otherwise fail a valid projector at the frozen FP32 tolerance.
    with torch.autocast(device_type=value.device.type, enabled=False):
        checked = value.float()
        if not torch.allclose(
            checked,
            checked.transpose(1, 2),
            atol=1.0e-6,
            rtol=1.0e-5,
        ):
            raise ValueError("projector must be symmetric")
        if not torch.allclose(
            checked @ checked,
            checked,
            atol=2.0e-5,
            rtol=2.0e-5,
        ):
            raise ValueError("projector must be idempotent")
    return value


def _frame_attenuation(observed: Tensor, attenuation: Tensor) -> Tensor:
    value = torch.as_tensor(
        attenuation,
        device=observed.device,
        dtype=observed.dtype,
    ).detach()
    if value.shape != (observed.shape[0], observed.shape[-1]):
        raise ValueError("attenuation must have shape (B,L)")
    if not bool(torch.isfinite(value).all()) or bool(
        ((value < 0.0) | (value > 1.0)).any()
    ):
        raise ValueError("attenuation must be finite and lie in [0,1]")
    return value


class TaskMatchedDeterministicUNet(nn.Module):
    """Deterministic residual U-Net with auditable matched inputs."""

    visible_input_fields = (
        "observed_query_eeg",
        "operator_projector",
        "framewise_external_eog_attenuation",
        "valid_time_mask",
    )

    def __init__(self, config: DeterministicUNetConfig) -> None:
        super().__init__()
        self.config = config
        # Features are y, Pi*y, framewise attenuation and every entry of Pi.
        # Broadcasting vec(Pi) over time preserves the full frozen operator
        # information; Pi*y alone would disclose less than the iterative state.
        backbone = ModelConfig(
            in_channels=(
                2 * config.eeg_channels
                + 1
                + config.eeg_channels * config.eeg_channels
            ),
            out_channels=config.eeg_channels,
            signal_length=config.signal_length,
            base_channels=config.base_channels,
            channel_mults=list(config.channel_mults),
            num_res_blocks=config.num_res_blocks,
            groupnorm_groups=config.groupnorm_groups,
            dropout=config.dropout,
            time_sinusoidal_dim=config.time_sinusoidal_dim,
            time_embed_dim=config.time_embed_dim,
            attention_length=config.attention_length,
            attention_heads=config.attention_heads,
        )
        self.unet = UNet1D(backbone, subject_conditioned=False)

    def forward(
        self,
        observed: Tensor,
        *,
        projector: Tensor,
        attenuation: Tensor,
        valid_time_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if observed.ndim != 3 or observed.shape[1] != self.config.eeg_channels:
            raise ValueError("observed EEG must have shape (B,C,L) matching montage")
        if observed.shape[-1] != self.config.signal_length:
            raise ValueError("observed EEG window length differs from model config")
        if not observed.dtype.is_floating_point or not bool(torch.isfinite(observed).all()):
            raise ValueError("observed EEG must be finite floating point")
        mask = canonical_valid_time_mask(observed, valid_time_mask)
        mask_float = mask.to(dtype=observed.dtype)
        observed = observed * mask_float
        projection = _projector_batch(observed, projector)
        attenuation_value = _frame_attenuation(observed, attenuation) * mask[:, 0, :]
        projected = torch.einsum("bij,bjl->bil", projection, observed)
        operator_features = projection.flatten(start_dim=1)[:, :, None].expand(
            -1, -1, observed.shape[-1]
        )
        features = torch.cat(
            [
                observed,
                projected,
                attenuation_value[:, None, :],
                operator_features,
            ],
            dim=1,
        ) * mask_float
        constant_condition = torch.zeros(
            observed.shape[0], device=observed.device, dtype=torch.long
        )
        residual = self.unet(
            features,
            constant_condition,
            valid_time_mask=mask,
        )
        return (observed + residual) * mask_float

    def task_loss(
        self,
        observed: Tensor,
        clean_target: Tensor,
        *,
        projector: Tensor,
        attenuation: Tensor,
        valid_time_mask: Optional[Tensor] = None,
        parallel_weight: float = 1.0,
        perpendicular_weight: float = 1.0,
        derivative_weight: float = 0.1,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if clean_target.shape != observed.shape:
            raise ValueError("clean target and observation must have identical shapes")
        if not clean_target.dtype.is_floating_point or not bool(
            torch.isfinite(clean_target).all()
        ):
            raise ValueError("clean target must be finite floating point")
        for name, value in (
            ("parallel_weight", parallel_weight),
            ("perpendicular_weight", perpendicular_weight),
            ("derivative_weight", derivative_weight),
        ):
            if not 0.0 <= float(value) < float("inf"):
                raise ValueError(f"{name} must be finite and non-negative")
        mask = canonical_valid_time_mask(observed, valid_time_mask)
        mask_float = mask.to(dtype=observed.dtype)
        restored = self(
            observed,
            projector=projector,
            attenuation=attenuation,
            valid_time_mask=mask,
        )
        target = clean_target * mask_float
        error = (restored - target) * mask_float
        projection = _projector_batch(observed, projector)
        parallel = torch.einsum("bij,bjl->bil", projection, error)
        perpendicular = error - parallel
        denominator = (mask_float.sum() * observed.shape[1]).clamp_min(1.0)
        parallel_mse = parallel.square().sum() / denominator
        perpendicular_mse = perpendicular.square().sum() / denominator
        derivative_mask = mask_float[:, :, 1:] * mask_float[:, :, :-1]
        derivative_error = (error[:, :, 1:] - error[:, :, :-1]) * derivative_mask
        derivative_denominator = (
            derivative_mask.sum() * observed.shape[1]
        ).clamp_min(1.0)
        derivative_mse = derivative_error.square().sum() / derivative_denominator
        loss = (
            float(parallel_weight) * parallel_mse
            + float(perpendicular_weight) * perpendicular_mse
            + float(derivative_weight) * derivative_mse
        )
        return loss, {
            "parallel_mse": parallel_mse.detach(),
            "perpendicular_mse": perpendicular_mse.detach(),
            "derivative_mse": derivative_mse.detach(),
        }


__all__ = ["DeterministicUNetConfig", "TaskMatchedDeterministicUNet"]
