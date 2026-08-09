"""Raw-temporal support conditioned clean-EEG estimators.

The diffusion state is the full clean waveform.  Query EOG and participant IDs
are deliberately absent from this module's interface.  Sixteen disjoint EEG
support patches are encoded independently; no cross-patch positional embedding
is used, so their order is a set permutation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from saddpm.models.unet1d import Downsample1D, ResBlock1D, TimeEmbedding, Upsample1D
from .artifact_latent_diffusion import cosine_alpha_bar


@dataclass(frozen=True)
class RawSupportCleanConfig:
    channels: int = 3
    signal_length: int = 512
    valid_length: int = 500
    support_patches: int = 16
    support_patch_length: int = 500
    token_dim: int = 64
    base_channels: int = 64
    timesteps: int = 1000
    ddim_steps: int = 25
    posterior_samples: int = 8
    dropout: float = 0.05

    def __post_init__(self) -> None:
        if self.channels != 3:
            raise ValueError("BCI2b clean-waveform route freezes three EEG channels")
        if self.support_patches != 16 or self.support_patch_length != 500:
            raise ValueError("the frozen support set is 16 non-overlapping 2-second patches")
        if self.token_dim != 64 or self.ddim_steps != 25 or self.posterior_samples != 8:
            raise ValueError("token width and primary sampler panel are frozen")


class RawTemporalSupportEncoder(nn.Module):
    """Encode each EEG patch independently into one 64-D set token."""

    forbidden_fields = (
        "support_EOG", "query_EOG", "participant_ID", "session_ID",
        "artifact_label", "MI_label", "query_clean_target", "query_outcome",
    )

    def __init__(self, config: RawSupportCleanConfig) -> None:
        super().__init__()
        width = config.token_dim
        self.config = config
        self.network = nn.Sequential(
            nn.Conv1d(config.channels, 32, 9, stride=4, padding=4),
            nn.SiLU(),
            nn.Conv1d(32, width, 9, stride=4, padding=4),
            nn.SiLU(),
            nn.Conv1d(width, width, 5, stride=2, padding=2),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.norm = nn.LayerNorm(width)

    def forward(self, support_eeg: Tensor) -> Tensor:
        expected = (self.config.support_patches, self.config.channels, self.config.support_patch_length)
        if support_eeg.ndim != 4 or tuple(support_eeg.shape[1:]) != expected:
            raise ValueError(f"support_eeg must have shape (B,{expected[0]},{expected[1]},{expected[2]})")
        batch, patches, channels, length = support_eeg.shape
        token = self.network(support_eeg.reshape(batch * patches, channels, length)).squeeze(-1)
        return self.norm(token).reshape(batch, patches, -1)


class _CrossAttention(nn.Module):
    def __init__(self, channels: int, token_dim: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(channels)
        self.token_proj = nn.Linear(token_dim, channels)
        self.token_norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, 4, dropout=0.0, batch_first=True)
        self.output = nn.Sequential(nn.LayerNorm(channels), nn.Linear(channels, 4 * channels), nn.SiLU(), nn.Linear(4 * channels, channels))

    def forward(self, feature: Tensor, tokens: Tensor) -> Tensor:
        query = self.query_norm(feature.transpose(1, 2))
        context = self.token_norm(self.token_proj(tokens))
        attended, _ = self.attention(query, context, context, need_weights=False)
        hidden = query + attended
        return (hidden + self.output(hidden)).transpose(1, 2)


class RawSupportCleanUNet(nn.Module):
    """Shared DET/DIFF backbone with cross-attention at two fixed scales."""

    visible_fields = ("corrupted_EEG", "support_EEG_patches", "diffusion_state", "timestep")
    forbidden_fields = RawTemporalSupportEncoder.forbidden_fields

    def __init__(self, config: RawSupportCleanConfig) -> None:
        super().__init__()
        self.config = config
        width, tdim = config.base_channels, 256
        self.support_encoder = RawTemporalSupportEncoder(config)
        self.time_embed = TimeEmbedding(64, tdim)
        self.stem = nn.Conv1d(2 * config.channels, width, 5, padding=2)
        self.enc0 = ResBlock1D(width, width, tdim, 8, config.dropout)
        self.down0 = Downsample1D(width)
        self.enc1 = ResBlock1D(width, width, tdim, 8, config.dropout)
        self.cross1 = _CrossAttention(width, config.token_dim)
        self.down1 = Downsample1D(width)
        self.enc2 = ResBlock1D(width, 2 * width, tdim, 8, config.dropout)
        self.cross2 = _CrossAttention(2 * width, config.token_dim)
        self.mid = ResBlock1D(2 * width, 2 * width, tdim, 8, config.dropout)
        self.up1 = Upsample1D(2 * width)
        self.dec1 = ResBlock1D(3 * width, width, tdim, 8, config.dropout)
        self.up0 = Upsample1D(width)
        self.dec0 = ResBlock1D(2 * width, width, tdim, 8, config.dropout)
        self.out = nn.Sequential(nn.GroupNorm(8, width), nn.SiLU(), nn.Conv1d(width, config.channels, 3, padding=1))

    def encode_support(self, support_eeg: Tensor) -> Tensor:
        return self.support_encoder(support_eeg)

    def forward_with_tokens(self, state: Tensor, timestep: Tensor, *, query_y: Tensor, tokens: Tensor) -> Tensor:
        temb = self.time_embed(timestep)
        e0 = self.enc0(self.stem(torch.cat((state, query_y), dim=1)), temb)
        e1 = self.cross1(self.enc1(self.down0(e0), temb), tokens)
        e2 = self.cross2(self.enc2(self.down1(e1), temb), tokens)
        hidden = self.mid(e2, temb)
        hidden = self.up1(hidden)
        if hidden.shape[-1] != e1.shape[-1]:
            hidden = torch.nn.functional.interpolate(hidden, size=e1.shape[-1], mode="nearest")
        hidden = self.dec1(torch.cat((hidden, e1), dim=1), temb)
        hidden = self.up0(hidden)
        if hidden.shape[-1] != e0.shape[-1]:
            hidden = torch.nn.functional.interpolate(hidden, size=e0.shape[-1], mode="nearest")
        return self.out(self.dec0(torch.cat((hidden, e0), dim=1), temb))

    def forward(self, state: Tensor, timestep: Tensor, *, query_y: Tensor, support_eeg: Tensor) -> Tensor:
        return self.forward_with_tokens(state, timestep, query_y=query_y, tokens=self.encode_support(support_eeg))


def _extract(values: Tensor, timestep: Tensor, ndim: int) -> Tensor:
    return values.gather(0, timestep).reshape(len(timestep), *((1,) * (ndim - 1)))


class DeterministicRawSupportCleaner(nn.Module):
    """One-step clean estimator using the exact same active backbone."""

    forbidden_fields = RawTemporalSupportEncoder.forbidden_fields

    def __init__(self, config: RawSupportCleanConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = RawSupportCleanUNet(config)

    def forward(self, *, query_y: Tensor, support_eeg: Tensor) -> Tensor:
        # The state path is active and carries the observation; no nominal all-zero input.
        timestep = torch.zeros(len(query_y), dtype=torch.long, device=query_y.device)
        return self.backbone(query_y, timestep, query_y=query_y, support_eeg=support_eeg)


class RawSupportCleanDiffusion(nn.Module):
    forbidden_fields = RawTemporalSupportEncoder.forbidden_fields

    def __init__(self, config: RawSupportCleanConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = RawSupportCleanUNet(config)
        _, alpha = cosine_alpha_bar(config.timesteps)
        self.register_buffer("alpha_bar", alpha.float())

    def training_loss(self, clean: Tensor, *, query_y: Tensor, support_eeg: Tensor,
                      generator: torch.Generator, timestep: Tensor | None = None,
                      noise: Tensor | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        if timestep is None:
            timestep = torch.randint(0, self.config.timesteps, (len(clean),), device=clean.device, generator=generator)
        if noise is None:
            noise = torch.randn(clean.shape, dtype=clean.dtype, device=clean.device, generator=generator)
        alpha = _extract(self.alpha_bar, timestep, clean.ndim)
        state = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
        target_v = alpha.sqrt() * noise - (1 - alpha).sqrt() * clean
        prediction = self.backbone(state, timestep, query_y=query_y, support_eeg=support_eeg)
        mask = torch.zeros_like(clean); mask[..., :self.config.valid_length] = 1
        loss = ((prediction - target_v).square() * mask).sum() / mask.sum().clamp_min(1)
        x0 = alpha.sqrt() * state - (1 - alpha).sqrt() * prediction
        return loss, {"predicted_x0": x0, "timestep": timestep, "noise": noise}

    @torch.no_grad()
    def sample(self, *, query_y: Tensor, support_eeg: Tensor, initial_noise: Tensor) -> Tensor:
        if initial_noise.shape != query_y.shape:
            raise ValueError("initial noise shape mismatch")
        tokens = self.backbone.encode_support(support_eeg)
        state = initial_noise.clone()
        schedule = torch.linspace(self.config.timesteps - 1, 0, self.config.ddim_steps, device=query_y.device).round().long()
        for index, t_value in enumerate(schedule):
            timestep = torch.full((len(query_y),), int(t_value), device=query_y.device, dtype=torch.long)
            alpha = _extract(self.alpha_bar, timestep, state.ndim)
            v = self.backbone.forward_with_tokens(state, timestep, query_y=query_y, tokens=tokens)
            x0 = alpha.sqrt() * state - (1 - alpha).sqrt() * v
            epsilon = (1 - alpha).sqrt() * state + alpha.sqrt() * v
            if index + 1 == len(schedule):
                state = x0
            else:
                next_t = torch.full_like(timestep, int(schedule[index + 1]))
                next_alpha = _extract(self.alpha_bar, next_t, state.ndim)
                state = next_alpha.sqrt() * x0 + (1 - next_alpha).sqrt() * epsilon
        state[..., self.config.valid_length:] = 0
        return state


class EMA:
    def __init__(self, module: nn.Module, decay: float = 0.999) -> None:
        self.decay = float(decay)
        self.shadow = {name: value.detach().clone() for name, value in module.state_dict().items()}

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        for name, value in module.state_dict().items():
            if torch.is_floating_point(value):
                self.shadow[name].lerp_(value.detach(), 1 - self.decay)
            else:
                self.shadow[name].copy_(value)

    def copy_to(self, module: nn.Module) -> None:
        module.load_state_dict(self.shadow)

    def state_dict(self) -> dict[str, object]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, value: dict[str, object]) -> None:
        self.decay = float(value["decay"]); self.shadow = value["shadow"]


def checkpoint_payload(config: RawSupportCleanConfig, det: nn.Module, diff: nn.Module,
                       det_ema: EMA, diff_ema: EMA, **extra: object) -> dict[str, object]:
    return {"config": asdict(config), "det": det.state_dict(), "diff": diff.state_dict(),
            "det_ema": det_ema.state_dict(), "diff_ema": diff_ema.state_dict(), **extra}


__all__ = ["RawSupportCleanConfig", "RawTemporalSupportEncoder", "RawSupportCleanUNet",
           "DeterministicRawSupportCleaner", "RawSupportCleanDiffusion", "EMA", "checkpoint_payload"]
