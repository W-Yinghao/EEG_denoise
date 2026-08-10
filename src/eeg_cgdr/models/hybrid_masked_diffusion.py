"""Capacity-matched masked clean EEG estimators for PhysioMotion."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from .artifact_latent_diffusion import cosine_alpha_bar


@dataclass(frozen=True)
class HybridMaskedConfig:
    channels: int = 34
    signal_length: int = 500
    base_channels: int = 32
    timesteps: int = 1000
    ddim_steps: int = 25
    posterior_samples: int = 8
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.channels != 34 or self.signal_length != 500:
            raise ValueError("PhysioMotion freezes 34 channels and 500 samples")
        if self.ddim_steps != 25 or self.posterior_samples != 8:
            raise ValueError("primary panel freezes DDIM25/K8")


class _TimeEmbedding(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.network = nn.Sequential(nn.Linear(width, 4 * width), nn.SiLU(), nn.Linear(4 * width, 4 * width))

    def forward(self, timestep: Tensor) -> Tensor:
        half = self.width // 2
        scale = torch.exp(-torch.log(torch.tensor(10000.0, device=timestep.device)) * torch.arange(half, device=timestep.device) / max(half - 1, 1))
        angle = timestep.float()[:, None] * scale[None]
        return self.network(torch.cat((angle.sin(), angle.cos()), dim=1))


class _Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_width: int, dropout: float) -> None:
        super().__init__()
        groups = 8 if out_channels % 8 == 0 else 4
        self.first = nn.Sequential(nn.Conv1d(in_channels, out_channels, 3, padding=1), nn.GroupNorm(groups, out_channels), nn.SiLU())
        self.time = nn.Linear(time_width, out_channels)
        self.second = nn.Sequential(nn.Dropout(dropout), nn.Conv1d(out_channels, out_channels, 3, padding=1), nn.GroupNorm(groups, out_channels), nn.SiLU())
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, value: Tensor, time: Tensor) -> Tensor:
        hidden = self.first(value)
        hidden = hidden + self.time(time)[:, :, None]
        return self.second(hidden) + self.skip(value)


class HybridMaskedUNet(nn.Module):
    """One shared active architecture for DET and DIFF."""

    visible_fields = ("state", "masked_query", "binary_mask", "population_retrieval", "subject_residual", "timestep")
    forbidden_fields = ("clean_target", "query_annotation", "oracle_indices", "participant_id", "evaluator_outcome")

    def __init__(self, config: HybridMaskedConfig) -> None:
        super().__init__()
        self.config = config
        width = config.base_channels
        self.time_embed = _TimeEmbedding(width)
        self.stem = nn.Conv1d(5 * config.channels, width, 5, padding=2)
        self.enc0 = _Block(width, width, 4 * width, config.dropout)
        self.down0 = nn.Conv1d(width, 2 * width, 4, stride=2, padding=1)
        self.enc1 = _Block(2 * width, 2 * width, 4 * width, config.dropout)
        self.down1 = nn.Conv1d(2 * width, 4 * width, 4, stride=2, padding=1)
        self.mid = _Block(4 * width, 4 * width, 4 * width, config.dropout)
        self.up1 = nn.ConvTranspose1d(4 * width, 2 * width, 4, stride=2, padding=1)
        self.dec1 = _Block(4 * width, 2 * width, 4 * width, config.dropout)
        self.up0 = nn.ConvTranspose1d(2 * width, width, 4, stride=2, padding=1)
        self.dec0 = _Block(2 * width, width, 4 * width, config.dropout)
        self.out = nn.Conv1d(width, config.channels, 3, padding=1)

    def forward(self, state: Tensor, timestep: Tensor, *, y_obs: Tensor, mask: Tensor, r_pop: Tensor, subject_residual: Tensor) -> Tensor:
        time = self.time_embed(timestep)
        hidden0 = self.enc0(self.stem(torch.cat((state, y_obs, mask, r_pop, subject_residual), dim=1)), time)
        hidden1 = self.enc1(self.down0(hidden0), time)
        middle = self.mid(self.down1(hidden1), time)
        up1 = self.up1(middle)
        if up1.shape[-1] != hidden1.shape[-1]: up1 = torch.nn.functional.interpolate(up1, size=hidden1.shape[-1], mode="nearest")
        decoded1 = self.dec1(torch.cat((up1, hidden1), dim=1), time)
        up0 = self.up0(decoded1)
        if up0.shape[-1] != hidden0.shape[-1]: up0 = torch.nn.functional.interpolate(up0, size=hidden0.shape[-1], mode="nearest")
        return self.out(self.dec0(torch.cat((up0, hidden0), dim=1), time))


class DeterministicHybridMasked(nn.Module):
    def __init__(self, config: HybridMaskedConfig) -> None:
        super().__init__(); self.config = config; self.backbone = HybridMaskedUNet(config)

    def forward(self, *, y_obs: Tensor, mask: Tensor, r_pop: Tensor, subject_residual: Tensor) -> Tensor:
        timestep = torch.zeros(len(y_obs), device=y_obs.device, dtype=torch.long)
        proposal = self.backbone(y_obs, timestep, y_obs=y_obs, mask=mask, r_pop=r_pop, subject_residual=subject_residual)
        return y_obs * (1 - mask) + proposal * mask


def _extract(values: Tensor, timestep: Tensor, ndim: int) -> Tensor:
    return values.gather(0, timestep).reshape(len(timestep), *((1,) * (ndim - 1)))


class HybridMaskedDiffusion(nn.Module):
    def __init__(self, config: HybridMaskedConfig) -> None:
        super().__init__(); self.config = config; self.backbone = HybridMaskedUNet(config)
        _, alpha = cosine_alpha_bar(config.timesteps); self.register_buffer("alpha_bar", alpha.float())

    def training_loss(self, clean: Tensor, *, y_obs: Tensor, mask: Tensor, r_pop: Tensor, subject_residual: Tensor, generator: torch.Generator, timestep: Tensor | None = None, noise: Tensor | None = None) -> tuple[Tensor, dict[str, Tensor]]:
        if timestep is None: timestep = torch.randint(0, self.config.timesteps, (len(clean),), device=clean.device, generator=generator)
        if noise is None: noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
        alpha = _extract(self.alpha_bar, timestep, clean.ndim)
        noised = alpha.sqrt() * clean + (1 - alpha).sqrt() * noise
        state = y_obs * (1 - mask) + noised * mask
        target_v = alpha.sqrt() * noise - (1 - alpha).sqrt() * clean
        prediction = self.backbone(state, timestep, y_obs=y_obs, mask=mask, r_pop=r_pop, subject_residual=subject_residual)
        loss = ((prediction - target_v).square() * mask).sum() / mask.sum().clamp_min(1)
        x0 = alpha.sqrt() * state - (1 - alpha).sqrt() * prediction
        x0 = y_obs * (1 - mask) + x0 * mask
        return loss, {"predicted_x0": x0, "timestep": timestep, "noise": noise}

    @torch.no_grad()
    def sample(self, *, y_obs: Tensor, mask: Tensor, r_pop: Tensor, subject_residual: Tensor, initial_noise: Tensor) -> Tensor:
        if initial_noise.shape != y_obs.shape: raise ValueError("initial noise shape mismatch")
        state = y_obs * (1 - mask) + initial_noise * mask
        schedule = torch.linspace(self.config.timesteps - 1, 0, self.config.ddim_steps, device=y_obs.device).round().long()
        for index, value in enumerate(schedule):
            timestep = torch.full((len(state),), int(value), device=state.device, dtype=torch.long)
            alpha = _extract(self.alpha_bar, timestep, state.ndim)
            prediction = self.backbone(state, timestep, y_obs=y_obs, mask=mask, r_pop=r_pop, subject_residual=subject_residual)
            x0 = alpha.sqrt() * state - (1 - alpha).sqrt() * prediction
            epsilon = (1 - alpha).sqrt() * state + alpha.sqrt() * prediction
            if index + 1 == len(schedule): state = x0
            else:
                next_t = torch.full_like(timestep, int(schedule[index + 1])); next_alpha = _extract(self.alpha_bar, next_t, state.ndim)
                state = next_alpha.sqrt() * x0 + (1 - next_alpha).sqrt() * epsilon
            state = y_obs * (1 - mask) + state * mask
        return state


class EMA:
    def __init__(self, module: nn.Module, decay: float = .999) -> None:
        self.decay = float(decay); self.shadow = {name: value.detach().clone() for name, value in module.state_dict().items()}

    @torch.no_grad()
    def update(self, module: nn.Module) -> None:
        for name, value in module.state_dict().items():
            if torch.is_floating_point(value): self.shadow[name].lerp_(value.detach(), 1 - self.decay)
            else: self.shadow[name].copy_(value)

    def copy_to(self, module: nn.Module) -> None: module.load_state_dict(self.shadow)
    def state_dict(self) -> dict[str, object]: return {"decay": self.decay, "shadow": self.shadow}
    def load_state_dict(self, value: dict[str, object]) -> None:
        self.decay = float(value["decay"])
        # A restored EMA must own its tensors.  Aliasing checkpoint storage
        # makes two independent resume processes mutate each other's state.
        self.shadow = {name: tensor.detach().clone() for name, tensor in value["shadow"].items()}


def parameter_count(module: nn.Module) -> int:
    return sum(value.numel() for value in module.parameters())


def config_dict(config: HybridMaskedConfig) -> dict[str, object]: return asdict(config)


__all__ = ["HybridMaskedConfig", "HybridMaskedUNet", "DeterministicHybridMasked", "HybridMaskedDiffusion", "EMA", "parameter_count", "config_dict"]
