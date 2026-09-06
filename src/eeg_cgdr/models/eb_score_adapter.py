"""Dynamic-transfer empirical-Bayes score adapter for the conditional v7 route."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from eeg_cgdr.models.dynamic_transfer_diffusion import DynamicTransferDiffusion, _time_embedding, apply_fir


def fir_adjoint(transfer: Tensor, eeg: Tensor) -> Tensor:
    """Apply the centered FIR adjoint H* to EEG, retaining lag/phase structure."""
    if transfer.ndim != 4 or eeg.ndim != 3 or transfer.shape[:2] != eeg.shape[:2]:
        raise ValueError("expected H B,C,E,L and EEG B,C,T")
    padding = transfer.shape[-1] // 2
    outputs = []
    for index in range(len(eeg)):
        weight = transfer[index].permute(1, 0, 2).flip(-1)
        outputs.append(torch.nn.functional.conv1d(eeg[index:index + 1], weight, padding=padding)[..., :eeg.shape[-1]])
    return torch.cat(outputs)


def normalized_fir_response(transfer: Tensor, eeg: Tensor) -> Tensor:
    ocular = fir_adjoint(transfer, eeg)
    ocular = ocular / ocular.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    response = apply_fir(transfer, ocular)
    return response / transfer.square().mean(dim=(2, 3), keepdim=False)[..., None].sqrt().clamp_min(1e-6)


@dataclass(frozen=True)
class EBAdapterConfig:
    eeg_channels: int
    eog_channels: int = 2
    width: int = 48
    blocks: int = 3


class DynamicTransferScoreAdapter(nn.Module):
    """Zero-initialized score deviation driven by full dynamic FIR responses."""

    def __init__(self, config: EBAdapterConfig) -> None:
        super().__init__(); self.config = config
        channels = config.eeg_channels
        self.observed_proxy = nn.Conv1d(channels, config.eog_channels, 5, padding=2)
        self.state_proxy = nn.Conv1d(channels, config.eog_channels, 5, padding=2)
        self.input = nn.Conv1d(4 * channels, config.width, 3, padding=1)
        self.time = nn.Sequential(nn.Linear(config.width, config.width), nn.SiLU(), nn.Linear(config.width, config.width))
        self.blocks = nn.ModuleList(nn.Sequential(nn.GroupNorm(8, config.width), nn.SiLU(), nn.Conv1d(config.width, config.width, 3, padding=2**(i % 3), dilation=2**(i % 3))) for i in range(config.blocks))
        self.output = nn.Conv1d(config.width, channels, 3, padding=1)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def dynamic_features(self, state: Tensor, observed: Tensor, delta_transfer: Tensor) -> tuple[Tensor, Tensor]:
        if delta_transfer.shape[2] != self.config.eog_channels:
            raise ValueError("EOG order/layout differs from adapter")
        # The learned EEG->ocular proxy is shared across subjects; applying the
        # subject delta FIR afterwards retains signed lag and phase.
        dynamic_response = apply_fir(delta_transfer, self.observed_proxy(observed))
        dynamic_state = apply_fir(delta_transfer, self.state_proxy(state))
        scale = delta_transfer.square().mean(dim=(2, 3), keepdim=False)[..., None].sqrt().clamp_min(1e-6)
        return dynamic_response / scale, dynamic_state / scale

    def forward(self, state: Tensor, timestep: Tensor, *, observed: Tensor, delta_transfer: Tensor) -> Tensor:
        dynamic_response, dynamic_state = self.dynamic_features(state, observed, delta_transfer)
        hidden = self.input(torch.cat((state, observed, dynamic_response, dynamic_state), dim=1))
        hidden = hidden + self.time(_time_embedding(timestep, self.config.width))[..., None]
        for block in self.blocks:
            hidden = hidden + block(hidden)
        return self.output(torch.nn.functional.silu(hidden))


class EBScoreAdapterDiffusion(nn.Module):
    def __init__(self, population: DynamicTransferDiffusion, adapter: DynamicTransferScoreAdapter) -> None:
        super().__init__(); self.population = population; self.adapter = adapter
        for parameter in self.population.parameters():
            parameter.requires_grad_(False)
        self.population.eval()

    def prediction(self, state: Tensor, timestep: Tensor, *, observed: Tensor, population_transfer: Tensor, delta_transfer: Tensor) -> Tensor:
        reliability = torch.ones(len(observed), device=observed.device, dtype=observed.dtype)
        with torch.no_grad():
            base = self.population.backbone(state, timestep, observed=observed, transfer=population_transfer, reliability=reliability)
        return base + self.adapter(state, timestep, observed=observed, delta_transfer=delta_transfer)


__all__ = ["DynamicTransferScoreAdapter", "EBAdapterConfig", "EBScoreAdapterDiffusion", "fir_adjoint", "normalized_fir_response"]
