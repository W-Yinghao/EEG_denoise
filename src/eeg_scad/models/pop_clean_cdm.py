"""Subject-agnostic same-backbone clean conditional diffusion."""
from __future__ import annotations

import torch
from torch import Tensor

from eeg_scad.models.support_clean_cdm import SupportCleanCDM


class PopCleanCDM(SupportCleanCDM):
    uses_subject_support = False

    def __init__(self, width: int = 64, timesteps: int = 1000) -> None:
        super().__init__(width, timesteps); self.register_buffer("population_context", torch.zeros(1, 128))

    def context(self, batch: int) -> Tensor:
        return self.population_context.expand(batch, -1)

    def training_prediction(self, clean: Tensor, y: Tensor, generator: torch.Generator):
        return super().training_prediction(clean, y, self.context(len(y)), generator)

    def sample(self, y: Tensor, noise: Tensor, steps: int = 25):
        return super().sample(y, self.context(len(y)), noise, steps)


__all__ = ["PopCleanCDM"]
