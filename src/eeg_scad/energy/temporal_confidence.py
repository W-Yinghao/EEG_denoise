"""EEG-only artifact confidence derived from frozen deterministic estimates."""
from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def artifact_score(artifact_det: Tensor, artifact_pop: Tensor) -> Tensor:
    return torch.maximum(artifact_det.square().mean(-2).sqrt(), artifact_pop.square().mean(-2).sqrt())


def calibrate_quantiles(artifact_det: Tensor, artifact_pop: Tensor) -> tuple[float, float]:
    score = artifact_score(artifact_det, artifact_pop).flatten()
    return float(torch.quantile(score, .5)), float(torch.quantile(score, .9))


def temporal_confidence(artifact_det: Tensor, artifact_pop: Tensor, q50: float, q90: float, smoothing_samples: int = 10) -> Tensor:
    score = artifact_score(artifact_det, artifact_pop)
    mask = ((score - q50) / max(q90 - q50, 1e-8)).clamp(0, 1)
    if smoothing_samples > 1:
        half = smoothing_samples // 2
        # Explicit construction avoids version-dependent negative-step slicing.
        kernel = torch.cat((torch.arange(1, half + 2, device=mask.device, dtype=mask.dtype), torch.arange(half, 0, -1, device=mask.device, dtype=mask.dtype)))
        kernel = (kernel / kernel.sum()).view(1, 1, -1)
        mask = F.conv1d(mask[:, None], kernel, padding=kernel.shape[-1] // 2)[:, 0, :mask.shape[-1]]
    return mask.clamp(0, 1)


__all__ = ["artifact_score", "calibrate_quantiles", "temporal_confidence"]
