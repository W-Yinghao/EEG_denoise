"""Closed-form partial-observation proximal energy."""
from __future__ import annotations

import torch
from torch import Tensor


def _project(value: Tensor, projector: Tensor) -> Tensor:
    return torch.einsum("bij,bjt->bit", projector, value)


def partial_observation_prox(candidate: Tensor, anchor: Tensor, projector: Tensor, mask: Tensor, lambda_anchor: float, lambda_measurement: float) -> Tensor:
    if candidate.shape != anchor.shape or candidate.shape[0] != projector.shape[0]:
        raise ValueError("incompatible energy tensors")
    if lambda_anchor == 0 and lambda_measurement == 0:
        return candidate.clone()
    parallel_c = _project(candidate, projector)
    parallel_a = _project(anchor, projector)
    perpendicular_c = candidate - parallel_c
    perpendicular_a = anchor - parallel_a
    numerator_parallel = parallel_c + lambda_anchor * parallel_a
    numerator_perpendicular = perpendicular_c + lambda_anchor * perpendicular_a
    denominator_parallel = 1 + lambda_anchor + lambda_measurement * (1-mask[:, None]) ** 2
    denominator_perpendicular = 1 + lambda_anchor + lambda_measurement
    return numerator_parallel / denominator_parallel + numerator_perpendicular / denominator_perpendicular


def partial_observation_solve(candidate: Tensor, anchor: Tensor, projector: Tensor, mask: Tensor, lambda_anchor: float, lambda_measurement: float) -> Tensor:
    """Independent dense normal-equation solve used only for fixtures."""
    output = torch.empty_like(candidate)
    identity = torch.eye(candidate.shape[1], dtype=candidate.dtype, device=candidate.device)
    complement = identity[None] - projector
    for batch in range(candidate.shape[0]):
        for time in range(candidate.shape[-1]):
            measurement = complement[batch] + (1-mask[batch, time]) * projector[batch]
            lhs = (1+lambda_anchor)*identity + lambda_measurement * measurement.T @ measurement
            rhs = candidate[batch, :, time] + lambda_anchor * anchor[batch, :, time]
            output[batch, :, time] = torch.linalg.solve(lhs, rhs)
    return output


def energy_diagnostics(candidate: Tensor, refined: Tensor, anchor: Tensor, projector: Tensor, mask: Tensor) -> dict[str, float]:
    before = candidate.square().mean().sqrt().clamp_min(1e-8)
    parallel = _project(refined, projector)
    complement = refined-parallel
    return {
        "energy_shrinkage_ratio": float(refined.square().mean().sqrt()/before),
        "parallel_correction_energy": float(parallel.square().mean()),
        "complement_correction_energy": float(complement.square().mean()),
        "mean_mask": float(mask.mean()),
        "mask_prevalence": float((mask > .5).float().mean()),
        "distance_to_anchor": float((refined-anchor).square().mean().sqrt()),
    }


__all__ = ["partial_observation_prox", "partial_observation_solve", "energy_diagnostics"]
