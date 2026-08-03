from __future__ import annotations

import torch

from eeg_cgdr.experiments.mainline_subject_residual_diffusion import task_rows
from eeg_cgdr.models.subject_residual_diffusion import (
    BoundedResidual,
    OneStepResidualEstimator,
    SubjectResidualConfig,
    SubjectResidualDiffusion,
    parameter_count,
)


def _config() -> SubjectResidualConfig:
    return SubjectResidualConfig(
        eeg_channels=4,
        signal_length=32,
        base_channels=8,
        num_timesteps=1000,
        posterior_samples=8,
        ddim_steps=50,
    )


def _condition(batch: int = 2):
    observed = torch.randn(batch, 4, 32)
    population = torch.randn(batch, 4, 2)
    subject = population + 0.2 * torch.randn(batch, 4, 2)
    return {
        "observed": observed,
        "population_anchor": 0.9 * observed,
        "population_transfer": population,
        "subject_transfer": subject,
        "reliability": torch.full((batch,), 0.7),
        "channel_mask": torch.ones(batch, 4, dtype=torch.bool),
        "context_present": torch.ones(batch),
        "valid_time_mask": torch.ones(batch, 32, dtype=torch.bool),
    }


def test_task_table_is_three_klados_plus_25_by_three_sge() -> None:
    rows = task_rows()
    assert len(rows) == 78
    assert sum(row["dataset"] == "klados" for row in rows) == 3
    assert sum(row["dataset"] == "sgeyesub" for row in rows) == 75
    assert {row["seed"] for row in rows} == {20260811, 20260812, 20260813}


def test_information_matched_backbones_have_exactly_equal_parameters() -> None:
    config = _config()
    one = OneStepResidualEstimator(config)
    diffusion = SubjectResidualDiffusion(config)
    assert parameter_count(one) == parameter_count(diffusion)


def test_context_swap_changes_output_without_query_eog_input() -> None:
    model = OneStepResidualEstimator(_config()).eval()
    condition = _condition()
    matching = model(**condition)
    population = model(
        **{
            **condition,
            "subject_transfer": condition["population_transfer"],
            "context_present": torch.zeros(2),
        }
    )
    assert not torch.equal(matching, population)
    assert "query_EOG" in SubjectResidualDiffusion.forbidden_input_fields
    assert "query_EOG" not in SubjectResidualDiffusion.visible_input_fields


def test_bounded_residual_is_finite_nonzero_and_bounded() -> None:
    bound = BoundedResidual(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    value, fraction = bound(torch.full((2, 4, 32), 100.0))
    assert torch.isfinite(value).all()
    assert value.abs().max() <= 4.0
    assert value.abs().min() > 0.0
    assert fraction == 1.0


def test_diffusion_loss_uses_common_residual_shape_and_is_finite() -> None:
    model = SubjectResidualDiffusion(_config())
    condition = _condition()
    target = torch.randn_like(condition["observed"])
    loss, diagnostics = model.training_loss(target, **condition)
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert torch.isfinite(diagnostics["raw_mse"])

