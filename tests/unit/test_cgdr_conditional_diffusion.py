"""Contracts for the exploratory task-matched conditional diffusion arm."""

from __future__ import annotations

import csv
import inspect
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from eeg_cgdr.experiments import stage3_conditional_diffusion as conditional_stage3
from eeg_cgdr.experiments.stage3_conditional_diffusion import (
    _comparison_cell_status,
    _frozen_common_eligible_records,
    _paired_cell_status,
    validate_conditional_config,
)
from eeg_cgdr.models.conditional_diffusion import (
    OperatorConditionedEEGDiffusion,
)
from eeg_cgdr.models.deterministic_unet import (
    DeterministicUNetConfig,
    TaskMatchedDeterministicUNet,
)
from eeg_cgdr.training import resume_training_checkpoint, save_training_checkpoint
from saddpm.diffusion.schedule import DiffusionConfig


CONFIG_PATH = Path("configs/cgdr/klados_stage3_conditional_diffusion_v3.yaml")


def _backbone() -> DeterministicUNetConfig:
    return DeterministicUNetConfig(
        eeg_channels=3,
        signal_length=64,
        base_channels=8,
        channel_mults=(1, 2, 4),
        num_res_blocks=1,
        groupnorm_groups=8,
        dropout=0.0,
        time_sinusoidal_dim=16,
        time_embed_dim=32,
        attention_length=8,
        attention_heads=4,
    )


def _model() -> OperatorConditionedEEGDiffusion:
    torch.manual_seed(991)
    return OperatorConditionedEEGDiffusion(
        _backbone(),
        DiffusionConfig(
            num_timesteps=8,
            beta_start=1.0e-4,
            beta_end=0.02,
            schedule="linear",
        ),
        enforce_scientific_schedule=False,
    ).eval()


def _inputs() -> tuple[torch.Tensor, ...]:
    observed = torch.randn(2, 3, 64)
    x_t = torch.randn_like(observed)
    projector = torch.diag(torch.tensor([1.0, 0.0, 0.0]))
    attenuation = torch.linspace(0.05, 1.0, 64).repeat(2, 1)
    mask = torch.zeros(2, 64, dtype=torch.bool)
    mask[:, :47] = True
    timesteps = torch.tensor([2, 6], dtype=torch.long)
    return x_t, timesteps, observed, projector, attenuation, mask


def test_conditional_protocol_freezes_fair_inputs_and_development_only_scope() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    validate_conditional_config(config)
    assert config["protocol_id"] == "klados_operator_conditioned_diffusion_matched_v3"
    assert config["source_record_split"]["training"] == list(range(1, 31))
    assert config["source_record_split"]["development"] == [
        31,
        32,
        33,
        34,
        35,
        36,
        44,
        45,
    ]
    assert config["source_record_split"]["historical_evaluation_allowed"] is False
    fairness = config["fair_comparison_contract"]
    assert fairness["common_eligibility_rule"] == (
        "matching_p0_eligible_records_shared_by_all_operator_scopes"
    )
    assert fairness["target_optimizer_updates"] == (
        "fixed_6000_successful_optimizer_updates_for_both_models"
    )
    assert fairness["no_development_or_evaluation_outcome_checkpoint_selection"] is True
    assert fairness["different_training_objective_disclosed"] == (
        "epsilon_prediction_vs_deterministic_task_loss"
    )
    assert config["training"]["amp_initial_scale"] == 1024.0
    assert config["training"]["maximum_skipped_optimizer_steps"] == 0
    assert fairness["broad_diffusion_family_claim_allowed"] is False


def test_conditional_model_has_same_visible_information_as_deterministic() -> None:
    model = _model()
    forward = inspect.signature(model.forward).parameters
    assert tuple(forward) == (
        "x_t",
        "timesteps",
        "observed",
        "projector",
        "attenuation",
        "valid_time_mask",
    )
    assert model.visible_input_fields == TaskMatchedDeterministicUNet.visible_input_fields
    assert "clean_target" not in forward
    assert model.algorithm_state_fields == (
        "diffused_clean_state_x_t",
        "diffusion_timestep",
    )


def test_conditioning_stack_is_exactly_the_deterministic_stack() -> None:
    backbone = _backbone()
    deterministic = TaskMatchedDeterministicUNet(backbone).eval()
    conditional = _model()
    x_t, timesteps, observed, projector, attenuation, mask = _inputs()
    deterministic_inputs: list[torch.Tensor] = []
    conditional_inputs: list[torch.Tensor] = []
    first = deterministic.unet.stem.register_forward_pre_hook(
        lambda _module, args: deterministic_inputs.append(args[0].detach().clone())
    )
    second = conditional.unet.stem.register_forward_pre_hook(
        lambda _module, args: conditional_inputs.append(args[0].detach().clone())
    )
    try:
        with torch.no_grad():
            deterministic(
                observed,
                projector=projector,
                attenuation=attenuation,
                valid_time_mask=mask,
            )
            conditional(
                x_t,
                timesteps,
                observed=observed,
                projector=projector,
                attenuation=attenuation,
                valid_time_mask=mask,
            )
    finally:
        first.remove()
        second.remove()
    assert len(deterministic_inputs) == len(conditional_inputs) == 1
    channels = backbone.eeg_channels
    assert conditional.unet.stem.in_channels == (
        deterministic.unet.stem.in_channels + channels
    )
    torch.testing.assert_close(
        conditional_inputs[0][:, channels:],
        deterministic_inputs[0],
        rtol=0.0,
        atol=0.0,
    )


def test_conditional_diffusion_is_internally_padding_safe() -> None:
    model = _model()
    x_t, timesteps, observed, projector, attenuation, mask = _inputs()
    changed_x_t = x_t.clone()
    changed_observed = observed.clone()
    changed_x_t[:, :, 47:] = 1.0e6 * torch.randn_like(changed_x_t[:, :, 47:])
    changed_observed[:, :, 47:] = 1.0e6 * torch.randn_like(
        changed_observed[:, :, 47:]
    )
    with torch.no_grad():
        first = model(
            x_t,
            timesteps,
            observed=observed,
            projector=projector,
            attenuation=attenuation,
            valid_time_mask=mask,
        )
        second = model(
            changed_x_t,
            timesteps,
            observed=changed_observed,
            projector=projector,
            attenuation=attenuation,
            valid_time_mask=mask,
        )
    torch.testing.assert_close(first[:, :, :47], second[:, :, :47], rtol=0.0, atol=0.0)
    assert torch.count_nonzero(first[:, :, 47:]) == 0
    assert torch.count_nonzero(second[:, :, 47:]) == 0


def test_conditional_training_loss_masks_target_and_has_finite_gradient() -> None:
    model = _model().train()
    x_t, timesteps, observed, projector, attenuation, mask = _inputs()
    clean = torch.randn_like(observed)
    noise = torch.randn_like(clean)
    loss = model.training_loss(
        clean,
        observed=observed,
        projector=projector,
        attenuation=attenuation,
        valid_time_mask=mask,
        timesteps=timesteps,
        noise=noise,
    )
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_conditional_ddim_steps_equal_actual_network_calls() -> None:
    model = _model()
    _x_t, _timesteps, observed, projector, attenuation, mask = _inputs()
    initial_noise = torch.randn_like(observed)
    result = model.sample_ddim(
        observed=observed,
        projector=projector,
        attenuation=attenuation,
        valid_time_mask=mask,
        ddim_steps=4,
        eta=0.0,
        initial_noise=initial_noise,
    )
    assert result.network_calls == 4
    assert result.restored.shape == observed.shape
    assert torch.count_nonzero(result.restored[:, :, 47:]) == 0


def test_scientific_schedule_terminal_and_standard_normal_initial_state(
    monkeypatch,
) -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    raw = config["conditional_diffusion"]
    model = OperatorConditionedEEGDiffusion(
        _backbone(),
        DiffusionConfig(
            num_timesteps=int(raw["num_timesteps"]),
            beta_start=float(raw["beta_start"]),
            beta_end=float(raw["beta_end"]),
            schedule=str(raw["schedule"]),
        ),
        enforce_scientific_schedule=True,
    ).eval()
    assert model.terminal_alpha_bar is not None
    assert model.terminal_alpha_bar <= float(raw["terminal_alpha_bar_maximum"])
    assert raw["initial_distribution"] == "standard_normal_at_timestep_999"

    _x_t, _timesteps, observed, projector, attenuation, mask = _inputs()
    sentinel = 0.375
    captured: list[torch.Tensor] = []
    hook = model.unet.stem.register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach().clone())
    )

    def fake_standard_normal(shape, *, device, dtype, generator):
        assert generator is not None
        return torch.full(shape, sentinel, device=device, dtype=dtype)

    monkeypatch.setattr(torch, "randn", fake_standard_normal)
    try:
        model.sample_ddim(
            observed=observed,
            projector=projector,
            attenuation=attenuation,
            valid_time_mask=mask,
            ddim_steps=2,
            eta=0.0,
            generator=torch.Generator().manual_seed(37),
        )
    finally:
        hook.remove()
    assert len(captured) == 2
    channels = observed.shape[1]
    expected = torch.full_like(observed, sentinel) * mask[:, None, :]
    torch.testing.assert_close(
        captured[0][:, :channels], expected, rtol=0.0, atol=0.0
    )


def test_conditional_checkpoint_resume_restores_exact_next_update(tmp_path) -> None:
    random.seed(812)
    np.random.seed(812)
    torch.manual_seed(812)
    model = _model().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-4, weight_decay=1.0e-4)
    _x_t, _timesteps, observed, projector, attenuation, mask = _inputs()
    clean = torch.randn_like(observed)

    optimizer.zero_grad(set_to_none=True)
    model.training_loss(
        clean,
        observed=observed,
        projector=projector,
        attenuation=attenuation,
        valid_time_mask=mask,
    ).backward()
    optimizer.step()
    checkpoint = tmp_path / "conditional.pt"
    contract = {"fixed_endpoint_update": 6000, "selection": "fixed"}
    save_training_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scaler=None,
        epoch=0,
        step=1,
        config=contract,
        normalizer={"source_records": list(range(1, 31))},
        extra={
            "fixed_endpoint_update": 6000,
            "checkpoint_selection_used_development_loss": False,
        },
    )

    optimizer.zero_grad(set_to_none=True)
    uninterrupted_loss = model.training_loss(
        clean,
        observed=observed,
        projector=projector,
        attenuation=attenuation,
        valid_time_mask=mask,
    )
    uninterrupted_loss.backward()
    optimizer.step()
    uninterrupted = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }

    resumed_model = _model().train()
    resumed_optimizer = torch.optim.AdamW(
        resumed_model.parameters(), lr=2.0e-4, weight_decay=1.0e-4
    )
    state = resume_training_checkpoint(
        checkpoint,
        model=resumed_model,
        optimizer=resumed_optimizer,
        expected_config=contract,
    )
    assert state.step == 1
    assert state.extra["fixed_endpoint_update"] == 6000
    resumed_optimizer.zero_grad(set_to_none=True)
    resumed_loss = resumed_model.training_loss(
        clean,
        observed=observed,
        projector=projector,
        attenuation=attenuation,
        valid_time_mask=mask,
    )
    resumed_loss.backward()
    resumed_optimizer.step()

    torch.testing.assert_close(
        resumed_loss, uninterrupted_loss, rtol=0.0, atol=0.0
    )
    for name, value in resumed_model.state_dict().items():
        torch.testing.assert_close(value, uninterrupted[name], rtol=0.0, atol=0.0)


def test_aggregate_status_helpers_retain_failures_and_unmatched_cells() -> None:
    assert (
        _comparison_cell_status(
            {"status": "failed_sampling_numerical"},
            record_is_eligible=True,
            family="conditional",
        )
        == "failed_sampling_numerical"
    )
    assert (
        _comparison_cell_status(
            None, record_is_eligible=True, family="reference"
        )
        == "unmatched_missing_reference_cell"
    )
    assert (
        _comparison_cell_status(
            None, record_is_eligible=True, family="conditional"
        )
        == "unmatched_missing_conditional_cell"
    )
    assert (
        _comparison_cell_status(
            None, record_is_eligible=False, family="conditional"
        )
        == "ineligible_common_record"
    )
    status = _paired_cell_status("success", "failed_method_numerical")
    assert status == "conditional=success;reference=failed_method_numerical"
    assert _paired_cell_status("success", "success") == "success_paired"


def test_common_eligibility_is_independent_of_conditional_job_status(tmp_path) -> None:
    deterministic = yaml.safe_load(
        Path("configs/cgdr/klados_stage3_deterministic_comparison_v4.yaml").read_text(
            encoding="utf-8"
        )
    )
    deterministic["outputs"]["root"] = str(tmp_path)
    coverage = {
        "requested_record_ids": [31, 32, 33, 34, 35, 36, 44, 45],
        "included_record_ids": [31, 32, 33, 34, 35, 36, 44],
        "skipped_record_ids": [45],
    }
    for scope in (
        "population_projector",
        "matching_p0",
        "query_derived_oracle_projector",
    ):
        path = tmp_path / "training" / scope / "result_summary.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "protocol_id": "klados_stage3_deterministic_scope_isolated_v4",
                    "status": "completed",
                    "validation_record_coverage": coverage,
                }
            ),
            encoding="utf-8",
        )
    assert _frozen_common_eligible_records(deterministic) == {
        "sim31",
        "sim32",
        "sim33",
        "sim34",
        "sim35",
        "sim36",
        "sim44",
    }


def test_conditional_aggregate_retains_eight_records_and_counts_missing_cells(
    tmp_path, monkeypatch
) -> None:
    eligible = {"sim31", "sim33", "sim34", "sim36", "sim44", "sim45"}
    ineligible = {"sim32", "sim35"}
    conditional_output_root = tmp_path / "conditional_output"
    conditional_root = conditional_output_root / "development"
    deterministic_root = tmp_path / "deterministic"
    config = {
        "claim_scope": "exploratory_test_fixture",
        "outputs": {
            "root": str(conditional_output_root),
            "development_root": str(conditional_root),
        },
    }
    deterministic = {
        "outputs": {"development_root": str(deterministic_root)}
    }
    prior_history = tmp_path / "prior_training_history.csv"
    prior_history.write_text(
        "epoch,step,train_loss\n0,15,1.0\n199,3000,0.1\n",
        encoding="utf-8",
    )
    base_config_path = tmp_path / "base_config.yaml"
    base_config_path.write_text(
        yaml.safe_dump(
            {"outputs": {"training_history": str(prior_history)}}
        ),
        encoding="utf-8",
    )
    deterministic["base_config"] = str(base_config_path)
    monkeypatch.setattr(
        conditional_stage3, "validate_conditional_config", lambda _config: None
    )
    monkeypatch.setattr(
        conditional_stage3, "_deterministic_config", lambda _config: deterministic
    )
    monkeypatch.setattr(
        conditional_stage3,
        "_frozen_common_eligible_records",
        lambda _config: set(eligible),
    )
    for scope in conditional_stage3.FROZEN_OPERATOR_SOURCES:
        training_summary = (
            conditional_output_root / "training" / scope / "result_summary.json"
        )
        training_summary.parent.mkdir(parents=True, exist_ok=True)
        training_summary.write_text(
            json.dumps(
                {
                    "status": "completed",
                    "protocol_id": conditional_stage3.PROTOCOL_ID,
                    "operator_scope": scope,
                    "target_optimizer_updates": 6000,
                    "actual_optimizer_updates": 6000,
                    "optimizer_step_attempts": 6000,
                    "skipped_optimizer_steps_amp_overflow": 0,
                    "exact_update_budget_matched": True,
                    "resumed": False,
                }
            ),
            encoding="utf-8",
        )

    def conditional_row(record: str, scope: str) -> dict[str, object]:
        row = {
            field: "" for field in conditional_stage3.REQUIRED_CONDITIONAL_ROW_FIELDS
        }
        row.update(
            {
                "source_record": record,
                "method_id": conditional_stage3.METHOD_ID,
                "status": "success",
                "operator_source": scope,
                "effective_operator_source": scope,
                "common_eligibility_status": "included",
                "conditional_training_windows": 154,
                "conditional_development_windows": 68,
                "deterministic_training_windows": 154,
                "deterministic_development_windows": 68,
                "fixed_optimizer_updates_each": 6000,
                "conditional_actual_optimizer_updates": 6000,
                "deterministic_fixed_checkpoint_updates": 6000,
                "deterministic_actual_training_updates": 6000,
                "conditional_model_parameters": 101,
                "deterministic_model_parameters": 99,
                "conditional_training_walltime_seconds": 10.0,
                "deterministic_training_walltime_seconds": 9.0,
                "latency_seconds": 1.0,
                "peak_memory_mb": 2.0,
                "function_evaluations_per_seed_per_window": 100,
                "total_function_evaluations_per_window": 500,
                "algorithmic_seed_count": 5,
                "same_paired_supervision_exposure": True,
                "conditional_training_objective": "valid_time_masked_epsilon_MSE",
                "deterministic_training_objective": "paired_task_loss",
                "training_objectives_equal": False,
                "e_parallel": 0.8,
                "e_perp": 0.1,
                "rrmse": 0.7,
                "correlation": 0.9,
                "psd_distortion": 0.2,
                "artifact_attenuation": 1.1,
                "clean_interval_preservation": 0.95,
            }
        )
        return row

    for record in sorted(eligible):
        record_root = conditional_root / record
        rows = [
            conditional_row(record, scope)
            for scope in conditional_stage3.FROZEN_OPERATOR_SOURCES
        ]
        conditional_stage3._write_csv(record_root / "metrics.csv", rows)
        (record_root / "result_summary.json").write_text(
            json.dumps(
                {
                    "status": "completed_exploratory_conditional_diffusion_development",
                    "common_eligibility_status": "included",
                    "successful_method_arms": 3,
                    "failed_method_arms": 0,
                }
            ),
            encoding="utf-8",
        )
    for record in sorted(ineligible):
        record_root = conditional_root / record
        record_root.mkdir(parents=True, exist_ok=True)
        (record_root / "result_summary.json").write_text(
            json.dumps(
                {
                    "status": "ineligible_common_record",
                    "common_eligibility_status": "excluded_matching_p0_ineligible",
                    "successful_method_arms": 0,
                    "failed_method_arms": 0,
                }
            ),
            encoding="utf-8",
        )

    for record_id in conditional_stage3.KLADOS_DEVELOPMENT_RECORDS:
        record = f"sim{record_id:02d}"
        rows = []
        for scope in conditional_stage3.FROZEN_OPERATOR_SOURCES:
            for method in conditional_stage3.FROZEN_METHODS:
                rows.append(
                    {
                        "source_record": record,
                        "operator_source": scope,
                        "method_id": method,
                        "status": "success",
                        "e_parallel": 0.9,
                        "e_perp": 0.1,
                        "rrmse": 0.8,
                        "correlation": 0.85,
                        "psd_distortion": 0.25,
                        "artifact_attenuation": 1.0,
                        "clean_interval_preservation": 0.9,
                        "training_model_parameters": 99,
                        "latency_seconds": 0.5,
                    }
                )
        conditional_stage3._write_csv(
            deterministic_root / record / "metrics.csv", rows
        )
        (deterministic_root / record / "result_summary.json").write_text(
            json.dumps({"diffusion_prior_parameters": 77}),
            encoding="utf-8",
        )

    complete = conditional_stage3.aggregate_conditional_development(
        config, run_dir=tmp_path / "complete_run"
    )
    assert complete["status"] == (
        "completed_exploratory_development_no_family_decision"
    )
    assert complete["available_source_records_denominator"] == 8
    assert complete["common_eligible_source_records"] == 6
    assert complete["common_eligible_source_record_ids"] == sorted(eligible)
    assert complete["ineligible_source_records"] == 2
    assert complete["records_without_performance_metrics"] == 2
    assert complete["nonterminal_or_missing_result_summaries"] == 0
    assert complete["expected_conditional_method_cells"] == 18
    assert complete["observed_conditional_method_cells"] == 18
    assert complete["unmatched_missing_conditional_method_cells"] == 0
    assert complete["unexpected_conditional_method_cells"] == 0
    assert complete["conditional_record_scope_cartesian_product_exact"] is True
    assert complete["conditional_zero_amp_skips_all_scopes"] is True
    assert complete["conditional_training_resumed_false_all_scopes"] is True
    assert complete["conditional_method_arm_failure_rate"] == 0.0
    assert (conditional_root / "resolved_config.yaml").is_file()
    common_summary_path = conditional_root / "common_eligible_arm_summary.csv"
    assert common_summary_path.is_file()
    with common_summary_path.open("r", encoding="utf-8", newline="") as stream:
        common_summary = list(csv.DictReader(stream))
    assert len(common_summary) == len(
        conditional_stage3.FROZEN_OPERATOR_SOURCES
    ) * len(conditional_stage3.COMPARISON_METHODS)
    assert {
        int(row["successful_source_records"]) for row in common_summary
    } == {6}
    assert {
        int(row["available_source_records_denominator"])
        for row in common_summary
    } == {8}
    assert {
        int(row["excluded_ineligible_source_records"])
        for row in common_summary
    } == {2}
    conditional_summary = next(
        row
        for row in common_summary
        if row["source_method_id"] == conditional_stage3.METHOD_ID
        and row["operator_source"] == "matching_p0"
    )
    assert float(conditional_summary["median_e_parallel"]) == 0.8
    assert float(conditional_summary["optimizer_updates"]) == 6000.0
    assert float(conditional_summary["model_parameters"]) == 101.0
    assert int(conditional_summary["n_e_parallel"]) == 6
    assert conditional_summary["confirmatory"] == "False"
    m1_summary = next(
        row
        for row in common_summary
        if row["source_method_id"] == "M1_observation_warm_start_sdedit"
        and row["operator_source"] == "matching_p0"
    )
    assert float(m1_summary["model_parameters"]) == 77.0
    assert float(m1_summary["optimizer_updates"]) == 3000.0
    assert m1_summary["training_walltime_seconds"] == ""
    assert m1_summary["training_cost_scope"] == "shared_pretrained_clean_prior"
    unet_summary = next(
        row
        for row in common_summary
        if row["source_method_id"]
        == "task_matched_multichannel_deterministic_UNet"
        and row["operator_source"] == "matching_p0"
    )
    assert float(unet_summary["model_parameters"]) == 99.0
    assert unet_summary["training_cost_scope"] == (
        "operator_scope_deterministic_training"
    )
    qy_summary = next(
        row
        for row in common_summary
        if row["source_method_id"] == "deterministic_Qy"
        and row["operator_source"] == "matching_p0"
    )
    assert qy_summary["model_parameters"] == ""
    assert float(qy_summary["median_total_function_evaluations_per_window"]) == 0.0

    failed_metric_path = deterministic_root / "sim31" / "metrics.csv"
    with failed_metric_path.open("r", encoding="utf-8", newline="") as stream:
        failed_rows = list(csv.DictReader(stream))
    failed_target = next(
        row
        for row in failed_rows
        if row["operator_source"] == "population_projector"
        and row["method_id"]
        == "M4_per_step_quadratic_proximal_q_consistency"
    )
    failed_target["status"] = "failed_method_numerical"
    conditional_stage3._write_csv(failed_metric_path, failed_rows)
    conditional_stage3.aggregate_conditional_development(
        config, run_dir=tmp_path / "failed_reference_run"
    )
    with common_summary_path.open("r", encoding="utf-8", newline="") as stream:
        failure_filtered_summary = list(csv.DictReader(stream))
    failed_m4_summary = next(
        row
        for row in failure_filtered_summary
        if row["source_method_id"]
        == "M4_per_step_quadratic_proximal_q_consistency"
        and row["operator_source"] == "population_projector"
    )
    assert int(failed_m4_summary["successful_source_records"]) == 5
    assert int(failed_m4_summary["failed_within_common_eligible"]) == 1
    assert int(failed_m4_summary["n_e_parallel"]) == 5

    failed_target["status"] = "success"
    failed_target["e_parallel"] = ""
    conditional_stage3._write_csv(failed_metric_path, failed_rows)
    with pytest.raises(
        ValueError,
        match="successful common-eligible row lacks finite required metric",
    ):
        conditional_stage3.aggregate_conditional_development(
            config, run_dir=tmp_path / "missing_metric_run"
        )
    failed_target["e_parallel"] = "0.9"
    conditional_stage3._write_csv(failed_metric_path, failed_rows)

    sim31_rows = [
        conditional_row("sim31", scope)
        for scope in conditional_stage3.FROZEN_OPERATOR_SOURCES[:-1]
    ]
    conditional_stage3._write_csv(
        conditional_root / "sim31" / "metrics.csv", sim31_rows
    )
    incomplete = conditional_stage3.aggregate_conditional_development(
        config, run_dir=tmp_path / "incomplete_run"
    )
    assert incomplete["status"] == "incomplete_exploratory_development_artifacts"
    assert incomplete["available_source_records_denominator"] == 8
    assert incomplete["common_eligible_source_record_ids"] == sorted(eligible)
    assert incomplete["expected_conditional_method_cells"] == 18
    assert incomplete["observed_conditional_method_cells"] == 17
    assert incomplete["unmatched_missing_conditional_method_cells"] == 1
    assert incomplete["unexpected_conditional_method_cells"] == 0
    assert incomplete["conditional_record_scope_cartesian_product_exact"] is False
    assert incomplete["failed_conditional_method_arms"] == 0
    assert incomplete["conditional_method_arm_failure_rate"] == 1.0 / 18.0
    assert incomplete["observed_conditional_method_arm_failure_rate"] == 0.0

    substituted_record = conditional_root / "sim32"
    conditional_stage3._write_csv(
        substituted_record / "metrics.csv",
        [
            conditional_row(
                "sim32", conditional_stage3.FROZEN_OPERATOR_SOURCES[0]
            )
        ],
    )
    (substituted_record / "result_summary.json").write_text(
        json.dumps(
            {
                "status": "completed_exploratory_conditional_diffusion_development",
                "common_eligibility_status": "included",
                "successful_method_arms": 1,
                "failed_method_arms": 0,
            }
        ),
        encoding="utf-8",
    )
    substituted = conditional_stage3.aggregate_conditional_development(
        config, run_dir=tmp_path / "substituted_run"
    )
    assert substituted["status"] == "incomplete_exploratory_development_artifacts"
    assert substituted["observed_conditional_method_cells"] == 18
    assert substituted["unmatched_missing_conditional_method_cells"] == 1
    assert substituted["unexpected_conditional_method_cells"] == 1
    assert substituted["conditional_record_scope_cartesian_product_exact"] is False
