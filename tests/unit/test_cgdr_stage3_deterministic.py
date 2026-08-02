"""Engineering contracts for the frozen deterministic-first stage-3 route."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from eeg_cgdr.data.mechanism import KLADOS_TRAIN_RECORDS
from eeg_cgdr.experiments.stage3_deterministic import (
    FROZEN_METHODS,
    FROZEN_OPERATOR_SOURCES,
    FROZEN_STATUS,
    _base_config,
    _checkpoint_contract,
    _metric_row,
    _scope_output_paths,
    _validate_deterministic_checkpoint_payload,
    _window_bundle,
    validate_stage3_config,
)
from eeg_cgdr.models import (
    DeterministicUNetConfig,
    TaskMatchedDeterministicUNet,
)
from saddpm.diffusion.gaussian_diffusion import GaussianDiffusion
from saddpm.diffusion.schedule import DiffusionConfig


CONFIG_PATH = Path("configs/cgdr/klados_stage3_deterministic_comparison.yaml")


def _config() -> dict:
    value = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _small_model() -> TaskMatchedDeterministicUNet:
    torch.manual_seed(8301)
    return TaskMatchedDeterministicUNet(
        DeterministicUNetConfig(
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
    ).eval()


def test_stage3_protocol_is_frozen_before_outcomes() -> None:
    config = _config()
    validate_stage3_config(config)

    assert config["frozen_current_status"] == FROZEN_STATUS
    assert tuple(config["frozen_comparison"]["methods"]) == FROZEN_METHODS
    assert (
        tuple(config["frozen_comparison"]["operator_sources"])
        == FROZEN_OPERATOR_SOURCES
    )
    assert config["frozen_comparison"]["broad_classifier_enabled"] is False
    assert (
        config["source_record_split"]["historical_records_are_fresh_evidence"]
        is False
    )
    assert config["deterministic_training"]["minimum_updates"] >= 3000
    assert (
        config["deterministic_training"]["operator_scope_isolated_checkpoints"]
        is True
    )
    assert config["deterministic_training"]["checkpoint_selection_scope"] == (
        "same_operator_scope_development_cells_only"
    )
    assert (
        config["deterministic_training"][
            "historical_query_clean_target_used_for_training_or_selection"
        ]
        is False
    )


def test_deterministic_model_has_only_frozen_deployment_inputs() -> None:
    parameters = inspect.signature(TaskMatchedDeterministicUNet.forward).parameters
    assert tuple(parameters) == (
        "self",
        "observed",
        "projector",
        "attenuation",
        "valid_time_mask",
    )
    assert TaskMatchedDeterministicUNet.visible_input_fields == (
        "observed_query_eeg",
        "operator_projector",
        "framewise_external_eog_attenuation",
        "valid_time_mask",
    )
    forbidden = {"clean_target", "participant_id", "query_outcome", "record_id"}
    assert forbidden.isdisjoint(parameters)


def test_deterministic_model_exposes_full_projector_and_is_padding_safe() -> None:
    model = _small_model()
    channels = model.config.eeg_channels
    assert model.unet.stem.in_channels == 2 * channels + 1 + channels * channels

    first = torch.randn(2, channels, 64)
    second = first.clone()
    second[:, :, 45:] = 1.0e6 * torch.randn_like(second[:, :, 45:])
    mask = torch.zeros(2, 64, dtype=torch.bool)
    mask[:, :45] = True
    attenuation = torch.linspace(0.1, 1.0, 64).repeat(2, 1)
    projector = torch.diag(torch.tensor([1.0, 0.0, 0.0]))

    with torch.no_grad():
        first_output = model(
            first,
            projector=projector,
            attenuation=attenuation,
            valid_time_mask=mask,
        )
        second_output = model(
            second,
            projector=projector,
            attenuation=attenuation,
            valid_time_mask=mask,
        )

    torch.testing.assert_close(
        first_output[:, :, :45], second_output[:, :, :45], rtol=0.0, atol=0.0
    )
    assert torch.count_nonzero(first_output[:, :, 45:]) == 0
    assert torch.count_nonzero(second_output[:, :, 45:]) == 0


def test_projector_entries_reach_backbone_even_when_projected_signal_is_zero() -> None:
    model = _small_model()
    captured: list[torch.Tensor] = []
    hook = model.unet.stem.register_forward_pre_hook(
        lambda _module, args: captured.append(args[0].detach().clone())
    )
    observed = torch.zeros(1, 3, 64)
    attenuation = torch.ones(1, 64)
    mask = torch.ones(1, 64, dtype=torch.bool)
    first_projector = torch.diag(torch.tensor([1.0, 0.0, 0.0]))
    second_projector = torch.diag(torch.tensor([0.0, 1.0, 0.0]))
    try:
        with torch.no_grad():
            model(
                observed,
                projector=first_projector,
                attenuation=attenuation,
                valid_time_mask=mask,
            )
            model(
                observed,
                projector=second_projector,
                attenuation=attenuation,
                valid_time_mask=mask,
            )
    finally:
        hook.remove()

    assert len(captured) == 2
    shared_channels = 2 * observed.shape[1] + 1
    torch.testing.assert_close(
        captured[0][:, :shared_channels],
        captured[1][:, :shared_channels],
        rtol=0.0,
        atol=0.0,
    )
    assert not torch.equal(
        captured[0][:, shared_channels:], captured[1][:, shared_channels:]
    )


def test_checkpoint_payload_requires_scope_isolated_selection_and_training_split() -> None:
    config = _config()
    base = _base_config(config)
    operator_scope = "population_projector"
    normalizer_state = {
        "mean": np.zeros(19, dtype=np.float64).tolist(),
        "standard_deviation": np.ones(19, dtype=np.float64).tolist(),
        "source_records": list(KLADOS_TRAIN_RECORDS),
        "sample_count": 1000,
    }
    payload = {
        "config": _checkpoint_contract(config, base, operator_scope),
        "step": 2999,
        "normalizer_state": normalizer_state,
        "extra": {
            "operator_scope": operator_scope,
            "operator_scope_deployable": True,
            "training_bundle_operator_sources": [operator_scope],
            "validation_bundle_operator_sources": [operator_scope],
        },
    }
    with pytest.raises(ValueError, match="minimum budget"):
        _validate_deterministic_checkpoint_payload(
            config, base, payload, operator_source=operator_scope
        )

    payload["step"] = 3000
    normalizer = _validate_deterministic_checkpoint_payload(
        config, base, payload, operator_source=operator_scope
    )
    assert normalizer.source_records == KLADOS_TRAIN_RECORDS

    with pytest.raises(ValueError, match="contract differs"):
        _validate_deterministic_checkpoint_payload(
            config, base, payload, operator_source="matching_p0"
        )

    payload["extra"]["validation_bundle_operator_sources"] = ["matching_p0"]
    with pytest.raises(ValueError, match="validation selection crossed"):
        _validate_deterministic_checkpoint_payload(
            config, base, payload, operator_source=operator_scope
        )
    payload["extra"]["validation_bundle_operator_sources"] = [operator_scope]

    payload["normalizer_state"] = {**normalizer_state, "source_records": [1, 2]}
    with pytest.raises(ValueError, match="sim01-sim30"):
        _validate_deterministic_checkpoint_payload(
            config, base, payload, operator_source=operator_scope
        )


def test_operator_scope_checkpoint_paths_are_new_and_disjoint() -> None:
    config = _config()
    base = _base_config(config)
    paths = {
        scope: _scope_output_paths(config, scope)
        for scope in FROZEN_OPERATOR_SOURCES
    }
    best_paths = {value["best_checkpoint"] for value in paths.values()}
    assert len(best_paths) == 3
    assert all("scope_isolated_v2" in str(path) for path in best_paths)
    assert all("deterministic_first/checkpoints" not in str(path) for path in best_paths)
    assert _checkpoint_contract(
        config, base, "query_derived_oracle_projector"
    )["operator_scope_deployable"] is False


def test_operator_scope_bundles_do_not_construct_other_scope_geometry(
    monkeypatch,
) -> None:
    config = _config()
    base = _base_config(config)
    channels = 3
    windows = 2
    length = 64
    projector = np.diag([1.0, 0.0, 0.0])
    native = type("Native", (), {"record_id": 1})()
    prepared = type(
        "Prepared",
        (),
        {
            "calibration": object(),
            "observed_windows": np.zeros((windows, channels, length)),
            "clean_windows": np.zeros((windows, channels, length)),
            "eog_windows": np.ones((windows, 2, length)),
            "valid_time_weight": np.ones((windows, length)),
            "observed_continuous": np.ones((channels, windows * length)),
            "clean_continuous": np.zeros((channels, windows * length)),
        },
    )()
    fit_calls: list[object] = []
    oracle_calls: list[object] = []

    def fake_fit(calibration, *_args, **_kwargs):
        fit_calls.append(calibration)
        transfer = type("Transfer", (), {"projector": projector})()
        return type("Outcome", (), {"transfer": transfer})()

    def fake_oracle(observed, clean, rank):
        oracle_calls.append((observed, clean, rank))
        return projector

    monkeypatch.setattr(
        "eeg_cgdr.experiments.stage3_deterministic.select_records",
        lambda _records, _source_records: [native],
    )
    monkeypatch.setattr(
        "eeg_cgdr.experiments.stage3_deterministic.prepare_mechanism_record",
        lambda *_args, **_kwargs: prepared,
    )
    monkeypatch.setattr(
        "eeg_cgdr.experiments.stage3_deterministic.fit_p0", fake_fit
    )
    monkeypatch.setattr(
        "eeg_cgdr.experiments.stage3_deterministic._oracle_projector", fake_oracle
    )

    def build(scope: str):
        return _window_bundle(
            config,
            base,
            records=(native,),
            normalizer=object(),
            population_projector=projector,
            source_records=(1,),
            operator_source=scope,
        )

    population = build("population_projector")
    assert set(population.operator_sources) == {"population_projector"}
    assert not fit_calls and not oracle_calls

    matching = build("matching_p0")
    assert set(matching.operator_sources) == {"matching_p0"}
    assert len(fit_calls) == 1 and not oracle_calls

    oracle = build("query_derived_oracle_projector")
    assert set(oracle.operator_sources) == {"query_derived_oracle_projector"}
    assert len(fit_calls) == 1 and len(oracle_calls) == 1


def test_query_derived_oracle_is_explicitly_nondeployable(monkeypatch) -> None:
    monkeypatch.setattr(
        "eeg_cgdr.experiments.stage3_deterministic._mechanism_metrics",
        lambda *_args, **_kwargs: {},
    )
    prepared = type(
        "Prepared",
        (),
        {
            "source_record": 31,
            "observed_continuous": np.zeros((2, 3)),
            "clean_continuous": np.zeros((2, 3)),
            "sampling_rate": 256,
        },
    )()
    common = {
        "partition": "development",
        "prepared": prepared,
        "method_id": "deterministic_Qy",
        "operator_source": "query_derived_oracle_projector",
        "effective_operator_source": "query_derived_oracle_projector",
        "restored": np.zeros((2, 3)),
        "projector": np.eye(2),
        "oracle": np.eye(2),
        "artifact_mask": np.zeros(3, dtype=bool),
        "runtime": {},
        "fallback_used": False,
    }
    oracle = _metric_row(**common, query_clean_target_used_by_method=True)
    assert oracle["query_clean_target_used_for_scoring_only"] is False
    assert oracle["deployable_operator_source"] is False
    assert oracle["operator_role"] == (
        "nondeployable_query_clean_mechanism_upper_bound"
    )

    matching = _metric_row(
        **{
            **common,
            "operator_source": "matching_p0",
            "effective_operator_source": "matching_p0",
        },
        query_clean_target_used_by_method=False,
    )
    assert matching["query_clean_target_used_for_scoring_only"] is True
    assert matching["deployable_operator_source"] is True


def test_m1_warm_start_supports_exact_frozen_network_call_budget() -> None:
    config = _config()
    base = _base_config(config)
    budget = config["frozen_comparison"]["budget"]["M1_network_calls"]
    warm_start = base["sampling"]["warm_start_timestep"]
    diffusion = GaussianDiffusion(
        DiffusionConfig(
            num_timesteps=base["diffusion"]["num_timesteps"],
            beta_start=base["diffusion"]["beta_start"],
            beta_end=base["diffusion"]["beta_end"],
            schedule=base["diffusion"]["schedule"],
        )
    )
    timesteps = diffusion.ddim_timesteps(budget, t_start=warm_start)

    assert len(timesteps) == 100
    assert timesteps[0] == 250
    assert timesteps[-1] == 0
    assert all(left > right for left, right in zip(timesteps, timesteps[1:]))


def test_real_record_integration_is_slurm_routed_and_not_scientific() -> None:
    config = _config()
    assert config["real_record_integration"] == {
        "source_record": "sim31",
        "complete_record": True,
        "operator_sources": list(FROZEN_OPERATOR_SOURCES),
        "batch_size": 4,
        "minimum_minibatches": 2,
        "checkpoint_role": "engineering_only_not_eligible_as_scientific_baseline",
    }
    job_script = Path("scripts/slurm/jobs/cgdr.sbatch").read_text(encoding="utf-8")
    submitter = Path("scripts/slurm/submit.sh").read_text(encoding="utf-8")
    assert "real-record-integration" in job_script
    assert "real-record-integration" in submitter
