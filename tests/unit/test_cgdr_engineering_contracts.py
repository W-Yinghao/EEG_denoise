"""Focused contracts for padding, W-rho consistency, and B6 runner provenance.

This module is executed only by the aggregate Slurm validation job.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import eeg_cgdr.experiments.b6_runner as b6_runner_module
from eeg_cgdr.data.klados import KladosRecord
from eeg_cgdr.data.mechanism import (
    KLADOS_NATIVE_CHANNEL_ORDER,
    write_mechanism_split_manifest,
)
from eeg_cgdr.experiments.b6_runner import run_deferred_b6_from_actual_split
from eeg_cgdr.inference.population import GuidanceStabilityConfig
from eeg_cgdr.inference.sampler_candidates import (
    RepairedSamplerRunner,
    SamplerMechanism,
    sampler_candidate,
)
from eeg_cgdr.inference.states import (
    CalibrationContextProjector,
    DatasetPopulationProjector,
    dataset_population_and_context_states,
    rho_interpolated_precision_state,
)
from saddpm.models.config import ModelConfig
from saddpm.models.unet1d import MaskedGroupNorm1D, UNet1D


def _small_unet() -> UNet1D:
    return UNet1D(
        ModelConfig(
            in_channels=3,
            out_channels=3,
            signal_length=64,
            base_channels=8,
            channel_mults=[1, 2, 4],
            num_res_blocks=1,
            groupnorm_groups=8,
            dropout=0.0,
            time_sinusoidal_dim=16,
            time_embed_dim=32,
            attention_length=8,
            attention_heads=4,
        ),
        subject_conditioned=False,
    ).eval()


def test_masked_groupnorm_excludes_invalid_time_from_statistics() -> None:
    torch.manual_seed(701)
    layer = MaskedGroupNorm1D(2, 4).eval()
    first = torch.randn(2, 4, 32)
    second = first.clone()
    second[:, :, 19:] = 1.0e6 * torch.randn_like(second[:, :, 19:])
    mask = torch.zeros(2, 1, 32, dtype=torch.bool)
    mask[:, :, :19] = True

    first_output = layer(first, mask)
    second_output = layer(second, mask)

    torch.testing.assert_close(
        first_output[:, :, :19], second_output[:, :, :19], rtol=0.0, atol=0.0
    )
    assert torch.count_nonzero(first_output[:, :, 19:]) == 0
    assert torch.count_nonzero(second_output[:, :, 19:]) == 0


def test_unet_internal_mask_makes_valid_output_invariant_to_padding_values() -> None:
    torch.manual_seed(702)
    model = _small_unet()
    first = torch.randn(2, 3, 64)
    second = first.clone()
    second[:, :, 45:] = 1.0e5 * torch.randn_like(second[:, :, 45:])
    mask = torch.zeros(2, 1, 64, dtype=torch.bool)
    mask[:, :, :45] = True
    timestep = torch.tensor([7, 13], dtype=torch.long)

    with torch.no_grad():
        first_output = model(first, timestep, valid_time_mask=mask)
        second_output = model(second, timestep, valid_time_mask=mask)

    torch.testing.assert_close(
        first_output[:, :, :45], second_output[:, :, :45], rtol=0.0, atol=0.0
    )
    assert torch.count_nonzero(first_output[:, :, 45:]) == 0
    assert torch.count_nonzero(second_output[:, :, 45:]) == 0


def _formal_precision_states(attenuation_value: float = 0.0):
    observation = torch.tensor(
        [[[2.0, -1.0, 0.5], [1.0, 3.0, -2.0]]], dtype=torch.float64
    )
    attenuation = torch.full(
        (1, 3), attenuation_value, dtype=torch.float64
    )
    valid = torch.ones(1, 3, dtype=torch.float64)
    pi0 = DatasetPopulationProjector(
        "fixture", "two_channel", torch.diag(torch.tensor([1.0, 0.0])), "train"
    )
    pic = CalibrationContextProjector(
        "fixture", "two_channel", torch.diag(torch.tensor([0.0, 1.0])), "support"
    )
    population, context = dataset_population_and_context_states(
        observation,
        attenuation=attenuation,
        valid_weight=valid,
        population_projector=pi0,
        context_projector=pic,
        base_precision=1.0,
        energy_scale=1.0,
    )
    return observation, population, context


def _runner_without_prior() -> RepairedSamplerRunner:
    inference = SimpleNamespace(stability=GuidanceStabilityConfig())
    return RepairedSamplerRunner(inference)  # type: ignore[arg-type]


def test_intermediate_rho_has_explicit_psd_wrho_proximal_name_only() -> None:
    _, population, context = _formal_precision_states(attenuation_value=0.5)
    state = rho_interpolated_precision_state(
        population,
        rho=0.25,
        calibration_accepted=True,
        context_state_factory=lambda: context,
    )
    runner = _runner_without_prior()
    geometry = runner._resolve_consistency(None, state)
    assert geometry.use_psd_precision
    assert geometry.projector is None
    assert geometry.semantics == "psd_precision_consistency_Wrho_not_a_projector"

    # At a=0.5, W0=diag(.25,1), WC=diag(1,.25), hence
    # W_rho(.25)=diag(.4375,.8125).  The state is PSD but not a projector.
    expected = torch.diag(torch.tensor([0.4375, 0.8125], dtype=torch.float64))
    torch.testing.assert_close(
        state.precision[0, 0], expected, rtol=0.0, atol=0.0
    )
    assert not torch.allclose(expected @ expected, expected)

    for hard_q_name in (
        SamplerMechanism.M2,
        SamplerMechanism.M3,
        SamplerMechanism.M4,
    ):
        with pytest.raises(ValueError, match="invalid for 0<rho<1"):
            runner._validate_mechanism_consistency(hard_q_name, geometry)
        with pytest.raises(ValueError, match="invalid for 0<rho<1"):
            runner.run(hard_q_name, state, seed=1, ddim_steps=2)
    assert runner._validate_mechanism_consistency(
        SamplerMechanism.WP, geometry
    ) == "intermediate_rho_psd_Wrho_quadratic_proximal_not_hard_Q"
    assert (
        sampler_candidate(
            "WP_per_step_quadratic_proximal_psd_wrho_consistency"
        ).mechanism
        == SamplerMechanism.WP
    )

    with pytest.raises(ValueError, match="not valid"):
        runner._resolve_consistency(context.context_consistency_projector, state)


def test_rho_endpoints_recover_pi0_and_pic_hard_q_geometry() -> None:
    _, population, context = _formal_precision_states(attenuation_value=0.5)
    runner = _runner_without_prior()
    torch.testing.assert_close(
        population.precision[0, 0],
        torch.diag(torch.tensor([0.25, 1.0], dtype=torch.float64)),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        context.precision[0, 0],
        torch.diag(torch.tensor([1.0, 0.25], dtype=torch.float64)),
        rtol=0.0,
        atol=0.0,
    )
    rho0 = runner._resolve_consistency(None, population)
    assert rho0.projector is not None
    assert rho0.semantics == "rho0_population_orthogonal_complement_projector"
    torch.testing.assert_close(
        rho0.projector, population.population_consistency_projector
    )
    rho1_state = rho_interpolated_precision_state(
        population,
        rho=1.0,
        calibration_accepted=True,
        context_state_factory=lambda: context,
    )
    rho1 = runner._resolve_consistency(None, rho1_state)
    assert rho1.projector is not None
    assert rho1.semantics == "rho1_context_orthogonal_complement_projector"
    torch.testing.assert_close(rho1.projector, context.context_consistency_projector)

    with pytest.raises(ValueError, match="reserved for a formal intermediate"):
        runner._validate_mechanism_consistency(SamplerMechanism.WP, rho0)
    with pytest.raises(ValueError, match="reserved for a formal intermediate"):
        runner._validate_mechanism_consistency(SamplerMechanism.WP, rho1)


def test_intermediate_rho_proximal_is_exact_psd_quadratic_solve() -> None:
    observation, population, context = _formal_precision_states()
    state = rho_interpolated_precision_state(
        population,
        rho=0.25,
        calibration_accepted=True,
        context_state_factory=lambda: context,
    )
    runner = _runner_without_prior()
    value = torch.full_like(observation, 0.4)
    strength = 2.0
    actual = runner._psd_precision_proximal(value, state, strength)
    precision = state.precision[0, 0]
    expected_frames = []
    identity = torch.eye(2, dtype=torch.float64)
    for frame in range(observation.shape[-1]):
        residual = observation[0, :, frame] - value[0, :, frame]
        update = torch.linalg.solve(
            identity + strength * precision,
            strength * precision @ residual,
        )
        expected_frames.append(value[0, :, frame] + update)
    expected = torch.stack(expected_frames, dim=1).unsqueeze(0)
    torch.testing.assert_close(actual, expected)


def _backup_config(enabled: bool = True) -> dict:
    return {
        "backup_id": "B6",
        "formal_name": "POP-SHRINK",
        "enabled": enabled,
        "selection": {
            "scope": "development_only",
            "one_global_gamma": True,
            "gamma_candidates": [0.25, 0.5, 0.75],
            "endpoints_as_controls": [0.0, 1.0],
        },
        "operator": {
            "rank": 2,
            "minimum_spectral_eigengap": 1.0e-6,
            "required_population_fit_scope": "outer_training_only",
            "required_context_fit_scope": "support_only",
        },
    }


def _b6_fixture(tmp_path: Path):
    split = tmp_path / "split.csv"
    write_mechanism_split_manifest(split)
    rng = np.random.default_rng(703)
    basis, _ = np.linalg.qr(rng.normal(size=(19, 2)))
    pi0 = basis @ basis.T
    population_path = tmp_path / "population_state.json"
    preprocessing_id = (
        "klados_mechanism_resample_200_to_256__window_512__"
        "train_clean_channel_moments__padding_zero__support_eog_zscore"
    )
    population_path.write_text(
        json.dumps(
            {
                "dataset_id": "klados_bamidis_v4",
                "montage_id": "klados_v4_19ch_native_order_256hz",
                "source": "all_training_source_records_sim01_sim30",
                "channel_order": list(KLADOS_NATIVE_CHANNEL_ORDER),
                "training_source_records": list(range(1, 31)),
                "source_sampling_rate": 200,
                "target_sampling_rate": 256,
                "reference_id": (
                    "linked_mastoids_native_odd_left_even_right_midline_average"
                ),
                "preprocessing_id": preprocessing_id,
                "fit_scope": "joint_concatenation_of_all_30_training_source_records",
                "records_total": 30,
                "projector": pi0.tolist(),
            }
        ),
        encoding="utf-8",
    )
    config = {
        "seed": 703,
        "klados": {
            "split_manifest": str(split),
            "data_root": "/fixture/klados_v4",
            "contaminated": "Contaminated_Data.mat",
            "clean": "Pure_Data.mat",
            "heog": "HEOG.mat",
            "veog": "VEOG.mat",
            "source_sampling_rate": 200,
            "channel_order": list(KLADOS_NATIVE_CHANNEL_ORDER),
            "reference_id": (
                "linked_mastoids_native_odd_left_even_right_midline_average"
            ),
            "training_source_records": list(range(1, 31)),
            "calibration_seconds": 10.0,
            "guard_seconds": 1.0,
        },
        "preprocessing": {
            "target_sampling_rate": 256,
            "window_samples": 512,
            "normalization": (
                "per-channel moments from complete clean training source records only"
            ),
            "padding_value_after_normalization": 0.0,
        },
        "p0": {
            "target_rank": 2,
            "ridge_lambda": 0.01,
            "maximum_reference_condition": 1.0e4,
            "minimum_singular_ratio": 0.0,
            "minimum_movement_coverage": 0.0,
            "movement_threshold": 0.0,
            "bootstrap_replicates": 0,
            "bootstrap_block_samples": 512,
            "minimum_bootstrap_success": 0.0,
            "maximum_bootstrap_median_distance": float("inf"),
            "maximum_bootstrap_q90_distance": float("inf"),
            "reference_standardization": "support_channel_zscore",
        },
        "outputs": {"population_state": str(population_path)},
    }
    samples = 2400
    raw_eog = rng.normal(size=(2, samples))
    clean = rng.normal(size=(19, samples))
    contaminated = clean + 4.0 * basis @ raw_eog
    records = [
        KladosRecord(
            record_id=record_id,
            clean=clean,
            contaminated=contaminated,
            veog=raw_eog[0],
            heog=raw_eog[1],
        )
        for record_id in range(1, 55)
    ]
    population = DatasetPopulationProjector(
        "klados_bamidis_v4",
        "klados_v4_19ch_native_order_256hz",
        pi0,
        "all_training_source_records_sim01_sim30",
    )
    return config, population, records


def test_b6_runner_derives_compatibility_and_fit_scopes_from_actual_split(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, population, records = _b6_fixture(tmp_path)
    loader_requests: list[dict] = []

    def formal_loader(loader_config: dict) -> list[KladosRecord]:
        loader_requests.append(loader_config)
        return records

    monkeypatch.setattr(b6_runner_module, "load_klados_records", formal_loader)
    result = run_deferred_b6_from_actual_split(
        config=config,
        backup_config=_backup_config(),
        population_projector=population,
        source_record=31,
        gamma=0.25,
    )
    assert result.outcome.status == "eligible"
    assert result.compatibility is not None
    assert result.compatibility.channel_order == KLADOS_NATIVE_CHANNEL_ORDER
    assert result.population_source_records == tuple(
        f"sim{index:02d}" for index in range(1, 31)
    )
    assert result.context_support_record == "sim31"
    assert result.partition == "development"
    assert result.population_fit_scope == "outer_training_only"
    assert result.context_fit_scope == "support_only"
    assert loader_requests == [
        {
            "data_root": "/fixture/klados_v4",
            "files": {
                "contaminated": "Contaminated_Data.mat",
                "clean": "Pure_Data.mat",
                "heog": "HEOG.mat",
                "veog": "VEOG.mat",
            },
            "official_description": {"records": 54},
        }
    ]


def test_b6_invalid_runner_contract_returns_pop_without_context(tmp_path: Path) -> None:
    config, population, _ = _b6_fixture(tmp_path)
    config["klados"]["reference_id"] = "caller_invented_reference"
    result = run_deferred_b6_from_actual_split(
        config=config,
        backup_config=_backup_config(),
        population_projector=population,
        source_record=31,
        gamma=0.25,
    )
    assert result.use_pop
    assert result.outcome.projector is None
    assert not result.outcome.context_projector_constructed


def test_b6_pop_endpoint_does_not_load_or_construct_context_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, population, _ = _b6_fixture(tmp_path)

    def forbidden_loader(_: dict) -> list[KladosRecord]:
        raise AssertionError("gamma=0 must short-circuit before formal data loading")

    monkeypatch.setattr(b6_runner_module, "load_klados_records", forbidden_loader)
    result = run_deferred_b6_from_actual_split(
        config=config,
        backup_config=_backup_config(),
        population_projector=population,
        source_record=31,
        gamma=0.0,
    )
    assert result.outcome.status == "eligible"
    assert np.array_equal(result.outcome.projector, population.projector)
    assert result.context_support_record is None
    assert result.context_fit_scope is None
    assert not result.outcome.context_projector_constructed


def test_b6_runner_rejects_record_outside_actual_development_split_before_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, population, _ = _b6_fixture(tmp_path)

    def forbidden_loader(_: dict) -> list[KladosRecord]:
        raise AssertionError("invalid split record must be rejected before loading")

    monkeypatch.setattr(b6_runner_module, "load_klados_records", forbidden_loader)
    result = run_deferred_b6_from_actual_split(
        config=config,
        backup_config=_backup_config(),
        population_projector=population,
        source_record=37,
        gamma=0.25,
    )
    assert result.use_pop
    assert result.outcome.projector is None
    assert result.outcome.reasons == (
        "runner_contract:B6_context_is_not_in_the_development_partition",
    )


def test_b6_runner_signature_has_no_caller_provenance_or_query_escape_hatch() -> None:
    parameters = inspect.signature(run_deferred_b6_from_actual_split).parameters
    forbidden = {
        "dataset_id",
        "montage_id",
        "reference_id",
        "preprocessing_id",
        "channel_order",
        "population_fit_scope",
        "context_fit_scope",
        "context_projector",
        "calibration",
        "support_record",
        "eeg",
        "eog",
        "query",
        "clean_target",
    }
    assert forbidden.isdisjoint(parameters)
    assert "source_record" in parameters
