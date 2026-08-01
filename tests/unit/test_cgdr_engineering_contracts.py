"""Focused contracts for padding, W-rho consistency, and B6 runner provenance.

This module is executed only by the aggregate Slurm validation job.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from eeg_cgdr.data.mechanism import (
    KLADOS_NATIVE_CHANNEL_ORDER,
    KladosMechanismRecord,
    support_view,
    write_mechanism_split_manifest,
)
from eeg_cgdr.experiments.b6_runner import run_deferred_b6_from_actual_split
from eeg_cgdr.inference.population import GuidanceStabilityConfig
from eeg_cgdr.inference.sampler_candidates import RepairedSamplerRunner
from eeg_cgdr.inference.states import (
    CalibrationContextProjector,
    DatasetPopulationProjector,
    dataset_population_and_context_states,
    rho_interpolated_precision_state,
)
from eeg_cgdr.operators.p0 import CalibrationBatch
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


def _formal_precision_states():
    observation = torch.tensor(
        [[[2.0, -1.0, 0.5], [1.0, 3.0, -2.0]]], dtype=torch.float64
    )
    attenuation = torch.zeros(1, 3, dtype=torch.float64)
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


def test_intermediate_rho_uses_psd_precision_and_rejects_fixed_pic() -> None:
    observation, population, context = _formal_precision_states()
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

    value = torch.zeros_like(observation)
    actual = runner._psd_precision_consistency(value, state)
    # W_rho=diag(0.25,0.75), normalized by lambda_max=0.75.
    expected_gain = torch.diag(torch.tensor([1.0 / 3.0, 1.0], dtype=torch.float64))
    expected = torch.einsum("cd,bdl->bcl", expected_gain, observation)
    torch.testing.assert_close(actual, expected)

    try:
        runner._resolve_consistency(context.context_consistency_projector, state)
    except ValueError as exc:
        assert "not valid" in str(exc)
    else:  # pragma: no cover - acceptance guard
        raise AssertionError("intermediate rho accepted a fixed context projector")


def test_rho_endpoints_recover_pi0_and_pic_hard_q_geometry() -> None:
    _, population, context = _formal_precision_states()
    runner = _runner_without_prior()
    rho0 = runner._resolve_consistency(None, population)
    assert rho0.projector is not None
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
    torch.testing.assert_close(rho1.projector, context.context_consistency_projector)


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
            "source_sampling_rate": 200,
            "channel_order": list(KLADOS_NATIVE_CHANNEL_ORDER),
            "reference_id": (
                "linked_mastoids_native_odd_left_even_right_midline_average"
            ),
            "training_source_records": list(range(1, 31)),
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
    raw_eog = rng.normal(size=(2, 2560))
    eog = (raw_eog - raw_eog.mean(axis=1, keepdims=True)) / raw_eog.std(
        axis=1, keepdims=True
    )
    eeg = basis @ eog + 0.001 * rng.normal(size=(19, 2560))
    calibration = CalibrationBatch(
        eeg=eeg,
        eog=eog,
        participant="unresolved_source_record",
        source_record="sim31",
        sampling_rate=256.0,
    )
    population = DatasetPopulationProjector(
        "klados_bamidis_v4",
        "klados_v4_19ch_native_order_256hz",
        pi0,
        "all_training_source_records_sim01_sim30",
    )
    query_samples = 512
    prepared = KladosMechanismRecord(
        source_record=31,
        calibration=calibration,
        observed_windows=np.zeros((1, 19, query_samples), dtype=np.float64),
        clean_windows=np.zeros((1, 19, query_samples), dtype=np.float64),
        eog_windows=np.zeros((1, 2, query_samples), dtype=np.float64),
        valid_time_weight=np.ones((1, query_samples), dtype=np.float64),
        observed_continuous=np.zeros((19, query_samples), dtype=np.float64),
        clean_continuous=np.zeros((19, query_samples), dtype=np.float64),
        eog_continuous=np.zeros((2, query_samples), dtype=np.float64),
        eog_calibration_mean=np.zeros((2, 1), dtype=np.float64),
        eog_calibration_standard_deviation=np.ones((2, 1), dtype=np.float64),
        sampling_rate=256,
        calibration_start_seconds=0.0,
        calibration_end_seconds=10.0,
        guard_seconds=1.0,
        query_start_seconds=11.0,
        query_end_seconds=13.0,
    )
    return config, population, prepared


def test_b6_runner_derives_compatibility_and_fit_scopes_from_actual_split(
    tmp_path: Path,
) -> None:
    config, population, prepared = _b6_fixture(tmp_path)
    result = run_deferred_b6_from_actual_split(
        config=config,
        backup_config=_backup_config(),
        population_projector=population,
        support_record=support_view(prepared),
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


def test_b6_invalid_runner_contract_returns_pop_without_context(tmp_path: Path) -> None:
    config, population, prepared = _b6_fixture(tmp_path)
    config["klados"]["reference_id"] = "caller_invented_reference"
    result = run_deferred_b6_from_actual_split(
        config=config,
        backup_config=_backup_config(),
        population_projector=population,
        support_record=support_view(prepared),
        gamma=0.25,
    )
    assert result.use_pop
    assert result.outcome.projector is None
    assert not result.outcome.context_projector_constructed


def test_b6_pop_endpoint_does_not_read_context_support(tmp_path: Path) -> None:
    config, population, _ = _b6_fixture(tmp_path)
    result = run_deferred_b6_from_actual_split(
        config=config,
        backup_config=_backup_config(),
        population_projector=population,
        support_record=None,
        gamma=0.0,
    )
    assert result.outcome.status == "eligible"
    assert np.array_equal(result.outcome.projector, population.projector)
    assert result.context_support_record is None
    assert result.context_fit_scope is None
    assert not result.outcome.context_projector_constructed


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
        "query",
        "clean_target",
    }
    assert forbidden.isdisjoint(parameters)
