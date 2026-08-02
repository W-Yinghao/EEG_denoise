"""Unit contracts for the frozen natural-EEG SGEYESUB comparator."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from eeg_cgdr.experiments.sgeyesub_diffusion import (
    CONDITIONAL_METHOD_ID,
    DETERMINISTIC_METHOD_ID,
    NATURAL_DECISION_FAIL,
    NATURAL_DECISION_INCONCLUSIVE,
    NATURAL_DECISION_PASS,
    PROTOCOL_ID,
    REPORTED_ARM_IDS,
    aggregate_sgeyesub_diffusion_metrics,
    assert_legal_inference_fields,
    build_frozen_sgeyesub_folds,
    build_within_stem_weak_pairs,
    eeg_only_frame_attenuation,
    fit_full_block1_p0,
    fit_outer_training_normalizer,
    fit_outer_training_population_p0,
    fit_outer_training_projected_energy_scale,
    freeze_query_arm_outputs,
    matching_soft_proximal,
    trial_local_nonoverlap_windows,
    validate_sgeyesub_diffusion_config,
)
from eeg_cgdr.operators import P0Config, P0Transfer


CONFIG_PATH = Path("configs/cgdr/sgeyesub_diffusion_incremental.yaml")


def _config() -> dict:
    value = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _protocol_rows(config: dict) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    split = config["split"]
    for partition in ("development", "evaluation"):
        for fold in split[f"{partition}_folds"]:
            layout_id = str(fold["layout_id"])
            for stem in fold["heldout_stems"]:
                rows.append(
                    {
                        "study": fold["study"],
                        "participant_stem": stem,
                        "recording_key": f"{fold['study']}/{stem}",
                        "layout_id": layout_id,
                        "reference_cell_id": "release_preprocessed_as_delivered",
                        "sampling_rate_hz": float(fold["sampling_rate_hz"]),
                        "partition": partition,
                        "status": "metadata_ready",
                    }
                )
    rows.append(
        {
            "study": "study05",
            "participant_stem": "study05_p42",
            "recording_key": "study05/study05_p42",
            "layout_id": "layout06",
            "reference_cell_id": "release_preprocessed_as_delivered",
            "sampling_rate_hz": 256.0,
            "partition": "evaluation",
            "status": "blocked_no_population",
        }
    )
    return rows


def test_frozen_config_is_natural_eeg_weak_supervision_not_clean_recovery() -> None:
    config = _config()
    validate_sgeyesub_diffusion_config(config)
    assert config["protocol_id"] == PROTOCOL_ID
    assert config["dataset"]["clean_target_available"] is False
    assert config["dataset"]["weak_target_semantics"] == (
        "low_artifact_observed_EEG_not_clean_truth"
    )
    assert config["matched_comparison"]["same_successful_optimizer_updates"] == 6000
    assert config["matched_comparison"]["gradient_clip_norm"] == 1.0
    assert config["matched_comparison"]["checkpoint_interval_successful_updates"] == 250
    assert config["matched_comparison"]["model_seed_by_arm"] == {
        DETERMINISTIC_METHOD_ID: 20260802,
        CONDITIONAL_METHOD_ID: 20260803,
    }
    assert config["matched_comparison"]["conditional_diffusion"][
        "ddim_network_calls_per_window"
    ] == 100
    assert config["evaluation"]["metric_coordinate"] == (
        "outer_training_channel_zscore"
    )
    assert config["evaluation"]["metrics"].count("observation_change_ratio") == 1
    assert config["evaluation"]["compute_metric_contract"] == {
        "deterministic_network_evaluations_per_window": 1,
        "conditional_DDIM_network_evaluations_per_window": 100,
        "report_batched_forward_invocations_separately": True,
        "report_total_and_per_window_latency": True,
    }


@pytest.mark.parametrize(
    ("path", "value", "message"),
    (
        (("dataset", "clean_target_available"), True, "no verified clean target"),
        (("matched_comparison", "same_successful_optimizer_updates"), 5999, "training budget"),
        (("matched_comparison", "gradient_clip_norm"), 0.5, "training budget"),
        (
            (
                "matched_comparison",
                "model_seed_by_arm",
                CONDITIONAL_METHOD_ID,
            ),
            20260802,
            "per-arm model seeds",
        ),
        (("outer_fold_fit", "eeg_only_attenuation", "query_EOG_input"), "allowed", "attenuation rule"),
        (("windowing", "padding"), "allowed", "window contract"),
        (("evaluation", "metric_coordinate"), "raw_microvolts", "metric coordinate"),
        (
            (
                "evaluation",
                "compute_metric_contract",
                "conditional_DDIM_network_evaluations_per_window",
            ),
            99,
            "compute metric contract",
        ),
    ),
)
def test_config_validator_fails_closed(path: tuple[str, ...], value: object, message: str) -> None:
    config = deepcopy(_config())
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=message):
        validate_sgeyesub_diffusion_config(config)


def test_config_metric_list_is_unique_and_uses_one_observation_change_name() -> None:
    config = deepcopy(_config())
    config["evaluation"]["metrics"].append("observation_change_ratio")
    with pytest.raises(ValueError, match="metric list changed or contains duplicates"):
        validate_sgeyesub_diffusion_config(config)


def test_explicit_five_fold_plan_is_participant_disjoint_and_complete() -> None:
    config = _config()
    plan = build_frozen_sgeyesub_folds(config, _protocol_rows(config))
    assert len(plan.development_folds) == 10
    assert len(plan.evaluation_folds) == 15
    assert plan.compatible_denominator == 43
    assert plan.availability_denominator == 44
    assert plan.blocked_recording_keys == ("study05/study05_p42",)

    development = []
    evaluation = []
    for fold in plan.folds:
        assert set(fold.training_recording_keys).isdisjoint(
            fold.heldout_recording_keys
        )
        assert all(key.startswith(f"{fold.study}/") for key in fold.training_recording_keys)
        assert all(key.startswith(f"{fold.study}/") for key in fold.heldout_recording_keys)
        (development if fold.partition == "development" else evaluation).extend(
            fold.heldout_recording_keys
        )
    assert len(development) == len(set(development)) == 15
    assert len(evaluation) == len(set(evaluation)) == 43
    assert "study05/study05_p42" not in evaluation


def test_trial_windows_never_cross_trials_and_have_no_padding() -> None:
    rate = 4
    samples_per_trial = 32
    signal = np.vstack(
        (
            np.concatenate((np.arange(32), 1000 + np.arange(32))),
            np.concatenate((100 + np.arange(32), 2000 + np.arange(32))),
        )
    )
    windows = trial_local_nonoverlap_windows(
        signal,
        samples_per_trial=samples_per_trial,
        sampling_rate_hz=rate,
        recording_key="study01/study01_p01",
    )
    assert windows.values.shape == (8, 2, 8)
    assert np.all(windows.valid_time_mask)
    assert [origin.trial_ordinal for origin in windows.origins] == [0] * 4 + [1] * 4
    np.testing.assert_array_equal(windows.values[3, 0], np.arange(24, 32))
    np.testing.assert_array_equal(windows.values[4, 0], 1000 + np.arange(8))

    with pytest.raises(ValueError, match="incomplete trial/window"):
        trial_local_nonoverlap_windows(
            signal[:, :-1],
            samples_per_trial=samples_per_trial,
            sampling_rate_hz=rate,
            recording_key="study01/study01_p01",
        )


def test_normalizer_uses_only_named_outer_training_block1() -> None:
    signals = {
        "study02/train_a": np.asarray([[0.0, 2.0], [10.0, 14.0]]),
        "study02/train_b": np.asarray([[4.0, 6.0], [18.0, 22.0]]),
        "study02/heldout": np.full((2, 50), 1.0e9),
    }
    normalizer = fit_outer_training_normalizer(
        signals,
        ("study02/train_a", "study02/train_b"),
    )
    np.testing.assert_allclose(normalizer.mean, [3.0, 16.0])
    assert normalizer.training_recording_keys == (
        "study02/train_a",
        "study02/train_b",
    )
    changed = dict(signals)
    changed["study02/heldout"] = np.full((2, 3), -1.0e12)
    second = fit_outer_training_normalizer(
        changed,
        ("study02/train_a", "study02/train_b"),
    )
    np.testing.assert_array_equal(normalizer.mean, second.mean)
    np.testing.assert_array_equal(
        normalizer.standard_deviation, second.standard_deviation
    )


def test_projected_energy_attenuation_is_eeg_only_and_matches_formula() -> None:
    observed = np.asarray([[3.0, 0.0], [4.0, 2.0]])
    projector = np.eye(2)
    scale = fit_outer_training_projected_energy_scale(((observed, projector),))
    assert scale == pytest.approx(3.5)
    attenuation = eeg_only_frame_attenuation(observed, projector, scale)
    expected_ratio = np.asarray([5.0, 2.0]) / 3.5
    np.testing.assert_allclose(
        attenuation,
        np.sqrt(1.0 / (1.0 + np.square(expected_ratio))),
    )
    assert_legal_inference_fields(
        (
            "observed_query_EEG",
            "operator_projector",
            "shared_framewise_attenuation",
            "valid_time_mask",
        )
    )
    with pytest.raises(ValueError, match="evaluation-only"):
        assert_legal_inference_fields(("observed_query_EEG", "query_external_EOG"))
    with pytest.raises(ValueError, match="evaluation-only"):
        assert_legal_inference_fields(("observed_query_EEG", "external_eog_signal"))


def _transfer_for_weak_pairs(samples: int, eog: np.ndarray) -> P0Transfer:
    mean = eog.mean(axis=1)
    scale = eog.std(axis=1)
    return P0Transfer(
        transfer_matrix=np.eye(2),
        eeg_subspace_basis=np.eye(2),
        projector=np.eye(2),
        predicted_contamination=np.zeros((2, samples)),
        eog_mean=np.zeros((2, 1)),
        eeg_mean=np.zeros((2, 1)),
        rank=2,
        diagnostics={
            "fit_scope": "full_block1",
            "samples": samples,
            "ridge_lambda": 0.01,
            "eog_fit_coordinate": (
                "standardized_from_source_full_block1_mean_std"
            ),
            "raw_eog_standardization_mean": mean.tolist(),
            "raw_eog_standardization_standard_deviation": scale.tolist(),
        },
    )


def test_weak_pairs_use_distinct_within_stem_windows_and_are_deterministic() -> None:
    rate = 4
    samples_per_trial = 32
    samples = samples_per_trial
    eeg = np.vstack((np.linspace(-1.0, 1.0, samples), np.linspace(1.0, 3.0, samples)))
    eog = np.vstack(
        (
            np.linspace(-2.0, 2.0, samples),
            np.sin(np.linspace(0.0, 4.0 * np.pi, samples)),
        )
    )
    labels = np.concatenate(
        (
            np.full(8, 6),
            np.full(8, 1),
            np.full(8, 6),
            np.full(8, 2),
        )
    )
    transfer = _transfer_for_weak_pairs(samples, eog)
    first = build_within_stem_weak_pairs(
        eeg,
        eog,
        labels,
        transfer=transfer,
        samples_per_trial=samples_per_trial,
        sampling_rate_hz=rate,
        recording_key="study01/study01_p01",
        projected_energy_scale=1.0,
    )
    second = build_within_stem_weak_pairs(
        eeg,
        eog,
        labels,
        transfer=transfer,
        samples_per_trial=samples_per_trial,
        sampling_rate_hz=rate,
        recording_key="study01/study01_p01",
        projected_energy_scale=1.0,
    )
    assert first.observed.shape == first.weak_target.shape == (2, 2, 8)
    assert len(first.target_origins) == len(first.artifact_origins)
    assert all(
        left != right
        for left, right in zip(first.target_origins, first.artifact_origins)
    )
    assert {origin.start_sample for origin in first.target_origins} == {0, 16}
    assert {origin.start_sample for origin in first.artifact_origins} == {8, 24}
    np.testing.assert_array_equal(first.observed, second.observed)
    np.testing.assert_array_equal(first.weak_target, second.weak_target)
    assert first.target_origins == second.target_origins
    assert first.artifact_origins == second.artifact_origins
    assert not np.array_equal(first.observed, first.weak_target)


def test_real_full_block_p0_fit_declares_standardized_coordinates_for_weak_pairs() -> None:
    rate = 4
    samples = 32
    rng = np.random.default_rng(20260802)
    eog = rng.normal(size=(2, samples))
    standardized = (eog - eog.mean(axis=1, keepdims=True)) / eog.std(
        axis=1, keepdims=True
    )
    transfer_matrix = np.asarray([[1.0, 0.2], [-0.3, 0.8], [0.4, -0.5]])
    eeg = transfer_matrix @ standardized + 0.01 * rng.normal(size=(3, samples))
    p0 = P0Config(
        target_rank=2,
        ridge_lambda=0.01,
        maximum_reference_condition=1.0e9,
        minimum_singular_ratio=0.0,
        minimum_movement_coverage=0.0,
        bootstrap_replicates=1,
        bootstrap_block_samples=8,
        minimum_bootstrap_success=0.0,
        maximum_bootstrap_median_distance=float("inf"),
        maximum_bootstrap_q90_distance=float("inf"),
        seed=20260802,
    )
    fitted = fit_full_block1_p0(
        eeg,
        eog,
        recording_key="study01/study01_p01",
        sampling_rate_hz=rate,
        config=p0,
        movement_threshold=0.0,
    )
    assert fitted.outcome.status == "eligible"
    assert fitted.outcome.transfer is not None
    assert fitted.outcome.transfer.diagnostics["fit_scope"] == (
        "full_block1_per_stem"
    )
    labels = np.concatenate(
        (
            np.full(8, 6),
            np.full(8, 1),
            np.full(8, 6),
            np.full(8, 2),
        )
    )
    bundle = build_within_stem_weak_pairs(
        eeg,
        eog,
        labels,
        transfer=fitted.outcome.transfer,
        samples_per_trial=samples,
        sampling_rate_hz=rate,
        recording_key="study01/study01_p01",
        projected_energy_scale=1.0,
    )
    assert bundle.observed.shape == (2, 3, 8)


def test_population_p0_standardizes_each_outer_training_source_separately() -> None:
    rng = np.random.default_rng(91)
    eog_a = rng.normal(size=(2, 32))
    eog_b = 100.0 + 5.0 * rng.normal(size=(2, 32))
    eeg_a = np.vstack((eog_a, eog_a[:1]))
    eeg_b = np.vstack((eog_b, eog_b[:1]))
    p0 = P0Config(
        target_rank=2,
        ridge_lambda=0.01,
        maximum_reference_condition=1.0e9,
        minimum_singular_ratio=0.0,
        minimum_movement_coverage=0.0,
        bootstrap_replicates=1,
        bootstrap_block_samples=8,
        minimum_bootstrap_success=0.0,
        maximum_bootstrap_median_distance=float("inf"),
        maximum_bootstrap_q90_distance=float("inf"),
        seed=20260802,
    )
    result = fit_outer_training_population_p0(
        {"study02/a": eeg_a, "study02/b": eeg_b},
        {"study02/a": eog_a, "study02/b": eog_b},
        ("study02/a", "study02/b"),
        sampling_rate_hz=200,
        p0_config=p0,
        movement_threshold=0.0,
    )
    assert result.training_recording_keys == ("study02/a", "study02/b")
    assert result.sample_count == 64
    assert result.outcome.diagnostics["source_eog_standardization"] == (
        "per_source_full_block1_mean_std"
    )
    assert result.source_eog_statistics["study02/a"]["mean"] != (
        result.source_eog_statistics["study02/b"]["mean"]
    )

    with pytest.raises(ValueError, match="at least two unique same-cell"):
        fit_outer_training_population_p0(
            {"study02/a": eeg_a},
            {"study02/a": eog_a},
            ("study02/a",),
            sampling_rate_hz=200,
            p0_config=p0,
            movement_threshold=0.0,
        )


def test_soft_proximal_uses_the_frozen_closed_form() -> None:
    observed = np.asarray([[2.0, 4.0], [3.0, 5.0]])
    projector = np.diag([1.0, 0.0])
    attenuation = np.asarray([0.0, 0.5])
    output = matching_soft_proximal(observed, projector, attenuation)
    expected = observed - (1.0 - attenuation**2)[None, :] * (
        projector @ observed
    )
    np.testing.assert_allclose(output, expected)


def test_all_arm_outputs_are_copied_and_frozen_before_annotation_open() -> None:
    source = {
        DETERMINISTIC_METHOD_ID: np.ones((2, 8)),
        CONDITIONAL_METHOD_ID: np.zeros((2, 8)),
    }
    frozen = freeze_query_arm_outputs(
        source,
        expected_arm_ids=(DETERMINISTIC_METHOD_ID, CONDITIONAL_METHOD_ID),
        recording_key="study02/study02_p02",
    )
    source[DETERMINISTIC_METHOD_ID][:] = 99.0
    assert not frozen.outputs[DETERMINISTIC_METHOD_ID].flags.writeable
    np.testing.assert_array_equal(
        frozen.outputs[DETERMINISTIC_METHOD_ID], np.ones((2, 8))
    )
    with pytest.raises(ValueError, match="all and only"):
        freeze_query_arm_outputs(
            {DETERMINISTIC_METHOD_ID: np.ones((2, 8))},
            expected_arm_ids=(DETERMINISTIC_METHOD_ID, CONDITIONAL_METHOD_ID),
            recording_key="study02/study02_p02",
        )


def _evaluation_rows(config: dict, *, mode: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fold in config["split"]["evaluation_folds"]:
        for stem in fold["heldout_stems"]:
            recording_key = f"{fold['study']}/{stem}"
            deterministic = {
                "eog_coherence_reduction": 0.10,
                "matching_projector_attenuation_db": 1.0,
                "nonartifact_observation_preservation": 0.90,
                "reference_free_psd_distortion": 0.12,
                "reference_free_covariance_distortion": 0.12,
                "condition_erp_observation_relative_preservation": 0.90,
            }
            conditional = {
                "eog_coherence_reduction": 0.20,
                "matching_projector_attenuation_db": 2.0,
                "nonartifact_observation_preservation": 0.91,
                "reference_free_psd_distortion": 0.10,
                "reference_free_covariance_distortion": 0.10,
                "condition_erp_observation_relative_preservation": 0.91,
            }
            if mode == "clear_primary_fail":
                conditional["eog_coherence_reduction"] = 0.0
                conditional["matching_projector_attenuation_db"] = 0.0
            elif mode == "safety_fail":
                conditional["nonartifact_observation_preservation"] = 0.50
            for method_id in REPORTED_ARM_IDS:
                metrics = (
                    conditional
                    if method_id == CONDITIONAL_METHOD_ID
                    else deterministic
                )
                resources = (
                    {
                        "latency_seconds": 1.0,
                        "peak_memory_mb": 200.0,
                        "network_calls_per_window": 100,
                        "parameter_count": 1000,
                        "training_walltime_seconds": 20.0,
                    }
                    if method_id == CONDITIONAL_METHOD_ID
                    else {
                        "latency_seconds": 0.1,
                        "peak_memory_mb": 100.0,
                        "network_calls_per_window": 1,
                        "parameter_count": 900,
                        "training_walltime_seconds": 10.0,
                    }
                )
                rows.append(
                    {
                        "partition": "evaluation",
                        "fold_id": fold["fold_id"],
                        "study": fold["study"],
                        "recording_key": recording_key,
                        "method_id": method_id,
                        "status": "success_complete_block2",
                        "fallback_used": False,
                        "query_evaluation_fields_opened_after_all_arm_outputs_frozen": True,
                        "query_evaluation_fields_used_for_fit_selection_or_inference": False,
                        **metrics,
                        **resources,
                    }
                )
    rows.extend(
        {
            "partition": "evaluation",
            "fold_id": "preblocked",
            "study": "study05",
            "recording_key": "study05/study05_p42",
            "method_id": method_id,
            "status": "blocked_no_population",
            "fallback_used": False,
        }
        for method_id in REPORTED_ARM_IDS
    )
    return rows


def _evaluation_endpoints(config: dict) -> list[dict[str, object]]:
    return [
        {
            "fold_id": fold["fold_id"],
            "method_id": method_id,
            "status": "success_fixed_6000_update_endpoint",
            "successful_optimizer_updates": 6000,
            "minibatch_sequence_updates": 6000,
            "minibatch_sequence_verified": True,
        }
        for fold in config["split"]["evaluation_folds"]
        for method_id in (CONDITIONAL_METHOD_ID, DETERMINISTIC_METHOD_ID)
    ]


@pytest.mark.parametrize(
    ("mode", "expected"),
    (
        ("pass", NATURAL_DECISION_PASS),
        ("clear_primary_fail", NATURAL_DECISION_FAIL),
        ("safety_fail", NATURAL_DECISION_INCONCLUSIVE),
    ),
)
def test_evaluation_aggregate_uses_frozen_scoped_decision_rules(
    mode: str,
    expected: str,
) -> None:
    config = _config()
    summary = aggregate_sgeyesub_diffusion_metrics(
        _evaluation_rows(config, mode=mode),
        config=config,
        fold_training_endpoints=_evaluation_endpoints(config),
        partition="evaluation",
    )
    assert summary["status"] == "completed_evaluation_aggregate"
    assert summary["evaluation_fold_count"] == 15
    assert len(summary["completed_fold_ids"]) == 15
    assert summary["availability_denominator"] == 44
    assert summary["compatible_performance_denominator"] == 43
    assert summary["preblocked_count"] == 1
    assert summary["preblocked_recording_key"] == "study05/study05_p42"
    assert summary["paired_primary_success_count"] == 43
    assert summary["natural_decision"]["status"] == expected
    assert summary["claim_boundary"]["clean_target_available"] is False
    assert summary["claim_boundary"]["clean_waveform_recovery_claim"] is False
    assert summary["claim_boundary"]["diffusion_family_wide_claim_allowed"] is False
    assert summary["information_boundary_audit"] == {
        "all_arm_outputs_frozen_before_query_evaluation_fields_opened": True,
        "query_evaluation_fields_used_for_fit_selection_or_inference": False,
    }
    assert summary["matched_comparison_audit"][
        "all_fold_arm_training_endpoints_valid"
    ] is True
    assert len(
        summary["matched_comparison_audit"]["fold_arm_training_endpoints"]
    ) == 30
    assert summary["conditional_minus_unet_resources"]["network_calls_per_window"][
        "mean_conditional_minus_unet"
    ] == pytest.approx(99.0)


def test_descriptive_bootstrap_intervals_do_not_become_unregistered_gates() -> None:
    config = _config()
    rows = _evaluation_rows(config, mode="pass")
    compatible_keys = sorted(
        {
            str(row["recording_key"])
            for row in rows
            if row["recording_key"] != "study05/study05_p42"
        }
    )
    winning = set(compatible_keys[:26])
    for row in rows:
        if row["method_id"] != CONDITIONAL_METHOD_ID:
            continue
        key = str(row["recording_key"])
        if key == "study05/study05_p42":
            continue
        if key in winning:
            row["eog_coherence_reduction"] = 0.30
            row["matching_projector_attenuation_db"] = 1.20
        else:
            row["eog_coherence_reduction"] = 0.0
            row["matching_projector_attenuation_db"] = 0.90
    summary = aggregate_sgeyesub_diffusion_metrics(
        rows,
        config=config,
        fold_training_endpoints=_evaluation_endpoints(config),
        partition="evaluation",
    )
    assert summary["natural_decision"]["primary_benefit_point_pass"] is True
    assert summary["natural_decision"][
        "bootstrap_intervals_used_as_decision_thresholds"
    ] is False
    assert summary["natural_decision"]["status"] == NATURAL_DECISION_PASS


def test_failed_primary_row_is_retained_but_not_used_as_a_performance_pair() -> None:
    config = _config()
    rows = _evaluation_rows(config, mode="pass")
    failed_key = "study02/study02_p02"
    for row in rows:
        if row["recording_key"] == failed_key and row["method_id"] == CONDITIONAL_METHOD_ID:
            row["status"] = "failed_inference"
            row.pop("query_evaluation_fields_opened_after_all_arm_outputs_frozen")
            break
    summary = aggregate_sgeyesub_diffusion_metrics(
        rows,
        config=config,
        fold_training_endpoints=_evaluation_endpoints(config),
        partition="evaluation",
    )
    assert summary["paired_primary_success_count"] == 42
    assert failed_key not in summary["paired_recording_keys"]
    assert summary["method_coverage"][CONDITIONAL_METHOD_ID]["failed_count"] == 1
    assert summary["natural_decision"]["conditional_diffusion_failure_count"] == 1


def test_incomplete_or_metric_incomplete_aggregate_cannot_emit_pass_or_fail() -> None:
    config = _config()
    endpoints = _evaluation_endpoints(config)
    rows = _evaluation_rows(config, mode="pass")
    rows = [
        row
        for row in rows
        if not (
            row["recording_key"] == "study02/study02_p02"
            and row["method_id"] == "raw_observation"
        )
    ]
    incomplete = aggregate_sgeyesub_diffusion_metrics(
        rows,
        config=config,
        fold_training_endpoints=endpoints,
        partition="evaluation",
    )
    assert incomplete["status"] == "incomplete_evaluation_aggregate"
    assert incomplete["natural_decision"]["status"] == NATURAL_DECISION_INCONCLUSIVE

    rows = _evaluation_rows(config, mode="pass")
    conditional = next(
        row
        for row in rows
        if row["recording_key"] == "study02/study02_p02"
        and row["method_id"] == CONDITIONAL_METHOD_ID
    )
    conditional.pop("eog_coherence_reduction")
    missing_metric = aggregate_sgeyesub_diffusion_metrics(
        rows,
        config=config,
        fold_training_endpoints=endpoints,
        partition="evaluation",
    )
    assert missing_metric["status"] == "completed_evaluation_aggregate"
    assert missing_metric["natural_decision"][
        "primary_metrics_complete_for_all_successful_pairs"
    ] is False
    assert missing_metric["natural_decision"]["status"] == NATURAL_DECISION_INCONCLUSIVE

    endpoints = _evaluation_endpoints(config)
    endpoints[0]["minibatch_sequence_verified"] = False
    unmatched_order = aggregate_sgeyesub_diffusion_metrics(
        _evaluation_rows(config, mode="pass"),
        config=config,
        fold_training_endpoints=endpoints,
        partition="evaluation",
    )
    assert unmatched_order["status"] == "incomplete_evaluation_aggregate"
    assert unmatched_order["natural_decision"]["status"] == (
        NATURAL_DECISION_INCONCLUSIVE
    )


def test_aggregate_rejects_rows_outside_frozen_fold_method_matrix() -> None:
    config = _config()
    rows = _evaluation_rows(config, mode="pass")
    extra = dict(rows[0])
    extra.update(
        {
            "recording_key": "study02/not_frozen",
            "method_id": CONDITIONAL_METHOD_ID,
        }
    )
    rows.append(extra)
    with pytest.raises(ValueError, match="outside the frozen participant matrix"):
        aggregate_sgeyesub_diffusion_metrics(
            rows,
            config=config,
            fold_training_endpoints=_evaluation_endpoints(config),
            partition="evaluation",
        )

    rows = _evaluation_rows(config, mode="pass")
    rows[0]["fold_id"] = "wrong_fold"
    with pytest.raises(ValueError, match="recording_key/fold_id"):
        aggregate_sgeyesub_diffusion_metrics(
            rows,
            config=config,
            fold_training_endpoints=_evaluation_endpoints(config),
            partition="evaluation",
        )


def test_preblocked_identity_row_stays_in_coverage_but_not_performance_mean() -> None:
    config = _config()
    rows = _evaluation_rows(config, mode="pass")
    blocked_raw = next(
        row
        for row in rows
        if row["recording_key"] == "study05/study05_p42"
        and row["method_id"] == "raw_observation"
    )
    blocked_raw["eog_coherence_reduction"] = 1.0e9
    summary = aggregate_sgeyesub_diffusion_metrics(
        rows,
        config=config,
        fold_training_endpoints=_evaluation_endpoints(config),
        partition="evaluation",
    )
    assert summary["method_coverage"]["raw_observation"] == {
        "requested_count": 44,
        "success_count": 43,
        "failed_count": 0,
        "blocked_or_ineligible_count": 1,
        "fallback_count": 0,
    }
    performance = summary["method_performance_success_rows_only"]["raw_observation"]
    assert performance["success_count"] == 43
    assert performance["metrics"]["eog_coherence_reduction"]["mean"] == pytest.approx(
        0.10
    )
