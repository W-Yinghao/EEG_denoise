"""Static/numeric contracts for the development-only natural-SGE factorial."""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import numpy as np
import pytest

from eeg_cgdr.experiments.subject_artifact_data import (
    HeldoutSubjectArtifactRecord,
    RuntimeArtifactContext,
    SealedQueryWindows,
)
from eeg_cgdr.experiments.subject_artifact_development_eval import (
    ArmInference,
    CHECKPOINT_ENDPOINTS,
    EVALUATION_ARM_IDS,
    EvaluationTask,
    FACTORIAL_ARM_IDS,
    TASK_COUNT,
    _context_provenance,
    _full_v0_scale_validity,
    _infer_arm,
    _performance_values_eligible,
    _posterior_reconstruction_metadata,
    _score_record,
    _uncertainty_window_rows,
    evaluation_task,
    factorial_context_plan,
    freeze_factorial_outputs,
    open_annotations_after_freeze,
    run_subject_artifact_evaluation,
    subject_artifact_checkpoint_path,
)


def _config(tmp_path):
    return {
        "training": {
            "seeds": [20260811, 20260812, 20260813],
            "equal_compute_updates": 8000,
        },
        "outputs": {"checkpoint_root": str(tmp_path / "checkpoints")},
        "calibration": {"calibration_duration_seconds": 0.5},
        "validity": {
            "V0": {
                "maximum_per_window_output_input_RMS_ratio": 10.0,
                "full_median_output_input_RMS_ratio": [0.5, 2.0],
                "low_artifact_median_output_input_RMS_ratio": [0.8, 1.2],
                "low_artifact_maximum_median_relative_observation_change": 0.20,
                "pure_operator_maximum_complement_consistency_relative_error": 1.0e-5,
                "pure_operator_maximum_union_span_consistency_relative_error": 1.0e-5,
            }
        },
    }


def _runtime(
    role: str,
    context_id: str,
    *,
    key: str,
    rho: float,
    swap: bool = False,
) -> RuntimeArtifactContext:
    normalized = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.25]], dtype=np.float64
    )
    if swap:
        normalized = normalized[:, ::-1].copy()
    scale = np.asarray([2.0, 3.0], dtype=np.float64)
    full = normalized * scale[None, :]
    basis = np.linalg.svd(full, full_matrices=False)[0][:, :2]
    return RuntimeArtifactContext(
        role=role,
        context_id=context_id,
        raw_transfer=full,
        full_transfer=full,
        normalized_transfer=normalized,
        transfer_scale=scale,
        singular_values=np.asarray([4.0, 2.0]),
        rank=2,
        projector=basis @ basis.T,
        rho=rho,
        calibration_duration_seconds=30.0,
        fit_recording_keys=(key,),
    )


def _record() -> tuple[RuntimeArtifactContext, HeldoutSubjectArtifactRecord]:
    rho = 0.6
    population = _runtime(
        "population", "fold:population", key="study02/train_p01", rho=0.0
    )
    # The preselected wrong source must be a member of the population library.
    population = RuntimeArtifactContext(
        **{
            **population.__dict__,
            "fit_recording_keys": ("study02/train_p01", "study02/train_p02"),
        }
    )
    matching = _runtime(
        "matching", "heldout:matching", key="study02/heldout", rho=rho
    )
    wrong = _runtime(
        "wrong_same_cell",
        "heldout:wrong",
        key="study02/train_p01",
        rho=rho,
        swap=True,
    )
    shuffled = _runtime(
        "shuffled_same_cell_severity_stratum",
        "heldout:shuffled",
        key="study02/heldout",
        rho=rho,
        swap=True,
    )
    query = SealedQueryWindows(
        recording_key="study02/heldout",
        observed=np.zeros((2, 3, 8), dtype=np.float32),
        valid_time_mask=np.ones((2, 8), dtype=bool),
        origins=(
            SimpleNamespace(),  # origin contents are not used by this unit contract
            SimpleNamespace(),
        ),
    )
    return population, HeldoutSubjectArtifactRecord(
        recording_key="study02/heldout",
        matching=matching,
        wrong_same_cell=wrong,
        shuffled_same_cell=shuffled,
        wrong_source_recording_key="study02/train_p01",
        query=query,
    )


def test_task_map_is_exactly_25_folds_times_three_frozen_seeds(tmp_path) -> None:
    config = _config(tmp_path)
    mapped = [evaluation_task(config, index) for index in range(TASK_COUNT)]

    assert TASK_COUNT == 75
    assert mapped[0].unified_fold_index == 0
    assert mapped[0].training_seed == 20260811
    assert mapped[2].training_seed == 20260813
    assert mapped[3].unified_fold_index == 1
    assert mapped[-1].unified_fold_index == 24
    assert mapped[-1].training_seed == 20260813
    assert len({(item.unified_fold_index, item.training_seed) for item in mapped}) == 75
    with pytest.raises(ValueError):
        evaluation_task(config, 75)


def test_checkpoint_handoff_exposes_best_and_equal_endpoints_per_model(tmp_path) -> None:
    config = _config(tmp_path)
    deterministic = subject_artifact_checkpoint_path(
        config,
        unified_fold_index=7,
        training_seed=20260812,
        model_kind="deterministic",
    )
    diffusion = subject_artifact_checkpoint_path(
        config,
        unified_fold_index=7,
        training_seed=20260812,
        model_kind="diffusion",
    )
    equal = subject_artifact_checkpoint_path(
        config,
        unified_fold_index=7,
        training_seed=20260812,
        model_kind="diffusion",
        checkpoint_endpoint="equal",
    )

    assert deterministic == (
        tmp_path
        / "checkpoints/fold_07/seed_20260812/deterministic/best.pt"
    )
    assert diffusion == (
        tmp_path / "checkpoints/fold_07/seed_20260812/diffusion/best.pt"
    )
    assert equal == (
        tmp_path / "checkpoints/fold_07/seed_20260812/diffusion/equal.pt"
    )
    assert CHECKPOINT_ENDPOINTS == ("best", "equal")


def test_factorial_keeps_original_rho_and_only_swaps_same_cell_C() -> None:
    population, heldout = _record()
    plan = factorial_context_plan(population, heldout)

    assert tuple(value.context_id for value in plan) == (
        "population",
        "matching",
        "wrong",
        "shuffled",
    )
    assert all(value.rho == heldout.matching.rho == 0.6 for value in plan)
    assert all(value.calibration_duration_seconds == 30.0 for value in plan)
    assert plan[0].subject is population
    assert plan[0].is_formal_population_arm
    assert not any(value.is_formal_population_arm for value in plan[1:])
    assert plan[1].subject is heldout.matching
    assert plan[2].subject is heldout.wrong_same_cell
    assert plan[3].subject is heldout.shuffled_same_cell
    assert {value.subject.full_transfer.shape for value in plan} == {(3, 2)}


def test_all_eight_outputs_freeze_before_annotation_opener_runs(tmp_path) -> None:
    events: list[str] = []
    outputs = {
        arm_id: np.full((3, 16), index, dtype=np.float64)
        for index, arm_id in enumerate(FACTORIAL_ARM_IDS)
    }
    manifest = tmp_path / "freeze.json"
    frozen = freeze_factorial_outputs(
        outputs,
        recording_key="study02/test",
        manifest_path=manifest,
    )
    events.append("frozen")

    def opener():
        assert manifest.is_file()
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        assert payload["status"] == "all_eight_factorial_outputs_frozen_before_scoring"
        assert len(payload["arm_ids"]) == 8
        assert payload["point_waveforms_persisted"] is True
        assert payload["point_output_dtype"] == "float32"
        assert frozen.output_archive_path.is_file()
        assert all(not value.flags.writeable for value in frozen.outputs.values())
        events.append("annotations_opened")
        return SimpleNamespace(
            query=SimpleNamespace(eeg=np.zeros((3, 16))),
            query_annotations=SimpleNamespace(external_eog=np.zeros((2, 16))),
        )

    annotated = open_annotations_after_freeze(frozen, opener)
    assert annotated.query_annotations is not None
    assert events == ["frozen", "annotations_opened"]


def test_all_endpoint_outputs_and_explicit_k8_samples_freeze_together(tmp_path) -> None:
    outputs = {
        arm_id: np.full((3, 16), index, dtype=np.float64)
        for index, arm_id in enumerate(EVALUATION_ARM_IDS)
    }
    samples = {
        f"{arm_id}__population_standardized_latent_samples": np.full(
            (8, 2, 2, 8), index, dtype=np.float32
        )
        for index, arm_id in enumerate(EVALUATION_ARM_IDS)
        if "__diffusion__" in arm_id
    }
    metadata = {
        arm_id: {"rho": 0.5, "sample_seeds_by_batch": [list(range(8))]}
        for arm_id in EVALUATION_ARM_IDS
        if "__diffusion__" in arm_id
    }
    manifest = tmp_path / "all_endpoint_freeze.json"

    frozen = freeze_factorial_outputs(
        outputs,
        recording_key="study02/test",
        manifest_path=manifest,
        expected_arm_ids=EVALUATION_ARM_IDS,
        posterior_samples=samples,
        required_posterior_sample_keys=tuple(samples),
        posterior_metadata=metadata,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert payload["status"] == "all_endpoint_factorial_outputs_frozen_before_scoring"
    assert payload["posterior_sample_count_K"] == 8
    assert payload["point_output_dtype"] == "float32"
    assert payload["posterior_storage"].startswith("standardized_latent_only")
    assert payload["posterior_point_rule"] == "strict_arithmetic_mean_no_best_of_K"
    assert payload["estimated_uncompressed_posterior_latent_bytes"] == sum(
        value.nbytes for value in samples.values()
    )
    assert len(payload["arm_ids"]) == 16
    assert all(value.dtype == np.float32 for value in frozen.outputs.values())
    assert set(frozen.posterior_samples or {}) == set(samples)
    assert all(
        not value.flags.writeable for value in (frozen.posterior_samples or {}).values()
    )
    with pytest.raises(ValueError, match="only low-dimensional"):
        freeze_factorial_outputs(
            {"arm": np.zeros((3, 16), dtype=np.float32)},
            recording_key="study02/test",
            manifest_path=tmp_path / "forbidden_EEG_samples.json",
            expected_arm_ids=("arm",),
            posterior_samples={
                "arm__mixed_correction_samples": np.zeros(
                    (8, 2, 3, 8), dtype=np.float32
                )
            },
        )


def test_evaluation_inference_surface_has_no_query_eog_label_or_outcome_args() -> None:
    public = inspect.signature(run_subject_artifact_evaluation)
    assert tuple(public.parameters) == (
        "config",
        "run_dir",
        "task_index",
        "implementation",
    )
    internal = inspect.signature(_infer_arm)
    forbidden = {
        "query_eog",
        "eye_tracking",
        "artifactclasses",
        "trial_labels",
        "outcome",
        "clean_target",
    }
    assert not (set(internal.parameters) & forbidden)


def test_fallback_failure_and_scale_unsafe_rows_are_not_performance() -> None:
    safe = {"full_V0_scale_validity_passed": True}
    unsafe = {"full_V0_scale_validity_passed": False}
    assert _performance_values_eligible("success", safe)
    assert not _performance_values_eligible(
        "success_population_fallback_rho_zero", safe
    )
    assert not _performance_values_eligible("failed_inference_runtime_or_contract", safe)
    assert not _performance_values_eligible("success", unsafe)


def test_full_v0_requires_all_frozen_scale_and_span_safety_rules(tmp_path) -> None:
    config = _config(tmp_path)
    observed = np.ones((3, 2, 4), dtype=np.float64)
    valid = np.ones((3, 4), dtype=bool)
    low_artifact = np.full((3, 4), 6, dtype=np.int64)

    safe = _full_v0_scale_validity(
        config,
        observed.copy(),
        observed,
        valid,
        low_artifact,
        span_consistency_relative_error=0.0,
        retained_samples_finite=True,
    )
    unsafe = observed.copy()
    unsafe[0] *= 11.0
    too_large = _full_v0_scale_validity(
        config,
        unsafe,
        observed,
        valid,
        low_artifact,
        span_consistency_relative_error=0.0,
        retained_samples_finite=True,
    )
    wrong_span = _full_v0_scale_validity(
        config,
        observed.copy(),
        observed,
        valid,
        low_artifact,
        span_consistency_relative_error=2.0e-5,
        retained_samples_finite=True,
    )

    assert safe["full_V0_scale_validity_passed"] is True
    assert safe["retained_posterior_samples_finite"] is True
    assert too_large["full_V0_scale_validity_passed"] is False
    assert (
        too_large["full_V0_checks"]["per_window_RMS_ratio_at_most_10"] is False
    )
    assert wrong_span["full_V0_scale_validity_passed"] is False


def test_wrong_and_shuffled_rows_record_actual_context_draw_provenance() -> None:
    population, heldout = _record()
    plan = {item.context_id: item for item in factorial_context_plan(population, heldout)}

    wrong = _context_provenance(plan["wrong"], heldout)
    shuffled = _context_provenance(plan["shuffled"], heldout)

    assert wrong["wrong_context_source_recording_key"] == "study02/train_p01"
    assert wrong["wrong_context_draw_id"] == "study02/train_p01"
    assert "same_cell" in wrong["wrong_context_draw_rule"]
    assert shuffled["shuffled_context_source_recording_key"] == heldout.recording_key
    assert shuffled["shuffled_context_draw_id"].endswith(
        "support_block1:within_artifactclass_half_roll"
    )
    assert "artifactclass" in shuffled["shuffled_context_draw_rule"]


def test_uncertainty_rows_preserve_k_axis_and_never_select_best_sample(tmp_path) -> None:
    population, heldout = _record()
    population_latent = np.stack(
        [np.full((2, 2, 8), float(index), dtype=np.float32) for index in range(8)]
    )
    subject_latent = population_latent + 0.5
    population_correction = np.einsum(
        "ce,knet->knct", population.normalized_transfer, population_latent
    )
    subject_correction = np.einsum(
        "ce,knet->knct", heldout.matching.normalized_transfer, subject_latent
    )
    correction = 0.4 * population_correction + 0.6 * subject_correction
    restored = heldout.query.observed[None, :, :, :] - correction
    point = restored.mean(axis=0).astype(np.float32)
    sample_seeds = (tuple(range(8)),)
    inference = ArmInference(
        windowed_output=point,
        status="success",
        latency_seconds=1.0,
        peak_memory_mb=2.0,
        posterior_standardized_latent_sd_rms=0.5,
        network_calls=800,
        complement_or_union_relative_error=0.0,
        branch="mixed",
        population_standardized_latent_samples=population_latent,
        subject_standardized_latent_samples=subject_latent,
        sample_seeds_by_batch=sample_seeds,
    )
    prepared = SimpleNamespace(
        population_context=population,
        latent_normalizer=SimpleNamespace(
            mean=np.zeros(2, dtype=np.float64),
            standard_deviation=np.ones(2, dtype=np.float64),
        ),
    )
    matching_context = {
        value.context_id: value
        for value in factorial_context_plan(population, heldout)
    }["matching"]
    evaluation_arm_id = "best__diffusion__matching"
    sample_arrays = {
        f"{evaluation_arm_id}__population_standardized_latent_samples": (
            population_latent
        ),
        f"{evaluation_arm_id}__subject_standardized_latent_samples": subject_latent,
    }
    frozen = freeze_factorial_outputs(
        {
            evaluation_arm_id: np.ascontiguousarray(
                point.transpose(1, 0, 2).reshape(3, -1)
            )
        },
        recording_key=heldout.recording_key,
        manifest_path=tmp_path / "uncertainty_freeze.json",
        expected_arm_ids=(evaluation_arm_id,),
        posterior_samples=sample_arrays,
        required_posterior_sample_keys=tuple(sample_arrays),
        posterior_metadata={
            evaluation_arm_id: _posterior_reconstruction_metadata(
                prepared, matching_context, inference
            )
        },
    )
    annotated_value = SimpleNamespace(
        query=SimpleNamespace(eeg=np.zeros((3, 16))),
        sampling_rate_hz=8.0,
        support=SimpleNamespace(
            external_eog=np.asarray(
                [[-1.0, 0.0, 1.0, 2.0], [2.0, 1.0, 0.0, -1.0]],
                dtype=np.float64,
            )
        ),
        query_annotations=SimpleNamespace(
            external_eog=np.zeros((2, 16), dtype=np.float64),
            artifactclasses=np.full((2, 8), 6, dtype=np.int64),
        ),
    )
    annotated = open_annotations_after_freeze(frozen, lambda: annotated_value)
    rows = _uncertainty_window_rows(
        _config(tmp_path),
        prepared,
        heldout,
        {evaluation_arm_id: inference},
        frozen,
        annotated,
        task=EvaluationTask(0, 0, 20260811, 0),
    )

    assert len(rows) == 2
    assert all(row["posterior_sample_count_K"] == 8 for row in rows)
    assert all(row["best_of_K_used"] is False for row in rows)
    assert all(row["risk_coverage_input_only_not_success_claim"] is True for row in rows)
    assert all(
        row["query_EOG_or_labels_used_for_inference_or_sample_selection"] is False
        for row in rows
    )
    assert all(row["posterior_correction_SD_RMS"] > 0.0 for row in rows)


def test_scoring_contract_aggregates_windows_before_stem_statistics() -> None:
    source = inspect.getsource(_score_record)

    assert '"statistical_unit": "participant_stem"' in source
    assert '"window_level_inference": False' in source
    assert '"windows_aggregated_within_stem"' in source
    assert '"window_index"' not in source
