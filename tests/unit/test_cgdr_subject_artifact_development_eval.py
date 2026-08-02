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
    FACTORIAL_ARM_IDS,
    TASK_COUNT,
    _infer_arm,
    _score_record,
    evaluation_task,
    factorial_context_plan,
    freeze_factorial_outputs,
    open_annotations_after_freeze,
    _performance_values_eligible,
    run_subject_artifact_evaluation,
    subject_artifact_checkpoint_path,
)


def _config(tmp_path):
    return {
        "training": {"seeds": [20260811, 20260812, 20260813]},
        "outputs": {"checkpoint_root": str(tmp_path / "checkpoints")},
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


def test_checkpoint_handoff_uses_one_fold_seed_checkpoint_per_model(tmp_path) -> None:
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

    assert deterministic == (
        tmp_path
        / "checkpoints/fold_07/seed_20260812/deterministic/best.pt"
    )
    assert diffusion == (
        tmp_path / "checkpoints/fold_07/seed_20260812/diffusion/best.pt"
    )


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
        assert payload["waveforms_persisted"] is True
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
    safe = {"absolute_scale_safety_passed": True}
    unsafe = {"absolute_scale_safety_passed": False}
    assert _performance_values_eligible("success", safe)
    assert not _performance_values_eligible(
        "success_population_fallback_rho_zero", safe
    )
    assert not _performance_values_eligible("failed_inference_runtime_or_contract", safe)
    assert not _performance_values_eligible("success", unsafe)


def test_scoring_contract_aggregates_windows_before_stem_statistics() -> None:
    source = inspect.getsource(_score_record)

    assert '"statistical_unit": "participant_stem"' in source
    assert '"window_level_inference": False' in source
    assert '"windows_aggregated_within_stem"' in source
    assert '"window_index"' not in source
