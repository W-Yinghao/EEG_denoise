"""Execution-adapter tests for the frozen natural SGEYESUB protocol."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

import eeg_cgdr.experiments.sgeyesub_diffusion_runner as runner
import eeg_cgdr.experiments.sgeyesub_diffusion as core
from eeg_cgdr.data.sgeyesub import (
    SgeyesubLoadedRecord,
    SgeyesubQuerySignals,
    SgeyesubReleaseRecord,
    SgeyesubSupportSignals,
)
from eeg_cgdr.experiments.sgeyesub_diffusion import (
    FullBlockP0Fit,
    FrozenSgeyesubFold,
    OuterTrainingPopulationP0Fit,
    SgeyesubFoldNormalizer,
    TrialWindowOrigin,
    WeakSupervisionBundle,
    matching_soft_proximal,
)
from eeg_cgdr.models.conditional_diffusion import OperatorConditionedEEGDiffusion
from eeg_cgdr.models.deterministic_unet import TaskMatchedDeterministicUNet
from eeg_cgdr.operators import P0FitOutcome, P0Transfer


CONFIG_PATH = Path("configs/cgdr/sgeyesub_diffusion_incremental.yaml")


def _config() -> dict[str, object]:
    value = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_runner_validates_frozen_config_and_p0_fields() -> None:
    runner.validate_sgeyesub_diffusion_runner_config(_config())
    config = _config()
    config["outer_fold_fit"] = dict(config["outer_fold_fit"])
    config["outer_fold_fit"]["weak_supervision"] = dict(
        config["outer_fold_fit"]["weak_supervision"]
    )
    config["outer_fold_fit"]["weak_supervision"][
        "p0_bootstrap_replicates"
    ] = 31
    with pytest.raises(ValueError, match="weak-supervision|P0"):
        runner.validate_sgeyesub_diffusion_runner_config(config)


def test_matched_models_expose_the_same_generic_inference_fields() -> None:
    expected = (
        "observed_query_eeg",
        "operator_projector",
        "shared_framewise_attenuation",
        "valid_time_mask",
    )
    assert TaskMatchedDeterministicUNet.visible_input_fields == expected
    assert OperatorConditionedEEGDiffusion.visible_input_fields == expected


def test_checkpoint_identity_requires_scheduled_40hex_git_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DENOISENET_GIT_HEAD", raising=False)
    with pytest.raises(RuntimeError, match="DENOISENET_GIT_HEAD"):
        runner._implementation_identity()
    monkeypatch.setenv("DENOISENET_GIT_HEAD", "not-a-commit")
    with pytest.raises(RuntimeError, match="40-hex"):
        runner._implementation_identity()
    monkeypatch.setenv("DENOISENET_GIT_HEAD", "A" * 40)
    assert runner._implementation_identity() == {
        "implementation_version": runner.IMPLEMENTATION_VERSION,
        "git_head": "a" * 40,
    }


def test_aggregate_rejects_fold_from_a_different_implementation() -> None:
    expected = {
        "implementation_version": runner.IMPLEMENTATION_VERSION,
        "git_head": "a" * 40,
    }
    summary = {
        "status": "completed_fold",
        "fold_id": "study01_fold01",
        "exact_shared_minibatch_sequence_verified": True,
        **expected,
    }
    runner._validate_completed_fold_for_aggregate(
        summary,
        fold_id="study01_fold01",
        expected_identity=expected,
    )
    summary["git_head"] = "b" * 40
    with pytest.raises(ValueError, match="implementation identity mismatch"):
        runner._validate_completed_fold_for_aggregate(
            summary,
            fold_id="study01_fold01",
            expected_identity=expected,
        )


def test_evaluation_rejects_incomplete_development_before_fold_preparation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    config["outputs"] = dict(config["outputs"])
    development_root = tmp_path / "development"
    config["outputs"]["development_root"] = str(development_root)
    development_root.mkdir(parents=True)
    identity = {
        "implementation_version": runner.IMPLEMENTATION_VERSION,
        "git_head": "d" * 40,
    }
    runner._write_json(
        development_root / "result_summary.json",
        {
            "protocol_id": runner.PROTOCOL_ID,
            "status": "incomplete_development_aggregate",
            "partition": "development",
            **identity,
        },
    )
    monkeypatch.setenv("DENOISENET_GIT_HEAD", identity["git_head"])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    preparation_called = False

    def fail_if_prepared(*_args, **_kwargs):
        nonlocal preparation_called
        preparation_called = True
        raise AssertionError("evaluation preparation must remain closed")

    monkeypatch.setattr(runner, "_prepare_fold", fail_if_prepared)
    with pytest.raises(RuntimeError, match="completed development aggregate"):
        runner.run_sgeyesub_diffusion_fold(
            config, "evaluation", 0, tmp_path / "run", torch.device("cuda")
        )
    assert preparation_called is False

    completed = {
        "protocol_id": runner.PROTOCOL_ID,
        "status": "completed_development_aggregate",
        "partition": "development",
        **identity,
    }
    runner._write_json(development_root / "result_summary.json", completed)
    assert runner._require_completed_development_aggregate(config, identity) == completed


def test_metric_contract_records_failure_task_boundary_and_config_seed() -> None:
    success = runner._metric_contract_fields("success", inference_seed=17)
    failure = runner._metric_contract_fields(
        "failed_inference_nonfinite", inference_seed=18
    )
    assert success["failure_status"] == ""
    assert failure["failure_status"] == "failed_inference_nonfinite"
    assert "ERP_proxy_only" in str(
        success["downstream_task_preservation_when_label_semantics_allow"]
    )
    assert failure["inference_seed"] == 18

    config = _config()
    first = runner._record_seed(config, "evaluation", "fold", "study/p01")
    changed = _config()
    changed["matched_comparison"] = dict(changed["matched_comparison"])
    changed["matched_comparison"]["conditional_diffusion"] = dict(
        changed["matched_comparison"]["conditional_diffusion"]
    )
    changed["matched_comparison"]["conditional_diffusion"][
        "initial_state_seed"
    ] += 1
    assert runner._record_seed(
        changed, "evaluation", "fold", "study/p01"
    ) == first + 1
    assert (
        runner._inference_failure_status(FloatingPointError("nan"))
        == "failed_inference_nonfinite"
    )
    assert (
        runner._inference_failure_status(RuntimeError("CUDA out of memory"))
        == "failed_inference_cuda_oom"
    )


def test_real_structure_contract_canonicalizes_layout_ids() -> None:
    layouts, records, folds = runner._protocol_contract(_config())
    assert len(layouts) == 6
    assert len(records) == 59
    assert len(folds) == 25
    first = next(value for value in folds if value.fold_id == "study01_fold01")
    assert first.layout_id == "layout_01"
    assert records[first.heldout_recording_keys[0]].layout_id == "layout_01"
    assert not set(first.training_recording_keys) & set(first.heldout_recording_keys)


def test_exact_6000_minibatch_sequence_is_reusable_by_both_models() -> None:
    left = runner._minibatch_schedule(17, 8)
    right = runner._minibatch_schedule(17, 8)
    assert len(left) == 6000
    assert all(np.array_equal(a, b) for a, b in zip(left, right))
    assert all(1 <= value.size <= 8 for value in left)
    np.testing.assert_array_equal(
        np.concatenate(left[:3]), np.random.default_rng(20260802).permutation(17)
    )


def test_soft_proximal_runner_path_uses_numpy_square_semantics() -> None:
    observed = np.asarray([[2.0, 4.0, 6.0], [3.0, 5.0, 7.0]])
    projector = np.asarray([[1.0, 0.0], [0.0, 0.0]])
    attenuation = np.asarray([0.0, 0.5, 1.0])
    actual = matching_soft_proximal(observed, projector, attenuation)
    expected = observed - (1.0 - np.square(attenuation))[None, :] * (
        projector @ observed
    )
    np.testing.assert_allclose(actual, expected)


def _record(study_stem: str) -> SgeyesubReleaseRecord:
    study = study_stem.split("_p", 1)[0]
    return SgeyesubReleaseRecord(
        study=study,
        participant_stem=study_stem,
        set_relative_path=f"{study}/{study_stem}_prep.set",
        fdt_relative_path=f"{study}/{study_stem}_prep.fdt",
        sampling_rate_hz=200.0,
        channel_count=5,
        samples_per_trial=16,
        trial_count=2,
        layout_id="layout_01",
        p0_layout_id="p0_layout_01",
        trial_block_counts={1: 1, 2: 1},
        trial_label_counts={1: 1, 2: 1},
        trial_id_count=2,
    )


def _loaded(study_stem: str, *, query: bool) -> SgeyesubLoadedRecord:
    support = SgeyesubSupportSignals(
        eeg=np.arange(32, dtype=np.float64).reshape(2, 16) + 1.0,
        native_eeg=np.arange(32, dtype=np.float64).reshape(2, 16) + 1.0,
        external_eog=np.vstack(
            [np.linspace(-1.0, 1.0, 16), np.linspace(1.0, -1.0, 16)]
        ),
        artifactclasses=np.asarray([6] * 8 + [1] * 8),
        trial_labels=np.asarray([1]),
        trial_ids=np.asarray([1]),
    )
    return SgeyesubLoadedRecord(
        study="study01",
        participant_stem=study_stem,
        release_layout_id="layout_01",
        p0_layout_id="p0_layout_01",
        p0_channel_labels=("C1", "C2"),
        native_channel_labels=("C1", "C2"),
        sampling_rate_hz=200.0,
        support=support,
        query=(
            SgeyesubQuerySignals(eeg=support.eeg.copy(), native_eeg=support.eeg.copy())
            if query
            else None
        ),
        query_annotations=None,
    )


def _transfer(samples: int) -> P0Transfer:
    projector = np.eye(2, dtype=np.float64)
    return P0Transfer(
        transfer_matrix=np.eye(2, dtype=np.float64),
        eeg_subspace_basis=np.eye(2, dtype=np.float64),
        projector=projector,
        predicted_contamination=np.zeros((2, samples), dtype=np.float64),
        eog_mean=np.zeros((2, 1), dtype=np.float64),
        eeg_mean=np.zeros((2, 1), dtype=np.float64),
        rank=2,
        diagnostics={"samples": samples, "ridge_lambda": 0.01},
    )


def test_prepare_fold_uses_outer_training_only_and_keeps_annotations_sealed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training_keys = ("study01/study01_p02", "study01/study01_p03")
    heldout_keys = ("study01/study01_p01",)
    fold = FrozenSgeyesubFold(
        partition="development",
        fold_id="study01_fold01",
        study="study01",
        layout_id="layout_01",
        reference_cell_id="release_preprocessed_as_delivered",
        sampling_rate_hz=200.0,
        eeg_channels=2,
        training_recording_keys=training_keys,
        heldout_recording_keys=heldout_keys,
    )
    records = {
        key: _record(key.split("/", 1)[1]) for key in (*training_keys, *heldout_keys)
    }
    loaded = {
        key: _loaded(key.split("/", 1)[1], query=key in heldout_keys)
        for key in records
    }
    load_calls: list[tuple[str, bool, bool | None]] = []

    monkeypatch.setattr(
        runner,
        "_protocol_contract",
        lambda _config: ({"layout_01": object()}, records, (fold,)),
    )

    def fake_load(_root, record, _layout, *, include_query, include_query_annotations=None):
        load_calls.append(
            (record.recording_key, include_query, include_query_annotations)
        )
        return loaded[record.recording_key]

    monkeypatch.setattr(runner, "load_sgeyesub_signal_record", fake_load)
    normalizer = SgeyesubFoldNormalizer(
        mean=np.zeros(2),
        standard_deviation=np.ones(2),
        training_recording_keys=training_keys,
        sample_count=32,
    )
    monkeypatch.setattr(
        runner, "fit_outer_training_normalizer", lambda _signals, _keys: normalizer
    )
    transfer = _transfer(16)
    fit = FullBlockP0Fit(
        outcome=P0FitOutcome("eligible", transfer, ()),
        eog_mean=np.zeros((2, 1)),
        eog_standard_deviation=np.ones((2, 1)),
        recording_key="record",
        sample_count=16,
    )
    monkeypatch.setattr(runner, "fit_full_block1_p0", lambda *args, **kwargs: fit)
    population = OuterTrainingPopulationP0Fit(
        outcome=P0FitOutcome("eligible", transfer, ()),
        training_recording_keys=training_keys,
        source_eog_statistics={},
        sample_count=32,
    )
    monkeypatch.setattr(
        runner, "fit_outer_training_population_p0", lambda *args, **kwargs: population
    )
    monkeypatch.setattr(
        runner, "fit_outer_training_projected_energy_scale", lambda _items: 1.0
    )
    origin_target = TrialWindowOrigin("record", 0, 0, 8)
    origin_source = TrialWindowOrigin("record", 0, 8, 16)

    def fake_bundle(*args, recording_key, **kwargs):
        return WeakSupervisionBundle(
            observed=np.ones((1, 2, 8), dtype=np.float32),
            weak_target=np.zeros((1, 2, 8), dtype=np.float32),
            projector=np.eye(2, dtype=np.float32)[None, :, :],
            attenuation=np.ones((1, 8), dtype=np.float32),
            valid_time_mask=np.ones((1, 8), dtype=np.float32),
            recording_keys=(recording_key,),
            target_origins=(
                TrialWindowOrigin(recording_key, origin_target.trial_ordinal, 0, 8),
            ),
            artifact_origins=(
                TrialWindowOrigin(recording_key, origin_source.trial_ordinal, 8, 16),
            ),
        )

    monkeypatch.setattr(runner, "build_within_stem_weak_pairs", fake_bundle)
    prepared = runner._prepare_fold(_config(), "development", 0)
    assert tuple(prepared.training) == training_keys
    assert tuple(prepared.heldout) == heldout_keys
    assert not set(prepared.training) & set(prepared.heldout)
    assert all(value.query_annotations is None for value in prepared.heldout.values())
    assert load_calls == [
        (training_keys[0], False, None),
        (training_keys[1], False, None),
        (heldout_keys[0], True, False),
    ]
    monkeypatch.setenv("DENOISENET_GIT_HEAD", "c" * 40)
    contract = runner._checkpoint_contract(
        _config(), prepared, core.DETERMINISTIC_METHOD_ID
    )
    assert contract["implementation_version"] == runner.IMPLEMENTATION_VERSION
    assert contract["git_head"] == "c" * 40


def test_runner_exposes_all_required_public_entry_points() -> None:
    for name in (
        "run_sgeyesub_diffusion_cpu_validation",
        "run_sgeyesub_diffusion_integration",
        "run_sgeyesub_diffusion_fold",
        "aggregate_sgeyesub_diffusion_partition",
    ):
        assert callable(getattr(runner, name))


def test_partition_aggregate_stamps_current_fold_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    git_head = "e" * 40
    monkeypatch.setenv("DENOISENET_GIT_HEAD", git_head)
    fold = SimpleNamespace(partition="development", fold_id="study01_fold01")
    development_root = tmp_path / "development"
    fold_root = development_root / fold.fold_id
    fold_root.mkdir(parents=True)
    fold_summary = {
        "status": "completed_fold",
        "fold_id": fold.fold_id,
        "exact_shared_minibatch_sequence_verified": True,
        "training_endpoints": [{"method_id": "learned"}],
        "implementation_version": runner.IMPLEMENTATION_VERSION,
        "git_head": git_head,
    }
    runner._write_json(fold_root / "result_summary.json", fold_summary)
    runner._write_csv(
        fold_root / "metrics.csv",
        [{"method_id": "raw_observation", "status": "success"}],
    )
    monkeypatch.setattr(
        runner,
        "_protocol_contract",
        lambda _config: ({}, {}, (fold,)),
    )

    def fake_write(_rows, *, config, fold_training_endpoints, partition):
        assert partition == "development"
        assert fold_training_endpoints == [{"method_id": "learned"}]
        summary_path = Path(config["outputs"]["development_root"]) / "result_summary.json"
        runner._write_json(
            summary_path,
            {
                "protocol_id": runner.PROTOCOL_ID,
                "status": "completed_development_aggregate",
                "partition": "development",
            },
        )
        return {"result_summary": summary_path}

    monkeypatch.setattr(runner, "write_sgeyesub_diffusion_aggregate", fake_write)
    config = {"outputs": {"development_root": str(development_root)}}
    summary = runner.aggregate_sgeyesub_diffusion_partition(
        config, "development", tmp_path / "run"
    )
    assert summary["implementation_version"] == runner.IMPLEMENTATION_VERSION
    assert summary["git_head"] == git_head
    canonical = json.loads(
        (development_root / "result_summary.json").read_text(encoding="utf-8")
    )
    assert canonical["git_head"] == git_head

    fold_summary["git_head"] = "f" * 40
    runner._write_json(fold_root / "result_summary.json", fold_summary)
    with pytest.raises(ValueError, match="implementation identity mismatch"):
        runner.aggregate_sgeyesub_diffusion_partition(
            config, "development", tmp_path / "second_run"
        )


def test_fold_installs_signal_before_preparation_and_writes_no_empty_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DENOISENET_GIT_HEAD", "b" * 40)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda _seed: None)
    installed: dict[str, object] = {}
    previous_handler = object()

    def fake_signal(_signal_number, handler):
        if callable(handler):
            installed["handler"] = handler
            return previous_handler
        assert handler is previous_handler
        installed["restored"] = True
        return handler

    monkeypatch.setattr(runner.signal, "signal", fake_signal)

    def fake_prepare(_config, partition, task_index):
        assert partition == "development" and task_index == 0
        handler = installed.get("handler")
        assert callable(handler)
        handler(0, None)
        return SimpleNamespace(fold=SimpleNamespace(fold_id="study01_fold01"))

    monkeypatch.setattr(runner, "_prepare_fold", fake_prepare)
    summary = runner.run_sgeyesub_diffusion_fold(
        _config(), "development", 0, tmp_path, torch.device("cuda")
    )
    assert summary["status"] == "checkpointed_for_resume"
    assert summary["checkpoint_written"] is False
    assert installed["restored"] is True
    assert not (tmp_path / "training_endpoints.csv").exists()
    saved = json.loads((tmp_path / "result_summary.json").read_text(encoding="utf-8"))
    assert saved["reason"] == "SIGUSR1_received_during_fold_preparation_before_training"


def test_csv_roundtrip_preserves_types_for_core_aggregate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config()
    rows: list[dict[str, object]] = []
    endpoints: list[dict[str, object]] = []
    failed_recording_key = "study02/study02_p02"
    for fold in config["split"]["evaluation_folds"]:
        fold_id = str(fold["fold_id"])
        for method_id in (core.DETERMINISTIC_METHOD_ID, core.CONDITIONAL_METHOD_ID):
            endpoints.append(
                {
                    "fold_id": fold_id,
                    "method_id": method_id,
                    "status": "success_fixed_6000_update_endpoint",
                    "successful_optimizer_updates": 6000,
                    "minibatch_sequence_updates": 6000,
                    "minibatch_sequence_verified": True,
                }
            )
        for stem in fold["heldout_stems"]:
            key = f"{fold['study']}/{stem}"
            for method_id in core.REPORTED_ARM_IDS:
                conditional = method_id == core.CONDITIONAL_METHOD_ID
                failed_inference = conditional and key == failed_recording_key
                rows.append(
                    {
                        "partition": "evaluation",
                        "fold_id": fold_id,
                        "study": fold["study"],
                        "participant_stem": stem,
                        "recording_key": key,
                        "method_id": method_id,
                        "status": (
                            "failed_inference_nonfinite"
                            if failed_inference
                            else "success"
                        ),
                        "failure_status": (
                            "failed_inference_nonfinite" if failed_inference else ""
                        ),
                        "inference_placeholder_used_for_freeze_only": failed_inference,
                        "performance_values_eligible": not failed_inference,
                        "fallback_used": False,
                        "query_evaluation_fields_opened_after_all_arm_outputs_frozen": True,
                        "query_evaluation_fields_used_for_fit_selection_or_inference": False,
                        "heldout_eog_prediction_remaining_ratio": 0.8,
                        "eog_coherence_reduction": 0.2 + (0.1 if conditional else 0.0),
                        "matching_projector_attenuation_db": 1.0 + (0.2 if conditional else 0.0),
                        "nonartifact_observation_preservation": 0.98,
                        "reference_free_psd_distortion": 0.03 - (0.01 if conditional else 0.0),
                        "reference_free_covariance_distortion": 0.04 - (0.01 if conditional else 0.0),
                        "condition_erp_observation_relative_preservation": 0.97,
                        "observation_change_ratio": 0.1,
                        "latency_seconds": 1.0,
                        "peak_memory_mb": 100.0,
                        "network_calls_per_window": 100 if conditional else 1,
                        "parameter_count": 1000,
                        "training_walltime_seconds": 20.0,
                    }
                )
    for method_id in core.REPORTED_ARM_IDS:
        rows.append(
            {
                "partition": "evaluation",
                "fold_id": "preblocked_singleton_layout06",
                "study": "study05",
                "participant_stem": "study05_p42",
                "recording_key": "study05/study05_p42",
                "method_id": method_id,
                "status": "blocked_no_population",
                "failure_status": "blocked_no_population",
                "fallback_used": False,
                "query_evaluation_fields_opened_after_all_arm_outputs_frozen": False,
                "query_evaluation_fields_used_for_fit_selection_or_inference": False,
            }
        )
    path = tmp_path / "metrics.csv"
    runner._write_csv(path, rows)
    reloaded = runner._read_csv(path)
    assert reloaded[0]["fallback_used"] is False
    assert (
        reloaded[0][
            "query_evaluation_fields_opened_after_all_arm_outputs_frozen"
        ]
        is True
    )
    assert reloaded[-1]["query_evaluation_fields_opened_after_all_arm_outputs_frozen"] is False

    def cheap_ci(values, *, replicates, seed, confidence):
        mean = float(np.mean(values)) if values else float("nan")
        return mean, mean

    monkeypatch.setattr(core, "_bootstrap_mean_ci", cheap_ci)
    summary = core.aggregate_sgeyesub_diffusion_metrics(
        reloaded,
        config=config,
        fold_training_endpoints=endpoints,
        partition="evaluation",
    )
    assert summary["status"] == "completed_evaluation_aggregate"
    assert summary["availability_denominator"] == 44
    assert summary["preblocked_count"] == 1
    assert summary["paired_primary_success_count"] == 42
    assert summary["method_coverage"][core.CONDITIONAL_METHOD_ID]["failed_count"] == 1
    assert summary["method_performance_success_rows_only"][
        core.CONDITIONAL_METHOD_ID
    ]["success_count"] == 42
    assert failed_recording_key not in summary["paired_recording_keys"]
    assert summary["information_boundary_audit"] == {
        "all_arm_outputs_frozen_before_query_evaluation_fields_opened": True,
        "query_evaluation_fields_used_for_fit_selection_or_inference": False,
    }
