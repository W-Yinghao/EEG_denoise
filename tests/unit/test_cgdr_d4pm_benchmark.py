"""Semantic tests for the frozen external D4PM EOG-scoped adapter.

These tests use deterministic algebraic arrays only.  A scheduled GPU smoke
stage must separately cover the real EEGdenoiseNet files and external model.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from torch import nn

from eeg_cgdr.experiments.d4pm_benchmark import (
    ARM_BRANCH_COUNT,
    BENCHMARK_ID,
    EXPECTED_FULL_EVALUATION_MIXTURES,
    EXPECTED_FULL_TRAIN_PAIRS,
    EXPECTED_FULL_UPDATES_PER_BRANCH,
    EXPECTED_FULL_VALIDATION_PAIRS,
    EXPECTED_SOURCE_FINDINGS,
    JOINT_ETA,
    JOINT_GAMMA,
    JOINT_LAMBDA_DC,
    OFFICIAL_BATCH_SIZE,
    OFFICIAL_COMBINATIONS,
    OFFICIAL_DIFFUSION_STEPS,
    OFFICIAL_EPOCHS,
    OFFICIAL_FEATURES,
    OFFICIAL_TEST_SNR_DB,
    SOURCE_COMMIT,
    SPECTRAL_METRIC_STATUS,
    TASK_MATRIX,
    MatchedConditionOnly,
    _training_config_view,
    aggregate_d4pm_full_cells,
    audit_d4pm_source_text,
    d4pm_rrmse_spectral_fft_magnitude,
    d4pm_rrmse_temporal,
    prepare_eog_scoped,
    source_split_manifest_rows,
    validate_d4pm_checkout,
    validate_d4pm_config,
)


CONFIG_PATH = Path("configs/baselines/d4pm_eog_scoped.yaml")
SOURCE_PATH = Path(".external/D4PM")


def _components(
    clean_count: int = 60, artifact_count: int = 40
) -> tuple[np.ndarray, np.ndarray]:
    time = np.linspace(0.0, 2.0 * np.pi, 512, endpoint=False)
    clean = np.stack(
        [
            np.sin((1.0 + index / 20.0) * time) + 0.01 * index * np.cos(3.0 * time)
            for index in range(clean_count)
        ]
    )
    artifact = np.stack(
        [
            np.cos((0.5 + index / 30.0) * time) + 0.02 * index * np.sin(5.0 * time)
            for index in range(artifact_count)
        ]
    )
    return clean.astype(np.float64), artifact.astype(np.float64)


def _config() -> dict[str, object]:
    value = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_frozen_config_preserves_upstream_budget_and_two_arm_matrix() -> None:
    config = _config()
    validate_d4pm_config(config)
    assert config["benchmark_id"] == BENCHMARK_ID
    assert config["source"]["commit"] == SOURCE_COMMIT
    training = config["protocols"]["eog_scoped_seeded_native"]["training"]
    diffusion = config["protocols"]["eog_scoped_seeded_native"]["diffusion"]
    sampler = config["protocols"]["eog_scoped_seeded_native"]["joint_sampler"]
    assert training["epochs"] == OFFICIAL_EPOCHS
    assert training["batch_size"] == OFFICIAL_BATCH_SIZE
    assert training["features"] == OFFICIAL_FEATURES
    assert training["combinations"] == OFFICIAL_COMBINATIONS
    assert diffusion["num_steps"] == OFFICIAL_DIFFUSION_STEPS
    assert (sampler["lambda_dc"], sampler["gamma"], sampler["eta"]) == (
        JOINT_LAMBDA_DC,
        JOINT_GAMMA,
        JOINT_ETA,
    )
    assert tuple(tuple(row) for row in config["task_matrix"]) == TASK_MATRIX
    assert config["budget"]["optimizer_updates_per_branch"] == (
        EXPECTED_FULL_UPDATES_PER_BRANCH
    )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda c: c["protocols"]["eog_scoped_seeded_native"]["training"].__setitem__(
            "epochs", 3999
        ), "4000 epochs"),
        (lambda c: c["protocols"]["eog_scoped_seeded_native"]["training"].__setitem__(
            "batch_size", 512
        ), "batch_size=1024"),
        (lambda c: c["protocols"]["eog_scoped_seeded_native"]["training"].__setitem__(
            "features", 64
        ), "features=128"),
        (lambda c: c["protocols"]["eog_scoped_seeded_native"]["diffusion"].__setitem__(
            "num_steps", 100
        ), "500 diffusion steps"),
        (lambda c: c["protocols"]["eog_scoped_seeded_native"]["joint_sampler"].__setitem__(
            "eta", 0.8
        ), "eta=0.3"),
        (lambda c: c["protocols"]["eog_scoped_seeded_native"]["split"].__setitem__(
            "train_fraction", 0.9
        ), "train fraction must remain 0.8"),
        (lambda c: c["protocols"]["eog_scoped_seeded_native"]["mixture"].__setitem__(
            "recipe_policy", "physically_corrected"
        ), "must not be physically corrected"),
        (lambda c: c["randomness"].__setitem__("adapter_seed", 1), "seed differs"),
        (lambda c: c["scope"].__setitem__("artifact_routes_executed", ["EOG", "EMG"]),
         "frozen to the EOG route"),
        (lambda c: c["data"].__setitem__("ecg", "/some/path/ECG_all_epochs.npy"),
         "declared absent"),
        (lambda c: c["arms"]["joint_dual_diffusion"].__setitem__(
            "artifact_branch_architecture", "DualBranchDenoisingModel_two_heads"
        ), "four-head noise class"),
        (lambda c: c["arms"]["matched_deterministic"].__setitem__(
            "optimizer_updates", "half_of_the_diffusion_arm"
        ), "cannot receive fewer optimizer updates"),
        (lambda c: c["budget"].__setitem__("optimizer_updates_per_branch", 100),
         "budget field optimizer_updates_per_branch"),
        (lambda c: c["known_upstream_issues"].__setitem__("ecg_not_reproducible", ""),
         "ecg_not_reproducible must stay disclosed"),
        (lambda c: c["execution"].__setitem__("gpu_walltime", "12:00:00"),
         "23:59:59"),
        (lambda c: c["execution"].__setitem__("gpu_excluded_nodes", []),
         "exclude node54"),
        (lambda c: c["execution"].__setitem__("array", "0-7%8"), "0-1%2"),
        (lambda c: c["source"].__setitem__("load_policy", "vendored_copy"),
         "dynamically loaded"),
        (lambda c: c["source"].__setitem__("commit", "0" * 40), "commit differs"),
        (lambda c: c.__setitem__("claim_scope", "clinical_multichannel_claim"),
         "participant-specific claims"),
    ],
)
def test_config_validation_rejects_frozen_budget_drift(mutation, match: str) -> None:
    config = deepcopy(_config())
    mutation(config)
    with pytest.raises(ValueError, match=match):
        validate_d4pm_config(config)


def test_config_validation_rejects_indivisible_gradient_accumulation() -> None:
    config = deepcopy(_config())
    config["protocols"]["eog_scoped_seeded_native"]["training"][
        "gradient_accumulation_steps"
    ] = 3
    with pytest.raises(ValueError, match="divide the frozen batch"):
        validate_d4pm_config(config)


def test_training_config_view_ignores_only_the_aggregate_route_annotation() -> None:
    current = _config()
    submitted = deepcopy(current)
    submitted["execution"]["stages"].remove("aggregate-full")
    submitted["execution"].pop("aggregate_command")
    assert _training_config_view(submitted) == _training_config_view(current)

    changed_seed = deepcopy(submitted)
    changed_seed["randomness"]["adapter_seed"] += 1
    assert _training_config_view(changed_seed) != _training_config_view(current)


def test_checkout_validation_rejects_a_foreign_path() -> None:
    with pytest.raises(ValueError, match="unexpected D4PM external checkout path"):
        validate_d4pm_checkout(Path("src"), expected_commit=SOURCE_COMMIT)


def test_checkout_validation_rejects_a_drifted_commit() -> None:
    assert SOURCE_PATH.is_dir(), "run the frozen D4PM checkout Slurm job first"
    with pytest.raises(RuntimeError, match="not at the frozen commit"):
        validate_d4pm_checkout(SOURCE_PATH, expected_commit="0" * 40)


def test_frozen_checkout_is_clean_unvendored_and_unlicensed() -> None:
    assert SOURCE_PATH.is_dir(), "run the frozen D4PM checkout Slurm job first"
    status = validate_d4pm_checkout(SOURCE_PATH, expected_commit=SOURCE_COMMIT)
    assert status["commit"] == SOURCE_COMMIT
    assert status["tracked_checkout_clean"] is True
    assert status["upstream_license_file"] == "absent_at_frozen_commit"
    assert status["vendored"] is False
    assert not list(Path("src/eeg_cgdr").rglob("DDPM_joint.py"))


def test_frozen_external_source_matches_the_recorded_defect_audit() -> None:
    assert SOURCE_PATH.is_dir(), "run the frozen D4PM checkout Slurm job first"
    findings = audit_d4pm_source_text(
        (SOURCE_PATH / "train_d4pm_artifacts.py").read_text(encoding="utf-8"),
        (SOURCE_PATH / "Data_Preparation" / "data_for_eegdnet.py").read_text(
            encoding="utf-8"
        ),
        (SOURCE_PATH / "test_joint.py").read_text(encoding="utf-8"),
        (SOURCE_PATH / "utils.py").read_text(encoding="utf-8"),
    )
    assert findings == EXPECTED_SOURCE_FINDINGS


def test_seeded_pairing_is_deterministic_and_arm_independent() -> None:
    clean, artifact = _components()
    first = prepare_eog_scoped(
        clean, artifact, seed=20260830, combinations=3, test_snr_db=(-5.0, 0.0, 5.0)
    )
    second = prepare_eog_scoped(
        clean, artifact, seed=20260830, combinations=3, test_snr_db=(-5.0, 0.0, 5.0)
    )
    np.testing.assert_array_equal(
        first.train.clean_source_epoch, second.train.clean_source_epoch
    )
    np.testing.assert_array_equal(
        first.train.artifact_source_epoch, second.train.artifact_source_epoch
    )
    np.testing.assert_array_equal(first.train.snr_db, second.train.snr_db)
    np.testing.assert_array_equal(first.train.noisy, second.train.noisy)
    np.testing.assert_array_equal(first.validation.noisy, second.validation.noisy)
    for left, right in zip(first.evaluation, second.evaluation, strict=True):
        assert left.snr_db == right.snr_db
        np.testing.assert_array_equal(left.pairs.noisy, right.pairs.noisy)

    other = prepare_eog_scoped(
        clean, artifact, seed=20260831, combinations=3, test_snr_db=(-5.0, 0.0, 5.0)
    )
    assert not np.array_equal(
        first.train.clean_source_epoch, other.train.clean_source_epoch
    )


def test_scoped_split_sizes_and_source_audit_match_the_upstream_eog_route() -> None:
    clean, artifact = _components(clean_count=60, artifact_count=40)
    prepared = prepare_eog_scoped(
        clean, artifact, seed=11, combinations=2, test_snr_db=(0.0,)
    )
    # 40 EOG rows: 32 train, 4 validation, 4 evaluation, each x2 mixtures.
    assert len(prepared.train.clean) == 64
    assert len(prepared.validation.clean) == 8
    assert len(prepared.evaluation[0].pairs.clean) == 8
    audit = prepared.source_audit
    assert audit["upstream_seed_defined"] is False
    assert audit["source_epoch_repetitions_within_split"] == 2
    assert audit["clean_source_epochs_dropped_by_truncation"] == 20
    assert audit["cross_artifact_class_overlap"] == "not_applicable_eog_only_scope"
    for key in (
        "train_validation_clean_overlap",
        "train_validation_artifact_overlap",
        "train_evaluation_clean_overlap",
        "train_evaluation_artifact_overlap",
        "validation_evaluation_clean_overlap",
        "validation_evaluation_artifact_overlap",
    ):
        assert audit[key] == 0


def test_mixture_preserves_the_upstream_mean_square_snr_recipe() -> None:
    clean, artifact = _components(clean_count=12, artifact_count=12)
    prepared = prepare_eog_scoped(
        clean, artifact, seed=5, combinations=1, test_snr_db=(3.0,)
    )
    pairs = prepared.evaluation[0].pairs
    np.testing.assert_allclose(
        pairs.noisy, pairs.clean + pairs.artifact, rtol=0.0, atol=1e-6
    )
    clean_power = np.mean(np.square(pairs.clean.astype(np.float64)), axis=1)
    artifact_scaled = pairs.artifact.astype(np.float64)
    residual_power = np.mean(np.square(artifact_scaled), axis=1)
    expected_amplitude = np.sqrt(10.0 ** (0.1 * 3.0))
    # coefficient = mean_square(clean) / (mean_square(artifact) * sqrt(10^(snr/10)))
    np.testing.assert_allclose(
        np.sqrt(residual_power / clean_power) * expected_amplitude,
        np.ones_like(clean_power),
        rtol=2e-3,
    )


def test_evaluation_reuses_exact_pairs_across_snr_levels() -> None:
    clean, artifact = _components()
    prepared = prepare_eog_scoped(
        clean, artifact, seed=29, combinations=2, test_snr_db=(-5.0, 0.0, 5.0)
    )
    reference = prepared.evaluation[0].pairs
    for level in prepared.evaluation[1:]:
        np.testing.assert_array_equal(
            reference.clean_source_epoch, level.pairs.clean_source_epoch
        )
        np.testing.assert_array_equal(
            reference.artifact_source_epoch, level.pairs.artifact_source_epoch
        )
        np.testing.assert_allclose(reference.clean, level.pairs.clean)
    assert not np.allclose(
        prepared.evaluation[0].pairs.noisy, prepared.evaluation[-1].pairs.noisy
    )


def test_split_manifest_never_invents_participant_identity() -> None:
    clean, artifact = _components(clean_count=40, artifact_count=30)
    prepared = prepare_eog_scoped(
        clean, artifact, seed=31, combinations=1, test_snr_db=(0.0,)
    )
    rows = source_split_manifest_rows(prepared)
    assert rows
    assert all(row["identity_unit"] == "source_epoch_not_participant" for row in rows)
    assert all("participant" not in row for row in rows)
    assert all("+" not in row["split_membership"] for row in rows)


class _FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_inputs: tuple[torch.Tensor, ...] | None = None

    def forward(
        self,
        latent: torch.Tensor,
        condition: torch.Tensor,
        level: torch.Tensor,
        label: torch.Tensor,
    ) -> torch.Tensor:
        self.last_inputs = (latent, condition, level, label)
        return condition * 0.5


def test_matched_deterministic_sees_only_the_noisy_condition_and_label() -> None:
    backbone = _FakeBackbone()
    model = MatchedConditionOnly(backbone)
    noisy = torch.randn(3, 1, 512)
    label = torch.zeros(3, 1)
    output = model(noisy, label)
    assert torch.equal(output, noisy * 0.5)
    assert backbone.last_inputs is not None
    latent, condition, level, seen_label = backbone.last_inputs
    assert torch.equal(latent, noisy)
    assert torch.equal(condition, noisy)
    assert torch.equal(level, torch.ones(3, 1))
    assert torch.equal(seen_label, label)
    assert model.visible_inputs == (
        "noisy_single_channel_epoch",
        "artifact_class_label",
    )


def test_spectral_metric_uses_the_shape_consistent_upstream_definition() -> None:
    clean = np.arange(2 * 512, dtype=np.float64).reshape(2, 512) / 100.0
    denoised = clean + 0.25
    clean_magnitude = np.abs(np.fft.fft(clean, axis=-1))
    denoised_magnitude = np.abs(np.fft.fft(denoised, axis=-1))
    expected = d4pm_rrmse_temporal(denoised_magnitude, clean_magnitude)
    assert d4pm_rrmse_spectral_fft_magnitude(denoised, clean) == pytest.approx(expected)
    assert clean_magnitude.shape == denoised_magnitude.shape == (2, 512)


def _metric_rows(arm: str) -> list[dict[str, object]]:
    offset = 0.1 if arm == "joint_dual_diffusion" else 0.0
    return [
        {
            "benchmark_id": BENCHMARK_ID,
            "protocol": TASK_MATRIX[0][0],
            "noise_type": "EOG",
            "arm": arm,
            "identity_unit": "source_epoch_not_participant",
            "evaluation_scope": "all_frozen_test_rows",
            "snr_db": snr_db,
            "evaluation_mixtures": EXPECTED_FULL_EVALUATION_MIXTURES,
            "rrmse_temporal": 0.4 - offset,
            "correlation": 0.8 + offset,
            "snr_output_db": 6.0 + offset,
            "snr_input_db": 0.0,
            "snr_improvement_db": 6.0 + offset,
            "rrmse_spectral_fft_magnitude": 0.5 - offset,
            "rrmse_spectral_metric_status": SPECTRAL_METRIC_STATUS,
            "evaluation_seconds": 10.0,
            "network_calls_per_output": (
                OFFICIAL_DIFFUSION_STEPS * 2
                if arm == "joint_dual_diffusion"
                else 1
            ),
        }
        for snr_db in OFFICIAL_TEST_SNR_DB
    ]


def _aggregate_cells() -> dict[tuple[str, str, str], dict[str, object]]:
    manifest = [
        {
            "dataset": "EEGdenoiseNet",
            "protocol": TASK_MATRIX[0][0],
            "noise_type": "EOG",
            "identity_unit": "source_epoch_not_participant",
            "source_kind": "clean_EEG",
            "source_epoch": "0",
            "split_membership": "evaluation",
        }
    ]
    audit = {
        "train_evaluation_clean_overlap": 0,
        "train_evaluation_artifact_overlap": 0,
        "source_epoch_repetitions_within_split": OFFICIAL_COMBINATIONS,
    }
    cells: dict[tuple[str, str, str], dict[str, object]] = {}
    for key in TASK_MATRIX:
        arm = key[2]
        branches = (
            ["clean_eeg", "artifact"]
            if arm == "joint_dual_diffusion"
            else ["clean_eeg"]
        )
        cells[key] = {
            "summary": {
                "status": "completed",
                "stage": "full",
                "scientific_result_eligible": True,
                "benchmark_id": BENCHMARK_ID,
                "protocol": key[0],
                "noise_type": key[1],
                "arm": arm,
                "identity_unit": "source_epoch_not_participant",
                "branches": branches,
                "optimizer_updates_per_branch": EXPECTED_FULL_UPDATES_PER_BRANCH,
                "planned_optimizer_updates_per_branch": (
                    EXPECTED_FULL_UPDATES_PER_BRANCH
                ),
                "optimizer_updates_total": (
                    EXPECTED_FULL_UPDATES_PER_BRANCH * len(branches)
                ),
                "matched_update_budget": True,
                "training_seconds": 100.0,
                "peak_gpu_memory_mb": 1000.0,
                "gpu_name": "same-test-gpu",
                "source_audit": dict(audit),
            },
            "metrics": _metric_rows(arm),
            "split_manifest": [dict(row) for row in manifest],
        }
    return cells


def _pairing_acceptance() -> dict[str, object]:
    return {
        "status": "passed_reconstructed_ordered_pairing_acceptance",
        "git_head": "fixture-head",
        "task_indices": [0, 1],
        "submitted_and_resolved_configs_equal": True,
        "both_arms_saw_identical_inputs": True,
        "reconstruction_basis": "frozen_config_plus_frozen_adapter_seed",
        "scientific_threshold_or_method_changed": False,
        "pairing_rows": [
            {
                "protocol": TASK_MATRIX[0][0],
                "noise_type": "EOG",
                "train_pairs": EXPECTED_FULL_TRAIN_PAIRS,
                "validation_pairs": EXPECTED_FULL_VALIDATION_PAIRS,
                "evaluation_mixtures_per_snr": EXPECTED_FULL_EVALUATION_MIXTURES,
                "snr_levels": len(OFFICIAL_TEST_SNR_DB),
                "ordered_clean_artifact_snr_pairing_equal": True,
            }
        ],
    }


def test_full_aggregate_pairs_both_arms_and_discloses_the_branch_asymmetry() -> None:
    result = aggregate_d4pm_full_cells(
        _aggregate_cells(), pairing_acceptance=_pairing_acceptance()
    )
    assert result["status"] == "completed_full_aggregate"
    assert result["matrix_cells_completed"] == len(TASK_MATRIX)
    assert len(result["all_metric_rows"]) == len(TASK_MATRIX) * len(
        OFFICIAL_TEST_SNR_DB
    )
    assert len(result["paired_rows"]) == len(OFFICIAL_TEST_SNR_DB)
    assert all(
        row["total_compute_asymmetry_favours"] == "joint_dual_diffusion"
        for row in result["paired_rows"]
    )
    assert all(
        row["delta_rrmse_temporal"] == pytest.approx(-0.1)
        for row in result["paired_rows"]
    )
    assert result["spectral_metric"][
        "upstream_denominator_shape_defect_present"
    ] is False
    joint_summary = next(
        row
        for row in result["cell_summary_rows"]
        if row["arm"] == "joint_dual_diffusion"
    )
    assert joint_summary["branches"] == ARM_BRANCH_COUNT["joint_dual_diffusion"]
    assert joint_summary["optimizer_updates_total"] == (
        2 * EXPECTED_FULL_UPDATES_PER_BRANCH
    )


def test_full_aggregate_rejects_a_short_optimizer_budget() -> None:
    short = _aggregate_cells()
    short[TASK_MATRIX[1]]["summary"]["optimizer_updates_per_branch"] = (
        EXPECTED_FULL_UPDATES_PER_BRANCH - 1
    )
    short[TASK_MATRIX[1]]["summary"]["planned_optimizer_updates_per_branch"] = (
        EXPECTED_FULL_UPDATES_PER_BRANCH - 1
    )
    short[TASK_MATRIX[1]]["summary"]["optimizer_updates_total"] = (
        EXPECTED_FULL_UPDATES_PER_BRANCH - 1
    )
    with pytest.raises(ValueError, match="optimizer budget is incomplete"):
        aggregate_d4pm_full_cells(short, pairing_acceptance=_pairing_acceptance())


def test_full_aggregate_rejects_a_mismatched_branch_accounting() -> None:
    mismatched = _aggregate_cells()
    mismatched[TASK_MATRIX[0]]["summary"]["branches"] = ["clean_eeg"]
    with pytest.raises(ValueError, match="branch count mismatch"):
        aggregate_d4pm_full_cells(mismatched, pairing_acceptance=_pairing_acceptance())


def test_full_aggregate_rejects_truncated_evaluation_and_missing_cells() -> None:
    truncated = _aggregate_cells()
    truncated[TASK_MATRIX[0]]["metrics"][0]["evaluation_mixtures"] -= 1
    with pytest.raises(ValueError, match="truncated D4PM evaluation"):
        aggregate_d4pm_full_cells(truncated, pairing_acceptance=_pairing_acceptance())

    missing = _aggregate_cells()
    del missing[TASK_MATRIX[-1]]
    with pytest.raises(ValueError, match="requires both cells"):
        aggregate_d4pm_full_cells(missing, pairing_acceptance=_pairing_acceptance())


def test_full_aggregate_rejects_unequal_inputs_or_manifests() -> None:
    acceptance = _pairing_acceptance()
    acceptance["both_arms_saw_identical_inputs"] = False
    with pytest.raises(ValueError, match="provably share one ordered pairing"):
        aggregate_d4pm_full_cells(_aggregate_cells(), pairing_acceptance=acceptance)

    failed = _pairing_acceptance()
    failed["pairing_rows"][0]["ordered_clean_artifact_snr_pairing_equal"] = False
    with pytest.raises(ValueError, match="ordered clean/artifact/SNR pairing"):
        aggregate_d4pm_full_cells(_aggregate_cells(), pairing_acceptance=failed)

    mismatched = _aggregate_cells()
    mismatched[TASK_MATRIX[1]]["split_manifest"] = [{"source_epoch": "different"}]
    with pytest.raises(ValueError, match="exact source manifest"):
        aggregate_d4pm_full_cells(mismatched, pairing_acceptance=_pairing_acceptance())


def test_full_aggregate_rejects_a_dropped_spectral_disclosure() -> None:
    dropped = _aggregate_cells()
    for row in dropped[TASK_MATRIX[0]]["metrics"]:
        row["rrmse_spectral_metric_status"] = "rrmse_s"
    with pytest.raises(ValueError, match="spectral metric disclosure"):
        aggregate_d4pm_full_cells(dropped, pairing_acceptance=_pairing_acceptance())


def test_full_aggregate_rejects_the_upstream_fifty_row_example_as_a_result() -> None:
    example = _aggregate_cells()
    for row in example[TASK_MATRIX[0]]["metrics"]:
        row["evaluation_scope"] = "upstream_example_first_50_rows_diagnostic_only"
    with pytest.raises(ValueError, match="evaluation_scope mismatch"):
        aggregate_d4pm_full_cells(example, pairing_acceptance=_pairing_acceptance())


def test_planned_update_budget_matches_the_frozen_config() -> None:
    config = _config()
    budget = config["budget"]
    train_pairs = int(budget["train_pairs"])
    updates_per_epoch = train_pairs // OFFICIAL_BATCH_SIZE
    assert updates_per_epoch == int(budget["optimizer_updates_per_epoch"])
    assert updates_per_epoch * OFFICIAL_EPOCHS == EXPECTED_FULL_UPDATES_PER_BRANCH
    assert int(budget["joint_dual_diffusion_total_updates"]) == (
        EXPECTED_FULL_UPDATES_PER_BRANCH * ARM_BRANCH_COUNT["joint_dual_diffusion"]
    )
    assert int(budget["matched_deterministic_total_updates"]) == (
        EXPECTED_FULL_UPDATES_PER_BRANCH * ARM_BRANCH_COUNT["matched_deterministic"]
    )


def test_cli_routes_all_four_stages_and_preserves_exit_75() -> None:
    cli = Path("src/eeg_cgdr/cli/main.py").read_text(encoding="utf-8")
    assert '"d4pm-benchmark",' in cli
    assert "run_d4pm_cpu_validation(config, run_dir=args.run_dir)" in cli
    assert "run_d4pm_stage(" in cli
    assert "run_d4pm_full_aggregate(config, run_dir=args.run_dir)" in cli
    assert '_array_task_index(f"d4pm-{args.stage}")' in cli
    assert (
        "d4pm-benchmark requires cpu-tests, smoke, full, or aggregate-full" in cli
    )


def test_job_script_routes_profiles_and_array_shape() -> None:
    job = Path("scripts/slurm/jobs/cgdr.sbatch").read_text(encoding="utf-8")
    assert "d4pm-benchmark)" in job
    assert "d4pm-benchmark cpu-tests requires cpu" in job
    assert "d4pm-benchmark smoke requires V100-32GB, L40S or gpu-any" in job
    assert "d4pm-benchmark full requires A100, H100 or gpu-any" in job
    assert "d4pm-benchmark aggregate-full requires cpu" in job
    assert "d4pm-benchmark %s requires array task 0 or 1" in job
    assert "tests/unit/test_cgdr_d4pm_benchmark.py" in job


def test_submitter_freezes_the_walltime_and_the_bad_node_exclusion() -> None:
    submitter = Path("scripts/slurm/submit.sh").read_text(encoding="utf-8")
    config = _config()
    assert '== d4pm-benchmark ]]' in submitter
    assert "d4pm full requires --array 0-1%%2 or one retry index 0-1" in submitter
    assert 'walltime="23:59:59"' in submitter
    assert "extra_sbatch_args+=(--exclude=node54)" in submitter
    execution = config["execution"]
    assert execution["gpu_walltime"] == "23:59:59"
    assert execution["gpu_excluded_nodes"] == ["node54"]
    for command in ("smoke_command", "full_command"):
        assert "--array 0-1%2" in execution[command]
    assert str(CONFIG_PATH) in execution["full_command"]


def test_planned_update_budget_matches_the_real_eog_row_arithmetic() -> None:
    # 3400 EOG rows: round(0.8 * 3400) = 2720 train rows, x11 mixtures.
    train_rows = round(0.8 * 3400) * OFFICIAL_COMBINATIONS
    assert train_rows == EXPECTED_FULL_TRAIN_PAIRS
    remaining = 3400 - round(0.8 * 3400)
    assert (remaining // 2) * OFFICIAL_COMBINATIONS == EXPECTED_FULL_VALIDATION_PAIRS
    assert (
        remaining - remaining // 2
    ) * OFFICIAL_COMBINATIONS == EXPECTED_FULL_EVALUATION_MIXTURES
    assert (
        train_rows // OFFICIAL_BATCH_SIZE
    ) * OFFICIAL_EPOCHS == EXPECTED_FULL_UPDATES_PER_BRANCH
