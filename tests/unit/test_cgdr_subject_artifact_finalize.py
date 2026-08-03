"""Tests for the read-only-input, mechanical J6 finalizer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eeg_cgdr.experiments import subject_artifact_finalize as finalize


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _config(tmp_path: Path) -> dict[str, object]:
    root = tmp_path / "results/cgdr/subject_calibrated_artifact_diffusion"
    revision = root / "revisions/j2_v1_identity_routing_r2"
    return {
        "harness_level": 1,
        "protocol_id": finalize.PROTOCOL_ID,
        "scientific_role": "development_exploratory_not_confirmation",
        "boundaries": {
            "query_eog_or_eye_tracking_input": "forbidden",
            "query_artifact_label_input": "forbidden",
            "query_outcome_input": "forbidden",
            "best_of_k_selection": "forbidden",
            "confirmation_outcomes_this_round": "forbidden",
        },
        "validity": {"execution_revision": finalize.EXECUTION_REVISION},
        "outputs": {
            "root": str(root),
            "validity_root": str(revision / "validity"),
            "development_root": str(revision / "development"),
            "checkpoint_root": str(revision / "checkpoints"),
            "metrics": str(revision / "metrics.csv"),
            "summary": str(revision / "result_summary.json"),
            "figures": str(revision / "figures"),
        },
    }


def _source_paths(config: dict[str, object]) -> finalize.FinalizerPaths:
    return finalize._paths(config)


def _j0() -> dict[str, object]:
    return {
        "status": "passed_j0_manifest_and_target_availability",
        "protocol_id": finalize.PROTOCOL_ID,
        "slurm_job_id": "101",
        "confirmation_signal_or_outcome_opened": False,
        "file_hashes_computed": False,
        "data_download_performed": False,
    }


def _j1() -> dict[str, object]:
    return {
        "status": "passed_j1_real_record_context_validation",
        "protocol_id": finalize.PROTOCOL_ID,
        "slurm_job_id": "102",
        "confirmation_signal_or_outcome_opened": False,
    }


def _validity_gate(
    paths: finalize.FinalizerPaths, *, passed: bool
) -> dict[str, object]:
    implementation = "primary_attempt_1"
    model_validity = "passed" if passed else "failed"
    attempt_status = "passed" if passed else "failed"
    gate_status = (
        "passed_V0_to_V3" if passed else "completed_model_validity_failed"
    )
    selected = {
        "implementation": implementation,
        "execution_revision": finalize.EXECUTION_REVISION,
        "attempt_result_path": str(
            paths.validity_gate.parent
            / finalize.EXECUTION_REVISION
            / implementation
            / "result_summary.json"
        ),
        "status": attempt_status,
        "passed": passed,
        "model_validity": model_validity,
        "validity": {
            level: {
                "status": "passed" if passed else "failed",
                "passed": passed,
            }
            for level in ("V0", "V1", "V2", "V3")
        },
    }
    return {
        "protocol_id": finalize.PROTOCOL_ID,
        "execution_revision": finalize.EXECUTION_REVISION,
        "status": gate_status,
        "passed": passed,
        "model_validity": model_validity,
        "selected_implementation": implementation,
        "selected_result": selected,
        "slurm_job_id": "103",
    }


def _j5(*, validity_passed: bool) -> dict[str, object]:
    if validity_passed:
        return {
            "status": "completed_development_aggregate_inconclusive",
            "protocol_id": finalize.PROTOCOL_ID,
            "execution_revision": finalize.EXECUTION_REVISION,
            "computational_completion": "passed",
            "model_validity": "passed",
            "scientific_comparison_eligibility": "development_descriptive_only",
            "protocol_decision": "inconclusive",
            "topic_status": "not_yet_testable",
            "confirmation_eligibility": False,
            "family_wide_status": "not_tested",
            "real_EEG_evidence_scope": "all_eligible_SGE_development_only",
            "query_confirmation_outcomes_opened": False,
            "G_calibration": {
                "status": "not_testable_missing_paired_mechanism_evidence",
                "overall_passed": False,
                "paired_mechanism": {"status": "not_run"},
            },
            "G_diffusion": {
                "status": "not_testable_primary_endpoint_not_operationalized",
                "overall_passed": False,
            },
            "uncertainty": {
                "status": "not_testable_missing_matched_uncertainty_comparison",
                "overall_passed": False,
            },
            "calibration_duration": {
                "status": "not_testable_single_30_second_condition"
            },
            "slurm_job_id": "104",
        }
    return {
        "status": "completed_fail_closed_model_validity_failed",
        "protocol_id": finalize.PROTOCOL_ID,
        "execution_revision": finalize.EXECUTION_REVISION,
        "computational_completion": "passed_fail_closed_terminal",
        "model_validity": "failed",
        "scientific_comparison_eligibility": "blocked",
        "protocol_decision": "inconclusive",
        "topic_status": "not_yet_testable",
        "confirmation_eligibility": False,
        "family_wide_status": "not_tested",
        "query_confirmation_outcomes_opened": False,
        "G_calibration": {"status": "not_run_blocked_by_V0_V3"},
        "G_diffusion": {"status": "not_run_blocked_by_V0_V3"},
        "uncertainty": {"status": "not_tested"},
        "slurm_job_id": "104",
    }


def _prepare_inputs(
    config: dict[str, object], *, validity_passed: bool
) -> tuple[finalize.FinalizerPaths, dict[str, object]]:
    paths = _source_paths(config)
    gate = _validity_gate(paths, passed=validity_passed)
    _write_json(paths.j0, _j0())
    _write_json(paths.j1, _j1())
    _write_json(paths.validity_gate, gate)
    _write_json(paths.j5, _j5(validity_passed=validity_passed))
    _write_json(
        paths.old_frozen_decision,
        {
            "current_M2_status": "current_M2_no_incremental_value",
            "diffusion_family_wide_status": "not_tested",
            "formal_G1_status": "NOT_RUN_BLOCKED",
            "formal_G3_status": "NOT_RUN_BLOCKED",
            "sentinel": "must_not_be_overwritten",
        },
    )
    paths.job_ledger.parent.mkdir(parents=True, exist_ok=True)
    paths.job_ledger.write_text(
        "J0 audit | 101 | passed\n"
        "J1 validation | 102 | passed\n"
        "J2 validity | 103 | terminal\n"
        "J5 aggregate | 104 | terminal\n"
        "J6 finalize | 105 | submitted\n",
        encoding="utf-8",
    )
    return paths, gate


def _implementation() -> dict[str, object]:
    return {
        "git_commit": "a" * 40,
        "slurm_job_id": "105",
        "slurm_profile": "cpu",
    }


def test_validity_failure_finalizes_without_opening_scientific_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    paths, _ = _prepare_inputs(config, validity_passed=False)
    original_old = paths.old_frozen_decision.read_text(encoding="utf-8")
    reads: list[Path] = []
    original_read_json = finalize._read_json

    def tracked_read(path: Path) -> dict[str, object]:
        reads.append(path)
        return original_read_json(path)

    monkeypatch.setattr(finalize, "_read_json", tracked_read)
    summary = finalize.run_subject_artifact_finalize(
        config, tmp_path / "run_j6", _implementation()
    )

    assert set(reads) == {
        paths.j0,
        paths.j1,
        paths.validity_gate,
        paths.j5,
        paths.old_frozen_decision,
    }
    assert summary["model_validity"] == "failed"
    assert summary["protocol_decision"] == "inconclusive"
    assert summary["confirmation_eligibility"] is False
    assert summary["confirmation_job_generated"] is False
    assert "V0_to_V3_not_passed" in summary["confirmation_blockers"]
    assert paths.output_summary.is_file()
    assert paths.development_report.is_file()
    assert paths.confirmation_plan.is_file()
    assert paths.terminal_manifest.is_file()
    assert paths.old_frozen_decision.read_text(encoding="utf-8") == original_old
    plan = paths.confirmation_plan.read_text(encoding="utf-8")
    assert "Status: `blocked`" in plan
    assert "generated no Slurm confirmation job" in plan
    assert "sbatch" not in plan

    terminal = json.loads(paths.terminal_manifest.read_text(encoding="utf-8"))
    assert set(terminal) == {
        "job_id",
        "implementation_git_sha",
        "final_report_git_sha",
        "config",
        "command",
        "conda_environment",
        "status",
        "runtime_seconds",
        "result_path",
    }
    assert terminal["job_id"] == "105"
    assert terminal["implementation_git_sha"] == "a" * 40
    assert terminal["final_report_git_sha"] == "pending_post_J6_report_commit"
    assert terminal["conda_environment"] == finalize.CPU_ENVIRONMENT
    assert terminal["status"] == "passed"
    assert terminal["result_path"] == str(paths.output_summary)
    assert terminal["runtime_seconds"] >= 0.0


def test_j6_does_not_race_on_its_own_ledger_entry(tmp_path: Path) -> None:
    config = _config(tmp_path)
    paths, _ = _prepare_inputs(config, validity_passed=False)
    ledger = paths.job_ledger.read_text(encoding="utf-8")
    paths.job_ledger.write_text(
        "\n".join(
            line for line in ledger.splitlines() if not line.startswith("J6 finalize")
        )
        + "\n",
        encoding="utf-8",
    )

    summary = finalize.run_subject_artifact_finalize(
        config, tmp_path / "run_j6", _implementation()
    )

    assert summary["slurm_job_id"] == "105"
    assert summary["confirmation_eligibility"] is False


def test_passed_validity_but_missing_paired_and_uncertainty_stays_blocked(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    paths, _ = _prepare_inputs(config, validity_passed=True)

    summary = finalize.run_subject_artifact_finalize(
        config, tmp_path / "run_j6", _implementation()
    )

    assert summary["model_validity"] == "passed"
    assert summary["confirmation_eligibility"] is False
    assert summary["confirmation_job_generated"] is False
    assert summary["topic_status"] == "not_yet_testable"
    assert summary["confirmation_blockers"] == [
        "paired_mechanism_evidence_missing_or_not_passed",
        "matched_uncertainty_comparison_missing_or_not_passed",
    ]
    report = paths.development_report.read_text(encoding="utf-8")
    assert "not_testable_missing_paired_mechanism_evidence" in report
    assert "not_testable_missing_matched_uncertainty_comparison" in report


def test_j5_cannot_promote_confirmation_across_hard_missing_evidence(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    paths, _ = _prepare_inputs(config, validity_passed=True)
    aggregate = json.loads(paths.j5.read_text(encoding="utf-8"))
    aggregate["confirmation_eligibility"] = True
    aggregate["protocol_decision"] = "eligible_for_confirmation"
    _write_json(paths.j5, aggregate)

    with pytest.raises(ValueError, match="contradicts its hard blockers"):
        finalize.run_subject_artifact_finalize(
            config, tmp_path / "run_j6", _implementation()
        )
    assert not paths.output_summary.exists()
    assert not paths.terminal_manifest.exists()


def test_legacy_non_revision_validity_attempt_path_is_rejected(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    paths, gate = _prepare_inputs(config, validity_passed=False)
    selected = dict(gate["selected_result"])
    selected["attempt_result_path"] = str(
        paths.validity_gate.parent
        / "primary_attempt_1"
        / "result_summary.json"
    )
    gate["selected_result"] = selected
    _write_json(paths.validity_gate, gate)

    with pytest.raises(ValueError, match="selected attempt disagree"):
        finalize.run_subject_artifact_finalize(
            config, tmp_path / "run_j6", _implementation()
        )
