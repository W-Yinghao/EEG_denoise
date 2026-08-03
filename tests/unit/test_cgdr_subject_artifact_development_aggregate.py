"""Narrow tests for the fail-closed J5 participant-stem aggregate."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from eeg_cgdr.experiments import subject_artifact_development_aggregate as aggregate


def _output_config(tmp_path: Path) -> dict[str, object]:
    revision = tmp_path / "revision"
    return {
        "protocol_id": "subject_calibrated_artifact_latent_diffusion_development_v1",
        "validity": {"execution_revision": "test_revision"},
        "training": {"seeds": [11, 12, 13]},
        "development_gates": {
            "bootstrap_replicates": 20_000,
            "bootstrap_seed": 20260802,
            "confidence_level": 0.95,
        },
        "outputs": {
            "validity_root": str(revision / "validity"),
            "development_root": str(revision / "development"),
            "checkpoint_root": str(revision / "checkpoints"),
            "metrics": str(revision / "metrics.csv"),
            "summary": str(revision / "result_summary.json"),
            "figures": str(revision / "figures"),
        },
    }


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fold() -> aggregate.FoldSpec:
    return aggregate.FoldSpec(
        unified_fold_index=0,
        fold_id="study02_fold00",
        study="study02",
        layout_id="layout_01",
        sampling_rate_hz=500.0,
        reference_cell="as_delivered",
        heldout_recording_keys=("study02/study02_p01",),
    )


def _row(
    *,
    seed: int,
    model: str,
    context: str,
    remaining: float,
    status: str = "success",
) -> dict[str, object]:
    return {
        "recording_key": "study02/study02_p01",
        "training_seed": seed,
        "checkpoint_endpoint": "best_validation",
        "model_id": model,
        "context_id": context,
        "status": status,
        "performance_values_eligible": True,
        "heldout_eog_prediction_remaining_ratio": remaining,
    }


def test_failed_validity_is_terminal_without_reading_j3_or_j4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _output_config(tmp_path)
    outputs = config["outputs"]
    assert isinstance(outputs, dict)
    validity_root = Path(str(outputs["validity_root"]))
    selected_detail = (
        validity_root
        / "test_revision"
        / "primary_attempt_1"
        / "result_summary.json"
    )
    selected_result = {
        "implementation": "primary_attempt_1",
        "execution_revision": "test_revision",
        "attempt_result_path": str(selected_detail),
        "status": "failed",
        "passed": False,
        "model_validity": "failed",
        "validity": {
            "V0": {"status": "failed"},
            "V1": {"status": "passed"},
            "V2": {"status": "blocked"},
            "V3": {"status": "blocked"},
        },
    }
    _write_json(
        validity_root / "result_summary.json",
        {
            "protocol_id": config["protocol_id"],
            "execution_revision": "test_revision",
            "status": "completed_model_validity_failed",
            "passed": False,
            "model_validity": "failed",
            "selected_implementation": "primary_attempt_1",
            "selected_result": selected_result,
        },
    )

    def forbidden_pass_branch(*args: object, **kwargs: object) -> None:
        raise AssertionError("J5 tried to read J3/J4 after validity failure")

    monkeypatch.setattr(
        aggregate, "_aggregate_passed_validity", forbidden_pass_branch
    )
    summary = aggregate.run_subject_artifact_development_aggregate(
        config, tmp_path / "run"
    )

    assert summary["status"] == "completed_fail_closed_model_validity_failed"
    assert summary["confirmation_eligibility"] is False
    assert summary["J3_training"] == "not_run_blocked_by_V0_V3"
    assert summary["J4_factorial"] == "not_run_blocked_by_V0_V3"
    assert summary["selected_validity_summary"] == str(selected_detail)
    assert summary["figures"]["trajectory_rms"] == str(
        selected_detail.with_name("diffusion_trajectory_rms.png")
    )
    assert not Path(str(outputs["checkpoint_root"])).exists()
    with Path(str(outputs["metrics"])).open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert rows[0]["status"] == "model_validity_failed"

    legacy = dict(selected_result)
    legacy["attempt_result_path"] = str(
        validity_root / "primary_attempt_1" / "result_summary.json"
    )
    _write_json(
        validity_root / "result_summary.json",
        {
            "protocol_id": config["protocol_id"],
            "execution_revision": "test_revision",
            "status": "completed_model_validity_failed",
            "passed": False,
            "model_validity": "failed",
            "selected_implementation": "primary_attempt_1",
            "selected_result": legacy,
        },
    )
    with pytest.raises(ValueError, match="revision-nested"):
        aggregate.run_subject_artifact_development_aggregate(
            config, tmp_path / "legacy_run"
        )


def test_scientific_performance_requires_exact_success_eligibility_and_finite() -> None:
    assert aggregate._canonical_endpoint({"checkpoint_endpoint": "best"}) == (
        "best_validation"
    )
    assert aggregate._canonical_endpoint({"checkpoint_endpoint": "equal"}) == (
        "equal_update"
    )
    accepted = _row(
        seed=11,
        model="diffusion",
        context="matching",
        remaining=0.4,
    )
    assert aggregate._performance_eligible(
        accepted, "heldout_eog_prediction_remaining_ratio"
    )

    fallback = dict(accepted, status="success_population_fallback_rho_zero")
    blocked = dict(accepted, status="blocked_no_population")
    ineligible = dict(accepted, performance_values_eligible=False)
    nonfinite = dict(accepted, heldout_eog_prediction_remaining_ratio="nan")
    for row in (fallback, blocked, ineligible, nonfinite):
        assert not aggregate._performance_eligible(
            row, "heldout_eog_prediction_remaining_ratio"
        )


def test_contrasts_pair_within_seed_then_require_all_three_seeds(
    tmp_path: Path,
) -> None:
    config = _output_config(tmp_path)
    rows: list[dict[str, object]] = []
    for seed, offset in zip((11, 12, 13), (0.00, 0.01, -0.01), strict=True):
        rows.extend(
            (
                _row(
                    seed=seed,
                    model="diffusion",
                    context="matching",
                    remaining=0.40 + offset,
                ),
                _row(
                    seed=seed,
                    model="diffusion",
                    context="population",
                    remaining=0.70 + offset,
                ),
            )
        )
    contrasts = aggregate._build_stem_contrasts(config, rows, (_fold(),))
    target = next(
        row
        for row in contrasts
        if row["checkpoint_endpoint"] == "best_validation"
        and row["contrast_id"] == "delta_cal_diff"
        and row["metric"] == "heldout_eog_prediction_remaining_ratio"
    )
    assert target["complete_three_seed_pair"] is True
    assert target["successful_seed_pair_count"] == 3
    assert target["mean_effect"] == pytest.approx(0.30)

    rows[-2]["status"] = "success_population_fallback_rho_zero"
    incomplete = aggregate._build_stem_contrasts(config, rows, (_fold(),))
    target = next(
        row
        for row in incomplete
        if row["contrast_id"] == "delta_cal_diff"
        and row["metric"] == "heldout_eog_prediction_remaining_ratio"
    )
    assert target["complete_three_seed_pair"] is False
    assert target["successful_seed_pair_count"] == 2
    assert target["mean_effect"] is None


def test_coverage_keeps_59_stem_denominator_and_preblocked_p42(
    tmp_path: Path,
) -> None:
    config = _output_config(tmp_path)
    rows = [
        _row(
            seed=seed,
            model=model,
            context=context,
            remaining=0.5,
        )
        for seed in (11, 12, 13)
        for model in aggregate.MODEL_IDS
        for context in aggregate.CONTEXT_IDS
    ]
    coverage = aggregate._build_coverage(config, rows, (_fold(),))
    assert len(coverage) == 8
    assert all(row["availability_stem_denominator"] == 59 for row in coverage)
    assert all(row["compatible_stem_denominator"] == 58 for row in coverage)
    assert all(
        row["preblocked_recording_key"] == "study05/study05_p42"
        for row in coverage
    )


def test_exact_cell_participant_bootstrap_is_seeded_and_cell_stratified() -> None:
    first = aggregate.stratified_participant_bootstrap(
        {"cell_a": [2.0, 2.0], "cell_b": [2.0]},
        replicates=20_000,
        seed=20260802,
        confidence_level=0.95,
    )
    second = aggregate.stratified_participant_bootstrap(
        {"cell_a": [2.0, 2.0], "cell_b": [2.0]},
        replicates=20_000,
        seed=20260802,
        confidence_level=0.95,
    )
    assert first == second
    assert first["participant_stem_count"] == 3
    assert first["cell_count"] == 2
    assert first["mean"] == pytest.approx(2.0)
    assert first["ci_low"] == pytest.approx(2.0)
    assert first["ci_high"] == pytest.approx(2.0)


def test_no_complete_pair_writes_fail_closed_bootstrap_status(tmp_path: Path) -> None:
    rows = aggregate._build_bootstrap_rows(
        _output_config(tmp_path),
        [
            {
                "checkpoint_endpoint": "best_validation",
                "contrast_id": "delta_cal_diff",
                "metric": "heldout_eog_prediction_remaining_ratio",
                "exact_cell_id": "cell_a",
                "complete_three_seed_pair": False,
                "mean_effect": None,
            }
        ],
    )
    assert rows == [
        {
            "scope": "none",
            "status": "no_complete_three_seed_participant_contrast",
            "bootstrap_replicates": 20_000,
            "bootstrap_seed": 20260802,
            "confidence_level": 0.95,
            "participant_stem_count": 0,
        }
    ]
