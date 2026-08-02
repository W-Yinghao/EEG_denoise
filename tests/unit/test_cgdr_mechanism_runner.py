"""Small semantic tests for the repaired mechanism runner (engineering only)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from eeg_cgdr.evaluation.metrics import subspace_parallel_error_side_by_side
from eeg_cgdr.experiments.mechanism_runner import (
    _OperatorArm,
    _base_row,
    _fallback_rows,
    _load_progress,
    _mechanism_metrics,
    _state_for_chunk,
)
from eeg_cgdr.inference import DatasetPopulationProjector
from eeg_cgdr.operators import P0FitOutcome


def _signals() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time = np.arange(512, dtype=np.float64) / 256.0
    clean = np.stack(
        [np.sin(2.0 * np.pi * 4.0 * time), np.cos(2.0 * np.pi * 7.0 * time)]
    )
    artifact = np.stack(
        [0.4 * np.sin(2.0 * np.pi * 2.0 * time), np.zeros_like(time)]
    )
    observed = clean + artifact
    restored = clean + 0.5 * artifact
    projector = np.diag([1.0, 0.0])
    return clean, observed, restored, projector


def test_paper_parallel_metric_and_empty_artifact_interval() -> None:
    clean, observed, restored, projector = _signals()
    metrics = _mechanism_metrics(
        restored,
        observed=observed,
        clean=clean,
        oracle_projector=projector,
        estimated_projector=projector,
        artifact_mask=np.zeros(clean.shape[1], dtype=bool),
        sampling_rate=256.0,
    )
    expected = np.linalg.norm(projector @ (restored - clean)) / np.linalg.norm(
        projector @ clean
    )
    assert metrics["e_parallel"] == pytest.approx(expected)
    assert metrics["e_parallel_neural_normalized"] == pytest.approx(expected)
    assert metrics["artifact_normalized_parallel_error"] == pytest.approx(0.5)
    assert metrics["e_parallel_artifact_normalized"] == pytest.approx(0.5)
    side_by_side = subspace_parallel_error_side_by_side(
        restored,
        observed,
        projector,
        clean=clean,
    )
    assert side_by_side["e_parallel_neural_normalized"] == pytest.approx(expected)
    assert side_by_side["e_parallel_artifact_normalized"] == pytest.approx(0.5)
    assert side_by_side["legacy_e_parallel_alias_value"] == pytest.approx(0.5)
    assert side_by_side["parallel_clean_denominator_valid"] is True
    assert side_by_side["parallel_artifact_denominator_valid"] is True
    assert metrics["artifact_attenuation"] == ""


def test_rejected_operator_keeps_exact_pop_output_and_failure_status() -> None:
    rejected = _OperatorArm(
        source="matching_p0",
        projector=None,
        p0_outcome=P0FitOutcome("ineligible", None, ("constant_reference",)),
        calibration_id="sim37",
    )
    pop_rows = []
    for index, seed in enumerate((1, 2, 3, 4, 5, -1)):
        row = _base_row(
            partition="untouched",
            source_record=37,
            seed=seed,
            aggregate=seed == -1,
            method_id="population_only__M0__trust_1",
            operator_source="population_only",
            candidate="M0",
            trust_radius=1.0,
            calibration_seconds=10.0,
            arm=None,
        )
        row.update({"status": "success", "e_parallel": 0.25})
        if index == 1:
            row.update({"status": "failed", "failure_reason": "POP_seed_failed"})
        pop_rows.append(row)
    fallback = _fallback_rows(
        population_rows=pop_rows,
        partition="untouched",
        source_record=37,
        method_id="matching_p0__M0__trust_1",
        operator_source="matching_p0",
        candidate="M0",
        trust_radius=1.0,
        arm=rejected,
        population_method_id="population_only__M0__trust_1",
    )
    assert fallback[0]["status"] == "fallback_POP"
    assert fallback[0]["e_parallel"] == 0.25
    assert fallback[0]["fallback_used"] is True
    assert fallback[1]["status"] == "failed"
    assert "POP_seed_failed" in fallback[1]["failure_reason"]


def test_rho_zero_does_not_validate_or_construct_context_projector() -> None:
    observation = torch.zeros(1, 2, 16)
    eog = torch.zeros(1, 2, 16)
    weight = torch.ones(1, 16)
    population = DatasetPopulationProjector(
        dataset_id="fixture",
        montage_id="two_channel",
        projector=np.diag([1.0, 0.0]),
        source="outer_training_fixture",
    )
    deliberately_invalid_context = _OperatorArm(
        source="matching_p0",
        projector=None,
        p0_outcome=None,
        calibration_id="must_not_be_touched",
    )
    state, consistency = _state_for_chunk(
        observation=observation,
        standardized_eog=eog,
        valid_weight=weight,
        population_projector=population,
        arm=deliberately_invalid_context,
        config={
            "observation": {
                "attenuation_scale": 2.0,
                "attenuation_floor": 0.05,
                "base_precision": 1.0,
                "energy_scale": 0.15,
                "rho": 0.0,
            }
        },
    )
    assert state.name == "population_E0"
    np.testing.assert_array_equal(consistency, population.projector)


def test_resume_rejects_changed_plan_contract(tmp_path) -> None:
    path = tmp_path / "method_progress.json"
    payload = _load_progress(
        path,
        partition="development",
        source_record=31,
        plan_contract={"seeds": [1, 2, 3, 4, 5]},
    )
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen source-record plan"):
        _load_progress(
            path,
            partition="development",
            source_record=31,
            plan_contract={"seeds": [6, 7, 8, 9, 10]},
        )
