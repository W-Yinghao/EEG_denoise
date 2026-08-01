"""Contract tests for repaired source-record mechanism aggregation.

These tests exercise only tiny CSV fixtures.  They are engineering checks run by
the aggregate J0 CPU Slurm job and are not mechanism or gate evidence.
"""

from __future__ import annotations

import copy
import csv
import inspect
import itertools
import json
from pathlib import Path
from typing import Any, Iterable

import pytest
import yaml

from eeg_cgdr.experiments.mechanism_aggregate import (
    _absolute_baseline_effect,
    _paired_effect,
    _per_record_method,
    _validate_record_replicates,
    aggregate_development,
    aggregate_untouched_and_decide,
    classify_absolute_mechanism_evidence,
)
from eeg_cgdr.experiments.mechanism_audit import _validate_protocol


ALGORITHM_SEEDS = (101, 102, 103, 104, 105)


def _row(
    *,
    partition: str,
    record: int,
    method_id: str,
    operator_source: str,
    sampler_candidate: str,
    trust_radius: float,
    seed: int,
    aggregate: bool,
    e_parallel: float,
    e_perp: float,
    status: str = "success",
) -> dict[str, str]:
    return {
        "partition": partition,
        "source_record": str(record),
        "seed": str(seed),
        "aggregate_across_seeds": "true" if aggregate else "false",
        "method_id": method_id,
        "operator_source": operator_source,
        "sampler_candidate": sampler_candidate,
        "trust_radius": str(trust_radius),
        "status": status,
        "e_parallel": str(e_parallel),
        "e_perp": str(e_perp),
        "rrmse": str(e_parallel + e_perp),
        "correlation": str(1.0 - min(e_parallel, 1.0) / 2.0),
        "psd_distortion": str(e_perp / 2.0),
        "d_perp_y": str(e_perp),
        "overlap_fraction": "0.1",
        "delta_snr_db": "1.0",
        "projector_distance": "0.2",
        "projector_max_angle_degrees": "10.0",
        "artifact_attenuation": "1.0",
        "clean_interval_preservation": "0.9",
        "p0_eligible": "true",
        "latency_seconds": "0.1",
        "peak_memory_mb": "1.0",
        "function_evaluations": "5",
        "failure_reason": "" if status == "success" else "fixture_failure",
        "fallback_used": "false",
    }


def _stochastic_rows(
    *,
    partition: str,
    record: int,
    method_id: str,
    operator_source: str,
    sampler_candidate: str,
    trust_radius: float,
    aggregate_e_parallel: float,
    aggregate_e_perp: float,
    seed_e_parallel: Iterable[float] | None = None,
) -> list[dict[str, str]]:
    seed_values = list(seed_e_parallel or [aggregate_e_parallel] * len(ALGORITHM_SEEDS))
    if len(seed_values) != len(ALGORITHM_SEEDS):
        raise AssertionError("fixture must name one value per configured seed")
    rows = [
        _row(
            partition=partition,
            record=record,
            method_id=method_id,
            operator_source=operator_source,
            sampler_candidate=sampler_candidate,
            trust_radius=trust_radius,
            seed=seed,
            aggregate=False,
            e_parallel=value,
            e_perp=aggregate_e_perp,
        )
        for seed, value in zip(ALGORITHM_SEEDS, seed_values)
    ]
    rows.append(
        _row(
            partition=partition,
            record=record,
            method_id=method_id,
            operator_source=operator_source,
            sampler_candidate=sampler_candidate,
            trust_radius=trust_radius,
            seed=-1,
            aggregate=True,
            e_parallel=aggregate_e_parallel,
            e_perp=aggregate_e_perp,
        )
    )
    return rows


def _deterministic_row(
    *,
    partition: str,
    record: int,
    method_id: str,
    operator_source: str,
    e_parallel: float,
    e_perp: float,
) -> dict[str, str]:
    return _row(
        partition=partition,
        record=record,
        method_id=method_id,
        operator_source=operator_source,
        sampler_candidate="deterministic",
        trust_radius=0.0,
        seed=-1,
        aggregate=True,
        e_parallel=e_parallel,
        e_perp=e_perp,
    )


def _config(
    tmp_path: Path,
    *,
    development_records: tuple[int, ...] = (31,),
    untouched_records: tuple[int, ...] = (37,),
) -> dict[str, Any]:
    root = tmp_path / "mechanism"
    return {
        "seed": 71,
        "klados": {
            "development_source_records": list(development_records),
            "untouched_source_records": list(untouched_records),
        },
        "sampling": {"seeds": list(ALGORITHM_SEEDS)},
        "development_selection": {
            "primary_metric": "e_parallel",
            "primary_direction": "lower",
            "safety_metric": "e_perp",
            "safety_direction": "lower_or_equal_POP",
            "unit": "source_record",
            "seeds_are_independent_units": False,
            "failure_rate_in_denominator": True,
            "minimum_paired_fraction": 0.75,
            "minimum_primary_improvement_fraction": 0.75,
            "maximum_median_safety_delta": 0.05,
            "maximum_seed_failure_rate_per_record": 0.10,
            "confidence_level": 0.95,
            "paired_bootstrap_replicates": 1000,
        },
        "outputs": {
            "root": str(root),
            "development_metrics": str(root / "development" / "metrics.csv"),
            "frozen_choice": str(root / "development" / "frozen_choice.json"),
            "untouched_metrics": str(root / "untouched" / "metrics.csv"),
            "decision": str(tmp_path / "cgdr_mechanism_decision.md"),
        },
    }


def _write_record_rows(
    config: dict[str, Any], partition: str, record: int, rows: list[dict[str, str]]
) -> Path:
    path = (
        Path(config["outputs"]["root"])
        / partition
        / f"sim{record:02d}"
        / "metrics.csv"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _contract() -> dict[str, Any]:
    return {
        "minimum_paired_fraction": 0.75,
        "minimum_primary_improvement_fraction": 0.75,
        "maximum_median_safety_delta": 0.05,
        "maximum_seed_failure_rate_per_record": 0.10,
        "confidence_level": 0.95,
        "paired_bootstrap_replicates": 10000,
    }


def test_replicate_contract_accepts_five_seeds_and_one_posterior_mean() -> None:
    rows = _stochastic_rows(
        partition="development",
        record=31,
        method_id="oracle_M0_t025",
        operator_source="oracle_projector",
        sampler_candidate="M0",
        trust_radius=0.25,
        aggregate_e_parallel=0.4,
        aggregate_e_perp=0.2,
    )
    _validate_record_replicates(rows, expected_seeds=set(ALGORITHM_SEEDS))
    assert len([row for row in rows if row["aggregate_across_seeds"] == "false"]) == 5
    assert len([row for row in rows if row["aggregate_across_seeds"] == "true"]) == 1


@pytest.mark.parametrize("fault", ["missing", "duplicate"])
def test_replicate_contract_rejects_missing_or_duplicate_seed(fault: str) -> None:
    rows = _stochastic_rows(
        partition="development",
        record=31,
        method_id="oracle_M0_t025",
        operator_source="oracle_projector",
        sampler_candidate="M0",
        trust_radius=0.25,
        aggregate_e_parallel=0.4,
        aggregate_e_perp=0.2,
    )
    if fault == "missing":
        rows = [row for row in rows if row["seed"] != str(ALGORITHM_SEEDS[-1])]
    else:
        duplicate = dict(next(row for row in rows if row["seed"] == str(ALGORITHM_SEEDS[0])))
        rows.append(duplicate)
    with pytest.raises(ValueError, match="exactly the configured algorithmic seeds"):
        _validate_record_replicates(rows, expected_seeds=set(ALGORITHM_SEEDS))


@pytest.mark.parametrize("fault", ["missing", "duplicate"])
def test_replicate_contract_requires_one_posterior_mean_row(fault: str) -> None:
    rows = _stochastic_rows(
        partition="development",
        record=31,
        method_id="oracle_M0_t025",
        operator_source="oracle_projector",
        sampler_candidate="M0",
        trust_radius=0.25,
        aggregate_e_parallel=0.4,
        aggregate_e_perp=0.2,
    )
    aggregate = next(row for row in rows if row["aggregate_across_seeds"] == "true")
    if fault == "missing":
        rows.remove(aggregate)
    else:
        rows.append(dict(aggregate))
    with pytest.raises(ValueError, match="exactly one seed=-1 posterior-mean row"):
        _validate_record_replicates(rows, expected_seeds=set(ALGORITHM_SEEDS))


def test_deterministic_identity_and_qy_are_one_aggregate_row_only() -> None:
    rows = [
        _deterministic_row(
            partition="untouched",
            record=37,
            method_id="corrupted_identity",
            operator_source="corrupted_identity",
            e_parallel=1.0,
            e_perp=0.1,
        ),
        _deterministic_row(
            partition="untouched",
            record=37,
            method_id="oracle_orthogonal_subtraction",
            operator_source="oracle_orthogonal_subtraction",
            e_parallel=0.2,
            e_perp=0.1,
        ),
    ]
    _validate_record_replicates(rows, expected_seeds=set(ALGORITHM_SEEDS))

    invalid = rows + [
        _row(
            partition="untouched",
            record=37,
            method_id="corrupted_identity",
            operator_source="corrupted_identity",
            sampler_candidate="deterministic",
            trust_radius=0.0,
            seed=ALGORITHM_SEEDS[0],
            aggregate=False,
            e_parallel=1.0,
            e_perp=0.1,
        )
    ]
    with pytest.raises(ValueError, match="deterministic method.*seed replicates"):
        _validate_record_replicates(invalid, expected_seeds=set(ALGORITHM_SEEDS))


def test_metrics_come_from_posterior_mean_waveform_row_not_seed_metric_average() -> None:
    rows = _stochastic_rows(
        partition="development",
        record=31,
        method_id="oracle_M0_t025",
        operator_source="oracle_projector",
        sampler_candidate="M0",
        trust_radius=0.25,
        aggregate_e_parallel=0.125,
        aggregate_e_perp=0.25,
        seed_e_parallel=[10.0, 20.0, 30.0, 40.0, 50.0],
    )
    summary = _per_record_method(rows)[(31, "oracle_M0_t025")]
    assert summary["e_parallel"] == pytest.approx(0.125)
    assert summary["e_parallel"] != pytest.approx(30.0)
    assert summary["failure_rate"] == 0.0


def test_fallback_rate_uses_explicit_fallback_field() -> None:
    rows = _stochastic_rows(
        partition="untouched",
        record=37,
        method_id="matching_M0_t025",
        operator_source="matching_p0",
        sampler_candidate="M0",
        trust_radius=0.25,
        aggregate_e_parallel=1.0,
        aggregate_e_perp=0.2,
    )
    for row in rows:
        row["fallback_used"] = "true"
        row["failure_reason"] = "bootstrap_q90"
    summary = _per_record_method(rows)[(37, "matching_M0_t025")]
    assert summary["failure_rate"] == 0.0
    assert summary["fallback_rate"] == 1.0


def test_protocol_consumes_exact_five_seed_and_posterior_mean_contract() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    config = yaml.safe_load(
        (repository_root / "configs/cgdr/mechanism_audit_klados.yaml").read_text(
            encoding="utf-8"
        )
    )
    _validate_protocol(config)
    assert len(set(config["sampling"]["seeds"])) == 5

    too_few = copy.deepcopy(config)
    too_few["sampling"]["seeds"] = too_few["sampling"]["seeds"][:4]
    with pytest.raises(ValueError, match="exactly five unique algorithmic seeds"):
        _validate_protocol(too_few)

    wrong_output = copy.deepcopy(config)
    wrong_output["sampling"]["output_rule"] = "mean_of_seed_metrics"
    with pytest.raises(ValueError, match="average waveforms before metrics"):
        _validate_protocol(wrong_output)


def test_aggregator_rejects_a_nonfive_seed_config_before_reading_results(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config["sampling"]["seeds"] = list(ALGORITHM_SEEDS[:4])
    with pytest.raises(ValueError, match="exactly five unique algorithmic seeds"):
        aggregate_development(config)


def test_unsupported_development_choice_is_explicitly_diagnostic(tmp_path: Path) -> None:
    config = _config(tmp_path)
    rows = _stochastic_rows(
        partition="development",
        record=31,
        method_id="pop_M0_t025",
        operator_source="population_only",
        sampler_candidate="M0",
        trust_radius=0.25,
        aggregate_e_parallel=1.0,
        aggregate_e_perp=0.2,
    )
    rows += _stochastic_rows(
        partition="development",
        record=31,
        method_id="oracle_M0_t025",
        operator_source="oracle_projector",
        sampler_candidate="M0",
        trust_radius=0.25,
        aggregate_e_parallel=1.2,
        aggregate_e_perp=0.2,
    )
    _write_record_rows(config, "development", 31, rows)

    result = aggregate_development(config)
    assert result["status"] == "frozen_diagnostic_no_supported_candidate"
    assert result["oracle_restoration_supported_on_development"] is False
    assert result["next_partition"] == "untouched_fixed_diagnostic_evaluation"
    assert result["formal_G1_status"] == "NOT_RUN_BLOCKED"


@pytest.mark.parametrize(
    ("operator_source", "sampler", "trust_radius", "message"),
    [
        ("population_only", "M0", 0.5, "non-frozen trust radius"),
        ("oracle_projector", "M1", 0.25, "non-frozen sampler candidate"),
    ],
)
def test_untouched_rejects_nonfrozen_sampler_or_trust(
    tmp_path: Path,
    operator_source: str,
    sampler: str,
    trust_radius: float,
    message: str,
) -> None:
    config = _config(tmp_path)
    frozen = {
        "status": "frozen_supported",
        "selected_sampler_candidate": "M0",
        "selected_trust_radius": 0.25,
    }
    frozen_path = Path(config["outputs"]["frozen_choice"])
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    frozen_path.write_text(json.dumps(frozen), encoding="utf-8")
    rows = _stochastic_rows(
        partition="untouched",
        record=37,
        method_id="bad_frozen_tuple",
        operator_source=operator_source,
        sampler_candidate=sampler,
        trust_radius=trust_radius,
        aggregate_e_parallel=0.5,
        aggregate_e_perp=0.2,
    )
    _write_record_rows(config, "untouched", 37, rows)

    with pytest.raises(ValueError, match=message):
        aggregate_untouched_and_decide(config)


def _paired_summaries(primary_deltas: list[float]) -> dict[tuple[int, str], dict[str, float]]:
    summaries: dict[tuple[int, str], dict[str, float]] = {}
    for record, delta in enumerate(primary_deltas, start=1):
        summaries[(record, "reference")] = {
            "e_parallel": 1.0,
            "e_perp": 0.2,
            "failure_rate": 0.0,
        }
        summaries[(record, "candidate")] = {
            "e_parallel": 1.0 + delta,
            "e_perp": 0.2,
            "failure_rate": 0.0,
        }
    return summaries


def test_paired_support_requires_confidence_interval_not_only_point_fraction() -> None:
    records = tuple(range(1, 9))
    supported = _paired_effect(
        _paired_summaries([-0.2] * 8),
        records,
        "candidate",
        "reference",
        contract=_contract(),
        bootstrap_seed=81,
    )
    assert supported["supported"] is True
    assert supported["median_primary_delta_confidence_interval"][1] < 0.0
    assert supported["statistical_unit"] == "source_record"
    assert supported["seed_semantics"] == (
        "posterior_mean_waveform_then_metric_within_source_record"
    )

    uncertain = _paired_effect(
        _paired_summaries([-0.2] * 6 + [0.2] * 2),
        records,
        "candidate",
        "reference",
        contract=_contract(),
        bootstrap_seed=83,
    )
    assert uncertain["median_primary_delta"] < 0.0
    assert uncertain["primary_improvement_fraction"] == pytest.approx(0.75)
    assert uncertain["median_primary_delta_confidence_interval"][1] >= 0.0
    assert uncertain["supported"] is False


def test_absolute_qy_audit_respects_all_four_metric_directions() -> None:
    summaries: dict[tuple[int, str], dict[str, float]] = {}
    for record, offset in ((37, 0.0), (38, 0.2)):
        summaries[(record, "Qy")] = {
            "e_parallel": 1.0 + offset,
            "e_perp": 0.1,
            "rrmse": 0.5,
            "correlation": 0.9,
            "failure_rate": 0.0,
            "fallback_rate": 0.0,
        }
        summaries[(record, "oracle_M2")] = {
            "e_parallel": 2.0 + offset,
            "e_perp": 0.2,
            "rrmse": 0.8,
            "correlation": 0.7,
            "failure_rate": 0.0,
            "fallback_rate": 0.0,
        }

    effect = _absolute_baseline_effect(
        summaries,
        (37, 38),
        "oracle_M2",
        "Qy",
    )

    assert effect["records_paired"] == 2
    assert effect["failed_records"] == 0
    assert effect["metrics"]["e_parallel"]["direction"] == "lower"
    assert effect["metrics"]["e_parallel"]["median_delta"] == pytest.approx(1.0)
    assert effect["metrics"]["e_perp"]["median_delta"] == pytest.approx(0.1)
    assert effect["metrics"]["rrmse"]["median_delta"] == pytest.approx(0.3)
    assert effect["metrics"]["correlation"]["direction"] == "higher"
    assert effect["metrics"]["correlation"]["median_delta"] == pytest.approx(-0.2)
    assert effect["metrics"]["correlation"]["better_or_equal_records"] == 0
    assert effect["metrics"]["correlation"]["worse_records"] == 2
    assert effect["all_metrics_worse_records"] == 2
    assert effect["all_metrics_better_or_equal_records"] == 0
    assert effect["noninferior_or_better_all_metrics"] is False


@pytest.mark.parametrize(
    (
        "geometry",
        "better_pop",
        "noninferior_qy",
        "better_trained_deterministic",
        "preservation",
        "expected",
    ),
    [
        (*values, (
            "C_geometry_not_supported"
            if not values[0]
            else (
                "A_diffusion_supported"
                if all(values[1:])
                else "B_geometry_only"
            )
        ))
        for values in itertools.product((False, True), repeat=5)
    ]
    + [
        # A missing trained U-Net comparison is an explicit missing A requirement.
        (True, True, True, None, True, "B_geometry_only"),
    ],
)
def test_absolute_classifier_truth_table(
    geometry: bool,
    better_pop: bool,
    noninferior_qy: bool,
    better_trained_deterministic: bool | None,
    preservation: bool,
    expected: str,
) -> None:
    result = classify_absolute_mechanism_evidence(
        geometry_supported=geometry,
        iterative_better_than_pop=better_pop,
        iterative_noninferior_or_better_than_qy=noninferior_qy,
        iterative_better_than_trained_deterministic=better_trained_deterministic,
        preservation_all_passed=preservation,
    )

    assert result["classification"] == expected
    assert result["sampler_id_is_evidence"] is False
    if better_trained_deterministic is None:
        assert "iterative_better_than_trained_deterministic_UNet" in result[
            "missing_requirements"
        ]


def test_absolute_classifier_has_no_sampler_id_argument() -> None:
    parameters = inspect.signature(classify_absolute_mechanism_evidence).parameters
    assert "sampler" not in parameters
    assert "sampler_candidate" not in parameters
    assert "method_id" not in parameters


def test_pop_pass_but_qy_fail_is_geometry_only_b() -> None:
    result = classify_absolute_mechanism_evidence(
        geometry_supported=True,
        iterative_better_than_pop=True,
        iterative_noninferior_or_better_than_qy=False,
        iterative_better_than_trained_deterministic=True,
        preservation_all_passed=True,
    )
    assert result["classification"] == "B_geometry_only"


def _write_untouched_decision_fixture(
    config: dict[str, Any],
    *,
    sampler: str,
    oracle_supported: bool,
    geometry_supported: bool,
) -> None:
    frozen = {
        "status": "frozen_supported",
        "selected_sampler_candidate": sampler,
        "selected_trust_radius": 0.25,
    }
    frozen_path = Path(config["outputs"]["frozen_choice"])
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    frozen_path.write_text(json.dumps(frozen), encoding="utf-8")

    record = int(config["klados"]["untouched_source_records"][0])
    rows = [
        _deterministic_row(
            partition="untouched",
            record=record,
            method_id="corrupted_identity",
            operator_source="corrupted_identity",
            e_parallel=1.0,
            e_perp=0.2,
        ),
        _deterministic_row(
            partition="untouched",
            record=record,
            method_id="oracle_orthogonal_subtraction",
            operator_source="oracle_orthogonal_subtraction",
            e_parallel=0.2 if geometry_supported else 1.2,
            e_perp=0.2,
        ),
    ]
    rows += _stochastic_rows(
        partition="untouched",
        record=record,
        method_id=f"population_only_{sampler}",
        operator_source="population_only",
        sampler_candidate=sampler,
        trust_radius=0.25,
        aggregate_e_parallel=1.0,
        aggregate_e_perp=0.2,
    )
    rows += _stochastic_rows(
        partition="untouched",
        record=record,
        method_id=f"oracle_{sampler}",
        operator_source="oracle_projector",
        sampler_candidate=sampler,
        trust_radius=0.25,
        aggregate_e_parallel=0.2 if oracle_supported else 1.2,
        aggregate_e_perp=0.2,
    )
    matching_value = 0.2 if oracle_supported else 0.9
    rows += _stochastic_rows(
        partition="untouched",
        record=record,
        method_id=f"matching_{sampler}",
        operator_source="matching_p0",
        sampler_candidate=sampler,
        trust_radius=0.25,
        aggregate_e_parallel=matching_value,
        aggregate_e_perp=0.2,
    )
    for operator_source, suffix in (
        ("population_projector", "population"),
        ("wrong_source_p0", "wrong"),
        ("shuffled_calibration_p0", "shuffled"),
    ):
        rows += _stochastic_rows(
            partition="untouched",
            record=record,
            method_id=f"{suffix}_{sampler}",
            operator_source=operator_source,
            sampler_candidate=sampler,
            trust_radius=0.25,
            aggregate_e_parallel=0.8,
            aggregate_e_perp=0.2,
        )
    _write_record_rows(config, "untouched", record, rows)


@pytest.mark.parametrize(
    ("sampler", "oracle_supported", "geometry_supported", "expected"),
    [
        ("M0", True, True, "B"),
        ("M5", True, True, "B"),
        ("M0", False, True, "B"),
        ("M0", False, False, "C"),
    ],
)
def test_untouched_absolute_decision_never_uses_sampler_id_as_a_shortcut(
    tmp_path: Path,
    sampler: str,
    oracle_supported: bool,
    geometry_supported: bool,
    expected: str,
) -> None:
    config = _config(tmp_path)
    _write_untouched_decision_fixture(
        config,
        sampler=sampler,
        oracle_supported=oracle_supported,
        geometry_supported=geometry_supported,
    )
    result = aggregate_untouched_and_decide(config)
    assert result["conclusion"] == expected
    assert result["formal_G1_status"] == "NOT_RUN_BLOCKED"
    assert result["conclusion"] != "A"
    if geometry_supported:
        assert result["post_absolute_baseline_audit"] == "B_geometry_only"
        assert result["next_stage"] == "deterministic_or_proximal_personalized_model"
        assert result["trained_deterministic_UNet_effect"] is None


def test_interpretation_rerun_does_not_overwrite_historical_result_summary(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_untouched_decision_fixture(
        config,
        sampler="M2",
        oracle_supported=True,
        geometry_supported=True,
    )
    historical = Path(config["outputs"]["root"]) / "result_summary.json"
    historical.parent.mkdir(parents=True, exist_ok=True)
    historical.write_text('{"conclusion":"A","immutable":true}\n', encoding="utf-8")

    result = aggregate_untouched_and_decide(config)

    assert json.loads(historical.read_text(encoding="utf-8")) == {
        "conclusion": "A",
        "immutable": True,
    }
    interpretation = historical.with_name(
        "interpretation_summary_after_absolute_baseline_audit.json"
    )
    assert interpretation.is_file()
    assert result["original_classifier_result"] == "A_limited"
    assert result["post_absolute_baseline_audit"] == "B_geometry_only"
