"""Preregistered source-record aggregation for the repaired mechanism audit."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np


REQUIRED_METRIC_FIELDS = {
    "partition",
    "source_record",
    "seed",
    "aggregate_across_seeds",
    "method_id",
    "operator_source",
    "sampler_candidate",
    "trust_radius",
    "status",
    "e_parallel",
    "e_perp",
    "rrmse",
    "correlation",
    "psd_distortion",
    "d_perp_y",
    "overlap_fraction",
    "delta_snr_db",
    "projector_distance",
    "projector_max_angle_degrees",
    "artifact_attenuation",
    "clean_interval_preservation",
    "p0_eligible",
    "latency_seconds",
    "peak_memory_mb",
    "function_evaluations",
    "failure_reason",
    "fallback_used",
}


DETERMINISTIC_OPERATOR_SOURCES = {
    "corrupted_identity",
    "oracle_orthogonal_subtraction",
}
VALID_OUTPUT_STATUSES = {"success", "fallback_POP"}


def _as_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid boolean field in mechanism metrics: {value!r}")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _validate_record_replicates(
    rows: list[dict[str, str]], *, expected_seeds: set[int]
) -> None:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["method_id"]].append(row)
    for method_id, values in grouped.items():
        aggregates = [row for row in values if _as_bool(row["aggregate_across_seeds"])]
        if len(aggregates) != 1 or int(aggregates[0]["seed"]) != -1:
            raise ValueError(
                f"{method_id} must have exactly one seed=-1 posterior-mean row"
            )
        operator_source = aggregates[0]["operator_source"]
        replicates = [row for row in values if not _as_bool(row["aggregate_across_seeds"])]
        if operator_source in DETERMINISTIC_OPERATOR_SOURCES:
            if replicates or len(values) != 1:
                raise ValueError(f"deterministic method {method_id} has seed replicates")
            continue
        actual_seeds = {int(row["seed"]) for row in replicates}
        if actual_seeds != expected_seeds or len(replicates) != len(expected_seeds):
            raise ValueError(
                f"{method_id} must preserve exactly the configured algorithmic seeds; "
                f"expected={sorted(expected_seeds)} actual={sorted(actual_seeds)}"
            )


def _read_record_rows(
    path: Path,
    partition: str,
    record_id: int,
    *,
    expected_seeds: set[int],
) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing {partition} result for sim{record_id:02d}: {path}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not REQUIRED_METRIC_FIELDS.issubset(reader.fieldnames):
            raise ValueError(f"mechanism metrics schema mismatch: {path}")
        rows = list(reader)
    if not rows or {row["partition"] for row in rows} != {partition}:
        raise ValueError(f"mechanism metrics partition mismatch: {path}")
    if {int(row["source_record"]) for row in rows} != {record_id}:
        raise ValueError(f"mechanism metrics source-record mismatch: {path}")
    _validate_record_replicates(rows, expected_seeds=expected_seeds)
    return rows


def _collect(config: dict[str, Any], partition: str) -> list[dict[str, str]]:
    klados = config["klados"]
    if partition == "development":
        records = [int(value) for value in klados["development_source_records"]]
    elif partition == "untouched":
        records = [int(value) for value in klados["untouched_source_records"]]
    else:
        raise ValueError(f"unknown mechanism partition: {partition}")
    root = Path(config["outputs"]["root"]) / partition
    configured_seeds = [int(value) for value in config["sampling"]["seeds"]]
    expected_seeds = set(configured_seeds)
    if len(configured_seeds) != 5 or len(expected_seeds) != 5:
        raise ValueError(
            "mechanism audit requires exactly five unique algorithmic seeds"
        )
    rows: list[dict[str, str]] = []
    for record_id in records:
        rows.extend(
            _read_record_rows(
                root / f"sim{record_id:02d}" / "metrics.csv",
                partition,
                record_id,
                expected_seeds=expected_seeds,
            )
        )
    if {int(row["source_record"]) for row in rows} != set(records):
        raise AssertionError("mechanism aggregation lost a preregistered source record")
    return rows


def _numeric(row: dict[str, str], field: str) -> float:
    value = float(row[field])
    if not np.isfinite(value):
        raise ValueError(f"non-finite {field} in successful mechanism row")
    return value


def _per_record_method(rows: Iterable[dict[str, str]]) -> dict[tuple[int, str], dict[str, float]]:
    grouped: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    seen_attempts: set[tuple[int, str, int, bool]] = set()
    for row in rows:
        record_id = int(row["source_record"])
        method_id = row["method_id"]
        aggregate = _as_bool(row["aggregate_across_seeds"])
        attempt = (record_id, method_id, int(row["seed"]), aggregate)
        if attempt in seen_attempts:
            raise ValueError(
                "duplicate mechanism attempt for "
                f"sim{record_id:02d}/{method_id}/seed={attempt[2]}/aggregate={aggregate}"
            )
        seen_attempts.add(attempt)
        grouped[(record_id, method_id)].append(row)
    output: dict[tuple[int, str], dict[str, float]] = {}
    for key, values in grouped.items():
        aggregate_rows = [row for row in values if _as_bool(row["aggregate_across_seeds"])]
        if len(aggregate_rows) != 1:
            raise ValueError(f"{key} has no unique posterior-mean output row")
        aggregate = aggregate_rows[0]
        seed_rows = [row for row in values if not _as_bool(row["aggregate_across_seeds"])]
        if seed_rows:
            successful_seeds = [
                row for row in seed_rows if row["status"] in VALID_OUTPUT_STATUSES
            ]
            fallback_seeds = [
                row for row in seed_rows if _as_bool(row["fallback_used"])
            ]
            failure_rate = 1.0 - len(successful_seeds) / len(seed_rows)
            fallback_rate = len(fallback_seeds) / len(seed_rows)
            attempts = len(seed_rows)
            successes = len(successful_seeds)
        else:
            failure_rate = (
                0.0 if aggregate["status"] in VALID_OUTPUT_STATUSES else 1.0
            )
            fallback_rate = float(_as_bool(aggregate["fallback_used"]))
            attempts = 1
            successes = int(aggregate["status"] in VALID_OUTPUT_STATUSES)
        summary: dict[str, float] = {
            "failure_rate": failure_rate,
            "fallback_rate": fallback_rate,
            "attempts": float(attempts),
            "successes": float(successes),
        }
        for field in ("e_parallel", "e_perp", "rrmse", "correlation", "psd_distortion"):
            summary[field] = (
                _numeric(aggregate, field)
                if aggregate["status"] in VALID_OUTPUT_STATUSES
                else float("nan")
            )
        output[key] = summary
    return output


def _method_metadata(rows: Iterable[dict[str, str]]) -> dict[str, dict[str, Any]]:
    """Require each method ID to denote one operator/sampler/trust tuple."""

    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        trust_radius = float(row["trust_radius"])
        deterministic = row["operator_source"] in DETERMINISTIC_OPERATOR_SOURCES
        if not np.isfinite(trust_radius) or (not deterministic and trust_radius <= 0.0):
            raise ValueError(f"invalid trust radius for method {row['method_id']!r}")
        current = {
            "sampler_candidate": row["sampler_candidate"],
            "trust_radius": trust_radius,
            "operator_source": row["operator_source"],
        }
        previous = output.setdefault(row["method_id"], current)
        if previous != current:
            raise ValueError(
                f"method ID {row['method_id']!r} maps to inconsistent metadata"
            )
    return output


def _bootstrap_median_interval(
    values: list[float], *, confidence: float, replicates: int, seed: int
) -> Optional[list[float]]:
    if not values:
        return None
    if not 0.0 < confidence < 1.0 or replicates < 1000:
        raise ValueError("paired bootstrap confidence/replicates are not defensible")
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, array.size, size=(replicates, array.size))
    draws = np.median(array[indices], axis=1)
    tail = (1.0 - confidence) / 2.0
    return [float(np.quantile(draws, tail)), float(np.quantile(draws, 1.0 - tail))]


def _selection_contract(config: dict[str, Any]) -> dict[str, Any]:
    raw = config["development_selection"]
    expected_labels = {
        "primary_metric": "e_parallel",
        "primary_direction": "lower",
        "safety_metric": "e_perp",
        "safety_direction": "lower_or_equal_POP",
        "unit": "source_record",
        "seeds_are_independent_units": False,
        "failure_rate_in_denominator": True,
    }
    for name, expected in expected_labels.items():
        if raw.get(name) != expected:
            raise ValueError(
                f"development selection field {name} must remain {expected!r}"
            )
    contract = {
        "minimum_paired_fraction": float(raw["minimum_paired_fraction"]),
        "minimum_primary_improvement_fraction": float(
            raw["minimum_primary_improvement_fraction"]
        ),
        "maximum_median_safety_delta": float(raw["maximum_median_safety_delta"]),
        "maximum_seed_failure_rate_per_record": float(
            raw["maximum_seed_failure_rate_per_record"]
        ),
        "confidence_level": float(raw["confidence_level"]),
        "paired_bootstrap_replicates": int(raw["paired_bootstrap_replicates"]),
    }
    if not 0.0 < contract["minimum_paired_fraction"] <= 1.0:
        raise ValueError("minimum_paired_fraction must lie in (0,1]")
    if not 0.0 < contract["minimum_primary_improvement_fraction"] <= 1.0:
        raise ValueError("minimum_primary_improvement_fraction must lie in (0,1]")
    if not 0.0 <= contract["maximum_seed_failure_rate_per_record"] < 1.0:
        raise ValueError("maximum seed failure rate must lie in [0,1)")
    if contract["maximum_median_safety_delta"] < 0.0:
        raise ValueError("safety delta margin must be non-negative")
    return contract


def _paired_effect(
    summaries: dict[tuple[int, str], dict[str, float]],
    records: Iterable[int],
    method: str,
    reference: str,
    *,
    contract: dict[str, Any],
    bootstrap_seed: int,
) -> dict[str, Any]:
    primary: list[float] = []
    safety: list[float] = []
    failed_records = 0
    for record_id in records:
        candidate = summaries.get((int(record_id), method))
        baseline = summaries.get((int(record_id), reference))
        if (
            candidate is None
            or baseline is None
            or not np.isfinite(candidate["e_parallel"])
            or not np.isfinite(baseline["e_parallel"])
            or not np.isfinite(candidate["e_perp"])
            or not np.isfinite(baseline["e_perp"])
            or candidate["failure_rate"]
            > contract["maximum_seed_failure_rate_per_record"]
            or baseline["failure_rate"]
            > contract["maximum_seed_failure_rate_per_record"]
            or candidate.get("fallback_rate", 0.0) > 0.0
            or baseline.get("fallback_rate", 0.0) > 0.0
        ):
            failed_records += 1
            continue
        primary.append(candidate["e_parallel"] - baseline["e_parallel"])
        safety.append(candidate["e_perp"] - baseline["e_perp"])
    count = len(primary)
    total = count + failed_records
    primary_interval = _bootstrap_median_interval(
        primary,
        confidence=contract["confidence_level"],
        replicates=contract["paired_bootstrap_replicates"],
        seed=bootstrap_seed,
    )
    safety_interval = _bootstrap_median_interval(
        safety,
        confidence=contract["confidence_level"],
        replicates=contract["paired_bootstrap_replicates"],
        seed=bootstrap_seed + 1,
    )
    safety_margin = contract["maximum_median_safety_delta"]
    minimum_paired = int(np.ceil(contract["minimum_paired_fraction"] * total))
    return {
        "method": method,
        "reference": reference,
        "records_total": total,
        "records_paired": count,
        "failed_records": failed_records,
        "median_primary_delta": float(np.median(primary)) if primary else None,
        "median_safety_delta": float(np.median(safety)) if safety else None,
        "median_primary_delta_confidence_interval": primary_interval,
        "median_safety_delta_confidence_interval": safety_interval,
        "confidence_level": contract["confidence_level"],
        "primary_improvement_fraction": (
            float(np.mean(np.asarray(primary) < 0.0)) if primary else 0.0
        ),
        "safety_preservation_fraction": (
            float(np.mean(np.asarray(safety) <= safety_margin)) if safety else 0.0
        ),
        "supported": bool(
            total > 0
            and count >= minimum_paired
            and float(np.median(primary)) < 0.0
            and float(np.mean(np.asarray(primary) < 0.0))
            >= contract["minimum_primary_improvement_fraction"]
            and float(np.median(safety)) <= safety_margin
            and float(np.mean(np.asarray(safety) <= safety_margin))
            >= contract["minimum_primary_improvement_fraction"]
            and primary_interval is not None
            and primary_interval[1] < 0.0
            and safety_interval is not None
            and safety_interval[1] <= safety_margin
        ) if primary else False,
        "safety_margin": safety_margin,
        "seed_semantics": "posterior_mean_waveform_then_metric_within_source_record",
        "statistical_unit": "source_record",
    }


def aggregate_development(config: dict[str, Any]) -> dict[str, Any]:
    """Aggregate every development record and freeze one sampler/trust radius."""

    rows = _collect(config, "development")
    metrics_path = Path(config["outputs"]["development_metrics"])
    _write_csv(metrics_path, rows)
    records = [int(value) for value in config["klados"]["development_source_records"]]
    contract = _selection_contract(config)
    summaries = _per_record_method(rows)
    method_metadata = _method_metadata(rows)
    oracle_methods = sorted(
        method
        for method, metadata in method_metadata.items()
        if metadata["operator_source"] == "oracle_projector"
        and metadata["sampler_candidate"].startswith("M")
    )
    pop_by_sampler_and_trust: dict[tuple[str, float], str] = {}
    for method, metadata in method_metadata.items():
        if metadata["operator_source"] == "population_only":
            trust = float(metadata["trust_radius"])
            key = (str(metadata["sampler_candidate"]), trust)
            if key in pop_by_sampler_and_trust:
                raise ValueError(
                    f"duplicate development POP method for sampler/trust {key}"
                )
            pop_by_sampler_and_trust[key] = method
    effects = []
    for method in oracle_methods:
        trust = float(method_metadata[method]["trust_radius"])
        pop_key = (str(method_metadata[method]["sampler_candidate"]), trust)
        if pop_key not in pop_by_sampler_and_trust:
            raise ValueError(
                f"oracle candidate has no sampler- and trust-matched POP: {pop_key}"
            )
        effects.append(
            _paired_effect(
                summaries,
                records,
                method,
                pop_by_sampler_and_trust[pop_key],
                contract=contract,
                bootstrap_seed=int(config["seed"]) + len(effects) * 2,
            )
        )
    evaluable = [
        effect
        for effect in effects
        if effect["records_paired"] > 0
        and effect["median_primary_delta"] is not None
        and effect["median_safety_delta"] is not None
    ]
    if not evaluable:
        raise RuntimeError(
            "development audit has no evaluable oracle/POP source-record pair"
        )
    supported = [effect for effect in evaluable if effect["supported"]]
    candidates = supported if supported else evaluable
    choice = min(
        candidates,
        key=lambda item: (
            item["median_primary_delta"] is None,
            float("inf") if item["median_primary_delta"] is None else item["median_primary_delta"],
            float("inf") if item["median_safety_delta"] is None else item["median_safety_delta"],
            item["method"],
        ),
    )
    metadata = method_metadata[str(choice["method"])]
    freeze_supported = bool(choice["supported"])
    result = {
        "status": (
            "frozen_supported"
            if freeze_supported
            else "frozen_diagnostic_no_supported_candidate"
        ),
        "development_records": records,
        "records_are_participants": False,
        "selection_criteria": contract,
        "selected_method_id": choice["method"],
        "selected_sampler_candidate": metadata["sampler_candidate"],
        "selected_trust_radius": metadata["trust_radius"],
        "oracle_restoration_supported_on_development": freeze_supported,
        "selected_effect": choice,
        "all_oracle_candidate_effects": effects,
        "next_partition": "untouched_fixed_diagnostic_evaluation",
        "formal_G1_status": "NOT_RUN_BLOCKED",
    }
    destination = Path(config["outputs"]["frozen_choice"])
    _write_json(destination, result)
    return result


def _method_for(
    rows: list[dict[str, str]],
    operator_source: str,
    sampler_candidate: str,
    *,
    trust_radius: Optional[float] = None,
) -> str:
    matches = {
        row["method_id"]
        for row in rows
        if row["operator_source"] == operator_source
        and row["sampler_candidate"] == sampler_candidate
        and (
            trust_radius is None
            or np.isclose(
                float(row["trust_radius"]),
                float(trust_radius),
                rtol=0.0,
                atol=1.0e-12,
            )
        )
    }
    if len(matches) != 1:
        raise ValueError(
            f"expected one {operator_source}/{sampler_candidate}/"
            f"trust={trust_radius} method, found {sorted(matches)}"
        )
    return next(iter(matches))


def aggregate_untouched_and_decide(config: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen choice once and emit the mandatory A/B/C conclusion."""

    frozen = json.loads(Path(config["outputs"]["frozen_choice"]).read_text(encoding="utf-8"))
    if frozen.get("status") not in {
        "frozen_supported",
        "frozen_diagnostic_no_supported_candidate",
    }:
        raise ValueError("development sampler choice is not frozen")
    rows = _collect(config, "untouched")
    method_metadata = _method_metadata(rows)
    metrics_path = Path(config["outputs"]["untouched_metrics"])
    _write_csv(metrics_path, rows)
    records = [int(value) for value in config["klados"]["untouched_source_records"]]
    summaries = _per_record_method(rows)
    sampler = str(frozen["selected_sampler_candidate"])
    trust_radius = float(frozen["selected_trust_radius"])
    for method_id, metadata in method_metadata.items():
        if metadata["operator_source"] in DETERMINISTIC_OPERATOR_SOURCES:
            continue
        if not np.isclose(
            float(metadata["trust_radius"]), trust_radius, rtol=0.0, atol=1.0e-12
        ):
            raise ValueError(
                f"untouched method {method_id} used a non-frozen trust radius"
            )
        expected_sampler = sampler
        if metadata["sampler_candidate"] != expected_sampler:
            raise ValueError(
                f"untouched method {method_id} used a non-frozen sampler candidate"
            )
    contract = _selection_contract(config)
    oracle_method = _method_for(
        rows, "oracle_projector", sampler, trust_radius=trust_radius
    )
    matching_method = _method_for(
        rows, "matching_p0", sampler, trust_radius=trust_radius
    )
    population_method = _method_for(
        rows, "population_projector", sampler, trust_radius=trust_radius
    )
    wrong_method = _method_for(
        rows, "wrong_source_p0", sampler, trust_radius=trust_radius
    )
    shuffled_method = _method_for(
        rows, "shuffled_calibration_p0", sampler, trust_radius=trust_radius
    )
    geometry = _paired_effect(
        summaries,
        records,
        "oracle_orthogonal_subtraction",
        "corrupted_identity",
        contract=contract,
        bootstrap_seed=int(config["seed"]) + 1000,
    )
    pop_method = _method_for(
        rows, "population_only", sampler, trust_radius=trust_radius
    )
    oracle_restoration = _paired_effect(
        summaries,
        records,
        oracle_method,
        pop_method,
        contract=contract,
        bootstrap_seed=int(config["seed"]) + 1100,
    )
    matching_controls = {
        name: _paired_effect(
            summaries,
            records,
            matching_method,
            reference,
            contract=contract,
            bootstrap_seed=int(config["seed"]) + 1200 + index * 2,
        )
        for index, (name, reference) in enumerate(
            (
            ("population", population_method),
            ("wrong", wrong_method),
            ("shuffled", shuffled_method),
            )
        )
    }
    specificity = all(item["supported"] for item in matching_controls.values())
    iterative_sampler = sampler in {"M0", "M1", "M2", "M3", "M4"}
    if oracle_restoration["supported"] and iterative_sampler:
        conclusion = "A"
        statement = "corrected diffusion mechanism supported at source-record level"
        next_stage = (
            "Eye-BCI_outer_folds"
            if specificity
            else "single_diagnostic_operator_repair_before_Eye_BCI"
        )
    elif geometry["supported"] or oracle_restoration["supported"]:
        conclusion = "B"
        statement = (
            "geometry useful at source-record level but the diffusion sampler has no "
            "demonstrated gain; use a deterministic/proximal personalized model"
        )
        next_stage = "deterministic_or_proximal_personalized_model"
    else:
        conclusion = "C"
        statement = "even oracle geometry has no demonstrated source-record-level gain"
        next_stage = "stop_CGDR_mainline"
    result = {
        "status": "completed",
        "conclusion": conclusion,
        "statement": statement,
        "next_stage": next_stage,
        "untouched_records": records,
        "records_are_participants": False,
        "selected_sampler_candidate": sampler,
        "selected_trust_radius": frozen["selected_trust_radius"],
        "development_freeze_status": frozen["status"],
        "selection_contract": contract,
        "geometry_effect": geometry,
        "oracle_restoration_effect": oracle_restoration,
        "matching_specificity_supported": specificity,
        "matching_control_effects": matching_controls,
        "formal_G1_status": "NOT_RUN_BLOCKED",
        "gate_boundary": "mechanism audit is not participant-level formal G1",
        "legacy_result_statement": (
            "the single-source-record exploratory effect-direction check failed; "
            "formal G1 was not executed."
        ),
    }
    root = Path(config["outputs"]["root"])
    summary = root / "result_summary.json"
    _write_json(summary, result)
    decision = Path(config["outputs"]["decision"])
    decision.parent.mkdir(parents=True, exist_ok=True)
    decision.write_text(
        "# CGDR repaired mechanism decision\n\n"
        f"Conclusion **{conclusion}**: {statement}.\n\n"
        f"Next stage: `{next_stage}`. Source records, not participants, are the "
        "statistical units. The posterior mean waveform across algorithmic seeds was "
        "formed before metrics. Formal G1 remains NOT RUN/BLOCKED.\n\n"
        "Historical boundary: the single-source-record exploratory effect-direction "
        "check failed; formal G1 was not executed.\n",
        encoding="utf-8",
    )
    return result
