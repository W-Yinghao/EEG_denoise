"""Exploratory Klados development-only calibration-duration diagnostics."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

from eeg_cgdr.data.klados import load_klados_records
from eeg_cgdr.data.mechanism import (
    KLADOS_DEVELOPMENT_RECORDS,
    fit_channel_normalizer,
    prepare_mechanism_record,
)
from eeg_cgdr.experiments.mechanism_audit import _loader_config, _p0_config
from eeg_cgdr.operators import fit_p0


PROTOCOL = "klados_development_diagnostics_v1"


def _relative_norm(numerator: np.ndarray, denominator: np.ndarray) -> float:
    scale = float(np.linalg.norm(denominator))
    if not np.isfinite(scale) or scale <= 1.0e-12:
        raise ValueError("diagnostic metric denominator is zero")
    return float(np.linalg.norm(numerator) / scale)


def _channel_correlation(restored: np.ndarray, clean: np.ndarray) -> float:
    fisher: list[float] = []
    for index in range(clean.shape[0]):
        left = restored[index] - restored[index].mean()
        right = clean[index] - clean[index].mean()
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator <= 1.0e-12:
            continue
        value = float(np.dot(left, right) / denominator)
        fisher.append(float(np.arctanh(np.clip(value, -0.999999, 0.999999))))
    if not fisher:
        raise ValueError("diagnostic correlation is undefined")
    return float(np.tanh(np.mean(fisher)))


def _oracle_projector(observed: np.ndarray, clean: np.ndarray, rank: int) -> np.ndarray:
    artifact = np.asarray(observed - clean, dtype=np.float64)
    basis, singular_values, _ = np.linalg.svd(artifact, full_matrices=False)
    if rank < 1 or rank > basis.shape[1] or singular_values[rank - 1] <= 0.0:
        raise ValueError("query artifact cannot define the requested oracle rank")
    retained = basis[:, :rank]
    return retained @ retained.T


def _projector_geometry(estimated: np.ndarray, oracle: np.ndarray) -> dict[str, float]:
    left_values, left_vectors = np.linalg.eigh(0.5 * (estimated + estimated.T))
    right_values, right_vectors = np.linalg.eigh(0.5 * (oracle + oracle.T))
    left = left_vectors[:, left_values > 0.5]
    right = right_vectors[:, right_values > 0.5]
    singular = np.linalg.svd(right.T @ left, compute_uv=False)
    angles = np.arccos(np.clip(singular, 0.0, 1.0))
    if left.shape[1] != right.shape[1]:
        angles = np.concatenate(
            [angles, np.full(abs(left.shape[1] - right.shape[1]), np.pi / 2.0)]
        )
    return {
        "projector_distance": float(
            np.linalg.norm(estimated - oracle, ord="fro")
            / np.sqrt(max(left.shape[1] + right.shape[1], 1))
        ),
        "projector_max_angle_degrees": float(np.rad2deg(angles).max()),
        "projector_overlap_fraction": float(
            np.clip(np.trace(oracle @ estimated) / max(right.shape[1], 1), 0.0, 1.0)
        ),
    }


def _qy_metrics(
    observed: np.ndarray,
    clean: np.ndarray,
    *,
    method_projector: np.ndarray,
    oracle_projector: np.ndarray,
) -> dict[str, float]:
    identity = np.eye(method_projector.shape[0], dtype=np.float64)
    restored = (identity - method_projector) @ observed
    error = restored - clean
    oracle_q = identity - oracle_projector
    return {
        "e_parallel": _relative_norm(oracle_projector @ error, oracle_projector @ clean),
        "e_perp": _relative_norm(oracle_q @ error, oracle_q @ clean),
        "rrmse": _relative_norm(error, clean),
        "correlation": _channel_correlation(restored, clean),
    }


def summarize_duration_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return descriptive duration summaries without treating records as participants."""

    grouped: dict[float, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(float(row["calibration_seconds"]), []).append(row)
    summaries: list[dict[str, Any]] = []
    for duration in sorted(grouped):
        values = grouped[duration]
        available = [row for row in values if row["status"] != "N/A"]
        eligible = [row for row in available if bool(row["p0_eligible"])]

        def median(field: str, source: list[dict[str, Any]]) -> float | None:
            numbers = [
                float(row[field])
                for row in source
                if row.get(field) not in (None, "")
                and math.isfinite(float(row[field]))
            ]
            return float(np.median(numbers)) if numbers else None

        summaries.append(
            {
                "calibration_seconds": duration,
                "records_total": len(values),
                "records_available": len(available),
                "records_NA": len(values) - len(available),
                "p0_eligible_records": len(eligible),
                "eligibility_fraction_of_available": (
                    len(eligible) / len(available) if available else None
                ),
                "median_bootstrap_median_distance": median(
                    "bootstrap_median_distance", available
                ),
                "median_bootstrap_q90_distance": median(
                    "bootstrap_q90_distance", available
                ),
                "median_projector_max_angle_degrees": median(
                    "projector_max_angle_degrees", eligible
                ),
                "median_matching_minus_population_e_parallel": median(
                    "matching_minus_population_e_parallel", eligible
                ),
                "median_matching_minus_population_e_perp": median(
                    "matching_minus_population_e_perp", eligible
                ),
                "median_matching_minus_population_rrmse": median(
                    "matching_minus_population_rrmse", eligible
                ),
                "median_matching_minus_population_correlation": median(
                    "matching_minus_population_correlation", eligible
                ),
            }
        )
    return summaries


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_calibration_duration_axis(
    config: dict[str, Any], *, run_dir: Path
) -> dict[str, Any]:
    """Run the development-only 5/10/20/30-second operator diagnostic."""

    if config.get("diagnostic_protocol") != PROTOCOL:
        raise ValueError(f"expected diagnostic_protocol={PROTOCOL}")
    records_requested = tuple(int(value) for value in config["development_records"])
    if records_requested != KLADOS_DEVELOPMENT_RECORDS:
        raise ValueError("Klados development records differ from the frozen partition")
    durations = tuple(float(value) for value in config["calibration_durations_seconds"])
    if durations != (5.0, 10.0, 20.0, 30.0):
        raise ValueError("calibration-duration axis must be exactly 5/10/20/30 seconds")
    base_path = Path(config["base_config"])
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if base.get("execution_scope") != "development_diagnostics_only":
        raise ValueError("calibration-duration audit requires a development-only base")
    common_query_start = float(config.get("common_query_start_seconds", -1.0))
    if not math.isclose(common_query_start, 31.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("calibration durations must share the frozen 31-second query start")
    records = load_klados_records(_loader_config(base))
    normalizer = fit_channel_normalizer(records)
    from eeg_cgdr.experiments.mechanism_training import load_population_projector

    population_projector = np.asarray(
        load_population_projector(base).projector, dtype=np.float64
    )
    target_rank = int(base["p0"]["target_rank"])
    rows: list[dict[str, Any]] = []
    for duration in durations:
        for record_id in records_requested:
            row: dict[str, Any] = {
                "source_record": record_id,
                "records_are_participants": False,
                "partition": "development",
                "calibration_seconds": duration,
                "guard_seconds": float(base["klados"]["guard_seconds"]),
                "status": "N/A",
                "failure_reason": "",
                "p0_eligible": False,
                "query_clean_target_used_for_diagnostic_only": True,
                "confirmatory_evidence": False,
            }
            try:
                prepared = prepare_mechanism_record(
                    records[record_id - 1],
                    normalizer,
                    source_rate=int(base["klados"]["source_sampling_rate"]),
                    target_rate=int(base["preprocessing"]["target_sampling_rate"]),
                    window_samples=int(base["preprocessing"]["window_samples"]),
                    calibration_seconds=duration,
                    guard_seconds=float(base["klados"]["guard_seconds"]),
                    query_start_seconds=common_query_start,
                )
            except ValueError as exc:
                row["failure_reason"] = str(exc)
                rows.append(row)
                continue
            outcome = fit_p0(
                prepared.calibration,
                _p0_config(base),
                movement_threshold=float(base["p0"]["movement_threshold"]),
            )
            diagnostics = (
                outcome.transfer.diagnostics
                if outcome.transfer is not None
                else outcome.diagnostics
            )
            row.update(
                {
                    "status": "eligible" if outcome.transfer is not None else "fallback_POP",
                    "failure_reason": ";".join(outcome.reasons),
                    "p0_eligible": outcome.transfer is not None,
                    "reference_condition": diagnostics.get("reference_condition", ""),
                    "movement_coverage": diagnostics.get("movement_coverage", ""),
                    "singular_ratio": diagnostics.get("singular_ratio", ""),
                    "bootstrap_success_rate": diagnostics.get("bootstrap_success_rate", ""),
                    "bootstrap_median_distance": diagnostics.get(
                        "bootstrap_median_projector_distance", ""
                    ),
                    "bootstrap_q90_distance": diagnostics.get(
                        "bootstrap_q90_projector_distance", ""
                    ),
                    "query_seconds": (
                        prepared.query_end_seconds - prepared.query_start_seconds
                    ),
                    "query_start_seconds": prepared.query_start_seconds,
                }
            )
            oracle = _oracle_projector(
                prepared.observed_continuous,
                prepared.clean_continuous,
                target_rank,
            )
            population_metrics = _qy_metrics(
                prepared.observed_continuous,
                prepared.clean_continuous,
                method_projector=population_projector,
                oracle_projector=oracle,
            )
            for metric in ("e_parallel", "e_perp", "rrmse", "correlation"):
                row[f"population_{metric}"] = population_metrics[metric]
            if outcome.transfer is None:
                rows.append(row)
                continue
            matching_projector = np.asarray(outcome.transfer.projector, dtype=np.float64)
            row.update(_projector_geometry(matching_projector, oracle))
            matching_metrics = _qy_metrics(
                prepared.observed_continuous,
                prepared.clean_continuous,
                method_projector=matching_projector,
                oracle_projector=oracle,
            )
            for metric in ("e_parallel", "e_perp", "rrmse", "correlation"):
                row[f"matching_{metric}"] = matching_metrics[metric]
                row[f"matching_minus_population_{metric}"] = (
                    matching_metrics[metric] - population_metrics[metric]
                )
            rows.append(row)

    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    row_path = output_root / "calibration_duration_metrics.csv"
    summary_path = output_root / "calibration_duration_summary.json"
    _write_csv(row_path, rows)
    summary = {
        "status": "completed_exploratory_development_only",
        "diagnostic_protocol": PROTOCOL,
        "development_records": list(records_requested),
        "records_are_participants": False,
        "evaluation_records_reused": False,
        "common_query_start_seconds": common_query_start,
        "formal_gate_evidence": False,
        "duration_summaries": summarize_duration_rows(rows),
        "metrics": str(row_path),
        "interpretation": (
            "distinguishes short-support information limits from P0 family failure; "
            "does not reopen or confirm G1/G2"
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "klados_duration_status.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
