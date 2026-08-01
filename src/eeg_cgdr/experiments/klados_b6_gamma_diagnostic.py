"""Development-only Klados B6 gamma diagnostic.

This module is deliberately not a confirmatory experiment.  Klados v4 exposes
aligned paired *source records*, but it does not expose a reliable participant
mapping.  The only records admitted here are the frozen development records
``sim31--sim36, sim44, sim45``.  The 16 records previously used for diagnostic
evaluation are rejected at the boundary and are never reused for gamma
selection; they are not described as untouched.

The module supplies three outputs for every preregistered gamma:

``deterministic_hard_Qy``
    The orthogonal subtraction ``(I - Pi_gamma) y``.

``deterministic_quadratic_soft_proximal``
    ``(I - Pi_gamma)y + tau Pi_gamma y``.  For ``0 < tau <= 1`` this is the
    closed-form minimizer of

    ``0.5 ||x-y||^2 + 0.5 lambda ||Pi_gamma x||^2``

    with ``lambda = tau**-1 - 1``.  ``tau=0`` is the hard-Q limit.  Tau is
    frozen by configuration and is not selected from paired query targets.

``M2_final_hard_Q_consistency``
    The existing repaired multichannel prior/sampler followed by final hard
    Q-consistency.  Its record-level entry point calls the B6 runner with the
    actual frozen split provenance; no compatibility or fit-scope strings are
    accepted from the caller.

Paired clean queries are used only to score development outputs and choose one
global gamma.  They are never passed to B6, P0, the prior, or the sampler.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PROTOCOL = "klados_b6_gamma_development_v1"
DATASET_ID = "klados_bamidis_v4"
DEVELOPMENT_SOURCE_RECORDS = (31, 32, 33, 34, 35, 36, 44, 45)
PREVIOUSLY_USED_EVALUATION_SOURCE_RECORDS = (
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
)
GAMMA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
METHOD_HARD_QY = "deterministic_hard_Qy"
METHOD_SOFT_PROXIMAL = "deterministic_quadratic_soft_proximal"
METHOD_M2 = "M2_final_hard_Q_consistency"
METHODS = (METHOD_HARD_QY, METHOD_SOFT_PROXIMAL, METHOD_M2)
CORE_METRICS = (
    "e_parallel",
    "e_perp",
    "d_perp_y",
    "rrmse",
    "correlation",
    "psd_distortion",
    "artifact_attenuation",
    "clean_interval_preservation",
)


@dataclass(frozen=True)
class GammaDiagnosticContract:
    """Validated immutable choices for the development diagnostic."""

    base_config: Path
    backup_config: Path
    development_records: tuple[int, ...]
    gamma_grid: tuple[float, ...]
    soft_tau: float
    m2_trust_radius: float
    m2_seeds: tuple[int, ...]
    selection_method: str
    maximum_median_preservation_delta: float
    minimum_valid_record_fraction: float
    minimum_operator_eligibility_fraction: float
    metric_tie_tolerance: float
    tie_break_order: tuple[float, ...]
    output_root: Path


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"missing mapping: {key}")
    return value


def _exact_bool(parent: Mapping[str, Any], key: str, expected: bool) -> None:
    if parent.get(key) is not expected:
        raise ValueError(f"{key} must be {expected}")


def _finite_unit_interval(value: object, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and lie in [0,1]")
    return number


def validate_diagnostic_config(
    config: Mapping[str, Any],
) -> GammaDiagnosticContract:
    """Validate claim boundaries and every choice used for gamma selection."""

    if config.get("diagnostic_protocol") != PROTOCOL:
        raise ValueError(f"diagnostic_protocol must be {PROTOCOL}")
    if int(config.get("harness_level", -1)) != 1:
        raise ValueError("the B6 development diagnostic requires HARNESS_LEVEL=1")

    records = tuple(int(value) for value in config.get("development_source_records", ()))
    if records != DEVELOPMENT_SOURCE_RECORDS:
        raise ValueError("development source records must be sim31-sim36, sim44, sim45")
    gamma_grid = tuple(float(value) for value in config.get("gamma_grid", ()))
    if gamma_grid != GAMMA_GRID:
        raise ValueError("gamma grid must be exactly 0/.25/.5/.75/1")

    boundary = _mapping(config, "claim_boundary")
    _exact_bool(boundary, "records_are_participants", False)
    _exact_bool(boundary, "confirmatory", False)
    _exact_bool(boundary, "evaluation_records_reused", False)
    if boundary.get("partition") != "development_only":
        raise ValueError("B6 gamma selection must remain development_only")
    previously_used = tuple(
        int(value)
        for value in boundary.get(
            "evaluation_source_records_previously_used_in_diagnosis", ()
        )
    )
    if previously_used != PREVIOUSLY_USED_EVALUATION_SOURCE_RECORDS:
        raise ValueError(
            "previously used evaluation records differ from the historical split"
        )
    if set(records) & set(previously_used):
        raise ValueError(
            "development and previously used evaluation records overlap"
        )

    activation = _mapping(config, "deferred_b6_activation")
    _exact_bool(activation, "activate_in_memory_for_this_diagnostic", True)
    _exact_bool(activation, "persisted_backup_config_must_remain_disabled", True)

    proximal = _mapping(config, "deterministic_soft_proximal")
    _exact_bool(proximal, "tau_frozen", True)
    soft_tau = _finite_unit_interval(proximal.get("tau"), name="soft proximal tau")
    if proximal.get("closed_form") != "x=Qy+tau*Pi*y":
        raise ValueError("soft proximal closed-form declaration is incorrect")
    configured_lambda = proximal.get("quadratic_penalty_lambda")
    expected_lambda = math.inf if soft_tau == 0.0 else 1.0 / soft_tau - 1.0
    if soft_tau == 0.0:
        if configured_lambda != "infinity_hard_Q_limit":
            raise ValueError("tau=0 must declare the hard-Q limiting penalty")
    elif not math.isclose(
        float(configured_lambda), expected_lambda, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError("soft proximal lambda is inconsistent with frozen tau")

    m2 = _mapping(config, "m2")
    if m2.get("candidate") != "M2":
        raise ValueError("the GPU diagnostic must use sampler candidate M2")
    if m2.get("output_rule") != "posterior_mean_of_seed_samples":
        raise ValueError("M2 development output must be the posterior seed mean")
    m2_trust = _finite_unit_interval(m2.get("trust_radius"), name="M2 trust radius")
    if m2.get("trust_radius_provenance") != (
        "historical_development_sampler_choice_reused_only_as_exploratory_constant"
    ):
        raise ValueError("M2 trust radius provenance is not the frozen exploratory rule")
    m2_seeds = tuple(int(value) for value in m2.get("seeds", ()))
    if len(m2_seeds) != 5 or len(set(m2_seeds)) != len(m2_seeds):
        raise ValueError("M2 requires exactly five distinct frozen seeds")

    selection = _mapping(config, "selection")
    if selection.get("method") != METHOD_M2:
        raise ValueError("global gamma must be selected from the frozen M2 output")
    if selection.get("primary_metric") != "e_parallel":
        raise ValueError("gamma primary metric must be e_parallel")
    if selection.get("primary_direction") != "lower":
        raise ValueError("e_parallel selection direction must be lower")
    if selection.get("preservation_metric") != "e_perp":
        raise ValueError("gamma preservation metric must be e_perp")
    if float(selection.get("preservation_reference_gamma", -1.0)) != 0.0:
        raise ValueError("preservation must be paired against gamma=0 POP geometry")
    preservation_limit = float(selection.get("maximum_median_preservation_delta"))
    if not math.isfinite(preservation_limit) or preservation_limit < 0.0:
        raise ValueError("preservation delta limit must be finite and non-negative")
    minimum_fraction = _finite_unit_interval(
        selection.get("minimum_valid_record_fraction"),
        name="minimum valid record fraction",
    )
    minimum_operator_fraction = _finite_unit_interval(
        selection.get("minimum_operator_eligibility_fraction"),
        name="minimum operator eligibility fraction",
    )
    tie_tolerance = float(selection.get("metric_tie_tolerance"))
    if not math.isfinite(tie_tolerance) or tie_tolerance < 0.0:
        raise ValueError("metric tie tolerance must be finite and non-negative")
    tie_order = tuple(float(value) for value in selection.get("tie_break_order", ()))
    if tie_order != GAMMA_GRID:
        raise ValueError("gamma tie-break order must be preregistered as 0/.25/.5/.75/1")

    return GammaDiagnosticContract(
        base_config=Path(str(config.get("base_config", ""))),
        backup_config=Path(str(config.get("backup_config", ""))),
        development_records=records,
        gamma_grid=gamma_grid,
        soft_tau=soft_tau,
        m2_trust_radius=m2_trust,
        m2_seeds=m2_seeds,
        selection_method=str(selection["method"]),
        maximum_median_preservation_delta=preservation_limit,
        minimum_valid_record_fraction=minimum_fraction,
        minimum_operator_eligibility_fraction=minimum_operator_fraction,
        metric_tie_tolerance=tie_tolerance,
        tie_break_order=tie_order,
        output_root=Path(str(config.get("output_root", ""))),
    )


def _validated_projector(projector: np.ndarray, channels: int) -> np.ndarray:
    value = np.asarray(projector, dtype=np.float64)
    if value.shape != (channels, channels) or not np.isfinite(value).all():
        raise ValueError("projector has invalid shape or non-finite values")
    if not np.allclose(value, value.T, atol=1.0e-10, rtol=0.0):
        raise ValueError("projector is not symmetric")
    if not np.allclose(value @ value, value, atol=1.0e-8, rtol=0.0):
        raise ValueError("projector is not idempotent")
    return value


def deterministic_qy(observed: np.ndarray, projector: np.ndarray) -> np.ndarray:
    """Return exact orthogonal subtraction ``(I-Pi)y`` in FP64."""

    value = np.asarray(observed, dtype=np.float64)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError("observed EEG must be a finite CxT array")
    projection = _validated_projector(projector, value.shape[0])
    return (np.eye(value.shape[0], dtype=np.float64) - projection) @ value


def deterministic_quadratic_soft_proximal(
    observed: np.ndarray,
    projector: np.ndarray,
    *,
    tau: float,
) -> np.ndarray:
    """Return the frozen quadratic-proximal closed form.

    For ``0 < tau <= 1`` this solves

    ``argmin_x 0.5||x-y||^2 + 0.5(1/tau-1)||Pi x||^2``.

    ``tau=0`` is defined by continuity as the hard-Q solution.
    """

    value = np.asarray(observed, dtype=np.float64)
    if value.ndim != 2 or not np.isfinite(value).all():
        raise ValueError("observed EEG must be a finite CxT array")
    shrink = _finite_unit_interval(tau, name="tau")
    projection = _validated_projector(projector, value.shape[0])
    projected = projection @ value
    return value - projected + shrink * projected


def _record_number(value: object) -> int:
    if isinstance(value, str) and value.startswith("sim"):
        value = value[3:]
    number = int(value)
    if number not in DEVELOPMENT_SOURCE_RECORDS:
        raise ValueError("only frozen Klados development source records are allowed")
    return number


def _metric_number(row: Mapping[str, Any], field: str) -> float:
    value = float(row[field])
    if not math.isfinite(value):
        raise ValueError(f"non-finite selection metric: {field}")
    return value


def select_global_gamma(
    rows: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Select one global development gamma under the frozen safety rule.

    Failures and B6 fallbacks remain in the denominator.  A successful POP
    fallback is included as the actual system output instead of dropping a
    difficult record from the metric.  Operator eligibility is a separate
    frozen constraint.  Gamma zero is the exact population-projector endpoint
    and is therefore the paired preservation reference.
    """

    contract = validate_diagnostic_config(config)
    selected_rows: dict[tuple[int, float], Mapping[str, Any]] = {}
    for row in rows:
        if str(row.get("method")) != contract.selection_method:
            continue
        record = _record_number(row.get("source_record"))
        gamma = float(row.get("gamma"))
        if gamma not in contract.gamma_grid:
            raise ValueError("metrics contain a gamma outside the frozen grid")
        key = (record, gamma)
        if key in selected_rows:
            raise ValueError("duplicate record/gamma M2 selection row")
        selected_rows[key] = row

    expected = {
        (record, gamma)
        for record in contract.development_records
        for gamma in contract.gamma_grid
    }
    missing = expected - set(selected_rows)
    extra = set(selected_rows) - expected
    if missing or extra:
        raise ValueError("M2 metrics do not cover the complete development gamma grid")

    baseline_e_perp: dict[int, float] = {}
    for record in contract.development_records:
        row = selected_rows[(record, 0.0)]
        if row.get("status") == "success":
            baseline_e_perp[record] = _metric_number(row, "e_perp")

    summaries: list[dict[str, Any]] = []
    for gamma in contract.gamma_grid:
        values = [selected_rows[(record, gamma)] for record in contract.development_records]
        usable: list[tuple[int, Mapping[str, Any]]] = []
        operator_eligible = 0
        for record, row in zip(contract.development_records, values):
            if gamma == 0.0 or row.get("b6_status") == "eligible":
                operator_eligible += 1
            if row.get("status") == "success" and record in baseline_e_perp:
                usable.append((record, row))
        valid_fraction = len(usable) / len(contract.development_records)
        operator_eligibility_fraction = operator_eligible / len(
            contract.development_records
        )
        primary_values = [_metric_number(row, "e_parallel") for _, row in usable]
        preservation_deltas = [
            _metric_number(row, "e_perp") - baseline_e_perp[record]
            for record, row in usable
        ]
        median_primary = float(np.median(primary_values)) if primary_values else None
        median_preservation = (
            float(np.median(preservation_deltas)) if preservation_deltas else None
        )
        preservation_passed = (
            median_preservation is not None
            and median_preservation
            <= contract.maximum_median_preservation_delta
        )
        fraction_passed = valid_fraction >= contract.minimum_valid_record_fraction
        operator_fraction_passed = (
            operator_eligibility_fraction
            >= contract.minimum_operator_eligibility_fraction
        )
        summaries.append(
            {
                "gamma": gamma,
                "records_total": len(contract.development_records),
                "records_valid": len(usable),
                "valid_record_fraction": valid_fraction,
                "operator_eligible_records": operator_eligible,
                "operator_eligibility_fraction": operator_eligibility_fraction,
                "median_e_parallel": median_primary,
                "median_e_perp_delta_vs_gamma0": median_preservation,
                "valid_fraction_passed": fraction_passed,
                "operator_eligibility_fraction_passed": operator_fraction_passed,
                "preservation_constraint_passed": preservation_passed,
                "eligible_for_global_selection": fraction_passed
                and operator_fraction_passed
                and preservation_passed,
            }
        )

    eligible = [row for row in summaries if row["eligible_for_global_selection"]]
    if not eligible:
        return {
            "status": "no_gamma_satisfied_preregistered_constraints",
            "best_global_gamma": None,
            "selection_method": contract.selection_method,
            "records_are_participants": False,
            "confirmatory": False,
            "evaluation_records_reused": False,
            "gamma_summaries": summaries,
        }

    best_primary = min(float(row["median_e_parallel"]) for row in eligible)
    tied = {
        float(row["gamma"])
        for row in eligible
        if float(row["median_e_parallel"])
        <= best_primary + contract.metric_tie_tolerance
    }
    best_gamma = next(gamma for gamma in contract.tie_break_order if gamma in tied)
    return {
        "status": "development_gamma_frozen",
        "best_global_gamma": best_gamma,
        "selection_method": contract.selection_method,
        "selection_primary_metric": "median_e_parallel",
        "preservation_constraint": (
            "median(e_perp_gamma-e_perp_gamma0)<="
            f"{contract.maximum_median_preservation_delta}"
        ),
        "statistical_unit": "source_record",
        "records_are_participants": False,
        "confirmatory": False,
        "evaluation_records_reused": False,
        "gamma_summaries": summaries,
    }


def summarize_record_metrics(
    rows: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return descriptive per-method/gamma source-record summaries."""

    contract = validate_diagnostic_config(config)
    grouped: dict[tuple[str, float], list[Mapping[str, Any]]] = {}
    for row in rows:
        method = str(row.get("method"))
        gamma = float(row.get("gamma"))
        _record_number(row.get("source_record"))
        if method not in METHODS or gamma not in contract.gamma_grid:
            raise ValueError("metric row is outside the frozen method/gamma grid")
        grouped.setdefault((method, gamma), []).append(row)

    summaries: list[dict[str, Any]] = []
    for method in METHODS:
        for gamma in contract.gamma_grid:
            values = grouped.get((method, gamma), [])
            successful = [row for row in values if row.get("status") == "success"]
            summary: dict[str, Any] = {
                "method": method,
                "gamma": gamma,
                "records_total": len(contract.development_records),
                "records_present": len(values),
                "records_successful": len(successful),
                "failure_rate": 1.0
                - len(successful) / len(contract.development_records),
                "records_are_participants": False,
                "confirmatory": False,
            }
            for metric in CORE_METRICS:
                numeric = [
                    float(row[metric])
                    for row in successful
                    if row.get(metric) not in (None, "")
                ]
                summary[f"median_{metric}"] = (
                    float(np.median(numeric)) if numeric else None
                )
            summaries.append(summary)
    return summaries


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    if not str(path) or not path.is_file():
        raise ValueError(f"required configuration is unavailable: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration is not a mapping: {path}")
    return payload


def _validate_runtime_protocol(
    diagnostic: Mapping[str, Any],
    contract: GammaDiagnosticContract,
    base: Mapping[str, Any],
    backup: Mapping[str, Any],
) -> dict[str, Any]:
    klados = _mapping(base, "klados")
    if tuple(int(value) for value in klados.get("training_source_records", ())) != tuple(
        range(1, 31)
    ):
        raise ValueError("population state must use exactly sim01-sim30")
    if tuple(int(value) for value in klados.get("development_source_records", ())) != (
        contract.development_records
    ):
        raise ValueError("base config development split differs from this diagnostic")
    if tuple(
        int(value)
        for value in klados.get(
            "historical_evaluation_source_records_already_used_in_diagnosis", ()
        )
    ) != (
        PREVIOUSLY_USED_EVALUATION_SOURCE_RECORDS
    ):
        raise ValueError(
            "base config historical evaluation split differs from this diagnostic"
        )
    if float(klados.get("calibration_seconds", -1.0)) != 10.0:
        raise ValueError("B6 diagnostic requires the frozen 10-second support")
    if backup.get("enabled") is not False:
        raise ValueError("persisted B6 config must remain disabled")
    if backup.get("backup_id") != "B6" or backup.get("formal_name") != "POP-SHRINK":
        raise ValueError("backup configuration is not B6 POP-SHRINK")
    configured_gamma = {
        float(value)
        for value in tuple(_mapping(backup, "selection").get("gamma_candidates", ()))
        + tuple(_mapping(backup, "selection").get("endpoints_as_controls", ()))
    }
    if configured_gamma != set(contract.gamma_grid):
        raise ValueError("B6 backup gamma choices differ from the diagnostic grid")

    if not math.isclose(
        float(_mapping(base, "observation").get("trust_radius_frozen")),
        contract.m2_trust_radius,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("diagnostic trust radius differs from the repaired base config")

    activated = copy.deepcopy(dict(backup))
    activated["enabled"] = True
    return activated


def _result_row(
    *,
    source_record: int,
    gamma: float,
    method: str,
    b6_status: str,
    b6_reasons: Sequence[str],
    context_projector_constructed: bool,
    effective_gamma: float,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "diagnostic_protocol": PROTOCOL,
        "dataset_id": DATASET_ID,
        "partition": "development",
        "source_record": f"sim{source_record:02d}",
        "records_are_participants": False,
        "confirmatory": False,
        "evaluation_records_reused": False,
        "gamma": gamma,
        "effective_gamma": effective_gamma,
        "method": method,
        "status": "pending",
        "failure_reason": "",
        "b6_status": b6_status,
        "b6_reasons": ";".join(str(value) for value in b6_reasons),
        "context_projector_constructed": context_projector_constructed,
        "query_clean_target_used_by_method": False,
        "query_clean_target_used_for_development_metrics_only": True,
        "latency_seconds": "",
        "peak_memory_mb": "",
        "function_evaluations": "",
        "seed_failure_rate": "",
    }
    for metric in CORE_METRICS:
        row[metric] = ""
    for metric in (
        "overlap_fraction",
        "artifact_normalized_parallel_error",
        "time_rrmse",
        "delta_snr_db",
        "projector_distance",
        "projector_max_angle_degrees",
        "projector_overlap_fraction",
    ):
        row[metric] = ""
    return row


def run_klados_b6_gamma_record(
    diagnostic_config: Mapping[str, Any],
    *,
    source_record: int,
    run_dir: Path,
    device: Any,
) -> dict[str, Any]:
    """Run all frozen gammas for one development source record on one GPU.

    This is a callable record-level GPU entry point, not a CLI.  It reuses the
    existing repaired prior and M2 sampler.  The B6 runner receives the
    loader-produced record, validates its support/query boundaries, and
    resolves split/provenance itself.  The sampler receives a query-only view
    with no paired clean fields.
    """

    contract = validate_diagnostic_config(diagnostic_config)
    record_id = _record_number(source_record)
    if getattr(device, "type", None) != "cuda":
        raise ValueError("record-level M2 diagnostic requires a CUDA device")
    base = _load_yaml(contract.base_config)
    backup = _load_yaml(contract.backup_config)
    activated_backup = _validate_runtime_protocol(
        diagnostic_config, contract, base, backup
    )

    # Imports stay inside the GPU entry point so algebra/selection tests do not
    # load the diffusion stack or inspect data.
    from eeg_cgdr.data.klados import load_klados_records
    from eeg_cgdr.data.mechanism import (
        inference_view,
        prepare_mechanism_record,
        select_records,
        support_view,
    )
    from eeg_cgdr.experiments.b6_runner import run_deferred_b6_from_actual_split
    from eeg_cgdr.experiments.mechanism_runner import (
        _OperatorArm,
        _loader_config,
        _mechanism_metrics,
        _oracle_projector,
        _sample_one_seed,
        _standardized_query_eog,
    )
    from eeg_cgdr.experiments.mechanism_training import (
        load_population_projector,
        load_repaired_prior,
    )

    prior, normalizer = load_repaired_prior(base, device=device)
    population = load_population_projector(base)
    records = load_klados_records(_loader_config(base))
    native = select_records(records, (record_id,))[0]
    prepared = prepare_mechanism_record(
        native,
        normalizer,
        source_rate=int(base["klados"]["source_sampling_rate"]),
        target_rate=int(base["preprocessing"]["target_sampling_rate"]),
        window_samples=int(base["preprocessing"]["window_samples"]),
        calibration_seconds=float(base["klados"]["calibration_seconds"]),
        guard_seconds=float(base["klados"]["guard_seconds"]),
    )
    standardized_windows, _, eog_magnitude = _standardized_query_eog(prepared)
    artifact_mask = eog_magnitude >= float(
        base["observation"]["artifact_eog_z_threshold"]
    )
    oracle, _ = _oracle_projector(prepared, int(base["p0"]["target_rank"]))
    expected_population_records = tuple(f"sim{value:02d}" for value in range(1, 31))
    population_value = np.asarray(population.projector, dtype=np.float64)
    rows: list[dict[str, Any]] = []

    for gamma in contract.gamma_grid:
        decision = run_deferred_b6_from_actual_split(
            config=base,
            backup_config=activated_backup,
            population_projector=population,
            support_record=(None if gamma == 0.0 else support_view(prepared)),
            gamma=gamma,
        )
        if decision.partition != "development":
            raise ValueError("B6 runner did not resolve the development partition")
        if decision.population_source_records != expected_population_records:
            raise ValueError("B6 runner did not use the frozen sim01-sim30 population state")
        if gamma == 0.0:
            if decision.context_support_record is not None:
                raise ValueError("gamma=0 must not inspect context support")
        elif decision.context_support_record != f"sim{record_id:02d}":
            raise ValueError("B6 runner context provenance differs from this support record")

        eligible = decision.outcome.status == "eligible"
        projector = (
            np.asarray(decision.outcome.projector, dtype=np.float64)
            if eligible and decision.outcome.projector is not None
            else population_value
        )
        projector = _validated_projector(projector, prepared.observed_continuous.shape[0])
        effective_gamma = gamma if eligible else 0.0
        shared = {
            "source_record": record_id,
            "gamma": gamma,
            "b6_status": decision.outcome.status,
            "b6_reasons": decision.outcome.reasons,
            "context_projector_constructed": (
                decision.outcome.context_projector_constructed
            ),
            "effective_gamma": effective_gamma,
        }

        deterministic_outputs = (
            (METHOD_HARD_QY, deterministic_qy(prepared.observed_continuous, projector)),
            (
                METHOD_SOFT_PROXIMAL,
                deterministic_quadratic_soft_proximal(
                    prepared.observed_continuous,
                    projector,
                    tau=contract.soft_tau,
                ),
            ),
        )
        for method, restored in deterministic_outputs:
            row = _result_row(method=method, **shared)
            row.update(
                _mechanism_metrics(
                    restored,
                    observed=prepared.observed_continuous,
                    clean=prepared.clean_continuous,
                    oracle_projector=oracle,
                    estimated_projector=projector,
                    artifact_mask=artifact_mask,
                    sampling_rate=float(prepared.sampling_rate),
                )
            )
            row["status"] = "success"
            row["soft_proximal_tau"] = (
                contract.soft_tau if method == METHOD_SOFT_PROXIMAL else ""
            )
            row["soft_proximal_lambda"] = (
                (1.0 / contract.soft_tau - 1.0)
                if method == METHOD_SOFT_PROXIMAL and contract.soft_tau > 0.0
                else ""
            )
            rows.append(row)

        m2_row = _result_row(method=METHOD_M2, **shared)
        m2_row["trust_radius"] = contract.m2_trust_radius
        m2_row["seeds"] = ";".join(str(value) for value in contract.m2_seeds)
        arm_source = "b6_pop_shrink" if eligible and gamma > 0.0 else "population_projector"
        arm = _OperatorArm(
            source=arm_source,
            projector=projector,
            p0_outcome=None,
            calibration_id=(
                f"sim{record_id:02d}_b6_gamma_{gamma:g}"
                if eligible and gamma > 0.0
                else "sim01_sim30_population_projector"
            ),
        )
        restored_by_seed: list[np.ndarray] = []
        runtimes: list[Mapping[str, Any]] = []
        failures: list[str] = []
        for seed in contract.m2_seeds:
            try:
                restored, runtime = _sample_one_seed(
                    prior=prior,
                    prepared=inference_view(prepared),
                    standardized_eog_windows=standardized_windows,
                    population_projector=population,
                    arm=arm,
                    candidate="M2",
                    trust_radius=contract.m2_trust_radius,
                    seed=seed,
                    config=base,
                    device=device,
                )
                restored_by_seed.append(restored)
                runtimes.append(runtime)
            except (RuntimeError, ValueError, FloatingPointError) as exc:
                failures.append(f"seed={seed}:{exc}")
        m2_row["seed_failure_rate"] = len(failures) / len(contract.m2_seeds)
        if failures or len(restored_by_seed) != len(contract.m2_seeds):
            m2_row["status"] = "failed"
            m2_row["failure_reason"] = ";".join(failures) or "incomplete_seed_outputs"
        else:
            posterior_mean = np.mean(np.stack(restored_by_seed, axis=0), axis=0)
            m2_row.update(
                _mechanism_metrics(
                    posterior_mean,
                    observed=prepared.observed_continuous,
                    clean=prepared.clean_continuous,
                    oracle_projector=oracle,
                    estimated_projector=projector,
                    artifact_mask=artifact_mask,
                    sampling_rate=float(prepared.sampling_rate),
                )
            )
            m2_row["latency_seconds"] = float(
                sum(float(value["latency_seconds"]) for value in runtimes)
            )
            m2_row["peak_memory_mb"] = float(
                max(float(value["peak_memory_mb"]) for value in runtimes)
            )
            m2_row["function_evaluations"] = int(
                sum(int(value["network_calls_total"]) for value in runtimes)
            )
            m2_row["status"] = "success"
        rows.append(m2_row)

    stable_record_dir = contract.output_root / "records" / f"sim{record_id:02d}"
    stable_metrics = stable_record_dir / "metrics.csv"
    _write_csv_atomic(stable_metrics, rows)
    _write_csv_atomic(run_dir / "metrics.csv", rows)
    record_summary = {
        "diagnostic_protocol": PROTOCOL,
        "dataset_id": DATASET_ID,
        "partition": "development",
        "source_record": f"sim{record_id:02d}",
        "records_are_participants": False,
        "confirmatory": False,
        "evaluation_records_reused": False,
        "gamma_grid": list(contract.gamma_grid),
        "methods": list(METHODS),
        "soft_proximal_tau": contract.soft_tau,
        "m2_trust_radius": contract.m2_trust_radius,
        "m2_seeds": list(contract.m2_seeds),
        "rows_total": len(rows),
        "rows_successful": sum(row["status"] == "success" for row in rows),
        "metrics_path": str(stable_metrics),
        "run_metrics_copy": str(run_dir / "metrics.csv"),
    }
    _write_json_atomic(stable_record_dir / "result_summary.json", record_summary)
    _write_json_atomic(run_dir / "result_summary.json", record_summary)
    return record_summary


def _write_csv_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty metrics table")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    fields = sorted({field for row in rows for field in row})
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def aggregate_klados_b6_gamma_development(
    diagnostic_config: Mapping[str, Any],
    *,
    record_metric_paths: Sequence[Path],
    output_dir: Path,
) -> dict[str, Any]:
    """Aggregate all eight development records and freeze one global gamma."""

    contract = validate_diagnostic_config(diagnostic_config)
    rows: list[dict[str, Any]] = []
    for path in record_metric_paths:
        if not path.is_file():
            raise ValueError(f"record metrics are unavailable: {path}")
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows.extend(dict(row) for row in csv.DictReader(stream))

    expected_keys = {
        (f"sim{record:02d}", gamma, method)
        for record in contract.development_records
        for gamma in contract.gamma_grid
        for method in METHODS
    }
    observed_keys: set[tuple[str, float, str]] = set()
    for row in rows:
        if row.get("partition") != "development":
            raise ValueError("aggregate input contains a non-development row")
        if str(row.get("confirmatory")).lower() not in {"false", "0"}:
            raise ValueError("aggregate input contains a confirmatory row")
        if str(row.get("evaluation_records_reused")).lower() not in {"false", "0"}:
            raise ValueError("aggregate input reused evaluation records")
        key = (str(row["source_record"]), float(row["gamma"]), str(row["method"]))
        if key in observed_keys:
            raise ValueError("aggregate input contains duplicate method rows")
        observed_keys.add(key)
    if observed_keys != expected_keys:
        raise ValueError("aggregate input does not cover the complete development grid")

    gamma_selection = select_global_gamma(rows, diagnostic_config)
    method_summaries = summarize_record_metrics(rows, diagnostic_config)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv_atomic(output_dir / "metrics.csv", rows)
    _write_csv_atomic(output_dir / "gamma_method_summary.csv", method_summaries)
    summary = {
        "diagnostic_protocol": PROTOCOL,
        "dataset_id": DATASET_ID,
        "partition": "development_only",
        "source_records": [f"sim{value:02d}" for value in contract.development_records],
        "records_are_participants": False,
        "confirmatory": False,
        "evaluation_records_reused": False,
        "evaluation_source_records_previously_used_in_diagnosis": [
            f"sim{value:02d}"
            for value in PREVIOUSLY_USED_EVALUATION_SOURCE_RECORDS
        ],
        "fresh_untouched_evaluation_available": False,
        "best_global_gamma": gamma_selection["best_global_gamma"],
        "selection": gamma_selection,
        "metrics_path": str(output_dir / "metrics.csv"),
        "method_summary_path": str(output_dir / "gamma_method_summary.csv"),
    }
    _write_json_atomic(output_dir / "result_summary.json", summary)
    return summary


__all__ = [
    "DEVELOPMENT_SOURCE_RECORDS",
    "GAMMA_GRID",
    "METHOD_HARD_QY",
    "METHOD_M2",
    "METHOD_SOFT_PROXIMAL",
    "PROTOCOL",
    "aggregate_klados_b6_gamma_development",
    "deterministic_qy",
    "deterministic_quadratic_soft_proximal",
    "run_klados_b6_gamma_record",
    "select_global_gamma",
    "summarize_record_metrics",
    "validate_diagnostic_config",
]
