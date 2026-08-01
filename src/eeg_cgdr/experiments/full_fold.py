"""Complete Klados v4 paired source-fold CGDR/P0 experiment."""

from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from eeg_cgdr.data.klados import load_klados_records
from eeg_cgdr.evaluation import (
    ContextIdentity,
    RuntimeEvaluation,
    evaluate_context,
    paired_bootstrap_ci,
)
from eeg_cgdr.experiments.common import load_best_prior, train_clean_prior
from eeg_cgdr.experiments.klados import (
    fit_source_controls,
    oracle_transfer,
    orthogonal_subtraction,
    prepare_query,
)
from eeg_cgdr.inference import (
    InformationMatchedOneStep,
    PopulationOnlyInference,
    attenuation_from_external_reference,
    matched_population_and_context_states,
    population_state_only,
)
from eeg_cgdr.operators import P0Config, P0FitOutcome


def _loader_config(config: dict[str, Any]) -> dict[str, Any]:
    klados = config["klados"]
    return {
        "data_root": klados["data_root"],
        "files": {
            "contaminated": klados["contaminated"],
            "clean": klados["clean"],
            "heog": klados["heog"],
            "veog": klados["veog"],
        },
        "official_description": {"records": 54},
    }


def _p0_config(config: dict[str, Any]) -> P0Config:
    raw = config["p0"]
    return P0Config(
        target_rank=int(raw["target_rank"]),
        ridge_lambda=float(raw["ridge_lambda"]),
        maximum_reference_condition=float(raw["maximum_reference_condition"]),
        minimum_singular_ratio=float(raw["minimum_singular_ratio"]),
        minimum_movement_coverage=float(raw["minimum_movement_coverage"]),
        bootstrap_replicates=int(raw["bootstrap_replicates"]),
        bootstrap_block_samples=int(raw["bootstrap_block_samples"]),
        minimum_bootstrap_success=float(raw["minimum_bootstrap_success"]),
        maximum_bootstrap_median_distance=float(raw["maximum_bootstrap_median_distance"]),
        maximum_bootstrap_q90_distance=float(raw["maximum_bootstrap_q90_distance"]),
        seed=int(config["seed"]),
    )


def _optional_float(row: dict[str, str], field: str) -> float | None:
    raw = row.get(field, "").strip()
    if not raw:
        return None
    value = float(raw)
    if not np.isfinite(value):
        raise ValueError(f"non-finite {field} in frozen split manifest")
    return value


def _source_record_number(value: str) -> int:
    if not value.startswith("sim") or not value[3:].isdigit():
        raise ValueError(f"invalid Klados source-record identifier: {value!r}")
    number = int(value[3:])
    if number < 1 or number > 54:
        raise ValueError(f"Klados source-record identifier is out of range: {value!r}")
    return number


def _load_frozen_split(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and return the one frozen held-out source-record fold.

    Participant provenance is unresolved in Klados v4, so the manifest is
    intentionally source-record level.  The support and query boundaries are
    authoritative here; redundant record-order assumptions are not used.
    """

    klados = config["klados"]
    path = Path(klados["split_manifest"])
    required = {
        "dataset_version",
        "outer_fold",
        "split",
        "participant",
        "session",
        "record",
        "calibration_start",
        "calibration_end",
        "query_start",
        "query_end",
        "sampling_rate",
        "status",
    }
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("frozen split manifest is missing required columns")
        rows = [dict(row) for row in reader]
    if len(rows) != 2:
        raise ValueError("Klados source fold must contain exactly support and query rows")
    if {row["dataset_version"] for row in rows} != {"klados_bamidis_v4"}:
        raise ValueError("unexpected dataset version in frozen split manifest")
    if {row["outer_fold"] for row in rows} != {str(config["experiment_id"])}:
        raise ValueError("frozen split outer-fold ID does not match the experiment")
    if {row["split"] for row in rows} != {"test"}:
        raise ValueError("held-out support and query must both be in the test source fold")
    support_rows = [row for row in rows if row["status"] == "held_out_calibration"]
    query_rows = [row for row in rows if row["status"] == "held_out_query"]
    if len(support_rows) != 1 or len(query_rows) != 1:
        raise ValueError("frozen split requires one held-out calibration and one query row")
    support = support_rows[0]
    query = query_rows[0]
    if any(
        _optional_float(support, field) is not None
        for field in ("query_start", "query_end")
    ) or any(
        _optional_float(query, field) is not None
        for field in ("calibration_start", "calibration_end")
    ):
        raise ValueError("support and query rows must not carry each other's intervals")
    identity_fields = ("participant", "session", "record", "sampling_rate")
    if any(support[field] != query[field] for field in identity_fields):
        raise ValueError("held-out support and query identity fields do not match")
    record_number = _source_record_number(support["record"])
    source_rate_value = float(support["sampling_rate"])
    source_rate = int(source_rate_value)
    if source_rate != source_rate_value or source_rate <= 0:
        raise ValueError("split sampling rate must be a positive integer")
    support_start = _optional_float(support, "calibration_start")
    support_end = _optional_float(support, "calibration_end")
    query_start = _optional_float(query, "query_start")
    query_end = _optional_float(query, "query_end")
    if None in (support_start, support_end, query_start, query_end):
        raise ValueError("held-out support/query boundaries must be explicit")
    assert support_start is not None
    assert support_end is not None
    assert query_start is not None
    assert query_end is not None
    if support_start != 0.0:
        raise ValueError("the current P0 calibration loader requires support to start at zero")
    if not support_start < support_end <= query_start < query_end:
        raise ValueError("support/query intervals are empty, overlapping, or reversed")
    guard = float(klados["guard_seconds"])
    if query_start - support_end < guard:
        raise ValueError("frozen split does not preserve the configured filter guard")
    durations = [int(value) for value in klados["calibration_seconds"]]
    if not durations or 0 not in durations or min(durations) < 0:
        raise ValueError("calibration durations must include POP duration zero")
    if max(durations) > support_end - support_start:
        raise ValueError("configured calibration duration exceeds frozen support")
    wrong_record = int(klados["wrong_source_record"])
    if wrong_record == record_number:
        raise ValueError("wrong-source control cannot equal the held-out source record")
    return {
        "path": str(path),
        "dataset_version": support["dataset_version"],
        "outer_fold": support["outer_fold"],
        "participant": support["participant"],
        "session": support["session"],
        "record": support["record"],
        "record_number": record_number,
        "source_rate": source_rate,
        "support_start": support_start,
        "support_end": support_end,
        "query_start": query_start,
        "query_end": query_end,
        "guard_seconds": query_start - support_end,
        "wrong_record_number": wrong_record,
    }


def _flatten_valid(windows: np.ndarray, valid_samples: np.ndarray) -> np.ndarray:
    if windows.ndim != 3 or windows.shape[0] != valid_samples.size:
        raise ValueError("window/valid-sample mismatch")
    pieces = [
        windows[index, :, : int(valid)]
        for index, valid in enumerate(valid_samples)
        if int(valid) > 0
    ]
    return np.concatenate(pieces, axis=1)


def _flatten_mask(windows: np.ndarray, valid_samples: np.ndarray) -> np.ndarray:
    if windows.ndim != 2 or windows.shape[0] != valid_samples.size:
        raise ValueError("mask/valid-sample mismatch")
    return np.concatenate(
        [windows[index, : int(valid)] for index, valid in enumerate(valid_samples)]
    ).astype(bool)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _timed_cuda_call(function, device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    output = function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak = int(torch.cuda.max_memory_allocated(device))
    else:
        peak = None
    return output, time.perf_counter() - start, peak


def _operator_source(method: str) -> str:
    return {
        "raw_observation": "none",
        "POP": "none",
        "matching_p0": "matching_source_support",
        "population_source_p0": "unavailable_participant_safe_population",
        "wrong_source_p0": "frozen_wrong_source_record",
        "shuffled_calibration_p0": "matching_support_block_shuffled",
        "oracle_projector_restoration": "paired_query_oracle_projector",
        "oracle_orthogonal_subtraction": "paired_query_oracle_projector",
        "oracle_information_matched_one_step": "paired_query_oracle_projector",
        "matching_information_matched_one_step": "matching_source_support",
    }[method]


def _context_row(
    *,
    source_id: str,
    participant_id: str,
    outer_fold: str,
    session_id: str,
    method: str,
    operator_source: str,
    seed: int | None,
    calibration_seconds: int,
    restored_windows: np.ndarray,
    observed: np.ndarray,
    clean: np.ndarray,
    valid: np.ndarray,
    artifact_mask: np.ndarray,
    oracle_projector_value: np.ndarray,
    estimated_projector: np.ndarray | None,
    fallback: str | None,
    failure_reason: str | None,
    latency: float,
    peak_memory: int | None,
    function_evaluations: int,
    p0_outcome: P0FitOutcome | None,
    sampling_rate: int,
    aggregate: bool,
) -> dict[str, Any]:
    restored = _flatten_valid(restored_windows, valid)
    identity = ContextIdentity(
        dataset_id="klados_bamidis_v4",
        source_id=source_id,
        participant_id=participant_id,
        outer_fold=outer_fold,
        session_id=session_id,
        context_id=(
            f"calibration_{calibration_seconds:02d}s_"
            + ("posterior_mean" if aggregate else "seed")
        ),
        method_id=method,
        operator_source=operator_source,
        seed=seed,
    )
    diagnostics = p0_outcome.transfer.diagnostics if p0_outcome and p0_outcome.transfer else {}
    extra = {
        "calibration_seconds": calibration_seconds,
        "query_seconds": observed.shape[1] / sampling_rate,
        "query_windows": int(valid.size),
        "aggregate_across_seeds": aggregate,
        "p0_eligibility": p0_outcome.status if p0_outcome else "not_applicable",
        "p0_rank": p0_outcome.transfer.rank if p0_outcome and p0_outcome.transfer else None,
        "p0_bootstrap_success": diagnostics.get("bootstrap_success_rate"),
        "p0_bootstrap_median_distance": diagnostics.get(
            "bootstrap_median_projector_distance"
        ),
        "participant_mapping_verified": False,
        "generalized_bayes": True,
        "mask_regime": "external",
        "artifact_mask_role": "reference_intervals_not_detector_prediction",
    }
    if method in ("raw_observation", "oracle_orthogonal_subtraction"):
        score_evaluations = 0
        energy_evaluations = 0
        model_forward_evaluations = 0
    elif method.endswith("information_matched_one_step"):
        score_evaluations = function_evaluations
        energy_evaluations = 0
        model_forward_evaluations = function_evaluations
    else:
        score_evaluations = function_evaluations
        energy_evaluations = function_evaluations
        model_forward_evaluations = function_evaluations
    return evaluate_context(
        identity,
        status="success" if failure_reason is None else "rolled_back",
        observed=observed,
        restored=restored,
        clean=clean,
        sampling_rate=float(sampling_rate),
        oracle_projector=oracle_projector_value,
        estimated_projector=estimated_projector,
        artifact_mask=artifact_mask,
        predicted_artifact_mask=None,
        clean_mask=~artifact_mask,
        frequency_band=(0.5, 40.0),
        fallback_method_id=fallback,
        failure_reason=failure_reason,
        runtime=RuntimeEvaluation(
            latency_seconds=latency,
            peak_memory_bytes=peak_memory,
            function_evaluations=function_evaluations,
            score_evaluations=score_evaluations,
            energy_evaluations=energy_evaluations,
            model_forward_evaluations=model_forward_evaluations,
        ),
        extra_fields=extra,
    )


def _aggregate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_fields = (
        "e_parallel",
        "e_perp",
        "d_perp_y",
        "overlap_fraction",
        "time_rrmse",
        "frequency_rrmse",
        "correlation",
        "delta_snr_db",
        "projector_distance",
        "projector_mean_angle_deg",
        "artifact_attenuation_db",
        "clean_interval_preservation",
        "latency_seconds",
        "peak_memory_bytes",
        "function_evaluations",
    )
    primary = [row for row in rows if row.get("aggregate_across_seeds")]
    table: dict[str, Any] = {}
    for row in primary:
        key = f"{int(row['calibration_seconds']):02d}s/{row['method_id']}"
        table[key] = {field: row.get(field) for field in numeric_fields}
        table[key]["status"] = row["status"]
        table[key]["fallback_method_id"] = row.get("fallback_method_id")
        table[key]["p0_eligibility"] = row.get("p0_eligibility")
    intervals: list[dict[str, Any]] = []
    metrics = (
        "e_parallel",
        "e_perp",
        "time_rrmse",
        "delta_snr_db",
        "artifact_attenuation_db",
    )
    pop_contrasts = (
        "matching_p0",
        "population_source_p0",
        "wrong_source_p0",
        "shuffled_calibration_p0",
        "oracle_projector_restoration",
        "raw_observation",
        "oracle_orthogonal_subtraction",
        "oracle_information_matched_one_step",
        "matching_information_matched_one_step",
    )
    durations = sorted({int(row["calibration_seconds"]) for row in primary})
    for duration in durations:
        duration_rows = [
            row for row in primary if int(row["calibration_seconds"]) == duration
        ]
        available = {str(row["method_id"]) for row in duration_rows}
        contrasts = [
            (method, "POP", "method_minus_POP")
            for method in pop_contrasts
            if method in available and "POP" in available
        ]
        if "matching_p0" in available and "population_source_p0" in available:
            contrasts.append(
                ("matching_p0", "population_source_p0", "G2_matching_minus_population")
            )
        if "matching_p0" in available and "shuffled_calibration_p0" in available:
            contrasts.append(
                ("matching_p0", "shuffled_calibration_p0", "G2_matching_minus_shuffled")
            )
        for metric in metrics:
            for method, reference, family in contrasts:
                for item in paired_bootstrap_ci(
                    duration_rows,
                    metric=metric,
                    method_id=method,
                    reference_method_id=reference,
                    minimum_participants=2,
                    bootstrap_replicates=2000,
                    seed=20260801 + duration,
                    include_overall=False,
                ):
                    interval = item.as_row()
                    interval["calibration_seconds"] = duration
                    interval["contrast_family"] = family
                    intervals.append(interval)
    return {
        "primary_posterior_mean_table": table,
        "paired_confidence_intervals": intervals,
        "confidence_interval_rows": "posterior_mean_only_separate_by_calibration_duration",
    }


def run_full_klados_fold(
    config: dict[str, Any], *, run_dir: Path, device: torch.device
) -> dict[str, Any]:
    if device.type != "cuda":
        raise RuntimeError("full fold requires one CUDA GPU")
    frozen_split = _load_frozen_split(config)
    population_precision = float(config["observation"]["population_precision"])
    context_precision = float(config["observation"]["context_precision"])
    if population_precision != context_precision:
        raise ValueError(
            "population_precision and context_precision must be exactly equal "
            "within a matched POP/P0 comparison"
        )
    energy_scale = float(config["observation"]["energy_scale"])
    if not np.isfinite(energy_scale) or energy_scale < 0.0:
        raise ValueError("energy_scale must be finite and non-negative")
    expected_methods = [
        "raw_observation",
        "POP",
        "matching_p0",
        "population_source_p0",
        "wrong_source_p0",
        "shuffled_calibration_p0",
        "oracle_projector_restoration",
        "oracle_orthogonal_subtraction",
        "oracle_information_matched_one_step",
        "matching_information_matched_one_step",
    ]
    if list(config["methods"]) != expected_methods:
        raise ValueError("full-fold method list differs from the frozen comparison set")
    output = Path(config["outputs"]["root"])
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(config["outputs"]["checkpoint"])
    best_checkpoint = Path(config["outputs"]["best_checkpoint"])
    training_result = train_clean_prior(
        config,
        device=device,
        checkpoint=checkpoint,
        best_checkpoint=best_checkpoint,
        history_path=output / "training_metrics.csv",
    )
    if training_result.stopped_for_signal:
        result = {
            "status": "checkpointed_for_resume",
            "checkpoint": str(checkpoint),
            "epochs_completed": training_result.epochs_completed,
            "steps_completed": training_result.steps_completed,
        }
        (run_dir / "full_fold_status.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        return result

    prior, normalizer = load_best_prior(config, best_checkpoint, device)
    for parameter in prior.parameters():
        parameter.requires_grad_(False)
    records = load_klados_records(_loader_config(config))
    klados = config["klados"]
    held_out = records[int(frozen_split["record_number"]) - 1]
    wrong = records[int(frozen_split["wrong_record_number"]) - 1]
    if held_out.record_id == wrong.record_id:
        raise AssertionError("held-out and wrong-source records are not disjoint")
    source_rate = int(frozen_split["source_rate"])
    if held_out.samples / source_rate < float(frozen_split["query_end"]):
        raise ValueError("frozen query extends beyond the held-out source record")
    target_rate = int(config["preprocessing"]["target_sampling_rate"])
    query = prepare_query(
        held_out,
        source_rate=source_rate,
        target_rate=target_rate,
        query_start_seconds=float(frozen_split["query_start"]),
        query_end_seconds=float(frozen_split["query_end"]),
        window_samples=int(config["preprocessing"]["window_samples"]),
        attenuation_scale=float(config["observation"]["attenuation_scale"]),
    )
    mean = float(normalizer["mean"])
    scale = float(normalizer["standard_deviation"])
    observed_windows = ((query.contaminated - mean) / scale).astype(np.float32)
    clean_windows = ((query.clean - mean) / scale).astype(np.float32)
    eog_windows = query.eog.astype(np.float64, copy=True)
    eog_flat = _flatten_valid(eog_windows, query.valid_samples)
    eog_mean = eog_flat.mean(axis=1, keepdims=True)
    eog_scale = np.maximum(eog_flat.std(axis=1, keepdims=True), 1.0e-8)
    for index, valid in enumerate(query.valid_samples):
        eog_windows[index, :, : int(valid)] = (
            eog_windows[index, :, : int(valid)] - eog_mean
        ) / eog_scale
        eog_windows[index, :, int(valid) :] = 0.0

    observed_tensor = torch.as_tensor(observed_windows, device=device)
    valid_time_mask = (
        torch.arange(observed_tensor.shape[-1], device=device)[None, :]
        < torch.as_tensor(query.valid_samples, device=device)[:, None]
    )
    eog_tensor = torch.as_tensor(eog_windows.astype(np.float32), device=device)
    attenuation = attenuation_from_external_reference(
        eog_tensor,
        scale=float(config["observation"]["attenuation_scale"]),
        floor=float(config["observation"]["attenuation_floor"]),
    )
    population = population_state_only(
        observed_tensor,
        attenuation=attenuation,
        base_precision=population_precision,
        energy_scale=energy_scale,
        valid_time_mask=valid_time_mask,
    )

    def context_state(projector: np.ndarray) -> Any:
        matched_population, context = matched_population_and_context_states(
            observed_tensor,
            attenuation=attenuation,
            projector=projector,
            base_precision=context_precision,
            energy_scale=energy_scale,
            valid_time_mask=valid_time_mask,
        )
        if not torch.equal(matched_population.precision, population.precision):
            raise AssertionError("matched E0 precision differs from direct POP precision")
        if float(matched_population.energy_scale) != float(population.energy_scale):
            raise AssertionError("matched E0 energy scale differs from direct POP")
        return context

    inference = PopulationOnlyInference(prior)
    one_step = InformationMatchedOneStep(prior)
    p0_config = _p0_config(config)
    oracle = oracle_transfer(
        held_out,
        start_seconds=float(frozen_split["query_start"]),
        stop_seconds=float(frozen_split["query_end"]),
        sampling_rate=source_rate,
        target_rank=int(config["p0"]["target_rank"]),
    )

    observed_flat = _flatten_valid(observed_windows, query.valid_samples)
    clean_flat = _flatten_valid(clean_windows, query.valid_samples)
    artifact_flat = _flatten_mask(query.artifact_mask, query.valid_samples)
    if not np.any(artifact_flat) or not np.any(~artifact_flat):
        raise AssertionError("external artifact mask lacks artifact or clean intervals")

    rows: list[dict[str, Any]] = []
    restored_by_condition: dict[tuple[int, str], list[np.ndarray]] = defaultdict(list)
    runtime_by_condition: dict[tuple[int, str], list[tuple[float, int | None, int]]] = defaultdict(list)
    outcome_by_condition: dict[tuple[int, str], P0FitOutcome | None] = {}
    projector_by_condition: dict[tuple[int, str], np.ndarray | None] = {}
    fallback_by_condition: dict[tuple[int, str], tuple[str | None, str | None]] = {}
    pop_cache: dict[int, tuple[np.ndarray, float, int | None]] = {}

    durations = [int(value) for value in klados["calibration_seconds"]]
    for duration in durations:
        raw_key = (duration, "raw_observation")
        restored_by_condition[raw_key].append(observed_windows.copy())
        runtime_by_condition[raw_key].append((0.0, 0, 0))
        outcome_by_condition[raw_key] = None
        projector_by_condition[raw_key] = None
        fallback_by_condition[raw_key] = (None, None)

    seed_values = [int(value) for value in config["sampling"]["seeds"]]
    ddim_steps = int(config["sampling"]["ddim_steps"])
    model_evaluations = ddim_steps
    for seed in seed_values:
        initial_noise = inference.make_initial_noise(population, seed=seed)
        pop_tensor, latency, peak = _timed_cuda_call(
            lambda: inference.sample(
                population, initial_noise=initial_noise, ddim_steps=ddim_steps
            ),
            device,
        )
        pop_cache[seed] = (pop_tensor.cpu().numpy(), latency, peak)

    for duration in durations:
        outcomes: dict[str, P0FitOutcome] = {}
        if duration > 0:
            outcomes.update(
                fit_source_controls(
                    matching_record=held_out,
                    wrong_record=wrong,
                    duration_seconds=float(duration),
                    source_rate=source_rate,
                    target_rate=target_rate,
                    config=p0_config,
                    movement_threshold=float(config["p0"]["movement_threshold"]),
                    seed=int(config["seed"]) + duration,
                )
            )
            outcomes["population_source_p0"] = P0FitOutcome(
                "ineligible",
                None,
                (
                    "participant_mapping_unresolved_population_operator_not_leakage_safe",
                ),
            )
            outcomes["oracle_projector_restoration"] = P0FitOutcome(
                "eligible", oracle, ()
            )

        methods = ["POP"] if duration == 0 else [
            "POP",
            "matching_p0",
            "population_source_p0",
            "wrong_source_p0",
            "shuffled_calibration_p0",
            "oracle_projector_restoration",
            "oracle_orthogonal_subtraction",
            "oracle_information_matched_one_step",
            "matching_information_matched_one_step",
        ]
        for seed in seed_values:
            initial_noise = inference.make_initial_noise(population, seed=seed)
            for method in methods:
                outcome = outcomes.get(method)
                projector = outcome.transfer.projector if outcome and outcome.transfer else None
                fallback: str | None = None
                failure_reason: str | None = None
                if method == "POP":
                    restored, latency, peak = pop_cache[seed]
                    evaluations = model_evaluations
                elif method == "oracle_orthogonal_subtraction":
                    start = time.perf_counter()
                    restored = orthogonal_subtraction(observed_windows, oracle.projector).astype(
                        np.float32
                    )
                    latency = time.perf_counter() - start
                    peak = 0
                    evaluations = 0
                    projector = oracle.projector
                elif method in (
                    "oracle_information_matched_one_step",
                    "matching_information_matched_one_step",
                ):
                    selected_outcome = outcomes[
                        "oracle_projector_restoration"
                        if method == "oracle_information_matched_one_step"
                        else "matching_p0"
                    ]
                    if selected_outcome.transfer is None:
                        restored, latency, peak = pop_cache[seed]
                        evaluations = model_evaluations
                        fallback = "POP"
                        failure_reason = ";".join(selected_outcome.reasons)
                        outcome = selected_outcome
                    else:
                        context = context_state(selected_outcome.transfer.projector)
                        tensor, latency, peak = _timed_cuda_call(
                            lambda context=context, seed=seed: one_step.restore(
                                observation=observed_tensor,
                                channel_precision=(
                                    context.precision * float(context.energy_scale)
                                ),
                                seed=seed,
                                timestep=int(config["one_step"]["timestep"]),
                                proximal_strength=float(
                                    config["one_step"]["proximal_strength"]
                                ),
                                valid_time_mask=context.valid_time_mask,
                            ),
                            device,
                        )
                        restored = tensor.cpu().numpy()
                        evaluations = 1
                        projector = selected_outcome.transfer.projector
                        outcome = selected_outcome
                else:
                    if outcome is None:
                        raise AssertionError(f"missing outcome for {method}")
                    if outcome.transfer is None:
                        restored, latency, peak = pop_cache[seed]
                        evaluations = model_evaluations
                        fallback = "POP"
                        failure_reason = ";".join(outcome.reasons)
                    else:
                        def factory(projector=outcome.transfer.projector):
                            return context_state(projector)

                        tensor, latency, peak = _timed_cuda_call(
                            lambda: inference.sample_cgdr(
                                population,
                                rho=float(config["observation"]["rho"]),
                                calibration_accepted=True,
                                context_state_factory=factory,
                                initial_noise=initial_noise,
                                ddim_steps=ddim_steps,
                            ),
                            device,
                        )
                        restored = tensor.cpu().numpy()
                        evaluations = model_evaluations
                key = (duration, method)
                restored_by_condition[key].append(restored)
                runtime_by_condition[key].append((latency, peak, evaluations))
                outcome_by_condition[key] = outcome
                projector_by_condition[key] = projector
                fallback_by_condition[key] = (fallback, failure_reason)
                rows.append(
                    _context_row(
                        source_id=str(frozen_split["record"]),
                        participant_id=str(frozen_split["participant"]),
                        outer_fold=str(frozen_split["outer_fold"]),
                        session_id=str(frozen_split["session"]),
                        method=method,
                        operator_source=_operator_source(method),
                        seed=seed,
                        calibration_seconds=duration,
                        restored_windows=restored,
                        observed=observed_flat,
                        clean=clean_flat,
                        valid=query.valid_samples,
                        artifact_mask=artifact_flat,
                        oracle_projector_value=oracle.projector,
                        estimated_projector=projector,
                        fallback=fallback,
                        failure_reason=failure_reason,
                        latency=latency,
                        peak_memory=peak,
                        function_evaluations=evaluations,
                        p0_outcome=outcome,
                        sampling_rate=target_rate,
                        aggregate=False,
                    )
                )

    for (duration, method), restored_samples in restored_by_condition.items():
        restored_mean = np.mean(np.stack(restored_samples, axis=0), axis=0)
        runtimes = runtime_by_condition[(duration, method)]
        fallback, failure_reason = fallback_by_condition[(duration, method)]
        rows.append(
            _context_row(
                source_id=str(frozen_split["record"]),
                participant_id=str(frozen_split["participant"]),
                outer_fold=str(frozen_split["outer_fold"]),
                session_id=str(frozen_split["session"]),
                method=method,
                operator_source=_operator_source(method),
                seed=None,
                calibration_seconds=duration,
                restored_windows=restored_mean,
                observed=observed_flat,
                clean=clean_flat,
                valid=query.valid_samples,
                artifact_mask=artifact_flat,
                oracle_projector_value=oracle.projector,
                estimated_projector=projector_by_condition[(duration, method)],
                fallback=fallback,
                failure_reason=failure_reason,
                latency=float(np.sum([item[0] for item in runtimes])),
                peak_memory=max(
                    (item[1] for item in runtimes if item[1] is not None), default=None
                ),
                function_evaluations=int(np.sum([item[2] for item in runtimes])),
                p0_outcome=outcome_by_condition[(duration, method)],
                sampling_rate=target_rate,
                aggregate=True,
            )
        )

    metrics_path = Path(config["outputs"]["metrics"])
    _write_rows(metrics_path, rows)
    summary = _aggregate_summary(rows)
    summary.update(
        {
            "status": "completed",
            "evidence_status": config["evidence_status"],
            "prior_contract": dict(config["prior_contract"]),
            "scientific_label": config["scientific_label"],
            "claim_scope": config["claim_scope"],
            "clean_prior_training": {
                "dataset": "EEGdenoiseNet clean EEG",
                "epochs_completed": training_result.epochs_completed,
                "steps_completed": training_result.steps_completed,
                "best_validation_loss": training_result.best_validation_loss,
                "resumed": training_result.resumed,
                "checkpoint": str(best_checkpoint),
            },
            "split": {
                "manifest": frozen_split["path"],
                "outer_fold": frozen_split["outer_fold"],
                "outer_unit": frozen_split["session"],
                "held_out_record": frozen_split["record"],
                "held_out_participant": frozen_split["participant"],
                "participant_mapping": "blocked_not_guessed",
                "support_start_seconds": frozen_split["support_start"],
                "support_end_seconds": frozen_split["support_end"],
                "query_start_seconds": frozen_split["query_start"],
                "query_end_seconds": frozen_split["query_end"],
                "guard_seconds": frozen_split["guard_seconds"],
                "calibration_seconds": durations,
                "query_seconds": observed_flat.shape[1] / target_rate,
                "query_windows": int(query.valid_samples.size),
                "query_samples": int(query.valid_samples.sum()),
                "calibration_query_disjoint": True,
                "wrong_source_record": f"sim{frozen_split['wrong_record_number']}",
                "wrong_source_scope": config["klados"]["wrong_source_role"],
            },
            "observation_semantics": {
                "population_and_context_base_precision": population_precision,
                "energy_scale": energy_scale,
                "attenuation_source": config["observation"]["attenuation_source"],
                "POP": "same query, clean prior, population E0, attenuation source, sampler and random stream; no calibration-derived projector",
                "population_operator": "N/A: participant mapping unresolved, so a leakage-safe population operator cannot be fit",
            },
            "one_step": dict(config["one_step"]),
            "mechanism_interpretation": (
                "not performed in train-fold; use the independent "
                "mechanism-audit route for exploratory diagnosis"
            ),
            "single_source_exploratory_check": "NOT_RUN_IN_THIS_ROUTE",
            "formal_gate_status": {
                "G1": "NOT_RUN_BLOCKED",
                "G2": "NOT_RUN_BLOCKED",
            },
            "g1_status": "NOT_RUN_BLOCKED",
            "g2_status": "NOT_RUN_BLOCKED",
            "participant_level_confidence": "inconclusive",
            "rows": len(rows),
            "metrics_path": str(metrics_path),
        }
    )
    summary_path = output / "result_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "RESULT.md").write_text(
        "# Corrected CGDR Klados v4 source fold\n\n"
        "This run is a complete real EEG/EOG-backed paired semi-simulation source-record fold. "
        "It uses all non-overlapping query samples after the frozen 30 s calibration and 1 s guard. "
        "The v4 release does not expose a reliable 54-to-27 participant map, so this result supports "
        "source-record debugging only. It is marked `exploratory_pre_repair_not_gate_evidence`. "
        "A leakage-safe population operator cannot be fit, and no participant-level gate is run. "
        "Formal G1 and G2 are NOT RUN/BLOCKED. Any single-source direction check belongs in the "
        "independent `mechanism-audit` route and cannot pass or fail a formal gate. Slurm 919385 is "
        "invalid inference evidence, while its "
        "independently trained clean-prior checkpoint is reused. See `result_summary.json` and "
        "`metrics.csv`.\n",
        encoding="utf-8",
    )
    run_status = {
        "status": "completed",
        "result_summary": str(summary_path),
        "metrics": str(metrics_path),
        "checkpoint": str(best_checkpoint),
    }
    (run_dir / "full_fold_status.json").write_text(
        json.dumps(run_status, indent=2) + "\n", encoding="utf-8"
    )
    return summary
