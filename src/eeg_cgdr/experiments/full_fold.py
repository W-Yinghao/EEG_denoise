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
    calibration_batch,
    fit_source_controls,
    oracle_transfer,
    orthogonal_subtraction,
    population_source_transfer,
    prepare_query,
)
from eeg_cgdr.inference import (
    InformationMatchedOneStep,
    PopulationOnlyInference,
    attenuation_from_external_reference,
    matched_population_and_context_states,
    population_state_only,
)
from eeg_cgdr.operators import P0Config, P0FitOutcome, fit_p0


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


def _context_row(
    *,
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
        source_id="sim45",
        participant_id="unresolved",
        outer_fold="klados_v4_source_fold_sim45",
        session_id="source_record_sim45",
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
    }
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
        predicted_artifact_mask=artifact_mask,
        clean_mask=~artifact_mask,
        frequency_band=(0.5, 40.0),
        fallback_method_id=fallback,
        failure_reason=failure_reason,
        runtime=RuntimeEvaluation(
            latency_seconds=latency,
            peak_memory_bytes=peak_memory,
            function_evaluations=function_evaluations,
            score_evaluations=function_evaluations,
            energy_evaluations=function_evaluations if function_evaluations > 1 else 0,
            model_forward_evaluations=function_evaluations,
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
    intervals = []
    for metric in ("e_parallel", "time_rrmse", "delta_snr_db", "artifact_attenuation_db"):
        for method in (
            "matching_p0",
            "population_source_p0",
            "wrong_source_p0",
            "shuffled_calibration_p0",
            "oracle_projector_restoration",
            "information_matched_one_step",
        ):
            intervals.extend(
                item.as_row()
                for item in paired_bootstrap_ci(
                    rows,
                    metric=metric,
                    method_id=method,
                    reference_method_id="POP",
                    minimum_participants=2,
                    bootstrap_replicates=2000,
                    seed=20260801,
                )
            )
    return {"primary_posterior_mean_table": table, "paired_confidence_intervals": intervals}


def run_full_klados_fold(
    config: dict[str, Any], *, run_dir: Path, device: torch.device
) -> dict[str, Any]:
    if device.type != "cuda":
        raise RuntimeError("full fold requires one CUDA GPU")
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
    held_out = records[int(klados["held_out_record"]) - 1]
    wrong = records[int(klados["wrong_source_record"]) - 1]
    source_rate = int(klados["sampling_rate"])
    target_rate = int(config["preprocessing"]["target_sampling_rate"])
    query = prepare_query(
        held_out,
        source_rate=source_rate,
        target_rate=target_rate,
        query_start_seconds=float(klados["query_start_seconds"]),
        query_end_seconds=float(klados["query_end_seconds"]),
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
    eog_tensor = torch.as_tensor(eog_windows.astype(np.float32), device=device)
    attenuation = attenuation_from_external_reference(
        eog_tensor,
        scale=float(config["observation"]["attenuation_scale"]),
        floor=float(config["observation"]["attenuation_floor"]),
    )
    population = population_state_only(
        observed_tensor,
        attenuation=attenuation,
        base_precision=float(config["observation"]["population_precision"]),
    )
    inference = PopulationOnlyInference(prior)
    one_step = InformationMatchedOneStep(prior)
    p0_config = _p0_config(config)
    oracle = oracle_transfer(
        held_out,
        start_seconds=float(klados["query_start_seconds"]),
        stop_seconds=float(klados["query_end_seconds"]),
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

    seed_values = [int(value) for value in config["sampling"]["seeds"]]
    ddim_steps = int(config["sampling"]["ddim_steps"])
    model_evaluations = ddim_steps + 1
    for seed in seed_values:
        initial_noise = inference.make_initial_noise(population, seed=seed)
        pop_tensor, latency, peak = _timed_cuda_call(
            lambda: inference.sample(
                population, initial_noise=initial_noise, ddim_steps=ddim_steps
            ),
            device,
        )
        pop_cache[seed] = (pop_tensor.cpu().numpy(), latency, peak)

    durations = [int(value) for value in klados["calibration_seconds"]]
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
            development: list[P0FitOutcome] = []
            for record_id in range(1, 9):
                support = calibration_batch(
                    records[record_id - 1],
                    duration_seconds=float(duration),
                    source_rate=source_rate,
                    target_rate=target_rate,
                    source_label=f"sim{record_id}",
                )
                development.append(
                    fit_p0(
                        support,
                        p0_config,
                        movement_threshold=float(config["p0"]["movement_threshold"]),
                    )
                )
            outcomes["population_source_p0"] = population_source_transfer(
                development, target_rank=int(config["p0"]["target_rank"])
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
            "information_matched_one_step",
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
                elif method == "information_matched_one_step":
                    matching = outcomes["matching_p0"]
                    if matching.transfer is None:
                        restored, latency, peak = pop_cache[seed]
                        evaluations = model_evaluations
                        fallback = "POP"
                        failure_reason = ";".join(matching.reasons)
                        outcome = matching
                    else:
                        _, context = matched_population_and_context_states(
                            observed_tensor,
                            attenuation=attenuation,
                            projector=matching.transfer.projector,
                            base_precision=float(config["observation"]["context_precision"]),
                        )
                        tensor, latency, peak = _timed_cuda_call(
                            lambda context=context, seed=seed: one_step.restore(
                                observation=observed_tensor,
                                channel_precision=context.precision,
                                seed=seed,
                                timestep=100,
                            ),
                            device,
                        )
                        restored = tensor.cpu().numpy()
                        evaluations = 1
                        projector = matching.transfer.projector
                        outcome = matching
                else:
                    if outcome is None:
                        raise AssertionError(f"missing outcome for {method}")
                    if outcome.transfer is None:
                        tensor, latency, peak = _timed_cuda_call(
                            lambda: inference.sample_cgdr(
                                population,
                                rho=1.0,
                                calibration_accepted=False,
                                context_state_factory=None,
                                initial_noise=initial_noise,
                                ddim_steps=ddim_steps,
                            ),
                            device,
                        )
                        restored = tensor.cpu().numpy()
                        evaluations = model_evaluations
                        fallback = "POP"
                        failure_reason = ";".join(outcome.reasons)
                    else:
                        def factory(projector=outcome.transfer.projector):
                            _, context = matched_population_and_context_states(
                                observed_tensor,
                                attenuation=attenuation,
                                projector=projector,
                                base_precision=float(
                                    config["observation"]["context_precision"]
                                ),
                            )
                            return context

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
                        method=method,
                        operator_source=(
                            "none" if method == "POP" else method.removesuffix("_p0")
                        ),
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
                method=method,
                operator_source="none" if method == "POP" else method.removesuffix("_p0"),
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
                latency=float(np.mean([item[0] for item in runtimes])),
                peak_memory=max(
                    (item[1] for item in runtimes if item[1] is not None), default=None
                ),
                function_evaluations=int(np.mean([item[2] for item in runtimes])),
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
                "outer_unit": "source_record_sim45",
                "participant_mapping": "blocked_not_guessed",
                "calibration_seconds": durations,
                "query_seconds": observed_flat.shape[1] / target_rate,
                "query_windows": int(query.valid_samples.size),
                "query_samples": int(query.valid_samples.sum()),
                "calibration_query_disjoint": True,
            },
            "g1_status": "source_record_mechanism_result_available",
            "g2_status": "source_specificity_controls_available_participant_claim_inconclusive",
            "participant_level_confidence": "inconclusive",
            "rows": len(rows),
            "metrics_path": str(metrics_path),
        }
    )
    summary_path = output / "result_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (output / "RESULT.md").write_text(
        "# CGDR Klados v4 source fold\n\n"
        "This run is a complete real EEG/EOG-backed paired semi-simulation source-record fold. "
        "It uses all non-overlapping query samples after the frozen 30 s calibration and 1 s guard. "
        "The v4 release does not expose a reliable 54-to-27 participant map, so this result supports "
        "record-level G1 mechanism diagnostics and source-operator controls only; participant-level "
        "G2 remains inconclusive. See `result_summary.json` and `metrics.csv`.\n",
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
