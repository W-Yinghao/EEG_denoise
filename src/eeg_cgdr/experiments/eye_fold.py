"""Natural Eye-BCI participant-held-out P0 fold."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from eeg_cgdr.data.eye_bci_full import (
    EyeBciNormalization,
    _resample_continuous,
    fit_outer_training_normalization,
    prepare_eye_bci_query,
    read_eye_bci_record,
    target_for,
    write_eye_bci_split_manifest,
)
from eeg_cgdr.evaluation import ContextIdentity, RuntimeEvaluation, evaluate_context
from eeg_cgdr.experiments.common import load_best_prior
from eeg_cgdr.experiments.klados import (
    block_shuffle_reference,
    population_source_transfer,
)
from eeg_cgdr.inference import (
    InformationMatchedOneStep,
    PopulationOnlyInference,
    attenuation_from_external_reference,
    matched_population_and_context_states,
    population_state_only,
)
from eeg_cgdr.operators import CalibrationBatch, P0Config, P0FitOutcome, fit_p0


def _p0_config(config: dict[str, Any], target_rate: int) -> P0Config:
    raw = config["p0"]
    return P0Config(
        target_rank=int(raw["target_rank"]),
        ridge_lambda=float(raw["ridge_lambda"]),
        maximum_reference_condition=float(raw["maximum_reference_condition"]),
        minimum_singular_ratio=float(raw["minimum_singular_ratio"]),
        minimum_movement_coverage=float(raw["minimum_movement_coverage"]),
        bootstrap_replicates=int(raw["bootstrap_replicates"]),
        bootstrap_block_samples=int(round(float(raw["bootstrap_block_seconds"]) * target_rate)),
        seed=int(config["seed"]),
    )


def _calibration_batch(
    record,
    normalization: EyeBciNormalization,
    *,
    seconds: float,
    target_rate: int,
    eog_override: np.ndarray | None = None,
) -> CalibrationBatch:
    source_rate = int(round(record.sampling_rate))
    samples = min(int(round(seconds * source_rate)), record.eeg.shape[1])
    eeg = _resample_continuous(record.eeg[:, :samples], source_rate, target_rate)
    eeg = (eeg - normalization.mean) / normalization.standard_deviation
    heo = _resample_continuous(record.heo[None, :samples], source_rate, target_rate)
    if eog_override is not None:
        if eog_override.shape != heo.shape:
            raise ValueError("Eye-BCI shuffled HEO changed shape")
        heo = eog_override
    return CalibrationBatch(
        eeg=np.asarray(eeg, dtype=np.float64),
        eog=np.asarray(heo, dtype=np.float64),
        participant=record.participant,
        source_record=str(target_for(record.participant, record.session).relative_path),
        sampling_rate=float(target_rate),
    )


def _flatten(windows: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [windows[index, ..., : int(count)] for index, count in enumerate(valid)],
        axis=-1,
    )


def _artifact_mask(query) -> np.ndarray:
    heo = _flatten(query.heo_windows, query.valid_samples)[0].astype(np.float64)
    heo = (heo - np.mean(heo)) / max(float(np.std(heo)), 1.0e-8)
    blinks = _flatten(query.blink_windows[:, None], query.valid_samples)[0]
    raw = blinks != 0
    if not np.any(raw):
        raw = np.abs(heo) >= 2.0
    radius = max(1, int(round(0.25 * query.sampling_rate)))
    expanded = np.convolve(raw.astype(np.int8), np.ones(2 * radius + 1), mode="same") > 0
    if not np.any(expanded) or not np.any(~expanded):
        raise ValueError("Eye-BCI external artifact mask has no two-regime support")
    return expanded


def _task_preservation(
    observed: np.ndarray,
    restored: np.ndarray,
    cues: np.ndarray,
    *,
    sampling_rate: int,
    clean_mask: np.ndarray,
) -> dict[str, float | None]:
    observed_clean = observed[:, clean_mask]
    restored_clean = restored[:, clean_mask]
    observed_psd = np.abs(np.fft.rfft(observed_clean, axis=1)) ** 2
    restored_psd = np.abs(np.fft.rfft(restored_clean, axis=1)) ** 2
    denominator = float(np.linalg.norm(observed_psd))
    spectral = None if denominator <= 1.0e-12 else 1.0 - float(
        np.linalg.norm(restored_psd - observed_psd) / denominator
    )
    onsets = np.flatnonzero((cues[1:] != cues[:-1]) & (cues[1:] != 0)) + 1
    before = int(round(0.2 * sampling_rate))
    after = int(round(0.8 * sampling_rate))
    onsets = onsets[(onsets >= before) & (onsets + after <= observed.shape[1])]
    cue_correlation = None
    if onsets.size:
        observed_erp = np.mean(
            np.stack([observed[:, onset - before : onset + after] for onset in onsets]), axis=0
        )
        restored_erp = np.mean(
            np.stack([restored[:, onset - before : onset + after] for onset in onsets]), axis=0
        )
        left = observed_erp.ravel() - float(observed_erp.mean())
        right = restored_erp.ravel() - float(restored_erp.mean())
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator > 1.0e-12:
            cue_correlation = float(np.dot(left, right) / denominator)
    return {
        "clean_interval_spectral_preservation": spectral,
        "cue_locked_erp_correlation": cue_correlation,
        "cue_onsets": int(onsets.size),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_eye_bci_fold(
    config: dict[str, Any], *, run_dir: Path, device: torch.device
) -> dict[str, Any]:
    if device.type != "cuda":
        raise RuntimeError("Eye-BCI full fold requires CUDA")
    eye = config["eye_bci"]
    root = Path(eye["root"])
    prior_config_path = Path(config["clean_prior_data"]["prior_config"])
    prior_config = yaml.safe_load(prior_config_path.read_text(encoding="utf-8"))
    checkpoint = Path(config["clean_prior_data"]["checkpoint"])
    prior, _ = load_best_prior(prior_config, checkpoint, device)
    for parameter in prior.parameters():
        parameter.requires_grad_(False)

    normalization = fit_outer_training_normalization(
        root,
        eye["training_participants"],
        session=eye["session"],
        seconds_per_participant=float(eye["calibration_end_seconds"]),
    )
    held_out_participant = str(eye["test_participants"][0])
    test_record = read_eye_bci_record(
        root, target_for(held_out_participant, eye["session"])
    )
    split_path = Path(config["outputs"]["split_manifest"])
    write_eye_bci_split_manifest(
        split_path, config=config, sampling_rate=test_record.sampling_rate
    )
    run_split = run_dir / "split_manifest.csv"
    run_split.write_text(split_path.read_text(encoding="utf-8"), encoding="utf-8")
    output_root = Path(config["outputs"]["root"])
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "outer_train_normalization.json").write_text(
        json.dumps(
            {
                "mean": normalization.mean,
                "standard_deviation": normalization.standard_deviation,
                "participants": list(normalization.participants),
                "samples": normalization.samples,
                "semantics": "outer_training_participants_first_30_seconds_only",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    target_rate = int(config["preprocessing"]["target_sampling_rate"])
    query = prepare_eye_bci_query(
        test_record,
        query_start_seconds=float(eye["query_start_seconds"]),
        target_sampling_rate=target_rate,
        window_samples=int(config["preprocessing"]["window_samples"]),
        normalization=normalization,
    )
    p0_config = _p0_config(config, target_rate)
    matching_batch = _calibration_batch(
        test_record,
        normalization,
        seconds=float(eye["calibration_end_seconds"]),
        target_rate=target_rate,
    )
    matching = fit_p0(
        matching_batch,
        p0_config,
        movement_threshold=float(config["p0"]["movement_threshold"]),
    )
    wrong_record = read_eye_bci_record(
        root,
        target_for(str(eye["training_participants"][0]), eye["session"]),
        seconds=float(eye["calibration_end_seconds"]),
    )
    wrong = fit_p0(
        _calibration_batch(
            wrong_record,
            normalization,
            seconds=float(eye["calibration_end_seconds"]),
            target_rate=target_rate,
        ),
        p0_config,
        movement_threshold=float(config["p0"]["movement_threshold"]),
    )
    shuffled_reference = block_shuffle_reference(
        matching_batch.eog,
        block_samples=2 * target_rate,
        seed=int(config["seed"]),
    )
    shuffled = fit_p0(
        CalibrationBatch(
            eeg=matching_batch.eeg,
            eog=shuffled_reference,
            participant=matching_batch.participant,
            source_record=f"{matching_batch.source_record}:shuffled",
            sampling_rate=matching_batch.sampling_rate,
        ),
        p0_config,
        movement_threshold=float(config["p0"]["movement_threshold"]),
    )
    development_outcomes: list[P0FitOutcome] = []
    for participant in eye["training_participants"][:8]:
        record = read_eye_bci_record(
            root,
            target_for(str(participant), eye["session"]),
            seconds=float(eye["calibration_end_seconds"]),
        )
        development_outcomes.append(
            fit_p0(
                _calibration_batch(
                    record,
                    normalization,
                    seconds=float(eye["calibration_end_seconds"]),
                    target_rate=target_rate,
                ),
                p0_config,
                movement_threshold=float(config["p0"]["movement_threshold"]),
            )
        )
    population_operator = population_source_transfer(development_outcomes, target_rank=1)
    outcomes = {
        "matching_p0": matching,
        "population_p0": population_operator,
        "wrong_participant_p0": wrong,
        "shuffled_calibration_p0": shuffled,
    }

    observed_flat = _flatten(query.eeg_windows, query.valid_samples)
    heo_flat = _flatten(query.heo_windows, query.valid_samples)
    heo_mean = float(np.mean(heo_flat))
    heo_scale = max(float(np.std(heo_flat)), 1.0e-8)
    standardized_heo = query.heo_windows.copy()
    for index, valid in enumerate(query.valid_samples):
        standardized_heo[index, :, : int(valid)] = (
            standardized_heo[index, :, : int(valid)] - heo_mean
        ) / heo_scale
        standardized_heo[index, :, int(valid) :] = 0.0
    artifact_mask = _artifact_mask(query)
    cues = _flatten(query.cue_windows[:, None], query.valid_samples)[0]
    inference = PopulationOnlyInference(prior)
    one_step = InformationMatchedOneStep(prior)
    methods = [
        "POP",
        "matching_p0",
        "population_p0",
        "wrong_participant_p0",
        "shuffled_calibration_p0",
        "information_matched_one_step",
    ]
    seeds = [int(value) for value in config["sampling"]["seeds"]]
    ddim_steps = int(config["sampling"]["ddim_steps"])
    chunk_windows = 4

    def restore(method: str, seed: int) -> tuple[np.ndarray, float, int | None, int, str | None, str | None]:
        restored = np.empty_like(query.eeg_windows, dtype=np.float32)
        latency_total = 0.0
        peak = 0
        evaluations = 0
        fallback = None
        failure = None
        method_outcome = outcomes.get(method)
        if method == "information_matched_one_step":
            method_outcome = matching
        if method != "POP" and (method_outcome is None or method_outcome.transfer is None):
            fallback = "POP"
            failure = ";".join(method_outcome.reasons) if method_outcome else "missing_operator"
        for start in range(0, query.eeg_windows.shape[0], chunk_windows):
            stop = min(start + chunk_windows, query.eeg_windows.shape[0])
            y = torch.as_tensor(query.eeg_windows[start:stop], device=device)
            eog = torch.as_tensor(standardized_heo[start:stop], device=device)
            attenuation = attenuation_from_external_reference(
                eog,
                scale=float(config["observation"]["attenuation_scale"]),
                floor=float(config["observation"]["attenuation_floor"]),
            )
            population = population_state_only(
                y,
                attenuation=attenuation,
                base_precision=float(config["observation"]["population_precision"]),
            )
            chunk_seed = seed + start * 1000003
            initial_noise = inference.make_initial_noise(population, seed=chunk_seed)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            begin = time.perf_counter()
            if method == "POP" or fallback == "POP":
                result = inference.sample(
                    population, initial_noise=initial_noise, ddim_steps=ddim_steps
                )
                evaluations += ddim_steps + 1
            elif method == "information_matched_one_step":
                _, context = matched_population_and_context_states(
                    y,
                    attenuation=attenuation,
                    projector=matching.transfer.projector,
                    base_precision=float(config["observation"]["context_precision"]),
                )
                result = one_step.restore(
                    observation=y,
                    channel_precision=context.precision,
                    seed=chunk_seed,
                    timestep=100,
                )
                evaluations += 1
            else:
                def factory(projector=method_outcome.transfer.projector):
                    _, context = matched_population_and_context_states(
                        y,
                        attenuation=attenuation,
                        projector=projector,
                        base_precision=float(config["observation"]["context_precision"]),
                    )
                    return context

                result = inference.sample_cgdr(
                    population,
                    rho=float(config["observation"]["rho"]),
                    calibration_accepted=True,
                    context_state_factory=factory,
                    initial_noise=initial_noise,
                    ddim_steps=ddim_steps,
                )
                evaluations += ddim_steps + 1
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                peak = max(peak, int(torch.cuda.max_memory_allocated(device)))
            latency_total += time.perf_counter() - begin
            restored[start:stop] = result.cpu().numpy()
        return restored, latency_total, peak, evaluations, fallback, failure

    rows: list[dict[str, Any]] = []
    condition_outputs: dict[str, list[np.ndarray]] = {method: [] for method in methods}
    condition_runtime: dict[str, list[tuple[float, int, int]]] = {method: [] for method in methods}
    condition_fallback: dict[str, tuple[str | None, str | None]] = {}
    for method in methods:
        outcome = matching if method == "information_matched_one_step" else outcomes.get(method)
        projector = outcome.transfer.projector if outcome and outcome.transfer else None
        for seed in seeds:
            restored_windows, latency, peak, evaluations, fallback, failure = restore(method, seed)
            condition_outputs[method].append(restored_windows)
            condition_runtime[method].append((latency, peak or 0, evaluations))
            condition_fallback[method] = (fallback, failure)
            restored_flat = _flatten(restored_windows, query.valid_samples)
            task = _task_preservation(
                observed_flat,
                restored_flat,
                cues,
                sampling_rate=target_rate,
                clean_mask=~artifact_mask,
            )
            rows.append(
                evaluate_context(
                    ContextIdentity(
                        dataset_id="eye_bci",
                        source_id="S01/Sess01/ME011",
                        participant_id="S01",
                        outer_fold=eye["outer_fold"],
                        session_id=eye["session"],
                        context_id="ME_query_after_35s",
                        method_id=method,
                        operator_source="none" if method == "POP" else method,
                        seed=seed,
                    ),
                    status="success" if failure is None else "rolled_back",
                    observed=observed_flat,
                    restored=restored_flat,
                    clean=None,
                    sampling_rate=target_rate,
                    estimated_projector=projector,
                    artifact_mask=artifact_mask,
                    predicted_artifact_mask=artifact_mask,
                    clean_mask=~artifact_mask,
                    fallback_method_id=fallback,
                    failure_reason=failure,
                    runtime=RuntimeEvaluation(
                        latency_seconds=latency,
                        peak_memory_bytes=peak,
                        function_evaluations=evaluations,
                        model_forward_evaluations=evaluations,
                    ),
                    extra_fields={
                        **task,
                        "paired_waveform_target_available": False,
                        "calibration_seconds": float(eye["calibration_end_seconds"]),
                        "query_windows": int(query.valid_samples.size),
                        "query_samples": int(query.valid_samples.sum()),
                        "outer_training_participants": len(eye["training_participants"]),
                        "validation_participants": len(eye["validation_participants"]),
                        "test_participants": len(eye["test_participants"]),
                    },
                )
            )
        mean_windows = np.mean(np.stack(condition_outputs[method]), axis=0)
        mean_flat = _flatten(mean_windows, query.valid_samples)
        runtimes = condition_runtime[method]
        fallback, failure = condition_fallback[method]
        task = _task_preservation(
            observed_flat,
            mean_flat,
            cues,
            sampling_rate=target_rate,
            clean_mask=~artifact_mask,
        )
        rows.append(
            evaluate_context(
                ContextIdentity(
                    dataset_id="eye_bci",
                    source_id="S01/Sess01/ME011",
                    participant_id="S01",
                    outer_fold=eye["outer_fold"],
                    session_id=eye["session"],
                    context_id="ME_query_after_35s_posterior_mean",
                    method_id=method,
                    operator_source="none" if method == "POP" else method,
                    seed=None,
                ),
                status="success" if failure is None else "rolled_back",
                observed=observed_flat,
                restored=mean_flat,
                clean=None,
                sampling_rate=target_rate,
                estimated_projector=projector,
                artifact_mask=artifact_mask,
                predicted_artifact_mask=artifact_mask,
                clean_mask=~artifact_mask,
                fallback_method_id=fallback,
                failure_reason=failure,
                runtime=RuntimeEvaluation(
                    latency_seconds=float(np.mean([item[0] for item in runtimes])),
                    peak_memory_bytes=max(item[1] for item in runtimes),
                    function_evaluations=int(np.mean([item[2] for item in runtimes])),
                    model_forward_evaluations=int(np.mean([item[2] for item in runtimes])),
                ),
                extra_fields={
                    **task,
                    "paired_waveform_target_available": False,
                    "calibration_seconds": float(eye["calibration_end_seconds"]),
                    "query_windows": int(query.valid_samples.size),
                    "query_samples": int(query.valid_samples.sum()),
                    "outer_training_participants": len(eye["training_participants"]),
                    "validation_participants": len(eye["validation_participants"]),
                    "test_participants": len(eye["test_participants"]),
                },
            )
        )
        # Keep only the aggregate row; seed-level restored arrays are no longer
        # needed once their metrics and posterior mean have been formed.
        condition_outputs[method].clear()

    metrics_path = Path(config["outputs"]["metrics"])
    _write_csv(metrics_path, rows)
    primary = [row for row in rows if row["seed"] is None]
    summary = {
        "status": "completed",
        "scientific_label": config["scientific_label"],
        "outer_fold": eye["outer_fold"],
        "split": {
            "training_participants": len(eye["training_participants"]),
            "validation_participants": len(eye["validation_participants"]),
            "test_participants": len(eye["test_participants"]),
            "held_out": "S01/Sess01/ME011",
            "calibration_query_disjoint": True,
        },
        "clean_waveform_metrics": "N/A_no_clean_target",
        "query_windows": int(query.valid_samples.size),
        "query_samples": int(query.valid_samples.sum()),
        "methods": {
            row["method_id"]: {
                "status": row["status"],
                "artifact_attenuation_db": row.get("artifact_attenuation_db"),
                "d_perp_y": row.get("d_perp_y"),
                "clean_interval_preservation": row.get("clean_interval_preservation"),
                "clean_interval_spectral_preservation": row.get(
                    "clean_interval_spectral_preservation"
                ),
                "cue_locked_erp_correlation": row.get("cue_locked_erp_correlation"),
                "latency_seconds": row.get("latency_seconds"),
                "peak_memory_bytes": row.get("peak_memory_bytes"),
                "function_evaluations": row.get("function_evaluations"),
            }
            for row in primary
        },
        "metrics": str(metrics_path),
        "split_manifest": str(split_path),
    }
    summary_path = output_root / "result_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (run_dir / "eye_fold_status.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "result_summary": str(summary_path),
                "metrics": str(metrics_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary
