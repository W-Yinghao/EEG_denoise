"""Natural Eye-BCI participant-held-out P0 fold."""

from __future__ import annotations

import csv
import json
import os
import signal
import tempfile
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
from eeg_cgdr.evaluation import (
    ContextIdentity,
    RuntimeEvaluation,
    artifact_attenuation,
    evaluate_context,
    projector_metrics,
)
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
    """Use blink annotations only to define evaluation intervals.

    The mask is never an attenuation or inference input. HEO remains the
    external observation input on the separate inference path.
    """

    blinks = _flatten(query.blink_windows[:, None], query.valid_samples)[0]
    raw = blinks != 0
    if not np.any(raw):
        raise ValueError("Eye-BCI query has no blink-annotation evaluation intervals")
    radius = max(1, int(round(0.25 * query.sampling_rate)))
    expanded = np.convolve(raw.astype(np.int8), np.ones(2 * radius + 1), mode="same") > 0
    if not np.any(expanded) or not np.any(~expanded):
        raise ValueError("Eye-BCI external artifact mask has no two-regime support")
    return expanded


def _contiguous_fixed_segments(mask: np.ndarray, length: int) -> list[slice]:
    """Return non-overlapping fixed-length slices wholly inside true runs."""

    if length < 2:
        raise ValueError("spectral segment length must be at least two samples")
    selection = np.asarray(mask, dtype=bool)
    if selection.ndim != 1:
        raise ValueError("non-artifact proxy mask must be one-dimensional")
    padded = np.pad(selection.astype(np.int8), (1, 1))
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    segments: list[slice] = []
    for run_start, run_stop in zip(starts, stops):
        for start in range(int(run_start), int(run_stop) - length + 1, length):
            segments.append(slice(start, start + length))
    return segments


def _task_preservation(
    observed: np.ndarray,
    restored: np.ndarray,
    cues: np.ndarray,
    *,
    sampling_rate: int,
    nonartifact_mask: np.ndarray,
    spectral_segment_seconds: float,
) -> dict[str, float | None]:
    # These are observation-relative proxies, not clean-target measurements.
    # Spectra are averaged over contiguous fixed-length runs; masked samples
    # are never concatenated into a discontinuous pseudo-signal.
    observed_proxy = observed[:, nonartifact_mask]
    restored_proxy = restored[:, nonartifact_mask]
    proxy_denominator = float(np.linalg.norm(observed_proxy))
    proxy_change = (
        None
        if proxy_denominator <= 1.0e-12
        else float(np.linalg.norm(restored_proxy - observed_proxy) / proxy_denominator)
    )
    left = observed_proxy.ravel() - float(observed_proxy.mean())
    right = restored_proxy.ravel() - float(restored_proxy.mean())
    correlation_denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    proxy_correlation = (
        None
        if correlation_denominator <= 1.0e-12
        else float(np.dot(left, right) / correlation_denominator)
    )

    segment_samples = int(round(float(spectral_segment_seconds) * sampling_rate))
    segments = _contiguous_fixed_segments(nonartifact_mask, segment_samples)
    spectral = None
    if segments:
        window = np.hanning(segment_samples).reshape(1, -1)
        frequency_bins = segment_samples // 2 + 1
        observed_psd = np.zeros((observed.shape[0], frequency_bins), dtype=np.float64)
        restored_psd = np.zeros_like(observed_psd)
        for segment in segments:
            observed_psd += np.abs(
                np.fft.rfft(observed[:, segment] * window, axis=-1)
            ) ** 2
            restored_psd += np.abs(
                np.fft.rfft(restored[:, segment] * window, axis=-1)
            ) ** 2
        observed_psd /= len(segments)
        restored_psd /= len(segments)
        denominator = float(np.linalg.norm(observed_psd))
        if denominator > 1.0e-12:
            spectral = 1.0 - float(
                np.linalg.norm(restored_psd - observed_psd) / denominator
            )

    onsets = np.flatnonzero((cues[1:] != cues[:-1]) & (cues[1:] != 0)) + 1
    before = int(round(0.2 * sampling_rate))
    after = int(round(0.8 * sampling_rate))
    onsets = onsets[(onsets >= before) & (onsets + after <= observed.shape[1])]
    proxy_onsets = np.asarray(
        [
            onset
            for onset in onsets
            if bool(np.all(nonartifact_mask[onset - before : onset + after]))
        ],
        dtype=np.int64,
    )
    cue_proxy_correlation = None
    if proxy_onsets.size:
        epoch_samples = before + after
        observed_erp = np.zeros((observed.shape[0], epoch_samples), dtype=np.float64)
        restored_erp = np.zeros_like(observed_erp)
        for onset in proxy_onsets:
            observed_erp += observed[:, onset - before : onset + after]
            restored_erp += restored[:, onset - before : onset + after]
        observed_erp /= proxy_onsets.size
        restored_erp /= proxy_onsets.size
        left = observed_erp.ravel() - float(observed_erp.mean())
        right = restored_erp.ravel() - float(restored_erp.mean())
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator > 1.0e-12:
            cue_proxy_correlation = float(np.dot(left, right) / denominator)
    return {
        "nonartifact_proxy_preservation": (
            None if proxy_change is None else 1.0 - proxy_change
        ),
        "nonartifact_proxy_relative_change": proxy_change,
        "nonartifact_proxy_correlation": proxy_correlation,
        "nonartifact_proxy_spectral_preservation": spectral,
        "nonartifact_proxy_spectral_segments": int(len(segments)),
        "nonartifact_proxy_spectral_segment_seconds": float(spectral_segment_seconds),
        "cue_locked_erp_observation_proxy_correlation": cue_proxy_correlation,
        "cue_onsets_total": int(onsets.size),
        "cue_onsets_nonartifact_proxy": int(proxy_onsets.size),
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(value, indent=2) + "\n")


def _atomic_write_npz(path: Path, *, mean_windows: np.ndarray, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            np.savez(
                stream,
                mean_windows=np.asarray(mean_windows, dtype=np.float32),
                metadata=np.asarray(json.dumps(metadata)),
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_method_progress(
    path: Path,
    *,
    method: str,
    contract: dict[str, Any],
    output_shape: tuple[int, ...],
) -> tuple[np.ndarray, dict[str, Any]]:
    if not path.is_file():
        return np.zeros(output_shape, dtype=np.float32), {
            "schema_version": 1,
            "method": method,
            "contract": contract,
            "completed_seeds": [],
            "seed_rows": [],
        }
    with np.load(path, allow_pickle=False) as archive:
        mean_windows = np.asarray(archive["mean_windows"], dtype=np.float32).copy()
        metadata = json.loads(str(archive["metadata"].item()))
    if metadata.get("schema_version") != 1:
        raise ValueError(f"unsupported Eye-BCI progress schema: {path}")
    if metadata.get("method") != method or metadata.get("contract") != contract:
        raise ValueError(f"Eye-BCI progress does not match frozen evaluation: {path}")
    if mean_windows.shape != output_shape:
        raise ValueError(f"Eye-BCI progress output shape changed: {path}")
    completed = [int(seed) for seed in metadata.get("completed_seeds", [])]
    rows = metadata.get("seed_rows", [])
    if len(completed) != len(set(completed)) or len(rows) != len(completed):
        raise ValueError(f"Eye-BCI progress seed ledger is inconsistent: {path}")
    if {int(row["seed"]) for row in rows} != set(completed):
        raise ValueError(f"Eye-BCI progress rows do not match completed seeds: {path}")
    return mean_windows, metadata


def _own_projector_diagnostics(
    common_projector: np.ndarray | None,
    own_projector: np.ndarray | None,
) -> dict[str, float | int | None]:
    fields = {
        "own_projector_overlap_with_matching": None,
        "own_projector_distance_from_matching": None,
        "own_projector_mean_angle_from_matching_deg": None,
        "own_projector_max_angle_from_matching_deg": None,
        "own_projector_rank": None,
        "matching_evaluation_projector_rank": None,
    }
    if common_projector is None or own_projector is None:
        return fields
    comparison = projector_metrics(
        estimated_projector=own_projector,
        oracle_projector=common_projector,
    )
    fields.update(
        {
            "own_projector_overlap_with_matching": comparison["overlap_fraction"],
            "own_projector_distance_from_matching": comparison["projector_distance"],
            "own_projector_mean_angle_from_matching_deg": comparison[
                "projector_mean_angle_deg"
            ],
            "own_projector_max_angle_from_matching_deg": comparison[
                "projector_max_angle_deg"
            ],
            "own_projector_rank": comparison["estimated_projector_rank"],
            "matching_evaluation_projector_rank": comparison["oracle_projector_rank"],
        }
    )
    return fields


def run_eye_bci_fold(
    config: dict[str, Any], *, run_dir: Path, device: torch.device
) -> dict[str, Any]:
    if device.type != "cuda":
        raise RuntimeError("Eye-BCI full fold requires CUDA")
    run_dir.mkdir(parents=True, exist_ok=True)
    output_root = Path(config["outputs"]["root"])
    output_root.mkdir(parents=True, exist_ok=True)
    progress_root = Path(
        config["outputs"].get("progress", output_root / "progress")
    )
    progress_root.mkdir(parents=True, exist_ok=True)
    stop_state: dict[str, Any] = {"requested": False, "signal": None}

    def request_boundary_stop(signum, _frame) -> None:
        # Signal handlers only set state.  All I/O happens after the active
        # method/seed has completed and its atomic cache has been published.
        stop_state["requested"] = True
        stop_state["signal"] = int(signum)

    previous_usr1 = signal.signal(signal.SIGUSR1, request_boundary_stop)

    def write_status(status: str, **fields: Any) -> None:
        payload = {
            "status": status,
            "experiment_id": config["experiment_id"],
            "progress_root": str(progress_root),
            **fields,
        }
        _atomic_write_json(run_dir / "eye_fold_status.json", payload)
        _atomic_write_json(progress_root / "status.json", payload)

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
    _atomic_write_text(run_split, split_path.read_text(encoding="utf-8"))
    _atomic_write_json(
        output_root / "outer_train_normalization.json",
        {
            "mean": normalization.mean,
            "standard_deviation": normalization.standard_deviation,
            "participants": list(normalization.participants),
            "samples": normalization.samples,
            "semantics": "outer_training_participants_first_30_seconds_only",
        },
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
    chunk_windows = int(config["evaluation"]["chunk_windows"])
    if chunk_windows < 1:
        raise ValueError("evaluation.chunk_windows must be positive")
    population_precision = float(config["observation"]["population_precision"])
    context_precision = float(config["observation"]["context_precision"])
    if population_precision != context_precision:
        raise ValueError(
            "POP and P0 must share one frozen base observation precision"
        )
    energy_scale = float(config["observation"]["energy_scale"])
    one_step_timestep = int(config["one_step"]["timestep"])
    if config["one_step"]["proximal_strength_source"] != "observation.energy_scale":
        raise ValueError("one-step must share the frozen observation energy scale")
    common_projector = (
        matching.transfer.projector if matching.transfer is not None else None
    )
    evaluation_config = config["evaluation"]
    if evaluation_config["common_subspace_basis"] != "matching_support_projector":
        raise ValueError("Eye-BCI evaluation basis must be matching support")
    spectral_segment_seconds = float(
        evaluation_config["nonartifact_proxy_spectral_segment_seconds"]
    )
    contract = {
        "experiment_id": config["experiment_id"],
        "evidence_status": config["evidence_status"],
        "prior_contract": dict(config["prior_contract"]),
        "semantics_revision": evaluation_config["semantics_revision"],
        "outer_fold": eye["outer_fold"],
        "held_out_record": "S01/Sess01/ME011",
        "query_shape": list(query.eeg_windows.shape),
        "query_samples": int(query.valid_samples.sum()),
        "seeds": seeds,
        "ddim_steps": ddim_steps,
        "output_rule": config["sampling"]["output_rule"],
        "population_precision": population_precision,
        "context_precision": context_precision,
        "energy_scale": energy_scale,
        "one_step_timestep": one_step_timestep,
        "one_step_proximal_strength": energy_scale,
        "rho": float(config["observation"]["rho"]),
        "attenuation_source": config["observation"]["attenuation_source"],
        "attenuation_floor": float(config["observation"]["attenuation_floor"]),
        "attenuation_scale": float(config["observation"]["attenuation_scale"]),
        "common_subspace_basis": evaluation_config["common_subspace_basis"],
        "chunk_windows": chunk_windows,
        "nonartifact_proxy_spectral_segment_seconds": spectral_segment_seconds,
        "methods": methods,
    }

    def restore(
        method: str, seed: int
    ) -> tuple[np.ndarray, float, int | None, int, str | None, str | None]:
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
            valid_time_mask = (
                torch.arange(y.shape[-1], device=device)[None, :]
                < torch.as_tensor(
                    query.valid_samples[start:stop], device=device
                )[:, None]
            )
            attenuation = attenuation_from_external_reference(
                eog,
                scale=float(config["observation"]["attenuation_scale"]),
                floor=float(config["observation"]["attenuation_floor"]),
            )
            population = population_state_only(
                y,
                attenuation=attenuation,
                base_precision=population_precision,
                energy_scale=energy_scale,
                valid_time_mask=valid_time_mask,
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
                evaluations += ddim_steps
            elif method == "information_matched_one_step":
                _, context = matched_population_and_context_states(
                    y,
                    attenuation=attenuation,
                    projector=matching.transfer.projector,
                    base_precision=context_precision,
                    energy_scale=energy_scale,
                    valid_time_mask=valid_time_mask,
                )
                result = one_step.restore(
                    observation=y,
                    channel_precision=context.precision,
                    seed=chunk_seed,
                    timestep=one_step_timestep,
                    proximal_strength=energy_scale,
                    valid_time_mask=context.valid_time_mask,
                )
                evaluations += 1
            else:
                def factory(projector=method_outcome.transfer.projector):
                    _, context = matched_population_and_context_states(
                        y,
                        attenuation=attenuation,
                        projector=projector,
                        base_precision=context_precision,
                        energy_scale=energy_scale,
                        valid_time_mask=valid_time_mask,
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
                evaluations += ddim_steps
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                peak = max(peak, int(torch.cuda.max_memory_allocated(device)))
            latency_total += time.perf_counter() - begin
            restored[start:stop] = result.cpu().numpy()
        return restored, latency_total, peak, evaluations, fallback, failure

    def method_outcome(method: str) -> P0FitOutcome | None:
        if method == "information_matched_one_step":
            return matching
        return outcomes.get(method)

    def evaluate_restoration(
        *,
        method: str,
        seed: int | None,
        restored_flat: np.ndarray,
        latency: float,
        peak: int,
        evaluations: int,
        fallback: str | None,
        failure: str | None,
        aggregate_across_seeds: bool,
    ) -> dict[str, Any]:
        outcome = method_outcome(method)
        own_projector = (
            outcome.transfer.projector
            if outcome is not None and outcome.transfer is not None
            else None
        )
        task = _task_preservation(
            observed_flat,
            restored_flat,
            cues,
            sampling_rate=target_rate,
            nonartifact_mask=~artifact_mask,
            spectral_segment_seconds=spectral_segment_seconds,
        )
        row = evaluate_context(
            ContextIdentity(
                dataset_id="eye_bci",
                source_id="S01/Sess01/ME011",
                participant_id="S01",
                outer_fold=eye["outer_fold"],
                session_id=eye["session"],
                context_id=(
                    "ME_query_after_35s_posterior_mean"
                    if aggregate_across_seeds
                    else "ME_query_after_35s"
                ),
                method_id=method,
                operator_source="none" if method == "POP" else method,
                seed=seed,
            ),
            status="success" if failure is None else "rolled_back",
            observed=observed_flat,
            restored=restored_flat,
            clean=None,
            sampling_rate=target_rate,
            # Every method is scored in the same frozen matching-support
            # subspace.  A method's own projector is diagnostic metadata only.
            estimated_projector=common_projector,
            artifact_mask=None,
            predicted_artifact_mask=None,
            clean_mask=None,
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
                **_own_projector_diagnostics(common_projector, own_projector),
                "paired_waveform_target_available": False,
                "external_artifact_reference_available": True,
                "external_artifact_reference_source": "blink_annotations_evaluation_only",
                "artifact_mask_is_model_prediction": False,
                "nonartifact_proxy_reference": "observed_EEG_no_clean_target",
                "aggregate_across_seeds": aggregate_across_seeds,
                "seed_samples_aggregated": len(seeds) if aggregate_across_seeds else 1,
                "calibration_seconds": float(eye["calibration_end_seconds"]),
                "query_windows": int(query.valid_samples.size),
                "query_samples": int(query.valid_samples.sum()),
                "outer_training_participants": len(eye["training_participants"]),
                "validation_participants": len(eye["validation_participants"]),
                "test_participants": len(eye["test_participants"]),
                "matching_evaluation_projector_eligible": common_projector is not None,
                "method_operator_eligible": own_projector is not None,
            },
        )
        # evaluate_context's legacy names are intentionally not reported as
        # clean-interval metrics here: Eye-BCI has no clean waveform target.
        row["clean_interval_preservation"] = None
        row["clean_interval_relative_change"] = None
        row["clean_interval_correlation"] = None
        if common_projector is not None:
            attenuation, attenuation_mode = artifact_attenuation(
                restored_flat,
                observed_flat,
                common_projector,
                clean=None,
                artifact_mask=artifact_mask,
            )
            row["artifact_attenuation_db"] = attenuation
            row["artifact_attenuation_mode"] = (
                f"{attenuation_mode}_external_reference_intervals"
            )
        else:
            row["artifact_attenuation_db"] = None
            row["artifact_attenuation_mode"] = None
        row["subspace_metric_basis"] = (
            "matching_support_projector"
            if common_projector is not None
            else "N/A_matching_support_ineligible"
        )
        return row

    completed_pairs = 0

    def stop_at_boundary(*, next_method: str, next_seed: int | None) -> None:
        if not stop_state["requested"]:
            return
        write_status(
            "checkpointed_for_resume",
            reason="SIGUSR1_deferred_to_method_seed_boundary",
            signal=stop_state["signal"],
            completed_method_seed_pairs=completed_pairs,
            next_method=next_method,
            next_seed=next_seed,
        )
        signal.signal(signal.SIGUSR1, previous_usr1)
        raise SystemExit(75)

    rows: list[dict[str, Any]] = []
    for method in methods:
        stop_at_boundary(next_method=method, next_seed=seeds[0])
        cache_path = progress_root / method / "streaming_mean.npz"
        mean_windows, progress = _load_method_progress(
            cache_path,
            method=method,
            contract=contract,
            output_shape=tuple(query.eeg_windows.shape),
        )
        completed = [int(seed) for seed in progress["completed_seeds"]]
        seed_rows_by_seed = {
            int(row["seed"]): row for row in progress["seed_rows"]
        }
        completed_pairs += len(completed)

        # The cumulative state is authoritative.  Recreate any marker whose
        # final atomic rename was interrupted after the cumulative save.
        for seed in completed:
            seed_marker = progress_root / method / f"seed_{seed}.json"
            if not seed_marker.is_file():
                _atomic_write_json(
                    seed_marker,
                    {
                        "status": "completed",
                        "method": method,
                        "seed": seed,
                        "state_cache": str(cache_path),
                        "posterior_mean_contribution_saved": True,
                        "metrics_row": seed_rows_by_seed[seed],
                    },
                )

        for seed in seeds:
            if seed in completed:
                continue
            stop_at_boundary(next_method=method, next_seed=seed)
            restored_windows, latency, peak, evaluations, fallback, failure = restore(
                method, seed
            )
            restored_flat = _flatten(restored_windows, query.valid_samples)
            row = evaluate_restoration(
                method=method,
                seed=seed,
                restored_flat=restored_flat,
                latency=latency,
                peak=peak or 0,
                evaluations=evaluations,
                fallback=fallback,
                failure=failure,
                aggregate_across_seeds=False,
            )

            previous_count = len(completed)
            mean_windows *= previous_count / float(previous_count + 1)
            np.multiply(
                restored_windows,
                1.0 / float(previous_count + 1),
                out=restored_windows,
            )
            mean_windows += restored_windows
            completed.append(seed)
            seed_rows_by_seed[seed] = row
            progress["completed_seeds"] = completed
            progress["seed_rows"] = [seed_rows_by_seed[value] for value in seeds if value in completed]
            _atomic_write_npz(
                cache_path,
                mean_windows=mean_windows,
                metadata=progress,
            )
            _atomic_write_json(
                progress_root / method / f"seed_{seed}.json",
                {
                    "status": "completed",
                    "method": method,
                    "seed": seed,
                    "state_cache": str(cache_path),
                    "posterior_mean_contribution_saved": True,
                    "metrics_row": row,
                },
            )
            completed_pairs += 1
            remaining_seeds = [value for value in seeds if value not in completed]
            if remaining_seeds:
                next_method = method
                next_seed: int | None = remaining_seeds[0]
            else:
                method_index = methods.index(method)
                next_method = (
                    methods[method_index + 1]
                    if method_index + 1 < len(methods)
                    else "finalize"
                )
                next_seed = seeds[0] if next_method != "finalize" else None
            stop_at_boundary(next_method=next_method, next_seed=next_seed)

        if completed != seeds:
            raise AssertionError(f"Eye-BCI method has incomplete configured seeds: {method}")
        seed_rows = [seed_rows_by_seed[seed] for seed in seeds]
        rows.extend(seed_rows)
        mean_flat = _flatten(mean_windows, query.valid_samples)
        fallback = seed_rows[0]["fallback_method_id"]
        failure = seed_rows[0]["failure_reason"]
        rows.append(
            evaluate_restoration(
                method=method,
                seed=None,
                restored_flat=mean_flat,
                latency=float(sum(float(row["latency_seconds"]) for row in seed_rows)),
                peak=max(int(row["peak_memory_bytes"]) for row in seed_rows),
                evaluations=sum(int(row["function_evaluations"]) for row in seed_rows),
                fallback=fallback,
                failure=failure,
                aggregate_across_seeds=True,
            )
        )

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
        "subspace_metric_basis": (
            "matching_support_projector"
            if common_projector is not None
            else "N/A_matching_support_ineligible"
        ),
        "external_artifact_reference": (
            "blink annotations define evaluation intervals only; HEO is the separate "
            "external inference input and no IoU is reported"
        ),
        "nonartifact_metrics": "observation-relative preservation proxies",
        "query_windows": int(query.valid_samples.size),
        "query_samples": int(query.valid_samples.sum()),
        "methods": {
            row["method_id"]: {
                "status": row["status"],
                "artifact_attenuation_db": row.get("artifact_attenuation_db"),
                "d_perp_y": row.get("d_perp_y"),
                "nonartifact_proxy_preservation": row.get(
                    "nonartifact_proxy_preservation"
                ),
                "nonartifact_proxy_spectral_preservation": row.get(
                    "nonartifact_proxy_spectral_preservation"
                ),
                "cue_locked_erp_observation_proxy_correlation": row.get(
                    "cue_locked_erp_observation_proxy_correlation"
                ),
                "latency_seconds": row.get("latency_seconds"),
                "peak_memory_bytes": row.get("peak_memory_bytes"),
                "function_evaluations": row.get("function_evaluations"),
            }
            for row in primary
        },
        "metrics": str(metrics_path),
        "split_manifest": str(split_path),
        "progress_root": str(progress_root),
        "resume_behavior": (
            "skip completed method-seed markers and continue from each method's "
            "atomic streaming posterior mean"
        ),
    }
    summary_path = output_root / "result_summary.json"
    _atomic_write_json(summary_path, summary)
    write_status(
        "completed",
        result_summary=str(summary_path),
        metrics=str(metrics_path),
        completed_method_seed_pairs=completed_pairs,
    )
    signal.signal(signal.SIGUSR1, previous_usr1)
    return summary
