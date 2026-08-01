"""GPU execution for the repaired Klados source-record mechanism audit.

This module is deliberately separate from the historical full-fold runner.  A
source record is the statistical unit: no participant identity is inferred
from the ordering of the Klados arrays.  Development chooses only a sampler
mechanism and trust radius using POP versus the paired-query oracle geometry;
the calibration-operator controls are evaluated once, on the untouched
records, under that frozen choice.
"""

from __future__ import annotations

import csv
import json
import os
import signal
import subprocess
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import torch
from scipy.signal import welch
from torch.optim import AdamW

from eeg_cgdr.data.klados import KladosRecord, load_klados_records
from eeg_cgdr.data.mechanism import (
    KLADOS_DEVELOPMENT_RECORDS,
    KLADOS_TRAIN_RECORDS,
    KLADOS_UNTOUCHED_RECORDS,
    ChannelNormalizer,
    KladosMechanismRecord,
    fit_channel_normalizer,
    prepare_mechanism_record,
    prepare_population_calibration,
    select_records,
)
from eeg_cgdr.experiments.klados import block_shuffle_reference
from eeg_cgdr.experiments.mechanism_training import (
    _build_prior,
    load_population_projector,
    load_repaired_prior,
)
from eeg_cgdr.inference import (
    CalibrationContextProjector,
    DatasetPopulationProjector,
    GuidanceStabilityConfig,
    PopulationOnlyInference,
    RepairedSamplerRunner,
    SamplerMechanism,
    dataset_population_and_context_states,
    dataset_population_state,
    frame_attenuation_from_external_reference,
    rho_interpolated_precision_state,
    sampler_candidate,
)
from eeg_cgdr.operators import CalibrationBatch, P0Config, P0FitOutcome, fit_p0


DATASET_ID = "klados_bamidis_v4"
MONTAGE_ID = "klados_v4_19ch_native_order_256hz"
REAL_RECORD_FD_EPSILON = 1.0e-3
REAL_RECORD_FD_RELATIVE_TOLERANCE = 5.0e-2
METRIC_SEMANTICS = {
    "e_parallel": "norm(P_star (x_hat-x)) / norm(P_star x)",
    "e_perp": "norm(Q_star (x_hat-x)) / norm(Q_star x)",
    "d_perp_y": "norm(Q_star (x_hat-y)) / norm(Q_star y)",
    "overlap_fraction": "norm(P_star x)^2 / norm(x)^2",
    "artifact_normalized_parallel_error": (
        "norm(P_star (x_hat-x)) / norm(P_star (y-x))"
    ),
    "rrmse": "norm(x_hat-x) / norm(x)",
    "correlation": "tanh(mean(channelwise Fisher-z Pearson correlation))",
    "psd_distortion": (
        "mean_channel norm(PSD(x_hat)-PSD(x))/norm(PSD(x)), Welch 0.5-40 Hz"
    ),
    "artifact_attenuation": (
        "20log10(norm(P_star(y-x))/norm(P_star(x_hat-x))) on high-EOG frames"
    ),
    "clean_interval_preservation": (
        "1-norm(x_hat-y)/norm(y) on low-EOG frames"
    ),
}


@dataclass(frozen=True)
class _OperatorArm:
    source: str
    projector: Optional[np.ndarray]
    p0_outcome: Optional[P0FitOutcome]
    calibration_id: str
    query_clean_target_used: bool = False

    @property
    def eligible(self) -> bool:
        if self.projector is None:
            return False
        return self.p0_outcome is None or self.p0_outcome.transfer is not None

    @property
    def failure_reason(self) -> str:
        if self.eligible:
            return ""
        if self.p0_outcome is None:
            return "missing_projector"
        return ";".join(self.p0_outcome.reasons) or "ineligible_p0"


def _loader_config(config: dict[str, Any]) -> dict[str, Any]:
    raw = config["klados"]
    return {
        "data_root": raw["data_root"],
        "files": {
            "contaminated": raw["contaminated"],
            "clean": raw["clean"],
            "heog": raw["heog"],
            "veog": raw["veog"],
        },
        "official_description": {"records": 54},
    }


def _p0_config(config: dict[str, Any], *, bootstrap: bool = True) -> P0Config:
    raw = config["p0"]
    replicates = int(raw["bootstrap_replicates"]) if bootstrap else 0
    return P0Config(
        target_rank=int(raw["target_rank"]),
        ridge_lambda=float(raw["ridge_lambda"]),
        maximum_reference_condition=float(raw["maximum_reference_condition"]),
        minimum_singular_ratio=float(raw["minimum_singular_ratio"]),
        minimum_movement_coverage=float(raw["minimum_movement_coverage"]),
        bootstrap_replicates=replicates,
        bootstrap_block_samples=int(raw["bootstrap_block_samples"]),
        minimum_bootstrap_success=(
            float(raw["minimum_bootstrap_success"]) if bootstrap else 0.0
        ),
        maximum_bootstrap_median_distance=(
            float(raw["maximum_bootstrap_median_distance"])
            if bootstrap
            else float("inf")
        ),
        maximum_bootstrap_q90_distance=(
            float(raw["maximum_bootstrap_q90_distance"])
            if bootstrap
            else float("inf")
        ),
        seed=int(config["seed"]),
    )


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _git_commit() -> str:
    code_root = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=code_root,
        check=True,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    if len(value) != 40:
        raise ValueError("git rev-parse did not return a full commit ID")
    return value


def _atomic_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to publish an empty mechanism metrics table")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({field for row in rows for field in row})
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _trust_slug(value: float) -> str:
    return f"{float(value):.12g}".replace("-", "m").replace(".", "p")


def _method_id(operator_source: str, candidate: str, trust_radius: float) -> str:
    return f"{operator_source}__{candidate}__trust_{_trust_slug(trust_radius)}"


def _continuous(windows: np.ndarray, samples: int) -> np.ndarray:
    value = np.asarray(windows)
    if value.ndim != 3:
        raise ValueError("windowed EEG must have shape (B,C,L)")
    return value.transpose(1, 0, 2).reshape(value.shape[1], -1)[:, :samples]


def _relative_norm(numerator: np.ndarray, denominator: np.ndarray) -> float:
    scale = float(np.linalg.norm(denominator))
    if not np.isfinite(scale) or scale <= 1.0e-12:
        raise ValueError("metric denominator is numerically zero")
    value = float(np.linalg.norm(numerator) / scale)
    if not np.isfinite(value):
        raise ValueError("non-finite relative norm")
    return value


def _fisher_channel_correlation(restored: np.ndarray, clean: np.ndarray) -> float:
    values: list[float] = []
    for channel in range(clean.shape[0]):
        left = restored[channel] - restored[channel].mean()
        right = clean[channel] - clean[channel].mean()
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator <= 1.0e-12:
            continue
        correlation = float(np.dot(left, right) / denominator)
        values.append(float(np.arctanh(np.clip(correlation, -0.999999, 0.999999))))
    if not values:
        raise ValueError("correlation is undefined for every EEG channel")
    return float(np.tanh(np.mean(values)))


def _psd_distortion(
    restored: np.ndarray, clean: np.ndarray, sampling_rate: float
) -> float:
    samples = clean.shape[1]
    if samples < 8:
        raise ValueError("record is too short for a PSD metric")
    frequencies, clean_psd = welch(
        clean, fs=sampling_rate, nperseg=min(512, samples), axis=-1
    )
    _, restored_psd = welch(
        restored, fs=sampling_rate, nperseg=min(512, samples), axis=-1
    )
    band = (frequencies >= 0.5) & (frequencies <= 40.0)
    if not np.any(band):
        raise ValueError("registered PSD band is empty")
    channel_values = []
    for channel in range(clean.shape[0]):
        channel_values.append(
            _relative_norm(
                restored_psd[channel, band] - clean_psd[channel, band],
                clean_psd[channel, band],
            )
        )
    return float(np.mean(channel_values))


def _projector_comparison(
    estimated: Optional[np.ndarray], oracle: np.ndarray
) -> dict[str, Any]:
    if estimated is None:
        return {
            "projector_distance": "",
            "projector_max_angle_degrees": "",
            "projector_overlap_fraction": "",
        }
    left = np.asarray(estimated, dtype=np.float64)
    right = np.asarray(oracle, dtype=np.float64)
    left_values, left_vectors = np.linalg.eigh(0.5 * (left + left.T))
    right_values, right_vectors = np.linalg.eigh(0.5 * (right + right.T))
    left_basis = left_vectors[:, left_values > 0.5]
    right_basis = right_vectors[:, right_values > 0.5]
    rank_sum = left_basis.shape[1] + right_basis.shape[1]
    distance = (
        0.0
        if rank_sum == 0
        else float(np.linalg.norm(left - right, ord="fro") / np.sqrt(rank_sum))
    )
    if left_basis.shape[1] and right_basis.shape[1]:
        singular = np.linalg.svd(right_basis.T @ left_basis, compute_uv=False)
        angles = np.arccos(np.clip(singular, 0.0, 1.0))
        missing = abs(left_basis.shape[1] - right_basis.shape[1])
        if missing:
            angles = np.concatenate([angles, np.full(missing, np.pi / 2.0)])
        maximum = float(np.rad2deg(angles).max())
        overlap = float(
            np.clip(np.trace(right @ left) / right_basis.shape[1], 0.0, 1.0)
        )
    else:
        maximum = 0.0 if rank_sum == 0 else 90.0
        overlap = 0.0 if right_basis.shape[1] else 1.0
    return {
        "projector_distance": distance,
        "projector_max_angle_degrees": maximum,
        "projector_overlap_fraction": overlap,
    }


def _mechanism_metrics(
    restored: np.ndarray,
    *,
    observed: np.ndarray,
    clean: np.ndarray,
    oracle_projector: np.ndarray,
    estimated_projector: Optional[np.ndarray],
    artifact_mask: np.ndarray,
    sampling_rate: float,
) -> dict[str, Any]:
    restored = np.asarray(restored, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.float64)
    clean = np.asarray(clean, dtype=np.float64)
    if restored.shape != observed.shape or clean.shape != observed.shape:
        raise ValueError("mechanism metric signals are not aligned")
    projection = np.asarray(oracle_projector, dtype=np.float64)
    complement = np.eye(projection.shape[0], dtype=np.float64) - projection
    error = restored - clean
    artifact = observed - clean
    parallel_clean = projection @ clean
    perpendicular_clean = complement @ clean
    parallel_error = projection @ error
    perpendicular_error = complement @ error
    overlap = float(
        np.linalg.norm(parallel_clean) ** 2
        / max(np.linalg.norm(clean) ** 2, 1.0e-24)
    )
    delta_snr = float(
        20.0
        * np.log10(
            max(np.linalg.norm(artifact), 1.0e-12)
            / max(np.linalg.norm(error), 1.0e-12)
        )
    )
    artifact_selection = np.asarray(artifact_mask, dtype=bool)
    if artifact_selection.shape != (observed.shape[1],):
        raise ValueError("artifact mask is not aligned to the continuous query")
    if np.any(artifact_selection):
        artifact_reference = projection @ artifact[:, artifact_selection]
        artifact_residual = projection @ error[:, artifact_selection]
        artifact_attenuation: float | str = float(
            20.0
            * np.log10(
                max(np.linalg.norm(artifact_reference), 1.0e-12)
                / max(np.linalg.norm(artifact_residual), 1.0e-12)
            )
        )
    else:
        artifact_attenuation = ""
    clean_selection = ~np.asarray(artifact_mask, dtype=bool)
    if np.any(clean_selection):
        preservation = 1.0 - _relative_norm(
            restored[:, clean_selection] - observed[:, clean_selection],
            observed[:, clean_selection],
        )
    else:
        preservation = ""
    metrics = {
        "e_parallel": _relative_norm(parallel_error, parallel_clean),
        "e_perp": _relative_norm(perpendicular_error, perpendicular_clean),
        "d_perp_y": _relative_norm(
            complement @ (restored - observed), complement @ observed
        ),
        "overlap_fraction": overlap,
        "artifact_normalized_parallel_error": _relative_norm(
            parallel_error, projection @ artifact
        ),
        "rrmse": _relative_norm(error, clean),
        "time_rrmse": _relative_norm(error, clean),
        "correlation": _fisher_channel_correlation(restored, clean),
        "psd_distortion": _psd_distortion(restored, clean, sampling_rate),
        "delta_snr_db": delta_snr,
        "artifact_attenuation": artifact_attenuation,
        "clean_interval_preservation": preservation,
    }
    metrics.update(_projector_comparison(estimated_projector, projection))
    for name in (
        "e_parallel",
        "e_perp",
        "d_perp_y",
        "overlap_fraction",
        "rrmse",
        "correlation",
        "psd_distortion",
    ):
        if not np.isfinite(float(metrics[name])):
            raise ValueError(f"non-finite core mechanism metric {name}")
    return metrics


def _oracle_projector(
    prepared: KladosMechanismRecord, target_rank: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """Take the rank-r thin SVD of paired-query artifact in EEG space.

    This deliberately does not use EOG or a ridge transfer: adding either
    would mix P0-estimation error into the clean-target mechanism upper bound.
    """

    artifact = np.asarray(
        prepared.observed_continuous - prepared.clean_continuous, dtype=np.float64
    )
    basis_full, singular_values, _ = np.linalg.svd(artifact, full_matrices=False)
    rank = min(int(target_rank), basis_full.shape[1])
    if rank < 1 or singular_values[rank - 1] <= 0.0:
        raise ValueError("query-derived oracle artifact is rank deficient")
    basis = basis_full[:, :rank]
    projector = basis @ basis.T
    predicted = projector @ artifact
    return projector, {
        "rank": rank,
        "singular_values": singular_values.tolist(),
        "paired_query_clean_target_used": True,
        "mechanism_upper_bound_only": True,
        "rank_truncation_relative_residual": float(
            np.linalg.norm(artifact - predicted)
            / max(np.linalg.norm(artifact), 1.0e-12)
        ),
        "construction": "thin_SVD_of_valid_query_artifact_y_minus_clean",
        "label": "query_clean_target_upper_bound",
    }


def _standardized_query_eog(
    prepared: KladosMechanismRecord,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Consume the support-standardized EOG prepared by the data boundary.

    ``prepare_mechanism_record`` has already fitted moments on the 30-second
    support, transformed support and query with those moments, and zeroed
    padded query frames.  Re-standardizing here from query statistics would be
    leakage and is therefore explicitly rejected by these checks.
    """

    support = np.asarray(prepared.calibration.eog, dtype=np.float64)
    if not np.allclose(support.mean(axis=1), 0.0, atol=1.0e-10, rtol=0.0):
        raise ValueError("P0 support EOG is not support-channel centered")
    if not np.allclose(support.std(axis=1), 1.0, atol=1.0e-10, rtol=0.0):
        raise ValueError("P0 support EOG is not support-channel standardized")
    moments_mean = np.asarray(prepared.eog_calibration_mean, dtype=np.float64)
    moments_scale = np.asarray(
        prepared.eog_calibration_standard_deviation, dtype=np.float64
    )
    if moments_mean.shape != (support.shape[0], 1) or moments_scale.shape != (
        support.shape[0],
        1,
    ):
        raise ValueError("stored support EOG moments have invalid shape")
    if not np.isfinite(moments_mean).all() or np.any(moments_scale <= 0.0):
        raise ValueError("stored support EOG moments are invalid")
    standardized_windows = np.asarray(prepared.eog_windows, dtype=np.float64).copy()
    standardized_windows *= prepared.valid_time_weight[:, None, :]
    standardized_continuous = np.asarray(
        prepared.eog_continuous, dtype=np.float64
    )
    magnitude = np.sqrt(np.mean(np.square(standardized_continuous), axis=0))
    return standardized_windows, standardized_continuous, magnitude


def _fit_calibration_arm(
    source: str,
    batch: CalibrationBatch,
    config: dict[str, Any],
) -> _OperatorArm:
    if config["p0"].get("reference_standardization") != "support_channel_zscore":
        raise ValueError("P0 reference_standardization must be support_channel_zscore")
    reference = np.asarray(batch.eog, dtype=np.float64)
    if not np.allclose(reference.mean(axis=1), 0.0, atol=1.0e-10, rtol=0.0):
        raise ValueError("P0 calibration reference is not support centered")
    if not np.allclose(reference.std(axis=1), 1.0, atol=1.0e-10, rtol=0.0):
        raise ValueError("P0 calibration reference is not support standardized")
    outcome = fit_p0(
        batch,
        _p0_config(config, bootstrap=True),
        movement_threshold=float(config["p0"]["movement_threshold"]),
    )
    projector = None if outcome.transfer is None else outcome.transfer.projector
    return _OperatorArm(source, projector, outcome, batch.source_record)


def _record_arms(
    config: dict[str, Any],
    *,
    prepared: KladosMechanismRecord,
    records: Sequence[KladosRecord],
    normalizer: ChannelNormalizer,
    population: DatasetPopulationProjector,
    include_controls: bool,
) -> tuple[dict[str, _OperatorArm], np.ndarray, dict[str, Any], int]:
    oracle, oracle_diagnostics = _oracle_projector(
        prepared, int(config["p0"]["target_rank"])
    )
    arms: dict[str, _OperatorArm] = {
        "population_projector": _OperatorArm(
            "population_projector",
            np.asarray(population.projector, dtype=np.float64),
            None,
            population.source,
        ),
        "oracle_projector": _OperatorArm(
            "oracle_projector",
            oracle,
            None,
            f"sim{prepared.source_record:02d}_paired_query_upper_bound",
            query_clean_target_used=True,
        ),
    }
    wrong_record_id = -1
    if include_controls:
        matching = _fit_calibration_arm("matching_p0", prepared.calibration, config)
        offset = int(config["klados"].get("wrong_source_offset", 1))
        training_pool = tuple(int(value) for value in KLADOS_TRAIN_RECORDS)
        wrong_index = (prepared.source_record - 1 + offset) % len(training_pool)
        wrong_record_id = training_pool[wrong_index]
        if wrong_record_id == prepared.source_record:
            raise ValueError("wrong-source control resolved to the matching record")
        wrong_prepared = prepare_mechanism_record(
            records[wrong_record_id - 1],
            normalizer,
            source_rate=int(config["klados"]["source_sampling_rate"]),
            target_rate=int(config["preprocessing"]["target_sampling_rate"]),
            window_samples=int(config["preprocessing"]["window_samples"]),
            calibration_seconds=float(config["klados"]["calibration_seconds"]),
            guard_seconds=float(config["klados"]["guard_seconds"]),
        )
        wrong = _fit_calibration_arm(
            "wrong_source_p0", wrong_prepared.calibration, config
        )
        shuffled_eog = block_shuffle_reference(
            prepared.calibration.eog,
            block_samples=max(
                1, int(round(prepared.calibration.sampling_rate * 2.0))
            ),
            seed=int(config["seed"]) + prepared.source_record,
        )
        shuffled_batch = replace(
            prepared.calibration,
            eog=shuffled_eog,
            source_record=f"sim{prepared.source_record:02d}_block_shuffled",
        )
        shuffled = _fit_calibration_arm(
            "shuffled_calibration_p0", shuffled_batch, config
        )
        arms.update(
            {
                "matching_p0": matching,
                "wrong_source_p0": wrong,
                "shuffled_calibration_p0": shuffled,
            }
        )
    return arms, oracle, oracle_diagnostics, wrong_record_id


def _candidate_steps(config: dict[str, Any], candidate: str) -> int:
    if candidate == "M1":
        return int(config["sampling"]["warm_start_steps"])
    if candidate == "M5":
        return 1
    return int(config["sampling"]["ddim_steps"])


def _trace_excerpt(trace: Iterable[Any]) -> list[dict[str, Any]]:
    values = [asdict(item) for item in trace]
    selected = [
        item
        for item in values
        if item["checkpoint_label"] in {"first", "middle", "last", "first_middle_last"}
    ]
    return selected


def _diagnostic_fields(arm: Optional[_OperatorArm]) -> dict[str, Any]:
    if arm is None:
        return {
            "p0_eligible": "",
            "p0_rank": "",
            "p0_ridge_lambda": "",
            "p0_reference_rank": "",
            "p0_reference_condition": "",
            "p0_movement_coverage": "",
            "p0_singular_ratio": "",
            "p0_singular_values": "",
            "p0_bootstrap_success_rate": "",
            "p0_bootstrap_median_distance": "",
            "p0_bootstrap_q90_distance": "",
        }
    outcome = arm.p0_outcome
    if outcome is None:
        return {
            "p0_eligible": "",
            "p0_rank": int(round(np.trace(arm.projector))) if arm.projector is not None else "",
            "p0_ridge_lambda": "",
            "p0_reference_rank": "",
            "p0_reference_condition": "",
            "p0_movement_coverage": "",
            "p0_singular_ratio": "",
            "p0_singular_values": "",
            "p0_bootstrap_success_rate": "",
            "p0_bootstrap_median_distance": "",
            "p0_bootstrap_q90_distance": "",
        }
    diagnostics = outcome.transfer.diagnostics if outcome.transfer is not None else {}
    return {
        "p0_eligible": outcome.transfer is not None,
        "p0_rank": outcome.transfer.rank if outcome.transfer is not None else "",
        "p0_ridge_lambda": diagnostics.get("ridge_lambda", ""),
        "p0_reference_rank": diagnostics.get("reference_rank", ""),
        "p0_reference_condition": diagnostics.get("reference_condition", ""),
        "p0_movement_coverage": diagnostics.get("movement_coverage", ""),
        "p0_singular_ratio": diagnostics.get("singular_ratio", ""),
        "p0_singular_values": json.dumps(diagnostics.get("singular_values", [])),
        "p0_bootstrap_success_rate": diagnostics.get("bootstrap_success_rate", ""),
        "p0_bootstrap_median_distance": diagnostics.get(
            "bootstrap_median_projector_distance", ""
        ),
        "p0_bootstrap_q90_distance": diagnostics.get(
            "bootstrap_q90_projector_distance", ""
        ),
    }


def _base_row(
    *,
    partition: str,
    source_record: int,
    seed: int,
    aggregate: bool,
    method_id: str,
    operator_source: str,
    candidate: str,
    trust_radius: float,
    arm: Optional[_OperatorArm],
) -> dict[str, Any]:
    row = {
        "partition": partition,
        "source_record": source_record,
        "records_are_participants": False,
        "seed": seed,
        "aggregate_across_seeds": aggregate,
        "method_id": method_id,
        "operator_source": operator_source,
        "sampler_candidate": candidate,
        "trust_radius": float(trust_radius),
        "status": "failed",
        "failure_reason": "",
        "fallback_used": False,
        "fallback_method_id": "",
        "calibration_id": (
            arm.calibration_id
            if arm is not None
            else "population_projector_Pi0"
            if operator_source == "population_only"
            else ""
        ),
        "query_clean_target_used_by_method": bool(
            arm.query_clean_target_used if arm is not None else False
        ),
        "calibration_seconds": 30.0,
        "calibration_duration_axis_complete": False,
        "latency_seconds": "",
        "peak_memory_mb": "",
        "function_evaluations": "",
        "network_calls_total": "",
    }
    row.update(_diagnostic_fields(arm))
    for field in (
        "e_parallel",
        "e_perp",
        "d_perp_y",
        "overlap_fraction",
        "artifact_normalized_parallel_error",
        "rrmse",
        "time_rrmse",
        "correlation",
        "psd_distortion",
        "delta_snr_db",
        "projector_distance",
        "projector_max_angle_degrees",
        "projector_overlap_fraction",
        "artifact_attenuation",
        "clean_interval_preservation",
    ):
        row[field] = ""
    return row


def _state_for_chunk(
    *,
    observation: torch.Tensor,
    standardized_eog: torch.Tensor,
    valid_weight: torch.Tensor,
    population_projector: DatasetPopulationProjector,
    arm: Optional[_OperatorArm],
    config: dict[str, Any],
) -> tuple[Any, np.ndarray]:
    attenuation = frame_attenuation_from_external_reference(
        standardized_eog,
        scale=float(config["observation"]["attenuation_scale"]),
        floor=float(config["observation"]["attenuation_floor"]),
    )
    population_state = dataset_population_state(
        observation,
        attenuation=attenuation,
        valid_weight=valid_weight,
        population_projector=population_projector,
        base_precision=float(config["observation"]["base_precision"]),
        energy_scale=float(config["observation"]["energy_scale"]),
    )
    rho = float(config["observation"]["rho"])
    if not np.isfinite(rho) or not 0.0 <= rho <= 1.0:
        raise ValueError("observation.rho must be finite and lie in [0,1]")
    if arm is None or rho == 0.0:
        return population_state, np.asarray(population_projector.projector)
    if not arm.eligible or arm.projector is None:
        raise ValueError(f"operator {arm.source} is not eligible")
    context_projector = CalibrationContextProjector(
        dataset_id=population_projector.dataset_id,
        montage_id=population_projector.montage_id,
        projector=arm.projector,
        calibration_id=arm.calibration_id,
    )
    population_again, context_state = dataset_population_and_context_states(
        observation,
        attenuation=attenuation,
        valid_weight=valid_weight,
        population_projector=population_projector,
        context_projector=context_projector,
        base_precision=float(config["observation"]["base_precision"]),
        energy_scale=float(config["observation"]["energy_scale"]),
    )
    state = rho_interpolated_precision_state(
        population_again,
        rho=rho,
        calibration_accepted=True,
        context_state_factory=lambda: context_state,
    )
    if arm.source == "population_projector":
        if not torch.equal(state.precision, population_state.precision):
            raise AssertionError("PiC=Pi0 did not recover exact POP precision")
        if not torch.equal(state.valid_time_mask, population_state.valid_time_mask):
            raise AssertionError("PiC=Pi0 did not recover exact POP valid-time mask")
    return state, np.asarray(arm.projector)


def _sample_one_seed(
    *,
    prior: Any,
    prepared: KladosMechanismRecord,
    standardized_eog_windows: np.ndarray,
    population_projector: DatasetPopulationProjector,
    arm: Optional[_OperatorArm],
    candidate: str,
    trust_radius: float,
    seed: int,
    config: dict[str, Any],
    device: torch.device,
    override_steps: Optional[int] = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    stability = GuidanceStabilityConfig(
        normalize_by_residual_dimension=bool(
            config["observation"]["residual_dimension_normalization"]
        ),
        trust_radius_ratio=float(trust_radius),
    )
    runner = RepairedSamplerRunner(PopulationOnlyInference(prior, stability=stability))
    mechanism = sampler_candidate(candidate).mechanism
    batch_size = int(config["sampling"].get("inference_batch_size", 8))
    if batch_size < 1:
        raise ValueError("sampling.inference_batch_size must be positive")
    restored_parts: list[np.ndarray] = []
    trace_excerpt: list[dict[str, Any]] = []
    network_calls = 0
    expected_steps = _candidate_steps(config, candidate)
    if override_steps is not None and candidate != "M5":
        expected_steps = int(override_steps)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for start in range(0, prepared.observed_windows.shape[0], batch_size):
        stop = min(start + batch_size, prepared.observed_windows.shape[0])
        observation = torch.as_tensor(
            prepared.observed_windows[start:stop],
            dtype=torch.float32,
            device=device,
        )
        eog = torch.as_tensor(
            standardized_eog_windows[start:stop],
            dtype=torch.float32,
            device=device,
        )
        weight = torch.as_tensor(
            prepared.valid_time_weight[start:stop],
            dtype=torch.float32,
            device=device,
        )
        state, consistency_projector = _state_for_chunk(
            observation=observation,
            standardized_eog=eog,
            valid_weight=weight,
            population_projector=population_projector,
            arm=arm,
            config=config,
        )
        stream_seed = int(seed) + 104729 * start
        result = runner.run(
            mechanism,
            state,
            seed=stream_seed,
            ddim_steps=expected_steps,
            projector=torch.as_tensor(
                consistency_projector, dtype=observation.dtype, device=device
            ),
            warm_start_timestep=(
                int(config["sampling"]["warm_start_timestep"])
                if mechanism == SamplerMechanism.M1
                else None
            ),
            one_step_timestep=(
                int(
                    config["sampling"].get(
                        "one_step_timestep",
                        config["sampling"]["warm_start_timestep"],
                    )
                )
                if mechanism == SamplerMechanism.M5
                else None
            ),
            proximal_strength=float(config["sampling"]["proximal_strength"]),
        )
        if result.network_evaluations != expected_steps:
            raise AssertionError(
                f"{candidate} recorded {result.network_evaluations} calls, "
                f"expected {expected_steps}"
            )
        network_calls += result.network_evaluations
        restored_parts.append(result.restored.detach().cpu().numpy())
        if not trace_excerpt:
            trace_excerpt = _trace_excerpt(result.trace)
        del observation, eog, weight, state, result
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory = float(torch.cuda.max_memory_allocated(device) / (1024.0**2))
    else:
        peak_memory = 0.0
    latency = time.perf_counter() - started
    windows = np.concatenate(restored_parts, axis=0)
    restored = _continuous(windows, prepared.observed_continuous.shape[1])
    return restored, {
        "latency_seconds": latency,
        "peak_memory_mb": peak_memory,
        "function_evaluations": expected_steps,
        "network_calls_total": network_calls,
        "trace_scope": "first_inference_chunk_first_middle_last",
        "trace": trace_excerpt,
    }


def _run_stochastic_method(
    *,
    partition: str,
    source_record: int,
    method_id: str,
    operator_source: str,
    candidate: str,
    trust_radius: float,
    arm: Optional[_OperatorArm],
    prior: Any,
    prepared: KladosMechanismRecord,
    standardized_eog_windows: np.ndarray,
    artifact_mask: np.ndarray,
    population_projector: DatasetPopulationProjector,
    oracle_projector: np.ndarray,
    seeds: Sequence[int],
    config: dict[str, Any],
    device: torch.device,
    override_steps: Optional[int] = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(seeds) != 5:
        raise ValueError("scientific mechanism methods require exactly five seeds")
    seed_rows: list[dict[str, Any]] = []
    successful_outputs: list[np.ndarray] = []
    trace_by_seed: dict[str, Any] = {}
    observed = prepared.observed_continuous
    clean = prepared.clean_continuous
    estimated_projector = (
        np.asarray(population_projector.projector)
        if arm is None
        else arm.projector
    )
    for seed in seeds:
        row = _base_row(
            partition=partition,
            source_record=source_record,
            seed=int(seed),
            aggregate=False,
            method_id=method_id,
            operator_source=operator_source,
            candidate=candidate,
            trust_radius=trust_radius,
            arm=arm,
        )
        restored, runtime = _sample_one_seed(
            prior=prior,
            prepared=prepared,
            standardized_eog_windows=standardized_eog_windows,
            population_projector=population_projector,
            arm=arm,
            candidate=candidate,
            trust_radius=trust_radius,
            seed=int(seed),
            config=config,
            device=device,
            override_steps=override_steps,
        )
        row.update(
            _mechanism_metrics(
                restored,
                observed=observed,
                clean=clean,
                oracle_projector=oracle_projector,
                estimated_projector=estimated_projector,
                artifact_mask=artifact_mask,
                sampling_rate=prepared.sampling_rate,
            )
        )
        row.update({
            key: runtime[key]
            for key in (
                "latency_seconds",
                "peak_memory_mb",
                "function_evaluations",
                "network_calls_total",
            )
        })
        row["status"] = "success"
        successful_outputs.append(restored)
        trace_by_seed[str(seed)] = {
            "trace_scope": runtime["trace_scope"],
            "points": runtime["trace"],
        }
        seed_rows.append(row)
    aggregate = _base_row(
        partition=partition,
        source_record=source_record,
        seed=-1,
        aggregate=True,
        method_id=method_id,
        operator_source=operator_source,
        candidate=candidate,
        trust_radius=trust_radius,
        arm=arm,
    )
    if len(successful_outputs) == len(seeds):
        posterior_mean = np.mean(np.stack(successful_outputs, axis=0), axis=0)
        aggregate.update(
            _mechanism_metrics(
                posterior_mean,
                observed=observed,
                clean=clean,
                oracle_projector=oracle_projector,
                estimated_projector=estimated_projector,
                artifact_mask=artifact_mask,
                sampling_rate=prepared.sampling_rate,
            )
        )
        aggregate["status"] = "success"
        aggregate["latency_seconds"] = float(
            sum(float(row["latency_seconds"]) for row in seed_rows)
        )
        aggregate["peak_memory_mb"] = float(
            max(float(row["peak_memory_mb"]) for row in seed_rows)
        )
        aggregate["function_evaluations"] = int(
            sum(int(row["function_evaluations"]) for row in seed_rows)
        )
        aggregate["network_calls_total"] = int(
            sum(int(row["network_calls_total"]) for row in seed_rows)
        )
    else:
        aggregate["failure_reason"] = (
            f"posterior_mean_requires_all_{len(seeds)}_configured_seeds;"
            f"successful={len(successful_outputs)}"
        )
    return seed_rows + [aggregate], trace_by_seed


def _fallback_rows(
    *,
    population_rows: Sequence[dict[str, Any]],
    partition: str,
    source_record: int,
    method_id: str,
    operator_source: str,
    candidate: str,
    trust_radius: float,
    arm: _OperatorArm,
    population_method_id: str,
) -> list[dict[str, Any]]:
    if len(population_rows) != 6:
        raise ValueError("POP fallback requires five seed rows and one posterior mean row")
    rows: list[dict[str, Any]] = []
    for source_row in population_rows:
        row = dict(source_row)
        row.update(
            _base_row(
                partition=partition,
                source_record=source_record,
                seed=int(source_row["seed"]),
                aggregate=bool(source_row["aggregate_across_seeds"]),
                method_id=method_id,
                operator_source=operator_source,
                candidate=candidate,
                trust_radius=trust_radius,
                arm=arm,
            )
        )
        # Preserve the exact POP output metrics while reporting rejection as a
        # separate eligibility/fallback event.
        for field, value in source_row.items():
            if field in METRIC_SEMANTICS or field in {
                "d_perp_y",
                "artifact_normalized_parallel_error",
                "time_rrmse",
                "delta_snr_db",
                "projector_distance",
                "projector_max_angle_degrees",
                "projector_overlap_fraction",
                "artifact_attenuation",
                "clean_interval_preservation",
                "latency_seconds",
                "peak_memory_mb",
                "function_evaluations",
                "network_calls_total",
            }:
                row[field] = value
        # A rejected calibration still produces the exact sampler-matched POP
        # waveform.  ``fallback_POP`` is a valid output status (not a failed
        # waveform) and lets aggregation preserve a separate fallback rate.
        if source_row["status"] == "success":
            row["status"] = "fallback_POP"
        else:
            row["status"] = "failed"
        row["fallback_used"] = True
        row["fallback_method_id"] = population_method_id
        source_failure = str(source_row.get("failure_reason", "")).strip()
        row["failure_reason"] = ";".join(
            value for value in (arm.failure_reason, source_failure) if value
        )
        rows.append(row)
    return rows


def _deterministic_rows(
    *,
    partition: str,
    prepared: KladosMechanismRecord,
    oracle: np.ndarray,
    artifact_mask: np.ndarray,
    trust_radius: float,
) -> list[dict[str, Any]]:
    outputs = {
        "corrupted_identity": (
            "corrupted_identity",
            lambda: prepared.observed_continuous.copy(),
            None,
        ),
        "oracle_orthogonal_subtraction": (
            "oracle_orthogonal_subtraction",
            lambda: (np.eye(oracle.shape[0]) - oracle)
            @ prepared.observed_continuous,
            oracle,
        ),
    }
    rows = []
    for method_id, (source, restore, estimated) in outputs.items():
        started = time.perf_counter()
        restored = restore()
        latency = time.perf_counter() - started
        row = _base_row(
            partition=partition,
            source_record=prepared.source_record,
            seed=-1,
            aggregate=True,
            method_id=method_id,
            operator_source=source,
            candidate="deterministic",
            trust_radius=trust_radius,
            arm=None,
        )
        row.update(
            _mechanism_metrics(
                restored,
                observed=prepared.observed_continuous,
                clean=prepared.clean_continuous,
                oracle_projector=oracle,
                estimated_projector=estimated,
                artifact_mask=artifact_mask,
                sampling_rate=prepared.sampling_rate,
            )
        )
        row.update(
            {
                "status": "success",
                "query_clean_target_used_by_method": (
                    method_id == "oracle_orthogonal_subtraction"
                ),
                "latency_seconds": latency,
                "peak_memory_mb": 0.0,
                "function_evaluations": 0,
                "network_calls_total": 0,
            }
        )
        rows.append(row)
    return rows


def _load_progress(
    path: Path,
    *,
    partition: str,
    source_record: int,
    plan_contract: dict[str, Any],
) -> dict[str, Any]:
    if not path.is_file():
        return {
            "schema_version": 2,
            "partition": partition,
            "source_record": source_record,
            "records_are_participants": False,
            "plan_contract": plan_contract,
            "completed_method_ids": [],
            "rows": [],
            "trace_summaries": {},
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 2
        or payload.get("partition") != partition
        or int(payload.get("source_record", -1)) != source_record
        or payload.get("records_are_participants") is not False
        or payload.get("plan_contract") != plan_contract
    ):
        raise ValueError(
            "mechanism progress file does not match this frozen source-record plan"
        )
    return payload


def _publish_method(
    progress_path: Path,
    progress: dict[str, Any],
    method_id: str,
    rows: Sequence[dict[str, Any]],
    trace: Any,
) -> None:
    if method_id in set(progress["completed_method_ids"]):
        return
    progress["rows"].extend(rows)
    progress["completed_method_ids"].append(method_id)
    progress["trace_summaries"][method_id] = trace
    _atomic_json(progress_path, progress)


def _rows_for_method(progress: dict[str, Any], method_id: str) -> list[dict[str, Any]]:
    return [row for row in progress["rows"] if row["method_id"] == method_id]


def _record_result_directory(
    config: dict[str, Any], partition: str, source_record: int
) -> Path:
    return Path(config["outputs"]["root"]) / partition / f"sim{source_record:02d}"


def _record_method_plan(
    config: dict[str, Any],
    *,
    partition: str,
) -> tuple[list[float], list[str], list[str]]:
    candidates = [sampler_candidate(value).candidate_id for value in config["sampling"]["candidates"]]
    if len(candidates) != 6 or set(candidates) != {f"M{i}" for i in range(6)}:
        raise ValueError("development mechanism audit requires exactly M0-M5")
    if partition == "development":
        trusts = [float(value) for value in config["observation"]["trust_radius_candidates"]]
        # Only this paired-query upper-bound scan is allowed to select a mechanism.
        return trusts, candidates, ["population_only", "oracle_projector"]
    frozen_path = Path(config["outputs"]["frozen_choice"])
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if not str(frozen.get("status", "")).startswith("frozen"):
        raise ValueError("untouched audit requires a frozen development choice")
    candidate = sampler_candidate(str(frozen["selected_sampler_candidate"])).candidate_id
    trust = float(frozen["selected_trust_radius"])
    return [trust], [candidate], [
        "population_only",
        "matching_p0",
        "population_projector",
        "wrong_source_p0",
        "shuffled_calibration_p0",
        "oracle_projector",
    ]


def _run_scientific_record(
    config: dict[str, Any],
    *,
    partition: str,
    source_record: int,
    run_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    prior, normalizer = load_repaired_prior(config, device=device)
    population = load_population_projector(config)
    records = load_klados_records(_loader_config(config))
    partition_records: Sequence[int] = (
        KLADOS_DEVELOPMENT_RECORDS
        if partition == "development"
        else KLADOS_UNTOUCHED_RECORDS
    )
    if source_record not in partition_records:
        raise ValueError(f"sim{source_record:02d} is not in the frozen {partition} partition")
    prepared = prepare_mechanism_record(
        records[source_record - 1],
        normalizer,
        source_rate=int(config["klados"]["source_sampling_rate"]),
        target_rate=int(config["preprocessing"]["target_sampling_rate"]),
        window_samples=int(config["preprocessing"]["window_samples"]),
        calibration_seconds=float(config["klados"]["calibration_seconds"]),
        guard_seconds=float(config["klados"]["guard_seconds"]),
    )
    include_controls = partition == "untouched"
    arms, oracle, oracle_diagnostics, wrong_record = _record_arms(
        config,
        prepared=prepared,
        records=records,
        normalizer=normalizer,
        population=population,
        include_controls=include_controls,
    )
    standardized_eog, _, eog_magnitude = _standardized_query_eog(prepared)
    artifact_mask = eog_magnitude >= float(
        config["observation"]["artifact_eog_z_threshold"]
    )
    trusts, candidates, operator_sources = _record_method_plan(
        config, partition=partition
    )
    seeds = tuple(int(value) for value in config["sampling"]["seeds"])
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError("mechanism audit requires five distinct configured seeds")
    expected_method_ids = {
        "corrupted_identity",
        "oracle_orthogonal_subtraction",
        *(
            _method_id(source, candidate, trust)
            for trust in trusts
            for candidate in candidates
            for source in operator_sources
        ),
    }
    plan_contract = {
        "git_commit": _git_commit(),
        "experiment_seed": int(config["seed"]),
        "audit_protocol": config["audit_protocol"],
        "scientific_label": config["scientific_label"],
        "partition": partition,
        "source_record": source_record,
        "method_ids": sorted(expected_method_ids),
        "operator_sources": list(operator_sources),
        "sampler_candidates": list(candidates),
        "trust_radii": list(trusts),
        "seeds": list(seeds),
        "calibration_seconds": float(config["klados"]["calibration_seconds"]),
        "guard_seconds": float(config["klados"]["guard_seconds"]),
        "wrong_source_offset": int(config["klados"].get("wrong_source_offset", 1)),
        "wrong_source_pool": list(KLADOS_TRAIN_RECORDS),
        "preprocessing": dict(config["preprocessing"]),
        "p0": dict(config["p0"]),
        "observation": dict(config["observation"]),
        "sampling": dict(config["sampling"]),
        "prior_checkpoint": str(config["outputs"]["best_checkpoint"]),
        "population_state": str(config["outputs"]["population_state"]),
    }

    destination = _record_result_directory(config, partition, source_record)
    progress_path = destination / "method_progress.json"
    progress = _load_progress(
        progress_path,
        partition=partition,
        source_record=source_record,
        plan_contract=plan_contract,
    )
    stop_requested = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    old_handler = signal.signal(signal.SIGUSR1, request_stop)
    try:
        deterministic_trust = trusts[0]
        for row in _deterministic_rows(
            partition=partition,
            prepared=prepared,
            oracle=oracle,
            artifact_mask=artifact_mask,
            trust_radius=deterministic_trust,
        ):
            method_id = str(row["method_id"])
            if method_id not in set(progress["completed_method_ids"]):
                _publish_method(progress_path, progress, method_id, [row], {})
            if stop_requested:
                break
        for trust_radius in trusts:
            if stop_requested:
                break
            for candidate in candidates:
                if stop_requested:
                    break
                population_method = _method_id(
                    "population_only", candidate, trust_radius
                )
                if population_method not in set(progress["completed_method_ids"]):
                    rows, trace = _run_stochastic_method(
                        partition=partition,
                        source_record=source_record,
                        method_id=population_method,
                        operator_source="population_only",
                        candidate=candidate,
                        trust_radius=trust_radius,
                        arm=None,
                        prior=prior,
                        prepared=prepared,
                        standardized_eog_windows=standardized_eog,
                        artifact_mask=artifact_mask,
                        population_projector=population,
                        oracle_projector=oracle,
                        seeds=seeds,
                        config=config,
                        device=device,
                    )
                    _publish_method(
                        progress_path, progress, population_method, rows, trace
                    )
                if stop_requested:
                    break
                for source in operator_sources:
                    if source == "population_only":
                        continue
                    method_id = _method_id(source, candidate, trust_radius)
                    if method_id in set(progress["completed_method_ids"]):
                        continue
                    arm = arms[source]
                    if not arm.eligible:
                        population_rows = _rows_for_method(progress, population_method)
                        rows = _fallback_rows(
                            population_rows=population_rows,
                            partition=partition,
                            source_record=source_record,
                            method_id=method_id,
                            operator_source=source,
                            candidate=candidate,
                            trust_radius=trust_radius,
                            arm=arm,
                            population_method_id=population_method,
                        )
                        trace = {"fallback": population_method, "reason": arm.failure_reason}
                    else:
                        rows, trace = _run_stochastic_method(
                            partition=partition,
                            source_record=source_record,
                            method_id=method_id,
                            operator_source=source,
                            candidate=candidate,
                            trust_radius=trust_radius,
                            arm=arm,
                            prior=prior,
                            prepared=prepared,
                            standardized_eog_windows=standardized_eog,
                            artifact_mask=artifact_mask,
                            population_projector=population,
                            oracle_projector=oracle,
                            seeds=seeds,
                            config=config,
                            device=device,
                        )
                    _publish_method(progress_path, progress, method_id, rows, trace)
                    if stop_requested:
                        break
    finally:
        signal.signal(signal.SIGUSR1, old_handler)

    completed_method_ids = set(progress["completed_method_ids"])
    unexpected = completed_method_ids - expected_method_ids
    if unexpected:
        raise ValueError(
            f"progress contains methods outside the frozen plan: {sorted(unexpected)}"
        )
    expected_methods = len(expected_method_ids)
    completed = len(completed_method_ids)
    status = (
        "completed"
        if completed_method_ids == expected_method_ids
        else "checkpointed_for_resume"
        if stop_requested
        else "incomplete"
    )
    summary = {
        "status": status,
        "partition": partition,
        "source_record": source_record,
        "records_are_participants": False,
        "scientific_label": config["scientific_label"],
        "calibration_seconds": float(config["klados"]["calibration_seconds"]),
        "calibration_duration_axis_complete": False,
        "query_start_seconds": prepared.query_start_seconds,
        "query_end_seconds": prepared.query_end_seconds,
        "query_windows": int(prepared.observed_windows.shape[0]),
        "valid_query_samples": int(prepared.observed_continuous.shape[1]),
        "reference_standardization": "support_channel_zscore",
        "query_reference_statistics_used": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "wrong_source_record": wrong_record if include_controls else None,
        "wrong_source_partition": "outer_training" if include_controls else None,
        "wrong_source_rule": (
            "KLADOS_TRAIN_RECORDS[(target_source_record-1+wrong_source_offset)"
            " % len(KLADOS_TRAIN_RECORDS)]"
            if include_controls
            else None
        ),
        "oracle_diagnostics": oracle_diagnostics,
        "metric_semantics": METRIC_SEMANTICS,
        "methods_expected": expected_methods,
        "methods_completed": completed,
        "five_seed_output_rule": "posterior_mean_of_seed_sample_tensors",
        "formal_G1_status": "NOT_RUN_BLOCKED",
    }
    destination.mkdir(parents=True, exist_ok=True)
    _atomic_json(destination / "result_summary.json", summary)
    run_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(run_dir / "record_status.json", summary)
    if status == "completed":
        _atomic_csv(destination / "metrics.csv", progress["rows"])
    elif status == "incomplete":
        raise RuntimeError(
            f"mechanism record stopped without signal: {completed}/{expected_methods} methods"
        )
    return summary


def _integration_population_projector(
    config: dict[str, Any], records: Sequence[KladosRecord], normalizer: ChannelNormalizer
) -> DatasetPopulationProjector:
    batches: list[CalibrationBatch] = []
    for record in select_records(records, KLADOS_TRAIN_RECORDS):
        batch = prepare_population_calibration(
            record,
            normalizer,
            source_rate=int(config["klados"]["source_sampling_rate"]),
            target_rate=int(config["preprocessing"]["target_sampling_rate"]),
        )
        batches.append(batch)
    if len(batches) != len(KLADOS_TRAIN_RECORDS):
        raise AssertionError("integration Pi0 did not consume sim01-sim30 exactly once")
    joint = CalibrationBatch(
        eeg=np.concatenate([batch.eeg for batch in batches], axis=1),
        eog=np.concatenate([batch.eog for batch in batches], axis=1),
        participant="outer_training_source_records",
        source_record="sim01_sim30_joint_population_integration",
        sampling_rate=float(config["preprocessing"]["target_sampling_rate"]),
    )
    outcome = fit_p0(
        joint,
        _p0_config(config, bootstrap=False),
        movement_threshold=float(config["p0"]["movement_threshold"]),
    )
    if outcome.transfer is None:
        raise RuntimeError(
            "integration joint all-training Pi0 is ineligible: "
            + ";".join(outcome.reasons)
        )
    return DatasetPopulationProjector(
        dataset_id=DATASET_ID,
        montage_id=MONTAGE_ID,
        projector=outcome.transfer.projector,
        source="integration_joint_all_training_source_records_sim01_sim30",
    )


def _real_record_directional_fd(
    config: dict[str, Any],
    *,
    prior: Any,
    prepared: KladosMechanismRecord,
    population: DatasetPopulationProjector,
    matching: _OperatorArm,
    device: torch.device,
) -> dict[str, Any]:
    """J1 finite difference through a real-window multichannel denoiser."""

    standardized_eog, _, _ = _standardized_query_eog(prepared)
    observation = torch.as_tensor(
        prepared.observed_windows[:1], dtype=torch.float32, device=device
    )
    eog = torch.as_tensor(
        standardized_eog[:1], dtype=torch.float32, device=device
    )
    weight = torch.as_tensor(
        prepared.valid_time_weight[:1], dtype=torch.float32, device=device
    )
    state, _ = _state_for_chunk(
        observation=observation,
        standardized_eog=eog,
        valid_weight=weight,
        population_projector=population,
        arm=matching,
        config=config,
    )
    timestep_value = int(config["diffusion"]["num_timesteps"]) // 2
    timesteps = torch.full((1,), timestep_value, device=device, dtype=torch.long)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(config["seed"]) + 99173)
    noise = torch.randn(
        observation.shape,
        device=device,
        dtype=observation.dtype,
        generator=generator,
    )
    x_t = prior.diffusion.q_sample(observation, timesteps, noise)
    inference = PopulationOnlyInference(
        prior,
        stability=GuidanceStabilityConfig(
            normalize_by_residual_dimension=bool(
                config["observation"]["residual_dimension_normalization"]
            ),
            trust_radius_ratio=float(
                config["observation"]["trust_radius_candidates"][0]
            ),
        ),
    )
    vjp = inference.full_energy_vjp(
        x_t,
        timesteps,
        energy=state.energy_per_sample,
        valid_time_mask=state.valid_time_mask,
    )
    gradient_rms = torch.sqrt(torch.mean(vjp.energy_vjp.square()))
    if not bool(torch.isfinite(gradient_rms)) or float(gradient_rms) <= 1.0e-12:
        raise AssertionError("real-record full energy VJP is zero or non-finite")
    direction = vjp.energy_vjp / gradient_rms
    analytic = float(torch.sum(vjp.energy_vjp * direction))

    def energy_at(value: torch.Tensor) -> float:
        with torch.no_grad():
            predicted_noise = prior.predict_noise(
                value, timesteps, valid_time_mask=state.valid_time_mask
            )
            clean = prior.predict_clean(
                value,
                timesteps,
                predicted_noise,
                valid_time_mask=state.valid_time_mask,
            )
            return float(state.energy_per_sample(clean).sum())

    epsilon = REAL_RECORD_FD_EPSILON
    plus = energy_at(x_t + epsilon * direction)
    minus = energy_at(x_t - epsilon * direction)
    finite_difference = (plus - minus) / (2.0 * epsilon)
    relative_error = abs(analytic - finite_difference) / max(
        abs(analytic), abs(finite_difference), 1.0e-6
    )
    if (
        not np.isfinite(relative_error)
        or relative_error > REAL_RECORD_FD_RELATIVE_TOLERANCE
    ):
        raise AssertionError(
            "real-record full-VJP finite difference failed: "
            f"analytic={analytic} finite_difference={finite_difference} "
            f"relative_error={relative_error}"
        )
    return {
        "source_record": prepared.source_record,
        "window_index": 0,
        "timestep": timestep_value,
        "epsilon": epsilon,
        "relative_tolerance": REAL_RECORD_FD_RELATIVE_TOLERANCE,
        "analytic_directional_derivative": analytic,
        "central_finite_difference": finite_difference,
        "relative_error": relative_error,
        "passed": True,
        "prior": "random_initialized_then_real_record_optimizer_updates_multichannel",
        "energy": "formal_rho_interpolated_population_context_observation_energy",
    }


def _run_sampler_integration(
    config: dict[str, Any], *, run_dir: Path, device: torch.device
) -> dict[str, Any]:
    """J1 real-record forward/backward/checkpoint and M0--M5 integration."""

    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    records = load_klados_records(_loader_config(config))
    normalizer = fit_channel_normalizer(records)
    prepared_records = [
        prepare_mechanism_record(
            records[record_id - 1],
            normalizer,
            source_rate=int(config["klados"]["source_sampling_rate"]),
            target_rate=int(config["preprocessing"]["target_sampling_rate"]),
            window_samples=int(config["preprocessing"]["window_samples"]),
            calibration_seconds=float(config["klados"]["calibration_seconds"]),
            guard_seconds=float(config["klados"]["guard_seconds"]),
        )
        for record_id in (1, 2)
    ]
    prior = _build_prior(config, device)
    optimizer = AdamW(prior.parameters(), lr=float(config["training"]["learning_rate"]))
    generator = torch.Generator(device=device)
    generator.manual_seed(int(config["seed"]) + 9000)
    batch_size = int(config["sampling"].get("inference_batch_size", 8))
    training_steps = 0
    prior.train()
    for prepared in prepared_records:
        for start in range(0, prepared.clean_windows.shape[0], batch_size):
            stop = min(start + batch_size, prepared.clean_windows.shape[0])
            clean = torch.as_tensor(
                prepared.clean_windows[start:stop], dtype=torch.float32, device=device
            )
            valid = torch.as_tensor(
                prepared.valid_time_weight[start:stop], dtype=torch.float32, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            loss = prior.training_loss(clean, generator=generator, valid_time_mask=valid)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("integration loss is non-finite")
            loss.backward()
            optimizer.step()
            training_steps += 1
    checkpoint = run_dir / "integration_checkpoint.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint.with_name(f".{checkpoint.name}.{os.getpid()}.tmp")
    torch.save(
        {
            "model_state": prior.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "training_steps": training_steps,
            "source_records": [1, 2],
        },
        temporary,
    )
    os.replace(temporary, checkpoint)
    reloaded = _build_prior(config, device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    reloaded.load_state_dict(payload["model_state"])
    reloaded_optimizer = AdamW(
        reloaded.parameters(), lr=float(config["training"]["learning_rate"])
    )
    reloaded_optimizer.load_state_dict(payload["optimizer_state"])
    resume_generator = torch.Generator(device=device)
    resume_generator.manual_seed(int(config["seed"]) + 9001)
    resume_stop = min(batch_size, prepared_records[0].clean_windows.shape[0])
    resume_clean = torch.as_tensor(
        prepared_records[0].clean_windows[:resume_stop],
        dtype=torch.float32,
        device=device,
    )
    resume_valid = torch.as_tensor(
        prepared_records[0].valid_time_weight[:resume_stop],
        dtype=torch.float32,
        device=device,
    )
    reloaded.train()
    reloaded_optimizer.zero_grad(set_to_none=True)
    resumed_loss = reloaded.training_loss(
        resume_clean,
        generator=resume_generator,
        valid_time_mask=resume_valid,
    )
    if not bool(torch.isfinite(resumed_loss)):
        raise FloatingPointError("integration resumed loss is non-finite")
    resumed_loss.backward()
    reloaded_optimizer.step()
    reloaded.eval()
    population = _integration_population_projector(config, records, normalizer)
    fd_matching = _fit_calibration_arm(
        "matching_p0", prepared_records[0].calibration, config
    )
    if not fd_matching.eligible:
        raise RuntimeError(
            "real-record finite-difference context P0 is ineligible: "
            f"{fd_matching.failure_reason}"
        )
    finite_difference = _real_record_directional_fd(
        config,
        prior=reloaded,
        prepared=prepared_records[0],
        population=population,
        matching=fd_matching,
        device=device,
    )

    integration_rows: list[dict[str, Any]] = []
    candidates = [sampler_candidate(value).candidate_id for value in config["sampling"]["candidates"]]
    trust = float(config["observation"]["trust_radius_candidates"][0])
    seed = int(config["sampling"]["seeds"][0])
    integration_steps = min(5, int(config["sampling"]["ddim_steps"]))
    for prepared in prepared_records:
        standardized_eog, _, eog_magnitude = _standardized_query_eog(prepared)
        artifact_mask = eog_magnitude >= float(
            config["observation"]["artifact_eog_z_threshold"]
        )
        oracle, _ = _oracle_projector(prepared, int(config["p0"]["target_rank"]))
        matching = _fit_calibration_arm("matching_p0", prepared.calibration, config)
        if not matching.eligible:
            raise RuntimeError(
                f"integration sim{prepared.source_record:02d} P0 ineligible: "
                f"{matching.failure_reason}"
            )
        for operator_source, arm in (
            ("population_only", None),
            ("matching_p0", matching),
        ):
            for candidate in candidates:
                restored, runtime = _sample_one_seed(
                    prior=reloaded,
                    prepared=prepared,
                    standardized_eog_windows=standardized_eog,
                    population_projector=population,
                    arm=arm,
                    candidate=candidate,
                    trust_radius=trust,
                    seed=seed,
                    config=config,
                    device=device,
                    override_steps=integration_steps,
                )
                row = {
                    "source_record": prepared.source_record,
                    "records_are_participants": False,
                    "operator_source": operator_source,
                    "sampler_candidate": candidate,
                    "status": "success",
                    **_mechanism_metrics(
                        restored,
                        observed=prepared.observed_continuous,
                        clean=prepared.clean_continuous,
                        oracle_projector=oracle,
                        estimated_projector=(
                            np.asarray(population.projector)
                            if arm is None
                            else matching.projector
                        ),
                        artifact_mask=artifact_mask,
                        sampling_rate=prepared.sampling_rate,
                    ),
                    **{key: runtime[key] for key in (
                        "latency_seconds",
                        "peak_memory_mb",
                        "function_evaluations",
                        "network_calls_total",
                    )},
                    "trace": runtime["trace"],
                }
                integration_rows.append(row)
    _atomic_json(run_dir / "sampler_integration.json", integration_rows)
    result = {
        "status": "completed",
        "real_source_records": [1, 2],
        "records_are_participants": False,
        "full_query_used": True,
        "optimizer_updates": training_steps,
        "checkpoint": str(checkpoint),
        "checkpoint_reloaded": True,
        "optimizer_state_reloaded": True,
        "resume_optimizer_updates": 1,
        "resumed_loss": float(resumed_loss.detach()),
        "real_record_full_vjp_finite_difference": finite_difference,
        "samplers_exercised": candidates,
        "branches_exercised": ["population_only", "matching_p0"],
        "integration_ddim_steps": integration_steps,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "scientific_result": False,
    }
    _atomic_json(run_dir / "result_summary.json", result)
    return result


def run_gpu_mechanism_stage(
    config: dict[str, Any],
    *,
    stage: str,
    run_dir: Path,
    device: torch.device,
    task_index: Optional[int],
) -> dict[str, Any]:
    """Run J1, one J3 development record, or one J5 untouched record."""

    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("GPU mechanism stage requires a scheduled CUDA allocation")
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if config["p0"].get("reference_standardization") != "support_channel_zscore":
        raise ValueError(
            "repaired mechanism audit requires support_channel_zscore EOG references"
        )
    event_threshold = float(config["observation"]["artifact_eog_z_threshold"])
    if not np.isfinite(event_threshold) or event_threshold <= 0.0:
        raise ValueError("artifact_eog_z_threshold must be finite and positive")
    if stage == "sampler-integration":
        if task_index is not None:
            raise ValueError("sampler-integration is not an array stage")
        return _run_sampler_integration(config, run_dir=run_dir, device=device)
    if stage == "development-record":
        records = KLADOS_DEVELOPMENT_RECORDS
        partition = "development"
    elif stage == "untouched-record":
        records = KLADOS_UNTOUCHED_RECORDS
        partition = "untouched"
    else:
        raise ValueError(f"unknown GPU mechanism stage: {stage}")
    if task_index is None or not 0 <= int(task_index) < len(records):
        raise ValueError(
            f"{stage} requires SLURM_ARRAY_TASK_ID in [0,{len(records) - 1}]"
        )
    source_record = int(records[int(task_index)])
    return _run_scientific_record(
        config,
        partition=partition,
        source_record=source_record,
        run_dir=run_dir,
        device=device,
    )
