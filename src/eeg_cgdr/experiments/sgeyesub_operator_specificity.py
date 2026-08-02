"""Deterministic SGEYESUB development operator-specificity experiment.

This stage contains no diffusion model.  Every operator is fitted from release
block 1, every output is frozen before query annotations are scored, and the
single B6 gamma is selected only from support-side stability/capture scores.
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import yaml

from eeg_cgdr.baselines.native_sgeyesub import (
    NativeSGEyeSubFitOutcome,
    REFERENCE_EQUIVALENCE_STATUS,
    fit_native_sgeyesub,
)
from eeg_cgdr.data.sgeyesub import (
    SGEYESUB_DEVELOPMENT_STUDIES,
    SGEYESUB_EVALUATION_STUDIES,
    SgeyesubLayout,
    SgeyesubLoadedRecord,
    SgeyesubProtocolRow,
    SgeyesubReleaseRecord,
    build_sgeyesub_protocol,
    load_sgeyesub_signal_record,
    load_sgeyesub_structure_audit,
)
from eeg_cgdr.experiments.sgeyesub_protocol import (
    validate_sgeyesub_protocol_config,
)
from eeg_cgdr.operators import (
    CalibrationBatch,
    P0Config,
    P0FitOutcome,
    ProjectorCompatibilityKey,
    fit_p0,
    spectral_projector_shrink,
)


SGEYESUB_METRIC_DIRECTIONS: tuple[tuple[str, str], ...] = (
    ("matching_projector_attenuation_db", "higher"),
    ("population_projector_attenuation_db", "higher"),
    ("nonartifact_observation_preservation", "higher"),
    ("eog_coherence_reduction", "higher"),
    ("reference_free_psd_distortion", "lower"),
    ("reference_free_covariance_distortion", "lower"),
    ("heldout_eog_prediction_remaining_ratio", "lower"),
    ("condition_erp_observation_relative_preservation", "higher"),
    ("observation_change_ratio", "lower"),
)

GAMMA_ZERO_STRUCTURAL_NOTE = (
    "gamma=0 sets the full and both split-half shrinkage projectors to the "
    "same population projector, so its stability component is structurally "
    "zero; this is an endpoint property, not evidence that participant "
    "calibration is stable. The support-only objective is a conservative "
    "selection rule, not an unbiased hypothesis test of personalization"
)


def _mapping(config: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"SGEYESUB config section {key!r} must be a mapping")
    return value


def _write_resolved_config(config: Mapping[str, object]) -> Path:
    """Persist the exact lightweight SGEYESUB config beside aggregate results."""

    development_parent = Path(str(config["development_output_root"])).parent
    evaluation_parent = Path(str(config["evaluation_output_root"])).parent
    if development_parent != evaluation_parent:
        raise ValueError("SGEYESUB development/evaluation output roots must share a parent")
    development_parent.mkdir(parents=True, exist_ok=True)
    path = development_parent / "resolved_config.yaml"
    path.write_text(yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8")
    return path


def _p0_config(config: Mapping[str, object], *, relaxed: bool) -> P0Config:
    p0 = _mapping(config, "p0")
    return P0Config(
        target_rank=int(p0["rank"]),
        ridge_lambda=float(p0["ridge_lambda"]),
        maximum_reference_condition=float(p0["maximum_reference_condition"]),
        minimum_singular_ratio=float(p0["minimum_singular_ratio"]),
        minimum_movement_coverage=float(p0["minimum_movement_coverage"]),
        bootstrap_replicates=(1 if relaxed else int(p0["bootstrap_replicates"])),
        bootstrap_block_samples=int(p0["bootstrap_block_samples"]),
        minimum_bootstrap_success=(0.0 if relaxed else float(p0["minimum_bootstrap_success"])),
        maximum_bootstrap_median_distance=(
            float("inf")
            if relaxed
            else float(p0["maximum_bootstrap_median_distance"])
        ),
        maximum_bootstrap_q90_distance=(
            float("inf")
            if relaxed
            else float(p0["maximum_bootstrap_q90_distance"])
        ),
        seed=20260802,
    )


def _standardize_eog(
    support: np.ndarray, query: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    support_value = np.asarray(support, dtype=np.float64)
    mean = support_value.mean(axis=1, keepdims=True)
    scale = support_value.std(axis=1, keepdims=True)
    if np.any(scale <= np.finfo(np.float64).eps):
        raise ValueError("support EOG contains a constant channel")
    standardized_support = (support_value - mean) / scale
    standardized_query = (
        None
        if query is None
        else (np.asarray(query, dtype=np.float64) - mean) / scale
    )
    return standardized_support, standardized_query


def _fit(
    eeg: np.ndarray,
    eog: np.ndarray,
    *,
    participant: str,
    sampling_rate: float,
    p0_config: P0Config,
    movement_threshold: float,
) -> P0FitOutcome:
    return fit_p0(
        CalibrationBatch(
            eeg=np.asarray(eeg, dtype=np.float64),
            eog=np.asarray(eog, dtype=np.float64),
            participant=participant,
            source_record="block1_support",
            sampling_rate=sampling_rate,
        ),
        p0_config,
        movement_threshold=movement_threshold,
    )


def _compatibility(
    record: SgeyesubLoadedRecord, *, reference_id: str
) -> ProjectorCompatibilityKey:
    return ProjectorCompatibilityKey(
        dataset_id="sgeyesub_osf_2qgrd_as_observed_2026_08_01",
        montage_id=record.release_layout_id,
        reference_id=reference_id,
        preprocessing_id="release_preprocessed_as_delivered",
        channel_order=record.p0_channel_labels,
    )


def _cell_key(record: SgeyesubReleaseRecord) -> tuple[str, str, float]:
    return record.study, record.layout_id, record.sampling_rate_hz


def _projector_distance(left: np.ndarray, right: np.ndarray, rank: int) -> float:
    return float(np.linalg.norm(left - right, ord="fro") / math.sqrt(2.0 * rank))


def _gamma_token(gamma: float) -> str:
    return f"{gamma:g}".replace(".", "p")


def _support_only_composite_score(
    split_half_stability: float,
    heldout_contamination_capture_loss: float,
    *,
    capture_weight: float,
) -> float:
    """Frozen B6 development score; smaller is better."""

    values = (
        float(split_half_stability),
        float(heldout_contamination_capture_loss),
        float(capture_weight),
    )
    if any(not np.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("support-only B6 score terms must be finite and non-negative")
    return values[0] + values[2] * values[1]


def _population_fit(
    *,
    root: Path,
    target: SgeyesubReleaseRecord,
    sources: Sequence[SgeyesubReleaseRecord],
    layout_map: Mapping[str, SgeyesubLayout],
    config: Mapping[str, object],
) -> tuple[P0FitOutcome, list[str]]:
    eeg_parts: list[np.ndarray] = []
    eog_parts: list[np.ndarray] = []
    used: list[str] = []
    for source in sources:
        if source.recording_key == target.recording_key or _cell_key(source) != _cell_key(target):
            raise AssertionError("population source is not another same-cell participant")
        loaded = load_sgeyesub_signal_record(
            root,
            source,
            layout_map[source.layout_id],
            include_query=False,
        )
        standardized, _ = _standardize_eog(loaded.support.external_eog)
        eeg_parts.append(np.asarray(loaded.support.eeg, dtype=np.float64))
        eog_parts.append(standardized)
        used.append(source.participant_stem)
    if not used:
        raise ValueError("population operator has no same-cell source participant")
    outcome = _fit(
        np.concatenate(eeg_parts, axis=1),
        np.concatenate(eog_parts, axis=1),
        participant="same_cell_other_participants",
        sampling_rate=target.sampling_rate_hz,
        p0_config=_p0_config(config, relaxed=True),
        movement_threshold=float(_mapping(config, "p0")["movement_threshold"]),
    )
    return outcome, used


def _gamma_support_score_row(
    *,
    gamma: float,
    status: str,
    stability: float | None,
    capture_loss: float | None,
    capture_weight: float,
    support_score: float | None,
) -> dict[str, object]:
    """Expose both score components and the structural gamma-zero endpoint."""

    weighted_capture = (
        None
        if capture_loss is None
        else float(capture_weight) * float(capture_loss)
    )
    if support_score is not None and (
        stability is None
        or weighted_capture is None
        or not math.isclose(
            float(support_score),
            float(stability) + weighted_capture,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError("gamma support score does not equal its registered components")
    return {
        "gamma": float(gamma),
        "status": status,
        "split_half_stability": stability,
        "heldout_contamination_capture_loss": capture_loss,
        "capture_weight": float(capture_weight),
        "weighted_capture_component": weighted_capture,
        "support_score": support_score,
        "support_score_formula": (
            "split_half_stability + capture_weight * "
            "heldout_contamination_capture_loss"
        ),
        "population_endpoint": float(gamma) == 0.0,
        "structural_zero_stability": float(gamma) == 0.0,
        "structural_zero_explanation": (
            GAMMA_ZERO_STRUCTURAL_NOTE if float(gamma) == 0.0 else None
        ),
    }


def _gamma_support_scores(
    *,
    population_projector: np.ndarray | None,
    context_outcome: P0FitOutcome,
    support_eeg: np.ndarray,
    support_eog: np.ndarray,
    compatibility: ProjectorCompatibilityKey,
    config: Mapping[str, object],
    sampling_rate: float,
) -> list[dict[str, object]]:
    b6 = _mapping(config, "b6_pop_shrink")
    p0 = _mapping(config, "p0")
    rank = int(p0["rank"])
    capture_weight = float(b6["heldout_contamination_capture_weight"])
    midpoint = support_eeg.shape[1] // 2
    split_indices = (
        np.arange(0, midpoint, dtype=int),
        np.arange(midpoint, support_eeg.shape[1], dtype=int),
    )
    half_outcomes = [
        _fit(
            support_eeg[:, indices],
            support_eog[:, indices],
            participant="support_half",
            sampling_rate=sampling_rate,
            p0_config=_p0_config(config, relaxed=True),
            movement_threshold=float(p0["movement_threshold"]),
        )
        for indices in split_indices
    ]
    if any(outcome.transfer is None for outcome in half_outcomes):
        return [
            _gamma_support_score_row(
                gamma=float(gamma),
                status="unavailable_split_half_context",
                stability=(0.0 if float(gamma) == 0.0 else None),
                capture_loss=None,
                capture_weight=capture_weight,
                support_score=None,
            )
            for gamma in b6["gamma_candidates"]
        ]

    rows: list[dict[str, object]] = []
    for value in b6["gamma_candidates"]:
        gamma = float(value)
        if gamma > 0.0 and context_outcome.transfer is None:
            rows.append(
                _gamma_support_score_row(
                    gamma=gamma,
                    status="unavailable_full_context",
                    stability=None,
                    capture_loss=None,
                    capture_weight=capture_weight,
                    support_score=None,
                )
            )
            continue
        full_context = (
            None if gamma == 0.0 else context_outcome.transfer.projector
        )
        full = spectral_projector_shrink(
            population_projector,
            full_context,
            rank=rank,
            gamma=gamma,
            context_eligible=gamma > 0.0,
            population_compatibility=compatibility,
            population_fit_scope="outer_training_only",
            context_compatibility=None if gamma == 0.0 else compatibility,
            context_fit_scope=None if gamma == 0.0 else "support_only",
        )
        half_projectors: list[np.ndarray] = []
        for outcome in half_outcomes:
            assert outcome.transfer is not None
            shrunk = spectral_projector_shrink(
                population_projector,
                None if gamma == 0.0 else outcome.transfer.projector,
                rank=rank,
                gamma=gamma,
                context_eligible=gamma > 0.0,
                population_compatibility=compatibility,
                population_fit_scope="outer_training_only",
                context_compatibility=None if gamma == 0.0 else compatibility,
                context_fit_scope=None if gamma == 0.0 else "support_only",
            )
            if shrunk.status == "eligible" and shrunk.projector is not None:
                half_projectors.append(shrunk.projector)
        if full.status != "eligible" or full.projector is None or len(half_projectors) != 2:
            rows.append(
                _gamma_support_score_row(
                    gamma=gamma,
                    status="unavailable_shrinkage",
                    stability=None,
                    capture_loss=None,
                    capture_weight=capture_weight,
                    support_score=None,
                )
            )
            continue
        stability = _projector_distance(
            half_projectors[0], half_projectors[1], rank
        )
        capture_losses: list[float] = []
        identity = np.eye(full.projector.shape[0], dtype=np.float64)
        for inner_index, heldout_index in ((0, 1), (1, 0)):
            transfer = half_outcomes[inner_index].transfer
            assert transfer is not None
            heldout_eog = support_eog[:, split_indices[heldout_index]]
            predicted_heldout = transfer.transfer_matrix @ (
                heldout_eog - transfer.eog_mean
            )
            denominator = max(
                float(np.linalg.norm(predicted_heldout) ** 2),
                np.finfo(np.float64).eps,
            )
            numerator = float(
                np.linalg.norm(
                    (identity - half_projectors[inner_index]) @ predicted_heldout
                )
                ** 2
            )
            capture_losses.append(numerator / denominator)
        capture_loss = float(np.mean(capture_losses))
        score = _support_only_composite_score(
            stability,
            capture_loss,
            capture_weight=capture_weight,
        )
        rows.append(
            _gamma_support_score_row(
                gamma=gamma,
                status="success",
                stability=stability,
                capture_loss=capture_loss,
                capture_weight=capture_weight,
                support_score=score,
            )
        )
    return rows


def _q_restore(projector: np.ndarray, observed: np.ndarray) -> np.ndarray:
    return observed - projector @ observed


def _soft_restore(projector: np.ndarray, observed: np.ndarray, tau: float) -> np.ndarray:
    # Explicit deterministic proximal endpoint requested by the protocol.
    return _q_restore(projector, observed) + float(tau) * (projector @ observed)


def _mean_abs_eog_correlation(
    eeg: np.ndarray, eog: np.ndarray, mask: np.ndarray
) -> float:
    if int(mask.sum()) < 4:
        return float("nan")
    x = np.asarray(eeg[:, mask], dtype=np.float64)
    z = np.asarray(eog[:, mask], dtype=np.float64)
    x -= x.mean(axis=1, keepdims=True)
    z -= z.mean(axis=1, keepdims=True)
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    z_norm = np.linalg.norm(z, axis=1, keepdims=True)
    denominator = np.maximum(x_norm @ z_norm.T, np.finfo(np.float64).eps)
    return float(np.mean(np.abs((x @ z.T) / denominator)))


def _psd_distortion(output: np.ndarray, observed: np.ndarray, mask: np.ndarray) -> float:
    if int(mask.sum()) < 8:
        return float("nan")
    output_power = np.abs(np.fft.rfft(output[:, mask], axis=1)) ** 2
    observed_power = np.abs(np.fft.rfft(observed[:, mask], axis=1)) ** 2
    numerator = np.linalg.norm(output_power - observed_power, axis=1)
    denominator = np.maximum(
        np.linalg.norm(observed_power, axis=1), np.finfo(np.float64).eps
    )
    return float(np.mean(numerator / denominator))


def _covariance_distortion(
    output: np.ndarray, observed: np.ndarray, mask: np.ndarray
) -> float:
    if int(mask.sum()) < 2:
        return float("nan")
    output_covariance = np.cov(output[:, mask], bias=False)
    observed_covariance = np.cov(observed[:, mask], bias=False)
    denominator = max(
        float(np.linalg.norm(observed_covariance, ord="fro")),
        np.finfo(np.float64).eps,
    )
    return float(
        np.linalg.norm(output_covariance - observed_covariance, ord="fro")
        / denominator
    )


def _predicted_contamination_remaining(
    output: np.ndarray,
    observed: np.ndarray,
    predicted_contamination: np.ndarray | None,
) -> float:
    if predicted_contamination is None:
        return float("nan")
    predicted = np.asarray(predicted_contamination, dtype=np.float64)
    if predicted.shape != observed.shape or not np.isfinite(predicted).all():
        raise ValueError("held-out predicted contamination shape/value mismatch")
    denominator = max(
        float(np.linalg.norm(predicted)),
        np.finfo(np.float64).eps,
    )
    return float(np.linalg.norm(output - observed + predicted) / denominator)


def _condition_erp_preservation(
    output: np.ndarray,
    observed: np.ndarray,
    *,
    trial_labels: np.ndarray,
    samples_per_trial: int,
    minimum_trials_per_condition: int,
) -> tuple[float, str]:
    labels = np.asarray(trial_labels, dtype=np.int64).reshape(-1)
    if output.shape != observed.shape or output.ndim != 2:
        raise ValueError("condition ERP arrays must be aligned channel-major matrices")
    if labels.size * samples_per_trial != output.shape[1]:
        raise ValueError("query trial labels do not align with flattened EEG")
    if minimum_trials_per_condition < 1:
        raise ValueError("minimum_trials_per_condition must be positive")
    if set(labels.tolist()) != {1, 2, 3, 4} or any(
        int(np.sum(labels == condition)) < minimum_trials_per_condition
        for condition in (1, 2, 3, 4)
    ):
        return float("nan"), "N/A_insufficient_condition_trials"
    output_trials = output.reshape(
        output.shape[0], labels.size, samples_per_trial
    )
    observed_trials = observed.reshape(
        observed.shape[0], labels.size, samples_per_trial
    )
    output_templates = np.stack(
        [output_trials[:, labels == condition].mean(axis=1) for condition in (1, 2, 3, 4)],
        axis=1,
    )
    observed_templates = np.stack(
        [
            observed_trials[:, labels == condition].mean(axis=1)
            for condition in (1, 2, 3, 4)
        ],
        axis=1,
    )
    denominator = max(
        float(np.linalg.norm(observed_templates)),
        np.finfo(np.float64).eps,
    )
    preservation = 1.0 - float(
        np.linalg.norm(output_templates - observed_templates) / denominator
    )
    return preservation, "success_observation_relative_four_condition_ERP"


def _evaluate_output(
    *,
    method_id: str,
    output: np.ndarray,
    observed: np.ndarray,
    matching_projector: np.ndarray | None,
    population_projector: np.ndarray | None,
    query_eog: np.ndarray,
    artifactclasses: np.ndarray,
    predicted_contamination: np.ndarray | None,
    trial_labels: np.ndarray,
    samples_per_trial: int,
    minimum_trials_per_condition: int,
    status: str,
    operator_source: str,
    gamma: float | None,
    fallback_used: bool,
    uses_query_external_eog: bool,
) -> dict[str, object]:
    # Label 0 is unlabelled in the public release, not an artifact interval.
    artifact_mask = np.asarray(
        (artifactclasses >= 1) & (artifactclasses <= 5), dtype=bool
    )
    rest_mask = np.asarray(artifactclasses == 6, dtype=bool)

    def projected_attenuation(projector: np.ndarray | None) -> float:
        if projector is None or not np.any(artifact_mask):
            return float("nan")
        projected_observed = projector @ observed
        projected_output = projector @ output
        return 20.0 * math.log10(
            max(
                float(np.linalg.norm(projected_observed[:, artifact_mask])),
                1.0e-12,
            )
            / max(
                float(np.linalg.norm(projected_output[:, artifact_mask])),
                1.0e-12,
            )
        )

    if np.any(rest_mask):
        rest_denominator = max(
            float(np.linalg.norm(observed[:, rest_mask])),
            np.finfo(np.float64).eps,
        )
        preservation = 1.0 - float(
            np.linalg.norm(output[:, rest_mask] - observed[:, rest_mask])
            / rest_denominator
        )
    else:
        preservation = float("nan")
    raw_coherence = _mean_abs_eog_correlation(observed, query_eog, artifact_mask)
    output_coherence = _mean_abs_eog_correlation(output, query_eog, artifact_mask)
    condition_preservation, condition_status = _condition_erp_preservation(
        output,
        observed,
        trial_labels=trial_labels,
        samples_per_trial=samples_per_trial,
        minimum_trials_per_condition=minimum_trials_per_condition,
    )
    return {
        "method_id": method_id,
        "status": status,
        "operator_source": operator_source,
        "gamma": "" if gamma is None else gamma,
        "fallback_used": fallback_used,
        "uses_query_external_eog": uses_query_external_eog,
        "matching_projector_attenuation_db": projected_attenuation(
            matching_projector
        ),
        "population_projector_attenuation_db": projected_attenuation(
            population_projector
        ),
        "nonartifact_observation_preservation": preservation,
        "artifact_sample_fraction": float(np.mean(artifact_mask)),
        "rest_sample_fraction": float(np.mean(rest_mask)),
        "eog_coherence_raw": raw_coherence,
        "eog_coherence_output": output_coherence,
        "eog_coherence_reduction": raw_coherence - output_coherence,
        "reference_free_psd_distortion": _psd_distortion(output, observed, rest_mask),
        "reference_free_covariance_distortion": _covariance_distortion(
            output, observed, rest_mask
        ),
        "heldout_eog_prediction_remaining_ratio": (
            _predicted_contamination_remaining(
                output,
                observed,
                predicted_contamination,
            )
        ),
        "condition_erp_observation_relative_preservation": condition_preservation,
        "condition_erp_proxy_status": condition_status,
        "observation_change_ratio": float(
            np.linalg.norm(output - observed)
            / max(float(np.linalg.norm(observed)), np.finfo(np.float64).eps)
        ),
        "clean_waveform_metric": "N/A_no_clean_target",
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def select_global_gamma(
    by_gamma: Mapping[float, Sequence[float]],
    *,
    candidates: Sequence[float],
    participant_count: int,
    minimum_fraction: float,
) -> tuple[float, list[dict[str, object]]]:
    """Select one development gamma from support-only scores.

    Both endpoints are ordinary candidates.  The deterministic tie break is
    toward the smaller gamma, so a best ``gamma=0`` result is not overridden
    merely to force a context contribution.
    """

    if participant_count < 1:
        raise ValueError("participant_count must be positive")
    if not 0.0 < minimum_fraction <= 1.0:
        raise ValueError("minimum_fraction must lie in (0, 1]")
    frozen_candidates = tuple(float(value) for value in candidates)
    if frozen_candidates != (0.0, 0.25, 0.5, 0.75, 1.0):
        raise ValueError("unexpected SGEYESUB gamma candidate set")
    minimum_count = math.ceil(participant_count * minimum_fraction)
    aggregate_scores: list[dict[str, object]] = []
    for gamma in frozen_candidates:
        values = [float(value) for value in by_gamma.get(gamma, ())]
        if any(not np.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("gamma support scores must be finite and non-negative")
        eligible = len(values) >= minimum_count
        aggregate_scores.append(
            {
                "gamma": gamma,
                "participant_count": len(values),
                "mean_support_score": float(np.mean(values)) if eligible else None,
                "eligible": eligible,
            }
        )
    eligible_rows = [row for row in aggregate_scores if row["eligible"]]
    if not eligible_rows:
        raise RuntimeError("no B6 gamma has enough support-only development scores")
    selected = min(
        eligible_rows,
        key=lambda row: (float(row["mean_support_score"]), float(row["gamma"])),
    )
    return float(selected["gamma"]), aggregate_scores


def _method_summary(
    rows: Sequence[Mapping[str, object]],
    *,
    partition: str,
    bootstrap_replicates: int = 2_000,
    bootstrap_seed: int = 20260802,
) -> list[dict[str, object]]:
    """Success-only performance; coverage and failures are reported separately."""

    if bootstrap_replicates < 1:
        raise ValueError("method summary bootstrap count must be positive")
    method_ids = sorted({str(row["method_id"]) for row in rows})
    summaries: list[dict[str, object]] = []
    for method_index, method_id in enumerate(method_ids):
        method_rows = [
            row
            for row in rows
            if str(row["method_id"]) == method_id
            and _is_success_performance_row(row)
        ]
        for metric_index, (metric_name, direction) in enumerate(
            SGEYESUB_METRIC_DIRECTIONS
        ):
            values: list[float] = []
            for row in method_rows:
                value = _finite_metric(row, metric_name)
                if value is not None:
                    values.append(value)
            if not values:
                continue
            numeric = np.asarray(values, dtype=np.float64)
            rng = np.random.default_rng(
                bootstrap_seed + 100 * method_index + metric_index
            )
            indices = rng.integers(
                0, numeric.size, size=(bootstrap_replicates, numeric.size)
            )
            bootstrap_means = numeric[indices].mean(axis=1)
            summaries.append(
                {
                    "partition": partition,
                    "method_id": method_id,
                    "metric": metric_name,
                    "direction": direction,
                    "participant_stem_count": int(numeric.size),
                    "mean": float(np.mean(numeric)),
                    "median": float(np.median(numeric)),
                    "ci95_low": float(np.quantile(bootstrap_means, 0.025)),
                    "ci95_high": float(np.quantile(bootstrap_means, 0.975)),
                    "bootstrap_replicates": bootstrap_replicates,
                    "bootstrap_seed": bootstrap_seed,
                    "inference_unit": "release_scoped_study_participant_stem",
                    "row_policy": "success_nonfallback_finite_only",
                }
            )
    return summaries


def _is_fallback_row(row: Mapping[str, object]) -> bool:
    value = row.get("fallback_used", False)
    return value is True or str(value).strip().lower() == "true" or str(
        row.get("status", "")
    ).startswith("fallback")


def _is_success_performance_row(row: Mapping[str, object]) -> bool:
    return str(row.get("status", "")).startswith("success") and not _is_fallback_row(
        row
    )


def _finite_metric(row: Mapping[str, object], metric: str) -> float | None:
    try:
        value = float(row.get(metric, "nan"))
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _development_gamma_component_audit(
    config: Mapping[str, object],
    protocol_rows: Sequence[SgeyesubProtocolRow],
) -> dict[str, object]:
    """Read existing development scores and expose their two score components."""

    if len(protocol_rows) != 15:
        raise ValueError("development gamma component audit requires 15 stems")
    development_root = Path(str(config["development_output_root"]))
    candidates = tuple(
        float(value) for value in _mapping(config, "b6_pop_shrink")["gamma_candidates"]
    )
    registered_capture_weight = float(
        _mapping(config, "b6_pop_shrink")["heldout_contamination_capture_weight"]
    )
    component_rows: list[dict[str, object]] = []
    for protocol_row in protocol_rows:
        path = development_root / protocol_row.participant_stem / "support_gamma_scores.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("development gamma component input must be a list")
        seen: set[float] = set()
        for raw in payload:
            if not isinstance(raw, Mapping):
                raise ValueError("development gamma component row must be a mapping")
            gamma = float(raw["gamma"])
            if gamma in seen or gamma not in candidates:
                raise ValueError("development gamma component candidates changed")
            seen.add(gamma)
            capture_weight = float(raw["capture_weight"])
            if not math.isclose(
                capture_weight,
                registered_capture_weight,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ):
                raise ValueError("development gamma capture weight changed")
            enriched = _gamma_support_score_row(
                gamma=gamma,
                status=str(raw["status"]),
                stability=(
                    None
                    if raw.get("split_half_stability") is None
                    else float(raw["split_half_stability"])
                ),
                capture_loss=(
                    None
                    if raw.get("heldout_contamination_capture_loss") is None
                    else float(raw["heldout_contamination_capture_loss"])
                ),
                capture_weight=capture_weight,
                support_score=(
                    None
                    if raw.get("support_score") is None
                    else float(raw["support_score"])
                ),
            )
            component_rows.append(
                {
                    "study": protocol_row.study,
                    "participant_stem": protocol_row.participant_stem,
                    "recording_key": protocol_row.recording_key,
                    **enriched,
                }
            )
        if seen != set(candidates):
            raise ValueError("development gamma component candidate set is incomplete")

    summary_rows: list[dict[str, object]] = []
    for gamma in candidates:
        rows = [row for row in component_rows if float(row["gamma"]) == gamma]
        successful = [
            row
            for row in rows
            if row["status"] == "success" and row["support_score"] is not None
        ]

        def mean(field: str) -> float | None:
            values = [float(row[field]) for row in successful if row[field] is not None]
            return float(np.mean(values)) if values else None

        summary_rows.append(
            {
                "gamma": gamma,
                "development_record_count": len(rows),
                "successful_component_count": len(successful),
                "mean_split_half_stability": mean("split_half_stability"),
                "mean_heldout_contamination_capture_loss": mean(
                    "heldout_contamination_capture_loss"
                ),
                "capture_weight": mean("capture_weight"),
                "mean_weighted_capture_component": mean(
                    "weighted_capture_component"
                ),
                "mean_support_score": mean("support_score"),
                "support_score_formula": (
                    "split_half_stability + capture_weight * "
                    "heldout_contamination_capture_loss"
                ),
                "population_endpoint": gamma == 0.0,
                "structural_zero_stability": gamma == 0.0,
                "structural_zero_explanation": (
                    GAMMA_ZERO_STRUCTURAL_NOTE if gamma == 0.0 else None
                ),
            }
        )
    return {"component_rows": component_rows, "summary_rows": summary_rows}


def _method_coverage_summary(
    rows: Sequence[Mapping[str, object]], *, partition: str
) -> list[dict[str, object]]:
    """Keep blocked/ineligible/fallback/failed rows in explicit denominators."""

    output: list[dict[str, object]] = []
    for method_id in sorted({str(row["method_id"]) for row in rows}):
        method_rows = [row for row in rows if str(row["method_id"]) == method_id]
        counts = {
            "success": 0,
            "fallback": 0,
            "blocked": 0,
            "ineligible": 0,
            "failed": 0,
            "other": 0,
        }
        for row in method_rows:
            status = str(row.get("status", ""))
            if _is_fallback_row(row):
                category = "fallback"
            elif status.startswith("success"):
                category = "success"
            elif status.startswith("blocked"):
                category = "blocked"
            elif status.startswith("ineligible"):
                category = "ineligible"
            elif status.startswith("failed"):
                category = "failed"
            else:
                category = "other"
            counts[category] += 1
        total = len(method_rows)
        output.append(
            {
                "partition": partition,
                "method_id": method_id,
                "record_count": total,
                **{f"{key}_count": value for key, value in counts.items()},
                "performance_eligible_count": counts["success"],
                "performance_eligible_fraction": counts["success"] / max(total, 1),
                "coverage_denominator_policy": "all_registered_recording_keys",
                "performance_policy": "success_nonfallback_only",
            }
        )
    return output


def _bootstrap_location_intervals(
    values: Sequence[float], *, replicates: int, seed: int
) -> dict[str, float] | None:
    numeric = np.asarray(values, dtype=np.float64)
    if numeric.size < 1 or not np.isfinite(numeric).all():
        return None
    if replicates != 20_000:
        raise ValueError("corrected SGEYESUB audit requires 20000 bootstraps")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, numeric.size, size=(replicates, numeric.size))
    samples = numeric[indices]
    means = samples.mean(axis=1)
    medians = np.median(samples, axis=1)
    return {
        "mean_ci95_low": float(np.quantile(means, 0.025)),
        "mean_ci95_high": float(np.quantile(means, 0.975)),
        "median_ci95_low": float(np.quantile(medians, 0.025)),
        "median_ci95_high": float(np.quantile(medians, 0.975)),
    }


def _matching_population_audit(
    rows: Sequence[Mapping[str, object]],
    protocol_rows: Sequence[SgeyesubProtocolRow],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    metric_directions: Sequence[tuple[str, str]] = SGEYESUB_METRIC_DIRECTIONS,
) -> dict[str, object]:
    """Paired matching-vs-population audit on the 43 compatible stems."""

    registered = tuple(protocol_rows)
    compatible = tuple(row for row in registered if row.status == "metadata_ready")
    blocked = tuple(row for row in registered if row.status == "blocked_no_population")
    if len(registered) != 44 or len(compatible) != 43 or len(blocked) != 1:
        raise ValueError("corrected audit requires 43 compatible plus one singleton stem")
    if {row.study for row in compatible} != {"study02", "study04", "study05"}:
        raise ValueError("corrected audit study partition is incomplete")
    compatible_study_counts = {
        study: sum(row.study == study for row in compatible)
        for study in ("study02", "study04", "study05")
    }
    if compatible_study_counts != {"study02": 15, "study04": 15, "study05": 13}:
        raise ValueError("corrected audit compatible study counts changed")

    by_key_method: dict[tuple[str, str], Mapping[str, object]] = {}
    for row in rows:
        key = (str(row.get("recording_key", "")), str(row.get("method_id", "")))
        if key in by_key_method:
            raise ValueError("corrected audit received duplicate recording/method rows")
        by_key_method[key] = row

    method_success_keys: list[str] = []
    for protocol_row in compatible:
        matching = by_key_method.get((protocol_row.recording_key, "matching_Qy"))
        population = by_key_method.get((protocol_row.recording_key, "pop_Qy"))
        if (
            matching is not None
            and population is not None
            and _is_success_performance_row(matching)
            and _is_success_performance_row(population)
        ):
            method_success_keys.append(protocol_row.recording_key)

    pair_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    heterogeneity_rows: list[dict[str, object]] = []
    for metric, direction in metric_directions:
        if direction not in {"lower", "higher"}:
            raise ValueError(f"unsupported SGEYESUB metric direction: {direction}")
        finite_pairs: list[dict[str, object]] = []
        for protocol_row in compatible:
            matching = by_key_method.get((protocol_row.recording_key, "matching_Qy"))
            population = by_key_method.get((protocol_row.recording_key, "pop_Qy"))
            method_pair_success = (
                matching is not None
                and population is not None
                and _is_success_performance_row(matching)
                and _is_success_performance_row(population)
            )
            matching_value = (
                _finite_metric(matching, metric) if matching is not None else None
            )
            population_value = (
                _finite_metric(population, metric) if population is not None else None
            )
            finite_pair = (
                method_pair_success
                and matching_value is not None
                and population_value is not None
            )
            raw_delta = (
                None
                if not finite_pair
                else float(matching_value) - float(population_value)
            )
            directional_improvement = (
                None
                if raw_delta is None
                else raw_delta if direction == "higher" else -raw_delta
            )
            pair = {
                "study": protocol_row.study,
                "participant_stem": protocol_row.participant_stem,
                "recording_key": protocol_row.recording_key,
                "metric": metric,
                "direction": direction,
                "matching_status": (
                    "missing" if matching is None else str(matching.get("status", ""))
                ),
                "population_status": (
                    "missing"
                    if population is None
                    else str(population.get("status", ""))
                ),
                "method_pair_success": method_pair_success,
                "finite_pair": finite_pair,
                "matching_value": matching_value,
                "population_value": population_value,
                "matching_minus_population": raw_delta,
                "directional_improvement_positive_is_matching_better": (
                    directional_improvement
                ),
                "pair_status": (
                    "success_finite"
                    if finite_pair
                    else "excluded_non_success_or_nonfinite"
                ),
            }
            pair_rows.append(pair)
            if finite_pair:
                finite_pairs.append(pair)

        improvements = [
            float(pair["directional_improvement_positive_is_matching_better"])
            for pair in finite_pairs
        ]
        raw_deltas = [float(pair["matching_minus_population"]) for pair in finite_pairs]
        matching_values = [float(pair["matching_value"]) for pair in finite_pairs]
        population_values = [float(pair["population_value"]) for pair in finite_pairs]
        wins = sum(value > 1.0e-12 for value in improvements)
        ties = sum(abs(value) <= 1.0e-12 for value in improvements)
        losses = sum(value < -1.0e-12 for value in improvements)
        directional_interval = _bootstrap_location_intervals(
            improvements,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        )
        raw_interval = _bootstrap_location_intervals(
            raw_deltas,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        )
        summary_rows.append(
            {
                "metric": metric,
                "direction": direction,
                "compatible_record_count": len(compatible),
                "method_success_paired_count": len(method_success_keys),
                "finite_metric_paired_count": len(finite_pairs),
                "finite_metric_paired_fraction": len(finite_pairs) / len(compatible),
                "matching_mean": (
                    float(np.mean(matching_values)) if matching_values else None
                ),
                "population_mean": (
                    float(np.mean(population_values)) if population_values else None
                ),
                "matching_median": (
                    float(np.median(matching_values)) if matching_values else None
                ),
                "population_median": (
                    float(np.median(population_values)) if population_values else None
                ),
                "mean_matching_minus_population": (
                    float(np.mean(raw_deltas)) if raw_deltas else None
                ),
                "median_matching_minus_population": (
                    float(np.median(raw_deltas)) if raw_deltas else None
                ),
                "mean_directional_improvement": (
                    float(np.mean(improvements)) if improvements else None
                ),
                "median_directional_improvement": (
                    float(np.median(improvements)) if improvements else None
                ),
                "matching_wins": wins,
                "ties": ties,
                "matching_losses": losses,
                "bootstrap_replicates": bootstrap_replicates,
                "bootstrap_seed": bootstrap_seed,
                "mean_matching_minus_population_ci95_low": (
                    None if raw_interval is None else raw_interval["mean_ci95_low"]
                ),
                "mean_matching_minus_population_ci95_high": (
                    None if raw_interval is None else raw_interval["mean_ci95_high"]
                ),
                "median_matching_minus_population_ci95_low": (
                    None if raw_interval is None else raw_interval["median_ci95_low"]
                ),
                "median_matching_minus_population_ci95_high": (
                    None if raw_interval is None else raw_interval["median_ci95_high"]
                ),
                "mean_directional_improvement_ci95_low": (
                    None
                    if directional_interval is None
                    else directional_interval["mean_ci95_low"]
                ),
                "mean_directional_improvement_ci95_high": (
                    None
                    if directional_interval is None
                    else directional_interval["mean_ci95_high"]
                ),
                "median_directional_improvement_ci95_low": (
                    None
                    if directional_interval is None
                    else directional_interval["median_ci95_low"]
                ),
                "median_directional_improvement_ci95_high": (
                    None
                    if directional_interval is None
                    else directional_interval["median_ci95_high"]
                ),
                "raw_delta_definition": "matching_minus_population",
                "win_definition": "positive_directional_improvement",
                "inference_unit": "release_scoped_study_participant_stem",
            }
        )

        for study in ("study02", "study04", "study05"):
            study_compatible = [row for row in compatible if row.study == study]
            study_pairs = [pair for pair in finite_pairs if pair["study"] == study]
            study_improvements = [
                float(pair["directional_improvement_positive_is_matching_better"])
                for pair in study_pairs
            ]
            study_raw_deltas = [
                float(pair["matching_minus_population"]) for pair in study_pairs
            ]
            study_directional_interval = _bootstrap_location_intervals(
                study_improvements,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed,
            )
            study_raw_interval = _bootstrap_location_intervals(
                study_raw_deltas,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed,
            )
            heterogeneity_rows.append(
                {
                    "study": study,
                    "metric": metric,
                    "direction": direction,
                    "compatible_record_count": len(study_compatible),
                    "finite_metric_paired_count": len(study_pairs),
                    "mean_matching_minus_population": (
                        float(np.mean(study_raw_deltas))
                        if study_raw_deltas
                        else None
                    ),
                    "median_matching_minus_population": (
                        float(np.median(study_raw_deltas))
                        if study_raw_deltas
                        else None
                    ),
                    "mean_directional_improvement": (
                        float(np.mean(study_improvements))
                        if study_improvements
                        else None
                    ),
                    "median_directional_improvement": (
                        float(np.median(study_improvements))
                        if study_improvements
                        else None
                    ),
                    "matching_wins": sum(
                        value > 1.0e-12 for value in study_improvements
                    ),
                    "ties": sum(abs(value) <= 1.0e-12 for value in study_improvements),
                    "matching_losses": sum(
                        value < -1.0e-12 for value in study_improvements
                    ),
                    "bootstrap_replicates": bootstrap_replicates,
                    "bootstrap_seed": bootstrap_seed,
                    "mean_matching_minus_population_ci95_low": (
                        None
                        if study_raw_interval is None
                        else study_raw_interval["mean_ci95_low"]
                    ),
                    "mean_matching_minus_population_ci95_high": (
                        None
                        if study_raw_interval is None
                        else study_raw_interval["mean_ci95_high"]
                    ),
                    "median_matching_minus_population_ci95_low": (
                        None
                        if study_raw_interval is None
                        else study_raw_interval["median_ci95_low"]
                    ),
                    "median_matching_minus_population_ci95_high": (
                        None
                        if study_raw_interval is None
                        else study_raw_interval["median_ci95_high"]
                    ),
                    "mean_directional_improvement_ci95_low": (
                        None
                        if study_directional_interval is None
                        else study_directional_interval["mean_ci95_low"]
                    ),
                    "mean_directional_improvement_ci95_high": (
                        None
                        if study_directional_interval is None
                        else study_directional_interval["mean_ci95_high"]
                    ),
                    "raw_delta_definition": "matching_minus_population",
                    "win_definition": "positive_directional_improvement",
                }
            )

    return {
        "status": (
            "complete_43_success_paired"
            if len(method_success_keys) == 43
            else "inconclusive_incomplete_success_pair_coverage"
        ),
        "registered_record_count": len(registered),
        "compatible_record_count": len(compatible),
        "blocked_singleton_recording_key": blocked[0].recording_key,
        "method_success_paired_count": len(method_success_keys),
        "method_success_paired_fraction": len(method_success_keys) / len(compatible),
        "method_success_paired_recording_keys": method_success_keys,
        "pair_rows": pair_rows,
        "summary_rows": summary_rows,
        "heterogeneity_rows": heterogeneity_rows,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
    }


def _absolute_safety_summary(
    rows: Sequence[Mapping[str, object]], *, config: Mapping[str, object]
) -> list[dict[str, object]]:
    thresholds = _mapping(config, "operator_specificity_decision")
    minimum_preservation = float(
        thresholds["minimum_nonartifact_observation_preservation"]
    )
    maximum_covariance = float(
        thresholds["maximum_reference_free_covariance_distortion"]
    )
    output: list[dict[str, object]] = []
    for method_id in sorted({str(row["method_id"]) for row in rows}):
        method_rows = [row for row in rows if str(row["method_id"]) == method_id]
        successful = [row for row in method_rows if _is_success_performance_row(row)]
        finite: list[tuple[float, float]] = []
        for row in successful:
            preservation = _finite_metric(row, "nonartifact_observation_preservation")
            covariance = _finite_metric(row, "reference_free_covariance_distortion")
            if preservation is not None and covariance is not None:
                finite.append((preservation, covariance))
        joint_passes = sum(
            preservation >= minimum_preservation and covariance <= maximum_covariance
            for preservation, covariance in finite
        )
        output.append(
            {
                "method_id": method_id,
                "registered_record_count": len(method_rows),
                "success_nonfallback_count": len(successful),
                "finite_joint_safety_count": len(finite),
                "minimum_nonartifact_observation_preservation": minimum_preservation,
                "maximum_reference_free_covariance_distortion": maximum_covariance,
                "nonartifact_preservation_pass_count": sum(
                    value >= minimum_preservation for value, _ in finite
                ),
                "covariance_distortion_pass_count": sum(
                    value <= maximum_covariance for _, value in finite
                ),
                "joint_safety_pass_count": joint_passes,
                "joint_safety_pass_fraction_finite": (
                    joint_passes / len(finite) if finite else None
                ),
                "joint_safety_pass_fraction_all_registered": (
                    joint_passes / len(method_rows) if method_rows else None
                ),
                "mean_nonartifact_observation_preservation": (
                    float(np.mean([value for value, _ in finite])) if finite else None
                ),
                "median_nonartifact_observation_preservation": (
                    float(np.median([value for value, _ in finite])) if finite else None
                ),
                "mean_reference_free_covariance_distortion": (
                    float(np.mean([value for _, value in finite])) if finite else None
                ),
                "median_reference_free_covariance_distortion": (
                    float(np.median([value for _, value in finite])) if finite else None
                ),
                "performance_policy": "success_nonfallback_finite_only",
                "coverage_denominator_policy": "all_registered_recording_keys",
            }
        )
    return output


def _corrected_method_groups(frozen_gamma: float) -> dict[str, tuple[str, ...]]:
    token = _gamma_token(frozen_gamma)
    b6_hard = f"B6_Qy__gamma_{token}"
    b6_soft = f"B6_soft_proximal__gamma_{token}"
    native = "native_sgeyesub_python_release_internal"
    return {
        "focus": ("matching_Qy", "pop_Qy", b6_hard, b6_soft, native),
        "controls": ("wrong_Qy", "shuffled_Qy"),
        "hard_q_safety": (
            "matching_Qy",
            "pop_Qy",
            b6_hard,
            "wrong_Qy",
            "shuffled_Qy",
        ),
        "additional_safety": (b6_soft, native),
    }


def _required_method_record_coverage(
    rows: Sequence[Mapping[str, object]],
    protocol_rows: Sequence[SgeyesubProtocolRow],
    *,
    method_ids: Sequence[str],
) -> dict[str, object]:
    """Require one row for every registered stem before an audit is complete."""

    registered_keys = [row.recording_key for row in protocol_rows]
    registered = set(registered_keys)
    if len(registered_keys) != 44 or len(registered) != 44:
        raise ValueError("corrected audit requires 44 unique registered stems")

    method_rows: list[dict[str, object]] = []
    for method_id in method_ids:
        keys = [
            str(row["recording_key"])
            for row in rows
            if str(row.get("method_id", "")) == method_id
        ]
        counts: dict[str, int] = {}
        for key in keys:
            counts[key] = counts.get(key, 0) + 1
        unique = set(keys)
        missing = sorted(registered - unique)
        unexpected = sorted(unique - registered)
        duplicates = sorted(key for key, count in counts.items() if count > 1)
        complete = (
            len(keys) == 44
            and len(unique) == 44
            and not missing
            and not unexpected
            and not duplicates
        )
        method_rows.append(
            {
                "method_id": method_id,
                "row_count": len(keys),
                "unique_recording_key_count": len(unique),
                "missing_recording_key_count": len(missing),
                "unexpected_recording_key_count": len(unexpected),
                "duplicate_recording_key_count": len(duplicates),
                "complete_44_unique_recording_keys": complete,
            }
        )

    all_complete = all(
        bool(row["complete_44_unique_recording_keys"]) for row in method_rows
    )
    return {
        "status": (
            "complete_44_unique_recording_keys"
            if all_complete
            else "inconclusive_incomplete_required_method_record_coverage"
        ),
        "all_required_methods_complete": all_complete,
        "registered_record_count": len(registered),
        "methods": method_rows,
    }


def _method_metric_status_view(
    performance: Sequence[Mapping[str, object]],
    coverage: Sequence[Mapping[str, object]],
    *,
    method_ids: Sequence[str],
) -> list[dict[str, object]]:
    """Complete method-by-nine-metric view with coverage beside performance."""

    performance_by_key = {
        (str(row["method_id"]), str(row["metric"])): row for row in performance
    }
    coverage_by_method = {str(row["method_id"]): row for row in coverage}
    output: list[dict[str, object]] = []
    for method_id in method_ids:
        coverage_row = coverage_by_method.get(method_id, {})
        for metric, direction in SGEYESUB_METRIC_DIRECTIONS:
            performance_row = performance_by_key.get((method_id, metric), {})
            output.append(
                {
                    "method_id": method_id,
                    "metric": metric,
                    "direction": direction,
                    "registered_record_count": int(
                        coverage_row.get("record_count", 0)
                    ),
                    "success_count": int(coverage_row.get("success_count", 0)),
                    "fallback_count": int(coverage_row.get("fallback_count", 0)),
                    "blocked_count": int(coverage_row.get("blocked_count", 0)),
                    "ineligible_count": int(
                        coverage_row.get("ineligible_count", 0)
                    ),
                    "failed_count": int(coverage_row.get("failed_count", 0)),
                    "performance_available": bool(performance_row),
                    "finite_performance_count": int(
                        performance_row.get("participant_stem_count", 0)
                    ),
                    "mean": performance_row.get("mean"),
                    "median": performance_row.get("median"),
                    "mean_ci95_low": performance_row.get("ci95_low"),
                    "mean_ci95_high": performance_row.get("ci95_high"),
                    "bootstrap_replicates": performance_row.get(
                        "bootstrap_replicates"
                    ),
                    "bootstrap_seed": performance_row.get("bootstrap_seed"),
                    "performance_policy": "success_nonfallback_finite_only",
                    "coverage_policy": "all_registered_recording_keys",
                }
            )
    return output


def _render_corrected_audit_report(
    *,
    audit: Mapping[str, object],
    performance: Sequence[Mapping[str, object]],
    coverage: Sequence[Mapping[str, object]],
    paired_summary: Sequence[Mapping[str, object]],
    heterogeneity: Sequence[Mapping[str, object]],
    safety: Sequence[Mapping[str, object]],
    gamma_summary: Sequence[Mapping[str, object]],
    frozen_gamma: float,
) -> str:
    lines = [
        "# SGEYESUB corrected operator audit",
        "",
        "This audit is additive and read-only with respect to earlier result files. "
        "Blocked, ineligible, failed, and fallback rows remain in coverage "
        "denominators but are excluded from performance means.",
        "",
        f"Audit status: `{audit['status']}`.",
        "",
        f"Frozen development gamma: `{frozen_gamma:g}`.",
        "",
        "Scientific interpretation: `hard_Q_P0_tradeoff_inconclusive`.",
        "This is a post-hoc descriptive audit, is non-preregistered, and is not "
        "formal gate evidence. Matching P0 showed descriptively lower held-out "
        "EOG remaining ratios and higher coherence reduction than population. "
        "Non-artifact preservation and covariance/PSD distortion were roughly "
        "tied (their descriptive paired confidence intervals spanned zero), "
        "while the ERP preservation proxy was slightly lower; the absolute "
        "hard-Q safety thresholds were not met. No broad category-level failure "
        "decision is generated.",
        "Required focus/control coverage: "
        f"`{audit['required_method_record_coverage_status']}`.",
    ]
    if frozen_gamma == 0.0:
        lines.extend(
            [
                "",
                "Interpretation: `development_selected_population_endpoint`. "
                + GAMMA_ZERO_STRUCTURAL_NOTE
                + ". Evaluation continues; this endpoint is not itself a negative "
                "held-out personalization result.",
            ]
        )
    if gamma_summary:
        lines.extend(
            [
                "",
                "## Development gamma score components",
                "",
                "| Gamma | Success records | Mean stability | Mean capture loss | Weighted capture | Mean score |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in gamma_summary:
            lines.append(
                f"| {row['gamma']} | {row['successful_component_count']} | "
                f"{row.get('mean_split_half_stability')} | "
                f"{row.get('mean_heldout_contamination_capture_loss')} | "
                f"{row.get('mean_weighted_capture_component')} | "
                f"{row.get('mean_support_score')} |"
            )
    method_groups = _corrected_method_groups(frozen_gamma)
    focus_view = _method_metric_status_view(
        performance, coverage, method_ids=method_groups["focus"]
    )
    control_view = _method_metric_status_view(
        performance, coverage, method_ids=method_groups["controls"]
    )
    lines.extend(
        [
            "",
            "## Required methods: nine metrics and status coverage",
            "",
            "Status columns are success/fallback/blocked/ineligible/failed over "
            "all registered records; numeric summaries use only successful, "
            "non-fallback finite rows.",
            "",
            "| Method | Metric | Direction | Finite N | Mean | Median | S/F/B/I/X |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in focus_view:
        lines.append(
            f"| {row['method_id']} | {row['metric']} | {row['direction']} | "
            f"{row['finite_performance_count']} | {row.get('mean')} | "
            f"{row.get('median')} | {row['success_count']}/"
            f"{row['fallback_count']}/{row['blocked_count']}/"
            f"{row['ineligible_count']}/{row['failed_count']} |"
        )
    lines.extend(
        [
            "",
            "## Wrong and shuffled controls",
            "",
            "| Method | Metric | Direction | Finite N | Mean | Median | S/F/B/I/X |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in control_view:
        lines.append(
            f"| {row['method_id']} | {row['metric']} | {row['direction']} | "
            f"{row['finite_performance_count']} | {row.get('mean')} | "
            f"{row.get('median')} | {row['success_count']}/"
            f"{row['fallback_count']}/{row['blocked_count']}/"
            f"{row['ineligible_count']}/{row['failed_count']} |"
        )
    lines.extend(
        [
            "",
            "## Matching versus population",
            "",
            f"Compatible stems: `{audit['compatible_record_count']}`; successful "
            f"method pairs: `{audit['method_success_paired_count']}`. Positive "
            "directional improvement always favors matching P0; raw delta is "
            "always `matching − population` before direction adjustment.",
            "",
            "| Metric | Finite pairs | Raw mean delta | Raw median delta | Raw mean 95% CI | Directional mean | Wins/Ties/Losses |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in paired_summary:
        lines.append(
            f"| {row['metric']} | {row['finite_metric_paired_count']} | "
            f"{row.get('mean_matching_minus_population')} | "
            f"{row.get('median_matching_minus_population')} | "
            f"[{row.get('mean_matching_minus_population_ci95_low')}, "
            f"{row.get('mean_matching_minus_population_ci95_high')}] | "
            f"{row.get('mean_directional_improvement')} | "
            f"{row['matching_wins']}/{row['ties']}/{row['matching_losses']} |"
        )
    lines.extend(
        [
            "",
            "## Study heterogeneity",
            "",
            "| Study | Metric | Finite/compatible | Raw mean delta | Raw median delta | Directional mean | Wins/Ties/Losses | Raw mean 95% CI |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in heterogeneity:
        lines.append(
            f"| {row['study']} | {row['metric']} | "
            f"{row['finite_metric_paired_count']}/{row['compatible_record_count']} | "
            f"{row.get('mean_matching_minus_population')} | "
            f"{row.get('median_matching_minus_population')} | "
            f"{row.get('mean_directional_improvement')} | "
            f"{row['matching_wins']}/{row['ties']}/{row['matching_losses']} | "
            f"[{row.get('mean_matching_minus_population_ci95_low')}, "
            f"{row.get('mean_matching_minus_population_ci95_high')}] |"
        )
    lines.extend(
        [
            "",
            "## Hard-Q absolute safety",
            "",
            "| Method | Finite rows | Mean/median preservation | Mean/median covariance distortion | Joint passes | Pass fraction finite/all |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    safety_by_method = {str(row["method_id"]): row for row in safety}
    for method_id in method_groups["hard_q_safety"]:
        row = safety_by_method.get(method_id)
        if row is None:
            continue
        lines.append(
            f"| {row['method_id']} | {row['finite_joint_safety_count']} | "
            f"{row.get('mean_nonartifact_observation_preservation')}/"
            f"{row.get('median_nonartifact_observation_preservation')} | "
            f"{row.get('mean_reference_free_covariance_distortion')}/"
            f"{row.get('median_reference_free_covariance_distortion')} | "
            f"{row['joint_safety_pass_count']} | "
            f"{row.get('joint_safety_pass_fraction_finite')}/"
            f"{row.get('joint_safety_pass_fraction_all_registered')} |"
        )
    lines.extend(
        [
            "",
            "## Additional focus-method safety",
            "",
            "| Method | Finite rows | Mean/median preservation | Mean/median covariance distortion | Joint passes | Pass fraction finite/all |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method_id in method_groups["additional_safety"]:
        row = safety_by_method.get(method_id)
        if row is None:
            continue
        lines.append(
            f"| {row['method_id']} | {row['finite_joint_safety_count']} | "
            f"{row.get('mean_nonartifact_observation_preservation')}/"
            f"{row.get('median_nonartifact_observation_preservation')} | "
            f"{row.get('mean_reference_free_covariance_distortion')}/"
            f"{row.get('median_reference_free_covariance_distortion')} | "
            f"{row['joint_safety_pass_count']} | "
            f"{row.get('joint_safety_pass_fraction_finite')}/"
            f"{row.get('joint_safety_pass_fraction_all_registered')} |"
        )
    lines.extend(
        [
            "",
            "The study heterogeneity table contains "
            f"`{len(heterogeneity)}` metric-study rows. The CSV safety table retains "
            "all methods; the report separately shows hard-Q and additional focus "
            "methods. No clean-target claim is made.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_corrected_audit(
    config: Mapping[str, object],
    *,
    protocol_rows: Sequence[SgeyesubProtocolRow],
    rows: Sequence[Mapping[str, object]],
    frozen_gamma: float,
    development_rows: Sequence[SgeyesubProtocolRow] = (),
) -> dict[str, object]:
    audit_config = _mapping(config, "corrected_audit")
    if int(audit_config.get("expected_compatible_records", -1)) != 43:
        raise ValueError("corrected audit must retain the 43 compatible stems")
    replicates = int(audit_config.get("bootstrap_replicates", -1))
    seed = int(audit_config.get("bootstrap_seed", -1))
    if replicates != 20_000 or seed != 20260802:
        raise ValueError("corrected audit bootstrap contract changed")
    if audit_config.get("historical_results_policy") != (
        "read_only_side_by_side_no_overwrite"
    ):
        raise ValueError("corrected audit cannot overwrite historical results")

    output_root = Path(str(audit_config["output_root"]))
    report_path = Path(str(audit_config["report_path"]))
    output_root.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    (output_root / "resolved_config.yaml").write_text(
        yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8"
    )
    performance = _method_summary(
        rows,
        partition="evaluation_corrected",
        bootstrap_replicates=replicates,
        bootstrap_seed=seed,
    )
    coverage = _method_coverage_summary(rows, partition="evaluation_corrected")
    paired = _matching_population_audit(
        rows,
        protocol_rows,
        bootstrap_replicates=replicates,
        bootstrap_seed=seed,
    )
    safety = _absolute_safety_summary(rows, config=config)
    gamma_components = (
        _development_gamma_component_audit(config, development_rows)
        if development_rows
        else {"component_rows": [], "summary_rows": []}
    )
    method_groups = _corrected_method_groups(frozen_gamma)
    required_method_coverage = _required_method_record_coverage(
        rows,
        protocol_rows,
        method_ids=method_groups["focus"] + method_groups["controls"],
    )
    final_status = (
        str(paired["status"])
        if required_method_coverage["all_required_methods_complete"]
        else "inconclusive_incomplete_required_method_record_coverage"
    )
    focus_method_view = _method_metric_status_view(
        performance, coverage, method_ids=method_groups["focus"]
    )
    control_method_view = _method_metric_status_view(
        performance, coverage, method_ids=method_groups["controls"]
    )
    safety_by_method = {str(row["method_id"]): row for row in safety}
    hard_q_safety = [
        safety_by_method[method_id]
        for method_id in method_groups["hard_q_safety"]
        if method_id in safety_by_method
    ]
    additional_focus_safety = [
        safety_by_method[method_id]
        for method_id in method_groups["additional_safety"]
        if method_id in safety_by_method
    ]
    _write_csv(output_root / "method_performance_success_only.csv", performance)
    _write_csv(output_root / "method_coverage.csv", coverage)
    _write_csv(
        output_root / "matching_population_pairs.csv",
        paired["pair_rows"],  # type: ignore[arg-type]
    )
    _write_csv(
        output_root / "matching_population_summary.csv",
        paired["summary_rows"],  # type: ignore[arg-type]
    )
    _write_csv(
        output_root / "matching_population_study_heterogeneity.csv",
        paired["heterogeneity_rows"],  # type: ignore[arg-type]
    )
    _write_csv(output_root / "absolute_safety.csv", safety)
    _write_csv(
        output_root / "required_focus_method_metric_status.csv", focus_method_view
    )
    _write_csv(output_root / "control_method_metric_status.csv", control_method_view)
    _write_csv(output_root / "hard_q_absolute_safety.csv", hard_q_safety)
    if development_rows:
        _write_csv(
            output_root / "development_gamma_score_components.csv",
            gamma_components["component_rows"],  # type: ignore[arg-type]
        )
        _write_csv(
            output_root / "development_gamma_score_summary.csv",
            gamma_components["summary_rows"],  # type: ignore[arg-type]
        )
    summary = {
        "status": final_status,
        "matching_population_pair_status": paired["status"],
        "required_method_record_coverage_status": required_method_coverage[
            "status"
        ],
        "required_method_record_coverage": required_method_coverage["methods"],
        "audit_version": "sgeyesub_corrected_operator_audit_v2",
        "historical_results_policy": "read_only_side_by_side_no_overwrite",
        "scientific_interpretation": "hard_Q_P0_tradeoff_inconclusive",
        "audit_scope": "post_hoc_descriptive_audit_non_preregistered",
        "formal_gate_evidence": False,
        "formal_operator_specificity_decision": "not_generated_post_hoc_audit",
        "descriptive_pattern": {
            "matching_heldout_eog_remaining": "post_hoc_lower_than_population",
            "matching_eog_coherence_reduction": "post_hoc_higher_than_population",
            "matching_nonartifact_preservation": "roughly_tied_ci_spans_zero",
            "matching_covariance_psd_distortion": "roughly_tied_ci_spans_zero",
            "matching_erp_preservation_proxy": "post_hoc_lower_than_population",
            "absolute_hard_q_safety_thresholds": "not_met",
        },
        "frozen_development_gamma": frozen_gamma,
        "gamma_interpretation": (
            "development_selected_population_endpoint"
            if frozen_gamma == 0.0
            else "development_selected_positive_context_weight"
        ),
        "gamma_zero_structural_note": (
            GAMMA_ZERO_STRUCTURAL_NOTE if frozen_gamma == 0.0 else None
        ),
        "development_gamma_score_summary": gamma_components["summary_rows"],
        "required_focus_method_ids": list(method_groups["focus"]),
        "required_focus_method_metric_status": focus_method_view,
        "control_method_ids": list(method_groups["controls"]),
        "control_method_metric_status": control_method_view,
        "hard_q_absolute_safety": hard_q_safety,
        "additional_focus_method_safety": additional_focus_safety,
        "matching_population_summary": paired["summary_rows"],
        "matching_population_study_heterogeneity": paired["heterogeneity_rows"],
        "registered_record_count": paired["registered_record_count"],
        "compatible_record_count": paired["compatible_record_count"],
        "method_success_paired_count": paired["method_success_paired_count"],
        "method_success_paired_fraction": paired["method_success_paired_fraction"],
        "blocked_singleton_recording_key": paired[
            "blocked_singleton_recording_key"
        ],
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "performance_row_policy": "success_nonfallback_finite_only",
        "coverage_denominator_policy": "all_44_registered_recording_keys",
        "paths": {
            "resolved_config": str(output_root / "resolved_config.yaml"),
            "method_performance": str(
                output_root / "method_performance_success_only.csv"
            ),
            "method_coverage": str(output_root / "method_coverage.csv"),
            "matching_population_pairs": str(
                output_root / "matching_population_pairs.csv"
            ),
            "matching_population_summary": str(
                output_root / "matching_population_summary.csv"
            ),
            "matching_population_study_heterogeneity": str(
                output_root / "matching_population_study_heterogeneity.csv"
            ),
            "absolute_safety": str(output_root / "absolute_safety.csv"),
            "required_focus_method_metric_status": str(
                output_root / "required_focus_method_metric_status.csv"
            ),
            "control_method_metric_status": str(
                output_root / "control_method_metric_status.csv"
            ),
            "hard_q_absolute_safety": str(
                output_root / "hard_q_absolute_safety.csv"
            ),
            "development_gamma_score_components": (
                str(output_root / "development_gamma_score_components.csv")
                if development_rows
                else None
            ),
            "development_gamma_score_summary": (
                str(output_root / "development_gamma_score_summary.csv")
                if development_rows
                else None
            ),
            "report": str(report_path),
        },
    }
    (output_root / "result_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    report_path.write_text(
        _render_corrected_audit_report(
            audit={
                **paired,
                "status": final_status,
                "required_method_record_coverage_status": (
                    required_method_coverage["status"]
                ),
            },
            performance=performance,
            coverage=coverage,
            paired_summary=paired["summary_rows"],  # type: ignore[arg-type]
            heterogeneity=paired["heterogeneity_rows"],  # type: ignore[arg-type]
            safety=safety,
            gamma_summary=gamma_components["summary_rows"],  # type: ignore[arg-type]
            frozen_gamma=frozen_gamma,
        ),
        encoding="utf-8",
    )
    return summary


def _native_baseline_output(
    loaded: SgeyesubLoadedRecord,
    observed: np.ndarray,
) -> tuple[
    NativeSGEyeSubFitOutcome,
    tuple[str, np.ndarray, str, str, None, bool, bool],
]:
    """Fit official block-1 labels and apply without block-2 labels."""

    if loaded.query is None:
        raise AssertionError("native baseline requires observed block-2 EEG")
    native_types = ("EEG",) * len(loaded.native_channel_labels)
    native_outcome = fit_native_sgeyesub(
        loaded.support.native_eeg,
        loaded.support.artifactclasses,
        channel_labels=loaded.native_channel_labels,
        channel_types=native_types,
        layout_id=loaded.release_layout_id,
        block_id=1,
    )
    if native_outcome.model is None:
        output = (
            "native_sgeyesub_python_release_internal",
            observed.copy(),
            "ineligible_native_fit_identity_no_claim",
            ";".join(native_outcome.reasons),
            None,
            False,
            False,
        )
    else:
        corrected_native = native_outcome.model.apply(
            loaded.query.native_eeg,
            channel_labels=loaded.native_channel_labels,
            channel_types=native_types,
            layout_id=loaded.release_layout_id,
            block_id=2,
        )
        native_p0_indices = np.asarray(
            [
                loaded.native_channel_labels.index(label)
                for label in loaded.p0_channel_labels
            ],
            dtype=int,
        )
        output = (
            "native_sgeyesub_python_release_internal",
            corrected_native[native_p0_indices],
            "success_source_faithful_not_matlab_cross_validated",
            "official_commit_2c95b4f_python_port",
            None,
            False,
            False,
        )
    return native_outcome, output


def _run_sgeyesub_singleton_record(
    config: Mapping[str, object],
    *,
    target: SgeyesubReleaseRecord,
    layout: SgeyesubLayout,
    run_dir: Path,
    partition: str,
    frozen_gamma: float | None,
    started: float,
) -> dict[str, object]:
    """Preserve the exact singleton cell without borrowing another layout."""

    if partition != "evaluation" or frozen_gamma is None:
        raise RuntimeError("a development record unexpectedly lacks a population cell")
    root = Path(str(config["data_root"]))
    loaded = load_sgeyesub_signal_record(
        root,
        target,
        layout,
        include_query=True,
        include_query_annotations=False,
    )
    if loaded.query is None or loaded.query_annotations is not None:
        raise AssertionError("singleton target query isolation failed")
    support_eog, _ = _standardize_eog(loaded.support.external_eog)
    p0 = _mapping(config, "p0")
    strict_context = _fit(
        loaded.support.eeg,
        support_eog,
        participant=target.participant_stem,
        sampling_rate=target.sampling_rate_hz,
        p0_config=_p0_config(config, relaxed=False),
        movement_threshold=float(p0["movement_threshold"]),
    )
    observed = np.asarray(loaded.query.eeg, dtype=np.float64)
    outputs: list[tuple[str, np.ndarray, str, str, float | None, bool, bool]] = [
        ("raw", observed.copy(), "success", "raw", None, False, False),
        (
            "pop_Qy",
            observed.copy(),
            "blocked_no_population_identity_no_claim",
            "singleton_exact_layout_cell",
            None,
            False,
            False,
        ),
        (
            "POP_fallback",
            observed.copy(),
            "blocked_no_population_identity_no_claim",
            "singleton_exact_layout_cell",
            None,
            False,
            False,
        ),
        (
            "wrong_Qy",
            observed.copy(),
            "blocked_no_same_cell_wrong_source",
            "singleton_exact_layout_cell",
            None,
            False,
            False,
        ),
    ]
    matching_projector = (
        None if strict_context.transfer is None else strict_context.transfer.projector
    )
    if matching_projector is None:
        outputs.append(
            (
                "matching_Qy",
                observed.copy(),
                "ineligible_matching_P0_identity_no_claim",
                ";".join(strict_context.reasons),
                None,
                False,
                False,
            )
        )
    else:
        outputs.append(
            (
                "matching_Qy",
                _q_restore(matching_projector, observed),
                "success",
                "matching_block1_P0",
                None,
                False,
                False,
            )
        )
    shift = max(
        1,
        int(
            round(
                support_eog.shape[1]
                * float(p0["shuffled_eog_fractional_shift"])
            )
        ),
    )
    shuffled_fit = _fit(
        loaded.support.eeg,
        np.roll(support_eog, shift=shift, axis=1),
        participant=target.participant_stem,
        sampling_rate=target.sampling_rate_hz,
        p0_config=_p0_config(config, relaxed=False),
        movement_threshold=float(p0["movement_threshold"]),
    )
    if shuffled_fit.transfer is None:
        outputs.append(
            (
                "shuffled_Qy",
                observed.copy(),
                "ineligible_shuffled_P0_identity_no_claim",
                ";".join(shuffled_fit.reasons),
                None,
                False,
                False,
            )
        )
    else:
        outputs.append(
            (
                "shuffled_Qy",
                _q_restore(shuffled_fit.transfer.projector, observed),
                "success",
                "circularly_shifted_block1_EOG",
                None,
                False,
                False,
            )
        )
    gamma_token = _gamma_token(frozen_gamma)
    for method_id in (
        f"B6_Qy__gamma_{gamma_token}",
        f"B6_soft_proximal__gamma_{gamma_token}",
    ):
        outputs.append(
            (
                method_id,
                observed.copy(),
                "blocked_no_population_identity_no_claim",
                "singleton_exact_layout_cell",
                frozen_gamma,
                False,
                False,
            )
        )
    native_outcome, native_output = _native_baseline_output(loaded, observed)
    outputs.append(native_output)

    annotated = load_sgeyesub_signal_record(
        root,
        target,
        layout,
        include_query=True,
        include_query_annotations=True,
    )
    if annotated.query_annotations is None:
        raise AssertionError("singleton query annotations were not loaded")
    query_annotations = annotated.query_annotations
    _, query_eog = _standardize_eog(
        loaded.support.external_eog,
        query_annotations.external_eog,
    )
    if query_eog is None:
        raise AssertionError("singleton query EOG standardization failed")
    heldout_predicted_contamination = None
    if strict_context.transfer is not None:
        heldout_predicted_contamination = (
            strict_context.transfer.transfer_matrix
            @ (query_eog - strict_context.transfer.eog_mean)
        )

    metric_rows = [
        {
            "partition": partition,
            "study": target.study,
            "participant_stem": target.participant_stem,
            "recording_key": target.recording_key,
            "release_layout_id": target.layout_id,
            "p0_layout_id": target.p0_layout_id,
            "sampling_rate_hz": target.sampling_rate_hz,
            "support_block": 1,
            "query_block": 2,
            "population_source_count": 0,
            **_evaluate_output(
                method_id=method_id,
                output=output,
                observed=observed,
                matching_projector=matching_projector,
                population_projector=None,
                query_eog=query_eog,
                artifactclasses=query_annotations.artifactclasses,
                predicted_contamination=heldout_predicted_contamination,
                trial_labels=query_annotations.trial_labels,
                samples_per_trial=target.samples_per_trial,
                minimum_trials_per_condition=int(
                    _mapping(config, "evaluation_metrics")[
                        "minimum_query_trials_per_condition"
                    ]
                ),
                status=status,
                operator_source=operator_source,
                gamma=gamma,
                fallback_used=fallback,
                uses_query_external_eog=uses_query_eog,
            ),
        }
        for method_id, output, status, operator_source, gamma, fallback, uses_query_eog in outputs
    ]
    output_root = Path(str(config["evaluation_output_root"])) / target.participant_stem
    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "metrics.csv", metric_rows)
    summary = {
        "status": "completed_with_blocked_no_population",
        "partition": partition,
        "study": target.study,
        "participant_stem": target.participant_stem,
        "recording_key": target.recording_key,
        "release_layout_id": target.layout_id,
        "population_source_count": 0,
        "population_status": "blocked_no_population",
        "cross_layout_pooling_used": False,
        "frozen_development_gamma": frozen_gamma,
        "matching_p0_status": strict_context.status,
        "matching_p0_reasons": list(strict_context.reasons),
        "shuffled_p0_status": shuffled_fit.status,
        "native_sgeyesub_status": native_outcome.status,
        "native_sgeyesub_reasons": list(native_outcome.reasons),
        "native_reference_equivalence_status": REFERENCE_EQUIVALENCE_STATUS,
        "query_annotations_used_for_gamma": False,
        "query_annotations_used_for_fit_or_method_selection": False,
        "query_annotations_opened_after_method_outputs_frozen": True,
        "elapsed_seconds": time.monotonic() - started,
        "metrics": str(output_root / "metrics.csv"),
    }
    (output_root / "result_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _run_sgeyesub_record(
    config: Mapping[str, object],
    *,
    task_index: int,
    run_dir: Path,
    partition: str,
    frozen_gamma: float | None,
) -> dict[str, object]:
    """Run deterministic controls for one non-singleton release record."""

    validate_sgeyesub_protocol_config(config)
    if partition not in {"development", "evaluation"}:
        raise ValueError("SGEYESUB partition must be development or evaluation")
    expected_tasks = 15 if partition == "development" else 44
    if not 0 <= task_index < expected_tasks:
        raise ValueError(
            f"SGEYESUB {partition} task index must lie in [0, {expected_tasks - 1}]"
        )
    if partition == "development" and frozen_gamma is not None:
        raise ValueError("development must score all gamma candidates")
    if partition == "evaluation" and frozen_gamma is None:
        raise ValueError("evaluation requires a frozen development gamma")
    metadata = _mapping(config, "metadata")
    layouts, records = load_sgeyesub_structure_audit(
        Path(str(metadata["structure_audit_result"]))
    )
    layout_map = {layout.layout_id: layout for layout in layouts}
    selected_studies = (
        SGEYESUB_DEVELOPMENT_STUDIES
        if partition == "development"
        else SGEYESUB_EVALUATION_STUDIES
    )
    partition_records = tuple(
        sorted(
            (record for record in records if record.study in selected_studies),
            key=lambda record: record.recording_key,
        )
    )
    target = partition_records[task_index]
    sources = tuple(
        record
        for record in records
        if record.recording_key != target.recording_key
        and _cell_key(record) == _cell_key(target)
    )
    root = Path(str(config["data_root"]))
    started = time.monotonic()
    if not sources:
        return _run_sgeyesub_singleton_record(
            config,
            target=target,
            layout=layout_map[target.layout_id],
            run_dir=run_dir,
            partition=partition,
            frozen_gamma=frozen_gamma,
            started=started,
        )
    loaded = load_sgeyesub_signal_record(
        root,
        target,
        layout_map[target.layout_id],
        include_query=True,
        include_query_annotations=False,
    )
    if loaded.query is None or loaded.query_annotations is not None:
        raise AssertionError("target query EEG/annotation isolation failed")
    support_eog, _ = _standardize_eog(loaded.support.external_eog)
    population, population_sources = _population_fit(
        root=root,
        target=target,
        sources=sources,
        layout_map=layout_map,
        config=config,
    )
    p0 = _mapping(config, "p0")
    strict_context = _fit(
        loaded.support.eeg,
        support_eog,
        participant=target.participant_stem,
        sampling_rate=target.sampling_rate_hz,
        p0_config=_p0_config(config, relaxed=False),
        movement_threshold=float(p0["movement_threshold"]),
    )
    relaxed_context = _fit(
        loaded.support.eeg,
        support_eog,
        participant=target.participant_stem,
        sampling_rate=target.sampling_rate_hz,
        p0_config=_p0_config(config, relaxed=True),
        movement_threshold=float(p0["movement_threshold"]),
    )
    compatibility = _compatibility(
        loaded,
        reference_id=str(_mapping(config, "compatibility_cell")["reference_cell_id"]),
    )
    support_scores = None
    if partition == "development":
        if population.transfer is None:
            support_scores = [
                _gamma_support_score_row(
                    gamma=float(gamma),
                    status="unavailable_population_operator",
                    stability=(0.0 if float(gamma) == 0.0 else None),
                    capture_loss=None,
                    capture_weight=float(
                        _mapping(config, "b6_pop_shrink")[
                            "heldout_contamination_capture_weight"
                        ]
                    ),
                    support_score=None,
                )
                for gamma in _mapping(config, "b6_pop_shrink")["gamma_candidates"]
            ]
        else:
            support_scores = _gamma_support_scores(
                population_projector=population.transfer.projector,
                context_outcome=relaxed_context,
                support_eeg=np.asarray(loaded.support.eeg, dtype=np.float64),
                support_eog=support_eog,
                compatibility=compatibility,
                config=config,
                sampling_rate=target.sampling_rate_hz,
            )
    output_root_key = (
        "development_output_root"
        if partition == "development"
        else "evaluation_output_root"
    )
    output_root = (
        Path(str(config[output_root_key])) / target.participant_stem
    )
    output_root.mkdir(parents=True, exist_ok=True)
    if support_scores is not None:
        (output_root / "support_gamma_scores.json").write_text(
            json.dumps(support_scores, indent=2) + "\n", encoding="utf-8"
        )

    observed = np.asarray(loaded.query.eeg, dtype=np.float64)
    pi0 = None if population.transfer is None else population.transfer.projector
    outputs: list[tuple[str, np.ndarray, str, str, float | None, bool, bool]] = []
    outputs.append(("raw", observed.copy(), "success", "raw", None, False, False))
    if pi0 is None:
        pop_output = observed.copy()
        outputs.append(
            (
                "pop_Qy",
                pop_output.copy(),
                "failed_population_operator_identity_no_claim",
                "same_cell_population_P0_ineligible",
                None,
                False,
                False,
            )
        )
        outputs.append(
            (
                "POP_fallback",
                pop_output.copy(),
                "failed_population_operator_identity_no_claim",
                "same_cell_population_P0_ineligible",
                None,
                False,
                False,
            )
        )
    else:
        pop_output = _q_restore(pi0, observed)
        outputs.append(("pop_Qy", pop_output, "success", "Pi0", None, False, False))
        outputs.append(
            ("POP_fallback", pop_output.copy(), "success", "Pi0", None, True, False)
        )

    if strict_context.transfer is None:
        matching_status = (
            "fallback_POP"
            if pi0 is not None
            else "ineligible_matching_P0_identity_no_claim"
        )
        outputs.append(
            (
                "matching_Qy",
                pop_output.copy(),
                matching_status,
                "matching_P0_ineligible",
                None,
                pi0 is not None,
                False,
            )
        )
        matching_projector = None
    else:
        outputs.append(
            (
                "matching_Qy",
                _q_restore(strict_context.transfer.projector, observed),
                "success",
                "matching_block1_P0",
                None,
                False,
                False,
            )
        )
        matching_projector = strict_context.transfer.projector

    wrong_source = sorted(sources, key=lambda record: record.recording_key)[0]
    wrong_loaded = load_sgeyesub_signal_record(
        root,
        wrong_source,
        layout_map[wrong_source.layout_id],
        include_query=False,
    )
    wrong_eog, _ = _standardize_eog(wrong_loaded.support.external_eog)
    wrong_fit = _fit(
        wrong_loaded.support.eeg,
        wrong_eog,
        participant=wrong_source.participant_stem,
        sampling_rate=wrong_source.sampling_rate_hz,
        p0_config=_p0_config(config, relaxed=False),
        movement_threshold=float(p0["movement_threshold"]),
    )
    if wrong_fit.transfer is None:
        wrong_status = (
            "fallback_POP"
            if pi0 is not None
            else "ineligible_wrong_P0_identity_no_claim"
        )
        outputs.append(
            (
                "wrong_Qy",
                pop_output.copy(),
                wrong_status,
                "wrong_P0_ineligible",
                None,
                pi0 is not None,
                False,
            )
        )
    else:
        outputs.append(
            (
                "wrong_Qy",
                _q_restore(wrong_fit.transfer.projector, observed),
                "success",
                wrong_source.participant_stem,
                None,
                False,
                False,
            )
        )

    shift = max(
        1,
        int(
            round(
                support_eog.shape[1]
                * float(p0["shuffled_eog_fractional_shift"])
            )
        ),
    )
    shuffled_fit = _fit(
        loaded.support.eeg,
        np.roll(support_eog, shift=shift, axis=1),
        participant=target.participant_stem,
        sampling_rate=target.sampling_rate_hz,
        p0_config=_p0_config(config, relaxed=False),
        movement_threshold=float(p0["movement_threshold"]),
    )
    if shuffled_fit.transfer is None:
        shuffled_status = (
            "fallback_POP"
            if pi0 is not None
            else "ineligible_shuffled_P0_identity_no_claim"
        )
        outputs.append(
            (
                "shuffled_Qy",
                pop_output.copy(),
                shuffled_status,
                "shuffled_P0_ineligible",
                None,
                pi0 is not None,
                False,
            )
        )
    else:
        outputs.append(
            (
                "shuffled_Qy",
                _q_restore(shuffled_fit.transfer.projector, observed),
                "success",
                "circularly_shifted_block1_EOG",
                None,
                False,
                False,
            )
        )

    b6 = _mapping(config, "b6_pop_shrink")
    gamma_values = (
        tuple(b6["gamma_candidates"])
        if partition == "development"
        else (float(frozen_gamma),)
    )
    for gamma_value in gamma_values:
        gamma = float(gamma_value)
        token = _gamma_token(gamma)
        if pi0 is None:
            for method_id in (
                f"B6_Qy__gamma_{token}",
                f"B6_soft_proximal__gamma_{token}",
            ):
                outputs.append(
                    (
                        method_id,
                        observed.copy(),
                        "failed_population_operator_identity_no_claim",
                        "same_cell_population_P0_ineligible",
                        gamma,
                        False,
                        False,
                    )
                )
            continue
        if gamma == 0.0:
            context_projector = None
            context_eligible = False
            context_compatibility = None
            context_fit_scope = None
        elif relaxed_context.transfer is not None:
            context_projector = relaxed_context.transfer.projector
            context_eligible = True
            context_compatibility = compatibility
            context_fit_scope = "support_only"
        else:
            context_projector = None
            context_eligible = False
            context_compatibility = None
            context_fit_scope = None
        if gamma > 0.0 and relaxed_context.transfer is None:
            outputs.append(
                (
                    f"B6_Qy__gamma_{token}",
                    pop_output.copy(),
                    "fallback_POP",
                    "B6_context_unavailable",
                    gamma,
                    True,
                    False,
                )
            )
            outputs.append(
                (
                    f"B6_soft_proximal__gamma_{token}",
                    pop_output.copy(),
                    "fallback_POP",
                    "B6_context_unavailable",
                    gamma,
                    True,
                    False,
                )
            )
            continue
        outcome = spectral_projector_shrink(
            pi0,
            context_projector,
            rank=int(p0["rank"]),
            gamma=gamma,
            context_eligible=context_eligible,
            population_compatibility=compatibility,
            population_fit_scope="outer_training_only",
            context_compatibility=context_compatibility,
            context_fit_scope=context_fit_scope,
        )
        if outcome.status != "eligible" or outcome.projector is None:
            outputs.append(
                (
                    f"B6_Qy__gamma_{token}",
                    pop_output.copy(),
                    "fallback_POP",
                    "B6_ineligible",
                    gamma,
                    True,
                    False,
                )
            )
            outputs.append(
                (
                    f"B6_soft_proximal__gamma_{token}",
                    pop_output.copy(),
                    "fallback_POP",
                    "B6_ineligible",
                    gamma,
                    True,
                    False,
                )
            )
        else:
            outputs.append(
                (
                    f"B6_Qy__gamma_{token}",
                    _q_restore(outcome.projector, observed),
                    "success",
                    "B6_POP_SHRINK",
                    gamma,
                    False,
                    False,
                )
            )
            outputs.append(
                (
                    f"B6_soft_proximal__gamma_{token}",
                    _soft_restore(
                        outcome.projector,
                        observed,
                        float(b6["soft_proximal_tau"]),
                    ),
                    "success",
                    "B6_POP_SHRINK",
                    gamma,
                    False,
                    False,
                )
            )

    native_outcome, native_output = _native_baseline_output(loaded, observed)
    outputs.append(native_output)

    # Query external-EOG annotations are first accessed only after every
    # support-only fit, gamma score, and method output above is frozen.  They
    # score held-out metrics only and cannot create or change a method output.
    annotated = load_sgeyesub_signal_record(
        root,
        target,
        layout_map[target.layout_id],
        include_query=True,
        include_query_annotations=True,
    )
    if annotated.query_annotations is None:
        raise AssertionError("target query annotations were not loaded")
    query_annotations = annotated.query_annotations
    _, query_eog = _standardize_eog(
        loaded.support.external_eog,
        query_annotations.external_eog,
    )
    if query_eog is None:
        raise AssertionError("query EOG standardization unexpectedly absent")
    heldout_predicted_contamination = None
    if strict_context.transfer is not None:
        centered_query_eog = query_eog - strict_context.transfer.eog_mean
        heldout_predicted_contamination = (
            strict_context.transfer.transfer_matrix @ centered_query_eog
        )

    metric_rows = [
        {
            "partition": partition,
            "study": target.study,
            "participant_stem": target.participant_stem,
            "recording_key": target.recording_key,
            "release_layout_id": target.layout_id,
            "p0_layout_id": target.p0_layout_id,
            "sampling_rate_hz": target.sampling_rate_hz,
            "support_block": 1,
            "query_block": 2,
            "population_source_count": len(population_sources),
            **_evaluate_output(
                method_id=method_id,
                output=output,
                observed=observed,
                matching_projector=matching_projector,
                population_projector=pi0,
                query_eog=query_eog,
                artifactclasses=query_annotations.artifactclasses,
                predicted_contamination=heldout_predicted_contamination,
                trial_labels=query_annotations.trial_labels,
                samples_per_trial=target.samples_per_trial,
                minimum_trials_per_condition=int(
                    _mapping(config, "evaluation_metrics")[
                        "minimum_query_trials_per_condition"
                    ]
                ),
                status=status,
                operator_source=operator_source,
                gamma=gamma,
                fallback_used=fallback,
                uses_query_external_eog=uses_query_eog,
            ),
        }
        for method_id, output, status, operator_source, gamma, fallback, uses_query_eog in outputs
    ]
    _write_csv(output_root / "metrics.csv", metric_rows)
    summary = {
        "status": (
            "completed"
            if population.transfer is not None
            else "completed_with_failed_population_operator"
        ),
        "partition": partition,
        "study": target.study,
        "participant_stem": target.participant_stem,
        "recording_key": target.recording_key,
        "release_layout_id": target.layout_id,
        "participant_level": True,
        "support_block": 1,
        "query_block": 2,
        "frozen_development_gamma": (
            frozen_gamma if partition == "evaluation" else None
        ),
        "exact_population_cell": {
            "study": target.study,
            "release_layout_id": target.layout_id,
            "reference_cell_id": str(
                _mapping(config, "compatibility_cell")["reference_cell_id"]
            ),
            "sampling_rate_hz": target.sampling_rate_hz,
        },
        "population_sources": population_sources,
        "population_source_count": len(population_sources),
        "cross_layout_pooling_used": False,
        "same_cell_population_verified": True,
        "population_status": (
            "available"
            if population.transfer is not None
            else "failed_population_operator"
        ),
        "population_p0_status": population.status,
        "population_p0_reasons": list(population.reasons),
        "population_p0_diagnostics": population.diagnostics,
        "matching_p0_status": strict_context.status,
        "matching_p0_reasons": list(strict_context.reasons),
        "matching_p0_diagnostics": strict_context.diagnostics,
        "wrong_p0_status": wrong_fit.status,
        "wrong_p0_reasons": list(wrong_fit.reasons),
        "shuffled_p0_status": shuffled_fit.status,
        "shuffled_p0_reasons": list(shuffled_fit.reasons),
        "native_sgeyesub_status": native_outcome.status,
        "native_sgeyesub_reasons": list(native_outcome.reasons),
        "native_sgeyesub_diagnostics": native_outcome.diagnostics,
        "native_reference_equivalence_status": REFERENCE_EQUIVALENCE_STATUS,
        "native_eeg_chan_idxs_status": (
            "resolved_official_exact_layout_channel_type_EEG_commit_2c95b4f"
        ),
        "gamma_selection_inputs": (
            "support_only_development"
            if partition == "development"
            else "frozen_development_gamma"
        ),
        "query_annotations_used_for_gamma": False,
        "query_annotations_used_for_fit_or_method_selection": False,
        "query_annotations_access_phase": (
            "after_support_fit_gamma_and_all_method_outputs"
        ),
        "query_annotations_opened_after_method_outputs_frozen": True,
        "query_clean_target": "not_available",
        "elapsed_seconds": time.monotonic() - started,
        "metrics": str(output_root / "metrics.csv"),
        "support_gamma_scores": (
            str(output_root / "support_gamma_scores.json")
            if support_scores is not None
            else "N/A_evaluation_uses_frozen_gamma"
        ),
    }
    (output_root / "result_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def run_sgeyesub_development_record(
    config: Mapping[str, object], *, task_index: int, run_dir: Path
) -> dict[str, object]:
    return _run_sgeyesub_record(
        config,
        task_index=task_index,
        run_dir=run_dir,
        partition="development",
        frozen_gamma=None,
    )


def _load_frozen_development_gamma(config: Mapping[str, object]) -> float:
    path = Path(str(config["development_output_root"])) / "frozen_gamma.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen":
        raise ValueError("SGEYESUB development gamma is not frozen")
    gamma = float(payload["gamma"])
    candidates = tuple(
        float(value) for value in _mapping(config, "b6_pop_shrink")["gamma_candidates"]
    )
    if gamma not in candidates:
        raise ValueError("frozen SGEYESUB gamma is not a registered candidate")
    if payload.get("query_annotations_used") is not False:
        raise ValueError("frozen SGEYESUB gamma used forbidden query annotations")
    return gamma


def run_sgeyesub_evaluation_record(
    config: Mapping[str, object], *, task_index: int, run_dir: Path
) -> dict[str, object]:
    return _run_sgeyesub_record(
        config,
        task_index=task_index,
        run_dir=run_dir,
        partition="evaluation",
        frozen_gamma=_load_frozen_development_gamma(config),
    )


def run_sgeyesub_development_aggregate(
    config: Mapping[str, object], *, run_dir: Path
) -> dict[str, object]:
    """Freeze one global support-only gamma and select matching metric rows."""

    validate_sgeyesub_protocol_config(config)
    _write_resolved_config(config)
    metadata = _mapping(config, "metadata")
    layouts, records = load_sgeyesub_structure_audit(
        Path(str(metadata["structure_audit_result"]))
    )
    plan = build_sgeyesub_protocol(
        layouts,
        records,
        protocol_id=str(config["protocol_id"]),
        reference_cell_id=str(_mapping(config, "compatibility_cell")["reference_cell_id"]),
        gamma_candidates=tuple(_mapping(config, "b6_pop_shrink")["gamma_candidates"]),
    )
    development_root = Path(str(config["development_output_root"]))
    candidates = [float(value) for value in _mapping(config, "b6_pop_shrink")["gamma_candidates"]]
    by_gamma: dict[float, list[float]] = {gamma: [] for gamma in candidates}
    component_rows_by_gamma: dict[float, list[Mapping[str, object]]] = {
        gamma: [] for gamma in candidates
    }
    for protocol_row in plan.development_rows:
        participant_root = development_root / protocol_row.participant_stem
        score_rows = json.loads(
            (participant_root / "support_gamma_scores.json").read_text(encoding="utf-8")
        )
        for row in score_rows:
            if row["status"] == "success" and row["support_score"] is not None:
                gamma = float(row["gamma"])
                by_gamma[gamma].append(float(row["support_score"]))
                component_rows_by_gamma[gamma].append(row)

    minimum_fraction = float(
        _mapping(config, "b6_pop_shrink")["minimum_gamma_score_participant_fraction"]
    )
    selected_gamma, aggregate_scores = select_global_gamma(
        by_gamma,
        candidates=candidates,
        participant_count=len(plan.development_rows),
        minimum_fraction=minimum_fraction,
    )
    for aggregate_row in aggregate_scores:
        gamma = float(aggregate_row["gamma"])
        component_rows = component_rows_by_gamma[gamma]
        stability = [
            float(row["split_half_stability"])
            for row in component_rows
            if row.get("split_half_stability") is not None
        ]
        capture_loss = [
            float(row["heldout_contamination_capture_loss"])
            for row in component_rows
            if row.get("heldout_contamination_capture_loss") is not None
        ]
        weighted_capture = [
            float(row["capture_weight"])
            * float(row["heldout_contamination_capture_loss"])
            for row in component_rows
            if row.get("capture_weight") is not None
            and row.get("heldout_contamination_capture_loss") is not None
        ]
        aggregate_row.update(
            {
                "mean_split_half_stability": (
                    float(np.mean(stability)) if stability else None
                ),
                "mean_heldout_contamination_capture_loss": (
                    float(np.mean(capture_loss)) if capture_loss else None
                ),
                "mean_weighted_capture_component": (
                    float(np.mean(weighted_capture)) if weighted_capture else None
                ),
                "support_score_formula": (
                    "split_half_stability + capture_weight * "
                    "heldout_contamination_capture_loss"
                ),
                "population_endpoint": gamma == 0.0,
                "structural_zero_stability": gamma == 0.0,
                "structural_zero_explanation": (
                    GAMMA_ZERO_STRUCTURAL_NOTE if gamma == 0.0 else None
                ),
            }
        )
    development_root.mkdir(parents=True, exist_ok=True)
    frozen = {
        "status": "frozen",
        "gamma": selected_gamma,
        "selection_partition": "development",
        "selection_objective": (
            "support_only_split_half_stability_plus_heldout_contamination_capture"
        ),
        "heldout_contamination_capture_weight": 0.5,
        "query_annotations_used": False,
        "candidates": aggregate_scores,
        "tie_break": "smallest_gamma",
        "selected_gamma_interpretation": (
            "development_selected_population_endpoint"
            if selected_gamma == 0.0
            else "development_selected_positive_context_weight"
        ),
        "gamma_zero_structural_note": (
            GAMMA_ZERO_STRUCTURAL_NOTE if selected_gamma == 0.0 else None
        ),
        "evaluation_policy": "continue_with_frozen_endpoint",
    }
    (development_root / "frozen_gamma.json").write_text(
        json.dumps(frozen, indent=2) + "\n", encoding="utf-8"
    )
    # Query-derived evaluation metrics are opened only after gamma is frozen.
    all_metric_rows: list[dict[str, str]] = []
    for protocol_row in plan.development_rows:
        participant_root = development_root / protocol_row.participant_stem
        with (participant_root / "metrics.csv").open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            all_metric_rows.extend(csv.DictReader(stream))
    token = _gamma_token(selected_gamma)
    selected_rows = [
        row
        for row in all_metric_rows
        if not row["method_id"].startswith("B6_")
        or row["method_id"].endswith(f"__gamma_{token}")
    ]
    _write_csv(development_root / "metrics.csv", selected_rows)
    method_summary = _method_summary(selected_rows, partition="development")
    _write_csv(development_root / "method_summary.csv", method_summary)
    method_coverage = _method_coverage_summary(
        selected_rows, partition="development"
    )
    _write_csv(development_root / "method_coverage.csv", method_coverage)
    method_status: dict[str, dict[str, object]] = {}
    for method_id in sorted({row["method_id"] for row in selected_rows}):
        method_rows = [row for row in selected_rows if row["method_id"] == method_id]
        status_counts: dict[str, int] = {}
        for row in method_rows:
            status = row["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
        fallback_count = sum(
            str(row.get("fallback_used", "")).lower() == "true"
            for row in method_rows
        )
        method_status[method_id] = {
            "record_count": len(method_rows),
            "status_counts": status_counts,
            "fallback_count": fallback_count,
            "fallback_rate": fallback_count / max(len(method_rows), 1),
        }
    summary = {
        "status": "completed",
        "development_participants": len(plan.development_rows),
        "selected_gamma": selected_gamma,
        "selected_gamma_interpretation": (
            "development_selected_population_endpoint"
            if selected_gamma == 0.0
            else "development_selected_positive_context_weight"
        ),
        "gamma_zero_is_not_heldout_personalization_failure": selected_gamma == 0.0,
        "gamma_zero_structural_note": (
            GAMMA_ZERO_STRUCTURAL_NOTE if selected_gamma == 0.0 else None
        ),
        "evaluation_continues": True,
        "gamma_selection_used_query_annotations": False,
        "native_sgeyesub_status": (
            "source_faithful_python_port_not_numerically_cross_validated_with_matlab"
        ),
        "native_eeg_chan_idxs_status": (
            "resolved_official_exact_layout_channel_type_EEG_commit_2c95b4f"
        ),
        "p0_b6_eog_regression_status": "development_completed",
        "method_status": method_status,
        "metrics": str(development_root / "metrics.csv"),
        "method_summary": str(development_root / "method_summary.csv"),
        "method_coverage": str(development_root / "method_coverage.csv"),
        "frozen_gamma": str(development_root / "frozen_gamma.json"),
    }
    (development_root / "result_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _required_evaluation_method_ids(frozen_gamma: float) -> tuple[str, ...]:
    token = _gamma_token(frozen_gamma)
    return (
        "raw",
        "pop_Qy",
        "POP_fallback",
        "matching_Qy",
        "wrong_Qy",
        "shuffled_Qy",
        f"B6_Qy__gamma_{token}",
        f"B6_soft_proximal__gamma_{token}",
        "native_sgeyesub_python_release_internal",
    )


def _validate_evaluation_record_artifacts(
    protocol_row: SgeyesubProtocolRow,
    record_summary: Mapping[str, object],
    participant_rows: Sequence[Mapping[str, str]],
    *,
    frozen_gamma: float,
) -> None:
    """Reject stale, duplicated, cross-wired, or incompletely blocked outputs."""

    expected_summary = {
        "partition": "evaluation",
        "study": protocol_row.study,
        "participant_stem": protocol_row.participant_stem,
        "recording_key": protocol_row.recording_key,
        "release_layout_id": protocol_row.release_layout_id,
    }
    for field, expected in expected_summary.items():
        if str(record_summary.get(field, "")) != str(expected):
            raise ValueError(
                f"evaluation result_summary {field} is stale or cross-wired for "
                f"{protocol_row.recording_key}"
            )
    if record_summary.get("cross_layout_pooling_used") is not False:
        raise ValueError("evaluation record did not preserve exact-layout isolation")
    if float(record_summary.get("frozen_development_gamma", float("nan"))) != (
        frozen_gamma
    ):
        raise ValueError("evaluation record used a different frozen gamma")

    expected_methods = _required_evaluation_method_ids(frozen_gamma)
    method_ids = [str(row.get("method_id", "")) for row in participant_rows]
    if len(method_ids) != len(set(method_ids)):
        raise ValueError(
            f"evaluation metrics contain duplicate methods for {protocol_row.recording_key}"
        )
    if set(method_ids) != set(expected_methods) or len(method_ids) != len(
        expected_methods
    ):
        raise ValueError(
            f"evaluation metrics method set is incomplete for {protocol_row.recording_key}"
        )

    expected_metric_fields = {
        "partition": "evaluation",
        "study": protocol_row.study,
        "participant_stem": protocol_row.participant_stem,
        "recording_key": protocol_row.recording_key,
        "release_layout_id": protocol_row.release_layout_id,
        "support_block": str(protocol_row.support_block),
        "query_block": str(protocol_row.query_block),
        "population_source_count": str(protocol_row.population_source_count),
    }
    for metric_row in participant_rows:
        for field, expected in expected_metric_fields.items():
            if str(metric_row.get(field, "")) != str(expected):
                raise ValueError(
                    f"evaluation metric {field} is stale or cross-wired for "
                    f"{protocol_row.recording_key}"
                )

    if protocol_row.status == "blocked_no_population":
        if record_summary.get("status") != "completed_with_blocked_no_population":
            raise ValueError("singleton evaluation result is not complete")
        if int(record_summary.get("population_source_count", -1)) != 0:
            raise ValueError("singleton evaluation result gained a population source")
        if record_summary.get("population_status") != "blocked_no_population":
            raise ValueError("singleton evaluation result did not remain blocked")
        population_dependent = {
            "pop_Qy",
            "POP_fallback",
            "wrong_Qy",
            f"B6_Qy__gamma_{_gamma_token(frozen_gamma)}",
            f"B6_soft_proximal__gamma_{_gamma_token(frozen_gamma)}",
        }
        by_method = {row["method_id"]: row for row in participant_rows}
        if any(
            not str(by_method[method_id].get("status", "")).startswith("blocked_")
            for method_id in population_dependent
        ):
            raise ValueError("singleton population-dependent method was not blocked")
    else:
        if int(record_summary.get("population_source_count", -1)) != (
            protocol_row.population_source_count
        ):
            raise ValueError("evaluation population source count is stale")
        if record_summary.get("population_status") not in {
            "available",
            "failed_population_operator",
        }:
            raise ValueError("evaluation population status is not explicit")
        expected_status = (
            "completed"
            if record_summary.get("population_status") == "available"
            else "completed_with_failed_population_operator"
        )
        if record_summary.get("status") != expected_status:
            raise ValueError("evaluation result completion status is stale")
        by_method = {row["method_id"]: row for row in participant_rows}
        if record_summary.get("population_status") == "available":
            if any(
                not str(by_method[method_id].get("status", "")).startswith(
                    "success"
                )
                for method_id in {"pop_Qy", "POP_fallback"}
            ):
                raise ValueError("available population operator rows are not complete")
        else:
            population_dependent = {
                "pop_Qy",
                "POP_fallback",
                f"B6_Qy__gamma_{_gamma_token(frozen_gamma)}",
                f"B6_soft_proximal__gamma_{_gamma_token(frozen_gamma)}",
            }
            if any(
                not str(by_method[method_id].get("status", "")).startswith(
                    "failed_population_operator"
                )
                for method_id in population_dependent
            ):
                raise ValueError("failed population operator was not retained in metrics")


def _operator_specificity_decision(
    rows: Sequence[Mapping[str, str]],
    *,
    frozen_gamma: float,
    config: Mapping[str, object],
) -> dict[str, object]:
    thresholds = _mapping(config, "operator_specificity_decision")
    participants = sorted({row["recording_key"] for row in rows})
    total = len(participants)
    if total < 1:
        raise ValueError("operator-specificity decision has no participants")
    if frozen_gamma == 0.0:
        return {
            "decision": "development_selected_population_endpoint",
            "reason": "support_only_development_objective_selected_gamma_zero",
            "participant_denominator": total,
            "failures_and_fallbacks_retained_in_denominator": True,
            "final_heldout_decision_only": True,
            "adaptation_reselection_or_method_change": False,
            "evaluation_continues": True,
            "structural_zero_stability": True,
            "structural_zero_explanation": GAMMA_ZERO_STRUCTURAL_NOTE,
            "next_route": "continue_frozen_population_endpoint_evaluation",
        }
    b6_id = f"B6_Qy__gamma_{_gamma_token(frozen_gamma)}"
    by_key = {(row["recording_key"], row["method_id"]): row for row in rows}

    def finite(row: Mapping[str, str] | None, metric: str) -> float | None:
        if row is None or not _is_success_performance_row(row):
            return None
        try:
            value = float(row.get(metric, "nan"))
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    comparisons: dict[str, dict[str, float | int]] = {}
    required_controls = tuple(thresholds["required_improvement_controls"])
    for control in required_controls:
        paired = 0
        improved = 0
        for participant in participants:
            b6_value = finite(
                by_key.get((participant, b6_id)),
                "heldout_eog_prediction_remaining_ratio",
            )
            control_value = finite(
                by_key.get((participant, control)),
                "heldout_eog_prediction_remaining_ratio",
            )
            if b6_value is None or control_value is None:
                continue
            paired += 1
            improved += int(b6_value < control_value)
        comparisons[control] = {
            "paired_count": paired,
            "paired_fraction": paired / total,
            "improvement_fraction_all_participants": improved / total,
        }

    safety_evaluable = 0
    safety_passes = 0
    for participant in participants:
        row = by_key.get((participant, b6_id))
        preservation = finite(row, "nonartifact_observation_preservation")
        covariance = finite(row, "reference_free_covariance_distortion")
        if (
            preservation is not None
            and covariance is not None
        ):
            safety_evaluable += 1
            if (
                preservation
                >= float(thresholds["minimum_nonartifact_observation_preservation"])
                and covariance
                <= float(thresholds["maximum_reference_free_covariance_distortion"])
            ):
                safety_passes += 1
    minimum_paired = float(thresholds["minimum_paired_participant_fraction"])
    minimum_improvement = float(thresholds["minimum_improvement_fraction"])
    paired_evidence_available = all(
        comparisons[control]["paired_fraction"] >= minimum_paired
        for control in required_controls
    )
    safety_evidence_available = safety_evaluable / total >= minimum_paired
    supported = paired_evidence_available and all(
        comparisons[control]["paired_fraction"] >= minimum_paired
        and comparisons[control]["improvement_fraction_all_participants"]
        >= minimum_improvement
        for control in required_controls
    )
    supported = (
        supported
        and safety_evidence_available
        and safety_passes / total
        >= float(thresholds["minimum_safety_pass_fraction"])
    )
    insufficient = not paired_evidence_available or not safety_evidence_available
    if insufficient:
        decision = "inconclusive_insufficient_finite_pairs"
        reason = "insufficient_finite_successful_pairs_or_safety_rows"
        next_route = "stop_inconclusive_use_population_deterministic"
    elif supported:
        decision = "b6_participant_specificity_supported"
        reason = "frozen_improvement_and_safety_thresholds_passed"
        next_route = "eye_bci_operator_specificity_eligible_but_not_submitted"
    else:
        decision = "frozen_b6_specificity_not_supported_under_tested_protocol"
        reason = "frozen_b6_improvement_or_safety_threshold_not_met"
        next_route = "stop_frozen_b6_route_retain_population_endpoint"
    return {
        "decision": decision,
        "reason": reason,
        "participant_denominator": total,
        "comparisons": comparisons,
        "safety_evaluable_count": safety_evaluable,
        "safety_evaluable_fraction": safety_evaluable / total,
        "safety_pass_count": safety_passes,
        "safety_pass_fraction": safety_passes / total,
        "failures_and_fallbacks_retained_in_denominator": True,
        "final_heldout_decision_only": True,
        "adaptation_reselection_or_method_change": False,
        "next_route": next_route,
    }


def run_sgeyesub_evaluation_aggregate(
    config: Mapping[str, object], *, run_dir: Path
) -> dict[str, object]:
    """Aggregate all 44 frozen-gamma evaluation stems without reselection."""

    validate_sgeyesub_protocol_config(config)
    _write_resolved_config(config)
    frozen_gamma = _load_frozen_development_gamma(config)
    metadata = _mapping(config, "metadata")
    layouts, records = load_sgeyesub_structure_audit(
        Path(str(metadata["structure_audit_result"]))
    )
    plan = build_sgeyesub_protocol(
        layouts,
        records,
        protocol_id=str(config["protocol_id"]),
        reference_cell_id=str(
            _mapping(config, "compatibility_cell")["reference_cell_id"]
        ),
        gamma_candidates=tuple(_mapping(config, "b6_pop_shrink")["gamma_candidates"]),
    )
    evaluation_root = Path(str(config["evaluation_output_root"]))
    all_metric_rows: list[dict[str, str]] = []
    gamma_token = _gamma_token(frozen_gamma)
    for protocol_row in plan.evaluation_rows:
        participant_root = evaluation_root / protocol_row.participant_stem
        record_summary = json.loads(
            (participant_root / "result_summary.json").read_text(encoding="utf-8")
        )
        with (participant_root / "metrics.csv").open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            participant_rows = list(csv.DictReader(stream))
        _validate_evaluation_record_artifacts(
            protocol_row,
            record_summary,
            participant_rows,
            frozen_gamma=frozen_gamma,
        )
        if any(
            row["method_id"].startswith("B6_")
            and not row["method_id"].endswith(f"__gamma_{gamma_token}")
            for row in participant_rows
        ):
            raise ValueError("evaluation record contains an unfrozen B6 gamma")
        all_metric_rows.extend(participant_rows)

    expected_recording_keys = {
        row.recording_key for row in plan.evaluation_rows
    }
    observed_recording_keys = {
        row["recording_key"] for row in all_metric_rows
    }
    if len(expected_recording_keys) != 44 or observed_recording_keys != (
        expected_recording_keys
    ):
        raise ValueError("evaluation aggregation does not contain the exact 44 records")

    _write_csv(evaluation_root / "metrics.csv", all_metric_rows)
    method_summary = _method_summary(all_metric_rows, partition="evaluation")
    _write_csv(evaluation_root / "method_summary.csv", method_summary)
    method_coverage = _method_coverage_summary(
        all_metric_rows, partition="evaluation"
    )
    _write_csv(evaluation_root / "method_coverage.csv", method_coverage)
    method_status: dict[str, dict[str, object]] = {}
    for method_id in sorted({row["method_id"] for row in all_metric_rows}):
        method_rows = [row for row in all_metric_rows if row["method_id"] == method_id]
        status_counts: dict[str, int] = {}
        for row in method_rows:
            status = row["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
        fallback_count = sum(
            str(row.get("fallback_used", "")).lower() == "true"
            for row in method_rows
        )
        method_status[method_id] = {
            "record_count": len(method_rows),
            "status_counts": status_counts,
            "fallback_count": fallback_count,
            "fallback_rate": fallback_count / max(len(method_rows), 1),
        }
    blocked = [
        row for row in plan.evaluation_rows if row.status == "blocked_no_population"
    ]
    if len(blocked) != 1:
        raise AssertionError("expected exactly one singleton evaluation cell")
    scientific_decision = _operator_specificity_decision(
        all_metric_rows,
        frozen_gamma=frozen_gamma,
        config=config,
    )
    corrected_audit = _write_corrected_audit(
        config,
        protocol_rows=plan.evaluation_rows,
        rows=all_metric_rows,
        frozen_gamma=frozen_gamma,
        development_rows=plan.development_rows,
    )
    summary = {
        "status": "completed",
        "claim_scope": "release_internal_block1_to_block2_not_native_replication",
        "evaluation_participant_stems": len(plan.evaluation_rows),
        "metadata_ready_population_records": sum(
            row.status == "metadata_ready" for row in plan.evaluation_rows
        ),
        "blocked_no_population_records": len(blocked),
        "blocked_no_population_recording_keys": [row.recording_key for row in blocked],
        "cross_layout_population_pooling_used": False,
        "frozen_development_gamma": frozen_gamma,
        "gamma_reselected_on_evaluation": False,
        "query_annotations_used_for_fit_gamma_or_method_selection": False,
        "query_annotations_used_for_single_final_automatic_decision": True,
        "final_decision_adaptation_reselection_or_method_change": False,
        "query_clean_target": "not_available",
        "native_reference_equivalence_status": REFERENCE_EQUIVALENCE_STATUS,
        "method_status": method_status,
        "performance_summary_policy": "success_nonfallback_finite_only",
        "coverage_denominator_policy": "all_44_registered_recording_keys",
        "operator_specificity_decision": scientific_decision,
        "corrected_operator_audit": corrected_audit,
        "decision": scientific_decision["decision"],
        "next_route": scientific_decision["next_route"],
        "metrics": str(evaluation_root / "metrics.csv"),
        "method_summary": str(evaluation_root / "method_summary.csv"),
        "method_coverage": str(evaluation_root / "method_coverage.csv"),
    }
    (evaluation_root / "result_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def run_sgeyesub_corrected_audit(
    config: Mapping[str, object], *, run_dir: Path
) -> dict[str, object]:
    """Read existing record outputs and write only the additive corrected audit."""

    validate_sgeyesub_protocol_config(config)
    frozen_gamma = _load_frozen_development_gamma(config)
    metadata = _mapping(config, "metadata")
    layouts, records = load_sgeyesub_structure_audit(
        Path(str(metadata["structure_audit_result"]))
    )
    plan = build_sgeyesub_protocol(
        layouts,
        records,
        protocol_id=str(config["protocol_id"]),
        reference_cell_id=str(
            _mapping(config, "compatibility_cell")["reference_cell_id"]
        ),
        gamma_candidates=tuple(_mapping(config, "b6_pop_shrink")["gamma_candidates"]),
    )
    evaluation_root = Path(str(config["evaluation_output_root"]))
    all_metric_rows: list[dict[str, str]] = []
    gamma_token = _gamma_token(frozen_gamma)
    for protocol_row in plan.evaluation_rows:
        participant_root = evaluation_root / protocol_row.participant_stem
        record_summary = json.loads(
            (participant_root / "result_summary.json").read_text(encoding="utf-8")
        )
        with (participant_root / "metrics.csv").open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            participant_rows = list(csv.DictReader(stream))
        _validate_evaluation_record_artifacts(
            protocol_row,
            record_summary,
            participant_rows,
            frozen_gamma=frozen_gamma,
        )
        if any(
            row["method_id"].startswith("B6_")
            and not row["method_id"].endswith(f"__gamma_{gamma_token}")
            for row in participant_rows
        ):
            raise ValueError("evaluation record contains an unfrozen B6 gamma")
        all_metric_rows.extend(participant_rows)

    expected_recording_keys = {row.recording_key for row in plan.evaluation_rows}
    observed_recording_keys = {row["recording_key"] for row in all_metric_rows}
    if len(expected_recording_keys) != 44 or observed_recording_keys != (
        expected_recording_keys
    ):
        raise ValueError("corrected audit does not contain the exact 44 records")

    corrected = _write_corrected_audit(
        config,
        protocol_rows=plan.evaluation_rows,
        rows=all_metric_rows,
        frozen_gamma=frozen_gamma,
        development_rows=plan.development_rows,
    )
    summary = {
        "status": corrected["status"],
        "stage": "corrected-audit",
        "source_policy": "read_existing_per_record_outputs_only",
        "historical_results_policy": "read_only_side_by_side_no_overwrite",
        "frozen_development_gamma": frozen_gamma,
        "corrected_operator_audit": corrected,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
