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


def _mapping(config: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"SGEYESUB config section {key!r} must be a mapping")
    return value


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
            {
                "gamma": float(gamma),
                "status": "unavailable_split_half_context",
                "split_half_stability": None,
                "heldout_contamination_capture_loss": None,
                "capture_weight": capture_weight,
                "support_score": None,
            }
            for gamma in b6["gamma_candidates"]
        ]

    rows: list[dict[str, object]] = []
    for value in b6["gamma_candidates"]:
        gamma = float(value)
        if gamma > 0.0 and context_outcome.transfer is None:
            rows.append(
                {
                    "gamma": gamma,
                    "status": "unavailable_full_context",
                    "split_half_stability": None,
                    "heldout_contamination_capture_loss": None,
                    "capture_weight": capture_weight,
                    "support_score": None,
                }
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
                {
                    "gamma": gamma,
                    "status": "unavailable_shrinkage",
                    "split_half_stability": None,
                    "heldout_contamination_capture_loss": None,
                    "capture_weight": capture_weight,
                    "support_score": None,
                }
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
            {
                "gamma": gamma,
                "status": "success",
                "split_half_stability": stability,
                "heldout_contamination_capture_loss": capture_loss,
                "capture_weight": capture_weight,
                "support_score": score,
            }
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
    population_projector: np.ndarray,
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
    artifact_mask = np.asarray(artifactclasses != 6, dtype=bool)
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
    rows: Sequence[Mapping[str, str]],
    *,
    partition: str,
) -> list[dict[str, object]]:
    """Participant-stem means and percentile intervals within one partition."""

    metric_names = (
        "matching_projector_attenuation_db",
        "population_projector_attenuation_db",
        "nonartifact_observation_preservation",
        "eog_coherence_reduction",
        "reference_free_psd_distortion",
        "reference_free_covariance_distortion",
        "heldout_eog_prediction_remaining_ratio",
        "condition_erp_observation_relative_preservation",
        "observation_change_ratio",
    )
    method_ids = sorted({row["method_id"] for row in rows})
    summaries: list[dict[str, object]] = []
    for method_index, method_id in enumerate(method_ids):
        method_rows = [row for row in rows if row["method_id"] == method_id]
        for metric_index, metric_name in enumerate(metric_names):
            values: list[float] = []
            for row in method_rows:
                try:
                    value = float(row.get(metric_name, "nan"))
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value):
                    values.append(value)
            if not values:
                continue
            numeric = np.asarray(values, dtype=np.float64)
            rng = np.random.default_rng(20260802 + 100 * method_index + metric_index)
            indices = rng.integers(0, numeric.size, size=(2000, numeric.size))
            bootstrap_means = numeric[indices].mean(axis=1)
            summaries.append(
                {
                    "partition": partition,
                    "method_id": method_id,
                    "metric": metric_name,
                    "participant_stem_count": int(numeric.size),
                    "mean": float(np.mean(numeric)),
                    "ci95_low": float(np.quantile(bootstrap_means, 0.025)),
                    "ci95_high": float(np.quantile(bootstrap_means, 0.975)),
                    "inference_unit": "release_scoped_study_participant_stem",
                }
            )
    return summaries


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
        channel_types=native_types,
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
            channel_types=native_types,
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
    if strict_context.transfer is None:
        outputs.append(
            (
                "external_query_eog_regression",
                observed.copy(),
                "ineligible_matching_P0_identity_no_claim",
                ";".join(strict_context.reasons),
                None,
                False,
                True,
            )
        )
    else:
        heldout_predicted_contamination = (
            strict_context.transfer.transfer_matrix
            @ (query_eog - strict_context.transfer.eog_mean)
        )
        outputs.append(
            (
                "external_query_eog_regression",
                observed - heldout_predicted_contamination,
                "success",
                "matching_block1_transfer_external_query_EOG",
                None,
                False,
                True,
            )
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
    if population.transfer is None:
        raise RuntimeError("same-cell population operator is unavailable")
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
    support_scores = (
        _gamma_support_scores(
            population_projector=population.transfer.projector,
            context_outcome=relaxed_context,
            support_eeg=np.asarray(loaded.support.eeg, dtype=np.float64),
            support_eog=support_eog,
            compatibility=compatibility,
            config=config,
            sampling_rate=target.sampling_rate_hz,
        )
        if partition == "development"
        else None
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
    pi0 = population.transfer.projector
    outputs: list[tuple[str, np.ndarray, str, str, float | None, bool, bool]] = []
    outputs.append(("raw", observed.copy(), "success", "raw", None, False, False))
    pop_output = _q_restore(pi0, observed)
    outputs.append(("pop_Qy", pop_output, "success", "Pi0", None, False, False))
    outputs.append(("POP_fallback", pop_output.copy(), "success", "Pi0", None, True, False))

    if strict_context.transfer is None:
        outputs.append(
            (
                "matching_Qy",
                pop_output.copy(),
                "fallback_POP",
                "matching_P0_ineligible",
                None,
                True,
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
        outputs.append(
            (
                "wrong_Qy",
                pop_output.copy(),
                "fallback_POP",
                "wrong_P0_ineligible",
                None,
                True,
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
        outputs.append(
            (
                "shuffled_Qy",
                pop_output.copy(),
                "fallback_POP",
                "shuffled_P0_ineligible",
                None,
                True,
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
        token = _gamma_token(gamma)
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

    # Query EOG is first accessed only after every support-only fit, gamma
    # score, and non-external output above is frozen.  This explicitly external
    # baseline is never a gamma-selection candidate.
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
    if strict_context.transfer is None:
        outputs.append(
            (
                "external_query_eog_regression",
                pop_output.copy(),
                "fallback_POP",
                "matching_P0_ineligible",
                None,
                True,
                True,
            )
        )
    else:
        centered_query_eog = query_eog - strict_context.transfer.eog_mean
        heldout_predicted_contamination = (
            strict_context.transfer.transfer_matrix @ centered_query_eog
        )
        outputs.append(
            (
                "external_query_eog_regression",
                observed - heldout_predicted_contamination,
                "success",
                "matching_block1_transfer_external_query_EOG",
                None,
                False,
                True,
            )
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
        "status": "completed",
        "partition": partition,
        "study": target.study,
        "participant_stem": target.participant_stem,
        "recording_key": target.recording_key,
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
        "same_cell_population_verified": True,
        "population_p0_status": population.status,
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
        "query_annotations_access_phase": (
            "after_support_fit_gamma_and_non_external_outputs"
        ),
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
    for protocol_row in plan.development_rows:
        participant_root = development_root / protocol_row.participant_stem
        score_rows = json.loads(
            (participant_root / "support_gamma_scores.json").read_text(encoding="utf-8")
        )
        for row in score_rows:
            if row["status"] == "success" and row["support_score"] is not None:
                by_gamma[float(row["gamma"])].append(float(row["support_score"]))

    minimum_fraction = float(
        _mapping(config, "b6_pop_shrink")["minimum_gamma_score_participant_fraction"]
    )
    selected_gamma, aggregate_scores = select_global_gamma(
        by_gamma,
        candidates=candidates,
        participant_count=len(plan.development_rows),
        minimum_fraction=minimum_fraction,
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
        "best_gamma_zero_supported": selected_gamma == 0.0,
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
            "decision": "personalization_failed_population_deterministic",
            "reason": "development_selected_gamma_zero",
            "participant_denominator": total,
            "next_route": "stop_personalization_use_population_deterministic",
        }
    b6_id = f"B6_Qy__gamma_{_gamma_token(frozen_gamma)}"
    by_key = {(row["recording_key"], row["method_id"]): row for row in rows}

    def finite(row: Mapping[str, str] | None, metric: str) -> float | None:
        if row is None or not row.get("status", "").startswith("success"):
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

    safety_passes = 0
    for participant in participants:
        row = by_key.get((participant, b6_id))
        preservation = finite(row, "nonartifact_observation_preservation")
        covariance = finite(row, "reference_free_covariance_distortion")
        if (
            preservation is not None
            and covariance is not None
            and preservation
            >= float(thresholds["minimum_nonartifact_observation_preservation"])
            and covariance
            <= float(thresholds["maximum_reference_free_covariance_distortion"])
        ):
            safety_passes += 1
    minimum_paired = float(thresholds["minimum_paired_participant_fraction"])
    minimum_improvement = float(thresholds["minimum_improvement_fraction"])
    supported = all(
        comparisons[control]["paired_fraction"] >= minimum_paired
        and comparisons[control]["improvement_fraction_all_participants"]
        >= minimum_improvement
        for control in required_controls
    )
    supported = supported and safety_passes / total >= float(
        thresholds["minimum_safety_pass_fraction"]
    )
    return {
        "decision": (
            "b6_participant_specificity_supported"
            if supported
            else "personalization_failed_population_deterministic"
        ),
        "participant_denominator": total,
        "comparisons": comparisons,
        "safety_pass_count": safety_passes,
        "safety_pass_fraction": safety_passes / total,
        "failures_and_fallbacks_retained_in_denominator": True,
        "next_route": (
            "eye_bci_operator_specificity_eligible_but_not_submitted"
            if supported
            else "stop_personalization_use_population_deterministic"
        ),
    }


def run_sgeyesub_evaluation_aggregate(
    config: Mapping[str, object], *, run_dir: Path
) -> dict[str, object]:
    """Aggregate all 44 frozen-gamma evaluation stems without reselection."""

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
        if float(record_summary["frozen_development_gamma"]) != frozen_gamma:
            raise ValueError("evaluation record used a different frozen gamma")
        with (participant_root / "metrics.csv").open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            participant_rows = list(csv.DictReader(stream))
        if any(
            row["method_id"].startswith("B6_")
            and not row["method_id"].endswith(f"__gamma_{gamma_token}")
            for row in participant_rows
        ):
            raise ValueError("evaluation record contains an unfrozen B6 gamma")
        all_metric_rows.extend(participant_rows)

    _write_csv(evaluation_root / "metrics.csv", all_metric_rows)
    method_summary = _method_summary(all_metric_rows, partition="evaluation")
    _write_csv(evaluation_root / "method_summary.csv", method_summary)
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
        "query_clean_target": "not_available",
        "native_reference_equivalence_status": REFERENCE_EQUIVALENCE_STATUS,
        "method_status": method_status,
        "operator_specificity_decision": scientific_decision,
        "decision": scientific_decision["decision"],
        "next_route": scientific_decision["next_route"],
        "metrics": str(evaluation_root / "metrics.csv"),
        "method_summary": str(evaluation_root / "method_summary.csv"),
    }
    (evaluation_root / "result_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
