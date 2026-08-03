"""Participant-stem aggregation for the subject-artifact development round.

This module is deliberately narrow.  It consumes the frozen J2 gate and, only
when that gate passed, the stable J3 summaries and J4 stem-level metric files.
It never opens EEG, EOG, annotations, waveform archives, or checkpoints.

The natural-SGE evidence remains development/exploratory.  Missing paired
mechanism evidence, an operational uncertainty target, an equal-update
evaluation endpoint, or a calibration-duration axis is reported as missing;
none of those omissions can be promoted to a successful scientific gate.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


MODEL_IDS = ("deterministic", "diffusion")
CONTEXT_IDS = ("population", "matching", "wrong", "shuffled")
TRAINING_SEED_COUNT = 3
FOLD_COUNT = 25
J3_SUMMARY_COUNT = FOLD_COUNT * TRAINING_SEED_COUNT * len(MODEL_IDS)
J4_SUMMARY_COUNT = FOLD_COUNT * TRAINING_SEED_COUNT
EXPECTED_COMPATIBLE_STEMS = 58
EXPECTED_PREBLOCKED_STEM = "study05/study05_p42"
EXPECTED_AVAILABILITY_STEMS = 59

# Every contrast below is expressed as a utility: positive means improvement.
METRIC_DIRECTIONS: dict[str, str] = {
    "matching_projector_attenuation_db": "higher",
    "population_projector_attenuation_db": "higher",
    "nonartifact_observation_preservation": "higher",
    "eog_coherence_reduction": "higher",
    "condition_erp_observation_relative_preservation": "higher",
    "heldout_eog_prediction_remaining_ratio": "lower",
    "reference_free_psd_distortion": "lower",
    "reference_free_covariance_distortion": "lower",
    "observation_change_ratio": "lower",
}


@dataclass(frozen=True)
class FoldSpec:
    unified_fold_index: int
    fold_id: str
    study: str
    layout_id: str
    sampling_rate_hz: float
    reference_cell: str
    heldout_recording_keys: tuple[str, ...]

    @property
    def cell_id(self) -> str:
        layout = self.layout_id.replace("_", "").lower()
        return (
            f"{self.study}|{layout}|{self.reference_cell}|"
            f"{self.sampling_rate_hz:g}Hz"
        )


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"missing mapping: {key}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if str(key) not in fields:
                fields.append(str(key))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for raw in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(raw.get(key), sort_keys=True)
                        if isinstance(raw.get(key), (dict, list, tuple))
                        else raw.get(key, "")
                    )
                    for key in fields
                }
            )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _truth(value: object) -> bool:
    return value is True or str(value).strip().lower() == "true"


def _finite(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _utility(metric: str, value: object) -> float | None:
    numeric = _finite(value)
    if numeric is None:
        return None
    direction = METRIC_DIRECTIONS[metric]
    return numeric if direction == "higher" else -numeric


def _canonical_endpoint(row: Mapping[str, Any]) -> str:
    raw = str(
        row.get("checkpoint_endpoint")
        or row.get("checkpoint_role")
        or "best_validation"
    ).strip()
    aliases = {
        "development_validation_best": "best_validation",
        "best": "best_validation",
        "equal": "equal_update",
        "equal_update_8000": "equal_update",
    }
    return aliases.get(raw, raw)


def _performance_eligible(row: Mapping[str, Any], metric: str | None = None) -> bool:
    if str(row.get("status", "")) != "success":
        return False
    if not _truth(row.get("performance_values_eligible", False)):
        return False
    return metric is None or _finite(row.get(metric)) is not None


def _output_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    outputs = _mapping(config, "outputs")
    return {
        name: Path(str(outputs[name]))
        for name in (
            "validity_root",
            "development_root",
            "checkpoint_root",
            "metrics",
            "summary",
            "figures",
        )
    }


def _validity_gate(config: Mapping[str, Any]) -> dict[str, Any]:
    paths = _output_paths(config)
    gate = _read_json(paths["validity_root"] / "result_summary.json")
    revision = str(_mapping(config, "validity").get("execution_revision", ""))
    if (
        gate.get("protocol_id") != config.get("protocol_id")
        or gate.get("execution_revision") != revision
        or gate.get("status")
        not in {"passed_V0_to_V3", "completed_model_validity_failed"}
    ):
        raise ValueError("J2 validity gate does not match this execution revision")
    _selected_validity_result(config, gate)
    return gate


def _selected_validity_result(
    config: Mapping[str, Any], gate: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Validate the embedded wrapper result; never probe a legacy attempt path."""

    selected = gate.get("selected_result")
    if not isinstance(selected, Mapping):
        raise ValueError("J2 wrapper gate lacks selected_result")
    revision = str(gate.get("execution_revision", ""))
    implementation = str(gate.get("selected_implementation", ""))
    passed = gate.get("passed") is True
    expected_gate_status = (
        "passed_V0_to_V3" if passed else "completed_model_validity_failed"
    )
    expected_result_status = "passed" if passed else "failed"
    expected_model_validity = "passed" if passed else "failed"
    validity = selected.get("validity")
    if (
        gate.get("status") != expected_gate_status
        or gate.get("model_validity") != expected_model_validity
        or selected.get("execution_revision") != revision
        or selected.get("implementation") != implementation
        or selected.get("status") != expected_result_status
        or (selected.get("passed") is True) is not passed
        or selected.get("model_validity") != expected_model_validity
        or not isinstance(validity, Mapping)
        or set(validity) != {"V0", "V1", "V2", "V3"}
    ):
        raise ValueError("J2 wrapper gate and selected_result disagree")
    expected_detail = (
        _output_paths(config)["validity_root"]
        / revision
        / implementation
        / "result_summary.json"
    )
    if Path(str(selected.get("attempt_result_path", ""))) != expected_detail:
        raise ValueError("J2 selected result does not use the revision-nested path")
    return selected


def _aggregate_root(config: Mapping[str, Any]) -> Path:
    return _output_paths(config)["development_root"] / "aggregate"


def _implementation_fields(implementation: Mapping[str, Any] | None) -> dict[str, Any]:
    return {} if implementation is None else dict(implementation)


def _fail_closed_after_validity(
    config: Mapping[str, Any],
    run_dir: Path,
    gate: Mapping[str, Any],
    implementation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Write a terminal J5 result without touching any J3/J4 path."""

    paths = _output_paths(config)
    aggregate_root = _aggregate_root(config)
    selected = str(gate.get("selected_implementation", "unknown"))
    revision = str(gate.get("execution_revision", ""))
    selected_result = _selected_validity_result(config, gate)
    attempt_summary = (
        paths["validity_root"]
        / revision
        / selected
        / "result_summary.json"
    )
    levels = selected_result["validity"]
    level_status = {
        name: (
            value.get("status", "unknown") if isinstance(value, Mapping) else "unknown"
        )
        for name, value in (
            levels.items() if isinstance(levels, Mapping) else ()
        )
    }
    status_row = {
        "row_type": "validity_gate_terminal",
        "status": "model_validity_failed",
        "execution_revision": gate.get("execution_revision"),
        "selected_implementation": selected,
        "scientific_comparison_eligibility": "blocked",
        "confirmation_eligibility": False,
    }
    _write_csv(paths["metrics"], [status_row])
    _write_csv(aggregate_root / "terminal_status.csv", [status_row])
    summary: dict[str, Any] = {
        "status": "completed_fail_closed_model_validity_failed",
        "stage": "J5_participant_stem_aggregate",
        "protocol_id": config.get("protocol_id"),
        "execution_revision": gate.get("execution_revision"),
        "scientific_role": "development_exploratory_not_confirmation",
        "computational_completion": "passed_fail_closed_terminal",
        "model_validity": "failed",
        "validity_levels": level_status,
        "selected_validity_implementation": selected,
        "selected_validity_summary": str(attempt_summary),
        "scientific_comparison_eligibility": "blocked",
        "full_real_factorial_executed": False,
        "J3_training": "not_run_blocked_by_V0_V3",
        "J4_factorial": "not_run_blocked_by_V0_V3",
        "G_calibration": {"status": "not_run_blocked_by_V0_V3"},
        "G_diffusion": {"status": "not_run_blocked_by_V0_V3"},
        "uncertainty": {"status": "not_tested"},
        "protocol_decision": "inconclusive",
        "topic_status": "not_yet_testable",
        "confirmation_eligibility": False,
        "family_wide_status": "not_tested",
        "query_confirmation_outcomes_opened": False,
        "coverage": {
            "availability_stem_denominator": EXPECTED_AVAILABILITY_STEMS,
            "performance_rows_read": 0,
            "reason": "hard_validity_gate_failed_before_J3_J4",
        },
        "figures": {
            "trajectory_rms": str(
                paths["validity_root"]
                / revision
                / selected
                / "diffusion_trajectory_rms.png"
            ),
            "paired_delta": "not_generated_hard_gate_failed",
            "attenuation_preservation_pareto": "not_generated_hard_gate_failed",
            "calibration_duration": "not_generated_hard_gate_failed",
        },
        "metrics": str(paths["metrics"]),
        **_implementation_fields(implementation),
    }
    _atomic_json(aggregate_root / "gate_status.json", summary)
    _atomic_json(aggregate_root / "result_summary.json", summary)
    run_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(run_dir / "result_summary.json", summary)
    return summary


def _load_fold_specs(config: Mapping[str, Any]) -> tuple[FoldSpec, ...]:
    sge = _mapping(_mapping(config, "data"), "sgeyesub")
    frozen_path = Path(str(sge["frozen_fold_source"]))
    frozen = yaml.safe_load(frozen_path.read_text(encoding="utf-8"))
    if not isinstance(frozen, Mapping):
        raise ValueError("frozen SGEYESUB fold source is not a mapping")
    split = _mapping(frozen, "split")
    compatibility = _mapping(frozen, "compatibility")
    reference = str(compatibility["reference_cell"])
    raw_folds = tuple(split.get("development_folds", ())) + tuple(
        split.get("evaluation_folds", ())
    )
    if len(raw_folds) != FOLD_COUNT:
        raise ValueError("J5 requires exactly 25 frozen SGEYESUB folds")
    result: list[FoldSpec] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_folds):
        if not isinstance(raw, Mapping):
            raise ValueError("frozen fold entry is not a mapping")
        study = str(raw["study"])
        heldout = tuple(
            f"{study}/{value}" for value in tuple(raw.get("heldout_stems", ()))
        )
        if not heldout or seen.intersection(heldout):
            raise ValueError("frozen held-out stems are empty or duplicated")
        seen.update(heldout)
        result.append(
            FoldSpec(
                unified_fold_index=index,
                fold_id=str(raw["fold_id"]),
                study=study,
                layout_id=str(raw["layout_id"]),
                sampling_rate_hz=float(raw["sampling_rate_hz"]),
                reference_cell=reference,
                heldout_recording_keys=heldout,
            )
        )
    if len(seen) != EXPECTED_COMPATIBLE_STEMS:
        raise ValueError("frozen compatible-stem count is not 58")
    preblocked = tuple(split.get("evaluation_preblocked", ()))
    blocked_keys = {
        str(value.get("recording_key"))
        for value in preblocked
        if isinstance(value, Mapping)
    }
    if blocked_keys != {EXPECTED_PREBLOCKED_STEM}:
        raise ValueError("the frozen 59-stem availability denominator changed")
    return tuple(result)


def _load_j3_summaries(
    config: Mapping[str, Any], folds: Sequence[FoldSpec]
) -> list[dict[str, Any]]:
    paths = _output_paths(config)
    seeds = tuple(int(value) for value in _mapping(config, "training")["seeds"])
    if len(seeds) != TRAINING_SEED_COUNT or len(set(seeds)) != len(seeds):
        raise ValueError("J5 requires exactly three unique training seeds")
    summaries: list[dict[str, Any]] = []
    for fold in folds:
        for seed in seeds:
            for model_id in MODEL_IDS:
                path = (
                    paths["checkpoint_root"]
                    / f"fold_{fold.unified_fold_index:02d}"
                    / f"seed_{seed}"
                    / model_id
                    / "result_summary.json"
                )
                value = _read_json(path)
                task = value.get("task")
                checkpoints = value.get("checkpoints")
                if (
                    value.get("status") != "success"
                    or not isinstance(task, Mapping)
                    or int(task.get("unified_fold_index", -1))
                    != fold.unified_fold_index
                    or int(task.get("seed", -1)) != seed
                    or task.get("model_kind") != model_id
                    or not isinstance(checkpoints, Mapping)
                ):
                    raise ValueError(f"J3 stable summary is invalid: {path}")
                for endpoint in ("equal_update", "best_validation"):
                    checkpoint = Path(str(checkpoints.get(endpoint, "")))
                    if not checkpoint.is_file():
                        raise ValueError(
                            f"J3 summary is missing {endpoint} checkpoint: {path}"
                        )
                summaries.append(value)
    if len(summaries) != J3_SUMMARY_COUNT:
        raise AssertionError("J3 stable-summary count is not 150")
    return summaries


def _load_j4_rows(
    config: Mapping[str, Any], folds: Sequence[FoldSpec]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paths = _output_paths(config)
    seeds = tuple(int(value) for value in _mapping(config, "training")["seeds"])
    summaries: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    expected_recordings = {
        recording: fold
        for fold in folds
        for recording in fold.heldout_recording_keys
    }
    seen_keys: set[tuple[str, int, str, str, str]] = set()
    for fold in folds:
        for seed in seeds:
            destination = (
                paths["development_root"]
                / "natural_sge_factorial"
                / f"fold_{fold.unified_fold_index:02d}"
                / f"seed_{seed}"
            )
            summary = _read_json(destination / "result_summary.json")
            if (
                summary.get("status")
                not in {
                    "success_complete_fold_seed_development_factorial",
                    "completed_with_failed_or_ineligible_arms",
                }
                or summary.get("protocol_id") != config.get("protocol_id")
                or int(summary.get("unified_fold_index", -1))
                != fold.unified_fold_index
                or int(summary.get("training_seed", -1)) != seed
            ):
                raise ValueError(f"J4 terminal summary is invalid: {destination}")
            summaries.append(summary)
            metric_rows = _read_csv(destination / "metrics.csv")
            expected_file_rows = len(fold.heldout_recording_keys) * (
                len(MODEL_IDS) * len(CONTEXT_IDS) * 2 + 1
            )
            if len(metric_rows) != expected_file_rows:
                raise ValueError(
                    f"J4 metrics row count is invalid: {destination}"
                )
            for raw in metric_rows:
                recording = str(raw.get("recording_key", ""))
                if recording not in fold.heldout_recording_keys:
                    raise ValueError("J4 metric row escaped its frozen held-out fold")
                if (
                    int(raw.get("training_seed", -1)) != seed
                    or int(raw.get("unified_fold_index", -1))
                    != fold.unified_fold_index
                ):
                    raise ValueError("J4 metric row has the wrong fold or seed")
                if (
                    _truth(
                        raw.get(
                            "query_evaluation_fields_used_for_fit_"
                            "selection_or_inference"
                        )
                    )
                    or _truth(raw.get("query_eog_used_for_inference"))
                    or _truth(raw.get("query_labels_used_for_inference"))
                    or str(raw.get("statistical_unit", "")) != "participant_stem"
                    or str(raw.get("clean_waveform_metric", ""))
                    != "N/A_no_clean_target"
                ):
                    raise ValueError("J4 row violates the frozen information boundary")
                model = str(raw.get("model_id", ""))
                context = str(raw.get("context_id", ""))
                endpoint = _canonical_endpoint(raw)
                key = (recording, seed, endpoint, model, context)
                if key in seen_keys:
                    raise ValueError(f"duplicate J4 factorial metric row: {key}")
                seen_keys.add(key)
                row: dict[str, Any] = dict(raw)
                row["training_seed"] = seed
                row["checkpoint_endpoint"] = endpoint
                row["exact_cell_id"] = expected_recordings[recording].cell_id
                rows.append(row)
    if len(summaries) != J4_SUMMARY_COUNT:
        raise AssertionError("J4 terminal-summary count is not 75")
    frozen_endpoints = {"best_validation", "equal_update"}
    expected_learned = {
        (recording, seed, endpoint, model, context)
        for recording in expected_recordings
        for seed in seeds
        for endpoint in frozen_endpoints
        for model in MODEL_IDS
        for context in CONTEXT_IDS
    }
    observed_learned = {
        (
            str(row["recording_key"]),
            int(row["training_seed"]),
            str(row["checkpoint_endpoint"]),
            str(row["model_id"]),
            str(row["context_id"]),
        )
        for row in rows
        if row.get("model_id") in MODEL_IDS
    }
    missing = expected_learned - observed_learned
    unexpected = observed_learned - expected_learned
    if missing or unexpected:
        example = sorted(missing or unexpected)[0]
        raise ValueError(
            "J4 learned factorial is structurally incomplete or expanded: "
            f"{example}"
        )
    return summaries, rows


def _audit_uncertainty_inputs(
    config: Mapping[str, Any],
    folds: Sequence[FoldSpec],
    factorial_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Audit post-freeze proxy inputs without turning them into a success gate."""

    paths = _output_paths(config)
    seeds = tuple(int(value) for value in _mapping(config, "training")["seeds"])
    present_files = 0
    row_count = 0
    scientifically_eligible_row_count = 0
    endpoints: set[str] = set()
    recordings: set[str] = set()
    arm_index = _arm_index(factorial_rows)
    required_finite = (
        "posterior_output_SD_RMS",
        "posterior_correction_SD_RMS",
        "point_EOG_contamination_proxy_RMSE",
        "posterior_sample_EOG_proxy_RMSE_mean",
        "posterior_sample_EOG_proxy_RMSE_SD",
    )
    for fold in folds:
        for seed in seeds:
            path = (
                paths["development_root"]
                / "natural_sge_factorial"
                / f"fold_{fold.unified_fold_index:02d}"
                / f"seed_{seed}"
                / "uncertainty_windows.csv"
            )
            if not path.is_file():
                continue
            present_files += 1
            for row in _read_csv(path):
                recording = str(row.get("recording_key", ""))
                endpoint = _canonical_endpoint(row)
                context = str(row.get("context_id", ""))
                arm = arm_index.get(
                    (recording, seed, endpoint, "diffusion", context)
                )
                if (
                    recording not in fold.heldout_recording_keys
                    or int(row.get("unified_fold_index", -1))
                    != fold.unified_fold_index
                    or int(row.get("training_seed", -1)) != seed
                    or row.get("model_id") != "diffusion"
                    or int(row.get("posterior_sample_count_K", -1)) != 8
                    or not _truth(
                        row.get("risk_coverage_input_only_not_success_claim")
                    )
                    or not _truth(
                        row.get(
                            "query_EOG_and_labels_opened_after_all_outputs_frozen"
                        )
                    )
                    or _truth(
                        row.get(
                            "query_EOG_or_labels_used_for_inference_or_sample_selection"
                        )
                    )
                    or _truth(row.get("best_of_K_used"))
                    or any(_finite(row.get(field)) is None for field in required_finite)
                ):
                    raise ValueError(
                        f"invalid post-freeze uncertainty input row: {path}"
                    )
                if (
                    row.get("inference_status") == "success"
                    and arm is not None
                    and _performance_eligible(arm)
                ):
                    scientifically_eligible_row_count += 1
                endpoints.add(endpoint)
                recordings.add(recording)
                row_count += 1
    all_files_present = present_files == J4_SUMMARY_COUNT
    return {
        "status": (
            "postfreeze_diffusion_proxy_inputs_present_"
            "matched_deterministic_ensemble_not_operationalized"
            if all_files_present and row_count > 0
            else "postfreeze_uncertainty_inputs_incomplete"
        ),
        "expected_fold_seed_files": J4_SUMMARY_COUNT,
        "present_fold_seed_files": present_files,
        "row_count": row_count,
        "scientifically_eligible_window_row_count": (
            scientifically_eligible_row_count
        ),
        "recording_count": len(recordings),
        "checkpoint_endpoints": sorted(endpoints),
        "query_fields_used_for_inference_or_sample_selection": False,
        "best_of_K_used": False,
        "go_no_go_uncertainty_contribution_established": False,
        "missing_for_fair_comparison": [
            "three_seed_deterministic_window_level_ensemble_uncertainty",
            "frozen_error_calibration_statistic",
            "frozen_risk_coverage_AUC_statistic",
        ],
    }


def _stem_metadata(folds: Sequence[FoldSpec]) -> dict[str, dict[str, Any]]:
    return {
        recording: {
            "recording_key": recording,
            "participant_stem": recording.split("/", 1)[-1],
            "study": fold.study,
            "layout_id": fold.layout_id,
            "sampling_rate_hz": fold.sampling_rate_hz,
            "reference_cell": fold.reference_cell,
            "exact_cell_id": fold.cell_id,
        }
        for fold in folds
        for recording in fold.heldout_recording_keys
    }


def _arm_index(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int, str, str, str], Mapping[str, Any]]:
    return {
        (
            str(row["recording_key"]),
            int(row["training_seed"]),
            str(row["checkpoint_endpoint"]),
            str(row["model_id"]),
            str(row["context_id"]),
        ): row
        for row in rows
        if row.get("model_id") in MODEL_IDS
    }


def _build_coverage(
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    folds: Sequence[FoldSpec],
) -> list[dict[str, Any]]:
    seeds = tuple(int(value) for value in _mapping(config, "training")["seeds"])
    recordings = tuple(sorted(_stem_metadata(folds)))
    endpoints = tuple(
        sorted(
            {
                str(row["checkpoint_endpoint"])
                for row in rows
                if row.get("model_id") in MODEL_IDS
            }
        )
    )
    index = _arm_index(rows)
    result: list[dict[str, Any]] = []
    for endpoint in endpoints:
        for model in MODEL_IDS:
            for context in CONTEXT_IDS:
                arm_rows = [
                    index.get((recording, seed, endpoint, model, context))
                    for recording in recordings
                    for seed in seeds
                ]
                present = [row for row in arm_rows if row is not None]
                statuses = Counter(str(row.get("status", "")) for row in present)
                eligible = sum(_performance_eligible(row) for row in present)
                complete = sum(
                    all(
                        (row := index.get((recording, seed, endpoint, model, context)))
                        is not None
                        and _performance_eligible(row)
                        for seed in seeds
                    )
                    for recording in recordings
                )
                result.append(
                    {
                        "checkpoint_endpoint": endpoint,
                        "model_id": model,
                        "context_id": context,
                        "availability_stem_denominator": EXPECTED_AVAILABILITY_STEMS,
                        "compatible_stem_denominator": EXPECTED_COMPATIBLE_STEMS,
                        "preblocked_stem_count": 1,
                        "preblocked_recording_key": EXPECTED_PREBLOCKED_STEM,
                        "planned_stem_seed_rows": EXPECTED_COMPATIBLE_STEMS
                        * len(seeds),
                        "present_stem_seed_rows": len(present),
                        "performance_eligible_stem_seed_rows": eligible,
                        "all_three_seed_complete_stems": complete,
                        "incomplete_stems": EXPECTED_COMPATIBLE_STEMS - complete,
                        "status_counts": dict(sorted(statuses.items())),
                    }
                )
    return result


def _build_stem_arm_metrics(
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    folds: Sequence[FoldSpec],
) -> list[dict[str, Any]]:
    seeds = tuple(int(value) for value in _mapping(config, "training")["seeds"])
    metadata = _stem_metadata(folds)
    index = _arm_index(rows)
    endpoints = sorted(
        {
            str(row["checkpoint_endpoint"])
            for row in rows
            if row.get("model_id") in MODEL_IDS
        }
    )
    result: list[dict[str, Any]] = []
    for recording, meta in sorted(metadata.items()):
        for endpoint in endpoints:
            for model in MODEL_IDS:
                for context in CONTEXT_IDS:
                    seed_rows = [
                        index.get((recording, seed, endpoint, model, context))
                        for seed in seeds
                    ]
                    complete = all(
                        row is not None and _performance_eligible(row)
                        for row in seed_rows
                    )
                    output: dict[str, Any] = {
                        **meta,
                        "checkpoint_endpoint": endpoint,
                        "model_id": model,
                        "context_id": context,
                        "complete_three_seed_arm": complete,
                        "successful_seed_count": sum(
                            row is not None and _performance_eligible(row)
                            for row in seed_rows
                        ),
                    }
                    for metric in METRIC_DIRECTIONS:
                        values = [
                            _finite(row.get(metric)) if row is not None else None
                            for row in seed_rows
                        ]
                        output[metric] = (
                            float(np.mean([float(value) for value in values]))
                            if complete and all(value is not None for value in values)
                            else None
                        )
                    for resource in (
                        "latency_seconds_per_window",
                        "peak_memory_mb",
                        "network_calls",
                    ):
                        values = [
                            _finite(row.get(resource)) if row is not None else None
                            for row in seed_rows
                        ]
                        output[resource] = (
                            float(np.mean([float(value) for value in values]))
                            if complete and all(value is not None for value in values)
                            else None
                        )
                    result.append(output)
    return result


_CONTRAST_TERMS: dict[
    str, tuple[tuple[float, str, str], ...]
] = {
    "delta_cal_diff": (
        (1.0, "diffusion", "matching"),
        (-1.0, "diffusion", "population"),
    ),
    "delta_cal_det": (
        (1.0, "deterministic", "matching"),
        (-1.0, "deterministic", "population"),
    ),
    "delta_diff": (
        (1.0, "diffusion", "matching"),
        (-1.0, "deterministic", "matching"),
    ),
    "delta_interaction": (
        (1.0, "diffusion", "matching"),
        (-1.0, "diffusion", "population"),
        (-1.0, "deterministic", "matching"),
        (1.0, "deterministic", "population"),
    ),
    "diff_matching_minus_wrong": (
        (1.0, "diffusion", "matching"),
        (-1.0, "diffusion", "wrong"),
    ),
    "diff_matching_minus_shuffled": (
        (1.0, "diffusion", "matching"),
        (-1.0, "diffusion", "shuffled"),
    ),
    "det_matching_minus_wrong": (
        (1.0, "deterministic", "matching"),
        (-1.0, "deterministic", "wrong"),
    ),
    "det_matching_minus_shuffled": (
        (1.0, "deterministic", "matching"),
        (-1.0, "deterministic", "shuffled"),
    ),
}


def _seed_contrast(
    index: Mapping[tuple[str, int, str, str, str], Mapping[str, Any]],
    *,
    recording: str,
    seed: int,
    endpoint: str,
    metric: str,
    terms: Sequence[tuple[float, str, str]],
) -> float | None:
    total = 0.0
    for coefficient, model, context in terms:
        row = index.get((recording, seed, endpoint, model, context))
        if row is None or not _performance_eligible(row, metric):
            return None
        value = _utility(metric, row.get(metric))
        if value is None:
            return None
        total += coefficient * value
    return float(total)


def _build_stem_contrasts(
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    folds: Sequence[FoldSpec],
) -> list[dict[str, Any]]:
    """Pair within seed first, then average the three seed effects per stem."""

    seeds = tuple(int(value) for value in _mapping(config, "training")["seeds"])
    metadata = _stem_metadata(folds)
    index = _arm_index(rows)
    endpoints = sorted(
        {
            str(row["checkpoint_endpoint"])
            for row in rows
            if row.get("model_id") in MODEL_IDS
        }
    )
    result: list[dict[str, Any]] = []
    for recording, meta in sorted(metadata.items()):
        for endpoint in endpoints:
            for contrast_id, terms in _CONTRAST_TERMS.items():
                for metric, direction in METRIC_DIRECTIONS.items():
                    seed_values = [
                        _seed_contrast(
                            index,
                            recording=recording,
                            seed=seed,
                            endpoint=endpoint,
                            metric=metric,
                            terms=terms,
                        )
                        for seed in seeds
                    ]
                    complete = all(value is not None for value in seed_values)
                    result.append(
                        {
                            **meta,
                            "checkpoint_endpoint": endpoint,
                            "contrast_id": contrast_id,
                            "metric": metric,
                            "metric_direction": direction,
                            "positive_is_improvement": True,
                            "complete_three_seed_pair": complete,
                            "successful_seed_pair_count": sum(
                                value is not None for value in seed_values
                            ),
                            "seed_effects": [
                                None if value is None else float(value)
                                for value in seed_values
                            ],
                            "mean_effect": (
                                float(
                                    np.mean(
                                        [float(value) for value in seed_values]
                                    )
                                )
                                if complete
                                else None
                            ),
                        }
                    )
    return result


def stratified_participant_bootstrap(
    values_by_cell: Mapping[str, Sequence[float]],
    *,
    replicates: int,
    seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    """Equal-stem mean with cell sizes held fixed in every bootstrap draw."""

    if replicates < 1 or not 0.0 < confidence_level < 1.0:
        raise ValueError("invalid participant bootstrap configuration")
    groups = {
        str(cell): np.asarray(tuple(values), dtype=np.float64)
        for cell, values in values_by_cell.items()
        if len(values) > 0
    }
    if not groups or any(
        values.ndim != 1 or not np.isfinite(values).all() for values in groups.values()
    ):
        raise ValueError("participant bootstrap requires finite one-dimensional cells")
    total = sum(values.size for values in groups.values())
    rng = np.random.default_rng(int(seed))
    boot = np.zeros(int(replicates), dtype=np.float64)
    # The largest frozen input is 58 x 20,000, so direct vectorization is small.
    for values in groups.values():
        indices = rng.integers(0, values.size, size=(int(replicates), values.size))
        boot += values[indices].sum(axis=1)
    boot /= float(total)
    observed = np.concatenate(tuple(groups.values()))
    alpha = (1.0 - float(confidence_level)) / 2.0
    return {
        "participant_stem_count": int(total),
        "cell_count": len(groups),
        "mean": float(np.mean(observed)),
        "median": float(np.median(observed)),
        "wins": int(np.sum(observed > 0.0)),
        "ties": int(np.sum(observed == 0.0)),
        "losses": int(np.sum(observed < 0.0)),
        "ci_low": float(np.quantile(boot, alpha)),
        "ci_high": float(np.quantile(boot, 1.0 - alpha)),
        "bootstrap_replicates": int(replicates),
        "bootstrap_seed": int(seed),
        "confidence_level": float(confidence_level),
        "resampling_unit": "participant_stem_stratified_by_exact_cell",
    }


def _build_bootstrap_rows(
    config: Mapping[str, Any], contrasts: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    gates = _mapping(config, "development_gates")
    replicates = int(gates["bootstrap_replicates"])
    seed = int(gates["bootstrap_seed"])
    confidence = float(gates["confidence_level"])
    if replicates != 20_000 or seed != 20260802 or confidence != 0.95:
        raise ValueError("frozen participant bootstrap configuration changed")
    grouped: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in contrasts:
        value = _finite(row.get("mean_effect"))
        if row.get("complete_three_seed_pair") is not True or value is None:
            continue
        key = (
            str(row["checkpoint_endpoint"]),
            str(row["contrast_id"]),
            str(row["metric"]),
        )
        grouped[key][str(row["exact_cell_id"])].append(value)
    result: list[dict[str, Any]] = []
    for (endpoint, contrast, metric), cells in sorted(grouped.items()):
        pooled = stratified_participant_bootstrap(
            cells,
            replicates=replicates,
            seed=seed,
            confidence_level=confidence,
        )
        result.append(
            {
                "scope": "all_compatible_cells_stratified",
                "checkpoint_endpoint": endpoint,
                "contrast_id": contrast,
                "metric": metric,
                "positive_is_improvement": True,
                **pooled,
            }
        )
        for cell, values in sorted(cells.items()):
            cell_result = stratified_participant_bootstrap(
                {cell: values},
                replicates=replicates,
                seed=seed,
                confidence_level=confidence,
            )
            result.append(
                {
                    "scope": "exact_cell",
                    "exact_cell_id": cell,
                    "checkpoint_endpoint": endpoint,
                    "contrast_id": contrast,
                    "metric": metric,
                    "positive_is_improvement": True,
                    **cell_result,
                }
            )
    if not result:
        return [
            {
                "scope": "none",
                "status": "no_complete_three_seed_participant_contrast",
                "bootstrap_replicates": replicates,
                "bootstrap_seed": seed,
                "confidence_level": confidence,
                "participant_stem_count": 0,
            }
        ]
    return result


def _bootstrap_lookup(
    rows: Sequence[Mapping[str, Any]],
    *,
    endpoint: str,
    contrast: str,
    metric: str,
) -> Mapping[str, Any] | None:
    matches = [
        row
        for row in rows
        if row.get("scope") == "all_compatible_cells_stratified"
        and row.get("checkpoint_endpoint") == endpoint
        and row.get("contrast_id") == contrast
        and row.get("metric") == metric
    ]
    if len(matches) > 1:
        raise ValueError("duplicate pooled bootstrap summary")
    return matches[0] if matches else None


def _natural_calibration_component(
    config: Mapping[str, Any],
    bootstrap: Sequence[Mapping[str, Any]],
    *,
    endpoint: str,
    contrast: str,
) -> dict[str, Any]:
    gates = _mapping(_mapping(config, "development_gates"), "G_calibration")
    primary = _bootstrap_lookup(
        bootstrap,
        endpoint=endpoint,
        contrast=contrast,
        metric="heldout_eog_prediction_remaining_ratio",
    )
    primary_pass = bool(
        primary is not None
        and float(primary["mean"])
        >= float(gates["natural_EOG_remaining_mean_improvement_minimum"])
        and float(primary["ci_low"]) > 0.0
    )
    safety_metrics = {
        "nonartifact_observation_preservation": float(
            gates["preservation_noninferiority_margin"]
        ),
        "condition_erp_observation_relative_preservation": float(
            gates["ERP_noninferiority_margin"]
        ),
        "reference_free_psd_distortion": float(
            gates["PSD_distortion_noninferiority_margin"]
        ),
        "reference_free_covariance_distortion": float(
            gates["covariance_distortion_noninferiority_margin"]
        ),
    }
    safety: dict[str, Any] = {}
    for metric, margin in safety_metrics.items():
        value = _bootstrap_lookup(
            bootstrap,
            endpoint=endpoint,
            contrast=contrast,
            metric=metric,
        )
        safety[metric] = {
            "margin": margin,
            "available": value is not None,
            "ci_low": None if value is None else value["ci_low"],
            "passed_noninferiority": bool(
                value is not None and float(value["ci_low"]) >= margin
            ),
        }
    safety_pass = all(value["passed_noninferiority"] for value in safety.values())
    return {
        "status": (
            "passed_natural_component"
            if primary_pass and safety_pass
            else "failed_or_inconclusive_natural_component"
        ),
        "checkpoint_endpoint": endpoint,
        "contrast_id": contrast,
        "primary": None if primary is None else dict(primary),
        "primary_passed": primary_pass,
        "safety": safety,
        "safety_passed": safety_pass,
        "EOG_coherence_is_supporting_not_sole_success_endpoint": True,
    }


def _gate_summary(
    config: Mapping[str, Any],
    endpoints: Sequence[str],
    bootstrap: Sequence[Mapping[str, Any]],
    uncertainty_inputs: Mapping[str, Any],
) -> dict[str, Any]:
    endpoint = (
        "best_validation" if "best_validation" in endpoints else str(endpoints[0])
    )
    diffusion_natural = _natural_calibration_component(
        config, bootstrap, endpoint=endpoint, contrast="delta_cal_diff"
    )
    deterministic_natural = _natural_calibration_component(
        config, bootstrap, endpoint=endpoint, contrast="delta_cal_det"
    )
    diff_candidate = _bootstrap_lookup(
        bootstrap,
        endpoint=endpoint,
        contrast="delta_diff",
        metric="heldout_eog_prediction_remaining_ratio",
    )
    equal_available = "equal_update" in endpoints
    return {
        "G_calibration": {
            "status": "not_testable_missing_paired_mechanism_evidence",
            "paired_mechanism": {
                "status": "not_run",
                "missing_fields": [
                    "artifact_latent_truth_error",
                    "clean_RRMSE",
                    "clean_correlation",
                    "clean_SNR",
                ],
            },
            "natural_diffusion_component": diffusion_natural,
            "natural_deterministic_component": deterministic_natural,
            "overall_passed": False,
        },
        "G_diffusion": {
            "status": "not_testable_primary_endpoint_not_operationalized",
            "descriptive_EOG_remaining_candidate": (
                None if diff_candidate is None else dict(diff_candidate)
            ),
            "safety_noninferiority_margin": _mapping(
                _mapping(config, "development_gates"), "G_diffusion"
            )["safety_noninferiority_margin"],
            "overall_passed": False,
        },
        "uncertainty": {
            **dict(uncertainty_inputs),
            "status": "not_testable_missing_matched_uncertainty_comparison",
            "posterior_SD_scalar_is_not_a_calibration_analysis": True,
            "overall_passed": False,
        },
        "equal_update_comparison": {
            "status": (
                "available_descriptive"
                if equal_available
                else "not_run_missing_endpoint"
            ),
            "required_for_complete_budget_audit": True,
        },
        "calibration_duration": {
            "status": "not_testable_single_30_second_condition",
            "rho_zero_is_not_a_zero_second_duration_arm": True,
        },
        "confirmation_eligibility": False,
        "protocol_decision": "inconclusive",
        "topic_status": "not_yet_testable",
    }


def _plot_paired_delta(
    path: Path,
    contrasts: Sequence[Mapping[str, Any]],
    *,
    endpoint: str,
) -> bool:
    selected = [
        row
        for row in contrasts
        if row.get("checkpoint_endpoint") == endpoint
        and row.get("metric") == "heldout_eog_prediction_remaining_ratio"
        and row.get("contrast_id") in {
            "delta_cal_diff",
            "delta_cal_det",
            "diff_matching_minus_wrong",
            "diff_matching_minus_shuffled",
            "det_matching_minus_wrong",
            "det_matching_minus_shuffled",
        }
        and _finite(row.get("mean_effect")) is not None
    ]
    if not selected:
        return False
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(11, 5.5))
    labels = sorted({str(row["contrast_id"]) for row in selected})
    for index, label in enumerate(labels):
        values = [
            float(row["mean_effect"])
            for row in selected
            if row["contrast_id"] == label
        ]
        axis.scatter(
            np.full(len(values), index),
            values,
            alpha=0.55,
            s=16,
            label=label,
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    axis.set_ylabel("EOG-remaining utility delta (positive is improvement)")
    axis.set_title("Participant-stem paired context deltas (three-seed means)")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return True


def _plot_pareto(
    path: Path,
    arm_rows: Sequence[Mapping[str, Any]],
    *,
    endpoint: str,
) -> bool:
    series: list[tuple[str, list[float], list[float]]] = []
    for model in MODEL_IDS:
        for context in CONTEXT_IDS:
            selected = [
                row
                for row in arm_rows
                if row.get("checkpoint_endpoint") == endpoint
                and row.get("model_id") == model
                and row.get("context_id") == context
                and _finite(row.get("heldout_eog_prediction_remaining_ratio"))
                is not None
                and _finite(row.get("nonartifact_observation_preservation"))
                is not None
            ]
            if not selected:
                continue
            x = [
                1.0 - float(row["heldout_eog_prediction_remaining_ratio"])
                for row in selected
            ]
            y = [
                float(row["nonartifact_observation_preservation"])
                for row in selected
            ]
            series.append((f"{model}:{context}", x, y))
    if not series:
        return False
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 6))
    for label, x, y in series:
        axis.scatter(x, y, alpha=0.35, s=14, label=label)
        axis.scatter([np.mean(x)], [np.mean(y)], marker="X", s=80)
    axis.set_xlabel("1 - held-out EOG prediction remaining ratio")
    axis.set_ylabel("Non-artifact observation preservation")
    axis.set_title("Artifact attenuation–preservation development Pareto")
    axis.legend(fontsize=7, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return True


def _aggregate_passed_validity(
    config: Mapping[str, Any],
    run_dir: Path,
    gate: Mapping[str, Any],
    implementation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    paths = _output_paths(config)
    folds = _load_fold_specs(config)
    j3 = _load_j3_summaries(config, folds)
    j4_summaries, rows = _load_j4_rows(config, folds)
    selected_result = _selected_validity_result(config, gate)
    selected_repair = str(gate.get("selected_implementation", ""))
    if not selected_repair:
        raise ValueError("J2 passed without a selected validity implementation")
    identity_repair = selected_result.get("identity_repair_active")
    if not isinstance(identity_repair, bool):
        raise ValueError("selected J2 result lacks the identity-repair decision")
    expected_clip = float(
        _mapping(config, "primary_diffusion")[
            "standardized_latent_absolute_clip"
        ]
    )
    if selected_repair == "primary_attempt_2":
        expected_clip = 3.0
    expected_sampler = "deterministic_DDIM_in_artifact_latent_space"
    if any(
        not isinstance(value.get("training"), Mapping)
        or value["training"].get("selected_validity_repair") != selected_repair
        or value["training"].get("identity_repair_active") is not identity_repair
        or _finite(
            value["training"].get(
                "effective_standardized_latent_absolute_clip"
            )
        )
        != expected_clip
        or value["training"].get("effective_inference_sampler")
        != expected_sampler
        for value in j3
    ):
        raise ValueError("J3 did not inherit the selected J2 validity repair")
    if any(
        value.get("validity_execution_revision")
        != gate.get("execution_revision")
        or value.get("validity_selected_implementation") != selected_repair
        or value.get("identity_repair_active") is not identity_repair
        or _finite(
            value.get("effective_standardized_latent_absolute_clip")
        )
        != expected_clip
        or value.get("effective_inference_sampler") != expected_sampler
        for value in j4_summaries
    ):
        raise ValueError("J4 did not inherit the selected J2 validity revision")
    coverage = _build_coverage(config, rows, folds)
    arms = _build_stem_arm_metrics(config, rows, folds)
    contrasts = _build_stem_contrasts(config, rows, folds)
    bootstrap = _build_bootstrap_rows(config, contrasts)
    uncertainty_inputs = _audit_uncertainty_inputs(config, folds, rows)
    endpoints = sorted(
        {
            str(row["checkpoint_endpoint"])
            for row in rows
            if row.get("model_id") in MODEL_IDS
        }
    )
    gates = _gate_summary(config, endpoints, bootstrap, uncertainty_inputs)
    aggregate_root = _aggregate_root(config)
    _write_csv(paths["metrics"], rows)
    _write_csv(aggregate_root / "coverage.csv", coverage)
    _write_csv(aggregate_root / "stem_arm_metrics.csv", arms)
    _write_csv(aggregate_root / "stem_contrasts.csv", contrasts)
    _write_csv(aggregate_root / "bootstrap_summary.csv", bootstrap)
    _atomic_json(
        aggregate_root / "uncertainty_input_audit.json", uncertainty_inputs
    )
    endpoint = "best_validation" if "best_validation" in endpoints else endpoints[0]
    paired_figure = paths["figures"] / "participant_paired_deltas.png"
    pareto_figure = paths["figures"] / "attenuation_preservation_pareto.png"
    paired_figure_written = _plot_paired_delta(
        paired_figure, contrasts, endpoint=endpoint
    )
    pareto_figure_written = _plot_pareto(
        pareto_figure, arms, endpoint=endpoint
    )
    training_implementations = sorted(
        {str(value.get("implementation", "")) for value in j3}
    )
    evaluation_implementations = sorted(
        {
            str(row.get("git_commit", ""))
            for row in rows
            if str(row.get("git_commit", ""))
        }
    )
    summary: dict[str, Any] = {
        "status": "completed_development_aggregate_inconclusive",
        "stage": "J5_participant_stem_aggregate",
        "protocol_id": config.get("protocol_id"),
        "execution_revision": gate.get("execution_revision"),
        "scientific_role": "development_exploratory_not_confirmation",
        "computational_completion": "passed",
        "model_validity": "passed",
        "validity_selected_implementation": gate.get("selected_implementation"),
        "J3_training": {
            "status": "complete",
            "stable_summary_count": len(j3),
            "expected_stable_summary_count": J3_SUMMARY_COUNT,
            "implementation_values": training_implementations,
        },
        "J4_factorial": {
            "status": "complete",
            "terminal_summary_count": len(j4_summaries),
            "expected_terminal_summary_count": J4_SUMMARY_COUNT,
            "metric_row_count": len(rows),
            "checkpoint_endpoints": endpoints,
            "implementation_values": evaluation_implementations,
        },
        "coverage": {
            "availability_stem_denominator": EXPECTED_AVAILABILITY_STEMS,
            "compatible_stem_denominator": EXPECTED_COMPATIBLE_STEMS,
            "preblocked_recording_key": EXPECTED_PREBLOCKED_STEM,
            "coverage_table": str(aggregate_root / "coverage.csv"),
        },
        **gates,
        "scientific_comparison_eligibility": "development_descriptive_only",
        "confirmation_eligibility": False,
        "real_EEG_evidence_scope": (
            "all_eligible_SGEYESUB_stems_development_exploratory_"
            "artifact_attenuation_and_preservation_proxies_no_clean_target"
        ),
        "family_wide_status": "not_tested",
        "query_confirmation_outcomes_opened": False,
        "outputs": {
            "metrics": str(paths["metrics"]),
            "coverage": str(aggregate_root / "coverage.csv"),
            "stem_arm_metrics": str(aggregate_root / "stem_arm_metrics.csv"),
            "stem_contrasts": str(aggregate_root / "stem_contrasts.csv"),
            "bootstrap": str(aggregate_root / "bootstrap_summary.csv"),
            "uncertainty_input_audit": str(
                aggregate_root / "uncertainty_input_audit.json"
            ),
        },
        "figures": {
            "trajectory_rms": str(
                paths["validity_root"]
                / str(gate.get("execution_revision"))
                / str(gate.get("selected_implementation"))
                / "diffusion_trajectory_rms.png"
            ),
            "paired_delta": (
                str(paired_figure)
                if paired_figure_written
                else "not_generated_no_complete_paired_stem_metric"
            ),
            "attenuation_preservation_pareto": (
                str(pareto_figure)
                if pareto_figure_written
                else "not_generated_no_complete_pareto_metric_pair"
            ),
            "calibration_duration": "not_generated_single_30_second_condition",
        },
        "unresolved_blockers": [
            "paired_mechanism_metrics_not_emitted_by_J4",
            "matched_deterministic_ensemble_uncertainty_and_frozen_"
            "risk_coverage_not_operationally_available",
            "G_diffusion_primary_endpoint_not_named_in_frozen_config",
            "calibration_duration_axis_not_run",
            *(
                []
                if "equal_update" in endpoints
                else ["equal_update_checkpoint_endpoint_not_evaluated"]
            ),
        ],
        **_implementation_fields(implementation),
    }
    _atomic_json(aggregate_root / "gate_status.json", gates)
    _atomic_json(aggregate_root / "result_summary.json", summary)
    run_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(run_dir / "result_summary.json", summary)
    return summary


def run_subject_artifact_development_aggregate(
    config: Mapping[str, Any],
    run_dir: str | Path,
    implementation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run J5, terminating safely whether J2 failed or passed."""

    gate = _validity_gate(config)
    destination = Path(run_dir)
    if gate.get("status") == "completed_model_validity_failed" or gate.get(
        "passed"
    ) is not True:
        return _fail_closed_after_validity(
            config, destination, gate, implementation
        )
    if gate.get("status") != "passed_V0_to_V3":
        raise ValueError("unrecognized terminal J2 validity state")
    return _aggregate_passed_validity(config, destination, gate, implementation)


__all__ = [
    "METRIC_DIRECTIONS",
    "run_subject_artifact_development_aggregate",
    "stratified_participant_bootstrap",
]
