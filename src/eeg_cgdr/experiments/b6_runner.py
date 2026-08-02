"""Runner boundary for deferred B6 with split-derived provenance.

The numerical B6 operator accepts already validated projector metadata.  This
module is the only experiment-side adapter allowed to construct that metadata:
it reads the frozen split and population-state artifact, derives the actual
support through the formal loader, fits P0 from that support, and only then
calls POP-SHRINK.  No
dataset, montage, reference, preprocessing, channel-order, or fit-scope string
is accepted from a call site.

B6 remains a development-only deferred diagnostic.  Any disabled, incompatible
or ineligible context returns an explicit POP decision with no context
projector.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from eeg_cgdr.data.klados import load_klados_records
from eeg_cgdr.data.mechanism import (
    KLADOS_NATIVE_CHANNEL_ORDER,
    fit_channel_normalizer,
    resample_signal,
    select_records,
    standardize_reference_from_support,
)
from eeg_cgdr.inference.states import DatasetPopulationProjector
from eeg_cgdr.operators.p0 import CalibrationBatch, P0Config, fit_p0
from eeg_cgdr.operators.pop_shrink import (
    PopShrinkOutcome,
    ProjectorCompatibilityKey,
    spectral_projector_shrink,
)


_DATASET_ID = "klados_bamidis_v4"
_MONTAGE_ID = "klados_v4_19ch_native_order_256hz"
_POPULATION_SOURCE = "all_training_source_records_sim01_sim30"
_POPULATION_FIT_SCOPE = "outer_training_only"
_CONTEXT_FIT_SCOPE = "support_only"


@dataclass(frozen=True)
class B6RunnerResult:
    """One runner decision; ``fallback_POP`` never carries a projector."""

    outcome: PopShrinkOutcome
    compatibility: Optional[ProjectorCompatibilityKey]
    population_source_records: tuple[str, ...]
    context_support_record: Optional[str]
    partition: Optional[str]
    population_fit_scope: str = _POPULATION_FIT_SCOPE
    context_fit_scope: Optional[str] = None

    @property
    def use_pop(self) -> bool:
        return self.outcome.status == "fallback_POP"


def _fallback(reason: str) -> B6RunnerResult:
    return B6RunnerResult(
        outcome=PopShrinkOutcome(
            status="fallback_POP",
            projector=None,
            reasons=(reason,),
            diagnostics={
                "operator": "B6_POP_SHRINK",
                "fallback": "POP",
                "runner_contract": "actual_split_derived_provenance",
            },
            context_projector_constructed=False,
        ),
        compatibility=None,
        population_source_records=(),
        context_support_record=None,
        partition=None,
    )


def _mapping(parent: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"missing mapping: {key}")
    return value


def _source_record(value: object) -> str:
    text = str(value)
    if len(text) != 5 or not text.startswith("sim") or not text[3:].isdigit():
        raise ValueError("source record must use simNN")
    number = int(text[3:])
    if not 1 <= number <= 54:
        raise ValueError("source record is outside sim01-sim54")
    return f"sim{number:02d}"


def _read_split(config: Mapping[str, Any]) -> tuple[list[dict[str, str]], Path]:
    klados = _mapping(config, "klados")
    path = Path(str(klados.get("split_manifest", "")))
    if not str(path) or not path.is_file():
        raise ValueError("frozen Klados split manifest is unavailable")
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "dataset_version",
            "outer_fold",
            "split",
            "record",
            "calibration_start",
            "calibration_end",
            "query_start",
            "sampling_rate",
            "status",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("split manifest is missing required fields")
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError("split manifest is empty")
    if {row["dataset_version"] for row in rows} != {_DATASET_ID}:
        raise ValueError("split dataset_version is incompatible")
    if len({row["outer_fold"] for row in rows}) != 1:
        raise ValueError("split manifest must describe exactly one frozen fold")
    records = [_source_record(row["record"]) for row in rows]
    if len(records) != len(set(records)):
        raise ValueError("split manifest contains duplicate source records")
    return rows, path


def _preprocessing_id(config: Mapping[str, Any]) -> str:
    klados = _mapping(config, "klados")
    preprocessing = _mapping(config, "preprocessing")
    p0 = _mapping(config, "p0")
    if int(klados.get("source_sampling_rate", -1)) != 200:
        raise ValueError("Klados native sampling rate must be 200 Hz")
    if int(preprocessing.get("target_sampling_rate", -1)) != 256:
        raise ValueError("Klados target sampling rate must be 256 Hz")
    if int(preprocessing.get("window_samples", -1)) != 512:
        raise ValueError("Klados mechanism window must contain 512 samples")
    if preprocessing.get("padding_value_after_normalization") != 0.0:
        raise ValueError("padding must be normalized zero")
    if p0.get("reference_standardization") != "support_channel_zscore":
        raise ValueError("P0 EOG standardization must use support only")
    normalization = str(preprocessing.get("normalization", ""))
    if normalization != "per-channel moments from complete clean training source records only":
        raise ValueError("normalization provenance is incompatible")
    return (
        "klados_mechanism_resample_200_to_256__window_512__"
        "train_clean_channel_moments__padding_zero__support_eog_zscore"
    )


def _compatibility(
    config: Mapping[str, Any], rows: list[dict[str, str]]
) -> ProjectorCompatibilityKey:
    klados = _mapping(config, "klados")
    channel_order = tuple(str(value) for value in klados.get("channel_order", ()))
    if channel_order != KLADOS_NATIVE_CHANNEL_ORDER:
        raise ValueError("Klados channel order is incompatible")
    reference_id = str(klados.get("reference_id", ""))
    if reference_id != "linked_mastoids_native_odd_left_even_right_midline_average":
        raise ValueError("Klados reference provenance is incompatible")
    if {row["dataset_version"] for row in rows} != {_DATASET_ID}:
        raise ValueError("split and acquisition dataset IDs differ")
    return ProjectorCompatibilityKey(
        dataset_id=_DATASET_ID,
        montage_id=_MONTAGE_ID,
        reference_id=reference_id,
        preprocessing_id=_preprocessing_id(config),
        channel_order=channel_order,
    )


def _validate_backup_config(
    backup_config: Mapping[str, Any], gamma: float, rank: int
) -> None:
    if backup_config.get("enabled") is not True:
        raise ValueError("b6_disabled")
    if backup_config.get("backup_id") != "B6" or backup_config.get("formal_name") != "POP-SHRINK":
        raise ValueError("b6_identity")
    selection = _mapping(backup_config, "selection")
    operator = _mapping(backup_config, "operator")
    if selection.get("scope") != "development_only":
        raise ValueError("b6_not_development_only")
    if selection.get("one_global_gamma") is not True:
        raise ValueError("b6_gamma_not_global")
    allowed_gamma = tuple(selection.get("gamma_candidates", ())) + tuple(
        selection.get("endpoints_as_controls", ())
    )
    if gamma not in tuple(float(value) for value in allowed_gamma):
        raise ValueError("b6_gamma_not_preregistered")
    if int(operator.get("rank", -1)) != rank:
        raise ValueError("b6_rank_mismatch")
    if operator.get("required_population_fit_scope") != _POPULATION_FIT_SCOPE:
        raise ValueError("b6_population_scope_config")
    if operator.get("required_context_fit_scope") != _CONTEXT_FIT_SCOPE:
        raise ValueError("b6_context_scope_config")


def _validate_population(
    config: Mapping[str, Any],
    rows: list[dict[str, str]],
    compatibility: ProjectorCompatibilityKey,
    population_projector: DatasetPopulationProjector,
) -> tuple[np.ndarray, tuple[str, ...]]:
    train_records = tuple(
        _source_record(row["record"]) for row in rows if row["split"] == "train"
    )
    expected_train = tuple(f"sim{index:02d}" for index in range(1, 31))
    if train_records != expected_train:
        raise ValueError("population split must be exactly sim01-sim30")
    if any(
        row["status"] != "population_prior_and_projector"
        for row in rows
        if row["split"] == "train"
    ):
        raise ValueError("population split status is incompatible")
    configured_train = tuple(
        f"sim{int(value):02d}"
        for value in _mapping(config, "klados").get("training_source_records", ())
    )
    if configured_train != train_records:
        raise ValueError("config training records differ from actual split")

    output_config = _mapping(config, "outputs")
    artifact_path = Path(str(output_config.get("population_state", "")))
    if not artifact_path.is_file():
        raise ValueError("population projector artifact is unavailable")
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    expected_payload = {
        "dataset_id": compatibility.dataset_id,
        "montage_id": compatibility.montage_id,
        "source": _POPULATION_SOURCE,
        "channel_order": list(compatibility.channel_order),
        "training_source_records": list(range(1, 31)),
        "source_sampling_rate": 200,
        "target_sampling_rate": 256,
        "reference_id": compatibility.reference_id,
        "preprocessing_id": compatibility.preprocessing_id,
        "fit_scope": "joint_concatenation_of_all_30_training_source_records",
        "records_total": 30,
    }
    for field_name, expected in expected_payload.items():
        if payload.get(field_name) != expected:
            raise ValueError(f"population artifact {field_name} provenance differs")
    artifact_projector = np.asarray(payload.get("projector"), dtype=np.float64)
    supplied_projector = np.asarray(population_projector.projector, dtype=np.float64)
    if (
        population_projector.dataset_id != compatibility.dataset_id
        or population_projector.montage_id != compatibility.montage_id
        or population_projector.source != _POPULATION_SOURCE
        or artifact_projector.shape != supplied_projector.shape
        or not np.array_equal(artifact_projector, supplied_projector)
    ):
        raise ValueError("runner population projector differs from its actual artifact")
    return supplied_projector, train_records


def _p0_config(config: Mapping[str, Any]) -> P0Config:
    raw = _mapping(config, "p0")
    return P0Config(
        target_rank=int(raw["target_rank"]),
        ridge_lambda=float(raw["ridge_lambda"]),
        maximum_reference_condition=float(raw["maximum_reference_condition"]),
        minimum_singular_ratio=float(raw["minimum_singular_ratio"]),
        minimum_movement_coverage=float(raw["minimum_movement_coverage"]),
        bootstrap_replicates=int(raw["bootstrap_replicates"]),
        bootstrap_block_samples=int(raw["bootstrap_block_samples"]),
        minimum_bootstrap_success=float(raw["minimum_bootstrap_success"]),
        maximum_bootstrap_median_distance=float(
            raw["maximum_bootstrap_median_distance"]
        ),
        maximum_bootstrap_q90_distance=float(raw["maximum_bootstrap_q90_distance"]),
        seed=int(config["seed"]),
    )


def _context_row(
    config: Mapping[str, Any],
    rows: list[dict[str, str]],
    source_record: int,
) -> dict[str, str]:
    if isinstance(source_record, bool) or not isinstance(
        source_record, (int, np.integer)
    ):
        raise ValueError("source_record must be an integer record ID")
    record = _source_record(f"sim{int(source_record):02d}")
    matches = [row for row in rows if _source_record(row["record"]) == record]
    if len(matches) != 1:
        raise ValueError("source record is absent or duplicated in split")
    row = matches[0]
    if row["split"] != "development":
        raise ValueError("B6 context is not in the development partition")
    if row["status"] != "source_record_only_participant_mapping_unavailable":
        raise ValueError("B6 context split status is incompatible")
    source_rate = int(_mapping(config, "klados")["source_sampling_rate"])
    if int(float(row["sampling_rate"])) != source_rate:
        raise ValueError("context split sampling rate differs from config")
    calibration_start = float(row["calibration_start"])
    calibration_end = float(row["calibration_end"])
    query_start = float(row["query_start"])
    if not (
        math.isfinite(calibration_start)
        and math.isfinite(calibration_end)
        and math.isfinite(query_start)
        and calibration_start == 0.0
        and calibration_end > calibration_start
        and query_start > calibration_end
    ):
        raise ValueError("context support/query timing is invalid")
    klados = _mapping(config, "klados")
    if not math.isclose(
        calibration_end - calibration_start,
        float(klados.get("calibration_seconds", float("nan"))),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("split support duration differs from config")
    if not math.isclose(
        query_start - calibration_end,
        float(klados.get("guard_seconds", float("nan"))),
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("split guard duration differs from config")
    return row


def _loader_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Build the only permitted Klados loader request from frozen config."""

    klados = _mapping(config, "klados")
    data_root = str(klados.get("data_root", ""))
    files = {
        key: str(klados.get(key, ""))
        for key in ("contaminated", "clean", "heog", "veog")
    }
    if not data_root or any(not value for value in files.values()):
        raise ValueError("formal Klados loader paths are incomplete")
    return {
        "data_root": data_root,
        "files": files,
        "official_description": {"records": 54},
    }


def _derive_calibration_from_split(
    config: Mapping[str, Any],
    rows: list[dict[str, str]],
    row: Mapping[str, str],
    source_record: int,
) -> CalibrationBatch:
    """Load and derive support internally; no caller-owned array is accepted.

    The formal loader necessarily opens the monolithic Klados MAT variables,
    but only clean EEG from the frozen training source records enters the
    normalizer.  The requested development record contributes only its
    contaminated EEG and VEOG/HEOG support interval to P0.
    """

    records = load_klados_records(_loader_config(config))
    actual_ids = tuple(record.record_id for record in records)
    if actual_ids != tuple(range(1, 55)):
        raise ValueError("formal Klados loader did not return sim01-sim54 in order")
    training_ids = tuple(
        int(_source_record(item["record"])[3:])
        for item in rows
        if item["split"] == "train"
    )
    normalizer = fit_channel_normalizer(records, training_ids)
    native = select_records(records, (source_record,))[0]

    source_rate = int(_mapping(config, "klados")["source_sampling_rate"])
    target_rate = int(_mapping(config, "preprocessing")["target_sampling_rate"])
    start_seconds = float(row["calibration_start"])
    stop_seconds = float(row["calibration_end"])
    query_start_seconds = float(row["query_start"])
    start_float = start_seconds * source_rate
    stop_float = stop_seconds * source_rate
    query_start_float = query_start_seconds * source_rate
    if (
        not math.isclose(start_float, round(start_float), abs_tol=1.0e-12)
        or not math.isclose(stop_float, round(stop_float), abs_tol=1.0e-12)
        or not math.isclose(
            query_start_float, round(query_start_float), abs_tol=1.0e-12
        )
    ):
        raise ValueError("split boundaries do not align to native samples")
    start = int(round(start_float))
    stop = int(round(stop_float))
    query_start = int(round(query_start_float))
    if start < 0 or stop <= start or query_start >= native.samples:
        raise ValueError("source record cannot supply the frozen support and query")
    if stop > native.samples:
        raise ValueError("source record support extends beyond record end")

    eeg = resample_signal(
        normalizer.transform(native.contaminated[:, start:stop]),
        source_rate,
        target_rate,
    )
    raw_eog = resample_signal(
        np.stack([native.veog, native.heog], axis=0)[:, start:stop],
        source_rate,
        target_rate,
    )
    eog, _, _, _ = standardize_reference_from_support(raw_eog, raw_eog)
    expected_samples = int(round((stop_seconds - start_seconds) * target_rate))
    if eeg.shape != (len(KLADOS_NATIVE_CHANNEL_ORDER), expected_samples):
        raise ValueError("formal loader produced an unexpected support EEG shape")
    if eog.shape != (2, expected_samples):
        raise ValueError("formal loader produced an unexpected support EOG shape")
    if not np.isfinite(eeg).all() or not np.isfinite(eog).all():
        raise ValueError("derived support contains non-finite values")
    if not np.allclose(eog.mean(axis=1), 0.0, atol=1.0e-10, rtol=0.0):
        raise ValueError("derived support EOG is not support-centered")
    if not np.allclose(eog.std(axis=1), 1.0, atol=1.0e-10, rtol=0.0):
        raise ValueError("derived support EOG is not support-standardized")
    return CalibrationBatch(
        eeg=eeg,
        eog=eog,
        participant="unresolved_source_record",
        source_record=f"sim{source_record:02d}",
        sampling_rate=float(target_rate),
    )


def run_deferred_b6_from_actual_split(
    *,
    config: Mapping[str, Any],
    backup_config: Mapping[str, Any],
    population_projector: DatasetPopulationProjector,
    source_record: int,
    gamma: float,
) -> B6RunnerResult:
    """Fit and route B6 from a record ID, frozen split, and formal loader.

    The signature intentionally has no caller-created support/calibration array,
    compatibility strings, fit-scope values, context projector, query,
    EOG-query, or clean-target argument.  For nonzero gamma the runner derives
    the exact support interval itself.  Gamma zero validates the split row but
    short-circuits before loading or constructing context data.
    """

    if backup_config.get("enabled") is not True:
        return _fallback("b6_disabled")
    try:
        gamma_value = float(gamma)
        if not math.isfinite(gamma_value):
            raise ValueError("b6_gamma_nonfinite")
        rows, _ = _read_split(config)
        row = _context_row(config, rows, source_record)
        compatibility = _compatibility(config, rows)
        rank = int(_mapping(config, "p0")["target_rank"])
        _validate_backup_config(backup_config, gamma_value, rank)
        population_value, population_records = _validate_population(
            config,
            rows,
            compatibility,
            population_projector,
        )

        # Exact POP endpoint: do not inspect or fit a context support batch.
        if gamma_value == 0.0:
            outcome = spectral_projector_shrink(
                population_value,
                None,
                rank=rank,
                gamma=0.0,
                context_eligible=False,
                population_compatibility=compatibility,
                population_fit_scope=_POPULATION_FIT_SCOPE,
                context_compatibility=None,
                context_fit_scope=None,
                minimum_eigengap=float(
                    _mapping(backup_config, "operator")["minimum_spectral_eigengap"]
                ),
            )
            return B6RunnerResult(
                outcome=outcome,
                compatibility=compatibility,
                population_source_records=population_records,
                context_support_record=None,
                partition=row["split"],
                context_fit_scope=None,
            )

        calibration = _derive_calibration_from_split(
            config, rows, row, int(source_record)
        )
        p0_outcome = fit_p0(
            calibration,
            _p0_config(config),
            movement_threshold=float(_mapping(config, "p0")["movement_threshold"]),
        )
        context_projector = (
            None if p0_outcome.transfer is None else p0_outcome.transfer.projector
        )
        outcome = spectral_projector_shrink(
            population_value,
            context_projector,
            rank=rank,
            gamma=gamma_value,
            context_eligible=p0_outcome.status == "eligible",
            population_compatibility=compatibility,
            population_fit_scope=_POPULATION_FIT_SCOPE,
            context_compatibility=(
                compatibility if p0_outcome.status == "eligible" else None
            ),
            context_fit_scope=(
                _CONTEXT_FIT_SCOPE if p0_outcome.status == "eligible" else None
            ),
            minimum_eigengap=float(
                _mapping(backup_config, "operator")["minimum_spectral_eigengap"]
            ),
        )
        return B6RunnerResult(
            outcome=outcome,
            compatibility=compatibility,
            population_source_records=population_records,
            context_support_record=_source_record(calibration.source_record),
            partition=row["split"],
            context_fit_scope=(
                _CONTEXT_FIT_SCOPE if p0_outcome.status == "eligible" else None
            ),
        )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
        reason = str(exc).strip().replace(" ", "_") or exc.__class__.__name__
        return _fallback(f"runner_contract:{reason}")


__all__ = ["B6RunnerResult", "run_deferred_b6_from_actual_split"]
