"""One aggregate CPU validation for the real CGDR data path."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .data.eegdenoise import load_clean_prior_split
from .data.eye_bci import (
    DEFAULT_EYE_BCI_TARGETS,
    EYE_BCI_ROOT,
    EYE_BCI_SCALP_CHANNELS,
    EyeBciTarget,
    read_default_eye_bci_targets,
)
from .data.klados import load_klados_records
from .experiments.klados import calibration_batch, prepare_query
from .operators.p0 import CalibrationBatch, P0Config, fit_p0


DEFAULT_CONFIG = Path("configs/cgdr/p0_klados_source_fold.yaml")


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


def _p0_config(raw: dict[str, Any], *, validation_replicates: int = 32) -> P0Config:
    return P0Config(
        target_rank=int(raw["target_rank"]),
        ridge_lambda=float(raw["ridge_lambda"]),
        maximum_reference_condition=float(raw["maximum_reference_condition"]),
        minimum_singular_ratio=float(raw["minimum_singular_ratio"]),
        minimum_movement_coverage=float(raw["minimum_movement_coverage"]),
        bootstrap_replicates=validation_replicates,
        bootstrap_block_samples=int(raw["bootstrap_block_samples"]),
        minimum_bootstrap_success=float(raw["minimum_bootstrap_success"]),
        maximum_bootstrap_median_distance=float(raw["maximum_bootstrap_median_distance"]),
        maximum_bootstrap_q90_distance=float(raw["maximum_bootstrap_q90_distance"]),
    )


def _validate_reference_invariance() -> dict[str, float]:
    rng = np.random.default_rng(20260801)
    samples = 1600
    eog = rng.normal(size=(2, samples))
    transfer = rng.normal(size=(6, 2))
    eeg = transfer @ eog + 0.05 * rng.normal(size=(6, samples))
    transform = np.asarray([[1.25, -0.4], [0.3, 0.9]], dtype=np.float64)
    config = P0Config(
        target_rank=2,
        bootstrap_replicates=8,
        bootstrap_block_samples=200,
        minimum_bootstrap_success=0.0,
        maximum_bootstrap_median_distance=float("inf"),
        maximum_bootstrap_q90_distance=float("inf"),
        minimum_movement_coverage=0.0,
    )
    original = fit_p0(
        CalibrationBatch(eeg, eog, "fixture", "original", 200.0),
        config,
        movement_threshold=0.0,
    )
    transformed = fit_p0(
        CalibrationBatch(eeg, transform @ eog, "fixture", "transformed", 200.0),
        config,
        movement_threshold=0.0,
    )
    if original.transfer is None or transformed.transfer is None:
        raise AssertionError("reference-invariance fixture was unexpectedly ineligible")
    prediction_error = float(
        np.linalg.norm(
            original.transfer.predicted_contamination
            - transformed.transfer.predicted_contamination
        )
        / np.linalg.norm(original.transfer.predicted_contamination)
    )
    projector_error = float(
        np.linalg.norm(original.transfer.projector - transformed.transfer.projector)
    )
    if prediction_error > 1.0e-10 or projector_error > 1.0e-10:
        raise AssertionError("P0 prediction/projector changed under invertible EOG coordinates")
    return {
        "prediction_relative_error": prediction_error,
        "projector_frobenius_error": projector_error,
    }


def _validate_klados(config: dict[str, Any]) -> dict[str, Any]:
    records = load_klados_records(_loader_config(config))
    if len(records) != 54:
        raise AssertionError(f"Klados loader returned {len(records)} records")
    for record in records:
        if (
            record.clean.shape != record.contaminated.shape
            or record.clean.shape[0] != 19
            or record.veog.size != record.samples
            or record.heog.size != record.samples
            or not all(
                np.isfinite(value).all()
                for value in (record.clean, record.contaminated, record.veog, record.heog)
            )
        ):
            raise AssertionError(f"real Klados record failed shape/finite: sim{record.record_id}")

    klados = config["klados"]
    held_out = records[int(klados["held_out_record"]) - 1]
    source_rate = int(klados["sampling_rate"])
    target_rate = int(config["preprocessing"]["target_sampling_rate"])
    calibration_end = max(float(value) for value in klados["calibration_seconds"])
    query_start = float(klados["query_start_seconds"])
    query_end = float(klados["query_end_seconds"])
    if calibration_end + float(klados["guard_seconds"]) > query_start:
        raise AssertionError("Klados calibration/query guard is not disjoint")
    if query_end * source_rate > held_out.samples + 1:
        raise AssertionError("Klados query extends beyond the source record")

    query = prepare_query(
        held_out,
        source_rate=source_rate,
        target_rate=target_rate,
        query_start_seconds=query_start,
        query_end_seconds=query_end,
        window_samples=int(config["preprocessing"]["window_samples"]),
        attenuation_scale=float(config["observation"]["attenuation_scale"]),
    )
    if int(query.valid_samples.sum()) < int((query_end - query_start) * target_rate) - 2:
        raise AssertionError("query windowing silently dropped a material tail")
    if not all(
        np.isfinite(value).all()
        for value in (query.clean, query.contaminated, query.eog)
    ):
        raise AssertionError("Klados query preprocessing produced non-finite values")

    support = calibration_batch(
        held_out,
        duration_seconds=calibration_end,
        source_rate=source_rate,
        target_rate=target_rate,
        source_label=f"sim{held_out.record_id}",
    )
    p0_config = _p0_config(config["p0"])
    outcome = fit_p0(
        support,
        p0_config,
        movement_threshold=float(config["p0"]["movement_threshold"]),
    )
    if outcome.status != "eligible" or outcome.transfer is None:
        raise AssertionError(f"real Klados P0 was ineligible: {outcome.reasons}")
    projector = outcome.transfer.projector
    symmetry = float(np.linalg.norm(projector - projector.T))
    idempotence = float(np.linalg.norm(projector @ projector - projector))
    if symmetry > 1.0e-10 or idempotence > 1.0e-10:
        raise AssertionError("P0 projector is not symmetric/idempotent")

    degenerate = CalibrationBatch(
        eeg=support.eeg[:, :512],
        eog=np.ones((2, 512), dtype=np.float64),
        participant="unresolved",
        source_record="constant_reference",
        sampling_rate=float(target_rate),
    )
    fallback = fit_p0(degenerate, p0_config, movement_threshold=1.0)
    if fallback.status != "ineligible" or fallback.fallback != "POP":
        raise AssertionError("degenerate calibration does not explicitly return POP")

    artifact = held_out.contaminated - held_out.clean
    eog = np.stack([held_out.veog, held_out.heog], axis=0)
    oracle_transfer = artifact @ eog.T @ np.linalg.pinv(eog @ eog.T)
    residual = float(
        np.linalg.norm(artifact - oracle_transfer @ eog)
        / max(np.linalg.norm(artifact), 1.0e-12)
    )
    if residual > 1.0e-6:
        raise AssertionError("Klados paired mixture recipe did not close")
    return {
        "records": len(records),
        "channels": 19,
        "sampling_rate_hz": source_rate,
        "signal_units": "unknown",
        "participant_mapping": "blocked_not_guessed",
        "outer_unit": "held_out_source_record_sim45",
        "calibration_seconds": calibration_end,
        "guard_seconds": float(klados["guard_seconds"]),
        "query_seconds": query_end - query_start,
        "query_windows": int(query.contaminated.shape[0]),
        "all_query_samples_retained_with_padding_mask": True,
        "calibration_query_disjoint": True,
        "paired_mixture_relative_residual": residual,
        "p0": {
            "status": outcome.status,
            "rank": outcome.transfer.rank,
            "projector_symmetry_error": symmetry,
            "projector_idempotence_error": idempotence,
            "bootstrap_stability": outcome.transfer.diagnostics,
            "degenerate_fallback": fallback.fallback,
        },
    }


def _validate_clean_prior(config: dict[str, Any]) -> dict[str, Any]:
    raw = config["clean_prior_data"]
    split = load_clean_prior_split(
        raw["path"],
        validation_fraction=float(raw["validation_fraction"]),
        seed=int(config["seed"]),
    )
    expected = int(raw["clean_epochs"])
    if split.train.shape[0] + split.validation.shape[0] != expected:
        raise AssertionError("clean-prior split omitted real epochs")
    if split.train.shape[1:] != (512,) or split.validation.shape[1:] != (512,):
        raise AssertionError("clean-prior epoch shape mismatch")
    if not np.isfinite(split.train).all() or not np.isfinite(split.validation).all():
        raise AssertionError("clean-prior preprocessing produced non-finite values")
    return {
        "dataset": "EEGdenoiseNet clean EEG",
        "train_epochs": int(split.train.shape[0]),
        "validation_epochs": int(split.validation.shape[0]),
        "sampling_rate_hz": split.sampling_rate,
        "normalization_fit": "population_prior_train_only",
        "held_out_klados_or_eye_identity_visible": False,
    }


def _validate_eye_bci(
    root: Path,
    targets: tuple[EyeBciTarget, ...],
    max_rows: int,
) -> dict[str, Any]:
    records = read_default_eye_bci_targets(root=root, targets=targets, max_rows=max_rows)
    if len(EYE_BCI_SCALP_CHANNELS) != 62 or len(records) != 2:
        raise AssertionError("Eye-BCI target/montage count mismatch")
    keys = {(record.participant_id, record.session_id) for record in records}
    if len(keys) != 2:
        raise AssertionError("Eye-BCI validation records are not participant/session independent")
    for record in records:
        if record.shape != (62, record.rows_read) or not record.finite:
            raise AssertionError(f"Eye-BCI shape/finite failure: {record.relative_path}")
        if record.signal_units != "unknown_not_encoded_in_csv":
            raise AssertionError("Eye-BCI units were inferred without evidence")
    return {
        "target_count": 2,
        "participant_session_keys": sorted("/".join(key) for key in keys),
        "directory_scan": False,
        "max_rows_per_target": max_rows,
        "sampling_rate_hz": [record.sampling_rate_hz for record in records],
        "shapes": [list(record.shape) for record in records],
        "finite": True,
        "signal_units": "unknown_not_encoded_in_csv",
    }


def validate_real_cpu_path(
    *,
    config_path: Path = DEFAULT_CONFIG,
    eye_bci_root: Path = EYE_BCI_ROOT,
    eye_bci_targets: tuple[EyeBciTarget, ...] = DEFAULT_EYE_BCI_TARGETS,
    eye_bci_max_rows: int = 4096,
) -> dict[str, Any]:
    """Run all algebra and bounded real-record checks in one Slurm job."""
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    result = {
        "status": "passed",
        "harness_level": 1,
        "hashes_computed": False,
        "clean_prior": _validate_clean_prior(config),
        "klados": _validate_klados(config),
        "eye_bci": _validate_eye_bci(eye_bci_root, eye_bci_targets, eye_bci_max_rows),
        "p0_reference_reparameterization": _validate_reference_invariance(),
    }
    return result
