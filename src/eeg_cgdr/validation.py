"""One aggregate CPU validation for the real CGDR data path."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
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
from .data.eye_bci_full import read_eye_bci_record, target_for
from .experiments.klados import calibration_batch, prepare_query
from .inference.states import (
    matched_population_and_context_states,
    population_state_only,
)
from .operators.p0 import CalibrationBatch, P0Config, fit_p0


DEFAULT_CONFIG = Path("configs/cgdr/p0_klados_source_fold.yaml")
DEFAULT_EYE_CONFIG = Path("configs/cgdr/p0_eye_bci_fold00.yaml")


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


def _read_split_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
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
    if not rows or set(rows[0]) != required:
        raise AssertionError(f"unexpected split manifest schema: {path}")
    return rows


def _klados_split_contract(config: dict[str, Any]) -> dict[str, Any]:
    rows = _read_split_rows(Path(config["klados"]["split_manifest"]))
    calibration = [row for row in rows if row["status"] == "held_out_calibration"]
    query = [row for row in rows if row["status"] == "held_out_query"]
    if len(calibration) != 1 or len(query) != 1 or len(rows) != 2:
        raise AssertionError("Klados source-fold manifest must contain one support/query pair")
    calibration_row, query_row = calibration[0], query[0]
    if calibration_row["record"] != query_row["record"]:
        raise AssertionError("Klados frozen source record differs across support/query")
    if not calibration_row["record"].startswith("sim"):
        raise AssertionError("Klados manifest record must use the simNN source ID")
    if calibration_row["sampling_rate"] != query_row["sampling_rate"]:
        raise AssertionError("Klados support/query sampling rates differ")
    return {
        "rows": rows,
        "record": calibration_row["record"],
        "record_id": int(calibration_row["record"].removeprefix("sim")),
        "sampling_rate": int(float(calibration_row["sampling_rate"])),
        "calibration_start": float(calibration_row["calibration_start"]),
        "calibration_end": float(calibration_row["calibration_end"]),
        "query_start": float(query_row["query_start"]),
        "query_end": float(query_row["query_end"]),
    }


def _validate_frozen_splits(
    klados_config: dict[str, Any], eye_config_path: Path
) -> dict[str, Any]:
    klados = klados_config["klados"]
    contract = _klados_split_contract(klados_config)
    if contract["record"] != "sim45":
        raise AssertionError("Klados frozen source fold is not sim45")
    calibration_end = float(contract["calibration_end"])
    query_start = float(contract["query_start"])
    if calibration_end + float(klados["guard_seconds"]) > query_start:
        raise AssertionError("Klados frozen calibration/query intervals overlap their guard")
    if (
        calibration_end != max(float(value) for value in klados["calibration_seconds"])
    ):
        raise AssertionError("Klados config diverges from the frozen split manifest")

    eye_config = yaml.safe_load(eye_config_path.read_text(encoding="utf-8"))
    eye = eye_config["eye_bci"]
    train = set(eye["training_participants"])
    validation = set(eye["validation_participants"])
    test = set(eye["test_participants"])
    if train & validation or train & test or validation & test:
        raise AssertionError("Eye-BCI participant outer splits overlap")
    if not train or not validation or not test:
        raise AssertionError("Eye-BCI participant outer split is incomplete")
    eye_calibration_end = float(eye["calibration_end_seconds"])
    eye_query_start = float(eye["query_start_seconds"])
    if eye_calibration_end + float(eye["guard_seconds"]) > eye_query_start:
        raise AssertionError("Eye-BCI calibration/query intervals overlap their guard")
    return {
        "klados": {
            "outer_unit": "source_record_only_participant_mapping_blocked",
            "support_query_rows": len(contract["rows"]),
            "support_query_time_disjoint": True,
        },
        "eye_bci": {
            "training_participants": len(train),
            "validation_participants": len(validation),
            "test_participants": len(test),
            "participant_disjoint": True,
            "heldout_support_query_time_disjoint": True,
        },
    }


def _validate_reference_invariance() -> dict[str, float]:
    rng = np.random.default_rng(20260801)
    samples = 1600
    eog = rng.normal(size=(2, samples))
    transfer = rng.normal(size=(6, 2))
    eeg = transfer @ eog + 0.05 * rng.normal(size=(6, samples))
    transform = np.asarray([[1.25, -0.4], [0.3, 0.9]], dtype=np.float64)
    config = P0Config(
        target_rank=2,
        # Exact isotropic ridge is intentionally coordinate-dependent under a
        # general non-orthogonal EOG transform.  The identifiable CE
        # reparameterization check is therefore the unregularized OLS endpoint.
        ridge_lambda=0.0,
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
    split_contract = _klados_split_contract(config)
    held_out = records[int(split_contract["record_id"]) - 1]
    source_rate = int(split_contract["sampling_rate"])
    target_rate = int(config["preprocessing"]["target_sampling_rate"])
    calibration_end = float(split_contract["calibration_end"])
    query_start = float(split_contract["query_start"])
    query_end = float(split_contract["query_end"])
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

    population_base = float(config["observation"]["population_precision"])
    context_base = float(config["observation"]["context_precision"])
    energy_scale = float(config["observation"]["energy_scale"])
    if population_base != context_base:
        raise AssertionError("E0 and EC must use the same base precision")
    endpoint_observation = torch.from_numpy(
        query.contaminated[:2].astype(np.float32, copy=False)
    )
    endpoint_valid_time_mask = (
        torch.arange(endpoint_observation.shape[-1])[None, :]
        < torch.as_tensor(query.valid_samples[:2])[:, None]
    )
    endpoint_attenuation = torch.tensor([0.0, 1.0], dtype=torch.float32)
    pop_only = population_state_only(
        endpoint_observation,
        attenuation=endpoint_attenuation,
        base_precision=population_base,
        energy_scale=energy_scale,
        valid_time_mask=endpoint_valid_time_mask,
    )
    matched_pop, endpoint_context = matched_population_and_context_states(
        endpoint_observation,
        attenuation=endpoint_attenuation,
        projector=projector,
        base_precision=context_base,
        energy_scale=energy_scale,
        valid_time_mask=endpoint_valid_time_mask,
    )
    channels = endpoint_observation.shape[1]
    identity = torch.eye(channels, dtype=endpoint_observation.dtype)
    projection_tensor = torch.as_tensor(projector, dtype=endpoint_observation.dtype)
    expected_pop = population_base * identity
    expected_a_zero = context_base * (identity - projection_tensor)
    expected_a_one = context_base * identity
    if not torch.allclose(
        pop_only.precision,
        expected_pop.expand(2, -1, -1),
        atol=1.0e-6,
        rtol=1.0e-6,
    ):
        raise AssertionError("POP precision is not fixed isotropic precision")
    if not torch.equal(matched_pop.precision, pop_only.precision):
        raise AssertionError("matched E0 differs from direct POP")
    if (
        matched_pop.energy_scale != energy_scale
        or endpoint_context.energy_scale != energy_scale
    ):
        raise AssertionError("E0 and EC energy scales differ")
    if not torch.allclose(
        endpoint_context.precision[0], expected_a_zero, atol=1.0e-6, rtol=1.0e-6
    ):
        raise AssertionError("a=0 did not preserve complement precision Q")
    if not torch.allclose(
        endpoint_context.precision[1], expected_a_one, atol=1.0e-6, rtol=1.0e-6
    ):
        raise AssertionError("a=1 did not restore isotropic precision I")
    alternate_pop = population_state_only(
        endpoint_observation,
        attenuation=torch.tensor([0.2, 0.8], dtype=torch.float32),
        base_precision=population_base,
        energy_scale=energy_scale,
        valid_time_mask=endpoint_valid_time_mask,
    )
    if not torch.equal(alternate_pop.precision, pop_only.precision):
        raise AssertionError("POP precision incorrectly depends on subspace attenuation")
    try:
        population_state_only(
            endpoint_observation,
            attenuation=torch.tensor([-0.01, 1.01], dtype=torch.float32),
            base_precision=population_base,
            energy_scale=energy_scale,
            valid_time_mask=endpoint_valid_time_mask,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range attenuation was accepted")

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
            "precision_endpoints": {
                "population": "base*w*I independent of calibrated subspace attenuation",
                "a_zero": "base*w*Q",
                "a_one": "base*w*I",
                "base_scales_matched": True,
                "energy_scale_matched": energy_scale,
                "real_query_windows_checked": 2,
            },
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
    full_loader_records = [
        read_eye_bci_record(
            root,
            target_for(participant, "Sess01"),
            seconds=5.0,
        )
        for participant in ("S01", "S02", "S04")
    ]
    for record in full_loader_records:
        if (
            record.eeg.shape[0] != 62
            or record.eeg.shape[1] != record.heo.size
            or not all(
                np.isfinite(value).all()
                for value in (
                    record.eeg,
                    record.heo,
                    record.triggers,
                    record.cues,
                    record.blinks,
                )
            )
        ):
            raise AssertionError(
                f"Eye-BCI full loader failed aligned finite streams: {record.participant}"
            )
    return {
        "target_count": 2,
        "participant_session_keys": sorted("/".join(key) for key in keys),
        "directory_scan": False,
        "max_rows_per_target": max_rows,
        "sampling_rate_hz": [record.sampling_rate_hz for record in records],
        "shapes": [list(record.shape) for record in records],
        "finite": True,
        "signal_units": "unknown_not_encoded_in_csv",
        "full_loader_event_schema": "numeric_or_stable_categorical_codes",
        "full_loader_participants": [record.participant for record in full_loader_records],
        "full_loader_stream_alignment": True,
    }


def validate_real_cpu_path(
    *,
    config_path: Path = DEFAULT_CONFIG,
    eye_bci_root: Path = EYE_BCI_ROOT,
    eye_bci_targets: tuple[EyeBciTarget, ...] = DEFAULT_EYE_BCI_TARGETS,
    eye_bci_max_rows: int = 4096,
    eye_config_path: Path = DEFAULT_EYE_CONFIG,
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
        "frozen_splits": _validate_frozen_splits(config, eye_config_path),
        "p0_reference_reparameterization": _validate_reference_invariance(),
    }
    return result
