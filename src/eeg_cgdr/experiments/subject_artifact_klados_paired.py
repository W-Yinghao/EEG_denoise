"""Paired Klados source-record development data for the r3 calibration screen.

This is mechanism evidence only: the 54 release entries do not expose a
defensible participant mapping.  The builder uses sim01--sim30 for training
and sim31--sim36/sim44/sim45 for development, always splitting before
windowing and fitting every non-oracle transfer on query-disjoint support.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from eeg_cgdr.data.klados import KladosRecord, load_klados_records
from eeg_cgdr.data.mechanism import (
    KLADOS_DEVELOPMENT_RECORDS,
    KLADOS_NATIVE_CHANNEL_ORDER,
    KLADOS_TRAIN_RECORDS,
    KladosMechanismRecord,
    fit_channel_normalizer,
    prepare_mechanism_record,
    select_records,
    window_after_normalization,
)
from eeg_cgdr.experiments.sgeyesub_diffusion import TrialWindowOrigin
from eeg_cgdr.experiments.subject_artifact_data import (
    ArtifactLatentTrainingArrays,
    HeldoutSubjectArtifactRecord,
    OuterTrainingLatentNormalizer,
    PreparedSubjectArtifactFold,
    RuntimeArtifactContext,
    SealedQueryWindows,
    SubjectArtifactModelDimensions,
    UnifiedDevelopmentFold,
    _runtime,
    _support_rho,
)
from eeg_cgdr.operators.artifact_context import ArtifactTransfer, fit_artifact_transfer


EOG_ORDER = ("VEOG", "HEOG")


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise ValueError(f"missing mapping: {key}")
    return result


def _mechanism_config(subject_config: Mapping[str, Any]) -> Mapping[str, Any]:
    stage3_path = Path(
        str(_mapping(_mapping(subject_config, "data"), "klados_v4")["mechanism_config"])
    )
    stage3 = yaml.safe_load(stage3_path.read_text(encoding="utf-8"))
    deterministic_path = Path(str(stage3["matched_deterministic_config"]))
    deterministic = yaml.safe_load(deterministic_path.read_text(encoding="utf-8"))
    base_path = Path(str(deterministic["base_config"]))
    value = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("Klados mechanism base config is invalid")
    return value


def _loader_config(mechanism: Mapping[str, Any]) -> dict[str, Any]:
    raw = _mapping(mechanism, "klados")
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


def _fit_transfer(
    subject_config: Mapping[str, Any],
    eeg: np.ndarray,
    eog: np.ndarray,
    *,
    fit_scope: str,
    fit_id: str,
) -> ArtifactTransfer:
    calibration = _mapping(subject_config, "calibration")
    return fit_artifact_transfer(
        eeg,
        eog,
        eeg_channel_order=KLADOS_NATIVE_CHANNEL_ORDER,
        eog_input_order=EOG_ORDER,
        eog_canonical_order=EOG_ORDER,
        eog_polarity=(1.0, 1.0),
        ridge_lambda=float(calibration["ridge_lambda"]),
        retained_rank=int(calibration["retained_rank"]),
        fit_scope=fit_scope,  # type: ignore[arg-type]
        fit_id=fit_id,
    )


def _origins(record_id: int, count: int) -> tuple[TrialWindowOrigin, ...]:
    # Klados provides continuous source records, not original trial IDs.  The
    # origin labels are provenance only and never used to invent participants.
    return tuple(
        TrialWindowOrigin(
            recording_key=f"sim{record_id:02d}",
            trial_ordinal=index,
            start_sample=index * 512,
            stop_sample=(index + 1) * 512,
        )
        for index in range(count)
    )


@dataclass(frozen=True)
class KladosPairedTruth:
    mechanism: KladosMechanismRecord
    raw_query_eog: np.ndarray
    matching_transfer: ArtifactTransfer


@dataclass(frozen=True)
class PreparedKladosPaired:
    prepared: PreparedSubjectArtifactFold
    truth: Mapping[str, KladosPairedTruth]
    transfers: Mapping[str, ArtifactTransfer]
    population_transfer: ArtifactTransfer


def _raw_query_eog(mechanism: KladosMechanismRecord) -> np.ndarray:
    return (
        mechanism.eog_continuous
        * mechanism.eog_calibration_standard_deviation[:, None]
        + mechanism.eog_calibration_mean[:, None]
    )


def _raw_support_eog(mechanism: KladosMechanismRecord) -> np.ndarray:
    return (
        mechanism.calibration.eog
        * mechanism.eog_calibration_standard_deviation[:, None]
        + mechanism.eog_calibration_mean[:, None]
    )


def prepare_klados_paired(subject_config: Mapping[str, Any]) -> PreparedKladosPaired:
    mechanism_config = _mechanism_config(subject_config)
    records = load_klados_records(_loader_config(mechanism_config))
    normalizer = fit_channel_normalizer(records, KLADOS_TRAIN_RECORDS)
    source_rate = int(_mapping(mechanism_config, "klados")["source_sampling_rate"])
    preprocessing = _mapping(mechanism_config, "preprocessing")
    target_rate = int(preprocessing["target_sampling_rate"])
    window_samples = int(preprocessing["window_samples"])
    calibration_seconds = 10.0
    guard_seconds = 1.0
    selected_ids = KLADOS_TRAIN_RECORDS + KLADOS_DEVELOPMENT_RECORDS
    mechanisms = {
        record.record_id: prepare_mechanism_record(
            record,
            normalizer,
            source_rate=source_rate,
            target_rate=target_rate,
            window_samples=window_samples,
            calibration_seconds=calibration_seconds,
            guard_seconds=guard_seconds,
        )
        for record in select_records(records, selected_ids)
    }
    transfers = {
        f"sim{record_id:02d}": _fit_transfer(
            subject_config,
            mechanisms[record_id].calibration.eeg,
            _raw_support_eog(mechanisms[record_id]),
            fit_scope="support_only",
            fit_id=f"sim{record_id:02d}:support",
        )
        for record_id in selected_ids
    }
    population_transfer = _fit_transfer(
        subject_config,
        np.concatenate(
            [mechanisms[value].calibration.eeg for value in KLADOS_TRAIN_RECORDS],
            axis=1,
        ),
        np.concatenate(
            [_raw_support_eog(mechanisms[value]) for value in KLADOS_TRAIN_RECORDS],
            axis=1,
        ),
        fit_scope="outer_training_only",
        fit_id="sim01_sim30:population",
    )
    population = _runtime(
        population_transfer,
        role="population",
        context_id="klados:population",
        rho=0.0,
        seconds=0.0,
        keys=tuple(f"sim{value:02d}" for value in KLADOS_TRAIN_RECORDS),
    )

    physical_latents: list[np.ndarray] = []
    observed: list[np.ndarray] = []
    valid: list[np.ndarray] = []
    full: list[np.ndarray] = []
    normalized: list[np.ndarray] = []
    scale: list[np.ndarray] = []
    singular: list[np.ndarray] = []
    ranks: list[int] = []
    rhos: list[float] = []
    keys: list[str] = []
    origins: list[TrialWindowOrigin] = []
    for record_id in KLADOS_TRAIN_RECORDS:
        key = f"sim{record_id:02d}"
        mechanism = mechanisms[record_id]
        transfer = transfers[key]
        raw_eog = _raw_query_eog(mechanism)
        physical = transfer.standardized_artifact_latent(
            raw_eog, input_order=EOG_ORDER
        )
        physical_windows = window_after_normalization(
            physical, window_samples
        ).values
        count = mechanism.observed_windows.shape[0]
        physical_latents.append(physical_windows)
        observed.append(mechanism.observed_windows)
        valid.append(mechanism.valid_time_weight.astype(bool))
        full.extend([transfer.transfer_matrix] * count)
        normalized.extend([transfer.transfer_normalized] * count)
        scale.extend([transfer.transfer_scale] * count)
        singular.extend([transfer.singular_values] * count)
        ranks.extend([transfer.rank] * count)
        rhos.extend([_support_rho(subject_config, transfer)] * count)
        keys.extend([key] * count)
        origins.extend(_origins(record_id, count))
    physical_values = np.concatenate(physical_latents, axis=0)
    mean = physical_values.mean(axis=(0, 2), dtype=np.float64)
    std = physical_values.std(axis=(0, 2), dtype=np.float64)
    latent_normalizer = OuterTrainingLatentNormalizer(
        mean, std, tuple(f"sim{value:02d}" for value in KLADOS_TRAIN_RECORDS)
    )
    observed_values = np.concatenate(observed, axis=0).astype(np.float32)
    training = ArtifactLatentTrainingArrays(
        observed=observed_values,
        standardized_artifact_latent=latent_normalizer.transform(
            physical_values
        ).astype(np.float32),
        valid_time_mask=np.concatenate(valid, axis=0),
        full_transfer=np.stack(full).astype(np.float32),
        normalized_transfer=np.stack(normalized).astype(np.float32),
        transfer_scale=np.stack(scale).astype(np.float32),
        singular_values=np.stack(singular).astype(np.float32),
        rank=np.asarray(ranks, dtype=np.int64),
        rho=np.asarray(rhos, dtype=np.float32),
        calibration_duration_seconds=np.full(
            len(keys), calibration_seconds, dtype=np.float32
        ),
        channel_mask=np.ones(
            (len(keys), len(KLADOS_NATIVE_CHANNEL_ORDER)), dtype=bool
        ),
        recording_keys=tuple(keys),
        target_origins=tuple(origins),
        artifact_origins=tuple(origins),
    )

    wrong_key = "sim01"
    wrong_transfer = transfers[wrong_key]
    heldout: dict[str, HeldoutSubjectArtifactRecord] = {}
    truth: dict[str, KladosPairedTruth] = {}
    for record_id in KLADOS_DEVELOPMENT_RECORDS:
        key = f"sim{record_id:02d}"
        mechanism = mechanisms[record_id]
        transfer = transfers[key]
        shuffled_eog = np.roll(
            _raw_support_eog(mechanism),
            shift=mechanism.calibration.eog.shape[1] // 2,
            axis=1,
        )
        shuffled = _fit_transfer(
            subject_config,
            mechanism.calibration.eeg,
            shuffled_eog,
            fit_scope="support_only",
            fit_id=f"{key}:support_shuffled",
        )
        rho = _support_rho(subject_config, transfer)
        heldout[key] = HeldoutSubjectArtifactRecord(
            recording_key=key,
            matching=_runtime(
                transfer,
                role="matching",
                context_id=f"{key}:matching",
                rho=rho,
                seconds=calibration_seconds,
                keys=(key,),
            ),
            wrong_same_cell=_runtime(
                wrong_transfer,
                role="wrong_same_cell",
                context_id=f"{key}:wrong:{wrong_key}",
                rho=rho,
                seconds=calibration_seconds,
                keys=(wrong_key,),
            ),
            shuffled_same_cell=_runtime(
                shuffled,
                role="shuffled_same_cell_severity_stratum",
                context_id=f"{key}:shuffled_support_only",
                rho=rho,
                seconds=calibration_seconds,
                keys=(key,),
            ),
            wrong_source_recording_key=wrong_key,
            query=SealedQueryWindows(
                recording_key=key,
                observed=mechanism.observed_windows.astype(np.float32),
                valid_time_mask=mechanism.valid_time_weight.astype(bool),
                origins=_origins(record_id, mechanism.observed_windows.shape[0]),
            ),
        )
        truth[key] = KladosPairedTruth(
            mechanism=mechanism,
            raw_query_eog=_raw_query_eog(mechanism),
            matching_transfer=transfer,
        )
    prepared = PreparedSubjectArtifactFold(
        fold=UnifiedDevelopmentFold(
            fold_id="klados_source_record_mechanism_development",
            unified_fold_index=0,
            original_partition="development",
            original_partition_index=0,
            study="klados_v4_source_records",
            layout_id="19ch_native_linked_mastoid_256hz",
            sampling_rate_hz=float(target_rate),
            training_recording_keys=tuple(
                f"sim{value:02d}" for value in KLADOS_TRAIN_RECORDS
            ),
            heldout_recording_keys=tuple(
                f"sim{value:02d}" for value in KLADOS_DEVELOPMENT_RECORDS
            ),
        ),
        model_dimensions=SubjectArtifactModelDimensions(
            eeg_channels=len(KLADOS_NATIVE_CHANNEL_ORDER),
            eog_coordinates=2,
            signal_length=window_samples,
        ),
        training=training,
        latent_normalizer=latent_normalizer,
        population_context=population,
        heldout=heldout,
    )
    return PreparedKladosPaired(
        prepared=prepared,
        truth=truth,
        transfers=transfers,
        population_transfer=population_transfer,
    )


__all__ = ["EOG_ORDER", "KladosPairedTruth", "PreparedKladosPaired", "prepare_klados_paired"]
