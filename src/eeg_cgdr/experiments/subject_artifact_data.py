"""Development-only SGEYESUB data surface for subject artifact latents.

This module deliberately reuses the frozen 25-fold SGEYESUB loader while
exposing every fold as development.  Block 1 is the only calibration and weak
target source.  Block 2 contributes observed EEG windows only; query EOG and
annotations never appear in the returned objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import yaml

from eeg_cgdr.experiments.sgeyesub_diffusion import (
    TrialWindowOrigin,
    trial_local_nonoverlap_windows,
)
from eeg_cgdr.experiments.sgeyesub_diffusion_runner import _prepare_fold
from eeg_cgdr.operators.artifact_context import ArtifactTransfer, fit_artifact_transfer


_CONTEXT_ROLES = (
    "population",
    "matching",
    "wrong_same_cell",
    "shuffled_same_cell_severity_stratum",
)


def _mapping(value: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    result = value.get(key)
    if not isinstance(result, Mapping):
        raise ValueError(f"missing mapping: {key}")
    return result


def _freeze(value: object, *, name: str, ndim: int | None = None) -> np.ndarray:
    result = np.ascontiguousarray(np.asarray(value))
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if result.dtype.kind not in "biuf" or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite numeric data")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class UnifiedDevelopmentFold:
    fold_id: str
    unified_fold_index: int
    original_partition: Literal["development", "evaluation"]
    original_partition_index: int
    study: str
    layout_id: str
    sampling_rate_hz: float
    training_recording_keys: tuple[str, ...]
    heldout_recording_keys: tuple[str, ...]
    partition: Literal["development"] = "development"


@dataclass(frozen=True)
class SubjectArtifactModelDimensions:
    eeg_channels: int
    eog_coordinates: int
    signal_length: int

    def __post_init__(self) -> None:
        if self.eeg_channels < 2 or self.eog_coordinates not in (2, 3):
            raise ValueError("model dimensions require multichannel EEG and 2/3 EOG")
        if self.signal_length < 1:
            raise ValueError("signal_length must be positive")


@dataclass(frozen=True)
class RuntimeArtifactContext:
    role: str
    context_id: str
    raw_transfer: np.ndarray
    full_transfer: np.ndarray
    normalized_transfer: np.ndarray
    transfer_scale: np.ndarray
    singular_values: np.ndarray
    rank: int
    projector: np.ndarray
    rho: float
    calibration_duration_seconds: float
    fit_recording_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.role not in _CONTEXT_ROLES:
            raise ValueError("unknown runtime artifact context role")
        raw = _freeze(self.raw_transfer, name="raw_transfer", ndim=2)
        full = _freeze(self.full_transfer, name="full_transfer", ndim=2)
        normalized = _freeze(
            self.normalized_transfer,
            name="normalized_transfer",
            ndim=2,
        )
        scale = _freeze(self.transfer_scale, name="transfer_scale", ndim=1)
        singular = _freeze(self.singular_values, name="singular_values", ndim=1)
        projector = _freeze(self.projector, name="projector", ndim=2)
        if raw.shape != full.shape or full.shape != normalized.shape:
            raise ValueError("runtime transfer shapes differ")
        if full.shape[1] not in (2, 3) or scale.shape != (full.shape[1],):
            raise ValueError("runtime context must retain all 2/3 EOG columns")
        if singular.shape != (full.shape[1],):
            raise ValueError("runtime singular values must retain all EOG entries")
        if projector.shape != (full.shape[0], full.shape[0]):
            raise ValueError("runtime projector shape differs from EEG channels")
        rank = int(self.rank)
        if isinstance(self.rank, bool) or rank != self.rank or not 1 <= rank <= full.shape[1]:
            raise ValueError("runtime retained rank is invalid")
        rho = float(self.rho)
        if not np.isfinite(rho) or not 0.0 <= rho <= 1.0:
            raise ValueError("runtime rho must lie in [0,1]")
        if not np.allclose(full, normalized * scale[None, :], atol=2e-10, rtol=2e-10):
            raise ValueError("runtime C/C_normalized/scale are inconsistent")
        if not self.context_id or not self.fit_recording_keys:
            raise ValueError("runtime context provenance is empty")
        for name, item in (
            ("raw_transfer", raw),
            ("full_transfer", full),
            ("normalized_transfer", normalized),
            ("transfer_scale", scale),
            ("singular_values", singular),
            ("projector", projector),
        ):
            object.__setattr__(self, name, item)
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "rho", rho)

    def model_kwargs(self) -> dict[str, object]:
        return {
            "full_transfer": self.full_transfer,
            "normalized_transfer": self.normalized_transfer,
            "transfer_scale": self.transfer_scale,
            "singular_values": self.singular_values,
            "rank": self.rank,
            "rho": self.rho,
            "calibration_duration_seconds": self.calibration_duration_seconds,
        }


@dataclass(frozen=True)
class OuterTrainingLatentNormalizer:
    mean: np.ndarray
    standard_deviation: np.ndarray
    training_recording_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        mean = _freeze(self.mean, name="latent mean", ndim=1)
        scale = _freeze(self.standard_deviation, name="latent scale", ndim=1)
        if mean.shape not in ((2,), (3,)) or scale.shape != mean.shape or np.any(scale <= 0):
            raise ValueError("latent normalizer must describe 2/3 positive coordinates")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "standard_deviation", scale)

    def transform(self, value: np.ndarray) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.ndim != 3 or array.shape[1] != self.mean.size:
            raise ValueError("latent array must have shape (N,E,L)")
        return (array - self.mean[None, :, None]) / self.standard_deviation[None, :, None]

    def inverse_transform(self, value: np.ndarray) -> np.ndarray:
        array = np.asarray(value, dtype=np.float64)
        if array.ndim != 3 or array.shape[1] != self.mean.size:
            raise ValueError("latent array must have shape (N,E,L)")
        return array * self.standard_deviation[None, :, None] + self.mean[None, :, None]


@dataclass(frozen=True)
class ArtifactLatentTrainingArrays:
    observed: np.ndarray
    standardized_artifact_latent: np.ndarray
    valid_time_mask: np.ndarray
    full_transfer: np.ndarray
    normalized_transfer: np.ndarray
    transfer_scale: np.ndarray
    singular_values: np.ndarray
    rank: np.ndarray
    rho: np.ndarray
    calibration_duration_seconds: np.ndarray
    channel_mask: np.ndarray
    recording_keys: tuple[str, ...]
    target_origins: tuple[TrialWindowOrigin, ...]
    artifact_origins: tuple[TrialWindowOrigin, ...]

    def __post_init__(self) -> None:
        observed = _freeze(self.observed, name="training observed", ndim=3)
        latent = _freeze(
            self.standardized_artifact_latent,
            name="training artifact latent",
            ndim=3,
        )
        count, channels, length = observed.shape
        eog = latent.shape[1]
        expected = {
            "valid_time_mask": (count, length),
            "full_transfer": (count, channels, eog),
            "normalized_transfer": (count, channels, eog),
            "transfer_scale": (count, eog),
            "singular_values": (count, eog),
            "rank": (count,),
            "rho": (count,),
            "calibration_duration_seconds": (count,),
            "channel_mask": (count, channels),
        }
        if latent.shape != (count, eog, length) or eog not in (2, 3):
            raise ValueError("training latent shape differs from observed")
        for name, shape in expected.items():
            value = _freeze(getattr(self, name), name=name)
            if value.shape != shape:
                raise ValueError(f"{name} shape differs from training count")
            object.__setattr__(self, name, value)
        if len(self.recording_keys) != count or len(self.target_origins) != count or len(self.artifact_origins) != count:
            raise ValueError("training provenance count differs from arrays")
        object.__setattr__(self, "observed", observed)
        object.__setattr__(self, "standardized_artifact_latent", latent)


@dataclass(frozen=True)
class SealedQueryWindows:
    recording_key: str
    observed: np.ndarray
    valid_time_mask: np.ndarray
    origins: tuple[TrialWindowOrigin, ...]
    annotations_sealed: Literal[True] = True

    def __post_init__(self) -> None:
        observed = _freeze(self.observed, name="sealed query observed", ndim=3)
        mask = _freeze(self.valid_time_mask, name="sealed query mask", ndim=2)
        if mask.shape != (observed.shape[0], observed.shape[2]) or len(self.origins) != observed.shape[0]:
            raise ValueError("sealed query window shapes differ")
        object.__setattr__(self, "observed", observed)
        object.__setattr__(self, "valid_time_mask", mask)


@dataclass(frozen=True)
class HeldoutSubjectArtifactRecord:
    recording_key: str
    matching: RuntimeArtifactContext
    wrong_same_cell: RuntimeArtifactContext
    shuffled_same_cell: RuntimeArtifactContext
    wrong_source_recording_key: str
    query: SealedQueryWindows

    def __post_init__(self) -> None:
        if self.query.recording_key != self.recording_key:
            raise ValueError("sealed query and held-out context keys differ")
        if self.matching.role != "matching":
            raise ValueError("matching runtime context has the wrong role")
        if self.wrong_same_cell.role != "wrong_same_cell":
            raise ValueError("wrong-source runtime context has the wrong role")
        if self.shuffled_same_cell.role != "shuffled_same_cell_severity_stratum":
            raise ValueError("shuffled runtime context has the wrong role")
        if self.wrong_source_recording_key not in self.wrong_same_cell.fit_recording_keys:
            raise ValueError("wrong-source provenance differs from its runtime context")


@dataclass(frozen=True)
class PreparedSubjectArtifactFold:
    fold: UnifiedDevelopmentFold
    model_dimensions: SubjectArtifactModelDimensions
    training: ArtifactLatentTrainingArrays
    latent_normalizer: OuterTrainingLatentNormalizer
    population_context: RuntimeArtifactContext
    heldout: Mapping[str, HeldoutSubjectArtifactRecord]


def _load_frozen_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    sge = _mapping(_mapping(config, "data"), "sgeyesub")
    path = Path(str(sge.get("frozen_fold_source", "")))
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("frozen SGEYESUB config is not a mapping")
    return value


def _unified_fold_route(frozen: Mapping[str, Any], index: int) -> tuple[str, int]:
    split = _mapping(frozen, "split")
    development = tuple(split.get("development_folds", ()))
    evaluation = tuple(split.get("evaluation_folds", ()))
    routes = tuple(("development", value) for value in range(len(development))) + tuple(
        ("evaluation", value) for value in range(len(evaluation))
    )
    if len(routes) != 25:
        raise ValueError("frozen SGEYESUB config must expose exactly 25 folds")
    if isinstance(index, bool) or not 0 <= int(index) < len(routes):
        raise ValueError("unified_fold_index must lie in [0,24]")
    return routes[int(index)]


def _loaded_record(prepared: Any, recording_key: str) -> Any:
    loaded = prepared.training.get(recording_key)
    if loaded is None:
        loaded = prepared.heldout.get(recording_key)
    if loaded is None:
        raise ValueError("unknown same-cell recording key")
    return loaded


def _eog_order(prepared: Any, recording_key: str) -> tuple[str, ...]:
    record = prepared.records[recording_key]
    layout = prepared.layouts[record.layout_id]
    available = tuple(layout.external_eog_labels)
    count = _loaded_record(prepared, recording_key).support.external_eog.shape[0]
    if count == 2 and {"HEOG", "VEOG"}.issubset(available):
        return ("HEOG", "VEOG")
    if count == 3 and len(available) == 3:
        return available
    raise ValueError("loaded support EOG count differs from the frozen layout")


def _calibration_samples(config: Mapping[str, Any], sampling_rate: float, available: int) -> tuple[int, float]:
    seconds = float(_mapping(config, "calibration")["calibration_duration_seconds"])
    samples = int(round(seconds * sampling_rate))
    if seconds <= 0 or samples < 4 or samples > available:
        raise ValueError("frozen calibration duration is unavailable in block 1")
    return samples, seconds


def _fit_subject_transfer(config: Mapping[str, Any], prepared: Any, recording_key: str, *, eog_override: np.ndarray | None = None, fit_id: str | None = None) -> ArtifactTransfer:
    loaded = _loaded_record(prepared, recording_key)
    eeg = prepared.normalizer.transform(loaded.support.eeg)
    eog = loaded.support.external_eog if eog_override is None else eog_override
    samples, _ = _calibration_samples(config, loaded.sampling_rate_hz, eeg.shape[1])
    order = _eog_order(prepared, recording_key)
    calibration = _mapping(config, "calibration")
    scope = "outer_training_only" if recording_key in prepared.fold.training_recording_keys else "support_only"
    return fit_artifact_transfer(
        eeg[:, :samples],
        eog[:, :samples],
        eeg_channel_order=loaded.p0_channel_labels,
        eog_input_order=order,
        eog_canonical_order=order,
        eog_polarity=(1.0,) * len(order),
        ridge_lambda=float(calibration["ridge_lambda"]),
        retained_rank=int(calibration["retained_rank"]),
        fit_scope=scope,
        fit_id=fit_id or recording_key,
    )


def _support_rho(config: Mapping[str, Any], transfer: ArtifactTransfer) -> float:
    """Map the support-only retained singular ratio to frozen reliability."""

    reliability = _mapping(
        _mapping(config, "calibration"),
        "support_only_reliability",
    )
    reference = float(reliability["reference_singular_ratio"])
    singular = np.asarray(transfer.singular_values, dtype=np.float64)
    if not np.isfinite(reference) or reference <= 0.0 or singular[0] <= 0.0:
        raise ValueError("support reliability reference/singular values are invalid")
    ratio = float(singular[transfer.rank - 1] / singular[0])
    return float(np.clip(ratio / reference, 0.0, 1.0))


def _runtime(transfer: ArtifactTransfer, *, role: str, context_id: str, rho: float, seconds: float, keys: Sequence[str]) -> RuntimeArtifactContext:
    return RuntimeArtifactContext(
        role=role,
        context_id=context_id,
        raw_transfer=transfer.raw_transfer_matrix,
        full_transfer=transfer.transfer_matrix,
        normalized_transfer=transfer.transfer_normalized,
        transfer_scale=transfer.transfer_scale,
        singular_values=transfer.singular_values,
        rank=transfer.rank,
        projector=transfer.projector,
        rho=rho,
        calibration_duration_seconds=seconds,
        fit_recording_keys=tuple(keys),
    )


def _artifact_window(eog: np.ndarray, origin: TrialWindowOrigin, samples_per_trial: int) -> np.ndarray:
    start = origin.trial_ordinal * samples_per_trial + origin.start_sample
    stop = origin.trial_ordinal * samples_per_trial + origin.stop_sample
    value = np.asarray(eog[:, start:stop], dtype=np.float64)
    if value.shape[1] != origin.stop_sample - origin.start_sample:
        raise ValueError("artifact origin lies outside support block 1")
    return value


def _severity_shuffle(eog: np.ndarray, labels: np.ndarray, samples: int) -> np.ndarray:
    result = np.array(eog, dtype=np.float64, copy=True)
    classes = np.rint(np.asarray(labels[:samples])).astype(int)
    changed = False
    for value in sorted(set(classes.tolist())):
        index = np.flatnonzero(classes == value)
        if index.size > 1:
            result[:, index] = result[:, np.roll(index, index.size // 2)]
            changed = True
    if not changed:
        result[:, :samples] = np.roll(result[:, :samples], max(1, samples // 3), axis=1)
    return result


def prepare_subject_artifact_fold(config: Mapping[str, Any], unified_fold_index: int) -> PreparedSubjectArtifactFold:
    """Prepare one of all 25 frozen folds as sealed development data."""

    sge = _mapping(_mapping(config, "data"), "sgeyesub")
    if sge.get("use_all_25_exact_cell_folds") is not True:
        raise ValueError("subject artifact development requires all 25 frozen folds")
    frozen = _load_frozen_config(config)
    original_partition, local_index = _unified_fold_route(frozen, unified_fold_index)
    prepared = _prepare_fold(frozen, original_partition, local_index)
    if set(prepared.training) & set(prepared.heldout) or any(
        value.query_annotations is not None for value in prepared.heldout.values()
    ):
        raise AssertionError("fold preparation leaked held-out identity or annotations")
    calibration_seconds = float(_mapping(config, "calibration")["calibration_duration_seconds"])
    training_transfer = {
        key: _fit_subject_transfer(config, prepared, key)
        for key in prepared.fold.training_recording_keys
    }
    # Population C is fitted only from concatenated same-cell outer-training
    # calibration blocks. Source IDs remain absent from model-visible arrays.
    eeg_parts: list[np.ndarray] = []
    eog_parts: list[np.ndarray] = []
    for key in prepared.fold.training_recording_keys:
        loaded = prepared.training[key]
        count, _ = _calibration_samples(config, loaded.sampling_rate_hz, loaded.support.eeg.shape[1])
        eeg_parts.append(prepared.normalizer.transform(loaded.support.eeg)[:, :count])
        source_eog = np.asarray(
            loaded.support.external_eog[:, :count],
            dtype=np.float64,
        )
        source_mean = source_eog.mean(axis=1, keepdims=True)
        source_scale = source_eog.std(axis=1, keepdims=True)
        if np.any(source_scale <= np.finfo(np.float64).eps):
            raise ValueError("outer-training population source has constant EOG")
        eog_parts.append((source_eog - source_mean) / source_scale)
    first_key = prepared.fold.training_recording_keys[0]
    first_loaded = prepared.training[first_key]
    order = _eog_order(prepared, first_key)
    calibration = _mapping(config, "calibration")
    population_transfer = fit_artifact_transfer(
        np.concatenate(eeg_parts, axis=1),
        np.concatenate(eog_parts, axis=1),
        eeg_channel_order=first_loaded.p0_channel_labels,
        eog_input_order=order,
        eog_canonical_order=order,
        eog_polarity=(1.0,) * len(order),
        ridge_lambda=float(calibration["ridge_lambda"]),
        retained_rank=int(calibration["retained_rank"]),
        fit_scope="outer_training_only",
        fit_id=f"{prepared.fold.fold_id}:population",
    )
    population = _runtime(
        population_transfer,
        role="population",
        context_id=f"{prepared.fold.fold_id}:population",
        rho=0.0,
        seconds=0.0,
        keys=prepared.fold.training_recording_keys,
    )

    raw_latent: list[np.ndarray] = []
    reconstructed_observed: list[np.ndarray] = []
    contexts: list[RuntimeArtifactContext] = []
    training_keys = set(prepared.fold.training_recording_keys)
    if any(
        key not in training_keys
        or prepared.pairs.target_origins[index].recording_key != key
        or prepared.pairs.artifact_origins[index].recording_key != key
        or prepared.pairs.target_origins[index] == prepared.pairs.artifact_origins[index]
        for index, key in enumerate(prepared.pairs.recording_keys)
    ):
        raise AssertionError("weak target/artifact origins escaped outer-training support")
    for index, key in enumerate(prepared.pairs.recording_keys):
        transfer = training_transfer[key]
        record = prepared.records[key]
        eog_window = _artifact_window(
            prepared.training[key].support.external_eog,
            prepared.pairs.artifact_origins[index],
            record.samples_per_trial,
        )
        latent = transfer.standardized_artifact_latent(eog_window, input_order=_eog_order(prepared, key))
        contamination = transfer.transfer_normalized @ latent
        raw_latent.append(latent)
        reconstructed_observed.append(
            np.asarray(prepared.pairs.weak_target[index], dtype=np.float64) + contamination
        )
        contexts.append(
            _runtime(
                transfer,
                role="matching",
                context_id=f"{key}:training_matching",
                rho=_support_rho(config, transfer),
                seconds=calibration_seconds,
                keys=(key,),
            )
        )
    latent_values = np.stack(raw_latent)
    latent_mean = latent_values.mean(axis=(0, 2), dtype=np.float64)
    latent_scale = latent_values.std(axis=(0, 2), dtype=np.float64)
    if np.any(latent_scale <= np.finfo(np.float64).eps):
        raise ValueError("outer-training artifact latent contains a constant coordinate")
    latent_normalizer = OuterTrainingLatentNormalizer(
        latent_mean,
        latent_scale,
        tuple(prepared.fold.training_recording_keys),
    )
    training = ArtifactLatentTrainingArrays(
        observed=np.stack(reconstructed_observed).astype(np.float32),
        standardized_artifact_latent=latent_normalizer.transform(latent_values).astype(np.float32),
        valid_time_mask=np.asarray(prepared.pairs.valid_time_mask, dtype=bool),
        full_transfer=np.stack([value.full_transfer for value in contexts]).astype(np.float32),
        normalized_transfer=np.stack([value.normalized_transfer for value in contexts]).astype(np.float32),
        transfer_scale=np.stack([value.transfer_scale for value in contexts]).astype(np.float32),
        singular_values=np.stack([value.singular_values for value in contexts]).astype(np.float32),
        rank=np.asarray([value.rank for value in contexts], dtype=np.int64),
        rho=np.asarray([value.rho for value in contexts], dtype=np.float32),
        calibration_duration_seconds=np.full(len(contexts), calibration_seconds, dtype=np.float32),
        channel_mask=np.ones((len(contexts), prepared.fold.eeg_channels), dtype=bool),
        recording_keys=tuple(prepared.pairs.recording_keys),
        target_origins=tuple(prepared.pairs.target_origins),
        artifact_origins=tuple(prepared.pairs.artifact_origins),
    )

    heldout: dict[str, HeldoutSubjectArtifactRecord] = {}
    wrong_key = sorted(prepared.fold.training_recording_keys)[0]
    wrong_transfer = training_transfer[wrong_key]
    for key in prepared.fold.heldout_recording_keys:
        loaded = prepared.heldout[key]
        if loaded.query is None or loaded.query_annotations is not None:
            raise AssertionError("held-out block2 must be signal-only and sealed")
        if (
            prepared.records[wrong_key].layout_id != prepared.records[key].layout_id
            or prepared.training[wrong_key].sampling_rate_hz
            != loaded.sampling_rate_hz
            or prepared.training[wrong_key].p0_channel_labels
            != loaded.p0_channel_labels
        ):
            raise AssertionError("wrong-source control crossed an exact compatibility cell")
        matching_transfer = _fit_subject_transfer(config, prepared, key)
        count, _ = _calibration_samples(config, loaded.sampling_rate_hz, loaded.support.eeg.shape[1])
        shuffled_eog = _severity_shuffle(
            loaded.support.external_eog,
            loaded.support.artifactclasses,
            count,
        )
        shuffled_transfer = _fit_subject_transfer(
            config,
            prepared,
            key,
            eog_override=shuffled_eog,
            fit_id=f"{key}:severity_shuffled",
        )
        rho = _support_rho(config, matching_transfer)
        query_windows = trial_local_nonoverlap_windows(
            prepared.normalizer.transform(loaded.query.eeg),
            samples_per_trial=prepared.records[key].samples_per_trial,
            sampling_rate_hz=loaded.sampling_rate_hz,
            recording_key=key,
        )
        heldout[key] = HeldoutSubjectArtifactRecord(
            recording_key=key,
            matching=_runtime(matching_transfer, role="matching", context_id=f"{key}:matching", rho=rho, seconds=calibration_seconds, keys=(key,)),
            wrong_same_cell=_runtime(wrong_transfer, role="wrong_same_cell", context_id=f"{key}:wrong:{wrong_key}", rho=rho, seconds=calibration_seconds, keys=(wrong_key,)),
            shuffled_same_cell=_runtime(shuffled_transfer, role="shuffled_same_cell_severity_stratum", context_id=f"{key}:shuffled", rho=rho, seconds=calibration_seconds, keys=(key,)),
            wrong_source_recording_key=wrong_key,
            query=SealedQueryWindows(
                recording_key=key,
                observed=query_windows.values.astype(np.float32),
                valid_time_mask=query_windows.valid_time_mask,
                origins=query_windows.origins,
            ),
        )
    eog_coordinates = next(iter(training_transfer.values())).transfer_matrix.shape[1]
    unified = UnifiedDevelopmentFold(
        fold_id=prepared.fold.fold_id,
        unified_fold_index=int(unified_fold_index),
        original_partition=original_partition,  # type: ignore[arg-type]
        original_partition_index=local_index,
        study=prepared.fold.study,
        layout_id=prepared.fold.layout_id,
        sampling_rate_hz=prepared.fold.sampling_rate_hz,
        training_recording_keys=tuple(prepared.fold.training_recording_keys),
        heldout_recording_keys=tuple(prepared.fold.heldout_recording_keys),
    )
    return PreparedSubjectArtifactFold(
        fold=unified,
        model_dimensions=SubjectArtifactModelDimensions(
            eeg_channels=prepared.fold.eeg_channels,
            eog_coordinates=eog_coordinates,
            signal_length=training.observed.shape[2],
        ),
        training=training,
        latent_normalizer=latent_normalizer,
        population_context=population,
        heldout=heldout,
    )


def validate_real_subject_artifact_inputs(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Load one real sealed representative from each frozen exact cell."""

    frozen = _load_frozen_config(config)
    split = _mapping(frozen, "split")
    entries = tuple(split.get("development_folds", ())) + tuple(split.get("evaluation_folds", ()))
    if len(entries) != 25:
        raise ValueError("real validator requires all 25 frozen fold entries")
    representative_indices: list[int] = []
    cells: set[tuple[object, ...]] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise ValueError("frozen fold entry must be a mapping")
        cell = (
            entry.get("study"),
            entry.get("layout_id"),
            entry.get("sampling_rate_hz"),
            entry.get("eeg_channels"),
        )
        if cell not in cells:
            cells.add(cell)
            representative_indices.append(index)
    rows: list[dict[str, object]] = []
    for index in representative_indices:
        prepared = prepare_subject_artifact_fold(config, index)
        query_windows = sum(value.query.observed.shape[0] for value in prepared.heldout.values())
        rho_values = [value.matching.rho for value in prepared.heldout.values()]
        rows.append(
            {
                "unified_fold_index": index,
                "fold_id": prepared.fold.fold_id,
                "study": prepared.fold.study,
                "layout_id": prepared.fold.layout_id,
                "training_stems": len(prepared.fold.training_recording_keys),
                "heldout_stems": len(prepared.fold.heldout_recording_keys),
                "eeg_channels": prepared.model_dimensions.eeg_channels,
                "eog_coordinates": prepared.model_dimensions.eog_coordinates,
                "training_windows": prepared.training.observed.shape[0],
                "query_windows": query_windows,
                "training_observed_shape": list(prepared.training.observed.shape),
                "training_latent_shape": list(
                    prepared.training.standardized_artifact_latent.shape
                ),
                "query_observed_shapes": {
                    key: list(value.query.observed.shape)
                    for key, value in prepared.heldout.items()
                },
                "support_rho_minimum": min(rho_values),
                "support_rho_maximum": max(rho_values),
                "annotations_sealed": all(value.query.annotations_sealed for value in prepared.heldout.values()),
                "training_heldout_disjoint": not bool(set(prepared.fold.training_recording_keys) & set(prepared.fold.heldout_recording_keys)),
            }
        )
    if len(rows) != len(cells) or not all(
        row["annotations_sealed"] and row["training_heldout_disjoint"] for row in rows
    ):
        raise AssertionError("real subject-artifact validation failed")
    all_rho = [
        float(row[field])
        for row in rows
        for field in ("support_rho_minimum", "support_rho_maximum")
    ]
    return {
        "status": "success_real_subject_artifact_inputs_sealed",
        "frozen_fold_count": 25,
        "representative_exact_cell_count": len(rows),
        "representatives": rows,
        "support_rho_range": [min(all_rho), max(all_rho)],
        "query_annotations_opened": False,
        "query_eog_used": False,
    }


__all__ = [
    "ArtifactLatentTrainingArrays",
    "HeldoutSubjectArtifactRecord",
    "OuterTrainingLatentNormalizer",
    "PreparedSubjectArtifactFold",
    "RuntimeArtifactContext",
    "SealedQueryWindows",
    "SubjectArtifactModelDimensions",
    "UnifiedDevelopmentFold",
    "prepare_subject_artifact_fold",
    "validate_real_subject_artifact_inputs",
]
