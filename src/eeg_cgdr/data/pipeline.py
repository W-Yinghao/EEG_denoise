"""Leakage-resistant Klados grouping, normalization, and windowing.

The ordering in :func:`build_klados_pipeline` is deliberate: source records are
first assigned to immutable participant groups, normalization is then fitted on
the unique outer-training clean records, and only then are overlapping windows
created.  Window overlap therefore cannot alter a split or give a record extra
weight in the normalization statistics.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence, Union

import numpy as np

from .klados import KladosRecord


SplitName = Literal["train", "validation", "test"]
ContextRole = Literal["population", "calibration", "query"]


@dataclass(frozen=True)
class GroupedKladosSource:
    """One whole source record after participant-level split assignment."""

    participant_id: str
    session_id: str
    source_record: str
    split: SplitName
    role: ContextRole
    start_sample: int
    end_sample: int
    sampling_rate: float
    record: KladosRecord

    @property
    def context_id(self) -> str:
        return (
            f"{self.participant_id}/{self.session_id}/{self.source_record}:"
            f"{self.start_sample}-{self.end_sample}"
        )


@dataclass(frozen=True)
class CleanNormalizer:
    """Per-channel EEG statistics fitted only on outer-training clean EEG."""

    mean: np.ndarray
    scale: np.ndarray
    fit_participant_ids: tuple[str, ...]
    fit_source_records: tuple[str, ...]
    fit_sample_count: int
    fit_semantics: Literal["outer_train_clean_only"] = "outer_train_clean_only"

    def transform(self, eeg: np.ndarray) -> np.ndarray:
        value = np.asarray(eeg, dtype=np.float64)
        if value.ndim != 2 or value.shape[0] != self.mean.shape[0]:
            raise ValueError(
                f"EEG channel mismatch: value={value.shape}, normalizer={self.mean.shape}"
            )
        if not np.isfinite(value).all():
            raise ValueError("cannot normalize non-finite EEG")
        return (value - self.mean) / self.scale


@dataclass(frozen=True)
class WindowKey:
    participant_id: str
    session_id: str
    source_record: str
    split: SplitName
    role: ContextRole
    start_sample: int
    end_sample: int

    @property
    def context_id(self) -> str:
        return (
            f"{self.participant_id}/{self.session_id}/{self.source_record}:"
            f"{self.start_sample}-{self.end_sample}"
        )


@dataclass(frozen=True)
class PopulationWindow:
    """Supervised outer-train/validation window."""

    key: WindowKey
    contaminated_eeg: np.ndarray
    clean_target: np.ndarray
    external_reference: np.ndarray


@dataclass(frozen=True)
class CalibrationWindow:
    """Held-out support window; intentionally has no clean-target field."""

    key: WindowKey
    eeg: np.ndarray
    external_reference: np.ndarray


@dataclass(frozen=True)
class QueryWindow:
    """Held-out evaluation window, never accepted by a calibration fit API."""

    key: WindowKey
    contaminated_eeg: np.ndarray
    clean_target: np.ndarray
    external_reference: np.ndarray


KladosWindow = Union[PopulationWindow, CalibrationWindow, QueryWindow]


@dataclass(frozen=True)
class KladosPipeline:
    sources: tuple[GroupedKladosSource, ...]
    normalizer: CleanNormalizer
    windows: tuple[KladosWindow, ...]


def _record_number(value: str) -> int:
    if not value.startswith("sim") or not value[3:].isdigit():
        raise ValueError(f"invalid Klados record name in split manifest: {value!r}")
    return int(value[3:])


def _interval_from_row(
    row: dict[str, str], role: ContextRole, samples: int, sampling_rate: float
) -> tuple[int, int]:
    if role == "population":
        return 0, samples
    prefix = "calibration" if role == "calibration" else "query"
    start_text = str(row.get(f"{prefix}_start") or "").strip()
    end_text = str(row.get(f"{prefix}_end") or "").strip()
    if not start_text or not end_text:
        raise ValueError(f"{role} row has no declared interval")
    start = int(round(float(start_text) * sampling_rate))
    end = int(round(float(end_text) * sampling_rate))
    if start < 0 or end <= start or end > samples:
        raise ValueError(
            f"invalid {role} interval {start}:{end} for a {samples}-sample record"
        )
    return start, end


def _role(status: str) -> ContextRole:
    mapping: dict[str, ContextRole] = {
        "population_source": "population",
        "held_out_calibration": "calibration",
        "held_out_query": "query",
    }
    try:
        return mapping[status]
    except KeyError as exc:
        raise ValueError(f"unknown split-manifest status: {status!r}") from exc


def assert_calibration_query_disjoint(
    sources: Sequence[GroupedKladosSource],
) -> None:
    """Reject shared IDs or overlapping sample intervals within one source."""

    calibration = [source for source in sources if source.role == "calibration"]
    queries = [source for source in sources if source.role == "query"]
    calibration_ids = {source.context_id for source in calibration}
    query_ids = {source.context_id for source in queries}
    duplicate_ids = calibration_ids & query_ids
    if duplicate_ids:
        raise ValueError(f"calibration/query context IDs overlap: {sorted(duplicate_ids)}")
    for support in calibration:
        for query in queries:
            if (
                support.participant_id == query.participant_id
                and support.source_record == query.source_record
                and max(support.start_sample, query.start_sample)
                < min(support.end_sample, query.end_sample)
            ):
                raise ValueError(
                    "calibration/query sample intervals overlap for "
                    f"{support.participant_id}/{support.source_record}"
                )


def group_klados_records(
    records: Sequence[KladosRecord],
    split_manifest: Path,
) -> tuple[GroupedKladosSource, ...]:
    """Attach the frozen participant/source split before any window exists."""

    by_number = {record.record_id: record for record in records}
    if len(by_number) != len(records):
        raise ValueError("duplicate Klados record IDs")
    with split_manifest.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("empty Klados split manifest")

    grouped: list[GroupedKladosSource] = []
    seen_records: set[int] = set()
    participant_splits: dict[str, SplitName] = {}
    for row in rows:
        record_name = str(row.get("record") or "")
        record_number = _record_number(record_name)
        if record_number in seen_records:
            raise ValueError(f"record occurs more than once in split manifest: {record_name}")
        try:
            record = by_number[record_number]
        except KeyError as exc:
            raise ValueError(f"split refers to an unloaded record: {record_name}") from exc
        seen_records.add(record_number)

        split_text = str(row.get("split") or "")
        if split_text not in ("train", "validation", "test"):
            raise ValueError(f"invalid outer split: {split_text!r}")
        split: SplitName = split_text  # type: ignore[assignment]
        participant = str(row.get("participant") or "").strip()
        session = str(row.get("session") or "").strip()
        if not participant or not session:
            raise ValueError("split row lacks participant or session grouping")
        prior = participant_splits.setdefault(participant, split)
        if prior != split:
            raise ValueError(f"participant crosses outer splits: {participant}")

        role = _role(str(row.get("status") or ""))
        if role == "population" and split == "test":
            raise ValueError("test record cannot be a population source")
        if role != "population" and split != "test":
            raise ValueError(f"{role} context must belong to the test split")
        sampling_rate = float(str(row.get("sampling_rate") or ""))
        if not np.isfinite(sampling_rate) or sampling_rate <= 0:
            raise ValueError(f"invalid sampling rate for {record_name}")
        start, end = _interval_from_row(row, role, record.samples, sampling_rate)
        grouped.append(
            GroupedKladosSource(
                participant_id=participant,
                session_id=session,
                source_record=record_name,
                split=split,
                role=role,
                start_sample=start,
                end_sample=end,
                sampling_rate=sampling_rate,
                record=record,
            )
        )

    if seen_records != set(by_number):
        missing = sorted(set(by_number) - seen_records)
        raise ValueError(f"loaded records absent from split manifest: {missing}")

    participant_sources: dict[str, list[GroupedKladosSource]] = {}
    for source in grouped:
        participant_sources.setdefault(source.participant_id, []).append(source)
    for participant, participant_group in participant_sources.items():
        if len(participant_group) != 2:
            raise ValueError(
                f"Klados participant group must contain two source records: {participant}"
            )
        sessions = {source.session_id for source in participant_group}
        records_in_group = {source.source_record for source in participant_group}
        if len(sessions) != 2 or len(records_in_group) != 2:
            raise ValueError(f"duplicate source/session within participant group: {participant}")

    test_participants = {
        source.participant_id for source in grouped if source.split == "test"
    }
    for participant in test_participants:
        roles = {source.role for source in grouped if source.participant_id == participant}
        if not {"calibration", "query"}.issubset(roles):
            raise ValueError(f"held-out participant lacks calibration/query: {participant}")
    assert_calibration_query_disjoint(grouped)
    return tuple(grouped)


def fit_outer_train_clean_normalizer(
    sources: Sequence[GroupedKladosSource],
) -> CleanNormalizer:
    """Fit once on unique raw outer-training clean records, never windows."""

    train = [
        source
        for source in sources
        if source.split == "train" and source.role == "population"
    ]
    if not train:
        raise ValueError("no outer-training clean source is available for normalization")
    channels = train[0].record.clean.shape[0]
    total = np.zeros((channels, 1), dtype=np.float64)
    total_square = np.zeros((channels, 1), dtype=np.float64)
    sample_count = 0
    for source in train:
        clean = np.asarray(
            source.record.clean[:, source.start_sample : source.end_sample],
            dtype=np.float64,
        )
        if clean.shape[0] != channels or not np.isfinite(clean).all():
            raise ValueError(f"invalid clean training source: {source.source_record}")
        total += clean.sum(axis=1, keepdims=True)
        total_square += np.square(clean).sum(axis=1, keepdims=True)
        sample_count += clean.shape[1]
    if sample_count < 2:
        raise ValueError("too few outer-training clean samples for normalization")
    mean = total / sample_count
    variance = total_square / sample_count - np.square(mean)
    variance = np.maximum(variance, 0.0)
    if np.any(variance <= np.finfo(np.float64).eps):
        raise ValueError("outer-training clean EEG contains a constant channel")
    scale = np.sqrt(variance)
    return CleanNormalizer(
        mean=mean,
        scale=scale,
        fit_participant_ids=tuple(sorted({source.participant_id for source in train})),
        fit_source_records=tuple(sorted(source.source_record for source in train)),
        fit_sample_count=sample_count,
    )


def _readonly(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value, dtype=np.float64)
    result.setflags(write=False)
    return result


def window_grouped_klados(
    sources: Sequence[GroupedKladosSource],
    normalizer: CleanNormalizer,
    *,
    window_samples: int,
    train_stride_samples: int,
    eval_stride_samples: int,
    filter_guard_samples: int = 0,
) -> tuple[KladosWindow, ...]:
    """Window already-grouped sources without making any split decision."""

    if min(window_samples, train_stride_samples, eval_stride_samples) <= 0:
        raise ValueError("window and stride sizes must be positive")
    if filter_guard_samples < 0:
        raise ValueError("filter guard cannot be negative")
    assert_calibration_query_disjoint(sources)
    windows: list[KladosWindow] = []
    for source in sources:
        first = source.start_sample + filter_guard_samples
        stop = source.end_sample - filter_guard_samples
        if stop - first < window_samples:
            raise ValueError(f"source is too short after guard: {source.context_id}")
        stride = train_stride_samples if source.split == "train" else eval_stride_samples
        eog_full = np.stack([source.record.veog, source.record.heog], axis=0)
        for start in range(first, stop - window_samples + 1, stride):
            end = start + window_samples
            key = WindowKey(
                participant_id=source.participant_id,
                session_id=source.session_id,
                source_record=source.source_record,
                split=source.split,
                role=source.role,
                start_sample=start,
                end_sample=end,
            )
            contaminated = _readonly(
                normalizer.transform(source.record.contaminated[:, start:end])
            )
            external = _readonly(eog_full[:, start:end])
            if source.role == "calibration":
                windows.append(CalibrationWindow(key, contaminated, external))
                continue
            clean = _readonly(normalizer.transform(source.record.clean[:, start:end]))
            if source.role == "query":
                windows.append(QueryWindow(key, contaminated, clean, external))
            else:
                windows.append(PopulationWindow(key, contaminated, clean, external))
    if not windows:
        raise ValueError("windowing produced no Klados examples")
    return tuple(windows)


def build_klados_pipeline(
    records: Sequence[KladosRecord],
    split_manifest: Path,
    config: dict[str, Any],
) -> KladosPipeline:
    """Build the lightweight real-data path in its required leakage-safe order."""

    sources = group_klados_records(records, split_manifest)
    normalizer = fit_outer_train_clean_normalizer(sources)
    split_config = config["split"] if "split" in config else config
    sampling_rate = float(
        config.get("official_description", {}).get(
            "sampling_rate", sources[0].sampling_rate
        )
    )
    guard = int(round(float(split_config.get("filter_guard_seconds", 0.0)) * sampling_rate))
    windows = window_grouped_klados(
        sources,
        normalizer,
        window_samples=int(split_config["window_samples"]),
        train_stride_samples=int(split_config["train_stride_samples"]),
        eval_stride_samples=int(split_config["eval_stride_samples"]),
        filter_guard_samples=guard,
    )
    return KladosPipeline(sources=sources, normalizer=normalizer, windows=windows)
