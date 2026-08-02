"""Release-internal SGEYESUB metadata and split contracts.

The public release has five ``studyXX`` folders, but the exact mapping from
those folders to the paper's EEGDS1--EEGDS4 datasets has not been verified.
This module therefore implements only the evidenced release-internal protocol:
block 1 is support and block 2 is query.  It consumes the compact structure
audit produced by Slurm job 919218 for metadata planning.  Its separate,
explicit signal loader opens only the named SET/FDT pair during scheduled real
development runs.

Signal loading is intentionally deferred until the metadata job has frozen an
exact ordered input list for every layout.  The official native channel rule
is ``channel_type == EEG`` at commit ``2c95b4f``; this does not resolve the
release-study to paper-EEGDS mapping.  EEGLAB ``trial_labels`` are four-class
trial metadata and are not interchangeable with the sample-wise six-class
``artifactclasses`` input used by native SGEYESUB.  Release label ``0`` is
unlabelled and is excluded from class-conditioned fitting and evaluation.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


SGEYESUB_DEVELOPMENT_STUDIES = ("study01", "study03")
SGEYESUB_EVALUATION_STUDIES = ("study02", "study04", "study05")
SGEYESUB_EXPECTED_STUDY_COUNTS = {
    "study01": 5,
    "study02": 15,
    "study03": 10,
    "study04": 15,
    "study05": 14,
}
SGEYESUB_SUPPORT_BLOCK = 1
SGEYESUB_QUERY_BLOCK = 2
SGEYESUB_RELEASE_CLAIM = (
    "release_internal_block1_to_block2_not_native_replication"
)
SGEYESUB_NATIVE_INPUT_STATUS = (
    "resolved_official_exact_layout_channel_type_EEG_commit_2c95b4f"
)

# Only these support-side objects may reach an operator fitting API.  Query
# annotations live in a separate evaluation object and cannot be smuggled in
# under a generic metadata field.
SUPPORT_FIT_FIELDS = frozenset(
    {
        "support_eeg",
        "support_native_eeg",
        "support_external_eog",
        "support_artifactclasses",
    }
)
SUPPORT_METADATA_ONLY_FIELDS = frozenset(
    {"support_trial_labels", "support_trial_ids"}
)
QUERY_EVALUATION_ONLY_FIELDS = frozenset(
    {
        "query_external_eog",
        "query_artifactclasses",
        "query_trial_labels",
        "query_trial_ids",
        "query_outcomes",
    }
)


def assert_operator_fit_fields(field_names: Iterable[str]) -> tuple[str, ...]:
    """Reject every query-side annotation from an operator fitting surface."""

    fields = tuple(str(value) for value in field_names)
    unknown = set(fields) - SUPPORT_FIT_FIELDS
    if unknown:
        raise ValueError(
            "operator fit fields are not support-only: "
            + ", ".join(sorted(unknown))
        )
    return fields


@dataclass(frozen=True)
class SgeyesubLayout:
    layout_id: str
    channel_labels: tuple[str, ...]
    channel_types: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.layout_id or not self.channel_labels:
            raise ValueError("SGEYESUB layout ID and channel list must be non-empty")
        if len(self.channel_labels) != len(self.channel_types):
            raise ValueError("SGEYESUB channel label/type counts differ")
        if len(set(self.channel_labels)) != len(self.channel_labels):
            raise ValueError("SGEYESUB channel labels must be unique within a layout")
        if self.channel_labels.count("artifactclasses") != 1:
            raise ValueError("SGEYESUB layout must expose exactly one artifactclasses channel")
        if not {"HEOG", "VEOG"}.issubset(self.channel_labels):
            raise ValueError("SGEYESUB layout lacks the audited HEOG/VEOG channels")

    @property
    def external_eog_labels(self) -> tuple[str, ...]:
        return tuple(
            label
            for label in ("HEOG", "VEOG", "REOG")
            if label in self.channel_labels
        )

    @property
    def native_eeg_labels(self) -> tuple[str, ...]:
        """Official demo mapping from ``eeg_decodechan(..., 'EEG', 'type')``."""

        labels = tuple(
            label
            for label, channel_type in zip(
                self.channel_labels, self.channel_types, strict=True
            )
            if channel_type == "EEG"
        )
        if not labels:
            raise ValueError("official native EEG type mapping is empty")
        return labels

    @property
    def release_internal_p0_eeg_labels(self) -> tuple[str, ...]:
        """Exact release-internal scalp list, separate from native eeg_chan_idxs.

        The audited release marks physical EOG electrodes as ``EEG`` in some
        layouts.  The P0 route therefore uses the frozen release rule
        ``type == EEG`` while excluding labels with an EOG prefix or suffix.  This
        P0-specific scalp rule is intentionally distinct from the official
        native ``eeg_chan_idxs`` rule exposed by :attr:`native_eeg_labels`.
        """

        labels = tuple(
            label
            for label, channel_type in zip(
                self.channel_labels, self.channel_types, strict=True
            )
            if channel_type == "EEG"
            and not (
                label.upper().startswith("EOG") or label.upper().endswith("EOG")
            )
        )
        if len(labels) < 2:
            raise ValueError("release-internal P0 EEG layout is empty")
        return labels


@dataclass(frozen=True)
class SgeyesubReleaseRecord:
    study: str
    participant_stem: str
    set_relative_path: str
    fdt_relative_path: str
    sampling_rate_hz: float
    channel_count: int
    samples_per_trial: int
    trial_count: int
    layout_id: str
    p0_layout_id: str
    trial_block_counts: Mapping[int, int]
    trial_label_counts: Mapping[int, int]
    trial_id_count: int

    @property
    def recording_key(self) -> str:
        return f"{self.study}/{self.participant_stem}"

    @property
    def trial_id_status(self) -> str:
        if self.trial_id_count == self.trial_count:
            return "present_complete"
        return "absent_release_metadata_use_epoch_ordinal_only"


@dataclass(frozen=True)
class SgeyesubProtocolRow:
    protocol_id: str
    claim_scope: str
    partition: str
    study: str
    participant_stem: str
    recording_key: str
    set_relative_path: str
    fdt_relative_path: str
    support_block: int
    query_block: int
    layout_id: str
    release_layout_id: str
    reference_cell_id: str
    sampling_rate_hz: float
    support_trial_count: int
    query_trial_count: int
    trial_id_status: str
    population_source_count: int
    population_source_participants: tuple[str, ...]
    status: str


@dataclass(frozen=True)
class SgeyesubProtocolPlan:
    protocol_id: str
    claim_scope: str
    rows: tuple[SgeyesubProtocolRow, ...]
    cells: tuple[dict[str, object], ...]
    layouts: tuple[SgeyesubLayout, ...]
    gamma_candidates: tuple[float, ...]

    @property
    def development_rows(self) -> tuple[SgeyesubProtocolRow, ...]:
        return tuple(row for row in self.rows if row.partition == "development")

    @property
    def evaluation_rows(self) -> tuple[SgeyesubProtocolRow, ...]:
        return tuple(row for row in self.rows if row.partition == "evaluation")

    def summary(self) -> dict[str, object]:
        blocked = [row for row in self.rows if row.status != "metadata_ready"]
        return {
            "status": "metadata_protocol_ready",
            "protocol_id": self.protocol_id,
            "claim_scope": self.claim_scope,
            "native_replication_claim_allowed": False,
            "native_input_mapping_status": SGEYESUB_NATIVE_INPUT_STATUS,
            "development_studies": list(SGEYESUB_DEVELOPMENT_STUDIES),
            "evaluation_studies": list(SGEYESUB_EVALUATION_STUDIES),
            "development_participant_stems": len(self.development_rows),
            "evaluation_participant_stems": len(self.evaluation_rows),
            "participant_unit": "release_scoped_study_participant_stem",
            "development_metadata_ready": sum(
                row.status == "metadata_ready" for row in self.development_rows
            ),
            "evaluation_metadata_ready": sum(
                row.status == "metadata_ready" for row in self.evaluation_rows
            ),
            "support_block": SGEYESUB_SUPPORT_BLOCK,
            "query_block": SGEYESUB_QUERY_BLOCK,
            "cell_fields": [
                "study",
                "layout_id",
                "reference_cell_id",
                "sampling_rate_hz",
            ],
            "cell_count": len(self.cells),
            "blocked_singleton_cells": sum(
                cell["participant_count"] == 1 for cell in self.cells
            ),
            "blocked_record_count": len(blocked),
            "study05_trial_ids": "absent_release_metadata",
            "trial_labels_semantics": (
                "four_class_trial_metadata_not_samplewise_native_labels"
            ),
            "artifactclasses_semantics": (
                "samplewise_classes_1_to_6_with_unlabelled_0_support_only"
            ),
            "population_operator_scope": (
                "same_exact_cell_other_participants_block1_only"
            ),
            "b6_family": "POP-SHRINK",
            "b6_gamma_selection": "one_global_gamma_development_only",
            "b6_gamma_candidates": list(self.gamma_candidates),
            "query_annotations_for_fit_gamma_or_method_selection": "forbidden",
            "query_annotations_for_reporting": (
                "allowed_after_all_method_outputs_frozen"
            ),
            "query_annotations_for_single_final_automatic_decision": (
                "allowed_without_adaptation_reselection_or_method_change"
            ),
        }


@dataclass(frozen=True)
class SgeyesubSupportSignals:
    """Block-1 fitting surface; trial metadata remain explicitly separate."""

    eeg: np.ndarray
    native_eeg: np.ndarray
    external_eog: np.ndarray
    artifactclasses: np.ndarray
    trial_labels: np.ndarray
    trial_ids: np.ndarray | None


@dataclass(frozen=True)
class SgeyesubQuerySignals:
    """Block-2 inference surface contains observed EEG and nothing else."""

    eeg: np.ndarray
    native_eeg: np.ndarray


@dataclass(frozen=True)
class SgeyesubQueryAnnotations:
    """Opened only after outputs freeze for metrics and one final decision."""

    external_eog: np.ndarray
    artifactclasses: np.ndarray
    trial_labels: np.ndarray
    trial_ids: np.ndarray | None


@dataclass(frozen=True)
class SgeyesubLoadedRecord:
    study: str
    participant_stem: str
    release_layout_id: str
    p0_layout_id: str
    p0_channel_labels: tuple[str, ...]
    native_channel_labels: tuple[str, ...]
    sampling_rate_hz: float
    support: SgeyesubSupportSignals
    query: SgeyesubQuerySignals | None
    query_annotations: SgeyesubQueryAnnotations | None


def _integer_counts(value: object, *, name: str) -> dict[int, int]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    result: dict[int, int] = {}
    for key, count in value.items():
        integer_key = int(key)
        integer_count = int(count)
        if integer_count < 1 or integer_count != count:
            raise ValueError(f"{name} contains a non-positive count")
        result[integer_key] = integer_count
    return result


def _positive_int(value: object, *, name: str) -> int:
    result = int(value)
    if isinstance(value, bool) or result != value or result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _h5_deref(h5_file: Any, node: Any, *, max_depth: int = 8) -> Any:
    import h5py

    depth = 0
    while (
        isinstance(node, h5py.Dataset)
        and node.size == 1
        and h5py.check_dtype(ref=node.dtype) is not None
    ):
        if depth >= max_depth:
            raise ValueError("SGEYESUB HDF5 reference depth exceeds safety limit")
        reference = np.asarray(node[()]).reshape(-1)[0]
        if not isinstance(reference, h5py.Reference) or not reference:
            raise ValueError("invalid SGEYESUB HDF5 reference")
        node = h5_file[reference]
        depth += 1
    return node


def _h5_field(group: Any, name: str) -> Any:
    import h5py

    if isinstance(group, h5py.Group) and name in group:
        return group[name]
    return None


def _h5_numeric(h5_file: Any, node: Any, *, maximum_elements: int) -> np.ndarray | None:
    import h5py

    node = _h5_deref(h5_file, node)
    if not isinstance(node, h5py.Dataset):
        return None
    if h5py.check_dtype(ref=node.dtype) is not None or node.size > maximum_elements:
        raise ValueError("unsupported or oversized SGEYESUB metadata array")
    value = np.asarray(node[()])
    if value.dtype.kind not in "iuf":
        raise ValueError("SGEYESUB protocol metadata must be numeric")
    return value


def _read_protocol_trial_blocks(
    set_path: Path, *, expected_trials: int
) -> np.ndarray:
    """Read the complete structural block vector needed to define the split."""

    import h5py

    with h5py.File(set_path, "r") as h5_file:
        if "EEG" not in h5_file:
            raise ValueError("SGEYESUB SET lacks EEG metadata")
        eeg = _h5_deref(h5_file, h5_file["EEG"])
        etc = _h5_deref(h5_file, _h5_field(eeg, "etc"))
        blocks = _h5_numeric(
            h5_file,
            _h5_field(etc, "trial_blocks"),
            maximum_elements=expected_trials,
        )
    if blocks is None:
        raise ValueError("SGEYESUB SET lacks trial_blocks")
    block_vector = np.asarray(blocks, dtype=np.int64).reshape(-1, order="F")
    if block_vector.size != expected_trials:
        raise ValueError("SGEYESUB trial_blocks length mismatch")
    return block_vector


def _h5_selected_numeric_vector(
    h5_file: Any,
    node: Any,
    *,
    indices: np.ndarray,
    name: str,
) -> np.ndarray | None:
    """Read only selected entries from a vector-shaped HDF5 metadata field."""

    import h5py

    node = _h5_deref(h5_file, node)
    if not isinstance(node, h5py.Dataset):
        return None
    if h5py.check_dtype(ref=node.dtype) is not None:
        raise ValueError(f"unsupported SGEYESUB {name} array")
    if np.dtype(node.dtype).kind not in "iuf":
        raise ValueError(f"SGEYESUB {name} metadata must be numeric")
    selected = np.asarray(indices, dtype=int).reshape(-1)
    if selected.size < 1 or np.any(selected < 0) or np.any(selected >= node.size):
        raise ValueError(f"SGEYESUB selected {name} indices are unavailable")
    if node.ndim == 1:
        value = np.asarray(node[selected])
    elif node.ndim == 2 and node.shape[0] == 1:
        value = np.asarray(node[0, selected])
    elif node.ndim == 2 and node.shape[1] == 1:
        value = np.asarray(node[selected, 0])
    else:
        raise ValueError(f"SGEYESUB {name} metadata must be vector-shaped")
    return value.reshape(-1)


def _integer_metadata(value: np.ndarray, *, name: str) -> np.ndarray:
    numeric = np.asarray(value, dtype=np.float64).reshape(-1)
    rounded = np.rint(numeric)
    if not np.isfinite(numeric).all() or not np.array_equal(numeric, rounded):
        raise ValueError(f"SGEYESUB selected {name} must contain finite integers")
    return rounded.astype(np.int64)


def _read_protocol_trial_annotations(
    set_path: Path,
    *,
    trial_indices: np.ndarray,
    expected_trials: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Read annotations for one already-frozen block and no other trials."""

    import h5py

    selected_trials = np.asarray(trial_indices, dtype=int).reshape(-1)
    if expected_trials < 1:
        raise ValueError("expected_trials must be positive")
    if (
        selected_trials.size < 1
        or np.any(selected_trials < 0)
        or np.any(selected_trials >= expected_trials)
    ):
        raise ValueError("selected SGEYESUB trial index lies outside the record")
    with h5py.File(set_path, "r") as h5_file:
        if "EEG" not in h5_file:
            raise ValueError("SGEYESUB SET lacks EEG metadata")
        eeg = _h5_deref(h5_file, h5_file["EEG"])
        etc = _h5_deref(h5_file, _h5_field(eeg, "etc"))
        labels = _h5_selected_numeric_vector(
            h5_file,
            _h5_field(etc, "trial_labels"),
            indices=selected_trials,
            name="trial_labels",
        )
        trial_ids = _h5_selected_numeric_vector(
            h5_file,
            _h5_field(etc, "trial_ids"),
            indices=selected_trials,
            name="trial_ids",
        )
    if labels is None:
        raise ValueError("SGEYESUB SET lacks trial_labels")
    return _integer_metadata(labels, name="trial_labels"), (
        None
        if trial_ids is None
        else _integer_metadata(trial_ids, name="trial_ids")
    )


def _contained_file(root: Path, relative: str) -> Path:
    relative_path = Path(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("SGEYESUB record path must be contained and relative")
    root_resolved = root.resolve(strict=True)
    path = root / relative_path
    if path.is_symlink():
        raise ValueError("SGEYESUB record payload cannot be a symlink")
    resolved = path.resolve(strict=True)
    if not resolved.is_file() or not resolved.is_relative_to(root_resolved):
        raise ValueError("SGEYESUB record payload lies outside the registered root")
    return resolved


def _concatenate_channel_trials(
    data: np.ndarray,
    channel_indices: np.ndarray,
    trial_indices: np.ndarray,
) -> np.ndarray:
    """Materialize only requested channel/trial intersections from the memmap."""

    channels = np.asarray(channel_indices, dtype=int).reshape(-1)
    trials = np.asarray(trial_indices, dtype=int).reshape(-1)
    if channels.size < 1:
        raise ValueError("SGEYESUB selected channel set is empty")
    if trials.size < 1:
        raise ValueError("SGEYESUB support/query block has no trials")
    if (
        np.any(channels < 0)
        or np.any(channels >= data.shape[0])
        or np.any(trials < 0)
        or np.any(trials >= data.shape[2])
    ):
        raise ValueError("SGEYESUB selected signal index lies outside the record")
    result = np.empty(
        (channels.size, trials.size * data.shape[1]), dtype=np.float64
    )
    for position, trial_index in enumerate(trials):
        start = position * data.shape[1]
        stop = start + data.shape[1]
        result[:, start:stop] = np.asarray(
            data[channels, :, int(trial_index)], dtype=np.float64
        )
    return np.ascontiguousarray(result)


def _freeze_array(value: np.ndarray) -> np.ndarray:
    result = np.ascontiguousarray(value)
    if not np.isfinite(result).all():
        raise ValueError("SGEYESUB selected signal contains NaN or Inf")
    result.setflags(write=False)
    return result


def _validated_artifactclasses(value: np.ndarray) -> np.ndarray:
    numeric = np.asarray(value, dtype=np.float64).reshape(-1)
    rounded = np.rint(numeric)
    if not np.isfinite(numeric).all():
        raise ValueError("artifactclasses contains NaN or Inf")
    if not np.allclose(numeric, rounded, atol=1.0e-5, rtol=0.0):
        raise ValueError("artifactclasses channel is not integer encoded")
    if not set(np.unique(rounded).astype(int)).issubset(set(range(0, 7))):
        raise ValueError("artifactclasses values fall outside release labels 0..6")
    return _freeze_array(rounded.astype(np.int64))


def load_sgeyesub_signal_record(
    root: Path,
    record: SgeyesubReleaseRecord,
    layout: SgeyesubLayout,
    *,
    include_query: bool = True,
    include_query_annotations: bool | None = None,
) -> SgeyesubLoadedRecord:
    """Load one explicit SET/FDT pair and partition trials before flattening.

    Population and wrong-source fits set ``include_query=False`` so no query
    signal row is sliced from those source FDT files.  A target run can load
    query EEG with annotations withheld, then explicitly reopen annotations
    only after all method outputs are frozen.
    """

    if record.layout_id != layout.layout_id:
        raise ValueError("SGEYESUB record/layout mismatch")
    annotation_flag = (
        include_query
        if include_query_annotations is None
        else bool(include_query_annotations)
    )
    if annotation_flag and not include_query:
        raise ValueError("query annotations require include_query=True")
    set_path = _contained_file(root, record.set_relative_path)
    fdt_path = _contained_file(root, record.fdt_relative_path)
    expected_values = record.channel_count * record.samples_per_trial * record.trial_count
    if fdt_path.stat().st_size != expected_values * np.dtype("<f4").itemsize:
        raise ValueError("SGEYESUB FDT byte size differs from audited dimensions")
    flat = np.memmap(fdt_path, mode="r", dtype="<f4", shape=(expected_values,))
    cube = np.asarray(flat).reshape(
        (record.channel_count, record.samples_per_trial, record.trial_count),
        order="F",
    )
    blocks = _read_protocol_trial_blocks(set_path, expected_trials=record.trial_count)
    if set(blocks.tolist()) != {SGEYESUB_SUPPORT_BLOCK, SGEYESUB_QUERY_BLOCK}:
        raise ValueError("SGEYESUB signal record does not contain blocks 1 and 2")

    support_trials = np.flatnonzero(blocks == SGEYESUB_SUPPORT_BLOCK)
    query_trials = np.flatnonzero(blocks == SGEYESUB_QUERY_BLOCK)
    support_trial_labels, support_trial_ids = _read_protocol_trial_annotations(
        set_path,
        trial_indices=support_trials,
        expected_trials=record.trial_count,
    )
    if record.study == "study05" and support_trial_ids is not None:
        raise ValueError("study05 support trial_ids unexpectedly appeared")
    if record.study != "study05" and support_trial_ids is None:
        raise ValueError("non-study05 support trial_ids unexpectedly disappeared")
    p0_labels = layout.release_internal_p0_eeg_labels
    p0_indices = np.asarray(
        [layout.channel_labels.index(label) for label in p0_labels], dtype=int
    )
    native_labels = layout.native_eeg_labels
    native_indices = np.asarray(
        [layout.channel_labels.index(label) for label in native_labels], dtype=int
    )
    eog_indices = np.asarray(
        [layout.channel_labels.index("HEOG"), layout.channel_labels.index("VEOG")],
        dtype=int,
    )
    artifact_index = layout.channel_labels.index("artifactclasses")

    support = SgeyesubSupportSignals(
        eeg=_freeze_array(
            _concatenate_channel_trials(cube, p0_indices, support_trials)
        ),
        native_eeg=_freeze_array(
            _concatenate_channel_trials(cube, native_indices, support_trials)
        ),
        external_eog=_freeze_array(
            _concatenate_channel_trials(cube, eog_indices, support_trials)
        ),
        artifactclasses=_validated_artifactclasses(
            _concatenate_channel_trials(
                cube, np.asarray([artifact_index], dtype=int), support_trials
            )
        ),
        trial_labels=_freeze_array(support_trial_labels),
        trial_ids=(
            None
            if support_trial_ids is None
            else _freeze_array(support_trial_ids)
        ),
    )
    query = None
    query_annotations = None
    if include_query:
        query = SgeyesubQuerySignals(
            eeg=_freeze_array(
                _concatenate_channel_trials(cube, p0_indices, query_trials)
            ),
            native_eeg=_freeze_array(
                _concatenate_channel_trials(cube, native_indices, query_trials)
            ),
        )
    if annotation_flag:
        query_trial_labels, query_trial_ids = _read_protocol_trial_annotations(
            set_path,
            trial_indices=query_trials,
            expected_trials=record.trial_count,
        )
        if record.study == "study05" and query_trial_ids is not None:
            raise ValueError("study05 query trial_ids unexpectedly appeared")
        if record.study != "study05" and query_trial_ids is None:
            raise ValueError("non-study05 query trial_ids unexpectedly disappeared")
        query_annotations = SgeyesubQueryAnnotations(
            external_eog=_freeze_array(
                _concatenate_channel_trials(cube, eog_indices, query_trials)
            ),
            artifactclasses=_validated_artifactclasses(
                _concatenate_channel_trials(
                    cube, np.asarray([artifact_index], dtype=int), query_trials
                )
            ),
            trial_labels=_freeze_array(query_trial_labels),
            trial_ids=(
                None
                if query_trial_ids is None
                else _freeze_array(query_trial_ids)
            ),
        )
    return SgeyesubLoadedRecord(
        study=record.study,
        participant_stem=record.participant_stem,
        release_layout_id=record.layout_id,
        p0_layout_id=record.p0_layout_id,
        p0_channel_labels=p0_labels,
        native_channel_labels=native_labels,
        sampling_rate_hz=record.sampling_rate_hz,
        support=support,
        query=query,
        query_annotations=query_annotations,
    )


def load_sgeyesub_structure_audit(path: Path) -> tuple[
    tuple[SgeyesubLayout, ...], tuple[SgeyesubReleaseRecord, ...]
]:
    """Load and validate the compact 919218 structure result, not raw EEG."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("mode") != "audit-sgeyesub-structure":
        raise ValueError("not an SGEYESUB structure-audit result")
    if payload.get("state") != "structure_read":
        raise ValueError("SGEYESUB structure audit is not complete")
    if payload.get("fdt_access") != "companion_exists_but_not_opened_by_code":
        raise ValueError("SGEYESUB structure audit has unexpected FDT access semantics")

    layouts = tuple(
        SgeyesubLayout(
            layout_id=str(item["layout_id"]),
            channel_labels=tuple(str(value) for value in item["channel_labels"]),
            channel_types=tuple(str(value) for value in item["channel_types"]),
        )
        for item in payload.get("channel_layouts", ())
    )
    layout_map = {layout.layout_id: layout for layout in layouts}
    if len(layout_map) != len(layouts) or len(layouts) != 6:
        raise ValueError("SGEYESUB audit must contain six unique exact layouts")

    # Acquisition cells stay distinct even when their selected release-internal
    # scalp subsequences happen to be identical.  This keeps the singleton
    # study05 layout_06 out of the layout_05 population library.
    p0_layout_ids = {
        layout.layout_id: f"p0_{layout.layout_id}" for layout in layouts
    }

    records: list[SgeyesubReleaseRecord] = []
    for item in payload.get("recordings", ()):
        study = str(item["study"])
        participant_stem = str(item["participant_stem"])
        if study not in SGEYESUB_EXPECTED_STUDY_COUNTS:
            raise ValueError(f"unexpected SGEYESUB study: {study}")
        if not participant_stem.startswith(f"{study}_p"):
            raise ValueError("participant stem is not scoped by its release study")
        layout_id = str(item["channel_layout_id"])
        layout = layout_map.get(layout_id)
        if layout is None:
            raise ValueError("record references an unknown SGEYESUB layout")
        trial_count = _positive_int(item["trials"], name="trials")
        block_counts = _integer_counts(
            item["trial_block_counts"], name="trial_block_counts"
        )
        label_counts = _integer_counts(
            item["trial_label_counts"], name="trial_label_counts"
        )
        if set(block_counts) != {SGEYESUB_SUPPORT_BLOCK, SGEYESUB_QUERY_BLOCK}:
            raise ValueError("release record must contain exactly blocks 1 and 2")
        if set(label_counts) != {1, 2, 3, 4}:
            raise ValueError("trial_labels must contain the four audited trial classes")
        if sum(block_counts.values()) != trial_count:
            raise ValueError("trial block counts do not match the trial count")
        if sum(label_counts.values()) != trial_count:
            raise ValueError("trial label counts do not match the trial count")
        trial_id_count = int(item["trial_id_count"])
        expected_trial_ids = 0 if study == "study05" else trial_count
        if trial_id_count != expected_trial_ids:
            raise ValueError("trial ID presence differs from the audited study rule")
        channel_count = _positive_int(item["channels"], name="channels")
        if channel_count != len(layout.channel_labels):
            raise ValueError("record channel count differs from its exact layout")
        expected_set_name = f"{participant_stem}_prep.set"
        expected_fdt_name = f"{participant_stem}_prep.fdt"
        if str(item["companion_fdt_candidate"]) != expected_fdt_name:
            raise ValueError("SET/FDT companion stem differs from participant metadata")
        records.append(
            SgeyesubReleaseRecord(
                study=study,
                participant_stem=participant_stem,
                set_relative_path=f"{study}/{expected_set_name}",
                fdt_relative_path=f"{study}/{expected_fdt_name}",
                sampling_rate_hz=float(item["sampling_hz"]),
                channel_count=channel_count,
                samples_per_trial=_positive_int(
                    item["samples_per_trial"], name="samples_per_trial"
                ),
                trial_count=trial_count,
                layout_id=layout_id,
                p0_layout_id=p0_layout_ids[layout.layout_id],
                trial_block_counts=block_counts,
                trial_label_counts=label_counts,
                trial_id_count=trial_id_count,
            )
        )

    keys = [record.recording_key for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate SGEYESUB study/participant recording key")
    observed_counts = {
        study: sum(record.study == study for record in records)
        for study in SGEYESUB_EXPECTED_STUDY_COUNTS
    }
    if observed_counts != SGEYESUB_EXPECTED_STUDY_COUNTS:
        raise ValueError("SGEYESUB study counts differ from the frozen release")
    if len(records) != 59:
        raise ValueError("SGEYESUB release must contain 59 participant stems")
    return layouts, tuple(sorted(records, key=lambda item: item.recording_key))


def _validated_gamma_candidates(values: Sequence[object]) -> tuple[float, ...]:
    candidates = tuple(float(value) for value in values)
    if not candidates or len(candidates) != len(set(candidates)):
        raise ValueError("B6 gamma candidates must be unique and non-empty")
    if any(not 0.0 <= value <= 1.0 for value in candidates):
        raise ValueError("B6 development gamma candidates must lie in [0, 1]")
    if candidates != (0.0, 0.25, 0.5, 0.75, 1.0):
        raise ValueError("B6 candidates must be exactly [0, .25, .5, .75, 1]")
    return candidates


def build_sgeyesub_protocol(
    layouts: Sequence[SgeyesubLayout],
    records: Sequence[SgeyesubReleaseRecord],
    *,
    protocol_id: str,
    reference_cell_id: str,
    gamma_candidates: Sequence[object],
) -> SgeyesubProtocolPlan:
    """Build leave-one-participant-out exact-cell metadata assignments."""

    if not protocol_id or not reference_cell_id:
        raise ValueError("protocol and reference-cell IDs must be non-empty")
    candidates = _validated_gamma_candidates(gamma_candidates)
    frozen_records = tuple(records)
    if len(frozen_records) != 59:
        raise ValueError("protocol construction requires all 59 release records")

    def cell_key(record: SgeyesubReleaseRecord) -> tuple[str, str, str, float]:
        return (
            record.study,
            record.layout_id,
            reference_cell_id,
            record.sampling_rate_hz,
        )

    by_cell: dict[tuple[str, str, str, float], list[SgeyesubReleaseRecord]] = {}
    for record in frozen_records:
        by_cell.setdefault(cell_key(record), []).append(record)

    rows: list[SgeyesubProtocolRow] = []
    for record in frozen_records:
        partition = (
            "development"
            if record.study in SGEYESUB_DEVELOPMENT_STUDIES
            else "evaluation"
        )
        same_cell = by_cell[cell_key(record)]
        population_sources = tuple(
            sorted(
                candidate.participant_stem
                for candidate in same_cell
                if candidate.recording_key != record.recording_key
            )
        )
        status = "metadata_ready" if population_sources else "blocked_no_population"
        rows.append(
            SgeyesubProtocolRow(
                protocol_id=protocol_id,
                claim_scope=SGEYESUB_RELEASE_CLAIM,
                partition=partition,
                study=record.study,
                participant_stem=record.participant_stem,
                recording_key=record.recording_key,
                set_relative_path=record.set_relative_path,
                fdt_relative_path=record.fdt_relative_path,
                support_block=SGEYESUB_SUPPORT_BLOCK,
                query_block=SGEYESUB_QUERY_BLOCK,
                layout_id=record.layout_id,
                release_layout_id=record.layout_id,
                reference_cell_id=reference_cell_id,
                sampling_rate_hz=record.sampling_rate_hz,
                support_trial_count=record.trial_block_counts[SGEYESUB_SUPPORT_BLOCK],
                query_trial_count=record.trial_block_counts[SGEYESUB_QUERY_BLOCK],
                trial_id_status=record.trial_id_status,
                population_source_count=len(population_sources),
                population_source_participants=population_sources,
                status=status,
            )
        )

    cells: list[dict[str, object]] = []
    for (study, layout_id, reference_id, sampling_rate), members in sorted(
        by_cell.items()
    ):
        cells.append(
            {
                "cell_id": (
                    f"{study}__{layout_id}__{reference_id}__{sampling_rate:g}hz"
                ),
                "study": study,
                "layout_id": layout_id,
                "reference_cell_id": reference_id,
                "sampling_rate_hz": sampling_rate,
                "participant_count": len(members),
                "participant_stems": sorted(
                    record.participant_stem for record in members
                ),
                "population_policy": (
                    "leave_one_participant_out_block1_same_exact_cell"
                ),
                "status": (
                    "metadata_ready" if len(members) > 1 else "blocked_no_population"
                ),
            }
        )

    plan = SgeyesubProtocolPlan(
        protocol_id=protocol_id,
        claim_scope=SGEYESUB_RELEASE_CLAIM,
        rows=tuple(rows),
        cells=tuple(cells),
        layouts=tuple(layouts),
        gamma_candidates=candidates,
    )
    if len(plan.development_rows) != 15 or len(plan.evaluation_rows) != 44:
        raise AssertionError("frozen SGEYESUB development/evaluation counts changed")
    return plan


def write_sgeyesub_protocol_outputs(
    plan: SgeyesubProtocolPlan, output_root: Path
) -> dict[str, str]:
    """Write the small metadata plan; no signal, checksum or dataset copy."""

    output_root.mkdir(parents=True, exist_ok=True)
    split_path = output_root / "split_manifest.csv"
    fields = [
        field.name for field in SgeyesubProtocolRow.__dataclass_fields__.values()
    ]
    with split_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in plan.rows:
            payload = asdict(row)
            payload["population_source_participants"] = ";".join(
                row.population_source_participants
            )
            writer.writerow(payload)

    cells_path = output_root / "protocol_cells.json"
    cells_path.write_text(
        json.dumps(list(plan.cells), indent=2) + "\n", encoding="utf-8"
    )
    layout_path = output_root / "layout_contracts.json"
    layout_path.write_text(
        json.dumps(
            [
                {
                    "layout_id": layout.layout_id,
                    "channel_count": len(layout.channel_labels),
                    "ordered_channel_labels": list(layout.channel_labels),
                    "ordered_channel_types": list(layout.channel_types),
                    "external_eog_labels": list(layout.external_eog_labels),
                    "artifactclasses_channel_index": layout.channel_labels.index(
                        "artifactclasses"
                    ),
                    "native_eeg_chan_idxs_status": SGEYESUB_NATIVE_INPUT_STATUS,
                    "native_eeg_chan_idxs_rule": (
                        "exact_layout_channel_type_equals_EEG"
                    ),
                    "native_ordered_eeg_labels": list(layout.native_eeg_labels),
                    "release_internal_p0_eeg_labels": list(
                        layout.release_internal_p0_eeg_labels
                    ),
                    "release_internal_p0_input_status": "metadata_ready",
                }
                for layout in plan.layouts
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    summary_path = output_root / "result_summary.json"
    summary_path.write_text(
        json.dumps(plan.summary(), indent=2) + "\n", encoding="utf-8"
    )
    input_policy_path = output_root / "input_policy.json"
    input_policy_path.write_text(
        json.dumps(
            {
                "support_fit_fields": sorted(SUPPORT_FIT_FIELDS),
                "support_metadata_only_fields": sorted(
                    SUPPORT_METADATA_ONLY_FIELDS
                ),
                "query_evaluation_only_fields": sorted(
                    QUERY_EVALUATION_ONLY_FIELDS
                ),
                "query_annotations_for_fit_gamma_or_method_selection": "forbidden",
                "query_annotations_for_reporting": (
                    "allowed_after_all_method_outputs_frozen"
                ),
                "query_annotations_for_single_final_automatic_decision": (
                    "allowed_without_adaptation_reselection_or_method_change"
                ),
                "trial_labels_are_artifactclasses": False,
                "native_input_mapping_status": SGEYESUB_NATIVE_INPUT_STATUS,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "split_manifest": str(split_path),
        "protocol_cells": str(cells_path),
        "layout_contracts": str(layout_path),
        "result_summary": str(summary_path),
        "input_policy": str(input_policy_path),
    }
