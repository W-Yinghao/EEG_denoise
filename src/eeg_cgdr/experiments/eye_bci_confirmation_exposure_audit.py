"""Manifest-only Eye-BCI confirmation-exposure audit.

The audit deliberately reads only the registered dataset metadata, the frozen
configuration, the frozen split manifest, and a small execution-status
manifest.  It never opens an Eye-BCI record, EOG stream, annotation, metrics
table, result summary, seed result, or other candidate outcome payload.

This is an exposure inventory, not a new scientific evaluation.  In
particular, a participant/session mentioned in a validation role is not called
fresh merely because the allowed manifests do not prove that its signal was
opened.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


PROTOCOL_ID = "eye_bci_confirmation_exposure_manifest_audit_v1"
DEFAULT_REGISTRY = Path("datasets/registry/eye_bci.json")
DEFAULT_CONFIG = Path("configs/cgdr/p0_eye_bci_fold00.yaml")
DEFAULT_SPLIT = Path("datasets/splits/eye_bci_me_outer_fold_00.csv")
DEFAULT_STATUS = Path(
    "results/cgdr/eye_bci_me_outer_fold_00/progress/status.json"
)
DEFAULT_OUTPUT = Path("reports/eye_bci_confirmation_exposure_audit.json")
DEFAULT_CATALOGUED_REPORTS = (
    Path("reports/cgdr_code_audit_after_f9e00ab.md"),
    Path("reports/cgdr_mechanism_decision.md"),
    Path("results/cgdr/eye_bci_me_outer_fold_00/RESULT.md"),
)

SPLIT_FIELDS = {
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
SPLIT_ROLES = ("train", "validation", "test")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    output = [str(item) for item in value]
    if not output or len(output) != len(set(output)):
        raise ValueError(f"{label} must be non-empty and unique")
    return output


def _participant(value: Any) -> str:
    text = str(value)
    if re.fullmatch(r"S\d{2}", text) is None:
        raise ValueError(f"invalid Eye-BCI participant ID: {text!r}")
    return text


def _session(value: Any) -> str:
    text = str(value)
    if re.fullmatch(r"Sess\d+", text) is None:
        raise ValueError(f"invalid Eye-BCI session ID: {text!r}")
    return text


def _pair_sort_key(pair: tuple[str, str]) -> tuple[int, int, str]:
    participant, session = pair
    session_number = int(session.removeprefix("Sess"))
    return int(participant.removeprefix("S")), session_number, session


def _load_json_object(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {path}: {error}") from error
    return _mapping(value, label)


def _load_yaml_object(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read {label}: {path}: {error}") from error
    return _mapping(value, label)


def _load_split(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or set(reader.fieldnames) != SPLIT_FIELDS:
                raise ValueError("Eye-BCI split does not use the frozen 12-field schema")
            rows = [dict(row) for row in reader]
    except OSError as error:
        raise ValueError(f"cannot read Eye-BCI split manifest: {path}: {error}") from error
    if not rows:
        raise ValueError("Eye-BCI split manifest is empty")
    return rows


def _validate_allowed_manifest_paths(
    *, registry_path: Path, config_path: Path, split_path: Path, status_path: Path
) -> None:
    expected = {
        "registry": (registry_path, "eye_bci.json"),
        "config": (config_path, "p0_eye_bci_fold00.yaml"),
        "split": (split_path, "eye_bci_me_outer_fold_00.csv"),
        "status": (status_path, "status.json"),
    }
    for label, (path, filename) in expected.items():
        if path.name != filename:
            raise ValueError(
                f"{label} input is not the allowlisted Eye-BCI manifest: {path}"
            )
    if status_path.parent.name != "progress":
        raise ValueError("Eye-BCI status input must be the progress manifest")


def _registry_inventory_counts(registry: Mapping[str, Any]) -> tuple[int, int]:
    """Read explicit counts, falling back to the existing registry note."""

    participant_count = registry.get("participant_count")
    session_count = registry.get("session_count")
    if isinstance(participant_count, int) and isinstance(session_count, int):
        if participant_count > 0 and session_count >= participant_count:
            return participant_count, session_count

    notes = registry.get("notes", [])
    if isinstance(notes, Sequence) and not isinstance(notes, (str, bytes)):
        evidence = "\n".join(str(value) for value in notes)
    else:
        evidence = str(notes)
    match = re.search(
        r"(?P<participants>\d+)\s+(?:subjects|participants),\s*"
        r"(?P<sessions>\d+)\s+sessions",
        evidence,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise ValueError("Eye-BCI registry lacks participant/session inventory counts")
    participants = int(match.group("participants"))
    sessions = int(match.group("sessions"))
    if participants <= 0 or sessions < participants:
        raise ValueError("Eye-BCI registry participant/session counts are invalid")
    return participants, sessions


def _pair_payload(
    pair: tuple[str, str],
    *,
    split_roles: Sequence[str],
    split_statuses: Sequence[str],
    exposure_status: str,
    confirmation_use: str,
) -> dict[str, Any]:
    return {
        "participant": pair[0],
        "session": pair[1],
        "split_roles": list(split_roles),
        "split_statuses": list(split_statuses),
        "exposure_status": exposure_status,
        "confirmation_use": confirmation_use,
    }


def audit_eye_bci_confirmation_exposure(
    *,
    registry_path: Path = DEFAULT_REGISTRY,
    config_path: Path = DEFAULT_CONFIG,
    split_path: Path = DEFAULT_SPLIT,
    status_path: Path = DEFAULT_STATUS,
    catalogued_reports: Sequence[Path] = DEFAULT_CATALOGUED_REPORTS,
) -> dict[str, Any]:
    """Return a conservative participant/session exposure inventory.

    Result and metric paths contained in the allowed manifests are recorded as
    references only.  They are never dereferenced by this function.
    """

    _validate_allowed_manifest_paths(
        registry_path=registry_path,
        config_path=config_path,
        split_path=split_path,
        status_path=status_path,
    )

    registry = _load_json_object(registry_path, "Eye-BCI registry")
    if registry.get("dataset_id") != "eye_bci":
        raise ValueError("dataset registry is not Eye-BCI")
    if registry.get("status") != "verified_available":
        raise ValueError("Eye-BCI registry is not verified_available")
    registered_participants, registered_sessions = _registry_inventory_counts(
        registry
    )

    config = _load_yaml_object(config_path, "Eye-BCI config")
    eye = _mapping(config.get("eye_bci"), "Eye-BCI config section")
    experiment_id = str(config.get("experiment_id", ""))
    outer_fold = str(eye.get("outer_fold", ""))
    if not experiment_id or experiment_id != outer_fold:
        raise ValueError("Eye-BCI experiment and outer-fold IDs differ")

    configured: dict[str, list[str]] = {
        "train": [
            _participant(value)
            for value in _string_list(
                eye.get("training_participants"), "training participants"
            )
        ],
        "validation": [
            _participant(value)
            for value in _string_list(
                eye.get("validation_participants"), "validation participants"
            )
        ],
        "test": [
            _participant(value)
            for value in _string_list(
                eye.get("test_participants"), "test participants"
            )
        ],
    }
    role_sets = {role: set(values) for role, values in configured.items()}
    if (
        role_sets["train"] & role_sets["validation"]
        or role_sets["train"] & role_sets["test"]
        or role_sets["validation"] & role_sets["test"]
    ):
        raise ValueError("Eye-BCI configured participant roles overlap")
    configured_participant_union = set().union(*role_sets.values())
    if len(configured_participant_union) != registered_participants:
        raise ValueError(
            "configured participant union does not match the registry participant count"
        )

    rows = _load_split(split_path)
    pair_rows: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    manifest_role_participants: dict[str, set[str]] = {
        role: set() for role in SPLIT_ROLES
    }
    for row in rows:
        if row["dataset_version"] != "eye_bci_syn64005218_neuroscan":
            raise ValueError("Eye-BCI split dataset version changed")
        if row["outer_fold"] != outer_fold:
            raise ValueError("Eye-BCI split outer-fold ID changed")
        role = row["split"]
        if role not in manifest_role_participants:
            raise ValueError(f"unexpected Eye-BCI split role: {role!r}")
        participant = _participant(row["participant"])
        session = _session(row["session"])
        manifest_role_participants[role].add(participant)
        pair_rows[(participant, session)].append(row)
    for role in SPLIT_ROLES:
        if manifest_role_participants[role] != role_sets[role]:
            raise ValueError(f"Eye-BCI {role} config/split participants differ")

    status_manifest_present = status_path.is_file()
    execution_state = "missing_status_manifest"
    completed_execution = False
    status: Mapping[str, Any] = {}
    if status_manifest_present:
        status = _load_json_object(status_path, "Eye-BCI execution status")
        if status.get("experiment_id") != experiment_id:
            raise ValueError("Eye-BCI execution status is cross-wired")
        execution_state = str(status.get("status", "unknown"))
        completed_execution = execution_state == "completed"

    calibration_pairs = {
        pair
        for pair, pair_members in pair_rows.items()
        if any(row["status"] == "held_out_calibration" for row in pair_members)
    }
    query_pairs = {
        pair
        for pair, pair_members in pair_rows.items()
        if any(row["status"] == "held_out_query" for row in pair_members)
    }
    train_pairs = {
        pair
        for pair, pair_members in pair_rows.items()
        if any(row["split"] == "train" for row in pair_members)
    }
    validation_pairs = {
        pair
        for pair, pair_members in pair_rows.items()
        if any(row["split"] == "validation" for row in pair_members)
    }

    definite_exposure: list[dict[str, Any]] = []
    access_unresolved: list[dict[str, Any]] = []
    possibly_query_unexposed: list[dict[str, str]] = []
    pair_audit: list[dict[str, Any]] = []
    for pair in sorted(pair_rows, key=_pair_sort_key):
        members = pair_rows[pair]
        roles = sorted({row["split"] for row in members})
        statuses = sorted({row["status"] for row in members})
        if completed_execution and pair in query_pairs:
            exposure_status = "query_evaluation_exposed"
            confirmation_use = "exclude_from_fresh_confirmation"
        elif completed_execution and pair in calibration_pairs:
            exposure_status = "heldout_calibration_signal_exposed"
            confirmation_use = "exclude_from_fresh_confirmation"
        elif completed_execution and pair in train_pairs:
            exposure_status = "population_training_signal_exposed"
            confirmation_use = "exclude_from_fresh_confirmation"
        elif status_manifest_present and pair in (query_pairs | calibration_pairs | train_pairs):
            exposure_status = "possibly_exposed_by_nonterminal_execution"
            confirmation_use = "not_fresh_without_stronger_execution_audit"
        elif pair in validation_pairs:
            exposure_status = "validation_role_declared_access_not_proven"
            confirmation_use = "not_fresh_without_stronger_access_audit"
        else:
            exposure_status = "split_role_declared_no_execution_evidence"
            confirmation_use = "not_fresh_without_stronger_access_audit"

        payload = _pair_payload(
            pair,
            split_roles=roles,
            split_statuses=statuses,
            exposure_status=exposure_status,
            confirmation_use=confirmation_use,
        )
        pair_audit.append(payload)
        if exposure_status in {
            "query_evaluation_exposed",
            "heldout_calibration_signal_exposed",
            "population_training_signal_exposed",
        }:
            definite_exposure.append(payload)
        else:
            access_unresolved.append(payload)
        if pair not in query_pairs or not completed_execution:
            possibly_query_unexposed.append(
                {"participant": pair[0], "session": pair[1]}
            )

    exact_cross_session_candidates: list[dict[str, str]] = []
    for participant in sorted(configured_participant_union):
        participant_calibrations = sorted(
            session for owner, session in calibration_pairs if owner == participant
        )
        participant_queries = sorted(
            session for owner, session in query_pairs if owner == participant
        )
        for calibration_session in participant_calibrations:
            for query_session in participant_queries:
                if calibration_session != query_session:
                    exact_cross_session_candidates.append(
                        {
                            "participant": participant,
                            "calibration_session": calibration_session,
                            "query_session": query_session,
                        }
                    )

    represented_pair_count = len(pair_rows)
    unmapped_session_pair_count = registered_sessions - represented_pair_count
    if unmapped_session_pair_count < 0:
        raise ValueError("split contains more participant/session pairs than the registry")
    if exact_cross_session_candidates:
        cross_session_status = "feasible_from_existing_split_manifest"
    elif unmapped_session_pair_count > 0:
        cross_session_status = (
            "potential_but_exact_participant_session_mapping_missing"
        )
    else:
        cross_session_status = "not_evidenced_by_allowed_manifests"

    referenced_result_payloads = sorted(
        {
            str(value)
            for value in (
                _mapping(config.get("outputs"), "Eye-BCI outputs").get("metrics"),
                status.get("metrics"),
                status.get("result_summary"),
            )
            if value
        }
    )
    record_references = sorted(
        {row["record"] for row in rows if row.get("record")}
    )

    return {
        "status": "completed_manifest_only_exposure_audit",
        "protocol_id": PROTOCOL_ID,
        "scientific_role": "confirmation_exposure_inventory_not_evaluation",
        "experiment_id": experiment_id,
        "existing_experiment_evidence_status": str(
            config.get("evidence_status", "unknown")
        ),
        "exposure_definition": {
            "identity_mention_is_signal_exposure": False,
            "download_or_header_audit_is_scientific_outcome_exposure": False,
            "completed_training_role_is_signal_exposure": True,
            "completed_heldout_query_role_is_query_evaluation_exposure": True,
            "validation_role_without_access_manifest": "unresolved_not_fresh",
        },
        "input_contract": {
            "opened_manifest_files": [
                str(registry_path),
                str(config_path),
                str(split_path),
                *([str(status_path)] if status_manifest_present else []),
            ],
            "catalogued_reports_seen_by_existence_only": [
                str(path) for path in catalogued_reports if path.is_file()
            ],
            "referenced_result_payloads_not_opened": referenced_result_payloads,
            "candidate_record_references_not_opened": record_references,
            "raw_eeg_or_eog_payload_opened": False,
            "candidate_label_or_annotation_payload_opened": False,
            "metric_or_outcome_payload_opened": False,
            "file_hashes_computed": False,
        },
        "execution_manifest": {
            "present": status_manifest_present,
            "state": execution_state,
            "completed": completed_execution,
            "completed_method_seed_pairs": status.get(
                "completed_method_seed_pairs"
            ),
        },
        "inventory": {
            "registered_participant_count": registered_participants,
            "registered_participant_session_count": registered_sessions,
            "split_participant_count": len(configured_participant_union),
            "split_participant_session_count": represented_pair_count,
            "unmapped_participant_session_count": unmapped_session_pair_count,
            "split_role_counts": {
                role: len(role_sets[role]) for role in SPLIT_ROLES
            },
        },
        "participant_session_audit": pair_audit,
        "definitely_exposed_participant_sessions": definite_exposure,
        "access_unresolved_participant_sessions": access_unresolved,
        "possibly_unexposed_participant_sessions": access_unresolved,
        "query_evaluation_exposed_participant_sessions": [
            {"participant": pair[0], "session": pair[1]}
            for pair in sorted(
                query_pairs if completed_execution else set(), key=_pair_sort_key
            )
        ],
        "possibly_query_outcome_unexposed_participant_sessions": (
            possibly_query_unexposed
        ),
        "known_fresh_confirmation_participant_sessions": [],
        "freshness_boundary": (
            "No known pair is promoted to fresh confirmation evidence from a "
            "manifest-only audit. Validation-only and unmapped sessions require "
            "a stronger metadata-only access audit first."
        ),
        "cross_session_calibration_to_query": {
            "status": cross_session_status,
            "existing_split_has_cross_session_pair": bool(
                exact_cross_session_candidates
            ),
            "exact_candidates": exact_cross_session_candidates,
            "registered_participant_session_count": registered_sessions,
            "represented_participant_session_count": represented_pair_count,
            "unmapped_participant_session_count": unmapped_session_pair_count,
            "schedulable_now": bool(exact_cross_session_candidates),
            "next_safe_metadata_step": (
                "create a participant/session/paradigm inventory without opening "
                "EEG, EOG, annotations, labels, metrics, or outcomes"
            ),
        },
        "confirmation_claim_allowed": False,
    }


def write_exposure_audit(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Manifest-only Eye-BCI confirmation exposure audit"
    )
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--status-manifest", type=Path, default=DEFAULT_STATUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    result = audit_eye_bci_confirmation_exposure(
        registry_path=args.registry,
        config_path=args.config,
        split_path=args.split,
        status_path=args.status_manifest,
    )
    write_exposure_audit(args.output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(args.output),
                "confirmation_claim_allowed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through Slurm CLI
    raise SystemExit(main())
