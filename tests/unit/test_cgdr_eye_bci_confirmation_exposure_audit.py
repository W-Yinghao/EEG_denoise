"""Tests for the manifest-only Eye-BCI confirmation exposure audit."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from eeg_cgdr.experiments.eye_bci_confirmation_exposure_audit import (
    SPLIT_FIELDS,
    audit_eye_bci_confirmation_exposure,
)


def _write_fixture(
    root: Path,
    *,
    completed: bool = True,
    cross_session: bool = False,
) -> dict[str, Path]:
    registry = root / "datasets/registry/eye_bci.json"
    config = root / "configs/cgdr/p0_eye_bci_fold00.yaml"
    split = root / "datasets/splits/eye_bci_me_outer_fold_00.csv"
    status = root / "results/cgdr/eye_bci_me_outer_fold_00/progress/status.json"
    registry.parent.mkdir(parents=True)
    config.parent.mkdir(parents=True)
    split.parent.mkdir(parents=True)
    status.parent.mkdir(parents=True)

    registry.write_text(
        json.dumps(
            {
                "dataset_id": "eye_bci",
                "status": "verified_available",
                "notes": [
                    "Metadata inventory found 4 subjects, 7 sessions and 35 files."
                ],
            }
        ),
        encoding="utf-8",
    )
    config.write_text(
        yaml.safe_dump(
            {
                "experiment_id": "eye_bci_me_outer_fold_00",
                "evidence_status": "exploratory_compatibility_only",
                "eye_bci": {
                    "outer_fold": "eye_bci_me_outer_fold_00",
                    "session": "Sess01",
                    "training_participants": ["S02", "S03"],
                    "validation_participants": ["S04"],
                    "test_participants": ["S01"],
                },
                "outputs": {
                    "metrics": str(root / "forbidden/metrics.csv"),
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    base_rows = [
        {
            "dataset_version": "eye_bci_syn64005218_neuroscan",
            "outer_fold": "eye_bci_me_outer_fold_00",
            "split": "train",
            "participant": "S02",
            "session": "Sess01",
            "record": "S02/Sess01/Neuroscan/ME021.csv",
            "calibration_start": "",
            "calibration_end": "",
            "query_start": "",
            "query_end": "",
            "sampling_rate": "1000",
            "status": "population_source",
        },
        {
            "dataset_version": "eye_bci_syn64005218_neuroscan",
            "outer_fold": "eye_bci_me_outer_fold_00",
            "split": "train",
            "participant": "S03",
            "session": "Sess01",
            "record": "S03/Sess01/Neuroscan/ME031.csv",
            "calibration_start": "",
            "calibration_end": "",
            "query_start": "",
            "query_end": "",
            "sampling_rate": "1000",
            "status": "population_source",
        },
        {
            "dataset_version": "eye_bci_syn64005218_neuroscan",
            "outer_fold": "eye_bci_me_outer_fold_00",
            "split": "validation",
            "participant": "S04",
            "session": "Sess01",
            "record": "S04/Sess01/Neuroscan/ME041.csv",
            "calibration_start": "",
            "calibration_end": "",
            "query_start": "",
            "query_end": "",
            "sampling_rate": "1000",
            "status": "population_source",
        },
        {
            "dataset_version": "eye_bci_syn64005218_neuroscan",
            "outer_fold": "eye_bci_me_outer_fold_00",
            "split": "test",
            "participant": "S01",
            "session": "Sess01",
            "record": "S01/Sess01/Neuroscan/ME011.csv",
            "calibration_start": "0",
            "calibration_end": "30",
            "query_start": "",
            "query_end": "",
            "sampling_rate": "1000",
            "status": "held_out_calibration",
        },
        {
            "dataset_version": "eye_bci_syn64005218_neuroscan",
            "outer_fold": "eye_bci_me_outer_fold_00",
            "split": "test",
            "participant": "S01",
            "session": "Sess02" if cross_session else "Sess01",
            "record": (
                "S01/Sess02/Neuroscan/ME012.csv"
                if cross_session
                else "S01/Sess01/Neuroscan/ME011.csv"
            ),
            "calibration_start": "",
            "calibration_end": "",
            "query_start": "35",
            "query_end": "EOF",
            "sampling_rate": "1000",
            "status": "held_out_query",
        },
    ]
    with split.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(SPLIT_FIELDS))
        writer.writeheader()
        writer.writerows(base_rows)

    forbidden_summary = root / "forbidden/result_summary.json"
    forbidden_metrics = root / "forbidden/metrics.csv"
    # Directories make an accidental read_text/open fail while allowing the
    # status manifest to reference the paths as opaque evidence only.
    forbidden_summary.mkdir(parents=True)
    forbidden_metrics.mkdir(parents=True)
    status.write_text(
        json.dumps(
            {
                "status": "completed" if completed else "checkpointed_for_resume",
                "experiment_id": "eye_bci_me_outer_fold_00",
                "result_summary": str(forbidden_summary),
                "metrics": str(forbidden_metrics),
                "completed_method_seed_pairs": 30 if completed else 7,
            }
        ),
        encoding="utf-8",
    )
    return {
        "registry": registry,
        "config": config,
        "split": split,
        "status": status,
    }


def _audit(paths: dict[str, Path]) -> dict[str, object]:
    return audit_eye_bci_confirmation_exposure(
        registry_path=paths["registry"],
        config_path=paths["config"],
        split_path=paths["split"],
        status_path=paths["status"],
        catalogued_reports=(),
    )


def test_completed_fold_separates_training_query_and_unresolved_exposure(
    tmp_path: Path,
) -> None:
    result = _audit(_write_fixture(tmp_path))

    exposed = {
        (row["participant"], row["session"]): row["exposure_status"]
        for row in result["definitely_exposed_participant_sessions"]
    }
    unresolved = {
        (row["participant"], row["session"]): row["exposure_status"]
        for row in result["access_unresolved_participant_sessions"]
    }
    assert exposed == {
        ("S01", "Sess01"): "query_evaluation_exposed",
        ("S02", "Sess01"): "population_training_signal_exposed",
        ("S03", "Sess01"): "population_training_signal_exposed",
    }
    assert unresolved == {
        ("S04", "Sess01"): "validation_role_declared_access_not_proven"
    }
    assert result["possibly_unexposed_participant_sessions"] == list(
        result["access_unresolved_participant_sessions"]
    )
    assert result["query_evaluation_exposed_participant_sessions"] == [
        {"participant": "S01", "session": "Sess01"}
    ]
    assert result["known_fresh_confirmation_participant_sessions"] == []
    assert result["confirmation_claim_allowed"] is False


def test_audit_does_not_open_result_metrics_or_candidate_records(tmp_path: Path) -> None:
    result = _audit(_write_fixture(tmp_path))
    contract = result["input_contract"]

    assert contract["raw_eeg_or_eog_payload_opened"] is False
    assert contract["candidate_label_or_annotation_payload_opened"] is False
    assert contract["metric_or_outcome_payload_opened"] is False
    assert len(contract["referenced_result_payloads_not_opened"]) == 2
    assert len(contract["candidate_record_references_not_opened"]) == 4


def test_cross_session_is_potential_but_not_schedulable_without_exact_mapping(
    tmp_path: Path,
) -> None:
    result = _audit(_write_fixture(tmp_path))
    cross_session = result["cross_session_calibration_to_query"]

    assert cross_session["status"] == (
        "potential_but_exact_participant_session_mapping_missing"
    )
    assert cross_session["existing_split_has_cross_session_pair"] is False
    assert cross_session["unmapped_participant_session_count"] == 3
    assert cross_session["schedulable_now"] is False


def test_exact_cross_session_split_is_reported_without_opening_payloads(
    tmp_path: Path,
) -> None:
    result = _audit(_write_fixture(tmp_path, cross_session=True))
    cross_session = result["cross_session_calibration_to_query"]

    assert cross_session["status"] == "feasible_from_existing_split_manifest"
    assert cross_session["schedulable_now"] is True
    assert cross_session["exact_candidates"] == [
        {
            "participant": "S01",
            "calibration_session": "Sess01",
            "query_session": "Sess02",
        }
    ]


def test_config_split_participant_mismatch_is_rejected(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    config = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))
    config["eye_bci"]["validation_participants"] = ["S05"]
    paths["config"].write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="validation config/split participants differ"):
        _audit(paths)


def test_nonterminal_execution_never_promotes_query_to_definitely_exposed(
    tmp_path: Path,
) -> None:
    result = _audit(_write_fixture(tmp_path, completed=False))

    assert result["definitely_exposed_participant_sessions"] == []
    assert result["query_evaluation_exposed_participant_sessions"] == []
    statuses = {
        row["exposure_status"]
        for row in result["access_unresolved_participant_sessions"]
    }
    assert "possibly_exposed_by_nonterminal_execution" in statuses
