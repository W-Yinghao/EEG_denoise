"""Structural tests for the release-internal SGEYESUB protocol."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from eeg_cgdr.data.sgeyesub import (
    QUERY_EVALUATION_ONLY_FIELDS,
    SGEYESUB_NATIVE_INPUT_STATUS,
    SGEYESUB_RELEASE_CLAIM,
    assert_operator_fit_fields,
    build_sgeyesub_protocol,
    load_sgeyesub_signal_record,
    load_sgeyesub_structure_audit,
    write_sgeyesub_protocol_outputs,
)
from eeg_cgdr.experiments.sgeyesub_protocol import (
    validate_sgeyesub_protocol_config,
)
from eeg_cgdr.experiments.sgeyesub_operator_specificity import (
    _condition_erp_preservation,
    _covariance_distortion,
    _load_frozen_development_gamma,
    _operator_specificity_decision,
    _predicted_contamination_remaining,
    _soft_restore,
    _support_only_composite_score,
    run_sgeyesub_evaluation_record,
    select_global_gamma,
)


STUDY_COUNTS = {
    "study01": 5,
    "study02": 15,
    "study03": 10,
    "study04": 15,
    "study05": 14,
}
STUDY_LAYOUT = {
    "study01": "layout_01",
    "study02": "layout_02",
    "study03": "layout_03",
    "study04": "layout_04",
    "study05": "layout_05",
}
STUDY_RATE = {
    "study01": 200,
    "study02": 200,
    "study03": 200,
    "study04": 100,
    "study05": 256,
}


def _structure_payload() -> dict[str, object]:
    layouts = []
    for index in range(1, 7):
        p0_label_index = 5 if index == 6 else index
        layouts.append(
            {
                "layout_id": f"layout_{index:02d}",
                "channel_labels": [
                    f"EEG_{p0_label_index}_A",
                    f"EEG_{p0_label_index}_B",
                    "EOG-R-Bottom",
                    "HEOG",
                    "VEOG",
                    "REOG",
                    "artifactclasses",
                ],
                # Some release layouts type physical EOG electrodes as EEG;
                # the release-internal P0 rule must still exclude them.
                "channel_types": [
                    "EEG",
                    "EEG",
                    "EEG",
                    "EEG",
                    "EEG",
                    "EEG",
                    "LABEL",
                ],
            }
        )
    records = []
    for study, count in STUDY_COUNTS.items():
        for participant_index in range(1, count + 1):
            layout_id = STUDY_LAYOUT[study]
            # The real release has exactly one study05/layout_06 recording.
            if study == "study05" and participant_index == count:
                layout_id = "layout_06"
            records.append(
                {
                    "study": study,
                    "participant_stem": f"{study}_p{participant_index:02d}",
                    "sampling_hz": STUDY_RATE[study],
                    "channels": 7,
                    "trials": 4,
                    "samples_per_trial": int(STUDY_RATE[study] * 8),
                    "channel_layout_id": layout_id,
                    "companion_fdt_candidate": (
                        f"{study}_p{participant_index:02d}_prep.fdt"
                    ),
                    "trial_block_counts": {"1": 2, "2": 2},
                    "trial_label_counts": {"1": 1, "2": 1, "3": 1, "4": 1},
                    "trial_id_count": 0 if study == "study05" else 4,
                }
            )
    return {
        "mode": "audit-sgeyesub-structure",
        "state": "structure_read",
        "recording_count": 59,
        "study_recording_counts": STUDY_COUNTS,
        "channel_layouts": layouts,
        "recordings": records,
        "fdt_access": "companion_exists_but_not_opened_by_code",
    }


def _load_plan(tmp_path: Path):
    source = tmp_path / "structure.json"
    source.write_text(json.dumps(_structure_payload()), encoding="utf-8")
    layouts, records = load_sgeyesub_structure_audit(source)
    plan = build_sgeyesub_protocol(
        layouts,
        records,
        protocol_id="sgeyesub_release_operator_specificity_v1",
        reference_cell_id="release_preprocessed_as_delivered",
        gamma_candidates=(0.0, 0.25, 0.5, 0.75, 1.0),
    )
    return layouts, records, plan


def test_frozen_study_partition_and_block_contract(tmp_path: Path) -> None:
    _, _, plan = _load_plan(tmp_path)
    assert len(plan.development_rows) == 15
    assert len(plan.evaluation_rows) == 44
    assert {row.study for row in plan.development_rows} == {"study01", "study03"}
    assert {row.study for row in plan.evaluation_rows} == {
        "study02",
        "study04",
        "study05",
    }
    assert all(row.support_block == 1 and row.query_block == 2 for row in plan.rows)
    assert all(row.claim_scope == SGEYESUB_RELEASE_CLAIM for row in plan.rows)


def test_pi0_sources_are_other_participants_in_the_exact_cell(tmp_path: Path) -> None:
    _, records, plan = _load_plan(tmp_path)
    for row in plan.rows:
        assert row.participant_stem not in row.population_source_participants
        same_cell_rows = [
            candidate
            for candidate in plan.rows
            if (
                candidate.study,
                candidate.layout_id,
                candidate.reference_cell_id,
                candidate.sampling_rate_hz,
            )
            == (
                row.study,
                row.layout_id,
                row.reference_cell_id,
                row.sampling_rate_hz,
            )
        ]
        assert row.population_source_count == len(same_cell_rows) - 1
        assert set(row.population_source_participants) == {
            candidate.participant_stem
            for candidate in same_cell_rows
            if candidate.recording_key != row.recording_key
        }

    blocked = [row for row in plan.rows if row.status == "blocked_no_population"]
    assert len(blocked) == 1
    assert blocked[0].study == "study05"
    assert blocked[0].release_layout_id == "layout_06"
    assert blocked[0].population_source_count == 0
    study05_layouts = {row.layout_id for row in plan.rows if row.study == "study05"}
    assert study05_layouts == {"layout_05", "layout_06"}
    study05_p0_layouts = {
        record.p0_layout_id for record in records if record.study == "study05"
    }
    assert study05_p0_layouts == {"p0_layout_05", "p0_layout_06"}


def test_study05_missing_trial_ids_remain_absent(tmp_path: Path) -> None:
    _, records, plan = _load_plan(tmp_path)
    study05_records = [record for record in records if record.study == "study05"]
    assert len(study05_records) == 14
    assert all(record.trial_id_count == 0 for record in study05_records)
    assert all(
        record.trial_id_status == "absent_release_metadata_use_epoch_ordinal_only"
        for record in study05_records
    )
    assert all(
        row.trial_id_status == "absent_release_metadata_use_epoch_ordinal_only"
        for row in plan.evaluation_rows
        if row.study == "study05"
    )


def test_signal_loader_splits_before_flatten_and_can_skip_query(
    tmp_path: Path,
) -> None:
    layouts, records, _ = _load_plan(tmp_path)
    record = records[0]
    layout = next(item for item in layouts if item.layout_id == record.layout_id)
    root = tmp_path / "release"
    study_root = root / record.study
    study_root.mkdir(parents=True)

    cube = np.zeros(
        (record.channel_count, record.samples_per_trial, record.trial_count),
        dtype=np.float32,
    )
    sample_axis = np.arange(record.samples_per_trial, dtype=np.float32)
    for trial in range(record.trial_count):
        cube[0, :, trial] = sample_axis + 10_000 * trial
        cube[1, :, trial] = 2 * sample_axis + 10_000 * trial
        cube[2, :, trial] = 10 * (trial + 1)
        cube[3, :, trial] = trial + 1
        cube[4, :, trial] = 2 * (trial + 1)
        cube[5, :, trial] = 3 * (trial + 1)
        cube[6, :, trial] = (1, 5, 6, 6)[trial]
    fdt_path = study_root / Path(record.fdt_relative_path).name
    cube.ravel(order="F").astype("<f4").tofile(fdt_path)
    set_path = study_root / Path(record.set_relative_path).name
    with h5py.File(set_path, "w") as h5_file:
        etc = h5_file.create_group("EEG").create_group("etc")
        etc.create_dataset("trial_blocks", data=np.asarray([1, 1, 2, 2]))
        etc.create_dataset("trial_labels", data=np.asarray([1, 2, 3, 4]))
        etc.create_dataset("trial_ids", data=np.asarray([11, 12, 13, 14]))

    support_only = load_sgeyesub_signal_record(
        root,
        record,
        layout,
        include_query=False,
    )
    assert support_only.query is None
    assert support_only.query_annotations is None
    assert support_only.support.eeg.shape == (2, 2 * record.samples_per_trial)
    assert support_only.support.native_eeg.shape == (
        6,
        2 * record.samples_per_trial,
    )
    assert support_only.support.artifactclasses.shape == (
        2 * record.samples_per_trial,
    )
    assert not support_only.support.eeg.flags.writeable
    assert not support_only.support.artifactclasses.flags.writeable

    query_without_annotations = load_sgeyesub_signal_record(
        root,
        record,
        layout,
        include_query=True,
        include_query_annotations=False,
    )
    assert query_without_annotations.query is not None
    assert query_without_annotations.query_annotations is None

    complete = load_sgeyesub_signal_record(root, record, layout)
    assert complete.query is not None
    assert complete.query_annotations is not None
    assert complete.query.eeg.shape == (2, 2 * record.samples_per_trial)
    assert complete.query.native_eeg.shape == (
        6,
        2 * record.samples_per_trial,
    )
    np.testing.assert_array_equal(
        complete.query_annotations.artifactclasses,
        np.full(2 * record.samples_per_trial, 6, dtype=np.int64),
    )


def test_trial_labels_and_query_annotations_cannot_enter_fit() -> None:
    assert_operator_fit_fields(
        (
            "support_eeg",
            "support_external_eog",
            "support_artifactclasses",
        )
    )
    for forbidden in sorted(QUERY_EVALUATION_ONLY_FIELDS):
        with pytest.raises(ValueError, match="support-only"):
            assert_operator_fit_fields(("support_eeg", forbidden))
    with pytest.raises(ValueError, match="support-only"):
        assert_operator_fit_fields(("trial_labels_as_artifactclasses",))
    with pytest.raises(ValueError, match="support-only"):
        assert_operator_fit_fields(("support_trial_labels",))


def test_metadata_outputs_preserve_release_internal_claim(tmp_path: Path) -> None:
    _, _, plan = _load_plan(tmp_path)
    outputs = write_sgeyesub_protocol_outputs(plan, tmp_path / "out")
    summary = json.loads(Path(outputs["result_summary"]).read_text(encoding="utf-8"))
    policy = json.loads(Path(outputs["input_policy"]).read_text(encoding="utf-8"))
    layouts = json.loads(
        Path(outputs["layout_contracts"]).read_text(encoding="utf-8")
    )
    assert summary["development_participant_stems"] == 15
    assert summary["evaluation_participant_stems"] == 44
    assert summary["development_metadata_ready"] == 15
    assert summary["evaluation_metadata_ready"] == 43
    assert summary["native_replication_claim_allowed"] is False
    assert summary["cell_count"] == 6
    assert summary["blocked_singleton_cells"] == 1
    assert summary["blocked_record_count"] == 1
    assert policy["trial_labels_are_artifactclasses"] is False
    assert policy["native_input_mapping_status"] == SGEYESUB_NATIVE_INPUT_STATUS
    assert len(layouts) == 6
    assert all(
        item["native_eeg_chan_idxs_status"] == SGEYESUB_NATIVE_INPUT_STATUS
        for item in layouts
    )
    assert all(
        all(
            not (label.startswith("EOG") or label.endswith("EOG"))
            for label in item["release_internal_p0_eeg_labels"]
        )
        for item in layouts
    )
    assert all(
        {"EOG-R-Bottom", "HEOG", "VEOG", "REOG"}.issubset(
            item["native_ordered_eeg_labels"]
        )
        for item in layouts
    )

    with Path(outputs["split_manifest"]).open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 59
    assert {row["partition"] for row in rows} == {"development", "evaluation"}
    assert all(row["support_block"] == "1" for row in rows)
    assert all(row["query_block"] == "2" for row in rows)


def test_repository_config_freezes_one_development_only_gamma() -> None:
    config = yaml.safe_load(
        Path("configs/cgdr/sgeyesub_operator_specificity.yaml").read_text(
            encoding="utf-8"
        )
    )
    validate_sgeyesub_protocol_config(config)
    assert config["b6_pop_shrink"]["one_global_gamma"] is True
    assert config["b6_pop_shrink"]["gamma_candidates"] == [
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    ]
    assert config["b6_pop_shrink"]["selection_partition"] == "development"
    assert config["b6_pop_shrink"]["selection_objective"] == (
        "support_only_split_half_stability_plus_heldout_contamination_capture"
    )
    assert config["b6_pop_shrink"]["heldout_contamination_capture_weight"] == 0.5
    assert config["b6_pop_shrink"]["evaluation_gamma_policy"] == (
        "load_frozen_development_choice"
    )
    assert config["native_sgeyesub"]["eeg_chan_idxs"] == (
        "exact_layout_channel_type_EEG_official_commit_2c95b4f"
    )
    assert config["native_sgeyesub"]["implementation_status"] == (
        "source_faithful_python_port_not_numerically_cross_validated_with_matlab"
    )
    assert "native_sgeyesub_python_release_internal" in (
        config["development_runner"]["methods"]
    )
    assert config["evaluation_metrics"]["clean_waveform_rrmse"] == (
        "forbidden_no_clean_target"
    )


def test_soft_proximal_is_exact_qy_plus_tau_pi_y() -> None:
    projector = np.diag([1.0, 0.0, 0.0])
    observed = np.arange(12, dtype=np.float64).reshape(3, 4)
    tau = 0.25
    expected = (np.eye(3) - projector) @ observed + tau * projector @ observed
    np.testing.assert_array_equal(_soft_restore(projector, observed, tau), expected)


def test_global_support_only_selector_can_freeze_gamma_zero() -> None:
    scores = {
        0.0: [0.10] * 15,
        0.25: [0.12] * 15,
        0.5: [0.20] * 15,
        0.75: [0.30] * 15,
        1.0: [0.40] * 15,
    }
    selected, rows = select_global_gamma(
        scores,
        candidates=(0.0, 0.25, 0.5, 0.75, 1.0),
        participant_count=15,
        minimum_fraction=0.8,
    )
    assert selected == 0.0
    assert all(row["eligible"] for row in rows)


def test_support_only_score_uses_frozen_half_capture_weight() -> None:
    score = _support_only_composite_score(
        0.2,
        0.4,
        capture_weight=0.5,
    )
    assert score == pytest.approx(0.4)


def test_reference_free_and_observation_relative_metric_formulas() -> None:
    observed = np.arange(48, dtype=np.float64).reshape(2, 24) + 1.0
    identical = observed.copy()
    mask = np.ones(24, dtype=bool)
    assert _covariance_distortion(identical, observed, mask) == pytest.approx(0.0)
    predicted = np.ones_like(observed)
    corrected = observed - predicted
    assert _predicted_contamination_remaining(
        corrected, observed, predicted
    ) == pytest.approx(0.0)
    labels = np.asarray([1, 1, 2, 2, 3, 3, 4, 4])
    preservation, status = _condition_erp_preservation(
        identical,
        observed,
        trial_labels=labels,
        samples_per_trial=3,
        minimum_trials_per_condition=2,
    )
    assert preservation == pytest.approx(1.0)
    assert status.startswith("success_observation_relative")


def test_evaluation_loads_frozen_development_gamma_without_reselection(
    tmp_path: Path,
) -> None:
    config = yaml.safe_load(
        Path("configs/cgdr/sgeyesub_operator_specificity.yaml").read_text(
            encoding="utf-8"
        )
    )
    development_root = tmp_path / "development"
    development_root.mkdir()
    config["development_output_root"] = str(development_root)
    frozen_path = development_root / "frozen_gamma.json"
    frozen_path.write_text(
        json.dumps(
            {
                "status": "frozen",
                "gamma": 0.0,
                "query_annotations_used": False,
            }
        ),
        encoding="utf-8",
    )
    assert _load_frozen_development_gamma(config) == 0.0

    frozen_path.write_text(
        json.dumps(
            {
                "status": "frozen",
                "gamma": 0.25,
                "query_annotations_used": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="forbidden query"):
        _load_frozen_development_gamma(config)


def test_gamma_zero_automatically_stops_personalization() -> None:
    config = yaml.safe_load(
        Path("configs/cgdr/sgeyesub_operator_specificity.yaml").read_text(
            encoding="utf-8"
        )
    )
    decision = _operator_specificity_decision(
        [{"recording_key": "study02/p01", "method_id": "raw"}],
        frozen_gamma=0.0,
        config=config,
    )
    assert decision["decision"] == "personalization_failed_population_deterministic"


def _specificity_rows(*, failed_b6_participants: int = 0) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index in range(10):
        recording_key = f"study02/p{index:02d}"
        rows.append(
            {
                "recording_key": recording_key,
                "method_id": "B6_Qy__gamma_0p5",
                "status": "fallback_POP" if index < failed_b6_participants else "success",
                "heldout_eog_prediction_remaining_ratio": "0.1",
                "nonartifact_observation_preservation": "0.95",
                "reference_free_covariance_distortion": "0.1",
            }
        )
        for control in ("pop_Qy", "matching_Qy", "wrong_Qy", "shuffled_Qy"):
            rows.append(
                {
                    "recording_key": recording_key,
                    "method_id": control,
                    "status": "success",
                    "heldout_eog_prediction_remaining_ratio": "0.2",
                }
            )
    return rows


def test_positive_gamma_requires_all_four_controls_and_safety() -> None:
    config = yaml.safe_load(
        Path("configs/cgdr/sgeyesub_operator_specificity.yaml").read_text(
            encoding="utf-8"
        )
    )

    supported = _operator_specificity_decision(
        _specificity_rows(), frozen_gamma=0.5, config=config
    )
    assert supported["decision"] == "b6_participant_specificity_supported"
    assert set(supported["comparisons"]) == {
        "pop_Qy",
        "matching_Qy",
        "wrong_Qy",
        "shuffled_Qy",
    }

    failed = _operator_specificity_decision(
        _specificity_rows(failed_b6_participants=4),
        frozen_gamma=0.5,
        config=config,
    )
    assert failed["decision"] == "personalization_failed_population_deterministic"
    assert failed["failures_and_fallbacks_retained_in_denominator"] is True
