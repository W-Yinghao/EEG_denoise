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
    _absolute_safety_summary,
    _condition_erp_preservation,
    _covariance_distortion,
    _development_gamma_component_audit,
    _gamma_support_score_row,
    _load_frozen_development_gamma,
    _matching_population_audit,
    _method_coverage_summary,
    _method_summary,
    _operator_specificity_decision,
    _predicted_contamination_remaining,
    _required_method_record_coverage,
    _required_evaluation_method_ids,
    _soft_restore,
    _support_only_composite_score,
    _validate_evaluation_record_artifacts,
    _write_corrected_audit,
    _write_resolved_config,
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
        cube[6, :, trial] = (0, 5, 6, 6)[trial]
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
    assert 0 in support_only.support.artifactclasses

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

    # Query artifactclass samples are not touched by the legal-query-EEG path.
    cube[6, :, 2:] = np.nan
    cube.ravel(order="F").astype("<f4").tofile(fdt_path)
    isolated_signal = load_sgeyesub_signal_record(
        root,
        record,
        layout,
        include_query=True,
        include_query_annotations=False,
    )
    assert isolated_signal.query is not None
    assert isolated_signal.query_annotations is None
    with pytest.raises(ValueError, match="artifactclasses contains NaN"):
        load_sgeyesub_signal_record(
            root,
            record,
            layout,
            include_query=True,
            include_query_annotations=True,
        )
    cube[6, :, 2:] = 6
    cube.ravel(order="F").astype("<f4").tofile(fdt_path)

    # Corrupt only query annotations.  Support-only fitting and legal query EEG
    # loading must remain available; the delayed annotation read must reject it.
    with h5py.File(set_path, "r+") as h5_file:
        etc = h5_file["EEG/etc"]
        del etc["trial_labels"]
        etc.create_dataset(
            "trial_labels", data=np.asarray([1.0, 2.0, np.nan, np.nan])
        )
        del etc["trial_ids"]
        etc.create_dataset("trial_ids", data=np.asarray([11, 12]))
    isolated = load_sgeyesub_signal_record(
        root,
        record,
        layout,
        include_query=True,
        include_query_annotations=False,
    )
    assert isolated.query is not None
    assert isolated.query_annotations is None
    np.testing.assert_array_equal(isolated.support.trial_labels, [1, 2])
    with pytest.raises(ValueError, match="selected trial_"):
        load_sgeyesub_signal_record(
            root,
            record,
            layout,
            include_query=True,
            include_query_annotations=True,
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
    assert policy["query_annotations_for_fit_gamma_or_method_selection"] == (
        "forbidden"
    )
    assert policy["query_annotations_for_reporting"] == (
        "allowed_after_all_method_outputs_frozen"
    )
    assert policy["query_annotations_for_single_final_automatic_decision"] == (
        "allowed_without_adaptation_reselection_or_method_change"
    )
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
    assert config["input_policy"][
        "query_annotations_for_fit_gamma_or_method_selection"
    ] == "forbidden"
    assert config["input_policy"][
        "query_annotations_for_single_final_automatic_decision"
    ] == "allowed_without_adaptation_reselection_or_method_change"
    assert "external_query_eog_regression" not in config["development_runner"][
        "methods"
    ]
    assert config["development_runner"]["query_annotation_opening"] == (
        "after_all_method_outputs_frozen"
    )
    assert config["development_runner"]["query_eog_method_exception"] == (
        "forbidden"
    )
    assert config["corrected_audit"]["expected_compatible_records"] == 43
    assert config["corrected_audit"]["bootstrap_replicates"] == 20_000
    assert config["corrected_audit"]["bootstrap_seed"] == 20260802
    assert config["corrected_audit"]["output_root"] == (
        "results/cgdr/sgeyesub_operator_specificity_corrected_audit"
    )
    assert config["corrected_audit"]["report_path"] == (
        "reports/cgdr_sgeyesub_corrected_audit.md"
    )
    assert config["operator_specificity_decision"]["gamma_zero_decision"] == (
        "development_selected_population_endpoint"
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


def test_gamma_score_exposes_components_and_structural_zero() -> None:
    row = _gamma_support_score_row(
        gamma=0.0,
        status="success",
        stability=0.0,
        capture_loss=0.4,
        capture_weight=0.5,
        support_score=0.2,
    )
    assert row["weighted_capture_component"] == pytest.approx(0.2)
    assert row["structural_zero_stability"] is True
    assert "endpoint property" in row["structural_zero_explanation"]
    with pytest.raises(ValueError, match="registered components"):
        _gamma_support_score_row(
            gamma=0.25,
            status="success",
            stability=0.1,
            capture_loss=0.4,
            capture_weight=0.5,
            support_score=9.0,
        )


def _write_gamma_score_fixtures(
    config, protocol_rows, development_root: Path
) -> None:
    config["development_output_root"] = str(development_root)
    candidates = config["b6_pop_shrink"]["gamma_candidates"]
    for protocol_row in protocol_rows:
        participant_root = development_root / protocol_row.participant_stem
        participant_root.mkdir(parents=True)
        payload = []
        for gamma in candidates:
            stability = float(gamma)
            capture = 0.4
            payload.append(
                {
                    "gamma": gamma,
                    "status": "success",
                    "split_half_stability": stability,
                    "heldout_contamination_capture_loss": capture,
                    "capture_weight": 0.5,
                    "support_score": stability + 0.5 * capture,
                }
            )
        (participant_root / "support_gamma_scores.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )


def test_development_gamma_component_audit_reads_existing_scores(
    tmp_path: Path,
) -> None:
    _, _, plan = _load_plan(tmp_path)
    config = yaml.safe_load(
        Path("configs/cgdr/sgeyesub_operator_specificity.yaml").read_text(
            encoding="utf-8"
        )
    )
    _write_gamma_score_fixtures(
        config, plan.development_rows, tmp_path / "development"
    )
    audit = _development_gamma_component_audit(config, plan.development_rows)
    assert len(audit["component_rows"]) == 15 * 5
    gamma_zero = next(
        row for row in audit["summary_rows"] if float(row["gamma"]) == 0.0
    )
    assert gamma_zero["successful_component_count"] == 15
    assert gamma_zero["mean_split_half_stability"] == pytest.approx(0.0)
    assert gamma_zero["mean_weighted_capture_component"] == pytest.approx(0.2)
    assert gamma_zero["mean_support_score"] == pytest.approx(0.2)
    assert gamma_zero["structural_zero_stability"] is True


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


def test_sgeyesub_aggregate_writes_one_shared_resolved_config(tmp_path: Path) -> None:
    config = yaml.safe_load(
        Path("configs/cgdr/sgeyesub_operator_specificity.yaml").read_text(
            encoding="utf-8"
        )
    )
    config["development_output_root"] = str(tmp_path / "protocol" / "development")
    config["evaluation_output_root"] = str(tmp_path / "protocol" / "evaluation")

    path = _write_resolved_config(config)

    assert path == tmp_path / "protocol" / "resolved_config.yaml"
    written = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert written["protocol_id"] == config["protocol_id"]


def test_gamma_zero_is_population_endpoint_and_evaluation_continues() -> None:
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
    assert decision["decision"] == "development_selected_population_endpoint"
    assert decision["evaluation_continues"] is True
    assert decision["next_route"] == "continue_frozen_population_endpoint_evaluation"
    assert "structurally zero" in decision["structural_zero_explanation"]


def test_method_performance_excludes_fallback_blocked_and_ineligible() -> None:
    rows = [
        {
            "method_id": "matching_Qy",
            "status": "success",
            "fallback_used": False,
            "heldout_eog_prediction_remaining_ratio": "0.2",
        },
        {
            "method_id": "matching_Qy",
            "status": "fallback_POP",
            "fallback_used": "True",
            "heldout_eog_prediction_remaining_ratio": "100.0",
        },
        {
            "method_id": "matching_Qy",
            "status": "blocked_no_population_identity_no_claim",
            "fallback_used": False,
            "heldout_eog_prediction_remaining_ratio": "200.0",
        },
        {
            "method_id": "matching_Qy",
            "status": "ineligible_matching_P0_identity_no_claim",
            "fallback_used": False,
            "heldout_eog_prediction_remaining_ratio": "300.0",
        },
    ]
    performance = _method_summary(rows, partition="evaluation")
    primary = next(
        row
        for row in performance
        if row["metric"] == "heldout_eog_prediction_remaining_ratio"
    )
    assert primary["participant_stem_count"] == 1
    assert primary["mean"] == pytest.approx(0.2)
    coverage = _method_coverage_summary(rows, partition="evaluation")
    assert coverage[0]["record_count"] == 4
    assert coverage[0]["success_count"] == 1
    assert coverage[0]["fallback_count"] == 1
    assert coverage[0]["blocked_count"] == 1
    assert coverage[0]["ineligible_count"] == 1


def _matching_population_rows(plan, *, fallback_first: bool = False):
    rows = []
    compatible = [row for row in plan.evaluation_rows if row.status == "metadata_ready"]
    for index, protocol_row in enumerate(compatible):
        rows.extend(
            [
                {
                    "study": protocol_row.study,
                    "participant_stem": protocol_row.participant_stem,
                    "recording_key": protocol_row.recording_key,
                    "method_id": "matching_Qy",
                    "status": "fallback_POP" if fallback_first and index == 0 else "success",
                    "fallback_used": fallback_first and index == 0,
                    "heldout_eog_prediction_remaining_ratio": "1.0",
                    "nonartifact_observation_preservation": "0.95",
                    "reference_free_covariance_distortion": "0.1",
                },
                {
                    "study": protocol_row.study,
                    "participant_stem": protocol_row.participant_stem,
                    "recording_key": protocol_row.recording_key,
                    "method_id": "pop_Qy",
                    "status": "success",
                    "fallback_used": False,
                    "heldout_eog_prediction_remaining_ratio": "2.0",
                    "nonartifact_observation_preservation": "0.95",
                    "reference_free_covariance_distortion": "0.1",
                },
                {
                    "study": protocol_row.study,
                    "participant_stem": protocol_row.participant_stem,
                    "recording_key": protocol_row.recording_key,
                    "method_id": "wrong_Qy",
                    "status": "success",
                    "fallback_used": False,
                    "heldout_eog_prediction_remaining_ratio": "1.5",
                    "nonartifact_observation_preservation": "0.95",
                    "reference_free_covariance_distortion": "0.1",
                },
                {
                    "study": protocol_row.study,
                    "participant_stem": protocol_row.participant_stem,
                    "recording_key": protocol_row.recording_key,
                    "method_id": "shuffled_Qy",
                    "status": "success",
                    "fallback_used": False,
                    "heldout_eog_prediction_remaining_ratio": "1.75",
                    "nonartifact_observation_preservation": "0.95",
                    "reference_free_covariance_distortion": "0.1",
                },
                {
                    "study": protocol_row.study,
                    "participant_stem": protocol_row.participant_stem,
                    "recording_key": protocol_row.recording_key,
                    "method_id": "B6_Qy__gamma_0",
                    "status": "success",
                    "fallback_used": False,
                    "heldout_eog_prediction_remaining_ratio": "2.0",
                    "nonartifact_observation_preservation": "0.95",
                    "reference_free_covariance_distortion": "0.1",
                },
                {
                    "study": protocol_row.study,
                    "participant_stem": protocol_row.participant_stem,
                    "recording_key": protocol_row.recording_key,
                    "method_id": "B6_soft_proximal__gamma_0",
                    "status": "success",
                    "fallback_used": False,
                    "heldout_eog_prediction_remaining_ratio": "1.25",
                    "nonartifact_observation_preservation": "0.96",
                    "reference_free_covariance_distortion": "0.09",
                },
                {
                    "study": protocol_row.study,
                    "participant_stem": protocol_row.participant_stem,
                    "recording_key": protocol_row.recording_key,
                    "method_id": "native_sgeyesub_python_release_internal",
                    "status": "success_source_faithful_not_matlab_cross_validated",
                    "fallback_used": False,
                    "heldout_eog_prediction_remaining_ratio": "0.5",
                    "nonartifact_observation_preservation": "0.97",
                    "reference_free_covariance_distortion": "0.08",
                },
            ]
        )
    singleton = [
        row for row in plan.evaluation_rows if row.status != "metadata_ready"
    ]
    assert len(singleton) == 1
    protocol_row = singleton[0]
    for method_id, status in (
        ("matching_Qy", "success"),
        ("pop_Qy", "blocked_no_population_identity_no_claim"),
        ("wrong_Qy", "blocked_no_population_identity_no_claim"),
        ("shuffled_Qy", "ineligible_shuffled_identity_no_claim"),
        ("B6_Qy__gamma_0", "blocked_no_population_identity_no_claim"),
        (
            "B6_soft_proximal__gamma_0",
            "blocked_no_population_identity_no_claim",
        ),
        (
            "native_sgeyesub_python_release_internal",
            "success_source_faithful_not_matlab_cross_validated",
        ),
    ):
        row = {
            "study": protocol_row.study,
            "participant_stem": protocol_row.participant_stem,
            "recording_key": protocol_row.recording_key,
            "method_id": method_id,
            "status": status,
            "fallback_used": False,
        }
        if status.startswith("success"):
            row.update(
                {
                    "heldout_eog_prediction_remaining_ratio": "1.0",
                    "nonartifact_observation_preservation": "0.95",
                    "reference_free_covariance_distortion": "0.1",
                }
            )
        rows.append(row)
    return rows


def test_corrected_matching_population_audit_and_outputs(tmp_path: Path) -> None:
    _, _, plan = _load_plan(tmp_path)
    rows = _matching_population_rows(plan)
    audit = _matching_population_audit(
        rows,
        plan.evaluation_rows,
        bootstrap_replicates=20_000,
        bootstrap_seed=20260802,
        metric_directions=(("heldout_eog_prediction_remaining_ratio", "lower"),),
    )
    assert audit["status"] == "complete_43_success_paired"
    assert audit["compatible_record_count"] == 43
    summary = audit["summary_rows"][0]
    assert summary["finite_metric_paired_count"] == 43
    assert summary["mean_matching_minus_population"] == pytest.approx(-1.0)
    assert summary["median_matching_minus_population"] == pytest.approx(-1.0)
    assert summary["mean_directional_improvement"] == pytest.approx(1.0)
    assert summary["median_directional_improvement"] == pytest.approx(1.0)
    assert summary["matching_wins"] == 43
    assert summary["mean_matching_minus_population_ci95_low"] == pytest.approx(-1.0)
    assert summary["mean_matching_minus_population_ci95_high"] == pytest.approx(-1.0)
    assert summary["mean_directional_improvement_ci95_low"] == pytest.approx(1.0)
    by_study = {
        row["study"]: row["compatible_record_count"]
        for row in audit["heterogeneity_rows"]
    }
    assert by_study == {"study02": 15, "study04": 15, "study05": 13}

    required_method_ids = (
        "matching_Qy",
        "pop_Qy",
        "B6_Qy__gamma_0",
        "B6_soft_proximal__gamma_0",
        "native_sgeyesub_python_release_internal",
        "wrong_Qy",
        "shuffled_Qy",
    )
    required_coverage = _required_method_record_coverage(
        rows,
        plan.evaluation_rows,
        method_ids=required_method_ids,
    )
    assert required_coverage["status"] == "complete_44_unique_recording_keys"
    assert required_coverage["all_required_methods_complete"] is True
    assert all(
        row["row_count"] == 44
        and row["unique_recording_key_count"] == 44
        and row["complete_44_unique_recording_keys"] is True
        for row in required_coverage["methods"]
    )
    missing_singleton = [
        row
        for row in rows
        if not (
            row["method_id"] == "native_sgeyesub_python_release_internal"
            and row["recording_key"] == audit["blocked_singleton_recording_key"]
        )
    ]
    incomplete_required_coverage = _required_method_record_coverage(
        missing_singleton,
        plan.evaluation_rows,
        method_ids=required_method_ids,
    )
    assert incomplete_required_coverage["all_required_methods_complete"] is False
    assert incomplete_required_coverage["status"] == (
        "inconclusive_incomplete_required_method_record_coverage"
    )

    incomplete = _matching_population_audit(
        _matching_population_rows(plan, fallback_first=True),
        plan.evaluation_rows,
        bootstrap_replicates=20_000,
        bootstrap_seed=20260802,
        metric_directions=(("heldout_eog_prediction_remaining_ratio", "lower"),),
    )
    assert incomplete["status"] == "inconclusive_incomplete_success_pair_coverage"
    assert incomplete["method_success_paired_count"] == 42

    config = yaml.safe_load(
        Path("configs/cgdr/sgeyesub_operator_specificity.yaml").read_text(
            encoding="utf-8"
        )
    )
    config["corrected_audit"]["output_root"] = str(tmp_path / "corrected")
    config["corrected_audit"]["report_path"] = str(tmp_path / "corrected.md")
    _write_gamma_score_fixtures(
        config, plan.development_rows, tmp_path / "gamma_development"
    )
    written = _write_corrected_audit(
        config,
        protocol_rows=plan.evaluation_rows,
        rows=rows,
        frozen_gamma=0.0,
        development_rows=plan.development_rows,
    )
    assert (tmp_path / "corrected" / "resolved_config.yaml").is_file()
    assert Path(written["paths"]["method_coverage"]).is_file()
    assert Path(written["paths"]["absolute_safety"]).is_file()
    assert Path(
        written["paths"]["required_focus_method_metric_status"]
    ).is_file()
    assert Path(written["paths"]["control_method_metric_status"]).is_file()
    assert Path(written["paths"]["hard_q_absolute_safety"]).is_file()
    assert Path(
        written["paths"]["development_gamma_score_components"]
    ).is_file()
    assert Path(written["paths"]["development_gamma_score_summary"]).is_file()
    report = (tmp_path / "corrected.md").read_text(encoding="utf-8")
    assert "development_selected_population_endpoint" in report
    assert "Development gamma score components" in report
    assert "hard_Q_P0_tradeoff_inconclusive" in report
    assert "post-hoc descriptive audit, is non-preregistered" in report
    assert "conservative selection rule" in report
    assert "not an unbiased hypothesis test of personalization" in report
    assert "covariance/PSD distortion were roughly tied" in report
    assert "Required methods: nine metrics and status coverage" in report
    assert "Hard-Q absolute safety" in report
    assert "| study02 |" in report
    assert "| study04 |" in report
    assert "| study05 |" in report
    for method_id in (
        "matching_Qy",
        "pop_Qy",
        "B6_Qy__gamma_0",
        "B6_soft_proximal__gamma_0",
        "native_sgeyesub_python_release_internal",
        "wrong_Qy",
        "shuffled_Qy",
    ):
        assert f"| {method_id} |" in report
    assert written["scientific_interpretation"] == (
        "hard_Q_P0_tradeoff_inconclusive"
    )
    assert written["audit_scope"] == (
        "post_hoc_descriptive_audit_non_preregistered"
    )
    assert written["formal_gate_evidence"] is False
    assert written["formal_operator_specificity_decision"] == (
        "not_generated_post_hoc_audit"
    )
    assert written["descriptive_pattern"] == {
        "matching_heldout_eog_remaining": "post_hoc_lower_than_population",
        "matching_eog_coherence_reduction": "post_hoc_higher_than_population",
        "matching_nonartifact_preservation": "roughly_tied_ci_spans_zero",
        "matching_covariance_psd_distortion": "roughly_tied_ci_spans_zero",
        "matching_erp_preservation_proxy": "post_hoc_lower_than_population",
        "absolute_hard_q_safety_thresholds": "not_met",
    }
    assert written["status"] == "complete_43_success_paired"
    assert written["matching_population_pair_status"] == (
        "complete_43_success_paired"
    )
    assert written["required_method_record_coverage_status"] == (
        "complete_44_unique_recording_keys"
    )
    assert set(written["required_focus_method_ids"]) == {
        "matching_Qy",
        "pop_Qy",
        "B6_Qy__gamma_0",
        "B6_soft_proximal__gamma_0",
        "native_sgeyesub_python_release_internal",
    }
    assert len(written["required_focus_method_metric_status"]) == 5 * 9
    assert all(
        row["registered_record_count"] == 44
        for row in written["required_focus_method_metric_status"]
        + written["control_method_metric_status"]
    )
    focus_remaining = [
        row
        for row in written["required_focus_method_metric_status"]
        if row["metric"] == "heldout_eog_prediction_remaining_ratio"
    ]
    assert len(focus_remaining) == 5
    assert all(row["performance_available"] for row in focus_remaining)
    assert {row["method_id"] for row in written["hard_q_absolute_safety"]} == {
        "matching_Qy",
        "pop_Qy",
        "B6_Qy__gamma_0",
        "wrong_Qy",
        "shuffled_Qy",
    }
    assert all(
        row["bootstrap_replicates"] == 20_000
        and row["bootstrap_seed"] == 20260802
        for row in focus_remaining
    )

    safety = _absolute_safety_summary(rows, config=config)
    matching_safety = next(row for row in safety if row["method_id"] == "matching_Qy")
    assert matching_safety["finite_joint_safety_count"] == 44
    assert matching_safety["joint_safety_pass_count"] == 44


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

    insufficient = _operator_specificity_decision(
        _specificity_rows(failed_b6_participants=4),
        frozen_gamma=0.5,
        config=config,
    )
    assert insufficient["decision"] == "inconclusive_insufficient_finite_pairs"
    assert insufficient["failures_and_fallbacks_retained_in_denominator"] is True

    negative_rows = _specificity_rows()
    for row in negative_rows:
        if row["method_id"] == "B6_Qy__gamma_0p5":
            row["heldout_eog_prediction_remaining_ratio"] = "0.3"
    failed = _operator_specificity_decision(
        negative_rows,
        frozen_gamma=0.5,
        config=config,
    )
    assert failed["decision"] == (
        "frozen_b6_specificity_not_supported_under_tested_protocol"
    )
    assert failed["reason"] == "frozen_b6_improvement_or_safety_threshold_not_met"
    assert failed["next_route"] == (
        "stop_frozen_b6_route_retain_population_endpoint"
    )


def _evaluation_artifacts(protocol_row, *, gamma: float):
    summary = {
        "status": "completed",
        "partition": "evaluation",
        "study": protocol_row.study,
        "participant_stem": protocol_row.participant_stem,
        "recording_key": protocol_row.recording_key,
        "release_layout_id": protocol_row.release_layout_id,
        "frozen_development_gamma": gamma,
        "cross_layout_pooling_used": False,
        "population_source_count": protocol_row.population_source_count,
        "population_status": "available",
    }
    rows = []
    for method_id in _required_evaluation_method_ids(gamma):
        rows.append(
            {
                "partition": "evaluation",
                "study": protocol_row.study,
                "participant_stem": protocol_row.participant_stem,
                "recording_key": protocol_row.recording_key,
                "release_layout_id": protocol_row.release_layout_id,
                "support_block": str(protocol_row.support_block),
                "query_block": str(protocol_row.query_block),
                "population_source_count": str(protocol_row.population_source_count),
                "method_id": method_id,
                "status": "success",
            }
        )
    return summary, rows


def test_evaluation_artifacts_reject_duplicates_crosswiring_and_unblocked_singleton(
    tmp_path: Path,
) -> None:
    _, _, plan = _load_plan(tmp_path)
    regular = next(row for row in plan.evaluation_rows if row.status == "metadata_ready")
    summary, rows = _evaluation_artifacts(regular, gamma=0.5)
    _validate_evaluation_record_artifacts(
        regular, summary, rows, frozen_gamma=0.5
    )

    failed_population_summary, failed_population_rows = _evaluation_artifacts(
        regular, gamma=0.5
    )
    failed_population_summary["status"] = (
        "completed_with_failed_population_operator"
    )
    failed_population_summary["population_status"] = "failed_population_operator"
    for row in failed_population_rows:
        if row["method_id"] in {
            "pop_Qy",
            "POP_fallback",
            "B6_Qy__gamma_0p5",
            "B6_soft_proximal__gamma_0p5",
        }:
            row["status"] = "failed_population_operator_identity_no_claim"
    _validate_evaluation_record_artifacts(
        regular,
        failed_population_summary,
        failed_population_rows,
        frozen_gamma=0.5,
    )

    with pytest.raises(ValueError, match="duplicate"):
        _validate_evaluation_record_artifacts(
            regular, summary, [*rows, dict(rows[0])], frozen_gamma=0.5
        )
    crosswired = [dict(row) for row in rows]
    crosswired[0]["recording_key"] = "study02/wrong"
    with pytest.raises(ValueError, match="cross-wired"):
        _validate_evaluation_record_artifacts(
            regular, summary, crosswired, frozen_gamma=0.5
        )
    incomplete_population_rows = [dict(row) for row in rows]
    next(
        row for row in incomplete_population_rows if row["method_id"] == "pop_Qy"
    )["status"] = "fallback_POP"
    with pytest.raises(ValueError, match="population operator rows"):
        _validate_evaluation_record_artifacts(
            regular,
            summary,
            incomplete_population_rows,
            frozen_gamma=0.5,
        )

    singleton = next(
        row for row in plan.evaluation_rows if row.status == "blocked_no_population"
    )
    singleton_summary, singleton_rows = _evaluation_artifacts(singleton, gamma=0.5)
    singleton_summary["status"] = "completed_with_blocked_no_population"
    singleton_summary["population_status"] = "blocked_no_population"
    blocked_methods = {
        "pop_Qy",
        "POP_fallback",
        "wrong_Qy",
        "B6_Qy__gamma_0p5",
        "B6_soft_proximal__gamma_0p5",
    }
    for row in singleton_rows:
        if row["method_id"] in blocked_methods:
            row["status"] = "blocked_no_population_identity_no_claim"
    _validate_evaluation_record_artifacts(
        singleton, singleton_summary, singleton_rows, frozen_gamma=0.5
    )
    next(row for row in singleton_rows if row["method_id"] == "pop_Qy")[
        "status"
    ] = "success"
    with pytest.raises(ValueError, match="not blocked"):
        _validate_evaluation_record_artifacts(
            singleton, singleton_summary, singleton_rows, frozen_gamma=0.5
        )
