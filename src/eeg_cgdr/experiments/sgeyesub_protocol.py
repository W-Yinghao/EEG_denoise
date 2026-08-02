"""One-CPU-job SGEYESUB release-protocol metadata entry point."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Mapping

from eeg_cgdr.data.sgeyesub import (
    QUERY_EVALUATION_ONLY_FIELDS,
    SGEYESUB_DEVELOPMENT_STUDIES,
    SGEYESUB_EVALUATION_STUDIES,
    SGEYESUB_NATIVE_INPUT_STATUS,
    SGEYESUB_QUERY_BLOCK,
    SGEYESUB_RELEASE_CLAIM,
    SGEYESUB_SUPPORT_BLOCK,
    assert_operator_fit_fields,
    build_sgeyesub_protocol,
    load_sgeyesub_structure_audit,
    write_sgeyesub_protocol_outputs,
)


def _mapping(config: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"SGEYESUB config section {key!r} must be a mapping")
    return value


def validate_sgeyesub_protocol_config(config: Mapping[str, object]) -> None:
    """Keep protocol choices explicit before the metadata job touches a result."""

    if config.get("protocol_id") != "sgeyesub_release_operator_specificity_v1":
        raise ValueError("unexpected SGEYESUB protocol_id")
    if config.get("claim_scope") != SGEYESUB_RELEASE_CLAIM:
        raise ValueError("SGEYESUB claim scope must remain release-internal")

    split = _mapping(config, "split")
    if tuple(split.get("development_studies", ())) != SGEYESUB_DEVELOPMENT_STUDIES:
        raise ValueError("SGEYESUB development studies must remain study01/study03")
    if tuple(split.get("evaluation_studies", ())) != SGEYESUB_EVALUATION_STUDIES:
        raise ValueError("SGEYESUB evaluation studies must remain study02/04/05")
    if split.get("support_block") != SGEYESUB_SUPPORT_BLOCK:
        raise ValueError("SGEYESUB support must be release block 1")
    if split.get("query_block") != SGEYESUB_QUERY_BLOCK:
        raise ValueError("SGEYESUB query must be release block 2")
    if split.get("cross_study_identity_linkage") != "unresolved_no_pooling":
        raise ValueError("cross-study participant identity must not be guessed")

    cell = _mapping(config, "compatibility_cell")
    expected_fields = (
        "study",
        "layout_id",
        "reference_cell_id",
        "sampling_rate_hz",
    )
    if tuple(cell.get("fields", ())) != expected_fields:
        raise ValueError("population cells must use the exact frozen fields")
    if cell.get("layout_semantics") != "exact_ordered_full_release_channel_layout":
        raise ValueError("population cells must retain the exact full release layout")
    reference_id = cell.get("reference_cell_id")
    if not isinstance(reference_id, str) or not reference_id:
        raise ValueError("reference_cell_id must be explicit")
    if cell.get("reference_semantics") != "as_delivered_value_unresolved":
        raise ValueError("SGEYESUB physical reference must not be guessed")
    if cell.get("cross_cell_population_pooling") != "forbidden":
        raise ValueError("population projector cannot pool across exact cells")
    if cell.get("singleton_cell_action") != "blocked_no_population_operator":
        raise ValueError("a singleton exact cell must remain blocked")

    population = _mapping(config, "population_operator")
    if population.get("fit_scope") != "same_cell_other_participants_block1_only":
        raise ValueError("Pi0 must use same-cell other-participant support only")
    if population.get("leave_one_participant_out") is not True:
        raise ValueError("Pi0 must exclude the current participant")
    assert_operator_fit_fields(population.get("visible_fit_fields", ()))
    if list(population.get("query_fields_visible", ())) != []:
        raise ValueError("population operator cannot see query fields")
    if tuple(population.get("support_metadata_only_fields", ())) != (
        "support_trial_labels",
        "support_trial_ids",
    ):
        raise ValueError("trial labels/IDs must remain support metadata only")

    p0 = _mapping(config, "p0")
    if tuple(p0.get("visible_fit_fields", ())) != (
        "support_eeg",
        "support_external_eog",
    ):
        raise ValueError("P0 may fit only support EEG and external EOG")
    assert_operator_fit_fields(p0.get("visible_fit_fields", ()))

    b6 = _mapping(config, "b6_pop_shrink")
    if b6.get("family") != "B6_POP_SHRINK":
        raise ValueError("only B6 POP-SHRINK is allowed in this protocol")
    if b6.get("one_global_gamma") is not True:
        raise ValueError("B6 must freeze one global gamma")
    if tuple(float(value) for value in b6.get("gamma_candidates", ())) != (
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    ):
        raise ValueError("B6 gamma candidates must include both endpoints")
    if b6.get("endpoint_controls_are_actual_candidates") is not True:
        raise ValueError("B6 endpoints must be actual selectable candidates")
    if b6.get("selection_partition") != "development":
        raise ValueError("B6 gamma may be selected only on development studies")
    if b6.get("selection_objective") != (
        "support_only_split_half_stability_plus_heldout_contamination_capture"
    ):
        raise ValueError("B6 gamma selection must use the frozen support-only score")
    if float(b6.get("heldout_contamination_capture_weight", -1.0)) != 0.5:
        raise ValueError("B6 held-out support capture weight must remain 0.5")
    if b6.get("query_annotations_visible_to_gamma_selection") is not False:
        raise ValueError("B6 gamma selection cannot see query annotations")
    if b6.get("evaluation_gamma_policy") != "load_frozen_development_choice":
        raise ValueError("evaluation may not refit B6 gamma")

    inputs = _mapping(config, "input_policy")
    if set(inputs.get("query_evaluation_only_fields", ())) != set(
        QUERY_EVALUATION_ONLY_FIELDS
    ):
        raise ValueError("query annotation isolation policy is incomplete")
    if inputs.get("query_annotations_for_fit_gamma_or_method_selection") != "forbidden":
        raise ValueError(
            "query annotations cannot influence fitting, gamma, or method selection"
        )
    if inputs.get("query_annotations_for_reporting") != (
        "allowed_after_all_method_outputs_frozen"
    ):
        raise ValueError("query annotations may be opened only after outputs freeze")
    if inputs.get("query_annotations_for_single_final_automatic_decision") != (
        "allowed_without_adaptation_reselection_or_method_change"
    ):
        raise ValueError("query annotations permit only one non-adaptive final decision")
    if inputs.get("trial_labels_are_artifactclasses") is not False:
        raise ValueError("trial_labels cannot be used as native artifactclasses")
    if inputs.get("native_input_mapping_status") != SGEYESUB_NATIVE_INPUT_STATUS:
        raise ValueError("native eeg_chan_idxs must use the frozen official type rule")
    if inputs.get("study05_missing_trial_ids") != "preserve_absent_no_fabrication":
        raise ValueError("study05 trial IDs cannot be fabricated")

    metrics = _mapping(config, "evaluation_metrics")
    if metrics.get("minimum_query_trials_per_condition") != 2:
        raise ValueError("condition ERP proxy requires two trials per class")
    if metrics.get("clean_waveform_rrmse") != "forbidden_no_clean_target":
        raise ValueError("SGEYESUB has no clean target for waveform RRMSE")

    metadata = _mapping(config, "metadata")
    if metadata.get("open_fdt_signal_payload") is not False:
        raise ValueError("metadata stage must not open FDT signal payloads")
    if metadata.get("official_study_mapping_status") != "unresolved":
        raise ValueError("official study-to-paper mapping is unresolved")
    if metadata.get("native_replication_claim_allowed") is not False:
        raise ValueError("metadata stage cannot claim native replication")

    native = _mapping(config, "native_sgeyesub")
    if tuple(native.get("visible_fit_fields", ())) != (
        "support_native_eeg",
        "support_artifactclasses",
    ):
        raise ValueError("native SGEYESUB must use sample-wise artifactclasses")
    if native.get("trial_labels_input") != "forbidden_not_native_artifactclasses":
        raise ValueError("trial_labels must not replace native artifactclasses")
    if native.get("eeg_chan_idxs") != (
        "exact_layout_channel_type_EEG_official_commit_2c95b4f"
    ):
        raise ValueError("native SGEYESUB EEG indices must use the official type rule")
    if native.get("implementation_status") != (
        "source_faithful_python_port_not_numerically_cross_validated_with_matlab"
    ):
        raise ValueError("native port equivalence status must remain explicit")
    assert_operator_fit_fields(native.get("visible_fit_fields", ()))

    development = _mapping(config, "development_runner")
    if "external_query_eog_regression" in tuple(development.get("methods", ())):
        raise ValueError("query EOG cannot create a held-out method output")
    if development.get("query_annotation_opening") != (
        "after_all_method_outputs_frozen"
    ):
        raise ValueError("query annotations must remain closed until outputs freeze")
    if tuple(development.get("query_annotation_uses", ())) != (
        "heldout_metric_scoring",
        "one_final_automatic_decision",
    ):
        raise ValueError("query annotations may only score and make the final decision")
    if development.get("query_eog_method_exception") != "forbidden":
        raise ValueError("query EOG method-output exceptions are forbidden")

    evaluation = _mapping(config, "evaluation_runner")
    if evaluation.get("task_count") != 44:
        raise ValueError("SGEYESUB evaluation must retain all 44 release stems")
    if evaluation.get("gamma_source") != (
        "development_output_root/frozen_gamma.json"
    ):
        raise ValueError("evaluation gamma must come from development")
    if evaluation.get("gamma_refit") != "forbidden":
        raise ValueError("evaluation cannot refit gamma")
    if evaluation.get("singleton_exact_cell_action") != (
        "blocked_no_population_without_cross_layout_pooling"
    ):
        raise ValueError("singleton exact cell action changed")

    decision = _mapping(config, "operator_specificity_decision")
    if decision.get("scope") != "one_final_frozen_heldout_decision":
        raise ValueError("operator-specificity decision must be single and final")
    if decision.get("query_annotation_role") != (
        "evaluation_metrics_only_after_outputs_frozen"
    ):
        raise ValueError("query annotations may only score frozen held-out outputs")
    if decision.get(
        "adaptation_reselection_or_method_change_after_opening_query_annotations"
    ) != "forbidden":
        raise ValueError("held-out annotations cannot cause adaptation or reselection")
    expected_decision_values = {
        "minimum_paired_participant_fraction": 0.80,
        "minimum_improvement_fraction": 0.60,
        "minimum_nonartifact_observation_preservation": 0.90,
        "maximum_reference_free_covariance_distortion": 0.25,
        "minimum_safety_pass_fraction": 0.90,
    }
    for key, expected in expected_decision_values.items():
        if float(decision.get(key, -1.0)) != expected:
            raise ValueError(f"SGEYESUB decision threshold changed: {key}")
    if tuple(decision.get("required_improvement_controls", ())) != (
        "pop_Qy",
        "matching_Qy",
        "wrong_Qy",
        "shuffled_Qy",
    ):
        raise ValueError("SGEYESUB decision controls changed")
    if decision.get("gamma_zero_decision") != (
        "personalization_failed_population_deterministic"
    ):
        raise ValueError("gamma=0 decision must remain automatic")


def run_sgeyesub_protocol_metadata(
    config: Mapping[str, object], *, run_dir: Path
) -> dict[str, object]:
    """Build protocol artifacts from the prior compact metadata audit only."""

    validate_sgeyesub_protocol_config(config)
    metadata = _mapping(config, "metadata")
    structure_audit = Path(str(metadata["structure_audit_result"]))
    layouts, records = load_sgeyesub_structure_audit(structure_audit)
    compatibility = _mapping(config, "compatibility_cell")
    b6 = _mapping(config, "b6_pop_shrink")
    plan = build_sgeyesub_protocol(
        layouts,
        records,
        protocol_id=str(config["protocol_id"]),
        reference_cell_id=str(compatibility["reference_cell_id"]),
        gamma_candidates=tuple(b6["gamma_candidates"]),
    )

    output_root = Path(str(config["output_root"]))
    outputs = write_sgeyesub_protocol_outputs(plan, output_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(outputs["split_manifest"], run_dir / "split_manifest.csv")

    result = plan.summary()
    result.update(
        {
            "structure_audit_result": str(structure_audit),
            "signal_payload_opened": False,
            "outputs": outputs,
            "next_stage": "submit_release_internal_p0_b6_development",
            "scientific_matrix_submitted": False,
        }
    )
    (run_dir / "result_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result
