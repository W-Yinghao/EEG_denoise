from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import yaml

from eeg_scad.cli import run_v31
from eeg_scad.evaluation.claim_package_v31 import (
    claim_rows,
    reviewer_rows,
    scope_comparison,
    write_scope_and_consultation_reports,
)
from eeg_scad.evaluation.support_duration_v31 import (
    attach_exact_support,
    duration_contract,
    exact_support_starts,
    standardized_prefix_support,
    validate_exact_manifest,
)


ROOT = Path(__file__).resolve().parents[2]


def test_base_sha_is_v30_terminal():
    assert run_v31.BASE == "220dcbaaabdef0cb8d1ac91b87b0d1cc8b7109cf"


def test_ledger_v23_or_terminal_v24():
    text = (ROOT / "docs/TAAS_SUBJECT_AWARE_DIFFUSION_PROJECT_LEDGER.md").read_text()
    assert ("**版本：** v2.3" in text or "**版本：** v2.4" in text) and "V31" in text


def test_protocol_is_development_only_without_training():
    cfg = yaml.safe_load((ROOT / "configs/claim_narrowing_v31.yaml").read_text())
    assert cfg["development_only"] is True and cfg["new_model_training"] is False and cfg["K"] == 1


def test_sealed_registry_exact():
    cfg = yaml.safe_load((ROOT / "configs/claim_narrowing_v31.yaml").read_text())
    assert cfg["sealed_participants"] == ["sub-01", "sub-04", "sub-08", "sub-10", "sub-13", "sub-16", "sub-20", "sub-22"]


def test_exact_nonoverlap_prefix_starts():
    assert exact_support_starts(0) == []
    assert exact_support_starts(5) == [0, 200]
    assert exact_support_starts(10) == [0, 200, 400, 600, 800]
    assert len(exact_support_starts(30)) == 15
    assert len(exact_support_starts(120)) == 60


def test_duration_contract_distinguishes_acquisition_and_effective_time():
    contract = duration_contract(5)
    assert contract["acquisition_span_seconds"] == 5
    assert contract["effective_seconds"] == 4
    assert contract["effective_samples"] == 400
    assert contract["window_count"] == 2


def test_prefix_normalization_does_not_read_future_support():
    rng = np.random.default_rng(31)
    eeg = rng.normal(size=(46, 12000))
    eog = rng.normal(size=(4, 12000))
    eeg_scale = np.linspace(1, 2, 46)
    first = standardized_prefix_support(eeg, eog, eeg_scale, 5)
    changed = eog.copy()
    changed[:, 500:] = 1e9
    second = standardized_prefix_support(eeg, changed, eeg_scale, 5)
    for left, right in zip(first[:4], second[:4]):
        assert np.array_equal(left, right)


def test_prefix_normalization_uses_fifth_second_but_model_windows_do_not():
    rng = np.random.default_rng(32)
    eeg = rng.normal(size=(46, 12000))
    eog = rng.normal(size=(4, 12000))
    eeg_scale = np.ones(46)
    first = standardized_prefix_support(eeg, eog, eeg_scale, 5)
    changed = eog.copy()
    changed[:, 400:500] += 20
    second = standardized_prefix_support(eeg, changed, eeg_scale, 5)
    assert not np.array_equal(first[1], second[1])
    assert first[4] == second[4] == [0, 200]


def test_no_overlap_or_repeated_samples():
    for duration in (5, 10, 30, 120):
        starts = exact_support_starts(duration)
        occupied = [sample for start in starts for sample in range(start, start + 200)]
        assert len(occupied) == len(set(occupied))
        assert max(occupied) < duration * 100


def test_manifest_validator_checks_exact_contract():
    rows = []
    for duration in (5, 10, 30, 120):
        contract = duration_contract(duration)
        rows.append({**contract, "starts": ";".join(map(str, contract["starts"])), "normalization_prefix_samples": duration * 100})
    result = validate_exact_manifest(rows)
    assert result["no_overlap"] and result["prefix_only_normalization"]


def test_manifest_validator_rejects_overlapping_v30_style():
    contract = duration_contract(5)
    row = {**contract, "starts": "0;150", "normalization_prefix_samples": 500}
    try:
        validate_exact_manifest([row])
    except RuntimeError:
        pass
    else:
        raise AssertionError("overlapping contract was accepted")


def test_zero_support_is_exact_population_bypass():
    batch = {"y": np.zeros((2, 46, 200)), "meta": [{}, {}]}
    result = attach_exact_support(batch, {}, [], 0)
    assert result["population_bypass"] is True
    assert "support_eeg" not in result and result["y"] is batch["y"]


def test_same_queries_checkpoints_noise_registered():
    cfg = yaml.safe_load((ROOT / "configs/claim_narrowing_v31.yaml").read_text())
    assert cfg["same_common_panel"] and cfg["same_checkpoint"] and cfg["same_query"] and cfg["same_diffusion_noise"]


def test_duration_inference_uses_frozen_v30_output_path():
    source = inspect.getsource(run_v31.duration_inference)
    assert "v30._encode" in source and "_support_outputs" in source
    assert "optimizer" not in source and "backward" not in source


def test_no_training_stage_in_cli():
    assert not any("train" in stage for stage in run_v31.STAGES)


def test_freeze_precedes_natural_evaluator():
    stages = list(run_v31.STAGES)
    assert stages.index("r3-freeze") < stages.index("r4-duration-evaluator")


def test_claim_matrix_has_all_required_claims_and_fields():
    rows = claim_rows()
    assert len(rows) >= 14
    required = {"claim_text", "evidence_source", "supporting_result", "contradicting_result", "scientific_status", "allowed_wording", "forbidden_wording", "manuscript_location", "reviewer_relevance"}
    assert all(required <= set(row) and all(str(row[key]).strip() for key in required) for row in rows)
    assert {row["scientific_status"] for row in rows} <= {"supported", "partially_supported", "mixed", "unsupported", "unavailable"}


def test_forbidden_claims_are_explicit():
    text = " ".join(row["forbidden_wording"] for row in claim_rows()).lower()
    for concept in ("specificity", "preservation", "safe deployment", "cross-session", "cross-montage"):
        assert concept in text


def test_scope_A_and_B_are_both_prepared_without_auto_selection():
    scopes = scope_comparison()
    assert scopes["scope_A"]["id"] == "A_audit_centric"
    assert scopes["scope_B"]["id"] == "B_method_centric"
    assert scopes["recommended_for_AE_consultation"] == "Scope A"
    assert scopes["automatic_selection"] is False


def test_scope_B_declares_higher_acceptance_risk():
    scopes = scope_comparison()
    assert scopes["scope_B"]["acceptance_risk"] == "higher than Scope A"


def test_reviewer_map_has_allowed_statuses_and_required_topics():
    rows = reviewer_rows()
    allowed = {"resolved", "partially_resolved", "unresolved_but_claim_withdrawn", "requires_AE_guidance"}
    assert all(row["status"] in allowed for row in rows)
    text = " ".join(row["comment"] for row in rows).lower()
    for topic in ("strong", "subject-agnostic", "raw", "statistics", "support amount", "latency", "privacy", "transductive", "montage", "physiology"):
        assert topic in text


def test_document_writer_prepares_but_does_not_send_email(tmp_path: Path):
    reports = tmp_path / "reports"
    results = tmp_path / "results"
    write_scope_and_consultation_reports(reports, results)
    email = (reports / "v31_AE_consultation_email.md").read_text()
    assert "NOT SENT" in email
    assert (reports / "v31_scope_A_audit_centric.md").is_file()
    assert (reports / "v31_scope_B_method_centric.md").is_file()


def test_decision_contract_is_review_and_consultation():
    if (ROOT / "results/claim_narrowing_v31/decision.json").is_file():
        decision = json.loads((ROOT / "results/claim_narrowing_v31/decision.json").read_text())
        assert decision["next_action"] == "USER_REVIEW_AND_AE_CONSULTATION"
        assert decision["AE_email_sent"] is False and decision["manuscript_modified"] is False


def test_query_auxiliary_and_sealed_reads_registered_zero():
    source = (ROOT / "src/eeg_scad/cli/run_v31.py").read_text()
    for field in ("query_EOG_inference_reads", "query_operator_inference_reads", "event_inference_reads", "sealed_reads"):
        assert f'"{field}": 0' in source


def test_v30_duration_rows_are_preserved_as_historical_invalid():
    source = inspect.getsource(run_v31.duration_aggregate)
    assert "historical_invalid_duration_contract" in source
    assert "results/frozen_candidate_v30/support_duration_effects.csv" in source


def test_diagnostic_aggregate_has_explicit_support_encoding_label():
    rows = [{"duration_seconds": "5", "context_stability_to_120": "1.5"}]
    result = run_v31._mean_rows(rows, "context_stability_to_120", "diagnostic")
    assert result[0]["method"] == "SUPPORT_ENCODING" and result[0]["mean"] == 1.5


def test_manuscript_is_not_an_edit_target():
    source = (ROOT / "src/eeg_scad/cli/run_v31.py").read_text()
    assert "manuscript_modified\": False" in source
    assert "taas_submission" in source  # governance diff check only


def test_clean_archive_import_surface():
    assert callable(run_v31.main) and callable(run_v31.duration_aggregate) and callable(run_v31.documents)
