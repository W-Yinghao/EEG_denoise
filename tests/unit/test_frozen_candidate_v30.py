from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import torch
import yaml

from eeg_scad.cli import run_v30
from eeg_scad.evaluation.all_donor_specificity import donor_summary, group_summary
from eeg_scad.evaluation.common_panel_v30 import (
    attach_support,
    content_digest,
    select_balanced_indices,
    support_bank_index,
    support_starts,
)
from eeg_scad.evaluation.linkage_diagnostic import linkage, projector_features
from eeg_scad.evaluation.support_duration import aggregate_duration, validate_duration_contract


ROOT = Path(__file__).resolve().parents[2]
OWNERS = [f"sub-{value:02d}" for value in range(1, 16)]


def test_base_sha_is_v29_terminal():
    assert run_v30.BASE == "9ca9c79b6f1549e89428e28c62ebbea6d3c0bb37"


def test_ledger_v21_or_terminal_v22():
    text = (ROOT / "docs/TAAS_SUBJECT_AWARE_DIFFUSION_PROJECT_LEDGER.md").read_text()
    assert ("**版本：** v2.1" in text or "**版本：** v2.2" in text) and "V30" in text


def test_protocol_is_development_only_and_k1():
    cfg = yaml.safe_load((ROOT / "configs/frozen_candidate_v30.yaml").read_text())
    assert cfg["development_only"] is True and cfg["K"] == 1


def test_sealed_registry_exact():
    cfg = yaml.safe_load((ROOT / "configs/frozen_candidate_v30.yaml").read_text())
    assert cfg["sealed_participants"] == ["sub-01", "sub-04", "sub-08", "sub-10", "sub-13", "sub-16", "sub-20", "sub-22"]


def test_support_bank_includes_registered_auxiliary_owner():
    owners = run_v30._support_owners()
    assert len(owners) == 16 and owners[-1] == "sub-24"


def test_support_starts_zero_is_exact_pop():
    assert support_starts(0) == []


def test_support_starts_are_unique_and_bounded():
    for seconds in (5, 10, 30, 120):
        starts = support_starts(seconds)
        assert len(starts) == len(set(starts))
        assert all(0 <= start <= seconds * 100 - 200 for start in starts)


def test_duration_contract_registered():
    contract = validate_duration_contract((0, 5, 10, 30, 120))
    assert contract["0"] == [] and len(contract["120"]) == 16


def test_panel_selection_balances_cells_without_outcomes():
    rows = [
        {"participant": participant, "session": "s", "task": "t", "outcome": str(index)}
        for participant in ("a", "b") for index in range(6)
    ]
    chosen = select_balanced_indices(rows, 4)
    assert chosen == [0, 1, 2, 3, 6, 7, 8, 9]


def test_content_digest_is_deterministic_and_content_sensitive():
    array = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert content_digest((array,)) == content_digest((array.copy(),))
    assert content_digest((array,)) != content_digest((array + 1,))


def test_support_bank_index_is_bijective():
    indices = {support_bank_index(owner, session, task, OWNERS) for owner in OWNERS for session in ("ses-02", "ses-03", "ses-04") for task in ("ERP", "SSVEP")}
    assert indices == set(range(90))


def test_attach_support_uses_requested_donor():
    owners = ["a", "b"]
    bank = {"eeg_5": np.arange(12).reshape(12, 1, 1, 1), "eog_5": np.arange(12).reshape(12, 1, 1, 1) + 100}
    batch = {"meta": [{"participant": "a", "wrong_owner": "b", "session": "ses-02", "task": "ERP"}], "y": np.zeros((1, 1, 1))}
    result = attach_support(batch, bank, owners, 5, donor="b")
    assert result["support_eeg"].item() == 6 and result["support_eog"].item() == 106


def _donor_rows():
    rows = []
    for recipient_index, recipient in enumerate(OWNERS):
        for donor_index, donor in enumerate(OWNERS):
            rows.append({"method": "M", "recipient": recipient, "donor": donor, "risk": float(abs(donor_index - recipient_index))})
    return rows


def test_all_fourteen_wrong_donors_required():
    summary = donor_summary(_donor_rows())
    assert len(summary) == 15 and all(row["correct_rank"] == 1 for row in summary)


def test_all_donor_group_is_participant_first():
    grouped = group_summary(donor_summary(_donor_rows()))[0]
    assert grouped["participants"] == 15 and grouped["correct_top1"] == 15


def test_linkage_identical_population_token_is_chance():
    features = {owner: (np.zeros(8), np.zeros(8)) for owner in OWNERS}
    result = linkage(features)[0]
    assert np.isclose(result["top1_accuracy"], 1 / 15)
    assert np.isclose(result["top3_accuracy"], 3 / 15)


def test_linkage_perfect_features_are_identified():
    features = {owner: (np.eye(15)[index], np.eye(15)[index]) for index, owner in enumerate(OWNERS)}
    result = linkage(features)[0]
    assert result["top1_accuracy"] == result["top3_accuracy"] == 1.0


def test_projector_feature_rotation_invariant():
    rng = np.random.default_rng(30); basis = rng.normal(size=(46, 8)); rotation = np.linalg.qr(rng.normal(size=(8, 8)))[0]
    projected = projector_features({"p": (basis, basis @ rotation)})["p"]
    assert np.allclose(projected[0], projected[1], atol=1e-10)


def test_duration_aggregate_keeps_panels_separate():
    rows = [
        {"panel": "paired", "method": "M", "duration_seconds": 5, "risk": value, "artifact_remaining": "", "retention": "", "context_stability": "", "projector_stability": "", "encoding_ms": 1}
        for value in (1, 3)
    ]
    result = aggregate_duration(rows)
    assert next(row["mean"] for row in result if row["metric"] == "risk") == 2


def test_energy_labels_match_protocol():
    assert [run_v30._lambda_label(value) for value in (.5, 2.0, 8.0)] == ["05", "2", "8"]


def test_candidate_ids_match_all_donor_ids():
    source = inspect.getsource(run_v30._donor_outputs)
    for method in ("V25_SET_CALIB_DET_MATCH", "V26_CALIB_SDEDIT_MATCH", "V29_PA_SC_DET_MATCH", "V29_PA_SC_CDM_MATCH"):
        assert method in source


def test_latency_binding_is_outside_timed_loop():
    source = inspect.getsource(run_v30.steps_latency)
    assert "predict = _benchmark_callable" in source and "_step_predictions(current" not in source


def test_no_training_stage_in_v30_cli():
    assert not any("train" in stage for stage in run_v30.STAGES)


def test_natural_freeze_precedes_evaluator():
    stages = list(run_v30.STAGES)
    assert stages.index("r10-freeze") < stages.index("r11-natural")


def test_query_auxiliary_zero_is_registered():
    source = (ROOT / "src/eeg_scad/cli/run_v30.py").read_text()
    assert '"query_EOG_inference_reads":0' in source and '"sealed_reads":0' in source


def test_v25_to_v29_model_sources_are_not_v30_edit_targets():
    plan = (ROOT / "reports/v30_project_plan.md").read_text()
    assert "No model training" in plan and "V25–V29" in plan


def test_common_panel_uses_fixed_rng_and_identity_rows():
    cfg = yaml.safe_load((ROOT / "configs/frozen_candidate_v30.yaml").read_text())
    assert cfg["panel_seed"] == 20260930 and cfg["same_noise"] is True


def test_same_noise_is_written_in_all_donor_rows():
    source = inspect.getsource(run_v30.all_donor)
    assert '"same_noise": 1' in source


def test_clean_archive_import_surface():
    assert callable(run_v30.main) and callable(run_v30.aggregate)
