from pathlib import Path

from eeg_cgdr.experiments.project_evidence_freeze import (
    CENTRAL_SENTENCE,
    _experiments,
    aliases,
    claim_whitelist,
    metric_schema,
    split_registry,
)


def test_frozen_state_vocabulary_and_claim_sentence() -> None:
    assert "did not provide reproducible incremental" in CENTRAL_SENTENCE
    assert "largely associated with averaging" in CENTRAL_SENTENCE
    unsupported = {r["claim"] for r in claim_whitelist() if r["status"] == "unsupported"}
    assert "diffusion-specific subject utility" in unsupported
    assert "independent confirmation" in unsupported


def test_canonical_replacements_and_scientific_units() -> None:
    experiments = _experiments()
    by_id = {e.experiment_id: e for e in experiments}
    assert by_id["sge_v8"].superseded_by == "sge_v8_1"
    assert by_id["sge_v9"].superseded_by == "sge_v9r"
    assert by_id["bci2b_v11"].superseded_by == "bci2b_v11_1"
    assert by_id["raw_support"].superseded_by == "raw_support_closure"
    assert all(e.scientific_unit for e in experiments)


def test_sealed_guards_and_shu_scope() -> None:
    rows = {r["dataset"]: r for r in split_registry()}
    assert "sealed unopened" in rows["MobileBCI"]["outcomes_opened"]
    assert "sealed 10 unopened" in rows["PhysioMotion"]["outcomes_opened"]
    assert "Day4/5 unopened" in rows["SHU MultiSession MI"]["outcomes_opened"]


def test_aliases_and_metric_direction_contract() -> None:
    known = {r["alias"] for r in aliases()}
    assert {"DET1", "DET2", "DET8", "DIFF-K1", "DIFF-K8", "policy", "evaluable"} <= known
    schema = metric_schema()
    assert schema["K_contract"]["NFE_DDIM25_K8"] == 200
    assert "DIFF-K8 vs DET1 as unique diffusion evidence" in schema["forbidden"]
    assert all(v["direction"] in {"higher_is_better", "lower_is_better"} for v in schema["metrics"].values())


def test_a_track_not_registered_as_mutable_experiment() -> None:
    assert all(not e.report.startswith("taas_submission/") for e in _experiments())
