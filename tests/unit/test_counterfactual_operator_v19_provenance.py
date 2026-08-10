from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

from eeg_cgdr.experiments.counterfactual_operator_v19_provenance import (
    exact_signflip, reference_joint_maxstat,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/cgdr/counterfactual_operator_v19_null_floor_audit.yaml"


def test_exact_null_fixture() -> None:
    raw = np.ones(16); match = np.ones(16)
    effect = raw - match
    np.testing.assert_array_equal(effect, np.zeros(16))
    assert float(np.mean(effect)) == 0.0


def test_sign_fixture_direction_and_exact_p() -> None:
    values = np.asarray([2.0, 1.0, -0.25, 0.5])
    assert float(np.mean(values)) > 0
    assert exact_signflip(values) == pytest.approx(0.125)
    assert exact_signflip(np.ones(16)) == pytest.approx(1 / 65536)
    assert exact_signflip(np.ones(16), two_sided=True) == pytest.approx(2 / 65536)


def test_maxstat_fixture_axis_and_linear_q95() -> None:
    # replicate x participant x endpoint: participant mean precedes endpoint max.
    fixture = np.asarray([[[1, 0], [3, 2]], [[0, 4], [2, 2]], [[2, 1], [2, 3]]], dtype=float)
    maxstat, q95 = reference_joint_maxstat(fixture)
    np.testing.assert_allclose(maxstat, [2.0, 3.0, 2.0])
    assert q95 == pytest.approx(np.quantile([2.0, 3.0, 2.0], .95, method="linear"))


def test_base_and_job_sets_are_frozen() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["source_v19_commit"] == "5ab1918ceebf3b9622ceb2806a274edd01205e8b"
    assert set(config["accepted_jobs"]) == {934192, 934209, 934229, 934230, 934231, 934248, 934265, 934266, 934267}
    assert set(config["excluded_jobs"]) == {934175, 934225, 934226, 934227, 934264}
    assert 934174 not in config["accepted_jobs"] and 934191 not in config["accepted_jobs"]


def test_original_decision_and_sealed_ledger_immutable() -> None:
    source = Path("/home/infres/yinwang/denoiseNet_counterfactual_operator_v19/results/cgdr/counterfactual_operator_headroom_v19")
    decision = json.loads((source / "route_decision.json").read_text(encoding="utf-8"))
    sealed = json.loads((source / "sealed_guard.json").read_text(encoding="utf-8"))
    assert decision["route"] == "SUPPORT_TO_QUERY_OPERATOR_TRANSFER_NO_GO"
    assert decision["O1"] == "NOT_RUN"
    assert all(int(sealed[key]) == 0 for key in ("mobile_sealed_reads", "physiomotion_sealed_reads", "shu_day4_day5_reads", "physiotrait_day200_reads"))


def test_prereg_has_no_permutation_or_maxstat_schedule() -> None:
    source = Path("/home/infres/yinwang/denoiseNet_counterfactual_operator_v19/results/cgdr/counterfactual_operator_headroom_v19/v19_preregistration.yaml")
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    assert "fold-local q95" in value["outer_null_floor"]
    text = source.read_text(encoding="utf-8").lower()
    assert "permutation" not in text and "max-stat" not in text and "interpolation" not in text


def test_cpu_submitter_rejects_gpu_profile() -> None:
    submitter = ROOT / "scripts/slurm/counterfactual_operator_v19_provenance/submit.sh"
    result = subprocess.run([str(submitter), "A100", "p0-freeze"], text=True, capture_output=True, check=False)
    assert result.returncode == 2
    assert "cpu/cpu-high" in result.stderr


def test_a_track_is_immutable() -> None:
    a_root = Path("/home/infres/yinwang/denoiseNet_taas_subject_diffusion")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=a_root, text=True).strip()
    assert head == "0c4f2301c1f873120fe54537cde3c76fff7ea3a2"
    assert subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "taas_submission"], cwd=a_root, check=False).returncode == 0
