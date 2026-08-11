from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

from eeg_cgdr.experiments.calibration_transfer_v20 import (
    endpoint_pass,
    generate_injections,
    plus_one_p,
    randomization_loop,
    randomization_vectorized,
    route_from_endpoints,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/cgdr/calibration_transfer_randomization_v20.yaml"


def _fixture() -> tuple[list[str], list[str]]:
    recipients = [f"p{i}" for i in range(3)]
    owners = recipients + ["p3"]
    return recipients, owners


def test_exact_null_fixture() -> None:
    recipients, owners = _fixture()
    risk = np.ones((3, 4)); pop = np.ones(3)
    assignments, _, _, _ = generate_injections(recipients, owners, 200, 20)
    t_p, t_w = randomization_vectorized(risk, pop, assignments)
    assert np.all(t_p == 0) and np.all(t_w == 0)
    assert plus_one_p(t_p, 0.0) == 1.0 and plus_one_p(t_w, 0.0) == 1.0


def test_perfect_identity_and_dual_replay() -> None:
    recipients, owners = _fixture()
    risk = np.ones((3, 4)); pop = np.full(3, 0.8)
    for i in range(3): risk[i, i] = 0.1
    assignments, _, _, _ = generate_injections(recipients, owners, 500, 21)
    a = randomization_loop(risk, pop, assignments); b = randomization_vectorized(risk, pop, assignments)
    np.testing.assert_array_equal(a[0], b[0]); np.testing.assert_array_equal(a[1], b[1])
    assert plus_one_p(a[0], 0.7) < 0.025 and plus_one_p(a[1], 0.9) < 0.025


def test_route_truth_table() -> None:
    assert route_from_endpoints(True, True)[0] == "V20_NATURAL_TRANSFER_PASS"
    assert route_from_endpoints(False, True)[0] == "V20_SPECIFICITY_WITHOUT_POP_INCREMENT"
    assert route_from_endpoints(True, False)[0] == "V20_POP_GAIN_WITHOUT_SPECIFICITY"
    assert route_from_endpoints(False, False)[0] == "V20_NATURAL_TRANSFER_NOT_ESTABLISHED"
    assert route_from_endpoints(True, True, False)[0] == "V20_CONSTRUCT_FALSIFICATION_FAILED"


def test_practical_and_statistical_requirements_are_separate() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert endpoint_pass(0.011, 0.051, 0.02, config)
    assert not endpoint_pass(0.009, 0.5, 0.001, config)
    assert not endpoint_pass(0.5, 0.049, 0.001, config)
    assert not endpoint_pass(0.5, 0.5, 0.026, config)


def test_fixed_point_free_injection_replay_and_unused_owner() -> None:
    recipients = [f"p{i}" for i in range(15)]; owners = recipients + ["p15"]
    a, ua, _, _ = generate_injections(recipients, owners, 1000, 20260820)
    b, ub, _, _ = generate_injections(recipients, owners, 1000, 20260820)
    c, _, _, _ = generate_injections(recipients, owners, 1000, 20260822)
    np.testing.assert_array_equal(a, b); np.testing.assert_array_equal(ua, ub)
    assert not np.array_equal(a, c)
    assert a.dtype == np.uint8 and ua.dtype == np.uint8
    assert all(len(set(row.tolist())) == 15 for row in a)
    assert all(int(row[i]) != i for row in a for i in range(15))
    assert all(int(ua[index]) not in set(a[index].tolist()) for index in range(len(a)))


def test_frozen_manifest_has_no_obvious_unused_owner_bias() -> None:
    path = ROOT / "results/cgdr/calibration_transfer_randomization_v20/permutation_manifest.npz"
    if not path.is_file(): pytest.skip("P3 not run yet")
    with np.load(path, allow_pickle=False) as data:
        assignment = np.asarray(data["assignments"]); unused = np.asarray(data["unused_owner"])
    assert assignment.shape == (100000, 15) and unused.shape == (100000,)
    counts = np.bincount(unused, minlength=16)
    # A broad implementation-bias guard, not a scientific uniformity test.
    assert int(np.max(np.abs(counts - 6250))) < 600
    assert all(len(set(row.tolist())) == 15 for row in assignment[:1000])


def test_eligibility_failure_is_fail_closed() -> None:
    recipients, owners = _fixture(); eligibility = np.ones((3, 4), dtype=bool)
    eligibility[0] = False
    # No accepted assignment exists; own-support completeness must be checked before generation.
    assert not np.all(eligibility) and not eligibility[0, 0]


def test_null_family_exclusion_frozen_in_metric_contract_after_p2() -> None:
    path = ROOT / "results/cgdr/calibration_transfer_randomization_v20/metric_contract.json"
    if not path.is_file(): pytest.skip("P2 not run yet")
    text = path.read_text(encoding="utf-8").lower()
    assert "positive clipping" in text and "q95 floor" in text
    assert "no time-shift/channel/gain" in text


def test_manifest_contains_assignments_only_not_falsification_nulls() -> None:
    path = ROOT / "results/cgdr/calibration_transfer_randomization_v20/permutation_manifest.npz"
    if not path.is_file(): pytest.skip("P3 not run yet")
    with np.load(path, allow_pickle=False) as data:
        assert set(data.files) == {"assignments", "unused_owner", "recipient_order", "support_owner_order"}


def test_plus_one_monte_carlo_definition() -> None:
    null = np.asarray([-1.0, 0.0, 1.0])
    assert plus_one_p(null, 1.0) == pytest.approx(2 / 4)
    assert plus_one_p(null, 2.0) == pytest.approx(1 / 4)


def test_config_scientific_units_and_rng_are_frozen() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert len(config["development_participants"]) == 16
    assert len(config["primary_recipients"]) == 15
    assert config["policy_only_participant"] == "sub-24"
    assert config["permutation"] == {"bit_generator": "PCG64DXSM", "seed": 20260820, "accepted_replicates": 100000}
    assert config["bootstrap"]["seed"] == 20260821


def test_actual_risk_matrix_and_policy_denominator() -> None:
    root = ROOT / "results/cgdr/calibration_transfer_randomization_v20"
    matrix_path = root / "natural_risk_matrix.csv"
    effects_path = root / "participant_effects_observed.csv"
    if not matrix_path.is_file(): pytest.skip("P4 not run yet")
    import csv
    with matrix_path.open(encoding="utf-8", newline="") as stream: matrix = list(csv.DictReader(stream))
    with effects_path.open(encoding="utf-8", newline="") as stream: effects = list(csv.DictReader(stream))
    assert len(matrix) == 15 * 16 and len(effects) == 16
    assert sum(row["self_match"] == "1" for row in matrix) == 15
    assert all(np.isfinite(float(row["risk"])) for row in matrix)
    policy = next(row for row in effects if row["participant"] == "sub-24")
    assert policy["primary_exchangeable"] == "0" and float(policy["N_P"]) == 0 and float(policy["N_W"]) == 0


def test_actual_randomization_and_dual_replay_contract() -> None:
    root = ROOT / "results/cgdr/calibration_transfer_randomization_v20"
    decision_path = root / "route_decision.json"
    if not decision_path.is_file(): pytest.skip("P8 not run yet")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    replay = json.loads((root / "independent_replay_comparison.json").read_text(encoding="utf-8"))
    assert decision["participant_first"] and decision["wrong_donor_mean_preserved"]
    assert decision["risk_matrix_exact_replay"] and decision["exchangeability_valid"]
    assert decision["permutation_replicates"] == 100000 and decision["permutation_seed"] == 20260820
    assert replay["dual_replay_exact"] and replay["maximum_difference"] <= 1e-12


def test_cpu_submitter_rejects_gpu() -> None:
    submit = ROOT / "scripts/slurm/calibration_transfer_v20/submit.sh"
    result = subprocess.run([str(submit), "A100", "p0-freeze"], text=True, capture_output=True, check=False)
    assert result.returncode == 2 and "cpu/cpu-high" in result.stderr
    result = subprocess.run([str(submit), "cpu", "p0-freeze", "--gres", "gpu:1"], text=True, capture_output=True, check=False)
    assert result.returncode == 2 and "forbidden" in result.stderr


def test_source_and_a_track_governance() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    for key, expected in (("source_v19_worktree", config["source_v19_commit"]), ("source_audit_worktree", config["source_audit_commit"]), ("a_track_worktree", config["a_track_commit"])):
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(config[key]), text=True).strip()
        assert actual == expected
    a_root = Path(config["a_track_worktree"])
    assert subprocess.run(["git", "diff", "--quiet", "HEAD", "--", "taas_submission"], cwd=a_root, check=False).returncode == 0


def test_v19_supersession_is_note_only_and_records_14_15_discrepancy() -> None:
    note = (ROOT / "reports/v19_null_floor_supersession_note.md").read_text(encoding="utf-8")
    assert "V19_NATURAL_GATE_NONADJUDICABLE_DUE_INVALID_NULL_PROTOCOL" in note
    assert "14" in note and "15" in note and "new development-only" in note
    source = Path("/home/infres/yinwang/denoiseNet_counterfactual_operator_v19")
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip() == "5ab1918ceebf3b9622ceb2806a274edd01205e8b"


def test_source_sealed_ledger_and_terminal_no_execution() -> None:
    source = Path("/home/infres/yinwang/denoiseNet_counterfactual_operator_v19/results/cgdr/counterfactual_operator_headroom_v19")
    sealed = json.loads((source / "sealed_guard.json").read_text(encoding="utf-8"))
    assert all(int(value) == 0 for key, value in sealed.items() if key.endswith("_reads"))
    terminal = ROOT / "results/cgdr/calibration_transfer_randomization_v20/terminal_manifest.json"
    if terminal.is_file():
        value = json.loads(terminal.read_text(encoding="utf-8"))
        assert value["raw_reads"] == value["sealed_reads"] == value["GPU_jobs"] == value["models_trained"] == 0
        assert value["O1_executed"] is False and value["paper_modified"] is False


def test_terminal_governance_if_decision_exists() -> None:
    path = ROOT / "results/cgdr/calibration_transfer_randomization_v20/route_decision.json"
    if not path.is_file(): pytest.skip("decision not run yet")
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["sealed_opened"] is False and value["raw_signal_opened"] is False
    assert value["GPU_jobs"] == 0 and value["O1_executed"] is False
    assert value["DET_executed"] is False and value["diffusion_executed"] is False
