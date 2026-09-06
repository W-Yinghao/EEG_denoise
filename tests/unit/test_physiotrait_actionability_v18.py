from pathlib import Path
import ast
import json

import numpy as np
import pytest

from eeg_cgdr.experiments.physiotrait_actionability_v18 import (
    _blend,
    _exact_sign_flip,
    block_distances,
    fold_members,
    guard_role,
)


ROOT = Path(__file__).parents[2]


def test_immutable_brainid_gates_remain_failures() -> None:
    old = json.loads(Path("/home/infres/yinwang/denoiseNet_brainid_gate_v17/results/cgdr/brainid_gate_v17/gate01_decision.json").read_text())
    repair = json.loads(Path("/home/infres/yinwang/denoiseNet_brainid_gate_v17r/results/cgdr/brainid_gate_v17r/gate01r_decision.json").read_text())
    assert old["PASS_01"] is False and old["M1_verifier"] == "FAIL"
    assert repair["PASS_01R"] is False and repair["M1R_verifier"] == "FAIL"


def test_day200_guard_is_fail_closed() -> None:
    for role in ("R", "T", "G"):
        guard_role(role)
    for role in ("F", "Day_200", "Day-200"):
        with pytest.raises(PermissionError):
            guard_role(role)


def test_outer_participant_folds_are_disjoint() -> None:
    seen = []
    for fold in range(5):
        training, evaluation = fold_members(fold)
        assert len(training) == 12 and len(evaluation) == 3
        assert set(training).isdisjoint(evaluation)
        seen.extend(evaluation)
    assert sorted(seen) == list(range(1, 16))


def test_trait_blend_and_distance_are_block_equal_weighted() -> None:
    pop = (np.zeros(3), np.zeros(2), np.zeros(4))
    match = (np.ones(3), np.ones(2) * 2, np.ones(4) * 3)
    half = _blend(pop, match, .5)
    assert np.allclose(block_distances(half, pop), [.5, 1., 1.5])


def test_exact_participant_sign_flip_not_trial_inflated() -> None:
    values = np.arange(1, 16, dtype=float)
    assert _exact_sign_flip(values) == pytest.approx(1 / 32768)


def test_no_identity_verifier_or_network_stage() -> None:
    source = (ROOT / "src/eeg_cgdr/experiments/physiotrait_actionability_v18.py").read_text(encoding="utf-8")
    cli = (ROOT / "src/eeg_cgdr/cli/physiotrait_actionability_v18.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any(name.startswith("torch") or "brainid_gate" in name for name in imports)
    assert "train" not in cli.lower()


def test_condition_mapping_is_metadata_frozen_when_available() -> None:
    root = ROOT / "results/cgdr/physiotrait_actionability_v18"
    path = root / "condition_mapping.csv"
    if not path.exists():
        pytest.skip("J0 freeze not run")
    text = path.read_text(encoding="utf-8")
    assert "signal_or_outcome_used" in text and "official task/event metadata" in text


def test_sealed_guard_and_negative_controls_when_available() -> None:
    root = ROOT / "results/cgdr/physiotrait_actionability_v18"
    guard = root / "sealed_guard.json"
    schema = root / "trait_schema.json"
    if not guard.exists():
        pytest.skip("J0 freeze not run")
    value = json.loads(guard.read_text())
    assert value["day200_opened"] is False and value["physiomotion_sealed_opened"] is False
    spec = json.loads(schema.read_text())
    assert set(spec["negative_controls"]) >= {"prestim", "hf_art", "label_shuffle", "time_shuffle", "wrong_condition"}


def test_gate_schema_and_participant_first_when_available() -> None:
    root = ROOT / "results/cgdr/physiotrait_actionability_v18"
    gate = root / "trait_gate_decision.json"
    if not gate.exists():
        pytest.skip("final gate not run")
    value = json.loads(gate.read_text())
    assert set(value) == {"brainid_gate01", "brainid_gate01r", "data_protocol", "trait_headroom", "trait_actionability", "PASS_TRAIT", "failed_criteria", "identity_verifier_used", "day200_opened", "physiomotion_sealed_opened", "denoiser_or_diffusion_trained"}
    assert value["identity_verifier_used"] is False and value["denoiser_or_diffusion_trained"] is False
    effects = root / "trait_headroom_participant_metrics.csv"
    if effects.exists():
        rows = effects.read_text(encoding="utf-8").splitlines()
        primary = [row for row in rows[1:] if ",primary," in row]
        assert len(primary) == 15
