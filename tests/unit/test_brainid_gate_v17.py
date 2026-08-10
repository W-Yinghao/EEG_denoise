from pathlib import Path
import csv
import json

import numpy as np
import pytest
import torch

from eeg_cgdr.experiments.brainid_gate_v17 import (
    BrainprintVerifierA,
    PCARestorer,
    fold_members,
    guard_physio,
    guard_role,
)


def test_day200_guard_is_fail_closed() -> None:
    for role in ("R", "T", "G"):
        guard_role(role)
    with pytest.raises(PermissionError):
        guard_role("F")
    with pytest.raises(PermissionError):
        guard_role("Day_200")


def test_outer_participant_folds_are_disjoint() -> None:
    seen = []
    for fold in range(5):
        training, evaluation = fold_members(fold)
        assert len(training) == 12 and len(evaluation) == 3
        assert set(training).isdisjoint(evaluation)
        assert all(donor != recipient for recipient in evaluation for donor in evaluation if donor != recipient)
        seen.extend(evaluation)
    assert sorted(seen) == list(range(1, 16))


def test_physiomotion_sealed_guard_is_fail_closed(tmp_path: Path) -> None:
    split = tmp_path / "split.csv"
    split.write_text("participant,role\n1,development\n2,sealed\n", encoding="utf-8")
    config = {"physiomotion_split": str(split)}
    guard_physio(config, 1)
    with pytest.raises(PermissionError):
        guard_physio(config, 2)
    with pytest.raises(PermissionError):
        guard_physio(config, 3)


def test_verifier_a_parameter_budget_and_unit_norm() -> None:
    model = BrainprintVerifierA()
    assert sum(v.numel() for v in model.parameters()) < 1_000_000
    model.eval()
    with torch.no_grad():
        output = model(torch.randn(3, 57, 200))
    assert output.shape == (3, 64)
    assert torch.allclose(output.norm(dim=1), torch.ones(3), atol=1e-5)


def test_pca_restorer_preserves_observed_region_exactly() -> None:
    rng = np.random.default_rng(2)
    shape = (3, 8)
    model = PCARestorer(np.zeros(np.prod(shape), np.float32), rng.normal(size=(4, np.prod(shape))).astype(np.float32), 0.1)
    clean = rng.normal(size=shape).astype(np.float32)
    mask = np.zeros(shape, bool); mask[1, 2:5] = True
    observed = clean.copy(); observed[mask] = 0
    restored = model.restore(observed, mask)
    assert np.array_equal(restored[~mask], observed[~mask])


def test_verifier_b_is_physically_separate_and_evaluator_only() -> None:
    source = Path(__file__).parents[2] / "src/eeg_cgdr/experiments/brainid_gate_v17.py"
    text = source.read_text(encoding="utf-8")
    selection = text[text.index("def select_m0_fold"):text.index("def evaluate_m0_fold")]
    training = text[text.index("def train_verifier_a"):text.index("def fit_verifier_b")]
    assert "load_verifier_b" not in selection
    assert "brainid_verifier_b" not in training


def test_no_denoiser_or_diffusion_stage_exposed() -> None:
    cli = (Path(__file__).parents[2] / "src/eeg_cgdr/cli/brainid_gate_v17.py").read_text(encoding="utf-8")
    assert '"train-diffusion"' not in cli
    assert '"denoise"' not in cli


def test_m0_selection_persists_fold_local_outer_only_summary() -> None:
    source = Path(__file__).parents[2] / "src/eeg_cgdr/experiments/brainid_gate_v17.py"
    text = source.read_text(encoding="utf-8")
    selection = text[text.index("def select_m0_fold"):text.index("def evaluate_m0_fold")]
    assert '_json(out/"selection.json",result)' in selection
    assert '"heldout_outcomes_used":False' in selection


def test_frozen_session_roles_are_disjoint_and_future_is_forbidden() -> None:
    root = Path(__file__).parents[2] / "results/cgdr/brainid_gate_v17/frozen"
    with (root / "session_role_manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 15 * 4
    for participant in range(1, 16):
        local = [row for row in rows if int(row["participant"]) == participant]
        assert {row["role"] for row in local} == {"R", "T", "G", "F"}
        assert len({row["acquisition"] for row in local}) == 4
        assert next(row for row in local if row["role"] == "F")["loader_access"] == "forbidden"


def test_m1_participant_first_summary_recomputes_exactly() -> None:
    root = Path(__file__).parents[2] / "results/cgdr/brainid_gate_v17"
    decision = json.loads((root / "m1_decision.json").read_text(encoding="utf-8"))
    for name in ("A", "B"):
        with (root / f"m1_verifier_{name.lower()}_subject_metrics.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 15
        assert len({int(row["participant"]) for row in rows}) == 15
        auroc = np.mean([float(row["auroc"]) for row in rows])
        positive = sum(float(row["identity_margin"]) > 0 for row in rows)
        assert auroc == pytest.approx(decision["verifiers"][name]["auroc"], abs=1e-12)
        assert positive == decision["verifiers"][name]["positive"]
