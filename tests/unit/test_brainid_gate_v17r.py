from pathlib import Path
import csv
import json

import numpy as np
import pytest
import torch

from eeg_cgdr.experiments.brainid_gate_v17r import (
    BrainprintVerifierAR,
    _counterfactual,
    fold_members,
    guard_role,
)


ROOT=Path(__file__).parents[2]


def test_original_gate01_is_immutable_failure() -> None:
    decision=json.loads(Path("/home/infres/yinwang/denoiseNet_brainid_gate_v17/results/cgdr/brainid_gate_v17/gate01_decision.json").read_text(encoding="utf-8"))
    assert decision["M1_verifier"]=="FAIL"
    assert decision["M0_actionability"]=="INSUFFICIENT"
    assert decision["PASS_01"] is False


def test_day200_guard_fail_closed() -> None:
    for role in ("R","T","G"): guard_role(role)
    for forbidden in ("F","Day_200"):
        with pytest.raises(PermissionError): guard_role(forbidden)


def test_outer_folds_are_participant_disjoint() -> None:
    seen=[]
    for fold in range(5):
        training,evaluation=fold_members(fold);assert len(training)==12 and len(evaluation)==3;assert set(training).isdisjoint(evaluation);seen.extend(evaluation)
    assert sorted(seen)==list(range(1,16))


def test_verifier_a_r_parameter_budget_and_embedding() -> None:
    model=BrainprintVerifierAR();assert sum(v.numel() for v in model.parameters())<1_000_000;model.eval()
    with torch.no_grad(): embedding=model.encode(torch.randn(3,57,200))
    assert embedding.shape==(3,64);assert torch.allclose(embedding.norm(dim=1),torch.ones(3),atol=1e-5)


def test_counterfactual_replay_is_exact() -> None:
    x=np.random.default_rng(2).normal(size=(4,57,200)).astype(np.float32);a=_counterfactual(x,"rereference",71);b=_counterfactual(x,"rereference",71);assert np.array_equal(a,b)


def test_verifier_b_r_is_evaluator_only() -> None:
    source=(ROOT/"src/eeg_cgdr/experiments/brainid_gate_v17r.py").read_text(encoding="utf-8")
    training=source[source.index("def train_verifier_a_r"):source.index("def _load_a_r")]
    assert "VerifierBR" not in training
    assert "fit_verifier_b_r" not in training


def test_no_denoiser_or_diffusion_training_stage() -> None:
    cli=(ROOT/"src/eeg_cgdr/cli/brainid_gate_v17r.py").read_text(encoding="utf-8")
    assert 'train-diffusion' not in cli and 'train-denoiser' not in cli and 'cachekv' not in cli.lower()


def test_frozen_controls_are_never_positives_when_manifest_exists() -> None:
    path=ROOT/"results/cgdr/brainid_gate_v17r/frozen/control_usage_manifest.csv"
    if not path.exists(): pytest.skip("freeze stage not run yet")
    with path.open(newline="",encoding="utf-8") as handle: rows=list(csv.DictReader(handle))
    assert len(rows)==52;assert all(int(row["positive_pair_allowed"])==0 for row in rows)


def test_physio_and_artifact_keys_are_physically_separate_when_prepared() -> None:
    phys=ROOT/"results/cgdr/brainid_gate_v17r/server_arrays/physiological/subject_01.npz";artifact=ROOT/"results/cgdr/brainid_gate_v17r/server_arrays/artifact/subject_01.npz"
    if not phys.exists(): pytest.skip("prepare stage not run yet")
    with np.load(phys) as p,np.load(artifact) as a:
        assert not any(key.startswith("artifact_") for key in p.files);assert all(key.startswith("artifact_") for key in a.files)


def test_participant_first_m1r_reaggregation_when_available() -> None:
    root=ROOT/"results/cgdr/brainid_gate_v17r";path=root/"m1r_decision.json"
    if not path.exists(): pytest.skip("M1R not run yet")
    decision=json.loads(path.read_text(encoding="utf-8"))
    with (root/"m1r_verifier_subject_metrics.csv").open(newline="",encoding="utf-8") as handle: rows=list(csv.DictReader(handle))
    for name in ("A-R","B-R"):
        selected=[r for r in rows if r["verifier"]==name and r["panel"]=="physiological"]
        assert len(selected)==15 and len({int(r["participant"]) for r in selected})==15
        assert np.mean([float(r["auroc"]) for r in selected])==pytest.approx(decision["verifiers"][name]["physiological_auroc"],abs=1e-12)
