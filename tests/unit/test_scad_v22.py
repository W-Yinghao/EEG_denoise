from __future__ import annotations
import json,subprocess
import csv
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml

from eeg_scad.context.operator_normalization import canonical_operator_features,canonicalize_operator
from eeg_scad.data.splits import load_folds,validate_folds
from eeg_scad.evaluation.paired_metrics import paired_metrics
from eeg_scad.evaluation.aggregate import participant_metrics
from eeg_scad.models.deterministic_artifact_unet import DeterministicArtifactEstimator
from eeg_scad.models.diffusion_schedule import cosine_alpha_bar
from eeg_scad.models.scad_artifact_diffusion import SCADArtifactDiffusion,SCADConfig,identity_postprocessor

ROOT=Path(__file__).resolve().parents[2]
DATA=yaml.safe_load((ROOT/"configs/scad_v22/data.yaml").read_text())


def test_five_fold_participant_disjointness()->None:
    folds=load_folds(ROOT/"configs/scad_v22/folds.yaml");validate_folds(folds,DATA["participants"])


def test_sealed_participants_absent()->None:assert not set(DATA["participants"])&set(DATA["sealed_participants"])


def test_support_query_contract_disjoint()->None:assert DATA["support_samples"]<DATA["qgen_start"]<DATA["qgen_stop"]<DATA["qnatural_start"]


def test_operator_feature_dimension_and_order()->None:
    value=np.arange(46*4,dtype=float).reshape(46,4);feature=canonical_operator_features(value);assert feature.shape==(189,)


def test_operator_standard_coordinate_equivalence()->None:
    rng=np.random.default_rng(1);c=rng.normal(size=(46,4));scale=np.arange(1,5);z=rng.normal(size=(4,20));np.testing.assert_allclose(c@(scale[:,None]*z),canonicalize_operator(c,scale)@z)


def test_det_output_shape_and_observation_anchor()->None:
    model=DeterministicArtifactEstimator(base_channels=8);y=torch.randn(2,46,64);ctx=torch.randn(2,189);artifact=model(y,ctx);assert artifact.shape==y.shape;torch.testing.assert_close(model.clean(y,ctx),y-artifact)


def test_scad_x0_artifact_parameterization()->None:
    model=SCADArtifactDiffusion(SCADConfig(base_channels=8,timesteps=20,ddim_steps=4));y=torch.randn(2,46,64);a=torch.randn_like(y);ctx=torch.randn(2,189);g=torch.Generator().manual_seed(2);loss,extra=model.training_loss(a,y,ctx,g);assert loss.ndim==0 and extra["predicted_x0"].shape==a.shape


def test_diffusion_forward_marginal_bounds()->None:
    alpha=cosine_alpha_bar(1000);assert alpha.shape==(1000,) and 0<alpha[-1]<alpha[0]<=1 and torch.all(alpha[1:]<=alpha[:-1])


def test_ddim_steps_and_fixed_noise_replay()->None:
    model=SCADArtifactDiffusion(SCADConfig(base_channels=8,timesteps=20,ddim_steps=4));y=torch.randn(1,46,64);ctx=torch.randn(1,189);noise=torch.randn_like(y);a,t=model.sample(y,ctx,noise,4,True);b,_=model.sample(y,ctx,noise,4,True);torch.testing.assert_close(a,b);assert len(t)==4


def test_k_sample_waveform_mean()->None:
    model=SCADArtifactDiffusion(SCADConfig(base_channels=8,timesteps=10,ddim_steps=2));y=torch.randn(1,46,64);ctx=torch.randn(1,189);bank=torch.randn(2,1,46,64);value=model.sample_mean(y,ctx,bank,2);expected=torch.stack([model.sample(y,ctx,bank[k],2)[0] for k in range(2)]).mean(0);torch.testing.assert_close(value,expected)


def test_film_context_swap_changes_output()->None:
    model=DeterministicArtifactEstimator(base_channels=8);y=torch.randn(1,46,64);a=model(y,torch.zeros(1,189));b=model(y,torch.ones(1,189));assert torch.linalg.vector_norm(a-b)>0


def test_same_checkpoint_supports_all_context_arms()->None:
    model=DeterministicArtifactEstimator(base_channels=8);y=torch.randn(1,46,64);contexts=[torch.randn(1,189) for _ in range(4)];assert all(model(y,c).shape==y.shape for c in contexts)


def test_identity_postprocessor_is_identity()->None:
    value=torch.randn(1,46,32);assert identity_postprocessor(observation=value,clean_estimate=value,artifact_estimate=value,context=torch.zeros(1,189)) is value


def test_paired_metrics_perfect_recovery()->None:
    rng=np.random.default_rng(3);x=rng.normal(size=(46,64));a=rng.normal(size=(46,64));m=paired_metrics(x,x+a,a,a);assert m["rrmse_temporal"]==pytest.approx(0) and m["artifact_correlation"]==pytest.approx(1)


def test_model_forbidden_query_fields()->None:
    assert "query_EOG" in DeterministicArtifactEstimator.forbidden_fields and "query_operator" in SCADArtifactDiffusion.forbidden_fields


def test_no_old_saddpm_import_in_authoritative_package()->None:
    for path in (ROOT/"src/eeg_scad").rglob("*.py"):assert "from saddpm" not in path.read_text() and "import saddpm" not in path.read_text()


def test_a_track_and_read_only_source_heads()->None:
    for key,commit in (("v19_worktree",DATA["v19_commit"]),("v20_worktree",DATA["v20_commit"]),("o1_worktree",DATA["o1_commit"]),("a_track_worktree",DATA["a_track_commit"])):assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=DATA[key],text=True).strip()==commit
    assert subprocess.run(["git","diff","--quiet","HEAD","--","taas_submission"],cwd=DATA["a_track_worktree"]).returncode==0


def test_natural_inference_bundle_has_no_eog_if_materialized()->None:
    path=Path(DATA["derived_root"])/"fold_0/natural_inference.npz"
    if not path.is_file():pytest.skip("R2 not run")
    with np.load(path,allow_pickle=False) as z:assert "eog" not in z.files and "C_query" not in z.files


def test_paired_inference_and_evaluator_are_physically_separate_if_materialized()->None:
    root=Path(DATA["derived_root"])/"fold_0";inference=root/"paired_test_inference.npz";evaluator=root/"paired_test_evaluator.npz"
    if not inference.is_file():pytest.skip("R2 not run")
    with np.load(inference,allow_pickle=False) as z:assert "x" not in z.files and "artifact" not in z.files and {"y","context_match","context_pop","context_wrong"}<=set(z.files)
    with np.load(evaluator,allow_pickle=False) as z:assert set(z.files)=={"x","artifact"}


def test_participant_first_reduction_does_not_use_windows_as_science_units()->None:
    rows=[]
    for session,value in (("s1",1.),("s2",3.)):
        for _ in range(9 if session=="s1" else 1):rows.append({"participant":"p1","session":session,"task":"ERP","seed":1,"method":"M","metric":value})
    reduced=participant_metrics(rows,{"metric":-1},"fixture");assert len(reduced)==1 and reduced[0]["metric"]==pytest.approx(2.)


def test_third_party_sources_are_pinned_and_not_vendored()->None:
    registry=yaml.safe_load((ROOT/"third_party/source_registry.yaml").read_text())
    commits={row["method"]:row["commit"] for row in registry["sources"]};assert len(commits["EEGDfus"])==40 and len(commits["D4PM"])==40
    assert not (ROOT/"third_party/EEGDfus").exists() and not (ROOT/"third_party/D4PM").exists()


def test_counterfactual_role_manifest_if_materialized()->None:
    path=ROOT/"results/scad_v22/counterfactual_role_manifest.csv"
    if not path.is_file():pytest.skip("R2 not run")
    with path.open(newline="") as f:rows=list(csv.DictReader(f))
    assert rows and all(r["support_query_disjoint"]==r["query_operator_evaluator_only"]=="1" for r in rows)


def test_terminal_governance_if_available()->None:
    path=ROOT/"results/scad_v22/terminal_manifest.json"
    if not path.is_file():pytest.skip("terminal not run")
    value=json.loads(path.read_text());assert value["sealed_reads"]==0 and value["manuscript_modified"] is False
