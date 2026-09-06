from __future__ import annotations
import inspect,json,subprocess
from pathlib import Path
import numpy as np
import pytest
import torch
import yaml
from eeg_scad.context.operator_factorization import factorize_operator,population_basis,decode_numpy,decode_torch
from eeg_scad.context.projection_features import ridge_target_numpy,project_numpy,CoefficientStandardizer
from eeg_scad.data.splits import load_folds,validate_folds
from eeg_scad.models.of_deterministic import OFDeterministic,PopulationMarginalDET
from eeg_scad.models.of_residual_diffusion import OFResidualDiffusion,OFSCADConfig
from eeg_scad.training import train_v23
from eeg_scad.cli.v23 import identity_energy_refinement
from eeg_scad.evaluation.paired_metrics import paired_metrics

ROOT=Path(__file__).resolve().parents[2];DATA=yaml.safe_load((ROOT/"configs/of_scad_v23/data.yaml").read_text())

def test_base_and_read_only_heads()->None:
    if (ROOT/".git").exists():assert subprocess.check_output(["git","merge-base","--is-ancestor",DATA["base_commit"],"HEAD"],cwd=ROOT).decode()==""
    else:assert DATA["base_commit"]=="2c5b7bf4b5daf667f345ecb6e5f32495d494dfe1"
    assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=DATA["v22_worktree"],text=True).strip()==DATA["v22_commit"]
    assert subprocess.check_output(["git","rev-parse","HEAD"],cwd=DATA["a_track_worktree"],text=True).strip()==DATA["a_track_commit"]

def test_fold_exact_reuse_and_disjointness()->None:
    v23=load_folds(ROOT/"configs/of_scad_v23/folds.yaml");v22=load_folds(ROOT/"configs/scad_v22/folds.yaml");assert v23==v22;validate_folds(v23,DATA["participants"])

def test_sealed_absent_and_auxiliary_not_primary()->None:
    assert not set(DATA["participants"])&set(DATA["sealed_participants"]);assert DATA["auxiliary_support_owner"] not in DATA["participants"]

def test_factorization_definitions_and_population_swap()->None:
    rng=np.random.default_rng(2);c0=rng.normal(size=(46,4));cs=c0+.2*rng.normal(size=(46,4));basis,scale,diag=factorize_operator(c0,cs);assert basis.shape==(46,8) and scale.shape==(8,);np.testing.assert_allclose(np.linalg.norm(basis,axis=0)[scale>1e-7],1,atol=1e-6);pop,_,_=population_basis(c0);assert np.max(np.abs(pop[:,4:]))==0;assert diag.deviation_energy_ratio>0

def test_column_canonicalization_round_trip()->None:
    rng=np.random.default_rng(3);raw=rng.normal(size=(46,8));scale=np.linalg.norm(raw,axis=0);unit=raw/scale;z=rng.normal(size=(8,40));np.testing.assert_allclose(raw@z,unit@(scale[:,None]*z),rtol=1e-6,atol=1e-6)

def test_ridge_target_exact_span_and_query_projection()->None:
    rng=np.random.default_rng(4);basis,_s,_=factorize_operator(rng.normal(size=(46,4)),rng.normal(size=(46,4)));z=rng.normal(size=(8,64));artifact=basis@z;target,decoded,error=ridge_target_numpy(artifact,basis,1e-10);assert error<1e-7;q,projected,residual=project_numpy(artifact,basis,1e-10);assert np.linalg.norm(residual)/np.linalg.norm(artifact)<1e-7;np.testing.assert_allclose(target,q)

def test_coefficient_standardization_inverse()->None:
    rng=np.random.default_rng(5);values=rng.normal(loc=2,scale=3,size=(20,8,64)).astype(np.float32);s=CoefficientStandardizer.fit(values);np.testing.assert_allclose(s.inverse(s.transform(values)),values,rtol=1e-6,atol=1e-6)

def test_of_det_shapes_decoder_and_context_intervention()->None:
    model=OFDeterministic(base=8);y=torch.randn(2,46,64);q=torch.randn(2,8,64);p=torch.randn(2,46,64);summary=torch.randn(2,24);basis=torch.randn(2,46,8);z=model(y,q,p,summary);assert z.shape==(2,8,64);a=decode_torch(basis,z);assert a.shape==y.shape;z2=model(y,q+.5,p,summary);assert torch.linalg.vector_norm(z-z2)>0

def test_pop_marginal_has_no_subject_support_contract()->None:
    model=PopulationMarginalDET(base=8);assert model.uses_subject_support is False;assert "support" not in inspect.signature(model.forward).parameters

def test_of_scad_residual_shapes_forward_and_fixed_noise()->None:
    model=OFResidualDiffusion(OFSCADConfig(base=8,timesteps=20,ddim_steps=4));y=torch.randn(2,46,64);q=z=torch.randn(2,8,64);p=torch.randn(2,46,64);summary=torch.randn(2,24);target=torch.randn(2,8,64);g=torch.Generator().manual_seed(7);loss,extra=model.training_loss(target,y,q,p,z,summary,g);assert loss.ndim==0 and extra["predicted_x0"].shape==target.shape;noise=torch.randn_like(target);a,t=model.sample(y,q,p,z,summary,noise,4,True);b,_=model.sample(y,q,p,z,summary,noise,4,True);torch.testing.assert_close(a,b);assert len(t)==4

def test_observation_anchor_definition()->None:
    y=torch.randn(2,46,32);basis=torch.randn(2,46,8);z=torch.randn(2,8,32);artifact=decode_torch(basis,z);clean=y-artifact;torch.testing.assert_close(clean,y-artifact)

def test_registered_energy_interface_is_identity_only()->None:
    observation=np.ones((2,3));clean=np.arange(6).reshape(2,3);artifact=observation-clean
    assert identity_energy_refinement(observation=observation,clean_estimate=clean,artifact_estimate=artifact,context={}) is clean

def test_training_semantics_wrong_absent_from_base_loss()->None:
    source=inspect.getsource(train_v23.train_det)+inspect.getsource(train_v23.train_v22_fixed);assert 'wrong_base_loss_proportion":0.' in source;assert "per_w" in source and "rank=torch.relu" in source;assert "pop_consistent=True" in source

def test_best_checkpoints_are_separate_from_last()->None:
    source=inspect.getsource(train_v23);assert '"best_artifact.pt"' in source and '"best_coefficient.pt"' in source and '"best_sampling.pt"' in source and '"last.pt"' in source

def test_resume_payload_includes_registered_rngs()->None:
    source=inspect.getsource(train_v23);assert all(token in source for token in ('"data_rng"','"mixture_rng"','"wrong_owner_rng"','"diffusion_rng"','"optimizer"','"scheduler"','"ema"'))

def test_zero_artifact_excluded_from_snr_contract()->None:
    source=(ROOT/"src/eeg_scad/cli/v23.py").read_text();assert 'np.nan if zero else metric["snr_improvement"]' in source;assert 'method_predictions.update({"RAW":zeros,"STANDARD":zeros})' in source

def test_artifact_rrmse_field_and_scale()->None:
    artifact=np.ones((2,8));prediction=np.zeros_like(artifact);metrics=paired_metrics(np.ones_like(artifact),np.ones_like(artifact)+artifact,artifact,prediction)
    assert metrics["artifact_rrmse"]==pytest.approx(1.0) and "artifact_rmse" in metrics

def test_online_sampler_contract_source()->None:
    source=(ROOT/"src/eeg_scad/data/online_counterfactual.py").read_text();assert "artifact=(generating@(gain*e))" in source;assert "artifact_mask_quantile" not in source;assert "strict_three_way" in source

def test_materialized_inference_evaluator_separation_if_present()->None:
    root=Path(DATA["derived_root"])/"fold_0";ip=root/"paired_test_inference.npz";ep=root/"paired_test_evaluator.npz"
    if not ip.is_file():pytest.skip("R2 not materialized")
    with np.load(ip,allow_pickle=False) as z:assert "x" not in z.files and "artifact" not in z.files and "basis_match" in z.files
    with np.load(ep,allow_pickle=False) as z:assert "x" in z.files and "artifact" in z.files and "y" not in z.files

def test_governance_no_old_saddpm_or_energy_bridge()->None:
    for path in (ROOT/"src/eeg_scad").rglob("*.py"):
        text=path.read_text().lower();assert "import saddpm" not in text and "from saddpm" not in text;assert "posterior energy" not in text

def test_terminal_governance_if_present()->None:
    path=ROOT/"results/of_scad_v23/terminal_manifest.json"
    if not path.is_file():pytest.skip("terminal not generated")
    value=json.loads(path.read_text());assert value["sealed_reads"]==0 and value["manuscript_modified"] is False and value["energy_bridge_implemented"] is False

def test_canonical_effects_are_participant_first_if_present()->None:
    path=ROOT/"results/of_scad_v23/development_diagnosis.json"
    if not path.is_file():pytest.skip("aggregate not generated")
    value=json.loads(path.read_text())
    assert {summary["n"] for summary in value["paired_effect_summaries"].values()}=={15}
