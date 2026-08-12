from __future__ import annotations
import inspect,json
from pathlib import Path
import numpy as np
import torch
import yaml
from eeg_scad.context.deepsets_encoder import DeepSetsSupportEncoder
from eeg_scad.context.set_transformer_encoder import SetTransformerSupportEncoder
from eeg_scad.context.learned_spatial_decoder import decode_residual,normalize_basis,ridge_latent
from eeg_scad.models.setcalib_det import SetCalibDET
from eeg_scad.models.setcalib_diff import SetCalibResidualDiffusion
from eeg_scad.data.v24_coordinate_contract import canonical_operator

ROOT=Path(__file__).resolve().parents[2]
def test_ledger_v11_and_active_route():
    text=(ROOT/"docs/TAAS_SUBJECT_AWARE_DIFFUSION_PROJECT_LEDGER.md").read_text();assert "**版本：** v1.1" in text and "V25 SetCalibDiff" in text
def test_base_sha():assert yaml.safe_load((ROOT/"configs/setcalibdiff_v25/data.yaml").read_text())["base_commit"]=="8dadb508fd2d50a089246c4e11c83b7b7628fa42"
def test_coordinate_contract_equivalence():
    rng=np.random.default_rng(2);c=rng.normal(size=(46,4));dy=np.exp(rng.normal(size=46));de=np.exp(rng.normal(size=4));e=rng.normal(size=(4,200));assert np.allclose((c@e)/dy[:,None],canonical_operator(c,dy,de)@(e/de[:,None]))
def test_deepsets_permutation_invariance():
    model=DeepSetsSupportEncoder().eval();eeg=torch.randn(2,5,46,40);eog=torch.randn(2,5,4,40);order=torch.tensor([3,0,4,1,2]);a=model(eeg,eog);b=model(eeg[:,order],eog[:,order]);assert torch.allclose(a["context"],b["context"],atol=1e-5)
def test_set_transformer_order_robustness():
    model=SetTransformerSupportEncoder().eval();eeg=torch.randn(2,5,46,40);eog=torch.randn(2,5,4,40);order=torch.tensor([3,0,4,1,2]);assert torch.allclose(model(eeg,eog)["context"],model(eeg[:,order],eog[:,order])["context"],atol=1e-5)
def test_basis_unit_norm():
    result=DeepSetsSupportEncoder()(torch.randn(2,4,46,32),torch.randn(2,4,4,32));assert torch.allclose(torch.linalg.vector_norm(result["basis"],dim=1),torch.ones(2,8),atol=1e-5)
def test_pop_exact_identity():
    y=torch.randn(2,46,64);a=torch.randn_like(y);assert torch.equal(SetCalibDET.population(y,a)["artifact"],a)
def test_match_wrong_change_output():
    model=SetCalibDET().eval();y=torch.randn(2,46,64);a=torch.randn_like(y);q=torch.randn(2,4,64);eeg=torch.randn(2,4,46,32);eog=torch.randn(2,4,4,32);assert not torch.equal(model(y,a,q,eeg,eog)["artifact"],model(y,a,q,eeg.flip(1),eog.roll(1,1))["artifact"])
def test_ridge_latent_and_decode():
    b=normalize_basis(torch.randn(2,46,8));h=torch.randn(2,8,64);target=torch.einsum("bcr,brt->bct",b,h);fit=ridge_latent(target,b,1e-8);assert torch.allclose(torch.einsum("bcr,brt->bct",b,fit),target,atol=1e-4)
def test_diffusion_shape_and_fixed_noise():
    model=SetCalibResidualDiffusion().eval();y=torch.randn(2,46,64);a=torch.randn_like(y);q=torch.randn(2,4,64);h=torch.randn(2,8,64);c=torch.randn(2,128);noise=torch.randn_like(h);x=model.sample(y,a,q,h,c,noise,10);z=model.sample(y,a,q,h,c,noise,10);assert x.shape==h.shape and torch.equal(x,z)
def test_zero_artifact_metric_contract():assert "artifact_rrmse" in yaml.safe_load((ROOT/"configs/setcalibdiff_v25/evaluation.yaml").read_text())["zero_artifact_excluded_from"]
def test_wrong_has_no_base_supervision():
    source=inspect.getsource(__import__("eeg_scad.training.train_v25",fromlist=["train_det"]).train_det);assert "wper" in source and "wrong" in source and "error=(out" in source
def test_temporal_query_has_no_auxiliary():assert SetCalibDET.forbidden_fields==("query_EOG","query_operator","query_event","subject_ID")
def test_fold_disjointness():
    folds=yaml.safe_load((ROOT/"configs/setcalibdiff_v25/folds.yaml").read_text())["folds"]
    for fold in folds:assert not(set(fold["train"])&set(fold["validation"])) and not(set(fold["train"])&set(fold["test"])) and not(set(fold["validation"])&set(fold["test"]))
def test_sealed_absent():
    data=yaml.safe_load((ROOT/"configs/setcalibdiff_v25/data.yaml").read_text());folds=yaml.safe_load((ROOT/"configs/setcalibdiff_v25/folds.yaml").read_text())["folds"];assert not set(data["sealed_participants"])&set(sum([f["train"]+f["validation"]+f["test"] for f in folds],[]))
def test_support_query_contract():
    data=yaml.safe_load((ROOT/"configs/setcalibdiff_v25/data.yaml").read_text());assert data["support_samples"]<data["qnatural_start"]
def test_checkpoint_resume_rng_fields():
    source=inspect.getsource(__import__("eeg_scad.training.train_v25",fromlist=["train_det"]).train_det)
    for key in ("support_window_rng","wrong_support_rng","paired_mixture_rng","natural_stream_rng"):assert key in source
def test_no_energy_or_identity_classifier():
    source=(ROOT/"src/eeg_scad/models/setcalib_det.py").read_text();assert "energy_bridge" not in source and "ArcFace" not in source
def test_v24_source_is_external_read_only():assert yaml.safe_load((ROOT/"configs/setcalibdiff_v25/data.yaml").read_text())["v24_worktree"]!="."
def test_no_k8_configured():assert "K8" not in (ROOT/"configs/setcalibdiff_v25/setcalib_diff.yaml").read_text()
def test_natural_inference_uses_auxiliary_free_v24_bundle():
    source=inspect.getsource(__import__("eeg_scad.cli.v25",fromlist=["natural_infer"]).natural_infer)
    assert "natural_test_inference.npz" in source
    assert "sample_natural" not in source
    assert "natural_test_evaluator.npz" not in source
def test_evaluator_namespace_opens_only_after_freeze():
    source=inspect.getsource(__import__("eeg_scad.cli.v25",fromlist=["natural_eval"]).natural_eval)
    assert "output_freeze.json" in source and "natural_test_evaluator.npz" in source
def test_support_bank_builder_has_no_query_operator_path():
    source=inspect.getsource(__import__("eeg_scad.data.support_set_episodes",fromlist=["NaturalSupportBankBuilder"]).NaturalSupportBankBuilder)
    assert "query_operator" not in source and "sample_natural" not in source
