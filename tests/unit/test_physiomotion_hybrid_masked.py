import numpy as np
import pytest
import torch

from eeg_cgdr.models.hybrid_masked_diffusion import EMA, DeterministicHybridMasked, HybridMaskedConfig, HybridMaskedDiffusion, HybridMaskedUNet, parameter_count
from eeg_cgdr.experiments.physiomotion_hybrid_masked import _guard, _interpolate, _retrieve


def test_capacity_and_hard_identity():
    cfg=HybridMaskedConfig(base_channels=8)
    det=DeterministicHybridMasked(cfg);diff=HybridMaskedDiffusion(cfg)
    assert parameter_count(det)==parameter_count(diff)
    y=torch.randn(2,34,500);mask=torch.zeros_like(y);mask[:,:,100:130]=1;rp=torch.randn_like(y);res=torch.randn_like(y)
    out=det(y_obs=y,mask=mask,r_pop=rp,subject_residual=res)
    assert torch.equal(out[~mask.bool()],y[~mask.bool()])
    zero=torch.zeros_like(mask);assert torch.equal(det(y_obs=y,mask=zero,r_pop=rp,subject_residual=res),y)


def test_diffusion_common_noise_and_context_response():
    cfg=HybridMaskedConfig(base_channels=8);model=HybridMaskedDiffusion(cfg).eval();y=torch.randn(1,34,500);mask=torch.zeros_like(y);mask[:,:,200:240]=1;rp=torch.randn_like(y);res=torch.randn_like(y);noise=torch.randn_like(y)
    a=model.sample(y_obs=y,mask=mask,r_pop=rp,subject_residual=res,initial_noise=noise);b=model.sample(y_obs=y,mask=mask,r_pop=rp,subject_residual=res,initial_noise=noise);wrong=model.sample(y_obs=y,mask=mask,r_pop=rp,subject_residual=-res,initial_noise=noise)
    assert torch.equal(a,b);assert (a-wrong).abs().max()>0;assert torch.equal(a[~mask.bool()],y[~mask.bool()])


def test_interpolation_preserves_observed():
    y=np.random.default_rng(1).normal(size=(34,500)).astype(np.float32);mask=np.zeros_like(y,bool);mask[:,50:70]=1;out=_interpolate(y,mask);assert np.array_equal(out[~mask],y[~mask])


def test_sealed_guard(tmp_path):
    # The real frozen split is deliberately used: no sealed file is opened.
    c={"source_result_root":"/home/infres/yinwang/denoiseNet_physiomotion_subject_restoration/results/cgdr/physiomotion_subject_restoration"}
    with pytest.raises(PermissionError):_guard(c,[3])


def test_retrieval_aggregate_is_bank_order_invariant():
    rng=np.random.default_rng(12);banks={1:rng.normal(size=(8,34,500)).astype(np.float32),2:rng.normal(size=(8,34,500)).astype(np.float32)};query=rng.normal(size=(34,500)).astype(np.float32);mask=np.zeros_like(query,bool);mask[:,100:150]=1
    first,_=_retrieve(query,mask,banks,8);second,_=_retrieve(query,mask={2:banks[2],1:banks[1]} if False else mask,banks={2:banks[2],1:banks[1]},k=8)
    assert np.array_equal(first,second)


def test_ema_reload_owns_independent_tensors():
    first=DeterministicHybridMasked(HybridMaskedConfig(base_channels=8));state=EMA(first).state_dict();left=EMA(first);right=EMA(first);left.load_state_dict(state);right.load_state_dict(state);name=next(iter(left.shadow));left.shadow[name].add_(1)
    assert not torch.equal(left.shadow[name],right.shadow[name])


def test_inference_boundary_excludes_evaluator_fields():
    assert "clean_target" not in HybridMaskedUNet.visible_fields
    assert {"clean_target","query_annotation","oracle_indices","evaluator_outcome"} <= set(HybridMaskedUNet.forbidden_fields)
