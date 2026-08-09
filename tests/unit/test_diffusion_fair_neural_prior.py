import numpy as np
import torch

from eeg_cgdr.experiments.diffusion_fair_neural_prior import _airm,_align,_covariance,_invsqrt,_logeuclidean,_sign_flip
from eeg_cgdr.models.eog_residual_diffusion import CapacityMatchedDeterministic,EOGResidualConfig,EOGResidualDiffusion

def test_capacity_matched_backbone_parameters_and_determinism():
    cfg=EOGResidualConfig(base_channels=8);det=CapacityMatchedDeterministic(cfg);diff=EOGResidualDiffusion(cfg)
    assert sum(p.numel() for p in det.backbone.parameters())==sum(p.numel() for p in diff.backbone.parameters())
    det.eval();fields={"y":torch.randn(2,3,512),"eog":torch.randn(2,3,512),"a0":torch.randn(2,3,512),"r_det":torch.randn(2,3,512)}
    with torch.no_grad():first=det(**fields);second=det(**fields)
    assert torch.equal(first,second)

def test_spd_distances_and_covariance_are_finite():
    rng=np.random.default_rng(2);a=_covariance(rng.normal(size=(4,1000)));b=_covariance(rng.normal(size=(4,1000)))
    assert np.all(np.linalg.eigvalsh(a)>0)
    assert _airm(a,a)<1e-10 and _logeuclidean(a,a)<1e-10
    assert _airm(a,b)>0 and _logeuclidean(a,b)>0

def test_exact_sign_flip_uses_participants():
    assert _sign_flip(np.ones(9),one_sided=True)==1/512
    assert _sign_flip(np.ones(9),one_sided=False)==2/512

def test_neural_alignment_roundtrip_and_arm_boundary():
    rng=np.random.default_rng(4);cov=_covariance(rng.normal(size=(3,2000)));matrix=_invsqrt(cov);value=rng.normal(size=(5,3,512)).astype(np.float32);restored=_align(_align(value,matrix),np.linalg.inv(matrix))
    assert np.max(np.abs(restored-value))<1e-4
    assert not np.allclose(_align(value,matrix),value)

def test_visible_fields_forbid_query_outcomes():
    assert "query_outcomes" in CapacityMatchedDeterministic.forbidden_fields
    assert "query_labels" in CapacityMatchedDeterministic.forbidden_fields
