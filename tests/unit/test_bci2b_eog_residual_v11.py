import numpy as np
import torch

from eeg_cgdr.experiments.bci2b_eog_residual_v11 import apply_transfer,gamma_correction,support_gamma,temporal_shuffle
from eeg_cgdr.models.eog_residual_diffusion import EOGResidualConfig,EOGResidualDiffusion


def test_gamma_zero_is_exact_raw():
    y=np.arange(24,dtype=np.float32).reshape(2,3,4);c=np.ones_like(y)
    assert np.array_equal(gamma_correction(y,c,0),y)


def test_support_gamma_recovers_known_scale():
    correction=np.arange(1,25,dtype=np.float32).reshape(2,3,4)
    assert abs(support_gamma(.4*correction,correction)-.4)<1e-7


def test_temporal_shuffle_preserves_values_but_not_time_order():
    eog=np.arange(2*3*32,dtype=np.float32).reshape(2,3,32);shuffled=temporal_shuffle(eog,19)
    assert not np.array_equal(eog,shuffled)
    for i in range(2):
        for c in range(3):assert np.array_equal(np.sort(eog[i,c]),np.sort(shuffled[i,c]))


def test_v10_correction_wording_is_exact():
    text=open("reports/v10_scientific_status_correction.md",encoding="utf-8").read()
    assert "OPERATOR_IDENTITY_SPECIFICITY_PRESENT_BUT_BASE_DENOISING_PIPELINE_INVALID" in text
    assert "EOG-guided" in text


def test_oracle_sampler_roundtrip():
    model=EOGResidualDiffusion(EOGResidualConfig(base_channels=8));target=torch.randn(2,3,512);noise=torch.randn_like(target)
    assert model.oracle_roundtrip(target,noise)<1e-6


def test_inference_does_not_open_evaluator_arrays():
    import inspect
    from eeg_cgdr.experiments.bci2b_eog_residual_v11 import stage_infer_fold
    source=inspect.getsource(stage_infer_fold)
    assert "evaluator.npz" not in source
    assert 'f"{panel}_eog"' in source
    assert 'query_eog_used":True' in source


def test_batched_transfer_matches_scalar_route():
    rng=np.random.default_rng(4);operators=rng.normal(size=(4,3,3)).astype(np.float32);eog=rng.normal(size=(4,3,16)).astype(np.float32)
    expected=np.stack([apply_transfer(operators[i],eog[i:i+1])[0] for i in range(4)])
    np.testing.assert_allclose(apply_transfer(operators,eog),expected,rtol=1e-6,atol=1e-6)
