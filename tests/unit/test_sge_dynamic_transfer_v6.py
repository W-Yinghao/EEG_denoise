import numpy as np
import torch
from eeg_cgdr.experiments.sge_dynamic_transfer_v6 import apply_dynamic_transfer, fit_dynamic_transfer
from eeg_cgdr.models.dynamic_transfer_diffusion import DynamicTransferModelConfig, DynamicTransferDeterministic, DynamicTransferDiffusion

def test_fir_two_eog_roundtrip_and_lag_axis():
    rng=np.random.default_rng(4);eog=rng.normal(size=(2,300));h=np.zeros((3,2,5));h[0,0,0]=.7;h[1,1,2]=-1.1;h[2,0,4]=.4
    eeg=apply_dynamic_transfer(h,eog);estimated=fit_dynamic_transfer(eeg,eog,taps=5,ridge=1e-8);reconstructed=apply_dynamic_transfer(estimated,eog)
    assert np.linalg.norm(reconstructed-eeg)/np.linalg.norm(eeg)<.08
    assert estimated.shape==(3,2,5)

def test_matched_models_finite_common_noise_and_context_change():
    cfg=DynamicTransferModelConfig(4,width=16,blocks=2,timesteps=20,ddim_steps=5);det=DynamicTransferDeterministic(cfg);diff=DynamicTransferDiffusion(cfg)
    y=torch.randn(2,4,32);h=torch.randn(2,4,2,5)*.1;rho=torch.ones(2);target=torch.randn_like(y)*.05;g=torch.Generator().manual_seed(2)
    loss=diff.training_loss(target,observed=y,transfer=h,reliability=rho,generator=g)+(det(y,transfer=h,reliability=rho)-target).square().mean();loss.backward();assert torch.isfinite(loss)
    # primary implementation enforces K=8 and matching context changes output.
    out1=det(y,transfer=h,reliability=rho);out2=det(y,transfer=torch.roll(h,1,0),reliability=rho);assert not torch.allclose(out1,out2)

def test_label_zero_is_neither_clean_nor_artifact():
    labels=np.array([0,0,6,6,1,2])
    assert np.mean(labels==6)<.95
    assert np.mean(np.isin(labels,np.arange(1,6)))==2/6
