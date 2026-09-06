import inspect
import numpy as np
import torch

from eeg_cgdr.experiments import bci2b_subject_diffusion_next as n

class Dummy:
    class C:
        timesteps=8;ddim_steps=4
    config=C()
    alpha_bar=torch.linspace(.99,.2,8)
    class B:
        def __call__(self,state,timestep,**kwargs):return torch.zeros_like(state)
    backbone=B()
    def sample(self,**kwargs):return kwargs["initial_noise"].clone()

def test_lambda_zero_is_exact_base_dispatch():
    d=Dummy();value=torch.randn(2,3,8)
    out=n.operator_posterior_sample(d,y=value,eog=value,a0=value,r_det=value,initial_noise=value,center=value,precision=torch.ones(1,3,1),strength=0)
    assert torch.equal(out,value)

def test_guided_padding_is_zero():
    d=Dummy();value=torch.randn(2,3,512)
    out=n.operator_posterior_sample(d,y=value,eog=value,a0=value,r_det=torch.zeros_like(value),initial_noise=value,center=torch.zeros_like(value),precision=torch.ones(1,3,1),strength=1)
    assert torch.count_nonzero(out[...,500:])==0

def test_task_mapping_and_primary_duration():
    c={"seeds":[20260808,20260810,20260811]}
    assert n._task(c,0)==(20260808,0)
    assert n._task(c,26)==(20260811,8)
    assert n.DURATIONS==(30,60,120)

def test_inference_does_not_open_evaluator():
    for fn in (n.stage_infer_pop8,n.stage_oppost_infer,n.stage_oppost_technical,n.stage_localized_infer):
        source=inspect.getsource(fn)
        assert "evaluator.npz" not in source and "paired_x" not in source and "natural_labels" not in source

def test_pop8_excludes_only_recipient():
    source=inspect.getsource(n.stage_prepare_pop8)
    assert "if x!=recipient" in source
    assert "training_participants\":8" in source

def test_wrong_is_scored_before_average():
    source=inspect.getsource(n._participant_effects)
    assert "wrong=[float(np.mean(v))" in source
    assert "np.mean(wrong)" in source

def test_causal_audit_has_oracle_and_det_cross():
    source=inspect.getsource(n.stage_causal_audit)
    assert "I_oracle" in source and "I_det_cross" in source and "I_full" in source

def test_frozen_score_fields_do_not_switch_with_context():
    source=inspect.getsource(n.stage_oppost_infer)
    assert 'a0=v11.apply_transfer(hp,eog)' in source
    assert 'rd=det(y=yt,eog=et,a0=at)' in source
    assert 'operator_posterior_sample' in source

def test_support_duration_is_not_silently_truncated():
    source=inspect.getsource(n.stage_infer_pop8)
    assert "support_duration_manifest.csv" in source
    assert "if not eligibility" in source
    assert "silently relabelling" in source
