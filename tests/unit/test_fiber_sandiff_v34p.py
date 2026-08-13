from pathlib import Path

import numpy as np
import torch

from eeg_scad.privacy.fiber import FiberOneStep, FiberSANDiff, HeadFiber
from eeg_scad.privacy.fiber_experiment import exact_preservation, sanitize_fiber
from eeg_scad.privacy.models import EEGNetRepresentation


ROOT=Path(__file__).resolve().parents[2]
BASE="5292c1a552ca3fd5980f37291cd53a98ab6d01ea"


def fixture():
    torch.manual_seed(34);head=torch.nn.Linear(128,4);geometry=HeadFiber.from_linear(head);z=np.random.default_rng(34).normal(size=(32,128)).astype(np.float32);return head,geometry,z


def test_centered_head_rank_and_fiber_dimension():
    _,geometry,_=fixture();assert geometry.rank==3 and geometry.fiber_dim==125


def test_exact_decomposition_round_trip():
    _,geometry,z=fixture();z_head,u,_=geometry.decompose(z);np.testing.assert_allclose(geometry.compose(z_head,u),z,rtol=1e-5,atol=2e-6)


def test_nullspace_contract():
    _,geometry,_=fixture();diagnostics=geometry.diagnostics();assert diagnostics["null_residual_max_abs"]<1e-10 and diagnostics["null_orthogonality_max_abs"]<1e-10


def test_arbitrary_fiber_change_preserves_function():
    _,geometry,z=fixture();z_head,u,_=geometry.decompose(z);changed=geometry.compose(z_head,np.random.default_rng(2).normal(size=u.shape));result=exact_preservation(geometry,z,changed,np.arange(len(z))%4);assert result["prediction_mismatch_count"]==0;assert result["max_softmax_probability_error"]<1e-6


def test_strength_zero_is_raw_and_strong_uses_replacement():
    _,geometry,z=fixture();model=FiberOneStep(geometry.fiber_dim)
    zero,_,_=sanitize_fiber(model,"Fiber-OneStep",geometry,z,torch.device("cpu"),1,0.0);strong,_,_=sanitize_fiber(model,"Fiber-OneStep",geometry,z,torch.device("cpu"),1,1.0)
    np.testing.assert_allclose(zero,z,rtol=1e-5,atol=2e-6);assert not np.allclose(strong,z)


def test_fiber_diffusion_k1_ten_step_fixed_replay():
    model=FiberSANDiff(125).eval();h=torch.randn(3,4);noise=torch.randn(3,125);first=model.sample(h,reverse_steps=10,noise=noise);second=model.sample(h,reverse_steps=10,noise=noise);torch.testing.assert_close(first,second)


def test_models_do_not_accept_subject_identity():
    one=FiberOneStep(125);diff=FiberSANDiff(125);assert one(torch.randn(2,4)).shape==(2,125);assert diff(torch.randn(2,125),torch.randn(2,4),torch.ones(2,dtype=torch.long)).shape==(2,125)


def test_eegnet_contract_is_unchanged():
    model=EEGNetRepresentation();logits,z=model(torch.randn(2,22,512),return_representation=True);assert logits.shape==(2,4) and z.shape==(2,128)


def test_config_governance_and_base():
    text=(ROOT/"configs"/"fiber_sandiff_v34p.yaml").read_text();assert BASE in text and "waveform_sealed_reads: 0" in text and "new_encoder: false" in text and "new_diffusion_family: false" in text


def test_ledger_increment_merge_preserves_v30_and_v31():
    text=(ROOT/"docs"/"TAAS_SUBJECT_AWARE_DIFFUSION_PROJECT_LEDGER.md").read_text();assert "# v3.1" in text and "# v3.0" in text and "V34P" in text


def test_outer_test_not_used_for_selection_source_contract():
    source=(ROOT/"src"/"eeg_scad"/"privacy"/"fiber_experiment.py").read_text();assert '"outer_test_used_for_selection":False' in source


def test_full_sampler_selection_source_contract():
    source=(ROOT/"src"/"eeg_scad"/"privacy"/"fiber_experiment.py").read_text();assert '"full_10_step"' in source and "reverse_steps=10" in source
