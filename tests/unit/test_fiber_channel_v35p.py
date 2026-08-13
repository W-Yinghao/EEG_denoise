import inspect
from pathlib import Path

import numpy as np
import torch

from eeg_scad.privacy.fiber import FiberOneStep, HeadFiber
from eeg_scad.privacy.fiber_channel import FiberStratifiedResampler, compose_strong_release, multisample_diagnostics, strong_model_replacement
from eeg_scad.privacy.fiber_experiment import exact_preservation


ROOT=Path(__file__).resolve().parents[2];BASE="e10dd40100e60f5e47c4d1a917ec4515880fc9ca"


def fixture():
    rng=np.random.default_rng(35);head=torch.nn.Linear(128,4);geometry=HeadFiber.from_linear(head);z=rng.normal(size=(60,128)).astype(np.float32);z_head,u,h=geometry.decompose(z);return rng,geometry,z,z_head,u,h


def test_base_and_governance():
    text=(ROOT/"configs"/"fiber_channel_v35p.yaml").read_text();assert BASE in text and "latency_used: false" in text and "retrain_eegnet: false" in text and "waveform_sealed_reads: 0" in text


def test_strong_replacement_signature_has_no_source_u():
    assert "source_u" not in inspect.signature(strong_model_replacement).parameters


def test_resampler_signature_has_no_source_u_or_subject():
    parameters=inspect.signature(FiberStratifiedResampler.sample).parameters;assert "source_u" not in parameters and "subject" not in parameters


def test_resampler_uses_training_fibers_only():
    _,_,_,_,u,h=fixture();sampler=FiberStratifiedResampler.fit(u[:40],h[:40]);released,coverage=sampler.sample(h[40:],seed=1);assert all(any(np.array_equal(row,training) for training in u[:40]) for row in released);assert max(item["donor_training_index"] for item in coverage)<40


def test_resample_strata_are_training_only():
    _,_,_,_,u,h=fixture();sampler=FiberStratifiedResampler.fit(u[:40],h[:40]);before=dict(sampler.tertiles);sampler.sample(h[40:]*1000,seed=2);assert sampler.tertiles==before


def test_resample_replay_and_seed_variation():
    _,_,_,_,u,h=fixture();sampler=FiberStratifiedResampler.fit(u[:40],h[:40]);a,_=sampler.sample(h[40:],seed=3);b,_=sampler.sample(h[40:],seed=3);c,_=sampler.sample(h[40:],seed=4);np.testing.assert_array_equal(a,b);assert not np.array_equal(a,c)


def test_exact_H_recovery_for_resample():
    _,geometry,z,z_head,u,h=fixture();sampler=FiberStratifiedResampler.fit(u[:40],h[:40]);replacement,_=sampler.sample(h[40:],seed=5);released=compose_strong_release(geometry,z_head[40:],replacement);result=exact_preservation(geometry,z[40:],released,np.arange(20)%4);assert result["prediction_mismatch_count"]==0 and result["max_softmax_probability_error"]<1e-6


def test_one_step_multisample_is_zero_diversity():
    _,_,_,_,u,_=fixture();release=np.repeat(u[40:][None],16,axis=0);result=multisample_diagnostics("one",release,u[:40]);assert result["within_H_sample_variance"]==0 and result["duplicate_rate"]==1 and result["sample_selection"]=="all_16_registered_releases"


def test_sixteen_release_contract():
    text=(ROOT/"configs"/"fiber_channel_v35p.yaml").read_text();assert "releases_per_query: 16" in text


def test_head_aware_features_are_explicit():
    source=(ROOT/"src"/"eeg_scad"/"privacy"/"fiber_channel.py").read_text();assert '"A_H"' in source and '"A_Z"' in source and '"A_HZ"' in source and '"A_HU"' in source


def test_participant_session_attack_contract():
    text=(ROOT/"configs"/"fiber_channel_v35p.yaml").read_text();assert "outer-test Session T" in text and "outer-test Session E" in text


def test_no_latency_benchmark_source():
    source=(ROOT/"src"/"eeg_scad"/"cli"/"run_v35p.py").read_text()+(ROOT/"src"/"eeg_scad"/"privacy"/"fiber_channel.py").read_text();assert "_latency(" not in source and "latency_benchmark_run\":False" in source


def test_ledger_increment_preserves_v32_and_v33():
    text=(ROOT/"docs"/"TAAS_SUBJECT_AWARE_DIFFUSION_PROJECT_LEDGER.md").read_text();assert "# v3.3" in text and "# v3.2" in text and "V35P" in text


def test_frozen_one_step_interface():
    model=FiberOneStep(125);assert model(torch.randn(2,4)).shape==(2,125)
