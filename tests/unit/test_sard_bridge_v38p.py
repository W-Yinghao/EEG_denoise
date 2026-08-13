from __future__ import annotations

import inspect
import subprocess
from pathlib import Path

import numpy as np
import torch

from eeg_scad.privacy.openbmi import OpenBMITrials, outer_folds, validate_folds
from eeg_scad.privacy.sard_bridge import BridgeScaler, DonorBank, FrozenSemantics, GaussianBridge, OneStepBridge, SARDBridge, support_context
from eeg_scad.privacy.sard_experiment import split_support


ROOT = Path(__file__).resolve().parents[2]
BASE = "89effec0abd8c0b3581c89dc6bfeed9e68b2cafe"


def fixture():
    rng=np.random.default_rng(38); subjects=np.repeat(np.arange(6),20); logits=np.tile(np.asarray([[3.,-1.],[-1.,3.]],dtype=np.float32),(60,1)); z=rng.normal(size=(120,128)).astype(np.float32); return rng,z,logits,subjects


def test_base_ledger_and_frozen_boundaries():
    if (ROOT/".git").exists(): assert subprocess.check_output(["git","merge-base","--is-ancestor",BASE,"HEAD"],cwd=ROOT,text=True)==""
    ledger=(ROOT/"docs/TAAS_SUBJECT_AWARE_DIFFUSION_PROJECT_LEDGER.md").read_text();assert "**版本：** v4.0" in ledger and "V38R已被supersede且未执行" in ledger
    changed=subprocess.check_output(["git","diff","--name-only",BASE,"--","taas_submission"],cwd=ROOT,text=True);assert changed==""


def test_54_participants_outer_tested_once():
    validate_folds();assert sorted(owner for fold in outer_folds() for owner in fold["test_subjects"])==list(range(54))


def test_chronological_balanced_support_and_disjoint_gallery():
    task=np.tile([0,1],50); data=OpenBMITrials(np.zeros((100,2,3),np.float32),task,np.zeros(100,int),np.zeros(100,int),np.arange(100));support,gallery=split_support(data)
    assert len(support.task)==20 and np.bincount(support.task).tolist()==[10,10] and not set(support.trial)&set(gallery.trial)


def test_frozen_predicted_semantics_and_support_context():
    _,z,logits,subject=fixture();sem=FrozenSemantics.fit(z,logits);owners,context=support_context(z,logits,subject,sem);assert len(owners)==6 and context.shape==(6,128)


def test_donor_is_cross_subject_and_semantically_matched():
    _,z,logits,subject=fixture();sem=FrozenSemantics.fit(z,logits);bank=DonorBank.fit(z,logits,subject,sem);_,indices,routes=bank.sample(z,logits,subject,3);assert np.all(subject[indices]!=subject) and set(routes)<={"exact_stratum","nearest_stratum","global_fallback"}


def test_donor_bank_signature_has_no_true_task_label():
    parameters=inspect.signature(DonorBank.sample).parameters;assert "task" not in parameters and "true_label" not in parameters


def test_bridge_shapes_and_k8_replay():
    one=OneStepBridge();diff=SARDBridge();condition=torch.randn(5,258);assert one(condition).shape==(5,128);noise=torch.randn(5,128);a=diff.sample(condition,noise,10);b=diff.sample(condition,noise,10);torch.testing.assert_close(a,b)


def test_dynamic_diffusion_target_and_forward_marginal():
    model=SARDBridge();x=torch.randn(4,128);t=torch.tensor([0,20,300,999]);noise=torch.randn_like(x);state=model.q_sample(x,t,noise);assert state.shape==x.shape and torch.isfinite(state).all()


def test_gaussian_is_model_only_and_stochastic():
    rng,z,logits,subject=fixture();sem=FrozenSemantics.fit(z,logits);bank=DonorBank.fit(z,logits,subject,sem);delta,_,_=bank.sample(z,logits,subject,4);scaler=BridgeScaler.fit(z,logits,delta);context=np.zeros((len(z),128),np.float32);condition=scaler.condition(z,logits,context);model=GaussianBridge.fit(condition,scaler.normalize_delta(delta),logits,sem);a=model.sample_many(condition[:5],logits[:5],8,4);b=model.sample_many(condition[:5],logits[:5],8,4);c=model.sample_many(condition[:5],logits[:5],8,5);np.testing.assert_array_equal(a,b);assert a.shape==(8,5,128) and not np.array_equal(a,c)


def test_source_adversary_is_training_only_condition_excludes_ids():
    source=(ROOT/"src/eeg_scad/privacy/sard_experiment.py").read_text();assert "scaler.condition(z, logits, context)" in source and "SourceAdversary" in source;assert "source subject ID" not in (ROOT/"reports/v38p_sard_bridge_method.md").read_text() if (ROOT/"reports/v38p_sard_bridge_method.md").exists() else True


def test_attackers_retrained_per_release_method():
    source=(ROOT/"src/eeg_scad/privacy/sard_experiment.py").read_text();assert "for method in METHODS" in source and "evaluate_representation(method" in source


def test_k8_all_retained_and_no_target_selected_sample():
    source=(ROOT/"src/eeg_scad/privacy/sard_experiment.py").read_text();config=(ROOT/"configs/sard_bridge_v38p.yaml").read_text();assert "count = 8" in source and "K: 8" in config and "target_selected" not in source


def test_outer_test_never_enters_donor_bank():
    source=(ROOT/"src/eeg_scad/privacy/sard_experiment.py").read_text();assert "DonorBank.fit(train_z, train_logits, train_subject" in source and '"outer_test_rows": 0' in source


def test_no_latency_or_waveform_method():
    source=(ROOT/"src/eeg_scad/privacy/sard_experiment.py").read_text();assert "latency" not in source.lower() and "waveform" not in source.lower()


def test_governance_contract():
    config=(ROOT/"configs/sard_bridge_v38p.yaml").read_text();assert "v36p_commit: a90cabf5ed7167e0bc6cfc01257e74592b6e7d85" in config
