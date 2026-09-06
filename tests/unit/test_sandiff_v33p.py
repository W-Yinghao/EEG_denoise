from pathlib import Path

import numpy as np
import torch

from eeg_scad.privacy.bci2a import outer_folds
from eeg_scad.privacy.consolidation import _privacy_balance, train_sanitizer_exact
from eeg_scad.privacy.leace import LEACE
from eeg_scad.privacy.models import EEGNetRepresentation
from eeg_scad.privacy.sandiff import SANDiff


ROOT=Path(__file__).resolve().parents[2]
BASE="2b1522e79a5b701389b1446f51589a9862fb5f15"


def test_v33p_outer_stage_contract():
    for split in outer_folds():
        train=set(split["train_subjects"]);validation=set(split["validation_subjects"]);test=set(split["test_subjects"]);full=train|validation
        assert len(train)==len(validation)==len(test)==3
        assert not train&validation and not full&test
        assert len(full)==6 and full|test==set(range(1,10))


def test_every_participant_is_outer_test_once():
    assert sorted(s for split in outer_folds() for s in split["test_subjects"])==list(range(1,10))


def test_privacy_balance_rewards_task_and_lower_attack():
    base={"fixed_head_balanced_accuracy":.40,"retrained_head_balanced_accuracy":.40,"adaptive_subject_attack_balanced_accuracy":.70,"cross_session_same_different_auroc":.60}
    better_task=dict(base,fixed_head_balanced_accuracy=.45)
    better_privacy=dict(base,adaptive_subject_attack_balanced_accuracy=.60)
    assert _privacy_balance(better_task)>_privacy_balance(base)
    assert _privacy_balance(better_privacy)>_privacy_balance(base)


def test_full_sampler_is_ten_step_k1():
    config=(ROOT/"configs"/"sandiff_v33p.yaml").read_text()
    assert "reverse_steps: 10" in config and "K: 1" in config
    model=SANDiff().eval();keep=torch.randn(2,128);logits=torch.randn(2,4);noise=torch.randn(2,128)
    first=model.sample(keep,logits,reverse_steps=10,noise=noise);second=model.sample(keep,logits,reverse_steps=10,noise=noise)
    torch.testing.assert_close(first,second)


def test_stage_b_exact_epoch_snapshots(tmp_path):
    rng=np.random.default_rng(33);subject=np.repeat(np.arange(3),8);task=np.tile(np.arange(4),6);z=rng.normal(size=(24,128)).astype(np.float32);logits=rng.normal(size=(24,4)).astype(np.float32);leace=LEACE.fit(z,subject);head=torch.nn.Linear(128,4)
    paths=train_sanitizer_exact("one_step",z,logits,task,subject,leace,head,torch.device("cpu"),7,{1,2},tmp_path)
    assert set(paths)=={1,2}
    assert torch.load(paths[1],weights_only=True)["epochs"]==1
    assert torch.load(paths[2],weights_only=True)["epochs"]==2


def test_full_pool_leace_can_use_five_subject_directions():
    rng=np.random.default_rng(8);subject=np.repeat(np.arange(6),40);z=rng.normal(size=(240,128));z[:,:6]+=np.eye(6)[subject]*2
    leace=LEACE.fit(z,subject)
    assert leace.rank==5


def test_primary_methods_and_strengths_are_frozen():
    text=(ROOT/"configs"/"sandiff_v33p.yaml").read_text()
    for method in ("RAW","LEACE","DANN","one_step","SANDiff"):assert method in text
    assert "primary: strong" in text and "weak, medium, strong" in text


def test_no_new_family_or_dataset():
    text=(ROOT/"configs"/"sandiff_v33p.yaml").read_text()
    assert "new_dataset: false" in text and "new_encoder: false" in text and "new_diffusion_family: false" in text


def test_ledger_v29_is_loaded():
    text=(ROOT/"docs"/"TAAS_SUBJECT_AWARE_DIFFUSION_PROJECT_LEDGER.md").read_text()
    assert "v2.9" in text and "V33P" in text


def test_exact_base_and_governance():
    text=(ROOT/"configs"/"sandiff_v33p.yaml").read_text()
    assert BASE in text and "waveform_sealed_reads: 0" in text and "manuscript_compiled: false" in text


def test_same_eegnet_representation_contract():
    model=EEGNetRepresentation();logits,z=model(torch.randn(2,22,512),return_representation=True)
    assert z.shape==(2,128) and logits.shape==(2,4)


def test_full_and_single_checkpoint_names_are_separate():
    source=(ROOT/"src"/"eeg_scad"/"privacy"/"consolidation.py").read_text()
    assert 'f"{kind}_single.pt"' in source and 'f"{kind}_full.pt"' in source
    assert '"full_10_step"' in source
