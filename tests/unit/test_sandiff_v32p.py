from pathlib import Path

import numpy as np
import torch

from eeg_scad.privacy.bci2a import load_bci2a_session
from eeg_scad.privacy.checkpoint import load_resume_state, save_resume_state
from eeg_scad.privacy.bci2a import outer_folds
from eeg_scad.privacy.leace import LEACE
from eeg_scad.privacy.models import EEGNetRepresentation, LatentDANN, OneStepSanitizer
from eeg_scad.privacy.sandiff import SANDiff, cosine_alpha_bar


ROOT = Path(__file__).resolve().parents[2]
BASE = "274b371ed2d3c7c105f2351f4dd88d4464fe3a66"


def test_outer_folds_are_participant_disjoint():
    for fold in outer_folds():
        train, validation, test = map(set, (fold["train_subjects"], fold["validation_subjects"], fold["test_subjects"]))
        assert not train & validation and not train & test and not validation & test
        assert sorted(train | validation | test) == list(range(1, 10))


def test_every_participant_is_test_once():
    test = [s for fold in outer_folds() for s in fold["test_subjects"]]
    assert sorted(test) == list(range(1, 10))


def test_sessions_are_separated_for_attack():
    for fold in outer_folds():
        assert fold["adaptive_attack_train_session"] == "T"
        assert fold["adaptive_attack_test_session"] == "E"


def test_leace_removes_linear_cross_covariance():
    rng = np.random.default_rng(4)
    subject = np.repeat(np.arange(4), 80)
    z = rng.normal(size=(320, 12)) + np.eye(4, 12)[subject] * 3
    leace = LEACE.fit(z, subject)
    keep = leace.transform(z)
    onehot = np.eye(4)[subject] - np.eye(4)[subject].mean(0)
    cross = (keep - keep.mean(0)).T @ onehot / (len(keep) - 1)
    assert np.max(np.abs(cross)) < 2e-4


def test_private_plus_keep_round_trip():
    rng = np.random.default_rng(7); z = rng.normal(size=(40, 8)); subject = np.repeat(np.arange(4), 10)
    leace = LEACE.fit(z, subject); keep = leace.transform(z)
    np.testing.assert_allclose(keep + leace.private(z), z, atol=1e-6)


def test_eegnet_representation_and_logits_shape():
    model = EEGNetRepresentation(); logits, z = model(torch.randn(3, 22, 512), return_representation=True)
    assert logits.shape == (3, 4) and z.shape == (3, 128)


def test_dann_preserves_shape():
    model = LatentDANN(128, 6); z, subject = model(torch.randn(5, 128), 0.2)
    assert z.shape == (5, 128) and subject.shape == (5, 6)


def test_one_step_is_support_free_and_subject_free():
    model = OneStepSanitizer(); out = model(torch.randn(4, 128), torch.randn(4, 4))
    assert out.shape == (4, 128)


def test_cosine_schedule_is_finite_and_decreasing():
    alpha = cosine_alpha_bar(1000)
    assert alpha.shape == (1000,) and torch.isfinite(alpha).all() and torch.all(alpha[1:] < alpha[:-1])


def test_diffusion_forward_marginal_shape():
    model = SANDiff(); x = torch.randn(7, 128); t = torch.arange(7) * 100; noise = torch.randn_like(x)
    assert model.q_sample(x, t, noise).shape == x.shape


def test_diffusion_x0_prediction_shape():
    model = SANDiff(); out = model(torch.randn(3, 128), torch.randn(3, 128), torch.randn(3, 4), torch.tensor([0, 50, 999]))
    assert out.shape == (3, 128) and torch.isfinite(out).all()


def test_ddim_k1_ten_step_replay():
    model = SANDiff().eval(); keep = torch.randn(2, 128); logits = torch.randn(2, 4); noise = torch.randn(2, 128)
    first = model.sample(keep, logits, reverse_steps=10, noise=noise); second = model.sample(keep, logits, reverse_steps=10, noise=noise)
    torch.testing.assert_close(first, second)


def test_config_has_exact_base_and_two_seeds():
    text = (ROOT / "configs" / "sandiff_v32p.yaml").read_text()
    assert BASE in text and "20260920" in text and "20260921" in text


def test_config_forbids_waveform_training_and_sealed_reads():
    text = (ROOT / "configs" / "sandiff_v32p.yaml").read_text()
    assert "waveform_sealed_reads: 0" in text and "waveform_denoiser_training: false" in text


def test_ledger_is_v27_at_start():
    text = (ROOT / "docs" / "TAAS_SUBJECT_AWARE_DIFFUSION_PROJECT_LEDGER.md").read_text()
    assert ("v2.7" in text or "v2.8" in text) and "V32P" in text


def test_official_two_session_trial_contract():
    for session in ("T", "E"):
        trials = load_bci2a_session(Path("/projects/EEG-foundation-model/BCI-IV"), 1, session)
        assert trials.eeg.shape == (288, 22, 512)
        assert sorted(np.unique(trials.task).tolist()) == [0, 1, 2, 3]
        assert np.isfinite(trials.eeg).all()


def test_checkpoint_resume_restores_optimizer_and_rng(tmp_path):
    import random
    random.seed(81); np.random.seed(82); torch.manual_seed(83)
    model=torch.nn.Linear(4,3);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    loss=model(torch.randn(5,4)).sum();loss.backward();optimizer.step()
    path=tmp_path/"resume.pt";save_resume_state(path,model=model,optimizer=optimizer,epoch=7,global_step=123,metadata={"fold":2})
    expected=(random.random(),np.random.rand(),torch.rand(1))
    restored=torch.nn.Linear(4,3);restored_optimizer=torch.optim.AdamW(restored.parameters(),lr=9e-2)
    state=load_resume_state(path,model=restored,optimizer=restored_optimizer)
    actual=(random.random(),np.random.rand(),torch.rand(1))
    assert state=={"epoch":7,"global_step":123,"metadata":{"fold":2}}
    assert expected[0]==actual[0] and expected[1]==actual[1]
    torch.testing.assert_close(expected[2],actual[2])
    for left,right in zip(model.parameters(),restored.parameters()):torch.testing.assert_close(left,right)
