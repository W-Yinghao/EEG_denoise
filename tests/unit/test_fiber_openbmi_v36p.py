import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from eeg_scad.privacy.fiber import HeadFiber
from eeg_scad.privacy.fiber_channel import FiberStratifiedResampler, compose_strong_release
from eeg_scad.privacy.fiber_experiment import exact_preservation
from eeg_scad.privacy.fiber_external import FiberGaussian
from eeg_scad.privacy.models import EEGNetRepresentation
from eeg_scad.privacy.openbmi import N_CHANNELS, N_SAMPLES, load_openbmi, outer_folds, validate_folds


ROOT = Path(__file__).resolve().parents[2]
BASE = "096b43fcb902e745811c953f1049b3e63fd90726"


def fiber_fixture():
    rng = np.random.default_rng(36)
    head = torch.nn.Linear(128, 2)
    geometry = HeadFiber.from_linear(head)
    z = rng.normal(size=(80, 128)).astype(np.float32)
    z_head, u, h = geometry.decompose(z)
    return rng, geometry, z, z_head, u, h


def test_base_ledger_and_governance_contract():
    config = (ROOT / "configs" / "fiber_openbmi_v36p.yaml").read_text()
    ledger = (ROOT / "docs" / "TAAS_SUBJECT_AWARE_DIFFUSION_PROJECT_LEDGER.md").read_text()
    assert BASE in config and "v3.5" in ledger and "V36P" in ledger
    assert "waveform_sealed_reads: 0" in config and "latency_benchmark_run: false" in config


def test_six_outer_folds_cover_54_once():
    validate_folds()
    folds = outer_folds()
    assert len(folds) == 6
    assert sorted(subject for fold in folds for subject in fold["test_subjects"]) == list(range(54))


def test_fold_participant_and_session_contract():
    for fold in outer_folds():
        assert (len(fold["train_subjects"]), len(fold["validation_subjects"]), len(fold["test_subjects"])) == (36, 9, 9)
        assert fold["privacy_gallery_session"] == "ses_0" and fold["privacy_query_session"] == "ses_1"


def test_openbmi_loader_synthetic_contract(tmp_path):
    channels = [f"C{i}" for i in range(N_CHANNELS)]
    (tmp_path / "recordings").mkdir()
    array = np.random.default_rng(1).normal(size=(1000, N_CHANNELS)).astype(np.float32)
    np.save(tmp_path / "recordings" / "a.npy", array)
    pd.DataFrame([{"recording_id": 0, "subject": 0, "session": "ses_0", "run": "run_0", "n_channels": N_CHANNELS, "n_timepoints": 1000, "channels": json.dumps(channels), "filepath": "recordings/a.npy"}]).to_parquet(tmp_path / "metadata.parquet")
    onsets = np.zeros(100, dtype=int)
    pd.DataFrame({"recording_id": 0, "onset_sample": onsets, "duration_samples": np.full(100, 4.0), "event_code": np.tile([1, 2], 50)}).to_parquet(tmp_path / "events.parquet")
    data = load_openbmi(tmp_path, [0], "ses_0")
    assert data.eeg.shape == (100, N_CHANNELS, N_SAMPLES)
    np.testing.assert_allclose(data.eeg.mean(axis=-1), 0, atol=2e-5)
    assert set(data.task) == {0, 1}


def test_openbmi_eegnet_shape_and_binary_head():
    model = EEGNetRepresentation(channels=62, samples=800, task_classes=2)
    logits, z = model(torch.randn(2, 62, 800), return_representation=True)
    assert logits.shape == (2, 2) and z.shape == (2, 128)


def test_historical_eegnet_default_shape_unchanged():
    model = EEGNetRepresentation()
    assert model(torch.randn(2, 22, 512)).shape == (2, 4)


def test_binary_head_exact_fiber_geometry():
    _, geometry, z, z_head, u, _ = fiber_fixture()
    assert geometry.rank == 1 and geometry.fiber_dim == 127
    released = geometry.compose(z_head, u[::-1])
    result = exact_preservation(geometry, z, released, np.arange(len(z)) % 2)
    assert result["prediction_mismatch_count"] == 0 and result["max_softmax_probability_error"] < 1e-6


def test_gaussian_signature_has_no_source_or_subject():
    parameters = inspect.signature(FiberGaussian.sample).parameters
    assert "source_u" not in parameters and "subject" not in parameters


def test_gaussian_replay_and_stochasticity():
    _, _, _, _, u, h = fiber_fixture()
    model = FiberGaussian.fit(u[:60], h[:60])
    a, _ = model.sample(h[60:], seed=1)
    b, _ = model.sample(h[60:], seed=1)
    c, _ = model.sample(h[60:], seed=2)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_gaussian_checkpoint_contains_no_training_fiber_bank(tmp_path):
    _, _, _, _, u, h = fiber_fixture()
    model = FiberGaussian.fit(u[:60], h[:60])
    path = tmp_path / "gaussian.npz"
    model.save(path)
    with np.load(path) as payload:
        assert not any(key in {"training_fibers", "exemplars", "fiber_bank"} for key in payload.files)


def test_gaussian_exact_preservation():
    _, geometry, z, z_head, u, h = fiber_fixture()
    model = FiberGaussian.fit(u[:60], h[:60])
    replacement, _ = model.sample(h[60:], seed=3)
    released = compose_strong_release(geometry, z_head[60:], replacement)
    result = exact_preservation(geometry, z[60:], released, np.arange(20) % 2)
    assert result["prediction_mismatch_count"] == 0


def test_resample_training_bank_only_and_exact_copy():
    _, _, _, _, u, h = fiber_fixture()
    model = FiberStratifiedResampler.fit(u[:60], h[:60])
    released, coverage = model.sample(h[60:], seed=4)
    assert all(any(np.array_equal(row, donor) for donor in u[:60]) for row in released)
    assert max(item["donor_training_index"] for item in coverage) < 60


def test_sandiff_deployment_contract_is_model_only():
    config = (ROOT / "configs" / "fiber_openbmi_v36p.yaml").read_text()
    source = (ROOT / "src" / "eeg_scad" / "privacy" / "openbmi_experiment.py").read_text()
    assert "sandiff_deployment_requires_training_bank\": False" in source
    assert "deployment_training_bank: false" in config


def test_no_latency_benchmark_or_second_encoder():
    source = (ROOT / "src" / "eeg_scad" / "privacy" / "openbmi_experiment.py").read_text()
    config = (ROOT / "configs" / "fiber_openbmi_v36p.yaml").read_text()
    assert "_latency(" not in source and "latency_benchmark_run\": False" in source
    assert config.count("encoder: EEGNet") == 1


def test_head_aware_and_exposure_fields_registered():
    source = (ROOT / "src" / "eeg_scad" / "privacy" / "openbmi_experiment.py").read_text()
    for token in ("A_H", "A_HU", "training_exposure", "Fiber-Gaussian", "Fiber-Stratified-Resample"):
        assert token in source


def test_sixteen_releases_are_not_target_selected():
    config = (ROOT / "configs" / "fiber_openbmi_v36p.yaml").read_text()
    source = (ROOT / "src" / "eeg_scad" / "privacy" / "openbmi_experiment.py").read_text()
    assert "multisample_releases: 16" in config and '"target_selected_multisample": False' in source


def test_manuscript_and_sealed_boundaries_are_explicit():
    config = (ROOT / "configs" / "fiber_openbmi_v36p.yaml").read_text()
    assert "manuscript_modified: false" in config and "manuscript_compiled: false" in config
    assert "waveform_sealed_reads: 0" in config
