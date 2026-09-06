from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import torch
import yaml

from eeg_scad.data.artifact_transfer_v41r import (
    TransferEpisodeSampler, TransferRegistry, bipolar_eog, flatten_channels,
    flatten_signatures, reassemble_channels, ridge_transfer,
)
from eeg_scad.models.calib_eegdfus_v41r import CalibEEGDfus, OfficialLinearSchedule, ancestral_sample


ROOT = Path(__file__).resolve().parents[2]
BASE = "ade827ebc587f4edf8c4eede11a5d4472116338f"


def configs():
    data = yaml.safe_load((ROOT / "configs/setcalibdiff_v25/data.yaml").read_text())
    data.update(yaml.safe_load((ROOT / "configs/calib_eegdfus_v41r/data.yaml").read_text()))
    data["v19_derived_root"] = data["source_root"]
    folds = yaml.safe_load((ROOT / "configs/setcalibdiff_v25/folds.yaml").read_text())["folds"]
    return data, folds


def test_base_and_ledger():
    assert subprocess.check_output(["git", "merge-base", "HEAD", BASE], cwd=ROOT, text=True).strip() == BASE
    assert (ROOT / "docs/TAAS_SUBJECT_AWARE_DIFFUSION_PROJECT_LEDGER.md").read_text().splitlines()[1].startswith("## v4.3")


def test_v40_tree_and_manuscript_unchanged():
    v40 = subprocess.check_output(["git", "rev-parse", f"{BASE}:results/official_support_diffusion_v40r"], cwd=ROOT, text=True).strip()
    assert v40 == "a9d6c300d44aebee2693129675167b24842f5cb4"
    manuscript = subprocess.check_output(["git", "rev-parse", f"{BASE}:taas_submission"], cwd=ROOT, text=True).strip()
    assert manuscript == subprocess.check_output(["git", "rev-parse", "HEAD:taas_submission"], cwd=ROOT, text=True).strip()


def test_exact_two_bipolar_regressors():
    eye = np.arange(40, dtype=float).reshape(4, 10)
    value = bipolar_eog(eye, ["HEOGL", "HEOGR", "VEOGU", "VEOGL"])
    np.testing.assert_allclose(value[0], eye[2] - eye[3])
    np.testing.assert_allclose(value[1], eye[0] - eye[1])
    assert value.shape == (2, 10)


def test_bipolar_rejects_four_source_alias():
    try:
        bipolar_eog(np.ones((4, 10)), ["EOG1", "EOG2", "EOG3", "EOG4"])
    except ValueError:
        pass
    else:
        raise AssertionError("unregistered four-source alias accepted")


def test_ridge_transfer_formula_and_shape():
    rng = np.random.default_rng(1)
    eog = rng.normal(size=(2, 1000))
    truth = rng.normal(size=(46, 2))
    transfer, diagnostics = ridge_transfer(truth @ eog, eog, 1e-8)
    assert transfer.shape == (46, 2)
    assert np.linalg.norm(transfer - truth) / np.linalg.norm(truth) < 1e-6
    assert diagnostics["fit_r2"] > 0.999999


def test_support_prefix_contract_and_query_independence():
    data, folds = configs()
    r10, r30 = TransferRegistry(data, folds[0], 10), TransferRegistry(data, folds[0], 30)
    c10 = r10.cells[("sub-02", "ses-02", "ERP")]
    c30 = r30.cells[("sub-02", "ses-02", "ERP")]
    assert c10.starts == tuple(range(0, 1000 - 199, 200))
    assert c30.starts == tuple(range(0, 3000 - 199, 200))
    assert c10.support_digest != c10.query_digest
    assert c30.support_digest != c30.query_digest
    assert c10.transfer.shape == c30.transfer.shape == (46, 2)


def test_training_only_signature_normalization():
    data, folds = configs()
    registry = TransferRegistry(data, folds[0], 30)
    assert all(owner in folds[0]["train"] for owner in folds[0]["train"])
    assert set(folds[0]["train"]).isdisjoint(folds[0]["test"])
    assert registry.continuous_center.shape == (7,)
    assert registry.continuous_scale.shape == (7,)


def test_dynamic_roles_and_outer_test_absent_from_sources():
    data, folds = configs()
    registry = TransferRegistry(data, folds[0], 30)
    sampler = TransferEpisodeSampler(data, folds[0], "test", 41, registry)
    assert {row[0] for row in sampler.sources}.issubset(set(folds[0]["train"]))
    bank = sampler.sample_balanced(1)
    assert set(row["participant"] for row in bank["meta"]) == set(folds[0]["test"])
    assert all(row["query_transfer_in_model_condition"] == 0 for row in bank["meta"])


def test_wrong_and_shuffled_are_not_query_owner():
    data, folds = configs()
    registry = TransferRegistry(data, folds[0], 30)
    sampler = TransferEpisodeSampler(data, folds[0], "test", 41, registry)
    meta = {"participant": "sub-02", "session": "ses-02", "task": "ERP"}
    for condition in ("WRONG", "SHUFFLED"):
        _, owner = sampler.condition_signature(meta, condition)
        assert owner != "sub-02"


def test_zero_seconds_is_exact_population_signature():
    data, folds = configs()
    registry = TransferRegistry(data, folds[0], 30)
    sampler = TransferEpisodeSampler(data, folds[0], "test", 41, registry)
    meta = {"participant": "sub-02", "session": "ses-02", "task": "ERP"}
    a, _ = sampler.condition_signature(meta, "POP")
    b = registry.signature("sub-02", "ses-02", "ERP", "POP")
    np.testing.assert_array_equal(a, b)


def test_oracle_differs_and_is_query_transfer():
    data, folds = configs()
    registry = TransferRegistry(data, folds[0], 30)
    match = registry.signature("sub-02", "ses-02", "ERP", "MATCH")
    oracle = registry.signature("sub-02", "ses-02", "ERP", "ORACLE")
    assert not np.array_equal(match, oracle)


def test_channel_flatten_reassembly():
    value = np.arange(2 * 46 * 512).reshape(2, 46, 512)
    flat = flatten_channels(value)
    assert flat.shape == (92, 1, 512)
    np.testing.assert_array_equal(reassemble_channels(flat, 2), value)
    assert flatten_signatures(np.zeros((2, 46, 53), np.float32)).shape == (92, 53)


def test_official_model_shape_and_zero_initial_transfer():
    torch.manual_seed(1)
    model = CalibEEGDfus()
    noisy = torch.randn(2, 1, 512)
    observed = torch.randn_like(noisy)
    a, b = torch.randn(2, 53), torch.randn(2, 53)
    out_a = model(noisy, observed, torch.full((2, 1), 0.5), a)
    out_b = model(noisy, observed, torch.full((2, 1), 0.5), b)
    assert out_a.shape == noisy.shape
    torch.testing.assert_close(out_a, out_b, rtol=0, atol=0)


def test_official_schedule():
    schedule = OfficialLinearSchedule()
    assert len(schedule.beta) == 500
    assert abs(float(schedule.beta[0]) - 1e-4) < 1e-8
    assert abs(float(schedule.beta[-1]) - 0.02) < 1e-7


def test_fixed_noise_replay():
    model = CalibEEGDfus()
    schedule = OfficialLinearSchedule(steps=2)
    observed = torch.randn(1, 1, 512)
    transfer = torch.randn(1, 53)
    a = ancestral_sample(model, observed, transfer, 99, schedule)
    b = ancestral_sample(model, observed, transfer, 99, schedule)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_channel_only_keeps_identity_and_removes_transfer_values():
    data, folds = configs()
    registry = TransferRegistry(data, folds[0], 30)
    sampler = TransferEpisodeSampler(data, folds[0], "test", 41, registry)
    value, _ = sampler.condition_signature({"participant": "sub-02", "session": "ses-02", "task": "ERP"}, "CHANNEL_ONLY")
    assert np.count_nonzero(value[:, :7]) == 0
    np.testing.assert_array_equal(value[:, 7:], np.eye(46))


def test_transfer_manifest_governance_fields():
    data, folds = configs()
    row = TransferRegistry(data, folds[0], 30).manifest_rows()[0]
    assert row["eog_regressors"] == 2
    assert row["query_transfer_in_support_estimate"] == 0
    assert row["query_eog_inference_reads"] == 0
    assert row["overlap_samples"] == row["repeated_samples"] == 0


def test_sealed_policy_is_zero():
    data, _ = configs()
    assert data["sealed_reads"] == 0
    assert set(data["sealed_participants"]).isdisjoint(data["participants"])


def test_official_checkout_binding_if_available():
    checkout = Path("/home/infres/yinwang/v40r_third_party/EEGDfus")
    if checkout.is_dir():
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
        assert sha == "a19a652b3b6346188ae77067e1daf8b90cad005f"
        assert not subprocess.check_output(["git", "status", "--short"], cwd=checkout, text=True).strip()
