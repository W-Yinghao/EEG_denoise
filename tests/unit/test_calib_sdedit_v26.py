from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import torch
import yaml

from eeg_scad.evaluation.aggregate_v26 import contrast, participant_first
from eeg_scad.evaluation.refinement_diagnostics import rotate_basis, rotation_fixture
from eeg_scad.models.calib_refine_det import CalibRefineDET, PopRefineDET
from eeg_scad.models.calib_sdedit import CalibSDEdit, PopSDEdit, sigma_to_timestep

ROOT = Path(__file__).resolve().parents[2]


def test_ledger_v13_loaded_and_v26_active():
    text = (ROOT / "docs/TAAS_SUBJECT_AWARE_DIFFUSION_PROJECT_LEDGER.md").read_text()
    assert "**版本：** v1.3" in text and "V26 CalibSDEdit" in text


def test_base_sha_and_k1():
    data = yaml.safe_load((ROOT / "configs/calib_sdedit_v26/data.yaml").read_text())
    evaluation = yaml.safe_load((ROOT / "configs/calib_sdedit_v26/evaluation.yaml").read_text())
    assert data["base_commit"] == "a7d9d647b69e152255b62dbca917a4b3ed082915"
    assert evaluation["K"] == 1


def test_fold_reuse_and_disjointness():
    folds = yaml.safe_load((ROOT / "configs/calib_sdedit_v26/folds.yaml").read_text())["folds"]
    assert len(folds) == 5
    for fold in folds:
        assert not set(fold["train"]) & set(fold["validation"])
        assert not set(fold["train"]) & set(fold["test"])
        assert not set(fold["validation"]) & set(fold["test"])


def test_basis_rotation_changes_latent_not_artifact():
    result = rotation_fixture()
    assert result["sensor_artifact_max_difference"] < 1e-10
    assert result["projector_distance"] < 1e-10
    assert result["latent_target_relative_difference"] > .1


def test_explicit_rotation_equivalence():
    rng = np.random.default_rng(3)
    u, _ = np.linalg.qr(rng.normal(size=(46, 8)))
    h = rng.normal(size=(8, 64)); r, _ = np.linalg.qr(rng.normal(size=(8, 8)))
    ur, hr = rotate_basis(u, h, r)
    assert np.allclose(u @ h, ur @ hr, atol=1e-10)


def _fixture():
    torch.manual_seed(1)
    return torch.randn(2, 46, 64), torch.randn(2, 46, 64), torch.randn(2, 46, 64), torch.randn(2, 128)


def test_one_step_shapes_finite_and_support_change():
    y, det, pop, context = _fixture(); model = CalibRefineDET(width=16).eval()
    a = model(y, det, pop, context); b = model(y, pop, pop, torch.zeros_like(context))
    assert a.shape == y.shape and torch.isfinite(a).all() and not torch.equal(a, b)
    assert PopRefineDET(width=16)(y, pop).shape == y.shape


def test_sdedit_sigma_zero_exact_anchor():
    y, det, pop, context = _fixture(); model = CalibSDEdit(width=16, timesteps=100).eval()
    output, trajectory = model.sample(y, det, pop, context, torch.randn_like(y), 0.0, 10)
    assert torch.equal(output, det) and len(trajectory) == 1


def test_sdedit_fixed_noise_and_call_count():
    y, det, pop, context = _fixture(); model = CalibSDEdit(width=16, timesteps=100).eval(); noise = torch.randn_like(y)
    a, ta = model.sample(y, det, pop, context, noise, .2, 5); b, tb = model.sample(y, det, pop, context, noise, .2, 5)
    assert torch.equal(a, b) and len(ta) == len(tb) <= 5 and torch.isfinite(a).all()


def test_sigma_mapping_nearest_and_monotone():
    model = CalibSDEdit(width=16, timesteps=100)
    indices = [sigma_to_timestep(model.alpha_bar, value) for value in (.05, .1, .2, .35)]
    assert indices == sorted(indices)


def test_population_models_have_no_support():
    assert PopRefineDET.uses_subject_support is False and PopSDEdit.uses_subject_support is False


def test_query_auxiliary_forbidden():
    assert CalibRefineDET.forbidden_fields == ("query_EOG", "query_operator", "query_event", "subject_ID")


def test_zero_artifact_metric_exclusions():
    config = yaml.safe_load((ROOT / "configs/calib_sdedit_v26/evaluation.yaml").read_text())
    assert set(config["zero_artifact_excluded_from"]) == {"snr_improvement", "artifact_rrmse"}


def test_diffusion_competition_is_not_retention_gate():
    config = yaml.safe_load((ROOT / "configs/calib_sdedit_v26/evaluation.yaml").read_text())
    assert config["retention_requires_diffusion_over_one_step"] is False
    assert config["primary_interpretive_priority"] == "natural_artifact_preservation_validity"
    source = inspect.getsource(__import__("eeg_scad.cli.v26", fromlist=["round_a_select"]).round_a_select)
    assert "validation_natural_teacher_rrmse" in source and "key[0] > 0" in source


def test_participant_first_and_contrast_sign():
    rows = [{"panel": "paired", "participant": "a", "method": "M", "rrmse_temporal": .2}, {"panel": "paired", "participant": "a", "method": "M", "rrmse_temporal": .4}, {"panel": "paired", "participant": "a", "method": "P", "rrmse_temporal": .5}]
    reduced = participant_first(rows, ["rrmse_temporal"])
    effect = contrast(reduced, "M", "P", "rrmse_temporal")
    assert np.isclose(effect[0]["effect"], .2)


def test_training_resume_state_contract():
    source = inspect.getsource(__import__("eeg_scad.training.train_v26", fromlist=["train_sdedit"]).train_sdedit)
    for key in ("optimizer", "scheduler", "ema", "amp_scaler", "diffusion_rng", "support_rng", "wrong_owner_rng"):
        assert key in source


def test_no_latent_diffusion_output_or_k8():
    source = (ROOT / "src/eeg_scad/models/calib_sdedit.py").read_text()
    assert "rank-8" not in source and "K8" not in source


def test_natural_boundary_declared():
    config = yaml.safe_load((ROOT / "configs/calib_sdedit_v26/evaluation.yaml").read_text())
    assert config["natural_evaluator_after_output_freeze"] is True
    assert config["query_auxiliary_inference_reads"] == 0 and config["sealed_reads"] == 0


def test_checkpoint_directory_and_file_are_distinct():
    module = __import__("eeg_scad.cli.v26", fromlist=["_model_dir", "_model_path"])
    directory = module._model_dir("calib_refine_det", 0, 20260828)
    path = module._model_path("calib_refine_det", 0, 20260828)
    assert path == directory / "best_joint.pt" and directory.suffix != ".pt"
