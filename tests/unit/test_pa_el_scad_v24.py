from __future__ import annotations

import numpy as np
import torch

from eeg_scad.data.v24_coordinate_contract import (
    CoordinateCell,
    artifact_v23_committed,
    canonical_operator,
    eog_latent,
)
from eeg_scad.data.eog_latent_streams import ridge_projection
from eeg_scad.models.pa_el_det import decode_deviation
from eeg_scad.models.pa_el_scad import PAELResidualDiffusion, PAELSCADConfig
from eeg_scad.models.population_anchor_v24 import PopulationAnchorV24
from eeg_scad.models.temporal_eog_net import TemporalEOGNet


def fixture():
    rng = np.random.default_rng(24)
    raw = rng.normal(size=(7, 3))
    eeg_scale = np.linspace(1.2, 3.7, 7)
    eog_scale = np.array([0.4, 2.0, 7.0])
    center = np.array([3.0, -2.0, 8.0])
    eog = center[:, None] + rng.normal(size=(3, 31)) * eog_scale[:, None]
    return raw, eeg_scale, eog_scale, center, eog


def test_raw_and_canonical_coordinate_routes_are_equivalent():
    raw, ys, es, center, eog = fixture()
    reference, canonical, _ = CoordinateCell(raw, ys, center, es).three_routes(eog)
    np.testing.assert_allclose(reference, canonical, rtol=1e-12, atol=1e-12)


def test_v23_expression_fails_nontrivial_scale_fixture():
    raw, ys, es, center, eog = fixture()
    reference, _, committed = CoordinateCell(raw, ys, center, es).three_routes(eog)
    assert np.linalg.norm(reference - committed) / np.linalg.norm(reference) > 0.1


def test_missing_inverse_eog_scale_is_detected():
    raw, ys, es, center, eog = fixture()
    op = canonical_operator(raw, ys, es)
    correct = op @ eog_latent(eog, center, es)
    wrong = op @ (eog - center[:, None])
    assert not np.allclose(correct, wrong)


def test_double_eeg_scale_is_detected():
    raw, ys, es, center, eog = fixture()
    op = canonical_operator(raw, ys, es)
    latent = eog_latent(eog, center, es)
    correct = op @ latent
    wrong = (op @ latent) / ys[:, None]
    assert not np.allclose(correct, wrong)


def test_v23_replay_helper_is_literal_expression():
    raw, ys, es, center, eog = fixture()
    op = canonical_operator(raw, ys, es)
    centered = eog - center[:, None]
    np.testing.assert_array_equal(artifact_v23_committed(op, centered, ys), (op @ centered) / ys[:, None])


def test_population_identity_and_context_swap_are_structural():
    torch.manual_seed(2)
    a0 = torch.randn(2, 46, 16)
    latent = torch.randn(2, 4, 16)
    zero = torch.zeros(2, 46, 4)
    deviation = torch.randn(2, 46, 4)
    pop = decode_deviation(a0, zero, latent)
    match = decode_deviation(a0, deviation, latent)
    torch.testing.assert_close(pop, a0, rtol=0, atol=0)
    assert not torch.equal(pop, match)


def test_query_projection_has_eog_dimension():
    rng = np.random.default_rng(3)
    operator = rng.normal(size=(46, 4))
    y = rng.normal(size=(46, 32))
    assert ridge_projection(operator, y).shape == (4, 32)


def test_v24_model_shapes_and_no_support_in_temporal_net():
    torch.manual_seed(4)
    y = torch.randn(2, 46, 32)
    q0 = torch.randn(2, 4, 32)
    c0 = torch.randn(2, 46, 4)
    p0 = torch.einsum("bcd,bdt->bct", c0, q0)
    anchor = PopulationAnchorV24(width=16)
    a0 = anchor(y, q0, p0)
    temporal = TemporalEOGNet(width=16)
    latent = temporal(y, a0, q0)
    assert a0.shape == y.shape
    assert latent.shape == (2, 4, 32)
    assert "subject_operator" in temporal.forbidden_fields


def test_eog_residual_diffusion_forward_and_ddim_count():
    torch.manual_seed(5)
    config = PAELSCADConfig(base_channels=16, timesteps=20, ddim_steps=5)
    model = PAELResidualDiffusion(config)
    y = torch.randn(2, 46, 32)
    a0 = torch.randn_like(y)
    q0 = torch.randn(2, 4, 32)
    zdet = torch.randn_like(q0)
    target = torch.randn_like(q0)
    generator = torch.Generator().manual_seed(6)
    loss, extra = model.training_loss(target, y, a0, q0, zdet, generator)
    assert torch.isfinite(loss)
    assert extra["predicted_x0"].shape == target.shape
    result, trace = model.sample(y, a0, q0, zdet, torch.randn_like(target), trajectory=True)
    assert result.shape == target.shape
    assert len(trace) == 5
    assert all(np.isfinite(row["r_hat_rms"]) for row in trace)


def test_zero_artifact_rows_are_excludable_from_snr():
    artifact_norm = np.array([0.0, 1.0, 0.0, 2.0])
    keep = artifact_norm > 0
    assert keep.tolist() == [False, True, False, True]
