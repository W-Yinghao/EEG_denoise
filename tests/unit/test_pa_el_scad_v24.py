from __future__ import annotations

import numpy as np

from eeg_scad.data.v24_coordinate_contract import (
    CoordinateCell,
    artifact_v23_committed,
    canonical_operator,
    eog_latent,
)


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

