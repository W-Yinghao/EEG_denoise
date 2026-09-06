"""Unit tests for V44 new code (raw-operator accessor, coherence helper)."""
from __future__ import annotations

import numpy as np
import pytest

from eeg_scad.data.eb_transfer_v43 import EBTransferRegistry
from eeg_scad.cli.run_v44 import _coherence, _natural_metrics, _rrmse
from tests.unit.test_v43 import SESSION, TASK, TRAIN, setup  # noqa: F401  (fixture)


def test_operator_blend_matches_gate(setup):  # noqa: F811
    data, fold, registry30 = setup
    eb120 = EBTransferRegistry(data, fold, registry30, 120)
    key = min(((cell.within, key) for key, cell in eb120.cells.items() if key[0] in TRAIN))[1]
    cell = eb120.cells[key]
    pop = registry30.population_transfer[(SESSION, TASK)]
    expected = pop + cell.lam * (cell.transfer - pop)
    np.testing.assert_array_equal(eb120.operator(*key, "EB"), expected)
    np.testing.assert_array_equal(eb120.operator(*key, "RAW"), cell.transfer)
    assert eb120.operator(*key, "EB").shape == (46, 2)


def test_operator_hard_gate_returns_population(setup):  # noqa: F811
    data, fold, registry30 = setup
    eb10 = EBTransferRegistry(data, fold, registry30, 10)
    key = next(iter(eb10.cells))
    pop = registry30.population_transfer[(SESSION, TASK)]
    np.testing.assert_array_equal(eb10.operator(*key, "EB"), pop)


def test_eog_model_anchor_identity_at_init():
    import torch
    from eeg_scad.models.calib_saddpm_eog_v44 import CalibSADDPMEOG, ddim_sample_eog
    from eeg_scad.models.calib_saddpm_cond_v42r import LinearX0Schedule

    torch.manual_seed(0)
    model = CalibSADDPMEOG(channels=4).eval()
    y = torch.randn(2, 4, 512)
    a0 = torch.randn(2, 4, 512)
    x_t = torch.randn(2, 4, 512)
    transfer = torch.randn(2, 4, 53)
    timestep = torch.tensor([500, 10])
    with torch.no_grad():
        prediction = model(x_t, y, a0, timestep, transfer)
    # Both residual heads are zero-initialized: prediction is exactly y - a0.
    torch.testing.assert_close(prediction, y - a0, rtol=0, atol=1e-6)
    schedule = LinearX0Schedule()
    noise = torch.randn(2, 4, 512)
    sampled = ddim_sample_eog(model, y, a0, transfer, noise, schedule, 5)
    torch.testing.assert_close(sampled, y - a0, rtol=0, atol=1e-5)


def test_eog_model_shape_check():
    import torch
    from eeg_scad.models.calib_saddpm_eog_v44 import CalibSADDPMEOG

    model = CalibSADDPMEOG(channels=4)
    y = torch.randn(1, 4, 512)
    with pytest.raises(ValueError):
        model(y, y, torch.randn(1, 4, 256), torch.tensor([0]), torch.randn(1, 4, 53))


def test_coherence_and_perfect_subtraction():
    rng = np.random.default_rng(3)
    drive = rng.normal(size=(2, 512))
    operator = rng.normal(size=(46, 2))
    artifact = operator @ drive
    clean = 0.1 * rng.normal(size=(46, 512))
    observed = clean + artifact
    assert _coherence(artifact, drive) > 0.99
    assert _coherence(clean, drive) < 0.1
    assert _rrmse(clean, observed - artifact) < 1e-12
    metrics = _natural_metrics(observed, drive, artifact)
    assert metrics["attenuation_db"] > 0
    assert metrics["coherence_reduction"] > 0.5
