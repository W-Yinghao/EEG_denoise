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
