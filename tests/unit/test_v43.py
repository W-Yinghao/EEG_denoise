"""Unit tests for the V43 EB gated state builder (new code only).

Synthetic records and a stub registry30 keep the tests off /projects.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from eeg_scad.data.artifact_transfer_v41r import TransferRegistry
from eeg_scad.data.eb_transfer_v43 import (EBTransferRegistry, HARD_GATE_MIN_SECONDS,
                                           eb_lambda, eb_lambda_rows)


SESSION, TASK = "ses-02", "ERP"
OWNERS = ("sub-a", "sub-b", "sub-c", "sub-d")
TRAIN = ["sub-a", "sub-b", "sub-c"]
EYE_NAMES = ["HEOGL", "HEOGR", "VEOGU", "VEOGL"]


class FakeRegistry30:
    """Just the registry30 surface the EB builder consumes."""

    def __init__(self, records):
        self.records = records
        self.eeg_scale = np.ones(46)
        self.ridge_ratio = 0.05
        rng = np.random.default_rng(7)
        self.cells = {}
        for key in records:
            quality = rng.normal(size=4)
            self.cells[key] = dataclasses.make_dataclass("Cell", ["quality"])(quality)
        self.population_transfer = {(SESSION, TASK): rng.normal(size=(46, 2))}
        self.population_quality = {(SESSION, TASK): rng.normal(size=4)}
        self.continuous_center = rng.normal(size=7) * 0.1
        self.continuous_scale = np.abs(rng.normal(size=7)) + 0.5

    _continuous = staticmethod(TransferRegistry._continuous)

    def _load(self, owner, session, task):
        return self.records[(owner, session, task)]

    def signature(self, owner, session, task, condition="MATCH"):
        assert condition == "POP"
        transfer = self.population_transfer[(session, task)]
        quality = self.population_quality[(session, task)]
        continuous = (self._continuous(transfer, quality) - self.continuous_center) / self.continuous_scale
        return np.concatenate((continuous, np.eye(len(transfer))), axis=1).astype(np.float32)


@pytest.fixture(scope="module")
def setup():
    rng = np.random.default_rng(20260814)
    records = {}
    for owner in OWNERS:
        true_transfer = rng.normal(size=(46, 2))
        eye = rng.normal(size=(4, 12000))
        bipolar = np.stack((eye[2] - eye[3], eye[0] - eye[1]))
        eeg = true_transfer @ bipolar + 0.3 * rng.normal(size=(46, 12000))
        records[(owner, SESSION, TASK)] = (eeg, eye, EYE_NAMES)
    registry30 = FakeRegistry30(records)
    fold = {"fold": 0, "train": TRAIN, "validation": [], "test": ["sub-d"]}
    data = {"sampling_rate": 100, "auxiliary_support_owner": "sub-none"}
    return data, fold, registry30


def test_gate_closed_form():
    lam, gate = eb_lambda(1.0, 4.0, 120, np.inf)
    assert lam == pytest.approx(0.5) and not gate
    assert eb_lambda(0.0, 4.0, 120, np.inf) == (0.0, False)
    assert 0.0 <= eb_lambda(5.0, 0.0, 120, np.inf)[0] <= 1.0


def test_hard_gate_short_support_and_within():
    assert eb_lambda(1.0, 0.1, 10, np.inf) == (0.0, True)
    assert eb_lambda(1.0, 0.1, HARD_GATE_MIN_SECONDS - 1e-9, np.inf) == (0.0, True)
    assert eb_lambda(1.0, 0.2, 120, 0.1) == (0.0, True)
    rows = eb_lambda_rows(np.ones(46), np.ones(46), hard_gate=True)
    assert np.array_equal(rows, np.zeros(46))


def test_eb10_bypass_identity(setup):
    data, fold, registry30 = setup
    eb10 = EBTransferRegistry(data, fold, registry30, 10)
    for key in registry30.records:
        assert eb10.cells[key].hard_gate and eb10.cells[key].lam == 0.0
        signature = eb10.signature(*key, "EB")
        pop = registry30.signature(*key, "POP")
        assert np.array_equal(signature, pop)
        assert signature.shape == (46, 53) and signature.dtype == np.float32


def test_lambda_zero_emits_pop_exactly(setup):
    data, fold, registry30 = setup
    eb120 = EBTransferRegistry(data, fold, registry30, 120)
    key = ("sub-a", SESSION, TASK)
    eb120.cells[key] = dataclasses.replace(eb120.cells[key], lam=0.0)
    assert np.array_equal(eb120.signature(*key, "EB"), registry30.signature(*key, "POP"))


def test_eb120_blend_and_normalization_reuse(setup):
    data, fold, registry30 = setup
    eb120 = EBTransferRegistry(data, fold, registry30, 120)
    key = min(((cell.within, key) for key, cell in eb120.cells.items() if key[0] in TRAIN))[1]
    cell = eb120.cells[key]
    assert not cell.hard_gate and 0.0 < cell.lam <= 1.0
    pop_transfer = registry30.population_transfer[(SESSION, TASK)]
    pop_quality = registry30.population_quality[(SESSION, TASK)]
    clamped = np.clip(cell.quality, eb120.quality_min, eb120.quality_max)
    transfer = pop_transfer + cell.lam * (cell.transfer - pop_transfer)
    quality = pop_quality + cell.lam * (clamped - pop_quality)
    expected = (registry30._continuous(transfer, quality) - registry30.continuous_center) / registry30.continuous_scale
    signature = eb120.signature(*key, "EB")
    np.testing.assert_array_equal(signature[:, :7], expected.astype(np.float32))
    np.testing.assert_array_equal(signature[:, 7:], np.eye(46, dtype=np.float32))


def test_raw_variant_is_unshrunk_full_state(setup):
    data, fold, registry30 = setup
    eb120 = EBTransferRegistry(data, fold, registry30, 120)
    key = ("sub-b", SESSION, TASK)
    cell = eb120.cells[key]
    clamped = np.clip(cell.quality, eb120.quality_min, eb120.quality_max)
    expected = (registry30._continuous(cell.transfer, clamped) - registry30.continuous_center) / registry30.continuous_scale
    raw = eb120.signature(*key, "RAW")
    np.testing.assert_allclose(raw[:, :7], expected.astype(np.float32), rtol=0, atol=1e-6)


def test_perrow_secondary_and_manifest(setup):
    data, fold, registry30 = setup
    eb120 = EBTransferRegistry(data, fold, registry30, 120)
    key = ("sub-c", SESSION, TASK)
    perrow = eb120.signature(*key, "PERROW")
    assert perrow.shape == (46, 53) and perrow.dtype == np.float32
    rows = eb120.manifest_rows()
    assert len(rows) == 4
    for row in rows:
        assert set(("fold", "seconds", "participant", "lambda", "tau2", "within",
                    "within_threshold", "hard_gate")) <= set(row)
        assert 0.0 <= row["lambda"] <= 1.0
