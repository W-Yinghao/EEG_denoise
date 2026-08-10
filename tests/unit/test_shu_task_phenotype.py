from __future__ import annotations

import numpy as np
import pytest

from eeg_cgdr.data.shu_mi import ShuMiStore, ShuTrialKey
from eeg_cgdr.experiments.shu_task_phenotype import (
    _exact_sign_flip_p,
    _masked_metrics,
    _physio_channel_to_shu_indices,
    _reconstruct,
    folds,
)


def test_folds_are_five_disjoint_groups() -> None:
    mapping = folds()
    assert sorted(mapping) == list(range(1, 26))
    assert {fold: list(mapping.values()).count(fold) for fold in range(5)} == {i: 5 for i in range(5)}


def test_session_seal_is_fail_closed() -> None:
    with pytest.raises(PermissionError):
        ShuMiStore._check_access(ShuTrialKey(1, 4, 0), final_evaluation=False)
    ShuMiStore._check_access(ShuTrialKey(1, 3, 0), final_evaluation=False)


def test_bipolar_channel_geometry_mapping() -> None:
    assert _physio_channel_to_shu_indices("Fp1-F7")
    assert _physio_channel_to_shu_indices("T7-P7")
    assert not _physio_channel_to_shu_indices("NOT-A-CHANNEL")


def test_reconstruction_preserves_unmasked_values() -> None:
    rng = np.random.default_rng(3)
    clean = rng.normal(size=(32, 1000)).astype(np.float32)
    mask = np.zeros_like(clean, bool); mask[[12, 13], 300:500] = True
    output = _reconstruct(clean, mask, np.eye(32))
    assert np.array_equal(output[~mask], clean[~mask])
    assert np.isfinite(output).all()
    assert np.isfinite(list(_masked_metrics(clean, output, mask).values())).all()


def test_exact_sign_flip_uses_effect_magnitudes() -> None:
    values = np.asarray([1.0, 2.0, 3.0])
    # Only the all-positive assignment reaches the observed mean.
    assert _exact_sign_flip_p(values) == 1 / 8
