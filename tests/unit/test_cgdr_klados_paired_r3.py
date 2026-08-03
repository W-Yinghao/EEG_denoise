from __future__ import annotations

import numpy as np

from eeg_cgdr.experiments.subject_artifact_klados_paired import _origins
from eeg_cgdr.experiments.subject_artifact_next_round import (
    _paired_masked_values,
    _stratified_bootstrap,
)


def test_klados_origins_remain_source_records_not_participants() -> None:
    values = _origins(31, 3)
    assert [value.recording_key for value in values] == ["sim31"] * 3
    assert [value.trial_ordinal for value in values] == [0, 1, 2]


def test_paired_masked_values_exclude_padding() -> None:
    values = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    mask = np.asarray([[True, True, False, False], [True, False, False, False]])
    selected = _paired_masked_values(values, mask)
    expected = values.transpose(1, 0, 2)[:, mask].reshape(-1)
    np.testing.assert_array_equal(selected, expected)
    assert selected.size == 9


def test_source_record_bootstrap_does_not_expand_seeds_or_windows() -> None:
    estimate, lower, upper = _stratified_bootstrap(
        [("klados", 1.0), ("klados", 2.0), ("klados", 3.0)],
        replicates=500,
        seed=17,
    )
    assert estimate == 2.0
    assert lower > 0.0
    assert upper >= lower
