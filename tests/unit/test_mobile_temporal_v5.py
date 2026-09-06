from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from eeg_cgdr.data.mobile_bci import SealedParticipantAccessError, assert_development_access
from eeg_cgdr.experiments.mobile_bci_v5 import _events_100hz
from eeg_cgdr.experiments.mobile_temporal_diffusion_v5 import (
    _continuous_true_runs,
    _continuous_welch_distortion,
    _kwargs,
)
from eeg_cgdr.experiments.mobile_v5_aggregate import aggregate
from eeg_cgdr.experiments.mobile_v5_report import run as write_report
from eeg_cgdr.models.temporal_support_diffusion import TemporalSupportCorrectionDiffusion


def test_official_event_sample_indices_are_interpreted_at_100_hz() -> None:
    rows = [
        {"onset": "0", "duration": "50", "trial_type": "start"},
        {"onset": "125", "duration": "25", "trial_type": "target"},
    ]
    onsets, labels, durations = _events_100hz(rows, 100.0)
    np.testing.assert_allclose(onsets, [0.0, 1.25])
    np.testing.assert_allclose(durations, [50.0, 25.0])

    assert labels == ["start", "target"]
    assert bool(np.all(onsets <= 2.0))


def test_sealed_participant_is_rejected_before_any_signal_open() -> None:
    with pytest.raises(SealedParticipantAccessError):
        assert_development_access("sub-01", ["sub-02", "sub-03"])


def test_eog_support_is_zero_and_masked() -> None:
    support = (
        np.ones((46, 7680), dtype=np.float32),
        np.ones((27, 7680), dtype=np.float32),
        np.zeros((4, 7680), dtype=np.float32),
    )
    kwargs = _kwargs(support, torch.device("cpu"), 1.0)
    assert kwargs["support_eeg"].shape[-1] == 60 * 128
    assert torch.count_nonzero(kwargs["support_eog"]) == 0
    assert kwargs["modality_present"].tolist() == [[1.0, 1.0, 0.0]]


def test_diffusion_surface_cannot_receive_query_eog_imu_or_labels() -> None:
    fields = set(inspect.signature(TemporalSupportCorrectionDiffusion.sample).parameters)
    assert not fields.intersection(
        {"query_eog", "query_imu", "query_event_label", "participant_id", "query_outcome"}
    )


def test_v5_aggregate_and_report_entrypoints_are_importable() -> None:
    assert callable(aggregate)
    assert callable(write_report)


def test_continuous_welch_does_not_join_separated_low_motion_samples() -> None:
    rng = np.random.default_rng(3)
    observed = rng.normal(size=(3, 1024))
    output = observed.copy()
    mask = np.zeros(1024, dtype=bool)
    mask[0:300] = True
    mask[700:1024] = True
    runs = _continuous_true_runs(mask, 128)
    assert [(item.start, item.stop) for item in runs] == [(0, 300), (700, 1024)]
    assert _continuous_welch_distortion(output, observed, mask, 128.0) == pytest.approx(0.0)
