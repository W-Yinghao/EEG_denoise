from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eeg_cgdr.data.mobile_bci import (
    SealedParticipantAccessError,
    assert_development_access,
    freeze_metadata_split,
    parse_brainvision_header,
)
from eeg_cgdr.experiments.mobile_bci_headroom_v4 import _event_arrays, _label_frequency, _support_block


def _metadata_rows() -> list[dict[str, object]]:
    rows = []
    for participant_number in range(1, 25):
        participant = f"sub-{participant_number:02d}"
        for session in ("ses-02", "ses-03", "ses-04", "ses-05"):
            for task in ("ERP", "SSVEP"):
                rows.append({
                    "participant": participant, "session": session, "task": task,
                    "header_exists": True, "data_exists": True, "marker_exists": True,
                    "channels_exists": True, "source_eog_channels": 4,
                    "processed_imu_channels": 27,
                })
    return rows


def test_metadata_split_has_one_sealed_per_three_and_four_balanced_folds() -> None:
    split = freeze_metadata_split(_metadata_rows())
    assert len(split.development) == 16
    assert len(split.sealed) == 8
    for start in range(1, 25, 3):
        group = {f"sub-{index:02d}" for index in range(start, start + 3)}
        assert len(group.intersection(split.sealed)) == 1
    assert {len(value) for value in split.folds.values()} == {4}
    assert set().union(*map(set, split.folds.values())) == set(split.development)


def test_sealed_participant_is_rejected_before_loader_open() -> None:
    with pytest.raises(SealedParticipantAccessError):
        assert_development_access("sub-03", ("sub-01", "sub-02"))


def test_brainvision_text_header_parser_does_not_open_binary(tmp_path: Path) -> None:
    path = tmp_path / "record.vhdr"
    path.write_text("DataFile=record.eeg\nMarkerFile=record.vmrk\nSamplingInterval=2000\n", encoding="latin-1")
    parsed = parse_brainvision_header(path)
    assert parsed == {"data_file": "record.eeg", "marker_file": "record.vmrk", "sampling_rate_hz": 500.0}


def test_official_mobile_event_sample_indices_and_ssvep_codes() -> None:
    onsets,labels,durations=_event_arrays([{"onset":"400","duration":"5","value":"11"}],100.0)
    assert onsets.tolist()==[4.0] and labels==["11"] and durations.tolist()==[5.0]
    assert _label_frequency("11")==5.45 and _label_frequency("12")==8.57 and _label_frequency("13")==12.0
    start,end,next_start,blocks=_support_block(
        np.asarray([4.0,5.5,7.0,14.0,15.5]),np.asarray([.5]*5),30.0
    )
    assert (start,end,next_start,blocks)==(0.0,7.5,14.0,2)
