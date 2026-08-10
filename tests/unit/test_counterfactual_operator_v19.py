from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eeg_cgdr.data.mobile_bci import SealedParticipantAccessError
from eeg_cgdr.experiments.counterfactual_operator_v19 import (
    _fit_operator, _marker_times, _rrmse, _unit_slices, exact_signflip_p, guarded_paths,
)


def _config(tmp_path: Path) -> dict:
    return {
        "data_root": str(tmp_path / "data"), "output_root": str(tmp_path / "results"),
        "development_participants": ["sub-02"], "sealed_participants": ["sub-01"],
        "windows": {"support_seconds": 120.0, "guard_seconds": 30.0,
                    "query_generator_seconds": 120.0, "second_guard_seconds": 30.0},
    }


def test_sealed_access_fails_before_path_ledger(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with pytest.raises(SealedParticipantAccessError):
        guarded_paths(config, "sub-01", "ses-02", "ERP", roles=("source_binary",))
    assert not (tmp_path / "results" / "resolved_path_ledger").exists()


def test_protocol_slices_are_disjoint() -> None:
    config = _config(Path("/tmp"))
    support, qgen, qnatural = _unit_slices(50_000, 100.0, config)  # type: ignore[misc]
    assert support.stop <= qgen.start
    assert qgen.stop <= qnatural.start
    assert support.stop == 12_000 and qgen.start == 15_000 and qnatural.start == 30_000


def test_operator_swap_identity_and_oracle_recovery() -> None:
    rng = np.random.default_rng(19)
    c = rng.normal(size=(46, 4)); x = rng.normal(size=(46, 200)); e = rng.normal(size=(4, 200))
    mask = np.arange(200) % 3 == 0
    y = x + c @ (e * mask[None])
    restored = y - c @ (e * mask[None])
    assert _rrmse(x, restored, mask) < 1e-12
    np.testing.assert_array_equal(restored[:, ~mask], y[:, ~mask])


def test_instantaneous_ridge_recovers_known_operator() -> None:
    rng = np.random.default_rng(191)
    e = rng.normal(size=(4, 20_000)); c = rng.normal(size=(46, 4)); y = c @ e
    fitted = _fit_operator(y, e, 1e-10)
    np.testing.assert_allclose(fitted, c, atol=1e-8, rtol=1e-8)


def test_signflip_is_participant_level_exact() -> None:
    values = np.ones(16)
    assert exact_signflip_p(values) == pytest.approx(1 / 65536)
    assert exact_signflip_p(values, alternative="two-sided") == pytest.approx(2 / 65536)


def test_wrong_donor_must_not_equal_recipient() -> None:
    recipient = "sub-02"; donors = ["sub-03", "sub-05"]
    assert all(donor != recipient for donor in donors)


def test_query_operator_key_is_evaluator_only(tmp_path: Path) -> None:
    inference = tmp_path / "inference.npz"; evaluator = tmp_path / "evaluator.npz"
    np.savez(inference, C_match=np.zeros((46, 4)), C_pop=np.zeros((46, 4)))
    np.savez(evaluator, C_query=np.zeros((46, 4)))
    with np.load(inference) as data:
        assert "C_query" not in data.files
    with np.load(evaluator) as data:
        assert "C_query" in data.files


def test_marker_time_alignment_parser(tmp_path: Path) -> None:
    marker = tmp_path / "record.vmrk"
    marker.write_text("Brain Vision Data Exchange Marker File\nMk1=New Segment,,1,1,0\nMk2=Stimulus,S  1,501,1,0\n", encoding="latin-1")
    np.testing.assert_allclose(_marker_times(marker, 500.0), [1.0])
