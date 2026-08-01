"""Contracts for Klados development-only calibration-duration diagnostics."""

from __future__ import annotations

import numpy as np

from eeg_cgdr.data.klados import KladosRecord
from eeg_cgdr.data.mechanism import ChannelNormalizer, prepare_mechanism_record
from eeg_cgdr.experiments.klados_development_diagnostics import (
    summarize_duration_rows,
)


def _row(duration: float, record: int, *, eligible: bool, status: str = "eligible"):
    return {
        "calibration_seconds": duration,
        "source_record": record,
        "status": status,
        "p0_eligible": eligible,
        "bootstrap_median_distance": 0.1 if eligible else 0.6,
        "bootstrap_q90_distance": 0.2 if eligible else 0.8,
        "projector_max_angle_degrees": 5.0 if eligible else "",
        "matching_minus_population_e_parallel": -0.2 if eligible else "",
        "matching_minus_population_e_perp": 0.01 if eligible else "",
        "matching_minus_population_rrmse": -0.1 if eligible else "",
        "matching_minus_population_correlation": 0.05 if eligible else "",
    }


def test_duration_summary_retains_na_and_ineligible_records_in_denominator() -> None:
    rows = [
        _row(5.0, 31, eligible=True),
        _row(5.0, 32, eligible=False, status="fallback_POP"),
        _row(5.0, 33, eligible=False, status="N/A"),
    ]
    summary = summarize_duration_rows(rows)[0]

    assert summary["records_total"] == 3
    assert summary["records_available"] == 2
    assert summary["records_NA"] == 1
    assert summary["p0_eligible_records"] == 1
    assert summary["eligibility_fraction_of_available"] == 0.5
    assert summary["median_matching_minus_population_e_parallel"] == -0.2


def test_duration_summary_is_sorted_and_does_not_create_participant_counts() -> None:
    rows = [
        _row(30.0, 31, eligible=False, status="N/A"),
        _row(10.0, 31, eligible=True),
    ]
    summaries = summarize_duration_rows(rows)

    assert [row["calibration_seconds"] for row in summaries] == [10.0, 30.0]
    assert "participants" not in summaries[0]


def test_duration_axis_changes_support_but_keeps_the_exact_same_query() -> None:
    samples = 40 * 200
    time = np.arange(samples, dtype=np.float64) / 200.0
    clean = np.stack(
        [np.sin((index + 1) * 0.05 * time) for index in range(19)], axis=0
    )
    veog = np.sin(1.3 * time)
    heog = np.cos(0.7 * time)
    contaminated = clean + 0.05 * veog[None, :] + 0.03 * heog[None, :]
    record = KladosRecord(31, clean, contaminated, veog, heog)
    normalizer = ChannelNormalizer(
        mean=np.zeros(19),
        standard_deviation=np.ones(19),
        source_records=(1,),
        sample_count=samples,
    )

    short = prepare_mechanism_record(
        record,
        normalizer,
        calibration_seconds=5.0,
        guard_seconds=1.0,
        query_start_seconds=31.0,
    )
    long = prepare_mechanism_record(
        record,
        normalizer,
        calibration_seconds=30.0,
        guard_seconds=1.0,
        query_start_seconds=31.0,
    )

    assert short.calibration.eeg.shape[-1] < long.calibration.eeg.shape[-1]
    assert short.query_start_seconds == long.query_start_seconds == 31.0
    assert np.array_equal(short.observed_continuous, long.observed_continuous)
    assert np.array_equal(short.clean_continuous, long.clean_continuous)
