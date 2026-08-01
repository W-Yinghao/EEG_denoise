"""Pure array checks for the frozen repaired mechanism data path."""

from __future__ import annotations

import csv

import numpy as np

from eeg_cgdr.data.mechanism import (
    KLADOS_DEVELOPMENT_RECORDS,
    KLADOS_TRAIN_RECORDS,
    KLADOS_UNTOUCHED_RECORDS,
    assert_frozen_source_partition,
    standardize_reference_from_support,
    window_after_normalization,
    write_mechanism_split_manifest,
)


def test_frozen_source_record_partition_is_disjoint_and_complete() -> None:
    assert_frozen_source_partition()
    groups = [
        set(KLADOS_TRAIN_RECORDS),
        set(KLADOS_DEVELOPMENT_RECORDS),
        set(KLADOS_UNTOUCHED_RECORDS),
    ]
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]
    assert set.union(*groups) == set(range(1, 55))


def test_windowing_pads_after_normalization_with_true_zero() -> None:
    normalized = np.arange(2 * 13, dtype=np.float64).reshape(2, 13) / 7.0 - 1.0
    windowed = window_after_normalization(normalized, 8)
    assert windowed.values.shape == (2, 2, 8)
    assert windowed.valid_time_weight.shape == (2, 8)
    np.testing.assert_array_equal(windowed.valid_time_weight[0], np.ones(8))
    np.testing.assert_array_equal(
        windowed.valid_time_weight[1], np.asarray([1, 1, 1, 1, 1, 0, 0, 0])
    )
    np.testing.assert_array_equal(windowed.values[1, :, 5:], 0.0)
    np.testing.assert_array_equal(windowed.values[1, :, :5], normalized[:, 8:])


def test_external_reference_uses_support_statistics_without_query_leakage() -> None:
    support = np.asarray([[1.0, 2.0, 4.0, 5.0], [-3.0, -1.0, 2.0, 6.0]])
    query = np.asarray([[100.0, 200.0], [-40.0, 80.0]])
    standardized_support, standardized_query, mean, scale = (
        standardize_reference_from_support(support, query)
    )
    np.testing.assert_allclose(standardized_support.mean(axis=1), 0.0, atol=1e-14)
    np.testing.assert_allclose(standardized_support.std(axis=1), 1.0, atol=1e-14)
    np.testing.assert_allclose(standardized_query, (query - mean) / scale)
    changed_query = query * 1000.0
    second_support, _, second_mean, second_scale = standardize_reference_from_support(
        support, changed_query
    )
    np.testing.assert_array_equal(second_support, standardized_support)
    np.testing.assert_array_equal(second_mean, mean)
    np.testing.assert_array_equal(second_scale, scale)


def test_mechanism_split_manifest_uses_requested_minimal_schema(tmp_path) -> None:
    destination = tmp_path / "split.csv"
    write_mechanism_split_manifest(destination)
    expected_fields = [
        "dataset_version",
        "outer_fold",
        "split",
        "participant",
        "session",
        "record",
        "calibration_start",
        "calibration_end",
        "query_start",
        "query_end",
        "sampling_rate",
        "status",
    ]
    with destination.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    assert reader.fieldnames == expected_fields
    assert len(rows) == 54
    assert {row["record"] for row in rows} == {
        f"sim{record_id:02d}" for record_id in range(1, 55)
    }
    assert {row["participant"] for row in rows} == {"unresolved_not_claimed"}
    assert {row["sampling_rate"] for row in rows} == {"200"}
    assert {row["split"] for row in rows} == {"train", "development", "untouched"}
    training = [row for row in rows if row["split"] == "train"]
    held_out = [row for row in rows if row["split"] != "train"]
    assert all(
        row[field] == "N/A"
        for row in training
        for field in (
            "calibration_start",
            "calibration_end",
            "query_start",
            "query_end",
        )
    )
    assert all(
        (
            row["calibration_start"],
            row["calibration_end"],
            row["query_start"],
            row["query_end"],
        )
        == ("0", "30", "31", "record_end")
        for row in held_out
    )
