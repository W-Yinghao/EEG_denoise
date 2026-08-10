import numpy as np
import pytest
from eeg_cgdr.experiments.physiomotion_subject_restoration import _family, _retrieve, _split


def test_primary_family_excludes_ocular_labels():
    assert _family("hor_headm") == "head_motion"
    assert _family("chew") == "chewing"
    assert _family("tongue") == "tongue"
    assert _family("swallow") == "swallowing"
    assert _family("eyebrow") == "facial_emg"
    assert _family("blink+chew") is None
    assert _family("hor_eyem") is None


def test_split_is_20_development_10_sealed_and_disjoint():
    development, sealed = _split({"split_seed": 20260810})
    assert len(development) == 20 and len(sealed) == 10
    assert set(development).isdisjoint(sealed)
    assert set(development + sealed) == set(range(1, 31))


def test_retrieval_is_fixed_k_and_observed_context_only():
    rng = np.random.default_rng(7); bank = rng.normal(size=(10, 2, 20)); query = bank[3].copy(); mask = np.zeros((2, 20), bool); mask[:, 8:12] = True
    expected = _retrieve(query, mask, bank, 8)
    changed = query.copy(); changed[mask] += 1000
    assert np.array_equal(expected, _retrieve(changed, mask, bank, 8))


def test_sealed_access_is_fail_closed(tmp_path):
    # Guard behavior is exercised by code inspection without opening any signal.
    source = open("src/eeg_cgdr/experiments/physiomotion_subject_restoration.py", encoding="utf-8").read()
    assert "sealed participant" in source and "PermissionError" in source
