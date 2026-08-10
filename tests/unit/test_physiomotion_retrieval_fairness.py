import numpy as np

from eeg_cgdr.experiments.physiomotion_retrieval_fairness import _balanced_codes, _channel_map, _map_annotation_channel, _norm_channel, _score_candidates, _topk_codes, _topk_from_score_lookup, _uniform_seeded_starts


def test_channel_mapping_is_explicit_and_normalized():
    names = ["Fp1", "Cz", "EOG 1"]
    assert _norm_channel(" eOg 1 ") == "eog1"
    assert _channel_map(names)["eog1"] == 2
    assert _map_annotation_channel(" cz ", names) == ([1], "normalized")
    assert _map_annotation_channel("missing", names) == ([], "failed")


def test_uniform_support_sampling_spans_complete_baseline():
    starts = _uniform_seeded_starts([(10, 50), (100, 140)], 2, 16, 7)
    assert len(starts) == len(set(starts)) == 16
    assert min(starts) < 50 and max(starts) >= 100
    assert starts == _uniform_seeded_starts([(10, 50), (100, 140)], 2, 16, 7)


def test_balanced_population_budget_is_exact_and_owner_bounded():
    banks = {owner: np.zeros((16, 2, 10)) for owner in range(1, 9)}
    codes = _balanced_codes(banks, 16, np.random.default_rng(3))
    owners = np.asarray(codes) // 100
    assert len(codes) == 16
    assert max(np.sum(owners == owner) for owner in banks) <= 2


def test_observable_score_ignores_masked_truth():
    rng = np.random.default_rng(4)
    banks = {1: rng.normal(size=(8, 2, 20))}
    query = rng.normal(size=(2, 20))
    mask = np.zeros_like(query, bool); mask[:, 5:10] = True
    codes = np.asarray([100 + index for index in range(8)])
    before = _score_candidates(query, mask, codes, banks)
    query[mask] += 1e6
    after = _score_candidates(query, mask, codes, banks)
    assert np.array_equal(before, after)
    lookup = {int(code): float(score) for code, score in zip(codes, before)}
    assert np.array_equal(_topk_codes(query, mask, codes, banks, 4), _topk_from_score_lookup(codes, lookup, 4))
