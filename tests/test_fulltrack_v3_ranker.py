import numpy as np
import pytest

from soundalike.ml.fulltrack_v3_ranker import (
    FEATURE_NAMES,
    FoldArrays,
    V3RankerError,
    _zscore_columns,
    fit_nonnegative_ranker,
    prepare_training_differences,
)
from soundalike.ml.fulltrack_v3 import CANDIDATE_POOL, QUERY_LIMIT, TRACKS_PER_FOLD


def _fold(artist_offset: int, *, held_artist: int | None = None) -> FoldArrays:
    track_ids = np.arange(TRACKS_PER_FOLD, dtype=np.int64) + 10_000 * artist_offset
    artist_ids = np.arange(TRACKS_PER_FOLD, dtype=np.int64) + artist_offset * 1_000
    if held_artist is not None:
        artist_ids[1] = held_artist
    query_positions = np.arange(QUERY_LIMIT, dtype=np.int64)
    global_orders = np.empty((QUERY_LIMIT, TRACKS_PER_FOLD - 1), dtype=np.int64)
    pools = np.empty((QUERY_LIMIT, CANDIDATE_POOL), dtype=np.int64)
    features = np.zeros(
        (QUERY_LIMIT, CANDIDATE_POOL, len(FEATURE_NAMES)), dtype=np.float32
    )
    relevance = np.zeros((QUERY_LIMIT, TRACKS_PER_FOLD), dtype=np.float32)
    shared = np.zeros((QUERY_LIMIT, CANDIDATE_POOL), dtype=np.int16)
    for query in range(QUERY_LIMIT):
        order = np.asarray(
            [position for position in range(TRACKS_PER_FOLD) if position != query],
            dtype=np.int64,
        )
        global_orders[query] = order
        pools[query] = order[:CANDIDATE_POOL]
        features[query, :, 0] = np.linspace(1.0, 0.0, CANDIDATE_POOL)
        features[query, :, 4] = np.linspace(0.0, 1.0, CANDIDATE_POOL)
        positive_position = int(pools[query, 2])
        relevance[query, positive_position] = 0.5
    arrays = FoldArrays(
        track_ids=track_ids,
        artist_ids=artist_ids,
        query_positions=query_positions,
        global_orders=global_orders,
        global_lengths=np.full(QUERY_LIMIT, TRACKS_PER_FOLD - 1, dtype=np.int64),
        pools=pools,
        features=features,
        relevance=relevance,
        shared_tags=shared,
    )
    arrays.validate()
    return arrays


def test_zscore_columns_is_finite_and_preserves_constant_column():
    values = np.asarray([[1.0, 3.0], [2.0, 3.0], [3.0, 3.0]])
    result = _zscore_columns(values)
    np.testing.assert_allclose(np.mean(result, axis=0), 0.0, atol=1e-12)
    np.testing.assert_array_equal(result[:, 1], 0.0)


def test_fold_arrays_rejects_pool_not_matching_global_prefix():
    arrays = _fold(1)
    arrays.pools[0, 0] = arrays.pools[0, 1]
    with pytest.raises(V3RankerError, match="global prefix"):
        arrays.validate()


def test_training_excludes_every_held_artist_and_is_deterministic():
    held = _fold(0)
    held_artist = int(held.artist_ids[0])
    folds = {
        0: held,
        1: _fold(1, held_artist=held_artist),
        2: _fold(2),
        3: _fold(3),
        4: _fold(4),
    }
    first = prepare_training_differences(folds, held_fold=0, seed=9)
    second = prepare_training_differences(folds, held_fold=0, seed=9)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert first[2] == second[2]
    assert first[2]["same_held_artist_training_count"] == 0
    assert first[2]["excluded_held_candidates"] > 0


def test_nonnegative_ranker_is_normalized_and_improves_loss():
    rng = np.random.default_rng(4)
    differences = rng.normal(size=(200, len(FEATURE_NAMES)))
    differences[:, 4] += 2.0
    targets = np.full(200, 0.5)
    weights, evidence = fit_nonnegative_ranker(
        differences, targets, seed=12, auxiliary_weight=0.10
    )
    assert np.all(weights >= 0.0)
    assert np.sum(weights) == pytest.approx(1.0)
    assert weights[0] == pytest.approx(0.90)
    assert weights[4] > weights[1]
    assert evidence["final_loss"] < evidence["initial_loss"]


def test_nonnegative_ranker_rejects_unsafe_auxiliary_share():
    differences = np.ones((2, len(FEATURE_NAMES)))
    targets = np.ones(2)
    with pytest.raises(V3RankerError, match="auxiliary weight"):
        fit_nonnegative_ranker(
            differences, targets, seed=1, auxiliary_weight=0.5
        )
