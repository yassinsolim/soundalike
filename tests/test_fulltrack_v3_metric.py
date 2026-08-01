from __future__ import annotations

import numpy as np
import pytest

from soundalike.ml.fulltrack_v3_metric import (
    K_NEIGHBORS,
    V3MetricError,
    gated_scores,
    mine_training_triplets,
    transform_inputs,
    weighted_knn_profiles,
)


def test_transform_inputs_uses_train_statistics_and_normalizes_rows():
    train = np.asarray(
        [
            [1.0, 2.0, 4.0],
            [2.0, 4.0, 4.0],
            [4.0, 8.0, 4.0],
        ]
    )
    values = np.asarray([[3.0, 5.0, 5.0]])
    transformed_train, transformed_values, mean, scale = transform_inputs(
        train,
        values,
    )
    np.testing.assert_allclose(
        np.linalg.norm(transformed_train, axis=1),
        1.0,
    )
    np.testing.assert_allclose(
        np.linalg.norm(transformed_values, axis=1),
        1.0,
    )
    np.testing.assert_allclose(mean, np.mean(train, axis=0))
    assert scale[-1] == 1.0


def test_triplet_mining_excludes_same_artist_and_shared_tag_negatives():
    inputs = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
            [0.0, 1.0],
            [0.1, 0.9],
            [0.2, 0.8],
        ]
    )
    targets = np.asarray(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=np.float64,
    )
    artists = np.asarray([1, 1, 2, 3, 4, 5], dtype=np.int64)
    query, positive, negative, evidence = mine_training_triplets(
        inputs,
        targets,
        artists,
    )
    assert len(query) > 0
    assert np.all(artists[query] != artists[positive])
    assert np.all(artists[query] != artists[negative])
    assert np.all(np.sum(targets[query] * targets[negative], axis=1) == 0)
    assert evidence["same_artist_pairs"] == 0
    assert evidence["shared_tag_negative_pairs"] == 0


def test_weighted_knn_profiles_are_deterministic_and_normalized():
    rng = np.random.default_rng(4)
    train = rng.normal(size=(K_NEIGHBORS + 4, 6))
    values = rng.normal(size=(3, 6))
    targets = np.zeros((len(train), 5))
    targets[np.arange(len(train)), np.arange(len(train)) % 5] = 1.0
    idf = np.linspace(1.0, 2.0, 5)
    first = weighted_knn_profiles(train, targets, values, idf)
    second = weighted_knn_profiles(train, targets, values, idf)
    np.testing.assert_allclose(first[0], second[0])
    np.testing.assert_allclose(first[1], second[1])
    np.testing.assert_allclose(np.linalg.norm(first[0], axis=1), 1.0)
    assert first[2] == second[2]


def test_gated_scores_select_high_confidence_rows():
    baseline = np.asarray([[3.0, 2.0, 1.0], [3.0, 2.0, 1.0]])
    metric = np.asarray([[1.0, 3.0, 2.0], [1.0, 3.0, 2.0]])
    knn = np.asarray([[2.0, 1.0, 3.0], [2.0, 1.0, 3.0]])
    scores, applied, threshold = gated_scores(
        baseline,
        metric,
        knn,
        np.asarray([0.2, 0.8]),
        threshold=0.5,
    )
    assert threshold == 0.5
    np.testing.assert_array_equal(applied, [False, True])
    assert not np.array_equal(scores[0], scores[1])


def test_gated_scores_reject_shape_drift():
    with pytest.raises(V3MetricError, match="gated score inputs"):
        gated_scores(
            np.ones((2, 3)),
            np.ones((2, 3)),
            np.ones((2, 2)),
            np.ones(2),
        )
