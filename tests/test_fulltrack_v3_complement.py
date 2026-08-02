from __future__ import annotations

import numpy as np
import pytest

from soundalike.ml.fulltrack_v3_complement import (
    V3ComplementError,
    _payload_sha256,
    _shadow_gate,
    complementary_profiles,
    complementary_scores,
)
from soundalike.ml.fulltrack_store import stable_json_sha256


class _ScoreData:
    def __init__(self) -> None:
        self.track_ids = np.asarray([11, 12, 13], dtype=np.int64)
        self.pools = np.asarray([[1, 2], [0, 2], [0, 1]], dtype=np.int32)
        self.baseline_scores = np.asarray(
            [[3.0, 1.0], [2.0, 1.0], [1.0, 4.0]],
            dtype=np.float64,
        )

    def validate(self) -> None:
        return None


def test_complementary_profiles_preserve_equal_channel_similarity():
    knn = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    musicfm = np.asarray([[0.6, 0.8], [0.8, 0.6]])
    combined = complementary_profiles(knn, musicfm)
    np.testing.assert_allclose(np.linalg.norm(combined, axis=1), 1.0)
    np.testing.assert_allclose(
        combined @ combined.T,
        0.5 * (knn @ knn.T) + 0.5 * (musicfm @ musicfm.T),
    )


def test_complementary_profiles_reject_unnormalized_channels():
    with pytest.raises(V3ComplementError, match="profile inputs"):
        complementary_profiles(np.ones((2, 2)), np.eye(2))


def test_complementary_scores_apply_fixed_half_residual():
    data = _ScoreData()
    profiles = np.asarray(
        [
            [1.0, 0.0],
            [0.8, 0.6],
            [0.0, 1.0],
        ]
    )
    scores = complementary_scores(data, profiles)
    assert scores.shape == data.baseline_scores.shape
    np.testing.assert_allclose(scores[0], [1.0, -1.0])


def _evaluation(*, recall_gain: float, ci_low: float = 0.001):
    return {
        "relative_delta": {
            "recall_at_k": recall_gain,
            "mrr": 0.01,
            "graded_ndcg_at_k": 0.01,
        },
        "paired_delta": {
            "recall_at_k": {
                "paired_bootstrap_ci95": [ci_low, ci_low + 0.01],
            }
        },
        "positive_folds": {"recall_at_k": 4},
        "worst_fold_relative_delta": {"recall_at_k": -0.05},
    }


def test_shadow_gate_requires_twenty_percent_and_stability():
    passing = _shadow_gate(_evaluation(recall_gain=0.20))
    assert passing["automated_passed"] is True
    assert passing["promotion_allowed"] is False
    failing = _shadow_gate(_evaluation(recall_gain=0.199, ci_low=-0.001))
    assert failing["automated_passed"] is False
    assert failing["checks"]["primary_relative_gain"] is False
    assert failing["checks"]["primary_paired_ci_above_zero"] is False


def test_payload_hash_excludes_its_checksum_field():
    document = {"artifact_kind": "test", "value": 7}
    checksum = stable_json_sha256(document)
    document["payload_sha256"] = checksum
    assert _payload_sha256(document) == checksum
