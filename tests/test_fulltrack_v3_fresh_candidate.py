from __future__ import annotations

import numpy as np
import pytest

from soundalike.ml.fulltrack_v3_fresh_candidate import (
    V3FreshCandidateError,
    _normalize_rows,
    candidate_scores,
    shadow_gate,
)


class _Data:
    def __init__(self) -> None:
        self.track_ids = np.asarray([1, 2, 3], dtype=np.int64)
        self.pools = np.asarray([[1, 2], [0, 2], [0, 1]], dtype=np.int32)
        self.baseline_scores = np.asarray(
            [[3.0, 1.0], [2.0, 1.0], [1.0, 4.0]],
            dtype=np.float64,
        )

    def validate(self) -> None:
        return None


def test_normalize_rows_rejects_zero_profile():
    with pytest.raises(V3FreshCandidateError, match="profile values"):
        _normalize_rows(np.asarray([[0.0, 0.0]]))


def test_candidate_scores_apply_seventy_percent_semantic_residual():
    data = _Data()
    profiles = np.asarray(
        [
            [1.0, 0.0],
            [0.8, 0.6],
            [0.0, 1.0],
        ]
    )
    scores = candidate_scores(data, profiles)
    np.testing.assert_allclose(scores[0], [1.0, -1.0])


def _evaluation(*, recall: float, ci_low: float = 0.001):
    return {
        "relative_delta": {
            "recall_at_k": recall,
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
    passing = shadow_gate(_evaluation(recall=0.20))
    assert passing["automated_passed"] is True
    assert passing["promotion_allowed"] is False
    failing = shadow_gate(_evaluation(recall=0.199, ci_low=-0.001))
    assert failing["automated_passed"] is False
    assert failing["checks"]["primary_relative_gain"] is False
    assert failing["checks"]["primary_paired_ci_above_zero"] is False
