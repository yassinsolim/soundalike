"""Focused tests for grouped V4 pairwise preference learning."""
from __future__ import annotations

import numpy as np

from soundalike.ml import v4_preference as preference


def test_grouped_validation_learns_signal_without_group_leakage():
    rng = np.random.default_rng(7)
    groups = np.repeat(np.arange(8), 6)
    features = rng.normal(size=(len(groups), len(preference.FEATURE_NAMES)))
    ratings = 3.0 * features[:, 1] - 2.0 * features[:, 3]
    acoustic = -features[:, 1]
    pacing = features[:, 0]
    report = preference.grouped_validation(
        features,
        ratings,
        groups,
        {"acoustic": acoustic, "pacing_v3": pacing},
    )
    assert len(report["outer_folds"]) == 8
    assert report["mean_pair_accuracy"]["learned_pair_accuracy"] > 0.9
    assert report["acceptance_rule"]["passed"] is True
    assert report["final_model"]["feature_names"] == list(
        preference.FEATURE_NAMES
    )


def test_score_features_uses_frozen_feature_order():
    artifact = {
        "validation": {
            "final_model": {
                "feature_names": list(preference.FEATURE_NAMES),
                "coefficients": [1.0] * len(preference.FEATURE_NAMES),
            }
        }
    }
    matrix = np.ones((2, len(preference.FEATURE_NAMES)))
    assert preference.score_features(matrix, artifact).tolist() == [
        float(len(preference.FEATURE_NAMES)),
        float(len(preference.FEATURE_NAMES)),
    ]
