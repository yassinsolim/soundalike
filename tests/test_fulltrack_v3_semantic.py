from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from soundalike.ml.fulltrack_store import stable_json_sha256
from soundalike.ml.fulltrack_v3_semantic import (
    BLEND_VALUES,
    MIN_DEVELOPMENT_PRIMARY_RELATIVE_GAIN,
    SemanticHead,
    V3SemanticError,
    _model_arrays,
    _normalized_inputs,
    _representation_inputs,
    development_gate,
    fit_semantic_head,
)


def test_normalized_inputs_use_train_statistics_and_unit_norm():
    train = np.asarray(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 3.0],
            [3.0, 8.0, 3.0],
        ]
    )
    transformed, mean, scale = _normalized_inputs(train)
    np.testing.assert_allclose(np.linalg.norm(transformed, axis=1), 1.0)
    np.testing.assert_allclose(mean, [2.0, 14.0 / 3.0, 3.0])
    assert scale[2] == 1.0


def test_semantic_head_fit_is_deterministic_and_predicts_unit_profiles():
    rng = np.random.default_rng(7)
    inputs = rng.normal(size=(120, 8))
    targets = np.zeros((120, 3), dtype=np.float64)
    classes = np.argmax(inputs[:, :3], axis=1)
    targets[np.arange(len(targets)), classes] = 1.0
    vocabulary = ("genre---a", "genre---b", "genre---c")
    first = fit_semantic_head(
        inputs,
        targets,
        vocabulary,
        representation="clap",
        ridge=1.0,
    )
    second = fit_semantic_head(
        inputs,
        targets,
        vocabulary,
        representation="clap",
        ridge=1.0,
    )
    np.testing.assert_allclose(first.coefficients, second.coefficients)
    profiles = first.predict(inputs)
    np.testing.assert_allclose(np.linalg.norm(profiles, axis=1), 1.0)
    assert np.mean(np.argmax(profiles, axis=1) == classes) > 0.80


def test_semantic_head_rejects_zero_norm_input():
    head = SemanticHead(
        representation="clap",
        ridge=1.0,
        vocabulary=("genre---a",),
        input_mean=np.zeros(2),
        input_scale=np.ones(2),
        coefficients=np.ones((2, 1)),
        prior=np.ones(1),
        idf=np.ones(1),
    )
    with pytest.raises(V3SemanticError, match="zero normalized norm"):
        head.predict(np.zeros((1, 2)))


def test_representation_inputs_keep_modalities_explicit():
    clap = np.ones((2, 3))
    musicfm = np.full((2, 4), 2.0)
    assert _representation_inputs("clap", clap, musicfm).shape == (2, 3)
    assert _representation_inputs("musicfm", clap, musicfm).shape == (2, 4)
    dual = _representation_inputs("dual", clap, musicfm)
    assert dual.shape == (2, 7)
    np.testing.assert_array_equal(dual[:, :3], clap)
    np.testing.assert_array_equal(dual[:, 3:], musicfm)


def _evaluation(
    *,
    recall_gain: float,
    recall_ci_low: float,
    recall_positive_folds: int,
    recall_worst_fold: float,
    mrr_gain: float = 0.01,
    ndcg_gain: float = 0.01,
):
    return {
        "relative_delta": {
            "recall_at_k": recall_gain,
            "mrr": mrr_gain,
            "graded_ndcg_at_k": ndcg_gain,
        },
        "paired_delta": {
            "recall_at_k": {
                "paired_bootstrap_ci95": [recall_ci_low, recall_ci_low + 0.01]
            }
        },
        "positive_folds": {"recall_at_k": recall_positive_folds},
        "worst_fold_relative_delta": {"recall_at_k": recall_worst_fold},
    }


def test_development_gate_requires_gain_confidence_stability_and_safety():
    passing = development_gate(
        _evaluation(
            recall_gain=MIN_DEVELOPMENT_PRIMARY_RELATIVE_GAIN,
            recall_ci_low=0.001,
            recall_positive_folds=4,
            recall_worst_fold=-0.05,
        )
    )
    assert passing["passed"] is True
    assert passing["decision"] == "freeze_for_one_time_shadow_audit"

    failing = development_gate(
        _evaluation(
            recall_gain=MIN_DEVELOPMENT_PRIMARY_RELATIVE_GAIN - 0.001,
            recall_ci_low=-0.001,
            recall_positive_folds=3,
            recall_worst_fold=-0.051,
            mrr_gain=-0.011,
        )
    )
    assert failing["passed"] is False
    assert not any(failing["checks"].values())


def test_model_arrays_are_pickle_free_and_hashable(tmp_path: Path):
    head = SemanticHead(
        representation="dual",
        ridge=10.0,
        vocabulary=("genre---a", "mood/theme---b"),
        input_mean=np.zeros(3),
        input_scale=np.ones(3),
        coefficients=np.ones((3, 2)),
        prior=np.full(2, 0.5),
        idf=np.ones(2),
    )
    arrays = _model_arrays(head)
    output = tmp_path / "model.npz"
    np.savez_compressed(output, **arrays)
    with np.load(output, allow_pickle=False) as archive:
        assert set(archive.files) == set(arrays)
        assert archive["vocabulary"].dtype.kind == "U"
    document = {
        "blend_values": list(BLEND_VALUES),
        "model_sha256": stable_json_sha256(
            {
                key: [*value.shape, str(value.dtype)]
                for key, value in arrays.items()
            }
        ),
    }
    json.loads(json.dumps(document))
