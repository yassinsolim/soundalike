from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from soundalike.ml import semantic_predictor
from soundalike.ml.fulltrack_store import sha256_path, stable_json_sha256
from soundalike.ml.semantic_predictor import (
    CATEGORIES,
    MODEL_KIND,
    MODEL_SCHEMA_VERSION,
    SENTINEL_TAG_INDEX,
    TAXONOMY_VERSION,
    CalibratedSemanticPredictor,
    LabelSplit,
    SemanticPredictorError,
    _model_arrays,
    _payload_sha256,
    calibration_metrics,
    calibration_partition,
    export_sparse_predictions,
    feature_domain_diagnostics,
    fit_calibrated_predictor,
    load_predictor,
    load_label_split,
    load_sparse_predictions,
    split_calibration_labels,
    tag_ranking_metrics,
)


VOCABULARY = (
    "genre---electronic",
    "genre---rock",
    "instrument---guitar",
    "instrument---synthesizer",
    "mood/theme---energetic",
    "mood/theme---melancholic",
)


def _targets(inputs: np.ndarray) -> np.ndarray:
    result = np.zeros((len(inputs), len(VOCABULARY)), dtype=np.float32)
    result[:, 0] = inputs[:, 0] > 0.0
    result[:, 1] = inputs[:, 0] <= 0.0
    result[:, 2] = inputs[:, 1] > 0.0
    result[:, 3] = inputs[:, 1] <= 0.0
    result[:, 4] = inputs[:, 2] > 0.0
    result[:, 5] = inputs[:, 2] <= 0.0
    return result


def _predictor() -> CalibratedSemanticPredictor:
    rng = np.random.default_rng(20260805)
    train = rng.normal(size=(240, 6))
    calibration = rng.normal(size=(160, 6))
    return fit_calibrated_predictor(
        train,
        _targets(train),
        calibration,
        _targets(calibration),
        VOCABULARY,
    )


def test_calibration_partition_is_deterministic_and_artist_bound():
    assert calibration_partition(42) == calibration_partition(42)
    assert {calibration_partition(value) for value in range(1, 100)} == {
        "fit",
        "audit",
    }
    with pytest.raises(SemanticPredictorError, match="positive integer"):
        calibration_partition(0)


def test_calibration_split_has_no_artist_overlap(tmp_path: Path):
    artist_ids = np.arange(1, 80, dtype=np.int64)
    split = LabelSplit(
        part="validation",
        source_path=tmp_path / "validation.tsv",
        track_ids=np.arange(100, 179, dtype=np.int64),
        artist_ids=artist_ids,
        tags=tuple((VOCABULARY[index % len(VOCABULARY)],) for index in range(79)),
    )
    fit, audit = split_calibration_labels(split)
    assert set(fit.artist_ids).isdisjoint(set(audit.artist_ids))
    assert set(fit.track_ids) | set(audit.track_ids) == set(split.track_ids)


def test_predictor_is_deterministic_calibrated_and_multilabel():
    first = _predictor()
    second = _predictor()
    np.testing.assert_allclose(first.coefficients, second.coefficients)
    np.testing.assert_allclose(
        first.calibrator_slopes, second.calibrator_slopes
    )
    values = np.asarray(
        [[2.0, 2.0, 2.0, 0.1, 0.2, 0.3], [-2.0, -2.0, -2.0, 0.1, 0.2, 0.3]]
    )
    probabilities = first.predict_proba(values)
    assert probabilities.shape == (2, len(VOCABULARY))
    assert np.all((probabilities > 0.0) & (probabilities < 1.0))
    assert probabilities[0, 0] > probabilities[0, 1]
    assert probabilities[0, 2] > probabilities[0, 3]
    assert probabilities[0, 4] > probabilities[0, 5]
    np.testing.assert_allclose(
        np.linalg.norm(first.semantic_profiles(values), axis=1), 1.0
    )


def test_rare_calibration_tag_uses_safe_constant_fallback():
    rng = np.random.default_rng(9)
    train = rng.normal(size=(80, 4))
    calibration = rng.normal(size=(30, 4))
    train_targets = np.zeros((80, 3), dtype=np.float32)
    calibration_targets = np.zeros((30, 3), dtype=np.float32)
    train_targets[:, 0] = train[:, 0] > 0
    train_targets[:, 1] = train[:, 1] > 0
    train_targets[:2, 2] = 1.0
    calibration_targets[:, 0] = calibration[:, 0] > 0
    calibration_targets[:, 1] = calibration[:, 1] > 0
    calibration_targets[:2, 2] = 1.0
    predictor = fit_calibrated_predictor(
        train,
        train_targets,
        calibration,
        calibration_targets,
        ("genre---a", "instrument---b", "mood/theme---c"),
    )
    assert predictor.calibration_supported[2] == np.bool_(False)
    rare = predictor.predict_proba(calibration[:5])[:, 2]
    np.testing.assert_allclose(rare, rare[0])


def test_sparse_predictions_apply_category_quotas_and_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    predictor = _predictor()
    monkeypatch.setattr(
        semantic_predictor,
        "TAXONOMY_COUNTS",
        {"genre": 2, "instrument": 2, "mood/theme": 2},
    )
    monkeypatch.setattr(
        semantic_predictor,
        "EXPECTED_VOCABULARY_SHA256",
        stable_json_sha256(predictor.vocabulary),
    )
    values = np.asarray(
        [[2.0, 2.0, 2.0, 0.1, 0.2, 0.3], [-2.0, -2.0, -2.0, 0.1, 0.2, 0.3]]
    )
    output = tmp_path / "sparse.npz"
    exported = export_sparse_predictions(
        predictor=predictor,
        embeddings=values,
        track_ids=np.asarray([101, 102], dtype=np.int64),
        output=output,
        category_limits={category: 1 for category in CATEGORIES},
        probability_threshold=0.0,
        batch_size=1,
    )
    assert exported.tag_indices.shape == (2, 3)
    assert exported.slot_categories == CATEGORIES
    assert np.all(exported.tag_indices != SENTINEL_TAG_INDEX)
    restored = load_sparse_predictions(output)
    np.testing.assert_array_equal(restored.track_ids, exported.track_ids)
    np.testing.assert_array_equal(restored.tag_indices, exported.tag_indices)
    np.testing.assert_array_equal(restored.probabilities, exported.probabilities)


def test_model_artifact_is_pickle_free_and_hash_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    predictor = _predictor()
    monkeypatch.setattr(
        semantic_predictor,
        "TAXONOMY_COUNTS",
        {"genre": 2, "instrument": 2, "mood/theme": 2},
    )
    monkeypatch.setattr(
        semantic_predictor,
        "EXPECTED_VOCABULARY_SHA256",
        stable_json_sha256(predictor.vocabulary),
    )
    model = tmp_path / "model.npz"
    with model.open("xb") as handle:
        np.savez_compressed(handle, **_model_arrays(predictor))
    report = tmp_path / "report.json"
    report.write_text("{}\n", encoding="utf-8")
    metadata = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "artifact_kind": MODEL_KIND,
        "taxonomy_version": TAXONOMY_VERSION,
        "taxonomy_counts": {
            "genre": 2,
            "instrument": 2,
            "mood/theme": 2,
        },
        "ridge": predictor.ridge,
        "input_dimension": len(predictor.input_mean),
        "tag_count": len(predictor.vocabulary),
        "vocabulary_sha256": stable_json_sha256(predictor.vocabulary),
        "model_npz_sha256": sha256_path(model),
        "report_file_sha256": sha256_path(report),
        "report_payload_sha256": stable_json_sha256({}),
        "test_labels_accessed": False,
        "production_ranking_changed": False,
        "promotion_allowed": False,
    }
    metadata["payload_sha256"] = _payload_sha256(metadata)
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8"
    )
    restored = load_predictor(model, metadata_path)
    np.testing.assert_allclose(restored.coefficients, predictor.coefficients, rtol=1e-6)
    with np.load(model, allow_pickle=False) as archive:
        assert archive["vocabulary"].dtype.kind == "U"
        assert archive["categories"].dtype.kind == "U"


def test_calibration_metrics_reject_probability_boundaries():
    targets = np.asarray([[0.0, 1.0]], dtype=np.float32)
    with pytest.raises(SemanticPredictorError, match="metric inputs"):
        calibration_metrics(targets, np.asarray([[0.0, 1.0]]))


def test_label_loader_rejects_nonzero_fold_before_file_access(tmp_path: Path):
    with pytest.raises(SemanticPredictorError, match="only fold 0"):
        load_label_split(tmp_path, fold=1, part="train")


def test_feature_domain_gate_rejects_shifted_inputs():
    rng = np.random.default_rng(71)
    train = rng.normal(size=(2_000, 6))
    calibration = rng.normal(size=(600, 6))
    predictor = fit_calibrated_predictor(
        train,
        _targets(train),
        calibration,
        _targets(calibration),
        VOCABULARY,
    )
    assert feature_domain_diagnostics(predictor, train)["passed"] is True
    assert feature_domain_diagnostics(predictor, train * 0.5 + 2.0)["passed"] is False


def test_tag_ranking_metrics_report_category_quotas():
    predictor = _predictor()
    values = np.asarray(
        [[2.0, 2.0, 2.0, 0.1, 0.2, 0.3], [-2.0, -2.0, -2.0, 0.1, 0.2, 0.3]]
    )
    metrics = tag_ranking_metrics(
        _targets(values),
        predictor.predict_proba(values),
        predictor.categories,
        category_limits={category: 1 for category in CATEGORIES},
    )
    assert set(metrics) == set(CATEGORIES)
    assert all(metrics[category]["hit_rate_at_quota"] == 1.0 for category in CATEGORIES)
