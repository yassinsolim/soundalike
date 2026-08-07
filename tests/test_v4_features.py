"""Focused tests for V4 semantic feature cache validation."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from soundalike.ml import v4_features as features


def write_cache(tmp_path):
    cache = tmp_path / "features.npz"
    metadata = tmp_path / "features.json"
    with cache.open("wb") as handle:
        np.savez_compressed(
            handle,
            track_ids=np.asarray([1, 2], dtype=np.int64),
            probabilities=np.ones((2, 3), dtype=np.float16),
            voice_scores=np.asarray([0.1, 0.9], dtype=np.float32),
            voice_states=np.asarray([2, 1], dtype=np.uint8),
        )
    document = {
        "schema_version": features.SCHEMA_VERSION,
        "cache_kind": features.CACHE_KIND,
        "source_fingerprint": "f" * 64,
        "arrays": {
            "semantic_dimensions": 3,
            "file_sha256": features._sha256(cache),
        },
    }
    document["payload_sha256"] = features._payload_sha256(document)
    metadata.write_text(json.dumps(document), encoding="utf-8")
    return cache, metadata, document


def test_load_semantic_cache_is_hash_and_identity_bound(tmp_path):
    cache, metadata, _ = write_cache(tmp_path)
    probabilities, scores, states = features.load_semantic_cache(
        cache,
        metadata,
        expected_source_fingerprint="f" * 64,
        expected_track_ids=np.asarray([1, 2], dtype=np.int64),
    )
    assert probabilities.shape == (2, 3)
    assert scores.tolist() == pytest.approx([0.1, 0.9])
    assert states.tolist() == [2, 1]
    with pytest.raises(features.V4FeatureError, match="identity"):
        features.load_semantic_cache(
            cache,
            metadata,
            expected_source_fingerprint="f" * 64,
            expected_track_ids=np.asarray([2, 1], dtype=np.int64),
        )


def test_load_semantic_cache_rejects_rehashed_metadata_tamper(tmp_path):
    cache, metadata, document = write_cache(tmp_path)
    tampered = copy.deepcopy(document)
    tampered["arrays"]["semantic_dimensions"] = 4
    tampered["payload_sha256"] = features._payload_sha256(tampered)
    metadata.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(features.V4FeatureError, match="shape drift"):
        features.load_semantic_cache(
            cache,
            metadata,
            expected_source_fingerprint="f" * 64,
            expected_track_ids=np.asarray([1, 2], dtype=np.int64),
        )
