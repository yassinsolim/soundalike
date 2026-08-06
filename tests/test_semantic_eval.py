"""Tests for the exploratory audio-versus-semantic listening study."""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from soundalike.ml.semantic_eval import (
    AUDIO_WEIGHT,
    CANDIDATE_POOL,
    EXPECTED_PREDICTOR_PAYLOAD_SHA256,
    EXPECTED_PREDICTOR_SHA256,
    EXPECTED_SOURCE_FINGERPRINT,
    EXPECTED_STORE_BINDING_SHA256,
    EXPECTED_TEST_TRACKS,
    EXPECTED_V2_PACK_SHA256,
    MAX_RESULTS_PER_ARTIST,
    METHODS,
    SECTION_BUDGET,
    SEMANTIC_WEIGHT,
    SemanticEvalConfig,
    SemanticEvalError,
    artist_diverse_top,
    build_blinded_documents,
    percentile_scores,
    rank_study_methods,
    semantic_blend_scores,
    validate_blinded_documents,
)


ROOT = Path(__file__).resolve().parents[1]
PUBLIC_PACK = ROOT / "webapp" / "evaluate" / "semantic-pack.json"
PUBLIC_PACK_CONTENT_SHA256 = (
    "4f3c34250d5c5fca35dcc671dae1c256f0d56d8ce404d7a758bbbf62a2e5b48a"
)
PUBLIC_PACK_FILE_SHA256 = (
    "7b05b1ecd74534c75148f1e5855a33b911e2f9337c26ace26ebb1e2448791acc"
)


def test_semantic_blend_is_rank_scaled_and_frozen():
    audio = np.asarray([0.3, 0.1, 0.2], dtype=np.float32)
    semantic = np.asarray([0.0, 1.0, 0.5], dtype=np.float32)
    assert np.allclose(percentile_scores(audio), [1.0, 0.0, 0.5])
    assert np.allclose(
        semantic_blend_scores(audio, semantic),
        AUDIO_WEIGHT * np.asarray([1.0, 0.0, 0.5])
        + SEMANTIC_WEIGHT * np.asarray([0.0, 1.0, 0.5]),
    )
    with pytest.raises(SemanticEvalError, match="frozen"):
        semantic_blend_scores(audio, semantic, semantic_weight=0.5)


def test_artist_diverse_top_retains_one_track_per_artist():
    track_ids = [1, 2, 3, 4, 5, 6]
    artists = {1: 10, 2: 10, 3: 20, 4: 30, 5: 40, 6: 50}
    selected = artist_diverse_top(
        track_ids,
        np.asarray([1.0, 0.9, 0.8, 0.7, 0.6, 0.5]),
        artists,
    )
    assert selected == (1, 3, 4, 5, 6)


def test_semantic_config_rejects_hidden_tuning_surfaces():
    SemanticEvalConfig().validate()
    with pytest.raises(SemanticEvalError, match="weight"):
        SemanticEvalConfig(semantic_weight=0.3).validate()
    with pytest.raises(SemanticEvalError, match="budget"):
        SemanticEvalConfig(section_budget=8).validate()


def _documents():
    source_seeds = []
    baseline = {}
    semantic = {}
    track_records = {}
    for seed_index in range(20):
        seed_track_id = 1000 + seed_index
        source_seeds.append(
            {
                "seed_id": f"old-seed-{seed_index}",
                "scene": f"scene-{seed_index}",
                "seed_track_id": seed_track_id,
                "tempo_bpm": 100.0 + seed_index,
                "tempo_region": "medium",
                "clap_texture_region": seed_index % 5,
            }
        )
        baseline[seed_track_id] = tuple(
            10_000 + seed_index * 20 + offset for offset in range(5)
        )
        semantic[seed_track_id] = tuple(
            10_000 + seed_index * 20 + offset for offset in range(5, 10)
        )
        for track_id in (seed_track_id, *baseline[seed_track_id], *semantic[seed_track_id]):
            track_records[str(track_id)] = {
                "track_id": track_id,
                "title": f"Track {track_id}",
                "artist": f"Artist {track_id}",
                "source_identity": {
                    "artist_id": track_id,
                    "fold": 0,
                    "fold_part": "test",
                },
            }
    shared = {
        "store_binding_sha256": EXPECTED_STORE_BINDING_SHA256,
        "candidate_pool": CANDIDATE_POOL,
        "section_budget": SECTION_BUDGET,
        "max_results_per_artist": MAX_RESULTS_PER_ARTIST,
        "published_v2_results_excluded": True,
        "source_seed_pack_sha256": EXPECTED_V2_PACK_SHA256,
        "test_labels_used_for_ranking": False,
        "language_metadata_used_for_ranking": False,
        "promoted": False,
    }
    method_bindings = {
        METHODS[0]: {
            **shared,
            "method": METHODS[0],
            "audio_weight": 1.0,
            "semantic_weight": 0.0,
            "semantic_profile_used_for_ranking": False,
        },
        METHODS[1]: {
            **shared,
            "method": METHODS[1],
            "predictor_payload_sha256": EXPECTED_PREDICTOR_PAYLOAD_SHA256,
            "predictor_model_sha256": EXPECTED_PREDICTOR_SHA256,
            "audio_weight": AUDIO_WEIGHT,
            "semantic_weight": SEMANTIC_WEIGHT,
            "semantic_profile_used_for_ranking": True,
            "domain_diagnostics": {
                "rows": EXPECTED_TEST_TRACKS,
                "dimensions": 512,
                "standardized_coordinate_mean_abs": 0.05,
                "standardized_coordinate_scale_mean": 1.02,
                "maximum_standardized_coordinate_mean_abs": 0.25,
                "minimum_standardized_coordinate_scale_mean": 0.8,
                "maximum_standardized_coordinate_scale_mean": 1.2,
                "passed": True,
            },
        },
    }
    store_binding = json.loads(PUBLIC_PACK.read_text(encoding="utf-8"))["store_binding"]
    return build_blinded_documents(
        source_seeds=source_seeds,
        track_records=track_records,
        audio_control_rankings=baseline,
        semantic_rankings=semantic,
        method_bindings=method_bindings,
        store_binding=store_binding,
        source_fingerprint=EXPECTED_SOURCE_FINGERPRINT,
        blinding_key=b"\x42" * 32,
    )


def test_blinded_documents_bind_methods_without_public_identity():
    public, private = _documents()
    validate_blinded_documents(public, private, require_frozen_artifacts=False)
    assert public["method_count"] == 2
    assert public["language_policy"]["evaluated_here"] is False
    text = str(public)
    assert not any(method in text for method in METHODS)
    assert private["methods"] == list(METHODS)


def test_blinded_documents_reject_rehashed_ranking_tampering():
    public, private = _documents()
    tampered = copy.deepcopy(public)
    tampered["seeds"][0]["lists"][0]["ranking"][0]["track_id"] += 1
    from soundalike.ml.fulltrack_pilot import _content_sha256

    tampered["content_sha256"] = _content_sha256(tampered)
    with pytest.raises(SemanticEvalError, match="ranking result identity"):
        validate_blinded_documents(
            tampered, private, require_frozen_artifacts=False
        )


def test_blinded_documents_reject_truncated_public_seed_set():
    public, private = _documents()
    tampered = copy.deepcopy(public)
    tampered["seeds"].pop()
    from soundalike.ml.fulltrack_pilot import _content_sha256

    tampered["content_sha256"] = _content_sha256(tampered)
    with pytest.raises(SemanticEvalError, match="seed count"):
        validate_blinded_documents(
            tampered, private, require_frozen_artifacts=False
        )


def _rehash_pair(public: dict, private: dict) -> None:
    from soundalike.ml.fulltrack_pilot import _content_sha256

    private["content_sha256"] = _content_sha256(private)
    public["blinding"]["private_unblinding_sha256"] = private["content_sha256"]
    public["content_sha256"] = _content_sha256(public)


@pytest.mark.parametrize("field", ["position", "result_id"])
def test_blinded_documents_reject_rehashed_result_identity_tampering(field):
    public, private = _documents()
    tampered = copy.deepcopy(public)
    row = tampered["seeds"][0]["lists"][0]["ranking"][0]
    row[field] = 99 if field == "position" else "semantic-result-" + "f" * 24
    from soundalike.ml.fulltrack_pilot import _content_sha256

    tampered["content_sha256"] = _content_sha256(tampered)
    with pytest.raises(SemanticEvalError, match="result identity"):
        validate_blinded_documents(
            tampered, private, require_frozen_artifacts=False
        )


def test_blinded_documents_reject_rehashed_private_method_metadata():
    public, private = map(copy.deepcopy, _documents())
    private["method_bindings"][METHODS[0]]["section_budget"] = 31
    _rehash_pair(public, private)
    with pytest.raises(SemanticEvalError, match="method binding policy"):
        validate_blinded_documents(
            public, private, require_frozen_artifacts=False
        )


def test_blinded_documents_reject_rehashed_duplicate_private_lists():
    public, private = map(copy.deepcopy, _documents())
    private["seeds"][0]["lists"][1] = copy.deepcopy(
        private["seeds"][0]["lists"][0]
    )
    _rehash_pair(public, private)
    with pytest.raises(SemanticEvalError, match="list identity map"):
        validate_blinded_documents(
            public, private, require_frozen_artifacts=False
        )


def test_blinded_documents_reject_rehashed_source_provenance():
    public, private = map(copy.deepcopy, _documents())
    public["source_fingerprint"] = "b" * 64
    private["source_fingerprint"] = "b" * 64
    _rehash_pair(public, private)
    with pytest.raises(SemanticEvalError, match="source/store provenance"):
        validate_blinded_documents(
            public, private, require_frozen_artifacts=False
        )


def test_verifier_requires_the_exact_frozen_artifacts_by_default():
    public, private = _documents()
    with pytest.raises(SemanticEvalError, match="frozen semantic study"):
        validate_blinded_documents(public, private)


def test_ranker_source_never_reads_test_tags():
    source = inspect.getsource(rank_study_methods)
    assert "track_tags" not in source
    assert ".tags" not in source


def test_committed_public_pack_is_frozen_and_schema_valid():
    payload = PUBLIC_PACK.read_bytes()
    document = json.loads(payload)
    content = {key: value for key, value in document.items() if key != "content_sha256"}
    content_sha = hashlib.sha256(
        json.dumps(
            content, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()

    assert hashlib.sha256(payload).hexdigest() == PUBLIC_PACK_FILE_SHA256
    assert document["content_sha256"] == content_sha == PUBLIC_PACK_CONTENT_SHA256
    assert document["schema_version"] == 1
    assert document["pack_kind"] == "fulltrack_semantic_blind_pilot_v1"
    assert document["method_count"] == 2
    assert document["seed_count"] == 20
    assert sum(len(seed["lists"]) for seed in document["seeds"]) == 40
