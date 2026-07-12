import json
from pathlib import Path

import numpy as np

from soundalike.ml.collaborative import CollaborativeIndex, _mask_edges
from soundalike.ml.collaborative_rerank import (
    FEATURE_NAMES,
    CollaborativeHybridRanker,
    LinearScorer,
)
from soundalike.ml.final_protocol import _bootstrap, validate_benchmark
from soundalike.ml.final_protocol import _verify_state_signature


def _asset(path: Path) -> Path:
    np.savez_compressed(
        path,
        catalog_rows=np.asarray([1, 2, 3], dtype=np.int32),
        vectors=np.asarray(
            [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float16
        ),
        artist_names=np.asarray(["artist a", "artist b"]),
        artist_vectors=np.asarray(
            [[1.0, 0.0], [0.0, 1.0]], dtype=np.float16
        ),
        metadata=np.asarray(json.dumps({"edge_masked": True})),
    )
    return path


def test_edge_mask_removes_deciding_pair_cooccurrence():
    tokens = ["query", "other", "target", "other"]
    masked, overlaps = _mask_edges(
        "user-1", tokens, {("query", "target")}
    )
    assert overlaps == 1
    assert not {"query", "target"} <= set(masked)
    assert "other" in masked


def test_collaborative_index_supports_track_artist_and_audio_bridge(tmp_path):
    index = CollaborativeIndex(_asset(tmp_path / "collab.npz"), 8)
    track, mode = index.query_vector(1, "unknown")
    assert mode == "track"
    assert np.allclose(track, [1.0, 0.0], atol=1e-3)

    artist, mode = index.query_vector(7, "Artist B")
    assert mode == "artist"
    assert np.allclose(artist, [0.0, 1.0], atol=1e-3)

    audio = np.zeros(8, dtype=np.float32)
    audio[1] = 1.0
    bridged, mode = index.query_vector(7, "unknown", audio_scores=audio)
    assert mode == "audio_bridge"
    assert bridged is not None
    assert bridged[0] > bridged[1]


def test_hybrid_union_is_deduplicated_and_balanced():
    pools = (
        np.asarray([1, 2, 3]),
        np.asarray([1, 4, 5]),
        np.asarray([6, 2, 7]),
    )
    union = CollaborativeHybridRanker._round_robin(pools, limit=7)
    assert union.tolist() == [1, 6, 2, 4, 3, 5, 7]
    assert len(union) == len(set(union))


def test_learned_scorer_has_no_notability_signal():
    coefficients = tuple(1.0 for _ in FEATURE_NAMES[:-1]) + (0.0,)
    scorer = LinearScorer(FEATURE_NAMES, coefficients, 0.0, 1.0)
    document = scorer.to_dict()
    assert document["global_notability_weight"] == 0.0
    assert FEATURE_NAMES[-1] == "global_notability_zero"
    features = np.zeros((1, len(FEATURE_NAMES)), dtype=np.float32)
    features[0, -1] = 999.0
    assert scorer.score(features)[0] == 0.0


def test_zero_baseline_has_no_relative_gain_claim():
    comparison = _bootstrap([0, 0, 0], [1, 0, 0], iterations=100, seed=1)
    assert comparison["baseline_primary"] == 0.0
    assert comparison["relative_gain"] is None
    assert comparison["absolute_delta"] > 0


def test_fresh_v6_benchmark_and_frozen_signature_exist():
    benchmark = json.loads(
        Path("benchmarks/soundalike_pairs.v6.json").read_text(encoding="utf-8")
    )
    audit = validate_benchmark(benchmark)
    assert audit["final_pairs"] >= 80
    assert audit["scenes"] >= 15
    protocol = Path(".goals/human-quality-recommendations/protocol-v6")
    metadata = json.loads(
        (protocol / "frozen-signature-metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["private_key_retained"] is False
    assert (protocol / "frozen-state.sig").is_file()


def test_v6_final_was_rankings_locked_and_opened_once():
    protocol = Path(".goals/human-quality-recommendations/protocol-v6")
    state_path = protocol / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    _verify_state_signature(state, state_path)
    assert state["status"] == "FINALIZED"
    assert state["final_open_count"] == 1
    assert state["rankings_locked_at"] < state["final_opened_at"]
    assert state["final_pass"] is False


def test_real_collaborative_training_removed_final_edges():
    report = json.loads(
        Path(
            ".goals/human-quality-recommendations/artifacts/"
            "collaborative-training-v6.json"
        ).read_text(encoding="utf-8")
    )
    assert report["source"]["license"] == "CC-BY-4.0"
    assert report["corpus"]["users_with_two_mapped_tracks"] > 100_000
    assert report["corpus"]["final_pair_user_overlaps_before_mask"] > 0
    assert report["corpus"]["final_pair_user_overlaps_after_mask"] == 0
    assert report["models"]["edge_masked"]["mapped_catalogue_rows"] > 10_000


def test_final_report_contains_all_candidate_generators():
    report = json.loads(
        Path(
            ".goals/human-quality-recommendations/artifacts/"
            "collaborative-final-once-v6.json"
        ).read_text(encoding="utf-8")
    )
    assert report["open_number"] == 1
    assert {
        "audio_only",
        "collaborative_edge_masked",
        "hybrid_union_edge_masked",
    } <= set(report["candidate_recall"])
    assert report["comparison_to_production_baseline"]["final_pass"] is False
