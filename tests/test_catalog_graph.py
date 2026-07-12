import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from soundalike.ml.catalog_graph import (
    CatalogArtistGraph,
    mask_final_topology,
)
from soundalike.ml.catalog_protocol import ProtocolError, validate_benchmark
from soundalike.ml.catalog_protocol import _graded_rows, _per_seed
from soundalike.ml.catalog_rerank import FEATURE_NAMES, HybridScorer
from soundalike.ml.real_benchmark import PairResolver


def _graph_asset(path: Path) -> Path:
    np.savez_compressed(
        path,
        artist_names=np.asarray(["a", "b", "c", "d"]),
        track_artist_ids=np.asarray([0, 1, 1, 2, 3], dtype=np.int32),
        track_rows=np.asarray([0, 1, 2, 3, 4], dtype=np.int32),
        track_indptr=np.asarray([0, 1, 3, 4, 5], dtype=np.int32),
        source_mapped=np.asarray([1, 1, 1, 0], dtype=np.uint8),
        artist_audio=np.asarray(
            [[1, 0], [0.9, 0.1], [0, 1], [0.8, 0.2]], dtype=np.float16
        ),
        full_indices=np.asarray(
            [[1, 2], [0, 2], [1, 0], [-1, -1]], dtype=np.int32
        ),
        full_weights=np.asarray(
            [[0.9, 0.2], [0.9, 0.8], [0.8, 0.2], [0, 0]],
            dtype=np.float16,
        ),
        direct_indices=np.asarray(
            [[1, 2], [0, 2], [1, 0], [-1, -1]], dtype=np.int32
        ),
        direct_weights=np.asarray(
            [[0.9, 0.2], [0.9, 0.8], [0.8, 0.2], [0, 0]],
            dtype=np.float16,
        ),
        twohop_indices=np.asarray(
            [[1, -1], [0, 2], [1, -1], [-1, -1]], dtype=np.int32
        ),
        twohop_weights=np.asarray(
            [[0.9, 0], [0.9, 0.8], [0.8, 0], [0, 0]],
            dtype=np.float16,
        ),
        metadata=np.asarray(json.dumps({"runtime_contains_secret": False})),
    )
    return path


def test_catalog_graph_covers_mapped_and_cold_start_queries(tmp_path):
    graph = CatalogArtistGraph(_graph_asset(tmp_path / "graph.npz"))
    rows, weights, mode = graph.artist_neighbors(
        "a", np.asarray([1.0, 0.0]), variant="twohop"
    )
    assert mode == "catalog_artist_graph"
    assert rows.tolist() == [1]
    assert weights[0] > 0

    rows, _, mode = graph.artist_neighbors(
        "d", np.asarray([1.0, 0.0]), variant="twohop", anchors=1
    )
    assert mode == "audio_artist_bridge"
    assert 1 in rows


def test_catalog_track_expansion_interleaves_artists(tmp_path):
    graph = CatalogArtistGraph(_graph_asset(tmp_path / "graph.npz"))
    audio = np.asarray([0.0, 0.8, 0.9, 0.2, 0.1], dtype=np.float32)
    rows, _, _ = graph.candidates(
        0,
        "a",
        np.asarray([1.0, 0.0]),
        audio,
        n=3,
        variant="full",
        max_tracks_per_artist=2,
    )
    assert rows.tolist()[:2] == [2, 3]
    assert len(rows) == len(set(rows.tolist()))


def test_two_hop_mask_breaks_direct_and_transitive_paths():
    indices = np.asarray(
        [[1, 2, -1], [0, 2, -1], [0, 1, -1]], dtype=np.int32
    )
    weights = np.where(indices >= 0, 1.0, 0.0).astype(np.float32)
    _, _, twohop_indices, _, audit = mask_final_topology(
        indices, weights, [(0, 1, "final-1")]
    )
    assert audit["exact_edges_present_before_mask"] == 1
    assert audit["exact_edges_present_after_mask"] == 0
    assert audit["two_hop_paths_before_mask"] == 1
    assert audit["two_hop_paths_after_mask"] == 0
    assert 1 not in set(twohop_indices[0])


def test_hybrid_scorer_cannot_use_static_popularity():
    coefficients = tuple([1.0] * (len(FEATURE_NAMES) - 1) + [0.0])
    scorer = HybridScorer(coefficients)
    features = np.zeros((1, len(FEATURE_NAMES)), dtype=np.float32)
    features[0, -1] = 1_000_000.0
    assert scorer.score(features)[0] == 0.0
    assert scorer.to_dict()["global_popularity_weight"] == 0.0


def test_artist_relevance_counts_once_across_catalog_tracks():
    resolver = PairResolver(
        ["query", "one", "two", "other"],
        ["seed", "related", "related", "unrelated"],
    )
    record = {
        "positives": [
            {
                "title": "one",
                "artist": "related",
                "grade": 3,
                "relevance_scope": "artist",
            }
        ]
    }
    relevance = _graded_rows(resolver, record)
    assert set(relevance) == {1, 2}
    metrics = _per_seed(
        [{"row": 1}, {"row": 2}, {"row": 3}], relevance
    )
    assert metrics["ndcg_at_10"] == 1.0
    assert metrics["recall_at_10"] == 1.0


def test_v7_benchmark_validator_accepts_multi_positive_components():
    records = []
    for split, count in (("development", 5), ("final", 50)):
        for number in range(count):
            prefix = "d" if split == "development" else "f"
            positives = [
                {
                    "title": f"{prefix}-target-{number}-{index}",
                    "artist": f"{prefix}-target-artist-{number}-{index}",
                    "grade": 3 if index < 2 else 1,
                }
                for index in range(5)
            ]
            records.append(
                {
                    "id": f"{prefix}-{number}",
                    "split": split,
                    "scene": f"scene-{number % 12}",
                    "catalog_tier": ("popular", "deep_cut", "niche")[number % 3],
                    "query": {
                        "title": f"{prefix}-query-{number}",
                        "artist": f"{prefix}-query-artist-{number}",
                    },
                    "positives": positives,
                    "evidence_axis": "taste_affinity",
                    "source": {
                        "url": "https://example.test/source",
                        "publisher": "Example",
                        "accessed_at": "2026-07-12",
                        "source_class": "independent_test",
                        "excerpt": "Independent multi-positive relevance evidence.",
                    },
                }
            )
    audit = validate_benchmark({"records": records})
    assert audit["final_seeds"] == 50
    assert audit["final_positives"] == 250
    assert audit["artist_overlap"] == []


def test_v7_validator_splits_featured_artist_credits():
    benchmark = {
        "records": [
            {
                "id": f"d-{number}",
                "split": "development",
                "scene": f"scene-{number % 12}",
                "catalog_tier": ("popular", "deep_cut", "niche")[number % 3],
                "query": {
                    "title": f"dev-query-{number}",
                    "artist": "Shared" if number == 0 else f"dev-{number}",
                },
                "positives": [
                    {
                        "title": f"dev-positive-{number}-{index}",
                        "artist": f"dev-positive-artist-{number}-{index}",
                        "grade": 1,
                    }
                    for index in range(5)
                ],
                "evidence_axis": "taste_affinity",
                "source": {
                    "url": "https://example.test",
                    "publisher": "Example",
                    "accessed_at": "2026-07-12",
                    "source_class": "test",
                    "excerpt": "test source",
                },
            }
            for number in range(5)
        ]
    }
    for number in range(50):
        benchmark["records"].append(
            {
                "id": f"f-{number}",
                "split": "final",
                "scene": f"scene-{number % 12}",
                "catalog_tier": ("popular", "deep_cut", "niche")[number % 3],
                "query": {
                    "title": f"final-query-{number}",
                    "artist": f"final-{number}",
                },
                "positives": [
                    {
                        "title": f"final-positive-{number}-{index}",
                        "artist": (
                            "Other feat. Shared"
                            if number == 0 and index == 0
                            else f"final-positive-artist-{number}-{index}"
                        ),
                        "grade": 1,
                    }
                    for index in range(5)
                ],
                "evidence_axis": "taste_affinity",
                "source": {
                    "url": "https://example.test",
                    "publisher": "Example",
                    "accessed_at": "2026-07-12",
                    "source_class": "test",
                    "excerpt": "test source",
                },
            }
        )
    with pytest.raises(ProtocolError, match="component overlap"):
        validate_benchmark(benchmark)


def test_real_v7_benchmark_is_fresh_graded_and_source_separated():
    benchmark = json.loads(
        Path("benchmarks/soundalike_multipositive.v7.json").read_text(
            encoding="utf-8"
        )
    )
    audit = validate_benchmark(benchmark)
    assert audit["final_seeds"] >= 50
    assert audit["final_scenes"] >= 12
    assert min(
        len(record["positives"]) for record in benchmark["records"]
    ) >= 5
    assert benchmark["source_policy"]["same_dataset_or_api"] is False
    assert benchmark["axis_policy"]["ship_requires_both"] is True


def test_real_v7_protocol_was_locked_opened_once_and_sealed():
    protocol = Path(".goals/human-quality-recommendations/protocol-v7")
    state = json.loads((protocol / "state.json").read_text(encoding="utf-8"))
    assert state["status"] == "FINALIZED"
    assert state["final_open_count"] == 1
    assert state["rankings_locked_before_open"] is True
    assert state["rankings_locked_at"] < state["final_opened_at"]
    assert state["retrieval_pass"] is False
    assert state["direct_pass"] is False
    assert state["final_pass"] is False
    verified = subprocess.run(
        [
            "ssh-keygen",
            "-Y",
            "verify",
            "-f",
            str(protocol / "allowed_signers"),
            "-I",
            "soundalike-protocol",
            "-n",
            "soundalike-protocol",
            "-s",
            str(protocol / "state.sig"),
        ],
        input=(protocol / "state.json").read_bytes(),
        capture_output=True,
        check=False,
    )
    assert verified.returncode == 0


def test_real_catalog_graph_meets_effective_coverage_and_masks_paths():
    report = json.loads(
        Path(
            ".goals/human-quality-recommendations/artifacts/"
            "catalog-graph-source-audit-v7.json"
        ).read_text(encoding="utf-8")
    )
    coverage = report["coverage"]
    assert coverage["effective_track_coverage"] >= 0.70
    assert coverage["effective_query_artist_coverage"] >= 0.80
    assert coverage["source_mapped_tracks"] > 100_000
    mask = report["leakage_mask_audit"]
    assert mask["exact_edges_present_before_mask"] > 0
    assert mask["exact_edges_present_after_mask"] == 0
    assert mask["two_hop_paths_before_mask"] > 0
    assert mask["two_hop_paths_after_mask"] == 0
    projection = report["projection_validation"]
    assert (
        projection["artists_with_nonempty_full_graph_neighborhood"]
        == projection["catalogue_artists"]
    )
    assert (
        projection["tracks_whose_artist_has_graph_neighborhood"]
        == projection["catalogue_tracks"]
    )


def test_real_v7_negative_result_blocks_deployment():
    root = Path(".goals/human-quality-recommendations/artifacts")
    final = json.loads(
        (root / "catalog-hybrid-final-once-v7.json").read_text(encoding="utf-8")
    )
    assert final["open_number"] == 1
    assert {
        "audio_only",
        "music4all_sparse",
        "catalog_graph_full",
        "catalog_graph_direct_masked",
        "catalog_graph_twohop_masked",
        "hybrid_union_twohop_masked",
    } <= set(final["candidate_recall"])
    assert (
        final["comparison_to_production_baseline"]["retrieval_pass"] is False
    )
    direct = json.loads(
        (root / "catalog-direct-judgments-v7.json").read_text(encoding="utf-8")
    )
    assert direct["summary"]["total"] == 20
    assert direct["summary"]["resolved_queries"] == 20
    assert direct["summary"]["passes_gate"] is False
    deployment = json.loads(
        (root / "catalog-deployment-status-v7.json").read_text(encoding="utf-8")
    )
    assert deployment["iteration6_deployed"] is False
    assert deployment["production_unchanged"] is True
