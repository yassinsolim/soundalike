from dataclasses import fields
from pathlib import Path

import numpy as np

from soundalike.ml.catalog_policy import (
    GRAPH_AUDIO_SCENE_POLICY,
    GRAPH_ONLY_POLICY,
    CatalogPolicy,
    CatalogPolicyRanker,
    graph_score,
    policy_score,
)


class FakeRecommender:
    def __init__(self):
        self.titles = np.asarray(
            ["Seed", "Graph One", "Graph Two", "Audio Star", "Seed (Slowed)"]
        )
        self.artists = np.asarray(["seed", "safe", "unsafe", "audio", "junk"])
        self.track_ids = np.arange(5)
        self._sonic = np.asarray([[1, 0], [0, 1], [0.2, 0.8], [1, 0], [1, 0]], np.float32)
        self._clap = self._sonic.copy()
        self._vscaled = np.asarray([[0], [2], [1], [0], [0]], np.float32)
        self.static_popularity = np.asarray([0, 0, 0, 1e9, 1e10])


class FakeGraph:
    artist_audio = np.asarray([[1, 0], [0, 1], [0.2, 0.8], [1, 0], [1, 0]], np.float32)
    artist_lookup = {"seed": 0, "safe": 1, "unsafe": 2, "audio": 3, "junk": 4}
    track_artist_ids = np.arange(5, dtype=np.int32)
    track_rows = np.arange(5, dtype=np.int32)
    track_indptr = np.arange(6, dtype=np.int32)

    def artist_neighbors(self, artist, audio, variant="twohop"):
        assert variant == "full"
        return (
            np.asarray([1, 2], np.int32),
            np.asarray([0.8, 0.4], np.float32),
            "catalog_artist_graph",
        )


class FakeStyles:
    values = {"safe": 0.9, "unsafe": 0.1, "audio": 0.8, "junk": 0.9}

    def style_overlap(self, query, candidate):
        return self.values.get(candidate, 0.0)


def test_exact_formula_and_policy_has_only_three_numeric_parameters():
    policy = CatalogPolicy(0.4, 0.2, 0.3)
    assert [field.name for field in fields(policy)] == [
        "audio_weight", "style_weight", "style_guard_min"
    ]
    assert all(isinstance(getattr(policy, field.name), float) for field in fields(policy))
    expected_g = 0.7 * 0.5 + 0.3 / np.log2(3)
    assert np.isclose(graph_score(0.5, 2), expected_g)
    assert np.isclose(policy_score(expected_g, 0.25, 0.75, policy),
                      expected_g + 0.4 * 0.25 + 0.2 * 0.75)


def test_graph_is_primary_over_audio_only_candidate():
    ranker = CatalogPolicyRanker(
        FakeRecommender(), FakeGraph(), FakeStyles(), CatalogPolicy(0.2, 0.0, 0.0)
    )
    result = ranker.recommend(0, 3)["results"]
    assert result[0]["artist"] == "safe"
    assert result[0]["rationale"]["source"] == "graph"
    assert next(item for item in result if item["artist"] == "audio")["rationale"]["G"] == 0


def test_style_guard_moves_safe_candidates_into_first_three_stably():
    ranker = CatalogPolicyRanker(
        FakeRecommender(), FakeGraph(), FakeStyles(), CatalogPolicy(0.9, 0.0, 0.5)
    )
    result = ranker.recommend(0, 2)["results"]
    assert [item["artist"] for item in result] == ["safe", "audio"]
    assert all(item["rationale"]["S"] >= 0.5 for item in result)


def test_graph_only_control_ignores_audio_and_style_scores():
    ranker = CatalogPolicyRanker(
        FakeRecommender(), FakeGraph(), FakeStyles(), GRAPH_ONLY_POLICY
    )
    result = ranker.recommend(0, 3)["results"]
    assert [item["artist"] for item in result[:2]] == ["safe", "unsafe"]
    assert all(np.isclose(item["score"], item["rationale"]["G"]) for item in result)


def test_quality_filter_removes_derivatives_and_outputs_target_blind_rationales():
    ranker = CatalogPolicyRanker(
        FakeRecommender(), FakeGraph(), FakeStyles(), GRAPH_AUDIO_SCENE_POLICY
    )
    payload = ranker.recommend(0, 5)
    assert "junk" not in [item["artist"] for item in payload["results"]]
    assert "target" not in repr(payload).casefold()
    for item in payload["results"]:
        assert set(item["rationale"]) == {"G", "A", "S", "source", "query_mode"}


def test_runtime_module_has_no_large_legacy_ranker_or_external_catalog_dependency():
    source = Path("src/soundalike/ml/catalog_policy.py").read_text(encoding="utf-8").casefold()
    forbidden = ("production" + "ranker", "music" + "4all")
    assert all(name not in source for name in forbidden)


def test_static_popularity_is_not_read_or_used():
    recommender = FakeRecommender()
    ranker = CatalogPolicyRanker(
        recommender, FakeGraph(), FakeStyles(), GRAPH_AUDIO_SCENE_POLICY
    )
    before = [item["track_id"] for item in ranker.recommend(0, 4)["results"]]
    recommender.static_popularity[:] = recommender.static_popularity[::-1]
    after = [item["track_id"] for item in ranker.recommend(0, 4)["results"]]
    assert after == before
