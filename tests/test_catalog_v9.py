import json
from dataclasses import fields

import numpy as np
import pytest

from soundalike.ml.catalog_cv_v9 import ListGoldScorer, summarize_predictions
from soundalike.ml.catalog_list_gold_v9 import (
    GoldBuildError,
    REQUIRED_SCENES,
    load_seed_specs,
    validate_gold,
    validate_music_map_snapshots,
)
from soundalike.ml.catalog_tier_v9 import _safe_cli
from soundalike.ml.catalog_policy_v9 import (
    LastfmListRanker,
    ListPolicy,
    OPTIONAL_MUSIC4ALL_WEIGHT,
)


class FakeRecommender:
    def __init__(self):
        self.titles = np.asarray(["Seed"] + [f"Track {i}" for i in range(1, 13)])
        self.artists = np.asarray(["seed"] + [f"artist {i}" for i in range(1, 13)])
        self.track_ids = np.arange(100, 113)
        self._sonic = np.ones((13, 2), np.float32)
        self._clap = np.ones((13, 2), np.float32)
        self._vscaled = np.zeros((13, 1), np.float32)
        self.alpha = 0.8

    def recommend(self, row, n=10, **kwargs):
        assert row == 0
        return {
            "results": [
                {
                    "deezer_id": int(self.track_ids[candidate]),
                    "title": str(self.titles[candidate]),
                    "artist": str(self.artists[candidate]),
                }
                for candidate in [12, 11, 10, 9, 8, 7, 6][:n]
            ]
        }


class FakeGraph:
    track_artist_ids = np.arange(13, dtype=np.int32)
    track_rows = np.arange(13, dtype=np.int32)
    track_indptr = np.arange(14, dtype=np.int32)

    def __init__(self, *, lastfm=True, music4all=False):
        left = np.arange(1, 9, dtype=np.int32) if lastfm else np.empty(0, np.int32)
        right = np.arange(1, 7, dtype=np.int32) if music4all else np.empty(0, np.int32)
        self.payload = {
            "lastfm": {
                "artist_ids": left,
                "weights": np.linspace(1.0, 0.65, len(left), dtype=np.float32),
            },
            "music4all": {
                "artist_ids": right,
                "weights": np.linspace(1.0, 0.7, len(right), dtype=np.float32),
            },
            "union_artist_ids": np.asarray(sorted(set(left) | set(right)), np.int32),
            "source_coverage": {
                "lastfm": bool(lastfm),
                "music4all": bool(music4all),
            },
        }

    def dual_source_neighbors(self, artist):
        assert artist == "seed"
        return self.payload


class FakeStyles:
    def __init__(self, value):
        self.value = value

    def style_overlap(self, query, candidate):
        return self.value


def test_powered_gold_is_real_multipositive_and_source_grounded():
    gold = json.load(open("benchmarks/soundalike_list_gold.v9.json", encoding="utf-8"))
    snapshots = json.load(
        open("benchmarks/evidence/v9/music-map.normalized.json", encoding="utf-8")
    )
    validation = validate_gold(gold)
    assert validation == {
        "passed": True,
        "errors": [],
        "seeds": 60,
        "scenes": 13,
        "positives": 815,
    }
    assert set(REQUIRED_SCENES) <= {record["scene"] for record in gold["records"]}
    assert len(snapshots["records"]) == 60
    assert all(item["http_status"] == 200 for item in snapshots["records"])
    assert all(item["source_url"].startswith("https://www.music-map.com/") for item in snapshots["records"])
    assert all(item["accessed_at"] == "2026-07-12" for item in snapshots["records"])
    assert all(len(item["neighbors"]) == 48 for item in snapshots["records"])
    assert all(len(record["positives"]) >= 5 for record in gold["records"])
    assert all(len(record["relevance_grades"]) >= 2 for record in gold["records"])


def test_policy_has_only_three_parameters_and_music4all_is_optional():
    assert [field.name for field in fields(ListPolicy)] == [
        "tau", "sigma", "audio_weight"
    ]
    assert OPTIONAL_MUSIC4ALL_WEIGHT == 0.15
    ranker = LastfmListRanker(
        FakeRecommender(), FakeGraph(music4all=False), FakeStyles(0.9)
    )
    cached = ranker.precompute_list_query(
        0, production_rows=[12, 11, 10, 9, 8, 7, 6]
    )
    result = ranker.apply_precomputed_list_policy(
        cached, ListPolicy(0.5, 0.5, 0.05), 7
    )
    assert result["fired"] is True
    assert result["source_coverage"] == {"lastfm": True, "music4all": False}
    assert result["music4all_shared_neighbors"] == 0
    assert result["ranking_rows"][:5] == [1, 2, 3, 4, 5]


def test_policy_applies_sigma_per_track_and_abstains_exactly():
    ranker = LastfmListRanker(
        FakeRecommender(), FakeGraph(music4all=True), FakeStyles(0.2)
    )
    production = [12, 11, 10, 9, 8, 7, 6]
    cached = ranker.precompute_list_query(0, production_rows=production)
    result = ranker.apply_precomputed_list_policy(
        cached, ListPolicy(0.5, 0.3, 0.05), 7
    )
    assert result["fired"] is False
    assert result["reason"] == "fewer_than_five_song_consistent_candidates"
    assert result["ranking_rows"] == production


def test_list_scorer_uses_actual_rows_and_independent_coherence():
    gold = {
        "records": [{
            "id": "s1",
            "query": {"artist": "seed"},
            "positives": [
                {
                    "relevance_scope": "artist", "artist": "artist 1",
                    "grade": 2, "rationale": "map", "uncertainty": "medium",
                },
                {
                    "relevance_scope": "artist", "artist": "artist 2",
                    "grade": 1, "rationale": "map", "uncertainty": "medium",
                },
            ],
        }]
    }
    snapshots = {
        "records": [{
            "seed_id": "s1",
            "neighbors": [{"artist": f"artist {i}"} for i in range(1, 6)],
        }]
    }
    titles = np.asarray(["Seed"] + [f"Track {i}" for i in range(1, 6)])
    artists = np.asarray(["seed"] + [f"artist {i}" for i in range(1, 6)])
    scorer = ListGoldScorer(
        gold, snapshots, titles, artists, np.arange(100, 106)
    )
    score = scorer.score(gold["records"][0], [1, 2, 3, 4, 5])
    assert score["ndcg_at_10"] == pytest.approx(1.0)
    assert score["mrr_at_10"] == 1.0
    assert score["coherence_pass"] is True
    assert score["unrelated_positions_1_to_3"] == 0
    assert score["junk_count"] == 0
    same_artist_at_five = scorer.score(
        gold["records"][0], [1, 2, 3, 4, 0]
    )
    assert same_artist_at_five["coherence_pass"] is False
    assert same_artist_at_five["top5_same_artist_count"] == 1


def test_powered_gate_requires_both_separate_co_primaries():
    baseline = {
        "ndcg_at_10": 0.10, "mrr_at_10": 0.1, "recall_at_10": 0.1,
        "coherence_fraction_at_5": 0.8, "coherence_pass": False,
        "candidate_recall_at_1000": 0.1, "junk_count": 0,
    }
    challenger = {
        "ndcg_at_10": 0.15, "mrr_at_10": 0.15, "recall_at_10": 0.15,
        "coherence_fraction_at_5": 1.0, "coherence_pass": True,
        "candidate_recall_at_1000": 0.2, "junk_count": 0,
    }
    predictions = [
        {
            "scene": f"scene-{index % 12}",
            "baseline": baseline,
            "challenger": challenger,
        }
        for index in range(60)
    ]
    summary = summarize_predictions(predictions)
    assert summary["relative_ndcg_gain"] == pytest.approx(0.5)
    assert summary["challenger"]["coherence_pass_rate"] == 1.0
    assert summary["gate_pass"] is True



def test_real_powered_dev_report_passes_retrieval_but_blocks_final():
    report = json.load(open(
        ".goals/human-quality-recommendations/artifacts/"
        "catalog-powered-sonic-dev-v9.json",
        encoding="utf-8",
    ))
    nested = report["nested_5fold"]["aggregate_outer_predictions"]
    assert nested["relative_ndcg_gain"] > 1.47
    assert nested["bootstrap"]["ci95_low"] > 0
    assert nested["improved"] == 34
    assert nested["worst_scene_relative_change"] >= 0
    assert nested["challenger"]["coherence_pass_rate"] == pytest.approx(16 / 60)
    assert nested["gates"]["top5_coherence_at_least_80pct"] is False
    assert report["all_powered_quality_dev_gates_passed"] is False
    assert report["final_open_count"] == 0
    assert report["fresh_final_created"] is False
    assert report["production_unchanged"] is True
    assert len(report["actual_selected_lists"]) == 60
    assert all(
        len(record["lists"][role]) == 10
        for record in report["actual_selected_lists"]
        for role in ("production_baseline", "challenger")
    )


def test_model_blind_judgments_are_hash_bound_and_still_fail_80pct():
    root = ".goals/human-quality-recommendations/artifacts/"
    blind = json.load(open(root + "catalog-powered-blind-lists-v9.json", encoding="utf-8"))
    judgments = json.load(open(
        root + "catalog-powered-model-blind-judgments-v9.json", encoding="utf-8"
    ))
    key = json.load(open(root + "catalog-powered-blind-key-v9.json", encoding="utf-8"))
    assert judgments["blind_content_sha256"] == blind["content_sha256"]
    mapping = {
        (item["id"], item["alias"]): item["method_role"]
        for item in key["records"]
    }
    passed = {"production_baseline": 0, "challenger": 0}
    for record in judgments["records"]:
        for alias in record["lists"]:
            passed[mapping[(record["id"], alias["alias"])]] += int(alias["passed"])
    assert passed == {"production_baseline": 1, "challenger": 5}
    assert passed["challenger"] / 60 < 0.8


def test_v9_tier_and_protocol_fail_closed_without_final():
    tier = json.load(open(
        ".goals/human-quality-recommendations/artifacts/"
        "catalog-vercel-tier-evidence-v9.json",
        encoding="utf-8",
    ))
    state = json.load(open(
        ".goals/human-quality-recommendations/"
        "protocol-v9-powered-development-r2/state.json",
        encoding="utf-8",
    ))
    assert tier["project_tier"] == "unknown"
    assert tier["actual_memory_limit_bytes"] is None
    assert tier["tier_verified"] is False
    assert tier["passed"] is False
    assert tier["credentials_or_token_values_recorded"] is False
    assert state["phase"] == "DEVELOPMENT_SCORER_LOCKED"
    assert state["final_open_count"] == 0
    assert state["fresh_final_blocked"] is True



def test_snapshot_validation_rejects_swapped_or_tampered_evidence():
    snapshots = json.load(
        open("benchmarks/evidence/v9/music-map.normalized.json", encoding="utf-8")
    )
    seeds = load_seed_specs(
        "benchmarks/soundalike_pairs.v6.json",
        ".goals/human-quality-recommendations/artifacts/"
        "catalog-gated-direct-seeds-v8.json",
    )
    assert len(validate_music_map_snapshots(seeds, snapshots)) == 60
    tampered = json.loads(json.dumps(snapshots))
    tampered["records"][0]["source_url"] = "https://example.invalid/swapped"
    with pytest.raises(GoldBuildError):
        validate_music_map_snapshots(seeds, tampered)


def test_missing_vercel_cli_is_recorded_instead_of_crashing():
    result = _safe_cli(["this-command-does-not-exist-v9"])
    assert result["status"] == "failed"
    assert result["returncode"] is None
