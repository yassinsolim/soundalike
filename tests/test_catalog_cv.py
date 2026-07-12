from pathlib import Path

import pytest

from soundalike.ml.catalog_cv import (
    DEFAULT_POLICY_GRID,
    apply_cached_policy,
    candidate_recall,
    nested_cross_validate,
    normalize_opened_benchmarks,
    paired_bootstrap,
    relative_change,
    scene_held_out_folds,
    scene_held_out_validate,
    sonic_primary,
)
from soundalike.ml.catalog_policy import CatalogPolicy


def test_real_v6_filter_is_exact_and_v7_never_becomes_a_label():
    result = normalize_opened_benchmarks(
        Path("benchmarks/soundalike_pairs.v6.json"),
        Path("benchmarks/soundalike_multipositive.v7.json"),
    )
    assert len(result["records"]) == 65
    inventory = result["benchmark_inventory"]
    assert inventory["v6_total_records"] == 195
    assert inventory["v6_excluded"] == 130
    assert inventory["supporting_only"]["soundalike_multipositive.v7"] == 100
    assert not inventory["eligible_sonic_multipositive_source_exists"]
    assert not inventory["deezer_used_for_selection"]
    assert {record["source_version"] for record in result["records"]} == {6}
    assert all(
        positive["audible_comparison_provenance_verified"]
        and positive["sources"][0]["excerpt"]
        for record in result["records"]
        for positive in record["positives"]
    )


def test_grid_has_only_three_predeclared_numeric_fields():
    assert len(DEFAULT_POLICY_GRID) == 27
    assert set(DEFAULT_POLICY_GRID[0].__dict__) == {"tau", "sigma", "audio_weight"}
    assert {policy.tau for policy in DEFAULT_POLICY_GRID} == {0.35, 0.50, 0.65}


def test_cached_gate_abstains_to_exact_production_and_firing_uses_full_union():
    cached = {
        "production_rows": [8, 9, 10],
        "graph_union_rows": [1, 2, 3, 4, 5, 6],
        "components": [
            {"row": row, "G": 1-row/100, "A": 0.9, "S": 0.9}
            for row in range(1, 7)
        ],
        "gate_components": {
            "source_coverage": {"lastfm": True, "music4all": True},
            "shared_count": 7,
            "agreement": 0.8,
        },
    }
    abstain = apply_cached_policy(cached, CatalogPolicy(0.9, 0.5, 0.1), 3)
    assert not abstain["fired"]
    assert abstain["ranking_rows"] == cached["production_rows"]
    assert abstain["candidate_rows"] == cached["production_rows"]
    fired = apply_cached_policy(cached, CatalogPolicy(0.5, 0.5, 0.1), 3)
    assert fired["fired"]
    assert fired["candidate_rows"] == [1, 2, 3, 4, 5, 6, 8, 9, 10]


def test_sonic_primary_candidate_recall_bootstrap_and_zero_baseline_are_exact():
    assert sonic_primary(0.5, 1.0, 0.0) == pytest.approx(0.5)
    relevance = {2: ("a", 3), 3: ("b", 1), 4: ("b", 1)}
    assert candidate_recall([9, 2, 4], relevance, 3) == 1.0
    assert relative_change(0, 0) == 0
    assert relative_change(0, 0.1) == float("inf")
    first = paired_bootstrap([0.1] * 20)
    assert first == paired_bootstrap([0.1] * 20)
    assert first["ci95_low"] > 0 and first["p_positive"] == 1


def test_scene_folds_have_complete_scene_isolation():
    records = [
        {"id": "a1", "scene": "a"},
        {"id": "a2", "scene": "a"},
        {"id": "b1", "scene": "b"},
    ]
    for fold in scene_held_out_folds(records):
        assert {record["scene"] for record in fold["test"]} == {fold["scene"]}
        assert fold["scene"] not in {record["scene"] for record in fold["train"]}


def test_nested_selection_never_receives_outer_labels():
    records = [
        {"id": str(index), "scene": str(index % 2), "label": index}
        for index in range(10)
    ]
    calls = []

    def evaluate(policy, train, validation):
        calls.append(({row["id"] for row in train}, {row["id"] for row in validation}))
        return {
            "sonic_primary": policy.audio_weight,
            "mrr_at_10": 0.0,
            "recall_at_10": 0.0,
            "ndcg_at_10": 0.0,
        }

    policy = CatalogPolicy(0.2, 0.3, 0.4)
    report = nested_cross_validate(records, evaluate, policies=(policy,))
    assert report["final_policy"]["audio_weight"] == 0.4
    offset = 0
    for fold in report["folds"]:
        allowed = set(fold["train_ids"])
        for train_ids, validation_ids in calls[offset:offset + 5]:
            assert train_ids | validation_ids <= allowed
        offset += 6  # five inner validation calls plus one outer evaluation


def test_scene_inner_selection_never_sees_held_scene():
    records = [
        {"id": f"{scene}{index}", "scene": scene}
        for scene in ("a", "b", "c") for index in range(5)
    ]
    calls = []

    def evaluate(_policy, train, validation):
        calls.append((list(train), list(validation)))
        return {"sonic_primary": 0.5, "mrr_at_10": 0.5,
                "recall_at_10": 0.5, "ndcg_at_10": 0.5}

    report = scene_held_out_validate(
        records, evaluate, policies=(CatalogPolicy(0.2, 0.3, 0.1),)
    )
    for fold in report["folds"]:
        assert fold["scene"] not in {
            row["scene"] for row in records if row["id"] in fold["train_ids"]
        }
