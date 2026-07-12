import json
from pathlib import Path

import pytest

from soundalike.ml.catalog_cv import (
    candidate_recall,
    composite_primary,
    gate_scene_fold,
    nested_cross_validate,
    normalize_opened_benchmarks,
    relative_change,
    rescore_components,
    scene_held_out_folds,
    select_policy,
)
from soundalike.ml.catalog_policy import GRAPH_ONLY_POLICY, CatalogPolicy


def test_inventory_uses_v6_as_superset_and_all_opened_v7():
    result = normalize_opened_benchmarks(
        Path("benchmarks/soundalike_pairs.v6.json"),
        Path("benchmarks/soundalike_multipositive.v7.json"),
    )
    assert len(result["records"]) == 295
    inventory = result["benchmark_inventory"]
    assert inventory["all_v7_opened_records_included"]
    assert "superseded" in inventory["excluded_or_superseded"][
        "soundalike_pairs.v1-v5"
    ]
    v6 = next(record for record in result["records"] if record["source_version"] == 6)
    assert v6["positives"][0]["grade"] == 3
    assert v6["positives"][0]["relevance_scope"] == "track"
    v7 = next(record for record in result["records"] if record["source_version"] == 7)
    assert v7["positives"] == v7["source_record"]["positives"]
    assert set(inventory["axis_counts"]) == {"sonic_editorial", "taste_affinity"}


def test_cached_components_adapt_to_policy_without_retrieval():
    components = [
        {"row": 1, "G": 0.8, "A": 0.0, "S": 0.0},
        {"row": 2, "G": 0.1, "A": 1.0, "S": 1.0},
    ]
    assert rescore_components(components, GRAPH_ONLY_POLICY)[0]["row"] == 1
    assert rescore_components(components, CatalogPolicy(1.0, 1.0, 0.0))[0]["row"] == 2


def test_composite_candidate_recall_and_zero_baseline_are_exact():
    assert composite_primary(0.5, 1.0) == pytest.approx(0.6)
    relevance = {2: ("a", 3), 3: ("b", 1), 4: ("b", 1)}
    assert candidate_recall([9, 2, 4], relevance, 3) == 1.0
    assert relative_change(0, 0) == 0
    assert relative_change(0, 0.1) == float("inf")


def test_scene_folds_have_complete_scene_isolation():
    records = [
        {"id": "a1", "scene": "a"},
        {"id": "a2", "scene": "a"},
        {"id": "b1", "scene": "b"},
    ]
    for fold in scene_held_out_folds(records):
        assert {record["scene"] for record in fold["test"]} == {fold["scene"]}
        assert fold["scene"] not in {record["scene"] for record in fold["train"]}


def test_policy_choice_is_deterministic_on_ties():
    a = CatalogPolicy(0.2, 0.0, 0.0)
    b = CatalogPolicy(0.1, 0.0, 0.0)
    scores = {a: [0.5], b: [0.5]}
    assert select_policy(scores) == b
    assert select_policy(dict(reversed(list(scores.items())))) == b


def test_nested_selection_never_receives_outer_labels():
    records = [
        {"id": str(index), "scene": str(index % 2), "label": index}
        for index in range(10)
    ]
    calls = []

    def evaluate(policy, train, validation):
        calls.append(({row["id"] for row in train}, {row["id"] for row in validation}))
        return {
            "composite_primary": policy.audio_weight,
            "mrr_at_10": 0.0,
            "recall_at_10": 0.0,
        }

    policy = CatalogPolicy(0.2, 0.0, 0.0)
    report = nested_cross_validate(records, evaluate, policies=(policy,))
    assert report["final_policy"]["audio_weight"] == 0.2
    # For every outer fold, all selection calls before its outer evaluation are
    # subsets of that fold's training ids.
    offset = 0
    for fold in report["folds"]:
        inner_call_count = 5 * 2  # graph-only plus supplied policy
        allowed = set(fold["train_ids"])
        for train_ids, validation_ids in calls[offset:offset + inner_call_count]:
            assert train_ids | validation_ids <= allowed
        offset += inner_call_count + 1


def test_gate_fails_when_any_scene_breaches_floor():
    baseline = {"composite_primary": 0.5, "mrr_at_10": 0.4, "recall_at_10": 0.3}
    challenger = {"composite_primary": 0.6, "mrr_at_10": 0.4, "recall_at_10": 0.3}
    passed = gate_scene_fold(baseline, challenger, 0.4, 0.5, {"a": (0.5, 0.45)})
    assert passed["gate_pass"]
    failed = gate_scene_fold(
        baseline, challenger, 0.4, 0.5, {"a": (0.5, 0.449999)}
    )
    assert not failed["passes"]["every_scene_above_minus_10_percent"]
    assert not failed["gate_pass"]
