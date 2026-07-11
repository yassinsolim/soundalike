"""Regression gates for the frozen 272,853-song real-world benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from soundalike.ml.quality_filter import TitleQualityFilter
from soundalike.ml.real_benchmark import (
    audit_leakage,
    bootstrap_delta,
    credited_artists,
    judged_top5_fingerprint,
    load_benchmark,
)
from soundalike.ml.related_artists_rerank import MANUAL_PAIRS

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "soundalike_pairs.v1.json"
JUDGMENTS = ROOT / "benchmarks" / "heldout_top5_judgments.v1.json"
ARTIFACTS = ROOT / ".goals" / "human-quality-recommendations" / "artifacts"
ALL_METHODS = ARTIFACTS / "held-out-all-methods-v1.json"
WINNER = ARTIFACTS / "held-out-winner-v1.json"
BASELINE = ARTIFACTS / "production-baseline-held-out-v1.json"
EXTERNAL = ARTIFACTS / "external-validation-v1.json"
LIVE_PARITY = ARTIFACTS / "live-baseline-parity-v1.json"
LIVE_BROWSER = ARTIFACTS / "live-browser-10-seeds-v1.json"


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_benchmark_is_sourced_versioned_and_diverse():
    benchmark = load_benchmark(BENCHMARK)
    assert benchmark["benchmark_version"] == "1.0.0"
    assert benchmark["frozen_at"] == "2026-07-11"
    assert len(benchmark["pairs"]) == 50
    assert len({pair["scene"] for pair in benchmark["pairs"]}) >= 12
    for pair in benchmark["pairs"]:
        assert pair["split"] in {"development", "held_out"}
        assert pair["sources"]
        for source in pair["sources"]:
            assert source["url"].startswith(("http://", "https://"))
            assert source["retrieved"] == "2026-07-11"
            assert len(source["context"]) >= 30


def test_artist_and_pair_split_has_no_leakage():
    benchmark = load_benchmark(BENCHMARK)
    audit = audit_leakage(benchmark, MANUAL_PAIRS)
    assert audit["passed"], audit
    assert audit["held_out_pair_count"] == 20
    assert MANUAL_PAIRS == []


def test_featured_artists_are_in_leakage_audit():
    assert credited_artists("Bad Bunny, Jowell & Randy feat. Ñengo Flow") == {
        "bad bunny", "jowell", "randy", "nengo flow"
    }


def test_frozen_baseline_is_the_actual_production_index():
    baseline = _json(BASELINE)
    assert baseline["index"]["tracks"] == 272_853
    assert baseline["index"]["sha256"] == (
        "89bfde6f622619a704462291b17f82bcb6508880210932b0a1548a433e1b7085"
    )
    method = baseline["frozen_method"]
    assert method["method"] == "production_baseline"
    assert len(method["pairs"]) == 20
    assert all(len(pair["ranked_outputs"]) == 50 for pair in method["pairs"])
    live = _json(LIVE_PARITY)
    assert live["matched"] == live["total"] == 10
    winner_baseline = _json(WINNER)["methods"]["production_baseline"]
    for frozen, current in zip(method["pairs"], winner_baseline["pairs"]):
        assert frozen["pair_id"] == current["pair_id"]
        assert frozen["query_catalogue"]["row"] == current["query_catalogue"]["row"]
        assert [row["row"] for row in frozen["ranked_outputs"]] == [
            row["row"] for row in current["ranked_outputs"]
        ]
    browser_rows = _json(LIVE_BROWSER)["rows"]
    by_id = {pair["pair_id"]: pair for pair in winner_baseline["pairs"]}
    for pair_id, browser in zip(live["pair_ids"], browser_rows):
        assert [row["title"] for row in by_id[pair_id]["ranked_outputs"][:5]] == [
            row["title"] for row in browser["body"]["results"]
        ]


def test_raw_encoder_report_has_required_retrieval_metrics():
    artifact = _json(ALL_METHODS)
    raw = artifact["methods"]["raw_encoder"]
    for name in (
        "recall_at_1", "recall_at_5", "recall_at_10", "recall_at_20",
        "recall_at_50", "mrr", "ndcg_at_50", "missing_catalogue_rate",
        "reciprocal_rank_distribution",
    ):
        assert name in raw["metrics"]
    assert len(raw["metrics"]["reciprocal_rank_distribution"]) == 20
    assert all(len(pair["ranked_outputs"]) == 50 for pair in raw["pairs"])


def test_three_materially_different_approaches_ran_on_production():
    artifact = _json(ALL_METHODS)
    assert artifact["index"]["tracks"] == 272_853
    methods = artifact["methods"]
    # Filtering, artist-centroid geometry, and unsupervised hubness correction
    # are independent interventions; query expansion is a fourth negative result.
    assert {"quality_filter", "artist_centroid", "hubness", "query_expansion"} <= set(methods)
    assert all(methods[name]["metrics"]["n_pairs"] == 20 for name in methods)


def test_winner_passes_human_aligned_gate_with_bootstrap():
    artifact = _json(WINNER)
    primary = artifact["human_aligned_primary"]
    assert primary["relative_gain"] >= 0.20
    assert primary["ci95_absolute_low"] > 0
    assert primary["passes_scene_guardrail"]
    assert primary["passes_80pct_direct"]
    assert primary["held_out_seeds_passing"] >= 16
    # Guarded reranking must preserve baseline known-pair Recall@50.
    baseline = artifact["methods"]["production_baseline"]["metrics"]
    winner = artifact["methods"]["guarded_centroid"]["metrics"]
    assert winner["recall_at_50"] >= baseline["recall_at_50"]


def test_all_20_top_fives_have_explicit_judgments():
    judgments = _json(JUDGMENTS)
    rows = judgments["judgments"]
    assert len(rows) == 20
    assert len({row["pair_id"] for row in rows}) == 20
    assert sum(row["winner_pass"] for row in rows) == 17
    assert all(row["baseline_reason"] and row["winner_reason"] for row in rows)
    assert judgments["ranked_top5_fingerprint"] == judged_top5_fingerprint(_json(WINNER))


def test_paired_bootstrap_aligns_by_pair_id():
    report = _json(WINNER)["methods"]["production_baseline"]
    reordered = {**report, "pairs": list(reversed(report["pairs"]))}
    result = bootstrap_delta(report, reordered, iterations=500)
    assert result["absolute_delta"] == 0
    assert result["ci95_low"] == result["ci95_high"] == 0


def test_winner_top_five_has_no_junk_or_seed_title_variants():
    artifact = _json(WINNER)
    method = artifact["methods"]["guarded_centroid"]
    quality = TitleQualityFilter()
    seen = set()
    for pair in method["pairs"]:
        top = pair["ranked_outputs"][:5]
        for result in top:
            assert not quality.is_junk(result["title"], result["artist"])
            assert not quality.seed_title_in_result(
                pair["query_catalogue"]["title"], result["title"]
            )
            key = (pair["pair_id"], result["title"].casefold(), result["artist"].casefold())
            assert key not in seen
            seen.add(key)


def test_external_validation_is_disjoint_and_non_regressing():
    artifact = _json(EXTERNAL)
    assert artifact["benchmark_artist_overlap"] == []
    assert artifact["index_tracks"] == 272_853
    for source in ("listenbrainz", "deezer"):
        comparison = artifact["comparisons"][source]
        assert comparison["winner_mean"] >= comparison["baseline_mean"]
        assert comparison["ci95_low"] >= -0.02


def test_measured_resources_fit_hosted_limits():
    resources = _json(WINNER)["resources"]
    assert resources["cold_load_seconds"] < 10
    assert resources["minimum_ranker_array_bytes"] < 512 * 1024 * 1024
    assert resources["reranker_bytes"] < 32 * 1024 * 1024
    assert resources["measured_reranker_rss_delta_bytes"] < 64 * 1024 * 1024
    latency = _json(WINNER)["methods"]["guarded_centroid"]["latency"]
    assert latency["p95_ms"] < 500
