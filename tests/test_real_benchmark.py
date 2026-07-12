"""Regression gates for the frozen 272,853-song pure-sonic benchmark."""

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
    PairResolver,
)
from soundalike.ml.related_artists_rerank import MANUAL_PAIRS

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "benchmarks" / "soundalike_pairs.v4.json"
JUDGMENTS = ROOT / "benchmarks" / "heldout_top5_judgments.v4.json"
ARTIFACTS = ROOT / ".goals" / "human-quality-recommendations" / "artifacts"
BASELINE = ARTIFACTS / "production-baseline-final-v4.json"
WINNER = ARTIFACTS / "held-out-final-winner-v4.json"
ALL_WINNER = WINNER
CHALLENGERS = ARTIFACTS / "representation-challengers-final-v4.json"
EXTERNAL = ARTIFACTS / "external-validation-final-v4.json"
RESOURCES = ARTIFACTS / "resource-metrics-final-v4.json"


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_benchmark_categorizes_every_sourced_pair():
    benchmark = load_benchmark(BENCHMARK)
    assert benchmark["benchmark_version"] == "4.0.0"
    assert benchmark["frozen_at"] == "2026-07-11"
    assert len(benchmark["pairs"]) == 93
    assert len({pair["scene"] for pair in benchmark["pairs"]}) >= 12
    allowed = set(benchmark["category_definitions"])
    assert allowed == {
        "pure_sonic",
        "sample_interpolation",
        "legal_plagiarism",
        "cover_remix",
        "weak_unsubstantiated",
    }
    for pair in benchmark["pairs"]:
        assert pair["split"] in {"development", "validation", "held_out"}
        assert pair["evidence_category"] in allowed
        assert pair["deciding_primary"] == (
            pair["evidence_category"] == "pure_sonic"
        )
        assert pair["sources"]
        for source in pair["sources"]:
            assert source["url"].startswith(("http://", "https://"))
            assert source["retrieved"] == "2026-07-11"
            assert len(source["context"]) >= 30


def test_deciding_subset_excludes_diagnostic_relationships():
    benchmark = load_benchmark(BENCHMARK)
    pure = [pair for pair in benchmark["pairs"] if pair["deciding_primary"]]
    assert sum(pair["split"] == "development" for pair in pure) == 7
    assert sum(pair["split"] == "held_out" for pair in pure) == 20
    assert all(pair["evidence_category"] == "pure_sonic" for pair in pure)
    for pair in pure:
        publishers = " ".join(
            source["publisher"].casefold() for source in pair["sources"]
        )
        assert "sounds just like" not in publishers
        assert publishers.strip() != "watchmojo"


def test_artist_pair_and_transitive_graph_split_has_no_leakage():
    benchmark = load_benchmark(BENCHMARK)
    audit = audit_leakage(benchmark, MANUAL_PAIRS)
    assert audit["passed"], audit
    assert audit["held_out_pair_count"] == 20
    assert audit["transitive_graph_overlap"] == []
    assert MANUAL_PAIRS == []

    dev_artist = next(
        pair["query"]["artist"]
        for pair in benchmark["pairs"]
        if pair["split"] == "development"
    )
    held_artist = next(
        pair["query"]["artist"]
        for pair in benchmark["pairs"]
        if pair["split"] == "held_out"
    )
    leaked = audit_leakage(
        benchmark,
        graph_edges=[(dev_artist, "Graph Bridge"), ("Graph Bridge", held_artist)],
    )
    assert not leaked["passed"]
    assert leaked["transitive_graph_overlap"]


def test_featured_artists_are_in_leakage_audit():
    assert credited_artists("Bad Bunny, Jowell & Randy feat. Ñengo Flow") == {
        "bad bunny", "jowell", "randy", "nengo flow"
    }


def test_target_resolution_rejects_derivative_substitutes():
    resolver = PairResolver(
        ["Physical (Remix)", "Physical (Live)", "Physical"],
        ["Dua Lipa", "Dua Lipa", "Different Artist"],
    )
    target = {"title": "Physical", "artist": "Dua Lipa"}
    assert resolver.rows(target) == [0, 1]
    assert resolver.target_rows(target) == []


def test_frozen_baseline_is_real_index_and_pure_only():
    baseline = _json(BASELINE)
    assert baseline["index"]["tracks"] == 272_853
    assert baseline["index"]["sha256"] == (
        "89bfde6f622619a704462291b17f82bcb6508880210932b0a1548a433e1b7085"
    )
    assert baseline["evidence_category"] == "pure_sonic"
    assert baseline["selection"] == {
        "selected_pairs": 20,
        "split_pairs": 20,
        "diagnostic_pairs_excluded": 0,
    }
    method = baseline["methods"]["production_baseline"]
    assert len(method["pairs"]) == 20
    assert all(len(pair["ranked_outputs"]) == 50 for pair in method["pairs"])
    assert method["metrics"]["primary_score"] == 0.028125
    assert method["metrics"]["n_rankable"] == 20


def test_materially_different_challengers_ran_on_real_index():
    artifact = _json(CHALLENGERS)
    assert artifact["corpus"]["rows"] == 272_853
    methods = artifact["final_held_out_challengers"]
    assert {
        "dual_sonic_without_priors",
        "pageview_heavy_learned_reranker",
        "source_notability_continuous",
        "specific_song_page_binary",
        "dual_sonic_guardrail",
    } <= set(methods)
    assert artifact["selection_policy"]["manual_judgments_blended"] is False
    builds = artifact["representation_builds"]
    assert {"laion_clap_calibrated_pca64", "panns_cnn14_calibrated",
            "efficientnet_multivector", "chroma_fft_dsp"} <= set(builds)
    assert len(artifact["negative_results"]) >= 7


def test_winner_passes_pure_retrieval_gate_without_manual_blend():
    artifact = _json(WINNER)
    baseline = artifact["methods"]["production_baseline"]["metrics"]
    winner = artifact["methods"]["dual_sonic"]["metrics"]
    comparison = artifact["comparisons_to_production_baseline"]["dual_sonic"]
    assert artifact["evidence_category"] == "pure_sonic"
    assert baseline["primary_score"] == 0.028125
    assert winner["primary_score"] == 0.05294840294840295
    assert comparison["relative_gain"] >= 0.20
    assert comparison["relative_gain"] > 0.88
    assert comparison["passes_scene_guardrail"]
    # The pair bootstrap is descriptive because sequential challengers reused
    # the held-out suite. Keep the negative lower bound visible.
    assert comparison["ci95_low"] < 0 < comparison["ci95_high"]
    assert winner["recall_at_50"] > baseline["recall_at_50"]


def test_all_20_final_top_fives_are_bound_to_direct_judgments():
    judgments = _json(JUDGMENTS)
    rows = judgments["judgments"]
    assert len(rows) == 20
    assert len({row["pair_id"] for row in rows}) == 20
    assert sum(row["winner_pass"] for row in rows) == 17
    assert all(row["reason"] and len(row["top5"]) == 5 for row in rows)
    fingerprint = judged_top5_fingerprint(
        _json(ALL_WINNER), ("dual_sonic",)
    )
    assert judgments["ranked_top5_fingerprint"] == fingerprint
    assert "never blended" in judgments["methodology"]["role"]
    assert judgments["retained_v2_ux_validation"]["winner_passes"] == 17


def test_paired_bootstrap_aligns_by_pair_id():
    report = _json(WINNER)["methods"]["production_baseline"]
    reordered = {**report, "pairs": list(reversed(report["pairs"]))}
    result = bootstrap_delta(report, reordered, iterations=500)
    assert result["absolute_delta"] == 0
    assert result["ci95_low"] == result["ci95_high"] == 0


def test_winner_top_five_has_no_junk_or_seed_title_variants():
    method = _json(ALL_WINNER)["methods"]["dual_sonic"]
    quality = TitleQualityFilter()
    for pair in method["pairs"]:
        seen = set()
        for result in pair["ranked_outputs"][:5]:
            assert not quality.is_junk(result["title"], result["artist"])
            assert not quality.seed_title_in_result(
                pair["query_catalogue"]["title"], result["title"]
            )
            key = (result["title"].casefold(), result["artist"].casefold())
            assert key not in seen
            seen.add(key)


def test_external_validation_is_disjoint_and_equivalent_or_better():
    artifact = _json(EXTERNAL)
    assert artifact["benchmark_artist_overlap"] == []
    assert artifact["index_tracks"] == 272_853
    assert artifact["winner_method"] == "dual_sonic"
    for source in ("listenbrainz", "deezer"):
        comparison = artifact["comparisons"][source]
        assert comparison["winner_mean"] >= comparison["baseline_mean"]
        assert comparison["ci95_low"] >= -0.05


def test_measured_resources_fit_hosted_limits():
    artifact = _json(RESOURCES)
    index = artifact["release_index"]
    assert index["cold_load_seconds"] < 10
    assert index["file_bytes"] == 299_288_526
    assert index["rss_after_bytes"] < 2 * 1024 * 1024 * 1024
    latency = artifact["local_recommendation_ms"]
    assert latency["p95"] < 500
