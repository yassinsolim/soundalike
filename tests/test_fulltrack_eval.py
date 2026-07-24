import copy
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from soundalike.ml.fulltrack_eval import (
    BENCHMARK_SCHEMA_VERSION,
    COMMERCIAL_EVIDENCE_SCOPE,
    DATASET_DESCRIPTION,
    EVALUATION_SCHEMA_VERSION,
    GROUPED_METRICS_NOTICE,
    LAWFUL_USE,
    EvaluationConfig,
    FullTrackEvaluationError,
    METHODS,
    METRICS,
    _benchmark_result_binding,
    _evaluation_protocol,
    _evaluation_store_binding,
    _grouped_metrics,
    _load_valid_benchmark_result,
    _method_ranking,
    _paired_bootstrap_delta,
    _query_metrics,
    _query_descriptor_sha256,
    _scene_for_tag,
    _score_trained_candidate_pool,
    _trained_method_id,
    _TrainedModelBinding,
    _write_benchmark_result,
    aggregate_all_fold_results,
    build_parser,
    evaluate_jamendo,
    fixed_budget_maxsim,
    freeze_ranked_section_budget,
    hybrid_score,
    load_commercial_v6_replay,
    load_trained_model_for_fold,
    run_all_folds_benchmark,
    write_evaluation_report,
)
from soundalike.ml.fulltrack_fusion import (
    FEATURE_DIM,
    FusionConfig,
    FusionModel,
    build_nonneg_linear,
)
from soundalike.ml.fulltrack_selection import (
    FullTrackSelectionError,
    build_selection_inputs,
    write_selection_inputs,
)
from soundalike.ml.fulltrack_store import (
    FullTrackStore,
    FullTrackStoreReader,
    TrackArtifacts,
    stable_json_sha256,
)
from soundalike.ml.jamendo_fulltrack import (
    EVIDENCE_SCOPE,
    ArtistFold,
    JamendoContext,
    JamendoTrack,
    TrackLicense,
)


HASH = hashlib.sha256(b"fixture").hexdigest()


def test_trained_pool_shares_pair_features_without_changing_scores(monkeypatch):
    def track(offset):
        windows = np.asarray(
            [[1.0, offset], [offset, 1.0]], dtype=np.float32
        )
        windows /= np.linalg.norm(windows, axis=1, keepdims=True)
        return SimpleNamespace(
            global_embedding=windows.mean(axis=0),
            window_embeddings=windows,
            repeated_sections=windows,
            salient_sections=windows[::-1],
            repeated_indices=np.asarray([0, 1], dtype=np.int64),
            salient_indices=np.asarray([1, 0], dtype=np.int64),
        )

    config = FusionConfig(
        kind="nonnegative_linear",
        embedding_dim=2,
        maxsim_budget=2,
        model_id="shared-feature-test",
        store_id="test-store",
        config_sha256=HASH,
    )
    first = build_nonneg_linear(np.ones(FEATURE_DIM), config)
    second = build_nonneg_linear(
        np.arange(1, FEATURE_DIM + 1, dtype=np.float64), config
    )
    query = track(0.1)
    candidates = [track(0.2), track(0.3), track(0.4)]
    expected = {
        "first": first.score_candidates(query, candidates),
        "second": second.score_candidates(query, candidates),
    }
    calls = 0
    original = FusionModel.extract_pair_features

    def counted_extract(model, query_track, candidate_track):
        nonlocal calls
        calls += 1
        return original(model, query_track, candidate_track)

    monkeypatch.setattr(FusionModel, "extract_pair_features", counted_extract)
    bindings = {
        "first": (SimpleNamespace(model=first), "none"),
        "second": (SimpleNamespace(model=second), "none"),
    }
    actual, latencies = _score_trained_candidate_pool(
        query, candidates, bindings
    )

    assert calls == len(candidates)
    assert set(latencies) == set(bindings)
    for method_id in bindings:
        np.testing.assert_array_equal(actual[method_id], expected[method_id])


def test_fixed_budget_maxsim_prevents_candidate_length_advantage():
    query = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    short = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    long = np.tile(short, (100, 1))
    assert fixed_budget_maxsim(query, short, budget=8) == pytest.approx(
        fixed_budget_maxsim(query, long, budget=8)
    )
    assert hybrid_score(0.2, 0.4, 0.6) == pytest.approx(0.35)


def test_ranked_section_budget_uses_prefix_and_repeats_only_when_short():
    sections = np.eye(4, dtype=np.float32)
    np.testing.assert_array_equal(
        freeze_ranked_section_budget(sections, 2), sections[:2]
    )
    repeated = freeze_ranked_section_budget(sections[:2], 4)
    assert repeated.shape == (4, 4)
    assert {tuple(row) for row in repeated} == {tuple(row) for row in sections[:2]}


def test_reranker_ties_preserve_frozen_global_order():
    ranking = _method_ranking(
        np.asarray([0.5, 0.5], dtype=np.float32),
        np.asarray([8, 3], dtype=np.int64),
        np.asarray([8, 3, 5], dtype=np.int64),
    )
    assert ranking.tolist() == [8, 3, 5]


def test_mrr_uses_complete_ranking_when_first_relevant_is_rank_11():
    ranking = list(range(1, 11)) + [99, 100]
    metrics = _query_metrics(
        ranking,
        {99: 1.0},
        recall_cutoff=10,
        ndcg_cutoff=10,
    )
    assert metrics.recall_at_k == 0.0
    assert metrics.mrr == pytest.approx(1.0 / 11.0)
    assert metrics.graded_ndcg_at_k == 0.0


def test_commercial_replay_is_strictly_labelled_read_only_and_outside_signed_state(
    tmp_path,
):
    replay = tmp_path / "commercial-v6-replay.json"
    replay.write_text(
        json.dumps(
            {
                "evidence_scope": COMMERCIAL_EVIDENCE_SCOPE,
                "benchmark_version": "v6",
                "metrics": {},
            }
        ),
        encoding="utf-8",
    )
    before = replay.read_bytes()
    loaded = load_commercial_v6_replay(replay)
    assert loaded["evidence_scope"] == COMMERCIAL_EVIDENCE_SCOPE
    assert replay.read_bytes() == before

    protected = tmp_path / ".goals" / "x" / "protocol-v6" / "state.json"
    with pytest.raises(FullTrackEvaluationError, match="must never be opened"):
        load_commercial_v6_replay(protected)
    with pytest.raises(FullTrackEvaluationError, match="invalid evidence"):
        write_evaluation_report(tmp_path / "bad.json", {"evidence_scope": "jamendo"})
    with pytest.raises(FullTrackEvaluationError, match="protected"):
        write_evaluation_report(protected, {"evidence_scope": EVIDENCE_SCOPE})
    assert not protected.exists()

    protected_parent = tmp_path / ".goals" / "x" / "protocol-v6"
    protected_parent.mkdir(parents=True)
    alias = tmp_path / "innocent-output"
    try:
        alias.symlink_to(protected_parent, target_is_directory=True)
    except OSError:
        pass
    else:
        with pytest.raises(FullTrackEvaluationError, match="protected"):
            write_evaluation_report(
                alias / "report.json", {"evidence_scope": EVIDENCE_SCOPE}
            )
        assert not (protected_parent / "report.json").exists()


def _track(index: int, tag: str) -> JamendoTrack:
    relative = f"{index:02d}/{index}.mp3"
    return JamendoTrack(
        row_index=index,
        track_id=100 + index,
        artist_id=200 + index,
        album_id=300 + index,
        relative_path=relative,
        audio_path=Path(relative),
        duration_seconds=60.0,
        tags=(tag,),
        title=f"Track {index}",
        artist_name=f"Artist {index}",
        album_name="Fixture",
        release_date="2026",
        jamendo_url=f"http://www.jamendo.com/track/{100 + index}",
        license=TrackLicense(
            path=relative,
            attribution="fixture",
            name="CC",
            url="http://creativecommons.org/licenses/by-nc-sa/3.0/",
            permits_commercial_use=False,
            permits_derivatives=True,
        ),
        expected_audio_sha256=HASH,
        expected_audio_bytes=1,
        fold_parts=("test",),
    )


def _unit(index: int) -> np.ndarray:
    value = np.zeros(4, dtype=np.float32)
    value[index] = 1
    return value


def test_artist_disjoint_evaluation_reports_baseline_and_trained_metrics(tmp_path):
    tracks = (
        _track(0, "genre---rock"),
        _track(1, "genre---rock"),
        _track(2, "genre---jazz"),
        _track(3, "genre---jazz"),
    )
    fold = ArtistFold(
        index=0,
        track_parts=MappingProxyType({track.track_id: "test" for track in tracks}),
        artist_parts=MappingProxyType({track.artist_id: "test" for track in tracks}),
        track_tags=MappingProxyType({track.track_id: track.tags for track in tracks}),
        tags=("genre---jazz", "genre---rock"),
    )
    context = JamendoContext(
        tracks=tracks,
        folds=(fold,),
        metadata_root=tmp_path,
        audio_root=tmp_path,
        state_root=tmp_path,
        metadata_commit="fixture",
        archive_manifest_sha256=HASH,
        track_manifest_sha256=HASH,
        metadata_hashes=MappingProxyType({}),
        source_fingerprint=HASH,
    )
    root = tmp_path / "store"
    with FullTrackStore(
        root,
        track_ids=[track.track_id for track in tracks],
        source_hashes=[HASH] * 4,
        source_fingerprint=HASH,
        config_sha256=HASH,
        model_sha256=HASH,
        model_id="fake",
        embedding_dim=4,
        shard_tracks=2,
        repetition_sections=32,
        salient_sections=32,
    ) as store:
        for index, track in enumerate(tracks):
            vector = _unit(index // 2)
            other = _unit((index // 2 + 1) % 4)
            window_count = 3 if index == 0 else 40
            windows = np.stack(
                [vector if position % 4 else other for position in range(window_count)]
            )
            sections = windows[: min(32, window_count)]
            store.write_track(
                track.track_id,
                HASH,
                TrackArtifacts(
                    global_embedding=vector,
                    window_embeddings=windows,
                    window_starts=np.arange(window_count, dtype=np.int64) * 5,
                    repeated_sections=sections,
                    salient_sections=sections,
                    repeated_indices=np.arange(len(sections), dtype=np.int64),
                    salient_indices=np.arange(len(sections), dtype=np.int64),
                    decoded_samples=window_count * 5 + 10,
                ),
            )
        store.seal()
    reports = []
    with FullTrackStoreReader(root) as reader:
        for budget in (8, 16, 32):
            reports.append(
                evaluate_jamendo(
                    context,
                    reader,
                    config=EvaluationConfig(
                        maxsim_budget=budget,
                        candidate_pool=3,
                        bootstrap_iterations=50,
                        max_feature_cache_bytes=1024 * 1024,
                        min_shared_tags=1,
                        min_tag_jaccard=1.0,
                    ),
                )
            )
        store_binding_sha256 = stable_json_sha256(_evaluation_store_binding(reader))
        trained_model = build_nonneg_linear(
            np.ones(FEATURE_DIM, dtype=np.float64),
            FusionConfig(
                kind="nonnegative_linear",
                embedding_dim=4,
                maxsim_budget=8,
                seed=17,
                model_id="fixture-trained",
                store_id=store_binding_sha256,
                config_sha256=HASH,
                fold_index=0,
            ),
        )
        trained_binding = _TrainedModelBinding(
            candidate_kind="nonnegative_linear",
            seed=17,
            fold_index=0,
            model=trained_model,
            report_sha256=HASH,
            model_artifact_sha256=HASH,
            model_json_sha256=HASH,
            weights_npz_sha256=HASH,
            source_fingerprint=HASH,
            store_binding_sha256=store_binding_sha256,
            training_config_sha256=HASH,
            job_config_sha256=HASH,
            fusion_metadata={},
            maxsim_budget=8,
            embedding_dim=4,
        )
        trained_config = EvaluationConfig(
            maxsim_budget=8,
            candidate_pool=3,
            bootstrap_iterations=50,
            max_feature_cache_bytes=1024 * 1024,
            min_shared_tags=1,
            min_tag_jaccard=1.0,
        )
        trained_report = evaluate_jamendo(
            context,
            reader,
            config=trained_config,
            trained_bindings=[trained_binding],
            include_ablations=False,
        )
        trained_cache_binding = _benchmark_result_binding(
            context,
            reader,
            trained_config,
            trained_bindings=[trained_binding],
            include_ablations=False,
        )
        changed_cache_binding = _benchmark_result_binding(
            context,
            reader,
            trained_config,
            trained_bindings=[
                replace(
                    trained_binding,
                    model_artifact_sha256=hashlib.sha256(b"other-model").hexdigest(),
                )
            ],
            include_ablations=False,
        )
    report = reports[-1]
    assert [item["protocol"]["maxsim_budget"] for item in reports] == [8, 16, 32]
    assert report["evidence_scope"] == EVIDENCE_SCOPE
    assert report["protocol"]["artist_disjoint_official_fold"] is True
    assert "shared-tag retrieval" in report["protocol"]["claim_scope"]
    assert report["protocol"]["min_shared_tags"] == 1
    assert report["protocol"]["metric_labels"] == {
        "recall_at_k": "Recall@10",
        "mrr": "standard MRR over the complete ranked list",
        "graded_ndcg_at_k": "graded NDCG@10",
    }
    assert report["protocol"]["hybrid_weights"]["global_cosine"] == 0.5
    assert report["protocol"]["effective_unique_section_limits"][
        "repeated_sections"
    ]["tracks_repeating_for_requested_budget"] == 1
    assert set(report["aggregate"]) == set(METHODS)
    assert report["per_scene"]["genre"]["queries"] == 4
    assert report["per_tag"]["genre---rock"]["queries"] == 2
    assert report["resources"]["feature_cache_bytes"] > 0
    assert "store_bytes" in report["resources"]
    assert set(
        report["aggregate"]["uniform_window_maxsim"]["comparison_to_global"]
    ) == set(METRICS)
    assert len(
        report["aggregate"]["uniform_window_maxsim"]["comparison_to_global"][
            "graded_ndcg_at_k"
        ]["paired_bootstrap_ci95"]
    ) == 2
    assert "multiple comparisons" in report["grouped_metrics_notice"]

    trained_method = _trained_method_id("nonnegative_linear", 17, "none")
    assert trained_report["trained_methods"] == [trained_method]
    assert trained_method in trained_report["aggregate"]
    assert trained_method in trained_report["query_records"][0]["metrics"]
    assert trained_method in trained_report["per_scene"]["genre"]["methods"]
    assert trained_method in trained_report["resources"]["latency_seconds"]
    assert trained_report["trained_model_bindings"][trained_method]["promoted"] is False
    assert set(trained_report["trained_paired_deltas"][trained_method]) == {
        "paired_candidate_minus_global",
        "paired_candidate_minus_frozen_hybrid",
    }
    assert trained_cache_binding["trained_evaluation"] != changed_cache_binding[
        "trained_evaluation"
    ]

    expected_metadata = {
        key: trained_report[key]
        for key in (
            "schema_version",
            "evidence_scope",
            "dataset",
            "lawful_use",
            "source_fingerprint",
            "store",
            "protocol",
            "candidate_tracks",
        )
    }
    cache_path = tmp_path / "trained-cache.json"
    _write_benchmark_result(cache_path, trained_cache_binding, trained_report)
    assert (
        _load_valid_benchmark_result(
            cache_path, trained_cache_binding, expected_metadata
        )
        == trained_report
    )

    artifact = json.loads(cache_path.read_text(encoding="utf-8"))
    artifact["result"]["trained_model_bindings"][trained_method][
        "model_artifact_sha256"
    ] = hashlib.sha256(b"forged-model").hexdigest()
    _rewrite_artifact(cache_path, artifact)
    assert (
        _load_valid_benchmark_result(
            cache_path, trained_cache_binding, expected_metadata
        )
        is None
    )

    _write_benchmark_result(cache_path, trained_cache_binding, trained_report)
    artifact = json.loads(cache_path.read_text(encoding="utf-8"))
    artifact["result"]["trained_paired_deltas"][trained_method][
        "paired_candidate_minus_frozen_hybrid"
    ]["mrr"]["mean_delta"] += 0.01
    _rewrite_artifact(cache_path, artifact)
    assert (
        _load_valid_benchmark_result(
            cache_path, trained_cache_binding, expected_metadata
        )
        is None
    )


def test_evaluation_rejects_store_declaring_too_few_sections(tmp_path):
    tracks = tuple(_track(index, "genre---rock") for index in range(4))
    fold = ArtistFold(
        index=0,
        track_parts=MappingProxyType({track.track_id: "test" for track in tracks}),
        artist_parts=MappingProxyType({track.artist_id: "test" for track in tracks}),
        track_tags=MappingProxyType({track.track_id: track.tags for track in tracks}),
        tags=("genre---rock",),
    )
    context = JamendoContext(
        tracks=tracks,
        folds=(fold,),
        metadata_root=tmp_path,
        audio_root=tmp_path,
        state_root=tmp_path,
        metadata_commit="fixture",
        archive_manifest_sha256=HASH,
        track_manifest_sha256=HASH,
        metadata_hashes=MappingProxyType({}),
        source_fingerprint=HASH,
    )
    root = tmp_path / "store-8"
    with FullTrackStore(
        root,
        track_ids=[track.track_id for track in tracks],
        source_hashes=[HASH] * len(tracks),
        source_fingerprint=HASH,
        config_sha256=HASH,
        model_sha256=HASH,
        model_id="fake",
        embedding_dim=4,
        shard_tracks=2,
        repetition_sections=8,
        salient_sections=8,
    ) as store:
        for index, track in enumerate(tracks):
            vector = _unit(index)
            windows = np.stack([vector] * 16)
            store.write_track(
                track.track_id,
                HASH,
                TrackArtifacts(
                    global_embedding=vector,
                    window_embeddings=windows,
                    window_starts=np.arange(16, dtype=np.int64) * 5,
                    repeated_sections=windows[:8],
                    salient_sections=windows[:8],
                    repeated_indices=np.arange(8, dtype=np.int64),
                    salient_indices=np.arange(8, dtype=np.int64),
                    decoded_samples=90,
                ),
            )
        store.seal()
    with FullTrackStoreReader(root) as reader:
        with pytest.raises(
            FullTrackEvaluationError,
            match="requested section/hybrid budget 16 exceeds store-declared",
        ):
            evaluate_jamendo(
                context,
                reader,
                config=EvaluationConfig(
                    maxsim_budget=16,
                    candidate_pool=3,
                    bootstrap_iterations=10,
                    min_shared_tags=1,
                    min_tag_jaccard=1.0,
                ),
            )


def test_trained_model_loader_rejects_unsafe_identity_before_read(tmp_path):
    root = tmp_path / "trained"
    root.mkdir()
    common = {
        "trained_root": root,
        "fold_index": 0,
        "candidate_kind": "nonnegative_linear",
        "seed": 17,
        "expected_source_fingerprint": HASH,
        "expected_store_binding_sha256": HASH,
        "store_embedding_dim": 4,
        "store_repetition_sections": 32,
        "store_salient_sections": 32,
    }
    with pytest.raises(FullTrackEvaluationError, match="unknown trained candidate"):
        load_trained_model_for_fold(
            **{**common, "candidate_kind": "../outside"}
        )
    with pytest.raises(FullTrackEvaluationError, match="non-negative"):
        load_trained_model_for_fold(**{**common, "seed": -1})

    outside = tmp_path / "outside"
    outside.mkdir()
    linked_job = root / "fold-0" / "nonnegative_linear" / "seed-17"
    linked_job.parent.mkdir(parents=True)
    try:
        linked_job.symlink_to(outside, target_is_directory=True)
    except OSError:
        return
    with pytest.raises(FullTrackEvaluationError, match="escapes"):
        load_trained_model_for_fold(**common)


def _all_fold_fixture(
    tmp_path, *, store_name="all-fold-store", vector_shift=0
):
    tracks = tuple(_track(index, "genre---rock") for index in range(4))
    folds = tuple(
        ArtistFold(
            index=fold,
            track_parts=MappingProxyType(
                {track.track_id: "test" for track in tracks}
            ),
            artist_parts=MappingProxyType(
                {track.artist_id: "test" for track in tracks}
            ),
            track_tags=MappingProxyType(
                {track.track_id: track.tags for track in tracks}
            ),
            tags=("genre---rock",),
        )
        for fold in range(5)
    )
    context = JamendoContext(
        tracks=tracks,
        folds=folds,
        metadata_root=tmp_path,
        audio_root=tmp_path,
        state_root=tmp_path,
        metadata_commit="fixture",
        archive_manifest_sha256=HASH,
        track_manifest_sha256=HASH,
        metadata_hashes=MappingProxyType({}),
        source_fingerprint=HASH,
    )
    root = tmp_path / store_name
    with FullTrackStore(
        root,
        track_ids=[track.track_id for track in tracks],
        source_hashes=[HASH] * len(tracks),
        source_fingerprint=HASH,
        config_sha256=HASH,
        model_sha256=HASH,
        model_id="fake",
        embedding_dim=4,
        shard_tracks=2,
        repetition_sections=32,
        salient_sections=32,
    ) as store:
        for index, track in enumerate(tracks):
            vector = _unit((index + vector_shift) % len(tracks))
            windows = np.stack([vector] * 40)
            store.write_track(
                track.track_id,
                HASH,
                TrackArtifacts(
                    global_embedding=vector,
                    window_embeddings=windows,
                    window_starts=np.arange(40, dtype=np.int64) * 5,
                    repeated_sections=windows[:32],
                    salient_sections=windows[:32],
                    repeated_indices=np.arange(32, dtype=np.int64),
                    salient_indices=np.arange(32, dtype=np.int64),
                    decoded_samples=210,
                ),
            )
        store.seal()
    return context, root


def _store_binding():
    return {
        "schema_version": 2,
        "source_fingerprint": HASH,
        "config_sha256": HASH,
        "model_sha256": HASH,
        "model_id": "fixture",
        "embedding_dim": 4,
        "track_count": 1_000,
        "shard_tracks": 100,
        "repetition_sections": 32,
        "salient_sections": 32,
        "track_plan_sha256": HASH,
        "sealed_manifest_sha256": HASH,
    }


def _effective_limits(budget: int, track_count: int):
    return {
        stream: {
            "store_declared_budget": 32,
            "requested_budget": budget,
            "minimum_selected_source_windows": 32,
            "median_selected_source_windows": 32.0,
            "maximum_selected_source_windows": 32,
            "tracks_repeating_for_requested_budget": 0,
            "track_count": track_count,
        }
        for stream in ("repeated_sections", "salient_sections")
    }


def _resource_metadata():
    return {
        "wall_seconds": 0.1,
        "rss_before_bytes": 1,
        "rss_after_bytes": 1,
        "rss_observed_peak_bytes": 1,
        "cuda_peak_allocated_bytes": 0,
        "feature_cache_bytes": 1,
        "store_bytes": 1,
        "latency_seconds": {
            method: {"mean": 0.0, "p50": 0.0, "p95": 0.0}
            for method in METHODS
        },
    }


def _synthetic_all_fold_reports():
    reports = []
    for fold in range(5):
        for budget in (8, 16, 32):
            candidate_tracks = 100 + fold
            config = EvaluationConfig(
                fold_index=fold,
                maxsim_budget=budget,
                bootstrap_iterations=100,
                bootstrap_seed=1234,
            )
            records = []
            for query in range(fold + 1):
                method_metrics = {}
                for method_index, method in enumerate(METHODS):
                    base = 0.1 + fold * 0.05 + query * 0.01 + budget * 0.0001
                    method_metrics[method] = {
                        metric: base + method_index * 0.01 for metric in METRICS
                    }
                records.append(
                    {
                        "track_id": 1_000 + fold * 10 + query,
                        "artist_id": 2_000 + fold * 10 + query,
                        "tags": ["genre---fixture"],
                        "relevant_candidates": 10,
                        "metrics": method_metrics,
                    }
                )
            aggregate = {
                method: {
                    "metrics": {
                        metric: float(
                            np.mean(
                                [
                                    record["metrics"][method][metric]
                                    for record in records
                                ]
                            )
                        )
                        for metric in METRICS
                    }
                }
                for method in METHODS
            }
            descriptors = [
                {
                    key: record[key]
                    for key in (
                        "track_id",
                        "artist_id",
                        "tags",
                        "relevant_candidates",
                    )
                }
                for record in records
            ]
            report = {
                "schema_version": EVALUATION_SCHEMA_VERSION,
                "evidence_scope": EVIDENCE_SCOPE,
                "dataset": DATASET_DESCRIPTION,
                "lawful_use": LAWFUL_USE,
                "source_fingerprint": HASH,
                "store": _store_binding(),
                "protocol": _evaluation_protocol(
                    config,
                    _effective_limits(budget, candidate_tracks),
                    query_descriptor_sha256=_query_descriptor_sha256(descriptors, 0),
                ),
                "candidate_tracks": candidate_tracks,
                "queries": len(records),
                "query_records": records,
                "aggregate": aggregate,
                "skipped_no_relevant": 0,
                "per_scene": _grouped_metrics(records, "scene"),
                "per_tag": _grouped_metrics(records, "tag"),
                "grouped_metrics_notice": GROUPED_METRICS_NOTICE,
                "resources": _resource_metadata(),
            }
            reports.append(
                report
            )
    return reports


def _with_synthetic_trained_method(reports):
    methods = [
        _trained_method_id("nonnegative_linear", seed, "none")
        for seed in (17, 29, 43)
    ]
    for report in reports:
        fold = report["protocol"]["fold_index"]
        for record in report["query_records"]:
            for method in methods:
                record["metrics"][method] = dict(record["metrics"]["hybrid"])
        for method in methods:
            report["aggregate"][method] = {
                "metrics": {
                    metric: float(
                        np.mean(
                            [
                                record["metrics"][method][metric]
                                for record in report["query_records"]
                            ]
                        )
                    )
                    for metric in METRICS
                }
            }
        report["trained_methods"] = methods
        report["trained_model_bindings"] = {}
        report["trained_paired_deltas"] = {}
        for seed, method in zip((17, 29, 43), methods):
            report["trained_model_bindings"][method] = {
                "candidate_kind": "nonnegative_linear",
                "seed": seed,
                "fold_index": fold,
                "ablation": "none",
                "model_artifact_sha256": hashlib.sha256(
                    f"model-fold-{fold}-seed-{seed}".encode()
                ).hexdigest(),
                "model_json_sha256": hashlib.sha256(
                    f"model-json-fold-{fold}-seed-{seed}".encode()
                ).hexdigest(),
                "weights_npz_sha256": hashlib.sha256(
                    f"weights-fold-{fold}-seed-{seed}".encode()
                ).hexdigest(),
                "report_sha256": hashlib.sha256(
                    f"report-fold-{fold}-seed-{seed}".encode()
                ).hexdigest(),
                "source_fingerprint": HASH,
                "store_binding_sha256": stable_json_sha256(report["store"]),
                "training_config_sha256": HASH,
                "job_config_sha256": HASH,
                "maxsim_budget": 8,
                "embedding_dim": 4,
                "promoted": False,
            }
            report["trained_paired_deltas"][method] = {
                "paired_candidate_minus_global": {
                    metric: _paired_bootstrap_delta(
                        [
                            record["metrics"]["global_cosine"][metric]
                            for record in report["query_records"]
                        ],
                        [
                            record["metrics"][method][metric]
                            for record in report["query_records"]
                        ],
                        iterations=report["protocol"]["bootstrap_iterations"],
                        seed=report["protocol"]["bootstrap_seed"],
                    )
                    for metric in METRICS
                },
                "paired_candidate_minus_frozen_hybrid": {
                    metric: _paired_bootstrap_delta(
                        [
                            record["metrics"]["hybrid"][metric]
                            for record in report["query_records"]
                        ],
                        [
                            record["metrics"][method][metric]
                            for record in report["query_records"]
                        ],
                        iterations=report["protocol"]["bootstrap_iterations"],
                        seed=report["protocol"]["bootstrap_seed"],
                    )
                    for metric in METRICS
                },
            }
        active_methods = list(METHODS) + methods
        report["per_scene"] = _grouped_metrics(
            report["query_records"], "scene", methods=active_methods
        )
        report["per_tag"] = _grouped_metrics(
            report["query_records"], "tag", methods=active_methods
        )
        for method in methods:
            report["resources"]["latency_seconds"][method] = {
                "mean": 0.0,
                "p50": 0.0,
                "p95": 0.0,
            }
    return reports


def test_all_fold_aggregation_is_aligned_weighted_and_deterministic():
    reports = _synthetic_all_fold_reports()
    first = aggregate_all_fold_results(
        reports, bootstrap_iterations=100, bootstrap_seed=1234
    )
    second = aggregate_all_fold_results(
        reports, bootstrap_iterations=100, bootstrap_seed=1234
    )
    assert first == second
    budget = first["by_budget"]["8"]
    assert set(budget["per_fold"]) == {"0", "1", "2", "3", "4"}
    assert budget["query_weighted"]["queries"] == 15
    assert budget["fold_macro"]["methods"]["global_cosine"][
        "mrr"
    ] != pytest.approx(
        budget["query_weighted"]["methods"]["global_cosine"]["metrics"]["mrr"]
    )
    comparison = budget["query_weighted"]["methods"]["hybrid"][
        "paired_method_minus_global"
    ]["mrr"]
    assert comparison["mean_delta"] == pytest.approx(0.03)
    assert comparison["improved_queries"] == 15
    assert "descriptive" in first["multiple_comparisons_notice"]


def test_all_fold_aggregation_rejects_trained_artifact_drift_across_budgets():
    reports = _with_synthetic_trained_method(_synthetic_all_fold_reports())
    aggregate_all_fold_results(
        reports, bootstrap_iterations=100, bootstrap_seed=1234
    )

    method = _trained_method_id("nonnegative_linear", 17, "none")
    target = next(
        report
        for report in reports
        if report["protocol"]["fold_index"] == 0
        and report["protocol"]["maxsim_budget"] == 16
    )
    target["trained_model_bindings"][method]["model_artifact_sha256"] = (
        hashlib.sha256(b"different-model-same-method").hexdigest()
    )
    with pytest.raises(FullTrackEvaluationError, match="trained model identity drift"):
        aggregate_all_fold_results(
            reports, bootstrap_iterations=100, bootstrap_seed=1234
        )


def test_trained_matrix_exports_selector_inputs_at_explicit_budget(tmp_path):
    reports = _with_synthetic_trained_method(_synthetic_all_fold_reports())
    candidate_list, evaluations = build_selection_inputs(
        reports,
        deciding_budget=8,
        primary_metric="recall_at_k",
        list_id="synthetic-trained-matrix",
    )

    assert candidate_list["deciding_budget"] == 8
    assert candidate_list["primary_metric"] == "recall_at_k"
    assert candidate_list["evaluation_identity"] == {
        "source_fingerprint": HASH,
        "store_binding_sha256": stable_json_sha256(_store_binding()),
    }
    assert candidate_list["candidates"] == [
        {
            "candidate_kind": "nonnegative_linear",
            "model_bundle_sha256": candidate_list["candidates"][0][
                "model_bundle_sha256"
            ],
        }
    ]
    assert len(evaluations) == 15
    assert {
        (item["candidate_kind"], item["fold"], item["seed"])
        for item in evaluations
    } == {
        ("nonnegative_linear", fold, seed)
        for fold in range(5)
        for seed in (17, 29, 43)
    }
    assert all(item["benchmark_budget"] == 8 for item in evaluations)
    assert all(item["primary_metric"] == "recall_at_k" for item in evaluations)
    assert all(
        item["candidate_list_sha256"] == candidate_list["content_sha256"]
        for item in evaluations
    )
    assert all(
        item["metrics"]["candidate"] == item["metrics"]["frozen_hybrid"]
        for item in evaluations
    )
    assert all(
        item["paired_candidate_minus_frozen_hybrid"]["mean_delta"] == 0.0
        for item in evaluations
    )
    manifest = write_selection_inputs(
        tmp_path / "selection-inputs", candidate_list, evaluations
    )
    output_dir = tmp_path / "selection-inputs"
    assert manifest["candidate_list"]["file"] == "candidate-list.json"
    assert (output_dir / "candidate-list.json").is_file()
    assert len(manifest["evaluation_reports"]) == 15
    assert all(
        (output_dir / item["file"]).is_file()
        for item in manifest["evaluation_reports"]
    )


def test_all_fold_aggregation_rejects_query_and_method_misalignment():
    reports = _synthetic_all_fold_reports()
    missing_query = copy.deepcopy(reports)
    target = next(
        report
        for report in missing_query
        if report["protocol"]["fold_index"] == 1
        and report["protocol"]["maxsim_budget"] == 16
    )
    target["query_records"].pop()
    target["queries"] -= 1
    remaining = target["query_records"][0]["metrics"]
    target["aggregate"] = {
        method: {"metrics": dict(remaining[method])} for method in METHODS
    }
    descriptors = [
        {
            key: record[key]
            for key in ("track_id", "artist_id", "tags", "relevant_candidates")
        }
        for record in target["query_records"]
    ]
    target["protocol"]["query_descriptor_sha256"] = _query_descriptor_sha256(
        descriptors, 0
    )
    target["per_scene"] = _grouped_metrics(target["query_records"], "scene")
    target["per_tag"] = _grouped_metrics(target["query_records"], "tag")
    with pytest.raises(FullTrackEvaluationError, match="query alignment drift"):
        aggregate_all_fold_results(
            missing_query, bootstrap_iterations=100, bootstrap_seed=1234
        )

    missing_method = copy.deepcopy(reports)
    missing_method[0]["query_records"][0]["metrics"].pop("hybrid")
    with pytest.raises(FullTrackEvaluationError, match="query alignment is incomplete"):
        aggregate_all_fold_results(
            missing_method, bootstrap_iterations=100, bootstrap_seed=1234
        )


def _write_valid_cached_result(tmp_path):
    config = EvaluationConfig(
        fold_index=0,
        maxsim_budget=8,
        bootstrap_iterations=10,
        bootstrap_seed=7,
    )
    store_binding = _store_binding()
    binding = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "evidence_scope": EVIDENCE_SCOPE,
        "source_fingerprint": HASH,
        "store_binding": store_binding,
        "evaluation_config": asdict(config),
        "metric_fields": list(METRICS),
    }
    query_metrics = {
        method: {metric: 0.5 for metric in METRICS} for method in METHODS
    }
    query_record = {
        "track_id": 123,
        "artist_id": 456,
        "tags": ["genre---fixture"],
        "relevant_candidates": 3,
        "metrics": query_metrics,
    }
    query_descriptor = {
        key: query_record[key]
        for key in ("track_id", "artist_id", "tags", "relevant_candidates")
    }
    expected_metadata = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "evidence_scope": EVIDENCE_SCOPE,
        "dataset": DATASET_DESCRIPTION,
        "lawful_use": LAWFUL_USE,
        "source_fingerprint": HASH,
        "store": store_binding,
        "protocol": _evaluation_protocol(
            config,
            _effective_limits(8, 4),
            query_descriptor_sha256=_query_descriptor_sha256(
                [query_descriptor], 0
            ),
        ),
        "candidate_tracks": 4,
    }
    result = {
        **expected_metadata,
        "queries": 1,
        "skipped_no_relevant": 0,
        "query_records": [query_record],
        "aggregate": {
            method: {
                "metrics": dict(query_metrics[method]),
                "bootstrap_ci95": {
                    metric: [0.5, 0.5] for metric in METRICS
                },
                "comparison_to_global": {
                    metric: {
                        "mean_delta": 0.0,
                        "paired_bootstrap_ci95": [0.0, 0.0],
                        "bootstrap_probability_delta_gt_zero": 0.0,
                        "improved_queries": 0,
                        "regressed_queries": 0,
                        "unchanged_queries": 1,
                    }
                    for metric in METRICS
                },
            }
            for method in METHODS
        },
        "per_scene": _grouped_metrics([query_record], "scene"),
        "per_tag": _grouped_metrics([query_record], "tag"),
        "grouped_metrics_notice": GROUPED_METRICS_NOTICE,
        "resources": _resource_metadata(),
    }
    path = tmp_path / "fold-0-budget-8.json"
    _write_benchmark_result(path, binding, result)
    return path, binding, expected_metadata, result


def _tamper_result(result, case):
    if case == "part":
        result["protocol"]["part"] = "validation"
    elif case == "fold":
        result["protocol"]["fold_index"] = 1
    elif case == "budget":
        result["protocol"]["maxsim_budget"] = 16
    elif case == "metric_labels":
        result["protocol"]["metric_labels"]["recall_at_k"] = "Recall@999"
    elif case == "effective_section_diversity":
        result["protocol"]["effective_unique_section_limits"][
            "repeated_sections"
        ]["minimum_selected_source_windows"] = 31
    elif case == "method_definition":
        result["protocol"]["method_definitions"]["hybrid"] = "tampered"
    elif case == "candidate_pool":
        result["protocol"]["candidate_pool"] = 999
    elif case == "source_binding":
        other_hash = hashlib.sha256(b"other").hexdigest()
        result["source_fingerprint"] = other_hash
        result["store"]["source_fingerprint"] = other_hash
    elif case == "query_descriptor":
        record = result["query_records"][0]
        record["track_id"] += 1
        descriptor = {
            key: record[key]
            for key in ("track_id", "artist_id", "tags", "relevant_candidates")
        }
        result["protocol"]["query_descriptor_sha256"] = _query_descriptor_sha256(
            [descriptor], result["skipped_no_relevant"]
        )
    elif case == "malformed_tag":
        result["query_records"][0]["tags"] = ["---orphan"]
    elif case == "empty_scene_group":
        result["per_scene"] = {"": result["per_scene"]["genre"]}
    else:
        raise AssertionError(case)


def _rewrite_artifact(path, artifact):
    artifact["result_sha256"] = stable_json_sha256(artifact["result"])
    payload = dict(artifact)
    payload.pop("artifact_payload_sha256", None)
    artifact["artifact_payload_sha256"] = stable_json_sha256(payload)
    path.write_text(json.dumps(artifact), encoding="utf-8")


def test_benchmark_artifact_reuse_requires_exact_config_binding(tmp_path):
    path, binding, expected_metadata, result = _write_valid_cached_result(tmp_path)
    assert _load_valid_benchmark_result(path, binding, expected_metadata) == result

    changed = copy.deepcopy(binding)
    changed["evaluation_config"]["candidate_pool"] = 999
    assert _load_valid_benchmark_result(path, changed, expected_metadata) is None

    artifact = json.loads(path.read_text(encoding="utf-8"))
    artifact["result"]["aggregate"]["global_cosine"]["metrics"]["mrr"] = 0.9
    _rewrite_artifact(path, artifact)
    assert _load_valid_benchmark_result(path, binding, expected_metadata) is None


@pytest.mark.parametrize(
    "case",
    [
        "part",
        "fold",
        "budget",
        "metric_labels",
        "effective_section_diversity",
        "method_definition",
        "candidate_pool",
        "source_binding",
        "query_descriptor",
        "malformed_tag",
        "empty_scene_group",
    ],
)
def test_cached_result_rejects_rehashed_protocol_or_label_tampering(tmp_path, case):
    path, binding, expected_metadata, _ = _write_valid_cached_result(tmp_path)
    artifact = json.loads(path.read_text(encoding="utf-8"))
    _tamper_result(artifact["result"], case)
    _rewrite_artifact(path, artifact)
    assert _load_valid_benchmark_result(path, binding, expected_metadata) is None


@pytest.mark.parametrize("hash_field", ["result_sha256", "artifact_payload_sha256"])
def test_cached_result_rejects_payload_hash_tampering(tmp_path, hash_field):
    path, binding, expected_metadata, _ = _write_valid_cached_result(tmp_path)
    artifact = json.loads(path.read_text(encoding="utf-8"))
    artifact[hash_field] = "0" * 64
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert _load_valid_benchmark_result(path, binding, expected_metadata) is None


def test_cached_result_treats_pathological_json_integer_as_cache_miss(tmp_path):
    path, binding, expected_metadata, _ = _write_valid_cached_result(tmp_path)
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        '"candidate_tracks": 4',
        '"candidate_tracks": ' + ("9" * 5_000),
        1,
    )
    path.write_text(text, encoding="utf-8")
    assert _load_valid_benchmark_result(path, binding, expected_metadata) is None


def test_cached_result_treats_unpaired_unicode_surrogate_as_cache_miss(tmp_path):
    path, binding, expected_metadata, _ = _write_valid_cached_result(tmp_path)
    artifact = json.loads(path.read_text(encoding="utf-8"))
    artifact["artifact_kind"] = "\ud800"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    assert _load_valid_benchmark_result(path, binding, expected_metadata) is None


def test_benchmark_all_recomputes_one_rehashed_stale_protocol_artifact(tmp_path):
    context, store_root = _all_fold_fixture(tmp_path)
    output_dir = tmp_path / "all-fold-results"
    config = EvaluationConfig(
        candidate_pool=3,
        bootstrap_iterations=2,
        bootstrap_seed=11,
        max_feature_cache_bytes=1024 * 1024,
        min_shared_tags=1,
        min_tag_jaccard=1.0,
    )
    with FullTrackStoreReader(store_root) as reader:
        _, first_resume = run_all_folds_benchmark(
            context, reader, output_dir=output_dir, base_config=config
        )
        assert first_resume == {"computed": 15, "reused": 0}

        path = output_dir / "fold-0-budget-8.json"
        artifact = json.loads(path.read_text(encoding="utf-8"))
        artifact["result"]["protocol"]["metric_labels"]["recall_at_k"] = "tampered"
        _rewrite_artifact(path, artifact)

        _, second_resume = run_all_folds_benchmark(
            context, reader, output_dir=output_dir, base_config=config
        )
    assert second_resume == {"computed": 1, "reused": 14}
    repaired = json.loads(path.read_text(encoding="utf-8"))
    assert repaired["result"]["protocol"]["metric_labels"]["recall_at_k"] == "Recall@10"


def test_trained_benchmark_all_writes_and_reuses_selector_inputs(tmp_path, monkeypatch):
    context, store_root = _all_fold_fixture(tmp_path)
    output_dir = tmp_path / "trained-all-fold-results"
    config = EvaluationConfig(
        candidate_pool=3,
        bootstrap_iterations=2,
        bootstrap_seed=11,
        max_feature_cache_bytes=1024 * 1024,
        min_shared_tags=1,
        min_tag_jaccard=1.0,
    )

    def fake_load_trained_model(
        trained_root,
        *,
        fold_index,
        candidate_kind,
        seed,
        expected_source_fingerprint,
        expected_store_binding_sha256,
        store_embedding_dim,
        store_repetition_sections,
        store_salient_sections,
    ):
        del (
            trained_root,
            store_repetition_sections,
            store_salient_sections,
        )
        model = build_nonneg_linear(
            np.ones(FEATURE_DIM, dtype=np.float64),
            FusionConfig(
                kind=candidate_kind,
                embedding_dim=store_embedding_dim,
                maxsim_budget=8,
                seed=seed,
                model_id=f"fixture-fold-{fold_index}-seed-{seed}",
                store_id=expected_store_binding_sha256,
                config_sha256=HASH,
                fold_index=fold_index,
            ),
        )
        identity = f"fold-{fold_index}-seed-{seed}".encode()
        return _TrainedModelBinding(
            candidate_kind=candidate_kind,
            seed=seed,
            fold_index=fold_index,
            model=model,
            report_sha256=hashlib.sha256(b"report-" + identity).hexdigest(),
            model_artifact_sha256=hashlib.sha256(b"artifact-" + identity).hexdigest(),
            model_json_sha256=hashlib.sha256(b"json-" + identity).hexdigest(),
            weights_npz_sha256=hashlib.sha256(b"weights-" + identity).hexdigest(),
            source_fingerprint=expected_source_fingerprint,
            store_binding_sha256=expected_store_binding_sha256,
            training_config_sha256=HASH,
            job_config_sha256=HASH,
            fusion_metadata={},
            maxsim_budget=8,
            embedding_dim=store_embedding_dim,
        )

    monkeypatch.setattr(
        "soundalike.ml.fulltrack_eval.load_trained_model_for_fold",
        fake_load_trained_model,
    )
    kwargs = {
        "output_dir": output_dir,
        "base_config": config,
        "trained_root": tmp_path / "trained-models",
        "trained_candidates": ["nonnegative_linear"],
        "trained_seeds": [17, 29, 43],
        "include_ablations": False,
    }
    with FullTrackStoreReader(store_root) as reader:
        first, first_resume = run_all_folds_benchmark(context, reader, **kwargs)
        second, second_resume = run_all_folds_benchmark(context, reader, **kwargs)

    assert first_resume == {"computed": 15, "reused": 0}
    assert second_resume == {"computed": 0, "reused": 15}
    assert first["selection_inputs"] == second["selection_inputs"]
    assert len(first["selection_inputs"]["evaluation_reports"]) == 15
    assert (output_dir / "benchmark-summary-trained.json").is_file()
    assert (output_dir / "selection" / "candidate-list.json").is_file()
    assert (output_dir / "selection" / "selection-inputs.json").is_file()

    worker_output = tmp_path / "trained-fold-worker-results"
    worker_kwargs = {
        **kwargs,
        "output_dir": worker_output,
        "worker_fold": 3,
    }
    with FullTrackStoreReader(store_root) as reader:
        worker_result, worker_resume = run_all_folds_benchmark(
            context, reader, **worker_kwargs
        )
    assert worker_resume == {"computed": 3, "reused": 0}
    assert worker_result["fold"] == 3
    assert [
        item["budget"] for item in worker_result["result_artifacts"]
    ] == [8, 16, 32]
    assert sorted(path.name for path in worker_output.glob("*.json")) == [
        "fold-3_budget-16_trained.json",
        "fold-3_budget-32_trained.json",
        "fold-3_budget-8_trained.json",
    ]
    assert not (worker_output / "benchmark-summary-trained.json").exists()
    assert not (worker_output / "selection").exists()

    def blocked_selector(*args, **kwargs):
        raise FullTrackSelectionError("synthetic selector failure")

    monkeypatch.setattr(
        "soundalike.ml.fulltrack_selection.build_selection_inputs",
        blocked_selector,
    )
    with FullTrackStoreReader(store_root) as reader:
        with pytest.raises(FullTrackEvaluationError, match="selector export failed"):
            run_all_folds_benchmark(context, reader, **kwargs)

    monkeypatch.setattr(
        "soundalike.ml.fulltrack_selection.build_selection_inputs",
        build_selection_inputs,
    )

    def blocked_write(*args, **kwargs):
        raise PermissionError("synthetic selector write denial")

    monkeypatch.setattr(
        "soundalike.ml.fulltrack_selection.write_selection_inputs",
        blocked_write,
    )
    with FullTrackStoreReader(store_root) as reader:
        with pytest.raises(FullTrackEvaluationError, match="selector export failed"):
            run_all_folds_benchmark(context, reader, **kwargs)


def test_store_binding_includes_sealed_embedding_content_identity(tmp_path):
    _, first_root = _all_fold_fixture(
        tmp_path, store_name="first-store", vector_shift=0
    )
    _, second_root = _all_fold_fixture(
        tmp_path, store_name="second-store", vector_shift=1
    )
    with FullTrackStoreReader(first_root) as first, FullTrackStoreReader(
        second_root
    ) as second:
        assert first.binding.as_dict() == second.binding.as_dict()
        assert _evaluation_store_binding(first) != _evaluation_store_binding(second)


@pytest.mark.parametrize(
    "case",
    [
        "part",
        "fold",
        "budget",
        "metric_labels",
        "effective_section_diversity",
        "method_definition",
        "candidate_pool",
        "source_binding",
        "query_descriptor",
        "malformed_tag",
        "empty_scene_group",
    ],
)
def test_all_fold_aggregation_rejects_protocol_and_invariant_tampering(case):
    reports = _synthetic_all_fold_reports()
    _tamper_result(reports[0], case)
    with pytest.raises(FullTrackEvaluationError):
        aggregate_all_fold_results(
            reports, bootstrap_iterations=100, bootstrap_seed=1234
        )


def test_all_fold_aggregation_rejects_valid_results_in_wrong_matrix_slots():
    reports = _synthetic_all_fold_reports()
    reports[0], reports[1] = reports[1], reports[0]
    with pytest.raises(FullTrackEvaluationError, match="matrix slot"):
        aggregate_all_fold_results(
            reports, bootstrap_iterations=100, bootstrap_seed=1234
        )


def test_malformed_scene_tag_never_becomes_an_empty_group():
    with pytest.raises(FullTrackEvaluationError, match="malformed scene tag"):
        _scene_for_tag("---orphan")


def test_all_fold_aggregation_rejects_out_of_range_metrics():
    reports = _synthetic_all_fold_reports()
    reports[0]["query_records"][0]["metrics"]["hybrid"]["mrr"] = 2.0
    with pytest.raises(FullTrackEvaluationError, match=r"within \[0, 1\]"):
        aggregate_all_fold_results(
            reports, bootstrap_iterations=100, bootstrap_seed=1234
        )


def test_all_fold_aggregation_rejects_impossible_relevance_counts():
    reports = _synthetic_all_fold_reports()
    for report in reports[:3]:
        record = report["query_records"][0]
        record["relevant_candidates"] = report["candidate_tracks"]
        descriptor = {
            key: record[key]
            for key in ("track_id", "artist_id", "tags", "relevant_candidates")
        }
        report["protocol"]["query_descriptor_sha256"] = _query_descriptor_sha256(
            [descriptor], 0
        )
    with pytest.raises(FullTrackEvaluationError, match="relevance count"):
        aggregate_all_fold_results(
            reports, bootstrap_iterations=100, bootstrap_seed=1234
        )


def test_all_fold_aggregation_rejects_contradictory_section_diversity():
    reports = _synthetic_all_fold_reports()
    for report in reports:
        for details in report["protocol"][
            "effective_unique_section_limits"
        ].values():
            details["minimum_selected_source_windows"] = 4
            details["tracks_repeating_for_requested_budget"] = 0
    with pytest.raises(FullTrackEvaluationError, match="diversity semantics"):
        aggregate_all_fold_results(
            reports, bootstrap_iterations=100, bootstrap_seed=1234
        )


def test_benchmark_all_cli_fixes_official_fold_budget_matrix():
    args = build_parser().parse_args(
        [
            "benchmark-all",
            "--metadata-root",
            "metadata",
            "--audio-root",
            "audio",
            "--state-root",
            "state",
            "--store",
            "store",
            "--output-dir",
            "results",
            "--worker-fold",
            "3",
        ]
    )
    assert args.bootstrap_seed == 20260714
    assert args.recall_cutoff == 10
    assert args.ndcg_cutoff == 10
    assert args.worker_fold == 3
    assert not hasattr(args, "fold")
    assert not hasattr(args, "maxsim_budget")
