"""Regression tests for the frozen CLAP catalogue development challenger."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from soundalike.ml.clap_catalog_v13 import (
    CHECKPOINT_SHA256,
    DIMENSIONS,
    EmbeddingStore,
    FrozenClapEmbedder,
    _geometry_passes,
    geometry_metrics,
    orthogonal_projection,
    validate_preregistration,
)
from soundalike.ml.human_eval_v10 import content_hash
from soundalike.ml.human_eval_v10 import approve_export
from soundalike.ml.human_eval_v13 import (
    freeze_pack,
    semantic_order_hash,
    verify_pack,
)
from soundalike.ml.human_aggregate_v10 import aggregate, sign_export


ROOT = Path(__file__).parents[1]
GOAL = ROOT / ".goals" / "human-quality-recommendations"
PREREG = GOAL / "protocol-v13-clap-development"
PACK = GOAL / "protocol-v13-clap-human-development"
ARTIFACTS = GOAL / "artifacts"


def test_preregistration_is_hash_bound_signed_and_zero_rating():
    document = validate_preregistration(PREREG / "preregistration-v13-r3.json")
    assert document["ratings_count_at_freeze"] == 0
    assert document["protocol_revision"] == 3
    assert document["audio_preprocessing"]["download_workers"] == 32
    assert document["development_only"] is True
    assert document["production_ranking_changed"] is False
    assert document["commercial_final_opened"] is False
    assert document["ac3_claimed"] is False
    assert document["encoder"]["checkpoint_sha256"] == CHECKPOINT_SHA256
    assert document["compression"]["candidate_dimensions"] == list(DIMENSIONS)
    assert "may overlap MagnaTagATune" in document["hypothesis"][
        "pretraining_caveat"
    ]


def test_prior_v10_v11_public_studies_are_byte_identical():
    expected = {
        "protocol-v10-human-development/protocol-v10.json":
            "d5b16b6268bc8675a97b66f945531e0b648beeaf637d9e931156ffdd60de059c",
        "protocol-v10-human-development/served-lists-v10.json":
            "05e50613e5c5e2e9633ebbf67adc318f45651b2ca93df8ab0c11e12a12b38f8b",
        "protocol-v11-audio-access-erratum/protocol-v11.json":
            "7e58035cf2737dfaa9bd16c152df3a875885b50e434d1cec2c480259d9dbada9",
        "protocol-v11-audio-access-erratum/served-lists-v11.json":
            "7626b09675c60b78840829e64902c75e34a3bcf8c8bd1ac900d67a9d4353810a",
    }
    for relative, digest in expected.items():
        assert hashlib.sha256((GOAL / relative).read_bytes()).hexdigest() == digest


def test_embedding_store_resumes_by_exact_row_and_checksum(tmp_path):
    ids = np.asarray([101, 202, 303], dtype=np.int64)
    store = EmbeddingStore(tmp_path, ids)
    vector = np.linspace(-1, 1, 512, dtype=np.float32)
    vector /= np.linalg.norm(vector)
    half = vector.astype(np.float16)
    store.embeddings[1] = half
    store.mark_terminal(
        0, "no_preview", attempts=1
    )
    store.mark_terminal(
        1,
        "available",
        attempts=1,
        preview_sha256="a" * 64,
        embedding_sha256=hashlib.sha256(half.tobytes()).hexdigest(),
        preview_bytes=12_345,
    )
    store.commit()
    store.close()

    resumed = EmbeddingStore(tmp_path, ids)
    resumed.verify_available()
    assert resumed.counts() == {
        "pending": 1, "available": 1, "no_preview": 1, "error": 0
    }
    assert resumed.pending() == [(2, 303)]
    resumed.mark_terminal(2, "error", attempts=4, error="temporary outage")
    resumed.commit()
    assert resumed.pending(retry_errors=True) == [(2, 303)]
    assert resumed.starting_attempt(2) == 0
    assert resumed.pending(retry_errors=False) == []
    resumed.close()
    with pytest.raises(ValueError, match="track IDs"):
        EmbeddingStore(tmp_path, np.asarray([101, 303, 202], dtype=np.int64))


def test_fixed_three_windows_are_deterministic_and_cover_preview():
    waveform = np.arange(1_440_000, dtype=np.float32)
    windows = FrozenClapEmbedder._fixed_windows(waveform)
    assert len(windows) == 3
    assert all(window.shape == (480_000,) for window in windows)
    assert windows[0][0] == 0
    assert windows[1][0] == 480_000
    assert windows[2][0] == 960_000
    short = FrozenClapEmbedder._fixed_windows(np.arange(100, dtype=np.float32))
    assert len(short) == 3 and np.array_equal(short[0], short[2])


def test_frozen_orthogonal_projection_is_nested_and_geometry_is_measured():
    projection64 = orthogonal_projection(64)
    projection128 = orthogonal_projection(128)
    assert np.array_equal(projection64, projection128[:, :64])
    assert np.allclose(
        projection128.T @ projection128, np.eye(128), atol=1e-5
    )
    rng = np.random.default_rng(20260713)
    full = rng.normal(size=(256, 512)).astype(np.float32)
    full /= np.linalg.norm(full, axis=1, keepdims=True)
    compact = full @ projection128
    metrics = geometry_metrics(
        full,
        compact,
        np.arange(8),
        np.arange(8, 256),
        top_k=20,
        pair_samples=1_000,
    )
    assert set(metrics) == {
        "sampled_pair_cosine_spearman",
        "mean_top50_overlap",
        "p05_top50_overlap",
        "mean_union_top50_rank_spearman",
    }
    assert all(-1.0 <= value <= 1.0 for value in metrics.values())
    assert _geometry_passes(
        {
            "sampled_pair_cosine_spearman": 0.991,
            "mean_top50_overlap": 0.76,
            "mean_union_top50_rank_spearman": 0.61,
            "p05_top50_overlap": 0.56,
        }
    )


def test_v13_evaluator_keeps_three_class_and_zero_to_ten_contract():
    html = (ROOT / "benchmarks/human_eval_v13.html").read_text(encoding="utf-8")
    assert "schema_version:13" in html
    assert "Expected schema version 13." in html
    assert all(
        value in html
        for value in ("not_similar", "somewhat_similar", "very_similar")
    )
    assert 'min="0" max="10"' in html
    assert "junk_or_version" in html
    assert "unrelated_positions_1_to_3" in html
    assert "production_baseline" not in html and "challenger" not in html
    assert "fresh Deezer preview" in html


def test_synthetic_v13_pack_freezes_verifies_and_aggregates(tmp_path, monkeypatch):
    import soundalike.ml.human_eval_v13 as module

    ids = np.arange(1001, 1007, dtype=np.int64)
    monkeypatch.setattr(module, "EXPECTED_ROWS", len(ids))
    monkeypatch.setattr(module, "EXPECTED_SEEDS", 1)
    monkeypatch.setattr(module, "EXPECTED_SCENES", 1)
    monkeypatch.setattr(
        module, "TRACK_IDS_SHA256", hashlib.sha256(ids.tobytes()).hexdigest()
    )
    index = tmp_path / "index.npz"
    np.savez(
        index,
        track_ids=ids,
        titles=np.asarray(["Seed", "One", "Two", "Three", "Four", "Five"]),
        artists=np.asarray(["Seed Artist", "A", "B", "C", "D", "E"]),
    )
    diagnostics = {
        "schema_version": 13,
        "artifact_kind": "test",
        "preregistration_content_sha256":
            "2c1bb55c85dfa8d1d344bba02868563c459ac743604f525ecb678598f3ef4ee7",
        "commercial_human_ratings_used": 0,
        "proxy_evidence_is_deciding": False,
        "compact_asset_sha256": "b" * 64,
        "selected_challenger": "conservative_clap_fallback",
        "safety": {"production_changed": False},
        "production_baseline": {
            "records": [{
                "seed_id": "S-1", "scene": "test", "query_row": 0,
                "rows": [1, 2, 3, 4, 5],
            }]
        },
        "variants": {
            "conservative_clap_fallback": {
                "metrics": {"passes_proxy_safety": True},
                "records": [{
                    "seed_id": "S-1", "scene": "test", "query_row": 0,
                    "rows": [5, 4, 3, 2, 1],
                }],
            }
        },
    }
    diagnostics["content_sha256"] = content_hash(diagnostics)
    diagnostics_path = tmp_path / "diagnostics.json"
    diagnostics_path.write_text(json.dumps(diagnostics), encoding="utf-8")
    compact = {
        "schema_version": 13,
        "preregistration_content_sha256":
            "2c1bb55c85dfa8d1d344bba02868563c459ac743604f525ecb678598f3ef4ee7",
        "coverage": {"pending": 0, "error": 0, "available": 6, "no_preview": 0},
        "asset": {"sha256": "b" * 64, "bytes": 128},
        "float16_reload_metrics": {"mean_top50_overlap": 0.8},
    }
    compact["content_sha256"] = content_hash(compact)
    compact_path = tmp_path / "compact.json"
    compact_path.write_text(json.dumps(compact), encoding="utf-8")
    public, private = tmp_path / "public", tmp_path / "private"
    paths = freeze_pack(
        diagnostics_path,
        index,
        compact_path,
        PREREG / "preregistration-v13-r3.json",
        ROOT / "benchmarks/human_eval_v13.html",
        public,
        private,
    )
    verified = verify_pack(public, private_key=paths["method_key"])
    assert verified["lists"]["ratings_count_at_freeze"] == 0
    assert semantic_order_hash(verified["lists"]) == verified["lists"][
        "semantic_order_sha256"
    ]
    assert "production_baseline" not in paths["lists"].read_text(encoding="utf-8")

    seed = verified["lists"]["seeds"][0]
    result, served = seed["results"][0], seed["lists"][0]
    started = datetime.now(timezone.utc) - timedelta(minutes=1)
    rated = (started + timedelta(seconds=10)).isoformat()
    export_doc = {
        "schema_version": 13,
        "source_kind": "human_listener",
        "provider": "standalone_local_evaluator",
        "anonymous_rater_id": "anon-v13-synthetic-rater",
        "session_id": "session-v13-synthetic",
        "protocol_sha256": verified["protocol"]["content_sha256"],
        "served_lists_sha256": verified["lists"]["content_sha256"],
        "local_session_key": "0123456789abcdef" * 4,
        "started_at": started.isoformat(),
        "exported_at": (started + timedelta(seconds=30)).isoformat(),
        "duration_ms": 30_000,
        "result_ratings": {
            result["result_id"]: {
                "similarity": "somewhat_similar",
                "score_0_10": 5,
                "junk_or_version": False,
                "rated_at": rated,
                "interaction_ms": 1000,
            }
        },
        "list_ratings": {
            served["list_id"]: {
                "whole_list_coherence": "somewhat_coherent",
                "unrelated_positions_1_to_3": 1,
                "rated_at": rated,
                "interaction_ms": 1000,
            }
        },
    }
    export_doc["integrity_hmac_sha256"] = sign_export(
        export_doc, export_doc["local_session_key"]
    )
    export = tmp_path / "ratings.json"
    export.write_text(json.dumps(export_doc), encoding="utf-8")
    approve_export(export, paths["collector_private"])
    import soundalike.ml.human_aggregate_v10 as aggregate_module

    monkeypatch.setattr(
        aggregate_module,
        "TRUSTED_V13_PROTOCOL",
        verified["protocol"]["content_sha256"],
    )
    monkeypatch.setattr(
        aggregate_module,
        "TRUSTED_V13_LISTS",
        verified["lists"]["content_sha256"],
    )
    monkeypatch.setattr(
        aggregate_module,
        "TRUSTED_V13_STATE",
        verified["state"]["content_sha256"],
    )
    monkeypatch.setattr(
        aggregate_module,
        "TRUSTED_V13_FILES",
        {
            name: hashlib.sha256((public / name).read_bytes()).hexdigest()
            for name in (
                "protocol-v13.json",
                "served-lists-v13.json",
                "state.json",
                "state.sig",
                "allowed_signers",
                "signer.pub",
                "collector_allowed_signers",
                "collector_signer.pub",
            )
        },
    )
    report = aggregate(
        paths["protocol"], paths["lists"], paths["method_key"], [export]
    )
    assert report["schema_version"] == 13
    assert report["valid_export_count"] == 1


def test_committed_v13_pack_and_evidence_are_fail_closed():
    if not (PACK / "protocol-v13.json").is_file():
        pytest.skip("full fresh-catalog build has not completed yet")
    verified = verify_pack(PACK, require_trusted=True)
    protocol, lists, state = (
        verified["protocol"],
        verified["lists"],
        verified["state"],
    )
    assert protocol["ratings_count_at_freeze"] == 0
    assert lists["seed_count"] == 60 and lists["scene_count"] == 13
    assert state["phase"] == "RANKINGS_LOCKED"
    assert state["production_deployment_blocked"] is True
    assert protocol["production_changed"] is False
    assert protocol["deployed"] is False
    assert protocol["commercial_final_opened"] is False
    assert protocol["ac3_claimed"] is False
    assert semantic_order_hash(lists) == lists["semantic_order_sha256"]
    public = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (PACK / "protocol-v13.json", PACK / "served-lists-v13.json")
    )
    assert '"method_role"' not in public
    for name in (
        "clap-catalog-coverage-v13.json",
        "clap-compact-geometry-v13.json",
        "clap-variant-diagnostics-v13.json",
        "human-eval-preview-audit-v13.json",
        "clap-prospective-resources-v13.json",
    ):
        artifact = json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))
        assert content_hash(artifact) == artifact["content_sha256"]
    coverage = json.loads(
        (ARTIFACTS / "clap-catalog-coverage-v13.json").read_text(encoding="utf-8")
    )
    assert sum(coverage["coverage"].values()) == 272_853
    assert coverage["coverage"]["pending"] == 0
    compact = json.loads(
        (ARTIFACTS / "clap-compact-geometry-v13.json").read_text(encoding="utf-8")
    )
    assert compact["asset"]["bytes"] <= 70_000_000
    assert compact["float16_reload_metrics"]["mean_top50_overlap"] >= 0.75
    variants = json.loads(
        (ARTIFACTS / "clap-variant-diagnostics-v13.json").read_text(encoding="utf-8")
    )
    assert variants["commercial_human_ratings_used"] == 0
    assert variants["selected_challenger"] in variants["selection_order"]
    assert variants["variants"][variants["selected_challenger"]]["metrics"][
        "passes_proxy_safety"
    ]
    preview = json.loads(
        (ARTIFACTS / "human-eval-preview-audit-v13.json").read_text(encoding="utf-8")
    )
    assert preview["ranked_positions"]["resolvable_fraction"] >= 0.90
