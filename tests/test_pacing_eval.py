"""Focused tests for the isolated pacing V3 study builder."""
from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

import soundalike.ml.pacing_eval as pacing
from soundalike.ml.fulltrack_pilot import _content_sha256


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "webapp" / "evaluate-pacing-v3" / "pacing-pack.json"
ARCHIVE = ROOT / "webapp" / "evaluate-semantic-v2"
PRIVATE = (
    Path(r"C:\soundalike-data")
    / "mtg-jamendo-fulltrack-artifacts"
    / "pacing-eval-v3"
    / "private-unblinding.private.json"
)


def test_frozen_scoring_weights_and_compatibilities():
    assert pacing.ACOUSTIC_WEIGHTS == {
        "global_cosine": 0.30,
        "uniform_window_maxsim": 0.25,
        "repeated_section_maxsim": 0.25,
        "salient_section_maxsim": 0.20,
    }
    assert pacing.RERANK_WEIGHTS == {
        "acoustic": 0.75,
        "pacing": 0.10,
        "tone": 0.05,
        "dynamics": 0.05,
        "instrument": 0.03,
        "mood_theme": 0.01,
        "genre": 0.01,
    }
    values = pacing.tempo_compatibility(np.asarray([60.0, 120.0, 240.0]), 120.0)
    assert values[1] == pytest.approx(1.0)
    assert values[0] == pytest.approx(values[2])


def test_robust_standardization_uses_median_and_has_29_dimensions():
    matrix = np.zeros((5, 29), dtype=np.float64)
    matrix[:, 1] = [1.0, 2.0, 3.0, 4.0, 1_000_000.0]
    standardized = pacing.robust_standardize_vibe(matrix)
    assert standardized.shape == (5, 29)
    assert standardized[2, 1] == pytest.approx(0.0)
    assert np.all(np.isfinite(standardized))
    with pytest.raises(pacing.PacingEvalError, match="matrix"):
        pacing.robust_standardize_vibe(np.zeros((5, 28)))


def test_percentile_blend_is_frozen():
    acoustic = np.asarray([0.1, 0.2, 0.3])
    components = {
        "pacing": np.asarray([0.3, 0.2, 0.1]),
        "tone": np.asarray([0.1, 0.2, 0.3]),
        "dynamics": np.asarray([0.1, 0.2, 0.3]),
        "instrument": np.asarray([0.1, 0.2, 0.3]),
        "mood_theme": np.asarray([0.1, 0.2, 0.3]),
        "genre": np.asarray([0.1, 0.2, 0.3]),
    }
    score = pacing.pacing_rerank_scores(acoustic, components)
    expected = 0.75 * np.asarray([0.0, 0.5, 1.0])
    expected += 0.10 * np.asarray([1.0, 0.5, 0.0])
    expected += 0.15 * np.asarray([0.0, 0.5, 1.0])
    assert np.allclose(score, expected)


def test_cache_identity_and_hash_are_fail_closed(tmp_path, monkeypatch):
    path = tmp_path / "cache.npz"
    ids = np.asarray([1, 2], dtype=np.int64)
    starts = np.asarray([0.0, 1.0])
    ends = np.asarray([20.0, 21.0])
    np.savez_compressed(
        path,
        track_ids=ids,
        starts=starts,
        ends=ends,
        vibe=np.zeros((2, 29), dtype=np.float32),
    )
    monkeypatch.setattr(
        pacing, "EXPECTED_VIBE_CACHE_FILE_SHA256", hashlib.sha256(path.read_bytes()).hexdigest()
    )
    assert pacing.load_vibe_cache(path, ids, starts, ends).shape == (2, 29)
    with pytest.raises(pacing.PacingEvalError, match="identity"):
        pacing.load_vibe_cache(path, ids[::-1], starts, ends)
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(pacing.PacingEvalError, match="hash"):
        pacing.load_vibe_cache(path, ids, starts, ends)


def test_frozen_source_packs_are_rehashed_and_all_published_results_are_excluded(
    tmp_path,
):
    semantic_path = ARCHIVE / "semantic-pack.json"
    fulltrack_path = ARCHIVE / "pilot-pack.json"
    semantic = pacing._load_frozen_pack(
        semantic_path,
        expected_file_sha256=pacing.EXPECTED_SEMANTIC_V2_PACK_FILE_SHA256,
        expected_content_sha256=pacing.EXPECTED_SEMANTIC_V2_PACK_SHA256,
        label="semantic-v2 pack",
    )
    fulltrack = pacing._load_frozen_pack(
        fulltrack_path,
        expected_file_sha256=pacing.EXPECTED_FULLTRACK_V2_PACK_FILE_SHA256,
        expected_content_sha256=pacing.EXPECTED_FULLTRACK_V2_PACK_SHA256,
        label="fulltrack-v2 pack",
    )
    excluded = pacing._published_result_exclusions(fulltrack, semantic)
    committed = json.loads(PUBLIC.read_text(encoding="utf-8"))
    for seed in committed["seeds"]:
        active = {
            row["track_id"]
            for candidate_list in seed["lists"]
            for row in candidate_list["ranking"]
        }
        assert active.isdisjoint(excluded[seed["seed_track_id"]])

    tampered = tmp_path / "semantic-pack.json"
    document = copy.deepcopy(semantic)
    document["notice"] += " tampered"
    tampered.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(pacing.PacingEvalError, match="file drift"):
        pacing._load_frozen_pack(
            tampered,
            expected_file_sha256=pacing.EXPECTED_SEMANTIC_V2_PACK_FILE_SHA256,
            expected_content_sha256=pacing.EXPECTED_SEMANTIC_V2_PACK_SHA256,
            label="semantic-v2 pack",
        )


def _synthetic_documents():
    committed = json.loads(PUBLIC.read_text(encoding="utf-8"))
    source_seeds = []
    audio = {}
    challenger = {}
    tracks = {}
    for seed_index in range(20):
        seed_id = 1_000 + seed_index
        source_seeds.append({"seed_track_id": seed_id})
        audio[seed_id] = tuple(10_000 + seed_index * 20 + offset for offset in range(5))
        challenger[seed_id] = (
            audio[seed_id][0],
            *(10_000 + seed_index * 20 + offset for offset in range(5, 9)),
        )
        for track_id in {seed_id, *audio[seed_id], *challenger[seed_id]}:
            tracks[str(track_id)] = {
                "track_id": track_id,
                "source_identity": {"artist_id": track_id},
            }
    ordered = pacing.prioritize_source_seeds(source_seeds, audio, challenger)
    return pacing.build_blinded_documents(
        source_seeds=ordered,
        track_records=tracks,
        audio_rankings=audio,
        pacing_rankings=challenger,
        store_binding=committed["store_binding"],
        blinding_key=b"\x42" * 32,
    )


def test_blinded_documents_synchronize_duplicate_results_without_public_methods():
    public, private = _synthetic_documents()
    pacing.validate_blinded_documents(public, private, require_frozen_artifacts=False)
    text = json.dumps(public, sort_keys=True)
    assert not any(method in text for method in pacing.METHODS)
    assert private["methods"] == list(pacing.METHODS)
    seed = public["seeds"][0]
    shared_track = set(
        row["track_id"] for row in seed["lists"][0]["ranking"]
    ) & set(row["track_id"] for row in seed["lists"][1]["ranking"])
    assert len(shared_track) == 1
    track_id = shared_track.pop()
    ids = {
        row["result_id"]
        for candidate in seed["lists"]
        for row in candidate["ranking"]
        if row["track_id"] == track_id
    }
    assert len(ids) == 1


def _rehash(public, private):
    private["content_sha256"] = _content_sha256(private)
    public["blinding"]["private_unblinding_sha256"] = private["content_sha256"]
    public["content_sha256"] = _content_sha256(public)


def test_blinded_documents_reject_rehashed_tampering():
    public, private = map(copy.deepcopy, _synthetic_documents())
    public["seeds"][0]["lists"][0]["ranking"][0]["result_id"] = (
        "pacing-result-" + "f" * 24
    )
    public["content_sha256"] = _content_sha256(public)
    with pytest.raises(pacing.PacingEvalError, match="result identity"):
        pacing.validate_blinded_documents(
            public, private, require_frozen_artifacts=False
        )

    public, private = map(copy.deepcopy, _synthetic_documents())
    private["method_bindings"][pacing.METHODS[1]]["reranking"]["weights"]["pacing"] = 0.11
    _rehash(public, private)
    with pytest.raises(pacing.PacingEvalError, match="method binding"):
        pacing.validate_blinded_documents(
            public, private, require_frozen_artifacts=False
        )


def test_committed_pack_is_frozen_blinded_and_prioritized():
    document = json.loads(PUBLIC.read_text(encoding="utf-8"))
    assert document["content_sha256"] == pacing.EXPECTED_PUBLIC_PACK_SHA256
    assert _content_sha256(document) == pacing.EXPECTED_PUBLIC_PACK_SHA256
    assert document["provenance"]["signal_dimensions"] == 29
    assert document["provenance"]["ratings_used"] is False
    assert document["research_only"] is True
    assert document["promotion_allowed"] is False
    assert document["production_recommendation_changed"] is False
    assert [seed["priority_rank"] for seed in document["seeds"]] == list(range(1, 21))
    overlaps = [
        (seed["matched_list_overlap"], seed["seed_track_id"])
        for seed in document["seeds"]
    ]
    assert overlaps == sorted(overlaps)
    text = PUBLIC.read_text(encoding="utf-8")
    assert not any(method in text for method in pacing.METHODS)


@pytest.mark.skipif(not PRIVATE.exists(), reason="private unblinding is not portable")
def test_generated_private_artifact_verifies_against_committed_pack():
    pacing.validate_blinded_documents(
        json.loads(PUBLIC.read_text(encoding="utf-8")),
        json.loads(PRIVATE.read_text(encoding="utf-8")),
    )


def test_semantic_v2_archive_is_byte_preserved_and_core_is_isolated():
    expected = {
        "index.html": "cd5f5c553eef7264a75a4bd80ff2987e39f8a1363a4bf7c7dccd8c7056960e85",
        "pilot-pack.json": "d23d66768f15fd5e37e01ad2a8905d181b4ff278c85674386edcd7dc50b267d3",
        "protocol-semantic-v2.json": "36919e57883fb54028c98e495431638edaecd899d533a0523026d3ce81fdaa20",
        "protocol-v2.json": "a88108894e3875159a9ae5b3fae61b01522e9c22647d9ff32748d53d0a5c981c",
        "semantic-pack.json": "f07bf814eab2a363aa9fbec5acd946e57cfad3d3c3eef6dea4027a190d0e13b3",
    }
    assert {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in ARCHIVE.iterdir()
    } == expected
    source = inspect.getsource(pacing)
    assert "from .semantic_eval import" not in source
    assert "listener_ratings_used_for_ranking" in source
    assert "webapp/evaluate-semantic-v2/* -text -whitespace" in (
        ROOT / ".gitattributes"
    ).read_text(encoding="utf-8")
    assert (ROOT / "webapp" / ".vercelignore").read_text(encoding="utf-8").splitlines() == [
        "test/",
        "tools/",
    ]
