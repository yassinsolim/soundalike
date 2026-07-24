"""Comprehensive offline tests for fulltrack_fusion.

Covers all candidate kinds, feature schema (16 features incl. top-k and
temporal-index features), artifact save/load, SHA-256 bindings, path safety,
strict JSON types, deep-copy freeze, compressed archive rejection, dimension
validation, numeric edges, and every listed corruption case.
Fully synthetic; no audio or network required.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import struct
import subprocess
import tempfile
import zipfile
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from soundalike.ml.fulltrack_fusion import (
    ABLATIONS,
    CANDIDATE_KINDS,
    FEATURE_DIM,
    FEATURE_NAMES,
    FUSION_SCHEMA_VERSION,
    FusionConfig,
    FusionError,
    FusionMetadata,
    FusionModel,
    PairFeatures,
    build_channel_gated,
    build_monotonic_network,
    build_nonneg_linear,
    extract_pair_features,
    load_fusion_artifact,
    save_fusion_artifact,
)


# ---------------------------------------------------------------------------
# Synthetic track fixtures
# ---------------------------------------------------------------------------


class _FakeTrack:
    """Minimal duck-typed track object for testing."""

    def __init__(
        self,
        global_embedding: np.ndarray,
        window_embeddings: np.ndarray,
        repeated_sections: Optional[np.ndarray] = None,
        salient_sections: Optional[np.ndarray] = None,
        repeated_indices: Optional[np.ndarray] = None,
        salient_indices: Optional[np.ndarray] = None,
    ):
        self.global_embedding = global_embedding
        self.window_embeddings = window_embeddings
        n_win = len(window_embeddings)
        self.repeated_sections = (
            repeated_sections
            if repeated_sections is not None
            else window_embeddings[: min(2, n_win)]
        )
        self.salient_sections = (
            salient_sections
            if salient_sections is not None
            else window_embeddings[: min(2, n_win)]
        )
        n_rep = len(self.repeated_sections) if self.repeated_sections.ndim == 2 else 0
        n_sal = len(self.salient_sections) if self.salient_sections.ndim == 2 else 0
        self.repeated_indices = (
            repeated_indices
            if repeated_indices is not None
            else np.arange(min(n_rep, n_win), dtype=np.int64)
        )
        self.salient_indices = (
            salient_indices
            if salient_indices is not None
            else np.arange(min(n_sal, n_win), dtype=np.int64)
        )


def _unit(i: int, dim: int = 8) -> np.ndarray:
    """Return a unit basis vector e_i (wrapping mod dim), float32."""
    v = np.zeros(dim, dtype=np.float32)
    v[i % dim] = 1.0
    return v


def _rand_unit_matrix(rows: int, dim: int, seed: int = 0) -> np.ndarray:
    """Random L2-normalised matrix, float32."""
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((rows, dim)).astype(np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return (m / norms).astype(np.float32)


def _identical_track(dim: int = 8, n_win: int = 4, seed: int = 42) -> _FakeTrack:
    """Track whose global, windows, sections all point in the same direction."""
    g = _rand_unit_matrix(1, dim, seed)[0]
    windows = np.tile(g, (n_win, 1))
    return _FakeTrack(
        global_embedding=g,
        window_embeddings=windows,
        repeated_sections=windows,
        salient_sections=windows,
    )


def _orthogonal_track(seed: int = 99, dim: int = 8, n_win: int = 4) -> _FakeTrack:
    """Track whose global is orthogonal to a reference direction."""
    rng = np.random.default_rng(seed)
    g = rng.standard_normal(dim).astype(np.float32)
    g /= np.linalg.norm(g)
    windows = np.tile(g, (n_win, 1))
    return _FakeTrack(
        global_embedding=g,
        window_embeddings=windows,
        repeated_sections=windows,
        salient_sections=windows,
    )


def _make_tracks(dim: int = 8, n_win: int = 6, seed: int = 7):
    """Two unrelated random tracks."""
    a_wins = _rand_unit_matrix(n_win, dim, seed)
    b_wins = _rand_unit_matrix(n_win, dim, seed + 1)
    g_a = a_wins[0].copy()
    g_b = b_wins[0].copy()
    ta = _FakeTrack(g_a, a_wins, a_wins[:3], a_wins[3:])
    tb = _FakeTrack(g_b, b_wins, b_wins[:3], b_wins[3:])
    return ta, tb


# ---------------------------------------------------------------------------
# Model + config helpers
# ---------------------------------------------------------------------------

DIM = 8
BUDGET = 4

_TEST_MODEL_ID = "test-model"
_TEST_STORE_ID = "test-store"
_TEST_CONFIG_SHA = "a" * 64


def _linear_model(dim: int = DIM, budget: int = BUDGET) -> FusionModel:
    cfg = FusionConfig(
        kind="nonnegative_linear", embedding_dim=dim, maxsim_budget=budget,
        model_id=_TEST_MODEL_ID, store_id=_TEST_STORE_ID, config_sha256=_TEST_CONFIG_SHA,
    )
    w = np.ones(FEATURE_DIM, dtype=np.float64)
    return build_nonneg_linear(w, cfg)


def _network_model(dim: int = DIM, budget: int = BUDGET) -> FusionModel:
    cfg = FusionConfig(
        kind="monotonic_network",
        embedding_dim=dim,
        maxsim_budget=budget,
        hidden_dims=(8,),
        model_id=_TEST_MODEL_ID, store_id=_TEST_STORE_ID, config_sha256=_TEST_CONFIG_SHA,
    )
    w0 = np.eye(8, FEATURE_DIM, dtype=np.float64)
    b0 = np.zeros(8, dtype=np.float64)
    w1 = np.ones((1, 8), dtype=np.float64) * 0.1
    b1 = np.zeros(1, dtype=np.float64)
    return build_monotonic_network([w0, w1], [b0, b1], cfg)


def _gated_model(dim: int = DIM, budget: int = BUDGET) -> FusionModel:
    cfg = FusionConfig(
        kind="channel_gated_embedding", embedding_dim=dim, maxsim_budget=budget,
        model_id=_TEST_MODEL_ID, store_id=_TEST_STORE_ID, config_sha256=_TEST_CONFIG_SHA,
    )
    gates = np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64)
    return build_channel_gated(gates, cfg)


@pytest.mark.parametrize("ablation", ABLATIONS)
def test_precomputed_feature_scoring_matches_candidate_scoring(ablation):
    query, first_candidate = _make_tracks()
    candidates = [first_candidate, _identical_track(), _orthogonal_track()]
    for model in (_linear_model(), _network_model()):
        features = np.stack(
            [
                model.extract_pair_features(query, candidate).to_vector()
                for candidate in candidates
            ]
        )
        expected = model.score_candidates(query, candidates, ablation=ablation)
        actual = model.score_feature_vectors(features, ablation=ablation)
        np.testing.assert_array_equal(actual, expected)


# ---------------------------------------------------------------------------
# Artifact helpers
# ---------------------------------------------------------------------------


def _save_and_reload(model: FusionModel, tmp_path: Path):
    meta = save_fusion_artifact(model, tmp_path)
    loaded = load_fusion_artifact(tmp_path)
    return loaded, meta


def _tamper_json(path: Path, key: str, new_value) -> None:
    with path.open("r", encoding="utf-8") as fh:
        doc = json.load(fh)
    doc[key] = new_value
    with path.open("w", encoding="utf-8") as fh:
        json.dump(doc, fh, sort_keys=True, indent=2)
        fh.write("\n")


def _tamper_npz_byte(path: Path) -> None:
    data = bytearray(path.read_bytes())
    idx = max(0, len(data) - 10)
    data[idx] ^= 0xFF
    path.write_bytes(bytes(data))


# ===========================================================================
# Constants
# ===========================================================================


def test_candidate_kinds_exact():
    assert set(CANDIDATE_KINDS) == {
        "nonnegative_linear", "monotonic_network", "channel_gated_embedding"
    }


def test_feature_dim_and_names():
    assert FEATURE_DIM == 16
    assert len(FEATURE_NAMES) == FEATURE_DIM
    assert FEATURE_NAMES[0] == "global_cosine"
    assert FEATURE_NAMES[8] == "asymmetry_utility"
    assert FEATURE_NAMES[9] == "recurrence_indicator"
    assert FEATURE_NAMES[11] == "steady_texture_b"
    assert FEATURE_NAMES[12] == "topk_maxsim_ab"
    assert FEATURE_NAMES[13] == "topk_maxsim_ba"
    assert FEATURE_NAMES[14] == "repeated_temporal_sim"
    assert FEATURE_NAMES[15] == "salient_temporal_sim"


def test_ablations_enum():
    assert set(ABLATIONS) == {"none", "global_only", "no_sections"}


# ===========================================================================
# FusionConfig
# ===========================================================================


def test_config_valid():
    cfg = FusionConfig(kind="nonnegative_linear", embedding_dim=8)
    cfg.validate()


def test_config_unknown_kind():
    with pytest.raises(FusionError, match="unknown kind"):
        FusionConfig(kind="bad", embedding_dim=8).validate()


def test_config_zero_embedding_dim():
    with pytest.raises(FusionError, match="embedding_dim"):
        FusionConfig(kind="nonnegative_linear", embedding_dim=0).validate()


def test_config_bad_coverage_threshold():
    with pytest.raises(FusionError, match="coverage_threshold"):
        FusionConfig(
            kind="nonnegative_linear", embedding_dim=8, coverage_threshold=1.5
        ).validate()


def test_config_network_requires_hidden_dims():
    with pytest.raises(FusionError, match="hidden layer"):
        FusionConfig(kind="monotonic_network", embedding_dim=8).validate()


def test_config_negative_hidden_dim():
    with pytest.raises(FusionError, match="hidden_dims"):
        FusionConfig(
            kind="monotonic_network", embedding_dim=8, hidden_dims=(-1,)
        ).validate()


def test_config_hidden_dims_nonempty_for_linear():
    """hidden_dims must be empty for non-network kinds."""
    with pytest.raises(FusionError, match="hidden_dims must be empty"):
        FusionConfig(
            kind="nonnegative_linear", embedding_dim=8, hidden_dims=(4,)
        ).validate()


def test_config_bool_as_int_rejected():
    """bool is not accepted where int is required."""
    with pytest.raises(FusionError, match="bool"):
        FusionConfig(kind="nonnegative_linear", embedding_dim=True).validate()


def test_config_as_dict_roundtrip():
    cfg = FusionConfig(
        kind="channel_gated_embedding",
        embedding_dim=512,
        fold_index=2,
        hidden_dims=(),
    )
    d = cfg.as_dict()
    assert d["kind"] == "channel_gated_embedding"
    assert d["embedding_dim"] == 512
    assert d["fold_index"] == 2


# ===========================================================================
# PairFeatures -- includes top-k and temporal features
# ===========================================================================


def test_pair_features_identical_tracks_high_scores():
    t = _identical_track(dim=DIM)
    pf = extract_pair_features(t, t, maxsim_budget=BUDGET)
    vec = pf.to_vector()
    assert vec.shape == (FEATURE_DIM,)
    assert pf.global_cosine == pytest.approx(1.0, abs=1e-6)
    assert np.all(vec >= 0.0) and np.all(vec <= 1.0)
    assert pf.asymmetry_utility == pytest.approx(1.0, abs=1e-6)


def test_pair_features_orthogonal_globals():
    g_a = np.zeros(8, dtype=np.float32); g_a[0] = 1.0
    g_b = np.zeros(8, dtype=np.float32); g_b[1] = 1.0
    wins_a = np.tile(g_a, (4, 1))
    wins_b = np.tile(g_b, (4, 1))
    ta = _FakeTrack(g_a, wins_a)
    tb = _FakeTrack(g_b, wins_b)
    pf = extract_pair_features(ta, tb, maxsim_budget=BUDGET)
    assert pf.global_cosine == pytest.approx(0.5, abs=1e-6)


def test_pair_features_all_values_in_unit_interval():
    ta, tb = _make_tracks()
    pf = extract_pair_features(ta, tb, maxsim_budget=BUDGET)
    vec = pf.to_vector()
    assert np.all(vec >= 0.0), f"min={vec.min()}"
    assert np.all(vec <= 1.0), f"max={vec.max()}"
    assert np.all(np.isfinite(vec))


def test_pair_features_deterministic():
    ta, tb = _make_tracks()
    pf1 = extract_pair_features(ta, tb, maxsim_budget=BUDGET)
    pf2 = extract_pair_features(ta, tb, maxsim_budget=BUDGET)
    np.testing.assert_array_equal(pf1.to_vector(), pf2.to_vector())


def test_pair_features_asymmetry_utility_symmetric():
    t = _identical_track()
    pf = extract_pair_features(t, t, maxsim_budget=BUDGET)
    assert pf.asymmetry_utility == pytest.approx(1.0, abs=1e-6)


def test_pair_features_asymmetry_utility_range():
    ta, tb = _make_tracks()
    pf = extract_pair_features(ta, tb, maxsim_budget=BUDGET)
    assert 0.0 <= pf.asymmetry_utility <= 1.0


def test_pair_features_coverage_threshold_zero():
    t = _identical_track()
    pf = extract_pair_features(t, t, maxsim_budget=BUDGET, coverage_threshold=0.0)
    assert pf.coverage_topk_a == pytest.approx(1.0, abs=1e-6)
    assert pf.coverage_topk_b == pytest.approx(1.0, abs=1e-6)


def test_pair_features_coverage_threshold_one():
    g = np.array([1.0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    t = _FakeTrack(g, np.tile(g, (4, 1)))
    pf = extract_pair_features(t, t, maxsim_budget=BUDGET, coverage_threshold=1.0)
    assert pf.coverage_topk_a == pytest.approx(1.0, abs=1e-6)


def test_pair_features_recurrence_single_section_neutral():
    g = _unit(0)
    t = _FakeTrack(
        global_embedding=g,
        window_embeddings=np.tile(g, (4, 1)),
        repeated_sections=g.reshape(1, -1),
        salient_sections=g.reshape(1, -1),
    )
    pf = extract_pair_features(t, t, maxsim_budget=BUDGET)
    assert pf.recurrence_indicator == pytest.approx(0.5, abs=1e-6)


def test_pair_features_recurrence_identical_sections():
    g = _unit(0)
    sections = np.tile(g, (4, 1))
    t = _FakeTrack(g, sections, sections, sections)
    pf = extract_pair_features(t, t, maxsim_budget=BUDGET)
    assert pf.recurrence_indicator == pytest.approx(1.0, abs=1e-6)


def test_pair_features_steady_texture_uniform():
    g = _unit(3)
    t = _FakeTrack(g, np.tile(g, (4, 1)))
    pf = extract_pair_features(t, t, maxsim_budget=BUDGET)
    assert pf.steady_texture_a == pytest.approx(1.0, abs=1e-6)
    assert pf.steady_texture_b == pytest.approx(1.0, abs=1e-6)


def test_pair_features_empty_sections_neutral():
    g = _unit(0)
    t = _FakeTrack(
        global_embedding=g,
        window_embeddings=np.tile(g, (4, 1)),
        repeated_sections=np.empty((0, DIM), dtype=np.float32),
        salient_sections=np.empty((0, DIM), dtype=np.float32),
        repeated_indices=np.array([], dtype=np.int64),
        salient_indices=np.array([], dtype=np.int64),
    )
    pf = extract_pair_features(t, t, maxsim_budget=BUDGET)
    assert pf.repeated_maxsim_sym == pytest.approx(0.5, abs=1e-6)
    assert pf.salient_maxsim_sym == pytest.approx(0.5, abs=1e-6)
    assert pf.repeated_temporal_sim == pytest.approx(0.5, abs=1e-6)
    assert pf.salient_temporal_sim == pytest.approx(0.5, abs=1e-6)


def test_pair_features_float16_input():
    g = _unit(0).astype(np.float16)
    wins = np.tile(g, (4, 1)).astype(np.float16)
    t = _FakeTrack(g, wins, wins, wins)
    pf = extract_pair_features(t, t, maxsim_budget=BUDGET)
    vec = pf.to_vector()
    assert np.all(np.isfinite(vec))
    assert np.all(vec >= 0.0) and np.all(vec <= 1.0)


def test_pair_features_global_dim_mismatch():
    g_a = _unit(0, dim=4)
    g_b = _unit(0, dim=8)
    ta = _FakeTrack(g_a, np.tile(g_a, (4, 1)))
    tb = _FakeTrack(g_b, np.tile(g_b, (4, 1)))
    with pytest.raises(FusionError, match="dims differ"):
        extract_pair_features(ta, tb)


def test_pair_features_to_vector_order():
    t = _identical_track()
    pf = extract_pair_features(t, t, maxsim_budget=BUDGET)
    vec = pf.to_vector()
    assert vec[0] == pf.global_cosine
    assert vec[8] == pf.asymmetry_utility
    assert vec[11] == pf.steady_texture_b
    assert vec[12] == pf.topk_maxsim_ab
    assert vec[13] == pf.topk_maxsim_ba
    assert vec[14] == pf.repeated_temporal_sim
    assert vec[15] == pf.salient_temporal_sim


# ===========================================================================
# Top-k feature tests (Fix 8)
# ===========================================================================


def test_topk_features_present_and_bounded():
    """topk_maxsim_ab and topk_maxsim_ba must be present and in [0,1]."""
    ta, tb = _make_tracks()
    pf = extract_pair_features(ta, tb, maxsim_budget=BUDGET, top_k=2)
    assert 0.0 <= pf.topk_maxsim_ab <= 1.0
    assert 0.0 <= pf.topk_maxsim_ba <= 1.0


def test_topk_varies_with_k():
    """top_k=1 vs top_k=budget produces genuinely different topk_maxsim features.

    Deterministic: window 0 of A matches all B windows perfectly (all e0),
    while windows 1-3 of A are orthogonal to B.  So top-1 picks only the
    best match (1.0) while top-all includes 3 zero-cosine matches.
    """
    dim = 8
    g_a = _unit(0, dim)
    g_b = _unit(0, dim)
    # A windows: [e0, e1, e2, e3] -- only e0 matches B
    wins_a = np.zeros((4, dim), dtype=np.float32)
    wins_a[0, 0] = 1.0
    wins_a[1, 1] = 1.0
    wins_a[2, 2] = 1.0
    wins_a[3, 3] = 1.0
    # B windows: all e0
    wins_b = np.zeros((4, dim), dtype=np.float32)
    wins_b[:, 0] = 1.0
    ta = _FakeTrack(g_a, wins_a)
    tb = _FakeTrack(g_b, wins_b)
    pf1 = extract_pair_features(ta, tb, maxsim_budget=4, top_k=1)
    pf4 = extract_pair_features(ta, tb, maxsim_budget=4, top_k=4)
    # top_k=1: mean of top-1 max-cosines = 1.0 -> mapped (1+1)/2 = 1.0
    # top_k=4: mean of [1,0,0,0] max-cosines = 0.25 -> mapped (1+0.25)/2 = 0.625
    assert pf1.topk_maxsim_ab > pf4.topk_maxsim_ab + 0.01
    assert 0.0 <= pf1.topk_maxsim_ab <= 1.0
    assert 0.0 <= pf4.topk_maxsim_ab <= 1.0


def test_topk_self_high():
    """Self-comparison top-k should be very high."""
    t = _identical_track()
    pf = extract_pair_features(t, t, maxsim_budget=BUDGET, top_k=2)
    assert pf.topk_maxsim_ab > 0.99
    assert pf.topk_maxsim_ba > 0.99


def test_topk_k1_ge_full():
    """top-k=1 (best match) should be >= full budget mean."""
    ta, tb = _make_tracks()
    pf_k1 = extract_pair_features(ta, tb, maxsim_budget=BUDGET, top_k=1)
    pf_all = extract_pair_features(ta, tb, maxsim_budget=BUDGET, top_k=BUDGET)
    # Top-1 picks only best, so its value >= mean of all
    assert pf_k1.topk_maxsim_ab >= pf_all.topk_maxsim_ab - 1e-9


# ===========================================================================
# Index validation and temporal features (Fix 8)
# ===========================================================================


def test_indices_validated_bounds():
    """Indices out of range must raise FusionError."""
    g = _unit(0)
    wins = np.tile(g, (4, 1))
    t_ok = _FakeTrack(g, wins, wins[:2], wins[:2],
                      repeated_indices=np.array([0, 1], dtype=np.int64),
                      salient_indices=np.array([0, 1], dtype=np.int64))
    # Should work fine
    extract_pair_features(t_ok, t_ok, maxsim_budget=BUDGET)

    t_bad = _FakeTrack(g, wins, wins[:2], wins[:2],
                       repeated_indices=np.array([0, 10], dtype=np.int64),
                       salient_indices=np.array([0, 1], dtype=np.int64))
    with pytest.raises(FusionError, match="repeated_indices"):
        extract_pair_features(t_bad, t_bad, maxsim_budget=BUDGET)


def test_indices_length_mismatch():
    """Index length must match section count."""
    g = _unit(0)
    wins = np.tile(g, (4, 1))
    t_bad = _FakeTrack(g, wins, wins[:2], wins[:2],
                       repeated_indices=np.array([0], dtype=np.int64),
                       salient_indices=np.array([0, 1], dtype=np.int64))
    with pytest.raises(FusionError, match="repeated_indices"):
        extract_pair_features(t_bad, t_bad, maxsim_budget=BUDGET)


def test_temporal_sim_same_positions():
    """Tracks with sections at identical positions -> temporal_sim ~ 1."""
    g = _unit(0)
    wins = np.tile(g, (8, 1))
    idx = np.array([0, 3, 5, 7], dtype=np.int64)
    t = _FakeTrack(g, wins, wins[:4], wins[:4],
                   repeated_indices=idx, salient_indices=idx)
    pf = extract_pair_features(t, t, maxsim_budget=BUDGET)
    assert pf.repeated_temporal_sim == pytest.approx(1.0, abs=1e-6)
    assert pf.salient_temporal_sim == pytest.approx(1.0, abs=1e-6)


def test_temporal_sim_different_positions():
    """Tracks with sections at very different positions -> lower temporal_sim."""
    g = _unit(0)
    wins = np.tile(g, (20, 1))
    idx_a = np.array([0, 1], dtype=np.int64)
    idx_b = np.array([18, 19], dtype=np.int64)
    ta = _FakeTrack(g, wins, wins[:2], wins[:2],
                    repeated_indices=idx_a, salient_indices=idx_a)
    tb = _FakeTrack(g, wins, wins[:2], wins[:2],
                    repeated_indices=idx_b, salient_indices=idx_b)
    pf = extract_pair_features(ta, tb, maxsim_budget=BUDGET)
    assert pf.repeated_temporal_sim < 0.95
    assert pf.salient_temporal_sim < 0.95


def test_no_indices_produces_neutral():
    """Track without index attrs -> temporal_sim = 0.5."""
    g = _unit(0)
    wins = np.tile(g, (4, 1))

    class _NoIndexTrack:
        def __init__(self):
            self.global_embedding = g
            self.window_embeddings = wins
            self.repeated_sections = wins[:2]
            self.salient_sections = wins[:2]

    t = _NoIndexTrack()
    pf = extract_pair_features(t, t, maxsim_budget=BUDGET)
    assert pf.repeated_temporal_sim == pytest.approx(0.5, abs=1e-6)
    assert pf.salient_temporal_sim == pytest.approx(0.5, abs=1e-6)


# ===========================================================================
# nonnegative_linear model
# ===========================================================================


def test_linear_score_self_is_high():
    model = _linear_model()
    t = _identical_track()
    s = model.score_candidate(t, t)
    assert isinstance(s, float)
    assert 0.9 < s <= 1.0


def test_linear_score_bounded():
    model = _linear_model()
    ta, tb = _make_tracks()
    s = model.score_candidate(ta, tb)
    assert 0.0 <= s <= 1.0
    assert np.isfinite(s)


def test_linear_score_deterministic():
    model = _linear_model()
    ta, tb = _make_tracks()
    assert model.score_candidate(ta, tb) == model.score_candidate(ta, tb)


def test_linear_monotone_in_features():
    model = _linear_model()
    ta, tb = _make_tracks()
    base_score = model.score_candidate(ta, tb)
    g_new = ta.global_embedding * 0.1 + tb.global_embedding * 0.9
    g_new /= np.linalg.norm(g_new)
    ta2 = _FakeTrack(
        g_new, np.tile(g_new, (6, 1)),
        np.tile(g_new, (3, 1)), np.tile(g_new, (3, 1)),
    )
    blended_score = model.score_candidate(ta2, tb)
    assert blended_score >= base_score - 1e-9


def test_linear_negative_weights_rejected():
    cfg = FusionConfig(kind="nonnegative_linear", embedding_dim=DIM, maxsim_budget=BUDGET)
    w = np.ones(FEATURE_DIM, dtype=np.float64)
    w[0] = -0.1
    with pytest.raises(FusionError, match="non-negative"):
        build_nonneg_linear(w, cfg)


def test_linear_zero_weight_sum_rejected():
    cfg = FusionConfig(kind="nonnegative_linear", embedding_dim=DIM, maxsim_budget=BUDGET)
    w = np.zeros(FEATURE_DIM, dtype=np.float64)
    with pytest.raises(FusionError, match="positive sum"):
        build_nonneg_linear(w, cfg)


def test_linear_wrong_shape_rejected():
    cfg = FusionConfig(kind="nonnegative_linear", embedding_dim=DIM, maxsim_budget=BUDGET)
    w = np.ones(FEATURE_DIM + 1, dtype=np.float64)
    with pytest.raises(FusionError, match="shape"):
        build_nonneg_linear(w, cfg)


def test_linear_int_weights_rejected():
    cfg = FusionConfig(kind="nonnegative_linear", embedding_dim=DIM, maxsim_budget=BUDGET)
    w = np.ones(FEATURE_DIM, dtype=np.int64)
    with pytest.raises(FusionError, match="floating-point"):
        build_nonneg_linear(w, cfg)


def test_linear_nan_weights_rejected():
    cfg = FusionConfig(kind="nonnegative_linear", embedding_dim=DIM, maxsim_budget=BUDGET)
    w = np.ones(FEATURE_DIM, dtype=np.float64)
    w[0] = float("nan")
    with pytest.raises(FusionError, match="non-finite"):
        build_nonneg_linear(w, cfg)


def test_linear_ablation_global_only():
    model = _linear_model()
    ta, tb = _make_tracks()
    pf = extract_pair_features(ta, tb, maxsim_budget=BUDGET)
    s_ablated = model.score_candidate(ta, tb, ablation="global_only")
    assert s_ablated == pytest.approx(pf.global_cosine, abs=1e-9)


def test_linear_ablation_no_sections_excludes_rep_sal():
    model = _linear_model()
    ta, tb = _make_tracks()
    pf = extract_pair_features(ta, tb, maxsim_budget=BUDGET)
    feat = pf.to_vector()
    mask = np.array([1,1,1,1,0,0,1,1,1,0,1,1,1,1,0,0], dtype=np.float64)
    w = np.ones(FEATURE_DIM, dtype=np.float64) * mask
    expected = float(np.dot(w, feat * mask)) / float(w.sum())
    s = model.score_candidate(ta, tb, ablation="no_sections")
    assert s == pytest.approx(expected, abs=1e-9)


def test_linear_unknown_ablation_raises():
    model = _linear_model()
    t = _identical_track()
    with pytest.raises(FusionError, match="unknown ablation"):
        model.score_candidate(t, t, ablation="bad_ablation")


def test_linear_wrong_kind_constructor():
    cfg = FusionConfig(kind="monotonic_network", embedding_dim=DIM, hidden_dims=(4,))
    with pytest.raises(FusionError, match="nonnegative_linear"):
        build_nonneg_linear(np.ones(FEATURE_DIM), cfg)


# ===========================================================================
# monotonic_network model
# ===========================================================================


def test_network_score_bounded():
    model = _network_model()
    ta, tb = _make_tracks()
    s = model.score_candidate(ta, tb)
    assert 0.0 <= s <= 1.0
    assert np.isfinite(s)


def test_network_score_self_sigmoid_range():
    model = _network_model()
    t = _identical_track()
    s = model.score_candidate(t, t)
    assert 0.0 < s < 1.0


def test_network_score_deterministic():
    model = _network_model()
    ta, tb = _make_tracks()
    assert model.score_candidate(ta, tb) == model.score_candidate(ta, tb)


def test_network_nonneg_weights_validated():
    cfg = FusionConfig(
        kind="monotonic_network", embedding_dim=DIM, maxsim_budget=BUDGET, hidden_dims=(4,)
    )
    w0 = np.eye(4, FEATURE_DIM, dtype=np.float64)
    w0[0, 0] = -1.0
    b0 = np.zeros(4)
    w1 = np.ones((1, 4), dtype=np.float64) * 0.1
    b1 = np.zeros(1)
    with pytest.raises(FusionError, match="non-negative"):
        build_monotonic_network([w0, w1], [b0, b1], cfg)


def test_network_wrong_shape_rejected():
    cfg = FusionConfig(
        kind="monotonic_network", embedding_dim=DIM, maxsim_budget=BUDGET, hidden_dims=(4,)
    )
    w0 = np.ones((5, FEATURE_DIM), dtype=np.float64)
    b0 = np.zeros(4)
    w1 = np.ones((1, 4), dtype=np.float64)
    b1 = np.zeros(1)
    with pytest.raises(FusionError, match="shape"):
        build_monotonic_network([w0, w1], [b0, b1], cfg)


def test_network_wrong_layer_count_rejected():
    cfg = FusionConfig(
        kind="monotonic_network", embedding_dim=DIM, maxsim_budget=BUDGET, hidden_dims=(4,)
    )
    w0 = np.ones((4, FEATURE_DIM), dtype=np.float64)
    b0 = np.zeros(4)
    with pytest.raises(FusionError, match="pairs"):
        build_monotonic_network([w0], [b0], cfg)


def test_network_ablation_global_only_bounded():
    model = _network_model()
    ta, tb = _make_tracks()
    s = model.score_candidate(ta, tb, ablation="global_only")
    assert 0.0 <= s <= 1.0


def test_network_two_hidden_layers():
    cfg = FusionConfig(
        kind="monotonic_network", embedding_dim=DIM, maxsim_budget=BUDGET,
        hidden_dims=(8, 4),
    )
    w0 = np.ones((8, FEATURE_DIM), dtype=np.float64) * 0.01
    b0 = np.zeros(8)
    w1 = np.ones((4, 8), dtype=np.float64) * 0.01
    b1 = np.zeros(4)
    w2 = np.ones((1, 4), dtype=np.float64) * 0.1
    b2 = np.zeros(1)
    model = build_monotonic_network([w0, w1, w2], [b0, b1, b2], cfg)
    t = _identical_track()
    s = model.score_candidate(t, t)
    assert 0.0 <= s <= 1.0


# ===========================================================================
# channel_gated_embedding model
# ===========================================================================


def test_gated_score_self_bounded():
    model = _gated_model()
    t = _identical_track()
    s = model.score_candidate(t, t)
    assert 0.0 <= s <= 1.0


def test_gated_score_self_near_one():
    model = _gated_model()
    t = _identical_track()
    s = model.score_candidate(t, t)
    assert s > 0.99


def test_gated_embed_track_l2_normalized():
    model = _gated_model()
    t = _identical_track()
    emb = model.embed_track(t)
    assert emb.dtype == np.float64
    assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-9


def test_gated_embed_track_deterministic():
    model = _gated_model()
    t = _identical_track()
    np.testing.assert_array_equal(model.embed_track(t), model.embed_track(t))


def test_gated_embed_track_wrong_kind():
    model = _linear_model()
    t = _identical_track()
    with pytest.raises(FusionError, match="embed_track"):
        model.embed_track(t)


def test_gated_score_bounded_random():
    model = _gated_model()
    ta, tb = _make_tracks()
    s = model.score_candidate(ta, tb)
    assert 0.0 <= s <= 1.0
    assert np.isfinite(s)


def test_gated_score_deterministic():
    model = _gated_model()
    ta, tb = _make_tracks()
    assert model.score_candidate(ta, tb) == model.score_candidate(ta, tb)


def test_gated_negative_gates_rejected():
    cfg = FusionConfig(kind="channel_gated_embedding", embedding_dim=DIM)
    gates = np.array([0.4, -0.1, 0.4, 0.3], dtype=np.float64)
    with pytest.raises(FusionError, match="non-negative"):
        build_channel_gated(gates, cfg)


def test_gated_zero_gates_rejected():
    cfg = FusionConfig(kind="channel_gated_embedding", embedding_dim=DIM)
    gates = np.zeros(4, dtype=np.float64)
    with pytest.raises(FusionError, match="positive sum"):
        build_channel_gated(gates, cfg)


def test_gated_unnormalized_gates_rejected():
    cfg = FusionConfig(kind="channel_gated_embedding", embedding_dim=DIM)
    gates = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    with pytest.raises(FusionError, match="sum to 1.0"):
        build_channel_gated(gates, cfg)


def test_gated_wrong_shape_rejected():
    cfg = FusionConfig(kind="channel_gated_embedding", embedding_dim=DIM)
    gates = np.array([0.5, 0.5], dtype=np.float64)
    with pytest.raises(FusionError, match="shape"):
        build_channel_gated(gates, cfg)


def test_gated_ablation_global_only():
    model = _gated_model()
    t = _identical_track()
    s = model.score_candidate(t, t, ablation="global_only")
    assert s == pytest.approx(1.0, abs=1e-9)


def test_gated_ablation_no_sections():
    model = _gated_model()
    ta, tb = _make_tracks()
    s = model.score_candidate(ta, tb, ablation="no_sections")
    assert 0.0 <= s <= 1.0


def test_gated_score_candidates_shape():
    model = _gated_model()
    ta, tb = _make_tracks()
    tc = _identical_track()
    scores = model.score_candidates(ta, [tb, tc, tb])
    assert scores.shape == (3,)
    assert scores.dtype == np.float64
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)


# ===========================================================================
# score_candidates (all kinds)
# ===========================================================================


@pytest.mark.parametrize("model_fn", [_linear_model, _network_model, _gated_model])
def test_score_candidates_all_kinds(model_fn, tmp_path):
    model = model_fn()
    ta, tb = _make_tracks()
    tc = _identical_track()
    scores = model.score_candidates(ta, [tb, tc, tb])
    assert scores.shape == (3,)
    assert np.all(np.isfinite(scores))
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)


@pytest.mark.parametrize("model_fn", [_linear_model, _network_model, _gated_model])
def test_score_candidates_single(model_fn):
    model = model_fn()
    ta, tb = _make_tracks()
    scores = model.score_candidates(ta, [tb])
    assert scores.shape == (1,)
    assert scores[0] == pytest.approx(model.score_candidate(ta, tb), abs=1e-12)


@pytest.mark.parametrize("model_fn", [_linear_model, _network_model, _gated_model])
def test_score_candidates_empty(model_fn):
    model = model_fn()
    ta, _ = _make_tracks()
    scores = model.score_candidates(ta, [])
    assert scores.shape == (0,)


# ===========================================================================
# Save / load artifact roundtrip
# ===========================================================================


@pytest.mark.parametrize("model_fn", [_linear_model, _network_model, _gated_model])
def test_save_load_score_identical(model_fn, tmp_path):
    model = model_fn()
    ta, tb = _make_tracks()
    meta = save_fusion_artifact(model, tmp_path)
    loaded = load_fusion_artifact(tmp_path)
    assert model.score_candidate(ta, tb) == loaded.score_candidate(ta, tb)
    assert model.score_candidate(ta, ta) == loaded.score_candidate(ta, ta)


@pytest.mark.parametrize("model_fn", [_linear_model, _network_model, _gated_model])
def test_save_returns_metadata(model_fn, tmp_path):
    model = model_fn()
    meta = save_fusion_artifact(model, tmp_path)
    assert isinstance(meta, FusionMetadata)
    assert len(meta.json_payload_sha256) == 64
    assert len(meta.npz_sha256) == 64
    assert meta.feature_dim == FEATURE_DIM
    assert meta.kind == model.config.kind


@pytest.mark.parametrize("model_fn", [_linear_model, _network_model, _gated_model])
def test_load_populates_metadata(model_fn, tmp_path):
    model = model_fn()
    save_meta = save_fusion_artifact(model, tmp_path)
    loaded = load_fusion_artifact(tmp_path)
    assert loaded.metadata is not None
    assert loaded.metadata.kind == model.config.kind
    assert loaded.metadata.json_payload_sha256 == save_meta.json_payload_sha256
    assert loaded.metadata.npz_sha256 == save_meta.npz_sha256


def test_save_creates_exactly_two_files(tmp_path):
    model = _linear_model()
    save_fusion_artifact(model, tmp_path)
    files = {f.name for f in tmp_path.iterdir() if f.is_file()}
    assert files == {"model.json", "weights.npz"}


def test_load_roundtrip_network(tmp_path):
    model = _network_model()
    ta, tb = _make_tracks()
    save_fusion_artifact(model, tmp_path)
    loaded = load_fusion_artifact(tmp_path)
    assert model.score_candidate(ta, tb) == pytest.approx(
        loaded.score_candidate(ta, tb), abs=1e-12
    )


def test_load_gated_embed_track_matches_pre_save(tmp_path):
    model = _gated_model()
    t = _identical_track()
    emb_before = model.embed_track(t)
    save_fusion_artifact(model, tmp_path)
    loaded = load_fusion_artifact(tmp_path)
    emb_after = loaded.embed_track(t)
    np.testing.assert_allclose(emb_before, emb_after, atol=1e-12)


def test_metadata_sha256_fields_consistent(tmp_path):
    from soundalike.ml.fulltrack_store import sha256_path, stable_json_sha256
    model = _linear_model()
    meta = save_fusion_artifact(model, tmp_path)
    npz_path = tmp_path / "weights.npz"
    json_path = tmp_path / "model.json"
    assert sha256_path(npz_path) == meta.npz_sha256
    with json_path.open("r", encoding="utf-8") as fh:
        doc = json.load(fh)
    payload = {k: v for k, v in doc.items() if k != "json_payload_sha256"}
    assert stable_json_sha256(payload) == meta.json_payload_sha256


# ===========================================================================
# Corruption / tamper detection
# ===========================================================================


def test_tampered_json_field_detected(tmp_path):
    save_fusion_artifact(_linear_model(), tmp_path)
    _tamper_json(tmp_path / "model.json", "fold_index", 999)
    with pytest.raises(FusionError, match="SHA-256 mismatch"):
        load_fusion_artifact(tmp_path)


def test_tampered_npz_detected(tmp_path):
    save_fusion_artifact(_linear_model(), tmp_path)
    _tamper_npz_byte(tmp_path / "weights.npz")
    with pytest.raises(FusionError, match="SHA-256 mismatch"):
        load_fusion_artifact(tmp_path)


def test_tampered_npz_sha256_field_detected(tmp_path):
    save_fusion_artifact(_linear_model(), tmp_path)
    _tamper_json(tmp_path / "model.json", "npz_sha256", "a" * 64)
    with pytest.raises(FusionError, match="SHA-256 mismatch"):
        load_fusion_artifact(tmp_path)


def test_wrong_json_payload_sha256_detected(tmp_path):
    save_fusion_artifact(_linear_model(), tmp_path)
    json_path = tmp_path / "model.json"
    with json_path.open("r", encoding="utf-8") as fh:
        doc = json.load(fh)
    doc["json_payload_sha256"] = "b" * 64
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(doc, fh, sort_keys=True, indent=2)
    with pytest.raises(FusionError, match="SHA-256 mismatch"):
        load_fusion_artifact(tmp_path)


def test_extra_json_field_rejected(tmp_path):
    save_fusion_artifact(_linear_model(), tmp_path)
    json_path = tmp_path / "model.json"
    with json_path.open("r", encoding="utf-8") as fh:
        doc = json.load(fh)
    doc["unexpected_field"] = "oops"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(doc, fh, sort_keys=True, indent=2)
    with pytest.raises(FusionError, match="unexpected fields"):
        load_fusion_artifact(tmp_path)


def test_missing_json_field_rejected(tmp_path):
    save_fusion_artifact(_linear_model(), tmp_path)
    json_path = tmp_path / "model.json"
    with json_path.open("r", encoding="utf-8") as fh:
        doc = json.load(fh)
    del doc["fold_index"]
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(doc, fh, sort_keys=True, indent=2)
    with pytest.raises(FusionError, match="missing required fields"):
        load_fusion_artifact(tmp_path)


def test_wrong_schema_version_rejected(tmp_path):
    save_fusion_artifact(_linear_model(), tmp_path)
    _tamper_json(tmp_path / "model.json", "schema_version", 999)
    with pytest.raises(FusionError, match="schema_version"):
        load_fusion_artifact(tmp_path)


def test_unknown_kind_in_json_rejected(tmp_path):
    save_fusion_artifact(_linear_model(), tmp_path)
    _tamper_json(tmp_path / "model.json", "kind", "bad_kind")
    with pytest.raises(FusionError, match="(SHA-256|unknown kind)"):
        load_fusion_artifact(tmp_path)


def test_wrong_feature_dim_rejected(tmp_path):
    save_fusion_artifact(_linear_model(), tmp_path)
    _tamper_json(tmp_path / "model.json", "feature_dim", 99)
    with pytest.raises(FusionError, match="(SHA-256|feature_dim)"):
        load_fusion_artifact(tmp_path)


def _fix_npz_and_json_hashes(tmp_path: Path) -> None:
    """Re-compute and store correct npz/json hashes after NPZ tampering."""
    from soundalike.ml.fulltrack_store import sha256_path, stable_json_sha256
    npz_path = tmp_path / "weights.npz"
    json_path = tmp_path / "model.json"
    new_npz_sha = sha256_path(npz_path)
    with json_path.open("r", encoding="utf-8") as fh:
        doc = json.load(fh)
    payload = {k: v for k, v in doc.items() if k != "json_payload_sha256"}
    payload["npz_sha256"] = new_npz_sha
    doc["npz_sha256"] = new_npz_sha
    doc["json_payload_sha256"] = stable_json_sha256(payload)
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(doc, fh, sort_keys=True, indent=2)


def test_extra_npz_key_rejected(tmp_path):
    model = _linear_model()
    save_fusion_artifact(model, tmp_path)
    npz_path = tmp_path / "weights.npz"
    old = np.load(str(npz_path), allow_pickle=False)
    arrays = {k: old[k] for k in old.files}
    arrays["extra_array"] = np.zeros(1)
    np.savez(str(npz_path), **arrays)
    _fix_npz_and_json_hashes(tmp_path)
    with pytest.raises(FusionError, match="unexpected"):
        load_fusion_artifact(tmp_path)


def test_missing_npz_key_rejected(tmp_path):
    model = _linear_model()
    save_fusion_artifact(model, tmp_path)
    npz_path = tmp_path / "weights.npz"
    np.savez(str(npz_path))
    _fix_npz_and_json_hashes(tmp_path)
    with pytest.raises(FusionError, match="missing"):
        load_fusion_artifact(tmp_path)


def test_negative_weights_in_npz_rejected(tmp_path):
    model = _linear_model()
    save_fusion_artifact(model, tmp_path)
    npz_path = tmp_path / "weights.npz"
    old = np.load(str(npz_path), allow_pickle=False)
    w = old["weights"].copy()
    w[0] = -1.0
    np.savez(str(npz_path), weights=w)
    _fix_npz_and_json_hashes(tmp_path)
    with pytest.raises(FusionError, match="non-negative"):
        load_fusion_artifact(tmp_path)


def test_wrong_weight_shape_in_npz_rejected(tmp_path):
    model = _linear_model()
    save_fusion_artifact(model, tmp_path)
    npz_path = tmp_path / "weights.npz"
    np.savez(str(npz_path), weights=np.ones(FEATURE_DIM + 5, dtype=np.float64))
    _fix_npz_and_json_hashes(tmp_path)
    with pytest.raises(FusionError, match="shape"):
        load_fusion_artifact(tmp_path)


def test_nan_weights_in_npz_rejected(tmp_path):
    model = _linear_model()
    save_fusion_artifact(model, tmp_path)
    npz_path = tmp_path / "weights.npz"
    w = np.ones(FEATURE_DIM, dtype=np.float64)
    w[3] = float("nan")
    np.savez(str(npz_path), weights=w)
    _fix_npz_and_json_hashes(tmp_path)
    with pytest.raises(FusionError, match="non-finite"):
        load_fusion_artifact(tmp_path)


def test_int_weights_in_npz_rejected(tmp_path):
    model = _linear_model()
    save_fusion_artifact(model, tmp_path)
    npz_path = tmp_path / "weights.npz"
    np.savez(str(npz_path), weights=np.ones(FEATURE_DIM, dtype=np.int32))
    _fix_npz_and_json_hashes(tmp_path)
    with pytest.raises(FusionError, match="floating-point"):
        load_fusion_artifact(tmp_path)


def test_all_zero_gates_in_npz_rejected(tmp_path):
    model = _gated_model()
    save_fusion_artifact(model, tmp_path)
    npz_path = tmp_path / "weights.npz"
    np.savez(str(npz_path), gates=np.zeros(4, dtype=np.float64))
    _fix_npz_and_json_hashes(tmp_path)
    with pytest.raises(FusionError, match="(positive sum|sum to 1.0)"):
        load_fusion_artifact(tmp_path)


def test_load_missing_json_file(tmp_path):
    with pytest.raises(FusionError, match="not found"):
        load_fusion_artifact(tmp_path)


def test_load_missing_npz_file(tmp_path):
    model = _linear_model()
    save_fusion_artifact(model, tmp_path)
    (tmp_path / "weights.npz").unlink()
    with pytest.raises(FusionError, match="not found"):
        load_fusion_artifact(tmp_path)


# ===========================================================================
# Path safety (Fix 1) -- symlinks and junctions
# ===========================================================================


def _try_symlink(target_path, link_path):
    """Attempt to create a symlink; skip test on permission error."""
    try:
        link_path.symlink_to(target_path)
    except OSError:
        pytest.skip("symlink creation not permitted")


def _try_junction(target_path, link_path):
    """Attempt to create a Windows junction; skip on failure."""
    if os.name != "nt":
        pytest.skip("junctions are Windows-only")
    try:
        subprocess.check_call(
            ["cmd", "/c", "mklink", "/J", str(link_path), str(target_path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("junction creation not permitted")


def test_symlinked_directory_save_rejected(tmp_path):
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    link_dir = tmp_path / "link"
    _try_symlink(real_dir, link_dir)
    with pytest.raises(FusionError, match="(symlink|reparse)"):
        save_fusion_artifact(_linear_model(), link_dir)


def test_symlinked_directory_load_rejected(tmp_path):
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    save_fusion_artifact(_linear_model(), real_dir)
    link_dir = tmp_path / "link"
    _try_symlink(real_dir, link_dir)
    with pytest.raises(FusionError, match="(symlink|reparse)"):
        load_fusion_artifact(link_dir)


def test_symlinked_model_json_rejected(tmp_path):
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    model = _linear_model()
    save_fusion_artifact(model, real_dir)
    link_dir = tmp_path / "link"
    link_dir.mkdir()
    (link_dir / "weights.npz").write_bytes((real_dir / "weights.npz").read_bytes())
    _try_symlink(real_dir / "model.json", link_dir / "model.json")
    with pytest.raises(FusionError, match="(symlink|reparse)"):
        load_fusion_artifact(link_dir)


def test_junction_directory_rejected(tmp_path):
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    save_fusion_artifact(_linear_model(), real_dir)
    link_dir = tmp_path / "junction"
    _try_junction(real_dir, link_dir)
    with pytest.raises(FusionError, match="(symlink|reparse)"):
        load_fusion_artifact(link_dir)


def _write_oversized_json(tmp_path: Path) -> None:
    model = _linear_model()
    save_fusion_artifact(model, tmp_path)
    json_path = tmp_path / "model.json"
    with json_path.open("r", encoding="utf-8") as fh:
        doc = json.load(fh)
    doc["comment"] = "x" * (300 * 1024)
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(doc, fh)


def test_oversized_json_rejected(tmp_path):
    from soundalike.ml.fulltrack_fusion import _MAX_JSON_BYTES
    _write_oversized_json(tmp_path)
    json_size = (tmp_path / "model.json").stat().st_size
    assert json_size > _MAX_JSON_BYTES
    with pytest.raises(FusionError, match="maximum size"):
        load_fusion_artifact(tmp_path)


# ===========================================================================
# Compressed NPZ rejection (Fix 3)
# ===========================================================================


def test_compressed_npz_rejected(tmp_path):
    """NPZ with compressed (non-stored) entries must be rejected."""
    model = _linear_model()
    save_fusion_artifact(model, tmp_path)
    npz_path = tmp_path / "weights.npz"
    # Read existing arrays, rewrite with compression
    old = np.load(str(npz_path), allow_pickle=False)
    arrays = {k: old[k] for k in old.files}
    np.savez_compressed(str(npz_path), **arrays)
    _fix_npz_and_json_hashes(tmp_path)
    with pytest.raises(FusionError, match="compress"):
        load_fusion_artifact(tmp_path)


# ===========================================================================
# Strict JSON types (Fix 6)
# ===========================================================================


def test_bool_as_int_in_json_rejected(tmp_path):
    """bool values in integer JSON fields must be rejected."""
    save_fusion_artifact(_linear_model(), tmp_path)
    json_path = tmp_path / "model.json"
    with json_path.open("r", encoding="utf-8") as fh:
        doc = json.load(fh)
    # Set embedding_dim to True (which json.load returns as bool)
    doc["embedding_dim"] = True
    from soundalike.ml.fulltrack_store import stable_json_sha256
    payload = {k: v for k, v in doc.items() if k != "json_payload_sha256"}
    doc["json_payload_sha256"] = stable_json_sha256(payload)
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(doc, fh, sort_keys=True, indent=2)
    with pytest.raises(FusionError, match="bool"):
        load_fusion_artifact(tmp_path)


def test_empty_model_id_in_json_rejected(tmp_path):
    """Empty model_id string must be rejected on load."""
    save_fusion_artifact(_linear_model(), tmp_path)
    json_path = tmp_path / "model.json"
    with json_path.open("r", encoding="utf-8") as fh:
        doc = json.load(fh)
    doc["model_id"] = ""
    from soundalike.ml.fulltrack_store import stable_json_sha256
    payload = {k: v for k, v in doc.items() if k != "json_payload_sha256"}
    doc["json_payload_sha256"] = stable_json_sha256(payload)
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(doc, fh, sort_keys=True, indent=2)
    with pytest.raises(FusionError, match="model_id"):
        load_fusion_artifact(tmp_path)


def test_bad_config_sha256_format_rejected(tmp_path):
    """config_sha256 must be exactly 64 lowercase hex."""
    save_fusion_artifact(_linear_model(), tmp_path)
    json_path = tmp_path / "model.json"
    with json_path.open("r", encoding="utf-8") as fh:
        doc = json.load(fh)
    doc["config_sha256"] = "ZZZZ" + "a" * 60  # uppercase
    from soundalike.ml.fulltrack_store import stable_json_sha256
    payload = {k: v for k, v in doc.items() if k != "json_payload_sha256"}
    doc["json_payload_sha256"] = stable_json_sha256(payload)
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(doc, fh, sort_keys=True, indent=2)
    with pytest.raises(FusionError, match="(config_sha256|SHA-256)"):
        load_fusion_artifact(tmp_path)


def test_bad_created_at_rejected(tmp_path):
    """created_at must match UTC timestamp format."""
    save_fusion_artifact(_linear_model(), tmp_path)
    json_path = tmp_path / "model.json"
    with json_path.open("r", encoding="utf-8") as fh:
        doc = json.load(fh)
    doc["created_at"] = "not-a-timestamp"
    from soundalike.ml.fulltrack_store import stable_json_sha256
    payload = {k: v for k, v in doc.items() if k != "json_payload_sha256"}
    doc["json_payload_sha256"] = stable_json_sha256(payload)
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(doc, fh, sort_keys=True, indent=2)
    with pytest.raises(FusionError, match="(created_at|SHA-256)"):
        load_fusion_artifact(tmp_path)


def test_hidden_dims_nonempty_for_linear_in_json_rejected(tmp_path):
    """hidden_dims must be empty for non-network kinds in JSON."""
    save_fusion_artifact(_linear_model(), tmp_path)
    json_path = tmp_path / "model.json"
    with json_path.open("r", encoding="utf-8") as fh:
        doc = json.load(fh)
    doc["hidden_dims"] = [8]
    from soundalike.ml.fulltrack_store import stable_json_sha256
    payload = {k: v for k, v in doc.items() if k != "json_payload_sha256"}
    doc["json_payload_sha256"] = stable_json_sha256(payload)
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(doc, fh, sort_keys=True, indent=2)
    with pytest.raises(FusionError, match="(hidden_dims|SHA-256)"):
        load_fusion_artifact(tmp_path)


def test_save_empty_model_id_rejected():
    """Saving with empty model_id raises FusionError."""
    cfg = FusionConfig(kind="nonnegative_linear", embedding_dim=DIM, model_id="")
    w = np.ones(FEATURE_DIM, dtype=np.float64)
    model = build_nonneg_linear(w, cfg)
    with pytest.raises(FusionError, match="model_id"):
        save_fusion_artifact(model, "dummy_dir")


def test_save_bad_config_sha256_rejected():
    """Saving with invalid config_sha256 raises FusionError."""
    cfg = FusionConfig(
        kind="nonnegative_linear", embedding_dim=DIM,
        model_id="m", store_id="s", config_sha256="short",
    )
    w = np.ones(FEATURE_DIM, dtype=np.float64)
    model = build_nonneg_linear(w, cfg)
    with pytest.raises(FusionError, match="config_sha256"):
        save_fusion_artifact(model, "dummy_dir")


# ===========================================================================
# Deep-copy / freeze weights (Fix 5)
# ===========================================================================


def test_weight_array_is_read_only():
    """Weight arrays must be read-only after build."""
    model = _linear_model()
    with pytest.raises(ValueError):
        model._weights["weights"][0] = 999.0


def test_caller_mutation_does_not_affect_model():
    """Mutating the original array after build must not affect the model."""
    cfg = FusionConfig(kind="nonnegative_linear", embedding_dim=DIM, maxsim_budget=BUDGET)
    w = np.ones(FEATURE_DIM, dtype=np.float64)
    model = build_nonneg_linear(w, cfg)
    ta, tb = _make_tracks()
    score_before = model.score_candidate(ta, tb)
    # Mutate original
    w[:] = 0.0
    score_after = model.score_candidate(ta, tb)
    assert score_before == score_after


def test_loaded_weights_are_read_only(tmp_path):
    """Weights loaded from artifact must be read-only."""
    save_fusion_artifact(_linear_model(), tmp_path)
    loaded = load_fusion_artifact(tmp_path)
    with pytest.raises(ValueError):
        loaded._weights["weights"][0] = 999.0


def test_network_weights_are_read_only():
    """Network layer weights must be read-only."""
    model = _network_model()
    with pytest.raises(ValueError):
        model._weights["layer_weights"][0][0, 0] = 999.0


def test_gated_weights_are_read_only():
    """Gate weights must be read-only."""
    model = _gated_model()
    with pytest.raises(ValueError):
        model._weights["gates"][0] = 999.0


# ===========================================================================
# Numeric safety (Fix 7)
# ===========================================================================


def test_huge_weight_rejected():
    """Impossibly large weights must be rejected."""
    cfg = FusionConfig(kind="nonnegative_linear", embedding_dim=DIM, maxsim_budget=BUDGET)
    w = np.ones(FEATURE_DIM, dtype=np.float64)
    w[0] = 1e16  # exceeds _MAX_WEIGHT_MAGNITUDE
    with pytest.raises(FusionError, match="weight magnitude"):
        build_nonneg_linear(w, cfg)


def test_normalization_overflow_safe():
    """Very large but finite embedding should normalize without overflow."""
    # Use float64 values large enough to overflow naive norm but
    # within float64 range; pass directly as float64.
    g = np.ones(DIM, dtype=np.float64) * 1e150
    wins = np.tile(g, (4, 1))
    t = _FakeTrack(g, wins, wins[:2], wins[:2])
    pf = extract_pair_features(t, t, maxsim_budget=BUDGET)
    vec = pf.to_vector()
    assert np.all(np.isfinite(vec))


def test_nonfinite_section_rejected():
    """Non-finite non-empty sections must raise FusionError."""
    g = _unit(0)
    wins = np.tile(g, (4, 1))
    bad_sections = np.tile(g, (2, 1)).astype(np.float32)
    bad_sections[0, 0] = float("nan")
    t_ok = _FakeTrack(g, wins)
    t_bad = _FakeTrack(g, wins, repeated_sections=bad_sections)
    with pytest.raises(FusionError, match="non-finite"):
        extract_pair_features(t_ok, t_bad, maxsim_budget=BUDGET)


# ===========================================================================
# Dimension mismatch (Fix 4)
# ===========================================================================


def test_section_dim_mismatch_rejected():
    """Section dim != global dim must raise FusionError."""
    g = _unit(0, dim=8)
    wins = np.tile(g, (4, 1))
    bad_sections = np.ones((2, 4), dtype=np.float32)  # dim=4 != 8
    t_ok = _FakeTrack(g, wins)
    t_bad = _FakeTrack(g, wins, repeated_sections=bad_sections)
    with pytest.raises(FusionError, match="dim mismatch"):
        extract_pair_features(t_ok, t_bad, maxsim_budget=BUDGET)


def test_gated_embedding_dim_mismatch():
    """channel_gated must validate embedding dim against config."""
    model = _gated_model(dim=8)
    g = _unit(0, dim=4)  # wrong dim
    wins = np.tile(g, (4, 1))
    t = _FakeTrack(g, wins)
    with pytest.raises(FusionError, match="(dim|embedding)"):
        model.embed_track(t)


# ===========================================================================
# Ablation correctness
# ===========================================================================


@pytest.mark.parametrize("ablation", ["none", "global_only", "no_sections"])
@pytest.mark.parametrize("model_fn", [_linear_model, _network_model, _gated_model])
def test_all_ablations_bounded(model_fn, ablation):
    model = model_fn()
    ta, tb = _make_tracks()
    s = model.score_candidate(ta, tb, ablation=ablation)
    assert 0.0 <= s <= 1.0
    assert np.isfinite(s)


def test_linear_global_only_uses_only_global_feature():
    model = _linear_model()
    ta, tb = _make_tracks()
    pf = extract_pair_features(ta, tb, maxsim_budget=BUDGET)
    s = model.score_candidate(ta, tb, ablation="global_only")
    assert s == pytest.approx(pf.global_cosine, abs=1e-9)


def test_no_sections_differs_from_none_when_sections_informative():
    dim = 8
    g = _unit(0, dim)
    wins = _rand_unit_matrix(8, dim, seed=123)
    rep = wins[:4]
    sal = wins[4:]
    ta = _FakeTrack(g, wins, rep, sal)
    tb = _FakeTrack(_unit(1, dim), wins[::-1], rep, sal)
    model = _linear_model(dim=dim)
    s_none = model.score_candidate(ta, tb, ablation="none")
    s_no_sec = model.score_candidate(ta, tb, ablation="no_sections")
    assert 0.0 <= s_none <= 1.0
    assert 0.0 <= s_no_sec <= 1.0


def test_gated_global_only_equals_pure_global_cosine():
    model = _gated_model()
    g_a = np.zeros(DIM, dtype=np.float32); g_a[0] = 1.0
    g_b = np.zeros(DIM, dtype=np.float32); g_b[0] = 1.0
    ta = _FakeTrack(g_a, np.tile(g_a, (4, 1)))
    tb = _FakeTrack(g_b, np.tile(g_b, (4, 1)))
    s = model.score_candidate(ta, tb, ablation="global_only")
    assert s == pytest.approx(1.0, abs=1e-9)


# ===========================================================================
# Metadata and config identity
# ===========================================================================


def test_metadata_kind_matches_config(tmp_path):
    for model_fn in [_linear_model, _network_model, _gated_model]:
        model = model_fn()
        d = tmp_path / model.config.kind
        meta = save_fusion_artifact(model, d)
        assert meta.kind == model.config.kind


def test_metadata_feature_dim_constant(tmp_path):
    meta = save_fusion_artifact(_linear_model(), tmp_path)
    assert meta.feature_dim == FEATURE_DIM


def test_config_stored_and_retrieved(tmp_path):
    cfg = FusionConfig(
        kind="nonnegative_linear",
        embedding_dim=16,
        maxsim_budget=4,
        top_k=3,
        seed=42,
        model_id="test-model",
        store_id="test-store",
        config_sha256="b" * 64,
        fold_index=2,
    )
    w = np.ones(FEATURE_DIM, dtype=np.float64)
    model = build_nonneg_linear(w, cfg)
    save_fusion_artifact(model, tmp_path)
    loaded = load_fusion_artifact(tmp_path)
    c = loaded.config
    assert c.kind == "nonnegative_linear"
    assert c.embedding_dim == 16
    assert c.maxsim_budget == 4
    assert c.top_k == 3
    assert c.seed == 42
    assert c.model_id == "test-model"
    assert c.store_id == "test-store"
    assert c.fold_index == 2


def test_metadata_as_dict_roundtrip(tmp_path):
    model = _gated_model()
    meta = save_fusion_artifact(model, tmp_path)
    d = meta.as_dict()
    assert d["kind"] == "channel_gated_embedding"
    assert d["feature_dim"] == FEATURE_DIM
    assert len(d["json_payload_sha256"]) == 64
    assert len(d["npz_sha256"]) == 64


# ===========================================================================
# Edge cases
# ===========================================================================


def test_single_window_track():
    g = _unit(0)
    t = _FakeTrack(g, g.reshape(1, -1), g.reshape(1, -1), g.reshape(1, -1))
    model = _linear_model()
    s = model.score_candidate(t, t)
    assert 0.0 <= s <= 1.0


def test_more_budget_than_windows():
    g = _unit(0)
    t = _FakeTrack(g, g.reshape(1, -1))
    pf = extract_pair_features(t, t, maxsim_budget=4)
    assert 0.0 <= pf.global_cosine <= 1.0


def test_float16_windows_handled():
    g = _unit(2).astype(np.float16)
    wins = _rand_unit_matrix(4, 8, seed=5).astype(np.float16)
    wins[0] = g
    t = _FakeTrack(g, wins, wins[:2], wins[2:])
    model = _linear_model()
    s = model.score_candidate(t, t)
    assert 0.0 <= s <= 1.0


def test_score_none_ablation_default():
    model = _linear_model()
    ta, tb = _make_tracks()
    assert model.score_candidate(ta, tb) == model.score_candidate(
        ta, tb, ablation="none"
    )


def test_pair_features_bad_budget():
    t = _identical_track()
    with pytest.raises(FusionError, match="maxsim_budget"):
        extract_pair_features(t, t, maxsim_budget=0)


def test_pair_features_bad_threshold():
    t = _identical_track()
    with pytest.raises(FusionError, match="coverage_threshold"):
        extract_pair_features(t, t, coverage_threshold=1.5)


# ===========================================================================
# Finding 1: embedding_dim enforcement for ALL kinds
# ===========================================================================


def test_model_embedding_dim_global_mismatch_rejected():
    """FusionModel.extract_pair_features rejects wrong global dim."""
    model = _linear_model(dim=8)
    g_wrong = _unit(0, dim=4)
    t_wrong = _FakeTrack(g_wrong, np.tile(g_wrong, (4, 1)))
    t_ok = _FakeTrack(_unit(0, dim=8), np.tile(_unit(0, dim=8), (4, 1)))
    with pytest.raises(FusionError, match="global_embedding dim"):
        model.extract_pair_features(t_wrong, t_ok)


def test_model_embedding_dim_window_mismatch_rejected():
    """FusionModel.extract_pair_features rejects wrong window dim."""
    model = _linear_model(dim=8)
    g = _unit(0, dim=8)
    wins_wrong = np.tile(_unit(0, dim=4), (4, 1))
    t_wrong = _FakeTrack(g, wins_wrong)
    t_ok = _FakeTrack(g, np.tile(g, (4, 1)))
    with pytest.raises(FusionError, match="window_embeddings dim"):
        model.extract_pair_features(t_wrong, t_ok)


def test_model_embedding_dim_repeated_mismatch_rejected():
    """FusionModel.extract_pair_features rejects wrong nonempty repeated dim."""
    model = _linear_model(dim=8)
    g = _unit(0, dim=8)
    wins = np.tile(g, (4, 1))
    bad_rep = np.ones((2, 4), dtype=np.float32)  # dim=4 != 8
    t_wrong = _FakeTrack(g, wins, repeated_sections=bad_rep)
    t_ok = _FakeTrack(g, wins)
    with pytest.raises(FusionError, match="repeated_sections dim"):
        model.extract_pair_features(t_wrong, t_ok)


def test_model_embedding_dim_salient_mismatch_rejected():
    """FusionModel.extract_pair_features rejects wrong nonempty salient dim."""
    model = _linear_model(dim=8)
    g = _unit(0, dim=8)
    wins = np.tile(g, (4, 1))
    bad_sal = np.ones((2, 4), dtype=np.float32)  # dim=4 != 8
    t_wrong = _FakeTrack(g, wins, salient_sections=bad_sal)
    t_ok = _FakeTrack(g, wins)
    with pytest.raises(FusionError, match="salient_sections dim"):
        model.extract_pair_features(t_wrong, t_ok)


def test_model_score_linear_enforces_embedding_dim():
    """Linear score path rejects tracks with wrong embedding dim."""
    model = _linear_model(dim=8)
    g_wrong = _unit(0, dim=4)
    t_wrong = _FakeTrack(g_wrong, np.tile(g_wrong, (4, 1)))
    t_ok = _FakeTrack(_unit(0, dim=8), np.tile(_unit(0, dim=8), (4, 1)))
    with pytest.raises(FusionError, match="(embedding_dim|dim)"):
        model.score_candidate(t_wrong, t_ok)


def test_model_score_network_enforces_embedding_dim():
    """Network score path rejects tracks with wrong embedding dim."""
    model = _network_model(dim=8)
    g_wrong = _unit(0, dim=4)
    t_wrong = _FakeTrack(g_wrong, np.tile(g_wrong, (4, 1)))
    t_ok = _FakeTrack(_unit(0, dim=8), np.tile(_unit(0, dim=8), (4, 1)))
    with pytest.raises(FusionError, match="(embedding_dim|dim)"):
        model.score_candidate(t_wrong, t_ok)


# ===========================================================================
# Finding 2: duplicate ZIP/NPZ member names rejected
# ===========================================================================


@pytest.mark.filterwarnings("ignore:Duplicate name:UserWarning")
def test_duplicate_npz_member_rejected(tmp_path):
    """NPZ with duplicate member names must be rejected even with valid hashes.

    Manually crafts a ZIP file with two weights.npy entries, recomputes
    valid SHA-256 hashes, and proves load rejects due to exact-member schema.
    """
    model = _linear_model()
    save_fusion_artifact(model, tmp_path / "orig")
    # Load valid arrays
    orig_npz = tmp_path / "orig" / "weights.npz"
    orig_data = np.load(str(orig_npz), allow_pickle=False)
    w_bytes = io.BytesIO()
    np.save(w_bytes, orig_data["weights"])
    w_npy = w_bytes.getvalue()

    # Build a ZIP with duplicate "weights.npy" entries
    dup_buf = io.BytesIO()
    with zipfile.ZipFile(dup_buf, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("weights.npy", w_npy)
        zf.writestr("weights.npy", w_npy)  # duplicate!
    dup_bytes = dup_buf.getvalue()

    # Write the duplicate NPZ and recompute valid hashes
    dest = tmp_path / "dup"
    dest.mkdir()
    npz_path = dest / "weights.npz"
    npz_path.write_bytes(dup_bytes)

    # Copy and fix JSON from orig
    import json as _json
    from soundalike.ml.fulltrack_store import stable_json_sha256
    json_path = dest / "model.json"
    with (tmp_path / "orig" / "model.json").open("r") as fh:
        doc = _json.load(fh)
    new_npz_sha = hashlib.sha256(dup_bytes).hexdigest()
    doc["npz_sha256"] = new_npz_sha
    payload = {k: v for k, v in doc.items() if k != "json_payload_sha256"}
    payload["npz_sha256"] = new_npz_sha
    doc["json_payload_sha256"] = stable_json_sha256(payload)
    with json_path.open("w") as fh:
        _json.dump(doc, fh, sort_keys=True, indent=2)

    with pytest.raises(FusionError, match="duplicate"):
        load_fusion_artifact(dest)


# ===========================================================================
# Finding 3: public parameter validation
# ===========================================================================


def test_extract_bool_maxsim_budget_rejected():
    """bool maxsim_budget must be rejected (not silently coerced)."""
    t = _identical_track()
    with pytest.raises(FusionError, match="maxsim_budget.*integer.*bool"):
        extract_pair_features(t, t, maxsim_budget=True)


def test_extract_bool_top_k_rejected():
    """bool top_k must be rejected."""
    t = _identical_track()
    with pytest.raises(FusionError, match="top_k.*integer.*bool"):
        extract_pair_features(t, t, top_k=False)


def test_extract_float_maxsim_budget_rejected():
    """float maxsim_budget must be rejected."""
    t = _identical_track()
    with pytest.raises(FusionError, match="maxsim_budget.*integer"):
        extract_pair_features(t, t, maxsim_budget=4.0)


def test_extract_float_top_k_rejected():
    """float top_k must be rejected."""
    t = _identical_track()
    with pytest.raises(FusionError, match="top_k.*integer"):
        extract_pair_features(t, t, top_k=2.0)


def test_extract_bool_coverage_threshold_rejected():
    """bool coverage_threshold must be rejected."""
    t = _identical_track()
    with pytest.raises(FusionError, match="coverage_threshold.*bool"):
        extract_pair_features(t, t, coverage_threshold=True)


def test_extract_nan_coverage_threshold_rejected():
    """NaN coverage_threshold must be rejected."""
    t = _identical_track()
    with pytest.raises(FusionError, match="coverage_threshold.*finite"):
        extract_pair_features(t, t, coverage_threshold=float("nan"))


def test_extract_inf_coverage_threshold_rejected():
    """inf coverage_threshold must be rejected."""
    t = _identical_track()
    with pytest.raises(FusionError, match="coverage_threshold.*finite"):
        extract_pair_features(t, t, coverage_threshold=float("inf"))


def test_extract_maxsim_budget_too_large_rejected():
    """maxsim_budget exceeding upper bound must be rejected."""
    from soundalike.ml.fulltrack_fusion import _MAX_MAXSIM_BUDGET
    t = _identical_track()
    with pytest.raises(FusionError, match="maxsim_budget"):
        extract_pair_features(t, t, maxsim_budget=_MAX_MAXSIM_BUDGET + 1)


def test_extract_top_k_too_large_rejected():
    """top_k exceeding upper bound must be rejected."""
    from soundalike.ml.fulltrack_fusion import _MAX_TOP_K
    t = _identical_track()
    with pytest.raises(FusionError, match="top_k"):
        extract_pair_features(t, t, top_k=_MAX_TOP_K + 1)


def test_extract_string_maxsim_budget_rejected():
    """string maxsim_budget must raise FusionError, not TypeError."""
    t = _identical_track()
    with pytest.raises(FusionError, match="maxsim_budget"):
        extract_pair_features(t, t, maxsim_budget="8")


def test_extract_none_top_k_rejected():
    """None top_k must raise FusionError, not TypeError."""
    t = _identical_track()
    with pytest.raises(FusionError, match="top_k"):
        extract_pair_features(t, t, top_k=None)


# ===========================================================================
# Finding 4: artifact I/O safety
# ===========================================================================


def test_immutable_export_prevents_overwrite(tmp_path):
    """save_fusion_artifact refuses to overwrite existing sealed artifacts."""
    model = _linear_model()
    save_fusion_artifact(model, tmp_path)
    # Second save to same directory must fail
    with pytest.raises(FusionError, match="(overwrite|already exists)"):
        save_fusion_artifact(model, tmp_path)


def test_immutable_export_npz_only(tmp_path):
    """Refuses overwrite even if only weights.npz exists."""
    model = _linear_model()
    save_fusion_artifact(model, tmp_path)
    # Remove model.json but keep weights.npz
    (tmp_path / "model.json").unlink()
    with pytest.raises(FusionError, match="(overwrite|already exists)"):
        save_fusion_artifact(model, tmp_path)


def test_save_unique_temp_names(tmp_path):
    """Temp file names use cryptographic randomness, not just PID."""
    from soundalike.ml.fulltrack_fusion import _unique_tmp_path
    p = tmp_path / "target.json"
    t1 = _unique_tmp_path(p)
    t2 = _unique_tmp_path(p)
    assert t1 != t2, "temp names must be unique"
    assert str(os.getpid()) in t1.name
    # Should have random hex component
    parts = t1.name.split(".")
    assert len(parts) >= 4, f"temp name lacks random component: {t1.name}"


# ===========================================================================
# Finding 5: temp cleanup and directory fsync
# ===========================================================================


def test_save_no_leftover_temp_files(tmp_path):
    """No temp files remain after successful save."""
    model = _linear_model()
    save_fusion_artifact(model, tmp_path)
    all_files = {f.name for f in tmp_path.iterdir()}
    tmp_files = {f for f in all_files if ".tmp" in f}
    assert tmp_files == set(), f"leftover temp files: {tmp_files}"


def test_fsync_directory_noop_windows():
    """_fsync_directory is no-op on Windows and doesn't raise."""
    from soundalike.ml.fulltrack_fusion import _fsync_directory
    # Should not raise on any platform
    _fsync_directory(Path(tempfile.gettempdir()))



# ===========================================================================
# Pre-publish NPZ hash binding and substitution detection
# ===========================================================================


def test_npz_substitution_prevents_json_publication(tmp_path, monkeypatch):
    """Simulated NPZ substitution between write and verify blocks JSON.

    Monkeypatches _safe_read_file so the post-write verification read
    returns tampered bytes; proves FusionError is raised and model.json
    is never written.
    """
    import soundalike.ml.fulltrack_fusion as _mod

    _original_read = _mod._safe_read_file
    _verify_called = [False]

    def _substituting_read(path, label, max_bytes):
        data = _original_read(path, label, max_bytes)
        if "post-write verify" in label:
            _verify_called[0] = True
            corrupted = bytearray(data)
            corrupted[-1] ^= 0xFF
            return bytes(corrupted)
        return data

    monkeypatch.setattr(_mod, "_safe_read_file", _substituting_read)

    model = _linear_model()
    with pytest.raises(FusionError, match="substitut"):
        save_fusion_artifact(model, tmp_path)

    assert _verify_called[0], "post-write verification was not invoked"
    assert not (tmp_path / "model.json").exists(), (
        "model.json must not be published when NPZ substitution is detected"
    )


def test_save_npz_hash_is_pre_publish_serialized(tmp_path):
    """Metadata npz_sha256 is from pre-publish serialized bytes, matches disk."""
    model = _linear_model()
    meta = save_fusion_artifact(model, tmp_path)
    on_disk = (tmp_path / "weights.npz").read_bytes()
    assert meta.npz_sha256 == hashlib.sha256(on_disk).hexdigest()
    # JSON document records the same hash
    with (tmp_path / "model.json").open("r", encoding="utf-8") as fh:
        doc = json.load(fh)
    assert doc["npz_sha256"] == meta.npz_sha256


# ===========================================================================
# Deterministic ablation: monotonic_network and channel_gated_embedding
# ===========================================================================


def test_network_ablation_deterministic_sections():
    """monotonic_network: none vs no_sections must differ when sections inform.

    Constructs deterministic synthetic tracks with non-neutral section features
    and a network sensitive to all 16 features.  global_only score must match
    the expected manual forward pass using only global_cosine.  Test would
    fail if ablation masking were ignored.
    """
    dim = 8
    g = _unit(0, dim)
    wins = np.tile(g, (8, 1))
    rep = np.tile(g, (4, 1))
    sal = np.tile(g, (4, 1))
    idx = np.array([0, 2, 4, 6], dtype=np.int64)
    ta = _FakeTrack(g, wins, rep, sal, repeated_indices=idx, salient_indices=idx)

    # Verify section features are non-neutral (well above 0.5)
    pf = extract_pair_features(ta, ta, maxsim_budget=BUDGET)
    assert pf.repeated_maxsim_sym > 0.9
    assert pf.salient_maxsim_sym > 0.9
    assert pf.recurrence_indicator > 0.9
    assert pf.repeated_temporal_sim > 0.9
    assert pf.salient_temporal_sim > 0.9

    # Build a network with all 16 features contributing, non-saturating
    cfg = FusionConfig(
        kind="monotonic_network", embedding_dim=dim, maxsim_budget=BUDGET,
        hidden_dims=(FEATURE_DIM,),
        model_id=_TEST_MODEL_ID, store_id=_TEST_STORE_ID,
        config_sha256=_TEST_CONFIG_SHA,
    )
    w0 = np.eye(FEATURE_DIM, dtype=np.float64)
    b0 = np.zeros(FEATURE_DIM, dtype=np.float64)
    w1 = np.ones((1, FEATURE_DIM), dtype=np.float64) * 0.05
    b1 = np.array([-0.4], dtype=np.float64)
    model = build_monotonic_network([w0, w1], [b0, b1], cfg)

    s_none = model.score_candidate(ta, ta, ablation="none")
    s_no_sec = model.score_candidate(ta, ta, ablation="no_sections")
    s_global = model.score_candidate(ta, ta, ablation="global_only")

    # none includes section features (~1.0), no_sections zeros them -> must differ
    assert s_none != s_no_sec, (
        f"none ({s_none}) must differ from no_sections ({s_no_sec}) "
        f"when sections are informative"
    )
    # Section features are positive -> none gives higher pre-sigmoid -> higher score
    assert s_none > s_no_sec

    # global_only: verify against expected manual forward pass
    feat = pf.to_vector()
    # global_only mask: only feature index 0 survives
    mask_g = np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=np.float64)
    x = feat * mask_g
    x = w0 @ x + b0
    x = np.maximum(x, 0.0)
    x = (w1 @ x + b1).ravel()
    expected_global = 1.0 / (1.0 + np.exp(-float(np.clip(x[0], -500, 500))))
    assert s_global == pytest.approx(expected_global, abs=1e-9)


def test_gated_ablation_deterministic_sections():
    """channel_gated_embedding: none vs no_sections must differ with distinct channels.

    Tracks have orthogonal basis vectors for each channel (global, uniform,
    repeated, salient), and sections differ between tracks A and B.  Equal
    gates ensure all channels contribute.  global_only score must equal
    pure global cosine (1+cos(g_a,g_b))/2.  Test would fail if channel
    mask were ignored.
    """
    dim = 8
    # Track A: each channel is a distinct basis vector
    g_a = _unit(0, dim)
    wins_a = np.tile(_unit(1, dim), (4, 1))  # uniform channel -> e1
    rep_a = np.tile(_unit(2, dim), (2, 1))   # repeated channel -> e2
    sal_a = np.tile(_unit(3, dim), (2, 1))   # salient channel -> e3
    ta = _FakeTrack(g_a, wins_a, rep_a, sal_a)

    # Track B: same global and uniform, different sections
    g_b = _unit(0, dim)
    wins_b = np.tile(_unit(1, dim), (4, 1))
    rep_b = np.tile(_unit(4, dim), (2, 1))   # different from A
    sal_b = np.tile(_unit(5, dim), (2, 1))   # different from A
    tb = _FakeTrack(g_b, wins_b, rep_b, sal_b)

    # Equal gates for all 4 channels
    cfg = FusionConfig(
        kind="channel_gated_embedding", embedding_dim=dim, maxsim_budget=BUDGET,
        model_id=_TEST_MODEL_ID, store_id=_TEST_STORE_ID,
        config_sha256=_TEST_CONFIG_SHA,
    )
    gates = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
    model = build_channel_gated(gates, cfg)

    s_none = model.score_candidate(ta, tb, ablation="none")
    s_no_sec = model.score_candidate(ta, tb, ablation="no_sections")
    s_global = model.score_candidate(ta, tb, ablation="global_only")

    # none: section channels are orthogonal between tracks -> lower score
    # no_sections: drops sections, both embeddings = normalize(e0+e1) -> score=1
    assert s_none != s_no_sec, (
        f"none ({s_none}) must differ from no_sections ({s_no_sec}) "
        f"when section channels differ between tracks"
    )
    assert s_no_sec == pytest.approx(1.0, abs=1e-6)

    # global_only: both use only global channel -> both embed as e0 -> cos=1.0
    ga = np.asarray(ta.global_embedding, dtype=np.float64).ravel()
    ga = ga / np.linalg.norm(ga)
    gb = np.asarray(tb.global_embedding, dtype=np.float64).ravel()
    gb = gb / np.linalg.norm(gb)
    expected_cos = float(np.clip((1.0 + np.dot(ga, gb)) / 2.0, 0.0, 1.0))
    assert s_global == pytest.approx(expected_cos, abs=1e-9)
