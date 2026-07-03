"""Tests for the recommendation-quality benchmark (pure numpy, no GPU/network)."""

from __future__ import annotations

import numpy as np

from soundalike.ml.benchmark import (
    balance_point,
    coverage_score,
    find_sweet_spot,
    fixed_pair_precision,
    library_size_sweep,
    same_artist_map,
    score_embeddings,
    _fit_whiten,
    _whiten,
)


def _synthetic(n_artists=200, per_artist=6, dim=32, seed=0):
    """Songs cluster tightly by artist — a good recommender should retrieve
    same-artist neighbours, and precision should fall as distractors grow."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_artists, dim))
    neural, artists = [], []
    for a in range(n_artists):
        for _ in range(per_artist):
            neural.append(centers[a] + 0.15 * rng.standard_normal(dim))
        artists += [f"artist{a}"] * per_artist
    neural = np.asarray(neural, np.float32)
    return neural, np.asarray(artists, dtype=object)


def test_whiten_unit_norm():
    neural, _ = _synthetic()
    mean, W = _fit_whiten(neural)
    w = _whiten(neural, mean, W)
    assert np.allclose(np.linalg.norm(w, axis=1), 1.0, atol=1e-4)


def test_coverage_in_range():
    neural, _ = _synthetic()
    mean, W = _fit_whiten(neural)
    w = _whiten(neural, mean, W)
    cov = coverage_score(w[:50], w[50:])
    assert -1.0 <= cov <= 1.0


def test_fixed_pair_precision_beats_chance():
    # With tight per-artist clusters, a same-artist target should land in top-K
    # far more often than chance.
    neural, artists = _synthetic(n_artists=100, per_artist=4, seed=1)
    mean, W = _fit_whiten(neural)
    w = _whiten(neural, mean, W)
    active = np.ones(len(w), dtype=bool)
    # one (query, target) pair per artist
    pairs = []
    for a in range(100):
        base = a * 4
        pairs.append((base, base + 1))
    recall = fixed_pair_precision(w, active, pairs, k=10)
    assert recall > 0.5  # clusters are tight, so siblings retrieve easily


def test_size_sweep_precision_falls_coverage_rises():
    neural, artists = _synthetic(n_artists=300, per_artist=5, seed=2)
    rows = library_size_sweep(neural, artists, sizes=[200, 600, 1200],
                              k=10, n_pairs=100, n_probe=100, seed=2)
    assert [r["size"] for r in rows] == [200, 600, 1200]
    # More distractors -> fixed-pair precision should not increase.
    assert rows[0]["recall_at_k"] >= rows[-1]["recall_at_k"]
    # Bigger library -> coverage should not decrease.
    assert rows[-1]["coverage"] >= rows[0]["coverage"] - 1e-6


def test_balance_and_sweet_spot_return_swept_sizes():
    rows = [
        {"size": 1000, "recall_at_k": 0.20, "coverage": 0.30},
        {"size": 5000, "recall_at_k": 0.10, "coverage": 0.42},
        {"size": 9000, "recall_at_k": 0.05, "coverage": 0.45},
    ]
    assert find_sweet_spot(rows) in {1000, 5000, 9000}
    assert balance_point(rows) in {1000, 5000, 9000}


def test_same_artist_map_rewards_clustered_space():
    # Tight per-artist clusters -> siblings rank first -> mAP near 1.
    good, artists = _synthetic(n_artists=80, per_artist=5, seed=3)
    mean, W = _fit_whiten(good)
    good_w = _whiten(good, mean, W)
    map_good = same_artist_map(good_w, artists, n_queries=200, seed=3)
    # Shuffle rows vs artists -> destroys structure -> mAP collapses.
    rng = np.random.default_rng(3)
    scrambled = good_w[rng.permutation(len(good_w))]
    map_bad = same_artist_map(scrambled, artists, n_queries=200, seed=3)
    assert 0.0 <= map_bad <= map_good <= 1.0
    assert map_good > 0.5
    assert map_good > map_bad + 0.2  # structure clearly beats noise


def test_same_artist_map_zero_when_no_multi_artist():
    # Every artist appears once -> no relevant siblings -> mAP defined as 0.
    neural = np.random.default_rng(0).standard_normal((10, 8)).astype(np.float32)
    artists = np.asarray([f"solo{i}" for i in range(10)], dtype=object)
    assert same_artist_map(neural, artists, n_queries=10) == 0.0


def test_score_embeddings_reports_all_metrics():
    neural, artists = _synthetic(n_artists=120, per_artist=5, seed=4)
    out = score_embeddings(neural, artists, n_queries=200, n_probe=100, seed=4)
    for key in ("map", "recall_at_k", "mrr", "coverage", "dim", "n_lib"):
        assert key in out
    assert out["dim"] == neural.shape[1]
    assert 0.0 <= out["map"] <= 1.0
    assert 0.0 <= out["recall_at_k"] <= 1.0
    # Clustered space should retrieve siblings well above chance.
    assert out["recall_at_k"] > 0.5

