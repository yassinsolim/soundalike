"""Tests for the recommendation-quality benchmark (pure numpy, no GPU/network)."""

from __future__ import annotations

import numpy as np

from soundalike.ml.benchmark import (
    balance_point,
    coverage_score,
    find_sweet_spot,
    fixed_pair_precision,
    library_size_sweep,
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
