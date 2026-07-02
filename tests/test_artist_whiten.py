"""Tests for embedding whitening in the deep-vibe recommender and the
artist-aware training helpers (pure logic, no GPU/network)."""

from __future__ import annotations

import numpy as np

from soundalike.ml.deepvibe import DeepVibeIndex, DeepVibeRecommender
from soundalike.audio.vibe import FEATURE_NAMES, VibeFeatures


def _rand_index(n=60, d=256, seed=0):
    rng = np.random.default_rng(seed)
    neural = rng.standard_normal((n, d)).astype(np.float32)
    vibe = rng.standard_normal((n, len(FEATURE_NAMES))).astype(np.float32)
    ids = list(range(n))
    titles = [f"t{i}" for i in range(n)]
    artists = [f"a{i%12}" for i in range(n)]
    return DeepVibeIndex(ids, titles, artists, neural, vibe)


def _vibe_from_vec(vec):
    vec = [float(x) for x in vec]
    return VibeFeatures(
        tempo=vec[0], brightness=vec[1], rolloff=vec[2], onset_rate=vec[3],
        rms_mean=vec[4], rms_std=vec[5], dynamic_range=vec[6], crest=vec[7],
        low_end_ratio=vec[8], bands=vec[9:16], mfcc=vec[16:29],
    )


def test_whiten_produces_unit_norm_rows():
    idx = _rand_index()
    rec = DeepVibeRecommender(idx, alpha=1.0, whiten=True)
    norms = np.linalg.norm(rec._neural, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_whiten_changes_ranking_vs_raw():
    idx = _rand_index(seed=1)
    seed_neural = idx.neural[0]
    seed_vibe = _vibe_from_vec(idx.vibe[0])
    raw = DeepVibeRecommender(idx, alpha=1.0, whiten=False)
    wht = DeepVibeRecommender(idx, alpha=1.0, whiten=True)
    r_raw = [r.track_id for r in raw.recommend(seed_neural, seed_vibe, n=10)]
    r_wht = [r.track_id for r in wht.recommend(seed_neural, seed_vibe, n=10)]
    # Whitening should reshape the neighbourhood (not identical ordering).
    assert r_raw != r_wht


def test_recommend_runs_with_whitening_and_excludes():
    idx = _rand_index(seed=2)
    rec = DeepVibeRecommender(idx, alpha=0.8, whiten=True)
    out = rec.recommend(idx.neural[3], _vibe_from_vec(idx.vibe[3]),
                        n=5, exclude_ids={0, 1})
    assert len(out) == 5
    assert all(r.track_id not in {0, 1} for r in out)


def test_pk_batches_shape():
    from soundalike.ml.train_artist import _pk_batches

    labels = np.array([i // 5 for i in range(200)])  # 40 artists, 5 songs each
    batches = list(_pk_batches(labels, p_artists=8, k_songs=4, seed=0))
    assert batches, "expected at least one PK batch"
    for b in batches:
        assert len(b) == 8 * 4
        # Each chosen artist appears exactly k times.
        _, counts = np.unique(labels[b], return_counts=True)
        assert set(counts.tolist()) == {4}


def test_pk_batches_empty_when_too_few_artists():
    # Regression: fewer eligible artists than p_artists yields no batches (the
    # trainer guards this by reducing p_artists / raising a clear error).
    from soundalike.ml.train_artist import _pk_batches

    labels = np.array([i // 5 for i in range(30)])  # 6 artists
    assert list(_pk_batches(labels, p_artists=128, k_songs=4, seed=0)) == []


def test_supcon_loss_rewards_grouping():
    import torch

    from soundalike.ml.train_artist import _supcon_loss

    labels = torch.tensor([0, 0, 1, 1])
    # Well-separated same-label pairs -> low loss.
    good = torch.tensor([[1.0, 0.0], [0.99, 0.14], [-1.0, 0.0], [-0.99, 0.14]])
    good = torch.nn.functional.normalize(good, dim=1)
    # Scrambled -> higher loss.
    bad = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.99, 0.14], [-0.99, 0.14]])
    bad = torch.nn.functional.normalize(bad, dim=1)
    assert float(_supcon_loss(good, labels)) < float(_supcon_loss(bad, labels))
