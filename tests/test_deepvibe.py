"""Tests for the deep-vibe fusion recommender (network-free)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from soundalike.audio.vibe import vibe_from_signal
from soundalike.ml.deepvibe import DeepVibeIndex, DeepVibeRecommender


def _tone(freq, sr=22050, seconds=4.0, amp=0.5):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _toy_index():
    # 4 tracks: 2 "bassy" (neural cluster A + bass vibe), 2 "bright".
    rng = np.random.default_rng(0)
    a = rng.normal(0, 0.01, size=256).astype(np.float32)
    b = rng.normal(5, 0.01, size=256).astype(np.float32)
    neural = np.stack([a, a + 0.01, b, b + 0.01])
    vibe = np.stack([
        vibe_from_signal(_tone(50), 22050).vector(),
        vibe_from_signal(_tone(55), 22050).vector(),
        vibe_from_signal(_tone(6000), 22050).vector(),
        vibe_from_signal(_tone(7000), 22050).vector(),
    ]).astype(np.float32)
    return DeepVibeIndex([1, 2, 3, 4], ["bassA", "bassB", "brightA", "brightB"],
                         ["x", "y", "z", "w"], neural, vibe)


def test_index_save_load_roundtrip(tmp_path):
    idx = _toy_index()
    p = tmp_path / "dv.npz"
    idx.save(p)
    loaded = DeepVibeIndex.load(p)
    assert len(loaded) == 4
    assert np.allclose(loaded.neural, idx.neural)
    assert np.allclose(loaded.vibe, idx.vibe)


def test_fusion_matches_both_signals():
    idx = _toy_index()
    rec = DeepVibeRecommender(idx, alpha=0.5)
    # Query aligned with the "bassy" cluster on both signals.
    seed_neural = idx.neural[0].copy()
    seed_vibe = vibe_from_signal(_tone(48), 22050)
    results = rec.recommend(seed_neural, seed_vibe, n=4, exclude_ids={1})
    # The other bassy track should rank first.
    assert results[0].title == "bassB"
    # bright tracks rank last
    assert results[-1].title.startswith("bright")


def test_alpha_extremes_select_different_signals():
    idx = _toy_index()
    seed_vibe = vibe_from_signal(_tone(48), 22050)   # bass-heavy vibe
    # Neural query pointing at the BRIGHT cluster, but vibe query bass-heavy:
    seed_neural = idx.neural[2].copy()

    pure_neural = DeepVibeRecommender(idx, alpha=1.0).recommend(seed_neural, seed_vibe, n=4)
    pure_vibe = DeepVibeRecommender(idx, alpha=0.0).recommend(seed_neural, seed_vibe, n=4)
    # Pure-neural should favour a bright track; pure-vibe a bassy track.
    assert pure_neural[0].title.startswith("bright")
    assert pure_vibe[0].title.startswith("bass")


def test_requires_nonempty_index():
    empty = DeepVibeIndex([], [], [], np.zeros((0, 256), np.float32), np.zeros((0, 29), np.float32))
    with pytest.raises(ValueError):
        DeepVibeRecommender(empty)


def test_exclude_ids_and_dedup():
    idx = _toy_index()
    rec = DeepVibeRecommender(idx, alpha=0.5)
    seed_vibe = vibe_from_signal(_tone(48), 22050)
    res = rec.recommend(idx.neural[0], seed_vibe, n=4, exclude_ids={1, 2})
    ids = [r.track_id for r in res]
    assert 1 not in ids and 2 not in ids
