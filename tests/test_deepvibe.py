"""Tests for the deep-vibe fusion recommender (network-free)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from soundalike.audio.vibe import VibeFeatures, vibe_from_signal
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
    assert loaded.sonic is None


def test_index_sonic_roundtrip(tmp_path):
    idx = _toy_index()
    idx.sonic = np.arange(4 * 64, dtype=np.float16).reshape(4, 64)
    path = tmp_path / "sonic.npz"
    idx.save(path, half=True)
    loaded = DeepVibeIndex.load(path)
    assert loaded.sonic.dtype == np.float16
    assert np.array_equal(loaded.sonic, idx.sonic)


def test_index_dual_sonic_roundtrip(tmp_path):
    idx = _toy_index()
    idx.sonic = np.arange(4 * 64, dtype=np.float16).reshape(4, 64)
    idx.clap = np.flip(idx.sonic, axis=1).copy()
    idx.wiki = np.arange(4, dtype=np.float16)
    idx.wiki_specific = np.array([0, 1, 0, 1], dtype=np.uint8)
    path = tmp_path / "dual.npz"
    idx.save(path, half=True)
    loaded = DeepVibeIndex.load(path)
    assert np.array_equal(loaded.clap, idx.clap)
    assert np.array_equal(loaded.wiki, idx.wiki)
    assert np.array_equal(loaded.wiki_specific, idx.wiki_specific)


def test_dual_sonic_preserves_guarded_head_and_baseline_top_ten():
    rng = np.random.default_rng(32)
    count = 40
    idx = DeepVibeIndex(
        np.arange(count),
        [f"title {i}" for i in range(count)],
        [f"artist {i}" for i in range(count)],
        rng.normal(size=(count, 24)).astype(np.float32),
        rng.normal(size=(count, 29)).astype(np.float32),
        rng.normal(size=(count, 64)).astype(np.float16),
        rng.normal(size=(count, 64)).astype(np.float16),
        rng.integers(0, 6, count).astype(np.float16),
        rng.integers(0, 2, count).astype(np.uint8),
    )
    rec = DeepVibeRecommender(idx, enhance=True)
    vibe = VibeFeatures.from_vector(idx.vibe[0])
    guarded = rec.recommend(
        idx.neural[0], vibe, n=5, exclude_ids={0}, diversity=.15,
        max_per_artist=1,
    )
    baseline = rec.recommend(
        idx.neural[0], vibe, n=10, exclude_ids={0}, diversity=.15,
        max_per_artist=1, genre_rerank=False,
    )
    dual = rec.recommend(
        idx.neural[0], vibe, n=25, exclude_ids={0},
        exclude_artist="artist 0", seed_title="title 0", diversity=.15,
        max_per_artist=1, seed_row=0,
    )
    ids = [item.track_id for item in dual]
    assert ids[:5] == [item.track_id for item in guarded]
    assert {item.track_id for item in baseline} <= set(ids[:15])
    assert rec.last_retrieval_mode == "dual_sonic64_guardrail"


def test_stable_sonic_preserves_head_and_changes_tail():
    rng = np.random.default_rng(31)
    count = 30
    idx = DeepVibeIndex(
        np.arange(count),
        [f"title {i}" for i in range(count)],
        [f"artist {i}" for i in range(count)],
        rng.normal(size=(count, 24)).astype(np.float32),
        rng.normal(size=(count, 29)).astype(np.float32),
        rng.normal(size=(count, 64)).astype(np.float16),
    )
    rec = DeepVibeRecommender(idx, enhance=True)
    vibe = VibeFeatures.from_vector(idx.vibe[0])
    legacy = rec.recommend(
        idx.neural[0], vibe, n=15, exclude_ids={0}, diversity=.15,
        max_per_artist=1,
    )
    sonic = rec.recommend(
        idx.neural[0], vibe, n=15, exclude_ids={0}, diversity=.15,
        max_per_artist=1, seed_row=0,
    )
    assert len(sonic) == 15
    assert [item.track_id for item in sonic[:5]] == [
        item.track_id for item in legacy[:5]
    ]
    assert [item.track_id for item in sonic[5:]] != [
        item.track_id for item in legacy[5:]
    ]
    short = rec.recommend(
        idx.neural[0], vibe, n=3, exclude_ids={0}, diversity=.15,
        max_per_artist=1, seed_row=0,
    )
    assert len(short) == 3
    assert rec.last_retrieval_mode == "sonic64_stable_head"


def test_sonic_index_without_query_reports_explicit_fallback():
    idx = _toy_index()
    idx.sonic = np.eye(4, 64, dtype=np.float16)
    rec = DeepVibeRecommender(idx)
    rec.recommend(idx.neural[0], VibeFeatures.from_vector(idx.vibe[0]), n=3)
    assert rec.last_retrieval_mode == "legacy_no_sonic_seed"


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
