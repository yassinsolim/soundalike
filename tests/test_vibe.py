"""Tests for the vibe engine (frequency bands + dynamics), network-free."""

from __future__ import annotations

import numpy as np
import pytest

from soundalike.audio.vibe import (
    DEFAULT_WEIGHTS,
    FEATURE_NAMES,
    VibeFeatures,
    vibe_from_signal,
    weight_vector,
)
from soundalike.audio.vibe_index import (
    VibeEntry,
    VibeIndex,
    VibeRecommender,
)


def _tone(freq: float, sr: int = 22050, seconds: float = 4.0, amp: float = 0.5) -> np.ndarray:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _dynamic_signal(sr: int = 22050) -> np.ndarray:
    """Quiet first half, loud second half — a big 'drop' for dynamics tests."""
    quiet = _tone(60, sr, 3.0, amp=0.05)
    loud = _tone(60, sr, 3.0, amp=0.9)
    return np.concatenate([quiet, loud])


# ------------------------------------------------------------------- features
def test_vector_length_matches_names():
    feats = vibe_from_signal(_tone(200), 22050)
    assert feats.vector().shape[0] == len(FEATURE_NAMES) == 29


def test_bands_identify_sub_vs_air():
    sub = vibe_from_signal(_tone(45), 22050)
    air = vibe_from_signal(_tone(8000), 22050)
    # A 45 Hz tone is almost all sub-band energy; an 8 kHz tone almost none.
    assert sub.bands[0] > 0.8
    assert sub.low_end_ratio > 0.8
    assert air.bands[0] < 0.05
    assert air.low_end_ratio < 0.1


def test_dynamics_capture_the_drop():
    steady = vibe_from_signal(_tone(60, seconds=6.0), 22050)
    dropping = vibe_from_signal(_dynamic_signal(), 22050)
    # The drop signal must show more movement and a bigger crest than the steady one.
    assert dropping.crest > steady.crest
    assert dropping.dynamic_range > steady.dynamic_range
    assert dropping.rms_std > steady.rms_std


def test_roundtrip_and_describe():
    feats = vibe_from_signal(_tone(120), 22050)
    restored = VibeFeatures.from_dict(feats.to_dict())
    assert np.allclose(restored.vector(), feats.vector())
    d = feats.describe()
    assert set(d) == {"tempo", "dynamics", "low_end", "tone"}


def test_default_weights_emphasize_lowend_and_dynamics():
    w = weight_vector(DEFAULT_WEIGHTS)
    names = {n: w[i] for i, n in enumerate(FEATURE_NAMES)}
    # The vibe-defining features should outweigh a default (1.0) feature.
    assert names["low_end_ratio"] > 1.0
    assert names["crest"] > 1.0
    assert names["band_sub"] > 1.0
    assert names["mfcc_5"] == 1.0


# ---------------------------------------------------------------- recommender
def _entry(track_id, freq, amp=0.5, dynamic=False, title=None, artist="A"):
    sig = _dynamic_signal() if dynamic else _tone(freq, amp=amp)
    feats = vibe_from_signal(sig, 22050)
    return VibeEntry(track_id, title or f"t{track_id}", artist, feats)


def test_recommender_matches_bass_heavy_to_bass_heavy():
    # Library: two bass-heavy tracks, two bright tracks.
    entries = [
        _entry(1, 50, title="bassy1"),
        _entry(2, 55, title="bassy2"),
        _entry(3, 6000, title="bright1"),
        _entry(4, 7000, title="bright2"),
    ]
    index = VibeIndex(entries)
    rec = VibeRecommender(index)
    seed = vibe_from_signal(_tone(48), 22050)  # bass-heavy query
    results = rec.recommend(seed, n=4)
    top2 = {r.title for r in results[:2]}
    assert top2 == {"bassy1", "bassy2"}  # bass-heavy neighbours rank first


def test_recommender_excludes_seed_id_and_dedups():
    entries = [_entry(1, 50, title="x"), _entry(2, 50, title="x"), _entry(3, 6000, title="y")]
    index = VibeIndex(entries)
    rec = VibeRecommender(index)
    seed = vibe_from_signal(_tone(50), 22050)
    results = rec.recommend(seed, n=5, exclude_ids={1})
    ids = [r.track_id for r in results]
    assert 1 not in ids
    # duplicate (title x, artist A) should appear once
    titles = [r.title for r in results]
    assert titles.count("x") <= 1


def test_index_save_load_roundtrip(tmp_path):
    entries = [_entry(1, 50), _entry(2, 6000)]
    index = VibeIndex(entries)
    p = tmp_path / "idx.json"
    index.save(p)
    loaded = VibeIndex.load(p)
    assert len(loaded) == 2
    assert np.allclose(loaded.matrix, index.matrix)


def test_recommender_requires_nonempty_index():
    with pytest.raises(ValueError):
        VibeRecommender(VibeIndex([]))
