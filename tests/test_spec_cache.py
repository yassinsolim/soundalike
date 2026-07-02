"""Tests for the harvest-once spec cache (data structure only, no network)."""

from __future__ import annotations

import numpy as np

from soundalike.ml.spec_cache import SpecCache


def _spec():
    return np.random.randn(128, 256).astype(np.float32)


def _vibe():
    return np.random.randn(29).astype(np.float32)


def test_add_and_len():
    c = SpecCache()
    c.add(1, "Song A", "Artist A", _spec(), _vibe())
    c.add(2, "Song B", "Artist B", _spec(), _vibe())
    assert len(c) == 2
    assert c.has(1) and c.has(2) and not c.has(3)


def test_dedup_by_track_id():
    c = SpecCache()
    c.add(7, "First", "A", _spec(), _vibe())
    c.add(7, "Duplicate", "A", _spec(), _vibe())
    assert len(c) == 1
    assert c.titles[0] == "First"


def test_specs_stored_float16():
    c = SpecCache()
    c.add(1, "S", "A", _spec(), _vibe())
    assert c.specs[0].dtype == np.float16
    assert c.specs[0].shape == (128, 256)


def test_save_load_roundtrip(tmp_path):
    c = SpecCache()
    for i in range(5):
        c.add(i, f"T{i}", f"A{i}", _spec(), _vibe())
    p = tmp_path / "cache.npz"
    c.save(p)
    d = SpecCache.load(p)
    assert len(d) == 5
    assert list(d.track_ids) == [0, 1, 2, 3, 4]
    assert d.specs[0].shape == (128, 256)
    assert d.vibe[0].shape == (29,)
    # Loading a saved cache preserves dedup state.
    d.add(2, "dup", "x", _spec(), _vibe())
    assert len(d) == 5


def test_load_numpy_array_ids(tmp_path):
    # Regression: constructor must accept numpy arrays (from np.load) without
    # tripping the ambiguous-truth-value error.
    c = SpecCache()
    c.add(100, "T", "A", _spec(), _vibe())
    p = tmp_path / "c.npz"
    c.save(p)
    d = SpecCache.load(p)
    assert isinstance(d.track_ids[0], int)
    assert d.has(100)
