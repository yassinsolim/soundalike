"""Tests for the broad-harvest helpers (data structures only, no network)."""

from __future__ import annotations

from soundalike.audio.previews import DeezerTrack
from soundalike.ml.grow_broad import (
    BROAD_SEED_ARTISTS,
    _load_candidates,
    _save_candidates,
)


def test_seed_list_spans_many_scenes():
    # A broad list is the whole point — guard against it shrinking back to one
    # scene. Should be large and unique-ish.
    assert len(BROAD_SEED_ARTISTS) >= 100
    # de-dup ratio sane (a couple accidental repeats are fine)
    assert len(set(BROAD_SEED_ARTISTS)) >= len(BROAD_SEED_ARTISTS) - 3


def test_candidates_roundtrip(tmp_path):
    tracks = [
        DeezerTrack(id=1, title="A", artist="X", artist_id=10, preview_url="http://p/1.mp3"),
        DeezerTrack(id=2, title="B", artist="Y", artist_id=20, preview_url="http://p/2.mp3"),
    ]
    p = tmp_path / "cands.json"
    _save_candidates(p, tracks)
    back = _load_candidates(p)
    assert [t.id for t in back] == [1, 2]
    assert [t.title for t in back] == ["A", "B"]
    assert [t.artist for t in back] == ["X", "Y"]
    assert all(t.preview_url for t in back)


def test_candidates_roundtrip_missing_preview(tmp_path):
    # A row without a preview should still load (empty url), not crash.
    tracks = [DeezerTrack(id=5, title="T", artist="Z", artist_id=0, preview_url="")]
    p = tmp_path / "c.json"
    _save_candidates(p, tracks)
    back = _load_candidates(p)
    assert len(back) == 1 and back[0].preview_url == ""
