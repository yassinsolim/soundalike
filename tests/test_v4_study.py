"""Focused tests for V4 active-study ranking and cache integrity."""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from soundalike.ml import v4_study


def test_artist_diverse_skips_gated_candidates():
    selected = v4_study._artist_diverse(
        [10, 20, 30, 40],
        np.asarray([0.8, -np.inf, 0.7, 0.6]),
        {10: 1, 20: 2, 30: 3, 40: 4},
        3,
    )
    assert selected == (10, 30, 40)
    with pytest.raises(v4_study.V4StudyError, match="lacks enough"):
        v4_study._artist_diverse(
            [10, 20],
            np.asarray([0.8, -np.inf]),
            {10: 1, 20: 2},
            2,
        )


def test_artist_unique_pool_keeps_closest_track_per_artist():
    tracks = [
        SimpleNamespace(track_id=10, artist_id=1),
        SimpleNamespace(track_id=11, artist_id=1),
        *[
            SimpleNamespace(track_id=20 + index, artist_id=2 + index)
            for index in range(15)
        ],
    ]
    positions = np.arange(len(tracks), dtype=np.int64)
    similarities = np.linspace(1.0, 0.0, len(tracks))
    selected = v4_study._artist_unique_pool(
        positions, similarities, tracks, 200
    )
    assert selected[:2].tolist() == [0, 2]
    assert len(selected) == 16
    assert len({tracks[position].artist_id for position in selected}) == 16


def test_seed_selection_excludes_non_song_length_edges():
    assert v4_study.MINIMUM_SEED_SECONDS == 90.0
    assert v4_study.MAXIMUM_SEED_SECONDS == 480.0


def test_repeated_vibe_cache_reuses_exact_rows_and_extracts_only_missing(
    tmp_path, monkeypatch
):
    cache = tmp_path / "vibe.npz"
    np.savez_compressed(
        cache,
        track_ids=np.asarray([1, 2], dtype=np.int64),
        starts=np.asarray([0.0, 1.0]),
        ends=np.asarray([20.0, 21.0]),
        vibe=np.stack(
            [
                np.full(29, 1.0, dtype=np.float32),
                np.full(29, 2.0, dtype=np.float32),
            ]
        ),
    )
    calls = []

    class Executor:
        def __init__(self, max_workers):
            assert max_workers == 2

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def map(self, function, tasks, chunksize):
            assert chunksize == 4
            for task in tasks:
                calls.append(task[0])
                yield task[0], np.full(29, task[0], dtype=np.float64)

    monkeypatch.setattr(v4_study, "ProcessPoolExecutor", Executor)
    result = v4_study._load_or_extract_vibe(
        cache,
        np.asarray([2, 3], dtype=np.int64),
        {
            2: {"start_seconds": 1.0, "end_seconds": 21.0},
            3: {"start_seconds": 2.0, "end_seconds": 22.0},
        },
        {
            2: SimpleNamespace(audio_path=tmp_path / "2.mp3"),
            3: SimpleNamespace(audio_path=tmp_path / "3.mp3"),
        },
        2,
    )
    assert calls == [3]
    assert result.dtype == np.float32
    assert np.all(result[0] == 2.0)
    assert np.all(result[1] == 3.0)
    with np.load(cache, allow_pickle=False) as archive:
        assert np.array_equal(archive["track_ids"], [2, 3])
        assert np.array_equal(archive["vibe"], result)
    warm = v4_study._load_or_extract_vibe(
        cache,
        np.asarray([2, 3], dtype=np.int64),
        {
            2: {"start_seconds": 1.0, "end_seconds": 21.0},
            3: {"start_seconds": 2.0, "end_seconds": 22.0},
        },
        {
            2: SimpleNamespace(audio_path=tmp_path / "2.mp3"),
            3: SimpleNamespace(audio_path=tmp_path / "3.mp3"),
        },
        2,
    )
    assert calls == [3]
    assert np.array_equal(warm, result)


def test_gate_cache_rejects_tampering(tmp_path):
    cache = {
        "schema_version": 2,
        "gate_kind": "soundalike_v4_study_track_gates_v2",
        "source_fingerprint": "source",
        "tracks": {
            "10": {"vocal_state": "vocal", "language": "en"},
            "20": {"vocal_state": "unknown", "language": "unknown"},
        },
    }
    cache["content_sha256"] = v4_study._content_sha256(cache)
    path = tmp_path / "gates.json"
    path.write_text(json.dumps(cache), encoding="utf-8")
    assert (
        v4_study._load_gate_cache(path, source_fingerprint="source") == cache
    )

    cache["tracks"]["20"]["language"] = "es"
    cache["content_sha256"] = v4_study._content_sha256(cache)
    path.write_text(json.dumps(cache), encoding="utf-8")
    with pytest.raises(v4_study.V4StudyError, match="binding failed"):
        v4_study._load_gate_cache(path, source_fingerprint="source")
