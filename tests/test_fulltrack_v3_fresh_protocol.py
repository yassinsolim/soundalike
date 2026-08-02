from __future__ import annotations

import pytest

from soundalike.ml.fulltrack_v3_fresh_protocol import (
    V3FreshProtocolError,
    fresh_artist_split,
)


def test_consumed_artists_are_always_training_only():
    consumed = {11, 22, 33}
    assert all(
        fresh_artist_split(artist_id, consumed) == "train"
        for artist_id in consumed
    )


def test_fresh_artist_split_is_deterministic_and_balanced():
    consumed = {1, 2, 3}
    first = {
        artist_id: fresh_artist_split(artist_id, consumed)
        for artist_id in range(4, 1_000)
    }
    second = {
        artist_id: fresh_artist_split(artist_id, consumed)
        for artist_id in reversed(range(4, 1_000))
    }
    assert first == second
    assert set(first.values()) == {"development", "shadow"}
    development = sum(split == "development" for split in first.values())
    assert 400 < development < 600


def test_fresh_artist_split_rejects_invalid_ids():
    with pytest.raises(V3FreshProtocolError, match="positive integers"):
        fresh_artist_split(0, set())
    with pytest.raises(V3FreshProtocolError, match="positive integers"):
        fresh_artist_split(4, {True})
