from pathlib import Path

import pytest

from soundalike.ml.fulltrack_v3_protocol import (
    DEVELOPMENT_CUTOFF,
    TRAIN_CUTOFF,
    V3ProtocolError,
    artist_split,
    select_tracks,
)
from soundalike.ml.jamendo_fulltrack import JamendoTrack, TrackLicense


def _track(track_id: int, artist_id: int) -> JamendoTrack:
    path = f"{track_id}.mp3"
    return JamendoTrack(
        row_index=track_id,
        track_id=track_id,
        artist_id=artist_id,
        album_id=artist_id,
        relative_path=path,
        audio_path=Path("X:/audio") / path,
        duration_seconds=180.0,
        tags=("genre---rock",),
        title=str(track_id),
        artist_name=str(artist_id),
        album_name=str(artist_id),
        release_date="2020-01-01",
        jamendo_url=f"https://www.jamendo.com/track/{track_id}",
        license=TrackLicense(
            path=path,
            attribution="test",
            name="CC BY 3.0",
            url="https://creativecommons.org/licenses/by/3.0/",
            permits_commercial_use=True,
            permits_derivatives=True,
        ),
        expected_audio_sha256="a" * 64,
        expected_audio_bytes=100,
    )


def test_artist_split_is_deterministic_and_artist_disjoint():
    first = {artist_id: artist_split(artist_id) for artist_id in range(1, 500)}
    second = {artist_id: artist_split(artist_id) for artist_id in range(1, 500)}
    assert first == second
    assert set(first.values()) == {"train", "development", "shadow"}
    assert 0 < TRAIN_CUTOFF < DEVELOPMENT_CUTOFF < 10_000


def test_track_selection_is_deterministic_and_unique():
    tracks = tuple(_track(index, index) for index in range(1, 30))
    first = select_tracks(tracks, limit=10, seed=5)
    second = select_tracks(tuple(reversed(tracks)), limit=10, seed=5)
    assert first == second
    assert len({track.track_id for track in first}) == 10


def test_track_selection_rejects_invalid_limit():
    with pytest.raises(V3ProtocolError, match="positive integer"):
        select_tracks((_track(1, 1),), limit=0)
    with pytest.raises(V3ProtocolError, match="not enough"):
        select_tracks((_track(1, 1),), limit=2)
