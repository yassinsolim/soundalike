"""Tests for the live-integration logic that can run without network access."""

from __future__ import annotations

import base64
import hashlib

import pytest

from soundalike.config import Config
from soundalike.lastfm.client import SimilarTrack
from soundalike.lastfm.recommender import LastFmRecommender
from soundalike.spotify.auth import (
    Token,
    build_authorize_url,
    code_challenge,
    generate_code_verifier,
)
from soundalike.spotify.client import _normalize_track


# ------------------------------------------------------------------- PKCE / auth
def test_code_challenge_matches_spec():
    verifier = generate_code_verifier()
    expected = base64.urlsafe_b64encode(
        hashlib.sha256(verifier.encode("ascii")).digest()
    ).decode("ascii").rstrip("=")
    assert code_challenge(verifier) == expected
    assert "=" not in code_challenge(verifier)  # url-safe, unpadded


def test_build_authorize_url_has_required_params():
    url = build_authorize_url(
        "CID", "http://127.0.0.1:8888/callback", ["user-top-read"], "state123", "chal"
    )
    assert url.startswith("https://accounts.spotify.com/authorize?")
    for fragment in ("client_id=CID", "code_challenge_method=S256", "state=state123",
                     "response_type=code"):
        assert fragment in url


def test_token_expiry_and_roundtrip():
    fresh = Token("a", "r", expires_at=9_999_999_999.0)
    assert not fresh.expired()
    stale = Token("a", "r", expires_at=0.0)
    assert stale.expired()
    restored = Token.from_json(fresh.to_json())
    assert restored == fresh


def test_token_from_response_keeps_previous_refresh():
    previous = Token("old", "REFRESH", expires_at=0.0, scope="s")
    new = Token.from_response({"access_token": "new", "expires_in": 3600}, previous=previous)
    assert new.access_token == "new"
    assert new.refresh_token == "REFRESH"  # carried over
    assert new.scope == "s"


# --------------------------------------------------------------- track normalize
def test_normalize_track():
    raw = {
        "id": "abc",
        "name": "One Dance",
        "uri": "spotify:track:abc",
        "artists": [{"id": "1", "name": "Drake"}, {"id": "2", "name": "WizKid"}],
    }
    track = _normalize_track(raw)
    assert track["title"] == "One Dance"
    assert track["artist"] == "Drake, WizKid"
    assert track["primary_artist"] == "Drake"
    assert track["uri"] == "spotify:track:abc"


def test_normalize_track_none_for_empty():
    assert _normalize_track({}) is None
    assert _normalize_track({"name": "x"}) is None  # no id


# ------------------------------------------------------------------ config guards
def test_config_requires(monkeypatch):
    monkeypatch.delenv("SPOTIFY_CLIENT_ID", raising=False)
    monkeypatch.delenv("LASTFM_API_KEY", raising=False)
    cfg = Config(spotify_client_id=None, spotify_redirect_uri="x", lastfm_api_key=None)
    with pytest.raises(RuntimeError, match="SPOTIFY_CLIENT_ID"):
        cfg.require_spotify()
    with pytest.raises(RuntimeError, match="LASTFM_API_KEY"):
        cfg.require_lastfm()


# ------------------------------------------------------------ lastfm aggregation
class _FakeLastFm:
    """Returns canned similar tracks keyed by seed title."""

    def __init__(self, table):
        self.table = table

    def similar_tracks(self, artist, title, limit=50):
        return self.table.get(title, [])


def test_lastfm_recommender_aggregates_and_ranks():
    table = {
        "Song A": [SimilarTrack("Shared", "X", 0.8), SimilarTrack("OnlyA", "Y", 0.9)],
        "Song B": [SimilarTrack("Shared", "X", 0.7), SimilarTrack("OnlyB", "Z", 0.6)],
    }
    engine = LastFmRecommender(_FakeLastFm(table))
    recs, skipped = engine.recommend(
        [("Song A", "Art1"), ("Song B", "Art2")], n=10, per_seed=50
    )
    by_title = {r.title: r for r in recs}
    # "Shared" appears for both seeds: score 0.8 + 0.7 = 1.5, hits = 2
    assert by_title["Shared"].seed_hits == 2
    assert by_title["Shared"].score == pytest.approx(1.5)
    # ranked highest first
    assert recs[0].title == "Shared"
    assert skipped == []


def test_lastfm_recommender_skips_seeds_without_artist_and_excludes_seeds():
    table = {"Song A": [SimilarTrack("Song B", "Art2", 0.5), SimilarTrack("New", "N", 0.4)]}
    engine = LastFmRecommender(_FakeLastFm(table))
    recs, skipped = engine.recommend(
        [("Song A", "Art1"), ("Song B", "Art2"), ("NoArtist", None)], n=10
    )
    titles = [r.title for r in recs]
    assert ("NoArtist", None) in skipped
    assert "Song B" not in titles  # excluded because it's a seed
    assert "New" in titles
