"""Tests for the local server's pure helpers (no model load, no network)."""

from __future__ import annotations

from soundalike.server import (
    DEMO_SEEDS,
    _DEEZER_RE,
    _SPOTIFY_RE,
    _SPOTIFY_URI_RE,
    _norm,
    _seed_suggestions,
    _split_text_query,
)


def test_norm_strips_accents_and_features():
    assert _norm("Café  Tacvba") == "cafe tacvba"
    assert _norm("Song (feat. Someone)") == "song"
    assert _norm("Track feat. X") == "track"
    assert _norm("  MiXeD  Case ") == "mixed case"


def test_split_text_query_formats():
    assert _split_text_query("Plastic Love — Mariya Takeuchi") == ("Plastic Love", "Mariya Takeuchi")
    assert _split_text_query("money machine - 100 gecs") == ("money machine", "100 gecs")
    assert _split_text_query("Sofia :: Clairo") == ("Sofia", "Clairo")
    assert _split_text_query("Redbone by Childish Gambino") == ("Redbone", "Childish Gambino")


def test_split_text_query_bare_title():
    assert _split_text_query("Windowlicker") == ("Windowlicker", "")


def test_spotify_link_regex_extracts_id():
    m = _SPOTIFY_RE.search("https://open.spotify.com/track/4uLU6hMCjMI75M1A2tKUQC?si=abc")
    assert m and m.group(1) == "4uLU6hMCjMI75M1A2tKUQC"
    assert _SPOTIFY_URI_RE.search("spotify:track:4uLU6hMCjMI75M1A2tKUQC").group(1) \
        == "4uLU6hMCjMI75M1A2tKUQC"


def test_deezer_link_regex_extracts_id():
    assert _DEEZER_RE.search("https://www.deezer.com/en/track/3135556").group(1) == "3135556"
    assert _DEEZER_RE.search("https://deezer.com/track/42").group(1) == "42"


def test_seed_suggestions_fall_back_to_demo_without_engine():
    # With no warm engine loaded, suggestions should be the curated demo set.
    import soundalike.server as srv

    srv._ENGINE = None
    seeds = _seed_suggestions()
    assert seeds == DEMO_SEEDS
    assert all("title" in s and "artist" in s for s in seeds)
