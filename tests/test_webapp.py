"""Tests for the Vercel web recommender (webapp/api/_reco.py).

The hosted library-mode recommender is a torch-free numpy reimplementation of
DeepVibeRecommender. These tests pin it to the canonical recommender so the two
can never silently diverge, and cover the query parser.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_API = Path(__file__).resolve().parents[1] / "webapp" / "api"
sys.path.insert(0, str(_API))


def _synthetic_index(tmp_path, n_artists=60, per=5, dim=48, seed=0):
    """Build + save a small DeepVibeIndex so both recommenders read the same data."""
    from soundalike.ml.deepvibe import DeepVibeIndex

    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_artists, dim))
    neural, vibe, titles, artists, tids = [], [], [], [], []
    k = 0
    for a in range(n_artists):
        for j in range(per):
            neural.append(centers[a] + 0.2 * rng.standard_normal(dim))
            vibe.append(rng.standard_normal(29))
            titles.append(f"song {k}")
            artists.append(f"artist {a}")
            tids.append(1000 + k)
            k += 1
    idx = DeepVibeIndex(np.array(tids), np.array(titles, object),
                        np.array(artists, object),
                        np.asarray(neural, np.float32), np.asarray(vibe, np.float32))
    p = tmp_path / "idx.npz"
    idx.save(p)
    return p, idx


def test_web_recommender_matches_canonical(tmp_path):
    import os
    os.environ["SOUNDALIKE_INDEX_PATH"] = ""  # force explicit path use
    from _reco import WebRecommender
    from soundalike.ml.deepvibe import DeepVibeIndex, DeepVibeRecommender
    from soundalike.audio.vibe import VibeFeatures

    path, idx = _synthetic_index(tmp_path)
    web = WebRecommender(str(path))
    canon = DeepVibeRecommender(DeepVibeIndex.load(path), alpha=0.8, whiten=True)

    for row in (0, 37, 111, 200, 250):
        w = web.recommend(row, n=15, alpha=0.8, diversity=0.15, max_per_artist=1)
        c = canon.recommend(
            np.asarray(idx.neural[row], np.float32),
            VibeFeatures.from_vector(np.asarray(idx.vibe[row], np.float32)),
            n=15, exclude_ids={int(idx.track_ids[row])},
            exclude_artist=str(idx.artists[row]), diversity=0.15, max_per_artist=1)
        assert [(x["title"], x["artist"]) for x in w["results"]] == \
               [(r.title, r.artist) for r in c], f"mismatch at row {row}"


def test_web_recommender_search_and_findrow(tmp_path):
    from _reco import WebRecommender

    path, _ = _synthetic_index(tmp_path)
    web = WebRecommender(str(path))
    assert web.find_row("song 0", "artist 0") == 0
    assert web.find_row("song 7") == 7  # unambiguous title
    hits = web.search("song 1", limit=5)
    assert hits and all("title" in h and "row" in h for h in hits)


def test_split_query_parsing():
    import recommend as rec
    assert rec._split("Plastic Love — Mariya Takeuchi") == ("Plastic Love", "Mariya Takeuchi")
    assert rec._split("Redbone by Childish Gambino") == ("Redbone", "Childish Gambino")
    assert rec._split("Windowlicker") == ("Windowlicker", "")
