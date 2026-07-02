"""Tests for the acoustic (DSP) engine that run without network or audio files.

Feature extraction is exercised on a synthesized tone; the recommender is
exercised with a fake catalog + analyzer, so no previews are downloaded.
"""

from __future__ import annotations

import numpy as np
import pytest

from soundalike.audio.features import (
    FEATURE_NAMES,
    AcousticFeatures,
    features_from_signal,
)
from soundalike.audio.previews import DeezerTrack, _parse_track
from soundalike.audio.recommender import AudioSimilarityRecommender
from soundalike.audio.store import FeatureStore


def _sine(freq: float, sr: int = 22050, seconds: float = 3.0) -> np.ndarray:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# --------------------------------------------------------------- DSP extraction
def test_features_from_signal_shape_and_order():
    feats = features_from_signal(_sine(220.0), 22050)
    vec = feats.vector()
    assert vec.shape == (len(FEATURE_NAMES),)
    assert len(FEATURE_NAMES) == 7 + 13  # 7 scalar features + 13 MFCCs
    assert np.all(np.isfinite(vec))


def test_brighter_tone_has_higher_centroid():
    low = features_from_signal(_sine(220.0), 22050)
    high = features_from_signal(_sine(3520.0), 22050)
    # A higher-frequency tone must have a higher spectral centroid (brightness).
    assert high.spectral_centroid > low.spectral_centroid


def test_features_from_empty_signal_raises():
    with pytest.raises(ValueError):
        features_from_signal(np.array([], dtype=np.float32), 22050)


def test_acoustic_features_roundtrip():
    feats = features_from_signal(_sine(440.0), 22050)
    restored = AcousticFeatures.from_dict(feats.to_dict())
    assert np.allclose(restored.vector(), feats.vector())


# --------------------------------------------------------------------- store
def test_feature_store_roundtrip(tmp_path):
    store = FeatureStore(path=tmp_path / "cache.json")
    feats = features_from_signal(_sine(440.0), 22050)
    key = FeatureStore.key("deezer", 123)
    assert key not in store
    store.put(key, feats)
    store.save()

    reloaded = FeatureStore(path=tmp_path / "cache.json")
    assert key in reloaded
    assert np.allclose(reloaded.get(key).vector(), feats.vector())


# ------------------------------------------------------------- previews parsing
def test_parse_track():
    raw = {"id": 42, "title": "X", "preview": "http://p", "artist": {"id": 7, "name": "A"}}
    track = _parse_track(raw)
    assert track.id == 42 and track.artist_id == 7 and track.has_preview
    assert _parse_track({}) is None
    no_prev = _parse_track({"id": 1, "title": "Y", "artist": {"id": 2, "name": "B"}})
    assert not no_prev.has_preview


# ----------------------------------------------------------------- recommender
class _FakeCatalog:
    """A tiny in-memory Deezer stand-in with two acoustic 'clusters'."""

    def __init__(self):
        # id -> (track, base_frequency for the fake analyzer)
        self.tracks = {
            1: (DeezerTrack(1, "Seed Low", "ArtLow", 10, "p1"), 220.0),
            2: (DeezerTrack(2, "Low Neighbour", "ArtLow", 10, "p2"), 233.0),
            3: (DeezerTrack(3, "Low Cousin", "ArtLow2", 11, "p3"), 210.0),
            4: (DeezerTrack(4, "High One", "ArtHigh", 20, "p4"), 3520.0),
            5: (DeezerTrack(5, "High Two", "ArtHigh", 20, "p5"), 3400.0),
        }

    def search_track(self, title, artist=None):
        for track, _ in self.tracks.values():
            if track.title.casefold() == title.casefold():
                return track
        return None

    def gather_candidates(self, seeds, per_artist=25, related_per_seed=6):
        return {tid: t for tid, (t, _) in self.tracks.items()}

    def download_preview(self, track, dest):  # not used (analyzer is faked)
        return dest


def test_audio_recommender_ranks_by_acoustic_similarity(tmp_path):
    catalog = _FakeCatalog()
    freq_by_id = {tid: freq for tid, (_, freq) in catalog.tracks.items()}

    def fake_analyzer(path: str) -> AcousticFeatures:
        # `path` ends with "<id>.mp3"; synthesize a tone at that track's freq.
        import os
        track_id = int(os.path.basename(path).split(".")[0])
        return features_from_signal(_sine(freq_by_id[track_id]), 22050)

    rec = AudioSimilarityRecommender(
        client=catalog,
        store=FeatureStore(path=tmp_path / "c.json"),
        analyzer=fake_analyzer,
    )
    results, unmatched = rec.recommend([("Seed Low", "ArtLow")], n=4)
    assert unmatched == []
    titles = [r.title for r in results]
    # The two other low-frequency tracks must rank above the high-frequency ones.
    assert titles.index("Low Neighbour") < titles.index("High One")
    assert titles.index("Low Cousin") < titles.index("High Two")
    assert "Seed Low" not in titles  # seed excluded


def test_audio_recommender_uses_cache(tmp_path):
    catalog = _FakeCatalog()
    calls = {"n": 0}

    def counting_analyzer(path: str) -> AcousticFeatures:
        calls["n"] += 1
        return features_from_signal(_sine(440.0), 22050)

    store = FeatureStore(path=tmp_path / "c.json")
    rec = AudioSimilarityRecommender(client=catalog, store=store, analyzer=counting_analyzer)
    rec.recommend([("Seed Low", "ArtLow")], n=3)
    first = calls["n"]
    assert first == len(catalog.tracks)  # analyzed everything once

    rec.recommend([("Seed Low", "ArtLow")], n=3)
    assert calls["n"] == first  # second run fully served from cache
