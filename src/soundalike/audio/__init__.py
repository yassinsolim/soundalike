"""Acoustic (DSP) similarity engine — the science core of soundalike.

Finds similar songs by measuring features directly from each track's audio
waveform (tempo, energy, brightness, timbre) and comparing those measurements,
rather than relying on Spotify's or any website's precomputed similarity.
"""

from .features import (
    AcousticFeatures,
    FEATURE_DESCRIPTIONS,
    FEATURE_NAMES,
    features_from_file,
    features_from_signal,
)
from .previews import DeezerClient, DeezerTrack
from .recommender import AudioRecommendation, AudioSimilarityRecommender
from .store import FeatureStore
from .vibe import VibeFeatures, vibe_from_file, vibe_from_signal
from .vibe_index import VibeIndex, VibeRecommendation, VibeRecommender, build_index

__all__ = [
    "AcousticFeatures",
    "FEATURE_NAMES",
    "FEATURE_DESCRIPTIONS",
    "features_from_file",
    "features_from_signal",
    "DeezerClient",
    "DeezerTrack",
    "FeatureStore",
    "AudioRecommendation",
    "AudioSimilarityRecommender",
    "VibeFeatures",
    "vibe_from_file",
    "vibe_from_signal",
    "VibeIndex",
    "VibeRecommender",
    "VibeRecommendation",
    "build_index",
]
