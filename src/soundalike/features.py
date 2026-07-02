"""Audio-feature definitions and configuration for the recommender."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

# Sonic features used for similarity. `bpm` lives on a very different scale
# (~40-220) than the 0-100 percentage features, so normalization (handled by the
# recommender via StandardScaler) is essential to make them comparable — a step
# the original class project skipped.
AUDIO_FEATURES: List[str] = [
    "bpm",
    "danceability",
    "valence",
    "energy",
    "acousticness",
    "instrumentalness",
    "liveness",
    "speechiness",
]

# Human-friendly aliases so the CLI can accept e.g. "dance" or "acoustic".
FEATURE_ALIASES: Dict[str, str] = {
    "tempo": "bpm",
    "bpm": "bpm",
    "dance": "danceability",
    "danceability": "danceability",
    "valence": "valence",
    "mood": "valence",
    "happiness": "valence",
    "energy": "energy",
    "acoustic": "acousticness",
    "acousticness": "acousticness",
    "instrumental": "instrumentalness",
    "instrumentalness": "instrumentalness",
    "live": "liveness",
    "liveness": "liveness",
    "speech": "speechiness",
    "speechiness": "speechiness",
}


def resolve_feature(name: str) -> str:
    """Resolve an alias or column name to a canonical feature name."""
    key = name.strip().lower()
    if key in FEATURE_ALIASES:
        return FEATURE_ALIASES[key]
    raise ValueError(f"Unknown feature '{name}'. Valid features: {AUDIO_FEATURES}")


@dataclass
class FeatureConfig:
    """Which features to use, how to weight them, and the distance metric.

    Attributes:
        features: Ordered list of feature columns to include.
        weights: Optional per-feature weight (default 1.0). Higher = matters more.
        metric: "euclidean" (default) or "cosine".
    """

    features: List[str] = field(default_factory=lambda: list(AUDIO_FEATURES))
    weights: Dict[str, float] = field(default_factory=dict)
    metric: str = "euclidean"

    def weight_vector(self) -> List[float]:
        return [float(self.weights.get(f, 1.0)) for f in self.features]

    def validate(self) -> "FeatureConfig":
        unknown = [f for f in self.features if f not in AUDIO_FEATURES]
        if unknown:
            raise ValueError(f"Unknown feature(s): {unknown}. Valid: {AUDIO_FEATURES}")
        if not self.features:
            raise ValueError("At least one feature is required.")
        if self.metric not in ("euclidean", "cosine"):
            raise ValueError("metric must be 'euclidean' or 'cosine'")
        for name, w in self.weights.items():
            if w < 0:
                raise ValueError(f"Weight for '{name}' must be non-negative.")
        return self
