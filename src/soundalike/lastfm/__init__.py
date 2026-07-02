"""Last.fm similarity engine (cross-catalog, works for any track)."""

from .client import LastFmClient, LastFmError, SimilarTrack
from .recommender import LastFmRecommender, Suggestion

__all__ = ["LastFmClient", "LastFmError", "SimilarTrack", "LastFmRecommender", "Suggestion"]
