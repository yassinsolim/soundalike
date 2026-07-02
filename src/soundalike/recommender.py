"""Content-based music recommender.

Finds songs whose audio-feature profile is closest to a seed song or to the
centroid of a set of liked songs (a "taste profile"). Features are standardized
so different scales (bpm vs. 0-100 percentages) are comparable, then optionally
weighted before computing similarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import StandardScaler

from .dataset import Dataset
from .features import FeatureConfig


@dataclass
class Recommendation:
    """A single recommended song and how strongly it matched."""

    title: str
    artist: str
    score: float
    index: int

    def __str__(self) -> str:
        return f"{self.title} — {self.artist}  ({self.score:.3f})"


class ContentBasedRecommender:
    """Recommends songs by audio-feature similarity within a Dataset."""

    def __init__(self, config: Optional[FeatureConfig] = None):
        self.config = (config or FeatureConfig()).validate()
        self.dataset: Optional[Dataset] = None
        self._scaler: Optional[StandardScaler] = None
        self._matrix: Optional[np.ndarray] = None

    # -------------------------------------------------------------------- fit
    def fit(self, dataset: Dataset) -> "ContentBasedRecommender":
        if len(dataset) == 0:
            raise ValueError("Cannot fit on an empty dataset.")
        self.dataset = dataset
        raw = dataset.feature_matrix(self.config.features)
        self._scaler = StandardScaler().fit(raw)
        self._matrix = self._weight(self._scaler.transform(raw))
        return self

    def _weight(self, scaled: np.ndarray) -> np.ndarray:
        weights = np.asarray(self.config.weight_vector(), dtype=float)
        # Scaling each standardized feature by sqrt(weight) makes squared
        # Euclidean distance a proper weighted distance, and emphasizes the
        # same dimensions under cosine similarity.
        return scaled * np.sqrt(np.clip(weights, 0.0, None))

    def _require_fit(self) -> None:
        if self._matrix is None or self.dataset is None:
            raise RuntimeError("Recommender is not fitted. Call fit(dataset) first.")

    # ---------------------------------------------------------------- scoring
    def _scores(self, query_vec: np.ndarray) -> np.ndarray:
        query = query_vec.reshape(1, -1)
        if self.config.metric == "cosine":
            return cosine_similarity(query, self._matrix).ravel()
        distances = euclidean_distances(query, self._matrix).ravel()
        return 1.0 / (1.0 + distances)

    def _rank(
        self,
        scores: np.ndarray,
        n: int,
        exclude_indices: Optional[Set[int]] = None,
        exclude_artist_keys: Optional[Set[str]] = None,
    ) -> List[Recommendation]:
        self._require_fit()
        frame = self.dataset.frame
        exclude_indices = exclude_indices or set()
        exclude_artist_keys = exclude_artist_keys or set()

        order = np.argsort(scores)[::-1]
        results: List[Recommendation] = []
        seen_keys: Set[str] = set()
        for idx in order:
            idx = int(idx)
            if idx in exclude_indices:
                continue
            dedup_key = frame.at[idx, "_dedup_key"]
            if dedup_key in seen_keys:
                continue
            if frame.at[idx, "_artist_key"] in exclude_artist_keys:
                continue
            seen_keys.add(dedup_key)
            results.append(
                Recommendation(
                    title=str(frame.at[idx, "title"]),
                    artist=str(frame.at[idx, "artist"]),
                    score=float(scores[idx]),
                    index=idx,
                )
            )
            if len(results) >= n:
                break
        return results

    # ------------------------------------------------------------ public API
    def similar_to(
        self,
        title: str,
        artist: Optional[str] = None,
        n: int = 10,
        exclude_same_artist: bool = False,
    ) -> List[Recommendation]:
        """Recommend songs similar to a single seed song in the dataset."""
        self._require_fit()
        idx = self.dataset.find_one(title, artist)
        if idx is None:
            raise LookupError(f"No song matching '{title}'" + (f" by '{artist}'" if artist else ""))
        frame = self.dataset.frame
        exclude_indices = {idx}
        exclude_artist_keys = {frame.at[idx, "_artist_key"]} if exclude_same_artist else set()
        scores = self._scores(self._matrix[idx])
        return self._rank(scores, n, exclude_indices, exclude_artist_keys)

    def recommend_for_profile(
        self,
        seeds: Sequence[Tuple[str, Optional[str]]],
        n: int = 20,
        exclude_known: bool = True,
        exclude_seed_artists: bool = False,
    ) -> Tuple[List[Recommendation], List[Tuple[str, Optional[str]]]]:
        """Recommend songs for a taste profile built from several seed songs.

        Returns (recommendations, unmatched_seeds).
        """
        self._require_fit()
        matched, unmatched = self.dataset.find_many(seeds)
        if not matched:
            raise LookupError("None of the seed songs were found in the dataset.")

        centroid = self._matrix[matched].mean(axis=0)
        scores = self._scores(centroid)

        frame = self.dataset.frame
        exclude_indices = set(matched) if exclude_known else set()
        exclude_artist_keys: Set[str] = set()
        if exclude_seed_artists:
            exclude_artist_keys = {frame.at[i, "_artist_key"] for i in matched}
        return self._rank(scores, n, exclude_indices, exclude_artist_keys), unmatched
