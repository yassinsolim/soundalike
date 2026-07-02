"""Acoustic similarity recommender — ranking is 100% computed DSP features.

Pipeline:
  1. Resolve seed songs to real tracks with audio previews (Deezer).
  2. Gather a candidate pool of other tracks (Deezer catalog — enumeration only).
  3. Measure the acoustic fingerprint of every track from its actual audio.
  4. Standardize + weight the fingerprints and rank candidates by closeness to
     the seed centroid.

No collaborative / "people also liked" signal enters the ranking. Similarity is
decided by the physics of the sound.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES, AcousticFeatures, features_from_file
from .previews import DeezerClient, DeezerTrack
from .store import FeatureStore

Seed = Tuple[str, Optional[str]]

# A default emphasis: tempo, energy and timbre (MFCCs) drive the "feel" of a
# track most, so they carry a bit more weight than raw spectral summaries.
DEFAULT_WEIGHTS: Dict[str, float] = {"tempo": 1.5, "rms_energy": 1.5}


@dataclass
class AudioRecommendation:
    title: str
    artist: str
    score: float
    track_id: int

    def __str__(self) -> str:
        return f"{self.title} — {self.artist}  ({self.score:.3f})"


def _weight_vector(weights: Dict[str, float]) -> np.ndarray:
    return np.array([float(weights.get(name, 1.0)) for name in FEATURE_NAMES], dtype=float)


class AudioSimilarityRecommender:
    """Recommends songs by measured acoustic similarity."""

    def __init__(
        self,
        client: Optional[DeezerClient] = None,
        store: Optional[FeatureStore] = None,
        weights: Optional[Dict[str, float]] = None,
        analyzer: Callable[[str], AcousticFeatures] = features_from_file,
        progress: Optional[Callable[[str], None]] = None,
    ):
        self.client = client or DeezerClient()
        self.store = store if store is not None else FeatureStore()
        self.weights = weights if weights is not None else dict(DEFAULT_WEIGHTS)
        self.analyzer = analyzer
        self.progress = progress or (lambda _msg: None)

    # ------------------------------------------------------- analysis + caching
    def _analyze_track(self, track: DeezerTrack, workdir: Path) -> Optional[AcousticFeatures]:
        key = FeatureStore.key("deezer", track.id)
        cached = self.store.get(key)
        if cached is not None:
            return cached
        if not track.has_preview:
            return None
        try:
            dest = workdir / f"{track.id}.mp3"
            if self.client.download_preview(track, dest) is None:
                return None
            features = self.analyzer(str(dest))
            dest.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001 - skip tracks we can't analyze
            self.progress(f"  skip {track.title}: {type(exc).__name__}")
            return None
        self.store.put(key, features)
        return features

    def _resolve_seeds(self, seeds: Sequence[Seed]) -> Tuple[List[DeezerTrack], List[Seed]]:
        resolved: List[DeezerTrack] = []
        unmatched: List[Seed] = []
        for title, artist in seeds:
            track = self.client.search_track(title, artist)
            if track and track.has_preview:
                resolved.append(track)
            else:
                unmatched.append((title, artist))
        return resolved, unmatched

    # ------------------------------------------------------------------- ranking
    def recommend(
        self,
        seeds: Sequence[Seed],
        n: int = 20,
        per_artist: int = 25,
        related_per_seed: int = 6,
    ) -> Tuple[List[AudioRecommendation], List[Seed]]:
        """Return (recommendations, unmatched_seeds)."""
        seed_tracks, unmatched = self._resolve_seeds(seeds)
        if not seed_tracks:
            raise LookupError("Could not resolve any seed song to a track with a preview.")

        self.progress(f"Resolved {len(seed_tracks)} seed track(s). Gathering candidates...")
        pool = self.client.gather_candidates(seed_tracks, per_artist, related_per_seed)
        seed_ids = {t.id for t in seed_tracks}
        for t in seed_tracks:
            pool.setdefault(t.id, t)
        self.progress(f"Analyzing {len(pool)} tracks (cached: {len(self.store)})...")

        with TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            analyzed: List[Tuple[DeezerTrack, AcousticFeatures]] = []
            for track in pool.values():
                feats = self._analyze_track(track, workdir)
                if feats is not None:
                    analyzed.append((track, feats))
        self.store.save()

        if len(analyzed) < 2:
            raise RuntimeError("Not enough analyzable tracks to compare.")

        tracks = [t for t, _ in analyzed]
        matrix = np.vstack([f.vector() for _, f in analyzed])
        scaled = StandardScaler().fit_transform(matrix)
        scaled *= np.sqrt(np.clip(_weight_vector(self.weights), 0.0, None))

        seed_rows = [i for i, t in enumerate(tracks) if t.id in seed_ids]
        centroid = scaled[seed_rows].mean(axis=0)

        distances = np.linalg.norm(scaled - centroid, axis=1)
        scores = 1.0 / (1.0 + distances)

        order = np.argsort(scores)[::-1]
        results: List[AudioRecommendation] = []
        seen_titles: set[str] = set()
        for idx in order:
            track = tracks[idx]
            if track.id in seed_ids:
                continue
            title_key = f"{track.title.casefold()}::{track.artist.casefold()}"
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)
            results.append(
                AudioRecommendation(
                    title=track.title,
                    artist=track.artist,
                    score=float(scores[idx]),
                    track_id=track.id,
                )
            )
            if len(results) >= n:
                break
        return results, unmatched
