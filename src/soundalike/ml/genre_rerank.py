"""Approach 2 — Artist-centroid genre-coherence reranker.

Problem: the neural embedding's whitened cosine similarity captures timbre
and texture well, but at 272k songs the top-50 candidates for a given seed
can span multiple unrelated genres.  A song by "The Weeknd" ends up near
rock ballads and reggae songs because they share some spectral property, not
because they're in the same scene.

Solution: build a **per-artist centroid** in the whitened embedding space.
Songs from the same artist cluster together, so the centroid captures the
"scene" of that artist.  The cross-artist cosine similarity between the seed
artist's centroid and each candidate artist's centroid is a direct measure of
*inter-artist genre coherence* — exactly what the `cross_artist_agreement`
metric in benchmark.py rewards.

We add this as a third term to the blend:

    final = (1 - γ) * blend + γ * genre_coherence

where ``blend`` is the existing (alpha * neural_z + (1-alpha) * vibe_z) and
``genre_coherence`` is the centroid-cosine between the seed artist and each
candidate artist (re-normalized to [0,1]).

Key properties:
  * **Numpy-only** — no PyTorch, no re-training.
  * **Scale-free** — works for any library size; centroids are O(n_artists×d).
  * **Complements the existing blend** — doesn't replace neural or vibe signal,
    just adds a scene-level prior.
  * **Graceful degradation** — if a candidate's artist has only 1 song in the
    library (no reliable centroid), the centroid cosine falls back to that
    song's own whitened embedding, so γ effectively does nothing for it.

The centroid matrix is computed once at construction time and costs O(n×d)
where n = library size, d = embedding dim — fast even at 272k tracks.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


class ArtistCentroidIndex:
    """Precomputed per-artist L2-normalized centroids in whitened embedding space.

    Parameters
    ----------
    neural_w : np.ndarray  shape (N, d), already whitened + L2-normalized
    artists : sequence of str  length N
    min_songs : int
        Minimum tracks an artist must have to get a reliable centroid.
        Artists with fewer tracks are mapped to their single-song embedding.
    """

    def __init__(
        self,
        neural_w: np.ndarray,
        artists,
        min_songs: int = 2,
    ):
        n = len(neural_w)
        assert n == len(artists), "neural_w and artists must have the same length"

        artists_cf = np.array([str(a).casefold() for a in artists])

        # Group song indices by casefolded artist name
        by_artist: Dict[str, List[int]] = {}
        for i, a in enumerate(artists_cf):
            by_artist.setdefault(a, []).append(i)

        # Build one compact centroid matrix plus an int32 song→centroid map.
        # The old implementation materialised a full (N,d) per-song centroid
        # matrix, duplicating ~419 MB on the 272,853 × 384 production index.
        centroid_names: List[str] = []
        centroid_values: List[np.ndarray] = []
        for a, rows in by_artist.items():
            if len(rows) >= min_songs:
                c = neural_w[rows].mean(axis=0)
                norm = np.linalg.norm(c)
                if norm > 1e-9:
                    c = c / norm
                centroid_names.append(a)
                centroid_values.append(c.astype(np.float32))

        self._centroid_position = {
            name: position for position, name in enumerate(centroid_names)
        }
        self._centroid_matrix = np.asarray(centroid_values, dtype=np.float32)
        if not centroid_values:
            self._centroid_matrix = np.empty(
                (0, neural_w.shape[1]), dtype=np.float32
            )
        self._song_centroid = np.asarray(
            [self._centroid_position.get(a, -1) for a in artists_cf],
            dtype=np.int32,
        )
        # Reference, not a copy: singleton artists fall back to their own
        # whitened embedding without another N×d allocation.
        self._neural_w = neural_w

        self.n_centroids = len(self._centroid_matrix)

    def seed_artist_centroid(self, artist: str) -> Optional[np.ndarray]:
        """Return the centroid for ``artist`` (casefolded lookup)."""
        position = self._centroid_position.get(str(artist).casefold())
        return None if position is None else self._centroid_matrix[position]

    def genre_similarity(
        self, seed_artist: str, seed_neural_w: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Cosine similarity between seed artist centroid and all library songs.

        Each song maps to its artist's centroid (or its own embedding if the
        artist appears fewer than ``min_songs`` times).

        Parameters
        ----------
        seed_artist : str
        seed_neural_w : np.ndarray (d,), optional
            If the seed artist is not in the library (new/live seed), fall back
            to the seed's own whitened embedding as the query centroid.

        Returns
        -------
        np.ndarray (N,)  values in [-1, 1]
        """
        position = self._centroid_position.get(str(seed_artist).casefold())
        qc = None if position is None else self._centroid_matrix[position]
        if qc is None:
            if seed_neural_w is not None:
                qc = seed_neural_w.astype(np.float32)
            else:
                # Unknown artist, no fallback → return zeros (neutral)
                return np.zeros(len(self._song_centroid), dtype=np.float32)
        centroid_scores = self._centroid_matrix @ qc
        out = np.empty(len(self._song_centroid), dtype=np.float32)
        mapped = self._song_centroid >= 0
        out[mapped] = centroid_scores[self._song_centroid[mapped]]
        if (~mapped).any():
            out[~mapped] = self._neural_w[~mapped] @ qc
        return out

    def blend_with_genre(
        self,
        blended: np.ndarray,
        seed_artist: str,
        seed_neural_w: Optional[np.ndarray] = None,
        gamma: float = 0.25,
    ) -> np.ndarray:
        """Add the genre-coherence term to an existing blend score.

        Parameters
        ----------
        blended : np.ndarray (N,)
            Current blend score (alpha*neural_z + (1-alpha)*vibe_z), already
            z-scored or on a comparable scale.
        seed_artist : str
        seed_neural_w : np.ndarray (d,), optional
            Whitened seed embedding for artists not in the library.
        gamma : float
            Weight of the genre-coherence term (0 = no effect, 1 = only genre).
            Tuned to 0.25 as default — adds a gentle scene bias without
            overriding fine-grained neural similarity.

        Returns
        -------
        np.ndarray (N,)  final scores (NOT z-scored; callers re-sort)
        """
        genre_sim = self.genre_similarity(seed_artist, seed_neural_w)
        # Normalize genre_sim to [0,1] for a fair weighted combination
        gs_min, gs_max = genre_sim.min(), genre_sim.max()
        genre_norm = (genre_sim - gs_min) / (gs_max - gs_min + 1e-9)
        # blend is already z-scored; rescale to [0,1] too
        bl_min, bl_max = blended.min(), blended.max()
        blend_norm = (blended - bl_min) / (bl_max - bl_min + 1e-9)
        return ((1.0 - gamma) * blend_norm + gamma * genre_norm).astype(np.float32)


def build_centroid_index(
    neural_w: np.ndarray,
    artists,
    min_songs: int = 2,
) -> ArtistCentroidIndex:
    """Convenience constructor — same as ArtistCentroidIndex(...)."""
    return ArtistCentroidIndex(neural_w, artists, min_songs=min_songs)
