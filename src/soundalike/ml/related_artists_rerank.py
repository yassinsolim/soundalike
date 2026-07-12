"""Approach 3 — Related-artist collaborative graph reranker.

Problem: artist-centroid genre coherence (genre_rerank.py) only uses what the
*embeddings* know about artist similarity — but the embedding can confuse heavy
guitar timbre across metal/shoegaze, or certain jazz flavours with soul, simply
because those genres share spectral properties.

Solution: augment the acoustic signal with a small **artist-relationship graph**
built from editorial / co-listening data (Deezer related-artist API, Last.fm
similar-artists, or manually-curated artist bundles). When the seed artist has
known related artists in the graph, candidates whose artists appear in that set
get a direct score boost.

This is materially different from Approaches 1 & 2:
  * Approach 1 (quality_filter): *removes* obvious junk derivatives from the
    candidate pool — it doesn't change ranking among genuine tracks.
  * Approach 2 (genre_rerank): boosts candidates whose *embedding centroid* is
    close to the seed artist's centroid — purely acoustic signal.
  * Approach 3 (this module): boosts candidates based on *editorial / social*
    artist similarity — an orthogonal collaborative signal.

Key properties:
  * Numpy-only: no PyTorch, no re-training.
  * Graceful degradation: if the seed artist is unknown to the graph, the
    score contribution is 0 and the existing blend is unchanged.
  * Scale-free: graph query time is O(1) per candidate (set membership).
  * Compatible with the hosted Vercel path (numpy-only backend).

Graph construction:
  The class can read explicit cache artifacts for offline experiments. Static
  ``MANUAL_PAIRS`` are intentionally empty because the previous list contained
  benchmark artists and invalidated evaluation. Serving no longer loads this
  graph; any future use must pass the artist-disjoint leakage audit.

The graph is bidirectional: if A is related to B, then B is also related to A.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

# ---------------------------------------------------------------------------
# Static supplementary pairs (fill gaps not covered by acc_cache)
# ---------------------------------------------------------------------------

# The former list mixed hand-curated evaluation artists into the serving graph.
# That was direct benchmark leakage, so static pairs are deliberately retired.
# A future graph may be loaded only from a versioned development-only artifact
# whose artist-disjointness is checked by real_benchmark.audit_leakage().
MANUAL_PAIRS: List[tuple] = []


def _nkfd_casefold(s: str) -> str:
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().casefold()


def _load_acc_cache(acc_cache_dir: Optional[Path]) -> Dict[str, List[str]]:
    """Load the pre-cached Deezer related-artist data from the acc_cache directory."""
    if acc_cache_dir is None or not acc_cache_dir.exists():
        return {}
    result: Dict[str, List[str]] = {}
    for f in acc_cache_dir.glob("dz_*.json"):
        try:
            data = json.loads(f.read_bytes().decode("utf-8", errors="replace"))
            names = data.get("names", [])
            if names:
                result[f.stem] = [str(n) for n in names]
        except Exception:
            pass
    return result


class RelatedArtistGraph:
    """Bidirectional artist-relationship graph for score boosting.

    Parameters
    ----------
    acc_cache_dir : Path, optional
        Directory containing ``dz_*.json`` Deezer related-artist cache files.
    use_manual : bool
        Whether to include the hard-coded ``MANUAL_PAIRS`` (default True).
    boost : float
        Score bonus applied to candidates whose artist is related to the seed.
        Applied multiplicatively on the normalised blend score.
    """

    def __init__(
        self,
        acc_cache_dir: Optional[Path] = None,
        use_manual: bool = True,
        boost: float = 0.15,
    ):
        self._graph: Dict[str, Set[str]] = {}

        # Load acc_cache data
        for stem, names in _load_acc_cache(acc_cache_dir).items():
            # stem is like "dz_kendrick_lamar"
            a = _nkfd_casefold(stem[3:].replace("_", " "))  # strip "dz_"
            for rel in names:
                b = _nkfd_casefold(rel)
                if b:
                    self._add_edge(a, b)

        # Add manual pairs (bidirectional by construction)
        if use_manual:
            for a_raw, b_raw in MANUAL_PAIRS:
                a = _nkfd_casefold(a_raw)
                b = _nkfd_casefold(b_raw)
                self._add_edge(a, b)
                self._add_edge(b, a)

        self.boost = float(boost)
        self.n_artists = len(self._graph)
        self.n_edges = sum(len(v) for v in self._graph.values()) // 2

    def _add_edge(self, a: str, b: str) -> None:
        self._graph.setdefault(a, set()).add(b)
        self._graph.setdefault(b, set()).add(a)

    def related_set(self, artist: str) -> Set[str]:
        """Return the set of casefolded artists related to ``artist``."""
        key = _nkfd_casefold(artist).split(",")[0].strip()
        return self._graph.get(key, set())

    def score_boost(
        self,
        blended: np.ndarray,
        artists: np.ndarray,
        seed_artist: str,
    ) -> np.ndarray:
        """Return a new score array with collaborative boosts applied.

        The boosted score for a candidate whose artist is in the related set
        is: score * (1 + boost). Candidates outside the related set are
        unchanged.

        Parameters
        ----------
        blended : np.ndarray (N,)
            Current blend scores (already normalised to [0, 1] or z-scored).
        artists : np.ndarray (N,) of str
            Library artist names.
        seed_artist : str
            The seed song's artist.

        Returns
        -------
        np.ndarray (N,) float32  — same scale as ``blended``, with boosts applied.
        """
        related = self.related_set(seed_artist)
        if not related:
            return blended  # unknown seed artist: no change

        boosted = np.array(blended, dtype=np.float32)
        for i, a in enumerate(artists):
            akey = _nkfd_casefold(str(a)).split(",")[0].strip()
            if akey in related:
                boosted[i] *= (1.0 + self.boost)
        return boosted

    def build_boost_vector(
        self,
        artists: np.ndarray,
        seed_artist: str,
    ) -> np.ndarray:
        """Boolean/float mask: 1+boost for related artists, 1.0 otherwise.

        This is cheaper than score_boost when you want to cache the mask across
        multiple blend scenarios.
        """
        related = self.related_set(seed_artist)
        if not related:
            return np.ones(len(artists), dtype=np.float32)
        mask = np.ones(len(artists), dtype=np.float32)
        for i, a in enumerate(artists):
            akey = _nkfd_casefold(str(a)).split(",")[0].strip()
            if akey in related:
                mask[i] = 1.0 + self.boost
        return mask

    def blend_with_related(
        self,
        blended: np.ndarray,
        artists: np.ndarray,
        seed_artist: str,
        gamma: float = 0.20,
    ) -> np.ndarray:
        """Add the related-artist term to an existing blend score.

        Final score = (1 - gamma) * blend_norm + gamma * related_norm

        where ``related_norm`` is a binary mask (1 for related artists, 0 for
        others) normalised to [0, 1].

        Parameters
        ----------
        blended : np.ndarray (N,)  current blend score (any scale)
        artists : np.ndarray (N,)  library artist names
        seed_artist : str
        gamma : float  weight of the collaborative term (0 = acoustic only)

        Returns
        -------
        np.ndarray (N,) float32
        """
        related = self.related_set(seed_artist)

        # Normalise blend to [0, 1]
        bl_min, bl_max = blended.min(), blended.max()
        blend_norm = (blended - bl_min) / (bl_max - bl_min + 1e-9)

        if not related:
            return blend_norm.astype(np.float32)  # unknown seed: passthrough

        # Build binary related-mask
        rel_mask = np.zeros(len(artists), dtype=np.float32)
        for i, a in enumerate(artists):
            akey = _nkfd_casefold(str(a)).split(",")[0].strip()
            if akey in related:
                rel_mask[i] = 1.0

        return ((1.0 - gamma) * blend_norm + gamma * rel_mask).astype(np.float32)


def build_related_graph(
    acc_cache_dir: Optional[Path] = None,
    use_manual: bool = True,
    boost: float = 0.15,
) -> RelatedArtistGraph:
    """Convenience constructor — same as RelatedArtistGraph(...)."""
    return RelatedArtistGraph(
        acc_cache_dir=acc_cache_dir,
        use_manual=use_manual,
        boost=boost,
    )
