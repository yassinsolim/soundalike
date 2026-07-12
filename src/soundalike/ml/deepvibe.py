"""Deep-vibe: fuse the learned neural embedding with hand-crafted vibe features.

Neither signal alone is "vibe":
  * the neural encoder (trained on 106k songs) captures **timbre and texture**
    deeply, but is partly blind to energy/dynamics and recommends by overall
    sonic character;
  * the hand-crafted vibe vector captures **bass profile and dynamics (the
    drops)** explicitly, but has no learned notion of texture.

This module stores BOTH for a library of real songs and ranks by a blend of the
two similarity scores, so a recommendation has to match on texture *and* on
energy/low-end. The blend is tunable (`alpha`): 1.0 = pure neural, 0.0 = pure
vibe, 0.5 = balanced.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

import numpy as np

from ..config import cache_dir
from ..audio.vibe import DEFAULT_WEIGHTS, FEATURE_NAMES, VibeFeatures, weight_vector


@dataclass
class DeepVibeRecommendation:
    title: str
    artist: str
    score: float
    track_id: int
    neural_sim: float
    vibe_sim: float

    def __str__(self) -> str:
        return f"{self.title} — {self.artist}  ({self.score:.3f})"


class DeepVibeIndex:
    """Parallel arrays of neural embeddings + vibe vectors for a library."""

    def __init__(
        self, track_ids, titles, artists, neural, vibe, sonic=None,
        clap=None, wiki=None, wiki_specific=None,
    ):
        self.track_ids = np.asarray(track_ids)
        self.titles = np.asarray(titles, dtype=object)
        self.artists = np.asarray(artists, dtype=object)
        self.neural = np.asarray(neural, dtype=np.float32)      # (N, d)
        self.vibe = np.asarray(vibe, dtype=np.float32)          # (N, 29)
        self.sonic = (
            None if sonic is None else np.asarray(sonic, dtype=np.float16)
        )                                                       # optional (N, 64)
        self.clap = None if clap is None else np.asarray(clap, dtype=np.float16)
        self.wiki = None if wiki is None else np.asarray(wiki, dtype=np.float16)
        self.wiki_specific = (
            None if wiki_specific is None
            else np.asarray(wiki_specific, dtype=np.uint8)
        )
        for name, values, ndim in (
            ("sonic", self.sonic, 2),
            ("clap", self.clap, 2),
            ("wiki", self.wiki, 1),
            ("wiki_specific", self.wiki_specific, 1),
        ):
            if values is not None and (
                values.ndim != ndim or len(values) != len(self.track_ids)
            ):
                raise ValueError(
                    f"{name} must be a {ndim}D array aligned with the index rows"
                )

    def __len__(self) -> int:
        return len(self.track_ids)

    def save(self, path: Path, half: bool = False) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # For the bundled artifact, storing the neural embeddings as float16 and
        # compressing halves the file with no effect on ranking (cosine on
        # L2-normalized vectors is insensitive to that precision).
        neural = self.neural.astype(np.float16) if half else self.neural
        saver = np.savez_compressed if half else np.savez
        arrays = {
            "track_ids": self.track_ids,
            "titles": self.titles.astype(str),
            "artists": self.artists.astype(str),
            "neural": neural,
            "vibe": self.vibe,
            "feature_names": np.array(FEATURE_NAMES),
        }
        if self.sonic is not None:
            arrays["sonic"] = self.sonic.astype(np.float16, copy=False)
        if self.clap is not None:
            arrays["clap"] = self.clap.astype(np.float16, copy=False)
        if self.wiki is not None:
            arrays["wiki"] = self.wiki.astype(np.float16, copy=False)
        if self.wiki_specific is not None:
            arrays["wiki_specific"] = self.wiki_specific.astype(np.uint8, copy=False)
        saver(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> "DeepVibeIndex":
        d = np.load(Path(path), allow_pickle=False)
        # Neural may be stored float16 (bundled) — upcast for downstream math.
        return cls(
            d["track_ids"], d["titles"], d["artists"],
            d["neural"].astype(np.float32), d["vibe"],
            d["sonic"] if "sonic" in d.files else None,
            d["clap"] if "clap" in d.files else None,
            d["wiki"] if "wiki" in d.files else None,
            d["wiki_specific"] if "wiki_specific" in d.files else None,
        )

    @classmethod
    def bundled_path(cls) -> Optional[Path]:
        try:
            from importlib import resources

            res = resources.files("soundalike").joinpath("data/deepvibe_index.npz")
            with resources.as_file(res) as p:
                if Path(p).exists():
                    return Path(p)
        except (ModuleNotFoundError, FileNotFoundError, AttributeError, TypeError):
            pass
        bundled = Path(__file__).resolve().parents[1] / "data" / "deepvibe_index.npz"
        return bundled if bundled.exists() else None

    @classmethod
    def user_path(cls) -> Path:
        return cache_dir() / "deepvibe_index.npz"

    @classmethod
    def default_path(cls) -> Path:
        user = cls.user_path()
        if user.exists():
            return user
        return cls.bundled_path() or user


class DeepVibeRecommender:
    """Rank a DeepVibeIndex by a tunable blend of neural + vibe similarity.

    Quality enhancements (all enabled by default, ``enhance=True``):

    * **Approach 1 — quality filter** (``quality_filter=True`` in recommend):
      Pre-filters the candidate pool to remove junk derivatives — slowed/reverb
      TikTok edits, nightcore, karaoke, tribute, and seed-title mashups.  These
      should never appear in recommendations.

    * **Approach 2 — guarded artist-centroid reranker** (``genre_rerank=True``):
      Reorders only the first 20 production candidates by artist-centroid
      coherence.  Ranks 21–50 remain frozen, preventing a known-pair regression.

    The old manual related-artist graph was retired because it directly leaked
    benchmark artists into serving and evaluation.

    Pass ``enhance=False`` to get the original unmodified baseline for ablation.
    """

    def __init__(
        self,
        index: DeepVibeIndex,
        alpha: float = 0.8,
        vibe_weights: Optional[Dict[str, float]] = None,
        whiten: bool = True,
        enhance: bool = True,
        acc_cache_dir: Optional[Path] = None,
    ):
        if len(index) < 2:
            raise ValueError("Deep-vibe index is empty — build it first.")
        self.index = index
        self.alpha = float(np.clip(alpha, 0.0, 1.0))

        # Neural: L2-normalize so a dot product is cosine similarity.
        neural = index.neural / (np.linalg.norm(index.neural, axis=1, keepdims=True) + 1e-9)

        # The learned embeddings pile into a tight cone (every pair ~0.9 cosine),
        # so at a large library size raw cosine can't rank finely and surfaces
        # cross-genre false matches. ZCA-whitening removes the dominant shared
        # direction and equalizes the variance of each dimension, so similarity
        # keys on what's *distinctive* about a track (its scene/vibe) — which
        # makes retrieval dramatically more coherent on a big, diverse library.
        self._whiten = whiten
        if whiten:
            self._nmean = neural.mean(axis=0)
            centered = neural - self._nmean
            cov = np.cov(centered.T)
            evals, evecs = np.linalg.eigh(cov)
            self._W = evecs @ np.diag(1.0 / np.sqrt(np.clip(evals, 1e-5, None))) @ evecs.T
            self._neural = self._apply_whiten(neural)
        else:
            self._nmean = np.zeros(neural.shape[1], np.float32)
            self._W = None
            self._neural = neural

        # Vibe: standardize across the library, then sqrt-weight.
        self._vmean = index.vibe.mean(axis=0)
        self._vstd = index.vibe.std(axis=0) + 1e-9
        w = np.sqrt(np.clip(weight_vector(vibe_weights or DEFAULT_WEIGHTS), 0.0, None))
        self._vscaled = ((index.vibe - self._vmean) / self._vstd) * w
        self._w = w
        # Keep the compact float16 matrix resident. Cosine normalization is done
        # in float32 chunks at query time, matching the measured PCA64 method
        # without retaining a second full matrix.
        self._sonic = index.sonic
        self._clap = index.clap
        self._wiki = (
            None if index.wiki is None
            else self._zscore(np.asarray(index.wiki, dtype=np.float32))
        )
        self._wiki_specific = (
            None if index.wiki_specific is None
            else self._zscore(np.asarray(index.wiki_specific, dtype=np.float32))
        )
        self.last_retrieval_mode = "legacy_no_sonic_seed"

        # ── Enhancement modules ──────────────────────────────────────────────
        self._qfilter = None
        self._qmask: Optional[np.ndarray] = None
        self._centroid_idx = None
        self._related_graph = None  # retired; kept as a compatibility sentinel

        if enhance:
            self._load_enhancements(acc_cache_dir)

    def _load_enhancements(self, acc_cache_dir: Optional[Path]) -> None:
        """Lazily build the two validated quality-improvement modules.

        Designed to degrade gracefully: each module is skipped silently if its
        import fails (e.g., running without the soundalike package installed as
        an editable install, or with a stripped deployment).
        """
        try:
            from .quality_filter import TitleQualityFilter
            self._qfilter = TitleQualityFilter()
            # Pre-compute mask once (fast; avoids per-call regex over 87k+ rows)
            self._qmask = self._qfilter.keep_mask(
                list(self.index.titles), list(self.index.artists))
        except Exception:
            pass

        try:
            from .genre_rerank import ArtistCentroidIndex
            self._centroid_idx = ArtistCentroidIndex(
                self._neural, self.index.artists, min_songs=2)
        except Exception:
            pass

    def _apply_whiten(self, vecs: np.ndarray) -> np.ndarray:
        """Center + ZCA-whiten + re-normalize (rows) of one or many embeddings."""
        x = (vecs - self._nmean) @ self._W
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

    def _project_vibe(self, feats: VibeFeatures) -> np.ndarray:
        return ((feats.vector() - self._vmean) / self._vstd) * self._w

    @staticmethod
    def _zscore(x: np.ndarray) -> np.ndarray:
        return (x - x.mean()) / (x.std() + 1e-9)

    def _sonic_cosine(self, query: np.ndarray) -> np.ndarray:
        query = np.asarray(query, dtype=np.float32)
        if self._sonic is None or query.shape != (self._sonic.shape[1],):
            raise ValueError("seed_sonic dimension does not match the sonic index")
        query = query / max(float(np.linalg.norm(query)), 1e-9)
        scores = np.empty(len(self.index), dtype=np.float32)
        for start in range(0, len(scores), 16384):
            stop = min(start + 16384, len(scores))
            block = np.asarray(self._sonic[start:stop], dtype=np.float32)
            block /= np.maximum(
                np.linalg.norm(block, axis=1, keepdims=True), 1e-9
            )
            scores[start:stop] = block @ query
        return scores

    @staticmethod
    def _compact_cosine(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
        """Chunked cosine against a compact float16 representation matrix."""
        query = np.asarray(query, dtype=np.float32)
        if query.shape != (matrix.shape[1],):
            raise ValueError("query dimension does not match compact index")
        query /= max(float(np.linalg.norm(query)), 1e-9)
        scores = np.empty(len(matrix), dtype=np.float32)
        for start in range(0, len(scores), 16384):
            stop = min(start + 16384, len(scores))
            block = np.asarray(matrix[start:stop], dtype=np.float32)
            block /= np.maximum(
                np.linalg.norm(block, axis=1, keepdims=True), 1e-9
            )
            scores[start:stop] = block @ query
        return scores

    def recommend(
        self,
        seed_neural: np.ndarray,
        seed_vibe: VibeFeatures,
        n: int = 15,
        exclude_ids: Optional[Set] = None,
        exclude_artist: Optional[str] = None,
        seed_title: Optional[str] = None,
        diversity: float = 0.0,
        max_per_artist: int = 0,
        # Enhancement flags (all True = best validated method)
        quality_filter: bool = True,
        genre_rerank: bool = True,
        related_boost: bool = False,
        genre_gamma: float = 0.25,
        related_gamma: float = 0.20,
        seed_sonic: Optional[np.ndarray] = None,
        seed_clap: Optional[np.ndarray] = None,
        seed_row: Optional[int] = None,
    ) -> List[DeepVibeRecommendation]:
        """Recommend songs similar to (seed_neural, seed_vibe).

        Three complementary quality improvements are applied when the enhancement
        modules are loaded (``enhance=True`` at construction):

        * **quality_filter**: removes junk derivatives from the candidate pool
          before ranking (approach 1 — operates on library track titles/artists).
        * **genre_rerank**: adds artist-centroid coherence to the blend so the
          same scene as the seed is boosted (approach 2 — acoustic signal).
        * **related_boost**: deprecated compatibility flag; the leaking manual
          graph is not loaded by serving.
        """
        sonic = self._sonic
        use_dual = (
            sonic is not None
            and self._clap is not None
            and self._wiki is not None
            and self._wiki_specific is not None
            and (seed_row is not None or (seed_sonic is not None and seed_clap is not None))
        )
        if use_dual:
            if seed_row is not None:
                if not 0 <= int(seed_row) < len(self.index):
                    raise ValueError("seed_row is outside the dual-sonic index")
                seed_sonic = np.asarray(sonic[int(seed_row)], dtype=np.float32)
                seed_clap = np.asarray(self._clap[int(seed_row)], dtype=np.float32)
            if seed_sonic is None or seed_clap is None:
                raise ValueError("dual-sonic retrieval requires both seed vectors")

            # Guardrail union: the already judged guarded top five stays visible,
            # then quality-filtered legacy positions preserve every frozen
            # baseline top-10 hit.  The independent learned tail can add exact
            # songs without an artist cap after that safety boundary.
            guarded = self.recommend(
                seed_neural, seed_vibe, n=5, exclude_ids=exclude_ids,
                exclude_artist=exclude_artist, seed_title=seed_title,
                diversity=diversity, max_per_artist=max_per_artist,
                quality_filter=True, genre_rerank=True,
            )
            baseline_guardrail = self.recommend(
                seed_neural, seed_vibe, n=10, exclude_ids=exclude_ids,
                exclude_artist=exclude_artist, seed_title=seed_title,
                diversity=diversity, max_per_artist=max_per_artist,
                quality_filter=True, genre_rerank=False,
            )
            tail = self._recommend_dual_tail(
                np.asarray(seed_sonic), np.asarray(seed_clap), max(n, 50),
                exclude_ids or set(), exclude_artist, seed_title,
            )
            merged: List[DeepVibeRecommendation] = []
            used_ids: Set[int] = set()
            for pool in (guarded[:5], baseline_guardrail[:10], tail):
                for item in pool:
                    if len(merged) >= n:
                        break
                    if item.track_id in used_ids:
                        continue
                    merged.append(item)
                    used_ids.add(item.track_id)
            self.last_retrieval_mode = "dual_sonic64_guardrail"
            return merged[:n]

        use_sonic = sonic is not None and (
            seed_sonic is not None or seed_row is not None
        )
        if use_sonic:
            if sonic is None:
                raise ValueError("Sonic retrieval requires a sonic index")
            if seed_sonic is None:
                if seed_row is None or not 0 <= int(seed_row) < len(self.index):
                    raise ValueError("seed_row is outside the sonic index")
                seed_sonic = np.asarray(sonic[int(seed_row)], dtype=np.float32)
            # The measured policy freezes the complete current guarded top five,
            # then fills from an independently ranked PCA64 tail.
            legacy = self.recommend(
                seed_neural, seed_vibe, n=5, exclude_ids=exclude_ids,
                exclude_artist=exclude_artist, seed_title=seed_title,
                diversity=diversity, max_per_artist=max_per_artist,
                quality_filter=quality_filter, genre_rerank=genre_rerank,
                related_boost=related_boost, genre_gamma=genre_gamma,
                related_gamma=related_gamma,
            )
            sonic_results = self._recommend_sonic_tail(
                seed_neural, seed_vibe, np.asarray(seed_sonic), max(n, 50),
                exclude_ids or set(), exclude_artist, seed_title,
                quality_filter, genre_rerank, genre_gamma,
            )
            merged = list(legacy[:min(5, n)])
            used_ids = {item.track_id for item in merged}
            used_artists = {item.artist.casefold() for item in merged}
            for item in sonic_results:
                if len(merged) >= n:
                    break
                artist = item.artist.casefold()
                if item.track_id in used_ids or artist in used_artists:
                    continue
                merged.append(item)
                used_ids.add(item.track_id)
                used_artists.add(artist)
            self.last_retrieval_mode = "sonic64_stable_head"
            return merged

        self.last_retrieval_mode = "legacy_no_sonic_seed"
        exclude_ids = exclude_ids or set()
        exclude_artist_key = (exclude_artist or "").casefold()

        qn = seed_neural / (np.linalg.norm(seed_neural) + 1e-9)
        if self._whiten:
            qn = self._apply_whiten(qn)
        neural_sim = self._neural @ qn                                   # cosine, -1..1
        qv = self._project_vibe(seed_vibe)
        vibe_sim = 1.0 / (1.0 + np.linalg.norm(self._vscaled - qv, axis=1))

        # Blend on comparable (z-scored) scales so alpha is meaningful.
        blended = self.alpha * self._zscore(neural_sim) + (1 - self.alpha) * self._zscore(vibe_sim)

        order = np.argsort(blended)[::-1]

        # ── Approach 1: pre-filter junk from candidate pool ──────────────────
        qmask = self._qmask if (quality_filter and self._qmask is not None) else None

        # Build a filtered candidate pool (dedup by title/artist, honour excludes,
        # optionally cap songs per artist so one artist can't dominate).
        cand: List[int] = []
        seen: set = set()
        artist_count: Dict[str, int] = {}
        guarded = genre_rerank and self._centroid_idx is not None
        pool_cap = max(n * 25, 500) if (
            diversity > 0 or max_per_artist or guarded
        ) else n
        for idx in order:
            i = int(idx)
            tid = int(self.index.track_ids[i])
            if tid in exclude_ids:
                continue
            title, artist = str(self.index.titles[i]), str(self.index.artists[i])
            akey = artist.casefold()
            if exclude_artist_key and exclude_artist_key in akey:
                continue
            if qmask is not None and not qmask[i]:
                continue
            if (
                qmask is not None
                and seed_title
                and self._qfilter is not None
                and self._qfilter.seed_title_in_result(seed_title, title)
            ):
                continue
            key = f"{title.casefold()}::{akey}"
            if key in seen:
                continue
            if max_per_artist and artist_count.get(akey, 0) >= max_per_artist:
                continue
            seen.add(key)
            artist_count[akey] = artist_count.get(akey, 0) + 1
            cand.append(i)
            if len(cand) >= pool_cap:
                break

        select_n = max(n, 50) if (genre_rerank and self._centroid_idx is not None) else n
        chosen = (
            self._mmr(cand, blended, select_n, diversity)
            if diversity > 0 else cand[:select_n]
        )

        # Guarded centroid rerank: improve the visible top results while freezing
        # the already-retrieved tail.  This preserved every baseline Recall@50
        # hit in the sourced held-out benchmark.
        if genre_rerank and self._centroid_idx is not None and chosen:
            centroid_score = self._centroid_idx.blend_with_genre(
                blended, exclude_artist or "", seed_neural_w=qn, gamma=genre_gamma)
            boundary = min(20, len(chosen))
            chosen = sorted(
                chosen[:boundary],
                key=lambda i: float(centroid_score[i]),
                reverse=True,
            ) + chosen[boundary:]
        chosen = chosen[:n]

        results: List[DeepVibeRecommendation] = []
        for i in chosen:
            results.append(DeepVibeRecommendation(
                title=str(self.index.titles[i]), artist=str(self.index.artists[i]),
                score=float(blended[i]), track_id=int(self.index.track_ids[i]),
                neural_sim=float(neural_sim[i]), vibe_sim=float(vibe_sim[i]),
            ))
        return results

    def _recommend_dual_tail(
        self,
        seed_sonic: np.ndarray,
        seed_clap: np.ndarray,
        n: int,
        exclude_ids: Set,
        exclude_artist: Optional[str],
        seed_title: Optional[str],
    ) -> List[DeepVibeRecommendation]:
        """Rank the learned dual-Sonic64 tail with source-independent priors."""
        if (
            self._sonic is None
            or self._clap is None
            or self._wiki is None
            or self._wiki_specific is None
        ):
            raise ValueError("Dual-Sonic64 tail requires all release arrays")
        efficientnet = self._compact_cosine(self._sonic, seed_sonic)
        clap = self._compact_cosine(self._clap, seed_clap)
        score = (
            0.25 * self._zscore(efficientnet)
            + 0.75 * self._zscore(clap)
            + 0.20 * self._wiki
            + 0.10 * self._wiki_specific
        )
        excluded_artist = (exclude_artist or "").casefold()
        seen_recordings: Set[str] = set()
        rows: List[int] = []
        for raw in np.argsort(score)[::-1]:
            row = int(raw)
            if int(self.index.track_ids[row]) in exclude_ids:
                continue
            title = str(self.index.titles[row])
            artist = str(self.index.artists[row])
            artist_key = artist.casefold()
            if excluded_artist and excluded_artist in artist_key:
                continue
            if self._qmask is not None and not bool(self._qmask[row]):
                continue
            if (
                seed_title
                and self._qfilter is not None
                and self._qfilter.seed_title_in_result(seed_title, title)
            ):
                continue
            recording = f"{title.casefold()}::{artist_key}"
            if recording in seen_recordings:
                continue
            seen_recordings.add(recording)
            rows.append(row)
            if len(rows) >= n:
                break
        return [
            DeepVibeRecommendation(
                title=str(self.index.titles[row]),
                artist=str(self.index.artists[row]),
                score=float(score[row]),
                track_id=int(self.index.track_ids[row]),
                neural_sim=float(efficientnet[row]),
                vibe_sim=float(clap[row]),
            )
            for row in rows
        ]

    def _recommend_sonic_tail(
        self,
        seed_neural: np.ndarray,
        seed_vibe: VibeFeatures,
        seed_sonic: np.ndarray,
        n: int,
        exclude_ids: Set,
        exclude_artist: Optional[str],
        seed_title: Optional[str],
        quality_filter: bool,
        genre_rerank: bool,
        genre_gamma: float,
    ) -> List[DeepVibeRecommendation]:
        """Fixed PCA64+vibe tail used by the measured stable-head winner."""
        sonic = self._sonic
        if sonic is None:
            raise ValueError("Sonic tail requires a sonic index")
        sonic_sim = self._sonic_cosine(seed_sonic)
        qv = self._project_vibe(seed_vibe)
        vibe_sim = 1.0 / (1.0 + np.linalg.norm(self._vscaled - qv, axis=1))
        blended = (
            self.alpha * self._zscore(sonic_sim)
            + (1 - self.alpha) * self._zscore(vibe_sim)
        )
        qmask = self._qmask if quality_filter else None
        excluded_artist = (exclude_artist or "").casefold()
        candidates: List[int] = []
        seen_recordings: Set[str] = set()
        seen_artists: Set[str] = set()
        for raw in np.argsort(blended)[::-1]:
            row = int(raw)
            if int(self.index.track_ids[row]) in exclude_ids:
                continue
            title = str(self.index.titles[row])
            artist = str(self.index.artists[row])
            artist_key = artist.casefold()
            if excluded_artist and excluded_artist in artist_key:
                continue
            if qmask is not None and not bool(qmask[row]):
                continue
            if (
                qmask is not None and seed_title and self._qfilter is not None
                and self._qfilter.seed_title_in_result(seed_title, title)
            ):
                continue
            recording = f"{title.casefold()}::{artist_key}"
            if recording in seen_recordings or artist_key in seen_artists:
                continue
            seen_recordings.add(recording)
            seen_artists.add(artist_key)
            candidates.append(row)
            if len(candidates) >= 1250:
                break

        # Measured MMR uses the PCA64 geometry, not the production embedding.
        rows = candidates
        chosen: List[int] = []
        if rows:
            vectors = np.asarray(sonic[rows], dtype=np.float32)
            vectors /= np.maximum(
                np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9
            )
            relevance = blended[rows]
            relevance = (
                (relevance - relevance.min())
                / (relevance.max() - relevance.min() + 1e-9)
            )
            positions = [int(np.argmax(relevance))]
            best = vectors @ vectors[positions[0]]
            while len(positions) < min(n, len(rows)):
                values = 0.85 * relevance - 0.15 * best
                values[positions] = -np.inf
                position = int(np.argmax(values))
                positions.append(position)
                best = np.maximum(best, vectors @ vectors[position])
            chosen = [rows[position] for position in positions]

        if genre_rerank and self._centroid_idx is not None and chosen:
            qn = seed_neural / (np.linalg.norm(seed_neural) + 1e-9)
            if self._whiten:
                qn = self._apply_whiten(qn)
            centroid = self._centroid_idx.blend_with_genre(
                blended, exclude_artist or "", qn, gamma=genre_gamma
            )
            boundary = min(20, len(chosen))
            chosen = sorted(
                chosen[:boundary], key=lambda row: float(centroid[row]), reverse=True
            ) + chosen[boundary:]

        return [
            DeepVibeRecommendation(
                title=str(self.index.titles[row]),
                artist=str(self.index.artists[row]),
                score=float(blended[row]),
                track_id=int(self.index.track_ids[row]),
                neural_sim=float(sonic_sim[row]),
                vibe_sim=float(vibe_sim[row]),
            )
            for row in chosen[:n]
        ]

    def _mmr(self, cand: List[int], blended: np.ndarray, n: int, diversity: float) -> List[int]:
        """Maximal Marginal Relevance re-ranking of candidate indices.

        Greedily picks the item maximizing ``(1-d)*relevance - d*max_similarity``
        to what's already chosen, so the list stays relevant but stops returning
        five near-identical songs. Similarity is cosine in the (whitened) neural
        space. ``diversity`` d in (0, 1]: 0 = pure relevance, ~0.3 = a good mix.
        """
        if not cand:
            return []
        d = float(np.clip(diversity, 0.0, 1.0))
        cand = list(cand)
        # Relevance normalized to [0, 1] over the candidate pool for a fair trade.
        rel_raw = blended[cand]
        rel = (rel_raw - rel_raw.min()) / (rel_raw.max() - rel_raw.min() + 1e-9)
        vecs = self._neural[cand]  # unit-norm rows

        chosen_pos = [int(np.argmax(rel))]
        best_sim = vecs @ vecs[chosen_pos[0]]  # running max similarity to chosen
        while len(chosen_pos) < min(n, len(cand)):
            scores = (1 - d) * rel - d * best_sim
            for p in chosen_pos:
                scores[p] = -np.inf
            nxt = int(np.argmax(scores))
            chosen_pos.append(nxt)
            best_sim = np.maximum(best_sim, vecs @ vecs[nxt])
        return [cand[p] for p in chosen_pos]


def build_deepvibe_index(
    model_dir: Path,
    per_genre: int = 150,
    per_artist: int = 12,
    genres: Optional[Dict[int, str]] = None,
    existing: Optional[DeepVibeIndex] = None,
    progress: Callable[[str], None] = print,
) -> DeepVibeIndex:
    """Harvest real songs and compute BOTH neural embedding and vibe features."""
    from tempfile import TemporaryDirectory

    from ..audio.previews import DeezerClient
    from ..audio.vibe import vibe_from_file
    from ..audio.vibe_index import HARVEST_GENRES
    from .encoder_infer import EncoderExtractor
    from .spectrogram import _fit_frames, load_audio, log_mel_full, SpectrogramConfig

    client = DeezerClient()
    extractor = EncoderExtractor(model_dir)
    cfg = SpectrogramConfig()
    genres = genres or HARVEST_GENRES

    have_ids: set = set()
    ids, titles, artists, neural, vibe = [], [], [], [], []
    if existing is not None:
        have_ids = set(int(t) for t in existing.track_ids)
        ids = list(existing.track_ids); titles = list(existing.titles)
        artists = list(existing.artists); neural = list(existing.neural); vibe = list(existing.vibe)

    # Gather candidates.
    candidates = {}
    for genre_id, label in genres.items():
        got = 0
        try:
            data = client._get(f"/genre/{genre_id}/artists", {"limit": max(30, per_genre // 3)})
        except Exception as exc:  # noqa: BLE001
            progress(f"[{label}] artist list failed: {exc}"); continue
        for a in data.get("data", []):
            if got >= per_genre:
                break
            for t in client.artist_top_tracks(int(a["id"]), per_artist):
                if t.has_preview and t.id not in candidates and int(t.id) not in have_ids:
                    candidates[t.id] = t
                    got += 1
                    if got >= per_genre:
                        break
        progress(f"[{label}] {got} new candidates")

    progress(f"Embedding {len(candidates)} tracks (neural + vibe)...")
    import time
    t0 = time.time()
    with TemporaryDirectory() as tmp:
        wd = Path(tmp)
        for i, track in enumerate(candidates.values(), 1):
            try:
                dest = wd / f"{track.id}.mp3"
                if client.download_preview(track, dest) is None:
                    continue
                y = load_audio(dest, cfg.sample_rate)
                spec = _fit_frames(log_mel_full(y, cfg), cfg.target_frames)
                nvec = extractor.embed_spec(spec)
                vfeat = vibe_from_file(str(dest))
                dest.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                continue
            ids.append(int(track.id)); titles.append(track.title); artists.append(track.artist)
            neural.append(nvec); vibe.append(vfeat.vector())
            if i % 100 == 0:
                progress(f"  {i}/{len(candidates)} ({i/(time.time()-t0):.1f}/s)")

    progress(f"Deep-vibe index size: {len(ids)} tracks")
    return DeepVibeIndex(ids, titles, artists, np.array(neural, np.float32),
                         np.array(vibe, np.float32))


def build_from_vibe_index(
    model_dir: Path,
    vibe_index_path: Path,
    progress: Callable[[str], None] = print,
) -> DeepVibeIndex:
    """Reuse an existing vibe library's songs and add neural embeddings.

    Keeps the same curated track set (and its vibe features) and re-fetches each
    preview only to compute the neural embedding, so the two signals cover an
    identical, already-curated song set.
    """
    import time
    from tempfile import TemporaryDirectory

    import requests

    from ..audio.previews import DeezerClient
    from ..audio.vibe_index import VibeIndex
    from .encoder_infer import EncoderExtractor
    from .spectrogram import SpectrogramConfig, _fit_frames, load_audio, log_mel_full

    vindex = VibeIndex.load(vibe_index_path)
    extractor = EncoderExtractor(model_dir)
    cfg = SpectrogramConfig()
    client = DeezerClient()
    session = requests.Session()

    ids, titles, artists, neural, vibe = [], [], [], [], []
    total = len(vindex.entries)
    progress(f"Adding neural embeddings to {total} vibe-library tracks...")
    t0 = time.time()
    with TemporaryDirectory() as tmp:
        wd = Path(tmp)
        for i, e in enumerate(vindex.entries, 1):
            try:
                meta = client._get(f"/track/{e.track_id}")
                preview = meta.get("preview")
                if not preview:
                    continue
                dest = wd / f"{e.track_id}.mp3"
                dest.write_bytes(session.get(preview, timeout=30).content)
                y = load_audio(dest, cfg.sample_rate)
                spec = _fit_frames(log_mel_full(y, cfg), cfg.target_frames)
                nvec = extractor.embed_spec(spec)
                dest.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                continue
            ids.append(int(e.track_id)); titles.append(e.title); artists.append(e.artist)
            neural.append(nvec); vibe.append(e.features.vector())
            if i % 100 == 0:
                progress(f"  {i}/{total} ({i/(time.time()-t0):.1f}/s, kept {len(ids)})")

    progress(f"Deep-vibe index size: {len(ids)} tracks")
    return DeepVibeIndex(ids, titles, artists, np.array(neural, np.float32),
                         np.array(vibe, np.float32))
