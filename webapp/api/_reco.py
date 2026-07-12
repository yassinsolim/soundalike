"""Torch-free numpy recommender for the Vercel deployment.

The full model needs PyTorch (~2.9 GB) to embed an *arbitrary* song, which can't
live in a serverless function. But recommending from a song that's already in the
272,853-track library needs **no torch at all** — every embedding is precomputed,
and ranking is pure numpy (whiten + cosine + vibe-blend + guarded reranking).

This module mirrors `soundalike.ml.deepvibe.DeepVibeRecommender` exactly (a test
asserts identical top-k), so the hosted library-mode results match the desktop
app. The index is fetched once from the public GitHub Release and cached in
``/tmp`` across warm invocations.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
import unicodedata
import urllib.request
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import numpy as np

# Where to get the index. A local path wins (dev); otherwise download the bundled
# pack asset from the public Release into the function's ephemeral /tmp.
_INDEX_URL = os.environ.get(
    "SOUNDALIKE_INDEX_URL",
    "https://github.com/yassinsolim/soundalike/releases/download/"
    "index-2026.07.11-dual-sonic64/deepvibe_index.npz",
)
# Bump this when the index changes so warm instances with an old /tmp copy
# re-download instead of serving stale data.
_INDEX_VERSION = "2026.07.11-dual-sonic64"
_INDEX_SHA256 = os.environ.get(
    "SOUNDALIKE_INDEX_SHA256",
    "f3ed57af1b8073f2872eed1e9192dee04d1089c7266fb98a157d1ea194526fb9",
)
_INDEX_PATH = os.environ.get("SOUNDALIKE_INDEX_PATH", "")

_LOCK = threading.Lock()
_RECO: Optional["WebRecommender"] = None


_PAREN = re.compile(r"[\(\[][^\)\]]*[\)\]]")   # (feat...)/(Remaster)/[Explicit]
_DASH_SUFFIX = re.compile(r"\s+-\s+.*$")        # "- 2011 Remaster" style suffix


def _norm(s: str) -> str:
    """Title/artist match key. Strips accents, parenthetical credits/versions, and
    trailing '- Remaster' suffixes — but KEEPS ordinary words like 'with' (so
    'Stay With Me' stays 'stay with me' instead of collapsing to 'stay')."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = s.casefold()
    s = _PAREN.sub(" ", s)
    s = _DASH_SUFFIX.sub("", s)
    for sep in (" feat. ", " feat ", " ft. ", " ft ", " featuring "):
        i = s.find(sep)
        if i > 0:
            s = s[:i]
    return " ".join(s.split())


def _version_penalty(title: str) -> Tuple[int, int]:
    """Prefer an original-looking catalogue row over remix/live derivatives."""
    derivative = int(bool(re.search(
        r"\b(?:karaoke|tribute|slowed|reverb|nightcore|instrumental|"
        r"remix|cover|live|acoustic)\b",
        str(title),
        re.IGNORECASE,
    )))
    return derivative, len(str(title))


# Vibe feature weights — must match soundalike.audio.vibe.DEFAULT_WEIGHTS exactly
# (verified against weight_vector(DEFAULT_WEIGHTS)). Anything not listed is 1.0.
_DEFAULT_WEIGHTS = {
    "tempo": 1.5, "brightness": 1.5, "onset_rate": 1.5, "rms_std": 2.0,
    "dynamic_range": 2.5, "crest": 2.5, "low_end_ratio": 3.0,
    "band_sub": 2.5, "band_bass": 2.0,
}

_TITLE_JUNK_RE = re.compile(
    r"\b(?:slowed|reverb|sped[- ]up|speed[- ]up|nightcore|karaoke|karaōke|"
    r"backing\s+track|instrumental\s+(?:version|mix|cover|track)|a\s+cappella|"
    r"tribute\s+(?:version|recording)|cover\s+(?:version|record)|"
    r"\(\s*cover(?:\s+of\b[^)]*)?\s*\)|\s+-\s+cover(?:\s+of\b.*)?$|"
    r"originally\s+performed\s+by|in\s+the\s+style\s+of|"
    r"as\s+made\s+famous\s+by|piano\s+version|"
    r"string\s+(?:quartet|version)|orchestral\s+version|remake|"
    r"marimba\s+remix|ringtone|8\s*bit\s+(?:version|cover)|"
    r"\w[\w']*\s+\w[\w' -]*\s+x\s+(?![^()]*\bremix\b)\w[\w']*\s+\w[\w' -]*|"
    r"medley|mashup|"
    r"sing(?:-|\s)?along|lo-?fi\s+(?:version|remix|cover|study))\b",
    re.IGNORECASE,
)
_ARTIST_JUNK_RE = re.compile(
    r"\b(?:karaoke|karaōke|tribute\s+(?:to|band|artists?)|covers?\s+band|"
    r"originally\s+performed\s+by|in\s+the\s+style\s+of|"
    r"instrumental\s+all\s+stars?|marimba\s+remix|nightcore|slowed)\b",
    re.IGNORECASE,
)
_MULTI_X_MASHUP_RE = re.compile(r"\bx\s+\w.*\bx\s+\w", re.IGNORECASE)
_LEADING_TRIBUTE_RE = re.compile(r"^tribute\s+to\b", re.IGNORECASE)
_CONTEXT_TITLE_JUNK_RE = re.compile(
    r"\(\s*cover(?:\s+of\b[^)]*)?\s*\)|\s+-\s+cover(?:\s+of\b.*)?$",
    re.IGNORECASE,
)
_CONTEXT_VERSION_JUNK_RE = re.compile(
    r"(?:\(|\[)[^)\]]*\b(?:re)?mix(?:es)?\b[^)\]]*(?:\)|\])"
    r"|\s+-\s+[^-]*\b(?:re)?mix(?:es)?\b[^-]*$"
    r"|\b(?:club|radio|extended|dub|dance|house|vocal)\s+mix\b"
    r"|(?:\(|\[|\s+-\s*)[^)\]]*\b(?:rework|bootleg|vip|edit)\b"
    r"[^)\]]*(?:\)|\]|$)|\bchopnotslop\b",
    re.IGNORECASE,
)


class _TitleQualityFilter:
    """Self-contained hosted copy; keeps Vercel independent of the torch package."""

    @staticmethod
    def _plain(value: str) -> str:
        return unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()

    def keep_mask(self, titles, artists) -> np.ndarray:
        return np.asarray([
            not (
                _TITLE_JUNK_RE.search(self._plain(title))
                or _CONTEXT_TITLE_JUNK_RE.search(self._plain(title))
                or _CONTEXT_VERSION_JUNK_RE.search(self._plain(title))
                or _MULTI_X_MASHUP_RE.search(self._plain(title))
                or _LEADING_TRIBUTE_RE.search(self._plain(title))
                or _ARTIST_JUNK_RE.search(self._plain(artist))
            )
            for title, artist in zip(titles, artists)
        ], dtype=bool)

    def seed_title_in_result(self, seed_title: str, result_title: str) -> bool:
        def title_key(value: str) -> str:
            value = self._plain(value).casefold()
            value = _PAREN.sub(" ", value)
            value = _DASH_SUFFIX.sub("", value)
            return " ".join(value.split())

        seed = title_key(seed_title)
        result = title_key(result_title)
        if not seed or not result:
            return False
        if seed == result or seed in result or result in seed:
            return True
        return (
            min(len(seed), len(result)) >= 8
            and SequenceMatcher(None, seed, result).ratio() >= 0.90
        )


class _ArtistCentroidIndex:
    """Compact numpy-only centroid index shared by warm hosted requests."""

    def __init__(self, neural: np.ndarray, artists, min_songs: int = 2):
        names = np.asarray([str(artist).casefold() for artist in artists])
        groups: Dict[str, List[int]] = {}
        for row, name in enumerate(names):
            groups.setdefault(name, []).append(row)
        centroid_names, values = [], []
        for name, rows in groups.items():
            if len(rows) < min_songs:
                continue
            centroid = neural[rows].mean(axis=0)
            centroid /= np.linalg.norm(centroid) + 1e-9
            centroid_names.append(name)
            values.append(centroid.astype(np.float32))
        self.positions = {name: i for i, name in enumerate(centroid_names)}
        self.matrix = (
            np.asarray(values, dtype=np.float32)
            if values else np.empty((0, neural.shape[1]), dtype=np.float32)
        )
        self.song_centroid = np.asarray(
            [self.positions.get(name, -1) for name in names], dtype=np.int32
        )
        self.neural = neural
        self.n_centroids = len(self.matrix)

    def blend_with_genre(
        self, blend: np.ndarray, seed_artist: str,
        seed_neural_w: np.ndarray, gamma: float,
    ) -> np.ndarray:
        position = self.positions.get(str(seed_artist).casefold())
        query = self.matrix[position] if position is not None else seed_neural_w
        centroid_scores = self.matrix @ query
        genre = np.empty(len(self.song_centroid), dtype=np.float32)
        mapped = self.song_centroid >= 0
        genre[mapped] = centroid_scores[self.song_centroid[mapped]]
        if (~mapped).any():
            genre[~mapped] = self.neural[~mapped] @ query
        genre = (genre - genre.min()) / (genre.max() - genre.min() + 1e-9)
        normalized = (blend - blend.min()) / (blend.max() - blend.min() + 1e-9)
        return ((1.0 - gamma) * normalized + gamma * genre).astype(np.float32)


class WebRecommender:
    """Loads a DeepVibeIndex .npz and ranks library songs, numpy-only.

    By default, the leakage-free quality filter and guarded centroid reranker
    are enabled (see ``recommend()``).
    Pass ``enhance=False`` to get the original unmodified baseline for ablation.
    """

    def __init__(self, path: str, alpha: float = 0.8, enhance: bool = True,
                 acc_cache_dir: Optional[str] = None):
        d = np.load(path, allow_pickle=False)
        self.track_ids = d["track_ids"]
        self.titles = d["titles"].astype(str)
        self.artists = d["artists"].astype(str)
        self.feature_names = [str(x) for x in d["feature_names"]]
        neural = d["neural"].astype(np.float32)
        vibe = d["vibe"].astype(np.float32)
        self._sonic = d["sonic"] if "sonic" in d.files else None
        self._clap = d["clap"] if "clap" in d.files else None
        self._wiki = (
            self._z(d["wiki"].astype(np.float32))
            if "wiki" in d.files else None
        )
        self._wiki_specific = (
            self._z(d["wiki_specific"].astype(np.float32))
            if "wiki_specific" in d.files else None
        )
        self.alpha = float(alpha)
        self.last_retrieval_mode = "legacy_no_sonic_seed"
        self.index_version = _INDEX_VERSION if self._sonic is not None else "legacy"

        # --- neural: L2-normalize, then ZCA-whiten (chunked + float32 to keep
        # peak memory well under a serverless function's limit on a large index;
        # mathematically the same transform the canonical recommender applies) ---
        neural /= np.linalg.norm(neural, axis=1, keepdims=True) + 1e-9
        self._nmean = neural.mean(axis=0)
        n, dim = neural.shape
        CH = 16384
        # Covariance via chunk accumulation (avoids a full centered copy / np.cov's
        # float64 temporary of the whole matrix).
        cov = np.zeros((dim, dim), dtype=np.float64)
        for i in range(0, n, CH):
            c = (neural[i:i + CH] - self._nmean).astype(np.float64)
            cov += c.T @ c
        cov /= max(n - 1, 1)
        evals, evecs = np.linalg.eigh(cov)
        self._W = (evecs @ np.diag(1.0 / np.sqrt(np.clip(evals, 1e-5, None)))
                   @ evecs.T).astype(np.float32)
        # Whiten in place, chunk by chunk (each row's transform is independent, so
        # overwriting as we go is safe and avoids a second full-size array), then
        # drop the on-disk npz. Only the whitened matrix stays resident.
        for i in range(0, n, CH):
            x = (neural[i:i + CH] - self._nmean) @ self._W
            x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
            neural[i:i + CH] = x
        self._neural = neural
        del d
        import gc
        gc.collect()

        # --- vibe: standardize, sqrt-weight (identical to production) ---
        self._vmean = vibe.mean(axis=0)
        self._vstd = vibe.std(axis=0) + 1e-9
        w = np.array([_DEFAULT_WEIGHTS.get(n, 1.0) for n in self.feature_names], np.float32)
        self._w = np.sqrt(np.clip(w, 0.0, None))
        self._vscaled = ((vibe - self._vmean) / self._vstd) * self._w

        # Fast library-hit lookups + precomputed normalized strings for search.
        self._nt = [_norm(t) for t in self.titles]     # normalized titles
        self._na = [_norm(a) for a in self.artists]     # normalized artists
        self._naprim = [a.split(",")[0].split(" & ")[0].strip() for a in self._na]
        self._by_pair: Dict[Tuple[str, str], int] = {}
        self._by_title: Dict[str, List[int]] = {}
        for i in range(len(self.titles)):
            t, a = self._nt[i], self._naprim[i]
            previous = self._by_pair.get((t, a))
            if previous is None or _version_penalty(self.titles[i]) < _version_penalty(
                self.titles[previous]
            ):
                self._by_pair[(t, a)] = i
            self._by_title.setdefault(t, []).append(i)

        # ── Enhancement modules (loaded only when enhance=True) ──────────────
        self._qfilter = None
        self._centroid_idx = None

        if enhance:
            self._load_enhancements(acc_cache_dir)

    def _load_enhancements(self, acc_cache_dir: Optional[str]) -> None:
        """Lazily load the two validated quality-improvement modules.

        These implementations intentionally live in this self-contained Vercel
        module.  The deployment installs numpy only and cannot import the desktop
        package from outside the ``webapp`` root.
        """
        self._qfilter = _TitleQualityFilter()
        self._qmask = self._qfilter.keep_mask(self.titles, self.artists)
        self._centroid_idx = _ArtistCentroidIndex(
            self._neural, self.artists, min_songs=2
        )

    def _apply_whiten(self, vecs: np.ndarray) -> np.ndarray:
        x = (vecs - self._nmean) @ self._W
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

    def __len__(self) -> int:
        return len(self.titles)

    # -------------------------------------------------------------- search/seed
    def find_row(self, title: str, artist: str = "") -> Optional[int]:
        t = _norm(title)
        a = _norm(artist)
        aprim = a.split(",")[0].split(" & ")[0].strip()
        if aprim and (t, aprim) in self._by_pair:
            return self._by_pair[(t, aprim)]
        if not a and len(self._by_title.get(t, [])) >= 1:
            return min(self._by_title[t], key=lambda row: _version_penalty(self.titles[row]))
        # loose: title substring, optionally constrained to the artist
        best = None
        for i in range(len(self._nt)):
            if t and t in self._nt[i]:
                if not a or a in self._na[i]:
                    # prefer an exact title match over a mere substring
                    if self._nt[i] == t:
                        return i
                    if best is None:
                        best = i
        return best

    def search(self, q: str, limit: int = 8) -> List[Dict]:
        """Ranked search over the library for the seed picker / autocomplete.

        Ranks exact-title matches first, then title prefix, then title substring,
        then artist-substring, then an all-tokens-present fallback (so 'miki stay'
        finds 'Mayonaka no Door / Stay With Me' by Miki Matsubara). Fixes the old
        behaviour where a query like 'Stay With Me' collapsed to 'stay' and matched
        unrelated songs.
        """
        nq = _norm(q)
        if not nq:
            return []
        toks = nq.split()
        scored: List[Tuple[int, int, int]] = []
        for i in range(len(self._nt)):
            nt = self._nt[i]
            if nq in nt:
                s = 0 if nt == nq else (1 if nt.startswith(nq) else 2)
            elif nq in nt + " " + self._na[i]:
                s = 3
            elif len(toks) > 1 and all(tok in (nt + " " + self._na[i]) for tok in toks):
                s = 4
            else:
                continue
            scored.append((s, len(nt), i))
        scored.sort()
        hits, seen = [], set()
        for _, __, i in scored:
            key = (self._nt[i], self._naprim[i])
            if key in seen:
                continue
            seen.add(key)
            hits.append({"row": i, "title": str(self.titles[i]),
                         "artist": str(self.artists[i])})
            if len(hits) >= limit:
                break
        return hits

    # -------------------------------------------------------------- recommend
    @staticmethod
    def _z(x: np.ndarray) -> np.ndarray:
        return (x - x.mean()) / (x.std() + 1e-9)

    def _sonic_cosine(self, query: np.ndarray) -> np.ndarray:
        query = np.asarray(query, dtype=np.float32)
        if self._sonic is None or query.shape != (self._sonic.shape[1],):
            raise ValueError("seed sonic dimension does not match index")
        query /= max(float(np.linalg.norm(query)), 1e-9)
        scores = np.empty(len(self), dtype=np.float32)
        for start in range(0, len(scores), 16384):
            stop = min(start + 16384, len(scores))
            block = np.asarray(self._sonic[start:stop], dtype=np.float32)
            block /= np.maximum(np.linalg.norm(block, axis=1, keepdims=True), 1e-9)
            scores[start:stop] = block @ query
        return scores

    @staticmethod
    def _compact_cosine(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
        query = np.asarray(query, dtype=np.float32)
        if query.shape != (matrix.shape[1],):
            raise ValueError("query dimension does not match compact index")
        query /= max(float(np.linalg.norm(query)), 1e-9)
        scores = np.empty(len(matrix), dtype=np.float32)
        for start in range(0, len(scores), 16384):
            stop = min(start + 16384, len(scores))
            block = np.asarray(matrix[start:stop], dtype=np.float32)
            block /= np.maximum(np.linalg.norm(block, axis=1, keepdims=True), 1e-9)
            scores[start:stop] = block @ query
        return scores

    def recommend(self, row: int, n: int = 20, alpha: Optional[float] = None,
                  diversity: float = 0.15, max_per_artist: int = 1,
                  # Enhancement flags (all True by default = best validated method)
                  quality_filter: bool = True,
                  genre_rerank: bool = True,
                  related_boost: bool = False,
                  genre_gamma: float = 0.25,
                  related_gamma: float = 0.20,
                  ) -> Dict:
        """Rank library songs for a seed row.

        Two validated improvements over the plain neural+vibe blend:

        * **quality_filter** (Approach 1): removes junk derivatives (slowed,
          karaoke, tribute, nightcore) from the candidate pool.
        * **genre_rerank** (Approach 2): guarded artist-centroid reordering of
          positions 1–20 while positions 21–50 remain frozen.

        The validated filter and guarded centroid flags default to True.
        ``related_boost`` is retained only for API compatibility and defaults
        to False because the manual graph was retired for leakage.
        """
        if (
            self._sonic is not None
            and self._clap is not None
            and self._wiki is not None
            and self._wiki_specific is not None
        ):
            guarded = self._recommend_legacy(
                row, 5, alpha, diversity, max_per_artist,
                True, True, genre_gamma,
            )
            baseline = self._recommend_legacy(
                row, 10, alpha, diversity, max_per_artist,
                True, False, genre_gamma,
            )
            tail = self._recommend_dual_tail(row, max(n, 50))
            results: List[Dict] = []
            used_ids = set()
            for pool in (
                guarded["results"][:5],
                baseline["results"][:10],
                tail,
            ):
                for item in pool:
                    if len(results) >= n:
                        break
                    if item["deezer_id"] in used_ids:
                        continue
                    results.append(item)
                    used_ids.add(item["deezer_id"])
            self.last_retrieval_mode = "dual_sonic64_guardrail"
            guarded["results"] = results[:n]
            guarded["retrieval_mode"] = self.last_retrieval_mode
            guarded["method"] = self.last_retrieval_mode
            guarded["index_version"] = self.index_version
            return guarded
        if self._sonic is not None:
            legacy = self._recommend_legacy(
                row, 5, alpha, diversity, max_per_artist,
                quality_filter, genre_rerank, genre_gamma,
            )
            tail = self._recommend_sonic_tail(
                row, max(n, 50), alpha, quality_filter, genre_rerank, genre_gamma
            )
            results = list(legacy["results"][:min(5, n)])
            used_ids = {item["deezer_id"] for item in results}
            used_artists = {item["artist"].casefold() for item in results}
            for item in tail:
                if len(results) >= n:
                    break
                artist = item["artist"].casefold()
                if item["deezer_id"] in used_ids or artist in used_artists:
                    continue
                results.append(item)
                used_ids.add(item["deezer_id"])
                used_artists.add(artist)
            self.last_retrieval_mode = "sonic64_stable_head"
            legacy["results"] = results
            legacy["retrieval_mode"] = self.last_retrieval_mode
            legacy["method"] = self.last_retrieval_mode
            legacy["index_version"] = self.index_version
            return legacy
        result = self._recommend_legacy(
            row, n, alpha, diversity, max_per_artist,
            quality_filter, genre_rerank, genre_gamma,
        )
        self.last_retrieval_mode = "legacy_no_sonic_seed"
        result["retrieval_mode"] = self.last_retrieval_mode
        result["method"] = self.last_retrieval_mode
        result["index_version"] = self.index_version
        return result

    def _recommend_dual_tail(self, row: int, n: int) -> List[Dict]:
        if (
            self._sonic is None
            or self._clap is None
            or self._wiki is None
            or self._wiki_specific is None
        ):
            raise ValueError("Dual-Sonic64 tail requires all release arrays")
        efficientnet = self._compact_cosine(
            self._sonic, np.asarray(self._sonic[row], dtype=np.float32)
        )
        clap = self._compact_cosine(
            self._clap, np.asarray(self._clap[row], dtype=np.float32)
        )
        score = (
            0.25 * self._z(efficientnet)
            + 0.75 * self._z(clap)
            + 0.20 * self._wiki
            + 0.10 * self._wiki_specific
        )
        seed_artist = str(self.artists[row]).casefold()
        seed_title = str(self.titles[row])
        seen_recordings, chosen = set(), []
        qfilter = self._qfilter
        qmask = self._qmask if qfilter is not None else None
        for raw in np.argsort(score)[::-1]:
            candidate = int(raw)
            if candidate == row:
                continue
            title = str(self.titles[candidate])
            artist = str(self.artists[candidate])
            artist_key = artist.casefold()
            if seed_artist and seed_artist in artist_key:
                continue
            if qmask is not None and not bool(qmask[candidate]):
                continue
            if qfilter is not None and qfilter.seed_title_in_result(seed_title, title):
                continue
            recording = (title.casefold(), artist_key)
            if recording in seen_recordings:
                continue
            seen_recordings.add(recording)
            chosen.append(candidate)
            if len(chosen) >= n:
                break
        from urllib.parse import quote
        return [{
            "title": str(self.titles[i]),
            "artist": str(self.artists[i]),
            "deezer_id": int(self.track_ids[i]),
            "neural_sim": round(float(efficientnet[i]), 4),
            "vibe_sim": round(float(clap[i]), 4),
            "spotify_url": (
                "https://open.spotify.com/search/"
                + quote(str(self.titles[i]) + " " + str(self.artists[i]))
            ),
        } for i in chosen]

    def _recommend_legacy(
        self, row: int, n: int, alpha: Optional[float], diversity: float,
        max_per_artist: int, quality_filter: bool, genre_rerank: bool,
        genre_gamma: float,
    ) -> Dict:
        a = self.alpha if alpha is None else float(alpha)
        qn = self._neural[row]
        neural_sim = self._neural @ qn
        qv = self._vscaled[row]
        vibe_sim = 1.0 / (1.0 + np.linalg.norm(self._vscaled - qv, axis=1))
        blended = a * self._z(neural_sim) + (1 - a) * self._z(vibe_sim)

        seed_artist_raw = str(self.artists[row])
        seed_artist = seed_artist_raw.casefold()
        seed_id = int(self.track_ids[row])
        seed_title = str(self.titles[row])

        # Re-normalise to [0,1] so downstream comparisons stay meaningful.
        bl_min, bl_max = blended.min(), blended.max()
        blended = (blended - bl_min) / (bl_max - bl_min + 1e-9)

        order = np.argsort(blended)[::-1]

        if quality_filter and hasattr(self, '_qmask') and self._qmask is not None:
            qmask = self._qmask
        else:
            qmask = None

        cand: List[int] = []
        seen: set = set()
        artist_count: Dict[str, int] = {}
        guarded = genre_rerank and self._centroid_idx is not None
        pool_cap = max(n * 25, 500) if (
            diversity > 0 or max_per_artist or guarded
        ) else n
        for idx in order:
            i = int(idx)
            if int(self.track_ids[i]) == seed_id:
                continue
            akey = str(self.artists[i]).casefold()
            if seed_artist and seed_artist in akey:
                continue
            title_i = str(self.titles[i])
            # Approach 1: skip junk tracks
            if qmask is not None and not qmask[i]:
                continue
            if (
                qmask is not None
                and self._qfilter is not None
                and self._qfilter.seed_title_in_result(seed_title, title_i)
            ):
                continue
            key = f"{title_i.casefold()}::{akey}"
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

        # Guarded centroid rerank: only reorder the first 20 already-retrieved
        # candidates.  Ranks 21–50 remain frozen, preventing held-out Recall@50
        # regressions while correcting obvious scene errors in the visible top 5.
        if genre_rerank and self._centroid_idx is not None and chosen:
            centroid_score = self._centroid_idx.blend_with_genre(
                blended, seed_artist_raw, seed_neural_w=qn, gamma=genre_gamma)
            boundary = min(20, len(chosen))
            chosen = sorted(
                chosen[:boundary],
                key=lambda i: float(centroid_score[i]),
                reverse=True,
            ) + chosen[boundary:]
        chosen = chosen[:n]
        results = []
        from urllib.parse import quote
        for i in chosen:
            results.append({
                "title": str(self.titles[i]), "artist": str(self.artists[i]),
                "deezer_id": int(self.track_ids[i]),
                "neural_sim": round(float(neural_sim[i]), 4),
                "vibe_sim": round(float(vibe_sim[i]), 4),
                "spotify_url": f"https://open.spotify.com/search/{quote(str(self.titles[i]) + ' ' + str(self.artists[i]))}",
            })
        v = self._describe_vibe(row)
        return {"ok": True,
                "seed": {"title": str(self.titles[row]), "artist": str(self.artists[row])},
                "vibe": v, "results": results, "library_size": len(self)}

    def _recommend_sonic_tail(
        self, row: int, n: int, alpha: Optional[float], quality_filter: bool,
        genre_rerank: bool, genre_gamma: float,
    ) -> List[Dict]:
        sonic = self._sonic
        if sonic is None:
            raise ValueError("Sonic tail requires a sonic index")
        a = self.alpha if alpha is None else float(alpha)
        sonic_sim = self._sonic_cosine(np.asarray(sonic[row], dtype=np.float32))
        qv = self._vscaled[row]
        vibe_sim = 1.0 / (1.0 + np.linalg.norm(self._vscaled - qv, axis=1))
        blended = a * self._z(sonic_sim) + (1 - a) * self._z(vibe_sim)
        seed_artist = str(self.artists[row]).casefold()
        seed_title = str(self.titles[row])
        candidates, seen_recordings, seen_artists = [], set(), set()
        qfilter = self._qfilter
        qmask = self._qmask if quality_filter and qfilter is not None else None
        for raw in np.argsort(blended)[::-1]:
            candidate = int(raw)
            if candidate == row:
                continue
            title = str(self.titles[candidate])
            artist = str(self.artists[candidate])
            artist_key = artist.casefold()
            if seed_artist and seed_artist in artist_key:
                continue
            if qmask is not None and not bool(qmask[candidate]):
                continue
            if (
                qmask is not None
                and qfilter is not None
                and qfilter.seed_title_in_result(seed_title, title)
            ):
                continue
            recording = (title.casefold(), artist_key)
            if recording in seen_recordings or artist_key in seen_artists:
                continue
            seen_recordings.add(recording)
            seen_artists.add(artist_key)
            candidates.append(candidate)
            if len(candidates) >= 1250:
                break

        vectors = np.asarray(sonic[candidates], dtype=np.float32)
        chosen = []
        if len(vectors):
            vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)
            relevance = blended[candidates]
            relevance = (
                (relevance - relevance.min())
                / (relevance.max() - relevance.min() + 1e-9)
            )
            positions = [int(np.argmax(relevance))]
            best = vectors @ vectors[positions[0]]
            while len(positions) < min(n, len(candidates)):
                scores = 0.85 * relevance - 0.15 * best
                scores[positions] = -np.inf
                position = int(np.argmax(scores))
                positions.append(position)
                best = np.maximum(best, vectors @ vectors[position])
            chosen = [candidates[position] for position in positions]

        if genre_rerank and self._centroid_idx is not None and chosen:
            centroid = self._centroid_idx.blend_with_genre(
                blended, str(self.artists[row]), self._neural[row], gamma=genre_gamma
            )
            boundary = min(20, len(chosen))
            chosen = sorted(
                chosen[:boundary], key=lambda candidate: float(centroid[candidate]),
                reverse=True,
            ) + chosen[boundary:]

        from urllib.parse import quote
        return [{
            "title": str(self.titles[candidate]),
            "artist": str(self.artists[candidate]),
            "deezer_id": int(self.track_ids[candidate]),
            "neural_sim": round(float(sonic_sim[candidate]), 4),
            "vibe_sim": round(float(vibe_sim[candidate]), 4),
            "spotify_url": (
                "https://open.spotify.com/search/"
                + quote(str(self.titles[candidate]) + " " + str(self.artists[candidate]))
            ),
        } for candidate in chosen[:n]]

    def _mmr(self, cand: List[int], blended: np.ndarray, n: int, diversity: float) -> List[int]:
        if not cand:
            return []
        d = float(np.clip(diversity, 0.0, 1.0))
        rel_raw = blended[cand]
        rel = (rel_raw - rel_raw.min()) / (rel_raw.max() - rel_raw.min() + 1e-9)
        vecs = self._neural[cand]
        chosen = [int(np.argmax(rel))]
        best = vecs @ vecs[chosen[0]]
        while len(chosen) < min(n, len(cand)):
            scores = (1 - d) * rel - d * best
            for p in chosen:
                scores[p] = -np.inf
            nxt = int(np.argmax(scores))
            chosen.append(nxt)
            best = np.maximum(best, vecs @ vecs[nxt])
        return [cand[p] for p in chosen]

    def _describe_vibe(self, row: int) -> Dict[str, str]:
        """Human tags from the standardized vibe vector (z>0 = above library avg)."""
        std = self._vscaled[row] / (self._w + 1e-9)  # recover (v-mean)/sd per feature
        z = {n: float(v) for n, v in zip(self.feature_names, std)}
        low = z.get("low_end_ratio", 0) + z.get("band_sub", 0) + z.get("band_bass", 0)
        def band(v, labels):
            return labels[0] if v < -0.5 else (labels[2] if v > 0.5 else labels[1])
        return {
            "low_end": band(low / 3.0, ["bass-light", "balanced low-end", "bass-heavy"]),
            "dynamics": band(z.get("dynamic_range", 0), ["compressed", "moderate dynamics", "very dynamic"]),
            "tone": band(z.get("brightness", 0), ["warm", "neutral", "bright"]),
        }


def get_recommender() -> WebRecommender:
    """Lazy singleton: fetch the index once, reuse across warm invocations."""
    global _RECO
    if _RECO is not None:
        return _RECO
    with _LOCK:
        if _RECO is not None:
            return _RECO
        path = _INDEX_PATH
        if not path:
            path = f"/tmp/deepvibe_index_{_INDEX_VERSION}.npz"
            if not os.path.exists(path) or _sha256(path) != _INDEX_SHA256:
                partial = f"{path}.part"
                try:
                    urllib.request.urlretrieve(_INDEX_URL, partial)
                    digest = _sha256(partial)
                    if digest != _INDEX_SHA256:
                        raise RuntimeError(
                            f"Index checksum mismatch: expected {_INDEX_SHA256}, got {digest}"
                        )
                    os.replace(partial, path)
                finally:
                    if os.path.exists(partial):
                        os.unlink(partial)
        _RECO = WebRecommender(path)
        return _RECO


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
