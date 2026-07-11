"""Torch-free numpy recommender for the Vercel deployment.

The full model needs PyTorch (~2.9 GB) to embed an *arbitrary* song, which can't
live in a serverless function. But recommending from a song that's already in the
87k-track library needs **no torch at all** — every embedding is precomputed, and
ranking is pure numpy (whiten + cosine + vibe-blend + MMR).

This module mirrors `soundalike.ml.deepvibe.DeepVibeRecommender` exactly (a test
asserts identical top-k), so the hosted library-mode results match the desktop
app. The index is fetched once from the public GitHub Release and cached in
``/tmp`` across warm invocations.
"""

from __future__ import annotations

import os
import threading
import unicodedata
import urllib.request
from typing import Dict, List, Optional, Tuple

import numpy as np

# Where to get the index. A local path wins (dev); otherwise download the bundled
# pack asset from the public Release into the function's ephemeral /tmp.
_INDEX_URL = os.environ.get(
    "SOUNDALIKE_INDEX_URL",
    "https://github.com/yassinsolim/soundalike/releases/download/"
    "index-2026.07.04-272k/deepvibe_index.npz",
)
# Bump this when the index changes so warm instances with an old /tmp copy
# re-download instead of serving stale data.
_INDEX_VERSION = "2026.07.04-272k"
_INDEX_PATH = os.environ.get("SOUNDALIKE_INDEX_PATH", "")

_LOCK = threading.Lock()
_RECO: Optional["WebRecommender"] = None


import re

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


# Vibe feature weights — must match soundalike.audio.vibe.DEFAULT_WEIGHTS exactly
# (verified against weight_vector(DEFAULT_WEIGHTS)). Anything not listed is 1.0.
_DEFAULT_WEIGHTS = {
    "tempo": 1.5, "brightness": 1.5, "onset_rate": 1.5, "rms_std": 2.0,
    "dynamic_range": 2.5, "crest": 2.5, "low_end_ratio": 3.0,
    "band_sub": 2.5, "band_bass": 2.0,
}


class WebRecommender:
    """Loads a DeepVibeIndex .npz and ranks library songs, numpy-only.

    By default, all three quality improvements are enabled (see ``recommend()``).
    Pass ``enhance=False`` to get the original unmodified baseline for ablation.
    """

    def __init__(self, path: str, alpha: float = 0.8, enhance: bool = True,
                 acc_cache_dir: Optional[str] = None):
        d = np.load(path, allow_pickle=True)
        self.track_ids = d["track_ids"]
        self.titles = d["titles"].astype(str)
        self.artists = d["artists"].astype(str)
        self.feature_names = [str(x) for x in d["feature_names"]]
        neural = d["neural"].astype(np.float32)
        vibe = d["vibe"].astype(np.float32)
        self.alpha = float(alpha)

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
            self._by_pair.setdefault((t, a), i)
            self._by_title.setdefault(t, []).append(i)

        # ── Enhancement modules (loaded only when enhance=True) ──────────────
        self._qfilter = None
        self._centroid_idx = None
        self._related_graph = None

        if enhance:
            self._load_enhancements(acc_cache_dir)

    def _load_enhancements(self, acc_cache_dir: Optional[str]) -> None:
        """Lazily load the three quality-improvement modules.

        Designed to degrade gracefully: each module is skipped silently if its
        dependency is unavailable (e.g. the soundalike package is not installed,
        or the acc_cache directory doesn't exist on the hosted runtime).
        """
        try:
            import sys
            import os
            # Add the src directory to path for soundalike package access.
            # In the Vercel runtime the package is installed; locally we also
            # search the development tree.
            _here = os.path.dirname(os.path.abspath(__file__))
            for candidate in (
                os.path.join(_here, "..", "..", "src"),  # local dev: webapp/api/../../src
                os.path.join(_here, "..", "..", "..", "src"),  # deeper nesting
            ):
                if os.path.isdir(os.path.join(candidate, "soundalike")):
                    if candidate not in sys.path:
                        sys.path.insert(0, candidate)
                    break

            # Approach 1: quality filter
            from soundalike.ml.quality_filter import TitleQualityFilter
            self._qfilter = TitleQualityFilter()
            # Pre-compute mask for the whole library once (fast boolean array)
            self._qmask = self._qfilter.keep_mask(self.titles, self.artists)
        except Exception:
            self._qmask = None

        try:
            # Approach 2: artist-centroid genre reranker
            from soundalike.ml.genre_rerank import ArtistCentroidIndex
            self._centroid_idx = ArtistCentroidIndex(
                self._neural, self.artists, min_songs=2)
        except Exception:
            pass

        try:
            # Approach 3: related-artist collaborative graph
            from soundalike.ml.related_artists_rerank import RelatedArtistGraph
            from pathlib import Path
            acd = Path(acc_cache_dir) if acc_cache_dir else None
            if acd is None:
                # Guess common locations (local dev)
                for candidate in (
                    Path(__file__).resolve().parents[3] / "ml_data" / "acc_cache",
                    Path(__file__).resolve().parents[2] / "ml_data" / "acc_cache",
                ):
                    if candidate.exists():
                        acd = candidate
                        break
            self._related_graph = RelatedArtistGraph(
                acc_cache_dir=acd, use_manual=True, boost=0.15)
        except Exception:
            pass

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
            return self._by_title[t][0]
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

    def recommend(self, row: int, n: int = 20, alpha: Optional[float] = None,
                  diversity: float = 0.15, max_per_artist: int = 1,
                  # Enhancement flags (all True by default = best validated method)
                  quality_filter: bool = True,
                  genre_rerank: bool = True,
                  related_boost: bool = True,
                  genre_gamma: float = 0.25,
                  related_gamma: float = 0.20,
                  ) -> Dict:
        """Rank library songs for a seed row.

        Three complementary improvements over the plain neural+vibe blend:

        * **quality_filter** (Approach 1): removes junk derivatives (slowed,
          karaoke, tribute, nightcore) from the candidate pool.
        * **genre_rerank** (Approach 2): adds an artist-centroid genre-coherence
          term so acoustically-similar but genre-incoherent artists (e.g. metal
          bands near shoegaze seeds) are gently demoted.
        * **related_boost** (Approach 3): adds a collaborative related-artist
          prior (Deezer editorial + curated pairs) that directly boosts candidates
          whose artist is editorially related to the seed artist.

        All three flags default to True. Pass ``quality_filter=False`` etc. to
        ablate individual contributions (useful for A/B evaluation).
        """
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

        # ── Approach 2: artist-centroid genre coherence ──────────────────────
        if genre_rerank and self._centroid_idx is not None:
            blended = self._centroid_idx.blend_with_genre(
                blended, seed_artist_raw, seed_neural_w=qn, gamma=genre_gamma)

        # ── Approach 3: related-artist collaborative boost ───────────────────
        if related_boost and self._related_graph is not None:
            blended = self._related_graph.blend_with_related(
                blended, self.artists, seed_artist_raw, gamma=related_gamma)

        # Re-normalise to [0,1] so downstream comparisons stay meaningful
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
        pool_cap = max(n * 25, 500) if (diversity > 0 or max_per_artist) else n
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

        chosen = self._mmr(cand, blended, n, diversity) if diversity > 0 else cand[:n]
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
            if not os.path.exists(path):
                urllib.request.urlretrieve(_INDEX_URL, path)
        _RECO = WebRecommender(path)
        return _RECO
