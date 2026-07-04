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
    "index-2026.07.04/deepvibe_index.npz",
)
# Bump this when the index changes so warm instances with an old /tmp copy
# re-download instead of serving stale data.
_INDEX_VERSION = "2026.07.04"
_INDEX_PATH = os.environ.get("SOUNDALIKE_INDEX_PATH", "")

_LOCK = threading.Lock()
_RECO: Optional["WebRecommender"] = None


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    s = s.casefold().strip()
    for sep in (" feat", " ft.", " ft ", " featuring", " (feat", " with "):
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
    """Loads a DeepVibeIndex .npz and ranks library songs, numpy-only."""

    def __init__(self, path: str, alpha: float = 0.8):
        d = np.load(path, allow_pickle=True)
        self.track_ids = d["track_ids"]
        self.titles = d["titles"].astype(str)
        self.artists = d["artists"].astype(str)
        self.feature_names = [str(x) for x in d["feature_names"]]
        neural = d["neural"].astype(np.float32)
        vibe = d["vibe"].astype(np.float32)
        self.alpha = float(alpha)

        # --- neural: L2-normalize, then ZCA-whiten (identical to production) ---
        neural /= np.linalg.norm(neural, axis=1, keepdims=True) + 1e-9
        self._nmean = neural.mean(axis=0)
        centered = neural - self._nmean
        cov = np.cov(centered.T)
        evals, evecs = np.linalg.eigh(cov)
        self._W = evecs @ np.diag(1.0 / np.sqrt(np.clip(evals, 1e-5, None))) @ evecs.T
        self._neural = self._apply_whiten(neural)

        # --- vibe: standardize, sqrt-weight (identical to production) ---
        self._vmean = vibe.mean(axis=0)
        self._vstd = vibe.std(axis=0) + 1e-9
        w = np.array([_DEFAULT_WEIGHTS.get(n, 1.0) for n in self.feature_names], np.float32)
        self._w = np.sqrt(np.clip(w, 0.0, None))
        self._vscaled = ((vibe - self._vmean) / self._vstd) * self._w

        # Fast library-hit lookups.
        self._by_pair: Dict[Tuple[str, str], int] = {}
        self._by_title: Dict[str, List[int]] = {}
        for i in range(len(self.titles)):
            t, a = _norm(self.titles[i]), _norm(self.artists[i])
            self._by_pair.setdefault((t, a), i)
            self._by_title.setdefault(t, []).append(i)

    def _apply_whiten(self, vecs: np.ndarray) -> np.ndarray:
        x = (vecs - self._nmean) @ self._W
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

    def __len__(self) -> int:
        return len(self.titles)

    # -------------------------------------------------------------- search/seed
    def find_row(self, title: str, artist: str = "") -> Optional[int]:
        t, a = _norm(title), _norm(artist)
        if a and (t, a) in self._by_pair:
            return self._by_pair[(t, a)]
        if not a and len(self._by_title.get(t, [])) >= 1:
            return self._by_title[t][0]
        # loose contains match on title
        for i in range(len(self.titles)):
            if t and t in _norm(self.titles[i]):
                if not a or a in _norm(self.artists[i]):
                    return i
        return None

    def search(self, q: str, limit: int = 8) -> List[Dict]:
        """Substring search over the library for the seed picker/autocomplete."""
        nq = _norm(q)
        if not nq:
            return []
        hits, seen = [], set()
        for i in range(len(self.titles)):
            hay = _norm(self.titles[i]) + " " + _norm(self.artists[i])
            if nq in hay:
                key = (_norm(self.titles[i]), _norm(self.artists[i]))
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
                  diversity: float = 0.15, max_per_artist: int = 1) -> Dict:
        a = self.alpha if alpha is None else float(alpha)
        qn = self._neural[row]
        neural_sim = self._neural @ qn
        qv = self._vscaled[row]
        vibe_sim = 1.0 / (1.0 + np.linalg.norm(self._vscaled - qv, axis=1))
        blended = a * self._z(neural_sim) + (1 - a) * self._z(vibe_sim)
        order = np.argsort(blended)[::-1]

        seed_artist = str(self.artists[row]).casefold()
        seed_id = int(self.track_ids[row])
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
            key = f"{str(self.titles[i]).casefold()}::{akey}"
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
