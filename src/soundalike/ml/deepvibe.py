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
from typing import Callable, Dict, List, Optional

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

    def __init__(self, track_ids, titles, artists, neural, vibe):
        self.track_ids = np.asarray(track_ids)
        self.titles = np.asarray(titles, dtype=object)
        self.artists = np.asarray(artists, dtype=object)
        self.neural = np.asarray(neural, dtype=np.float32)      # (N, d)
        self.vibe = np.asarray(vibe, dtype=np.float32)          # (N, 29)

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
        saver(
            path, track_ids=self.track_ids, titles=self.titles.astype(str),
            artists=self.artists.astype(str), neural=neural, vibe=self.vibe,
            feature_names=np.array(FEATURE_NAMES),
        )

    @classmethod
    def load(cls, path: Path) -> "DeepVibeIndex":
        d = np.load(Path(path), allow_pickle=True)
        # Neural may be stored float16 (bundled) — upcast for downstream math.
        return cls(d["track_ids"], d["titles"], d["artists"],
                   d["neural"].astype(np.float32), d["vibe"])

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
    """Rank a DeepVibeIndex by a tunable blend of neural + vibe similarity."""

    def __init__(
        self,
        index: DeepVibeIndex,
        alpha: float = 0.8,
        vibe_weights: Optional[Dict[str, float]] = None,
        whiten: bool = True,
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

    def _apply_whiten(self, vecs: np.ndarray) -> np.ndarray:
        """Center + ZCA-whiten + re-normalize (rows) of one or many embeddings."""
        x = (vecs - self._nmean) @ self._W
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)

    def _project_vibe(self, feats: VibeFeatures) -> np.ndarray:
        return ((feats.vector() - self._vmean) / self._vstd) * self._w

    @staticmethod
    def _zscore(x: np.ndarray) -> np.ndarray:
        return (x - x.mean()) / (x.std() + 1e-9)

    def recommend(
        self,
        seed_neural: np.ndarray,
        seed_vibe: VibeFeatures,
        n: int = 15,
        exclude_ids: Optional[set] = None,
        exclude_artist: Optional[str] = None,
    ) -> List[DeepVibeRecommendation]:
        exclude_ids = exclude_ids or set()
        exclude_artist = (exclude_artist or "").casefold()

        qn = seed_neural / (np.linalg.norm(seed_neural) + 1e-9)
        if self._whiten:
            qn = self._apply_whiten(qn)
        neural_sim = self._neural @ qn                                   # cosine, -1..1
        qv = self._project_vibe(seed_vibe)
        vibe_sim = 1.0 / (1.0 + np.linalg.norm(self._vscaled - qv, axis=1))

        # Blend on comparable (z-scored) scales so alpha is meaningful.
        blended = self.alpha * self._zscore(neural_sim) + (1 - self.alpha) * self._zscore(vibe_sim)

        order = np.argsort(blended)[::-1]
        results: List[DeepVibeRecommendation] = []
        seen: set = set()
        for idx in order:
            i = int(idx)
            tid = int(self.index.track_ids[i])
            if tid in exclude_ids:
                continue
            title, artist = str(self.index.titles[i]), str(self.index.artists[i])
            key = f"{title.casefold()}::{artist.casefold()}"
            if key in seen:
                continue
            if exclude_artist and exclude_artist in artist.casefold():
                continue
            seen.add(key)
            results.append(DeepVibeRecommendation(
                title=title, artist=artist, score=float(blended[i]), track_id=tid,
                neural_sim=float(neural_sim[i]), vibe_sim=float(vibe_sim[i]),
            ))
            if len(results) >= n:
                break
        return results


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

