"""Build and query a library of real songs by their "vibe" fingerprint.

To recommend actual listenable music (not one artist's handful of neighbours),
we harvest a broad, genre-diverse set of tracks from Deezer, measure each one's
vibe features once, and cache them as an index. A recommendation then analyses
your seed song and returns the library tracks whose bass-profile + dynamics +
tempo + timbre are closest — weighted so the low-end and the drops matter most.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ..config import cache_dir
from .previews import DeezerClient, DeezerTrack
from .vibe import (
    DEFAULT_WEIGHTS,
    FEATURE_NAMES,
    VibeFeatures,
    vibe_from_file,
    weight_vector,
)

# Deezer genre ids -> label, chosen to span very different sonic territory so a
# bass-heavy dynamic track has bass-heavy dynamic neighbours to match.
HARVEST_GENRES: Dict[int, str] = {
    116: "rap_hiphop",
    113: "dance_edm",
    106: "electro",
    132: "pop",
    152: "rock",
    165: "rnb",
    85: "alternative",
    464: "metal",
}


@dataclass
class VibeEntry:
    track_id: int
    title: str
    artist: str
    features: VibeFeatures


class VibeIndex:
    def __init__(self, entries: List[VibeEntry]):
        self.entries = entries
        self.matrix = np.vstack([e.features.vector() for e in entries]) if entries else np.zeros((0,))

    def __len__(self) -> int:
        return len(self.entries)

    # ------------------------------------------------------------------ persist
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "feature_names": FEATURE_NAMES,
            "entries": [
                {"track_id": e.track_id, "title": e.title, "artist": e.artist,
                 "features": e.features.to_dict()}
                for e in self.entries
            ],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "VibeIndex":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        entries = [
            VibeEntry(e["track_id"], e["title"], e["artist"],
                      VibeFeatures.from_dict(e["features"]))
            for e in data["entries"]
        ]
        return cls(entries)

    @classmethod
    def bundled_path(cls) -> Optional[Path]:
        """The library that ships with the package, if present."""
        try:
            from importlib import resources

            resource = resources.files("soundalike").joinpath("data/vibe_index.json")
            with resources.as_file(resource) as p:
                if Path(p).exists():
                    return Path(p)
        except (ModuleNotFoundError, FileNotFoundError, AttributeError, TypeError):
            pass
        bundled = Path(__file__).resolve().parents[1] / "data" / "vibe_index.json"
        return bundled if bundled.exists() else None

    @classmethod
    def user_path(cls) -> Path:
        """Writable location for a user-built library (takes precedence for reads)."""
        return cache_dir() / "vibe_index.json"

    @classmethod
    def default_path(cls) -> Path:
        """For reads: prefer a user-built library, else the bundled default."""
        user = cls.user_path()
        if user.exists():
            return user
        return cls.bundled_path() or user


def build_index(
    per_genre: int = 120,
    per_artist: int = 12,
    client: Optional[DeezerClient] = None,
    analyzer: Callable[[str], VibeFeatures] = vibe_from_file,
    genres: Optional[Dict[int, str]] = None,
    existing: Optional[VibeIndex] = None,
    progress: Callable[[str], None] = print,
) -> VibeIndex:
    """Harvest tracks across genres and compute each one's vibe features."""
    from tempfile import TemporaryDirectory

    client = client or DeezerClient()
    genres = genres or HARVEST_GENRES
    have: Dict[int, VibeEntry] = {}
    if existing:
        have = {e.track_id: e for e in existing.entries}

    # Collect candidate tracks first (fast), then analyze (slow).
    candidates: Dict[int, DeezerTrack] = {}
    for genre_id, label in genres.items():
        got = 0
        try:
            data = client._get(f"/genre/{genre_id}/artists", {"limit": max(30, per_genre // 3)})
        except Exception as exc:  # noqa: BLE001
            progress(f"[{label}] artist list failed: {exc}")
            continue
        for a in data.get("data", []):
            if got >= per_genre:
                break
            for t in client.artist_top_tracks(int(a["id"]), per_artist):
                if t.has_preview and t.id not in candidates and t.id not in have:
                    candidates[t.id] = t
                    got += 1
                    if got >= per_genre:
                        break
        progress(f"[{label}] {got} new candidates")

    progress(f"Analyzing {len(candidates)} tracks (cached: {len(have)})...")
    entries = list(have.values())
    t0 = time.time()
    with TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        for i, track in enumerate(candidates.values(), 1):
            try:
                dest = workdir / f"{track.id}.mp3"
                if client.download_preview(track, dest) is None:
                    continue
                feats = analyzer(str(dest))
                dest.unlink(missing_ok=True)
                entries.append(VibeEntry(track.id, track.title, track.artist, feats))
            except Exception as exc:  # noqa: BLE001
                continue
            if i % 100 == 0:
                progress(f"  analyzed {i}/{len(candidates)} ({i/(time.time()-t0):.1f}/s)")
    progress(f"Index size: {len(entries)} tracks")
    return VibeIndex(entries)


@dataclass
class VibeRecommendation:
    title: str
    artist: str
    score: float
    track_id: int

    def __str__(self) -> str:
        return f"{self.title} — {self.artist}  ({self.score:.3f})"


class VibeRecommender:
    """Rank a VibeIndex by weighted vibe-distance to a seed track."""

    def __init__(self, index: VibeIndex, weights: Optional[Dict[str, float]] = None):
        if len(index) < 2:
            raise ValueError("Vibe index is empty — build it first.")
        self.index = index
        self.weights = weights if weights is not None else dict(DEFAULT_WEIGHTS)
        # Standardize features across the library, then apply sqrt-weights so
        # euclidean distance becomes a weighted distance.
        self._mean = index.matrix.mean(axis=0)
        self._std = index.matrix.std(axis=0) + 1e-9
        w = np.sqrt(np.clip(weight_vector(self.weights), 0.0, None))
        self._scaled = ((index.matrix - self._mean) / self._std) * w
        self._w = w

    def _project(self, feats: VibeFeatures) -> np.ndarray:
        return ((feats.vector() - self._mean) / self._std) * self._w

    def recommend(
        self,
        seed_feats: VibeFeatures,
        n: int = 15,
        exclude_ids: Optional[set] = None,
        exclude_artist: Optional[str] = None,
    ) -> List[VibeRecommendation]:
        exclude_ids = exclude_ids or set()
        exclude_artist = (exclude_artist or "").casefold()
        q = self._project(seed_feats)
        dist = np.linalg.norm(self._scaled - q, axis=1)
        scores = 1.0 / (1.0 + dist)
        order = np.argsort(scores)[::-1]

        results: List[VibeRecommendation] = []
        seen: set = set()
        for idx in order:
            e = self.index.entries[int(idx)]
            if e.track_id in exclude_ids:
                continue
            key = f"{e.title.casefold()}::{e.artist.casefold()}"
            if key in seen:
                continue
            if exclude_artist and exclude_artist in e.artist.casefold():
                continue
            seen.add(key)
            results.append(VibeRecommendation(e.title, e.artist, float(scores[int(idx)]), e.track_id))
            if len(results) >= n:
                break
        return results
