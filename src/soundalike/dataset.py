"""Dataset loading and normalization.

Wraps a pandas DataFrame with a canonical schema so the recommender can work on
the bundled dataset or any user-supplied CSV that carries audio-feature columns.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .features import AUDIO_FEATURES

# Maps assorted incoming column names -> canonical names.
_CANONICAL_COLUMNS = {
    "title": "title",
    "track": "title",
    "name": "title",
    "artist(s)": "artist",
    "artist": "artist",
    "artists": "artist",
    "release": "release",
    "released_year": "release",
    "num_of_streams": "streams",
    "streams": "streams",
    "bpm": "bpm",
    "tempo": "bpm",
    "key": "key",
    "mode": "mode",
    "danceability": "danceability",
    "danceability_%": "danceability",
    "valence": "valence",
    "valence_%": "valence",
    "energy": "energy",
    "energy_%": "energy",
    "acousticness": "acousticness",
    "acousticness_%": "acousticness",
    "instrumentalness": "instrumentalness",
    "instrumentalness_%": "instrumentalness",
    "liveness": "liveness",
    "liveness_%": "liveness",
    "speechiness": "speechiness",
    "speechiness_%": "speechiness",
}

# Multi-artist rows in the bundled CSV use " . " as a separator.
_MULTI_ARTIST_SEP = " . "


def _normalize_key(text: str) -> str:
    return " ".join(str(text).strip().casefold().split())


class Dataset:
    """A normalized table of songs with audio features."""

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame.reset_index(drop=True)

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_csv(cls, path: str | Path) -> "Dataset":
        path = Path(path)
        raw = pd.read_csv(path)
        return cls._from_raw(raw)

    @classmethod
    def _from_raw(cls, raw: pd.DataFrame) -> "Dataset":
        renamed = {}
        for col in raw.columns:
            key = str(col).strip().lower()
            if key in _CANONICAL_COLUMNS:
                renamed[col] = _CANONICAL_COLUMNS[key]
        frame = raw.rename(columns=renamed)

        missing = [f for f in AUDIO_FEATURES if f not in frame.columns]
        if missing:
            raise ValueError(
                "CSV is missing required audio-feature column(s): "
                f"{missing}. Present columns: {list(frame.columns)}"
            )
        if "title" not in frame.columns:
            raise ValueError("CSV must have a 'title' column.")
        if "artist" not in frame.columns:
            frame["artist"] = ""

        # Coerce feature columns to numeric and drop rows we cannot use.
        for feat in AUDIO_FEATURES:
            frame[feat] = pd.to_numeric(frame[feat], errors="coerce")
        frame = frame.dropna(subset=AUDIO_FEATURES).reset_index(drop=True)

        frame["primary_artist"] = (
            frame["artist"].astype(str).str.split(_MULTI_ARTIST_SEP).str[0].str.strip()
        )
        frame["_title_key"] = frame["title"].map(_normalize_key)
        frame["_artist_key"] = frame["primary_artist"].map(_normalize_key)
        frame["_dedup_key"] = frame["_title_key"] + " :: " + frame["_artist_key"]
        return cls(frame)

    # --------------------------------------------------------------- accessors
    def __len__(self) -> int:
        return len(self.frame)

    def feature_matrix(self, features: Optional[Sequence[str]] = None) -> np.ndarray:
        features = list(features) if features else list(AUDIO_FEATURES)
        return self.frame[features].to_numpy(dtype=float)

    def label(self, index: int) -> str:
        row = self.frame.iloc[index]
        return f"{row['title']} — {row['artist']}"

    # ---------------------------------------------------------------- matching
    def find_one(self, title: str, artist: Optional[str] = None) -> Optional[int]:
        """Return the row index best matching a title (and optional artist)."""
        matches = self.find_all(title, artist)
        return matches[0] if matches else None

    def find_all(self, title: str, artist: Optional[str] = None) -> List[int]:
        tkey = _normalize_key(title)
        frame = self.frame
        exact = frame.index[frame["_title_key"] == tkey].tolist()
        if not exact:
            # Fall back to substring matching on the title.
            contains = frame.index[
                frame["_title_key"].str.contains(tkey, regex=False)
            ].tolist()
            exact = contains
        if artist:
            akey = _normalize_key(artist)
            filtered = [
                i for i in exact if akey in frame.at[i, "_artist_key"] or akey in _normalize_key(frame.at[i, "artist"])
            ]
            if filtered:
                return filtered
        return exact

    def find_many(
        self, seeds: Iterable[Tuple[str, Optional[str]]]
    ) -> Tuple[List[int], List[Tuple[str, Optional[str]]]]:
        """Resolve (title, artist) seeds to row indices.

        Returns (matched_indices, unmatched_seeds).
        """
        matched: List[int] = []
        unmatched: List[Tuple[str, Optional[str]]] = []
        seen: set[int] = set()
        for title, artist in seeds:
            idx = self.find_one(title, artist)
            if idx is None:
                unmatched.append((title, artist))
            elif idx not in seen:
                seen.add(idx)
                matched.append(idx)
        return matched, unmatched


def _bundled_csv_path() -> Path:
    try:
        resource = resources.files("soundalike").joinpath("data/spotify_data.csv")
        with resources.as_file(resource) as p:
            if Path(p).exists():
                return Path(p)
    except (ModuleNotFoundError, FileNotFoundError, AttributeError, TypeError):
        pass

    here = Path(__file__).resolve()
    candidates = [
        here.parent / "data" / "spotify_data.csv",
        here.parents[2] / "spotify_data.csv",  # repo root
        Path.cwd() / "spotify_data.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate the bundled spotify_data.csv dataset.")


def load_bundled_dataset() -> Dataset:
    """Load the dataset that ships with the package."""
    return Dataset.from_csv(_bundled_csv_path())
