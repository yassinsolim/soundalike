"""Helpers for building a taste profile from a list of seed songs.

A "seed" is a (title, artist) pair. Seeds can come from a simple text file
(one song per line), a CSV with title/artist columns, or an inline string.
This module is also the bridge point for live Spotify data later: whatever
source produces (title, artist) pairs can feed the recommender.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

Seed = Tuple[str, Optional[str]]

# Separators accepted between a title and an artist on one line, e.g.
# "Blinding Lights - The Weeknd" or "Shape of You | Ed Sheeran".
_LINE_SEPARATORS = (" - ", " – ", " — ", " | ", "\t")


def parse_seed_line(line: str) -> Optional[Seed]:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    for sep in _LINE_SEPARATORS:
        if sep in line:
            title, artist = line.split(sep, 1)
            return title.strip(), artist.strip() or None
    return line, None


def parse_seed_string(text: str) -> List[Seed]:
    """Parse a ';'- or newline-separated list of seeds."""
    chunks: List[str] = []
    for part in text.replace("\r", "\n").split("\n"):
        chunks.extend(part.split(";"))
    seeds: List[Seed] = []
    for chunk in chunks:
        seed = parse_seed_line(chunk)
        if seed:
            seeds.append(seed)
    return seeds


def read_seed_file(path: str | Path) -> List[Seed]:
    """Read seeds from a .csv (title/artist columns) or a plain text file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Seed file not found: {path}")

    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        cols = {c.lower(): c for c in frame.columns}
        title_col = cols.get("title") or cols.get("track") or cols.get("name")
        if title_col is None:
            raise ValueError("Seed CSV must have a 'title' (or 'track'/'name') column.")
        artist_col = cols.get("artist") or cols.get("artist(s)") or cols.get("artists")
        seeds: List[Seed] = []
        for _, row in frame.iterrows():
            title = str(row[title_col]).strip()
            if not title or title.lower() == "nan":
                continue
            artist = str(row[artist_col]).strip() if artist_col else None
            seeds.append((title, artist or None))
        return seeds

    return parse_seed_string(path.read_text(encoding="utf-8"))
