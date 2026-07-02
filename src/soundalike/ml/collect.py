"""Harvest a training dataset of track previews across genres (Deezer).

Produces a manifest of tracks — id, title, artist, genre label, preview URL —
that later stages download and turn into mel-spectrograms. Genre labels are used
only to *color and sanity-check* the learned embedding space, never as training
targets (the model is self-supervised).

Deezer exposes editorial genres and per-genre chart/artist listings for free with
no auth, which is enough to assemble a broad, stylistically diverse set.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import requests

from ..audio.previews import DeezerClient, DeezerTrack

API_ROOT = "https://api.deezer.com"

# A spread of Deezer genre ids -> readable labels. These cover very different
# sonic territory so the embedding space has something to separate.
DEFAULT_GENRES: Dict[int, str] = {
    132: "pop",
    116: "rap_hiphop",
    152: "rock",
    113: "dance_edm",
    165: "rnb",
    98: "classical",
    129: "jazz",
    85: "alternative",
    106: "electro",
    144: "reggae",
    466: "folk",
    464: "metal",
}


@dataclass
class TrackEntry:
    track_id: int
    title: str
    artist: str
    genre: str
    preview_url: str


class DatasetCollector:
    def __init__(self, client: Optional[DeezerClient] = None):
        self.client = client or DeezerClient()
        self.session = self.client.session

    def _genre_artists(self, genre_id: int, limit: int = 40) -> List[int]:
        data = self.client._get(f"/genre/{genre_id}/artists", {"limit": limit})
        return [int(a["id"]) for a in data.get("data", []) if a.get("id")]

    def collect(
        self,
        per_genre: int = 60,
        per_artist: int = 8,
        genres: Optional[Dict[int, str]] = None,
        progress=print,
    ) -> List[TrackEntry]:
        genres = genres or DEFAULT_GENRES
        entries: List[TrackEntry] = []
        seen: set[int] = set()

        for genre_id, label in genres.items():
            got = 0
            progress(f"[{label}] gathering...")
            for artist_id in self._genre_artists(genre_id, limit=max(40, per_genre // 2)):
                if got >= per_genre:
                    break
                try:
                    tracks = self.client.artist_top_tracks(artist_id, per_artist)
                except requests.RequestException:
                    continue
                for track in tracks:
                    if got >= per_genre:
                        break
                    if track.id in seen or not track.has_preview:
                        continue
                    seen.add(track.id)
                    entries.append(
                        TrackEntry(track.id, track.title, track.artist, label, track.preview_url)
                    )
                    got += 1
                # Pace requests to stay under Deezer's ~50-per-5s free quota
                # (~5 req/s is comfortably safe and avoids the quota backoff).
                time.sleep(0.2)
            progress(f"[{label}] {got} tracks")
            time.sleep(1.0)  # breathe between genres
        return entries

    @staticmethod
    def save(entries: List[TrackEntry], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["track_id", "title", "artist", "genre", "preview_url"]
            )
            writer.writeheader()
            for entry in entries:
                writer.writerow(asdict(entry))

    @staticmethod
    def load(path: Path) -> List[TrackEntry]:
        with Path(path).open(encoding="utf-8") as handle:
            return [
                TrackEntry(
                    track_id=int(row["track_id"]),
                    title=row["title"],
                    artist=row["artist"],
                    genre=row["genre"],
                    preview_url=row["preview_url"],
                )
                for row in csv.DictReader(handle)
            ]


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Harvest a Deezer preview dataset.")
    parser.add_argument("--out", default="ml_data/manifest.csv", help="Output manifest CSV.")
    parser.add_argument("--per-genre", type=int, default=60)
    parser.add_argument("--per-artist", type=int, default=8)
    args = parser.parse_args(argv)

    collector = DatasetCollector()
    entries = collector.collect(per_genre=args.per_genre, per_artist=args.per_artist)
    out = Path(args.out)
    collector.save(entries, out)
    by_genre: Dict[str, int] = {}
    for e in entries:
        by_genre[e.genre] = by_genre.get(e.genre, 0) + 1
    print(f"\nSaved {len(entries)} tracks to {out}")
    print("Per genre:", json.dumps(by_genre))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
