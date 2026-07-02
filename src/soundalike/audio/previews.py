"""Deezer catalog client — used ONLY to enumerate songs and fetch audio.

Deezer's public API is free and needs no authentication. We use it strictly to:
  1. resolve a song to a real track (+ its 30s MP3 preview URL), and
  2. gather a *candidate pool* of tracks that might be worth recommending.

Crucially, we never use Deezer's opinion of what's "similar" for ranking — that
is done entirely by our own acoustic analysis. Deezer only answers "what songs
exist and where's their audio", which is unavoidable for any recommender.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import time

import requests

API_ROOT = "https://api.deezer.com"


@dataclass
class DeezerTrack:
    id: int
    title: str
    artist: str
    artist_id: int
    preview_url: str

    @property
    def has_preview(self) -> bool:
        return bool(self.preview_url)


def _parse_track(item: dict) -> Optional[DeezerTrack]:
    if not item or not item.get("id"):
        return None
    artist = item.get("artist") or {}
    return DeezerTrack(
        id=int(item["id"]),
        title=item.get("title", ""),
        artist=artist.get("name", ""),
        artist_id=int(artist.get("id", 0) or 0),
        preview_url=item.get("preview", "") or "",
    )


class DeezerClient:
    def __init__(self, session: Optional[requests.Session] = None):
        self.session = session or requests.Session()

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        # Deezer returns HTTP 200 with a JSON error body (code 4) when the free
        # rate limit is hit; back off and retry rather than failing the harvest.
        for attempt in range(6):
            response = self.session.get(f"{API_ROOT}{path}", params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and data.get("error"):
                code = data["error"].get("code") if isinstance(data["error"], dict) else None
                if code == 4:  # quota / rate limit
                    time.sleep(min(2 ** attempt, 20))
                    continue
                raise RuntimeError(f"Deezer API error: {data['error']}")
            return data
        raise RuntimeError("Deezer rate limit: retries exhausted.")

    def search_track(self, title: str, artist: Optional[str] = None) -> Optional[DeezerTrack]:
        query = f'track:"{title}"'
        if artist:
            query += f' artist:"{artist}"'
        data = self._get("/search", {"q": query, "limit": 1})
        items = data.get("data", [])
        if not items:  # fall back to a looser free-text search
            data = self._get("/search", {"q": f"{artist or ''} {title}".strip(), "limit": 1})
            items = data.get("data", [])
        return _parse_track(items[0]) if items else None

    def artist_top_tracks(self, artist_id: int, limit: int = 25) -> List[DeezerTrack]:
        if not artist_id:
            return []
        data = self._get(f"/artist/{artist_id}/top", {"limit": limit})
        return [t for t in map(_parse_track, data.get("data", [])) if t]

    def related_artists(self, artist_id: int, limit: int = 8) -> List[int]:
        if not artist_id:
            return []
        data = self._get(f"/artist/{artist_id}/related", {"limit": limit})
        return [int(a["id"]) for a in data.get("data", []) if a.get("id")]

    def download_preview(self, track: DeezerTrack, dest: Path) -> Optional[Path]:
        if not track.has_preview:
            return None
        dest.parent.mkdir(parents=True, exist_ok=True)
        response = self.session.get(track.preview_url, timeout=30)
        response.raise_for_status()
        dest.write_bytes(response.content)
        return dest

    def gather_candidates(
        self,
        seeds: List[DeezerTrack],
        per_artist: int = 25,
        related_per_seed: int = 6,
    ) -> Dict[int, DeezerTrack]:
        """Build a candidate pool: each seed artist's catalog plus the catalogs
        of neighbouring artists, so there's breadth beyond the seed.

        Returns a dict keyed by Deezer track id (deduplicated). Neighbouring
        artists are used only to *widen the net*; every candidate is still
        ranked purely by our acoustic analysis downstream.
        """
        pool: Dict[int, DeezerTrack] = {}
        seen_artists: set[int] = set()

        def add_artist(artist_id: int) -> None:
            if not artist_id or artist_id in seen_artists:
                return
            seen_artists.add(artist_id)
            for track in self.artist_top_tracks(artist_id, per_artist):
                if track.has_preview:
                    pool.setdefault(track.id, track)

        for seed in seeds:
            add_artist(seed.artist_id)
            for related_id in self.related_artists(seed.artist_id, related_per_seed):
                add_artist(related_id)
        return pool
