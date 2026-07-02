"""Minimal Last.fm API client.

Last.fm exposes crowd-sourced "similar tracks"/"similar artists" and tags built
from real listening behaviour. Since Spotify removed its Recommendations and
Related-Artists endpoints for new apps, Last.fm is our cross-catalog similarity
signal — and it works for *any* track, not just songs in the bundled dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import requests

API_ROOT = "https://ws.audioscrobbler.com/2.0/"


@dataclass
class SimilarTrack:
    title: str
    artist: str
    match: float  # 0..1 similarity as reported by Last.fm


class LastFmError(RuntimeError):
    pass


class LastFmClient:
    def __init__(self, api_key: str, session: Optional[requests.Session] = None):
        if not api_key:
            raise ValueError("A Last.fm API key is required.")
        self.api_key = api_key
        self.session = session or requests.Session()

    def _get(self, method: str, **params) -> dict:
        query = {"method": method, "api_key": self.api_key, "format": "json", **params}
        response = self.session.get(API_ROOT, params=query, timeout=30)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and "error" in data:
            raise LastFmError(f"Last.fm error {data['error']}: {data.get('message', '')}")
        return data

    def similar_tracks(
        self, artist: str, title: str, limit: int = 50, autocorrect: bool = True
    ) -> List[SimilarTrack]:
        data = self._get(
            "track.getsimilar",
            artist=artist,
            track=title,
            limit=limit,
            autocorrect=1 if autocorrect else 0,
        )
        raw = data.get("similartracks", {}).get("track", [])
        results: List[SimilarTrack] = []
        for item in raw:
            name = item.get("name")
            artist_name = (item.get("artist") or {}).get("name", "")
            if not name:
                continue
            try:
                match = float(item.get("match", 0.0) or 0.0)
            except (TypeError, ValueError):
                match = 0.0
            results.append(SimilarTrack(title=name, artist=artist_name, match=match))
        return results

    def similar_artists(self, artist: str, limit: int = 50, autocorrect: bool = True) -> List[str]:
        data = self._get(
            "artist.getsimilar", artist=artist, limit=limit, autocorrect=1 if autocorrect else 0
        )
        raw = data.get("similarartists", {}).get("artist", [])
        return [a["name"] for a in raw if a.get("name")]

    def track_tags(self, artist: str, title: str, autocorrect: bool = True) -> List[str]:
        try:
            data = self._get(
                "track.gettoptags", artist=artist, track=title,
                autocorrect=1 if autocorrect else 0,
            )
        except LastFmError:
            return []
        raw = data.get("toptags", {}).get("tag", [])
        return [t["name"] for t in raw if t.get("name")]
