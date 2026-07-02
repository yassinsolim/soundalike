"""Thin Spotify Web API client.

Only uses endpoints that remain available to new apps after the 2024-11-27
lockdown: user library / top / recently-played, artists, search, and playlist
creation. It deliberately does NOT touch Recommendations or Audio Features
(both removed for new apps) — similarity comes from our own engine + Last.fm.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import requests

from .auth import SpotifyAuth

API_BASE = "https://api.spotify.com/v1"


class SpotifyAPIError(RuntimeError):
    """A Spotify Web API request failed with a helpful, user-facing message."""


def _describe_http_error(response: requests.Response) -> str:
    status = response.status_code
    try:
        message = response.json().get("error", {}).get("message", "")
    except ValueError:
        message = (response.text or "").strip()
    detail = f"Spotify API {status}" + (f": {message}" if message else "")

    if status == 403:
        detail += (
            "\nThis usually means your Spotify app is in Development mode and the "
            "authorizing account isn't on its allowlist. Fix: open the app at "
            "https://developer.spotify.com/dashboard -> Settings -> User Management, "
            "add your Spotify account's name and email, then run `soundalike login` again. "
            "See SETUP.md."
        )
    elif status == 401:
        detail += "\nYour session expired. Run `soundalike login` again."
    return detail


def _normalize_track(track: dict) -> Optional[Dict[str, str]]:
    if not track or not track.get("id"):
        return None
    artists = [a["name"] for a in track.get("artists", []) if a.get("name")]
    return {
        "id": track["id"],
        "uri": track.get("uri", f"spotify:track:{track['id']}"),
        "title": track.get("name", ""),
        "artist": ", ".join(artists),
        "primary_artist": artists[0] if artists else "",
        "artist_ids": ",".join(a["id"] for a in track.get("artists", []) if a.get("id")),
    }


class SpotifyClient:
    def __init__(self, auth: SpotifyAuth):
        self.auth = auth
        self.session = requests.Session()

    # --------------------------------------------------------------- transport
    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        url = path if path.startswith("http") else f"{API_BASE}{path}"
        for attempt in range(4):
            token = self.auth.get_valid_token()
            headers = kwargs.pop("headers", {})
            headers["Authorization"] = f"Bearer {token.access_token}"
            response = self.session.request(method, url, headers=headers, timeout=30, **kwargs)

            if response.status_code == 429:  # rate limited
                wait = int(response.headers.get("Retry-After", "1")) + 1
                time.sleep(min(wait, 30))
                continue
            if response.status_code == 401 and attempt == 0:  # token just expired
                self.auth.refresh(token)
                continue
            if not response.ok:
                raise SpotifyAPIError(_describe_http_error(response))
            return response
        raise SpotifyAPIError(_describe_http_error(response))

    def _get(self, path: str, params: Optional[dict] = None) -> dict:
        return self._request("GET", path, params=params).json()

    # ------------------------------------------------------------------ user
    def current_user(self) -> dict:
        return self._get("/me")

    def _paged_tracks(self, path: str, params: dict, limit: int, item_key: Optional[str]) -> List[Dict[str, str]]:
        collected: List[Dict[str, str]] = []
        page_params = dict(params)
        page_params["limit"] = min(50, limit)
        url: Optional[str] = path
        while url and len(collected) < limit:
            data = self._get(url, params=page_params) if url == path else self._get(url)
            for item in data.get("items", []):
                raw = item[item_key] if item_key else item
                track = _normalize_track(raw)
                if track:
                    collected.append(track)
                if len(collected) >= limit:
                    break
            url = data.get("next")
            page_params = {}  # `next` already carries pagination params
        return collected[:limit]

    def liked_tracks(self, limit: int = 100) -> List[Dict[str, str]]:
        return self._paged_tracks("/me/tracks", {}, limit, item_key="track")

    def top_tracks(self, time_range: str = "medium_term", limit: int = 50) -> List[Dict[str, str]]:
        return self._paged_tracks(
            "/me/top/tracks", {"time_range": time_range}, limit, item_key=None
        )

    def recently_played(self, limit: int = 50) -> List[Dict[str, str]]:
        return self._paged_tracks(
            "/me/player/recently-played", {}, limit, item_key="track"
        )

    def artist_genres(self, artist_ids: List[str]) -> Dict[str, List[str]]:
        genres: Dict[str, List[str]] = {}
        unique = [a for a in dict.fromkeys(artist_ids) if a]
        for start in range(0, len(unique), 50):
            batch = unique[start : start + 50]
            data = self._get("/artists", params={"ids": ",".join(batch)})
            for artist in data.get("artists", []) or []:
                if artist:
                    genres[artist["id"]] = artist.get("genres", [])
        return genres

    # ---------------------------------------------------------------- search
    def search_track(self, title: str, artist: Optional[str] = None) -> Optional[Dict[str, str]]:
        query = f"track:{title}"
        if artist:
            query += f" artist:{artist}"
        data = self._get("/search", params={"q": query, "type": "track", "limit": 1})
        items = data.get("tracks", {}).get("items", [])
        return _normalize_track(items[0]) if items else None

    # -------------------------------------------------------------- playlists
    def create_playlist(
        self, name: str, track_uris: List[str], description: str = "", public: bool = False
    ) -> dict:
        user_id = self.current_user()["id"]
        playlist = self._request(
            "POST",
            f"/users/{user_id}/playlists",
            json={"name": name, "description": description, "public": public},
        ).json()
        for start in range(0, len(track_uris), 100):
            self._request(
                "POST",
                f"/playlists/{playlist['id']}/tracks",
                json={"uris": track_uris[start : start + 100]},
            )
        return playlist
