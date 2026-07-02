"""Runtime configuration loaded from environment / a local .env file.

Secrets never live in the repo: copy .env.example to .env (git-ignored) and fill
in your own values. See SETUP.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:  # python-dotenv is an optional ("live") dependency.
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - exercised only when extra is absent
    load_dotenv = None  # type: ignore[assignment]

DEFAULT_REDIRECT_URI = "http://127.0.0.1:8888/callback"


def _load_dotenv(dotenv_path: Optional[str] = None) -> None:
    if load_dotenv is None:
        return
    if dotenv_path:
        load_dotenv(dotenv_path)
    else:
        # Search upward from the CWD for a .env file.
        load_dotenv()


def cache_dir() -> Path:
    """Directory for token caches and other local state (git-ignored, in $HOME)."""
    path = Path.home() / ".soundalike"
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class Config:
    spotify_client_id: Optional[str]
    spotify_redirect_uri: str
    lastfm_api_key: Optional[str]

    @classmethod
    def from_env(cls, dotenv_path: Optional[str] = None) -> "Config":
        _load_dotenv(dotenv_path)
        return cls(
            spotify_client_id=os.getenv("SPOTIFY_CLIENT_ID") or None,
            spotify_redirect_uri=os.getenv("SPOTIFY_REDIRECT_URI") or DEFAULT_REDIRECT_URI,
            lastfm_api_key=os.getenv("LASTFM_API_KEY") or None,
        )

    def require_spotify(self) -> None:
        if not self.spotify_client_id:
            raise RuntimeError(
                "SPOTIFY_CLIENT_ID is not set. Copy .env.example to .env and add your "
                "Spotify app's Client ID. See SETUP.md for the 2-minute walkthrough."
            )

    def require_lastfm(self) -> None:
        if not self.lastfm_api_key:
            raise RuntimeError(
                "LASTFM_API_KEY is not set. Get a free key at "
                "https://www.last.fm/api/account/create and add it to .env. See SETUP.md."
            )
