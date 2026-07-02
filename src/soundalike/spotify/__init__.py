"""Live Spotify integration (OAuth PKCE + Web API client)."""

from .auth import SpotifyAuth, Token, DEFAULT_SCOPES
from .client import SpotifyClient, SpotifyAPIError

__all__ = ["SpotifyAuth", "SpotifyClient", "SpotifyAPIError", "Token", "DEFAULT_SCOPES"]
