"""Spotify OAuth 2.0 with PKCE (no client secret, no password).

Flow:
  1. Generate a code verifier/challenge pair (PKCE).
  2. Open the Spotify consent page in the user's browser.
  3. Catch the redirect on a tiny local HTTP server to read the auth `code`.
  4. Exchange the code for access + refresh tokens.
  5. Cache tokens locally and transparently refresh them when they expire.

This is the correct, ToS-compliant way to access a user's own Spotify data —
we never see or store the user's password.
"""

from __future__ import annotations

import base64
import hashlib
import json
import secrets
import threading
import time
import urllib.parse
import webbrowser
from dataclasses import dataclass, asdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import List, Optional

import requests

from ..config import Config, cache_dir

AUTHORIZE_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"

DEFAULT_SCOPES: List[str] = [
    "user-library-read",
    "user-top-read",
    "user-read-recently-played",
    "playlist-modify-public",
    "playlist-modify-private",
]


def generate_code_verifier() -> str:
    return base64.urlsafe_b64encode(secrets.token_bytes(64)).decode("ascii").rstrip("=")


def code_challenge(verifier: str) -> str:
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def build_authorize_url(
    client_id: str,
    redirect_uri: str,
    scopes: List[str],
    state: str,
    challenge: str,
) -> str:
    query = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri,
        "scope": " ".join(scopes),
        "state": state,
        "code_challenge_method": "S256",
        "code_challenge": challenge,
    }
    return f"{AUTHORIZE_URL}?{urllib.parse.urlencode(query)}"


@dataclass
class Token:
    access_token: str
    refresh_token: Optional[str]
    expires_at: float
    scope: str = ""
    token_type: str = "Bearer"

    def expired(self, skew_seconds: int = 60) -> bool:
        return time.time() >= (self.expires_at - skew_seconds)

    @classmethod
    def from_response(cls, data: dict, previous: Optional["Token"] = None) -> "Token":
        return cls(
            access_token=data["access_token"],
            # Spotify omits refresh_token on some refreshes; keep the previous one.
            refresh_token=data.get("refresh_token")
            or (previous.refresh_token if previous else None),
            expires_at=time.time() + float(data.get("expires_in", 3600)),
            scope=data.get("scope", previous.scope if previous else ""),
            token_type=data.get("token_type", "Bearer"),
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "Token":
        return cls(**json.loads(text))


class _CallbackHandler(BaseHTTPRequestHandler):
    result: dict = {}

    def do_GET(self):  # noqa: N802 (http.server API)
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)
        _CallbackHandler.result = {k: v[0] for k, v in params.items()}
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        ok = "code" in _CallbackHandler.result
        body = (
            "<html><body style='font-family:sans-serif'>"
            f"<h2>{'Authorization complete.' if ok else 'Authorization failed.'}</h2>"
            "<p>You can close this tab and return to the terminal.</p>"
            "</body></html>"
        )
        self.wfile.write(body.encode("utf-8"))

    def log_message(self, *args):  # silence the default stderr logging
        return


def _wait_for_callback(redirect_uri: str, expected_state: str, timeout: int = 300) -> str:
    parsed = urllib.parse.urlparse(redirect_uri)
    host, port = parsed.hostname or "127.0.0.1", parsed.port or 80
    _CallbackHandler.result = {}
    server = HTTPServer((host, port), _CallbackHandler)
    server.timeout = timeout

    thread = threading.Thread(target=server.handle_request)
    thread.start()
    thread.join(timeout)
    server.server_close()

    result = _CallbackHandler.result
    if not result:
        raise TimeoutError("Timed out waiting for the Spotify authorization redirect.")
    if "error" in result:
        raise RuntimeError(f"Spotify authorization error: {result['error']}")
    if result.get("state") != expected_state:
        raise RuntimeError("State mismatch in OAuth callback (possible CSRF).")
    return result["code"]


class SpotifyAuth:
    """Manages the PKCE flow and a cached token."""

    def __init__(
        self,
        config: Config,
        scopes: Optional[List[str]] = None,
        token_path: Optional[Path] = None,
    ):
        config.require_spotify()
        self.config = config
        self.scopes = scopes or DEFAULT_SCOPES
        self.token_path = token_path or (cache_dir() / "spotify_token.json")

    # ------------------------------------------------------------- token cache
    def load_cached_token(self) -> Optional[Token]:
        if self.token_path.exists():
            try:
                return Token.from_json(self.token_path.read_text(encoding="utf-8"))
            except (ValueError, KeyError):
                return None
        return None

    def save_token(self, token: Token) -> None:
        self.token_path.write_text(token.to_json(), encoding="utf-8")

    # ------------------------------------------------------------------- flow
    def authorize_interactive(self, open_browser: bool = True) -> Token:
        verifier = generate_code_verifier()
        challenge = code_challenge(verifier)
        state = secrets.token_urlsafe(16)
        url = build_authorize_url(
            self.config.spotify_client_id,
            self.config.spotify_redirect_uri,
            self.scopes,
            state,
            challenge,
        )
        print("Opening Spotify authorization in your browser...")
        print(f"If it doesn't open, visit:\n{url}\n")
        if open_browser:
            webbrowser.open(url)

        code = _wait_for_callback(self.config.spotify_redirect_uri, state)
        token = self._exchange_code(code, verifier)
        self.save_token(token)
        return token

    def _exchange_code(self, code: str, verifier: str) -> Token:
        data = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.config.spotify_redirect_uri,
            "client_id": self.config.spotify_client_id,
            "code_verifier": verifier,
        }
        response = requests.post(TOKEN_URL, data=data, timeout=30)
        response.raise_for_status()
        return Token.from_response(response.json())

    def refresh(self, token: Token) -> Token:
        if not token.refresh_token:
            raise RuntimeError("No refresh token available; re-run `soundalike login`.")
        data = {
            "grant_type": "refresh_token",
            "refresh_token": token.refresh_token,
            "client_id": self.config.spotify_client_id,
        }
        response = requests.post(TOKEN_URL, data=data, timeout=30)
        response.raise_for_status()
        refreshed = Token.from_response(response.json(), previous=token)
        self.save_token(refreshed)
        return refreshed

    def get_valid_token(self, interactive: bool = True) -> Token:
        token = self.load_cached_token()
        if token is None:
            if not interactive:
                raise RuntimeError("Not logged in. Run `soundalike login` first.")
            return self.authorize_interactive()
        if token.expired():
            return self.refresh(token)
        return token
