"""Vercel serverless function: GET /api/preview?id=<deezer_track_id>

Returns a fresh 30-second preview URL for a track. Deezer preview URLs are signed
and expire, so we can't bake them into the index — we fetch one on demand by
track id. The browser then plays it directly from Deezer's CDN in an <audio>
element (media playback doesn't need CORS).
"""

import json
import re
import urllib.request
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse


PRODUCTION_ORIGIN = "https://soundalike.yassin.app"
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
MAX_SAFE_TRACK_ID = (1 << 53) - 1


def _allowed_cors_origin(origin):
    """Return a narrowly allowed browser origin, or None."""
    if (
        not isinstance(origin, str)
        or not origin
        or any(ord(character) < 32 or ord(character) == 127 for character in origin)
    ):
        return None
    if origin == PRODUCTION_ORIGIN:
        return origin
    try:
        parsed = urlparse(origin)
        port = parsed.port
    except ValueError:
        return None
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname not in LOOPBACK_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path
        or parsed.params
        or parsed.query
        or parsed.fragment
        or not parsed.netloc
        or (port is not None and not 1 <= port <= 65535)
    ):
        return None
    return origin


def _valid_track_id(value):
    return (
        isinstance(value, str)
        and re.fullmatch(r"[1-9][0-9]{0,15}", value) is not None
        and int(value) <= MAX_SAFE_TRACK_ID
    )


def _trusted_dzcdn_url(value):
    """Only permit HTTPS media URLs on dzcdn.net or its subdomains."""
    if (
        not isinstance(value, str)
        or not value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        return False
    try:
        parsed = urlparse(value)
        port = parsed.port
    except ValueError:
        return False
    hostname = (parsed.hostname or "").lower()
    return (
        parsed.scheme == "https"
        and (hostname == "dzcdn.net" or hostname.endswith(".dzcdn.net"))
        and parsed.username is None
        and parsed.password is None
        and port in {None, 443}
    )


class handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        origin = _allowed_cors_origin(self.headers.get("Origin"))
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Vary", "Origin")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        origin = self.headers.get("Origin")
        if origin and not _allowed_cors_origin(origin):
            return self._send(403, {"ok": False, "error": "origin not allowed"})
        params = parse_qs(urlparse(self.path).query)
        tid = params.get("id", [""])[0]
        if not _valid_track_id(tid):
            return self._send(400, {"ok": False, "error": "bad id"})
        try:
            with urllib.request.urlopen(
                f"https://api.deezer.com/track/{tid}", timeout=15
            ) as r:
                data = json.loads(r.read().decode("utf-8"))
            preview = data.get("preview") or ""
            cover = ((data.get("album") or {}).get("cover_medium")) or ""
            if not preview:
                return self._send(404, {"ok": False, "error": "no preview"})
            if not _trusted_dzcdn_url(preview):
                return self._send(502, {"ok": False, "error": "untrusted preview"})
            if cover and not _trusted_dzcdn_url(cover):
                cover = ""
            self._send(200, {"ok": True, "preview": preview, "cover": cover})
        except Exception as e:
            self._send(502, {"ok": False, "error": f"{type(e).__name__}"})
