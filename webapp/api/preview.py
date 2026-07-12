"""Vercel serverless function: GET /api/preview?id=<deezer_track_id>

Returns a fresh 30-second preview URL for a track. Deezer preview URLs are signed
and expire, so we can't bake them into the index — we fetch one on demand by
track id. The browser then plays it directly from Deezer's CDN in an <audio>
element (media playback doesn't need CORS).
"""

import json
import urllib.request
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlsplit


def _track_id(path):
    parsed = urlsplit(path)
    params = parse_qs(parsed.query, keep_blank_values=True)
    if (
        parsed.path != "/api/preview"
        or set(params) != {"id"}
        or len(params["id"]) != 1
    ):
        return None
    value = params["id"][0]
    if not value.isdigit() or len(value) > 20 or int(value) <= 0:
        return None
    return value


class handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        tid = _track_id(self.path)
        if tid is None:
            return self._send(400, {"ok": False, "error": "bad id"})
        try:
            with urllib.request.urlopen(
                f"https://api.deezer.com/track/{tid}", timeout=15
            ) as r:
                data = json.loads(r.read().decode("utf-8"))
            preview = data.get("preview") or ""
            if not preview:
                return self._send(404, {"ok": False, "error": "no preview"})
            origin = urlsplit(preview)
            if origin.scheme != "https" or not (
                origin.hostname == "dzcdn.net"
                or (origin.hostname or "").endswith(".dzcdn.net")
            ):
                return self._send(502, {"ok": False, "error": "preview provider"})
            self._send(200, {"ok": True, "preview": preview})
        except Exception:
            self._send(502, {"ok": False, "error": "preview provider"})
