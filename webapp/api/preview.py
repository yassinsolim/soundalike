"""Vercel serverless function: GET /api/preview?id=<deezer_track_id>

Returns a fresh 30-second preview URL for a track. Deezer preview URLs are signed
and expire, so we can't bake them into the index — we fetch one on demand by
track id. The browser then plays it directly from Deezer's CDN in an <audio>
element (media playback doesn't need CORS).
"""

import json
import urllib.request
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse


class handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "public, max-age=600")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        params = parse_qs(urlparse(self.path).query)
        tid = (params.get("id", [""])[0]).strip()
        if not tid.isdigit():
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
            self._send(200, {"ok": True, "preview": preview, "cover": cover})
        except Exception as e:
            self._send(502, {"ok": False, "error": f"{type(e).__name__}"})
