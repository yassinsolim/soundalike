"""Vercel serverless function: GET /api/search?q=...

Substring search over the 272,853-track library — powers the seed picker /
autocomplete in the web UI. Returns up to `limit` {row, title, artist}.
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _reco import get_recommender


class handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        params = parse_qs(urlparse(self.path).query)
        q = (params.get("q", [""])[0]).strip()
        if not q:
            return self._send(200, {"ok": True, "results": []})
        try:
            reco = get_recommender()
            hits = reco.search(q, limit=int(params.get("limit", ["8"])[0]))
            self._send(200, {"ok": True, "results": hits})
        except Exception as e:
            self._send(500, {"ok": False, "error": f"{type(e).__name__}: {e}"})
