"""Vercel serverless function: GET /api/stats

Lightweight library metadata for the UI (so copy like the song count stays
correct across index rebuilds instead of being hard-coded). Returns the current
library size and index version.
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _reco import get_recommender, _INDEX_VERSION


class handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "public, max-age=300")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        try:
            reco = get_recommender()
            self._send(200, {"ok": True, "library_size": len(reco),
                             "version": _INDEX_VERSION})
        except Exception as e:
            self._send(500, {"ok": False, "error": f"{type(e).__name__}: {e}"})
