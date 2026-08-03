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
from _search import _INDEX_VERSION, get_search_catalog


class handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header(
            "Cache-Control",
            "public, max-age=300, s-maxage=86400, stale-while-revalidate=604800",
        )
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        params = parse_qs(urlparse(self.path).query)
        q = (params.get("q", [""])[0]).strip()
        if not q:
            return self._send(
                200, {"ok": True, "version": _INDEX_VERSION, "results": []}
            )
        if len(q) > 120:
            return self._send(
                400, {"ok": False, "error": "query is too long"}
            )
        try:
            limit = max(1, min(int(params.get("limit", ["8"])[0]), 20))
        except ValueError:
            return self._send(400, {"ok": False, "error": "invalid limit"})
        try:
            hits = get_search_catalog().search(q, limit=limit)
            self._send(
                200,
                {"ok": True, "version": _INDEX_VERSION, "results": hits},
            )
        except Exception as e:
            self._send(500, {"ok": False, "error": f"{type(e).__name__}: {e}"})
