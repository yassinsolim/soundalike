"""Vercel serverless function: POST /api/recommend

Body: {"query": "Title - Artist"}  or  {"row": <int>}  plus optional
      {"n", "alpha", "diversity", "max_per_artist"}.

Returns library-mode soundalikes (numpy only, no torch). Only songs already in
the 87k library can be seeded here; the desktop app (`soundalike serve`) handles
arbitrary songs via on-the-fly neural embedding.
"""

import json
from http.server import BaseHTTPRequestHandler

from _reco import get_recommender


def _split(q):
    q = (q or "").strip()
    for sep in (" — ", " – ", " :: ", " - ", " by "):
        if sep in q:
            a, b = q.split(sep, 1)
            return a.strip(), b.strip()
    return q, ""


class handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", "0") or "0")
            data = json.loads(self.rfile.read(n).decode("utf-8")) if n > 0 else {}
        except (ValueError, TypeError):
            return self._send(400, {"ok": False, "error": "bad JSON body"})
        try:
            reco = get_recommender()
            row = data.get("row")
            if row is None:
                title, artist = _split(data.get("query", ""))
                if not title:
                    return self._send(400, {"ok": False, "error": "empty query"})
                row = reco.find_row(title, artist)
                if row is None:
                    return self._send(422, {
                        "ok": False,
                        "error": f"“{title}” isn't in the hosted library. "
                                 "The hosted demo covers 87k songs; for anything else, "
                                 "run the desktop app (soundalike serve).",
                    })
            res = reco.recommend(
                int(row), n=int(data.get("n", 20)),
                alpha=float(data["alpha"]) if data.get("alpha") is not None else None,
                diversity=float(data.get("diversity", 0.15)),
                max_per_artist=int(data.get("max_per_artist", 1)))
            self._send(200, res)
        except Exception as e:
            self._send(500, {"ok": False, "error": f"{type(e).__name__}: {e}"})
