"""Local dev server that mimics Vercel's routing for webapp/.

Serves index.html and dispatches /api/search + /api/recommend to the same
`_reco` engine the serverless functions use — so you can test the whole UI
without deploying. Not shipped to Vercel (Vercel uses api/*.py directly).

    set SOUNDALIKE_INDEX_PATH=../src/soundalike/data/deepvibe_index.npz
    python webapp/dev_server.py
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).parent / "api"))
os.environ.setdefault(
    "SOUNDALIKE_INDEX_PATH",
    str(Path(__file__).resolve().parents[1] / "src" / "soundalike" / "data" / "deepvibe_index.npz"),
)
from _reco import get_recommender  # noqa: E402

HTML = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")


def _split(q):
    for sep in (" — ", " – ", " :: ", " - ", " by "):
        if sep in q:
            a, b = q.split(sep, 1)
            return a.strip(), b.strip()
    return q.strip(), ""


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            b = HTML.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        elif u.path == "/api/search":
            q = parse_qs(u.query).get("q", [""])[0]
            self._json(200, {"ok": True, "results": get_recommender().search(q, 8) if q else []})
        elif u.path == "/api/stats":
            from _reco import _INDEX_VERSION
            reco = get_recommender()
            self._json(200, {"ok": True, "library_size": len(reco), "version": _INDEX_VERSION})
        elif u.path == "/api/preview":
            import urllib.request
            tid = parse_qs(u.query).get("id", [""])[0]
            if not tid.isdigit():
                return self._json(400, {"ok": False, "error": "bad id"})
            try:
                with urllib.request.urlopen(f"https://api.deezer.com/track/{tid}", timeout=15) as r:
                    d = json.loads(r.read().decode())
                self._json(200, {"ok": bool(d.get("preview")), "preview": d.get("preview", ""),
                                 "cover": (d.get("album") or {}).get("cover_medium", "")})
            except Exception:
                self._json(502, {"ok": False, "error": "deezer"})
        else:
            self._json(404, {"ok": False})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0") or "0")
        data = json.loads(self.rfile.read(n).decode()) if n else {}
        if self.path == "/api/recommend":
            reco = get_recommender()
            row = data.get("row")
            if row is None:
                t, a = _split(data.get("query", ""))
                row = reco.find_row(t, a)
                if row is None:
                    return self._json(422, {"ok": False, "error": "not in hosted library"})
            self._json(200, reco.recommend(int(row), n=int(data.get("n", 20)),
                                           diversity=float(data.get("diversity", 0.15))))
        else:
            self._json(404, {"ok": False})


if __name__ == "__main__":
    print("Loading index…")
    get_recommender()
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8788
    print(f"webapp dev server → http://127.0.0.1:{port}/")
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
