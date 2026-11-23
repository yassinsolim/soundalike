"""Local demo page for matching a part of a song (not deployed, not linked from the site).

    python demo/part-match/serve.py [port]        # http://127.0.0.1:8797/

Serves index.html and /preview?id=<deezer track id>, which relays the song's 30 s Deezer preview so the page can draw
its waveform (Deezer's preview host sends no CORS header). Nothing is stored. The page talks to a local engine started
with --parts --excerpts (default http://127.0.0.1:8796, or ?api=<url>).
"""
import json
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

PAGE = Path(__file__).with_name("index.html")


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "soundalike-demo"})
    with urllib.request.urlopen(request, timeout=20) as r:
        return r.read()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code: int, body: bytes, kind: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlsplit(self.path)
        if url.path in ("/", "/index.html"):
            return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        if url.path == "/preview":
            tid = (parse_qs(url.query).get("id") or [""])[0]
            if not tid.isdigit():
                return self._send(400, b"bad id", "text/plain")
            try:
                preview = json.loads(fetch(f"https://api.deezer.com/track/{tid}")).get("preview")
                if not preview:
                    return self._send(404, b"no preview", "text/plain")
                return self._send(200, fetch(preview), "audio/mpeg")
            except Exception:  # noqa: BLE001 - a demo relay: any upstream failure is a 502
                return self._send(502, b"deezer", "text/plain")
        self._send(404, b"not found", "text/plain")


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8797
    print(f"part-match demo at http://127.0.0.1:{port}/", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
