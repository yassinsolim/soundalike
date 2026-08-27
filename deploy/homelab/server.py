"""Always-on HTTP entry point for the hosted recommendation API."""

from __future__ import annotations

import os
import sys
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
API_DIR = ROOT / "webapp" / "api"
sys.path.insert(0, str(API_DIR))

from _reco import get_recommender  # noqa: E402
from spicetify_recommend import handler as RecommendationHandler  # noqa: E402


class Handler(RecommendationHandler):
    def log_message(self, format, *args):
        print(f"{self.address_string()} - {format % args}", flush=True)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/healthz":
            return self._send(200, {"ok": True}, cacheable=False)
        if path == "/api/spicetify_recommend":
            return super().do_GET()
        return self._send(404, {"ok": False, "error": "not found"}, cacheable=False)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    host = os.environ.get("SOUNDALIKE_HOST", "127.0.0.1")
    port = int(os.environ.get("SOUNDALIKE_PORT", "8788"))
    if (host, port) != ("127.0.0.1", 8788):
        raise ValueError("the homelab tunnel contract requires 127.0.0.1:8788")

    recommender = get_recommender()
    print(
        f"Loaded {len(recommender):,} tracks; listening on http://{host}:{port}",
        flush=True,
    )
    Server((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
