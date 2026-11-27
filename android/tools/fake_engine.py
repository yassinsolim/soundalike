"""Stand-in for the Soundalike engine that always fails with one status, for testing the app's error states.

    python android/tools/fake_engine.py 429      # every request: 429 with Retry-After: 42
    python android/tools/fake_engine.py 503      # every request: 503

Listens on 127.0.0.1:8799. In a debug build set Settings > Debug build to http://10.0.2.2:8799.
"""
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

status = int(sys.argv[1]) if len(sys.argv) > 1 else 429


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b'{"ok":false,"error":"test"}'
        self.send_response(status)
        if status == 429:
            self.send_header("Retry-After", "42")
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


ThreadingHTTPServer(("127.0.0.1", 8799), Handler).serve_forever()
