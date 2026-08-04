"""Cacheable recommendation endpoint for the Spicetify extension."""

import json
import math
import os
import sys
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlencode, urlsplit

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _reco import get_recommender


def _tempo_bpm(recommender, row):
    try:
        index = recommender.feature_names.index("tempo")
        standardized = recommender._vscaled[row, index] / recommender._w[index]
        value = standardized * recommender._vstd[index] + recommender._vmean[index]
        bpm = float(value)
    except (AttributeError, IndexError, TypeError, ValueError, ZeroDivisionError):
        return None
    return round(bpm) if math.isfinite(bpm) and bpm > 0 else None


def _enrich_result_tempos(recommender, result):
    for item in result.get("results", []):
        row = None
        try:
            track_id = int(item["deezer_id"])
            matches = np.flatnonzero(recommender.track_ids == track_id)
            if matches.size:
                row = int(matches[0])
        except (KeyError, TypeError, ValueError):
            pass
        if row is None:
            row = recommender.find_row(item.get("title", ""), item.get("artist", ""))
        item["bpm"] = _tempo_bpm(recommender, row) if row is not None else None
    return result


def _needs_canonical_redirect(params, query, count, diversity):
    return (
        params["query"][0] != query
        or params.get("n", ["20"])[0] != str(count)
        or params.get("diversity", ["0.15"])[0] != format(diversity, "g")
    )


class handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        if code == 200:
            self.send_header(
                "Cache-Control",
                "public, max-age=0, s-maxage=86400, stale-while-revalidate=604800",
            )
        else:
            self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        request = urlsplit(self.path)
        params = parse_qs(request.query, keep_blank_values=True)
        if set(params) - {"query", "n", "diversity", "v"} or any(
            len(values) != 1 for values in params.values()
        ):
            return self._send(400, {"ok": False, "error": "invalid query parameters"})
        if params.get("v", ["2"])[0] != "2":
            return self._send(400, {"ok": False, "error": "unsupported API version"})
        query = params.get("query", [""])[0].strip()
        if not query or len(query) > 300:
            return self._send(400, {"ok": False, "error": "empty query"})
        try:
            count = int(params.get("n", ["20"])[0])
            diversity = float(params.get("diversity", ["0.15"])[0])
        except (TypeError, ValueError):
            return self._send(
                400, {"ok": False, "error": "invalid recommendation parameters"}
            )
        if not 1 <= count <= 50 or not 0 <= diversity <= 1:
            return self._send(
                400, {"ok": False, "error": "invalid recommendation parameters"}
            )
        canonical_query = urlencode({
            "query": query,
            "n": str(count),
            "diversity": format(diversity, "g"),
            "v": "2",
        })
        if _needs_canonical_redirect(params, query, count, diversity):
            return self._redirect(f"{request.path}?{canonical_query}")
        try:
            recommender = get_recommender()
            title, _, artist = query.partition(" — ")
            row = recommender.find_row(title.strip(), artist.strip())
            if row is None:
                return self._send(422, {
                    "ok": False,
                    "error": f"“{title}” isn't in the hosted library.",
                })
            result = recommender.recommend(
                row,
                n=count,
                diversity=diversity,
                max_per_artist=1,
            )
            seed_bpm = _tempo_bpm(recommender, row)
            if seed_bpm:
                result.setdefault("vibe", {})["tempo"] = f"{seed_bpm} BPM"
            self._send(200, _enrich_result_tempos(recommender, result))
        except Exception as error:
            print(f"Spicetify recommendation failed: {error}", file=sys.stderr)
            self._send(500, {
                "ok": False,
                "error": "recommendation failed",
            })

    def _redirect(self, location):
        self.send_response(307)
        self.send_header("Content-Length", "0")
        self.send_header("Location", location)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "public, max-age=86400")
        self.end_headers()
