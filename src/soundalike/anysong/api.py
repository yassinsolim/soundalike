"""Any-song HTTP API: Deezer search plus sound-alike ranking for any Deezer track.

    python -m soundalike.anysong.api --index /opt/soundalike/index/full --meta /opt/soundalike/index/catalog_meta.npz

GET endpoints (JSON):

- ``/healthz``
- ``/api/anysong/search?q=<text>&limit=10``  Deezer track search; ``in_catalog`` marks indexed tracks
- ``/api/anysong/similar?id=<deezer id>&k=20&dedupe=1``  top matches per channel, plus ``best``: the melody and
  samples lists fused into one ranking (``fuse``)

Indexed tracks are ranked from their stored windows. Any other track has its 30 s preview fetched and
embedded live on the GPU, so it never has to be in the catalog. Standard library HTTP only: bind to
localhost and put a reverse proxy in front for TLS and public exposure.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import re
import shutil
import sys
import tempfile
import threading
import time
import traceback
from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterator
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import numpy as np

FETCH = 200  # per-channel depth ranked before de-duplication
K_MAX = 50
# "best" list: reciprocal-rank fusion of the channels' song lists, score = sum(w / (FUSE_K + rank)).
# Chosen on development splits only; see docs/ANYSONG_EVAL.md.
FUSE_K = 5
FUSE_WEIGHTS = {"melody": 1.0, "samples": 1.0}
_PARENS = re.compile(r"[\(\[][^\)\]]*[\)\]]")
_DASH_SUFFIX = re.compile(r"\s+-\s+.*$")
_FEAT = re.compile(r"\s(?:feat\.?|ft\.?|featuring)\s.*$")
_ARTIST_SPLIT = re.compile(r",|&|\s(?:x|feat\.?|ft\.?|featuring|and|with)\s")
_NON_WORD = re.compile(r"[\W_]+")


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def _norm(s: str) -> str:
    return _NON_WORD.sub(" ", s).strip()


def song_key(artist: str, title: str) -> tuple[str, str]:
    """Identity of a song across releases: primary artist plus title without version tags."""
    t = title.casefold()
    core = _norm(_FEAT.sub("", _DASH_SUFFIX.sub("", _PARENS.sub(" ", t))))
    a = _norm(_ARTIST_SPLIT.split(artist.casefold())[0])
    return a, core or _norm(t)


def fuse(orders: dict[str, list[Hashable]], weights: dict[str, float] | None = None, k: int = FUSE_K
         ) -> list[tuple[Hashable, float, dict[str, int]]]:
    """Reciprocal-rank fusion of per-channel orders of distinct keys.

    Returns ``(key, score, {channel: 1-based rank})`` best first; ties go to the better single-channel rank,
    then to first appearance (channels in ``orders`` order).
    """
    weights = FUSE_WEIGHTS if weights is None else weights
    score: dict[Hashable, float] = {}
    ranks: dict[Hashable, dict[str, int]] = {}
    for ch, order in orders.items():
        w = weights.get(ch, 0.0)
        if not w:
            continue
        for r, key in enumerate(order, 1):
            score[key] = score.get(key, 0.0) + w / (k + r)
            ranks.setdefault(key, {})[ch] = r
    return sorted(((key, s, ranks[key]) for key, s in score.items()), key=lambda x: (-x[1], min(x[2].values())))


class LRU:
    """Thread-safe LRU map with an optional time-to-live."""

    def __init__(self, size: int, ttl: float | None = None, clock: Callable[[], float] = time.monotonic):
        self.size, self.ttl, self.clock = size, ttl, clock
        self._data: OrderedDict[Hashable, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._data)

    def get(self, key: Hashable) -> Any:
        with self._lock:
            hit = self._data.get(key)
            if hit is None:
                return None
            if self.ttl is not None and self.clock() - hit[0] > self.ttl:
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return hit[1]

    def put(self, key: Hashable, value: Any) -> None:
        with self._lock:
            self._data[key] = (self.clock(), value)
            self._data.move_to_end(key)
            while len(self._data) > self.size:
                self._data.popitem(last=False)


def client_key(address: str) -> str | None:
    """Rate-limit key for a client address, or None if it is not an IP.

    IPv6 clients are keyed by their /64: one subscriber usually controls a whole /64, so per-address
    buckets would let a single client rotate through unlimited fresh budgets.
    """
    try:
        ip = ipaddress.ip_address(address.strip())
    except ValueError:
        return None
    if ip.version == 6:
        if ip.ipv4_mapped:
            return str(ip.ipv4_mapped)
        return str(ipaddress.IPv6Network((ip, 64), strict=False))
    return str(ip)


class TokenBucket:
    """Per-client request budget: ``burst`` requests at once, refilled at ``per_min`` per minute."""

    def __init__(self, per_min: float, burst: int, clock: Callable[[], float] = time.monotonic,
                 max_clients: int = 10_000):
        if per_min <= 0 or burst < 1:
            raise ValueError("rate and burst must be positive")
        self.rate, self.burst, self.clock, self.max_clients = per_min / 60.0, float(burst), clock, max_clients
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, client: str) -> bool:
        with self._lock:
            now = self.clock()
            if len(self._buckets) >= self.max_clients:
                self._prune(now)
            tokens, last = self._buckets.get(client, (self.burst, now))
            tokens = min(self.burst, tokens + (now - last) * self.rate)
            ok = tokens >= 1.0
            self._buckets[client] = (tokens - 1.0 if ok else tokens, now)
            return ok

    def _prune(self, now: float) -> None:
        full = [c for c, (tok, last) in self._buckets.items() if tok + (now - last) * self.rate >= self.burst]
        for c in full:
            del self._buckets[c]
        if len(self._buckets) >= self.max_clients:  # everyone is active: forget the stalest half
            for c in sorted(self._buckets, key=lambda c: self._buckets[c][1])[: len(self._buckets) // 2]:
                del self._buckets[c]


def _card(track: dict) -> dict:
    album, artist = track.get("album") or {}, track.get("artist") or {}
    return {"id": track["id"], "title": track.get("title") or "", "artist": artist.get("name") or "",
            "album": album.get("title") or "", "cover": album.get("cover_medium") or "",
            "duration": track.get("duration"), "has_preview": bool(track.get("preview"))}


class AnySong:
    """Request logic shared by HTTP threads. One GPU lock serialises the (non-thread-safe) engine."""

    def __init__(self, engine, deezer, meta: dict[int, tuple[str, str]], cache_size: int = 2048,
                 max_pending: int = 8, search_ttl: float = 3600.0, keep_previews: bool = False,
                 follower_timeout: float = 180.0, known_ttl: float = 600.0):
        self.engine, self.deezer, self.meta, self.keep_previews = engine, deezer, meta, keep_previews
        self.gpu = threading.Lock()
        self.pending = threading.BoundedSemaphore(max_pending)
        self.results = LRU(cache_size)
        self.searches = LRU(cache_size, ttl=search_ttl)
        # Track objects from recent searches carry signed preview URLs, so a click on a search result
        # skips the Deezer track lookup. Kept briefly because the URLs expire.
        self.known = LRU(cache_size * 4, ttl=known_ttl)
        self.follower_timeout = follower_timeout
        self._flight: dict[int, Future] = {}
        self._lock = threading.Lock()
        self.served = {"catalog": 0, "live": 0, "cached": 0}

    def _count(self, kind: str) -> None:
        with self._lock:
            self.served[kind] += 1

    def health(self) -> dict:
        with self._lock:
            served = dict(self.served)
        return {"ok": True, "device": self.engine.device,
                "channels": {name: len(ix) for name, ix in self.engine.indexes.items()},
                "cached_results": len(self.results), "served": served}

    def search(self, q: str, limit: int = 10) -> dict:
        q = " ".join(q.split())
        if not q or len(q) > 200:
            raise ApiError(400, "q must be 1-200 characters")
        key = (q.casefold(), limit)
        cards = self.searches.get(key)
        if cards is None:
            try:
                found = self.deezer.search(q, limit)
            except RuntimeError as exc:
                raise ApiError(502, "track search is unavailable") from exc
            found = [t for t in found if isinstance(t, dict) and type(t.get("id")) is int]
            for t in found:
                self.known.put(t["id"], t)
            cards = [_card(t) for t in found]
            self.searches.put(key, cards)
        return {"query": q, "tracks": [c | {"in_catalog": self.engine.indexed(c["id"])} for c in cards]}

    def cached(self, track_id: int) -> bool:
        return self.results.get(track_id) is not None

    def similar(self, track_id: int, k: int = 20, dedupe: bool = True) -> dict:
        res = self.results.get(track_id)
        if res is not None:
            self._count("cached")
            return self._view(res, k, dedupe, cached=True)
        with self._lock:
            fut = self._flight.get(track_id)
            owner = fut is None
            if owner:
                fut = self._flight[track_id] = Future()
        if not owner:  # the same track is already being ranked: share that result
            try:
                res = fut.result(timeout=self.follower_timeout)
            except FutureTimeout:
                raise ApiError(504, "timed out waiting for the result") from None
            self._count("cached")
            return self._view(res, k, dedupe, cached=True)
        try:
            res = self._compute(track_id)
            self.results.put(track_id, res)
            fut.set_result(res)
        except BaseException as exc:
            fut.set_exception(exc)
            raise
        finally:
            with self._lock:
                self._flight.pop(track_id, None)
        return self._view(res, k, dedupe, cached=False)

    def _compute(self, track_id: int) -> dict:
        if not self.pending.acquire(blocking=False):
            raise ApiError(503, "busy, retry shortly")
        try:
            t0 = time.perf_counter()
            if self.engine.indexed(track_id):
                artist, title = self.meta.get(track_id, ("", ""))
                query = {"id": track_id, "artist": artist, "title": title, "source": "catalog"}
                timings: dict[str, float] = {}
                with self._gpu(timings):
                    raw, t = self.engine.search_stored(track_id, FETCH)
            else:
                query, raw, timings, t = self._live(track_id)
            self._count(query["source"])
            timings |= t
            timings["total"] = (time.perf_counter() - t0) * 1000
            return {"query": query, "raw": raw, "timings_ms": {k: round(v, 1) for k, v in timings.items()}}
        finally:
            self.pending.release()

    @contextmanager
    def _gpu(self, timings: dict[str, float]) -> Iterator[None]:
        t = time.perf_counter()
        with self.gpu:
            timings["queue"] = (time.perf_counter() - t) * 1000
            yield

    def _live(self, track_id: int):
        timings: dict[str, float] = {}
        info = self.known.get(track_id)
        if info is None:
            t = time.perf_counter()
            try:
                info = self.deezer.track(track_id)
            except RuntimeError as exc:
                raise ApiError(502, "track lookup failed") from exc
            timings["lookup"] = (time.perf_counter() - t) * 1000
        if not info or info.get("id") is None:
            raise ApiError(404, "unknown track")
        card = _card(info)
        query = {"id": track_id, "artist": card["artist"], "title": card["title"], "cover": card["cover"],
                 "source": "live"}
        t = time.perf_counter()
        try:
            path = self.deezer.fetch_preview(track_id, info)
        except RuntimeError as exc:
            raise ApiError(502, "preview download failed") from exc
        if path is None:
            raise ApiError(404, "no preview available for this track")
        timings["fetch"] = (time.perf_counter() - t) * 1000
        try:
            clips, t_decode = self.engine.decode(path)
        except Exception as exc:
            raise ApiError(422, "could not decode the preview") from exc
        finally:
            if not self.keep_previews:
                Path(path).unlink(missing_ok=True)
        timings |= t_decode
        with self._gpu(timings):
            wins, t_embed = self.engine.embed_clips(clips)
            raw, t_search = self.engine.search_windows(wins, FETCH, [track_id])
        return query, raw, timings | t_embed, t_search

    def _view(self, res: dict, k: int, dedupe: bool, cached: bool) -> dict:
        q = res["query"]
        qkey = song_key(q["artist"], q["title"]) if q.get("title") else None
        out: dict[str, list[dict]] = {}
        songs: dict[str, dict[Hashable, dict]] = {}
        for channel, rows in res["raw"].items():
            order: dict[Hashable, dict] = {}
            for tid, score in rows:
                artist, title = self.meta.get(tid, (None, None))
                key: Hashable = ("id", tid)
                if dedupe and title is not None:
                    key = song_key(artist, title)
                    if key == qkey or key in order:
                        continue
                order[key] = {"id": tid, "score": round(float(score), 4), "artist": artist, "title": title}
            songs[channel] = order
            out[channel] = list(order.values())[:k]
        best = []
        for key, score, ranks in fuse({ch: list(order) for ch, order in songs.items()})[:k]:
            # shown as the release the strongest channel found
            ch = min(ranks, key=lambda c: ranks[c])
            best.append(songs[ch][key] | {"score": round(score, 5), "ranks": ranks})
        out = {"best": best, **out} if best else out
        return {"ok": True, "query": q, "results": out, "timings_ms": res["timings_ms"], "cached": cached}


def _int_param(params: dict[str, list[str]], name: str, default: int | None, lo: int, hi: int) -> int:
    vals = params.get(name)
    if not vals:
        if default is None:
            raise ApiError(400, f"{name} is required")
        return default
    try:
        v = int(vals[0])
    except ValueError:
        raise ApiError(400, f"{name} must be an integer") from None
    if not lo <= v <= hi:
        raise ApiError(400, f"{name} must be between {lo} and {hi}")
    return v


class Handler(BaseHTTPRequestHandler):
    server: Server
    server_version = "anysong"
    sys_version = ""
    timeout = 30

    def log_message(self, format, *args):
        print(f"{self._client()} - {format % args}", flush=True)

    def _client(self) -> str:
        header = self.server.client_header
        forwarded = client_key(self.headers.get(header, "")) if header else None
        return forwarded or client_key(self.client_address[0]) or self.client_address[0]

    def _send(self, code: int, obj: dict, max_age: int = 0) -> None:
        body = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", f"public, max-age={max_age}" if max_age else "no-store")
        if self.server.cors_origin:
            self.send_header("Access-Control-Allow-Origin", self.server.cors_origin)
            self.send_header("Vary", "Origin")
        self.end_headers()
        self.wfile.write(body)

    def _limit(self, path: str) -> None:
        if not self.server.limiters[path].allow(self._client()):
            raise ApiError(429, "too many requests")

    def do_GET(self):
        url = urlsplit(self.path)
        svc = self.server.service
        try:
            params = parse_qs(url.query, max_num_fields=8)
        except ValueError:
            return self._send(400, {"ok": False, "error": "bad query string"})
        try:
            if url.path == "/healthz":
                return self._send(200, svc.health())
            if url.path == "/api/anysong/search":
                self._limit(url.path)
                q = (params.get("q") or [""])[0]
                return self._send(200, svc.search(q, _int_param(params, "limit", 10, 1, 25)), max_age=300)
            if url.path == "/api/anysong/similar":
                tid = _int_param(params, "id", None, 1, 2**53)
                k = _int_param(params, "k", 20, 1, K_MAX)
                dedupe = _int_param(params, "dedupe", 1, 0, 1) == 1
                if not svc.cached(tid):  # only new rankings cost GPU time and Deezer requests
                    self._limit(url.path)
                return self._send(200, svc.similar(tid, k, dedupe), max_age=3600)
            raise ApiError(404, "not found")
        except ApiError as exc:
            self._send(exc.status, {"ok": False, "error": exc.message})
        except Exception:
            traceback.print_exc()
            self._send(500, {"ok": False, "error": "internal error"})


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, service: AnySong, search_per_min: float = 60, similar_per_min: float = 20,
                 cors_origin: str | None = None, client_header: str | None = None):
        super().__init__(addr, Handler)
        self.service, self.cors_origin, self.client_header = service, cors_origin, client_header
        self.limiters = {"/api/anysong/search": TokenBucket(search_per_min, max(1, int(search_per_min // 4))),
                         "/api/anysong/similar": TokenBucket(similar_per_min, max(1, int(similar_per_min // 4)))}


def _warmup(engine) -> float:
    """One synthetic query so CUDA kernels and allocator pools are ready before real traffic."""
    from .embedders import REGISTRY

    t = time.perf_counter()
    rng = np.random.default_rng(0)
    rates = sorted({REGISTRY[ch.model].sr for ch in engine.channels})
    clips = {sr: (rng.standard_normal(sr * 30) * 0.05).astype(np.float32) for sr in rates}
    engine.search_windows(engine.embed_clips(clips)[0], FETCH)
    return time.perf_counter() - t


def main(argv: list[str] | None = None) -> None:
    import torch

    from .deezer import Deezer
    from .serve import CHANNELS, Engine, load_meta

    p = argparse.ArgumentParser(prog="anysong-api")
    p.add_argument("--index", required=True, help="index_build output or a shard directory (see serve)")
    p.add_argument("--meta", type=Path, default=None, help="catalog_meta npz with track_ids/artists/titles")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8790)
    p.add_argument("--channels", nargs="+", default=list(CHANNELS), choices=list(CHANNELS))
    p.add_argument("--candidates", type=int, default=None)
    p.add_argument("--hub-alpha", type=float, default=None,
                   help="CSLS hub penalty weight (default: as recorded by hub_build; 0 disables)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--preview-dir", type=Path, default=None, help="scratch space for live previews")
    p.add_argument("--cors-origin", default=None)
    p.add_argument("--client-ip-header", default=None,
                   help="trusted proxy header with the client address, e.g. CF-Connecting-IP")
    p.add_argument("--search-per-min", type=float, default=60)
    p.add_argument("--similar-per-min", type=float, default=20)
    p.add_argument("--max-pending", type=int, default=8)
    p.add_argument("--cache-size", type=int, default=4096)
    args = p.parse_args(argv)

    scratch = args.preview_dir or Path(tempfile.mkdtemp(prefix="anysong-"))
    t = time.perf_counter()
    engine = Engine(args.index, args.channels, args.device, candidates=args.candidates, hub_alpha=args.hub_alpha)
    engine.warm()
    warm = _warmup(engine)
    service = AnySong(engine, Deezer(scratch), load_meta(args.meta), args.cache_size, args.max_pending)
    server = Server((args.host, args.port), service, args.search_per_min, args.similar_per_min,
                    args.cors_origin, args.client_ip_header)
    sizes = {name: len(ix) for name, ix in engine.indexes.items()}
    hubs = {name: getattr(ix, "hub_alpha", 0.0) for name, ix in engine.indexes.items()}
    print(f"ready in {time.perf_counter() - t:.1f}s (warmup {warm:.1f}s) {sizes} hub_alpha={hubs} "
          f"on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        if args.preview_dir is None:
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
