"""Any-song HTTP API: Deezer search plus sound-alike ranking for any Deezer track.

    python -m soundalike.anysong.api --index /opt/soundalike/index/full --meta /opt/soundalike/index/catalog_meta.npz

GET endpoints (JSON):

- ``/healthz``
- ``/api/anysong/search?q=<text>&limit=10``  Deezer track search; ``in_catalog`` marks indexed tracks
- ``/api/anysong/similar?id=<deezer id>&k=20&dedupe=1``  top matches per channel, plus ``best``: the melody and
  samples lists fused into one ranking (``fuse``). With a scene table (``--scene``) the fused list also weighs the
  artist scene (same artist, related artists, region, genre; ``scene.scene_fuse``); ``scene=0`` turns that off
- ``/api/anysong/similar?title=<title>&artist=<artist>&k=20``  the same for a song named by title and artist
  (``AnySong.resolve``): the catalog release of that song, else the first matching Deezer search hit

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
import unicodedata
from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import numpy as np

from .deezer import version_words
from .scene import SceneParams, SceneTable, scene_fuse

FETCH = 200  # per-channel depth ranked before de-duplication
K_MAX = 50
# "best" list: reciprocal-rank fusion of the channels' song lists, score = sum(w / (FUSE_K + rank)).
# Chosen on development splits only; see docs/ANYSONG_EVAL.md.
FUSE_K = 5
FUSE_WEIGHTS = {"melody": 1.0, "samples": 1.0}
RESOLVE_LIMIT = 10  # Deezer search depth for a title and artist that are not in the catalog
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


def fuse(orders: Mapping[str, Sequence[Hashable]], weights: dict[str, float] | None = None, k: int = FUSE_K
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


def _text_param(value: str, name: str) -> str:
    value = " ".join(unicodedata.normalize("NFC", value).split())
    if not value or len(value) > 200:
        raise ApiError(400, f"{name} must be 1-200 characters")
    return value


def _identity(artist: str, title: str) -> tuple[str, str]:
    """``song_key`` of NFC text, so composed and decomposed accents name the same song."""
    return song_key(unicodedata.normalize("NFC", artist), unicodedata.normalize("NFC", title))


def _card(track: dict) -> dict:
    album, artist = track.get("album") or {}, track.get("artist") or {}
    return {"id": track["id"], "title": track.get("title") or "", "artist": artist.get("name") or "",
            "album": album.get("title") or "", "cover": album.get("cover_medium") or "",
            "duration": track.get("duration"), "has_preview": bool(track.get("preview"))}


class AnySong:
    """Request logic shared by HTTP threads. One GPU lock serialises the (non-thread-safe) engine."""

    def __init__(self, engine, deezer, meta: dict[int, tuple[str, str]], cache_size: int = 2048,
                 max_pending: int = 8, search_ttl: float = 3600.0, keep_previews: bool = False,
                 follower_timeout: float = 180.0, known_ttl: float = 600.0, languages=None,
                 scene: SceneTable | None = None, scene_params: SceneParams | None = None,
                 related_ttl: float = 86400.0):
        self.engine, self.deezer, self.meta, self.keep_previews = engine, deezer, meta, keep_previews
        self.languages = languages
        self.scene, self.scene_params = scene, scene_params or SceneParams()
        # Deezer related artists for query artists whose own list is not in the scene table (e.g. unseen artists).
        self.related = LRU(cache_size, ttl=related_ttl)
        self._related_jobs: dict[int, Future] = {}
        self._background = ThreadPoolExecutor(max_workers=2, thread_name_prefix="anysong-related")
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
        # Song identity -> indexed catalog releases, for clients that only know a title and artist: titles without
        # version tags first ("One More Time" before "One More Time (Unplugged)"), then low ids. Releases without
        # windows usually had no preview, so the Deezer search decides about those.
        entries = []
        for tid, (artist, title) in meta.items():
            if title and engine.indexed(tid):
                key = _identity(artist, title)
                entries.append((key[1] != _norm(unicodedata.normalize("NFC", title).casefold()), tid, key))
        releases: dict[tuple[str, str], list[int]] = {}
        for _, tid, key in sorted(entries):
            releases.setdefault(key, []).append(tid)
        self.by_key: dict[tuple[str, str], tuple[int, ...]] = {key: tuple(ids) for key, ids in releases.items()}

    def _count(self, kind: str) -> None:
        with self._lock:
            self.served[kind] += 1

    def health(self) -> dict:
        with self._lock:
            served = dict(self.served)
        scene = None if self.scene is None else {"tracks": int((self.scene.track_artist > 0).sum()),
                                                 "artists": len(self.scene.artist_ids)}
        return {"ok": True, "device": self.engine.device,
                "channels": {name: len(ix) for name, ix in self.engine.indexes.items()},
                "cached_results": len(self.results), "served": served, "scene": scene}

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

    def resolve(self, artist: str, title: str, on_search: Callable[[], None] | None = None) -> int:
        """Deezer id of the recording named by ``artist`` and ``title``.

        Songs match by ``song_key``. Among a song's releases, one with the same ``version_words`` wins, so
        "Song - Live" asks for a live recording, while "Song" or "Song - Remastered" will not be served by a remix.
        An indexed catalog release answers without any Deezer request. Otherwise ``on_search`` runs (the caller's
        search rate limit) and the first Deezer search hit with a preview and that version wins. Failing that, the
        song's best indexed release, then its first search hit, stands in.
        """
        artist, title = _text_param(artist, "artist"), _text_param(title, "title")
        key, version = _identity(artist, title), version_words(title)
        known = self.by_key.get(key, ())
        for tid in known:
            if version_words(self.meta[tid][1]) == version:
                return tid
        try:
            if on_search is not None:
                on_search()
            cards = self.search(f"{artist} {title}"[:200], RESOLVE_LIMIT)["tracks"]
        except ApiError:
            if known:
                return known[0]
            raise
        same = [c for c in cards if c["has_preview"] and _identity(c["artist"], c["title"]) == key]
        for card in same:
            if version_words(card["title"]) == version:
                return card["id"]
        if known:
            return known[0]
        if same:
            return same[0]["id"]
        raise ApiError(404, "song not found")

    def cached(self, track_id: int) -> bool:
        return self.results.get(track_id) is not None

    def similar(self, track_id: int, k: int = 20, dedupe: bool = True, lang_mode: str = "same",
                scene: bool = True) -> dict:
        res = self.results.get(track_id)
        if res is not None:
            self._count("cached")
            return self._view(res, k, dedupe, cached=True, lang_mode=lang_mode, scene=scene)
        with self._lock:
            fut = self._flight.get(track_id)
            owner = fut is None
            if owner:
                fut = self._flight[track_id] = Future()
        if not owner:
            try:
                res = fut.result(timeout=self.follower_timeout)
            except FutureTimeout:
                raise ApiError(504, "timed out waiting for the result") from None
            self._count("cached")
            return self._view(res, k, dedupe, cached=True, lang_mode=lang_mode, scene=scene)
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
        return self._view(res, k, dedupe, cached=False, lang_mode=lang_mode, scene=scene)

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
            related = self._scene_context(query, timings)
            timings["total"] = (time.perf_counter() - t0) * 1000
            return {"query": query, "raw": raw, "related": related,
                    "timings_ms": {k: round(v, 1) for k, v in timings.items()}}
        finally:
            self.pending.release()

    def _scene_context(self, query: dict, timings: dict[str, float]) -> list[int]:
        """Set the query's Deezer artist id; fetch Deezer's related artists when the scene table lacks that list.

        A live query's list is usually fetched already (``_start_related``, during the preview download). Fails open:
        without a related list the scene layer uses whatever the table knows about the artist.
        """
        if self.scene is None:
            return []
        artist_id = query.get("artist_id") or self.scene.artist(int(query["id"]))
        if not artist_id:
            return []
        query["artist_id"] = int(artist_id)
        if self.scene.has_related(artist_id):
            return []
        with self._lock:
            job = self._related_jobs.pop(artist_id, None)
        related = self.related.get(artist_id)
        if related is None:
            t = time.perf_counter()
            try:
                related = job.result(timeout=30) if job is not None else self._fetch_related(artist_id)
            except (RuntimeError, ValueError, OSError, FutureTimeout):
                return []
            finally:
                timings["related"] = (time.perf_counter() - t) * 1000  # time spent waiting for Deezer
        elif job is not None:
            timings["related"] = 0.0  # fetched during the preview download
        return list(related)

    def _fetch_related(self, artist_id: int) -> list[int]:
        related = self.deezer.related_artists(artist_id)
        self.related.put(artist_id, related)
        return related

    def _start_related(self, artist_id: int | None) -> None:
        """Start fetching a live seed artist's related list while its preview is downloaded and embedded."""
        if self.scene is None or not artist_id or self.scene.has_related(artist_id) \
                or self.related.get(artist_id) is not None:
            return
        with self._lock:
            if len(self._related_jobs) > 256:  # jobs of live queries that failed before using them
                self._related_jobs = {a: job for a, job in self._related_jobs.items() if not job.done()}
            if artist_id not in self._related_jobs:
                self._related_jobs[artist_id] = self._background.submit(self._fetch_related, artist_id)

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
        artist_id = (info.get("artist") or {}).get("id")
        if type(artist_id) is int and artist_id > 0:
            query["artist_id"] = artist_id
            self._start_related(artist_id)
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

    def _view(self, res: dict, k: int, dedupe: bool, cached: bool, lang_mode: str = "same",
              scene: bool = True) -> dict:
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

        language = "unknown"
        if self.languages is not None:
            language = self.languages.language(int(q["id"]))
            if language == "unknown":
                language = self.languages.artist_language(q.get("artist"))

        scene_state = "off"
        artists: dict[Hashable, int | None] = {}
        profile = None
        if self.scene is not None and scene:
            for order in songs.values():
                for key, item in order.items():
                    if key not in artists:
                        artists[key] = self.scene.artist(int(item["id"]))
            profile = self.scene.profile(q.get("artist_id"), res.get("related"))
            scene_state = "on" if profile is not None else "unknown_artist"

        def rank(view: dict[str, dict[Hashable, dict]]) -> list:
            orders = {channel: list(order) for channel, order in view.items()}
            if scene_state == "off":
                return fuse(orders)
            scores = {channel: [item["score"] for item in order.values()] for channel, order in view.items()}
            return scene_fuse(orders, artists, profile, self.scene, self.scene_params, scores=scores)

        all_fused = rank(songs)
        hidden = 0
        filtered = songs
        if language != "unknown" and self.languages is not None:
            def keep(item: dict) -> bool:
                candidate = self.languages.language(int(item["id"]))
                return candidate in {"unknown", language}

            hidden = sum(
                1
                for key, _, _ in all_fused[:k]
                if not keep(next(songs[channel][key] for channel in songs if key in songs[channel]))
            )
            if lang_mode == "same":
                filtered = {
                    channel: {key: item for key, item in order.items() if keep(item)}
                    for channel, order in songs.items()
                }
        best = []
        for key, score, ranks in rank(filtered)[:k]:
            channel = min(ranks, key=lambda name: ranks[name])
            best.append(filtered[channel][key] | {"score": round(score, 5), "ranks": ranks})
        query = q | {"language": language, "language_mode": lang_mode, "hidden": hidden, "scene": scene_state}
        out = {"best": best, **out} if best else out
        return {"ok": True, "query": query, "results": out, "timings_ms": res["timings_ms"], "cached": cached}


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
                k = _int_param(params, "k", 20, 1, K_MAX)
                dedupe = _int_param(params, "dedupe", 1, 0, 1) == 1
                lang_mode = (params.get("lang") or ["same"])[0]
                if lang_mode not in {"same", "all"}:
                    raise ApiError(400, "lang must be 'same' or 'all'")
                scene = _int_param(params, "scene", 1, 0, 1) == 1
                if "id" not in params and ("title" in params or "artist" in params):
                    tid = svc.resolve((params.get("artist") or [""])[0], (params.get("title") or [""])[0],
                                      lambda: self._limit("/api/anysong/search"))
                else:
                    tid = _int_param(params, "id", None, 1, 2**53)
                if not svc.cached(tid):  # only new rankings cost GPU time and Deezer requests
                    self._limit(url.path)
                return self._send(200, svc.similar(tid, k, dedupe, lang_mode, scene), max_age=3600)
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
    p.add_argument("--languages", type=Path, default=None, help="language labels from anysong.languages")
    p.add_argument("--scene", type=Path, default=None, help="scene table from anysong.scene (metadata for Best)")
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
    meta = load_meta(args.meta)
    languages = None
    if args.languages is not None:
        from .languages import LanguageTable

        languages = LanguageTable.load(
            args.languages,
            artist_names={track_id: artist for track_id, (artist, _) in meta.items()},
        )
    scene = SceneTable.load(args.scene) if args.scene is not None else None
    service = AnySong(engine, Deezer(scratch), meta, args.cache_size, args.max_pending,
                      languages=languages, scene=scene)
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
