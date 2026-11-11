"""Any-song HTTP API: Deezer search plus sound-alike ranking for any Deezer track.

    python -m soundalike.anysong.api --index /opt/soundalike/index/full --meta /opt/soundalike/index/catalog_meta.npz

GET endpoints (JSON):

- ``/healthz``
- ``/api/anysong/search?q=<text>&limit=10``  Deezer track search; ``in_catalog`` marks indexed tracks
- ``/api/anysong/similar?id=<deezer id>&k=20&dedupe=1``  top matches per channel, plus ``best``: the melody and
  samples lists fused into one ranking (``fuse``). With a scene table (``--scene``) the fused list also weighs the
  artist scene (same artist, related artists, region, genre; ``scene.scene_fuse``); ``scene=0`` turns that off.
  With a tempo table (``--tempo``), ``vibe`` is the Same vibe list: Best matches that share the seed's tempo and
  sound, one song per artist (``samevibe``). ``sounds`` is the Sounds like list: the sonic channel with Best's
  language filter, one song per artist, each with its catalog ``bpm`` when known. Every listed song with a sound score
  also carries ``match``: how alike it sounds to the seed, 0 to 100, the percentile of its sound score among the songs
  these lists show for typical seeds (``sound_match``)
- ``/api/anysong/similar?title=<title>&artist=<artist>&k=20``  the same for a song named by title and artist
  (``AnySong.resolve``): the catalog release of that song, else the first matching Deezer search hit
- ``/api/anysong/similar?isrc=<ISRC>&title=<title>&artist=<artist>``  the same for the recording with that ISRC
  (``AnySong.resolve_isrc``): the catalog release of the named song, else Deezer's ISRC lookup, else ``resolve``
- ``/api/anysong/dissect?id=<deezer id>&with=<deezer id>``  the stem dissection of a song (``stems.payload``: six
  stems' loudness lanes, drum hits and pattern, vocal and bass pitch, beats, key and tempo) and, with ``with``, of a
  second song plus how alike their stems and measurements are (``stems.compare``). Rows come from the stem table
  (``--stems``) or earlier live runs; with ``--stems-model`` any other song's preview is dissected on the GPU

Indexed tracks are ranked from their stored windows. Any other track has its 30 s preview fetched and
embedded live on the GPU, so it never has to be in the catalog. Standard library HTTP only: bind to
localhost and put a reverse proxy in front for TLS and public exposure.

With ``--learn-queue``, each song answered live (so not yet in the catalog) is appended to that file as its Deezer
track id, artist id and UTC day, nothing about who asked (``LearnQueue``); ``anysong.crawl`` reads it to grow the
catalog toward what people look up.

With ``--core-ids`` (and ``--scene``), indexed songs outside the core catalog, the ones ``anysong.crawl`` added, are
only ranked for seeds by their own artist or by an artist whose Deezer related list in the scene table names theirs
(``tiers.TierGate``); every list, the channel lists included, is gated the same way.
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
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import numpy as np

from . import listener, samevibe
from . import stems as stemlib
from .deezer import version_words
from .scene import SceneParams, SceneTable, affinity, arrange_artists, audio_head, scene_fuse
from .tempo import TempoTable, decode22

FETCH = 200  # per-channel depth ranked before de-duplication
K_MAX = 50
# Another artist's version of the seed's song this high in the samples list reuses the seed's recording (a re-upload,
# a sped-up edit, a remix, the same song under another credit) rather than covering it (``AnySong._limit_versions``).
# About 3 % of benchmark covers (development split) score this high.
COPY_SAMPLES_SCORE = 0.55
MAX_VERSIONS = 3  # other artists' versions of the seed's song kept in the lists users see
SOUNDS_BOOST = 8.0  # Sounds like: lift by the artist's scene affinity
# With a listener ranker (``listener.Ranker``): Best after its audio head orders by z(listener logit) +
# BEST_SOUND_WEIGHT * z(sound score) + BEST_AUDIO_WEIGHT * z(melody/samples fusion score); Sounds like by
# z(sound score) + SOUNDS_LISTENER_WEIGHT * z(listener logit).
BEST_SOUND_WEIGHT = 0.5
BEST_AUDIO_WEIGHT = 0.25
SOUNDS_LISTENER_WEIGHT = 2.0
# With a listener ranker, Best's songs after its audio head are spread out (``scene.arrange_artists``): an artist's
# songs at least BEST_ARTIST_GAP places apart, and each artist once in the first BEST_ARTIST_ONCE places.
BEST_ARTIST_GAP = 2
BEST_ARTIST_ONCE = 10
# With a listener ranker, a melody or samples match extends Best's audio head only if it also stands out from the seed's
# other matches: a z-score of at least BEST_HEAD_STANDOUT among the first BEST_HEAD_WINDOW songs of its list (0 = off).
BEST_HEAD_STANDOUT = 2.0
BEST_HEAD_WINDOW = 20
# With a listener ranker, Best's audio head when none of its first songs has cover/sample evidence: BEST_HEAD_BARE
# songs; with BEST_HEAD_EVIDENCE_ONLY, a head with evidence holds only those songs (``scene.audio_head``).
BEST_HEAD_BARE = 1
BEST_HEAD_EVIDENCE_ONLY = True
# Sound match (``sound_match``): sonic (MuQ-MuLan) score quantiles 0, 5, ..., 100 % over every song shown in the top
# 20 of Best matches, Same vibe or Sounds like for the development seeds: 62,362 songs from 1,474 seeds with the
# served lists (grow/qa/head_dev.py calib, 2026-11-11). A typical seed's median shown song scores 0.419 (50).
MATCH_QUANTILES = (-0.4515, 0.1873, 0.2729, 0.3209, 0.3498, 0.3695, 0.3834, 0.3933, 0.4023, 0.4106, 0.4177, 0.4248,
                   0.4312, 0.4374, 0.4437, 0.4500, 0.4563, 0.4628, 0.4701, 0.4784, 0.5354)
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
_LEADING_THE = re.compile(r"^the\s+")
_BRACKETED = re.compile(r"[\(\[]([^\)\]]*)[\)\]]")
# Title parts that decorate a release rather than name other content (``_content_tags``).
_DECORATION = re.compile(
    r"^(?:feat\b|ft\b|featuring\b|with\b|prod\b|produced\b|original\b|mono\b|stereo\b|explicit\b|clean\b|"
    r"bonus\b|deluxe\b|single\b|album\b|radio\b|mixed\b|\d{4}\b)", re.I)
_DECORATION_ANY = re.compile(r"\b(?:from|soundtrack|ost|remaster(?:ed)?|motion picture|edition|anniversary)\b", re.I)
_ISRC = re.compile(r"^[A-Z]{2}[A-Z0-9]{3}\d{7}$")


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


def _fold(s: str) -> str:
    """Casefolded text without accents on Latin letters ("Beyoncé" = "Beyonce"); other scripts keep their marks."""
    out: list[str] = []
    for c in unicodedata.normalize("NFKD", s.casefold()):
        if unicodedata.combining(c) and out and out[-1].isascii():
            continue
        out.append(c)
    return unicodedata.normalize("NFC", "".join(out))


def _lookup_key(artist: str, title: str) -> tuple[str, str]:
    """``song_key`` for matching a requested title and artist to releases. Services credit artists differently
    ("The Black Eyed Peas" and "Black Eyed Peas", "Beyoncé" and "Beyonce", "††† (Crosses)" and "✝✝✝ (Crosses)"),
    so accents on Latin letters, symbols and a leading "The" are ignored. Lists keep ``song_key``."""
    a, t = song_key(_fold(artist), _fold(title))
    return _LEADING_THE.sub("", a) or a, t


def _content_tags(title: str) -> frozenset[str]:
    """Bracketed or dash-suffixed parts of a title that name different content rather than a version, a credit or a
    release decoration: the "Santa Clause is Coming to Town" of a medley, not "Live", "feat. X", "2011 Remaster" or
    "From ...". Releases of one song whose content tags differ are probably not the same piece."""
    title = unicodedata.normalize("NFKC", title)
    parts = [m.group(1) for m in _BRACKETED.finditer(title)]
    dash = _DASH_SUFFIX.search(title)
    if dash:
        parts.append(dash.group(0).split("-", 1)[1])
    tags = set()
    for part in parts:
        part = " ".join(part.split()).strip(" .")
        if part and not _DECORATION.match(part) and not _DECORATION_ANY.search(part) and not version_words(part):
            tags.add(_norm(_fold(part)))
    return frozenset(t for t in tags if t)


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


def sound_match(score: float, quantiles: Sequence[float] = MATCH_QUANTILES) -> int:
    """A song's sonic score against the seed as 0 to 100: the share of songs shown for typical seeds that it sounds
    more alike than (``MATCH_QUANTILES``)."""
    return int(round(float(np.interp(score, quantiles, np.linspace(0, 100, len(quantiles))))))


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


class LearnQueue:
    """Append-only queue of songs answered live, for growing the catalog (read by ``anysong.crawl``).

    A line is ``{"id": <Deezer track id>, "artist_id": <Deezer artist id or null>, "day": "YYYY-MM-DD"}`` (UTC). No
    address, header, title or time of day is written. Each song is written once per process and at most ``per_day``
    lines a day. A failed write never fails the request.
    """

    def __init__(self, path: Path | str, per_day: int = 5000, remember: int = 200_000,
                 clock: Callable[[], float] = time.time):
        self.path, self.per_day, self.remember, self.clock = Path(path), per_day, remember, clock
        self._seen: set[int] = set()
        self._day: str | None = None
        self._count = 0
        self._lock = threading.Lock()

    def add(self, track_id: int, artist_id: int | None) -> bool:
        day = time.strftime("%Y-%m-%d", time.gmtime(self.clock()))
        line = json.dumps({"id": int(track_id), "artist_id": int(artist_id) if artist_id else None, "day": day})
        with self._lock:
            if day != self._day:
                self._day, self._count = day, 0
            if int(track_id) in self._seen or self._count >= self.per_day:
                return False
            try:
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except OSError:
                return False
            if len(self._seen) >= self.remember:
                self._seen.clear()
            self._seen.add(int(track_id))
            self._count += 1
            return True


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


def _gpu_mib(device: str) -> dict | None:
    """The CUDA allocator in MiB (``peak``: most allocated at once since start; ``free``: on the whole card)."""
    if not str(device).startswith("cuda"):
        return None
    import torch

    mib = 1 << 20
    free, _ = torch.cuda.mem_get_info(device)
    return {"allocated": torch.cuda.memory_allocated(device) // mib,
            "reserved": torch.cuda.memory_reserved(device) // mib,
            "peak": torch.cuda.max_memory_allocated(device) // mib, "free": free // mib}


def _release_cuda(device: str) -> None:
    """Hand cached blocks back after a dissection so ranking queries never find the card full of them."""
    if str(device).startswith("cuda"):
        import torch

        torch.cuda.empty_cache()


class AnySong:
    """Request logic shared by HTTP threads. One GPU lock serialises the (non-thread-safe) engine."""

    def __init__(self, engine, deezer, meta: dict[int, tuple[str, str]], cache_size: int = 2048,
                 max_pending: int = 8, search_ttl: float = 3600.0, keep_previews: bool = False,
                 follower_timeout: float = 180.0, known_ttl: float = 600.0, languages=None,
                 scene: SceneTable | None = None, scene_params: SceneParams | None = None,
                 related_ttl: float = 86400.0, tempo: TempoTable | None = None, tracker=None,
                 vibe_params: samevibe.SameVibeParams | None = None, learn: LearnQueue | None = None,
                 tiers=None, artist_languages: bool = True, copy_score: float | None = COPY_SAMPLES_SCORE,
                 max_versions: int | None = MAX_VERSIONS, sounds_boost: float = SOUNDS_BOOST,
                 ranker: listener.Ranker | None = None, best_sound_weight: float = BEST_SOUND_WEIGHT,
                 sounds_listener_weight: float = SOUNDS_LISTENER_WEIGHT, best_audio_weight: float = BEST_AUDIO_WEIGHT,
                 best_artist_gap: int = BEST_ARTIST_GAP, best_artist_once: int = BEST_ARTIST_ONCE,
                 best_head_standout: float = BEST_HEAD_STANDOUT, best_head_window: int = BEST_HEAD_WINDOW,
                 best_head_bare: int = BEST_HEAD_BARE, best_head_evidence_only: bool = BEST_HEAD_EVIDENCE_ONLY,
                 stems: stemlib.StemStore | None = None, dissector: stemlib.Dissector | None = None,
                 catalog_previews: str | Path | None = None):
        self.engine, self.deezer, self.meta, self.keep_previews = engine, deezer, meta, keep_previews
        # Stem dissections (``dissect``): stored rows, else (with ``dissector``) the song's preview separated on the
        # GPU, read from ``catalog_previews/<id % 1000:03d>/<id>.mp3`` when there, else fetched from Deezer.
        self.stems = stems if stems is not None or dissector is None else stemlib.StemStore()
        self.dissector = dissector
        self.catalog_previews = Path(catalog_previews) if catalog_previews is not None else None
        self._stem_flight: dict[int, Future] = {}
        self.dissected = {"stored": 0, "live": 0}
        self.languages = languages
        # Songs without a confident language label take their artist's majority language in the language gate.
        self.artist_languages = artist_languages
        # Other artists' versions of the seed's own song: copies of its recording (samples score at least
        # ``copy_score``) dropped, at most ``max_versions`` of the rest kept.
        self.copy_score, self.max_versions = copy_score, max_versions
        # Sounds like: lift songs by their artist's scene affinity (0 = the sonic channel's order).
        self.sounds_boost = sounds_boost
        # Listener-aware ranking (``listener``): needs the scene table; replaces the scene slots and the Sounds like
        # lift for seeds whose artist the table (or Deezer) knows.
        self.ranker = ranker if scene is not None else None
        self.best_sound_weight, self.sounds_listener_weight = best_sound_weight, sounds_listener_weight
        self.best_audio_weight = best_audio_weight
        self.best_artist_gap, self.best_artist_once = best_artist_gap, best_artist_once
        self.best_head_standout, self.best_head_window = best_head_standout, best_head_window
        self.best_head_bare, self.best_head_evidence_only = best_head_bare, best_head_evidence_only
        self._artist_tracks = listener.ArtistTracks(scene) if self.ranker is not None else None
        self._sonic_core: np.ndarray | None = None
        self.learn = learn
        # Catalog tiers (``tiers.TierGate``): songs added by the crawl only for seeds in their artist's scene.
        self.tiers = tiers
        self.scene, self.scene_params = scene, scene_params or SceneParams()
        # Same vibe needs catalog tempos and the sonic channel; ``tracker`` estimates the tempo of unseen seeds.
        self.tempo, self.tracker = tempo, tracker
        self.vibe_params = vibe_params or samevibe.SameVibeParams()
        self.vibe = tempo is not None and "sonic" in engine.indexes
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
                key = _lookup_key(artist, title)
                entries.append((key[1] != _norm(_fold(unicodedata.normalize("NFC", title))), tid, key))
        releases: dict[tuple[str, str], list[int]] = {}
        for _, tid, key in sorted(entries):
            releases.setdefault(key, []).append(tid)
        self.by_key: dict[tuple[str, str], tuple[int, ...]] = {key: tuple(ids) for key, ids in releases.items()}
        # ISRC -> Deezer id answered by Deezer's ISRC lookup (``resolve_isrc``).
        self.isrcs = LRU(cache_size * 4)

    def _count(self, kind: str) -> None:
        with self._lock:
            self.served[kind] += 1

    def health(self) -> dict:
        with self._lock:
            served = dict(self.served)
            dissected = dict(self.dissected)
        scene = None if self.scene is None else {"tracks": int((self.scene.track_artist > 0).sum()),
                                                 "artists": len(self.scene.artist_ids)}
        tempo = None if not self.vibe else {"tracks": len(self.tempo.by_id), "live": self.tracker is not None}
        tiers = None if self.tiers is None else self.tiers.sizes
        stems = None
        if self.stems is not None:
            stems = {"songs": 0 if self.stems.table is None else len(self.stems.table),
                     "live": self.dissector is not None, "graded": self.stems.calib is not None,
                     "dissected": dissected}
        return {"ok": True, "device": self.engine.device,
                "channels": {name: len(ix) for name, ix in self.engine.indexes.items()},
                "cached_results": len(self.results), "served": served, "scene": scene, "tempo": tempo,
                "tiers": tiers, "ranker": None if self.ranker is None else self.ranker.meta or "on", "stems": stems,
                "gpu_mib": _gpu_mib(self.engine.device)}

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

    def _catalog_release(self, key: tuple[str, str], version: set[str], tags: frozenset[str]) -> int | None:
        """The indexed release of song ``key`` with the requested version words and content tags, if any."""
        for tid in self.by_key.get(key, ()):
            title = self.meta[tid][1]
            if version_words(title) == version and _content_tags(title) == tags:
                return tid
        return None

    def _search_song(self, artist: str, title: str, key: tuple[str, str]) -> list[dict]:
        """Deezer search cards with a preview for song ``key``: first for "artist title" as given, then, if that finds
        none, for the normalized key text, which Deezer's search handles better ("crosses bitches brew" finds
        "††† (Crosses)" where "✝✝✝ (Crosses) Bitches Brew" finds nothing)."""
        queries = [f"{artist} {title}"[:200]]
        plain = f"{key[0]} {key[1]}".strip()[:200]
        if plain and " ".join(queries[0].casefold().split()) != plain:
            queries.append(plain)
        for q in queries:
            cards = self.search(q, RESOLVE_LIMIT)["tracks"]
            same = [c for c in cards if c["has_preview"] and _lookup_key(c["artist"], c["title"]) == key]
            if same:
                return same
        return []

    def resolve(self, artist: str, title: str, on_search: Callable[[], None] | None = None) -> int:
        """Deezer id of the recording named by ``artist`` and ``title``.

        Songs match by ``_lookup_key``. Among a song's releases, one with the same ``version_words`` and
        ``_content_tags`` wins, so "Song - Live" asks for a live recording, while "Song" or "Song - Remastered" will
        not be served by a remix or a medley. An indexed catalog release answers without any Deezer request.
        Otherwise ``on_search`` runs (the caller's search rate limit) and Deezer is searched; a hit with a preview,
        that version and those content tags wins. Failing that, an indexed release with that version, a hit with
        that version, the song's best indexed release, then its first search hit stand in.
        """
        artist, title = _text_param(artist, "artist"), _text_param(title, "title")
        key, version, tags = _lookup_key(artist, title), version_words(title), _content_tags(title)
        found = self._catalog_release(key, version, tags)
        if found is not None:
            return found
        known = self.by_key.get(key, ())
        same_version = [tid for tid in known if version_words(self.meta[tid][1]) == version]
        try:
            if on_search is not None:
                on_search()
            cards = self._search_song(artist, title, key)
        except ApiError:
            if known:
                return (same_version or known)[0]
            raise
        for card in cards:
            if version_words(card["title"]) == version and _content_tags(card["title"]) == tags:
                return card["id"]
        if same_version:
            return same_version[0]
        for card in cards:
            if version_words(card["title"]) == version:
                return card["id"]
        if known:
            return known[0]
        if cards:
            return cards[0]["id"]
        raise ApiError(404, "song not found")

    def resolve_isrc(self, isrc: str, artist: str | None = None, title: str | None = None,
                     on_search: Callable[[], None] | None = None) -> int:
        """Deezer id of the recording with this ISRC, which the Spotify extension reads from Spotify.

        With ``artist`` and ``title``, an indexed release of that exact version answers first, without Deezer.
        Then Deezer's ISRC lookup (cached), which, unlike its search, also finds tracks that Deezer does not license
        where this server runs; a hit counts if it is indexed or has a preview. Failing that, ``resolve``.
        """
        code = "".join(isrc.split()).upper()
        if not _ISRC.match(code):
            raise ApiError(400, "isrc must be a 12-character ISRC")
        named = bool(artist and title)
        if named:
            artist, title = _text_param(artist, "artist"), _text_param(title, "title")
            found = self._catalog_release(_lookup_key(artist, title), version_words(title), _content_tags(title))
            if found is not None:
                return found
        tid = self.isrcs.get(code)
        if tid is None:
            if on_search is not None:
                on_search()
                on_search = None
            try:
                track = self.deezer.track_by_isrc(code)
            except RuntimeError:
                track = {}
            if isinstance(track, dict) and type(track.get("id")) is int and (
                    self.engine.indexed(track["id"]) or track.get("preview")):
                tid = track["id"]
                self.known.put(tid, track)
                self.isrcs.put(code, tid)
        if tid is not None:
            return tid
        if named:
            return self.resolve(artist, title, on_search)
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
            windows = None
            if self.engine.indexed(track_id):
                artist, title = self.meta.get(track_id, ("", ""))
                query = {"id": track_id, "artist": artist, "title": title, "source": "catalog"}
                timings: dict[str, float] = {}
                with self._gpu(timings):
                    raw, t = self.engine.search_stored(track_id, FETCH, **self._allow(query))
                    vibe = self._vibe_inputs(raw, t, query_track=track_id) if self.vibe else None
                if vibe is not None:
                    vibe["bpm"] = self.tempo.bpm(track_id)
            else:
                query, raw, timings, t, vibe, windows = self._live(track_id)
                if self.learn is not None:
                    self.learn.add(track_id, query.get("artist_id"))
            self._count(query["source"])
            timings |= t
            related = self._scene_context(query, timings)
            res = {"query": query, "raw": raw, "related": related, "vibe": vibe}
            if self.ranker is not None:
                res["listener"] = self._listener(res, windows, timings)
            timings["total"] = (time.perf_counter() - t0) * 1000
            return res | {"timings_ms": {k: round(v, 1) for k, v in timings.items()}}
        finally:
            self.pending.release()

    def _listener(self, res: dict, windows, timings: dict[str, float]) -> dict | None:
        """Listener-aware inputs, computed once per seed: each candidate's listener probability (``ranker``) and served
        sound score, and the scene candidates (``listener.scene_candidates``). None without a known seed artist."""
        query = res["query"]
        artist_id = query.get("artist_id")
        profile = self.scene.profile(artist_id, res.get("related")) if artist_id else None
        if profile is None:
            return None
        t = time.perf_counter()
        qid = int(query["id"])
        query_track = None if windows is not None else qid
        vibe = res.get("vibe")
        son = dict(vibe["muq"]) if vibe is not None else {}
        missing = {int(tid) for rows in res["raw"].values() for tid, _ in rows} - set(son)
        allowed = self._tier_filter(artist_id)

        def score(tids):
            return self.engine.score_tracks("sonic", tids, windows=windows, query_track=query_track)

        waited: dict[str, float] = {}
        with self._gpu(waited):
            if missing:
                son |= score(missing)
            scene = listener.scene_candidates(profile, self._artist_tracks, score, self.scene.artist, allowed)
        son |= scene
        seed_bpm = vibe["bpm"] if vibe is not None else (self.tempo.bpm(qid) if self.tempo is not None else None)
        tids, X = listener.features(
            res["raw"], son, scene, qid, profile, self.scene.related(artist_id) or res.get("related") or (),
            self.scene, self.scene_params, seed_bpm,
            self.tempo.bpm if self.tempo is not None else (lambda tid: None), self._language_known)
        p = self.ranker.predict(X)
        timings["listener"] = (time.perf_counter() - t) * 1000
        timings["listener_queue"] = waited.get("queue", 0.0)
        return {"p": dict(zip(tids, p.tolist())), "son": {tid: float(son[tid]) for tid in tids},
                "scene": sorted(scene)}

    def _tier_filter(self, artist_id: int) -> Callable[[int], bool] | None:
        """The catalog tiers for scene candidates: core songs, and crawled songs of the artists the search allows."""
        if self.tiers is None:
            return None
        if self._sonic_core is None:
            self._sonic_core = self.tiers.rows["sonic"][0].cpu().numpy()
        eligible = set(self.tiers.artists(artist_id))
        pos, core = self.engine.indexes["sonic"].pos, self._sonic_core
        return lambda tid: (tid in pos and bool(core[pos[tid]])) or self.scene.artist(tid) in eligible

    def _language_known(self, track_id: int) -> bool:
        if self.languages is None:
            return False
        item = {"id": int(track_id), "artist": self.meta.get(int(track_id), ("", ""))[0]}
        return self._item_language(item) != "unknown"

    def _allow(self, query: dict) -> dict:
        """Engine keyword for the catalog tiers: row masks for the seed artist's scene as the scene table records it.
        No Deezer request, so the search never waits for one (none without tiers)."""
        if self.tiers is None:
            return {}
        artist_id = query.get("artist_id") or self.tiers.scene.artist(int(query["id"]))
        return {"allow": self.tiers.masks(self.tiers.artists(artist_id))}

    def _vibe_inputs(self, raw: dict, timings: dict[str, float], windows=None, query_track: int | None = None) -> dict:
        """Served MuQ-MuLan scores against the seed for every track in the channel lists (Same vibe's sound gate):
        the sonic list's own scores, and exact scores for the melody and samples tracks it does not hold."""
        t = time.perf_counter()
        muq = {int(tid): float(score) for tid, score in raw.get("sonic", ())}
        others = {int(tid) for channel, rows in raw.items() if channel != "sonic" for tid, _ in rows} - set(muq)
        muq |= self.engine.score_tracks("sonic", others, windows=windows, query_track=query_track)
        timings["vibe_scores"] = (time.perf_counter() - t) * 1000
        return {"muq": muq, "bpm": None}

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
        clip22 = None
        try:
            clips, t_decode = self.engine.decode(path)
            if self.vibe and self.tracker is not None:
                t = time.perf_counter()
                try:
                    clip22 = decode22(path)
                except Exception:  # noqa: BLE001 - an unknown seed tempo lets every song pass the tempo gate
                    clip22 = None
                timings["decode_tempo"] = (time.perf_counter() - t) * 1000
        except Exception as exc:
            raise ApiError(422, "could not decode the preview") from exc
        finally:
            if not self.keep_previews:
                Path(path).unlink(missing_ok=True)
        timings |= t_decode
        vibe = None
        with self._gpu(timings):
            wins, t_embed = self.engine.embed_clips(clips)
            raw, t_search = self.engine.search_windows(wins, FETCH, [track_id], **self._allow(query))
            if self.vibe:
                vibe = self._vibe_inputs(raw, t_search, windows=wins["sonic"])
                if clip22 is not None:
                    t = time.perf_counter()
                    bpm = self.tracker.tempo(clip22)[0]
                    vibe["bpm"] = bpm if np.isfinite(bpm) else None
                    t_search["tempo"] = (time.perf_counter() - t) * 1000
        return query, raw, timings | t_embed, t_search, vibe, wins.get("sonic")

    def stems_missing(self, track_id: int) -> bool:
        """True when ``dissect`` would separate this song live (the costly case the HTTP layer rate-limits)."""
        return self.dissector is not None and self.stems.get(track_id)[0] is None

    def dissect(self, track_id: int, other: int | None = None) -> dict:
        if self.stems is None:
            raise ApiError(404, "song dissection is not enabled")
        ids = [track_id] if other is None or other == track_id else [track_id, other]
        rows, songs = [], []
        for tid in ids:
            row, source, ms = self._stem_row(tid)
            rows.append(row)
            songs.append(stemlib.payload(tid, row) | {"source": source} | ({"ms": round(ms)} if ms else {}))
        out: dict[str, Any] = {"ok": True, "songs": songs}
        if len(rows) == 2:
            out["compare"] = stemlib.compare(rows[0], rows[1], self.stems.mu, self.stems.calib)
        return out

    def _stem_row(self, track_id: int) -> tuple[dict[str, np.ndarray], str, float | None]:
        row, source = self.stems.get(track_id)
        if row is not None:
            with self._lock:
                self.dissected["stored"] += 1
            return row, source, None
        if self.dissector is None:
            raise ApiError(404, "this song has not been dissected yet")
        with self._lock:
            fut = self._stem_flight.get(track_id)
            owner = fut is None
            if owner:
                fut = self._stem_flight[track_id] = Future()
        if not owner:
            try:
                return fut.result(timeout=self.follower_timeout), "live", None
            except FutureTimeout:
                raise ApiError(504, "timed out waiting for the dissection") from None
        t = time.perf_counter()
        try:
            row = self._dissect_live(track_id)
            self.stems.put(track_id, row)
            fut.set_result(row)
        except BaseException as exc:
            fut.set_exception(exc)
            raise
        finally:
            with self._lock:
                self._stem_flight.pop(track_id, None)
        with self._lock:
            self.dissected["live"] += 1
        return row, "live", (time.perf_counter() - t) * 1000

    def _stem_preview(self, track_id: int) -> tuple[Path, bool]:
        """The song's 30 s preview and whether it is a temporary download."""
        if self.catalog_previews is not None:
            path = self.catalog_previews / f"{track_id % 1000:03d}" / f"{track_id}.mp3"
            if path.is_file():
                return path, False
        info = self.known.get(track_id)
        if info is None:
            try:
                info = self.deezer.track(track_id)
            except RuntimeError as exc:
                raise ApiError(502, "track lookup failed") from exc
        if not info or info.get("id") is None:
            raise ApiError(404, "unknown track")
        try:
            path = self.deezer.fetch_preview(track_id, info)
        except RuntimeError as exc:
            raise ApiError(502, "preview download failed") from exc
        if path is None:
            raise ApiError(404, "no preview available for this track")
        return Path(path), True

    def _dissect_live(self, track_id: int) -> dict[str, np.ndarray]:
        if not self.pending.acquire(blocking=False):
            raise ApiError(503, "busy, retry shortly")
        try:
            path, temporary = self._stem_preview(track_id)
            try:
                mixes = stemlib.decode_stereo(path)[None]
            except Exception as exc:
                raise ApiError(422, "could not decode the preview") from exc
            finally:
                if temporary and not self.keep_previews:
                    path.unlink(missing_ok=True)
            try:
                with self._gpu({}):
                    try:
                        parts = self.dissector.gpu(mixes)
                    finally:
                        _release_cuda(self.engine.device)
            except Exception as exc:
                if "out of memory" in str(exc).lower():
                    raise ApiError(503, "the GPU is busy, retry shortly") from exc
                raise
            return self.dissector.cpu(parts[0], mixes, parts[1], parts[2])[0]  # measurements outside the GPU lock
        finally:
            self.pending.release()

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
        if dedupe and qkey is not None and (self.copy_score is not None or self.max_versions is not None):
            songs = self._limit_versions(songs, qkey)

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
                return self._item_language(item) in {"unknown", language}

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
        query = q | {"language": language, "language_mode": lang_mode, "hidden": hidden, "scene": scene_state}
        if res.get("vibe") is not None and "sonic" in filtered:
            bpm = res["vibe"]["bpm"]
            query["bpm"] = round(bpm, 1) if bpm else None
        if scene_state == "on" and res.get("listener") is not None:
            keep = (lambda item: self._item_language(item) in {"unknown", language}) \
                if lang_mode == "same" and language != "unknown" and self.languages is not None else None
            best, vibe, sounds = self._listener_lists(res, filtered, artists, profile, qkey, keep, language,
                                                      lang_mode, k, dedupe)
        else:
            fused = rank(filtered)
            best = []
            for key, score, ranks in fused[:k]:
                channel = min(ranks, key=lambda name: ranks[name])
                best.append(filtered[channel][key] | {"score": round(score, 5), "ranks": ranks})
            vibe = []
            if res.get("vibe") is not None and "sonic" in filtered:
                vibe = self._same_vibe(res["vibe"], filtered, [key for key, _, _ in fused], language, lang_mode, k)
            sounds = self._sounds(filtered, k, profile, artists)
        lists = {"best": best} if best else {}
        if vibe:
            lists["vibe"] = vibe
        if sounds:
            lists["sounds"] = sounds
        sound_of = {int(tid): float(score) for tid, score in res["raw"].get("sonic", ())}
        sound_of |= (res.get("vibe") or {}).get("muq") or {}
        sound_of |= (res.get("listener") or {}).get("son") or {}
        for items in (*lists.values(), *out.values()):
            for item in items:
                score = sound_of.get(int(item["id"]))
                if score is not None:
                    item["match"] = sound_match(score)
        return {"ok": True, "query": query, "results": lists | out, "timings_ms": res["timings_ms"],
                "cached": cached}

    def _listener_lists(self, res: dict, filtered: dict[str, dict[Hashable, dict]], artists: dict[Hashable, int | None],
                        profile, qkey, keep: Callable[[dict], bool] | None, language: str, lang_mode: str, k: int,
                        dedupe: bool) -> tuple[list[dict], list[dict], list[dict]]:
        """Best matches, Same vibe and Sounds like with the listener ranker (``res["listener"]``, from ``_listener``).

        Best: the audio head (``scene.audio_head``, likely covers and samples), then every other candidate, the channel
        lists' songs and the scene candidates, by z(listener logit) + ``best_sound_weight`` * z(sound score), then the
        per-artist caps. Sounds like: the sonic list by z(sound score) + ``sounds_listener_weight`` * z(listener logit),
        one song per artist. Same vibe: ``samevibe.same_vibe`` over those two orders. Songs by the seed's own artist
        get the median listener logit, so sound alone places them.
        """
        data = res["listener"]
        p_of, son_of = data["p"], data["son"]
        extra: dict[Hashable, dict] = {}
        for tid in data["scene"]:
            artist, title = self.meta.get(int(tid), (None, None))
            if title is None or int(tid) not in son_of:
                continue
            key: Hashable = song_key(artist, title) if dedupe else ("id", int(tid))
            if key in extra or any(key in order for order in filtered.values()):
                continue
            if dedupe and qkey is not None and key[1] == qkey[1]:
                continue  # the seed's own song or another artist's version of it: Best's head is for those
            item = {"id": int(tid), "score": round(son_of[int(tid)], 4), "artist": artist, "title": title}
            if keep is None or keep(item):
                extra[key] = item
                artists[key] = self.scene.artist(int(tid))
        pos = {channel: {key: i for i, key in enumerate(order, 1)} for channel, order in filtered.items()}
        keys = list(dict.fromkeys([key for order in filtered.values() for key in order] + list(extra)))
        shown: dict[Hashable, tuple[dict, dict[str, int]]] = {}
        for key in keys:
            ranks = {channel: p[key] for channel, p in pos.items() if key in p}
            if ranks:
                channel = min(ranks, key=lambda name: (ranks[name], name))
                shown[key] = (filtered[channel][key], ranks)
            else:
                shown[key] = (extra[key], {})

        def logits(ks: list, item_of: Callable[[Hashable], dict]) -> np.ndarray:
            p = np.array([p_of.get(int(item_of(key)["id"]), np.nan) for key in ks], dtype=np.float64)
            out = listener.logit(np.nan_to_num(p, nan=0.5))
            neutral = np.isnan(p) | np.array([artists.get(key) == profile.artist_id for key in ks], dtype=bool)
            out[neutral] = np.median(out[~neutral]) if (~neutral).any() else 0.0
            return out

        def sound(ks: list, item_of: Callable[[Hashable], dict]) -> np.ndarray:
            s = np.array([son_of.get(int(item_of(key)["id"]), np.nan) for key in ks], dtype=np.float64)
            return np.nan_to_num(s, nan=np.nanmin(s) if np.isfinite(s).any() else 0.0)

        def ordered(ks: list, score: np.ndarray) -> list:
            return [ks[i] for i in np.lexsort((np.arange(len(ks)), -score))]

        orders = {channel: list(order) for channel, order in filtered.items()}
        scores = {channel: [item["score"] for item in order.values()] for channel, order in filtered.items()}
        head_params = replace(self.scene_params, head_bare=self.best_head_bare,
                              head_evidence_only=self.best_head_evidence_only)
        head = [key for key in audio_head(orders, scores, head_params, standout_min=self.best_head_standout,
                                          standout_window=self.best_head_window) if key in shown]
        rest = [key for key in keys if key not in set(head)]
        if rest:
            item_of = lambda key: shown[key][0]  # noqa: E731
            audio = [sum(1.0 / (FUSE_K + pos[channel][key]) for channel in ("melody", "samples")
                         if key in pos.get(channel, {})) for key in rest]
            rest = ordered(rest, listener.zscore(logits(rest, item_of))
                           + self.best_sound_weight * listener.zscore(sound(rest, item_of))
                           + self.best_audio_weight * listener.zscore(audio))
        best_order = arrange_artists(head + rest, artists.get, profile.artist_id, self.scene_params, head=len(head),
                                     gap=self.best_artist_gap, once=self.best_artist_once)
        best = [shown[key][0] | {"score": round(1.0 / (FUSE_K + i), 5), "ranks": shown[key][1]}
                for i, key in enumerate(best_order[:k], 1)]

        sonic = filtered.get("sonic", {})
        sonic_order = list(sonic)
        if sonic_order:
            item_of = sonic.__getitem__
            sonic_order = ordered(sonic_order, listener.blend(sound(sonic_order, item_of),
                                                              logits(sonic_order, item_of),
                                                              self.sounds_listener_weight))
        sounds = self._sounds(filtered, k, order=sonic_order)

        vibe: list[dict] = []
        inputs = res.get("vibe")
        if inputs is not None and sonic:
            rep: dict[Hashable, dict] = {}
            for channel in samevibe.CHANNEL_ORDER:
                for key, item in filtered.get(channel, {}).items():
                    rep.setdefault(key, item)
            rep |= {key: item for key, item in extra.items() if key not in rep}
            cands = {}
            for key, item in rep.items():
                tid = int(item["id"])
                cands[key] = samevibe.Candidate(
                    track_id=tid, bpm=self.tempo.bpm(tid), muq=son_of.get(tid, inputs["muq"].get(tid, -np.inf)),
                    artist_id=artists.get(key) or 0,
                    language=self._item_language(item) if self.languages is not None else "unknown")
            chosen = samevibe.same_vibe([key for key in best_order if key in rep], sonic_order, cands, inputs["bpm"],
                                        language, lang_mode, self.vibe_params)
            vibe = [rep[key] | {"score": round(1.0 / (FUSE_K + i), 5)} for i, key in enumerate(chosen[:k], 1)]
        return best, vibe, sounds

    def _item_language(self, item: dict) -> str:
        """A listed song's language: its own label, else (with ``artist_languages``) its artist's majority language."""
        language = self.languages.language(int(item["id"]))
        if language == "unknown" and self.artist_languages:
            language = self.languages.artist_language(item.get("artist"))
        return language

    def _limit_versions(self, songs: dict[str, dict[Hashable, dict]], qkey: tuple[str, str]
                        ) -> dict[str, dict[Hashable, dict]]:
        """Other artists' versions of the seed's own song (same ``song_key`` title, another artist). With
        ``copy_score``, those scoring at least that in the samples list are dropped as copies of the seed's recording;
        with ``max_versions``, only that many of the rest stay (the best placed in any channel), so covers and edits of
        one song cannot fill the lists. The channel lists themselves are left whole."""
        best: dict[Hashable, int] = {}
        for order in songs.values():
            for i, key in enumerate(order, 1):
                if isinstance(key, tuple) and key[0] != "id" and key[1] == qkey[1] and key[0] != qkey[0]:
                    best[key] = min(best.get(key, i), i)
        drop: set[Hashable] = set()
        if self.copy_score is not None:
            samples = songs.get("samples", {})
            drop = {key for key in best if key in samples and samples[key]["score"] >= self.copy_score}
        if self.max_versions is not None:
            kept = [key for key in sorted(best, key=lambda key: best[key]) if key not in drop]
            drop.update(kept[self.max_versions:])
        if not drop:
            return songs
        return {channel: {key: item for key, item in order.items() if key not in drop}
                for channel, order in songs.items()}

    def _sounds(self, view: dict[str, dict[Hashable, dict]], k: int, profile=None,
                artists: Mapping[Hashable, int | None] | None = None, order: Sequence[Hashable] | None = None
                ) -> list[dict]:
        """The Sounds like list: the sonic channel as filtered for Best (no seed song and, in same-language mode,
        no confidently other-language song), one song per artist, each with its catalog tempo when known. With
        ``order`` (the listener ranker's), songs follow it. Otherwise, with ``sounds_boost`` and a scene profile, songs
        move up by their artist's scene affinity: 1 / (k + rank) times (1 + sounds_boost * affinity)."""
        items = list(view.get("sonic", {}).items())
        if order is not None:
            sonic = view.get("sonic", {})
            items = [(key, sonic[key]) for key in order]
        elif self.sounds_boost > 0 and profile is not None and self.scene is not None and artists is not None:
            def lifted(i: int, key: Hashable) -> float:
                artist_id = artists.get(key)
                candidate = self.scene.stored(artist_id) if artist_id else None
                return (1.0 / (FUSE_K + i)) * (1.0 + self.sounds_boost * affinity(
                    profile, candidate, artist_id, self.scene_params))

            lift = {key: lifted(i, key) for i, (key, _) in enumerate(items, 1)}
            items.sort(key=lambda pair: -lift[pair[0]])
        seen: set[Hashable] = set()
        sounds = []
        for _, item in items:
            tid = int(item["id"])
            artist_id = self.scene.artist(tid) if self.scene is not None else None
            artist: Hashable = ("id", artist_id) if artist_id else ("name", (item["artist"] or "").casefold() or tid)
            if artist in seen:
                continue
            seen.add(artist)
            bpm = self.tempo.bpm(tid) if self.tempo is not None else None
            sounds.append(item | {"bpm": round(bpm, 1) if bpm else None})
            if len(sounds) == k:
                break
        return sounds

    def _same_vibe(self, inputs: dict, view: dict[str, dict[Hashable, dict]], best_order: list, language: str,
                   lang_mode: str, k: int) -> list[dict]:
        """Same vibe list (``samevibe.same_vibe``); each song is read from its representative track."""
        rep: dict[Hashable, dict] = {}
        for channel in samevibe.CHANNEL_ORDER:
            for key, item in view.get(channel, {}).items():
                rep.setdefault(key, item)
        cands = {}
        for key, item in rep.items():
            tid = int(item["id"])
            cands[key] = samevibe.Candidate(
                track_id=tid, bpm=self.tempo.bpm(tid), muq=inputs["muq"].get(tid, -np.inf),
                artist_id=(self.scene.artist(tid) or 0) if self.scene is not None else 0,
                language=self._item_language(item) if self.languages is not None else "unknown")
        keys = samevibe.same_vibe(best_order, list(view["sonic"]), cands, inputs["bpm"], language, lang_mode,
                                  self.vibe_params)
        return [rep[key] | {"score": round(1.0 / (FUSE_K + i), 5)} for i, key in enumerate(keys[:k], 1)]


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
                if "id" not in params and "isrc" in params:
                    tid = svc.resolve_isrc((params.get("isrc") or [""])[0], (params.get("artist") or [None])[0],
                                           (params.get("title") or [None])[0],
                                           lambda: self._limit("/api/anysong/search"))
                elif "id" not in params and ("title" in params or "artist" in params):
                    tid = svc.resolve((params.get("artist") or [""])[0], (params.get("title") or [""])[0],
                                      lambda: self._limit("/api/anysong/search"))
                else:
                    tid = _int_param(params, "id", None, 1, 2**53)
                if not svc.cached(tid):  # only new rankings cost GPU time and Deezer requests
                    self._limit(url.path)
                return self._send(200, svc.similar(tid, k, dedupe, lang_mode, scene), max_age=3600)
            if url.path == "/api/anysong/dissect":
                tid = _int_param(params, "id", None, 1, 2**53)
                other = _int_param(params, "with", 0, 1, 2**53) or None
                for t in dict.fromkeys((tid, other)):
                    if t is not None and svc.stems_missing(t):  # only live dissections cost GPU time
                        self._limit(url.path)
                return self._send(200, svc.dissect(tid, other), max_age=3600)
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
                 cors_origin: str | None = None, client_header: str | None = None, dissect_per_min: float = 12):
        super().__init__(addr, Handler)
        self.service, self.cors_origin, self.client_header = service, cors_origin, client_header
        self.limiters = {"/api/anysong/search": TokenBucket(search_per_min, max(1, int(search_per_min // 4))),
                         "/api/anysong/similar": TokenBucket(similar_per_min, max(1, int(similar_per_min // 4))),
                         "/api/anysong/dissect": TokenBucket(dissect_per_min, max(1, int(dissect_per_min // 4)))}


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
    p.add_argument("--core-ids", type=Path, default=None,
                   help="npy of the core catalog's track ids: other indexed songs only for seeds in their artist's "
                        "scene (anysong.tiers; needs --scene)")
    p.add_argument("--tempo", type=Path, default=None,
                   help="catalog tempo table from anysong.tempo; enables the Same vibe list")
    p.add_argument("--tempo-model", default=None,
                   help="Beat This! checkpoint for the tempo of unseen seeds (without it their tempo is unknown)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8790)
    p.add_argument("--channels", nargs="+", default=list(CHANNELS), choices=list(CHANNELS))
    p.add_argument("--candidates", type=int, default=None)
    p.add_argument("--hub-alpha", type=float, default=None,
                   help="CSLS hub penalty weight (default: as recorded by hub_build; 0 disables)")
    p.add_argument("--half-stage1", nargs="*", default=[], choices=list(CHANNELS),
                   help="channels whose quantized stage 1 runs as an fp16 GEMM (serve.TwoStageIndex)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--preview-dir", type=Path, default=None, help="scratch space for live previews")
    p.add_argument("--cors-origin", default=None)
    p.add_argument("--client-ip-header", default=None,
                   help="trusted proxy header with the client address, e.g. CF-Connecting-IP")
    p.add_argument("--search-per-min", type=float, default=60)
    p.add_argument("--similar-per-min", type=float, default=20)
    p.add_argument("--max-pending", type=int, default=8)
    p.add_argument("--cache-size", type=int, default=4096)
    p.add_argument("--learn-queue", type=Path, default=None,
                   help="append songs answered live (Deezer track and artist ids, UTC day) for anysong.crawl")
    p.add_argument("--ranker", type=Path, default=None,
                   help="listener ranker (anysong.listener) for the lists users see; needs --scene")
    p.add_argument("--stems", type=Path, default=None,
                   help="stem table from anysong.stems pack: stored dissections for /api/anysong/dissect")
    p.add_argument("--stems-model", type=Path, default=None,
                   help="folder with the htdemucs_6s weights: dissect other songs live (needs --tempo-model)")
    p.add_argument("--stems-cache", type=Path, default=None, help="keep live dissections here across restarts")
    p.add_argument("--stems-fp32", action="store_true", help="separate in float32 (default: float16 autocast)")
    p.add_argument("--catalog-previews", type=Path, default=None,
                   help="catalog previews as <id %% 1000>/<id>.mp3, dissected without a Deezer download")
    p.add_argument("--dissect-per-min", type=float, default=12)
    args = p.parse_args(argv)
    if args.stems_model is not None and not args.tempo_model:
        p.error("--stems-model needs --tempo-model")
    if args.core_ids is not None and args.scene is None:
        p.error("--core-ids needs --scene")
    if args.ranker is not None and args.scene is None:
        p.error("--ranker needs --scene")

    scratch = args.preview_dir or Path(tempfile.mkdtemp(prefix="anysong-"))
    t = time.perf_counter()
    engine = Engine(args.index, args.channels, args.device, candidates=args.candidates, hub_alpha=args.hub_alpha,
                    half_stage1=args.half_stage1)
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
    tiers = None
    if args.core_ids is not None:
        from .tiers import TierGate

        tiers = TierGate.load(args.core_ids, scene, engine.indexes)
    tempo = TempoTable.load(args.tempo) if args.tempo is not None else None
    tracker = None
    if tempo is not None and args.tempo_model:
        from .tempo import Tracker

        tracker = Tracker(args.tempo_model, args.device)
        tracker.tempo(np.zeros(30 * 22050, np.float32))  # warm-up
    store, dissector, stems_warm = None, None, None
    if args.stems is not None or args.stems_model is not None:
        store = stemlib.StemStore(stemlib.StemTable.load(args.stems) if args.stems is not None else None,
                                  args.stems_cache)
    if args.stems_model is not None:
        from .tempo import Tracker

        t_stems = time.perf_counter()
        # One song at a time, sharing the service's Beat This! and MuQ-MuLan models; two stems per MuQ call keep the
        # spike small on an 8 GB card.
        dissector = stemlib.Dissector(stemlib.Separator(args.stems_model, args.device, fp16=not args.stems_fp32,
                                                        batch=1),
                                      tracker or Tracker(args.tempo_model, args.device),
                                      engine.embedder("muq_mulan"), workers=0, embed_batch=2)
        noise = np.random.default_rng(0).standard_normal((1, 2, stemlib.SAMPLES)).astype(np.float32) * 0.05
        dissector.gpu(noise)  # warm-up
        _release_cuda(args.device)
        stems_warm = round(time.perf_counter() - t_stems, 1)
        threading.Thread(target=stemlib.warm_up, name="stems-warm-up", daemon=True).start()
    service = AnySong(engine, Deezer(scratch), meta, args.cache_size, args.max_pending,
                      languages=languages, scene=scene, tempo=tempo, tracker=tracker,
                      learn=LearnQueue(args.learn_queue) if args.learn_queue else None, tiers=tiers,
                      ranker=listener.Ranker.load(args.ranker) if args.ranker is not None else None,
                      stems=store, dissector=dissector, catalog_previews=args.catalog_previews)
    server = Server((args.host, args.port), service, args.search_per_min, args.similar_per_min,
                    args.cors_origin, args.client_ip_header, dissect_per_min=args.dissect_per_min)
    sizes = {name: len(ix) for name, ix in engine.indexes.items()}
    hubs = {name: getattr(ix, "hub_alpha", 0.0) for name, ix in engine.indexes.items()}
    print(f"ready in {time.perf_counter() - t:.1f}s (warmup {warm:.1f}s) {sizes} hub_alpha={hubs} "
          f"tiers={None if tiers is None else tiers.sizes} ranker={service.health()['ranker']} "
          f"stems={service.health()['stems']} (warmup {stems_warm}s) gpu={_gpu_mib(args.device)} "
          f"on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        if args.preview_dir is None:
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
