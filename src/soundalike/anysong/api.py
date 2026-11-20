"""Any-song HTTP API: Deezer search plus sound-alike ranking for any Deezer track.

    python -m soundalike.anysong.api --index /opt/soundalike/index/full --meta /opt/soundalike/index/catalog_meta.npz

GET endpoints (JSON):

- ``/healthz``
- ``/api/anysong/search?q=<text>&limit=10``  Deezer track search; ``in_catalog`` marks indexed tracks
- ``/api/anysong/similar?id=<deezer id>&k=20&dedupe=1``  top matches per channel, plus ``best``: the melody and
  samples lists fused into one ranking (``fuse``). With a scene table (``--scene``) the fused list also weighs the
  artist scene (same artist, related artists, region, genre; ``scene.scene_fuse``); ``scene=0`` turns that off.
  With the listener ranker (``--ranker``), each ``best`` item also carries ``basis``: ``audio`` for the audio head
  (likely covers, samples and shared melodies) and ``listeners`` for the songs placed after it by what the seed's
  listeners also play blended with sound; lists without the listener ranker (``scene=0`` among them) carry no
  ``basis``, and its absence means audio only. With a tempo table (``--tempo``), ``vibe`` is the Same vibe list:
  Best matches that share the seed's tempo and sound, one song per artist (``samevibe``). ``sounds`` is the Sounds
  like list: the sonic channel with Best's language filter, one song per artist, each with its catalog ``bpm`` when
  known. ``main`` and ``listeners`` serve a two-tab layout: ``main`` is the audio head, then the songs that sound
  most alike with what the seed's listeners play blended in, one song per artist (``basis`` ``audio`` or
  ``sound``); ``listeners``, only with the listener ranker, is other artists' songs that the seed's listeners also
  play, one song per artist. Every listed song with a sound score also carries ``match``: how alike it sounds to the
  seed, 0 to 100, the percentile of its sound score among the songs these lists show for typical seeds
  (``sound_match``)
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

With ``--blend-head`` and ``--blend-cf`` the sonic search also ranks the whole catalog by a blended score
(``blend.Blend``: sound, scene head and playlist co-occurrence); ``--main-pool blend`` makes that ranking main's body.

With ``--consensus-model`` (and the blend's files) main's body is the consensus body (``consensus.Consensus``): the
songs the sound, scene head, playlist, melody and samples signals agree on, ordered by a small model; those rows
carry ``basis`` ``consensus``, the raw body score as ``consensus``, ``match`` (its percentile on
``--consensus-quantiles``) and each signal's rank as ``ranks_all``. Main's audio head and every other list are
unchanged, and so is main with ``scene=0``.

With ``--languages`` the searches also rank each channel (and the blended score) among the songs of the seed's language
alone (``languages.LanguageRows``), so the same-language lists are as full for a seed in a small language as for an
English one; ``query["language_pool"]`` then says how many indexed songs that language has. With ``--language-model``
a live seed's language is read from its preview by the model and rule that labelled the catalog
(``languages.LanguageDetector``); ``query["language_source"]`` says where a seed's language comes from: its own
"label", "detected", its "artist"'s majority or the table's "guess".
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
from dataclasses import dataclass, replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import numpy as np

from . import listener, samevibe
from . import stems as stemlib
from .deezer import version_words
from .scene import SceneParams, SceneTable, affinity, arrange_artists, audio_head, scene_fuse, standout
from .scenehead import SceneHead
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
# Two-tab layout, served next to the lists above. Main: the audio head (Best's rules, but MAIN_HEAD_BARE songs when
# none has cover or sample evidence), then songs by z(sound score) + MAIN_LISTENER_WEIGHT * z(listener logit), one per
# artist. MAIN_POOL "sonic" takes them from the sonic list; "all" adds Best's other candidates (melody, samples and
# scene songs) whose sound match is at least MAIN_MATCH_FLOOR. Without listener data: the head, then Sounds like's
# order. Listeners also play (listener data only): Best's candidates by other artists than the seed's, by
# z(listener logit) + LISTENERS_SOUND_WEIGHT * z(sound score), one per artist. With ``main_scale`` (sound sd,
# logit sd) main's body is ordered by sound / sound sd + MAIN_LISTENER_WEIGHT * logit / logit sd instead: fixed
# scales, so sound decides where it separates a seed's candidates and the ranker where it does not. MAIN_POOL "blend"
# (needs the engine's ``blend``): main's body is the catalog's blended ranking (``blend.Blend``) in its own order, with
# or without listener data, filtered like the channel lists and without the songs whose sound match is below
# MAIN_MATCH_FLOOR (the head is not asked); main then carries ``match`` but no ``overall``. MAIN_POOL "sonic+blend"
# (listener data and the engine's ``blend``): the sonic list's songs plus the blended ranking's (filtered and floored
# the same way), each with a listener probability, ordered as for "sonic" plus MAIN_BLEND_WEIGHT * the song's blended
# score; ``overall`` as for "sonic" (the blended term is not in it). Without listener data it is the "sonic" main.
# With the in-language lists (``languages.LanguageRows``) the same-language lists take their songs from the searches
# among the seed's language alone, and a "blend" main that the floor leaves short of songs goes on with the sonic
# list's other songs in its order.
MAIN_BLEND_WEIGHT = 0.0
MAIN_LISTENER_WEIGHT = 2.0
MAIN_POOL = "sonic"
MAIN_MATCH_FLOOR = 50
MAIN_HEAD_BARE = 0
# Main's head can be stricter than Best's: a song whose sound match is below MAIN_HEAD_MATCH also needs a standout
# z-score of at least MAIN_HEAD_STANDOUT (0 = off, Best's rule alone; MAIN_HEAD_MATCH 101 = every song). Z 3 and M 20
# passed the held-out gates on 2026-11-12 (docs/ANYSONG_EVAL.md, "Held-out result for the stricter head").
MAIN_HEAD_STANDOUT = 3.0
MAIN_HEAD_MATCH = 20
LISTENERS_SOUND_WEIGHT = 0.25
# With LANGUAGE_GUESS, a seed without a language of its own or an artist majority gets ``LanguageTable.guess``
# (docs/ANYSONG_EVAL.md, "A best-guess seed language"). True gates every list with it; it failed held-out rule 2 for
# Listeners also play on 2026-11-13 ("Held-out result for the seed language guess and overall match") and again in the
# retest. "main" gates only main (Best matches): every other list and ``query["language"]`` stay as with the guess
# off, and ``query["main_language_guess"]`` names the guess. It passed its held-out gates on 2026-11-13 ("Held-out
# result for the main-only seed language guess") and is served. False: no guess.
LANGUAGE_GUESS: bool | str = "main"
# ``head_all_languages``: an other-language song joins the audio head only when it stands in at least this many of the
# catalog's channel lists. On 2,517 seeds of fresh online claims (2026-11-18, anysong-lab v8/retr/head_study.py) rows
# listed in two or more channels were the song people named 13.5 % of the time (170 rows), rows in one list 2.4 %
# (165 rows, mean sound match 21).
CROSS_HEAD_CHANNELS = 2
# ``ranks_all`` of a consensus row: each signal's rank among the songs the seed's body counts, in
# ``consensus.SIGNALS`` order (0: the signal has no score for the song).
CONSENSUS_SIGNALS = ("sound", "head", "playlists", "melody", "samples")
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
    more alike than (``MATCH_QUANTILES``; a service with its own quantiles uses ``AnySong.sound_match``)."""
    return int(round(float(np.interp(score, quantiles, np.linspace(0, 100, len(quantiles))))))


def checked_match_quantiles(values: Sequence[float]) -> tuple[float, ...]:
    """Checked sound-match quantiles: as many as ``MATCH_QUANTILES`` (0, 5, ..., 100 %), finite and ascending."""
    try:
        out = tuple(float(v) for v in values)
    except (TypeError, ValueError) as exc:
        raise ValueError("match quantiles must be a list of numbers") from exc
    if len(out) != len(MATCH_QUANTILES) or not np.isfinite(out).all() or (np.diff(out) < 0).any():
        raise ValueError(f"match quantiles must be {len(MATCH_QUANTILES)} finite ascending sonic scores "
                         f"(the 0, 5, ..., 100 % quantiles)")
    return out


@dataclass(frozen=True)
class OverallScale:
    """How the blend a list is ordered by becomes its ``overall`` match. The list ranks one seed's candidates by z-scores
    of their sound score and listener logit; ``overall`` uses the same weights on a fixed scale instead,
    sound_weight * sound / ``sound_sd`` + listener_weight * logit / ``logit_sd`` (the typical spread of each within
    one seed's candidates), so it can be compared across seeds. That value, as the share of songs shown in this list
    for typical seeds that it beats (``quantiles``, 0 to 100 %), then goes through ``overall_match``."""

    sound_sd: float
    logit_sd: float
    quantiles: tuple[float, ...]


def overall_match(values: Sequence[float], head: int = 0) -> list[int]:
    """A list's ``overall`` match, 0 to 100, from its songs' overall percentiles in list order: made non-increasing down
    the list, so a song never shows a higher overall match than one placed above it. After the head, runs where the
    order disagrees with the percentiles share their mean (isotonic regression). The first ``head`` songs, placed
    first on cover or sample evidence, show at least the song after them."""
    blocks: list[list[float]] = []
    for value in values[head:]:
        blocks.append([float(value), 1.0])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] < blocks[-1][0] / blocks[-1][1]:
            total, count = blocks.pop()
            blocks[-1][0] += total
            blocks[-1][1] += count
    body = [total / count for total, count in blocks for _ in range(int(count))]
    floor = body[0] if body else 0.0
    top = []
    for value in reversed(values[:head]):
        floor = max(float(value), floor)
        top.append(floor)
    return [int(round(min(100.0, max(0.0, v)))) for v in top[::-1] + body]


# Overall match (``overall``) on main and Listeners also play: each list's own blend on a fixed scale (``OverallScale``).
# Calibrated on the development seeds (dev + sweep, 825 with listener data) with out-of-fold listener probabilities:
# sd = the median over seeds of each part's standard deviation among the list's candidates; quantiles 0, 2.5, ..., 100 %
# of the blend over the songs shown in that list's top 20 (grow/qa/overall_dev.py calib, 2026-11-13). A typical seed's
# first song shows 84 % (main) and 89 % (Listeners also play), its tenth 50 %.
MAIN_OVERALL: OverallScale | None = OverallScale(0.0138, 1.1893, (
    -37.1158, 22.238, 24.2477, 25.4065, 26.1839, 26.7744, 27.2681, 27.6908, 28.0597, 28.3819, 28.6687, 28.9419,
    29.2149, 29.4891, 29.7295, 29.9864, 30.2257, 30.4553, 30.7055, 30.9401, 31.1728, 31.3906, 31.6178, 31.8479,
    32.0672, 32.2702, 32.4923, 32.7144, 32.9226, 33.1546, 33.3879, 33.6076, 33.8413, 34.1028, 34.3816, 34.6812,
    35.0505, 35.4527, 35.9831, 36.7741, 40.261))
LISTENERS_OVERALL: OverallScale | None = OverallScale(0.1768, 1.4899, (
    -4.421, -0.2872, -0.0005, 0.1452, 0.2426, 0.3238, 0.3788, 0.4231, 0.4732, 0.5133, 0.5509, 0.584, 0.6188, 0.6525,
    0.6848, 0.7101, 0.738, 0.7652, 0.7919, 0.8206, 0.847, 0.8737, 0.9004, 0.9308, 0.9598, 0.9904, 1.0219, 1.0532,
    1.0868, 1.1221, 1.1592, 1.2022, 1.2486, 1.3042, 1.3684, 1.4401, 1.5281, 1.6263, 1.7672, 1.9689, 2.729))


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
                 tiers=None, artist_languages: bool = True, language_guess: bool | str = LANGUAGE_GUESS,
                 language_rows=None,
                 copy_score: float | None = COPY_SAMPLES_SCORE,
                 max_versions: int | None = MAX_VERSIONS, sounds_boost: float = SOUNDS_BOOST,
                 ranker: listener.Ranker | None = None, best_sound_weight: float = BEST_SOUND_WEIGHT,
                 sounds_listener_weight: float = SOUNDS_LISTENER_WEIGHT, best_audio_weight: float = BEST_AUDIO_WEIGHT,
                 best_artist_gap: int = BEST_ARTIST_GAP, best_artist_once: int = BEST_ARTIST_ONCE,
                 best_head_standout: float = BEST_HEAD_STANDOUT, best_head_window: int = BEST_HEAD_WINDOW,
                 best_head_bare: int = BEST_HEAD_BARE, best_head_evidence_only: bool = BEST_HEAD_EVIDENCE_ONLY,
                 main_listener_weight: float = MAIN_LISTENER_WEIGHT, main_pool: str = MAIN_POOL,
                 main_match_floor: float = MAIN_MATCH_FLOOR, main_scale: tuple[float, float] | None = None,
                 main_head_bare: int = MAIN_HEAD_BARE, main_blend_weight: float = MAIN_BLEND_WEIGHT,
                 main_head_standout: float = MAIN_HEAD_STANDOUT, main_head_match: float = MAIN_HEAD_MATCH,
                 listeners_sound_weight: float = LISTENERS_SOUND_WEIGHT,
                 main_overall: OverallScale | None = MAIN_OVERALL,
                 listeners_overall: OverallScale | None = LISTENERS_OVERALL,
                 stems: stemlib.StemStore | None = None, dissector: stemlib.Dissector | None = None,
                 catalog_previews: str | Path | None = None, head: SceneHead | None = None,
                 language_detector=None, head_all_languages: bool = False,
                 popularity: listener.Popularity | None = None, match_quantiles: Sequence[float] | None = None,
                 consensus=None, consensus_quantiles: Sequence[float] | None = None):
        self.engine, self.deezer, self.meta, self.keep_previews = engine, deezer, meta, keep_previews
        # Consensus body of main (``consensus.Consensus``): computed once per seed from the searches' stage-1 scores
        # (the engine's ``keep_base``); ``consensus_quantiles`` turn a body score into the row's ``match``.
        if consensus is not None and consensus_quantiles is None:
            raise ValueError("a consensus body needs the quantiles of its score (--consensus-quantiles)")
        self.consensus = consensus
        self.consensus_quantiles = None if consensus_quantiles is None else \
            checked_match_quantiles(consensus_quantiles)
        if consensus is not None:
            engine.keep_base = True
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
        # A seed neither places gets the table's best guess (``LanguageTable.guess``); the query says so. "main": for
        # main alone (LANGUAGE_GUESS).
        if language_guess not in (False, True, "main"):
            raise ValueError(f"language_guess must be False, True or 'main', not {language_guess!r}")
        self.language_guess = language_guess
        # Other artists' versions of the seed's own song: copies of its recording (samples score at least
        # ``copy_score``) dropped, at most ``max_versions`` of the rest kept.
        self.copy_score, self.max_versions = copy_score, max_versions
        # In-language lists (``languages.LanguageRows``, built with the same ``artist_languages`` rule): the searches
        # also rank the songs of the seed's language alone, and the same-language lists are made from those instead
        # of from what the gate leaves of the whole catalog's lists. None: the gate filters the catalog's lists.
        self.language_rows = language_rows if languages is not None else None
        # A live seed's language from its audio (``languages.LanguageDetector``: the catalog's labelling model and
        # rule on the seed's preview), as sure as a catalog song's own label and used like one (``_seed_language``).
        self.language_detector = language_detector if languages is not None else None
        self.detections = {"detected": 0, "unsure": 0, "failed": 0}
        # The audio head of main and of Best also takes other-language songs with cover or sample evidence in the
        # catalog's lists when the lists are gated to the seed's language (``_cross_head``).
        self.head_all_languages = head_all_languages
        # Sounds like: lift songs by their artist's scene affinity (0 = the sonic channel's order).
        self.sounds_boost = sounds_boost
        # Listener-aware ranking (``listener``): needs the scene table; replaces the scene slots and the Sounds like
        # lift for seeds whose artist the table (or Deezer) knows.
        self.ranker = ranker if scene is not None else None
        # The scene head and the popularity table feed the rankers trained with them (``listener.FEATURES_HEAD``,
        # ``FEATURES_POP``): the ranker's own feature list decides the columns (``listener.features``).
        self.head = head
        self.popularity = popularity
        if self.ranker is not None and "pop" in self.ranker.features and popularity is None:
            raise ValueError("this ranker reads the 'pop' feature and needs a popularity table (--popularity)")
        # ``match``: the served sound-match quantiles (``sound_match``); None: ``MATCH_QUANTILES``.
        self.custom_match_quantiles = match_quantiles is not None
        self.match_quantiles = MATCH_QUANTILES if match_quantiles is None else \
            checked_match_quantiles(match_quantiles)
        self.best_sound_weight, self.sounds_listener_weight = best_sound_weight, sounds_listener_weight
        self.best_audio_weight = best_audio_weight
        self.best_artist_gap, self.best_artist_once = best_artist_gap, best_artist_once
        self.best_head_standout, self.best_head_window = best_head_standout, best_head_window
        self.best_head_bare, self.best_head_evidence_only = best_head_bare, best_head_evidence_only
        if main_pool not in ("sonic", "all", "blend", "sonic+blend"):
            raise ValueError(f"main_pool must be 'sonic', 'all', 'blend' or 'sonic+blend', not {main_pool!r}")
        if "blend" in main_pool and getattr(engine, "blend", None) is None:
            raise ValueError(f"main_pool {main_pool!r} needs an engine with a blend")
        if main_pool == "blend" and getattr(engine.blend, "wp", 0.0):
            # "blend" shows the blended ranking as main's body without listener data too (``scene=0``), which must
            # stay free of popularity; "sonic+blend" only reads it next to the listener ranker.
            raise ValueError("main_pool 'blend' cannot take a blend with a popularity weight: use 'sonic+blend'")
        self.main_listener_weight, self.main_pool = main_listener_weight, main_pool
        self.main_blend_weight = main_blend_weight
        self.main_match_floor, self.main_head_bare = main_match_floor, main_head_bare
        if main_scale is not None and min(main_scale) <= 0:
            raise ValueError("main_scale needs two positive standard deviations")
        self.main_scale = main_scale
        self.main_head_standout, self.main_head_match = main_head_standout, main_head_match
        self.listeners_sound_weight = listeners_sound_weight
        # ``overall`` on main and Listeners also play (``OverallScale``); None leaves it out.
        self.main_overall, self.listeners_overall = main_overall, listeners_overall
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

    def sound_match(self, score: float) -> int:
        """``sound_match`` on this service's quantiles."""
        return sound_match(score, self.match_quantiles)

    def health(self) -> dict:
        with self._lock:
            served = dict(self.served)
            dissected = dict(self.dissected)
            detections = dict(self.detections)
        scene = None if self.scene is None else {"tracks": int((self.scene.track_artist > 0).sum()),
                                                 "artists": len(self.scene.artist_ids)}
        tempo = None if not self.vibe else {"tracks": len(self.tempo.by_id), "live": self.tracker is not None}
        tiers = None if self.tiers is None else self.tiers.sizes
        stems = None
        if self.stems is not None:
            stems = {"songs": 0 if self.stems.table is None else len(self.stems.table),
                     "live": self.dissector is not None, "graded": self.stems.calib is not None,
                     "dissected": dissected}
        blend = getattr(self.engine, "blend", None)
        return {"ok": True, "device": self.engine.device,
                "channels": {name: len(ix) for name, ix in self.engine.indexes.items()},
                "cached_results": len(self.results), "served": served, "scene": scene, "tempo": tempo,
                "tiers": tiers, "ranker": None if self.ranker is None else self.ranker.meta or "on", "stems": stems,
                "blend": None if blend is None else blend.info() | {"main": "blend" in self.main_pool},
                "popularity": None if self.popularity is None else {"songs": len(self.popularity)},
                "consensus": None if self.consensus is None else self.consensus.info(),
                "match_quantiles": "custom" if self.custom_match_quantiles else "default",
                "language_search": self.language_rows is not None,
                "language_detector": None if self.language_detector is None else detections,
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
                detected = None
                with self._gpu(timings):
                    raw, t = self.engine.search_stored(track_id, FETCH, **self._allow(query), **self._within(query))
                    bases = raw.pop("base", None)
                    vibe = self._vibe_inputs(raw, t, query_track=track_id) if self.vibe else None
                    consensus = self._consensus(query, bases, t)
                if vibe is not None:
                    vibe["bpm"] = self.tempo.bpm(track_id)
            else:
                query, raw, timings, t, vibe, windows, detected, consensus = self._live(track_id)
                if self.learn is not None:
                    self.learn.add(track_id, query.get("artist_id"))
            self._count(query["source"])
            timings |= t
            related = self._scene_context(query, timings)
            within = raw.pop("within", None)
            res = {"query": query, "raw": raw, "related": related, "vibe": vibe}
            if detected is not None:
                res["detected"] = detected
            if consensus is not None:
                res["consensus"] = consensus
            if within is not None:
                # The lists among the songs of the seed's gate language alone (``_within``), next to the catalog's.
                language = self._seed_language(query, detected)[0]
                res["lang"], res["lang_pool"] = language, self.language_rows.pool(language)
                res["blend_lang"] = within.pop("blend", None)
                res["raw_lang"] = within
            if "blend" in raw:  # not channels: only main reads them (``main_pool`` "blend" and "sonic+blend")
                res["blend"] = raw.pop("blend")
                res["blendz"] = {int(tid): float(score) for tid, score in raw.pop("blendz", ())}
            if self.ranker is not None:
                res["listener"] = self._listener(res, windows, timings)
            timings["total"] = (time.perf_counter() - t0) * 1000
            return res | {"timings_ms": {k: round(v, 1) for k, v in timings.items()}}
        finally:
            self.pending.release()

    def _consensus(self, query: dict, bases, timings: dict[str, float], detected: str | None = None, windows=None,
                   bpm: float | None = None) -> dict | None:
        """The seed's consensus bodies (``consensus.Consensus.bodies``), computed once per seed under the GPU lock:
        "same" among the songs of the language that gates main (``_seed_language``, the guess included; the codes
        are those of ``language_rows``) and "all" among every language's. ``bases``: the searches' stage-1 scores;
        ``windows`` and ``bpm``: a live seed's raw sonic windows and tempo. None without a consensus."""
        if self.consensus is None or bases is None:
            return None
        t = time.perf_counter()
        code = -1
        if self.language_rows is not None:
            code = self.language_rows.vocab.get(self._seed_language(query, detected)[0], -1)
        if query["source"] == "catalog":
            out = self.consensus.bodies(bases, code, row=self.engine.indexes["sonic"].pos[int(query["id"])])
        else:
            out = self.consensus.bodies(bases, code, windows=windows, artist=query.get("artist"),
                                        artist_id=query.get("artist_id"), bpm=bpm)
        timings["consensus"] = (time.perf_counter() - t) * 1000
        return out

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
        lang_raw = res.get("raw_lang") or {}
        missing = {int(tid) for rows in (*res["raw"].values(), *lang_raw.values()) for tid, _ in rows} - set(son)
        allowed = self._tier_filter(artist_id)

        def score(tids):
            return self.engine.score_tracks("sonic", tids, windows=windows, query_track=query_track)

        waited: dict[str, float] = {}
        with self._gpu(waited):
            if missing:
                son |= score(missing)
            scene = listener.scene_candidates(profile, self._artist_tracks, score, self.scene.artist, allowed)
        son |= scene
        # ``main_pool`` "sonic+blend": the blended ranking's songs are candidates too, with the exact sound scores
        # that came with them.
        blended: dict[int, float] = {}
        if self.main_pool == "sonic+blend":
            blended = {int(tid): float(s) for tid, s in (*(res.get("blend") or ()), *(res.get("blend_lang") or ()))
                       if int(tid) not in son}
            son |= blended
        # The in-language lists' songs that the catalog's lists do not hold are candidates as well.
        inside = {int(tid) for rows in lang_raw.values() for tid, _ in rows}
        seed_bpm = vibe["bpm"] if vibe is not None else (self.tempo.bpm(qid) if self.tempo is not None else None)
        head = None
        if self.head is not None and "head" in self.ranker.features:
            seed_head = self.head.seed(windows, query_track)
            head = self.head.cosines(seed_head, son) if seed_head is not None else {}
        tids, X = listener.features(
            res["raw"], son, scene, qid, profile, self.scene.related(artist_id) or res.get("related") or (),
            self.scene, self.scene_params, seed_bpm,
            self.tempo.bpm if self.tempo is not None else (lambda tid: None), self._language_known, head,
            extra={*blended, *inside}, columns=self.ranker.features,
            pop=None if self.popularity is None else self.popularity.pop)
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

    def _within(self, query: dict, detected: str | None = None) -> dict:
        """Engine keyword for the in-language lists: row masks of the songs the same-language gate would keep for this
        seed (``_seed_language``; ``detected``: a live seed's detected language); none without ``language_rows`` or
        such a language."""
        if self.language_rows is None:
            return {}
        masks = self.language_rows.masks(self._seed_language(query, detected)[0])
        return {} if masks is None else {"within": masks}

    def _vibe_inputs(self, raw: dict, timings: dict[str, float], windows=None, query_track: int | None = None) -> dict:
        """Served MuQ-MuLan scores against the seed for every track in the channel lists, the in-language ones
        (``raw["within"]``) included (Same vibe's sound gate): the sonic lists' own scores, and exact scores for the
        melody and samples tracks they do not hold."""
        t = time.perf_counter()
        within = raw.get("within") or {}
        muq = {int(tid): float(score) for tid, score in (*within.get("sonic", ()), *raw.get("sonic", ()))}
        others = {int(tid) for lists, skip in ((raw, ("sonic", "blend", "blendz", "within")),
                                                (within, ("sonic", "blend")))
                  for channel, rows in lists.items() if channel not in skip for tid, _ in rows} - set(muq)
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
        clip22 = excerpt = None
        detect = self.language_detector is not None and self.languages.language(track_id) == "unknown"
        try:
            clips, t_decode = self.engine.decode(path)
            if self.vibe and self.tracker is not None:
                t = time.perf_counter()
                try:
                    clip22 = decode22(path)
                except Exception:  # noqa: BLE001 - an unknown seed tempo lets every song pass the tempo gate
                    clip22 = None
                timings["decode_tempo"] = (time.perf_counter() - t) * 1000
            if detect:
                t = time.perf_counter()
                try:
                    excerpt = self.language_detector.excerpt(path)
                except Exception:  # noqa: BLE001 - without a detected language the seed is placed as before
                    self._detection("failed")
                timings["decode_language"] = (time.perf_counter() - t) * 1000
        except Exception as exc:
            raise ApiError(422, "could not decode the preview") from exc
        finally:
            if not self.keep_previews:
                Path(path).unlink(missing_ok=True)
        timings |= t_decode
        vibe = detected = None
        with self._gpu(timings):
            if excerpt is not None:
                # Before the search: the in-language lists need the seed's language.
                t = time.perf_counter()
                try:
                    detected = self.language_detector.detect(excerpt)[0]
                    self._detection("unsure" if detected == "unknown" else "detected")
                except Exception:  # noqa: BLE001 - as above
                    self._detection("failed")
                timings["language"] = (time.perf_counter() - t) * 1000
            wins, t_embed = self.engine.embed_clips(clips)
            raw, t_search = self.engine.search_windows(wins, FETCH, [track_id], **self._allow(query),
                                                       **self._within(query, detected))
            bases = raw.pop("base", None)
            if self.vibe:
                vibe = self._vibe_inputs(raw, t_search, windows=wins["sonic"])
                if clip22 is not None:
                    t = time.perf_counter()
                    bpm = self.tracker.tempo(clip22)[0]
                    vibe["bpm"] = bpm if np.isfinite(bpm) else None
                    t_search["tempo"] = (time.perf_counter() - t) * 1000
            consensus = self._consensus(query, bases, t_search, detected, wins.get("sonic"),
                                        vibe["bpm"] if vibe is not None else None)
        return query, raw, timings | t_embed, t_search, vibe, wins.get("sonic"), detected, consensus

    def _detection(self, outcome: str) -> None:
        with self._lock:
            self.detections[outcome] += 1

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

        def listed(rows) -> dict[Hashable, dict]:
            order: dict[Hashable, dict] = {}
            for tid, score in rows:
                artist, title = self.meta.get(tid, (None, None))
                key: Hashable = ("id", tid)
                if dedupe and title is not None:
                    key = song_key(artist, title)
                    if key == qkey or key in order:
                        continue
                order[key] = {"id": tid, "score": round(float(score), 4), "artist": artist, "title": title}
            return order

        for channel, rows in res["raw"].items():
            songs[channel] = listed(rows)
            out[channel] = list(songs[channel].values())[:k]
        sound_of = self._sound_of(res)
        exact: dict[int, float] = {}

        def floored(rows) -> dict[Hashable, dict]:
            """A blended ranking without the songs whose sound match is below ``main_match_floor``."""
            exact.update((int(tid), float(score)) for tid, score in rows)
            return {key: item for key, item in listed(rows).items()
                    if self.sound_match(sound_of.get(int(item["id"]), exact[int(item["id"])])) >= self.main_match_floor}

        def versions(lists: dict[str, dict[Hashable, dict]], ranking: dict[Hashable, dict] | None):
            """``_limit_versions`` for channel lists and the blended ranking that came with them."""
            if dedupe and qkey is not None and (self.copy_score is not None or self.max_versions is not None):
                limited = self._limit_versions(lists, qkey)
                if ranking:
                    ranking = self._limit_blend_versions(ranking, lists, limited, qkey)
                lists = limited
            return lists, ranking

        # The blended ranking (``main_pool`` "blend" and "sonic+blend"), filtered below like the channel lists; None:
        # not used.
        catalog = songs
        songs, blend = versions(songs, floored(res.get("blend") or ()) if "blend" in self.main_pool else None)

        language, origin = self._seed_language(q, res.get("detected"))
        guessed = origin == "guess"
        # "main": the guess gates main alone, below; everything else is as with the guess off.
        main_language = None
        if guessed and self.language_guess == "main":
            main_language, language, guessed, origin = language, "unknown", False, None
        # The in-language lists (``_compute``): the same-language lists are made from them, as they hold every song
        # the gate would leave of the catalog's lists and the language's next best ones. None: the gate filters the
        # catalog's lists (every language asked for, no gate language, or a search without them).
        inside = inside_blend = None
        if lang_mode == "same" and self.languages is not None and res.get("raw_lang") is not None \
                and res.get("lang") == (main_language or language):
            language_lists = {channel: listed(rows) for channel, rows in res["raw_lang"].items()}
            inside, inside_blend = versions(
                language_lists, floored(res.get("blend_lang") or ()) if "blend" in self.main_pool else None)

        scene_state = "off"
        artists: dict[Hashable, int | None] = {}
        profile = None
        if self.scene is not None and scene:
            for order in (*songs.values(), *(inside or {}).values()):
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
        filtered, blend_view = songs, blend
        if language != "unknown" and self.languages is not None:
            def keep(item: dict) -> bool:
                return self._item_language(item) in {"unknown", language}

            hidden = sum(
                1
                for key, _, _ in all_fused[:k]
                if not keep(next(songs[channel][key] for channel in songs if key in songs[channel]))
            )
            if lang_mode == "same":
                source, source_blend = (songs, blend) if inside is None else (inside, inside_blend)
                filtered = {
                    channel: {key: item for key, item in order.items() if keep(item)}
                    for channel, order in source.items()
                }
                if source_blend is not None:
                    blend_view = {key: item for key, item in source_blend.items() if keep(item)}
        query = q | {"language": language, "language_mode": lang_mode, "hidden": hidden, "scene": scene_state}
        if origin is not None:
            query["language_source"] = origin
        if guessed:
            query["language_guessed"] = True
        if inside is not None:
            query["language_pool"] = res["lang_pool"]
        # With the in-language lists, a main whose body is the blended ranking is filled up from the sonic list.
        fill = inside is not None and language != "unknown"
        if res.get("vibe") is not None and "sonic" in filtered:
            bpm = res["vibe"]["bpm"]
            query["bpm"] = round(bpm, 1) if bpm else None
        keep = (lambda item: self._item_language(item) in {"unknown", language}) \
            if lang_mode == "same" and language != "unknown" and self.languages is not None else None
        head_view = songs if self.head_all_languages and keep is not None else None
        # The consensus body (``_consensus``) replaces main's body, not with ``scene=0``: the songs of the asked
        # language mode, as song keys like the lists, without the seed's own song and artist, and with the seed
        # song's versions limited as in the lists main's head comes from.
        consensus = None
        if scene and self.consensus_quantiles is not None and res.get("consensus") is not None:
            consensus = {}
            for tid, score, ranks in res["consensus"][lang_mode]:
                for key, item in listed([(tid, score)]).items():
                    if key not in consensus and not (q.get("artist_id") and self.scene is not None
                                                     and self.scene.artist(tid) == q["artist_id"]):
                        consensus[key] = item | {
                            "consensus": item["score"], "match": sound_match(score, self.consensus_quantiles),
                            "ranks_all": dict(zip(CONSENSUS_SIGNALS, ranks))}
            if dedupe and qkey is not None and (self.copy_score is not None or self.max_versions is not None):
                before, after = (catalog, songs) if inside is None or lang_mode != "same" else \
                    (language_lists, inside)
                consensus = self._limit_blend_versions(consensus, before, after, qkey)
        if scene_state == "on" and res.get("listener") is not None:
            best, vibe, sounds, main, listeners = self._listener_lists(res, filtered, artists, profile, qkey, keep,
                                                                       language, lang_mode, k, dedupe, sound_of,
                                                                       blend_view, fill, head_view, consensus)
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
            main, listeners = self._main_without_listeners(filtered, profile, artists, k, sound_of, blend_view,
                                                           fill, head_view, keep, consensus), []
        if main_language is not None:
            query["main_language_guess"] = main_language
            if lang_mode == "same":
                main = self._guessed_main(res, songs if inside is None else inside, artists, profile, qkey,
                                          main_language, scene_state, k, dedupe, sound_of,
                                          blend if inside is None else inside_blend, inside is not None,
                                          songs if self.head_all_languages else None, consensus)
        lists = {"best": best} if best else {}
        if vibe:
            lists["vibe"] = vibe
        if sounds:
            lists["sounds"] = sounds
        if main:
            lists["main"] = main
        if listeners:
            lists["listeners"] = listeners
        for items in (*lists.values(), *out.values()):
            for item in items:
                score = sound_of.get(int(item["id"]))
                if score is not None and "consensus" not in item:  # a consensus row's match is its body score's
                    item["match"] = self.sound_match(score)
        if blend is not None:
            # Blended songs outside every other list: their exact sonic score came with the ranking.
            for item in main:
                if "match" not in item and int(item["id"]) in exact:
                    item["match"] = self.sound_match(exact[int(item["id"])])
        if self.main_pool != "blend" and self.main_overall is not None and main and "overall" not in main[0]:
            # Without listener data main follows sound alone, so its overall match is its sound match.
            head = next((i for i, item in enumerate(main) if item.get("basis") != "audio"), len(main))
            for item, value in zip(main, overall_match([item.get("match", 0) for item in main], head)):
                item["overall"] = value
        return {"ok": True, "query": query, "results": lists | out, "timings_ms": res["timings_ms"],
                "cached": cached}

    def _guessed_main(self, res: dict, songs: dict[str, dict[Hashable, dict]], artists: dict[Hashable, int | None],
                      profile, qkey, language: str, scene_state: str, k: int, dedupe: bool,
                      sound_of: Mapping[int, float], blend: dict[Hashable, dict] | None = None,
                      fill: bool = False, head_view: dict[str, dict[Hashable, dict]] | None = None,
                      consensus: dict[Hashable, dict] | None = None) -> list[dict]:
        """Main with the seed's guessed ``language`` gating it, built exactly as with ``language_guess`` True (the
        "main" mode keeps that list and nothing else). ``songs`` and ``blend``: the in-language lists (then with
        ``fill``), else the catalog's. ``head_view``: the catalog's lists for ``_cross_head``. ``consensus``: the
        consensus body, searched in that language already."""
        def keep(item: dict) -> bool:
            return self._item_language(item) in {"unknown", language}

        filtered = {channel: {key: item for key, item in order.items() if keep(item)}
                    for channel, order in songs.items()}
        if blend is not None:
            blend = {key: item for key, item in blend.items() if keep(item)}
        if scene_state == "on" and res.get("listener") is not None:
            return self._listener_lists(res, filtered, artists, profile, qkey, keep, language, "same", k, dedupe,
                                        sound_of, blend, fill, head_view, consensus)[3]
        return self._main_without_listeners(filtered, profile, artists, k, sound_of, blend, fill, head_view, keep,
                                            consensus)

    @staticmethod
    def _sound_of(res: dict) -> dict[int, float]:
        """Each candidate's sound (sonic) score against the seed: the sonic lists (the catalog's and the in-language
        one), then Same vibe's and the listener inputs, the later winning (what ``match`` shows)."""
        sound_of = {int(tid): float(score) for tid, score in (res.get("raw_lang") or {}).get("sonic", ())}
        sound_of |= {int(tid): float(score) for tid, score in res["raw"].get("sonic", ())}
        sound_of |= (res.get("vibe") or {}).get("muq") or {}
        sound_of |= (res.get("listener") or {}).get("son") or {}
        return sound_of

    def _match(self, sound_of: Mapping[int, float], item: dict) -> int:
        """``match`` of a listed song, 0 without a sound score."""
        score = sound_of.get(int(item["id"]))
        return self.sound_match(score) if score is not None else 0

    def _listener_lists(self, res: dict, filtered: dict[str, dict[Hashable, dict]], artists: dict[Hashable, int | None],
                        profile, qkey, keep: Callable[[dict], bool] | None, language: str, lang_mode: str, k: int,
                        dedupe: bool, sound_of: Mapping[int, float], blend: dict[Hashable, dict] | None = None,
                        fill: bool = False, head_view: dict[str, dict[Hashable, dict]] | None = None,
                        consensus: dict[Hashable, dict] | None = None
                        ) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict]]:
        """Best matches, Same vibe, Sounds like, main and Listeners also play with the listener ranker
        (``res["listener"]``, from ``_listener``).

        Best: the audio head (``scene.audio_head``, likely covers and samples), then every other candidate, the channel
        lists' songs and the scene candidates, by z(listener logit) + ``best_sound_weight`` * z(sound score), then the
        per-artist caps. Sounds like: the sonic list by z(sound score) + ``sounds_listener_weight`` * z(listener logit),
        one song per artist. Same vibe: ``samevibe.same_vibe`` over those two orders. Main: the audio head (with
        ``main_head_bare``, ``main_head_standout`` and ``main_head_match``), then the ``main_pool`` songs by z(sound
        score) + ``main_listener_weight`` * z(listener logit), one per artist. Listeners also play: Best's candidates
        by other artists than the seed's by z(listener logit) + ``listeners_sound_weight`` * z(sound score), one per
        artist. Songs by the seed's own artist get the median listener logit, so sound alone places them. With
        ``blend`` (``main_pool`` "blend": the blended ranking, filtered like the channel lists), main is the same head,
        then those songs in their order, one per artist, without ``overall``. With ``main_pool`` "sonic+blend" they join
        the sonic list's songs instead, and the order adds ``main_blend_weight`` * each song's blended score. With
        ``consensus`` (the consensus body's songs by key, in its order) main is the same head, then those songs
        (``basis`` "consensus"), one per artist; its ``overall`` then comes from each row's ``match`` (``_view``).
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

        head = [key for key in self._audio_head(filtered, self.best_head_bare) if key in shown]
        # ``head_view`` (``head_all_languages``): other-language songs with evidence join the heads; they are in no
        # body, as the gate keeps them out of every candidate list.
        listed = set(shown)
        if head_view is not None:
            head, cross = self._cross_head(head_view, head, keep, listed)
            shown |= cross
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
        head_set = set(head)
        best = [shown[key][0] | {"score": round(1.0 / (FUSE_K + i), 5), "ranks": shown[key][1],
                                 "basis": "audio" if key in head_set else "listeners"}
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

        if self.main_head_bare == self.best_head_bare and not self.main_head_standout:
            main_head = head
        else:
            main_head = [key for key in self._audio_head(
                filtered, self.main_head_bare, lambda key: self._match(sound_of, shown[key][0]) if key in shown else 0)
                if key in listed]
            if head_view is not None:
                main_head, cross = self._cross_head(head_view, main_head, keep, listed, sound_of)
                shown |= cross

        blend_body = blend is not None and self.main_pool == "blend"
        blend_only: dict[Hashable, dict] = {}
        if blend is not None and self.main_pool == "sonic+blend":
            blend_only = {key: item for key, item in blend.items() if key not in sonic}
            for key, item in blend_only.items():
                artists.setdefault(key, self.scene.artist(int(item["id"])))

        def main_item(key: Hashable) -> dict:
            return sonic[key] if key in sonic else shown[key][0] if key in shown else blend_only[key]

        def main_ranks(key: Hashable) -> dict[str, int]:
            return shown[key][1] if key in shown else {}

        def shown_item(key: Hashable) -> dict:
            return shown[key][0]

        pool = [] if blend_body else list(sonic) if self.main_pool == "sonic" else [*sonic, *blend_only] \
            if self.main_pool == "sonic+blend" else [
            key for key in keys
            if key in sonic or self.sound_match(son_of.get(int(shown[key][0]["id"]), -np.inf)) >= self.main_match_floor]
        raw_main: dict[int, tuple[float, float]] = {}
        neutral = 0.0
        if pool:
            pool_sound, pool_logit = sound(pool, main_item), logits(pool, main_item)
            raw_main = {int(main_item(key)["id"]): (float(s), float(lg))
                        for key, s, lg in zip(pool, pool_sound, pool_logit)}
            neutral = float(np.median(pool_logit))
            if self.main_scale is None:
                score = listener.blend(pool_sound, pool_logit, self.main_listener_weight)
            else:
                score = (pool_sound / self.main_scale[0]
                         + self.main_listener_weight * pool_logit / self.main_scale[1])
            if blend is not None and self.main_blend_weight:
                # Each song's blended score (its own release's, else that of the release the blended ranking holds);
                # a song without one gets the pool's lowest.
                blendz = res.get("blendz") or {}
                z = np.array([blendz.get(int(main_item(key)["id"]),
                                         blendz.get(int(blend[key]["id"]), np.nan) if key in blend else np.nan)
                              for key in pool], dtype=np.float64)
                score = score + self.main_blend_weight * np.nan_to_num(
                    z, nan=np.nanmin(z) if np.isfinite(z).any() else 0.0)
            pool = ordered(pool, score)
        main_set = set(main_head)
        body = [(main_item(key), main_ranks(key)) for key in pool if key not in main_set] if not blend_body else \
            [(item, main_ranks(key)) for key, item in blend.items() if key not in main_set]
        if blend_body and fill:
            # ``fill`` (the in-language lists): the sonic list's other songs, in its order, follow the blended ones,
            # so the match floor does not leave main short while the language has songs.
            body += [(item, main_ranks(key)) for key, item in sonic.items() if key not in main_set and key not in blend]
        if consensus is not None:
            body = [(item, main_ranks(key)) for key, item in consensus.items() if key not in main_set]
        main = self._main([(shown[key][0], shown[key][1]) for key in main_head], body, k,
                          "sound" if consensus is None else "consensus")
        if self.main_overall is not None and main and not blend_body and consensus is None:
            floor = min((s for s, _ in raw_main.values()), default=0.0)
            for key in main_head:
                tid = int(shown[key][0]["id"])
                p = p_of.get(tid)
                plain = p is None or artists.get(key) == profile.artist_id
                raw_main.setdefault(tid, (float(son_of.get(tid, floor)),
                                          neutral if plain else float(listener.logit(p))))
            self._set_overall(main, raw_main, self.main_overall, 1.0, self.main_listener_weight)

        others = [key for key in keys if artists.get(key) != profile.artist_id]
        raw_listeners: dict[int, tuple[float, float]] = {}
        if others:
            others_logit, others_sound = logits(others, shown_item), sound(others, shown_item)
            raw_listeners = {int(shown_item(key)["id"]): (float(s), float(lg))
                             for key, s, lg in zip(others, others_sound, others_logit)}
            others = ordered(others, listener.zscore(others_logit)
                             + self.listeners_sound_weight * listener.zscore(others_sound))
        picks = self._one_per_artist(
            [(shown[key][0], {"ranks": shown[key][1], "basis": "listeners"}) for key in others], k)
        listeners = [item | {"score": round(1.0 / (FUSE_K + i), 5)} for i, item in enumerate(picks, 1)]
        if self.listeners_overall is not None and listeners:
            self._set_overall(listeners, raw_listeners, self.listeners_overall, self.listeners_sound_weight, 1.0)
        return best, vibe, sounds, main, listeners

    def _set_overall(self, items: list[dict], raw: Mapping[int, tuple[float, float]], scale: OverallScale,
                     sound_weight: float, listener_weight: float) -> None:
        """Each listed song's ``overall`` (``OverallScale``, ``overall_match``) from its (sound score, listener logit)
        in ``raw``, the values its list was ordered by."""
        blend = np.array([sound_weight * raw[int(item["id"])][0] / scale.sound_sd
                          + listener_weight * raw[int(item["id"])][1] / scale.logit_sd for item in items])
        head = next((i for i, item in enumerate(items) if item.get("basis") != "audio"), len(items))
        for item, value in zip(items, overall_match(self._overall_percentile(blend, scale), head)):
            item["overall"] = value

    @staticmethod
    def _overall_percentile(blend: np.ndarray, scale: OverallScale) -> list[float]:
        return np.interp(blend, scale.quantiles, np.linspace(0, 100, len(scale.quantiles))).tolist()

    def _cross_head(self, ungated: dict[str, dict[Hashable, dict]], head: list[Hashable],
                    keep: Callable[[dict], bool], listed: set[Hashable], sound_of: Mapping[int, float] | None = None
                    ) -> tuple[list[Hashable], dict[Hashable, tuple[dict, dict[str, int]]]]:
        """An audio ``head`` of gated lists plus the other-language songs with cover or sample evidence
        (``head_all_languages``): the songs of the audio head of the ``ungated`` lists (the catalog's, as every language
        shows them) under the same evidence and standout rules, evidence songs alone and none for a bare head (with
        ``sound_of``: main's rules, else Best's), that the gate removes (``keep``) and that are not ``listed``. They
        stand where that head has them among the given head's songs. Returns the head's keys and each added song's
        (item with ``other_language``, channel ranks in the ungated lists)."""
        pos = {channel: {key: i for i, key in enumerate(order, 1)} for channel, order in ungated.items()}

        def entry(key: Hashable) -> tuple[dict, dict[str, int]]:
            ranks = {channel: p[key] for channel, p in pos.items() if key in p}
            return ungated[min(ranks, key=lambda name: (ranks[name], name))][key], ranks

        evident = self._audio_head(ungated, 0, None if sound_of is None else
                                   (lambda key: self._match(sound_of, entry(key)[0])), evidence_only=True)
        cross = {}
        for key in evident:
            item, ranks = entry(key)
            # Evidence in one list alone is right too rarely across languages (CROSS_HEAD_CHANNELS).
            if key not in listed and not keep(item) and len(ranks) >= CROSS_HEAD_CHANNELS:
                cross[key] = (item | {"other_language": True}, ranks)
        if not cross:
            return head, {}
        at = {key: i for i, key in enumerate(evident)}
        place = {key: i for i, key in enumerate(head)}
        # An added song goes after the last of the head's songs that the ungated head has before it.
        slot = {key: max((place[g] + 1 for g in head if at.get(g, len(at)) < at[key]), default=0) for key in cross}
        return sorted([*cross, *head], key=lambda key: (slot[key], 0, at[key]) if key in cross
                      else (place[key], 1, 0)), cross

    def _audio_head(self, view: dict[str, dict[Hashable, dict]], bare: int,
                    match_of: Callable[[Hashable], float] | None = None, evidence_only: bool | None = None
                    ) -> list[Hashable]:
        """Best's audio head (``scene.audio_head``) over the filtered channel lists, ``bare`` songs long when none of
        them has cover or sample evidence. With ``match_of`` (a song's sound match; main's head), a song whose match is
        below ``main_head_match`` also needs a standout z-score of at least ``main_head_standout``. ``evidence_only``
        overrides ``best_head_evidence_only``."""
        orders = {channel: list(order) for channel, order in view.items()}
        scores = {channel: [item["score"] for item in order.values()] for channel, order in view.items()}
        params = replace(self.scene_params, head_bare=bare, head_evidence_only=self.best_head_evidence_only
                         if evidence_only is None else evidence_only)
        head = audio_head(orders, scores, params, standout_min=self.best_head_standout,
                          standout_window=self.best_head_window)
        if match_of is None or not self.main_head_standout:
            return head
        z = standout(orders, scores, self.best_head_window)
        return [key for key in head
                if z.get(key, -np.inf) >= self.main_head_standout or match_of(key) >= self.main_head_match]

    def _main(self, head: Sequence[tuple[dict, dict[str, int]]], body: Sequence[tuple[dict, dict[str, int]]],
              k: int, basis: str = "sound") -> list[dict]:
        """The main list from ``(item, channel ranks)`` pairs: the head's songs (``basis`` "audio"), then the body's
        (``basis``), one song per artist."""
        entries = [(item, {"ranks": ranks, "basis": "audio"}) for item, ranks in head]
        entries += [(item, {"ranks": ranks, "basis": basis}) for item, ranks in body]
        return [item | {"score": round(1.0 / (FUSE_K + i), 5)}
                for i, item in enumerate(self._one_per_artist(entries, k), 1)]

    def _main_without_listeners(self, view: dict[str, dict[Hashable, dict]], profile,
                                artists: Mapping[Hashable, int | None], k: int,
                                sound_of: Mapping[int, float], blend: dict[Hashable, dict] | None = None,
                                fill: bool = False, head_view: dict[str, dict[Hashable, dict]] | None = None,
                                keep: Callable[[dict], bool] | None = None,
                                consensus: dict[Hashable, dict] | None = None) -> list[dict]:
        """The main list without listener data (``scene=0``, an unknown artist or no ranker): the audio head, then
        Sounds like's order, or the ``blend`` songs in theirs (``main_pool`` "blend" only; with ``fill``, the sonic
        list's other songs after them), or the ``consensus`` body's songs (as in ``_listener_lists``). ``head_view``
        and ``keep``: as for ``_cross_head``."""
        pos = {channel: {key: i for i, key in enumerate(order, 1)} for channel, order in view.items()}

        def ranks(key: Hashable) -> dict[str, int]:
            return {channel: p[key] for channel, p in pos.items() if key in p}

        def item_of(key: Hashable) -> dict:
            r = ranks(key)
            return view[min(r, key=lambda name: (r[name], name))][key]

        head_keys = self._audio_head(view, self.main_head_bare, lambda key: self._match(sound_of, item_of(key)))
        head = [(item_of(key), ranks(key)) for key in head_keys]
        if head_view is not None:
            head_keys, cross = self._cross_head(head_view, head_keys, keep,
                                                {key for order in view.values() for key in order}, sound_of)
            head = [cross[key] if key in cross else (item_of(key), ranks(key)) for key in head_keys]
        head_set = set(head_keys)
        blend_body = blend is not None and self.main_pool == "blend"
        order = list(blend.items()) if blend_body else self._sound_order(view, profile, artists)
        if blend_body and fill:  # as in ``_listener_lists``
            order += [(key, item) for key, item in view.get("sonic", {}).items() if key not in blend]
        if consensus is not None:
            order = list(consensus.items())
        body = [(item, ranks(key)) for key, item in order if key not in head_set]
        return self._main(head, body, k, "sound" if consensus is None else "consensus")

    def _seed_language(self, q: dict, detected: str | None = None) -> tuple[str, str | None]:
        """The seed's language for the gate and where it comes from: its own "label", else the one "detected" in a live
        seed's audio, else its "artist"'s majority language, else (with ``language_guess``) the table's "guess";
        "unknown" (no filter) and None without any. A detected language passed the rule a song's own label passes
        (``languages.LanguageDetector``), so it stands where that label would: before the artist's majority, which is
        there for songs whose own audio does not say; an unsure detection is "unknown" and leaves it to the artist."""
        if self.languages is None:
            return "unknown", None
        language = self.languages.language(int(q["id"]))
        if language != "unknown":
            return language, "label"
        if detected and detected != "unknown":
            return detected, "detected"
        language = self.languages.artist_language(q.get("artist"))
        if language != "unknown":
            return language, "artist"
        if self.language_guess:
            language = self.languages.guess(int(q["id"]), q.get("artist"))
            if language != "unknown":
                return language, "guess"
        return "unknown", None

    def _item_language(self, item: dict) -> str:
        """A listed song's language: its own label, else (with ``artist_languages``) its artist's majority language."""
        language = self.languages.language(int(item["id"]))
        if language == "unknown" and self.artist_languages:
            language = self.languages.artist_language(item.get("artist"))
        return language

    @staticmethod
    def _is_version(key: Hashable, qkey: tuple[str, str]) -> bool:
        """Whether a song key names another artist's version of the seed's own song."""
        return isinstance(key, tuple) and key[0] != "id" and key[1] == qkey[1] and key[0] != qkey[0]

    def _limit_blend_versions(self, blend: dict[Hashable, dict], songs: dict[str, dict[Hashable, dict]],
                              limited: dict[str, dict[Hashable, dict]], qkey: tuple[str, str]) -> dict[Hashable, dict]:
        """The blended ranking under ``_limit_versions``: a version the channel lists (``songs``) dropped (``limited``
        is what they kept) is dropped here too, and versions only this ranking holds fill what is left of
        ``max_versions``."""
        listed = {key for order in songs.values() for key in order if self._is_version(key, qkey)}
        kept = {key for order in limited.values() for key in order if self._is_version(key, qkey)}
        room = None if self.max_versions is None else max(0, self.max_versions - len(kept))
        out: dict[Hashable, dict] = {}
        for key, item in blend.items():
            if self._is_version(key, qkey) and key not in kept:
                if key in listed or room == 0:
                    continue
                if room is not None:
                    room -= 1
            out[key] = item
        return out

    def _limit_versions(self, songs: dict[str, dict[Hashable, dict]], qkey: tuple[str, str]
                        ) -> dict[str, dict[Hashable, dict]]:
        """Other artists' versions of the seed's own song (same ``song_key`` title, another artist). With
        ``copy_score``, those scoring at least that in the samples list are dropped as copies of the seed's recording;
        with ``max_versions``, only that many of the rest stay (the best placed in any channel), so covers and edits of
        one song cannot fill the lists. The channel lists themselves are left whole."""
        best: dict[Hashable, int] = {}
        for order in songs.values():
            for i, key in enumerate(order, 1):
                if self._is_version(key, qkey):
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
        return self._one_per_artist([(item, {}) for _, item in self._sound_order(view, profile, artists, order)], k)

    def _sound_order(self, view: dict[str, dict[Hashable, dict]], profile=None,
                     artists: Mapping[Hashable, int | None] | None = None, order: Sequence[Hashable] | None = None
                     ) -> list[tuple[Hashable, dict]]:
        """Sounds like's songs in order, before one song per artist is kept (see ``_sounds``)."""
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
        return items

    def _one_per_artist(self, entries: Sequence[tuple[dict, dict]], k: int) -> list[dict]:
        """The first song of each artist among ``(item, fields)`` entries, in order, up to ``k``: each item with those
        fields and its catalog tempo. An artist is the scene table's Deezer artist id, else the credited name."""
        seen: set[Hashable] = set()
        out = []
        for item, fields in entries:
            tid = int(item["id"])
            artist_id = self.scene.artist(tid) if self.scene is not None else None
            artist: Hashable = ("id", artist_id) if artist_id else ("name", (item["artist"] or "").casefold() or tid)
            if artist in seen:
                continue
            seen.add(artist)
            bpm = self.tempo.bpm(tid) if self.tempo is not None else None
            out.append(item | fields | {"bpm": round(bpm, 1) if bpm else None})
            if len(out) == k:
                break
        return out

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


def _overall_scales(path: Path | None) -> dict:
    """``main_overall`` and ``listeners_overall`` from an ``--overall`` JSON file; empty (the defaults) without one."""
    if path is None:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {f"{name}_overall": OverallScale(float(data[name]["sound_sd"]), float(data[name]["logit_sd"]),
                                            tuple(float(q) for q in data[name]["quantiles"]))
            for name in ("main", "listeners")}


def main(argv: list[str] | None = None) -> None:
    import torch

    from .deezer import Deezer
    from .serve import CHANNELS, Engine, load_meta

    p = argparse.ArgumentParser(prog="anysong-api")
    p.add_argument("--index", required=True, help="index_build output or a shard directory (see serve)")
    p.add_argument("--meta", type=Path, default=None, help="catalog_meta npz with track_ids/artists/titles")
    p.add_argument("--languages", type=Path, default=None, help="language labels from anysong.languages")
    p.add_argument("--language-model", type=Path, default=None,
                   help="local snapshot of the Whisper model that labelled the catalog (anysong.languages --model-dir): "
                        "the language of songs outside the catalog is detected from their preview (needs --languages)")
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
    p.add_argument("--head", type=Path, default=None,
                   help="scene head (anysong.scenehead): path prefix of <prefix>.npz and <prefix>.emb.npy; for rankers "
                        "trained with the 'head' feature")
    p.add_argument("--popularity", type=Path, default=None,
                   help="npz with ids (ascending Deezer track ids) and rank (Deezer popularity; absent songs count "
                        "as 0): for rankers trained with the 'pop' feature and for the blend's popularity weight")
    p.add_argument("--match-quantiles", type=Path, default=None,
                   help="JSON list of the 21 sonic-score quantiles (0, 5, ..., 100 %%) behind 'match', for an index "
                        "whose sonic scores MATCH_QUANTILES was not fitted on")
    p.add_argument("--main-pool", choices=("sonic", "all", "blend", "sonic+blend"), default=MAIN_POOL,
                   help="main's body: the sonic list, every candidate at or above --main-match-floor, the blended "
                        "catalog ranking at or above it, or the sonic list plus that ranking in the listener order "
                        "(the last two need --blend-head and --blend-cf)")
    p.add_argument("--blend-head", type=Path, default=None,
                   help="scene head of the blended ranking (anysong.blend): path prefix as for --head")
    p.add_argument("--blend-cf", type=Path, default=None,
                   help="folder with cooc.npz and cf128.npy: playlist co-occurrence vectors of the blended ranking")
    p.add_argument("--blend-weights", type=float, nargs="+", default=None, metavar="W",
                   help="WH WC [WP]: weights of the scene head, playlist and popularity terms next to sound "
                        "(default 2 1 0; WP needs --popularity and never reaches lists without the listener ranker)")
    p.add_argument("--consensus-model", type=Path, default=None,
                   help="consensus model (anysong.consensus export): main's body becomes the consensus body; needs "
                        "--blend-head, --blend-cf, --consensus-quantiles and the three channels as built indexes")
    p.add_argument("--consensus-sound-weight", type=float, default=None,
                   help="body score = prediction - this * min(log10 sound rank, 5) (default 0.1)")
    p.add_argument("--consensus-sound-cap", type=int, default=None,
                   help="songs past this sound rank are in no consensus body (default 5000)")
    p.add_argument("--consensus-quantiles", type=Path, default=None,
                   help="JSON list of the 21 body-score quantiles (0, 5, ..., 100 %%) behind a consensus row's "
                        "'match'")
    p.add_argument("--main-blend-weight", type=float, default=MAIN_BLEND_WEIGHT,
                   help="--main-pool sonic+blend: weight of a song's blended score in main's order")
    p.add_argument("--main-match-floor", type=float, default=MAIN_MATCH_FLOOR)
    p.add_argument("--head-all-languages", action="store_true",
                   help="same-language lists: the audio head of main and of Best matches also takes songs in other "
                        "languages that have cover or sample evidence (each marked other_language)")
    p.add_argument("--main-listener-weight", type=float, default=MAIN_LISTENER_WEIGHT)
    p.add_argument("--main-scale", type=float, nargs=2, default=None, metavar=("SOUND_SD", "LOGIT_SD"),
                   help="order main's body on these fixed scales instead of per-seed z-scores")
    p.add_argument("--overall", type=Path, default=None,
                   help="JSON with 'main' and 'listeners' overall scales (sound_sd, logit_sd, quantiles) for a ranker "
                        "other than the one MAIN_OVERALL and LISTENERS_OVERALL were calibrated on")
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
    if args.language_model is not None and args.languages is None:
        p.error("--language-model needs --languages")
    if args.ranker is not None and args.scene is None:
        p.error("--ranker needs --scene")
    if (args.blend_head is None) != (args.blend_cf is None):
        p.error("--blend-head and --blend-cf go together")
    if "blend" in args.main_pool and args.blend_head is None:
        p.error(f"--main-pool {args.main_pool} needs --blend-head and --blend-cf")
    if args.blend_head is not None and "sonic" not in args.channels:
        p.error("--blend-head needs the sonic channel")
    if args.blend_weights is not None and len(args.blend_weights) not in (2, 3):
        p.error("--blend-weights takes WH WC or WH WC WP")
    blend_wp = args.blend_weights[2] if args.blend_weights is not None and len(args.blend_weights) == 3 else 0.0
    if blend_wp and args.popularity is None:
        p.error("a popularity weight in --blend-weights needs --popularity")
    if blend_wp and args.main_pool == "blend":
        p.error("--main-pool blend cannot take a popularity weight (it would reach audio-only lists): "
                "use sonic+blend")
    consensus_quantiles = None
    if args.consensus_model is not None:
        if args.blend_head is None or not {"melody", "samples", "sonic"} <= set(args.channels):
            p.error("--consensus-model needs --blend-head, --blend-cf and the melody, samples and sonic channels")
        if args.consensus_quantiles is None:
            p.error("--consensus-model needs --consensus-quantiles")
        if not (Path(args.index) / "sonic" / "full.npy").exists():
            p.error("--consensus-model needs a built sonic index (its full.npy)")
        try:
            consensus_quantiles = checked_match_quantiles(
                json.loads(args.consensus_quantiles.read_text(encoding="utf-8")))
        except ValueError as exc:
            p.error(f"--consensus-quantiles {args.consensus_quantiles}: {exc}")
    elif any(v is not None for v in (args.consensus_sound_weight, args.consensus_sound_cap,
                                     args.consensus_quantiles)):
        p.error("--consensus-sound-weight, --consensus-sound-cap and --consensus-quantiles need --consensus-model")
    popularity = listener.Popularity.load(args.popularity) if args.popularity is not None else None
    quantiles = None
    if args.match_quantiles is not None:
        try:
            quantiles = checked_match_quantiles(json.loads(args.match_quantiles.read_text(encoding="utf-8")))
        except ValueError as exc:
            p.error(f"--match-quantiles {args.match_quantiles}: {exc}")

    scratch = args.preview_dir or Path(tempfile.mkdtemp(prefix="anysong-"))
    t = time.perf_counter()
    engine = Engine(args.index, args.channels, args.device, candidates=args.candidates, hub_alpha=args.hub_alpha,
                    half_stage1=args.half_stage1)
    if args.blend_head is not None:
        from .blend import WEIGHTS, Blend

        t_blend = time.perf_counter()
        sonic = engine.indexes["sonic"]
        engine.blend = Blend.load(args.blend_head, args.blend_cf, sonic.pos, sonic.id_tensor.device,
                                  tuple(args.blend_weights) if args.blend_weights else WEIGHTS,
                                  popularity.of(sonic.ids) if blend_wp else None)
        engine.load_secs["blend"] = round(time.perf_counter() - t_blend, 2)
    engine.warm()
    warm = _warmup(engine)
    meta = load_meta(args.meta)
    languages = language_rows = None
    if args.languages is not None:
        from .languages import LanguageRows, LanguageTable

        languages = LanguageTable.load(
            args.languages,
            artist_names={track_id: artist for track_id, (artist, _) in meta.items()},
        )
        t_rows = time.perf_counter()
        language_rows = LanguageRows(languages, engine.indexes, lambda track_id: meta.get(track_id, (None,))[0])
        engine.load_secs["language_rows"] = round(time.perf_counter() - t_rows, 2)
    detector = None
    if args.language_model is not None:
        from .languages import EXCERPT_SECONDS, SAMPLE_RATE, LanguageDetector

        t_detector = time.perf_counter()
        detector = LanguageDetector.load(args.language_model, args.device)
        detector.detect(np.zeros(int(SAMPLE_RATE * EXCERPT_SECONDS), np.float32))  # warm-up
        engine.load_secs["language_model"] = round(time.perf_counter() - t_detector, 2)
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
    consensus = None
    if args.consensus_model is not None:
        from .consensus import SOUND_CAP, SOUND_WEIGHT, Consensus

        t_consensus = time.perf_counter()
        sonic_ids = engine.indexes["sonic"].ids.tolist()
        consensus = Consensus.load(
            args.consensus_model, Path(args.index) / "sonic" / "full.npy", args.blend_cf / "cf128.npy", engine,
            artists=[meta[tid][0] if tid in meta else None for tid in sonic_ids],
            codes=None if language_rows is None else language_rows.codes["sonic"],
            bpm=None if tempo is None else np.array([tempo.by_id.get(tid, np.nan) for tid in sonic_ids], np.float32),
            scene=scene,
            sound_weight=SOUND_WEIGHT if args.consensus_sound_weight is None else args.consensus_sound_weight,
            sound_cap=SOUND_CAP if args.consensus_sound_cap is None else args.consensus_sound_cap)
        engine.load_secs["consensus"] = round(time.perf_counter() - t_consensus, 2)
    service = AnySong(engine, Deezer(scratch), meta, args.cache_size, args.max_pending,
                      languages=languages, scene=scene, tempo=tempo, tracker=tracker,
                      learn=LearnQueue(args.learn_queue) if args.learn_queue else None, tiers=tiers,
                      language_rows=language_rows, language_detector=detector,
                      ranker=listener.Ranker.load(args.ranker) if args.ranker is not None else None,
                      head=SceneHead(f"{args.head}.npz", f"{args.head}.emb.npy", engine.indexes["sonic"].pos)
                      if args.head is not None else None,
                      main_pool=args.main_pool, main_match_floor=args.main_match_floor,
                      head_all_languages=args.head_all_languages, popularity=popularity, match_quantiles=quantiles,
                      main_listener_weight=args.main_listener_weight, main_blend_weight=args.main_blend_weight,
                      main_scale=tuple(args.main_scale) if args.main_scale else None,
                      stems=store, dissector=dissector, catalog_previews=args.catalog_previews,
                      consensus=consensus, consensus_quantiles=consensus_quantiles,
                      **_overall_scales(args.overall))
    server = Server((args.host, args.port), service, args.search_per_min, args.similar_per_min,
                    args.cors_origin, args.client_ip_header, dissect_per_min=args.dissect_per_min)
    sizes = {name: len(ix) for name, ix in engine.indexes.items()}
    hubs = {name: getattr(ix, "hub_alpha", 0.0) for name, ix in engine.indexes.items()}
    print(f"ready in {time.perf_counter() - t:.1f}s (warmup {warm:.1f}s) {sizes} hub_alpha={hubs} "
          f"tiers={None if tiers is None else tiers.sizes} ranker={service.health()['ranker']} "
          f"language_search={engine.load_secs.get('language_rows')}s "
          f"language_model={engine.load_secs.get('language_model')}s "
          f"stems={service.health()['stems']} (warmup {stems_warm}s) blend={service.health()['blend']} "
          f"consensus={service.health()['consensus']} ({engine.load_secs.get('consensus')}s) "
          f"gpu={_gpu_mib(args.device)} "
          f"on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        if args.preview_dir is None:
            shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
