"""Grow the any-song catalog along Deezer's related-artist graph.

    python -m soundalike.anysong.crawl run --scene /opt/soundalike/index/scene.npz \
        --meta /opt/soundalike/index/catalog_meta.npz --db /data/crawl/crawl.sqlite \
        --previews /data/previews --target 730000
    python -m soundalike.anysong.crawl status --db /data/crawl/crawl.sqlite
    python -m soundalike.anysong.crawl export --db /data/crawl/crawl.sqlite --out /data/crawl/export

The scene table supplies the starting point: the catalog's artists and every artist their related lists link to.
Catalog artists are crawled first; that tops them up with popular songs the catalog lacks, and their related lists
give fan counts for the artists around them. The rest of the frontier then goes most-followed first (``--min-fans``
skips the smallest). Ordering by links instead would favour dense scenes, such as classical performers who all
list each other. Each crawled artist costs one top-tracks and one related-artists request (up to ``--fetchers``
in flight); its related list adds the next ring of artists to the frontier.

From an artist's own top tracks, in Deezer's popularity order, the crawl keeps songs until the artist has 8 in all
at 100 fans, 12 at 1,000, 16 at 10,000 and so on up to ``--per-artist``, counting the songs it already has in the
catalog. A song needs a playable preview, one to fifteen minutes of audio and a minimum Deezer rank. Karaoke,
tribute, sped-up, slowed, 8D, nightcore, frequency, instrumental and a cappella uploads are left out, as are
artists named like karaoke, sleep or study-music channels. Each song is kept once: live, edit, version and
remaster variants share the original's key (remixes and named mixes stay distinct), songs already in the catalog
are skipped, and a catalogue-numbered classical work (Op., BWV, K., RV, ...) is kept for its first performer only.

Signed preview links expire within about fifteen minutes, so previews download right after their artist's
metadata, and the crawl pauses while too many downloads are pending. The preview CDN rate-limits each IP at
roughly ten downloads a second, so the pace starts at ``--preview-rate``, drops by a fifth after each burst of
rate limiting and recovers by a tenth per quiet half minute. Failed downloads get one more pass at the end.
Everything is recorded in SQLite: stop it (SIGTERM or Ctrl-C) and rerun the same command to resume; previews that
were still pending get fresh links.

``export`` writes ``catalog_meta.npz`` (the API's track_ids/artists/titles layout), ``ids.txt``, and the scene
builder's ``deezer_tracks.jsonl`` and ``deezer_related.jsonl`` for the new songs and every artist whose related
list was fetched.

The public Deezer API is used for this personal, non-commercial project at a polite rate (``--rate``, default
4 requests/s of Deezer's ~10/s per IP, leaving room for the live API).
"""

from __future__ import annotations

import argparse
import errno
import json
import math
import queue
import re
import shutil
import signal
import sqlite3
import threading
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Callable, Collection, Iterable, Sequence

import numpy as np

from .deezer import Deezer, RateLimiter, base_title, normalize
from .scene import SceneTable

_DECORATION = re.compile(r"\(([^()]*)\)|\[([^\[\]]*)\]|\s+-\s+([^()\[\]]+?)\s*$")
_ALWAYS_JUNK = re.compile(r"karaoke|made famous|in the style of|nightcore|\b\d{3,4}\s?hz\b|binaural", re.I)
_JUNK_DECORATION = re.compile(
    r"\btribute\b|\b8d\b|slowed|sped up|speed up|reverb|instrumental|a ?cap+el+a|backing track|lullaby|ringtone",
    re.I)
_JUNK_ARTIST = re.compile(
    r"karaoke|\btribute\b|made famous|in the style of|lullab|white noise|\brain sounds?\b|"
    r"\bsleep (music|sounds|aid|therapy)\b|\b(relaxing|relaxation|meditation|study|workout|yoga|spa) music\b|"
    r"ringtone|\b8d audio\b|nightcore|nursery rhyme|\b\d{3,4}\s?hz\b|binaural", re.I)
_PLAIN_MIX = re.compile(r"\b(original|extended|radio|club|album|single|main|short|long|full)\s+mix\b", re.I)
_DISTINCT = re.compile(r"remix|\brmx\b|bootleg|\bflip\b|\bvip\b|rework|\bmix\b", re.I)
_SAME_RECORDING = re.compile(
    r"\blive\b|\bvivo\b|\bedit\b|\bversion\b|\bremaster|\bexplicit\b|\bclean\b|\bmono\b|\bstereo\b|\bbonus\b|"
    r"\bdemo\b|\bacoustic\b|\bunplugged\b|\bsessions?\b|\bfrom\b|\bprod\b", re.I)
_WORK_NUMBER = re.compile(r"\b(?:op|bwv|kv|rv|hob|woo|hwv|sz)\.?\s*[ivx\d]|\b[kd]\.\s*\d", re.I)

SCHEMA = """
CREATE TABLE IF NOT EXISTS artists (
    id INTEGER PRIMARY KEY, name TEXT, hop INTEGER NOT NULL, links INTEGER NOT NULL DEFAULT 0,
    fans INTEGER NOT NULL DEFAULT -1, status TEXT NOT NULL, linked INTEGER NOT NULL DEFAULT 0,
    songs INTEGER NOT NULL DEFAULT 0, attempts INTEGER NOT NULL DEFAULT 0, crawled_at REAL, error TEXT);
CREATE INDEX IF NOT EXISTS artists_next ON artists (status, fans DESC, links DESC, id);
CREATE INDEX IF NOT EXISTS artists_hop ON artists (status, hop, id);
CREATE TABLE IF NOT EXISTS related (
    artist_id INTEGER NOT NULL, rel_id INTEGER NOT NULL, position INTEGER NOT NULL,
    PRIMARY KEY (artist_id, rel_id));
CREATE TABLE IF NOT EXISTS tracks (
    id INTEGER PRIMARY KEY, artist_id INTEGER NOT NULL, artist TEXT NOT NULL, title TEXT NOT NULL,
    song_key TEXT NOT NULL UNIQUE, rank INTEGER, duration INTEGER, album_id INTEGER,
    status TEXT NOT NULL DEFAULT 'queued', bytes INTEGER NOT NULL DEFAULT 0,
    attempts INTEGER NOT NULL DEFAULT 0, error TEXT, added_at REAL);
CREATE INDEX IF NOT EXISTS tracks_status ON tracks (status);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT);
"""


def _inner(match: re.Match) -> str:
    return next(group for group in match.groups() if group is not None)


def _same_recording(decoration: str) -> bool:
    if _PLAIN_MIX.search(decoration):
        return True
    return not _DISTINCT.search(decoration) and bool(_SAME_RECORDING.search(decoration))


def core_title(title: str) -> str:
    """The song's title without credits, remaster tags or same-recording decorations; remixes stay distinct."""
    kept = _DECORATION.sub(lambda m: " " if _same_recording(_inner(m)) else m.group(0), title or "")
    return base_title(kept) or base_title(title or "")


def song_key(artist: str, title: str) -> str:
    return f"{normalize(artist)}\t{core_title(title)}"


def work_key(title: str) -> str | None:
    """Performer-independent key for a catalogue-numbered classical work (movements stay distinct)."""
    return f"work\t{core_title(title)}" if _WORK_NUMBER.search(title or "") else None


def junk_title(title: str, version: str = "") -> bool:
    text = f"{title or ''} {version or ''}"
    if _ALWAYS_JUNK.search(text):
        return True
    return any(_JUNK_DECORATION.search(_inner(m)) for m in _DECORATION.finditer(text))


def junk_artist(name: str) -> bool:
    return bool(_JUNK_ARTIST.search(name or ""))


def song_cap(fans: int, per_artist: int) -> int:
    """More songs for bigger artists: 8 at 100 fans, 12 at 1,000, 16 at 10,000, ... up to ``per_artist``."""
    if fans < 0:
        return max(1, per_artist // 2)
    return max(1, min(per_artist, max(4, round(4 * math.log10(max(fans, 1))))))


def pick_tracks(artist_id: int, top: Sequence[dict], taken: set[str], catalog_ids: set[int], per_artist: int,
                min_rank: int, min_seconds: int = 60, max_seconds: int = 900) -> list[dict]:
    """The artist's own songs worth adding, most popular first; ``taken`` gains their keys."""
    picked = []
    for item in top:
        if len(picked) >= per_artist:
            break
        artist = item.get("artist") or {}
        if artist.get("id") != artist_id or item.get("id") in catalog_ids:
            continue
        if not isinstance(item.get("preview"), str) or not item["preview"] or item.get("readable") is False:
            continue
        duration, rank = int(item.get("duration") or 0), int(item.get("rank") or 0)
        title = item.get("title") or item.get("title_short") or ""
        if not min_seconds <= duration <= max_seconds or rank < min_rank or not title:
            continue
        if junk_title(title, item.get("title_version") or ""):
            continue
        key, work = song_key(artist.get("name") or "", title), work_key(title)
        if key in taken or work in taken:
            continue
        taken.update(k for k in (key, work) if k)
        picked.append({"id": int(item["id"]), "artist_id": artist_id, "artist": artist.get("name") or "",
                       "title": title, "key": key, "rank": rank, "duration": duration,
                       "album_id": (item.get("album") or {}).get("id"), "preview": item["preview"]})
    return picked


def catalog_keys(meta: Path) -> tuple[set[int], set[str]]:
    with np.load(meta, allow_pickle=True) as z:
        ids = {int(t) for t in z["track_ids"]}
        keys = set()
        for artist, title in zip(z["artists"], z["titles"]):
            keys.add(song_key(str(artist), str(title)))
            work = work_key(str(title))
            if work:
                keys.add(work)
    return ids, keys


class AdaptiveLimiter(RateLimiter):
    """Request pacing that slows by a fifth per burst of rate limiting and recovers a tenth per quiet minute."""

    def __init__(self, rate_per_s: float, floor_per_s: float = 1.0, quiet_s: float = 60.0, burst_s: float = 5.0):
        super().__init__(rate_per_s)
        self.fastest, self.slowest = self.min_gap, max(self.min_gap, 1.0 / floor_per_s)
        self.quiet_s, self.burst_s = quiet_s, burst_s
        self.last_limited = self.last_change = -math.inf
        self.slowdowns = 0

    @property
    def rate(self) -> float:
        return 1.0 / self.min_gap

    def limited(self, now: float) -> None:
        with self._lock:
            if now - self.last_limited >= self.burst_s:
                self.min_gap = min(self.min_gap * 1.25, self.slowest)
                self.slowdowns += 1
                self.last_change = now
            self.last_limited = now

    def recover(self, now: float) -> None:
        with self._lock:
            if self.min_gap > self.fastest and now - max(self.last_limited, self.last_change) >= self.quiet_s:
                self.min_gap = max(self.fastest, self.min_gap / 1.1)
                self.last_change = now

    def cooldown(self, seconds: float) -> None:
        self.limited(time.monotonic())
        super().cooldown(seconds)

    def wait(self) -> None:
        self.recover(time.monotonic())
        super().wait()


class CrawlDB:
    def __init__(self, path: Path, readonly: bool = False):
        if readonly:
            self.conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=60)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, timeout=60)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)

    def setting(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set_setting(self, key: str, value: str) -> None:
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)", (key, value))

    def seed(self, scene: SceneTable) -> tuple[int, int]:
        """Catalog artists (hop 0) and every artist their related lists link to (hop 1), all to crawl (once)."""
        catalog = {int(a) for a in np.unique(scene.track_artist) if a > 0}
        links: Counter[int] = Counter()
        for row, aid in enumerate(scene.artist_ids.tolist()):
            if aid not in catalog:
                continue
            for j in scene.nbr_idx[scene.nbr_ptr[row]:scene.nbr_ptr[row + 1]].tolist():
                other = int(scene.artist_ids[j])
                if other != aid and other not in catalog:
                    links[other] += 1
        with self.conn:
            self.conn.executemany("INSERT OR IGNORE INTO artists (id, hop, status) VALUES (?, 0, 'frontier')",
                                  ((a,) for a in sorted(catalog)))
            self.conn.executemany("INSERT OR IGNORE INTO artists (id, hop, links, status) VALUES (?, 1, ?, 'frontier')",
                                  sorted(links.items()))
            self.conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('seeded', ?)",
                              (json.dumps({"catalog": len(catalog), "frontier": len(links), "at": time.time()}),))
        return len(catalog), len(links)

    def taken_keys(self) -> set[str]:
        keys = set()
        for key, title in self.conn.execute("SELECT song_key, title FROM tracks"):
            keys.add(key)
            work = work_key(title)
            if work:
                keys.add(work)
        return keys

    def next_artist(self, min_fans: int, exclude: Collection[int] = ()) -> tuple[int, str | None, int, int] | None:
        """Catalog artists first, then the most-followed frontier artist; ``exclude`` holds those in flight."""
        limit = len(exclude) + 1
        rows = self.conn.execute("SELECT id, name, hop, fans FROM artists WHERE status = 'frontier' AND hop = 0 "
                                 "ORDER BY id LIMIT ?", (limit,)).fetchall()
        rows += self.conn.execute(
            "SELECT id, name, hop, fans FROM artists WHERE status = 'frontier' AND hop > 0 AND (fans >= ? OR fans < 0) "
            "ORDER BY fans DESC, links DESC, id LIMIT ?", (min_fans, limit)).fetchall()
        return next((row for row in rows if row[0] not in exclude), None)

    def count(self, table: str, where: str = "1", params: Sequence = ()) -> int:
        return self.conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {where}", params).fetchone()[0]

    def _link(self, artist_id: int, hop: int, related: Sequence[dict]) -> None:
        for position, item in enumerate(related):
            rel_id = int(item["id"])
            if rel_id == artist_id:
                continue
            new_edge = self.conn.execute("INSERT OR IGNORE INTO related (artist_id, rel_id, position) "
                                         "VALUES (?, ?, ?)", (artist_id, rel_id, position)).rowcount
            fans = item.get("nb_fan")
            fans = fans if type(fans) is int and fans >= 0 else -1
            self.conn.execute(
                "INSERT INTO artists (id, name, hop, links, fans, status) VALUES (?, ?, ?, ?, ?, 'frontier') "
                "ON CONFLICT (id) DO UPDATE SET links = links + excluded.links, "
                "fans = CASE WHEN excluded.fans >= 0 THEN excluded.fans ELSE fans END, "
                "name = COALESCE(name, excluded.name), hop = MIN(hop, excluded.hop)",
                (rel_id, item.get("name"), hop + 1, new_edge, fans))
        self.conn.execute("UPDATE artists SET linked = 1 WHERE id = ?", (artist_id,))

    def record_artist(self, artist_id: int, hop: int, name: str | None, status: str, picked: Sequence[dict],
                      related: Sequence[dict]) -> None:
        now = time.time()
        with self.conn:
            if status == "done":
                self._link(artist_id, hop, related)
            self.conn.executemany(
                "INSERT OR IGNORE INTO tracks (id, artist_id, artist, title, song_key, rank, duration, album_id, "
                "added_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [(t["id"], t["artist_id"], t["artist"], t["title"], t["key"], t["rank"], t["duration"],
                  t["album_id"], now) for t in picked])
            self.conn.execute("UPDATE artists SET name = COALESCE(?, name), status = ?, songs = ?, crawled_at = ?, "
                              "error = NULL WHERE id = ?", (name, status, len(picked), now, artist_id))

    def artist_failed(self, artist_id: int, error: str, max_attempts: int) -> None:
        """Set a failed artist aside ('retry') until the next pause or the frontier's end; 'error' when spent."""
        with self.conn:
            self.conn.execute("UPDATE artists SET attempts = attempts + 1, error = ?, status = CASE "
                              "WHEN attempts + 1 >= ? THEN 'error' ELSE 'retry' END WHERE id = ?",
                              (error[:300], max_attempts, artist_id))

    def requeue_failed(self, include_spent: bool = False) -> int:
        statuses = ("retry", "error") if include_spent else ("retry",)
        with self.conn:
            return self.conn.execute(
                f"UPDATE artists SET status = 'frontier', attempts = CASE WHEN status = 'error' THEN 0 "
                f"ELSE attempts END WHERE status IN ({', '.join('?' * len(statuses))})", statuses).rowcount

    def record_downloads(self, results: Iterable[tuple[int, str, int, str | None]]) -> None:
        with self.conn:
            self.conn.executemany("UPDATE tracks SET status = ?, bytes = ?, error = ?, attempts = attempts + 1 "
                                  "WHERE id = ?", [(status, size, error, tid) for tid, status, size, error in results])

    def unfinished_tracks(self, max_attempts: int, statuses: Sequence[str] = ("queued", "error")) -> list[tuple]:
        marks = ", ".join("?" * len(statuses))
        return self.conn.execute(
            f"SELECT id, status FROM tracks WHERE status IN ({marks}) AND (status = 'queued' OR attempts < ?) "
            f"ORDER BY id", (*statuses, max_attempts)).fetchall()


class Crawler:
    def __init__(self, args: argparse.Namespace, client: Callable[..., Deezer] = Deezer,
                 log: Callable[[str], None] = print):
        self.args, self.log = args, log
        self.db = CrawlDB(Path(args.db))
        Path(args.previews).mkdir(parents=True, exist_ok=True)
        self.dz = client(args.previews, rate_per_s=args.rate, preview_rate_per_s=args.preview_rate)
        self.pacing = AdaptiveLimiter(args.preview_rate, floor_per_s=min(1.0, args.preview_rate), quiet_s=30.0)
        if hasattr(self.dz, "preview_limiter"):
            self.dz.preview_limiter = self.pacing
        self.stop = threading.Event()
        self.results: queue.Queue = queue.Queue()
        self.pending: set[int] = set()
        self.fatal: BaseException | None = None
        self.stats: Counter = Counter()
        self.failures = 0

    def _download(self, tid: int, meta: dict | None) -> None:
        try:
            path = self.dz.fetch_preview(tid, meta=meta) if meta else self.dz.fetch_preview(tid)
            self.results.put((tid, "ok" if path else "no_preview", path.stat().st_size if path else 0, None))
        except Exception as exc:  # recorded either way so a rerun retries it
            if isinstance(exc, OSError) and exc.errno in (errno.ENOSPC, errno.EDQUOT):
                self.fatal = exc
                self.stop.set()
            self.results.put((tid, "error", 0, str(exc)[:300]))

    def _drain(self, block: float = 0.0) -> None:
        done = []
        try:
            done.append(self.results.get(timeout=block) if block else self.results.get_nowait())
            while True:
                done.append(self.results.get_nowait())
        except queue.Empty:
            pass
        if not done:
            return
        self.db.record_downloads(done)
        for tid, status, size, _ in done:
            self.pending.discard(tid)
            self.stats[status] += 1
            if status != "ok":
                self.selected -= 1

    def _submit(self, pool: ThreadPoolExecutor, tid: int, meta: dict | None) -> None:
        self.pending.add(tid)
        pool.submit(self._download, tid, meta)

    def _failed(self, what: str, exc: Exception) -> None:
        """Count a failed request; pause with backoff while Deezer keeps failing (outage or quota)."""
        self.failures += 1
        self.stats["request_errors"] += 1
        self.log(f"{what} failed ({self.failures} in a row): {exc}")
        if self.failures >= 5:
            pause = min(600, 30 * 2 ** min(self.failures - 5, 4))
            self.log(f"pausing {pause}s after {self.failures} failures in a row")
            self.stop.wait(pause)
            self.db.requeue_failed()

    def _fetch(self, artist_id: int, known_name: str | None) -> tuple[list[dict], list[dict] | None, str | None]:
        """Top tracks, then related artists unless the name marks a junk channel (runs on a fetch thread)."""
        top = self.dz.artist_top_tracks(artist_id, min(100, max(50, 3 * self.args.per_artist)))
        name = next((t["artist"].get("name") for t in top
                     if isinstance(t.get("artist"), dict) and t["artist"].get("id") == artist_id), None) or known_name
        if name and junk_artist(name):
            return top, None, name
        return top, self.dz.related(artist_id), name

    def _record(self, artist_id: int, hop: int, fans: int, fetched: tuple) -> list[dict]:
        top, related, name = fetched
        self.stats["api"] += 1 if related is None else 2
        if related is None:
            self.db.record_artist(artist_id, hop, name, "skipped", [], [])
            self.stats["skipped"] += 1
            return []
        cap = max(0, song_cap(fans, self.args.per_artist) - self.existing.get(artist_id, 0))
        picked = pick_tracks(artist_id, top, self.taken, self.catalog_ids, cap, self.args.min_rank)
        self.db.record_artist(artist_id, hop, name, "done", picked, related)
        self.stats["artists"] += 1
        return picked

    def _progress(self, started: float, final: bool = False) -> None:
        db, elapsed = self.db, max(time.time() - started, 1e-6)
        saved = db.count("tracks", "status = 'ok'")
        gigabytes = db.conn.execute("SELECT COALESCE(SUM(bytes), 0) FROM tracks").fetchone()[0] / 1e9
        rate = self.stats["ok"] / elapsed
        eta = f"{max(self.args.target - saved, 0) / rate / 3600:.1f}h" if rate else "unknown"
        line = (f"{'done' if final else 'progress'} {elapsed / 3600:.2f}h | artists +{self.stats['artists']} "
                f"(skipped {self.stats['skipped']}, failed requests {self.stats['request_errors']}), "
                f"{db.count('artists', 'status = ?', ('done',))} crawled, "
                f"{db.count('artists', 'status = ?', ('frontier',))} in frontier | songs {self.selected}/"
                f"{self.args.target} selected, {saved} saved ({gigabytes:.1f} GB), {self.stats['no_preview']} "
                f"without preview, {self.stats['error']} failed, {len(self.pending)} pending | "
                f"{rate:.1f} previews/s (pacing {self.pacing.rate:.1f}/s, {self.pacing.slowdowns} slowdowns), "
                f"{self.stats['api'] / elapsed:.1f} API/s | ETA {eta}")
        self.log(line)
        self.db.set_setting("progress", json.dumps({"at": time.time(), "line": line}))

    def _wait_for_downloads(self) -> None:
        stop_deadline = None
        while self.pending and not self.fatal:
            if self.stop.is_set():
                stop_deadline = stop_deadline or time.time() + self.args.stop_timeout
                if time.time() >= stop_deadline:
                    return
            self._drain(block=1.0)

    def _fetch_again(self, pool: ThreadPoolExecutor, rows: Sequence[tuple[int, str]]) -> None:
        for tid, track_status in rows:
            if self.stop.is_set():
                return
            if track_status == "error":
                self.selected += 1
            self._submit(pool, tid, None)
            while len(self.pending) >= self.args.max_pending and not self.stop.is_set():
                self._drain(block=1.0)

    def run(self) -> str:
        args = self.args
        scene = SceneTable.load(args.scene)
        if not self.db.setting("seeded"):
            catalog, frontier = self.db.seed(scene)
            self.log(f"seeded {catalog} catalog artists and a frontier of {frontier} related artists")
        self.existing = Counter(int(a) for a in scene.track_artist.tolist() if a > 0)
        self.catalog_ids, known_keys = catalog_keys(Path(args.meta))
        self.taken = known_keys | self.db.taken_keys()
        self.selected = self.db.count("tracks", "status IN ('ok', 'queued')")
        requeued = self.db.requeue_failed(include_spent=True)
        resume = self.db.unfinished_tracks(args.max_attempts)
        self.log(f"{len(self.catalog_ids)} catalog songs; {self.selected} new songs selected so far; "
                 f"{len(resume)} previews to fetch again; {requeued} failed artists to retry")
        started, last_log, reason = time.time(), time.time(), "stopped"
        pool = ThreadPoolExecutor(args.workers)
        fetchers = ThreadPoolExecutor(args.fetchers)
        inflight: dict[int, tuple[Future, tuple]] = {}
        try:
            self._fetch_again(pool, resume)
            crawled = 0
            while not self.stop.is_set():
                self._drain()
                if time.time() - last_log >= args.log_every:
                    self._progress(started)
                    last_log = time.time()
                if len(self.pending) >= args.max_pending:
                    self._drain(block=1.0)
                    continue
                if self.selected >= args.target:
                    reason = "target reached"
                    break
                if args.max_artists and crawled >= args.max_artists:
                    reason = "artist limit reached"
                    break
                if shutil.disk_usage(args.previews).free < args.min_free_gb * 1e9:
                    reason = f"less than {args.min_free_gb} GB free"
                    break
                window = min(args.fetchers, args.max_artists - crawled) if args.max_artists else args.fetchers
                while len(inflight) < window:
                    nxt = self.db.next_artist(args.min_fans, inflight)
                    if nxt is None:
                        break
                    inflight[nxt[0]] = (fetchers.submit(self._fetch, nxt[0], nxt[1]), nxt)
                if not inflight:
                    if self.db.requeue_failed():
                        continue
                    reason = "frontier exhausted"
                    break
                done, _ = wait([future for future, _ in inflight.values()], timeout=1.0, return_when=FIRST_COMPLETED)
                for artist_id in [a for a, (future, _) in inflight.items() if future in done]:
                    future, (_, _, hop, fans) = inflight.pop(artist_id)
                    try:
                        picked = self._record(artist_id, hop, fans, future.result())
                        self.failures = 0
                    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                        self.db.artist_failed(artist_id, str(exc), args.max_attempts)
                        self._failed(f"artist {artist_id}", exc)
                        continue
                    crawled += 1
                    self.selected += len(picked)
                    for track in picked:
                        self._submit(pool, track["id"], {"preview": track["preview"]})
            if self.fatal:
                reason = f"stopped: {self.fatal}"
            self.log(f"crawl ended ({reason}); finishing {len(self.pending)} downloads")
            self._wait_for_downloads()
            if not self.stop.is_set():
                retry = self.db.unfinished_tracks(args.max_attempts, statuses=("error",))
                if retry:
                    self.log(f"fetching {len(retry)} failed previews once more")
                    self._fetch_again(pool, retry)
                    self._wait_for_downloads()
        finally:
            fetchers.shutdown(wait=False, cancel_futures=True)
            pool.shutdown(wait=False, cancel_futures=True)
            self._drain()
            self._progress(started, final=True)
            self.db.conn.close()
        if self.fatal:
            raise self.fatal
        return reason


def status(db_path: Path) -> dict:
    db = CrawlDB(db_path, readonly=True)
    artists = dict(db.conn.execute("SELECT status, COUNT(*) FROM artists GROUP BY status").fetchall())
    tracks = dict(db.conn.execute("SELECT status, COUNT(*) FROM tracks GROUP BY status").fetchall())
    hops = dict(db.conn.execute("SELECT hop, COUNT(*) FROM artists WHERE status = 'done' GROUP BY hop").fetchall())
    size = db.conn.execute("SELECT COALESCE(SUM(bytes), 0) FROM tracks").fetchone()[0]
    recent = [name for (name,) in db.conn.execute(
        "SELECT name FROM artists WHERE status = 'done' ORDER BY crawled_at DESC LIMIT 8")]
    progress = json.loads(db.setting("progress") or "{}")
    db.conn.close()
    return {"artists": artists, "crawled_by_hop": hops, "tracks": tracks, "gigabytes": round(size / 1e9, 2),
            "recent_artists": recent, "last_progress": progress.get("line"),
            "last_progress_age_s": round(time.time() - progress["at"]) if progress.get("at") else None}


def export(db_path: Path, out: Path) -> dict:
    db = CrawlDB(db_path, readonly=True)
    out.mkdir(parents=True, exist_ok=True)
    rows = db.conn.execute("SELECT id, artist_id, artist, title, rank, duration, album_id FROM tracks "
                           "WHERE status = 'ok' ORDER BY id").fetchall()
    np.savez(out / "catalog_meta.npz", track_ids=np.array([r[0] for r in rows], dtype=np.int64),
             artists=np.array([r[2] for r in rows], dtype=object), titles=np.array([r[3] for r in rows], dtype=object))
    (out / "ids.txt").write_text("".join(f"{r[0]}\n" for r in rows), encoding="utf-8")
    with (out / "deezer_tracks.jsonl").open("w", encoding="utf-8") as fh:
        for tid, aid, artist, _, rank, duration, album in rows:
            fh.write(json.dumps({"track_id": tid, "artist_id": aid, "artist_name": artist, "album_id": album,
                                 "rank": rank, "duration": duration}, ensure_ascii=False) + "\n")
    related: dict[int, list] = {aid: [] for (aid,) in db.conn.execute(
        "SELECT id FROM artists WHERE linked = 1 ORDER BY id")}
    for aid, rel, name, fans in db.conn.execute(
            "SELECT r.artist_id, r.rel_id, a.name, a.fans FROM related r JOIN artists a ON a.id = r.rel_id "
            "ORDER BY r.artist_id, r.position"):
        related.setdefault(aid, []).append([rel, name, fans if fans >= 0 else None])
    with (out / "deezer_related.jsonl").open("w", encoding="utf-8") as fh:
        for aid, items in related.items():
            fh.write(json.dumps({"artist_id": aid, "related": items}, ensure_ascii=False) + "\n")
    db.conn.close()
    summary = {"tracks": len(rows), "artists_with_related": len(related)}
    (out / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="anysong-crawl", description="grow the catalog along related artists")
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--scene", type=Path, required=True, help="scene table npz (catalog artists and related)")
    run.add_argument("--meta", type=Path, required=True, help="catalog_meta.npz (songs already in the catalog)")
    run.add_argument("--db", type=Path, required=True)
    run.add_argument("--previews", type=Path, required=True, help="preview cache (<id %% 1000>/<id>.mp3)")
    run.add_argument("--target", type=int, default=730_000, help="new songs to add")
    run.add_argument("--per-artist", type=int, default=25, help="most songs from one artist")
    run.add_argument("--min-rank", type=int, default=10_000, help="minimum Deezer track rank")
    run.add_argument("--min-fans", type=int, default=100, help="skip artists with fewer known fans")
    run.add_argument("--rate", type=float, default=4.0, help="Deezer API requests per second")
    run.add_argument("--fetchers", type=int, default=3, help="artist requests in flight")
    run.add_argument("--preview-rate", type=float, default=12.0, help="fastest preview downloads per second")
    run.add_argument("--workers", type=int, default=24, help="parallel preview downloads")
    run.add_argument("--max-pending", type=int, default=320, help="downloads queued before the crawl waits")
    run.add_argument("--max-attempts", type=int, default=3)
    run.add_argument("--min-free-gb", type=float, default=100.0)
    run.add_argument("--max-artists", type=int, default=0, help="stop after this many artists (0: no limit)")
    run.add_argument("--log-every", type=float, default=60.0)
    run.add_argument("--stop-timeout", type=float, default=90.0, help="seconds to finish downloads when stopping")
    st = sub.add_parser("status")
    st.add_argument("--db", type=Path, required=True)
    ex = sub.add_parser("export")
    ex.add_argument("--db", type=Path, required=True)
    ex.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    if args.command == "status":
        print(json.dumps(status(args.db), indent=2, ensure_ascii=False))
        return
    if args.command == "export":
        print(json.dumps(export(args.db, args.out)))
        return
    if min(args.target, args.per_artist, args.workers, args.fetchers, args.max_pending, args.max_attempts) < 1 \
            or args.rate <= 0 or args.preview_rate <= 0:
        p.error("target, per-artist, workers, fetchers, max-pending, max-attempts and rates must be positive")
    crawler = Crawler(args, log=lambda line: print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}", flush=True))

    def on_signal(signum, _frame):
        if crawler.stop.is_set():
            raise KeyboardInterrupt
        crawler.log(f"signal {signum}: stopping after pending downloads (signal again to quit now)")
        crawler.stop.set()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    print(f"finished: {crawler.run()}", flush=True)


if __name__ == "__main__":
    main()
