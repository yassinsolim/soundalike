"""Rate-limited Deezer public API client with an on-disk preview cache.

Deezer allows ~50 requests per 5 seconds per IP; we stay well below that.
Preview URLs are signed and expire after a few minutes, so a preview is always
downloaded right after its metadata request.
"""

from __future__ import annotations

import json
import errno
import math
import re
import threading
import time
import unicodedata
import urllib.parse
import urllib.request
import warnings
from dataclasses import dataclass
from datetime import timezone
from email.utils import parsedate_to_datetime
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError

API = "https://api.deezer.com"
USER_AGENT = "soundalike-anysong/0.1 (+https://soundalike.yassin.app)"

_VERSION_WORDS = (
    "karaoke", "instrumental", "8d", "slowed", "sped up", "nightcore", "reverb",
    "piano", "lullaby", "rendition", "tribute", "cover", "remix", "acoustic", "live",
    "edit", "mix", "version", "made famous", "in the style",
)


class RateLimiter:
    def __init__(self, rate_per_s: float = 8.0):
        if not math.isfinite(rate_per_s) or rate_per_s <= 0:
            raise ValueError("Request rate must be finite and positive")
        self.min_gap = 1.0 / rate_per_s
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                delay = self._next - now
                if delay <= 0:
                    self._next = now + self.min_gap
                    return
            time.sleep(delay)

    def cooldown(self, seconds: float) -> None:
        with self._lock:
            self._next = max(self._next, time.monotonic() + seconds)


def _retry_delay(error: HTTPError, default: float) -> float:
    value = (error.headers or {}).get("Retry-After")
    if value:
        try:
            delay = float(value)
        except ValueError:
            try:
                stamp = parsedate_to_datetime(value)
                if stamp.tzinfo is None:
                    stamp = stamp.replace(tzinfo=timezone.utc)
                delay = stamp.timestamp() - time.time()
            except (ValueError, TypeError, OverflowError):
                return default
        if math.isfinite(delay):
            return max(default, delay)
    return default


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def base_title(title: str) -> str:
    """Strip feat./remaster/version decorations that vary between services."""
    t = re.sub(r"\((feat|ft|with)\.?[^)]*\)", "", title, flags=re.I)
    t = re.sub(r"\[(feat|ft|with)\.?[^\]]*\]", "", t, flags=re.I)
    t = re.sub(r"\s+-\s+(\d{4}\s+)?remaster(ed)?(\s+\d{4})?.*$", "", t, flags=re.I)
    t = re.sub(r"\((\d{4}\s+)?remaster(ed)?(\s+\d{4})?[^)]*\)", "", t, flags=re.I)
    return normalize(t)


def version_words(title: str) -> set[str]:
    n = normalize(title)
    return {w for w in _VERSION_WORDS if re.search(rf"\b{re.escape(w)}\b", n)}


@dataclass
class Match:
    deezer_id: int
    title: str
    artist: str
    isrc: str | None
    duration: int
    score: float
    rank: int


class Deezer:
    def __init__(self, cache_dir: str | Path, rate_per_s: float = 8.0, timeout: float = 20.0,
                 preview_rate_per_s: float = 16.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.limiter = RateLimiter(rate_per_s)
        self.preview_limiter = RateLimiter(preview_rate_per_s)
        self.timeout = timeout

    def _get(self, path: str, params: dict | None = None, retries: int = 4) -> dict:
        url = f"{API}/{path.lstrip('/')}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        last_err: Exception | None = None
        for attempt in range(retries):
            self.limiter.wait()
            try:
                req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.load(resp)
                err = data.get("error") if isinstance(data, dict) else None
                if err:
                    if err.get("code") == 4:  # quota exceeded
                        last_err = RuntimeError(f"deezer error {err}")
                        delay = 5 * (attempt + 1)
                        self.limiter.cooldown(delay)
                        warnings.warn(f"Deezer API quota exceeded; pausing metadata workers for {delay}s",
                                      RuntimeWarning, stacklevel=2)
                        continue
                    if err.get("code") == 800:  # no data
                        return {}
                    raise RuntimeError(f"deezer error {err}")
                return data
            except (OSError, ValueError, HTTPException) as exc:
                last_err = exc
                if isinstance(exc, HTTPError) and exc.code == 429:
                    delay = _retry_delay(exc, 5 * (attempt + 1))
                    self.limiter.cooldown(delay)
                    warnings.warn(f"Deezer API rate limited; pausing metadata workers for {delay}s",
                                  RuntimeWarning, stacklevel=2)
                elif attempt < retries - 1:
                    time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"deezer request failed: {url}: {last_err}")

    def track(self, track_id: int) -> dict:
        return self._get(f"track/{int(track_id)}")

    def artist_top_tracks(self, artist_id: int, limit: int = 100) -> list[dict]:
        response = self._get(f"artist/{int(artist_id)}/top", {"limit": limit})
        if not isinstance(response, dict):
            raise ValueError(f"Invalid track metadata for artist {artist_id}")
        data = response.get("data", [])
        if not isinstance(data, list) or any(
            not isinstance(item, dict) or type(item.get("id")) is not int for item in data
        ):
            raise ValueError(f"Invalid track metadata for artist {artist_id}")
        return data

    def related_artists(self, artist_id: int, limit: int = 50) -> list[int]:
        """Deezer's related-artist list (most related first; Deezer currently returns up to 20)."""
        response = self._get(f"artist/{int(artist_id)}/related", {"limit": limit})
        data = response.get("data", []) if isinstance(response, dict) else []
        return [int(item["id"]) for item in data if isinstance(item, dict) and type(item.get("id")) is int]

    def track_by_isrc(self, isrc: str) -> dict:
        return self._get(f"track/isrc:{isrc.strip().upper()}")

    def search(self, query: str, limit: int = 25) -> list[dict]:
        return self._get("search", {"q": query, "limit": limit}).get("data", [])

    def match(self, title: str, artists: list[str], duration_s: float | None = None,
              allow_versions: bool = False) -> Match | None:
        """Find the original recording for a (title, artists) pair.

        Requires the normalized base title to match and at least one credited
        artist to overlap; penalises karaoke/cover/remix uploads unless the
        requested title itself is such a version.
        """
        want_title = base_title(title)
        want_artists = {normalize(a) for a in artists if a}
        want_versions = version_words(title)
        queries = [
            f'artist:"{artists[0]}" track:"{re.sub(chr(34), "", title)}"' if artists else title,
            f"{title} {' '.join(artists[:2])}",
        ]
        if base_title(title) != normalize(title):
            queries.append(f"{want_title} {artists[0] if artists else ''}")
        seen: dict[int, Match] = {}
        for q in queries:
            for rank, item in enumerate(self.search(q)):
                tid = int(item["id"])
                if tid in seen:
                    continue
                cand_title = item.get("title", "")
                cand_artist = item.get("artist", {}).get("name", "")
                cand_artist_n = normalize(cand_artist)
                artist_ok = any(
                    a == cand_artist_n or (len(a) > 3 and (a in cand_artist_n or cand_artist_n in a))
                    for a in want_artists
                )
                if not artist_ok:
                    continue
                cb = base_title(cand_title)
                if cb != want_title and not (cb.startswith(want_title) or want_title.startswith(cb)):
                    continue
                extra = version_words(cand_title) - want_versions
                if extra and not allow_versions:
                    continue
                score = 1.0 - 0.02 * rank
                if cb == want_title:
                    score += 0.5
                if normalize(cand_title) == normalize(title):
                    score += 0.25
                if duration_s and item.get("duration"):
                    score -= min(abs(float(item["duration"]) - duration_s) / 60.0, 1.0)
                seen[tid] = Match(tid, cand_title, cand_artist, item.get("isrc"),
                                  int(item.get("duration") or 0), score, rank)
            if seen and max(m.score for m in seen.values()) >= 1.6:
                break
        if not seen:
            return None
        return max(seen.values(), key=lambda m: m.score)

    def preview_path(self, track_id: int) -> Path:
        tid = int(track_id)
        return self.cache_dir / f"{tid % 1000:03d}" / f"{tid}.mp3"

    def fetch_preview(self, track_id: int, meta: dict | None = None) -> Path | None:
        """Download (once) the 30 s preview MP3 for a track; returns None if unavailable."""
        path = self.preview_path(track_id)
        if path.exists() and path.stat().st_size > 10_000:
            return path
        meta = meta if meta and meta.get("preview") else self.track(track_id)
        url = (meta or {}).get("preview")
        if not url:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        last_error: OSError | HTTPException | None = None
        try:
            self.preview_limiter.wait()
            for attempt in range(3):
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
                    with urllib.request.urlopen(req, timeout=self.timeout) as resp, open(tmp, "wb") as fh:
                        fh.write(resp.read())
                    if tmp.stat().st_size <= 10_000:
                        raise OSError(f"Preview download too small for track {track_id}")
                    tmp.replace(path)
                    return path
                except (OSError, HTTPException) as exc:
                    if isinstance(exc, OSError) and exc.errno in (errno.ENOSPC, errno.EDQUOT):
                        raise
                    last_error = exc
                    limited = isinstance(exc, HTTPError) and exc.code == 429
                    if limited:
                        delay = _retry_delay(exc, 5 * (attempt + 1))
                        self.preview_limiter.cooldown(delay)
                        warnings.warn(f"Preview CDN rate limited; pausing audio workers for {delay}s",
                                      RuntimeWarning, stacklevel=2)
                    if attempt < 2:
                        if limited:
                            self.preview_limiter.wait()
                        else:
                            time.sleep(1.5 * (attempt + 1))
                        meta = self.track(track_id)
                        url = (meta or {}).get("preview")
                        if not url:
                            return None
            raise RuntimeError(f"Preview download failed for track {track_id}: {last_error}") from last_error
        finally:
            tmp.unlink(missing_ok=True)
