"""Lightweight, row-aligned title and artist search for the hosted catalog."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import threading
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

_INDEX_VERSION = "2026.07.11-dual-sonic64"
_INDEX_SHA256 = os.environ.get(
    "SOUNDALIKE_INDEX_SHA256",
    "f3ed57af1b8073f2872eed1e9192dee04d1089c7266fb98a157d1ea194526fb9",
)
_INDEX_PATH = os.environ.get("SOUNDALIKE_INDEX_PATH", "")
_PRODUCTION_LIBRARY_SIZE = 272_853

_PACKAGED_CATALOG_PATH = Path(__file__).with_name("search_catalog.json.gz")
_PACKAGED_CATALOG_SHA256 = (
    "c9ce8b8fcae8ed49498ef24e06f697b12b60a24a583acf291a0ccac8b37adbdf"
)
_SEARCH_CATALOG_PATH = os.environ.get("SOUNDALIKE_SEARCH_CATALOG_PATH", "")
_SEARCH_CATALOG_SHA256 = os.environ.get("SOUNDALIKE_SEARCH_CATALOG_SHA256", "")

_PAREN = re.compile(r"[\(\[][^\)\]]*[\)\]]")
_DASH_SUFFIX = re.compile(r"\s+-\s+.*$")

_SEARCH_LOCK = threading.Lock()
_SEARCH_CATALOG: Optional["SearchCatalog"] = None


def _norm(value: str) -> str:
    """Build the title/artist key shared by autocomplete and recommendation."""
    value = (
        unicodedata.normalize("NFKD", str(value))
        .encode("ascii", "ignore")
        .decode()
        .casefold()
    )
    value = _PAREN.sub(" ", value)
    value = _DASH_SUFFIX.sub("", value)
    for separator in (" feat. ", " feat ", " ft. ", " ft ", " featuring "):
        position = value.find(separator)
        if position > 0:
            value = value[:position]
    return " ".join(value.split())


def _version_penalty(title: str) -> Tuple[int, int]:
    derivative = int(
        bool(
            re.search(
                r"\b(?:karaoke|tribute|slowed|reverb|nightcore|instrumental|"
                r"remix|cover|live|acoustic)\b",
                str(title),
                re.IGNORECASE,
            )
        )
    )
    return derivative, len(str(title))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SearchCatalog:
    """Search-only metadata that preserves production recommendation row IDs."""

    def __init__(self, titles: Sequence[str], artists: Sequence[str]):
        if len(titles) != len(artists):
            raise ValueError("search catalog title and artist lengths differ")
        self.titles = titles
        self.artists = artists
        self._nt = [_norm(title) for title in titles]
        self._na = [_norm(artist) for artist in artists]
        self._naprim = [
            artist.split(",")[0].split(" & ")[0].strip() for artist in self._na
        ]
        self._by_pair: Optional[Dict[Tuple[str, str], int]] = None
        self._by_title: Optional[Dict[str, List[int]]] = None

    @classmethod
    def from_npz(cls, path: str) -> "SearchCatalog":
        import numpy as np

        with np.load(path, allow_pickle=False) as data:
            titles = data["titles"].tolist()
            artists = data["artists"].tolist()
        return cls(titles, artists)

    @classmethod
    def from_gzip_json(
        cls, path: Path, expected_sha256: str = ""
    ) -> "SearchCatalog":
        if expected_sha256:
            actual = _sha256(path)
            if actual != expected_sha256:
                raise RuntimeError(
                    "Search catalog checksum mismatch: "
                    f"expected {expected_sha256}, got {actual}"
                )
        rows = json.loads(gzip.decompress(path.read_bytes()).decode("utf-8"))
        if not isinstance(rows, list):
            raise ValueError("search catalog must contain a list")
        titles: List[str] = []
        artists: List[str] = []
        for row in rows:
            if (
                not isinstance(row, list)
                or len(row) != 2
                or not all(isinstance(value, str) for value in row)
            ):
                raise ValueError("search catalog rows must be [title, artist]")
            titles.append(row[0])
            artists.append(row[1])
        return cls(titles, artists)

    def __len__(self) -> int:
        return len(self.titles)

    def _ensure_lookups(self) -> None:
        if self._by_pair is not None:
            return
        by_pair: Dict[Tuple[str, str], int] = {}
        by_title: Dict[str, List[int]] = {}
        for row, (title, artist) in enumerate(zip(self._nt, self._naprim)):
            previous = by_pair.get((title, artist))
            if previous is None or _version_penalty(
                self.titles[row]
            ) < _version_penalty(self.titles[previous]):
                by_pair[(title, artist)] = row
            by_title.setdefault(title, []).append(row)
        self._by_pair = by_pair
        self._by_title = by_title

    def find_row(self, title: str, artist: str = "") -> Optional[int]:
        self._ensure_lookups()
        assert self._by_pair is not None and self._by_title is not None
        normalized_title = _norm(title)
        normalized_artist = _norm(artist)
        primary_artist = (
            normalized_artist.split(",")[0].split(" & ")[0].strip()
        )
        if primary_artist and (normalized_title, primary_artist) in self._by_pair:
            return self._by_pair[(normalized_title, primary_artist)]
        title_rows = self._by_title.get(normalized_title, [])
        if not normalized_artist and title_rows:
            return min(
                title_rows,
                key=lambda row: _version_penalty(self.titles[row]),
            )
        best = None
        for row, candidate_title in enumerate(self._nt):
            if normalized_title and normalized_title in candidate_title:
                if not normalized_artist or normalized_artist in self._na[row]:
                    if candidate_title == normalized_title:
                        return row
                    if best is None:
                        best = row
        return best

    def search(self, query: str, limit: int = 8) -> List[Dict]:
        normalized_query = _norm(query)
        if not normalized_query or limit <= 0:
            return []
        rows = self._search_normalized(normalized_query, int(limit))
        return [
            {"row": row, "title": title, "artist": artist}
            for row, title, artist in rows
        ]

    @lru_cache(maxsize=256)
    def _search_normalized(
        self, normalized_query: str, limit: int
    ) -> Tuple[Tuple[int, str, str], ...]:
        tokens = normalized_query.split()
        scored: List[Tuple[int, int, int]] = []
        for row, title in enumerate(self._nt):
            artist = self._na[row]
            combined = title + " " + artist
            if title == normalized_query:
                score = 0
            elif artist == normalized_query:
                score = 1
            elif title.startswith(normalized_query):
                score = 2
            elif normalized_query in artist:
                score = 3
            elif normalized_query in title:
                score = 4
            elif normalized_query in combined:
                score = 5
            elif len(tokens) > 1 and all(
                token in combined for token in tokens
            ):
                score = 6
            else:
                continue
            scored.append((score, len(title), row))
        scored.sort()

        results: List[Tuple[int, str, str]] = []
        seen = set()
        for _, __, row in scored:
            key = (self._nt[row], self._naprim[row])
            if key in seen:
                continue
            seen.add(key)
            results.append(
                (row, str(self.titles[row]), str(self.artists[row]))
            )
            if len(results) >= limit:
                break
        return tuple(results)

    def cache_info(self):
        return self._search_normalized.cache_info()


def get_search_catalog() -> SearchCatalog:
    """Return the search-only singleton without constructing the recommender."""
    global _SEARCH_CATALOG
    if _SEARCH_CATALOG is not None:
        return _SEARCH_CATALOG
    with _SEARCH_LOCK:
        if _SEARCH_CATALOG is not None:
            return _SEARCH_CATALOG
        if _SEARCH_CATALOG_PATH:
            if not _SEARCH_CATALOG_SHA256:
                raise RuntimeError(
                    "SOUNDALIKE_SEARCH_CATALOG_SHA256 is required with "
                    "SOUNDALIKE_SEARCH_CATALOG_PATH"
                )
            catalog = SearchCatalog.from_gzip_json(
                Path(_SEARCH_CATALOG_PATH), _SEARCH_CATALOG_SHA256
            )
        elif _INDEX_PATH:
            catalog = SearchCatalog.from_npz(_INDEX_PATH)
        else:
            catalog = SearchCatalog.from_gzip_json(
                _PACKAGED_CATALOG_PATH, _PACKAGED_CATALOG_SHA256
            )
            if len(catalog) != _PRODUCTION_LIBRARY_SIZE:
                raise RuntimeError(
                    "Packaged search catalog row count does not match "
                    f"{_INDEX_VERSION}"
                )
        _SEARCH_CATALOG = catalog
        return catalog


def get_library_size() -> int:
    """Avoid catalog/model initialization for the default stats response."""
    if _SEARCH_CATALOG_PATH or _INDEX_PATH:
        return len(get_search_catalog())
    return _PRODUCTION_LIBRARY_SIZE


def write_search_catalog(index_path: Path, output_path: Path) -> Dict[str, object]:
    """Write deterministic row-aligned metadata from a DeepVibe NPZ."""
    import numpy as np

    with np.load(index_path, allow_pickle=False) as data:
        titles = data["titles"]
        artists = data["artists"]
        if len(titles) != len(artists):
            raise ValueError("index title and artist lengths differ")
        partial = output_path.with_name(output_path.name + ".part")
        try:
            with partial.open("wb") as raw:
                with gzip.GzipFile(
                    filename="",
                    mode="wb",
                    fileobj=raw,
                    compresslevel=9,
                    mtime=0,
                ) as compressed:
                    with io.TextIOWrapper(
                        compressed, encoding="utf-8", write_through=True
                    ) as stream:
                        stream.write("[")
                        for row, (title, artist) in enumerate(
                            zip(titles, artists)
                        ):
                            if row:
                                stream.write(",")
                            json.dump(
                                [str(title), str(artist)],
                                stream,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            )
                        stream.write("]")
            os.replace(partial, output_path)
        finally:
            if partial.exists():
                partial.unlink()
    return {
        "rows": len(titles),
        "bytes": output_path.stat().st_size,
        "sha256": _sha256(output_path),
    }
