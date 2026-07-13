"""Artist-identity disambiguation module for soundalike v14.

Solves the artist name-homonym problem: a single normalised name (e.g.
"nothing") can map to multiple distinct Deezer artist IDs from entirely
different scenes.

Design
------
IdentityCache  — resumable SQLite builder.  Ingests Deezer track metadata,
                 existing candidate JSONs, and the Last.fm-360K MBID TSV.
                 Never persists preview URLs or signed tokens.  Produces a
                 compact NPZ identity asset.

IdentityAsset  — read-only NPZ loader for zero-network runtime use.
                   row_identity()          per-row stable-ID dict
                   contributor_intersection() shared contributor IDs
                   artist_centroid()       per-artist audio centroid
                   name_to_deezer_ids()    distinct artist IDs for a name
                   name_to_mbids()         source MBIDs for a name cluster
                   disambiguate()          centroid-based selection

MBID attribution policy
-----------------------
MBIDs from Last.fm are stable source identities associated at the
*name-cluster level*. They are never redistributed to Deezer IDs: even a
one-name/one-ID join is not a verified cross-source relationship. Accordingly,
this build records ``name_level_mbids`` with
``mbid_attribution="name_level_unlinked"`` and leaves ``direct_mbids`` empty.
A future verified Deezer↔MusicBrainz relationship table may populate direct
links; the public Deezer track response does not provide one.

run_identity_audit — deterministic full audit of ALL normalised keys
                     (not only homonyms), with per-name MBID evidence status,
                     v13 seed/result coverage unioned across all variants, and
                     quantified metrics.

build_identity_network / CLI — resumable network builder with injectable
                               fetcher, bounded workers, rate limiting, and
                               retries.  Fails on unresolved catalog rows by
                               default (--allow-unresolved for dev/CI).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sqlite3
import tarfile
import threading
import time
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 14
_LASTFM360K_MEMBER = "lastfm-dataset-360K/usersha1-artmbid-artname-plays.tsv"
_MBID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
_SENTINEL_ARTIST_ID = -1  # used in arrays for "unknown primary artist"

SUGGESTED_MIN_CONFIDENCE = 0.25
SUGGESTED_MIN_MARGIN = 0.05

# Minimum tracks per artist required to compute within-artist multimodality metrics
MIN_TRACKS_FOR_MULTIMODAL: int = 4
# Maximum tracks sampled per artist for O(N²) pairwise cosine (performance cap)
MAX_PAIRWISE_TRACKS_MULTIMODAL: int = 64
# Within-artist min cosine below this threshold signals a multimodal audio distribution
WITHIN_ARTIST_MULTIMODAL_COSINE_THRESHOLD: float = 0.70
# Min centroid-pair cosine separation to consider audio disambiguation feasible
CENTROID_SEPARATION_FEASIBLE_THRESHOLD: float = 0.10
# Print fetch progress every this many completed fetches
_FETCH_PROGRESS_INTERVAL: int = 1000


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ArtistIdentityError(ValueError):
    """Raised when the identity cache or asset is inconsistent."""


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def normalize_key(value: str) -> str:
    """Stable, deterministic normalised key for artist name comparisons.

    Applies NFKD Unicode decomposition, ASCII transliteration, casefolding,
    parenthetical-suffix removal, and whitespace/punctuation collapse.

    >>> normalize_key("Björk")
    'bjork'
    >>> normalize_key("NOTHING")
    'nothing'
    >>> normalize_key("Jay-Z")
    'jay z'
    >>> normalize_key("Sigur Rós")
    'sigur ros'
    """
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    value = value.casefold()
    value = re.sub(r"[\(\[][^\)\]]*[\)\]]", " ", value)
    value = re.sub(
        r"\s+-\s+(?:\d{4}\s+)?(?:re)?master(?:ed)?(?:\s+\d{4})?.*$", "", value
    )
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value).split())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalise_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True).clip(min=1e-8)
    return matrix / norms


def _valid_mbid(value: str) -> bool:
    return bool(_MBID_RE.fullmatch(str(value).strip()))


def _coerce_int(value: Any) -> Optional[int]:
    """Coerce int or numeric-string to int; return None on failure."""
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value == int(value):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except (ValueError, TypeError):
            pass
    return None


def _extract_deezer_track_fields(
    payload: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """Extract stable identity fields from Deezer API or harvest-cache rows.

    The catalog harvest caches use flat ``artist``/``artist_id`` fields while
    ``/track/{id}`` uses a nested artist object.  Only stable identity fields
    are returned; preview URLs, links, tokens, and signed parameters are
    structurally excluded.
    """
    if not isinstance(payload, dict):
        return None
    track_id = _coerce_int(payload.get("id"))
    if track_id is None:
        return None

    artist_block = payload.get("artist")
    primary_id: Optional[int] = None
    primary_name: Optional[str] = None
    if isinstance(artist_block, dict):
        primary_id = _coerce_int(artist_block.get("id"))
        pname = artist_block.get("name")
        if isinstance(pname, str):
            primary_name = pname.strip() or None
    elif isinstance(artist_block, str):
        primary_name = artist_block.strip() or None
        primary_id = _coerce_int(payload.get("artist_id"))

    if primary_id is None:
        primary_id = _coerce_int(payload.get("artist_id"))
    if primary_name is None and isinstance(payload.get("artist_name"), str):
        primary_name = str(payload["artist_name"]).strip() or None

    title_raw = payload.get("title") or payload.get("title_short")
    title = str(title_raw).strip() if title_raw else None
    contributors: List[Dict[str, Any]] = []
    for contrib in payload.get("contributors") or []:
        if not isinstance(contrib, dict):
            continue
        cid = _coerce_int(contrib.get("id"))
        cname = contrib.get("name")
        role = contrib.get("role") or contrib.get("type") or "Unknown"
        if cid is not None:
            contributors.append(
                {
                    "deezer_id": cid,
                    "name": str(cname).strip() if cname else None,
                    "role": str(role).strip(),
                }
            )
    return {
        "track_id": int(track_id),
        "primary_artist_deezer_id": primary_id,
        "artist_name": primary_name,
        "title": title,
        "contributors": contributors,
    }


def _walk_for_deezer_track_objects(
    obj: Any,
) -> Iterable[Dict[str, Any]]:
    """Recursively walk arbitrary JSON and yield Deezer-like track dicts."""
    if isinstance(obj, dict):
        extracted = _extract_deezer_track_fields(obj)
        if extracted is not None:
            yield extracted
        else:
            for child in obj.values():
                yield from _walk_for_deezer_track_objects(child)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk_for_deezer_track_objects(item)


# ---------------------------------------------------------------------------
# SQLite schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS cache_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS track_identity (
    track_id                  INTEGER PRIMARY KEY,
    row_index                 INTEGER,
    primary_artist_deezer_id  INTEGER,
    artist_name               TEXT,
    title                     TEXT,
    source                    TEXT NOT NULL,
    ingested_at               TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS track_contributors (
    track_id              INTEGER NOT NULL,
    contributor_deezer_id INTEGER NOT NULL,
    contributor_name      TEXT,
    role                  TEXT,
    PRIMARY KEY (track_id, contributor_deezer_id)
);
CREATE TABLE IF NOT EXISTS artist_name_variants (
    artist_deezer_id  INTEGER NOT NULL,
    normalized_name   TEXT NOT NULL,
    raw_name          TEXT NOT NULL,
    source            TEXT NOT NULL,
    PRIMARY KEY (artist_deezer_id, normalized_name)
);
CREATE TABLE IF NOT EXISTS mbid_mappings (
    mbid            TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    raw_name        TEXT,
    source          TEXT NOT NULL DEFAULT 'lastfm360k',
    PRIMARY KEY (mbid, normalized_name)
);
CREATE INDEX IF NOT EXISTS idx_track_identity_row
    ON track_identity (row_index);
CREATE INDEX IF NOT EXISTS idx_name_variants_norm
    ON artist_name_variants (normalized_name);
CREATE INDEX IF NOT EXISTS idx_mbid_norm
    ON mbid_mappings (normalized_name);
"""


# ---------------------------------------------------------------------------
# IdentityCache — SQLite-backed builder
# ---------------------------------------------------------------------------


class IdentityCache:
    """Resumable local identity cache for catalog artist disambiguation.

    Parameters
    ----------
    cache_path:
        Path to the SQLite database file (created if absent).
    track_ids:
        Optional int64 array of catalog track IDs in row order.  Provided on
        first creation to seed the row-order alignment; validated on resume.
    """

    def __init__(
        self,
        cache_path: Path,
        track_ids: Optional[np.ndarray] = None,
    ) -> None:
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.cache_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA_SQL)

        if track_ids is not None:
            track_ids = np.asarray(track_ids, dtype=np.int64)
            ids_sha = _sha256(track_ids.tobytes())
            existing = self._conn.execute(
                "SELECT value FROM cache_meta WHERE key='track_ids_sha256'"
            ).fetchone()
            if existing is None:
                self._conn.execute(
                    "INSERT INTO cache_meta(key,value) VALUES('track_ids_sha256',?)",
                    (ids_sha,),
                )
                self._conn.execute(
                    "INSERT INTO cache_meta(key,value) VALUES('row_count',?)",
                    (str(len(track_ids)),),
                )
                self._conn.execute(
                    "INSERT INTO cache_meta(key,value) VALUES('schema_version',?)",
                    (str(SCHEMA_VERSION),),
                )
                self._conn.executemany(
                    """
                    INSERT OR IGNORE INTO track_identity
                        (track_id, row_index, source, ingested_at)
                    VALUES (?, ?, 'catalog_seed', ?)
                    """,
                    (
                        (int(tid), int(i), _now())
                        for i, tid in enumerate(track_ids)
                    ),
                )
            elif existing[0] != ids_sha:
                raise ArtistIdentityError(
                    "Resumed cache track_ids_sha256 mismatch: "
                    f"stored {existing[0]!r}, got {ids_sha!r}"
                )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_deezer_metadata(
        self,
        track_id: int,
        payload: Mapping[str, Any],
        *,
        source: str = "deezer_api",
    ) -> bool:
        """Ingest a single Deezer /track/{id} API response.

        Uses COALESCE so that a malformed or transient response (null artist
        block) never overwrites an already-stored valid primary artist ID.
        Preview URLs and tokens are never written.

        Returns True if the primary artist ID is now known (either from this
        payload or from a prior ingestion).
        """
        extracted = _extract_deezer_track_fields(payload)
        if extracted is None or extracted["track_id"] != int(track_id):
            return False
        primary_id = extracted["primary_artist_deezer_id"]
        # COALESCE: preserve existing valid primary_id when new value is NULL.
        self._conn.execute(
            """
            UPDATE track_identity
            SET primary_artist_deezer_id = COALESCE(?, primary_artist_deezer_id),
                artist_name  = COALESCE(?, artist_name),
                title        = COALESCE(?, title),
                source       = ?,
                ingested_at  = ?
            WHERE track_id = ?
            """,
            (
                primary_id,
                extracted["artist_name"],
                extracted["title"],
                source,
                _now(),
                int(track_id),
            ),
        )
        self._conn.execute(
            """
            INSERT OR IGNORE INTO track_identity
                (track_id, primary_artist_deezer_id, artist_name,
                 title, source, ingested_at)
            VALUES (?,?,?,?,?,?)
            """,
            (
                int(track_id),
                primary_id,
                extracted["artist_name"],
                extracted["title"],
                source,
                _now(),
            ),
        )
        for contrib in extracted["contributors"]:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO track_contributors
                    (track_id, contributor_deezer_id, contributor_name, role)
                VALUES (?,?,?,?)
                """,
                (
                    int(track_id),
                    contrib["deezer_id"],
                    contrib["name"],
                    contrib["role"],
                ),
            )
            if contrib["name"]:
                self._upsert_name_variant(contrib["deezer_id"], contrib["name"], source)
        if primary_id is not None and extracted["artist_name"]:
            self._upsert_name_variant(primary_id, extracted["artist_name"], source)
        self._conn.commit()
        # Report whether primary ID is now known (may have come from prior run)
        row = self._conn.execute(
            "SELECT primary_artist_deezer_id FROM track_identity WHERE track_id=?",
            (int(track_id),),
        ).fetchone()
        return row is not None and row[0] is not None

    def _upsert_name_variant(
        self,
        deezer_artist_id: int,
        raw_name: str,
        source: str,
    ) -> None:
        normalized = normalize_key(raw_name)
        if not normalized:
            return
        self._conn.execute(
            """
            INSERT OR IGNORE INTO artist_name_variants
                (artist_deezer_id, normalized_name, raw_name, source)
            VALUES (?,?,?,?)
            """,
            (int(deezer_artist_id), normalized, raw_name.strip(), source),
        )

    def ingest_candidate_json(
        self,
        path: Path,
        *,
        source: str = "candidate_json",
    ) -> int:
        """Ingest stable identity fields from an existing harvest JSON.

        When this cache is bound to a catalog, non-catalog track records are
        ignored rather than inflating the cache or unresolved-row accounting.
        Preview URLs and tokens are never read into the persistence schema.
        """
        try:
            obj = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return 0
        catalog_bound = self._conn.execute(
            "SELECT 1 FROM cache_meta WHERE key='track_ids_sha256'"
        ).fetchone() is not None
        count = 0
        for extracted in _walk_for_deezer_track_objects(obj):
            tid = extracted["track_id"]
            primary_id = extracted["primary_artist_deezer_id"]
            cursor = self._conn.execute(
                """
                UPDATE track_identity
                SET primary_artist_deezer_id = COALESCE(primary_artist_deezer_id, ?),
                    artist_name = COALESCE(artist_name, ?),
                    title = COALESCE(title, ?)
                WHERE track_id = ?
                """,
                (primary_id, extracted["artist_name"], extracted["title"], tid),
            )
            if cursor.rowcount == 0:
                if catalog_bound:
                    continue
                self._conn.execute(
                    """
                    INSERT INTO track_identity
                        (track_id, primary_artist_deezer_id, artist_name,
                         title, source, ingested_at)
                    VALUES (?,?,?,?,?,?)
                    """,
                    (
                        tid, primary_id, extracted["artist_name"],
                        extracted["title"], source, _now(),
                    ),
                )
            for contrib in extracted["contributors"]:
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO track_contributors
                        (track_id, contributor_deezer_id, contributor_name, role)
                    VALUES (?,?,?,?)
                    """,
                    (
                        tid, contrib["deezer_id"], contrib["name"], contrib["role"],
                    ),
                )
                if contrib["name"]:
                    self._upsert_name_variant(
                        contrib["deezer_id"], contrib["name"], source
                    )
            if primary_id is not None and extracted["artist_name"]:
                self._upsert_name_variant(primary_id, extracted["artist_name"], source)
            count += 1
        self._conn.commit()
        return count

    def ingest_lastfm_mbids(
        self,
        archive_path: Path,
        *,
        max_rows: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Parse the Last.fm-360K TSV for MBID→name mappings.

        Column format: user_sha1 \\t artist_mbid \\t artist_name \\t plays

        Two rows with the same MBID but different artist names → same artist
        (alternate spellings).  Two rows with the same normalised name but
        different valid MBIDs → homonyms.
        """
        archive_path = Path(archive_path)
        source_rows = 0
        valid_mbid_rows = 0
        inserted = 0
        skipped_invalid_mbid = 0
        seen: Set[Tuple[str, str]] = set()

        with tarfile.open(archive_path, "r:gz") as tar:
            try:
                member = tar.extractfile(_LASTFM360K_MEMBER)
            except KeyError:
                member = None
            if member is None:
                raise ArtistIdentityError(
                    f"Last.fm-360K archive missing member {_LASTFM360K_MEMBER!r}"
                )
            text = io.TextIOWrapper(member, encoding="utf-8", errors="replace")
            for line in text:
                if max_rows is not None and source_rows >= max_rows:
                    break
                source_rows += 1
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                mbid_raw = parts[1].strip()
                artist_name = parts[2].strip()
                if not _valid_mbid(mbid_raw):
                    skipped_invalid_mbid += 1
                    continue
                mbid = mbid_raw.lower()
                normalized = normalize_key(artist_name)
                if not normalized:
                    continue
                valid_mbid_rows += 1
                key = (mbid, normalized)
                if key in seen:
                    continue
                seen.add(key)
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO mbid_mappings
                        (mbid, normalized_name, raw_name, source)
                    VALUES (?,?,?,'lastfm360k')
                    """,
                    (mbid, normalized, artist_name or None),
                )
                inserted += 1

        self._conn.commit()
        homonyms = self._conn.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT normalized_name FROM mbid_mappings
                GROUP BY normalized_name HAVING COUNT(DISTINCT mbid) > 1
            )
            """
        ).fetchone()[0]
        return {
            "source_rows_read": source_rows,
            "valid_mbid_rows": valid_mbid_rows,
            "skipped_invalid_mbid": skipped_invalid_mbid,
            "distinct_mbid_name_pairs_inserted": inserted,
            "homonym_normalized_names": int(homonyms),
        }

    # ------------------------------------------------------------------
    # NPZ builder
    # ------------------------------------------------------------------

    def build_npz(
        self,
        track_ids: np.ndarray,
        clap_asset_hash: str,
        output_path: Path,
        embeddings: Optional[np.ndarray] = None,
        *,
        embedding_dim: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Compile the compact NPZ identity asset.

        MBID attribution policy
        -----------------------
        Source MBIDs are stored only in per-name arrays
        (``name_mbid_flat`` / ``name_mbid_indptr``). Per-artist direct arrays
        remain empty because this build has no verified Deezer↔MBID bridge.
        """
        track_ids = np.asarray(track_ids, dtype=np.int64)
        n = len(track_ids)
        ids_sha = _sha256(track_ids.tobytes())

        existing_sha = self._conn.execute(
            "SELECT value FROM cache_meta WHERE key='track_ids_sha256'"
        ).fetchone()
        if existing_sha is not None and existing_sha[0] != ids_sha:
            raise ArtistIdentityError(
                "build_npz track_ids_sha256 mismatch with cached identity"
            )

        track_id_to_row = {int(tid): i for i, tid in enumerate(track_ids)}

        # ------------------------------------------------------------------
        # 1. Per-row primary artist IDs and contributor CSR
        # ------------------------------------------------------------------
        rows_data = self._conn.execute(
            """
            SELECT track_id, primary_artist_deezer_id
            FROM track_identity
            ORDER BY CASE WHEN row_index IS NOT NULL THEN row_index ELSE 999999999 END,
                     track_id
            """
        ).fetchall()
        primary_artist_ids = np.full(n, _SENTINEL_ARTIST_ID, dtype=np.int32)
        for tid, paid in rows_data:
            row = track_id_to_row.get(int(tid))
            if row is not None and paid is not None:
                primary_artist_ids[row] = int(paid)

        contrib_rows = self._conn.execute(
            """
            SELECT ti.track_id, tc.contributor_deezer_id
            FROM track_identity ti
            JOIN track_contributors tc ON ti.track_id = tc.track_id
            ORDER BY ti.row_index, ti.track_id, tc.contributor_deezer_id
            """
        ).fetchall()
        contrib_by_row: Dict[int, List[int]] = defaultdict(list)
        for tid, cid in contrib_rows:
            row = track_id_to_row.get(int(tid))
            if row is not None:
                contrib_by_row[row].append(int(cid))
        contrib_flat_list: List[int] = []
        contrib_indptr = np.zeros(n + 1, dtype=np.int32)
        for row in range(n):
            contrib_indptr[row + 1] = contrib_indptr[row] + len(
                contrib_by_row.get(row, [])
            )
            contrib_flat_list.extend(contrib_by_row.get(row, []))
        contrib_flat = np.asarray(contrib_flat_list, dtype=np.int32)

        # ------------------------------------------------------------------
        # 2. Unique artist IDs, audio centroids, per-artist track stats
        # ------------------------------------------------------------------
        all_artist_ids: Set[int] = set()
        for paid in primary_artist_ids:
            if int(paid) != _SENTINEL_ARTIST_ID:
                all_artist_ids.add(int(paid))
        for cids in contrib_by_row.values():
            all_artist_ids.update(cids)
        artist_ids_sorted = np.asarray(sorted(all_artist_ids), dtype=np.int32)
        m = len(artist_ids_sorted)
        artist_id_to_pos: Dict[int, int] = {
            int(aid): i for i, aid in enumerate(artist_ids_sorted)
        }

        # Single pass: build per-artist row lists for track counts + multimodality
        _artist_rows: Dict[int, List[int]] = defaultdict(list)
        for _ri in range(n):
            _paid = int(primary_artist_ids[_ri])
            if _paid != _SENTINEL_ARTIST_ID:
                _pos = artist_id_to_pos.get(_paid)
                if _pos is not None:
                    _artist_rows[_pos].append(_ri)

        artist_track_count_arr = np.zeros(m, dtype=np.int32)
        artist_within_min_cosine_arr = np.full(m, np.nan, dtype=np.float32)
        for _pos in range(m):
            artist_track_count_arr[_pos] = len(_artist_rows.get(_pos, []))

        if embeddings is not None:
            embs = np.asarray(embeddings, dtype=np.float32)
            if embs.shape[0] != n:
                raise ArtistIdentityError(
                    f"embeddings has {embs.shape[0]} rows but track_ids has {n}"
                )
            dim = embs.shape[1] if embs.ndim == 2 else (embedding_dim or 128)
            embs_norm = _normalise_rows(embs) if embs.ndim == 2 else embs
            centroid_sums = np.zeros((m, dim), dtype=np.float64)
            centroid_counts = np.zeros(m, dtype=np.int64)
            for row in range(n):
                paid = int(primary_artist_ids[row])
                if paid != _SENTINEL_ARTIST_ID:
                    pos = artist_id_to_pos.get(paid)
                    if pos is not None:
                        centroid_sums[pos] += embs_norm[row].astype(np.float64)
                        centroid_counts[pos] += 1
            valid = centroid_counts > 0
            centroids_f32 = np.zeros((m, dim), dtype=np.float32)
            if valid.any():
                centroids_f32[valid] = _normalise_rows(
                    (centroid_sums[valid] / centroid_counts[valid, None]).astype(
                        np.float32
                    )
                )
            artist_centroids = centroids_f32.astype(np.float16)
            # Within-artist multimodality: min pairwise cosine (deterministic sample)
            for _pos in range(m):
                _rows = _artist_rows.get(_pos, [])
                if len(_rows) >= MIN_TRACKS_FOR_MULTIMODAL:
                    _sample = _rows[:MAX_PAIRWISE_TRACKS_MULTIMODAL]
                    _emb_s = embs_norm[_sample]
                    _sims = _emb_s @ _emb_s.T
                    _idx_u = np.triu_indices(len(_sample), k=1)
                    if _idx_u[0].size > 0:
                        artist_within_min_cosine_arr[_pos] = float(
                            np.min(_sims[_idx_u])
                        )
        else:
            dim = embedding_dim or 0
            artist_centroids = np.zeros((0, max(dim, 1)), dtype=np.float16)

        # ------------------------------------------------------------------
        # 3. Catalog name clusters: API-resolved variants plus every aligned
        #    catalog row name. Source-only Last.fm names remain in SQLite and
        #    are not runtime candidate clusters.
        # ------------------------------------------------------------------
        name_variant_rows = self._conn.execute(
            """
            SELECT normalized_name, artist_deezer_id
            FROM artist_name_variants
            WHERE normalized_name != ''
            ORDER BY normalized_name, artist_deezer_id
            """
        ).fetchall()
        name_to_artist_ids: Dict[str, Set[int]] = defaultdict(set)
        for norm_name, aid in name_variant_rows:
            name_to_artist_ids[str(norm_name)].add(int(aid))

        # Add names from track_identity.artist_name (even for unresolved rows)
        row_name_rows = self._conn.execute(
            "SELECT track_id, artist_name FROM track_identity "
            "WHERE artist_name IS NOT NULL AND artist_name != ''"
        ).fetchall()
        row_name_by_tid: Dict[int, str] = {}
        for _tid, _aname in row_name_rows:
            if _aname:
                _norm = normalize_key(str(_aname))
                if _norm:
                    row_name_by_tid[int(_tid)] = _norm
                    # Ensure key exists; may remain empty if no Deezer ID ever found
                    name_to_artist_ids.setdefault(_norm, set())


        # Aligned row_name_keys array (empty string where no artist_name)
        row_name_keys_list: List[str] = [""] * n
        for _tid, _norm in row_name_by_tid.items():
            _row = track_id_to_row.get(int(_tid))
            if _row is not None:
                row_name_keys_list[_row] = _norm
        row_name_keys = np.asarray(row_name_keys_list)

        # Classify homonym vs unique at name level
        homonym_names_set: Set[str] = {
            name for name, aids in name_to_artist_ids.items() if len(aids) > 1
        }

        name_keys_list = sorted(name_to_artist_ids.keys())
        c = len(name_keys_list)
        name_cluster_flat_list: List[int] = []
        name_cluster_indptr = np.zeros(c + 1, dtype=np.int32)
        for i, name in enumerate(name_keys_list):
            cluster = sorted(name_to_artist_ids[name])
            name_cluster_indptr[i + 1] = name_cluster_indptr[i] + len(cluster)
            name_cluster_flat_list.extend(cluster)
        name_cluster_flat = np.asarray(name_cluster_flat_list, dtype=np.int32)
        name_keys = np.asarray(name_keys_list)

        # ------------------------------------------------------------------
        # Per-key spelling variants (CSR): catalog names + API names + MBID names
        # ------------------------------------------------------------------
        key_to_spellings: Dict[str, Set[str]] = defaultdict(set)
        # From artist_name_variants (API-confirmed)
        for _sp_row in self._conn.execute(
            "SELECT normalized_name, raw_name FROM artist_name_variants "
            "WHERE normalized_name != '' AND raw_name IS NOT NULL"
        ).fetchall():
            key_to_spellings[str(_sp_row[0])].add(str(_sp_row[1]))
        # From track_identity.artist_name (catalog seeds, including unresolved)
        for _tid, _aname in row_name_rows:
            if _aname:
                _norm = normalize_key(str(_aname))
                if _norm:
                    key_to_spellings[_norm].add(str(_aname).strip())
        # Source spelling variants are retained only for catalog keys.
        for _mr in self._conn.execute(
            "SELECT normalized_name, raw_name FROM mbid_mappings "
            "WHERE normalized_name != '' AND raw_name IS NOT NULL"
        ).fetchall():
            if _mr[1] and str(_mr[0]) in name_to_artist_ids:
                key_to_spellings[str(_mr[0])].add(str(_mr[1]))

        spelling_flat_list: List[str] = []
        spelling_indptr = np.zeros(c + 1, dtype=np.int32)
        for i, name in enumerate(name_keys_list):
            sorted_sp = sorted(key_to_spellings.get(name, set()))
            spelling_indptr[i + 1] = spelling_indptr[i] + len(sorted_sp)
            spelling_flat_list.extend(sorted_sp)
        spelling_flat = (
            np.asarray(spelling_flat_list) if spelling_flat_list
            else np.asarray([], dtype="<U1")
        )

        # ------------------------------------------------------------------
        # 4. MBID mappings — name-level and (for unambiguous names) per-artist
        # ------------------------------------------------------------------
        mbid_rows = self._conn.execute(
            "SELECT mbid, normalized_name FROM mbid_mappings ORDER BY normalized_name, mbid"
        ).fetchall()
        name_to_mbids_map: Dict[str, Set[str]] = defaultdict(set)
        for mbid, norm_name in mbid_rows:
            key = str(norm_name)
            if key in name_to_artist_ids:
                name_to_mbids_map[key].add(str(mbid))

        # No Deezer↔MusicBrainz relationship table is available here. Even a
        # one-name/one-ID join remains ambiguous, so source MBIDs stay name-level
        # and
        # are never represented as direct cross-source links.
        artist_to_direct_mbids: Dict[int, Set[str]] = defaultdict(set)

        mbid_flat_list: List[str] = []
        mbid_indptr = np.zeros(m + 1, dtype=np.int32)
        for i, aid in enumerate(artist_ids_sorted):
            mbids = sorted(artist_to_direct_mbids.get(int(aid), set()))
            mbid_indptr[i + 1] = mbid_indptr[i] + len(mbids)
            mbid_flat_list.extend(mbids)
        mbid_flat = (
            np.asarray(mbid_flat_list) if mbid_flat_list
            else np.asarray([], dtype="<U36")
        )

        # Per-name MBIDs: ALL names (including homonyms), name-cluster indexed.
        name_mbid_flat_list: List[str] = []
        name_mbid_indptr = np.zeros(c + 1, dtype=np.int32)
        for i, name in enumerate(name_keys_list):
            nm_mbids = sorted(name_to_mbids_map.get(name, set()))
            name_mbid_indptr[i + 1] = name_mbid_indptr[i] + len(nm_mbids)
            name_mbid_flat_list.extend(nm_mbids)
        name_mbid_flat = (
            np.asarray(name_mbid_flat_list) if name_mbid_flat_list
            else np.asarray([], dtype="<U36")
        )

        # ------------------------------------------------------------------
        # 5. Unresolved catalog rows
        # ------------------------------------------------------------------
        unresolved_tids = self.unresolved_track_ids()
        keys_with_zero_ids = sum(
            1 for aids in name_to_artist_ids.values() if len(aids) == 0
        )

        # ------------------------------------------------------------------
        # 6. Write NPZ
        # ------------------------------------------------------------------
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "track_ids_sha256": ids_sha,
            "clap_asset_hash": clap_asset_hash,
            "total_rows": n,
            "total_artists": m,
            "total_name_clusters": c,
            "keys_with_zero_deezer_ids": keys_with_zero_ids,
            "embedding_dim": int(dim) if embeddings is not None else 0,
            "has_centroids": embeddings is not None,
            "created_at": _now(),
        }
        np.savez_compressed(
            output_path,
            track_ids=track_ids,
            row_name_keys=row_name_keys,
            primary_artist_ids=primary_artist_ids,
            contrib_flat=contrib_flat,
            contrib_indptr=contrib_indptr,
            artist_ids=artist_ids_sorted,
            artist_centroids=artist_centroids,
            artist_track_count=artist_track_count_arr,
            artist_within_min_cosine=artist_within_min_cosine_arr,
            name_keys=name_keys,
            name_cluster_flat=name_cluster_flat,
            name_cluster_indptr=name_cluster_indptr,
            spelling_flat=spelling_flat,
            spelling_indptr=spelling_indptr,
            mbid_flat=mbid_flat,
            mbid_indptr=mbid_indptr,
            name_mbid_flat=name_mbid_flat,
            name_mbid_indptr=name_mbid_indptr,
            metadata=np.array(json.dumps(metadata, sort_keys=True)),
        )
        asset_bytes = output_path.stat().st_size
        asset_sha = _sha256(output_path.read_bytes())
        return {
            "schema_version": SCHEMA_VERSION,
            "track_ids_sha256": ids_sha,
            "clap_asset_hash": clap_asset_hash,
            "asset_path": str(output_path),
            "asset_bytes": asset_bytes,
            "asset_sha256": asset_sha,
            "total_rows": n,
            "rows_with_primary_artist_id": int(
                np.sum(primary_artist_ids != _SENTINEL_ARTIST_ID)
            ),
            "rows_unresolved": len(unresolved_tids),
            "unresolved_track_ids": unresolved_tids,
            "total_unique_artist_ids": m,
            "total_name_clusters": c,
            "keys_with_zero_deezer_ids": keys_with_zero_ids,
            "homonym_name_clusters": len(homonym_names_set),
            "total_mbid_mappings": len(mbid_flat),
            "total_name_mbid_mappings": len(name_mbid_flat),
            "has_centroids": embeddings is not None,
        }

    # ------------------------------------------------------------------
    # Utility / query helpers
    # ------------------------------------------------------------------

    def unresolved_track_ids(self) -> List[int]:
        """Return bound catalog track IDs that lack a primary artist ID."""
        bound = self._conn.execute(
            "SELECT 1 FROM cache_meta WHERE key='track_ids_sha256'"
        ).fetchone() is not None
        where = "row_index IS NOT NULL AND " if bound else ""
        rows = self._conn.execute(
            "SELECT track_id FROM track_identity WHERE "
            + where
            + "primary_artist_deezer_id IS NULL ORDER BY track_id"
        ).fetchall()
        return [int(row[0]) for row in rows]

    def close(self) -> None:
        self._conn.commit()
        self._conn.close()

    def __enter__(self) -> "IdentityCache":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def homonym_names(self) -> List[Tuple[str, List[int]]]:
        """Return (normalised_name, [distinct_artist_ids]) for all homonyms."""
        rows = self._conn.execute(
            """
            SELECT normalized_name, GROUP_CONCAT(artist_deezer_id)
            FROM artist_name_variants
            GROUP BY normalized_name
            HAVING COUNT(DISTINCT artist_deezer_id) > 1
            ORDER BY normalized_name
            """
        ).fetchall()
        return [
            (str(name), [int(x) for x in str(ids).split(",")])
            for name, ids in rows
        ]

    def mbid_homonyms(self) -> List[Tuple[str, List[str]]]:
        """Return (normalised_name, [distinct_mbids]) for Last.fm MBID homonyms."""
        rows = self._conn.execute(
            """
            SELECT normalized_name, GROUP_CONCAT(mbid)
            FROM mbid_mappings
            GROUP BY normalized_name
            HAVING COUNT(DISTINCT mbid) > 1
            ORDER BY normalized_name
            """
        ).fetchall()
        return [
            (str(name), str(mbids).split(","))
            for name, mbids in rows
        ]

    def name_variants_for_id(self, deezer_artist_id: int) -> List[str]:
        """Return all known raw name spellings for a Deezer artist ID."""
        rows = self._conn.execute(
            "SELECT raw_name FROM artist_name_variants WHERE artist_deezer_id=?",
            (int(deezer_artist_id),),
        ).fetchall()
        return [str(r[0]) for r in rows]

    def stats(self) -> Dict[str, int]:
        return {
            "tracks_with_primary_id": self._conn.execute(
                "SELECT COUNT(*) FROM track_identity "
                "WHERE primary_artist_deezer_id IS NOT NULL"
            ).fetchone()[0],
            "tracks_total": self._conn.execute(
                "SELECT COUNT(*) FROM track_identity"
            ).fetchone()[0],
            "unique_artist_ids": self._conn.execute(
                "SELECT COUNT(DISTINCT artist_deezer_id) FROM artist_name_variants"
            ).fetchone()[0],
            "unique_normalized_names": self._conn.execute(
                "SELECT COUNT(DISTINCT normalized_name) FROM artist_name_variants"
            ).fetchone()[0],
            "homonym_names": len(self.homonym_names()),
            "mbid_mappings": self._conn.execute(
                "SELECT COUNT(*) FROM mbid_mappings"
            ).fetchone()[0],
            "mbid_homonym_names": len(self.mbid_homonyms()),
        }


# ---------------------------------------------------------------------------
# IdentityAsset — read-only runtime loader
# ---------------------------------------------------------------------------


class IdentityAsset:
    """Read-only runtime identity asset loaded from an NPZ file.

    Designed for zero-network, zero-allocation repeated lookups.

    Parameters
    ----------
    npz_path:
        Path to the .npz file produced by ``IdentityCache.build_npz``.
    allow_pickle:
        Must remain False; enforced for security.
    """

    def __init__(self, npz_path: Path, *, allow_pickle: bool = False) -> None:
        if allow_pickle:
            raise ArtistIdentityError("allow_pickle must remain False")
        npz_path = Path(npz_path)
        with np.load(npz_path, allow_pickle=False) as data:
            self._track_ids = np.asarray(data["track_ids"], dtype=np.int64)
            self._primary_artist_ids = np.asarray(
                data["primary_artist_ids"], dtype=np.int32
            )
            self._contrib_flat = np.asarray(data["contrib_flat"], dtype=np.int32)
            self._contrib_indptr = np.asarray(data["contrib_indptr"], dtype=np.int32)
            self._artist_ids = np.asarray(data["artist_ids"], dtype=np.int32)
            self._artist_centroids = np.asarray(
                data["artist_centroids"], dtype=np.float16
            )
            self._name_keys = np.asarray(data["name_keys"])
            self._name_cluster_flat = np.asarray(
                data["name_cluster_flat"], dtype=np.int32
            )
            self._name_cluster_indptr = np.asarray(
                data["name_cluster_indptr"], dtype=np.int32
            )
            # Per-artist MBIDs (non-homonym only — direct attribution)
            self._mbid_flat = np.asarray(data["mbid_flat"])
            self._mbid_indptr = np.asarray(data["mbid_indptr"], dtype=np.int32)
            # Per-name MBIDs (all names including homonyms — uncertain attribution)
            files = set(data.files)
            c = len(self._name_keys)
            self._name_mbid_flat = (
                np.asarray(data["name_mbid_flat"])
                if "name_mbid_flat" in files
                else np.asarray([], dtype="<U36")
            )
            self._name_mbid_indptr = (
                np.asarray(data["name_mbid_indptr"], dtype=np.int32)
                if "name_mbid_indptr" in files
                else np.zeros(c + 1, dtype=np.int32)
            )
            # Row-aligned normalized name keys (empty string where no artist_name)
            self._row_name_keys = (
                np.asarray(data["row_name_keys"])
                if "row_name_keys" in files
                else np.asarray([""] * len(self._track_ids))
            )
            # Per-artist statistics
            m = len(self._artist_ids)
            self._artist_track_count = (
                np.asarray(data["artist_track_count"], dtype=np.int32)
                if "artist_track_count" in files
                else np.zeros(m, dtype=np.int32)
            )
            self._artist_within_min_cosine = (
                np.asarray(data["artist_within_min_cosine"], dtype=np.float32)
                if "artist_within_min_cosine" in files
                else np.full(m, np.nan, dtype=np.float32)
            )
            # Per-name spelling variants CSR (raw spellings from all sources)
            self._spelling_flat = (
                np.asarray(data["spelling_flat"])
                if "spelling_flat" in files
                else np.asarray([], dtype="<U1")
            )
            self._spelling_indptr = (
                np.asarray(data["spelling_indptr"], dtype=np.int32)
                if "spelling_indptr" in files
                else np.zeros(c + 1, dtype=np.int32)
            )
            self._meta: Dict[str, Any] = json.loads(str(data["metadata"]))

        # Compact lookup tables (track IDs are already row-aligned, so no
        # 272k-entry Python row dictionary is retained).
        self._artist_pos: Dict[int, int] = {
            int(aid): i for i, aid in enumerate(self._artist_ids)
        }
        self._name_cluster_pos: Dict[str, int] = {
            str(name): i for i, name in enumerate(self._name_keys)
        }
        # artist_id → list of name-cluster positions (an artist can have multiple names)
        self._artist_to_name_positions: Dict[int, List[int]] = defaultdict(list)
        for i in range(len(self._name_keys)):
            start = int(self._name_cluster_indptr[i])
            stop = int(self._name_cluster_indptr[i + 1])
            for aid in self._name_cluster_flat[start:stop]:
                self._artist_to_name_positions[int(aid)].append(i)

    @classmethod
    def load(cls, npz_path: Path) -> "IdentityAsset":
        return cls(npz_path)

    @property
    def metadata(self) -> Dict[str, Any]:
        return dict(self._meta)

    def verify_bindings(
        self,
        *,
        track_ids_sha256: Optional[str] = None,
        clap_asset_hash: Optional[str] = None,
    ) -> None:
        """Raise ArtistIdentityError if any provided hash does not match."""
        if track_ids_sha256 is not None:
            stored = self._meta.get("track_ids_sha256")
            if stored != track_ids_sha256:
                raise ArtistIdentityError(
                    f"track_ids_sha256 mismatch: stored {stored!r}, "
                    f"expected {track_ids_sha256!r}"
                )
        if clap_asset_hash is not None:
            stored = self._meta.get("clap_asset_hash")
            if stored != clap_asset_hash:
                raise ArtistIdentityError(
                    f"clap_asset_hash mismatch: stored {stored!r}, "
                    f"expected {clap_asset_hash!r}"
                )

    # ------------------------------------------------------------------
    # Row-level lookups
    # ------------------------------------------------------------------

    def primary_artist_id(self, row_idx: int) -> Optional[int]:
        """Return the row's stable primary Deezer artist ID when available."""
        if row_idx < 0 or row_idx >= len(self._primary_artist_ids):
            raise IndexError(f"row_idx {row_idx} out of range")
        value = int(self._primary_artist_ids[row_idx])
        return None if value == _SENTINEL_ARTIST_ID else value

    def row_identity(self, row_idx: int) -> Dict[str, Any]:
        """Return all stable identity fields for a single catalog row.

        Fields
        ------
        row_idx, track_id, primary_artist_deezer_id,
        contributor_deezer_ids,
        direct_mbids       — MBIDs from a verified cross-source link (none in
                             this build).
        name_level_mbids   — stable source MBIDs joined only by normalized name.
        mbid_attribution   — "direct" | "name_level_unlinked" | "none"
        """
        n = len(self._track_ids)
        if row_idx < 0 or row_idx >= n:
            raise IndexError(f"row_idx {row_idx} out of range [0, {n})")
        track_id = int(self._track_ids[row_idx])
        paid_raw = int(self._primary_artist_ids[row_idx])
        primary_id: Optional[int] = (
            None if paid_raw == _SENTINEL_ARTIST_ID else paid_raw
        )
        c_start = int(self._contrib_indptr[row_idx])
        c_stop = int(self._contrib_indptr[row_idx + 1])
        contrib_ids = [int(x) for x in self._contrib_flat[c_start:c_stop]]

        direct_mbids: List[str] = []
        name_level_mbids: List[str] = []
        mbid_attribution = "none"

        if primary_id is not None:
            # Reserved for future verified cross-source relationship rows.
            apos = self._artist_pos.get(primary_id)
            if apos is not None:
                m_start = int(self._mbid_indptr[apos])
                m_stop = int(self._mbid_indptr[apos + 1])
                direct_mbids = [str(x) for x in self._mbid_flat[m_start:m_stop]]

            # Name-cluster-level MBIDs (union across all name clusters for artist)
            for name_idx in self._artist_to_name_positions.get(primary_id, []):
                nm_start = int(self._name_mbid_indptr[name_idx])
                nm_stop = int(self._name_mbid_indptr[name_idx + 1])
                name_level_mbids.extend(
                    str(x) for x in self._name_mbid_flat[nm_start:nm_stop]
                )
            name_level_mbids = sorted(set(name_level_mbids))

            if direct_mbids:
                mbid_attribution = "direct"
            elif name_level_mbids:
                mbid_attribution = "name_level_unlinked"

        return {
            "row_idx": row_idx,
            "track_id": track_id,
            "primary_artist_deezer_id": primary_id,
            "contributor_deezer_ids": contrib_ids,
            "direct_mbids": direct_mbids,
            "name_level_mbids": name_level_mbids,
            "mbid_attribution": mbid_attribution,
        }

    def contributor_intersection(
        self,
        rows_a: Sequence[int],
        rows_b: Sequence[int],
    ) -> Set[int]:
        """Return contributor Deezer IDs shared across two row groups."""
        n = len(self._contrib_indptr) - 1

        def _ids(rows: Sequence[int]) -> Set[int]:
            out: Set[int] = set()
            for r in rows:
                if r < 0 or r >= n:
                    continue
                start = int(self._contrib_indptr[r])
                stop = int(self._contrib_indptr[r + 1])
                for cid in self._contrib_flat[start:stop]:
                    out.add(int(cid))
            return out

        return _ids(rows_a) & _ids(rows_b)

    # ------------------------------------------------------------------
    # Artist-level lookups
    # ------------------------------------------------------------------

    def artist_centroid(self, deezer_artist_id: int) -> Optional[np.ndarray]:
        """Return float32 L2-normalised audio centroid, or None if unavailable."""
        if self._artist_centroids.size == 0:
            return None
        apos = self._artist_pos.get(int(deezer_artist_id))
        if apos is None:
            return None
        vec = np.asarray(self._artist_centroids[apos], dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 1e-8 else None

    def name_to_deezer_ids(self, name: str) -> List[int]:
        """Sorted distinct Deezer artist IDs for a name (auto-normalised)."""
        pos = self._name_cluster_pos.get(normalize_key(name))
        if pos is None:
            return []
        start = int(self._name_cluster_indptr[pos])
        stop = int(self._name_cluster_indptr[pos + 1])
        return sorted(int(x) for x in self._name_cluster_flat[start:stop])

    def name_to_mbids(self, name: str) -> List[str]:
        """Source MBIDs for a name cluster (name-cluster level, auto-normalised).

        Returns MBIDs from the name cluster regardless of whether the name is
        a homonym.  Multiple MBIDs confirm homonym status; single MBID
        confirms unambiguous identity.
        """
        pos = self._name_cluster_pos.get(normalize_key(name))
        if pos is None:
            return []
        start = int(self._name_mbid_indptr[pos])
        stop = int(self._name_mbid_indptr[pos + 1])
        return sorted(str(x) for x in self._name_mbid_flat[start:stop])

    def spellings_for_key(self, normalized_name: str) -> List[str]:
        """Raw spelling variants for a normalized key (from NPZ, no network).

        Spellings are seeded from catalog names, API-confirmed names, and
        Last.fm MBID raw names.  No external argument needed.
        """
        pos = self._name_cluster_pos.get(normalized_name)
        if pos is None:
            return []
        start = int(self._spelling_indptr[pos])
        stop = int(self._spelling_indptr[pos + 1])
        return [str(x) for x in self._spelling_flat[start:stop]]

    # ------------------------------------------------------------------
    # Disambiguation
    # ------------------------------------------------------------------

    def disambiguate(
        self,
        name: str,
        query_centroid: np.ndarray,
        *,
        min_confidence: float = 0.0,
        min_margin: float = 0.0,
    ) -> Optional[Tuple[int, float, float]]:
        """Select the best Deezer artist ID for a name using audio proximity.

        Returns (artist_id, confidence, margin) or None to abstain.
        margin = 0.0 for single candidates (min_margin check is skipped).
        Deterministic: highest cosine first, smallest artist ID as tiebreak.
        """
        key = normalize_key(name)
        pos = self._name_cluster_pos.get(key)
        if pos is None:
            return None
        start = int(self._name_cluster_indptr[pos])
        stop = int(self._name_cluster_indptr[pos + 1])
        candidates = [int(x) for x in self._name_cluster_flat[start:stop]]
        if not candidates:
            return None

        q = np.asarray(query_centroid, dtype=np.float32).reshape(-1)
        q_norm = float(np.linalg.norm(q))
        if q_norm < 1e-8:
            return None
        q = q / q_norm

        scores: List[Tuple[int, float]] = []
        for aid in candidates:
            c = self.artist_centroid(aid)
            scores.append((aid, -2.0 if c is None else float(np.dot(q, c))))
        scores.sort(key=lambda pair: (-pair[1], pair[0]))

        best_id, best_score = scores[0]
        if best_score < -1.5:
            return None  # no centroid data at all

        confidence = float(best_score)
        margin = float(best_score - scores[1][1]) if len(scores) > 1 else 0.0

        if confidence < min_confidence:
            return None
        if len(scores) > 1 and margin < min_margin:
            return None
        return (best_id, confidence, margin)

    # ------------------------------------------------------------------
    # Iteration helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._track_ids)

    def homonym_names(self) -> List[Tuple[str, List[int]]]:
        """(normalised_name, [distinct_artist_ids]) for all homonyms."""
        out: List[Tuple[str, List[int]]] = []
        for i, name in enumerate(self._name_keys):
            start = int(self._name_cluster_indptr[i])
            stop = int(self._name_cluster_indptr[i + 1])
            cluster = [int(x) for x in self._name_cluster_flat[start:stop]]
            if len(cluster) > 1:
                out.append((str(name), cluster))
        return out


# ---------------------------------------------------------------------------
# Full normalised-key audit
# ---------------------------------------------------------------------------


def run_identity_audit(
    asset: "IdentityAsset",
    *,
    v13_diagnostics: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Deterministic audit of ALL normalised keys in the identity asset.

    Covers every key: unique-ID, multi-ID/homonym, and zero-ID keys from
    unresolved rows or MBID-only sources.  Quantifies multi-Deezer IDs,
    multi-MBIDs, spelling variants (from NPZ — no external argument needed),
    multimodal audio metrics, and v13 seed/result coverage.

    v13 variant records are **unioned** across all variants (production
    baseline + every named variant), never last-write-wins.

    MBID evidence status
    --------------------
    ``has_disjoint_mbid_evidence`` is always False.  Without a direct
    Deezer↔MBID bridge, equal MBID/Deezer counts are unlinked name-level
    evidence, not proof of disjoint attribution.  Statuses:

    * ``"none"`` — no source MBID for this name
    * ``"source_mbids_stable_but_cross_source_unlinked"`` — one or more stable
      source MBIDs exist, but no verified relationship links them to Deezer IDs
    """
    n = len(asset)
    meta = asset.metadata
    primary_ids = asset._primary_artist_ids

    # ------------------------------------------------------------------
    # Coverage
    # ------------------------------------------------------------------
    rows_with_primary = int(np.sum(primary_ids != _SENTINEL_ARTIST_ID))
    rows_without_primary = n - rows_with_primary
    unresolved_track_ids_list: List[int] = [
        int(asset._track_ids[i])
        for i in range(n)
        if int(primary_ids[i]) == _SENTINEL_ARTIST_ID
    ]
    # Distinct non-empty catalog name keys from row_name_keys
    unique_catalog_name_keys = len(
        set(str(k) for k in asset._row_name_keys if str(k) != "")
    )

    # ------------------------------------------------------------------
    # MBID coverage across all artists
    # ------------------------------------------------------------------
    m = len(asset._artist_ids)
    artist_mbid_counts = [
        int(asset._mbid_indptr[i + 1]) - int(asset._mbid_indptr[i])
        for i in range(m)
    ]
    artists_with_direct_mbids = sum(1 for c in artist_mbid_counts if c > 0)
    total_direct_mbid_mappings = sum(artist_mbid_counts)

    # ------------------------------------------------------------------
    # Union v13 seed and result rows across ALL variants (not last-write)
    # ------------------------------------------------------------------
    v13_seed_rows: Dict[str, int] = {}
    v13_result_rows: Dict[str, List[int]] = {}

    if v13_diagnostics is not None:
        def _ingest_records(records: List[Any]) -> None:
            for rec in records or []:
                sid = str(rec.get("seed_id", ""))
                if not sid:
                    continue
                qrow = int(rec.get("query_row", -1))
                if sid not in v13_seed_rows:
                    v13_seed_rows[sid] = qrow
                v13_result_rows.setdefault(sid, []).extend(
                    int(r) for r in rec.get("rows", [])
                )

        baseline = v13_diagnostics.get("production_baseline") or {}
        _ingest_records(baseline.get("records", []))
        for vdata in (v13_diagnostics.get("variants") or {}).values():
            _ingest_records((vdata or {}).get("records", []))

    # ------------------------------------------------------------------
    # Per-name-key analysis (ALL keys: unique, homonym, and zero-ID)
    # ------------------------------------------------------------------
    def _row_has_artist(row_idx: int, artist_id_set: Set[int]) -> bool:
        if row_idx < 0 or row_idx >= n:
            return False
        return int(primary_ids[row_idx]) in artist_id_set

    all_key_details: List[Dict[str, Any]] = []
    homonym_details: List[Dict[str, Any]] = []
    unresolved_by_audio = 0
    total_spelling_variants = 0

    artist_track_count: Dict[int, int] = defaultdict(int)
    for paid in primary_ids:
        if int(paid) != _SENTINEL_ARTIST_ID:
            artist_track_count[int(paid)] += 1

    name_keys = asset._name_keys
    num_keys = len(name_keys)

    for i, name in enumerate(name_keys):
        name_str = str(name)
        nc_start = int(asset._name_cluster_indptr[i])
        nc_stop = int(asset._name_cluster_indptr[i + 1])
        deezer_ids = sorted(int(x) for x in asset._name_cluster_flat[nc_start:nc_stop])
        n_dids = len(deezer_ids)

        # Name-level MBIDs
        nm_start = int(asset._name_mbid_indptr[i])
        nm_stop = int(asset._name_mbid_indptr[i + 1])
        name_mbids = sorted(str(x) for x in asset._name_mbid_flat[nm_start:nm_stop])
        n_mbids = len(name_mbids)

        # Spelling variants — read directly from asset NPZ (no external arg)
        sp_start = int(asset._spelling_indptr[i])
        sp_stop = int(asset._spelling_indptr[i + 1])
        spellings: List[str] = sorted(str(x) for x in asset._spelling_flat[sp_start:sp_stop])
        total_spelling_variants += len(spellings)

        # Key type classification
        if n_dids == 0:
            key_type = "unknown"  # unresolved row or MBID-only name, no Deezer ID
        elif n_dids == 1:
            key_type = "unique"
        else:
            key_type = "homonym"

        # MBIDs are stable source identities, but this asset has no verified
        # Deezer↔MusicBrainz relationship table. Name joins never become direct
        # links merely because each side has one identifier.
        mbid_evidence_status = (
            "source_mbids_stable_but_cross_source_unlinked"
            if n_mbids > 0
            else "none"
        )
        has_disjoint_mbid_evidence = False

        # Centroid metrics for multi-ID keys
        centroid_risk: Optional[float] = None
        min_centroid_cosine: Optional[float] = None
        max_centroid_cosine: Optional[float] = None
        centroid_separation: Optional[float] = None
        audio_disambiguation_feasible: Optional[bool] = None
        if n_dids >= 2:
            centroids_for_name = []
            for aid in deezer_ids:
                c_vec = asset.artist_centroid(aid)
                if c_vec is not None:
                    centroids_for_name.append(c_vec)
            if len(centroids_for_name) >= 2:
                pairs = [
                    float(np.dot(centroids_for_name[a], centroids_for_name[b]))
                    for a in range(len(centroids_for_name))
                    for b in range(a + 1, len(centroids_for_name))
                ]
                min_centroid_cosine = float(min(pairs))
                max_centroid_cosine = float(max(pairs))
                centroid_risk = max_centroid_cosine  # kept for backward compat
                centroid_separation = 1.0 - min_centroid_cosine
                audio_disambiguation_feasible = (
                    centroid_separation >= CENTROID_SEPARATION_FEASIBLE_THRESHOLD
                )

        # Within-artist multimodality (single-ID keys only, from stored stats)
        within_min_cosine: Optional[float] = None
        if n_dids == 1:
            apos = asset._artist_pos.get(deezer_ids[0])
            if apos is not None:
                val = float(asset._artist_within_min_cosine[apos])
                if not np.isnan(val):
                    within_min_cosine = val

        # v13 affected seeds for this name cluster
        artist_id_set = set(deezer_ids)
        affected_seeds: List[str] = []
        affected_result_count = 0
        for sid, qrow in sorted(v13_seed_rows.items()):
            if _row_has_artist(qrow, artist_id_set):
                affected_seeds.append(sid)
            for rrow in v13_result_rows.get(sid, []):
                if _row_has_artist(rrow, artist_id_set):
                    affected_result_count += 1

        key_record: Dict[str, Any] = {
            "normalized_name": name_str,
            "key_type": key_type,
            "distinct_deezer_ids": deezer_ids,
            "n_deezer_ids": n_dids,
            "name_level_mbids": name_mbids,
            "n_name_mbids": n_mbids,
            "mbid_evidence_status": mbid_evidence_status,
            "raw_spelling_variants": spellings,
            "centroid_cosine_risk": centroid_risk,
            "within_min_cosine": within_min_cosine,
        }
        all_key_details.append(key_record)

        if key_type == "homonym":
            unresolvable = (
                centroid_separation is None
                or centroid_separation < CENTROID_SEPARATION_FEASIBLE_THRESHOLD
            )
            if unresolvable:
                unresolved_by_audio += 1

            hd: Dict[str, Any] = {
                **key_record,
                "has_disjoint_mbid_evidence": has_disjoint_mbid_evidence,
                "min_centroid_cosine": min_centroid_cosine,
                "max_centroid_cosine": max_centroid_cosine,
                "centroid_separation": centroid_separation,
                "audio_disambiguation_feasible": audio_disambiguation_feasible,
                "unresolvable_by_audio": unresolvable,
                "affected_v13_seeds": sorted(affected_seeds),
                "affected_v13_result_rows": affected_result_count,
            }
            homonym_details.append(hd)

    # Sort homonym details by number of distinct IDs desc, then name
    homonym_details.sort(
        key=lambda d: (-d["n_deezer_ids"], d["normalized_name"])
    )

    # Summary stats across ALL keys
    n_unique_keys = sum(1 for d in all_key_details if d["key_type"] == "unique")
    n_homonym_keys = len(homonym_details)
    n_unknown_keys = sum(1 for d in all_key_details if d["key_type"] == "unknown")
    n_keys_with_mbids = sum(1 for d in all_key_details if d["n_name_mbids"] > 0)
    n_keys_multi_mbids = sum(1 for d in all_key_details if d["n_name_mbids"] > 1)

    # Keys with multimodal audio across ALL key types
    n_keys_multimodal_audio = 0
    # Single-ID keys: within-artist spread below threshold
    for d in all_key_details:
        if d["key_type"] == "unique":
            wmc = d["within_min_cosine"]
            if wmc is not None and wmc < WITHIN_ARTIST_MULTIMODAL_COSINE_THRESHOLD:
                n_keys_multimodal_audio += 1
    # Homonym keys: centroid separation feasible (from homonym_details which carries the field)
    for d in homonym_details:
        if d.get("audio_disambiguation_feasible"):
            n_keys_multimodal_audio += 1

    n_v13_seeds_affected = sum(
        1 for d in homonym_details if d["affected_v13_seeds"]
    )
    total_v13_result_rows_affected = sum(
        d["affected_v13_result_rows"] for d in homonym_details
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now(),
        "track_ids_sha256": meta.get("track_ids_sha256"),
        "clap_asset_hash": meta.get("clap_asset_hash"),
        "total_rows": n,
        "coverage": {
            "rows_with_primary_artist_id": rows_with_primary,
            "rows_without_primary_artist_id": rows_without_primary,
            "rows_with_primary_artist_id_fraction": (
                rows_with_primary / n if n else 0.0
            ),
            "unique_primary_artist_ids": int(
                np.unique(
                    primary_ids[primary_ids != _SENTINEL_ARTIST_ID]
                ).size
            ),
            "unique_catalog_name_keys": unique_catalog_name_keys,
            "rows_with_contributors": int(np.sum(np.diff(asset._contrib_indptr) > 0)),
            "unresolved_track_ids": unresolved_track_ids_list,
        },
        "mbid_coverage": {
            "total_artist_ids": m,
            "artist_ids_with_direct_mbids": artists_with_direct_mbids,
            "artist_ids_without_direct_mbids": m - artists_with_direct_mbids,
            "total_direct_mbid_mappings": total_direct_mbid_mappings,
        },
        "name_clusters": {
            "unique_normalized_names": num_keys,
            "unique_id_names": n_unique_keys,
            "homonym_names": n_homonym_keys,
            "unknown_id_names": n_unknown_keys,
            "homonym_fraction": n_homonym_keys / num_keys if num_keys else 0.0,
            "keys_with_any_mbids": n_keys_with_mbids,
            "keys_with_multi_mbids": n_keys_multi_mbids,
            "keys_with_multimodal_audio": n_keys_multimodal_audio,
            "unresolved_by_audio": unresolved_by_audio,
            "total_unresolved": unresolved_by_audio,
            "all_key_details": all_key_details,
            "homonym_details": homonym_details,
        },
        "v13_impact": {
            "homonym_names_affecting_v13_seeds": n_v13_seeds_affected,
            "total_v13_result_rows_affected": total_v13_result_rows_affected,
        },
    }


# ---------------------------------------------------------------------------
# Primitive fallbacks (used when disambiguation abstains)
# ---------------------------------------------------------------------------


def fallback_by_track_count(
    deezer_ids: Sequence[int],
    track_counts: Mapping[int, int],
) -> Optional[int]:
    """Select the artist ID with the most catalog tracks.  Abstains on tie."""
    if not deezer_ids:
        return None
    counts = [(track_counts.get(int(did), 0), int(did)) for did in deezer_ids]
    counts.sort(key=lambda pair: (-pair[0], pair[1]))
    best_count, best_id = counts[0]
    if len(counts) > 1 and counts[1][0] == best_count:
        return None
    return best_id if best_count > 0 else None


def fallback_by_mbid_coverage(
    deezer_ids: Sequence[int],
    asset: "IdentityAsset",
) -> Optional[int]:
    """Prefer the artist ID with richer direct MBID evidence.  Abstains on tie."""
    if not deezer_ids:
        return None
    counts = []
    for did in deezer_ids:
        apos = asset._artist_pos.get(int(did))
        if apos is None:
            counts.append((0, int(did)))
            continue
        n_mbids = int(asset._mbid_indptr[apos + 1]) - int(asset._mbid_indptr[apos])
        counts.append((n_mbids, int(did)))
    counts.sort(key=lambda pair: (-pair[0], pair[1]))
    best_n, best_id = counts[0]
    if len(counts) > 1 and counts[1][0] == best_n:
        return None
    return best_id if best_n > 0 else None


# ---------------------------------------------------------------------------
# Network fetcher (injectable — never imported or instantiated at module level)
# ---------------------------------------------------------------------------


class DeezerFetcher:
    """Base class / injectable protocol for Deezer /track/{id} fetching.

    Override ``fetch`` to inject mocks in tests.  The default implementation
    is a no-op; use ``DefaultDeezerFetcher`` for real network calls.
    """

    def fetch(self, track_id: int) -> Optional[Dict[str, Any]]:
        raise NotImplementedError


class _TokenBucket:
    """Thread-safe token bucket for rate limiting."""

    def __init__(self, rate_hz: float) -> None:
        self._interval = 1.0 / max(float(rate_hz), 1e-3)
        self._lock = threading.Lock()
        self._next_allowed = time.monotonic()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait > 0:
                time.sleep(wait)
            self._next_allowed = max(self._next_allowed, time.monotonic()) + self._interval


class DefaultDeezerFetcher(DeezerFetcher):
    """Production Deezer fetcher: real HTTP with rate limiting and retries.

    Requires ``requests``.  Import is deferred so the module remains
    network-free when only offline functionality is used.

    Parameters
    ----------
    rate_hz:
        Maximum API requests per second (Deezer free tier: ≤ 10/s; default 4).
    max_retries:
        Number of retry attempts on transient errors (5xx, timeout).
    timeout:
        HTTP request timeout in seconds.
    """

    def __init__(
        self,
        rate_hz: float = 4.0,
        max_retries: int = 3,
        timeout: float = 10.0,
    ) -> None:
        self._bucket = _TokenBucket(rate_hz)
        self._max_retries = max_retries
        self._timeout = timeout

    def fetch(self, track_id: int) -> Optional[Dict[str, Any]]:
        import requests  # deferred import — keeps module network-free by default

        url = f"https://api.deezer.com/track/{track_id}"
        for attempt in range(self._max_retries + 1):
            self._bucket.acquire()
            try:
                resp = requests.get(url, timeout=self._timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, dict) and "error" not in data:
                        return data
                    return None
                if resp.status_code in (429, 500, 502, 503, 504):
                    if attempt < self._max_retries:
                        time.sleep(2.0 ** attempt)
                        continue
                return None
            except Exception:
                if attempt < self._max_retries:
                    time.sleep(2.0 ** attempt)
                    continue
                return None
        return None


# ---------------------------------------------------------------------------
# Resumable network builder
# ---------------------------------------------------------------------------


def build_identity_network(
    catalog_track_ids: np.ndarray,
    cache_path: Path,
    output_npz: Path,
    clap_asset_hash: str,
    *,
    catalog_names: Optional[Sequence[str]] = None,
    candidate_jsons: Sequence[Path] = (),
    lastfm_archive: Optional[Path] = None,
    embeddings: Optional[np.ndarray] = None,
    v13_diagnostics: Optional[Dict[str, Any]] = None,
    audit_path: Optional[Path] = None,
    fetcher: Optional[DeezerFetcher] = None,
    max_workers: int = 4,
    rate_hz: float = 4.0,
    max_retries: int = 3,
    max_fetches: Optional[int] = None,
    progress_callback: Optional[Any] = None,
    allow_unresolved: bool = False,
) -> Dict[str, Any]:
    """Build or resume an identity NPZ for the given catalog.

    Steps (all resumable — already-resolved rows are never re-fetched):
    1. Seed cache with all catalog track IDs and optional names.
    2. Ingest candidate JSON caches (no network).
    3. Fetch unresolved track IDs via the injected fetcher (single rate-limiting
       layer — the fetcher itself handles rate limiting and retries).
    4. Ingest Last.fm MBID archive.
    5. Build NPZ and write deterministic JSON audit.

    Parameters
    ----------
    catalog_track_ids:
        Ordered int64 array of Deezer track IDs (catalog row order).
    cache_path:
        SQLite cache path (created/resumed).
    output_npz:
        Destination identity NPZ path.
    clap_asset_hash:
        SHA-256 of the CLAP compact embedding asset.
    catalog_names:
        Optional artist names aligned with catalog_track_ids.
    candidate_jsons:
        Existing JSON caches to ingest before hitting the network.
    lastfm_archive:
        Path to lastfm-dataset-360K.tar.gz.
    embeddings:
        Float32 array (N, D) aligned with catalog_track_ids.
    v13_diagnostics:
        Parsed content of clap-variant-diagnostics-v13.json.
    audit_path:
        If provided, write deterministic JSON audit to this path.
    fetcher:
        Injectable fetcher (default: DefaultDeezerFetcher).  Rate limiting
        and retries are the fetcher's responsibility — no duplicate layer here.
    max_workers:
        Thread pool size for concurrent Deezer API requests.
    rate_hz:
        API rate cap passed to DefaultDeezerFetcher when no fetcher provided.
    max_retries:
        Per-request retries passed to DefaultDeezerFetcher when no fetcher provided.
    max_fetches:
        If set, stop after this many fetch calls (for bounded preflight/resume
        runs).  Combine with allow_unresolved=True.
    progress_callback:
        Optional callable(fetched, resolved, total) called after each fetch.
        When None, periodic progress is printed to stdout.
    allow_unresolved:
        If False (default), raise ArtistIdentityError when any catalog row
        remains unresolved after all ingestion steps.
    """
    catalog_track_ids = np.asarray(catalog_track_ids, dtype=np.int64)
    n = len(catalog_track_ids)

    with IdentityCache(cache_path, catalog_track_ids) as cache:
        # Phase 1: seed catalog names if provided
        if catalog_names is not None:
            if len(catalog_names) != n:
                raise ArtistIdentityError(
                    f"catalog_names has {len(catalog_names)} entries but "
                    f"catalog_track_ids has {n}"
                )
            for tid, name in zip(catalog_track_ids, catalog_names):
                if name:
                    norm = normalize_key(str(name))
                    if norm:
                        cache._conn.execute(
                            """
                            UPDATE track_identity
                            SET artist_name = COALESCE(artist_name, ?)
                            WHERE track_id = ?
                            """,
                            (str(name).strip(), int(tid)),
                        )
            cache._conn.commit()

        # Phase 2: ingest candidate JSONs (no network)
        candidate_counts = []
        for cj_path in candidate_jsons:
            count = cache.ingest_candidate_json(Path(cj_path))
            candidate_counts.append((str(cj_path), count))

        # Phase 3: fetch unresolved IDs. When frozen v13 diagnostics are
        # supplied, hydrate its seed and served-result rows first so semantic
        # comparisons use stable identity without depending on track-ID order.
        unresolved_before = cache.unresolved_track_ids()
        priority_ids: Set[int] = set()
        if v13_diagnostics is not None:
            for section in [
                v13_diagnostics.get("production_baseline", {}),
                *(v13_diagnostics.get("variants", {}) or {}).values(),
            ]:
                for record in section.get("records", []):
                    rows = [record.get("query_row"), *record.get("rows", [])]
                    for row in rows:
                        if isinstance(row, int) and 0 <= row < n:
                            priority_ids.add(int(catalog_track_ids[row]))
        unresolved_set = set(unresolved_before)
        priority_unresolved = sorted(priority_ids & unresolved_set)
        unresolved_before = priority_unresolved + [
            track_id
            for track_id in unresolved_before
            if track_id not in priority_ids
        ]
        fetch_results: Dict[str, Any] = {
            "tracks_fetched": 0,
            "tracks_resolved": 0,
            "tracks_failed": 0,
            "priority_unresolved": len(priority_unresolved),
        }

        if unresolved_before:
            if fetcher is None:
                fetcher = DefaultDeezerFetcher(
                    rate_hz=rate_hz, max_retries=max_retries
                )
            # Bound fetch list if max_fetches requested
            to_fetch = (
                unresolved_before
                if max_fetches is None
                else unresolved_before[:max_fetches]
            )
            total_to_fetch = len(to_fetch)

            def _fetch_one(tid: int) -> Tuple[int, Optional[Dict[str, Any]]]:
                # Delegate all rate limiting and retries to the fetcher
                try:
                    result = fetcher.fetch(tid)
                except Exception:
                    result = None
                return (tid, result)

            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(_fetch_one, int(tid)): int(tid)
                    for tid in to_fetch
                }
                for future in as_completed(futures):
                    tid, payload = future.result()
                    fetch_results["tracks_fetched"] += 1
                    if payload is not None:
                        ok = cache.ingest_deezer_metadata(tid, payload)
                        if ok:
                            fetch_results["tracks_resolved"] += 1
                        else:
                            fetch_results["tracks_failed"] += 1
                    else:
                        fetch_results["tracks_failed"] += 1

                    fetched = fetch_results["tracks_fetched"]
                    if progress_callback is not None:
                        progress_callback(
                            fetched,
                            fetch_results["tracks_resolved"],
                            total_to_fetch,
                        )
                    elif (
                        fetched % _FETCH_PROGRESS_INTERVAL == 0
                        or fetched == total_to_fetch
                    ):
                        print(
                            f"[identity] {fetched}/{total_to_fetch} fetched, "
                            f"{fetch_results['tracks_resolved']} resolved",
                            flush=True,
                        )

        # Phase 4: ingest Last.fm MBIDs
        lastfm_stats: Dict[str, Any] = {}
        if lastfm_archive is not None:
            lastfm_stats = cache.ingest_lastfm_mbids(Path(lastfm_archive))

        # Phase 5: check resolution and build NPZ
        still_unresolved = cache.unresolved_track_ids()
        if still_unresolved and not allow_unresolved:
            raise ArtistIdentityError(
                f"{len(still_unresolved)} catalog rows remain unresolved after "
                f"all ingestion steps: {still_unresolved[:10]}"
                + ("..." if len(still_unresolved) > 10 else "")
            )

        build_result = cache.build_npz(
            catalog_track_ids, clap_asset_hash, output_npz, embeddings
        )

    # Phase 6: audit
    asset = IdentityAsset.load(output_npz)
    audit = run_identity_audit(asset, v13_diagnostics=v13_diagnostics)
    if audit_path is not None:
        Path(audit_path).parent.mkdir(parents=True, exist_ok=True)
        Path(audit_path).write_text(
            json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
        )

    return {
        **build_result,
        "candidate_json_ingestion": candidate_counts,
        "fetch_stats": fetch_results,
        "lastfm_stats": lastfm_stats,
        "still_unresolved": len(still_unresolved),
        "audit_path": str(audit_path) if audit_path else None,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> None:
    """CLI for building and inspecting identity NPZ assets.

    Commands
    --------
    build   Build or resume an identity NPZ for a catalog.

    Example
    -------
    python -m soundalike.ml.artist_identity_v14 build \\
        --index catalog_track_ids.npy \\
        --embeddings clap_compact.npz \\
        --clap-hash <sha256> \\
        --sqlite identity.sqlite3 \\
        --output-npz identity_v14.npz \\
        --output-audit audit.json
    """
    parser = argparse.ArgumentParser(
        prog="artist_identity_v14",
        description="Artist identity NPZ builder/inspector",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    bp = sub.add_parser("build", help="Build or resume an identity NPZ")
    bp.add_argument("--index", required=True, help="catalog NPZ or aligned track-ID NPY")
    bp.add_argument("--sqlite", required=True, help="SQLite cache path")
    bp.add_argument("--output-npz", required=True, help="output identity NPZ path")
    bp.add_argument("--clap-hash", default="", help="CLAP compact asset SHA-256")
    bp.add_argument(
        "--embeddings", default=None,
        help="aligned compact CLAP embeddings NPY/NPZ"
    )
    bp.add_argument(
        "--candidate-json", nargs="*", default=[],
        help="existing candidate JSON caches to ingest"
    )
    bp.add_argument("--lastfm-archive", default=None, help="lastfm-360K.tar.gz")
    bp.add_argument(
        "--v13-diag", default=None,
        help="clap-variant-diagnostics-v13.json"
    )
    bp.add_argument("--output-audit", default=None, help="output audit JSON path")
    bp.add_argument(
        "--allow-unresolved", action="store_true",
        help="do not fail when catalog rows are unresolved (dev/CI use)"
    )
    bp.add_argument("--max-workers", type=int, default=4)
    bp.add_argument("--rate-hz", type=float, default=4.0)
    bp.add_argument(
        "--max-fetches", type=int, default=None,
        help="cap API fetch calls (for bounded preflight/resume runs)"
    )

    args = parser.parse_args(argv)

    if args.command == "build":
        index_data = np.load(args.index, allow_pickle=False)
        catalog_names: Optional[np.ndarray] = None
        if isinstance(index_data, np.lib.npyio.NpzFile):
            catalog_track_ids = np.asarray(index_data["track_ids"], dtype=np.int64)
            if "artists" in index_data.files:
                catalog_names = np.asarray(index_data["artists"])
            index_data.close()
        else:
            catalog_track_ids = np.asarray(index_data, dtype=np.int64)

        embeddings: Optional[np.ndarray] = None
        if args.embeddings:
            emb_data = np.load(args.embeddings, allow_pickle=False)
            if isinstance(emb_data, np.lib.npyio.NpzFile):
                key = "embeddings" if "embeddings" in emb_data.files else emb_data.files[0]
                embeddings = np.asarray(emb_data[key])
                emb_data.close()
            else:
                embeddings = np.asarray(emb_data)

        v13_diag: Optional[Dict[str, Any]] = None
        if args.v13_diag:
            v13_diag = json.loads(Path(args.v13_diag).read_text(encoding="utf-8"))

        clap_hash = args.clap_hash
        if not clap_hash and args.embeddings:
            clap_hash = _sha256(Path(args.embeddings).read_bytes())
        if not clap_hash:
            raise ArtistIdentityError("--clap-hash or --embeddings is required")

        result = build_identity_network(
            catalog_track_ids=catalog_track_ids,
            catalog_names=catalog_names,
            cache_path=Path(args.sqlite),
            output_npz=Path(args.output_npz),
            clap_asset_hash=clap_hash,
            candidate_jsons=[Path(p) for p in (args.candidate_json or [])],
            lastfm_archive=Path(args.lastfm_archive) if args.lastfm_archive else None,
            embeddings=embeddings,
            v13_diagnostics=v13_diag,
            audit_path=Path(args.output_audit) if args.output_audit else None,
            allow_unresolved=args.allow_unresolved,
            max_workers=args.max_workers,
            rate_hz=args.rate_hz,
            max_fetches=args.max_fetches,
        )
        print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
