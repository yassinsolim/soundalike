"""Resumable, integrity-bound storage for full-track embedding artifacts.

The store uses a SQLite ledger, a raw float16 global-embedding memmap, and
fixed-track-range NPZ shards for variable-length window/section embeddings.
Rows become complete only in the same SQLite transaction that publishes a
sealed shard generation.  A crash can therefore cause bounded rework, never a
silent skip.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


STORE_SCHEMA_VERSION = 2


class FullTrackStoreError(RuntimeError):
    """Store corruption, drift, unsafe state, or invalid artifacts."""


@dataclass(frozen=True)
class StoreBinding:
    source_fingerprint: str
    config_sha256: str
    model_sha256: str
    model_id: str
    embedding_dim: int
    track_count: int
    shard_tracks: int
    repetition_sections: int
    salient_sections: int
    track_plan_sha256: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "schema_version": STORE_SCHEMA_VERSION,
            "source_fingerprint": self.source_fingerprint,
            "config_sha256": self.config_sha256,
            "model_sha256": self.model_sha256,
            "model_id": self.model_id,
            "embedding_dim": self.embedding_dim,
            "track_count": self.track_count,
            "shard_tracks": self.shard_tracks,
            "repetition_sections": self.repetition_sections,
            "salient_sections": self.salient_sections,
            "track_plan_sha256": self.track_plan_sha256,
        }


@dataclass(frozen=True)
class TrackArtifacts:
    global_embedding: np.ndarray
    window_embeddings: np.ndarray
    window_starts: np.ndarray
    repeated_sections: np.ndarray
    salient_sections: np.ndarray
    repeated_indices: np.ndarray
    salient_indices: np.ndarray
    decoded_samples: int


@dataclass
class _BufferedTrack:
    row_index: int
    track_id: int
    source_sha256: str
    artifacts: TrackArtifacts


@dataclass(frozen=True)
class StoredTrack:
    row_index: int
    track_id: int
    global_embedding: np.ndarray
    window_embeddings: np.ndarray
    window_starts: np.ndarray
    repeated_sections: np.ndarray
    salient_sections: np.ndarray
    repeated_indices: np.ndarray
    salient_indices: np.ndarray
    decoded_samples: int


def sha256_path(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json_sha256(value: Mapping[str, object]) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _track_plan_sha256(
    track_ids: Sequence[int], source_hashes: Sequence[str]
) -> str:
    raw = json.dumps(
        [
            [int(track_id), source_hash]
            for track_id, source_hash in zip(track_ids, source_hashes)
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _acquire_writer_lock(path: Path):
    handle = path.open("a+b")
    if path.stat().st_size == 0:
        handle.write(b"\0")
        handle.flush()
    handle.seek(0)
    try:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:  # pragma: no cover - exercised on non-Windows CI
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (OSError, BlockingIOError) as exc:
        handle.close()
        raise FullTrackStoreError("another writer owns this store") from exc
    return handle


def _release_writer_lock(handle) -> None:
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    else:  # pragma: no cover - exercised on non-Windows CI
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    handle.close()


def _validate_hash(value: str, label: str) -> None:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise FullTrackStoreError(f"{label} must be a lowercase SHA-256")


def _row_sha256(value: np.ndarray) -> str:
    little_endian = np.asarray(value, dtype="<f2")
    return hashlib.sha256(little_endian.tobytes(order="C")).hexdigest()


def _normalised_matrix(
    value: np.ndarray, dimension: int, label: str, *, allow_empty: bool = False
) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1:] != (dimension,):
        raise FullTrackStoreError(
            f"{label} must have shape (N, {dimension}), got {matrix.shape}"
        )
    if not allow_empty and not len(matrix):
        raise FullTrackStoreError(f"{label} may not be empty")
    if not np.all(np.isfinite(matrix)):
        raise FullTrackStoreError(f"{label} contains non-finite values")
    if len(matrix):
        norms = np.linalg.norm(matrix, axis=1)
        if not np.allclose(norms, 1.0, atol=2e-3, rtol=2e-3):
            raise FullTrackStoreError(f"{label} rows must be L2-normalized")
    return matrix.astype(np.float16)


def _validate_artifacts(
    value: TrackArtifacts,
    dimension: int,
    repetition_sections: int,
    salient_sections: int,
) -> TrackArtifacts:
    global_value = np.asarray(value.global_embedding, dtype=np.float32).reshape(-1)
    if global_value.shape != (dimension,) or not np.all(np.isfinite(global_value)):
        raise FullTrackStoreError(f"global embedding must have shape ({dimension},)")
    norm = float(np.linalg.norm(global_value))
    if not np.isclose(norm, 1.0, atol=2e-3, rtol=2e-3):
        raise FullTrackStoreError("global embedding must be L2-normalized")
    windows = _normalised_matrix(value.window_embeddings, dimension, "windows")
    starts = np.asarray(value.window_starts, dtype=np.int64).reshape(-1)
    if len(starts) != len(windows):
        raise FullTrackStoreError("window starts and embeddings have different lengths")
    if len(starts) and (starts[0] < 0 or np.any(np.diff(starts) < 0)):
        raise FullTrackStoreError("window starts must be non-negative and sorted")
    repeated = _normalised_matrix(
        value.repeated_sections, dimension, "repeated sections", allow_empty=True
    )
    salient = _normalised_matrix(
        value.salient_sections, dimension, "salient sections", allow_empty=True
    )
    expected_repeated = min(repetition_sections, len(windows))
    expected_salient = min(salient_sections, len(windows))
    if len(repeated) != expected_repeated:
        raise FullTrackStoreError(
            "repeated-section count must equal min(declared budget, source windows)"
        )
    if len(salient) != expected_salient:
        raise FullTrackStoreError(
            "salient-section count must equal min(declared budget, source windows)"
        )

    def validate_indices(
        raw: np.ndarray, sections: np.ndarray, expected: int, label: str
    ) -> np.ndarray:
        values = np.asarray(raw)
        if values.dtype.kind not in "iu" or values.ndim != 1 or len(values) != expected:
            raise FullTrackStoreError(f"{label} indices are invalid")
        indices = values.astype(np.int64, copy=False)
        if (
            np.any(indices < 0)
            or np.any(indices >= len(windows))
            or len(np.unique(indices)) != len(indices)
        ):
            raise FullTrackStoreError(
                f"{label} indices must be unique in-range source windows"
            )
        if not np.array_equal(sections, windows[indices]):
            raise FullTrackStoreError(
                f"{label} vectors do not match their declared source windows"
            )
        return indices

    repeated_indices = validate_indices(
        value.repeated_indices, repeated, expected_repeated, "repeated-section"
    )
    salient_indices = validate_indices(
        value.salient_indices, salient, expected_salient, "salient-section"
    )
    if isinstance(value.decoded_samples, bool) or int(value.decoded_samples) <= 0:
        raise FullTrackStoreError("decoded_samples must be positive")
    return TrackArtifacts(
        global_embedding=global_value.astype(np.float16),
        window_embeddings=windows,
        window_starts=starts,
        repeated_sections=repeated,
        salient_sections=salient,
        repeated_indices=repeated_indices,
        salient_indices=salient_indices,
        decoded_samples=int(value.decoded_samples),
    )


def _concat_variable(
    records: Sequence[_BufferedTrack],
    attribute: str,
    dimension: int,
) -> Tuple[np.ndarray, np.ndarray]:
    offsets = [0]
    values = []
    for record in records:
        matrix = np.asarray(getattr(record.artifacts, attribute), dtype=np.float16)
        values.append(matrix)
        offsets.append(offsets[-1] + len(matrix))
    if values and offsets[-1]:
        flat = np.concatenate(values, axis=0).astype(np.float16, copy=False)
    else:
        flat = np.empty((0, dimension), dtype=np.float16)
    return flat, np.asarray(offsets, dtype=np.int64)


def _binding_bytes(binding: StoreBinding) -> np.ndarray:
    raw = json.dumps(
        binding.as_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return np.frombuffer(raw, dtype=np.uint8).copy()


class FullTrackStore:
    """Writable resumable store.

    ``track_ids`` and ``source_hashes`` define the immutable row plan.  They
    must already be in deterministic extraction order.
    """

    def __init__(
        self,
        root: Path,
        *,
        track_ids: Sequence[int],
        source_hashes: Sequence[str],
        source_fingerprint: str,
        config_sha256: str,
        model_sha256: str,
        model_id: str,
        embedding_dim: int,
        shard_tracks: int = 256,
        repetition_sections: int = 32,
        salient_sections: int = 32,
    ) -> None:
        if len(track_ids) != len(source_hashes) or not track_ids:
            raise FullTrackStoreError("track plan is empty or misaligned")
        if len(set(int(value) for value in track_ids)) != len(track_ids):
            raise FullTrackStoreError("track IDs must be unique")
        if embedding_dim <= 0 or shard_tracks <= 0:
            raise FullTrackStoreError("embedding_dim and shard_tracks must be positive")
        if repetition_sections <= 0 or salient_sections <= 0:
            raise FullTrackStoreError("declared section budgets must be positive")
        for label, value in (
            ("source fingerprint", source_fingerprint),
            ("config hash", config_sha256),
            ("model hash", model_sha256),
        ):
            _validate_hash(value, label)
        for value in source_hashes:
            _validate_hash(value, "source hash")

        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink():
            raise FullTrackStoreError("store root may not be a symlink")
        self.shard_root = self.root / "shards"
        self.shard_root.mkdir(exist_ok=True)
        self._lock_handle = _acquire_writer_lock(self.root / ".writer.lock")
        self.ledger_path = self.root / "ledger.sqlite3"
        self.global_path = self.root / "global.f16"
        self.manifest_path = self.root / "store.sealed.json"
        self.binding = StoreBinding(
            source_fingerprint=source_fingerprint,
            config_sha256=config_sha256,
            model_sha256=model_sha256,
            model_id=model_id,
            embedding_dim=int(embedding_dim),
            track_count=len(track_ids),
            shard_tracks=int(shard_tracks),
            repetition_sections=int(repetition_sections),
            salient_sections=int(salient_sections),
            track_plan_sha256=_track_plan_sha256(track_ids, source_hashes),
        )
        self._track_ids = tuple(int(value) for value in track_ids)
        self._source_hashes = tuple(source_hashes)
        self._connection: Optional[sqlite3.Connection] = None
        try:
            self._connection = sqlite3.connect(str(self.ledger_path), timeout=30.0)
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA foreign_keys=ON")
            self._connection.execute("PRAGMA synchronous=FULL")
            self._connection.execute("PRAGMA journal_mode=DELETE")
            self._buffer_start: Optional[int] = None
            self._buffer: List[_BufferedTrack] = []
            self._closed = False
            self._initialize_or_validate()
            self._global = np.memmap(
                self.global_path,
                mode="r+",
                dtype="<f2",
                shape=(self.binding.track_count, self.binding.embedding_dim),
            )
            self.validate_completed()
            self._quarantine_orphan_generations()
        except BaseException:
            if self._connection is not None:
                self._connection.close()
            _release_writer_lock(self._lock_handle)
            raise

    def _initialize_or_validate(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS tracks (
                row_index INTEGER PRIMARY KEY,
                track_id INTEGER NOT NULL UNIQUE,
                source_sha256 TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('pending', 'done')),
                global_sha256 TEXT,
                shard_start INTEGER,
                local_index INTEGER,
                decoded_samples INTEGER,
                completed_at REAL
            );
            CREATE TABLE IF NOT EXISTS shards (
                shard_start INTEGER PRIMARY KEY,
                generation INTEGER NOT NULL,
                file_name TEXT NOT NULL UNIQUE,
                file_sha256 TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                first_row INTEGER NOT NULL,
                last_row INTEGER NOT NULL,
                sealed_at REAL NOT NULL
            );
            """
        )
        existing = dict(self._connection.execute("SELECT key, value FROM metadata"))
        expected = {
            key: str(value)
            for key, value in self.binding.as_dict().items()
        }
        if not existing:
            temporary = self.global_path.with_name(f".{self.global_path.name}.tmp")
            expected_bytes = (
                self.binding.track_count * self.binding.embedding_dim * np.dtype("<f2").itemsize
            )
            try:
                with temporary.open("xb") as handle:
                    handle.truncate(expected_bytes)
                    handle.flush()
                    os.fsync(handle.fileno())
                initial = np.memmap(
                    temporary,
                    mode="r+",
                    dtype="<f2",
                    shape=(self.binding.track_count, self.binding.embedding_dim),
                )
                initial[:] = np.nan
                initial.flush()
                del initial
                os.replace(temporary, self.global_path)
            finally:
                if temporary.exists():
                    temporary.unlink()
            with self._connection:
                self._connection.executemany(
                    "INSERT INTO metadata(key, value) VALUES (?, ?)", expected.items()
                )
                self._connection.executemany(
                    """
                    INSERT INTO tracks(
                        row_index, track_id, source_sha256, status
                    ) VALUES (?, ?, ?, 'pending')
                    """,
                    (
                        (index, track_id, self._source_hashes[index])
                        for index, track_id in enumerate(self._track_ids)
                    ),
                )
        else:
            for key, value in expected.items():
                if existing.get(key) != value:
                    raise FullTrackStoreError(
                        f"store binding drift for {key}: "
                        f"expected {value!r}, found {existing.get(key)!r}"
                    )
            rows = self._connection.execute(
                "SELECT row_index, track_id, source_sha256 FROM tracks ORDER BY row_index"
            ).fetchall()
            plan = tuple(
                (int(row["track_id"]), str(row["source_sha256"])) for row in rows
            )
            if plan != tuple(zip(self._track_ids, self._source_hashes)):
                raise FullTrackStoreError("track plan drift")
        expected_size = (
            self.binding.track_count * self.binding.embedding_dim * np.dtype("<f2").itemsize
        )
        if not self.global_path.is_file() or self.global_path.stat().st_size != expected_size:
            raise FullTrackStoreError("global memmap is missing or has the wrong size")

    @property
    def completed_count(self) -> int:
        row = self._connection.execute(
            "SELECT COUNT(*) AS count FROM tracks WHERE status='done'"
        ).fetchone()
        return int(row["count"])

    @property
    def pending_count(self) -> int:
        return self.binding.track_count - self.completed_count

    def pending_track_ids(self) -> Tuple[int, ...]:
        rows = self._connection.execute(
            "SELECT track_id FROM tracks WHERE status='pending' ORDER BY row_index"
        )
        return tuple(int(row["track_id"]) for row in rows)

    def _quarantine_orphan_generations(self) -> None:
        referenced = {
            str(row["file_name"])
            for row in self._connection.execute("SELECT file_name FROM shards")
        }
        for path in sorted(self.shard_root.glob("shard-*.npz")):
            if path.name not in referenced:
                quarantine = path.with_suffix(path.suffix + ".orphan")
                if quarantine.exists():
                    quarantine.unlink()
                os.replace(path, quarantine)

    def _load_shard_records(self, shard_start: int) -> List[_BufferedTrack]:
        row = self._connection.execute(
            "SELECT file_name, file_sha256 FROM shards WHERE shard_start=?",
            (shard_start,),
        ).fetchone()
        if row is None:
            return []
        path = self.shard_root / str(row["file_name"])
        if not path.is_file() or sha256_path(path) != str(row["file_sha256"]):
            raise FullTrackStoreError(f"shard corruption at range {shard_start}")
        arrays = _read_npz(path, self.binding)
        records = []
        for local, track_id in enumerate(arrays["track_ids"].tolist()):
            window_start = int(arrays["window_offsets"][local])
            window_end = int(arrays["window_offsets"][local + 1])
            repeat_start = int(arrays["repeat_offsets"][local])
            repeat_end = int(arrays["repeat_offsets"][local + 1])
            salient_start = int(arrays["salient_offsets"][local])
            salient_end = int(arrays["salient_offsets"][local + 1])
            records.append(
                _BufferedTrack(
                    row_index=int(arrays["row_indices"][local]),
                    track_id=int(track_id),
                    source_sha256=str(arrays["source_hashes"][local]),
                    artifacts=TrackArtifacts(
                        global_embedding=np.asarray(
                            self._global[int(arrays["row_indices"][local])],
                            dtype=np.float16,
                        ).copy(),
                        window_embeddings=np.asarray(
                            arrays["windows"][window_start:window_end], dtype=np.float16
                        ).copy(),
                        window_starts=np.asarray(
                            arrays["window_starts"][window_start:window_end],
                            dtype=np.int64,
                        ).copy(),
                        repeated_sections=np.asarray(
                            arrays["repeated"][repeat_start:repeat_end], dtype=np.float16
                        ).copy(),
                        salient_sections=np.asarray(
                            arrays["salient"][salient_start:salient_end], dtype=np.float16
                        ).copy(),
                        repeated_indices=np.asarray(
                            arrays["repeated_indices"][repeat_start:repeat_end],
                            dtype=np.int64,
                        ).copy(),
                        salient_indices=np.asarray(
                            arrays["salient_indices"][salient_start:salient_end],
                            dtype=np.int64,
                        ).copy(),
                        decoded_samples=int(arrays["decoded_samples"][local]),
                    ),
                )
            )
        return records

    def write_track(
        self,
        track_id: int,
        source_sha256: str,
        artifacts: TrackArtifacts,
    ) -> None:
        """Buffer one deterministic pending row for the next atomic shard seal."""
        if self._closed:
            raise FullTrackStoreError("store is closed")
        row = self._connection.execute(
            """
            SELECT row_index, source_sha256, status
            FROM tracks WHERE track_id=?
            """,
            (int(track_id),),
        ).fetchone()
        if row is None:
            raise FullTrackStoreError(f"track {track_id} is not in the immutable plan")
        if str(row["source_sha256"]) != source_sha256:
            raise FullTrackStoreError(f"source hash drift for track {track_id}")
        if row["status"] == "done":
            raise FullTrackStoreError(f"track {track_id} is already complete; refusing skip")
        row_index = int(row["row_index"])
        if any(record.track_id == int(track_id) for record in self._buffer):
            raise FullTrackStoreError(f"track {track_id} already exists in shard buffer")
        buffered_rows = {record.row_index for record in self._buffer}
        expected_pending = next(
            (
                int(candidate["row_index"])
                for candidate in self._connection.execute(
                    "SELECT row_index FROM tracks WHERE status='pending' ORDER BY row_index"
                )
                if int(candidate["row_index"]) not in buffered_rows
            ),
            None,
        )
        if expected_pending is None or row_index != expected_pending:
            raise FullTrackStoreError(
                f"out-of-order extraction: row {row_index} is not the next pending row"
            )
        shard_start = (row_index // self.binding.shard_tracks) * self.binding.shard_tracks
        if self._buffer_start is not None and self._buffer_start != shard_start:
            self.flush()
        if self._buffer_start is None:
            self._buffer_start = shard_start
            self._buffer = self._load_shard_records(shard_start)
        validated = _validate_artifacts(
            artifacts,
            self.binding.embedding_dim,
            self.binding.repetition_sections,
            self.binding.salient_sections,
        )
        self._buffer.append(
            _BufferedTrack(row_index, int(track_id), source_sha256, validated)
        )
        self._buffer.sort(key=lambda item: item.row_index)
        range_end = min(
            shard_start + self.binding.shard_tracks, self.binding.track_count
        )
        if len(self._buffer) == range_end - shard_start:
            self.flush()

    def flush(self) -> None:
        """Atomically publish the current fixed-row-range shard generation."""
        if self._buffer_start is None or not self._buffer:
            return
        start = self._buffer_start
        records = tuple(sorted(self._buffer, key=lambda item: item.row_index))
        if records[0].row_index != start:
            raise FullTrackStoreError("shard range does not begin at its fixed boundary")
        expected_rows = list(range(start, start + len(records)))
        if [record.row_index for record in records] != expected_rows:
            raise FullTrackStoreError("shard rows are not contiguous")

        old = self._connection.execute(
            "SELECT generation, file_name FROM shards WHERE shard_start=?", (start,)
        ).fetchone()
        generation = (int(old["generation"]) + 1) if old is not None else 1
        file_name = f"shard-{start:08d}-g{generation:08d}.npz"
        path = self.shard_root / file_name
        windows, window_offsets = _concat_variable(
            records, "window_embeddings", self.binding.embedding_dim
        )
        repeated, repeat_offsets = _concat_variable(
            records, "repeated_sections", self.binding.embedding_dim
        )
        salient, salient_offsets = _concat_variable(
            records, "salient_sections", self.binding.embedding_dim
        )
        arrays = {
            "binding_json": _binding_bytes(self.binding),
            "row_indices": np.asarray(
                [record.row_index for record in records], dtype=np.int64
            ),
            "track_ids": np.asarray(
                [record.track_id for record in records], dtype=np.int64
            ),
            "source_hashes": np.asarray(
                [record.source_sha256 for record in records], dtype="<U64"
            ),
            "decoded_samples": np.asarray(
                [record.artifacts.decoded_samples for record in records],
                dtype=np.int64,
            ),
            "windows": windows,
            "window_starts": np.concatenate(
                [
                    np.asarray(record.artifacts.window_starts, dtype=np.int64)
                    for record in records
                ]
            ),
            "window_offsets": window_offsets,
            "repeated": repeated,
            "repeated_indices": np.concatenate(
                [
                    np.asarray(record.artifacts.repeated_indices, dtype=np.int64)
                    for record in records
                ]
            ),
            "repeat_offsets": repeat_offsets,
            "salient": salient,
            "salient_indices": np.concatenate(
                [
                    np.asarray(record.artifacts.salient_indices, dtype=np.int64)
                    for record in records
                ]
            ),
            "salient_offsets": salient_offsets,
        }
        _atomic_npz(path, arrays)
        checksum = sha256_path(path)

        for record in records:
            self._global[record.row_index] = record.artifacts.global_embedding
        self._global.flush()
        now = time.time()
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO shards(
                    shard_start, generation, file_name, file_sha256, row_count,
                    first_row, last_row, sealed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(shard_start) DO UPDATE SET
                    generation=excluded.generation,
                    file_name=excluded.file_name,
                    file_sha256=excluded.file_sha256,
                    row_count=excluded.row_count,
                    first_row=excluded.first_row,
                    last_row=excluded.last_row,
                    sealed_at=excluded.sealed_at
                """,
                (
                    start,
                    generation,
                    file_name,
                    checksum,
                    len(records),
                    records[0].row_index,
                    records[-1].row_index,
                    now,
                ),
            )
            for local_index, record in enumerate(records):
                self._connection.execute(
                    """
                    UPDATE tracks SET
                        status='done',
                        global_sha256=?,
                        shard_start=?,
                        local_index=?,
                        decoded_samples=?,
                        completed_at=?
                    WHERE row_index=?
                    """,
                    (
                        _row_sha256(record.artifacts.global_embedding),
                        start,
                        local_index,
                        record.artifacts.decoded_samples,
                        now,
                        record.row_index,
                    ),
                )
        if old is not None:
            old_path = self.shard_root / str(old["file_name"])
            if old_path != path and old_path.exists():
                old_path.unlink()
        self._buffer_start = None
        self._buffer = []

    def abort_unsealed(self) -> None:
        """Discard only in-memory work; no row was marked complete."""
        self._buffer_start = None
        self._buffer = []

    def validate_completed(self) -> None:
        """Verify every ledger-complete global row and every referenced shard."""
        shard_rows = self._connection.execute(
            "SELECT shard_start, file_name, file_sha256, row_count FROM shards"
        ).fetchall()
        shard_counts: Dict[int, int] = {}
        for row in shard_rows:
            path = self.shard_root / str(row["file_name"])
            if not path.is_file() or sha256_path(path) != str(row["file_sha256"]):
                raise FullTrackStoreError(
                    f"shard checksum mismatch: {row['file_name']}"
                )
            arrays = _read_npz(path, self.binding)
            count = len(arrays["track_ids"])
            if count != int(row["row_count"]):
                raise FullTrackStoreError(f"shard row count drift: {path}")
            shard_counts[int(row["shard_start"])] = count
        completed = self._connection.execute(
            """
            SELECT row_index, global_sha256, shard_start, local_index
            FROM tracks WHERE status='done' ORDER BY row_index
            """
        ).fetchall()
        for row in completed:
            if row["shard_start"] is None or row["local_index"] is None:
                raise FullTrackStoreError("completed row has no shard location")
            start = int(row["shard_start"])
            local = int(row["local_index"])
            if start not in shard_counts or not 0 <= local < shard_counts[start]:
                raise FullTrackStoreError("completed row points outside a shard")
            actual = _row_sha256(self._global[int(row["row_index"])])
            if actual != str(row["global_sha256"]):
                raise FullTrackStoreError(
                    f"global embedding corruption at row {row['row_index']}"
                )

    def seal(self) -> Mapping[str, object]:
        """Flush and atomically publish the final read-only manifest."""
        self.flush()
        if self.pending_count:
            raise FullTrackStoreError(
                f"cannot seal store with {self.pending_count} pending tracks"
            )
        self.validate_completed()
        shards = [
            {
                "shard_start": int(row["shard_start"]),
                "file_name": str(row["file_name"]),
                "sha256": str(row["file_sha256"]),
                "row_count": int(row["row_count"]),
            }
            for row in self._connection.execute(
                "SELECT * FROM shards ORDER BY shard_start"
            )
        ]
        manifest: Dict[str, object] = {
            **self.binding.as_dict(),
            "global_file": self.global_path.name,
            "global_sha256": sha256_path(self.global_path),
            "shards": shards,
            "completed_tracks": self.completed_count,
        }
        manifest["manifest_sha256"] = stable_json_sha256(manifest)
        _atomic_json(self.manifest_path, manifest)
        with self._connection:
            self._connection.execute(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES ('sealed', '1')"
            )
        return manifest

    def close(self, *, flush: bool = True) -> None:
        if self._closed:
            return
        if flush:
            self.flush()
        else:
            self.abort_unsealed()
        self._global.flush()
        del self._global
        self._connection.close()
        _release_writer_lock(self._lock_handle)
        self._closed = True

    def __enter__(self) -> "FullTrackStore":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close(flush=exc_type is None)


def _read_npz(path: Path, binding: StoreBinding) -> Dict[str, np.ndarray]:
    required = {
        "binding_json",
        "row_indices",
        "track_ids",
        "source_hashes",
        "decoded_samples",
        "windows",
        "window_starts",
        "window_offsets",
        "repeated",
        "repeated_indices",
        "repeat_offsets",
        "salient",
        "salient_indices",
        "salient_offsets",
    }
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != required:
                raise FullTrackStoreError(f"unexpected arrays in shard {path}")
            arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    except (OSError, ValueError, EOFError) as exc:
        raise FullTrackStoreError(f"cannot load shard {path}: {exc}") from exc
    try:
        raw_binding = bytes(np.asarray(arrays["binding_json"], dtype=np.uint8).tolist())
        stored_binding = json.loads(raw_binding.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FullTrackStoreError(f"invalid shard binding in {path}") from exc
    if stored_binding != binding.as_dict():
        raise FullTrackStoreError(f"shard binding drift in {path}")
    count = len(arrays["track_ids"])
    for name in ("row_indices", "source_hashes", "decoded_samples"):
        if len(arrays[name]) != count:
            raise FullTrackStoreError(f"misaligned {name} in {path}")
    for values_name, offsets_name in (
        ("windows", "window_offsets"),
        ("repeated", "repeat_offsets"),
        ("salient", "salient_offsets"),
    ):
        values = arrays[values_name]
        offsets = np.asarray(arrays[offsets_name], dtype=np.int64)
        if (
            values.ndim != 2
            or values.shape[1:] != (binding.embedding_dim,)
            or len(offsets) != count + 1
            or offsets[0] != 0
            or offsets[-1] != len(values)
            or np.any(np.diff(offsets) < 0)
        ):
            raise FullTrackStoreError(f"invalid {values_name} offsets in {path}")
    if len(arrays["window_starts"]) != len(arrays["windows"]):
        raise FullTrackStoreError(f"misaligned window starts in {path}")
    for values_name, indices_name in (
        ("repeated", "repeated_indices"),
        ("salient", "salient_indices"),
    ):
        indices = arrays[indices_name]
        if indices.dtype.kind not in "iu" or indices.ndim != 1:
            raise FullTrackStoreError(f"invalid {indices_name} in {path}")
        if len(indices) != len(arrays[values_name]):
            raise FullTrackStoreError(f"misaligned {indices_name} in {path}")
    return arrays


class FullTrackStoreReader:
    """Strict read-only access to a completely sealed store."""

    def __init__(
        self,
        root: Path,
        *,
        expected_source_fingerprint: Optional[str] = None,
        expected_config_sha256: Optional[str] = None,
        expected_model_sha256: Optional[str] = None,
    ) -> None:
        self.root = Path(root).resolve(strict=True)
        self.manifest_path = self.root / "store.sealed.json"
        if not self.manifest_path.is_file():
            raise FullTrackStoreError("store is not sealed")
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FullTrackStoreError(f"invalid store manifest: {exc}") from exc
        if not isinstance(manifest, dict):
            raise FullTrackStoreError("store manifest must be an object")
        declared_manifest_hash = manifest.pop("manifest_sha256", None)
        if declared_manifest_hash != stable_json_sha256(manifest):
            raise FullTrackStoreError("store manifest checksum mismatch")
        self.manifest = manifest
        if manifest.get("schema_version") != STORE_SCHEMA_VERSION:
            raise FullTrackStoreError(
                f"unsupported store schema: expected {STORE_SCHEMA_VERSION}, "
                f"found {manifest.get('schema_version')!r}"
            )
        self.binding = StoreBinding(
            source_fingerprint=str(manifest["source_fingerprint"]),
            config_sha256=str(manifest["config_sha256"]),
            model_sha256=str(manifest["model_sha256"]),
            model_id=str(manifest["model_id"]),
            embedding_dim=int(manifest["embedding_dim"]),
            track_count=int(manifest["track_count"]),
            shard_tracks=int(manifest["shard_tracks"]),
            repetition_sections=int(manifest["repetition_sections"]),
            salient_sections=int(manifest["salient_sections"]),
            track_plan_sha256=str(manifest["track_plan_sha256"]),
        )
        if self.binding.repetition_sections <= 0 or self.binding.salient_sections <= 0:
            raise FullTrackStoreError("sealed store declares invalid section budgets")
        for expected, actual, label in (
            (
                expected_source_fingerprint,
                self.binding.source_fingerprint,
                "source fingerprint",
            ),
            (expected_config_sha256, self.binding.config_sha256, "config hash"),
            (expected_model_sha256, self.binding.model_sha256, "model hash"),
        ):
            if expected is not None and expected != actual:
                raise FullTrackStoreError(f"{label} drift")
        global_path = self.root / str(manifest["global_file"])
        if sha256_path(global_path) != str(manifest["global_sha256"]):
            raise FullTrackStoreError("sealed global memmap checksum mismatch")
        self.global_embeddings = np.memmap(
            global_path,
            mode="r",
            dtype="<f2",
            shape=(self.binding.track_count, self.binding.embedding_dim),
        )
        uri = f"file:{(self.root / 'ledger.sqlite3').as_posix()}?mode=ro"
        self._connection = sqlite3.connect(uri, uri=True)
        self._connection.row_factory = sqlite3.Row
        rows = self._connection.execute(
            """
            SELECT row_index, track_id, source_sha256, shard_start, local_index,
                   decoded_samples
            FROM tracks WHERE status='done' ORDER BY row_index
            """
        ).fetchall()
        if len(rows) != self.binding.track_count:
            raise FullTrackStoreError("sealed ledger is not complete")
        ledger_plan_hash = _track_plan_sha256(
            [int(row["track_id"]) for row in rows],
            [str(row["source_sha256"]) for row in rows],
        )
        if ledger_plan_hash != self.binding.track_plan_sha256:
            raise FullTrackStoreError("sealed ledger track plan checksum mismatch")
        self._rows = {int(row["track_id"]): row for row in rows}
        self.track_ids = tuple(int(row["track_id"]) for row in rows)
        self._shards = {
            int(row["shard_start"]): row
            for row in self._connection.execute("SELECT * FROM shards")
        }
        manifest_items = manifest.get("shards")
        if not isinstance(manifest_items, list):
            raise FullTrackStoreError("sealed manifest shard list is invalid")
        manifest_shards: Dict[int, Mapping[str, object]] = {}
        for item in manifest_items:
            if not isinstance(item, dict):
                raise FullTrackStoreError("sealed manifest shard entry is invalid")
            start = int(item.get("shard_start", -1))
            if start < 0 or start in manifest_shards:
                raise FullTrackStoreError("sealed manifest shard ranges are invalid")
            manifest_shards[start] = item
        if set(self._shards) != set(manifest_shards):
            raise FullTrackStoreError("sealed manifest/ledger shard ranges differ")
        for start, item in manifest_shards.items():
            ledger = self._shards[start]
            expected_file = str(item.get("file_name"))
            expected_hash = str(item.get("sha256"))
            expected_rows = int(item.get("row_count", -1))
            if (
                str(ledger["file_name"]) != expected_file
                or str(ledger["file_sha256"]) != expected_hash
                or int(ledger["row_count"]) != expected_rows
            ):
                raise FullTrackStoreError(
                    f"sealed manifest/ledger shard routing differs at range {start}"
                )
            path = self.root / "shards" / expected_file
            if sha256_path(path) != expected_hash:
                raise FullTrackStoreError(f"sealed shard checksum mismatch: {path}")
        self._cached_start: Optional[int] = None
        self._cached_arrays: Optional[Dict[str, np.ndarray]] = None

    @property
    def storage_bytes(self) -> int:
        return sum(
            path.stat().st_size
            for path in self.root.rglob("*")
            if path.is_file()
        )

    def _arrays(self, start: int) -> Dict[str, np.ndarray]:
        if self._cached_start != start:
            row = self._shards.get(start)
            if row is None:
                raise FullTrackStoreError(f"missing shard range {start}")
            path = self.root / "shards" / str(row["file_name"])
            if sha256_path(path) != str(row["file_sha256"]):
                raise FullTrackStoreError(f"shard checksum mismatch: {path}")
            self._cached_arrays = _read_npz(path, self.binding)
            self._cached_start = start
        if self._cached_arrays is None:
            raise FullTrackStoreError("internal shard cache failure")
        return self._cached_arrays

    def read_track(self, track_id: int) -> StoredTrack:
        row = self._rows.get(int(track_id))
        if row is None:
            raise KeyError(track_id)
        start = int(row["shard_start"])
        local = int(row["local_index"])
        arrays = self._arrays(start)
        if (
            not 0 <= local < len(arrays["track_ids"])
            or int(arrays["track_ids"][local]) != int(track_id)
            or int(arrays["row_indices"][local]) != int(row["row_index"])
            or str(arrays["source_hashes"][local]) != str(row["source_sha256"])
            or int(arrays["decoded_samples"][local]) != int(row["decoded_samples"])
        ):
            raise FullTrackStoreError("ledger/shard track routing mismatch")

        def section(name: str, offsets_name: str) -> np.ndarray:
            offsets = arrays[offsets_name]
            first, last = int(offsets[local]), int(offsets[local + 1])
            return np.asarray(arrays[name][first:last], dtype=np.float32)

        window_offsets = arrays["window_offsets"]
        window_first = int(window_offsets[local])
        window_last = int(window_offsets[local + 1])
        windows = section("windows", "window_offsets")
        repeated = section("repeated", "repeat_offsets")
        salient = section("salient", "salient_offsets")
        repeat_offsets = arrays["repeat_offsets"]
        repeat_first, repeat_last = (
            int(repeat_offsets[local]),
            int(repeat_offsets[local + 1]),
        )
        salient_offsets = arrays["salient_offsets"]
        salient_first, salient_last = (
            int(salient_offsets[local]),
            int(salient_offsets[local + 1]),
        )
        repeated_indices = np.asarray(
            arrays["repeated_indices"][repeat_first:repeat_last], dtype=np.int64
        )
        salient_indices = np.asarray(
            arrays["salient_indices"][salient_first:salient_last], dtype=np.int64
        )
        if len(repeated) != min(self.binding.repetition_sections, len(windows)):
            raise FullTrackStoreError(
                f"track {track_id} repeated sections violate the declared budget"
            )
        if len(salient) != min(self.binding.salient_sections, len(windows)):
            raise FullTrackStoreError(
                f"track {track_id} salient sections violate the declared budget"
            )
        for label, sections, indices in (
            ("repeated", repeated, repeated_indices),
            ("salient", salient, salient_indices),
        ):
            if (
                len(indices) != len(sections)
                or np.any(indices < 0)
                or np.any(indices >= len(windows))
                or len(np.unique(indices)) != len(indices)
                or not np.array_equal(
                    sections.astype(np.float16),
                    windows[indices].astype(np.float16),
                )
            ):
                raise FullTrackStoreError(
                    f"track {track_id} {label} source-window provenance is invalid"
                )
        return StoredTrack(
            row_index=int(row["row_index"]),
            track_id=int(track_id),
            global_embedding=np.asarray(
                self.global_embeddings[int(row["row_index"])], dtype=np.float32
            ),
            window_embeddings=windows,
            window_starts=np.asarray(
                arrays["window_starts"][window_first:window_last], dtype=np.int64
            ),
            repeated_sections=repeated,
            salient_sections=salient,
            repeated_indices=repeated_indices,
            salient_indices=salient_indices,
            decoded_samples=int(row["decoded_samples"]),
        )

    def close(self) -> None:
        self._cached_arrays = None
        del self.global_embeddings
        self._connection.close()

    def __enter__(self) -> "FullTrackStoreReader":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
