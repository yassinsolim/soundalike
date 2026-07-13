"""Frozen, resumable CLAP catalogue challenger for human development.

This module is deliberately isolated from the production recommender.  It
downloads fresh public Deezer previews, embeds and deletes them, writes only
gitignored derived state, compresses with the pre-registered JL projection,
and produces proxy-safety diagnostics.  Nothing here is a promotion gate.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import sqlite3
import tempfile
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import numpy as np

from .human_eval_v10 import canonical_bytes, content_hash, file_hash
from .quality_filter import TitleQualityFilter
from .real_benchmark import PairResolver, ProductionRanker, normalize_text


SCHEMA_VERSION = 13
SEED = 20260713
EXPECTED_ROWS = 272_853
FULL_DIM = 512
DIMENSIONS = (64, 96, 112, 128)
CHECKPOINT_SHA256 = "8053c9775516af2f4902e1e8281e356cc1bf7a85e8b761908170767b77c3f037"
PREREGISTRATION_SHA256 = (
    "2c1bb55c85dfa8d1d344bba02868563c459ac743604f525ecb678598f3ef4ee7"
)
TRUSTED_PREREGISTRATION_R3_FILES = {
    "preregistration-v13-r3.sig":
        "6bf100fb04a6cc61ef09022401e0bac31b4da8e41b50b83229ff14835bc86014",
    "prereg-r3-signer.pub":
        "cbd59f964170085de0415d70768743983b89ed1ad8d34cd4f72ce2aefb34ef4f",
    "prereg-r3-allowed-signers":
        "8eeab58ec50697c851bdbe2d5a50aefe5d131431c6da86802ed7d99e3883816c",
}
TRACK_IDS_SHA256 = "a20632fc8fb4beff406c1858714b14eb0303802a3c3829b085454d10900555f7"
MAX_ASSET_BYTES = 70_000_000
API_RATE = 20.0
# Deezer currently begins returning API error 4 above roughly ten metadata
# requests/s. The signed protocol fixes a maximum of 20; operating below that
# ceiling avoids wasteful provider-limit retries without changing the method.
EFFECTIVE_API_RATE = 10.0
DOWNLOAD_WORKERS = 32
METADATA_WORKERS = 64
NETWORK_ATTEMPTS = 4
BACKOFF = (0.5, 1.0, 2.0, 4.0)
WINDOW_SAMPLES = 480_000
WINDOWS_PER_TRACK = 3
WINDOW_BATCH = 96
GEOMETRY_GATES = {
    "sampled_pair_cosine_spearman_min": 0.94,
    "mean_top50_overlap_min": 0.74,
    "mean_union_top50_rank_spearman_min": 0.55,
    "p05_top50_overlap_min": 0.55,
}
VARIANT_ORDER = (
    "conservative_clap_fallback",
    "graph_clap_union",
    "pure_clap",
)


class ClapCatalogError(ValueError):
    """The frozen catalogue build or its safety evidence is invalid."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _normalise_rows(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    return result / np.linalg.norm(result, axis=1, keepdims=True).clip(min=1e-8)


def validate_preregistration(path: Path) -> Dict[str, Any]:
    """Validate the immutable protocol and its detached Ed25519 signature."""
    document = json.loads(path.read_text(encoding="utf-8"))
    if (
        document.get("schema_version") != SCHEMA_VERSION
        or content_hash(document) != document.get("content_sha256")
        or document.get("content_sha256") != PREREGISTRATION_SHA256
        or document.get("ratings_count_at_freeze") != 0
        or document.get("development_only") is not True
        or document.get("ac3_claimed") is not False
    ):
        raise ClapCatalogError("CLAP v13 preregistration hash or safety state is invalid")
    revision = (
        "r3"
        if path.name == "preregistration-v13-r3.json"
        else ("r2" if path.name == "preregistration-v13-r2.json" else "r1")
    )
    signature = path.with_name(
        (
            f"preregistration-v13-{revision}.sig"
            if revision != "r1"
            else "preregistration-v13.sig"
        )
    )
    allowed = path.with_name(
        (
            f"prereg-{revision}-allowed-signers"
            if revision != "r1"
            else "prereg-allowed-signers"
        )
    )
    executable = shutil.which("ssh-keygen")
    if executable is None or not signature.is_file() or not allowed.is_file():
        raise ClapCatalogError("signed CLAP preregistration is incomplete")
    if revision == "r3" and any(
        not path.with_name(name).is_file()
        or _sha256_path(path.with_name(name)) != digest
        for name, digest in TRUSTED_PREREGISTRATION_R3_FILES.items()
    ):
        raise ClapCatalogError("CLAP preregistration signer differs from trust anchors")
    import subprocess

    verified = subprocess.run(
        [
            executable,
            "-Y",
            "verify",
            "-f",
            str(allowed),
            "-I",
            "soundalike-clap-prereg",
            "-n",
            "soundalike-clap-prereg",
            "-s",
            str(signature),
        ],
        input=path.read_bytes(),
        capture_output=True,
        check=False,
    )
    if verified.returncode:
        raise ClapCatalogError("CLAP preregistration signature is invalid")
    return document


def load_catalog_identity(index_path: Path) -> Dict[str, Any]:
    """Return immutable source arrays and fail on any row-order drift."""
    if _sha256_path(index_path) != (
        "f3ed57af1b8073f2872eed1e9192dee04d1089c7266fb98a157d1ea194526fb9"
    ):
        raise ClapCatalogError("catalog source index hash differs from preregistration")
    with np.load(index_path, allow_pickle=False) as index:
        ids = np.asarray(index["track_ids"], dtype=np.int64)
        titles = np.asarray(index["titles"])
        artists = np.asarray(index["artists"])
    if len(ids) != EXPECTED_ROWS or len(set(map(int, ids))) != EXPECTED_ROWS:
        raise ClapCatalogError("catalog must contain 272,853 unique stable Deezer IDs")
    digest = _sha256_bytes(ids.tobytes())
    if digest != TRACK_IDS_SHA256:
        raise ClapCatalogError("catalog track-ID row order differs from preregistration")
    return {
        "track_ids": ids,
        "titles": titles,
        "artists": artists,
        "track_ids_tobytes_sha256": digest,
    }


class RateLimiter:
    """Simple process-local monotonic limiter shared by download workers."""

    def __init__(self, rate: float = API_RATE):
        if rate <= 0:
            raise ValueError("rate must be positive")
        self.interval = 1.0 / float(rate)
        self.next_at = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next_at - now)
            self.next_at = max(self.next_at, now) + self.interval
        if delay:
            time.sleep(delay)


class EmbeddingStore:
    """SQLite status/checksum ledger plus exactly aligned float16 memmap."""

    def __init__(
        self,
        directory: Path,
        track_ids: np.ndarray,
        *,
        reset: bool = False,
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.database_path = self.directory / "status.sqlite3"
        self.embedding_path = self.directory / "full-clap512.f16.npy"
        if reset:
            self.database_path.unlink(missing_ok=True)
            self.embedding_path.unlink(missing_ok=True)
        create = not self.embedding_path.is_file()
        self.embeddings = (
            np.lib.format.open_memmap(
                self.embedding_path,
                mode="w+",
                dtype=np.float16,
                shape=(len(track_ids), FULL_DIM),
            )
            if create
            else np.load(self.embedding_path, mmap_mode="r+")
        )
        if self.embeddings.shape != (len(track_ids), FULL_DIM):
            raise ClapCatalogError("resumed CLAP embedding memmap has the wrong shape")
        self.connection = sqlite3.connect(self.database_path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS rows (
                row_index INTEGER PRIMARY KEY,
                track_id INTEGER NOT NULL UNIQUE,
                status TEXT NOT NULL DEFAULT 'pending',
                attempts INTEGER NOT NULL DEFAULT 0,
                preview_sha256 TEXT,
                embedding_sha256 TEXT,
                preview_bytes INTEGER,
                error TEXT,
                updated_at TEXT
            )
            """
        )
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS retry_failures (
                row_index INTEGER NOT NULL,
                pass_attempt INTEGER NOT NULL,
                error TEXT NOT NULL,
                observed_at TEXT NOT NULL
            )
            """
        )
        metadata = {
            "schema_version": str(SCHEMA_VERSION),
            "row_count": str(len(track_ids)),
            "full_dimension": str(FULL_DIM),
            "track_ids_tobytes_sha256": TRACK_IDS_SHA256,
            "preregistration_content_sha256": PREREGISTRATION_SHA256,
            "checkpoint_sha256": CHECKPOINT_SHA256,
        }
        for key, value in metadata.items():
            old = self.connection.execute(
                "SELECT value FROM metadata WHERE key=?", (key,)
            ).fetchone()
            if old is not None and old[0] != value:
                raise ClapCatalogError(f"resumed status metadata differs for {key}")
            self.connection.execute(
                "INSERT OR IGNORE INTO metadata(key,value) VALUES(?,?)", (key, value)
            )
        count = self.connection.execute("SELECT COUNT(*) FROM rows").fetchone()[0]
        if count == 0:
            self.connection.executemany(
                "INSERT INTO rows(row_index,track_id) VALUES(?,?)",
                ((row, int(track_id)) for row, track_id in enumerate(track_ids)),
            )
        elif count != len(track_ids):
            raise ClapCatalogError("resumed status row count differs from the catalog")
        for row, track_id in self.connection.execute(
            "SELECT row_index,track_id FROM rows ORDER BY row_index"
        ):
            if int(track_ids[row]) != int(track_id):
                raise ClapCatalogError("resumed status track IDs are not row-aligned")
        self.connection.commit()

    def pending(self, *, retry_errors: bool = True) -> List[tuple[int, int]]:
        if retry_errors:
            rows = self.connection.execute(
                """
                SELECT row_index,track_id FROM rows
                WHERE (status='pending' AND attempts < ?) OR status='error'
                ORDER BY row_index
                """,
                (NETWORK_ATTEMPTS,),
            ).fetchall()
        else:
            rows = self.connection.execute(
                """
                SELECT row_index,track_id FROM rows
                WHERE status='pending' AND attempts < ? ORDER BY row_index
                """,
                (NETWORK_ATTEMPTS,),
            ).fetchall()
        return [(int(row), int(track_id)) for row, track_id in rows]

    def starting_attempt(self, row: int) -> int:
        """Resume an interrupted pending row; give failed rows a fresh retry pass."""
        attempts, status = self.connection.execute(
            "SELECT attempts,status FROM rows WHERE row_index=?", (int(row),)
        ).fetchone()
        return 0 if status == "error" else int(attempts)

    def mark_terminal(
        self,
        row: int,
        status: str,
        *,
        attempts: int,
        preview_sha256: Optional[str] = None,
        embedding_sha256: Optional[str] = None,
        preview_bytes: Optional[int] = None,
        error: Optional[str] = None,
    ) -> None:
        if status not in {"available", "no_preview", "error"}:
            raise ValueError("invalid terminal status")
        self.connection.execute(
            """
            UPDATE rows SET status=?,attempts=?,preview_sha256=?,
              embedding_sha256=?,preview_bytes=?,error=?,updated_at=?
            WHERE row_index=?
            """,
            (
                status,
                int(attempts),
                preview_sha256,
                embedding_sha256,
                preview_bytes,
                error,
                _now(),
                int(row),
            ),
        )

    def record_retry_failures(self, item: "DownloadResult") -> None:
        self.connection.executemany(
            """
            INSERT INTO retry_failures(row_index,pass_attempt,error,observed_at)
            VALUES(?,?,?,?)
            """,
            (
                (int(item.row), int(attempt), str(error), _now())
                for attempt, error in enumerate(item.retry_failures, start=1)
            ),
        )

    def commit(self) -> None:
        self.embeddings.flush()
        self.connection.commit()

    def counts(self) -> Dict[str, int]:
        result = {
            str(status): int(count)
            for status, count in self.connection.execute(
                "SELECT status,COUNT(*) FROM rows GROUP BY status"
            )
        }
        for name in ("pending", "available", "no_preview", "error"):
            result.setdefault(name, 0)
        return result

    def failed_rows(self) -> List[Dict[str, Any]]:
        return [
            {
                "row": int(row),
                "deezer_track_id": int(track_id),
                "attempts": int(attempts),
                "error": str(error or ""),
            }
            for row, track_id, attempts, error in self.connection.execute(
                """
                SELECT row_index,track_id,attempts,error FROM rows
                WHERE status='error' ORDER BY row_index
                """
            )
        ]

    def retry_summary(self) -> Dict[str, Any]:
        rows = self.connection.execute(
            "SELECT row_index,error FROM retry_failures ORDER BY rowid"
        ).fetchall()
        error_types = Counter(
            str(error).split(":", 1)[0] for _, error in rows
        )
        return {
            "failed_attempts": len(rows),
            "rows_with_failed_attempts": len({int(row) for row, _ in rows}),
            "error_types": dict(sorted(error_types.items())),
            "detailed_failures_retained_in": str(self.database_path),
        }

    def cumulative_run_summary(self) -> Dict[str, Any]:
        first, last, preview_bytes, attempts = self.connection.execute(
            """
            SELECT MIN(updated_at),MAX(updated_at),
              COALESCE(SUM(preview_bytes),0),COALESCE(SUM(attempts),0)
            FROM rows WHERE updated_at IS NOT NULL
            """
        ).fetchone()
        elapsed = 0.0
        if first and last:
            elapsed = (
                datetime.fromisoformat(str(last))
                - datetime.fromisoformat(str(first))
            ).total_seconds()
        counts = self.counts()
        return {
            "first_completed_at": first,
            "last_completed_at": last,
            "wall_span_seconds": elapsed,
            "available_tracks_per_second": (
                counts["available"] / elapsed if elapsed > 0 else 0.0
            ),
            "downloaded_bytes": int(preview_bytes),
            "network_attempts": int(attempts),
        }

    def available_mask(self) -> np.ndarray:
        mask = np.zeros(len(self.embeddings), dtype=bool)
        rows = self.connection.execute(
            "SELECT row_index FROM rows WHERE status='available'"
        ).fetchall()
        if rows:
            mask[np.asarray([row[0] for row in rows], dtype=np.int64)] = True
        return mask

    def verify_available(self) -> None:
        for row, expected in self.connection.execute(
            """
            SELECT row_index,embedding_sha256 FROM rows
            WHERE status='available' ORDER BY row_index
            """
        ):
            vector = np.asarray(self.embeddings[int(row)])
            if (
                expected is None
                or not np.isfinite(vector).all()
                or float(np.linalg.norm(vector.astype(np.float32))) <= 0.0
                or _sha256_bytes(vector.tobytes()) != expected
            ):
                raise ClapCatalogError(f"available embedding row {row} is invalid")

    def close(self) -> None:
        self.commit()
        self.connection.close()


@dataclass
class DownloadResult:
    row: int
    track_id: int
    status: str
    attempts: int
    path: Optional[Path] = None
    preview_sha256: Optional[str] = None
    preview_bytes: Optional[int] = None
    error: Optional[str] = None
    retry_failures: List[str] = field(default_factory=list)


_NETWORK_LOCAL = threading.local()
_DOWNLOAD_SLOTS = threading.BoundedSemaphore(DOWNLOAD_WORKERS)


def _network_session() -> Any:
    """Return one keep-alive HTTP session per bounded worker thread."""
    session = getattr(_NETWORK_LOCAL, "session", None)
    if session is None:
        import requests

        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=4, pool_maxsize=4, max_retries=0
        )
        session.mount("https://", adapter)
        session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "soundalike-clap-development/13.0",
            }
        )
        _NETWORK_LOCAL.session = session
    return session


def _preview_url(track_id: int, limiter: RateLimiter, session: Any) -> Optional[str]:
    limiter.wait()
    response = session.get(
        f"https://api.deezer.com/track/{int(track_id)}",
        timeout=30,
    )
    if response.status_code != 200:
        raise OSError(f"Deezer metadata HTTP {response.status_code}")
    payload = response.json()
    if payload.get("error"):
        code = payload["error"].get("code")
        if code in {800, 803}:
            return None
        raise OSError(f"Deezer API error {code}")
    value = payload.get("preview")
    if not value:
        return None
    parsed = urlsplit(str(value))
    if parsed.scheme != "https" or not (
        parsed.hostname == "dzcdn.net"
        or (parsed.hostname or "").endswith(".dzcdn.net")
    ):
        raise ClapCatalogError("Deezer returned an untrusted preview origin")
    return str(value)


def _download_one(
    row: int,
    track_id: int,
    start_attempt: int,
    directory: Path,
    limiter: RateLimiter,
) -> DownloadResult:
    last_error = "unknown"
    retry_failures: List[str] = []
    attempts = int(start_attempt)
    while attempts < NETWORK_ATTEMPTS:
        attempts += 1
        path = directory / f"{row:06d}-{track_id}.mp3"
        try:
            session = _network_session()
            url = _preview_url(track_id, limiter, session)
            if url is None:
                return DownloadResult(
                    row,
                    track_id,
                    "no_preview",
                    attempts,
                    retry_failures=retry_failures,
                )
            # Signed URLs are used immediately and never written to status or logs.
            with _DOWNLOAD_SLOTS:
                response = session.get(url, timeout=45, stream=True)
                response.raise_for_status()
                digest = hashlib.sha256()
                size = 0
                first = True
                with path.open("wb") as stream:
                    for block in response.iter_content(chunk_size=64 * 1024):
                        if not block:
                            continue
                        digest.update(block)
                        size += len(block)
                        decoded_block = block
                        # Deezer previews currently carry an empty ten-byte
                        # ID3v2 header. Removing only that semantically empty
                        # header avoids one libmpg123 warning per track; the
                        # checksum and byte count still describe the exact
                        # downloaded response.
                        if (
                            first
                            and len(block) >= 10
                            and block[:3] == b"ID3"
                            and block[6:10] == b"\0\0\0\0"
                        ):
                            decoded_block = block[10:]
                        first = False
                        stream.write(decoded_block)
            if size < 1024:
                raise OSError("preview response was unexpectedly small")
            return DownloadResult(
                row,
                track_id,
                "downloaded",
                attempts,
                path=path,
                preview_sha256=digest.hexdigest(),
                preview_bytes=size,
                retry_failures=retry_failures,
            )
        except (HTTPError, OSError, TimeoutError, json.JSONDecodeError) as error:
            path.unlink(missing_ok=True)
            last_error = f"{type(error).__name__}: {str(error)[:160]}"
            retry_failures.append(last_error)
            if attempts < NETWORK_ATTEMPTS:
                time.sleep(BACKOFF[attempts - 1])
        except Exception as error:
            path.unlink(missing_ok=True)
            last_error = f"{type(error).__name__}: {str(error)[:160]}"
            retry_failures.append(last_error)
            if isinstance(error, ClapCatalogError):
                break
            if attempts < NETWORK_ATTEMPTS:
                time.sleep(BACKOFF[attempts - 1])
    return DownloadResult(
        row,
        track_id,
        "error",
        attempts,
        error=last_error,
        retry_failures=retry_failures,
    )


class FrozenClapEmbedder:
    """Exact pre-registered deterministic three-window CLAP extractor."""

    def __init__(self):
        import importlib.metadata
        import laion_clap
        import torch

        if importlib.metadata.version("laion-clap") != "1.1.7":
            raise ClapCatalogError("laion-clap must be exactly version 1.1.7")
        if not torch.cuda.is_available():
            raise ClapCatalogError("the frozen CLAP catalog build requires CUDA")
        self.torch = torch
        self.model = laion_clap.CLAP_Module(enable_fusion=False, device="cuda")
        self.model.load_ckpt(model_id=1, verbose=False)
        checkpoint = (
            Path(laion_clap.__file__).resolve().parent / "630k-audioset-best.pt"
        )
        if _sha256_path(checkpoint) != CHECKPOINT_SHA256:
            raise ClapCatalogError("LAION-CLAP checkpoint hash differs from preregistration")
        self.checkpoint = checkpoint
        self.gpu = torch.cuda.get_device_name(0)

    @staticmethod
    def _fixed_windows(waveform: np.ndarray) -> List[np.ndarray]:
        value = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if not len(value):
            raise ClapCatalogError("decoded preview is empty")
        if len(value) < WINDOW_SAMPLES:
            repeats = int(math.ceil(WINDOW_SAMPLES / len(value)))
            value = np.tile(value, repeats)[:WINDOW_SAMPLES]
            return [value.copy(), value.copy(), value.copy()]
        overflow = len(value) - WINDOW_SAMPLES
        offsets = (0, overflow // 2, overflow)
        return [
            np.asarray(value[offset : offset + WINDOW_SAMPLES], dtype=np.float32)
            for offset in offsets
        ]

    def embed_files(self, paths: Sequence[Path]) -> np.ndarray:
        import librosa
        from laion_clap.hook import float32_to_int16, int16_to_float32
        from laion_clap.training.data import get_audio_features

        features: List[Dict[str, Any]] = []
        for path in paths:
            waveform, _ = librosa.load(str(path), sr=48_000, mono=True)
            for window in self._fixed_windows(waveform):
                quantized = int16_to_float32(float32_to_int16(window))
                tensor = self.torch.from_numpy(quantized).float()
                sample: Dict[str, Any] = {}
                features.append(
                    get_audio_features(
                        sample,
                        tensor,
                        WINDOW_SAMPLES,
                        data_truncating="rand_trunc",
                        data_filling="repeatpad",
                        audio_cfg=self.model.model_cfg["audio_cfg"],
                        require_grad=False,
                    )
                )
        outputs: List[np.ndarray] = []
        self.model.model.eval()
        with self.torch.inference_mode():
            for start in range(0, len(features), WINDOW_BATCH):
                embedding = self.model.model.get_audio_embedding(
                    features[start : start + WINDOW_BATCH]
                )
                outputs.append(embedding.float().cpu().numpy())
        windows = _normalise_rows(np.concatenate(outputs))
        pooled = windows.reshape(len(paths), WINDOWS_PER_TRACK, FULL_DIM).mean(axis=1)
        return _normalise_rows(pooled)


def build_full_embeddings(
    index_path: Path,
    output_dir: Path,
    preregistration: Path,
    *,
    chunk_tracks: int = 32,
    retry_errors: bool = True,
    reset: bool = False,
    max_rows: Optional[int] = None,
    embedder_factory: Callable[[], Any] = FrozenClapEmbedder,
    downloader: Callable[..., DownloadResult] = _download_one,
) -> Dict[str, Any]:
    """Resume fresh preview extraction until all requested reachable rows finish."""
    validate_preregistration(preregistration)
    catalog = load_catalog_identity(index_path)
    store = EmbeddingStore(output_dir, catalog["track_ids"], reset=reset)
    pending = store.pending(retry_errors=retry_errors)
    if max_rows is not None:
        pending = pending[: max(0, int(max_rows))]
    stale = output_dir / "tmp"
    shutil.rmtree(stale, ignore_errors=True)
    stale.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    embedded = 0
    downloaded_bytes = 0
    embedder = None
    limiter = RateLimiter(EFFECTIVE_API_RATE)
    try:
        with ThreadPoolExecutor(max_workers=METADATA_WORKERS) as executor:
            step = max(1, int(chunk_tracks))
            offsets = list(range(0, len(pending), step))

            def submit_batch(offset: int) -> List[Any]:
                batch = pending[offset : offset + step]
                return [
                    executor.submit(
                        downloader,
                        row,
                        track_id,
                        store.starting_attempt(row),
                        stale,
                        limiter,
                    )
                    for row, track_id in batch
                ]

            futures = submit_batch(offsets[0]) if offsets else []
            for position, offset in enumerate(offsets):
                results = [future.result() for future in futures]
                # Resolve/download the next bounded batch while CPU decoding and
                # GPU inference consume this one. Signed URLs still remain
                # worker-local and are used immediately.
                futures = (
                    submit_batch(offsets[position + 1])
                    if position + 1 < len(offsets)
                    else []
                )
                downloadable = [item for item in results if item.status == "downloaded"]
                for item in results:
                    store.record_retry_failures(item)
                    if item.status == "no_preview":
                        store.mark_terminal(
                            item.row, "no_preview", attempts=item.attempts
                        )
                    elif item.status == "error":
                        store.mark_terminal(
                            item.row,
                            "error",
                            attempts=item.attempts,
                            error=item.error,
                        )
                if downloadable:
                    if embedder is None:
                        embedder = embedder_factory()
                    paths = [item.path for item in downloadable]
                    if any(path is None for path in paths):
                        raise ClapCatalogError("downloaded batch lacks a temporary path")
                    try:
                        values = embedder.embed_files(paths)  # type: ignore[arg-type]
                        if values.shape != (len(downloadable), FULL_DIM):
                            raise ClapCatalogError("CLAP extractor returned the wrong shape")
                        for item, vector in zip(downloadable, values):
                            half = np.asarray(vector, dtype=np.float16)
                            if (
                                not np.isfinite(half).all()
                                or float(np.linalg.norm(half.astype(np.float32))) <= 0
                            ):
                                raise ClapCatalogError("CLAP extractor returned an invalid row")
                            store.embeddings[item.row] = half
                            store.mark_terminal(
                                item.row,
                                "available",
                                attempts=item.attempts,
                                preview_sha256=item.preview_sha256,
                                embedding_sha256=_sha256_bytes(half.tobytes()),
                                preview_bytes=item.preview_bytes,
                            )
                            embedded += 1
                            downloaded_bytes += int(item.preview_bytes or 0)
                    except Exception as error:
                        for item in downloadable:
                            store.embeddings[item.row] = 0
                            store.mark_terminal(
                                item.row,
                                "error",
                                attempts=item.attempts,
                                preview_sha256=item.preview_sha256,
                                preview_bytes=item.preview_bytes,
                                error=f"embedding {type(error).__name__}: {str(error)[:160]}",
                            )
                        if isinstance(error, ClapCatalogError):
                            raise
                    finally:
                        for item in downloadable:
                            if item.path is not None:
                                item.path.unlink(missing_ok=True)
                store.commit()
                if embedder is not None and offset and offset % (chunk_tracks * 25) == 0:
                    elapsed = max(time.perf_counter() - started, 1e-9)
                    counts = store.counts()
                    print(
                        f"CLAP v13 {sum(counts.values()) - counts['pending']:,}/"
                        f"{EXPECTED_ROWS:,} terminal; {embedded / elapsed:.2f} embedded/s",
                        flush=True,
                    )
        store.verify_available()
        counts = store.counts()
        elapsed = time.perf_counter() - started
        torch = getattr(embedder, "torch", None)
        report: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": "clap_catalog_embedding_coverage",
            "created_at": _now(),
            "preregistration_content_sha256": PREREGISTRATION_SHA256,
            "catalog": {
                "rows": EXPECTED_ROWS,
                "track_ids_tobytes_sha256": TRACK_IDS_SHA256,
            },
            "encoder": {
                "checkpoint_sha256": CHECKPOINT_SHA256,
                "gpu": getattr(embedder, "gpu", None),
                "dimension": FULL_DIM,
            },
            "coverage": counts,
            "available_fraction": counts["available"] / EXPECTED_ROWS,
            "terminal_fraction": (
                counts["available"] + counts["no_preview"] + counts["error"]
            )
            / EXPECTED_ROWS,
            "retry_failures": store.failed_rows(),
            "retry_history": store.retry_summary(),
            "run": {
                "requested_rows": len(pending),
                "embedded_rows": embedded,
                "wall_seconds": elapsed,
                "embedded_tracks_per_second": embedded / max(elapsed, 1e-9),
                "downloaded_bytes": downloaded_bytes,
                "download_workers": DOWNLOAD_WORKERS,
                "metadata_workers": METADATA_WORKERS,
                "api_rate_limit_per_second": API_RATE,
                "effective_api_requests_per_second": EFFECTIVE_API_RATE,
                "temporary_files_remaining": len(list(stale.iterdir())),
                "cumulative": store.cumulative_run_summary(),
            },
            "gpu": {
                "max_allocated_bytes": (
                    int(torch.cuda.max_memory_allocated()) if torch is not None else 0
                ),
                "max_reserved_bytes": (
                    int(torch.cuda.max_memory_reserved()) if torch is not None else 0
                ),
            },
            "cache": {
                "embedding_path": str(store.embedding_path),
                "embedding_bytes": int(store.embedding_path.stat().st_size),
                "embedding_sha256": _sha256_path(store.embedding_path),
                "status_path": str(store.database_path),
                "status_bytes": int(store.database_path.stat().st_size),
                "retained_audio_files": 0,
                "signed_preview_urls_persisted": False,
            },
            "production_changed": False,
            "commercial_final_opened": False,
            "ac3_claimed": False,
        }
        report["content_sha256"] = content_hash(report)
        _write_json(output_dir / "coverage-report.json", report)
        return report
    finally:
        shutil.rmtree(stale, ignore_errors=True)
        store.close()


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks with deterministic tie handling (SciPy-free)."""
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        result[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return result


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    a, b = _rankdata(np.asarray(left).ravel()), _rankdata(np.asarray(right).ravel())
    a -= a.mean()
    b -= b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / denominator) if denominator else 0.0


def orthogonal_projection(dimension: int, seed: int = SEED) -> np.ndarray:
    """Return the frozen first ``dimension`` columns of a 512-D Gaussian QR."""
    if dimension not in DIMENSIONS:
        raise ValueError(f"dimension must be one of {DIMENSIONS}")
    rng = np.random.default_rng(seed)
    gaussian = rng.standard_normal((FULL_DIM, max(DIMENSIONS)), dtype=np.float32)
    orthogonal, _ = np.linalg.qr(gaussian)
    return np.asarray(orthogonal[:, :dimension], dtype=np.float32)


def geometry_metrics(
    full: np.ndarray,
    compact: np.ndarray,
    query_rows: np.ndarray,
    reference_rows: np.ndarray,
    *,
    top_k: int = 50,
    pair_samples: int = 100_000,
) -> Dict[str, float]:
    full_queries = _normalise_rows(full[query_rows])
    full_refs = _normalise_rows(full[reference_rows])
    compact_queries = _normalise_rows(compact[query_rows])
    compact_refs = _normalise_rows(compact[reference_rows])
    exact = full_queries @ full_refs.T
    reduced = compact_queries @ compact_refs.T
    rng = np.random.default_rng(SEED)
    positions = rng.choice(
        exact.size, size=min(int(pair_samples), exact.size), replace=False
    )
    overlaps: List[float] = []
    rank_correlations: List[float] = []
    count = min(int(top_k), exact.shape[1])
    for left, right in zip(exact, reduced):
        left_top = np.argpartition(-left, count - 1)[:count]
        right_top = np.argpartition(-right, count - 1)[:count]
        overlaps.append(len(set(left_top) & set(right_top)) / count)
        union = np.asarray(sorted(set(left_top) | set(right_top)), dtype=np.int64)
        rank_correlations.append(_spearman(left[union], right[union]))
    return {
        "sampled_pair_cosine_spearman": _spearman(
            exact.ravel()[positions], reduced.ravel()[positions]
        ),
        "mean_top50_overlap": float(np.mean(overlaps)),
        "p05_top50_overlap": float(np.quantile(overlaps, 0.05)),
        "mean_union_top50_rank_spearman": float(np.mean(rank_correlations)),
    }


def _geometry_passes(metrics: Mapping[str, float]) -> bool:
    return bool(
        metrics["sampled_pair_cosine_spearman"]
        >= GEOMETRY_GATES["sampled_pair_cosine_spearman_min"]
        and metrics["mean_top50_overlap"]
        >= GEOMETRY_GATES["mean_top50_overlap_min"]
        and metrics["mean_union_top50_rank_spearman"]
        >= GEOMETRY_GATES["mean_union_top50_rank_spearman_min"]
        and metrics["p05_top50_overlap"]
        >= GEOMETRY_GATES["p05_top50_overlap_min"]
    )


def compress_embeddings(
    index_path: Path,
    cache_dir: Path,
    preregistration: Path,
    *,
    query_count: int = 256,
    reference_count: int = 20_000,
) -> Dict[str, Any]:
    """Select and materialize the smallest pre-registered passing JL dimension."""
    validate_preregistration(preregistration)
    load_catalog_identity(index_path)
    store = EmbeddingStore(cache_dir, load_catalog_identity(index_path)["track_ids"])
    try:
        store.verify_available()
        mask = store.available_mask()
        counts = store.counts()
        if (
            counts["pending"] != 0
            or counts["error"] != 0
            or counts["available"] + counts["no_preview"] != EXPECTED_ROWS
        ):
            raise ClapCatalogError(
                "compression requires a complete catalog: zero pending/error rows"
            )
        coverage_path = cache_dir / "coverage-report.json"
        if not coverage_path.is_file():
            raise ClapCatalogError("complete embedding coverage report is missing")
        coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
        if (
            content_hash(coverage) != coverage.get("content_sha256")
            or coverage.get("coverage") != counts
        ):
            raise ClapCatalogError("embedding coverage report is stale or invalid")
        available = np.flatnonzero(mask)
        needed = int(query_count) + int(reference_count)
        if len(available) < needed:
            raise ClapCatalogError(
                f"geometry audit needs {needed:,} available rows, got {len(available):,}"
            )
        rng = np.random.default_rng(SEED)
        sample = rng.choice(available, size=needed, replace=False)
        queries = sample[:query_count]
        references = sample[query_count:]
        full = store.embeddings
        diagnostics: Dict[str, Any] = {}
        selected: Optional[int] = None
        projection = orthogonal_projection(max(DIMENSIONS))
        for dimension in DIMENSIONS:
            subset = np.concatenate((queries, references))
            projected_subset = _normalise_rows(
                _normalise_rows(full[subset]) @ projection[:, :dimension]
            )
            compact_subset = np.zeros(
                (len(full), dimension), dtype=np.float32
            )
            compact_subset[subset] = projected_subset
            metrics = geometry_metrics(
                full,
                compact_subset,
                queries,
                references,
            )
            metrics["passes"] = _geometry_passes(metrics)
            metrics["asset_bytes"] = EXPECTED_ROWS * dimension * 2 + 128
            diagnostics[str(dimension)] = metrics
            if selected is None and metrics["passes"]:
                selected = dimension
        if selected is None:
            raise ClapCatalogError("no pre-registered compact dimension passed geometry")
        output = cache_dir / f"compact-clap{selected}.f16.npy"
        compact = np.lib.format.open_memmap(
            output,
            mode="w+",
            dtype=np.float16,
            shape=(EXPECTED_ROWS, selected),
        )
        matrix = projection[:, :selected]
        for start in range(0, EXPECTED_ROWS, 4096):
            stop = min(start + 4096, EXPECTED_ROWS)
            valid = mask[start:stop]
            block = np.zeros((stop - start, selected), dtype=np.float32)
            if np.any(valid):
                block[valid] = _normalise_rows(
                    _normalise_rows(full[start:stop][valid]) @ matrix
                )
            compact[start:stop] = block.astype(np.float16)
        compact.flush()
        if output.stat().st_size > MAX_ASSET_BYTES:
            raise ClapCatalogError("compact CLAP asset exceeds the 70 MB gate")
        reloaded = np.load(output, mmap_mode="r")
        if reloaded.shape != (EXPECTED_ROWS, selected) or reloaded.dtype != np.float16:
            raise ClapCatalogError("compact CLAP asset shape/dtype is invalid")
        reloaded_metrics = geometry_metrics(
            full, reloaded, queries, references
        )
        if not _geometry_passes(reloaded_metrics):
            raise ClapCatalogError("float16 reload failed compact geometry gates")
        projection_path = cache_dir / f"compact-clap{selected}-projection.npz"
        np.savez_compressed(
            projection_path,
            matrix=matrix,
            seed=np.asarray(SEED, dtype=np.int64),
            source_dimension=np.asarray(FULL_DIM, dtype=np.int64),
            target_dimension=np.asarray(selected, dtype=np.int64),
        )
        report: Dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "artifact_kind": "clap_catalog_compact_geometry",
            "created_at": _now(),
            "preregistration_content_sha256": PREREGISTRATION_SHA256,
            "algorithm": "Gaussian orthogonal Johnson-Lindenstrauss projection",
            "seed": SEED,
            "selection_uses_labels": False,
            "available_rows": int(mask.sum()),
            "unavailable_rows": int((~mask).sum()),
            "coverage_content_sha256": coverage["content_sha256"],
            "coverage": counts,
            "sample": {
                "queries": int(query_count),
                "references": int(reference_count),
                "query_reference_disjoint": True,
                "row_selection_sha256": _sha256_bytes(sample.tobytes()),
            },
            "gates": GEOMETRY_GATES,
            "dimensions": diagnostics,
            "selected_dimension": selected,
            "float16_reload_metrics": reloaded_metrics,
            "asset": {
                "path": str(output),
                "bytes": int(output.stat().st_size),
                "sha256": _sha256_path(output),
                "dtype": "float16",
                "shape": [EXPECTED_ROWS, selected],
                "normalised": True,
            },
            "projection": {
                "path": str(projection_path),
                "bytes": int(projection_path.stat().st_size),
                "sha256": _sha256_path(projection_path),
            },
            "production_changed": False,
            "deployed": False,
        }
        report["content_sha256"] = content_hash(report)
        _write_json(cache_dir / "compact-report.json", report)
        return report
    finally:
        store.close()


def _top_indices(scores: np.ndarray, count: int) -> np.ndarray:
    count = min(max(int(count), 0), len(scores))
    if not count:
        return np.empty(0, dtype=np.int64)
    selected = np.argpartition(-scores, count - 1)[:count]
    return selected[np.lexsort((selected, -scores[selected]))]


class ClapDevelopmentRanker:
    """Three frozen CLAP list variants with shared quality/diversity controls."""

    def __init__(
        self,
        index_path: Path,
        compact_path: Path,
        status_path: Path,
        graph_path: Path,
    ):
        from .catalog_graph import CatalogArtistGraph
        from webapp.api._reco import WebRecommender

        self.production = WebRecommender(str(index_path), enhance=False)
        self.production_ranker = ProductionRanker(self.production, set(), seed=SEED)
        self.track_ids = np.asarray(self.production.track_ids, dtype=np.int64)
        self.titles = np.asarray(self.production.titles)
        self.artists = np.asarray(self.production.artists)
        self.compact = np.load(compact_path, mmap_mode="r")
        if len(self.compact) != len(self.track_ids):
            raise ClapCatalogError("compact CLAP rows are not aligned to production")
        connection = sqlite3.connect(status_path)
        try:
            metadata = dict(connection.execute("SELECT key,value FROM metadata"))
            expected_metadata = {
                "schema_version": str(SCHEMA_VERSION),
                "row_count": str(len(self.track_ids)),
                "full_dimension": str(FULL_DIM),
                "track_ids_tobytes_sha256": TRACK_IDS_SHA256,
                "preregistration_content_sha256": PREREGISTRATION_SHA256,
                "checkpoint_sha256": CHECKPOINT_SHA256,
            }
            if any(metadata.get(key) != value for key, value in expected_metadata.items()):
                raise ClapCatalogError(
                    "CLAP availability ledger is not bound to this catalog/protocol"
                )
            row_contract = connection.execute(
                "SELECT COUNT(*),MIN(row_index),MAX(row_index) FROM rows"
            ).fetchone()
            if row_contract != (len(self.track_ids), 0, len(self.track_ids) - 1):
                raise ClapCatalogError("CLAP availability ledger rows are incomplete")
            self.available = np.zeros(len(self.track_ids), dtype=bool)
            rows = connection.execute(
                "SELECT row_index FROM rows WHERE status='available'"
            ).fetchall()
            if rows:
                self.available[np.asarray([row[0] for row in rows], dtype=np.int64)] = True
        finally:
            connection.close()
        compact_report_path = compact_path.parent / "compact-report.json"
        if not compact_report_path.is_file():
            raise ClapCatalogError("compact CLAP report is missing")
        compact_report = json.loads(compact_report_path.read_text(encoding="utf-8"))
        if (
            content_hash(compact_report) != compact_report.get("content_sha256")
            or compact_report.get("preregistration_content_sha256")
            != PREREGISTRATION_SHA256
            or compact_report.get("asset", {}).get("sha256")
            != _sha256_path(compact_path)
            or compact_report.get("coverage", {}).get("pending") != 0
            or compact_report.get("coverage", {}).get("error") != 0
        ):
            raise ClapCatalogError("compact CLAP asset/report binding is invalid")
        compact_available = np.zeros(len(self.compact), dtype=bool)
        for start in range(0, len(self.compact), 8192):
            stop = min(start + 8192, len(self.compact))
            compact_available[start:stop] = np.any(
                np.asarray(self.compact[start:stop]) != 0, axis=1
            )
        if not np.array_equal(compact_available, self.available):
            raise ClapCatalogError(
                "compact CLAP nonzero rows differ from the availability ledger"
            )
        self.graph = CatalogArtistGraph(graph_path)
        if not np.array_equal(
            np.sort(np.asarray(self.graph.track_rows, dtype=np.int64)),
            np.arange(len(self.track_ids), dtype=np.int64),
        ):
            raise ClapCatalogError("catalog graph track rows are not complete/aligned")
        self.quality = TitleQualityFilter()
        self.rows_by_track_id = {
            int(track_id): row for row, track_id in enumerate(self.track_ids)
        }

    def query_row(self, track_id: int) -> int:
        try:
            return int(self.rows_by_track_id[int(track_id)])
        except KeyError:
            raise ClapCatalogError(f"seed Deezer ID {track_id} is absent") from None

    def production_rows(self, row: int, n: int = 5) -> List[int]:
        return self.production_ranker.rank(row, "dual_sonic", n=n)

    def clap_scores(self, row: int) -> Optional[np.ndarray]:
        if not self.available[row]:
            return None
        query = np.asarray(self.compact[row], dtype=np.float32)
        query /= max(float(np.linalg.norm(query)), 1e-8)
        scores = np.empty(len(self.compact), dtype=np.float32)
        for start in range(0, len(scores), 8192):
            stop = min(start + 8192, len(scores))
            scores[start:stop] = _normalise_rows(
                np.asarray(self.compact[start:stop], dtype=np.float32)
            ) @ query
        scores[~self.available] = -np.inf
        scores[row] = -np.inf
        return scores

    def _graph_candidates(
        self, row: int
    ) -> tuple[np.ndarray, Dict[int, float], Dict[int, float], float, bool]:
        neighborhood = self.graph.dual_source_neighbors(str(self.artists[row]))
        last_ids = np.asarray(neighborhood["lastfm"]["artist_ids"], dtype=np.int64)
        last_weights = np.asarray(neighborhood["lastfm"]["weights"], dtype=np.float32)
        music_ids = np.asarray(neighborhood["music4all"]["artist_ids"], dtype=np.int64)
        music_weights = np.asarray(
            neighborhood["music4all"]["weights"], dtype=np.float32
        )
        last_max = max(float(last_weights.max()) if len(last_weights) else 0.0, 1e-8)
        music_max = max(
            float(music_weights.max()) if len(music_weights) else 0.0, 1e-8
        )
        last = {
            int(artist_id): float(weight / last_max)
            for artist_id, weight in zip(last_ids, last_weights)
        }
        music = {
            int(artist_id): float(weight / music_max)
            for artist_id, weight in zip(music_ids, music_weights)
        }
        artist_ids = sorted(set(last) | set(music))
        rows: List[int] = []
        for artist_id in artist_ids:
            start = int(self.graph.track_indptr[artist_id])
            stop = int(self.graph.track_indptr[artist_id + 1])
            rows.extend(map(int, self.graph.track_rows[start:stop]))
        confidence = (
            float(np.mean(np.sort(last_weights / last_max)[-5:]))
            if len(last_weights)
            else 0.0
        )
        return (
            np.asarray(sorted(set(rows)), dtype=np.int64),
            last,
            music,
            confidence,
            bool(neighborhood["source_coverage"]["lastfm"]),
        )

    def _eligible(
        self, query_row: int, candidates: Iterable[int], relevance: np.ndarray
    ) -> List[Dict[str, Any]]:
        from .catalog_policy import _artist_parts

        seed_title = str(self.titles[query_row])
        seed_artist = str(self.artists[query_row])
        seed_parts = _artist_parts(seed_artist)
        values: List[Dict[str, Any]] = []
        for raw in candidates:
            row = int(raw)
            title, artist = str(self.titles[row]), str(self.artists[row])
            if (
                row == query_row
                or not self.available[row]
                or seed_parts & _artist_parts(artist)
                or not self.quality.is_eligible_for_query(
                    seed_title, seed_artist, title, artist
                )
                or self.quality.seed_title_in_result(seed_title, title)
            ):
                continue
            values.append(
                {
                    "row": row,
                    "title": title,
                    "artist": artist,
                    "artist_key": normalize_text(artist),
                    "relevance": float(relevance[row]),
                }
            )
        values.sort(key=lambda item: (-item["relevance"], item["row"]))
        return [dict(item) for item in self.quality.prefer_canonical(values)]

    def _mmr(
        self, query_row: int, candidates: List[Dict[str, Any]], n: int = 5
    ) -> List[int]:
        selected: List[Dict[str, Any]] = []
        used_artists = set()
        remaining = list(candidates)
        while remaining and len(selected) < n:
            best = None
            best_key = None
            for item in remaining:
                if item["artist_key"] in used_artists:
                    continue
                row = int(item["row"])
                diversity = 0.0
                if selected:
                    candidate = np.asarray(self.compact[row], dtype=np.float32)
                    candidate /= max(float(np.linalg.norm(candidate)), 1e-8)
                    diversity = max(
                        (
                            float(
                                candidate
                                @ (
                                    np.asarray(
                                        self.compact[int(chosen["row"])],
                                        dtype=np.float32,
                                    )
                                    / max(
                                        float(
                                            np.linalg.norm(
                                                np.asarray(
                                                    self.compact[int(chosen["row"])],
                                                    dtype=np.float32,
                                                )
                                            )
                                        ),
                                        1e-8,
                                    )
                                )
                            )
                            + 1.0
                        )
                        / 2.0
                        for chosen in selected
                    )
                score = 0.85 * float(item["relevance"]) - 0.15 * diversity
                key = (score, -row)
                if best_key is None or key > best_key:
                    best, best_key = item, key
            if best is None:
                break
            selected.append(best)
            used_artists.add(best["artist_key"])
            remaining.remove(best)
        return [int(item["row"]) for item in selected]

    def rank_all(self, query_row: int) -> Dict[str, Dict[str, Any]]:
        production = self.production_rows(query_row, 5)
        clap = self.clap_scores(query_row)
        if clap is None:
            return {
                name: {
                    "rows": list(production),
                    "query_available": False,
                    "gate_fired": False,
                    "fallback_reason": "seed_preview_unavailable",
                    "candidate_count": 0,
                    "candidate_rows": [],
                }
                for name in VARIANT_ORDER
            }
        clap01 = (clap + 1.0) / 2.0
        pure_candidates = _top_indices(clap, 500)
        pure_eligible = self._eligible(query_row, pure_candidates, clap01)
        pure = self._mmr(query_row, pure_eligible)

        graph_rows, last, music, confidence, has_lastfm = self._graph_candidates(
            query_row
        )
        union_rows = np.asarray(
            sorted(set(map(int, graph_rows)) | set(map(int, _top_indices(clap, 200)))),
            dtype=np.int64,
        )
        graph_relevance = np.full(len(self.track_ids), -np.inf, dtype=np.float32)
        graph_only_relevance = np.full(len(self.track_ids), -np.inf, dtype=np.float32)
        for row in union_rows:
            artist_id = int(self.graph.track_artist_ids[int(row)])
            graph_relevance[int(row)] = (
                0.70 * float(clap01[int(row)])
                + 0.25 * last.get(artist_id, 0.0)
                + 0.05 * music.get(artist_id, 0.0)
            )
        for row in graph_rows:
            artist_id = int(self.graph.track_artist_ids[int(row)])
            graph_only_relevance[int(row)] = (
                0.80 * last.get(artist_id, 0.0)
                + 0.20 * float(clap01[int(row)])
            )
        graph_eligible = self._eligible(query_row, union_rows, graph_relevance)
        graph_ranked = self._mmr(query_row, graph_eligible)
        conservative_eligible = self._eligible(
            query_row, graph_rows, graph_only_relevance
        )
        consistent = [
            item
            for item in conservative_eligible
            if float(clap01[int(item["row"])]) >= 0.65
        ]
        fired = has_lastfm and confidence >= 0.55 and len(consistent) >= 5
        conservative = (
            self._mmr(query_row, consistent)
            if fired
            else list(production)
        )
        return {
            "pure_clap": {
                "rows": pure,
                "query_available": True,
                "gate_fired": True,
                "fallback_reason": None,
                "candidate_count": len(pure_eligible),
                "candidate_rows": [
                    int(item["row"]) for item in pure_eligible[:200]
                ],
            },
            "graph_clap_union": {
                "rows": graph_ranked,
                "query_available": True,
                "gate_fired": True,
                "fallback_reason": None,
                "candidate_count": len(graph_eligible),
                "candidate_rows": [
                    int(item["row"]) for item in graph_eligible[:200]
                ],
            },
            "conservative_clap_fallback": {
                "rows": conservative,
                "query_available": True,
                "gate_fired": fired,
                "fallback_reason": (
                    None
                    if fired
                    else (
                        "missing_lastfm"
                        if not has_lastfm
                        else (
                            "lastfm_confidence"
                            if confidence < 0.55
                            else "fewer_than_five_consistent_candidates"
                        )
                    )
                ),
                "candidate_count": len(consistent),
                "candidate_rows": [
                    int(item["row"]) for item in consistent[:200]
                ],
                "lastfm_confidence": confidence,
            },
        }


def _deezer_related(artist: str) -> List[str]:
    from urllib.parse import urlencode

    headers = {
        "Accept": "application/json",
        "User-Agent": "soundalike-clap-development/13.0",
    }
    search = Request(
        "https://api.deezer.com/search/artist?" + urlencode({"q": artist}),
        headers=headers,
        method="GET",
    )
    with urlopen(search, timeout=30) as response:
        rows = json.loads(response.read().decode("utf-8")).get("data", [])
    if not rows:
        return []
    related = Request(
        f"https://api.deezer.com/artist/{int(rows[0]['id'])}/related",
        headers=headers,
        method="GET",
    )
    with urlopen(related, timeout=30) as response:
        values = json.loads(response.read().decode("utf-8")).get("data", [])
    return [normalize_text(str(item.get("name", ""))) for item in values]


def _list_metrics(
    rows_by_seed: Sequence[Mapping[str, Any]],
    ranker: ClapDevelopmentRanker,
    style: Any,
    deezer_truth: Mapping[str, set[str]],
) -> Dict[str, Any]:
    all_rows = [
        int(row)
        for seed in rows_by_seed
        for row in seed.get("rows", ())
    ]
    artists = [normalize_text(str(ranker.artists[row])) for row in all_rows]
    track_counts = Counter(all_rows)
    artist_counts = Counter(artists)
    junk = 0
    same_artist = 0
    style_values: List[float] = []
    affinity_hits = 0
    affinity_total = 0
    affinity_seed_count = 0
    for seed in rows_by_seed:
        query_row = int(seed["query_row"])
        query_artist = str(ranker.artists[query_row])
        related = deezer_truth.get(str(seed["seed_id"]), set())
        affinity_seed_count += int(bool(related))
        for row in seed.get("rows", ()):
            row = int(row)
            title, artist = str(ranker.titles[row]), str(ranker.artists[row])
            junk += int(
                not ranker.quality.is_eligible_for_query(
                    str(ranker.titles[query_row]), query_artist, title, artist
                )
            )
            same_artist += int(
                normalize_text(query_artist) == normalize_text(artist)
            )
            style_values.append(style.style_overlap(query_artist, artist))
            if related:
                affinity_total += 1
                affinity_hits += int(normalize_text(artist) in related)
    slots = max(len(all_rows), 1)
    return {
        "seed_count": len(rows_by_seed),
        "complete_top5_count": sum(len(seed.get("rows", ())) == 5 for seed in rows_by_seed),
        "slots": len(all_rows),
        "junk_or_version_count": junk,
        "same_artist_count": same_artist,
        "unique_tracks": len(track_counts),
        "unique_artists": len(artist_counts),
        "unique_artist_slot_fraction": len(artist_counts) / slots,
        "maximum_track_slot_fraction": max(track_counts.values(), default=0) / slots,
        "maximum_artist_slot_fraction": max(artist_counts.values(), default=0) / slots,
        "mean_style_overlap": float(np.mean(style_values)) if style_values else 0.0,
        "deezer_related_artist_hits": affinity_hits,
        "deezer_related_artist_total": affinity_total,
        "deezer_related_artist_seed_count": affinity_seed_count,
        "deezer_related_artist_hit_rate": (
            affinity_hits / affinity_total if affinity_total else 0.0
        ),
    }


def _proxy_passes(metrics: Mapping[str, Any], baseline: Mapping[str, Any]) -> bool:
    return bool(
        metrics["seed_count"] == 60
        and metrics["complete_top5_count"] == 60
        and metrics["junk_or_version_count"] == 0
        and metrics["same_artist_count"] == 0
        and metrics["maximum_track_slot_fraction"] <= 0.03
        and metrics["maximum_artist_slot_fraction"] <= 0.05
        and metrics["unique_artist_slot_fraction"] >= 0.60
        and metrics["deezer_related_artist_seed_count"] >= 45
        and metrics["mean_style_overlap"] - baseline["mean_style_overlap"] >= -0.05
        and metrics["deezer_related_artist_hit_rate"]
        - baseline["deezer_related_artist_hit_rate"]
        >= -0.05
    )


def run_variant_diagnostics(
    index_path: Path,
    compact_path: Path,
    status_path: Path,
    graph_path: Path,
    style_path: Path,
    seed_lists_path: Path,
    output_path: Path,
    *,
    deezer_fetcher: Callable[[str], List[str]] = _deezer_related,
) -> Dict[str, Any]:
    """Run non-deciding proxy safety and freeze one human-test challenger."""
    from .catalog_style import CatalogStyleIndex

    ranker = ClapDevelopmentRanker(
        index_path, compact_path, status_path, graph_path
    )
    style = CatalogStyleIndex(style_path)
    source = json.loads(seed_lists_path.read_text(encoding="utf-8"))
    if source.get("seed_count") != 60 or source.get("scene_count") != 13:
        raise ClapCatalogError("v13 diagnostics require the frozen 60-seed/13-scene suite")
    deezer_truth: Dict[str, set[str]] = {}
    for seed in source["seeds"]:
        try:
            deezer_truth[str(seed["seed_id"])] = set(
                deezer_fetcher(str(seed["query"]["artist"]))
            )
        except Exception:
            deezer_truth[str(seed["seed_id"])] = set()

    baseline_records: List[Dict[str, Any]] = []
    variants: Dict[str, List[Dict[str, Any]]] = {
        name: [] for name in VARIANT_ORDER
    }
    scene_counts = Counter()
    latencies: List[float] = []
    for seed in source["seeds"]:
        scene_counts[str(seed["scene"])] += 1
        query_row = ranker.query_row(int(seed["query"]["deezer_track_id"]))
        baseline = ranker.production_rows(query_row, 5)
        baseline_records.append(
            {
                "seed_id": seed["seed_id"],
                "scene": seed["scene"],
                "query_row": query_row,
                "rows": baseline,
            }
        )
        started = time.perf_counter()
        ranked = ranker.rank_all(query_row)
        latencies.append(time.perf_counter() - started)
        for name in VARIANT_ORDER:
            variants[name].append(
                {
                    "seed_id": seed["seed_id"],
                    "scene": seed["scene"],
                    "query_row": query_row,
                    **ranked[name],
                }
            )
    baseline_metrics = _list_metrics(
        baseline_records, ranker, style, deezer_truth
    )
    variant_metrics: Dict[str, Any] = {}
    selected = None
    for name in VARIANT_ORDER:
        metrics = _list_metrics(variants[name], ranker, style, deezer_truth)
        metrics["style_delta_vs_production"] = (
            metrics["mean_style_overlap"] - baseline_metrics["mean_style_overlap"]
        )
        metrics["deezer_affinity_delta_vs_production"] = (
            metrics["deezer_related_artist_hit_rate"]
            - baseline_metrics["deezer_related_artist_hit_rate"]
        )
        metrics["gate_fired_count"] = sum(
            bool(item.get("gate_fired")) for item in variants[name]
        )
        metrics["exact_production_fallback_count"] = sum(
            item["rows"] == baseline_records[position]["rows"]
            for position, item in enumerate(variants[name])
        )
        metrics["passes_proxy_safety"] = _proxy_passes(
            metrics, baseline_metrics
        )
        variant_metrics[name] = metrics
        if selected is None and metrics["passes_proxy_safety"]:
            selected = name
    if selected is None:
        raise ClapCatalogError("all three CLAP variants failed proxy collapse gates")

    # Category-A target recall is diagnostic only and never selects the variant.
    target_rows: Dict[str, List[int]] = {}
    try:
        from .catalog_list_gold_v9 import load_seed_specs

        seeds = load_seed_specs(
            "benchmarks/soundalike_pairs.v6.json",
            ".goals/human-quality-recommendations/artifacts/"
            "catalog-gated-direct-seeds-v8.json",
        )
        resolver = PairResolver(ranker.titles, ranker.artists)
        for seed in seeds:
            pair = seed.get("category_a_pair")
            if pair:
                resolved = resolver.target_rows(pair["target"])
                target_rows[str(seed["id"])] = list(map(int, resolved))
    except Exception:
        target_rows = {}
    candidate_recall: Dict[str, Any] = {}
    for name in VARIANT_ORDER:
        found50 = found200 = total = 0
        for record in variants[name]:
            targets = set(target_rows.get(str(record["seed_id"]), ()))
            if not targets:
                continue
            total += 1
            candidates = list(map(int, record.get("candidate_rows", ())))
            found50 += int(bool(targets & set(candidates[:50])))
            found200 += int(bool(targets & set(candidates[:200])))
        candidate_recall[name] = {
            "known_category_a_targets": total,
            "recall_at_50": found50 / total if total else None,
            "recall_at_200": found200 / total if total else None,
            "selection_use": False,
        }

    compact_sha = _sha256_path(compact_path)
    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "clap_catalog_proxy_safety_and_variant_selection",
        "created_at": _now(),
        "preregistration_content_sha256": PREREGISTRATION_SHA256,
        "commercial_human_ratings_used": 0,
        "proxy_evidence_is_deciding": False,
        "old_gnod_co_primary_used": False,
        "catalog": {
            "rows": EXPECTED_ROWS,
            "track_ids_tobytes_sha256": TRACK_IDS_SHA256,
        },
        "compact_asset_sha256": compact_sha,
        "scene_distribution": dict(sorted(scene_counts.items())),
        "deezer_affinity": {
            "seeds_requested": 60,
            "seeds_with_related_artists": sum(bool(value) for value in deezer_truth.values()),
            "fresh_supporting_only": True,
        },
        "production_baseline": {
            "metrics": baseline_metrics,
            "records": baseline_records,
        },
        "variants": {
            name: {"metrics": variant_metrics[name], "records": variants[name]}
            for name in VARIANT_ORDER
        },
        "candidate_recall_diagnostic": candidate_recall,
        "selection_order": list(VARIANT_ORDER),
        "selected_challenger": selected,
        "selection_rule": (
            "first pre-registered variant in conservative, graph, pure order "
            "passing every proxy safety gate"
        ),
        "latency": {
            "queries": len(latencies),
            "mean_ms": float(np.mean(latencies) * 1000),
            "p50_ms": float(np.quantile(latencies, 0.50) * 1000),
            "p95_ms": float(np.quantile(latencies, 0.95) * 1000),
        },
        "safety": {
            "obvious_collapse_rejected": True,
            "human_ab_required": True,
            "production_changed": False,
            "deployed": False,
            "commercial_final_opened": False,
            "ac3_claimed": False,
        },
    }
    report["content_sha256"] = content_hash(report)
    _write_json(output_path, report)
    return report


def measure_resources(
    index_path: Path,
    compact_path: Path,
    diagnostics_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    """Measure prospective isolated asset load RSS and compact query latency."""
    try:
        import psutil
    except ImportError as error:  # pragma: no cover - ML extra
        raise ClapCatalogError("psutil is required for resource measurement") from error

    process = psutil.Process(os.getpid())
    gc.collect()
    start_rss = int(process.memory_info().rss)
    from webapp.api._reco import WebRecommender

    production_started = time.perf_counter()
    production = WebRecommender(str(index_path), enhance=True)
    production_load_seconds = time.perf_counter() - production_started
    production_rss = int(process.memory_info().rss)
    started = time.perf_counter()
    compact = np.load(compact_path, mmap_mode="r")
    mmap_seconds = time.perf_counter() - started
    mmap_rss = int(process.memory_info().rss)
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    query_rows = [
        int(item["query_row"])
        for item in diagnostics["production_baseline"]["records"]
    ]
    latencies = []
    production_latencies = []
    for row in query_rows:
        production_before = time.perf_counter()
        production.recommend(row, n=5)
        production_latencies.append(time.perf_counter() - production_before)
        before = time.perf_counter()
        query = np.asarray(compact[row], dtype=np.float32)
        query /= max(float(np.linalg.norm(query)), 1e-8)
        scores = np.empty(len(compact), dtype=np.float32)
        for start in range(0, len(scores), 8192):
            stop = min(start + 8192, len(scores))
            scores[start:stop] = _normalise_rows(
                np.asarray(compact[start:stop], dtype=np.float32)
            ) @ query
        _top_indices(scores, 500)
        latencies.append(time.perf_counter() - before)
    touched_rss = int(process.memory_info().rss)
    production_bytes = int(index_path.stat().st_size)
    compact_bytes = int(compact_path.stat().st_size)
    report: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact_kind": "clap_catalog_prospective_resources",
        "created_at": _now(),
        "assets": {
            "production_index": {
                "path": str(index_path),
                "bytes": production_bytes,
                "sha256": _sha256_path(index_path),
                "unchanged": True,
            },
            "isolated_compact_clap": {
                "path": str(compact_path),
                "bytes": compact_bytes,
                "sha256": _sha256_path(compact_path),
                "release_uploaded": False,
                "wired": False,
            },
            "prospective_combined_file_bytes": production_bytes + compact_bytes,
        },
        "process": {
            "start_rss_bytes": start_rss,
            "production_loaded_rss_bytes": production_rss,
            "mmap_rss_bytes": mmap_rss,
            "touched_rss_bytes": touched_rss,
            "incremental_touched_rss_bytes": max(0, touched_rss - start_rss),
            "incremental_compact_touched_rss_bytes": max(
                0, touched_rss - production_rss
            ),
            "production_load_seconds": production_load_seconds,
            "mmap_load_seconds": mmap_seconds,
        },
        "production_query_latency": {
            "queries": len(production_latencies),
            "mean_ms": float(np.mean(production_latencies) * 1000),
            "p50_ms": float(np.quantile(production_latencies, 0.50) * 1000),
            "p95_ms": float(np.quantile(production_latencies, 0.95) * 1000),
        },
        "compact_query_latency": {
            "queries": len(latencies),
            "mean_ms": float(np.mean(latencies) * 1000),
            "p50_ms": float(np.quantile(latencies, 0.50) * 1000),
            "p95_ms": float(np.quantile(latencies, 0.95) * 1000),
        },
        "existing_production_measurement": {
            "artifact": (
                ".goals/human-quality-recommendations/artifacts/"
                "catalog-gated-resources-v8.json"
            ),
            "peak_rss_bytes": 1_494_294_528,
            "post_gc_resident_rss_bytes": 1_327_017_984,
            "load_seconds": 7.958867499997723,
        },
        "vercel": {
            "evidence_artifact": (
                ".goals/human-quality-recommendations/artifacts/"
                "catalog-vercel-tier-evidence-v9.json"
            ),
            "project_specific_tier": "unknown",
            "actual_memory_limit_bytes": None,
            "official_documented_hobby_bytes": 2_147_483_648,
            "fit_claimed": False,
        },
        "production_changed": False,
        "deployed": False,
    }
    report["content_sha256"] = content_hash(report)
    _write_json(output_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[3]
    protocol = (
        root
        / ".goals"
        / "human-quality-recommendations"
        / "protocol-v13-clap-development"
        / "preregistration-v13-r3.json"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    embed = sub.add_parser("embed")
    embed.add_argument("--index", type=Path, default=root / "ml_data/deepvibe_index_v5.npz")
    embed.add_argument("--cache", type=Path, default=root / "ml_data/clap_v13")
    embed.add_argument("--preregistration", type=Path, default=protocol)
    embed.add_argument("--chunk-tracks", type=int, default=32)
    embed.add_argument("--max-rows", type=int)
    embed.add_argument("--reset", action="store_true")
    compact = sub.add_parser("compress")
    compact.add_argument("--index", type=Path, default=root / "ml_data/deepvibe_index_v5.npz")
    compact.add_argument("--cache", type=Path, default=root / "ml_data/clap_v13")
    compact.add_argument("--preregistration", type=Path, default=protocol)
    diagnose = sub.add_parser("diagnose")
    diagnose.add_argument("--index", type=Path, default=root / "ml_data/deepvibe_index_v5.npz")
    diagnose.add_argument("--compact", type=Path, required=True)
    diagnose.add_argument("--status", type=Path, default=root / "ml_data/clap_v13/status.sqlite3")
    diagnose.add_argument(
        "--graph",
        type=Path,
        default=root / "ml_data/iteration8/catalog-artist-graph-dual-v8.npz",
    )
    diagnose.add_argument(
        "--style", type=Path, default=root / "ml_data/iteration7/catalog-style-v8.npz"
    )
    diagnose.add_argument(
        "--seeds",
        type=Path,
        default=(
            root
            / ".goals/human-quality-recommendations/"
            "protocol-v11-audio-access-erratum/served-lists-v11.json"
        ),
    )
    diagnose.add_argument("--output", type=Path, required=True)
    resources = sub.add_parser("resources")
    resources.add_argument("--index", type=Path, default=root / "ml_data/deepvibe_index_v5.npz")
    resources.add_argument("--compact", type=Path, required=True)
    resources.add_argument("--diagnostics", type=Path, required=True)
    resources.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "embed":
        report = build_full_embeddings(
            args.index,
            args.cache,
            args.preregistration,
            chunk_tracks=args.chunk_tracks,
            reset=args.reset,
            max_rows=args.max_rows,
        )
        print(json.dumps(report["coverage"], sort_keys=True))
    elif args.command == "compress":
        report = compress_embeddings(args.index, args.cache, args.preregistration)
        print(
            f"compact CLAP{report['selected_dimension']}: "
            f"{report['asset']['sha256']}"
        )
    elif args.command == "diagnose":
        report = run_variant_diagnostics(
            args.index,
            args.compact,
            args.status,
            args.graph,
            args.style,
            args.seeds,
            args.output,
        )
        print(f"selected challenger: {report['selected_challenger']}")
    else:
        report = measure_resources(
            args.index, args.compact, args.diagnostics, args.output
        )
        print(json.dumps(report["process"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
