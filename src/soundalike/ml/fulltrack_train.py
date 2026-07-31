"""Self-supervised trainer for sealed MTG-Jamendo full-track CLAP stores.

This module is intentionally narrow:

* Inputs are an audited :class:`JamendoContext` and a sealed
  :class:`FullTrackStoreReader`.
* Training supervision is only same-track/cross-temporal-view ranking.
* Tags, ratings, external graphs, same-artist positives, and audio decoding are
  not imported or used.
* Exported models are NumPy-only :mod:`soundalike.ml.fulltrack_fusion`
  artifacts; PyTorch is imported lazily only while fitting.

The public helpers are designed to be testable with tiny synthetic
``JamendoContext``/``FullTrackStoreReader`` fixtures while enforcing the same
split isolation and artifact integrity rules used by the production CLI.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as _dt
import hashlib
import io
import json
import math
import os
import secrets
import stat
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np

from .fulltrack_fusion import (
    CANDIDATE_KINDS,
    FEATURE_DIM,
    FEATURE_NAMES,
    FusionConfig,
    FusionError,
    FusionModel,
    build_channel_gated,
    build_monotonic_network,
    build_nonneg_linear,
    extract_pair_features,
    load_fusion_artifact,
    save_fusion_artifact,
)
from .fulltrack_store import FullTrackStoreReader, stable_json_sha256
from .jamendo_fulltrack import JamendoContext, load_jamendo_context


OFFICIAL_FOLDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
OFFICIAL_PARTS: Tuple[str, ...] = ("train", "validation", "test")
TRAIN_SCHEMA_VERSION = 1
CHECKPOINT_SCHEMA_VERSION = 1
DEFAULT_SEEDS: Tuple[int, int, int] = (17, 29, 43)

NO_TAG_SELF_SUPERVISION_NOTICE = (
    "Training is self-supervised from same-track disjoint temporal views only; "
    "fold.track_tags, JamendoTrack.tags, tag Jaccard, ratings, external graphs, "
    "audio decoding, and same-artist positives are not read or used."
)

_MAX_JSON_BYTES = 8 * 1024 * 1024
_MAX_NPZ_BYTES = 256 * 1024 * 1024
_UTC_RE_SUFFIX = "Z"


class FullTrackTrainingError(RuntimeError):
    """Invalid split, unsafe path, stale artifact, or training failure."""


class ViewFormationRejection(FullTrackTrainingError):
    """A valid finite track cannot form the required disjoint temporal views."""

    def __init__(self, message: str, *, reason: str) -> None:
        super().__init__(message)
        self.reason = reason


# ---------------------------------------------------------------------------
# Stable hashing / validation helpers
# ---------------------------------------------------------------------------


def _now_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _seed64(*parts: object) -> int:
    raw = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big", signed=False)


def _rng_for(*parts: object) -> np.random.Generator:
    return np.random.default_rng(_seed64(*parts))


def _validate_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise FullTrackTrainingError(f"{label} must be a 64-character SHA-256 hex string")
    if any(ch not in "0123456789abcdef" for ch in value):
        raise FullTrackTrainingError(f"{label} must be lowercase SHA-256 hex")
    return value


def _validate_json_safe(value: object, label: str = "json") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FullTrackTrainingError(f"{label} contains a non-finite float")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise FullTrackTrainingError(f"{label} contains a non-string object key")
            _validate_json_safe(item, f"{label}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_json_safe(item, f"{label}[{index}]")
        return
    raise FullTrackTrainingError(f"{label} contains unsupported JSON value {type(value).__name__}")


def _strict_int(value: object, label: str, *, minimum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FullTrackTrainingError(f"{label} must be an integer (not bool)")
    out = int(value)
    if minimum is not None and out < minimum:
        raise FullTrackTrainingError(f"{label} must be >= {minimum}")
    return out


def _strict_float(
    value: object,
    label: str,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FullTrackTrainingError(f"{label} must be a finite float")
    out = float(value)
    if not math.isfinite(out):
        raise FullTrackTrainingError(f"{label} must be finite")
    if minimum is not None and out < minimum:
        raise FullTrackTrainingError(f"{label} must be >= {minimum}")
    if maximum is not None and out > maximum:
        raise FullTrackTrainingError(f"{label} must be <= {maximum}")
    return out


def _array_sha256(array: np.ndarray) -> str:
    arr = np.asarray(array)
    if arr.dtype.hasobject:
        raise FullTrackTrainingError("object arrays are not allowed in hashes/artifacts")
    contiguous = np.ascontiguousarray(arr)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _track_plan_sha256(track_plan: Sequence[Tuple[int, str]]) -> str:
    """Reproduce the sealed store's public track-plan hash canonically.

    This intentionally mirrors the JSON bytes used by ``fulltrack_store``
    without importing its private helper: a list of ``[track_id, source_sha256]``
    pairs in audited context order, compact separators, UTF-8 bytes.
    """

    raw = json.dumps(
        [[int(track_id), source_sha256] for track_id, source_sha256 in track_plan],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _finite_array(array: object, label: str, *, ndim: Optional[int] = None) -> np.ndarray:
    arr = np.asarray(array)
    if arr.dtype.hasobject:
        raise FullTrackTrainingError(f"{label} may not have object dtype")
    if ndim is not None and arr.ndim != ndim:
        raise FullTrackTrainingError(f"{label} must be {ndim}-D, got shape {arr.shape}")
    if arr.dtype.kind in "fc" and not np.all(np.isfinite(arr)):
        raise FullTrackTrainingError(f"{label} contains non-finite values")
    if arr.dtype.kind not in "fciu":
        raise FullTrackTrainingError(f"{label} must be numeric")
    return arr


def _freeze_array(array: np.ndarray) -> np.ndarray:
    out = np.asarray(array).copy()
    out.setflags(write=False)
    return out


def _normalise_vector(value: object, label: str = "vector") -> np.ndarray:
    vec = np.asarray(value, dtype=np.float64).reshape(-1)
    if vec.ndim != 1 or len(vec) == 0 or not np.all(np.isfinite(vec)):
        raise FullTrackTrainingError(f"{label} must be a finite non-empty vector")
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        raise FullTrackTrainingError(f"{label} has near-zero norm")
    return vec / norm


def _normalise_rows(value: object, label: str = "matrix") -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise FullTrackTrainingError(f"{label} must be a non-empty 2-D matrix")
    if not np.all(np.isfinite(matrix)):
        raise FullTrackTrainingError(f"{label} contains non-finite values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise FullTrackTrainingError(f"{label} contains near-zero rows")
    return matrix / norms


def _fixed_budget_indices(count: int, budget: int) -> np.ndarray:
    if count <= 0 or budget <= 0:
        raise FullTrackTrainingError("count and budget must be positive")
    return np.rint(np.linspace(0, count - 1, num=budget)).astype(np.int64)


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    max_value = np.max(values, axis=axis, keepdims=True)
    stable = max_value + np.log(np.sum(np.exp(values - max_value), axis=axis, keepdims=True))
    return np.squeeze(stable, axis=axis)


# ---------------------------------------------------------------------------
# Path safety and atomic JSON/NPZ I/O
# ---------------------------------------------------------------------------


def _is_reparse_point(path: Path) -> bool:
    if os.name != "nt":
        return False
    try:
        st = os.lstat(str(path))
    except OSError:
        return False
    attr = getattr(st, "st_file_attributes", 0)
    rp_bit = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attr & rp_bit)


def _reject_link_or_reparse(path: Path, label: str) -> None:
    try:
        if path.is_symlink():
            raise FullTrackTrainingError(f"{label} may not be a symlink/junction: {path}")
    except OSError as exc:
        raise FullTrackTrainingError(f"cannot inspect {label}: {path}") from exc
    if _is_reparse_point(path):
        raise FullTrackTrainingError(f"{label} may not be a reparse point/junction: {path}")


def _check_path_safety(raw: Union[str, Path], label: str) -> Path:
    path = Path(raw)
    # Reject any existing symlink/junction in the path chain before resolving.
    candidates = [path] + list(path.parents)
    for candidate in candidates:
        if candidate.exists() or candidate.is_symlink():
            _reject_link_or_reparse(candidate, f"{label} component")
    try:
        return path.resolve(strict=False)
    except OSError as exc:
        raise FullTrackTrainingError(f"cannot resolve {label}: {path}") from exc


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_write_bytes(path: Union[str, Path], data: bytes, label: str) -> None:
    target = _check_path_safety(path, label)
    parent = _check_path_safety(target.parent, f"{label} parent")
    parent.mkdir(parents=True, exist_ok=True)
    _reject_link_or_reparse(parent, f"{label} parent")
    if target.exists():
        _reject_link_or_reparse(target, label)
    tmp = target.with_name(f".{target.name}.{secrets.token_hex(12)}.tmp")
    try:
        with tmp.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        _reject_link_or_reparse(tmp, f"{label} temp")
        os.replace(str(tmp), str(target))
        _reject_link_or_reparse(target, label)
        _fsync_directory(parent)
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def atomic_write_json(path: Union[str, Path], value: Mapping[str, object]) -> str:
    """Atomically write canonical JSON and return its SHA-256."""

    _validate_json_safe(value, "json artifact")
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    _atomic_write_bytes(path, raw, "json artifact")
    return _sha256_bytes(raw)


def atomic_write_npz(path: Union[str, Path], arrays: Mapping[str, np.ndarray]) -> str:
    """Atomically write a numeric NPZ with pickle disabled by construction."""

    if not arrays:
        raise FullTrackTrainingError("NPZ artifact must contain at least one array")
    clean: Dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        if not isinstance(name, str) or not name or "/" in name or "\\" in name:
            raise FullTrackTrainingError(f"invalid NPZ array name: {name!r}")
        arr = np.asarray(value)
        if arr.dtype.hasobject:
            raise FullTrackTrainingError(f"NPZ array {name!r} may not have object dtype")
        if arr.dtype.kind in "fc" and not np.all(np.isfinite(arr)):
            raise FullTrackTrainingError(f"NPZ array {name!r} contains non-finite values")
        clean[name] = np.ascontiguousarray(arr)
    buffer = io.BytesIO()
    np.savez(buffer, **clean)
    data = buffer.getvalue()
    if len(data) > _MAX_NPZ_BYTES:
        raise FullTrackTrainingError("NPZ artifact exceeds size limit")
    digest = _sha256_bytes(data)
    _atomic_write_bytes(path, data, "npz artifact")
    return digest


def _safe_read_bytes(path: Union[str, Path], label: str, max_bytes: int) -> bytes:
    target = _check_path_safety(path, label)
    _reject_link_or_reparse(target, label)
    if not target.is_file():
        raise FullTrackTrainingError(f"{label} is missing: {target}")
    size = target.stat().st_size
    if size <= 0 or size > max_bytes:
        raise FullTrackTrainingError(f"{label} has invalid size: {size}")
    try:
        return target.read_bytes()
    except OSError as exc:
        raise FullTrackTrainingError(f"cannot read {label}: {target}") from exc


def _read_json(path: Union[str, Path], label: str) -> Dict[str, object]:
    raw = _safe_read_bytes(path, label, _MAX_JSON_BYTES)
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise FullTrackTrainingError(f"{label} is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise FullTrackTrainingError(f"{label} must be a JSON object")
    return parsed


def _read_npz_bytes(raw: bytes, label: str, expected_keys: Sequence[str]) -> Dict[str, np.ndarray]:
    """Parse a strict numeric NPZ from an already-read byte snapshot."""

    if not isinstance(raw, (bytes, bytearray, memoryview)):
        raise FullTrackTrainingError(f"{label} byte snapshot is invalid")
    snapshot = bytes(raw)
    if len(snapshot) <= 0 or len(snapshot) > _MAX_NPZ_BYTES:
        raise FullTrackTrainingError(f"{label} byte snapshot has invalid size: {len(snapshot)}")
    expected = frozenset(expected_keys)
    try:
        with np.load(io.BytesIO(snapshot), allow_pickle=False) as archive:
            actual = frozenset(archive.files)
            if actual != expected:
                raise FullTrackTrainingError(
                    f"{label} arrays differ: expected {sorted(expected)}, found {sorted(actual)}"
                )
            arrays = {name: np.asarray(archive[name]).copy() for name in archive.files}
    except FullTrackTrainingError:
        raise
    except Exception as exc:
        raise FullTrackTrainingError(f"cannot parse {label}: {exc}") from exc
    for name, arr in arrays.items():
        if arr.dtype.hasobject:
            raise FullTrackTrainingError(f"{label} array {name!r} has object dtype")
        if arr.dtype.kind in "fc" and not np.all(np.isfinite(arr)):
            raise FullTrackTrainingError(f"{label} array {name!r} contains non-finite values")
    return arrays


def _read_npz(path: Union[str, Path], label: str, expected_keys: Sequence[str]) -> Dict[str, np.ndarray]:
    raw = _safe_read_bytes(path, label, _MAX_NPZ_BYTES)
    return _read_npz_bytes(raw, label, expected_keys)


# ---------------------------------------------------------------------------
# Configuration and official split isolation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainingConfig:
    """Fitting/mining configuration.

    ``max_train_tracks`` and ``max_validation_tracks`` are only legal when
    ``non_production=True``.  Production ``train-all`` always uses complete
    official train/validation partitions.
    """

    max_epochs: int = 64
    patience: int = 8
    min_delta: float = 1e-5
    learning_rate: float = 1e-2
    weight_decay: float = 1e-3
    margin: float = 0.05
    temperature: float = 0.2
    gradient_clip_norm: float = 5.0
    hard_negatives: int = 2
    random_negatives: int = 2
    maxsim_budget: int = 8
    top_k: int = 4
    coverage_threshold: float = 0.5
    monotonic_hidden_dims: Tuple[int, ...] = (8,)
    min_train_tracks: int = 2
    min_validation_tracks: int = 2
    max_train_tracks: Optional[int] = None
    max_validation_tracks: Optional[int] = None
    device: str = "auto"
    non_production: bool = False

    def validate(self, *, production_train_all: bool = False) -> None:
        _strict_int(self.max_epochs, "max_epochs", minimum=1)
        _strict_int(self.patience, "patience", minimum=0)
        _strict_float(self.min_delta, "min_delta", minimum=0.0)
        _strict_float(self.learning_rate, "learning_rate", minimum=1e-8)
        _strict_float(self.weight_decay, "weight_decay", minimum=0.0)
        _strict_float(self.margin, "margin", minimum=0.0, maximum=1.0)
        _strict_float(self.temperature, "temperature", minimum=1e-6)
        _strict_float(self.gradient_clip_norm, "gradient_clip_norm", minimum=1e-8)
        _strict_int(self.hard_negatives, "hard_negatives", minimum=0)
        _strict_int(self.random_negatives, "random_negatives", minimum=0)
        if self.hard_negatives + self.random_negatives <= 0:
            raise FullTrackTrainingError("at least one hard or random negative is required")
        _strict_int(self.maxsim_budget, "maxsim_budget", minimum=1)
        _strict_int(self.top_k, "top_k", minimum=1)
        _strict_float(self.coverage_threshold, "coverage_threshold", minimum=0.0, maximum=1.0)
        _strict_int(self.min_train_tracks, "min_train_tracks", minimum=2)
        _strict_int(self.min_validation_tracks, "min_validation_tracks", minimum=2)
        if self.device not in ("auto", "cpu", "cuda"):
            raise FullTrackTrainingError("device must be 'auto', 'cpu', or 'cuda'")
        if not isinstance(self.non_production, bool):
            raise FullTrackTrainingError("non_production must be a bool")
        if not self.non_production and int(self.max_epochs) != int(type(self).max_epochs):
            raise FullTrackTrainingError(
                "max_epochs overrides are allowed only in explicit non-production mode"
            )
        if not self.monotonic_hidden_dims:
            raise FullTrackTrainingError("monotonic_hidden_dims must be non-empty")
        for dim in self.monotonic_hidden_dims:
            _strict_int(dim, "monotonic_hidden_dims entry", minimum=1)
            if dim > 4096:
                raise FullTrackTrainingError("monotonic hidden dims must be <= 4096")
        for label, value in (
            ("max_train_tracks", self.max_train_tracks),
            ("max_validation_tracks", self.max_validation_tracks),
        ):
            if value is not None:
                _strict_int(value, label, minimum=2)
        if not self.non_production and (
            self.max_train_tracks is not None or self.max_validation_tracks is not None
        ):
            raise FullTrackTrainingError("track limits are allowed only in explicit non-production mode")

    def as_dict(self) -> Dict[str, object]:
        return {
            "max_epochs": int(self.max_epochs),
            "patience": int(self.patience),
            "min_delta": float(self.min_delta),
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "margin": float(self.margin),
            "temperature": float(self.temperature),
            "gradient_clip_norm": float(self.gradient_clip_norm),
            "hard_negatives": int(self.hard_negatives),
            "random_negatives": int(self.random_negatives),
            "maxsim_budget": int(self.maxsim_budget),
            "top_k": int(self.top_k),
            "coverage_threshold": float(self.coverage_threshold),
            "monotonic_hidden_dims": list(self.monotonic_hidden_dims),
            "min_train_tracks": int(self.min_train_tracks),
            "min_validation_tracks": int(self.min_validation_tracks),
            "max_train_tracks": self.max_train_tracks,
            "max_validation_tracks": self.max_validation_tracks,
            "device": self.device,
            "non_production": bool(self.non_production),
        }

    @property
    def sha256(self) -> str:
        return stable_json_sha256(self.as_dict())


@dataclass(frozen=True)
class FoldSplit:
    fold_index: int
    train_track_ids: Tuple[int, ...]
    validation_track_ids: Tuple[int, ...]
    train_artist_ids: Tuple[int, ...]
    validation_artist_ids: Tuple[int, ...]
    train_track_count: int
    validation_track_count: int
    test_track_count: int
    train_artist_count: int
    validation_artist_count: int
    test_artist_count: int
    fold_hash: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "fold_index": self.fold_index,
            "train_track_ids": list(self.train_track_ids),
            "validation_track_ids": list(self.validation_track_ids),
            "train_artist_ids": list(self.train_artist_ids),
            "validation_artist_ids": list(self.validation_artist_ids),
            "train_track_count": self.train_track_count,
            "validation_track_count": self.validation_track_count,
            "test_track_count": self.test_track_count,
            "train_artist_count": self.train_artist_count,
            "validation_artist_count": self.validation_artist_count,
            "test_artist_count": self.test_artist_count,
            "fold_hash": self.fold_hash,
            "test_track_ids_materialized": False,
        }


def _fold_map(context: JamendoContext) -> Dict[int, object]:
    out: Dict[int, object] = {}
    for fold in context.folds:
        idx = int(fold.index)
        if idx in out:
            raise FullTrackTrainingError(f"duplicate fold index {idx}")
        out[idx] = fold
    return out


def _context_artist_by_track(context: JamendoContext) -> Dict[int, int]:
    artists: Dict[int, int] = {}
    for track in context.tracks:
        tid = int(track.track_id)
        if tid in artists:
            raise FullTrackTrainingError(f"duplicate context track_id {tid}")
        artists[tid] = int(track.artist_id)
    return artists


def _normalised_fold_part_mapping(fold: object, fold_index: int, label: str) -> Dict[int, str]:
    mapping = getattr(fold, label)
    parts: Dict[int, str] = {}
    for raw_id, raw_part in mapping.items():
        item_id = int(raw_id)
        if item_id in parts:
            raise FullTrackTrainingError(
                f"fold {fold_index} {label} id {item_id} is duplicated after integer normalization"
            )
        parts[item_id] = str(raw_part)
    return parts


def _context_track_plan(context: JamendoContext) -> Tuple[Tuple[int, str], ...]:
    plan: List[Tuple[int, str]] = []
    seen: set = set()
    for track in context.tracks:
        track_id = int(track.track_id)
        if track_id in seen:
            raise FullTrackTrainingError(f"duplicate context track_id {track_id}")
        seen.add(track_id)
        source_hash = _validate_sha256(
            getattr(track, "expected_audio_sha256", None),
            f"context track {track_id} expected_audio_sha256",
        )
        plan.append((track_id, source_hash))
    if not plan:
        raise FullTrackTrainingError("JamendoContext has no tracks")
    return tuple(plan)


def store_binding_dict(reader: FullTrackStoreReader) -> Dict[str, object]:
    """Return the exact sealed-store binding used in training reports."""

    if not hasattr(reader, "binding") or not hasattr(reader.binding, "as_dict"):
        raise FullTrackTrainingError("reader must be a sealed FullTrackStoreReader-like object")
    if not hasattr(reader, "manifest"):
        raise FullTrackTrainingError("reader must expose the sealed store manifest")
    binding = dict(reader.binding.as_dict())
    binding["sealed_manifest_sha256"] = stable_json_sha256(reader.manifest)
    return binding


def store_binding_sha256(reader: FullTrackStoreReader) -> str:
    return stable_json_sha256(store_binding_dict(reader))


def validate_store_context_binding(context: JamendoContext, reader: FullTrackStoreReader) -> Dict[str, object]:
    """Require exact context/source binding to the sealed store."""

    binding = store_binding_dict(reader)
    source = _validate_sha256(context.source_fingerprint, "context source_fingerprint")
    if binding.get("source_fingerprint") != source:
        raise FullTrackTrainingError("current JamendoContext source fingerprint and sealed store differ")
    for field_name in ("config_sha256", "model_sha256", "track_plan_sha256"):
        _validate_sha256(binding.get(field_name), f"store binding {field_name}")
    if not isinstance(binding.get("model_id"), str) or not binding["model_id"]:
        raise FullTrackTrainingError("store binding model_id must be nonempty")
    track_count = _strict_int(binding.get("track_count"), "store binding track_count", minimum=1)
    plan = _context_track_plan(context)
    context_track_ids = tuple(track_id for track_id, _ in plan)
    reader_track_ids = tuple(int(track_id) for track_id in getattr(reader, "track_ids", ()))
    if reader_track_ids != context_track_ids:
        raise FullTrackTrainingError(
            "sealed store track IDs must exactly match JamendoContext track IDs/order"
        )
    if track_count != len(context_track_ids):
        raise FullTrackTrainingError("sealed store track count does not match JamendoContext")
    expected_plan_hash = _track_plan_sha256(plan)
    if binding.get("track_plan_sha256") != expected_plan_hash:
        raise FullTrackTrainingError(
            "sealed store track plan checksum does not match JamendoContext source hashes"
        )
    if track_count <= 0:
        raise FullTrackTrainingError("sealed store has no tracks")
    return binding


def validate_official_artist_splits(
    context: JamendoContext,
    reader: Optional[FullTrackStoreReader] = None,
    *,
    required_folds: Sequence[int] = OFFICIAL_FOLDS,
    require_all_official: bool = True,
) -> Tuple[FoldSplit, ...]:
    """Validate official artist-disjoint train/validation/test folds.

    The returned object intentionally contains train/validation track IDs only.
    Test partition IDs are counted for isolation audits but are not returned and
    are never read from the store by this module.
    """

    required = tuple(int(f) for f in required_folds)
    if len(set(required)) != len(required):
        raise FullTrackTrainingError("required fold indices must be unique")
    if require_all_official and required != OFFICIAL_FOLDS:
        raise FullTrackTrainingError("production training must validate folds 0..4")
    folds = _fold_map(context)
    missing = [fold for fold in required if fold not in folds]
    if missing:
        raise FullTrackTrainingError(f"context is missing required folds: {missing}")
    if require_all_official and set(folds) != set(OFFICIAL_FOLDS):
        raise FullTrackTrainingError("production context must contain exactly folds 0..4")

    artist_by_track = _context_artist_by_track(context)
    store_track_ids: Optional[frozenset] = None
    if reader is not None:
        validate_store_context_binding(context, reader)
        store_track_ids = frozenset(int(tid) for tid in reader.track_ids)

    splits: List[FoldSplit] = []
    for fold_index in required:
        fold = folds[fold_index]
        counts = {"train": 0, "validation": 0, "test": 0}
        artist_sets: Dict[str, set] = {"train": set(), "validation": set(), "test": set()}
        train_ids: List[int] = []
        validation_ids: List[int] = []

        # Validate every official track assignment without reading tags.
        track_parts = _normalised_fold_part_mapping(fold, fold_index, "track_parts")
        expected_track_ids = frozenset(
            int(track.track_id)
            for track in context.tracks
            if len(track.fold_parts) > fold_index and track.fold_parts[fold_index] is not None
        )
        missing_tracks = sorted(expected_track_ids - frozenset(track_parts))
        extra_tracks = sorted(frozenset(track_parts) - expected_track_ids)
        if missing_tracks or extra_tracks:
            raise FullTrackTrainingError(
                f"fold {fold_index} track_parts coverage must exactly match official split track IDs "
                f"(missing={missing_tracks[:10]}, extra={extra_tracks[:10]})"
            )
        for track_id, part in track_parts.items():
            if part not in OFFICIAL_PARTS:
                raise FullTrackTrainingError(
                    f"fold {fold_index} track {track_id} has invalid partition {part!r}"
                )
            artist_sets[part].add(artist_by_track[track_id])
            counts[part] += 1

        # Validate official artist partition mapping and artist-disjointness.
        artist_parts = _normalised_fold_part_mapping(fold, fold_index, "artist_parts")
        expected_artist_ids = frozenset(artist_by_track[track_id] for track_id in expected_track_ids)
        missing_artists = sorted(expected_artist_ids - frozenset(artist_parts))
        extra_artists = sorted(frozenset(artist_parts) - expected_artist_ids)
        if missing_artists or extra_artists:
            raise FullTrackTrainingError(
                f"fold {fold_index} artist_parts coverage must exactly match official split artists "
                f"(missing={missing_artists[:10]}, extra={extra_artists[:10]})"
            )
        for artist_id, part in artist_parts.items():
            if part not in OFFICIAL_PARTS:
                raise FullTrackTrainingError(
                    f"fold {fold_index} artist {artist_id} has invalid partition {part!r}"
                )
        for part, artists in artist_sets.items():
            for artist_id in artists:
                declared = artist_parts.get(artist_id)
                if declared != part:
                    raise FullTrackTrainingError(
                        f"fold {fold_index} artist {artist_id} track partition {part} "
                        f"differs from artist partition {declared!r}"
                    )
        if artist_sets["train"] & artist_sets["validation"]:
            raise FullTrackTrainingError(f"fold {fold_index} train/validation artists overlap")
        if artist_sets["train"] & artist_sets["test"]:
            raise FullTrackTrainingError(f"fold {fold_index} train/test artists overlap")
        if artist_sets["validation"] & artist_sets["test"]:
            raise FullTrackTrainingError(f"fold {fold_index} validation/test artists overlap")
        if any(counts[part] <= 0 for part in OFFICIAL_PARTS):
            raise FullTrackTrainingError(f"fold {fold_index} must contain train/validation/test tracks")

        # Preserve context order for deterministic fitting.  Do not materialize
        # or return test IDs.
        for track in context.tracks:
            part = track_parts.get(int(track.track_id))
            if part == "train":
                train_ids.append(int(track.track_id))
            elif part == "validation":
                validation_ids.append(int(track.track_id))
        if store_track_ids is not None:
            missing_train = [tid for tid in train_ids if tid not in store_track_ids]
            missing_validation = [tid for tid in validation_ids if tid not in store_track_ids]
            if missing_train or missing_validation:
                raise FullTrackTrainingError(
                    f"fold {fold_index} train/validation tracks are missing from sealed store"
                )

        payload = {
            "fold_index": fold_index,
            "train_track_ids": train_ids,
            "validation_track_ids": validation_ids,
            "train_artist_ids": sorted(artist_sets["train"]),
            "validation_artist_ids": sorted(artist_sets["validation"]),
            "train_track_count": counts["train"],
            "validation_track_count": counts["validation"],
            "test_track_count": counts["test"],
            "train_artist_count": len(artist_sets["train"]),
            "validation_artist_count": len(artist_sets["validation"]),
            "test_artist_count": len(artist_sets["test"]),
            "no_tag_supervision": True,
        }
        splits.append(
            FoldSplit(
                fold_index=fold_index,
                train_track_ids=tuple(train_ids),
                validation_track_ids=tuple(validation_ids),
                train_artist_ids=tuple(sorted(artist_sets["train"])),
                validation_artist_ids=tuple(sorted(artist_sets["validation"])),
                train_track_count=counts["train"],
                validation_track_count=counts["validation"],
                test_track_count=counts["test"],
                train_artist_count=len(artist_sets["train"]),
                validation_artist_count=len(artist_sets["validation"]),
                test_artist_count=len(artist_sets["test"]),
                fold_hash=stable_json_sha256(payload),
            )
        )
    return tuple(splits)


def _limited_ids(ids: Sequence[int], limit: Optional[int], label: str, config: TrainingConfig) -> Tuple[int, ...]:
    values = tuple(int(v) for v in ids)
    if limit is None:
        return values
    if not config.non_production:
        raise FullTrackTrainingError(f"{label} limit is allowed only in non-production mode")
    return values[: int(limit)]


def _enforce_production_view_completeness(dataset: ViewDataset, part: str, config: TrainingConfig) -> None:
    if config.non_production:
        return
    if int(dataset.rejected_track_count) > 0:
        raise FullTrackTrainingError(
            f"production training rejected {dataset.rejected_track_count} official {part} tracks; "
            "all official train/validation tracks must form disjoint temporal views"
        )


# ---------------------------------------------------------------------------
# Disjoint temporal views
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TemporalView:
    track_id: int
    artist_id: int
    view_index: int
    source_indices: Tuple[int, ...]
    time_starts: Tuple[int, ...]
    global_embedding: np.ndarray
    window_embeddings: np.ndarray
    window_starts: np.ndarray
    repeated_sections: np.ndarray
    salient_sections: np.ndarray
    repeated_indices: np.ndarray
    salient_indices: np.ndarray
    view_hash: str


@dataclass(frozen=True)
class ViewPair:
    track_id: int
    artist_id: int
    view_a: TemporalView
    view_b: TemporalView
    track_window_count: int
    source_overlap_count: int
    time_overlap_count: int
    source_coverage: float
    positive_cosine: float
    pair_hash: str


@dataclass(frozen=True)
class ViewDataset:
    fold_index: int
    part: str
    seed: int
    pairs: Tuple[ViewPair, ...]
    rejected_track_count: int
    rejected_reasons: Mapping[str, int]
    dataset_hash: str
    stats: Mapping[str, object]
    embedding_dim: int

    @property
    def track_ids(self) -> Tuple[int, ...]:
        return tuple(pair.track_id for pair in self.pairs)

    @property
    def artist_ids(self) -> Tuple[int, ...]:
        return tuple(pair.artist_id for pair in self.pairs)


def _select_nonempty_subset(pool: np.ndarray, *seed_parts: object) -> np.ndarray:
    pool = np.asarray(pool, dtype=np.int64)
    if len(pool) == 0:
        raise FullTrackTrainingError("cannot select from an empty pool")
    if len(pool) == 1:
        return pool.copy()
    rng = _rng_for("view-subset", *seed_parts)
    size = 1 + int(_seed64("view-subset-size", *seed_parts) % len(pool))
    selected = np.sort(rng.permutation(pool)[:size]).astype(np.int64)
    return selected


def _validate_stored_track_integrity(track: object) -> Tuple[int, np.ndarray, np.ndarray]:
    """Validate store data before classifying any view-formation rejection."""

    try:
        track_id = int(getattr(track, "track_id"))
    except Exception as exc:
        raise FullTrackTrainingError("stored track is missing a valid track_id") from exc
    try:
        global_embedding = np.asarray(getattr(track, "global_embedding"), dtype=np.float64).reshape(-1)
        windows = np.asarray(getattr(track, "window_embeddings"), dtype=np.float64)
    except Exception as exc:
        raise FullTrackTrainingError(f"track {track_id} has malformed embedding arrays") from exc
    if global_embedding.ndim != 1 or len(global_embedding) == 0:
        raise FullTrackTrainingError(f"track {track_id} has invalid global embedding")
    if not np.all(np.isfinite(global_embedding)):
        raise FullTrackTrainingError(f"track {track_id} global embedding contains non-finite values")
    if float(np.linalg.norm(global_embedding)) <= 1e-12:
        raise FullTrackTrainingError(f"track {track_id} global embedding has near-zero norm")
    if windows.ndim != 2 or windows.shape[0] == 0 or windows.shape[1] == 0:
        raise FullTrackTrainingError(f"track {track_id} has invalid window embeddings")
    if not np.all(np.isfinite(windows)):
        raise FullTrackTrainingError(f"track {track_id} window embeddings contain non-finite values")
    if np.any(np.linalg.norm(windows, axis=1) <= 1e-12):
        raise FullTrackTrainingError(f"track {track_id} window embeddings contain near-zero rows")

    try:
        raw_starts = np.asarray(getattr(track, "window_starts"))
    except Exception as exc:
        raise FullTrackTrainingError(f"track {track_id} window_starts are missing/misaligned") from exc
    if raw_starts.ndim != 1 or raw_starts.dtype.kind not in "iu":
        raise FullTrackTrainingError(f"track {track_id} window_starts are missing/misaligned")
    starts = raw_starts.astype(np.int64, copy=False)
    if len(starts) != len(windows):
        raise FullTrackTrainingError(f"track {track_id} window_starts are missing/misaligned")
    if len(starts) and (np.any(starts < 0) or np.any(np.diff(starts) < 0)):
        raise FullTrackTrainingError(f"track {track_id} window_starts are invalid")

    dim = int(windows.shape[1])
    for section_name, indices_name in (
        ("repeated_sections", "repeated_indices"),
        ("salient_sections", "salient_indices"),
    ):
        try:
            sections = np.asarray(getattr(track, section_name), dtype=np.float64)
            raw_indices = np.asarray(getattr(track, indices_name))
        except Exception as exc:
            raise FullTrackTrainingError(
                f"track {track_id} {section_name}/{indices_name} are malformed"
            ) from exc
        if sections.ndim != 2 or sections.shape[1] != dim:
            raise FullTrackTrainingError(f"track {track_id} {section_name} has invalid shape")
        if not np.all(np.isfinite(sections)):
            raise FullTrackTrainingError(f"track {track_id} {section_name} contains non-finite values")
        if raw_indices.ndim != 1 or raw_indices.dtype.kind not in "iu":
            raise FullTrackTrainingError(f"track {track_id} {indices_name} is misaligned")
        indices = raw_indices.astype(np.int64, copy=False)
        if len(indices) != len(sections):
            raise FullTrackTrainingError(f"track {track_id} {indices_name} is misaligned")
        if (
            np.any(indices < 0)
            or np.any(indices >= len(windows))
            or len(np.unique(indices)) != len(indices)
        ):
            raise FullTrackTrainingError(f"track {track_id} {indices_name} is misaligned")
        if len(indices) and not np.array_equal(
            sections.astype(np.float16),
            windows[indices].astype(np.float16),
        ):
            raise FullTrackTrainingError(f"track {track_id} {section_name} source windows are misaligned")
    return track_id, windows, starts


def _make_temporal_view(
    track: object,
    artist_id: int,
    view_index: int,
    selected: np.ndarray,
) -> TemporalView:
    track_id = int(track.track_id)
    windows = np.asarray(track.window_embeddings, dtype=np.float64)
    starts = np.asarray(track.window_starts, dtype=np.int64)
    dim = int(windows.shape[1])
    selected = np.asarray(selected, dtype=np.int64)
    selected_windows = windows[selected]
    mean = np.mean(selected_windows, axis=0)
    norm = float(np.linalg.norm(mean))
    if norm <= 1e-12:
        mean = selected_windows[0]
        norm = float(np.linalg.norm(mean))
    if norm <= 1e-12:
        raise FullTrackTrainingError(f"track {track_id} view {view_index} has zero global norm")
    global_embedding = (mean / norm).astype(np.float32)
    source_to_local = {int(src): local for local, src in enumerate(selected.tolist())}

    def section(section_name: str, indices_name: str) -> Tuple[np.ndarray, np.ndarray]:
        raw_sections = np.asarray(getattr(track, section_name), dtype=np.float64)
        raw_indices = np.asarray(getattr(track, indices_name), dtype=np.int64)
        if raw_sections.ndim != 2 or raw_sections.shape[1] != dim:
            raise FullTrackTrainingError(f"track {track_id} {section_name} has invalid shape")
        if raw_indices.ndim != 1 or len(raw_indices) != len(raw_sections):
            raise FullTrackTrainingError(f"track {track_id} {indices_name} is misaligned")
        keep_positions = [i for i, src in enumerate(raw_indices.tolist()) if int(src) in source_to_local]
        if keep_positions:
            kept = raw_sections[keep_positions].astype(np.float32)
            remapped = np.asarray(
                [source_to_local[int(raw_indices[i])] for i in keep_positions],
                dtype=np.int64,
            )
        else:
            kept = np.zeros((0, dim), dtype=np.float32)
            remapped = np.zeros((0,), dtype=np.int64)
        return kept, remapped

    repeated_sections, repeated_indices = section("repeated_sections", "repeated_indices")
    salient_sections, salient_indices = section("salient_sections", "salient_indices")
    source_indices = tuple(int(v) for v in selected.tolist())
    time_starts = tuple(int(starts[v]) for v in selected.tolist())
    arrays = {
        "global": global_embedding,
        "windows": selected_windows.astype(np.float32),
        "starts": starts[selected].astype(np.int64),
        "repeated": repeated_sections,
        "repeated_indices": repeated_indices,
        "salient": salient_sections,
        "salient_indices": salient_indices,
    }
    payload = {
        "track_id": track_id,
        "artist_id": int(artist_id),
        "view_index": int(view_index),
        "source_indices": list(source_indices),
        "time_starts": list(time_starts),
        "array_hashes": {name: _array_sha256(value) for name, value in arrays.items()},
    }
    return TemporalView(
        track_id=track_id,
        artist_id=int(artist_id),
        view_index=int(view_index),
        source_indices=source_indices,
        time_starts=time_starts,
        global_embedding=_freeze_array(global_embedding),
        window_embeddings=_freeze_array(selected_windows.astype(np.float32)),
        window_starts=_freeze_array(starts[selected].astype(np.int64)),
        repeated_sections=_freeze_array(repeated_sections),
        salient_sections=_freeze_array(salient_sections),
        repeated_indices=_freeze_array(repeated_indices),
        salient_indices=_freeze_array(salient_indices),
        view_hash=stable_json_sha256(payload),
    )


def make_disjoint_temporal_views(
    track: object,
    *,
    artist_id: int,
    seed: int,
    fold_index: int = 0,
) -> ViewPair:
    """Create two deterministic non-empty, source/time-disjoint views.

    Tracks with fewer than two distinct ``window_starts`` are rejected instead
    of duplicating evidence across views.
    """

    track_id, windows, starts = _validate_stored_track_integrity(track)
    if len(windows) < 2:
        raise ViewFormationRejection(
            f"track {track_id} needs at least two windows for views",
            reason="fewer_than_two_windows",
        )
    unique_starts = np.unique(starts)
    if len(unique_starts) < 2:
        raise ViewFormationRejection(
            f"track {track_id} cannot form two non-overlapping temporal views",
            reason="single_temporal_start",
        )
    # Contiguous split in start-time space.  The modulo form gives predictable
    # seed variation for adjacent seeds while remaining deterministic.
    split_span = len(unique_starts) - 1
    split_at = 1 + ((int(seed) + track_id * 1009 + int(fold_index) * 9173) % split_span)
    left_starts = set(int(v) for v in unique_starts[:split_at].tolist())
    right_starts = set(int(v) for v in unique_starts[split_at:].tolist())
    left_pool = np.asarray([i for i, value in enumerate(starts.tolist()) if int(value) in left_starts], dtype=np.int64)
    right_pool = np.asarray([i for i, value in enumerate(starts.tolist()) if int(value) in right_starts], dtype=np.int64)
    if len(left_pool) == 0 or len(right_pool) == 0:
        raise ViewFormationRejection(
            f"track {track_id} produced an empty temporal view",
            reason="empty_temporal_view",
        )
    view_a_indices = _select_nonempty_subset(left_pool, seed, fold_index, track_id, "a")
    view_b_indices = _select_nonempty_subset(right_pool, seed, fold_index, track_id, "b")
    if _seed64("view-swap", seed, fold_index, track_id) % 2:
        view_a_indices, view_b_indices = view_b_indices, view_a_indices
    source_overlap = set(view_a_indices.tolist()) & set(view_b_indices.tolist())
    time_overlap = set(int(starts[i]) for i in view_a_indices.tolist()) & set(
        int(starts[i]) for i in view_b_indices.tolist()
    )
    if source_overlap or time_overlap:
        raise FullTrackTrainingError(f"track {track_id} produced overlapping views")
    view_a = _make_temporal_view(track, int(artist_id), 0, view_a_indices)
    view_b = _make_temporal_view(track, int(artist_id), 1, view_b_indices)
    cos = float(np.clip(np.dot(_normalise_vector(view_a.global_embedding), _normalise_vector(view_b.global_embedding)), -1.0, 1.0))
    coverage = float(len(set(view_a.source_indices) | set(view_b.source_indices)) / float(len(windows)))
    payload = {
        "track_id": track_id,
        "artist_id": int(artist_id),
        "view_hashes": [view_a.view_hash, view_b.view_hash],
        "source_overlap_count": 0,
        "time_overlap_count": 0,
        "source_coverage": coverage,
        "positive_cosine": cos,
    }
    return ViewPair(
        track_id=track_id,
        artist_id=int(artist_id),
        view_a=view_a,
        view_b=view_b,
        track_window_count=int(len(windows)),
        source_overlap_count=0,
        time_overlap_count=0,
        source_coverage=coverage,
        positive_cosine=cos,
        pair_hash=stable_json_sha256(payload),
    )


def _mean_pairwise_cosine_linear(matrix: np.ndarray) -> float:
    vectors = np.asarray(matrix, dtype=np.float64)
    if vectors.ndim != 2 or len(vectors) == 0 or not np.all(np.isfinite(vectors)):
        raise FullTrackTrainingError("pairwise cosine matrix must be finite and two-dimensional")
    if len(vectors) == 1:
        return 1.0
    vector_sum = np.sum(vectors, axis=0, dtype=np.float64)
    sum_squared_norm = float(np.dot(vector_sum, vector_sum))
    individual_squared_norms = float(np.einsum("ij,ij->", vectors, vectors))
    return float(
        (sum_squared_norm - individual_squared_norms)
        / float(len(vectors) * (len(vectors) - 1))
    )


def _summarise_view_pairs(
    pairs: Sequence[ViewPair],
    *,
    rejected_count: int,
    rejected_reasons: Mapping[str, int],
    pairwise_cosine_mode: str = "legacy-v1",
) -> Dict[str, object]:
    if not pairs:
        raise FullTrackTrainingError("view dataset is empty")
    if pairwise_cosine_mode not in ("legacy-v1", "linear-v2"):
        raise FullTrackTrainingError(
            f"unknown pairwise cosine mode {pairwise_cosine_mode!r}"
        )
    source_overlap = int(sum(pair.source_overlap_count for pair in pairs))
    time_overlap = int(sum(pair.time_overlap_count for pair in pairs))
    globals_ = []
    for pair in pairs:
        globals_.append(_normalise_vector(pair.view_a.global_embedding))
        globals_.append(_normalise_vector(pair.view_b.global_embedding))
    matrix = np.stack(globals_)
    if len(matrix) > 1:
        if pairwise_cosine_mode == "legacy-v1":
            sim = matrix @ matrix.T
            upper = sim[np.triu_indices(len(matrix), k=1)]
            mean_pairwise_cosine = float(np.mean(upper))
        else:
            mean_pairwise_cosine = _mean_pairwise_cosine_linear(matrix)
    else:
        mean_pairwise_cosine = 1.0
    positive = np.asarray([pair.positive_cosine for pair in pairs], dtype=np.float64)
    coverage = np.asarray([pair.source_coverage for pair in pairs], dtype=np.float64)
    windows_per_view = [
        len(pair.view_a.source_indices) for pair in pairs
    ] + [
        len(pair.view_b.source_indices) for pair in pairs
    ]
    stats: Dict[str, object] = {
        "track_count": int(len(pairs)),
        "view_count": int(2 * len(pairs)),
        "rejected_track_count": int(rejected_count),
        "rejected_reasons": dict(rejected_reasons),
        "source_overlap_count": source_overlap,
        "time_overlap_count": time_overlap,
        "overlap": 0 if source_overlap == 0 and time_overlap == 0 else source_overlap + time_overlap,
        "mean_windows_per_view": float(np.mean(windows_per_view)),
        "min_windows_per_view": int(min(windows_per_view)),
        "positive_cosine_mean": float(np.mean(positive)),
        "positive_cosine_std": float(np.std(positive)),
        "source_coverage_mean": float(np.mean(coverage)),
        "source_coverage_min": float(np.min(coverage)),
        "diversity_pairwise_cosine_mean": mean_pairwise_cosine,
        "diversity_score": float(np.clip(1.0 - ((mean_pairwise_cosine + 1.0) / 2.0), 0.0, 1.0)),
        "no_tag_supervision": True,
    }
    if pairwise_cosine_mode == "linear-v2":
        stats["diversity_pairwise_cosine_algorithm"] = "sum-vector-v2"
    return stats


def build_view_dataset(
    context: JamendoContext,
    reader: FullTrackStoreReader,
    track_ids: Sequence[int],
    *,
    fold_index: int,
    part: str,
    seed: int,
    min_tracks: int = 2,
    pairwise_cosine_mode: str = "legacy-v1",
) -> ViewDataset:
    """Read only the supplied train/validation IDs and build temporal views."""

    if part not in ("train", "validation"):
        raise FullTrackTrainingError("view datasets may only be built for train/validation parts")
    if pairwise_cosine_mode not in ("legacy-v1", "linear-v2"):
        raise FullTrackTrainingError(
            f"unknown pairwise cosine mode {pairwise_cosine_mode!r}"
        )
    fold_index = int(fold_index)
    requested_track_ids = tuple(int(tid) for tid in track_ids)
    folds = _fold_map(context)
    if fold_index not in folds:
        raise FullTrackTrainingError(f"context is missing fold {fold_index}")
    fold_track_parts = _normalised_fold_part_mapping(folds[fold_index], fold_index, "track_parts")
    artist_by_track = _context_artist_by_track(context)
    for track_id in requested_track_ids:
        if track_id not in artist_by_track:
            raise FullTrackTrainingError(f"unknown track {track_id}")
        assigned_part = fold_track_parts.get(track_id)
        if assigned_part != part:
            raise FullTrackTrainingError(
                f"track {track_id} is officially assigned to {assigned_part!r} "
                f"in fold {fold_index}, not requested {part!r}"
            )

    validate_store_context_binding(context, reader)
    pairs: List[ViewPair] = []
    rejected_reasons: Dict[str, int] = {}
    embedding_dim: Optional[int] = None
    for track_id in requested_track_ids:
        try:
            stored = reader.read_track(track_id)
            if embedding_dim is None:
                embedding_dim = int(np.asarray(stored.global_embedding).reshape(-1).shape[0])
            pair = make_disjoint_temporal_views(
                stored,
                artist_id=artist_by_track[track_id],
                seed=int(seed),
                fold_index=int(fold_index),
            )
            pairs.append(pair)
        except ViewFormationRejection as exc:
            reason = str(getattr(exc, "reason", "view_formation_rejected"))
            rejected_reasons[reason] = rejected_reasons.get(reason, 0) + 1
    if len(pairs) < int(min_tracks):
        raise FullTrackTrainingError(
            f"{part} view dataset has {len(pairs)} usable tracks; need at least {min_tracks}"
        )
    if embedding_dim is None:
        raise FullTrackTrainingError(f"{part} view dataset has no readable tracks")
    stats = _summarise_view_pairs(
        pairs,
        rejected_count=len(requested_track_ids) - len(pairs),
        rejected_reasons=rejected_reasons,
        pairwise_cosine_mode=pairwise_cosine_mode,
    )
    if pairwise_cosine_mode == "legacy-v1":
        payload = {
            "fold_index": int(fold_index),
            "part": part,
            "seed": int(seed),
            "pair_hashes": [pair.pair_hash for pair in pairs],
            "stats": stats,
            "embedding_dim": int(embedding_dim),
            "no_tag_supervision": True,
        }
    else:
        payload = {
            "dataset_hash_schema": "structural-objective-v2",
            "fold_index": int(fold_index),
            "part": part,
            "seed": int(seed),
            "pair_hashes": [pair.pair_hash for pair in pairs],
            "rejected_track_count": len(requested_track_ids) - len(pairs),
            "rejected_reasons": dict(rejected_reasons),
            "embedding_dim": int(embedding_dim),
            "no_tag_supervision": True,
        }
    return ViewDataset(
        fold_index=int(fold_index),
        part=part,
        seed=int(seed),
        pairs=tuple(pairs),
        rejected_track_count=len(requested_track_ids) - len(pairs),
        rejected_reasons=dict(rejected_reasons),
        dataset_hash=stable_json_sha256(payload),
        stats=stats,
        embedding_dim=int(embedding_dim),
    )


def _dataset_views(dataset: ViewDataset) -> Tuple[TemporalView, ...]:
    views: List[TemporalView] = []
    for pair in dataset.pairs:
        views.append(pair.view_a)
        views.append(pair.view_b)
    return tuple(views)


# ---------------------------------------------------------------------------
# Negative mining and ranking data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RankingData:
    dataset_hash: str
    ranking_hash: str
    pos_features: np.ndarray
    neg_features: np.ndarray
    query_indices: np.ndarray
    positive_indices: np.ndarray
    negative_indices: np.ndarray
    hard_negative_mask: np.ndarray
    query_track_ids: Tuple[int, ...]
    positive_track_ids: Tuple[int, ...]
    negative_track_ids: Tuple[Tuple[int, ...], ...]
    negative_artist_ids: Tuple[Tuple[int, ...], ...]
    stats: Mapping[str, object]

    @property
    def example_count(self) -> int:
        return int(self.query_indices.shape[0])

    @property
    def negatives_per_example(self) -> int:
        return int(self.negative_indices.shape[1])


@dataclass(frozen=True)
class PreparedTrainingData:
    train_dataset: ViewDataset
    validation_dataset: ViewDataset
    train_ranking: RankingData
    validation_ranking: RankingData


def mine_negatives(
    dataset: ViewDataset,
    *,
    config: Optional[TrainingConfig] = None,
    seed: int = 0,
) -> RankingData:
    """Mine hard/random negatives within one split.

    Every negative is a different track and different artist.  Hard negatives
    are selected by frozen view-global cosine with deterministic tie-breaking by
    track ID and view index; random negatives are drawn from the same safeguarded
    eligible set.
    """

    cfg = config or TrainingConfig()
    cfg.validate(production_train_all=False)
    views = _dataset_views(dataset)
    if len(views) < 4:
        raise FullTrackTrainingError("negative mining requires at least two view pairs")
    view_globals = np.stack([_normalise_vector(v.global_embedding) for v in views])
    hard_n = int(cfg.hard_negatives)
    random_n = int(cfg.random_negatives)
    total_n = hard_n + random_n
    query_indices: List[int] = []
    positive_indices: List[int] = []
    negative_indices: List[List[int]] = []
    hard_masks: List[List[bool]] = []
    pos_features: List[np.ndarray] = []
    neg_features: List[np.ndarray] = []
    query_track_ids: List[int] = []
    positive_track_ids: List[int] = []
    negative_track_ids: List[Tuple[int, ...]] = []
    negative_artist_ids: List[Tuple[int, ...]] = []
    same_artist_violations = 0
    same_track_violations = 0

    for pair_index, pair in enumerate(dataset.pairs):
        for query_side in (0, 1):
            q_idx = 2 * pair_index + query_side
            p_idx = 2 * pair_index + (1 - query_side)
            q_view = views[q_idx]
            p_view = views[p_idx]
            candidate_side = 1 - query_side
            eligible: List[int] = []
            for other_pair_index, other_pair in enumerate(dataset.pairs):
                if other_pair_index == pair_index:
                    continue
                if int(other_pair.artist_id) == int(pair.artist_id):
                    continue
                eligible.append(2 * other_pair_index + candidate_side)
            if not eligible:
                raise FullTrackTrainingError(
                    f"no eligible different-artist negatives for track {pair.track_id}"
                )
            scored = []
            for cand_idx in eligible:
                score = float(np.dot(view_globals[q_idx], view_globals[cand_idx]))
                cand = views[cand_idx]
                scored.append((-score, int(cand.track_id), int(cand.view_index), int(cand_idx)))
            scored.sort()
            hard = [item[3] for item in scored[: min(hard_n, len(scored))]]
            hard_set = set(hard)
            remaining = [idx for idx in eligible if idx not in hard_set]
            remaining.sort(key=lambda idx: (int(views[idx].track_id), int(views[idx].view_index), int(idx)))
            rng = _rng_for("random-negatives", dataset.dataset_hash, int(seed), pair_index, query_side)
            random_choices: List[int] = []
            if random_n > 0:
                if remaining:
                    perm = rng.permutation(np.asarray(remaining, dtype=np.int64))
                    random_choices.extend(int(v) for v in perm[: min(random_n, len(perm))].tolist())
                deficit = random_n - len(random_choices)
                if deficit > 0:
                    # Deterministic replacement is allowed only after all
                    # non-hard eligible negatives have been consumed once.
                    pool = sorted(
                        eligible,
                        key=lambda idx: (int(views[idx].track_id), int(views[idx].view_index), int(idx)),
                    )
                    random_choices.extend(
                        int(pool[int(rng.integers(0, len(pool)))]) for _ in range(deficit)
                    )
            negs = hard + random_choices
            masks = [True] * len(hard) + [False] * len(random_choices)
            if len(negs) != total_n:
                # hard_n may exceed eligible count; fill from the eligible pool
                # while preserving different-track/different-artist safeguards.
                fill_pool = [item[3] for item in scored]
                while len(negs) < total_n:
                    negs.append(fill_pool[len(negs) % len(fill_pool)])
                    masks.append(len(masks) < hard_n)
            for neg_idx in negs:
                neg_view = views[neg_idx]
                if neg_view.track_id == q_view.track_id:
                    same_track_violations += 1
                if neg_view.artist_id == q_view.artist_id:
                    same_artist_violations += 1
            if same_track_violations or same_artist_violations:
                raise FullTrackTrainingError("negative mining safeguard violation")

            query_indices.append(q_idx)
            positive_indices.append(p_idx)
            negative_indices.append([int(v) for v in negs])
            hard_masks.append([bool(v) for v in masks])
            query_track_ids.append(int(q_view.track_id))
            positive_track_ids.append(int(p_view.track_id))
            negative_track_ids.append(tuple(int(views[idx].track_id) for idx in negs))
            negative_artist_ids.append(tuple(int(views[idx].artist_id) for idx in negs))
            pf_pos = extract_pair_features(
                q_view,
                p_view,
                maxsim_budget=cfg.maxsim_budget,
                top_k=cfg.top_k,
                coverage_threshold=cfg.coverage_threshold,
            ).to_vector()
            pos_features.append(pf_pos)
            neg_vecs = []
            for neg_idx in negs:
                neg_vecs.append(
                    extract_pair_features(
                        q_view,
                        views[neg_idx],
                        maxsim_budget=cfg.maxsim_budget,
                        top_k=cfg.top_k,
                        coverage_threshold=cfg.coverage_threshold,
                    ).to_vector()
                )
            neg_features.append(np.stack(neg_vecs))

    pos_arr = np.asarray(pos_features, dtype=np.float32)
    neg_arr = np.asarray(neg_features, dtype=np.float32)
    _finite_array(pos_arr, "positive ranking features", ndim=2)
    _finite_array(neg_arr, "negative ranking features", ndim=3)
    q_arr = np.asarray(query_indices, dtype=np.int64)
    p_arr = np.asarray(positive_indices, dtype=np.int64)
    n_arr = np.asarray(negative_indices, dtype=np.int64)
    hard_arr = np.asarray(hard_masks, dtype=np.bool_)
    stats = {
        "example_count": int(len(query_indices)),
        "negatives_per_example": int(total_n),
        "hard_negatives_per_example": int(hard_n),
        "random_negatives_per_example": int(random_n),
        "hard_random_ratio": float(hard_n / float(max(1, total_n))),
        "same_track_negative_count": int(same_track_violations),
        "same_artist_negative_count": int(same_artist_violations),
        "false_negative_safeguards": {
            "different_track": True,
            "different_artist": True,
            "same_artist_negatives_excluded": True,
            "tags_not_used": True,
        },
        "hard_mining_signal": "frozen_view_global_cosine_only",
        "deterministic_tie_breaking": "score_desc_then_track_id_then_view_index",
    }
    payload = {
        "dataset_hash": dataset.dataset_hash,
        "seed": int(seed),
        "query_indices": q_arr.tolist(),
        "positive_indices": p_arr.tolist(),
        "negative_indices": n_arr.tolist(),
        "hard_negative_mask": hard_arr.astype(np.int8).tolist(),
        "pos_features_sha256": _array_sha256(pos_arr),
        "neg_features_sha256": _array_sha256(neg_arr),
        "stats": stats,
    }
    return RankingData(
        dataset_hash=dataset.dataset_hash,
        ranking_hash=stable_json_sha256(payload),
        pos_features=_freeze_array(pos_arr),
        neg_features=_freeze_array(neg_arr),
        query_indices=_freeze_array(q_arr),
        positive_indices=_freeze_array(p_arr),
        negative_indices=_freeze_array(n_arr),
        hard_negative_mask=_freeze_array(hard_arr),
        query_track_ids=tuple(query_track_ids),
        positive_track_ids=tuple(positive_track_ids),
        negative_track_ids=tuple(negative_track_ids),
        negative_artist_ids=tuple(negative_artist_ids),
        stats=stats,
    )


def prepare_training_data(
    context: JamendoContext,
    reader: FullTrackStoreReader,
    split: FoldSplit,
    *,
    seed: int,
    config: TrainingConfig,
    pairwise_cosine_mode: str = "legacy-v1",
) -> PreparedTrainingData:
    """Build one immutable fold/seed objective shared by all candidates."""

    config.validate(production_train_all=False)
    train_ids = _limited_ids(
        split.train_track_ids, config.max_train_tracks, "train", config
    )
    validation_ids = _limited_ids(
        split.validation_track_ids,
        config.max_validation_tracks,
        "validation",
        config,
    )
    train_dataset = build_view_dataset(
        context,
        reader,
        train_ids,
        fold_index=split.fold_index,
        part="train",
        seed=int(seed),
        min_tracks=config.min_train_tracks,
        pairwise_cosine_mode=pairwise_cosine_mode,
    )
    validation_dataset = build_view_dataset(
        context,
        reader,
        validation_ids,
        fold_index=split.fold_index,
        part="validation",
        seed=int(seed),
        min_tracks=config.min_validation_tracks,
        pairwise_cosine_mode=pairwise_cosine_mode,
    )
    _enforce_production_view_completeness(train_dataset, "train", config)
    _enforce_production_view_completeness(
        validation_dataset, "validation", config
    )
    train_ranking = mine_negatives(
        train_dataset, config=config, seed=int(seed) + 101
    )
    validation_ranking = mine_negatives(
        validation_dataset, config=config, seed=int(seed) + 202
    )
    return PreparedTrainingData(
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        train_ranking=train_ranking,
        validation_ranking=validation_ranking,
    )


def ranking_metrics_from_scores(
    positive_scores: np.ndarray,
    negative_scores: np.ndarray,
    *,
    margin: float,
    temperature: float,
) -> Dict[str, float]:
    pos = np.asarray(positive_scores, dtype=np.float64).reshape(-1)
    neg = np.asarray(negative_scores, dtype=np.float64)
    if neg.ndim != 2 or len(pos) != neg.shape[0]:
        raise FullTrackTrainingError("score arrays are misaligned")
    if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(neg)):
        raise FullTrackTrainingError("scores contain non-finite values")
    diffs = pos[:, None] - neg
    pair_loss = float(np.mean(np.logaddexp(0.0, float(margin) - diffs)))
    logits = np.concatenate([pos[:, None], neg], axis=1) / float(temperature)
    list_loss = float(np.mean(_logsumexp(logits, axis=1) - logits[:, 0]))
    wins = (diffs > 0.0).astype(np.float64) + 0.5 * (diffs == 0.0).astype(np.float64)
    auc = float(np.mean(wins))
    accuracy = float(np.mean(pos > np.max(neg, axis=1)))
    return {
        "loss": pair_loss + list_loss,
        "pairwise_loss": pair_loss,
        "listwise_loss": list_loss,
        "ranking_accuracy": accuracy,
        "pairwise_auc": auc,
    }


# ---------------------------------------------------------------------------
# Lazy torch training
# ---------------------------------------------------------------------------


def _import_torch():
    try:
        import torch  # type: ignore
    except ImportError as exc:
        raise FullTrackTrainingError(
            "PyTorch is required for fitting but is optional for inference; "
            "install the project ML extras to train models"
        ) from exc
    return torch


def configure_deterministic_torch(seed: int, *, device: str = "auto"):
    """Import torch lazily and configure deterministic execution."""

    torch = _import_torch()
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    try:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except Exception:
        pass
    if device == "auto":
        selected = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda":
        if not torch.cuda.is_available():
            raise FullTrackTrainingError("device='cuda' requested but CUDA is unavailable")
        selected = "cuda"
    elif device == "cpu":
        selected = "cpu"
    else:
        raise FullTrackTrainingError("device must be 'auto', 'cpu', or 'cuda'")
    return torch, torch.device(selected)


def assert_finite_gradients(parameters: Iterable[object]) -> None:
    """Raise if any torch parameter has a non-finite gradient."""

    for index, parameter in enumerate(parameters):
        grad = getattr(parameter, "grad", None)
        if grad is None:
            continue
        try:
            finite = bool(grad.detach().isfinite().all().item())
        except AttributeError as exc:
            raise FullTrackTrainingError("assert_finite_gradients expects torch parameters") from exc
        if not finite:
            raise FullTrackTrainingError(f"non-finite gradient detected in parameter {index}")


def _peak_rss_bytes() -> int:
    try:
        import resource  # type: ignore

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # Linux reports KiB; macOS reports bytes.  Use a conservative heuristic.
        return value if value > 10 * 1024 * 1024 else value * 1024
    except Exception:
        if os.name == "nt":
            try:
                import ctypes
                import ctypes.wintypes

                class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                    _fields_ = [
                        ("cb", ctypes.wintypes.DWORD),
                        ("PageFaultCount", ctypes.wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t),
                    ]

                counters = PROCESS_MEMORY_COUNTERS()
                counters.cb = ctypes.sizeof(counters)
                handle = ctypes.windll.kernel32.GetCurrentProcess()
                ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                    handle, ctypes.byref(counters), counters.cb
                )
                if ok:
                    return int(counters.PeakWorkingSetSize)
            except Exception:
                return 0
        return 0


def _cuda_peak_bytes(torch, device) -> int:
    try:
        if getattr(device, "type", None) == "cuda":
            return int(torch.cuda.max_memory_allocated(device))
    except Exception:
        return 0
    return 0


def _channel_summaries_numpy(view: TemporalView, budget: int, embedding_dim: int) -> np.ndarray:
    g = _normalise_vector(view.global_embedding)
    if len(g) != embedding_dim:
        raise FullTrackTrainingError("view global embedding dimension mismatch")
    win = _normalise_rows(view.window_embeddings, "view windows")
    bud = win[_fixed_budget_indices(len(win), int(budget))]
    u_raw = np.mean(bud, axis=0)
    u = _normalise_vector(u_raw) if float(np.linalg.norm(u_raw)) > 1e-12 else g.copy()

    def section_summary(sections: np.ndarray) -> np.ndarray:
        sec = np.asarray(sections, dtype=np.float64)
        if sec.ndim == 2 and len(sec) > 0:
            sec_norm = _normalise_rows(sec, "view sections")
            raw = np.mean(sec_norm, axis=0)
            return _normalise_vector(raw) if float(np.linalg.norm(raw)) > 1e-12 else g.copy()
        return g.copy()

    r = section_summary(view.repeated_sections)
    s = section_summary(view.salient_sections)
    return np.stack([g, u, r, s]).astype(np.float32)


def _all_channel_summaries(dataset: ViewDataset, budget: int) -> np.ndarray:
    return np.stack(
        [_channel_summaries_numpy(view, budget, dataset.embedding_dim) for view in _dataset_views(dataset)]
    ).astype(np.float32)


def _score_exported_features_numpy(kind: str, arrays: Mapping[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    if kind == "nonnegative_linear":
        w = np.asarray(arrays["weights"], dtype=np.float64)
        denom = float(np.sum(w))
        return np.clip((x * w).sum(axis=-1) / denom, 0.0, 1.0)
    if kind == "monotonic_network":
        out = x
        layer = 0
        while f"l{layer}_weight" in arrays:
            w = np.asarray(arrays[f"l{layer}_weight"], dtype=np.float64)
            b = np.asarray(arrays[f"l{layer}_bias"], dtype=np.float64)
            out = out @ w.T + b
            if f"l{layer + 1}_weight" in arrays:
                out = np.maximum(out, 0.0)
            layer += 1
        out = np.clip(out.reshape(-1), -500.0, 500.0)
        return 1.0 / (1.0 + np.exp(-out))
    raise FullTrackTrainingError(f"feature scoring is not available for {kind}")


def _score_exported_channel_numpy(gates: np.ndarray, query: TemporalView, candidate: TemporalView, budget: int, dim: int) -> float:
    gates = np.asarray(gates, dtype=np.float64)

    def embed(view: TemporalView) -> np.ndarray:
        channels = _channel_summaries_numpy(view, budget, dim).astype(np.float64)
        combined = np.sum(channels * gates[:, None], axis=0)
        return _normalise_vector(combined)

    cos = float(np.clip(np.dot(embed(query), embed(candidate)), -1.0, 1.0))
    return float(np.clip((1.0 + cos) / 2.0, 0.0, 1.0))


def _validate_exported_arrays(kind: str, arrays: Mapping[str, np.ndarray], hidden_dims: Tuple[int, ...]) -> Dict[str, np.ndarray]:
    if kind == "nonnegative_linear":
        expected = {"weights"}
    elif kind == "channel_gated_embedding":
        expected = {"gates"}
    elif kind == "monotonic_network":
        expected = set()
        for i in range(len(hidden_dims) + 1):
            expected.add(f"l{i}_weight")
            expected.add(f"l{i}_bias")
    else:
        raise FullTrackTrainingError(f"unknown candidate kind {kind!r}")
    if set(arrays) != expected:
        raise FullTrackTrainingError(f"exported arrays differ: expected {sorted(expected)}, got {sorted(arrays)}")
    clean = {name: np.asarray(value, dtype=np.float64).copy() for name, value in arrays.items()}
    for name, arr in clean.items():
        if not np.all(np.isfinite(arr)):
            raise FullTrackTrainingError(f"exported array {name} contains non-finite values")
    if kind == "nonnegative_linear":
        if clean["weights"].shape != (FEATURE_DIM,) or np.any(clean["weights"] < 0.0) or float(clean["weights"].sum()) <= 0.0:
            raise FullTrackTrainingError("invalid nonnegative_linear weights")
    elif kind == "channel_gated_embedding":
        gates = clean["gates"]
        if gates.shape != (4,) or np.any(gates < 0.0) or abs(float(gates.sum()) - 1.0) > 1e-5:
            raise FullTrackTrainingError("invalid channel gates")
    else:
        dims = (FEATURE_DIM,) + tuple(hidden_dims) + (1,)
        for i in range(len(dims) - 1):
            w = clean[f"l{i}_weight"]
            b = clean[f"l{i}_bias"]
            if w.shape != (dims[i + 1], dims[i]) or b.shape != (dims[i + 1],):
                raise FullTrackTrainingError(f"invalid monotonic layer {i} shapes")
            if np.any(w < 0.0):
                raise FullTrackTrainingError(f"monotonic layer {i} has negative weights")
    return {name: _freeze_array(arr) for name, arr in clean.items()}


@dataclass(frozen=True)
class CandidateTrainingResult:
    kind: str
    seed: int
    fold_index: int
    model: FusionModel
    arrays: Mapping[str, np.ndarray]
    report: Mapping[str, object]
    train_ranking: RankingData
    validation_ranking: RankingData


def train_candidate_from_datasets(
    kind: str,
    train_dataset: ViewDataset,
    validation_dataset: ViewDataset,
    *,
    config: TrainingConfig,
    seed: int,
    store_binding_hash: str,
    source_fingerprint: str,
    job_config_sha256: Optional[str] = None,
    train_ranking: Optional[RankingData] = None,
    validation_ranking: Optional[RankingData] = None,
    dedicated_cuda_stream: bool = False,
) -> CandidateTrainingResult:
    """Fit one fusion candidate from fixed train/validation view datasets."""

    if kind not in CANDIDATE_KINDS:
        raise FullTrackTrainingError(f"unknown candidate kind {kind!r}")
    config.validate(production_train_all=False)
    _validate_sha256(store_binding_hash, "store_binding_hash")
    _validate_sha256(source_fingerprint, "source_fingerprint")
    if train_dataset.embedding_dim != validation_dataset.embedding_dim:
        raise FullTrackTrainingError("train/validation embedding dimensions differ")
    if train_dataset.fold_index != validation_dataset.fold_index:
        raise FullTrackTrainingError("train/validation datasets are from different folds")
    start_time = time.perf_counter()
    torch, device = configure_deterministic_torch(int(seed), device=config.device)
    previous_cuda_stream = None
    candidate_cuda_stream = None
    if getattr(device, "type", None) == "cuda":
        if dedicated_cuda_stream:
            previous_cuda_stream = torch.cuda.current_stream(device)
            candidate_cuda_stream = torch.cuda.Stream(device=device)
            torch.cuda.set_stream(candidate_cuda_stream)
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            pass

    if train_ranking is None:
        train_ranking = mine_negatives(train_dataset, config=config, seed=int(seed) + 101)
    elif train_ranking.dataset_hash != train_dataset.dataset_hash:
        raise FullTrackTrainingError("train ranking does not match train dataset")
    if validation_ranking is None:
        validation_ranking = mine_negatives(
            validation_dataset, config=config, seed=int(seed) + 202
        )
    elif validation_ranking.dataset_hash != validation_dataset.dataset_hash:
        raise FullTrackTrainingError("validation ranking does not match validation dataset")
    hidden_dims = tuple(config.monotonic_hidden_dims) if kind == "monotonic_network" else ()
    if job_config_sha256 is None:
        job_config_sha256 = stable_json_sha256(
            {
                "kind": kind,
                "seed": int(seed),
                "fold_index": int(train_dataset.fold_index),
                "training_config_sha256": config.sha256,
                "train_dataset_hash": train_dataset.dataset_hash,
                "validation_dataset_hash": validation_dataset.dataset_hash,
                "train_ranking_hash": train_ranking.ranking_hash,
                "validation_ranking_hash": validation_ranking.ranking_hash,
                "store_binding_hash": store_binding_hash,
                "source_fingerprint": source_fingerprint,
            }
        )
    else:
        _validate_sha256(job_config_sha256, "job_config_sha256")

    dtype = torch.float32
    pos_train = torch.tensor(train_ranking.pos_features, dtype=dtype, device=device)
    neg_train = torch.tensor(train_ranking.neg_features, dtype=dtype, device=device)
    pos_val = torch.tensor(validation_ranking.pos_features, dtype=dtype, device=device)
    neg_val = torch.tensor(validation_ranking.neg_features, dtype=dtype, device=device)
    for label, tensor in (
        ("pos_train", pos_train),
        ("neg_train", neg_train),
        ("pos_val", pos_val),
        ("neg_val", neg_val),
    ):
        if not bool(torch.isfinite(tensor).all().item()):
            raise FullTrackTrainingError(f"{label} contains non-finite values")

    rng = np.random.default_rng(_seed64("torch-init", kind, seed, train_dataset.dataset_hash))
    params: List[object] = []
    raw_linear = None
    raw_gates = None
    raw_weights: List[object] = []
    biases: List[object] = []

    if kind == "nonnegative_linear":
        init = torch.tensor(0.01 * rng.standard_normal(FEATURE_DIM), dtype=dtype, device=device)
        raw_linear = torch.nn.Parameter(init)
        params = [raw_linear]
    elif kind == "channel_gated_embedding":
        init = torch.tensor(0.01 * rng.standard_normal(4), dtype=dtype, device=device)
        raw_gates = torch.nn.Parameter(init)
        params = [raw_gates]
        train_channels = torch.tensor(
            _all_channel_summaries(train_dataset, config.maxsim_budget), dtype=dtype, device=device
        )
        val_channels = torch.tensor(
            _all_channel_summaries(validation_dataset, config.maxsim_budget), dtype=dtype, device=device
        )
        train_q_idx = torch.tensor(train_ranking.query_indices, dtype=torch.long, device=device)
        train_p_idx = torch.tensor(train_ranking.positive_indices, dtype=torch.long, device=device)
        train_n_idx = torch.tensor(train_ranking.negative_indices, dtype=torch.long, device=device)
        val_q_idx = torch.tensor(validation_ranking.query_indices, dtype=torch.long, device=device)
        val_p_idx = torch.tensor(validation_ranking.positive_indices, dtype=torch.long, device=device)
        val_n_idx = torch.tensor(validation_ranking.negative_indices, dtype=torch.long, device=device)
    else:
        dims = (FEATURE_DIM,) + tuple(hidden_dims) + (1,)
        for i in range(len(dims) - 1):
            scale = 0.02
            raw_w = torch.nn.Parameter(
                torch.tensor(
                    -2.0 + scale * rng.standard_normal((dims[i + 1], dims[i])),
                    dtype=dtype,
                    device=device,
                )
            )
            bias = torch.nn.Parameter(torch.zeros((dims[i + 1],), dtype=dtype, device=device))
            raw_weights.append(raw_w)
            biases.append(bias)
        params = list(raw_weights) + list(biases)

    optimizer = torch.optim.AdamW(params, lr=float(config.learning_rate), weight_decay=float(config.weight_decay))
    softplus = torch.nn.Softplus()

    def score_feature_tensors(pos_features, neg_features):
        if kind == "nonnegative_linear":
            weights = softplus(raw_linear) + 1e-8  # type: ignore[arg-type]
            denom = torch.clamp(weights.sum(), min=1e-12)
            pos_scores = (pos_features * weights).sum(dim=-1) / denom
            neg_scores = (neg_features * weights.view(1, 1, -1)).sum(dim=-1) / denom
            reg = 1e-3 * torch.mean((weights / torch.clamp(weights.mean(), min=1e-12) - 1.0) ** 2)
            return pos_scores.clamp(0.0, 1.0), neg_scores.clamp(0.0, 1.0), reg
        if kind == "monotonic_network":
            def network(features):
                x = features
                original_shape = x.shape[:-1]
                x = x.reshape(-1, FEATURE_DIM)
                for layer_index, (raw_w, bias) in enumerate(zip(raw_weights, biases)):
                    weight = softplus(raw_w) + 1e-8
                    x = x @ weight.t() + bias
                    if layer_index < len(raw_weights) - 1:
                        x = torch.relu(x)
                x = torch.sigmoid(x.reshape(original_shape))
                return x

            pos_scores = network(pos_features)
            neg_scores = network(neg_features)
            reg_terms = [torch.mean((softplus(raw_w) + 1e-8) ** 2) for raw_w in raw_weights]
            reg = 1e-4 * torch.stack(reg_terms).mean()
            return pos_scores, neg_scores, reg
        raise FullTrackTrainingError("feature scoring called for non-feature model")

    def score_channel_tensors(channels, q_idx, p_idx, n_idx):
        gates = torch.softmax(raw_gates, dim=0)  # type: ignore[arg-type]

        def embed(indices):
            selected = channels[indices]
            combined = (selected * gates.view(*([1] * (selected.ndim - 2)), 4, 1)).sum(dim=-2)
            return torch.nn.functional.normalize(combined, p=2, dim=-1, eps=1e-12)

        q = embed(q_idx)
        p = embed(p_idx)
        n = embed(n_idx)
        pos_scores = ((q * p).sum(dim=-1).clamp(-1.0, 1.0) + 1.0) / 2.0
        neg_scores = ((q[:, None, :] * n).sum(dim=-1).clamp(-1.0, 1.0) + 1.0) / 2.0
        reg = 1e-3 * torch.sum(gates * torch.log(torch.clamp(gates * 4.0, min=1e-12)))
        return pos_scores, neg_scores, reg

    def ranking_loss(pos_scores, neg_scores, reg):
        diffs = pos_scores[:, None] - neg_scores
        pair_loss = torch.nn.functional.softplus(float(config.margin) - diffs).mean()
        logits = torch.cat([pos_scores[:, None], neg_scores], dim=1) / float(config.temperature)
        target = torch.zeros((logits.shape[0],), dtype=torch.long, device=logits.device)
        list_loss = torch.nn.functional.cross_entropy(logits, target)
        loss = pair_loss + list_loss + reg
        if not bool(torch.isfinite(loss).item()):
            raise FullTrackTrainingError("non-finite training loss")
        return loss, pair_loss, list_loss

    def current_scores(which: str):
        if kind == "channel_gated_embedding":
            if which == "train":
                return score_channel_tensors(train_channels, train_q_idx, train_p_idx, train_n_idx)
            return score_channel_tensors(val_channels, val_q_idx, val_p_idx, val_n_idx)
        if which == "train":
            return score_feature_tensors(pos_train, neg_train)
        return score_feature_tensors(pos_val, neg_val)

    def export_arrays() -> Dict[str, np.ndarray]:
        if kind == "nonnegative_linear":
            with torch.no_grad():
                weights = (softplus(raw_linear) + 1e-8).detach().cpu().numpy().astype(np.float64)  # type: ignore[arg-type]
            return {"weights": weights}
        if kind == "channel_gated_embedding":
            with torch.no_grad():
                gates = torch.softmax(raw_gates, dim=0).detach().cpu().numpy().astype(np.float64)  # type: ignore[arg-type]
            gates = gates / float(np.sum(gates))
            return {"gates": gates}
        arrays: Dict[str, np.ndarray] = {}
        with torch.no_grad():
            for i, (raw_w, bias) in enumerate(zip(raw_weights, biases)):
                arrays[f"l{i}_weight"] = (softplus(raw_w) + 1e-8).detach().cpu().numpy().astype(np.float64)
                arrays[f"l{i}_bias"] = bias.detach().cpu().numpy().astype(np.float64)
        return arrays

    history: List[Dict[str, float]] = []
    best_arrays: Optional[Dict[str, np.ndarray]] = None
    best_epoch = 0
    best_val = float("inf")
    stale_epochs = 0
    epochs_ran = 0

    for epoch in range(1, int(config.max_epochs) + 1):
        epochs_ran = epoch
        optimizer.zero_grad(set_to_none=True)
        train_pos, train_neg, train_reg = current_scores("train")
        loss, _, _ = ranking_loss(train_pos, train_neg, train_reg)
        loss.backward()
        assert_finite_gradients(params)
        torch.nn.utils.clip_grad_norm_(params, float(config.gradient_clip_norm))
        optimizer.step()

        with torch.no_grad():
            tr_pos, tr_neg, tr_reg = current_scores("train")
            tr_loss, _, _ = ranking_loss(tr_pos, tr_neg, tr_reg)
            va_pos, va_neg, va_reg = current_scores("validation")
            va_loss, _, _ = ranking_loss(va_pos, va_neg, va_reg)
            train_metrics = ranking_metrics_from_scores(
                tr_pos.detach().cpu().numpy(),
                tr_neg.detach().cpu().numpy(),
                margin=float(config.margin),
                temperature=float(config.temperature),
            )
            val_metrics = ranking_metrics_from_scores(
                va_pos.detach().cpu().numpy(),
                va_neg.detach().cpu().numpy(),
                margin=float(config.margin),
                temperature=float(config.temperature),
            )
        row = {
            "epoch": float(epoch),
            "train_loss": float(tr_loss.detach().cpu().item()),
            "validation_loss": float(va_loss.detach().cpu().item()),
            "train_ranking_accuracy": float(train_metrics["ranking_accuracy"]),
            "validation_ranking_accuracy": float(val_metrics["ranking_accuracy"]),
            "train_pairwise_auc": float(train_metrics["pairwise_auc"]),
            "validation_pairwise_auc": float(val_metrics["pairwise_auc"]),
        }
        history.append(row)
        current_val = float(row["validation_loss"])
        if current_val < best_val - float(config.min_delta):
            best_val = current_val
            best_epoch = epoch
            best_arrays = export_arrays()
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs > int(config.patience):
                break

    if best_arrays is None:
        best_arrays = export_arrays()
        best_epoch = epochs_ran
    best_arrays = _validate_exported_arrays(kind, best_arrays, hidden_dims)
    fusion_config = FusionConfig(
        kind=kind,
        embedding_dim=int(train_dataset.embedding_dim),
        maxsim_budget=int(config.maxsim_budget),
        top_k=int(config.top_k),
        coverage_threshold=float(config.coverage_threshold),
        seed=int(seed),
        model_id=f"fulltrack-{kind}-fold{train_dataset.fold_index}-seed{seed}-{job_config_sha256[:12]}",
        store_id=store_binding_hash,
        config_sha256=job_config_sha256,
        fold_index=int(train_dataset.fold_index),
        hidden_dims=hidden_dims,
    )
    if kind == "nonnegative_linear":
        model = build_nonneg_linear(best_arrays["weights"], fusion_config)
    elif kind == "channel_gated_embedding":
        model = build_channel_gated(best_arrays["gates"], fusion_config)
    else:
        weights = [best_arrays[f"l{i}_weight"] for i in range(len(hidden_dims) + 1)]
        biases = [best_arrays[f"l{i}_bias"] for i in range(len(hidden_dims) + 1)]
        model = build_monotonic_network(weights, biases, fusion_config)

    # Runtime parity on a fixed validation positive pair.
    first_q = int(validation_ranking.query_indices[0])
    first_p = int(validation_ranking.positive_indices[0])
    val_views = _dataset_views(validation_dataset)
    runtime_score = float(model.score_candidate(val_views[first_q], val_views[first_p]))
    if kind == "channel_gated_embedding":
        exported_score = _score_exported_channel_numpy(
            best_arrays["gates"],
            val_views[first_q],
            val_views[first_p],
            int(config.maxsim_budget),
            int(train_dataset.embedding_dim),
        )
    else:
        exported_score = float(
            _score_exported_features_numpy(
                kind,
                best_arrays,
                validation_ranking.pos_features[0:1],
            )[0]
        )
    parity_diff = abs(runtime_score - exported_score)
    if not math.isfinite(parity_diff) or parity_diff > 1e-5:
        raise FullTrackTrainingError(f"runtime/export parity failure: {parity_diff}")

    selected_metrics = history[best_epoch - 1] if 0 < best_epoch <= len(history) else history[-1]
    parameter_count = int(sum(int(p.numel()) for p in params))
    model_bytes = int(sum(np.asarray(value).nbytes for value in best_arrays.values()))
    if candidate_cuda_stream is not None:
        candidate_cuda_stream.synchronize()
    cuda_peak_bytes = int(_cuda_peak_bytes(torch, device))
    if previous_cuda_stream is not None:
        torch.cuda.set_stream(previous_cuda_stream)
    report = {
        "candidate_kind": kind,
        "seed": int(seed),
        "fold": int(train_dataset.fold_index),
        "device": str(device),
        "epochs_ran": int(epochs_ran),
        "best_epoch": int(best_epoch),
        "early_stopping_metric": "validation_self_supervised_ranking_loss",
        "train_loss": float(selected_metrics["train_loss"]),
        "validation_loss": float(selected_metrics["validation_loss"]),
        "train_ranking_accuracy": float(selected_metrics["train_ranking_accuracy"]),
        "validation_ranking_accuracy": float(selected_metrics["validation_ranking_accuracy"]),
        "train_pairwise_auc": float(selected_metrics["train_pairwise_auc"]),
        "validation_pairwise_auc": float(selected_metrics["validation_pairwise_auc"]),
        "history": history,
        "parameter_count": parameter_count,
        "model_bytes": model_bytes,
        "runtime_parity_abs_diff": float(parity_diff),
        "wall_time_seconds": float(time.perf_counter() - start_time),
        "cpu_rss_peak_bytes": int(_peak_rss_bytes()),
        "cuda_peak_bytes": cuda_peak_bytes,
        "job_config_sha256": job_config_sha256,
        "training_config_sha256": config.sha256,
        "train_dataset_hash": train_dataset.dataset_hash,
        "validation_dataset_hash": validation_dataset.dataset_hash,
        "train_view_hashes": [pair.pair_hash for pair in train_dataset.pairs],
        "validation_view_hashes": [pair.pair_hash for pair in validation_dataset.pairs],
        "train_view_stats": dict(train_dataset.stats),
        "validation_view_stats": dict(validation_dataset.stats),
        "train_ranking_hash": train_ranking.ranking_hash,
        "validation_ranking_hash": validation_ranking.ranking_hash,
        "no_tag_self_supervision_notice": NO_TAG_SELF_SUPERVISION_NOTICE,
    }
    return CandidateTrainingResult(
        kind=kind,
        seed=int(seed),
        fold_index=int(train_dataset.fold_index),
        model=model,
        arrays=best_arrays,
        report=report,
        train_ranking=train_ranking,
        validation_ranking=validation_ranking,
    )


# ---------------------------------------------------------------------------
# Checkpoints, reports, and reusable jobs
# ---------------------------------------------------------------------------


CHECKPOINT_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "job_id",
        "candidate_kind",
        "fold",
        "seed",
        "hidden_dims",
        "embedding_dim",
        "training_config_sha256",
        "job_config_sha256",
        "source_fingerprint",
        "store_binding_sha256",
        "arrays_npz_sha256",
        "created_at",
        "checkpoint_sha256",
    }
)


@dataclass(frozen=True)
class TrainingCheckpoint:
    metadata: Mapping[str, object]
    arrays: Mapping[str, np.ndarray]


def _checkpoint_array_keys(kind: str, hidden_dims: Tuple[int, ...]) -> Tuple[str, ...]:
    if kind == "nonnegative_linear":
        return ("weights",)
    if kind == "channel_gated_embedding":
        return ("gates",)
    if kind == "monotonic_network":
        keys: List[str] = []
        for i in range(len(hidden_dims) + 1):
            keys.append(f"l{i}_weight")
            keys.append(f"l{i}_bias")
        return tuple(keys)
    raise FullTrackTrainingError(f"unknown candidate kind {kind!r}")


def save_training_checkpoint(
    directory: Union[str, Path],
    *,
    job_id: str,
    kind: str,
    fold: int,
    seed: int,
    hidden_dims: Sequence[int],
    embedding_dim: int,
    training_config_sha256: str,
    job_config_sha256: str,
    source_fingerprint: str,
    store_binding_sha256: str,
    arrays: Mapping[str, np.ndarray],
) -> TrainingCheckpoint:
    """Save a JSON+NPZ checkpoint with cross-checksums (never pickle)."""

    hidden = tuple(int(v) for v in hidden_dims)
    clean_arrays = _validate_exported_arrays(kind, arrays, hidden)
    root = _check_path_safety(directory, "checkpoint directory")
    root.mkdir(parents=True, exist_ok=True)
    _reject_link_or_reparse(root, "checkpoint directory")
    npz_sha = atomic_write_npz(root / "checkpoint.npz", clean_arrays)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "artifact_kind": "fulltrack_train_checkpoint",
        "job_id": str(job_id),
        "candidate_kind": kind,
        "fold": int(fold),
        "seed": int(seed),
        "hidden_dims": list(hidden),
        "embedding_dim": int(embedding_dim),
        "training_config_sha256": _validate_sha256(training_config_sha256, "training_config_sha256"),
        "job_config_sha256": _validate_sha256(job_config_sha256, "job_config_sha256"),
        "source_fingerprint": _validate_sha256(source_fingerprint, "source_fingerprint"),
        "store_binding_sha256": _validate_sha256(store_binding_sha256, "store_binding_sha256"),
        "arrays_npz_sha256": npz_sha,
        "created_at": _now_utc(),
    }
    payload["checkpoint_sha256"] = stable_json_sha256(payload)
    _validate_json_safe(payload, "checkpoint")
    atomic_write_json(root / "checkpoint.json", payload)
    return TrainingCheckpoint(metadata=payload, arrays=clean_arrays)


def load_training_checkpoint(
    directory: Union[str, Path],
    *,
    expected_kind: Optional[str] = None,
    expected_job_config_sha256: Optional[str] = None,
) -> TrainingCheckpoint:
    root = _check_path_safety(directory, "checkpoint directory")
    meta = _read_json(root / "checkpoint.json", "checkpoint.json")
    if frozenset(meta.keys()) != CHECKPOINT_REQUIRED_FIELDS:
        raise FullTrackTrainingError("checkpoint.json schema fields differ")
    if meta.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise FullTrackTrainingError("checkpoint schema version drift")
    if meta.get("artifact_kind") != "fulltrack_train_checkpoint":
        raise FullTrackTrainingError("checkpoint artifact kind drift")
    _validate_json_safe(meta, "checkpoint")
    payload = {key: value for key, value in meta.items() if key != "checkpoint_sha256"}
    if meta.get("checkpoint_sha256") != stable_json_sha256(payload):
        raise FullTrackTrainingError("checkpoint JSON checksum mismatch")
    kind = str(meta["candidate_kind"])
    if expected_kind is not None and kind != expected_kind:
        raise FullTrackTrainingError("checkpoint candidate kind drift")
    if expected_job_config_sha256 is not None and meta["job_config_sha256"] != expected_job_config_sha256:
        raise FullTrackTrainingError("checkpoint job config drift")
    hidden = tuple(int(v) for v in meta["hidden_dims"])
    keys = _checkpoint_array_keys(kind, hidden)
    npz_path = root / "checkpoint.npz"
    raw = _safe_read_bytes(npz_path, "checkpoint.npz", _MAX_NPZ_BYTES)
    if _sha256_bytes(raw) != str(meta["arrays_npz_sha256"]):
        raise FullTrackTrainingError("checkpoint NPZ checksum mismatch")
    arrays = _read_npz_bytes(raw, "checkpoint.npz", keys)
    clean = _validate_exported_arrays(kind, arrays, hidden)
    return TrainingCheckpoint(metadata=meta, arrays=clean)


REPORT_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "job_status",
        "job_id",
        "fold",
        "seed",
        "candidate_kind",
        "created_at",
        "source_fingerprint",
        "store_binding",
        "store_binding_sha256",
        "store_manifest_sha256",
        "training_config",
        "training_config_sha256",
        "job_config_sha256",
        "dataset_hashes",
        "ranking_hashes",
        "view_hashes",
        "view_stats",
        "negative_mining",
        "metrics",
        "history",
        "resources",
        "model",
        "checkpoint",
        "notices",
        "report_sha256",
    }
)


def _report_with_hash(payload: Mapping[str, object]) -> Dict[str, object]:
    doc = dict(payload)
    if "report_sha256" in doc:
        doc.pop("report_sha256")
    doc["report_sha256"] = stable_json_sha256(doc)
    return doc


def save_training_report(path: Union[str, Path], payload: Mapping[str, object]) -> Dict[str, object]:
    doc = _report_with_hash(payload)
    if frozenset(doc.keys()) != REPORT_REQUIRED_FIELDS:
        raise FullTrackTrainingError("training report schema fields differ")
    _validate_json_safe(doc, "training report")
    atomic_write_json(path, doc)
    return doc


def load_training_report(path: Union[str, Path]) -> Dict[str, object]:
    report = _read_json(path, "training report")
    if frozenset(report.keys()) != REPORT_REQUIRED_FIELDS:
        raise FullTrackTrainingError("training report schema fields differ")
    if report.get("schema_version") != TRAIN_SCHEMA_VERSION:
        raise FullTrackTrainingError("training report schema version drift")
    if report.get("artifact_kind") != "fulltrack_train_report":
        raise FullTrackTrainingError("training report artifact kind drift")
    if report.get("job_status") != "complete":
        raise FullTrackTrainingError("training report is not complete")
    _validate_json_safe(report, "training report")
    payload = {key: value for key, value in report.items() if key != "report_sha256"}
    if report.get("report_sha256") != stable_json_sha256(payload):
        raise FullTrackTrainingError("training report checksum mismatch")
    return report


def _model_artifact_hashes(model_dir: Union[str, Path]) -> Dict[str, str]:
    root = _check_path_safety(model_dir, "model artifact directory")
    json_bytes = _safe_read_bytes(root / "model.json", "model.json", _MAX_JSON_BYTES)
    npz_bytes = _safe_read_bytes(root / "weights.npz", "weights.npz", _MAX_NPZ_BYTES)
    digest = hashlib.sha256()
    digest.update(b"model.json\0")
    digest.update(json_bytes)
    digest.update(b"weights.npz\0")
    digest.update(npz_bytes)
    return {
        "model_json_sha256": _sha256_bytes(json_bytes),
        "weights_npz_sha256": _sha256_bytes(npz_bytes),
        "artifact_sha256": digest.hexdigest(),
    }


@dataclass(frozen=True)
class TrainJobSpec:
    fold_index: int
    candidate_kind: str
    seed: int
    job_id: str
    relative_dir: str


@dataclass(frozen=True)
class TrainAllPlan:
    seeds: Tuple[int, ...]
    folds: Tuple[int, ...]
    candidates: Tuple[str, ...]
    jobs: Tuple[TrainJobSpec, ...]

    @property
    def job_count(self) -> int:
        return len(self.jobs)

    def as_dict(self) -> Dict[str, object]:
        return {
            "seeds": list(self.seeds),
            "folds": list(self.folds),
            "candidates": list(self.candidates),
            "job_count": len(self.jobs),
            "jobs": [
                {
                    "fold_index": job.fold_index,
                    "candidate_kind": job.candidate_kind,
                    "seed": job.seed,
                    "job_id": job.job_id,
                    "relative_dir": job.relative_dir,
                }
                for job in self.jobs
            ],
        }


def validate_train_all_seeds(seeds: Optional[Sequence[int]] = None) -> Tuple[int, ...]:
    values = tuple(DEFAULT_SEEDS if seeds is None else tuple(int(s) for s in seeds))
    if len(values) < 3:
        raise FullTrackTrainingError("train-all requires at least three distinct seeds")
    if len(set(values)) != len(values):
        raise FullTrackTrainingError("train-all seeds must be distinct")
    for seed in values:
        _strict_int(seed, "seed")
    return values


def build_train_all_plan(seeds: Optional[Sequence[int]] = None) -> TrainAllPlan:
    seed_values = validate_train_all_seeds(seeds)
    jobs: List[TrainJobSpec] = []
    for fold in OFFICIAL_FOLDS:
        for kind in CANDIDATE_KINDS:
            for seed in seed_values:
                job_id = f"fold-{fold}__{kind}__seed-{seed}"
                relative = f"fold-{fold}/{kind}/seed-{seed}"
                jobs.append(
                    TrainJobSpec(
                        fold_index=int(fold),
                        candidate_kind=kind,
                        seed=int(seed),
                        job_id=job_id,
                        relative_dir=relative,
                    )
                )
    return TrainAllPlan(
        seeds=seed_values,
        folds=OFFICIAL_FOLDS,
        candidates=tuple(CANDIDATE_KINDS),
        jobs=tuple(jobs),
    )


@dataclass(frozen=True)
class JobRunResult:
    spec: TrainJobSpec
    job_dir: Path
    status: str
    report: Mapping[str, object]


def _job_dir(output_dir: Union[str, Path], spec: TrainJobSpec) -> Path:
    root = _check_path_safety(output_dir, "output directory")
    if Path(spec.relative_dir).is_absolute() or ".." in Path(spec.relative_dir).parts:
        raise FullTrackTrainingError("unsafe job relative path")
    return _check_path_safety(root / Path(spec.relative_dir), "job directory")


def _expected_job_config_sha256(
    spec: TrainJobSpec,
    config: TrainingConfig,
    train_dataset_hash: str,
    validation_dataset_hash: str,
    train_ranking_hash: str,
    validation_ranking_hash: str,
    store_hash: str,
    source_fingerprint: str,
) -> str:
    return _expected_job_config_sha256_from_hash(
        spec,
        config.sha256,
        train_dataset_hash,
        validation_dataset_hash,
        train_ranking_hash,
        validation_ranking_hash,
        store_hash,
        source_fingerprint,
    )


def _expected_job_config_sha256_from_hash(
    spec: TrainJobSpec,
    training_config_sha256: str,
    train_dataset_hash: str,
    validation_dataset_hash: str,
    train_ranking_hash: str,
    validation_ranking_hash: str,
    store_hash: str,
    source_fingerprint: str,
) -> str:
    return stable_json_sha256(
        {
            "job_id": spec.job_id,
            "fold_index": spec.fold_index,
            "candidate_kind": spec.candidate_kind,
            "seed": spec.seed,
            "training_config_sha256": training_config_sha256,
            "train_dataset_hash": train_dataset_hash,
            "validation_dataset_hash": validation_dataset_hash,
            "train_ranking_hash": train_ranking_hash,
            "validation_ranking_hash": validation_ranking_hash,
            "store_binding_sha256": store_hash,
            "source_fingerprint": source_fingerprint,
            "no_tag_supervision": True,
        }
    )


def validate_training_report_bindings(
    report: Mapping[str, object],
    *,
    spec: TrainJobSpec,
    source_fingerprint: str,
    store_binding_hash: str,
    store_manifest_sha256: str,
    expected_training_config_sha256: Optional[str] = None,
) -> Mapping[str, object]:
    """Recompute semantic report bindings without opening model/checkpoint files."""
    declared_training_hash = _validate_sha256(
        report.get("training_config_sha256"), "training_config_sha256"
    )
    expected_scalars = {
        "job_id": spec.job_id,
        "fold": int(spec.fold_index),
        "seed": int(spec.seed),
        "candidate_kind": spec.candidate_kind,
        "source_fingerprint": source_fingerprint,
        "store_binding_sha256": store_binding_hash,
        "store_manifest_sha256": store_manifest_sha256,
    }
    if expected_training_config_sha256 is not None:
        expected_scalars["training_config_sha256"] = (
            expected_training_config_sha256
        )
    for key, expected in expected_scalars.items():
        if report.get(key) != expected:
            raise FullTrackTrainingError(f"stale job {spec.job_id}: {key} drift")

    store_binding = report.get("store_binding")
    if (
        not isinstance(store_binding, Mapping)
        or stable_json_sha256(store_binding) != store_binding_hash
        or store_binding.get("source_fingerprint") != source_fingerprint
        or store_binding.get("sealed_manifest_sha256") != store_manifest_sha256
    ):
        raise FullTrackTrainingError("store binding content/hash drift")
    training_config = report.get("training_config")
    if (
        not isinstance(training_config, Mapping)
        or stable_json_sha256(training_config) != declared_training_hash
    ):
        raise FullTrackTrainingError("training config content/hash drift")

    dataset_hashes = report.get("dataset_hashes")
    ranking_hashes = report.get("ranking_hashes")
    view_hashes = report.get("view_hashes")
    if (
        not isinstance(dataset_hashes, dict)
        or not isinstance(ranking_hashes, dict)
        or not isinstance(view_hashes, dict)
    ):
        raise FullTrackTrainingError(
            "report dataset/ranking/view hash sections are invalid"
        )
    for section, values in (
        ("dataset_hashes", dataset_hashes),
        ("ranking_hashes", ranking_hashes),
    ):
        if set(values) != {"train", "validation"}:
            raise FullTrackTrainingError(f"report {section} keys are invalid")
        _validate_sha256(values["train"], f"report {section} train")
        _validate_sha256(values["validation"], f"report {section} validation")
    if set(view_hashes) != {"train", "validation"}:
        raise FullTrackTrainingError("report view_hashes keys are invalid")
    for part in ("train", "validation"):
        hashes = view_hashes[part]
        if not isinstance(hashes, list) or not hashes:
            raise FullTrackTrainingError(
                f"report view_hashes {part} must be a non-empty list"
            )
        for value in hashes:
            _validate_sha256(value, f"report view_hashes {part}")

    expected_job_config = _expected_job_config_sha256_from_hash(
        spec,
        declared_training_hash,
        str(dataset_hashes["train"]),
        str(dataset_hashes["validation"]),
        str(ranking_hashes["train"]),
        str(ranking_hashes["validation"]),
        store_binding_hash,
        source_fingerprint,
    )
    if report.get("job_config_sha256") != expected_job_config:
        raise FullTrackTrainingError("report job config hash drift")
    return report


def try_load_reusable_job(
    job_dir: Union[str, Path],
    *,
    spec: TrainJobSpec,
    training_config_sha256: str,
    source_fingerprint: str,
    store_binding_hash: str,
    store_manifest_sha256: str,
) -> Optional[Dict[str, object]]:
    """Return a validated complete report, or None if the job can start cleanly.

    An empty directory can be left between job-directory creation and the first
    artifact write when a process is interrupted. Remove only that recoverable
    state; any nonempty stale/tampered/incomplete directory fails loudly.
    """

    root = _check_path_safety(job_dir, "job directory")
    if not root.exists():
        return None
    _reject_link_or_reparse(root, "job directory")
    report_path = root / "report.json"
    if not report_path.exists():
        try:
            root.rmdir()
        except OSError as exc:
            raise FullTrackTrainingError(
                f"stale/incomplete job directory without report: {root}"
            ) from exc
        return None
    report = load_training_report(report_path)
    validate_training_report_bindings(
        report,
        spec=spec,
        source_fingerprint=source_fingerprint,
        store_binding_hash=store_binding_hash,
        store_manifest_sha256=store_manifest_sha256,
        expected_training_config_sha256=training_config_sha256,
    )
    checkpoint = load_training_checkpoint(
        root / "checkpoint",
        expected_kind=spec.candidate_kind,
        expected_job_config_sha256=str(report["job_config_sha256"]),
    )
    checkpoint_report = report.get("checkpoint")
    if not isinstance(checkpoint_report, dict) or checkpoint_report != {
        "relative_dir": "checkpoint",
        "checkpoint_sha256": checkpoint.metadata["checkpoint_sha256"],
        "arrays_npz_sha256": checkpoint.metadata["arrays_npz_sha256"],
    }:
        raise FullTrackTrainingError("checkpoint/report hash binding drift")
    model_report = report.get("model")
    if not isinstance(model_report, dict) or not isinstance(
        model_report.get("fusion_metadata"), dict
    ):
        raise FullTrackTrainingError("report model section is invalid")
    fusion_metadata = model_report["fusion_metadata"]
    expected_checkpoint_metadata = {
        "job_id": spec.job_id,
        "candidate_kind": spec.candidate_kind,
        "fold": int(spec.fold_index),
        "seed": int(spec.seed),
        "hidden_dims": fusion_metadata.get("hidden_dims"),
        "embedding_dim": report["store_binding"].get("embedding_dim"),
        "training_config_sha256": training_config_sha256,
        "job_config_sha256": report["job_config_sha256"],
        "source_fingerprint": source_fingerprint,
        "store_binding_sha256": store_binding_hash,
    }
    for field, expected in expected_checkpoint_metadata.items():
        if checkpoint.metadata.get(field) != expected:
            raise FullTrackTrainingError(f"checkpoint {field} drift")
    model = load_fusion_artifact(root / "model")
    if (
        model.config.kind != spec.candidate_kind
        or int(model.config.seed) != int(spec.seed)
        or int(model.config.fold_index) != int(spec.fold_index)
        or model.config.config_sha256 != report["job_config_sha256"]
        or model.config.store_id != store_binding_hash
    ):
        raise FullTrackTrainingError("model artifact metadata drift")
    hashes = _model_artifact_hashes(root / "model")
    if model.metadata is None or model.metadata.as_dict() != model_report.get(
        "fusion_metadata"
    ):
        raise FullTrackTrainingError("model fusion metadata drift")
    for key, value in hashes.items():
        if model_report.get(key) != value:
            raise FullTrackTrainingError(f"model artifact {key} drift")
    return report


def _build_strict_job_report(
    *,
    spec: TrainJobSpec,
    result: CandidateTrainingResult,
    config: TrainingConfig,
    source_fingerprint: str,
    store_binding: Mapping[str, object],
    store_hash: str,
    checkpoint: TrainingCheckpoint,
    model_metadata: Mapping[str, object],
    model_hashes: Mapping[str, str],
) -> Dict[str, object]:
    metrics = {
        "train_loss": result.report["train_loss"],
        "validation_loss": result.report["validation_loss"],
        "train_ranking_accuracy": result.report["train_ranking_accuracy"],
        "validation_ranking_accuracy": result.report["validation_ranking_accuracy"],
        "train_pairwise_auc": result.report["train_pairwise_auc"],
        "validation_pairwise_auc": result.report["validation_pairwise_auc"],
        "early_stopping_metric": result.report["early_stopping_metric"],
        "epochs_ran": result.report["epochs_ran"],
        "best_epoch": result.report["best_epoch"],
    }
    payload = {
        "schema_version": TRAIN_SCHEMA_VERSION,
        "artifact_kind": "fulltrack_train_report",
        "job_status": "complete",
        "job_id": spec.job_id,
        "fold": int(spec.fold_index),
        "seed": int(spec.seed),
        "candidate_kind": spec.candidate_kind,
        "created_at": _now_utc(),
        "source_fingerprint": source_fingerprint,
        "store_binding": dict(store_binding),
        "store_binding_sha256": store_hash,
        "store_manifest_sha256": str(store_binding["sealed_manifest_sha256"]),
        "training_config": config.as_dict(),
        "training_config_sha256": config.sha256,
        "job_config_sha256": result.report["job_config_sha256"],
        "dataset_hashes": {
            "train": result.report["train_dataset_hash"],
            "validation": result.report["validation_dataset_hash"],
        },
        "ranking_hashes": {
            "train": result.report["train_ranking_hash"],
            "validation": result.report["validation_ranking_hash"],
        },
        "view_hashes": {
            "train": result.report["train_view_hashes"],
            "validation": result.report["validation_view_hashes"],
        },
        "view_stats": {
            "train": result.report["train_view_stats"],
            "validation": result.report["validation_view_stats"],
        },
        "negative_mining": {
            "train": dict(result.train_ranking.stats),
            "validation": dict(result.validation_ranking.stats),
        },
        "metrics": metrics,
        "history": result.report["history"],
        "resources": {
            "wall_time_seconds": result.report["wall_time_seconds"],
            "cpu_rss_peak_bytes": result.report["cpu_rss_peak_bytes"],
            "cuda_peak_bytes": result.report["cuda_peak_bytes"],
            "device": result.report["device"],
        },
        "model": {
            **dict(model_hashes),
            "fusion_metadata": dict(model_metadata),
            "parameter_count": result.report["parameter_count"],
            "model_bytes": result.report["model_bytes"],
            "runtime_parity_abs_diff": result.report["runtime_parity_abs_diff"],
        },
        "checkpoint": {
            "relative_dir": "checkpoint",
            "checkpoint_sha256": checkpoint.metadata["checkpoint_sha256"],
            "arrays_npz_sha256": checkpoint.metadata["arrays_npz_sha256"],
        },
        "notices": [NO_TAG_SELF_SUPERVISION_NOTICE],
    }
    return _report_with_hash(payload)


def run_train_job(
    context: JamendoContext,
    reader: FullTrackStoreReader,
    split: FoldSplit,
    spec: TrainJobSpec,
    *,
    config: TrainingConfig,
    output_dir: Union[str, Path],
    prepared_data: Optional[PreparedTrainingData] = None,
    dedicated_cuda_stream: bool = False,
) -> JobRunResult:
    """Run or strictly reuse one fold/candidate/seed job."""

    if spec.fold_index != split.fold_index:
        raise FullTrackTrainingError("job spec fold does not match split")
    if spec.candidate_kind not in CANDIDATE_KINDS:
        raise FullTrackTrainingError(f"unknown candidate kind {spec.candidate_kind}")
    config.validate(production_train_all=False)
    binding = validate_store_context_binding(context, reader)
    store_hash = stable_json_sha256(binding)
    source = _validate_sha256(context.source_fingerprint, "source_fingerprint")
    root = _job_dir(output_dir, spec)
    existing = try_load_reusable_job(
        root,
        spec=spec,
        training_config_sha256=config.sha256,
        source_fingerprint=source,
        store_binding_hash=store_hash,
        store_manifest_sha256=str(binding["sealed_manifest_sha256"]),
    )
    if existing is not None:
        return JobRunResult(spec=spec, job_dir=root, status="reused", report=existing)

    root.mkdir(parents=True, exist_ok=False)
    _reject_link_or_reparse(root, "job directory")
    prepared = prepared_data or prepare_training_data(
        context,
        reader,
        split,
        seed=spec.seed,
        config=config,
    )
    train_dataset = prepared.train_dataset
    validation_dataset = prepared.validation_dataset
    train_ranking_preview = prepared.train_ranking
    validation_ranking_preview = prepared.validation_ranking
    expected_train_ids = _limited_ids(
        split.train_track_ids, config.max_train_tracks, "train", config
    )
    expected_validation_ids = _limited_ids(
        split.validation_track_ids,
        config.max_validation_tracks,
        "validation",
        config,
    )
    if (
        train_dataset.fold_index != spec.fold_index
        or validation_dataset.fold_index != spec.fold_index
        or train_dataset.seed != spec.seed
        or validation_dataset.seed != spec.seed
        or train_dataset.part != "train"
        or validation_dataset.part != "validation"
        or train_dataset.track_ids != expected_train_ids
        or validation_dataset.track_ids != expected_validation_ids
        or train_ranking_preview.dataset_hash != train_dataset.dataset_hash
        or validation_ranking_preview.dataset_hash
        != validation_dataset.dataset_hash
    ):
        raise FullTrackTrainingError(
            "prepared training data does not match the requested fold/seed split"
        )
    job_config_sha = _expected_job_config_sha256(
        spec,
        config,
        train_dataset.dataset_hash,
        validation_dataset.dataset_hash,
        train_ranking_preview.ranking_hash,
        validation_ranking_preview.ranking_hash,
        store_hash,
        source,
    )
    result = train_candidate_from_datasets(
        spec.candidate_kind,
        train_dataset,
        validation_dataset,
        config=config,
        seed=spec.seed,
        store_binding_hash=store_hash,
        source_fingerprint=source,
        job_config_sha256=job_config_sha,
        train_ranking=train_ranking_preview,
        validation_ranking=validation_ranking_preview,
        dedicated_cuda_stream=dedicated_cuda_stream,
    )
    hidden = tuple(config.monotonic_hidden_dims) if spec.candidate_kind == "monotonic_network" else ()
    checkpoint = save_training_checkpoint(
        root / "checkpoint",
        job_id=spec.job_id,
        kind=spec.candidate_kind,
        fold=spec.fold_index,
        seed=spec.seed,
        hidden_dims=hidden,
        embedding_dim=train_dataset.embedding_dim,
        training_config_sha256=config.sha256,
        job_config_sha256=job_config_sha,
        source_fingerprint=source,
        store_binding_sha256=store_hash,
        arrays=result.arrays,
    )
    metadata = save_fusion_artifact(result.model, root / "model")
    model_hashes = _model_artifact_hashes(root / "model")
    report = _build_strict_job_report(
        spec=spec,
        result=result,
        config=config,
        source_fingerprint=source,
        store_binding=binding,
        store_hash=store_hash,
        checkpoint=checkpoint,
        model_metadata=metadata.as_dict(),
        model_hashes=model_hashes,
    )
    saved = save_training_report(root / "report.json", report)
    validated = try_load_reusable_job(
        root,
        spec=spec,
        training_config_sha256=config.sha256,
        source_fingerprint=source,
        store_binding_hash=store_hash,
        store_manifest_sha256=str(binding["sealed_manifest_sha256"]),
    )
    if validated is None:
        raise FullTrackTrainingError("job validation failed after save")
    return JobRunResult(spec=spec, job_dir=root, status="trained", report=saved)


def train_all(
    context: JamendoContext,
    reader: FullTrackStoreReader,
    *,
    output_dir: Union[str, Path],
    seeds: Optional[Sequence[int]] = None,
    config: Optional[TrainingConfig] = None,
    candidate_workers: int = 1,
    pairwise_cosine_mode: str = "legacy-v1",
    dry_run: bool = False,
) -> Mapping[str, object]:
    """Run/reuse the exact 5 x 3 x >=3 train-all matrix."""

    cfg = config or TrainingConfig()
    cfg.validate(production_train_all=True)
    if pairwise_cosine_mode not in ("legacy-v1", "linear-v2"):
        raise FullTrackTrainingError(
            f"unknown pairwise cosine mode {pairwise_cosine_mode!r}"
        )
    if (
        isinstance(candidate_workers, bool)
        or not isinstance(candidate_workers, int)
        or not 1 <= candidate_workers <= len(CANDIDATE_KINDS)
    ):
        raise FullTrackTrainingError(
            f"candidate_workers must be an integer in [1, {len(CANDIDATE_KINDS)}]"
        )
    plan = build_train_all_plan(seeds)
    splits = validate_official_artist_splits(
        context,
        reader,
        required_folds=OFFICIAL_FOLDS,
        require_all_official=True,
    )
    binding = validate_store_context_binding(context, reader)
    if dry_run:
        return {
            "dry_run": True,
            "plan": plan.as_dict(),
            "splits": [split.as_dict() for split in splits],
            "store_binding_sha256": stable_json_sha256(binding),
            "source_fingerprint": context.source_fingerprint,
            "pairwise_cosine_mode": pairwise_cosine_mode,
            "notices": [NO_TAG_SELF_SUPERVISION_NOTICE],
        }
    split_by_fold = {split.fold_index: split for split in splits}
    results_by_job: Dict[str, Mapping[str, object]] = {}
    specs_by_group: Dict[Tuple[int, int], List[TrainJobSpec]] = {}
    for spec in plan.jobs:
        specs_by_group.setdefault((spec.fold_index, spec.seed), []).append(spec)
    for fold_index in plan.folds:
        for seed in plan.seeds:
            pending: List[TrainJobSpec] = []
            for spec in specs_by_group[(fold_index, seed)]:
                root = _job_dir(output_dir, spec)
                if not (root / "report.json").is_file():
                    pending.append(spec)
                    continue
                result = run_train_job(
                    context,
                    reader,
                    split_by_fold[fold_index],
                    spec,
                    config=cfg,
                    output_dir=output_dir,
                )
                results_by_job[spec.job_id] = {
                    "job_id": spec.job_id,
                    "relative_dir": spec.relative_dir,
                    "status": result.status,
                    "report_sha256": result.report["report_sha256"],
                }
            if pending:
                for spec in pending:
                    try_load_reusable_job(
                        _job_dir(output_dir, spec),
                        spec=spec,
                        training_config_sha256=cfg.sha256,
                        source_fingerprint=context.source_fingerprint,
                        store_binding_hash=stable_json_sha256(binding),
                        store_manifest_sha256=str(
                            binding["sealed_manifest_sha256"]
                        ),
                    )
                prepared = prepare_training_data(
                    context,
                    reader,
                    split_by_fold[fold_index],
                    seed=seed,
                    config=cfg,
                    pairwise_cosine_mode=pairwise_cosine_mode,
                )

                def run_pending(spec: TrainJobSpec) -> JobRunResult:
                    return run_train_job(
                        context,
                        reader,
                        split_by_fold[fold_index],
                        spec,
                        config=cfg,
                        output_dir=output_dir,
                        prepared_data=prepared,
                        dedicated_cuda_stream=candidate_workers > 1,
                    )

                if candidate_workers == 1 or len(pending) == 1:
                    pending_results = [run_pending(spec) for spec in pending]
                else:
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=min(candidate_workers, len(pending)),
                        thread_name_prefix="fulltrack-candidate",
                    ) as executor:
                        futures = [executor.submit(run_pending, spec) for spec in pending]
                        pending_results = [future.result() for future in futures]
                for result in pending_results:
                    spec = result.spec
                    results_by_job[spec.job_id] = {
                        "job_id": spec.job_id,
                        "relative_dir": spec.relative_dir,
                        "status": result.status,
                        "report_sha256": result.report["report_sha256"],
                    }
    results = [results_by_job[spec.job_id] for spec in plan.jobs]
    return {
        "dry_run": False,
        "plan": plan.as_dict(),
        "results": results,
        "completed_jobs": len(results),
        "pairwise_cosine_mode": pairwise_cosine_mode,
        "notices": [NO_TAG_SELF_SUPERVISION_NOTICE],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m soundalike.ml.fulltrack_train")
    sub = parser.add_subparsers(dest="command", required=True)
    train_all_parser = sub.add_parser("train-all", help="train/reuse all full-track fusion candidates")
    train_all_parser.add_argument("--metadata", type=Path)
    train_all_parser.add_argument("--audio", type=Path)
    train_all_parser.add_argument("--state", type=Path)
    train_all_parser.add_argument("--store", type=Path)
    train_all_parser.add_argument("--output", type=Path)
    train_all_parser.add_argument("--seed", type=int, action="append", default=None)
    train_all_parser.add_argument("--epochs", type=int, default=TrainingConfig.max_epochs)
    train_all_parser.add_argument("--patience", type=int, default=TrainingConfig.patience)
    train_all_parser.add_argument("--learning-rate", type=float, default=TrainingConfig.learning_rate)
    train_all_parser.add_argument("--weight-decay", type=float, default=TrainingConfig.weight_decay)
    train_all_parser.add_argument("--hard-negatives", type=int, default=TrainingConfig.hard_negatives)
    train_all_parser.add_argument("--random-negatives", type=int, default=TrainingConfig.random_negatives)
    train_all_parser.add_argument("--maxsim-budget", type=int, default=TrainingConfig.maxsim_budget)
    train_all_parser.add_argument("--top-k", type=int, default=TrainingConfig.top_k)
    train_all_parser.add_argument("--coverage-threshold", type=float, default=TrainingConfig.coverage_threshold)
    train_all_parser.add_argument("--hidden-dim", type=int, action="append", default=None)
    train_all_parser.add_argument("--max-train-tracks", type=int, default=None)
    train_all_parser.add_argument("--max-validation-tracks", type=int, default=None)
    train_all_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    train_all_parser.add_argument(
        "--candidate-workers", type=int, choices=range(1, len(CANDIDATE_KINDS) + 1), default=1
    )
    train_all_parser.add_argument(
        "--pairwise-cosine-mode",
        choices=("legacy-v1", "linear-v2"),
        default="legacy-v1",
    )
    train_all_parser.add_argument("--non-production", action="store_true")
    train_all_parser.add_argument("--dry-run", action="store_true")
    train_all_parser.add_argument("--plan", action="store_true", help="print the 45+ job matrix and exit")
    return parser.parse_args(argv)


def _config_from_args(args: argparse.Namespace) -> TrainingConfig:
    hidden = tuple(args.hidden_dim) if args.hidden_dim else TrainingConfig.monotonic_hidden_dims
    cfg = TrainingConfig(
        max_epochs=int(args.epochs),
        patience=int(args.patience),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        hard_negatives=int(args.hard_negatives),
        random_negatives=int(args.random_negatives),
        maxsim_budget=int(args.maxsim_budget),
        top_k=int(args.top_k),
        coverage_threshold=float(args.coverage_threshold),
        monotonic_hidden_dims=tuple(int(v) for v in hidden),
        max_train_tracks=args.max_train_tracks,
        max_validation_tracks=args.max_validation_tracks,
        device=str(args.device),
        non_production=bool(args.non_production),
    )
    cfg.validate(production_train_all=True)
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    if args.command != "train-all":
        raise FullTrackTrainingError(f"unknown command {args.command!r}")
    seeds = validate_train_all_seeds(args.seed)
    cfg = _config_from_args(args)
    plan = build_train_all_plan(seeds)
    if args.plan:
        print(json.dumps({"plan": plan.as_dict(), "notices": [NO_TAG_SELF_SUPERVISION_NOTICE]}, indent=2, sort_keys=True))
        return 0
    for label in ("metadata", "audio", "state", "store", "output"):
        if getattr(args, label) is None:
            raise FullTrackTrainingError(f"--{label} is required unless --plan is used")
    context = load_jamendo_context(
        args.metadata,
        args.audio,
        args.state,
        production=not bool(args.non_production),
    )
    reader = FullTrackStoreReader(args.store, expected_source_fingerprint=context.source_fingerprint)
    try:
        result = train_all(
            context,
            reader,
            output_dir=args.output,
            seeds=seeds,
            config=cfg,
            candidate_workers=int(args.candidate_workers),
            pairwise_cosine_mode=str(args.pairwise_cosine_mode),
            dry_run=bool(args.dry_run),
        )
    finally:
        reader.close()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except FullTrackTrainingError as exc:
        print(f"fulltrack_train: {exc}", file=sys.stderr)
        raise SystemExit(2)
