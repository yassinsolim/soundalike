"""NumPy-only frozen sealed full-track fusion model for inference/runtime.

Candidate kinds
---------------
``nonnegative_linear``
    Non-negative weighted sum of 16 oriented utility features.
``monotonic_network``
    Low-capacity positive-weight MLP: ReLU hidden, sigmoid output.
``channel_gated_embedding``
    Nonneg sum-to-one gates blend 4 channel summaries into a
    deployable L2-normalised track embedding; pair score in [0,1].

Pair-feature schema  (FEATURE_NAMES / FEATURE_DIM = 16)
---------------------------------------------------------
All features clipped to [0,1], higher = more similar.

  0  global_cosine           (1+cos(g_a,g_b))/2
  1  uniform_maxsim_sym      symmetric budget-window MaxSim
  2  uniform_maxsim_ab       directional A->B MaxSim
  3  uniform_maxsim_ba       directional B->A MaxSim
  4  repeated_maxsim_sym     symmetric repeated-section MaxSim
  5  salient_maxsim_sym      symmetric salient-section MaxSim
  6  coverage_topk_a         fraction of A budget windows matched
  7  coverage_topk_b         fraction of B budget windows matched
  8  asymmetry_utility       1 - |uniform_ab - uniform_ba|
  9  recurrence_indicator    mean pairwise self-recurrence of repeated
 10  steady_texture_a        clip(1-std(cos(budget_a, g_a)), 0, 1)
 11  steady_texture_b        clip(1-std(cos(budget_b, g_b)), 0, 1)
 12  topk_maxsim_ab          mean of top-k A->B per-window max cosines
 13  topk_maxsim_ba          mean of top-k B->A per-window max cosines
 14  repeated_temporal_sim   temporal-position similarity of repeated indices
 15  salient_temporal_sim    temporal-position similarity of salient indices

Track duck-type: global_embedding, window_embeddings, repeated_sections,
salient_sections, and optional repeated_indices / salient_indices.

Ablations: 'none' | 'global_only' | 'no_sections'

Artifact format: model.json + weights.npz with SHA-256 cross-binding,
atomic fsync+replace writes, ZIP preflight, symlink/junction protection,
deep-copied read-only weight arrays.
"""
from __future__ import annotations

import copy
import datetime
import hashlib
import io
import json
import os
import re
import secrets
import stat
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .fulltrack_store import stable_json_sha256


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

FUSION_SCHEMA_VERSION: int = 1

CANDIDATE_KINDS: Tuple[str, ...] = (
    "nonnegative_linear",
    "monotonic_network",
    "channel_gated_embedding",
)

FEATURE_NAMES: Tuple[str, ...] = (
    "global_cosine",
    "uniform_maxsim_sym",
    "uniform_maxsim_ab",
    "uniform_maxsim_ba",
    "repeated_maxsim_sym",
    "salient_maxsim_sym",
    "coverage_topk_a",
    "coverage_topk_b",
    "asymmetry_utility",
    "recurrence_indicator",
    "steady_texture_a",
    "steady_texture_b",
    "topk_maxsim_ab",
    "topk_maxsim_ba",
    "repeated_temporal_sim",
    "salient_temporal_sim",
)

FEATURE_DIM: int = len(FEATURE_NAMES)

ABLATIONS: Tuple[str, ...] = ("none", "global_only", "no_sections")

_CHANNEL_COUNT: int = 4  # global / uniform / repeated / salient

_MAX_JSON_BYTES: int = 256 * 1024
_MAX_NPZ_BYTES: int = 64 * 1024 * 1024
_MAX_WEIGHT_MAGNITUDE: float = 1e15

_DEFAULT_MAXSIM_BUDGET: int = 8
_DEFAULT_TOP_K: int = 4
_DEFAULT_COVERAGE_THRESHOLD: float = 0.5

_MAX_MAXSIM_BUDGET: int = 100_000
_MAX_TOP_K: int = 100_000

_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

# Feature-ablation masks (1=keep, 0=zero-out).  16 features.
_FEATURE_MASK: Dict[str, Tuple[int, ...]] = {
    "none":        (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
    "global_only": (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    "no_sections": (1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0),
}

_CHANNEL_MASK: Dict[str, Tuple[int, ...]] = {
    "none":        (1, 1, 1, 1),
    "global_only": (1, 0, 0, 0),
    "no_sections": (1, 1, 0, 0),
}

_JSON_REQUIRED_FIELDS: frozenset = frozenset({
    "schema_version", "kind", "model_id", "store_id", "config_sha256",
    "fold_index", "embedding_dim", "feature_dim", "feature_names",
    "maxsim_budget", "top_k", "coverage_threshold", "seed", "hidden_dims",
    "created_at", "npz_sha256", "json_payload_sha256",
})


# ---------------------------------------------------------------------------
# Error
# ---------------------------------------------------------------------------


class FusionError(RuntimeError):
    """Invalid artifact, config, shape, tamper, or scoring failure."""


# ---------------------------------------------------------------------------
# FusionConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FusionConfig:
    """Hyper-parameters bound to a FusionModel artifact.

    Parameters
    ----------
    kind : str
        One of CANDIDATE_KINDS.
    embedding_dim : int
        Track embedding dimensionality (must match stored tracks).
    maxsim_budget : int
        Number of budget windows for MaxSim/coverage computations.
    top_k : int
        Number of top per-window max-cosines used for top-k features.
    coverage_threshold : float
        Post-(1+cos)/2 mapped cosine threshold in [0,1] for coverage.
    hidden_dims : tuple of int
        Hidden-layer sizes for monotonic_network (must be empty for others).
    """

    kind: str
    embedding_dim: int
    maxsim_budget: int = _DEFAULT_MAXSIM_BUDGET
    top_k: int = _DEFAULT_TOP_K
    coverage_threshold: float = _DEFAULT_COVERAGE_THRESHOLD
    seed: int = 0
    model_id: str = ""
    store_id: str = ""
    config_sha256: str = ""
    fold_index: int = 0
    hidden_dims: Tuple[int, ...] = ()

    def validate(self) -> None:
        if self.kind not in CANDIDATE_KINDS:
            raise FusionError(
                f"unknown kind {self.kind!r}; must be one of {CANDIDATE_KINDS}"
            )
        if not isinstance(self.embedding_dim, int) or isinstance(self.embedding_dim, bool):
            raise FusionError("embedding_dim must be an integer (not bool)")
        if self.embedding_dim <= 0:
            raise FusionError("embedding_dim must be positive")
        if not isinstance(self.maxsim_budget, int) or isinstance(self.maxsim_budget, bool):
            raise FusionError("maxsim_budget must be an integer (not bool)")
        if self.maxsim_budget <= 0:
            raise FusionError("maxsim_budget must be positive")
        if not isinstance(self.top_k, int) or isinstance(self.top_k, bool):
            raise FusionError("top_k must be an integer (not bool)")
        if self.top_k <= 0:
            raise FusionError("top_k must be positive")
        if not isinstance(self.coverage_threshold, (int, float)) or isinstance(self.coverage_threshold, bool):
            raise FusionError("coverage_threshold must be numeric")
        ct = float(self.coverage_threshold)
        if not (0.0 <= ct <= 1.0) or not np.isfinite(ct):
            raise FusionError("coverage_threshold must be a finite value in [0, 1]")
        if not isinstance(self.fold_index, int) or isinstance(self.fold_index, bool):
            raise FusionError("fold_index must be an integer (not bool)")
        if self.fold_index < 0:
            raise FusionError("fold_index must be non-negative")
        for dim in self.hidden_dims:
            if not isinstance(dim, int) or isinstance(dim, bool):
                raise FusionError("all hidden_dims must be integers (not bool)")
            if dim <= 0 or dim > 4096:
                raise FusionError("all hidden_dims must be in [1, 4096]")
        if self.kind == "monotonic_network" and not self.hidden_dims:
            raise FusionError("monotonic_network requires at least one hidden layer")
        if self.kind != "monotonic_network" and self.hidden_dims:
            raise FusionError(
                f"hidden_dims must be empty for {self.kind}"
            )

    def as_dict(self) -> Dict[str, object]:
        return {
            "kind": self.kind,
            "embedding_dim": self.embedding_dim,
            "maxsim_budget": self.maxsim_budget,
            "top_k": self.top_k,
            "coverage_threshold": float(self.coverage_threshold),
            "seed": self.seed,
            "model_id": self.model_id,
            "store_id": self.store_id,
            "config_sha256": self.config_sha256,
            "fold_index": self.fold_index,
            "hidden_dims": list(self.hidden_dims),
        }


# ---------------------------------------------------------------------------
# FusionMetadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FusionMetadata:
    """Read-only metadata attached to every saved/loaded FusionModel."""

    kind: str
    model_id: str
    store_id: str
    config_sha256: str
    fold_index: int
    embedding_dim: int
    feature_dim: int
    maxsim_budget: int
    top_k: int
    coverage_threshold: float
    seed: int
    hidden_dims: Tuple[int, ...]
    json_payload_sha256: str
    npz_sha256: str
    created_at: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "kind": self.kind,
            "model_id": self.model_id,
            "store_id": self.store_id,
            "config_sha256": self.config_sha256,
            "fold_index": self.fold_index,
            "embedding_dim": self.embedding_dim,
            "feature_dim": self.feature_dim,
            "maxsim_budget": self.maxsim_budget,
            "top_k": self.top_k,
            "coverage_threshold": float(self.coverage_threshold),
            "seed": self.seed,
            "hidden_dims": list(self.hidden_dims),
            "json_payload_sha256": self.json_payload_sha256,
            "npz_sha256": self.npz_sha256,
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# PairFeatures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairFeatures:
    """Stable bounded pair-feature schema.  All values in [0, 1].

    Orientation: higher = more similar for every feature, including
    asymmetry_utility = 1 - raw_asymmetry (perfect symmetry -> 1).
    Empty 2-D section arrays produce neutral 0.5 for section features.
    """

    global_cosine: float
    uniform_maxsim_sym: float
    uniform_maxsim_ab: float
    uniform_maxsim_ba: float
    repeated_maxsim_sym: float
    salient_maxsim_sym: float
    coverage_topk_a: float
    coverage_topk_b: float
    asymmetry_utility: float
    recurrence_indicator: float
    steady_texture_a: float
    steady_texture_b: float
    topk_maxsim_ab: float
    topk_maxsim_ba: float
    repeated_temporal_sim: float
    salient_temporal_sim: float

    def to_vector(self) -> np.ndarray:
        """Return float64 array of shape (FEATURE_DIM,) in canonical order."""
        return np.array(
            [
                self.global_cosine,
                self.uniform_maxsim_sym,
                self.uniform_maxsim_ab,
                self.uniform_maxsim_ba,
                self.repeated_maxsim_sym,
                self.salient_maxsim_sym,
                self.coverage_topk_a,
                self.coverage_topk_b,
                self.asymmetry_utility,
                self.recurrence_indicator,
                self.steady_texture_a,
                self.steady_texture_b,
                self.topk_maxsim_ab,
                self.topk_maxsim_ba,
                self.repeated_temporal_sim,
                self.salient_temporal_sim,
            ],
            dtype=np.float64,
        )


# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------


def _is_reparse_point(p: Path) -> bool:
    """Check for Windows reparse point (junction/mount) via lstat."""
    if os.name != "nt":
        return False
    try:
        st = os.lstat(str(p))
        attr = getattr(st, "st_file_attributes", 0)
        rp_bit = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
        return bool(attr & rp_bit)
    except OSError:
        return False


def _reject_link_or_reparse(p: Path, label: str) -> None:
    """Raise FusionError if *p* is a symlink, junction, or reparse point."""
    if p.is_symlink():
        raise FusionError(f"{label} may not be a symlink or junction: {p}")
    if _is_reparse_point(p):
        raise FusionError(f"{label} may not be a reparse point/junction: {p}")


def _check_path_safety(raw: Union[str, Path], label: str) -> Path:
    """Validate unresolved path and all existing parents for symlinks.

    Returns the resolved path.  Checks the *original* unresolved path
    and every existing ancestor before calling ``resolve()``.
    """
    p = Path(raw)
    # Check the path itself (before resolution) if it exists
    if p.exists() or p.is_symlink():
        _reject_link_or_reparse(p, label)
    # Check all existing parent components
    for parent in p.parents:
        if parent.exists() or parent.is_symlink():
            _reject_link_or_reparse(parent, f"{label} (ancestor)")
    return p.resolve()


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _normalize_l2(v: np.ndarray) -> np.ndarray:
    """L2-normalise a 1-D float64 vector; overflow-safe via pre-scaling."""
    if not np.all(np.isfinite(v)):
        raise FusionError("embedding contains non-finite values")
    max_abs = float(np.max(np.abs(v)))
    if max_abs == 0.0:
        raise FusionError("embedding is a zero or near-zero vector")
    scaled = v / max_abs
    norm = float(np.linalg.norm(scaled))
    if norm <= 1e-12:
        raise FusionError("embedding is a zero or near-zero vector")
    result = scaled / norm
    if not np.all(np.isfinite(result)):
        raise FusionError("normalization produced non-finite values")
    return result


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise each row of a 2-D float64 matrix; overflow-safe."""
    if matrix.ndim != 2 or len(matrix) == 0:
        raise FusionError("embeddings must be a non-empty 2-D matrix")
    if not np.all(np.isfinite(matrix)):
        raise FusionError("embeddings contain non-finite values")
    row_max = np.max(np.abs(matrix), axis=1, keepdims=True)
    row_max = np.where(row_max == 0.0, 1.0, row_max)
    scaled = matrix / row_max
    norms = np.linalg.norm(scaled, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise FusionError("embeddings contain a zero row")
    result = scaled / norms
    if not np.all(np.isfinite(result)):
        raise FusionError("row normalization produced non-finite values")
    return result


def _fixed_budget_indices(count: int, budget: int) -> np.ndarray:
    """Select exactly *budget* indices from [0, count)."""
    if count <= 0 or budget <= 0:
        raise FusionError("count and budget must be positive")
    return np.rint(np.linspace(0, count - 1, num=budget)).astype(np.int64)


def _self_recurrence(sections: np.ndarray) -> float:
    """Mean off-diagonal pairwise cosine of L2-normalised rows -> [0,1].

    Returns 0.5 (neutral) when fewer than 2 sections are present.
    """
    k = len(sections)
    if k < 2:
        return 0.5
    sim = sections @ sections.T
    off_diag = (float(sim.sum()) - k) / float(k * (k - 1))
    return float(np.clip((1.0 + off_diag) / 2.0, 0.0, 1.0))


def _freeze_array(arr: np.ndarray) -> np.ndarray:
    """Deep-copy and mark read-only."""
    out = arr.copy()
    out.flags.writeable = False
    return out



def _validate_track_embedding_dim(
    track: object, expected_dim: int, label: str
) -> None:
    """Validate all embedding widths on *track* against *expected_dim*."""
    g = np.asarray(track.global_embedding, dtype=np.float64).reshape(-1)
    if len(g) != expected_dim:
        raise FusionError(
            f"{label}: global_embedding dim {len(g)} != "
            f"config.embedding_dim {expected_dim}"
        )
    win = np.asarray(track.window_embeddings, dtype=np.float64)
    if win.ndim >= 2 and win.shape[-1] != expected_dim:
        raise FusionError(
            f"{label}: window_embeddings dim {win.shape[-1]} != "
            f"config.embedding_dim {expected_dim}"
        )
    rep = np.asarray(track.repeated_sections, dtype=np.float64)
    if rep.ndim == 2 and len(rep) > 0 and rep.shape[1] != expected_dim:
        raise FusionError(
            f"{label}: repeated_sections dim {rep.shape[1]} != "
            f"config.embedding_dim {expected_dim}"
        )
    sal = np.asarray(track.salient_sections, dtype=np.float64)
    if sal.ndim == 2 and len(sal) > 0 and sal.shape[1] != expected_dim:
        raise FusionError(
            f"{label}: salient_sections dim {sal.shape[1]} != "
            f"config.embedding_dim {expected_dim}"
        )

def _deep_copy_weights(weights: Dict[str, object]) -> Dict[str, object]:
    """Deep-copy all weight arrays and mark them read-only."""
    result: Dict[str, object] = {}
    for key, val in weights.items():
        if isinstance(val, np.ndarray):
            result[key] = _freeze_array(val)
        elif isinstance(val, list):
            result[key] = [
                _freeze_array(a) if isinstance(a, np.ndarray) else copy.deepcopy(a)
                for a in val
            ]
        else:
            result[key] = copy.deepcopy(val)
    return result


def _fsync_directory(directory: Path) -> None:
    """Best-effort fsync of a directory (no-op on Windows)."""
    if os.name == "nt":
        return
    try:
        fd = os.open(str(directory), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _safe_remove_tmp(tmp: Path) -> None:
    """Remove a temp file, ignoring FileNotFoundError."""
    try:
        tmp.unlink()
    except FileNotFoundError:
        pass



def _stat_identity_match(a: os.stat_result, b: os.stat_result) -> bool:
    """Best-effort check whether two stat results refer to the same file.

    Uses (st_dev, st_ino) when both inodes are non-zero.  Falls back
    to (st_dev, st_size, st_mtime_ns) on platforms where st_ino is
    zero.  Not race-free; provides practical TOCTOU mitigation.
    """
    if a.st_dev != b.st_dev:
        return False
    if a.st_ino != 0 and b.st_ino != 0:
        return a.st_ino == b.st_ino
    return a.st_size == b.st_size and a.st_mtime_ns == b.st_mtime_ns


def _safe_read_file(path: Path, label: str, max_bytes: int) -> bytes:
    """Read file into bounded byte snapshot with safety checks.

    Uses O_NOFOLLOW on Unix, lstat/fstat identity verification,
    and reparse-point rejection on Windows.  Not race-free but
    significantly reduces the TOCTOU attack surface.
    """
    _reject_link_or_reparse(path, label)

    try:
        st_before = os.lstat(str(path))
    except OSError as exc:
        raise FusionError(f"cannot stat {label}: {exc}") from exc

    if not stat.S_ISREG(st_before.st_mode):
        raise FusionError(f"{label} is not a regular file")

    if st_before.st_size > max_bytes:
        raise FusionError(
            f"{label} exceeds maximum size "
            f"({st_before.st_size} > {max_bytes})"
        )

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    if os.name != "nt":
        flags |= getattr(os, "O_NOFOLLOW", 0)

    try:
        fd = os.open(str(path), flags)
    except OSError as exc:
        raise FusionError(f"cannot open {label}: {exc}") from exc

    try:
        st_fd = os.fstat(fd)
        if not _stat_identity_match(st_before, st_fd):
            raise FusionError(
                f"{label}: file identity changed between lstat and open"
            )

        chunks: list = []
        total = 0
        while True:
            to_read = min(65536, max_bytes + 1 - total)
            if to_read <= 0:
                break
            chunk = os.read(fd, to_read)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise FusionError(f"{label} exceeds maximum size")
        data = b"".join(chunks)
    finally:
        os.close(fd)

    _reject_link_or_reparse(path, label)
    try:
        st_after = os.lstat(str(path))
    except OSError as exc:
        raise FusionError(
            f"cannot stat {label} after read: {exc}"
        ) from exc

    if not _stat_identity_match(st_before, st_after):
        raise FusionError(f"{label}: file identity changed during read")

    return data


def _unique_tmp_path(target: Path) -> Path:
    """Generate a temp path with PID + cryptographic random suffix."""
    return target.with_name(
        f".{target.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    )

def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    """Atomically write a JSON document with fsync+replace.

    Uses a cryptographically-unique temp name and revalidates
    the target directory before replace to mitigate TOCTOU races.
    """
    _reject_link_or_reparse(path.parent, "artifact directory")
    tmp = _unique_tmp_path(path)
    _safe_remove_tmp(tmp)
    try:
        with tmp.open("x", encoding="utf-8", newline="\n") as fh:
            json.dump(value, fh, ensure_ascii=False, indent=2, sort_keys=True)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        _reject_link_or_reparse(path.parent, "artifact directory")
        os.replace(tmp, path)
        _fsync_directory(path.parent)
    finally:
        _safe_remove_tmp(tmp)


def _atomic_npz_hashed(path: Path, arrays: Mapping[str, np.ndarray]) -> str:
    """Atomically write a NumPy .npz file with pre-publish SHA-256 binding.

    Serializes to a bounded in-memory buffer, computes SHA-256 from
    those exact bytes before any filesystem write, then writes via
    exclusive temp + fsync + replace.  After replace, verifies that
    the final path contents match the pre-computed hash.

    Returns the hex SHA-256 digest of the serialized bytes.
    Raises FusionError on size violation or post-replace hash mismatch.
    """
    # Serialize to bounded in-memory buffer
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    npz_bytes = buf.getvalue()

    if len(npz_bytes) > _MAX_NPZ_BYTES:
        raise FusionError(
            f"serialized weights.npz ({len(npz_bytes)} bytes) exceeds "
            f"maximum size ({_MAX_NPZ_BYTES} bytes)"
        )

    # Hash the exact serialized bytes before any filesystem write
    pre_hash = hashlib.sha256(npz_bytes).hexdigest()

    # Atomic write: unique temp -> exclusive open -> write -> fsync -> replace
    _reject_link_or_reparse(path.parent, "artifact directory")
    tmp = _unique_tmp_path(path)
    _safe_remove_tmp(tmp)
    try:
        with tmp.open("xb") as fh:
            fh.write(npz_bytes)
            fh.flush()
            os.fsync(fh.fileno())
        _reject_link_or_reparse(path.parent, "artifact directory")
        os.replace(tmp, path)
        _fsync_directory(path.parent)
    finally:
        _safe_remove_tmp(tmp)

    # Post-replace verification: read back and check against pre-hash
    _reject_link_or_reparse(path, "weights.npz")
    verify_bytes = _safe_read_file(
        path, "weights.npz (post-write verify)", _MAX_NPZ_BYTES
    )
    post_hash = hashlib.sha256(verify_bytes).hexdigest()

    if post_hash != pre_hash:
        _safe_remove_tmp(path)
        raise FusionError(
            "weights.npz post-write SHA-256 mismatch: file contents were "
            "substituted between atomic write and verification"
        )

    return pre_hash


def _as_f64(arr: object, label: str) -> np.ndarray:
    """Cast to float64; reject integer/bool dtypes and non-finite values."""
    a = np.asarray(arr)
    if a.dtype.kind not in ("f",):
        raise FusionError(
            f"{label}: expected floating-point dtype, got {a.dtype}"
        )
    out = a.astype(np.float64, copy=False)
    if not np.all(np.isfinite(out)):
        raise FusionError(f"{label}: contains non-finite values (NaN or inf)")
    return out


def _require_nonneg(arr: np.ndarray, label: str) -> None:
    if np.any(arr < 0.0):
        raise FusionError(f"{label}: must be non-negative, found {arr.min():.6g}")


def _check_weight_bounds(arr: np.ndarray, label: str) -> None:
    """Reject impossibly huge weights that would break stable arithmetic."""
    mx = float(np.max(np.abs(arr)))
    if mx > _MAX_WEIGHT_MAGNITUDE:
        raise FusionError(
            f"{label}: weight magnitude {mx:.2e} exceeds limit {_MAX_WEIGHT_MAGNITUDE:.0e}"
        )


def _validate_track_indices(
    track: object, n_sections: int, attr_name: str, n_windows: int
) -> Optional[np.ndarray]:
    """Validate optional index array on a track; return it or None."""
    idx = getattr(track, attr_name, None)
    if idx is None:
        return None
    arr = np.asarray(idx)
    if arr.ndim != 1:
        raise FusionError(f"{attr_name}: must be a 1-D array, got ndim={arr.ndim}")
    if arr.dtype.kind not in ("i", "u"):
        raise FusionError(f"{attr_name}: must be integer dtype, got {arr.dtype}")
    if len(arr) != n_sections:
        raise FusionError(
            f"{attr_name}: length {len(arr)} != section count {n_sections}"
        )
    if len(arr) > 0:
        if int(arr.min()) < 0:
            raise FusionError(f"{attr_name}: indices must be >= 0")
        if int(arr.max()) >= n_windows:
            raise FusionError(
                f"{attr_name}: index {int(arr.max())} >= window count {n_windows}"
            )
    return arr


def _temporal_position_sim(
    idx_a: Optional[np.ndarray], n_a: int,
    idx_b: Optional[np.ndarray], n_b: int,
) -> float:
    """Compute [0,1] temporal-position similarity between two index sets.

    Returns 0.5 (neutral) when either index set is absent or empty.
    Higher values indicate that sections occur at similar relative
    temporal positions in both tracks.
    """
    if idx_a is None or idx_b is None or len(idx_a) == 0 or len(idx_b) == 0:
        return 0.5
    norm_a = np.sort(np.asarray(idx_a, dtype=np.float64)) / max(n_a - 1, 1)
    norm_b = np.sort(np.asarray(idx_b, dtype=np.float64)) / max(n_b - 1, 1)
    k = min(len(norm_a), len(norm_b))
    mad = float(np.mean(np.abs(norm_a[:k] - norm_b[:k])))
    return float(np.clip(1.0 - mad, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Public feature extraction
# ---------------------------------------------------------------------------


def extract_pair_features(
    track_a: object,
    track_b: object,
    *,
    maxsim_budget: int = _DEFAULT_MAXSIM_BUDGET,
    top_k: int = _DEFAULT_TOP_K,
    coverage_threshold: float = _DEFAULT_COVERAGE_THRESHOLD,
) -> "PairFeatures":
    """Extract the 16-dimensional bounded pair-feature vector.

    Parameters
    ----------
    track_a, track_b :
        Duck-typed objects exposing ``global_embedding`` (1-D),
        ``window_embeddings`` (2-D, >=1 row), ``repeated_sections`` (2-D),
        ``salient_sections`` (2-D).  Optional ``repeated_indices`` and
        ``salient_indices`` (1-D int) are validated and used for
        temporal-position similarity features.
    maxsim_budget : int
        Number of uniformly-spaced budget windows.
    top_k : int
        Number of top per-window max-cosines for top-k features.
    coverage_threshold : float
        Post-(1+cos)/2 mapped cosine threshold in [0,1].

    Returns
    -------
    PairFeatures
        All values deterministic, finite, clipped to [0, 1].
    """
    if isinstance(maxsim_budget, bool) or not isinstance(maxsim_budget, int):
        raise FusionError("maxsim_budget must be an integer (not bool)")
    if maxsim_budget <= 0 or maxsim_budget > _MAX_MAXSIM_BUDGET:
        raise FusionError(
            f"maxsim_budget must be a positive integer <= {_MAX_MAXSIM_BUDGET}"
        )
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise FusionError("top_k must be an integer (not bool)")
    if top_k <= 0 or top_k > _MAX_TOP_K:
        raise FusionError(
            f"top_k must be a positive integer <= {_MAX_TOP_K}"
        )
    if isinstance(coverage_threshold, bool) or not isinstance(coverage_threshold, (int, float)):
        raise FusionError("coverage_threshold must be a real number (not bool)")
    ct_val = float(coverage_threshold)
    if not np.isfinite(ct_val) or not (0.0 <= ct_val <= 1.0):
        raise FusionError("coverage_threshold must be a finite value in [0, 1]")

    # ---- global embeddings ------------------------------------------------
    g_a = _normalize_l2(
        np.asarray(track_a.global_embedding, dtype=np.float64).reshape(-1)
    )
    g_b = _normalize_l2(
        np.asarray(track_b.global_embedding, dtype=np.float64).reshape(-1)
    )
    dim_a = len(g_a)
    dim_b = len(g_b)
    if dim_a != dim_b:
        raise FusionError(
            f"global embedding dims differ: {dim_a} vs {dim_b}"
        )
    global_cos = float(np.clip(np.dot(g_a, g_b), -1.0, 1.0))
    global_cosine_01 = (1.0 + global_cos) / 2.0

    # ---- budget windows ---------------------------------------------------
    win_a = _normalize_rows(
        np.asarray(track_a.window_embeddings, dtype=np.float64)
    )
    win_b = _normalize_rows(
        np.asarray(track_b.window_embeddings, dtype=np.float64)
    )
    if win_a.shape[1] != dim_a:
        raise FusionError("window_embeddings dim mismatch with global_embedding (track_a)")
    if win_b.shape[1] != dim_b:
        raise FusionError("window_embeddings dim mismatch with global_embedding (track_b)")
    n_win_a = len(win_a)
    n_win_b = len(win_b)
    bud_a = win_a[_fixed_budget_indices(n_win_a, maxsim_budget)]
    bud_b = win_b[_fixed_budget_indices(n_win_b, maxsim_budget)]

    # ---- uniform MaxSim ---------------------------------------------------
    sim_u = bud_a @ bud_b.T
    fwd_raw = float(np.mean(np.max(sim_u, axis=1)))
    bwd_raw = float(np.mean(np.max(sim_u, axis=0)))
    uni_ab_01 = (1.0 + fwd_raw) / 2.0
    uni_ba_01 = (1.0 + bwd_raw) / 2.0
    uni_sym_01 = (uni_ab_01 + uni_ba_01) / 2.0

    # ---- top-k MaxSim (uses top_k) ---------------------------------------
    max_cos_ab = np.max(sim_u, axis=1)  # (B,) best match per A window
    max_cos_ba = np.max(sim_u, axis=0)  # (B,) best match per B window
    k = min(top_k, maxsim_budget)
    topk_ab_vals = np.sort(max_cos_ab)[-k:]
    topk_ba_vals = np.sort(max_cos_ba)[-k:]
    topk_ab_01 = (1.0 + float(np.mean(topk_ab_vals))) / 2.0
    topk_ba_01 = (1.0 + float(np.mean(topk_ba_vals))) / 2.0

    # ---- section MaxSim helpers -------------------------------------------
    def _section_sym_01(sa: object, sb: object) -> float:
        ra = np.asarray(sa, dtype=np.float64)
        rb = np.asarray(sb, dtype=np.float64)
        if ra.ndim != 2 or rb.ndim != 2:
            raise FusionError("section embeddings must be 2-D arrays")
        if len(ra) == 0 or len(rb) == 0:
            return 0.5  # neutral fallback for empty section arrays
        if not np.all(np.isfinite(ra)):
            raise FusionError("section embeddings contain non-finite values")
        if not np.all(np.isfinite(rb)):
            raise FusionError("section embeddings contain non-finite values")
        ra = _normalize_rows(ra)
        rb = _normalize_rows(rb)
        if ra.shape[1] != dim_a:
            raise FusionError("repeated/salient section dim mismatch with global")
        if rb.shape[1] != dim_b:
            raise FusionError("repeated/salient section dim mismatch with global")
        sim = ra @ rb.T
        fwd = float(np.mean(np.max(sim, axis=1)))
        bwd = float(np.mean(np.max(sim, axis=0)))
        return float(np.clip((1.0 + (fwd + bwd) / 2.0) / 2.0, 0.0, 1.0))

    repeated_sym_01 = _section_sym_01(
        track_a.repeated_sections, track_b.repeated_sections
    )
    salient_sym_01 = _section_sym_01(
        track_a.salient_sections, track_b.salient_sections
    )

    # ---- coverage (reuses uniform sim matrix) -----------------------------
    max_01_a = (1.0 + np.max(sim_u, axis=1)) / 2.0
    max_01_b = (1.0 + np.max(sim_u, axis=0)) / 2.0
    coverage_a = float(np.mean(max_01_a >= coverage_threshold))
    coverage_b = float(np.mean(max_01_b >= coverage_threshold))

    # ---- asymmetry utility ------------------------------------------------
    asymmetry_utility = float(np.clip(1.0 - abs(uni_ab_01 - uni_ba_01), 0.0, 1.0))

    # ---- recurrence indicator ---------------------------------------------
    def _safe_recur(sections: object) -> float:
        s = np.asarray(sections, dtype=np.float64)
        if s.ndim != 2:
            raise FusionError("section embeddings must be 2-D arrays")
        if len(s) < 2:
            return 0.5
        if not np.all(np.isfinite(s)):
            raise FusionError("repeated section embeddings contain non-finite values")
        s = _normalize_rows(s)
        return _self_recurrence(s)

    recurrence_indicator = float(
        (_safe_recur(track_a.repeated_sections)
         + _safe_recur(track_b.repeated_sections)) / 2.0
    )

    # ---- steady texture ---------------------------------------------------
    cos_a = bud_a @ g_a
    cos_b = bud_b @ g_b
    steady_a = float(np.clip(1.0 - float(np.std(cos_a)), 0.0, 1.0))
    steady_b = float(np.clip(1.0 - float(np.std(cos_b)), 0.0, 1.0))

    # ---- validate and use indices -----------------------------------------
    rep_a_raw = np.asarray(track_a.repeated_sections, dtype=np.float64)
    rep_b_raw = np.asarray(track_b.repeated_sections, dtype=np.float64)
    sal_a_raw = np.asarray(track_a.salient_sections, dtype=np.float64)
    sal_b_raw = np.asarray(track_b.salient_sections, dtype=np.float64)
    n_rep_a = len(rep_a_raw) if rep_a_raw.ndim == 2 else 0
    n_rep_b = len(rep_b_raw) if rep_b_raw.ndim == 2 else 0
    n_sal_a = len(sal_a_raw) if sal_a_raw.ndim == 2 else 0
    n_sal_b = len(sal_b_raw) if sal_b_raw.ndim == 2 else 0
    rep_idx_a = _validate_track_indices(track_a, n_rep_a, "repeated_indices", n_win_a)
    rep_idx_b = _validate_track_indices(track_b, n_rep_b, "repeated_indices", n_win_b)
    sal_idx_a = _validate_track_indices(track_a, n_sal_a, "salient_indices", n_win_a)
    sal_idx_b = _validate_track_indices(track_b, n_sal_b, "salient_indices", n_win_b)

    repeated_temporal = _temporal_position_sim(rep_idx_a, n_win_a, rep_idx_b, n_win_b)
    salient_temporal = _temporal_position_sim(sal_idx_a, n_win_a, sal_idx_b, n_win_b)

    pf = PairFeatures(
        global_cosine=float(np.clip(global_cosine_01, 0.0, 1.0)),
        uniform_maxsim_sym=float(np.clip(uni_sym_01, 0.0, 1.0)),
        uniform_maxsim_ab=float(np.clip(uni_ab_01, 0.0, 1.0)),
        uniform_maxsim_ba=float(np.clip(uni_ba_01, 0.0, 1.0)),
        repeated_maxsim_sym=repeated_sym_01,
        salient_maxsim_sym=salient_sym_01,
        coverage_topk_a=float(np.clip(coverage_a, 0.0, 1.0)),
        coverage_topk_b=float(np.clip(coverage_b, 0.0, 1.0)),
        asymmetry_utility=asymmetry_utility,
        recurrence_indicator=float(np.clip(recurrence_indicator, 0.0, 1.0)),
        steady_texture_a=steady_a,
        steady_texture_b=steady_b,
        topk_maxsim_ab=float(np.clip(topk_ab_01, 0.0, 1.0)),
        topk_maxsim_ba=float(np.clip(topk_ba_01, 0.0, 1.0)),
        repeated_temporal_sim=float(np.clip(repeated_temporal, 0.0, 1.0)),
        salient_temporal_sim=float(np.clip(salient_temporal, 0.0, 1.0)),
    )
    # Final finite-output guard
    vec = pf.to_vector()
    if not np.all(np.isfinite(vec)):
        raise FusionError("extract_pair_features produced non-finite output")
    return pf


# ---------------------------------------------------------------------------
# Weight validation
# ---------------------------------------------------------------------------


def _expected_npz_keys(kind: str, hidden_dims: Tuple[int, ...]) -> frozenset:
    """Return the exact set of expected NPZ array keys for the given kind."""
    if kind == "nonnegative_linear":
        return frozenset({"weights"})
    if kind == "channel_gated_embedding":
        return frozenset({"gates"})
    if kind == "monotonic_network":
        n = len(hidden_dims) + 1
        keys: set = set()
        for i in range(n):
            keys.add(f"l{i}_weight")
            keys.add(f"l{i}_bias")
        return frozenset(keys)
    raise FusionError(f"unknown kind {kind!r}")


def _expected_layer_shapes(
    feature_dim: int, hidden_dims: Tuple[int, ...]
) -> List[Tuple[Tuple[int, int], Tuple[int, ...]]]:
    dims = (feature_dim,) + tuple(hidden_dims) + (1,)
    return [
        ((dims[i + 1], dims[i]), (dims[i + 1],))
        for i in range(len(dims) - 1)
    ]


def _validate_linear_weights(weights: object) -> np.ndarray:
    w = _as_f64(weights, "weights")
    if w.ndim != 1 or w.shape[0] != FEATURE_DIM:
        raise FusionError(
            f"nonnegative_linear weights must have shape ({FEATURE_DIM},), got {w.shape}"
        )
    _require_nonneg(w, "weights")
    _check_weight_bounds(w, "weights")
    if float(w.sum()) < 1e-12:
        raise FusionError("nonnegative_linear weights must have positive sum")
    return _freeze_array(w)


def _validate_network_weights(
    raw: Mapping[str, object],
    hidden_dims: Tuple[int, ...],
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    shapes = _expected_layer_shapes(FEATURE_DIM, hidden_dims)
    layer_w: List[np.ndarray] = []
    layer_b: List[np.ndarray] = []
    for i, (wshape, bshape) in enumerate(shapes):
        w = _as_f64(raw[f"l{i}_weight"], f"l{i}_weight")
        b = _as_f64(raw[f"l{i}_bias"], f"l{i}_bias")
        if w.shape != wshape:
            raise FusionError(f"l{i}_weight: expected shape {wshape}, got {w.shape}")
        if b.shape != bshape:
            raise FusionError(f"l{i}_bias: expected shape {bshape}, got {b.shape}")
        _require_nonneg(w, f"l{i}_weight")
        _check_weight_bounds(w, f"l{i}_weight")
        _check_weight_bounds(b, f"l{i}_bias")
        layer_w.append(_freeze_array(w))
        layer_b.append(_freeze_array(b))
    return layer_w, layer_b


def _validate_channel_gated_weights(gates: object) -> np.ndarray:
    g = _as_f64(gates, "gates")
    if g.ndim != 1 or g.shape[0] != _CHANNEL_COUNT:
        raise FusionError(
            f"gates must have shape ({_CHANNEL_COUNT},), got {g.shape}"
        )
    _require_nonneg(g, "gates")
    _check_weight_bounds(g, "gates")
    total = float(g.sum())
    if total < 1e-12:
        raise FusionError("gates must have positive sum (cannot be all-zero)")
    if abs(total - 1.0) > 1e-4:
        raise FusionError(
            f"gates must sum to 1.0 (got {total:.6f}); normalise before saving"
        )
    return _freeze_array(g)


# ---------------------------------------------------------------------------
# Internal scoring helpers
# ---------------------------------------------------------------------------


def _apply_feature_mask(features: np.ndarray, ablation: str) -> np.ndarray:
    if ablation not in ABLATIONS:
        raise FusionError(f"unknown ablation {ablation!r}; must be one of {ABLATIONS}")
    return features * np.array(_FEATURE_MASK[ablation], dtype=np.float64)


def _score_nonneg_linear(
    weights: np.ndarray,
    features: np.ndarray,
    ablation: str,
) -> float:
    mask = np.array(_FEATURE_MASK[ablation], dtype=np.float64)
    w = weights * mask
    total = float(w.sum())
    if total < 1e-12:
        return 0.5
    raw = float(np.dot(w, features * mask)) / total
    out = float(np.clip(raw, 0.0, 1.0))
    if not np.isfinite(out):
        raise FusionError("nonneg_linear scoring produced non-finite output")
    return out


def _score_monotonic_network(
    layer_weights: List[np.ndarray],
    layer_biases: List[np.ndarray],
    features: np.ndarray,
    ablation: str,
) -> float:
    x = _apply_feature_mask(features, ablation)
    n = len(layer_weights)
    for i, (W, b) in enumerate(zip(layer_weights, layer_biases)):
        x = W @ x + b
        if i < n - 1:
            x = np.maximum(x, 0.0)
    val = float(np.clip(x.ravel()[0], -500.0, 500.0))
    out = 1.0 / (1.0 + float(np.exp(-val)))
    out = float(np.clip(out, 0.0, 1.0))
    if not np.isfinite(out):
        raise FusionError("monotonic_network scoring produced non-finite output")
    return out


def _channel_summaries(
    track: object, budget: int, embedding_dim: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (global, uniform, repeated, salient) L2-normalised channel vectors."""
    g = _normalize_l2(
        np.asarray(track.global_embedding, dtype=np.float64).reshape(-1)
    )
    if len(g) != embedding_dim:
        raise FusionError(
            f"global_embedding dim {len(g)} != config.embedding_dim {embedding_dim}"
        )
    win = _normalize_rows(np.asarray(track.window_embeddings, dtype=np.float64))
    if win.shape[1] != embedding_dim:
        raise FusionError(
            f"window_embeddings dim {win.shape[1]} != config.embedding_dim {embedding_dim}"
        )
    bud = win[_fixed_budget_indices(len(win), budget)]
    u_raw = np.mean(bud, axis=0)
    u = _normalize_l2(u_raw) if float(np.linalg.norm(u_raw)) > 1e-12 else g.copy()

    rep = np.asarray(track.repeated_sections, dtype=np.float64)
    if rep.ndim == 2 and len(rep) > 0:
        if rep.shape[1] != embedding_dim:
            raise FusionError(
                f"repeated_sections dim {rep.shape[1]} != config.embedding_dim {embedding_dim}"
            )
        if not np.all(np.isfinite(rep)):
            raise FusionError("repeated_sections contain non-finite values")
        r_raw = np.mean(_normalize_rows(rep), axis=0)
        r = _normalize_l2(r_raw) if float(np.linalg.norm(r_raw)) > 1e-12 else g.copy()
    else:
        r = g.copy()

    sal = np.asarray(track.salient_sections, dtype=np.float64)
    if sal.ndim == 2 and len(sal) > 0:
        if sal.shape[1] != embedding_dim:
            raise FusionError(
                f"salient_sections dim {sal.shape[1]} != config.embedding_dim {embedding_dim}"
            )
        if not np.all(np.isfinite(sal)):
            raise FusionError("salient_sections contain non-finite values")
        s_raw = np.mean(_normalize_rows(sal), axis=0)
        s = _normalize_l2(s_raw) if float(np.linalg.norm(s_raw)) > 1e-12 else g.copy()
    else:
        s = g.copy()

    return g, u, r, s


def _embed_track_channel_gated(
    gates: np.ndarray,
    track: object,
    budget: int,
    ablation: str,
    embedding_dim: int,
) -> np.ndarray:
    """Produce L2-normalised float64 track embedding via gated channel blend."""
    g, u, r, s = _channel_summaries(track, budget, embedding_dim)
    ch_mask = np.array(_CHANNEL_MASK[ablation], dtype=np.float64)
    eff_gates = gates * ch_mask
    total = float(eff_gates.sum())
    if total < 1e-12:
        return g.copy()
    eff_gates = eff_gates / total
    combined = eff_gates[0] * g + eff_gates[1] * u + eff_gates[2] * r + eff_gates[3] * s
    norm = float(np.linalg.norm(combined))
    if norm <= 1e-12:
        return g.copy()
    result = (combined / norm).astype(np.float64)
    if not np.all(np.isfinite(result)):
        return g.copy()
    return result


# ---------------------------------------------------------------------------
# FusionModel
# ---------------------------------------------------------------------------


class FusionModel:
    """Frozen sealed inference model for full-track fusion scoring.

    Do not construct directly; use build_nonneg_linear,
    build_monotonic_network, build_channel_gated, or load_fusion_artifact.
    """

    def __init__(
        self,
        *,
        config: "FusionConfig",
        weights: Dict[str, object],
        metadata: Optional["FusionMetadata"] = None,
    ) -> None:
        config.validate()
        self._config = config
        self._weights = _deep_copy_weights(weights)
        self._metadata = metadata

    @property
    def config(self) -> "FusionConfig":
        return self._config

    @property
    def metadata(self) -> Optional["FusionMetadata"]:
        return self._metadata

    def extract_pair_features(
        self,
        track_a: object,
        track_b: object,
    ) -> "PairFeatures":
        """Extract PairFeatures using this model's configured parameters.

        Validates that all track embedding widths match config.embedding_dim.
        """
        edim = self._config.embedding_dim
        _validate_track_embedding_dim(track_a, edim, "track_a")
        _validate_track_embedding_dim(track_b, edim, "track_b")
        return extract_pair_features(
            track_a,
            track_b,
            maxsim_budget=self._config.maxsim_budget,
            top_k=self._config.top_k,
            coverage_threshold=self._config.coverage_threshold,
        )

    def embed_track(
        self,
        track: object,
        *,
        ablation: str = "none",
    ) -> np.ndarray:
        """Return L2-normalised float64 track embedding.

        Only valid for channel_gated_embedding kind.
        """
        if self._config.kind != "channel_gated_embedding":
            raise FusionError(
                "embed_track is only available for channel_gated_embedding models"
            )
        if ablation not in ABLATIONS:
            raise FusionError(f"unknown ablation {ablation!r}")
        return _embed_track_channel_gated(
            self._weights["gates"], track, self._config.maxsim_budget,
            ablation, self._config.embedding_dim,
        )

    def score_candidate(
        self,
        query: object,
        candidate: object,
        *,
        ablation: str = "none",
    ) -> float:
        """Score a single candidate against a query.  Returns finite [0, 1]."""
        if ablation not in ABLATIONS:
            raise FusionError(f"unknown ablation {ablation!r}")
        kind = self._config.kind

        if kind == "channel_gated_embedding":
            gates = self._weights["gates"]
            emb_q = _embed_track_channel_gated(
                gates, query, self._config.maxsim_budget,
                ablation, self._config.embedding_dim,
            )
            emb_c = _embed_track_channel_gated(
                gates, candidate, self._config.maxsim_budget,
                ablation, self._config.embedding_dim,
            )
            cos = float(np.clip(np.dot(emb_q, emb_c), -1.0, 1.0))
            out = float(np.clip((1.0 + cos) / 2.0, 0.0, 1.0))
            if not np.isfinite(out):
                raise FusionError("channel_gated scoring produced non-finite output")
            return out

        feat = self.extract_pair_features(query, candidate).to_vector()

        if kind == "nonnegative_linear":
            return _score_nonneg_linear(self._weights["weights"], feat, ablation)

        if kind == "monotonic_network":
            return _score_monotonic_network(
                self._weights["layer_weights"],
                self._weights["layer_biases"],
                feat,
                ablation,
            )
        raise FusionError(f"unknown kind {kind!r}")

    def score_feature_vectors(
        self,
        features: np.ndarray,
        *,
        ablation: str = "none",
    ) -> np.ndarray:
        """Score canonical pair-feature rows for a feature-based model."""
        if ablation not in ABLATIONS:
            raise FusionError(f"unknown ablation {ablation!r}")
        if self._config.kind == "channel_gated_embedding":
            raise FusionError(
                "channel_gated_embedding does not score pair-feature vectors"
            )
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != FEATURE_DIM:
            raise FusionError(
                f"features must have shape (N, {FEATURE_DIM}), got {matrix.shape}"
            )
        if not np.all(np.isfinite(matrix)):
            raise FusionError("features contain non-finite values")
        if np.any(matrix < 0.0) or np.any(matrix > 1.0):
            raise FusionError("features must be bounded in [0, 1]")

        out = np.empty(len(matrix), dtype=np.float64)
        if self._config.kind == "nonnegative_linear":
            for index, row in enumerate(matrix):
                out[index] = _score_nonneg_linear(
                    self._weights["weights"], row, ablation
                )
            return out
        if self._config.kind == "monotonic_network":
            for index, row in enumerate(matrix):
                out[index] = _score_monotonic_network(
                    self._weights["layer_weights"],
                    self._weights["layer_biases"],
                    row,
                    ablation,
                )
            return out
        raise FusionError(f"unknown kind {self._config.kind!r}")

    def score_candidates(
        self,
        query: object,
        candidates: Sequence[object],
        *,
        ablation: str = "none",
    ) -> np.ndarray:
        """Score multiple candidates.  Returns (N,) float64, all in [0, 1]."""
        if ablation not in ABLATIONS:
            raise FusionError(f"unknown ablation {ablation!r}")
        out = np.empty(len(candidates), dtype=np.float64)
        if self._config.kind == "channel_gated_embedding":
            gates = self._weights["gates"]
            emb_q = _embed_track_channel_gated(
                gates, query, self._config.maxsim_budget,
                ablation, self._config.embedding_dim,
            )
            for i, c in enumerate(candidates):
                emb_c = _embed_track_channel_gated(
                    gates, c, self._config.maxsim_budget,
                    ablation, self._config.embedding_dim,
                )
                cos = float(np.clip(np.dot(emb_q, emb_c), -1.0, 1.0))
                out[i] = float(np.clip((1.0 + cos) / 2.0, 0.0, 1.0))
        else:
            for i, c in enumerate(candidates):
                out[i] = self.score_candidate(query, c, ablation=ablation)
        return out


# ---------------------------------------------------------------------------
# Constructor functions
# ---------------------------------------------------------------------------


def build_nonneg_linear(
    weights: np.ndarray,
    config: "FusionConfig",
) -> "FusionModel":
    """Build a nonnegative_linear FusionModel from a weight vector."""
    if config.kind != "nonnegative_linear":
        raise FusionError(
            f"build_nonneg_linear requires kind=nonnegative_linear, "
            f"got {config.kind!r}"
        )
    w = _validate_linear_weights(weights)
    return FusionModel(config=config, weights={"weights": w})


def build_monotonic_network(
    layer_weights: Sequence[np.ndarray],
    layer_biases: Sequence[np.ndarray],
    config: "FusionConfig",
) -> "FusionModel":
    """Build a monotonic_network FusionModel."""
    if config.kind != "monotonic_network":
        raise FusionError(
            f"build_monotonic_network requires kind=monotonic_network, "
            f"got {config.kind!r}"
        )
    config.validate()
    expected_n = len(config.hidden_dims) + 1
    if len(layer_weights) != expected_n or len(layer_biases) != expected_n:
        raise FusionError(
            f"monotonic_network expects {expected_n} weight+bias pairs "
            f"for hidden_dims={config.hidden_dims}"
        )
    raw: Dict[str, object] = {}
    for i, (w, b) in enumerate(zip(layer_weights, layer_biases)):
        raw[f"l{i}_weight"] = w
        raw[f"l{i}_bias"] = b
    lw, lb = _validate_network_weights(raw, config.hidden_dims)
    return FusionModel(
        config=config, weights={"layer_weights": lw, "layer_biases": lb}
    )


def build_channel_gated(
    gates: np.ndarray,
    config: "FusionConfig",
) -> "FusionModel":
    """Build a channel_gated_embedding FusionModel."""
    if config.kind != "channel_gated_embedding":
        raise FusionError(
            f"build_channel_gated requires kind=channel_gated_embedding, "
            f"got {config.kind!r}"
        )
    g = _validate_channel_gated_weights(gates)
    return FusionModel(config=config, weights={"gates": g})


# ---------------------------------------------------------------------------
# Artifact I/O helpers
# ---------------------------------------------------------------------------


def _collect_arrays(model: "FusionModel") -> Dict[str, np.ndarray]:
    kind = model.config.kind
    if kind == "nonnegative_linear":
        return {"weights": model._weights["weights"].astype(np.float64)}
    if kind == "channel_gated_embedding":
        return {"gates": model._weights["gates"].astype(np.float64)}
    if kind == "monotonic_network":
        out: Dict[str, np.ndarray] = {}
        for i, (w, b) in enumerate(
            zip(model._weights["layer_weights"], model._weights["layer_biases"])
        ):
            out[f"l{i}_weight"] = w.astype(np.float64)
            out[f"l{i}_bias"] = b.astype(np.float64)
        return out
    raise FusionError(f"unknown kind {kind!r}")


def _build_json_payload(
    config: "FusionConfig",
    npz_sha256: str,
    created_at: str,
) -> Dict[str, object]:
    return {
        "schema_version": FUSION_SCHEMA_VERSION,
        "kind": config.kind,
        "model_id": config.model_id,
        "store_id": config.store_id,
        "config_sha256": config.config_sha256,
        "fold_index": config.fold_index,
        "embedding_dim": config.embedding_dim,
        "feature_dim": FEATURE_DIM,
        "feature_names": list(FEATURE_NAMES),
        "maxsim_budget": config.maxsim_budget,
        "top_k": config.top_k,
        "coverage_threshold": float(config.coverage_threshold),
        "seed": config.seed,
        "hidden_dims": list(config.hidden_dims),
        "created_at": created_at,
        "npz_sha256": npz_sha256,
    }


def _strict_int_check(val: object, label: str) -> None:
    """Raise FusionError if val is not int or is bool."""
    if isinstance(val, bool):
        raise FusionError(f"{label}: bool is not accepted as integer")
    if not isinstance(val, int):
        raise FusionError(f"{label} must be an integer")


def _validate_json_schema(raw: object) -> None:
    """Raise FusionError if raw does not conform to the strict JSON schema."""
    if not isinstance(raw, dict):
        raise FusionError("model.json must be a JSON object")
    keys = frozenset(raw.keys())
    missing = _JSON_REQUIRED_FIELDS - keys
    extra = keys - _JSON_REQUIRED_FIELDS
    if missing:
        raise FusionError(f"model.json missing required fields: {sorted(missing)}")
    if extra:
        raise FusionError(f"model.json contains unexpected fields: {sorted(extra)}")

    # --- schema_version ---
    _strict_int_check(raw["schema_version"], "schema_version")
    if raw["schema_version"] != FUSION_SCHEMA_VERSION:
        raise FusionError(
            f"schema_version mismatch: expected {FUSION_SCHEMA_VERSION}, "
            f"got {raw['schema_version']!r}"
        )

    # --- kind ---
    if raw["kind"] not in CANDIDATE_KINDS:
        raise FusionError(f"unknown kind {raw['kind']!r}")

    # --- feature_dim / feature_names ---
    _strict_int_check(raw["feature_dim"], "feature_dim")
    if raw["feature_dim"] != FEATURE_DIM:
        raise FusionError(
            f"feature_dim mismatch: expected {FEATURE_DIM}, got {raw['feature_dim']}"
        )
    if list(raw["feature_names"]) != list(FEATURE_NAMES):
        raise FusionError("feature_names mismatch")

    # --- integer fields (reject bool-as-int) ---
    _strict_int_check(raw["embedding_dim"], "embedding_dim")
    if raw["embedding_dim"] <= 0:
        raise FusionError("embedding_dim must be a positive integer")
    _strict_int_check(raw["fold_index"], "fold_index")
    if raw["fold_index"] < 0:
        raise FusionError("fold_index must be a non-negative integer")
    _strict_int_check(raw["maxsim_budget"], "maxsim_budget")
    if raw["maxsim_budget"] <= 0:
        raise FusionError("maxsim_budget must be a positive integer")
    _strict_int_check(raw["top_k"], "top_k")
    if raw["top_k"] <= 0:
        raise FusionError("top_k must be a positive integer")
    _strict_int_check(raw["seed"], "seed")

    # --- coverage_threshold ---
    ct = raw["coverage_threshold"]
    if isinstance(ct, bool):
        raise FusionError("coverage_threshold: bool is not accepted")
    if not isinstance(ct, (int, float)):
        raise FusionError("coverage_threshold must be numeric")
    ct_f = float(ct)
    if not np.isfinite(ct_f) or not (0.0 <= ct_f <= 1.0):
        raise FusionError("coverage_threshold must be a finite number in [0, 1]")

    # --- string provenance fields ---
    for field in ("model_id", "store_id"):
        if not isinstance(raw[field], str) or not raw[field]:
            raise FusionError(f"{field} must be a nonempty string")

    # --- config_sha256: 64 lowercase hex ---
    cs = raw["config_sha256"]
    if not isinstance(cs, str) or len(cs) != 64:
        raise FusionError("config_sha256 must be a 64-char lowercase hex string")
    if any(c not in "0123456789abcdef" for c in cs):
        raise FusionError("config_sha256 must be lowercase hex")

    # --- created_at: UTC timestamp ---
    ca = raw["created_at"]
    if not isinstance(ca, str) or not _UTC_RE.match(ca):
        raise FusionError("created_at must be a UTC timestamp (YYYY-MM-DDTHH:MM:SSZ)")

    # --- hidden_dims ---
    if not isinstance(raw["hidden_dims"], list):
        raise FusionError("hidden_dims must be a list")
    for dim in raw["hidden_dims"]:
        _strict_int_check(dim, "hidden_dims entry")
        if dim <= 0 or dim > 4096:
            raise FusionError("hidden_dims entries must be positive integers in [1, 4096]")
    if raw["kind"] == "monotonic_network" and not raw["hidden_dims"]:
        raise FusionError("monotonic_network requires at least one hidden dim in JSON")
    if raw["kind"] != "monotonic_network" and raw["hidden_dims"]:
        raise FusionError(f"hidden_dims must be empty for {raw['kind']}")

    # --- SHA-256 fields ---
    for sha_field in ("npz_sha256", "json_payload_sha256"):
        val = raw[sha_field]
        if not isinstance(val, str) or len(val) != 64:
            raise FusionError(f"{sha_field} must be a 64-char lowercase hex string")
        if any(c not in "0123456789abcdef" for c in val):
            raise FusionError(f"{sha_field} must be lowercase hex")


def _validate_npz_zip(data: bytes, expected_npy_names: frozenset) -> None:
    """Pre-flight ZIP inspection of NPZ data.

    Rejects compressed entries, directories, unexpected members,
    and enforces per-member / aggregate uncompressed-size limits.
    """
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise FusionError(f"weights.npz is not a valid ZIP archive: {exc}") from exc

    seen: set = set()
    total_uncompressed: int = 0

    for info in zf.infolist():
        name = info.filename
        if name.endswith("/"):
            raise FusionError(f"weights.npz contains directory entry: {name!r}")
        if not name.endswith(".npy"):
            raise FusionError(f"weights.npz contains non-.npy entry: {name!r}")
        array_name = name[:-4]
        if array_name not in expected_npy_names:
            raise FusionError(f"weights.npz contains unexpected entry: {array_name!r}")
        if array_name in seen:
            raise FusionError(
                f"weights.npz contains duplicate entry: {array_name!r}"
            )
        seen.add(array_name)
        if info.compress_type != zipfile.ZIP_STORED:
            raise FusionError(
                f"weights.npz entry {name!r} uses compression (must be stored/uncompressed)"
            )
        if info.file_size > _MAX_NPZ_BYTES:
            raise FusionError(f"weights.npz entry {name!r} exceeds size limit")
        total_uncompressed += info.file_size

    if total_uncompressed > _MAX_NPZ_BYTES:
        raise FusionError("weights.npz total uncompressed size exceeds limit")

    miss = expected_npy_names - seen
    if miss:
        raise FusionError(f"weights.npz missing arrays: {sorted(miss)}")

    zf.close()


# ---------------------------------------------------------------------------
# Public artifact API
# ---------------------------------------------------------------------------


def _is_valid_sha256_hex(s: str) -> bool:
    return (
        isinstance(s, str)
        and len(s) == 64
        and all(c in "0123456789abcdef" for c in s)
    )


def save_fusion_artifact(
    model: "FusionModel",
    directory: Union[str, Path],
) -> "FusionMetadata":
    """Atomically write a fusion artifact to *directory*.

    Creates model.json and weights.npz with SHA-256 cross-binding.
    Uses cryptographically-unique temp names, revalidates path safety
    at each stage, and refuses to overwrite existing sealed artifacts
    (immutable export) to prevent mixed-version corruption.

    Raises FusionError if model_id, store_id, or config_sha256 are
    missing / malformed, or if the directory already contains
    artifact files.
    """
    model.config.validate()
    cfg = model.config

    # Validate provenance fields required for artifacts
    if not cfg.model_id:
        raise FusionError("model_id must be nonempty for artifact save")
    if not cfg.store_id:
        raise FusionError("store_id must be nonempty for artifact save")
    if not _is_valid_sha256_hex(cfg.config_sha256):
        raise FusionError("config_sha256 must be 64 lowercase hex chars")

    # Path safety: check unresolved path + parents before resolve
    directory = _check_path_safety(directory, "artifact directory")
    directory.mkdir(parents=True, exist_ok=True)

    # Revalidate directory after mkdir
    _reject_link_or_reparse(directory, "artifact directory")

    json_path = directory / "model.json"
    npz_path = directory / "weights.npz"

    # Immutable export: refuse to overwrite existing sealed artifacts.
    # Prevents mixed-version corruption where one file is from version A
    # and the other from version B.
    for p, lbl in ((json_path, "model.json"), (npz_path, "weights.npz")):
        if p.is_symlink():
            raise FusionError(f"{lbl} may not be a symlink or junction: {p}")
        if _is_reparse_point(p):
            raise FusionError(f"{lbl} may not be a reparse point/junction: {p}")
        if p.exists():
            raise FusionError(
                f"refusing to overwrite sealed artifact: "
                f"{lbl} already exists at {directory}"
            )

    # Step 1: serialize, write, and hash arrays -> NPZ (atomic, pre-publish)
    arrays = _collect_arrays(model)
    npz_sha256 = _atomic_npz_hashed(npz_path, arrays)
    _reject_link_or_reparse(npz_path, "weights.npz")

    # Step 2: build payload
    created_at = datetime.datetime.now(
        datetime.timezone.utc
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    payload = _build_json_payload(cfg, npz_sha256, created_at)

    # Step 3: sha256 of canonical payload
    json_payload_sha256 = stable_json_sha256(payload)

    # Step 4: full document; write atomically
    doc = dict(payload)
    doc["json_payload_sha256"] = json_payload_sha256
    _atomic_json(json_path, doc)
    _reject_link_or_reparse(json_path, "model.json")

    return FusionMetadata(
        kind=cfg.kind,
        model_id=cfg.model_id,
        store_id=cfg.store_id,
        config_sha256=cfg.config_sha256,
        fold_index=cfg.fold_index,
        embedding_dim=cfg.embedding_dim,
        feature_dim=FEATURE_DIM,
        maxsim_budget=cfg.maxsim_budget,
        top_k=cfg.top_k,
        coverage_threshold=float(cfg.coverage_threshold),
        seed=cfg.seed,
        hidden_dims=tuple(cfg.hidden_dims),
        json_payload_sha256=json_payload_sha256,
        npz_sha256=npz_sha256,
        created_at=created_at,
    )


def load_fusion_artifact(
    directory: Union[str, Path],
) -> "FusionModel":
    """Load and fully validate a fusion artifact from *directory*.

    Reads JSON and NPZ into bounded byte snapshots via safely opened
    handles with lstat/fstat identity verification and O_NOFOLLOW
    (where available).  Validates symlink/junction/reparse-point
    protection, strict JSON schema, SHA-256 cross-bindings, ZIP
    preflight (no compression, exact members, no duplicates), and
    weight dtype/shape/finite/nonneg checks.  Deep-copies and
    freezes all weight arrays.

    Not race-free; provides practical TOCTOU mitigation.
    """
    directory = _check_path_safety(directory, "artifact directory")

    json_path = directory / "model.json"
    npz_path = directory / "weights.npz"

    for path, label in ((json_path, "model.json"), (npz_path, "weights.npz")):
        _reject_link_or_reparse(path, label)
        if not path.is_file():
            raise FusionError(f"{label} not found in {directory}")

    # ---- read JSON into bounded byte snapshot with safety -----------------
    json_bytes = _safe_read_file(json_path, "model.json", _MAX_JSON_BYTES)

    try:
        raw = json.loads(json_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise FusionError(f"model.json is not valid JSON: {exc}") from exc

    _validate_json_schema(raw)

    stored_json_sha256 = str(raw["json_payload_sha256"])
    stored_npz_sha256 = str(raw["npz_sha256"])

    # ---- verify JSON payload SHA-256 -------------------------------------
    payload = {k: v for k, v in raw.items() if k != "json_payload_sha256"}
    computed_json_sha256 = stable_json_sha256(payload)
    if computed_json_sha256 != stored_json_sha256:
        raise FusionError(
            "model.json payload SHA-256 mismatch -- file may have been tampered with"
        )

    # ---- read NPZ into bounded byte snapshot with safety ------------------
    npz_bytes = _safe_read_file(npz_path, "weights.npz", _MAX_NPZ_BYTES)

    # Hash the pinned byte snapshot
    computed_npz_sha256 = hashlib.sha256(npz_bytes).hexdigest()
    if computed_npz_sha256 != stored_npz_sha256:
        raise FusionError(
            "weights.npz SHA-256 mismatch -- file may have been tampered with"
        )

    # ---- reconstruct config ----------------------------------------------
    hidden_dims = tuple(int(d) for d in raw["hidden_dims"])
    config = FusionConfig(
        kind=str(raw["kind"]),
        embedding_dim=int(raw["embedding_dim"]),
        maxsim_budget=int(raw["maxsim_budget"]),
        top_k=int(raw["top_k"]),
        coverage_threshold=float(raw["coverage_threshold"]),
        seed=int(raw["seed"]),
        model_id=str(raw["model_id"]),
        store_id=str(raw["store_id"]),
        config_sha256=str(raw["config_sha256"]),
        fold_index=int(raw["fold_index"]),
        hidden_dims=hidden_dims,
    )
    config.validate()

    # ---- ZIP preflight + load arrays from same bytes ---------------------
    expected_keys = _expected_npz_keys(config.kind, hidden_dims)
    _validate_npz_zip(npz_bytes, expected_keys)

    try:
        with np.load(io.BytesIO(npz_bytes), allow_pickle=False) as npz:
            actual_keys = frozenset(npz.files)
            miss = expected_keys - actual_keys
            extra = actual_keys - expected_keys
            if miss:
                raise FusionError(f"weights.npz missing arrays: {sorted(miss)}")
            if extra:
                raise FusionError(f"weights.npz has unexpected arrays: {sorted(extra)}")
            raw_arrays = {k: npz[k].copy() for k in expected_keys}
    except FusionError:
        raise
    except Exception as exc:
        raise FusionError(f"cannot parse weights.npz: {exc}") from exc

    if config.kind == "nonnegative_linear":
        w = _validate_linear_weights(raw_arrays["weights"])
        weights_store: Dict[str, object] = {"weights": w}
    elif config.kind == "monotonic_network":
        lw, lb = _validate_network_weights(raw_arrays, hidden_dims)
        weights_store = {"layer_weights": lw, "layer_biases": lb}
    else:
        g = _validate_channel_gated_weights(raw_arrays["gates"])
        weights_store = {"gates": g}

    metadata = FusionMetadata(
        kind=config.kind,
        model_id=config.model_id,
        store_id=config.store_id,
        config_sha256=config.config_sha256,
        fold_index=config.fold_index,
        embedding_dim=config.embedding_dim,
        feature_dim=FEATURE_DIM,
        maxsim_budget=config.maxsim_budget,
        top_k=config.top_k,
        coverage_threshold=float(config.coverage_threshold),
        seed=config.seed,
        hidden_dims=hidden_dims,
        json_payload_sha256=stored_json_sha256,
        npz_sha256=stored_npz_sha256,
        created_at=str(raw["created_at"]),
    )

    return FusionModel(config=config, weights=weights_store, metadata=metadata)
