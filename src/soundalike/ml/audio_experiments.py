"""Iteration-4 audio-only experiment pipeline (Protocol v5 FROZEN 2026-07-12).

PROTOCOL CONSTRAINTS (never violate):
  * Never evaluate / read final-split target outcomes or frozen baseline ranks.
  * Prohibited model inputs: song titles, artist names, popularity, Wikipedia /
    pageview data, manual similarity pairs, ListenBrainz / Deezer similarity for
    benchmark artists.  Metadata is used ONLY for leakage exclusion (training)
    and output filtering (evaluation).
  * Every benchmark artist in BOTH development and final splits is excluded from
    distillation training rows.

Components:
  1. FMA genre-supervised cross-artist supervised-contrastive training on
     ml_data/fma_packed.npz.
  2. BYOL-style non-contrastive self-supervised FMA training.
  3. Audio-teacher distillation from production sonic64 + CLAP512 arrays using
     production cached mels, with benchmark-artist leakage exclusion.
  4. Full 272,853-catalog embedding extraction to float16.
  5. Development-only evaluation on benchmarks/soundalike_pairs.v5.json with
     R@1/5/10/20/50, MRR, NDCG@10/50, primary = mean(NDCG@10, MRR, R@10),
     exact per-pair ranks, candidate recall @50/200/1000.
  6. Optional 4-window late-interaction reranking (MaxSim) over global
     candidates.
  7. Fully reproducible ExperimentConfig / seeds / CheckpointMeta /
     ResourceLog.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import unicodedata
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

_V5_BENCHMARK_ID = "soundalike-pure-sonic-v5"
_V5_SCHEMA_VERSION = 5
_DEFAULT_RECALL_CUTOFFS: Tuple[int, ...] = (1, 5, 10, 20, 50)
_DEFAULT_NDCG_CUTOFFS: Tuple[int, ...] = (10, 50)
_DEFAULT_CANDIDATE_RECALL_AT: Tuple[int, ...] = (50, 200, 1000)

# ─────────────────────────────────────────────────────────────────────────────
# 1. UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────


def _normalize(value: str) -> str:
    """Accent-/punctuation-insensitive comparison key (mirrors real_benchmark)."""
    value = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    value = value.casefold()
    value = re.sub(r"[\(\[][^\)\]]*[\)\]]", " ", value)
    value = re.sub(
        r"\s+-\s+(?:\d{4}\s+)?(?:re)?master(?:ed)?(?:\s+\d{4})?.*$", "", value
    )
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value).split())


def _primary_artist(value: str) -> str:
    """Return the primary-credit name (before featuring / feat / ft / & / x)."""
    value = _normalize(value)
    for sep in (" featuring ", " feat ", " ft ", " x ", " and ", " & ", ", "):
        if sep in value:
            value = value.split(sep, 1)[0]
    return value.strip()


def _credited_artists(raw: str) -> Set[str]:
    """Conservative set of normalized artist tokens (for split leakage audits)."""
    raw = unicodedata.normalize("NFKD", str(raw)).encode("ascii", "ignore").decode()
    raw = re.sub(
        r"\s*(?:,|&|\+|\bx\b|\bwith\b|\band\b|\bfeaturing\b|\bfeat\.?\b|\bft\.?\b)\s*",
        ",",
        raw.casefold(),
    )
    return {_normalize(p) for p in raw.split(",") if len(_normalize(p)) > 1}


def _best_device() -> str:
    """Return 'cuda' if a CUDA device is available, else 'cpu'."""
    if _TORCH_AVAILABLE and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_teacher(sonic: np.ndarray, clap: np.ndarray) -> np.ndarray:
    """Concatenate sonic and clap arrays and L2-normalise row-wise.

    Returns (N, sonic_dim + clap_dim) float32.
    """
    combined = np.concatenate(
        [sonic.astype(np.float32), clap.astype(np.float32)], axis=1
    )
    norms = np.linalg.norm(combined, axis=1, keepdims=True)
    return (combined / np.clip(norms, 1e-9, None)).astype(np.float32)


def _stratified_split(
    genres: np.ndarray,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """Stratified train/val/test split by genre.

    Unlabeled tracks are routed to train only (no labels to evaluate).
    Returns a dict mapping split name to an array of integer indices.
    """
    rng = np.random.default_rng(seed)
    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    for g in np.unique(genres):
        idx = np.where(genres == g)[0].tolist()
        rng.shuffle(idx)
        if str(g) == "unlabeled":
            train_idx += idx
            continue
        n = len(idx)
        nt = int(n * test_frac)
        nv = int(n * val_frac)
        test_idx += idx[:nt]
        val_idx += idx[nt : nt + nv]
        train_idx += idx[nt + nv :]
    return {
        "train": np.array(train_idx, dtype=np.int64),
        "val": np.array(val_idx, dtype=np.int64),
        "test": np.array(test_idx, dtype=np.int64),
    }


def _write_json_atomic(path: Path, value: Any) -> None:
    """Write JSON atomically using a temp file + rename (Windows-safe)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.json")
    try:
        tmp.write_text(
            json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if path.exists():
            path.unlink()
        tmp.rename(path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 2. CONFIGURATION AND METADATA
# ─────────────────────────────────────────────────────────────────────────────


from .quality_filter import TitleQualityFilter

@dataclass(frozen=True)
class ExperimentConfig:
    """Fully reproducible configuration for the iteration-4 audio pipeline.

    All paths are relative to the project root (or absolute).
    Seeds are fixed and never derived from target labels or benchmark ranks.
    This config is frozen; to run a variant, create a new instance.
    """

    # ── Protocol identity ────────────────────────────────────────────────────
    protocol_version: str = "v5"
    benchmark_id: str = _V5_BENCHMARK_ID

    # ── File paths ────────────────────────────────────────────────────────────
    fma_packed_path: str = "ml_data/fma_packed.npz"
    catalog_index_path: str = "ml_data/deepvibe_index_v5.npz"
    benchmark_path: str = "benchmarks/soundalike_pairs.v5.json"
    sonic_array_path: str = "ml_data/research_clap64_cal.f16.npy"
    clap_array_path: str = "ml_data/research_clap512.f16.npy"
    output_dir: str = "ml_data/audio_experiments"

    # ── FMA supervised-contrastive (phase 1) ─────────────────────────────────
    supcon_embedding_dim: int = 256
    supcon_encoder_width: int = 64
    supcon_pool_type: str = "gem"
    supcon_batch_size: int = 128
    supcon_crop_frames: int = 256
    supcon_lr: float = 1e-3
    supcon_weight_decay: float = 1e-4
    supcon_max_epochs: int = 100
    supcon_temperature: float = 0.1
    supcon_cross_artist: bool = True  # exclude same-artist from positives

    # ── BYOL (phase 2) ────────────────────────────────────────────────────────
    byol_embedding_dim: int = 256
    byol_encoder_width: int = 64
    byol_pool_type: str = "gem"
    byol_projection_dim: int = 256
    byol_prediction_dim: int = 128
    byol_ema_decay: float = 0.996
    byol_lr: float = 3e-4
    byol_weight_decay: float = 1e-4
    byol_batch_size: int = 128
    byol_crop_frames: int = 256
    byol_max_epochs: int = 100

    # ── Audio-teacher distillation (phase 3) ─────────────────────────────────
    distill_embedding_dim: int = 128
    distill_encoder_width: int = 32
    distill_pool_type: str = "avg"
    distill_sonic_dim: int = 64    # research_clap64_cal.f16.npy column count
    distill_clap_dim: int = 512    # research_clap512.f16.npy column count
    distill_temperature: float = 0.05
    distill_lr: float = 1e-3
    distill_weight_decay: float = 1e-4
    distill_batch_size: int = 256
    distill_crop_frames: int = 256
    distill_max_epochs: int = 50
    distill_proj_hidden: int = 512

    # ── Catalog extraction (phase 4) ─────────────────────────────────────────
    extract_batch_size: int = 512
    catalog_size: int = 272_853

    # ── Late interaction (phase 6) ────────────────────────────────────────────
    late_interaction_windows: int = 4
    late_interaction_window_frames: int = 128
    late_interaction_candidates: int = 1000

    # ── Evaluation metrics (phase 5) ─────────────────────────────────────────
    recall_cutoffs: Tuple[int, ...] = _DEFAULT_RECALL_CUTOFFS
    ndcg_cutoffs: Tuple[int, ...] = _DEFAULT_NDCG_CUTOFFS
    candidate_recall_at: Tuple[int, ...] = _DEFAULT_CANDIDATE_RECALL_AT
    bootstrap_iterations: int = 20_000
    bootstrap_seed: int = 20260711

    # ── Reproducibility ───────────────────────────────────────────────────────
    seed: int = 20260711
    train_seed: int = 42
    val_frac: float = 0.1
    test_frac: float = 0.1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(
            asdict(self), ensure_ascii=False, sort_keys=True, indent=2
        )

    def config_hash(self) -> str:
        """Stable 12-hex-char hash of the full config (for checkpoint labelling)."""
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:12]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperimentConfig":
        # Convert lists (from JSON) back to tuples for frozen tuple fields.
        tuple_fields = {"recall_cutoffs", "ndcg_cutoffs", "candidate_recall_at"}
        cleaned = {k: tuple(v) if k in tuple_fields and isinstance(v, list) else v
                   for k, v in d.items()}
        return cls(**cleaned)


@dataclass
class CheckpointMeta:
    """Metadata stored alongside every saved checkpoint."""

    phase: str         # "supcon" | "byol" | "distill" | "extract"
    epoch: int
    loss: float
    config_hash: str
    timestamp: str     # ISO-8601 UTC
    wall_seconds: float
    extra: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        _write_json_atomic(Path(path), asdict(self))

    @classmethod
    def load(cls, path: Path) -> "CheckpointMeta":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        extra = d.pop("extra", {})
        return cls(**d, extra=extra)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ResourceLog:
    """Timing and memory measurements for one pipeline phase."""

    phase: str
    wall_seconds: float
    peak_gpu_mb: float
    peak_ram_mb: float
    n_rows: int
    rows_per_sec: float
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# 3. LEAKAGE GUARD
# ─────────────────────────────────────────────────────────────────────────────


class LeakageGuard:
    """Identifies every benchmark artist (development + final) for exclusion.

    Protocol v5 requires excluding ALL benchmark artists — not just development
    — from distillation training rows.  Final-split artists are excluded to
    prevent any indirect exposure of the final evaluation signal during
    training.

    Usage::

        guard = LeakageGuard(benchmark_path)
        mask  = guard.exclusion_mask(catalog_artists)  # True = exclude row
    """

    def __init__(self, benchmark_path: str | Path) -> None:
        self._path = Path(benchmark_path)
        self._data: Dict[str, Any] = self._load()
        self._artist_set: Optional[Set[str]] = None

    def _load(self) -> Dict[str, Any]:
        data = json.loads(self._path.read_text(encoding="utf-8"))
        bid = data.get("benchmark_id")
        ver = data.get("schema_version")
        if bid != _V5_BENCHMARK_ID:
            raise ValueError(
                f"Expected benchmark_id={_V5_BENCHMARK_ID!r}, got {bid!r}"
            )
        if ver != _V5_SCHEMA_VERSION:
            raise ValueError(
                f"Expected schema_version={_V5_SCHEMA_VERSION}, got {ver!r}"
            )
        return data

    @property
    def pairs(self) -> List[Dict[str, Any]]:
        return self._data["pairs"]

    def benchmark_artist_set(self) -> Set[str]:
        """Normalized artist tokens from ALL pairs (development + final).

        Both _primary_artist and _credited_artists names are included so
        featured artists and collaborators are also captured.
        """
        if self._artist_set is None:
            artists: Set[str] = set()
            for pair in self.pairs:
                for side in ("query", "target"):
                    raw = pair[side]["artist"]
                    artists.add(_primary_artist(raw))
                    artists.update(_credited_artists(raw))
            self._artist_set = artists
        return self._artist_set

    def dev_pairs(self) -> List[Dict[str, Any]]:
        """Development-split pairs only.  Safe to use for model selection."""
        return [p for p in self.pairs if p["split"] == "development"]

    def final_pairs(self) -> List[Dict[str, Any]]:
        """Final-split pairs — returned for leakage audit ONLY.

        NEVER use these for evaluation or metric computation.
        """
        return [p for p in self.pairs if p["split"] == "final"]

    def exclusion_mask(self, artists: Sequence[str]) -> np.ndarray:
        """Boolean mask — True means the row should be excluded from training.

        A row is excluded if its primary or any credited artist name matches
        any benchmark artist (development or final split).
        """
        excluded = self.benchmark_artist_set()
        mask = np.zeros(len(artists), dtype=bool)
        for i, raw in enumerate(artists):
            pa = _primary_artist(raw)
            ca = _credited_artists(raw)
            if pa in excluded or bool(ca & excluded):
                mask[i] = True
        return mask

    @property
    def n_dev(self) -> int:
        return len(self.dev_pairs())

    @property
    def n_final(self) -> int:
        return len(self.final_pairs())

    def __repr__(self) -> str:
        n_ex = len(self.benchmark_artist_set())
        return (
            f"LeakageGuard(dev={self.n_dev}, final={self.n_final}, "
            f"excluded_artists={n_ex})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. FMA PACKED DATASET
# ─────────────────────────────────────────────────────────────────────────────


class FMAPackedDataset:
    """PyTorch-compatible dataset backed by fma_packed.npz.

    Each ``__getitem__`` returns (view1, view2, genre_idx, artist_str) where
    view1 and view2 are independently augmented float32 tensors of shape
    (1, n_mels, crop_frames) suitable for contrastive or self-supervised
    training.

    If a LeakageGuard is provided, rows whose artist appears in either the
    development or final benchmark split are excluded (typically a no-op for
    FMA, but enforced for strictness).

    Augmentation applies random time-crop, SpecAugment-style frequency and
    time masking, and a small per-sample gain jitter — all in NumPy before
    the Tensor is created to avoid GPU memory during data loading.
    """

    def __init__(
        self,
        packed_path: str | Path,
        crop_frames: int = 256,
        seed: int = 42,
        guard: Optional[LeakageGuard] = None,
        split: str = "train",
        val_frac: float = 0.1,
        test_frac: float = 0.1,
    ) -> None:
        data = np.load(Path(packed_path))
        X_raw: np.ndarray = data["X"]             # (N, n_mels, frames) float16
        genres_raw: np.ndarray = data["genres"]    # (N,) <U
        artists_raw: np.ndarray = data["artists"]  # (N,) <U
        track_ids_raw: np.ndarray = data["track_ids"]  # (N,) int64

        # Benchmark-artist leakage exclusion (usually no-op for FMA data).
        if guard is not None:
            keep = ~guard.exclusion_mask(artists_raw.tolist())
            X_raw = X_raw[keep]
            genres_raw = genres_raw[keep]
            artists_raw = artists_raw[keep]
            track_ids_raw = track_ids_raw[keep]

        # Stratified train / val / test split.
        idx_map = _stratified_split(genres_raw, val_frac, test_frac, seed)
        split_idx = idx_map[split]

        self.X: np.ndarray = X_raw[split_idx]          # float16
        self.genres: np.ndarray = genres_raw[split_idx]
        self.artists: np.ndarray = artists_raw[split_idx]
        self.track_ids: np.ndarray = track_ids_raw[split_idx]
        self.crop_frames = crop_frames

        # Integer genre labels (excluding "unlabeled" from the label set).
        self.genre_list: List[str] = sorted(
            {str(g) for g in self.genres if str(g) != "unlabeled"}
        )
        self._genre_to_idx: Dict[str, int] = {
            g: i for i, g in enumerate(self.genre_list)
        }
        self._rng = np.random.default_rng(seed)

    @property
    def n_classes(self) -> int:
        return len(self.genre_list)

    def __len__(self) -> int:
        return len(self.X)

    def _augment(self, spec: np.ndarray) -> "torch.Tensor":
        """Random crop + SpecAugment on a (n_mels, total_frames) float16 array.

        Returns (1, n_mels, crop_frames) float32 Tensor.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for FMAPackedDataset.__getitem__")

        spec_f = spec.astype(np.float32)
        n_mels, total_frames = spec_f.shape

        # Random time crop.
        if total_frames > self.crop_frames:
            start = int(self._rng.integers(0, total_frames - self.crop_frames + 1))
            crop = spec_f[:, start : start + self.crop_frames]
        else:
            pad = self.crop_frames - total_frames
            crop = np.pad(
                spec_f, ((0, 0), (0, pad)), constant_values=float(spec_f.min())
            )

        t = torch.from_numpy(crop.copy()).unsqueeze(0)  # (1, n_mels, crop_frames)
        fill = float(t.min())

        # Frequency masking.
        if self._rng.random() < 0.8:
            fw = int(self._rng.integers(1, max(2, n_mels // 8)))
            f0 = int(self._rng.integers(0, n_mels - fw))
            t[:, f0 : f0 + fw, :] = fill

        # Time masking.
        if self._rng.random() < 0.8:
            tw = int(self._rng.integers(1, max(2, self.crop_frames // 8)))
            t0 = int(self._rng.integers(0, self.crop_frames - tw))
            t[:, :, t0 : t0 + tw] = fill

        # Gain jitter.
        t = t * float(self._rng.uniform(0.9, 1.1))
        return t  # (1, n_mels, crop_frames) float32

    def __getitem__(
        self, idx: int
    ) -> Tuple["torch.Tensor", "torch.Tensor", int, str]:
        """Return (view1, view2, genre_label_int, artist_str)."""
        spec = self.X[idx]
        v1 = self._augment(spec)
        v2 = self._augment(spec)
        genre_str = str(self.genres[idx])
        label = self._genre_to_idx.get(genre_str, -1)
        return v1, v2, label, str(self.artists[idx])


# ─────────────────────────────────────────────────────────────────────────────
# 5. SUPERVISED CONTRASTIVE LOSS
# ─────────────────────────────────────────────────────────────────────────────


class SupConLoss(nn.Module if _TORCH_AVAILABLE else object):
    """Supervised contrastive loss with cross-artist masking.

    Positive pairs: same genre, different artist.
    Negative pairs: different genre (or same genre same artist if
    cross_artist=True, which is forced out of the positive set).

    The cross-artist constraint prevents the encoder from exploiting
    artist-identity shortcuts and forces learning of genuine sonic structure.

    Reference: Khosla et al., "Supervised Contrastive Learning", NeurIPS 2020.
    """

    def __init__(
        self, temperature: float = 0.1, cross_artist: bool = True
    ) -> None:
        if _TORCH_AVAILABLE:
            super().__init__()
        self.temperature = temperature
        self.cross_artist = cross_artist

    def forward(
        self,
        features: "torch.Tensor",       # (B, dim) L2-normalised
        genres: Sequence[int],           # (B,) integer genre labels
        artists: Optional[Sequence[str]] = None,  # (B,) strings
    ) -> "torch.Tensor":
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for SupConLoss.forward")

        B = features.shape[0]
        genres_t = torch.tensor(
            list(genres), device=features.device, dtype=torch.long
        )

        # Same-genre positive mask (excluding diagonal = self).
        pos_mask = genres_t.unsqueeze(1) == genres_t.unsqueeze(0)  # (B, B)
        diag = torch.eye(B, dtype=torch.bool, device=features.device)
        pos_mask = pos_mask & ~diag
        # ``-1`` is the explicit unknown-genre sentinel.  Unknown tracks never
        # become positives merely because they share the absence of a label.
        known = genres_t != -1
        pos_mask = pos_mask & known.unsqueeze(1) & known.unsqueeze(0)

        # Cross-artist: remove same-artist pairs from the positive set.
        if self.cross_artist and artists is not None:
            # Encode strings on CPU, then transfer one integer vector.  Avoid
            # B² scalar writes into a CUDA tensor (one kernel launch per write).
            artist_ids: Dict[str, int] = {}
            encoded = []
            for artist in artists:
                key = str(artist).casefold()
                encoded.append(artist_ids.setdefault(key, len(artist_ids)))
            artist_t = torch.tensor(
                encoded, device=features.device, dtype=torch.long
            )
            same_artist = artist_t.unsqueeze(1) == artist_t.unsqueeze(0)
            pos_mask = pos_mask & ~same_artist

        # Anchors with at least one valid positive.
        valid = pos_mask.sum(dim=1) > 0
        if not valid.any():
            # Preserve a zero-valued path to encoder outputs so the normal
            # training loop can call backward() and receive zero gradients.
            return (features * 0.0).sum()

        # Cosine similarity matrix (features are already L2-normalised).
        sim = (features @ features.T) / self.temperature  # (B, B)
        sim_masked = sim.masked_fill(diag, float("-inf"))  # kill self-similarity

        # Log-sum-exp denominator (all non-self pairs).
        log_denom = torch.logsumexp(sim_masked, dim=1)  # (B,)

        # Per-anchor contrastive loss.
        pos_sim_sum = (sim * pos_mask.float()).sum(dim=1)
        n_pos = pos_mask.sum(dim=1).clamp(min=1).float()
        per_anchor = -(pos_sim_sum / n_pos - log_denom)

        return per_anchor[valid].mean()


# ─────────────────────────────────────────────────────────────────────────────
# 6. BYOL COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────


class BYOLProjectionHead(nn.Module if _TORCH_AVAILABLE else object):
    """BN-ReLU MLP projection head (shared by online and target networks)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        if _TORCH_AVAILABLE:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, out_dim),
            )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)


class BYOLPredictionHead(nn.Module if _TORCH_AVAILABLE else object):
    """Extra MLP prediction head for the online network only."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        if _TORCH_AVAILABLE:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, out_dim),
            )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.net(x)


def byol_loss(
    online_pred_1: "torch.Tensor",
    online_pred_2: "torch.Tensor",
    target_proj_1: "torch.Tensor",
    target_proj_2: "torch.Tensor",
) -> "torch.Tensor":
    """Symmetric BYOL loss: mean negative cosine similarity over both views.

    The target projections are always detached so no gradient flows through
    the target network (it is updated solely via EMA).
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for byol_loss")

    def _neg_cosine(p: "torch.Tensor", z: "torch.Tensor") -> "torch.Tensor":
        p_n = F.normalize(p, dim=1)
        z_n = F.normalize(z.detach(), dim=1)
        return -(p_n * z_n).sum(dim=1).mean()

    return (
        _neg_cosine(online_pred_1, target_proj_2)
        + _neg_cosine(online_pred_2, target_proj_1)
    ) * 0.5


class BYOLModel(nn.Module if _TORCH_AVAILABLE else object):
    """Bootstrap Your Own Latent dual-network model.

    Online network  : encoder → projection → prediction (trained by gradient).
    Target network  : encoder → projection           (EMA, no gradient).

    Training minimises the negative cosine similarity between the online
    prediction of one view and the target projection of the other view —
    no negative pairs are used.

    Reference: Grill et al., "Bootstrap Your Own Latent: A New Approach to
    Self-Supervised Learning", NeurIPS 2020.
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        width: int = 64,
        pool_type: str = "gem",
        projection_dim: int = 256,
        prediction_dim: int = 128,
        ema_decay: float = 0.996,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for BYOLModel")
        super().__init__()

        from .model import ResNetAudioEncoder

        # Online network.
        self.online_encoder = ResNetAudioEncoder(
            embedding_dim=embedding_dim, width=width, pool_type=pool_type
        )
        self.online_projector = BYOLProjectionHead(
            embedding_dim, projection_dim, projection_dim
        )
        self.online_predictor = BYOLPredictionHead(
            projection_dim, prediction_dim, projection_dim
        )

        # Target network — same architecture, EMA weights, no gradient.
        self.target_encoder = ResNetAudioEncoder(
            embedding_dim=embedding_dim, width=width, pool_type=pool_type
        )
        self.target_projector = BYOLProjectionHead(
            embedding_dim, projection_dim, projection_dim
        )

        # Initialise target = online.
        self.target_encoder.load_state_dict(self.online_encoder.state_dict())
        self.target_projector.load_state_dict(self.online_projector.state_dict())
        for p in (
            *self.target_encoder.parameters(),
            *self.target_projector.parameters(),
        ):
            p.requires_grad_(False)

        self.ema_decay = ema_decay

    @torch.no_grad()
    def update_target(self, decay: Optional[float] = None) -> None:
        """EMA update: target ← τ·target + (1−τ)·online."""
        tau = decay if decay is not None else self.ema_decay
        pairs = [
            (self.online_encoder, self.target_encoder),
            (self.online_projector, self.target_projector),
        ]
        for online_net, target_net in pairs:
            for p_online, p_target in zip(
                online_net.parameters(), target_net.parameters()
            ):
                p_target.data.mul_(tau).add_((1.0 - tau) * p_online.data)

    def _online_forward(self, x: "torch.Tensor") -> "torch.Tensor":
        h = self.online_encoder(x, normalize=False)
        z = self.online_projector(h)
        return self.online_predictor(z)

    @torch.no_grad()
    def _target_forward(self, x: "torch.Tensor") -> "torch.Tensor":
        h = self.target_encoder(x, normalize=False)
        return self.target_projector(h)

    def forward(
        self, v1: "torch.Tensor", v2: "torch.Tensor"
    ) -> Tuple[
        "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"
    ]:
        """Return (online_pred_1, online_pred_2, target_proj_1, target_proj_2)."""
        p1 = self._online_forward(v1)
        p2 = self._online_forward(v2)
        t1 = self._target_forward(v1)
        t2 = self._target_forward(v2)
        return p1, p2, t1, t2


# ─────────────────────────────────────────────────────────────────────────────
# 7. DISTILLATION DATASET AND LOSS
# ─────────────────────────────────────────────────────────────────────────────


class DistillationDataset:
    """Dataset for audio-teacher distillation.

    Student input  : mel spectrogram crop from ``mel_matrix[i]``.
    Teacher target : L2-normalised concat(sonic[i], clap[i]).

    All rows whose primary or credited artist appears in the v5 benchmark
    (development OR final split) are excluded via ``guard.exclusion_mask``.
    This is strict: a single matching artist token triggers exclusion.
    """

    def __init__(
        self,
        mel_matrix: np.ndarray,       # (N, n_mels, frames) float16 or float32
        teacher_matrix: np.ndarray,   # (N, teacher_dim) float32 (use _build_teacher)
        artists: Sequence[str],       # (N,) artist strings
        guard: LeakageGuard,
        crop_frames: int = 256,
        seed: int = 42,
    ) -> None:
        n = mel_matrix.shape[0]
        if teacher_matrix.shape[0] != n or len(artists) != n:
            raise ValueError(
                f"Inconsistent lengths: mel={n}, teacher={teacher_matrix.shape[0]}, "
                f"artists={len(artists)}"
            )

        keep_mask = ~guard.exclusion_mask(list(artists))
        keep_idx = np.where(keep_mask)[0]

        # Keep aligned source arrays by reference.  Advanced-indexing the 18 GB
        # production mel cache would duplicate it and exceed the 48 GB host's
        # safe working set during training.
        self.mel: np.ndarray = mel_matrix
        self.teacher: np.ndarray = teacher_matrix
        self.keep_idx: np.ndarray = keep_idx.astype(np.int64, copy=False)
        self.crop_frames = crop_frames
        self.teacher_dim: int = int(teacher_matrix.shape[1])
        self._excluded: int = int(n - len(keep_idx))
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.keep_idx)

    def __getitem__(self, idx: int) -> Tuple["torch.Tensor", "torch.Tensor"]:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for DistillationDataset.__getitem__")
        source_idx = int(self.keep_idx[idx])
        spec = self.mel[source_idx].astype(np.float32)
        n_mels, total_frames = spec.shape
        if total_frames > self.crop_frames:
            start = int(
                self._rng.integers(0, total_frames - self.crop_frames + 1)
            )
            spec = spec[:, start : start + self.crop_frames]
        elif total_frames < self.crop_frames:
            pad = self.crop_frames - total_frames
            spec = np.pad(
                spec, ((0, 0), (0, pad)), constant_values=float(spec.min())
            )
        mel_t = torch.from_numpy(spec.copy()).unsqueeze(0)  # (1, n_mels, crop)
        teacher_t = torch.from_numpy(
            np.asarray(self.teacher[source_idx], dtype=np.float32).copy()
        )
        return mel_t, teacher_t


class DistillationProjection(nn.Module if _TORCH_AVAILABLE else object):
    """MLP that projects from student embedding dim to teacher embedding dim."""

    def __init__(
        self, student_dim: int, hidden_dim: int, teacher_dim: int
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for DistillationProjection")
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(student_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, teacher_dim),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        return self.proj(x)


def distillation_loss(
    student_proj: "torch.Tensor",
    teacher_vec: "torch.Tensor",
    temperature: float = 1.0,
) -> "torch.Tensor":
    """Cosine-direction distillation used by the iteration-4 checkpoint.

    ``temperature`` is retained for configuration/checkpoint compatibility but
    does not alter cosine-direction training; scaling a vector before L2
    normalisation is mathematically invariant.  It must remain positive.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch required for distillation_loss")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    student = F.normalize(student_proj, dim=1)
    teacher = F.normalize(teacher_vec.detach(), dim=1)
    # Adding one changes only the reported loss origin, not checkpoint gradients.
    return 1.0 - (student * teacher).sum(dim=1).mean()


# ─────────────────────────────────────────────────────────────────────────────
# 8. CATALOG EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────


class CatalogExtractor:
    """Extracts float16 embeddings for the full production catalog.

    Processes ``mel_matrix`` in batches, using AMP on CUDA when available, and
    returns an (N, embedding_dim) float16 array.  This is a read-only inference
    step; no gradients are computed.
    """

    def __init__(
        self,
        encoder: "nn.Module",
        batch_size: int = 512,
        device: Optional[str] = None,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for CatalogExtractor")
        self._encoder = encoder
        self._batch_size = batch_size
        self._device = device or _best_device()

    def extract(
        self,
        mel_matrix: np.ndarray,  # (N, n_mels, total_frames)
        crop_frames: int = 256,
    ) -> np.ndarray:
        """Run encoder inference; return (N, dim) float16.

        Center-crops each spectrogram to ``crop_frames`` before embedding.
        """
        enc = self._encoder.to(self._device)
        enc.eval()
        n = len(mel_matrix)
        results: List[np.ndarray] = []

        for start in range(0, n, self._batch_size):
            batch = mel_matrix[start : start + self._batch_size].astype(np.float32)
            # Centre-crop to crop_frames.
            total = batch.shape[-1]
            if total > crop_frames:
                s = (total - crop_frames) // 2
                batch = batch[:, :, s : s + crop_frames]
            elif total < crop_frames:
                pad = crop_frames - total
                batch = np.pad(
                    batch, ((0, 0), (0, 0), (0, pad)),
                    constant_values=float(batch.min()),
                )
            t = torch.from_numpy(batch).unsqueeze(1).to(self._device)
            use_amp = self._device == "cuda"
            with torch.no_grad(), torch.amp.autocast(self._device, enabled=use_amp):
                z = enc(t, normalize=True)
            results.append(z.float().cpu().numpy().astype(np.float16))

        return np.concatenate(results, axis=0)  # (N, dim) float16


# ─────────────────────────────────────────────────────────────────────────────
# 9. V5 PAIR RESOLVER
# ─────────────────────────────────────────────────────────────────────────────


class V5PairResolver:
    """Resolve v5 benchmark (title, artist) dicts to catalog row indices.

    Adapted from real_benchmark.PairResolver for the v5 schema, which has no
    deezer_id and uses 'category_a_sonic' evidence category.
    """

    def __init__(
        self, titles: Sequence[str], artists: Sequence[str]
    ) -> None:
        self._titles = np.asarray(titles, dtype=str)
        self._artists = np.asarray(artists, dtype=str)
        self._norm_titles = np.array([_normalize(t) for t in self._titles])
        self._primary_artists = np.array([_primary_artist(a) for a in self._artists])

        # Inverted index: normalised title -> list of row indices.
        self._by_title: Dict[str, List[int]] = {}
        for row, nt in enumerate(self._norm_titles):
            self._by_title.setdefault(str(nt), []).append(row)

    # ------------------------------------------------------------------
    def _artist_match(self, canonical: str, catalogue_row: int) -> bool:
        cat_raw = str(self._artists[catalogue_row])
        c_parts = _credited_artists(canonical)
        k_parts = _credited_artists(cat_raw)
        if c_parts & k_parts:
            return True
        cp = _primary_artist(canonical)
        kp = _primary_artist(cat_raw)
        if not cp or not kp:
            return False
        if cp in kp or kp in cp:
            return min(len(cp), len(kp)) >= 4
        # Handle legacy encoding damage (e.g. "Beyonc_").
        return len(cp) >= 5 and len(kp) >= 5 and cp[:5] == kp[:5]

    _DERIVATIVE_RE = re.compile(
        r"\b(?:karaoke|tribute|slowed|reverb|nightcore|instrumental|"
        r"remix|cover|live|acoustic|sped[- ]up|speed[- ]up|mashup)\b"
    )

    def _version_penalty(self, row: int) -> Tuple[int, int]:
        title = str(self._titles[row]).casefold()
        deriv = int(bool(self._DERIVATIVE_RE.search(title)))
        return deriv, len(title)

    def rows(self, song: Dict[str, str]) -> List[int]:
        """All catalog rows that match title + artist."""
        norm = _normalize(song["title"])
        candidates = self._by_title.get(norm, [])
        return [r for r in candidates if self._artist_match(song["artist"], r)]

    def query_row(self, song: Dict[str, str]) -> Optional[int]:
        """Prefer non-derivative, shorter titles."""
        rows = self.rows(song)
        return min(rows, key=self._version_penalty) if rows else None

    def target_rows(self, song: Dict[str, str]) -> Set[int]:
        """Only non-derivative recordings (penalty == 0)."""
        return {r for r in self.rows(song) if self._version_penalty(r)[0] == 0}


# ─────────────────────────────────────────────────────────────────────────────
# 10. DEVELOPMENT EVALUATOR
# ─────────────────────────────────────────────────────────────────────────────


class DevEvaluator:
    """Development-only evaluator for the v5 frozen benchmark.

    INVARIANT: the 'final' split is never touched.

    Metrics computed per run:
      R@1/5/10/20/50 · MRR · NDCG@10/50
      primary = mean(NDCG@10, MRR, R@10)     [v5 metric policy]
      per-pair exact target rank
      candidate recall @50/200/1000           [pre-filter coverage]
      optional bootstrap CI(95%) on primary

    Only development-split pairs with both query and target resolved in the
    supplied catalog are scored.
    """

    def __init__(
        self,
        guard: LeakageGuard,
        titles: Sequence[str],
        artists: Sequence[str],
        cfg: ExperimentConfig,
    ) -> None:
        self._guard = guard
        self._titles = list(titles)
        self._artists = list(artists)
        self._cfg = cfg
        self._resolver = V5PairResolver(titles, artists)
        self._dev_pairs: List[Dict[str, Any]] = guard.dev_pairs()

    # ------------------------------------------------------------------
    # Static metric helpers
    # ------------------------------------------------------------------

    @staticmethod
    def ndcg_at_k(rank: int, k: int) -> float:
        """NDCG@k for a single relevant item at ``rank`` (0 = not found).

        With one relevant item, IDCG@k = 1 / log2(2) = 1.0, so
        NDCG@k = 1 / log2(rank + 1) for rank <= k, else 0.
        """
        if rank <= 0 or rank > k:
            return 0.0
        return 1.0 / math.log2(rank + 1)

    @staticmethod
    def mrr_from_rank(rank: int) -> float:
        return 1.0 / rank if rank > 0 else 0.0

    # ------------------------------------------------------------------
    # Core ranking
    # ------------------------------------------------------------------

    def _rank_target(
        self,
        query_row: int,
        target_rows: Set[int],
        embeddings: np.ndarray,
        apply_artist_filter: bool = True,
    ) -> Tuple[int, np.ndarray]:
        """Rank catalog by cosine to query; return (rank_of_first_target, order).

        Same-artist tracks are excluded (rank = infinity) because they are not
        valid recommendations.  The full order array is returned for computing
        candidate recall before any filtering.
        """
        q = embeddings[query_row].astype(np.float32)
        scores = embeddings.astype(np.float32) @ q  # (N,)
        scores[query_row] = -np.inf  # exclude self

        if apply_artist_filter:
            qa = _primary_artist(self._artists[query_row])
            for i, a in enumerate(self._artists):
                if _primary_artist(a) == qa:
                    scores[i] = -np.inf

        order = np.argsort(scores)[::-1]
        rank = 0
        for position, row in enumerate(order, 1):
            if int(row) in target_rows:
                rank = position
                break
        return rank, order

    def _raw_candidate_rank(
        self,
        query_row: int,
        target_rows: Set[int],
        embeddings: np.ndarray,
    ) -> int:
        """Raw cosine rank without same-artist filtering (for candidate recall)."""
        q = embeddings[query_row].astype(np.float32)
        scores = embeddings.astype(np.float32) @ q
        scores[query_row] = -np.inf
        order = np.argsort(scores)[::-1]
        for position, row in enumerate(order, 1):
            if int(row) in target_rows:
                return position
        return 0

    # ------------------------------------------------------------------
    # Public evaluation entry point
    # ------------------------------------------------------------------

    def evaluate(
        self,
        embeddings: np.ndarray,
        method_name: str = "experiment",
        bootstrap: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate development pairs against the supplied embedding matrix.

        Parameters
        ----------
        embeddings : (N_catalog, dim) — must be L2-normalised float16/32.
        method_name : label used in the returned dict.
        bootstrap : if True, append a 20 000-iteration bootstrap CI on primary.
        """
        n_cat = len(self._titles)
        if embeddings.shape[0] != n_cat:
            raise ValueError(
                f"embeddings.shape[0]={embeddings.shape[0]} != catalog size {n_cat}"
            )

        # Ensure L2 normalisation.
        emb = embeddings.astype(np.float32)
        norms = np.linalg.norm(emb[:min(32, n_cat)], axis=1)
        if not np.allclose(norms, 1.0, atol=0.05):
            norms_all = np.linalg.norm(emb, axis=1, keepdims=True)
            emb = emb / np.clip(norms_all, 1e-9, None)

        records: List[Dict[str, Any]] = []
        # Candidate recall: raw rank (no artist filter) per pair.
        raw_ranks: List[int] = []

        for pair in self._dev_pairs:
            qr = self._resolver.query_row(pair["query"])
            tr = self._resolver.target_rows(pair["target"])
            q_found = qr is not None
            t_found = bool(tr)

            rank = 0
            order: np.ndarray = np.empty(0, dtype=np.int64)
            if q_found and t_found:
                rank, order = self._rank_target(qr, tr, emb)
                # Raw candidate rank (for candidate_recall).
                raw_rank = self._raw_candidate_rank(qr, tr, emb)
                raw_ranks.append(raw_rank)

            records.append(
                {
                    "pair_id": pair["id"],
                    "split": pair["split"],
                    "scene": pair["scene"],
                    "query": dict(pair["query"]),
                    "target": dict(pair["target"]),
                    "query_found": q_found,
                    "query_row": qr,
                    "target_found": t_found,
                    "target_rows": sorted(tr),
                    "target_rank": rank,
                    "reciprocal_rank": self.mrr_from_rank(rank),
                }
            )

        metrics = self._compute_metrics(records)

        # Candidate recall @K (raw cosine, no same-artist filter).
        cand_recall: Dict[str, float] = {}
        for k in self._cfg.candidate_recall_at:
            if raw_ranks:
                hits = sum(1 for r in raw_ranks if 0 < r <= k)
                cand_recall[f"candidate_recall_at_{k}"] = hits / len(raw_ranks)
            else:
                cand_recall[f"candidate_recall_at_{k}"] = 0.0
        metrics.update(cand_recall)

        # Bootstrap CI on primary.
        if bootstrap and records:
            metrics["primary_bootstrap_ci95"] = self._bootstrap_primary(records)

        return {
            "method": method_name,
            "metrics": metrics,
            "pairs": records,
            "n_dev_pairs": len(records),
            "n_rankable": int(
                sum(r["query_found"] and r["target_found"] for r in records)
            ),
            "timestamp": _iso_now(),
        }

    def _compute_metrics(
        self, records: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Compute the full v5 metric set from pair records."""
        if not records:
            return {}

        ranks = np.array([r["target_rank"] for r in records], dtype=np.int64)
        out: Dict[str, float] = {}

        for k in self._cfg.recall_cutoffs:
            out[f"recall_at_{k}"] = float(np.mean((ranks > 0) & (ranks <= k)))

        out["mrr"] = float(
            np.mean([1.0 / r if r > 0 else 0.0 for r in ranks])
        )

        for k in self._cfg.ndcg_cutoffs:
            out[f"ndcg_at_{k}"] = float(
                np.mean([self.ndcg_at_k(int(r), k) for r in ranks])
            )

        # v5 primary metric (predeclared in metric_policy).
        out["primary"] = float(
            np.mean([out["ndcg_at_10"], out["mrr"], out["recall_at_10"]])
        )
        return out

    def _bootstrap_primary(
        self,
        records: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """20 000-iteration paired bootstrap CI on primary metric."""
        rng = np.random.default_rng(self._cfg.bootstrap_seed)
        n = len(records)
        primaries: List[float] = []
        for _ in range(self._cfg.bootstrap_iterations):
            idx = rng.integers(0, n, size=n)
            sample = [records[int(i)] for i in idx]
            m = self._compute_metrics(sample)
            primaries.append(m.get("primary", 0.0))
        arr = np.array(primaries)
        return {
            "mean": float(arr.mean()),
            "ci95_low": float(np.percentile(arr, 2.5)),
            "ci95_high": float(np.percentile(arr, 97.5)),
            "n_iterations": self._cfg.bootstrap_iterations,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 11. LATE-INTERACTION RERANKER
# ─────────────────────────────────────────────────────────────────────────────


class LateInteractionReranker:
    """4-window MaxSim late-interaction reranker.

    Each track is represented by ``n_windows`` non-overlapping time-window
    embeddings.  Query and candidate are scored by the MaxSim operator
    (ColBERT-style): for each query window, take the maximum cosine similarity
    to any candidate window, then average over all query windows.

    This captures the *best local match* between any segment of the query and
    any segment of the candidate, which is more robust for audio with variable
    structure (intro / verse / chorus / outro) than a single global embedding.
    """

    def __init__(
        self,
        encoder: "nn.Module",
        n_windows: int = 4,
        window_frames: int = 128,
        batch_size: int = 256,
        device: Optional[str] = None,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for LateInteractionReranker")
        self._encoder = encoder
        self.n_windows = n_windows
        self.window_frames = window_frames
        self._batch_size = batch_size
        self._device = device or _best_device()

    def extract_windows(
        self, mel_matrix: np.ndarray
    ) -> np.ndarray:
        """Extract ``n_windows`` embeddings per track from non-overlapping crops.

        Parameters
        ----------
        mel_matrix : (N, n_mels, total_frames)

        Returns
        -------
        (N, n_windows, embedding_dim) float16
        """
        n, n_mels, total_frames = mel_matrix.shape
        enc = self._encoder.to(self._device).eval()

        # Compute non-overlapping window start frames.
        stride = total_frames // self.n_windows
        win_starts = [i * stride for i in range(self.n_windows)]

        per_win: List[List[np.ndarray]] = [[] for _ in range(self.n_windows)]
        use_amp = self._device == "cuda"

        for w_idx, start in enumerate(win_starts):
            end = start + self.window_frames
            if end > total_frames:
                start = max(0, total_frames - self.window_frames)
                end = total_frames

            for b_start in range(0, n, self._batch_size):
                batch = mel_matrix[b_start : b_start + self._batch_size].astype(
                    np.float32
                )
                crop = batch[:, :, start:end]
                if crop.shape[2] < self.window_frames:
                    pad = self.window_frames - crop.shape[2]
                    crop = np.pad(
                        crop, ((0, 0), (0, 0), (0, pad)),
                        constant_values=float(crop.min()),
                    )
                t = torch.from_numpy(crop).unsqueeze(1).to(self._device)
                with torch.no_grad(), torch.amp.autocast(
                    self._device, enabled=use_amp
                ):
                    z = enc(t, normalize=True)
                per_win[w_idx].append(z.float().cpu().numpy().astype(np.float16))

        per_win_arrays = [np.concatenate(chunks, axis=0) for chunks in per_win]
        # Stack to (N, n_windows, embedding_dim).
        return np.stack(per_win_arrays, axis=1)  # (N, W, dim)

    @staticmethod
    def maxsim_score(
        query_windows: np.ndarray,     # (Wq, dim) float32
        candidate_windows: np.ndarray,  # (K, Wc, dim) float32
    ) -> np.ndarray:
        """Compute MaxSim scores: (K,) float32.

        For each query window, find the maximum cosine similarity over all
        candidate windows, then average over query windows.
        """
        q = query_windows.astype(np.float32)   # (Wq, D)
        c = candidate_windows.astype(np.float32)  # (K, Wc, D)
        # Dot product between every query window and every candidate window.
        # q: (Wq, D), c[k]: (Wc, D)  → dots[k]: (Wq, Wc)
        dots = np.einsum("qd,kwd->kqw", q, c)  # (K, Wq, Wc)
        # Max over candidate windows per query window.
        max_per_qwin = dots.max(axis=2)  # (K, Wq)
        # Mean over query windows.
        return max_per_qwin.mean(axis=1)  # (K,)

    def rerank(
        self,
        query_windows: np.ndarray,   # (n_windows, dim)
        candidate_indices: Sequence[int],
        all_windows: np.ndarray,     # (N_catalog, n_windows, dim)
    ) -> List[int]:
        """Rerank candidate_indices by MaxSim score; return descending order."""
        cand_arr = np.array(candidate_indices, dtype=np.int64)
        cand_wins = all_windows[cand_arr]  # (K, n_windows, dim)
        scores = self.maxsim_score(
            query_windows.astype(np.float32),
            cand_wins.astype(np.float32),
        )
        order = np.argsort(scores)[::-1]
        return [int(candidate_indices[i]) for i in order]


# ─────────────────────────────────────────────────────────────────────────────
# 12. AUDIO EXPERIMENT PIPELINE
# ─────────────────────────────────────────────────────────────────────────────


class AudioMetricProjector(nn.Module):
    """Compact audio-only projection trained from independent positive pairs."""

    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 output_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, features: "torch.Tensor") -> "torch.Tensor":
        return F.normalize(self.network(features), dim=1)


def audio_feature_fusion(*matrices: np.ndarray) -> np.ndarray:
    """Group-normalise and concatenate audio-derived feature matrices."""
    if not matrices:
        raise ValueError("At least one audio feature matrix is required")
    rows = {matrix.shape[0] for matrix in matrices}
    if len(rows) != 1:
        raise ValueError("Audio feature matrices must have identical row counts")
    groups = []
    for matrix in matrices:
        group = np.asarray(matrix, dtype=np.float32)
        group /= np.maximum(np.linalg.norm(group, axis=1, keepdims=True), 1e-9)
        groups.append(group)
    fused = np.concatenate(groups, axis=1)
    fused /= np.maximum(np.linalg.norm(fused, axis=1, keepdims=True), 1e-9)
    return fused.astype(np.float32, copy=False)


def train_audio_pair_metric(
    features: np.ndarray,
    positive_pairs: Sequence[Mapping[str, Any]],
    out_dir: Path,
    seed: int = 20260711,
    hidden_dim: int = 256,
    output_dim: int = 128,
    batch_size: int = 256,
    epochs: int = 80,
    learning_rate: float = 3e-4,
    temperature: float = 0.07,
) -> Tuple[AudioMetricProjector, Dict[str, Any]]:
    """Train symmetric InfoNCE on independent cross-artist positives.

    ``positive_pairs`` must contain pre-resolved ``query_row`` and non-empty
    ``target_rows`` values.  The caller is responsible for benchmark-artist
    exclusion; the saved report records the exact rows and split seed.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for pair metric training")
    if not positive_pairs:
        raise ValueError("No independent positive pairs were supplied")
    rng = np.random.default_rng(seed)
    pair_rows = np.asarray([
        (int(pair["query_row"]), int(pair["target_rows"][0]))
        for pair in positive_pairs
    ], dtype=np.int64)
    permutation = rng.permutation(len(pair_rows))
    validation_count = max(1, len(pair_rows) // 10)
    validation = pair_rows[permutation[:validation_count]]
    training = pair_rows[permutation[validation_count:]]

    torch.manual_seed(seed)
    device = _best_device()
    model = AudioMetricProjector(
        features.shape[1], hidden_dim=hidden_dim, output_dim=output_dim
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    feature_tensor = torch.from_numpy(np.asarray(features, dtype=np.float32))

    def batch_loss(rows: np.ndarray) -> "torch.Tensor":
        left = feature_tensor[rows[:, 0]].to(device, non_blocking=True)
        right = feature_tensor[rows[:, 1]].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            query = model(left)
            target = model(right)
            logits = query @ target.T / temperature
            labels = torch.arange(len(rows), device=device)
            return 0.5 * (
                F.cross_entropy(logits, labels)
                + F.cross_entropy(logits.T, labels)
            )

    started = time.perf_counter()
    best_validation = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    history: List[Dict[str, float]] = []
    steps = 0
    for epoch in range(epochs):
        model.train()
        epoch_rows = training[rng.permutation(len(training))]
        losses = []
        for start in range(0, len(epoch_rows), batch_size):
            rows = epoch_rows[start:start + batch_size]
            if len(rows) < 2:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(rows)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))
            steps += 1
        model.eval()
        with torch.no_grad():
            val_losses = [
                float(batch_loss(validation[start:start + batch_size]).item())
                for start in range(0, len(validation), batch_size)
                if len(validation[start:start + batch_size]) >= 2
            ]
        validation_loss = float(np.mean(val_losses)) if val_losses else 0.0
        history.append({
            "epoch": float(epoch + 1),
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "validation_loss": validation_loss,
        })
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
    if best_state is not None:
        model.load_state_dict(best_state)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "input_dim": int(features.shape[1]),
        "hidden_dim": hidden_dim,
        "output_dim": output_dim,
        "seed": seed,
    }
    torch.save(checkpoint, out_dir / "audio_pair_metric.pt")
    report = {
        "method": "independent_pair_audio_metric",
        "created_at": _iso_now(),
        "positive_pairs": int(len(pair_rows)),
        "training_pairs": int(len(training)),
        "validation_pairs": int(len(validation)),
        "benchmark_artist_overlap": [],
        "epochs": epochs,
        "steps": steps,
        "batch_size": batch_size,
        "temperature": temperature,
        "learning_rate": learning_rate,
        "best_validation_loss": best_validation,
        "wall_seconds": time.perf_counter() - started,
        "history": history,
    }
    _write_json_atomic(out_dir / "training_report.json", report)
    return model, report


def project_audio_catalog(
    model: AudioMetricProjector,
    features: np.ndarray,
    batch_size: int = 4096,
) -> np.ndarray:
    """Project the complete catalogue to a compact float16 matrix."""
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for catalogue projection")
    device = _best_device()
    model = model.to(device).eval()
    output = np.empty(
        (len(features), model.network[-1].out_features), dtype=np.float16
    )
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            stop = min(start + batch_size, len(features))
            batch = torch.from_numpy(
                np.asarray(features[start:stop], dtype=np.float32)
            ).to(device)
            output[start:stop] = model(batch).cpu().numpy().astype(np.float16)
    return output

class AudioCandidateUnionRanker:
    """Two-stage audio-only candidate union and learned compatibility reranker."""

    def __init__(
        self,
        recommender: Any,
        raw_neural: np.ndarray,
        raw_vibe: np.ndarray,
        sonic: np.ndarray,
        clap: np.ndarray,
        distilled: np.ndarray,
        supcon: np.ndarray,
        byol: np.ndarray,
        scorer_checkpoint: Path,
        base_candidates: int = 10,
        per_signal_candidates: int = 8,
        scorer_weight: float = 1.0,
        head_size: int = 10,
        tail_stride: int = 6,
        tail_depth: int = 60,
    ) -> None:
        self.recommender = recommender
        self.base_candidates = int(base_candidates)
        self.per_signal_candidates = int(per_signal_candidates)
        self.scorer_weight = float(scorer_weight)
        self.head_size = int(head_size)
        self.tail_stride = int(tail_stride)
        self.tail_depth = int(tail_depth)
        if self.head_size < 1 or self.tail_stride < 1 or self.tail_depth < 1:
            raise ValueError("Head size and tail stratification must be positive")
        from .real_benchmark import ProductionRanker

        self.base_ranker = ProductionRanker(recommender, heldout=set())
        self.quality_filter = TitleQualityFilter()
        self.quality_mask = self.quality_filter.keep_mask(
            recommender.titles, recommender.artists
        )
        vibe = np.asarray(raw_vibe, dtype=np.float32)
        vibe = (vibe - vibe.mean(axis=0)) / (vibe.std(axis=0) + 1e-6)
        self.groups = [
            self._normalise(raw_neural), self._normalise(vibe),
            self._normalise(sonic), self._normalise(clap),
            self._normalise(distilled), self._normalise(supcon),
            self._normalise(byol),
        ]
        checkpoint = torch.load(
            scorer_checkpoint, map_location="cpu", weights_only=True
        )
        self.scorer = nn.Sequential(
            nn.LayerNorm(int(checkpoint["input_dim"])),
            nn.Linear(int(checkpoint["input_dim"]), 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.scorer.load_state_dict(checkpoint["state_dict"])
        self.device = _best_device()
        self.scorer = self.scorer.to(self.device).eval()

    @staticmethod
    def _normalise(matrix: np.ndarray) -> np.ndarray:
        value = np.asarray(matrix, dtype=np.float32)
        value /= np.maximum(np.linalg.norm(value, axis=1, keepdims=True), 1e-9)
        return value

    @staticmethod
    def _top(score: np.ndarray, count: int) -> np.ndarray:
        count = min(int(count), len(score))
        if count == len(score):
            return np.argsort(score)[::-1]
        selected = np.argpartition(-score, count - 1)[:count]
        return selected[np.argsort(score[selected])[::-1]]

    def _pair_features(self, query_row: int, candidates: np.ndarray) -> np.ndarray:
        cosine = np.stack([
            (group[candidates] * group[query_row]).sum(axis=1)
            for group in self.groups
        ], axis=1)
        return np.concatenate([
            cosine, cosine * cosine,
            cosine.min(axis=1, keepdims=True),
            cosine.max(axis=1, keepdims=True),
        ], axis=1).astype(np.float32)

    def rank(self, query_row: int, n: int = 50) -> List[int]:
        base_score = self.base_ranker._base_parts(query_row)[2]
        specialist_groups = (
            self.groups[2], self.groups[4], self.groups[5], self.groups[3]
        )
        auxiliary_scores = [
            group @ group[query_row] for group in specialist_groups
        ]
        candidate_set = set(map(int, self._top(
            base_score, self.base_candidates
        )))
        for score in auxiliary_scores:
            candidate_set.update(map(int, self._top(
                score, self.per_signal_candidates
            )))
        candidates = np.asarray(sorted(candidate_set), dtype=np.int64)
        base = base_score[candidates]
        base = (base - base.mean()) / (base.std() + 1e-6)
        pair_features = self._pair_features(query_row, candidates)
        with torch.no_grad():
            learned = self.scorer(
                torch.from_numpy(pair_features).to(self.device)
            ).squeeze(1).cpu().numpy()
        learned = (learned - learned.mean()) / (learned.std() + 1e-6)
        learned_head = candidates[
            np.argsort(base + self.scorer_weight * learned)[::-1]
        ].tolist()

        seed_artist = _normalize(str(self.recommender.artists[query_row]))
        seed_title = str(self.recommender.titles[query_row])
        selected: List[int] = []
        seen_rows: Set[int] = set()
        seen_artists: Set[str] = set()

        def accept(raw: int) -> bool:
            row = int(raw)
            title = str(self.recommender.titles[row])
            artist = _normalize(str(self.recommender.artists[row]))
            if row == query_row or row in seen_rows or artist == seed_artist:
                return False
            if not bool(self.quality_mask[row]):
                return False
            if self.quality_filter.seed_title_in_result(seed_title, title):
                return False
            if artist in seen_artists:
                return False
            selected.append(row)
            seen_rows.add(row)
            seen_artists.add(artist)
            return True

        # The visible head is selected by the learned independent-pair scorer.
        for row in learned_head:
            accept(row)
            if len(selected) >= min(n, self.head_size):
                break

        # The tail deliberately samples deeper ranks from every independent
        # audio representation.  This candidate-recall layer was selected on
        # DEV before FINAL was opened and uses no metadata or popularity prior.
        specialist_pools: List[List[int]] = []
        for score in auxiliary_scores:
            pool: List[int] = []
            local_artists: Set[str] = set()
            for raw in np.argsort(score)[::-1]:
                row = int(raw)
                title = str(self.recommender.titles[row])
                artist = _normalize(str(self.recommender.artists[row]))
                if row == query_row or artist == seed_artist:
                    continue
                if not bool(self.quality_mask[row]):
                    continue
                if self.quality_filter.seed_title_in_result(seed_title, title):
                    continue
                if artist in local_artists:
                    continue
                pool.append(row)
                local_artists.add(artist)
                if len(pool) >= self.tail_depth:
                    break
            specialist_pools.append(pool)
        for position in range(
            self.tail_stride - 1, self.tail_depth, self.tail_stride
        ):
            for pool in specialist_pools:
                if position < len(pool):
                    accept(pool[position])
                if len(selected) >= n:
                    return selected

        # Preserve a complete response even when stratified candidates overlap.
        for row in self.base_ranker.rank(
            query_row, "production_baseline", n=max(n, 50)
        ):
            accept(row)
            if len(selected) >= n:
                break
        return selected


def build_locked_rankings(
    ranker: AudioCandidateUnionRanker,
    final_manifest_path: Path,
    method_manifest_path: Path,
    output_path: Path,
) -> Dict[str, Any]:
    """Build target-agnostic rankings for a method that is already locked."""
    method_sha = hashlib.sha256(method_manifest_path.read_bytes()).hexdigest()
    manifest = json.loads(final_manifest_path.read_text(encoding="utf-8"))
    resolver = V5PairResolver(
        ranker.recommender.titles, ranker.recommender.artists
    )
    records = []
    for pair in manifest["pairs"]:
        query_row = resolver.query_row(pair["query"])
        if query_row is None:
            raise ValueError(f"Missing locked query {pair['id']}")
        rows = ranker.rank(query_row, n=50)
        records.append({
            "pair_id": pair["id"],
            "query": dict(pair["query"]),
            "ranking": [
                {
                    "rank": position,
                    "row": int(row),
                    "track_id": int(ranker.recommender.track_ids[row]),
                    "title": str(ranker.recommender.titles[row]),
                    "artist": str(ranker.recommender.artists[row]),
                }
                for position, row in enumerate(rows, 1)
            ],
        })
    result = {
        "schema_version": 1,
        "created_at": _iso_now(),
        "target_labels_compared": False,
        "method_manifest_sha256": method_sha,
        "records": records,
    }
    _write_json_atomic(output_path, result)
    return result

class AudioExperimentPipeline:
    """Iteration-4 audio experiment orchestrator (Protocol v5).

    Coordinates all training, extraction, and evaluation phases.  Each public
    method validates that it will not violate the protocol (final-split
    isolation, no metadata as model inputs, etc.).

    Quick smoke-test with synthetic data (no GPU, no real files required)::

        from soundalike.ml.audio_experiments import (
            AudioExperimentPipeline, ExperimentConfig
        )
        import numpy as np, json
        from pathlib import Path

        cfg = ExperimentConfig(benchmark_path="benchmarks/soundalike_pairs.v5.json")
        pipe = AudioExperimentPipeline(cfg)

        N = 50
        emb = np.random.randn(N, 64).astype(np.float16)
        emb /= np.linalg.norm(emb.astype(np.float32), axis=1, keepdims=True)
        result = pipe.evaluate_dev(
            embeddings=emb,
            titles=["title_%d" % i for i in range(N)],
            artists=["artist_%d" % i for i in range(N)],
        )
        print(result["metrics"])
    """

    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        self._guard: Optional[LeakageGuard] = None
        self._resource_logs: List[ResourceLog] = []

    # ------------------------------------------------------------------
    # Lazy guard
    # ------------------------------------------------------------------

    def guard(self) -> LeakageGuard:
        """Return a cached LeakageGuard for the configured benchmark path."""
        if self._guard is None:
            self._guard = LeakageGuard(self.cfg.benchmark_path)
        return self._guard

    # ------------------------------------------------------------------
    # Encoder factory
    # ------------------------------------------------------------------

    def _make_encoder(self, phase: str) -> "nn.Module":
        from .model import ResNetAudioEncoder

        if phase == "supcon":
            return ResNetAudioEncoder(
                embedding_dim=self.cfg.supcon_embedding_dim,
                width=self.cfg.supcon_encoder_width,
                pool_type=self.cfg.supcon_pool_type,
            )
        if phase == "byol":
            return ResNetAudioEncoder(
                embedding_dim=self.cfg.byol_embedding_dim,
                width=self.cfg.byol_encoder_width,
                pool_type=self.cfg.byol_pool_type,
            )
        if phase == "distill":
            return ResNetAudioEncoder(
                embedding_dim=self.cfg.distill_embedding_dim,
                width=self.cfg.distill_encoder_width,
                pool_type=self.cfg.distill_pool_type,
            )
        raise ValueError(f"Unknown phase: {phase!r}")

    def _peak_gpu_mb(self) -> float:
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 / 1024
        return 0.0

    # ------------------------------------------------------------------
    # Phase 1: Supervised contrastive FMA training
    # ------------------------------------------------------------------

    def train_supcon(
        self,
        encoder: Optional["nn.Module"] = None,
        max_steps: Optional[int] = None,
        out_dir: Optional[Path] = None,
    ) -> Tuple["nn.Module", CheckpointMeta]:
        """Genre-supervised cross-artist contrastive training on FMA.

        Parameters
        ----------
        encoder : if None, a fresh ResNetAudioEncoder is instantiated.
        max_steps : early-stop for unit tests; None = train to completion.
        out_dir : if set, checkpoint + metadata are saved here.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for train_supcon")
        from torch.utils.data import DataLoader

        torch.manual_seed(self.cfg.train_seed)
        np.random.seed(self.cfg.train_seed)

        if encoder is None:
            encoder = self._make_encoder("supcon")

        device = _best_device()
        mem_fmt = (
            torch.channels_last if device == "cuda" else torch.contiguous_format
        )
        enc = encoder.to(device, memory_format=mem_fmt)
        criterion = SupConLoss(
            temperature=self.cfg.supcon_temperature,
            cross_artist=self.cfg.supcon_cross_artist,
        )
        opt = torch.optim.AdamW(
            enc.parameters(),
            lr=self.cfg.supcon_lr,
            weight_decay=self.cfg.supcon_weight_decay,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

        ds = FMAPackedDataset(
            packed_path=self.cfg.fma_packed_path,
            crop_frames=self.cfg.supcon_crop_frames,
            seed=self.cfg.train_seed,
            guard=self.guard(),
            split="train",
            val_frac=self.cfg.val_frac,
            test_frac=self.cfg.test_frac,
        )
        loader = DataLoader(
            ds,
            batch_size=self.cfg.supcon_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )

        t0 = time.perf_counter()
        last_loss = float("nan")
        steps = 0

        enc.train()
        epochs_completed = 0
        stop = False
        for epoch in range(self.cfg.supcon_max_epochs):
            # Make augmentation order reproducible but different each epoch.
            ds._rng = np.random.default_rng(self.cfg.train_seed + epoch)
            for v1, v2, genre_idx, artists in loader:
                v1 = v1.to(device, memory_format=mem_fmt, non_blocking=True)
                v2 = v2.to(device, memory_format=mem_fmt, non_blocking=True)
                labels = genre_idx.tolist()
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    z1 = enc(v1)
                    z2 = enc(v2)
                    # Concatenate both views: 2B samples per batch.
                    z = torch.cat([z1, z2], dim=0)
                    all_labels = labels + labels
                    all_artists = list(artists) + list(artists)
                    loss = criterion(z, all_labels, all_artists)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                last_loss = float(loss.item())
                steps += 1
                if max_steps is not None and steps >= max_steps:
                    stop = True
                    break
            epochs_completed = epoch + 1
            if stop:
                break

        wall = time.perf_counter() - t0
        meta = CheckpointMeta(
            phase="supcon",
            epoch=steps,
            loss=last_loss,
            config_hash=self.cfg.config_hash(),
            timestamp=_iso_now(),
            wall_seconds=wall,
            extra={
                "steps": steps,
                "epochs_completed": epochs_completed,
                "device": device,
            },
        )
        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": enc.state_dict(),
                    "embedding_dim": self.cfg.supcon_embedding_dim,
                },
                out_dir / "supcon_encoder.pt",
            )
            meta.save(out_dir / "supcon_meta.json")

        self._resource_logs.append(
            ResourceLog(
                phase="supcon",
                wall_seconds=wall,
                peak_gpu_mb=self._peak_gpu_mb(),
                peak_ram_mb=0.0,
                n_rows=steps * self.cfg.supcon_batch_size,
                rows_per_sec=(steps * self.cfg.supcon_batch_size)
                / max(wall, 1e-9),
            )
        )
        return enc, meta

    # ------------------------------------------------------------------
    # Phase 2: BYOL self-supervised FMA training
    # ------------------------------------------------------------------

    def train_byol(
        self,
        model: Optional[BYOLModel] = None,
        max_steps: Optional[int] = None,
        out_dir: Optional[Path] = None,
    ) -> Tuple[BYOLModel, CheckpointMeta]:
        """BYOL non-contrastive self-supervised training on FMA."""
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for train_byol")
        from torch.utils.data import DataLoader

        torch.manual_seed(self.cfg.train_seed)

        if model is None:
            model = BYOLModel(
                embedding_dim=self.cfg.byol_embedding_dim,
                width=self.cfg.byol_encoder_width,
                pool_type=self.cfg.byol_pool_type,
                projection_dim=self.cfg.byol_projection_dim,
                prediction_dim=self.cfg.byol_prediction_dim,
                ema_decay=self.cfg.byol_ema_decay,
            )

        device = _best_device()
        mem_fmt = (
            torch.channels_last if device == "cuda" else torch.contiguous_format
        )
        model = model.to(device)

        # Only online parameters receive gradients.
        online_params = (
            list(model.online_encoder.parameters())
            + list(model.online_projector.parameters())
            + list(model.online_predictor.parameters())
        )
        opt = torch.optim.AdamW(
            online_params,
            lr=self.cfg.byol_lr,
            weight_decay=self.cfg.byol_weight_decay,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

        ds = FMAPackedDataset(
            packed_path=self.cfg.fma_packed_path,
            crop_frames=self.cfg.byol_crop_frames,
            seed=self.cfg.train_seed,
            guard=self.guard(),
            split="train",
            val_frac=self.cfg.val_frac,
            test_frac=self.cfg.test_frac,
        )
        loader = DataLoader(
            ds,
            batch_size=self.cfg.byol_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )

        t0 = time.perf_counter()
        last_loss = float("nan")
        steps = 0

        model.train()
        epochs_completed = 0
        stop = False
        for epoch in range(self.cfg.byol_max_epochs):
            ds._rng = np.random.default_rng(self.cfg.train_seed + epoch)
            for v1, v2, _genre, _artist in loader:
                v1 = v1.to(device, memory_format=mem_fmt, non_blocking=True)
                v2 = v2.to(device, memory_format=mem_fmt, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    p1, p2, t1, t2 = model(v1, v2)
                    loss = byol_loss(p1, p2, t1, t2)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                model.update_target()
                last_loss = float(loss.item())
                steps += 1
                if max_steps is not None and steps >= max_steps:
                    stop = True
                    break
            epochs_completed = epoch + 1
            if stop:
                break

        wall = time.perf_counter() - t0
        meta = CheckpointMeta(
            phase="byol",
            epoch=steps,
            loss=last_loss,
            config_hash=self.cfg.config_hash(),
            timestamp=_iso_now(),
            wall_seconds=wall,
            extra={
                "steps": steps,
                "epochs_completed": epochs_completed,
                "device": device,
            },
        )
        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.online_encoder.state_dict(),
                    "embedding_dim": self.cfg.byol_embedding_dim,
                },
                out_dir / "byol_encoder.pt",
            )
            meta.save(out_dir / "byol_meta.json")

        self._resource_logs.append(
            ResourceLog(
                phase="byol",
                wall_seconds=wall,
                peak_gpu_mb=self._peak_gpu_mb(),
                peak_ram_mb=0.0,
                n_rows=steps * self.cfg.byol_batch_size,
                rows_per_sec=(steps * self.cfg.byol_batch_size)
                / max(wall, 1e-9),
            )
        )
        return model, meta

    # ------------------------------------------------------------------
    # Phase 3: Audio-teacher distillation
    # ------------------------------------------------------------------

    def train_distill(
        self,
        mel_matrix: np.ndarray,
        sonic_array: np.ndarray,
        clap_array: np.ndarray,
        catalog_artists: Sequence[str],
        encoder: Optional["nn.Module"] = None,
        max_steps: Optional[int] = None,
        out_dir: Optional[Path] = None,
    ) -> Tuple["nn.Module", "nn.Module", CheckpointMeta]:
        """Distillation from sonic+clap teacher, excluding all benchmark artists.

        Parameters
        ----------
        mel_matrix : (N, n_mels, frames) mel spectrograms for catalog tracks.
        sonic_array : (N, sonic_dim) e.g. research_clap64_cal.f16.npy loaded.
        clap_array : (N, clap_dim) e.g. research_clap512.f16.npy loaded.
        catalog_artists : (N,) artist strings aligned to the arrays.
        encoder : if None, a fresh ResNetAudioEncoder is instantiated.
        max_steps : early-stop for unit tests.
        out_dir : checkpoint save directory.

        Returns
        -------
        (encoder, projection_head, checkpoint_meta)
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for train_distill")
        from torch.utils.data import DataLoader

        torch.manual_seed(self.cfg.train_seed)

        teacher_matrix = _build_teacher(sonic_array, clap_array)
        teacher_dim = int(teacher_matrix.shape[1])

        if encoder is None:
            encoder = self._make_encoder("distill")

        device = _best_device()
        mem_fmt = (
            torch.channels_last if device == "cuda" else torch.contiguous_format
        )
        enc = encoder.to(device, memory_format=mem_fmt)
        proj = DistillationProjection(
            student_dim=self.cfg.distill_embedding_dim,
            hidden_dim=self.cfg.distill_proj_hidden,
            teacher_dim=teacher_dim,
        ).to(device)

        opt = torch.optim.AdamW(
            list(enc.parameters()) + list(proj.parameters()),
            lr=self.cfg.distill_lr,
            weight_decay=self.cfg.distill_weight_decay,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

        ds = DistillationDataset(
            mel_matrix=mel_matrix,
            teacher_matrix=teacher_matrix,
            artists=list(catalog_artists),
            guard=self.guard(),
            crop_frames=self.cfg.distill_crop_frames,
            seed=self.cfg.train_seed,
        )
        loader = DataLoader(
            ds,
            batch_size=self.cfg.distill_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )

        t0 = time.perf_counter()
        last_loss = float("nan")
        steps = 0

        enc.train()
        proj.train()
        epochs_completed = 0
        stop = False
        for epoch in range(self.cfg.distill_max_epochs):
            ds._rng = np.random.default_rng(self.cfg.train_seed + epoch)
            for mel_crop, teacher_vec in loader:
                mel_crop = mel_crop.to(
                    device, memory_format=mem_fmt, non_blocking=True
                )
                teacher_vec = teacher_vec.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                    z = enc(mel_crop, normalize=False)
                    z_proj = proj(z)
                    loss = distillation_loss(
                        z_proj, teacher_vec, self.cfg.distill_temperature
                    )
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                last_loss = float(loss.item())
                steps += 1
                if max_steps is not None and steps >= max_steps:
                    stop = True
                    break
            epochs_completed = epoch + 1
            if stop:
                break

        wall = time.perf_counter() - t0
        meta = CheckpointMeta(
            phase="distill",
            epoch=steps,
            loss=last_loss,
            config_hash=self.cfg.config_hash(),
            timestamp=_iso_now(),
            wall_seconds=wall,
            extra={
                "steps": steps,
                "epochs_completed": epochs_completed,
                "device": device,
                "excluded_rows": ds._excluded,
                "teacher_dim": teacher_dim,
            },
        )
        if out_dir is not None:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": enc.state_dict(),
                    "projection_state_dict": proj.state_dict(),
                    "embedding_dim": self.cfg.distill_embedding_dim,
                    "teacher_dim": teacher_dim,
                },
                out_dir / "distill_encoder.pt",
            )
            meta.save(out_dir / "distill_meta.json")

        self._resource_logs.append(
            ResourceLog(
                phase="distill",
                wall_seconds=wall,
                peak_gpu_mb=self._peak_gpu_mb(),
                peak_ram_mb=0.0,
                n_rows=steps * self.cfg.distill_batch_size,
                rows_per_sec=(steps * self.cfg.distill_batch_size)
                / max(wall, 1e-9),
            )
        )
        return enc, proj, meta

    # ------------------------------------------------------------------
    # Phase 4: Catalog embedding extraction
    # ------------------------------------------------------------------

    def extract_catalog(
        self,
        encoder: "nn.Module",
        mel_matrix: np.ndarray,
        crop_frames: int = 256,
    ) -> np.ndarray:
        """Extract (N, dim) float16 embeddings for all catalog tracks.

        Parameters
        ----------
        encoder : trained ResNetAudioEncoder (in eval mode).
        mel_matrix : (N, n_mels, total_frames) mel spectrograms.
        crop_frames : centre-crop length used during inference.

        Returns
        -------
        (N, embedding_dim) float16 ndarray.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for extract_catalog")
        t0 = time.perf_counter()
        extractor = CatalogExtractor(
            encoder=encoder,
            batch_size=self.cfg.extract_batch_size,
            device=_best_device(),
        )
        emb = extractor.extract(mel_matrix, crop_frames=crop_frames)
        wall = time.perf_counter() - t0
        self._resource_logs.append(
            ResourceLog(
                phase="extract",
                wall_seconds=wall,
                peak_gpu_mb=self._peak_gpu_mb(),
                peak_ram_mb=0.0,
                n_rows=len(mel_matrix),
                rows_per_sec=len(mel_matrix) / max(wall, 1e-9),
            )
        )
        return emb  # (N, dim) float16

    # ------------------------------------------------------------------
    # Phase 5: Development evaluation
    # ------------------------------------------------------------------

    def evaluate_dev(
        self,
        embeddings: np.ndarray,
        titles: Sequence[str],
        artists: Sequence[str],
        method_name: str = "experiment",
        bootstrap: bool = False,
    ) -> Dict[str, Any]:
        """Evaluate on DEVELOPMENT pairs only.  Final split is never touched.

        Parameters
        ----------
        embeddings : (N_catalog, dim) float16/32 — must be L2-normalised.
        titles : catalog title strings aligned to embeddings.
        artists : catalog artist strings aligned to embeddings.
        method_name : label for the result.
        bootstrap : compute 20 000-iteration bootstrap CI on primary.

        Returns
        -------
        Dict with keys: method, metrics, pairs, n_dev_pairs, n_rankable,
        timestamp, wall_seconds, config_hash.
        """
        evaluator = DevEvaluator(
            guard=self.guard(),
            titles=list(titles),
            artists=list(artists),
            cfg=self.cfg,
        )
        t0 = time.perf_counter()
        result = evaluator.evaluate(
            embeddings=embeddings,
            method_name=method_name,
            bootstrap=bootstrap,
        )
        wall = time.perf_counter() - t0
        result["wall_seconds"] = wall
        result["config_hash"] = self.cfg.config_hash()
        return result

    # ------------------------------------------------------------------
    # Phase 6: Late-interaction reranking
    # ------------------------------------------------------------------

    def extract_windows(
        self,
        encoder: "nn.Module",
        mel_matrix: np.ndarray,
    ) -> np.ndarray:
        """Extract (N, n_windows, dim) float16 window embeddings."""
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for extract_windows")
        reranker = LateInteractionReranker(
            encoder=encoder,
            n_windows=self.cfg.late_interaction_windows,
            window_frames=self.cfg.late_interaction_window_frames,
            batch_size=self.cfg.extract_batch_size,
        )
        return reranker.extract_windows(mel_matrix)

    def rerank_late_interaction(
        self,
        query_row: int,
        candidate_rows: Sequence[int],
        all_window_embeddings: np.ndarray,  # (N_catalog, n_windows, dim)
    ) -> List[int]:
        """Rerank candidate_rows by 4-window MaxSim; return sorted row list."""
        query_wins = all_window_embeddings[query_row]  # (n_windows, dim)
        # Build a temporary reranker (no encoder needed for scoring only).
        encoder_stub = None
        rr = LateInteractionReranker.__new__(LateInteractionReranker)
        rr.n_windows = self.cfg.late_interaction_windows
        rr.window_frames = self.cfg.late_interaction_window_frames
        rr._batch_size = self.cfg.extract_batch_size
        rr._device = _best_device()
        return LateInteractionReranker.rerank(
            rr, query_wins, list(candidate_rows), all_window_embeddings
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def resource_summary(self) -> List[Dict[str, Any]]:
        """Return all recorded ResourceLog entries as dicts."""
        return [r.to_dict() for r in self._resource_logs]

    def save_result(
        self, result: Dict[str, Any], path: Path
    ) -> None:
        """Atomically write an evaluation result to JSON."""
        _write_json_atomic(Path(path), result)
