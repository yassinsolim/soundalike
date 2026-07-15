"""Bounded, resumable full-track extraction with optional pinned PyAV decoding."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from .fulltrack_store import FullTrackStore, TrackArtifacts, stable_json_sha256
from .jamendo_fulltrack import (
    EVIDENCE_SCOPE,
    JamendoContext,
    JamendoTrack,
    JamendoValidationError,
    load_jamendo_context,
    sha256_file,
)


PINNED_AV_VERSION = "18.0.0"
PINNED_CLAP_VERSION = "1.1.7"
CLAP_CHECKPOINT_SHA256 = (
    "8053c9775516af2f4902e1e8281e356cc1bf7a85e8b761908170767b77c3f037"
)


class FullTrackExtractionError(RuntimeError):
    """Decoder/model capability or bounded extraction failure."""


@contextmanager
def _offline_model_environment() -> Iterator[None]:
    values = {
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@dataclass(frozen=True)
class DecoderWindow:
    index: int
    start_sample: int
    samples: np.ndarray
    valid_samples: int


@dataclass(frozen=True)
class ExtractionConfig:
    sample_rate: int = 48_000
    window_seconds: float = 10.0
    hop_seconds: float = 5.0
    short_track_policy: str = "repeatpad"
    decoder_chunk_seconds: float = 2.0
    model_batch_size: int = 32
    max_windows_per_track: int = 2_048
    repetition_sections: int = 32
    salient_sections: int = 32
    section_min_gap_windows: int = 2
    verify_source_sha256: bool = True
    shard_tracks: int = 256

    @property
    def window_samples(self) -> int:
        return int(round(self.sample_rate * self.window_seconds))

    @property
    def hop_samples(self) -> int:
        return int(round(self.sample_rate * self.hop_seconds))

    @property
    def decoder_chunk_samples(self) -> int:
        return int(round(self.sample_rate * self.decoder_chunk_seconds))

    def validate(self) -> None:
        if self.sample_rate <= 0:
            raise FullTrackExtractionError("sample_rate must be positive")
        if self.window_samples <= 0 or not 0 < self.hop_samples <= self.window_samples:
            raise FullTrackExtractionError("require 0 < hop <= window")
        if self.short_track_policy not in ("repeatpad", "zero_pad", "reject"):
            raise FullTrackExtractionError("unsupported short-track policy")
        if self.decoder_chunk_samples <= 0:
            raise FullTrackExtractionError("decoder chunk size must be positive")
        if self.model_batch_size <= 0 or self.max_windows_per_track <= 0:
            raise FullTrackExtractionError("batch and window bounds must be positive")
        if self.repetition_sections <= 0 or self.salient_sections <= 0:
            raise FullTrackExtractionError("section budgets must be positive")
        if self.section_min_gap_windows < 0 or self.shard_tracks <= 0:
            raise FullTrackExtractionError("invalid section gap or shard size")

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)

    def sha256(self) -> str:
        self.validate()
        return stable_json_sha256(self.as_dict())


@dataclass(frozen=True)
class ModelCapability:
    available: bool
    model_id: str
    checkpoint_path: Optional[str]
    checkpoint_sha256: Optional[str]
    package_versions: Dict[str, str]
    cuda_device: Optional[str]
    license: Optional[str]
    reasons: Tuple[str, ...]

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ExtractionSummary:
    processed_tracks: int
    completed_tracks: int
    pending_tracks: int
    decoded_samples: int
    windows: int
    wall_seconds: float
    evidence_scope: str = EVIDENCE_SCOPE


class AudioChunkDecoder(Protocol):
    """Incremental mono decoder contract."""

    def decode(
        self, path: Path, *, sample_rate: int, chunk_samples: int
    ) -> Iterable[np.ndarray]:
        ...


class MusicModelAdapter(Protocol):
    """Generic frozen music-window model; no second model is implied."""

    @property
    def model_id(self) -> str:
        ...

    @property
    def checkpoint_sha256(self) -> str:
        ...

    @property
    def embedding_dim(self) -> int:
        ...

    @property
    def sample_rate(self) -> int:
        ...

    @property
    def max_batch_size(self) -> int:
        ...

    def embed_windows(self, windows: np.ndarray) -> np.ndarray:
        ...


class PyAVAudioDecoder:
    """Decode with PyAV in bounded chunks; no WAV or other audio copy is written."""

    def __init__(self, *, required_version: str = PINNED_AV_VERSION) -> None:
        try:
            version = importlib.metadata.version("av")
        except importlib.metadata.PackageNotFoundError as exc:
            raise FullTrackExtractionError(
                'PyAV is unavailable; install the "fulltrack" optional extra'
            ) from exc
        if version != required_version:
            raise FullTrackExtractionError(
                f"PyAV must be exactly {required_version}, found {version}"
            )

    def decode(
        self, path: Path, *, sample_rate: int, chunk_samples: int
    ) -> Iterator[np.ndarray]:
        if sample_rate <= 0 or chunk_samples <= 0:
            raise FullTrackExtractionError("invalid decoder sample/chunk size")
        try:
            import av
        except ImportError as exc:  # pragma: no cover - constructor gates this
            raise FullTrackExtractionError("PyAV import failed") from exc
        pending = np.empty(0, dtype=np.float32)
        try:
            container = av.open(str(path), mode="r")
        except (av.error.FFmpegError, OSError) as exc:
            raise FullTrackExtractionError(f"cannot open audio {path}: {exc}") from exc
        with container:
            audio_streams = list(container.streams.audio)
            if not audio_streams:
                raise FullTrackExtractionError(f"audio stream is missing: {path}")
            stream = audio_streams[0]
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=sample_rate)
            try:
                frames = container.decode(stream)
                for frame in frames:
                    for converted in resampler.resample(frame):
                        samples = np.asarray(
                            converted.to_ndarray(), dtype=np.float32
                        ).reshape(-1)
                        if not np.all(np.isfinite(samples)):
                            raise FullTrackExtractionError(
                                f"decoder produced non-finite audio: {path}"
                            )
                        pending = np.concatenate((pending, samples))
                        while len(pending) >= chunk_samples:
                            yield pending[:chunk_samples].copy()
                            pending = pending[chunk_samples:]
                for converted in resampler.resample(None):
                    samples = np.asarray(
                        converted.to_ndarray(), dtype=np.float32
                    ).reshape(-1)
                    pending = np.concatenate((pending, samples))
                    while len(pending) >= chunk_samples:
                        yield pending[:chunk_samples].copy()
                        pending = pending[chunk_samples:]
            except av.error.FFmpegError as exc:
                raise FullTrackExtractionError(
                    f"audio decode failed for {path}: {exc}"
                ) from exc
        if len(pending):
            if not np.all(np.isfinite(pending)):
                raise FullTrackExtractionError(
                    f"decoder produced non-finite tail audio: {path}"
                )
            yield pending.copy()


def _pad_short(samples: np.ndarray, size: int, policy: str) -> np.ndarray:
    if not len(samples):
        raise FullTrackExtractionError("decoded track is empty")
    if policy == "reject":
        raise FullTrackExtractionError(
            f"decoded track has {len(samples)} samples, shorter than window {size}"
        )
    if policy == "zero_pad":
        return np.pad(samples, (0, size - len(samples))).astype(np.float32)
    if policy == "repeatpad":
        repeats = int(math.ceil(size / len(samples)))
        return np.tile(samples, repeats)[:size].astype(np.float32)
    raise FullTrackExtractionError(f"unknown short-track policy {policy!r}")


def iter_overlapping_windows(
    chunks: Iterable[np.ndarray],
    *,
    window_samples: int,
    hop_samples: int,
    short_track_policy: str = "repeatpad",
    max_chunk_samples: Optional[int] = None,
    max_windows: int = 2_048,
) -> Iterator[DecoderWindow]:
    """Stream deterministic regular windows plus one end-aligned tail window."""
    if window_samples <= 0 or not 0 < hop_samples <= window_samples:
        raise FullTrackExtractionError("require 0 < hop_samples <= window_samples")
    buffer = np.empty(0, dtype=np.float32)
    tail = np.empty(0, dtype=np.float32)
    total_samples = 0
    base_sample = 0
    last_start: Optional[int] = None
    count = 0
    for raw_chunk in chunks:
        chunk = np.asarray(raw_chunk, dtype=np.float32).reshape(-1)
        if not len(chunk):
            continue
        if max_chunk_samples is not None and len(chunk) > max_chunk_samples:
            raise FullTrackExtractionError(
                f"decoder chunk exceeds bound: {len(chunk)} > {max_chunk_samples}"
            )
        if not np.all(np.isfinite(chunk)):
            raise FullTrackExtractionError("decoder chunk contains non-finite samples")
        total_samples += len(chunk)
        buffer = np.concatenate((buffer, chunk))
        tail = np.concatenate((tail, chunk))[-window_samples:]
        while len(buffer) >= window_samples:
            if count >= max_windows:
                raise FullTrackExtractionError(
                    f"track exceeds max_windows_per_track={max_windows}"
                )
            yield DecoderWindow(
                count, base_sample, buffer[:window_samples].copy(), window_samples
            )
            count += 1
            last_start = base_sample
            buffer = buffer[hop_samples:]
            base_sample += hop_samples
    if total_samples == 0:
        raise FullTrackExtractionError("decoded track is empty")
    if total_samples < window_samples:
        if count:
            raise FullTrackExtractionError("internal short-track windowing error")
        yield DecoderWindow(
            0,
            0,
            _pad_short(tail, window_samples, short_track_policy),
            total_samples,
        )
        return
    end_start = total_samples - window_samples
    if last_start != end_start:
        if count >= max_windows:
            raise FullTrackExtractionError(
                f"track exceeds max_windows_per_track={max_windows}"
            )
        if len(tail) != window_samples:
            raise FullTrackExtractionError("internal tail-window length error")
        yield DecoderWindow(count, end_start, tail.copy(), window_samples)


def normalize_rows(value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.ndim != 2 or not len(matrix) or not np.all(np.isfinite(matrix)):
        raise FullTrackExtractionError("model embeddings must be a finite non-empty matrix")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise FullTrackExtractionError("model returned a zero embedding")
    return (matrix / norms).astype(np.float32)


def length_aware_global_pool(
    windows: np.ndarray,
    starts: Sequence[int],
    *,
    decoded_samples: int,
    window_samples: int,
) -> np.ndarray:
    """Pool by each window center's unique temporal coverage, then normalize."""
    embeddings = normalize_rows(windows)
    positions = np.asarray(starts, dtype=np.int64)
    if len(positions) != len(embeddings) or decoded_samples <= 0:
        raise FullTrackExtractionError("pooling positions are misaligned")
    if len(positions) == 1:
        return embeddings[0].copy()
    centers = np.minimum(
        positions.astype(np.float64) + window_samples / 2.0,
        float(decoded_samples),
    )
    boundaries = np.empty(len(centers) + 1, dtype=np.float64)
    boundaries[0] = 0.0
    boundaries[-1] = float(decoded_samples)
    boundaries[1:-1] = (centers[:-1] + centers[1:]) / 2.0
    weights = np.diff(boundaries)
    if np.any(weights <= 0) or not np.isclose(weights.sum(), decoded_samples):
        raise FullTrackExtractionError("invalid temporal pooling weights")
    pooled = np.sum(embeddings * weights[:, None], axis=0)
    norm = float(np.linalg.norm(pooled))
    if norm <= 1e-12:
        raise FullTrackExtractionError("global pooling produced a zero vector")
    return (pooled / norm).astype(np.float32)


def fixed_budget_indices(count: int, budget: int) -> np.ndarray:
    """Deterministically select exactly ``budget`` indices, repeating if needed."""
    if count <= 0 or budget <= 0:
        raise FullTrackExtractionError("count and budget must be positive")
    return np.rint(np.linspace(0, count - 1, num=budget)).astype(np.int64)


def _diverse_top_indices(
    scores: np.ndarray, budget: int, minimum_gap: int
) -> np.ndarray:
    if budget <= 0 or not len(scores):
        return np.empty(0, dtype=np.int64)
    order = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))
    selected: List[int] = []
    for index in order:
        if all(abs(index - other) >= minimum_gap for other in selected):
            selected.append(index)
            if len(selected) == min(budget, len(scores)):
                break
    if len(selected) < min(budget, len(scores)):
        for index in order:
            if index not in selected:
                selected.append(index)
                if len(selected) == min(budget, len(scores)):
                    break
    return np.asarray(selected, dtype=np.int64)


def select_repeated_section_indices(
    windows: np.ndarray,
    *,
    budget: int,
    minimum_gap: int = 2,
    similarity_block: int = 256,
) -> np.ndarray:
    """Return distinct deterministic source-window indices with non-local recurrence."""
    embeddings = normalize_rows(windows)
    count = len(embeddings)
    if budget <= 0:
        return np.empty(0, dtype=np.int64)
    if similarity_block <= 0:
        raise FullTrackExtractionError("similarity block must be positive")
    scores = np.full(count, -1.0, dtype=np.float32)
    for query_start in range(0, count, similarity_block):
        query_end = min(query_start + similarity_block, count)
        best = np.full(query_end - query_start, -1.0, dtype=np.float32)
        query_indices = np.arange(query_start, query_end)[:, None]
        for candidate_start in range(0, count, similarity_block):
            candidate_end = min(candidate_start + similarity_block, count)
            similarities = (
                embeddings[query_start:query_end]
                @ embeddings[candidate_start:candidate_end].T
            )
            candidate_indices = np.arange(candidate_start, candidate_end)[None, :]
            similarities[
                np.abs(query_indices - candidate_indices) < minimum_gap
            ] = -1.0
            best = np.maximum(best, np.max(similarities, axis=1))
        scores[query_start:query_end] = best
    return _diverse_top_indices(scores, budget, minimum_gap)


def select_repeated_sections(
    windows: np.ndarray,
    *,
    budget: int,
    minimum_gap: int = 2,
    similarity_block: int = 256,
) -> np.ndarray:
    """Pick windows with strong non-local recurrence using fixed-size blocks."""
    embeddings = normalize_rows(windows)
    indices = select_repeated_section_indices(
        embeddings,
        budget=budget,
        minimum_gap=minimum_gap,
        similarity_block=similarity_block,
    )
    return embeddings[indices]


def select_salient_section_indices(
    windows: np.ndarray,
    global_embedding: np.ndarray,
    *,
    budget: int,
    minimum_gap: int = 2,
) -> np.ndarray:
    """Return distinct deterministic distinctive/change-heavy source indices."""
    embeddings = normalize_rows(windows)
    if budget <= 0:
        return np.empty(0, dtype=np.int64)
    global_value = np.asarray(global_embedding, dtype=np.float32).reshape(-1)
    global_value = global_value / max(float(np.linalg.norm(global_value)), 1e-12)
    novelty = 1.0 - embeddings @ global_value
    changes = np.zeros(len(embeddings), dtype=np.float32)
    if len(embeddings) > 1:
        adjacent = 1.0 - np.sum(embeddings[1:] * embeddings[:-1], axis=1)
        changes[:-1] = np.maximum(changes[:-1], adjacent)
        changes[1:] = np.maximum(changes[1:], adjacent)
    scores = novelty + 0.5 * changes
    return _diverse_top_indices(scores, budget, minimum_gap)


def select_salient_sections(
    windows: np.ndarray,
    global_embedding: np.ndarray,
    *,
    budget: int,
    minimum_gap: int = 2,
) -> np.ndarray:
    """Pick deterministic distinctive/change-heavy windows."""
    embeddings = normalize_rows(windows)
    indices = select_salient_section_indices(
        embeddings,
        global_embedding,
        budget=budget,
        minimum_gap=minimum_gap,
    )
    return embeddings[indices]


class FrozenClapAdapter:
    """Frozen LAION-CLAP music-audio adapter with a no-download capability gate."""

    MODEL_ID = "laion_clap_htsat_tiny_music_audioset_630k_nonfusion"

    @classmethod
    def capability(cls, checkpoint_path: Optional[Path] = None) -> ModelCapability:
        reasons = []
        versions: Dict[str, str] = {}
        cuda_device: Optional[str] = None
        license_name: Optional[str] = None
        resolved_checkpoint: Optional[Path] = None
        checkpoint_hash: Optional[str] = None
        try:
            versions["laion-clap"] = importlib.metadata.version("laion-clap")
        except importlib.metadata.PackageNotFoundError:
            reasons.append("laion-clap is not installed")
        else:
            if versions["laion-clap"] != PINNED_CLAP_VERSION:
                reasons.append(
                    f"laion-clap must be {PINNED_CLAP_VERSION}, "
                    f"found {versions['laion-clap']}"
                )
            metadata = importlib.metadata.metadata("laion-clap")
            raw_license = metadata.get("License")
            if raw_license:
                license_name = (
                    "CC0-1.0 (package metadata)"
                    if "CC0 1.0 Universal" in raw_license
                    else raw_license.splitlines()[0]
                )
            spec = importlib.util.find_spec("laion_clap")
            if checkpoint_path is None and spec is not None and spec.origin:
                resolved_checkpoint = Path(spec.origin).resolve().parent / (
                    "630k-audioset-best.pt"
                )
            elif checkpoint_path is not None:
                resolved_checkpoint = Path(checkpoint_path).resolve()
        try:
            versions["torch"] = importlib.metadata.version("torch")
            import torch
        except (importlib.metadata.PackageNotFoundError, ImportError):
            reasons.append("torch is not installed")
        else:
            if not torch.cuda.is_available():
                reasons.append("CUDA is unavailable")
            else:
                cuda_device = torch.cuda.get_device_name(0)
        if resolved_checkpoint is None or not resolved_checkpoint.is_file():
            reasons.append("the frozen CLAP checkpoint is not installed locally")
        else:
            checkpoint_hash = sha256_file(resolved_checkpoint)
            if checkpoint_hash != CLAP_CHECKPOINT_SHA256:
                reasons.append("the frozen CLAP checkpoint SHA-256 is not approved")
        return ModelCapability(
            available=not reasons,
            model_id=cls.MODEL_ID,
            checkpoint_path=(
                str(resolved_checkpoint) if resolved_checkpoint is not None else None
            ),
            checkpoint_sha256=checkpoint_hash,
            package_versions=versions,
            cuda_device=cuda_device,
            license=license_name,
            reasons=tuple(reasons),
        )

    def __init__(self, checkpoint_path: Optional[Path] = None) -> None:
        capability = self.capability(checkpoint_path)
        if not capability.available or capability.checkpoint_path is None:
            raise FullTrackExtractionError(
                "CLAP capability gate failed: " + "; ".join(capability.reasons)
            )
        with _offline_model_environment():
            import laion_clap
            import torch

            self._torch = torch
            self._checkpoint = Path(capability.checkpoint_path)
            self._model = laion_clap.CLAP_Module(enable_fusion=False, device="cuda")
            self._model.load_ckpt(ckpt=str(self._checkpoint), verbose=False)
        self._model.model.eval()

    @property
    def model_id(self) -> str:
        return self.MODEL_ID

    @property
    def checkpoint_sha256(self) -> str:
        return CLAP_CHECKPOINT_SHA256

    @property
    def embedding_dim(self) -> int:
        return 512

    @property
    def sample_rate(self) -> int:
        return 48_000

    @property
    def max_batch_size(self) -> int:
        return 96

    def embed_windows(self, windows: np.ndarray) -> np.ndarray:
        from laion_clap.hook import float32_to_int16, int16_to_float32
        from laion_clap.training.data import get_audio_features

        waveforms = np.asarray(windows, dtype=np.float32)
        if waveforms.ndim != 2 or waveforms.shape[1] != 480_000:
            raise FullTrackExtractionError(
                "frozen CLAP requires a batch of exact 10-second/48 kHz windows"
            )
        if len(waveforms) > self.max_batch_size:
            raise FullTrackExtractionError("CLAP batch exceeds the capability bound")
        features = []
        for waveform in waveforms:
            quantized = int16_to_float32(float32_to_int16(waveform))
            sample: Dict[str, object] = {}
            features.append(
                get_audio_features(
                    sample,
                    self._torch.from_numpy(quantized).float(),
                    480_000,
                    data_truncating="rand_trunc",
                    data_filling="repeatpad",
                    audio_cfg=self._model.model_cfg["audio_cfg"],
                    require_grad=False,
                )
            )
        with self._torch.inference_mode():
            embeddings = self._model.model.get_audio_embedding(features)
        return normalize_rows(embeddings.float().cpu().numpy())


def extract_track(
    track: JamendoTrack,
    *,
    decoder: AudioChunkDecoder,
    encoder: MusicModelAdapter,
    config: ExtractionConfig,
) -> TrackArtifacts:
    """Decode and embed one full track with bounded waveform/model batches."""
    config.validate()
    if encoder.sample_rate != config.sample_rate:
        raise FullTrackExtractionError("encoder/config sample-rate mismatch")
    if config.model_batch_size > encoder.max_batch_size:
        raise FullTrackExtractionError("configured batch exceeds encoder capability")
    if config.verify_source_sha256:
        actual_hash = sha256_file(track.audio_path)
        if actual_hash != track.expected_audio_sha256:
            raise FullTrackExtractionError(
                f"source SHA-256 drift for track {track.track_id}"
            )
    chunks = decoder.decode(
        track.audio_path,
        sample_rate=config.sample_rate,
        chunk_samples=config.decoder_chunk_samples,
    )
    generator = iter_overlapping_windows(
        chunks,
        window_samples=config.window_samples,
        hop_samples=config.hop_samples,
        short_track_policy=config.short_track_policy,
        max_chunk_samples=config.decoder_chunk_samples,
        max_windows=config.max_windows_per_track,
    )
    starts: List[int] = []
    embedding_batches: List[np.ndarray] = []
    waveform_batch: List[np.ndarray] = []
    short_decoded_samples: Optional[int] = None

    def flush_batch() -> None:
        if not waveform_batch:
            return
        values = encoder.embed_windows(np.stack(waveform_batch))
        if values.shape != (len(waveform_batch), encoder.embedding_dim):
            raise FullTrackExtractionError(
                f"encoder returned unexpected shape {values.shape}"
            )
        embedding_batches.append(normalize_rows(values))
        waveform_batch.clear()

    for window in generator:
        starts.append(window.start_sample)
        if window.valid_samples < config.window_samples:
            short_decoded_samples = window.valid_samples
        waveform_batch.append(window.samples)
        if len(waveform_batch) == config.model_batch_size:
            flush_batch()
    flush_batch()
    if not embedding_batches:
        raise FullTrackExtractionError("track produced no model windows")
    embeddings = np.concatenate(embedding_batches, axis=0)
    decoded_samples = (
        config.window_samples
        if len(starts) == 1
        else starts[-1] + config.window_samples
    )
    if short_decoded_samples is not None:
        decoded_samples = short_decoded_samples
    global_embedding = length_aware_global_pool(
        embeddings,
        starts,
        decoded_samples=decoded_samples,
        window_samples=config.window_samples,
    )
    repeated_indices = select_repeated_section_indices(
        embeddings,
        budget=config.repetition_sections,
        minimum_gap=config.section_min_gap_windows,
    )
    salient_indices = select_salient_section_indices(
        embeddings,
        global_embedding,
        budget=config.salient_sections,
        minimum_gap=config.section_min_gap_windows,
    )
    return TrackArtifacts(
        global_embedding=global_embedding,
        window_embeddings=embeddings,
        window_starts=np.asarray(starts, dtype=np.int64),
        repeated_sections=embeddings[repeated_indices],
        salient_sections=embeddings[salient_indices],
        repeated_indices=repeated_indices,
        salient_indices=salient_indices,
        decoded_samples=decoded_samples,
    )


def extract_context(
    context: JamendoContext,
    store: FullTrackStore,
    *,
    decoder: AudioChunkDecoder,
    encoder: MusicModelAdapter,
    config: ExtractionConfig,
    max_tracks: Optional[int] = None,
) -> ExtractionSummary:
    """Resume extraction in immutable context order."""
    started = time.perf_counter()
    pending = set(store.pending_track_ids())
    processed = 0
    decoded_samples = 0
    windows = 0
    for track in context.tracks:
        if track.track_id not in pending:
            continue
        artifacts = extract_track(
            track, decoder=decoder, encoder=encoder, config=config
        )
        store.write_track(
            track.track_id, track.expected_audio_sha256, artifacts
        )
        processed += 1
        decoded_samples += artifacts.decoded_samples
        windows += len(artifacts.window_embeddings)
        if max_tracks is not None and processed >= max_tracks:
            break
    store.flush()
    if store.pending_count == 0:
        store.seal()
    return ExtractionSummary(
        processed_tracks=processed,
        completed_tracks=store.completed_count,
        pending_tracks=store.pending_count,
        decoded_samples=decoded_samples,
        windows=windows,
        wall_seconds=time.perf_counter() - started,
    )


def _extract_command(args: argparse.Namespace) -> int:
    # The completion/provenance gate intentionally precedes CUDA model loading.
    context = load_jamendo_context(
        Path(args.metadata_root),
        Path(args.audio_root),
        Path(args.state_root),
        production=True,
    )
    config = ExtractionConfig(
        window_seconds=args.window_seconds,
        hop_seconds=args.hop_seconds,
        model_batch_size=args.batch_size,
        repetition_sections=args.repetition_sections,
        salient_sections=args.salient_sections,
        shard_tracks=args.shard_tracks,
    )
    config.validate()
    encoder = FrozenClapAdapter(
        Path(args.checkpoint) if args.checkpoint else None
    )
    decoder = PyAVAudioDecoder()
    with FullTrackStore(
        Path(args.output),
        track_ids=[track.track_id for track in context.tracks],
        source_hashes=[track.expected_audio_sha256 for track in context.tracks],
        source_fingerprint=context.source_fingerprint,
        config_sha256=config.sha256(),
        model_sha256=encoder.checkpoint_sha256,
        model_id=encoder.model_id,
        embedding_dim=encoder.embedding_dim,
        shard_tracks=config.shard_tracks,
        repetition_sections=config.repetition_sections,
        salient_sections=config.salient_sections,
    ) as store:
        summary = extract_context(
            context,
            store,
            decoder=decoder,
            encoder=encoder,
            config=config,
            max_tracks=args.max_tracks,
        )
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))
    return 0


def _capability_command(args: argparse.Namespace) -> int:
    capability = FrozenClapAdapter.capability(
        Path(args.checkpoint) if args.checkpoint else None
    )
    av_reason = None
    try:
        PyAVAudioDecoder()
    except FullTrackExtractionError as exc:
        av_reason = str(exc)
    output = capability.as_dict()
    output["pyav_version"] = (
        importlib.metadata.version("av")
        if importlib.util.find_spec("av") is not None
        else None
    )
    output["pyav_error"] = av_reason
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if capability.available and av_reason is None else 2


def _smoke_decode_command(args: argparse.Namespace) -> int:
    path = Path(args.audio).resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise FullTrackExtractionError("smoke input must be a concrete local file")
    expected = args.expected_sha256.lower()
    actual = sha256_file(path)
    if actual != expected:
        raise FullTrackExtractionError("smoke input SHA-256 does not match verified marker")
    config = ExtractionConfig()
    decoder = PyAVAudioDecoder()
    total = 0
    peak_chunk = 0
    chunks = 0
    for chunk in decoder.decode(
        path,
        sample_rate=config.sample_rate,
        chunk_samples=config.decoder_chunk_samples,
    ):
        total += len(chunk)
        peak_chunk = max(peak_chunk, len(chunk))
        chunks += 1
    print(
        json.dumps(
            {
                "path": str(path),
                "sha256": actual,
                "decoded_samples": total,
                "sample_rate": config.sample_rate,
                "chunks": chunks,
                "peak_chunk_samples": peak_chunk,
                "persistent_audio_outputs": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    capability = subparsers.add_parser("capability", help="validate local model/decode assets")
    capability.add_argument("--checkpoint")
    capability.set_defaults(handler=_capability_command)

    extract = subparsers.add_parser("extract", help="run production full-track extraction")
    extract.add_argument("--metadata-root", required=True)
    extract.add_argument("--audio-root", required=True)
    extract.add_argument("--state-root", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--checkpoint")
    extract.add_argument("--window-seconds", type=float, default=10.0)
    extract.add_argument("--hop-seconds", type=float, default=5.0)
    extract.add_argument("--batch-size", type=int, default=32)
    extract.add_argument("--repetition-sections", type=int, default=32)
    extract.add_argument("--salient-sections", type=int, default=32)
    extract.add_argument("--shard-tracks", type=int, default=256)
    extract.add_argument("--max-tracks", type=int)
    extract.set_defaults(handler=_extract_command)

    smoke = subparsers.add_parser(
        "smoke-decode", help="decode one already SHA-verified local track; write nothing"
    )
    smoke.add_argument("--audio", required=True)
    smoke.add_argument("--expected-sha256", required=True)
    smoke.set_defaults(handler=_smoke_decode_command)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (FullTrackExtractionError, JamendoValidationError) as exc:
        raise SystemExit(f"full-track extraction blocked: {exc}") from exc


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
