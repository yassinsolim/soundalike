"""Build and query a compact catalogue-wide artist affinity graph.

The primary graph is learned from Last.fm-360K user/artist play histories and
then projected onto every artist in the 272k-track catalogue.  Music4All remains
an independent sparse track-level candidate source.  Runtime assets contain no
users, play histories, API keys, or credentials.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import tarfile
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np

from .real_benchmark import normalize_text

LASTFM_360K_DOI = "https://doi.org/10.5281/zenodo.6090214"
LASTFM_360K_URL = (
    "https://zenodo.org/api/records/6090214/files/"
    "lastfm-dataset-360K.tar.gz/content"
)
LASTFM_360K_MD5 = "635e6ed3fc873aa4ba33aba0ebce02b1"
LASTFM_360K_LICENSE = (
    "Last.fm permission; non-commercial use only (Zenodo license other-nc)"
)
_MEMBER = "lastfm-dataset-360K/usersha1-artmbid-artname-plays.tsv"


@dataclass(frozen=True)
class GraphBuildConfig:
    vector_size: int = 64
    neighbors: int = 96
    epochs: int = 6
    window: int = 30
    negative: int = 15
    min_count: int = 2
    max_artists_per_user: int = 80
    workers: int = 20
    seed: int = 20260712


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _digest(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compact_full_graph(source: str | Path, output: str | Path) -> Path:
    """Write a deterministic, full-signal-only runtime graph asset."""
    source_path = Path(source)
    output_path = Path(output)
    required = (
        "artist_names",
        "track_artist_ids",
        "track_rows",
        "track_indptr",
        "source_mapped",
        "artist_audio",
        "full_indices",
        "full_weights",
    )
    source_sha256 = _digest(source_path)
    with np.load(source_path, allow_pickle=False) as asset:
        missing = [name for name in required if name not in asset.files]
        if missing:
            raise ValueError(
                "Cannot compact catalog graph; missing required arrays: "
                + ", ".join(missing)
            )
        arrays = {name: np.asarray(asset[name]) for name in required}

    artist_count = len(arrays["artist_names"])
    indices = arrays["full_indices"]
    if np.any(indices < -1) or np.any(indices >= artist_count):
        raise ValueError("Full graph contains an out-of-range artist index")
    index_dtype = (
        np.int16
        if artist_count <= np.iinfo(np.int16).max + 1
        else np.int32
    )
    metadata = {
        "schema_version": 2,
        "asset_type": "catalog_artist_graph_runtime",
        "available_variants": ["full"],
        "intended_signal": "full_unmasked",
        "full_unmasked_intended_signal": True,
        "masked_variants": {
            "included": False,
            "available_in": "research_source_only",
        },
        "source_sha256": source_sha256,
        "missing_variant_policy": "error",
        "silent_fallback": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        artist_names=arrays["artist_names"],
        track_artist_ids=arrays["track_artist_ids"],
        track_rows=arrays["track_rows"],
        track_indptr=arrays["track_indptr"],
        source_mapped=arrays["source_mapped"],
        artist_audio=np.asarray(
            _normalise_rows(arrays["artist_audio"]), dtype=np.float16
        ),
        full_indices=np.asarray(indices, dtype=index_dtype),
        full_weights=np.asarray(arrays["full_weights"], dtype=np.float16),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return output_path


def compact_dual_source_graph(
    source: str | Path,
    music4all_full: str | Path,
    output: str | Path,
) -> Path:
    """Add exact Music4All item2vec artist neighborhoods to a graph asset.

    Only catalogue-aligned artist ids and their deterministic top-96 cosine
    neighborhoods are written.  The item2vec vectors are build-time inputs and
    are deliberately absent from the runtime asset.
    """
    source_path = Path(source)
    music_path = Path(music4all_full)
    output_path = Path(output)
    required = (
        "artist_names",
        "track_artist_ids",
        "track_rows",
        "track_indptr",
        "source_mapped",
        "artist_audio",
        "full_indices",
        "full_weights",
    )
    with np.load(source_path, allow_pickle=False) as asset:
        missing = [name for name in required if name not in asset.files]
        if missing:
            raise ValueError(
                "Cannot build dual-source graph; missing graph arrays: "
                + ", ".join(missing)
            )
        arrays = {name: np.asarray(asset[name]) for name in required}

    with np.load(music_path, allow_pickle=False) as asset:
        if "artist_names" not in asset.files:
            raise ValueError("Music4All item2vec-full asset lacks artist_names")
        vector_key = (
            "artist_vectors" if "artist_vectors" in asset.files else "vectors"
        )
        if vector_key not in asset.files:
            raise ValueError("Music4All item2vec-full asset lacks artist vectors")
        music_names = np.asarray(asset["artist_names"])
        music_vectors = np.asarray(asset[vector_key], dtype=np.float32)
    if music_vectors.ndim != 2 or len(music_names) != len(music_vectors):
        raise ValueError("Music4All artist names and vectors are misaligned")

    graph_lookup = {
        normalize_text(str(name)): artist_id
        for artist_id, name in enumerate(arrays["artist_names"])
    }
    # Sorting by catalogue id makes both rows and tie-breaking independent of
    # the source file's incidental row order.
    aligned: Dict[int, np.ndarray] = {}
    for name, vector in zip(music_names, music_vectors):
        artist_id = graph_lookup.get(normalize_text(str(name)))
        if artist_id is not None and artist_id not in aligned:
            aligned[artist_id] = vector
    query_ids = np.asarray(sorted(aligned), dtype=np.int32)
    vectors = (
        _normalise_rows(np.asarray([aligned[int(key)] for key in query_ids]))
        if len(query_ids)
        else np.empty((0, music_vectors.shape[1]), dtype=np.float32)
    )
    width = min(96, max(len(query_ids) - 1, 0))
    indices = np.full((len(query_ids), width), -1, dtype=np.int32)
    weights = np.zeros((len(query_ids), width), dtype=np.float32)
    for start in range(0, len(query_ids), 256):
        stop = min(start + 256, len(query_ids))
        score = vectors[start:stop] @ vectors.T
        score[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        for offset, values in enumerate(score):
            # lexsort uses catalogue id as the deterministic tie-break.
            order = np.lexsort((query_ids, -values))[:width]
            positive = values[order] > 0.0
            count = int(positive.sum())
            if count:
                indices[start + offset, :count] = query_ids[order[positive]]
                weights[start + offset, :count] = values[order[positive]]

    artist_count = len(arrays["artist_names"])
    index_dtype = (
        np.int16
        if artist_count <= np.iinfo(np.int16).max + 1
        else np.int32
    )
    metadata = {
        "schema_version": 3,
        "asset_type": "catalog_artist_graph_dual_source_runtime",
        "available_variants": ["full"],
        "lastfm_candidate_policy": "source_mapped_only",
        "music4all_signal": "item2vec-full-exact-cosine-top96",
        "music4all_aligned_artists": int(len(query_ids)),
        "source_sha256": _digest(source_path),
        "music4all_full_sha256": _digest(music_path),
        "runtime_contains_raw_vectors": False,
        "silent_fallback": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        artist_names=arrays["artist_names"],
        track_artist_ids=arrays["track_artist_ids"],
        track_rows=arrays["track_rows"],
        track_indptr=arrays["track_indptr"],
        source_mapped=np.asarray(arrays["source_mapped"], dtype=np.uint8),
        artist_audio=np.asarray(
            _normalise_rows(arrays["artist_audio"]), dtype=np.float16
        ),
        full_indices=np.asarray(arrays["full_indices"], dtype=index_dtype),
        full_weights=np.asarray(arrays["full_weights"], dtype=np.float16),
        music4all_query_artist_ids=query_ids,
        music4all_indices=np.asarray(indices, dtype=index_dtype),
        music4all_weights=np.asarray(weights, dtype=np.float16),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return output_path


def _normalise_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.linalg.norm(values, axis=1, keepdims=True).clip(min=1e-8)


def catalogue_artists(
    index_path: Path,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[str, int],
    np.ndarray,
]:
    """Return normalized artists, row mappings, track order, and audio centroids."""
    with np.load(index_path, allow_pickle=False) as index:
        raw_artists = np.asarray(index["artists"])
        sonic = _normalise_rows(index["sonic"])
        clap = _normalise_rows(index["clap"])
        vibe = _normalise_rows(index["vibe"])
    normalized = np.asarray([normalize_text(str(value)) for value in raw_artists])
    artist_names = np.asarray(sorted(set(normalized.tolist())))
    lookup = {str(name): position for position, name in enumerate(artist_names)}
    track_artist_ids = np.asarray(
        [lookup[str(name)] for name in normalized], dtype=np.int32
    )
    order = np.argsort(track_artist_ids, kind="stable").astype(np.int32)
    counts = np.bincount(track_artist_ids, minlength=len(artist_names))
    indptr = np.concatenate(([0], np.cumsum(counts))).astype(np.int32)
    audio = np.concatenate((sonic, clap, vibe), axis=1)
    centroids = np.zeros((len(artist_names), audio.shape[1]), dtype=np.float32)
    np.add.at(centroids, track_artist_ids, audio)
    centroids /= counts[:, None].clip(min=1)
    centroids = _normalise_rows(centroids)
    return artist_names, track_artist_ids, order, indptr, lookup, centroids


def build_artist_corpus(
    archive_path: Path,
    corpus_path: Path,
    artist_lookup: Mapping[str, int],
    config: GraphBuildConfig,
) -> Dict[str, Any]:
    """Stream Last.fm-360K into deterministic catalogue-artist sentences."""
    if _digest(archive_path, "md5") != LASTFM_360K_MD5:
        raise ValueError("Last.fm-360K archive checksum mismatch")
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    users_total = 0
    users_mapped = 0
    source_rows = 0
    mapped_rows = 0
    tokens = 0
    seen_artists: Set[int] = set()
    current_user: str | None = None
    current: Dict[int, int] = {}

    def flush(user: str | None, values: Dict[int, int], handle: Any) -> None:
        nonlocal users_total, users_mapped, tokens
        if user is None:
            return
        users_total += 1
        if len(values) < 2:
            return
        selected = sorted(
            values.items(), key=lambda item: (-item[1], item[0])
        )[: config.max_artists_per_user]
        artist_ids = [artist_id for artist_id, _ in selected]
        rng = random.Random(f"{config.seed}:{user}")
        rng.shuffle(artist_ids)
        handle.write(" ".join(f"a{artist_id}" for artist_id in artist_ids) + "\n")
        users_mapped += 1
        tokens += len(artist_ids)

    with tarfile.open(archive_path, "r:gz") as archive:
        raw = archive.extractfile(_MEMBER)
        if raw is None:
            raise ValueError("Last.fm-360K artist-play member is missing")
        source = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
        with corpus_path.open("w", encoding="ascii") as output:
            for line in source:
                source_rows += 1
                parts = line.rstrip("\n").split("\t")
                if len(parts) != 4:
                    continue
                user, _, artist, raw_plays = parts
                if current_user is None:
                    current_user = user
                elif user != current_user:
                    flush(current_user, current, output)
                    current_user = user
                    current = {}
                artist_id = artist_lookup.get(normalize_text(artist))
                if artist_id is None:
                    continue
                try:
                    plays = int(raw_plays)
                except ValueError:
                    continue
                current[artist_id] = current.get(artist_id, 0) + plays
                seen_artists.add(artist_id)
                mapped_rows += 1
            flush(current_user, current, output)
    return {
        "source_rows": source_rows,
        "mapped_rows": mapped_rows,
        "mapping_rate": mapped_rows / max(source_rows, 1),
        "users_total": users_total,
        "users_with_two_mapped_artists": users_mapped,
        "corpus_tokens": tokens,
        "source_mapped_artists": len(seen_artists),
        "corpus_path": str(corpus_path),
        "corpus_sha256": _digest(corpus_path),
    }


def train_artist_vectors(
    corpus_path: Path,
    artist_count: int,
    config: GraphBuildConfig,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Train skip-gram artist vectors and return catalogue-aligned rows."""
    from gensim.models import Word2Vec

    started = time.perf_counter()
    model = Word2Vec(
        corpus_file=str(corpus_path),
        vector_size=config.vector_size,
        window=config.window,
        min_count=config.min_count,
        workers=config.workers,
        sg=1,
        negative=config.negative,
        hs=0,
        sample=1e-4,
        epochs=config.epochs,
        seed=config.seed,
        sorted_vocab=1,
    )
    vectors = np.zeros((artist_count, config.vector_size), dtype=np.float32)
    mapped = np.zeros(artist_count, dtype=bool)
    for artist_id in range(artist_count):
        token = f"a{artist_id}"
        if token not in model.wv:
            continue
        vectors[artist_id] = np.asarray(model.wv[token], dtype=np.float32)
        mapped[artist_id] = True
    vectors[mapped] = _normalise_rows(vectors[mapped])
    return vectors, mapped, {
        "training_seconds": time.perf_counter() - started,
        "vocabulary_artists": int(mapped.sum()),
        "configuration": asdict(config),
    }


def _exact_neighbors(
    vectors: np.ndarray,
    mapped: np.ndarray,
    neighbors: int,
) -> Tuple[np.ndarray, np.ndarray, str, float]:
    """Compute exact cosine k-NN in GPU blocks when CUDA is available."""
    started = time.perf_counter()
    mapped_ids = np.flatnonzero(mapped).astype(np.int32)
    compact = np.asarray(vectors[mapped], dtype=np.float32)
    count = min(neighbors, max(len(mapped_ids) - 1, 1))
    indices = np.full((len(vectors), neighbors), -1, dtype=np.int32)
    weights = np.zeros((len(vectors), neighbors), dtype=np.float32)
    backend = "numpy"
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        backend = f"torch-{torch.cuda.get_device_name(0)}"
        matrix = torch.from_numpy(compact).cuda()
        for start in range(0, len(compact), 512):
            stop = min(start + 512, len(compact))
            score = matrix[start:stop] @ matrix.T
            rows = torch.arange(start, stop, device=matrix.device)
            score[torch.arange(stop - start, device=matrix.device), rows] = -2.0
            values, positions = torch.topk(score, k=count, dim=1)
            values_np = values.float().cpu().numpy()
            positions_np = positions.int().cpu().numpy()
            target_rows = mapped_ids[start:stop]
            indices[target_rows, :count] = mapped_ids[positions_np]
            weights[target_rows, :count] = values_np
        del matrix
        torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        for start in range(0, len(compact), 256):
            stop = min(start + 256, len(compact))
            score = compact[start:stop] @ compact.T
            score[np.arange(stop - start), np.arange(start, stop)] = -2.0
            positions = np.argpartition(score, -count, axis=1)[:, -count:]
            values = np.take_along_axis(score, positions, axis=1)
            order = np.argsort(values, axis=1)[:, ::-1]
            positions = np.take_along_axis(positions, order, axis=1)
            values = np.take_along_axis(values, order, axis=1)
            target_rows = mapped_ids[start:stop]
            indices[target_rows, :count] = mapped_ids[positions]
            weights[target_rows, :count] = values
    weights = np.maximum(weights, 0.0)
    indices[weights <= 0.0] = -1
    return indices, weights, backend, time.perf_counter() - started


def _project_cold_artists(
    vectors: np.ndarray,
    source_mapped: np.ndarray,
    artist_audio: np.ndarray,
    anchors: int = 8,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Distill collaborative neighborhoods to unmapped artists via audio anchors."""
    started = time.perf_counter()
    projected = vectors.copy()
    source_ids = np.flatnonzero(source_mapped).astype(np.int32)
    cold_ids = np.flatnonzero(~source_mapped).astype(np.int32)
    source_audio = np.asarray(artist_audio[source_ids], dtype=np.float32)
    cold_audio = np.asarray(artist_audio[cold_ids], dtype=np.float32)
    backend = "numpy"
    positions_all = np.empty((len(cold_ids), anchors), dtype=np.int32)
    values_all = np.empty((len(cold_ids), anchors), dtype=np.float32)
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        backend = f"torch-{torch.cuda.get_device_name(0)}"
        source_tensor = torch.from_numpy(source_audio).cuda()
        for start in range(0, len(cold_ids), 512):
            stop = min(start + 512, len(cold_ids))
            query = torch.from_numpy(cold_audio[start:stop]).cuda()
            values, positions = torch.topk(
                query @ source_tensor.T, k=anchors, dim=1
            )
            positions_all[start:stop] = positions.int().cpu().numpy()
            values_all[start:stop] = values.float().cpu().numpy()
        del source_tensor
        torch.cuda.empty_cache()
    except (ImportError, RuntimeError):
        for start in range(0, len(cold_ids), 256):
            stop = min(start + 256, len(cold_ids))
            score = cold_audio[start:stop] @ source_audio.T
            positions = np.argpartition(score, -anchors, axis=1)[:, -anchors:]
            values = np.take_along_axis(score, positions, axis=1)
            order = np.argsort(values, axis=1)[:, ::-1]
            positions_all[start:stop] = np.take_along_axis(
                positions, order, axis=1
            )
            values_all[start:stop] = np.take_along_axis(
                values, order, axis=1
            )
    weights = np.maximum(values_all, 0.0)
    empty = ~np.any(weights > 0, axis=1)
    weights[empty] = 1.0
    anchor_vectors = vectors[source_ids[positions_all]]
    projected[cold_ids] = np.sum(
        anchor_vectors * weights[:, :, None], axis=1
    ) / weights.sum(axis=1, keepdims=True)
    projected = _normalise_rows(projected)
    return projected, {
        "method": "audio-centroid-to-collaborative-anchor-distillation",
        "anchors": anchors,
        "projected_artists": len(cold_ids),
        "backend": backend,
        "seconds": time.perf_counter() - started,
        "static_popularity_used": False,
    }


def _final_artist_edges(
    benchmark_path: Path,
    artist_lookup: Mapping[str, int],
) -> List[Tuple[int, int, str]]:
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    edges = []
    for record in benchmark["records"]:
        if record.get("split") != "final":
            continue
        query = artist_lookup.get(normalize_text(record["query"]["artist"]))
        if query is None:
            continue
        for positive in record["positives"]:
            target = artist_lookup.get(normalize_text(positive["artist"]))
            if target is not None and target != query:
                edges.append((query, target, record["id"]))
    return edges


def _remove_edge(
    indices: np.ndarray,
    weights: np.ndarray,
    left: int,
    right: int,
) -> int:
    removed = 0
    for source, target in ((left, right), (right, left)):
        positions = np.flatnonzero(indices[source] == target)
        if len(positions):
            indices[source, positions] = -1
            weights[source, positions] = 0.0
            removed += len(positions)
    return removed


def _compact_neighbors(indices: np.ndarray, weights: np.ndarray) -> None:
    for row in range(len(indices)):
        valid = np.flatnonzero(indices[row] >= 0)
        invalid = len(indices[row]) - len(valid)
        if invalid:
            indices[row] = np.concatenate(
                (
                    indices[row, valid],
                    np.full(invalid, -1, dtype=np.int32),
                )
            )
            weights[row] = np.concatenate(
                (
                    weights[row, valid],
                    np.zeros(invalid, dtype=np.float32),
                )
            )


def mask_final_topology(
    full_indices: np.ndarray,
    full_weights: np.ndarray,
    edges: Sequence[Tuple[int, int, str]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Create direct-edge and strict two-hop-masked graph variants."""
    direct_indices = full_indices.copy()
    direct_weights = full_weights.copy()
    direct_before = 0
    direct_removed = 0
    for query, target, _ in edges:
        direct_before += int(target in set(map(int, full_indices[query])))
        direct_removed += _remove_edge(
            direct_indices, direct_weights, query, target
        )
    _compact_neighbors(direct_indices, direct_weights)

    twohop_indices = direct_indices.copy()
    twohop_weights = direct_weights.copy()
    paths_before = 0
    for query, target, _ in edges:
        query_neighbors = set(
            map(int, direct_indices[query][direct_indices[query] >= 0])
        )
        target_neighbors = set(
            map(int, direct_indices[target][direct_indices[target] >= 0])
        )
        paths_before += len(query_neighbors & target_neighbors)
    broken = 0
    for query, target, record_id in edges:
        query_neighbors = set(
            map(int, twohop_indices[query][twohop_indices[query] >= 0])
        )
        target_neighbors = set(
            map(int, twohop_indices[target][twohop_indices[target] >= 0])
        )
        for middle in sorted(query_neighbors & target_neighbors):
            chooser = hashlib.sha256(
                f"{record_id}:{query}:{middle}:{target}".encode("utf-8")
            ).digest()[0]
            if chooser % 2:
                broken += _remove_edge(
                    twohop_indices, twohop_weights, query, middle
                )
            else:
                broken += _remove_edge(
                    twohop_indices, twohop_weights, middle, target
                )
    _compact_neighbors(twohop_indices, twohop_weights)
    direct_after = 0
    paths_after = 0
    for query, target, _ in edges:
        query_neighbors = set(
            map(int, twohop_indices[query][twohop_indices[query] >= 0])
        )
        target_neighbors = set(
            map(int, twohop_indices[target][twohop_indices[target] >= 0])
        )
        direct_after += int(target in query_neighbors)
        paths_after += len(query_neighbors & target_neighbors)
    return (
        direct_indices,
        direct_weights,
        twohop_indices,
        twohop_weights,
        {
            "final_positive_edges": len(edges),
            "exact_edges_present_before_mask": direct_before,
            "directed_edge_slots_removed": direct_removed,
            "exact_edges_present_after_mask": direct_after,
            "two_hop_paths_before_mask": paths_before,
            "edge_slots_removed_to_break_two_hop_paths": broken,
            "two_hop_paths_after_mask": paths_after,
        },
    )


def build_graph(
    archive_path: Path,
    index_path: Path,
    benchmark_path: Path,
    output_dir: Path,
    config: GraphBuildConfig,
) -> Dict[str, Any]:
    """Train, map, mask, and distill the catalogue-wide static graph."""
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    (
        artist_names,
        track_artist_ids,
        track_rows,
        track_indptr,
        artist_lookup,
        artist_audio,
    ) = catalogue_artists(index_path)
    corpus_path = output_dir / "lastfm360-catalog-artists.txt"
    corpus = build_artist_corpus(
        archive_path, corpus_path, artist_lookup, config
    )
    vectors, source_mapped, training = train_artist_vectors(
        corpus_path, len(artist_names), config
    )
    effective_vectors, projection = _project_cold_artists(
        vectors, source_mapped, artist_audio
    )
    effective_mapped = np.linalg.norm(effective_vectors, axis=1) > 1e-8
    full_indices, full_weights, backend, knn_seconds = _exact_neighbors(
        effective_vectors,
        np.ones(len(source_mapped), dtype=bool),
        config.neighbors,
    )
    final_edges = _final_artist_edges(benchmark_path, artist_lookup)
    (
        direct_indices,
        direct_weights,
        twohop_indices,
        twohop_weights,
        mask_audit,
    ) = mask_final_topology(
        full_indices, full_weights, final_edges
    )
    covered_tracks = int(source_mapped[track_artist_ids].sum())
    effective_tracks = int(effective_mapped[track_artist_ids].sum())
    cold_mask = ~source_mapped
    cold_with_neighbors = np.any(full_indices[cold_mask] >= 0, axis=1)
    coverage = {
        "catalogue_tracks": len(track_artist_ids),
        "catalogue_artists": len(artist_names),
        "source_mapped_tracks": covered_tracks,
        "source_mapped_artists": int(source_mapped.sum()),
        "track_coverage": covered_tracks / len(track_artist_ids),
        "query_artist_coverage": float(source_mapped.mean()),
        "effective_graph_tracks": effective_tracks,
        "effective_graph_artists": int(effective_mapped.sum()),
        "effective_track_coverage": effective_tracks / len(track_artist_ids),
        "effective_query_artist_coverage": float(effective_mapped.mean()),
        "cold_start_bridge_coverage": (
            float(cold_with_neighbors.mean()) if np.any(cold_mask) else 1.0
        ),
    }
    metadata = {
        "schema_version": 1,
        "method": "lastfm360-artist-skipgram-catalog-graph",
        "created_at": _now(),
        "source": {
            "name": "Last.fm-360K",
            "doi": LASTFM_360K_DOI,
            "url": LASTFM_360K_URL,
            "archive_md5": LASTFM_360K_MD5,
            "license": LASTFM_360K_LICENSE,
            "statistics": (
                "17,559,530 user-artist-play tuples from 359,347 users"
            ),
        },
        "index_sha256": _digest(index_path),
        "benchmark_sha256": _digest(benchmark_path),
        "configuration": asdict(config),
        "corpus": corpus,
        "training": training,
        "cold_start_projection": projection,
        "knn": {
            "backend": backend,
            "seconds": knn_seconds,
            "exact": True,
        },
        "coverage": coverage,
        "leakage_mask_audit": mask_audit,
        "runtime_contains_user_data": False,
        "runtime_contains_secret": False,
    }
    asset_path = output_dir / "catalog-artist-graph.npz"
    np.savez_compressed(
        asset_path,
        artist_names=artist_names,
        track_artist_ids=track_artist_ids,
        track_rows=track_rows,
        track_indptr=track_indptr,
        source_mapped=source_mapped.astype(np.uint8),
        artist_audio=artist_audio.astype(np.float16),
        full_indices=full_indices,
        full_weights=full_weights.astype(np.float16),
        direct_indices=direct_indices,
        direct_weights=direct_weights.astype(np.float16),
        twohop_indices=twohop_indices,
        twohop_weights=twohop_weights.astype(np.float16),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    report = dict(metadata)
    report.update(
        {
            "asset_path": str(asset_path),
            "asset_bytes": asset_path.stat().st_size,
            "asset_sha256": _digest(asset_path),
            "wall_seconds": time.perf_counter() - started,
        }
    )
    report_path = output_dir / "catalog-graph-report.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


class CatalogArtistGraph:
    """Compact runtime retrieval over artist affinity and audio bridge."""

    def __init__(self, path: str | Path):
        started = time.perf_counter()
        with np.load(path, allow_pickle=False) as asset:
            if "full_indices" not in asset.files or "full_weights" not in asset.files:
                raise ValueError(
                    "Catalog graph requires the full variant; no fallback is allowed"
                )
            legacy_three_variant = all(
                f"{name}_{suffix}" in asset.files
                for name in ("full", "direct", "twohop")
                for suffix in ("indices", "weights")
            )
            self.artist_names = np.asarray(asset["artist_names"])
            track_dtype = np.int32 if legacy_three_variant else None
            self.track_artist_ids = np.asarray(
                asset["track_artist_ids"], dtype=track_dtype
            )
            self.track_rows = np.asarray(asset["track_rows"], dtype=track_dtype)
            self.track_indptr = np.asarray(
                asset["track_indptr"], dtype=track_dtype
            )
            self.source_mapped = np.asarray(
                asset["source_mapped"], dtype=bool
            )
            self.artist_audio = (
                _normalise_rows(asset["artist_audio"])
                if legacy_three_variant
                else np.asarray(asset["artist_audio"])
            )
            self.variants = {}
            for name in ("full", "direct", "twohop"):
                index_key = f"{name}_indices"
                weight_key = f"{name}_weights"
                present = (index_key in asset.files, weight_key in asset.files)
                if any(present) and not all(present):
                    raise ValueError(
                        f"Catalog graph variant '{name}' is incomplete"
                    )
                if all(present):
                    self.variants[name] = (
                        np.asarray(
                            asset[index_key],
                            dtype=np.int32 if legacy_three_variant else None,
                        ),
                        np.asarray(
                            asset[weight_key],
                            dtype=np.float32 if legacy_three_variant else None,
                        ),
                    )
            self.metadata = json.loads(str(asset["metadata"].item()))
            dual_keys = (
                "music4all_query_artist_ids",
                "music4all_indices",
                "music4all_weights",
            )
            present = [name in asset.files for name in dual_keys]
            if any(present) and not all(present):
                raise ValueError("Music4All dual-source graph arrays are incomplete")
            self.music4all_query_artist_ids = (
                np.asarray(asset[dual_keys[0]], dtype=np.int32)
                if all(present)
                else np.empty(0, dtype=np.int32)
            )
            self.music4all_indices = (
                np.asarray(asset[dual_keys[1]], dtype=np.int32)
                if all(present)
                else np.empty((0, 0), dtype=np.int32)
            )
            self.music4all_weights = (
                np.asarray(asset[dual_keys[2]], dtype=np.float32)
                if all(present)
                else np.empty((0, 0), dtype=np.float32)
            )
            if all(present) and (
                len(self.music4all_query_artist_ids)
                != len(self.music4all_indices)
                or self.music4all_indices.shape != self.music4all_weights.shape
            ):
                raise ValueError("Music4All dual-source graph arrays are misaligned")
        self.artist_lookup = {
            str(name): position
            for position, name in enumerate(self.artist_names)
        }
        self._music4all_rows = {
            int(artist_id): row
            for row, artist_id in enumerate(self.music4all_query_artist_ids)
        }
        self.has_dual_source = bool(len(self.music4all_query_artist_ids))
        self.load_seconds = time.perf_counter() - started

    def dual_source_neighbors(self, query_artist: str) -> Dict[str, Any]:
        """Expose the independent Last.fm/Music4All top-96 union.

        Last.fm evidence is accepted only when both query and candidate were
        directly mapped in Last.fm.  Audio-projected rows never count as an
        independent source and no source is substituted when either is absent.
        """
        artist_id = self.artist_lookup.get(normalize_text(query_artist))
        coverage = {"lastfm": False, "music4all": False}
        empty = {
            "artist_ids": np.empty(0, dtype=np.int32),
            "weights": np.empty(0, dtype=np.float32),
        }
        lastfm = dict(empty)
        music4all = dict(empty)
        if artist_id is not None and bool(self.source_mapped[artist_id]):
            raw_indices, raw_weights = self._variant("full")
            valid = (
                (raw_indices[artist_id] >= 0)
                & self.source_mapped[
                    np.maximum(raw_indices[artist_id].astype(np.int64), 0)
                ]
            )
            lastfm = {
                "artist_ids": np.asarray(
                    raw_indices[artist_id, valid][:96], dtype=np.int32
                ),
                "weights": np.asarray(
                    raw_weights[artist_id, valid][:96], dtype=np.float32
                ),
            }
            coverage["lastfm"] = True
        music_row = self._music4all_rows.get(int(artist_id)) if artist_id is not None else None
        if music_row is not None:
            valid = self.music4all_indices[music_row] >= 0
            music4all = {
                "artist_ids": np.asarray(
                    self.music4all_indices[music_row, valid][:96], dtype=np.int32
                ),
                "weights": np.asarray(
                    self.music4all_weights[music_row, valid][:96], dtype=np.float32
                ),
            }
            coverage["music4all"] = True
        union = np.asarray(
            sorted(
                set(map(int, lastfm["artist_ids"]))
                | set(map(int, music4all["artist_ids"]))
            ),
            dtype=np.int32,
        )
        return {
            "artist_id": artist_id,
            "lastfm": lastfm,
            "music4all": music4all,
            "union_artist_ids": union,
            "source_coverage": coverage,
            "mode": (
                "dual_source_union"
                if all(coverage.values())
                else "dual_source_unavailable"
            ),
        }

    def _variant(self, variant: str) -> Tuple[np.ndarray, np.ndarray]:
        try:
            return self.variants[variant]
        except KeyError:
            available = ", ".join(sorted(self.variants))
            raise ValueError(
                f"Catalog graph variant '{variant}' is absent; "
                f"available variants: {available}. No fallback is allowed."
            ) from None

    def _bridge(
        self,
        audio_query: np.ndarray,
        variant: str,
        anchors: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        score = self.artist_audio @ np.asarray(audio_query, dtype=np.float32)
        score[~self.source_mapped] = -2.0
        count = min(anchors, int(self.source_mapped.sum()))
        anchor_ids = np.argpartition(score, -count)[-count:]
        anchor_ids = anchor_ids[np.argsort(score[anchor_ids])[::-1]]
        graph_indices, graph_weights = self._variant(variant)
        combined: Dict[int, float] = {}
        for anchor in anchor_ids:
            anchor_weight = max(float(score[anchor]), 0.0)
            combined[int(anchor)] = max(
                combined.get(int(anchor), 0.0), 0.35 * anchor_weight
            )
            for neighbor, weight in zip(
                graph_indices[anchor], graph_weights[anchor]
            ):
                if neighbor < 0:
                    continue
                value = anchor_weight * max(float(weight), 0.0)
                combined[int(neighbor)] = max(
                    combined.get(int(neighbor), 0.0), value
                )
        ordered = sorted(combined, key=lambda key: -combined[key])
        return (
            np.asarray(ordered, dtype=np.int32),
            np.asarray([combined[key] for key in ordered], dtype=np.float32),
        )

    def artist_neighbors(
        self,
        query_artist: str,
        audio_query: np.ndarray,
        variant: str = "twohop",
        anchors: int = 8,
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        indices, weights = self._variant(variant)
        artist_id = self.artist_lookup.get(normalize_text(query_artist))
        if artist_id is not None:
            valid = indices[artist_id] >= 0
            if np.any(valid):
                return (
                    np.asarray(indices[artist_id, valid], dtype=np.int32),
                    np.asarray(weights[artist_id, valid], dtype=np.float32),
                    (
                        "catalog_artist_graph"
                        if self.source_mapped[artist_id]
                        else "audio_projected_artist_bridge"
                    ),
                )
        rows, weights = self._bridge(audio_query, variant, anchors)
        return rows, weights, "audio_artist_bridge"

    def candidates(
        self,
        query_row: int,
        query_artist: str,
        audio_query: np.ndarray,
        track_audio_scores: np.ndarray,
        n: int = 1000,
        variant: str = "twohop",
        max_tracks_per_artist: int = 16,
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """Expand graph artists to tracks, audio-rank within each, and interleave."""
        neighbors, weights, mode = self.artist_neighbors(
            query_artist, audio_query, variant=variant
        )
        per_artist: List[Tuple[np.ndarray, float]] = []
        for artist_id, weight in zip(neighbors, weights):
            start = int(self.track_indptr[int(artist_id)])
            stop = int(self.track_indptr[int(artist_id) + 1])
            rows = self.track_rows[start:stop]
            if not len(rows):
                continue
            order = np.argsort(track_audio_scores[rows])[::-1]
            rows = rows[order[:max_tracks_per_artist]]
            per_artist.append((rows, float(weight)))
        selected: List[int] = []
        scores: List[float] = []
        seen: Set[int] = {int(query_row)}
        for depth in range(max_tracks_per_artist):
            layer = []
            for rows, graph_weight in per_artist:
                if depth >= len(rows):
                    continue
                row = int(rows[depth])
                value = graph_weight + 0.15 * float(track_audio_scores[row])
                layer.append((value, row))
            for value, row in sorted(layer, reverse=True):
                if row in seen:
                    continue
                seen.add(row)
                selected.append(row)
                scores.append(value)
                if len(selected) >= n:
                    return (
                        np.asarray(selected, dtype=np.int32),
                        np.asarray(scores, dtype=np.float32),
                        mode,
                    )
        return (
            np.asarray(selected, dtype=np.int32),
            np.asarray(scores, dtype=np.float32),
            mode,
        )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--neighbors", type=int, default=96)
    args = parser.parse_args(argv)
    config = GraphBuildConfig(epochs=args.epochs, neighbors=args.neighbors)
    report = build_graph(
        args.archive,
        args.index,
        args.benchmark,
        args.output_dir,
        config,
    )
    print(
        json.dumps(
            {
                "asset_path": report["asset_path"],
                "asset_bytes": report["asset_bytes"],
                "coverage": report["coverage"],
                "leakage_mask_audit": report["leakage_mask_audit"],
                "wall_seconds": report["wall_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
