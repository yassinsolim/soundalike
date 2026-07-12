"""Music4All-Onion collaborative candidate training and compact retrieval.

Training consumes user-track play counts, maps the public Music4All track IDs
to the shipped catalogue, and learns skip-gram item embeddings over listening
histories.  The compact runtime asset contains only mapped catalogue rows and
normalized vectors; no user data ships with the application.
"""
from __future__ import annotations

import argparse
import bz2
import csv
import hashlib
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Sequence, Set, Tuple

import numpy as np

from .real_benchmark import PairResolver, normalize_text

MUSIC4ALL_ONION_DOI = "https://doi.org/10.5281/zenodo.6609677"
MUSIC4ALL_METADATA_URL = (
    "https://huggingface.co/datasets/Leon299/music4all/raw/main/"
    "id_information.csv"
)
MUSIC4ALL_COUNTS_URL = (
    "https://zenodo.org/api/records/6609677/files/"
    "userid_trackid_count.tsv.bz2/content"
)
MUSIC4ALL_COUNTS_MD5 = "314b51196a9c8f333c7fefc0711760a1"


@dataclass(frozen=True)
class TrainingConfig:
    vector_size: int = 64
    window: int = 30
    negative: int = 15
    epochs: int = 8
    min_count: int = 2
    max_tracks_per_user: int = 160
    workers: int = max(1, min(24, (os.cpu_count() or 4) - 2))
    seed: int = 20260712


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def map_music4all_to_catalogue(
    metadata_path: Path,
    index_path: Path,
) -> Tuple[Dict[str, int], Dict[str, Any]]:
    """Map public Music4All IDs to one quality-preferred catalogue row."""
    with np.load(index_path, allow_pickle=False) as index:
        resolver = PairResolver(index["titles"], index["artists"])
    mapping: Dict[str, int] = {}
    total = 0
    duplicate_rows = 0
    rows_seen: Set[int] = set()
    with metadata_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        for record in csv.DictReader(handle, delimiter="\t"):
            total += 1
            row = resolver.query_row({
                "title": record.get("song", ""),
                "artist": record.get("artist", ""),
            })
            if row is None:
                continue
            track_id = record.get("id", "").strip()
            if not track_id:
                continue
            mapping[track_id] = int(row)
            if row in rows_seen:
                duplicate_rows += 1
            rows_seen.add(int(row))
    report = {
        "metadata_rows": total,
        "mapped_music4all_tracks": len(mapping),
        "unique_catalogue_rows": len(rows_seen),
        "duplicate_catalogue_mappings": duplicate_rows,
        "mapping_rate": len(mapping) / max(total, 1),
    }
    return mapping, report


def _final_internal_edges(
    benchmark_path: Path,
    index_path: Path,
    track_mapping: Mapping[str, int],
) -> Tuple[Set[Tuple[str, str]], Dict[str, Any]]:
    """Resolve fresh FINAL pairs to Music4All IDs for pre-training masking."""
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    with np.load(index_path, allow_pickle=False) as index:
        resolver = PairResolver(index["titles"], index["artists"])
    row_to_ids: Dict[int, List[str]] = defaultdict(list)
    for internal_id, row in track_mapping.items():
        row_to_ids[int(row)].append(internal_id)
    edges: Set[Tuple[str, str]] = set()
    mapped_pairs = []
    for pair in benchmark["pairs"]:
        if pair.get("split") != "final" or not pair.get("deciding_primary"):
            continue
        query_row = resolver.query_row(pair["query"])
        target_rows = resolver.target_rows(pair["target"])
        query_ids = row_to_ids.get(int(query_row), []) if query_row is not None else []
        target_ids = [
            internal_id
            for row in target_rows
            for internal_id in row_to_ids.get(int(row), [])
        ]
        pair_edges = {
            tuple(sorted((query_id, target_id)))
            for query_id in query_ids
            for target_id in target_ids
            if query_id != target_id
        }
        edges |= pair_edges
        if pair_edges:
            mapped_pairs.append({
                "pair_id": pair["id"],
                "internal_edge_count": len(pair_edges),
            })
    return edges, {
        "final_pairs": sum(
            pair.get("split") == "final" and pair.get("deciding_primary")
            for pair in benchmark["pairs"]
        ),
        "final_pairs_mapped_to_training_items": len(mapped_pairs),
        "internal_edges_masked": len(edges),
        "mapped_pairs": mapped_pairs,
    }


def _expanded_tokens(
    items: List[Tuple[str, int]],
    max_tracks: int,
) -> List[str]:
    selected = sorted(items, key=lambda value: (-value[1], value[0]))[:max_tracks]
    tokens = []
    for track_id, count in selected:
        repeats = min(3, max(1, int(math.log2(max(count, 1))) + 1))
        tokens.extend([track_id] * repeats)
    return tokens


def _mask_edges(
    user_id: str,
    tokens: List[str],
    masked_edges: Set[Tuple[str, str]],
) -> Tuple[List[str], int]:
    present = set(tokens)
    remove: Set[str] = set()
    overlaps = 0
    for left, right in masked_edges:
        if left not in present or right not in present:
            continue
        overlaps += 1
        chooser = hashlib.sha256(
            f"{user_id}\t{left}\t{right}".encode("utf-8")
        ).digest()[0]
        remove.add(right if chooser % 2 else left)
    if not remove:
        return tokens, overlaps
    return [token for token in tokens if token not in remove], overlaps


def build_user_corpora(
    counts_path: Path,
    track_mapping: Mapping[str, int],
    full_corpus_path: Path,
    masked_corpus_path: Path,
    masked_edges: Set[Tuple[str, str]],
    config: TrainingConfig,
) -> Dict[str, Any]:
    """Stream 50M interactions into deterministic full and edge-masked corpora."""
    full_corpus_path.parent.mkdir(parents=True, exist_ok=True)
    mapped_ids = set(track_mapping)
    users_total = 0
    users_mapped = 0
    full_tokens = 0
    masked_tokens = 0
    pair_user_overlaps = 0
    current_user = None
    current_items: List[Tuple[str, int]] = []

    def flush(
        user_id: str | None,
        items: List[Tuple[str, int]],
        full_handle: Any,
        masked_handle: Any,
    ) -> None:
        nonlocal users_total, users_mapped, full_tokens, masked_tokens
        nonlocal pair_user_overlaps
        if user_id is None:
            return
        users_total += 1
        tokens = _expanded_tokens(items, config.max_tracks_per_user)
        if len(set(tokens)) < 2:
            return
        users_mapped += 1
        rng = random.Random(f"{config.seed}:{user_id}")
        rng.shuffle(tokens)
        masked, overlap = _mask_edges(user_id, tokens, masked_edges)
        pair_user_overlaps += overlap
        full_handle.write(" ".join(tokens) + "\n")
        full_tokens += len(tokens)
        if len(set(masked)) >= 2:
            masked_handle.write(" ".join(masked) + "\n")
            masked_tokens += len(masked)

    with (
        bz2.open(counts_path, "rt", encoding="utf-8", errors="replace") as source,
        full_corpus_path.open("w", encoding="ascii") as full_handle,
        masked_corpus_path.open("w", encoding="ascii") as masked_handle,
    ):
        header = next(source, "")
        if not header.startswith("user_id\ttrack_id\tcount"):
            raise ValueError("Unexpected Music4All-Onion count schema")
        for line in source:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            user_id, track_id, raw_count = parts
            if current_user is None:
                current_user = user_id
            elif user_id != current_user:
                flush(current_user, current_items, full_handle, masked_handle)
                current_user = user_id
                current_items = []
            if track_id in mapped_ids:
                try:
                    current_items.append((track_id, int(raw_count)))
                except ValueError:
                    continue
        flush(current_user, current_items, full_handle, masked_handle)
    return {
        "users_total": users_total,
        "users_with_two_mapped_tracks": users_mapped,
        "full_corpus_tokens": full_tokens,
        "masked_corpus_tokens": masked_tokens,
        "final_pair_user_overlaps_before_mask": pair_user_overlaps,
        "final_pair_user_overlaps_after_mask": 0,
        "max_tracks_per_user": config.max_tracks_per_user,
    }


def train_item2vec(
    corpus_path: Path,
    output_path: Path,
    track_mapping: Mapping[str, int],
    index_path: Path,
    config: TrainingConfig,
    edge_masked: bool,
) -> Dict[str, Any]:
    """Train skip-gram item2vec and distill it to catalogue-row vectors."""
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
    by_row: Dict[int, List[np.ndarray]] = defaultdict(list)
    for internal_id, row in track_mapping.items():
        if internal_id in model.wv:
            by_row[int(row)].append(np.asarray(model.wv[internal_id], dtype=np.float32))
    rows = np.asarray(sorted(by_row), dtype=np.int32)
    vectors = np.stack([
        np.mean(by_row[int(row)], axis=0) for row in rows
    ]).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-8)
    with np.load(index_path, allow_pickle=False) as index:
        catalogue_size = len(index["titles"])
        artists = np.asarray(index["artists"])
    artist_groups: Dict[str, List[int]] = defaultdict(list)
    for position, row in enumerate(rows):
        artist_groups[normalize_text(str(artists[int(row)]))].append(position)
    artist_names = np.asarray(sorted(artist_groups))
    artist_vectors = np.stack([
        np.mean(vectors[artist_groups[str(name)]], axis=0)
        for name in artist_names
    ]).astype(np.float32)
    artist_vectors /= np.linalg.norm(
        artist_vectors, axis=1, keepdims=True
    ).clip(min=1e-8)
    metadata = {
        "schema_version": 1,
        "method": "music4all-onion-skipgram-item2vec",
        "created_at": _now(),
        "edge_masked": edge_masked,
        "catalogue_size": catalogue_size,
        "mapped_catalogue_rows": len(rows),
        "mapped_artists": len(artist_names),
        "configuration": asdict(config),
        "training_seconds": time.perf_counter() - started,
        "corpus_path": str(corpus_path),
        "corpus_sha256": _sha256(corpus_path),
        "source_doi": MUSIC4ALL_ONION_DOI,
        "source_license": "CC-BY-4.0",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        catalog_rows=rows,
        vectors=vectors.astype(np.float16),
        artist_names=artist_names,
        artist_vectors=artist_vectors.astype(np.float16),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    metadata["asset_path"] = str(output_path)
    metadata["asset_bytes"] = output_path.stat().st_size
    metadata["asset_sha256"] = _sha256(output_path)
    return metadata


class CollaborativeIndex:
    """Compact query-conditioned item2vec candidate retrieval."""

    def __init__(self, path: str | Path, catalogue_size: int):
        started = time.perf_counter()
        with np.load(path, allow_pickle=False) as asset:
            self.rows = np.asarray(asset["catalog_rows"], dtype=np.int32)
            self.vectors = np.asarray(asset["vectors"], dtype=np.float32)
            self.artist_names = np.asarray(asset["artist_names"])
            self.artist_vectors = np.asarray(
                asset["artist_vectors"], dtype=np.float32
            )
            self.metadata = json.loads(str(asset["metadata"]))
        self.vectors /= np.linalg.norm(
            self.vectors, axis=1, keepdims=True
        ).clip(min=1e-8)
        self.artist_vectors /= np.linalg.norm(
            self.artist_vectors, axis=1, keepdims=True
        ).clip(min=1e-8)
        self.row_lookup = np.full(catalogue_size, -1, dtype=np.int32)
        self.row_lookup[self.rows] = np.arange(len(self.rows), dtype=np.int32)
        self.artist_lookup = {
            str(name): position for position, name in enumerate(self.artist_names)
        }
        self.load_seconds = time.perf_counter() - started

    def query_vector(
        self,
        query_row: int,
        query_artist: str,
        audio_scores: np.ndarray | None = None,
        bridge_size: int = 24,
    ) -> Tuple[np.ndarray | None, str]:
        position = int(self.row_lookup[int(query_row)])
        if position >= 0:
            return self.vectors[position], "track"
        artist_position = self.artist_lookup.get(normalize_text(query_artist))
        if artist_position is not None:
            return self.artist_vectors[artist_position], "artist"
        if audio_scores is None:
            return None, "unavailable"
        mapped_scores = np.asarray(audio_scores[self.rows], dtype=np.float32)
        count = min(bridge_size, len(mapped_scores))
        positions = np.argpartition(mapped_scores, -count)[-count:]
        weights = np.maximum(mapped_scores[positions], 0.0)
        if not np.any(weights):
            weights = np.ones_like(weights)
        vector = np.average(self.vectors[positions], axis=0, weights=weights)
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-8:
            return None, "unavailable"
        return (vector / norm).astype(np.float32), "audio_bridge"

    def candidates(
        self,
        query_row: int,
        query_artist: str,
        audio_scores: np.ndarray | None = None,
        n: int = 1000,
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        vector, mode = self.query_vector(query_row, query_artist, audio_scores)
        if vector is None:
            return (
                np.empty(0, dtype=np.int32),
                np.empty(0, dtype=np.float32),
                mode,
            )
        scores = self.vectors @ vector
        position = int(self.row_lookup[int(query_row)])
        if position >= 0:
            scores[position] = -np.inf
        count = min(n, len(scores))
        selected = np.argpartition(scores, -count)[-count:]
        selected = selected[np.argsort(scores[selected])[::-1]]
        return self.rows[selected], scores[selected], mode


def train_pipeline(
    metadata_path: Path,
    counts_path: Path,
    index_path: Path,
    benchmark_path: Path,
    output_dir: Path,
    config: TrainingConfig,
) -> Dict[str, Any]:
    """Map, mask, train full/masked models, and write an auditable report."""
    if _md5(counts_path) != MUSIC4ALL_COUNTS_MD5:
        raise ValueError("Music4All-Onion listening-count checksum mismatch")
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping, mapping_report = map_music4all_to_catalogue(
        metadata_path, index_path
    )
    mapping_path = output_dir / "catalogue-mapping.json"
    mapping_path.write_text(
        json.dumps(mapping, sort_keys=True) + "\n", encoding="utf-8"
    )
    masked_edges, edge_report = _final_internal_edges(
        benchmark_path, index_path, mapping
    )
    full_corpus = output_dir / "users-full.txt"
    masked_corpus = output_dir / "users-final-edges-masked.txt"
    corpus_report = build_user_corpora(
        counts_path,
        mapping,
        full_corpus,
        masked_corpus,
        masked_edges,
        config,
    )
    full_model = train_item2vec(
        full_corpus,
        output_dir / "item2vec-full.npz",
        mapping,
        index_path,
        config,
        edge_masked=False,
    )
    masked_model = train_item2vec(
        masked_corpus,
        output_dir / "item2vec-final-edges-masked.npz",
        mapping,
        index_path,
        config,
        edge_masked=True,
    )
    report = {
        "schema_version": 1,
        "created_at": _now(),
        "source": {
            "name": "Music4All-Onion",
            "doi": MUSIC4ALL_ONION_DOI,
            "license": "CC-BY-4.0",
            "provenance": (
                "252,984,396 Last.fm listening records from 119,140 users; "
                "the count subset has 50,016,042 user-track interactions."
            ),
            "counts_url": MUSIC4ALL_COUNTS_URL,
            "counts_md5": MUSIC4ALL_COUNTS_MD5,
            "metadata_url": MUSIC4ALL_METADATA_URL,
        },
        "configuration": asdict(config),
        "mapping": mapping_report,
        "edge_mask_audit": edge_report,
        "corpus": corpus_report,
        "models": {"full": full_model, "edge_masked": masked_model},
    }
    report_path = output_dir / "training-report.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--counts", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--vector-size", type=int, default=64)
    args = parser.parse_args(argv)
    config = TrainingConfig(epochs=args.epochs, vector_size=args.vector_size)
    report = train_pipeline(
        args.metadata,
        args.counts,
        args.index,
        args.benchmark,
        args.output_dir,
        config,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
