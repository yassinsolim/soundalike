"""Development diagnostics for the next full-track retrieval experiment.

This module is deliberately separate from ``fulltrack_eval`` so diagnostics
cannot alter a running or frozen benchmark. It accepts only official train or
validation partitions and never opens audio.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .fulltrack_extract import normalize_rows
from .fulltrack_store import FullTrackStoreReader, stable_json_sha256
from .jamendo_fulltrack import EVIDENCE_SCOPE, JamendoContext, load_jamendo_context


DIAGNOSTIC_SCHEMA_VERSION = 1
DEFAULT_POOL_SIZES: Tuple[int, ...] = (200, 500, 1_000, 2_000)
_TAG_PATTERN = re.compile(r"(genre|instrument|mood/theme)---([^\t\r\n]+)\Z")


class FullTrackDiagnosticError(RuntimeError):
    """Invalid split, store, relevance definition, or diagnostic output."""


@dataclass(frozen=True)
class CandidateCeilingConfig:
    fold_index: int = 0
    part: str = "validation"
    pool_sizes: Tuple[int, ...] = DEFAULT_POOL_SIZES
    recall_cutoff: int = 10
    min_shared_tags: int = 2
    min_tag_jaccard: float = 0.25

    def validate(self) -> None:
        if isinstance(self.fold_index, bool) or not isinstance(self.fold_index, int):
            raise FullTrackDiagnosticError("fold_index must be an integer")
        if self.fold_index < 0:
            raise FullTrackDiagnosticError("fold_index must be non-negative")
        if self.part not in ("train", "validation"):
            raise FullTrackDiagnosticError(
                "v2 diagnostics may inspect only train or validation partitions"
            )
        if not self.pool_sizes:
            raise FullTrackDiagnosticError("at least one candidate pool size is required")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in self.pool_sizes
        ):
            raise FullTrackDiagnosticError("candidate pool sizes must be positive integers")
        if tuple(sorted(set(self.pool_sizes))) != self.pool_sizes:
            raise FullTrackDiagnosticError(
                "candidate pool sizes must be unique and strictly increasing"
            )
        if isinstance(self.recall_cutoff, bool) or self.recall_cutoff <= 0:
            raise FullTrackDiagnosticError("recall_cutoff must be a positive integer")
        if isinstance(self.min_shared_tags, bool) or self.min_shared_tags <= 0:
            raise FullTrackDiagnosticError("min_shared_tags must be a positive integer")
        if not math.isfinite(float(self.min_tag_jaccard)) or not (
            0.0 < float(self.min_tag_jaccard) <= 1.0
        ):
            raise FullTrackDiagnosticError("min_tag_jaccard must be in (0, 1]")

    def as_dict(self) -> Dict[str, object]:
        return {
            "fold_index": self.fold_index,
            "part": self.part,
            "pool_sizes": list(self.pool_sizes),
            "recall_cutoff": self.recall_cutoff,
            "min_shared_tags": self.min_shared_tags,
            "min_tag_jaccard": float(self.min_tag_jaccard),
        }


def _validated_tags(value: object, *, where: str) -> Tuple[str, ...]:
    if (
        not isinstance(value, (list, tuple))
        or not value
        or any(not isinstance(tag, str) for tag in value)
    ):
        raise FullTrackDiagnosticError(f"{where} tags must be a non-empty string list")
    tags = tuple(value)
    if len(tags) != len(set(tags)):
        raise FullTrackDiagnosticError(f"{where} tags must be unique")
    if any(_TAG_PATTERN.fullmatch(tag) is None for tag in tags):
        raise FullTrackDiagnosticError(f"{where} contains a malformed tag")
    return tags


def _tag_jaccard_relevance(
    query_tags: Sequence[str],
    candidate_tags: Sequence[str],
    *,
    min_shared_tags: int,
    min_tag_jaccard: float,
) -> float:
    query = set(query_tags)
    candidate = set(candidate_tags)
    shared = len(query.intersection(candidate))
    union = len(query.union(candidate))
    if shared < min_shared_tags or not union:
        return 0.0
    jaccard = shared / union
    return float(jaccard if jaccard >= min_tag_jaccard else 0.0)


def _mean(values: Sequence[float]) -> float:
    return 0.0 if not values else float(np.mean(np.asarray(values, dtype=np.float64)))


def candidate_ceiling_report(
    context: JamendoContext,
    reader: FullTrackStoreReader,
    *,
    config: CandidateCeilingConfig = CandidateCeilingConfig(),
) -> Mapping[str, object]:
    """Measure the global candidate pool's maximum possible retrieval recall.

    The oracle may reorder only tracks already present in each requested pool.
    This isolates first-stage candidate recall from reranker quality.
    """

    config.validate()
    if context.evidence_scope != EVIDENCE_SCOPE:
        raise FullTrackDiagnosticError("context evidence scope is not full-track Jamendo")
    fold = next(
        (item for item in context.folds if int(item.index) == config.fold_index),
        None,
    )
    if fold is None:
        raise FullTrackDiagnosticError(f"fold {config.fold_index} is not loaded")
    selected = tuple(
        track
        for track in context.tracks
        if fold.track_parts.get(int(track.track_id)) == config.part
    )
    if len(selected) < 2:
        raise FullTrackDiagnosticError("selected diagnostic partition is too small")

    selected_ids = tuple(int(track.track_id) for track in selected)
    store_rows = {int(track_id): row for row, track_id in enumerate(reader.track_ids)}
    missing = sorted(set(selected_ids) - set(store_rows))
    if missing:
        raise FullTrackDiagnosticError(
            f"sealed store is missing diagnostic tracks: {missing[:10]}"
        )
    tags_by_id = {
        track_id: _validated_tags(
            fold.track_tags.get(track_id),
            where=f"fold {config.fold_index} track {track_id}",
        )
        for track_id in selected_ids
    }
    positions = np.asarray([store_rows[track_id] for track_id in selected_ids], dtype=np.int64)
    embeddings = normalize_rows(np.asarray(reader.global_embeddings[positions], dtype=np.float32))
    artist_ids = np.asarray([int(track.artist_id) for track in selected], dtype=np.int64)

    pool_values: Dict[int, Dict[str, list]] = {
        size: {
            "relevant_coverage": [],
            "oracle_recall_at_k": [],
            "has_relevant": [],
            "relevant_in_pool": [],
            "effective_pool_size": [],
        }
        for size in config.pool_sizes
    }
    query_records = []
    relevant_counts = []
    first_relevant_ranks = []
    global_recall = []
    skipped_no_relevant = 0

    for query_position, query in enumerate(selected):
        eligible = np.flatnonzero(
            (np.arange(len(selected), dtype=np.int64) != query_position)
            & (artist_ids != artist_ids[query_position])
        )
        if not len(eligible):
            raise FullTrackDiagnosticError(
                f"track {query.track_id} has no different-artist candidates"
            )
        relevance = {
            int(selected[candidate_position].track_id): grade
            for candidate_position in eligible.tolist()
            if (
                grade := _tag_jaccard_relevance(
                    tags_by_id[int(query.track_id)],
                    tags_by_id[int(selected[candidate_position].track_id)],
                    min_shared_tags=config.min_shared_tags,
                    min_tag_jaccard=config.min_tag_jaccard,
                )
            )
        }
        if not relevance:
            skipped_no_relevant += 1
            continue

        scores = embeddings[eligible] @ embeddings[query_position]
        order = eligible[np.lexsort((eligible, -scores))]
        ranked_ids = [int(selected[position].track_id) for position in order.tolist()]
        first_rank = next(
            rank
            for rank, track_id in enumerate(ranked_ids, 1)
            if track_id in relevance
        )
        relevant_count = len(relevance)
        actual_hits = sum(
            track_id in relevance for track_id in ranked_ids[: config.recall_cutoff]
        )
        actual_recall = actual_hits / relevant_count
        relevant_counts.append(float(relevant_count))
        first_relevant_ranks.append(float(first_rank))
        global_recall.append(float(actual_recall))

        pools = {}
        for requested_size in config.pool_sizes:
            effective_size = min(requested_size, len(ranked_ids))
            relevant_in_pool = sum(
                track_id in relevance for track_id in ranked_ids[:effective_size]
            )
            coverage = relevant_in_pool / relevant_count
            oracle_recall = min(relevant_in_pool, config.recall_cutoff) / relevant_count
            values = pool_values[requested_size]
            values["relevant_coverage"].append(float(coverage))
            values["oracle_recall_at_k"].append(float(oracle_recall))
            values["has_relevant"].append(float(relevant_in_pool > 0))
            values["relevant_in_pool"].append(float(relevant_in_pool))
            values["effective_pool_size"].append(float(effective_size))
            pools[str(requested_size)] = {
                "effective_pool_size": effective_size,
                "relevant_in_pool": relevant_in_pool,
                "relevant_coverage": float(coverage),
                "oracle_recall_at_k": float(oracle_recall),
            }
        query_records.append(
            {
                "track_id": int(query.track_id),
                "artist_id": int(query.artist_id),
                "relevant_candidates": relevant_count,
                "first_relevant_global_rank": first_rank,
                "global_recall_at_k": float(actual_recall),
                "pools": pools,
            }
        )

    if not query_records:
        raise FullTrackDiagnosticError("no queries have cross-artist relevant candidates")

    aggregate_pools = {}
    for requested_size, values in pool_values.items():
        aggregate_pools[str(requested_size)] = {
            "mean_effective_pool_size": _mean(values["effective_pool_size"]),
            "mean_relevant_in_pool": _mean(values["relevant_in_pool"]),
            "mean_relevant_coverage": _mean(values["relevant_coverage"]),
            "query_hit_rate": _mean(values["has_relevant"]),
            "mean_oracle_recall_at_k": _mean(values["oracle_recall_at_k"]),
            "oracle_minus_global_recall_at_k": float(
                _mean(values["oracle_recall_at_k"]) - _mean(global_recall)
            ),
        }

    store_binding = dict(reader.binding.as_dict())
    payload: Dict[str, object] = {
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "artifact_kind": "fulltrack_candidate_ceiling_diagnostic",
        "evidence_scope": EVIDENCE_SCOPE,
        "deciding": False,
        "test_partition_accessed": False,
        "source_fingerprint": context.source_fingerprint,
        "store_binding": store_binding,
        "store_binding_sha256": stable_json_sha256(store_binding),
        "config": config.as_dict(),
        "config_sha256": stable_json_sha256(config.as_dict()),
        "selected_track_count": len(selected),
        "query_count": len(query_records),
        "skipped_no_relevant": skipped_no_relevant,
        "global_recall_at_k": _mean(global_recall),
        "relevant_candidates": {
            "mean": _mean(relevant_counts),
            "minimum": int(min(relevant_counts)),
            "maximum": int(max(relevant_counts)),
        },
        "first_relevant_global_rank": {
            "mean": _mean(first_relevant_ranks),
            "median": float(np.median(np.asarray(first_relevant_ranks))),
            "maximum": int(max(first_relevant_ranks)),
        },
        "pools": aggregate_pools,
        "queries": query_records,
        "notice": (
            "Development diagnostic only. It measures a global candidate pool ceiling "
            "on an official train/validation partition and cannot authorize promotion."
        ),
    }
    payload["report_payload_sha256"] = stable_json_sha256(payload)
    return payload


def write_candidate_ceiling_report(path: Path, report: Mapping[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.parent.is_symlink():
        raise FullTrackDiagnosticError("diagnostic output may not use a symlink")
    raw = json.dumps(
        dict(report),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    temporary = target.with_name(f".{target.name}.{secrets.token_hex(8)}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    ceiling = subparsers.add_parser(
        "candidate-ceiling",
        help="measure validation-only global candidate recall ceilings",
    )
    ceiling.add_argument("--metadata-root", required=True)
    ceiling.add_argument("--audio-root", required=True)
    ceiling.add_argument("--state-root", required=True)
    ceiling.add_argument("--store", required=True)
    ceiling.add_argument("--output", required=True)
    ceiling.add_argument("--fold", type=int, default=0)
    ceiling.add_argument("--part", choices=("train", "validation"), default="validation")
    ceiling.add_argument("--pool-size", type=int, action="append", dest="pool_sizes")
    ceiling.add_argument("--recall-cutoff", type=int, default=10)
    ceiling.add_argument("--min-shared-tags", type=int, default=2)
    ceiling.add_argument("--min-tag-jaccard", type=float, default=0.25)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "candidate-ceiling":
        raise FullTrackDiagnosticError(f"unknown command {args.command!r}")
    config = CandidateCeilingConfig(
        fold_index=args.fold,
        part=args.part,
        pool_sizes=tuple(args.pool_sizes or DEFAULT_POOL_SIZES),
        recall_cutoff=args.recall_cutoff,
        min_shared_tags=args.min_shared_tags,
        min_tag_jaccard=args.min_tag_jaccard,
    )
    context = load_jamendo_context(
        Path(args.metadata_root),
        Path(args.audio_root),
        Path(args.state_root),
        production=True,
    )
    with FullTrackStoreReader(
        Path(args.store),
        expected_source_fingerprint=context.source_fingerprint,
    ) as reader:
        report = candidate_ceiling_report(context, reader, config=config)
    write_candidate_ceiling_report(Path(args.output), report)
    print(
        json.dumps(
            {
                "output": args.output,
                "fold": args.fold,
                "part": args.part,
                "queries": report["query_count"],
                "report_payload_sha256": report["report_payload_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())