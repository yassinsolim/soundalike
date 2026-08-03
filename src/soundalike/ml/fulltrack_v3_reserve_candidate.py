"""Build, freeze, and audit the final reserve CLAP/MusicFM V3 candidate."""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .fulltrack_eval import (
    METRICS,
    _BudgetCache,
    _method_ranking,
    _paired_bootstrap_delta,
    _query_metrics,
    _tag_jaccard_relevance,
)
from .fulltrack_store import (
    FullTrackStoreReader,
    sha256_path,
    stable_json_sha256,
)
from .fulltrack_v3 import (
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    CANDIDATE_POOL,
    CLAP_MANIFEST_FILE_SHA256,
    EXPECTED_CLAP_BINDING,
    MAXSIM_BUDGET,
    MAX_FEATURE_CACHE_BYTES,
    SOURCE_FINGERPRINT,
    _open_bound_store,
    _replace_json,
)
from .fulltrack_v3_ranker import (
    MUSICFM_MODEL_SHA256,
    _score_channels,
    _write_json_exclusive,
    _write_npz_exclusive,
)
from .fulltrack_v3_reserve_protocol import (
    EXPECTED_SELECTION_SHA256,
    EXPECTED_SPLITS,
    load_reserve_protocol,
)
from .fulltrack_v3_semantic import (
    LABEL_HEADER,
    SemanticHead,
    fit_semantic_head,
)
from .jamendo_fulltrack import EVIDENCE_SCOPE, _ID_PATTERNS, _TAG


MODEL_SCHEMA_VERSION = 1
MODEL_KIND = "v3_final_reserve_clap_musicfm"
DEVELOPMENT_REPORT_KIND = "v3_final_reserve_development_report"
FREEZE_KIND = "v3_final_reserve_shadow_freeze"
SHADOW_REPORT_KIND = "v3_final_reserve_shadow_audit"
SHADOW_STATE_KIND = "v3_final_reserve_shadow_audit_state"
PROTOCOL_FILE_SHA256 = (
    "e04e65a44dc102423595e76d67b962b9b4822e61871716cbe3900414ca954018"
)
PROTOCOL_PAYLOAD_SHA256 = (
    "ac8c649a4cfd2672015f587bfb204105d63dcdffe0f1d8a6d1142737127e89be"
)
REFINEMENT_KIND = "v3_reserve_clap_window_max_audio_boundary"
REFINEMENT_PAYLOAD_SHA256 = (
    "e18b598aa432e5b6fd7ad340747a804641445a17366cca8480bf2ed7f3c50102"
)
MUSICFM_TRAIN_CONFIG_SHA256 = (
    "c2b0c316a36226abd26b85912c136967ca7df5ebfcb59c5bff0b9f42ed169ea3"
)
MUSICFM_SPARSE_CONFIG_SHA256 = (
    "debe0a5cd5c2daed70426cbe5c095d272d97541e5a7c0fff977a5e012471bfaa"
)
MUSICFM_EXTRACTION_PLAN_KIND = "v3_final_reserve_musicfm_sparse_four_plan"
MUSICFM_TRAIN_TRACKS = 8_192
CLAP_HEAD_NAMES = ("global", "salient", "repeated", "window_max")
RIDGE = 10.0
FOLD_SEED = 20260814
SHADOW_FOLDS = 5
TOP_TAGS = 5
TOP_IDF_POWER = 1.5
TOP_SHARE = 0.25
CLAP_GLOBAL_SALIENT_SHARE = 0.5
CLAP_FOUR_VIEW_SHARE = 0.5
CLAP_WINDOW_MAX_AUDIO_SHARE = 0.3
MUSICFM_AUDIO_SHARE = 0.075
MUSICFM_SEMANTIC_SHARE = 0.1
SEMANTIC_WEIGHT = 0.4
PRIMARY_METRIC = "recall_at_k"
MIN_RECALL_GAIN = 0.20
MIN_POSITIVE_FOLDS = 4
MAX_FOLD_RECALL_REGRESSION = 0.05
MAX_SAFETY_REGRESSION = 0.01
EXPECTED_DEVELOPMENT_RELATIVE = {
    "recall_at_k": 0.20004991370059355,
    "mrr": 0.08695668584973792,
    "graded_ndcg_at_k": 0.11295960528352023,
}


class V3ReserveCandidateError(RuntimeError):
    """Invalid, changed, leaky, or prematurely opened reserve candidate."""


@dataclass(frozen=True)
class PartitionData:
    split: str
    track_ids: np.ndarray
    artist_ids: np.ndarray
    query_folds: np.ndarray
    global_orders: np.ndarray
    global_lengths: np.ndarray
    pools: np.ndarray
    baseline_scores: np.ndarray
    relevance: np.ndarray

    def validate(self) -> None:
        expected = EXPECTED_SPLITS.get(self.split)
        if expected is None:
            raise V3ReserveCandidateError("unknown reserve evaluation split")
        count = int(expected["tracks"])
        if (
            self.track_ids.shape != (count,)
            or self.artist_ids.shape != (count,)
            or self.query_folds.shape != (count,)
            or self.global_orders.shape != (count, count - 1)
            or self.global_lengths.shape != (count,)
            or self.pools.shape != (count, CANDIDATE_POOL)
            or self.baseline_scores.shape != self.pools.shape
            or self.relevance.shape != (count, count)
            or len(np.unique(self.track_ids)) != count
            or len(np.unique(self.artist_ids)) != int(expected["artists"])
            or np.any(self.query_folds < 0)
            or np.any(self.query_folds >= SHADOW_FOLDS)
            or not np.all(np.isfinite(self.baseline_scores))
            or not np.all(np.isfinite(self.relevance))
        ):
            raise V3ReserveCandidateError(
                "reserve partition shape or value drift"
            )
        for query, length in enumerate(self.global_lengths):
            order = self.global_orders[query, : int(length)]
            if (
                length < CANDIDATE_POOL
                or np.any(order < 0)
                or len(np.unique(order)) != len(order)
                or not np.array_equal(order[:CANDIDATE_POOL], self.pools[query])
            ):
                raise V3ReserveCandidateError(
                    "reserve partition global order drift"
                )


@dataclass(frozen=True)
class ReserveCandidateModel:
    vocabulary: Tuple[str, ...]
    clap_heads: Mapping[str, SemanticHead]
    musicfm_head: SemanticHead

    def validate(self) -> None:
        tag_count = len(self.vocabulary)
        if (
            not tag_count
            or tuple(sorted(self.vocabulary)) != self.vocabulary
            or len(set(self.vocabulary)) != tag_count
            or set(self.clap_heads) != set(CLAP_HEAD_NAMES)
        ):
            raise V3ReserveCandidateError("reserve model identity drift")
        for name, head in self.clap_heads.items():
            head.validate()
            if (
                name not in CLAP_HEAD_NAMES
                or head.representation != "clap"
                or head.ridge != RIDGE
                or head.vocabulary != self.vocabulary
            ):
                raise V3ReserveCandidateError("CLAP head identity drift")
        self.musicfm_head.validate()
        if (
            self.musicfm_head.representation != "musicfm"
            or self.musicfm_head.ridge != RIDGE
            or self.musicfm_head.vocabulary != self.vocabulary
        ):
            raise V3ReserveCandidateError("MusicFM head identity drift")


def _payload_sha256(document: Mapping[str, object]) -> str:
    payload = dict(document)
    payload.pop("payload_sha256", None)
    return stable_json_sha256(payload)


def _implementation_sha256() -> str:
    return stable_json_sha256(
        {"source": Path(__file__).read_text(encoding="utf-8")}
    )


def _read_json(path: Path, label: str) -> Mapping[str, object]:
    candidate = Path(path).absolute()
    if candidate.is_symlink() or not candidate.is_file():
        raise V3ReserveCandidateError(f"{label} must be a concrete file")
    try:
        document = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V3ReserveCandidateError(f"{label} is invalid: {exc}") from exc
    if not isinstance(document, dict):
        raise V3ReserveCandidateError(f"{label} must contain a JSON object")
    return document


def _validate_protocol(path: Path) -> Mapping[str, object]:
    if sha256_path(Path(path)) != PROTOCOL_FILE_SHA256:
        raise V3ReserveCandidateError("reserve protocol file hash drift")
    protocol = load_reserve_protocol(Path(path))
    if (
        protocol.get("payload_sha256") != PROTOCOL_PAYLOAD_SHA256
        or protocol.get("selection_sha256") != EXPECTED_SELECTION_SHA256
        or protocol.get("shadow_labels_accessed") is not False
    ):
        raise V3ReserveCandidateError("reserve protocol binding drift")
    return protocol


def _validate_refinement(path: Path) -> Mapping[str, object]:
    document = _read_json(path, "reserve refinement")
    if (
        document.get("artifact_kind") != REFINEMENT_KIND
        or document.get("payload_sha256") != _payload_sha256(document)
        or document.get("payload_sha256") != REFINEMENT_PAYLOAD_SHA256
        or document.get("shadow_labels_accessed") is not False
        or document.get("development_gate", {}).get("passed") is not True
    ):
        raise V3ReserveCandidateError("reserve refinement binding drift")
    return document


def _validate_shadow_extraction_plan(
    path: Path, protocol: Mapping[str, object]
) -> Mapping[str, object]:
    document = _read_json(path, "shadow MusicFM extraction plan")
    binding = document.get("binding")
    if (
        document.get("artifact_kind") != MUSICFM_EXTRACTION_PLAN_KIND
        or document.get("payload_sha256") != _payload_sha256(document)
        or document.get("source_fingerprint") != SOURCE_FINGERPRINT
        or document.get("protocol_payload_sha256")
        != protocol["payload_sha256"]
        or document.get("protocol_selection_sha256")
        != protocol["selection_sha256"]
        or document.get("split") != "shadow"
        or document.get("tracks") != int(EXPECTED_SPLITS["shadow"]["tracks"])
        or document.get("track_ids_sha256")
        != EXPECTED_SPLITS["shadow"]["track_ids_sha256"]
        or not isinstance(binding, dict)
        or document.get("config_sha256") != stable_json_sha256(binding)
        or document.get("model_sha256") != MUSICFM_MODEL_SHA256
        or document.get("shadow_labels_accessed") is not False
        or document.get("opened_label_splits") != []
    ):
        raise V3ReserveCandidateError("shadow MusicFM extraction-plan drift")
    return document


def _protocol_entries(
    protocol: Mapping[str, object], split: str
) -> Tuple[Mapping[str, object], ...]:
    values = protocol.get("tracks")
    if not isinstance(values, list) or split not in EXPECTED_SPLITS:
        raise V3ReserveCandidateError("reserve track plan drift")
    selected = tuple(
        item
        for item in values
        if isinstance(item, dict) and item.get("split") == split
    )
    expected = EXPECTED_SPLITS[split]
    if (
        len(selected) != int(expected["tracks"])
        or len({int(item["artist_id"]) for item in selected})
        != int(expected["artists"])
    ):
        raise V3ReserveCandidateError(f"{split} reserve identity drift")
    return selected


def load_protocol_tags(
    metadata_root: Path,
    protocol: Mapping[str, object],
    splits: Sequence[str],
) -> Mapping[int, Tuple[str, ...]]:
    selected_splits = tuple(splits)
    if (
        not selected_splits
        or len(set(selected_splits)) != len(selected_splits)
        or any(split not in EXPECTED_SPLITS for split in selected_splits)
    ):
        raise V3ReserveCandidateError("reserve label selection is invalid")
    expected_artists = {
        int(item["track_id"]): int(item["artist_id"])
        for split in selected_splits
        for item in _protocol_entries(protocol, split)
    }
    excluded = {
        int(item["track_id"])
        for split in EXPECTED_SPLITS
        if split not in selected_splits
        for item in _protocol_entries(protocol, split)
    }
    files = {
        "train": "autotagging-train.tsv",
        "development": "autotagging-validation.tsv",
        "shadow": "autotagging-test.tsv",
    }
    labels: Dict[int, Tuple[str, ...]] = {}
    for split in selected_splits:
        path = (
            Path(metadata_root).absolute()
            / "data"
            / "splits"
            / "split-0"
            / files[split]
        )
        if path.is_symlink():
            raise V3ReserveCandidateError("label source may not be a symlink")
        path = path.resolve(strict=True)
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            if next(reader, None) != list(LABEL_HEADER):
                raise V3ReserveCandidateError("label source header drift")
            for line_number, row in enumerate(reader, 2):
                if len(row) < len(LABEL_HEADER):
                    raise V3ReserveCandidateError(
                        f"label row {line_number} is short"
                    )
                match = _ID_PATTERNS["track"].fullmatch(row[0])
                if match is None:
                    raise V3ReserveCandidateError(
                        f"label row {line_number} has malformed track ID"
                    )
                track_id = int(match.group(1))
                if track_id not in expected_artists:
                    continue
                artist = _ID_PATTERNS["artist"].fullmatch(row[1])
                tags = tuple(sorted(row[5:]))
                if (
                    artist is None
                    or int(artist.group(1)) != expected_artists[track_id]
                    or not tags
                    or len(tags) != len(set(tags))
                    or any(_TAG.fullmatch(tag) is None for tag in tags)
                    or track_id in labels
                ):
                    raise V3ReserveCandidateError(
                        f"label row {line_number} differs from reserve"
                    )
                labels[track_id] = tags
    if set(labels) != set(expected_artists) or set(labels).intersection(excluded):
        raise V3ReserveCandidateError("reserve label firewall failure")
    return labels


def build_label_targets(
    entries: Sequence[Mapping[str, object]],
    labels: Mapping[int, Sequence[str]],
) -> Tuple[Tuple[str, ...], np.ndarray]:
    vocabulary = tuple(
        sorted(
            {
                tag
                for item in entries
                for tag in labels[int(item["track_id"])]
            }
        )
    )
    positions = {tag: index for index, tag in enumerate(vocabulary)}
    targets = np.zeros((len(entries), len(vocabulary)), dtype=np.float32)
    for row, item in enumerate(entries):
        for tag in labels[int(item["track_id"])]:
            targets[row, positions[tag]] = 1.0
    if not len(vocabulary) or np.any(np.sum(targets, axis=0) <= 0.0):
        raise V3ReserveCandidateError("reserve training targets are invalid")
    return vocabulary, targets


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not len(array) or not np.all(np.isfinite(array)):
        raise V3ReserveCandidateError("embedding matrix is invalid")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise V3ReserveCandidateError("embedding matrix is invalid")
    return array / norms


def _pool(values: np.ndarray, mode: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or not len(array):
        raise V3ReserveCandidateError("CLAP section matrix is empty")
    pooled = np.max(array, axis=0) if mode == "max" else np.mean(array, axis=0)
    return _normalize_rows(pooled[None, :])[0].astype(np.float32)


def _clap_views(
    reader: FullTrackStoreReader,
    track_ids: Sequence[int],
    *,
    progress_label: Optional[str] = None,
) -> Mapping[str, np.ndarray]:
    positions = {track_id: row for row, track_id in enumerate(reader.track_ids)}
    try:
        rows = [positions[int(track_id)] for track_id in track_ids]
    except KeyError as exc:
        raise V3ReserveCandidateError(
            f"CLAP store is missing track {exc.args[0]}"
        ) from exc
    global_values = np.asarray(
        reader.global_embeddings[rows], dtype=np.float32
    ).copy()
    global_values /= np.linalg.norm(global_values, axis=1, keepdims=True)
    dimension = reader.binding.embedding_dim
    views = {
        "global": global_values,
        "salient": np.empty((len(rows), dimension), dtype=np.float32),
        "repeated": np.empty((len(rows), dimension), dtype=np.float32),
        "window_max": np.empty((len(rows), dimension), dtype=np.float32),
    }
    for row, track_id in enumerate(track_ids):
        track = reader.read_track(int(track_id))
        views["salient"][row] = _pool(track.salient_sections, "mean")
        views["repeated"][row] = _pool(track.repeated_sections, "mean")
        views["window_max"][row] = _pool(track.window_embeddings, "max")
        if progress_label and (row + 1) % 5_000 == 0:
            print(
                f"{progress_label}: {row + 1}/{len(track_ids)} CLAP tracks",
                flush=True,
            )
    return views


def _sparse_four(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    indices = np.unique(
        np.linspace(0, len(array) - 1, min(4, len(array))).astype(np.int32)
    )
    return _normalize_rows(np.mean(array[indices], axis=0, keepdims=True))[
        0
    ].astype(np.float32)


def _open_clap_store(path: Path) -> FullTrackStoreReader:
    return _open_bound_store(
        Path(path),
        expected_manifest_file_sha256=CLAP_MANIFEST_FILE_SHA256,
        expected_binding=EXPECTED_CLAP_BINDING,
    )


def _open_musicfm_store(
    path: Path,
    *,
    config_sha256: str,
    expected_track_ids: Optional[Sequence[int]] = None,
) -> FullTrackStoreReader:
    reader = FullTrackStoreReader(
        Path(path),
        expected_source_fingerprint=SOURCE_FINGERPRINT,
        expected_config_sha256=config_sha256,
        expected_model_sha256=MUSICFM_MODEL_SHA256,
    )
    if (
        reader.binding.embedding_dim != 1024
        or (
            expected_track_ids is not None
            and set(reader.track_ids)
            != {int(track_id) for track_id in expected_track_ids}
        )
    ):
        reader.close()
        raise V3ReserveCandidateError("MusicFM store identity drift")
    return reader


def _musicfm_training(
    reader: FullTrackStoreReader,
) -> Tuple[np.ndarray, np.ndarray]:
    if len(reader.track_ids) != MUSICFM_TRAIN_TRACKS:
        raise V3ReserveCandidateError("MusicFM training-store size drift")
    values = np.empty((len(reader.track_ids), 1024), dtype=np.float32)
    for row, track_id in enumerate(reader.track_ids):
        values[row] = _sparse_four(
            reader.read_track(int(track_id)).window_embeddings
        )
        if (row + 1) % 2_000 == 0:
            print(
                f"training: {row + 1}/{len(reader.track_ids)} MusicFM tracks",
                flush=True,
            )
    return np.asarray(reader.track_ids, dtype=np.int64), values


def _musicfm_partition(
    reader: FullTrackStoreReader, track_ids: Sequence[int]
) -> np.ndarray:
    positions = {track_id: row for row, track_id in enumerate(reader.track_ids)}
    try:
        rows = [positions[int(track_id)] for track_id in track_ids]
    except KeyError as exc:
        raise V3ReserveCandidateError(
            f"MusicFM store is missing track {exc.args[0]}"
        ) from exc
    return _normalize_rows(
        np.asarray(reader.global_embeddings[rows], dtype=np.float32)
    )


def _partition_fold(artist_id: int) -> int:
    return (
        int(
            stable_json_sha256(
                {"seed": FOLD_SEED, "artist_id": int(artist_id)}
            )[:16],
            16,
        )
        % SHADOW_FOLDS
    )


def build_partition_data(
    entries: Sequence[Mapping[str, object]],
    labels: Mapping[int, Sequence[str]],
    clap_reader: FullTrackStoreReader,
    *,
    split: str,
) -> PartitionData:
    positions = {
        track_id: position for position, track_id in enumerate(clap_reader.track_ids)
    }
    try:
        ordered = tuple(
            sorted(entries, key=lambda item: positions[int(item["track_id"])])
        )
    except KeyError as exc:
        raise V3ReserveCandidateError(
            f"CLAP store is missing track {exc.args[0]}"
        ) from exc
    track_ids = np.asarray(
        [int(item["track_id"]) for item in ordered], dtype=np.int64
    )
    artist_ids = np.asarray(
        [int(item["artist_id"]) for item in ordered], dtype=np.int64
    )
    if set(labels) != {int(value) for value in track_ids}:
        raise V3ReserveCandidateError("partition labels differ from reserve")
    count = len(track_ids)
    budget = _BudgetCache(
        clap_reader,
        track_ids,
        budget=MAXSIM_BUDGET,
        max_bytes=MAX_FEATURE_CACHE_BYTES,
    )
    store_rows = [positions[int(track_id)] for track_id in track_ids]
    globals_ = np.asarray(
        clap_reader.global_embeddings[store_rows], dtype=np.float32
    ).copy()
    globals_ /= np.linalg.norm(globals_, axis=1, keepdims=True)
    globals_ = globals_.astype(np.float64)
    global_orders = np.full((count, count - 1), -1, dtype=np.int32)
    global_lengths = np.zeros(count, dtype=np.int32)
    pools = np.empty((count, CANDIDATE_POOL), dtype=np.int32)
    baseline_scores = np.empty((count, CANDIDATE_POOL), dtype=np.float32)
    relevance = np.zeros((count, count), dtype=np.float32)
    for query in range(count):
        eligible = np.flatnonzero(
            (track_ids != track_ids[query])
            & (artist_ids != artist_ids[query])
        )
        global_scores = globals_[eligible] @ globals_[query]
        order = eligible[np.lexsort((track_ids[eligible], -global_scores))]
        if len(order) < CANDIDATE_POOL:
            raise V3ReserveCandidateError("candidate universe is too small")
        global_orders[query, : len(order)] = order
        global_lengths[query] = len(order)
        pool = order[:CANDIDATE_POOL]
        pools[query] = pool
        baseline_scores[query] = _score_channels(
            query, pool, globals_, budget
        )[3].astype(np.float32)
        query_tags = labels[int(track_ids[query])]
        for candidate in eligible:
            relevance[query, candidate] = _tag_jaccard_relevance(
                query_tags,
                labels[int(track_ids[candidate])],
                min_shared_tags=2,
                min_tag_jaccard=0.25,
            )
    data = PartitionData(
        split=split,
        track_ids=track_ids,
        artist_ids=artist_ids,
        query_folds=np.asarray(
            [_partition_fold(int(artist)) for artist in artist_ids],
            dtype=np.int8,
        ),
        global_orders=global_orders,
        global_lengths=global_lengths,
        pools=pools,
        baseline_scores=baseline_scores,
        relevance=relevance,
    )
    data.validate()
    return data


def _raw_predictions(head: SemanticHead, values: np.ndarray) -> np.ndarray:
    head.validate()
    transformed = (
        np.asarray(values, dtype=np.float64) - head.input_mean
    ) / head.input_scale
    transformed = _normalize_rows(transformed)
    return np.clip(
        transformed @ head.coefficients + head.prior,
        0.0,
        1.0,
    ).astype(np.float32)


def _selected_profile(predictions: np.ndarray, idf: np.ndarray) -> np.ndarray:
    indices = np.argpartition(-predictions, TOP_TAGS - 1, axis=1)[
        :, :TOP_TAGS
    ]
    values = np.zeros_like(predictions)
    np.put_along_axis(values, indices, 1.0, axis=1)
    return _normalize_rows(values * np.power(idf, TOP_IDF_POWER))


def set_aware_similarity(
    predictions: np.ndarray, idf: np.ndarray
) -> np.ndarray:
    values = np.asarray(predictions, dtype=np.float32)
    weights = np.asarray(idf, dtype=np.float32)
    if (
        values.ndim != 2
        or weights.shape != (values.shape[1],)
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(weights))
    ):
        raise V3ReserveCandidateError("semantic predictions are invalid")
    top = _selected_profile(values, weights)
    weighted = values * weights
    intersection = weighted @ values.T
    mass = np.sum(weighted, axis=1)
    union = mass[:, None] + mass[None, :] - intersection
    soft_jaccard = intersection / np.maximum(union, 1e-12)
    return TOP_SHARE * (top @ top.T) + (1.0 - TOP_SHARE) * soft_jaccard


def _standardize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    deviation = float(np.std(array))
    return (array - float(np.mean(array))) / (
        deviation if deviation > 1e-8 else 1.0
    )


def calibrated_candidate_scores(
    clap_audio: np.ndarray,
    musicfm_audio: np.ndarray,
    clap_semantic: np.ndarray,
    musicfm_semantic: np.ndarray,
) -> np.ndarray:
    channels = tuple(
        np.asarray(values, dtype=np.float64)
        for values in (
            clap_audio,
            musicfm_audio,
            clap_semantic,
            musicfm_semantic,
        )
    )
    if (
        any(values.ndim != 1 for values in channels)
        or len({values.shape for values in channels}) != 1
        or not np.all(np.isfinite(np.stack(channels)))
    ):
        raise V3ReserveCandidateError("candidate score channels are invalid")
    audio = (
        (1.0 - MUSICFM_AUDIO_SHARE) * _standardize(channels[0])
        + MUSICFM_AUDIO_SHARE * _standardize(channels[1])
    )
    semantic = (
        (1.0 - MUSICFM_SEMANTIC_SHARE) * _standardize(channels[2])
        + MUSICFM_SEMANTIC_SHARE * _standardize(channels[3])
    )
    return (1.0 - SEMANTIC_WEIGHT) * audio + SEMANTIC_WEIGHT * semantic


def reserve_candidate_rankings(
    data: PartitionData,
    model: ReserveCandidateModel,
    clap_views: Mapping[str, np.ndarray],
    musicfm: np.ndarray,
) -> Sequence[np.ndarray]:
    data.validate()
    model.validate()
    if set(clap_views) != set(CLAP_HEAD_NAMES):
        raise V3ReserveCandidateError("CLAP partition views are incomplete")
    count = len(data.track_ids)
    if any(np.asarray(values).shape[0] != count for values in clap_views.values()):
        raise V3ReserveCandidateError("CLAP partition view length drift")
    music = _normalize_rows(musicfm)
    global_clap = _normalize_rows(clap_views["global"])
    window_max = _normalize_rows(clap_views["window_max"])
    global_clap_similarity = global_clap @ global_clap.T
    clap_audio = (
        (1.0 - CLAP_WINDOW_MAX_AUDIO_SHARE) * global_clap_similarity
        + CLAP_WINDOW_MAX_AUDIO_SHARE * (window_max @ window_max.T)
    )
    music_audio = music @ music.T
    clap_predictions = {
        name: _raw_predictions(model.clap_heads[name], clap_views[name])
        for name in CLAP_HEAD_NAMES
    }
    global_salient = (
        CLAP_GLOBAL_SALIENT_SHARE * clap_predictions["global"]
        + (1.0 - CLAP_GLOBAL_SALIENT_SHARE)
        * clap_predictions["salient"]
    )
    four_view_max = np.max(
        np.stack(tuple(clap_predictions.values())), axis=0
    )
    clap_semantic = (
        (1.0 - CLAP_FOUR_VIEW_SHARE)
        * set_aware_similarity(global_salient, model.clap_heads["global"].idf)
        + CLAP_FOUR_VIEW_SHARE
        * set_aware_similarity(four_view_max, model.clap_heads["global"].idf)
    )
    music_predictions = _raw_predictions(model.musicfm_head, music)
    music_semantic = set_aware_similarity(
        music_predictions, model.musicfm_head.idf
    )
    rankings = []
    for query in range(count):
        eligible = data.global_orders[
            query, : int(data.global_lengths[query])
        ]
        scores = calibrated_candidate_scores(
            clap_audio[query, eligible],
            music_audio[query, eligible],
            clap_semantic[query, eligible],
            music_semantic[query, eligible],
        )
        top = np.argpartition(-scores, 9)[:10]
        top = top[np.argsort(-scores[top], kind="stable")]
        rankings.append(eligible[top].astype(np.int32, copy=False))
    return rankings


def evaluate_rankings(
    data: PartitionData, candidate_rankings: Sequence[np.ndarray]
) -> Mapping[str, object]:
    data.validate()
    if len(candidate_rankings) != len(data.track_ids):
        raise V3ReserveCandidateError("candidate ranking count drift")
    baseline_values: Dict[str, list[float]] = {metric: [] for metric in METRICS}
    candidate_values: Dict[str, list[float]] = {metric: [] for metric in METRICS}
    folds = []
    for query in range(len(data.track_ids)):
        relevant = {
            int(data.track_ids[position]): float(grade)
            for position, grade in enumerate(data.relevance[query])
            if grade > 0.0
        }
        if not relevant:
            continue
        pool = data.pools[query]
        global_order = data.global_orders[
            query, : int(data.global_lengths[query])
        ]
        baseline_order = _method_ranking(
            data.baseline_scores[query], pool, global_order
        )[:10]
        candidate_order = np.asarray(candidate_rankings[query], dtype=np.int64)
        if (
            candidate_order.shape != (10,)
            or len(np.unique(candidate_order)) != 10
            or np.any(candidate_order < 0)
            or np.any(candidate_order >= len(data.track_ids))
        ):
            raise V3ReserveCandidateError("candidate ranking is invalid")
        for destination, order in (
            (baseline_values, baseline_order),
            (candidate_values, candidate_order),
        ):
            metrics = _query_metrics(
                [int(data.track_ids[position]) for position in order],
                relevant,
                recall_cutoff=10,
                ndcg_cutoff=10,
            )
            for metric in METRICS:
                destination[metric].append(float(getattr(metrics, metric)))
        folds.append(int(data.query_folds[query]))
    fold_array = np.asarray(folds, dtype=np.int8)
    if not len(fold_array) or set(fold_array) != set(range(SHADOW_FOLDS)):
        raise V3ReserveCandidateError("evaluation folds are incomplete")
    baseline_mean = {
        metric: float(np.mean(baseline_values[metric])) for metric in METRICS
    }
    candidate_mean = {
        metric: float(np.mean(candidate_values[metric])) for metric in METRICS
    }
    positive_folds = {metric: 0 for metric in METRICS}
    worst_fold = {metric: float("inf") for metric in METRICS}
    fold_results = {}
    for fold_index in range(SHADOW_FOLDS):
        selected = fold_array == fold_index
        relative = {}
        for metric in METRICS:
            baseline_fold = float(
                np.mean(np.asarray(baseline_values[metric])[selected])
            )
            candidate_fold = float(
                np.mean(np.asarray(candidate_values[metric])[selected])
            )
            relative[metric] = candidate_fold / baseline_fold - 1.0
            positive_folds[metric] += int(relative[metric] > 0.0)
            worst_fold[metric] = min(worst_fold[metric], relative[metric])
        fold_results[str(fold_index)] = {
            "queries": int(np.count_nonzero(selected)),
            "relative_delta": relative,
        }
    return {
        "queries": len(fold_array),
        "baseline": baseline_mean,
        "candidate": candidate_mean,
        "relative_delta": {
            metric: candidate_mean[metric] / baseline_mean[metric] - 1.0
            for metric in METRICS
        },
        "paired_delta": {
            metric: _paired_bootstrap_delta(
                baseline_values[metric],
                candidate_values[metric],
                iterations=BOOTSTRAP_ITERATIONS,
                seed=BOOTSTRAP_SEED,
            )
            for metric in METRICS
        },
        "positive_folds": positive_folds,
        "worst_fold_relative_delta": worst_fold,
        "folds": fold_results,
    }


def _gate(evaluation: Mapping[str, object], *, shadow: bool) -> Mapping[str, object]:
    relative = evaluation["relative_delta"]
    checks = {
        "primary_relative_gain": (
            float(relative[PRIMARY_METRIC]) >= MIN_RECALL_GAIN
        ),
        "primary_paired_ci_above_zero": (
            float(
                evaluation["paired_delta"][PRIMARY_METRIC][
                    "paired_bootstrap_ci95"
                ][0]
            )
            > 0.0
        ),
        "primary_positive_folds": (
            int(evaluation["positive_folds"][PRIMARY_METRIC])
            >= MIN_POSITIVE_FOLDS
        ),
        "primary_worst_fold": (
            float(evaluation["worst_fold_relative_delta"][PRIMARY_METRIC])
            >= -MAX_FOLD_RECALL_REGRESSION
        ),
        "safety_metrics": all(
            float(relative[metric]) >= -MAX_SAFETY_REGRESSION
            for metric in METRICS
            if metric != PRIMARY_METRIC
        ),
    }
    passed = all(checks.values())
    result: Dict[str, object] = {"checks": checks}
    if shadow:
        result.update(
            {
                "automated_passed": passed,
                "human_pilot_required": True,
                "promotion_allowed": False,
            }
        )
    else:
        result.update(
            {
                "passed": passed,
                "decision": (
                    "freeze_for_one_time_final_shadow_audit"
                    if passed
                    else "do_not_open_final_shadow"
                ),
            }
        )
    return result


def development_gate(evaluation: Mapping[str, object]) -> Mapping[str, object]:
    return _gate(evaluation, shadow=False)


def shadow_gate(evaluation: Mapping[str, object]) -> Mapping[str, object]:
    return _gate(evaluation, shadow=True)


def _head_arrays(prefix: str, head: SemanticHead) -> Mapping[str, np.ndarray]:
    head.validate()
    return {
        f"{prefix}__input_mean": head.input_mean.astype(np.float64),
        f"{prefix}__input_scale": head.input_scale.astype(np.float64),
        f"{prefix}__coefficients": head.coefficients.astype(np.float64),
        f"{prefix}__prior": head.prior.astype(np.float64),
        f"{prefix}__idf": head.idf.astype(np.float64),
    }


def _model_arrays(model: ReserveCandidateModel) -> Mapping[str, np.ndarray]:
    model.validate()
    arrays: Dict[str, np.ndarray] = {
        "vocabulary": np.asarray(model.vocabulary, dtype=np.str_)
    }
    for name, head in model.clap_heads.items():
        arrays.update(_head_arrays(f"clap_{name}", head))
    arrays.update(_head_arrays("musicfm", model.musicfm_head))
    return arrays


def _load_head(
    archive: np.lib.npyio.NpzFile,
    prefix: str,
    representation: str,
    vocabulary: Tuple[str, ...],
) -> SemanticHead:
    return SemanticHead(
        representation=representation,
        ridge=RIDGE,
        vocabulary=vocabulary,
        input_mean=np.asarray(archive[f"{prefix}__input_mean"], dtype=np.float64),
        input_scale=np.asarray(
            archive[f"{prefix}__input_scale"], dtype=np.float64
        ),
        coefficients=np.asarray(
            archive[f"{prefix}__coefficients"], dtype=np.float64
        ),
        prior=np.asarray(archive[f"{prefix}__prior"], dtype=np.float64),
        idf=np.asarray(archive[f"{prefix}__idf"], dtype=np.float64),
    )


def load_candidate_model(path: Path) -> ReserveCandidateModel:
    with np.load(Path(path), allow_pickle=False) as archive:
        vocabulary = tuple(str(value) for value in archive["vocabulary"])
        model = ReserveCandidateModel(
            vocabulary=vocabulary,
            clap_heads={
                name: _load_head(
                    archive, f"clap_{name}", "clap", vocabulary
                )
                for name in CLAP_HEAD_NAMES
            },
            musicfm_head=_load_head(
                archive, "musicfm", "musicfm", vocabulary
            ),
        )
    model.validate()
    return model


def _verify_expected_development(evaluation: Mapping[str, object]) -> None:
    for metric, expected in EXPECTED_DEVELOPMENT_RELATIVE.items():
        if not np.isclose(
            float(evaluation["relative_delta"][metric]),
            expected,
            rtol=0.0,
            atol=1e-12,
        ):
            raise V3ReserveCandidateError(
                f"reserve candidate does not reproduce {metric}"
            )


def build_development_candidate(
    *,
    metadata_root: Path,
    protocol_path: Path,
    refinement_report_path: Path,
    clap_store: Path,
    musicfm_train_store: Path,
    musicfm_development_store: Path,
    model_output: Path,
    metadata_output: Path,
    report_output: Path,
) -> Mapping[str, object]:
    outputs = (Path(model_output), Path(metadata_output), Path(report_output))
    if any(path.exists() for path in outputs):
        raise V3ReserveCandidateError("candidate outputs already exist")
    protocol = _validate_protocol(protocol_path)
    refinement = _validate_refinement(refinement_report_path)
    train_entries = _protocol_entries(protocol, "train")
    development_entries = _protocol_entries(protocol, "development")
    labels = load_protocol_tags(
        metadata_root, protocol, ("train", "development")
    )
    clap_reader = _open_clap_store(clap_store)
    train_music_reader = _open_musicfm_store(
        musicfm_train_store,
        config_sha256=MUSICFM_TRAIN_CONFIG_SHA256,
    )
    development_music_reader = _open_musicfm_store(
        musicfm_development_store,
        config_sha256=MUSICFM_SPARSE_CONFIG_SHA256,
        expected_track_ids=[
            int(item["track_id"]) for item in development_entries
        ],
    )
    try:
        positions = {
            track_id: row for row, track_id in enumerate(clap_reader.track_ids)
        }
        train_entries = tuple(
            sorted(
                train_entries,
                key=lambda item: positions[int(item["track_id"])],
            )
        )
        vocabulary, targets = build_label_targets(train_entries, labels)
        train_ids = np.asarray(
            [int(item["track_id"]) for item in train_entries], dtype=np.int64
        )
        train_views = _clap_views(
            clap_reader, train_ids, progress_label="training"
        )
        clap_heads = {}
        for name in CLAP_HEAD_NAMES:
            print(f"fitting CLAP {name} head", flush=True)
            clap_heads[name] = fit_semantic_head(
                train_views[name],
                targets,
                vocabulary,
                representation="clap",
                ridge=RIDGE,
            )
        del train_views
        music_ids, music_values = _musicfm_training(train_music_reader)
        target_rows = {
            int(track_id): row for row, track_id in enumerate(train_ids)
        }
        try:
            music_targets = targets[
                [target_rows[int(track_id)] for track_id in music_ids]
            ]
        except KeyError as exc:
            raise V3ReserveCandidateError(
                f"MusicFM training track outside reserve: {exc.args[0]}"
            ) from exc
        print("fitting MusicFM head", flush=True)
        musicfm_head = fit_semantic_head(
            music_values,
            music_targets,
            vocabulary,
            representation="musicfm",
            ridge=RIDGE,
        )
        del music_values, music_targets
        model = ReserveCandidateModel(
            vocabulary=vocabulary,
            clap_heads=clap_heads,
            musicfm_head=musicfm_head,
        )
        model.validate()
        development_labels = {
            int(item["track_id"]): labels[int(item["track_id"])]
            for item in development_entries
        }
        data = build_partition_data(
            development_entries,
            development_labels,
            clap_reader,
            split="development",
        )
        development_views = _clap_views(
            clap_reader, data.track_ids, progress_label="development"
        )
        development_music = _musicfm_partition(
            development_music_reader, data.track_ids
        )
        evaluation = evaluate_rankings(
            data,
            reserve_candidate_rankings(
                data, model, development_views, development_music
            ),
        )
        _verify_expected_development(evaluation)
        gate = development_gate(evaluation)
        if not gate["passed"]:
            raise V3ReserveCandidateError("reserve development gate failed")
        _write_npz_exclusive(Path(model_output), _model_arrays(model))
        report: Dict[str, object] = {
            "schema_version": 1,
            "artifact_kind": DEVELOPMENT_REPORT_KIND,
            "evidence_scope": EVIDENCE_SCOPE,
            "evidence_status": "development_only",
            "source_fingerprint": SOURCE_FINGERPRINT,
            "protocol_file_sha256": PROTOCOL_FILE_SHA256,
            "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
            "protocol_selection_sha256": EXPECTED_SELECTION_SHA256,
            "refinement_file_sha256": sha256_path(refinement_report_path),
            "refinement_payload_sha256": refinement["payload_sha256"],
            "opened_label_splits": ["train", "development"],
            "shadow_labels_accessed": False,
            "shadow_evaluation_accessed": False,
            "development_consumed_for_selection": True,
            "frozen_method": {
                "clap_heads": list(CLAP_HEAD_NAMES),
                "ridge": RIDGE,
                "top_tags": TOP_TAGS,
                "top_idf_power": TOP_IDF_POWER,
                "top_share": TOP_SHARE,
                "global_salient_prediction_share": (
                    CLAP_GLOBAL_SALIENT_SHARE
                ),
                "four_view_semantic_share": CLAP_FOUR_VIEW_SHARE,
                "clap_window_max_audio_share": (
                    CLAP_WINDOW_MAX_AUDIO_SHARE
                ),
                "musicfm_audio_share": MUSICFM_AUDIO_SHARE,
                "musicfm_semantic_share": MUSICFM_SEMANTIC_SHARE,
                "semantic_weight": SEMANTIC_WEIGHT,
                "score_calibration": "per_query_channel_zscore",
            },
            "train_tracks": len(train_entries),
            "musicfm_train_tracks": len(music_ids),
            "development_tracks": len(development_entries),
            "tag_count": len(vocabulary),
            "evaluation": evaluation,
            "development_gate": gate,
            "model_npz_sha256": sha256_path(model_output),
            "clap_manifest_file_sha256": sha256_path(
                Path(clap_store) / "store.sealed.json"
            ),
            "musicfm_train_manifest_file_sha256": sha256_path(
                Path(musicfm_train_store) / "store.sealed.json"
            ),
            "musicfm_development_manifest_file_sha256": sha256_path(
                Path(musicfm_development_store) / "store.sealed.json"
            ),
            "promotion_allowed": False,
        }
        report["payload_sha256"] = stable_json_sha256(report)
        _write_json_exclusive(Path(report_output), report)
        metadata: Dict[str, object] = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "artifact_kind": MODEL_KIND,
            "source_fingerprint": SOURCE_FINGERPRINT,
            "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
            "protocol_selection_sha256": EXPECTED_SELECTION_SHA256,
            "model_npz_sha256": sha256_path(model_output),
            "development_report_file_sha256": sha256_path(report_output),
            "development_report_payload_sha256": report["payload_sha256"],
            "tag_count": len(vocabulary),
            "shadow_labels_accessed": False,
            "promotion_allowed": False,
        }
        metadata["payload_sha256"] = stable_json_sha256(metadata)
        _write_json_exclusive(Path(metadata_output), metadata)
        return report
    finally:
        clap_reader.close()
        train_music_reader.close()
        development_music_reader.close()


def freeze_candidate(
    *,
    protocol_path: Path,
    refinement_report_path: Path,
    clap_store: Path,
    musicfm_train_store: Path,
    musicfm_development_store: Path,
    model_path: Path,
    metadata_path: Path,
    development_report_path: Path,
    shadow_extraction_plan_path: Path,
    output: Path,
) -> Mapping[str, object]:
    if Path(output).exists():
        raise V3ReserveCandidateError("freeze output already exists")
    protocol = _validate_protocol(protocol_path)
    refinement = _validate_refinement(refinement_report_path)
    extraction_plan = _validate_shadow_extraction_plan(
        shadow_extraction_plan_path, protocol
    )
    metadata = _read_json(metadata_path, "candidate metadata")
    report = _read_json(development_report_path, "development report")
    if (
        metadata.get("artifact_kind") != MODEL_KIND
        or report.get("artifact_kind") != DEVELOPMENT_REPORT_KIND
        or metadata.get("payload_sha256") != _payload_sha256(metadata)
        or report.get("payload_sha256") != _payload_sha256(report)
        or metadata.get("model_npz_sha256") != sha256_path(model_path)
        or metadata.get("development_report_file_sha256")
        != sha256_path(development_report_path)
        or report.get("refinement_payload_sha256")
        != refinement["payload_sha256"]
        or not report.get("development_gate", {}).get("passed")
        or report.get("shadow_labels_accessed") is not False
    ):
        raise V3ReserveCandidateError("candidate is not eligible to freeze")
    load_candidate_model(model_path)
    document: Dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": FREEZE_KIND,
        "source_fingerprint": SOURCE_FINGERPRINT,
        "protocol_file_sha256": sha256_path(protocol_path),
        "protocol_payload_sha256": protocol["payload_sha256"],
        "protocol_selection_sha256": protocol["selection_sha256"],
        "refinement_file_sha256": sha256_path(refinement_report_path),
        "refinement_payload_sha256": refinement["payload_sha256"],
        "candidate_model_file_sha256": sha256_path(model_path),
        "candidate_metadata_file_sha256": sha256_path(metadata_path),
        "candidate_metadata_payload_sha256": metadata["payload_sha256"],
        "development_report_file_sha256": sha256_path(
            development_report_path
        ),
        "development_report_payload_sha256": report["payload_sha256"],
        "shadow_extraction_plan_payload_sha256": extraction_plan[
            "payload_sha256"
        ],
        "shadow_musicfm_config_sha256": extraction_plan["config_sha256"],
        "clap_manifest_file_sha256": sha256_path(
            Path(clap_store) / "store.sealed.json"
        ),
        "musicfm_train_manifest_file_sha256": sha256_path(
            Path(musicfm_train_store) / "store.sealed.json"
        ),
        "musicfm_development_manifest_file_sha256": sha256_path(
            Path(musicfm_development_store) / "store.sealed.json"
        ),
        "frozen_method": report["frozen_method"],
        "scoring_method_sha256": stable_json_sha256(report["frozen_method"]),
        "implementation_sha256": _implementation_sha256(),
        "development_evaluation": {
            "relative_delta": report["evaluation"]["relative_delta"],
            "paired_recall_ci95": report["evaluation"]["paired_delta"][
                PRIMARY_METRIC
            ]["paired_bootstrap_ci95"],
            "positive_folds": report["evaluation"]["positive_folds"],
            "worst_fold_relative_delta": report["evaluation"][
                "worst_fold_relative_delta"
            ],
        },
        "shadow_track_ids_sha256": EXPECTED_SPLITS["shadow"][
            "track_ids_sha256"
        ],
        "shadow_gate": {
            "minimum_recall_relative_gain": MIN_RECALL_GAIN,
            "paired_recall_ci_must_exclude_zero": True,
            "minimum_positive_recall_folds": MIN_POSITIVE_FOLDS,
            "maximum_fold_recall_regression": MAX_FOLD_RECALL_REGRESSION,
            "maximum_mrr_ndcg_regression": MAX_SAFETY_REGRESSION,
        },
        "development_result_seen": True,
        "shadow_labels_accessed": False,
        "promotion_allowed": False,
    }
    document["payload_sha256"] = stable_json_sha256(document)
    _write_json_exclusive(Path(output), document)
    return document


def _verify_freeze(
    freeze_path: Path,
    *,
    protocol: Mapping[str, object],
    model_path: Path,
    metadata_path: Path,
    development_report_path: Path,
    shadow_extraction_plan_path: Path,
) -> Mapping[str, object]:
    freeze = _read_json(freeze_path, "reserve shadow freeze")
    extraction_plan = _read_json(
        shadow_extraction_plan_path, "shadow MusicFM extraction plan"
    )
    expected = {
        "artifact_kind": FREEZE_KIND,
        "payload_sha256": _payload_sha256(freeze),
        "protocol_payload_sha256": protocol["payload_sha256"],
        "protocol_selection_sha256": protocol["selection_sha256"],
        "candidate_model_file_sha256": sha256_path(model_path),
        "candidate_metadata_file_sha256": sha256_path(metadata_path),
        "development_report_file_sha256": sha256_path(
            development_report_path
        ),
        "shadow_extraction_plan_payload_sha256": extraction_plan[
            "payload_sha256"
        ],
        "shadow_musicfm_config_sha256": extraction_plan["config_sha256"],
        "implementation_sha256": _implementation_sha256(),
        "shadow_labels_accessed": False,
    }
    drift = {
        key: (value, freeze.get(key))
        for key, value in expected.items()
        if freeze.get(key) != value
    }
    if drift:
        raise V3ReserveCandidateError(f"shadow freeze binding drift: {drift}")
    return freeze


def audit_frozen_shadow(
    *,
    metadata_root: Path,
    protocol_path: Path,
    clap_store: Path,
    musicfm_shadow_store: Path,
    model_path: Path,
    metadata_path: Path,
    development_report_path: Path,
    shadow_extraction_plan_path: Path,
    freeze_path: Path,
    output: Path,
    audit_state_path: Path,
) -> Mapping[str, object]:
    if Path(output).exists():
        raise V3ReserveCandidateError("shadow audit output already exists")
    if Path(audit_state_path).exists():
        raise V3ReserveCandidateError(
            "shadow audit state already exists; refusing reopen"
        )
    protocol = _validate_protocol(protocol_path)
    metadata = _read_json(metadata_path, "candidate metadata")
    development = _read_json(development_report_path, "development report")
    extraction_plan = _validate_shadow_extraction_plan(
        shadow_extraction_plan_path, protocol
    )
    if (
        metadata.get("payload_sha256") != _payload_sha256(metadata)
        or development.get("payload_sha256") != _payload_sha256(development)
        or not development.get("development_gate", {}).get("passed")
    ):
        raise V3ReserveCandidateError("pre-audit candidate evidence is invalid")
    freeze = _verify_freeze(
        freeze_path,
        protocol=protocol,
        model_path=model_path,
        metadata_path=metadata_path,
        development_report_path=development_report_path,
        shadow_extraction_plan_path=shadow_extraction_plan_path,
    )
    model = load_candidate_model(model_path)
    shadow_entries = _protocol_entries(protocol, "shadow")
    clap_reader = _open_clap_store(clap_store)
    music_reader = _open_musicfm_store(
        musicfm_shadow_store,
        config_sha256=str(extraction_plan["config_sha256"]),
        expected_track_ids=[
            int(item["track_id"]) for item in shadow_entries
        ],
    )
    opened_state: Dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": SHADOW_STATE_KIND,
        "status": "opened",
        "protocol_payload_sha256": protocol["payload_sha256"],
        "freeze_file_sha256": sha256_path(freeze_path),
        "freeze_payload_sha256": freeze["payload_sha256"],
        "candidate_model_file_sha256": sha256_path(model_path),
        "shadow_labels_accessed": True,
        "shadow_evaluation_completed": False,
        "promotion_allowed": False,
    }
    opened_state["payload_sha256"] = stable_json_sha256(opened_state)
    try:
        _write_json_exclusive(Path(audit_state_path), opened_state)
        labels = load_protocol_tags(metadata_root, protocol, ("shadow",))
        data = build_partition_data(
            shadow_entries, labels, clap_reader, split="shadow"
        )
        views = _clap_views(
            clap_reader, data.track_ids, progress_label="shadow"
        )
        music = _musicfm_partition(music_reader, data.track_ids)
        evaluation = evaluate_rankings(
            data, reserve_candidate_rankings(data, model, views, music)
        )
        gate = shadow_gate(evaluation)
        report: Dict[str, object] = {
            "schema_version": 1,
            "artifact_kind": SHADOW_REPORT_KIND,
            "evidence_scope": EVIDENCE_SCOPE,
            "evidence_status": "independent_shadow_audit",
            "source_fingerprint": SOURCE_FINGERPRINT,
            "protocol_payload_sha256": protocol["payload_sha256"],
            "protocol_selection_sha256": protocol["selection_sha256"],
            "freeze_file_sha256": sha256_path(freeze_path),
            "freeze_payload_sha256": freeze["payload_sha256"],
            "candidate_model_file_sha256": sha256_path(model_path),
            "musicfm_shadow_manifest_file_sha256": sha256_path(
                Path(musicfm_shadow_store) / "store.sealed.json"
            ),
            "opened_label_splits": ["shadow"],
            "shadow_labels_accessed": True,
            "shadow_evaluation_accessed": True,
            "shadow_tracks": len(shadow_entries),
            "shadow_fold_seed": FOLD_SEED,
            "evaluation": evaluation,
            "shadow_gate": gate,
            "listening_pack_allowed": gate["automated_passed"],
            "human_pilot_required": True,
            "promotion_allowed": False,
        }
        report["payload_sha256"] = stable_json_sha256(report)
        _write_json_exclusive(Path(output), report)
        completed = dict(opened_state)
        completed.update(
            {
                "status": "completed",
                "shadow_evaluation_completed": True,
                "report_file_sha256": sha256_path(output),
                "report_payload_sha256": report["payload_sha256"],
            }
        )
        completed.pop("payload_sha256", None)
        completed["payload_sha256"] = stable_json_sha256(completed)
        _replace_json(Path(audit_state_path), completed)
        return report
    finally:
        clap_reader.close()
        music_reader.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-development")
    for name in (
        "metadata-root",
        "protocol",
        "refinement-report",
        "clap-store",
        "musicfm-train-store",
        "musicfm-development-store",
        "model-output",
        "metadata-output",
        "report-output",
    ):
        build.add_argument(f"--{name}", required=True)
    freeze = commands.add_parser("freeze-shadow")
    for name in (
        "protocol",
        "refinement-report",
        "clap-store",
        "musicfm-train-store",
        "musicfm-development-store",
        "model",
        "metadata",
        "development-report",
        "shadow-extraction-plan",
        "output",
    ):
        freeze.add_argument(f"--{name}", required=True)
    audit = commands.add_parser("audit-shadow")
    for name in (
        "metadata-root",
        "protocol",
        "clap-store",
        "musicfm-shadow-store",
        "model",
        "metadata",
        "development-report",
        "shadow-extraction-plan",
        "freeze",
        "output",
        "audit-state",
    ):
        audit.add_argument(f"--{name}", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "build-development":
        result = build_development_candidate(
            metadata_root=Path(args.metadata_root),
            protocol_path=Path(args.protocol),
            refinement_report_path=Path(args.refinement_report),
            clap_store=Path(args.clap_store),
            musicfm_train_store=Path(args.musicfm_train_store),
            musicfm_development_store=Path(args.musicfm_development_store),
            model_output=Path(args.model_output),
            metadata_output=Path(args.metadata_output),
            report_output=Path(args.report_output),
        )
        summary = {
            "evaluation": result["evaluation"]["relative_delta"],
            "development_gate": result["development_gate"],
        }
    elif args.command == "freeze-shadow":
        result = freeze_candidate(
            protocol_path=Path(args.protocol),
            refinement_report_path=Path(args.refinement_report),
            clap_store=Path(args.clap_store),
            musicfm_train_store=Path(args.musicfm_train_store),
            musicfm_development_store=Path(args.musicfm_development_store),
            model_path=Path(args.model),
            metadata_path=Path(args.metadata),
            development_report_path=Path(args.development_report),
            shadow_extraction_plan_path=Path(args.shadow_extraction_plan),
            output=Path(args.output),
        )
        summary = {
            "payload_sha256": result["payload_sha256"],
            "shadow_labels_accessed": False,
        }
    else:
        result = audit_frozen_shadow(
            metadata_root=Path(args.metadata_root),
            protocol_path=Path(args.protocol),
            clap_store=Path(args.clap_store),
            musicfm_shadow_store=Path(args.musicfm_shadow_store),
            model_path=Path(args.model),
            metadata_path=Path(args.metadata),
            development_report_path=Path(args.development_report),
            shadow_extraction_plan_path=Path(args.shadow_extraction_plan),
            freeze_path=Path(args.freeze),
            output=Path(args.output),
            audit_state_path=Path(args.audit_state),
        )
        summary = {
            "evaluation": result["evaluation"]["relative_delta"],
            "shadow_gate": result["shadow_gate"],
        }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
