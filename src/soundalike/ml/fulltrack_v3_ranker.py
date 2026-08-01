"""Development-only supervised CLAP/MusicFM reranker canary.

This module never opens official test labels. It uses the five already-consumed
validation canaries, removes all held-fold artists from each training split,
and reports leave-one-fold-out evidence only.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from .fulltrack_eval import (
    METRICS,
    OFFICIAL_FOLDS,
    _BudgetCache,
    _method_ranking,
    _paired_bootstrap_delta,
    _query_metrics,
    _tag_jaccard_relevance,
    batch_fixed_budget_maxsim,
)
from .fulltrack_extract import normalize_rows
from .fulltrack_store import (
    FullTrackStoreReader,
    StoreBinding,
    TrackArtifacts,
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
    QUERY_LIMIT,
    SELECTION_SEED,
    SOURCE_FINGERPRINT,
    TRACKS_PER_FOLD,
    _open_bound_store,
)
from .jamendo_fulltrack import (
    EVIDENCE_SCOPE,
    JamendoContext,
    JamendoTrack,
    load_jamendo_context,
)


FEATURE_SCHEMA_VERSION = 1
FEATURE_KIND = "musicfm_dual_encoder_validation_features"
REPORT_SCHEMA_VERSION = 1
REPORT_KIND = "musicfm_dual_encoder_supervised_lofo_canary"
FEATURE_NAMES = (
    "clap_hybrid",
    "clap_global",
    "clap_uniform",
    "clap_section",
    "musicfm_global",
    "musicfm_uniform",
    "musicfm_section",
    "musicfm_hybrid",
)
EXPECTED_VALIDATION_SELECTION_SHA256 = {
    0: "c0d2fbef7448f94649a2e01061210a22baf4ea72ed7e35e716214e1f1c90cee3",
    1: "4db5c2fb8da603424b1ddd6403fc60a5ba17ea7e09df8f321b5a01683bc4cfcc",
    2: "5054dd3ecbeb1aa464049ac14673d2610f242e5897c6e05862318937aeaba727",
    3: "901a3d5a556ce06a9db8be20f97ef493778c5f93cc6df97436dd459778dfaf59",
    4: "0cd74219325442394ff0128f33066c51913bdc201f5bbe9c7b670491d2f813c9",
}
VALIDATION_STORE_MANIFEST_FILE_SHA256 = {
    0: "abb269cba1935e1ad132f37a72da63b69ada751c09929814e3394d92a23d7a04",
    1: "12c1611137758f1bbe69fd1b3a9b951f898d7e5b52d23a8eaa3f28535ee59488",
    2: "8ba4e1d11d4937c1658f83efa0e58ba357a535d478da40be683a5e498a4aa510",
    3: "3e82e6571a21b352ad44d966af216bf15c3a29bb5c7d3f29632bd5aff87f0b07",
    4: "1da30eecfc63e4c25e0f18c553e243d4a95c5abb92da840de23d08c3b3b5a368",
}
MUSICFM_MODEL_SHA256 = (
    "68392eee13d34c2941b3761934abb6b1e67b2e9df498695bda2ea5c1087d4b96"
)
MUSICFM_MODEL_ID = "musicfm_fma_b83ebed_layer7"
HARD_NEGATIVES = 8
RANDOM_NEGATIVES = 4
MAX_POSITIVES_PER_QUERY = 8
TRAINING_EPOCHS = 800
LEARNING_RATE = 0.03
PAIRWISE_MARGIN = 0.10
MARGIN_MSE_WEIGHT = 0.25


class V3RankerError(RuntimeError):
    """Invalid, leaky, non-reproducible, or non-finite ranker experiment."""


@dataclass(frozen=True)
class FoldArrays:
    track_ids: np.ndarray
    artist_ids: np.ndarray
    query_positions: np.ndarray
    global_orders: np.ndarray
    global_lengths: np.ndarray
    pools: np.ndarray
    features: np.ndarray
    relevance: np.ndarray
    shared_tags: np.ndarray

    def validate(self) -> None:
        if self.track_ids.shape != (TRACKS_PER_FOLD,):
            raise V3RankerError("track ID shape drift")
        if self.artist_ids.shape != self.track_ids.shape:
            raise V3RankerError("artist ID shape drift")
        if self.query_positions.shape != (QUERY_LIMIT,):
            raise V3RankerError("query position shape drift")
        if self.global_orders.shape != (QUERY_LIMIT, TRACKS_PER_FOLD - 1):
            raise V3RankerError("global order shape drift")
        if self.global_lengths.shape != (QUERY_LIMIT,):
            raise V3RankerError("global length shape drift")
        if self.pools.shape != (QUERY_LIMIT, CANDIDATE_POOL):
            raise V3RankerError("candidate-pool shape drift")
        if self.features.shape != (
            QUERY_LIMIT,
            CANDIDATE_POOL,
            len(FEATURE_NAMES),
        ):
            raise V3RankerError("feature shape drift")
        if self.relevance.shape != (QUERY_LIMIT, TRACKS_PER_FOLD):
            raise V3RankerError("relevance shape drift")
        if self.shared_tags.shape != (QUERY_LIMIT, CANDIDATE_POOL):
            raise V3RankerError("shared-tag shape drift")
        for label, values in (
            ("features", self.features),
            ("relevance", self.relevance),
        ):
            if not np.all(np.isfinite(values)):
                raise V3RankerError(f"{label} contains non-finite values")
        if np.any(self.global_lengths <= 0) or np.any(
            self.global_lengths > TRACKS_PER_FOLD - 1
        ):
            raise V3RankerError("global order length drift")
        for query_index, length in enumerate(self.global_lengths):
            order = self.global_orders[query_index, : int(length)]
            if np.any(order < 0) or len(np.unique(order)) != len(order):
                raise V3RankerError("global order contains invalid positions")
            if not np.array_equal(order[:CANDIDATE_POOL], self.pools[query_index]):
                raise V3RankerError("candidate pool is not the global prefix")


class _MusicFMUnion:
    def __init__(self, readers: Sequence[FullTrackStoreReader]) -> None:
        if not readers:
            raise V3RankerError("MusicFM reader union is empty")
        self._readers = tuple(readers)
        self._owners: Dict[int, FullTrackStoreReader] = {}
        for reader in readers:
            if (
                reader.binding.source_fingerprint != SOURCE_FINGERPRINT
                or reader.binding.model_sha256 != MUSICFM_MODEL_SHA256
                or reader.binding.model_id != MUSICFM_MODEL_ID
                or reader.binding.embedding_dim != 1024
                or reader.binding.repetition_sections < MAXSIM_BUDGET
                or reader.binding.salient_sections < MAXSIM_BUDGET
            ):
                raise V3RankerError("MusicFM validation store binding drift")
            for track_id in reader.track_ids:
                self._owners.setdefault(int(track_id), reader)
        first = readers[0].binding
        self.binding = StoreBinding(
            source_fingerprint=SOURCE_FINGERPRINT,
            config_sha256="0" * 64,
            model_sha256=MUSICFM_MODEL_SHA256,
            model_id=MUSICFM_MODEL_ID,
            embedding_dim=1024,
            track_count=len(self._owners),
            shard_tracks=first.shard_tracks,
            repetition_sections=min(
                reader.binding.repetition_sections for reader in readers
            ),
            salient_sections=min(
                reader.binding.salient_sections for reader in readers
            ),
            track_plan_sha256="0" * 64,
        )

    @property
    def track_ids(self) -> Tuple[int, ...]:
        return tuple(sorted(self._owners))

    def read_track(self, track_id: int) -> TrackArtifacts:
        try:
            reader = self._owners[int(track_id)]
        except KeyError as exc:
            raise V3RankerError(f"MusicFM track {track_id} is missing") from exc
        return reader.read_track(int(track_id))


def selected_validation_tracks(
    context: JamendoContext, fold_index: int
) -> Tuple[JamendoTrack, ...]:
    fold = next(
        (item for item in context.folds if item.index == fold_index),
        None,
    )
    if fold is None:
        raise V3RankerError(f"official fold {fold_index} is missing")
    eligible = [
        track
        for track in context.tracks
        if fold.track_parts.get(track.track_id) == "validation"
    ]
    selected = tuple(
        sorted(
            eligible,
            key=lambda track: stable_json_sha256(
                {"seed": SELECTION_SEED, "track_id": track.track_id}
            ),
        )[:TRACKS_PER_FOLD]
    )
    selection_hash = stable_json_sha256(
        tuple(track.track_id for track in selected)
    )
    if selection_hash != EXPECTED_VALIDATION_SELECTION_SHA256[fold_index]:
        raise V3RankerError(f"fold {fold_index} validation selection drift")
    return selected


def _zscore_columns(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not len(array) or not np.all(np.isfinite(array)):
        raise V3RankerError("features must be a finite matrix")
    standard_deviation = np.std(array, axis=0)
    divisor = np.where(standard_deviation > 1e-8, standard_deviation, 1.0)
    return (array - np.mean(array, axis=0)) / divisor


def _score_channels(
    query_position: int,
    pool: np.ndarray,
    globals_: np.ndarray,
    budget: _BudgetCache,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    global_scores = globals_[pool] @ globals_[query_position]
    uniform = batch_fixed_budget_maxsim(
        budget.uniform[query_position],
        budget.uniform[pool].astype(np.float32),
    )
    repeated = batch_fixed_budget_maxsim(
        budget.repeated[query_position],
        budget.repeated[pool].astype(np.float32),
    )
    salient = batch_fixed_budget_maxsim(
        budget.salient[query_position],
        budget.salient[pool].astype(np.float32),
    )
    section = 0.5 * (repeated + salient)
    hybrid = 0.50 * global_scores + 0.25 * uniform + 0.25 * section
    return global_scores, uniform, section, hybrid


def _extract_fold_arrays(
    context: JamendoContext,
    fold_index: int,
    selected: Sequence[JamendoTrack],
    clap_reader: FullTrackStoreReader,
    music_reader: _MusicFMUnion,
) -> FoldArrays:
    fold = next(item for item in context.folds if item.index == fold_index)
    track_ids = np.asarray([track.track_id for track in selected], dtype=np.int64)
    artist_ids = np.asarray([track.artist_id for track in selected], dtype=np.int64)
    clap_budget = _BudgetCache(
        clap_reader,
        track_ids,
        budget=MAXSIM_BUDGET,
        max_bytes=MAX_FEATURE_CACHE_BYTES,
    )
    music_budget = _BudgetCache(
        music_reader,  # type: ignore[arg-type]
        track_ids,
        budget=MAXSIM_BUDGET,
        max_bytes=MAX_FEATURE_CACHE_BYTES,
    )
    clap_globals = normalize_rows(
        np.stack(
            [
                clap_reader.read_track(int(track_id)).global_embedding
                for track_id in track_ids
            ]
        )
    )
    music_globals = normalize_rows(
        np.stack(
            [
                music_reader.read_track(int(track_id)).global_embedding
                for track_id in track_ids
            ]
        )
    )
    query_positions = np.arange(QUERY_LIMIT, dtype=np.int64)
    global_orders = np.full(
        (QUERY_LIMIT, TRACKS_PER_FOLD - 1), -1, dtype=np.int64
    )
    global_lengths = np.zeros(QUERY_LIMIT, dtype=np.int64)
    pools = np.empty((QUERY_LIMIT, CANDIDATE_POOL), dtype=np.int64)
    features = np.empty(
        (QUERY_LIMIT, CANDIDATE_POOL, len(FEATURE_NAMES)), dtype=np.float32
    )
    relevance = np.zeros((QUERY_LIMIT, TRACKS_PER_FOLD), dtype=np.float32)
    shared_tags = np.zeros((QUERY_LIMIT, CANDIDATE_POOL), dtype=np.int16)
    for query_index, query_position in enumerate(query_positions):
        query = selected[int(query_position)]
        eligible = np.asarray(
            [
                position
                for position, candidate in enumerate(selected)
                if candidate.track_id != query.track_id
                and candidate.artist_id != query.artist_id
            ],
            dtype=np.int64,
        )
        global_scores = clap_globals[eligible] @ clap_globals[query_position]
        global_order = eligible[np.lexsort((eligible, -global_scores))]
        global_orders[query_index, : len(global_order)] = global_order
        global_lengths[query_index] = len(global_order)
        pool = global_order[:CANDIDATE_POOL]
        pools[query_index] = pool
        clap = _score_channels(
            int(query_position), pool, clap_globals, clap_budget
        )
        music = _score_channels(
            int(query_position), pool, music_globals, music_budget
        )
        features[query_index] = np.stack(
            (
                clap[3],
                clap[0],
                clap[1],
                clap[2],
                music[0],
                music[1],
                music[2],
                music[3],
            ),
            axis=1,
        ).astype(np.float32)
        query_tags = set(fold.track_tags[query.track_id])
        for candidate_position, candidate in enumerate(selected):
            if (
                candidate.track_id == query.track_id
                or candidate.artist_id == query.artist_id
            ):
                continue
            relevance[query_index, candidate_position] = _tag_jaccard_relevance(
                tuple(query_tags),
                fold.track_tags[candidate.track_id],
                min_shared_tags=2,
                min_tag_jaccard=0.25,
            )
        for pool_index, candidate_position in enumerate(pool):
            candidate = selected[int(candidate_position)]
            shared_tags[query_index, pool_index] = len(
                query_tags.intersection(fold.track_tags[candidate.track_id])
            )
    arrays = FoldArrays(
        track_ids=track_ids,
        artist_ids=artist_ids,
        query_positions=query_positions,
        global_orders=global_orders,
        global_lengths=global_lengths,
        pools=pools,
        features=features,
        relevance=relevance,
        shared_tags=shared_tags,
    )
    arrays.validate()
    return arrays


def _write_npz_exclusive(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path = Path(path).absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise V3RankerError("feature archive already exists; refusing overwrite")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_exclusive(path: Path, document: Mapping[str, object]) -> None:
    path = Path(path).absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def extract_validation_features(
    *,
    metadata_root: Path,
    audio_root: Path,
    state_root: Path,
    clap_store: Path,
    validation_store_root: Path,
    output_npz: Path,
    output_metadata: Path,
) -> Mapping[str, object]:
    if Path(output_npz).exists() or Path(output_metadata).exists():
        raise V3RankerError("feature outputs already exist; refusing overwrite")
    context = load_jamendo_context(
        Path(metadata_root),
        Path(audio_root),
        Path(state_root),
        production=True,
    )
    if context.source_fingerprint != SOURCE_FINGERPRINT:
        raise V3RankerError("Jamendo source fingerprint drift")
    clap_reader = _open_bound_store(
        clap_store,
        expected_manifest_file_sha256=CLAP_MANIFEST_FILE_SHA256,
        expected_binding=EXPECTED_CLAP_BINDING,
    )
    music_readers = []
    try:
        for fold_index in OFFICIAL_FOLDS:
            root = (
                Path(validation_store_root)
                / f"musicfm-fma-canary-f{fold_index}-val-512"
            )
            manifest_path = root / "store.sealed.json"
            if sha256_path(manifest_path) != VALIDATION_STORE_MANIFEST_FILE_SHA256[
                fold_index
            ]:
                raise V3RankerError(
                    f"fold {fold_index} validation manifest file SHA-256 drift"
                )
            music_readers.append(
                FullTrackStoreReader(
                    root,
                    expected_source_fingerprint=SOURCE_FINGERPRINT,
                    expected_model_sha256=MUSICFM_MODEL_SHA256,
                )
            )
        music_union = _MusicFMUnion(music_readers)
        selections = {
            fold_index: selected_validation_tracks(context, fold_index)
            for fold_index in OFFICIAL_FOLDS
        }
        for fold_index, selected in selections.items():
            if {track.track_id for track in selected} != set(
                music_readers[fold_index].track_ids
            ):
                raise V3RankerError(
                    f"fold {fold_index} MusicFM store selection drift"
                )
        archive: Dict[str, np.ndarray] = {}
        for fold_index in OFFICIAL_FOLDS:
            arrays = _extract_fold_arrays(
                context,
                fold_index,
                selections[fold_index],
                clap_reader,
                music_union,
            )
            for field in FoldArrays.__dataclass_fields__:
                archive[f"f{fold_index}_{field}"] = getattr(arrays, field)
        _write_npz_exclusive(output_npz, archive)
        metadata: Dict[str, object] = {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "artifact_kind": FEATURE_KIND,
            "evidence_scope": EVIDENCE_SCOPE,
            "part": "validation",
            "held_out_test_accessed": False,
            "source_fingerprint": SOURCE_FINGERPRINT,
            "feature_names": list(FEATURE_NAMES),
            "selection_seed": SELECTION_SEED,
            "selection_sha256": dict(EXPECTED_VALIDATION_SELECTION_SHA256),
            "tracks_per_fold": TRACKS_PER_FOLD,
            "query_limit": QUERY_LIMIT,
            "candidate_pool": CANDIDATE_POOL,
            "maxsim_budget": MAXSIM_BUDGET,
            "clap_manifest_file_sha256": CLAP_MANIFEST_FILE_SHA256,
            "musicfm_manifest_file_sha256": dict(
                VALIDATION_STORE_MANIFEST_FILE_SHA256
            ),
            "npz_sha256": sha256_path(Path(output_npz)),
        }
        metadata["payload_sha256"] = stable_json_sha256(metadata)
        _write_json_exclusive(output_metadata, metadata)
        return metadata
    finally:
        clap_reader.close()
        for reader in music_readers:
            reader.close()


def _load_feature_cache(
    npz_path: Path, metadata_path: Path
) -> Tuple[Mapping[str, object], Dict[int, FoldArrays]]:
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    declared = metadata.pop("payload_sha256", None)
    if declared != stable_json_sha256(metadata):
        raise V3RankerError("feature metadata payload checksum mismatch")
    metadata["payload_sha256"] = declared
    if (
        metadata.get("schema_version") != FEATURE_SCHEMA_VERSION
        or metadata.get("artifact_kind") != FEATURE_KIND
        or metadata.get("evidence_scope") != EVIDENCE_SCOPE
        or metadata.get("part") != "validation"
        or metadata.get("held_out_test_accessed") is not False
        or metadata.get("feature_names") != list(FEATURE_NAMES)
        or metadata.get("selection_sha256")
        != {str(key): value for key, value in EXPECTED_VALIDATION_SELECTION_SHA256.items()}
        or metadata.get("npz_sha256") != sha256_path(Path(npz_path))
    ):
        raise V3RankerError("feature metadata binding drift")
    with np.load(Path(npz_path), allow_pickle=False) as archive:
        expected = {
            f"f{fold_index}_{field}"
            for fold_index in OFFICIAL_FOLDS
            for field in FoldArrays.__dataclass_fields__
        }
        if set(archive.files) != expected:
            raise V3RankerError("feature archive member drift")
        folds = {
            fold_index: FoldArrays(
                **{
                    field: np.asarray(archive[f"f{fold_index}_{field}"]).copy()
                    for field in FoldArrays.__dataclass_fields__
                }
            )
            for fold_index in OFFICIAL_FOLDS
        }
    for arrays in folds.values():
        arrays.validate()
    return metadata, folds


def prepare_training_differences(
    folds: Mapping[int, FoldArrays],
    *,
    held_fold: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, Mapping[str, object]]:
    held_artists = set(int(value) for value in folds[held_fold].artist_ids)
    owner: Dict[int, int] = {}
    for fold_index in OFFICIAL_FOLDS:
        if fold_index == held_fold:
            continue
        arrays = folds[fold_index]
        for query_position in arrays.query_positions:
            track_id = int(arrays.track_ids[int(query_position)])
            owner.setdefault(track_id, fold_index)

    differences = []
    targets = []
    used_queries = 0
    excluded_held_query_artists = 0
    excluded_held_candidates = 0
    false_negative_candidates = 0
    pair_keys = set()
    for fold_index in OFFICIAL_FOLDS:
        if fold_index == held_fold:
            continue
        arrays = folds[fold_index]
        for query_index, query_position in enumerate(arrays.query_positions):
            query_position = int(query_position)
            query_track_id = int(arrays.track_ids[query_position])
            query_artist_id = int(arrays.artist_ids[query_position])
            if owner[query_track_id] != fold_index:
                continue
            if query_artist_id in held_artists:
                excluded_held_query_artists += 1
                continue
            pool = arrays.pools[query_index]
            candidate_artists = arrays.artist_ids[pool]
            allowed = np.asarray(
                [int(artist) not in held_artists for artist in candidate_artists],
                dtype=np.bool_,
            )
            excluded_held_candidates += int(np.count_nonzero(~allowed))
            if np.count_nonzero(allowed) < 2:
                continue
            allowed_pool = pool[allowed]
            values = arrays.features[query_index, allowed]
            labels = arrays.relevance[query_index, allowed_pool]
            shared = arrays.shared_tags[query_index, allowed]
            normalized = _zscore_columns(values)
            positive_indices = np.flatnonzero(labels > 0.0)
            negative_indices = np.flatnonzero((labels == 0.0) & (shared == 0))
            false_negative_candidates += int(
                np.count_nonzero((labels == 0.0) & (shared > 0))
            )
            if not len(positive_indices) or not len(negative_indices):
                continue
            positive_indices = positive_indices[
                np.lexsort(
                    (
                        allowed_pool[positive_indices],
                        -labels[positive_indices],
                    )
                )
            ][:MAX_POSITIVES_PER_QUERY]
            negative_indices = negative_indices[
                np.lexsort(
                    (
                        allowed_pool[negative_indices],
                        -values[negative_indices, 0],
                    )
                )
            ]
            hard = negative_indices[: min(HARD_NEGATIVES, len(negative_indices))]
            remaining = negative_indices[len(hard) :]
            rng = np.random.default_rng(
                int(
                    stable_json_sha256(
                        {
                            "seed": seed,
                            "held_fold": held_fold,
                            "query_track_id": query_track_id,
                        }
                    )[:16],
                    16,
                )
            )
            random_values = (
                rng.permutation(remaining)[:RANDOM_NEGATIVES]
                if len(remaining)
                else np.empty(0, dtype=np.int64)
            )
            negatives = np.concatenate((hard, random_values))
            for positive_index in positive_indices:
                positive_track_id = int(allowed_pool[positive_index])
                for negative_index in negatives:
                    negative_track_id = int(allowed_pool[int(negative_index)])
                    key = (
                        query_track_id,
                        positive_track_id,
                        negative_track_id,
                    )
                    if key in pair_keys:
                        continue
                    pair_keys.add(key)
                    differences.append(
                        normalized[positive_index] - normalized[int(negative_index)]
                    )
                    targets.append(float(labels[positive_index]))
            used_queries += 1
    if not differences:
        raise V3RankerError(f"held fold {held_fold} produced no training pairs")
    difference_array = np.asarray(differences, dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.float64)
    if not np.all(np.isfinite(difference_array)) or not np.all(
        np.isfinite(target_array)
    ):
        raise V3RankerError("training pairs contain non-finite values")
    stats = {
        "held_fold": held_fold,
        "held_artist_count": len(held_artists),
        "used_queries": used_queries,
        "pair_count": len(difference_array),
        "excluded_held_query_artists": excluded_held_query_artists,
        "excluded_held_candidates": excluded_held_candidates,
        "excluded_possible_false_negatives": false_negative_candidates,
        "unique_pair_count": len(pair_keys),
        "same_held_artist_training_count": 0,
    }
    return difference_array, target_array, stats


def fit_nonnegative_ranker(
    differences: np.ndarray,
    targets: np.ndarray,
    *,
    seed: int,
    auxiliary_weight: float = 0.10,
) -> Tuple[np.ndarray, Mapping[str, object]]:
    import torch

    difference_array = np.asarray(differences, dtype=np.float64)
    target_array = np.asarray(targets, dtype=np.float64)
    if (
        difference_array.ndim != 2
        or difference_array.shape[1] != len(FEATURE_NAMES)
        or target_array.shape != (len(difference_array),)
        or not len(difference_array)
    ):
        raise V3RankerError("ranker training arrays have invalid shape")
    if (
        not np.isfinite(auxiliary_weight)
        or auxiliary_weight <= 0.0
        or auxiliary_weight > 0.25
    ):
        raise V3RankerError("auxiliary weight must be in (0, 0.25]")
    torch.manual_seed(int(seed))
    dtype = torch.float64
    raw = torch.nn.Parameter(torch.zeros(len(FEATURE_NAMES) - 1, dtype=dtype))
    x = torch.tensor(difference_array, dtype=dtype)
    y = torch.tensor(target_array, dtype=dtype)
    optimizer = torch.optim.Adam([raw], lr=LEARNING_RATE)
    initial_loss = None
    final_loss = None
    for _ in range(TRAINING_EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        auxiliary = torch.softmax(raw, dim=0)
        weights = torch.cat(
            (
                torch.tensor([1.0 - auxiliary_weight], dtype=dtype),
                auxiliary_weight * auxiliary,
            )
        )
        predicted_margin = x @ weights
        pairwise = torch.nn.functional.softplus(
            PAIRWISE_MARGIN - predicted_margin
        ).mean()
        margin_mse = torch.mean((predicted_margin - y) ** 2)
        loss = pairwise + MARGIN_MSE_WEIGHT * margin_mse
        if not bool(torch.isfinite(loss).item()):
            raise V3RankerError("ranker training produced a non-finite loss")
        if initial_loss is None:
            initial_loss = float(loss.detach().item())
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().item())
    with torch.no_grad():
        auxiliary = torch.softmax(raw, dim=0)
        weights = torch.cat(
            (
                torch.tensor([1.0 - auxiliary_weight], dtype=dtype),
                auxiliary_weight * auxiliary,
            )
        ).cpu().numpy()
    if (
        not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
        or not np.isclose(np.sum(weights), 1.0, atol=1e-10)
    ):
        raise V3RankerError("ranker weights are invalid")
    return weights.astype(np.float64), {
        "seed": seed,
        "epochs": TRAINING_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "pairwise_margin": PAIRWISE_MARGIN,
        "margin_mse_weight": MARGIN_MSE_WEIGHT,
        "auxiliary_weight": auxiliary_weight,
        "fixed_clap_hybrid_weight": 1.0 - auxiliary_weight,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
    }


def _evaluate_fold(
    arrays: FoldArrays, weights: np.ndarray
) -> Tuple[Mapping[str, object], Sequence[Mapping[str, object]]]:
    records = []
    skipped = 0
    for query_index, query_position in enumerate(arrays.query_positions):
        pool = arrays.pools[query_index]
        global_order = arrays.global_orders[
            query_index, : int(arrays.global_lengths[query_index])
        ]
        relevant = {
            int(arrays.track_ids[position]): float(grade)
            for position, grade in enumerate(arrays.relevance[query_index])
            if grade > 0.0
        }
        if not relevant:
            skipped += 1
            continue
        baseline_scores = arrays.features[query_index, :, 0].astype(np.float64)
        candidate_scores = _zscore_columns(
            arrays.features[query_index]
        ) @ np.asarray(weights, dtype=np.float64)
        baseline_order = _method_ranking(
            baseline_scores, pool, global_order
        )
        candidate_order = _method_ranking(
            candidate_scores, pool, global_order
        )

        def metrics(order: np.ndarray) -> Mapping[str, float]:
            ranked_ids = [int(arrays.track_ids[position]) for position in order]
            return vars(
                _query_metrics(
                    ranked_ids,
                    relevant,
                    recall_cutoff=10,
                    ndcg_cutoff=10,
                )
            )

        records.append(
            {
                "track_id": int(arrays.track_ids[int(query_position)]),
                "metrics": {
                    "clap_hybrid": metrics(baseline_order),
                    "trained_reranker": metrics(candidate_order),
                },
            }
        )
    if not records:
        raise V3RankerError("held fold has no evaluable queries")
    methods = {}
    for method in ("clap_hybrid", "trained_reranker"):
        methods[method] = {
            metric: float(
                np.mean(
                    [
                        record["metrics"][method][metric]
                        for record in records
                    ]
                )
            )
            for metric in METRICS
        }
    deltas = {
        metric: methods["trained_reranker"][metric]
        - methods["clap_hybrid"][metric]
        for metric in METRICS
    }
    relative = {
        metric: (
            methods["trained_reranker"][metric]
            / methods["clap_hybrid"][metric]
            - 1.0
        )
        for metric in METRICS
    }
    return {
        "queries": len(records),
        "skipped_no_relevant": skipped,
        "methods": methods,
        "absolute_delta": deltas,
        "relative_delta": relative,
    }, records


def train_lofo_canary(
    *,
    features_npz: Path,
    features_metadata: Path,
    output_path: Path,
    seed: int = 20260801,
    auxiliary_weight: float = 0.10,
) -> Mapping[str, object]:
    if Path(output_path).exists():
        raise V3RankerError("ranker report already exists; refusing overwrite")
    metadata, folds = _load_feature_cache(features_npz, features_metadata)
    fold_results = []
    all_records = []
    for held_fold in OFFICIAL_FOLDS:
        differences, targets, mining_stats = prepare_training_differences(
            folds, held_fold=held_fold, seed=seed
        )
        weights, training = fit_nonnegative_ranker(
            differences,
            targets,
            seed=seed + held_fold,
            auxiliary_weight=auxiliary_weight,
        )
        evaluation, records = _evaluate_fold(folds[held_fold], weights)
        fold_results.append(
            {
                "held_fold": held_fold,
                "weights": {
                    name: float(weight)
                    for name, weight in zip(FEATURE_NAMES, weights)
                },
                "mining": mining_stats,
                "training": training,
                "evaluation": evaluation,
            }
        )
        all_records.extend(
            {"fold": held_fold, **record} for record in records
        )
    pooled = {}
    for method in ("clap_hybrid", "trained_reranker"):
        pooled[method] = {
            metric: float(
                np.mean(
                    [
                        record["metrics"][method][metric]
                        for record in all_records
                    ]
                )
            )
            for metric in METRICS
        }
    pooled["absolute_delta"] = {
        metric: pooled["trained_reranker"][metric]
        - pooled["clap_hybrid"][metric]
        for metric in METRICS
    }
    pooled["relative_delta"] = {
        metric: pooled["trained_reranker"][metric]
        / pooled["clap_hybrid"][metric]
        - 1.0
        for metric in METRICS
    }
    pooled["paired_delta"] = {
        metric: _paired_bootstrap_delta(
            [
                record["metrics"]["clap_hybrid"][metric]
                for record in all_records
            ],
            [
                record["metrics"]["trained_reranker"][metric]
                for record in all_records
            ],
            iterations=BOOTSTRAP_ITERATIONS,
            seed=BOOTSTRAP_SEED,
        )
        for metric in METRICS
    }
    positive_folds = {
        metric: sum(
            result["evaluation"]["absolute_delta"][metric] > 0.0
            for result in fold_results
        )
        for metric in METRICS
    }
    report: Dict[str, object] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_kind": REPORT_KIND,
        "evidence_scope": EVIDENCE_SCOPE,
        "evidence_status": "development_only_consumed_validation_folds",
        "held_out_test_accessed": False,
        "promotion_allowed": False,
        "feature_cache_payload_sha256": metadata["payload_sha256"],
        "feature_cache_npz_sha256": metadata["npz_sha256"],
        "feature_names": list(FEATURE_NAMES),
        "protocol": {
            "held_artist_exclusion": True,
            "deduplicate_training_queries": True,
            "false_negative_rule": "exclude zero-grade candidates sharing any tag",
            "hard_negatives": HARD_NEGATIVES,
            "random_negatives": RANDOM_NEGATIVES,
            "max_positives_per_query": MAX_POSITIVES_PER_QUERY,
            "seed": seed,
            "auxiliary_weight": auxiliary_weight,
        },
        "fold_results": fold_results,
        "pooled": pooled,
        "positive_folds": positive_folds,
        "scale_candidate": (
            pooled["relative_delta"]["recall_at_k"] >= 0.05
            and pooled["relative_delta"]["mrr"] >= -0.01
            and pooled["relative_delta"]["graded_ndcg_at_k"] >= -0.01
            and positive_folds["recall_at_k"] >= 4
        ),
    }
    report["payload_sha256"] = stable_json_sha256(report)
    _write_json_exclusive(output_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser("extract-validation-features")
    extract.add_argument("--metadata-root", required=True)
    extract.add_argument("--audio-root", required=True)
    extract.add_argument("--state-root", required=True)
    extract.add_argument("--clap-store", required=True)
    extract.add_argument("--validation-store-root", required=True)
    extract.add_argument("--output-npz", required=True)
    extract.add_argument("--output-metadata", required=True)
    train = subparsers.add_parser("train-lofo")
    train.add_argument("--features-npz", required=True)
    train.add_argument("--features-metadata", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--seed", type=int, default=20260801)
    train.add_argument("--auxiliary-weight", type=float, default=0.10)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "extract-validation-features":
            result = extract_validation_features(
                metadata_root=Path(args.metadata_root),
                audio_root=Path(args.audio_root),
                state_root=Path(args.state_root),
                clap_store=Path(args.clap_store),
                validation_store_root=Path(args.validation_store_root),
                output_npz=Path(args.output_npz),
                output_metadata=Path(args.output_metadata),
            )
        else:
            result = train_lofo_canary(
                features_npz=Path(args.features_npz),
                features_metadata=Path(args.features_metadata),
                output_path=Path(args.output),
                seed=int(args.seed),
                auxiliary_weight=float(args.auxiliary_weight),
            )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except (OSError, ValueError, V3RankerError) as exc:
        raise SystemExit(f"V3 ranker blocked: {exc}") from exc


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
