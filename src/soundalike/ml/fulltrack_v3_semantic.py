"""Train and evaluate the frozen artist-disjoint V3 semantic tag head."""
from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse.linalg import lsqr

from .fulltrack_eval import (
    METRICS,
    _BudgetCache,
    _method_ranking,
    _paired_bootstrap_delta,
    _query_metrics,
    _tag_jaccard_relevance,
)
from .fulltrack_extract import normalize_rows
from .fulltrack_store import FullTrackStoreReader, sha256_path, stable_json_sha256
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
)
from .fulltrack_v3_protocol import (
    BASE_FOLD,
    BASE_PART,
    EXPECTED_SELECTION_SHA256,
    EXPECTED_SPLITS,
    PROTOCOL_KIND,
    load_protocol,
)
from .fulltrack_v3_ranker import (
    MUSICFM_MODEL_ID,
    MUSICFM_MODEL_SHA256,
    _score_channels,
    _write_json_exclusive,
    _write_npz_exclusive,
    _zscore_columns,
)
from .jamendo_fulltrack import EVIDENCE_SCOPE, _ID_PATTERNS, _TAG


MODEL_SCHEMA_VERSION = 1
MODEL_KIND = "v3_artist_disjoint_semantic_tag_head"
REPORT_SCHEMA_VERSION = 1
REPORT_KIND = "v3_semantic_tag_head_development_report"
SCALE_PROTOCOL_PAYLOAD_SHA256 = (
    "d697240384003ba1a7d9e00d281462b005a3e02abac49964cd3ba4e128292738"
)
SCALE_MUSICFM_CONFIG_SHA256 = (
    "c2b0c316a36226abd26b85912c136967ca7df5ebfcb59c5bff0b9f42ed169ea3"
)
SCALE_MUSICFM_TRACK_PLAN_SHA256 = (
    "43d93666e0c25708f7b7b1c25f3c504c80bc5f7b2edd553b284d7fda0de4a74b"
)
REPRESENTATIONS = ("clap", "musicfm", "dual")
RIDGE_VALUES = (1.0, 10.0, 100.0)
BLEND_VALUES = (0.05, 0.10, 0.20, 0.30)
DEVELOPMENT_FOLDS = 5
DEVELOPMENT_FOLD_SEED = 20260804
MIN_DEVELOPMENT_PRIMARY_RELATIVE_GAIN = 0.15
MAX_SAFETY_RELATIVE_REGRESSION = 0.01
MIN_POSITIVE_FOLDS = 4
MAX_FOLD_PRIMARY_RELATIVE_REGRESSION = 0.05
PRIMARY_METRIC = "recall_at_k"
INPUT_SCALE_EPSILON = 1e-6
PROFILE_NORM_EPSILON = 1e-12
LSQR_TARGET_WORKERS = 16
LABEL_HEADER = ("TRACK_ID", "ARTIST_ID", "ALBUM_ID", "PATH", "DURATION", "TAGS")


class V3SemanticError(RuntimeError):
    """Invalid, leaky, non-reproducible, or non-finite semantic-head run."""


@dataclass(frozen=True)
class SemanticHead:
    representation: str
    ridge: float
    vocabulary: Tuple[str, ...]
    input_mean: np.ndarray
    input_scale: np.ndarray
    coefficients: np.ndarray
    prior: np.ndarray
    idf: np.ndarray

    def validate(self) -> None:
        if self.representation not in REPRESENTATIONS:
            raise V3SemanticError("unknown semantic-head representation")
        if self.ridge not in RIDGE_VALUES:
            raise V3SemanticError("semantic-head ridge value drift")
        dimension = self.input_mean.shape
        tag_count = len(self.vocabulary)
        if (
            self.input_mean.ndim != 1
            or self.input_scale.shape != dimension
            or self.coefficients.shape != (dimension[0], tag_count)
            or self.prior.shape != (tag_count,)
            or self.idf.shape != (tag_count,)
            or not tag_count
            or len(set(self.vocabulary)) != tag_count
            or tuple(sorted(self.vocabulary)) != self.vocabulary
        ):
            raise V3SemanticError("semantic-head array shape drift")
        for name, values in (
            ("input_mean", self.input_mean),
            ("input_scale", self.input_scale),
            ("coefficients", self.coefficients),
            ("prior", self.prior),
            ("idf", self.idf),
        ):
            if not np.all(np.isfinite(values)):
                raise V3SemanticError(f"{name} contains non-finite values")
        if np.any(self.input_scale <= 0.0) or np.any(self.idf <= 0.0):
            raise V3SemanticError("semantic-head scale or IDF is non-positive")

    def predict(self, values: np.ndarray) -> np.ndarray:
        self.validate()
        inputs = _transform_with_statistics(
            values,
            self.input_mean,
            self.input_scale,
        )
        predictions = np.clip(inputs @ self.coefficients + self.prior, 0.0, 1.0)
        weighted = predictions * self.idf
        norms = np.linalg.norm(weighted, axis=1, keepdims=True)
        if np.any(norms <= PROFILE_NORM_EPSILON):
            raise V3SemanticError("semantic head produced an empty profile")
        return weighted / norms


@dataclass(frozen=True)
class DevelopmentData:
    track_ids: np.ndarray
    artist_ids: np.ndarray
    query_folds: np.ndarray
    global_orders: np.ndarray
    global_lengths: np.ndarray
    pools: np.ndarray
    baseline_scores: np.ndarray
    relevance: np.ndarray

    def validate(self) -> None:
        count = len(self.track_ids)
        if (
            count != EXPECTED_SPLITS["development"]["tracks"]
            or self.artist_ids.shape != (count,)
            or self.query_folds.shape != (count,)
            or self.global_orders.shape != (count, count - 1)
            or self.global_lengths.shape != (count,)
            or self.pools.shape != (count, CANDIDATE_POOL)
            or self.baseline_scores.shape != self.pools.shape
            or self.relevance.shape != (count, count)
        ):
            raise V3SemanticError("development array shape drift")
        if (
            len(np.unique(self.track_ids)) != count
            or np.any(self.query_folds < 0)
            or np.any(self.query_folds >= DEVELOPMENT_FOLDS)
            or not np.all(np.isfinite(self.baseline_scores))
            or not np.all(np.isfinite(self.relevance))
        ):
            raise V3SemanticError("development arrays contain invalid values")
        for query_index, length in enumerate(self.global_lengths):
            order = self.global_orders[query_index, : int(length)]
            if (
                length < CANDIDATE_POOL
                or np.any(order < 0)
                or len(np.unique(order)) != len(order)
                or not np.array_equal(order[:CANDIDATE_POOL], self.pools[query_index])
            ):
                raise V3SemanticError("development global order drift")


def _normalized_inputs(
    train: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(train, dtype=np.float64)
    if values.ndim != 2 or not len(values) or not np.all(np.isfinite(values)):
        raise V3SemanticError("training inputs must be a finite matrix")
    mean = np.mean(values, axis=0)
    standard_deviation = np.std(values, axis=0)
    scale = np.where(
        standard_deviation > INPUT_SCALE_EPSILON,
        standard_deviation,
        1.0,
    )
    return _transform_with_statistics(values, mean, scale), mean, scale


def _transform_with_statistics(
    values: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if (
        array.ndim != 2
        or array.shape[1:] != mean.shape
        or scale.shape != mean.shape
        or not np.all(np.isfinite(array))
        or not np.all(np.isfinite(mean))
        or not np.all(np.isfinite(scale))
        or np.any(scale <= 0.0)
    ):
        raise V3SemanticError("semantic-head inputs or statistics are invalid")
    centered = (array - mean) / scale
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    if np.any(norms <= PROFILE_NORM_EPSILON):
        raise V3SemanticError("semantic-head input has zero normalized norm")
    return centered / norms


def fit_semantic_head(
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    vocabulary: Sequence[str],
    *,
    representation: str,
    ridge: float,
) -> SemanticHead:
    if representation not in REPRESENTATIONS or ridge not in RIDGE_VALUES:
        raise V3SemanticError("semantic-head hyperparameter drift")
    inputs, mean, scale = _normalized_inputs(train_inputs)
    targets = np.asarray(train_targets, dtype=np.float64)
    vocabulary_tuple = tuple(vocabulary)
    if (
        targets.shape != (len(inputs), len(vocabulary_tuple))
        or not len(vocabulary_tuple)
        or not np.all(np.isfinite(targets))
        or np.any(targets < 0.0)
        or np.any(targets > 1.0)
        or np.any(np.sum(targets, axis=1) <= 0.0)
    ):
        raise V3SemanticError("semantic-head targets are invalid")
    prior = np.mean(targets, axis=0)
    centered_targets = targets - prior

    def solve_target(index: int) -> np.ndarray:
        return np.asarray(
            lsqr(
                inputs,
                centered_targets[:, index],
                damp=np.sqrt(ridge),
                atol=1e-6,
                btol=1e-6,
                iter_lim=2_000,
            )[0],
            dtype=np.float64,
        )

    with ThreadPoolExecutor(
        max_workers=min(LSQR_TARGET_WORKERS, len(vocabulary_tuple))
    ) as executor:
        coefficients = np.stack(
            tuple(executor.map(solve_target, range(len(vocabulary_tuple)))),
            axis=1,
        )
    idf = np.log(
        (len(targets) + 1.0) / (np.sum(targets, axis=0) + 1.0)
    ) + 1.0
    head = SemanticHead(
        representation=representation,
        ridge=float(ridge),
        vocabulary=vocabulary_tuple,
        input_mean=mean,
        input_scale=scale,
        coefficients=coefficients,
        prior=prior,
        idf=idf,
    )
    head.validate()
    return head


def _protocol_entries(
    protocol: Mapping[str, object],
    split: str,
) -> Tuple[Mapping[str, object], ...]:
    entries = protocol.get("tracks")
    if not isinstance(entries, list):
        raise V3SemanticError("protocol has no track plan")
    selected = tuple(
        entry
        for entry in entries
        if isinstance(entry, dict) and entry.get("split") == split
    )
    expected = EXPECTED_SPLITS[split]
    if (
        len(selected) != expected["tracks"]
        or len({int(entry["artist_id"]) for entry in selected})
        != expected["artists"]
    ):
        raise V3SemanticError(f"{split} protocol split drift")
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
        raise V3SemanticError("label split selection is invalid")
    allowed_entries = tuple(
        entry
        for split in selected_splits
        for entry in _protocol_entries(protocol, split)
    )
    expected_artists = {
        int(entry["track_id"]): int(entry["artist_id"]) for entry in allowed_entries
    }
    excluded_ids = {
        int(entry["track_id"])
        for split in EXPECTED_SPLITS
        if split not in selected_splits
        for entry in _protocol_entries(protocol, split)
    }
    path = (
        Path(metadata_root).absolute()
        / "data"
        / "splits"
        / f"split-{BASE_FOLD}"
        / f"autotagging-{BASE_PART}.tsv"
    )
    if path.is_symlink():
        raise V3SemanticError("label source may not be a symlink")
    path = path.resolve(strict=True)
    labels: Dict[int, Tuple[str, ...]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        if next(reader, None) != list(LABEL_HEADER):
            raise V3SemanticError("label source header drift")
        for line_number, row in enumerate(reader, 2):
            if len(row) < len(LABEL_HEADER):
                raise V3SemanticError(f"label source row {line_number} is short")
            match = _ID_PATTERNS["track"].fullmatch(row[0])
            if match is None:
                raise V3SemanticError(
                    f"label source row {line_number} has malformed track ID"
                )
            track_id = int(match.group(1))
            if track_id not in expected_artists:
                continue
            artist_match = _ID_PATTERNS["artist"].fullmatch(row[1])
            tags = tuple(sorted(row[5:]))
            if (
                artist_match is None
                or int(artist_match.group(1)) != expected_artists[track_id]
                or not tags
                or len(tags) != len(set(tags))
                or any(_TAG.fullmatch(tag) is None for tag in tags)
                or track_id in labels
            ):
                raise V3SemanticError(
                    f"label source row {line_number} differs from protocol"
                )
            labels[track_id] = tags
    if set(labels) != set(expected_artists):
        raise V3SemanticError("selected protocol labels are incomplete")
    if set(labels).intersection(excluded_ids):
        raise V3SemanticError("labels outside the selected splits were loaded")
    return labels


def load_train_development_tags(
    metadata_root: Path,
    protocol: Mapping[str, object],
) -> Mapping[int, Tuple[str, ...]]:
    return load_protocol_tags(
        metadata_root,
        protocol,
        ("train", "development"),
    )


def build_label_targets(
    train_entries: Sequence[Mapping[str, object]],
    labels: Mapping[int, Sequence[str]],
) -> Tuple[Tuple[str, ...], np.ndarray]:
    vocabulary = tuple(
        sorted(
            {
                tag
                for entry in train_entries
                for tag in labels[int(entry["track_id"])]
            }
        )
    )
    tag_positions = {tag: position for position, tag in enumerate(vocabulary)}
    targets = np.zeros((len(train_entries), len(vocabulary)), dtype=np.float64)
    for row, entry in enumerate(train_entries):
        for tag in labels[int(entry["track_id"])]:
            targets[row, tag_positions[tag]] = 1.0
    if np.any(np.sum(targets, axis=0) <= 0.0):
        raise V3SemanticError("training vocabulary contains an empty tag")
    return vocabulary, targets


def _representation_inputs(
    representation: str,
    clap: np.ndarray,
    musicfm: np.ndarray,
) -> np.ndarray:
    if clap.ndim != 2 or musicfm.ndim != 2 or len(clap) != len(musicfm):
        raise V3SemanticError("embedding matrices are misaligned")
    if representation == "clap":
        return clap
    if representation == "musicfm":
        return musicfm
    if representation == "dual":
        return np.concatenate((clap, musicfm), axis=1)
    raise V3SemanticError("unknown representation")


def _global_embeddings(
    reader: FullTrackStoreReader,
    track_ids: Sequence[int],
) -> np.ndarray:
    positions = {track_id: row for row, track_id in enumerate(reader.track_ids)}
    try:
        rows = [positions[int(track_id)] for track_id in track_ids]
    except KeyError as exc:
        raise V3SemanticError(f"store is missing track {exc.args[0]}") from exc
    values = np.asarray(reader.global_embeddings[rows], dtype=np.float64)
    if values.shape != (len(track_ids), reader.binding.embedding_dim):
        raise V3SemanticError("global embedding shape drift")
    return normalize_rows(values)


def _development_fold(artist_id: int) -> int:
    return (
        int(
            stable_json_sha256(
                {"seed": DEVELOPMENT_FOLD_SEED, "artist_id": int(artist_id)}
            )[:16],
            16,
        )
        % DEVELOPMENT_FOLDS
    )


def build_development_data(
    development_entries: Sequence[Mapping[str, object]],
    labels: Mapping[int, Sequence[str]],
    clap_reader: FullTrackStoreReader,
) -> DevelopmentData:
    row_positions = {
        track_id: position for position, track_id in enumerate(clap_reader.track_ids)
    }
    try:
        entries = tuple(
            sorted(
                development_entries,
                key=lambda entry: row_positions[int(entry["track_id"])],
            )
        )
    except KeyError as exc:
        raise V3SemanticError(f"CLAP store is missing track {exc.args[0]}") from exc
    track_ids = np.asarray(
        [int(entry["track_id"]) for entry in entries], dtype=np.int64
    )
    artist_ids = np.asarray(
        [int(entry["artist_id"]) for entry in entries], dtype=np.int64
    )
    query_folds = np.asarray(
        [_development_fold(int(artist)) for artist in artist_ids], dtype=np.int8
    )
    count = len(track_ids)
    budget = _BudgetCache(
        clap_reader,
        track_ids,
        budget=MAXSIM_BUDGET,
        max_bytes=MAX_FEATURE_CACHE_BYTES,
    )
    globals_ = _global_embeddings(clap_reader, track_ids)
    global_orders = np.full((count, count - 1), -1, dtype=np.int32)
    global_lengths = np.zeros(count, dtype=np.int32)
    pools = np.empty((count, CANDIDATE_POOL), dtype=np.int32)
    baseline_scores = np.empty((count, CANDIDATE_POOL), dtype=np.float32)
    relevance = np.zeros((count, count), dtype=np.float32)
    for query_position in range(count):
        eligible = np.flatnonzero(
            (track_ids != track_ids[query_position])
            & (artist_ids != artist_ids[query_position])
        )
        global_scores = globals_[eligible] @ globals_[query_position]
        order = eligible[
            np.lexsort((track_ids[eligible], -global_scores))
        ]
        if len(order) < CANDIDATE_POOL:
            raise V3SemanticError("development candidate universe is too small")
        global_orders[query_position, : len(order)] = order
        global_lengths[query_position] = len(order)
        pool = order[:CANDIDATE_POOL]
        pools[query_position] = pool
        baseline_scores[query_position] = _score_channels(
            query_position,
            pool,
            globals_,
            budget,
        )[3].astype(np.float32)
        query_tags = labels[int(track_ids[query_position])]
        for candidate_position in eligible:
            relevance[query_position, candidate_position] = _tag_jaccard_relevance(
                query_tags,
                labels[int(track_ids[candidate_position])],
                min_shared_tags=2,
                min_tag_jaccard=0.25,
            )
    data = DevelopmentData(
        track_ids=track_ids,
        artist_ids=artist_ids,
        query_folds=query_folds,
        global_orders=global_orders,
        global_lengths=global_lengths,
        pools=pools,
        baseline_scores=baseline_scores,
        relevance=relevance,
    )
    data.validate()
    return data


def _relative_delta(baseline: float, candidate: float) -> float:
    if baseline <= 0.0:
        raise V3SemanticError("baseline metric is not positive")
    return float(candidate / baseline - 1.0)


def evaluate_profiles(
    data: DevelopmentData,
    profiles: np.ndarray,
    *,
    blend: float,
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
) -> Mapping[str, object]:
    if (
        isinstance(bootstrap_iterations, bool)
        or not isinstance(bootstrap_iterations, int)
        or bootstrap_iterations < 0
    ):
        raise V3SemanticError("development profile evaluation inputs are invalid")
    data.validate()
    values = np.asarray(profiles, dtype=np.float64)
    if (
        blend not in BLEND_VALUES
        or values.ndim != 2
        or len(values) != len(data.track_ids)
        or not np.all(np.isfinite(values))
    ):
        raise V3SemanticError("development profile evaluation inputs are invalid")
    baseline_values: Dict[str, list[float]] = {metric: [] for metric in METRICS}
    candidate_values: Dict[str, list[float]] = {metric: [] for metric in METRICS}
    query_folds = []
    for query_position in range(len(data.track_ids)):
        relevant = {
            int(data.track_ids[position]): float(grade)
            for position, grade in enumerate(data.relevance[query_position])
            if grade > 0.0
        }
        if not relevant:
            continue
        pool = data.pools[query_position]
        baseline = data.baseline_scores[query_position].astype(np.float64)
        semantic = values[pool] @ values[query_position]
        scores = (1.0 - blend) * _zscore_columns(
            baseline[:, None]
        )[:, 0] + blend * _zscore_columns(semantic[:, None])[:, 0]
        global_order = data.global_orders[
            query_position, : int(data.global_lengths[query_position])
        ]
        baseline_order = _method_ranking(baseline, pool, global_order)
        candidate_order = _method_ranking(scores, pool, global_order)
        for method, order, destination in (
            ("baseline", baseline_order, baseline_values),
            ("candidate", candidate_order, candidate_values),
        ):
            del method
            metrics = _query_metrics(
                [int(data.track_ids[position]) for position in order],
                relevant,
                recall_cutoff=10,
                ndcg_cutoff=10,
            )
            for metric in METRICS:
                destination[metric].append(float(getattr(metrics, metric)))
        query_folds.append(int(data.query_folds[query_position]))
    if not query_folds or set(query_folds) != set(range(DEVELOPMENT_FOLDS)):
        raise V3SemanticError("development reporting folds are incomplete")
    baseline_means = {
        metric: float(np.mean(baseline_values[metric])) for metric in METRICS
    }
    candidate_means = {
        metric: float(np.mean(candidate_values[metric])) for metric in METRICS
    }
    fold_results: Dict[str, object] = {}
    positive_folds = {metric: 0 for metric in METRICS}
    worst_fold_relative_delta = {metric: float("inf") for metric in METRICS}
    folds = np.asarray(query_folds, dtype=np.int8)
    for fold_index in range(DEVELOPMENT_FOLDS):
        selected = folds == fold_index
        fold_baseline = {
            metric: float(np.mean(np.asarray(baseline_values[metric])[selected]))
            for metric in METRICS
        }
        fold_candidate = {
            metric: float(np.mean(np.asarray(candidate_values[metric])[selected]))
            for metric in METRICS
        }
        fold_delta = {
            metric: _relative_delta(
                fold_baseline[metric],
                fold_candidate[metric],
            )
            for metric in METRICS
        }
        for metric in METRICS:
            positive_folds[metric] += int(fold_delta[metric] > 0.0)
            worst_fold_relative_delta[metric] = min(
                worst_fold_relative_delta[metric],
                fold_delta[metric],
            )
        fold_results[str(fold_index)] = {
            "queries": int(np.count_nonzero(selected)),
            "baseline": fold_baseline,
            "candidate": fold_candidate,
            "relative_delta": fold_delta,
        }
    result = {
        "queries": len(query_folds),
        "baseline": baseline_means,
        "candidate": candidate_means,
        "relative_delta": {
            metric: _relative_delta(
                baseline_means[metric],
                candidate_means[metric],
            )
            for metric in METRICS
        },
        "positive_folds": positive_folds,
        "worst_fold_relative_delta": worst_fold_relative_delta,
        "folds": fold_results,
    }
    if bootstrap_iterations:
        result["paired_delta"] = {
            metric: _paired_bootstrap_delta(
                baseline_values[metric],
                candidate_values[metric],
                iterations=bootstrap_iterations,
                seed=BOOTSTRAP_SEED,
            )
            for metric in METRICS
        }
    return result


def _selection_key(result: Mapping[str, object]) -> Tuple[object, ...]:
    evaluation = result["evaluation"]
    relative = evaluation["relative_delta"]
    positive = evaluation["positive_folds"]
    worst = evaluation["worst_fold_relative_delta"]
    safe = all(
        float(relative[metric]) >= -MAX_SAFETY_RELATIVE_REGRESSION
        for metric in METRICS
    )
    stable = (
        int(positive[PRIMARY_METRIC]) >= MIN_POSITIVE_FOLDS
        and float(worst[PRIMARY_METRIC])
        >= -MAX_FOLD_PRIMARY_RELATIVE_REGRESSION
    )
    return (
        safe and stable,
        stable,
        float(relative[PRIMARY_METRIC]),
        float(relative["graded_ndcg_at_k"]),
        float(relative["mrr"]),
        -float(result["blend"]),
        -float(result["ridge"]),
        str(result["representation"]) == "dual",
    )


def development_gate(evaluation: Mapping[str, object]) -> Mapping[str, object]:
    relative = evaluation["relative_delta"]
    paired = evaluation["paired_delta"][PRIMARY_METRIC]
    checks = {
        "primary_relative_gain": (
            float(relative[PRIMARY_METRIC])
            >= MIN_DEVELOPMENT_PRIMARY_RELATIVE_GAIN
        ),
        "primary_paired_ci_above_zero": (
            float(paired["paired_bootstrap_ci95"][0]) > 0.0
        ),
        "primary_positive_folds": (
            int(evaluation["positive_folds"][PRIMARY_METRIC])
            >= MIN_POSITIVE_FOLDS
        ),
        "primary_worst_fold": (
            float(evaluation["worst_fold_relative_delta"][PRIMARY_METRIC])
            >= -MAX_FOLD_PRIMARY_RELATIVE_REGRESSION
        ),
        "safety_metrics": all(
            float(relative[metric]) >= -MAX_SAFETY_RELATIVE_REGRESSION
            for metric in METRICS
            if metric != PRIMARY_METRIC
        ),
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "decision": (
            "freeze_for_one_time_shadow_audit"
            if all(checks.values())
            else "continue_development_without_shadow_access"
        ),
    }


def _validate_musicfm_store(
    reader: FullTrackStoreReader,
    protocol: Mapping[str, object],
) -> None:
    expected = {
        "source_fingerprint": SOURCE_FINGERPRINT,
        "config_sha256": SCALE_MUSICFM_CONFIG_SHA256,
        "model_sha256": MUSICFM_MODEL_SHA256,
        "model_id": MUSICFM_MODEL_ID,
        "embedding_dim": 1024,
        "track_count": 8_192,
        "shard_tracks": 64,
        "repetition_sections": 32,
        "salient_sections": 32,
        "track_plan_sha256": SCALE_MUSICFM_TRACK_PLAN_SHA256,
    }
    actual = reader.binding.as_dict()
    actual["source_fingerprint"] = reader.binding.source_fingerprint
    drift = {
        key: (value, actual.get(key))
        for key, value in expected.items()
        if actual.get(key) != value
    }
    entries = protocol.get("tracks")
    if drift:
        raise V3SemanticError(f"MusicFM scale store binding drift: {drift}")
    if not isinstance(entries, list) or tuple(reader.track_ids) != tuple(
        int(entry["track_id"]) for entry in entries
    ):
        raise V3SemanticError("MusicFM scale store track order drift")


def _model_arrays(head: SemanticHead) -> Mapping[str, np.ndarray]:
    head.validate()
    return {
        "input_mean": head.input_mean.astype(np.float32),
        "input_scale": head.input_scale.astype(np.float32),
        "coefficients": head.coefficients.astype(np.float32),
        "prior": head.prior.astype(np.float32),
        "idf": head.idf.astype(np.float32),
        "vocabulary": np.asarray(head.vocabulary, dtype=np.str_),
    }


def train_scaled_semantic_head(
    *,
    metadata_root: Path,
    protocol_path: Path,
    clap_store: Path,
    musicfm_store: Path,
    model_output: Path,
    metadata_output: Path,
    report_output: Path,
) -> Mapping[str, object]:
    outputs = (Path(model_output), Path(metadata_output), Path(report_output))
    if any(path.exists() for path in outputs):
        raise V3SemanticError("semantic-head output already exists; refusing overwrite")
    protocol = load_protocol(Path(protocol_path))
    if (
        protocol.get("artifact_kind") != PROTOCOL_KIND
        or protocol.get("payload_sha256") != SCALE_PROTOCOL_PAYLOAD_SHA256
        or protocol.get("selection_sha256") != EXPECTED_SELECTION_SHA256
    ):
        raise V3SemanticError("scale protocol binding drift")
    train_entries = _protocol_entries(protocol, "train")
    development_entries = _protocol_entries(protocol, "development")
    labels = load_train_development_tags(Path(metadata_root), protocol)
    vocabulary, train_targets = build_label_targets(train_entries, labels)
    clap_reader = _open_bound_store(
        Path(clap_store),
        expected_manifest_file_sha256=CLAP_MANIFEST_FILE_SHA256,
        expected_binding=EXPECTED_CLAP_BINDING,
    )
    music_reader = FullTrackStoreReader(
        Path(musicfm_store),
        expected_source_fingerprint=SOURCE_FINGERPRINT,
        expected_config_sha256=SCALE_MUSICFM_CONFIG_SHA256,
        expected_model_sha256=MUSICFM_MODEL_SHA256,
    )
    try:
        _validate_musicfm_store(music_reader, protocol)
        train_ids = [int(entry["track_id"]) for entry in train_entries]
        development_ids = [
            int(entry["track_id"]) for entry in development_entries
        ]
        clap_train = _global_embeddings(clap_reader, train_ids)
        music_train = _global_embeddings(music_reader, train_ids)
        development_data = build_development_data(
            development_entries,
            labels,
            clap_reader,
        )
        clap_development = _global_embeddings(
            clap_reader,
            development_data.track_ids,
        )
        music_development = _global_embeddings(
            music_reader,
            development_data.track_ids,
        )
        if set(development_ids) != set(int(value) for value in development_data.track_ids):
            raise V3SemanticError("development embedding selection drift")
        heads: Dict[Tuple[str, float], SemanticHead] = {}
        results = []
        for representation in REPRESENTATIONS:
            train_inputs = _representation_inputs(
                representation,
                clap_train,
                music_train,
            )
            development_inputs = _representation_inputs(
                representation,
                clap_development,
                music_development,
            )
            for ridge in RIDGE_VALUES:
                head = fit_semantic_head(
                    train_inputs,
                    train_targets,
                    vocabulary,
                    representation=representation,
                    ridge=ridge,
                )
                heads[(representation, ridge)] = head
                profiles = head.predict(development_inputs)
                for blend in BLEND_VALUES:
                    results.append(
                        {
                            "representation": representation,
                            "ridge": ridge,
                            "blend": blend,
                            "evaluation": evaluate_profiles(
                                development_data,
                                profiles,
                                blend=blend,
                                bootstrap_iterations=0,
                            ),
                        }
                    )
        results.sort(key=_selection_key, reverse=True)
        best = results[0]
        best_head = heads[(str(best["representation"]), float(best["ridge"]))]
        best_inputs = _representation_inputs(
            best_head.representation,
            clap_development,
            music_development,
        )
        best["evaluation"] = evaluate_profiles(
            development_data,
            best_head.predict(best_inputs),
            blend=float(best["blend"]),
        )
        _write_npz_exclusive(Path(model_output), _model_arrays(best_head))
        report: Dict[str, object] = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "artifact_kind": REPORT_KIND,
            "evidence_scope": EVIDENCE_SCOPE,
            "evidence_status": "development_only",
            "source_fingerprint": SOURCE_FINGERPRINT,
            "protocol_payload_sha256": SCALE_PROTOCOL_PAYLOAD_SHA256,
            "protocol_selection_sha256": EXPECTED_SELECTION_SHA256,
            "opened_label_splits": ["train", "development"],
            "shadow_labels_accessed": False,
            "shadow_evaluation_accessed": False,
            "train_tracks": len(train_entries),
            "development_tracks": len(development_entries),
            "tag_count": len(vocabulary),
            "representations": list(REPRESENTATIONS),
            "ridge_values": list(RIDGE_VALUES),
            "blend_values": list(BLEND_VALUES),
            "development_fold_seed": DEVELOPMENT_FOLD_SEED,
            "candidate_pool": CANDIDATE_POOL,
            "maxsim_budget": MAXSIM_BUDGET,
            "selection_rule": (
                "prefer safety-plus-stability, then Recall@10, NDCG@10, MRR, "
                "lower blend, lower ridge, and dual representation"
            ),
            "clap_manifest_file_sha256": sha256_path(
                Path(clap_store) / "store.sealed.json"
            ),
            "musicfm_manifest_file_sha256": sha256_path(
                Path(musicfm_store) / "store.sealed.json"
            ),
            "label_source_sha256": sha256_path(
                Path(metadata_root)
                / "data"
                / "splits"
                / f"split-{BASE_FOLD}"
                / f"autotagging-{BASE_PART}.tsv"
            ),
            "model_npz_sha256": sha256_path(Path(model_output)),
            "results": results,
            "best": best,
            "development_gate": development_gate(best["evaluation"]),
            "promotion_allowed": False,
        }
        report["payload_sha256"] = stable_json_sha256(report)
        _write_json_exclusive(Path(report_output), report)
        metadata: Dict[str, object] = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "artifact_kind": MODEL_KIND,
            "evidence_scope": EVIDENCE_SCOPE,
            "source_fingerprint": SOURCE_FINGERPRINT,
            "protocol_payload_sha256": SCALE_PROTOCOL_PAYLOAD_SHA256,
            "protocol_selection_sha256": EXPECTED_SELECTION_SHA256,
            "representation": best_head.representation,
            "ridge": best_head.ridge,
            "blend": best["blend"],
            "input_dimension": int(best_head.input_mean.shape[0]),
            "tag_count": len(best_head.vocabulary),
            "vocabulary_sha256": stable_json_sha256(best_head.vocabulary),
            "model_npz_sha256": sha256_path(Path(model_output)),
            "development_report_file_sha256": sha256_path(Path(report_output)),
            "development_report_payload_sha256": report["payload_sha256"],
            "shadow_labels_accessed": False,
            "promotion_allowed": False,
        }
        metadata["payload_sha256"] = stable_json_sha256(metadata)
        _write_json_exclusive(Path(metadata_output), metadata)
        return report
    finally:
        clap_reader.close()
        music_reader.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--clap-store", required=True)
    parser.add_argument("--musicfm-store", required=True)
    parser.add_argument("--model-output", required=True)
    parser.add_argument("--metadata-output", required=True)
    parser.add_argument("--report-output", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = train_scaled_semantic_head(
            metadata_root=Path(args.metadata_root),
            protocol_path=Path(args.protocol),
            clap_store=Path(args.clap_store),
            musicfm_store=Path(args.musicfm_store),
            model_output=Path(args.model_output),
            metadata_output=Path(args.metadata_output),
            report_output=Path(args.report_output),
        )
    except (OSError, ValueError, V3SemanticError) as exc:
        raise SystemExit(f"V3 semantic-head training failed: {exc}") from exc
    print(
        json.dumps(
            {
                "best": report["best"],
                "development_gate": report["development_gate"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
