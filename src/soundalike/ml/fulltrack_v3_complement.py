"""Freeze and audit the complementary CLAP-neighbor/MusicFM V3 candidate."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .fulltrack_eval import (
    METRICS,
    _BudgetCache,
    _tag_jaccard_relevance,
)
from .fulltrack_store import (
    FullTrackStoreError,
    FullTrackStoreReader,
    sha256_path,
    stable_json_sha256,
)
from .fulltrack_v3 import (
    CANDIDATE_POOL,
    CLAP_MANIFEST_FILE_SHA256,
    EXPECTED_CLAP_BINDING,
    MAXSIM_BUDGET,
    MAX_FEATURE_CACHE_BYTES,
    SOURCE_FINGERPRINT,
    _open_bound_store,
    _replace_json,
)
from .fulltrack_v3_metric import _evaluate_scores, weighted_knn_profiles
from .fulltrack_v3_protocol import (
    BASE_FOLD,
    BASE_PART,
    EXPECTED_SELECTION_SHA256,
    EXPECTED_SPLITS,
    PROTOCOL_KIND,
    load_protocol,
)
from .fulltrack_v3_ranker import (
    MUSICFM_MODEL_SHA256,
    _score_channels,
    _write_json_exclusive,
    _write_npz_exclusive,
    _zscore_columns,
)
from .fulltrack_v3_semantic import (
    SCALE_MUSICFM_CONFIG_SHA256,
    SCALE_PROTOCOL_PAYLOAD_SHA256,
    DevelopmentData,
    SemanticHead,
    V3SemanticError,
    _global_embeddings,
    _protocol_entries,
    _validate_musicfm_store,
    build_development_data,
    build_label_targets,
    development_gate,
    load_protocol_tags,
    load_train_development_tags,
)
from .jamendo_fulltrack import EVIDENCE_SCOPE


MODEL_SCHEMA_VERSION = 1
MODEL_KIND = "v3_complementary_clap_knn_musicfm"
REPORT_SCHEMA_VERSION = 1
DEVELOPMENT_REPORT_KIND = "v3_complementary_development_report"
FREEZE_KIND = "v3_complementary_shadow_freeze"
SHADOW_REPORT_KIND = "v3_complementary_shadow_audit"
SHADOW_STATE_KIND = "v3_complementary_shadow_audit_state"
MUSICFM_HEAD_SHA256 = (
    "88ac87b9fcecd6f0ddf071ff687891d176a360e7f8ea62ca29996b16d873b4d7"
)
COMPLEMENT_PROBE_PAYLOAD_SHA256 = (
    "1852c7b3b0b2788e5e206d399f0ce8cc6a913bb6c4fc83de2cc1b65c81e2bb83"
)
PROFILE_SHARES = (0.5, 0.5)
BASELINE_BLEND = 0.5
MUSICFM_RIDGE = 10.0
SHADOW_FOLDS = 5
SHADOW_FOLD_SEED = 20260808
MIN_SHADOW_PRIMARY_RELATIVE_GAIN = 0.20
MIN_POSITIVE_SHADOW_FOLDS = 4
MAX_SHADOW_FOLD_PRIMARY_REGRESSION = 0.05
MAX_SHADOW_SAFETY_REGRESSION = 0.01
PRIMARY_METRIC = "recall_at_k"
EXPECTED_DEVELOPMENT_RELATIVE = {
    "recall_at_k": 0.15056579301514383,
    "mrr": 0.011778451599638418,
    "graded_ndcg_at_k": 0.049755280225821874,
}


class V3ComplementError(RuntimeError):
    """Invalid, unfrozen, leaky, or non-reproducible complementary V3 run."""


@dataclass(frozen=True)
class ComplementModel:
    vocabulary: Tuple[str, ...]
    train_track_ids: np.ndarray
    train_clap: np.ndarray
    train_targets: np.ndarray
    idf: np.ndarray
    musicfm_head: SemanticHead

    def validate(self) -> None:
        train_count = EXPECTED_SPLITS["train"]["tracks"]
        tag_count = len(self.vocabulary)
        self.musicfm_head.validate()
        if (
            self.musicfm_head.representation != "musicfm"
            or self.musicfm_head.ridge != MUSICFM_RIDGE
            or self.musicfm_head.vocabulary != self.vocabulary
            or self.train_track_ids.shape != (train_count,)
            or self.train_clap.ndim != 2
            or self.train_clap.shape[0] != train_count
            or self.train_targets.shape != (train_count, tag_count)
            or self.idf.shape != (tag_count,)
            or len(np.unique(self.train_track_ids)) != train_count
            or tuple(sorted(self.vocabulary)) != self.vocabulary
            or len(set(self.vocabulary)) != tag_count
            or not tag_count
        ):
            raise V3ComplementError("complement model shape or identity drift")
        for values in (self.train_clap, self.train_targets, self.idf):
            if not np.all(np.isfinite(values)):
                raise V3ComplementError("complement model contains non-finite values")
        if (
            not np.all((self.train_targets == 0.0) | (self.train_targets == 1.0))
            or np.any(np.sum(self.train_targets, axis=1) <= 0.0)
            or np.any(self.idf <= 0.0)
        ):
            raise V3ComplementError("complement model labels are invalid")


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
            raise V3ComplementError("unknown evaluation split")
        count = int(expected["tracks"])
        if (
            self.track_ids.shape != (count,)
            or self.artist_ids.shape != (count,)
            or self.query_folds.shape != (count,)
            or self.global_orders.shape != (count, count - 1)
            or self.global_lengths.shape != (count,)
            or self.pools.ndim != 2
            or self.pools.shape[0] != count
            or self.baseline_scores.shape != self.pools.shape
            or self.relevance.shape != (count, count)
            or len(np.unique(self.track_ids)) != count
            or len(np.unique(self.artist_ids)) != int(expected["artists"])
            or np.any(self.query_folds < 0)
            or np.any(self.query_folds >= SHADOW_FOLDS)
            or not np.all(np.isfinite(self.baseline_scores))
            or not np.all(np.isfinite(self.relevance))
        ):
            raise V3ComplementError("evaluation partition shape or value drift")
        for query_position, length in enumerate(self.global_lengths):
            order = self.global_orders[query_position, : int(length)]
            if (
                length < self.pools.shape[1]
                or np.any(order < 0)
                or len(np.unique(order)) != len(order)
                or not np.array_equal(
                    order[: self.pools.shape[1]],
                    self.pools[query_position],
                )
            ):
                raise V3ComplementError("evaluation partition order drift")


def _load_musicfm_head(path: Path, vocabulary: Sequence[str]) -> SemanticHead:
    model_path = Path(path)
    if sha256_path(model_path) != MUSICFM_HEAD_SHA256:
        raise V3ComplementError("selected MusicFM head file hash drift")
    with np.load(model_path, allow_pickle=False) as archive:
        model_vocabulary = tuple(str(value) for value in archive["vocabulary"])
        if model_vocabulary != tuple(vocabulary):
            raise V3ComplementError("selected MusicFM head vocabulary drift")
        head = SemanticHead(
            representation="musicfm",
            ridge=MUSICFM_RIDGE,
            vocabulary=model_vocabulary,
            input_mean=np.asarray(archive["input_mean"], dtype=np.float64),
            input_scale=np.asarray(archive["input_scale"], dtype=np.float64),
            coefficients=np.asarray(archive["coefficients"], dtype=np.float64),
            prior=np.asarray(archive["prior"], dtype=np.float64),
            idf=np.asarray(archive["idf"], dtype=np.float64),
        )
    head.validate()
    return head


def _model_arrays(model: ComplementModel) -> Mapping[str, np.ndarray]:
    model.validate()
    head = model.musicfm_head
    return {
        "vocabulary": np.asarray(model.vocabulary, dtype=np.str_),
        "train_track_ids": model.train_track_ids.astype(np.int64),
        "train_clap": model.train_clap.astype(np.float64),
        "train_targets": model.train_targets.astype(np.uint8),
        "idf": model.idf.astype(np.float64),
        "musicfm_input_mean": head.input_mean.astype(np.float64),
        "musicfm_input_scale": head.input_scale.astype(np.float64),
        "musicfm_coefficients": head.coefficients.astype(np.float64),
        "musicfm_prior": head.prior.astype(np.float64),
        "musicfm_idf": head.idf.astype(np.float64),
    }


def load_complement_model(path: Path) -> ComplementModel:
    with np.load(Path(path), allow_pickle=False) as archive:
        vocabulary = tuple(str(value) for value in archive["vocabulary"])
        model = ComplementModel(
            vocabulary=vocabulary,
            train_track_ids=np.asarray(archive["train_track_ids"], dtype=np.int64),
            train_clap=np.asarray(archive["train_clap"], dtype=np.float64),
            train_targets=np.asarray(archive["train_targets"], dtype=np.float64),
            idf=np.asarray(archive["idf"], dtype=np.float64),
            musicfm_head=SemanticHead(
                representation="musicfm",
                ridge=MUSICFM_RIDGE,
                vocabulary=vocabulary,
                input_mean=np.asarray(
                    archive["musicfm_input_mean"], dtype=np.float64
                ),
                input_scale=np.asarray(
                    archive["musicfm_input_scale"], dtype=np.float64
                ),
                coefficients=np.asarray(
                    archive["musicfm_coefficients"], dtype=np.float64
                ),
                prior=np.asarray(archive["musicfm_prior"], dtype=np.float64),
                idf=np.asarray(archive["musicfm_idf"], dtype=np.float64),
            ),
        )
    model.validate()
    return model


def complementary_profiles(
    knn_profiles: np.ndarray,
    musicfm_profiles: np.ndarray,
) -> np.ndarray:
    knn = np.asarray(knn_profiles, dtype=np.float64)
    musicfm = np.asarray(musicfm_profiles, dtype=np.float64)
    if (
        knn.ndim != 2
        or musicfm.ndim != 2
        or len(knn) != len(musicfm)
        or not len(knn)
        or not np.all(np.isfinite(knn))
        or not np.all(np.isfinite(musicfm))
        or not np.allclose(np.linalg.norm(knn, axis=1), 1.0, atol=1e-7)
        or not np.allclose(np.linalg.norm(musicfm, axis=1), 1.0, atol=1e-7)
    ):
        raise V3ComplementError("complementary profile inputs are invalid")
    return np.concatenate(
        (
            np.sqrt(PROFILE_SHARES[0]) * knn,
            np.sqrt(PROFILE_SHARES[1]) * musicfm,
        ),
        axis=1,
    )


def complementary_scores(
    data: object,
    profiles: np.ndarray,
) -> np.ndarray:
    data.validate()
    values = np.asarray(profiles, dtype=np.float64)
    if (
        values.ndim != 2
        or len(values) != len(data.track_ids)
        or not np.all(np.isfinite(values))
    ):
        raise V3ComplementError("complementary score inputs are invalid")
    scores = np.empty_like(data.baseline_scores, dtype=np.float64)
    for query_position in range(len(data.track_ids)):
        pool = data.pools[query_position]
        semantic = values[pool] @ values[query_position]
        scores[query_position] = (
            (1.0 - BASELINE_BLEND)
            * _zscore_columns(
                data.baseline_scores[query_position].astype(np.float64)[:, None]
            )[:, 0]
            + BASELINE_BLEND * _zscore_columns(semantic[:, None])[:, 0]
        )
    return scores


def _candidate_profiles(
    model: ComplementModel,
    clap_values: np.ndarray,
    musicfm_values: np.ndarray,
) -> Tuple[np.ndarray, Mapping[str, object]]:
    knn, _, evidence = weighted_knn_profiles(
        model.train_clap,
        model.train_targets,
        clap_values,
        model.idf,
    )
    musicfm = model.musicfm_head.predict(musicfm_values)
    return complementary_profiles(knn, musicfm), evidence


def _partition_fold(artist_id: int) -> int:
    return (
        int(
            stable_json_sha256(
                {"seed": SHADOW_FOLD_SEED, "artist_id": int(artist_id)}
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
            sorted(entries, key=lambda entry: positions[int(entry["track_id"])])
        )
    except KeyError as exc:
        raise V3ComplementError(f"CLAP store is missing track {exc.args[0]}") from exc
    track_ids = np.asarray(
        [int(entry["track_id"]) for entry in ordered], dtype=np.int64
    )
    artist_ids = np.asarray(
        [int(entry["artist_id"]) for entry in ordered], dtype=np.int64
    )
    if set(labels) != set(int(value) for value in track_ids):
        raise V3ComplementError("evaluation labels differ from partition")
    query_folds = np.asarray(
        [_partition_fold(int(artist)) for artist in artist_ids], dtype=np.int8
    )
    count = len(track_ids)
    pool_size = CANDIDATE_POOL
    budget = _BudgetCache(
        clap_reader,
        track_ids,
        budget=MAXSIM_BUDGET,
        max_bytes=MAX_FEATURE_CACHE_BYTES,
    )
    globals_ = _global_embeddings(clap_reader, track_ids)
    global_orders = np.full((count, count - 1), -1, dtype=np.int32)
    global_lengths = np.zeros(count, dtype=np.int32)
    pools = np.empty((count, pool_size), dtype=np.int32)
    baseline_scores = np.empty((count, pool_size), dtype=np.float32)
    relevance = np.zeros((count, count), dtype=np.float32)
    for query_position in range(count):
        eligible = np.flatnonzero(
            (track_ids != track_ids[query_position])
            & (artist_ids != artist_ids[query_position])
        )
        global_scores = globals_[eligible] @ globals_[query_position]
        order = eligible[np.lexsort((track_ids[eligible], -global_scores))]
        if len(order) < pool_size:
            raise V3ComplementError("evaluation candidate universe is too small")
        global_orders[query_position, : len(order)] = order
        global_lengths[query_position] = len(order)
        pool = order[:pool_size]
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
    data = PartitionData(
        split=split,
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


def _validate_protocol(path: Path) -> Mapping[str, object]:
    protocol = load_protocol(Path(path))
    if (
        protocol.get("artifact_kind") != PROTOCOL_KIND
        or protocol.get("payload_sha256") != SCALE_PROTOCOL_PAYLOAD_SHA256
        or protocol.get("selection_sha256") != EXPECTED_SELECTION_SHA256
    ):
        raise V3ComplementError("scale protocol binding drift")
    return protocol


def _open_stores(
    clap_store: Path,
    musicfm_store: Path,
    protocol: Mapping[str, object],
) -> Tuple[FullTrackStoreReader, FullTrackStoreReader]:
    clap_reader = _open_bound_store(
        Path(clap_store),
        expected_manifest_file_sha256=CLAP_MANIFEST_FILE_SHA256,
        expected_binding=EXPECTED_CLAP_BINDING,
    )
    try:
        music_reader = FullTrackStoreReader(
            Path(musicfm_store),
            expected_source_fingerprint=SOURCE_FINGERPRINT,
            expected_config_sha256=SCALE_MUSICFM_CONFIG_SHA256,
            expected_model_sha256=MUSICFM_MODEL_SHA256,
        )
        _validate_musicfm_store(music_reader, protocol)
    except (OSError, ValueError, FullTrackStoreError, V3SemanticError):
        clap_reader.close()
        raise
    return clap_reader, music_reader


def _label_source(metadata_root: Path) -> Path:
    return (
        Path(metadata_root)
        / "data"
        / "splits"
        / f"split-{BASE_FOLD}"
        / f"autotagging-{BASE_PART}.tsv"
    )


def _verify_expected_development(evaluation: Mapping[str, object]) -> None:
    relative = evaluation["relative_delta"]
    for metric, expected in EXPECTED_DEVELOPMENT_RELATIVE.items():
        if not np.isclose(float(relative[metric]), expected, rtol=0.0, atol=1e-12):
            raise V3ComplementError(
                f"production candidate does not reproduce probe {metric}"
            )


def build_complement_candidate(
    *,
    metadata_root: Path,
    protocol_path: Path,
    clap_store: Path,
    musicfm_store: Path,
    musicfm_head_path: Path,
    model_output: Path,
    metadata_output: Path,
    report_output: Path,
) -> Mapping[str, object]:
    outputs = (Path(model_output), Path(metadata_output), Path(report_output))
    if any(path.exists() for path in outputs):
        raise V3ComplementError("candidate output already exists; refusing overwrite")
    protocol = _validate_protocol(protocol_path)
    train_entries = _protocol_entries(protocol, "train")
    development_entries = _protocol_entries(protocol, "development")
    labels = load_train_development_tags(Path(metadata_root), protocol)
    vocabulary, targets = build_label_targets(train_entries, labels)
    train_ids = np.asarray(
        [int(entry["track_id"]) for entry in train_entries], dtype=np.int64
    )
    idf = np.log(
        (len(targets) + 1.0) / (np.sum(targets, axis=0) + 1.0)
    ) + 1.0
    head = _load_musicfm_head(Path(musicfm_head_path), vocabulary)
    clap_reader, music_reader = _open_stores(
        clap_store,
        musicfm_store,
        protocol,
    )
    try:
        train_clap = _global_embeddings(clap_reader, train_ids)
        model = ComplementModel(
            vocabulary=vocabulary,
            train_track_ids=train_ids,
            train_clap=train_clap,
            train_targets=targets,
            idf=idf,
            musicfm_head=head,
        )
        model.validate()
        development_data = build_development_data(
            development_entries,
            labels,
            clap_reader,
        )
        clap_development = _global_embeddings(
            clap_reader,
            development_data.track_ids,
        )
        musicfm_development = _global_embeddings(
            music_reader,
            development_data.track_ids,
        )
        profiles, knn_evidence = _candidate_profiles(
            model,
            clap_development,
            musicfm_development,
        )
        evaluation = _evaluate_scores(
            development_data,
            complementary_scores(development_data, profiles),
        )
        _verify_expected_development(evaluation)
        gate = development_gate(evaluation)
        if not gate["passed"]:
            raise V3ComplementError("reproduced candidate failed development gate")
        _write_npz_exclusive(Path(model_output), _model_arrays(model))
        report: Dict[str, object] = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "artifact_kind": DEVELOPMENT_REPORT_KIND,
            "evidence_scope": EVIDENCE_SCOPE,
            "evidence_status": "development_only",
            "source_fingerprint": SOURCE_FINGERPRINT,
            "protocol_payload_sha256": SCALE_PROTOCOL_PAYLOAD_SHA256,
            "protocol_selection_sha256": EXPECTED_SELECTION_SHA256,
            "opened_label_splits": ["train", "development"],
            "shadow_labels_accessed": False,
            "shadow_evaluation_accessed": False,
            "development_consumed_for_selection": True,
            "complement_probe_payload_sha256": COMPLEMENT_PROBE_PAYLOAD_SHA256,
            "frozen_method": {
                "profile_mode": "concatenated_weighted_channels",
                "clap_profile": "train_only_idf_weighted_16_neighbor_tags",
                "clap_profile_share": PROFILE_SHARES[0],
                "musicfm_profile": "selected_layer7_ridge_10_tag_head",
                "musicfm_profile_share": PROFILE_SHARES[1],
                "baseline_blend": 1.0 - BASELINE_BLEND,
                "semantic_blend": BASELINE_BLEND,
            },
            "train_tracks": len(train_entries),
            "development_tracks": len(development_entries),
            "tag_count": len(vocabulary),
            "knn": dict(knn_evidence),
            "evaluation": evaluation,
            "development_gate": gate,
            "model_npz_sha256": sha256_path(Path(model_output)),
            "musicfm_source_head_sha256": sha256_path(Path(musicfm_head_path)),
            "clap_manifest_file_sha256": sha256_path(
                Path(clap_store) / "store.sealed.json"
            ),
            "musicfm_manifest_file_sha256": sha256_path(
                Path(musicfm_store) / "store.sealed.json"
            ),
            "label_source_sha256": sha256_path(_label_source(metadata_root)),
            "promotion_allowed": False,
        }
        report["payload_sha256"] = stable_json_sha256(report)
        _write_json_exclusive(Path(report_output), report)
        metadata: Dict[str, object] = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "artifact_kind": MODEL_KIND,
            "source_fingerprint": SOURCE_FINGERPRINT,
            "protocol_payload_sha256": SCALE_PROTOCOL_PAYLOAD_SHA256,
            "protocol_selection_sha256": EXPECTED_SELECTION_SHA256,
            "model_npz_sha256": sha256_path(Path(model_output)),
            "development_report_file_sha256": sha256_path(Path(report_output)),
            "development_report_payload_sha256": report["payload_sha256"],
            "train_tracks": len(train_entries),
            "tag_count": len(vocabulary),
            "shadow_labels_accessed": False,
            "promotion_allowed": False,
        }
        metadata["payload_sha256"] = stable_json_sha256(metadata)
        _write_json_exclusive(Path(metadata_output), metadata)
        return report
    finally:
        clap_reader.close()
        music_reader.close()


def _read_json(path: Path, label: str) -> Mapping[str, object]:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise V3ComplementError(f"{label} must be a concrete file")
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V3ComplementError(f"{label} is invalid: {exc}") from exc
    if not isinstance(value, dict):
        raise V3ComplementError(f"{label} must contain a JSON object")
    return value


def _payload_sha256(document: Mapping[str, object]) -> str:
    payload = dict(document)
    payload.pop("payload_sha256", None)
    return stable_json_sha256(payload)


def freeze_complement_candidate(
    *,
    protocol_path: Path,
    clap_store: Path,
    musicfm_store: Path,
    model_path: Path,
    metadata_path: Path,
    development_report_path: Path,
    output: Path,
) -> Mapping[str, object]:
    if Path(output).exists():
        raise V3ComplementError("freeze output already exists; refusing overwrite")
    protocol = _validate_protocol(protocol_path)
    metadata = _read_json(metadata_path, "candidate metadata")
    report = _read_json(development_report_path, "development report")
    if (
        metadata.get("artifact_kind") != MODEL_KIND
        or report.get("artifact_kind") != DEVELOPMENT_REPORT_KIND
        or metadata.get("payload_sha256") != _payload_sha256(metadata)
        or report.get("payload_sha256") != _payload_sha256(report)
        or metadata.get("model_npz_sha256") != sha256_path(Path(model_path))
        or metadata.get("development_report_file_sha256")
        != sha256_path(Path(development_report_path))
        or not report.get("development_gate", {}).get("passed")
        or report.get("shadow_labels_accessed") is not False
    ):
        raise V3ComplementError("candidate evidence is not eligible for freezing")
    load_complement_model(model_path)
    document: Dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": FREEZE_KIND,
        "source_fingerprint": SOURCE_FINGERPRINT,
        "protocol_payload_sha256": protocol["payload_sha256"],
        "protocol_selection_sha256": protocol["selection_sha256"],
        "candidate_model_file_sha256": sha256_path(Path(model_path)),
        "candidate_metadata_file_sha256": sha256_path(Path(metadata_path)),
        "candidate_metadata_payload_sha256": metadata["payload_sha256"],
        "development_report_file_sha256": sha256_path(
            Path(development_report_path)
        ),
        "development_report_payload_sha256": report["payload_sha256"],
        "clap_manifest_file_sha256": sha256_path(
            Path(clap_store) / "store.sealed.json"
        ),
        "musicfm_manifest_file_sha256": sha256_path(
            Path(musicfm_store) / "store.sealed.json"
        ),
        "frozen_method": report["frozen_method"],
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
        "shadow_gate": {
            "minimum_recall_relative_gain": MIN_SHADOW_PRIMARY_RELATIVE_GAIN,
            "paired_recall_ci_must_exclude_zero": True,
            "minimum_positive_recall_folds": MIN_POSITIVE_SHADOW_FOLDS,
            "maximum_fold_recall_regression": (
                MAX_SHADOW_FOLD_PRIMARY_REGRESSION
            ),
            "maximum_mrr_ndcg_regression": MAX_SHADOW_SAFETY_REGRESSION,
        },
        "shadow_fold_seed": SHADOW_FOLD_SEED,
        "development_result_seen": True,
        "shadow_labels_accessed": False,
        "promotion_allowed": False,
    }
    document["payload_sha256"] = stable_json_sha256(document)
    _write_json_exclusive(Path(output), document)
    return document


def _shadow_gate(evaluation: Mapping[str, object]) -> Mapping[str, object]:
    relative = evaluation["relative_delta"]
    checks = {
        "primary_relative_gain": (
            float(relative[PRIMARY_METRIC])
            >= MIN_SHADOW_PRIMARY_RELATIVE_GAIN
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
            >= MIN_POSITIVE_SHADOW_FOLDS
        ),
        "primary_worst_fold": (
            float(evaluation["worst_fold_relative_delta"][PRIMARY_METRIC])
            >= -MAX_SHADOW_FOLD_PRIMARY_REGRESSION
        ),
        "safety_metrics": all(
            float(relative[metric]) >= -MAX_SHADOW_SAFETY_REGRESSION
            for metric in METRICS
            if metric != PRIMARY_METRIC
        ),
    }
    return {
        "checks": checks,
        "automated_passed": all(checks.values()),
        "human_pilot_required": True,
        "promotion_allowed": False,
    }


def _verify_freeze(
    freeze_path: Path,
    *,
    protocol: Mapping[str, object],
    clap_store: Path,
    musicfm_store: Path,
    model_path: Path,
    metadata_path: Path,
    development_report_path: Path,
) -> Mapping[str, object]:
    freeze = _read_json(freeze_path, "shadow freeze")
    expected = {
        "artifact_kind": FREEZE_KIND,
        "payload_sha256": _payload_sha256(freeze),
        "protocol_payload_sha256": protocol["payload_sha256"],
        "protocol_selection_sha256": protocol["selection_sha256"],
        "candidate_model_file_sha256": sha256_path(Path(model_path)),
        "candidate_metadata_file_sha256": sha256_path(Path(metadata_path)),
        "development_report_file_sha256": sha256_path(
            Path(development_report_path)
        ),
        "clap_manifest_file_sha256": sha256_path(
            Path(clap_store) / "store.sealed.json"
        ),
        "musicfm_manifest_file_sha256": sha256_path(
            Path(musicfm_store) / "store.sealed.json"
        ),
        "shadow_labels_accessed": False,
    }
    drift = {
        key: (value, freeze.get(key))
        for key, value in expected.items()
        if freeze.get(key) != value
    }
    if drift:
        raise V3ComplementError(f"shadow freeze binding drift: {drift}")
    return freeze


def audit_frozen_shadow(
    *,
    metadata_root: Path,
    protocol_path: Path,
    clap_store: Path,
    musicfm_store: Path,
    model_path: Path,
    metadata_path: Path,
    development_report_path: Path,
    freeze_path: Path,
    output: Path,
    audit_state_path: Path,
) -> Mapping[str, object]:
    if Path(output).exists():
        raise V3ComplementError("shadow audit output already exists")
    if Path(audit_state_path).exists():
        raise V3ComplementError("shadow audit state already exists; refusing reopen")
    protocol = _validate_protocol(protocol_path)
    metadata = _read_json(metadata_path, "candidate metadata")
    development = _read_json(development_report_path, "development report")
    if (
        metadata.get("payload_sha256") != _payload_sha256(metadata)
        or development.get("payload_sha256") != _payload_sha256(development)
        or not development.get("development_gate", {}).get("passed")
    ):
        raise V3ComplementError("candidate evidence failed pre-audit validation")
    freeze = _verify_freeze(
        freeze_path,
        protocol=protocol,
        clap_store=clap_store,
        musicfm_store=musicfm_store,
        model_path=model_path,
        metadata_path=metadata_path,
        development_report_path=development_report_path,
    )
    model = load_complement_model(model_path)
    clap_reader, music_reader = _open_stores(
        clap_store,
        musicfm_store,
        protocol,
    )
    opened_state: Dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": SHADOW_STATE_KIND,
        "status": "opened",
        "protocol_payload_sha256": protocol["payload_sha256"],
        "freeze_file_sha256": sha256_path(Path(freeze_path)),
        "freeze_payload_sha256": freeze["payload_sha256"],
        "candidate_model_file_sha256": sha256_path(Path(model_path)),
        "shadow_labels_accessed": True,
        "shadow_evaluation_completed": False,
        "promotion_allowed": False,
    }
    opened_state["payload_sha256"] = stable_json_sha256(opened_state)
    try:
        _write_json_exclusive(Path(audit_state_path), opened_state)
        shadow_entries = _protocol_entries(protocol, "shadow")
        labels = load_protocol_tags(Path(metadata_root), protocol, ("shadow",))
        shadow_data = build_partition_data(
            shadow_entries,
            labels,
            clap_reader,
            split="shadow",
        )
        clap_shadow = _global_embeddings(clap_reader, shadow_data.track_ids)
        musicfm_shadow = _global_embeddings(music_reader, shadow_data.track_ids)
        profiles, knn_evidence = _candidate_profiles(
            model,
            clap_shadow,
            musicfm_shadow,
        )
        evaluation = _evaluate_scores(
            shadow_data,
            complementary_scores(shadow_data, profiles),
        )
        gate = _shadow_gate(evaluation)
        report: Dict[str, object] = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "artifact_kind": SHADOW_REPORT_KIND,
            "evidence_scope": EVIDENCE_SCOPE,
            "evidence_status": "independent_shadow_audit",
            "source_fingerprint": SOURCE_FINGERPRINT,
            "protocol_payload_sha256": protocol["payload_sha256"],
            "protocol_selection_sha256": protocol["selection_sha256"],
            "freeze_file_sha256": sha256_path(Path(freeze_path)),
            "freeze_payload_sha256": freeze["payload_sha256"],
            "candidate_model_file_sha256": sha256_path(Path(model_path)),
            "opened_label_splits": ["shadow"],
            "shadow_labels_accessed": True,
            "shadow_evaluation_accessed": True,
            "shadow_tracks": len(shadow_entries),
            "shadow_fold_seed": SHADOW_FOLD_SEED,
            "knn": dict(knn_evidence),
            "evaluation": evaluation,
            "shadow_gate": gate,
            "listening_pack_allowed": gate["automated_passed"],
            "human_pilot_required": True,
            "promotion_allowed": False,
        }
        report["payload_sha256"] = stable_json_sha256(report)
        _write_json_exclusive(Path(output), report)
        completed_state = dict(opened_state)
        completed_state.update(
            {
                "status": "completed",
                "shadow_evaluation_completed": True,
                "report_file_sha256": sha256_path(Path(output)),
                "report_payload_sha256": report["payload_sha256"],
            }
        )
        completed_state.pop("payload_sha256", None)
        completed_state["payload_sha256"] = stable_json_sha256(completed_state)
        _replace_json(Path(audit_state_path), completed_state)
        return report
    finally:
        clap_reader.close()
        music_reader.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-development")
    for name in ("metadata-root", "protocol", "clap-store", "musicfm-store"):
        build.add_argument(f"--{name}", required=True)
    build.add_argument("--musicfm-head", required=True)
    build.add_argument("--model-output", required=True)
    build.add_argument("--metadata-output", required=True)
    build.add_argument("--report-output", required=True)

    freeze = subparsers.add_parser("freeze-shadow")
    for name in (
        "protocol",
        "clap-store",
        "musicfm-store",
        "model",
        "metadata",
        "development-report",
        "output",
    ):
        freeze.add_argument(f"--{name}", required=True)

    audit = subparsers.add_parser("audit-shadow")
    for name in (
        "metadata-root",
        "protocol",
        "clap-store",
        "musicfm-store",
        "model",
        "metadata",
        "development-report",
        "freeze",
        "output",
        "audit-state",
    ):
        audit.add_argument(f"--{name}", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "build-development":
            result = build_complement_candidate(
                metadata_root=Path(args.metadata_root),
                protocol_path=Path(args.protocol),
                clap_store=Path(args.clap_store),
                musicfm_store=Path(args.musicfm_store),
                musicfm_head_path=Path(args.musicfm_head),
                model_output=Path(args.model_output),
                metadata_output=Path(args.metadata_output),
                report_output=Path(args.report_output),
            )
            summary = {
                "evaluation": result["evaluation"]["relative_delta"],
                "development_gate": result["development_gate"],
            }
        elif args.command == "freeze-shadow":
            result = freeze_complement_candidate(
                protocol_path=Path(args.protocol),
                clap_store=Path(args.clap_store),
                musicfm_store=Path(args.musicfm_store),
                model_path=Path(args.model),
                metadata_path=Path(args.metadata),
                development_report_path=Path(args.development_report),
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
                musicfm_store=Path(args.musicfm_store),
                model_path=Path(args.model),
                metadata_path=Path(args.metadata),
                development_report_path=Path(args.development_report),
                freeze_path=Path(args.freeze),
                output=Path(args.output),
                audit_state_path=Path(args.audit_state),
            )
            summary = {
                "evaluation": result["evaluation"]["relative_delta"],
                "shadow_gate": result["shadow_gate"],
                "listening_pack_allowed": result["listening_pack_allowed"],
            }
    except (OSError, ValueError, V3SemanticError, V3ComplementError) as exc:
        raise SystemExit(f"V3 complementary candidate failed: {exc}") from exc
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
