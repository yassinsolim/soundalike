"""Build, freeze, and audit the fresh scaled CLAP kNN-MLP V3 candidate."""
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
    _tag_jaccard_relevance,
)
from .fulltrack_store import FullTrackStoreReader, sha256_path, stable_json_sha256
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
from .fulltrack_v3_fresh_protocol import (
    BASE_FOLD,
    BASE_PART,
    EXPECTED_SELECTION_SHA256,
    EXPECTED_SPLITS,
    PROTOCOL_KIND,
    load_fresh_protocol,
)
from .fulltrack_v3_metric import _evaluate_scores
from .fulltrack_v3_ranker import (
    _score_channels,
    _write_json_exclusive,
    _write_npz_exclusive,
    _zscore_columns,
)
from .fulltrack_v3_semantic import LABEL_HEADER
from .jamendo_fulltrack import EVIDENCE_SCOPE, _ID_PATTERNS, _TAG


MODEL_SCHEMA_VERSION = 1
MODEL_KIND = "v3_fresh_scaled_clap_knn_mlp"
DEVELOPMENT_REPORT_KIND = "v3_fresh_scaled_clap_development_report"
FREEZE_KIND = "v3_fresh_scaled_clap_shadow_freeze"
SHADOW_REPORT_KIND = "v3_fresh_scaled_clap_shadow_audit"
SHADOW_STATE_KIND = "v3_fresh_scaled_clap_shadow_audit_state"
PROTOCOL_FILE_SHA256 = (
    "c5c1fc5bf23f088e51262f2c811659ea0c30e01e43a3ac102c9f86112b4d9af4"
)
PROTOCOL_PAYLOAD_SHA256 = (
    "8d43ca8099ffc604555eef5b180ee1aee38520ddb9e7e269a672533767e566a3"
)
REFINEMENT_PAYLOAD_SHA256 = (
    "7535071cdceb44db9fcceab30ef2700adc0834bb537b160bd5a1b64257da0e87"
)
SEED = 20260812
FOLD_SEED = 20260811
HIDDEN_DIMENSION = 768
LATENT_DIMENSION = 256
EPOCHS = 150
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-4
K_NEIGHBORS = 8
KNN_TEMPERATURE = 0.05
KNN_PREDICTION_POWER = 0.75
KNN_IDF_POWER = 1.0
KNN_SHARE = 0.50
MLP_SHARE = 0.50
SEMANTIC_BLEND = 0.70
MIN_DEVELOPMENT_RECALL_GAIN = 0.20
MIN_SHADOW_RECALL_GAIN = 0.20
MIN_POSITIVE_FOLDS = 4
MAX_FOLD_RECALL_REGRESSION = 0.05
MAX_SAFETY_REGRESSION = 0.01
PRIMARY_METRIC = "recall_at_k"
EXPECTED_DEVELOPMENT_RELATIVE = {
    "recall_at_k": 0.25266108522218267,
    "mrr": 0.12765247292025128,
    "graded_ndcg_at_k": 0.14176583176612456,
}


class V3FreshCandidateError(RuntimeError):
    """Invalid, leaky, changed, or prematurely opened fresh V3 candidate."""


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
            raise V3FreshCandidateError("unknown evaluation split")
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
            or np.any(self.query_folds >= 5)
            or not np.all(np.isfinite(self.baseline_scores))
            or not np.all(np.isfinite(self.relevance))
        ):
            raise V3FreshCandidateError("evaluation partition shape or value drift")
        for query_position, length in enumerate(self.global_lengths):
            order = self.global_orders[query_position, : int(length)]
            if (
                length < CANDIDATE_POOL
                or np.any(order < 0)
                or len(np.unique(order)) != len(order)
                or not np.array_equal(
                    order[:CANDIDATE_POOL],
                    self.pools[query_position],
                )
            ):
                raise V3FreshCandidateError("evaluation partition order drift")


@dataclass(frozen=True)
class FreshCandidateModel:
    vocabulary: Tuple[str, ...]
    train_track_ids: np.ndarray
    train_clap: np.ndarray
    train_targets: np.ndarray
    idf: np.ndarray
    mlp_state: Mapping[str, np.ndarray]

    def validate(self) -> None:
        train_count = int(EXPECTED_SPLITS["train"]["tracks"])
        tag_count = len(self.vocabulary)
        input_dimension = self.train_clap.shape[1] if self.train_clap.ndim == 2 else 0
        expected_state = {
            "hidden.0.weight": (HIDDEN_DIMENSION, input_dimension),
            "hidden.0.bias": (HIDDEN_DIMENSION,),
            "hidden.2.weight": (HIDDEN_DIMENSION,),
            "hidden.2.bias": (HIDDEN_DIMENSION,),
            "projection.weight": (LATENT_DIMENSION, HIDDEN_DIMENSION),
            "projection.bias": (LATENT_DIMENSION,),
            "classifier.weight": (tag_count, LATENT_DIMENSION),
            "classifier.bias": (tag_count,),
        }
        if (
            self.train_track_ids.shape != (train_count,)
            or self.train_clap.ndim != 2
            or self.train_clap.shape[0] != train_count
            or self.train_targets.shape != (train_count, tag_count)
            or self.idf.shape != (tag_count,)
            or len(np.unique(self.train_track_ids)) != train_count
            or tuple(sorted(self.vocabulary)) != self.vocabulary
            or len(set(self.vocabulary)) != tag_count
            or not tag_count
            or set(self.mlp_state) != set(expected_state)
            or any(
                np.asarray(self.mlp_state[name]).shape != shape
                for name, shape in expected_state.items()
            )
        ):
            raise V3FreshCandidateError("candidate model shape or identity drift")
        for values in (
            self.train_clap,
            self.train_targets,
            self.idf,
            *self.mlp_state.values(),
        ):
            if not np.all(np.isfinite(values)):
                raise V3FreshCandidateError("candidate model has non-finite values")
        if (
            not np.all((self.train_targets == 0.0) | (self.train_targets == 1.0))
            or np.any(np.sum(self.train_targets, axis=1) <= 0.0)
            or np.any(self.idf <= 0.0)
        ):
            raise V3FreshCandidateError("candidate model labels are invalid")


def _payload_sha256(document: Mapping[str, object]) -> str:
    payload = dict(document)
    payload.pop("payload_sha256", None)
    return stable_json_sha256(payload)


def _protocol_entries(
    protocol: Mapping[str, object],
    split: str,
) -> Tuple[Mapping[str, object], ...]:
    entries = protocol.get("tracks")
    if not isinstance(entries, list) or split not in EXPECTED_SPLITS:
        raise V3FreshCandidateError("fresh protocol track plan drift")
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
        raise V3FreshCandidateError(f"{split} protocol split drift")
    return selected


def _validate_protocol(path: Path) -> Mapping[str, object]:
    candidate = Path(path)
    if sha256_path(candidate) != PROTOCOL_FILE_SHA256:
        raise V3FreshCandidateError("fresh protocol file hash drift")
    protocol = load_fresh_protocol(candidate)
    if (
        protocol.get("artifact_kind") != PROTOCOL_KIND
        or protocol.get("payload_sha256") != PROTOCOL_PAYLOAD_SHA256
        or protocol.get("selection_sha256") != EXPECTED_SELECTION_SHA256
    ):
        raise V3FreshCandidateError("fresh protocol binding drift")
    return protocol


def _validate_refinement(path: Path) -> Mapping[str, object]:
    document = _read_json(path, "refinement report")
    if (
        document.get("artifact_kind") != "v3_fresh_clap_refinement"
        or document.get("payload_sha256") != _payload_sha256(document)
        or document.get("payload_sha256") != REFINEMENT_PAYLOAD_SHA256
        or document.get("shadow_labels_accessed") is not False
        or document.get("development_gate", {}).get("passed") is not True
    ):
        raise V3FreshCandidateError("refinement report binding drift")
    return document


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
        raise V3FreshCandidateError("label split selection is invalid")
    allowed = tuple(
        entry
        for split in selected_splits
        for entry in _protocol_entries(protocol, split)
    )
    expected_artists = {
        int(entry["track_id"]): int(entry["artist_id"]) for entry in allowed
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
        raise V3FreshCandidateError("label source may not be a symlink")
    path = path.resolve(strict=True)
    labels: Dict[int, Tuple[str, ...]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        if next(reader, None) != list(LABEL_HEADER):
            raise V3FreshCandidateError("label source header drift")
        for line_number, row in enumerate(reader, 2):
            if len(row) < len(LABEL_HEADER):
                raise V3FreshCandidateError(
                    f"label source row {line_number} is short"
                )
            match = _ID_PATTERNS["track"].fullmatch(row[0])
            if match is None:
                raise V3FreshCandidateError(
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
                raise V3FreshCandidateError(
                    f"label source row {line_number} differs from protocol"
                )
            labels[track_id] = tags
    if set(labels) != set(expected_artists):
        raise V3FreshCandidateError("selected protocol labels are incomplete")
    if set(labels).intersection(excluded_ids):
        raise V3FreshCandidateError("labels outside selected splits were loaded")
    return labels


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
    positions = {tag: position for position, tag in enumerate(vocabulary)}
    targets = np.zeros((len(train_entries), len(vocabulary)), dtype=np.float32)
    for row, entry in enumerate(train_entries):
        for tag in labels[int(entry["track_id"])]:
            targets[row, positions[tag]] = 1.0
    if np.any(np.sum(targets, axis=0) <= 0.0):
        raise V3FreshCandidateError("training vocabulary contains an empty tag")
    return vocabulary, targets


def _global_embeddings(
    reader: FullTrackStoreReader,
    track_ids: Sequence[int],
) -> np.ndarray:
    positions = {track_id: row for row, track_id in enumerate(reader.track_ids)}
    try:
        rows = [positions[int(track_id)] for track_id in track_ids]
    except KeyError as exc:
        raise V3FreshCandidateError(
            f"CLAP store is missing track {exc.args[0]}"
        ) from exc
    values = np.asarray(reader.global_embeddings[rows], dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if (
        values.shape != (len(track_ids), reader.binding.embedding_dim)
        or not np.all(np.isfinite(values))
        or np.any(norms <= 1e-12)
    ):
        raise V3FreshCandidateError("global embedding shape or value drift")
    return values / norms


def _partition_fold(artist_id: int) -> int:
    return (
        int(
            stable_json_sha256(
                {"seed": FOLD_SEED, "artist_id": int(artist_id)}
            )[:16],
            16,
        )
        % 5
    )


def build_partition_data(
    partition_entries: Sequence[Mapping[str, object]],
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
            sorted(
                partition_entries,
                key=lambda entry: positions[int(entry["track_id"])],
            )
        )
    except KeyError as exc:
        raise V3FreshCandidateError(
            f"CLAP store is missing track {exc.args[0]}"
        ) from exc
    track_ids = np.asarray(
        [int(entry["track_id"]) for entry in ordered], dtype=np.int64
    )
    artist_ids = np.asarray(
        [int(entry["artist_id"]) for entry in ordered], dtype=np.int64
    )
    if set(labels) != set(int(value) for value in track_ids):
        raise V3FreshCandidateError("evaluation labels differ from partition")
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
        order = eligible[np.lexsort((track_ids[eligible], -global_scores))]
        if len(order) < CANDIDATE_POOL:
            raise V3FreshCandidateError(
                "evaluation candidate universe is too small"
            )
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


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if (
        array.ndim != 2
        or not len(array)
        or not np.all(np.isfinite(array))
        or np.any(norms <= 1e-12)
    ):
        raise V3FreshCandidateError("profile values are invalid")
    return array / norms


def _new_mlp(input_dimension: int, tag_count: int):
    try:
        from torch import nn
        from torch.nn import functional
    except ImportError as exc:
        raise V3FreshCandidateError("torch is required for the fresh candidate") from exc

    class MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hidden = nn.Sequential(
                nn.Linear(input_dimension, HIDDEN_DIMENSION),
                nn.GELU(),
                nn.LayerNorm(HIDDEN_DIMENSION),
                nn.Dropout(0.10),
            )
            self.projection = nn.Linear(HIDDEN_DIMENSION, LATENT_DIMENSION)
            self.classifier = nn.Linear(LATENT_DIMENSION, tag_count)

        def forward(self, values):
            hidden = self.hidden(values)
            latent = functional.normalize(self.projection(hidden), dim=1)
            return latent, self.classifier(latent)

    return MLP()


def train_mlp(
    train_inputs: np.ndarray,
    targets: np.ndarray,
) -> Tuple[Mapping[str, np.ndarray], Sequence[Mapping[str, object]]]:
    try:
        import torch
        from torch.nn import functional
    except ImportError as exc:
        raise V3FreshCandidateError("torch is required for the fresh candidate") from exc
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _new_mlp(train_inputs.shape[1], targets.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    train_tensor = torch.as_tensor(
        train_inputs,
        dtype=torch.float32,
        device=device,
    )
    target_tensor = torch.as_tensor(
        targets,
        dtype=torch.float32,
        device=device,
    )
    positive = torch.sum(target_tensor, dim=0)
    positive_weight = torch.clamp(
        (len(targets) - positive) / torch.clamp(positive, min=1.0),
        max=20.0,
    )
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        _, logits = model(train_tensor)
        loss = functional.binary_cross_entropy_with_logits(
            logits,
            target_tensor,
            pos_weight=positive_weight,
        )
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 25 == 0:
            history.append({"epoch": epoch, "loss": float(loss.detach().cpu())})
    return (
        {
            key: value.detach().cpu().numpy()
            for key, value in model.state_dict().items()
        },
        history,
    )


def mlp_tag_profiles(
    state: Mapping[str, np.ndarray],
    values: np.ndarray,
    idf: np.ndarray,
) -> np.ndarray:
    try:
        import torch
    except ImportError as exc:
        raise V3FreshCandidateError("torch is required for the fresh candidate") from exc
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _new_mlp(values.shape[1], len(idf)).to(device)
    model.load_state_dict(
        {
            key: torch.as_tensor(array, device=device)
            for key, array in state.items()
        }
    )
    model.eval()
    with torch.no_grad():
        _, logits = model(
            torch.as_tensor(values, dtype=torch.float32, device=device)
        )
    return _normalize_rows(torch.sigmoid(logits).cpu().numpy() * idf)


def knn_tag_profiles(
    train_inputs: np.ndarray,
    targets: np.ndarray,
    values: np.ndarray,
    idf: np.ndarray,
) -> np.ndarray:
    train = _normalize_rows(train_inputs)
    queries = _normalize_rows(values)
    similarities = queries @ train.T
    indices = np.argpartition(
        -similarities,
        K_NEIGHBORS - 1,
        axis=1,
    )[:, :K_NEIGHBORS]
    scores = np.take_along_axis(similarities, indices, axis=1)
    order = np.argsort(-scores, axis=1, kind="stable")
    indices = np.take_along_axis(indices, order, axis=1)
    scores = np.take_along_axis(scores, order, axis=1)
    weights = np.exp((scores - scores[:, :1]) / KNN_TEMPERATURE)
    weights /= np.sum(weights, axis=1, keepdims=True)
    predictions = np.einsum(
        "nk,nkt->nt",
        weights,
        targets[indices],
        optimize=True,
    )
    return _normalize_rows(
        np.power(predictions, KNN_PREDICTION_POWER)
        * np.power(idf, KNN_IDF_POWER)
    )


def candidate_profiles(
    model: FreshCandidateModel,
    values: np.ndarray,
) -> np.ndarray:
    model.validate()
    knn = knn_tag_profiles(
        model.train_clap,
        model.train_targets,
        values,
        model.idf,
    )
    mlp = mlp_tag_profiles(model.mlp_state, values, model.idf)
    return np.concatenate(
        (
            np.sqrt(KNN_SHARE) * knn,
            np.sqrt(MLP_SHARE) * mlp,
        ),
        axis=1,
    )


def candidate_scores(
    data: PartitionData,
    profiles: np.ndarray,
) -> np.ndarray:
    data.validate()
    values = np.asarray(profiles, dtype=np.float64)
    if (
        values.ndim != 2
        or len(values) != len(data.track_ids)
        or not np.all(np.isfinite(values))
    ):
        raise V3FreshCandidateError("candidate score inputs are invalid")
    scores = np.empty_like(data.baseline_scores, dtype=np.float64)
    for query_position in range(len(data.track_ids)):
        pool = data.pools[query_position]
        semantic = values[pool] @ values[query_position]
        scores[query_position] = (
            (1.0 - SEMANTIC_BLEND)
            * _zscore_columns(
                data.baseline_scores[query_position].astype(np.float64)[:, None]
            )[:, 0]
            + SEMANTIC_BLEND * _zscore_columns(semantic[:, None])[:, 0]
        )
    return scores


def _model_arrays(model: FreshCandidateModel) -> Mapping[str, np.ndarray]:
    model.validate()
    arrays: Dict[str, np.ndarray] = {
        "vocabulary": np.asarray(model.vocabulary, dtype=np.str_),
        "train_track_ids": model.train_track_ids.astype(np.int64),
        "train_clap": model.train_clap.astype(np.float32),
        "train_targets": model.train_targets.astype(np.uint8),
        "idf": model.idf.astype(np.float64),
    }
    arrays.update(
        {
            f"mlp__{key.replace('.', '__')}": np.asarray(value)
            for key, value in model.mlp_state.items()
        }
    )
    return arrays


def load_candidate_model(path: Path) -> FreshCandidateModel:
    with np.load(Path(path), allow_pickle=False) as archive:
        vocabulary = tuple(str(value) for value in archive["vocabulary"])
        state = {
            key.removeprefix("mlp__").replace("__", "."): np.asarray(
                archive[key],
                dtype=np.float32,
            )
            for key in archive.files
            if key.startswith("mlp__")
        }
        model = FreshCandidateModel(
            vocabulary=vocabulary,
            train_track_ids=np.asarray(
                archive["train_track_ids"], dtype=np.int64
            ),
            train_clap=np.asarray(archive["train_clap"], dtype=np.float64),
            train_targets=np.asarray(
                archive["train_targets"], dtype=np.float64
            ),
            idf=np.asarray(archive["idf"], dtype=np.float64),
            mlp_state=state,
        )
    model.validate()
    return model


def development_gate(evaluation: Mapping[str, object]) -> Mapping[str, object]:
    relative = evaluation["relative_delta"]
    checks = {
        "primary_relative_gain": (
            float(relative[PRIMARY_METRIC]) >= MIN_DEVELOPMENT_RECALL_GAIN
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
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "decision": (
            "freeze_for_one_time_fresh_shadow_audit"
            if all(checks.values())
            else "continue_development_without_fresh_shadow_access"
        ),
    }


def shadow_gate(evaluation: Mapping[str, object]) -> Mapping[str, object]:
    relative = evaluation["relative_delta"]
    checks = {
        "primary_relative_gain": (
            float(relative[PRIMARY_METRIC]) >= MIN_SHADOW_RECALL_GAIN
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
    return {
        "checks": checks,
        "automated_passed": all(checks.values()),
        "human_pilot_required": True,
        "promotion_allowed": False,
    }


def _open_clap_store(path: Path) -> FullTrackStoreReader:
    return _open_bound_store(
        Path(path),
        expected_manifest_file_sha256=CLAP_MANIFEST_FILE_SHA256,
        expected_binding=EXPECTED_CLAP_BINDING,
    )


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
            raise V3FreshCandidateError(
                f"production candidate does not reproduce refinement {metric}"
            )


def build_development_candidate(
    *,
    metadata_root: Path,
    protocol_path: Path,
    refinement_report_path: Path,
    clap_store: Path,
    model_output: Path,
    metadata_output: Path,
    report_output: Path,
) -> Mapping[str, object]:
    outputs = (Path(model_output), Path(metadata_output), Path(report_output))
    if any(path.exists() for path in outputs):
        raise V3FreshCandidateError(
            "candidate output already exists; refusing overwrite"
        )
    protocol = _validate_protocol(protocol_path)
    refinement = _validate_refinement(refinement_report_path)
    train_entries = _protocol_entries(protocol, "train")
    development_entries = _protocol_entries(protocol, "development")
    labels = load_protocol_tags(
        metadata_root,
        protocol,
        ("train", "development"),
    )
    reader = _open_clap_store(clap_store)
    try:
        positions = {
            track_id: position for position, track_id in enumerate(reader.track_ids)
        }
        train_entries = tuple(
            sorted(
                train_entries,
                key=lambda entry: positions[int(entry["track_id"])],
            )
        )
        vocabulary, targets = build_label_targets(train_entries, labels)
        train_ids = np.asarray(
            [int(entry["track_id"]) for entry in train_entries],
            dtype=np.int64,
        )
        train_clap = _global_embeddings(reader, train_ids)
        idf = np.log(
            (len(targets) + 1.0) / (np.sum(targets, axis=0) + 1.0)
        ) + 1.0
        state, history = train_mlp(train_clap, targets)
        model = FreshCandidateModel(
            vocabulary=vocabulary,
            train_track_ids=train_ids,
            train_clap=train_clap,
            train_targets=targets,
            idf=idf,
            mlp_state=state,
        )
        model.validate()
        development_labels = {
            int(entry["track_id"]): labels[int(entry["track_id"])]
            for entry in development_entries
        }
        data = build_partition_data(
            development_entries,
            development_labels,
            reader,
            split="development",
        )
        development_clap = _global_embeddings(reader, data.track_ids)
        evaluation = _evaluate_scores(
            data,
            candidate_scores(
                data,
                candidate_profiles(model, development_clap),
            ),
        )
        _verify_expected_development(evaluation)
        gate = development_gate(evaluation)
        if not gate["passed"]:
            raise V3FreshCandidateError(
                "reproduced candidate failed development gate"
            )
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
            "refinement_payload_sha256": REFINEMENT_PAYLOAD_SHA256,
            "refinement_file_sha256": sha256_path(Path(refinement_report_path)),
            "opened_label_splits": ["train", "development"],
            "shadow_labels_accessed": False,
            "shadow_evaluation_accessed": False,
            "development_consumed_for_selection": True,
            "frozen_method": {
                "seed": SEED,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "hidden_dimension": HIDDEN_DIMENSION,
                "latent_dimension": LATENT_DIMENSION,
                "neighbors": K_NEIGHBORS,
                "neighbor_temperature": KNN_TEMPERATURE,
                "knn_prediction_power": KNN_PREDICTION_POWER,
                "knn_idf_power": KNN_IDF_POWER,
                "knn_profile_share": KNN_SHARE,
                "mlp_tag_profile_share": MLP_SHARE,
                "baseline_blend": 1.0 - SEMANTIC_BLEND,
                "semantic_blend": SEMANTIC_BLEND,
            },
            "train_tracks": len(train_entries),
            "development_tracks": len(development_entries),
            "tag_count": len(vocabulary),
            "training_history": list(history),
            "evaluation": evaluation,
            "development_gate": gate,
            "model_npz_sha256": sha256_path(Path(model_output)),
            "clap_manifest_file_sha256": sha256_path(
                Path(clap_store) / "store.sealed.json"
            ),
            "label_source_sha256": sha256_path(_label_source(metadata_root)),
            "promotion_allowed": False,
        }
        if report["refinement_payload_sha256"] != refinement["payload_sha256"]:
            raise V3FreshCandidateError("refinement payload changed during build")
        report["payload_sha256"] = stable_json_sha256(report)
        _write_json_exclusive(Path(report_output), report)
        metadata: Dict[str, object] = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "artifact_kind": MODEL_KIND,
            "source_fingerprint": SOURCE_FINGERPRINT,
            "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
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
        reader.close()


def _read_json(path: Path, label: str) -> Mapping[str, object]:
    candidate = Path(path)
    if candidate.is_symlink() or not candidate.is_file():
        raise V3FreshCandidateError(f"{label} must be a concrete file")
    try:
        document = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V3FreshCandidateError(f"{label} is invalid: {exc}") from exc
    if not isinstance(document, dict):
        raise V3FreshCandidateError(f"{label} must contain a JSON object")
    return document


def freeze_candidate(
    *,
    protocol_path: Path,
    refinement_report_path: Path,
    clap_store: Path,
    model_path: Path,
    metadata_path: Path,
    development_report_path: Path,
    output: Path,
) -> Mapping[str, object]:
    if Path(output).exists():
        raise V3FreshCandidateError("freeze output already exists")
    protocol = _validate_protocol(protocol_path)
    refinement = _validate_refinement(refinement_report_path)
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
        or report.get("refinement_payload_sha256") != refinement["payload_sha256"]
    ):
        raise V3FreshCandidateError("candidate evidence is not eligible to freeze")
    load_candidate_model(model_path)
    document: Dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": FREEZE_KIND,
        "source_fingerprint": SOURCE_FINGERPRINT,
        "protocol_file_sha256": sha256_path(Path(protocol_path)),
        "protocol_payload_sha256": protocol["payload_sha256"],
        "protocol_selection_sha256": protocol["selection_sha256"],
        "refinement_file_sha256": sha256_path(Path(refinement_report_path)),
        "refinement_payload_sha256": refinement["payload_sha256"],
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
            "minimum_recall_relative_gain": MIN_SHADOW_RECALL_GAIN,
            "paired_recall_ci_must_exclude_zero": True,
            "minimum_positive_recall_folds": MIN_POSITIVE_FOLDS,
            "maximum_fold_recall_regression": MAX_FOLD_RECALL_REGRESSION,
            "maximum_mrr_ndcg_regression": MAX_SAFETY_REGRESSION,
        },
        "shadow_labels_accessed": False,
        "promotion_allowed": False,
    }
    document["payload_sha256"] = stable_json_sha256(document)
    _write_json_exclusive(Path(output), document)
    return document


def _verify_freeze(
    freeze_path: Path,
    *,
    protocol_path: Path,
    refinement_report_path: Path,
    clap_store: Path,
    model_path: Path,
    metadata_path: Path,
    development_report_path: Path,
) -> Mapping[str, object]:
    freeze = _read_json(freeze_path, "shadow freeze")
    expected = {
        "artifact_kind": FREEZE_KIND,
        "payload_sha256": _payload_sha256(freeze),
        "protocol_file_sha256": sha256_path(Path(protocol_path)),
        "protocol_payload_sha256": PROTOCOL_PAYLOAD_SHA256,
        "protocol_selection_sha256": EXPECTED_SELECTION_SHA256,
        "refinement_file_sha256": sha256_path(Path(refinement_report_path)),
        "refinement_payload_sha256": REFINEMENT_PAYLOAD_SHA256,
        "candidate_model_file_sha256": sha256_path(Path(model_path)),
        "candidate_metadata_file_sha256": sha256_path(Path(metadata_path)),
        "development_report_file_sha256": sha256_path(
            Path(development_report_path)
        ),
        "clap_manifest_file_sha256": sha256_path(
            Path(clap_store) / "store.sealed.json"
        ),
        "shadow_labels_accessed": False,
    }
    drift = {
        key: (value, freeze.get(key))
        for key, value in expected.items()
        if freeze.get(key) != value
    }
    if drift:
        raise V3FreshCandidateError(f"shadow freeze binding drift: {drift}")
    return freeze


def audit_shadow(
    *,
    metadata_root: Path,
    protocol_path: Path,
    refinement_report_path: Path,
    clap_store: Path,
    model_path: Path,
    metadata_path: Path,
    development_report_path: Path,
    freeze_path: Path,
    output: Path,
    audit_state_path: Path,
) -> Mapping[str, object]:
    if Path(output).exists():
        raise V3FreshCandidateError("shadow audit output already exists")
    if Path(audit_state_path).exists():
        raise V3FreshCandidateError(
            "shadow audit state already exists; refusing reopen"
        )
    protocol = _validate_protocol(protocol_path)
    _validate_refinement(refinement_report_path)
    metadata = _read_json(metadata_path, "candidate metadata")
    development = _read_json(development_report_path, "development report")
    if (
        metadata.get("payload_sha256") != _payload_sha256(metadata)
        or development.get("payload_sha256") != _payload_sha256(development)
        or not development.get("development_gate", {}).get("passed")
    ):
        raise V3FreshCandidateError("candidate evidence failed audit validation")
    freeze = _verify_freeze(
        freeze_path,
        protocol_path=protocol_path,
        refinement_report_path=refinement_report_path,
        clap_store=clap_store,
        model_path=model_path,
        metadata_path=metadata_path,
        development_report_path=development_report_path,
    )
    model = load_candidate_model(model_path)
    reader = _open_clap_store(clap_store)
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
        labels = load_protocol_tags(
            metadata_root,
            protocol,
            ("shadow",),
        )
        data = build_partition_data(
            shadow_entries,
            labels,
            reader,
            split="shadow",
        )
        shadow_clap = _global_embeddings(reader, data.track_ids)
        evaluation = _evaluate_scores(
            data,
            candidate_scores(
                data,
                candidate_profiles(model, shadow_clap),
            ),
        )
        gate = shadow_gate(evaluation)
        report: Dict[str, object] = {
            "schema_version": 1,
            "artifact_kind": SHADOW_REPORT_KIND,
            "evidence_scope": EVIDENCE_SCOPE,
            "evidence_status": "independent_fresh_shadow_audit",
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
                "report_file_sha256": sha256_path(Path(output)),
                "report_payload_sha256": report["payload_sha256"],
            }
        )
        completed.pop("payload_sha256", None)
        completed["payload_sha256"] = stable_json_sha256(completed)
        _replace_json(Path(audit_state_path), completed)
        return report
    finally:
        reader.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build-development")
    for name in ("metadata-root", "protocol", "clap-store"):
        build.add_argument(f"--{name}", required=True)
    build.add_argument("--model-output", required=True)
    build.add_argument("--refinement-report", required=True)
    build.add_argument("--metadata-output", required=True)
    build.add_argument("--report-output", required=True)

    freeze = subparsers.add_parser("freeze-shadow")
    for name in (
        "protocol",
        "refinement-report",
        "clap-store",
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
        "refinement-report",
        "clap-store",
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
            result = build_development_candidate(
                metadata_root=Path(args.metadata_root),
                protocol_path=Path(args.protocol),
                refinement_report_path=Path(args.refinement_report),
                clap_store=Path(args.clap_store),
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
            result = audit_shadow(
                metadata_root=Path(args.metadata_root),
                protocol_path=Path(args.protocol),
                refinement_report_path=Path(args.refinement_report),
                clap_store=Path(args.clap_store),
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
    except (OSError, ValueError, V3FreshCandidateError) as exc:
        raise SystemExit(f"fresh V3 candidate failed: {exc}") from exc
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
