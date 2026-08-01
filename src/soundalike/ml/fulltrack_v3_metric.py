"""Train the frozen nonlinear dual-representation V3 development candidate."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .fulltrack_eval import (
    METRICS,
    _method_ranking,
    _paired_bootstrap_delta,
    _query_metrics,
)
from .fulltrack_store import (
    FullTrackStoreError,
    FullTrackStoreReader,
    sha256_path,
    stable_json_sha256,
)
from .fulltrack_v3 import (
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    CLAP_MANIFEST_FILE_SHA256,
    EXPECTED_CLAP_BINDING,
    SOURCE_FINGERPRINT,
    _open_bound_store,
)
from .fulltrack_v3_protocol import (
    EXPECTED_SELECTION_SHA256,
    PROTOCOL_KIND,
    load_protocol,
)
from .fulltrack_v3_ranker import (
    _write_json_exclusive,
    _write_npz_exclusive,
    _zscore_columns,
)
from .fulltrack_v3_semantic import (
    MODEL_KIND,
    SCALE_MUSICFM_CONFIG_SHA256,
    SCALE_PROTOCOL_PAYLOAD_SHA256,
    DevelopmentData,
    V3SemanticError,
    _global_embeddings,
    _protocol_entries,
    _validate_musicfm_store,
    build_development_data,
    build_label_targets,
    development_gate,
    load_train_development_tags,
)
from .fulltrack_v3_text import clap_text_profiles, load_text_artifact
from .jamendo_fulltrack import EVIDENCE_SCOPE


MODEL_SCHEMA_VERSION = 1
METRIC_MODEL_KIND = "v3_dual_metric_tag_head"
REPORT_SCHEMA_VERSION = 1
REPORT_KIND = "v3_dual_metric_knn_development_report"
MUSICFM_MODEL_SHA256 = (
    "68392eee13d34c2941b3761934abb6b1e67b2e9df498695bda2ea5c1087d4b96"
)
SEED = 20260807
HIDDEN_DIMENSION = 384
LATENT_DIMENSION = 128
EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
TRIPLET_MARGIN = 0.10
TRIPLET_WEIGHT = 0.25
POSITIVES_PER_QUERY = 2
NEGATIVES_PER_QUERY = 2
K_NEIGHBORS = 16
KNN_TEMPERATURE = 0.05
METRIC_TAG_SHARE = 0.25
LEARNED_METRIC_SHARE = 0.75
METRIC_BLEND = 0.20
KNN_BLEND = 0.40
GATE_QUANTILE = 0.40


class V3MetricError(RuntimeError):
    """Invalid, leaky, non-reproducible, or non-finite metric-head run."""


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not len(array) or not np.all(np.isfinite(array)):
        raise V3MetricError("values must be a finite non-empty matrix")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise V3MetricError("values contain a zero row")
    return array / norms


def transform_inputs(
    train: np.ndarray,
    values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_array = np.asarray(train, dtype=np.float64)
    values_array = np.asarray(values, dtype=np.float64)
    if (
        train_array.ndim != 2
        or values_array.ndim != 2
        or train_array.shape[1:] != values_array.shape[1:]
        or not len(train_array)
        or not len(values_array)
        or not np.all(np.isfinite(train_array))
        or not np.all(np.isfinite(values_array))
    ):
        raise V3MetricError("metric-head input matrices are invalid")
    mean = np.mean(train_array, axis=0)
    standard_deviation = np.std(train_array, axis=0)
    scale = np.where(standard_deviation > 1e-6, standard_deviation, 1.0)
    return (
        _normalize_rows((train_array - mean) / scale),
        _normalize_rows((values_array - mean) / scale),
        mean,
        scale,
    )


def mine_training_triplets(
    clap_inputs: np.ndarray,
    targets: np.ndarray,
    artist_ids: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Mapping[str, object]]:
    inputs = _normalize_rows(clap_inputs)
    labels = np.asarray(targets, dtype=np.float64)
    artists = np.asarray(artist_ids, dtype=np.int64)
    if (
        labels.ndim != 2
        or labels.shape[0] != len(inputs)
        or artists.shape != (len(inputs),)
        or not np.all((labels == 0.0) | (labels == 1.0))
    ):
        raise V3MetricError("triplet mining inputs are invalid")
    tag_counts = np.sum(labels, axis=1)
    queries = []
    positives = []
    negatives = []
    eligible_queries = 0
    block_size = 256
    for start in range(0, len(inputs), block_size):
        stop = min(start + block_size, len(inputs))
        similarity = inputs[start:stop] @ inputs.T
        shared = labels[start:stop] @ labels.T
        union = tag_counts[start:stop, None] + tag_counts[None, :] - shared
        for local, query in enumerate(range(start, stop)):
            cross_artist = artists != artists[query]
            positive_ids = np.flatnonzero(
                cross_artist
                & (shared[local] >= 2.0)
                & (shared[local] / np.maximum(union[local], 1.0) >= 0.25)
            )
            negative_ids = np.flatnonzero(
                cross_artist & (shared[local] == 0.0)
            )
            if not len(positive_ids) or not len(negative_ids):
                continue
            eligible_queries += 1
            positive_ids = positive_ids[
                np.argsort(
                    -similarity[local, positive_ids],
                    kind="stable",
                )
            ][:POSITIVES_PER_QUERY]
            negative_ids = negative_ids[
                np.argsort(
                    -similarity[local, negative_ids],
                    kind="stable",
                )
            ][:NEGATIVES_PER_QUERY]
            for positive in positive_ids:
                for negative in negative_ids:
                    queries.append(query)
                    positives.append(int(positive))
                    negatives.append(int(negative))
    if not queries:
        raise V3MetricError("triplet mining found no eligible queries")
    query_array = np.asarray(queries, dtype=np.int64)
    positive_array = np.asarray(positives, dtype=np.int64)
    negative_array = np.asarray(negatives, dtype=np.int64)
    if (
        np.any(artists[query_array] == artists[positive_array])
        or np.any(artists[query_array] == artists[negative_array])
        or np.any(
            np.sum(labels[query_array] * labels[negative_array], axis=1) != 0.0
        )
    ):
        raise V3MetricError("triplet mining leaked an artist or false negative")
    return (
        query_array,
        positive_array,
        negative_array,
        {
            "eligible_queries": eligible_queries,
            "triplets": len(query_array),
            "same_artist_pairs": 0,
            "shared_tag_negative_pairs": 0,
        },
    )


def weighted_knn_profiles(
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    values: np.ndarray,
    idf: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Mapping[str, object]]:
    train = _normalize_rows(train_inputs)
    queries = _normalize_rows(values)
    targets = np.asarray(train_targets, dtype=np.float64)
    idf_values = np.asarray(idf, dtype=np.float64)
    if (
        targets.ndim != 2
        or targets.shape[0] != len(train)
        or idf_values.shape != (targets.shape[1],)
        or len(train) < K_NEIGHBORS
        or np.any(idf_values <= 0.0)
    ):
        raise V3MetricError("k-neighbor profile inputs are invalid")
    similarity = queries @ train.T
    indices = np.argpartition(-similarity, K_NEIGHBORS - 1, axis=1)[
        :, :K_NEIGHBORS
    ]
    scores = np.take_along_axis(similarity, indices, axis=1)
    order = np.argsort(-scores, axis=1, kind="stable")
    indices = np.take_along_axis(indices, order, axis=1)
    scores = np.take_along_axis(scores, order, axis=1)
    shifted = (scores - scores[:, :1]) / KNN_TEMPERATURE
    weights = np.exp(shifted)
    weights /= np.sum(weights, axis=1, keepdims=True)
    predictions = np.einsum(
        "nk,nkt->nt",
        weights,
        targets[indices],
        optimize=True,
    )
    profiles = _normalize_rows(predictions * idf_values)
    confidence = np.max(predictions, axis=1)
    return (
        profiles,
        confidence,
        {
            "neighbors": K_NEIGHBORS,
            "temperature": KNN_TEMPERATURE,
            "mean_top1_similarity": float(np.mean(scores[:, 0])),
            "mean_effective_neighbors": float(
                np.mean(
                    np.exp(
                        -np.sum(
                            weights * np.log(np.maximum(weights, 1e-12)),
                            axis=1,
                        )
                    )
                )
            ),
        },
    )


def gated_scores(
    baseline_scores: np.ndarray,
    metric_semantic_scores: np.ndarray,
    knn_semantic_scores: np.ndarray,
    confidence: np.ndarray,
    *,
    threshold: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    baseline = np.asarray(baseline_scores, dtype=np.float64)
    metric = np.asarray(metric_semantic_scores, dtype=np.float64)
    knn = np.asarray(knn_semantic_scores, dtype=np.float64)
    confidence_values = np.asarray(confidence, dtype=np.float64)
    if (
        baseline.ndim != 2
        or metric.shape != baseline.shape
        or knn.shape != baseline.shape
        or confidence_values.shape != (len(baseline),)
        or not np.all(np.isfinite(baseline))
        or not np.all(np.isfinite(metric))
        or not np.all(np.isfinite(knn))
        or not np.all(np.isfinite(confidence_values))
    ):
        raise V3MetricError("gated score inputs are invalid")
    if threshold is None:
        threshold = float(np.quantile(confidence_values, GATE_QUANTILE))
    if not np.isfinite(threshold):
        raise V3MetricError("gate threshold is invalid")

    def normalize_rows(values: np.ndarray) -> np.ndarray:
        mean = np.mean(values, axis=1, keepdims=True)
        standard_deviation = np.std(values, axis=1, keepdims=True)
        return (values - mean) / np.where(
            standard_deviation > 1e-8,
            standard_deviation,
            1.0,
        )

    base = normalize_rows(baseline)
    metric_candidate = (
        (1.0 - METRIC_BLEND) * base
        + METRIC_BLEND * normalize_rows(metric)
    )
    knn_candidate = (
        (1.0 - KNN_BLEND) * base
        + KNN_BLEND * normalize_rows(knn)
    )
    applied = confidence_values >= threshold
    return (
        np.where(applied[:, None], knn_candidate, metric_candidate),
        applied,
        float(threshold),
    )


def _evaluate_scores(
    data: DevelopmentData,
    candidate_scores: np.ndarray,
) -> Mapping[str, object]:
    data.validate()
    candidate_matrix = np.asarray(candidate_scores, dtype=np.float64)
    if (
        candidate_matrix.shape != data.baseline_scores.shape
        or not np.all(np.isfinite(candidate_matrix))
    ):
        raise V3MetricError("candidate score matrix is invalid")
    baseline_values: Dict[str, list[float]] = {metric: [] for metric in METRICS}
    candidate_values: Dict[str, list[float]] = {metric: [] for metric in METRICS}
    folds = []
    for query_position in range(len(data.track_ids)):
        relevant = {
            int(data.track_ids[position]): float(grade)
            for position, grade in enumerate(data.relevance[query_position])
            if grade > 0.0
        }
        if not relevant:
            continue
        pool = data.pools[query_position]
        global_order = data.global_orders[
            query_position, : int(data.global_lengths[query_position])
        ]
        orders = {
            "baseline": _method_ranking(
                data.baseline_scores[query_position],
                pool,
                global_order,
            ),
            "candidate": _method_ranking(
                candidate_matrix[query_position],
                pool,
                global_order,
            ),
        }
        for method, destination in (
            ("baseline", baseline_values),
            ("candidate", candidate_values),
        ):
            metrics = _query_metrics(
                [
                    int(data.track_ids[position])
                    for position in orders[method]
                ],
                relevant,
                recall_cutoff=10,
                ndcg_cutoff=10,
            )
            for metric in METRICS:
                destination[metric].append(float(getattr(metrics, metric)))
        folds.append(int(data.query_folds[query_position]))
    fold_array = np.asarray(folds, dtype=np.int8)
    if not len(fold_array) or set(fold_array) != set(range(5)):
        raise V3MetricError("metric evaluation folds are incomplete")
    baseline_mean = {
        metric: float(np.mean(baseline_values[metric])) for metric in METRICS
    }
    candidate_mean = {
        metric: float(np.mean(candidate_values[metric])) for metric in METRICS
    }
    positive_folds = {metric: 0 for metric in METRICS}
    worst_fold = {metric: float("inf") for metric in METRICS}
    fold_results = {}
    for fold_index in range(5):
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


def _train_metric_head(
    train_inputs: np.ndarray,
    development_inputs: np.ndarray,
    targets: np.ndarray,
    triplets: Tuple[np.ndarray, np.ndarray, np.ndarray],
    idf: np.ndarray,
) -> Tuple[np.ndarray, Mapping[str, np.ndarray], Sequence[Mapping[str, object]]]:
    try:
        import torch
        from torch import nn
        from torch.nn import functional as functional
    except ImportError as exc:
        raise V3MetricError("torch is required for the metric-head run") from exc

    class MetricTagHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hidden = nn.Sequential(
                nn.Linear(train_inputs.shape[1], HIDDEN_DIMENSION),
                nn.GELU(),
                nn.LayerNorm(HIDDEN_DIMENSION),
                nn.Dropout(0.10),
            )
            self.projection = nn.Linear(HIDDEN_DIMENSION, LATENT_DIMENSION)
            self.classifier = nn.Linear(LATENT_DIMENSION, targets.shape[1])

        def forward(self, values):
            hidden = self.hidden(values)
            latent = functional.normalize(self.projection(hidden), dim=1)
            return latent, self.classifier(latent)

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tensor = torch.as_tensor(
        train_inputs,
        dtype=torch.float32,
        device=device,
    )
    development_tensor = torch.as_tensor(
        development_inputs,
        dtype=torch.float32,
        device=device,
    )
    target_tensor = torch.as_tensor(targets, dtype=torch.float32, device=device)
    query = torch.as_tensor(triplets[0], device=device)
    positive = torch.as_tensor(triplets[1], device=device)
    negative = torch.as_tensor(triplets[2], device=device)
    positive_counts = torch.sum(target_tensor, dim=0)
    positive_weight = torch.clamp(
        (len(targets) - positive_counts)
        / torch.clamp(positive_counts, min=1.0),
        max=20.0,
    )
    model = MetricTagHead().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        latent, logits = model(train_tensor)
        classification_loss = functional.binary_cross_entropy_with_logits(
            logits,
            target_tensor,
            pos_weight=positive_weight,
        )
        positive_similarity = torch.sum(
            latent[query] * latent[positive],
            dim=1,
        )
        negative_similarity = torch.sum(
            latent[query] * latent[negative],
            dim=1,
        )
        triplet_loss = torch.mean(
            functional.relu(
                TRIPLET_MARGIN
                - positive_similarity
                + negative_similarity
            )
        )
        loss = classification_loss + TRIPLET_WEIGHT * triplet_loss
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 25 == 0 or epoch == EPOCHS:
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(loss.detach().cpu()),
                    "classification_loss": float(
                        classification_loss.detach().cpu()
                    ),
                    "triplet_loss": float(triplet_loss.detach().cpu()),
                    "triplet_accuracy": float(
                        torch.mean(
                            (positive_similarity > negative_similarity).float()
                        )
                        .detach()
                        .cpu()
                    ),
                }
            )
    model.eval()
    with torch.no_grad():
        latent, logits = model(development_tensor)
    latent_profile = latent.cpu().numpy().astype(np.float64)
    tag_probabilities = torch.sigmoid(logits).cpu().numpy()
    tag_profile = _normalize_rows(tag_probabilities * idf)
    profile = np.concatenate(
        (
            np.sqrt(METRIC_TAG_SHARE) * tag_profile,
            np.sqrt(1.0 - METRIC_TAG_SHARE) * latent_profile,
        ),
        axis=1,
    )
    state = {
        key.replace(".", "__"): value.detach().cpu().numpy()
        for key, value in model.state_dict().items()
    }
    state["device_was_cuda"] = np.asarray(
        [int(device.type == "cuda")],
        dtype=np.int8,
    )
    return profile, state, history


def train_dual_metric_candidate(
    *,
    metadata_root: Path,
    protocol_path: Path,
    clap_store: Path,
    musicfm_store: Path,
    clap_text_embeddings: Path,
    model_output: Path,
    metadata_output: Path,
    report_output: Path,
) -> Mapping[str, object]:
    outputs = (Path(model_output), Path(metadata_output), Path(report_output))
    if any(path.exists() for path in outputs):
        raise V3MetricError("metric-head output already exists; refusing overwrite")
    protocol = load_protocol(Path(protocol_path))
    if (
        protocol.get("artifact_kind") != PROTOCOL_KIND
        or protocol.get("payload_sha256") != SCALE_PROTOCOL_PAYLOAD_SHA256
        or protocol.get("selection_sha256") != EXPECTED_SELECTION_SHA256
    ):
        raise V3MetricError("scale protocol binding drift")
    train_entries = _protocol_entries(protocol, "train")
    development_entries = _protocol_entries(protocol, "development")
    labels = load_train_development_tags(Path(metadata_root), protocol)
    vocabulary, targets = build_label_targets(train_entries, labels)
    text_embeddings, text_prompts, text_evidence = load_text_artifact(
        Path(clap_text_embeddings),
        expected_vocabulary=vocabulary,
    )
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
    started = time.perf_counter()
    try:
        _validate_musicfm_store(music_reader, protocol)
        train_ids = [int(entry["track_id"]) for entry in train_entries]
        artist_ids = np.asarray(
            [int(entry["artist_id"]) for entry in train_entries],
            dtype=np.int64,
        )
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
        dual_train = np.concatenate((clap_train, music_train), axis=1)
        dual_development = np.concatenate(
            (clap_development, music_development),
            axis=1,
        )
        transformed_train, transformed_development, input_mean, input_scale = (
            transform_inputs(dual_train, dual_development)
        )
        clap_mining_inputs = transform_inputs(clap_train, clap_train)[0]
        query, positive, negative, mining = mine_training_triplets(
            clap_mining_inputs,
            targets,
            artist_ids,
        )
        idf = (
            np.log(
                (len(targets) + 1.0)
                / (np.sum(targets, axis=0) + 1.0)
            )
            + 1.0
        )
        learned_metric_profile, model_state, history = _train_metric_head(
            transformed_train,
            transformed_development,
            targets,
            (query, positive, negative),
            idf,
        )
        text_profile = clap_text_profiles(clap_development, text_embeddings)
        metric_profile = np.concatenate(
            (
                np.sqrt(LEARNED_METRIC_SHARE) * learned_metric_profile,
                np.sqrt(1.0 - LEARNED_METRIC_SHARE) * text_profile,
            ),
            axis=1,
        )
        knn_profile, confidence, knn_evidence = weighted_knn_profiles(
            transformed_train,
            targets,
            transformed_development,
            idf,
        )
        metric_semantic = np.asarray(
            [
                metric_profile[development_data.pools[position]]
                @ metric_profile[position]
                for position in range(len(development_data.track_ids))
            ]
        )
        knn_semantic = np.asarray(
            [
                knn_profile[development_data.pools[position]]
                @ knn_profile[position]
                for position in range(len(development_data.track_ids))
            ]
        )
        evaluable = np.any(development_data.relevance > 0.0, axis=1)
        threshold = float(np.quantile(confidence[evaluable], GATE_QUANTILE))
        candidate_scores, applied, threshold = gated_scores(
            development_data.baseline_scores,
            metric_semantic,
            knn_semantic,
            confidence,
            threshold=threshold,
        )
        evaluation = _evaluate_scores(development_data, candidate_scores)
        arrays: Dict[str, np.ndarray] = {
            "input_mean": input_mean.astype(np.float32),
            "input_scale": input_scale.astype(np.float32),
            "idf": idf.astype(np.float32),
            "vocabulary": np.asarray(vocabulary, dtype=np.str_),
            "gate_threshold": np.asarray([threshold], dtype=np.float64),
            "knn_train_inputs": transformed_train.astype(np.float32),
            "knn_train_targets": targets.astype(np.uint8),
            "knn_train_track_ids": np.asarray(train_ids, dtype=np.int64),
            "clap_text_embeddings": text_embeddings.astype(np.float32),
            "clap_text_prompts": np.asarray(text_prompts, dtype=np.str_),
        }
        arrays.update(
            {
                f"state__{key}": np.asarray(value)
                for key, value in model_state.items()
            }
        )
        _write_npz_exclusive(Path(model_output), arrays)
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
            "train_track_ids_sha256": stable_json_sha256(tuple(train_ids)),
            "development_tracks": len(development_entries),
            "tag_count": len(vocabulary),
            "frozen_method": {
                "input": "standardized_clap_musicfm_global_concatenation",
                "hard_negative_source": "clap_global_neighborhood",
                "seed": SEED,
                "hidden_dimension": HIDDEN_DIMENSION,
                "latent_dimension": LATENT_DIMENSION,
                "epochs": EPOCHS,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "triplet_margin": TRIPLET_MARGIN,
                "triplet_weight": TRIPLET_WEIGHT,
                "metric_tag_share": METRIC_TAG_SHARE,
                "learned_metric_share": LEARNED_METRIC_SHARE,
                "clap_text_share": 1.0 - LEARNED_METRIC_SHARE,
                "metric_blend": METRIC_BLEND,
                "knn_blend": KNN_BLEND,
                "gate_feature": "maximum_weighted_neighbor_tag_probability",
                "gate_direction": "ge",
                "gate_quantile": GATE_QUANTILE,
                "gate_threshold": threshold,
            },
            "triplet_mining": dict(mining),
            "knn": dict(knn_evidence),
            "clap_text": dict(text_evidence),
            "training_history": list(history),
            "gate_applied_queries": int(np.count_nonzero(applied & evaluable)),
            "gate_evaluable_queries": int(np.count_nonzero(evaluable)),
            "evaluation": evaluation,
            "development_gate": development_gate(evaluation),
            "model_npz_sha256": sha256_path(Path(model_output)),
            "clap_manifest_file_sha256": sha256_path(
                Path(clap_store) / "store.sealed.json"
            ),
            "musicfm_manifest_file_sha256": sha256_path(
                Path(musicfm_store) / "store.sealed.json"
            ),
            "wall_seconds": time.perf_counter() - started,
            "promotion_allowed": False,
        }
        report["payload_sha256"] = stable_json_sha256(report)
        _write_json_exclusive(Path(report_output), report)
        metadata: Dict[str, object] = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "artifact_kind": METRIC_MODEL_KIND,
            "semantic_base_kind": MODEL_KIND,
            "evidence_scope": EVIDENCE_SCOPE,
            "source_fingerprint": SOURCE_FINGERPRINT,
            "protocol_payload_sha256": SCALE_PROTOCOL_PAYLOAD_SHA256,
            "protocol_selection_sha256": EXPECTED_SELECTION_SHA256,
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
    parser.add_argument("--clap-text-embeddings", required=True)
    parser.add_argument("--model-output", required=True)
    parser.add_argument("--metadata-output", required=True)
    parser.add_argument("--report-output", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = train_dual_metric_candidate(
            metadata_root=Path(args.metadata_root),
            protocol_path=Path(args.protocol),
            clap_store=Path(args.clap_store),
            musicfm_store=Path(args.musicfm_store),
            clap_text_embeddings=Path(args.clap_text_embeddings),
            model_output=Path(args.model_output),
            metadata_output=Path(args.metadata_output),
            report_output=Path(args.report_output),
        )
    except (
        OSError,
        ValueError,
        FullTrackStoreError,
        V3MetricError,
        V3SemanticError,
    ) as exc:
        raise SystemExit(f"V3 dual metric run failed: {exc}") from exc
    print(
        json.dumps(
            {
                "evaluation": report["evaluation"],
                "development_gate": report["development_gate"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
