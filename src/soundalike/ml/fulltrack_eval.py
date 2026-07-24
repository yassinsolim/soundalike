"""Artist-disjoint MTG-Jamendo full-track retrieval evaluation.

The full-track evidence scope is always ``full_track_jamendo_research``.
Commercial v6 replay evidence is a separate, caller-exported, read-only
``preview_30s_commercial`` document.  This module refuses to open anything
inside a signed protocol-v6 directory.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .fulltrack_extract import fixed_budget_indices, normalize_rows
from .fulltrack_store import (
    STORE_SCHEMA_VERSION,
    FullTrackStoreReader,
    sha256_path,
    stable_json_sha256,
)
from .jamendo_fulltrack import (
    EVIDENCE_SCOPE,
    ArtistFold,
    JamendoContext,
    JamendoTrack,
    load_jamendo_context,
)


COMMERCIAL_EVIDENCE_SCOPE = "preview_30s_commercial"
METHODS = ("global_cosine", "uniform_window_maxsim", "section_maxsim", "hybrid")
HYBRID_WEIGHTS = {
    "global_cosine": 0.50,
    "uniform_window_maxsim": 0.25,
    "section_maxsim": 0.25,
}
METRICS = ("recall_at_k", "mrr", "graded_ndcg_at_k")
OFFICIAL_FOLDS = (0, 1, 2, 3, 4)
OFFICIAL_BUDGETS = (8, 16, 32)
EVALUATION_SCHEMA_VERSION = 3
BENCHMARK_SCHEMA_VERSION = 2
DATASET_DESCRIPTION = (
    "MTG-Jamendo raw_30s (duration >30s), full-quality full tracks"
)
LAWFUL_USE = "non-commercial research evaluation only"
TAG_PATTERN = re.compile(r"(genre|instrument|mood/theme)---([^\t\r\n]+)\Z")
METHOD_DEFINITIONS = {
    "global_cosine": "cosine similarity over frozen global track embeddings",
    "uniform_window_maxsim": (
        "symmetric mean-MaxSim over rounded-linspace fixed-budget windows"
    ),
    "section_maxsim": (
        "equal-weight mean of repeated-section and salient-section symmetric MaxSim"
    ),
    "hybrid": (
        "weighted sum of global cosine, uniform-window MaxSim, and section MaxSim"
    ),
}
SECTION_COMPONENT_WEIGHTS = {
    "repeated_sections": 0.5,
    "salient_sections": 0.5,
}
GROUPED_METRICS_NOTICE = (
    "Per-scene and per-tag results are descriptive and uncorrected for "
    "multiple comparisons."
)


class FullTrackEvaluationError(RuntimeError):
    """Evaluation input, evidence, metric, or resource-bound failure."""


@dataclass(frozen=True)
class EvaluationConfig:
    fold_index: int = 0
    part: str = "test"
    maxsim_budget: int = 8
    candidate_pool: int = 200
    recall_cutoff: int = 10
    ndcg_cutoff: int = 10
    bootstrap_iterations: int = 2_000
    bootstrap_seed: int = 20260714
    max_feature_cache_bytes: int = 2 * 1024**3
    query_limit: Optional[int] = None
    min_shared_tags: int = 2
    min_tag_jaccard: float = 0.25

    def validate(self) -> None:
        if self.fold_index < 0 or self.part not in ("train", "validation", "test"):
            raise FullTrackEvaluationError("invalid fold or split part")
        if self.maxsim_budget <= 0 or self.candidate_pool <= 0:
            raise FullTrackEvaluationError("MaxSim budget/pool must be positive")
        if self.recall_cutoff <= 0 or self.ndcg_cutoff <= 0:
            raise FullTrackEvaluationError("metric cutoffs must be positive")
        if self.bootstrap_iterations <= 0 or self.max_feature_cache_bytes <= 0:
            raise FullTrackEvaluationError("bootstrap/cache bounds must be positive")
        if self.bootstrap_seed < 0:
            raise FullTrackEvaluationError("bootstrap seed must be non-negative")
        if self.min_shared_tags <= 0:
            raise FullTrackEvaluationError("minimum shared tags must be positive")
        if not 0.0 < self.min_tag_jaccard <= 1.0:
            raise FullTrackEvaluationError("tag Jaccard threshold must be in (0, 1]")


@dataclass(frozen=True)
class QueryMetrics:
    recall_at_k: float
    mrr: float
    graded_ndcg_at_k: float


def _normalise_vector(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if not len(vector) or not np.all(np.isfinite(vector)):
        raise FullTrackEvaluationError("embedding must be finite and non-empty")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise FullTrackEvaluationError("embedding may not be zero")
    return vector / norm


def global_cosine(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    query_value = _normalise_vector(query)
    candidate_values = normalize_rows(candidates)
    if candidate_values.shape[1] != len(query_value):
        raise FullTrackEvaluationError("global embedding dimensions differ")
    return candidate_values @ query_value


def freeze_fixed_budget(windows: np.ndarray, budget: int) -> np.ndarray:
    values = normalize_rows(windows)
    indices = fixed_budget_indices(len(values), budget)
    return values[indices]


def freeze_ranked_section_budget(sections: np.ndarray, budget: int) -> np.ndarray:
    """Take the top-ranked section prefix; repeat only if source windows are fewer."""
    values = normalize_rows(sections)
    if len(values) >= budget:
        return values[:budget]
    return values[fixed_budget_indices(len(values), budget)]


def fixed_budget_maxsim(
    query_windows: np.ndarray,
    candidate_windows: np.ndarray,
    *,
    budget: int,
) -> float:
    """Symmetric mean-MaxSim after freezing both sides to the same budget."""
    query = freeze_fixed_budget(query_windows, budget)
    candidate = freeze_fixed_budget(candidate_windows, budget)
    similarities = query @ candidate.T
    return float(
        0.5
        * (
            np.mean(np.max(similarities, axis=1))
            + np.mean(np.max(similarities, axis=0))
        )
    )


def batch_fixed_budget_maxsim(
    query_budget: np.ndarray, candidate_budgets: np.ndarray
) -> np.ndarray:
    """Vectorized symmetric MaxSim for candidates shaped ``(N, B, D)``."""
    query = normalize_rows(query_budget)
    candidates = np.asarray(candidate_budgets, dtype=np.float32)
    if (
        candidates.ndim != 3
        or candidates.shape[1:] != query.shape
        or not np.all(np.isfinite(candidates))
    ):
        raise FullTrackEvaluationError("candidate fixed-budget tensor has invalid shape")
    norms = np.linalg.norm(candidates, axis=2, keepdims=True)
    if np.any(norms <= 1e-12):
        raise FullTrackEvaluationError("candidate fixed-budget tensor contains zero rows")
    candidates = candidates / norms
    # This remains bounded by candidate_pool * B * B.
    similarities = np.einsum("id,njd->nij", query, candidates, optimize=True)
    return 0.5 * (
        np.mean(np.max(similarities, axis=2), axis=1)
        + np.mean(np.max(similarities, axis=1), axis=1)
    )


def section_maxsim(
    query_repeated: np.ndarray,
    query_salient: np.ndarray,
    candidate_repeated: np.ndarray,
    candidate_salient: np.ndarray,
    *,
    budget: int,
) -> float:
    def score(query_sections: np.ndarray, candidate_sections: np.ndarray) -> float:
        query = freeze_ranked_section_budget(query_sections, budget)
        candidate = freeze_ranked_section_budget(candidate_sections, budget)
        similarities = query @ candidate.T
        return float(
            0.5
            * (
                np.mean(np.max(similarities, axis=1))
                + np.mean(np.max(similarities, axis=0))
            )
        )

    return 0.5 * (
        score(query_repeated, candidate_repeated)
        + score(query_salient, candidate_salient)
    )


def hybrid_score(global_score: float, uniform_score: float, section_score: float) -> float:
    """Frozen hybrid: 0.50 global + 0.25 uniform + 0.25 section MaxSim."""
    return float(
        HYBRID_WEIGHTS["global_cosine"] * global_score
        + HYBRID_WEIGHTS["uniform_window_maxsim"] * uniform_score
        + HYBRID_WEIGHTS["section_maxsim"] * section_score
    )


def _query_metrics(
    ranked_track_ids: Sequence[int],
    relevance: Mapping[int, float],
    *,
    recall_cutoff: int,
    ndcg_cutoff: int,
) -> QueryMetrics:
    if not relevance:
        raise FullTrackEvaluationError("query has no cross-artist relevant candidates")
    top_recall = ranked_track_ids[:recall_cutoff]
    hits = sum(track_id in relevance for track_id in top_recall)
    recall_at_k = hits / len(relevance)
    first_rank = next(
        (
            rank
            for rank, track_id in enumerate(ranked_track_ids, 1)
            if track_id in relevance
        ),
        None,
    )
    mrr = 0.0 if first_rank is None else 1.0 / first_rank
    dcg = sum(
        (relevance.get(track_id, 0.0) / math.log2(rank + 1))
        for rank, track_id in enumerate(ranked_track_ids[:ndcg_cutoff], 1)
        if track_id in relevance
    )
    ideal_relevance = sorted(relevance.values(), reverse=True)[:ndcg_cutoff]
    ideal = sum(
        grade / math.log2(rank + 1)
        for rank, grade in enumerate(ideal_relevance, 1)
    )
    graded_ndcg_at_k = dcg / ideal
    return QueryMetrics(
        recall_at_k=float(recall_at_k),
        mrr=float(mrr),
        graded_ndcg_at_k=float(graded_ndcg_at_k),
    )


def _mean_metrics(values: Sequence[QueryMetrics]) -> Dict[str, float]:
    if not values:
        return {name: 0.0 for name in METRICS}
    return {
        name: float(np.mean([getattr(value, name) for value in values]))
        for name in METRICS
    }


def _bootstrap_ci(
    values: Sequence[float], *, iterations: int, seed: int
) -> Tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        indices = rng.integers(0, len(array), size=len(array))
        samples[iteration] = float(np.mean(array[indices]))
    low, high = np.quantile(samples, (0.025, 0.975))
    return float(low), float(high)


def _paired_bootstrap_delta(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    iterations: int,
    seed: int,
) -> Dict[str, object]:
    baseline_values = np.asarray(baseline, dtype=np.float64)
    candidate_values = np.asarray(candidate, dtype=np.float64)
    if (
        not len(baseline_values)
        or baseline_values.shape != candidate_values.shape
        or not np.all(np.isfinite(baseline_values))
        or not np.all(np.isfinite(candidate_values))
    ):
        raise FullTrackEvaluationError("paired bootstrap inputs are invalid")
    differences = candidate_values - baseline_values
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        indices = rng.integers(0, len(differences), size=len(differences))
        samples[iteration] = float(np.mean(differences[indices]))
    low, high = np.quantile(samples, (0.025, 0.975))
    return {
        "mean_delta": float(np.mean(differences)),
        "paired_bootstrap_ci95": [float(low), float(high)],
        "bootstrap_probability_delta_gt_zero": float(np.mean(samples > 0.0)),
        "improved_queries": int(np.count_nonzero(differences > 1e-12)),
        "regressed_queries": int(np.count_nonzero(differences < -1e-12)),
        "unchanged_queries": int(np.count_nonzero(np.abs(differences) <= 1e-12)),
    }


def _tag_jaccard_relevance(
    query_tags: Sequence[str],
    candidate_tags: Sequence[str],
    *,
    min_shared_tags: int,
    min_tag_jaccard: float,
) -> float:
    query_tags = set(query_tags)
    candidate_tags = set(candidate_tags)
    shared = len(query_tags.intersection(candidate_tags))
    union = len(query_tags.union(candidate_tags))
    if shared < min_shared_tags or not union:
        return 0.0
    jaccard = shared / union
    return float(jaccard if jaccard >= min_tag_jaccard else 0.0)


def _validated_tags(value: object, *, where: str) -> Tuple[str, ...]:
    if (
        not isinstance(value, (list, tuple))
        or not value
        or any(not isinstance(tag, str) for tag in value)
    ):
        raise FullTrackEvaluationError(f"{where} tags must be a non-empty string list")
    tags = tuple(value)
    if len(tags) != len(set(tags)):
        raise FullTrackEvaluationError(f"{where} tags must be unique")
    if any(TAG_PATTERN.fullmatch(tag) is None for tag in tags):
        raise FullTrackEvaluationError(f"{where} contains a malformed scene tag")
    return tags


def _scene_for_tag(tag: str) -> str:
    match = TAG_PATTERN.fullmatch(tag)
    if match is None:
        raise FullTrackEvaluationError(f"malformed scene tag: {tag!r}")
    return match.group(1)


def _grouped_metrics(
    query_records: Sequence[Mapping[str, object]], group: str,
    *, methods: Sequence[str] = METHODS,
) -> Dict[str, object]:
    if group not in {"scene", "tag"}:
        raise FullTrackEvaluationError("unknown grouped-metric dimension")
    output: Dict[str, object] = {}
    keys = set()
    for record in query_records:
        tags = _validated_tags(
            record.get("tags"), where=f"grouped metrics track {record.get('track_id')}"
        )
        if group == "scene":
            keys.update(_scene_for_tag(tag) for tag in tags)
        else:
            keys.update(tags)
    for key in sorted(keys):
        records = [
            record
            for record in query_records
            if (
                key in record["tags"]
                if group == "tag"
                else any(tag.startswith(key + "---") for tag in record["tags"])
            )
        ]
        output[key] = {
            "queries": len(records),
            "methods": {
                method: _mean_metrics(
                    [
                        QueryMetrics(**record["metrics"][method])
                        for record in records
                    ]
                )
                for method in methods
            },
        }
    return output


def _query_descriptor_sha256(
    descriptors: Sequence[Mapping[str, object]], skipped_no_relevant: int
) -> str:
    fields = {"track_id", "artist_id", "tags", "relevant_candidates"}
    normalized = []
    for descriptor in descriptors:
        if not isinstance(descriptor, Mapping) or set(descriptor) != fields:
            raise FullTrackEvaluationError("query descriptor schema is incomplete")
        track_id = descriptor["track_id"]
        artist_id = descriptor["artist_id"]
        relevant = descriptor["relevant_candidates"]
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in (track_id, artist_id, relevant)
        ):
            raise FullTrackEvaluationError("query descriptor values are invalid")
        tags = _validated_tags(
            descriptor["tags"], where=f"query descriptor track {track_id}"
        )
        normalized.append(
            {
                "track_id": track_id,
                "artist_id": artist_id,
                "tags": list(tags),
                "relevant_candidates": relevant,
            }
        )
    if (
        isinstance(skipped_no_relevant, bool)
        or not isinstance(skipped_no_relevant, int)
        or skipped_no_relevant < 0
    ):
        raise FullTrackEvaluationError("invalid skipped-query count")
    return stable_json_sha256(
        {
            "query_descriptors": normalized,
            "skipped_no_relevant": skipped_no_relevant,
        }
    )


def _expected_query_descriptor_sha256(
    fold: ArtistFold,
    selected: Sequence[JamendoTrack],
    config: EvaluationConfig,
) -> str:
    queries = (
        selected[: config.query_limit]
        if config.query_limit is not None
        else selected
    )
    descriptors = []
    skipped = 0
    for query in queries:
        relevant = sum(
            bool(
                _tag_jaccard_relevance(
                    fold.track_tags[query.track_id],
                    fold.track_tags[candidate.track_id],
                    min_shared_tags=config.min_shared_tags,
                    min_tag_jaccard=config.min_tag_jaccard,
                )
            )
            for candidate in selected
            if candidate.track_id != query.track_id
            and candidate.artist_id != query.artist_id
        )
        if not relevant:
            skipped += 1
            continue
        descriptors.append(
            {
                "track_id": query.track_id,
                "artist_id": query.artist_id,
                "tags": list(fold.track_tags[query.track_id]),
                "relevant_candidates": relevant,
            }
        )
    return _query_descriptor_sha256(descriptors, skipped)


def _effective_unique_section_limits(
    selected_counts: Mapping[str, Sequence[int]],
    declared: Mapping[str, int],
    *,
    budget: int,
) -> Dict[str, object]:
    output: Dict[str, object] = {}
    for name in ("repeated_sections", "salient_sections"):
        counts = tuple(int(value) for value in selected_counts.get(name, ()))
        if not counts or any(value <= 0 for value in counts):
            raise FullTrackEvaluationError(
                f"{name} source-window counts must be positive and non-empty"
            )
        output[name] = {
            "store_declared_budget": int(declared[name]),
            "requested_budget": budget,
            "minimum_selected_source_windows": min(counts),
            "median_selected_source_windows": float(statistics.median(counts)),
            "maximum_selected_source_windows": max(counts),
            "tracks_repeating_for_requested_budget": sum(
                count < budget for count in counts
            ),
            "track_count": len(counts),
        }
    return output


def _evaluation_protocol(
    config: EvaluationConfig,
    effective_unique_section_limits: Mapping[str, object],
    *,
    query_descriptor_sha256: str,
) -> Dict[str, object]:
    return {
        **asdict(config),
        "artist_disjoint_official_fold": True,
        "same_artist_candidates_excluded": True,
        "metric_labels": {
            "recall_at_k": f"Recall@{config.recall_cutoff}",
            "mrr": "standard MRR over the complete ranked list",
            "graded_ndcg_at_k": f"graded NDCG@{config.ndcg_cutoff}",
        },
        "relevance": (
            "cross-artist candidates with at least min_shared_tags and tag "
            "Jaccard >= min_tag_jaccard; NDCG uses Jaccard as graded relevance"
        ),
        "claim_scope": (
            "artist-disjoint shared-tag retrieval on MTG-Jamendo; not a direct "
            "measure of perceptual similarity or annotated chorus retrieval"
        ),
        "uniform_window_selection": "rounded linspace to an exact fixed budget",
        "budget_interpretation": (
            "equalizes draw count, not effective temporal diversity; deterministic "
            "index repetition is used only when a track has fewer source windows "
            "than the requested, store-declared section budget"
        ),
        "effective_unique_section_limits": {
            name: dict(details)
            for name, details in effective_unique_section_limits.items()
        },
        "query_descriptor_sha256": query_descriptor_sha256,
        "method_definitions": dict(METHOD_DEFINITIONS),
        "maxsim": "symmetric mean of per-window maxima at equal fixed budgets",
        "section_score": (
            "0.5 embedding-self-recurrence MaxSim + 0.5 salient MaxSim; "
            "recurrence is not an annotated chorus/verse label"
        ),
        "section_component_weights": dict(SECTION_COMPONENT_WEIGHTS),
        "hybrid_weights": dict(HYBRID_WEIGHTS),
        "late_interaction_candidate_source": (
            "frozen global-cosine top candidate_pool; remaining global order unchanged"
        ),
    }


def _expected_result_metadata(
    context: JamendoContext,
    reader: FullTrackStoreReader,
    config: EvaluationConfig,
    effective_unique_section_limits: Mapping[str, object],
    *,
    candidate_tracks: int,
    query_descriptor_sha256: str,
) -> Dict[str, object]:
    if reader.binding.source_fingerprint != context.source_fingerprint:
        raise FullTrackEvaluationError(
            "current source fingerprint and store binding disagree"
        )
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "evidence_scope": EVIDENCE_SCOPE,
        "dataset": DATASET_DESCRIPTION,
        "lawful_use": LAWFUL_USE,
        "source_fingerprint": context.source_fingerprint,
        "store": _evaluation_store_binding(reader),
        "protocol": _evaluation_protocol(
            config,
            effective_unique_section_limits,
            query_descriptor_sha256=query_descriptor_sha256,
        ),
        "candidate_tracks": candidate_tracks,
    }


def _evaluation_store_binding(reader: FullTrackStoreReader) -> Dict[str, object]:
    binding = reader.binding.as_dict()
    binding["sealed_manifest_sha256"] = stable_json_sha256(reader.manifest)
    return binding


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _rss_bytes() -> int:
    try:
        import psutil
    except ImportError:
        return 0
    return int(psutil.Process().memory_info().rss)


def _cuda_peak_bytes(reset: bool = False) -> int:
    try:
        import torch
    except ImportError:
        return 0
    if not torch.cuda.is_available():
        return 0
    if reset:
        torch.cuda.reset_peak_memory_stats()
    return int(torch.cuda.max_memory_allocated())


class _BudgetCache:
    def __init__(
        self,
        reader: FullTrackStoreReader,
        track_ids: Sequence[int],
        *,
        budget: int,
        max_bytes: int,
    ) -> None:
        declared = {
            "repeated_sections": reader.binding.repetition_sections,
            "salient_sections": reader.binding.salient_sections,
        }
        underspecified = [
            f"{name}={value}" for name, value in declared.items() if budget > value
        ]
        if underspecified:
            raise FullTrackEvaluationError(
                f"requested section/hybrid budget {budget} exceeds store-declared "
                + ", ".join(underspecified)
            )
        dimension = reader.binding.embedding_dim
        required = len(track_ids) * budget * dimension * 2 * 3
        if required > max_bytes:
            raise FullTrackEvaluationError(
                f"fixed-budget feature cache requires {required} bytes, "
                f"exceeding bound {max_bytes}"
            )
        shape = (len(track_ids), budget, dimension)
        self.uniform = np.empty(shape, dtype=np.float16)
        self.repeated = np.empty(shape, dtype=np.float16)
        self.salient = np.empty(shape, dtype=np.float16)
        self.rows: Dict[int, int] = {}
        selected_counts = {"repeated_sections": [], "salient_sections": []}
        for row, track_id in enumerate(track_ids):
            track = reader.read_track(track_id)
            if not len(track.repeated_sections) or not len(track.salient_sections):
                raise FullTrackEvaluationError(
                    f"track {track_id} has no section embeddings"
                )
            self.rows[int(track_id)] = row
            selected_counts["repeated_sections"].append(len(track.repeated_indices))
            selected_counts["salient_sections"].append(len(track.salient_indices))
            self.uniform[row] = freeze_fixed_budget(
                track.window_embeddings, budget
            ).astype(np.float16)
            self.repeated[row] = freeze_ranked_section_budget(
                track.repeated_sections, budget
            ).astype(np.float16)
            self.salient[row] = freeze_ranked_section_budget(
                track.salient_sections, budget
            ).astype(np.float16)
        self.bytes = int(self.uniform.nbytes + self.repeated.nbytes + self.salient.nbytes)
        self.effective_unique_section_limits = _effective_unique_section_limits(
            selected_counts, declared, budget=budget
        )


def _method_ranking(
    method_scores: np.ndarray,
    candidate_indices: np.ndarray,
    global_order: np.ndarray,
) -> np.ndarray:
    scores = np.asarray(method_scores)
    if scores.shape != candidate_indices.shape or not np.all(np.isfinite(scores)):
        raise FullTrackEvaluationError("reranker scores are invalid")
    reranked = candidate_indices[np.argsort(-scores, kind="stable")]
    candidate_set = set(int(value) for value in candidate_indices)
    tail = [int(value) for value in global_order if int(value) not in candidate_set]
    return np.concatenate((reranked, np.asarray(tail, dtype=np.int64)))



# ---------------------------------------------------------------------------
# Trained model loading and validation for evaluator
# ---------------------------------------------------------------------------


def _trained_method_id(candidate_kind: str, seed: int, ablation: str) -> str:
    """Deterministically encode candidate/seed/ablation into a method ID."""
    if ablation == "none":
        return f"trained_{candidate_kind}_s{seed}"
    return f"trained_{candidate_kind}_s{seed}_{ablation}"


def _validate_sha256_hex(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise FullTrackEvaluationError(f"{label} must be a 64-char lowercase hex string")
    return value


@dataclass(frozen=True)
class _TrainedModelBinding:
    """Validated per-fold/candidate/seed model loaded for evaluation."""
    candidate_kind: str
    seed: int
    fold_index: int
    model: object  # FusionModel
    report_sha256: str
    model_artifact_sha256: str
    model_json_sha256: str
    weights_npz_sha256: str
    source_fingerprint: str
    store_binding_sha256: str
    training_config_sha256: str
    job_config_sha256: str
    fusion_metadata: Mapping[str, object]
    maxsim_budget: int
    embedding_dim: int


_TRAINED_CACHE_MODEL_FIELDS = frozenset(
    {
        "candidate_kind",
        "seed",
        "fold_index",
        "report_sha256",
        "model_artifact_sha256",
        "model_json_sha256",
        "weights_npz_sha256",
        "source_fingerprint",
        "store_binding_sha256",
        "training_config_sha256",
        "job_config_sha256",
        "maxsim_budget",
        "embedding_dim",
    }
)
_TRAINED_RESULT_MODEL_FIELDS = frozenset(
    {
        "candidate_kind",
        "seed",
        "fold_index",
        "ablation",
        "model_artifact_sha256",
        "model_json_sha256",
        "weights_npz_sha256",
        "report_sha256",
        "source_fingerprint",
        "store_binding_sha256",
        "training_config_sha256",
        "job_config_sha256",
        "maxsim_budget",
        "embedding_dim",
        "promoted",
    }
)


def _trained_cache_model_identity(
    binding: _TrainedModelBinding,
) -> Dict[str, object]:
    return {
        "candidate_kind": binding.candidate_kind,
        "seed": binding.seed,
        "fold_index": binding.fold_index,
        "report_sha256": binding.report_sha256,
        "model_artifact_sha256": binding.model_artifact_sha256,
        "model_json_sha256": binding.model_json_sha256,
        "weights_npz_sha256": binding.weights_npz_sha256,
        "source_fingerprint": binding.source_fingerprint,
        "store_binding_sha256": binding.store_binding_sha256,
        "training_config_sha256": binding.training_config_sha256,
        "job_config_sha256": binding.job_config_sha256,
        "maxsim_budget": binding.maxsim_budget,
        "embedding_dim": binding.embedding_dim,
    }


def _trained_result_model_binding(
    identity: Mapping[str, object], ablation: str
) -> Dict[str, object]:
    return {
        "candidate_kind": identity["candidate_kind"],
        "seed": identity["seed"],
        "fold_index": identity["fold_index"],
        "ablation": ablation,
        "model_artifact_sha256": identity["model_artifact_sha256"],
        "model_json_sha256": identity["model_json_sha256"],
        "weights_npz_sha256": identity["weights_npz_sha256"],
        "report_sha256": identity["report_sha256"],
        "source_fingerprint": identity["source_fingerprint"],
        "store_binding_sha256": identity["store_binding_sha256"],
        "training_config_sha256": identity["training_config_sha256"],
        "job_config_sha256": identity["job_config_sha256"],
        "maxsim_budget": identity["maxsim_budget"],
        "embedding_dim": identity["embedding_dim"],
        "promoted": False,
    }


def load_trained_model_for_fold(
    trained_root: Path,
    *,
    fold_index: int,
    candidate_kind: str,
    seed: int,
    expected_source_fingerprint: str,
    expected_store_binding_sha256: str,
    store_embedding_dim: int,
    store_repetition_sections: int,
    store_salient_sections: int,
) -> _TrainedModelBinding:
    """Safely load ONE fold training report + model artifacts.

    Opens ONLY fold-{fold_index}/{candidate_kind}/seed-{seed}/
    Never opens any other fold directory.
    """
    from .fulltrack_fusion import CANDIDATE_KINDS as FUSION_CANDIDATE_KINDS
    from .fulltrack_train import (
        FullTrackTrainingError,
        TrainJobSpec,
        _model_artifact_hashes,
        load_training_report as _ltr,
        validate_training_report_bindings,
    )
    from .fulltrack_fusion import FusionError, load_fusion_artifact as _lfa

    if fold_index not in OFFICIAL_FOLDS:
        raise FullTrackEvaluationError(
            f"trained fold must be one of {OFFICIAL_FOLDS}, found {fold_index}"
        )
    if candidate_kind not in FUSION_CANDIDATE_KINDS:
        raise FullTrackEvaluationError(
            f"unknown trained candidate kind: {candidate_kind!r}"
        )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise FullTrackEvaluationError("trained seed must be a non-negative integer")
    try:
        resolved_root = Path(trained_root).resolve(strict=True)
        fold_dir = (
            resolved_root
            / f"fold-{fold_index}"
            / candidate_kind
            / f"seed-{seed}"
        ).resolve(strict=True)
    except OSError as exc:
        raise FullTrackEvaluationError(
            "trained model directory is missing or cannot be resolved"
        ) from exc
    if resolved_root != fold_dir and resolved_root not in fold_dir.parents:
        raise FullTrackEvaluationError(
            "trained model directory escapes the configured trained root"
        )
    try:
        report = _ltr(fold_dir / "report.json")
    except FullTrackTrainingError as exc:
        raise FullTrackEvaluationError(f"training report invalid: {exc}") from exc
    if report.get("fold") != fold_index:
        raise FullTrackEvaluationError(f"training report fold mismatch: {report.get('fold')} != {fold_index}")
    if report.get("candidate_kind") != candidate_kind:
        raise FullTrackEvaluationError(f"training report candidate_kind mismatch")
    if report.get("seed") != seed:
        raise FullTrackEvaluationError(f"training report seed mismatch: {report.get('seed')} != {seed}")
    rp_src = _validate_sha256_hex(report.get("source_fingerprint"), "report source_fingerprint")
    if rp_src != expected_source_fingerprint:
        raise FullTrackEvaluationError("training report source_fingerprint mismatch")
    rp_store = _validate_sha256_hex(report.get("store_binding_sha256"), "report store_binding_sha256")
    if rp_store != expected_store_binding_sha256:
        raise FullTrackEvaluationError("training report store_binding_sha256 mismatch")
    sb = report.get("store_binding")
    if not isinstance(sb, dict) or stable_json_sha256(sb) != rp_store:
        raise FullTrackEvaluationError("training report store_binding content hash mismatch")
    store_manifest_sha256 = _validate_sha256_hex(
        sb.get("sealed_manifest_sha256"), "store sealed_manifest_sha256"
    )
    spec = TrainJobSpec(
        fold_index=fold_index,
        candidate_kind=candidate_kind,
        seed=seed,
        job_id=f"fold-{fold_index}__{candidate_kind}__seed-{seed}",
        relative_dir=f"fold-{fold_index}/{candidate_kind}/seed-{seed}",
    )
    try:
        validate_training_report_bindings(
            report,
            spec=spec,
            source_fingerprint=rp_src,
            store_binding_hash=rp_store,
            store_manifest_sha256=store_manifest_sha256,
        )
    except FullTrackTrainingError as exc:
        raise FullTrackEvaluationError(f"training report binding invalid: {exc}") from exc
    model_dir = fold_dir / "model"
    try:
        model = _lfa(model_dir)
    except FusionError as exc:
        raise FullTrackEvaluationError(f"trained model artifact invalid: {exc}") from exc
    ms = report.get("model")
    if not isinstance(ms, dict):
        raise FullTrackEvaluationError("training report model section invalid")
    mj_sha = _validate_sha256_hex(ms.get("model_json_sha256"), "model_json_sha256")
    wn_sha = _validate_sha256_hex(ms.get("weights_npz_sha256"), "weights_npz_sha256")
    art_sha = _validate_sha256_hex(ms.get("artifact_sha256"), "artifact_sha256")
    try:
        ah = _model_artifact_hashes(model_dir)
    except FullTrackTrainingError as exc:
        raise FullTrackEvaluationError(f"trained model hashes invalid: {exc}") from exc
    if ah["model_json_sha256"] != mj_sha:
        raise FullTrackEvaluationError("model.json hash mismatch with training report")
    if ah["weights_npz_sha256"] != wn_sha:
        raise FullTrackEvaluationError("weights.npz hash mismatch with training report")
    if ah["artifact_sha256"] != art_sha:
        raise FullTrackEvaluationError("artifact hash mismatch with training report")
    if model.config.kind != candidate_kind or int(model.config.seed) != seed or int(model.config.fold_index) != fold_index:
        raise FullTrackEvaluationError("loaded model config does not match expected")
    if model.config.store_id != rp_store:
        raise FullTrackEvaluationError("loaded model store_id mismatch")
    if model.config.config_sha256 != report.get("job_config_sha256"):
        raise FullTrackEvaluationError("loaded model config_sha256 mismatch")
    mb = int(model.config.maxsim_budget)
    md = int(model.config.embedding_dim)
    if md != store_embedding_dim:
        raise FullTrackEvaluationError(f"model embedding_dim {md} != store {store_embedding_dim}")
    if mb > store_repetition_sections:
        raise FullTrackEvaluationError(f"model maxsim_budget {mb} exceeds store repetition_sections")
    if mb > store_salient_sections:
        raise FullTrackEvaluationError(f"model maxsim_budget {mb} exceeds store salient_sections")
    fm = ms.get("fusion_metadata", {})
    if not isinstance(fm, dict):
        raise FullTrackEvaluationError("fusion_metadata invalid")
    if model.metadata is None or model.metadata.as_dict() != fm:
        raise FullTrackEvaluationError(
            "loaded model fusion metadata does not match training report"
        )
    r_sha = _validate_sha256_hex(report.get("report_sha256"), "report_sha256")
    tc_sha = _validate_sha256_hex(report.get("training_config_sha256"), "training_config_sha256")
    jc_sha = _validate_sha256_hex(report.get("job_config_sha256"), "job_config_sha256")
    return _TrainedModelBinding(
        candidate_kind=candidate_kind, seed=seed, fold_index=fold_index,
        model=model, report_sha256=r_sha, model_artifact_sha256=art_sha,
        model_json_sha256=mj_sha, weights_npz_sha256=wn_sha,
        source_fingerprint=rp_src, store_binding_sha256=rp_store,
        training_config_sha256=tc_sha, job_config_sha256=jc_sha,
        fusion_metadata=fm, maxsim_budget=mb, embedding_dim=md,
    )


def _score_trained_candidate_pool(
    query: object,
    candidates: Sequence[object],
    trained_bindings: Mapping[
        str, Tuple["_TrainedModelBinding", str]
    ],
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    feature_groups: Dict[Tuple[int, int, int, float], List[str]] = {}
    channel_methods: List[str] = []
    for method_id, (binding, _) in trained_bindings.items():
        config = binding.model.config
        if config.kind == "channel_gated_embedding":
            channel_methods.append(method_id)
            continue
        feature_key = (
            int(config.embedding_dim),
            int(config.maxsim_budget),
            int(config.top_k),
            float(config.coverage_threshold),
        )
        feature_groups.setdefault(feature_key, []).append(method_id)

    scores: Dict[str, np.ndarray] = {}
    latencies: Dict[str, float] = {}
    for method_ids in feature_groups.values():
        representative = trained_bindings[method_ids[0]][0].model
        feature_started = time.perf_counter()
        feature_vectors = np.stack(
            [
                representative.extract_pair_features(query, candidate).to_vector()
                for candidate in candidates
            ]
        )
        feature_seconds = time.perf_counter() - feature_started
        for method_id in method_ids:
            binding, ablation = trained_bindings[method_id]
            scoring_started = time.perf_counter()
            scores[method_id] = binding.model.score_feature_vectors(
                feature_vectors, ablation=ablation
            )
            latencies[method_id] = (
                feature_seconds + time.perf_counter() - scoring_started
            )

    for method_id in channel_methods:
        binding, ablation = trained_bindings[method_id]
        scoring_started = time.perf_counter()
        scores[method_id] = binding.model.score_candidates(
            query, candidates, ablation=ablation
        )
        latencies[method_id] = time.perf_counter() - scoring_started
    return scores, latencies


def evaluate_jamendo(
    context: JamendoContext,
    reader: FullTrackStoreReader,
    *,
    config: EvaluationConfig = EvaluationConfig(),
    expected_query_descriptor_hash: Optional[str] = None,
    trained_bindings: Optional[Sequence["_TrainedModelBinding"]] = None,
    include_ablations: bool = True,
) -> Mapping[str, object]:
    """Evaluate one official artist-disjoint fold partition without audio access."""
    config.validate()
    fold_position = next(
        (
            position
            for position, fold in enumerate(context.folds)
            if fold.index == config.fold_index
        ),
        None,
    )
    if fold_position is None:
        raise FullTrackEvaluationError(f"fold {config.fold_index} is not loaded")
    fold = context.folds[fold_position]
    selected = [
        track
        for track in context.tracks
        if fold.track_parts.get(track.track_id) == config.part
    ]
    for track in selected:
        _validated_tags(
            fold.track_tags.get(track.track_id),
            where=f"fold {config.fold_index} track {track.track_id}",
        )
    if config.query_limit is not None:
        queries = selected[: config.query_limit]
    else:
        queries = selected
    if len(selected) < 2 or not queries:
        raise FullTrackEvaluationError("selected evaluation partition is too small")
    if expected_query_descriptor_hash is None:
        expected_query_descriptor_hash = _expected_query_descriptor_sha256(
            fold, selected, config
        )
    selected_ids = [track.track_id for track in selected]
    reader_ids = set(reader.track_ids)
    if not set(selected_ids).issubset(reader_ids):
        raise FullTrackEvaluationError("store does not cover the evaluation partition")

    # --- Build trained method list ---
    from .fulltrack_fusion import ABLATIONS as FUSION_ABLATIONS
    trained_methods: List[str] = []
    trained_bindings_map: Dict[str, Tuple["_TrainedModelBinding", str]] = {}
    if trained_bindings:
        for tb in trained_bindings:
            if tb.fold_index != config.fold_index:
                raise FullTrackEvaluationError(
                    f"trained binding fold {tb.fold_index} != eval fold {config.fold_index}"
                )
            ablations_to_use = list(FUSION_ABLATIONS) if include_ablations else ["none"]
            for abl in ablations_to_use:
                mid = _trained_method_id(tb.candidate_kind, tb.seed, abl)
                if mid in trained_bindings_map:
                    raise FullTrackEvaluationError(f"duplicate trained method ID: {mid}")
                trained_methods.append(mid)
                trained_bindings_map[mid] = (tb, abl)
    all_methods = list(METHODS) + trained_methods

    started = time.perf_counter()
    rss_before = _rss_bytes()
    rss_peak = rss_before
    _cuda_peak_bytes(reset=True)
    cache = _BudgetCache(
        reader,
        selected_ids,
        budget=config.maxsim_budget,
        max_bytes=config.max_feature_cache_bytes,
    )
    id_to_position = {track_id: index for index, track_id in enumerate(selected_ids)}
    globals_matrix = np.asarray(
        reader.global_embeddings[
            [reader.read_track(track_id).row_index for track_id in selected_ids]
        ],
        dtype=np.float32,
    )
    globals_matrix = normalize_rows(globals_matrix)
    track_by_id = {track.track_id: track for track in selected}
    # --- Preload stored tracks for trained models ---
    stored_tracks: Dict[int, object] = {}
    if trained_bindings:
        for track_id in selected_ids:
            stored_tracks[int(track_id)] = reader.read_track(track_id)
    stored_track_list = [stored_tracks.get(tid) for tid in selected_ids] if trained_bindings else []

    method_metrics: Dict[str, List[QueryMetrics]] = {method: [] for method in all_methods}
    method_latencies: Dict[str, List[float]] = {method: [] for method in all_methods}
    query_records = []
    skipped_no_relevant = 0

    for query in queries:
        query_position = id_to_position[query.track_id]
        eligible = np.asarray(
            [
                index
                for index, candidate in enumerate(selected)
                if candidate.track_id != query.track_id
                and candidate.artist_id != query.artist_id
            ],
            dtype=np.int64,
        )
        relevant = {
            candidate.track_id: grade
            for candidate in selected
            if (
                grade := _tag_jaccard_relevance(
                    fold.track_tags[query.track_id],
                    fold.track_tags[candidate.track_id],
                    min_shared_tags=config.min_shared_tags,
                    min_tag_jaccard=config.min_tag_jaccard,
                )
            )
            if candidate.track_id != query.track_id
            and candidate.artist_id != query.artist_id
        }
        if not relevant:
            skipped_no_relevant += 1
            continue
        global_started = time.perf_counter()
        global_scores = globals_matrix[eligible] @ globals_matrix[query_position]
        global_order_local = np.lexsort((eligible, -global_scores))
        global_order = eligible[global_order_local]
        method_latencies["global_cosine"].append(
            time.perf_counter() - global_started
        )
        pool_count = min(config.candidate_pool, len(global_order))
        pool = global_order[:pool_count]
        query_cache_row = cache.rows[query.track_id]

        uniform_started = time.perf_counter()
        uniform_scores = batch_fixed_budget_maxsim(
            cache.uniform[query_cache_row],
            cache.uniform[pool].astype(np.float32),
        )
        uniform_order = _method_ranking(uniform_scores, pool, global_order)
        method_latencies["uniform_window_maxsim"].append(
            time.perf_counter() - uniform_started
        )

        section_started = time.perf_counter()
        repeated_scores = batch_fixed_budget_maxsim(
            cache.repeated[query_cache_row],
            cache.repeated[pool].astype(np.float32),
        )
        salient_scores = batch_fixed_budget_maxsim(
            cache.salient[query_cache_row],
            cache.salient[pool].astype(np.float32),
        )
        section_scores = 0.5 * (repeated_scores + salient_scores)
        section_order = _method_ranking(section_scores, pool, global_order)
        method_latencies["section_maxsim"].append(
            time.perf_counter() - section_started
        )

        hybrid_started = time.perf_counter()
        global_pool_scores = globals_matrix[pool] @ globals_matrix[query_position]
        hybrid_scores = (
            HYBRID_WEIGHTS["global_cosine"] * global_pool_scores
            + HYBRID_WEIGHTS["uniform_window_maxsim"] * uniform_scores
            + HYBRID_WEIGHTS["section_maxsim"] * section_scores
        )
        hybrid_order = _method_ranking(hybrid_scores, pool, global_order)
        method_latencies["hybrid"].append(time.perf_counter() - hybrid_started)

        orders = {
            "global_cosine": global_order,
            "uniform_window_maxsim": uniform_order,
            "section_maxsim": section_order,
            "hybrid": hybrid_order,
        }
        # --- Trained methods rerank the SAME frozen-global candidate pool ---
        if trained_bindings:
            query_stored = stored_tracks[query.track_id]
            candidate_tracks = [stored_track_list[int(idx)] for idx in pool]
            trained_scores, trained_latencies = _score_trained_candidate_pool(
                query_stored,
                candidate_tracks,
                trained_bindings_map,
            )
            for mid in trained_methods:
                t_scores = np.asarray(trained_scores[mid], dtype=np.float64)
                if not np.all(np.isfinite(t_scores)) or np.any(t_scores < 0.0) or np.any(t_scores > 1.0):
                    raise FullTrackEvaluationError(f"trained model scores not finite/bounded for {mid}")
                orders[mid] = _method_ranking(t_scores, pool, global_order)
                method_latencies[mid].append(trained_latencies[mid])

        record_metrics = {}
        for method, order in orders.items():
            ranked_ids = [selected[int(index)].track_id for index in order]
            metrics = _query_metrics(
                ranked_ids,
                relevant,
                recall_cutoff=config.recall_cutoff,
                ndcg_cutoff=config.ndcg_cutoff,
            )
            method_metrics[method].append(metrics)
            record_metrics[method] = asdict(metrics)
        query_records.append(
            {
                "track_id": query.track_id,
                "artist_id": query.artist_id,
                "tags": list(fold.track_tags[query.track_id]),
                "relevant_candidates": len(relevant),
                "metrics": record_metrics,
            }
        )
        rss_peak = max(rss_peak, _rss_bytes())

    if not query_records:
        raise FullTrackEvaluationError("no evaluable cross-artist queries")
    actual_query_descriptor_hash = _query_descriptor_sha256(
        [
            {
                key: record[key]
                for key in (
                    "track_id",
                    "artist_id",
                    "tags",
                    "relevant_candidates",
                )
            }
            for record in query_records
        ],
        skipped_no_relevant,
    )
    if actual_query_descriptor_hash != expected_query_descriptor_hash:
        raise FullTrackEvaluationError(
            "evaluated query descriptors differ from current fold labels"
        )

    aggregate = {}
    for method in all_methods:
        values = method_metrics[method]
        metrics = _mean_metrics(values)
        cis = {
            name: list(
                _bootstrap_ci(
                    [getattr(value, name) for value in values],
                    iterations=config.bootstrap_iterations,
                    seed=config.bootstrap_seed,
                )
            )
            for name in METRICS
        }
        comparisons = {
            name: _paired_bootstrap_delta(
                [getattr(value, name) for value in method_metrics["global_cosine"]],
                [getattr(value, name) for value in values],
                iterations=config.bootstrap_iterations,
                seed=config.bootstrap_seed,
            )
            for name in METRICS
        }
        aggregate[method] = {
            "metrics": metrics,
            "bootstrap_ci95": cis,
            "comparison_to_global": comparisons,
        }

    rss_after = _rss_bytes()
    expected_metadata = _expected_result_metadata(
        context,
        reader,
        config,
        cache.effective_unique_section_limits,
        candidate_tracks=len(selected),
        query_descriptor_sha256=expected_query_descriptor_hash,
    )
    report: Dict[str, object] = {
        **expected_metadata,
        "queries": len(query_records),
        "skipped_no_relevant": skipped_no_relevant,
        "query_records": query_records,
        "aggregate": aggregate,
        "per_scene": _grouped_metrics(query_records, "scene", methods=all_methods),
        "per_tag": _grouped_metrics(query_records, "tag", methods=all_methods),
        "grouped_metrics_notice": GROUPED_METRICS_NOTICE,
        "resources": {
            "wall_seconds": time.perf_counter() - started,
            "rss_before_bytes": rss_before,
            "rss_after_bytes": rss_after,
            "rss_observed_peak_bytes": max(rss_peak, rss_after),
            "cuda_peak_allocated_bytes": _cuda_peak_bytes(),
            "feature_cache_bytes": cache.bytes,
            "store_bytes": reader.storage_bytes,
            "latency_seconds": {
                method: {
                    "mean": (
                        statistics.fmean(method_latencies[method])
                        if method_latencies[method]
                        else 0.0
                    ),
                    "p50": _percentile(method_latencies[method], 50),
                    "p95": _percentile(method_latencies[method], 95),
                }
                for method in all_methods
            },
        },
    }

    # --- Add trained_methods section only when opt-in ---
    if trained_methods:
        tmb = {}
        for mid in trained_methods:
            tb, abl = trained_bindings_map[mid]
            tmb[mid] = _trained_result_model_binding(
                _trained_cache_model_identity(tb), abl
            )
        tpd = {}
        for mid in trained_methods:
            hvs = {}
            for metric in METRICS:
                hvs[metric] = _paired_bootstrap_delta(
                    [getattr(v, metric) for v in method_metrics["hybrid"]],
                    [getattr(v, metric) for v in method_metrics[mid]],
                    iterations=config.bootstrap_iterations,
                    seed=config.bootstrap_seed,
                )
            tpd[mid] = {
                "paired_candidate_minus_global": aggregate[mid]["comparison_to_global"],
                "paired_candidate_minus_frozen_hybrid": hvs,
            }
        report["trained_methods"] = trained_methods
        report["trained_model_bindings"] = tmb
        report["trained_paired_deltas"] = tpd

    return report


def _benchmark_result_binding(
    context: JamendoContext,
    reader: FullTrackStoreReader,
    config: EvaluationConfig,
    *,
    trained_bindings: Optional[Sequence[_TrainedModelBinding]] = None,
    include_ablations: bool = True,
) -> Dict[str, object]:
    """Return the exact source/store/protocol identity for one result artifact."""
    binding: Dict[str, object] = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "evidence_scope": EVIDENCE_SCOPE,
        "source_fingerprint": context.source_fingerprint,
        "store_binding": _evaluation_store_binding(reader),
        "evaluation_config": asdict(config),
        "metric_fields": list(METRICS),
    }
    if trained_bindings:
        from .fulltrack_fusion import ABLATIONS as FUSION_ABLATIONS

        binding["trained_evaluation"] = {
            "schema_version": 1,
            "ablations": list(FUSION_ABLATIONS) if include_ablations else ["none"],
            "models": [_trained_cache_model_identity(item) for item in trained_bindings],
        }
    return binding


def _benchmark_result_path(output_dir: Path, fold: int, budget: int) -> Path:
    return Path(output_dir) / f"fold-{fold}-budget-{budget}.json"


def _load_valid_benchmark_result(
    path: Path,
    expected_binding: Mapping[str, object],
    expected_metadata: Mapping[str, object],
) -> Optional[Mapping[str, object]]:
    """Return a result only when its full current-input protocol envelope verifies."""
    path = Path(path)
    assert_not_signed_protocol_path(path)
    if not path.exists():
        return None
    try:
        resolved = path.resolve(strict=True)
        assert_not_signed_protocol_path(resolved)
        if path.is_symlink() or not resolved.is_file():
            return None
        if resolved.stat().st_size > 1024**3:
            return None
        artifact = json.loads(resolved.read_text(encoding="utf-8"))
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
        RecursionError,
    ):
        return None
    if not isinstance(artifact, dict):
        return None
    payload_hash = artifact.get("artifact_payload_sha256")
    payload = dict(artifact)
    payload.pop("artifact_payload_sha256", None)
    try:
        payload_valid = payload_hash == stable_json_sha256(payload)
    except (
        TypeError,
        ValueError,
        OverflowError,
        RecursionError,
        UnicodeEncodeError,
    ):
        return None
    if not payload_valid:
        return None
    try:
        binding_hash = stable_json_sha256(expected_binding)
    except (
        TypeError,
        ValueError,
        OverflowError,
        RecursionError,
        UnicodeEncodeError,
    ):
        return None
    if (
        artifact.get("schema_version") != BENCHMARK_SCHEMA_VERSION
        or artifact.get("evidence_scope") != EVIDENCE_SCOPE
        or artifact.get("artifact_kind") != "fulltrack_fold_budget_result"
        or artifact.get("binding") != expected_binding
        or artifact.get("binding_sha256") != binding_hash
    ):
        return None
    result = artifact.get("result")
    if not isinstance(result, dict):
        return None
    try:
        result_valid = artifact.get("result_sha256") == stable_json_sha256(result)
    except (
        TypeError,
        ValueError,
        OverflowError,
        RecursionError,
        UnicodeEncodeError,
    ):
        return None
    if not result_valid:
        return None
    if not isinstance(expected_metadata, Mapping) or any(
        result.get(key) != value for key, value in expected_metadata.items()
    ):
        return None
    try:
        _validate_aggregation_envelope(
            result,
            expected_slot=(
                int(expected_metadata["protocol"]["fold_index"]),
                int(expected_metadata["protocol"]["maxsim_budget"]),
            ),
            bootstrap_iterations=int(
                expected_metadata["protocol"]["bootstrap_iterations"]
            ),
            bootstrap_seed=int(expected_metadata["protocol"]["bootstrap_seed"]),
            expected_metadata=expected_metadata,
            expected_binding=expected_binding,
            require_uncertainty=True,
        )
    except (
        FullTrackEvaluationError,
        TypeError,
        ValueError,
        OverflowError,
        RecursionError,
        UnicodeEncodeError,
    ):
        return None
    return result


def _write_benchmark_result(
    path: Path,
    binding: Mapping[str, object],
    result: Mapping[str, object],
) -> Mapping[str, object]:
    artifact: Dict[str, object] = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "evidence_scope": EVIDENCE_SCOPE,
        "artifact_kind": "fulltrack_fold_budget_result",
        "binding": dict(binding),
        "binding_sha256": stable_json_sha256(binding),
        "result": dict(result),
        "result_sha256": stable_json_sha256(result),
    }
    artifact["artifact_payload_sha256"] = stable_json_sha256(artifact)
    write_evaluation_report(path, artifact)
    return artifact


def _validated_report_key(report: Mapping[str, object]) -> Tuple[int, int]:
    protocol = report.get("protocol")
    if not isinstance(protocol, dict):
        raise FullTrackEvaluationError("benchmark result has no protocol object")
    fold = protocol.get("fold_index")
    budget = protocol.get("maxsim_budget")
    if isinstance(fold, bool) or not isinstance(fold, int):
        raise FullTrackEvaluationError("benchmark result has an invalid fold index")
    if isinstance(budget, bool) or not isinstance(budget, int):
        raise FullTrackEvaluationError("benchmark result has an invalid budget")
    return fold, budget


def _query_metric_maps(
    report: Mapping[str, object], fold: int,
    *, methods: Sequence[str] = METHODS,
) -> Dict[int, Dict[str, Dict[str, float]]]:
    records = report.get("query_records")
    if not isinstance(records, list) or not records:
        raise FullTrackEvaluationError(
            f"fold {fold} benchmark result has no query records"
        )
    output: Dict[int, Dict[str, Dict[str, float]]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise FullTrackEvaluationError("query record must be an object")
        if set(record) != {
            "track_id",
            "artist_id",
            "tags",
            "relevant_candidates",
            "metrics",
        }:
            raise FullTrackEvaluationError("query record schema is incomplete")
        track_id = record.get("track_id")
        if isinstance(track_id, bool) or not isinstance(track_id, int):
            raise FullTrackEvaluationError("query record has an invalid track ID")
        if track_id in output:
            raise FullTrackEvaluationError(
                f"duplicate aligned query key fold={fold}, track={track_id}"
            )
        _validated_tags(record.get("tags"), where=f"fold {fold} track {track_id}")
        for field in ("artist_id", "relevant_candidates"):
            value = record.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise FullTrackEvaluationError(
                    f"query record has an invalid {field}"
                )
        raw_metrics = record.get("metrics")
        if not isinstance(raw_metrics, dict) or set(raw_metrics) != set(methods):
            raise FullTrackEvaluationError(
                f"query alignment is incomplete for fold={fold}, track={track_id}"
            )
        method_values: Dict[str, Dict[str, float]] = {}
        for method in methods:
            raw_values = raw_metrics.get(method)
            if not isinstance(raw_values, dict) or set(raw_values) != set(METRICS):
                raise FullTrackEvaluationError(
                    f"metric alignment is incomplete for {method}, fold={fold}, "
                    f"track={track_id}"
                )
            if any(
                isinstance(raw_values[name], bool)
                or not isinstance(raw_values[name], (int, float))
                for name in METRICS
            ):
                raise FullTrackEvaluationError("query metrics must be numeric")
            try:
                values = {name: float(raw_values[name]) for name in METRICS}
            except (TypeError, ValueError, OverflowError) as exc:
                raise FullTrackEvaluationError("query metrics must be numeric") from exc
            if not all(0.0 <= value <= 1.0 for value in values.values()):
                raise FullTrackEvaluationError(
                    "query metrics must be finite and within [0, 1]"
                )
            method_values[method] = values
        output[track_id] = method_values
    return output


def _validate_fold_result_consistency(
    report: Mapping[str, object],
    *,
    require_uncertainty: bool = False,
    methods: Sequence[str] = METHODS,
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """Validate one cached fold result and recompute its aggregate means."""
    fold, budget = _validated_report_key(report)
    query_map = _query_metric_maps(report, fold, methods=methods)
    queries = report.get("queries")
    if isinstance(queries, bool) or not isinstance(queries, int):
        raise FullTrackEvaluationError("fold result has an invalid query count")
    if queries != len(query_map):
        raise FullTrackEvaluationError(
            f"fold {fold}, budget {budget} query count disagrees with query records"
        )
    aggregate = report.get("aggregate")
    if not isinstance(aggregate, dict) or set(aggregate) != set(methods):
        raise FullTrackEvaluationError(
            f"fold {fold}, budget {budget} has incomplete aggregate methods"
        )
    for method in methods:
        method_result = aggregate.get(method)
        if not isinstance(method_result, dict):
            raise FullTrackEvaluationError("method aggregate must be an object")
        raw_metrics = method_result.get("metrics")
        if not isinstance(raw_metrics, dict) or set(raw_metrics) != set(METRICS):
            raise FullTrackEvaluationError("method aggregate metrics are incomplete")
        for metric in METRICS:
            if isinstance(raw_metrics[metric], bool) or not isinstance(
                raw_metrics[metric], (int, float)
            ):
                raise FullTrackEvaluationError(
                    "method aggregate metrics must be numeric"
                )
            declared = float(raw_metrics[metric])
            recomputed = float(
                np.mean(
                    [query_map[track_id][method][metric] for track_id in query_map]
                )
            )
            if not math.isfinite(declared) or not math.isclose(
                declared, recomputed, rel_tol=0.0, abs_tol=1e-12
            ):
                raise FullTrackEvaluationError(
                    f"fold {fold}, budget {budget} aggregate disagrees with "
                    f"aligned query records for {method}/{metric}"
                )
    if require_uncertainty:
        protocol = report.get("protocol")
        if not isinstance(protocol, dict):
            raise FullTrackEvaluationError("fold result has no protocol object")
        iterations = protocol.get("bootstrap_iterations")
        seed = protocol.get("bootstrap_seed")
        if (
            isinstance(iterations, bool)
            or not isinstance(iterations, int)
            or iterations <= 0
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed < 0
        ):
            raise FullTrackEvaluationError("fold result has invalid bootstrap config")
        baseline = {
            metric: [
                query_map[track_id]["global_cosine"][metric]
                for track_id in query_map
            ]
            for metric in METRICS
        }
        for method in methods:
            method_result = aggregate[method]
            expected_cis = {
                metric: list(
                    _bootstrap_ci(
                        [query_map[track_id][method][metric] for track_id in query_map],
                        iterations=iterations,
                        seed=seed,
                    )
                )
                for metric in METRICS
            }
            expected_comparisons = {
                metric: _paired_bootstrap_delta(
                    baseline[metric],
                    [query_map[track_id][method][metric] for track_id in query_map],
                    iterations=iterations,
                    seed=seed,
                )
                for metric in METRICS
            }
            if method_result.get("bootstrap_ci95") != expected_cis:
                raise FullTrackEvaluationError(
                    f"fold {fold}, budget {budget} has invalid bootstrap intervals"
                )
            if method_result.get("comparison_to_global") != expected_comparisons:
                raise FullTrackEvaluationError(
                    f"fold {fold}, budget {budget} has invalid paired comparisons"
                )
    return query_map


def _validate_store_binding(value: object) -> Dict[str, object]:
    fields = {
        "schema_version",
        "source_fingerprint",
        "config_sha256",
        "model_sha256",
        "model_id",
        "embedding_dim",
        "track_count",
        "shard_tracks",
        "repetition_sections",
        "salient_sections",
        "track_plan_sha256",
        "sealed_manifest_sha256",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise FullTrackEvaluationError("fold result has an incomplete store binding")
    for field in (
        "schema_version",
        "embedding_dim",
        "track_count",
        "shard_tracks",
        "repetition_sections",
        "salient_sections",
    ):
        item = value[field]
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise FullTrackEvaluationError(
                f"fold result store binding has invalid {field}"
            )
    if value["schema_version"] != STORE_SCHEMA_VERSION:
        raise FullTrackEvaluationError("fold result store schema version drift")
    for field in (
        "source_fingerprint",
        "config_sha256",
        "model_sha256",
        "model_id",
        "track_plan_sha256",
        "sealed_manifest_sha256",
    ):
        if not isinstance(value[field], str) or not value[field]:
            raise FullTrackEvaluationError(
                f"fold result store binding has invalid {field}"
            )
    return dict(value)


def _validate_effective_section_metadata(
    value: object,
    *,
    store: Mapping[str, object],
    budget: int,
    candidate_tracks: int,
) -> Dict[str, Dict[str, object]]:
    streams = ("repeated_sections", "salient_sections")
    fields = {
        "store_declared_budget",
        "requested_budget",
        "minimum_selected_source_windows",
        "median_selected_source_windows",
        "maximum_selected_source_windows",
        "tracks_repeating_for_requested_budget",
        "track_count",
    }
    if not isinstance(value, dict) or set(value) != set(streams):
        raise FullTrackEvaluationError(
            "effective section diversity metadata is incomplete"
        )
    output: Dict[str, Dict[str, object]] = {}
    for stream in streams:
        store_field = (
            "repetition_sections"
            if stream == "repeated_sections"
            else "salient_sections"
        )
        details = value.get(stream)
        if not isinstance(details, dict) or set(details) != fields:
            raise FullTrackEvaluationError(
                f"effective section diversity metadata is incomplete for {stream}"
            )
        integer_fields = fields.difference({"median_selected_source_windows"})
        if any(
            isinstance(details[field], bool)
            or not isinstance(details[field], int)
            for field in integer_fields
        ):
            raise FullTrackEvaluationError(
                f"effective section diversity metadata has invalid types for {stream}"
            )
        median = details["median_selected_source_windows"]
        if (
            isinstance(median, bool)
            or not isinstance(median, (int, float))
            or not math.isfinite(float(median))
        ):
            raise FullTrackEvaluationError(
                f"effective section diversity median is invalid for {stream}"
            )
        declared = details["store_declared_budget"]
        minimum = details["minimum_selected_source_windows"]
        maximum = details["maximum_selected_source_windows"]
        repeating = details["tracks_repeating_for_requested_budget"]
        if (
            declared != store[store_field]
            or details["requested_budget"] != budget
            or details["track_count"] != candidate_tracks
            or not 1 <= minimum <= float(median) <= maximum <= declared
            or not 0 <= repeating <= candidate_tracks
            or (minimum >= budget and repeating != 0)
            or (minimum < budget and repeating == 0)
            or (maximum < budget and repeating != candidate_tracks)
            or (maximum >= budget and repeating == candidate_tracks)
        ):
            raise FullTrackEvaluationError(
                f"effective section diversity semantics drift for {stream}"
            )
        output[stream] = dict(details)
    return output


def _validate_resource_metadata(
    value: object, *, methods: Sequence[str] = METHODS
) -> None:
    fields = {
        "wall_seconds",
        "rss_before_bytes",
        "rss_after_bytes",
        "rss_observed_peak_bytes",
        "cuda_peak_allocated_bytes",
        "feature_cache_bytes",
        "store_bytes",
        "latency_seconds",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise FullTrackEvaluationError("fold result resource metadata is incomplete")
    for field in fields.difference({"latency_seconds"}):
        item = value[field]
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or item < 0
        ):
            raise FullTrackEvaluationError(
                f"fold result resource metadata has invalid {field}"
            )
    latencies = value["latency_seconds"]
    if not isinstance(latencies, dict) or set(latencies) != set(methods):
        raise FullTrackEvaluationError("fold result latency metadata is incomplete")
    for method, stats in latencies.items():
        if not isinstance(stats, dict) or set(stats) != {"mean", "p50", "p95"}:
            raise FullTrackEvaluationError(
                f"fold result latency metadata is incomplete for {method}"
            )
        if any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            or item < 0
            for item in stats.values()
        ):
            raise FullTrackEvaluationError(
                f"fold result latency metadata is invalid for {method}"
            )


def _expected_trained_result_metadata(
    value: object, *, fold: int
) -> Tuple[Tuple[str, ...], Dict[str, Dict[str, object]]]:
    from .fulltrack_fusion import ABLATIONS as FUSION_ABLATIONS
    from .fulltrack_fusion import CANDIDATE_KINDS as FUSION_CANDIDATE_KINDS

    fields = {"schema_version", "ablations", "models"}
    if not isinstance(value, dict) or set(value) != fields:
        raise FullTrackEvaluationError("trained cache binding schema is incomplete")
    if value.get("schema_version") != 1:
        raise FullTrackEvaluationError("trained cache binding schema version drift")
    ablations = value.get("ablations")
    if (
        not isinstance(ablations, list)
        or not ablations
        or len(ablations) != len(set(ablations))
        or any(item not in FUSION_ABLATIONS for item in ablations)
    ):
        raise FullTrackEvaluationError("trained cache ablations are invalid")
    models = value.get("models")
    if not isinstance(models, list) or not models:
        raise FullTrackEvaluationError("trained cache models are incomplete")

    method_ids: List[str] = []
    result_bindings: Dict[str, Dict[str, object]] = {}
    for index, model in enumerate(models):
        if not isinstance(model, dict) or set(model) != set(_TRAINED_CACHE_MODEL_FIELDS):
            raise FullTrackEvaluationError(
                f"trained cache model {index} schema is incomplete"
            )
        candidate_kind = model.get("candidate_kind")
        seed = model.get("seed")
        fold_index = model.get("fold_index")
        if candidate_kind not in FUSION_CANDIDATE_KINDS:
            raise FullTrackEvaluationError("trained cache candidate kind is invalid")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise FullTrackEvaluationError("trained cache seed is invalid")
        if fold_index != fold:
            raise FullTrackEvaluationError("trained cache fold identity drift")
        for field in (
            "report_sha256",
            "model_artifact_sha256",
            "model_json_sha256",
            "weights_npz_sha256",
            "source_fingerprint",
            "store_binding_sha256",
            "training_config_sha256",
            "job_config_sha256",
        ):
            _validate_sha256_hex(model.get(field), f"trained cache {field}")
        for field in ("maxsim_budget", "embedding_dim"):
            item = model.get(field)
            if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
                raise FullTrackEvaluationError(
                    f"trained cache {field} must be a positive integer"
                )
        for ablation in ablations:
            method_id = _trained_method_id(candidate_kind, seed, ablation)
            if method_id in result_bindings:
                raise FullTrackEvaluationError(
                    f"duplicate trained cache method ID: {method_id}"
                )
            method_ids.append(method_id)
            result_bindings[method_id] = _trained_result_model_binding(
                model, ablation
            )
    return tuple(method_ids), result_bindings


def _validate_trained_result_metadata(
    report: Mapping[str, object],
    *,
    fold: int,
    expected_binding: Optional[Mapping[str, object]],
) -> Tuple[str, ...]:
    optional_fields = {
        "trained_methods",
        "trained_model_bindings",
        "trained_paired_deltas",
    }
    present = set(report).intersection(optional_fields)
    if present and present != optional_fields:
        raise FullTrackEvaluationError("fold result has partial trained fields")

    expected_trained = None
    if expected_binding is not None:
        expected_trained = expected_binding.get("trained_evaluation")
        if (expected_trained is None) != (not present):
            raise FullTrackEvaluationError(
                "trained result does not match benchmark cache binding"
            )
    if not present:
        return ()

    raw_methods = report.get("trained_methods")
    if (
        not isinstance(raw_methods, list)
        or not raw_methods
        or any(not isinstance(method, str) or not method for method in raw_methods)
        or len(raw_methods) != len(set(raw_methods))
        or set(raw_methods).intersection(METHODS)
    ):
        raise FullTrackEvaluationError("trained_methods are invalid")
    model_bindings = report.get("trained_model_bindings")
    paired_deltas = report.get("trained_paired_deltas")
    if (
        not isinstance(model_bindings, dict)
        or set(model_bindings) != set(raw_methods)
        or not isinstance(paired_deltas, dict)
        or set(paired_deltas) != set(raw_methods)
    ):
        raise FullTrackEvaluationError("trained result method alignment is incomplete")

    store = report.get("store")
    if not isinstance(store, dict):
        raise FullTrackEvaluationError("trained result store binding is invalid")
    store_binding_sha256 = stable_json_sha256(store)
    for method in raw_methods:
        details = model_bindings.get(method)
        if (
            not isinstance(details, dict)
            or set(details) != set(_TRAINED_RESULT_MODEL_FIELDS)
        ):
            raise FullTrackEvaluationError(
                f"trained result binding schema is incomplete for {method}"
            )
        candidate_kind = details.get("candidate_kind")
        seed = details.get("seed")
        ablation = details.get("ablation")
        if (
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed < 0
            or not isinstance(candidate_kind, str)
            or not isinstance(ablation, str)
            or method != _trained_method_id(candidate_kind, seed, ablation)
        ):
            raise FullTrackEvaluationError(
                f"trained result method identity drift for {method}"
            )
        if details.get("fold_index") != fold or details.get("promoted") is not False:
            raise FullTrackEvaluationError(
                f"trained result fold/promotion identity drift for {method}"
            )
        for field in (
            "model_artifact_sha256",
            "model_json_sha256",
            "weights_npz_sha256",
            "report_sha256",
            "source_fingerprint",
            "store_binding_sha256",
            "training_config_sha256",
            "job_config_sha256",
        ):
            _validate_sha256_hex(details.get(field), f"trained result {field}")
        if (
            details.get("source_fingerprint") != report.get("source_fingerprint")
            or details.get("store_binding_sha256") != store_binding_sha256
        ):
            raise FullTrackEvaluationError(
                f"trained result source/store identity drift for {method}"
            )
        for field in ("maxsim_budget", "embedding_dim"):
            item = details.get(field)
            if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
                raise FullTrackEvaluationError(
                    f"trained result {field} is invalid for {method}"
                )
        if (
            details["embedding_dim"] != store.get("embedding_dim")
            or details["maxsim_budget"] > store.get("repetition_sections", 0)
            or details["maxsim_budget"] > store.get("salient_sections", 0)
        ):
            raise FullTrackEvaluationError(
                f"trained result model/store dimensions drift for {method}"
            )

    if expected_trained is not None:
        expected_methods, expected_model_bindings = _expected_trained_result_metadata(
            expected_trained, fold=fold
        )
        if tuple(raw_methods) != expected_methods or model_bindings != expected_model_bindings:
            raise FullTrackEvaluationError(
                "trained result model identity differs from benchmark cache binding"
            )
    return tuple(raw_methods)


def _validate_trained_paired_deltas(
    report: Mapping[str, object],
    query_map: Mapping[int, Mapping[str, Mapping[str, float]]],
    *,
    methods: Sequence[str],
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> None:
    if not methods:
        return
    expected: Dict[str, object] = {}
    for method in methods:
        expected[method] = {
            "paired_candidate_minus_global": {
                metric: _paired_bootstrap_delta(
                    [query_map[key]["global_cosine"][metric] for key in query_map],
                    [query_map[key][method][metric] for key in query_map],
                    iterations=bootstrap_iterations,
                    seed=bootstrap_seed,
                )
                for metric in METRICS
            },
            "paired_candidate_minus_frozen_hybrid": {
                metric: _paired_bootstrap_delta(
                    [query_map[key]["hybrid"][metric] for key in query_map],
                    [query_map[key][method][metric] for key in query_map],
                    iterations=bootstrap_iterations,
                    seed=bootstrap_seed,
                )
                for metric in METRICS
            },
        }
    if report.get("trained_paired_deltas") != expected:
        raise FullTrackEvaluationError("trained paired comparisons are invalid")


def _validate_aggregation_envelope(
    report: Mapping[str, object],
    *,
    expected_slot: Tuple[int, int],
    bootstrap_iterations: int,
    bootstrap_seed: int,
    expected_metadata: Optional[Mapping[str, object]] = None,
    expected_binding: Optional[Mapping[str, object]] = None,
    require_uncertainty: bool = False,
) -> Dict[int, Dict[str, Dict[str, float]]]:
    result_fields = {
        "schema_version",
        "evidence_scope",
        "dataset",
        "lawful_use",
        "source_fingerprint",
        "store",
        "protocol",
        "queries",
        "skipped_no_relevant",
        "candidate_tracks",
        "query_records",
        "aggregate",
        "per_scene",
        "per_tag",
        "grouped_metrics_notice",
        "resources",
    }
    if not isinstance(report, Mapping):
        raise FullTrackEvaluationError("fold result schema is incomplete")
    report_keys = set(report)
    trained_optional = {
        "trained_methods",
        "trained_model_bindings",
        "trained_paired_deltas",
    }
    if report_keys - trained_optional != result_fields:
        raise FullTrackEvaluationError("fold result schema is incomplete")
    fold, budget = _validated_report_key(report)
    trained_methods = _validate_trained_result_metadata(
        report, fold=fold, expected_binding=expected_binding
    )
    active_methods_list = list(METHODS) + list(trained_methods)
    if (fold, budget) != expected_slot:
        raise FullTrackEvaluationError(
            "fold/budget identifiers do not match benchmark matrix slot "
            f"{expected_slot}; found {(fold, budget)}"
        )
    if (
        report.get("schema_version") != EVALUATION_SCHEMA_VERSION
        or report.get("evidence_scope") != EVIDENCE_SCOPE
        or report.get("dataset") != DATASET_DESCRIPTION
        or report.get("lawful_use") != LAWFUL_USE
    ):
        raise FullTrackEvaluationError(
            f"fold {fold}, budget {budget} evaluation schema/descriptors drift"
        )
    source_fingerprint = report.get("source_fingerprint")
    if not isinstance(source_fingerprint, str) or not source_fingerprint:
        raise FullTrackEvaluationError("fold result has an invalid source fingerprint")
    store = _validate_store_binding(report.get("store"))
    if store["source_fingerprint"] != source_fingerprint:
        raise FullTrackEvaluationError(
            "fold result source fingerprint/store binding disagree"
        )
    candidate_tracks = report.get("candidate_tracks")
    if (
        isinstance(candidate_tracks, bool)
        or not isinstance(candidate_tracks, int)
        or candidate_tracks <= 0
    ):
        raise FullTrackEvaluationError("fold result has an invalid candidate count")
    protocol = report.get("protocol")
    if not isinstance(protocol, dict):
        raise FullTrackEvaluationError("fold result has no protocol object")
    config_fields = set(EvaluationConfig.__dataclass_fields__)
    try:
        config = EvaluationConfig(
            **{field: protocol[field] for field in config_fields}
        )
        config.validate()
    except (KeyError, TypeError, ValueError) as exc:
        raise FullTrackEvaluationError(
            "fold result has incomplete evaluation configuration"
        ) from exc
    if (
        config.fold_index != fold
        or config.maxsim_budget != budget
        or config.part != "test"
        or config.query_limit is not None
    ):
        raise FullTrackEvaluationError(
            f"fold {fold}, budget {budget} is not the official test protocol"
        )
    if (
        config.bootstrap_iterations != bootstrap_iterations
        or config.bootstrap_seed != bootstrap_seed
    ):
        raise FullTrackEvaluationError(
            f"fold {fold}, budget {budget} bootstrap protocol drift"
        )
    effective = _validate_effective_section_metadata(
        protocol.get("effective_unique_section_limits"),
        store=store,
        budget=budget,
        candidate_tracks=candidate_tracks,
    )
    query_descriptor_hash = protocol.get("query_descriptor_sha256")
    if (
        not isinstance(query_descriptor_hash, str)
        or len(query_descriptor_hash) != 64
        or any(
            character not in "0123456789abcdef"
            for character in query_descriptor_hash
        )
    ):
        raise FullTrackEvaluationError("query descriptor hash is invalid")
    expected_protocol = _evaluation_protocol(
        config,
        effective,
        query_descriptor_sha256=query_descriptor_hash,
    )
    if protocol != expected_protocol:
        raise FullTrackEvaluationError(
            f"fold {fold}, budget {budget} descriptive/method protocol drift"
        )
    if expected_metadata is not None and any(
        report.get(key) != value for key, value in expected_metadata.items()
    ):
        raise FullTrackEvaluationError(
            f"fold {fold}, budget {budget} differs from current input metadata"
        )
    query_map = _validate_fold_result_consistency(
        report, require_uncertainty=require_uncertainty,
        methods=active_methods_list,
    )
    _validate_trained_paired_deltas(
        report,
        query_map,
        methods=trained_methods,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )
    records = report["query_records"]
    descriptors = [
        {
            key: record[key]
            for key in ("track_id", "artist_id", "tags", "relevant_candidates")
        }
        for record in records
    ]
    if any(
        descriptor["relevant_candidates"] >= candidate_tracks
        for descriptor in descriptors
    ):
        raise FullTrackEvaluationError(
            f"fold {fold}, budget {budget} relevance count exceeds candidate pool"
        )
    skipped = report.get("skipped_no_relevant")
    if _query_descriptor_sha256(descriptors, skipped) != query_descriptor_hash:
        raise FullTrackEvaluationError(
            f"fold {fold}, budget {budget} query descriptor binding drift"
        )
    if (
        report.get("per_scene") != _grouped_metrics(records, "scene", methods=active_methods_list)
        or report.get("per_tag") != _grouped_metrics(records, "tag", methods=active_methods_list)
        or report.get("grouped_metrics_notice") != GROUPED_METRICS_NOTICE
    ):
        raise FullTrackEvaluationError(
            f"fold {fold}, budget {budget} grouped result metadata drift"
        )
    _validate_resource_metadata(report.get("resources"), methods=active_methods_list)
    return query_map


def _protocol_invariant(protocol: Mapping[str, object]) -> Dict[str, object]:
    return {
        key: value
        for key, value in protocol.items()
        if key
        not in {
            "fold_index",
            "maxsim_budget",
            "effective_unique_section_limits",
            "query_descriptor_sha256",
        }
    }


def aggregate_all_fold_results(
    reports: Sequence[Mapping[str, object]],
    *,
    bootstrap_iterations: int,
    bootstrap_seed: int,
    expected_metadata_by_slot: Optional[
        Mapping[Tuple[int, int], Mapping[str, object]]
    ] = None,
) -> Mapping[str, object]:
    """Aggregate all official fold/budget results with strict query alignment."""
    if bootstrap_iterations <= 0:
        raise FullTrackEvaluationError("bootstrap iterations must be positive")
    if bootstrap_seed < 0:
        raise FullTrackEvaluationError("bootstrap seed must be non-negative")
    expected_slots = tuple(
        (fold, budget) for fold in OFFICIAL_FOLDS for budget in OFFICIAL_BUDGETS
    )
    if len(reports) != len(expected_slots):
        raise FullTrackEvaluationError(
            f"all-fold benchmark requires {len(expected_slots)} ordered results"
        )
    if (
        expected_metadata_by_slot is not None
        and set(expected_metadata_by_slot) != set(expected_slots)
    ):
        raise FullTrackEvaluationError("current-input metadata matrix is incomplete")
    keyed: Dict[Tuple[int, int], Mapping[str, object]] = {}
    query_maps: Dict[Tuple[int, int], Dict[int, Dict[str, Dict[str, float]]]] = {}
    reference_source: Optional[object] = None
    reference_store: Optional[object] = None
    reference_protocol: Optional[object] = None
    for expected_slot, report in zip(expected_slots, reports):
        expected_metadata = (
            expected_metadata_by_slot.get(expected_slot)
            if expected_metadata_by_slot is not None
            else None
        )
        query_map = _validate_aggregation_envelope(
            report,
            expected_slot=expected_slot,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_seed=bootstrap_seed,
            expected_metadata=expected_metadata,
        )
        protocol = report["protocol"]
        invariant_protocol = _protocol_invariant(protocol)
        if reference_source is None:
            reference_source = report["source_fingerprint"]
            reference_store = report["store"]
            reference_protocol = invariant_protocol
        elif (
            report["source_fingerprint"] != reference_source
            or report["store"] != reference_store
            or invariant_protocol != reference_protocol
        ):
            raise FullTrackEvaluationError(
                "source/protocol/store invariants differ across benchmark runs"
            )
        keyed[expected_slot] = report
        query_maps[expected_slot] = query_map

    reference_trained_methods = keyed[expected_slots[0]].get("trained_methods", [])
    for slot in expected_slots[1:]:
        if keyed[slot].get("trained_methods", []) != reference_trained_methods:
            raise FullTrackEvaluationError(
                "trained method identity drift across benchmark runs"
            )

    for fold in OFFICIAL_FOLDS:
        reference = set(query_maps[(fold, OFFICIAL_BUDGETS[0])])
        reference_candidate_tracks = keyed[(fold, OFFICIAL_BUDGETS[0])][
            "candidate_tracks"
        ]
        reference_query_descriptor_hash = keyed[
            (fold, OFFICIAL_BUDGETS[0])
        ]["protocol"]["query_descriptor_sha256"]
        reference_trained_bindings = keyed[(fold, OFFICIAL_BUDGETS[0])].get(
            "trained_model_bindings"
        )
        reference_diversity: Dict[str, object] = {}
        previous_repeating = {"repeated_sections": -1, "salient_sections": -1}
        for budget in OFFICIAL_BUDGETS[1:]:
            if set(query_maps[(fold, budget)]) != reference:
                raise FullTrackEvaluationError(
                    f"query alignment drift across budgets for fold {fold}"
                )
            if keyed[(fold, budget)]["candidate_tracks"] != reference_candidate_tracks:
                raise FullTrackEvaluationError(
                    f"candidate partition drift across budgets for fold {fold}"
                )
            if (
                keyed[(fold, budget)]["protocol"]["query_descriptor_sha256"]
                != reference_query_descriptor_hash
            ):
                raise FullTrackEvaluationError(
                    f"query descriptor drift across budgets for fold {fold}"
                )
            if (
                keyed[(fold, budget)].get("trained_model_bindings")
                != reference_trained_bindings
            ):
                raise FullTrackEvaluationError(
                    f"trained model identity drift across budgets for fold {fold}"
                )
        for budget in OFFICIAL_BUDGETS:
            effective = keyed[(fold, budget)]["protocol"][
                "effective_unique_section_limits"
            ]
            for stream in ("repeated_sections", "salient_sections"):
                details = effective[stream]
                diversity_base = {
                    key: value
                    for key, value in details.items()
                    if key
                    not in {
                        "requested_budget",
                        "tracks_repeating_for_requested_budget",
                    }
                }
                if stream not in reference_diversity:
                    reference_diversity[stream] = diversity_base
                elif diversity_base != reference_diversity[stream]:
                    raise FullTrackEvaluationError(
                        f"effective section diversity drift across budgets for "
                        f"fold {fold}/{stream}"
                    )
                repeating = details["tracks_repeating_for_requested_budget"]
                if repeating < previous_repeating[stream]:
                    raise FullTrackEvaluationError(
                        f"effective section repetition is not monotonic for "
                        f"fold {fold}/{stream}"
                    )
                previous_repeating[stream] = repeating

    by_budget: Dict[str, object] = {}
    for budget in OFFICIAL_BUDGETS:
        per_fold: Dict[str, object] = {}
        active_agg = list(METHODS)
        first = keyed[(OFFICIAL_FOLDS[0], budget)]
        if "trained_methods" in first:
            active_agg = list(METHODS) + list(first["trained_methods"])
        pooled: Dict[str, Dict[str, List[float]]] = {
            method: {metric: [] for metric in METRICS} for method in active_agg
        }
        fold_metric_values: Dict[str, Dict[str, List[float]]] = {
            method: {metric: [] for metric in METRICS} for method in active_agg
        }
        pooled_query_keys: List[str] = []
        for fold in OFFICIAL_FOLDS:
            report = keyed[(fold, budget)]
            aggregate = report.get("aggregate")
            if not isinstance(aggregate, dict) or set(aggregate) != set(active_agg):
                raise FullTrackEvaluationError(
                    f"fold {fold}, budget {budget} has incomplete aggregate methods"
                )
            fold_methods: Dict[str, Dict[str, float]] = {}
            for method in active_agg:
                method_result = aggregate.get(method)
                if not isinstance(method_result, dict):
                    raise FullTrackEvaluationError("method aggregate must be an object")
                raw_metrics = method_result.get("metrics")
                if not isinstance(raw_metrics, dict) or set(raw_metrics) != set(METRICS):
                    raise FullTrackEvaluationError("method aggregate metrics are incomplete")
                fold_methods[method] = {
                    metric: float(raw_metrics[metric]) for metric in METRICS
                }
                for metric in METRICS:
                    fold_metric_values[method][metric].append(
                        fold_methods[method][metric]
                    )
            records = query_maps[(fold, budget)]
            for track_id in sorted(records):
                pooled_query_keys.append(f"{fold}:{track_id}")
                for method in active_agg:
                    for metric in METRICS:
                        pooled[method][metric].append(
                            records[track_id][method][metric]
                        )
            skipped = report.get("skipped_no_relevant", 0)
            if isinstance(skipped, bool) or not isinstance(skipped, int) or skipped < 0:
                raise FullTrackEvaluationError("invalid skipped-query count")
            per_fold[str(fold)] = {
                "queries": len(records),
                "skipped_no_relevant": skipped,
                "methods": fold_methods,
            }

        fold_macro = {
            method: {
                metric: float(np.mean(fold_metric_values[method][metric]))
                for metric in METRICS
            }
            for method in active_agg
        }
        query_weighted: Dict[str, object] = {}
        for method_index, method in enumerate(active_agg):
            metrics = {
                metric: float(np.mean(pooled[method][metric])) for metric in METRICS
            }
            comparisons = {
                metric: _paired_bootstrap_delta(
                    pooled["global_cosine"][metric],
                    pooled[method][metric],
                    iterations=bootstrap_iterations,
                    seed=(
                        bootstrap_seed
                        + budget * 100
                        + method_index * len(METRICS)
                        + metric_index
                    ),
                )
                for metric_index, metric in enumerate(METRICS)
            }
            query_weighted[method] = {
                "metrics": metrics,
                "paired_method_minus_global": comparisons,
            }
        by_budget[str(budget)] = {
            "per_fold": per_fold,
            "fold_macro": {
                "definition": "unweighted mean of the five official fold metrics",
                "methods": fold_macro,
            },
            "query_weighted": {
                "definition": (
                    "mean over pooled per-query observations; paired uncertainty "
                    "resamples aligned (fold, track_id) observations"
                ),
                "queries": len(pooled_query_keys),
                "aligned_query_keys_sha256": stable_json_sha256(
                    {"keys": pooled_query_keys}
                ),
                "methods": query_weighted,
            },
        }

    return {
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "source_fingerprint": keyed[(OFFICIAL_FOLDS[0], OFFICIAL_BUDGETS[0])][
            "source_fingerprint"
        ],
        "store_binding": keyed[(OFFICIAL_FOLDS[0], OFFICIAL_BUDGETS[0])]["store"],
        "protocol_invariants": _protocol_invariant(
            keyed[(OFFICIAL_FOLDS[0], OFFICIAL_BUDGETS[0])]["protocol"]
        ),
        "folds": list(OFFICIAL_FOLDS),
        "budgets": list(OFFICIAL_BUDGETS),
        "metrics": dict(
            keyed[(OFFICIAL_FOLDS[0], OFFICIAL_BUDGETS[0])]["protocol"][
                "metric_labels"
            ]
        ),
        "bootstrap": {
            "method": "pooled paired bootstrap over aligned per-query observations",
            "iterations": bootstrap_iterations,
            "base_seed": bootstrap_seed,
        },
        "by_budget": by_budget,
        "multiple_comparisons_notice": (
            "All method-minus-global and budget comparisons are descriptive; no "
            "multiple-comparison correction or confirmatory significance claim is made."
        ),
    }


def run_all_folds_benchmark(
    context: JamendoContext,
    reader: FullTrackStoreReader,
    *,
    output_dir: Path,
    base_config: EvaluationConfig = EvaluationConfig(),
    trained_root: Optional[Path] = None,
    trained_candidates: Optional[Sequence[str]] = None,
    trained_seeds: Optional[Sequence[int]] = None,
    include_ablations: bool = True,
    worker_fold: Optional[int] = None,
    selection_budget: int = 8,
    selection_primary_metric: str = "recall_at_k",
    selection_list_id: str = "fulltrack-trained-candidates-v1",
    selection_stability_threshold: float = 0.05,
) -> Tuple[Mapping[str, object], Mapping[str, int]]:
    """Run/resume the fixed five-fold by three-budget official benchmark matrix."""
    base_config.validate()
    loaded_folds = tuple(sorted(fold.index for fold in context.folds))
    if loaded_folds != OFFICIAL_FOLDS:
        raise FullTrackEvaluationError(
            f"all-fold benchmark requires official folds {OFFICIAL_FOLDS}, "
            f"found {loaded_folds}"
        )
    if base_config.part != "test":
        raise FullTrackEvaluationError(
            "official all-fold benchmark requires part='test'"
        )
    if base_config.query_limit is not None:
        raise FullTrackEvaluationError("official all-fold benchmark forbids query limits")
    if worker_fold is not None and (
        isinstance(worker_fold, bool)
        or not isinstance(worker_fold, int)
        or worker_fold not in OFFICIAL_FOLDS
    ):
        raise FullTrackEvaluationError(
            f"worker fold must be one of {OFFICIAL_FOLDS}"
        )
    benchmark_folds = (
        OFFICIAL_FOLDS if worker_fold is None else (worker_fold,)
    )
    required_budget = max(OFFICIAL_BUDGETS)
    if (
        reader.binding.repetition_sections < required_budget
        or reader.binding.salient_sections < required_budget
    ):
        raise FullTrackEvaluationError(
            "all-fold benchmark requires store-declared repeated and salient "
            f"section budgets >= {required_budget}; found "
            f"{reader.binding.repetition_sections} and "
            f"{reader.binding.salient_sections}"
        )
    requested_output_dir = Path(output_dir)
    if requested_output_dir.is_symlink():
        raise FullTrackEvaluationError("benchmark output directory may not be a symlink")
    output_dir = _safe_output_path(
        requested_output_dir / "benchmark-summary.json"
    ).parent

    fold_track_ids: Dict[int, Tuple[int, ...]] = {}
    fold_tracks: Dict[int, Tuple[JamendoTrack, ...]] = {}
    fold_by_index = {fold.index: fold for fold in context.folds}
    for fold in context.folds:
        selected_tracks = tuple(
            track
            for track in context.tracks
            if fold.track_parts.get(track.track_id) == "test"
        )
        track_ids = tuple(track.track_id for track in selected_tracks)
        if len(track_ids) < 2:
            raise FullTrackEvaluationError(
                f"official test partition for fold {fold.index} is too small"
            )
        for track_id in track_ids:
            _validated_tags(
                fold.track_tags.get(track_id),
                where=f"fold {fold.index} track {track_id}",
            )
        fold_track_ids[fold.index] = track_ids
        fold_tracks[fold.index] = selected_tracks
    required_track_ids = {
        track_id for track_ids in fold_track_ids.values() for track_id in track_ids
    }
    if not required_track_ids.issubset(set(reader.track_ids)):
        raise FullTrackEvaluationError(
            "store does not cover every official test partition"
        )
    section_counts_by_track: Dict[int, Tuple[int, int]] = {}
    for track_id in reader.track_ids:
        if track_id in required_track_ids:
            stored = reader.read_track(track_id)
            section_counts_by_track[track_id] = (
                len(stored.repeated_indices),
                len(stored.salient_indices),
            )
    declared_sections = {
        "repeated_sections": reader.binding.repetition_sections,
        "salient_sections": reader.binding.salient_sections,
    }
    expected_metadata_by_slot: Dict[
        Tuple[int, int], Mapping[str, object]
    ] = {}
    for fold in OFFICIAL_FOLDS:
        track_ids = fold_track_ids[fold]
        query_descriptor_hash = _expected_query_descriptor_sha256(
            fold_by_index[fold],
            fold_tracks[fold],
            replace(base_config, fold_index=fold),
        )
        selected_counts = {
            "repeated_sections": [
                section_counts_by_track[track_id][0] for track_id in track_ids
            ],
            "salient_sections": [
                section_counts_by_track[track_id][1] for track_id in track_ids
            ],
        }
        for budget in OFFICIAL_BUDGETS:
            config = replace(
                base_config, fold_index=fold, maxsim_budget=budget, query_limit=None
            )
            effective = _effective_unique_section_limits(
                selected_counts, declared_sections, budget=budget
            )
            expected_metadata_by_slot[(fold, budget)] = _expected_result_metadata(
                context,
                reader,
                config,
                effective,
                candidate_tracks=len(track_ids),
                query_descriptor_sha256=query_descriptor_hash,
            )

    # --- Load trained models if --trained-root supplied ---
    trained_bindings_by_fold: Dict[int, List[_TrainedModelBinding]] = {}
    if trained_root is not None:
        from .fulltrack_train import DEFAULT_SEEDS
        from .fulltrack_fusion import CANDIDATE_KINDS as FUSION_CANDIDATE_KINDS
        actual_candidates = list(trained_candidates or FUSION_CANDIDATE_KINDS)
        actual_seeds = list(trained_seeds or DEFAULT_SEEDS)
        if (
            not actual_candidates
            or len(actual_candidates) != len(set(actual_candidates))
            or any(item not in FUSION_CANDIDATE_KINDS for item in actual_candidates)
        ):
            raise FullTrackEvaluationError(
                "trained candidates must be distinct supported fusion kinds"
            )
        if (
            not actual_seeds
            or len(actual_seeds) != len(set(actual_seeds))
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in actual_seeds
            )
        ):
            raise FullTrackEvaluationError(
                "trained seeds must be distinct non-negative integers"
            )
        src_fp = context.source_fingerprint
        sb_sha = stable_json_sha256(_evaluation_store_binding(reader))
        store_edim = int(reader.binding.embedding_dim)
        store_rep = int(reader.binding.repetition_sections)
        store_sal = int(reader.binding.salient_sections)
        for fold in benchmark_folds:
            bindings: List[_TrainedModelBinding] = []
            for ck in actual_candidates:
                for sd in actual_seeds:
                    tb = load_trained_model_for_fold(
                        Path(trained_root),
                        fold_index=fold, candidate_kind=ck, seed=sd,
                        expected_source_fingerprint=src_fp,
                        expected_store_binding_sha256=sb_sha,
                        store_embedding_dim=store_edim,
                        store_repetition_sections=store_rep,
                        store_salient_sections=store_sal,
                    )
                    bindings.append(tb)
            trained_bindings_by_fold[fold] = bindings

    reports: List[Mapping[str, object]] = []
    artifacts: List[Mapping[str, object]] = []
    computed = 0
    reused = 0
    for fold in benchmark_folds:
        for budget in OFFICIAL_BUDGETS:
            config = replace(
                base_config, fold_index=fold, maxsim_budget=budget, query_limit=None
            )
            tb_fold = trained_bindings_by_fold.get(fold)
            binding = _benchmark_result_binding(
                context,
                reader,
                config,
                trained_bindings=tb_fold,
                include_ablations=include_ablations,
            )
            if trained_root is not None:
                path = _safe_output_path(output_dir / f"fold-{fold}_budget-{budget}_trained.json")
            else:
                path = _benchmark_result_path(output_dir, fold, budget)
            expected_metadata = expected_metadata_by_slot[(fold, budget)]
            result = _load_valid_benchmark_result(
                path, binding, expected_metadata
            )
            if result is None:
                result = evaluate_jamendo(
                    context,
                    reader,
                    config=config,
                    expected_query_descriptor_hash=expected_metadata["protocol"][
                        "query_descriptor_sha256"
                    ],
                    trained_bindings=tb_fold if tb_fold else None,
                    include_ablations=include_ablations,
                )
                _write_benchmark_result(path, binding, result)
                computed += 1
            else:
                reused += 1
            reports.append(result)
            artifacts.append(
                {
                    "fold": fold,
                    "budget": budget,
                    "file": path.name,
                    "sha256": sha256_path(path),
                    "binding_sha256": stable_json_sha256(binding),
                }
            )

    if worker_fold is not None:
        worker_result: Dict[str, object] = {
            "schema_version": 1,
            "artifact_kind": "fulltrack_fold_benchmark_worker",
            "source_fingerprint": context.source_fingerprint,
            "fold": worker_fold,
            "result_artifacts": artifacts,
        }
        worker_result["worker_payload_sha256"] = stable_json_sha256(
            worker_result
        )
        return worker_result, {"computed": computed, "reused": reused}

    aggregate = aggregate_all_fold_results(
        reports,
        bootstrap_iterations=base_config.bootstrap_iterations,
        bootstrap_seed=base_config.bootstrap_seed,
        expected_metadata_by_slot=expected_metadata_by_slot,
    )
    benchmark_config = asdict(base_config)
    benchmark_config.update(
        {
            "fold_index": list(OFFICIAL_FOLDS),
            "maxsim_budget": list(OFFICIAL_BUDGETS),
            "query_limit": None,
        }
    )
    if trained_root is not None:
        benchmark_config["selection"] = {
            "deciding_budget": selection_budget,
            "primary_metric": selection_primary_metric,
            "list_id": selection_list_id,
            "cross_seed_stability_threshold": selection_stability_threshold,
        }
    summary: Dict[str, object] = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "evidence_scope": EVIDENCE_SCOPE,
        "artifact_kind": "fulltrack_all_fold_benchmark",
        "source_fingerprint": context.source_fingerprint,
        "store_binding": _evaluation_store_binding(reader),
        "benchmark_config": benchmark_config,
        "result_artifacts": artifacts,
        "aggregate": aggregate,
    }
    if trained_root is not None:
        trained_summary_entries: List[Dict[str, object]] = []
        for fold in OFFICIAL_FOLDS:
            for tb in trained_bindings_by_fold.get(fold, []):
                trained_summary_entries.append({
                    "schema_version": 1,
                    "artifact_kind": "fulltrack_trained_candidate_identity",
                    "candidate_kind": tb.candidate_kind,
                    "fold": tb.fold_index,
                    "seed": tb.seed,
                    "model_artifact_sha256": tb.model_artifact_sha256,
                    "evaluation_identity": {
                        "source_fingerprint": tb.source_fingerprint,
                        "store_binding_sha256": tb.store_binding_sha256,
                    },
                    "evaluation_identity_sha256": stable_json_sha256({
                        "source_fingerprint": tb.source_fingerprint,
                        "store_binding_sha256": tb.store_binding_sha256,
                    }),
                    "promoted": False,
                })
        summary["trained_cross_seed_entries"] = trained_summary_entries
        from .fulltrack_selection import (
            FullTrackSelectionError,
            build_selection_inputs,
            write_selection_inputs,
        )

        try:
            candidate_list, candidate_evaluations = build_selection_inputs(
                reports,
                deciding_budget=selection_budget,
                primary_metric=selection_primary_metric,
                list_id=selection_list_id,
                stability_threshold=selection_stability_threshold,
            )
            selection_manifest = write_selection_inputs(
                output_dir / "selection", candidate_list, candidate_evaluations
            )
        except (FullTrackSelectionError, OSError) as exc:
            raise FullTrackEvaluationError(
                f"selector export failed: {exc}"
            ) from exc
        summary["selection_inputs"] = {
            "directory": "selection",
            **selection_manifest,
        }
        summary_path = output_dir / "benchmark-summary-trained.json"
    else:
        summary_path = output_dir / "benchmark-summary.json"
    summary["summary_payload_sha256"] = stable_json_sha256(summary)
    write_evaluation_report(summary_path, summary)
    return summary, {"computed": computed, "reused": reused}


def _path_components(path: Path) -> Tuple[str, ...]:
    return tuple(part.casefold() for part in path.parts)


def assert_not_signed_protocol_path(path: Path) -> None:
    """Fail before opening or writing any signed protocol-v6 path."""
    absolute = Path(path).absolute()
    components = _path_components(absolute)
    if "protocol-v6" in components:
        raise FullTrackEvaluationError(
            "signed protocol-v6 state is protected and must never be opened"
        )
    protected_names = {
        "state.json",
        "frozen-state.json",
        "final-test-manifest.json",
        "frozen-baseline-rankings.json",
        "winner-rankings.json",
    }
    if ".goals" in components and absolute.name.casefold() in protected_names:
        raise FullTrackEvaluationError("signed protocol state path is protected")


def load_commercial_v6_replay(path: Path) -> Mapping[str, object]:
    """Read an exported commercial replay report without touching signed state."""
    path = Path(path)
    assert_not_signed_protocol_path(path)
    resolved = path.resolve(strict=True)
    assert_not_signed_protocol_path(resolved)
    if path.is_symlink() or not resolved.is_file() or resolved.suffix.casefold() != ".json":
        raise FullTrackEvaluationError("commercial replay must be a concrete JSON file")
    flags = os.O_RDONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor = os.open(resolved, flags)
    try:
        before = os.fstat(descriptor)
        raw = bytearray()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            raw.extend(block)
            if len(raw) > 100 * 1024 * 1024:
                raise FullTrackEvaluationError("commercial replay exceeds 100 MiB bound")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise FullTrackEvaluationError("commercial replay changed during read")
    try:
        value = json.loads(bytes(raw).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FullTrackEvaluationError(f"invalid commercial replay JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise FullTrackEvaluationError("commercial replay must contain an object")
    if value.get("evidence_scope") != COMMERCIAL_EVIDENCE_SCOPE:
        raise FullTrackEvaluationError(
            f"commercial replay must be labelled {COMMERCIAL_EVIDENCE_SCOPE!r}"
        )
    if value.get("benchmark_version") != "v6":
        raise FullTrackEvaluationError("commercial replay must explicitly identify v6")
    return value


def _safe_output_path(path: Path) -> Path:
    """Resolve existing ancestry before any write and reject protected destinations."""
    absolute = Path(path).absolute()
    assert_not_signed_protocol_path(absolute)
    ancestor = absolute.parent
    while not ancestor.exists():
        parent = ancestor.parent
        if parent == ancestor:
            raise FullTrackEvaluationError("cannot resolve benchmark output ancestry")
        ancestor = parent
    try:
        resolved_ancestor = ancestor.resolve(strict=True)
    except OSError as exc:
        raise FullTrackEvaluationError(f"cannot resolve output ancestry: {exc}") from exc
    assert_not_signed_protocol_path(resolved_ancestor)
    if absolute.exists():
        if absolute.is_symlink():
            raise FullTrackEvaluationError("evaluation output may not be a symlink")
        assert_not_signed_protocol_path(absolute.resolve(strict=True))
    absolute.parent.mkdir(parents=True, exist_ok=True)
    try:
        resolved_parent = absolute.parent.resolve(strict=True)
    except OSError as exc:
        raise FullTrackEvaluationError(f"cannot resolve output directory: {exc}") from exc
    assert_not_signed_protocol_path(resolved_parent)
    return resolved_parent / absolute.name


def write_evaluation_report(path: Path, report: Mapping[str, object]) -> None:
    if report.get("evidence_scope") not in (
        EVIDENCE_SCOPE,
        COMMERCIAL_EVIDENCE_SCOPE,
    ):
        raise FullTrackEvaluationError("report has an invalid evidence scope")
    path = _safe_output_path(Path(path))
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _evaluate_command(args: argparse.Namespace) -> int:
    context = load_jamendo_context(
        Path(args.metadata_root),
        Path(args.audio_root),
        Path(args.state_root),
        production=True,
    )
    config = EvaluationConfig(
        fold_index=args.fold,
        part=args.part,
        maxsim_budget=args.maxsim_budget,
        candidate_pool=args.candidate_pool,
        bootstrap_iterations=args.bootstrap_iterations,
        query_limit=args.query_limit,
        min_shared_tags=args.min_shared_tags,
        min_tag_jaccard=args.min_tag_jaccard,
    )
    with FullTrackStoreReader(
        Path(args.store),
        expected_source_fingerprint=context.source_fingerprint,
    ) as reader:
        report = evaluate_jamendo(context, reader, config=config)
    write_evaluation_report(Path(args.output), report)
    print(json.dumps({"output": args.output, "queries": report["queries"]}, indent=2))
    return 0


def _all_folds_command(args: argparse.Namespace) -> int:
    context = load_jamendo_context(
        Path(args.metadata_root),
        Path(args.audio_root),
        Path(args.state_root),
        production=True,
    )
    config = EvaluationConfig(
        part="test",
        candidate_pool=args.candidate_pool,
        recall_cutoff=args.recall_cutoff,
        ndcg_cutoff=args.ndcg_cutoff,
        bootstrap_iterations=args.bootstrap_iterations,
        bootstrap_seed=args.bootstrap_seed,
        max_feature_cache_bytes=args.max_feature_cache_bytes,
        min_shared_tags=args.min_shared_tags,
        min_tag_jaccard=args.min_tag_jaccard,
    )
    with FullTrackStoreReader(
        Path(args.store),
        expected_source_fingerprint=context.source_fingerprint,
    ) as reader:
        trained_root = Path(args.trained_root) if args.trained_root else None
        summary, resume = run_all_folds_benchmark(
            context,
            reader,
            output_dir=Path(args.output_dir),
            base_config=config,
            trained_root=trained_root,
            trained_candidates=args.trained_candidate,
            trained_seeds=args.trained_seed,
            include_ablations=not args.no_ablations,
            worker_fold=args.worker_fold,
            selection_budget=args.selection_budget,
            selection_primary_metric=args.selection_primary_metric,
            selection_list_id=args.selection_list_id,
            selection_stability_threshold=args.selection_stability_threshold,
        )
    if args.worker_fold is not None:
        print(
            json.dumps(
                {
                    "output_dir": str(Path(args.output_dir)),
                    "worker_fold": args.worker_fold,
                    "worker_payload_sha256": summary["worker_payload_sha256"],
                    **resume,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    summary_name = (
        "benchmark-summary-trained.json" if trained_root is not None
        else "benchmark-summary.json"
    )
    print(
        json.dumps(
            {
                "output": str(Path(args.output_dir) / summary_name),
                "summary_payload_sha256": summary["summary_payload_sha256"],
                **resume,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _commercial_command(args: argparse.Namespace) -> int:
    replay = load_commercial_v6_replay(Path(args.replay))
    print(
        json.dumps(
            {
                "evidence_scope": replay["evidence_scope"],
                "benchmark_version": replay["benchmark_version"],
                "read_only": True,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--metadata-root", required=True)
    evaluate.add_argument("--audio-root", required=True)
    evaluate.add_argument("--state-root", required=True)
    evaluate.add_argument("--store", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--fold", type=int, default=0)
    evaluate.add_argument(
        "--part", choices=("train", "validation", "test"), default="test"
    )
    evaluate.add_argument("--maxsim-budget", type=int, default=8)
    evaluate.add_argument("--candidate-pool", type=int, default=200)
    evaluate.add_argument("--bootstrap-iterations", type=int, default=2_000)
    evaluate.add_argument("--query-limit", type=int)
    evaluate.add_argument("--min-shared-tags", type=int, default=2)
    evaluate.add_argument("--min-tag-jaccard", type=float, default=0.25)
    evaluate.set_defaults(handler=_evaluate_command)

    all_folds = subparsers.add_parser(
        "benchmark-all", help="run/resume all five official folds at budgets 8, 16, and 32"
    )
    all_folds.add_argument("--metadata-root", required=True)
    all_folds.add_argument("--audio-root", required=True)
    all_folds.add_argument("--state-root", required=True)
    all_folds.add_argument("--store", required=True)
    all_folds.add_argument("--output-dir", required=True)
    all_folds.add_argument("--candidate-pool", type=int, default=200)
    all_folds.add_argument("--recall-cutoff", type=int, default=10)
    all_folds.add_argument("--ndcg-cutoff", type=int, default=10)
    all_folds.add_argument("--bootstrap-iterations", type=int, default=2_000)
    all_folds.add_argument("--bootstrap-seed", type=int, default=20260714)
    all_folds.add_argument(
        "--max-feature-cache-bytes", type=int, default=2 * 1024**3
    )
    all_folds.add_argument("--min-shared-tags", type=int, default=2)
    all_folds.add_argument("--min-tag-jaccard", type=float, default=0.25)
    all_folds.add_argument(
        "--trained-root", type=str, default=None,
        help="training matrix root with fold-N/candidate/seed-S/ layout",
    )
    all_folds.add_argument(
        "--trained-candidate",
        action="append",
        choices=(
            "nonnegative_linear",
            "monotonic_network",
            "channel_gated_embedding",
        ),
        default=None,
        help="repeatable: candidate kinds to evaluate (default: all three)",
    )
    all_folds.add_argument(
        "--trained-seed", action="append", type=int, default=None,
        help="repeatable: seeds to evaluate (default: trainer default seeds)",
    )
    all_folds.add_argument(
        "--no-ablations", action="store_true", default=False,
        help="disable ablation variants (global_only, no_sections)",
    )
    all_folds.add_argument(
        "--worker-fold",
        type=int,
        choices=OFFICIAL_FOLDS,
        default=None,
        help="compute one fold's reusable artifacts without aggregate finalization",
    )
    all_folds.add_argument(
        "--selection-budget",
        type=int,
        choices=OFFICIAL_BUDGETS,
        default=8,
        help="preregister the benchmark budget exported to model selection",
    )
    all_folds.add_argument(
        "--selection-primary-metric",
        choices=METRICS,
        default="recall_at_k",
        help="preregister the primary automated metric for model selection",
    )
    all_folds.add_argument(
        "--selection-list-id",
        default="fulltrack-trained-candidates-v1",
        help="stable identifier bound into the candidate list",
    )
    all_folds.add_argument(
        "--selection-stability-threshold",
        type=float,
        default=0.05,
        help="maximum cross-seed standard deviation (cannot exceed 0.05)",
    )
    all_folds.set_defaults(handler=_all_folds_command)

    commercial = subparsers.add_parser("commercial-v6-replay")
    commercial.add_argument("--replay", required=True)
    commercial.set_defaults(handler=_commercial_command)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except FullTrackEvaluationError as exc:
        raise SystemExit(f"full-track evaluation blocked: {exc}") from exc


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
