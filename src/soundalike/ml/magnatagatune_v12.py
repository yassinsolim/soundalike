"""Iteration-12 nested evaluation for all accepted MagnaTagATune triads.

This is deliberately a *multi-membership purged grouped cross-fit*, not
GroupKFold.  Triad communities are packed into test folds; any training triad
sharing a clip or artist with that fold is purged.  The same rule is used by
inner model selection, and every accepted row receives exactly one outer OOF
prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from . import magnatagatune_v10 as v10

SCHEMA_VERSION = 12
SEED = 20260712
OUTER_FOLDS = 5
INNER_FOLDS = 3
LOUVAIN_RESOLUTION = 0.7
EXPECTED_CONSTRAINTS = 307
EXPECTED_CLIPS = 611
STRATA = ("low", "medium", "high")
FIXED_NAMES = ("artist_supcon", "fma_supcon", "vibe_dsp", "clap")
FAMILY_ORDER = (
    "linear_triplet",
    "orthogonal_linear_triplet",
    "mlp_triplet",
    "smooth_linear",
    "fma_distilled_linear",
)


class MTATV12Error(v10.MTATError):
    """Raised when v12 coverage, leakage, resources, or caches are invalid."""


def confidence_stratum(value: float) -> str:
    """Return the fixed, predeclared vote-strength stratum."""
    if value < 0.25:
        return "low"
    if value < 0.5:
        return "medium"
    return "high"


def _record(item: Any) -> dict[str, Any]:
    value = asdict(item) if is_dataclass(item) else dict(item)
    value["source_row"] = int(value["source_row"])
    value["clip_ids"] = tuple(map(int, value["clip_ids"]))
    value["artists"] = tuple(str(x).strip().casefold() for x in value["artists"])
    value["confidence"] = float(value["confidence"])
    return value


def _entities(rows: Sequence[Mapping[str, Any]]) -> tuple[set[int], set[str]]:
    clips = {int(x) for row in rows for x in row["clip_ids"]}
    artists = {str(x).strip().casefold() for row in rows for x in row["artists"]}
    return clips, artists


def _assert_disjoint(
    train: Sequence[Mapping[str, Any]], test: Sequence[Mapping[str, Any]]
) -> None:
    train_clips, train_artists = _entities(train)
    test_clips, test_artists = _entities(test)
    clip_overlap = train_clips & test_clips
    artist_overlap = train_artists & test_artists
    if clip_overlap or artist_overlap:
        raise MTATV12Error(
            f"purge leakage: {len(clip_overlap)} clips, "
            f"{len(artist_overlap)} artists"
        )


def artist_graph_component_count(constraints: Sequence[Any]) -> int:
    """Count connected components in the raw artist co-occurrence graph."""
    rows = [_record(item) for item in constraints]
    artists = sorted({artist for row in rows for artist in row["artists"]})
    parent = {artist: artist for artist in artists}

    def find(artist: str) -> str:
        while parent[artist] != artist:
            parent[artist] = parent[parent[artist]]
            artist = parent[artist]
        return artist

    for row in rows:
        members = sorted(set(row["artists"]))
        for left, right in itertools.combinations(members, 2):
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[max(left_root, right_root)] = min(left_root, right_root)
    return len({find(artist) for artist in artists})


def _triad_communities(
    rows: Sequence[Mapping[str, Any]], *, seed: int
) -> list[list[int]]:
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover - ML extra
        raise MTATV12Error("networkx>=3.2 is required for v12 folds") from exc

    graph = nx.Graph()
    graph.add_nodes_from(range(len(rows)))
    clips = [set(map(int, row["clip_ids"])) for row in rows]
    artists = [
        {str(x).strip().casefold() for x in row["artists"]} for row in rows
    ]
    clip_members: dict[int, list[int]] = {}
    artist_members: dict[str, list[int]] = {}
    for index in range(len(rows)):
        for clip in clips[index]:
            clip_members.setdefault(clip, []).append(index)
        for artist in artists[index]:
            artist_members.setdefault(artist, []).append(index)
    weights: dict[tuple[int, int], int] = {}
    for members in (*clip_members.values(), *artist_members.values()):
        for left, right in itertools.combinations(sorted(set(members)), 2):
            weights[(left, right)] = weights.get((left, right), 0) + 1
    graph.add_weighted_edges_from(
        (left, right, weight) for (left, right), weight in sorted(weights.items())
    )
    communities = nx.community.louvain_communities(
        graph, weight="weight", resolution=LOUVAIN_RESOLUTION, seed=int(seed)
    )
    return sorted(
        (sorted(map(int, community)) for community in communities),
        key=lambda values: (
            -len(values),
            min(int(rows[index]["source_row"]) for index in values),
        ),
    )


def _pack_communities(
    rows: Sequence[Mapping[str, Any]],
    communities: Sequence[Sequence[int]],
    n_folds: int,
) -> list[list[int]]:
    """Deterministically balance rows, confidence mass, and fixed strata."""
    if n_folds < 2 or len(rows) < n_folds:
        raise ValueError("fold count must be >=2 and no greater than row count")
    totals = np.zeros((n_folds, 5), dtype=np.float64)
    target = np.asarray([
        len(rows),
        sum(float(row["confidence"]) for row in rows),
        *[
            sum(confidence_stratum(float(row["confidence"])) == stratum for row in rows)
            for stratum in STRATA
        ],
    ], dtype=np.float64) / n_folds
    target = np.maximum(target, 1e-9)
    packed: list[list[int]] = [[] for _ in range(n_folds)]
    for position, community in enumerate(communities):
        vector = np.asarray([
            len(community),
            sum(float(rows[index]["confidence"]) for index in community),
            *[
                sum(
                    confidence_stratum(float(rows[index]["confidence"])) == stratum
                    for index in community
                )
                for stratum in STRATA
            ],
        ], dtype=np.float64)
        # Seed each fold before greedy balancing, avoiding empty folds even when
        # one community dominates a connected raw graph.
        choices = [position] if position < n_folds else list(range(n_folds))
        selected = min(
            choices,
            key=lambda fold: (
                float(np.square((totals[fold] + vector - target) / target).sum()),
                totals[fold, 0],
                totals[fold, 1],
                fold,
            ),
        )
        packed[selected].extend(community)
        totals[selected] += vector
    if any(not values for values in packed):
        raise MTATV12Error("Louvain packing produced an empty fold")
    return [sorted(values, key=lambda i: int(rows[i]["source_row"])) for values in packed]


def build_purged_folds(
    constraints: Sequence[Any],
    *,
    n_folds: int = OUTER_FOLDS,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    """Build deterministic multi-membership purged grouped cross-fit folds."""
    rows = [_record(item) for item in constraints]
    source_rows = [int(row["source_row"]) for row in rows]
    if len(source_rows) != len(set(source_rows)):
        raise MTATV12Error("source_row values must be unique")
    communities = _triad_communities(rows, seed=seed)
    packed = _pack_communities(rows, communities, n_folds)
    folds: list[dict[str, Any]] = []
    for fold_index, test_indices in enumerate(packed):
        test_set = set(test_indices)
        test = [rows[index] for index in test_indices]
        test_clips, test_artists = _entities(test)
        eligible: list[dict[str, Any]] = []
        purged: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            if index in test_set:
                continue
            clips, artists = _entities([row])
            (purged if clips & test_clips or artists & test_artists else eligible).append(row)
        _assert_disjoint(eligible, test)
        purged_clips, purged_artists = _entities(purged)
        eligible_clips, eligible_artists = _entities(eligible)
        folds.append({
            "fold": fold_index,
            "seed": int(seed),
            "method": "multi-membership purged grouped cross-fit; not GroupKFold",
            "test_source_rows": [int(row["source_row"]) for row in test],
            "eligible_train_source_rows": [
                int(row["source_row"]) for row in eligible
            ],
            "purged_source_rows": [int(row["source_row"]) for row in purged],
            "test_clip_ids": sorted(test_clips),
            "test_artists": sorted(test_artists),
            "eligible_train_clip_ids": sorted(eligible_clips),
            "eligible_train_artists": sorted(eligible_artists),
            "purged_clip_ids": sorted(purged_clips),
            "purged_artists": sorted(purged_artists),
            "counts": {
                "test": len(test),
                "eligible_train": len(eligible),
                "purged": len(purged),
                "test_clips": len(test_clips),
                "test_artists": len(test_artists),
                "eligible_train_clips": len(eligible_clips),
                "eligible_train_artists": len(eligible_artists),
                "purged_clips": len(purged_clips),
                "purged_artists": len(purged_artists),
            },
            "test_strata": {
                name: sum(
                    confidence_stratum(float(row["confidence"])) == name for row in test
                )
                for name in STRATA
            },
        })
    observed = [source for fold in folds for source in fold["test_source_rows"]]
    if sorted(observed) != sorted(source_rows) or len(observed) != len(set(observed)):
        raise MTATV12Error("OOF fold coverage is not exactly once per source row")
    return folds


def materialize_fold(
    constraints: Sequence[Any], fold: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Resolve a fold manifest to eligible train, test, and purged records."""
    rows = {int(row["source_row"]): row for row in map(_record, constraints)}
    groups = tuple(
        [rows[int(source)] for source in fold[key]]
        for key in (
            "eligible_train_source_rows",
            "test_source_rows",
            "purged_source_rows",
        )
    )
    _assert_disjoint(groups[0], groups[1])
    return groups  # type: ignore[return-value]


def build_benchmark_all(root: str | Path, output: str | Path) -> dict[str, Any]:
    """Reconcile all accepted v10 constraints and write the v12 manifest."""
    comparisons, clips, audit = v10.load_inputs(root)
    constraints, rejected = v10.parse_constraints(comparisons, clips)
    if len(constraints) != EXPECTED_CONSTRAINTS:
        raise MTATV12Error(
            f"expected {EXPECTED_CONSTRAINTS} accepted constraints, got {len(constraints)}"
        )
    accepted_clips = {clip for row in constraints for clip in row.clip_ids}
    if len(accepted_clips) != EXPECTED_CLIPS:
        raise MTATV12Error(
            f"expected {EXPECTED_CLIPS} accepted clips, got {len(accepted_clips)}"
        )
    component_count = artist_graph_component_count(constraints)
    if component_count != 1:
        raise MTATV12Error(
            f"expected raw artist graph to be connected, got {component_count} components"
        )
    folds = build_purged_folds(constraints)
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_id": "magnatagatune-human-odd-one-out-v12-all",
        "cv_method": "multi-membership purged grouped cross-fit; not GroupKFold",
        "raw_artist_graph_connected_components": component_count,
        "fold_count": OUTER_FOLDS,
        "louvain": {"resolution": LOUVAIN_RESOLUTION, "seed": SEED},
        "minimum_total_votes": v10.MIN_TOTAL_VOTES,
        "ties": "excluded; never broken algorithmically",
        "vote_confidence": {
            "definition": "(winner votes - runner-up votes) / total votes",
            "training_weight": True,
            "fold_strata": {"low": "<0.25", "medium": "[0.25,0.50)", "high": ">=0.50"},
        },
        "input_audit": audit,
        "vote_audit": rejected,
        "accepted_clip_count": len(accepted_clips),
        "constraints": [asdict(item) for item in constraints],
        "outer_folds": folds,
        "created_at": v10._now(),
    }
    document["content_sha256"] = hashlib.sha256(v10._canonical(document)).hexdigest()
    v10._write_json(output, document)
    return document


def prepare_features_all(
    benchmark_path: str | Path,
    metadata_root: str | Path,
    audio_root: str | Path,
    output: str | Path,
    workers: int = 12,
) -> dict[str, Any]:
    """Reuse v10's production-compatible audio extraction for all 611 clips."""
    result = v10.prepare_audio_features(
        benchmark_path, metadata_root, audio_root, output, workers
    )
    if int(result["clips"]) != EXPECTED_CLIPS:
        raise MTATV12Error(f"feature extraction covered {result['clips']} clips, not 611")
    return result


def _require_cuda() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional ML extra
        raise MTATV12Error("PyTorch with CUDA is required") from exc
    if not torch.cuda.is_available():
        raise MTATV12Error("v12 extraction/training is predeclared to require CUDA")
    return torch


def _accepted_clip_ids(benchmark: Mapping[str, Any]) -> list[int]:
    return sorted({
        int(clip)
        for constraint in benchmark["constraints"]
        for clip in constraint["clip_ids"]
    })


def extract_fixed_representations(
    benchmark_path: str | Path,
    feature_cache: str | Path,
    metadata_root: str | Path,
    audio_root: str | Path,
    artist_checkpoint: str | Path,
    fma_checkpoint: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Extract reusable incumbent, FMA, CLAP, and raw DSP arrays."""
    torch = _require_cuda()
    benchmark = json.loads(Path(benchmark_path).read_text(encoding="utf-8"))
    expected = _accepted_clip_ids(benchmark)
    with np.load(feature_cache, allow_pickle=False) as cache:
        clip_ids = cache["clip_ids"].astype(np.int64)
        mels = cache["mels"].astype(np.float32)
        vibe = cache["vibe"].astype(np.float32)
    if sorted(map(int, clip_ids)) != expected or len(set(map(int, clip_ids))) != len(clip_ids):
        raise MTATV12Error("feature cache does not exactly cover accepted clips")
    artist, artist_meta = v10.embed_mels(mels, artist_checkpoint, device="cuda")
    fma, fma_meta = v10.embed_mels(mels, fma_checkpoint, device="cuda")
    _, clips, _ = v10.load_inputs(metadata_root)
    paths = [Path(audio_root) / clips[int(clip)]["mp3_path"] for clip in clip_ids]
    clap, clap_meta = v10.extract_clap(paths)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_sha256": v10.sha256_path(benchmark_path),
        "feature_cache_sha256": v10.sha256_path(feature_cache),
        "cuda": True,
        "gpu": torch.cuda.get_device_name(0),
        "artist_supcon": artist_meta,
        "fma_supcon": fma_meta,
        "clap": clap_meta,
    }
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        clip_ids=clip_ids,
        artist_supcon=artist.astype(np.float32),
        fma_supcon=fma.astype(np.float32),
        vibe_raw=vibe.astype(np.float32),
        clap=clap.astype(np.float32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return {"output": str(target), "sha256": v10.sha256_path(target), **metadata}


def extract_fma_regularizer_cache(
    fma_path: str | Path,
    artist_checkpoint: str | Path,
    output: str | Path,
    *,
    sample_count: int = 512,
) -> dict[str, Any]:
    """Cache independent FMA embeddings used only for geometry distillation."""
    _require_cuda()
    values, metadata = v10._fma_regularizer_embeddings(
        fma_path, artist_checkpoint, sample_count=sample_count, seed=SEED
    )
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        embeddings=values.astype(np.float32),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return {"output": str(target), "sha256": v10.sha256_path(target), **metadata}


def _labels(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray([
        tuple(map(int, row["clip_ids"])).index(int(row["odd_clip_id"]))
        for row in rows
    ], dtype=np.int8)


def _prediction_details(
    rows: Sequence[Mapping[str, Any]], predicted: Sequence[int]
) -> list[dict[str, Any]]:
    actual = _labels(rows)
    return [
        {
            "source_row": int(row["source_row"]),
            "fold": int(row.get("_fold", -1)),
            "predicted_odd_index": int(prediction),
            "actual_odd_index": int(label),
            "correct": int(prediction == label),
            "confidence": float(row["confidence"]),
            "stratum": confidence_stratum(float(row["confidence"])),
            "artists": sorted(set(str(x).strip().casefold() for x in row["artists"])),
        }
        for row, prediction, label in zip(rows, predicted, actual)
    ]


def summarize_predictions(predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize auditable source-row predictions overall and by fold/stratum."""
    ordered = sorted(predictions, key=lambda row: int(row["source_row"]))

    def score(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        correct = np.asarray([int(row["correct"]) for row in rows], dtype=np.float64)
        weights = np.asarray([float(row["confidence"]) for row in rows])
        return {
            "count": len(rows),
            "accuracy": float(correct.mean()) if len(rows) else 0.0,
            "confidence_weighted_accuracy": (
                float(np.average(correct, weights=weights))
                if len(rows) and weights.sum() > 0 else 0.0
            ),
            "source_rows": [int(row["source_row"]) for row in rows],
            "predicted_odd_indices": [
                int(row["predicted_odd_index"]) for row in rows
            ],
            "correct_vector": [int(row["correct"]) for row in rows],
        }

    result = score(ordered)
    result["per_fold"] = {
        str(fold): score([row for row in ordered if int(row["fold"]) == fold])
        for fold in sorted({int(row["fold"]) for row in ordered})
    }
    result["calibration_by_vote_strength"] = {
        stratum: score([row for row in ordered if row["stratum"] == stratum])
        for stratum in STRATA
    }
    result["oof_predictions"] = ordered
    return result


def paired_triad_bootstrap(
    challenger: Sequence[int],
    baseline: Sequence[int],
    *,
    iterations: int = 50_000,
    seed: int = SEED + 81,
) -> dict[str, Any]:
    result = v10.paired_bootstrap_delta(
        challenger, baseline, iterations=iterations, seed=seed
    )
    result["kind"] = "paired triad bootstrap"
    return result


def artist_cluster_bootstrap(
    challenger: Sequence[int],
    baseline: Sequence[int],
    artists: Sequence[Sequence[str]],
    *,
    iterations: int = 50_000,
    seed: int = SEED + 82,
) -> dict[str, Any]:
    """Multi-membership artist bootstrap with mean artist multiplicity weights."""
    delta = np.asarray(challenger, dtype=np.float64) - np.asarray(
        baseline, dtype=np.float64
    )
    if len(delta) == 0 or len(delta) != len(artists):
        raise ValueError("paired vectors/artists must be non-empty and equally sized")
    unique = sorted({str(a).casefold() for group in artists for a in set(group)})
    artist_index = {artist: index for index, artist in enumerate(unique)}
    memberships = [
        np.asarray(sorted({artist_index[str(a).casefold()] for a in group}), dtype=int)
        for group in artists
    ]
    if any(len(group) == 0 for group in memberships):
        raise ValueError("every triad needs at least one artist")
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        counts = np.bincount(
            rng.integers(0, len(unique), size=len(unique)), minlength=len(unique)
        )
        weights = np.asarray([counts[group].mean() for group in memberships])
        draws[iteration] = float(np.average(delta, weights=weights))
    return {
        "kind": "multi-membership artist-cluster bootstrap",
        "delta": float(delta.mean()),
        "ci95": [float(x) for x in np.quantile(draws, (0.025, 0.975))],
        "iterations": iterations,
        "seed": seed,
        "artists": len(unique),
        "triad_weight": "mean resampled multiplicity of unique triad artists",
    }


def paired_sign_flip_test(
    challenger: Sequence[int],
    baseline: Sequence[int],
    *,
    iterations: int = 100_000,
    seed: int = SEED + 83,
) -> dict[str, Any]:
    delta = np.asarray(challenger, dtype=np.float64) - np.asarray(
        baseline, dtype=np.float64
    )
    if not len(delta) or len(delta) != len(baseline):
        raise ValueError("paired vectors must be non-empty and equally sized")
    observed = abs(float(delta.mean()))
    rng = np.random.default_rng(seed)
    extreme = 0
    for start in range(0, iterations, 2000):
        count = min(2000, iterations - start)
        signs = rng.choice((-1.0, 1.0), size=(count, len(delta)))
        extreme += int((np.abs((signs * delta).mean(axis=1)) >= observed - 1e-15).sum())
    return {
        "kind": "paired sign-flip permutation",
        "p_value_two_sided": float((extreme + 1) / (iterations + 1)),
        "iterations": iterations,
        "seed": seed,
    }


def exact_mcnemar(
    challenger: Sequence[int], baseline: Sequence[int]
) -> dict[str, Any]:
    left = np.asarray(challenger, dtype=np.int8)
    right = np.asarray(baseline, dtype=np.int8)
    if len(left) != len(right):
        raise ValueError("paired vectors must be equally sized")
    challenger_only = int(np.sum((left == 1) & (right == 0)))
    baseline_only = int(np.sum((left == 0) & (right == 1)))
    discordant = challenger_only + baseline_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(discordant, k) for k in range(
            min(challenger_only, baseline_only) + 1
        )) / (2 ** discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "p_value_two_sided": float(p_value),
        "challenger_only_correct": challenger_only,
        "baseline_only_correct": baseline_only,
        "discordant": discordant,
    }


def evaluate_win_gate(
    delta: float,
    triad_ci: Sequence[float],
    artist_ci: Sequence[float],
    fold_deltas: Sequence[float],
) -> dict[str, Any]:
    """Apply every predeclared gate condition and return explicit reasons."""
    checks = {
        "positive_absolute_delta_at_least_0.05": float(delta) >= 0.05,
        "triad_bootstrap_lower_above_zero": float(triad_ci[0]) > 0.0,
        "artist_cluster_bootstrap_lower_above_zero": float(artist_ci[0]) > 0.0,
        "positive_delta_in_at_least_3_of_5_folds": (
            len(fold_deltas) == 5 and sum(float(value) > 0 for value in fold_deltas) >= 3
        ),
    }
    reasons = [name for name, passed in checks.items() if not passed]
    return {"passed": all(checks.values()), "checks": checks, "reasons": reasons}


def family_grids() -> dict[str, list[dict[str, Any]]]:
    """Return the exact fixed, intentionally small v12 configuration grids."""
    linear = [
        {"dim": dim, "margin": margin, "lr": 1e-3, "weight_decay": 1e-2,
         "epochs": 140}
        for dim in (32, 64) for margin in (0.1, 0.2)
    ]
    return {
        "linear_triplet": linear,
        "orthogonal_linear_triplet": [
            {**config, "orthogonal_lambda": value}
            for config, value in zip(linear, (0.05, 0.1, 0.05, 0.1))
        ],
        "mlp_triplet": [
            {"dim": dim, "hidden": 64, "margin": margin, "dropout": 0.4,
             "lr": 5e-4, "weight_decay": 2e-2, "epochs": 160}
            for dim in (32, 64) for margin in (0.1, 0.2)
        ],
        "smooth_linear": [
            {"dim": dim, "temperature": temperature, "lr": 1e-3,
             "weight_decay": 1e-2, "epochs": 140}
            for dim in (32, 64) for temperature in (0.1, 0.2)
        ],
        "fma_distilled_linear": [
            {**config, "fma_lambda": value}
            for config, value in zip(linear, (0.05, 0.2, 0.05, 0.2))
        ],
    }


def _parameter_count(input_dim: int, family: str, config: Mapping[str, Any]) -> int:
    output = int(config["dim"])
    if family == "mlp_triplet":
        hidden = int(config["hidden"])
        return input_dim * hidden + hidden + hidden * output
    return input_dim * output


def select_config_nested(
    constraints: Sequence[Any],
    configs: Sequence[Mapping[str, Any]],
    evaluator: Callable[
        [Mapping[str, Any], Sequence[dict[str, Any]], Sequence[dict[str, Any]], int],
        Sequence[int],
    ],
    *,
    input_dim: int,
    family: str,
    seed: int,
) -> dict[str, Any]:
    """Select from inner OOF predictions; evaluator never receives outer rows."""
    rows = [_record(item) for item in constraints]
    inner_folds = build_purged_folds(rows, n_folds=INNER_FOLDS, seed=seed)
    candidates = []
    for config_index, config_value in enumerate(configs):
        config = dict(config_value)
        predictions: list[dict[str, Any]] = []
        losses = []
        fit_seeds = []
        for inner in inner_folds:
            train, validation, _ = materialize_fold(rows, inner)
            fit_seed = seed + config_index * 100 + int(inner["fold"])
            predicted = evaluator(
                config, train, validation, fit_seed
            )
            fit_seeds.append(fit_seed)
            if len(predicted) != len(validation):
                raise MTATV12Error("inner evaluator prediction count mismatch")
            predictions.extend(_prediction_details(validation, predicted))
            if hasattr(evaluator, "last_loss"):
                losses.append(float(getattr(evaluator, "last_loss")))
        summary = summarize_predictions(predictions)
        if sorted(summary["source_rows"]) != sorted(int(row["source_row"]) for row in rows):
            raise MTATV12Error("inner OOF coverage mismatch")
        candidate = {
            "config_id": f"{family}-{config_index:02d}",
            "config": config,
            "parameter_count": _parameter_count(input_dim, family, config),
            "inner_oof_accuracy": summary["accuracy"],
            "inner_oof_confidence_weighted_accuracy":
                summary["confidence_weighted_accuracy"],
            "inner_oof_source_rows": summary["source_rows"],
            "inner_loss_summaries": losses,
            "inner_fit_seeds": fit_seeds,
        }
        candidates.append(candidate)
    selected = max(
        candidates,
        key=lambda item: (
            item["inner_oof_confidence_weighted_accuracy"],
            item["inner_oof_accuracy"],
            -item["parameter_count"],
            # max with inverted lexical rank is awkward; fixed index is exact.
            -int(str(item["config_id"]).rsplit("-", 1)[1]),
        ),
    )
    return {
        "objective": [
            "confidence_weighted_accuracy",
            "unweighted_accuracy",
            "fewer_parameters",
            "config_id",
        ],
        "seed": seed,
        "inner_folds": inner_folds,
        "candidates": candidates,
        "selected": selected,
    }


def _fit_projection(
    base_embeddings: np.ndarray,
    clip_to_row: Mapping[int, int],
    train_rows: Sequence[Mapping[str, Any]],
    family: str,
    config: Mapping[str, Any],
    seed: int,
    fma_embeddings: np.ndarray | None = None,
) -> tuple[Callable[[np.ndarray], np.ndarray], dict[str, np.ndarray], float]:
    """Fit one compact projection. CUDA is intentionally mandatory."""
    torch = _require_cuda()
    import torch.nn.functional as F

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    device = "cuda"
    base = v10._normalise_rows(base_embeddings)
    train_clip_ids = sorted({
        int(clip) for row in train_rows for clip in row["clip_ids"]
    })
    if not train_clip_ids:
        raise MTATV12Error("purging left no training constraints")
    local_row = {clip: index for index, clip in enumerate(train_clip_ids)}
    x = torch.from_numpy(np.stack([
        base[clip_to_row[clip]] for clip in train_clip_ids
    ])).to(device)
    dim = int(config["dim"])
    if family == "mlp_triplet":
        model = torch.nn.Sequential(
            torch.nn.Linear(base.shape[1], int(config["hidden"])),
            torch.nn.GELU(),
            torch.nn.Dropout(float(config["dropout"])),
            torch.nn.Linear(int(config["hidden"]), dim, bias=False),
        ).to(device)
    else:
        model = torch.nn.Linear(base.shape[1], dim, bias=False).to(device)
        torch.nn.init.orthogonal_(model.weight)
    indices = torch.as_tensor([
        [
            local_row[int(row["similar_clip_ids"][0])],
            local_row[int(row["similar_clip_ids"][1])],
            local_row[int(row["odd_clip_id"])],
        ]
        for row in train_rows
    ], dtype=torch.long, device=device)
    weights = torch.as_tensor(
        [float(row["confidence"]) for row in train_rows],
        dtype=torch.float32, device=device,
    )
    fma_x = None
    if family == "fma_distilled_linear":
        if fma_embeddings is None or fma_embeddings.shape[1] != base.shape[1]:
            raise MTATV12Error("compatible independent FMA geometry cache is required")
        fma_x = torch.from_numpy(v10._normalise_rows(fma_embeddings)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"])
    )
    generator = torch.Generator(device=device).manual_seed(seed + 1)
    final_loss = math.nan
    model.train()
    for _ in range(int(config["epochs"])):
        projected = F.normalize(model(x), dim=1)
        similar_left, similar_right, odd = (
            projected[indices[:, 0]], projected[indices[:, 1]], projected[indices[:, 2]]
        )
        positive = 1.0 - (similar_left * similar_right).sum(1)
        negative_left = 1.0 - (similar_left * odd).sum(1)
        negative_right = 1.0 - (similar_right * odd).sum(1)
        if family == "smooth_linear":
            temperature = float(config["temperature"])
            row_loss = 0.5 * (
                F.softplus((positive - negative_left) / temperature)
                + F.softplus((positive - negative_right) / temperature)
            )
        else:
            margin = float(config["margin"])
            row_loss = 0.5 * (
                F.relu(positive - negative_left + margin)
                + F.relu(positive - negative_right + margin)
            )
        loss = (row_loss * weights).sum() / weights.sum().clamp_min(1e-8)
        if family == "orthogonal_linear_triplet":
            weight = model.weight
            identity = torch.eye(weight.shape[0], device=device)
            loss = loss + float(config["orthogonal_lambda"]) * (
                (weight @ weight.T - identity).square().mean()
            )
        if family == "fma_distilled_linear":
            assert fma_x is not None
            pairs = torch.randint(
                len(fma_x), (256, 2), generator=generator, device=device
            )
            original = (fma_x[pairs[:, 0]] * fma_x[pairs[:, 1]]).sum(1)
            projected_fma = F.normalize(model(fma_x), dim=1)
            compressed = (
                projected_fma[pairs[:, 0]] * projected_fma[pairs[:, 1]]
            ).sum(1)
            loss = loss + float(config["fma_lambda"]) * F.mse_loss(
                compressed, original
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu())
    model.eval()

    def transform(values: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            tensor = torch.from_numpy(
                v10._normalise_rows(values).astype(np.float32)
            ).to(device)
            return F.normalize(model(tensor), dim=1).cpu().numpy()

    state = {
        key: value.detach().cpu().numpy().astype(np.float32)
        for key, value in model.state_dict().items()
    }
    return transform, state, final_loss


def _save_checkpoint(
    path: Path,
    state: Mapping[str, np.ndarray],
    *,
    family: str,
    config_id: str,
    config: Mapping[str, Any],
    seed: int,
    loss: float,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "family": family,
        "config_id": config_id,
        "config": dict(config),
        "seed": seed,
        "final_training_loss": loss,
    }
    np.savez_compressed(
        path,
        **state,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return {
        **metadata,
        "path": str(path),
        "sha256": v10.sha256_path(path),
    }


def _fit_dsp_fold(
    vibe: np.ndarray,
    clip_ids: Sequence[int],
    train_rows: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    row = {int(clip): index for index, clip in enumerate(clip_ids)}
    train_clips = sorted({int(x) for item in train_rows for x in item["clip_ids"]})
    if not train_clips:
        raise MTATV12Error("DSP scaler has no eligible training clips")
    train = vibe[[row[clip] for clip in train_clips]]
    mean = train.mean(axis=0)
    std = train.std(axis=0) + 1e-6
    try:
        from soundalike.audio.vibe import DEFAULT_WEIGHTS, weight_vector
        weights = np.sqrt(weight_vector(DEFAULT_WEIGHTS)).astype(np.float32)
        if len(weights) != vibe.shape[1]:
            weights = np.ones(vibe.shape[1], dtype=np.float32)
    except (ImportError, ValueError):
        weights = np.ones(vibe.shape[1], dtype=np.float32)
    return v10._normalise_rows(((vibe - mean) / std) * weights)


def _load_run_caches(
    benchmark: Mapping[str, Any],
    fixed_cache: str | Path,
    fma_regularizer_cache: str | Path,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    expected = _accepted_clip_ids(benchmark)
    with np.load(fixed_cache, allow_pickle=False) as cache:
        clip_ids = cache["clip_ids"].astype(np.int64)
        arrays = {name: cache[name].astype(np.float32) for name in FIXED_NAMES
                  if name in cache}
        if "vibe_dsp" not in arrays and "vibe_raw" in cache:
            arrays["vibe_dsp"] = cache["vibe_raw"].astype(np.float32)
    required = set(FIXED_NAMES)
    if set(arrays) != required:
        raise MTATV12Error(f"fixed cache missing arrays: {sorted(required - set(arrays))}")
    if sorted(map(int, clip_ids)) != expected or len(clip_ids) != len(expected):
        raise MTATV12Error("fixed cache does not exactly cover all accepted clips")
    if any(len(values) != len(clip_ids) for values in arrays.values()):
        raise MTATV12Error("fixed representation row count mismatch")
    with np.load(fma_regularizer_cache, allow_pickle=False) as cache:
        fma_regularizer = cache["embeddings"].astype(np.float32)
    return clip_ids, arrays, fma_regularizer


def run_nested_cv(
    benchmark_path: str | Path,
    fixed_cache: str | Path,
    fma_regularizer_cache: str | Path,
    work_dir: str | Path,
    report_path: str | Path,
    *,
    bootstrap_iterations: int = 50_000,
    permutation_iterations: int = 100_000,
) -> dict[str, Any]:
    """Run leakage-asserted nested CV and write checkpoints plus full OOF report."""
    torch = _require_cuda()
    torch.cuda.reset_peak_memory_stats()
    try:
        import psutil
        process = psutil.Process()
        rss_before = int(process.memory_info().rss)
    except ImportError:  # pragma: no cover - psutil is part of the ML extra
        process = None
        rss_before = None
    benchmark = json.loads(Path(benchmark_path).read_text(encoding="utf-8"))
    constraints = [_record(item) for item in benchmark["constraints"]]
    if len(constraints) != EXPECTED_CONSTRAINTS:
        raise MTATV12Error("benchmark does not contain exactly all 307 constraints")
    folds = benchmark.get("outer_folds") or build_purged_folds(constraints)
    observed = [source for fold in folds for source in fold["test_source_rows"]]
    expected_sources = [int(row["source_row"]) for row in constraints]
    if sorted(observed) != sorted(expected_sources) or len(set(observed)) != len(observed):
        raise MTATV12Error("outer OOF source-row coverage differs from exactly all 307")
    clip_ids, arrays, fma_regularizer = _load_run_caches(
        benchmark, fixed_cache, fma_regularizer_cache
    )
    clip_to_row = {int(clip): index for index, clip in enumerate(clip_ids)}
    work = Path(work_dir)
    all_predictions: dict[str, list[dict[str, Any]]] = {
        name: [] for name in (*FIXED_NAMES, *FAMILY_ORDER, "nested_selected_learned")
    }
    fold_logs = []
    grids = family_grids()
    started = time.perf_counter()
    for outer in folds:
        fold_index = int(outer["fold"])
        train, test, _ = materialize_fold(constraints, outer)
        test = [{**row, "_fold": fold_index} for row in test]
        fixed_values = {
            "artist_supcon": arrays["artist_supcon"],
            "fma_supcon": arrays["fma_supcon"],
            "clap": arrays["clap"],
            "vibe_dsp": _fit_dsp_fold(arrays["vibe_dsp"], clip_ids, train),
        }
        for name, values in fixed_values.items():
            predicted = v10.odd_predictions(values, clip_to_row, test)
            all_predictions[name].extend(_prediction_details(test, predicted))
        selections: dict[str, Any] = {}
        family_fold_predictions: dict[str, list[dict[str, Any]]] = {}
        checkpoints: dict[str, Any] = {}
        for family_index, family in enumerate(FAMILY_ORDER):
            outer_seed = SEED + fold_index * 10_000 + family_index * 1_000
            inner_split_seed = SEED + fold_index * 10_000 + 100

            class Evaluator:
                last_loss = math.nan

                def __call__(
                    self,
                    config: Mapping[str, Any],
                    inner_train: Sequence[dict[str, Any]],
                    inner_validation: Sequence[dict[str, Any]],
                    fit_seed: int,
                ) -> Sequence[int]:
                    transform, _, loss = _fit_projection(
                        arrays["artist_supcon"], clip_to_row, inner_train,
                        family, config, fit_seed, fma_regularizer
                    )
                    self.last_loss = loss
                    return v10.odd_predictions(
                        transform(arrays["artist_supcon"]),
                        clip_to_row, inner_validation,
                    )

            selection = select_config_nested(
                train, grids[family], Evaluator(),
                input_dim=arrays["artist_supcon"].shape[1],
                family=family, seed=inner_split_seed,
            )
            selections[family] = selection
            chosen = selection["selected"]
            retrain_seed = outer_seed + 900
            transform, state, loss = _fit_projection(
                arrays["artist_supcon"], clip_to_row, train, family,
                chosen["config"], retrain_seed, fma_regularizer
            )
            checkpoint = _save_checkpoint(
                work / f"fold-{fold_index}" / f"{family}.npz", state,
                family=family, config_id=chosen["config_id"],
                config=chosen["config"], seed=retrain_seed, loss=loss,
            )
            checkpoints[family] = checkpoint
            predicted = v10.odd_predictions(
                transform(arrays["artist_supcon"]), clip_to_row, test
            )
            details = _prediction_details(test, predicted)
            family_fold_predictions[family] = details
            all_predictions[family].extend(details)
        nested_family = max(
            FAMILY_ORDER,
            key=lambda family: (
                selections[family]["selected"][
                    "inner_oof_confidence_weighted_accuracy"
                ],
                selections[family]["selected"]["inner_oof_accuracy"],
                -selections[family]["selected"]["parameter_count"],
                -FAMILY_ORDER.index(family),
            ),
        )
        all_predictions["nested_selected_learned"].extend(
            family_fold_predictions[nested_family]
        )
        fold_logs.append({
            "fold": fold_index,
            "outer_seed": SEED + fold_index * 10_000,
            "test_source_rows": outer["test_source_rows"],
            "eligible_train_source_rows": outer["eligible_train_source_rows"],
            "purged_source_rows": outer["purged_source_rows"],
            "selections": selections,
            "nested_selected_family": nested_family,
            "checkpoints": checkpoints,
        })
    expected_sorted = sorted(expected_sources)
    for name, predictions in all_predictions.items():
        sources = sorted(int(row["source_row"]) for row in predictions)
        if sources != expected_sorted or len(sources) != len(set(sources)):
            raise MTATV12Error(f"{name} OOF source-row coverage mismatch")
    scores = {
        name: summarize_predictions(predictions)
        for name, predictions in all_predictions.items()
    }
    learned = scores["nested_selected_learned"]["correct_vector"]
    incumbent = scores["artist_supcon"]["correct_vector"]
    artists = [
        row["artists"] for row in scores["nested_selected_learned"]["oof_predictions"]
    ]
    triad_bootstrap = paired_triad_bootstrap(
        learned, incumbent, iterations=bootstrap_iterations
    )
    artist_bootstrap = artist_cluster_bootstrap(
        learned, incumbent, artists, iterations=bootstrap_iterations
    )
    sign_flip = paired_sign_flip_test(
        learned, incumbent, iterations=permutation_iterations
    )
    mcnemar = exact_mcnemar(learned, incumbent)
    fold_deltas = [
        scores["nested_selected_learned"]["per_fold"][str(fold)]["accuracy"]
        - scores["artist_supcon"]["per_fold"][str(fold)]["accuracy"]
        for fold in range(OUTER_FOLDS)
    ]
    gate = evaluate_win_gate(
        triad_bootstrap["delta"], triad_bootstrap["ci95"],
        artist_bootstrap["ci95"], fold_deltas
    )
    delta_by_vote_strength = {
        stratum: (
            scores["nested_selected_learned"]["calibration_by_vote_strength"][stratum]["accuracy"]
            - scores["artist_supcon"]["calibration_by_vote_strength"][stratum]["accuracy"]
        )
        for stratum in STRATA
    }
    checkpoint_bytes = sum(
        int(Path(checkpoint["path"]).stat().st_size)
        for fold in fold_logs for checkpoint in fold["checkpoints"].values()
    )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "magnatagatune-human-audio-nested-purged-cross-fit",
        "cv_method": "multi-membership purged grouped cross-fit; not GroupKFold",
        "predeclared_decision": {
            "promotion_eligible_model": "nested_selected_learned",
            "family_and_hyperparameter_selection": "inner folds only",
            "outer_fold_results_used_for_selection": False,
            "fixed_and_family_results": "reported as comparators; not substituted post hoc",
            "training_weight": "MTAT vote confidence",
            "fold_strata": list(STRATA),
        },
        "benchmark": {
            "path": str(benchmark_path),
            "sha256": v10.sha256_path(benchmark_path),
            "constraints": len(constraints),
            "clips": len(clip_ids),
        },
        "cache_hashes": {
            "fixed": v10.sha256_path(fixed_cache),
            "fma_regularizer": v10.sha256_path(fma_regularizer_cache),
        },
        "fold_logs": fold_logs,
        "scores": scores,
        "primary_comparison": {
            "challenger": "nested_selected_learned",
            "baseline": "artist_supcon",
            "fold_accuracy_deltas": fold_deltas,
            "triad_bootstrap": triad_bootstrap,
            "artist_cluster_bootstrap": artist_bootstrap,
            "sign_flip_permutation": sign_flip,
            "mcnemar_exact": mcnemar,
            "accuracy_delta_by_vote_strength": delta_by_vote_strength,
        },
        "win_gate": gate,
        "final_all_data_projection_trained": False,
        "catalog_embeddings_changed": False,
        "commercial_evaluator_changed": False,
        "commercial_final_opened": False,
        "production_ranking_changed": False,
        "resources": {
            "device": "cuda",
            "gpu": torch.cuda.get_device_name(0),
            "seconds": time.perf_counter() - started,
            "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "process_rss_before_bytes": rss_before,
            "process_rss_after_bytes": (
                int(process.memory_info().rss) if process is not None else None
            ),
            "fixed_cache_bytes": int(Path(fixed_cache).stat().st_size),
            "fma_regularizer_cache_bytes": int(
                Path(fma_regularizer_cache).stat().st_size
            ),
            "selected_fold_checkpoint_count": sum(
                len(fold["checkpoints"]) for fold in fold_logs
            ),
            "selected_fold_checkpoint_bytes": checkpoint_bytes,
        },
        "created_at": v10._now(),
    }
    report["content_sha256"] = hashlib.sha256(v10._canonical(report)).hexdigest()
    v10._write_json(report_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("benchmark-all", "build"):
        command = commands.add_parser(name, help="build all-constraint v12 benchmark")
        command.add_argument("--root", default="ml_data/magnatagatune")
        command.add_argument(
            "--output", default="ml_data/magnatagatune/benchmark-v12-all.json"
        )
    for name in ("features-all", "prepare"):
        command = commands.add_parser(name, help="prepare all accepted audio features")
        command.add_argument(
            "--benchmark", default="ml_data/magnatagatune/benchmark-v12-all.json"
        )
        command.add_argument("--metadata-root", default="ml_data/magnatagatune")
        command.add_argument("--audio-root", default="ml_data/magnatagatune/audio")
        command.add_argument(
            "--output", default="ml_data/magnatagatune/features-v12-all.npz"
        )
        command.add_argument("--workers", type=int, default=12)
    extract = commands.add_parser(
        "extract-fixed", help="extract/cache fixed representations"
    )
    extract.add_argument(
        "--benchmark", default="ml_data/magnatagatune/benchmark-v12-all.json"
    )
    extract.add_argument(
        "--features", default="ml_data/magnatagatune/features-v12-all.npz"
    )
    extract.add_argument("--metadata-root", default="ml_data/magnatagatune")
    extract.add_argument("--audio-root", default="ml_data/magnatagatune/audio")
    extract.add_argument(
        "--artist-checkpoint", default="ml_data/model_artist384/encoder_best.pt"
    )
    extract.add_argument(
        "--fma-checkpoint", default="ml_data/iteration4/supcon/supcon_encoder.pt"
    )
    extract.add_argument(
        "--output", default="ml_data/magnatagatune/fixed-v12-all.npz"
    )
    extract.add_argument("--fma", default="ml_data/fma_packed.npz")
    extract.add_argument(
        "--fma-regularizer-output",
        default="ml_data/magnatagatune/fma-regularizer-v12.npz",
    )
    run = commands.add_parser("run-nested", help="run v12 nested purged CV")
    run.add_argument(
        "--benchmark", default="ml_data/magnatagatune/benchmark-v12-all.json"
    )
    run.add_argument("--fixed", default="ml_data/magnatagatune/fixed-v12-all.npz")
    run.add_argument(
        "--fma-regularizer",
        default="ml_data/magnatagatune/fma-regularizer-v12.npz",
    )
    run.add_argument("--work-dir", default="ml_data/magnatagatune/v12-run")
    run.add_argument(
        "--report", default="ml_data/magnatagatune/v12-run/report.json"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command in ("benchmark-all", "build"):
        result = build_benchmark_all(args.root, args.output)
    elif args.command in ("features-all", "prepare"):
        result = prepare_features_all(
            args.benchmark, args.metadata_root, args.audio_root,
            args.output, args.workers
        )
    elif args.command == "extract-fixed":
        result = {
            "fixed": extract_fixed_representations(
                args.benchmark, args.features, args.metadata_root, args.audio_root,
                args.artist_checkpoint, args.fma_checkpoint, args.output,
            ),
            "fma_regularizer": extract_fma_regularizer_cache(
                args.fma, args.artist_checkpoint, args.fma_regularizer_output
            ),
        }
    else:
        result = run_nested_cv(
            args.benchmark, args.fixed, args.fma_regularizer,
            args.work_dir, args.report
        )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
