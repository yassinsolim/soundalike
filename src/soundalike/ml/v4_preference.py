"""Fit the compact V4 pairwise reranker from development-only listener evidence."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression

from .fulltrack_extract import normalize_rows
from .fulltrack_store import FullTrackStoreReader
from .jamendo_fulltrack import load_jamendo_context
from .pacing_eval import (
    acoustic_scores,
    compatibility_components,
    load_vibe_cache,
    pacing_rerank_scores,
    robust_standardize_vibe,
)
from .semantic_predictor import load_predictor
from .v4_features import load_semantic_cache


SCHEMA_VERSION = 1
MODEL_KIND = "soundalike_v4_pairwise_linear_reranker"
FEATURE_NAMES = (
    "acoustic",
    "pacing",
    "tone",
    "dynamics",
    "instrument",
    "mood_theme",
    "genre",
    "voice_compatibility",
)
C_VALUES = (0.01, 0.1, 1.0)


class V4PreferenceError(RuntimeError):
    """V4 preference evidence or grouped validation is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pairwise_rows(
    features: np.ndarray,
    ratings: np.ndarray,
    groups: np.ndarray,
    included_groups: set[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = []
    labels = []
    weights = []
    for group in sorted(included_groups):
        positions = np.flatnonzero(groups == group)
        for left, right in itertools.combinations(positions, 2):
            difference = float(ratings[left] - ratings[right])
            if difference == 0.0:
                continue
            rows.append(features[left] - features[right])
            labels.append(1 if difference > 0.0 else 0)
            weights.append(abs(difference))
    if not rows or len(set(labels)) != 2:
        raise V4PreferenceError("pairwise evidence lacks both preference classes")
    return (
        np.asarray(rows, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        np.asarray(weights, dtype=np.float64),
    )


def _fit(
    features: np.ndarray,
    ratings: np.ndarray,
    groups: np.ndarray,
    included_groups: set[int],
    regularization: float,
) -> tuple[np.ndarray, np.ndarray]:
    differences, labels, weights = _pairwise_rows(
        features, ratings, groups, included_groups
    )
    scale = np.std(differences, axis=0)
    scale = np.maximum(scale, 1e-6)
    model = LogisticRegression(
        C=regularization,
        fit_intercept=False,
        solver="liblinear",
        max_iter=10_000,
        random_state=0,
    )
    model.fit(differences / scale, labels, sample_weight=weights)
    return model.coef_[0] / scale, scale


def _group_pair_accuracy(
    scores: np.ndarray,
    ratings: np.ndarray,
    groups: np.ndarray,
    group: int,
) -> float:
    positions = np.flatnonzero(groups == group)
    outcomes = []
    for left, right in itertools.combinations(positions, 2):
        expected = np.sign(ratings[left] - ratings[right])
        if expected == 0.0:
            continue
        predicted = np.sign(scores[left] - scores[right])
        outcomes.append(0.5 if predicted == 0.0 else float(predicted == expected))
    if not outcomes:
        raise V4PreferenceError("a validation group has no non-tied preferences")
    return float(np.mean(outcomes))


def _choose_c(
    features: np.ndarray,
    ratings: np.ndarray,
    groups: np.ndarray,
    training_groups: set[int],
) -> float:
    candidates = []
    for regularization in C_VALUES:
        scores = []
        for validation_group in sorted(training_groups):
            inner_groups = training_groups - {validation_group}
            coefficients, _ = _fit(
                features,
                ratings,
                groups,
                inner_groups,
                regularization,
            )
            scores.append(
                _group_pair_accuracy(
                    features @ coefficients,
                    ratings,
                    groups,
                    validation_group,
                )
            )
        candidates.append((float(np.mean(scores)), -regularization, regularization))
    return max(candidates)[2]


def grouped_validation(
    features: np.ndarray,
    ratings: np.ndarray,
    groups: np.ndarray,
    baseline_scores: Mapping[str, np.ndarray],
) -> Mapping[str, object]:
    matrix = np.asarray(features, dtype=np.float64)
    targets = np.asarray(ratings, dtype=np.float64)
    group_values = np.asarray(groups, dtype=np.int64)
    if (
        matrix.ndim != 2
        or matrix.shape[1] != len(FEATURE_NAMES)
        or targets.shape != (len(matrix),)
        or group_values.shape != targets.shape
        or len(set(group_values)) < 4
        or not np.all(np.isfinite(matrix))
        or not np.all(np.isfinite(targets))
    ):
        raise V4PreferenceError("grouped validation inputs are invalid")
    unique_groups = set(int(value) for value in group_values)
    outer = []
    selected_cs = []
    for held_out in sorted(unique_groups):
        training = unique_groups - {held_out}
        regularization = _choose_c(matrix, targets, group_values, training)
        coefficients, _ = _fit(
            matrix, targets, group_values, training, regularization
        )
        learned = _group_pair_accuracy(
            matrix @ coefficients, targets, group_values, held_out
        )
        row = {
            "group": held_out,
            "selected_c": regularization,
            "learned_pair_accuracy": learned,
        }
        for name, values in baseline_scores.items():
            scores = np.asarray(values, dtype=np.float64)
            if scores.shape != targets.shape or not np.all(np.isfinite(scores)):
                raise V4PreferenceError("baseline score shape is invalid")
            row[f"{name}_pair_accuracy"] = _group_pair_accuracy(
                scores, targets, group_values, held_out
            )
        outer.append(row)
        selected_cs.append(regularization)
    means = {
        key: float(np.mean([row[key] for row in outer]))
        for key in outer[0]
        if key.endswith("_pair_accuracy")
    }
    chosen_c = sorted(
        Counter(selected_cs).items(), key=lambda item: (-item[1], item[0])
    )[0][0]
    coefficients, scale = _fit(
        matrix, targets, group_values, unique_groups, chosen_c
    )
    acoustic_gain = (
        means["learned_pair_accuracy"] - means["acoustic_pair_accuracy"]
    )
    v3_gain = means["learned_pair_accuracy"] - means["pacing_v3_pair_accuracy"]
    accepted = acoustic_gain >= 0.03 and v3_gain >= 0.03
    return {
        "outer_folds": outer,
        "mean_pair_accuracy": means,
        "learned_gain_over_acoustic": acoustic_gain,
        "learned_gain_over_pacing_v3": v3_gain,
        "acceptance_rule": {
            "minimum_gain_over_each_baseline": 0.03,
            "passed": accepted,
        },
        "final_model": {
            "regularization_c": chosen_c,
            "feature_names": list(FEATURE_NAMES),
            "coefficients": coefficients.tolist(),
            "difference_scale": scale.tolist(),
        },
    }


def _rated_track_map(pack: Mapping[str, object]) -> Mapping[str, tuple[int, int]]:
    result = {}
    for seed in pack["seeds"]:
        seed_id = int(seed["seed_track_id"])
        for candidate_list in seed["lists"]:
            for row in candidate_list["ranking"]:
                result[str(row["result_id"])] = (seed_id, int(row["track_id"]))
    return result


def build_preference_artifact(
    *,
    metadata_root: Path,
    audio_root: Path,
    state_root: Path,
    store_root: Path,
    predictor_model: Path,
    predictor_metadata: Path,
    semantic_cache: Path,
    semantic_cache_metadata: Path,
    vibe_cache: Path,
    public_pack: Path,
    receipt: Path,
) -> Mapping[str, object]:
    context = load_jamendo_context(
        metadata_root, audio_root, state_root, production=True
    )
    predictor = load_predictor(predictor_model, predictor_metadata)
    pack = json.loads(public_pack.read_text(encoding="utf-8"))
    ratings_document = json.loads(receipt.read_text(encoding="utf-8"))
    if (
        ratings_document.get("pilot_pack_sha256") != pack.get("content_sha256")
        or ratings_document.get("canonical_payload_sha256") != receipt.stem
    ):
        raise V4PreferenceError("rating receipt/public pack binding drift")
    result_map = _rated_track_map(pack)
    rating_rows = []
    for result_id, rating in ratings_document["result_ratings"].items():
        if result_id not in result_map:
            raise V4PreferenceError("rating references an unknown result")
        seed_id, track_id = result_map[result_id]
        rating_rows.append(
            (seed_id, track_id, float(rating["score_0_10"]), result_id)
        )
    rating_rows.sort(key=lambda row: (row[0], row[1], row[3]))
    if len(rating_rows) != 69:
        raise V4PreferenceError("development receipt coverage drift")

    with np.load(vibe_cache, allow_pickle=False) as archive:
        vibe_ids = np.asarray(archive["track_ids"], dtype=np.int64)
        vibe_starts = np.asarray(archive["starts"], dtype=np.float64)
        vibe_ends = np.asarray(archive["ends"], dtype=np.float64)
    vibe = load_vibe_cache(
        vibe_cache, vibe_ids, vibe_starts, vibe_ends
    ).astype(np.float64)
    standardized_vibe = robust_standardize_vibe(vibe)
    vibe_row = {int(track_id): row for row, track_id in enumerate(vibe_ids)}

    with FullTrackStoreReader(
        store_root, expected_source_fingerprint=context.source_fingerprint
    ) as reader:
        store_ids = np.asarray(reader.track_ids, dtype=np.int64)
        store_row = {
            int(track_id): row for row, track_id in enumerate(store_ids)
        }
        globals_matrix = normalize_rows(
            np.asarray(reader.global_embeddings, dtype=np.float32)
        )
        probabilities, voice_scores, _ = load_semantic_cache(
            semantic_cache,
            semantic_cache_metadata,
            expected_source_fingerprint=context.source_fingerprint,
            expected_track_ids=store_ids,
        )
        features = []
        targets = []
        groups = []
        acoustic_baseline = []
        pacing_baseline = []
        for seed_id in sorted({row[0] for row in rating_rows}):
            rows = [row for row in rating_rows if row[0] == seed_id]
            candidate_ids = [row[1] for row in rows]
            candidate_store_rows = np.asarray(
                [store_row[track_id] for track_id in candidate_ids],
                dtype=np.int64,
            )
            query_store_row = store_row[seed_id]
            global_scores = (
                globals_matrix[candidate_store_rows]
                @ globals_matrix[query_store_row]
            )
            acoustic = acoustic_scores(
                reader, seed_id, candidate_ids, global_scores
            )
            candidate_vibe_rows = np.asarray(
                [vibe_row[track_id] for track_id in candidate_ids],
                dtype=np.int64,
            )
            query_vibe_row = vibe_row[seed_id]
            components = compatibility_components(
                vibe[candidate_vibe_rows],
                vibe[query_vibe_row],
                standardized_vibe[candidate_vibe_rows],
                standardized_vibe[query_vibe_row],
                probabilities[candidate_store_rows],
                probabilities[query_store_row],
                predictor,
            )
            voice_compatibility = np.exp(
                -np.abs(
                    voice_scores[candidate_store_rows]
                    - voice_scores[query_store_row]
                )
                / 0.05
            )
            current_features = np.column_stack(
                [
                    acoustic,
                    components["pacing"],
                    components["tone"],
                    components["dynamics"],
                    components["instrument"],
                    components["mood_theme"],
                    components["genre"],
                    voice_compatibility,
                ]
            )
            features.extend(current_features.tolist())
            targets.extend(row[2] for row in rows)
            groups.extend([seed_id] * len(rows))
            acoustic_baseline.extend(acoustic.tolist())
            pacing_baseline.extend(
                pacing_rerank_scores(acoustic, components).tolist()
            )

    validation = grouped_validation(
        np.asarray(features, dtype=np.float64),
        np.asarray(targets, dtype=np.float64),
        np.asarray(groups, dtype=np.int64),
        {
            "acoustic": np.asarray(acoustic_baseline, dtype=np.float64),
            "pacing_v3": np.asarray(pacing_baseline, dtype=np.float64),
        },
    )
    artifact: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "model_kind": MODEL_KIND,
        "research_only": True,
        "production_ranking_changed": False,
        "promotion_allowed": False,
        "evidence": {
            "receipt_sha256": _sha256(receipt),
            "receipt_payload_sha256": receipt.stem,
            "public_pack_sha256": pack["content_sha256"],
            "rated_results": len(rating_rows),
            "seed_groups": len(set(groups)),
            "evaluation_reuse_allowed": False,
        },
        "bindings": {
            "source_fingerprint": context.source_fingerprint,
            "semantic_cache_sha256": _sha256(semantic_cache),
            "vibe_cache_sha256": _sha256(vibe_cache),
            "predictor_model_sha256": _sha256(predictor_model),
        },
        "validation": validation,
    }
    artifact["content_sha256"] = hashlib.sha256(
        json.dumps(
            artifact,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return artifact


def score_features(features: np.ndarray, artifact: Mapping[str, object]) -> np.ndarray:
    model = artifact.get("validation", {}).get("final_model", {})
    if (
        model.get("feature_names") != list(FEATURE_NAMES)
        or len(model.get("coefficients", [])) != len(FEATURE_NAMES)
    ):
        raise V4PreferenceError("V4 preference model binding drift")
    matrix = np.asarray(features, dtype=np.float64)
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[1] != len(FEATURE_NAMES)
        or not np.all(np.isfinite(matrix))
    ):
        raise V4PreferenceError("V4 preference scoring input is invalid")
    return matrix @ coefficients


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit the V4 pairwise reranker.")
    for name in (
        "metadata_root",
        "audio_root",
        "state_root",
        "store_root",
        "predictor_model",
        "predictor_metadata",
        "semantic_cache",
        "semantic_cache_metadata",
        "vibe_cache",
        "public_pack",
        "receipt",
        "output",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifact = build_preference_artifact(
        metadata_root=args.metadata_root,
        audio_root=args.audio_root,
        state_root=args.state_root,
        store_root=args.store_root,
        predictor_model=args.predictor_model,
        predictor_metadata=args.predictor_metadata,
        semantic_cache=args.semantic_cache,
        semantic_cache_metadata=args.semantic_cache_metadata,
        vibe_cache=args.vibe_cache,
        public_pack=args.public_pack,
        receipt=args.receipt,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps(artifact["validation"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
