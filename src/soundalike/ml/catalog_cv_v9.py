"""Powered served-list scorer and leakage-safe DEV cross-validation."""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from dataclasses import asdict
from statistics import mean
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from .catalog_list_gold_v9 import artist_key
from .catalog_policy_v9 import ListPolicy
from .catalog_cv import paired_bootstrap
from .quality_filter import TitleQualityFilter


MIN_RELATIVE_GAIN = 0.20
MIN_ABSOLUTE_GAIN = 0.02
MIN_IMPROVED_SEEDS = 10
MAX_SCENE_REGRESSION = -0.10
MIN_COHERENCE = 0.80
MIN_COHERENCE_MARGIN = 0.10
PRIMARY_K = 10
COHERENCE_K = 5


def _gain(grade: int) -> float:
    return float(2 ** int(grade) - 1)


def _dcg(grades: Sequence[int], k: int = PRIMARY_K) -> float:
    return sum(
        _gain(grade) / math.log2(position + 1.0)
        for position, grade in enumerate(grades[:k], start=1)
    )


def _track_id(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


class ListGoldScorer:
    """Score actual catalogue rows against frozen source-derived relevance."""

    def __init__(
        self,
        gold: Mapping[str, Any],
        snapshots: Mapping[str, Any],
        titles: Sequence[str],
        artists: Sequence[str],
        track_ids: Sequence[Any],
    ):
        self.records = {
            str(item["id"]): item for item in gold.get("records", ())
        }
        self.snapshots = {
            str(item["seed_id"]): item for item in snapshots.get("records", ())
        }
        self.titles = np.asarray(titles)
        self.artists = np.asarray(artists)
        self.track_ids = np.asarray(track_ids)
        self.quality = TitleQualityFilter()

    def _rules(
        self, record: Mapping[str, Any]
    ) -> Tuple[Dict[Any, Mapping[str, Any]], Dict[str, Mapping[str, Any]]]:
        tracks: Dict[Any, Mapping[str, Any]] = {}
        artists: Dict[str, Mapping[str, Any]] = {}
        for positive in record.get("positives", ()):
            if positive["relevance_scope"] == "track":
                tracks[_track_id(positive["track_id"])] = positive
            else:
                artists[artist_key(str(positive["artist"]))] = positive
        return tracks, artists

    def _coherence_artists(self, record: Mapping[str, Any]) -> set[str]:
        snapshot = self.snapshots[str(record["id"])]
        values = {
            artist_key(str(item["artist"]))
            for item in snapshot.get("neighbors", ())
        }
        for positive in record.get("positives", ()):
            values.add(artist_key(str(positive["artist"])))
        return values

    def score(
        self, record: Mapping[str, Any], rows: Sequence[int]
    ) -> Dict[str, Any]:
        tracks, artists = self._rules(record)
        coherence_artists = self._coherence_artists(record)
        seen_entities: set[str] = set()
        grades: List[int] = []
        result_evidence: List[Dict[str, Any]] = []
        junk_count = 0
        seed_artist = artist_key(str(record["query"]["artist"]))
        for position, raw_row in enumerate(rows[:PRIMARY_K], start=1):
            row = int(raw_row)
            title, artist = str(self.titles[row]), str(self.artists[row])
            item_track_id = _track_id(self.track_ids[row])
            artist_id = artist_key(artist)
            junk = bool(self.quality.is_junk(title, artist))
            same_artist = artist_id == seed_artist
            if junk:
                junk_count += 1
            rule = tracks.get(item_track_id)
            entity = f"track:{item_track_id}" if rule else f"artist:{artist_id}"
            if rule is None:
                rule = artists.get(artist_id)
            grade = int(rule["grade"]) if rule and entity not in seen_entities else 0
            if grade:
                seen_entities.add(entity)
            grades.append(grade)
            coherence_supported = bool(
                not junk and not same_artist and artist_id in coherence_artists
            )
            result_evidence.append({
                "position": position,
                "row": row,
                "track_id": item_track_id,
                "title": title,
                "artist": artist,
                "grade": grade,
                "matched_relevance_scope": (
                    str(rule["relevance_scope"]) if rule else None
                ),
                "relevance_rationale": (
                    str(rule["rationale"]) if rule
                    else "No frozen independent sonic source supports this result."
                ),
                "coherence_supported": coherence_supported,
                "coherence_rationale": (
                    "Artist appears on the frozen Music-Map similar-artist snapshot "
                    "or is the category-A track counterpart."
                    if coherence_supported else
                    "Artist is not supported by the frozen independent sonic map."
                ),
                "uncertainty": str(rule.get("uncertainty", "high")) if rule else "high",
                "junk": junk,
                "same_artist": same_artist,
                "preview_availability": "pending_public_deezer_check",
            })
        ideal_entities: Dict[str, int] = {}
        for positive in record.get("positives", ()):
            if positive["relevance_scope"] == "track":
                key = f"track:{positive['track_id']}"
            else:
                key = f"artist:{artist_key(str(positive['artist']))}"
            ideal_entities[key] = max(
                ideal_entities.get(key, 0), int(positive["grade"])
            )
        ideal = sorted(ideal_entities.values(), reverse=True)
        ideal_dcg = _dcg(ideal)
        ndcg = _dcg(grades) / ideal_dcg if ideal_dcg else 0.0
        first = next((i for i, grade in enumerate(grades, 1) if grade), None)
        matched = sum(grade > 0 for grade in grades)
        coherent = [item["coherence_supported"] for item in result_evidence[:5]]
        coherent_count = sum(coherent)
        unrelated_top3 = sum(not value for value in coherent[:3])
        top5_junk_count = sum(
            bool(item["junk"]) for item in result_evidence[:COHERENCE_K]
        )
        top5_same_artist_count = sum(
            bool(item["same_artist"]) for item in result_evidence[:COHERENCE_K]
        )
        coherence_pass = bool(
            len(coherent) == COHERENCE_K
            and coherent_count >= 4
            and unrelated_top3 == 0
            and top5_junk_count == 0
            and top5_same_artist_count == 0
        )
        return {
            "ndcg_at_10": float(ndcg),
            "mrr_at_10": float(1.0 / first) if first else 0.0,
            "recall_at_10": float(matched / len(ideal_entities))
            if ideal_entities else 0.0,
            "matched_positive_entities": int(matched),
            "positive_entities": len(ideal_entities),
            "coherence_pass": coherence_pass,
            "coherence_fraction_at_5": float(coherent_count / 5.0),
            "unrelated_positions_1_to_3": int(unrelated_top3),
            "junk_count": int(junk_count),
            "top5_junk_count": int(top5_junk_count),
            "top5_same_artist_count": int(top5_same_artist_count),
            "result_evidence": result_evidence,
        }

    def candidate_recall(
        self, record: Mapping[str, Any], rows: Sequence[int]
    ) -> float:
        scored = self.score(record, rows)
        return float(scored["recall_at_10"]) if len(rows) <= 10 else self._candidate_recall(
            record, rows
        )

    def _candidate_recall(
        self, record: Mapping[str, Any], rows: Sequence[int]
    ) -> float:
        tracks, artists = self._rules(record)
        matched: set[str] = set()
        possible: set[str] = set()
        for positive in record.get("positives", ()):
            if positive["relevance_scope"] == "track":
                possible.add(f"track:{positive['track_id']}")
            else:
                possible.add(f"artist:{artist_key(str(positive['artist']))}")
        for raw_row in rows:
            row = int(raw_row)
            item_track_id = _track_id(self.track_ids[row])
            artist_id = artist_key(str(self.artists[row]))
            if item_track_id in tracks:
                matched.add(f"track:{item_track_id}")
            if artist_id in artists:
                matched.add(f"artist:{artist_id}")
        return float(len(matched) / len(possible)) if possible else 0.0


PredictionEvaluator = Callable[
    [ListPolicy, Sequence[Mapping[str, Any]]], Sequence[Mapping[str, Any]]
]


def deterministic_folds(
    records: Sequence[Mapping[str, Any]],
    n_splits: int = 5,
    seed: int = 20260712,
) -> List[List[Mapping[str, Any]]]:
    """Deterministic scene-round-robin folds with complete record isolation."""
    if n_splits < 2:
        raise ValueError("n_splits must be at least two")
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["scene"])].append(record)
    folds: List[List[Mapping[str, Any]]] = [[] for _ in range(n_splits)]
    for scene, values in sorted(grouped.items()):
        ordered = sorted(values, key=lambda item: hashlib.sha256(
            f"{seed}:{scene}:{item['id']}".encode()
        ).hexdigest())
        for index, record in enumerate(ordered):
            folds[index % n_splits].append(record)
    return folds


def _mean_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    fields = (
        "ndcg_at_10", "mrr_at_10", "recall_at_10",
        "coherence_fraction_at_5", "candidate_recall_at_1000",
    )
    return {
        field: float(mean(float(row.get(field, 0.0)) for row in rows))
        if rows else 0.0
        for field in fields
    }


def _relative(baseline: float, challenger: float) -> float:
    if baseline > 0:
        return float((challenger - baseline) / baseline)
    return 0.0 if challenger >= baseline else -1.0


def summarize_predictions(
    predictions: Sequence[Mapping[str, Any]],
    *,
    bootstrap_seed: int = 20260712,
) -> Dict[str, Any]:
    baseline_rows = [item["baseline"] for item in predictions]
    challenger_rows = [item["challenger"] for item in predictions]
    baseline = _mean_metrics(baseline_rows)
    challenger = _mean_metrics(challenger_rows)
    deltas = [
        float(item["challenger"]["ndcg_at_10"])
        - float(item["baseline"]["ndcg_at_10"])
        for item in predictions
    ]
    bootstrap = paired_bootstrap(deltas, seed=bootstrap_seed)
    improved = sum(delta > 1e-12 for delta in deltas)
    worsened = sum(delta < -1e-12 for delta in deltas)
    scene_rows: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for item in predictions:
        scene_rows[str(item["scene"])].append(item)
    per_scene = {}
    for scene, values in sorted(scene_rows.items()):
        left = _mean_metrics([item["baseline"] for item in values])
        right = _mean_metrics([item["challenger"] for item in values])
        per_scene[scene] = {
            "seeds": len(values),
            "baseline_ndcg_at_10": left["ndcg_at_10"],
            "challenger_ndcg_at_10": right["ndcg_at_10"],
            "relative_change": _relative(
                left["ndcg_at_10"], right["ndcg_at_10"]
            ),
        }
    baseline_coherence = float(mean(
        bool(item["baseline"]["coherence_pass"]) for item in predictions
    )) if predictions else 0.0
    challenger_coherence = float(mean(
        bool(item["challenger"]["coherence_pass"]) for item in predictions
    )) if predictions else 0.0
    relative_gain = _relative(
        baseline["ndcg_at_10"], challenger["ndcg_at_10"]
    )
    absolute_gain = challenger["ndcg_at_10"] - baseline["ndcg_at_10"]
    scene_floor = min(
        (item["relative_change"] for item in per_scene.values()), default=-1.0
    )
    junk_count = sum(
        int(item["challenger"]["junk_count"]) for item in predictions
    )
    gates = {
        "relative_gain_at_least_20pct": relative_gain >= MIN_RELATIVE_GAIN,
        "meaningful_absolute_gain": absolute_gain >= MIN_ABSOLUTE_GAIN,
        "ci95_excludes_zero": bootstrap["ci95_low"] > 0.0,
        "at_least_10_seeds_improve": improved >= MIN_IMPROVED_SEEDS,
        "every_scene_at_least_minus_10pct": scene_floor >= MAX_SCENE_REGRESSION,
        "top5_coherence_at_least_80pct":
            challenger_coherence >= MIN_COHERENCE,
        "top5_coherence_clear_margin":
            challenger_coherence - baseline_coherence >= MIN_COHERENCE_MARGIN,
        "candidate_recall_improves": (
            challenger["candidate_recall_at_1000"]
            > baseline["candidate_recall_at_1000"]
        ),
        "mrr_nonregression":
            challenger["mrr_at_10"] >= baseline["mrr_at_10"],
        "no_junk": junk_count == 0,
    }
    return {
        "seeds": len(predictions),
        "baseline": {
            **baseline,
            "coherence_pass_rate": baseline_coherence,
        },
        "challenger": {
            **challenger,
            "coherence_pass_rate": challenger_coherence,
        },
        "absolute_ndcg_gain": float(absolute_gain),
        "relative_ndcg_gain": float(relative_gain),
        "bootstrap": bootstrap,
        "improved": int(improved),
        "worsened": int(worsened),
        "unchanged": len(predictions) - int(improved) - int(worsened),
        "per_scene": per_scene,
        "worst_scene_relative_change": float(scene_floor),
        "challenger_junk_count": int(junk_count),
        "gates": gates,
        "gate_pass": all(gates.values()),
        "per_record": list(predictions),
    }


def _policy_key(policy: ListPolicy) -> Tuple[float, float, float]:
    return (policy.tau, policy.sigma, policy.audio_weight)


def select_policy(
    records: Sequence[Mapping[str, Any]],
    policies: Sequence[ListPolicy],
    evaluator: PredictionEvaluator,
    *,
    seed: int,
) -> Tuple[ListPolicy, Dict[str, Any]]:
    folds = deterministic_folds(records, 5, seed)
    scores: Dict[Tuple[float, float, float], Dict[str, Any]] = {}
    for policy in policies:
        predictions: List[Mapping[str, Any]] = []
        for validation in folds:
            predictions.extend(evaluator(policy, validation))
        summary = summarize_predictions(predictions, bootstrap_seed=seed)
        scores[_policy_key(policy)] = {
            "ndcg_at_10": summary["challenger"]["ndcg_at_10"],
            "coherence_pass_rate":
                summary["challenger"]["coherence_pass_rate"],
            "gate_firing_rate": float(mean(
                bool(item["gate"]["fired"]) for item in predictions
            )) if predictions else 0.0,
        }
    selected = max(
        policies,
        key=lambda policy: (
            scores[_policy_key(policy)]["ndcg_at_10"],
            scores[_policy_key(policy)]["coherence_pass_rate"],
            -policy.tau,
            -policy.sigma,
            -policy.audio_weight,
        ),
    )
    return selected, {
        str(_policy_key(policy)): scores[_policy_key(policy)]
        for policy in policies
    }


def nested_cross_validate(
    records: Sequence[Mapping[str, Any]],
    policies: Sequence[ListPolicy],
    evaluator: PredictionEvaluator,
    *,
    seed: int = 20260712,
) -> Dict[str, Any]:
    outer = deterministic_folds(records, 5, seed)
    predictions: List[Mapping[str, Any]] = []
    fold_reports = []
    for fold_number, test in enumerate(outer):
        test_ids = {str(item["id"]) for item in test}
        train = [item for item in records if str(item["id"]) not in test_ids]
        selected, inner_scores = select_policy(
            train, policies, evaluator, seed=seed + fold_number + 1
        )
        fold_predictions = list(evaluator(selected, test))
        predictions.extend(fold_predictions)
        fold_reports.append({
            "fold": fold_number,
            "train_ids": [str(item["id"]) for item in train],
            "test_ids": [str(item["id"]) for item in test],
            "selected_policy": asdict(selected),
            "inner_scores": inner_scores,
        })
    selected, full_scores = select_policy(
        records, policies, evaluator, seed=seed + 1000
    )
    return {
        "folds": fold_reports,
        "aggregate_outer_predictions": summarize_predictions(
            predictions, bootstrap_seed=seed
        ),
        "final_policy": asdict(selected),
        "full_dev_inner_scores": full_scores,
        "outer_labels_used_only_after_policy_selection": True,
        "selection_primary": "graded_nDCG@10",
        "coherence_role": "predeclared independent hard co-primary gate",
    }


def scene_held_out_validate(
    records: Sequence[Mapping[str, Any]],
    policies: Sequence[ListPolicy],
    evaluator: PredictionEvaluator,
    *,
    seed: int = 20260712,
) -> Dict[str, Any]:
    predictions: List[Mapping[str, Any]] = []
    folds = []
    for number, scene in enumerate(sorted({str(item["scene"]) for item in records})):
        train = [item for item in records if str(item["scene"]) != scene]
        test = [item for item in records if str(item["scene"]) == scene]
        selected, _ = select_policy(
            train, policies, evaluator, seed=seed + 2000 + number
        )
        values = list(evaluator(selected, test))
        predictions.extend(values)
        folds.append({
            "held_out_scene": scene,
            "train_scenes": sorted({str(item["scene"]) for item in train}),
            "selected_policy": asdict(selected),
            "summary": summarize_predictions(
                values, bootstrap_seed=seed + number
            ),
        })
    return {
        "folds": folds,
        "aggregate_predictions": summarize_predictions(
            predictions, bootstrap_seed=seed + 3000
        ),
        "complete_scene_isolation": all(
            fold["held_out_scene"] not in fold["train_scenes"] for fold in folds
        ),
    }


__all__ = [
    "COHERENCE_K",
    "ListGoldScorer",
    "MIN_ABSOLUTE_GAIN",
    "MIN_COHERENCE",
    "MIN_COHERENCE_MARGIN",
    "MIN_IMPROVED_SEEDS",
    "MIN_RELATIVE_GAIN",
    "PRIMARY_K",
    "deterministic_folds",
    "nested_cross_validate",
    "scene_held_out_validate",
    "select_policy",
    "summarize_predictions",
]
