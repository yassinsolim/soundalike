"""Target-blind catalogue policy selection on all legitimately opened DEV data.

The v6 file supersedes v1--v5: it is their de-duplicated Category-A superset.
Consequently those older files are inventoried, but are not concatenated (which
would overweight repeated evidence).  All v6 and all now-opened v7 records are
DEV; this module never reads or opens a protocol manifest.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .catalog_policy import (
    GRAPH_ONLY_POLICY,
    CatalogPolicy,
    CatalogPolicyRanker,
    policy_score,
)
from .catalog_protocol import _candidate_recall, _graded_rows, _per_seed
from .catalog_style import CatalogStyleIndex
from .real_benchmark import PairResolver, ProductionRanker

DEFAULT_POLICY_GRID: Tuple[CatalogPolicy, ...] = tuple(
    CatalogPolicy(audio, style, guard)
    for audio in (0.10, 0.20, 0.30)
    for style in (0.0, 0.20, 0.35)
    for guard in (0.0, 0.15, 0.25)
)
PRIMARY_WEIGHTS = {"ndcg_at_10": 0.80, "style_coherence_at_3": 0.20}


def _read_document(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return json.loads(Path(value).read_text(encoding="utf-8"))


def broad_scene(value: Any) -> str:
    """Collapse historical fine-grained labels into stable held-out scenes."""
    text = str(value or "unknown").casefold().replace("&", " and ")
    text = " ".join(
        part for part in "".join(
            character if character.isalnum() else " " for character in text
        ).split()
    )
    checks = (
        ("classical", ("classical", "baroque", "orchestral", "opera")),
        ("shoegaze-dream-pop", ("shoegaze", "dream pop")),
        ("hyperpop-digicore", ("hyperpop", "digicore")),
        ("asian-pop", ("k pop", "j pop", "city pop", "asian pop", "c pop")),
        ("latin", ("latin", "reggaeton", "salsa", "cumbia", "bachata")),
        ("african", ("afrobeat", "afrobeats", "amapiano", "african")),
        ("reggae-dub-ska", ("reggae", "dancehall", "dub", "ska")),
        ("r-and-b-soul", ("r and b", "rnb", "soul", "motown")),
        ("hip-hop", ("hip hop", "rap", "crunk", "boom bap", "trap", "drill")),
        ("metal", ("metal", "doom", "thrash", "grindcore", "metalcore")),
        ("jazz", ("jazz", "bebop", "hard bop", "big band")),
        ("electronic", ("electronic", "electro", "dance", "house", "techno", "idm", "ambient", "garage", "synthwave")),
        ("folk-country", ("folk", "country", "americana", "bluegrass")),
        ("indie-alternative", ("indie", "alternative", "post punk", "emo", "grunge", "dream pop")),
        ("rock", ("rock", "punk", "new wave", "britpop")),
        ("pop", ("pop", "boy band", "girl group")),
    )
    for scene, keywords in checks:
        if any(keyword in text for keyword in keywords):
            return scene
    return "other"


def normalize_opened_benchmarks(v6: Any, v7: Any) -> Dict[str, Any]:
    """Normalize the v6 superset and every opened v7 record into DEV records."""
    pair_doc, multi_doc = _read_document(v6), _read_document(v7)
    pairs, multipositives = pair_doc.get("pairs", []), multi_doc.get("records", [])
    records: List[Dict[str, Any]] = []
    seen = set()
    for pair in pairs:
        key = ("v6", str(pair["id"]))
        if key in seen:
            continue
        seen.add(key)
        category = str(pair.get("evidence_category", ""))
        axis = (
            "taste_affinity"
            if category == "category_a_human_songs_like"
            else "sonic_editorial"
        )
        records.append(
            {
                "id": "V6-" + str(pair["id"]),
                "split": "development",
                "opened_source_split": pair.get("split"),
                "source_version": 6,
                "scene": broad_scene(pair.get("scene", "unknown")),
                "source_scene": pair.get("scene", "unknown"),
                "catalog_tier": pair.get("catalog_tier", "unknown"),
                "query": dict(pair["query"]),
                "positives": [
                    dict(pair["target"], grade=3, relevance_scope="track")
                ],
                "evidence_axis": axis,
                "source_record": pair,
            }
        )
    for record in multipositives:
        key = ("v7", str(record["id"]))
        if key in seen:
            continue
        seen.add(key)
        normalized = dict(record)
        normalized.update(
            {
                "id": "V7-" + str(record["id"]),
                "split": "development",
                "opened_source_split": record.get("split"),
                "source_version": 7,
                "scene": broad_scene(record.get("scene", "unknown")),
                "source_scene": record.get("scene", "unknown"),
                "evidence_axis": "taste_affinity",
                "query": dict(record["query"]),
                "positives": [dict(value) for value in record["positives"]],
                "source_record": record,
            }
        )
        records.append(normalized)

    axes: Dict[str, int] = {}
    for record in records:
        axis = record["evidence_axis"]
        axes[axis] = axes.get(axis, 0) + 1
    return {
        "records": records,
        "benchmark_inventory": {
            "included": {
                "soundalike_pairs.v6": len(pairs),
                "soundalike_multipositive.v7": len(multipositives),
            },
            "excluded_or_superseded": {
                "soundalike_pairs.v1-v5": (
                    "superseded by de-duplicated Category-A v6 superset; "
                    "concatenating them would duplicate opened evidence"
                )
            },
            "normalized_dev_records": len(records),
            "axis_counts": axes,
            "all_v7_opened_records_included": len(multipositives) == 100,
            "no_unopened_final_labels_compared": True,
        },
    }


def policy_key(policy: CatalogPolicy) -> Tuple[float, float, float]:
    return (policy.audio_weight, policy.style_weight, policy.style_guard_min)


def precompute_query_components(
    ranker: CatalogPolicyRanker,
    production: ProductionRanker,
    query_row: int,
    candidate_limit: int = 1000,
) -> Dict[str, Any]:
    """Compute graph/audio/style once and capture live ``dual_sonic`` output."""
    # Capture the complete fixed pool (<= 96*16 graph tracks + 1000 audio)
    # before any policy is selected, so later rescoring cannot be truncated by
    # the default policy used for target-blind component extraction.
    payload = ranker.recommend(query_row, n=max(3000, int(candidate_limit)))
    production_rows = [
        int(row)
        for row in production.rank(query_row, "dual_sonic", n=candidate_limit)
    ]
    components = [
        {
            "row": int(item["row"]),
            "G": float(item["rationale"]["G"]),
            "A": float(item["rationale"]["A"]),
            "S": float(item["rationale"]["S"]),
            "source": item["rationale"]["source"],
        }
        for item in payload["results"]
    ]
    return {
        "query_row": int(query_row),
        "components": components,
        "production_method": "dual_sonic",
        "production_rows": production_rows,
        "graph_rows": [
            value["row"] for value in components if value["source"] == "graph"
        ],
        # Fixed graph candidates plus the ranker's audio bridge/fallback pool.
        "graph_union_rows": list(
            dict.fromkeys(value["row"] for value in components)
        ),
    }


def rescore_components(
    components: Sequence[Mapping[str, Any]],
    policy: CatalogPolicy,
    n: int = 10,
) -> List[Dict[str, Any]]:
    """Rescore cached target-blind components without catalogue audio work."""
    ranked = []
    for component in components:
        item = dict(component)
        item["score"] = policy_score(item["G"], item["A"], item["S"], policy)
        ranked.append(item)
    ranked.sort(key=lambda item: (-item["score"], int(item["row"])))
    guarded = min(3, max(0, int(n)))
    safe = [item for item in ranked if item["S"] >= policy.style_guard_min]
    if len(safe) >= guarded:
        protected = safe[:guarded]
        protected_rows = {int(item["row"]) for item in protected}
        ranked = protected + [
            item for item in ranked if int(item["row"]) not in protected_rows
        ]
    return ranked[: max(0, int(n))]


def composite_primary(ndcg_at_10: float, style_coherence_at_3: float) -> float:
    """Predeclared primary; affinity relevance and sonic/editorial style stay separate."""
    return 0.80 * float(ndcg_at_10) + 0.20 * float(style_coherence_at_3)


def style_coherence_at_3(
    ranking: Sequence[Mapping[str, Any]],
    styles: Optional[CatalogStyleIndex] = None,
    query_artist: Optional[str] = None,
) -> float:
    """Mean independent MusicBrainz style overlap for the first three results."""
    if styles is not None and query_artist is not None:
        values = [
            styles.style_overlap(query_artist, str(item["artist"]))
            for item in ranking[:3]
        ]
    else:
        values = [
            float(item.get("S", item.get("style_overlap", 0.0)))
            for item in ranking[:3]
        ]
    return float(mean(values)) if values else 0.0


def candidate_recall(
    candidate_rows: Sequence[int],
    relevance: Mapping[int, Tuple[str, int]],
    cutoff: int = 1000,
) -> float:
    return float(_candidate_recall(candidate_rows, relevance, cutoff))


def candidate_recall_comparison(
    production_rows: Sequence[int],
    graph_union_rows: Sequence[int],
    relevance: Mapping[int, Tuple[str, int]],
    cutoff: int = 1000,
) -> Dict[str, Any]:
    production_value = candidate_recall(production_rows, relevance, cutoff)
    union_value = candidate_recall(graph_union_rows, relevance, cutoff)
    return {
        "production_dual_sonic": production_value,
        "graph_union": union_value,
        "improves": union_value > production_value,
    }


def evaluate_seed(
    ranking: Sequence[Mapping[str, Any]],
    relevance: Mapping[int, Tuple[str, int]],
    styles: Optional[CatalogStyleIndex] = None,
    query_artist: Optional[str] = None,
) -> Dict[str, float]:
    serialized = [{"row": int(item["row"])} for item in ranking]
    result = dict(_per_seed(serialized, relevance))
    result["style_coherence_at_3"] = style_coherence_at_3(
        ranking, styles, query_artist
    )
    result["composite_primary"] = composite_primary(
        result["ndcg_at_10"], result["style_coherence_at_3"]
    )
    return result


def resolve_relevance(
    resolver: PairResolver, record: Mapping[str, Any]
) -> Dict[int, Tuple[str, int]]:
    return _graded_rows(resolver, record)


def deterministic_folds(
    records: Sequence[Mapping[str, Any]], n_splits: int = 5, seed: int = 20260712
) -> List[List[Mapping[str, Any]]]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least two")
    ordered = sorted(
        records,
        key=lambda record: hashlib.sha256(
            ("%s:%s" % (seed, record["id"])).encode("utf-8")
        ).hexdigest(),
    )
    return [ordered[index::n_splits] for index in range(n_splits)]


def scene_held_out_folds(
    records: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Return folds whose training side contains no held-out scene."""
    scenes = sorted({str(record["scene"]) for record in records})
    return [
        {
            "scene": scene,
            "train": [record for record in records if str(record["scene"]) != scene],
            "test": [record for record in records if str(record["scene"]) == scene],
        }
        for scene in scenes
    ]


def select_policy(
    scores: Mapping[CatalogPolicy, Sequence[float]],
) -> CatalogPolicy:
    """Select deterministically by mean score, then lexicographic policy values."""
    if not scores:
        raise ValueError("at least one policy is required")
    return min(
        scores,
        key=lambda policy: (
            -float(mean(scores[policy])) if scores[policy] else math.inf,
            policy_key(policy),
        ),
    )


Evaluator = Callable[
    [CatalogPolicy, Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]],
    Mapping[str, float],
]


def _selection_primary(metrics: Mapping[str, Any]) -> float:
    values = metrics.get("challenger", metrics)
    return float(values["composite_primary"])


def _inner_select(
    training: Sequence[Mapping[str, Any]],
    policies: Sequence[CatalogPolicy],
    evaluator: Evaluator,
    seed: int,
) -> Tuple[CatalogPolicy, Dict[CatalogPolicy, List[float]]]:
    fold_count = min(5, len(training))
    if fold_count < 2:
        raise ValueError("policy selection needs at least two training records")
    folds = deterministic_folds(training, fold_count, seed)
    scores = {policy: [] for policy in policies}
    for validation in folds:
        validation_ids = {record["id"] for record in validation}
        fit = [record for record in training if record["id"] not in validation_ids]
        for policy in policies:
            metrics = evaluator(policy, fit, validation)
            scores[policy].append(_selection_primary(metrics))
    return select_policy(scores), scores


def nested_cross_validate(
    records: Sequence[Mapping[str, Any]],
    evaluator: Evaluator,
    policies: Sequence[CatalogPolicy] = DEFAULT_POLICY_GRID,
    seed: int = 20260712,
) -> Dict[str, Any]:
    """Deterministic nested five-fold CV; outer labels are used only after selection."""
    policies = tuple(policies)
    if GRAPH_ONLY_POLICY not in policies:
        policies = (GRAPH_ONLY_POLICY,) + policies
    outer = deterministic_folds(records, 5, seed)
    results = []
    for number, test in enumerate(outer):
        test_ids = {record["id"] for record in test}
        train = [record for record in records if record["id"] not in test_ids]
        selected, inner = _inner_select(train, policies, evaluator, seed + number + 1)
        metrics = dict(evaluator(selected, train, test))
        results.append(
            {
                "fold": number,
                "train_ids": [record["id"] for record in train],
                "test_ids": [record["id"] for record in test],
                "selected_policy": asdict(selected),
                "inner_scores": {
                    str(policy_key(policy)): values for policy, values in inner.items()
                },
                "outer_metrics": metrics,
            }
        )
    # Exact deployable policy is chosen from inner validation over all opened DEV.
    final_policy, full_scores = _inner_select(records, policies, evaluator, seed + 1000)
    return {
        "folds": results,
        "final_policy": asdict(final_policy),
        "full_dev_inner_scores": {
            str(policy_key(policy)): values for policy, values in full_scores.items()
        },
        "selection_data": "opened_DEV_only",
    }


def relative_change(baseline: float, challenger: float) -> float:
    baseline, challenger = float(baseline), float(challenger)
    if baseline == 0.0:
        if challenger == 0.0:
            return 0.0
        return math.inf if challenger > 0.0 else -math.inf
    return (challenger - baseline) / baseline


def gate_scene_fold(
    baseline: Mapping[str, float],
    challenger: Mapping[str, float],
    baseline_candidate_recall: float,
    challenger_candidate_recall: float,
    per_scene: Mapping[str, Tuple[float, float]],
) -> Dict[str, Any]:
    scene_values = {
        scene: {
            "baseline_composite_primary": float(values[0]),
            "challenger_composite_primary": float(values[1]),
            "relative_change": relative_change(values[0], values[1]),
        }
        for scene, values in sorted(per_scene.items())
    }
    primary_change = relative_change(
        baseline["composite_primary"], challenger["composite_primary"]
    )
    checks = {
        "composite_relative_gain_at_least_20_percent": primary_change >= 0.20
        or math.isclose(primary_change, 0.20, rel_tol=0.0, abs_tol=1e-12),
        "every_scene_above_minus_10_percent": all(
            value["relative_change"] >= -0.10
            or math.isclose(
                value["relative_change"], -0.10, rel_tol=0.0, abs_tol=1e-12
            )
            for value in scene_values.values()
        ),
        "candidate_recall_at_1000_improves": (
            float(challenger_candidate_recall) > float(baseline_candidate_recall)
        ),
        "mrr_at_10_non_regression": (
            challenger["mrr_at_10"] >= baseline["mrr_at_10"]
        ),
        "recall_at_10_non_regression": (
            challenger["recall_at_10"] >= baseline["recall_at_10"]
        ),
    }
    return {
        "passes": checks,
        "per_scene": scene_values,
        "gate_pass": all(checks.values()),
    }


def scene_held_out_validate(
    records: Sequence[Mapping[str, Any]],
    evaluator: Evaluator,
    policies: Sequence[CatalogPolicy] = DEFAULT_POLICY_GRID,
    seed: int = 20260712,
) -> Dict[str, Any]:
    """Select inside each training partition, then evaluate its absent scene."""
    policies = tuple(policies)
    if GRAPH_ONLY_POLICY not in policies:
        policies = (GRAPH_ONLY_POLICY,) + policies
    output = []
    for number, fold in enumerate(scene_held_out_folds(records)):
        selected, _ = _inner_select(
            fold["train"], policies, evaluator, seed + number + 2000
        )
        metrics = dict(evaluator(selected, fold["train"], fold["test"]))
        row = {
            "scene": fold["scene"],
            "train_ids": [record["id"] for record in fold["train"]],
            "test_ids": [record["id"] for record in fold["test"]],
            "selected_policy": asdict(selected),
            "metrics": metrics,
        }
        if "baseline" in metrics and "challenger" in metrics:
            row["gate"] = gate_scene_fold(
                metrics["baseline"],
                metrics["challenger"],
                metrics["baseline_candidate_recall_at_1000"],
                metrics["challenger_candidate_recall_at_1000"],
                {
                    fold["scene"]: (
                        metrics["baseline"]["composite_primary"],
                        metrics["challenger"]["composite_primary"],
                    )
                },
            )
        output.append(row)
    gates = [row["gate"]["gate_pass"] for row in output if "gate" in row]
    return {
        "folds": output,
        "scene_isolation": True,
        "all_reported_gates_pass": all(gates) if gates else None,
    }


def build_catalog_cv_report(
    v6: Any,
    v7: Any,
    evaluator: Evaluator,
    policies: Sequence[CatalogPolicy] = DEFAULT_POLICY_GRID,
    seed: int = 20260712,
) -> Dict[str, Any]:
    """Build the complete opened-DEV report without consulting any FINAL protocol."""
    normalized = normalize_opened_benchmarks(v6, v7)
    records = normalized["records"]
    return {
        "benchmark_inventory": normalized["benchmark_inventory"],
        "axis_policy": {
            "sonic_editorial": "v6 documented sonic/editorial evidence",
            "taste_affinity": "human songs-like and v7 graded artist positives",
            "reported_separately": True,
        },
        "primary": {
            "name": "composite_primary",
            "formula": "0.80*nDCG@10 + 0.20*style_coherence@3",
        },
        "nested_5fold": nested_cross_validate(records, evaluator, policies, seed),
        "scene_held_out": scene_held_out_validate(
            records, evaluator, policies, seed
        ),
        "final_policy_selection_source": "full opened DEV nested procedure only",
        "no_unopened_final_labels_compared": True,
    }


__all__ = [
    "CatalogPolicy", "CatalogPolicyRanker", "CatalogStyleIndex", "PairResolver",
    "ProductionRanker", "GRAPH_ONLY_POLICY", "DEFAULT_POLICY_GRID",
    "normalize_opened_benchmarks", "precompute_query_components",
    "rescore_components", "composite_primary", "style_coherence_at_3",
    "candidate_recall", "candidate_recall_comparison", "evaluate_seed",
    "resolve_relevance",
    "deterministic_folds", "scene_held_out_folds", "select_policy",
    "nested_cross_validate", "scene_held_out_validate", "relative_change",
    "gate_scene_fold", "build_catalog_cv_report",
]
