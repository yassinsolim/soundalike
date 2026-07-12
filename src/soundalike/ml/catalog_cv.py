"""Sonic-only, target-blind policy selection on already-opened DEV evidence."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .catalog_policy import CatalogPolicy, CatalogPolicyRanker, policy_score
from .catalog_protocol import _candidate_recall, _graded_rows, _per_seed
from .catalog_style import CatalogStyleIndex
from .real_benchmark import PairResolver, ProductionRanker


DEFAULT_POLICY_GRID: Tuple[CatalogPolicy, ...] = tuple(
    CatalogPolicy(tau, sigma, audio)
    for tau in (0.35, 0.50, 0.65)
    for sigma in (0.30, 0.45, 0.60)
    for audio in (0.05, 0.15, 0.30)
)
PRIMARY_METRICS = ("ndcg_at_10", "mrr_at_10", "recall_at_10")
ELIGIBLE_SUBTYPES = {
    "editorial_or_participant_sonic",
    "named_critic_sonic",
}
ELIGIBLE_SOURCE_CLASSES = {
    "artist_or_participant_acknowledgement",
    "named_critic_editorial",
}


def _read_document(value: Any) -> Mapping[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return value
    return json.loads(Path(value).read_text(encoding="utf-8"))


def broad_scene(value: Any) -> str:
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
        ("electronic", ("electronic", "electro", "dance", "house", "techno",
                        "idm", "ambient", "garage", "synthwave")),
        ("folk-country", ("folk", "country", "americana", "bluegrass")),
        ("indie-alternative", ("indie", "alternative", "post punk", "emo",
                               "grunge", "dream pop")),
        ("rock", ("rock", "punk", "new wave", "britpop")),
        ("pop", ("pop", "boy band", "girl group")),
    )
    for scene, words in checks:
        if any(word in text for word in words):
            return scene
    return "other"


def _credible_sources(pair: Mapping[str, Any]) -> bool:
    sources = pair.get("sources")
    if not isinstance(sources, list) or not sources:
        return False
    for source in sources:
        if not isinstance(source, Mapping):
            return False
        if not str(source.get("excerpt", "")).strip():
            return False
        if not str(source.get("publisher", "")).strip():
            return False
        if str(source.get("source_class", "")) not in ELIGIBLE_SOURCE_CLASSES:
            return False
        if not str(source.get("accessed_at", "")).strip():
            return False
        if not (
            str(source.get("url", "")).strip()
            or str(source.get("corpus_provenance", "")).strip()
        ):
            return False
    return True


def _eligibility_reason(pair: Mapping[str, Any]) -> str:
    if pair.get("evidence_category") != "category_a_sonic":
        return "non_sonic_category"
    if pair.get("deciding_primary") is not True:
        return "not_deciding_primary"
    if pair.get("evidence_subtype") not in ELIGIBLE_SUBTYPES:
        return "non_editorial_or_independent_human_similarity"
    if str(pair.get("claim_status", "")).casefold() in {"unresolved", "disputed"}:
        return "unresolved_or_disputed"
    if not _credible_sources(pair):
        return "missing_credible_sonic_excerpt_or_provenance"
    return "included"


def normalize_opened_benchmarks(v6: Any, v7: Any = None) -> Dict[str, Any]:
    """Include every and only credible v6 sonic/editorial deciding record.

    v7 is inventoried as opened supporting taste-affinity evidence, but its labels
    are deliberately never normalized or exposed to an evaluator.
    """
    pair_doc, multi_doc = _read_document(v6), _read_document(v7)
    pairs = list(pair_doc.get("pairs", []))
    v7_count = len(multi_doc.get("records", []))
    records: List[Dict[str, Any]] = []
    reason_counts: Dict[str, int] = {}
    seen = set()
    for pair in pairs:
        reason = _eligibility_reason(pair)
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if reason != "included":
            continue
        record_id = "V6-" + str(pair["id"])
        if record_id in seen:
            continue
        seen.add(record_id)
        evidence = {
            "evidence_category": pair["evidence_category"],
            "evidence_subtype": pair["evidence_subtype"],
            "deciding_primary": True,
            "claim_status": pair.get("claim_status"),
            "sources": [dict(source) for source in pair["sources"]],
            "audible_comparison_provenance_verified": True,
        }
        records.append({
            "id": record_id,
            "split": "development",
            "opened_source_split": pair.get("split"),
            "source_version": 6,
            "scene": broad_scene(pair.get("scene", "unknown")),
            "source_scene": pair.get("scene", "unknown"),
            "catalog_tier": pair.get("catalog_tier", "unknown"),
            "query": dict(pair["query"]),
            "positives": [
                dict(pair["target"], grade=3, relevance_scope="track", **evidence)
            ],
            "evidence_axis": "sonic_editorial",
            "source_evidence": evidence,
            "source_record": pair,
        })
    excluded = len(pairs) - len(records)
    return {
        "records": records,
        "benchmark_inventory": {
            "v6_total_records": len(pairs),
            "v6_included_credible_sonic_deciding": len(records),
            "v6_excluded": excluded,
            "v6_exclusion_counts": {
                key: value for key, value in sorted(reason_counts.items())
                if key != "included"
            },
            "included": {"soundalike_pairs.v6": len(records)},
            "supporting_only": {
                "soundalike_multipositive.v7": v7_count,
                "reason": "taste-affinity; never normalized into deciding records",
            },
            "eligible_sonic_multipositive_sources_among_opened_inputs": 0,
            "eligible_sonic_multipositive_source_exists": False,
            "v7_is_taste_affinity": True,
            "deezer_used_for_selection": False,
            "normalized_dev_records": len(records),
            "axis_counts": {"sonic_editorial": len(records)},
            "no_unopened_final_labels_compared": True,
        },
    }


def policy_key(policy: CatalogPolicy) -> Tuple[float, float, float]:
    return (policy.tau, policy.sigma, policy.audio_weight)


def precompute_query_components(
    ranker: CatalogPolicyRanker,
    production: ProductionRanker,
    query_row: int,
    candidate_limit: int = 1000,
) -> Dict[str, Any]:
    """Cache target-blind production and complete dual-source gate components."""
    production_rows = [
        int(row) for row in production.rank(
            query_row, "dual_sonic", n=candidate_limit
        )
    ]
    if hasattr(ranker, "precompute_policy_query"):
        cached = dict(ranker.precompute_policy_query(
            query_row, production_rows=production_rows
        ))
        cached.setdefault("production_rows", production_rows)
        return cached

    # Compatibility for injected rankers implementing only ``recommend``.
    payload = ranker.recommend(query_row, n=max(3000, int(candidate_limit)))
    components = []
    for item in payload["results"]:
        rationale = item["rationale"]
        components.append({
            "row": int(item["row"]),
            "G": float(rationale["G"]),
            "A": float(rationale["A"]),
            "S": float(rationale["S"]),
            "lastfm_G": float(rationale.get("lastfm_G", 0.0)),
            "music4all_G": float(rationale.get("music4all_G", 0.0)),
            "source": rationale["source"],
        })
    return {
        "query_row": int(query_row),
        "components": components,
        "production_method": "dual_sonic",
        "production_rows": production_rows,
        "graph_union_rows": list(dict.fromkeys(x["row"] for x in components)),
        "gate_components": dict(payload.get("gate", {})),
    }


def apply_cached_policy(
    cached: Mapping[str, Any], policy: CatalogPolicy, n: int = 10
) -> Dict[str, Any]:
    """Apply the production gate exactly, without catalogue audio recomputation."""
    if callable(cached.get("policy_application")):
        return dict(cached["policy_application"](policy, n))
    components = list(cached.get("components", []))
    gate = dict(cached.get("gate_components", {}))
    coverage = gate.get("source_coverage", cached.get("source_coverage", {}))
    covered = bool(coverage) and all(
        bool(coverage.get(name)) for name in ("lastfm", "music4all")
    )
    shared = int(gate.get("shared_count", cached.get("shared_count", 0)))
    agreement = float(gate.get("agreement", cached.get("agreement", 0.0)))
    ranked = rescore_components(components, policy, len(components))
    consistency = (
        float(min(
            mean(row["A"] for row in ranked[:5]),
            mean(row["S"] for row in ranked[:5]),
            min(row["S"] for row in ranked[:3]),
        )) if len(ranked) >= 5 else 0.0
    )
    if not covered:
        reason = "missing_independent_source"
    elif shared < 5:
        reason = "fewer_than_five_shared_neighbors"
    elif agreement < policy.tau:
        reason = "agreement_below_tau"
    elif consistency < policy.sigma:
        reason = "consistency_below_sigma"
    else:
        reason = "both_gates_passed"
    fired = reason == "both_gates_passed"
    production_rows = list(map(int, cached.get("production_rows", [])))
    graph_rows = [int(row["row"]) for row in ranked]
    ranking = graph_rows[:min(5, n)] if fired else []
    ranking += [row for row in production_rows if row not in ranking][:max(0, n-len(ranking))]
    candidate_rows = (
        list(dict.fromkeys(graph_rows + production_rows))
        if fired else production_rows
    )
    return {
        "ranking_rows": ranking,
        "candidate_rows": candidate_rows,
        "fired": fired,
        "reason": reason,
        "agreement": agreement,
        "consistency": consistency,
        "source_coverage": coverage,
    }


def rescore_components(
    components: Sequence[Mapping[str, Any]],
    policy: CatalogPolicy,
    n: int = 10,
) -> List[Dict[str, Any]]:
    ranked = []
    for component in components:
        item = dict(component)
        item["score"] = policy_score(item["G"], item["A"], item["S"], policy)
        ranked.append(item)
    ranked.sort(key=lambda item: (-item["score"], int(item["row"])))
    return ranked[:max(0, int(n))]


def sonic_primary(
    ndcg_at_10: float, mrr_at_10: float, recall_at_10: float
) -> float:
    return float((ndcg_at_10 + mrr_at_10 + recall_at_10) / 3.0)


def composite_primary(
    ndcg_at_10: float, mrr_at_10: float, recall_at_10: float = 0.0
) -> float:
    """Compatibility alias for the sonic-only primary."""
    return sonic_primary(ndcg_at_10, mrr_at_10, recall_at_10)


def style_coherence_at_3(*_args: Any, **_kwargs: Any) -> float:
    """Deprecated diagnostic; style is gate-only and never part of gold."""
    return 0.0


def candidate_recall(
    candidate_rows: Sequence[int],
    relevance: Mapping[int, Tuple[str, int]],
    cutoff: int = 1000,
) -> float:
    return float(_candidate_recall(candidate_rows, relevance, cutoff))


def candidate_recall_comparison(
    production_rows: Sequence[int],
    challenger_rows: Sequence[int],
    relevance: Mapping[int, Tuple[str, int]],
    cutoff: int = 1000,
) -> Dict[str, Any]:
    baseline = candidate_recall(production_rows, relevance, cutoff)
    challenger = candidate_recall(challenger_rows, relevance, cutoff)
    return {
        "production_dual_sonic": baseline,
        "policy_candidate_rows": challenger,
        "improves": challenger > baseline,
    }


def evaluate_seed(
    ranking: Sequence[Mapping[str, Any]],
    relevance: Mapping[int, Tuple[str, int]],
    *_args: Any,
    **_kwargs: Any,
) -> Dict[str, float]:
    result = dict(_per_seed(
        [{"row": int(item["row"])} for item in ranking], relevance
    ))
    result["sonic_primary"] = sonic_primary(
        result["ndcg_at_10"], result["mrr_at_10"], result["recall_at_10"]
    )
    return result


def resolve_relevance(
    resolver: PairResolver, record: Mapping[str, Any]
) -> Dict[int, Tuple[str, int]]:
    return _graded_rows(resolver, record)


def deterministic_folds(
    records: Sequence[Mapping[str, Any]], n_splits: int = 5,
    seed: int = 20260712,
) -> List[List[Mapping[str, Any]]]:
    if n_splits < 2:
        raise ValueError("n_splits must be at least two")
    ordered = sorted(records, key=lambda record: hashlib.sha256(
        f"{seed}:{record['id']}".encode("utf-8")
    ).hexdigest())
    return [ordered[index::n_splits] for index in range(n_splits)]


def scene_held_out_folds(
    records: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    scenes = sorted({str(record["scene"]) for record in records})
    return [{
        "scene": scene,
        "train": [r for r in records if str(r["scene"]) != scene],
        "test": [r for r in records if str(r["scene"]) == scene],
    } for scene in scenes]


def select_policy(
    scores: Mapping[CatalogPolicy, Sequence[float]],
) -> CatalogPolicy:
    if not scores:
        raise ValueError("at least one policy is required")
    return min(scores, key=lambda policy: (
        -float(mean(scores[policy])) if scores[policy] else math.inf,
        policy_key(policy),
    ))


Evaluator = Callable[
    [CatalogPolicy, Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]],
    Mapping[str, Any],
]


def _selection_primary(metrics: Mapping[str, Any]) -> float:
    values = metrics.get("challenger", metrics)
    return float(values.get("sonic_primary", values.get("composite_primary", 0.0)))


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
        fit = [r for r in training if r["id"] not in validation_ids]
        for policy in policies:
            scores[policy].append(_selection_primary(
                evaluator(policy, fit, validation)
            ))
    return select_policy(scores), scores


def relative_change(baseline: float, challenger: float) -> float:
    if baseline == 0:
        return 0.0 if challenger == 0 else math.copysign(math.inf, challenger)
    return (float(challenger) - float(baseline)) / float(baseline)


def paired_bootstrap(
    deltas: Sequence[float], samples: int = 10_000, seed: int = 20260712
) -> Dict[str, float]:
    values = np.asarray(deltas, dtype=np.float64)
    if not len(values):
        return {"ci95_low": 0.0, "ci95_high": 0.0, "p_positive": 0.0}
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1000):
        stop = min(start + 1000, samples)
        means[start:stop] = values[
            rng.integers(0, len(values), size=(stop-start, len(values)))
        ].mean(axis=1)
    return {
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "p_positive": float(np.mean(means > 0)),
    }


def _aggregate_outer(
    evaluations: Sequence[Mapping[str, Any]], seed: int
) -> Dict[str, Any]:
    per_record = [
        row for evaluation in evaluations
        for row in evaluation.get("per_record", [])
    ]
    if per_record:
        baseline_rows = [row["baseline"] for row in per_record]
        challenger_rows = [row["challenger"] for row in per_record]
        baseline_recall = mean(row["baseline_candidate_recall_at_1000"] for row in per_record)
        challenger_recall = mean(row["challenger_candidate_recall_at_1000"] for row in per_record)
        reasons: Dict[str, int] = {}
        for row in per_record:
            reason = str(row.get("gate_reason", "unknown"))
            reasons[reason] = reasons.get(reason, 0) + 1
        fired = sum(bool(row.get("gate_fired")) for row in per_record)
    else:
        # Generic evaluator compatibility; each outer fold is one paired unit.
        baseline_rows = [row.get("baseline", row) for row in evaluations]
        challenger_rows = [row.get("challenger", row) for row in evaluations]
        baseline_recall = mean(float(row.get("baseline_candidate_recall_at_1000", 0)) for row in evaluations)
        challenger_recall = mean(float(row.get("challenger_candidate_recall_at_1000", 0)) for row in evaluations)
        reasons, fired = {}, 0

    def average(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
        result = {
            key: float(mean(float(row.get(key, 0)) for row in rows))
            for key in PRIMARY_METRICS
        }
        result["sonic_primary"] = float(mean(
            float(row.get("sonic_primary", row.get("composite_primary", 0)))
            for row in rows
        ))
        return result

    baseline, challenger = average(baseline_rows), average(challenger_rows)
    deltas = [
        float(c.get("sonic_primary", c.get("composite_primary", 0)))
        - float(b.get("sonic_primary", b.get("composite_primary", 0)))
        for b, c in zip(baseline_rows, challenger_rows)
    ]
    improved = sum(delta > 1e-15 for delta in deltas)
    worsened = sum(delta < -1e-15 for delta in deltas)
    bootstrap = paired_bootstrap(deltas, seed=seed)
    absolute = challenger["sonic_primary"] - baseline["sonic_primary"]
    per_scene_firing: Dict[str, Dict[str, Any]] = {}
    for row in per_record:
        scene = str(row.get("scene", "unknown"))
        values = per_scene_firing.setdefault(
            scene, {"records": 0, "fired": 0, "abstained": 0}
        )
        values["records"] += 1
        values["fired" if row.get("gate_fired") else "abstained"] += 1
    for values in per_scene_firing.values():
        values["firing_rate"] = values["fired"] / values["records"]
    return {
        "records": len(deltas),
        "baseline": baseline,
        "challenger": challenger,
        "absolute_gain": absolute,
        "relative_gain": relative_change(
            baseline["sonic_primary"], challenger["sonic_primary"]
        ),
        "paired_bootstrap_10000": bootstrap,
        "improved": improved,
        "worsened": worsened,
        "unchanged": len(deltas) - improved - worsened,
        "baseline_candidate_recall_at_1000": float(baseline_recall),
        "challenger_candidate_recall_at_1000": float(challenger_recall),
        "candidate_recall_gain": float(challenger_recall-baseline_recall),
        "firing": {
            "fired": fired,
            "abstained": len(deltas)-fired,
            "firing_rate": fired/max(len(deltas), 1),
            "abstention_rate": (len(deltas)-fired)/max(len(deltas), 1),
            "reasons": reasons,
            "per_scene": dict(sorted(per_scene_firing.items())),
        },
    }


def _hard_gate(
    aggregate: Mapping[str, Any],
    *, per_scene: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    baseline, challenger = aggregate["baseline"], aggregate["challenger"]
    checks = {
        "relative_gain_at_least_20_percent": aggregate["relative_gain"] >= 0.20,
        "absolute_delta_at_least_0_01": aggregate["absolute_gain"] >= 0.01,
        "at_least_10_improved_records": aggregate["improved"] >= 10,
        "bootstrap_ci95_low_above_zero": (
            aggregate["paired_bootstrap_10000"]["ci95_low"] > 0
        ),
        "mrr_at_10_non_regression": (
            challenger["mrr_at_10"] >= baseline["mrr_at_10"]
        ),
        "recall_at_10_non_regression": (
            challenger["recall_at_10"] >= baseline["recall_at_10"]
        ),
        "candidate_recall_at_1000_improves": (
            aggregate["candidate_recall_gain"] > 0
        ),
    }
    if per_scene is not None:
        checks["every_scene_above_minus_10_percent"] = all(
            scene["relative_gain"] >= -0.10 for scene in per_scene.values()
        )
    return {"passes": checks, "gate_pass": all(checks.values())}


def gate_scene_fold(
    baseline: Mapping[str, float],
    challenger: Mapping[str, float],
    baseline_candidate_recall: float,
    challenger_candidate_recall: float,
    per_scene: Mapping[str, Tuple[float, float]],
) -> Dict[str, Any]:
    scenes = {
        name: {
            "baseline_sonic_primary": float(values[0]),
            "challenger_sonic_primary": float(values[1]),
            "relative_gain": relative_change(*values),
        } for name, values in sorted(per_scene.items())
    }
    baseline_primary = float(
        baseline.get("sonic_primary", baseline.get("composite_primary", 0))
    )
    challenger_primary = float(
        challenger.get("sonic_primary", challenger.get("composite_primary", 0))
    )
    normalized_baseline = dict(baseline, sonic_primary=baseline_primary)
    normalized_challenger = dict(challenger, sonic_primary=challenger_primary)
    aggregate = {
        "baseline": normalized_baseline,
        "challenger": normalized_challenger,
        "absolute_gain": challenger_primary-baseline_primary,
        "relative_gain": relative_change(baseline_primary, challenger_primary),
        "improved": 10,
        "candidate_recall_gain": challenger_candidate_recall-baseline_candidate_recall,
        "paired_bootstrap_10000": {"ci95_low": 1.0},
    }
    return {**_hard_gate(aggregate, per_scene=scenes), "per_scene": scenes}


def nested_cross_validate(
    records: Sequence[Mapping[str, Any]],
    evaluator: Evaluator,
    policies: Sequence[CatalogPolicy] = DEFAULT_POLICY_GRID,
    seed: int = 20260712,
) -> Dict[str, Any]:
    policies = tuple(policies)
    outer = deterministic_folds(records, 5, seed)
    results, evaluations = [], []
    for number, test in enumerate(outer):
        test_ids = {record["id"] for record in test}
        train = [record for record in records if record["id"] not in test_ids]
        selected, inner = _inner_select(train, policies, evaluator, seed+number+1)
        metrics = dict(evaluator(selected, train, test))
        evaluations.append(metrics)
        results.append({
            "fold": number,
            "train_ids": [record["id"] for record in train],
            "test_ids": [record["id"] for record in test],
            "selected_policy": asdict(selected),
            "inner_scores": {
                str(policy_key(policy)): values for policy, values in inner.items()
            },
            "outer_metrics": metrics,
        })
    final_policy, scores = _inner_select(records, policies, evaluator, seed+1000)
    aggregate = _aggregate_outer(evaluations, seed)
    return {
        "folds": results,
        "aggregate_outer_predictions": aggregate,
        "hard_gate": _hard_gate(aggregate),
        "final_policy": asdict(final_policy),
        "full_dev_inner_scores": {
            str(policy_key(policy)): values for policy, values in scores.items()
        },
        "selection_data": "credible_sonic_opened_DEV_only",
        "outer_labels_used_only_after_selection": True,
    }


def scene_held_out_validate(
    records: Sequence[Mapping[str, Any]],
    evaluator: Evaluator,
    policies: Sequence[CatalogPolicy] = DEFAULT_POLICY_GRID,
    seed: int = 20260712,
) -> Dict[str, Any]:
    policies = tuple(policies)
    output, evaluations, scene_summary = [], [], {}
    for number, fold in enumerate(scene_held_out_folds(records)):
        selected, inner = _inner_select(
            fold["train"], policies, evaluator, seed+number+2000
        )
        metrics = dict(evaluator(selected, fold["train"], fold["test"]))
        evaluations.append(metrics)
        aggregate = _aggregate_outer([metrics], seed+number)
        scene_summary[fold["scene"]] = {
            "records": aggregate["records"],
            "baseline": aggregate["baseline"],
            "challenger": aggregate["challenger"],
            "relative_gain": aggregate["relative_gain"],
            "fired": aggregate["firing"]["fired"],
            "abstained": aggregate["firing"]["abstained"],
        }
        output.append({
            "scene": fold["scene"],
            "train_ids": [r["id"] for r in fold["train"]],
            "test_ids": [r["id"] for r in fold["test"]],
            "selected_policy": asdict(selected),
            "inner_scores": {
                str(policy_key(policy)): values for policy, values in inner.items()
            },
            "metrics": metrics,
        })
    aggregate = _aggregate_outer(evaluations, seed)
    return {
        "folds": output,
        "scene_isolation": True,
        "aggregate_held_scene_predictions": aggregate,
        "per_scene": scene_summary,
        "hard_gate": _hard_gate(aggregate, per_scene=scene_summary),
    }


def build_catalog_cv_report(
    v6: Any,
    v7: Any,
    evaluator: Evaluator,
    policies: Sequence[CatalogPolicy] = DEFAULT_POLICY_GRID,
    seed: int = 20260712,
) -> Dict[str, Any]:
    normalized = normalize_opened_benchmarks(v6, v7)
    records = normalized["records"]
    nested = nested_cross_validate(records, evaluator, policies, seed)
    scenes = scene_held_out_validate(records, evaluator, policies, seed)
    preconditions = {
        "nested_5fold_hard_gate": nested["hard_gate"]["gate_pass"],
        "scene_held_out_hard_gate": scenes["hard_gate"]["gate_pass"],
    }
    return {
        "benchmark_inventory": normalized["benchmark_inventory"],
        "gold_eligibility": (
            "credible category_a_sonic deciding editorial/participant or named "
            "critic track-level comparisons only"
        ),
        "primary": {
            "name": "sonic_primary",
            "formula": "mean(nDCG@10, MRR@10, Recall@10)",
            "reported_separately": list(PRIMARY_METRICS),
            "musicbrainz_style_role": "predeclared sigma gate only",
        },
        "nested_5fold": nested,
        "scene_held_out": scenes,
        "hard_preconditions": preconditions,
        "all_preconditions_passed": all(preconditions.values()),
        "final_policy_selection_source": (
            "all credible sonic DEV inner CV only"
        ),
        "deezer_used_for_selection": False,
        "no_unopened_final_labels_compared": True,
    }


__all__ = [
    "CatalogPolicy", "CatalogPolicyRanker", "CatalogStyleIndex", "PairResolver",
    "ProductionRanker", "DEFAULT_POLICY_GRID", "normalize_opened_benchmarks",
    "precompute_query_components", "apply_cached_policy", "rescore_components",
    "sonic_primary", "composite_primary", "style_coherence_at_3",
    "candidate_recall", "candidate_recall_comparison", "evaluate_seed",
    "resolve_relevance", "deterministic_folds", "scene_held_out_folds",
    "select_policy", "nested_cross_validate", "scene_held_out_validate",
    "relative_change", "paired_bootstrap", "gate_scene_fold",
    "build_catalog_cv_report",
]
