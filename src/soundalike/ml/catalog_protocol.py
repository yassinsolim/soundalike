"""Hash-bound one-open protocol for graded multi-positive retrieval.

FINAL labels are frozen before graph training.  Baseline and challenger lists
are generated target-blind, then the selected method, assets, DEV report, and
rankings are hash-locked before FINAL can be scored exactly once.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np

from .final_protocol import (
    ProtocolError,
    _now,
    _rank_audio_priors_zero,
    _serialise_ranking,
    _verify_state_signature,
    _write_json,
    content_sha256,
    file_sha256,
)
from .real_benchmark import (
    PairResolver,
    ProductionRanker,
    credited_artists,
    normalize_text,
)

BASELINE_METHODS = (
    "production_baseline",
    "iteration3_deployed",
    "audio_priors_zero",
)
_CANDIDATE_CUTOFFS = (100, 500, 1000)


def _verify_state(
    state: Mapping[str, Any],
    state_path: Path | None = None,
) -> None:
    expected = state.get("integrity_signature")
    unsigned = dict(state)
    unsigned.pop("integrity_signature", None)
    unsigned.pop("signature_algorithm", None)
    if not expected or content_sha256(unsigned) != expected:
        raise ProtocolError("Protocol state integrity signature mismatch")
    if state_path is None:
        return
    signature = state_path.parent / "state.sig"
    allowed = state_path.parent / "allowed_signers"
    if state.get("detached_signature_required") and not (
        signature.is_file() and allowed.is_file()
    ):
        raise ProtocolError("Required detached protocol signature is missing")
    if signature.exists() or allowed.exists():
        _verify_state_signature(state, state_path)


def _set_signature(state: Dict[str, Any]) -> None:
    state.pop("integrity_signature", None)
    state.pop("signature_algorithm", None)
    state["integrity_signature"] = content_sha256(state)
    state["signature_algorithm"] = "SHA-256 canonical JSON"


def _verify_file(path: Path, digest: str, label: str) -> None:
    if not path.is_file() or file_sha256(path) != digest:
        raise ProtocolError(f"Frozen {label} is missing or changed: {path}")


def _verify_inputs(state: Mapping[str, Any], locked: bool = False) -> None:
    for path_key, hash_key, label in (
        ("benchmark_path", "benchmark_sha256", "benchmark"),
        ("index_path", "index_sha256", "index"),
        ("manifest_path", "manifest_sha256", "manifest"),
        ("baseline_path", "baseline_sha256", "baseline"),
    ):
        _verify_file(Path(state[path_key]), state[hash_key], label)
    if not locked:
        return
    _verify_file(
        Path(state["method_manifest_path"]),
        state["method_manifest_sha256"],
        "method manifest",
    )
    _verify_file(
        Path(state["dev_report_path"]),
        state["dev_report_sha256"],
        "DEV report",
    )
    for raw_path, digest in state.get("asset_hashes", {}).items():
        _verify_file(Path(raw_path), digest, f"method asset {raw_path}")


def _record_artists(record: Mapping[str, Any]) -> Set[str]:
    result = {
        normalize_text(artist)
        for artist in credited_artists(record["query"]["artist"])
    }
    result |= {
        normalize_text(artist)
        for positive in record["positives"]
        for artist in credited_artists(positive["artist"])
    }
    return result


def validate_benchmark(benchmark: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate multi-positive scale, provenance, and split isolation."""
    records = benchmark.get("records", [])
    development = [
        record for record in records if record.get("split") == "development"
    ]
    final = [record for record in records if record.get("split") == "final"]
    if len(final) < 50:
        raise ProtocolError("FINAL must contain at least 50 seeds")
    if len({record["scene"] for record in final}) < 12:
        raise ProtocolError("FINAL must span at least 12 scenes")
    if not {"popular", "deep_cut", "niche"} <= {
        record["catalog_tier"] for record in final
    }:
        raise ProtocolError("FINAL must span popular, deep-cut, and niche tiers")
    ids: Set[str] = set()
    query_tracks: Set[Tuple[str, str]] = set()
    for record in records:
        if record["id"] in ids:
            raise ProtocolError(f"Duplicate benchmark id: {record['id']}")
        ids.add(record["id"])
        if not 5 <= len(record.get("positives", [])) <= 20:
            raise ProtocolError(f"{record['id']} must have 5-20 positives")
        source = record.get("source", {})
        for field in ("url", "publisher", "accessed_at", "source_class", "excerpt"):
            if not source.get(field):
                raise ProtocolError(f"{record['id']} source lacks {field}")
        if record.get("evidence_axis") != "taste_affinity":
            raise ProtocolError(f"{record['id']} axis is not explicit")
        query_artist = normalize_text(record["query"]["artist"])
        query_key = (
            normalize_text(record["query"]["title"]),
            query_artist,
        )
        if query_key in query_tracks:
            raise ProtocolError(f"Repeated benchmark query: {query_key}")
        query_tracks.add(query_key)
        record_tracks: Set[Tuple[str, str]] = set()
        for positive in record["positives"]:
            key = (
                normalize_text(positive["title"]),
                normalize_text(positive["artist"]),
            )
            if key in record_tracks:
                raise ProtocolError(
                    f"Repeated positive in {record['id']}: {key}"
                )
            record_tracks.add(key)
            if not 1 <= int(positive["grade"]) <= 3:
                raise ProtocolError(f"Invalid relevance grade in {record['id']}")
            if normalize_text(positive["artist"]) == query_artist:
                raise ProtocolError(f"Same-artist positive in {record['id']}")
    dev_artists = set().union(*map(_record_artists, development))
    final_artists = set().union(*map(_record_artists, final))
    overlap = sorted(dev_artists & final_artists)
    if overlap:
        raise ProtocolError(
            f"Development/FINAL artist component overlap: {overlap[:5]}"
        )
    return {
        "development_seeds": len(development),
        "final_seeds": len(final),
        "final_scenes": len({record["scene"] for record in final}),
        "final_positives": sum(len(record["positives"]) for record in final),
        "artist_overlap": overlap,
    }


def _final_records(benchmark: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [
        record
        for record in benchmark["records"]
        if record.get("split") == "final"
    ]


def freeze_protocol(
    benchmark_path: Path,
    index_path: Path,
    protocol_dir: Path,
) -> Dict[str, Any]:
    """Freeze labels and baseline lists before any v7 model selection."""
    state_path = protocol_dir / "state.json"
    if state_path.exists():
        raise ProtocolError("Protocol state already exists; freeze is immutable")
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    audit = validate_benchmark(benchmark)
    final = _final_records(benchmark)
    manifest = {
        "schema_version": 7,
        "benchmark_id": benchmark["benchmark_id"],
        "created_at": _now(),
        "records": final,
    }
    manifest["content_sha256"] = content_sha256(final)

    from webapp.api._reco import WebRecommender

    recommender = WebRecommender(str(index_path), enhance=False)
    resolver = PairResolver(recommender.titles, recommender.artists)
    production = ProductionRanker(recommender, heldout=set())
    frozen_records = []
    for record in final:
        query_row = resolver.query_row(record["query"])
        if query_row is None:
            raise ProtocolError(f"Missing FINAL query: {record['id']}")
        for positive in record["positives"]:
            if not resolver.target_rows(positive):
                raise ProtocolError(
                    f"Missing FINAL positive in {record['id']}"
                )
        method_rows = {
            "production_baseline": production.rank(
                query_row, "production_baseline", n=100
            ),
            "iteration3_deployed": production.rank(
                query_row, "dual_sonic", n=100
            ),
            "audio_priors_zero": _rank_audio_priors_zero(
                recommender, query_row, n=100
            ),
        }
        frozen_records.append(
            {
                "record_id": record["id"],
                "query": dict(record["query"]),
                "query_row": int(query_row),
                "rankings": {
                    method: _serialise_ranking(recommender, rows)
                    for method, rows in method_rows.items()
                },
            }
        )
    baseline = {
        "schema_version": 7,
        "created_at": _now(),
        "target_labels_compared": False,
        "methods": list(BASELINE_METHODS),
        "records": frozen_records,
        "content_sha256": content_sha256(frozen_records),
    }
    manifest_path = protocol_dir / "final-test-manifest.json"
    baseline_path = protocol_dir / "frozen-baseline-rankings.json"
    _write_json(manifest_path, manifest)
    _write_json(baseline_path, baseline)
    state = {
        "schema_version": 7,
        "status": "FROZEN",
        "created_at": _now(),
        "benchmark_path": str(benchmark_path),
        "benchmark_sha256": file_sha256(benchmark_path),
        "index_path": str(index_path),
        "index_sha256": file_sha256(index_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "baseline_path": str(baseline_path),
        "baseline_sha256": file_sha256(baseline_path),
        "final_open_count": 0,
        "audit": audit,
        "rankings_locked_before_open": False,
    }
    _set_signature(state)
    _write_json(state_path, state)
    return state


def lock_method(
    protocol_dir: Path,
    method_manifest_path: Path,
    dev_report_path: Path,
) -> Dict[str, Any]:
    state_path = protocol_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    _verify_state(state, state_path)
    _verify_inputs(state)
    if state["status"] != "FROZEN" or state["final_open_count"] != 0:
        raise ProtocolError("Method can be locked only from unopened FROZEN")
    method = json.loads(method_manifest_path.read_text(encoding="utf-8"))
    for field in ("method_id", "configuration", "assets"):
        if not method.get(field):
            raise ProtocolError(f"Method manifest lacks {field}")
    asset_hashes = {}
    for asset in method["assets"]:
        path = Path(asset["path"])
        digest = file_sha256(path)
        if asset.get("sha256") and asset["sha256"] != digest:
            raise ProtocolError(f"Asset hash mismatch: {path}")
        asset_hashes[str(path)] = digest
    state.update(
        {
            "status": "METHOD_LOCKED",
            "locked_at": _now(),
            "method_id": method["method_id"],
            "method_manifest_path": str(method_manifest_path),
            "method_manifest_sha256": file_sha256(method_manifest_path),
            "dev_report_path": str(dev_report_path),
            "dev_report_sha256": file_sha256(dev_report_path),
            "asset_hashes": asset_hashes,
        }
    )
    _set_signature(state)
    _write_json(state_path, state)
    return state


def commit_rankings(
    protocol_dir: Path,
    rankings_path: Path,
) -> Dict[str, Any]:
    state_path = protocol_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    _verify_state(state, state_path)
    _verify_inputs(state, locked=True)
    if state["status"] != "METHOD_LOCKED" or state["final_open_count"] != 0:
        raise ProtocolError("Rankings require unopened METHOD_LOCKED state")
    rankings = json.loads(rankings_path.read_text(encoding="utf-8"))
    if rankings.get("target_labels_compared") is not False:
        raise ProtocolError("Rankings must be target-blind")
    if rankings.get("method_manifest_sha256") != state["method_manifest_sha256"]:
        raise ProtocolError("Rankings do not match locked method")
    manifest = json.loads(Path(state["manifest_path"]).read_text(encoding="utf-8"))
    expected = {record["id"] for record in manifest["records"]}
    actual = {record["record_id"] for record in rankings["records"]}
    if expected != actual:
        raise ProtocolError("Rankings and FINAL manifest differ")
    state.update(
        {
            "status": "RANKINGS_LOCKED",
            "rankings_locked_at": _now(),
            "rankings_locked_before_open": True,
            "winner_rankings_path": str(rankings_path),
            "winner_rankings_sha256": file_sha256(rankings_path),
            "winner_rankings_content_sha256": content_sha256(
                rankings["records"]
            ),
        }
    )
    _set_signature(state)
    _write_json(state_path, state)
    return state


def _graded_rows(
    resolver: PairResolver,
    record: Mapping[str, Any],
) -> Dict[int, Tuple[str, int]]:
    """Resolve rows to relevance-group ids so artist labels count only once."""
    if not hasattr(resolver, "_catalog_protocol_artist_rows"):
        artist_rows: Dict[str, List[int]] = defaultdict(list)
        for row, raw_artist in enumerate(resolver.artists):
            for artist in credited_artists(str(raw_artist)):
                artist_rows[normalize_text(artist)].append(row)
        setattr(resolver, "_catalog_protocol_artist_rows", artist_rows)
    artist_rows = getattr(resolver, "_catalog_protocol_artist_rows")
    relevance: Dict[int, Tuple[str, int]] = {}
    for number, positive in enumerate(record["positives"]):
        grade = int(positive["grade"])
        if positive.get("relevance_scope") == "artist":
            artist_keys = {
                normalize_text(artist)
                for artist in credited_artists(positive["artist"])
            }
            rows = {
                int(row)
                for artist in artist_keys
                for row in artist_rows.get(artist, [])
            }
            group = "artist:" + "|".join(sorted(artist_keys))
        else:
            rows = set(map(int, resolver.target_rows(positive)))
            group = (
                f"track:{normalize_text(positive['title'])}:"
                f"{normalize_text(positive['artist'])}:{number}"
            )
        for row in rows:
            previous = relevance.get(row)
            if previous is None or grade > previous[1]:
                relevance[row] = (group, grade)
    return relevance


def _per_seed(
    ranking: Sequence[Mapping[str, Any]],
    relevance: Mapping[int, Tuple[str, int]],
) -> Dict[str, float]:
    ranked_rows = [int(item["row"]) for item in ranking[:10]]
    group_grades: Dict[str, int] = {}
    for group, grade in relevance.values():
        group_grades[group] = max(group_grades.get(group, 0), grade)
    found_groups: Set[str] = set()
    gains = []
    relevant_ranks = []
    for rank, row in enumerate(ranked_rows, 1):
        value = relevance.get(row)
        if value is None or value[0] in found_groups:
            gains.append(0)
            continue
        group, grade = value
        found_groups.add(group)
        gains.append(2 ** grade - 1)
        relevant_ranks.append(rank)
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, 1))
    ideal = sorted(
        (2 ** grade - 1 for grade in group_grades.values()), reverse=True
    )[:10]
    idcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(ideal, 1))
    return {
        "ndcg_at_10": dcg / idcg if idcg else 0.0,
        "mrr_at_10": 1.0 / relevant_ranks[0] if relevant_ranks else 0.0,
        "recall_at_10": len(found_groups) / max(len(group_grades), 1),
    }


def _aggregate(values: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    return {
        metric: float(np.mean([value[metric] for value in values]))
        for metric in ("ndcg_at_10", "mrr_at_10", "recall_at_10")
    } | {"primary": float(np.mean([value["ndcg_at_10"] for value in values]))}


def _bootstrap(
    baseline: Sequence[float],
    winner: Sequence[float],
    iterations: int = 20_000,
) -> Dict[str, float | None]:
    base = np.asarray(baseline, dtype=np.float64)
    test = np.asarray(winner, dtype=np.float64)
    rng = np.random.default_rng(20260712)
    deltas = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        sample = rng.integers(0, len(base), len(base))
        deltas[iteration] = float(np.mean(test[sample] - base[sample]))
    absolute = float(np.mean(test - base))
    baseline_mean = float(np.mean(base))
    return {
        "baseline_primary": baseline_mean,
        "winner_primary": float(np.mean(test)),
        "absolute_delta": absolute,
        "relative_gain": (
            absolute / baseline_mean if baseline_mean > 0 else None
        ),
        "ci95_low": float(np.percentile(deltas, 2.5)),
        "ci95_high": float(np.percentile(deltas, 97.5)),
        "probability_positive": float(np.mean(deltas > 0)),
    }


def _candidate_recall(
    candidate_rows: Sequence[int],
    relevance: Mapping[int, Tuple[str, int]],
    cutoff: int,
) -> float:
    all_groups = {group for group, _ in relevance.values()}
    found = {
        relevance[int(row)][0]
        for row in candidate_rows[:cutoff]
        if int(row) in relevance
    }
    return len(found) / max(len(all_groups), 1)


def open_final_once(
    protocol_dir: Path,
    rankings_path: Path,
    report_path: Path,
) -> Dict[str, Any]:
    """Open FINAL once and atomically compute every predeclared gate."""
    state_path = protocol_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    _verify_state(state, state_path)
    if state["status"] != "RANKINGS_LOCKED" or state["final_open_count"] != 0:
        raise ProtocolError("FINAL requires unopened RANKINGS_LOCKED state")
    _verify_inputs(state, locked=True)
    _verify_file(
        rankings_path, state["winner_rankings_sha256"], "winner rankings"
    )
    rankings_doc = json.loads(rankings_path.read_text(encoding="utf-8"))
    if content_sha256(rankings_doc["records"]) != state[
        "winner_rankings_content_sha256"
    ]:
        raise ProtocolError("Winner rankings content hash mismatch")
    manifest = json.loads(Path(state["manifest_path"]).read_text(encoding="utf-8"))
    baseline_doc = json.loads(Path(state["baseline_path"]).read_text(encoding="utf-8"))
    manifest_by_id = {record["id"]: record for record in manifest["records"]}
    baseline_by_id = {
        record["record_id"]: record for record in baseline_doc["records"]
    }
    winner_by_id = {
        record["record_id"]: record for record in rankings_doc["records"]
    }
    if set(manifest_by_id) != set(baseline_by_id) or set(manifest_by_id) != set(
        winner_by_id
    ):
        raise ProtocolError("FINAL record identities differ")

    with np.load(state["index_path"], allow_pickle=False) as index:
        resolver = PairResolver(index["titles"], index["artists"])
    method_names = sorted(
        {
            method
            for record in winner_by_id.values()
            for method in record.get("diagnostic_rankings", {})
        }
    )
    metric_names = list(BASELINE_METHODS) + ["winner"] + method_names
    per_method: Dict[str, List[Dict[str, float]]] = {
        method: [] for method in metric_names
    }
    candidate_methods = sorted(
        {
            method
            for record in winner_by_id.values()
            for method in record.get("candidate_sets", {})
        }
    )
    candidate_values = {
        method: {cutoff: [] for cutoff in _CANDIDATE_CUTOFFS}
        for method in candidate_methods
    }
    rows = []
    for record_id, record in manifest_by_id.items():
        grades = _graded_rows(resolver, record)
        baseline_record = baseline_by_id[record_id]
        winner_record = winner_by_id[record_id]
        rankings = {
            method: baseline_record["rankings"][method]
            for method in BASELINE_METHODS
        }
        rankings["winner"] = winner_record["ranking"]
        rankings.update(winner_record.get("diagnostic_rankings", {}))
        seed_metrics = {
            method: _per_seed(ranking, grades)
            for method, ranking in rankings.items()
        }
        for method, values in seed_metrics.items():
            per_method[method].append(values)
        seed_candidate = {}
        for method in candidate_methods:
            candidate_rows = winner_record["candidate_sets"].get(method, [])
            seed_candidate[method] = {}
            for cutoff in _CANDIDATE_CUTOFFS:
                recall = _candidate_recall(candidate_rows, grades, cutoff)
                candidate_values[method][cutoff].append(recall)
                seed_candidate[method][f"recall_at_{cutoff}"] = recall
        rows.append(
            {
                "record_id": record_id,
                "scene": record["scene"],
                "catalog_tier": record["catalog_tier"],
                "positive_count": len({group for group, _ in grades.values()}),
                "metrics": seed_metrics,
                "candidate_recall": seed_candidate,
            }
        )
    metrics = {
        method: _aggregate(values) for method, values in per_method.items()
    }
    candidate_recall = {
        method: {
            f"recall_at_{cutoff}": float(np.mean(values))
            for cutoff, values in cutoffs.items()
        }
        for method, cutoffs in candidate_values.items()
    }
    baseline_primary = [
        value["ndcg_at_10"] for value in per_method["production_baseline"]
    ]
    winner_primary = [
        value["ndcg_at_10"] for value in per_method["winner"]
    ]
    comparison = _bootstrap(baseline_primary, winner_primary)
    comparison.update(
        {
            "improved_seeds": int(
                np.sum(np.asarray(winner_primary) > np.asarray(baseline_primary))
            ),
            "worsened_seeds": int(
                np.sum(np.asarray(winner_primary) < np.asarray(baseline_primary))
            ),
            "unchanged_seeds": int(
                np.sum(np.asarray(winner_primary) == np.asarray(baseline_primary))
            ),
        }
    )
    scenes: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: {"baseline": [], "winner": []}
    )
    for row in rows:
        scenes[row["scene"]]["baseline"].append(
            row["metrics"]["production_baseline"]["ndcg_at_10"]
        )
        scenes[row["scene"]]["winner"].append(
            row["metrics"]["winner"]["ndcg_at_10"]
        )
    per_scene = {}
    for scene, values in sorted(scenes.items()):
        base = float(np.mean(values["baseline"]))
        winner = float(np.mean(values["winner"]))
        relative = (winner - base) / base if base > 0 else (0.0 if winner == 0 else 1.0)
        per_scene[scene] = {
            "baseline_primary": base,
            "winner_primary": winner,
            "relative_delta": relative,
        }
    comparison["per_scene"] = per_scene
    policy = json.loads(Path(state["benchmark_path"]).read_text(encoding="utf-8"))[
        "metric_policy"
    ]["success"]
    comparison["passes"] = {
        "relative_gain": comparison["relative_gain"] is not None
        and comparison["relative_gain"] >= policy["minimum_relative_primary_gain"],
        "minimum_absolute_gain": comparison["absolute_delta"]
        >= policy["minimum_absolute_primary_gain"],
        "ci_excludes_zero": comparison["ci95_low"]
        > policy["paired_bootstrap_ci95_low_must_exceed"],
        "minimum_improved_seeds": comparison["improved_seeds"]
        >= policy["minimum_improved_seeds"],
        "recall_at_10_non_regression": metrics["winner"]["recall_at_10"]
        >= metrics["production_baseline"]["recall_at_10"],
        "mrr_at_10_non_regression": metrics["winner"]["mrr_at_10"]
        >= metrics["production_baseline"]["mrr_at_10"],
        "scene_no_regression": all(
            value["relative_delta"]
            >= policy["maximum_scene_relative_regression"]
            for value in per_scene.values()
        ),
    }
    comparison["retrieval_pass"] = all(comparison["passes"].values())
    report = {
        "schema_version": 7,
        "opened_at": _now(),
        "open_number": 1,
        "method_id": state["method_id"],
        "primary_axis": "taste_affinity",
        "metrics": metrics,
        "candidate_recall": candidate_recall,
        "comparison_to_production_baseline": comparison,
        "records": rows,
    }
    _write_json(report_path, report)
    state.update(
        {
            "status": "FINALIZED",
            "final_open_count": 1,
            "final_opened_at": report["opened_at"],
            "final_report_path": str(report_path),
            "final_report_sha256": file_sha256(report_path),
            "retrieval_pass": comparison["retrieval_pass"],
            "detached_signature_required": True,
        }
    )
    _set_signature(state)
    _write_json(state_path, state)
    return report


def attach_direct_gate(
    protocol_dir: Path,
    direct_report_path: Path,
) -> Dict[str, Any]:
    """Bind the separately locked 20-seed coherence gate to final ship status."""
    state_path = protocol_dir / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    _verify_state(state, state_path)
    if state.get("status") != "FINALIZED" or state.get("final_open_count") != 1:
        raise ProtocolError("Direct gate requires exactly one finalized opening")
    if state.get("direct_report_path"):
        raise ProtocolError("Direct gate has already been attached")
    direct = json.loads(direct_report_path.read_text(encoding="utf-8"))
    summary = direct.get("summary", {})
    benchmark = json.loads(Path(state["benchmark_path"]).read_text(encoding="utf-8"))
    required = benchmark["metric_policy"]["success"][
        "minimum_direct_top5_passes"
    ]
    if not direct.get("method_locked_before_judgment"):
        raise ProtocolError("Direct judgments were not method-locked")
    if int(summary.get("total", 0)) != 20:
        raise ProtocolError("Direct judgment report must contain 20 seeds")
    direct_pass = int(summary.get("passes", 0)) >= int(required)
    state.update(
        {
            "direct_report_path": str(direct_report_path),
            "direct_report_sha256": file_sha256(direct_report_path),
            "direct_top5_passes": int(summary.get("passes", 0)),
            "direct_top5_required": int(required),
            "direct_pass": direct_pass,
            "final_pass": bool(state.get("retrieval_pass")) and direct_pass,
            "detached_signature_required": True,
        }
    )
    _set_signature(state)
    _write_json(state_path, state)
    return state


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--benchmark", type=Path, required=True)
    freeze.add_argument("--index", type=Path, required=True)
    freeze.add_argument("--protocol-dir", type=Path, required=True)
    lock = commands.add_parser("lock")
    lock.add_argument("--protocol-dir", type=Path, required=True)
    lock.add_argument("--method-manifest", type=Path, required=True)
    lock.add_argument("--dev-report", type=Path, required=True)
    rankings = commands.add_parser("commit-rankings")
    rankings.add_argument("--protocol-dir", type=Path, required=True)
    rankings.add_argument("--rankings", type=Path, required=True)
    final = commands.add_parser("open-final")
    final.add_argument("--protocol-dir", type=Path, required=True)
    final.add_argument("--rankings", type=Path, required=True)
    final.add_argument("--report", type=Path, required=True)
    direct = commands.add_parser("attach-direct")
    direct.add_argument("--protocol-dir", type=Path, required=True)
    direct.add_argument("--direct-report", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        result = freeze_protocol(args.benchmark, args.index, args.protocol_dir)
    elif args.command == "lock":
        result = lock_method(
            args.protocol_dir, args.method_manifest, args.dev_report
        )
    elif args.command == "commit-rankings":
        result = commit_rankings(args.protocol_dir, args.rankings)
    elif args.command == "open-final":
        result = open_final_once(args.protocol_dir, args.rankings, args.report)
    else:
        result = attach_direct_gate(args.protocol_dir, args.direct_report)
    summary = {
        key: result[key]
        for key in (
            "status",
            "schema_version",
            "open_number",
            "retrieval_pass",
            "direct_pass",
            "final_pass",
        )
        if key in result
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
