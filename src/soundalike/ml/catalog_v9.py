"""Freeze and execute the powered served-list DEVELOPMENT protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from .catalog_cv_v9 import (
    ListGoldScorer,
    nested_cross_validate,
    scene_held_out_validate,
    summarize_predictions,
)
from .catalog_graph import CatalogArtistGraph
from .catalog_list_gold_v9 import (
    canonical_bytes,
    sha256_bytes,
    sha256_path,
    validate_gold,
    write_json,
)
from .catalog_policy_v9 import (
    DEFAULT_LIST_POLICY_GRID,
    LastfmListRanker,
    ListPolicy,
)
from .catalog_style import CatalogStyleIndex
from .real_benchmark import ProductionRanker


class PoweredDevelopmentError(RuntimeError):
    """Raised when the locked DEVELOPMENT protocol cannot be honored."""


def _load_json(path: Any) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_recommender(index_path: Any) -> Any:
    from webapp.api._reco import WebRecommender

    return WebRecommender(str(index_path))


def _sign_state(directory: Path, state_path: Path) -> Dict[str, Any]:
    executable = shutil.which("ssh-keygen")
    if executable is None:
        raise PoweredDevelopmentError("ssh-keygen is required; freeze fails closed")
    with tempfile.TemporaryDirectory(prefix="soundalike-v9-key-") as temporary:
        private = Path(temporary) / "signer"
        generated = subprocess.run(
            [
                executable, "-q", "-t", "ed25519", "-N", "",
                "-C", "soundalike-protocol-v9-development", "-f", str(private),
            ],
            capture_output=True,
            check=False,
        )
        if generated.returncode:
            raise PoweredDevelopmentError("Ed25519 key generation failed")
        public = private.with_suffix(".pub").read_text(encoding="utf-8").strip()
        fields = public.split()
        (directory / "signer.pub").write_text(public + "\n", encoding="utf-8")
        (directory / "allowed_signers").write_text(
            f"soundalike-protocol {fields[0]} {fields[1]}\n", encoding="utf-8"
        )
        signed = subprocess.run(
            [
                executable, "-Y", "sign", "-f", str(private),
                "-n", "soundalike-protocol", str(state_path),
            ],
            capture_output=True,
            check=False,
        )
        generated_signature = Path(str(state_path) + ".sig")
        if signed.returncode or not generated_signature.is_file():
            raise PoweredDevelopmentError("detached state signing failed")
        os.replace(generated_signature, directory / "state.sig")
    return {
        "algorithm": "Ed25519 detached SSH signature",
        "namespace": "soundalike-protocol",
        "identity": "soundalike-protocol",
        "state_sha256": sha256_path(state_path),
        "public_key_sha256": sha256_path(directory / "signer.pub"),
        "allowed_signers_sha256": sha256_path(directory / "allowed_signers"),
        "signature_sha256": sha256_path(directory / "state.sig"),
    }


def freeze_development_protocol(
    protocol_dir: Any,
    *,
    gold_path: Any,
    snapshots_path: Any,
    index_path: Any,
    graph_path: Any,
    style_path: Any,
    policies: Sequence[ListPolicy] = DEFAULT_LIST_POLICY_GRID,
) -> Dict[str, Any]:
    """Create the signed scorer/policy lock before any policy evaluation."""
    directory = Path(protocol_dir)
    if directory.exists():
        raise PoweredDevelopmentError("development protocol directory must be new")
    gold = _load_json(gold_path)
    validation = validate_gold(gold)
    if not validation["passed"]:
        raise PoweredDevelopmentError("powered gold validation failed")
    code_paths = (
        Path("src/soundalike/ml/catalog_list_gold_v9.py"),
        Path("src/soundalike/ml/catalog_policy_v9.py"),
        Path("src/soundalike/ml/catalog_cv_v9.py"),
        Path("src/soundalike/ml/catalog_v9.py"),
    )
    inputs = {
        str(Path(gold_path)): sha256_path(gold_path),
        str(Path(snapshots_path)): sha256_path(snapshots_path),
        str(Path(index_path)): sha256_path(index_path),
        str(Path(graph_path)): sha256_path(graph_path),
        str(Path(style_path)): sha256_path(style_path),
        **{str(path): sha256_path(path) for path in code_paths},
    }
    protocol = {
        "schema_version": 9,
        "kind": "powered-served-list-development-protocol",
        "phase": "DEVELOPMENT_SCORER_LOCKED",
        "locked_before_policy_tuning": True,
        "powered_gold": {
            "seeds": validation["seeds"],
            "scenes": validation["scenes"],
            "positives": validation["positives"],
            "source_classes": [
                "Gnod Music-Map crowd similar-artist maps",
                "category-A named critic/editorial track comparisons",
            ],
            "candidate_graph_independent": True,
            "deezer_listenbrainz_musicbrainz": "supporting_only",
        },
        "co_primary": {
            "one": {
                "name": "graded_nDCG@10",
                "unit": "actual current-production or challenger served top-10",
                "gain": "2^grade-1; one credit per frozen positive entity",
                "minimum_relative_gain": 0.20,
                "minimum_absolute_gain": 0.02,
                "ci95_low_must_exceed": 0.0,
                "minimum_improved_seeds": 10,
                "maximum_scene_relative_regression": -0.10,
            },
            "two": {
                "name": "source-grounded top5 coherence pass rate",
                "pass_rule": (
                    "all positions 1-3 independently supported, at least 4/5 "
                    "supported, and zero junk/same-artist variants"
                ),
                "minimum_challenger": 0.80,
                "minimum_absolute_margin_over_production": 0.10,
            },
            "selection_rule": (
                "select by inner-fold nDCG; coherence remains a separate hard "
                "co-primary and is never blended into nDCG"
            ),
            "exact_pair_retrieval": "diagnostic_only",
        },
        "blind_judgment": {
            "method_aliases": "deterministic per-seed A/B aliases",
            "scoring": "method-blind deterministic frozen source evidence",
            "unsupported_subjective_assertions": False,
            "uncertainty": "low/medium/high recorded per result",
            "preview_availability": "public Deezer track endpoint; supporting only",
        },
        "policy": {
            "numeric_parameters": ["tau", "sigma", "audio_weight"],
            "numeric_parameter_count": 3,
            "fixed_grid": [asdict(policy) for policy in policies],
            "tau": "Last.fm neighborhood confidence threshold",
            "sigma": "per-track min(audio similarity, style consistency) threshold",
            "audio_weight": "single graph-ranking audio tie-break",
            "music4all": (
                "optional fixed 0.15 score corroborator and fixed 0.05 confidence "
                "bonus where shared; never required for coverage"
            ),
            "production_default": "exact current dual_sonic list on abstention",
            "scene_artist_popularity_boosts": False,
        },
        "validation": {
            "nested_5fold": True,
            "scene_held_out": True,
            "candidate_recall_at_1000_must_improve": True,
            "mrr_must_not_regress": True,
            "no_junk": True,
        },
        "final_and_deployment": {
            "fresh_final_creation": "BLOCKED until every DEV and tier gate passes",
            "final_open_count": 0,
            "deployment": "BLOCKED until fresh FINAL passes",
        },
        "input_sha256": inputs,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    directory.mkdir(parents=True, exist_ok=False)
    protocol_path = directory / "development-protocol.json"
    write_json(protocol_path, protocol)
    state = {
        "schema_version": 9,
        "phase": "DEVELOPMENT_SCORER_LOCKED",
        "policy_tuning_started": False,
        "final_open_count": 0,
        "fresh_final_blocked": True,
        "deployment_blocked": True,
        "input_sha256": inputs,
        "protocol_sha256": sha256_path(protocol_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    state["integrity_signature"] = sha256_bytes(canonical_bytes(state))
    state_path = directory / "state.json"
    write_json(state_path, state)
    metadata = _sign_state(directory, state_path)
    write_json(directory / "signature-metadata.json", metadata)
    return {"protocol": protocol, "state": state, "signature": metadata}


def _verify_protocol_inputs(protocol_dir: Any) -> Dict[str, Any]:
    directory = Path(protocol_dir)
    state_path = directory / "state.json"
    protocol_path = directory / "development-protocol.json"
    state = _load_json(state_path)
    protocol = _load_json(protocol_path)
    if state.get("phase") != "DEVELOPMENT_SCORER_LOCKED":
        raise PoweredDevelopmentError("locked state has the wrong phase")
    if state.get("policy_tuning_started") or int(state.get("final_open_count", -1)) != 0:
        raise PoweredDevelopmentError("locked state is not untouched DEVELOPMENT")
    unsigned = dict(state)
    declared_integrity = str(unsigned.pop("integrity_signature", ""))
    if sha256_bytes(canonical_bytes(unsigned)) != declared_integrity:
        raise PoweredDevelopmentError("state canonical integrity signature is invalid")
    if str(state.get("protocol_sha256")) != sha256_path(protocol_path):
        raise PoweredDevelopmentError("locked protocol hash mismatch")
    if state.get("input_sha256") != protocol.get("input_sha256"):
        raise PoweredDevelopmentError("state and protocol input locks disagree")
    metadata = _load_json(directory / "signature-metadata.json")
    expected_files = {
        "state_sha256": state_path,
        "public_key_sha256": directory / "signer.pub",
        "allowed_signers_sha256": directory / "allowed_signers",
        "signature_sha256": directory / "state.sig",
    }
    for field, path in expected_files.items():
        if str(metadata.get(field)) != sha256_path(path):
            raise PoweredDevelopmentError(f"detached signature metadata mismatch: {field}")
    executable = shutil.which("ssh-keygen")
    if executable is None:
        raise PoweredDevelopmentError("ssh-keygen is required to verify the DEV lock")
    verified = subprocess.run(
        [
            executable, "-Y", "verify", "-f", str(directory / "allowed_signers"),
            "-I", "soundalike-protocol", "-n", "soundalike-protocol",
            "-s", str(directory / "state.sig"),
        ],
        input=state_path.read_bytes(),
        capture_output=True,
        check=False,
    )
    if verified.returncode != 0:
        raise PoweredDevelopmentError("detached DEV state signature did not verify")
    mismatches = {}
    for raw_path, expected in state["input_sha256"].items():
        path = Path(raw_path)
        actual = sha256_path(path) if path.is_file() else None
        if actual != expected:
            mismatches[raw_path] = {"expected": expected, "actual": actual}
    if mismatches:
        raise PoweredDevelopmentError(f"locked input hash mismatch: {mismatches}")
    return {"state": state, "protocol": protocol}


def _serialize_prediction_list(
    scored: Mapping[str, Any],
    component_by_row: Mapping[int, Mapping[str, Any]],
    *,
    method_role: str,
) -> List[Dict[str, Any]]:
    results = []
    for item in scored["result_evidence"]:
        row = int(item["row"])
        component = dict(component_by_row.get(row, {}))
        results.append({
            **dict(item),
            "method_role": method_role,
            "ranking_rationale": {
                "source": (
                    "lastfm_graph_optional_music4all"
                    if component else "actual_current_production_order"
                ),
                "G": float(component.get("G", 0.0)),
                "A": float(component.get("A", 0.0)),
                "S": float(component.get("S", 0.0)),
                "song_consistency":
                    float(component.get("song_consistency", 0.0)),
                "lastfm_G": float(component.get("lastfm_G", 0.0)),
                "music4all_G": float(component.get("music4all_G", 0.0)),
                "policy_score": float(component.get("score", 0.0)),
            },
        })
    return results


def _preview_status(track_id: Any) -> Dict[str, Any]:
    import requests

    try:
        response = requests.get(
            f"https://api.deezer.com/track/{int(track_id)}",
            timeout=15,
            headers={"User-Agent": "soundalike-evaluation/9.0"},
        )
        payload = response.json() if response.status_code == 200 else {}
        return {
            "status": "available" if bool(payload.get("preview")) else "unavailable",
            "http_status": int(response.status_code),
            "checked_at": "2026-07-12",
            "supporting_only": True,
        }
    except Exception as exc:  # network errors are evidence, never silent availability
        return {
            "status": "unknown_api_error",
            "http_status": None,
            "checked_at": "2026-07-12",
            "supporting_only": True,
            "error_class": type(exc).__name__,
        }


def _attach_previews(predictions: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    ids = {
        int(item["track_id"])
        for prediction in predictions
        for role in ("production_baseline", "challenger")
        for item in prediction["lists"][role]
    }
    statuses: Dict[int, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = {pool.submit(_preview_status, value): value for value in ids}
        for future in as_completed(futures):
            statuses[futures[future]] = future.result()
    for prediction in predictions:
        for role in ("production_baseline", "challenger"):
            for item in prediction["lists"][role]:
                item["preview_availability"] = statuses[int(item["track_id"])]
    return {
        "unique_tracks_checked": len(ids),
        "counts": dict(sorted(Counter(
            item["status"] for item in statuses.values()
        ).items())),
        "source": "public Deezer track endpoint",
        "used_for_selection": False,
    }


def _blind_artifacts(
    predictions: Sequence[Mapping[str, Any]]
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    blinded, judgments, key = [], [], []
    for prediction in predictions:
        record_id = str(prediction["id"])
        swap = int(hashlib.sha256(record_id.encode()).hexdigest(), 16) % 2 == 1
        roles = (
            ("A", "challenger"), ("B", "production_baseline")
        ) if swap else (
            ("A", "production_baseline"), ("B", "challenger")
        )
        lists = []
        for alias, role in roles:
            values = [
                {
                    key_name: item[key_name]
                    for key_name in (
                        "position", "track_id", "title", "artist", "grade",
                        "coherence_supported", "coherence_rationale",
                        "uncertainty", "junk", "same_artist",
                        "preview_availability",
                    )
                }
                for item in prediction["lists"][role]
            ]
            lists.append({"alias": alias, "results": values})
            score = prediction[
                "challenger" if role == "challenger" else "baseline"
            ]
            judgments.append({
                "id": record_id,
                "alias": alias,
                "coherence_pass": bool(score["coherence_pass"]),
                "coherence_fraction_at_5":
                    float(score["coherence_fraction_at_5"]),
                "unrelated_positions_1_to_3":
                    int(score["unrelated_positions_1_to_3"]),
                "junk_count": int(score["junk_count"]),
                "judgment_basis": (
                    "frozen independent Music-Map/category-A evidence only; "
                    "method identity hidden"
                ),
            })
            key.append({"id": record_id, "alias": alias, "method_role": role})
        blinded.append({
            "id": record_id,
            "query": dict(prediction["query"]),
            "scene": str(prediction["scene"]),
            "lists": lists,
        })
    blind_doc = {
        "schema_version": 9,
        "protocol": "deterministic method-blind source-grounded scoring",
        "records": blinded,
    }
    blind_doc["content_sha256"] = sha256_bytes(canonical_bytes(blind_doc))
    judgment_doc = {
        "schema_version": 9,
        "blind_lists_sha256": blind_doc["content_sha256"],
        "records": judgments,
    }
    judgment_doc["content_sha256"] = sha256_bytes(canonical_bytes(judgment_doc))
    key_doc = {
        "schema_version": 9,
        "blind_lists_sha256": blind_doc["content_sha256"],
        "judgments_sha256": judgment_doc["content_sha256"],
        "revealed_after_judgment_hash_locked": True,
        "records": key,
    }
    key_doc["content_sha256"] = sha256_bytes(canonical_bytes(key_doc))
    return blind_doc, judgment_doc, key_doc


def run_powered_development(
    *,
    protocol_dir: Any,
    gold_path: Any,
    snapshots_path: Any,
    index_path: Any,
    graph_path: Any,
    style_path: Any,
    report_path: Any,
    blind_lists_path: Any,
    judgments_path: Any,
    blind_key_path: Any,
    policies: Sequence[ListPolicy] = DEFAULT_LIST_POLICY_GRID,
) -> Dict[str, Any]:
    started = time.perf_counter()
    locked = _verify_protocol_inputs(protocol_dir)
    state, protocol = locked["state"], locked["protocol"]
    frozen_grid = list(protocol["policy"]["fixed_grid"])
    requested_grid = [asdict(policy) for policy in policies]
    if requested_grid != frozen_grid:
        raise PoweredDevelopmentError(
            "evaluation policy grid differs from the signed protocol"
        )
    gold, snapshots = _load_json(gold_path), _load_json(snapshots_path)
    records = list(gold["records"])
    rec = _load_recommender(index_path)
    graph = CatalogArtistGraph(graph_path)
    styles = CatalogStyleIndex(style_path)
    production = ProductionRanker(rec, set())
    ranker = LastfmListRanker(rec, graph, styles)
    scorer = ListGoldScorer(
        gold, snapshots, rec.titles, rec.artists, rec.track_ids
    )
    record_data: Dict[str, Dict[str, Any]] = {}
    for record in records:
        query_row = int(record["query"]["catalog_row"])
        if int(rec.track_ids[query_row]) != int(record["query"]["track_id"]):
            raise PoweredDevelopmentError("gold query row no longer matches index")
        production_rows = [
            int(row) for row in production.rank(
                query_row, "dual_sonic", n=1000
            )
        ]
        if len(production_rows) < 10:
            raise PoweredDevelopmentError("production returned fewer than ten rows")
        cached = ranker.precompute_list_query(
            query_row, production_rows=production_rows
        )
        baseline = scorer.score(record, production_rows[:10])
        baseline["candidate_recall_at_1000"] = scorer._candidate_recall(
            record, production_rows
        )
        record_data[str(record["id"])] = {
            "record": record,
            "cached": cached,
            "baseline": baseline,
            "production_rows": production_rows,
        }
    policy_cache: Dict[tuple[tuple[float, float, float], str], Dict[str, Any]] = {}

    def policy_result(policy: ListPolicy, record_id: str) -> Dict[str, Any]:
        key = ((policy.tau, policy.sigma, policy.audio_weight), record_id)
        if key not in policy_cache:
            data = record_data[record_id]
            applied = ranker.apply_precomputed_list_policy(
                data["cached"], policy, 10
            )
            score = scorer.score(data["record"], applied["ranking_rows"])
            score["candidate_recall_at_1000"] = scorer._candidate_recall(
                data["record"], applied["candidate_rows"][:1000]
            )
            policy_cache[key] = {"applied": applied, "score": score}
        return policy_cache[key]

    def evaluator(
        policy: ListPolicy, subset: Sequence[Mapping[str, Any]]
    ) -> Sequence[Mapping[str, Any]]:
        predictions = []
        for record in subset:
            record_id = str(record["id"])
            data = record_data[record_id]
            result = policy_result(policy, record_id)
            applied, challenger = result["applied"], result["score"]
            graph_head = set(
                applied["ranking_rows"][:5] if applied["fired"] else []
            )
            component_by_row = {
                int(item["row"]): item
                for item in applied["ranked_components"]
                if int(item["row"]) in graph_head
            }
            predictions.append({
                "id": record_id,
                "scene": str(record["scene"]),
                "query": dict(record["query"]),
                "baseline": data["baseline"],
                "challenger": challenger,
                "gate": {
                    key_name: value for key_name, value in applied.items()
                    if key_name not in {
                        "ranking_rows", "candidate_rows", "ranked_components"
                    }
                },
                "lists": {
                    "production_baseline": _serialize_prediction_list(
                        data["baseline"], {}, method_role="production_baseline"
                    ),
                    "challenger": _serialize_prediction_list(
                        challenger,
                        component_by_row,
                        method_role="challenger",
                    ),
                },
            })
        return predictions

    nested = nested_cross_validate(records, policies, evaluator)
    scene_held_out = scene_held_out_validate(records, policies, evaluator)
    selected = ListPolicy(**nested["final_policy"])
    selected_predictions = list(evaluator(selected, records))
    selected_summary = summarize_predictions(selected_predictions)
    preview_summary = _attach_previews(selected_predictions)
    blind_doc, judgment_doc, key_doc = _blind_artifacts(selected_predictions)
    write_json(blind_lists_path, blind_doc)
    write_json(judgments_path, judgment_doc)
    write_json(blind_key_path, key_doc)
    exact_pair = {}
    for role, metric_key in (
        ("production_baseline", "baseline"),
        ("challenger", "challenger"),
    ):
        eligible = [
            prediction for prediction in selected_predictions
            if any(
                item["relevance_scope"] == "track"
                for item in record_data[str(prediction["id"])]
                ["record"]["positives"]
            )
        ]
        exact_pair[role] = {
            "eligible_seeds": len(eligible),
            "top10_hits": sum(
                any(
                    item["matched_relevance_scope"] == "track"
                    and int(item["grade"]) > 0
                    for item in prediction["lists"][role]
                )
                for prediction in eligible
            ),
            "diagnostic_only": True,
        }
    quality_pass = bool(
        nested["aggregate_outer_predictions"]["gate_pass"]
        and scene_held_out["aggregate_predictions"]["gate_pass"]
        and selected_summary["gate_pass"]
    )
    gate_reasons = Counter(
        str(item["gate"]["reason"]) for item in selected_predictions
    )
    report = {
        "schema_version": 9,
        "phase": "DEVELOPMENT_EVALUATED",
        "protocol_state_sha256": sha256_path(Path(protocol_dir) / "state.json"),
        "scorer_locked_before_policy_tuning":
            not bool(state.get("policy_tuning_started")),
        "gold": {
            "path": str(Path(gold_path)),
            "sha256": sha256_path(gold_path),
            "counts": gold["counts"],
        },
        "co_primary": {
            "graded_ndcg_at_10": nested["aggregate_outer_predictions"],
            "top5_coherence": {
                "baseline":
                    nested["aggregate_outer_predictions"]["baseline"]
                    ["coherence_pass_rate"],
                "challenger":
                    nested["aggregate_outer_predictions"]["challenger"]
                    ["coherence_pass_rate"],
                "minimum": 0.80,
                "minimum_margin": 0.10,
            },
        },
        "nested_5fold": nested,
        "scene_held_out": scene_held_out,
        "selected_policy": asdict(selected),
        "selected_full_dev_evaluation": selected_summary,
        "actual_selected_lists": selected_predictions,
        "blind_judgment_protocol": {
            "lists": str(Path(blind_lists_path)),
            "lists_sha256": sha256_path(blind_lists_path),
            "judgments": str(Path(judgments_path)),
            "judgments_sha256": sha256_path(judgments_path),
            "unblinding_key": str(Path(blind_key_path)),
            "unblinding_key_sha256": sha256_path(blind_key_path),
            "method_blind": True,
            "model_assisted": False,
            "deterministic_source_grounded": True,
        },
        "gate_firing": {
            "fired": sum(bool(item["gate"]["fired"]) for item in selected_predictions),
            "abstained": sum(
                not bool(item["gate"]["fired"]) for item in selected_predictions
            ),
            "reasons": dict(sorted(gate_reasons.items())),
            "music4all_mandatory": False,
        },
        "supporting_only": {
            "preview_availability": preview_summary,
            "exact_pair_retrieval": exact_pair,
            "deezer_listenbrainz_musicbrainz": (
                "excluded from selection; prior source-isolation artifact retained"
            ),
            "source_isolation_artifact": (
                ".goals/human-quality-recommendations/artifacts/"
                "catalog-source-independence-v8.json"
            ),
            "source_isolation_sha256": sha256_path(
                ".goals/human-quality-recommendations/artifacts/"
                "catalog-source-independence-v8.json"
            ),
        },
        "all_powered_quality_dev_gates_passed": quality_pass,
        "verified_hosted_tier_gate_pending": True,
        "all_dev_preconditions_passed": False,
        "fresh_final_created": False,
        "final_open_count": 0,
        "deployment_attempted": False,
        "production_unchanged": True,
        "execution": {
            "seconds": time.perf_counter() - started,
            "policies": len(policies),
            "policy_record_cache_entries": len(policy_cache),
            "actual_served_top10_per_method_per_seed": True,
            "unopened_final_labels_compared": False,
        },
    }
    write_json(report_path, report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "evaluate"))
    parser.add_argument(
        "--protocol",
        default=(
            ".goals/human-quality-recommendations/"
            "protocol-v9-powered-development-r2"
        ),
    )
    parser.add_argument("--gold", default="benchmarks/soundalike_list_gold.v9.json")
    parser.add_argument(
        "--snapshots", default="benchmarks/evidence/v9/music-map.normalized.json"
    )
    parser.add_argument("--index", default="ml_data/deepvibe_index_v5.npz")
    parser.add_argument(
        "--graph", default="ml_data/iteration8/catalog-artist-graph-dual-v8.npz"
    )
    parser.add_argument(
        "--style", default="ml_data/iteration7/catalog-style-v8.npz"
    )
    artifact = ".goals/human-quality-recommendations/artifacts/"
    parser.add_argument(
        "--report", default=artifact + "catalog-powered-sonic-dev-v9.json"
    )
    parser.add_argument(
        "--blind-lists", default=artifact + "catalog-powered-blind-lists-v9.json"
    )
    parser.add_argument(
        "--judgments", default=artifact + "catalog-powered-blind-judgments-v9.json"
    )
    parser.add_argument(
        "--blind-key", default=artifact + "catalog-powered-blind-key-v9.json"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    common = {
        "protocol_dir": args.protocol,
        "gold_path": args.gold,
        "snapshots_path": args.snapshots,
        "index_path": args.index,
        "graph_path": args.graph,
        "style_path": args.style,
    }
    if args.action == "freeze":
        result = freeze_development_protocol(**common)
        print(json.dumps({
            "phase": result["state"]["phase"],
            "final_open_count": 0,
            "state_sha256": result["signature"]["state_sha256"],
        }, sort_keys=True))
    else:
        result = run_powered_development(
            **common,
            report_path=args.report,
            blind_lists_path=args.blind_lists,
            judgments_path=args.judgments,
            blind_key_path=args.blind_key,
        )
        print(json.dumps({
            "quality_pass": result["all_powered_quality_dev_gates_passed"],
            "selected_policy": result["selected_policy"],
            "final_open_count": result["final_open_count"],
        }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
