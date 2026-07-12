"""Protocol-v8 source audit and opened-data-only catalogue development.

This module deliberately has no operation that creates, opens, scores, or deploys
a FINAL.  Its only benchmark operation is cross-validation over previously opened
v6/v7 material.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .catalog_cv import (
    DEFAULT_POLICY_GRID,
    build_catalog_cv_report,
    candidate_recall,
    evaluate_seed,
    normalize_opened_benchmarks,
    policy_key,
    precompute_query_components,
    resolve_relevance,
    rescore_components,
)
from .catalog_graph import CatalogArtistGraph
from .catalog_policy import CatalogPolicy, CatalogPolicyRanker
from .catalog_style import CatalogStyleIndex
from .real_benchmark import PairResolver, ProductionRanker, normalize_text


class DevelopmentProtocolError(RuntimeError):
    """A fail-closed v8 development lock or execution error."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, str(path))
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_json(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return json.loads(Path(value).read_text(encoding="utf-8"))


def _record_primary_provider(record: Mapping[str, Any]) -> str:
    explicit = record.get("primary_source") or record.get("source_provider")
    if explicit:
        return str(explicit)
    source = record.get("source", {})
    if isinstance(source, Mapping):
        provider = source.get("publisher") or source.get("provider")
        if provider:
            return str(provider)
    sources = record.get("sources", [])
    if sources and isinstance(sources[0], Mapping):
        return str(
            sources[0].get("publisher") or sources[0].get("provider") or ""
        )
    positives = record.get("positives", [])
    if positives and isinstance(positives[0], Mapping):
        return str(positives[0].get("source_provider") or "")
    return ""


def _positive_artist(positive: Mapping[str, Any]) -> str:
    return str(
        positive.get("source_related_artist")
        or positive.get("artist")
        or ""
    )


def _edge_set(records: Sequence[Mapping[str, Any]]) -> set:
    edges = set()
    for record in records:
        query = normalize_text(str(record.get("query", {}).get("artist", "")))
        for positive in record.get("positives", []):
            target = normalize_text(_positive_artist(positive))
            if query and target and query != target:
                edges.add((query, target))
    return edges


def _graph_overlap(
    edges: Iterable[Tuple[str, str]],
    names: np.ndarray,
    indices: np.ndarray,
) -> Tuple[int, int, List[List[str]]]:
    lookup = {normalize_text(str(name)): row for row, name in enumerate(names)}
    resolvable = overlap = 0
    examples: List[List[str]] = []
    for query, target in sorted(edges):
        left, right = lookup.get(query), lookup.get(target)
        if left is None or right is None:
            continue
        resolvable += 1
        if int(right) in set(map(int, indices[int(left)])):
            overlap += 1
            if len(examples) < 20:
                examples.append([query, target])
    return resolvable, overlap, examples


def _music4all_top_neighbors(
    names: np.ndarray, vectors: np.ndarray, count: int = 96
) -> Tuple[np.ndarray, np.ndarray]:
    names = np.asarray([normalize_text(str(value)) for value in names])
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or len(names) != len(vectors):
        raise DevelopmentProtocolError("Music4All artist vectors are misaligned")
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-8)
    width = min(max(len(vectors) - 1, 0), int(count))
    neighbors = np.full((len(vectors), width), -1, dtype=np.int32)
    if not width:
        return names, neighbors
    for start in range(0, len(vectors), 256):
        stop = min(start + 256, len(vectors))
        scores = vectors[start:stop] @ vectors.T
        scores[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        if width == len(vectors) - 1:
            selected = np.argsort(-scores, axis=1, kind="stable")[:, :width]
        else:
            selected = np.argpartition(scores, -width, axis=1)[:, -width:]
            values = np.take_along_axis(scores, selected, axis=1)
            order = np.argsort(-values, axis=1, kind="stable")
            selected = np.take_along_axis(selected, order, axis=1)
        neighbors[start:stop] = selected
    return names, neighbors


def audit_source_independence(
    v7_benchmark_path: Any,
    catalog_graph_path: Any,
    music4all_full_path: Any,
) -> Dict[str, Any]:
    """Correct the v7 provenance erratum and quantify independent-source overlap."""
    benchmark_path = Path(v7_benchmark_path)
    graph_path = Path(catalog_graph_path)
    music_path = Path(music4all_full_path)
    benchmark = _load_json(benchmark_path)
    records = list(benchmark.get("records", []))
    if not records:
        raise DevelopmentProtocolError("v7 benchmark contains no records")

    providers = [_record_primary_provider(record) for record in records]
    non_deezer = [
        str(record.get("id", index))
        for index, (record, provider) in enumerate(zip(records, providers))
        if "deezer" not in provider.casefold()
    ]
    if non_deezer:
        raise DevelopmentProtocolError(
            "record primary source is not Deezer: " + ", ".join(non_deezer)
        )
    listenbrainz = sum(
        any(
            "listenbrainz"
            in str(source.get("publisher") or source.get("provider") or "").casefold()
            for source in record.get("sources", [])
            if isinstance(source, Mapping)
        )
        for record in records
    )
    edges = _edge_set(records)

    with np.load(graph_path, allow_pickle=False) as graph:
        graph_names = np.asarray(graph["artist_names"])
        graph_indices = np.asarray(graph["full_indices"], dtype=np.int32)
        graph_keys = list(graph.files)
    graph_resolved, graph_overlap, graph_examples = _graph_overlap(
        edges, graph_names, graph_indices
    )

    with np.load(music_path, allow_pickle=False) as music:
        music_names = np.asarray(music["artist_names"])
        vector_key = (
            "artist_vectors" if "artist_vectors" in music.files else "vectors"
        )
        music_vectors = np.asarray(music[vector_key], dtype=np.float32)
        music_keys = list(music.files)
    music_names, music_neighbors = _music4all_top_neighbors(
        music_names, music_vectors, 96
    )
    music_resolved, music_overlap, music_examples = _graph_overlap(
        edges, music_names, music_neighbors
    )

    forbidden = ("deezer", "source_artist_id", "deezer_track_id")
    graph_id_fields = [key for key in graph_keys if any(x in key.casefold() for x in forbidden)]
    music_id_fields = [key for key in music_keys if any(x in key.casefold() for x in forbidden)]
    no_shared_ids = not graph_id_fields and not music_id_fields
    return {
        "schema_version": 8,
        "audit_kind": "record-level-source-independence",
        "inputs": {
            "v7_benchmark_path": str(benchmark_path),
            "v7_benchmark_sha256": _sha256(benchmark_path),
            "catalog_graph_path": str(graph_path),
            "catalog_graph_sha256": _sha256(graph_path),
            "music4all_full_path": str(music_path),
            "music4all_full_sha256": _sha256(music_path),
        },
        "signed_v7_erratum": {
            "frozen_incorrect_declaration": (
                benchmark.get("source_policy", {}).get("automated_evaluation")
            ),
            "actual_primary_source": "Deezer related artists",
            "records": len(records),
            "records_with_deezer_primary": len(records),
            "records_with_listenbrainz_secondary": listenbrainz,
            "all_record_primary_sources_are_deezer": True,
        },
        "deezer_directed_edges": {
            "unique_query_positive_artist_edges": len(edges),
            "identity": "normalized artist names, directed query->positive",
        },
        "lastfm_360k_vs_deezer": {
            "dataset": "Last.fm-360K user/artist play histories vs Deezer related artists",
            "operator": "Last.fm dataset publisher vs Deezer service",
            "api": "offline Last.fm-360K archive vs api.deezer.com related-artists API",
            "id_namespace": "catalogue artist rows/names vs Deezer numeric artist/track IDs",
            "same_dataset": False,
            "same_operator": False,
            "same_api": False,
            "shared_numeric_id_namespace": False,
            "resolved_deezer_edges": graph_resolved,
            "full_top_neighbor_edge_overlap": graph_overlap,
            "overlap_fraction_of_resolved": graph_overlap / max(graph_resolved, 1),
            "overlap_examples": graph_examples,
        },
        "music4all_onion_vs_deezer": {
            "dataset": "Music4All-Onion listening histories vs Deezer related artists",
            "operator": "Music4All-Onion researchers/Last.fm extraction vs Deezer service",
            "api": "offline Music4All-Onion corpus vs api.deezer.com related-artists API",
            "id_namespace": "Music4All track IDs/catalogue rows vs Deezer numeric IDs",
            "same_dataset": False,
            "same_operator": False,
            "same_api": False,
            "shared_numeric_id_namespace": False,
            "overlap_kind": (
                "learned top-96 artist-neighborhood overlap from L2-normalized "
                "Music4All artist vectors; not raw cooccurrence"
            ),
            "raw_cooccurrence_available": False,
            "resolved_deezer_edges": music_resolved,
            "learned_top96_artist_neighborhood_overlap": music_overlap,
            "overlap_fraction_of_resolved": music_overlap / max(music_resolved, 1),
            "overlap_examples": music_examples,
        },
        "id_isolation": {
            "passed": no_shared_ids,
            "catalog_graph_forbidden_id_fields": graph_id_fields,
            "music4all_forbidden_id_fields": music_id_fields,
            "statement": (
                "Deezer numeric artist and track IDs never enter either training "
                "asset. Normalized artist names are used only for this read-only "
                "overlap audit."
            ),
        },
        "decision": {
            "unmasked_lastfm_full_direct_edges_allowed": no_shared_ids,
            "rationale": (
                "Unmasked full direct edges are intended legitimate query-conditioned "
                "collaborative signal because dataset, operator, API, and ID namespace "
                "are independent."
            ),
            "mask_policy": (
                "Direct and two-hop masks are mechanism diagnostics only and never "
                "deciding constraints."
            ),
        },
    }


def _referenced_paths(value: Any, parent_key: str = "") -> Iterable[Path]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _referenced_paths(child, str(key))
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _referenced_paths(child, parent_key)
    elif isinstance(value, (str, Path)) and (
        parent_key == "path"
        or parent_key.endswith("_path")
        or parent_key.endswith("_paths")
        or parent_key in {"cache", "asset", "index", "benchmark"}
    ):
        yield Path(value)


def _hash_references(*documents: Mapping[str, Any]) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    for document in documents:
        for path in _referenced_paths(document):
            if not path.is_file():
                raise DevelopmentProtocolError(
                    "referenced development input is missing: %s" % path
                )
            hashes[str(path)] = _sha256(path)
    return dict(sorted(hashes.items()))


def _ssh_sign_development(protocol_dir: Path, state_path: Path) -> Dict[str, Any]:
    executable = shutil.which("ssh-keygen")
    if executable is None:
        raise DevelopmentProtocolError("ssh-keygen is required; signing failed closed")
    with tempfile.TemporaryDirectory(prefix="soundalike-v8-key-") as temporary:
        private = Path(temporary) / "signer"
        generated = subprocess.run(
            [
                executable, "-q", "-t", "ed25519", "-N", "",
                "-C", "soundalike-protocol-v8-development", "-f", str(private),
            ],
            capture_output=True,
            check=False,
        )
        if generated.returncode != 0:
            raise DevelopmentProtocolError("ssh-keygen Ed25519 key generation failed")
        public_text = private.with_suffix(".pub").read_text(encoding="utf-8").strip()
        public_path = protocol_dir / "signer.pub"
        public_path.write_text(public_text + "\n", encoding="utf-8")
        fields = public_text.split()
        allowed_path = protocol_dir / "allowed_signers"
        allowed_path.write_text(
            "soundalike-protocol %s %s\n" % (fields[0], fields[1]),
            encoding="utf-8",
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
        if signed.returncode != 0 or not generated_signature.is_file():
            raise DevelopmentProtocolError("ssh-keygen detached signing failed")
        signature_path = protocol_dir / "state.sig"
        os.replace(str(generated_signature), str(signature_path))
    return {
        "algorithm": "Ed25519 detached SSH signature",
        "namespace": "soundalike-protocol",
        "identity": "soundalike-protocol",
        "state_sha256": _sha256(state_path),
        "public_key_sha256": _sha256(public_path),
        "allowed_signers_sha256": _sha256(allowed_path),
        "signature_sha256": _sha256(signature_path),
    }


def write_signed_development_protocol(
    protocol_dir: Any,
    audit: Mapping[str, Any],
    style_metadata: Mapping[str, Any],
    policy_grid: Sequence[CatalogPolicy] = DEFAULT_POLICY_GRID,
    *,
    signing_helper: Optional[Callable[[Path, Path], Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Create a new signed DEVELOPMENT-only v8 lock; an existing dir is rejected."""
    directory = Path(protocol_dir)
    if "protocol-v7" in {part.casefold() for part in directory.parts}:
        raise DevelopmentProtocolError("v7 is immutable and may not be read or written")
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise DevelopmentProtocolError("protocol-v8 directory must be new") from exc

    try:
        policies = tuple(policy_grid)
        if not policies:
            raise DevelopmentProtocolError("fixed policy grid may not be empty")
        inputs = _hash_references(audit, style_metadata)
        protocol = {
            "schema_version": 8,
            "kind": "development-protocol",
            "phase": "DEVELOPMENT_LOCKED",
            "corrected_provenance": audit,
            "style_metadata": style_metadata,
            "source_decision": {
                "full_unmasked_lastfm_edges": (
                    "legitimate deciding collaborative signal"
                ),
                "direct_and_twohop_masks": (
                    "mechanism diagnostics only; never constraints"
                ),
            },
            "scoring": {
                "G": (
                    "0.7*normalized_graph_edge_weight + "
                    "0.3/log2(graph_edge_rank+1)"
                ),
                "A": (
                    "mean(Sonic64 cosine mapped to [0,1], CLAP cosine mapped "
                    "to [0,1], 1/(1+vibe_distance))"
                ),
                "S": "MusicBrainz style-vector cosine overlap clipped to [0,1]",
                "rank_score": "G + audio_weight*A + style_weight*S",
                "top3_guard": (
                    "prefer S >= style_guard_min when at least three candidates "
                    "qualify"
                ),
            },
            "policy": {
                "numeric_parameters": [
                    "audio_weight", "style_weight", "style_guard_min"
                ],
                "numeric_parameter_count": 3,
                "fixed_grid": [asdict(policy) for policy in policies],
                "selection": "nested five-fold CV over opened DEV only",
            },
            "development_primary": {
                "formula": "0.80*nDCG@10 + 0.20*MusicBrainz style@3",
                "axis_metrics_reported_separately": True,
            },
            "gates": {
                "nested_5fold_required": True,
                "scene_held_out_required": True,
                "minimum_relative_composite_gain": 0.20,
                "maximum_per_scene_relative_regression": -0.10,
                "candidate_recall_at_1000_must_improve": True,
                "mrr_at_10_non_regression": True,
                "recall_at_10_non_regression": True,
                "direct_review_prerequisite": "at least 16/20",
            },
            "resources": {
                "peak_limit_gb": 1.5,
                "resident_target_gb": 1.1,
            },
            "final_and_deployment": {
                "fresh_final_creation_or_opening": (
                    "BLOCKED until every precondition passes"
                ),
                "deployment": "BLOCKED until every precondition passes",
            },
            "development_input_sha256": inputs,
        }
        protocol_path = directory / "development-protocol.json"
        _write_json(protocol_path, protocol)
        state = {
            "schema_version": 8,
            "phase": "DEVELOPMENT_LOCKED",
            "final_open_count": 0,
            "fresh_final_blocked": True,
            "deployment_blocked": True,
            "all_preconditions_passed": False,
            "protocol_path": str(protocol_path),
            "protocol_sha256": _sha256(protocol_path),
            "development_input_sha256": inputs,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "detached_signature_required": True,
        }
        state["integrity_signature"] = hashlib.sha256(_canonical_bytes(state)).hexdigest()
        state["signature_algorithm"] = "SHA-256 canonical JSON plus detached Ed25519"
        state_path = directory / "state.json"
        _write_json(state_path, state)
        metadata = dict((signing_helper or _ssh_sign_development)(directory, state_path))
        required = ("signer.pub", "allowed_signers", "state.sig")
        if not all((directory / name).is_file() for name in required):
            raise DevelopmentProtocolError("signing helper did not create required files")
        metadata.setdefault("state_sha256", _sha256(state_path))
        _write_json(directory / "signature-metadata.json", metadata)
        return {
            "protocol_dir": str(directory),
            "protocol": protocol,
            "state": state,
            "signature_metadata": metadata,
        }
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise


def _validate_opened_v7(path: Path, state_value: Any) -> Mapping[str, Any]:
    state = _load_json(state_value)
    if int(state.get("final_open_count", 0)) < 1:
        raise DevelopmentProtocolError("v7 is not already marked opened")
    expected = state.get("benchmark_sha256")
    if expected and _sha256(path) != expected:
        raise DevelopmentProtocolError("opened v7 benchmark hash does not match state")
    benchmark = _load_json(path)
    bad = [
        str(record.get("id", "?"))
        for record in benchmark.get("records", [])
        if "deezer" not in _record_primary_provider(record).casefold()
    ]
    if bad:
        raise DevelopmentProtocolError(
            "v7 record primary sources must be Deezer: " + ", ".join(bad)
        )
    return benchmark


def _average_metrics(rows: Sequence[Mapping[str, float]]) -> Dict[str, float]:
    keys = (
        "ndcg_at_10", "mrr_at_10", "recall_at_10",
        "style_coherence_at_3", "composite_primary",
    )
    return {
        key: float(mean(float(row.get(key, 0.0)) for row in rows)) if rows else 0.0
        for key in keys
    }


def _load_web_recommender(index_path: Path) -> Any:
    from webapp.api._reco import WebRecommender
    return WebRecommender(str(index_path))


def run_development_cv(
    v6_benchmark_path: Any,
    v7_benchmark_path: Any,
    v7_state_path: Any,
    index_path: Any,
    catalog_graph_path: Any,
    style_index_path: Any,
    report_path: Any,
    *,
    policies: Sequence[CatalogPolicy] = DEFAULT_POLICY_GRID,
    recommender_factory: Optional[Callable[[Path], Any]] = None,
    graph_factory: Callable[[Any], Any] = CatalogArtistGraph,
    style_factory: Callable[[Any], Any] = CatalogStyleIndex,
    resolver_factory: Callable[[Sequence[str], Sequence[str]], Any] = PairResolver,
    production_factory: Callable[..., Any] = ProductionRanker,
    ranker_factory: Callable[..., Any] = CatalogPolicyRanker,
    component_precomputer: Callable[..., Mapping[str, Any]] = precompute_query_components,
    report_builder: Callable[..., Dict[str, Any]] = build_catalog_cv_report,
) -> Dict[str, Any]:
    """Execute cached real DEV CV using production ``dual_sonic`` as baseline."""
    started = time.perf_counter()
    v6_path, v7_path = Path(v6_benchmark_path), Path(v7_benchmark_path)
    v7 = _validate_opened_v7(v7_path, v7_state_path)
    v6 = _load_json(v6_path)
    normalized = normalize_opened_benchmarks(v6, v7)
    all_records = normalized["records"]
    records: List[Mapping[str, Any]] = []
    unresolved_queries: List[str] = []
    unresolved_relevance: List[str] = []

    rec = (recommender_factory or _load_web_recommender)(Path(index_path))
    graph = graph_factory(catalog_graph_path)
    styles = style_factory(style_index_path)
    resolver = resolver_factory(rec.titles, rec.artists)
    production = production_factory(rec, set())
    component_ranker = ranker_factory(
        rec, graph, styles, CatalogPolicy(0.0, 0.0, 0.0)
    )

    query_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
    record_data: Dict[str, Dict[str, Any]] = {}
    precompute_started = time.perf_counter()
    for record in all_records:
        query = record["query"]
        query_key = (
            normalize_text(str(query.get("title", ""))),
            normalize_text(str(query.get("artist", ""))),
        )
        query_row = resolver.query_row(query)
        if query_row is None:
            unresolved_queries.append(str(record["id"]))
            continue
        if query_key not in query_cache:
            query_cache[query_key] = dict(
                component_precomputer(
                    component_ranker, production, int(query_row), candidate_limit=1000
                )
            )
        cached = query_cache[query_key]
        relevance = resolve_relevance(resolver, record)
        if not relevance:
            unresolved_relevance.append(str(record["id"]))
            continue
        records.append(record)
        baseline_ranking = [
            {
                "row": int(row),
                "S": styles.style_overlap(
                    str(rec.artists[int(query_row)]), str(rec.artists[int(row)])
                ),
            }
            for row in cached["production_rows"][:10]
        ]
        baseline_metrics = evaluate_seed(baseline_ranking, relevance)
        baseline_recall = candidate_recall(
            cached["production_rows"], relevance, 1000
        )
        union_recall = candidate_recall(
            cached["graph_union_rows"], relevance, 1000
        )
        record_data[str(record["id"])] = {
            "record": record,
            "query_artist": str(query.get("artist", "")),
            "query": cached,
            "relevance": relevance,
            "baseline": baseline_metrics,
            "baseline_recall": baseline_recall,
            "union_recall": union_recall,
        }
    precompute_seconds = time.perf_counter() - precompute_started
    if len(records) < 5:
        raise DevelopmentProtocolError(
            "nested five-fold CV requires at least five resolvable records"
        )

    policy_record_cache: Dict[Tuple[Tuple[float, float, float], str], Dict[str, float]] = {}

    def policy_metrics(policy: CatalogPolicy, record_id: str) -> Dict[str, float]:
        key = (policy_key(policy), record_id)
        if key not in policy_record_cache:
            data = record_data[record_id]
            ranking = rescore_components(data["query"]["components"], policy, 10)
            policy_record_cache[key] = evaluate_seed(ranking, data["relevance"])
        return policy_record_cache[key]

    def evaluator(
        policy: CatalogPolicy,
        _training: Sequence[Mapping[str, Any]],
        validation: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        ids = [str(record["id"]) for record in validation]
        baseline_rows = [record_data[value]["baseline"] for value in ids]
        challenger_rows = [policy_metrics(policy, value) for value in ids]
        baseline = _average_metrics(baseline_rows)
        challenger = _average_metrics(challenger_rows)

        def grouped(field: str) -> Dict[str, Any]:
            groups: Dict[str, List[int]] = defaultdict(list)
            for position, record in enumerate(validation):
                groups[str(record.get(field, "unknown"))].append(position)
            return {
                group: {
                    "records": len(positions),
                    "baseline": _average_metrics([baseline_rows[p] for p in positions]),
                    "challenger": _average_metrics([challenger_rows[p] for p in positions]),
                }
                for group, positions in sorted(groups.items())
            }

        improved = sum(
            challenger_rows[i]["composite_primary"]
            > baseline_rows[i]["composite_primary"]
            for i in range(len(ids))
        )
        worsened = sum(
            challenger_rows[i]["composite_primary"]
            < baseline_rows[i]["composite_primary"]
            for i in range(len(ids))
        )
        baseline_recall = float(mean(record_data[x]["baseline_recall"] for x in ids))
        union_recall = float(mean(record_data[x]["union_recall"] for x in ids))
        return {
            "baseline": baseline,
            "challenger": challenger,
            "per_axis": grouped("evidence_axis"),
            "per_scene": grouped("scene"),
            "improved": int(improved),
            "worsened": int(worsened),
            "unchanged": len(ids) - int(improved) - int(worsened),
            "baseline_candidate_recall_at_1000": baseline_recall,
            "challenger_candidate_recall_at_1000": union_recall,
        }

    cv_started = time.perf_counter()
    filtered_v6 = dict(v6)
    filtered_v6["pairs"] = [
        record["source_record"]
        for record in records
        if int(record["source_version"]) == 6
    ]
    filtered_v7 = dict(v7)
    filtered_v7["records"] = [
        record["source_record"]
        for record in records
        if int(record["source_version"]) == 7
    ]
    report = report_builder(
        filtered_v6, filtered_v7, evaluator, policies=policies
    )
    report["opened_evidence_resolution"] = {
        "considered_records": len(all_records),
        "evaluated_records": len(records),
        "unresolved_queries": unresolved_queries,
        "unresolved_relevance": unresolved_relevance,
        "all_opened_records_considered": True,
        "exclusion_rule": "only catalogue-unresolvable query or positive labels",
    }
    cv_seconds = time.perf_counter() - cv_started
    selected_value = report.get("nested_5fold", {}).get("final_policy")
    if isinstance(selected_value, Mapping):
        selected = CatalogPolicy(
            float(selected_value["audio_weight"]),
            float(selected_value["style_weight"]),
            float(selected_value["style_guard_min"]),
        )
        report["selected_policy_evaluation"] = {
            "exact_policy": asdict(selected),
            "aggregate_and_slices": evaluator(selected, records, records),
        }
    state_hashes = {}
    if isinstance(v7_state_path, Mapping):
        state_hashes["inline_v7_opened_state"] = hashlib.sha256(
            _canonical_bytes(v7_state_path)
        ).hexdigest()
    else:
        state_hashes[str(Path(v7_state_path))] = _sha256(Path(v7_state_path))
    report["execution"] = {
        "phase": "DEVELOPMENT_LOCKED",
        "production_baseline": "dual_sonic",
        "v7_already_opened": True,
        "fresh_final_inputs_accepted": False,
        "unique_queries": len(query_cache),
        "records": len(records),
        "policy_record_cache_entries": len(policy_record_cache),
        "candidate_components": "complete full graph plus fixed 1000-audio pool",
        "relevance_computed_separately_per_record": True,
        "timing_seconds": {
            "precompute": precompute_seconds,
            "cross_validation": cv_seconds,
            "total": time.perf_counter() - started,
        },
        "input_sha256": {
            str(v6_path): _sha256(v6_path),
            str(v7_path): _sha256(v7_path),
            str(Path(index_path)): _sha256(Path(index_path)),
            str(Path(catalog_graph_path)): _sha256(Path(catalog_graph_path)),
            str(Path(style_index_path)): _sha256(Path(style_index_path)),
            **state_hashes,
        },
    }
    output = Path(report_path)
    _write_json(output, report)
    return report


execute_development_cv = run_development_cv


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit")
    audit.add_argument("v7_benchmark", type=Path)
    audit.add_argument("catalog_graph", type=Path)
    audit.add_argument("music4all_full", type=Path)
    audit.add_argument("--output", type=Path)

    lock = subparsers.add_parser("lock-development")
    lock.add_argument("protocol_dir", type=Path)
    lock.add_argument("audit_json", type=Path)
    lock.add_argument("style_metadata_json", type=Path)

    dev = subparsers.add_parser("dev-cv")
    dev.add_argument("v6_benchmark", type=Path)
    dev.add_argument("v7_benchmark", type=Path)
    dev.add_argument("v7_state", type=Path)
    dev.add_argument("index", type=Path)
    dev.add_argument("catalog_graph", type=Path)
    dev.add_argument("style_index", type=Path)
    dev.add_argument("report", type=Path)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "audit":
        result = audit_source_independence(
            args.v7_benchmark, args.catalog_graph, args.music4all_full
        )
        if args.output:
            _write_json(args.output, result)
        print(json.dumps(result, indent=2))
    elif args.command == "lock-development":
        result = write_signed_development_protocol(
            args.protocol_dir, _load_json(args.audit_json),
            _load_json(args.style_metadata_json),
        )
        print(json.dumps(result, indent=2))
    elif args.command == "dev-cv":
        result = run_development_cv(
            args.v6_benchmark, args.v7_benchmark, args.v7_state, args.index,
            args.catalog_graph, args.style_index, args.report,
        )
        print(json.dumps(result["execution"], indent=2))
    else:  # pragma: no cover - argparse makes this unreachable
        raise DevelopmentProtocolError("unsupported DEVELOPMENT command")


__all__ = [
    "DEFAULT_POLICY_GRID",
    "DevelopmentProtocolError",
    "audit_source_independence",
    "write_signed_development_protocol",
    "run_development_cv",
    "execute_development_cv",
    "main",
]


if __name__ == "__main__":
    main()
