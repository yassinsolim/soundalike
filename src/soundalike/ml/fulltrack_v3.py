"""Frozen V3 MusicFM selective-reranker audit and promotion gates."""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .fulltrack_eval import (
    METRICS,
    OFFICIAL_FOLDS,
    _BudgetCache,
    _method_ranking,
    _paired_bootstrap_delta,
    _query_metrics,
    _tag_jaccard_relevance,
    batch_fixed_budget_maxsim,
    write_evaluation_report,
)
from .fulltrack_extract import normalize_rows
from .fulltrack_store import (
    FullTrackStoreError,
    FullTrackStoreReader,
    sha256_path,
    stable_json_sha256,
)
from .jamendo_fulltrack import (
    EVIDENCE_SCOPE,
    JamendoContext,
    JamendoTrack,
    JamendoValidationError,
    load_jamendo_context,
)


AUDIT_SCHEMA_VERSION = 1
AUDIT_KIND = "musicfm_selective_reranker_frozen_test_audit"
POLICY_KIND = "musicfm_selective_gate_nested_leave_one_fold_out"
POLICY_FILE_SHA256 = "3a90f0a7b5f4776ae8f450473b2ebb1504b3439ed9c0010baf3c47151d5eda64"
CLAP_MANIFEST_FILE_SHA256 = (
    "82f593136408a893a71d350af8e3356356e8ea5c041f3c1293abe65183388409"
)
MUSICFM_MANIFEST_FILE_SHA256 = (
    "cbdd8d43d7b00bc923fa80ab3d6b262b28630da83fc6b4394e0e828e8f257428"
)
SOURCE_FINGERPRINT = "060f43ed0fa12e5a583e26a7728be14a5334c7daffebe2289f08875e9ec0c709"
SELECTION_SEED = 20260731
TRACKS_PER_FOLD = 512
QUERY_LIMIT = 128
CANDIDATE_POOL = 200
MAXSIM_BUDGET = 8
MAX_FEATURE_CACHE_BYTES = 2 * 1024**3
BOOTSTRAP_ITERATIONS = 5_000
BOOTSTRAP_SEED = 20260714
PRIMARY_METRIC = "recall_at_k"
MIN_PRIMARY_RELATIVE_GAIN = 0.20
MAX_SAFETY_RELATIVE_REGRESSION = 0.01
MIN_POSITIVE_FOLDS = 4
MAX_FOLD_PRIMARY_RELATIVE_REGRESSION = 0.05

EXPECTED_SELECTION_SHA256 = {
    0: "831f52e8161e09850923a52216e69eec40f8ef1fa7c45abae0e24d2ab021a199",
    1: "7de2df27ae7d6fa4cd6f56a37c0dc4ad5d6a9d9771670a47945c630161f32b0e",
    2: "6e3555064b8ca5d8a8275aadad2612eb02a0f2e960bcfcae6a3a98c524914073",
    3: "963ed21a1442bc14a3f9112983d55a98d0c8cbf34802d909b1206c99f14eb33a",
    4: "10564b5f8a00808b028bab8bca1d8cc4687cebfec903c778248d54e537bcce4d",
}
EXPECTED_UNION_SHA256 = "c43dea0e8032c2e900786d3b7c5898a74317b8c4b12867889196402440718bc5"

EXPECTED_CLAP_BINDING = {
    "config_sha256": "32f29427f8b8c19d809f13c4d062baec18461c3b71d63a40b09aa0788572a0d9",
    "model_sha256": "8053c9775516af2f4902e1e8281e356cc1bf7a85e8b761908170767b77c3f037",
    "model_id": "laion_clap_htsat_tiny_music_audioset_630k_nonfusion",
    "embedding_dim": 512,
    "track_count": 55_701,
    "shard_tracks": 256,
    "repetition_sections": 32,
    "salient_sections": 32,
    "track_plan_sha256": "6aaff026be51a7edb48aff80bc993460eaca65e793bff1545851d5511cebb244",
}
EXPECTED_MUSICFM_BINDING = {
    "config_sha256": "3525e71717800dde5091fbd27e29f753b62055a2631f4a905909bea2801ddffd",
    "model_sha256": "68392eee13d34c2941b3761934abb6b1e67b2e9df498695bda2ea5c1087d4b96",
    "model_id": "musicfm_fma_b83ebed_layer7",
    "embedding_dim": 1024,
    "track_count": 1_702,
    "shard_tracks": 64,
    "repetition_sections": 32,
    "salient_sections": 32,
    "track_plan_sha256": "6e052aff0fb20961dbb206b2e060abe1c3cfd02ce3f295cb61a2e56dcb1dc000",
}


class FullTrackV3Error(RuntimeError):
    """Frozen V3 input, protocol, evidence, or promotion-gate failure."""


@dataclass(frozen=True)
class SelectivePolicy:
    weight: float
    feature: str
    direction: str
    threshold: float


def _validate_policy_document(document: object) -> SelectivePolicy:
    if not isinstance(document, dict):
        raise FullTrackV3Error("policy must be a JSON object")
    if document.get("artifact_kind") != POLICY_KIND:
        raise FullTrackV3Error("policy artifact kind drift")
    if document.get("evidence_scope") != EVIDENCE_SCOPE:
        raise FullTrackV3Error("policy evidence scope drift")
    if document.get("held_out_test_accessed") is not False:
        raise FullTrackV3Error("policy was not frozen before held-out test access")
    raw = document.get("final_validation_policy")
    if not isinstance(raw, dict):
        raise FullTrackV3Error("policy has no final validation policy")
    if (
        raw.get("weight") != "0.25"
        or raw.get("feature") != "music_std"
        or raw.get("direction") != "le"
        or raw.get("threshold") != 0.05948563385754824
    ):
        raise FullTrackV3Error("frozen selective policy drift")
    return SelectivePolicy(
        weight=0.25,
        feature="music_std",
        direction="le",
        threshold=0.05948563385754824,
    )


def load_frozen_policy(path: Path) -> Tuple[SelectivePolicy, Mapping[str, object]]:
    candidate = Path(path).absolute()
    if candidate.is_symlink():
        raise FullTrackV3Error("policy must be a concrete file")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file():
        raise FullTrackV3Error("policy must be a concrete file")
    if sha256_path(resolved) != POLICY_FILE_SHA256:
        raise FullTrackV3Error("policy file SHA-256 drift")
    try:
        document = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FullTrackV3Error(f"invalid policy JSON: {exc}") from exc
    return _validate_policy_document(document), document


def _open_bound_store(
    root: Path,
    *,
    expected_manifest_file_sha256: str,
    expected_binding: Mapping[str, object],
) -> FullTrackStoreReader:
    candidate = Path(root).absolute()
    if candidate.is_symlink():
        raise FullTrackV3Error("sealed store root must be a concrete directory")
    root = candidate.resolve(strict=True)
    manifest_candidate = root / "store.sealed.json"
    if manifest_candidate.is_symlink():
        raise FullTrackV3Error("sealed store manifest must be a concrete file")
    manifest = manifest_candidate.resolve(strict=True)
    if not manifest.is_file():
        raise FullTrackV3Error("sealed store manifest must be a concrete file")
    if sha256_path(manifest) != expected_manifest_file_sha256:
        raise FullTrackV3Error("sealed store manifest file SHA-256 drift")
    reader = FullTrackStoreReader(
        root,
        expected_source_fingerprint=SOURCE_FINGERPRINT,
        expected_config_sha256=str(expected_binding["config_sha256"]),
        expected_model_sha256=str(expected_binding["model_sha256"]),
    )
    actual = reader.binding.as_dict()
    drift = {
        key: (expected, actual.get(key))
        for key, expected in expected_binding.items()
        if actual.get(key) != expected
    }
    if drift:
        reader.close()
        raise FullTrackV3Error(f"sealed store binding drift: {drift}")
    return reader


def _fold(context: JamendoContext, fold_index: int):
    return next(
        (item for item in context.folds if item.index == fold_index),
        None,
    )


def selected_test_tracks(
    context: JamendoContext, fold_index: int
) -> Tuple[JamendoTrack, ...]:
    fold = _fold(context, fold_index)
    if fold is None:
        raise FullTrackV3Error(f"official fold {fold_index} is missing")
    eligible = [
        track
        for track in context.tracks
        if fold.track_parts.get(track.track_id) == "test"
    ]
    if len(eligible) < TRACKS_PER_FOLD:
        raise FullTrackV3Error(f"fold {fold_index} test partition is too small")
    selected = tuple(
        sorted(
            eligible,
            key=lambda track: stable_json_sha256(
                {"seed": SELECTION_SEED, "track_id": track.track_id}
            ),
        )[:TRACKS_PER_FOLD]
    )
    selection_hash = stable_json_sha256(
        tuple(track.track_id for track in selected)
    )
    if selection_hash != EXPECTED_SELECTION_SHA256[fold_index]:
        raise FullTrackV3Error(f"fold {fold_index} frozen selection drift")
    return selected


def _zscore(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.all(np.isfinite(array)):
        raise FullTrackV3Error("reranker scores must be a finite vector")
    standard_deviation = float(np.std(array))
    divisor = standard_deviation if standard_deviation > 1e-8 else 1.0
    return (array - float(np.mean(array))) / divisor


def selective_reranker_scores(
    clap_hybrid: np.ndarray,
    music_uniform: np.ndarray,
    policy: SelectivePolicy,
) -> Tuple[np.ndarray, bool, float]:
    clap = np.asarray(clap_hybrid, dtype=np.float64)
    music = np.asarray(music_uniform, dtype=np.float64)
    if clap.shape != music.shape or clap.ndim != 1:
        raise FullTrackV3Error("CLAP and MusicFM candidate scores differ in shape")
    if not np.all(np.isfinite(clap)) or not np.all(np.isfinite(music)):
        raise FullTrackV3Error("CLAP and MusicFM candidate scores must be finite")
    music_std = float(np.std(music))
    applied = music_std <= policy.threshold
    if not applied:
        return clap.copy(), False, music_std
    fused = (1.0 - policy.weight) * _zscore(clap) + policy.weight * _zscore(music)
    return fused, True, music_std


def _mean_metrics(records: Sequence[Mapping[str, object]], method: str) -> Dict[str, float]:
    if not records:
        raise FullTrackV3Error("no evaluable queries")
    return {
        metric: float(
            np.mean([record["metrics"][method][metric] for record in records])
        )
        for metric in METRICS
    }


def _relative_delta(baseline: float, candidate: float) -> Optional[float]:
    if baseline == 0.0:
        return None
    return candidate / baseline - 1.0


def _aggregate(records: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    baseline = _mean_metrics(records, "clap_hybrid")
    candidate = _mean_metrics(records, "selective_reranker")
    paired = {
        metric: _paired_bootstrap_delta(
            [record["metrics"]["clap_hybrid"][metric] for record in records],
            [record["metrics"]["selective_reranker"][metric] for record in records],
            iterations=BOOTSTRAP_ITERATIONS,
            seed=BOOTSTRAP_SEED,
        )
        for metric in METRICS
    }
    return {
        "clap_hybrid": baseline,
        "selective_reranker": candidate,
        "absolute_delta": {
            metric: candidate[metric] - baseline[metric] for metric in METRICS
        },
        "relative_delta": {
            metric: _relative_delta(baseline[metric], candidate[metric])
            for metric in METRICS
        },
        "paired_delta": paired,
    }


def _evaluate_fold(
    context: JamendoContext,
    fold_index: int,
    selected: Sequence[JamendoTrack],
    clap_reader: FullTrackStoreReader,
    music_reader: FullTrackStoreReader,
    policy: SelectivePolicy,
) -> Tuple[Dict[str, object], int]:
    fold = _fold(context, fold_index)
    if fold is None:
        raise FullTrackV3Error(f"official fold {fold_index} is missing")
    selected_ids = [track.track_id for track in selected]
    position = {track_id: index for index, track_id in enumerate(selected_ids)}
    clap_cache = _BudgetCache(
        clap_reader,
        selected_ids,
        budget=MAXSIM_BUDGET,
        max_bytes=MAX_FEATURE_CACHE_BYTES,
    )
    music_cache = _BudgetCache(
        music_reader,
        selected_ids,
        budget=MAXSIM_BUDGET,
        max_bytes=MAX_FEATURE_CACHE_BYTES,
    )
    clap_globals = normalize_rows(
        np.asarray(
            clap_reader.global_embeddings[
                [
                    clap_reader.read_track(track_id).row_index
                    for track_id in selected_ids
                ]
            ],
            dtype=np.float32,
        )
    )
    records = []
    skipped = 0
    for query in selected[:QUERY_LIMIT]:
        query_position = position[query.track_id]
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
            if candidate.track_id != query.track_id
            and candidate.artist_id != query.artist_id
            and (
                grade := _tag_jaccard_relevance(
                    fold.track_tags[query.track_id],
                    fold.track_tags[candidate.track_id],
                    min_shared_tags=2,
                    min_tag_jaccard=0.25,
                )
            )
        }
        if not relevant:
            skipped += 1
            continue
        global_scores = clap_globals[eligible] @ clap_globals[query_position]
        global_order = eligible[np.lexsort((eligible, -global_scores))]
        pool = global_order[: min(CANDIDATE_POOL, len(global_order))]
        clap_uniform = batch_fixed_budget_maxsim(
            clap_cache.uniform[clap_cache.rows[query.track_id]],
            clap_cache.uniform[pool].astype(np.float32),
        )
        clap_section = 0.5 * (
            batch_fixed_budget_maxsim(
                clap_cache.repeated[clap_cache.rows[query.track_id]],
                clap_cache.repeated[pool].astype(np.float32),
            )
            + batch_fixed_budget_maxsim(
                clap_cache.salient[clap_cache.rows[query.track_id]],
                clap_cache.salient[pool].astype(np.float32),
            )
        )
        clap_hybrid = (
            0.50 * (clap_globals[pool] @ clap_globals[query_position])
            + 0.25 * clap_uniform
            + 0.25 * clap_section
        )
        music_uniform = batch_fixed_budget_maxsim(
            music_cache.uniform[music_cache.rows[query.track_id]],
            music_cache.uniform[pool].astype(np.float32),
        )
        selective_scores, applied, music_std = selective_reranker_scores(
            clap_hybrid, music_uniform, policy
        )
        baseline_order = _method_ranking(clap_hybrid, pool, global_order)
        candidate_order = _method_ranking(selective_scores, pool, global_order)

        def metrics(order: np.ndarray) -> Dict[str, float]:
            ranked_ids = [selected[int(index)].track_id for index in order]
            return asdict(
                _query_metrics(
                    ranked_ids,
                    relevant,
                    recall_cutoff=10,
                    ndcg_cutoff=10,
                )
            )

        records.append(
            {
                "track_id": query.track_id,
                "artist_id": query.artist_id,
                "tags": list(fold.track_tags[query.track_id]),
                "relevant_candidates": len(relevant),
                "policy_applied": applied,
                "music_uniform_std": music_std,
                "metrics": {
                    "clap_hybrid": metrics(baseline_order),
                    "selective_reranker": metrics(candidate_order),
                },
            }
        )
    aggregate = _aggregate(records)
    aggregate.update(
        {
            "fold_index": fold_index,
            "queries": len(records),
            "skipped_no_relevant": skipped,
            "policy_applied_queries": sum(
                bool(record["policy_applied"]) for record in records
            ),
            "query_records": records,
        }
    )
    return aggregate, clap_cache.bytes + music_cache.bytes


def promotion_gates(
    fold_results: Sequence[Mapping[str, object]],
    pooled: Mapping[str, object],
) -> Dict[str, object]:
    primary_relative = pooled["relative_delta"][PRIMARY_METRIC]
    primary_ci_low = pooled["paired_delta"][PRIMARY_METRIC][
        "paired_bootstrap_ci95"
    ][0]
    fold_primary_relative = [
        result["relative_delta"][PRIMARY_METRIC] for result in fold_results
    ]
    if primary_relative is None or any(value is None for value in fold_primary_relative):
        raise FullTrackV3Error("primary relative gain is undefined")
    positive_folds = sum(value > 0.0 for value in fold_primary_relative)
    primary_checks = {
        "relative_gain_at_least_20_percent": (
            primary_relative >= MIN_PRIMARY_RELATIVE_GAIN
        ),
        "paired_ci_lower_bound_above_zero": primary_ci_low > 0.0,
        "positive_on_at_least_four_folds": positive_folds >= MIN_POSITIVE_FOLDS,
        "no_fold_regresses_more_than_five_percent": (
            min(fold_primary_relative) >= -MAX_FOLD_PRIMARY_RELATIVE_REGRESSION
        ),
    }
    safety_checks = {
        metric: (
            pooled["relative_delta"][metric] is not None
            and pooled["relative_delta"][metric]
            >= -MAX_SAFETY_RELATIVE_REGRESSION
        )
        for metric in ("recall_at_k", "mrr")
    }
    automated_passed = all(primary_checks.values()) and all(safety_checks.values())
    return {
        "primary_metric": PRIMARY_METRIC,
        "thresholds": {
            "minimum_relative_gain": MIN_PRIMARY_RELATIVE_GAIN,
            "maximum_safety_relative_regression": MAX_SAFETY_RELATIVE_REGRESSION,
            "minimum_positive_folds": MIN_POSITIVE_FOLDS,
            "maximum_fold_primary_relative_regression": (
                MAX_FOLD_PRIMARY_RELATIVE_REGRESSION
            ),
        },
        "observed": {
            "pooled_primary_relative_gain": primary_relative,
            "pooled_primary_paired_ci95": pooled["paired_delta"][PRIMARY_METRIC][
                "paired_bootstrap_ci95"
            ],
            "positive_primary_folds": positive_folds,
            "fold_primary_relative_gain": fold_primary_relative,
        },
        "primary_checks": primary_checks,
        "safety_checks": safety_checks,
        "automated_passed": automated_passed,
        "human_pilot_passed": False,
        "human_pilot_required": True,
        "promotion_allowed": False,
    }


def _write_json_exclusive(path: Path, value: Mapping[str, object]) -> None:
    path = Path(path).absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _replace_json(path: Path, value: Mapping[str, object]) -> None:
    path = Path(path).absolute()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _resolve_input_directory(path: Path, label: str) -> Path:
    candidate = Path(path).absolute()
    if candidate.is_symlink():
        raise FullTrackV3Error(f"{label} must be a concrete directory")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise FullTrackV3Error(f"{label} does not exist: {candidate}") from exc
    if not resolved.is_dir():
        raise FullTrackV3Error(f"{label} is not a directory: {resolved}")
    return resolved


def run_frozen_audit(
    *,
    metadata_root: Path,
    audio_root: Path,
    state_root: Path,
    clap_store: Path,
    musicfm_store: Path,
    policy_path: Path,
    output_path: Path,
    audit_state_path: Path,
) -> Mapping[str, object]:
    output_path = Path(output_path).absolute()
    audit_state_path = Path(audit_state_path).absolute()
    if output_path.exists():
        raise FullTrackV3Error("audit output already exists; refusing overwrite")
    if audit_state_path.exists():
        raise FullTrackV3Error("held-out audit state already exists; refusing reopen")
    metadata_root = _resolve_input_directory(metadata_root, "metadata root")
    audio_root = _resolve_input_directory(audio_root, "audio root")
    state_root = _resolve_input_directory(state_root, "state root")

    policy, policy_document = load_frozen_policy(policy_path)
    clap_reader = _open_bound_store(
        clap_store,
        expected_manifest_file_sha256=CLAP_MANIFEST_FILE_SHA256,
        expected_binding=EXPECTED_CLAP_BINDING,
    )
    try:
        music_reader = _open_bound_store(
            musicfm_store,
            expected_manifest_file_sha256=MUSICFM_MANIFEST_FILE_SHA256,
            expected_binding=EXPECTED_MUSICFM_BINDING,
        )
    except Exception:
        clap_reader.close()
        raise
    opened_state = {
        "schema_version": 1,
        "status": "opened",
        "policy_file_sha256": POLICY_FILE_SHA256,
        "clap_manifest_file_sha256": CLAP_MANIFEST_FILE_SHA256,
        "musicfm_manifest_file_sha256": MUSICFM_MANIFEST_FILE_SHA256,
        "output_path": str(output_path),
    }
    _write_json_exclusive(audit_state_path, opened_state)
    started = time.perf_counter()
    try:
        context = load_jamendo_context(
            Path(metadata_root),
            Path(audio_root),
            Path(state_root),
            production=True,
        )
        if context.source_fingerprint != SOURCE_FINGERPRINT:
            raise FullTrackV3Error("Jamendo source fingerprint drift")
        selections = {
            fold_index: selected_test_tracks(context, fold_index)
            for fold_index in OFFICIAL_FOLDS
        }
        union_ids = tuple(
            sorted(
                {
                    track.track_id
                    for selected in selections.values()
                    for track in selected
                }
            )
        )
        if stable_json_sha256(union_ids) != EXPECTED_UNION_SHA256:
            raise FullTrackV3Error("frozen test-union selection drift")
        if tuple(music_reader.track_ids) != union_ids:
            raise FullTrackV3Error("MusicFM store does not exactly cover the test union")
        if not set(union_ids).issubset(set(clap_reader.track_ids)):
            raise FullTrackV3Error("CLAP store does not cover the test union")

        fold_results = []
        feature_cache_bytes = 0
        for fold_index in OFFICIAL_FOLDS:
            result, cache_bytes = _evaluate_fold(
                context,
                fold_index,
                selections[fold_index],
                clap_reader,
                music_reader,
                policy,
            )
            fold_results.append(result)
            feature_cache_bytes = max(feature_cache_bytes, cache_bytes)
        pooled_records = [
            record
            for result in fold_results
            for record in result["query_records"]
        ]
        pooled = _aggregate(pooled_records)
        pooled["queries"] = len(pooled_records)
        pooled["policy_applied_queries"] = sum(
            bool(record["policy_applied"]) for record in pooled_records
        )
        gates = promotion_gates(fold_results, pooled)
        report: Dict[str, object] = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "artifact_kind": AUDIT_KIND,
            "evidence_scope": EVIDENCE_SCOPE,
            "protocol": {
                "folds": list(OFFICIAL_FOLDS),
                "part": "test",
                "selection_seed": SELECTION_SEED,
                "tracks_per_fold": TRACKS_PER_FOLD,
                "query_limit": QUERY_LIMIT,
                "candidate_pool": CANDIDATE_POOL,
                "maxsim_budget": MAXSIM_BUDGET,
                "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "min_shared_tags": 2,
                "min_tag_jaccard": 0.25,
                "selection_sha256": dict(EXPECTED_SELECTION_SHA256),
                "union_sha256": EXPECTED_UNION_SHA256,
            },
            "policy": asdict(policy),
            "policy_file_sha256": POLICY_FILE_SHA256,
            "policy_validation_evidence": {
                "artifact_kind": policy_document["artifact_kind"],
                "outer_mean_deltas": policy_document["outer_mean_deltas"],
                "held_out_test_accessed_at_freeze": policy_document[
                    "held_out_test_accessed"
                ],
            },
            "stores": {
                "clap": {
                    **clap_reader.binding.as_dict(),
                    "manifest_file_sha256": CLAP_MANIFEST_FILE_SHA256,
                    "storage_bytes": clap_reader.storage_bytes,
                },
                "musicfm": {
                    **music_reader.binding.as_dict(),
                    "manifest_file_sha256": MUSICFM_MANIFEST_FILE_SHA256,
                    "storage_bytes": music_reader.storage_bytes,
                },
            },
            "fold_results": fold_results,
            "pooled": pooled,
            "promotion_gates": gates,
            "resources": {
                "wall_seconds": time.perf_counter() - started,
                "feature_cache_peak_bytes": feature_cache_bytes,
            },
            "held_out_test_accessed": True,
            "promotion_allowed": False,
        }
        report["artifact_payload_sha256"] = stable_json_sha256(report)
        write_evaluation_report(output_path, report)
        completed_state = {
            **opened_state,
            "status": "completed",
            "report_file_sha256": sha256_path(output_path),
            "report_payload_sha256": report["artifact_payload_sha256"],
            "automated_passed": gates["automated_passed"],
            "promotion_allowed": False,
        }
        _replace_json(audit_state_path, completed_state)
        return report
    finally:
        clap_reader.close()
        music_reader.close()


def verify_audit_report(path: Path) -> Mapping[str, object]:
    candidate = Path(path).absolute()
    if candidate.is_symlink():
        raise FullTrackV3Error("audit report must be a concrete file")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file():
        raise FullTrackV3Error("audit report must be a concrete file")
    try:
        report = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FullTrackV3Error(f"invalid audit report: {exc}") from exc
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != AUDIT_SCHEMA_VERSION
        or report.get("artifact_kind") != AUDIT_KIND
        or report.get("evidence_scope") != EVIDENCE_SCOPE
    ):
        raise FullTrackV3Error("audit report envelope drift")
    declared = report.pop("artifact_payload_sha256", None)
    actual = stable_json_sha256(report)
    report["artifact_payload_sha256"] = declared
    if declared != actual:
        raise FullTrackV3Error("audit report payload checksum mismatch")
    if report.get("promotion_allowed") is not False:
        raise FullTrackV3Error("automated audit may not authorize promotion")
    return report


def _audit_command(args: argparse.Namespace) -> int:
    report = run_frozen_audit(
        metadata_root=Path(args.metadata_root),
        audio_root=Path(args.audio_root),
        state_root=Path(args.state_root),
        clap_store=Path(args.clap_store),
        musicfm_store=Path(args.musicfm_store),
        policy_path=Path(args.policy),
        output_path=Path(args.output),
        audit_state_path=Path(args.audit_state),
    )
    print(
        json.dumps(
            {
                "output": str(Path(args.output).absolute()),
                "pooled": report["pooled"],
                "promotion_gates": report["promotion_gates"],
                "promotion_allowed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _verify_command(args: argparse.Namespace) -> int:
    report = verify_audit_report(Path(args.report))
    print(
        json.dumps(
            {
                "report": str(Path(args.report).resolve()),
                "artifact_payload_sha256": report["artifact_payload_sha256"],
                "promotion_gates": report["promotion_gates"],
                "promotion_allowed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser(
        "audit-frozen-test",
        help="open the hash-frozen five-fold test protocol exactly once",
    )
    audit.add_argument("--metadata-root", required=True)
    audit.add_argument("--audio-root", required=True)
    audit.add_argument("--state-root", required=True)
    audit.add_argument("--clap-store", required=True)
    audit.add_argument("--musicfm-store", required=True)
    audit.add_argument("--policy", required=True)
    audit.add_argument("--output", required=True)
    audit.add_argument("--audit-state", required=True)
    audit.set_defaults(handler=_audit_command)
    verify = subparsers.add_parser(
        "verify-audit",
        help="verify a completed audit without reopening held-out labels",
    )
    verify.add_argument("--report", required=True)
    verify.set_defaults(handler=_verify_command)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (
        FullTrackStoreError,
        FullTrackV3Error,
        JamendoValidationError,
        OSError,
    ) as exc:
        raise SystemExit(f"V3 audit blocked: {exc}") from exc


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
