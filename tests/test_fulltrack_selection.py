"""Synthetic tests for soundalike.ml.fulltrack_selection.

All test data is synthetic-test-fixture-only; no real audio, network, or
credentials are used.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from soundalike.ml.fulltrack_selection import (
    AUTOMATED_PROMOTION_PROHIBITED_NOTICE,
    CANDIDATE_EVALUATION_SCHEMA_VERSION,
    CANDIDATE_LIST_SCHEMA_VERSION,
    DEFAULT_CROSS_SEED_STABILITY_THRESHOLD,
    HUMAN_EVIDENCE_SCHEMA_VERSION,
    JAMENDO_TAG_DESCRIPTIVE_NOTICE,
    MIN_SEEDS_PER_CANDIDATE_FOLD,
    OFFICIAL_FOLDS,
    REASON_CODE_ACCEPTED,
    REASON_CODE_AUTOMATED_GATES_FAILED,
    REASON_CODE_NOT_SUPPLIED,
    REASON_CODE_REJECTED,
    SELECTION_SCHEMA_VERSION,
    FullTrackSelectionError,
    _canonical_sha256,
    _compute_model_bundle_sha256,
    build_and_write_selection_report,
    build_selection_report,
    build_selection_report_from_manifest,
    load_trusted_human_evidence,
    write_selection_inputs,
    write_selection_report,
)


# ---------------------------------------------------------------------------
# Synthetic fixture constants  (SYNTHETIC-TEST-FIXTURE-ONLY)
# ---------------------------------------------------------------------------

def _h(tag: str) -> str:
    return hashlib.sha256(tag.encode()).hexdigest()


# Store binding (StoreBinding.as_dict() + sealed_manifest_sha256):
# schema_version=2 and 11 other fields.
SYNTH_SOURCE_FP = _h("synth-test-source-fingerprint")
SYNTH_MANIFEST = _h("synth-test-manifest")
_SYNTH_STORE_CONFIG = _h("synth-test-store-config-sha256")
_SYNTH_STORE_MODEL = _h("synth-test-store-model-sha256")
_SYNTH_STORE_TRACK_PLAN = _h("synth-test-store-track-plan-sha256")
_SYNTH_STORE_BINDING: Dict[str, Any] = {
    "schema_version": 2,
    "source_fingerprint": SYNTH_SOURCE_FP,
    "config_sha256": _SYNTH_STORE_CONFIG,
    "model_sha256": _SYNTH_STORE_MODEL,
    "model_id": "fixture-model",
    "embedding_dim": 4,
    "track_count": 8,
    "shard_tracks": 3,
    "repetition_sections": 32,
    "salient_sections": 32,
    "track_plan_sha256": _SYNTH_STORE_TRACK_PLAN,
    "sealed_manifest_sha256": SYNTH_MANIFEST,
}
# SYNTH_STORE_SHA = canonical SHA-256 of the store binding (required by strict validation)
SYNTH_STORE_SHA = _canonical_sha256(_SYNTH_STORE_BINDING)

# Training config (TrainingConfig.as_dict() with non_production=True):
_SYNTH_TRAINING_CONFIG: Dict[str, Any] = {
    "max_epochs": 64,
    "patience": 8,
    "min_delta": 1e-5,
    "learning_rate": 0.01,
    "weight_decay": 0.001,
    "margin": 0.05,
    "temperature": 0.2,
    "gradient_clip_norm": 5.0,
    "hard_negatives": 2,
    "random_negatives": 2,
    "maxsim_budget": 8,
    "top_k": 4,
    "coverage_threshold": 0.5,
    "monotonic_hidden_dims": [8],
    "min_train_tracks": 2,
    "min_validation_tracks": 2,
    "max_train_tracks": None,
    "max_validation_tracks": None,
    "device": "cpu",
    "non_production": True,
}
# SYNTH_CONFIG_SHA = canonical SHA-256 of training_config (required by strict validation)
SYNTH_CONFIG_SHA = _canonical_sha256(_SYNTH_TRAINING_CONFIG)

SYNTH_MODEL_JSON = _h("synth-test-model-json-sha256")
SYNTH_TRAIN_DS = _h("synth-test-train-dataset")
SYNTH_VAL_DS = _h("synth-test-val-dataset")
SYNTH_TRAIN_RK = _h("synth-test-train-ranking")
SYNTH_VAL_RK = _h("synth-test-val-ranking")
SYNTH_CKPT = _h("synth-test-checkpoint")
SYNTH_NPZ = _h("synth-test-npz")
SYNTH_VIEW = _h("synth-test-view")
SYNTH_FOLD_QUERY = _h("synth-test-fold-query")

CANDIDATE_KIND = "nonnegative_linear"
FOLDS = list(OFFICIAL_FOLDS)
SEEDS = [17, 29, 43]

# Presentation-order permutations for blinded randomized study.
# G: blinded_label must appear in presentation_order, so we use the same label set.
_BLINDED_LABELS = ["A", "B", "C"]
_ALL_ORDERS_3 = [list(p) for p in itertools.permutations(_BLINDED_LABELS)]
_RATERS = ["rater_alpha", "rater_beta", "rater_gamma", "rater_delta"]


def _artifact_sha(candidate_kind: str, fold: int, seed: int) -> str:
    return _h(f"synth-artifact-{candidate_kind}-fold{fold}-seed{seed}")


def _eval_identity() -> Dict[str, Any]:
    return {"source_fingerprint": SYNTH_SOURCE_FP, "store_binding_sha256": SYNTH_STORE_SHA}


def _eval_identity_sha256() -> str:
    return _canonical_sha256(_eval_identity())


# ---------------------------------------------------------------------------
# Synthetic artifact builders
# ---------------------------------------------------------------------------


def _make_training_report(
    fold: int, seed: int, candidate_kind: str = CANDIDATE_KIND,
    *,
    artifact_sha: Optional[str] = None,
    bad_resources: bool = False,
) -> Dict[str, Any]:
    arti = artifact_sha or _artifact_sha(candidate_kind, fold, seed)
    # C: resources must have exactly {wall_time_seconds, cpu_rss_peak_bytes,
    # cuda_peak_bytes, device}
    res = {
        "wall_time_seconds": float("nan") if bad_resources else 1.0,
        "cpu_rss_peak_bytes": 1000000,
        "cuda_peak_bytes": 0,
        "device": "cpu",
    }
    job_id = f"fold-{fold}__{candidate_kind}__seed-{seed}"
    job_config_sha256 = _canonical_sha256(
        {
            "job_id": job_id,
            "fold_index": fold,
            "candidate_kind": candidate_kind,
            "seed": seed,
            "training_config_sha256": SYNTH_CONFIG_SHA,
            "train_dataset_hash": SYNTH_TRAIN_DS,
            "validation_dataset_hash": SYNTH_VAL_DS,
            "train_ranking_hash": SYNTH_TRAIN_RK,
            "validation_ranking_hash": SYNTH_VAL_RK,
            "store_binding_sha256": SYNTH_STORE_SHA,
            "source_fingerprint": SYNTH_SOURCE_FP,
            "no_tag_supervision": True,
        }
    )
    # C: store_binding must match _SYNTH_STORE_BINDING exactly, and
    # store_binding_sha256 == _canonical_sha256(store_binding).
    # training_config must match _SYNTH_TRAINING_CONFIG exactly, and
    # training_config_sha256 == _canonical_sha256(training_config).
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": "fulltrack_train_report",
        "job_status": "complete",
        "job_id": job_id,
        "fold": fold,
        "seed": seed,
        "candidate_kind": candidate_kind,
        "created_at": "2024-01-01T00:00:00Z",
        "source_fingerprint": SYNTH_SOURCE_FP,
        "store_binding": dict(_SYNTH_STORE_BINDING),
        "store_binding_sha256": SYNTH_STORE_SHA,
        "store_manifest_sha256": SYNTH_MANIFEST,
        "training_config": dict(_SYNTH_TRAINING_CONFIG),
        "training_config_sha256": SYNTH_CONFIG_SHA,
        "job_config_sha256": job_config_sha256,
        "dataset_hashes": {"train": SYNTH_TRAIN_DS, "validation": SYNTH_VAL_DS},
        "ranking_hashes": {"train": SYNTH_TRAIN_RK, "validation": SYNTH_VAL_RK},
        "view_hashes": {"train": [SYNTH_VIEW], "validation": [SYNTH_VIEW]},
        "view_stats": {
            "train": {"no_tag_supervision": True, "track_count": 3},
            "validation": {"no_tag_supervision": True, "track_count": 2},
        },
        "negative_mining": {
            "train": {"example_count": 6},
            "validation": {"example_count": 4},
        },
        "metrics": {
            "train_loss": 0.5,
            "validation_loss": 0.6,
            "train_ranking_accuracy": 0.7,
            "validation_ranking_accuracy": 0.65,
            "train_pairwise_auc": 0.72,
            "validation_pairwise_auc": 0.68,
            "early_stopping_metric": "validation_self_supervised_ranking_loss",
            "epochs_ran": 3,
            "best_epoch": 2,
        },
        "history": [
            {
                "epoch": 1,
                "train_loss": 0.7,
                "validation_loss": 0.8,
                "train_ranking_accuracy": 0.6,
                "validation_ranking_accuracy": 0.55,
                "train_pairwise_auc": 0.62,
                "validation_pairwise_auc": 0.58,
            }
        ],
        "resources": res,
        "model": {
            "artifact_sha256": arti,
            "model_json_sha256": SYNTH_MODEL_JSON,
            "weights_npz_sha256": SYNTH_NPZ,
            "fusion_metadata": {"kind": candidate_kind},
            "parameter_count": 16,
            "model_bytes": 128,
            "runtime_parity_abs_diff": 0.0,
        },
        "checkpoint": {
            "relative_dir": "checkpoint",
            "checkpoint_sha256": SYNTH_CKPT,
            "arrays_npz_sha256": SYNTH_NPZ,
        },
        "notices": [
            "Training is self-supervised from same-track disjoint temporal views only; "
            "fold.track_tags, JamendoTrack.tags, tag Jaccard, ratings, external graphs, "
            "audio decoding, and same-artist positives are not read or used."
        ],
    }
    payload["report_sha256"] = _canonical_sha256(payload)
    return payload


def _make_eval_report(
    fold: int,
    seed: int,
    candidate_kind: str = CANDIDATE_KIND,
    *,
    artifact_sha: Optional[str] = None,
    primary_metric_val: float = 0.50,
    clist_sha: Optional[str] = None,
    eid_override: Optional[Dict[str, Any]] = None,
    bad_paired: bool = False,
    fold_query_sha256_override: Optional[str] = None,
    benchmark_budget: int = 8,
    primary_metric: str = "recall_at_k",
) -> Dict[str, Any]:
    arti = artifact_sha or _artifact_sha(candidate_kind, fold, seed)
    training_report = _make_training_report(
        fold, seed, candidate_kind, artifact_sha=arti
    )
    eid = eid_override or _eval_identity()
    eid_sha = _canonical_sha256(eid)
    cl_sha = clist_sha or _h("placeholder-clist-sha")
    fq_sha = fold_query_sha256_override or SYNTH_FOLD_QUERY
    pmval = primary_metric_val
    metrics = {
        "candidate": {
            "recall_at_k": pmval,
            "mrr": min(pmval + 0.05, 1.0),
            "graded_ndcg_at_k": max(pmval - 0.02, 0.0),
        },
        "global": {
            "recall_at_k": max(pmval - 0.12, 0.0),
            "mrr": max(pmval - 0.10, 0.0),
            "graded_ndcg_at_k": max(pmval - 0.12, 0.0),
        },
        "frozen_hybrid": {
            "recall_at_k": max(pmval - 0.06, 0.0),
            "mrr": max(pmval - 0.05, 0.0),
            "graded_ndcg_at_k": max(pmval - 0.08, 0.0),
        },
    }
    paired_mg = {
        "mean_delta": 0.12,
        "paired_bootstrap_ci95": [0.05, 0.19],
        "bootstrap_probability_delta_gt_zero": 0.97,
    }
    paired_mh = {
        "mean_delta": 0.06,
        "paired_bootstrap_ci95": [0.01, 0.11],
        "bootstrap_probability_delta_gt_zero": 0.89,
    }
    if bad_paired:
        paired_mg = {"incomplete": True}
    payload: Dict[str, Any] = {
        "schema_version": CANDIDATE_EVALUATION_SCHEMA_VERSION,
        "artifact_kind": "fulltrack_trained_candidate_evaluation",
        "candidate_kind": candidate_kind,
        "fold": fold,
        "seed": seed,
        "model_artifact_sha256": arti,
        "model_json_sha256": training_report["model"]["model_json_sha256"],
        "weights_npz_sha256": training_report["model"]["weights_npz_sha256"],
        "training_report_sha256": training_report["report_sha256"],
        "job_config_sha256": training_report["job_config_sha256"],
        "evaluation_identity": eid,
        "evaluation_identity_sha256": eid_sha,
        "fold_query_sha256": fq_sha,
        "candidate_list_sha256": cl_sha,
        "benchmark_budget": benchmark_budget,
        "primary_metric": primary_metric,
        "metrics": metrics,
        "paired_candidate_minus_global": paired_mg,
        "paired_candidate_minus_frozen_hybrid": paired_mh,
        "resources": {"wall_seconds": 1.5, "rss_bytes": 1024000},
        "content_sha256": "placeholder",
    }
    payload["content_sha256"] = _canonical_sha256(
        {k: v for k, v in payload.items() if k != "content_sha256"}
    )
    return payload


def _make_candidate_list(
    candidate_kind: str = CANDIDATE_KIND,
    *,
    training_reports: Optional[List[Dict[str, Any]]] = None,
    stability_threshold: float = DEFAULT_CROSS_SEED_STABILITY_THRESHOLD,
    wrong_bundle_sha: bool = False,
    deciding_budget: int = 8,
    primary_metric: str = "recall_at_k",
) -> Dict[str, Any]:
    if training_reports is None:
        training_reports = [
            _make_training_report(fold, seed, candidate_kind)
            for fold in FOLDS
            for seed in SEEDS
        ]
    bundle_sha = _compute_model_bundle_sha256(candidate_kind, training_reports)
    if wrong_bundle_sha:
        bundle_sha = _h("wrong-bundle-sha")
    payload: Dict[str, Any] = {
        "schema_version": CANDIDATE_LIST_SCHEMA_VERSION,
        "artifact_kind": "fulltrack_selection_candidate_list",
        "list_id": "synthetic-test-fixture-only",
        "evaluation_identity": _eval_identity(),
        "candidates": [{"candidate_kind": candidate_kind, "model_bundle_sha256": bundle_sha}],
        "cross_seed_stability_threshold": stability_threshold,
        "deciding_budget": deciding_budget,
        "primary_metric": primary_metric,
        "content_sha256": "placeholder",
    }
    payload["content_sha256"] = _canonical_sha256(
        {k: v for k, v in payload.items() if k != "content_sha256"}
    )
    return payload


def _make_eval_reports_for_clist(
    clist: Dict[str, Any],
    candidate_kind: str = CANDIDATE_KIND,
    *,
    primary_metric_val: float = 0.50,
    seed_override: Optional[List[int]] = None,
    fold_override: Optional[List[int]] = None,
    vary_by: float = 0.001,
) -> List[Dict[str, Any]]:
    cl_sha = clist["content_sha256"]
    seeds = seed_override if seed_override is not None else SEEDS
    folds = fold_override if fold_override is not None else FOLDS
    reports = []
    for fold in folds:
        for si, seed in enumerate(seeds):
            val = primary_metric_val + vary_by * (si - len(seeds) // 2)
            reports.append(
                _make_eval_report(fold, seed, candidate_kind, primary_metric_val=val, clist_sha=cl_sha)
            )
    return reports


def _standard_fixtures(tmp_path: Path, candidate_kind: str = CANDIDATE_KIND):
    """Return a complete passing scenario (train_paths, eval_paths, clist_path, clist, tr, er)."""
    training_reports = [
        _make_training_report(fold, seed, candidate_kind)
        for fold in FOLDS
        for seed in SEEDS
    ]
    tmp_path.mkdir(parents=True, exist_ok=True)
    clist = _make_candidate_list(candidate_kind, training_reports=training_reports)
    clist_path = tmp_path / "candidates.json"
    clist_path.write_text(json.dumps(clist), encoding="utf-8")
    eval_reports = _make_eval_reports_for_clist(clist, candidate_kind)
    eval_paths = []
    for i, er in enumerate(eval_reports):
        p = tmp_path / f"eval_{i}.json"
        p.write_text(json.dumps(er), encoding="utf-8")
        eval_paths.append(p)
    train_paths = []
    for i, tr in enumerate(training_reports):
        p = tmp_path / f"train_{i}.json"
        p.write_text(json.dumps(tr), encoding="utf-8")
        train_paths.append(p)
    return train_paths, eval_paths, clist_path, clist, training_reports, eval_reports


def _write_json(path: Path, data: Dict[str, Any]) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Human evidence builder
# ---------------------------------------------------------------------------


def _make_human_evidence(
    candidate_kind: str,
    clist: Dict[str, Any],
    training_reports: List[Dict[str, Any]],
    eval_reports: List[Dict[str, Any]],
    *,
    n_raters: int = 3,
    n_seeds: int = 20,
    primary_gain: float = 0.25,
    scene_regression: float = 0.05,
    coherent_top5: float = 0.85,
    unrelated_in_top3: bool = False,
    aggregate_kind: Optional[str] = None,
    self_authored_rater: Optional[str] = None,
    wrong_model_bundle: bool = False,
    wrong_eval_bundle: bool = False,
    wrong_clist_sha: bool = False,
    wrong_eid: bool = False,
    wrong_selected: Optional[str] = None,
    min_raters_override: Optional[int] = None,
    min_seeds_override: Optional[int] = None,
    sparse_rater: Optional[str] = None,
    sparse_coverage_frac: float = 0.2,
) -> Dict[str, Any]:
    from soundalike.ml.fulltrack_selection import (
        _compute_model_bundle_sha256,
        _compute_evaluation_bundle_sha256,
    )
    rater_ids = _RATERS[:n_raters]
    seeds = [f"synth-seed-{i:03d}" for i in range(n_seeds)]
    actual_model_bundle = _compute_model_bundle_sha256(candidate_kind, training_reports)
    actual_eval_bundle = _compute_evaluation_bundle_sha256(candidate_kind, eval_reports)
    cl_sha = clist["content_sha256"]
    eid_sha = _canonical_sha256(_eval_identity())
    bindings = {
        "selected_candidate": wrong_selected or candidate_kind,
        "model_bundle_sha256": _h("wrong") if wrong_model_bundle else actual_model_bundle,
        "evaluation_bundle_sha256": _h("wrong") if wrong_eval_bundle else actual_eval_bundle,
        "candidate_list_sha256": _h("wrong") if wrong_clist_sha else cl_sha,
        "evaluation_identity_sha256": _h("wrong") if wrong_eid else eid_sha,
    }
    protocol = {
        "min_primary_gain_rel": 0.20,
        "max_scene_regression_abs": 0.10,
        "min_coherent_top5_frac": 0.80,
        "zero_unrelated_in_top3": True,
        "blinded_randomized_labels": True,
        "min_independent_raters": min_raters_override if min_raters_override is not None else 3,
        "min_difficult_seeds": min_seeds_override if min_seeds_override is not None else 20,
        "min_rater_seed_coverage": 0.80,
    }
    irds = {}
    for rid in rater_ids:
        is_self = rid == self_authored_rater
        irds[rid] = {
            "is_independent": not is_self,
            "not_self_authored": not is_self,
            "affiliation_declared": "external",
        }
    # Varied presentation orders so blinded_randomized_labels check passes
    # G: blinded_label must appear in presentation_order.
    # Use _BLINDED_LABELS = ["A","B","C"] for both, ensuring consistency.
    # At least 2 distinct orders are guaranteed by cycling through all 6 permutations.
    difficult_seeds = [
        {
            "seed_id": sid,
            "blinded_label": _BLINDED_LABELS[i % len(_BLINDED_LABELS)],
            "presentation_order": _ALL_ORDERS_3[i % len(_ALL_ORDERS_3)],
        }
        for i, sid in enumerate(seeds)
    ]
    raw_ratings = [
        {
            "rater_id": rid,
            "seed_id": sid,
            "primary_gain_relative": primary_gain,
            "scene_regression_max_abs": scene_regression,
            "coherent_top5_frac": coherent_top5,
            "unrelated_in_top3": unrelated_in_top3,
        }
        for rid in rater_ids
        for sid in seeds
    ]
    # Apply sparse rater: keep only first sparse_coverage_frac * n_seeds ratings for that rater
    if sparse_rater is not None:
        n_keep = max(1, int(n_seeds * sparse_coverage_frac))
        kept, sparse_count = [], 0
        for r in raw_ratings:
            if r["rater_id"] == sparse_rater:
                if sparse_count < n_keep:
                    kept.append(r)
                    sparse_count += 1
            else:
                kept.append(r)
        raw_ratings = kept
    payload: Dict[str, Any] = {
        "schema_version": HUMAN_EVIDENCE_SCHEMA_VERSION,
        "artifact_kind": "trusted_fulltrack_human_evidence",
        "aggregate_kind": aggregate_kind,
        "bindings": bindings,
        "protocol": protocol,
        "rater_ids": rater_ids,
        "independent_rater_declarations": irds,
        "difficult_seeds": difficult_seeds,
        "raw_ratings": raw_ratings,
        "content_sha256": "placeholder",
    }
    payload["content_sha256"] = _canonical_sha256(
        {k: v for k, v in payload.items() if k != "content_sha256"}
    )
    return payload


# ===========================================================================
# Tests: automated gates - all passing, promotion_allowed=False (no human)
# ===========================================================================


def test_passing_automated_gates_no_human_evidence_promotion_false(tmp_path):
    """Deterministic: all automated gates pass but no human evidence -> False."""
    train_paths, eval_paths, clist_path, clist, _, _ = _standard_fixtures(tmp_path)
    report = build_selection_report(train_paths, eval_paths, clist_path)
    assert report["promotion_allowed"] is False
    assert report["schema_version"] == SELECTION_SCHEMA_VERSION
    assert report["artifact_kind"] == "fulltrack_selection_report"
    cdet = report["candidate_gate_details"][CANDIDATE_KIND]
    assert cdet["passed"] is True
    assert report["human_decision"]["provided"] is False
    assert report["human_decision"]["reason_code"] == REASON_CODE_NOT_SUPPLIED
    assert AUTOMATED_PROMOTION_PROHIBITED_NOTICE in report["notices"]
    assert JAMENDO_TAG_DESCRIPTIVE_NOTICE in report["notices"]


def test_report_is_deterministic(tmp_path):
    """Two identical calls produce identical report_sha256."""
    train_paths, eval_paths, clist_path, _, _, _ = _standard_fixtures(tmp_path)
    r1 = build_selection_report(train_paths, eval_paths, clist_path)
    r2 = build_selection_report(train_paths, eval_paths, clist_path)
    assert r1["report_sha256"] == r2["report_sha256"]
    assert r1["promotion_allowed"] is False


def test_all_automated_gates_pass(tmp_path):
    """All individual automated gates pass in the standard fixture."""
    train_paths, eval_paths, clist_path, _, _, _ = _standard_fixtures(tmp_path)
    report = build_selection_report(train_paths, eval_paths, clist_path)
    gates = report["candidate_gate_details"][CANDIDATE_KIND]["gates"]
    for gate_name, gate in gates.items():
        assert gate["passed"] is True, f"Gate {gate_name!r} failed: {gate['reason']}"


# ===========================================================================
# Tests: individual automated gate failures
# ===========================================================================


def test_unstable_cross_seed_gate(tmp_path):
    """Cross-seed variance exceeds threshold -> gate fails."""
    training_reports = [_make_training_report(f, s) for f in FOLDS for s in SEEDS]
    clist = _make_candidate_list(training_reports=training_reports)
    clist_path = _write_json(tmp_path / "clist.json", clist)
    cl_sha = clist["content_sha256"]
    eval_reports = []
    for fold in FOLDS:
        for i, seed in enumerate(SEEDS):
            big_val = 0.0 + 0.3 * i
            eval_reports.append(
                _make_eval_report(fold, seed, primary_metric_val=big_val, clist_sha=cl_sha)
            )
    train_paths = [_write_json(tmp_path / f"tr{i}.json", r) for i, r in enumerate(training_reports)]
    eval_paths = [_write_json(tmp_path / f"ev{i}.json", r) for i, r in enumerate(eval_reports)]
    report = build_selection_report(train_paths, eval_paths, clist_path)
    gates = report["candidate_gate_details"][CANDIDATE_KIND]["gates"]
    assert gates["cross_seed_stability"]["passed"] is False
    assert report["candidate_gate_details"][CANDIDATE_KIND]["passed"] is False
    assert report["promotion_allowed"] is False


def test_missing_fold_gate_fails(tmp_path):
    """Missing fold 4 in eval reports -> all_official_folds gate fails."""
    training_reports = [_make_training_report(f, s) for f in FOLDS for s in SEEDS]
    clist = _make_candidate_list(training_reports=training_reports)
    clist_path = _write_json(tmp_path / "clist.json", clist)
    cl_sha = clist["content_sha256"]
    eval_reports = [_make_eval_report(f, s, clist_sha=cl_sha) for f in FOLDS[:-1] for s in SEEDS]
    train_paths = [_write_json(tmp_path / f"tr{i}.json", r) for i, r in enumerate(training_reports)]
    eval_paths = [_write_json(tmp_path / f"ev{i}.json", r) for i, r in enumerate(eval_reports)]
    report = build_selection_report(train_paths, eval_paths, clist_path)
    gates = report["candidate_gate_details"][CANDIDATE_KIND]["gates"]
    assert gates["all_official_folds_present"]["passed"] is False
    assert report["promotion_allowed"] is False


def test_insufficient_seeds_per_fold_gate_fails(tmp_path):
    """Only 2 seeds per fold -> min_seeds_per_fold gate fails."""
    only_2_seeds = SEEDS[:2]
    training_reports = [_make_training_report(f, s) for f in FOLDS for s in only_2_seeds]
    clist = _make_candidate_list(training_reports=training_reports)
    clist_path = _write_json(tmp_path / "clist.json", clist)
    cl_sha = clist["content_sha256"]
    eval_reports = [_make_eval_report(f, s, clist_sha=cl_sha) for f in FOLDS for s in only_2_seeds]
    train_paths = [_write_json(tmp_path / f"tr{i}.json", r) for i, r in enumerate(training_reports)]
    eval_paths = [_write_json(tmp_path / f"ev{i}.json", r) for i, r in enumerate(eval_reports)]
    report = build_selection_report(train_paths, eval_paths, clist_path)
    gates = report["candidate_gate_details"][CANDIDATE_KIND]["gates"]
    assert gates["min_seeds_per_fold"]["passed"] is False
    assert report["promotion_allowed"] is False


def test_model_hash_mismatch_gate_fails(tmp_path):
    """Eval reports use a different artifact_sha256 -> model_hash_match fails."""
    training_reports = [_make_training_report(f, s) for f in FOLDS for s in SEEDS]
    clist = _make_candidate_list(training_reports=training_reports)
    clist_path = _write_json(tmp_path / "clist.json", clist)
    cl_sha = clist["content_sha256"]
    eval_reports = [
        _make_eval_report(f, s, artifact_sha=_h("wrong-model"), clist_sha=cl_sha)
        for f in FOLDS
        for s in SEEDS
    ]
    train_paths = [_write_json(tmp_path / f"tr{i}.json", r) for i, r in enumerate(training_reports)]
    eval_paths = [_write_json(tmp_path / f"ev{i}.json", r) for i, r in enumerate(eval_reports)]
    report = build_selection_report(train_paths, eval_paths, clist_path)
    gates = report["candidate_gate_details"][CANDIDATE_KIND]["gates"]
    assert gates["model_hash_match"]["passed"] is False
    assert report["promotion_allowed"] is False


def test_same_model_artifact_cannot_substitute_different_training_report(tmp_path):
    train_paths, eval_paths, clist_path, _, _, _ = _standard_fixtures(tmp_path)
    substituted = json.loads(train_paths[0].read_text(encoding="utf-8"))
    original_artifact = substituted["model"]["artifact_sha256"]
    substituted["created_at"] = "2024-01-02T00:00:00Z"
    substituted["report_sha256"] = _canonical_sha256(
        {key: value for key, value in substituted.items() if key != "report_sha256"}
    )
    assert substituted["model"]["artifact_sha256"] == original_artifact
    train_paths[0].write_text(json.dumps(substituted), encoding="utf-8")

    report = build_selection_report(train_paths, eval_paths, clist_path)
    gates = report["candidate_gate_details"][CANDIDATE_KIND]["gates"]
    assert gates["model_hash_match"]["passed"] is False
    assert gates["model_bundle_hash_match"]["passed"] is False
    assert report["promotion_allowed"] is False


def test_unrelated_candidate_list_in_eval_rejected(tmp_path):
    """Eval report binds to wrong candidate list -> FullTrackSelectionError."""
    training_reports = [_make_training_report(f, s) for f in FOLDS for s in SEEDS]
    clist = _make_candidate_list(training_reports=training_reports)
    clist_path = _write_json(tmp_path / "clist.json", clist)
    wrong_cl_sha = _h("other-candidate-list-sha")
    er = _make_eval_report(0, 17, clist_sha=wrong_cl_sha)
    train_paths = [_write_json(tmp_path / f"tr{i}.json", r) for i, r in enumerate(training_reports)]
    eval_paths = [_write_json(tmp_path / "ev_bad.json", er)]
    with pytest.raises(FullTrackSelectionError, match="unrelated evaluation"):
        build_selection_report(train_paths, eval_paths, clist_path)


# ===========================================================================
# Tests: malformed / tampered artifacts
# ===========================================================================


def test_duplicate_keys_rejected(tmp_path):
    """JSON with duplicate keys raises FullTrackSelectionError."""
    bad_json = '{"schema_version": 1, "schema_version": 2}'
    bad_path = tmp_path / "bad.json"
    bad_path.write_text(bad_json, encoding="utf-8")
    _, _, clist_path, _, _, _ = _standard_fixtures(tmp_path / "std")
    with pytest.raises(FullTrackSelectionError, match="duplicate key"):
        build_selection_report([bad_path], [bad_path], clist_path)


def test_nonfinite_float_in_eval_report_rejected(tmp_path):
    """JSON NaN in eval report raises FullTrackSelectionError."""
    training_reports = [_make_training_report(f, s) for f in FOLDS for s in SEEDS]
    clist = _make_candidate_list(training_reports=training_reports)
    clist_path = _write_json(tmp_path / "clist.json", clist)
    cl_sha = clist["content_sha256"]
    er = _make_eval_report(0, 17, clist_sha=cl_sha)
    bad_json = json.dumps(er).replace('"recall_at_k": 0.5', '"recall_at_k": NaN')
    bad_path = tmp_path / "bad_eval.json"
    bad_path.write_bytes(bad_json.encode("utf-8"))
    train_paths = [_write_json(tmp_path / f"tr{i}.json", r) for i, r in enumerate(training_reports)]
    with pytest.raises(FullTrackSelectionError, match="non-finite|not valid JSON"):
        build_selection_report(train_paths, [bad_path], clist_path)


def test_tampered_training_report_checksum(tmp_path):
    """Modifying a training report field breaks report_sha256."""
    tr = _make_training_report(0, 17)
    tr_tampered = dict(tr)
    tr_tampered["fold"] = 99
    bad_path = _write_json(tmp_path / "tr_bad.json", tr_tampered)
    training_reports = [_make_training_report(f, s) for f in FOLDS for s in SEEDS]
    clist = _make_candidate_list(training_reports=training_reports)
    clist_path = _write_json(tmp_path / "clist.json", clist)
    with pytest.raises(FullTrackSelectionError, match="report_sha256 mismatch"):
        build_selection_report([bad_path], [], clist_path)


def test_rehashed_training_dataset_forgery_rejected_by_job_binding(tmp_path):
    train_paths, eval_paths, clist_path, _, _, _ = _standard_fixtures(tmp_path)
    forged = json.loads(train_paths[0].read_text(encoding="utf-8"))
    forged["dataset_hashes"]["train"] = _h("forged-train-dataset")
    forged["report_sha256"] = _canonical_sha256(
        {key: value for key, value in forged.items() if key != "report_sha256"}
    )
    train_paths[0].write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(FullTrackSelectionError, match="job config"):
        build_selection_report(train_paths, eval_paths, clist_path)


def test_tampered_candidate_list_checksum(tmp_path):
    """Modifying candidate list breaks content_sha256."""
    training_reports = [_make_training_report(f, s) for f in FOLDS for s in SEEDS]
    clist = _make_candidate_list(training_reports=training_reports)
    clist_tampered = dict(clist)
    clist_tampered["list_id"] = "tampered"
    bad_path = _write_json(tmp_path / "clist_bad.json", clist_tampered)
    with pytest.raises(FullTrackSelectionError, match="content_sha256 mismatch"):
        build_selection_report([], [], bad_path)


def test_tampered_eval_report_checksum(tmp_path):
    """Modifying eval report breaks content_sha256."""
    training_reports = [_make_training_report(f, s) for f in FOLDS for s in SEEDS]
    clist = _make_candidate_list(training_reports=training_reports)
    clist_path = _write_json(tmp_path / "clist.json", clist)
    cl_sha = clist["content_sha256"]
    er = _make_eval_report(0, 17, clist_sha=cl_sha)
    er_tampered = dict(er)
    er_tampered["fold"] = 1
    bad_path = _write_json(tmp_path / "er_bad.json", er_tampered)
    train_paths = [_write_json(tmp_path / f"tr{i}.json", r) for i, r in enumerate(training_reports)]
    with pytest.raises(FullTrackSelectionError, match="content_sha256 mismatch"):
        build_selection_report(train_paths, [bad_path], clist_path)


def test_tampered_human_evidence_checksum(tmp_path):
    """Modifying human evidence breaks content_sha256."""
    _, _, _, clist, training_reports, eval_reports = _standard_fixtures(tmp_path)
    he = _make_human_evidence(CANDIDATE_KIND, clist, training_reports, eval_reports)
    he_tampered = dict(he)
    he_tampered["rater_ids"] = ["rater_alpha", "rater_beta", "rater_gamma", "extra"]
    bad_path = _write_json(tmp_path / "he_bad.json", he_tampered)
    with pytest.raises(FullTrackSelectionError, match="content_sha256 mismatch"):
        load_trusted_human_evidence(bad_path)


def test_wrong_model_bundle_in_candidate_list_fails(tmp_path):
    """Candidate list declares wrong model_bundle_sha256 -> gate fails."""
    training_reports = [_make_training_report(f, s) for f in FOLDS for s in SEEDS]
    clist = _make_candidate_list(training_reports=training_reports, wrong_bundle_sha=True)
    clist_path = _write_json(tmp_path / "clist.json", clist)
    eval_reports = _make_eval_reports_for_clist(clist)
    train_paths = [_write_json(tmp_path / f"tr{i}.json", r) for i, r in enumerate(training_reports)]
    eval_paths = [_write_json(tmp_path / f"ev{i}.json", r) for i, r in enumerate(eval_reports)]
    report = build_selection_report(train_paths, eval_paths, clist_path)
    gates = report["candidate_gate_details"][CANDIDATE_KIND]["gates"]
    assert gates["model_bundle_hash_match"]["passed"] is False
    assert report["promotion_allowed"] is False


# ===========================================================================
# Tests: missing human evidence -> always False
# ===========================================================================


def test_missing_ratings_promotion_false(tmp_path):
    """No trusted_ratings_path -> promotion_allowed always False."""
    train_paths, eval_paths, clist_path, _, _, _ = _standard_fixtures(tmp_path)
    report = build_selection_report(train_paths, eval_paths, clist_path)
    assert report["promotion_allowed"] is False
    assert report["human_decision"]["provided"] is False


def test_zero_count_ratings_coverage_fails(tmp_path):
    """Empty raw_ratings -> per-seed gate fails -> promotion False."""
    train_paths, eval_paths, clist_path, clist, training_reports, eval_reports = _standard_fixtures(tmp_path)
    he = _make_human_evidence(CANDIDATE_KIND, clist, training_reports, eval_reports)
    he["raw_ratings"] = []
    he["content_sha256"] = _canonical_sha256(
        {k: v for k, v in he.items() if k != "content_sha256"}
    )
    he_path = _write_json(tmp_path / "human.json", he)
    report = build_selection_report(train_paths, eval_paths, clist_path, trusted_ratings_path=he_path)
    assert report["promotion_allowed"] is False


# ===========================================================================
# Tests: human evidence rejection cases
# ===========================================================================


def test_aggregate_kind_mismatch_rejected(tmp_path):
    """aggregate_kind='aggregate_ratings_v16' -> rejected with deterministic reason."""
    train_paths, eval_paths, clist_path, clist, training_reports, eval_reports = _standard_fixtures(tmp_path)
    he = _make_human_evidence(
        CANDIDATE_KIND, clist, training_reports, eval_reports,
        aggregate_kind="aggregate_ratings_v16",
    )
    he["content_sha256"] = _canonical_sha256(
        {k: v for k, v in he.items() if k != "content_sha256"}
    )
    he_path = _write_json(tmp_path / "human.json", he)
    report = build_selection_report(train_paths, eval_paths, clist_path, trusted_ratings_path=he_path)
    assert report["promotion_allowed"] is False
    hd = report["human_decision"]
    assert "human evidence rejected" in hd["reason"]
    assert "not recognized" in hd["reason"]
    assert hd["reason_code"] == REASON_CODE_REJECTED


def test_blinded_human_ratings_analysis_recognized_and_passes(tmp_path):
    """aggregate_kind='blinded_human_ratings_analysis' recognized; all gates pass -> True."""
    train_paths, eval_paths, clist_path, clist, training_reports, eval_reports = _standard_fixtures(tmp_path)
    he = _make_human_evidence(
        CANDIDATE_KIND, clist, training_reports, eval_reports,
        aggregate_kind="blinded_human_ratings_analysis",
    )
    he_path = _write_json(tmp_path / "human.json", he)
    report = build_selection_report(train_paths, eval_paths, clist_path, trusted_ratings_path=he_path)
    assert report["promotion_allowed"] is True
    hd = report["human_decision"]
    assert hd.get("aggregate_kind") == "blinded_human_ratings_analysis"
    gates = hd["human_gate_details"]
    assert "aggregate_kind_note" in gates
    assert gates["aggregate_kind_note"]["passed"] is True


def test_self_authored_rater_rejected(tmp_path):
    """Rater declared non-independent -> artifact rejected."""
    train_paths, eval_paths, clist_path, clist, training_reports, eval_reports = _standard_fixtures(tmp_path)
    he = _make_human_evidence(
        CANDIDATE_KIND, clist, training_reports, eval_reports,
        self_authored_rater="rater_alpha",
    )
    he["content_sha256"] = _canonical_sha256(
        {k: v for k, v in he.items() if k != "content_sha256"}
    )
    he_path = _write_json(tmp_path / "human.json", he)
    report = build_selection_report(train_paths, eval_paths, clist_path, trusted_ratings_path=he_path)
    assert report["promotion_allowed"] is False
    reason = report["human_decision"]["reason"]
    assert "self-authored" in reason.lower() or "human evidence rejected" in reason


def test_insufficient_raters_2_rejected(tmp_path):
    """Only 2 raters declared -> human_gate min_independent_raters gate fails."""
    train_paths, eval_paths, clist_path, clist, training_reports, eval_reports = _standard_fixtures(tmp_path)
    he = _make_human_evidence(CANDIDATE_KIND, clist, training_reports, eval_reports, n_raters=2)
    he_path = _write_json(tmp_path / "human.json", he)
    report = build_selection_report(train_paths, eval_paths, clist_path, trusted_ratings_path=he_path)
    assert report["promotion_allowed"] is False


def test_insufficient_seeds_19_human_gate_fails(tmp_path):
    """Only 19 difficult seeds (< 20) -> human gate fails."""
    train_paths, eval_paths, clist_path, clist, training_reports, eval_reports = _standard_fixtures(tmp_path)
    he = _make_human_evidence(CANDIDATE_KIND, clist, training_reports, eval_reports, n_seeds=19)
    he["content_sha256"] = _canonical_sha256(
        {k: v for k, v in he.items() if k != "content_sha256"}
    )
    he_path = _write_json(tmp_path / "human.json", he)
    report = build_selection_report(train_paths, eval_paths, clist_path, trusted_ratings_path=he_path)
    assert report["promotion_allowed"] is False
    hd = report["human_decision"]
    gates = hd.get("human_gate_details", {})
    if gates:
        assert gates["min_difficult_seeds"]["passed"] is False


def test_insufficient_coverage_60pct_fails(tmp_path):
    """Coverage 60% -> rater_seed_coverage gate fails."""
    train_paths, eval_paths, clist_path, clist, training_reports, eval_reports = _standard_fixtures(tmp_path)
    he = _make_human_evidence(CANDIDATE_KIND, clist, training_reports, eval_reports)
    all_ratings = he["raw_ratings"]
    keep = int(len(all_ratings) * 0.6)
    he["raw_ratings"] = all_ratings[:keep]
    he["content_sha256"] = _canonical_sha256(
        {k: v for k, v in he.items() if k != "content_sha256"}
    )
    he_path = _write_json(tmp_path / "human.json", he)
    report = build_selection_report(train_paths, eval_paths, clist_path, trusted_ratings_path=he_path)
    assert report["promotion_allowed"] is False
    hd = report["human_decision"]
    gates = hd.get("human_gate_details", {})
    if gates:
        assert gates["rater_seed_coverage"]["passed"] is False


def test_stale_model_bundle_binding_fails(tmp_path):
    """Human evidence binds to wrong model_bundle_sha256 -> binding gate fails."""
    train_paths, eval_paths, clist_path, clist, training_reports, eval_reports = _standard_fixtures(tmp_path)
    he = _make_human_evidence(
        CANDIDATE_KIND, clist, training_reports, eval_reports, wrong_model_bundle=True
    )
    he_path = _write_json(tmp_path / "human.json", he)
    report = build_selection_report(train_paths, eval_paths, clist_path, trusted_ratings_path=he_path)
    assert report["promotion_allowed"] is False
    hd = report["human_decision"]
    gates = hd.get("human_gate_details", {})
    if gates:
        assert gates["model_bundle_sha256_binding"]["passed"] is False


def test_unrelated_list_in_human_evidence_fails(tmp_path):
    """Human evidence references wrong candidate_list_sha256 -> binding gate fails."""
    train_paths, eval_paths, clist_path, clist, training_reports, eval_reports = _standard_fixtures(tmp_path)
    he = _make_human_evidence(
        CANDIDATE_KIND, clist, training_reports, eval_reports, wrong_clist_sha=True
    )
    he_path = _write_json(tmp_path / "human.json", he)
    report = build_selection_report(train_paths, eval_paths, clist_path, trusted_ratings_path=he_path)
    assert report["promotion_allowed"] is False
    hd = report["human_decision"]
    gates = hd.get("human_gate_details", {})
    if gates:
        assert gates["candidate_list_sha256_binding"]["passed"] is False


def test_primary_gain_below_threshold_fails(tmp_path):
    """primary_gain_relative = 0.10 < 0.20 -> primary_gain_gate fails."""
    train_paths, eval_paths, clist_path, clist, training_reports, eval_reports = _standard_fixtures(tmp_path)
    he = _make_human_evidence(
        CANDIDATE_KIND, clist, training_reports, eval_reports, primary_gain=0.10
    )
    he_path = _write_json(tmp_path / "human.json", he)
    report = build_selection_report(train_paths, eval_paths, clist_path, trusted_ratings_path=he_path)
    assert report["promotion_allowed"] is False
    hd = report["human_decision"]
    gates = hd.get("human_gate_details", {})
    if gates:
        assert gates["primary_gain_gate"]["passed"] is False


def test_scene_regression_above_threshold_fails(tmp_path):
    """scene_regression_max_abs = 0.15 > 0.10 -> scene_regression_gate fails."""
    train_paths, eval_paths, clist_path, clist, training_reports, eval_reports = _standard_fixtures(tmp_path)
    he = _make_human_evidence(
        CANDIDATE_KIND, clist, training_reports, eval_reports, scene_regression=0.15
    )
    he_path = _write_json(tmp_path / "human.json", he)
    report = build_selection_report(train_paths, eval_paths, clist_path, trusted_ratings_path=he_path)
    assert report["promotion_allowed"] is False
    hd = report["human_decision"]
    gates = hd.get("human_gate_details", {})
    if gates:
        assert gates["scene_regression_gate"]["passed"] is False


def test_unrelated_in_top3_fails(tmp_path):
    """unrelated_in_top3=True -> zero_unrelated_in_top3 gate fails."""
    train_paths, eval_paths, clist_path, clist, training_reports, eval_reports = _standard_fixtures(tmp_path)
    he = _make_human_evidence(
        CANDIDATE_KIND, clist, training_reports, eval_reports, unrelated_in_top3=True
    )
    he_path = _write_json(tmp_path / "human.json", he)
    report = build_selection_report(train_paths, eval_paths, clist_path, trusted_ratings_path=he_path)
    assert report["promotion_allowed"] is False
    hd = report["human_decision"]
    gates = hd.get("human_gate_details", {})
    if gates:
        assert gates["zero_unrelated_in_top3"]["passed"] is False


def test_coherent_top5_below_threshold_fails(tmp_path):
    """coherent_top5_frac = 0.70 < 0.80 -> coherent_top5_gate fails."""
    train_paths, eval_paths, clist_path, clist, training_reports, eval_reports = _standard_fixtures(tmp_path)
    he = _make_human_evidence(
        CANDIDATE_KIND, clist, training_reports, eval_reports, coherent_top5=0.70
    )
    he_path = _write_json(tmp_path / "human.json", he)
    report = build_selection_report(train_paths, eval_paths, clist_path, trusted_ratings_path=he_path)
    assert report["promotion_allowed"] is False
    hd = report["human_decision"]
    gates = hd.get("human_gate_details", {})
    if gates:
        assert gates["coherent_top5_gate"]["passed"] is False


def test_wrong_selected_candidate_not_in_list(tmp_path):
    """Human evidence points to nonexistent candidate -> promotion False."""
    train_paths, eval_paths, clist_path, clist, training_reports, eval_reports = _standard_fixtures(tmp_path)
    he = _make_human_evidence(
        CANDIDATE_KIND, clist, training_reports, eval_reports,
        wrong_selected="nonexistent_kind",
    )
    he_path = _write_json(tmp_path / "human.json", he)
    report = build_selection_report(train_paths, eval_paths, clist_path, trusted_ratings_path=he_path)
    assert report["promotion_allowed"] is False
    assert "not in candidate list" in report["human_decision"]["reason"]


def test_automated_gates_failed_human_evidence_insufficient(tmp_path):
    """Even correct human evidence cannot override failed automated gates."""
    training_reports = [_make_training_report(f, s) for f in FOLDS for s in SEEDS]
    clist = _make_candidate_list(training_reports=training_reports)
    clist_path = _write_json(tmp_path / "clist.json", clist)
    cl_sha = clist["content_sha256"]
    pairs = [(f, s) for f in FOLDS for s in SEEDS]
    eval_reports = [
        _make_eval_report(f, s, primary_metric_val=0.3 * (i % 3), clist_sha=cl_sha)
        for i, (f, s) in enumerate(pairs)
    ]
    train_paths = [_write_json(tmp_path / f"tr{i}.json", r) for i, r in enumerate(training_reports)]
    eval_paths = [_write_json(tmp_path / f"ev{i}.json", r) for i, r in enumerate(eval_reports)]
    he = _make_human_evidence(CANDIDATE_KIND, clist, training_reports, eval_reports)
    he_path = _write_json(tmp_path / "human.json", he)
    report = build_selection_report(train_paths, eval_paths, clist_path, trusted_ratings_path=he_path)
    assert report["promotion_allowed"] is False
    assert "automated gates failed" in report["human_decision"]["reason"]
    assert report["human_decision"]["reason_code"] == REASON_CODE_AUTOMATED_GATES_FAILED


# ===========================================================================
# Test: full synthetic passing scenario -> promotion_allowed=True
# ===========================================================================


def test_full_synthetic_passing_human_evidence_promotion_true(tmp_path):
    """
    Fully synthetic exact-bound independent test-fixture artifact.
    All automated and human gates pass -> promotion_allowed=True.
    Raw ratings are NOT copied into the report.
    """
    train_paths, eval_paths, clist_path, clist, training_reports, eval_reports = _standard_fixtures(tmp_path)
    he = _make_human_evidence(
        CANDIDATE_KIND, clist, training_reports, eval_reports,
        n_raters=3, n_seeds=20,
        primary_gain=0.25, scene_regression=0.05,
        coherent_top5=0.85, unrelated_in_top3=False,
        aggregate_kind=None,
    )
    he_path = _write_json(tmp_path / "human.json", he)
    report = build_selection_report(train_paths, eval_paths, clist_path, trusted_ratings_path=he_path)

    assert report["promotion_allowed"] is True
    assert report["schema_version"] == SELECTION_SCHEMA_VERSION
    assert report["artifact_kind"] == "fulltrack_selection_report"

    hd = report["human_decision"]
    assert hd["provided"] is True
    assert hd["promotion_allowed"] is True
    assert hd["selected_candidate"] == CANDIDATE_KIND
    assert hd["reason_code"] == REASON_CODE_ACCEPTED

    gates = hd["human_gate_details"]
    for gate_name, gate in gates.items():
        assert gate["passed"] is True, f"Human gate {gate_name!r} failed: {gate['reason']}"

    cdet = report["candidate_gate_details"][CANDIDATE_KIND]
    assert cdet["passed"] is True

    # Raw ratings must NOT be in the report
    report_text = json.dumps(report)
    assert "rater_alpha" not in report_text
    assert "rater_beta" not in report_text
    assert "synth-seed-000" not in report_text

    # Report sha256 must verify
    payload = {k: v for k, v in report.items() if k != "report_sha256"}
    assert report["report_sha256"] == _canonical_sha256(payload)


# ===========================================================================
# Tests: load_trusted_human_evidence public API
# ===========================================================================


def test_load_trusted_human_evidence_pass(tmp_path):
    """load_trusted_human_evidence succeeds on a valid artifact."""
    _, _, _, clist, training_reports, eval_reports = _standard_fixtures(tmp_path)
    he = _make_human_evidence(CANDIDATE_KIND, clist, training_reports, eval_reports)
    he_path = _write_json(tmp_path / "human.json", he)
    loaded = load_trusted_human_evidence(he_path)
    assert loaded["schema_version"] == HUMAN_EVIDENCE_SCHEMA_VERSION
    assert loaded["artifact_kind"] == "trusted_fulltrack_human_evidence"


def test_load_trusted_human_evidence_expected_bindings_pass(tmp_path):
    """load_trusted_human_evidence with matching expected_bindings succeeds."""
    _, _, _, clist, training_reports, eval_reports = _standard_fixtures(tmp_path)
    he = _make_human_evidence(CANDIDATE_KIND, clist, training_reports, eval_reports)
    he_path = _write_json(tmp_path / "human.json", he)
    loaded = load_trusted_human_evidence(
        he_path,
        expected_bindings={
            "selected_candidate": CANDIDATE_KIND,
            "candidate_list_sha256": clist["content_sha256"],
        },
    )
    assert loaded is not None


def test_load_trusted_human_evidence_expected_bindings_mismatch(tmp_path):
    """load_trusted_human_evidence with wrong expected_bindings raises error."""
    _, _, _, clist, training_reports, eval_reports = _standard_fixtures(tmp_path)
    he = _make_human_evidence(CANDIDATE_KIND, clist, training_reports, eval_reports)
    he_path = _write_json(tmp_path / "human.json", he)
    with pytest.raises(FullTrackSelectionError, match="mismatch"):
        load_trusted_human_evidence(he_path, expected_bindings={"selected_candidate": "wrong_kind"})


# ===========================================================================
# Tests: write_selection_report + checksum
# ===========================================================================


def test_write_selection_report_produces_correct_sha256(tmp_path):
    """Written file SHA-256 matches return value and is deterministic."""
    train_paths, eval_paths, clist_path, _, _, _ = _standard_fixtures(tmp_path)
    report = build_selection_report(train_paths, eval_paths, clist_path)
    out_path = tmp_path / "report.json"
    file_sha = write_selection_report(out_path, report)
    assert out_path.exists()
    actual_sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    assert actual_sha == file_sha
    assert report["report_sha256"] is not None
    out_path2 = tmp_path / "report2.json"
    file_sha2 = write_selection_report(out_path2, report)
    assert file_sha2 == file_sha


def test_write_selection_report_is_valid_json(tmp_path):
    """Written file is valid, parseable JSON with correct report_sha256."""
    train_paths, eval_paths, clist_path, _, _, _ = _standard_fixtures(tmp_path)
    report = build_selection_report(train_paths, eval_paths, clist_path)
    out_path = tmp_path / "report.json"
    write_selection_report(out_path, report)
    loaded = json.loads(out_path.read_text(encoding="utf-8"))
    assert loaded["promotion_allowed"] is False
    assert loaded["report_sha256"] == report["report_sha256"]


# ===========================================================================
# Tests: CLI
# ===========================================================================


def _project_root() -> str:
    import soundalike
    return str(Path(soundalike.__file__).parent.parent.parent)


def test_cli_report_basic(tmp_path):
    """CLI builds a report and writes it; stdout contains valid JSON."""
    train_paths, eval_paths, clist_path, _, _, _ = _standard_fixtures(tmp_path)
    out_path = tmp_path / "cli_report.json"
    cmd = [
        sys.executable, "-m", "soundalike.ml.fulltrack_selection", "report",
        "--candidate-list", str(clist_path),
        "--output", str(out_path),
    ]
    for p in train_paths:
        cmd += ["--training-report", str(p)]
    for p in eval_paths:
        cmd += ["--evaluation-report", str(p)]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=_project_root())
    assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
    out_json = json.loads(result.stdout)
    assert out_json["promotion_allowed"] is False
    assert out_path.exists()


def test_cli_report_with_trusted_ratings_promotion_true(tmp_path):
    """CLI with --trusted-ratings produces promotion_allowed=True."""
    train_paths, eval_paths, clist_path, clist, training_reports, eval_reports = _standard_fixtures(tmp_path)
    he = _make_human_evidence(CANDIDATE_KIND, clist, training_reports, eval_reports)
    he_path = _write_json(tmp_path / "human.json", he)
    out_path = tmp_path / "cli_report2.json"
    cmd = [
        sys.executable, "-m", "soundalike.ml.fulltrack_selection", "report",
        "--candidate-list", str(clist_path),
        "--trusted-ratings", str(he_path),
        "--output", str(out_path),
    ]
    for p in train_paths:
        cmd += ["--training-report", str(p)]
    for p in eval_paths:
        cmd += ["--evaluation-report", str(p)]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=_project_root())
    assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
    out_json = json.loads(result.stdout)
    assert out_json["promotion_allowed"] is True


def test_manifest_drives_selection_and_rejects_listed_file_tampering(tmp_path):
    _, _, _, clist, training_reports, eval_reports = _standard_fixtures(
        tmp_path / "fixtures"
    )
    training_root = tmp_path / "training"
    for report in training_reports:
        path = (
            training_root
            / f"fold-{report['fold']}"
            / str(report["candidate_kind"])
            / f"seed-{report['seed']}"
            / "report.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report), encoding="utf-8")
    selection_dir = tmp_path / "selection"
    manifest = write_selection_inputs(selection_dir, clist, eval_reports)
    manifest_path = selection_dir / "selection-inputs.json"

    report = build_selection_report_from_manifest(training_root, manifest_path)
    assert report["evaluation_report_count"] == 15
    assert report["training_report_count"] == 15
    assert report["promotion_allowed"] is False
    cli_output = tmp_path / "manifest-selection-report.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "soundalike.ml.fulltrack_selection",
            "report-from-manifest",
            "--training-root",
            str(training_root),
            "--manifest",
            str(manifest_path),
            "--output",
            str(cli_output),
        ],
        capture_output=True,
        text=True,
        cwd=_project_root(),
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["promotion_allowed"] is False
    assert cli_output.is_file()

    first_eval = selection_dir / manifest["evaluation_reports"][0]["file"]
    first_eval.write_bytes(first_eval.read_bytes() + b" ")
    with pytest.raises(FullTrackSelectionError, match="file SHA-256"):
        build_selection_report_from_manifest(training_root, manifest_path)


def test_cli_missing_output_exits_nonzero(tmp_path):
    """CLI without --output exits with non-zero code."""
    result = subprocess.run(
        [
            sys.executable, "-m", "soundalike.ml.fulltrack_selection", "report",
            "--training-report", str(tmp_path / "x.json"),
            "--evaluation-report", str(tmp_path / "y.json"),
            "--candidate-list", str(tmp_path / "z.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0


# ===========================================================================
# Tests: misc edge cases
# ===========================================================================


def test_jamendo_tag_notice_non_deciding(tmp_path):
    """JAMENDO_TAG_DESCRIPTIVE_NOTICE appears in report.notices."""
    train_paths, eval_paths, clist_path, _, _, _ = _standard_fixtures(tmp_path)
    report = build_selection_report(train_paths, eval_paths, clist_path)
    assert JAMENDO_TAG_DESCRIPTIVE_NOTICE in report["notices"]


def test_no_current_time_in_report(tmp_path):
    """Report contains no current-time field."""
    train_paths, eval_paths, clist_path, _, _, _ = _standard_fixtures(tmp_path)
    report = build_selection_report(train_paths, eval_paths, clist_path)
    for key in ("created_at", "timestamp", "generated_at", "run_at"):
        assert key not in report, f"Time field {key!r} found in report"


def test_report_sha256_consistency(tmp_path):
    """report_sha256 verifies correctly against the payload."""
    train_paths, eval_paths, clist_path, _, _, _ = _standard_fixtures(tmp_path)
    report = build_selection_report(train_paths, eval_paths, clist_path)
    payload = {k: v for k, v in report.items() if k != "report_sha256"}
    assert report["report_sha256"] == _canonical_sha256(payload)


def test_candidate_details_bundle_hashes_present(tmp_path):
    """Selection report includes computed bundle hashes in candidate details."""
    train_paths, eval_paths, clist_path, _, _, _ = _standard_fixtures(tmp_path)
    report = build_selection_report(train_paths, eval_paths, clist_path)
    cdet = report["candidate_gate_details"][CANDIDATE_KIND]
    assert "computed_model_bundle_sha256" in cdet
    assert "computed_evaluation_bundle_sha256" in cdet
    assert len(cdet["computed_model_bundle_sha256"]) == 64


def test_human_evidence_aggregate_kind_none_works(tmp_path):
    """aggregate_kind=None is accepted and promotion proceeds normally."""
    train_paths, eval_paths, clist_path, clist, training_reports, eval_reports = _standard_fixtures(tmp_path)
    he = _make_human_evidence(
        CANDIDATE_KIND, clist, training_reports, eval_reports, aggregate_kind=None
    )
    he_path = _write_json(tmp_path / "human.json", he)
    report = build_selection_report(train_paths, eval_paths, clist_path, trusted_ratings_path=he_path)
    assert report["promotion_allowed"] is True


def test_human_evidence_eval_bundle_binding_mismatch(tmp_path):
    """Wrong evaluation_bundle_sha256 binding -> human gate fails."""
    train_paths, eval_paths, clist_path, clist, training_reports, eval_reports = _standard_fixtures(tmp_path)
    he = _make_human_evidence(
        CANDIDATE_KIND, clist, training_reports, eval_reports, wrong_eval_bundle=True
    )
    he_path = _write_json(tmp_path / "human.json", he)
    report = build_selection_report(train_paths, eval_paths, clist_path, trusted_ratings_path=he_path)
    assert report["promotion_allowed"] is False
    hd = report["human_decision"]
    gates = hd.get("human_gate_details", {})
    if gates:
        assert gates["evaluation_bundle_sha256_binding"]["passed"] is False


def test_evaluation_identity_sha_binding_mismatch(tmp_path):
    """Wrong evaluation_identity_sha256 binding -> human gate fails."""
    train_paths, eval_paths, clist_path, clist, training_reports, eval_reports = _standard_fixtures(tmp_path)
    he = _make_human_evidence(
        CANDIDATE_KIND, clist, training_reports, eval_reports, wrong_eid=True
    )
    he_path = _write_json(tmp_path / "human.json", he)
    report = build_selection_report(train_paths, eval_paths, clist_path, trusted_ratings_path=he_path)
    assert report["promotion_allowed"] is False
    hd = report["human_decision"]
    gates = hd.get("human_gate_details", {})
    if gates:
        assert gates["evaluation_identity_sha256_binding"]["passed"] is False


# ===========================================================================
# Regression tests (Issues 1, 2, 3, 4, 7, 8, 9, 10)
# ===========================================================================


def test_tampered_mapping_training_report_rejected(tmp_path):
    """Mapping training report with wrong report_sha256 is rejected (Issue 1)."""
    _, _, clist_path, _, _, _ = _standard_fixtures(tmp_path / "std")
    tr = _make_training_report(0, 17)
    tr_bad = dict(tr)
    tr_bad["fold"] = 99  # tamper without recomputing checksum
    with pytest.raises(FullTrackSelectionError, match="report_sha256"):
        build_selection_report([tr_bad], [], clist_path)


def test_tampered_mapping_eval_report_rejected(tmp_path):
    """Mapping eval report with wrong content_sha256 is rejected (Issue 1)."""
    train_paths, _, clist_path, clist, _, _ = _standard_fixtures(tmp_path)
    er = _make_eval_report(0, 17, clist_sha=clist["content_sha256"])
    er_bad = dict(er)
    er_bad["seed"] = 999  # tamper: still valid fold but wrong checksum
    with pytest.raises(FullTrackSelectionError, match="content_sha256"):
        build_selection_report(list(train_paths), [er_bad], clist_path)


def test_eval_identity_differs_from_candidate_list_gate_fails(tmp_path):
    """Eval reports with a different evaluation_identity than clist -> gate fails (Issue 2)."""
    training_reports = [_make_training_report(f, s) for f in FOLDS for s in SEEDS]
    clist = _make_candidate_list(training_reports=training_reports)
    clist_path = _write_json(tmp_path / "clist.json", clist)
    cl_sha = clist["content_sha256"]
    # Different identity that is self-consistent but doesn't match the candidate list
    different_eid = {
        "source_fingerprint": _h("different-source"),
        "store_binding_sha256": _h("different-store"),
    }
    eval_reports = [
        _make_eval_report(f, s, clist_sha=cl_sha, eid_override=different_eid)
        for f in FOLDS
        for s in SEEDS
    ]
    train_paths = [_write_json(tmp_path / f"tr{i}.json", r) for i, r in enumerate(training_reports)]
    eval_paths = [_write_json(tmp_path / f"ev{i}.json", r) for i, r in enumerate(eval_reports)]
    report = build_selection_report(train_paths, eval_paths, clist_path)
    gates = report["candidate_gate_details"][CANDIDATE_KIND]["gates"]
    assert gates["eval_identity_matches_candidate_list"]["passed"] is False
    assert report["promotion_allowed"] is False


def test_fold_query_sha256_drift_gate_fails(tmp_path):
    """Different fold_query_sha256 within same fold -> gate fails (Issue 2)."""
    training_reports = [_make_training_report(f, s) for f in FOLDS for s in SEEDS]
    clist = _make_candidate_list(training_reports=training_reports)
    clist_path = _write_json(tmp_path / "clist.json", clist)
    cl_sha = clist["content_sha256"]
    different_fq = _h("different-fold-query")
    eval_reports = []
    for fold in FOLDS:
        for i, seed in enumerate(SEEDS):
            # First seed in fold 0 gets a different fold_query_sha256 -> drift
            fq = different_fq if (fold == 0 and i == 0) else None
            eval_reports.append(
                _make_eval_report(fold, seed, clist_sha=cl_sha, fold_query_sha256_override=fq)
            )
    train_paths = [_write_json(tmp_path / f"tr{i}.json", r) for i, r in enumerate(training_reports)]
    eval_paths = [_write_json(tmp_path / f"ev{i}.json", r) for i, r in enumerate(eval_reports)]
    report = build_selection_report(train_paths, eval_paths, clist_path)
    gates = report["candidate_gate_details"][CANDIDATE_KIND]["gates"]
    assert gates["fold_query_sha256_aligned"]["passed"] is False
    assert report["promotion_allowed"] is False


def test_one_declared_seed_zero_ratings_fails(tmp_path):
    """A declared seed with zero ratings fails min_raters_per_seed (Issue 3, 6)."""
    train_paths, eval_paths, clist_path, clist, training_reports, eval_reports = _standard_fixtures(tmp_path)
    he = _make_human_evidence(CANDIDATE_KIND, clist, training_reports, eval_reports)
    # Remove ALL ratings for the first declared seed
    first_seed = he["difficult_seeds"][0]["seed_id"]
    he["raw_ratings"] = [r for r in he["raw_ratings"] if r["seed_id"] != first_seed]
    he["content_sha256"] = _canonical_sha256(
        {k: v for k, v in he.items() if k != "content_sha256"}
    )
    he_path = _write_json(tmp_path / "human.json", he)
    report = build_selection_report(train_paths, eval_paths, clist_path, trusted_ratings_path=he_path)
    assert report["promotion_allowed"] is False
    hd = report["human_decision"]
    gates = hd.get("human_gate_details", {})
    assert gates["min_raters_per_seed"]["passed"] is False


def test_one_rater_low_per_rater_coverage_fails(tmp_path):
    """One of four raters at 20% coverage while aggregate is ~80% -> fails (Issue 3)."""
    train_paths, eval_paths, clist_path, clist, training_reports, eval_reports = _standard_fixtures(tmp_path)
    # 4 raters: rater_alpha covers 20% (4/20 seeds), others cover 100% (20/20)
    # Aggregate: (4 + 20 + 20 + 20) / (4 * 20) = 64/80 = 80% (exactly meets threshold)
    # Per-rater: rater_alpha at 20% < 80% -> per_rater_coverage gate fails
    he = _make_human_evidence(
        CANDIDATE_KIND, clist, training_reports, eval_reports,
        n_raters=4,
        sparse_rater="rater_alpha",
        sparse_coverage_frac=0.2,  # 4 out of 20 seeds
    )
    he_path = _write_json(tmp_path / "human.json", he)
    report = build_selection_report(train_paths, eval_paths, clist_path, trusted_ratings_path=he_path)
    assert report["promotion_allowed"] is False
    hd = report["human_decision"]
    gates = hd.get("human_gate_details", {})
    assert gates["per_rater_coverage"]["passed"] is False
    # Aggregate coverage should meet threshold (4+20+20+20)/80 = 0.80
    assert gates["rater_seed_coverage"]["passed"] is True


def test_stricter_declared_min_raters_enforced(tmp_path):
    """Protocol declares min_independent_raters=4 but only 3 raters -> gate fails (Issue 3)."""
    train_paths, eval_paths, clist_path, clist, training_reports, eval_reports = _standard_fixtures(tmp_path)
    # Protocol says 4 required, but only 3 raters provided
    he = _make_human_evidence(
        CANDIDATE_KIND, clist, training_reports, eval_reports,
        n_raters=3,
        min_raters_override=4,
    )
    he_path = _write_json(tmp_path / "human.json", he)
    report = build_selection_report(train_paths, eval_paths, clist_path, trusted_ratings_path=he_path)
    assert report["promotion_allowed"] is False
    hd = report["human_decision"]
    gates = hd.get("human_gate_details", {})
    assert gates["min_independent_raters"]["passed"] is False


def test_stricter_declared_min_seeds_enforced(tmp_path):
    """Protocol declares min_difficult_seeds=25 but only 20 seeds -> gate fails (Issue 3)."""
    train_paths, eval_paths, clist_path, clist, training_reports, eval_reports = _standard_fixtures(tmp_path)
    he = _make_human_evidence(
        CANDIDATE_KIND, clist, training_reports, eval_reports,
        n_seeds=20,
        min_seeds_override=25,
    )
    he_path = _write_json(tmp_path / "human.json", he)
    report = build_selection_report(train_paths, eval_paths, clist_path, trusted_ratings_path=he_path)
    assert report["promotion_allowed"] is False
    hd = report["human_decision"]
    gates = hd.get("human_gate_details", {})
    assert gates["min_difficult_seeds"]["passed"] is False


def test_write_selection_report_rejects_tampered(tmp_path):
    """write_selection_report raises FullTrackSelectionError for tampered report (Issue 7)."""
    train_paths, eval_paths, clist_path, _, _, _ = _standard_fixtures(tmp_path)
    report = build_selection_report(train_paths, eval_paths, clist_path)
    # Tamper: flip promotion_allowed without recomputing report_sha256
    tampered = dict(report)
    tampered["promotion_allowed"] = True
    with pytest.raises(FullTrackSelectionError):
        write_selection_report(tmp_path / "out.json", tampered)


def test_write_selection_report_rejects_rechecksummed_forged_promotion(tmp_path):
    """Passing-looking summaries cannot replace revalidation of source evidence."""
    train_paths, eval_paths, clist_path, _, _, _ = _standard_fixtures(tmp_path)
    forged = build_selection_report(train_paths, eval_paths, clist_path)
    forged["promotion_allowed"] = True
    forged["candidate_gate_details"][CANDIDATE_KIND] = {
        "passed": True,
        "gates": {"fabricated_automated_gate": {"passed": True}},
    }
    forged["human_decision"] = {
        **forged["human_decision"],
        "provided": True,
        "promotion_allowed": True,
        "reason_code": REASON_CODE_ACCEPTED,
        "selected_candidate": CANDIDATE_KIND,
        "human_gate_details": {"fabricated_human_gate": {"passed": True}},
    }
    forged["report_sha256"] = _canonical_sha256(
        {k: v for k, v in forged.items() if k != "report_sha256"}
    )
    with pytest.raises(FullTrackSelectionError, match="rebuilt from source evidence"):
        write_selection_report(tmp_path / "forged.json", forged)


def test_promoted_build_and_write_rejects_preloaded_source_mappings(tmp_path):
    _, _, clist_path, clist, training_reports, eval_reports = _standard_fixtures(
        tmp_path
    )
    human = _make_human_evidence(
        CANDIDATE_KIND, clist, training_reports, eval_reports
    )
    human_path = _write_json(tmp_path / "human.json", human)
    with pytest.raises(FullTrackSelectionError, match="filesystem paths"):
        build_and_write_selection_report(
            tmp_path / "selection.json",
            training_reports,
            eval_reports,
            clist_path,
            trusted_ratings_path=human_path,
        )


def test_atomic_selection_write_translates_filesystem_error(tmp_path, monkeypatch):
    train_paths, eval_paths, clist_path, _, _, _ = _standard_fixtures(tmp_path)
    report = build_selection_report(train_paths, eval_paths, clist_path)

    def blocked_replace(*args, **kwargs):
        raise PermissionError("synthetic write denial")

    monkeypatch.setattr(os, "replace", blocked_replace)
    with pytest.raises(FullTrackSelectionError, match="cannot write selection report"):
        write_selection_report(tmp_path / "blocked.json", report)


def test_human_decision_reason_codes_present(tmp_path):
    """reason_code fields are populated correctly for all cases (Issue 8)."""
    train_paths, eval_paths, clist_path, clist, training_reports, eval_reports = _standard_fixtures(tmp_path)
    # not_supplied
    report_ns = build_selection_report(train_paths, eval_paths, clist_path)
    assert report_ns["human_decision"]["reason_code"] == REASON_CODE_NOT_SUPPLIED
    # rejected (tampered evidence)
    he = _make_human_evidence(CANDIDATE_KIND, clist, training_reports, eval_reports)
    he["aggregate_kind"] = "aggregate_ratings_v16"
    he["content_sha256"] = _canonical_sha256(
        {k: v for k, v in he.items() if k != "content_sha256"}
    )
    he_path = _write_json(tmp_path / "he_bad.json", he)
    report_rj = build_selection_report(train_paths, eval_paths, clist_path, trusted_ratings_path=he_path)
    assert report_rj["human_decision"]["reason_code"] == REASON_CODE_REJECTED
    # accepted
    he_ok = _make_human_evidence(CANDIDATE_KIND, clist, training_reports, eval_reports)
    he_ok_path = _write_json(tmp_path / "he_ok.json", he_ok)
    report_ac = build_selection_report(train_paths, eval_paths, clist_path, trusted_ratings_path=he_ok_path)
    assert report_ac["human_decision"]["reason_code"] == REASON_CODE_ACCEPTED


def test_aggregate_ratings_schema_not_trusted_fulltrack(tmp_path):
    """aggregate_ratings.py output schema is explicitly recognized and rejected with
    a stable binding-gap reason (not generic 'schema fields differ') (Issue 9, req A)."""
    agg_output = {
        "schema_version": 1,
        "aggregate_kind": "blinded_human_ratings_analysis",
        "session_count": 3,
        "complete_result_ratings": 42,
        "complete_list_ratings": 18,
        "sessions": [],
    }
    agg_path = tmp_path / "agg.json"
    agg_path.write_text(json.dumps(agg_output), encoding="utf-8")
    # Must fail with explicit binding-gap reason, NOT generic "schema fields differ"
    with pytest.raises(FullTrackSelectionError, match="cannot authorize promotion"):
        load_trusted_human_evidence(agg_path)


def test_aggregate_ratings_schema_as_trusted_path_rejected_in_report(tmp_path):
    """aggregate_ratings.py output fails closed as trusted_ratings_path (Issue 9)."""
    train_paths, eval_paths, clist_path, _, _, _ = _standard_fixtures(tmp_path)
    agg_output = {
        "schema_version": 1,
        "aggregate_kind": "blinded_human_ratings_analysis",
        "session_count": 3,
        "complete_result_ratings": 42,
        "complete_list_ratings": 18,
        "sessions": [],
    }
    agg_path = tmp_path / "agg.json"
    agg_path.write_text(json.dumps(agg_output), encoding="utf-8")
    report = build_selection_report(
        train_paths, eval_paths, clist_path, trusted_ratings_path=agg_path
    )
    assert report["promotion_allowed"] is False
    assert report["human_decision"]["reason_code"] == REASON_CODE_REJECTED
    # A: Must fail with explicit binding-gap reason, not generic 'schema fields differ'
    assert "cannot authorize promotion" in report["human_decision"]["reason"]


# ===========================================================================
# NEW REGRESSION TESTS (requirements A-H)
# ===========================================================================


# --- A: aggregate_ratings.py zero-count explicit rejection ---

def test_aggregate_ratings_zero_count_fails_closed(tmp_path):
    """A zero-count blinded aggregate fails closed with an explicit zero/no-evidence reason (req A)."""
    agg_output = {
        "schema_version": 1,
        "aggregate_kind": "blinded_human_ratings_analysis",
        "session_count": 0,
        "complete_result_ratings": 0,
        "complete_list_ratings": 0,
        "sessions": [],
    }
    agg_path = tmp_path / "agg_zero.json"
    agg_path.write_text(json.dumps(agg_output), encoding="utf-8")
    # load_trusted_human_evidence must raise with zero/no-evidence wording
    with pytest.raises(FullTrackSelectionError, match="zero sessions|no.*evidence|cannot authorize"):
        load_trusted_human_evidence(agg_path)
    # build_selection_report must also fail closed
    train_paths, eval_paths, clist_path, _, _, _ = _standard_fixtures(tmp_path / "std")
    report = build_selection_report(
        train_paths, eval_paths, clist_path, trusted_ratings_path=agg_path
    )
    assert report["promotion_allowed"] is False
    assert report["human_decision"]["reason_code"] == REASON_CODE_REJECTED
    reason = report["human_decision"]["reason"]
    assert "zero" in reason or "no" in reason or "cannot authorize" in reason


def test_aggregate_ratings_enriched_artifact_kind_passes(tmp_path):
    """A trusted_fulltrack_human_evidence artifact with aggregate_kind=
    blinded_human_ratings_analysis (enriched with full bindings) must still pass (req A)."""
    train_paths, eval_paths, clist_path, clist, training_reports, eval_reports = _standard_fixtures(tmp_path)
    he = _make_human_evidence(
        CANDIDATE_KIND, clist, training_reports, eval_reports,
        aggregate_kind="blinded_human_ratings_analysis",
    )
    he_path = _write_json(tmp_path / "human.json", he)
    report = build_selection_report(train_paths, eval_paths, clist_path, trusted_ratings_path=he_path)
    # Enriched trusted artifact must be allowed to promote
    assert report["promotion_allowed"] is True


# --- B/D: paired summary exact-key rejection ---

def test_paired_summary_extra_key_rejected(tmp_path):
    """An extra key in paired_candidate_minus_global fails schema validation (req B/D)."""
    training_reports = [_make_training_report(f, s) for f in FOLDS for s in SEEDS]
    clist = _make_candidate_list(training_reports=training_reports)
    clist_path = _write_json(tmp_path / "clist.json", clist)
    cl_sha = clist["content_sha256"]
    er = _make_eval_report(0, 17, clist_sha=cl_sha)
    # Inject an extra field into paired summary
    er["paired_candidate_minus_global"]["extra_field"] = "sneaky"
    er["content_sha256"] = _canonical_sha256(
        {k: v for k, v in er.items() if k != "content_sha256"}
    )
    bad_path = _write_json(tmp_path / "er_extra.json", er)
    train_paths = [_write_json(tmp_path / f"tr{i}.json", r) for i, r in enumerate(training_reports)]
    with pytest.raises(FullTrackSelectionError, match="exact fields|extra="):
        build_selection_report(train_paths, [bad_path], clist_path)


# --- C: store_binding hash cross-check ---

def test_store_binding_sha256_mismatch_rejected(tmp_path):
    """A training report where store_binding_sha256 != canonical SHA-256 of store_binding is
    rejected even if report_sha256 was recomputed from the tampered data (req C)."""
    tr = _make_training_report(0, 17)
    # Change store_binding_sha256 to a wrong value and rehash the report
    tr["store_binding_sha256"] = _h("wrong-store-binding-sha256")
    tr.pop("report_sha256")
    tr["report_sha256"] = _canonical_sha256(tr)
    bad_path = _write_json(tmp_path / "tr_bad.json", tr)
    training_reports = [_make_training_report(f, s) for f in FOLDS for s in SEEDS]
    clist = _make_candidate_list(training_reports=training_reports)
    clist_path = _write_json(tmp_path / "clist.json", clist)
    with pytest.raises(FullTrackSelectionError, match="store_binding_sha256"):
        build_selection_report([bad_path], [], clist_path)


def test_training_config_sha256_mismatch_rejected(tmp_path):
    """A training report where training_config_sha256 != canonical SHA-256 of
    training_config is rejected even if top-level report_sha256 is rehashed (req C)."""
    tr = _make_training_report(0, 17)
    # Poison training_config_sha256 and rehash top-level
    tr["training_config_sha256"] = _h("wrong-training-config-sha256")
    tr.pop("report_sha256")
    tr["report_sha256"] = _canonical_sha256(tr)
    bad_path = _write_json(tmp_path / "tr_bad.json", tr)
    training_reports = [_make_training_report(f, s) for f in FOLDS for s in SEEDS]
    clist = _make_candidate_list(training_reports=training_reports)
    clist_path = _write_json(tmp_path / "clist.json", clist)
    with pytest.raises(FullTrackSelectionError, match="training_config_sha256"):
        build_selection_report([bad_path], [], clist_path)


def test_store_binding_wrong_fields_rejected(tmp_path):
    """A training report with malformed store_binding (wrong fields) is rejected (req C)."""
    tr = _make_training_report(0, 17)
    # Replace store_binding with wrong-field object and rehash everything
    bad_sb = {"schema_version": 2, "bad_field": "oops"}
    tr["store_binding"] = bad_sb
    tr["store_binding_sha256"] = _canonical_sha256(bad_sb)
    tr.pop("report_sha256")
    tr["report_sha256"] = _canonical_sha256(tr)
    bad_path = _write_json(tmp_path / "tr_bad.json", tr)
    training_reports = [_make_training_report(f, s) for f in FOLDS for s in SEEDS]
    clist = _make_candidate_list(training_reports=training_reports)
    clist_path = _write_json(tmp_path / "clist.json", clist)
    with pytest.raises(FullTrackSelectionError, match="store_binding"):
        build_selection_report([bad_path], [], clist_path)


def test_training_config_wrong_fields_rejected(tmp_path):
    """A training report with malformed training_config (wrong fields) is rejected (req C)."""
    tr = _make_training_report(0, 17)
    bad_tc = {"non_production": True}  # missing 19 required fields
    tr["training_config"] = bad_tc
    tr["training_config_sha256"] = _canonical_sha256(bad_tc)
    tr.pop("report_sha256")
    tr["report_sha256"] = _canonical_sha256(tr)
    bad_path = _write_json(tmp_path / "tr_bad.json", tr)
    training_reports = [_make_training_report(f, s) for f in FOLDS for s in SEEDS]
    clist = _make_candidate_list(training_reports=training_reports)
    clist_path = _write_json(tmp_path / "clist.json", clist)
    with pytest.raises(FullTrackSelectionError, match="training_config"):
        build_selection_report([bad_path], [], clist_path)


# --- D: evaluation_identity exact-field check ---

def test_eval_identity_extra_field_rejected(tmp_path):
    """Evaluation report with extra field in evaluation_identity is rejected (req D)."""
    training_reports = [_make_training_report(f, s) for f in FOLDS for s in SEEDS]
    clist = _make_candidate_list(training_reports=training_reports)
    clist_path = _write_json(tmp_path / "clist.json", clist)
    cl_sha = clist["content_sha256"]
    eid_extra = {**_eval_identity(), "extra": "bad"}
    er = _make_eval_report(0, 17, clist_sha=cl_sha, eid_override=eid_extra)
    train_paths = [_write_json(tmp_path / f"tr{i}.json", r) for i, r in enumerate(training_reports)]
    bad_path = _write_json(tmp_path / "er_bad.json", er)
    with pytest.raises(FullTrackSelectionError, match="evaluation_identity"):
        build_selection_report(train_paths, [bad_path], clist_path)


# --- D: eval resources exact-field check ---

def test_eval_resources_extra_field_rejected(tmp_path):
    """Evaluation report with extra field in resources is rejected (req D)."""
    training_reports = [_make_training_report(f, s) for f in FOLDS for s in SEEDS]
    clist = _make_candidate_list(training_reports=training_reports)
    clist_path = _write_json(tmp_path / "clist.json", clist)
    cl_sha = clist["content_sha256"]
    er = _make_eval_report(0, 17, clist_sha=cl_sha)
    er["resources"]["extra_field"] = 42
    er["content_sha256"] = _canonical_sha256(
        {k: v for k, v in er.items() if k != "content_sha256"}
    )
    bad_path = _write_json(tmp_path / "er_res.json", er)
    train_paths = [_write_json(tmp_path / f"tr{i}.json", r) for i, r in enumerate(training_reports)]
    with pytest.raises(FullTrackSelectionError, match="resources"):
        build_selection_report(train_paths, [bad_path], clist_path)


# --- D: stability threshold upper bound ---

def test_stability_threshold_too_loose_rejected(tmp_path):
    """Candidate list with stability_threshold > DEFAULT fails validation (req D)."""
    training_reports = [_make_training_report(f, s) for f in FOLDS for s in SEEDS]
    clist = _make_candidate_list(
        training_reports=training_reports,
        stability_threshold=DEFAULT_CROSS_SEED_STABILITY_THRESHOLD * 2.0,
    )
    clist_path = _write_json(tmp_path / "clist.json", clist)
    with pytest.raises(FullTrackSelectionError, match="stability_threshold"):
        build_selection_report([], [], clist_path)


def test_stability_threshold_stricter_allowed(tmp_path):
    """Candidate list with stability_threshold < DEFAULT (stricter) is accepted (req D)."""
    training_reports = [_make_training_report(f, s) for f in FOLDS for s in SEEDS]
    clist = _make_candidate_list(
        training_reports=training_reports,
        stability_threshold=DEFAULT_CROSS_SEED_STABILITY_THRESHOLD / 2.0,
    )
    clist_path = _write_json(tmp_path / "clist.json", clist)
    eval_reports = _make_eval_reports_for_clist(clist)
    train_paths = [_write_json(tmp_path / f"tr{i}.json", r) for i, r in enumerate(training_reports)]
    eval_paths = [_write_json(tmp_path / f"ev{i}.json", r) for i, r in enumerate(eval_reports)]
    # Should not raise (may or may not pass the stability gate, but must load)
    report = build_selection_report(train_paths, eval_paths, clist_path)
    assert "cross_seed_stability" in report["candidate_gate_details"][CANDIDATE_KIND]["gates"]


# --- E: cross-candidate fold_query_sha256 alignment ---

def _make_two_candidate_clist(
    kind_a: str,
    kind_b: str,
    tr_a: List[Dict[str, Any]],
    tr_b: List[Dict[str, Any]],
) -> Dict[str, Any]:
    from soundalike.ml.fulltrack_selection import _compute_model_bundle_sha256
    bundle_a = _compute_model_bundle_sha256(kind_a, tr_a)
    bundle_b = _compute_model_bundle_sha256(kind_b, tr_b)
    payload: Dict[str, Any] = {
        "schema_version": CANDIDATE_LIST_SCHEMA_VERSION,
        "artifact_kind": "fulltrack_selection_candidate_list",
        "list_id": "synthetic-two-candidate-test",
        "evaluation_identity": _eval_identity(),
        "candidates": [
            {"candidate_kind": kind_a, "model_bundle_sha256": bundle_a},
            {"candidate_kind": kind_b, "model_bundle_sha256": bundle_b},
        ],
        "cross_seed_stability_threshold": DEFAULT_CROSS_SEED_STABILITY_THRESHOLD,
        "deciding_budget": 8,
        "primary_metric": "recall_at_k",
        "content_sha256": "placeholder",
    }
    payload["content_sha256"] = _canonical_sha256(
        {k: v for k, v in payload.items() if k != "content_sha256"}
    )
    return payload


def test_cross_candidate_fold_query_drift_gate_fails(tmp_path):
    """When two candidates have different fold_query_sha256 for the same fold,
    fold_query_sha256_cross_candidate_aligned gate fails for both (req E)."""
    kind_a = "nonneg_a"
    kind_b = "nonneg_b"
    tr_a = [_make_training_report(f, s, kind_a) for f in FOLDS for s in SEEDS]
    tr_b = [_make_training_report(f, s, kind_b) for f in FOLDS for s in SEEDS]
    clist = _make_two_candidate_clist(kind_a, kind_b, tr_a, tr_b)
    clist_path = _write_json(tmp_path / "clist.json", clist)
    cl_sha = clist["content_sha256"]
    # kind_a uses the standard fold query for all folds
    ev_a = [_make_eval_report(f, s, kind_a, clist_sha=cl_sha) for f in FOLDS for s in SEEDS]
    # kind_b uses a different fold_query_sha256 for fold 0 only -> cross-candidate conflict
    different_fq = _h("different-fold-query-kind-b")
    ev_b = [
        _make_eval_report(
            f, s, kind_b, clist_sha=cl_sha,
            fold_query_sha256_override=different_fq if f == 0 else None,
        )
        for f in FOLDS for s in SEEDS
    ]
    all_tr = tr_a + tr_b
    all_ev = ev_a + ev_b
    train_paths = [_write_json(tmp_path / f"tr{i}.json", r) for i, r in enumerate(all_tr)]
    eval_paths = [_write_json(tmp_path / f"ev{i}.json", r) for i, r in enumerate(all_ev)]
    report = build_selection_report(train_paths, eval_paths, clist_path)
    # Both candidates must fail the cross-candidate fold_query gate
    for ck in (kind_a, kind_b):
        gates = report["candidate_gate_details"][ck]["gates"]
        assert "fold_query_sha256_cross_candidate_aligned" in gates, f"{ck} missing gate"
        assert gates["fold_query_sha256_cross_candidate_aligned"]["passed"] is False, (
            f"{ck} cross-candidate fold_query gate should fail"
        )
    assert report["promotion_allowed"] is False


def test_two_candidate_consistent_cross_candidate_gates_pass(tmp_path):
    """Two candidates with identical fold_query_sha256 and primary_metric:
    cross-candidate gates pass for both (req E)."""
    kind_a = "nonneg_a"
    kind_b = "nonneg_b"
    tr_a = [_make_training_report(f, s, kind_a) for f in FOLDS for s in SEEDS]
    tr_b = [_make_training_report(f, s, kind_b) for f in FOLDS for s in SEEDS]
    clist = _make_two_candidate_clist(kind_a, kind_b, tr_a, tr_b)
    clist_path = _write_json(tmp_path / "clist.json", clist)
    cl_sha = clist["content_sha256"]
    # Both candidates use the standard fold query (SYNTH_FOLD_QUERY) -> no conflict
    ev_a = [_make_eval_report(f, s, kind_a, clist_sha=cl_sha) for f in FOLDS for s in SEEDS]
    ev_b = [_make_eval_report(f, s, kind_b, clist_sha=cl_sha) for f in FOLDS for s in SEEDS]
    all_tr = tr_a + tr_b
    all_ev = ev_a + ev_b
    train_paths = [_write_json(tmp_path / f"tr{i}.json", r) for i, r in enumerate(all_tr)]
    eval_paths = [_write_json(tmp_path / f"ev{i}.json", r) for i, r in enumerate(all_ev)]
    report = build_selection_report(train_paths, eval_paths, clist_path)
    for ck in (kind_a, kind_b):
        gates = report["candidate_gate_details"][ck]["gates"]
        if "fold_query_sha256_cross_candidate_aligned" in gates:
            assert gates["fold_query_sha256_cross_candidate_aligned"]["passed"] is True
        if "primary_metric_cross_candidate_aligned" in gates:
            assert gates["primary_metric_cross_candidate_aligned"]["passed"] is True


# --- F: cross-seed stability covers all metrics (non-primary instability) ---

def test_cross_seed_non_primary_metric_instability_fails(tmp_path):
    """When mrr is highly unstable across seeds (while recall_at_k is stable),
    the cross_seed_stability gate still fails (req F)."""
    training_reports = [_make_training_report(f, s) for f in FOLDS for s in SEEDS]
    clist = _make_candidate_list(training_reports=training_reports)
    clist_path = _write_json(tmp_path / "clist.json", clist)
    cl_sha = clist["content_sha256"]
    eval_reports = []
    for fold in FOLDS:
        for i, seed in enumerate(SEEDS):
            er = _make_eval_report(fold, seed, primary_metric_val=0.50, clist_sha=cl_sha)
            # Make mrr wildly unstable (0.0, 0.3, 0.6) while recall_at_k stays stable at 0.50
            er["metrics"]["candidate"]["mrr"] = 0.0 + 0.3 * i
            er["content_sha256"] = _canonical_sha256(
                {k: v for k, v in er.items() if k != "content_sha256"}
            )
            eval_reports.append(er)
    train_paths = [_write_json(tmp_path / f"tr{i}.json", r) for i, r in enumerate(training_reports)]
    eval_paths = [_write_json(tmp_path / f"ev{i}.json", r) for i, r in enumerate(eval_reports)]
    report = build_selection_report(train_paths, eval_paths, clist_path)
    gates = report["candidate_gate_details"][CANDIDATE_KIND]["gates"]
    assert gates["cross_seed_stability"]["passed"] is False, (
        "Expected gate to fail due to mrr instability"
    )
    reason = gates["cross_seed_stability"]["reason"]
    assert "mrr" in reason or "unstable" in reason
    # Should report all checked metrics
    assert "checked_metrics" in gates["cross_seed_stability"]
    assert report["promotion_allowed"] is False


def test_cross_seed_stability_checks_all_metrics_list(tmp_path):
    """checked_metrics in gate details includes all three candidate metrics
    and both paired delta fields (req F)."""
    train_paths, eval_paths, clist_path, _, _, _ = _standard_fixtures(tmp_path)
    report = build_selection_report(train_paths, eval_paths, clist_path)
    gates = report["candidate_gate_details"][CANDIDATE_KIND]["gates"]
    checked = gates["cross_seed_stability"]["checked_metrics"]
    for metric in ("recall_at_k", "mrr", "graded_ndcg_at_k"):
        assert metric in checked, f"{metric} not in checked_metrics"
    for paired in ("paired_candidate_minus_global.mean_delta",
                   "paired_candidate_minus_frozen_hybrid.mean_delta"):
        assert paired in checked, f"{paired} not in checked_metrics"


# --- G: blinded_label must be in presentation_order ---

def test_blinded_label_not_in_presentation_order_rejected(tmp_path):
    """A difficult seed whose blinded_label is absent from presentation_order
    causes load_trusted_human_evidence to fail (req G)."""
    _, _, _, clist, training_reports, eval_reports = _standard_fixtures(tmp_path)
    he = _make_human_evidence(CANDIDATE_KIND, clist, training_reports, eval_reports)
    # Corrupt the first seed: blinded_label = "Z" which is not in ["A", "B", "C"]
    he["difficult_seeds"][0]["blinded_label"] = "Z"
    he["content_sha256"] = _canonical_sha256(
        {k: v for k, v in he.items() if k != "content_sha256"}
    )
    he_path = _write_json(tmp_path / "he_bad.json", he)
    with pytest.raises(FullTrackSelectionError, match="blinded_label|not found in presentation_order"):
        load_trusted_human_evidence(he_path)


def test_blinded_label_in_presentation_order_required(tmp_path):
    """All seeds in standard fixture have blinded_label present in presentation_order (req G)."""
    _, _, _, clist, training_reports, eval_reports = _standard_fixtures(tmp_path)
    he = _make_human_evidence(CANDIDATE_KIND, clist, training_reports, eval_reports)
    # Verify the fixture is correct: every seed has its label in the order
    for seed in he["difficult_seeds"]:
        assert seed["blinded_label"] in seed["presentation_order"], (
            f"seed {seed['seed_id']} has blinded_label={seed['blinded_label']!r} "
            f"not in {seed['presentation_order']}"
        )
    he_path = _write_json(tmp_path / "he_ok.json", he)
    # Must load without error
    loaded = load_trusted_human_evidence(he_path)
    assert loaded["artifact_kind"] == "trusted_fulltrack_human_evidence"


# --- H: no raw ratings in output (regression, verify still holds) ---

def test_no_raw_ratings_in_selection_report_output(tmp_path):
    """Private rater IDs and individual seed ratings are never present in the
    selection report output (req H).  Gate details may say computed_from_raw_ratings=True
    as a metadata note, but no actual rating rows or rater identities appear."""
    train_paths, eval_paths, clist_path, clist, training_reports, eval_reports = _standard_fixtures(tmp_path)
    he = _make_human_evidence(CANDIDATE_KIND, clist, training_reports, eval_reports)
    he_path = _write_json(tmp_path / "human.json", he)
    report = build_selection_report(train_paths, eval_paths, clist_path, trusted_ratings_path=he_path)
    report_text = json.dumps(report)
    # Private rater IDs must not appear in the report
    assert "rater_alpha" not in report_text
    assert "rater_beta" not in report_text
    # Individual seed IDs (private identifiers) must not appear
    assert "synth-seed-000" not in report_text
    assert "synth-seed-001" not in report_text
    # The actual rating data array must not appear (only gate metadata may reference the
    # computation method via computed_from_raw_ratings boolean, which is fine)
    assert '"rater_id"' not in report_text
    assert '"seed_id"' not in report_text


# --- H: reason_code is always stable string ---

def test_reason_code_is_stable_string(tmp_path):
    """reason_code values are the module-level string constants (req H)."""
    train_paths, eval_paths, clist_path, _, _, _ = _standard_fixtures(tmp_path)
    report = build_selection_report(train_paths, eval_paths, clist_path)
    rc = report["human_decision"]["reason_code"]
    assert rc == REASON_CODE_NOT_SUPPLIED
    assert isinstance(rc, str)
