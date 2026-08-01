import copy
import json

import numpy as np
import pytest

from soundalike.ml.fulltrack_v3 import (
    EXPECTED_SELECTION_SHA256,
    FullTrackV3Error,
    SelectivePolicy,
    _resolve_input_directory,
    _validate_policy_document,
    _zscore,
    promotion_gates,
    selective_reranker_scores,
    verify_audit_report,
)
from soundalike.ml.fulltrack_store import stable_json_sha256


def _policy_document():
    return {
        "artifact_kind": "musicfm_selective_gate_nested_leave_one_fold_out",
        "evidence_scope": "full_track_jamendo_research",
        "held_out_test_accessed": False,
        "final_validation_policy": {
            "weight": "0.25",
            "feature": "music_std",
            "direction": "le",
            "threshold": 0.05948563385754824,
        },
    }


def _aggregate(recall, mrr, ndcg, *, ci=(0.01, 0.02)):
    baseline = {
        "recall_at_k": 0.10,
        "mrr": 0.20,
        "graded_ndcg_at_k": 0.15,
    }
    candidate = {
        "recall_at_k": recall,
        "mrr": mrr,
        "graded_ndcg_at_k": ndcg,
    }
    return {
        "clap_hybrid": baseline,
        "selective_reranker": candidate,
        "relative_delta": {
            metric: candidate[metric] / baseline[metric] - 1.0
            for metric in baseline
        },
        "paired_delta": {
            metric: {"paired_bootstrap_ci95": list(ci)}
            for metric in baseline
        },
    }


def test_policy_validation_accepts_only_the_frozen_selective_policy():
    policy = _validate_policy_document(_policy_document())
    assert policy == SelectivePolicy(
        weight=0.25,
        feature="music_std",
        direction="le",
        threshold=0.05948563385754824,
    )
    for field, value in (
        ("weight", "0.10"),
        ("feature", "top10_overlap"),
        ("direction", "ge"),
        ("threshold", 0.06),
    ):
        drifted = copy.deepcopy(_policy_document())
        drifted["final_validation_policy"][field] = value
        with pytest.raises(FullTrackV3Error, match="drift"):
            _validate_policy_document(drifted)


def test_selective_reranker_applies_only_below_frozen_music_std():
    clap = np.asarray([0.2, 0.4, 0.6], dtype=np.float64)
    policy = SelectivePolicy(0.25, "music_std", "le", 0.05948563385754824)
    low_variance = np.asarray([0.50, 0.51, 0.52], dtype=np.float64)
    scores, applied, observed = selective_reranker_scores(
        clap, low_variance, policy
    )
    assert applied is True
    assert observed < policy.threshold
    np.testing.assert_allclose(
        scores, 0.75 * _zscore(clap) + 0.25 * _zscore(low_variance)
    )
    high_variance = np.asarray([0.1, 0.5, 0.9], dtype=np.float64)
    scores, applied, observed = selective_reranker_scores(
        clap, high_variance, policy
    )
    assert applied is False
    assert observed > policy.threshold
    np.testing.assert_array_equal(scores, clap)


def test_promotion_gates_require_primary_gain_confidence_and_safety():
    passing = _aggregate(0.125, 0.20, 0.16)
    folds = [_aggregate(0.12, 0.20, 0.16) for _ in range(5)]
    gates = promotion_gates(folds, passing)
    assert gates["automated_passed"] is True
    assert gates["human_pilot_passed"] is False
    assert gates["promotion_allowed"] is False

    weak_gain = _aggregate(0.119, 0.20, 0.16)
    assert promotion_gates(folds, weak_gain)["automated_passed"] is False
    unsafe_mrr = _aggregate(0.125, 0.19, 0.16)
    assert promotion_gates(folds, unsafe_mrr)["automated_passed"] is False
    regressed_fold = list(folds)
    regressed_fold[0] = _aggregate(0.09, 0.20, 0.16)
    assert promotion_gates(regressed_fold, passing)["automated_passed"] is False


def test_verify_audit_report_rejects_tampering_and_promotion(tmp_path):
    report = {
        "schema_version": 1,
        "artifact_kind": "musicfm_selective_reranker_frozen_test_audit",
        "evidence_scope": "full_track_jamendo_research",
        "promotion_allowed": False,
        "promotion_gates": {"automated_passed": False},
    }
    report["artifact_payload_sha256"] = stable_json_sha256(report)
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert verify_audit_report(path)["promotion_allowed"] is False

    tampered = dict(report)
    tampered["promotion_allowed"] = True
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(FullTrackV3Error, match="checksum"):
        verify_audit_report(path)


def test_frozen_selection_hashes_cover_all_official_folds():
    assert set(EXPECTED_SELECTION_SHA256) == set(range(5))
    assert all(len(value) == 64 for value in EXPECTED_SELECTION_SHA256.values())


def test_preflight_rejects_missing_root_before_audit_lock(tmp_path):
    with pytest.raises(FullTrackV3Error, match="does not exist"):
        _resolve_input_directory(tmp_path / "missing", "state root")
