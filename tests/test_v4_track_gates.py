"""Focused tests for V4 study track-gate composition and bindings."""
from __future__ import annotations

import json

import pytest

from soundalike.ml import v4_track_gates as track_gates
from soundalike.ml.v4_gates import INSTRUMENTAL, UNKNOWN, VOCAL


@pytest.mark.parametrize(
    ("semantic", "panns", "expected"),
    [
        (VOCAL, VOCAL, VOCAL),
        (INSTRUMENTAL, INSTRUMENTAL, INSTRUMENTAL),
        (VOCAL, INSTRUMENTAL, UNKNOWN),
        (UNKNOWN, VOCAL, UNKNOWN),
        (UNKNOWN, UNKNOWN, UNKNOWN),
    ],
)
def test_conservative_vocal_state_requires_detector_agreement(
    semantic: str, panns: str, expected: str
):
    assert track_gates.conservative_vocal_state(semantic, panns) == expected


def test_conservative_vocal_state_rejects_invalid_state():
    with pytest.raises(track_gates.V4TrackGateError, match="state is invalid"):
        track_gates.conservative_vocal_state("speech", VOCAL)


def test_bound_report_rejects_tampering(tmp_path):
    report = {
        "probe_kind": "expected-kind",
        "thresholds": {"minimum_confidence": 0.8},
    }
    report["content_sha256"] = track_gates._content_sha256(report)
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert track_gates._load_bound_report(path, "expected-kind") == report

    report["thresholds"]["minimum_confidence"] = 0.1
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(track_gates.V4TrackGateError, match="binding failed"):
        track_gates._load_bound_report(path, "expected-kind")
