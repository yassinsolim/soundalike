"""Focused tests for fail-closed V5 multi-segment gates."""
from __future__ import annotations

import pytest

from soundalike.ml import v5_track_gates as gates
from soundalike.ml.v4_gates import INSTRUMENTAL, UNKNOWN, VOCAL


@pytest.mark.parametrize(
    ("semantic", "panns", "language", "expected"),
    [
        (VOCAL, VOCAL, "en", VOCAL),
        (VOCAL, UNKNOWN, "en", VOCAL),
        (UNKNOWN, VOCAL, "es", VOCAL),
        (VOCAL, INSTRUMENTAL, "en", UNKNOWN),
        (INSTRUMENTAL, VOCAL, "en", UNKNOWN),
        (UNKNOWN, UNKNOWN, "en", UNKNOWN),
        (INSTRUMENTAL, INSTRUMENTAL, UNKNOWN, INSTRUMENTAL),
        (VOCAL, VOCAL, UNKNOWN, UNKNOWN),
    ],
)
def test_strict_resolved_vocal_state_fails_closed_on_conflict(
    semantic: str,
    panns: str,
    language: str,
    expected: str,
):
    assert gates.strict_resolved_vocal_state(semantic, panns, language) == expected


def test_strict_resolved_vocal_state_rejects_invalid_language():
    with pytest.raises(gates.V5TrackGateError, match="language"):
        gates.strict_resolved_vocal_state(VOCAL, VOCAL, "")


def test_aggregate_language_probabilities_averages_and_normalizes():
    result = gates.aggregate_language_probabilities(
        [
            {"en": 0.8, "es": 0.2},
            {"en": 0.4, "es": 0.6},
            {"en": 0.6, "es": 0.4},
        ]
    )
    assert result == pytest.approx({"en": 0.6, "es": 0.4})
    assert sum(result.values()) == pytest.approx(1.0)


def test_aggregate_language_probabilities_rejects_invalid_rows():
    with pytest.raises(gates.V5TrackGateError, match="probability"):
        gates.aggregate_language_probabilities([{"en": -0.1}])


@pytest.mark.parametrize(
    ("segments", "aggregate", "expected"),
    [
        (["en", "en", "en"], "en", "en"),
        (["en", "unknown", "en"], "en", UNKNOWN),
        (["en", "es", "en"], "en", UNKNOWN),
        (["en", "en", "en"], "es", UNKNOWN),
    ],
)
def test_stable_language_requires_three_matching_known_decisions(
    segments: list[str],
    aggregate: str,
    expected: str,
):
    assert gates.stable_language(segments, aggregate) == expected


def _gate_row(
    *,
    semantic: str = VOCAL,
    panns: str = VOCAL,
    final_state: str = VOCAL,
    language: str = "en",
) -> dict[str, object]:
    return {
        "semantic_vocal_state": semantic,
        "panns_vocal_state": panns,
        "vocal_state": final_state,
        "language": language,
        "multisegment_audited": True,
        "language_segment_starts": [10.0, 40.0, 70.0],
        "language_segment_decisions": ["en", "en", "en"],
        "language_aggregate_decision": "en",
        "language_confidence": 0.9,
        "language_margin": 0.8,
    }


def test_multisegment_rows_reproduce_strict_final_decisions():
    instrumental = _gate_row(
        semantic=INSTRUMENTAL,
        panns=INSTRUMENTAL,
        final_state=INSTRUMENTAL,
        language=UNKNOWN,
    )
    instrumental.update(
        {
            "multisegment_audited": False,
            "language_segment_starts": [],
            "language_segment_decisions": [],
            "language_aggregate_decision": UNKNOWN,
            "language_confidence": 0.0,
            "language_margin": 0.0,
        }
    )
    gates.validate_multisegment_gate_rows(
        {"1": _gate_row(), "2": instrumental}
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("language_segment_decisions", ["en", "es", "en"]),
        ("language_aggregate_decision", "es"),
        ("panns_vocal_state", INSTRUMENTAL),
        ("vocal_state", UNKNOWN),
        ("language", "es"),
    ],
)
def test_multisegment_rows_reject_inconsistent_final_decisions(
    field: str,
    value: object,
):
    row = _gate_row()
    row[field] = value
    with pytest.raises(gates.V5TrackGateError):
        gates.validate_multisegment_gate_rows({"1": row})


@pytest.mark.parametrize(
    "rows",
    [
        {1: _gate_row()},
        {"1": {**_gate_row(), "semantic_vocal_state": []}},
    ],
)
def test_multisegment_rows_reject_malformed_types(rows: dict[object, object]):
    with pytest.raises(gates.V5TrackGateError):
        gates.validate_multisegment_gate_rows(rows)  # type: ignore[arg-type]
