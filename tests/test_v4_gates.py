"""Focused tests for conservative V4 compatibility gates."""
from __future__ import annotations

import numpy as np
import pytest

from soundalike.ml import v4_gates as gates


def test_voice_probability_uses_all_bound_human_voice_labels():
    labels = [*gates.VOICE_LABELS, "Guitar"]
    values = np.zeros((2, len(labels)), dtype=np.float32)
    values[0, labels.index("Singing")] = 0.8
    values[1, labels.index("Speech")] = 0.6
    assert np.allclose(gates.voice_probability(values, labels), [0.8, 0.6])
    missing_voice = labels[: labels.index("A capella")] + labels[labels.index("A capella") + 1 :]
    with pytest.raises(gates.V4GateError, match="vocabulary drift"):
        gates.voice_probability(
            np.zeros((2, len(missing_voice)), dtype=np.float32),
            missing_voice,
        )


def test_thresholds_leave_ambiguous_scores_unknown():
    thresholds = gates.calibrate_vocal_thresholds(
        [0.7, 0.8, 0.9, 0.95],
        [0.01, 0.02, 0.05, 0.1],
        maximum_false_exclusion_rate=0.1,
    )
    assert gates.classify_vocal(0.01, thresholds) == gates.INSTRUMENTAL
    assert gates.classify_vocal(0.9, thresholds) == gates.VOCAL
    assert gates.classify_vocal(
        (thresholds.instrumental_max + thresholds.vocal_min) / 2.0,
        thresholds,
    ) == gates.UNKNOWN


def test_language_and_compatibility_fail_open_on_unknown():
    known = gates.decide_language(
        {"en": 0.9, "es": 0.05},
        minimum_confidence=0.8,
        minimum_margin=0.5,
    )
    ambiguous = gates.decide_language(
        {"en": 0.5, "es": 0.4},
        minimum_confidence=0.8,
        minimum_margin=0.5,
    )
    assert known.language == "en"
    assert ambiguous.language == gates.UNKNOWN
    assert gates.compatibility_allowed(gates.VOCAL, gates.UNKNOWN, "en", "es")
    assert not gates.compatibility_allowed(gates.VOCAL, gates.INSTRUMENTAL)
    assert not gates.compatibility_allowed(gates.VOCAL, gates.VOCAL, "en", "es")
    assert gates.compatibility_allowed(gates.VOCAL, gates.VOCAL, "en", gates.UNKNOWN)


def test_language_threshold_calibration_preserves_precision_floor():
    rows = [
        {"en": 0.95, "es": 0.05},
        {"es": 0.9, "en": 0.1},
    ] * 10
    thresholds = gates.calibrate_language_thresholds(
        rows,
        ["en", "es"] * 10,
        minimum_selected=10,
    )
    assert thresholds.calibration_accuracy == 1.0
    assert thresholds.calibration_coverage == 1.0


def test_representative_starts_are_bounded_and_deterministic():
    assert gates.representative_starts(10.0) == (0.0,)
    assert gates.representative_starts(100.0) == (12.0, 40.0, 68.0)
