"""Calibrate conservative vocal and language compatibility gates for V4."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


VOICE_LABELS = (
    "Speech",
    "Male speech, man speaking",
    "Female speech, woman speaking",
    "Child speech, kid speaking",
    "Conversation",
    "Narration, monologue",
    "Singing",
    "Choir",
    "Male singing",
    "Female singing",
    "Child singing",
    "Synthetic singing",
    "Vocal music",
    "A capella",
)
VOCAL = "vocal"
INSTRUMENTAL = "instrumental"
UNKNOWN = "unknown"


class V4GateError(RuntimeError):
    """A V4 gate input, threshold, or detector binding is invalid."""


@dataclass(frozen=True)
class VocalThresholds:
    instrumental_max: float
    vocal_min: float

    def validate(self) -> None:
        if not (
            0.0 <= self.instrumental_max < self.vocal_min <= 1.0
        ):
            raise V4GateError("vocal thresholds must define a non-empty unknown band")


@dataclass(frozen=True)
class LanguageDecision:
    language: str
    confidence: float
    margin: float


@dataclass(frozen=True)
class LanguageThresholds:
    minimum_confidence: float
    minimum_margin: float
    calibration_accuracy: float
    calibration_coverage: float

    def validate(self) -> None:
        if not (
            0.0 < self.minimum_confidence <= 1.0
            and 0.0 <= self.minimum_margin < 1.0
            and 0.0 <= self.calibration_accuracy <= 1.0
            and 0.0 <= self.calibration_coverage <= 1.0
        ):
            raise V4GateError("language thresholds are invalid")


def _scores(values: Sequence[float], label: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if (
        result.ndim != 1
        or not len(result)
        or not np.all(np.isfinite(result))
        or np.any((result < 0.0) | (result > 1.0))
    ):
        raise V4GateError(f"{label} scores are invalid")
    return result


def voice_probability(
    clipwise_output: np.ndarray,
    labels: Sequence[str],
) -> np.ndarray:
    """Return the strongest human speech/singing probability per excerpt."""
    matrix = np.asarray(clipwise_output, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[1] != len(labels)
        or not len(matrix)
        or not np.all(np.isfinite(matrix))
    ):
        raise V4GateError("clipwise detector output is invalid")
    positions = [labels.index(label) for label in VOICE_LABELS if label in labels]
    if len(positions) != len(VOICE_LABELS):
        raise V4GateError("PANNs voice label vocabulary drift")
    return np.max(matrix[:, np.asarray(positions, dtype=np.int64)], axis=1)


def calibrate_vocal_thresholds(
    known_vocal: Sequence[float],
    known_instrumental: Sequence[float],
    *,
    maximum_false_exclusion_rate: float = 0.05,
) -> VocalThresholds:
    """Create high-precision known classes and leave the overlap unknown."""
    vocal = _scores(known_vocal, "known-vocal")
    instrumental = _scores(known_instrumental, "known-instrumental")
    if not 0.0 < maximum_false_exclusion_rate < 0.5:
        raise V4GateError("false-exclusion rate must be between zero and one half")
    instrumental_max = float(
        np.quantile(vocal, maximum_false_exclusion_rate, method="lower")
    )
    vocal_min = float(
        np.quantile(
            instrumental,
            1.0 - maximum_false_exclusion_rate,
            method="higher",
        )
    )
    if instrumental_max >= vocal_min:
        midpoint = (instrumental_max + vocal_min) / 2.0
        instrumental_max = float(np.nextafter(midpoint, 0.0))
        vocal_min = float(np.nextafter(midpoint, 1.0))
    thresholds = VocalThresholds(instrumental_max, vocal_min)
    thresholds.validate()
    return thresholds


def classify_vocal(score: float, thresholds: VocalThresholds) -> str:
    thresholds.validate()
    if not np.isfinite(score) or not 0.0 <= score <= 1.0:
        raise V4GateError("vocal score is invalid")
    if score <= thresholds.instrumental_max:
        return INSTRUMENTAL
    if score >= thresholds.vocal_min:
        return VOCAL
    return UNKNOWN


def decide_language(
    probabilities: Mapping[str, float],
    *,
    minimum_confidence: float,
    minimum_margin: float,
) -> LanguageDecision:
    """Return a language only when both probability and margin are convincing."""
    if (
        not 0.0 < minimum_confidence <= 1.0
        or not 0.0 <= minimum_margin < 1.0
        or not probabilities
    ):
        raise V4GateError("language decision thresholds or probabilities are invalid")
    ordered = sorted(
        ((str(language), float(value)) for language, value in probabilities.items()),
        key=lambda item: (-item[1], item[0]),
    )
    if any(
        not language
        or not np.isfinite(value)
        or not 0.0 <= value <= 1.0
        for language, value in ordered
    ):
        raise V4GateError("language probabilities are invalid")
    best_language, confidence = ordered[0]
    runner_up = ordered[1][1] if len(ordered) > 1 else 0.0
    margin = confidence - runner_up
    language = (
        best_language
        if confidence >= minimum_confidence and margin >= minimum_margin
        else UNKNOWN
    )
    return LanguageDecision(language, confidence, margin)


def calibrate_language_thresholds(
    probability_rows: Sequence[Mapping[str, float]],
    expected_languages: Sequence[str],
    *,
    minimum_accuracy: float = 0.95,
    minimum_selected: int = 20,
    confidence_floor: float = 0.8,
    margin_floor: float = 0.5,
) -> LanguageThresholds:
    """Choose the broadest confidence/margin region meeting a precision floor."""
    if (
        len(probability_rows) != len(expected_languages)
        or len(probability_rows) < minimum_selected
        or not 0.5 < minimum_accuracy <= 1.0
        or minimum_selected <= 0
        or not 0.0 < confidence_floor <= 1.0
        or not 0.0 <= margin_floor < 1.0
    ):
        raise V4GateError("language calibration inputs are invalid")
    observations = []
    for probabilities, expected in zip(probability_rows, expected_languages):
        decision = decide_language(
            probabilities,
            minimum_confidence=np.nextafter(0.0, 1.0),
            minimum_margin=0.0,
        )
        observations.append(
            (
                decision.confidence,
                decision.margin,
                decision.language == expected,
            )
        )
    confidence_grid = sorted(
        {round(value[0], 3) for value in observations}
        | {confidence_floor, 0.9}
    )
    margin_grid = sorted(
        {round(value[1], 3) for value in observations}
        | {margin_floor, 0.75}
    )
    eligible = []
    for confidence in confidence_grid:
        for margin in margin_grid:
            if confidence < confidence_floor or margin < margin_floor:
                continue
            selected = [
                correct
                for observed_confidence, observed_margin, correct in observations
                if observed_confidence >= confidence and observed_margin >= margin
            ]
            if len(selected) < minimum_selected:
                continue
            accuracy = float(np.mean(selected))
            if accuracy >= minimum_accuracy:
                eligible.append(
                    (
                        len(selected),
                        accuracy,
                        -confidence,
                        -margin,
                        confidence,
                        margin,
                    )
                )
    if not eligible:
        raise V4GateError("language detector cannot meet the precision floor")
    selected_count, accuracy, _, _, confidence, margin = max(eligible)
    thresholds = LanguageThresholds(
        minimum_confidence=float(confidence),
        minimum_margin=float(margin),
        calibration_accuracy=float(accuracy),
        calibration_coverage=float(selected_count / len(observations)),
    )
    thresholds.validate()
    return thresholds


def compatibility_allowed(
    query_vocal: str,
    candidate_vocal: str,
    query_language: str = UNKNOWN,
    candidate_language: str = UNKNOWN,
) -> bool:
    """Require a known matching voice class and, for vocals, language."""
    valid_vocal = {VOCAL, INSTRUMENTAL, UNKNOWN}
    if query_vocal not in valid_vocal or candidate_vocal not in valid_vocal:
        raise V4GateError("vocal state is invalid")
    if UNKNOWN in {query_vocal, candidate_vocal} or query_vocal != candidate_vocal:
        return False
    if query_vocal == INSTRUMENTAL:
        return query_language == candidate_language == UNKNOWN
    return query_language != UNKNOWN and query_language == candidate_language


def representative_starts(
    duration_seconds: float,
    *,
    excerpt_seconds: float = 20.0,
) -> tuple[float, ...]:
    if (
        not np.isfinite(duration_seconds)
        or duration_seconds <= 0.0
        or not np.isfinite(excerpt_seconds)
        or excerpt_seconds <= 0.0
    ):
        raise V4GateError("excerpt duration is invalid")
    maximum = max(0.0, duration_seconds - excerpt_seconds)
    return tuple(
        sorted(
            {
                round(maximum * fraction, 3)
                for fraction in (0.15, 0.5, 0.85)
            }
        )
    )


def detector_binding(
    checkpoint_path: Path,
    report: Mapping[str, object],
) -> Mapping[str, object]:
    digest = hashlib.sha256()
    with Path(checkpoint_path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    payload = json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "detector": "PANNs Cnn14 AudioSet",
        "checkpoint_sha256": digest.hexdigest(),
        "voice_labels": list(VOICE_LABELS),
        "calibration_payload_sha256": hashlib.sha256(payload).hexdigest(),
        "unknown_fallback": True,
        "promotion_allowed": False,
    }
