"""Calibrated, artist-disjoint semantic prediction for offline research.

The predictor learns MTG-Jamendo genre, instrument, and mood/theme labels from
frozen CLAP embeddings. It deliberately does not evaluate recommendation
quality, open the official test split, or modify the production ranker.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression

from .fulltrack_store import (
    FullTrackStoreReader,
    sha256_path,
    stable_json_sha256,
)
from .jamendo_fulltrack import JamendoValidationError, _parse_metadata_tracks


MODEL_SCHEMA_VERSION = 1
MODEL_KIND = "calibrated_jamendo_semantic_predictor"
REPORT_SCHEMA_VERSION = 1
REPORT_KIND = "calibrated_jamendo_semantic_predictor_report"
SPARSE_SCHEMA_VERSION = 1
SPARSE_KIND = "sparse_semantic_predictions"
SPARSE_METADATA_KIND = "sparse_semantic_predictions_metadata"
TAXONOMY_VERSION = "mtg-jamendo-split-0-183-v1"
TAXONOMY_COUNTS = {"genre": 87, "instrument": 40, "mood/theme": 56}
EXPECTED_VOCABULARY_SHA256 = (
    "f2439dcaef8e77f5a5158e31376bca598e22b49a1198ba2183e6085b50c16734"
)
CATEGORIES = tuple(TAXONOMY_COUNTS)
OFFICIAL_FOLD = 0
EXPECTED_LABEL_SHA256 = {
    "train": "3b6766ee0a0aea01fa2c2c34769cb3873c0adc04378bba35912edad37077f42b",
    "validation": "b93639bc4f716148e7a41e35e9a1d73058a0b9e65c1cc723b66e9e72726072a7",
}
EXPECTED_STORE_MANIFEST_FILE_SHA256 = (
    "82f593136408a893a71d350af8e3356356e8ea5c041f3c1293abe65183388409"
)
EXPECTED_STORE_BINDING = {
    "schema_version": 2,
    "source_fingerprint": (
        "060f43ed0fa12e5a583e26a7728be14a5334c7daffebe2289f08875e9ec0c709"
    ),
    "config_sha256": (
        "32f29427f8b8c19d809f13c4d062baec18461c3b71d63a40b09aa0788572a0d9"
    ),
    "model_sha256": (
        "8053c9775516af2f4902e1e8281e356cc1bf7a85e8b761908170767b77c3f037"
    ),
    "model_id": "laion_clap_htsat_tiny_music_audioset_630k_nonfusion",
    "embedding_dim": 512,
    "track_count": 55_701,
    "shard_tracks": 256,
    "repetition_sections": 32,
    "salient_sections": 32,
    "track_plan_sha256": (
        "6aaff026be51a7edb48aff80bc993460eaca65e793bff1545851d5511cebb244"
    ),
}
EXPECTED_PRODUCTION_ROWS = 272_853
EXPECTED_PRODUCTION_EMBEDDINGS_SHA256 = (
    "52962644b6a5601df97b27f7a7f16f266826851f5193000c512470ad316bde30"
)
EXPECTED_PRODUCTION_EMBEDDINGS_BUILD_SHA256 = (
    "ab79987944b9228354302f9c98e3f89a301fae0f0eb7b08ddbf35a324147bc43"
)
EXPECTED_PRODUCTION_INDEX_SHA256 = (
    "f3ed57af1b8073f2872eed1e9192dee04d1089c7266fb98a157d1ea194526fb9"
)
EXPECTED_PRODUCTION_TRACK_IDS_SHA256 = (
    "a7d673821cc2117de4a2a13f185cc0273a9a684ce4d271ae07831fccb3d61cc0"
)
TRAIN_PART = "train"
CALIBRATION_PART = "validation"
CALIBRATION_SEED = 20260805
CALIBRATION_FIT_PERCENT = 70
DEFAULT_RIDGE = 10.0
MIN_CALIBRATION_CLASS_COUNT = 8
PROBABILITY_EPSILON = 1e-6
NORM_EPSILON = 1e-12
SENTINEL_TAG_INDEX = np.iinfo(np.uint16).max
DEFAULT_CATEGORY_LIMITS = {"genre": 4, "instrument": 2, "mood/theme": 2}
DEFAULT_PROBABILITY_THRESHOLD = 0.05
DEFAULT_BATCH_SIZE = 4_096
MAX_TARGET_STANDARDIZED_MEAN_ABS = 0.25
MIN_TARGET_STANDARDIZED_SCALE_MEAN = 0.8
MAX_TARGET_STANDARDIZED_SCALE_MEAN = 1.2
MODEL_ARRAYS = {
    "calibration_supported",
    "calibrator_intercepts",
    "calibrator_slopes",
    "categories",
    "coefficients",
    "idf",
    "input_mean",
    "input_scale",
    "prior",
    "vocabulary",
}
SPARSE_ARRAYS = {
    "counts",
    "probabilities",
    "slot_categories",
    "tag_indices",
    "taxonomy_version",
    "track_ids",
    "vocabulary",
}


class SemanticPredictorError(RuntimeError):
    """Invalid data, leakage, model drift, or unsafe artifact handling."""


def _payload_sha256(document: Mapping[str, object]) -> str:
    return stable_json_sha256(
        {key: value for key, value in document.items() if key != "payload_sha256"}
    )


def _category(tag: str) -> str:
    prefix, separator, label = str(tag).partition("---")
    if separator != "---" or prefix not in CATEGORIES or not label:
        raise SemanticPredictorError(f"invalid semantic tag: {tag!r}")
    return prefix


def _stable_sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _logit(probability: float) -> float:
    value = float(np.clip(probability, PROBABILITY_EPSILON, 1.0 - PROBABILITY_EPSILON))
    return math.log(value / (1.0 - value))


@dataclass(frozen=True)
class LabelSplit:
    """Ordered labels from one explicitly allowed Jamendo split file."""

    part: str
    source_path: Path
    track_ids: np.ndarray
    artist_ids: np.ndarray
    tags: Tuple[Tuple[str, ...], ...]

    def validate(self) -> None:
        count = len(self.track_ids)
        if (
            self.part not in (TRAIN_PART, CALIBRATION_PART)
            or self.track_ids.shape != (count,)
            or self.artist_ids.shape != (count,)
            or len(self.tags) != count
            or not count
            or len(np.unique(self.track_ids)) != count
            or np.any(self.track_ids <= 0)
            or np.any(self.artist_ids <= 0)
            or any(
                not row
                or len(row) != len(set(row))
                or tuple(sorted(row)) != row
                or any(_category(tag) not in CATEGORIES for tag in row)
                for row in self.tags
            )
        ):
            raise SemanticPredictorError("semantic label split is invalid")

    def subset(self, selected: np.ndarray) -> "LabelSplit":
        mask = np.asarray(selected, dtype=bool)
        if mask.shape != self.track_ids.shape or not np.any(mask):
            raise SemanticPredictorError("semantic label subset is empty or misaligned")
        rows = np.flatnonzero(mask)
        result = LabelSplit(
            part=self.part,
            source_path=self.source_path,
            track_ids=self.track_ids[rows].copy(),
            artist_ids=self.artist_ids[rows].copy(),
            tags=tuple(self.tags[int(row)] for row in rows),
        )
        result.validate()
        return result


def load_label_split(
    metadata_root: Path,
    *,
    fold: int = OFFICIAL_FOLD,
    part: str,
) -> LabelSplit:
    """Load train or validation labels while making test access impossible."""
    if fold != OFFICIAL_FOLD:
        raise SemanticPredictorError(
            "only fold 0 is allowed; other folds can expose fold-0 test labels"
        )
    if part not in (TRAIN_PART, CALIBRATION_PART):
        raise SemanticPredictorError("only train and validation labels may be opened")
    path = (
        Path(metadata_root).absolute()
        / "data"
        / "splits"
        / f"split-{fold}"
        / f"autotagging-{part}.tsv"
    )
    if path.is_symlink():
        raise SemanticPredictorError("semantic label source may not be a symlink")
    try:
        path = path.resolve(strict=True)
    except OSError as exc:
        raise SemanticPredictorError(f"semantic label source is unavailable: {path}") from exc
    if sha256_path(path) != EXPECTED_LABEL_SHA256[part]:
        raise SemanticPredictorError(f"official fold-0 {part} label hash drift")
    try:
        rows = _parse_metadata_tracks(path)
    except JamendoValidationError as exc:
        raise SemanticPredictorError(str(exc)) from exc
    ordered = tuple(sorted(rows.items()))
    split = LabelSplit(
        part=part,
        source_path=path,
        track_ids=np.asarray([track_id for track_id, _ in ordered], dtype=np.int64),
        artist_ids=np.asarray(
            [int(row["artist_id"]) for _, row in ordered], dtype=np.int64
        ),
        tags=tuple(tuple(row["tags"]) for _, row in ordered),
    )
    split.validate()
    return split


def calibration_partition(artist_id: int) -> str:
    """Deterministically split validation artists into fit and audit groups."""
    if isinstance(artist_id, bool) or not isinstance(artist_id, int) or artist_id <= 0:
        raise SemanticPredictorError("artist ID must be a positive integer")
    bucket = (
        int(
            stable_json_sha256(
                {"seed": CALIBRATION_SEED, "artist_id": artist_id}
            )[:16],
            16,
        )
        % 100
    )
    return "fit" if bucket < CALIBRATION_FIT_PERCENT else "audit"


def split_calibration_labels(validation: LabelSplit) -> Tuple[LabelSplit, LabelSplit]:
    validation.validate()
    fit_mask = np.asarray(
        [calibration_partition(int(artist)) == "fit" for artist in validation.artist_ids],
        dtype=bool,
    )
    fit = validation.subset(fit_mask)
    audit = validation.subset(~fit_mask)
    if set(int(value) for value in fit.artist_ids).intersection(
        int(value) for value in audit.artist_ids
    ):
        raise SemanticPredictorError("calibration fit/audit artist leakage")
    return fit, audit


def build_targets(
    split: LabelSplit,
    vocabulary: Optional[Sequence[str]] = None,
) -> Tuple[Tuple[str, ...], np.ndarray]:
    split.validate()
    selected = (
        tuple(sorted({tag for row in split.tags for tag in row}))
        if vocabulary is None
        else tuple(vocabulary)
    )
    if (
        not selected
        or tuple(sorted(selected)) != selected
        or len(set(selected)) != len(selected)
    ):
        raise SemanticPredictorError("semantic vocabulary is invalid")
    positions = {tag: index for index, tag in enumerate(selected)}
    targets = np.zeros((len(split.track_ids), len(selected)), dtype=np.float32)
    for row_index, tags in enumerate(split.tags):
        for tag in tags:
            position = positions.get(tag)
            if position is not None:
                targets[row_index, position] = 1.0
    if vocabulary is None and np.any(np.sum(targets, axis=0) <= 0.0):
        raise SemanticPredictorError("training vocabulary contains an empty tag")
    return selected, targets


def _transform(
    values: np.ndarray,
    mean: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if (
        matrix.ndim != 2
        or not len(matrix)
        or matrix.shape[1:] != mean.shape
        or scale.shape != mean.shape
        or not np.all(np.isfinite(matrix))
        or not np.all(np.isfinite(mean))
        or not np.all(np.isfinite(scale))
        or np.any(scale <= 0.0)
    ):
        raise SemanticPredictorError("semantic input matrix or statistics are invalid")
    standardized = (matrix - mean) / scale
    norms = np.linalg.norm(standardized, axis=1, keepdims=True)
    if np.any(norms <= NORM_EPSILON):
        raise SemanticPredictorError("semantic input has zero normalized norm")
    return standardized / norms


def _fit_transform(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not len(matrix) or not np.all(np.isfinite(matrix)):
        raise SemanticPredictorError("semantic training inputs must be a finite matrix")
    mean = np.mean(matrix, axis=0)
    standard_deviation = np.std(matrix, axis=0)
    scale = np.where(standard_deviation > 1e-6, standard_deviation, 1.0)
    return _transform(matrix, mean, scale), mean, scale


@dataclass(frozen=True)
class SparseSemanticPredictions:
    track_ids: np.ndarray
    tag_indices: np.ndarray
    probabilities: np.ndarray
    counts: np.ndarray
    slot_categories: Tuple[str, ...]

    def validate(self, vocabulary_size: int) -> None:
        rows = len(self.track_ids)
        slots = len(self.slot_categories)
        valid = self.tag_indices != SENTINEL_TAG_INDEX
        if (
            vocabulary_size <= 0
            or self.track_ids.shape != (rows,)
            or self.tag_indices.shape != (rows, slots)
            or self.probabilities.shape != (rows, slots)
            or self.counts.shape != (rows,)
            or self.tag_indices.dtype != np.uint16
            or self.probabilities.dtype != np.float16
            or self.counts.dtype != np.uint8
            or len(np.unique(self.track_ids)) != rows
            or any(category not in CATEGORIES for category in self.slot_categories)
            or np.any(self.counts != np.sum(valid, axis=1))
            or np.any(self.tag_indices[valid] >= vocabulary_size)
            or np.any(self.probabilities[~valid] != 0.0)
            or np.any(self.probabilities[valid] <= 0.0)
            or np.any(self.probabilities[valid] > 1.0)
        ):
            raise SemanticPredictorError("sparse semantic predictions are invalid")


@dataclass(frozen=True)
class CalibratedSemanticPredictor:
    taxonomy_version: str
    ridge: float
    vocabulary: Tuple[str, ...]
    categories: Tuple[str, ...]
    input_mean: np.ndarray
    input_scale: np.ndarray
    coefficients: np.ndarray
    prior: np.ndarray
    idf: np.ndarray
    calibrator_slopes: np.ndarray
    calibrator_intercepts: np.ndarray
    calibration_supported: np.ndarray

    def validate(self) -> None:
        dimension = self.input_mean.shape
        tag_count = len(self.vocabulary)
        vectors = (
            self.prior,
            self.idf,
            self.calibrator_slopes,
            self.calibrator_intercepts,
            self.calibration_supported,
        )
        if (
            self.taxonomy_version != TAXONOMY_VERSION
            or isinstance(self.ridge, bool)
            or not math.isfinite(self.ridge)
            or self.ridge <= 0.0
            or self.input_mean.ndim != 1
            or self.input_scale.shape != dimension
            or self.coefficients.shape != (dimension[0], tag_count)
            or any(vector.shape != (tag_count,) for vector in vectors)
            or self.categories != tuple(_category(tag) for tag in self.vocabulary)
            or tuple(sorted(self.vocabulary)) != self.vocabulary
            or len(set(self.vocabulary)) != tag_count
            or not tag_count
            or np.any(self.input_scale <= 0.0)
            or np.any(self.idf <= 0.0)
            or np.any(self.prior <= 0.0)
            or np.any(self.prior >= 1.0)
            or np.any(self.calibrator_slopes < 0.0)
            or self.calibration_supported.dtype != np.bool_
        ):
            raise SemanticPredictorError("semantic predictor shape or schema drift")
        for values in (
            self.input_mean,
            self.input_scale,
            self.coefficients,
            self.prior,
            self.idf,
            self.calibrator_slopes,
            self.calibrator_intercepts,
        ):
            if not np.all(np.isfinite(values)):
                raise SemanticPredictorError("semantic predictor contains non-finite values")

    def raw_scores(self, values: np.ndarray) -> np.ndarray:
        self.validate()
        return (
            _transform(values, self.input_mean, self.input_scale) @ self.coefficients
            + self.prior
        )

    def predict_proba(self, values: np.ndarray) -> np.ndarray:
        logits = (
            self.raw_scores(values) * self.calibrator_slopes
            + self.calibrator_intercepts
        )
        return np.clip(
            _stable_sigmoid(logits),
            PROBABILITY_EPSILON,
            1.0 - PROBABILITY_EPSILON,
        ).astype(np.float32)

    def semantic_profiles(self, values: np.ndarray) -> np.ndarray:
        weighted = self.predict_proba(values).astype(np.float64) * self.idf
        norms = np.linalg.norm(weighted, axis=1, keepdims=True)
        if np.any(norms <= NORM_EPSILON):
            raise SemanticPredictorError("semantic predictor produced an empty profile")
        return (weighted / norms).astype(np.float32)

    def sparse(
        self,
        values: np.ndarray,
        *,
        category_limits: Mapping[str, int] = DEFAULT_CATEGORY_LIMITS,
        probability_threshold: float = DEFAULT_PROBABILITY_THRESHOLD,
        track_ids: Optional[np.ndarray] = None,
    ) -> SparseSemanticPredictions:
        self.validate()
        limits = _validate_category_limits(category_limits)
        if (
            isinstance(probability_threshold, bool)
            or not math.isfinite(probability_threshold)
            or not 0.0 <= probability_threshold <= 1.0
        ):
            raise SemanticPredictorError("semantic probability threshold is invalid")
        probabilities = self.predict_proba(values)
        row_count = len(probabilities)
        slots = sum(limits.values())
        indices = np.full(
            (row_count, slots), SENTINEL_TAG_INDEX, dtype=np.uint16
        )
        sparse_probabilities = np.zeros((row_count, slots), dtype=np.float16)
        slot_categories = tuple(
            category
            for category in CATEGORIES
            for _ in range(limits[category])
        )
        slot_start = 0
        category_array = np.asarray(self.categories)
        for category in CATEGORIES:
            limit = limits[category]
            positions = np.flatnonzero(category_array == category)
            if limit > len(positions):
                raise SemanticPredictorError(
                    f"semantic quota exceeds {category} vocabulary"
                )
            selected_scores = probabilities[:, positions]
            local = np.argpartition(
                -selected_scores, kth=limit - 1, axis=1
            )[:, :limit]
            top_scores = np.take_along_axis(selected_scores, local, axis=1)
            top_positions = positions[local]
            order = np.argsort(-top_scores, axis=1, kind="stable")
            top_scores = np.take_along_axis(top_scores, order, axis=1)
            top_positions = np.take_along_axis(top_positions, order, axis=1)
            accepted = top_scores >= probability_threshold
            destination = slice(slot_start, slot_start + limit)
            indices[:, destination] = np.where(
                accepted, top_positions, SENTINEL_TAG_INDEX
            ).astype(np.uint16)
            sparse_probabilities[:, destination] = np.where(
                accepted, top_scores, 0.0
            ).astype(np.float16)
            slot_start += limit
        ids = (
            np.arange(row_count, dtype=np.int64)
            if track_ids is None
            else np.asarray(track_ids, dtype=np.int64)
        )
        result = SparseSemanticPredictions(
            track_ids=ids,
            tag_indices=indices,
            probabilities=sparse_probabilities,
            counts=np.sum(indices != SENTINEL_TAG_INDEX, axis=1).astype(np.uint8),
            slot_categories=slot_categories,
        )
        result.validate(len(self.vocabulary))
        return result


def _validate_category_limits(limits: Mapping[str, int]) -> Dict[str, int]:
    if not isinstance(limits, Mapping) or set(limits) != set(CATEGORIES):
        raise SemanticPredictorError("semantic category limits must cover the taxonomy")
    result = {}
    for category in CATEGORIES:
        value = limits[category]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise SemanticPredictorError("semantic category limits must be positive")
        result[category] = value
    if sum(result.values()) > np.iinfo(np.uint8).max:
        raise SemanticPredictorError("semantic category limits exceed sparse count range")
    return result


def fit_calibrated_predictor(
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    calibration_inputs: np.ndarray,
    calibration_targets: np.ndarray,
    vocabulary: Sequence[str],
    *,
    ridge: float = DEFAULT_RIDGE,
) -> CalibratedSemanticPredictor:
    if isinstance(ridge, bool) or not math.isfinite(ridge) or ridge <= 0.0:
        raise SemanticPredictorError("ridge must be a positive finite number")
    transformed, mean, scale = _fit_transform(train_inputs)
    targets = np.asarray(train_targets, dtype=np.float64)
    calibration = np.asarray(calibration_targets, dtype=np.float64)
    vocabulary_tuple = tuple(vocabulary)
    if (
        targets.shape != (len(transformed), len(vocabulary_tuple))
        or calibration.shape
        != (len(np.asarray(calibration_inputs)), len(vocabulary_tuple))
        or not len(vocabulary_tuple)
        or tuple(sorted(vocabulary_tuple)) != vocabulary_tuple
        or len(set(vocabulary_tuple)) != len(vocabulary_tuple)
        or not np.all(np.isfinite(targets))
        or not np.all(np.isfinite(calibration))
        or np.any((targets < 0.0) | (targets > 1.0))
        or np.any((calibration < 0.0) | (calibration > 1.0))
        or np.any(np.sum(targets, axis=0) <= 0.0)
    ):
        raise SemanticPredictorError("semantic predictor targets are invalid")
    prior = np.clip(
        (np.sum(targets, axis=0) + 1.0) / (len(targets) + 2.0),
        PROBABILITY_EPSILON,
        1.0 - PROBABILITY_EPSILON,
    )
    gram = transformed.T @ transformed
    gram.flat[:: gram.shape[0] + 1] += float(ridge)
    coefficients = np.linalg.solve(
        gram,
        transformed.T @ (targets - prior),
    )
    calibration_transformed = _transform(
        calibration_inputs,
        mean,
        scale,
    )
    calibration_scores = calibration_transformed @ coefficients + prior
    slopes = np.zeros(len(vocabulary_tuple), dtype=np.float64)
    intercepts = np.empty(len(vocabulary_tuple), dtype=np.float64)
    supported = np.zeros(len(vocabulary_tuple), dtype=bool)
    for index in range(len(vocabulary_tuple)):
        labels = calibration[:, index]
        positives = int(np.count_nonzero(labels))
        negatives = len(labels) - positives
        smoothed_prevalence = (positives + 1.0) / (len(labels) + 2.0)
        intercepts[index] = _logit(smoothed_prevalence)
        scores = calibration_scores[:, index]
        if (
            positives < MIN_CALIBRATION_CLASS_COUNT
            or negatives < MIN_CALIBRATION_CLASS_COUNT
            or float(np.std(scores)) <= 1e-12
        ):
            continue
        calibrator = LogisticRegression(
            C=100.0,
            fit_intercept=True,
            max_iter=1_000,
            random_state=CALIBRATION_SEED,
            solver="lbfgs",
        )
        calibrator.fit(scores[:, None], labels.astype(np.int8))
        slope = float(calibrator.coef_[0, 0])
        intercept = float(calibrator.intercept_[0])
        if slope <= 0.0 or not math.isfinite(slope) or not math.isfinite(intercept):
            continue
        slopes[index] = slope
        intercepts[index] = intercept
        supported[index] = True
    idf = np.log((len(targets) + 1.0) / (np.sum(targets, axis=0) + 1.0)) + 1.0
    predictor = CalibratedSemanticPredictor(
        taxonomy_version=TAXONOMY_VERSION,
        ridge=float(ridge),
        vocabulary=vocabulary_tuple,
        categories=tuple(_category(tag) for tag in vocabulary_tuple),
        input_mean=mean,
        input_scale=scale,
        coefficients=coefficients,
        prior=prior,
        idf=idf,
        calibrator_slopes=slopes,
        calibrator_intercepts=intercepts,
        calibration_supported=supported,
    )
    predictor.validate()
    return predictor


def _embeddings(
    reader: FullTrackStoreReader,
    track_ids: Sequence[int],
) -> np.ndarray:
    positions = {track_id: row for row, track_id in enumerate(reader.track_ids)}
    try:
        rows = [positions[int(track_id)] for track_id in track_ids]
    except KeyError as exc:
        raise SemanticPredictorError(f"sealed store is missing track {exc.args[0]}") from exc
    values = np.asarray(reader.global_embeddings[rows], dtype=np.float32)
    if values.shape != (len(track_ids), reader.binding.embedding_dim):
        raise SemanticPredictorError("sealed-store embedding selection drift")
    return values


def _taxonomy_summary(vocabulary: Sequence[str]) -> Dict[str, int]:
    result = {category: 0 for category in CATEGORIES}
    for tag in vocabulary:
        result[_category(tag)] += 1
    return result


def _split_summary(split: LabelSplit) -> Dict[str, object]:
    split.validate()
    artists = tuple(sorted({int(value) for value in split.artist_ids}))
    return {
        "tracks": len(split.track_ids),
        "artists": len(artists),
        "track_ids_sha256": stable_json_sha256(
            tuple(int(value) for value in split.track_ids)
        ),
        "artist_ids_sha256": stable_json_sha256(artists),
    }


def calibration_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, object]:
    labels = np.asarray(targets, dtype=np.float64)
    values = np.asarray(probabilities, dtype=np.float64)
    if (
        labels.shape != values.shape
        or labels.ndim != 2
        or not len(labels)
        or not np.all(np.isfinite(labels))
        or not np.all(np.isfinite(values))
        or np.any((labels < 0.0) | (labels > 1.0))
        or np.any((values <= 0.0) | (values >= 1.0))
    ):
        raise SemanticPredictorError("calibration metric inputs are invalid")
    brier_by_tag = np.mean(np.square(values - labels), axis=0)
    log_loss = -np.mean(
        labels * np.log(values) + (1.0 - labels) * np.log(1.0 - values)
    )
    flattened_labels = labels.ravel()
    flattened_values = values.ravel()
    expected_calibration_error = 0.0
    for lower in np.linspace(0.0, 0.9, 10):
        upper = lower + 0.1
        selected = (flattened_values >= lower) & (
            flattened_values <= upper if upper >= 1.0 else flattened_values < upper
        )
        if np.any(selected):
            expected_calibration_error += (
                float(np.count_nonzero(selected))
                / len(flattened_values)
                * abs(
                    float(np.mean(flattened_values[selected]))
                    - float(np.mean(flattened_labels[selected]))
                )
            )
    return {
        "micro_brier": float(np.mean(np.square(values - labels))),
        "macro_brier": float(np.mean(brier_by_tag)),
        "micro_log_loss": float(log_loss),
        "expected_calibration_error_10_bins": expected_calibration_error,
    }


def tag_ranking_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    categories: Sequence[str],
    *,
    category_limits: Mapping[str, int] = DEFAULT_CATEGORY_LIMITS,
) -> Mapping[str, object]:
    labels = np.asarray(targets, dtype=np.float64)
    values = np.asarray(probabilities, dtype=np.float64)
    category_tuple = tuple(categories)
    limits = _validate_category_limits(category_limits)
    if (
        labels.shape != values.shape
        or labels.ndim != 2
        or len(category_tuple) != labels.shape[1]
        or any(category not in CATEGORIES for category in category_tuple)
        or not np.all(np.isfinite(labels))
        or not np.all(np.isfinite(values))
        or np.any((labels < 0.0) | (labels > 1.0))
    ):
        raise SemanticPredictorError("tag-ranking metric inputs are invalid")
    result: Dict[str, object] = {}
    category_array = np.asarray(category_tuple)
    for category in CATEGORIES:
        positions = np.flatnonzero(category_array == category)
        limit = limits[category]
        if limit > len(positions):
            raise SemanticPredictorError(
                f"tag-ranking quota exceeds {category} vocabulary"
            )
        local_order = np.argsort(-values[:, positions], axis=1)[:, :limit]
        selected_targets = np.take_along_axis(
            labels[:, positions], local_order, axis=1
        )
        positives = np.sum(labels[:, positions], axis=1)
        eligible = positives > 0.0
        if not np.any(eligible):
            raise SemanticPredictorError(
                f"tag-ranking metrics have no positive {category} labels"
            )
        hits = np.sum(selected_targets, axis=1)
        selected_global = positions[local_order]
        result[category] = {
            "tracks_with_labels": int(np.count_nonzero(eligible)),
            "mean_recall_at_quota": float(
                np.mean(hits[eligible] / positives[eligible])
            ),
            "hit_rate_at_quota": float(np.mean(hits[eligible] > 0.0)),
            "quota": limit,
            "selected_tag_coverage": int(len(np.unique(selected_global))),
            "category_tag_count": len(positions),
        }
    return result


def feature_domain_diagnostics(
    predictor: CalibratedSemanticPredictor,
    values: np.ndarray,
) -> Mapping[str, object]:
    predictor.validate()
    matrix = np.asarray(values)
    if (
        matrix.ndim != 2
        or matrix.shape[1] != predictor.input_mean.shape[0]
        or not len(matrix)
        or not np.all(np.isfinite(matrix))
    ):
        raise SemanticPredictorError("feature-domain input matrix is invalid")
    target_mean = np.mean(matrix, axis=0, dtype=np.float64)
    target_scale = np.std(matrix, axis=0, dtype=np.float64)
    if np.any(target_scale <= 1e-6):
        raise SemanticPredictorError("feature-domain input has constant coordinates")
    standardized_mean = (
        target_mean - predictor.input_mean
    ) / predictor.input_scale
    standardized_scale = target_scale / predictor.input_scale
    coordinate_mean_abs = float(np.mean(np.abs(standardized_mean)))
    coordinate_scale_mean = float(np.mean(standardized_scale))
    passed = (
        coordinate_mean_abs <= MAX_TARGET_STANDARDIZED_MEAN_ABS
        and MIN_TARGET_STANDARDIZED_SCALE_MEAN
        <= coordinate_scale_mean
        <= MAX_TARGET_STANDARDIZED_SCALE_MEAN
    )
    return {
        "rows": len(matrix),
        "dimensions": matrix.shape[1],
        "standardized_coordinate_mean_abs": coordinate_mean_abs,
        "standardized_coordinate_scale_mean": coordinate_scale_mean,
        "maximum_standardized_coordinate_mean_abs": (
            MAX_TARGET_STANDARDIZED_MEAN_ABS
        ),
        "minimum_standardized_coordinate_scale_mean": (
            MIN_TARGET_STANDARDIZED_SCALE_MEAN
        ),
        "maximum_standardized_coordinate_scale_mean": (
            MAX_TARGET_STANDARDIZED_SCALE_MEAN
        ),
        "passed": passed,
    }


def _model_arrays(
    predictor: CalibratedSemanticPredictor,
) -> Mapping[str, np.ndarray]:
    predictor.validate()
    return {
        "input_mean": predictor.input_mean.astype(np.float32),
        "input_scale": predictor.input_scale.astype(np.float32),
        "coefficients": predictor.coefficients.astype(np.float32),
        "prior": predictor.prior.astype(np.float32),
        "idf": predictor.idf.astype(np.float32),
        "calibrator_slopes": predictor.calibrator_slopes.astype(np.float32),
        "calibrator_intercepts": predictor.calibrator_intercepts.astype(np.float32),
        "calibration_supported": predictor.calibration_supported.astype(np.uint8),
        "vocabulary": np.asarray(predictor.vocabulary, dtype=np.str_),
        "categories": np.asarray(predictor.categories, dtype=np.str_),
    }


def _write_npz_exclusive(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    destination = Path(path).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_exclusive(path: Path, document: Mapping[str, object]) -> None:
    destination = Path(path).absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_json(path: Path, label: str) -> Mapping[str, object]:
    candidate = Path(path).absolute()
    if candidate.is_symlink():
        raise SemanticPredictorError(f"{label} may not be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
        document = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SemanticPredictorError(f"cannot read valid {label}: {exc}") from exc
    if not isinstance(document, dict):
        raise SemanticPredictorError(f"{label} must be a JSON object")
    return document


def load_predictor(
    model_path: Path,
    metadata_path: Path,
) -> CalibratedSemanticPredictor:
    metadata = _read_json(metadata_path, "semantic predictor metadata")
    if (
        metadata.get("schema_version") != MODEL_SCHEMA_VERSION
        or metadata.get("artifact_kind") != MODEL_KIND
        or metadata.get("taxonomy_version") != TAXONOMY_VERSION
        or metadata.get("taxonomy_counts") != TAXONOMY_COUNTS
        or metadata.get("tag_count") != sum(TAXONOMY_COUNTS.values())
        or metadata.get("vocabulary_sha256") != EXPECTED_VOCABULARY_SHA256
        or metadata.get("test_labels_accessed") is not False
        or metadata.get("production_ranking_changed") is not False
        or metadata.get("promotion_allowed") is not False
        or metadata.get("payload_sha256") != _payload_sha256(metadata)
        or metadata.get("model_npz_sha256") != sha256_path(Path(model_path))
    ):
        raise SemanticPredictorError("semantic predictor metadata drift")
    try:
        ridge = float(metadata["ridge"])
        input_dimension = int(metadata["input_dimension"])
        with np.load(Path(model_path), allow_pickle=False) as archive:
            if set(archive.files) != MODEL_ARRAYS:
                raise SemanticPredictorError("semantic predictor model array drift")
            predictor = CalibratedSemanticPredictor(
                taxonomy_version=TAXONOMY_VERSION,
                ridge=ridge,
                vocabulary=tuple(str(value) for value in archive["vocabulary"]),
                categories=tuple(str(value) for value in archive["categories"]),
                input_mean=np.asarray(archive["input_mean"], dtype=np.float64),
                input_scale=np.asarray(archive["input_scale"], dtype=np.float64),
                coefficients=np.asarray(archive["coefficients"], dtype=np.float64),
                prior=np.asarray(archive["prior"], dtype=np.float64),
                idf=np.asarray(archive["idf"], dtype=np.float64),
                calibrator_slopes=np.asarray(
                    archive["calibrator_slopes"], dtype=np.float64
                ),
                calibrator_intercepts=np.asarray(
                    archive["calibrator_intercepts"], dtype=np.float64
                ),
                calibration_supported=np.asarray(
                    archive["calibration_supported"], dtype=bool
                ),
            )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise SemanticPredictorError(f"cannot load semantic predictor model: {exc}") from exc
    predictor.validate()
    if (
        len(predictor.vocabulary) != sum(TAXONOMY_COUNTS.values())
        or predictor.input_mean.shape[0] != input_dimension
        or stable_json_sha256(predictor.vocabulary)
        != EXPECTED_VOCABULARY_SHA256
        or _taxonomy_summary(predictor.vocabulary) != TAXONOMY_COUNTS
    ):
        raise SemanticPredictorError("semantic predictor model/metadata mismatch")
    return predictor


def train_jamendo_predictor(
    *,
    metadata_root: Path,
    store_path: Path,
    model_output: Path,
    metadata_output: Path,
    report_output: Path,
    ridge: float = DEFAULT_RIDGE,
) -> Mapping[str, object]:
    outputs = tuple(
        Path(path) for path in (model_output, metadata_output, report_output)
    )
    if any(path.exists() for path in outputs):
        raise SemanticPredictorError("predictor output already exists; refusing overwrite")
    train = load_label_split(metadata_root, fold=OFFICIAL_FOLD, part=TRAIN_PART)
    validation = load_label_split(
        metadata_root, fold=OFFICIAL_FOLD, part=CALIBRATION_PART
    )
    if set(int(value) for value in train.artist_ids).intersection(
        int(value) for value in validation.artist_ids
    ):
        raise SemanticPredictorError("train/validation artist leakage")
    calibration_fit, calibration_audit = split_calibration_labels(validation)
    vocabulary, train_targets = build_targets(train)
    taxonomy_counts = _taxonomy_summary(vocabulary)
    if taxonomy_counts != TAXONOMY_COUNTS:
        raise SemanticPredictorError(
            f"expected the fixed 183-tag taxonomy, found {taxonomy_counts}"
        )
    _, calibration_targets = build_targets(calibration_fit, vocabulary)
    _, audit_targets = build_targets(calibration_audit, vocabulary)
    manifest_path = Path(store_path) / "store.sealed.json"
    if sha256_path(manifest_path) != EXPECTED_STORE_MANIFEST_FILE_SHA256:
        raise SemanticPredictorError("official sealed CLAP store manifest hash drift")
    with FullTrackStoreReader(
        Path(store_path),
        expected_source_fingerprint=str(
            EXPECTED_STORE_BINDING["source_fingerprint"]
        ),
        expected_config_sha256=str(EXPECTED_STORE_BINDING["config_sha256"]),
        expected_model_sha256=str(EXPECTED_STORE_BINDING["model_sha256"]),
    ) as reader:
        if reader.binding.as_dict() != EXPECTED_STORE_BINDING:
            raise SemanticPredictorError("official sealed CLAP store binding drift")
        train_inputs = _embeddings(reader, train.track_ids)
        calibration_inputs = _embeddings(reader, calibration_fit.track_ids)
        audit_inputs = _embeddings(reader, calibration_audit.track_ids)
        store_binding = reader.binding.as_dict()
        predictor = fit_calibrated_predictor(
            train_inputs,
            train_targets,
            calibration_inputs,
            calibration_targets,
            vocabulary,
            ridge=ridge,
        )
        raw_audit = np.clip(
            predictor.raw_scores(audit_inputs),
            PROBABILITY_EPSILON,
            1.0 - PROBABILITY_EPSILON,
        )
        calibrated_audit = predictor.predict_proba(audit_inputs)
    _write_npz_exclusive(Path(model_output), _model_arrays(predictor))
    report: Dict[str, object] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_kind": REPORT_KIND,
        "evidence_status": "calibration_audit_only",
        "taxonomy_version": TAXONOMY_VERSION,
        "taxonomy_counts": taxonomy_counts,
        "fold": OFFICIAL_FOLD,
        "ridge": float(ridge),
        "calibration_seed": CALIBRATION_SEED,
        "calibration_fit_percent": CALIBRATION_FIT_PERCENT,
        "minimum_calibration_class_count": MIN_CALIBRATION_CLASS_COUNT,
        "train": _split_summary(train),
        "calibration_fit": _split_summary(calibration_fit),
        "calibration_audit": _split_summary(calibration_audit),
        "train_validation_artist_overlap": 0,
        "calibration_fit_audit_artist_overlap": 0,
        "train_label_source_sha256": sha256_path(train.source_path),
        "validation_label_source_sha256": sha256_path(validation.source_path),
        "test_label_source_opened": False,
        "test_labels_accessed": False,
        "store_manifest_file_sha256": sha256_path(
            Path(store_path) / "store.sealed.json"
        ),
        "store_binding": store_binding,
        "input_dimension": int(predictor.input_mean.shape[0]),
        "tag_count": len(predictor.vocabulary),
        "calibrated_tag_count": int(
            np.count_nonzero(predictor.calibration_supported)
        ),
        "fallback_tag_count": int(
            len(predictor.vocabulary)
            - np.count_nonzero(predictor.calibration_supported)
        ),
        "audit_metrics": {
            "uncalibrated_clipped_ridge": calibration_metrics(
                audit_targets, raw_audit
            ),
            "calibrated": calibration_metrics(audit_targets, calibrated_audit),
            "tag_ranking": {
                "smoothed_prior_baseline": tag_ranking_metrics(
                    audit_targets,
                    np.broadcast_to(predictor.prior, audit_targets.shape),
                    predictor.categories,
                ),
                "calibrated_predictor": tag_ranking_metrics(
                    audit_targets,
                    calibrated_audit,
                    predictor.categories,
                ),
            },
        },
        "fallback_tags": [
            tag
            for tag, supported in zip(
                predictor.vocabulary, predictor.calibration_supported
            )
            if not supported
        ],
        "model_npz_sha256": sha256_path(Path(model_output)),
        "recommendation_quality_evaluated": False,
        "production_ranking_changed": False,
        "promotion_allowed": False,
    }
    report["payload_sha256"] = _payload_sha256(report)
    _write_json_exclusive(Path(report_output), report)
    metadata: Dict[str, object] = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "artifact_kind": MODEL_KIND,
        "taxonomy_version": TAXONOMY_VERSION,
        "taxonomy_counts": taxonomy_counts,
        "ridge": float(ridge),
        "input_dimension": int(predictor.input_mean.shape[0]),
        "tag_count": len(predictor.vocabulary),
        "vocabulary_sha256": stable_json_sha256(predictor.vocabulary),
        "model_npz_sha256": sha256_path(Path(model_output)),
        "report_file_sha256": sha256_path(Path(report_output)),
        "report_payload_sha256": report["payload_sha256"],
        "test_labels_accessed": False,
        "production_ranking_changed": False,
        "promotion_allowed": False,
    }
    metadata["payload_sha256"] = _payload_sha256(metadata)
    _write_json_exclusive(Path(metadata_output), metadata)
    return report


def export_sparse_predictions(
    *,
    predictor: CalibratedSemanticPredictor,
    embeddings: np.ndarray,
    track_ids: np.ndarray,
    output: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    category_limits: Mapping[str, int] = DEFAULT_CATEGORY_LIMITS,
    probability_threshold: float = DEFAULT_PROBABILITY_THRESHOLD,
) -> SparseSemanticPredictions:
    if Path(output).exists():
        raise SemanticPredictorError("sparse prediction output exists; refusing overwrite")
    matrix = np.asarray(embeddings)
    ids = np.asarray(track_ids, dtype=np.int64)
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
        or matrix.ndim != 2
        or matrix.shape[0] != len(ids)
        or matrix.shape[1] != predictor.input_mean.shape[0]
        or len(np.unique(ids)) != len(ids)
        or np.any(ids <= 0)
    ):
        raise SemanticPredictorError("sparse export inputs are invalid")
    limits = _validate_category_limits(category_limits)
    slot_categories = tuple(
        category for category in CATEGORIES for _ in range(limits[category])
    )
    slots = len(slot_categories)
    all_indices = np.empty((len(ids), slots), dtype=np.uint16)
    all_probabilities = np.empty((len(ids), slots), dtype=np.float16)
    all_counts = np.empty(len(ids), dtype=np.uint8)
    for start in range(0, len(ids), batch_size):
        end = min(start + batch_size, len(ids))
        batch = predictor.sparse(
            matrix[start:end],
            category_limits=limits,
            probability_threshold=probability_threshold,
            track_ids=ids[start:end],
        )
        all_indices[start:end] = batch.tag_indices
        all_probabilities[start:end] = batch.probabilities
        all_counts[start:end] = batch.counts
    result = SparseSemanticPredictions(
        track_ids=ids.copy(),
        tag_indices=all_indices,
        probabilities=all_probabilities,
        counts=all_counts,
        slot_categories=slot_categories,
    )
    result.validate(len(predictor.vocabulary))
    _write_npz_exclusive(
        Path(output),
        {
            "track_ids": result.track_ids,
            "tag_indices": result.tag_indices,
            "probabilities": result.probabilities,
            "counts": result.counts,
            "slot_categories": np.asarray(result.slot_categories, dtype=np.str_),
            "vocabulary": np.asarray(predictor.vocabulary, dtype=np.str_),
            "taxonomy_version": np.asarray(
                [predictor.taxonomy_version], dtype=np.str_
            ),
        },
    )
    return result


def export_production_sparse_predictions(
    *,
    model_path: Path,
    model_metadata_path: Path,
    embeddings_path: Path,
    embeddings_build_path: Path,
    index_path: Path,
    output: Path,
    output_metadata: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
    category_limits: Mapping[str, int] = DEFAULT_CATEGORY_LIMITS,
    probability_threshold: float = DEFAULT_PROBABILITY_THRESHOLD,
) -> Tuple[SparseSemanticPredictions, Mapping[str, object]]:
    if Path(output_metadata).exists():
        raise SemanticPredictorError(
            "sparse metadata output exists; refusing overwrite"
        )
    source_hashes = {
        "embeddings_file_sha256": sha256_path(Path(embeddings_path)),
        "embeddings_build_file_sha256": sha256_path(Path(embeddings_build_path)),
        "production_index_file_sha256": sha256_path(Path(index_path)),
    }
    expected_hashes = {
        "embeddings_file_sha256": EXPECTED_PRODUCTION_EMBEDDINGS_SHA256,
        "embeddings_build_file_sha256": (
            EXPECTED_PRODUCTION_EMBEDDINGS_BUILD_SHA256
        ),
        "production_index_file_sha256": EXPECTED_PRODUCTION_INDEX_SHA256,
    }
    if source_hashes != expected_hashes:
        raise SemanticPredictorError("pinned production semantic source hash drift")
    predictor = load_predictor(model_path, model_metadata_path)
    embeddings = np.load(
        Path(embeddings_path), mmap_mode="r", allow_pickle=False
    )
    with np.load(Path(index_path), allow_pickle=False) as index:
        if "track_ids" not in index:
            raise SemanticPredictorError("production index has no track IDs")
        track_ids = np.asarray(index["track_ids"], dtype=np.int64)
    track_ids_sha256 = stable_json_sha256(
        tuple(int(value) for value in track_ids)
    )
    if (
        embeddings.shape
        != (EXPECTED_PRODUCTION_ROWS, predictor.input_mean.shape[0])
        or embeddings.dtype != np.float16
        or track_ids.shape != (EXPECTED_PRODUCTION_ROWS,)
        or len(np.unique(track_ids)) != EXPECTED_PRODUCTION_ROWS
        or np.any(track_ids <= 0)
        or track_ids_sha256 != EXPECTED_PRODUCTION_TRACK_IDS_SHA256
    ):
        raise SemanticPredictorError("pinned production semantic row alignment drift")
    domain_diagnostics = feature_domain_diagnostics(predictor, embeddings)
    if not domain_diagnostics["passed"]:
        raise SemanticPredictorError(
            "production CLAP feature domain is incompatible with the Jamendo "
            f"predictor: {json.dumps(domain_diagnostics, sort_keys=True)}"
        )
    sparse = export_sparse_predictions(
        predictor=predictor,
        embeddings=embeddings,
        track_ids=track_ids,
        output=output,
        batch_size=batch_size,
        category_limits=category_limits,
        probability_threshold=probability_threshold,
    )
    model_metadata = _read_json(
        model_metadata_path, "semantic predictor metadata"
    )
    limits = _validate_category_limits(category_limits)
    metadata: Dict[str, object] = {
        "schema_version": SPARSE_SCHEMA_VERSION,
        "artifact_kind": SPARSE_METADATA_KIND,
        "taxonomy_version": TAXONOMY_VERSION,
        "taxonomy_counts": TAXONOMY_COUNTS,
        "vocabulary_sha256": EXPECTED_VOCABULARY_SHA256,
        "model_npz_sha256": sha256_path(Path(model_path)),
        "model_metadata_file_sha256": sha256_path(Path(model_metadata_path)),
        "model_metadata_payload_sha256": model_metadata["payload_sha256"],
        **source_hashes,
        "production_track_ids_sha256": track_ids_sha256,
        "feature_domain_diagnostics": domain_diagnostics,
        "tracks": len(sparse.track_ids),
        "category_limits": limits,
        "slot_categories": list(sparse.slot_categories),
        "slots_per_track": len(sparse.slot_categories),
        "probability_threshold": float(probability_threshold),
        "nonempty_predictions": int(np.sum(sparse.counts)),
        "sparse_npz_sha256": sha256_path(Path(output)),
        "test_labels_accessed": False,
        "recommendation_quality_evaluated": False,
        "production_ranking_changed": False,
        "promotion_allowed": False,
    }
    metadata["payload_sha256"] = _payload_sha256(metadata)
    _write_json_exclusive(Path(output_metadata), metadata)
    return sparse, metadata


def load_sparse_predictions(path: Path) -> SparseSemanticPredictions:
    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            if set(archive.files) != SPARSE_ARRAYS:
                raise SemanticPredictorError("sparse semantic array drift")
            taxonomy = tuple(str(value) for value in archive["taxonomy_version"])
            vocabulary = tuple(str(value) for value in archive["vocabulary"])
            if (
                taxonomy != (TAXONOMY_VERSION,)
                or len(vocabulary) != sum(TAXONOMY_COUNTS.values())
                or stable_json_sha256(vocabulary)
                != EXPECTED_VOCABULARY_SHA256
                or _taxonomy_summary(vocabulary) != TAXONOMY_COUNTS
            ):
                raise SemanticPredictorError("sparse semantic taxonomy drift")
            result = SparseSemanticPredictions(
                track_ids=np.asarray(archive["track_ids"], dtype=np.int64),
                tag_indices=np.asarray(archive["tag_indices"], dtype=np.uint16),
                probabilities=np.asarray(
                    archive["probabilities"], dtype=np.float16
                ),
                counts=np.asarray(archive["counts"], dtype=np.uint8),
                slot_categories=tuple(
                    str(value) for value in archive["slot_categories"]
                ),
            )
    except (OSError, ValueError) as exc:
        raise SemanticPredictorError(
            f"cannot load sparse semantic predictions: {exc}"
        ) from exc
    result.validate(len(vocabulary))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    train = subparsers.add_parser("train")
    train.add_argument("--metadata-root", required=True)
    train.add_argument("--store", required=True)
    train.add_argument("--model-output", required=True)
    train.add_argument("--metadata-output", required=True)
    train.add_argument("--report-output", required=True)
    train.add_argument("--ridge", type=float, default=DEFAULT_RIDGE)
    export = subparsers.add_parser("export")
    export.add_argument("--model", required=True)
    export.add_argument("--metadata", required=True)
    export.add_argument("--embeddings", required=True)
    export.add_argument("--embeddings-build", required=True)
    export.add_argument("--index", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--output-metadata", required=True)
    export.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    export.add_argument(
        "--probability-threshold",
        type=float,
        default=DEFAULT_PROBABILITY_THRESHOLD,
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "train":
            report = train_jamendo_predictor(
                metadata_root=Path(args.metadata_root),
                store_path=Path(args.store),
                model_output=Path(args.model_output),
                metadata_output=Path(args.metadata_output),
                report_output=Path(args.report_output),
                ridge=args.ridge,
            )
            result = {
                "model_npz_sha256": report["model_npz_sha256"],
                "audit_metrics": report["audit_metrics"],
                "calibrated_tag_count": report["calibrated_tag_count"],
                "test_labels_accessed": False,
                "production_ranking_changed": False,
            }
        else:
            sparse, metadata = export_production_sparse_predictions(
                model_path=Path(args.model),
                model_metadata_path=Path(args.metadata),
                embeddings_path=Path(args.embeddings),
                embeddings_build_path=Path(args.embeddings_build),
                index_path=Path(args.index),
                output=Path(args.output),
                output_metadata=Path(args.output_metadata),
                batch_size=args.batch_size,
                probability_threshold=args.probability_threshold,
            )
            result = {
                "output": str(Path(args.output).absolute()),
                "output_sha256": sha256_path(Path(args.output)),
                "tracks": len(sparse.track_ids),
                "slots_per_track": len(sparse.slot_categories),
                "nonempty_predictions": int(np.sum(sparse.counts)),
                "metadata_payload_sha256": metadata["payload_sha256"],
                "production_ranking_changed": False,
            }
    except (OSError, ValueError, SemanticPredictorError) as exc:
        raise SystemExit(f"Semantic predictor blocked: {exc}") from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
