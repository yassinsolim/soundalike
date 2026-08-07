"""Seal reusable semantic and vocal features for V4 ranking."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .fulltrack_extract import normalize_rows
from .fulltrack_store import FullTrackStoreReader
from .jamendo_fulltrack import JamendoContext, load_jamendo_context
from .semantic_predictor import load_predictor
from .v4_gates import (
    INSTRUMENTAL,
    UNKNOWN,
    VOCAL,
    VocalThresholds,
    calibrate_vocal_thresholds,
    classify_vocal,
)


SCHEMA_VERSION = 1
CACHE_KIND = "soundalike_v4_semantic_feature_cache"
VOICE_TAG = "instrument---voice"
INSTRUMENTAL_PREFIXES = (
    "genre---hiphopinstrumental",
    "genre---instrumental",
)
STATE_CODES = {UNKNOWN: 0, VOCAL: 1, INSTRUMENTAL: 2}


class V4FeatureError(RuntimeError):
    """A V4 feature cache is incomplete, inconsistent, or mutable."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_sha256(value: Mapping[str, object]) -> str:
    payload = dict(value)
    payload.pop("payload_sha256", None)
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _voice_position(vocabulary: Sequence[str]) -> int:
    positions = [
        index for index, value in enumerate(vocabulary) if value == VOICE_TAG
    ]
    if len(positions) != 1:
        raise V4FeatureError("semantic voice label identity drift")
    return positions[0]


def _calibration_ids(
    context: JamendoContext,
    development_ids: set[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    vocal = []
    instrumental = []
    for track in context.tracks:
        if int(track.track_id) not in development_ids:
            continue
        tags = set(track.tags)
        if VOICE_TAG in tags:
            vocal.append(int(track.track_id))
        if (
            VOICE_TAG not in tags
            and "genre---singersongwriter" not in tags
            and any(
                tag.startswith(prefix)
                for tag in tags
                for prefix in INSTRUMENTAL_PREFIXES
            )
        ):
            instrumental.append(int(track.track_id))
    if len(vocal) < 100 or len(instrumental) < 100:
        raise V4FeatureError("semantic voice calibration coverage is insufficient")
    return tuple(sorted(vocal)), tuple(sorted(instrumental))


def build_semantic_cache(
    *,
    context: JamendoContext,
    population_manifest: Mapping[str, object],
    store_root: Path,
    predictor_model: Path,
    predictor_metadata: Path,
    cache_path: Path,
    metadata_path: Path,
) -> Mapping[str, object]:
    if population_manifest.get("source_fingerprint") != context.source_fingerprint:
        raise V4FeatureError("population/source fingerprint drift")
    development = population_manifest.get("development")
    if not isinstance(development, Mapping) or not isinstance(
        development.get("track_ids"), list
    ):
        raise V4FeatureError("population development section is invalid")
    development_ids = set(int(value) for value in development["track_ids"])
    predictor = load_predictor(predictor_model, predictor_metadata)
    with FullTrackStoreReader(
        store_root, expected_source_fingerprint=context.source_fingerprint
    ) as reader:
        track_ids = np.asarray(reader.track_ids, dtype=np.int64)
        globals_matrix = normalize_rows(
            np.asarray(reader.global_embeddings, dtype=np.float32)
        )
        probabilities = predictor.predict_proba(globals_matrix)
        voice_position = _voice_position(predictor.vocabulary)
        voice_scores = probabilities[:, voice_position].astype(np.float32)
        row_by_track = {
            int(track_id): row for row, track_id in enumerate(track_ids)
        }
        vocal_ids, instrumental_ids = _calibration_ids(
            context, development_ids
        )
        thresholds = calibrate_vocal_thresholds(
            [voice_scores[row_by_track[track_id]] for track_id in vocal_ids],
            [
                voice_scores[row_by_track[track_id]]
                for track_id in instrumental_ids
            ],
        )
        states = np.asarray(
            [
                STATE_CODES[classify_vocal(float(score), thresholds)]
                for score in voice_scores
            ],
            dtype=np.uint8,
        )
        store_binding = {
            "source_fingerprint": reader.binding.source_fingerprint,
            "config_sha256": reader.binding.config_sha256,
            "model_sha256": reader.binding.model_sha256,
            "track_plan_sha256": reader.binding.track_plan_sha256,
            "track_count": reader.binding.track_count,
        }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            track_ids=track_ids,
            probabilities=probabilities.astype(np.float16),
            voice_scores=voice_scores,
            voice_states=states,
        )
    temporary.replace(cache_path)
    metadata: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "cache_kind": CACHE_KIND,
        "research_only": True,
        "promotion_allowed": False,
        "source_fingerprint": context.source_fingerprint,
        "population_sha256": population_manifest.get("content_sha256"),
        "store_binding": store_binding,
        "predictor": {
            "model_file_sha256": _sha256(predictor_model),
            "metadata_file_sha256": _sha256(predictor_metadata),
            "voice_tag": VOICE_TAG,
            "voice_position": voice_position,
        },
        "calibration": {
            "fold_parts": ["train", "validation"],
            "vocal_track_count": len(vocal_ids),
            "instrumental_track_count": len(instrumental_ids),
            "reserve_labels_accessed": False,
            "maximum_false_exclusion_rate": 0.05,
        },
        "thresholds": {
            "instrumental_max": thresholds.instrumental_max,
            "vocal_min": thresholds.vocal_min,
            "middle_state": UNKNOWN,
        },
        "states": STATE_CODES,
        "arrays": {
            "track_count": len(track_ids),
            "semantic_dimensions": probabilities.shape[1],
            "file_sha256": _sha256(cache_path),
        },
    }
    metadata["payload_sha256"] = _payload_sha256(metadata)
    temporary_metadata = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary_metadata.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary_metadata.replace(metadata_path)
    load_semantic_cache(
        cache_path,
        metadata_path,
        expected_source_fingerprint=context.source_fingerprint,
        expected_track_ids=track_ids,
    )
    return metadata


def load_semantic_cache(
    cache_path: Path,
    metadata_path: Path,
    *,
    expected_source_fingerprint: str,
    expected_track_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise V4FeatureError("cannot read V4 feature metadata") from exc
    if (
        metadata.get("schema_version") != SCHEMA_VERSION
        or metadata.get("cache_kind") != CACHE_KIND
        or metadata.get("source_fingerprint") != expected_source_fingerprint
        or metadata.get("payload_sha256") != _payload_sha256(metadata)
        or metadata.get("arrays", {}).get("file_sha256") != _sha256(cache_path)
    ):
        raise V4FeatureError("V4 feature cache binding drift")
    try:
        with np.load(cache_path, allow_pickle=False) as archive:
            if set(archive.files) != {
                "track_ids",
                "probabilities",
                "voice_scores",
                "voice_states",
            }:
                raise V4FeatureError("V4 feature array set drift")
            track_ids = np.asarray(archive["track_ids"], dtype=np.int64)
            probabilities = np.asarray(
                archive["probabilities"], dtype=np.float32
            )
            voice_scores = np.asarray(archive["voice_scores"], dtype=np.float32)
            voice_states = np.asarray(archive["voice_states"], dtype=np.uint8)
    except (OSError, ValueError) as exc:
        raise V4FeatureError("cannot load V4 feature arrays") from exc
    if (
        not np.array_equal(track_ids, expected_track_ids)
        or probabilities.shape
        != (len(track_ids), int(metadata["arrays"]["semantic_dimensions"]))
        or voice_scores.shape != (len(track_ids),)
        or voice_states.shape != (len(track_ids),)
        or not np.all(np.isfinite(probabilities))
        or not np.all(np.isfinite(voice_scores))
        or not set(np.unique(voice_states)).issubset(set(STATE_CODES.values()))
    ):
        raise V4FeatureError("V4 feature array identity or shape drift")
    return probabilities, voice_scores, voice_states


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build V4 semantic features.")
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--store-root", type=Path, required=True)
    parser.add_argument("--predictor-model", type=Path, required=True)
    parser.add_argument("--predictor-metadata", type=Path, required=True)
    parser.add_argument("--cache-output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    context = load_jamendo_context(
        args.metadata_root, args.audio_root, args.state_root, production=True
    )
    population = json.loads(args.population.read_text(encoding="utf-8"))
    metadata = build_semantic_cache(
        context=context,
        population_manifest=population,
        store_root=args.store_root,
        predictor_model=args.predictor_model,
        predictor_metadata=args.predictor_metadata,
        cache_path=args.cache_output,
        metadata_path=args.metadata_output,
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
