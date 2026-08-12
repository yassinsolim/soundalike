"""Build fail-closed multi-segment vocal and language gates for V5 studies."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Mapping, Sequence, cast

import numpy as np

from .jamendo_fulltrack import JamendoTrack, load_jamendo_context
from .v4_gates import INSTRUMENTAL, UNKNOWN, VOCAL, decide_language, representative_starts
from .v4_language_probe import (
    MODEL_ID,
    MODEL_REVISION,
    PROBE_KIND,
    _language_probabilities,
    _load_audio,
)
from .v4_track_gates import (
    GATE_KIND as V4_GATE_KIND,
    SCHEMA_VERSION as V4_SCHEMA_VERSION,
    _content_sha256,
    _load_bound_report,
    _sha256,
    _write,
    conservative_vocal_state,
)


SCHEMA_VERSION = 3
GATE_KIND = "soundalike_v5_multisegment_track_gates_v1"
SEGMENTS_PER_TRACK = 3


class V5TrackGateError(RuntimeError):
    """The V5 gate inputs or multi-segment detector output are invalid."""


def strict_resolved_vocal_state(
    semantic_state: str,
    panns_state: str,
    language: str,
) -> str:
    """Require known language plus vocal evidence without instrumental conflict."""
    agreed = conservative_vocal_state(semantic_state, panns_state)
    if not isinstance(language, str) or not language:
        raise V5TrackGateError("language detector state is invalid")
    if language == UNKNOWN:
        return INSTRUMENTAL if agreed == INSTRUMENTAL else UNKNOWN
    if INSTRUMENTAL in (semantic_state, panns_state):
        return UNKNOWN
    return VOCAL if VOCAL in (semantic_state, panns_state) else UNKNOWN


def aggregate_language_probabilities(
    excerpts: Sequence[Mapping[str, float]],
) -> Mapping[str, float]:
    if not excerpts or any(
        not row
        or any(
            not isinstance(language, str)
            or not language
            or not np.isfinite(probability)
            or probability < 0.0
            for language, probability in row.items()
        )
        for row in excerpts
    ):
        raise V5TrackGateError("language probability rows are invalid")
    languages = sorted({language for row in excerpts for language in row})
    averaged = {
        language: float(np.mean([row.get(language, 0.0) for row in excerpts]))
        for language in languages
    }
    total = sum(averaged.values())
    if not np.isfinite(total) or total <= 0.0:
        raise V5TrackGateError("language probability mass is invalid")
    return {language: probability / total for language, probability in averaged.items()}


def stable_language(
    segment_languages: Sequence[str],
    aggregate_language: str,
) -> str:
    if (
        len(segment_languages) != SEGMENTS_PER_TRACK
        or any(not isinstance(language, str) or not language for language in segment_languages)
        or not isinstance(aggregate_language, str)
        or not aggregate_language
    ):
        raise V5TrackGateError("multi-segment language decisions are invalid")
    distinct = set(segment_languages)
    if (
        len(distinct) == 1
        and aggregate_language in distinct
        and aggregate_language != UNKNOWN
    ):
        return aggregate_language
    return UNKNOWN


def validate_multisegment_gate_rows(
    rows: Mapping[str, Mapping[str, object]],
) -> None:
    """Verify that final states can be reproduced from retained detector decisions."""
    valid_states = {VOCAL, INSTRUMENTAL, UNKNOWN}
    if not rows:
        raise V5TrackGateError("multi-segment gate rows are missing")
    for track_id, row in rows.items():
        if (
            not isinstance(track_id, str)
            or not track_id.isdigit()
            or int(track_id) <= 0
            or not isinstance(row, Mapping)
        ):
            raise V5TrackGateError("multi-segment gate track identity is invalid")
        semantic_state = row.get("semantic_vocal_state")
        panns_state = row.get("panns_vocal_state")
        vocal_state = row.get("vocal_state")
        language = row.get("language")
        audited = row.get("multisegment_audited")
        starts = row.get("language_segment_starts")
        decisions = row.get("language_segment_decisions")
        aggregate = row.get("language_aggregate_decision")
        confidence = row.get("language_confidence")
        margin = row.get("language_margin")
        if (
            not isinstance(semantic_state, str)
            or semantic_state not in valid_states
            or not isinstance(panns_state, str)
            or panns_state not in valid_states
            or not isinstance(vocal_state, str)
            or vocal_state not in valid_states
            or not isinstance(language, str)
            or not isinstance(audited, bool)
            or not isinstance(starts, list)
            or not isinstance(decisions, list)
            or not isinstance(aggregate, str)
            or isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or isinstance(margin, bool)
            or not isinstance(margin, (int, float))
            or not np.isfinite(confidence)
            or not np.isfinite(margin)
            or not 0.0 <= float(confidence) <= 1.0
            or not 0.0 <= float(margin) <= 1.0
        ):
            raise V5TrackGateError("multi-segment gate row structure is invalid")
        if audited:
            if (
                len(starts) != SEGMENTS_PER_TRACK
                or any(
                    isinstance(start, bool)
                    or not isinstance(start, (int, float))
                    or not np.isfinite(start)
                    or start < 0.0
                    for start in starts
                )
                or list(starts) != sorted(starts)
                or len(decisions) != SEGMENTS_PER_TRACK
            ):
                raise V5TrackGateError("multi-segment audit coverage is invalid")
            resolved_language = stable_language(
                cast(Sequence[str], decisions), aggregate
            )
        else:
            if starts or decisions or aggregate != UNKNOWN:
                raise V5TrackGateError("unaudited track retains segment decisions")
            resolved_language = UNKNOWN
        expected_state = strict_resolved_vocal_state(
            cast(str, semantic_state),
            cast(str, panns_state),
            resolved_language,
        )
        expected_language = (
            resolved_language if expected_state == VOCAL else UNKNOWN
        )
        if vocal_state != expected_state or language != expected_language:
            raise V5TrackGateError("multi-segment final gate decision is invalid")
        if expected_state == VOCAL:
            if confidence <= 0.0 or margin <= 0.0:
                raise V5TrackGateError("resolved vocal language confidence is invalid")
        elif confidence != 0.0 or margin != 0.0:
            raise V5TrackGateError("unresolved track retains language confidence")


def _decode_track(track: JamendoTrack) -> tuple[int, list[float], list[np.ndarray]]:
    starts = list(
        representative_starts(float(track.duration_seconds), excerpt_seconds=30.0)
    )
    if len(starts) != SEGMENTS_PER_TRACK:
        raise V5TrackGateError("multi-segment language positions drifted")
    waveforms = [_load_audio(Path(track.audio_path), start) for start in starts]
    return int(track.track_id), starts, waveforms


def _load_v4_cache(path: Path, source_fingerprint: str) -> Mapping[str, object]:
    cache = json.loads(path.read_text(encoding="utf-8"))
    if (
        cache.get("schema_version") != V4_SCHEMA_VERSION
        or cache.get("gate_kind") != V4_GATE_KIND
        or cache.get("source_fingerprint") != source_fingerprint
        or cache.get("content_sha256") != _content_sha256(cache)
        or not isinstance(cache.get("tracks"), Mapping)
    ):
        raise V5TrackGateError("V4 source gate cache binding failed")
    return cache


def build_multisegment_gate_cache(
    *,
    metadata_root: Path,
    audio_root: Path,
    state_root: Path,
    source_gate_path: Path,
    whisper_report_path: Path,
    model_root: Path,
    batch_tracks: int,
    workers: int,
) -> Mapping[str, object]:
    if batch_tracks <= 0 or workers <= 0:
        raise V5TrackGateError("V5 gate batch settings are invalid")
    context = load_jamendo_context(
        metadata_root, audio_root, state_root, production=True
    )
    source = _load_v4_cache(source_gate_path, context.source_fingerprint)
    source_rows = cast(Mapping[str, Mapping[str, object]], source["tracks"])
    by_id = {int(track.track_id): track for track in context.tracks}
    track_ids = sorted(int(track_id) for track_id in source_rows)
    if not set(track_ids).issubset(by_id):
        raise V5TrackGateError("V5 gate source tracks are absent from the corpus")

    whisper_report = _load_bound_report(whisper_report_path, PROBE_KIND)
    thresholds = whisper_report.get("thresholds")
    model_binding = whisper_report.get("model")
    if not isinstance(thresholds, Mapping) or not isinstance(model_binding, Mapping):
        raise V5TrackGateError("Whisper calibration structure is invalid")
    minimum_confidence = float(thresholds.get("minimum_confidence", -1.0))
    minimum_margin = float(thresholds.get("minimum_margin", -1.0))
    if minimum_confidence < 0.8 or minimum_margin < 0.5:
        raise V5TrackGateError("Whisper confidence floors are too permissive")
    model_file = model_root / "model.safetensors"
    if (
        model_binding.get("model_id") != MODEL_ID
        or model_binding.get("model_revision") != MODEL_REVISION
        or not model_file.is_file()
        or _sha256(model_file)
        != model_binding.get("model_file_sha256")
    ):
        raise V5TrackGateError("Whisper model binding failed")

    provisional_ids = []
    for track_id in track_ids:
        row = source_rows[str(track_id)]
        if (
            strict_resolved_vocal_state(
                str(row["semantic_vocal_state"]),
                str(row["panns_vocal_state"]),
                str(row["language"]),
            )
            == VOCAL
        ):
            provisional_ids.append(track_id)
    eligible_ids = [
        track_id
        for track_id in provisional_ids
        if len(
            representative_starts(
                float(by_id[track_id].duration_seconds), excerpt_seconds=30.0
            )
        )
        == SEGMENTS_PER_TRACK
    ]

    import torch
    from transformers import AutoProcessor, WhisperForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_root, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_root,
        local_files_only=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to("cuda")
    model.eval()

    audited: dict[int, Mapping[str, object]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for offset in range(0, len(eligible_ids), batch_tracks):
            current_ids = eligible_ids[offset : offset + batch_tracks]
            decoded = list(
                executor.map(
                    _decode_track,
                    [by_id[track_id] for track_id in current_ids],
                )
            )
            waveforms = [
                waveform for _, _, track_waveforms in decoded for waveform in track_waveforms
            ]
            probabilities = _language_probabilities(model, processor, waveforms)
            if len(probabilities) != len(current_ids) * SEGMENTS_PER_TRACK:
                raise V5TrackGateError("Whisper multi-segment coverage drifted")
            for index, (track_id, starts, _) in enumerate(decoded):
                begin = index * SEGMENTS_PER_TRACK
                excerpts = probabilities[begin : begin + SEGMENTS_PER_TRACK]
                aggregate = aggregate_language_probabilities(excerpts)
                combined = decide_language(
                    aggregate,
                    minimum_confidence=minimum_confidence,
                    minimum_margin=minimum_margin,
                )
                segment_decisions = [
                    decide_language(
                        row,
                        minimum_confidence=minimum_confidence,
                        minimum_margin=minimum_margin,
                    )
                    for row in excerpts
                ]
                segment_languages = [
                    decision.language for decision in segment_decisions
                ]
                language = stable_language(segment_languages, combined.language)
                audited[track_id] = {
                    "segment_starts": [round(float(value), 3) for value in starts],
                    "segment_languages": segment_languages,
                    "aggregate_language": combined.language,
                    "language": language,
                    "language_confidence": (
                        round(combined.confidence, 8) if language != UNKNOWN else 0.0
                    ),
                    "language_margin": (
                        round(combined.margin, 8) if language != UNKNOWN else 0.0
                    ),
                }
            print(
                f"V5 language audit: {min(offset + batch_tracks, len(eligible_ids))}"
                f"/{len(eligible_ids)} tracks",
                flush=True,
            )

    rows: dict[str, Mapping[str, object]] = {}
    for track_id in track_ids:
        source_row = source_rows[str(track_id)]
        semantic_state = str(source_row["semantic_vocal_state"])
        panns_state = str(source_row["panns_vocal_state"])
        decision = audited.get(track_id)
        language = str(decision["language"]) if decision is not None else UNKNOWN
        vocal_state = strict_resolved_vocal_state(
            semantic_state, panns_state, language
        )
        rows[str(track_id)] = {
            **source_row,
            "recurrence_language": source_row["language"],
            "recurrence_language_confidence": source_row["language_confidence"],
            "recurrence_language_margin": source_row["language_margin"],
            "multisegment_audited": decision is not None,
            "language_segment_starts": (
                decision["segment_starts"] if decision is not None else []
            ),
            "language_segment_decisions": (
                decision["segment_languages"] if decision is not None else []
            ),
            "language_aggregate_decision": (
                decision["aggregate_language"] if decision is not None else UNKNOWN
            ),
            "vocal_state": vocal_state,
            "language": language if vocal_state == VOCAL else UNKNOWN,
            "language_confidence": (
                decision["language_confidence"] if vocal_state == VOCAL else 0.0
            ),
            "language_margin": (
                decision["language_margin"] if vocal_state == VOCAL else 0.0
            ),
        }

    state_counts = Counter(str(row["vocal_state"]) for row in rows.values())
    language_counts = Counter(str(row["language"]) for row in rows.values())
    report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "gate_kind": GATE_KIND,
        "research_only": True,
        "promotion_allowed": False,
        "source_fingerprint": context.source_fingerprint,
        "source_gate_sha256": source["content_sha256"],
        "whisper_calibration_sha256": whisper_report["content_sha256"],
        "policy": {
            "language_positions": [0.15, 0.5, 0.85],
            "language_aggregation": (
                "all three segment decisions must match the known "
                "mean-probability decision"
            ),
            "vocal_state": (
                "known language plus vocal evidence and no instrumental conflict"
            ),
            "unknown_fallback": False,
            "transcription_saved": False,
        },
        "observed": {
            "track_count": len(rows),
            "provisional_vocal_tracks": len(provisional_ids),
            "multisegment_audited_tracks": len(audited),
            "insufficient_duration_tracks": len(provisional_ids) - len(eligible_ids),
            "vocal_state_counts": dict(sorted(state_counts.items())),
            "known_language_count": sum(
                row["language"] != UNKNOWN for row in rows.values()
            ),
            "language_counts": dict(sorted(language_counts.items())),
        },
        "tracks": rows,
    }
    validate_multisegment_gate_rows(rows)
    report["content_sha256"] = _content_sha256(report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build fail-closed multi-segment V5 track gates."
    )
    for name in (
        "metadata_root",
        "audio_root",
        "state_root",
        "source_gate",
        "whisper_report",
        "model_root",
        "output",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--batch-tracks", type=int, default=8)
    parser.add_argument("--workers", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_multisegment_gate_cache(
        metadata_root=args.metadata_root,
        audio_root=args.audio_root,
        state_root=args.state_root,
        source_gate_path=args.source_gate,
        whisper_report_path=args.whisper_report,
        model_root=args.model_root,
        batch_tracks=args.batch_tracks,
        workers=args.workers,
    )
    _write(args.output, report)
    print(json.dumps(report["observed"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
