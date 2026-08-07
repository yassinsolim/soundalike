"""Apply calibrated vocal and singing-language gates to V4 study tracks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .fulltrack_store import FullTrackStoreReader
from .jamendo_fulltrack import load_jamendo_context
from .v4_features import load_semantic_cache
from .v4_gate_probe import (
    EXCERPT_SECONDS as PANNS_EXCERPT_SECONDS,
    SAMPLE_RATE as PANNS_SAMPLE_RATE,
)
from .v4_gates import (
    INSTRUMENTAL,
    UNKNOWN,
    VOCAL,
    VocalThresholds,
    classify_vocal,
    decide_language,
)
from .v4_language_probe import (
    EXCERPT_SECONDS as WHISPER_EXCERPT_SECONDS,
    MODEL_ID,
    MODEL_REVISION,
    SAMPLE_RATE as WHISPER_SAMPLE_RATE,
    _language_probabilities,
)
from .v4_study import CODE_STATES, PLAN_KIND, _excerpt


SCHEMA_VERSION = 1
GATE_KIND = "soundalike_v4_study_track_gates"


class V4TrackGateError(RuntimeError):
    """The V4 study gate inputs or detector output are invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_sha256(value: Mapping[str, object]) -> str:
    payload = {key: item for key, item in value.items() if key != "content_sha256"}
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _load_bound_report(path: Path, expected_kind: str) -> Mapping[str, object]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("probe_kind") != expected_kind
        or report.get("content_sha256") != _content_sha256(report)
    ):
        raise V4TrackGateError(f"calibration report binding failed: {path.name}")
    return report


def conservative_vocal_state(semantic_state: str, panns_state: str) -> str:
    """Return a known state only when the independent detectors agree."""
    valid = {VOCAL, INSTRUMENTAL, UNKNOWN}
    if semantic_state not in valid or panns_state not in valid:
        raise V4TrackGateError("vocal detector state is invalid")
    return semantic_state if semantic_state == panns_state else UNKNOWN


def _load_audio(
    path: Path,
    *,
    start: float,
    seconds: float,
    sample_rate: int,
) -> np.ndarray:
    import librosa

    values, _ = librosa.load(
        path,
        sr=sample_rate,
        mono=True,
        offset=float(start),
        duration=float(seconds),
    )
    expected = int(sample_rate * seconds)
    if not len(values):
        raise V4TrackGateError(f"decoded empty study excerpt: {path}")
    return np.pad(values, (0, max(0, expected - len(values))))[:expected].astype(
        np.float32
    )


def _panns_scores(
    model: object,
    labels: Sequence[str],
    paths: Sequence[Path],
    starts: Sequence[float],
    *,
    batch_size: int,
) -> list[float]:
    import torch

    if len(paths) != len(starts) or batch_size <= 0:
        raise V4TrackGateError("PANNs study inputs are invalid")
    scores: list[float] = []
    for offset in range(0, len(paths), batch_size):
        batch = np.stack(
            [
                _load_audio(
                    path,
                    start=start,
                    seconds=PANNS_EXCERPT_SECONDS,
                    sample_rate=PANNS_SAMPLE_RATE,
                )
                for path, start in zip(
                    paths[offset : offset + batch_size],
                    starts[offset : offset + batch_size],
                )
            ]
        )
        with torch.inference_mode():
            output, _ = model.inference(batch)
        from .v4_gates import voice_probability

        scores.extend(float(value) for value in voice_probability(output, labels))
    return scores


def _whisper_decisions(
    model: object,
    processor: object,
    paths: Sequence[Path],
    starts: Sequence[float],
    *,
    minimum_confidence: float,
    minimum_margin: float,
    batch_size: int,
) -> list[Mapping[str, object]]:
    if len(paths) != len(starts) or batch_size <= 0:
        raise V4TrackGateError("Whisper study inputs are invalid")
    results: list[Mapping[str, object]] = []
    for offset in range(0, len(paths), batch_size):
        waveforms = [
            _load_audio(
                path,
                start=start,
                seconds=WHISPER_EXCERPT_SECONDS,
                sample_rate=WHISPER_SAMPLE_RATE,
            )
            for path, start in zip(
                paths[offset : offset + batch_size],
                starts[offset : offset + batch_size],
            )
        ]
        probabilities = _language_probabilities(model, processor, waveforms)
        for row in probabilities:
            decision = decide_language(
                row,
                minimum_confidence=minimum_confidence,
                minimum_margin=minimum_margin,
            )
            results.append(
                {
                    "language": decision.language,
                    "confidence": round(decision.confidence, 8),
                    "margin": round(decision.margin, 8),
                }
            )
    return results


def build_track_gate_cache(
    *,
    metadata_root: Path,
    audio_root: Path,
    state_root: Path,
    store_root: Path,
    semantic_cache_path: Path,
    semantic_metadata_path: Path,
    plan_path: Path,
    panns_report_path: Path,
    panns_checkpoint_path: Path,
    whisper_report_path: Path,
    whisper_model_root: Path,
    existing_cache_path: Path | None,
    panns_batch_size: int,
    whisper_batch_size: int,
) -> Mapping[str, object]:
    context = load_jamendo_context(
        metadata_root, audio_root, state_root, production=True
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if (
        plan.get("plan_kind") != PLAN_KIND
        or plan.get("source_fingerprint") != context.source_fingerprint
        or plan.get("content_sha256") != _content_sha256(plan)
    ):
        raise V4TrackGateError("V4 study plan binding failed")
    track_ids = tuple(int(value) for value in plan["gate_track_ids"])
    if len(track_ids) != len(set(track_ids)) or not track_ids:
        raise V4TrackGateError("V4 study gate track identities are invalid")

    panns_report = _load_bound_report(
        panns_report_path, "soundalike_v4_panns_vocal_calibration"
    )
    whisper_report = _load_bound_report(
        whisper_report_path, "soundalike_v4_whisper_singing_language_calibration"
    )
    if (
        panns_report["source_fingerprint"] != context.source_fingerprint
        or panns_report["detector"]["checkpoint_sha256"]
        != _sha256(panns_checkpoint_path)
    ):
        raise V4TrackGateError("PANNs checkpoint or corpus binding failed")
    model_file = whisper_model_root / "model.safetensors"
    if (
        whisper_report["model"]["model_id"] != MODEL_ID
        or whisper_report["model"]["model_revision"] != MODEL_REVISION
        or whisper_report["model"]["model_file_sha256"] != _sha256(model_file)
    ):
        raise V4TrackGateError("Whisper model binding failed")
    existing_rows: Mapping[str, object] = {}
    if existing_cache_path is not None:
        existing = json.loads(existing_cache_path.read_text(encoding="utf-8"))
        if (
            existing.get("gate_kind") != GATE_KIND
            or existing.get("source_fingerprint") != context.source_fingerprint
            or existing.get("panns_calibration_sha256")
            != panns_report["content_sha256"]
            or existing.get("whisper_calibration_sha256")
            != whisper_report["content_sha256"]
            or existing.get("content_sha256") != _content_sha256(existing)
            or not isinstance(existing.get("tracks"), Mapping)
        ):
            raise V4TrackGateError("existing study gate cache binding failed")
        existing_rows = existing["tracks"]

    by_id = {int(track.track_id): track for track in context.tracks}
    if any(track_id not in by_id for track_id in track_ids):
        raise V4TrackGateError("study gate track is absent from the source corpus")
    with FullTrackStoreReader(
        store_root, expected_source_fingerprint=context.source_fingerprint
    ) as reader:
        store_ids = np.asarray(reader.track_ids, dtype=np.int64)
        store_row = {int(track_id): row for row, track_id in enumerate(store_ids)}
        if any(track_id not in store_row for track_id in track_ids):
            raise V4TrackGateError("study gate track is absent from the CLAP store")
        _, _, semantic_codes = load_semantic_cache(
            semantic_cache_path,
            semantic_metadata_path,
            expected_source_fingerprint=context.source_fingerprint,
            expected_track_ids=store_ids,
        )
        excerpts = {track_id: _excerpt(reader, track_id) for track_id in track_ids}

    pending_track_ids = [
        track_id for track_id in track_ids if str(track_id) not in existing_rows
    ]
    thresholds = VocalThresholds(
        instrumental_max=float(panns_report["thresholds"]["instrumental_max"]),
        vocal_min=float(panns_report["thresholds"]["vocal_min"]),
    )
    rows: dict[str, dict[str, object]] = {
        str(track_id): dict(existing_rows[str(track_id)])
        for track_id in track_ids
        if str(track_id) in existing_rows
    }
    vocal_track_ids = []
    if pending_track_ids:
        from panns_inference import AudioTagging, labels

        panns_model = AudioTagging(
            checkpoint_path=str(panns_checkpoint_path), device="cuda"
        )
        panns_scores = _panns_scores(
            panns_model,
            labels,
            [by_id[track_id].audio_path for track_id in pending_track_ids],
            [
                float(excerpts[track_id]["start_seconds"])
                for track_id in pending_track_ids
            ],
            batch_size=panns_batch_size,
        )
        for track_id, score in zip(pending_track_ids, panns_scores):
            semantic_state = CODE_STATES[int(semantic_codes[store_row[track_id]])]
            panns_state = classify_vocal(score, thresholds)
            vocal_state = conservative_vocal_state(semantic_state, panns_state)
            rows[str(track_id)] = {
                "semantic_vocal_state": semantic_state,
                "panns_voice_score": round(score, 8),
                "panns_vocal_state": panns_state,
                "vocal_state": vocal_state,
                "language": UNKNOWN,
                "language_confidence": 0.0,
                "language_margin": 0.0,
            }
            if vocal_state == VOCAL:
                vocal_track_ids.append(track_id)

    if vocal_track_ids:
        import torch
        from transformers import AutoProcessor, WhisperForConditionalGeneration

        processor = AutoProcessor.from_pretrained(
            whisper_model_root, local_files_only=True
        )
        whisper_model = WhisperForConditionalGeneration.from_pretrained(
            whisper_model_root,
            local_files_only=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        ).to("cuda")
        whisper_model.eval()
        language_thresholds = whisper_report["thresholds"]
        language_starts = [
            max(
                0.0,
                float(excerpts[track_id]["start_seconds"])
                + (
                    float(excerpts[track_id]["end_seconds"])
                    - float(excerpts[track_id]["start_seconds"])
                )
                / 2.0
                - WHISPER_EXCERPT_SECONDS / 2.0,
            )
            for track_id in vocal_track_ids
        ]
        decisions = _whisper_decisions(
            whisper_model,
            processor,
            [by_id[track_id].audio_path for track_id in vocal_track_ids],
            language_starts,
            minimum_confidence=float(
                language_thresholds["minimum_confidence"]
            ),
            minimum_margin=float(language_thresholds["minimum_margin"]),
            batch_size=whisper_batch_size,
        )
        for track_id, decision in zip(vocal_track_ids, decisions):
            rows[str(track_id)].update(
                {
                    "language": decision["language"],
                    "language_confidence": decision["confidence"],
                    "language_margin": decision["margin"],
                }
            )

    report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "gate_kind": GATE_KIND,
        "research_only": True,
        "promotion_allowed": False,
        "source_fingerprint": context.source_fingerprint,
        "study_plan_sha256": plan["content_sha256"],
        "panns_calibration_sha256": panns_report["content_sha256"],
        "whisper_calibration_sha256": whisper_report["content_sha256"],
        "policy": {
            "excerpt": "same strongest-recurrence section used by the evaluator",
            "vocal_state": "known only when semantic and PANNs detectors agree",
            "language": "Whisper classification only for confidently vocal tracks",
            "unknown_fallback": True,
            "transcription_saved": False,
        },
        "observed": {
            "track_count": len(rows),
            "reused_track_count": len(rows) - len(pending_track_ids),
            "new_track_count": len(pending_track_ids),
            "vocal_state_counts": {
                state: sum(row["vocal_state"] == state for row in rows.values())
                for state in (VOCAL, INSTRUMENTAL, UNKNOWN)
            },
            "known_language_count": sum(
                row["language"] != UNKNOWN for row in rows.values()
            ),
        },
        "tracks": rows,
    }
    report["content_sha256"] = _content_sha256(report)
    return report


def _write(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build V4 study detector gates.")
    for name in (
        "metadata_root",
        "audio_root",
        "state_root",
        "store_root",
        "semantic_cache",
        "semantic_cache_metadata",
        "plan",
        "panns_report",
        "panns_checkpoint",
        "whisper_report",
        "whisper_model_root",
        "output",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--panns-batch-size", type=int, default=8)
    parser.add_argument("--whisper-batch-size", type=int, default=4)
    parser.add_argument("--existing-cache", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_track_gate_cache(
        metadata_root=args.metadata_root,
        audio_root=args.audio_root,
        state_root=args.state_root,
        store_root=args.store_root,
        semantic_cache_path=args.semantic_cache,
        semantic_metadata_path=args.semantic_cache_metadata,
        plan_path=args.plan,
        panns_report_path=args.panns_report,
        panns_checkpoint_path=args.panns_checkpoint,
        whisper_report_path=args.whisper_report,
        whisper_model_root=args.whisper_model_root,
        existing_cache_path=args.existing_cache,
        panns_batch_size=args.panns_batch_size,
        whisper_batch_size=args.whisper_batch_size,
    )
    _write(args.output, report)
    print(json.dumps(report["observed"], indent=2))
    print(report["content_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
