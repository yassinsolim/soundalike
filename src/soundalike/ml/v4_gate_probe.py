"""Run the bounded GPU probe used to calibrate the V4 vocal gate."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .jamendo_fulltrack import load_jamendo_context
from .v4_gates import (
    INSTRUMENTAL,
    UNKNOWN,
    VOCAL,
    calibrate_vocal_thresholds,
    classify_vocal,
    detector_binding,
    representative_starts,
    voice_probability,
)


PROBE_SCHEMA_VERSION = 1
PROBE_KIND = "soundalike_v4_panns_vocal_calibration"
JAM_ALT_REVISION = "28302224954ef050fe752d1628dd9bac4fc8c02b"
SAMPLE_RATE = 32_000
EXCERPT_SECONDS = 20.0
CONTROL_COUNT = 79
CONTROL_TAG_PREFIXES = (
    "genre---hiphopinstrumental",
    "genre---instrumental",
)
CONTROL_DISALLOWED_TAGS = frozenset(
    {"instrument---voice", "genre---singersongwriter"}
)


class V4GateProbeError(RuntimeError):
    """The bounded V4 gate probe cannot be reproduced safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _language_code(value: str) -> str:
    codes = {"English": "en", "French": "fr", "German": "de", "Spanish": "es"}
    try:
        return codes[value]
    except KeyError as exc:
        raise V4GateProbeError(f"unsupported Jam-ALT language: {value}") from exc


def _load_excerpt(path: Path, start: float) -> np.ndarray:
    import librosa

    values, _ = librosa.load(
        path,
        sr=SAMPLE_RATE,
        mono=True,
        offset=float(start),
        duration=EXCERPT_SECONDS,
    )
    expected = int(SAMPLE_RATE * EXCERPT_SECONDS)
    if not len(values):
        raise V4GateProbeError(f"decoded empty audio excerpt: {path}")
    return np.pad(values, (0, max(0, expected - len(values))))[:expected].astype(
        np.float32
    )


def _score_paths(
    model: object,
    labels: Sequence[str],
    paths: Sequence[Path],
    *,
    batch_size: int,
) -> list[float]:
    import librosa
    import torch

    per_track: list[float] = []
    pending: list[np.ndarray] = []
    owners: list[int] = []
    excerpts_by_track: list[list[float]] = [[] for _ in paths]

    def flush() -> None:
        if not pending:
            return
        batch = np.stack(pending)
        with torch.inference_mode():
            output, _ = model.inference(batch)
        scores = voice_probability(output, labels)
        for owner, score in zip(owners, scores):
            excerpts_by_track[owner].append(float(score))
        pending.clear()
        owners.clear()

    for index, path in enumerate(paths):
        duration = float(librosa.get_duration(path=path))
        for start in representative_starts(
            duration, excerpt_seconds=EXCERPT_SECONDS
        ):
            pending.append(_load_excerpt(path, start))
            owners.append(index)
            if len(pending) >= batch_size:
                flush()
    flush()
    for scores in excerpts_by_track:
        if not scores:
            raise V4GateProbeError("a calibration track has no detector score")
        per_track.append(max(scores))
    return per_track


def _instrumental_controls(context: object) -> tuple[object, ...]:
    candidates = []
    for track in context.tracks:
        tags = set(track.tags)
        if (
            track.fold_parts[0] in {"train", "validation"}
            and any(
                tag.startswith(prefix)
                for tag in tags
                for prefix in CONTROL_TAG_PREFIXES
            )
            and tags.isdisjoint(CONTROL_DISALLOWED_TAGS)
        ):
            order = hashlib.sha256(
                f"soundalike-v4-instrumental-control\0{track.track_id}".encode()
            ).hexdigest()
            candidates.append((order, track))
    selected = tuple(track for _, track in sorted(candidates)[:CONTROL_COUNT])
    if len(selected) != CONTROL_COUNT:
        raise V4GateProbeError("insufficient development-only instrumental controls")
    return selected


def build_probe_report(
    *,
    metadata_root: Path,
    audio_root: Path,
    state_root: Path,
    jam_alt_root: Path,
    checkpoint_path: Path,
    batch_size: int,
) -> Mapping[str, object]:
    from panns_inference import AudioTagging, labels

    context = load_jamendo_context(
        metadata_root, audio_root, state_root, production=True
    )
    metadata_path = jam_alt_root / "metadata.csv"
    rows = list(
        csv.DictReader(metadata_path.open(encoding="utf-8-sig", newline=""))
    )
    if len(rows) != CONTROL_COUNT:
        raise V4GateProbeError("Jam-ALT calibration coverage drift")
    positive_paths = [
        jam_alt_root
        / "subsets"
        / _language_code(row["Language"])
        / "audio"
        / row["Filepath"]
        for row in rows
    ]
    if any(not path.is_file() or path.stat().st_size < 1024 for path in positive_paths):
        raise V4GateProbeError("Jam-ALT audio is missing or still a symlink stub")
    controls = _instrumental_controls(context)
    model = AudioTagging(checkpoint_path=str(checkpoint_path), device="cuda")
    vocal_scores = _score_paths(
        model, labels, positive_paths, batch_size=batch_size
    )
    instrumental_scores = _score_paths(
        model,
        labels,
        [track.audio_path for track in controls],
        batch_size=batch_size,
    )
    thresholds = calibrate_vocal_thresholds(vocal_scores, instrumental_scores)
    positive_states = [
        classify_vocal(score, thresholds) for score in vocal_scores
    ]
    negative_states = [
        classify_vocal(score, thresholds) for score in instrumental_scores
    ]
    report: dict[str, object] = {
        "schema_version": PROBE_SCHEMA_VERSION,
        "probe_kind": PROBE_KIND,
        "research_only": True,
        "promotion_allowed": False,
        "source_fingerprint": context.source_fingerprint,
        "detector": {
            "name": "PANNs Cnn14 AudioSet",
            "checkpoint_path_name": checkpoint_path.name,
            "checkpoint_sha256": _sha256(checkpoint_path),
            "code_license": "MIT",
            "training_code_license_url": (
                "https://github.com/qiuqiangkong/audioset_tagging_cnn/"
                "blob/master/LICENSE.MIT"
            ),
            "inference_code_license_url": (
                "https://github.com/qiuqiangkong/panns_inference/"
                "blob/master/LICENSE.MIT"
            ),
        },
        "calibration": {
            "known_vocal": {
                "dataset": "Jam-ALT",
                "revision": JAM_ALT_REVISION,
                "metadata_sha256": _sha256(metadata_path),
                "track_count": len(rows),
                "languages": sorted({row["Language"] for row in rows}),
                "scores": [
                    {
                        "track_id": int(row["URL"].split("/track/")[1].split("/")[0]),
                        "language": row["Language"],
                        "license": row["LicenseType"],
                        "score": round(score, 8),
                    }
                    for row, score in zip(rows, vocal_scores)
                ],
            },
            "known_instrumental": {
                "selection": (
                    "deterministic fold-0 train/validation tracks carrying an "
                    "instrumental genre tag and no voice/singer-songwriter tag"
                ),
                "track_count": len(controls),
                "reserve_labels_accessed": False,
                "scores": [
                    {
                        "track_id": int(track.track_id),
                        "artist_id": int(track.artist_id),
                        "fold_part": track.fold_parts[0],
                        "tags": sorted(
                            tag
                            for tag in track.tags
                            if any(
                                tag.startswith(prefix)
                                for prefix in CONTROL_TAG_PREFIXES
                            )
                        ),
                        "score": round(score, 8),
                    }
                    for track, score in zip(controls, instrumental_scores)
                ],
            },
            "excerpt_policy": {
                "seconds": EXCERPT_SECONDS,
                "positions": [0.15, 0.5, 0.85],
                "aggregation": "maximum voice probability",
            },
            "maximum_false_exclusion_rate": 0.05,
        },
        "thresholds": {
            "instrumental_max": thresholds.instrumental_max,
            "vocal_min": thresholds.vocal_min,
            "middle_state": UNKNOWN,
        },
        "observed": {
            "known_vocal_states": {
                state: positive_states.count(state)
                for state in (VOCAL, UNKNOWN, INSTRUMENTAL)
            },
            "known_instrumental_states": {
                state: negative_states.count(state)
                for state in (VOCAL, UNKNOWN, INSTRUMENTAL)
            },
            "known_vocal_score_median": float(np.median(vocal_scores)),
            "known_instrumental_score_median": float(
                np.median(instrumental_scores)
            ),
        },
    }
    report["binding"] = detector_binding(checkpoint_path, report)
    report["content_sha256"] = hashlib.sha256(
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return report


def write_report(path: Path, report: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe the V4 vocal gate.")
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--jam-alt-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_probe_report(
        metadata_root=args.metadata_root,
        audio_root=args.audio_root,
        state_root=args.state_root,
        jam_alt_root=args.jam_alt_root,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
    )
    write_report(args.output, report)
    print(json.dumps({"output": str(args.output), **report["observed"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
