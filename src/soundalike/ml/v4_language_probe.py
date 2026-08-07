"""Calibrate audio-derived singing-language confidence for the V4 gate."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .v4_gates import (
    UNKNOWN,
    calibrate_language_thresholds,
    decide_language,
    representative_starts,
)
from .v4_gate_probe import JAM_ALT_REVISION, _language_code


SCHEMA_VERSION = 1
PROBE_KIND = "soundalike_v4_whisper_singing_language_calibration"
SAMPLE_RATE = 16_000
EXCERPT_SECONDS = 30.0
MODEL_ID = "openai/whisper-large-v3-turbo"
MODEL_REVISION = "41f01f3fe87f28c78e2fbf8b568835947dd65ed9"


class V4LanguageProbeError(RuntimeError):
    """The V4 language probe inputs, model, or calibration are invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_audio(path: Path, start: float) -> np.ndarray:
    import librosa

    values, _ = librosa.load(
        path,
        sr=SAMPLE_RATE,
        mono=True,
        offset=start,
        duration=EXCERPT_SECONDS,
    )
    expected = int(SAMPLE_RATE * EXCERPT_SECONDS)
    if not len(values):
        raise V4LanguageProbeError(f"decoded empty audio excerpt: {path}")
    return np.pad(values, (0, max(0, expected - len(values))))[:expected]


def _language_probabilities(
    model: object,
    processor: object,
    waveforms: Sequence[np.ndarray],
) -> list[Mapping[str, float]]:
    import torch

    features = processor.feature_extractor(
        list(waveforms),
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    ).input_features.to(device=model.device, dtype=model.dtype)
    decoder_input_ids = torch.full(
        (len(waveforms), 1),
        int(model.generation_config.decoder_start_token_id),
        device=model.device,
        dtype=torch.long,
    )
    with torch.inference_mode():
        logits = model(
            input_features=features,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
        ).logits[:, -1]
    language_items = sorted(model.generation_config.lang_to_id.items())
    token_ids = torch.asarray(
        [token_id for _, token_id in language_items],
        device=model.device,
        dtype=torch.long,
    )
    probabilities = torch.softmax(logits[:, token_ids], dim=-1).float().cpu().numpy()
    result = []
    for row in probabilities:
        result.append(
            {
                token.removeprefix("<|").removesuffix("|>"): float(value)
                for (token, _), value in zip(language_items, row)
            }
        )
    return result


def build_language_report(
    *,
    jam_alt_root: Path,
    model_root: Path,
    batch_size: int,
) -> Mapping[str, object]:
    import librosa
    import torch
    from transformers import AutoProcessor, WhisperForConditionalGeneration

    metadata_path = jam_alt_root / "metadata.csv"
    rows = list(
        csv.DictReader(metadata_path.open(encoding="utf-8-sig", newline=""))
    )
    if len(rows) != 79 or batch_size <= 0:
        raise V4LanguageProbeError("Jam-ALT coverage or batch size drift")
    paths = [
        jam_alt_root
        / "subsets"
        / _language_code(row["Language"])
        / "audio"
        / row["Filepath"]
        for row in rows
    ]
    if any(not path.is_file() or path.stat().st_size < 1024 for path in paths):
        raise V4LanguageProbeError("Jam-ALT audio is missing")
    processor = AutoProcessor.from_pretrained(model_root, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_root,
        local_files_only=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to("cuda")
    model.eval()

    track_probabilities: list[dict[str, float]] = []
    pending: list[np.ndarray] = []
    owners: list[int] = []
    excerpt_probabilities: list[list[Mapping[str, float]]] = [
        [] for _ in rows
    ]

    def flush() -> None:
        if not pending:
            return
        predictions = _language_probabilities(model, processor, pending)
        for owner, probabilities in zip(owners, predictions):
            excerpt_probabilities[owner].append(probabilities)
        pending.clear()
        owners.clear()

    for index, path in enumerate(paths):
        duration = float(librosa.get_duration(path=path))
        for start in representative_starts(
            duration, excerpt_seconds=EXCERPT_SECONDS
        ):
            pending.append(_load_audio(path, start))
            owners.append(index)
            if len(pending) >= batch_size:
                flush()
    flush()
    for excerpts in excerpt_probabilities:
        if not excerpts:
            raise V4LanguageProbeError("a calibration track has no language score")
        languages = sorted({key for item in excerpts for key in item})
        averaged = {
            language: float(np.mean([item.get(language, 0.0) for item in excerpts]))
            for language in languages
        }
        total = sum(averaged.values())
        track_probabilities.append(
            {language: value / total for language, value in averaged.items()}
        )

    expected = [_language_code(row["Language"]) for row in rows]
    thresholds = calibrate_language_thresholds(
        track_probabilities,
        expected,
        minimum_accuracy=0.95,
        minimum_selected=20,
    )
    decisions = [
        decide_language(
            probabilities,
            minimum_confidence=thresholds.minimum_confidence,
            minimum_margin=thresholds.minimum_margin,
        )
        for probabilities in track_probabilities
    ]
    selected = [
        decision.language != UNKNOWN for decision in decisions
    ]
    correct = [
        decision.language == language
        for decision, language in zip(decisions, expected)
        if decision.language != UNKNOWN
    ]
    model_file = model_root / "model.safetensors"
    report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "probe_kind": PROBE_KIND,
        "research_only": True,
        "promotion_allowed": False,
        "model": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_file_sha256": _sha256(model_file),
            "model_file_bytes": model_file.stat().st_size,
            "dtype": "float16",
            "device": torch.cuda.get_device_name(0),
        },
        "calibration": {
            "dataset": "Jam-ALT",
            "revision": JAM_ALT_REVISION,
            "metadata_sha256": _sha256(metadata_path),
            "track_count": len(rows),
            "languages": sorted(set(expected)),
            "excerpt_policy": {
                "seconds": EXCERPT_SECONDS,
                "positions": [0.15, 0.5, 0.85],
                "aggregation": "mean language probability",
            },
            "minimum_precision": 0.95,
            "transcription_saved": False,
        },
        "thresholds": {
            "minimum_confidence": thresholds.minimum_confidence,
            "minimum_margin": thresholds.minimum_margin,
            "unknown_fallback": True,
        },
        "observed": {
            "coverage": float(np.mean(selected)),
            "selected_accuracy": float(np.mean(correct)) if correct else 0.0,
            "selected_tracks": int(sum(selected)),
            "unknown_tracks": int(len(selected) - sum(selected)),
            "predicted_language_counts": dict(
                sorted(Counter(decision.language for decision in decisions).items())
            ),
            "by_expected_language": {
                language: {
                    "tracks": sum(value == language for value in expected),
                    "selected": sum(
                        value == language and decision.language != UNKNOWN
                        for value, decision in zip(expected, decisions)
                    ),
                    "correct": sum(
                        value == language and decision.language == language
                        for value, decision in zip(expected, decisions)
                    ),
                }
                for language in sorted(set(expected))
            },
        },
        "tracks": [
            {
                "track_id": int(row["URL"].split("/track/")[1].split("/")[0]),
                "expected_language": language,
                "predicted_language": decision.language,
                "confidence": round(decision.confidence, 8),
                "margin": round(decision.margin, 8),
            }
            for row, language, decision in zip(rows, expected, decisions)
        ],
    }
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
    parser = argparse.ArgumentParser(description="Probe V4 singing language.")
    parser.add_argument("--jam-alt-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = build_language_report(
        jam_alt_root=args.jam_alt_root,
        model_root=args.model_root,
        batch_size=args.batch_size,
    )
    write_report(args.output, report)
    print(json.dumps({"output": str(args.output), **report["observed"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
