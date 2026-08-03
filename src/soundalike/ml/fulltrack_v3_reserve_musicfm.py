"""Plan and resume sparse-four MusicFM extraction for the final V3 reserve."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

from .fulltrack_extract import (
    ExtractionConfig,
    PyAVAudioDecoder,
    iter_overlapping_windows,
    normalize_rows,
)
from .fulltrack_musicfm import (
    DEFAULT_ASSET_ROOT,
    HOP_SECONDS,
    SAMPLE_RATE,
    WINDOW_SECONDS,
    FrozenMusicFMAdapter,
)
from .fulltrack_store import (
    FullTrackStore,
    TrackArtifacts,
    stable_json_sha256,
)
from .fulltrack_v3_reserve_protocol import (
    EXPECTED_SELECTION_SHA256,
    EXPECTED_SPLITS,
    load_reserve_protocol,
)
from .jamendo_fulltrack import (
    load_jamendo_context,
    sha256_file,
)


PLAN_KIND = "v3_final_reserve_musicfm_sparse_four_plan"
EXPERIMENT = "musicfm_fma_reserve_sparse_four_anchor"
ANCHORS = 4
SHARD_TRACKS = 32


class V3ReserveMusicFMError(RuntimeError):
    """Invalid or changed final-reserve MusicFM extraction."""


def _payload_sha256(document: Mapping[str, object]) -> str:
    payload = dict(document)
    payload.pop("payload_sha256", None)
    return stable_json_sha256(payload)


def _read_json(path: Path, label: str) -> Mapping[str, object]:
    candidate = Path(path).absolute()
    if candidate.is_symlink() or not candidate.is_file():
        raise V3ReserveMusicFMError(f"{label} must be a concrete file")
    try:
        document = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V3ReserveMusicFMError(f"{label} is invalid: {exc}") from exc
    if not isinstance(document, dict):
        raise V3ReserveMusicFMError(f"{label} must contain a JSON object")
    return document


def _extraction_config() -> ExtractionConfig:
    config = ExtractionConfig(
        sample_rate=SAMPLE_RATE,
        window_seconds=WINDOW_SECONDS,
        hop_seconds=HOP_SECONDS,
        model_batch_size=2,
        repetition_sections=ANCHORS,
        salient_sections=ANCHORS,
        shard_tracks=SHARD_TRACKS,
    )
    config.validate()
    return config


def _binding(
    protocol: Mapping[str, object],
    *,
    split: str,
    config: ExtractionConfig,
    model_binding: Mapping[str, object],
) -> Mapping[str, object]:
    return {
        "schema_version": 1,
        "experiment": EXPERIMENT,
        "anchors": ANCHORS,
        "extraction": config.as_dict(),
        "model": model_binding,
        "track_protocol": {
            "artifact_kind": protocol["artifact_kind"],
            "payload_sha256": protocol["payload_sha256"],
            "selection_sha256": protocol["selection_sha256"],
            "split": split,
            "track_ids_sha256": protocol["split_summary"][split][
                "track_ids_sha256"
            ],
        },
        "promotion_allowed": False,
    }


def build_extraction_plan(
    *,
    metadata_root: Path,
    audio_root: Path,
    state_root: Path,
    protocol_path: Path,
    split: str,
    asset_root: Path,
    output: Path,
) -> Mapping[str, object]:
    if split not in ("development", "shadow"):
        raise V3ReserveMusicFMError("only development or shadow may be extracted")
    if Path(output).exists():
        raise V3ReserveMusicFMError("extraction plan already exists")
    context = load_jamendo_context(
        Path(metadata_root),
        Path(audio_root),
        Path(state_root),
        production=True,
    )
    protocol = load_reserve_protocol(Path(protocol_path), context=context)
    if (
        protocol.get("selection_sha256") != EXPECTED_SELECTION_SHA256
        or protocol.get("shadow_labels_accessed") is not False
    ):
        raise V3ReserveMusicFMError("reserve protocol binding drift")
    entries = tuple(
        item for item in protocol["tracks"] if item["split"] == split
    )
    if len(entries) != int(EXPECTED_SPLITS[split]["tracks"]):
        raise V3ReserveMusicFMError("reserve split size drift")
    encoder = FrozenMusicFMAdapter(Path(asset_root))
    config = _extraction_config()
    binding = _binding(
        protocol,
        split=split,
        config=config,
        model_binding=encoder.capability.binding(),
    )
    document: Dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": PLAN_KIND,
        "source_fingerprint": context.source_fingerprint,
        "protocol_payload_sha256": protocol["payload_sha256"],
        "protocol_selection_sha256": protocol["selection_sha256"],
        "split": split,
        "tracks": len(entries),
        "track_ids_sha256": EXPECTED_SPLITS[split]["track_ids_sha256"],
        "binding": binding,
        "config_sha256": stable_json_sha256(binding),
        "model_id": encoder.model_id,
        "model_sha256": encoder.checkpoint_sha256,
        "embedding_dim": encoder.embedding_dim,
        "opened_label_splits": [],
        "shadow_labels_accessed": False,
        "promotion_allowed": False,
    }
    document["payload_sha256"] = stable_json_sha256(document)
    target = Path(output).absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return document


def sparse_artifacts(
    track,
    decoder: PyAVAudioDecoder,
    encoder: FrozenMusicFMAdapter,
    config: ExtractionConfig,
) -> TrackArtifacts:
    if sha256_file(track.audio_path) != track.expected_audio_sha256:
        raise V3ReserveMusicFMError(
            f"source SHA-256 drift for track {track.track_id}"
        )
    windows = tuple(
        iter_overlapping_windows(
            decoder.decode(
                track.audio_path,
                sample_rate=config.sample_rate,
                chunk_samples=config.decoder_chunk_samples,
            ),
            window_samples=config.window_samples,
            hop_samples=config.hop_samples,
            short_track_policy=config.short_track_policy,
            max_chunk_samples=config.decoder_chunk_samples,
            max_windows=config.max_windows_per_track,
        )
    )
    if not windows:
        raise V3ReserveMusicFMError(
            f"track {track.track_id} produced no windows"
        )
    selected = np.unique(
        np.linspace(0, len(windows) - 1, min(ANCHORS, len(windows))).astype(
            np.int32
        )
    )
    encoded = []
    for start in range(0, len(selected), encoder.max_batch_size):
        batch = np.stack(
            [
                windows[int(index)].samples
                for index in selected[start : start + encoder.max_batch_size]
            ]
        )
        encoded.append(encoder.embed_windows(batch))
    embeddings = normalize_rows(np.concatenate(encoded, axis=0))
    global_embedding = normalize_rows(
        np.mean(embeddings, axis=0, keepdims=True)
    )[0]
    starts = np.asarray(
        [windows[int(index)].start_sample for index in selected],
        dtype=np.int64,
    )
    decoded_samples = (
        windows[0].valid_samples
        if len(windows) == 1
        else windows[-1].start_sample + config.window_samples
    )
    indices = np.arange(len(embeddings), dtype=np.int64)
    return TrackArtifacts(
        global_embedding=global_embedding,
        window_embeddings=embeddings,
        window_starts=starts,
        repeated_sections=embeddings,
        salient_sections=embeddings,
        repeated_indices=indices,
        salient_indices=indices,
        decoded_samples=decoded_samples,
    )


def run_extraction(
    *,
    metadata_root: Path,
    audio_root: Path,
    state_root: Path,
    protocol_path: Path,
    plan_path: Path,
    asset_root: Path,
    output: Path,
    max_tracks: Optional[int] = None,
) -> Mapping[str, object]:
    plan = _read_json(plan_path, "MusicFM extraction plan")
    if (
        plan.get("artifact_kind") != PLAN_KIND
        or plan.get("payload_sha256") != _payload_sha256(plan)
        or plan.get("shadow_labels_accessed") is not False
    ):
        raise V3ReserveMusicFMError("MusicFM extraction plan binding drift")
    context = load_jamendo_context(
        Path(metadata_root),
        Path(audio_root),
        Path(state_root),
        production=True,
    )
    protocol = load_reserve_protocol(Path(protocol_path), context=context)
    split = str(plan["split"])
    entries = tuple(
        item for item in protocol["tracks"] if item["split"] == split
    )
    encoder = FrozenMusicFMAdapter(Path(asset_root))
    config = _extraction_config()
    binding = _binding(
        protocol,
        split=split,
        config=config,
        model_binding=encoder.capability.binding(),
    )
    expected = {
        "source_fingerprint": context.source_fingerprint,
        "protocol_payload_sha256": protocol["payload_sha256"],
        "protocol_selection_sha256": protocol["selection_sha256"],
        "tracks": len(entries),
        "track_ids_sha256": EXPECTED_SPLITS[split]["track_ids_sha256"],
        "binding": binding,
        "config_sha256": stable_json_sha256(binding),
        "model_id": encoder.model_id,
        "model_sha256": encoder.checkpoint_sha256,
        "embedding_dim": encoder.embedding_dim,
    }
    drift = {
        key: (value, plan.get(key))
        for key, value in expected.items()
        if plan.get(key) != value
    }
    if drift:
        raise V3ReserveMusicFMError(f"MusicFM extraction plan drift: {drift}")
    tracks = tuple(
        context.by_track_id[int(item["track_id"])] for item in entries
    )
    decoder = PyAVAudioDecoder()
    started = time.perf_counter()
    processed = 0
    with FullTrackStore(
        Path(output),
        track_ids=[track.track_id for track in tracks],
        source_hashes=[track.expected_audio_sha256 for track in tracks],
        source_fingerprint=context.source_fingerprint,
        config_sha256=str(plan["config_sha256"]),
        model_sha256=encoder.checkpoint_sha256,
        model_id=encoder.model_id,
        embedding_dim=encoder.embedding_dim,
        shard_tracks=SHARD_TRACKS,
        repetition_sections=ANCHORS,
        salient_sections=ANCHORS,
    ) as store:
        pending = set(store.pending_track_ids())
        initial_completed = store.completed_count
        for track in tracks:
            if track.track_id not in pending:
                continue
            store.write_track(
                track.track_id,
                track.expected_audio_sha256,
                sparse_artifacts(track, decoder, encoder, config),
            )
            processed += 1
            if processed % 16 == 0:
                elapsed = time.perf_counter() - started
                rate = processed / elapsed
                print(
                    json.dumps(
                        {
                            "completed": initial_completed + processed,
                            "total": len(tracks),
                            "tracks_per_second": rate,
                            "eta_seconds": (
                                len(tracks) - initial_completed - processed
                            )
                            / rate,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if max_tracks is not None and processed >= max_tracks:
                break
        store.flush()
        if store.pending_count:
            result: Mapping[str, object] = {
                "sealed": False,
                "completed": store.completed_count,
                "pending": store.pending_count,
            }
        else:
            manifest = store.seal()
            result = {
                "sealed": True,
                "completed": store.completed_count,
                "manifest_sha256": manifest["manifest_sha256"],
            }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    for name in (
        "metadata-root",
        "audio-root",
        "state-root",
        "protocol",
        "split",
        "asset-root",
        "output",
    ):
        plan.add_argument(f"--{name}", required=True)
    extract = commands.add_parser("extract")
    for name in (
        "metadata-root",
        "audio-root",
        "state-root",
        "protocol",
        "plan",
        "asset-root",
        "output",
    ):
        extract.add_argument(f"--{name}", required=True)
    extract.add_argument("--max-tracks", type=int)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "plan":
        result = build_extraction_plan(
            metadata_root=Path(args.metadata_root),
            audio_root=Path(args.audio_root),
            state_root=Path(args.state_root),
            protocol_path=Path(args.protocol),
            split=args.split,
            asset_root=Path(args.asset_root),
            output=Path(args.output),
        )
        print(
            json.dumps(
                {
                    "output": str(Path(args.output).absolute()),
                    "payload_sha256": result["payload_sha256"],
                    "config_sha256": result["config_sha256"],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        run_extraction(
            metadata_root=Path(args.metadata_root),
            audio_root=Path(args.audio_root),
            state_root=Path(args.state_root),
            protocol_path=Path(args.protocol),
            plan_path=Path(args.plan),
            asset_root=Path(args.asset_root),
            output=Path(args.output),
            max_tracks=args.max_tracks,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
