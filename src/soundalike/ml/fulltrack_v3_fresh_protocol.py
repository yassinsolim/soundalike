"""Freeze a fresh V3 protocol after the first scaled shadow was consumed."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Set, Tuple

from .fulltrack_store import sha256_path, stable_json_sha256
from .fulltrack_v3 import SOURCE_FINGERPRINT
from .fulltrack_v3_protocol import load_protocol as load_consumed_protocol
from .jamendo_fulltrack import (
    EVIDENCE_SCOPE,
    JamendoContext,
    JamendoTrack,
    JamendoValidationError,
    load_jamendo_context,
)


PROTOCOL_SCHEMA_VERSION = 1
PROTOCOL_KIND = "v3_fresh_artist_disjoint_scaled_supervision_protocol"
BASE_FOLD = 0
BASE_PART = "train"
FRESH_SPLIT_SEED = 20260809
TRACK_ORDER_SEED = 20260810
BUCKETS = 10_000
DEVELOPMENT_CUTOFF = 5_000
TRACK_COUNT = 32_859
EXPECTED_SELECTION_SHA256 = (
    "788354f41e6588cd3c41a59167d9ade0274cefb0907033819595db9609823ae6"
)
CONSUMED_PROTOCOL_FILE_SHA256 = (
    "f00a67569967dba9a2f16bfd8effcb93729847618f1270e98d245ac74c184901"
)
CONSUMED_PROTOCOL_PAYLOAD_SHA256 = (
    "d697240384003ba1a7d9e00d281462b005a3e02abac49964cd3ba4e128292738"
)
CONSUMED_SHADOW_RESULT_PAYLOAD_SHA256 = (
    "17d0a1ecc9c997c2b44d8bd40f2fda777d73fa916fe54870b1d59fc16e498c76"
)
EXPECTED_SPLITS = {
    "train": {
        "tracks": 31_473,
        "artists": 1_577,
        "track_ids_sha256": (
            "9a3d988433a8dacd915d27248bbfa56ce3f70eee14d5ce22970e80f357e274d2"
        ),
    },
    "development": {
        "tracks": 742,
        "artists": 297,
        "track_ids_sha256": (
            "cb3573f18648733f1b2292f2dddf037996e6a4d9c2d3bde0723350a77c6a2ffb"
        ),
    },
    "shadow": {
        "tracks": 644,
        "artists": 265,
        "track_ids_sha256": (
            "d6908d7de0604031017dd2d74c731b6770be3d0d9cc630cd94b303f3beffedfb"
        ),
    },
}


class V3FreshProtocolError(RuntimeError):
    """Invalid, overlapping, changed, or prematurely opened fresh protocol."""


def _payload_sha256(document: Mapping[str, object]) -> str:
    payload = dict(document)
    payload.pop("payload_sha256", None)
    return stable_json_sha256(payload)


def fresh_artist_split(artist_id: int, consumed_artists: Set[int]) -> str:
    if (
        isinstance(artist_id, bool)
        or not isinstance(artist_id, int)
        or artist_id <= 0
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in consumed_artists
        )
    ):
        raise V3FreshProtocolError("artist IDs must be positive integers")
    if artist_id in consumed_artists:
        return "train"
    bucket = (
        int(
            stable_json_sha256(
                {"seed": FRESH_SPLIT_SEED, "artist_id": artist_id}
            )[:16],
            16,
        )
        % BUCKETS
    )
    return "development" if bucket < DEVELOPMENT_CUTOFF else "shadow"


def order_tracks(tracks: Sequence[JamendoTrack]) -> Tuple[JamendoTrack, ...]:
    ordered = tuple(
        sorted(
            tracks,
            key=lambda track: stable_json_sha256(
                {"seed": TRACK_ORDER_SEED, "track_id": track.track_id}
            ),
        )
    )
    if len(ordered) != TRACK_COUNT:
        raise V3FreshProtocolError("fresh protocol track count drift")
    if len({track.track_id for track in ordered}) != len(ordered):
        raise V3FreshProtocolError("fresh protocol contains duplicate tracks")
    return ordered


def _read_consumed_result(path: Path) -> Mapping[str, object]:
    candidate = Path(path).absolute()
    if candidate.is_symlink():
        raise V3FreshProtocolError("consumed result may not be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
        document = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V3FreshProtocolError(f"consumed result is invalid: {exc}") from exc
    if (
        not isinstance(document, dict)
        or document.get("artifact_kind") != "v3_complementary_shadow_audit"
        or document.get("payload_sha256") != _payload_sha256(document)
        or document.get("payload_sha256")
        != CONSUMED_SHADOW_RESULT_PAYLOAD_SHA256
        or document.get("shadow_labels_accessed") is not True
        or document.get("shadow_gate", {}).get("automated_passed") is not False
        or document.get("promotion_allowed") is not False
    ):
        raise V3FreshProtocolError("consumed result envelope drift")
    return document


def _eligible_tracks(context: JamendoContext) -> Tuple[JamendoTrack, ...]:
    if context.source_fingerprint != SOURCE_FINGERPRINT:
        raise V3FreshProtocolError("Jamendo source fingerprint drift")
    fold = next((item for item in context.folds if item.index == BASE_FOLD), None)
    if fold is None:
        raise V3FreshProtocolError("base fold is missing")
    return tuple(
        track
        for track in context.tracks
        if fold.track_parts.get(track.track_id) == BASE_PART
    )


def build_fresh_protocol(
    context: JamendoContext,
    consumed_protocol_path: Path,
    consumed_result_path: Path,
) -> Dict[str, object]:
    if sha256_path(Path(consumed_protocol_path)) != CONSUMED_PROTOCOL_FILE_SHA256:
        raise V3FreshProtocolError("consumed protocol file hash drift")
    consumed = load_consumed_protocol(Path(consumed_protocol_path))
    if consumed.get("payload_sha256") != CONSUMED_PROTOCOL_PAYLOAD_SHA256:
        raise V3FreshProtocolError("consumed protocol payload drift")
    result = _read_consumed_result(Path(consumed_result_path))
    consumed_entries = consumed.get("tracks")
    if not isinstance(consumed_entries, list):
        raise V3FreshProtocolError("consumed protocol track plan drift")
    consumed_artists = {int(entry["artist_id"]) for entry in consumed_entries}
    ordered = order_tracks(_eligible_tracks(context))
    selection_hash = stable_json_sha256(
        tuple(track.track_id for track in ordered)
    )
    if selection_hash != EXPECTED_SELECTION_SHA256:
        raise V3FreshProtocolError("fresh track selection drift")
    entries = [
        {
            "track_id": track.track_id,
            "artist_id": track.artist_id,
            "source_sha256": track.expected_audio_sha256,
            "split": fresh_artist_split(track.artist_id, consumed_artists),
        }
        for track in ordered
    ]
    split_summary: Dict[str, object] = {}
    artist_sets = {}
    for split, expected in EXPECTED_SPLITS.items():
        split_entries = [entry for entry in entries if entry["split"] == split]
        track_ids = tuple(sorted(int(entry["track_id"]) for entry in split_entries))
        artists = {int(entry["artist_id"]) for entry in split_entries}
        summary = {
            "tracks": len(split_entries),
            "artists": len(artists),
            "track_ids_sha256": stable_json_sha256(track_ids),
        }
        if summary != expected:
            raise V3FreshProtocolError(f"{split} split drift: {summary}")
        split_summary[split] = summary
        artist_sets[split] = artists
    if artist_sets["train"] != consumed_artists:
        raise V3FreshProtocolError("training artists differ from consumed universe")
    if any(
        artist_sets[left].intersection(artist_sets[right])
        for left, right in (
            ("train", "development"),
            ("train", "shadow"),
            ("development", "shadow"),
        )
    ):
        raise V3FreshProtocolError("fresh protocol contains artist leakage")
    protocol: Dict[str, object] = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "artifact_kind": PROTOCOL_KIND,
        "evidence_scope": EVIDENCE_SCOPE,
        "source_fingerprint": SOURCE_FINGERPRINT,
        "base_fold": BASE_FOLD,
        "base_part": BASE_PART,
        "track_order_seed": TRACK_ORDER_SEED,
        "fresh_split_seed": FRESH_SPLIT_SEED,
        "selection_sha256": selection_hash,
        "split_summary": split_summary,
        "tracks": entries,
        "consumed_artist_count": len(consumed_artists),
        "consumed_protocol_file_sha256": sha256_path(Path(consumed_protocol_path)),
        "consumed_protocol_payload_sha256": consumed["payload_sha256"],
        "consumed_shadow_result_file_sha256": sha256_path(
            Path(consumed_result_path)
        ),
        "consumed_shadow_result_payload_sha256": result["payload_sha256"],
        "shadow_labels_accessed": False,
        "promotion_allowed": False,
    }
    protocol["payload_sha256"] = stable_json_sha256(protocol)
    return protocol


def validate_fresh_protocol(
    document: object,
    *,
    context: Optional[JamendoContext] = None,
) -> Mapping[str, object]:
    if not isinstance(document, dict):
        raise V3FreshProtocolError("fresh protocol must be a JSON object")
    if document.get("payload_sha256") != _payload_sha256(document):
        raise V3FreshProtocolError("fresh protocol payload checksum mismatch")
    if (
        document.get("schema_version") != PROTOCOL_SCHEMA_VERSION
        or document.get("artifact_kind") != PROTOCOL_KIND
        or document.get("evidence_scope") != EVIDENCE_SCOPE
        or document.get("source_fingerprint") != SOURCE_FINGERPRINT
        or document.get("base_fold") != BASE_FOLD
        or document.get("base_part") != BASE_PART
        or document.get("track_order_seed") != TRACK_ORDER_SEED
        or document.get("fresh_split_seed") != FRESH_SPLIT_SEED
        or document.get("selection_sha256") != EXPECTED_SELECTION_SHA256
        or document.get("split_summary") != EXPECTED_SPLITS
        or document.get("consumed_artist_count")
        != EXPECTED_SPLITS["train"]["artists"]
        or document.get("consumed_protocol_file_sha256")
        != CONSUMED_PROTOCOL_FILE_SHA256
        or document.get("consumed_protocol_payload_sha256")
        != CONSUMED_PROTOCOL_PAYLOAD_SHA256
        or document.get("consumed_shadow_result_payload_sha256")
        != CONSUMED_SHADOW_RESULT_PAYLOAD_SHA256
        or document.get("shadow_labels_accessed") is not False
        or document.get("promotion_allowed") is not False
    ):
        raise V3FreshProtocolError("fresh protocol envelope drift")
    entries = document.get("tracks")
    if not isinstance(entries, list) or len(entries) != TRACK_COUNT:
        raise V3FreshProtocolError("fresh protocol track plan drift")
    required = {"track_id", "artist_id", "source_sha256", "split"}
    track_ids = []
    split_artists = {split: set() for split in EXPECTED_SPLITS}
    split_track_ids = {split: [] for split in EXPECTED_SPLITS}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != required:
            raise V3FreshProtocolError("fresh protocol track entry schema drift")
        track_id = entry["track_id"]
        artist_id = entry["artist_id"]
        source_sha256 = entry["source_sha256"]
        split = entry["split"]
        if (
            isinstance(track_id, bool)
            or not isinstance(track_id, int)
            or track_id <= 0
            or isinstance(artist_id, bool)
            or not isinstance(artist_id, int)
            or artist_id <= 0
            or not isinstance(source_sha256, str)
            or len(source_sha256) != 64
            or any(character not in "0123456789abcdef" for character in source_sha256)
            or split not in EXPECTED_SPLITS
        ):
            raise V3FreshProtocolError("invalid fresh protocol track entry")
        track_ids.append(track_id)
        split_artists[split].add(artist_id)
        split_track_ids[split].append(track_id)
    actual_splits = {
        split: {
            "tracks": len(split_track_ids[split]),
            "artists": len(split_artists[split]),
            "track_ids_sha256": stable_json_sha256(
                tuple(sorted(split_track_ids[split]))
            ),
        }
        for split in EXPECTED_SPLITS
    }
    if (
        len(set(track_ids)) != TRACK_COUNT
        or stable_json_sha256(tuple(track_ids)) != EXPECTED_SELECTION_SHA256
        or actual_splits != EXPECTED_SPLITS
        or any(
            split_artists[left].intersection(split_artists[right])
            for left, right in (
                ("train", "development"),
                ("train", "shadow"),
                ("development", "shadow"),
            )
        )
    ):
        raise V3FreshProtocolError("fresh protocol identity or leakage drift")
    if context is not None:
        if context.source_fingerprint != SOURCE_FINGERPRINT:
            raise V3FreshProtocolError("context source fingerprint drift")
        tracks_by_id = context.by_track_id
        for entry in entries:
            track = tracks_by_id.get(int(entry["track_id"]))
            if (
                track is None
                or track.artist_id != entry["artist_id"]
                or track.expected_audio_sha256 != entry["source_sha256"]
            ):
                raise V3FreshProtocolError(
                    f"fresh protocol/source mismatch for track {entry['track_id']}"
                )
    return document


def load_fresh_protocol(
    path: Path,
    *,
    context: Optional[JamendoContext] = None,
) -> Mapping[str, object]:
    candidate = Path(path).absolute()
    if candidate.is_symlink():
        raise V3FreshProtocolError("fresh protocol path may not be a symlink")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file() or resolved.stat().st_size > 16 * 1024 * 1024:
        raise V3FreshProtocolError("fresh protocol file is missing or too large")
    try:
        document = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V3FreshProtocolError(f"invalid fresh protocol JSON: {exc}") from exc
    return validate_fresh_protocol(document, context=context)


def _write_protocol(path: Path, protocol: Mapping[str, object]) -> None:
    target = Path(path).absolute()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(protocol, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--state-root", required=True)
    parser.add_argument("--consumed-protocol", required=True)
    parser.add_argument("--consumed-result", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        context = load_jamendo_context(
            Path(args.metadata_root),
            Path(args.audio_root),
            Path(args.state_root),
            production=True,
        )
        protocol = build_fresh_protocol(
            context,
            Path(args.consumed_protocol),
            Path(args.consumed_result),
        )
        _write_protocol(Path(args.output), protocol)
    except (
        JamendoValidationError,
        OSError,
        ValueError,
        V3FreshProtocolError,
    ) as exc:
        raise SystemExit(f"fresh V3 protocol blocked: {exc}") from exc
    print(
        json.dumps(
            {
                "output": str(Path(args.output).absolute()),
                "payload_sha256": protocol["payload_sha256"],
                "selection_sha256": protocol["selection_sha256"],
                "split_summary": protocol["split_summary"],
                "shadow_labels_accessed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
