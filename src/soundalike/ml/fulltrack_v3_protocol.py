"""Freeze and validate the artist-disjoint V3 scale protocol."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

from .fulltrack_store import stable_json_sha256
from .fulltrack_v3 import SOURCE_FINGERPRINT
from .jamendo_fulltrack import (
    EVIDENCE_SCOPE,
    JamendoContext,
    JamendoTrack,
    JamendoValidationError,
    load_jamendo_context,
)


PROTOCOL_SCHEMA_VERSION = 1
PROTOCOL_KIND = "v3_artist_disjoint_semantic_head_scale_protocol"
BASE_FOLD = 0
BASE_PART = "train"
TRACK_SELECTION_SEED = 20260802
ARTIST_SPLIT_SEED = 20260803
TRACK_LIMIT = 8_192
BUCKETS = 10_000
TRAIN_CUTOFF = 7_000
DEVELOPMENT_CUTOFF = 8_500
EXPECTED_SELECTION_SHA256 = (
    "8097b84f87c9f662157fe19bd93c00f51ea6703e0db964b864ecb188ae5481fe"
)
EXPECTED_SPLITS = {
    "train": {
        "tracks": 5_864,
        "artists": 1_119,
        "track_ids_sha256": (
            "357639db357de2ec464e01f683bbd401503b195b6c927db87dce610ba169a7f6"
        ),
    },
    "development": {
        "tracks": 1_134,
        "artists": 229,
        "track_ids_sha256": (
            "94dcd748a1b5d758029f486b5bf2a76c1c15a3e4a26015ab12b2c9b4fcfdd2f5"
        ),
    },
    "shadow": {
        "tracks": 1_194,
        "artists": 229,
        "track_ids_sha256": (
            "a029fa7b3d66df281f1d76fdf7dfabf6e2af043bda5591188a7f02b498196424"
        ),
    },
}


class V3ProtocolError(RuntimeError):
    """Invalid, changed, overlapping, or prematurely accessed V3 protocol."""


def artist_split(artist_id: int) -> str:
    if isinstance(artist_id, bool) or not isinstance(artist_id, int) or artist_id <= 0:
        raise V3ProtocolError("artist ID must be a positive integer")
    bucket = (
        int(
            stable_json_sha256(
                {"seed": ARTIST_SPLIT_SEED, "artist_id": artist_id}
            )[:16],
            16,
        )
        % BUCKETS
    )
    if bucket < TRAIN_CUTOFF:
        return "train"
    if bucket < DEVELOPMENT_CUTOFF:
        return "development"
    return "shadow"


def select_tracks(
    tracks: Sequence[JamendoTrack],
    *,
    limit: int = TRACK_LIMIT,
    seed: int = TRACK_SELECTION_SEED,
) -> Tuple[JamendoTrack, ...]:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise V3ProtocolError("track limit must be a positive integer")
    if len(tracks) < limit:
        raise V3ProtocolError("not enough tracks for the frozen protocol")
    selected = tuple(
        sorted(
            tracks,
            key=lambda track: stable_json_sha256(
                {"seed": seed, "track_id": track.track_id}
            ),
        )[:limit]
    )
    if len({track.track_id for track in selected}) != len(selected):
        raise V3ProtocolError("track selection contains duplicate IDs")
    return selected


def build_protocol(context: JamendoContext) -> Dict[str, object]:
    if context.source_fingerprint != SOURCE_FINGERPRINT:
        raise V3ProtocolError("Jamendo source fingerprint drift")
    fold = next(
        (item for item in context.folds if item.index == BASE_FOLD),
        None,
    )
    if fold is None:
        raise V3ProtocolError("base fold is missing")
    eligible = [
        track
        for track in context.tracks
        if fold.track_parts.get(track.track_id) == BASE_PART
    ]
    selected = select_tracks(eligible)
    selection_hash = stable_json_sha256(
        tuple(track.track_id for track in selected)
    )
    if selection_hash != EXPECTED_SELECTION_SHA256:
        raise V3ProtocolError("frozen track selection drift")
    entries = [
        {
            "track_id": track.track_id,
            "artist_id": track.artist_id,
            "source_sha256": track.expected_audio_sha256,
            "split": artist_split(track.artist_id),
        }
        for track in selected
    ]
    split_summary = {}
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
            raise V3ProtocolError(f"{split} split drift: {summary}")
        split_summary[split] = summary
        artist_sets[split] = artists
    for left, right in (
        ("train", "development"),
        ("train", "shadow"),
        ("development", "shadow"),
    ):
        if artist_sets[left].intersection(artist_sets[right]):
            raise V3ProtocolError(f"{left}/{right} artist leakage")
    protocol: Dict[str, object] = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "artifact_kind": PROTOCOL_KIND,
        "evidence_scope": EVIDENCE_SCOPE,
        "source_fingerprint": SOURCE_FINGERPRINT,
        "base_fold": BASE_FOLD,
        "base_part": BASE_PART,
        "track_selection_seed": TRACK_SELECTION_SEED,
        "artist_split_seed": ARTIST_SPLIT_SEED,
        "track_limit": TRACK_LIMIT,
        "selection_sha256": selection_hash,
        "split_summary": split_summary,
        "tracks": entries,
        "shadow_labels_accessed": False,
        "promotion_allowed": False,
    }
    protocol["payload_sha256"] = stable_json_sha256(protocol)
    return protocol


def validate_protocol(
    document: object,
    *,
    context: Optional[JamendoContext] = None,
) -> Mapping[str, object]:
    if not isinstance(document, dict):
        raise V3ProtocolError("protocol must be a JSON object")
    declared = document.pop("payload_sha256", None)
    actual = stable_json_sha256(document)
    document["payload_sha256"] = declared
    if declared != actual:
        raise V3ProtocolError("protocol payload checksum mismatch")
    if (
        document.get("schema_version") != PROTOCOL_SCHEMA_VERSION
        or document.get("artifact_kind") != PROTOCOL_KIND
        or document.get("evidence_scope") != EVIDENCE_SCOPE
        or document.get("source_fingerprint") != SOURCE_FINGERPRINT
        or document.get("base_fold") != BASE_FOLD
        or document.get("base_part") != BASE_PART
        or document.get("track_selection_seed") != TRACK_SELECTION_SEED
        or document.get("artist_split_seed") != ARTIST_SPLIT_SEED
        or document.get("track_limit") != TRACK_LIMIT
        or document.get("selection_sha256") != EXPECTED_SELECTION_SHA256
        or document.get("split_summary") != EXPECTED_SPLITS
        or document.get("shadow_labels_accessed") is not False
        or document.get("promotion_allowed") is not False
    ):
        raise V3ProtocolError("protocol envelope drift")
    entries = document.get("tracks")
    if not isinstance(entries, list) or len(entries) != TRACK_LIMIT:
        raise V3ProtocolError("protocol track plan drift")
    required = {"track_id", "artist_id", "source_sha256", "split"}
    track_ids = []
    split_artists = {name: set() for name in EXPECTED_SPLITS}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != required:
            raise V3ProtocolError("protocol track entry schema drift")
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
            or any(ch not in "0123456789abcdef" for ch in source_sha256)
            or split not in EXPECTED_SPLITS
            or split != artist_split(artist_id)
        ):
            raise V3ProtocolError("invalid protocol track entry")
        track_ids.append(track_id)
        split_artists[split].add(artist_id)
    if len(set(track_ids)) != TRACK_LIMIT:
        raise V3ProtocolError("protocol track IDs are not unique")
    if stable_json_sha256(tuple(track_ids)) != EXPECTED_SELECTION_SHA256:
        raise V3ProtocolError("protocol track order drift")
    if any(
        split_artists[left].intersection(split_artists[right])
        for left, right in (
            ("train", "development"),
            ("train", "shadow"),
            ("development", "shadow"),
        )
    ):
        raise V3ProtocolError("protocol contains cross-split artist leakage")
    if context is not None:
        if context.source_fingerprint != SOURCE_FINGERPRINT:
            raise V3ProtocolError("context source fingerprint drift")
        tracks_by_id = context.by_track_id
        for entry in entries:
            track = tracks_by_id.get(int(entry["track_id"]))
            if (
                track is None
                or track.artist_id != entry["artist_id"]
                or track.expected_audio_sha256 != entry["source_sha256"]
            ):
                raise V3ProtocolError(
                    f"protocol/source mismatch for track {entry['track_id']}"
                )
    return document


def load_protocol(
    path: Path,
    *,
    context: Optional[JamendoContext] = None,
) -> Mapping[str, object]:
    candidate = Path(path).absolute()
    if candidate.is_symlink():
        raise V3ProtocolError("protocol path may not be a symlink")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_file() or resolved.stat().st_size > 4 * 1024 * 1024:
        raise V3ProtocolError("protocol file is missing or too large")
    try:
        document = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V3ProtocolError(f"invalid protocol JSON: {exc}") from exc
    return validate_protocol(document, context=context)


def _write_protocol(path: Path, protocol: Mapping[str, object]) -> None:
    path = Path(path).absolute()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(protocol, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-root", required=True)
    parser.add_argument("--audio-root", required=True)
    parser.add_argument("--state-root", required=True)
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
        protocol = build_protocol(context)
        _write_protocol(Path(args.output), protocol)
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
    except (JamendoValidationError, OSError, V3ProtocolError) as exc:
        raise SystemExit(f"V3 protocol blocked: {exc}") from exc


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
