"""Freeze the final untouched artist reserve for V3 development and shadow."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Set, Tuple

from .fulltrack_store import sha256_path, stable_json_sha256
from .fulltrack_v3 import SOURCE_FINGERPRINT, selected_test_tracks
from .fulltrack_v3_ranker import selected_validation_tracks
from .jamendo_fulltrack import (
    EVIDENCE_SCOPE,
    JamendoContext,
    JamendoTrack,
    JamendoValidationError,
    load_jamendo_context,
)


PROTOCOL_SCHEMA_VERSION = 1
PROTOCOL_KIND = "v3_final_untouched_artist_reserve_protocol"
BASE_FOLD = 0
TRACK_ORDER_SEED = 20260813
TRACK_COUNT = 38_956
EXPECTED_SELECTION_SHA256 = (
    "aed5188cd0fc3839181f10385318ad9964164dd54bb4629ad3bcd771ecf9ef5a"
)
EXPECTED_SPLITS = {
    "train": {
        "tracks": 32_859,
        "artists": 2_139,
        "track_ids_sha256": (
            "8da179c6a59b26a7b430ac9717b86d491bf2490a967573202b40fd7eff6b8658"
        ),
        "artist_ids_sha256": (
            "dbb9dc0853a29fe1d34636999907f164b455541aae77f8173ca700ec6378ae02"
        ),
    },
    "development": {
        "tracks": 3_074,
        "artists": 471,
        "track_ids_sha256": (
            "aed55cbb65cd461281f61c22abaf7bb6d9c00348157f7d2403992e2c572ed16d"
        ),
        "artist_ids_sha256": (
            "65a5dd45c658ba7a930bd38c71c4c6cbbe093ef934dbf4ff158fa2a39d9e6453"
        ),
    },
    "shadow": {
        "tracks": 3_023,
        "artists": 466,
        "track_ids_sha256": (
            "39986abcea529ce6b606b9151b2b7406ca8ebaf8e250b86e5a33b13bf539d698"
        ),
        "artist_ids_sha256": (
            "543b9542a32ec9bdfa8f0a44cd35da06a69e41347aeb3eb6d663b7c628d6d96c"
        ),
    },
}
EXPECTED_HISTORY = {
    "validation_tracks": 1_710,
    "validation_track_ids_sha256": (
        "31b4320186b55f5300260c77f62281323ddf737d6cc01b03c24489d8164123f6"
    ),
    "test_tracks": 1_702,
    "test_track_ids_sha256": (
        "c43dea0e8032c2e900786d3b7c5898a74317b8c4b12867889196402440718bc5"
    ),
    "artists": 1_098,
    "artist_ids_sha256": (
        "a4171afc82fd3edeaa128bbdaeea5e45650badc11b6d19d7b60071f9869aaaf6"
    ),
}
EXCLUDED_FOLD0_NONTRAIN_TRACKS = 16_569
UNASSIGNED_TRACKS = 176
FRESH_RESULT_PAYLOAD_SHA256 = (
    "97ea4f673f0d2edba417e83ece1187b7c2d34d3beda11040e6b040988bce3eaa"
)
OFFICIAL_AUDIT_FILE_SHA256 = (
    "aac1e050007393da897a11d077012f2621d6dcad17079a836d0e5aec57bd63b7"
)
OFFICIAL_AUDIT_PAYLOAD_SHA256 = (
    "e201582e1425b61a38e9b7af4ba111f65cd1f3e8778c2470616466416aff8868"
)


class V3ReserveProtocolError(RuntimeError):
    """Invalid, overlapping, changed, or prematurely opened reserve protocol."""


def _payload_sha256(document: Mapping[str, object]) -> str:
    payload = dict(document)
    payload.pop("payload_sha256", None)
    return stable_json_sha256(payload)


def reserve_split(
    base_part: Optional[str],
    artist_id: int,
    train_artists: Set[int],
    historical_artists: Set[int],
) -> Optional[str]:
    if (
        isinstance(artist_id, bool)
        or not isinstance(artist_id, int)
        or artist_id <= 0
    ):
        raise V3ReserveProtocolError("artist IDs must be positive integers")
    if base_part == "train":
        return "train"
    if artist_id in train_artists or artist_id in historical_artists:
        return None
    if base_part == "validation":
        return "development"
    if base_part == "test":
        return "shadow"
    return None


def _read_json(path: Path, label: str) -> Mapping[str, object]:
    candidate = Path(path).absolute()
    if candidate.is_symlink():
        raise V3ReserveProtocolError(f"{label} may not be a symlink")
    try:
        resolved = candidate.resolve(strict=True)
        if not resolved.is_file() or resolved.stat().st_size > 16 * 1024 * 1024:
            raise V3ReserveProtocolError(f"{label} is missing or too large")
        document = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V3ReserveProtocolError(f"{label} is invalid: {exc}") from exc
    if not isinstance(document, dict):
        raise V3ReserveProtocolError(f"{label} must be a JSON object")
    return document


def _verify_fresh_result(path: Path) -> Mapping[str, object]:
    document = _read_json(path, "fresh shadow result")
    if (
        document.get("artifact_kind") != "v3_fresh_scaled_clap_shadow_audit"
        or document.get("payload_sha256") != _payload_sha256(document)
        or document.get("payload_sha256") != FRESH_RESULT_PAYLOAD_SHA256
        or document.get("shadow_labels_accessed") is not True
        or document.get("shadow_gate", {}).get("automated_passed") is not False
        or document.get("promotion_allowed") is not False
    ):
        raise V3ReserveProtocolError("fresh shadow result envelope drift")
    return document


def _verify_official_audit(path: Path) -> Mapping[str, object]:
    if sha256_path(Path(path)) != OFFICIAL_AUDIT_FILE_SHA256:
        raise V3ReserveProtocolError("official audit file hash drift")
    document = _read_json(path, "official test audit")
    payload = dict(document)
    payload.pop("artifact_payload_sha256", None)
    if (
        document.get("artifact_kind")
        != "musicfm_selective_reranker_frozen_test_audit"
        or document.get("artifact_payload_sha256") != stable_json_sha256(payload)
        or document.get("artifact_payload_sha256")
        != OFFICIAL_AUDIT_PAYLOAD_SHA256
        or document.get("held_out_test_accessed") is not True
        or document.get("promotion_allowed") is not False
    ):
        raise V3ReserveProtocolError("official test audit envelope drift")
    return document


def _historical_selection(
    context: JamendoContext,
) -> Tuple[Set[int], Dict[str, object]]:
    validation = {
        track.track_id
        for fold_index in range(5)
        for track in selected_validation_tracks(context, fold_index)
    }
    test = {
        track.track_id
        for fold_index in range(5)
        for track in selected_test_tracks(context, fold_index)
    }
    selected_ids = validation | test
    artists = {
        context.by_track_id[track_id].artist_id for track_id in selected_ids
    }
    summary = {
        "validation_tracks": len(validation),
        "validation_track_ids_sha256": stable_json_sha256(
            tuple(sorted(validation))
        ),
        "test_tracks": len(test),
        "test_track_ids_sha256": stable_json_sha256(tuple(sorted(test))),
        "artists": len(artists),
        "artist_ids_sha256": stable_json_sha256(tuple(sorted(artists))),
    }
    if summary != EXPECTED_HISTORY:
        raise V3ReserveProtocolError(f"historical selection drift: {summary}")
    return artists, summary


def _summary(entries: Sequence[Mapping[str, object]]) -> Dict[str, object]:
    track_ids = tuple(sorted(int(entry["track_id"]) for entry in entries))
    artist_ids = tuple(sorted({int(entry["artist_id"]) for entry in entries}))
    return {
        "tracks": len(track_ids),
        "artists": len(artist_ids),
        "track_ids_sha256": stable_json_sha256(track_ids),
        "artist_ids_sha256": stable_json_sha256(artist_ids),
    }


def build_reserve_protocol(
    context: JamendoContext,
    fresh_result_path: Path,
    official_audit_path: Path,
) -> Dict[str, object]:
    if context.source_fingerprint != SOURCE_FINGERPRINT:
        raise V3ReserveProtocolError("Jamendo source fingerprint drift")
    fresh_result = _verify_fresh_result(fresh_result_path)
    official_audit = _verify_official_audit(official_audit_path)
    fold = next((item for item in context.folds if item.index == BASE_FOLD), None)
    if fold is None:
        raise V3ReserveProtocolError("base fold is missing")
    train_artists = {
        track.artist_id
        for track in context.tracks
        if fold.track_parts.get(track.track_id) == "train"
    }
    historical_artists, history = _historical_selection(context)
    classified = [
        (
            track,
            reserve_split(
                fold.track_parts.get(track.track_id),
                track.artist_id,
                train_artists,
                historical_artists,
            ),
        )
        for track in context.tracks
    ]
    ordered = sorted(
        ((track, split) for track, split in classified if split is not None),
        key=lambda item: stable_json_sha256(
            {"seed": TRACK_ORDER_SEED, "track_id": item[0].track_id}
        ),
    )
    if len(ordered) != TRACK_COUNT:
        raise V3ReserveProtocolError("reserve protocol track count drift")
    entries = [
        {
            "track_id": track.track_id,
            "artist_id": track.artist_id,
            "source_sha256": track.expected_audio_sha256,
            "split": split,
        }
        for track, split in ordered
    ]
    selection_hash = stable_json_sha256(
        tuple(int(entry["track_id"]) for entry in entries)
    )
    split_summary = {
        split: _summary(
            [entry for entry in entries if entry["split"] == split]
        )
        for split in EXPECTED_SPLITS
    }
    if (
        selection_hash != EXPECTED_SELECTION_SHA256
        or split_summary != EXPECTED_SPLITS
    ):
        raise V3ReserveProtocolError("reserve selection drift")
    split_artists = {
        split: {
            int(entry["artist_id"])
            for entry in entries
            if entry["split"] == split
        }
        for split in EXPECTED_SPLITS
    }
    if any(
        split_artists[left].intersection(split_artists[right])
        for left, right in (
            ("train", "development"),
            ("train", "shadow"),
            ("development", "shadow"),
        )
    ):
        raise V3ReserveProtocolError("reserve protocol contains artist leakage")
    excluded_nontrain = sum(
        fold.track_parts.get(track.track_id) in {"validation", "test"}
        and split is None
        for track, split in classified
    )
    unassigned = sum(
        fold.track_parts.get(track.track_id) is None
        for track in context.tracks
    )
    if (
        excluded_nontrain != EXCLUDED_FOLD0_NONTRAIN_TRACKS
        or unassigned != UNASSIGNED_TRACKS
    ):
        raise V3ReserveProtocolError("reserve exclusion count drift")
    protocol: Dict[str, object] = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "artifact_kind": PROTOCOL_KIND,
        "evidence_scope": EVIDENCE_SCOPE,
        "source_fingerprint": SOURCE_FINGERPRINT,
        "base_fold": BASE_FOLD,
        "track_order_seed": TRACK_ORDER_SEED,
        "selection_sha256": selection_hash,
        "split_summary": split_summary,
        "historical_selection": history,
        "excluded_fold0_nontrain_tracks": excluded_nontrain,
        "unassigned_tracks": unassigned,
        "tracks": entries,
        "fresh_shadow_result_payload_sha256": fresh_result["payload_sha256"],
        "official_test_audit_file_sha256": sha256_path(
            Path(official_audit_path)
        ),
        "official_test_audit_payload_sha256": official_audit[
            "artifact_payload_sha256"
        ],
        "development_labels_accessed": False,
        "shadow_labels_accessed": False,
        "promotion_allowed": False,
    }
    protocol["payload_sha256"] = stable_json_sha256(protocol)
    return protocol


def validate_reserve_protocol(
    document: object,
    *,
    context: Optional[JamendoContext] = None,
) -> Mapping[str, object]:
    if not isinstance(document, dict):
        raise V3ReserveProtocolError("reserve protocol must be a JSON object")
    if document.get("payload_sha256") != _payload_sha256(document):
        raise V3ReserveProtocolError("reserve protocol payload checksum mismatch")
    expected_envelope = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "artifact_kind": PROTOCOL_KIND,
        "evidence_scope": EVIDENCE_SCOPE,
        "source_fingerprint": SOURCE_FINGERPRINT,
        "base_fold": BASE_FOLD,
        "track_order_seed": TRACK_ORDER_SEED,
        "selection_sha256": EXPECTED_SELECTION_SHA256,
        "split_summary": EXPECTED_SPLITS,
        "historical_selection": EXPECTED_HISTORY,
        "excluded_fold0_nontrain_tracks": EXCLUDED_FOLD0_NONTRAIN_TRACKS,
        "unassigned_tracks": UNASSIGNED_TRACKS,
        "fresh_shadow_result_payload_sha256": FRESH_RESULT_PAYLOAD_SHA256,
        "official_test_audit_file_sha256": OFFICIAL_AUDIT_FILE_SHA256,
        "official_test_audit_payload_sha256": OFFICIAL_AUDIT_PAYLOAD_SHA256,
        "development_labels_accessed": False,
        "shadow_labels_accessed": False,
        "promotion_allowed": False,
    }
    if any(document.get(key) != value for key, value in expected_envelope.items()):
        raise V3ReserveProtocolError("reserve protocol envelope drift")
    entries = document.get("tracks")
    if not isinstance(entries, list) or len(entries) != TRACK_COUNT:
        raise V3ReserveProtocolError("reserve protocol track plan drift")
    required = {"track_id", "artist_id", "source_sha256", "split"}
    track_ids = []
    split_entries = {split: [] for split in EXPECTED_SPLITS}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != required:
            raise V3ReserveProtocolError("reserve protocol track schema drift")
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
            raise V3ReserveProtocolError("invalid reserve protocol track entry")
        track_ids.append(track_id)
        split_entries[split].append(entry)
    split_summary = {
        split: _summary(split_entries[split]) for split in EXPECTED_SPLITS
    }
    split_artists = {
        split: {int(entry["artist_id"]) for entry in split_entries[split]}
        for split in EXPECTED_SPLITS
    }
    if (
        len(set(track_ids)) != TRACK_COUNT
        or stable_json_sha256(tuple(track_ids)) != EXPECTED_SELECTION_SHA256
        or split_summary != EXPECTED_SPLITS
        or any(
            split_artists[left].intersection(split_artists[right])
            for left, right in (
                ("train", "development"),
                ("train", "shadow"),
                ("development", "shadow"),
            )
        )
    ):
        raise V3ReserveProtocolError("reserve protocol identity or leakage drift")
    if context is not None:
        if context.source_fingerprint != SOURCE_FINGERPRINT:
            raise V3ReserveProtocolError("context source fingerprint drift")
        for entry in entries:
            track = context.by_track_id.get(int(entry["track_id"]))
            if (
                track is None
                or track.artist_id != entry["artist_id"]
                or track.expected_audio_sha256 != entry["source_sha256"]
            ):
                raise V3ReserveProtocolError(
                    f"reserve/source mismatch for track {entry['track_id']}"
                )
    return document


def load_reserve_protocol(
    path: Path,
    *,
    context: Optional[JamendoContext] = None,
) -> Mapping[str, object]:
    document = _read_json(path, "reserve protocol")
    return validate_reserve_protocol(document, context=context)


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
    parser.add_argument("--fresh-result", required=True)
    parser.add_argument("--official-audit", required=True)
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
        protocol = build_reserve_protocol(
            context,
            Path(args.fresh_result),
            Path(args.official_audit),
        )
        _write_protocol(Path(args.output), protocol)
    except (
        JamendoValidationError,
        OSError,
        ValueError,
        V3ReserveProtocolError,
    ) as exc:
        raise SystemExit(f"reserve V3 protocol blocked: {exc}") from exc
    print(
        json.dumps(
            {
                "output": str(Path(args.output).absolute()),
                "payload_sha256": protocol["payload_sha256"],
                "selection_sha256": protocol["selection_sha256"],
                "split_summary": protocol["split_summary"],
                "development_labels_accessed": False,
                "shadow_labels_accessed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
