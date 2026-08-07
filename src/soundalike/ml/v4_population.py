"""Freeze the artist-disjoint populations used by the V4 preference study."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .jamendo_fulltrack import JamendoContext, load_jamendo_context


SCHEMA_VERSION = 1
MANIFEST_KIND = "soundalike_v4_population"
FOLD_INDEX = 0
DEVELOPMENT_PARTS = frozenset({"train", "validation"})
RESERVE_PART = "test"
TRACK_ID_KEYS = frozenset(
    {
        "candidate_track_id",
        "result_track_id",
        "seed_track_id",
        "track_id",
    }
)
TRACK_IDS_KEYS = frozenset(
    {
        "candidate_track_ids",
        "result_track_ids",
        "seed_track_ids",
        "track_ids",
    }
)


class V4PopulationError(RuntimeError):
    """The V4 population is overlapping, mutable, or inconsistent."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _content_sha256(document: Mapping[str, object]) -> str:
    payload = dict(document)
    payload.pop("content_sha256", None)
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _collect_track_ids(value: object, output: set[int]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if key in TRACK_ID_KEYS:
                track_id = _positive_int(child)
                if track_id is not None:
                    output.add(track_id)
            elif key in TRACK_IDS_KEYS and isinstance(child, Sequence):
                for item in child:
                    track_id = _positive_int(item)
                    if track_id is not None:
                        output.add(track_id)
            _collect_track_ids(child, output)
    elif isinstance(value, list):
        for child in value:
            _collect_track_ids(child, output)


def _json_sources(paths: Iterable[Path]) -> tuple[Path, ...]:
    sources: set[Path] = set()
    for raw_path in paths:
        path = Path(raw_path).resolve()
        if path.is_dir():
            sources.update(candidate.resolve() for candidate in path.rglob("*.json"))
        elif path.is_file() and path.suffix.lower() == ".json":
            sources.add(path)
        else:
            raise V4PopulationError(f"exclusion source is not JSON: {path}")
    return tuple(sorted(sources, key=lambda value: str(value).casefold()))


def collect_exposed_tracks(
    paths: Iterable[Path],
    known_track_ids: set[int],
) -> tuple[tuple[int, ...], tuple[Mapping[str, object], ...]]:
    """Collect known Jamendo track IDs from immutable prior evidence."""
    found: set[int] = set()
    sources = []
    for path in _json_sources(paths):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise V4PopulationError(f"cannot parse exclusion source: {path}") from exc
        current: set[int] = set()
        _collect_track_ids(document, current)
        matched = current & known_track_ids
        found.update(matched)
        sources.append(
            {
                "name": path.name,
                "file_sha256": _file_sha256(path),
                "matched_track_count": len(matched),
            }
        )
    return tuple(sorted(found)), tuple(sources)


def build_population_manifest(
    context: JamendoContext,
    exclusion_paths: Iterable[Path],
) -> Mapping[str, object]:
    """Build a deterministic train/validation development and test reserve split."""
    by_track = context.by_track_id
    exposed_track_ids, sources = collect_exposed_tracks(
        exclusion_paths, set(by_track)
    )
    exposed_artists = {
        int(by_track[track_id].artist_id) for track_id in exposed_track_ids
    }
    eligible = [
        track
        for track in context.tracks
        if int(track.artist_id) not in exposed_artists
    ]
    development = sorted(
        int(track.track_id)
        for track in eligible
        if track.fold_parts[FOLD_INDEX] in DEVELOPMENT_PARTS
    )
    reserve = sorted(
        int(track.track_id)
        for track in eligible
        if track.fold_parts[FOLD_INDEX] == RESERVE_PART
    )
    unassigned = sorted(
        int(track.track_id)
        for track in eligible
        if track.fold_parts[FOLD_INDEX] is None
    )
    development_artists = {
        int(by_track[track_id].artist_id) for track_id in development
    }
    reserve_artists = {int(by_track[track_id].artist_id) for track_id in reserve}
    if not development or not reserve:
        raise V4PopulationError("V4 development or reserve population is empty")
    if development_artists & reserve_artists:
        raise V4PopulationError("official fold is not artist-disjoint")

    document: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "manifest_kind": MANIFEST_KIND,
        "source_fingerprint": context.source_fingerprint,
        "policy": {
            "fold_index": FOLD_INDEX,
            "development_parts": sorted(DEVELOPMENT_PARTS),
            "human_reserve_part": RESERVE_PART,
            "exclude_every_artist_with_prior_public_or_human_evidence": True,
            "reserve_labels_opened": False,
            "promotion_allowed": False,
        },
        "exclusion_sources": list(sources),
        "excluded": {
            "track_ids": list(exposed_track_ids),
            "artist_ids": sorted(exposed_artists),
        },
        "development": {
            "track_ids": development,
            "artist_ids": sorted(development_artists),
        },
        "human_reserve": {
            "track_ids": reserve,
            "artist_ids": sorted(reserve_artists),
        },
        "unassigned_track_ids": unassigned,
        "counts": {
            "source_tracks": len(context.tracks),
            "source_artists": len({track.artist_id for track in context.tracks}),
            "excluded_tracks": len(exposed_track_ids),
            "excluded_artists": len(exposed_artists),
            "development_tracks": len(development),
            "development_artists": len(development_artists),
            "human_reserve_tracks": len(reserve),
            "human_reserve_artists": len(reserve_artists),
            "unassigned_tracks": len(unassigned),
        },
    }
    document["content_sha256"] = _content_sha256(document)
    validate_population_manifest(document, context)
    return document


def _strict_ids(value: object, label: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise V4PopulationError(f"{label} is not a list")
    result = tuple(value)
    if (
        any(_positive_int(item) is None for item in result)
        or len(set(result)) != len(result)
        or tuple(sorted(result)) != result
    ):
        raise V4PopulationError(f"{label} is not sorted unique positive integers")
    return result


def validate_population_manifest(
    document: Mapping[str, object],
    context: JamendoContext,
) -> None:
    """Fail closed if a population manifest drifts or loses disjointness."""
    if (
        document.get("schema_version") != SCHEMA_VERSION
        or document.get("manifest_kind") != MANIFEST_KIND
        or document.get("source_fingerprint") != context.source_fingerprint
        or document.get("content_sha256") != _content_sha256(document)
    ):
        raise V4PopulationError("V4 population identity or hash drift")
    policy = document.get("policy")
    expected_policy = {
        "fold_index": FOLD_INDEX,
        "development_parts": sorted(DEVELOPMENT_PARTS),
        "human_reserve_part": RESERVE_PART,
        "exclude_every_artist_with_prior_public_or_human_evidence": True,
        "reserve_labels_opened": False,
        "promotion_allowed": False,
    }
    if policy != expected_policy:
        raise V4PopulationError("V4 population policy drift")
    excluded = document.get("excluded")
    development = document.get("development")
    reserve = document.get("human_reserve")
    if not all(isinstance(value, Mapping) for value in (excluded, development, reserve)):
        raise V4PopulationError("V4 population sections are invalid")
    excluded_tracks = _strict_ids(excluded.get("track_ids"), "excluded track IDs")
    excluded_artists = _strict_ids(excluded.get("artist_ids"), "excluded artist IDs")
    development_tracks = _strict_ids(
        development.get("track_ids"), "development track IDs"
    )
    development_artists = _strict_ids(
        development.get("artist_ids"), "development artist IDs"
    )
    reserve_tracks = _strict_ids(reserve.get("track_ids"), "reserve track IDs")
    reserve_artists = _strict_ids(reserve.get("artist_ids"), "reserve artist IDs")
    unassigned = _strict_ids(
        document.get("unassigned_track_ids"), "unassigned track IDs"
    )
    by_track = context.by_track_id
    if not set((*excluded_tracks, *development_tracks, *reserve_tracks, *unassigned)) <= set(
        by_track
    ):
        raise V4PopulationError("V4 population references an unknown track")
    if set(development_tracks) & set(reserve_tracks):
        raise V4PopulationError("V4 development and reserve tracks overlap")
    if set(development_artists) & set(reserve_artists):
        raise V4PopulationError("V4 development and reserve artists overlap")
    if set(excluded_artists) & (set(development_artists) | set(reserve_artists)):
        raise V4PopulationError("excluded artist appears in a V4 population")
    actual_development_artists = {
        int(by_track[track_id].artist_id) for track_id in development_tracks
    }
    actual_reserve_artists = {
        int(by_track[track_id].artist_id) for track_id in reserve_tracks
    }
    if actual_development_artists != set(development_artists):
        raise V4PopulationError("development artist identity drift")
    if actual_reserve_artists != set(reserve_artists):
        raise V4PopulationError("reserve artist identity drift")
    if any(
        by_track[track_id].fold_parts[FOLD_INDEX] not in DEVELOPMENT_PARTS
        for track_id in development_tracks
    ):
        raise V4PopulationError("development contains a non-development fold part")
    if any(
        by_track[track_id].fold_parts[FOLD_INDEX] != RESERVE_PART
        for track_id in reserve_tracks
    ):
        raise V4PopulationError("reserve contains a non-test fold part")
    expected_counts = {
        "source_tracks": len(context.tracks),
        "source_artists": len({track.artist_id for track in context.tracks}),
        "excluded_tracks": len(excluded_tracks),
        "excluded_artists": len(excluded_artists),
        "development_tracks": len(development_tracks),
        "development_artists": len(development_artists),
        "human_reserve_tracks": len(reserve_tracks),
        "human_reserve_artists": len(reserve_artists),
        "unassigned_tracks": len(unassigned),
    }
    if document.get("counts") != expected_counts:
        raise V4PopulationError("V4 population count drift")


def write_manifest(path: Path, document: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the V4 population manifest.")
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--exclude", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    context = load_jamendo_context(
        args.metadata_root,
        args.audio_root,
        args.state_root,
        production=True,
    )
    document = build_population_manifest(context, args.exclude)
    write_manifest(args.output, document)
    print(json.dumps({"output": str(args.output), **document["counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
