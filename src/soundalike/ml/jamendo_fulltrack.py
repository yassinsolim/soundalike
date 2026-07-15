"""Strict MTG-Jamendo full-track metadata and collection provenance.

This module deliberately does not download audio.  Production callers must
present the completion marker emitted by the repository's verified downloader.
The marker is bound to the two official checksum manifests, and every archive
marker, metadata join, license record, split, and local audio path is audited
before extraction can start.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse


EXPECTED_TRACKS = 55_701
EXPECTED_ARCHIVES = 100
EXPECTED_SPLIT_TRACKS = 55_525
EXPECTED_METADATA_COMMIT = "cafd8e20c265ed84f1e61f1c875327971f43a62f"
EXPECTED_METADATA_TREE = "e7ae994f981a54a3a64d6af7e2762844356cc8a7"
EXPECTED_ARCHIVE_MANIFEST_SHA256 = (
    "de704285860adae62024aef684c9d1c9f62e6a1711b5772db5cc7a0fdb1a556f"
)
EXPECTED_TRACK_MANIFEST_SHA256 = (
    "570404f1c8ff3571069439fae88ee9c854cdd6f5286fded085d23c19b7d53c38"
)
EVIDENCE_SCOPE = "full_track_jamendo_research"

_ARCHIVE_MANIFEST = "raw_30s_audio_sha256_tars.txt"
_TRACK_MANIFEST = "raw_30s_audio_sha256_tracks.txt"
_ID_PATTERNS = {
    "track": re.compile(r"track_(\d+)\Z"),
    "artist": re.compile(r"artist_(\d+)\Z"),
    "album": re.compile(r"album_(\d+)\Z"),
}
_CHECKSUM_LINE = re.compile(r"([0-9a-fA-F]{64})[ \t]+([^ \t\r\n]+)\Z")
_ARCHIVE_NAME = re.compile(r"raw_30s_audio-(\d{2})\.tar\Z")
_TAG = re.compile(r"(genre|instrument|mood/theme)---([^\t\r\n]+)\Z")
_LICENSE_LINE = re.compile(r"Available under (.+): (https?://\S+)\Z")
_ATTRIBUTION_LINE = re.compile(
    r".+ by .+ from Jamendo: https?://(?:www\.)?jamendo\.com/track/(\d+)\Z"
)
_PARTS = ("train", "validation", "test")


class JamendoValidationError(RuntimeError):
    """The local dataset is incomplete, unsafe, corrupt, or provenance-drifted."""


@dataclass(frozen=True)
class ChecksumEntry:
    path: str
    sha256: str


@dataclass(frozen=True)
class TrackLicense:
    path: str
    attribution: str
    name: str
    url: str
    permits_commercial_use: bool
    permits_derivatives: bool


@dataclass(frozen=True)
class JamendoTrack:
    row_index: int
    track_id: int
    artist_id: int
    album_id: int
    relative_path: str
    audio_path: Path
    duration_seconds: float
    tags: Tuple[str, ...]
    title: str
    artist_name: str
    album_name: str
    release_date: str
    jamendo_url: str
    license: TrackLicense
    expected_audio_sha256: str
    expected_audio_bytes: int
    fold_parts: Tuple[Optional[str], ...] = ()


@dataclass(frozen=True)
class ArtistFold:
    index: int
    track_parts: Mapping[int, str]
    artist_parts: Mapping[int, str]
    track_tags: Mapping[int, Tuple[str, ...]]
    tags: Tuple[str, ...]


@dataclass(frozen=True)
class JamendoContext:
    tracks: Tuple[JamendoTrack, ...]
    folds: Tuple[ArtistFold, ...]
    metadata_root: Path
    audio_root: Path
    state_root: Path
    metadata_commit: str
    archive_manifest_sha256: str
    track_manifest_sha256: str
    metadata_hashes: Mapping[str, str]
    source_fingerprint: str
    evidence_scope: str = EVIDENCE_SCOPE

    @property
    def by_track_id(self) -> Mapping[int, JamendoTrack]:
        return MappingProxyType({track.track_id: track for track in self.tracks})


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Hash a file using bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json_object(path: Path) -> Dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise JamendoValidationError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise JamendoValidationError(f"expected a JSON object in {path}")
    return value


def _is_link(path: Path) -> bool:
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    if is_junction is not None and is_junction():
        return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if not reparse_flag:
        return False
    try:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
    except OSError:
        return False
    return bool(attributes & reparse_flag)


def _require_local_root(path: Path, label: str) -> Path:
    raw = str(path)
    parsed = urlparse(raw)
    if parsed.scheme and len(parsed.scheme) > 1:
        raise JamendoValidationError(f"{label} must be a local path, not a URL: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise JamendoValidationError(f"{label} does not exist: {path}") from exc
    if not resolved.is_dir():
        raise JamendoValidationError(f"{label} is not a directory: {resolved}")
    if _is_link(path):
        raise JamendoValidationError(f"{label} may not be a symlink or junction: {path}")
    return resolved


def _require_concrete_file(root: Path, relative: str, label: str) -> Path:
    safe = safe_relative_path(relative)
    candidate = root.joinpath(*PurePosixPath(safe).parts)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise JamendoValidationError(
            f"{label} is missing or escapes its root: {candidate}"
        ) from exc
    current = root
    for part in PurePosixPath(safe).parts:
        current = current / part
        if _is_link(current):
            raise JamendoValidationError(
                f"{label} contains a symlink or junction: {current}"
            )
    if not resolved.is_file():
        raise JamendoValidationError(f"{label} is not a regular file: {resolved}")
    return resolved


def safe_relative_path(value: str) -> str:
    """Return a canonical local POSIX relative path or fail closed."""
    if not isinstance(value, str) or not value:
        raise JamendoValidationError("empty dataset path")
    if "\x00" in value or "\\" in value:
        raise JamendoValidationError(f"unsafe dataset path: {value!r}")
    parsed = urlparse(value)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        raise JamendoValidationError(f"URLs are not dataset paths: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts:
        raise JamendoValidationError(f"dataset path is not safely relative: {value!r}")
    if any(part in ("", ".", "..") for part in path.parts):
        raise JamendoValidationError(f"dataset path contains traversal: {value!r}")
    if ":" in path.parts[0]:
        raise JamendoValidationError(f"dataset path contains a drive: {value!r}")
    canonical = path.as_posix()
    if canonical != value:
        raise JamendoValidationError(f"dataset path is not canonical: {value!r}")
    return canonical


def _parse_id(value: str, kind: str, *, where: str) -> int:
    match = _ID_PATTERNS[kind].fullmatch(value)
    if match is None:
        raise JamendoValidationError(f"{where}: malformed {kind} id {value!r}")
    return int(match.group(1))


def _read_checksum_manifest(
    path: Path, *, archives: bool
) -> Tuple[Tuple[ChecksumEntry, ...], str]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise JamendoValidationError(f"manifest is not UTF-8: {path}") from exc
    if not text or not text.endswith("\n"):
        raise JamendoValidationError(f"manifest must be non-empty and newline-ended: {path}")
    entries = []
    seen: Dict[str, int] = {}
    for line_number, line in enumerate(text.splitlines(), 1):
        match = _CHECKSUM_LINE.fullmatch(line)
        if match is None:
            raise JamendoValidationError(f"{path}:{line_number}: malformed checksum row")
        checksum, name = match.groups()
        if archives:
            if _ARCHIVE_NAME.fullmatch(name) is None:
                raise JamendoValidationError(
                    f"{path}:{line_number}: unsafe archive name {name!r}"
                )
        else:
            name = safe_relative_path(name)
            if len(PurePosixPath(name).parts) != 2 or not name.lower().endswith(".mp3"):
                raise JamendoValidationError(
                    f"{path}:{line_number}: expected an MP3 path"
                )
        collision = name.casefold()
        if collision in seen:
            raise JamendoValidationError(
                f"{path}:{line_number}: duplicate/colliding path {name!r}"
            )
        seen[collision] = line_number
        entries.append(ChecksumEntry(name, checksum.lower()))
    return tuple(entries), digest


def _require_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise JamendoValidationError(f"{label} must be an integer")
    return value


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise JamendoValidationError(f"{label} must be a lowercase SHA-256")
    return value


def _marker_manifests(marker: Mapping[str, object], label: str) -> Tuple[str, str]:
    raw = marker.get("manifests")
    if not isinstance(raw, dict):
        raise JamendoValidationError(f"{label}.manifests must be an object")
    return (
        _require_sha256(raw.get("archives_sha256"), f"{label}.archives_sha256"),
        _require_sha256(raw.get("tracks_sha256"), f"{label}.tracks_sha256"),
    )


def validate_completion_marker(
    state_root: Path,
    *,
    archive_manifest_sha256: str,
    track_manifest_sha256: str,
    expected_archives: int,
    expected_tracks: int,
) -> Mapping[str, object]:
    """Validate the external downloader's collection-wide completion marker."""
    marker_path = state_root / "collection.complete.json"
    if not marker_path.is_file():
        raise JamendoValidationError(
            "production extraction is blocked: state/collection.complete.json "
            "does not exist"
        )
    if _is_link(marker_path):
        raise JamendoValidationError("completion marker may not be a symlink or junction")
    marker = _read_json_object(marker_path)
    if _require_int(marker.get("schema_version"), "completion schema_version") != 1:
        raise JamendoValidationError("unsupported completion marker schema")
    if marker.get("collection") != "raw_30s/audio":
        raise JamendoValidationError("completion marker names the wrong collection")
    if _require_int(marker.get("archive_count"), "completion archive_count") != expected_archives:
        raise JamendoValidationError("completion marker archive count mismatch")
    if _require_int(marker.get("track_count"), "completion track_count") != expected_tracks:
        raise JamendoValidationError("completion marker track count mismatch")
    if _require_int(marker.get("track_bytes"), "completion track_bytes") <= 0:
        raise JamendoValidationError("completion marker has invalid track bytes")
    archive_hash, track_hash = _marker_manifests(marker, "completion")
    if archive_hash != archive_manifest_sha256:
        raise JamendoValidationError("completion marker archive manifest hash mismatch")
    if track_hash != track_manifest_sha256:
        raise JamendoValidationError("completion marker track manifest hash mismatch")
    return MappingProxyType(marker)


def _audit_archive_markers(
    state_root: Path,
    archive_entries: Sequence[ChecksumEntry],
    track_entries: Mapping[str, str],
    *,
    archive_manifest_sha256: str,
    track_manifest_sha256: str,
) -> Mapping[str, int]:
    claimed: Dict[str, int] = {}
    total_bytes = 0
    for archive in archive_entries:
        path = _require_concrete_file(
            state_root, f"{archive.path}.verified.json", "archive marker"
        )
        marker = _read_json_object(path)
        if _require_int(marker.get("schema_version"), "archive marker schema_version") != 1:
            raise JamendoValidationError(f"unsupported archive marker schema: {path}")
        if marker.get("archive") != archive.path:
            raise JamendoValidationError(f"archive marker identity mismatch: {path}")
        if _require_sha256(marker.get("archive_sha256"), "archive SHA-256") != archive.sha256:
            raise JamendoValidationError(f"archive marker hash mismatch: {path}")
        if _require_int(marker.get("archive_bytes"), "archive bytes") <= 0:
            raise JamendoValidationError(f"archive marker has invalid byte count: {path}")
        marker_archive_hash, marker_track_hash = _marker_manifests(marker, str(path))
        if marker_archive_hash != archive_manifest_sha256:
            raise JamendoValidationError(f"archive manifest binding mismatch: {path}")
        if marker_track_hash != track_manifest_sha256:
            raise JamendoValidationError(f"track manifest binding mismatch: {path}")
        tracks = marker.get("tracks")
        if not isinstance(tracks, list):
            raise JamendoValidationError(f"archive marker tracks must be a list: {path}")
        if _require_int(marker.get("track_count"), "archive track_count") != len(tracks):
            raise JamendoValidationError(f"archive marker track count mismatch: {path}")
        marker_track_bytes = 0
        for item in tracks:
            if not isinstance(item, dict):
                raise JamendoValidationError(f"invalid track marker in {path}")
            relative = safe_relative_path(str(item.get("path", "")))
            checksum = _require_sha256(item.get("sha256"), "track SHA-256")
            size = _require_int(item.get("bytes"), "track bytes")
            if size <= 0:
                raise JamendoValidationError(f"non-positive track bytes in {path}")
            if relative in claimed:
                raise JamendoValidationError(f"track claimed by two archive markers: {relative}")
            if track_entries.get(relative) != checksum:
                raise JamendoValidationError(
                    f"archive marker track differs from official manifest: {relative}"
                )
            claimed[relative] = size
            marker_track_bytes += size
        if _require_int(marker.get("track_bytes"), "archive track_bytes") != marker_track_bytes:
            raise JamendoValidationError(f"archive marker byte total mismatch: {path}")
        total_bytes += marker_track_bytes
    if set(claimed) != set(track_entries):
        missing = sorted(set(track_entries) - set(claimed))[:3]
        extra = sorted(set(claimed) - set(track_entries))[:3]
        raise JamendoValidationError(
            f"archive marker/track manifest join mismatch; missing={missing}, extra={extra}"
        )
    return MappingProxyType(claimed)


def _read_tsv(path: Path, expected_header: Sequence[str]) -> Iterable[Tuple[int, list[str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, None)
        if header != list(expected_header):
            raise JamendoValidationError(
                f"{path}: unexpected header {header!r}; expected {list(expected_header)!r}"
            )
        for line_number, row in enumerate(reader, 2):
            if len(row) < len(expected_header):
                raise JamendoValidationError(f"{path}:{line_number}: short TSV row")
            yield line_number, row


def _parse_metadata_tracks(path: Path) -> Dict[int, Dict[str, object]]:
    header = ("TRACK_ID", "ARTIST_ID", "ALBUM_ID", "PATH", "DURATION", "TAGS")
    tracks: Dict[int, Dict[str, object]] = {}
    seen_paths: Dict[str, int] = {}
    for line_number, row in _read_tsv(path, header):
        where = f"{path}:{line_number}"
        track_id = _parse_id(row[0], "track", where=where)
        if track_id in tracks:
            raise JamendoValidationError(f"{where}: duplicate track id {track_id}")
        artist_id = _parse_id(row[1], "artist", where=where)
        album_id = _parse_id(row[2], "album", where=where)
        relative = safe_relative_path(row[3])
        if (
            len(PurePosixPath(relative).parts) != 2
            or not relative.casefold().endswith(".mp3")
        ):
            raise JamendoValidationError(f"{where}: audio path must be NN/id.mp3")
        collision = relative.casefold()
        if collision in seen_paths:
            raise JamendoValidationError(f"{where}: duplicate/colliding audio path")
        seen_paths[collision] = line_number
        try:
            duration = float(row[4])
        except ValueError as exc:
            raise JamendoValidationError(f"{where}: invalid duration") from exc
        # Membership in raw_30s.tsv is authoritative. Some official durations
        # round to exactly/slightly below 30, so reject only impossible values.
        if not math.isfinite(duration) or not 0.0 < duration <= 24 * 60 * 60:
            raise JamendoValidationError(f"{where}: duration is outside safe bounds")
        tags = tuple(row[5:])
        if not tags or len(tags) != len(set(tags)):
            raise JamendoValidationError(f"{where}: tags must be non-empty and unique")
        if any(_TAG.fullmatch(tag) is None for tag in tags):
            raise JamendoValidationError(f"{where}: malformed tag")
        tracks[track_id] = {
            "artist_id": artist_id,
            "album_id": album_id,
            "relative_path": relative,
            "duration": duration,
            "tags": tuple(sorted(tags)),
        }
    return tracks


def _parse_descriptive_metadata(path: Path) -> Dict[int, Dict[str, str]]:
    header = (
        "TRACK_ID",
        "ARTIST_ID",
        "ALBUM_ID",
        "TRACK_NAME",
        "ARTIST_NAME",
        "ALBUM_NAME",
        "RELEASEDATE",
        "URL",
    )
    result: Dict[int, Dict[str, str]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise JamendoValidationError(
            f"cannot read descriptive metadata {path}: {exc}"
        ) from exc
    if not lines or lines[0].split("\t") != list(header):
        raise JamendoValidationError(f"{path}: unexpected descriptive metadata header")
    for line_number, line in enumerate(lines[1:], 2):
        # raw.meta.tsv contains unmatched literal quote characters, so CSV quote
        # interpretation joins unrelated lines. Tabs are the authoritative
        # delimiters. Two official rows have one documented empty extra field
        # between title and artist; normalize only that exact shape.
        row = line.split("\t")
        if len(row) == 9 and row[4] == "":
            del row[4]
        if len(row) != len(header):
            raise JamendoValidationError(
                f"{path}:{line_number}: unexpected descriptive columns"
            )
        where = f"{path}:{line_number}"
        track_id = _parse_id(row[0], "track", where=where)
        if track_id in result:
            raise JamendoValidationError(f"{where}: duplicate descriptive track")
        artist_id = _parse_id(row[1], "artist", where=where)
        album_id = _parse_id(row[2], "album", where=where)
        if not row[3] or not row[4]:
            raise JamendoValidationError(f"{where}: title and artist are required")
        parsed = urlparse(row[7])
        if (
            parsed.scheme not in ("http", "https")
            or parsed.hostname not in ("jamendo.com", "www.jamendo.com")
            or parsed.path.rstrip("/") != f"/track/{track_id}"
            or parsed.query
            or parsed.fragment
        ):
            raise JamendoValidationError(f"{where}: invalid official Jamendo URL")
        result[track_id] = {
            "artist_id": str(artist_id),
            "album_id": str(album_id),
            "title": row[3],
            "artist_name": row[4],
            "album_name": row[5],
            "release_date": row[6],
            "url": row[7],
        }
    return result


def _parse_licenses(path: Path) -> Dict[str, TrackLicense]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise JamendoValidationError(f"cannot read license file {path}: {exc}") from exc
    normalized = text.replace("\r\n", "\n")
    if not normalized.endswith("\n"):
        raise JamendoValidationError("license file must be newline-terminated")
    blocks = normalized.strip("\n").split("\n\n")
    result: Dict[str, TrackLicense] = {}
    seen_paths = set()
    for number, block in enumerate(blocks, 1):
        lines = block.splitlines()
        if len(lines) != 3:
            raise JamendoValidationError(f"license block {number} must contain three lines")
        relative = safe_relative_path(lines[0])
        if (
            len(PurePosixPath(relative).parts) != 2
            or not relative.casefold().endswith(".mp3")
        ):
            raise JamendoValidationError(
                f"license block {number} has an invalid audio path"
            )
        track_name = PurePosixPath(relative).stem
        attribution_match = _ATTRIBUTION_LINE.fullmatch(lines[1])
        if attribution_match is None or int(attribution_match.group(1)) != int(track_name):
            raise JamendoValidationError(f"license attribution mismatch for {relative}")
        license_match = _LICENSE_LINE.fullmatch(lines[2])
        if license_match is None:
            raise JamendoValidationError(f"malformed license declaration for {relative}")
        name, url = license_match.groups()
        parsed = urlparse(url)
        host = (parsed.hostname or "").casefold()
        if parsed.scheme not in ("http", "https") or host not in (
            "creativecommons.org",
            "www.creativecommons.org",
            "artlibre.org",
            "www.artlibre.org",
        ):
            raise JamendoValidationError(f"unrecognized license authority for {relative}")
        lowered = url.casefold()
        license_slug = next(
            (
                part
                for part in parsed.path.casefold().split("/")
                if part == "by" or part.startswith("by-")
            ),
            "",
        )
        if host.endswith("creativecommons.org") and "/licenses/" not in parsed.path:
            raise JamendoValidationError(f"invalid Creative Commons URL for {relative}")
        if host.endswith("artlibre.org") and "licence/lal" not in parsed.path.casefold():
            raise JamendoValidationError(f"invalid Art Libre URL for {relative}")
        collision = relative.casefold()
        if collision in seen_paths:
            raise JamendoValidationError(f"duplicate/colliding license path {relative}")
        seen_paths.add(collision)
        result[relative] = TrackLicense(
            path=relative,
            attribution=lines[1],
            name=name,
            url=url,
            permits_commercial_use="/by-nc" not in lowered,
            permits_derivatives=not license_slug.endswith("-nd"),
        )
    return result


def _parse_fold(
    split_root: Path,
    fold_index: int,
    metadata: Mapping[int, Mapping[str, object]],
) -> ArtistFold:
    track_parts: Dict[int, str] = {}
    artist_parts: Dict[int, str] = {}
    track_tags: Dict[int, Tuple[str, ...]] = {}
    all_tags = set()
    for part in _PARTS:
        relative = f"split-{fold_index}/autotagging-{part}.tsv"
        path = _require_concrete_file(split_root, relative, f"fold {fold_index} {part}")
        parsed = _parse_metadata_tracks(path)
        for track_id, split_track in parsed.items():
            if track_id in track_parts:
                raise JamendoValidationError(
                    f"fold {fold_index}: track {track_id} occurs in multiple parts"
                )
            source = metadata.get(track_id)
            if source is None:
                raise JamendoValidationError(
                    f"fold {fold_index}: unknown track {track_id}"
                )
            for field in ("artist_id", "album_id", "relative_path"):
                if split_track[field] != source[field]:
                    raise JamendoValidationError(
                        f"fold {fold_index}: {field} drift for track {track_id}"
                    )
            if not math.isclose(
                float(split_track["duration"]),
                float(source["duration"]),
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise JamendoValidationError(
                    f"fold {fold_index}: duration drift for track {track_id}"
                )
            artist_id = int(split_track["artist_id"])
            previous = artist_parts.get(artist_id)
            if previous is not None and previous != part:
                raise JamendoValidationError(
                    f"fold {fold_index}: artist {artist_id} crosses {previous}/{part}"
                )
            artist_parts[artist_id] = part
            track_parts[track_id] = part
            track_tags[track_id] = tuple(sorted(split_track["tags"]))
            all_tags.update(track_tags[track_id])
    if not track_parts:
        raise JamendoValidationError(f"fold {fold_index} is empty")
    return ArtistFold(
        index=fold_index,
        track_parts=MappingProxyType(track_parts),
        artist_parts=MappingProxyType(artist_parts),
        track_tags=MappingProxyType(track_tags),
        tags=tuple(sorted(all_tags)),
    )


def _read_small_ascii(path: Path, label: str, *, max_bytes: int = 1024) -> str:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise JamendoValidationError(f"cannot read {label}: {path}") from exc
    if not raw or len(raw) > max_bytes or b"\x00" in raw:
        raise JamendoValidationError(f"{label} has an invalid size or encoding")
    try:
        return raw.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise JamendoValidationError(f"{label} is not ASCII") from exc


def _safe_git_ref(value: str) -> str:
    if (
        not value.startswith("refs/")
        or value.endswith("/")
        or "//" in value
        or "\\" in value
        or any(part in ("", ".", "..") for part in value.split("/"))
        or re.fullmatch(r"[A-Za-z0-9._/-]+", value) is None
    ):
        raise JamendoValidationError("Git HEAD contains an unsafe reference")
    return value


def _resolve_git_head(git_dir: Path) -> str:
    head_path = _require_concrete_file(git_dir, "HEAD", "Git HEAD")
    head = _read_small_ascii(head_path, "Git HEAD")
    if head.startswith("ref: "):
        ref = _safe_git_ref(head[5:])
        loose = git_dir.joinpath(*PurePosixPath(ref).parts)
        if loose.is_file():
            commit = _read_small_ascii(
                _require_concrete_file(git_dir, ref, "Git HEAD reference"),
                "Git HEAD reference",
            )
        else:
            packed_path = _require_concrete_file(
                git_dir, "packed-refs", "packed Git references"
            )
            commit = ""
            for line in _read_small_ascii(
                packed_path, "packed Git references", max_bytes=16 * 1024 * 1024
            ).splitlines():
                if not line or line.startswith(("#", "^")):
                    continue
                fields = line.split(" ")
                if len(fields) == 2 and fields[1] == ref:
                    commit = fields[0]
                    break
    else:
        commit = head
    commit = commit.lower()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise JamendoValidationError("cannot resolve official metadata Git commit")
    return commit


@dataclass(frozen=True)
class _GitIndexEntry:
    mode: int
    object_id: bytes
    path: str


def _read_git_index(git_dir: Path) -> Tuple[_GitIndexEntry, ...]:
    index_path = _require_concrete_file(git_dir, "index", "Git index")
    try:
        raw = index_path.read_bytes()
    except OSError as exc:
        raise JamendoValidationError("cannot read official metadata Git index") from exc
    if len(raw) < 32 or raw[:4] != b"DIRC":
        raise JamendoValidationError("official metadata Git index is invalid")
    if hashlib.sha1(raw[:-20]).digest() != raw[-20:]:
        raise JamendoValidationError("official metadata Git index checksum mismatch")
    version = int.from_bytes(raw[4:8], "big")
    if version not in (2, 3):
        raise JamendoValidationError(
            f"unsupported official metadata Git index version {version}"
        )
    count = int.from_bytes(raw[8:12], "big")
    offset = 12
    entries = []
    seen = set()
    for _ in range(count):
        start = offset
        if offset + 62 > len(raw) - 20:
            raise JamendoValidationError("official metadata Git index is truncated")
        fixed = raw[offset : offset + 62]
        mode = int.from_bytes(fixed[24:28], "big")
        object_id = fixed[40:60]
        flags = int.from_bytes(fixed[60:62], "big")
        offset += 62
        if flags & 0x4000:
            if version < 3 or offset + 2 > len(raw) - 20:
                raise JamendoValidationError("Git index has invalid extended flags")
            offset += 2
        if (flags >> 12) & 0x3:
            raise JamendoValidationError("official metadata Git index is unmerged")
        encoded_length = flags & 0x0FFF
        nul = raw.find(b"\x00", offset, len(raw) - 20)
        if nul < 0:
            raise JamendoValidationError("Git index path is not terminated")
        path_bytes = raw[offset:nul]
        if encoded_length != 0x0FFF and encoded_length != len(path_bytes):
            raise JamendoValidationError("Git index path length is inconsistent")
        try:
            path = path_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise JamendoValidationError("Git index path is not UTF-8") from exc
        path = safe_relative_path(path)
        if path.casefold() in seen:
            raise JamendoValidationError("Git index contains duplicate/colliding paths")
        seen.add(path.casefold())
        entry_prefix = offset - start
        entry_size = ((entry_prefix + len(path_bytes) + 8) // 8) * 8
        offset = start + entry_size
        if offset > len(raw) - 20:
            raise JamendoValidationError("Git index entry padding is invalid")
        if mode not in (0o100644, 0o100755, 0o120000, 0o160000):
            raise JamendoValidationError(f"unsupported tracked Git mode {mode:o}")
        entries.append(_GitIndexEntry(mode=mode, object_id=object_id, path=path))
    return tuple(entries)


def _git_tree_id(entries: Sequence[_GitIndexEntry]) -> str:
    root: Dict[bytes, object] = {}
    for entry in entries:
        parts = tuple(part.encode("utf-8") for part in PurePosixPath(entry.path).parts)
        node = root
        for part in parts[:-1]:
            existing = node.setdefault(part, {})
            if not isinstance(existing, dict):
                raise JamendoValidationError("tracked Git path conflicts with a file")
            node = existing
        if parts[-1] in node:
            raise JamendoValidationError("tracked Git path occurs more than once")
        node[parts[-1]] = entry

    def hash_tree(node: Mapping[bytes, object]) -> bytes:
        records = []
        for name, value in node.items():
            if isinstance(value, dict):
                mode = b"40000"
                object_id = hash_tree(value)
                sort_name = name + b"/"
            else:
                assert isinstance(value, _GitIndexEntry)
                mode = f"{value.mode:o}".encode("ascii")
                object_id = value.object_id
                sort_name = name
            records.append((sort_name, mode + b" " + name + b"\x00" + object_id))
        content = b"".join(record for _sort_name, record in sorted(records))
        header = f"tree {len(content)}\0".encode("ascii")
        return hashlib.sha1(header + content).digest()

    return hash_tree(root).hex()


def _git_blob_id(path: Path) -> bytes:
    digest = hashlib.sha1()
    try:
        size = path.stat().st_size
        digest.update(f"blob {size}\0".encode("ascii"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise JamendoValidationError(f"cannot hash tracked metadata file: {path}") from exc
    return digest.digest()


def _git_blob_id_from_bytes(payload: bytes) -> bytes:
    digest = hashlib.sha1()
    digest.update(f"blob {len(payload)}\0".encode("ascii"))
    digest.update(payload)
    return digest.digest()


def _git_symlink_blob_id(metadata_root: Path, relative: str) -> bytes:
    """Hash a tracked symlink payload without dereferencing the link."""
    safe = safe_relative_path(relative)
    parts = PurePosixPath(safe).parts
    candidate = metadata_root.joinpath(*parts)
    current = metadata_root
    for part in parts[:-1]:
        current = current / part
        if _is_link(current):
            raise JamendoValidationError(
                f"tracked metadata file contains a symlink or junction: {current}"
            )
    try:
        info = candidate.lstat()
    except OSError as exc:
        raise JamendoValidationError(
            f"tracked metadata file is missing or escapes its root: {candidate}"
        ) from exc
    if candidate.is_symlink():
        try:
            target = os.readlink(candidate)
        except OSError as exc:
            raise JamendoValidationError(
                f"cannot read tracked metadata symlink: {candidate}"
            ) from exc
        return _git_blob_id_from_bytes(os.fsencode(target))
    if _is_link(candidate):
        raise JamendoValidationError(
            f"tracked metadata file contains an unsupported reparse point: {candidate}"
        )
    if not stat.S_ISREG(info.st_mode):
        raise JamendoValidationError(
            f"tracked metadata file is not a symlink or regular placeholder: {candidate}"
        )
    return _git_blob_id(candidate)


def _git_crlf_normalized_blob_id(path: Path) -> Optional[bytes]:
    """Hash Git's built-in CRLF-to-LF checkout form without running filters."""
    try:
        raw_size = path.stat().st_size
        crlf_count = 0
        has_nul = False
        previous_cr = False
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                has_nul = has_nul or b"\x00" in block
                if previous_cr and block.startswith(b"\n"):
                    crlf_count += 1
                crlf_count += block.count(b"\r\n")
                previous_cr = block.endswith(b"\r")
        if has_nul or not crlf_count:
            return None
        digest = hashlib.sha1()
        digest.update(f"blob {raw_size - crlf_count}\0".encode("ascii"))
        pending = b""
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                value = pending + block
                if value.endswith(b"\r"):
                    pending = b"\r"
                    value = value[:-1]
                else:
                    pending = b""
                digest.update(value.replace(b"\r\n", b"\n"))
        if pending:
            digest.update(pending)
        return digest.digest()
    except OSError as exc:
        raise JamendoValidationError(f"cannot hash tracked metadata file: {path}") from exc


def _verify_clean_worktree(
    metadata_root: Path, entries: Sequence[_GitIndexEntry]
) -> None:
    for entry in entries:
        if entry.mode == 0o160000:
            raise JamendoValidationError("metadata checkout may not contain Git submodules")
        if entry.mode == 0o120000:
            object_id = _git_symlink_blob_id(metadata_root, entry.path)
            if object_id != entry.object_id:
                raise JamendoValidationError(
                    f"official metadata checkout has tracked modifications: {entry.path}"
                )
            continue
        path = _require_concrete_file(
            metadata_root, entry.path, "tracked metadata file"
        )
        if (
            _git_blob_id(path) != entry.object_id
            and _git_crlf_normalized_blob_id(path) != entry.object_id
        ):
            raise JamendoValidationError(
                f"official metadata checkout has tracked modifications: {entry.path}"
            )


def _metadata_commit(metadata_root: Path, *, production: bool) -> str:
    """Validate pinned Git provenance without executing the Git program."""
    git_dir = metadata_root / ".git"
    if not git_dir.exists():
        if production:
            raise JamendoValidationError("official metadata provenance requires a Git checkout")
        return "synthetic-fixture"
    if not git_dir.is_dir() or _is_link(git_dir):
        raise JamendoValidationError("official metadata .git must be a concrete directory")
    for redirect in ("commondir", "objects/info/alternates"):
        if (git_dir / Path(redirect)).exists():
            raise JamendoValidationError(
                f"official metadata Git repository may not use {redirect}"
            )
    commit = _resolve_git_head(git_dir)
    if production and commit != EXPECTED_METADATA_COMMIT:
        raise JamendoValidationError(
            f"official metadata commit drift: expected {EXPECTED_METADATA_COMMIT}, got {commit}"
        )
    if production:
        entries = _read_git_index(git_dir)
        tree = _git_tree_id(entries)
        if tree != EXPECTED_METADATA_TREE:
            raise JamendoValidationError(
                f"official metadata index tree drift: expected {EXPECTED_METADATA_TREE}, "
                f"got {tree}"
            )
        _verify_clean_worktree(metadata_root, entries)
    return commit


def _explicit_autotagging_source(data_root: Path) -> Path:
    """Handle Git's 31-byte Windows symlink placeholder without following it."""
    placeholder = data_root / "autotagging.tsv"
    concrete = data_root / "raw_30s_cleantags_50artists.tsv"
    if placeholder.is_file() and not _is_link(placeholder):
        raw = placeholder.read_bytes()
        if len(raw) == 31:
            try:
                target = raw.decode("utf-8").strip()
            except UnicodeDecodeError as exc:
                raise JamendoValidationError("autotagging placeholder is not UTF-8") from exc
            if target != "raw_30s_cleantags_50artists.tsv":
                raise JamendoValidationError(
                    "unexpected 31-byte autotagging link placeholder target"
                )
            return _require_concrete_file(
                data_root,
                "raw_30s_cleantags_50artists.tsv",
                "concrete autotagging metadata",
            )
    if _is_link(placeholder):
        # Dataset paths are never followed through links.  The concrete official
        # file is present in normal checkouts and is always preferred.
        return _require_concrete_file(
            data_root,
            "raw_30s_cleantags_50artists.tsv",
            "concrete autotagging metadata",
        )
    if placeholder.is_file() and placeholder.stat().st_size > 31:
        return _require_concrete_file(data_root, "autotagging.tsv", "autotagging metadata")
    return _require_concrete_file(
        data_root,
        "raw_30s_cleantags_50artists.tsv",
        "concrete autotagging metadata",
    )


def load_jamendo_context(
    metadata_root: Path,
    audio_root: Path,
    state_root: Path,
    *,
    production: bool = True,
    expected_tracks: Optional[int] = None,
    expected_archives: Optional[int] = None,
    fold_indices: Sequence[int] = (0, 1, 2, 3, 4),
    verify_audio_hashes: bool = False,
) -> JamendoContext:
    """Load and fully audit a local full-track MTG-Jamendo collection.

    ``production=True`` pins the official repository commit, manifest hashes,
    and 55,701/100 cardinalities.  Fixture callers may pass smaller expected
    counts, but still receive all path, marker, join, license, and split audits.
    """
    metadata_root = _require_local_root(Path(metadata_root), "metadata root")
    audio_root = _require_local_root(Path(audio_root), "audio root")
    state_root = _require_local_root(Path(state_root), "state root")
    data_root = _require_local_root(metadata_root / "data", "metadata data root")
    manifest_root = _require_local_root(data_root / "download", "manifest root")

    track_count = EXPECTED_TRACKS if expected_tracks is None else expected_tracks
    archive_count = EXPECTED_ARCHIVES if expected_archives is None else expected_archives
    if production and (track_count != EXPECTED_TRACKS or archive_count != EXPECTED_ARCHIVES):
        raise JamendoValidationError("production cardinalities cannot be overridden")
    if production and tuple(fold_indices) != (0, 1, 2, 3, 4):
        raise JamendoValidationError("production must audit all five official folds")
    if len(set(int(index) for index in fold_indices)) != len(fold_indices):
        raise JamendoValidationError("fold indices must be unique")

    archive_manifest_path = _require_concrete_file(
        manifest_root, _ARCHIVE_MANIFEST, "official archive manifest"
    )
    track_manifest_path = _require_concrete_file(
        manifest_root, _TRACK_MANIFEST, "official track manifest"
    )
    archives, archive_manifest_hash = _read_checksum_manifest(
        archive_manifest_path, archives=True
    )
    track_entries_list, track_manifest_hash = _read_checksum_manifest(
        track_manifest_path, archives=False
    )
    if len(archives) != archive_count or len(track_entries_list) != track_count:
        raise JamendoValidationError(
            f"official manifest cardinality mismatch: archives={len(archives)}, "
            f"tracks={len(track_entries_list)}"
        )
    if production and archive_manifest_hash != EXPECTED_ARCHIVE_MANIFEST_SHA256:
        raise JamendoValidationError("official archive manifest content hash drift")
    if production and track_manifest_hash != EXPECTED_TRACK_MANIFEST_SHA256:
        raise JamendoValidationError("official track manifest content hash drift")

    # This is intentionally before metadata/model work: an incomplete collection
    # must fail closed as soon as the manifest identities are known.
    completion = validate_completion_marker(
        state_root,
        archive_manifest_sha256=archive_manifest_hash,
        track_manifest_sha256=track_manifest_hash,
        expected_archives=archive_count,
        expected_tracks=track_count,
    )
    track_manifest = {entry.path: entry.sha256 for entry in track_entries_list}
    track_bytes = _audit_archive_markers(
        state_root,
        archives,
        track_manifest,
        archive_manifest_sha256=archive_manifest_hash,
        track_manifest_sha256=track_manifest_hash,
    )
    if sum(track_bytes.values()) != _require_int(
        completion.get("track_bytes"), "completion track_bytes"
    ):
        raise JamendoValidationError("completion marker byte total differs from archive markers")

    metadata_commit = _metadata_commit(metadata_root, production=production)
    raw_path = _require_concrete_file(data_root, "raw_30s.tsv", "raw_30s metadata")
    meta_path = _require_concrete_file(data_root, "raw.meta.tsv", "descriptive metadata")
    license_path = _require_concrete_file(
        metadata_root, "audio_licenses.txt", "audio licenses"
    )
    autotagging_path = _explicit_autotagging_source(data_root)

    metadata = _parse_metadata_tracks(raw_path)
    autotagging = _parse_metadata_tracks(autotagging_path)
    descriptive = _parse_descriptive_metadata(meta_path)
    licenses = _parse_licenses(license_path)
    if len(metadata) != track_count:
        raise JamendoValidationError(
            f"raw_30s metadata count mismatch: expected {track_count}, got {len(metadata)}"
        )
    metadata_paths = {str(value["relative_path"]) for value in metadata.values()}
    if metadata_paths != set(track_manifest):
        missing = sorted(set(track_manifest) - metadata_paths)[:3]
        extra = sorted(metadata_paths - set(track_manifest))[:3]
        raise JamendoValidationError(
            f"metadata/track manifest join mismatch; missing={missing}, extra={extra}"
        )
    if metadata_paths != set(licenses):
        raise JamendoValidationError(
            "license/metadata path join mismatch; "
            f"missing={sorted(metadata_paths - set(licenses))[:3]}, "
            f"extra={sorted(set(licenses) - metadata_paths)[:3]}"
        )
    if not set(autotagging).issubset(metadata):
        raise JamendoValidationError("autotagging metadata is not a raw_30s subset")
    for track_id, tagged in autotagging.items():
        source = metadata[track_id]
        for field in ("artist_id", "album_id", "relative_path"):
            if tagged[field] != source[field]:
                raise JamendoValidationError(
                    f"autotagging {field} drift for track {track_id}"
                )

    folds = tuple(
        _parse_fold(data_root / "splits", int(index), metadata)
        for index in fold_indices
    )
    if folds:
        split_track_ids = set(folds[0].track_parts)
        if any(set(fold.track_parts) != split_track_ids for fold in folds[1:]):
            raise JamendoValidationError("official folds do not cover the same track set")
        if any(dict(fold.track_tags) != dict(folds[0].track_tags) for fold in folds[1:]):
            raise JamendoValidationError(
                "official folds do not preserve aligned per-track tag labels"
            )
        if not split_track_ids.issubset(autotagging):
            raise JamendoValidationError("official folds are not an autotagging subset")
        if production and len(split_track_ids) != EXPECTED_SPLIT_TRACKS:
            raise JamendoValidationError(
                f"expected {EXPECTED_SPLIT_TRACKS} official split tracks, "
                f"found {len(split_track_ids)}"
            )
    fold_lookup = [fold.track_parts for fold in folds]
    tracks = []
    for row_index, track_id in enumerate(sorted(metadata)):
        source = metadata[track_id]
        relative = str(source["relative_path"])
        detail = descriptive.get(track_id)
        if detail is None:
            raise JamendoValidationError(f"missing descriptive metadata for track {track_id}")
        if int(detail["artist_id"]) != int(source["artist_id"]):
            raise JamendoValidationError(f"artist join mismatch for track {track_id}")
        if int(detail["album_id"]) != int(source["album_id"]):
            raise JamendoValidationError(f"album join mismatch for track {track_id}")
        audio_path = _require_concrete_file(audio_root, relative, "audio track")
        expected_size = track_bytes[relative]
        if audio_path.stat().st_size != expected_size:
            raise JamendoValidationError(f"audio byte-size drift for {relative}")
        expected_hash = track_manifest[relative]
        if verify_audio_hashes and sha256_file(audio_path) != expected_hash:
            raise JamendoValidationError(f"audio SHA-256 drift for {relative}")
        tracks.append(
            JamendoTrack(
                row_index=row_index,
                track_id=track_id,
                artist_id=int(source["artist_id"]),
                album_id=int(source["album_id"]),
                relative_path=relative,
                audio_path=audio_path,
                duration_seconds=float(source["duration"]),
                tags=tuple(source["tags"]),
                title=detail["title"],
                artist_name=detail["artist_name"],
                album_name=detail["album_name"],
                release_date=detail["release_date"],
                jamendo_url=detail["url"],
                license=licenses[relative],
                expected_audio_sha256=expected_hash,
                expected_audio_bytes=expected_size,
                fold_parts=tuple(mapping.get(track_id) for mapping in fold_lookup),
            )
        )

    metadata_hashes = {
        "data/raw_30s.tsv": sha256_file(raw_path),
        "data/raw.meta.tsv": sha256_file(meta_path),
        "audio_licenses.txt": sha256_file(license_path),
        f"data/{autotagging_path.name}": sha256_file(autotagging_path),
    }
    for fold in folds:
        for part in _PARTS:
            relative = f"splits/split-{fold.index}/autotagging-{part}.tsv"
            metadata_hashes[f"data/{relative}"] = sha256_file(data_root / relative)
    fingerprint = _canonical_hash(
        {
            "schema_version": 1,
            "evidence_scope": EVIDENCE_SCOPE,
            "metadata_commit": metadata_commit,
            "archive_manifest_sha256": archive_manifest_hash,
            "track_manifest_sha256": track_manifest_hash,
            "metadata_hashes": metadata_hashes,
            "track_count": len(tracks),
        }
    )
    return JamendoContext(
        tracks=tuple(tracks),
        folds=folds,
        metadata_root=metadata_root,
        audio_root=audio_root,
        state_root=state_root,
        metadata_commit=metadata_commit,
        archive_manifest_sha256=archive_manifest_hash,
        track_manifest_sha256=track_manifest_hash,
        metadata_hashes=MappingProxyType(metadata_hashes),
        source_fingerprint=fingerprint,
    )
