"""Deterministic, offline coverage audit for the packaged vibe index.

The index stores artists but no genre labels.  Consequently, category results are
explicitly *curated artist-anchor proxies*, never inferred genre classifications.
This module only reads local JSON/NPZ files and never constructs a network client.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np


class CoverageAuditError(ValueError):
    """Raised when an index, target configuration, or crawl plan is malformed."""


def normalize_artist_name(value: str) -> str:
    """Return a stable, accent- and punctuation-insensitive artist lookup key."""
    decomposed = unicodedata.normalize("NFKD", value)
    without_marks = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"[\W_]+", " ", without_marks.casefold()).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json_object(path: Path, label: str) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CoverageAuditError(f"Cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise CoverageAuditError(f"{label} must contain a JSON object: {path}")
    return value


def _required_string(value: Mapping[str, Any], key: str, label: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result.strip():
        raise CoverageAuditError(f"{label}.{key} must be a non-empty string")
    return result


def _required_positive_int(value: Mapping[str, Any], key: str, label: str) -> int:
    result = value.get(key)
    if isinstance(result, bool) or not isinstance(result, int) or result < 1:
        raise CoverageAuditError(f"{label}.{key} must be a positive integer")
    return result


def _normalized_aliases(artist: str, aliases: Iterable[str], label: str) -> List[str]:
    names = [artist, *aliases]
    normalized = [normalize_artist_name(name) for name in names]
    if not all(normalized):
        raise CoverageAuditError(f"{label} has an empty artist alias")
    return sorted(set(normalized))


def _validated_targets(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    if config.get("schema_version") != 1:
        raise CoverageAuditError("targets.schema_version must be 1")
    _required_string(config, "category_model", "targets")
    categories = config.get("categories")
    if not isinstance(categories, list) or not categories:
        raise CoverageAuditError("targets.categories must be a non-empty list")

    result: List[Dict[str, Any]] = []
    known_artists: Dict[str, str] = {}
    for raw_category in categories:
        if not isinstance(raw_category, dict):
            raise CoverageAuditError("each category must be an object")
        category = _required_string(raw_category, "name", "category")
        description = _required_string(raw_category, "description", f"category {category!r}")
        anchors = raw_category.get("anchors")
        if not isinstance(anchors, list) or not anchors:
            raise CoverageAuditError(f"category {category!r}.anchors must be a non-empty list")
        validated_anchors = []
        for raw_anchor in anchors:
            if not isinstance(raw_anchor, dict):
                raise CoverageAuditError(f"category {category!r} anchor must be an object")
            artist = _required_string(raw_anchor, "artist", f"category {category!r} anchor")
            aliases = raw_anchor.get("aliases", [])
            if not isinstance(aliases, list) or not all(isinstance(item, str) for item in aliases):
                raise CoverageAuditError(f"artist {artist!r}.aliases must be a list of strings")
            normalized = _normalized_aliases(artist, aliases, f"artist {artist!r}")
            for key in normalized:
                owner = known_artists.get(key)
                if owner and owner != artist:
                    raise CoverageAuditError(f"alias {key!r} is assigned to both {owner!r} and {artist!r}")
                known_artists[key] = artist
            validated_anchors.append({
                "artist": artist,
                "aliases": sorted(set(aliases)),
                "lookup_keys": normalized,
                "minimum_tracks": _required_positive_int(raw_anchor, "minimum_tracks", f"artist {artist!r}"),
            })
        result.append({"name": category, "description": description, "anchors": validated_anchors})
    return sorted(result, key=lambda item: item["name"].casefold())


def _json_index_artists(index: Mapping[str, Any]) -> List[str]:
    entries = index.get("entries")
    if not isinstance(entries, list):
        raise CoverageAuditError("index.entries must be a list")
    artists = []
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise CoverageAuditError(f"index.entries[{position}] must be an object")
        artist = entry.get("artist")
        if not isinstance(artist, str) or not normalize_artist_name(artist):
            raise CoverageAuditError(f"index.entries[{position}].artist must be a non-empty string")
        artists.append(artist)
    return artists


def _index_artist_counts(index_path: Path) -> Tuple[Counter, int, str, int]:
    if index_path.suffix.casefold() == ".npz":
        try:
            with np.load(index_path, allow_pickle=False) as index:
                if "artists" not in index.files:
                    raise CoverageAuditError("NPZ index must contain an artists array")
                artists = index["artists"]
        except (OSError, ValueError) as exc:
            raise CoverageAuditError(f"Cannot read index: {index_path}") from exc
        if artists.ndim != 1:
            raise CoverageAuditError("NPZ index artists must be one-dimensional")
        values = artists.astype(str).tolist()
        index_format = "npz"
    else:
        index = _load_json_object(index_path, "index")
        values = _json_index_artists(index)
        index_format = "json"

    counts: Counter = Counter()
    unknown = 0
    for position, artist in enumerate(values):
        if not isinstance(artist, str):
            raise CoverageAuditError(
                f"index artist at position {position} must be a string"
            )
        normalized = normalize_artist_name(artist)
        if not normalized:
            unknown += 1
            continue
        counts[normalized] += 1
    return counts, len(values), index_format, unknown


def _artist_result(category: str, anchor: Mapping[str, Any], counts: Counter) -> Dict[str, Any]:
    observed = sum(counts[key] for key in anchor["lookup_keys"])
    minimum = anchor["minimum_tracks"]
    if observed == 0:
        status = "missing"
    elif observed < minimum:
        status = "thin"
    else:
        status = "covered"
    return {
        "category": category,
        "artist": anchor["artist"],
        "aliases": anchor["aliases"],
        "observed": observed,
        "minimum": minimum,
        "status": status,
    }


def build_audit_report(index_path: Path, targets_path: Path) -> Dict[str, Any]:
    """Build a stable report from a local JSON/NPZ index without network access."""
    index_path = Path(index_path)
    targets_path = Path(targets_path)
    targets = _load_json_object(targets_path, "targets")
    categories = _validated_targets(targets)
    artist_counts, entry_count, index_format, unknown_artists = (
        _index_artist_counts(index_path)
    )

    artist_results: List[Dict[str, Any]] = []
    category_results: List[Dict[str, Any]] = []
    for category in categories:
        results = [_artist_result(category["name"], anchor, artist_counts) for anchor in category["anchors"]]
        artist_results.extend(results)
        category_results.append({
            "category": category["name"],
            "description": category["description"],
            "anchor_count": len(results),
            "artists_present": sum(item["observed"] > 0 for item in results),
            "artists_meeting_minimum": sum(item["status"] == "covered" for item in results),
            "tracks_observed": sum(item["observed"] for item in results),
            "artist_presence_ratio": sum(item["observed"] > 0 for item in results) / len(results),
            "minimum_coverage_ratio": sum(item["status"] == "covered" for item in results) / len(results),
        })

    artist_results.sort(key=lambda item: (item["category"].casefold(), item["artist"].casefold()))
    priority = {"missing": 0, "thin": 1}
    crawl_plan = []
    for item in artist_results:
        if item["status"] not in priority:
            continue
        deficit = item["minimum"] - item["observed"]
        crawl_plan.append({
            "artist": item["artist"],
            "category": item["category"],
            "observed": item["observed"],
            "minimum": item["minimum"],
            "reason": item["status"],
            "budget": {
                "artists": 1,
                "tracks": deficit,
                "api_calls": 2 + deficit,
            },
        })
    crawl_plan.sort(key=lambda item: (
        priority[item["reason"]], item["category"].casefold(), item["artist"].casefold()
    ))

    return {
        "schema_version": 1,
        "category_model": targets["category_model"],
        "index": {
            "sha256": _sha256(index_path),
            "entries": entry_count,
            "format": index_format,
            "unknown_artist_entries": unknown_artists,
        },
        "targets": {"sha256": _sha256(targets_path), "schema_version": targets["schema_version"]},
        "artist_presence_and_thinness": artist_results,
        "category_proxy_coverage": category_results,
        "targeted_crawl_plan": crawl_plan,
    }


def load_targeted_crawl_plan(path: Path) -> List[Dict[str, Any]]:
    """Load and validate the bounded plan emitted by :func:`build_audit_report`."""
    report = _load_json_object(Path(path), "audit report")
    plan = report.get("targeted_crawl_plan")
    if not isinstance(plan, list):
        raise CoverageAuditError("audit report.targeted_crawl_plan must be a list")
    validated = []
    for position, item in enumerate(plan):
        if not isinstance(item, dict):
            raise CoverageAuditError(f"crawl plan item {position} must be an object")
        artist = _required_string(item, "artist", f"crawl plan item {position}")
        category = _required_string(item, "category", f"crawl plan item {position}")
        observed = item.get("observed")
        minimum = item.get("minimum")
        if isinstance(observed, bool) or not isinstance(observed, int) or observed < 0:
            raise CoverageAuditError(f"crawl plan item {position}.observed must be a non-negative integer")
        if isinstance(minimum, bool) or not isinstance(minimum, int) or minimum <= observed:
            raise CoverageAuditError(f"crawl plan item {position}.minimum must exceed observed")
        reason = item.get("reason")
        if reason not in {"missing", "thin"}:
            raise CoverageAuditError(f"crawl plan item {position}.reason must be missing or thin")
        validated.append({"artist": artist, "category": category, "observed": observed,
                          "minimum": minimum, "reason": reason})
    return sorted(validated, key=lambda item: (
        0 if item["reason"] == "missing" else 1,
        item["category"].casefold(), item["artist"].casefold(),
    ))


def _default_data_file(name: str) -> Path:
    return Path(__file__).resolve().parents[1] / "data" / name


def main(argv: Optional[List[str]] = None) -> int:
    """Write an offline audit report; this command never makes network calls."""
    parser = argparse.ArgumentParser(description="Audit local artist-anchor coverage without network access.")
    parser.add_argument("--index", type=Path, default=_default_data_file("deepvibe_index.npz"))
    parser.add_argument(
        "--targets",
        type=Path,
        default=_default_data_file("coverage_targets.v1.json"),
    )
    parser.add_argument("--output", type=Path, required=True, help="Path for the deterministic JSON report.")
    args = parser.parse_args(argv)

    report = build_audit_report(args.index, args.targets)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
