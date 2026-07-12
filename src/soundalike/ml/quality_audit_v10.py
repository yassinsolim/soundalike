"""Reproducible real-catalog audit for generic version/canonical filtering."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .magnatagatune_v10 import sha256_path
from .quality_filter import TitleQualityFilter


EXPLICIT_DERIVATIVE = re.compile(
    r"\b(?:slowed|reverb|nightcore|mashup|medley|chopnotslop)\b"
    r"|\b(?:sped|speed)[- ]up\b"
    r"|\bkara(?:oke|ōke)\s+(?:version|mix|edit|track|instrumental)\b"
    r"|\b(?:cover|tribute|instrumental)\s+(?:version|recording|track|mix)\b"
    r"|(?:\(|\[)[^)\]]*\b(?:remix|club mix|radio mix|extended mix|"
    r"rework|bootleg|vip edit)\b[^)\]]*(?:\)|\])"
    r"|\s+-\s+[^-]*\b(?:remix|club mix|radio mix|extended mix)\b[^-]*$",
    re.IGNORECASE,
)
LEGITIMATE_CONTROLS = {
    ("cover me", "bruce springsteen"),
    ("love x love", "george benson"),
    ("a tribute to someone", "herbie hancock"),
    ("karaoke", "drake"),
    ("karaoke", "cass mccombs"),
    ("karaoke bar", "angus & julia stone"),
}


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def audit_real_index(
    index_path: str | Path,
    output_path: str | Path,
    *,
    sample_size: int = 25,
) -> dict[str, Any]:
    quality = TitleQualityFilter()
    with np.load(index_path, allow_pickle=False) as index:
        titles = index["titles"].astype(str)
        artists = index["artists"].astype(str)
        track_ids = index["track_ids"].tolist()
    kept = quality.keep_mask(titles, artists)
    independently_explicit = np.asarray([
        bool(EXPLICIT_DERIVATIVE.search(f"{title} {artist}"))
        for title, artist in zip(titles, artists)
    ])
    false_negative_rows = np.where(independently_explicit & kept)[0]
    legitimate_rows = np.asarray([
        (title.casefold(), artist.casefold()) in LEGITIMATE_CONTROLS
        for title, artist in zip(titles, artists)
    ])
    false_positive_rows = np.where(legitimate_rows & ~kept)[0]

    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    title_artists: dict[str, set[str]] = defaultdict(set)
    for row, (title, artist) in enumerate(zip(titles, artists)):
        canonical = quality.canonical_title(title)
        artist_key = " ".join(artist.casefold().split())
        if canonical:
            groups[(canonical, artist_key)].append(row)
            title_artists[canonical].add(artist_key)
    canonical_preference_groups = []
    for (canonical, artist), rows in groups.items():
        priorities = [quality.version_priority(titles[row], artists[row]) for row in rows]
        if len(rows) > 1 and min(priorities) == 0 and max(priorities) >= 20:
            winner = rows[priorities.index(min(priorities))]
            canonical_preference_groups.append({
                "canonical_title": canonical,
                "artist": artist,
                "available_versions": len(rows),
                "preferred_track_id": track_ids[winner],
                "preferred_title": titles[winner],
                "rejected_derivative_titles": [
                    titles[row] for row, priority in zip(rows, priorities)
                    if priority >= 20
                ][:5],
            })
    filtered_rows = np.where(~kept)[0]

    def samples(rows: Sequence[int]) -> list[dict[str, Any]]:
        return [{
            "row": int(row),
            "track_id": track_ids[int(row)],
            "title": titles[int(row)],
            "artist": artists[int(row)],
            "version_tags": sorted(
                quality.version_tags(titles[int(row)], artists[int(row)])
            ),
        } for row in list(rows)[:sample_size]]

    report: dict[str, Any] = {
        "schema_version": 10,
        "kind": "real-catalog-generic-version-quality-audit",
        "index_path": str(index_path),
        "index_sha256": sha256_path(index_path),
        "catalog_rows": len(titles),
        "generic_rules_only": True,
        "artist_specific_rules": False,
        "filtered_rows": int((~kept).sum()),
        "filtered_fraction": float((~kept).mean()),
        "independent_explicit_derivative_rows": int(independently_explicit.sum()),
        "explicit_derivative_false_negatives": int(len(false_negative_rows)),
        "explicit_derivative_false_negative_samples": samples(false_negative_rows),
        "curated_legitimate_controls_present": int(legitimate_rows.sum()),
        "curated_legitimate_false_positives": int(len(false_positive_rows)),
        "curated_legitimate_false_positive_samples": samples(false_positive_rows),
        "filtered_samples": samples(filtered_rows),
        "canonical_original_preference_groups": len(canonical_preference_groups),
        "canonical_original_preference_samples": canonical_preference_groups[:sample_size],
        "cross_artist_same_canonical_title_groups": int(sum(
            len(values) > 1 for values in title_artists.values()
        )),
        "unlabelled_cover_policy": (
            "Do not guess original ownership from title/artist strings. Explicit "
            "cover metadata is filtered; unlabelled cross-artist recordings are "
            "flagged in blinded human evaluation. A future automatic exclusion "
            "requires trustworthy work/original-release attribution."
        ),
        "query_version_exception": (
            "canonical queries exclude derivatives; a derivative query may admit "
            "only candidates carrying the same derivative tag classes"
        ),
        "production_deployed": False,
    }
    report["content_sha256"] = hashlib.sha256(_canonical(report)).hexdigest()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index", default="ml_data/deepvibe_index_v5.npz", type=Path
    )
    parser.add_argument(
        "--output",
        default=(
            ".goals/human-quality-recommendations/artifacts/"
            "catalog-version-quality-audit-v10.json"
        ),
        type=Path,
    )
    args = parser.parse_args(argv)
    report = audit_real_index(args.index, args.output)
    print(json.dumps({
        key: report[key] for key in (
            "catalog_rows", "filtered_rows",
            "explicit_derivative_false_negatives",
            "curated_legitimate_false_positives",
            "canonical_original_preference_groups",
            "content_sha256",
        )
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
