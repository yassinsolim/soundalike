"""Build the unopened multi-positive benchmark used by protocol v7.

The candidate graph is trained from Last.fm-360K and Music4All-Onion.  This
benchmark therefore uses ListenBrainz session similarity as an independent
source of taste-affinity labels.  Existing v6 FINAL queries are expanded into
multi-positive DEV records; fresh, artist-disjoint queries become FINAL.

The command prints counts only.  It never prints FINAL identities or labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

import numpy as np
import requests

from .collaborative_benchmark import (
    LISTENBRAINZ_URL,
    USER_AGENT,
    _load_cache,
    _recording_mbids,
    _save_cache,
    _similar_recordings,
)
from .final_protocol import content_sha256
from .quality_filter import TitleQualityFilter
from .real_benchmark import PairResolver, credited_artists, normalize_text

_VERSION_RE = re.compile(
    r"(?i)\b(?:instrumental|karaoke|tribute|slowed|reverb|sped[ -]?up|"
    r"nightcore|cover|remix|mashup|medley|version|edit)\b"
)
_GRADE_POLICY = {
    "rank_1_to_3": 3,
    "rank_4_to_8": 2,
    "rank_9_to_12": 1,
}
V7_LISTENBRAINZ_ALGORITHM = (
    "session_based_days_180_session_300_contribution_5_threshold_15_limit_50_skip_30"
)
_DEEZER_GENRES = {
    "rap/hip hop": "hip-hop",
    "r&b": "r&b-soul",
    "alternative": "indie-alternative",
    "rock": "rock",
    "metal": "metal",
    "jazz": "jazz",
    "electro": "electronic",
    "dance": "dance-electronic",
    "pop": "pop",
    "reggae": "reggae-dub-ska",
    "latin music": "latin",
    "country": "folk-country",
    "classical": "classical",
    "soul & funk": "funk-soul",
    "asian music": "asian-pop",
    "african music": "african",
    "blues": "blues",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _artists(value: str) -> Set[str]:
    return {normalize_text(name) for name in credited_artists(value)}


def _record_artists(record: Mapping[str, Any]) -> Set[str]:
    result = _artists(record["query"]["artist"])
    for positive in record["positives"]:
        result |= _artists(positive["artist"])
    return result


def _source_url(mbid: str) -> str:
    from urllib.parse import urlencode

    return LISTENBRAINZ_URL + "?" + urlencode(
        {"recording_mbids": mbid, "algorithm": V7_LISTENBRAINZ_ALGORITHM}
    )


def _grade(rank: int) -> int:
    if rank <= 3:
        return 3
    if rank <= 8:
        return 2
    return 1


def _resolve_positives(
    similar: Iterable[Mapping[str, Any]],
    resolver: PairResolver,
    quality_mask: np.ndarray,
    titles: np.ndarray,
    artists: np.ndarray,
    artist_rows: Mapping[str, Sequence[int]],
    query_row: int,
    blocked_artists: Set[str],
    minimum: int,
    maximum: int,
) -> List[Dict[str, Any]]:
    query_artist = _artists(str(artists[query_row]))
    query_title = str(titles[query_row])
    quality = TitleQualityFilter()
    positives: List[Dict[str, Any]] = []
    seen_tracks: Set[Tuple[str, str]] = set()
    seen_artists: Set[str] = set()
    for item in similar:
        raw_title = str(item.get("recording_name", ""))
        raw_artist = str(item.get("artist_credit_name", ""))
        target_artists = _artists(raw_artist)
        if (
            not raw_title
            or not target_artists
            or target_artists & (query_artist | blocked_artists | seen_artists)
            or _VERSION_RE.search(raw_title)
        ):
            continue
        exact_rows = resolver.target_rows(
            {"title": raw_title, "artist": raw_artist}
        )
        catalog_artist_rows = [
            int(row)
            for artist in target_artists
            for row in artist_rows.get(artist, [])
            if quality_mask[int(row)]
        ]
        target_row = next(
            (int(row) for row in exact_rows if quality_mask[int(row)]),
            min(
                catalog_artist_rows,
                key=lambda row: (
                    int(_VERSION_RE.search(str(titles[row])) is not None),
                    len(str(titles[row])),
                    row,
                ),
                default=None,
            ),
        )
        if target_row is None:
            continue
        title = str(titles[target_row])
        artist = str(artists[target_row])
        key = (normalize_text(title), normalize_text(artist))
        if (
            key in seen_tracks
            or quality.seed_title_in_result(query_title, title)
            or _VERSION_RE.search(title)
        ):
            continue
        rank = len(positives) + 1
        positives.append(
            {
                "title": title,
                "artist": artist,
                "recording_mbid": item.get("recording_mbid"),
                "source_recording_title": raw_title,
                "source_recording_artist": raw_artist,
                "relevance_scope": "artist",
                "exact_source_recording_in_catalog": bool(exact_rows),
                "grade": _grade(rank),
                "source_rank": rank,
                "source_score": int(item.get("score", 0)),
            }
        )
        seen_tracks.add(key)
        seen_artists |= target_artists
        if len(positives) >= maximum:
            break
    return positives if len(positives) >= minimum else []


def _resolve_deezer_artists(
    related: Sequence[Mapping[str, Any]],
    artist_rows: Mapping[str, Sequence[int]],
    quality_mask: np.ndarray,
    titles: np.ndarray,
    artists: np.ndarray,
    blocked_artists: Set[str],
    maximum: int,
) -> List[Dict[str, Any]]:
    positives: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for item in related:
        source_artist = str(item.get("name", ""))
        keys = _artists(source_artist)
        if not keys or keys & (blocked_artists | seen):
            continue
        rows = [
            int(row)
            for key in keys
            for row in artist_rows.get(key, [])
            if quality_mask[int(row)]
        ]
        row = min(
            rows,
            key=lambda value: (
                int(_VERSION_RE.search(str(titles[value])) is not None),
                len(str(titles[value])),
                value,
            ),
            default=None,
        )
        if row is None:
            continue
        rank = len(positives) + 1
        positives.append(
            {
                "title": str(titles[row]),
                "artist": str(artists[row]),
                "source_related_artist": source_artist,
                "source_artist_id": item.get("id"),
                "relevance_scope": "artist",
                "exact_source_recording_in_catalog": False,
                "grade": _grade(rank),
                "source_rank": rank,
                "source_score": None,
                "source_provider": "Deezer related artists",
            }
        )
        seen |= keys
        if len(positives) >= maximum:
            break
    return positives


def _build_record(
    seed: Mapping[str, str],
    split: str,
    number: int,
    resolver: PairResolver,
    session: requests.Session,
    cache: Dict[str, Any],
    cache_path: Path,
    titles: np.ndarray,
    artists: np.ndarray,
    track_ids: np.ndarray,
    artist_rows: Mapping[str, Sequence[int]],
    quality_mask: np.ndarray,
    blocked_artists: Set[str],
    minimum_positives: int,
    maximum_positives: int,
    known_mbid: str | None = None,
) -> Dict[str, Any] | None:
    query_row = resolver.query_row(seed)
    if query_row is None or not quality_mask[int(query_row)]:
        return None
    if _VERSION_RE.search(str(titles[int(query_row)])):
        return None
    track_id = str(int(track_ids[int(query_row)]))
    deezer_track = _deezer_json(
        session,
        f"https://api.deezer.com/track/{track_id}",
        cache,
        "deezer_tracks",
        track_id,
    )
    deezer_artist_id = str(deezer_track.get("artist", {}).get("id", ""))
    related_url = (
        f"https://api.deezer.com/artist/{deezer_artist_id}/related"
        if deezer_artist_id
        else ""
    )
    related = (
        _deezer_json(
            session,
            related_url,
            cache,
            "deezer_related",
            deezer_artist_id,
        ).get("data", [])
        if deezer_artist_id
        else []
    )
    deezer_positives = _resolve_deezer_artists(
        related,
        artist_rows,
        quality_mask,
        titles,
        artists,
        blocked_artists | _artists(str(artists[int(query_row)])),
        maximum_positives,
    )
    mbids = (
        [known_mbid]
        if known_mbid
        else _recording_mbids(
            session, seed["title"], seed["artist"], cache
        )
    )
    _save_cache(cache_path, cache)
    listenbrainz_positives: List[Dict[str, Any]] = []
    chosen_mbid = str(mbids[0]) if mbids else ""
    for mbid in mbids[:3]:
        similar = _similar_recordings(session, str(mbid), cache)
        _save_cache(cache_path, cache)
        listenbrainz_positives = _resolve_positives(
            similar,
            resolver,
            quality_mask,
            titles,
            artists,
            artist_rows,
            int(query_row),
            blocked_artists,
            1,
            maximum_positives,
        )
        if listenbrainz_positives:
            chosen_mbid = str(mbid)
            break
    combined: List[Dict[str, Any]] = []
    used_artists: Set[str] = set()
    for positive in (*deezer_positives, *listenbrainz_positives):
        keys = _artists(positive["artist"])
        if keys & used_artists:
            continue
        copied = dict(positive)
        copied["grade"] = _grade(len(combined) + 1)
        copied["source_rank"] = len(combined) + 1
        combined.append(copied)
        used_artists |= keys
        if len(combined) >= maximum_positives:
            break
    if len(combined) < minimum_positives:
        return None
    sources = [
        {
            "url": related_url,
            "publisher": "Deezer",
            "accessed_at": date.today().isoformat(),
            "source_class": "independent_service_related_artists",
            "excerpt": (
                "The public related-artists response supplies ranked, "
                "query-conditioned artist affinity."
            ),
        }
    ]
    if chosen_mbid and listenbrainz_positives:
        sources.append(
            {
                "url": _source_url(chosen_mbid),
                "publisher": "ListenBrainz Labs",
                "accessed_at": date.today().isoformat(),
                "source_class": (
                    "independent_human_listening_session_similarity"
                ),
                "excerpt": (
                    "The session-based similar-recordings response supplies "
                    "additional query-conditioned affinity evidence."
                ),
                "algorithm": V7_LISTENBRAINZ_ALGORITHM,
            }
        )
    prefix = "DEV-OPENED-MP" if split == "development" else "FINAL-MP"
    return {
        "id": f"{prefix}-{number:03d}",
        "split": split,
        "scene": seed["scene"],
        "catalog_tier": seed["catalog_tier"],
        "query": {
            "title": str(titles[int(query_row)]),
            "artist": str(artists[int(query_row)]),
            "recording_mbid": chosen_mbid or None,
            "deezer_track_id": int(track_id),
        },
        "positives": combined,
        "evidence_axis": "taste_affinity",
        "source": sources[0],
        "sources": sources,
    }


def _v6_dev_seeds(v6_path: Path) -> List[Dict[str, str]]:
    benchmark = json.loads(v6_path.read_text(encoding="utf-8"))
    seeds = []
    for pair in benchmark["pairs"]:
        if pair.get("split") != "final":
            continue
        query = pair["query"]
        if not query.get("recording_mbid"):
            continue
        seeds.append(
            {
                "title": query["title"],
                "artist": query["artist"],
                "scene": pair["scene"],
                "catalog_tier": pair.get("catalog_tier", "deep_cut"),
                "recording_mbid": query["recording_mbid"],
            }
        )
    return seeds


def _blocked_v6_artists(v6_path: Path) -> Set[str]:
    benchmark = json.loads(v6_path.read_text(encoding="utf-8"))
    result: Set[str] = set()
    for pair in benchmark["pairs"]:
        result |= _artists(pair["query"]["artist"])
        result |= _artists(pair["target"]["artist"])
    return result


def _balanced(records: Sequence[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
    by_scene: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        by_scene.setdefault(record["scene"], []).append(record)
    selected: List[Dict[str, Any]] = []
    depth = 0
    while len(selected) < count:
        added = False
        for scene in sorted(by_scene):
            if depth < len(by_scene[scene]):
                selected.append(by_scene[scene][depth])
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
        depth += 1
    return selected


def _deezer_json(
    session: requests.Session,
    url: str,
    cache: Dict[str, Any],
    cache_group: str,
    cache_key: str,
) -> Dict[str, Any]:
    group = cache.setdefault(cache_group, {})
    if cache_key not in group:
        response = session.get(url, timeout=30)
        response.raise_for_status()
        group[cache_key] = response.json()
        time.sleep(0.05)
    value = group[cache_key]
    return value if isinstance(value, dict) else {}


def discover_fresh_seeds(
    titles: np.ndarray,
    artists: np.ndarray,
    track_ids: np.ndarray,
    quality_mask: np.ndarray,
    blocked_artists: Set[str],
    session: requests.Session,
    cache: Dict[str, Any],
    cache_path: Path,
    per_scene: int = 18,
    minimum_scenes: int = 12,
) -> List[Dict[str, str]]:
    """Freeze a fresh query catalogue using Deezer genre metadata only."""
    rng = np.random.default_rng(20260712)
    candidate_rows = rng.permutation(len(titles))
    counts: Counter[str] = Counter()
    seen_artists: Set[str] = set()
    seeds: List[Dict[str, str]] = []
    for raw_row in candidate_rows[:5000]:
        row = int(raw_row)
        artist_key = normalize_text(str(artists[row]))
        if (
            not quality_mask[row]
            or artist_key in blocked_artists
            or artist_key in seen_artists
            or _VERSION_RE.search(str(titles[row]))
        ):
            continue
        track_id = str(int(track_ids[row]))
        track = _deezer_json(
            session,
            f"https://api.deezer.com/track/{track_id}",
            cache,
            "deezer_tracks",
            track_id,
        )
        album_id = str(track.get("album", {}).get("id", ""))
        if not album_id or track.get("error"):
            continue
        album = _deezer_json(
            session,
            f"https://api.deezer.com/album/{album_id}",
            cache,
            "deezer_albums",
            album_id,
        )
        raw_genres = [
            normalize_text(item.get("name", ""))
            for item in album.get("genres", {}).get("data", [])
        ]
        scene = next(
            (
                mapped
                for raw_genre in raw_genres
                for label, mapped in _DEEZER_GENRES.items()
                if normalize_text(label) == raw_genre
            ),
            None,
        )
        if scene is None or counts[scene] >= per_scene:
            continue
        rank = int(track.get("rank", 0) or 0)
        tier = "popular" if rank >= 700_000 else (
            "deep_cut" if rank >= 350_000 else "niche"
        )
        seeds.append(
            {
                "scene": scene,
                "catalog_tier": tier,
                "title": str(titles[row]),
                "artist": str(artists[row]),
                "deezer_track_id": track_id,
                "scene_source_url": f"https://api.deezer.com/album/{album_id}",
                "scene_source_excerpt": (
                    f"Deezer album metadata assigns genre "
                    f"{album.get('genres', {}).get('data', [{}])[0].get('name', scene)}."
                ),
            }
        )
        counts[scene] += 1
        seen_artists.add(artist_key)
        if (
            len(counts) >= minimum_scenes
            and sum(min(value, 12) for value in counts.values()) >= 120
        ):
            break
        if len(seeds) % 10 == 0:
            _save_cache(cache_path, cache)
    _save_cache(cache_path, cache)
    return seeds


def build_benchmark(
    index_path: Path,
    v6_path: Path,
    cache_path: Path,
    output_path: Path,
    dev_count: int = 60,
    final_count: int = 60,
    minimum_positives: int = 5,
    maximum_positives: int = 12,
) -> Dict[str, Any]:
    """Build source-separated DEV and unopened FINAL multi-positive records."""
    with np.load(index_path, allow_pickle=False) as index:
        titles = np.asarray(index["titles"])
        artists = np.asarray(index["artists"])
        track_ids = np.asarray(index["track_ids"])
    resolver = PairResolver(titles, artists)
    artist_rows: Dict[str, List[int]] = {}
    for row, raw_artist in enumerate(artists):
        for artist in credited_artists(str(raw_artist)):
            artist_rows.setdefault(normalize_text(artist), []).append(row)
    quality = TitleQualityFilter()
    quality_mask = quality.keep_mask(titles, artists)
    cache = _load_cache(cache_path)
    cache.setdefault("deezer_tracks", {})
    cache.setdefault("deezer_albums", {})
    cache.setdefault("deezer_related", {})
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT.replace("/6.0", "/7.0")})

    development: List[Dict[str, Any]] = []
    development_artists: Set[str] = set()
    for seed in _v6_dev_seeds(v6_path):
        record = _build_record(
            seed,
            "development",
            len(development) + 1,
            resolver,
            session,
            cache,
            cache_path,
            titles,
            artists,
            track_ids,
            artist_rows,
            quality_mask,
            development_artists,
            minimum_positives,
            maximum_positives,
            known_mbid=seed["recording_mbid"],
        )
        if record is None:
            continue
        development.append(record)
        development_artists |= _record_artists(record)
        if len(development) >= dev_count:
            break

    blocked = _blocked_v6_artists(v6_path) | development_artists
    fresh_seeds = discover_fresh_seeds(
        titles,
        artists,
        track_ids,
        quality_mask,
        blocked,
        session,
        cache,
        cache_path,
    )
    final_candidates: List[Dict[str, Any]] = []
    final_query_artists: Set[str] = set()
    for seed in fresh_seeds:
        if _artists(seed["artist"]) & (blocked | final_query_artists):
            continue
        record = _build_record(
            seed,
            "final",
            len(final_candidates) + 1,
            resolver,
            session,
            cache,
            cache_path,
            titles,
            artists,
            track_ids,
            artist_rows,
            quality_mask,
            blocked,
            minimum_positives,
            maximum_positives,
        )
        if record is None:
            continue
        record["query_scene_source"] = {
            "url": seed["scene_source_url"],
            "publisher": "Deezer",
            "accessed_at": date.today().isoformat(),
            "source_class": "independent_catalog_genre_metadata",
            "excerpt": seed["scene_source_excerpt"],
        }
        final_candidates.append(record)
        final_query_artists |= _artists(record["query"]["artist"])
        if (
            len(final_candidates) >= final_count + 12
            and len({item["scene"] for item in final_candidates}) >= 12
        ):
            break
    final = _balanced(final_candidates, final_count)
    if len(development) < dev_count:
        raise RuntimeError(f"Only {len(development)} DEV records; need {dev_count}")
    if len(final) < final_count:
        raise RuntimeError(f"Only {len(final)} FINAL records; need {final_count}")
    scenes = Counter(record["scene"] for record in final)
    if len(scenes) < 12:
        raise RuntimeError(f"Only {len(scenes)} FINAL scenes")

    document = {
        "schema_version": 7,
        "benchmark_id": "catalog-wide-multipositive-v7",
        "benchmark_version": "7.0.0",
        "created_at": _now(),
        "frozen_at": _now(),
        "source_policy": {
            "graph_training": [
                "Last.fm-360K artist play histories",
                "Music4All-Onion Last.fm listening extraction",
            ],
            "automated_evaluation": [
                "Deezer related artists",
                "ListenBrainz session-based similar recordings",
            ],
            "same_dataset_or_api": False,
            "previous_finals": "opened diagnostics used only as DEV queries",
            "junk_samples_legal_covers_remixes_excluded": True,
            "component_artist_overlap": False,
        },
        "metric_policy": {
            "primary": "graded_nDCG@10",
            "secondary": ["MRR@10", "Recall@10"],
            "candidate_recall_at": [100, 500, 1000],
            "grade_policy": _GRADE_POLICY,
            "success": {
                "minimum_relative_primary_gain": 0.20,
                "minimum_absolute_primary_gain": 0.02,
                "paired_bootstrap_ci95_low_must_exceed": 0.0,
                "minimum_improved_seeds": 10,
                "maximum_scene_relative_regression": -0.10,
                "recall_at_10_must_not_regress": True,
                "mrr_at_10_must_not_regress": True,
                "minimum_direct_top5_passes": 16,
            },
        },
        "axis_policy": {
            "taste_affinity": (
                "graded independent ListenBrainz multi-positive retrieval"
            ),
            "sonic_similarity": (
                "separate blind preview/list coherence review; never relabelled "
                "as taste affinity"
            ),
            "ship_requires_both": True,
        },
        "counts": {
            "development": len(development),
            "final": len(final),
            "final_scenes": len(scenes),
            "minimum_positives": min(
                len(record["positives"]) for record in development + final
            ),
            "maximum_positives": max(
                len(record["positives"]) for record in development + final
            ),
        },
        "records": development + final,
    }
    document["records_sha256"] = content_sha256(document["records"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "benchmark_path": str(output_path),
        "benchmark_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "counts": document["counts"],
        "final_labels_printed": False,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--v6", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dev-count", type=int, default=60)
    parser.add_argument("--final-count", type=int, default=60)
    args = parser.parse_args(argv)
    summary = build_benchmark(
        args.index,
        args.v6,
        args.cache,
        args.output,
        dev_count=args.dev_count,
        final_count=args.final_count,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
