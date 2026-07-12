"""Post-lock direct, external, and resource validation for protocol v7."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Set, Tuple

import numpy as np
import psutil
import requests

from .catalog_graph import CatalogArtistGraph
from .catalog_rerank import CatalogHybridRanker, HybridScorer
from .collaborative import CollaborativeIndex
from .external_validation import _bootstrap_delta
from .real_benchmark import (
    PairResolver,
    ProductionRanker,
    credited_artists,
    normalize_text,
    primary_artist,
)

DIRECT_SEEDS: List[Tuple[str, str, str]] = [
    ("Redbone", "Childish Gambino", "alternative-r&b"),
    ("Nights", "Frank Ocean", "alternative-r&b"),
    ("Chanel", "Frank Ocean", "alternative-r&b"),
    ("Exit Music (For a Film)", "Radiohead", "art-rock"),
    ("Kid A", "Radiohead", "idm-art-rock"),
    ("Alison", "Slowdive", "shoegaze"),
    ("Souvlaki Space Station", "Slowdive", "shoegaze"),
    ("money machine", "100 gecs", "hyperpop"),
    ("venus fly trap", "brakence", "digicore"),
    ("Remember Summer Days", "Anri", "city-pop"),
    ("Enter Sandman", "Metallica", "metal"),
    ("Fade to Black", "Metallica", "metal-ballad"),
    ("Round Midnight", "Thelonious Monk", "jazz"),
    ("So What", "Miles Davis", "modal-jazz"),
    ("Where Is My Mind?", "Pixies", "indie-rock"),
    ("Heroin", "The Velvet Underground", "art-rock"),
    ("Starboy", "The Weeknd", "r&b-pop"),
    ("Save Your Tears", "The Weeknd", "synth-pop"),
    ("Chemical", "Post Malone", "guitar-pop"),
    ("Golden Hour", "JVKE", "piano-pop"),
]

EXTERNAL_SEEDS: List[Tuple[str, str, str]] = [
    ("Lose Yourself", "Eminem", "hip-hop"),
    ("Rock the Boat", "Aaliyah", "r&b"),
    ("Reptilia", "The Strokes", "indie"),
    ("Sometimes", "My Bloody Valentine", "shoegaze"),
    ("Xtal", "Aphex Twin", "electronic"),
    ("Lateralus", "Tool", "metal"),
    ("Take Five", "The Dave Brubeck Quartet", "jazz"),
    ("Plastic Love", "Mariya Takeuchi", "city-pop"),
    ("Ditto", "NewJeans", "k-pop"),
    ("Tití Me Preguntó", "Bad Bunny", "latin"),
    ("Essence", "Wizkid", "afrobeats"),
    ("Drew Barrymore", "SZA", "r&b"),
    ("Hard Times", "Paramore", "pop-rock"),
    ("Oblivion", "Grimes", "electronic"),
    ("Kyoto", "Phoebe Bridgers", "folk-rock"),
    ("Heart-Shaped Box", "Nirvana", "grunge"),
    ("Human Nature", "Michael Jackson", "pop"),
    ("Hung Up", "Madonna", "pop"),
    ("Them Changes", "Thundercat", "funk"),
    ("Crystal Mountain", "Death", "metal"),
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_rankers(
    index_path: Path,
    sparse_path: Path,
    catalog_path: Path,
    scorer_path: Path,
) -> Tuple[Any, PairResolver, ProductionRanker, CatalogHybridRanker]:
    from webapp.api._reco import WebRecommender

    recommender = WebRecommender(str(index_path), enhance=False)
    resolver = PairResolver(recommender.titles, recommender.artists)
    production = ProductionRanker(recommender, heldout=set())
    scorer = HybridScorer.from_dict(
        json.loads(scorer_path.read_text(encoding="utf-8"))["scorer"]
    )
    hybrid = CatalogHybridRanker(
        recommender,
        CollaborativeIndex(sparse_path, len(recommender)),
        CatalogArtistGraph(catalog_path),
        scorer=scorer,
    )
    return recommender, resolver, production, hybrid


def _preview(session: requests.Session, track_id: int) -> Dict[str, Any]:
    response = session.get(
        f"https://api.deezer.com/track/{track_id}", timeout=30
    )
    if response.status_code != 200:
        return {"http_status": response.status_code, "available": False}
    preview = response.json().get("preview")
    return {
        "http_status": response.status_code,
        "available": bool(preview),
        "url": preview,
    }


def run_direct_lists(
    index_path: Path,
    sparse_path: Path,
    catalog_path: Path,
    scorer_path: Path,
) -> Dict[str, Any]:
    """Record locked top fives and preview availability for 20 hard seeds."""
    rec, resolver, production, hybrid = _load_rankers(
        index_path, sparse_path, catalog_path, scorer_path
    )
    session = requests.Session()
    records = []
    mode_counts: Dict[str, int] = {}
    for title, artist, scene in DIRECT_SEEDS:
        query_row = resolver.query_row({"title": title, "artist": artist})
        record: Dict[str, Any] = {
            "requested_query": {"title": title, "artist": artist, "scene": scene},
            "query_found": query_row is not None,
        }
        if query_row is None:
            records.append(record)
            continue
        context = hybrid.context(query_row, variant="twohop")
        mode_counts[context.query_mode] = mode_counts.get(context.query_mode, 0) + 1
        record["resolved_query"] = {
            "title": str(rec.titles[query_row]),
            "artist": str(rec.artists[query_row]),
            "row": int(query_row),
        }
        methods = {
            "current_production": production.rank(query_row, "dual_sonic", n=5),
            "locked_catalog_hybrid": hybrid.rank_context(
                context, "hybrid", n=5
            ),
        }
        record["query_mode"] = context.query_mode
        record["methods"] = {}
        for method, rows in methods.items():
            results = []
            for row in rows:
                track_id = int(rec.track_ids[int(row)])
                results.append(
                    {
                        "title": str(rec.titles[int(row)]),
                        "artist": str(rec.artists[int(row)]),
                        "track_id": track_id,
                        "preview": _preview(session, track_id),
                    }
                )
            record["methods"][method] = results
        records.append(record)
    return {
        "schema_version": 1,
        "created_at": _now(),
        "method_locked_before_judgment": True,
        "used_for_tuning": False,
        "axis": "sonic_similarity_and_scene_coherence",
        "pass_rule": (
            "at least four coherent top-five results and no unrelated-scene "
            "result in positions 1-3; junk and seed-title variants fail"
        ),
        "preview_policy": (
            "Deezer preview availability is HTTP-verified. Metadata and preview "
            "inspection are separate from taste-affinity FINAL labels."
        ),
        "query_modes": mode_counts,
        "records": records,
    }


def _benchmark_artists(path: Path) -> Set[str]:
    benchmark = json.loads(path.read_text(encoding="utf-8"))
    return {
        normalize_text(artist)
        for record in benchmark["records"]
        for song in (record["query"], *record["positives"])
        for artist in credited_artists(song["artist"])
    }


def _musicbrainz_tags(
    session: requests.Session,
    artist: str,
    cache: Dict[str, List[str]],
) -> Set[str]:
    key = normalize_text(artist)
    if key not in cache:
        response = session.get(
            "https://musicbrainz.org/ws/2/artist/",
            params={
                "query": f'artist:"{artist}"',
                "fmt": "json",
                "limit": 1,
            },
            timeout=30,
        )
        response.raise_for_status()
        values = response.json().get("artists", [])
        tags = values[0].get("tags", []) if values else []
        cache[key] = [
            normalize_text(tag["name"])
            for tag in sorted(
                tags, key=lambda value: -int(value.get("count", 0))
            )
            if int(tag.get("count", 0)) > 0
        ][:12]
        time.sleep(1.05)
    return set(cache[key])


def _tag_score(
    query_tags: Set[str],
    result_artists: Sequence[str],
    session: requests.Session,
    cache: Dict[str, List[str]],
) -> float:
    values = []
    for artist in result_artists:
        result_tags = _musicbrainz_tags(session, artist, cache)
        union = query_tags | result_tags
        values.append(len(query_tags & result_tags) / len(union) if union else 0.0)
    return float(np.mean(values)) if values else 0.0


def run_external_tags(
    index_path: Path,
    sparse_path: Path,
    catalog_path: Path,
    scorer_path: Path,
    benchmark_path: Path,
    cache_path: Path,
) -> Dict[str, Any]:
    """Compare methods against independent MusicBrainz community tags."""
    rec, resolver, production, hybrid = _load_rankers(
        index_path, sparse_path, catalog_path, scorer_path
    )
    blocked = _benchmark_artists(benchmark_path)
    benchmark_overlap = sorted(
        {
            normalize_text(artist)
            for _, artist, _ in EXTERNAL_SEEDS
            if normalize_text(artist) in blocked
        }
    )
    cache = (
        json.loads(cache_path.read_text(encoding="utf-8"))
        if cache_path.exists()
        else {}
    )
    session = requests.Session()
    session.headers.update(
        {"User-Agent": "soundalike-external-validation/1.0 (contact: github.com)"}
    )
    baseline_values = []
    hybrid_values = []
    records = []
    for title, artist, scene in EXTERNAL_SEEDS:
        if normalize_text(artist) in blocked:
            continue
        query_row = resolver.query_row({"title": title, "artist": artist})
        if query_row is None:
            continue
        query_tags = _musicbrainz_tags(session, artist, cache)
        if len(query_tags) < 2:
            continue
        context = hybrid.context(query_row, variant="twohop")
        baseline_rows = production.rank(query_row, "dual_sonic", n=10)
        hybrid_rows = hybrid.rank_context(context, "hybrid", n=10)
        baseline_artists = [
            str(rec.artists[int(row)]) for row in baseline_rows
        ]
        hybrid_artists = [
            str(rec.artists[int(row)]) for row in hybrid_rows
        ]
        baseline = _tag_score(
            query_tags, baseline_artists, session, cache
        )
        winner = _tag_score(query_tags, hybrid_artists, session, cache)
        baseline_values.append(baseline)
        hybrid_values.append(winner)
        records.append(
            {
                "query": {
                    "title": str(rec.titles[query_row]),
                    "artist": str(rec.artists[query_row]),
                    "scene": scene,
                },
                "query_tags": sorted(query_tags),
                "production_tag_jaccard_at_10": baseline,
                "hybrid_tag_jaccard_at_10": winner,
            }
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if len(records) >= 12:
            break
    comparison = _bootstrap_delta(baseline_values, hybrid_values)
    return {
        "schema_version": 1,
        "created_at": _now(),
        "source": {
            "provider": "MusicBrainz community artist tags",
            "endpoint": "https://musicbrainz.org/ws/2/artist/",
            "relationship": "independent scene/style metadata",
            "used_by_graph_or_benchmark": False,
        },
        "metric_scope": "scene/style consistency, not co-listening",
        "benchmark_artist_overlap": benchmark_overlap,
        "resolved_seeds": len(records),
        "baseline_mean": float(np.mean(baseline_values)),
        "hybrid_mean": float(np.mean(hybrid_values)),
        "comparison": comparison,
        "records": records,
    }


def run_resources(
    index_path: Path,
    sparse_path: Path,
    catalog_path: Path,
    scorer_path: Path,
) -> Dict[str, Any]:
    from webapp.api._reco import WebRecommender

    process = psutil.Process()
    rss_start = process.memory_info().rss
    started = time.perf_counter()
    rec = WebRecommender(str(index_path), enhance=False)
    index_load_seconds = time.perf_counter() - started
    rss_production = process.memory_info().rss
    resolver = PairResolver(rec.titles, rec.artists)
    scorer = HybridScorer.from_dict(
        json.loads(scorer_path.read_text(encoding="utf-8"))["scorer"]
    )
    hybrid_started = time.perf_counter()
    hybrid = CatalogHybridRanker(
        rec,
        CollaborativeIndex(sparse_path, len(rec)),
        CatalogArtistGraph(catalog_path),
        scorer=scorer,
    )
    hybrid_load_seconds = time.perf_counter() - hybrid_started
    rss_loaded = process.memory_info().rss
    total_load_seconds = time.perf_counter() - started
    rows = []
    for title, artist, _ in DIRECT_SEEDS:
        row = resolver.query_row({"title": title, "artist": artist})
        if row is not None:
            rows.append(row)
    cold_started = time.perf_counter()
    first = hybrid.context(rows[0])
    hybrid.rank_context(first, "hybrid", n=10)
    first_seconds = time.perf_counter() - cold_started
    timings = []
    modes: Dict[str, int] = {}
    for row in rows[:16]:
        started = time.perf_counter()
        context = hybrid.context(row)
        hybrid.rank_context(context, "hybrid", n=10)
        timings.append(time.perf_counter() - started)
        modes[context.query_mode] = modes.get(context.query_mode, 0) + 1
    added_bytes = (
        sparse_path.stat().st_size
        + catalog_path.stat().st_size
        + scorer_path.stat().st_size
    )
    return {
        "schema_version": 1,
        "created_at": _now(),
        "runtime_assets": {
            "sparse_graph_bytes": sparse_path.stat().st_size,
            "catalog_graph_bytes": catalog_path.stat().st_size,
            "scorer_bytes": scorer_path.stat().st_size,
            "added_runtime_bytes": added_bytes,
        },
        "load_seconds": {
            "production_index": index_load_seconds,
            "hybrid_assets": hybrid_load_seconds,
            "total": total_load_seconds,
        },
        "rss_bytes": {
            "start": rss_start,
            "after_production_index": rss_production,
            "after_hybrid": rss_loaded,
            "hybrid_delta": rss_loaded - rss_production,
        },
        "latency_seconds": {
            "first_recommendation": first_seconds,
            "warm_mean": float(np.mean(timings)),
            "warm_p50": float(np.percentile(timings, 50)),
            "warm_p95": float(np.percentile(timings, 95)),
            "samples": len(timings),
        },
        "query_modes": modes,
        "deployment_assessment": (
            "Measured against the existing serverless path. Shipping remains "
            "forbidden unless both fresh FINAL and direct coherence pass."
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("direct", "external", "resources"))
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--sparse", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "direct":
        result = run_direct_lists(
            args.index, args.sparse, args.catalog, args.scorer
        )
    elif args.command == "external":
        if args.benchmark is None or args.cache is None:
            parser.error("external requires --benchmark and --cache")
        result = run_external_tags(
            args.index,
            args.sparse,
            args.catalog,
            args.scorer,
            args.benchmark,
            args.cache,
        )
    else:
        result = run_resources(
            args.index, args.sparse, args.catalog, args.scorer
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "command": args.command,
        "records": len(result.get("records", [])),
        "resolved_seeds": result.get("resolved_seeds"),
        "runtime_assets": result.get("runtime_assets"),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
