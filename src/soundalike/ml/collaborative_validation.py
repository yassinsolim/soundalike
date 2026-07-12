"""Independent validation and direct-list inspection for the locked hybrid."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import psutil
import requests

from .collaborative import CollaborativeIndex
from .collaborative_rerank import CollaborativeHybridRanker, LinearScorer
from .eval_suite import HELD_OUT_SEEDS
from .external_validation import _bootstrap_delta, _deezer
from .real_benchmark import (
    PairResolver,
    ProductionRanker,
    credited_artists,
    normalize_text,
    primary_artist,
)

EXTERNAL_SEEDS: List[Tuple[str, str, str]] = [
    ("Lose Yourself", "Eminem", "hip-hop"),
    ("The Light", "Common", "hip-hop"),
    ("Rock the Boat", "Aaliyah", "r&b"),
    ("Ascension", "Maxwell", "r&b"),
    ("Reptilia", "The Strokes", "indie"),
    ("My Girls", "Animal Collective", "indie"),
    ("Sometimes", "My Bloody Valentine", "shoegaze"),
    ("Sweet Trip", "Velocity : Design : Comfort", "shoegaze"),
    ("Xtal", "Aphex Twin", "electronic"),
    ("Eple", "Röyksopp", "electronic"),
    ("Paranoid Android", "Radiohead", "art-rock"),
    ("Lateralus", "Tool", "metal"),
    ("Crystal Mountain", "Death", "metal"),
    ("Take Five", "The Dave Brubeck Quartet", "jazz"),
    ("Blue Rondo à la Turk", "Dave Brubeck", "jazz"),
    ("Plastic Love", "Mariya Takeuchi", "city-pop"),
    ("Ditto", "NewJeans", "k-pop"),
    ("Tití Me Preguntó", "Bad Bunny", "latin"),
    ("Essence", "Wizkid", "afrobeats"),
    ("Drew Barrymore", "SZA", "r&b"),
    ("Hard Times", "Paramore", "pop-rock"),
    ("Your Best American Girl", "Mitski", "indie"),
    ("Oblivion", "Grimes", "electronic"),
    ("Kyoto", "Phoebe Bridgers", "folk-rock"),
    ("Heart-Shaped Box", "Nirvana", "grunge"),
    ("Freeee (Ghost Town Pt. 2)", "KIDS SEE GHOSTS", "hip-hop"),
    ("Human Nature", "Michael Jackson", "pop"),
    ("Hung Up", "Madonna", "pop"),
    ("Rock With You", "Michael Jackson", "disco"),
    ("Them Changes", "Thundercat", "funk"),
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_rankers(
    index_path: Path,
    collaborative_path: Path,
    scorer_path: Path,
) -> Tuple[Any, PairResolver, ProductionRanker, CollaborativeHybridRanker]:
    from webapp.api._reco import WebRecommender

    recommender = WebRecommender(str(index_path), enhance=False)
    resolver = PairResolver(recommender.titles, recommender.artists)
    production = ProductionRanker(recommender, heldout=set())
    scorer_document = json.loads(scorer_path.read_text(encoding="utf-8"))
    scorer = LinearScorer.from_dict(scorer_document["scorer"])
    hybrid = CollaborativeHybridRanker(
        recommender,
        CollaborativeIndex(collaborative_path, len(recommender)),
        scorer=scorer,
    )
    return recommender, resolver, production, hybrid


def _benchmark_artists(path: Path) -> set[str]:
    benchmark = json.loads(path.read_text(encoding="utf-8"))
    return {
        normalize_text(artist)
        for pair in benchmark["pairs"]
        for side in ("query", "target")
        for artist in credited_artists(pair[side]["artist"])
    }


def _artist_rows(recommender: Any, rows: Sequence[int]) -> List[str]:
    return [
        primary_artist(str(recommender.artists[int(row)])) for row in rows
    ]


def run_external(
    index_path: Path,
    collaborative_path: Path,
    scorer_path: Path,
    benchmark_path: Path,
) -> Dict[str, Any]:
    """Compare current production and locked hybrid against Deezer affinity."""
    recommender, resolver, production, hybrid = _load_rankers(
        index_path, collaborative_path, scorer_path
    )
    blocked = _benchmark_artists(benchmark_path)
    session = requests.Session()
    session.headers.update({
        "User-Agent": "soundalike-independent-validation/1.0"
    })
    records = []
    baseline_values = []
    hybrid_values = []
    for title, artist, scene in EXTERNAL_SEEDS:
        if normalize_text(artist) in blocked:
            continue
        query_row = resolver.query_row({"title": title, "artist": artist})
        if query_row is None:
            continue
        truth = _deezer(session, artist)
        relevant = {primary_artist(value) for value in truth}
        if not relevant:
            continue
        baseline_rows = production.rank(query_row, "dual_sonic", n=15)
        hybrid_rows = hybrid.rank(query_row, "hybrid", n=15)
        baseline_artists = _artist_rows(recommender, baseline_rows)
        hybrid_artists = _artist_rows(recommender, hybrid_rows)
        baseline_overlap = float(np.mean([
            value in relevant for value in baseline_artists
        ]))
        hybrid_overlap = float(np.mean([
            value in relevant for value in hybrid_artists
        ]))
        baseline_values.append(baseline_overlap)
        hybrid_values.append(hybrid_overlap)
        records.append({
            "query": {
                "title": str(recommender.titles[query_row]),
                "artist": str(recommender.artists[query_row]),
                "scene": scene,
            },
            "source": {
                "provider": "Deezer",
                "endpoint": "https://api.deezer.com/artist/{id}/related",
                "relationship": "artist-level taste affinity",
            },
            "truth_artists": truth,
            "current_production": {
                "artists": baseline_artists,
                "overlap_at_15": baseline_overlap,
            },
            "locked_hybrid": {
                "artists": hybrid_artists,
                "overlap_at_15": hybrid_overlap,
            },
        })
    comparison = _bootstrap_delta(baseline_values, hybrid_values)
    return {
        "schema_version": 1,
        "created_at": _now(),
        "source_independence": {
            "collaborative_training": "Music4All-Onion / Last.fm extraction",
            "validation_source": "Deezer related artists",
            "same_dataset_or_api": False,
            "listenbrainz_or_lastfm_claimed_as_external": False,
        },
        "metric_scope": "artist-level taste affinity, not sonic similarity",
        "benchmark_artist_overlap": [],
        "resolved_seeds": len(records),
        "comparison": comparison,
        "baseline_mean": float(np.mean(baseline_values)),
        "hybrid_mean": float(np.mean(hybrid_values)),
        "records": records,
    }


def _preview(session: requests.Session, track_id: int) -> Dict[str, Any]:
    response = session.get(
        f"https://api.deezer.com/track/{track_id}", timeout=30
    )
    if response.status_code != 200:
        return {"http_status": response.status_code, "available": False}
    payload = response.json()
    preview = payload.get("preview")
    return {
        "http_status": response.status_code,
        "available": bool(preview),
        "url": preview,
    }


def run_direct_lists(
    index_path: Path,
    collaborative_path: Path,
    scorer_path: Path,
) -> Dict[str, Any]:
    """Record actual top fives and preview availability for direct judgment."""
    recommender, resolver, production, hybrid = _load_rankers(
        index_path, collaborative_path, scorer_path
    )
    session = requests.Session()
    records = []
    for title, artist, scene in HELD_OUT_SEEDS:
        query_row = resolver.query_row({"title": title, "artist": artist})
        record: Dict[str, Any] = {
            "requested_query": {"title": title, "artist": artist, "scene": scene},
            "query_found": query_row is not None,
        }
        if query_row is None:
            records.append(record)
            continue
        record["resolved_query"] = {
            "title": str(recommender.titles[query_row]),
            "artist": str(recommender.artists[query_row]),
            "row": int(query_row),
        }
        methods = {
            "current_production": production.rank(
                query_row, "dual_sonic", n=5
            ),
            "locked_hybrid": hybrid.rank(query_row, "hybrid", n=5),
        }
        record["methods"] = {}
        for method, rows in methods.items():
            results = []
            for row in rows:
                track_id = int(recommender.track_ids[row])
                results.append({
                    "title": str(recommender.titles[row]),
                    "artist": str(recommender.artists[row]),
                    "track_id": track_id,
                    "preview": _preview(session, track_id),
                })
            record["methods"][method] = results
        records.append(record)
    return {
        "schema_version": 1,
        "created_at": _now(),
        "evaluation_role": (
            "separate difficult-seed direct inspection; not used for tuning"
        ),
        "pass_rule": (
            "at least four coherent top-five results and no unrelated scene "
            "result in positions 1-3; automatic fail for junk/seed variants"
        ),
        "preview_policy": (
            "Deezer preview URL availability and HTTP metadata were checked; "
            "no claim of audible playback is made."
        ),
        "records": records,
    }


def run_resources(
    index_path: Path,
    collaborative_path: Path,
    scorer_path: Path,
) -> Dict[str, Any]:
    """Measure cold load, RSS, and warm recommendation latency."""
    from webapp.api._reco import WebRecommender

    process = psutil.Process()
    rss_start = process.memory_info().rss
    started = time.perf_counter()
    recommender = WebRecommender(str(index_path), enhance=False)
    index_load_seconds = time.perf_counter() - started
    rss_after_index = process.memory_info().rss
    started = time.perf_counter()
    collaborative = CollaborativeIndex(collaborative_path, len(recommender))
    collaborative_load_seconds = time.perf_counter() - started
    scorer_doc = json.loads(scorer_path.read_text(encoding="utf-8"))
    scorer = LinearScorer.from_dict(scorer_doc["scorer"])
    ranker = CollaborativeHybridRanker(
        recommender, collaborative, scorer=scorer
    )
    rss_after_hybrid = process.memory_info().rss
    resolver = PairResolver(recommender.titles, recommender.artists)
    rows = []
    for title, artist, _ in HELD_OUT_SEEDS:
        row = resolver.query_row({"title": title, "artist": artist})
        if row is not None and row not in rows:
            rows.append(row)
    started = time.perf_counter()
    ranker.rank(rows[0], "hybrid", n=20)
    cold_recommend_seconds = time.perf_counter() - started
    latencies = []
    modes = []
    for row in rows:
        started = time.perf_counter()
        context = ranker.context(row)
        ranker.rank_context(context, "hybrid", n=20)
        latencies.append(time.perf_counter() - started)
        modes.append(context.query_mode)
    return {
        "schema_version": 1,
        "created_at": _now(),
        "runtime_assets": {
            "production_index_bytes": index_path.stat().st_size,
            "collaborative_index_bytes": collaborative_path.stat().st_size,
            "scorer_bytes": scorer_path.stat().st_size,
            "added_runtime_bytes": (
                collaborative_path.stat().st_size + scorer_path.stat().st_size
            ),
        },
        "research_only_unmasked_asset_ships": False,
        "load_seconds": {
            "production_index": index_load_seconds,
            "collaborative_index": collaborative_load_seconds,
            "collaborative_internal": collaborative.load_seconds,
        },
        "rss_bytes": {
            "start": rss_start,
            "after_production_index": rss_after_index,
            "after_hybrid": rss_after_hybrid,
            "hybrid_delta": rss_after_hybrid - rss_after_index,
        },
        "latency_seconds": {
            "first_recommendation": cold_recommend_seconds,
            "warm_mean": float(np.mean(latencies)),
            "warm_p50": float(np.percentile(latencies, 50)),
            "warm_p95": float(np.percentile(latencies, 95)),
            "samples": len(latencies),
        },
        "query_modes": {
            str(mode): int(count)
            for mode, count in zip(*np.unique(modes, return_counts=True))
        },
        "deployment_assessment": (
            "The compact collaborative asset is suitable for serverless shipping; "
            "deployment remains prohibited because the once-opened FINAL failed."
        ),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("external", "direct", "resource")
    )
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--collaborative", type=Path, required=True)
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "external":
        if args.benchmark is None:
            parser.error("--benchmark is required for external validation")
        result = run_external(
            args.index, args.collaborative, args.scorer, args.benchmark
        )
    elif args.command == "direct":
        result = run_direct_lists(
            args.index, args.collaborative, args.scorer
        )
    else:
        result = run_resources(
            args.index, args.collaborative, args.scorer
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        key: value for key, value in result.items() if key != "records"
    }, indent=2))


if __name__ == "__main__":
    main()
