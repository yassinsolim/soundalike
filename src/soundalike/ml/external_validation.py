"""Independent artist-similarity validation for the guarded reranker.

ListenBrainz and Deezer data collected here is validation-only.  The winner is
an unsupervised centroid reranker and never reads these responses.  All seed
artists are disjoint from both splits of the sourced-pair benchmark.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import requests

from .real_benchmark import (
    PairResolver,
    ProductionRanker,
    credited_artists,
    held_out_artists,
    load_benchmark,
    primary_artist,
)

VALIDATION_SEEDS: List[Tuple[str, str, str]] = [
    ("Money Trees", "Kendrick Lamar", "rap"),
    ("Snooze", "SZA", "r&b"),
    ("So What", "Miles Davis", "jazz"),
    ("Alison", "Slowdive", "shoegaze"),
    ("Last Last", "Burna Boy", "afrobeats"),
    ("Bags", "Clairo", "indie"),
    ("Bangarang", "Skrillex", "electronic"),
    ("Windowlicker", "Aphex Twin", "electronic"),
    ("Holocene", "Bon Iver", "folk"),
    ("Change (In the House of Flies)", "Deftones", "metal"),
    ("Provenza", "Karol G", "latin"),
    ("Chamber of Reflection", "Mac DeMarco", "indie"),
]

LB_ALGORITHM = (
    "session_based_days_7500_session_300_contribution_5_"
    "threshold_10_limit_100_filter_True_skip_30"
)
USER_AGENT = "soundalike-public-benchmark/1.0"


def _musicbrainz_id(session: requests.Session, artist: str) -> Optional[str]:
    response = session.get(
        "https://musicbrainz.org/ws/2/artist/",
        params={"query": f'artist:"{artist}"', "fmt": "json", "limit": 1},
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    response.raise_for_status()
    rows = response.json().get("artists", [])
    return rows[0]["id"] if rows else None


def _listenbrainz(session: requests.Session, artist: str) -> List[str]:
    mbid = _musicbrainz_id(session, artist)
    time.sleep(1.05)  # MusicBrainz asks clients to stay at or below 1 request/s.
    if not mbid:
        return []
    response = session.get(
        "https://labs.api.listenbrainz.org/similar-artists/json",
        params={"artist_mbids": mbid, "algorithm": LB_ALGORITHM},
        headers={"User-Agent": USER_AGENT},
        timeout=30,
    )
    response.raise_for_status()
    return [str(row["name"]) for row in response.json() if row.get("name")]


def _deezer(session: requests.Session, artist: str) -> List[str]:
    search = session.get(
        "https://api.deezer.com/search/artist",
        params={"q": artist},
        timeout=30,
    )
    search.raise_for_status()
    rows = search.json().get("data", [])
    if not rows:
        return []
    response = session.get(
        f"https://api.deezer.com/artist/{rows[0]['id']}/related",
        timeout=30,
    )
    response.raise_for_status()
    return [str(row["name"]) for row in response.json().get("data", [])]


def _bootstrap_delta(
    baseline: Sequence[float], winner: Sequence[float], iterations: int = 10_000
) -> Dict[str, float]:
    base = np.asarray(baseline, dtype=np.float64)
    test = np.asarray(winner, dtype=np.float64)
    rng = np.random.default_rng(20260711)
    deltas = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        sample = rng.integers(0, len(base), size=len(base))
        deltas[iteration] = (test[sample] - base[sample]).mean()
    return {
        "baseline_mean": float(base.mean()),
        "winner_mean": float(test.mean()),
        "absolute_delta": float((test - base).mean()),
        "ci95_low": float(np.percentile(deltas, 2.5)),
        "ci95_high": float(np.percentile(deltas, 97.5)),
    }


def run(
    index_path: Path,
    benchmark_path: Path,
    truth_path: Optional[Path] = None,
) -> Dict[str, Any]:
    from webapp.api._reco import WebRecommender

    benchmark = load_benchmark(benchmark_path)
    all_benchmark_artists = held_out_artists(benchmark)
    for pair in benchmark["pairs"]:
        if pair["split"] == "development":
            all_benchmark_artists.update(credited_artists(pair["query"]["artist"]))
            all_benchmark_artists.update(credited_artists(pair["target"]["artist"]))
    overlap = sorted(
        {primary_artist(artist) for _, artist, _ in VALIDATION_SEEDS}
        & all_benchmark_artists
    )
    if overlap:
        raise RuntimeError(f"External validation leaks benchmark artists: {overlap}")

    session = requests.Session()
    if truth_path and Path(truth_path).exists():
        truth = json.loads(Path(truth_path).read_text(encoding="utf-8"))
    else:
        rows = []
        for title, artist, scene in VALIDATION_SEEDS:
            rows.append({
                "title": title,
                "artist": artist,
                "scene": scene,
                "listenbrainz": _listenbrainz(session, artist),
                "deezer": _deezer(session, artist),
            })
        truth = {
            "schema_version": 1,
            "retrieved_at": "2026-07-11",
            "validation_only": True,
            "sources": {
                "listenbrainz": "https://labs.api.listenbrainz.org/similar-artists/json",
                "deezer": "https://api.deezer.com/artist/{id}/related",
                "musicbrainz_resolution": "https://musicbrainz.org/ws/2/artist/",
            },
            "rows": rows,
        }
        if truth_path:
            Path(truth_path).parent.mkdir(parents=True, exist_ok=True)
            Path(truth_path).write_text(
                json.dumps(truth, indent=2, ensure_ascii=False), encoding="utf-8"
            )

    recommender = WebRecommender(str(index_path), enhance=False)
    resolver = PairResolver(recommender.titles, recommender.artists)
    ranker = ProductionRanker(recommender, held_out_artists(benchmark))
    records = []
    aggregates = {
        "listenbrainz": {"production_baseline": [], "guarded_centroid": []},
        "deezer": {"production_baseline": [], "guarded_centroid": []},
    }
    for row in truth["rows"]:
        query = {"title": row["title"], "artist": row["artist"]}
        query_row = resolver.query_row(query)
        record = {**row, "query_found": query_row is not None, "methods": {}}
        if query_row is None:
            records.append(record)
            continue
        for method in ("production_baseline", "guarded_centroid"):
            ranked = ranker.rank(query_row, method, n=15)
            artists = [primary_artist(str(recommender.artists[i])) for i in ranked]
            record["methods"][method] = {
                "artists": artists,
                "titles": [str(recommender.titles[i]) for i in ranked],
            }
            for source in ("listenbrainz", "deezer"):
                relevant = {primary_artist(name) for name in row[source]}
                if relevant:
                    overlap_score = float(np.mean([artist in relevant for artist in artists]))
                    record["methods"][method][f"{source}_overlap_at_15"] = overlap_score
                    aggregates[source][method].append(overlap_score)
        records.append(record)

    comparisons = {
        source: _bootstrap_delta(
            values["production_baseline"], values["guarded_centroid"]
        )
        for source, values in aggregates.items()
        if values["production_baseline"]
    }
    return {
        "schema_version": 1,
        "created_at": "2026-07-11",
        "index_tracks": len(recommender),
        "benchmark_artist_overlap": overlap,
        "truth_sources": truth["sources"],
        "comparisons": comparisons,
        "records": records,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--truth", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run(args.index, args.benchmark, args.truth)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(result["comparisons"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
