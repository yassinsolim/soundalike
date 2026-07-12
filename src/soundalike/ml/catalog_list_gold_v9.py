"""Build the powered, independently sourced sonic list gold for iteration 9."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence
from urllib.parse import quote_plus

import numpy as np

from .quality_filter import TitleQualityFilter
from .real_benchmark import PairResolver, normalize_text


SCHEMA_VERSION = 9
ACCESS_DATE = "2026-07-12"
MUSIC_MAP_CLAIM = (
    "People who like {artist} might also like these artists. "
    "The closer two names are, the greater the probability people will like both artists."
)
REQUIRED_SCENES = (
    "rap",
    "rnb",
    "indie",
    "shoegaze",
    "hyperpop",
    "electronic",
    "metal",
    "jazz",
    "city_pop_jpop_kpop",
    "latin_afrobeats",
    "pop",
    "rock",
    "difficult_blend",
)

# Forty already-opened category-A editorial comparisons. The remaining twenty
# seeds are the locked difficult-list cases from iteration 8.
V6_PAIR_IDS = (
    "DEV-OPENED-DEV-L007", "DEV-OPENED-DEV-L002",
    "DEV-OPENED-DEV-L003", "DEV-OPENED-DEV-L004",
    "DEV-OPENED-DEV-L005", "DEV-OPENED-DEV-L006",
    "DEV-OPENED-DEV-L009", "DEV-OPENED-DEV-L010",
    "DEV-OPENED-DEV-L011", "DEV-OPENED-DEV-L012",
    "DEV-OPENED-DEV-L013", "DEV-OPENED-DEV-L014",
    "DEV-OPENED-DEV-L016", "DEV-OPENED-DEV-L017",
    "DEV-OPENED-DEV-L019", "DEV-OPENED-DEV-L020",
    "DEV-OPENED-DEV-L021", "DEV-OPENED-DEV-L023",
    "DEV-OPENED-DEV-L026", "DEV-OPENED-DEV-L027",
    "DEV-OPENED-DEV-L029", "DEV-OPENED-DEV-L030",
    "DEV-OPENED-DEV-L031", "DEV-OPENED-DEV-L032",
    "DEV-OPENED-DEV-L034", "DEV-OPENED-DEV-L037",
    "DEV-OPENED-DEV-L038", "DEV-OPENED-DEV-L041",
    "DEV-OPENED-DEV-L045", "DEV-OPENED-DEV-L048",
    "DEV-OPENED-DEV-L051", "DEV-OPENED-DEV-P029",
    "DEV-OPENED-DEV-P053", "DEV-OPENED-DEV-P055",
    "DEV-OPENED-DEV-P063", "DEV-OPENED-DEV-P140",
    "DEV-OPENED-DEV-P160", "DEV-OPENED-DEV-P165",
    "DEV-OPENED-DEV-P175", "DEV-OPENED-DEV-P180",
)

SCENE_BY_ARTIST = {
    "lady gaga": "pop", "katy perry": "pop", "nirvana": "rock", "the beatles": "pop",
    "iggy pop": "rock", "green day": "rock", "the killers": "indie",
    "the strokes": "indie", "avenged sevenfold": "metal",
    "mariya takeuchi": "city_pop_jpop_kpop", "beyonce": "rnb",
    "the weeknd": "rnb", "arctic monkeys": "rock",
    "michael jackson": "pop", "twice": "city_pop_jpop_kpop",
    "fontaines d c": "indie", "my chemical romance": "rock",
    "m83": "electronic", "muse": "rock", "chris stapleton": "pop",
    "editors": "indie", "maroon 5": "pop", "rag n bone man": "rnb",
    "rihanna": "rnb", "taio cruz": "electronic",
    "red velvet": "city_pop_jpop_kpop", "vampire weekend": "indie",
    "ciara": "rnb", "rina sawayama": "hyperpop", "the cure": "indie",
    "mgmt": "electronic", "kanye west": "rap",
    "chicago underground quartet": "jazz", "dizzy gillespie": "jazz",
    "partynextdoor": "rnb",
    "the world is a beautiful place and i am no longer afraid to die": "indie",
    "jhene aiko": "rnb", "king gizzard and the lizard wizard": "difficult_blend",
    "nothing": "shoegaze", "shygirl": "hyperpop", "taylor swift": "pop",
    "theon cross": "jazz", "pixies": "indie", "anri": "city_pop_jpop_kpop",
    "miki matsubara": "city_pop_jpop_kpop", "kali uchis": "latin_afrobeats",
    "bad bunny": "latin_afrobeats", "100 gecs": "hyperpop",
    "brakence": "hyperpop", "glaive": "hyperpop", "daft punk": "electronic",
    "gorillaz": "difficult_blend", "massive attack": "electronic",
    "my bloody valentine": "shoegaze", "deftones": "metal",
    "a tribe called quest": "rap", "frank ocean": "rnb",
    "metallica": "metal", "miles davis": "jazz",
    "burna boy": "latin_afrobeats", "newjeans": "city_pop_jpop_kpop",
    "fka twigs": "difficult_blend",
}

SOURCE_SLUGS = {
    "beyonce": "beyonc%C3%A9+knowles",
    "fontaines d c": "fontaines+d.c.",
    "rag n bone man": "rag%27n%27bone+man",
    "the world is a beautiful place i am no longer afraid to die":
        "the+world+is+a+beautiful+place+and+i+am+no+longer+afraid+to+die",
    "king gizzard the lizard wizard": "king+gizzard+",
}


class GoldBuildError(RuntimeError):
    """Raised when evidence or catalogue eligibility is insufficient."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Any) -> str:
    return sha256_bytes(Path(path).read_bytes())


def write_json(path: Any, value: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def artist_key(value: str) -> str:
    text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    text = re.sub(r"\b(feat|featuring|ft)\b.*$", "", text, flags=re.I)
    text = text.replace("&", " and ")
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def artist_keys(value: str) -> Sequence[str]:
    key = artist_key(value)
    keys = {key, key.removeprefix("the ")}
    keys.add(key.replace(" and ", " "))
    return tuple(sorted(item for item in keys if item))


def music_map_url(artist: str) -> str:
    key = artist_key(artist)
    slug = SOURCE_SLUGS.get(key, quote_plus(key))
    return "https://www.music-map.com/" + slug


def parse_music_map_html(raw: bytes, *, url: str, accessed_at: str) -> Dict[str, Any]:
    text = raw.decode("utf-8", errors="replace")
    title_match = re.search(
        r"<span id=the_title class=the_title>(.*?)</span>", text, re.S
    )
    block_match = re.search(r"<div id=gnodMap>(.*?)</div>", text, re.S)
    affinity_match = re.search(r"Aid\[0\]=new Array\((.*?)\);", text, re.S)
    if not title_match or not block_match or not affinity_match:
        raise GoldBuildError(f"Music-Map page did not contain a map: {url}")
    page_artist = html.unescape(re.sub("<.*?>", "", title_match.group(1))).strip()
    names = [
        html.unescape(re.sub("<.*?>", "", value)).strip()
        for value in re.findall(
            r"<a[^>]+class=S[^>]*>(.*?)</a>", block_match.group(1), re.S
        )
    ]
    scores = [
        float(value) for value in affinity_match.group(1).split(",")
        if value.strip() and float(value) >= 0.0
    ]
    neighbors = [
        {"artist": name, "source_rank": rank, "affinity": score}
        for rank, (name, score) in enumerate(zip(names[1:], scores), start=1)
    ]
    if len(neighbors) < 12:
        raise GoldBuildError(f"Music-Map returned fewer than 12 neighbors: {url}")
    normalized = {
        "source_url": url,
        "accessed_at": accessed_at,
        "source_class": "gnod_music_map_crowd_similarity",
        "page_artist": page_artist,
        "retrieved_claim": MUSIC_MAP_CLAIM.format(artist=page_artist),
        "ranking_semantics": (
            "The page orders artists by Gnod crowd proximity; lower source_rank "
            "is treated as stronger artist-level similarity."
        ),
        "neighbors": neighbors,
    }
    return {
        **normalized,
        "http_status": 200,
        "raw_response_sha256": sha256_bytes(raw),
        "normalized_snapshot_sha256": sha256_bytes(canonical_bytes(normalized)),
    }


def fetch_music_map_snapshots(
    seeds: Sequence[Mapping[str, Any]],
    *,
    accessed_at: str = ACCESS_DATE,
    session: Any = None,
) -> Dict[str, Any]:
    if session is None:
        import requests

        session = requests.Session()
        session.headers["User-Agent"] = (
            "soundalike-evaluation/9.0 (+https://github.com/yassinsolim/soundalike)"
        )
    snapshots: List[Dict[str, Any]] = []
    for seed in seeds:
        url = music_map_url(str(seed["query"]["artist"]))
        response = session.get(url, timeout=30)
        if int(response.status_code) != 200:
            raise GoldBuildError(f"Music-Map HTTP {response.status_code}: {url}")
        snapshot = parse_music_map_html(
            bytes(response.content), url=url, accessed_at=accessed_at
        )
        snapshot["seed_id"] = str(seed["id"])
        snapshots.append(snapshot)
    document = {
        "schema_version": SCHEMA_VERSION,
        "snapshot_type": "normalized_source_snapshot",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "accessed_at": accessed_at,
        "source": "Gnod Music-Map",
        "source_independence": {
            "independent_of_candidate_graph": True,
            "candidate_graph_sources": ["Last.fm-360K", "Music4All-Onion"],
            "not_from_deezer_supporting_validation": True,
        },
        "records": snapshots,
    }
    document["content_sha256"] = sha256_bytes(canonical_bytes(document))
    return document



def validate_music_map_snapshots(
    seeds: Sequence[Mapping[str, Any]], snapshots: Mapping[str, Any]
) -> Dict[str, Mapping[str, Any]]:
    """Authenticate normalized snapshots and bind each one to its seed page."""
    records = list(snapshots.get("records", ()))
    ids = [str(item.get("seed_id", "")) for item in records]
    if len(ids) != len(set(ids)):
        raise GoldBuildError("duplicate Music-Map snapshot seed IDs")
    unsigned_document = dict(snapshots)
    declared_content = str(unsigned_document.pop("content_sha256", ""))
    if sha256_bytes(canonical_bytes(unsigned_document)) != declared_content:
        raise GoldBuildError("Music-Map snapshot document hash mismatch")
    by_id = {str(item["seed_id"]): item for item in records}
    if set(by_id) != {str(seed["id"]) for seed in seeds}:
        raise GoldBuildError("snapshot seed IDs do not match the powered manifest")
    for seed in seeds:
        seed_id = str(seed["id"])
        snapshot = by_id[seed_id]
        expected_url = music_map_url(str(seed["query"]["artist"]))
        if str(snapshot.get("source_url")) != expected_url:
            raise GoldBuildError(f"snapshot URL does not match {seed_id}")
        if snapshot.get("source_class") != "gnod_music_map_crowd_similarity":
            raise GoldBuildError(f"snapshot source class does not match {seed_id}")
        if int(snapshot.get("http_status", 0)) != 200:
            raise GoldBuildError(f"snapshot HTTP status does not match {seed_id}")
        neighbors = list(snapshot.get("neighbors", ()))
        if [int(item.get("source_rank", 0)) for item in neighbors] != list(
            range(1, len(neighbors) + 1)
        ):
            raise GoldBuildError(f"snapshot ranks are not contiguous for {seed_id}")
        normalized = {
            "source_url": snapshot["source_url"],
            "accessed_at": snapshot["accessed_at"],
            "source_class": snapshot["source_class"],
            "page_artist": snapshot["page_artist"],
            "retrieved_claim": snapshot["retrieved_claim"],
            "ranking_semantics": snapshot["ranking_semantics"],
            "neighbors": neighbors,
        }
        if sha256_bytes(canonical_bytes(normalized)) != str(
            snapshot.get("normalized_snapshot_sha256", "")
        ):
            raise GoldBuildError(f"normalized snapshot hash mismatch for {seed_id}")
        if not re.fullmatch(r"[0-9a-f]{64}", str(
            snapshot.get("raw_response_sha256", "")
        )):
            raise GoldBuildError(f"raw snapshot hash is invalid for {seed_id}")
    return by_id


def load_seed_specs(v6: Any, direct: Any) -> List[Dict[str, Any]]:
    v6_doc = json.loads(Path(v6).read_text(encoding="utf-8"))
    direct_doc = json.loads(Path(direct).read_text(encoding="utf-8"))
    pairs = {str(item["id"]): item for item in v6_doc["pairs"]}
    seeds: List[Dict[str, Any]] = []
    for position, pair_id in enumerate(V6_PAIR_IDS, start=1):
        pair = pairs.get(pair_id)
        if pair is None:
            raise GoldBuildError(f"missing category-A pair {pair_id}")
        pair = dict(pair)
        if pair_id == "DEV-OPENED-DEV-P029":
            pair["query"], pair["target"] = dict(pair["target"]), dict(pair["query"])
            pair["direction_reversed_for_seed_coverage"] = True
        query = dict(pair["query"])
        seeds.append({
            "id": f"DEV-SONIC-{position:03d}",
            "query": query,
            "scene": SCENE_BY_ARTIST[artist_key(str(query["artist"]))],
            "source_scene": str(pair["scene"]),
            "category_a_pair": pair,
            "known_failure_class": None,
        })
    direct_records = direct_doc.get("seeds", direct_doc.get("records", []))
    if len(direct_records) != 20:
        raise GoldBuildError("locked difficult manifest must contain exactly 20 seeds")
    for offset, seed in enumerate(direct_records, start=len(seeds) + 1):
        query = {"title": str(seed["title"]), "artist": str(seed["artist"])}
        seeds.append({
            "id": f"DEV-SONIC-{offset:03d}",
            "query": query,
            "scene": SCENE_BY_ARTIST[artist_key(query["artist"])],
            "source_scene": str(seed["scene"]),
            "category_a_pair": None,
            "known_failure_class": str(seed["failure_class"]),
        })
    if len(seeds) != 60 or len({item["id"] for item in seeds}) != 60:
        raise GoldBuildError("powered DEV seed manifest must contain 60 unique seeds")
    missing = set(REQUIRED_SCENES) - {str(item["scene"]) for item in seeds}
    if missing:
        raise GoldBuildError(f"required scenes are missing: {sorted(missing)}")
    return seeds


def _catalog_artist_lookup(artists: Sequence[str]) -> Dict[str, List[int]]:
    lookup: MutableMapping[str, List[int]] = defaultdict(list)
    for row, artist in enumerate(artists):
        for key in artist_keys(str(artist)):
            lookup[key].append(int(row))
    return dict(lookup)


def _matched_artist_rows(
    source_artist: str, lookup: Mapping[str, Sequence[int]]
) -> List[int]:
    rows: List[int] = []
    for key in artist_keys(source_artist):
        rows.extend(map(int, lookup.get(key, ())))
    return list(dict.fromkeys(rows))


def _eligible_track_count(
    rows: Iterable[int],
    titles: Sequence[str],
    artists: Sequence[str],
    quality: TitleQualityFilter,
) -> int:
    return sum(
        not quality.is_junk(str(titles[row]), str(artists[row])) for row in rows
    )


def build_gold(
    seeds: Sequence[Mapping[str, Any]],
    snapshots: Mapping[str, Any],
    index_path: Any,
    *,
    source_inputs: Mapping[str, str],
) -> Dict[str, Any]:
    with np.load(index_path, allow_pickle=False) as data:
        titles = np.asarray(data["titles"])
        artists = np.asarray(data["artists"])
        track_ids = np.asarray(data["track_ids"])
    resolver = PairResolver(titles, artists)
    artist_lookup = _catalog_artist_lookup(artists)
    quality = TitleQualityFilter()
    evidence = validate_music_map_snapshots(seeds, snapshots)
    records: List[Dict[str, Any]] = []
    source_agreements = 0
    for seed in seeds:
        seed_id = str(seed["id"])
        query = dict(seed["query"])
        query_row = resolver.query_row(query)
        if query_row is None:
            raise GoldBuildError(f"unresolved served query: {seed_id} {query}")
        snapshot = evidence.get(seed_id)
        if snapshot is None:
            raise GoldBuildError(f"missing source snapshot for {seed_id}")
        positive_by_rule: Dict[tuple[str, str], Dict[str, Any]] = {}
        pair = seed.get("category_a_pair")
        if pair:
            target = dict(pair["target"])
            target_rows = resolver.target_rows(target)
            for row in target_rows:
                key = ("track", str(track_ids[row]))
                positive_by_rule[key] = {
                    "relevance_scope": "track",
                    "title": str(titles[row]),
                    "artist": str(artists[row]),
                    "track_id": int(track_ids[row]),
                    "grade": 3,
                    "rationale": "independent category-A track-level sonic comparison",
                    "source_refs": [
                        {
                            "url": str(source["url"]),
                            "publisher": str(source["publisher"]),
                            "accessed_at": str(source["accessed_at"]),
                            "source_class": str(source["source_class"]),
                            "retrieved_evidence": str(source["excerpt"]),
                        }
                        for source in pair["sources"]
                    ],
                    "uncertainty": "low",
                }
        for neighbor in snapshot["neighbors"][:24]:
            source_rank = int(neighbor["source_rank"])
            grade = 2 if source_rank <= 6 else 1 if source_rank <= 16 else 0
            if grade == 0:
                continue
            rows = _matched_artist_rows(str(neighbor["artist"]), artist_lookup)
            if not rows or artist_key(str(neighbor["artist"])) in artist_keys(
                str(query["artist"])
            ):
                continue
            count = _eligible_track_count(rows, titles, artists, quality)
            if count == 0:
                continue
            canonical_artist = Counter(str(artists[row]) for row in rows).most_common(1)[0][0]
            key = ("artist", artist_key(canonical_artist))
            positive_by_rule[key] = {
                "relevance_scope": "artist",
                "artist": canonical_artist,
                "grade": grade,
                "source_rank": source_rank,
                "source_affinity": float(neighbor["affinity"]),
                "eligible_catalog_tracks": count,
                "rationale": (
                    "Music-Map explicitly places this artist on the seed's "
                    "similar-artists map; any eligible served track inherits "
                    "artist-level relevance, not track-specific endorsement."
                ),
                "source_refs": [{
                    "url": str(snapshot["source_url"]),
                    "publisher": "Gnod Music-Map",
                    "accessed_at": str(snapshot["accessed_at"]),
                    "source_class": "gnod_music_map_crowd_similarity",
                    "retrieved_evidence": str(snapshot["retrieved_claim"]),
                    "normalized_snapshot_sha256":
                        str(snapshot["normalized_snapshot_sha256"]),
                }],
                "uncertainty": "medium",
            }
        positives = sorted(
            positive_by_rule.values(),
            key=lambda item: (
                -int(item["grade"]),
                int(item.get("source_rank", 0)),
                str(item.get("artist", "")),
            ),
        )
        if len(positives) < 5 or len({item["grade"] for item in positives}) < 2:
            raise GoldBuildError(
                f"{seed_id} has insufficient powered gold: {len(positives)} positives"
            )
        if pair:
            target_artist = artist_key(str(pair["target"]["artist"]))
            if any(
                item["relevance_scope"] == "artist"
                and artist_key(str(item["artist"])) == target_artist
                for item in positives
            ):
                source_agreements += 1
        records.append({
            "id": seed_id,
            "split": "development",
            "scene": str(seed["scene"]),
            "source_scene": str(seed["source_scene"]),
            "known_failure_class": seed.get("known_failure_class"),
            "query": {
                **query,
                "catalog_row": int(query_row),
                "track_id": int(track_ids[query_row]),
            },
            "positives": positives,
            "positive_count": len(positives),
            "relevance_grades": sorted(
                {int(item["grade"]) for item in positives}, reverse=True
            ),
            "music_map_snapshot_sha256":
                str(snapshot["normalized_snapshot_sha256"]),
            "category_a_source_present": bool(pair),
            "source_agreement": {
                "independent_sources": 2 if pair else 1,
                "category_a_target_also_on_music_map": bool(
                    pair and any(
                        item["relevance_scope"] == "artist"
                        and artist_key(str(item["artist"]))
                        == artist_key(str(pair["target"]["artist"]))
                        for item in positives
                    )
                ),
            },
        })
    scene_counts = Counter(str(item["scene"]) for item in records)
    document: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "benchmark_id": "powered-served-list-sonic-dev-v9",
        "created_at": str(snapshots["created_at"]),
        "access_date": ACCESS_DATE,
        "split": "DEVELOPMENT_ONLY",
        "counts": {
            "seeds": len(records),
            "scenes": len(scene_counts),
            "scene_counts": dict(sorted(scene_counts.items())),
            "positives": sum(int(item["positive_count"]) for item in records),
            "minimum_positives_per_seed": min(
                int(item["positive_count"]) for item in records
            ),
            "seeds_with_two_independent_source_agreement": source_agreements,
        },
        "source_protocol": {
            "deciding_sources": [
                "Gnod Music-Map crowd similar-artist maps",
                "existing opened category-A editorial/critic track comparisons",
            ],
            "independent_of_lastfm_music4all_candidate_graph": True,
            "deezer_listenbrainz_musicbrainz_role": "supporting_only",
            "samples_legal_covers_remixes_weak_listicles_excluded": True,
            "artist_scope_disclosure": (
                "An eligible recommended track receives artist-level credit only "
                "when Music-Map explicitly lists that artist as similar to the "
                "seed. This is not a track-specific endorsement."
            ),
            "model_assisted_subjective_labels": False,
        },
        "metric_protocol": {
            "co_primary_1": "graded exponential-gain nDCG@10 on actual served lists",
            "co_primary_2": (
                "top5 scene/style coherence: no unrelated positions 1-3, "
                "at least 4/5 coherent, junk-free"
            ),
            "policy_selection": (
                "Both co-primaries are reported separately and both must pass; "
                "neither is blended with exact-pair retrieval."
            ),
            "exact_pair_retrieval": "diagnostic_only",
            "junk_penalty": (
                "Any duplicate, slowed/reverb, karaoke, tribute, cover/remix, "
                "or seed-title mashup makes coherence zero."
            ),
        },
        "eligibility": {
            "required_scenes": list(REQUIRED_SCENES),
            "automated_catalog_resolution": True,
            "automated_junk_and_same_artist_exclusion": True,
            "one_credit_per_positive_rule": True,
        },
        "input_sha256": {
            **dict(source_inputs),
            str(Path(index_path)): sha256_path(index_path),
        },
        "records": records,
    }
    document["content_sha256"] = sha256_bytes(canonical_bytes(document))
    return document


def validate_gold(document: Mapping[str, Any]) -> Dict[str, Any]:
    records = list(document.get("records", []))
    errors: List[str] = []
    if len(records) < 50:
        errors.append("fewer than 50 seeds")
    scenes = {str(item.get("scene")) for item in records}
    missing = set(REQUIRED_SCENES) - scenes
    if missing:
        errors.append(f"missing scenes: {sorted(missing)}")
    for record in records:
        positives = list(record.get("positives", []))
        if len(positives) < 5:
            errors.append(f"{record.get('id')}: fewer than five positives")
        if len({int(item.get("grade", 0)) for item in positives}) < 2:
            errors.append(f"{record.get('id')}: fewer than two grades")
        if any(int(item.get("grade", 0)) not in (1, 2, 3) for item in positives):
            errors.append(f"{record.get('id')}: invalid grade")
    return {
        "passed": not errors,
        "errors": errors,
        "seeds": len(records),
        "scenes": len(scenes),
        "positives": sum(len(item.get("positives", [])) for item in records),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v6", default="benchmarks/soundalike_pairs.v6.json")
    parser.add_argument(
        "--direct",
        default=(
            ".goals/human-quality-recommendations/artifacts/"
            "catalog-gated-direct-seeds-v8.json"
        ),
    )
    parser.add_argument("--index", default="ml_data/deepvibe_index_v5.npz")
    parser.add_argument(
        "--snapshots", default="benchmarks/evidence/v9/music-map.normalized.json"
    )
    parser.add_argument(
        "--output", default="benchmarks/soundalike_list_gold.v9.json"
    )
    parser.add_argument("--fetch", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    seeds = load_seed_specs(args.v6, args.direct)
    snapshots_path = Path(args.snapshots)
    if args.fetch:
        write_json(snapshots_path, fetch_music_map_snapshots(seeds))
    snapshots = json.loads(snapshots_path.read_text(encoding="utf-8"))
    gold = build_gold(
        seeds,
        snapshots,
        args.index,
        source_inputs={
            str(Path(args.v6)): sha256_path(args.v6),
            str(Path(args.direct)): sha256_path(args.direct),
            str(snapshots_path): sha256_path(snapshots_path),
        },
    )
    validation = validate_gold(gold)
    if not validation["passed"]:
        raise GoldBuildError("; ".join(validation["errors"]))
    write_json(args.output, gold)
    print(json.dumps(validation, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
