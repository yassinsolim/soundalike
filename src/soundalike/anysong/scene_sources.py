"""Collect the scene table's sources: Deezer artists and related artists, Wikidata countries and genres.

    python -m soundalike.anysong.scene_sources --ids index/samples/ids.npy \
        --catalog-artists catalog_artist_ids.json --out scene_data
    python -m soundalike.anysong.scene --data scene_data --catalog-artists catalog_artist_ids.json --out scene.npz

Phases (resumable JSONL under ``--out``; a rerun skips finished ids):

- ``tracks``: catalog tracks without a known artist -> Deezer ``/track/{id}`` (artist, contributors, album).
- ``related``: every catalog artist -> Deezer ``/artist/{id}/related``.
- ``wikidata``: catalog and related artists by Deezer artist id (P2722) -> citizenship (P27), origin (P495),
  formation country (P740/P17), birth country (P19/P17), genres (P136); then country ISO codes and genre parents.

Deezer's public API is used for this personal, non-commercial project at a polite rate (``--rate``, default
5 requests/s of Deezer's ~10/s per IP, leaving room for the live API). Wikidata is CC0; one SPARQL request at a time
with a descriptive User-Agent.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError

import numpy as np

from .deezer import Deezer

WDQS = "https://query.wikidata.org/sparql"
USER_AGENT = "soundalike-anysong/0.1 (+https://soundalike.yassin.app; personal non-commercial music research)"
PROPS = {"P27": "?item wdt:P27 ?val .", "P495": "?item wdt:P495 ?val .",
         "P740": "?item wdt:P740 ?loc . ?loc wdt:P17 ?val .", "P19": "?item wdt:P19 ?b . ?b wdt:P17 ?val .",
         "P136": "?item wdt:P136 ?val .", "P31": "?item wdt:P31 ?val ."}


def _rows(path: Path):
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    yield json.loads(line)
                except ValueError:
                    continue  # a torn final line is fetched again


def _done(path: Path, key: str) -> set[int]:
    return {int(row[key]) for row in _rows(path) if "error" not in row}


class _Sink:
    def __init__(self, path: Path):
        self.fh, self.lock, self.count = path.open("a", encoding="utf-8"), threading.Lock(), 0

    def write(self, row: dict) -> None:
        with self.lock:
            self.fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.fh.flush()
            self.count += 1


def _run(items: list[int], fetch, path: Path, key: str, workers: int, label: str) -> None:
    sink, start = _Sink(path), time.time()

    def one(item: int) -> None:
        try:
            sink.write(fetch(item))
        except Exception as exc:  # recorded, so the next run retries it
            sink.write({key: item, "error": str(exc)[:300]})
        if sink.count % 500 == 0:
            print(f"{label} {sink.count}/{len(items)} ({sink.count / max(time.time() - start, 1e-6):.2f}/s)",
                  flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(one, items))
    print(f"{label}: {sink.count} rows", flush=True)


def track_row(dz: Deezer, tid: int) -> dict:
    data = dz.track(tid)
    if not data or not data.get("id"):
        return {"track_id": tid, "missing": True}
    artist = data.get("artist") or {}
    return {"track_id": tid, "artist_id": artist.get("id"), "artist_name": artist.get("name"),
            "contributors": [[c.get("id"), c.get("name")] for c in data.get("contributors") or []],
            "album_id": (data.get("album") or {}).get("id"), "release_date": data.get("release_date")}


def catalog_artists(ids: list[int], known: dict[int, int], out: Path) -> dict[int, int]:
    artists = {t: known[t] for t in ids if t in known}
    for row in _rows(out / "deezer_tracks.jsonl"):
        if row.get("artist_id"):
            artists.setdefault(int(row["track_id"]), int(row["artist_id"]))
    return artists


def sparql(query: str, retries: int = 6) -> list[dict]:
    body = urllib.parse.urlencode({"query": query, "format": "json"}).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(WDQS, data=body, headers={
                "User-Agent": USER_AGENT, "Accept": "application/sparql-results+json",
                "Content-Type": "application/x-www-form-urlencoded"})
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.load(resp)["results"]["bindings"]
        except HTTPError as exc:
            time.sleep(float(exc.headers.get("Retry-After") or 10 * (attempt + 1)))
        except (OSError, ValueError):
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("Wikidata query failed after retries")


def _qid(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def artist_facts(ids: list[int], query=sparql) -> dict[int, dict]:
    """Wikidata facts for Deezer artist ids; ids without an item come back with empty lists."""
    unions = " UNION ".join(f'{{ {pattern} BIND("{p}" AS ?p) }}' for p, pattern in PROPS.items())
    values = " ".join(f'"{int(i)}"' for i in ids)
    out = {int(i): {"dz": int(i), "items": [], **{p: [] for p in PROPS}} for i in ids}
    for b in query(f"SELECT ?dz ?item ?p ?val WHERE {{ VALUES ?dz {{ {values} }} ?item wdt:P2722 ?dz . "
                   f"OPTIONAL {{ {unions} }} }}"):
        row = out[int(b["dz"]["value"])]
        if _qid(b["item"]["value"]) not in row["items"]:
            row["items"].append(_qid(b["item"]["value"]))
        if "p" in b and "val" in b and _qid(b["val"]["value"]) not in row[b["p"]["value"]]:
            row[b["p"]["value"]].append(_qid(b["val"]["value"]))
    return out


def label_values(out: Path, batch: int = 200, query=sparql) -> None:
    path = out / "wikidata_values.json"
    known = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"countries": {}, "genres": {}}
    countries, genres = set(), set()
    for row in _rows(out / "wikidata_artists.jsonl"):
        countries.update(q for p in ("P27", "P495", "P740", "P19") for q in row[p])
        genres.update(row["P136"])
    todo_c, todo_g = sorted(countries - set(known["countries"])), sorted(genres - set(known["genres"]))
    for start in range(0, len(todo_c), batch):
        vals = " ".join(f"wd:{q}" for q in todo_c[start:start + batch])
        for b in query(f'SELECT ?v ?label ?iso ?parentIso WHERE {{ VALUES ?v {{ {vals} }} '
                       f'OPTIONAL {{ ?v rdfs:label ?label FILTER(LANG(?label)="en") }} OPTIONAL {{ ?v wdt:P297 ?iso }} '
                       f'OPTIONAL {{ ?v wdt:P17 ?pc . ?pc wdt:P297 ?parentIso }} }}'):
            entry = known["countries"].setdefault(_qid(b["v"]["value"]), {"label": None, "iso": [], "parent_iso": []})
            entry["label"] = b.get("label", {}).get("value", entry["label"])
            for name, field in (("iso", "iso"), ("parentIso", "parent_iso")):
                if name in b and b[name]["value"] not in entry[field]:
                    entry[field].append(b[name]["value"])
    for start in range(0, len(todo_g), batch):
        vals = " ".join(f"wd:{q}" for q in todo_g[start:start + batch])
        for b in query(f'SELECT ?v ?label ?parent WHERE {{ VALUES ?v {{ {vals} }} '
                       f'OPTIONAL {{ ?v rdfs:label ?label FILTER(LANG(?label)="en") }} OPTIONAL {{ ?v wdt:P279 ?parent }} }}'):
            entry = known["genres"].setdefault(_qid(b["v"]["value"]), {"label": None, "parents": []})
            entry["label"] = b.get("label", {}).get("value", entry["label"])
            if "parent" in b and _qid(b["parent"]["value"]) not in entry["parents"]:
                entry["parents"].append(_qid(b["parent"]["value"]))
    path.write_text(json.dumps(known, ensure_ascii=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="anysong-scene-sources", description="collect scene-table sources")
    p.add_argument("--ids", type=Path, required=True, help="served track ids (an index channel's ids.npy)")
    p.add_argument("--catalog-artists", type=Path, required=True, help="JSON track id -> Deezer artist id")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--phases", nargs="+", default=["tracks", "related", "wikidata"],
                   choices=["tracks", "related", "wikidata"])
    p.add_argument("--rate", type=float, default=5.0, help="Deezer requests per second")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--batch", type=int, default=200, help="Deezer artist ids per Wikidata query")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    ids = [int(t) for t in np.load(args.ids)]
    known = {int(k): int(v) for k, v in json.loads(args.catalog_artists.read_text(encoding="utf-8")).items()}
    dz = Deezer(args.out / "deezer_cache", rate_per_s=args.rate)
    if "tracks" in args.phases:
        done = _done(args.out / "deezer_tracks.jsonl", "track_id")
        todo = [t for t in ids if t not in known and t not in done]
        _run(todo, lambda t: track_row(dz, t), args.out / "deezer_tracks.jsonl", "track_id", args.workers, "tracks")
    artists = catalog_artists(ids, known, args.out)
    if "related" in args.phases:
        counts = Counter(artists.values())
        done = _done(args.out / "deezer_related.jsonl", "artist_id")
        todo = sorted((a for a in counts if a not in done), key=lambda a: (-counts[a], a))
        _run(todo, lambda a: {"artist_id": a, "related": [[r, None, None] for r in dz.related_artists(a)]},
             args.out / "deezer_related.jsonl", "artist_id", args.workers, "related")
    if "wikidata" in args.phases:
        wanted = set(artists.values())
        for row in _rows(args.out / "deezer_related.jsonl"):
            wanted.update(int(r[0]) for r in row.get("related") or [] if r and r[0])
        path = args.out / "wikidata_artists.jsonl"
        todo = sorted(wanted - {int(row["dz"]) for row in _rows(path)})
        with path.open("a", encoding="utf-8") as fh:
            for start in range(0, len(todo), args.batch):
                for row in artist_facts(todo[start:start + args.batch]).values():
                    fh.write(json.dumps(row) + "\n")
                fh.flush()
                time.sleep(1.0)
        label_values(args.out)
    print(f"done: {len(artists)} catalog tracks with artists -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
