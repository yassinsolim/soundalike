"""Build the any-song benchmark (v7) from lawful, attributable sources.

Groups / pairs are stored as facts only (titles, artists, Deezer ids, source
URLs); no audio or lyrics are committed.

Sources
- Gen Alpha Melody playlist (Carl E. Martin, "The Gen Alpha Melody", public
  Spotify playlist 5LF1VGZdTKEoFT9WVR7Pwm): songs sharing one melodic contour.
- MusicBrainz "samples material" recording relationships (CC0 data dump).
- Wikidata P5707 "samples from work" statements (CC0).
- The frozen v4 sourced pairs (benchmarks/soundalike_pairs.v4.json).

Usage (on the GPU host):
    python -m soundalike.anysong.bench_build genalpha --playlist genalpha_embed.json --out genalpha.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

from .deezer import Deezer, base_title, normalize


def _load_catalog(path: str | Path) -> dict[int, int]:
    z = np.load(path, allow_pickle=True)
    return {int(t): i for i, t in enumerate(z["track_ids"])}


def _split_artists(subtitle: str) -> list[str]:
    return [a.replace("\xa0", " ").strip() for a in subtitle.split(",") if a.strip()]


def build_genalpha(args: argparse.Namespace) -> None:
    raw = json.loads(Path(args.playlist).read_text(encoding="utf-8-sig"))
    tracks = raw["props"]["pageProps"]["state"]["data"]["entity"]["trackList"]
    catalog = _load_catalog(args.catalog)
    dz = Deezer(args.previews)
    members = []
    for i, t in enumerate(tracks):
        artists = _split_artists(t.get("subtitle", ""))
        dur = (t.get("duration") or 0) / 1000.0 or None
        m = dz.match(t["title"], artists, dur)
        row = {
            "spotify_uri": t["uri"],
            "title": t["title"],
            "artists": artists,
            "deezer_id": m.deezer_id if m else None,
            "deezer_title": m.title if m else None,
            "deezer_artist": m.artist if m else None,
            "isrc": m.isrc if m else None,
            "match_score": round(m.score, 3) if m else None,
            "in_catalogue": bool(m and m.deezer_id in catalog),
        }
        members.append(row)
        flag = "OK " if m else "-- "
        print(f"{flag}{i:02d} {t['title']} | {', '.join(artists)} -> "
              f"{(m.title + ' | ' + m.artist + ' | ' + str(m.deezer_id)) if m else 'NO MATCH'}"
              f"{' [cat]' if row['in_catalogue'] else ''}", flush=True)
    out = {
        "group_id": "gen_alpha_melody",
        "relation": "shared_melody",
        "source": {
            "name": "The Gen Alpha Melody (Carl E. Martin) companion playlist",
            "video": "https://www.youtube.com/watch?v=DW0XUsyBBuY",
            "playlist": "https://open.spotify.com/playlist/5LF1VGZdTKEoFT9WVR7Pwm",
        },
        "members": members,
    }
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    ok = sum(1 for r in members if r["deezer_id"])
    print(f"matched {ok}/{len(members)}; in catalogue {sum(r['in_catalogue'] for r in members)}")


def _stable_split(key: str, held_out_fraction: float = 0.5) -> str:
    h = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "held_out" if h < held_out_fraction else "development"


def _resolve_isrcs(dz: Deezer, isrcs: list[str], title: str, length_ms: int | None,
                   cache: dict) -> dict | None:
    """Resolve a MusicBrainz recording to a Deezer track via its ISRCs.

    Guards against ISRC reuse/mislabels by requiring a loose title match and,
    when MusicBrainz knows the length, a duration within 20 s.
    """
    want = base_title(title)
    for isrc in isrcs[:3]:
        if isrc in cache:
            meta = cache[isrc]
        else:
            try:
                meta = dz.track_by_isrc(isrc)
            except RuntimeError:
                meta = {}
            cache[isrc] = {k: meta.get(k) for k in ("id", "title", "duration", "preview", "artist", "isrc")} if meta else {}
            meta = cache[isrc]
        if not meta or not meta.get("id"):
            continue
        got = base_title(meta.get("title") or "")
        if not got or not (got == want or got.startswith(want) or want.startswith(got)):
            continue
        if length_ms and meta.get("duration") and abs(meta["duration"] - length_ms / 1000) > 20:
            continue
        return meta
    return None


def _split_credit(artist: str) -> list[str]:
    parts = re.split(r",|;|\s+feat\.?\s+|\s+ft\.?\s+|\s+featuring\s+|\s+x\s+|\s+&\s+|\s+and\s+|\s+with\s+",
                     artist or "", flags=re.I)
    names = [p.strip() for p in parts if p.strip()]
    return [artist.strip()] + [n for n in names if n != artist.strip()]


def _catalog_title_index(path: str | Path) -> dict[tuple[str, str], int]:
    z = np.load(path, allow_pickle=True)
    idx: dict[tuple[str, str], int] = {}
    for tid, title, artist in zip(z["track_ids"], z["titles"], z["artists"]):
        for a in _split_credit(str(artist))[:3]:
            idx.setdefault((base_title(str(title)), normalize(a)), int(tid))
    return idx


def _resolve_title_artist(dz: Deezer, cat: dict, title: str, artist: str) -> tuple[int | None, str]:
    names = _split_credit(artist)
    for a in names:
        tid = cat.get((base_title(title), normalize(a)))
        if tid:
            return tid, "catalogue"
    m = dz.match(title, names)
    return (m.deezer_id, "search") if m else (None, "unresolved")


def build_title_pairs(args: argparse.Namespace) -> None:
    """Resolve title/artist pairs (frozen v4 JSON or sampleid's samplepairs.tsv) to Deezer ids."""
    src = Path(args.input)
    if src.suffix == ".tsv":
        rows = []
        try:
            text = src.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            warnings.warn(f"{src}: reading SamplePairs metadata as MacRoman, not UTF-8",
                          RuntimeWarning, stacklevel=2)
            text = src.read_text(encoding="mac_roman")
        for line_no, c in enumerate(csv.reader(text.splitlines(), delimiter="\t"), 1):
            if line_no == 1 or not c:
                continue
            if len(c) < 13:
                raise ValueError(f"{src}:{line_no}: expected 13 SamplePairs columns, got {len(c)}")
            rows.append({"id": f"SID{int(c[0]):03d}", "split": "held_out", "relation": "samples",
                         "query": {"title": c[4], "artist": c[3]}, "target": {"title": c[10], "artist": c[9]},
                         "source": {"name": "sony/sampleid samplepairs.tsv (MIT)", "query_url": c[2],
                                    "target_url": c[8], "query_start": c[6], "target_start": c[12]}})
    else:
        raw = json.loads(src.read_text(encoding="utf-8"))
        rows = raw["pairs"] if isinstance(raw, dict) else raw
    cat = _catalog_title_index(args.catalog)
    dz = Deezer(args.previews, rate_per_s=args.rate)
    out, how = [], defaultdict(int)
    for r in rows:
        ids = []
        for side in ("query", "target"):
            tid, via = _resolve_title_artist(dz, cat, r[side]["title"], r[side]["artist"])
            how[via] += 1
            if tid and not dz.fetch_preview(tid):
                tid, via = None, "no_preview"
            ids.append(tid)
        row = {k: r[k] for k in r if k not in ("query", "target")}
        row["query"] = {**r["query"], "deezer_id": ids[0]}
        row["target"] = {**r["target"], "deezer_id": ids[1]}
        out.append(row)
        print(f"{r['id']}: {r['query']['title']} -> {ids[0]} | {r['target']['title']} -> {ids[1]}", flush=True)
    Path(args.out).write_text(json.dumps({"pairs": out}, indent=1, ensure_ascii=False), encoding="utf-8")
    ok = sum(1 for r in out if r["query"]["deezer_id"] and r["target"]["deezer_id"])
    print(f"resolved {ok}/{len(out)} pairs; sides by method {dict(how)}")


def build_mb_samples(args: argparse.Namespace) -> None:
    rows = [json.loads(l) for l in Path(args.relations).read_text(encoding="utf-8").splitlines() if l]
    rows = [r for r in rows if r["entity0"]["isrcs"] and r["entity1"]["isrcs"]]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    per_target: dict[str, int] = defaultdict(int)
    per_query: dict[str, int] = defaultdict(int)
    picked = []
    for r in rows:
        q, t = r["entity0"], r["entity1"]
        if q["mbid"] == t["mbid"] or normalize(q["artist"]) == normalize(t["artist"]):
            continue
        if per_target[t["mbid"]] >= args.cap_per_target or per_query[q["mbid"]] >= args.cap_per_query:
            continue
        per_target[t["mbid"]] += 1
        per_query[q["mbid"]] += 1
        picked.append(r)
        if len(picked) >= args.max_pairs:
            break
    print(f"{len(rows)} ISRC-complete relations -> {len(picked)} after caps", flush=True)
    dz = Deezer(args.previews, rate_per_s=args.rate)
    cache_path = Path(args.isrc_cache)
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    pairs = []
    for n, r in enumerate(picked):
        sides = []
        for ent in (r["entity0"], r["entity1"]):
            meta = _resolve_isrcs(dz, ent["isrcs"], ent["title"], ent["length_ms"], cache)
            sides.append(meta)
        if all(sides):
            ok = all(dz.fetch_preview(m["id"], m) for m in sides)
            if ok:
                q, t = r["entity0"], r["entity1"]
                pairs.append({
                    "id": f"MBS{len(pairs) + 1:05d}",
                    "split": _stable_split(normalize(t["artist"])),
                    "relation": "samples",
                    "query": {"title": q["title"], "artist": q["artist"], "mbid": q["mbid"],
                              "deezer_id": int(sides[0]["id"])},
                    "target": {"title": t["title"], "artist": t["artist"], "mbid": t["mbid"],
                               "deezer_id": int(sides[1]["id"])},
                    "source": {"name": "MusicBrainz", "license": "CC0-1.0",
                               "relationship": "samples material",
                               "url": f"https://musicbrainz.org/recording/{q['mbid']}"},
                })
        if (n + 1) % 200 == 0:
            cache_path.write_text(json.dumps(cache))
            print(f"{n + 1}/{len(picked)} resolved pairs={len(pairs)}", flush=True)
    cache_path.write_text(json.dumps(cache))
    Path(args.out).write_text(json.dumps({"pairs": pairs}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(pairs)} pairs ({sum(p['split'] == 'held_out' for p in pairs)} held-out)")


def build_mb_covers(args: argparse.Namespace) -> None:
    works = [json.loads(l) for l in Path(args.works).read_text(encoding="utf-8").splitlines() if l]
    dz = Deezer(args.previews, rate_per_s=args.rate)
    cache_path = Path(args.isrc_cache)
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    rng = random.Random(args.seed)
    groups = []
    for n, w in enumerate(works[: args.max_works]):
        recs = list(w["recordings"])
        rng.shuffle(recs)
        covers = [r for r in recs if r["cover"]][: args.max_covers]
        origs = [r for r in recs if not r["cover"]][: args.max_originals]
        members = []
        for r in covers + origs:
            meta = _resolve_isrcs(dz, r["isrcs"], r["title"], r["length_ms"], cache)
            if meta and dz.fetch_preview(meta["id"], meta):
                members.append({"title": r["title"], "artist": r["artist"], "mbid": r["mbid"],
                                "cover": r["cover"], "deezer_id": int(meta["id"])})
        ids = {m["deezer_id"] for m in members}
        if len(ids) == len(members) and any(m["cover"] for m in members) and any(not m["cover"] for m in members):
            groups.append({
                "id": f"MBC{len(groups) + 1:05d}",
                "split": _stable_split(w["work"].get("mbid", str(n))),
                "relation": "same_work",
                "work": w["work"],
                "members": members,
                "source": {"name": "MusicBrainz", "license": "CC0-1.0",
                           "url": f"https://musicbrainz.org/work/{w['work'].get('mbid', '')}"},
            })
        if (n + 1) % 100 == 0:
            cache_path.write_text(json.dumps(cache))
            print(f"{n + 1} works -> {len(groups)} groups", flush=True)
    cache_path.write_text(json.dumps(cache))
    Path(args.out).write_text(json.dumps({"groups": groups}, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(groups)} cover groups")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="bench_build")
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("genalpha")
    g.add_argument("--playlist", required=True)
    g.add_argument("--catalog", default="/data/catalog_meta.npz")
    g.add_argument("--previews", default="/data/previews")
    g.add_argument("--out", required=True)
    g.set_defaults(func=build_genalpha)
    s = sub.add_parser("mb-samples")
    s.add_argument("--relations", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--previews", default="/data/previews")
    s.add_argument("--isrc-cache", default="/data/bench/isrc_cache.json")
    s.add_argument("--max-pairs", type=int, default=2500)
    s.add_argument("--cap-per-target", type=int, default=3)
    s.add_argument("--cap-per-query", type=int, default=2)
    s.add_argument("--rate", type=float, default=8.5)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=build_mb_samples)
    c = sub.add_parser("mb-covers")
    c.add_argument("--works", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--previews", default="/data/previews")
    c.add_argument("--isrc-cache", default="/data/bench/isrc_cache.json")
    c.add_argument("--max-works", type=int, default=1500)
    c.add_argument("--max-covers", type=int, default=2)
    c.add_argument("--max-originals", type=int, default=2)
    c.add_argument("--rate", type=float, default=8.5)
    c.add_argument("--seed", type=int, default=0)
    c.set_defaults(func=build_mb_covers)
    t = sub.add_parser("title-pairs")
    t.add_argument("--input", required=True, help="v4 pairs JSON or sampleid samplepairs.tsv")
    t.add_argument("--out", required=True)
    t.add_argument("--catalog", default="/data/catalog_meta.npz")
    t.add_argument("--previews", default="/data/previews")
    t.add_argument("--rate", type=float, default=8.5)
    t.set_defaults(func=build_title_pairs)
    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
