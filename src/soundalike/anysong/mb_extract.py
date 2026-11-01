"""Extract recording relationships from a MusicBrainz full-export dump (CC0).

Streams the PostgreSQL COPY files extracted from ``mbdump.tar.bz2`` so memory
stays bounded; only rows touching the selected relationships are kept.

Output JSON lines: one per (source recording, target recording) relation with
titles, artist credits, ISRCs and the relation name/direction.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

SAMPLES_MATERIAL = "9efd9ce9-e702-448b-8e76-641515e8fe62"


def _rows(path: Path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            yield line.rstrip("\n").split("\t")


def extract(dump_dir: Path, link_type_gids: list[str]) -> list[dict]:
    lt = {}
    for r in _rows(dump_dir / "link_type"):
        if r[3] in link_type_gids:
            lt[r[0]] = {"gid": r[3], "e0": r[4], "e1": r[5], "name": r[6],
                        "link_phrase": r[8], "reverse_phrase": r[9]}
    print("link types:", {v["name"]: (v["e0"], v["e1"], v["link_phrase"], v["reverse_phrase"]) for v in lt.values()})
    links = {}
    for r in _rows(dump_dir / "link"):
        if r[1] in lt:
            links[r[0]] = r[1]
    print("links:", len(links))
    rels = []
    for r in _rows(dump_dir / "l_recording_recording"):
        if r[1] in links:
            rels.append((int(r[2]), int(r[3]), links[r[1]]))
    print("recording-recording relations:", len(rels))
    want = {a for a, _, _ in rels} | {b for _, b, _ in rels}
    rec, credits = {}, set()
    for r in _rows(dump_dir / "recording"):
        rid = int(r[0])
        if rid in want:
            rec[rid] = {"mbid": r[1], "title": r[2], "ac": int(r[3]),
                        "length_ms": int(r[4]) if r[4] != "\\N" else None,
                        "video": r[8] == "t"}
            credits.add(int(r[3]))
    ac = {}
    for r in _rows(dump_dir / "artist_credit"):
        if int(r[0]) in credits:
            ac[int(r[0])] = r[1]
    isrc = defaultdict(list)
    for r in _rows(dump_dir / "isrc"):
        rid = int(r[1])
        if rid in want:
            isrc[rid].append(r[2])
    out = []
    for e0, e1, ltid in rels:
        a, b = rec.get(e0), rec.get(e1)
        if not a or not b or a["video"] or b["video"]:
            continue
        out.append({
            "relation": lt[ltid]["name"],
            "relation_gid": lt[ltid]["gid"],
            "entity0": {"mbid": a["mbid"], "title": a["title"], "artist": ac.get(a["ac"], ""),
                        "length_ms": a["length_ms"], "isrcs": sorted(set(isrc.get(e0, [])))},
            "entity1": {"mbid": b["mbid"], "title": b["title"], "artist": ac.get(b["ac"], ""),
                        "length_ms": b["length_ms"], "isrcs": sorted(set(isrc.get(e1, [])))},
        })
    return out


def extract_covers(dump_dir: Path, max_works: int, seed: int = 0) -> list[dict]:
    """Works that have both a cover recording and a non-cover recording, all with ISRCs.

    Uses the recording->work "performance" relationship and its "cover" attribute.
    """
    perf_types = {r[0] for r in _rows(dump_dir / "link_type")
                  if r[4] == "recording" and r[5] == "work" and r[6] == "performance"}
    cover_attr = {r[0] for r in _rows(dump_dir / "link_attribute_type") if r[5] == "cover"}
    print("performance link types:", perf_types, "cover attribute:", cover_attr)
    perf_links = {r[0] for r in _rows(dump_dir / "link") if r[1] in perf_types}
    cover_links = {r[0] for r in _rows(dump_dir / "link_attribute")
                   if r[1] in cover_attr and r[0] in perf_links}
    print("performance links:", len(perf_links), "cover links:", len(cover_links))
    has_cover, has_orig = set(), set()
    for r in _rows(dump_dir / "l_recording_work"):
        if r[1] in perf_links:
            (has_cover if r[1] in cover_links else has_orig).add(int(r[3]))
    works = has_cover & has_orig
    print("works with cover and non-cover recordings:", len(works))
    members: dict[int, list[tuple[int, bool]]] = defaultdict(list)
    for r in _rows(dump_dir / "l_recording_work"):
        w = int(r[3])
        if r[1] in perf_links and w in works:
            members[w].append((int(r[2]), r[1] in cover_links))
    recs = {rid for ms in members.values() for rid, _ in ms}
    isrc = defaultdict(list)
    for r in _rows(dump_dir / "isrc"):
        rid = int(r[1])
        if rid in recs:
            isrc[rid].append(r[2])
    eligible = [w for w, ms in members.items()
                if any(c and isrc.get(rid) for rid, c in ms) and any(not c and isrc.get(rid) for rid, c in ms)]
    random.Random(seed).shuffle(eligible)
    eligible = eligible[:max_works]
    keep = {rid for w in eligible for rid, _ in members[w] if isrc.get(rid)}
    rec, credits = {}, set()
    for r in _rows(dump_dir / "recording"):
        rid = int(r[0])
        if rid in keep:
            rec[rid] = {"mbid": r[1], "title": r[2], "ac": int(r[3]),
                        "length_ms": int(r[4]) if r[4] != "\\N" else None, "video": r[8] == "t"}
            credits.add(int(r[3]))
    ac = {int(r[0]): r[1] for r in _rows(dump_dir / "artist_credit") if int(r[0]) in credits}
    work_meta = {}
    wanted_works = set(eligible)
    for r in _rows(dump_dir / "work"):
        if int(r[0]) in wanted_works:
            work_meta[int(r[0])] = {"mbid": r[1], "title": r[2]}
    out = []
    for w in eligible:
        recs_out = []
        for rid, is_cover in members[w]:
            m = rec.get(rid)
            if not m or m["video"]:
                continue
            recs_out.append({"mbid": m["mbid"], "title": m["title"], "artist": ac.get(m["ac"], ""),
                             "length_ms": m["length_ms"], "isrcs": sorted(set(isrc[rid])), "cover": is_cover})
        out.append({"relation": "same_work_cover", "work": work_meta.get(w, {}), "recordings": recs_out})
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="mb_extract")
    p.add_argument("--dump-dir", required=True, help="directory containing extracted mbdump/* tables")
    p.add_argument("--out", required=True)
    p.add_argument("--link-type", action="append", default=None)
    p.add_argument("--covers", type=int, default=0, help="extract N cover works instead of sample relations")
    args = p.parse_args(argv)
    if args.covers:
        works = extract_covers(Path(args.dump_dir), args.covers)
        with open(args.out, "w", encoding="utf-8") as fh:
            for r in works:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(works)} cover works, {sum(len(w['recordings']) for w in works)} recordings")
        return
    rows = extract(Path(args.dump_dir), args.link_type or [SAMPLES_MATERIAL])
    random.Random(0).shuffle(rows)
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    both = sum(1 for r in rows if r["entity0"]["isrcs"] and r["entity1"]["isrcs"])
    print(f"wrote {len(rows)} relations; both sides with ISRC: {both}")


if __name__ == "__main__":
    sys.exit(main())
