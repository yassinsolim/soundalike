"""Bulk, resumable Deezer preview downloader.

    python -m soundalike.anysong.previews --ids ids.txt --out /data/previews
    python -m soundalike.anysong.previews --random-catalog 20000 --seed 7 --catalog /data/catalog_meta.npz

Writes ``<out>/manifest.jsonl`` entries ({id, ok, path|reason}) so reruns skip work.
"""

from __future__ import annotations

import argparse
import errno
import json
import sys
import threading
import time
import warnings
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Callable, Iterable, Iterator, TypeVar

import numpy as np

from .deezer import Deezer

T = TypeVar("T")
R = TypeVar("R")


def bounded_results(executor: ThreadPoolExecutor, fn: Callable[[T], R],
                    items: Iterable[T], limit: int) -> Iterator[R]:
    """Keep workers fed without queuing the entire catalog or aging signed URLs."""
    if limit < 1:
        raise ValueError("Pending-job limit must be positive")
    iterator = iter(items)
    pending = set()
    exhausted = False
    try:
        while pending or not exhausted:
            while not exhausted and len(pending) < limit:
                try:
                    item = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                pending.add(executor.submit(fn, item))
            if pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    yield future.result()
    finally:
        for future in pending:
            future.cancel()


def artist_groups(path: Path, todo: list[int]) -> list[tuple[int, set[int]]]:
    mapping = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(mapping, dict):
        raise ValueError("Artist map must be an object mapping track IDs to positive artist IDs")
    wanted = set(todo)
    groups: dict[int, set[int]] = defaultdict(set)
    for tid, aid in mapping.items():
        if (not tid.isdecimal() or int(tid) < 1 or str(int(tid)) != tid
                or type(aid) is not int or aid < 1):
            raise ValueError(f"Invalid artist mapping for track {tid}")
        if int(tid) in wanted:
            groups[aid].add(int(tid))
    return sorted(((aid, ids) for aid, ids in groups.items() if len(ids) >= 3),
                  key=lambda group: (-len(group[1]), group[0]))


def load_ids(args: argparse.Namespace) -> list[int]:
    ids: list[int] = []
    if args.ids:
        for line in Path(args.ids).read_text().split():
            ids.append(int(line))
    if args.random_catalog:
        cat = np.load(args.catalog, allow_pickle=True)["track_ids"].astype(np.int64)
        rng = np.random.default_rng(args.seed)
        ids.extend(int(x) for x in rng.choice(cat, size=args.random_catalog, replace=False))
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="previews")
    p.add_argument("--ids")
    p.add_argument("--random-catalog", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--catalog", default="/data/catalog_meta.npz")
    p.add_argument("--out", default="/data/previews")
    p.add_argument("--rate", type=float, default=8.0)
    p.add_argument("--preview-rate", type=float, default=16.0, help="Maximum preview CDN requests per second")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--artist-map", help="JSON track ID -> artist ID map for batched preview metadata")
    p.add_argument("--write-ids")
    args = p.parse_args(argv)
    if not args.ids and not args.random_catalog:
        p.error("provide --ids or --random-catalog")
    if args.rate <= 0 or args.preview_rate <= 0 or args.workers < 1 or args.random_catalog < 0:
        p.error("rate and workers must be positive; random-catalog must be non-negative")

    ids = load_ids(args)
    if args.write_ids:
        Path(args.write_ids).write_text("\n".join(map(str, ids)) + "\n")
    dz = Deezer(args.out, rate_per_s=args.rate, preview_rate_per_s=args.preview_rate)
    manifest = Path(args.out) / "manifest.jsonl"
    done: set[int] = set()
    if manifest.exists():
        with manifest.open(encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                try:
                    record = json.loads(line)
                    tid = int(record["id"])
                    if record.get("reason") == "no_preview":
                        done.add(tid)
                    else:
                        done.discard(tid)
                except (ValueError, KeyError, TypeError):
                    warnings.warn(f"{manifest}:{line_no}: ignoring malformed manifest record",
                                  RuntimeWarning, stacklevel=2)
    todo = [i for i in ids if i not in done and not (
        dz.preview_path(i).exists() and dz.preview_path(i).stat().st_size > 10_000)]
    groups = artist_groups(Path(args.artist_map), todo) if args.artist_map else []
    print(f"{len(ids)} ids, {len(ids) - len(todo)} already done, {len(todo)} to fetch", flush=True)
    if args.artist_map:
        print(f"{len(groups)} artist batches covering {sum(len(ids) for _, ids in groups)} pending IDs; "
              "unmatched IDs use individual lookups", flush=True)
    lock = threading.Lock()
    stats = {"ok": 0, "miss": 0, "err": 0}
    batch_stats = {"requests": 0, "matched": 0, "errors": 0}
    phase = "batches" if groups else "individual"
    downloaded_bytes = 0
    t0 = time.time()

    def work(item: tuple[int, dict | None]) -> None:
        nonlocal downloaded_bytes
        tid, meta = item
        size = 0
        try:
            path = dz.fetch_preview(tid, meta=meta) if meta is not None else dz.fetch_preview(tid)
            size = path.stat().st_size if path else 0
            rec = {"id": tid, "ok": bool(path), **({"path": str(path)} if path else {"reason": "no_preview"})}
            key = "ok" if path else "miss"
        except (OSError, RuntimeError, ValueError) as exc:
            if isinstance(exc, OSError) and exc.errno in (errno.ENOSPC, errno.EDQUOT):
                raise
            rec = {"id": tid, "ok": False, "reason": str(exc)[:200]}
            key = "err"
            print(f"preview failed {tid}: {exc}", flush=True)
        with lock:
            stats[key] += 1
            downloaded_bytes += size
            with open(manifest, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
            n = sum(stats.values())
            if n % 500 == 0:
                rate = n / max(time.time() - t0, 1e-6)
                mbps = downloaded_bytes * 8 / max(time.time() - t0, 1e-6) / 1e6
                print(f"{n}/{len(todo)} {stats} {rate:.1f}/s phase={phase} "
                      f"current-rate ETA {(len(todo) - n) / max(rate, 1e-6) / 60:.1f} min",
                      flush=True)
                print(f"  audio={mbps:.1f} Mbps metadata_batches={batch_stats}", flush=True)

    def resolve_group(group: tuple[int, set[int]]) -> dict[int, dict]:
        aid, wanted = group
        try:
            rows = dz.artist_top_tracks(aid)
            matches = {r["id"]: r for r in rows if r["id"] in wanted
                       and isinstance(r.get("preview"), str) and r["preview"]}
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"artist metadata failed {aid}; using individual lookups: {exc}", flush=True)
            with lock:
                batch_stats["errors"] += 1
            matches = {}
        with lock:
            batch_stats["requests"] += 1
            batch_stats["matched"] += len(matches)
        return matches

    def jobs() -> Iterator[tuple[int, dict | None]]:
        nonlocal phase
        scheduled: set[int] = set()
        if groups:
            with ThreadPoolExecutor(min(args.workers, 4)) as metadata:
                for matches in bounded_results(metadata, resolve_group, groups, min(args.workers, 4)):
                    for tid, meta in matches.items():
                        scheduled.add(tid)
                        yield tid, meta
            with lock:
                phase = "individual"
                print(f"Artist batches resolved {len(scheduled)} previews; "
                      f"{len(todo) - len(scheduled)} IDs need individual lookups. "
                      "The remaining phase is API-rate limited.", flush=True)
        for tid in todo:
            if tid not in scheduled:
                yield tid, None

    with ThreadPoolExecutor(args.workers) as ex:
        for _ in bounded_results(ex, work, jobs(), args.workers * 4):
            pass
    print("done", stats, flush=True)
    print("metadata batches", batch_stats, flush=True)
    if stats["err"]:
        raise RuntimeError(f"{stats['err']} preview downloads failed; rerun to retry errors")


if __name__ == "__main__":
    sys.exit(main())
