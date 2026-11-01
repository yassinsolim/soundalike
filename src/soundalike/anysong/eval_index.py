"""Evaluate a built any-song index the way the API presents results.

Every benchmark query that is in the index is ranked from its stored windows with itself excluded: the
offline twin of an unseen live query (``serve_check`` shows live and offline rankings agree).

raw      every catalog row, as ``evaluate``: other releases of the query or of a positive count.
songs    the API view: rows collapsed to one entry per song (``api.song_key``) and the query's own song
         removed; a positive song counts as found when any of its releases is.
any_tab  a positive song in the top k of any channel (the page shows three tabs of 20).

Also reported: exact brute force vs two-stage agreement on a sample (``--exact``), hubness over random
catalog queries (``--hubs``) and named example rankings (``--examples``).

    python -m soundalike.anysong.eval_index --index index_full --meta bench/catalog_meta_full_v1.npz \
        --bench bench --exact 256 --hubs 2000 --out logs/eval_full.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from .api import song_key
from .evaluate import Query, load_benchmarks, score_block
from .serve import CHANNELS, ChannelIndex, Engine, TwoStageIndex, load_meta

TAB_K = 20
HEADLINE = {
    "melody": ("genalpha/test", "mb_covers/held_out"),
    "samples": ("mb_samples/held_out", "mb_samples_rev/held_out", "sampleid/held_out"),
    "sonic": ("v4_pure_sonic/held_out", "v4_pure_sonic/validation"),
    "any_tab": ("genalpha/test", "v4/held_out", "v4/validation", "mb_covers/held_out", "mb_samples/held_out",
                "sampleid/held_out"),
}
METRICS = ("ap", "rr", "hr@1", "hr@10", "hr@20", "p@10", "r@100")


class SongKeys:
    """Integer song identity per track ID; a track without metadata is its own song."""

    def __init__(self, meta: dict[int, tuple[str, str]]):
        self.meta = meta
        self._ids: dict[tuple[str, str], int] = {}
        self.names: list[str] = []
        self.of: dict[int, int] = {}
        for tid, (artist, title) in meta.items():
            self.of[int(tid)] = self._key(song_key(artist, title), f"{artist} - {title}")

    def _key(self, key: tuple[str, str], name: str) -> int:
        if key not in self._ids:
            self._ids[key] = len(self._ids)
            self.names.append(name)
        return self._ids[key]

    def get(self, track_id: int) -> int:
        tid = int(track_id)
        if tid not in self.of:
            self.of[tid] = self._key(("#", str(tid)), f"#{tid}")
        return self.of[tid]

    def rows(self, ids: np.ndarray, device) -> torch.Tensor:
        return torch.tensor([self.get(t) for t in ids.tolist()], dtype=torch.long, device=device)


def song_best(row: torch.Tensor, rowkey: torch.Tensor, qkey: int | None) -> torch.Tensor:
    """Best score per song (``-inf`` for songs with no ranked row), the query's own song removed."""
    best = torch.full((int(rowkey.max()) + 1,), -torch.inf, device=row.device)
    best.scatter_reduce_(0, rowkey, row.float(), reduce="amax")
    if qkey is not None and qkey < len(best):
        best[qkey] = -torch.inf
    return best


def song_ranks(best: torch.Tensor, pkeys: set[int]) -> np.ndarray:
    """Sorted 1-based ranks of the positive songs present; ties go to the lower song key."""
    present = sorted(k for k in pkeys if k < len(best) and best[k] > -torch.inf)
    if not present:
        return np.array([], dtype=np.int64)
    pi = torch.tensor(present, device=best.device)
    pv = best[pi]
    order = torch.arange(len(best), device=best.device)
    ahead = (best[None, :] > pv[:, None]) | ((best[None, :] == pv[:, None]) & (order[None, :] < pi[:, None]))
    return (ahead.sum(1) + 1).sort().values.cpu().numpy()


def song_top(best: torch.Tensor, k: int) -> list[int]:
    vals, idx = best.topk(min(k, len(best)))
    return [int(i) for i, v in zip(idx.tolist(), vals.tolist()) if v > -np.inf]


def metrics(ranks: np.ndarray) -> dict[str, float]:
    n = len(ranks)
    return {"ap": float(np.mean(np.arange(1, n + 1) / ranks)), "rr": 1.0 / float(ranks[0]),
            "hr@1": float(ranks[0] <= 1), "hr@10": float(ranks[0] <= 10), "hr@20": float(ranks[0] <= TAB_K),
            "p@10": float((ranks <= 10).sum()) / 10, "r@100": float((ranks <= 100).mean()),
            "first": float(ranks[0])}


def aggregate(rows: list[dict[str, float]]) -> dict[str, float | int | None]:
    if not rows:
        return {"n": 0}
    out: dict[str, float | int | None] = {k: round(float(np.mean([r[k] for r in rows])), 5) for k in METRICS
                                          if k in rows[0]}
    out["median_first"] = float(np.median([r["first"] for r in rows]))
    out["n"] = len(rows)
    return out


def stored_batch(ix: ChannelIndex | TwoStageIndex, ids: list[int]) -> torch.Tensor:
    return torch.stack([ix.stored_query(t) for t in ids])


@torch.inference_mode()
def evaluate_indexes(indexes: dict[str, ChannelIndex | TwoStageIndex], queries: list[Query], keys: SongKeys,
                     batch: int = 32, log=print) -> dict:
    by_id: dict[int, list[Query]] = defaultdict(list)
    for q in queries:
        by_id[q.q].append(q)
    report: dict = {"channels": {}, "coverage": {}}
    firsts: dict[tuple[int, str], dict[str, int | None]] = defaultdict(dict)
    for name, ix in indexes.items():
        t0 = time.perf_counter()
        rowkey = keys.rows(ix.ids, ix.id_tensor.device)
        raw: dict[str, list] = defaultdict(list)
        songs: dict[str, list] = defaultdict(list)
        cov: dict[str, Counter] = defaultdict(Counter)
        for q in queries:
            if q.q not in ix.pos:
                cov[f"{q.bench}/{q.split}"]["missing_query"] += 1
        ids = sorted(t for t in by_id if t in ix.pos)
        for s in range(0, len(ids), batch):
            b = ids[s:s + batch]
            sc = ix.scores(stored_batch(ix, b))
            for j, t in enumerate(b):
                row = sc[j]
                row[ix.pos[t]] = -torch.inf
                qkey = keys.get(t)
                best = song_best(row, rowkey, qkey)
                for q in by_id[t]:
                    key = f"{q.bench}/{q.split}"
                    rr = ix.rank_of(row, q.pos, [t])
                    pkeys = {keys.get(p) for p in q.pos if p != t} - {qkey}
                    sr = song_ranks(best, pkeys)
                    if len(rr):
                        raw[key].append(metrics(rr))
                    else:
                        cov[key]["no_positive_raw"] += 1
                    if len(sr):
                        songs[key].append(metrics(sr))
                    elif not pkeys:
                        cov[key]["same_song_only"] += 1
                    else:
                        cov[key]["no_positive_songs"] += 1
                    firsts[(t, key)][name] = int(sr[0]) if len(sr) else None
            if (s // batch) % 20 == 0:
                log(f"{name}: {min(s + batch, len(ids))}/{len(ids)} queries")
        report["channels"][name] = {k: {"raw": aggregate(raw[k]), "songs": aggregate(songs[k])}
                                    for k in sorted(set(raw) | set(songs))}
        report["coverage"][name] = {k: dict(v) for k, v in sorted(cov.items())}
        report.setdefault("secs", {})[name] = round(time.perf_counter() - t0, 1)
    tabs: dict[str, list] = defaultdict(list)
    for (t, key), per in firsts.items():
        if len(per) != len(indexes):
            continue
        found = [r for r in per.values() if r is not None]
        if found:
            best = min(found)
            tabs[key].append({"rr": 1.0 / best, "hr@1": float(best <= 1), "hr@10": float(best <= 10),
                              "hr@20": float(best <= TAB_K), "first": float(best)})
    report["any_tab"] = {k: aggregate(v) for k, v in sorted(tabs.items())}
    return report


@torch.inference_mode()
def exact_scores(ix: ChannelIndex | TwoStageIndex, q: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
    """Brute-force scores of prepared queries [B, W, D] against every stored row (hub penalty included)."""
    if isinstance(ix, ChannelIndex):
        return ix.scores(q)
    out = torch.empty(len(q), len(ix), device=ix.device)
    for s in range(0, len(ix), chunk):
        df = torch.from_numpy(np.array(ix.full.mm[s:s + chunk])).to(ix.device).float()
        out[:, s:s + chunk] = score_block(q, df, ix.channel.mode)
    if ix.penalty is not None:
        out -= ix.penalty
    return out


@torch.inference_mode()
def exact_check(ix: ChannelIndex | TwoStageIndex, sample: list[int], keys: SongKeys, batch: int = 64) -> dict:
    """How often the served two-stage ranking matches brute force over the full windows."""
    rowkey = keys.rows(ix.ids, ix.id_tensor.device)
    rows: dict[str, list[float]] = defaultdict(list)
    for s in range(0, len(sample), batch):
        b = sample[s:s + batch]
        q = stored_batch(ix, b)
        staged, exact = ix.scores(q), exact_scores(ix, q)
        for j, t in enumerate(b):
            got, want = staged[j], exact[j]
            for k in (10, 50):
                a = {x for x, _ in ix.top(got, k, [t])}
                e = {x for x, _ in ix.top(want, k, [t])}
                rows[f"top{k}_overlap"].append(len(a & e) / max(len(e), 1))
            got[ix.pos[t]] = want[ix.pos[t]] = -torch.inf
            qk = keys.get(t)
            gs, es = song_top(song_best(got, rowkey, qk), 10), song_top(song_best(want, rowkey, qk), 10)
            rows["songs_top10_overlap"].append(len(set(gs) & set(es)) / max(len(es), 1))
            rows["songs_top10_identical"].append(float(gs == es))
    return {"n": len(sample), **{k: round(float(np.mean(v)), 5) for k, v in rows.items()},
            "min_top10_overlap": round(float(np.min(rows["top10_overlap"])), 3)}


@torch.inference_mode()
def hubness(ix: ChannelIndex | TwoStageIndex, sample: list[int], keys: SongKeys, k: int = 10, batch: int = 32,
            show: int = 15) -> dict:
    """k-occurrence of songs in random catalog queries' song-view top k (hubs crowd out real matches)."""
    rowkey = keys.rows(ix.ids, ix.id_tensor.device)
    n_songs = int(torch.unique(rowkey).numel())
    counts: Counter = Counter()
    for s in range(0, len(sample), batch):
        b = sample[s:s + batch]
        sc = ix.scores(stored_batch(ix, b))
        for j, t in enumerate(b):
            row = sc[j]
            row[ix.pos[t]] = -torch.inf
            counts.update(song_top(song_best(row, rowkey, keys.get(t)), k))
    occ = np.zeros(n_songs, np.float64)
    occ[:len(counts)] = sorted(counts.values(), reverse=True)
    slots = occ.sum()
    std = occ.std()
    top = counts.most_common(show)
    return {"queries": len(sample), "songs": n_songs, "k": k,
            "distinct_in_topk": len(counts), "coverage": round(len(counts) / n_songs, 4),
            "skewness": round(float(((occ - occ.mean()) ** 3).mean() / std ** 3), 2) if std else 0.0,
            "top100_share": round(float(occ[:100].sum() / slots), 4) if slots else 0.0,
            "hubs": [{"song": keys.names[key], "count": c, "share_of_queries": round(c / len(sample), 4)}
                     for key, c in top]}


@torch.inference_mode()
def examples(indexes: dict[str, ChannelIndex | TwoStageIndex], ids: list[int], keys: SongKeys, k: int = 10
             ) -> dict:
    out = {}
    for t in ids:
        entry = {"query": keys.names[keys.get(t)] if t in keys.of else f"#{t}"}
        for name, ix in indexes.items():
            if t not in ix.pos:
                continue
            rowkey = keys.rows(ix.ids, ix.id_tensor.device)
            row = ix.scores(ix.stored_query(t))
            row[ix.pos[t]] = -torch.inf
            best = song_best(row, rowkey, keys.get(t))
            entry[name] = [{"song": keys.names[s], "score": round(float(best[s]), 4)} for s in song_top(best, k)]
        out[str(t)] = entry
    return out


def headline(report: dict, baseline: dict | None = None) -> str:
    lines = []
    for name, benches in HEADLINE.items():
        src = report["any_tab"] if name == "any_tab" else report["channels"].get(name, {})
        for key in benches:
            m = src.get(key, {})
            m = m.get("songs", m) if name != "any_tab" else m
            if not m.get("n"):
                continue
            line = (f"{name:8s} {key:28s} n={m['n']:<5d} " + (f"AP={m['ap']:.3f} " if "ap" in m else "")
                    + f"HR@1={m['hr@1']:.3f} HR@10={m['hr@10']:.3f} HR@20={m['hr@20']:.3f} "
                    f"median_first={m['median_first']:g}")
            if baseline:
                b = baseline["any_tab"] if name == "any_tab" else baseline["channels"].get(name, {})
                bm = b.get(key, {})
                bm = bm.get("songs", bm) if name != "any_tab" else bm
                if bm.get("n"):
                    line += f"   | baseline HR@10={bm['hr@10']:.3f}" + (f" AP={bm['ap']:.3f}" if "ap" in bm else "")
            lines.append(line)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="eval_index")
    p.add_argument("--index", required=True)
    p.add_argument("--meta", type=Path, required=True, help="catalog_meta.npz (song identity for the API view)")
    p.add_argument("--bench", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--channels", nargs="+", default=list(CHANNELS), choices=list(CHANNELS))
    p.add_argument("--benches", nargs="*", default=None, help="benchmark names to keep (default all)")
    p.add_argument("--splits", nargs="+", default=["development", "test", "held_out", "validation"])
    p.add_argument("--max-per-bench", type=int, default=0, help="seeded sample per bench/split (0 = all)")
    p.add_argument("--exact", type=int, default=0, help="queries per channel for the brute-force check")
    p.add_argument("--hubs", type=int, default=0, help="random catalog queries per channel for hubness")
    p.add_argument("--examples", nargs="*", type=int, default=[], help="track IDs to list top songs for")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--candidates", type=int, default=None)
    p.add_argument("--hub-alpha", type=float, default=None,
                   help="CSLS hub penalty weight (default: as recorded by hub_build; 0 disables)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--baseline", type=Path, default=None, help="earlier eval_index report to compare against")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    rng = random.Random(args.seed)
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)  # noqa: E731

    keys = SongKeys(load_meta(args.meta))
    queries, _ = load_benchmarks(args.bench)
    queries = [q for q in queries if q.split in args.splits and (not args.benches or q.bench in args.benches)]
    if args.max_per_bench:
        groups: dict[tuple[str, str], list[Query]] = defaultdict(list)
        for q in queries:
            groups[(q.bench, q.split)].append(q)
        queries = [q for g in groups.values() for q in (rng.sample(g, args.max_per_bench)
                                                        if len(g) > args.max_per_bench else g)]
    eng = Engine(args.index, args.channels, args.device, candidates=args.candidates, hub_alpha=args.hub_alpha)
    report: dict = {"args": {k: str(v) for k, v in vars(args).items()},
                    "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "index_sizes": {k: len(v) for k, v in eng.indexes.items()},
                    "candidates": {k: getattr(v, "candidates", None) for k, v in eng.indexes.items()},
                    "hub_alpha": {k: getattr(v, "hub_alpha", None) for k, v in eng.indexes.items()},
                    "queries": len(queries)}
    log(f"index {report['index_sizes']} hub_alpha={report['hub_alpha']} queries={len(queries)}")
    report |= evaluate_indexes(eng.indexes, queries, keys, args.batch, log)
    bench_ids = sorted({q.q for q in queries})
    if args.exact:
        report["exact_check"] = {}
        for name, ix in eng.indexes.items():
            pool_b = [t for t in bench_ids if t in ix.pos]
            sample = rng.sample(pool_b, min(args.exact // 2, len(pool_b)))
            sample += rng.sample(ix.ids.tolist(), min(args.exact - len(sample), len(ix)))
            log(f"exact check {name} n={len(sample)}")
            report["exact_check"][name] = exact_check(ix, sample, keys)
    if args.hubs:
        report["hubness"] = {}
        for name, ix in eng.indexes.items():
            log(f"hubness {name}")
            report["hubness"][name] = hubness(ix, rng.sample(ix.ids.tolist(), min(args.hubs, len(ix))), keys)
    if args.examples:
        report["examples"] = examples(eng.indexes, args.examples, keys)
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    baseline = json.loads(args.baseline.read_text(encoding="utf-8")) if args.baseline else None
    print(headline(report, baseline), flush=True)
    for name, r in report.get("exact_check", {}).items():
        print(f"exact {name}: {r}", flush=True)
    for name, r in report.get("hubness", {}).items():
        print(f"hubs {name}: coverage={r['coverage']} skew={r['skewness']} top100_share={r['top100_share']} "
              f"top={r['hubs'][:3]}", flush=True)
    log(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
