"""Check the any-song serving path on the serving GPU against the offline research embeddings.

parity  Re-embed indexed tracks from their cached previews and compare with the stored windows.
        Every track should retrieve itself first.
unseen  Embed each benchmark query live, remove it from the index and rank. The same ranking with the
        stored (offline) query vector is reported beside it, so a difference isolates the serving
        GPU/decoder rather than the model choice. This is a systems check, never model selection.

    python -m soundalike.anysong.serve_check --index /opt/soundalike/index/research \
        --previews /data/previews --bench /data/bench --out /data/logs/serve_check.json
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from .embed_run import preview_path
from .evaluate import load_benchmarks
from .serve import CHANNELS, Engine


def gpu_used_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10, check=True)
        return int(out.stdout.split()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def summarize(values: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    return {k: {"n": len(v), "p50": round(float(np.percentile(v, 50)), 2),
                "p95": round(float(np.percentile(v, 95)), 2), "mean": round(float(np.mean(v)), 2),
                "max": round(float(np.max(v)), 2)} for k, v in sorted(values.items()) if v}


def rank_metrics(ranks: np.ndarray) -> dict[str, float]:
    n = len(ranks)
    return {"ap": float(np.mean(np.arange(1, n + 1) / ranks)), "rr": 1.0 / float(ranks[0]),
            "hr@1": float(ranks[0] <= 1), "hr@10": float(ranks[0] <= 10), "p@10": float((ranks <= 10).sum()) / 10}


def mean_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    return {k: round(float(np.mean([r[k] for r in rows])), 5) for k in rows[0]} if rows else {}


def window_agreement(live: torch.Tensor, stored: torch.Tensor) -> torch.Tensor:
    """Per-window cosine between two [W, D] window sets."""
    return torch.nn.functional.cosine_similarity(live, stored, dim=-1)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="serve_check")
    p.add_argument("--index", required=True)
    p.add_argument("--previews", required=True)
    p.add_argument("--bench", required=True)
    p.add_argument("--channels", nargs="+", default=list(CHANNELS), choices=list(CHANNELS))
    p.add_argument("--parity", type=int, default=64, help="indexed tracks to re-embed for parity")
    p.add_argument("--benches", nargs="*", default=None, help="benchmark names to keep (default all)")
    p.add_argument("--splits", nargs="+", default=["development", "test", "held_out", "validation"])
    p.add_argument("--max-per-bench", type=int, default=0, help="seeded sample per bench/split (0 = all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=3, help="CPU decode prefetch threads")
    p.add_argument("--candidates", type=int, default=None,
                   help="exact re-rank depth for built indexes (default: per-channel value from meta.json)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--half-stage1", nargs="*", default=[], choices=list(CHANNELS),
                   help="channels whose quantized stage 1 runs as an fp16 GEMM, as served with api --half-stage1")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    previews = Path(args.previews)
    rng = random.Random(args.seed)
    report: dict = {"args": vars(args), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z")}

    report["gpu_used_mib_before"] = gpu_used_mib()
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    eng = Engine(args.index, args.channels, args.device, candidates=args.candidates, half_stage1=args.half_stage1)
    eng.warm()
    report["load_secs"] = eng.load_secs
    report["index_sizes"] = {k: len(v) for k, v in eng.indexes.items()}
    report["gpu_used_mib_loaded"] = gpu_used_mib()
    if args.device.startswith("cuda"):
        report["torch_mib_loaded"] = round(torch.cuda.memory_allocated() / 2**20)
        report["torch_reserved_mib_loaded"] = round(torch.cuda.memory_reserved() / 2**20)
    print(json.dumps({k: report[k] for k in report if k != "args"}), flush=True)

    timings: dict[str, list[float]] = defaultdict(list)
    failures: list[dict] = []
    pool = ThreadPoolExecutor(max(1, args.workers))

    def safe_decode(path: Path):
        try:
            return eng.decode(path)
        except Exception as exc:  # noqa: BLE001 - recorded and reported, never silently dropped
            return exc

    def stream(ids: list[int]):
        """Yield (id, clips, decode timings) in order with bounded decode prefetch."""
        window = max(1, args.workers) * 2
        futs = [pool.submit(safe_decode, preview_path(previews, t)) for t in ids[:window]]
        for i, t in enumerate(ids):
            if i + window < len(ids):
                futs.append(pool.submit(safe_decode, preview_path(previews, ids[i + window])))
            res = futs[i].result()
            futs[i] = None
            if isinstance(res, Exception):
                failures.append({"id": t, "error": f"{type(res).__name__}: {res}"})
                continue
            clips, td = res
            yield t, clips, td

    # ---------------------------------------------------------------- parity
    common = set.intersection(*(set(map(int, ix.ids)) for ix in eng.indexes.values()))
    pool_ids = sorted(t for t in common if preview_path(previews, t).exists())
    parity_ids = rng.sample(pool_ids, min(args.parity, len(pool_ids)))
    par: dict[str, dict[str, list[float]]] = {c: defaultdict(list) for c in eng.indexes}
    for tid, clips, td in stream(parity_ids):
        wins, te = eng.embed_clips(clips)
        for name, ix in eng.indexes.items():
            live, stored = ix.prep_query(wins[name]), ix.stored_query(tid)
            cos = window_agreement(live, stored)
            sc = ix.scores(live)
            ranks = ix.rank_of(sc, [tid])
            par[name]["win_cos_mean"].append(float(cos.mean()))
            par[name]["win_cos_min"].append(float(cos.min()))
            par[name]["self_rank"].append(int(ranks[0]))
            par[name]["self_score_delta"].append(float(sc[ix.pos[tid]] - ix.scores(stored)[ix.pos[tid]]))
    report["parity"] = {name: {
        "n": len(v["self_rank"]),
        "win_cos_mean": round(float(np.mean(v["win_cos_mean"])), 6),
        "win_cos_min": round(float(np.min(v["win_cos_min"])), 6),
        "self_at_1": round(float(np.mean(np.array(v["self_rank"]) == 1)), 4),
        "self_rank_max": int(np.max(v["self_rank"])),
        "self_score_delta_abs_mean": round(float(np.mean(np.abs(v["self_score_delta"]))), 6),
    } for name, v in par.items() if v["self_rank"]}
    print("parity", json.dumps(report["parity"]), flush=True)

    # ---------------------------------------------------------------- unseen (leave-one-out) queries
    queries, _ = load_benchmarks(Path(args.bench))
    queries = [q for q in queries if q.split in args.splits and (not args.benches or q.bench in args.benches)]
    if args.max_per_bench:
        groups: dict[tuple[str, str], list] = defaultdict(list)
        for q in queries:
            groups[(q.bench, q.split)].append(q)
        queries = [q for g in groups.values() for q in (rng.sample(g, args.max_per_bench)
                                                        if len(g) > args.max_per_bench else g)]
    by_id: dict[int, list] = defaultdict(list)
    for q in queries:
        by_id[q.q].append(q)
    ids = sorted(t for t in by_id if preview_path(previews, t).exists())
    report["unseen_queries"] = {"queries": len(queries), "unique_ids": len(by_id), "with_preview": len(ids)}
    print("unseen", json.dumps(report["unseen_queries"]), flush=True)

    live_rows: dict[tuple[str, str], list] = defaultdict(list)
    off_rows: dict[tuple[str, str], list] = defaultdict(list)
    overlap: dict[tuple[str, str], list] = defaultdict(list)
    skipped: dict[tuple[str, str], int] = defaultdict(int)
    peak_gpu = report["gpu_used_mib_loaded"] or 0
    t0 = time.perf_counter()
    for n, (tid, clips, td) in enumerate(stream(ids), 1):
        wins, te = eng.embed_clips(clips)
        for k, v in (td | te).items():
            timings[k].append(v)
        _, ts = eng.search_windows(wins, 10, [tid])
        for k, v in ts.items():
            timings[k].append(v)
        timings["total_serial"].append(sum(td.values()) + sum(te.values()) + sum(ts.values()))
        for name, ix in eng.indexes.items():
            if tid not in ix.pos:
                for q in by_id[tid]:
                    skipped[(name, f"{q.bench}/{q.split}")] += 1
                continue
            sc = ix.scores(torch.stack([ix.prep_query(wins[name]), ix.stored_query(tid)]))
            top_live = {t for t, _ in ix.top(sc[0], 10, [tid])}
            top_off = {t for t, _ in ix.top(sc[1], 10, [tid])}
            for q in by_id[tid]:
                key = (name, f"{q.bench}/{q.split}")
                rl, ro = ix.rank_of(sc[0], q.pos, [tid]), ix.rank_of(sc[1], q.pos, [tid])
                if not len(rl):
                    skipped[key] += 1
                    continue
                live_rows[key].append(rank_metrics(rl))
                off_rows[key].append(rank_metrics(ro))
                overlap[key].append(len(top_live & top_off) / 10)
        if n % 100 == 0 or n == len(ids):
            used = gpu_used_mib()
            peak_gpu = max(peak_gpu, used or 0)
            rate = n / (time.perf_counter() - t0)
            print(f"{n}/{len(ids)} {rate:.2f} q/s eta {(len(ids) - n) / rate / 60:.1f} min gpu_used={used}MiB",
                  flush=True)
    pool.shutdown()
    report["unseen"] = {f"{name}|{key}": {"n": len(live_rows[(name, key)]),
                                         "skipped": skipped.get((name, key), 0),
                                         "live": mean_rows(live_rows[(name, key)]),
                                         "offline": mean_rows(off_rows[(name, key)]),
                                         "top10_overlap": round(float(np.mean(overlap[(name, key)])), 4)}
                        for name, key in sorted(live_rows)}
    report["latency_ms"] = summarize(timings)
    report["decode_failures"] = failures
    report["gpu_used_mib_peak_sampled"] = peak_gpu
    if args.device.startswith("cuda"):
        report["torch_peak_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
        report["torch_reserved_mib_end"] = round(torch.cuda.memory_reserved() / 2**20)
    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    sys.exit(main())
