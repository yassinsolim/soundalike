"""Embed cached previews with one candidate model into resumable shards.

    python -m soundalike.anysong.embed_run --model muq_mulan --ids pool.txt \
        --previews C:/anysong/previews --out C:/anysong/emb
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypeVar

import numpy as np
from soundfile import LibsndfileError

from .audio import load_clip
from .embedders import REGISTRY

Decoded = TypeVar("Decoded")


def preview_path(root: Path, tid: int) -> Path:
    return root / f"{tid % 1000:03d}" / f"{tid}.mp3"


def load_store(model_dir: Path) -> dict[str, np.ndarray]:
    """Concatenate all shards for a model -> {'ids': [N], key: [N, ...]}."""
    shards = sorted(model_dir.glob("part-*.npz"))
    if not shards:
        return {}
    parts = [dict(np.load(s)) for s in shards]
    keys = parts[0].keys()
    out = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    _, first = np.unique(out["ids"], return_index=True)
    first.sort()
    return {k: v[first] for k, v in out.items()}


def done_ids(model_dir: Path) -> set[int]:
    ids: set[int] = set()
    for s in model_dir.glob("part-*.npz"):
        with np.load(s) as z:
            ids.update(int(x) for x in z["ids"])
    return ids


def decoded_batches(executor: ThreadPoolExecutor, decode: Callable[[int], Decoded],
                    ids: Sequence[int], batch_size: int) -> Iterator[list[Decoded]]:
    """Keep only the current and next batch scheduled, overlapping decode with inference."""
    starts = iter(range(0, len(ids), batch_size))
    first = next(starts, None)
    if first is None:
        return

    def submit(start):
        return [executor.submit(decode, tid) for tid in ids[start:start + batch_size]]

    pending = submit(first)
    for start in starts:
        current = [future.result() for future in pending]
        pending = submit(start)
        yield current
    yield [future.result() for future in pending]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="embed_run")
    p.add_argument("--model", required=True, choices=sorted(REGISTRY))
    p.add_argument("--ids", required=True, nargs="+")
    p.add_argument("--previews", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--shard", type=int, default=256)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args(argv)
    if min(args.batch, args.shard, args.workers) < 1 or args.limit < 0:
        p.error("batch, shard and workers must be positive; limit must be non-negative")

    root = Path(args.previews)
    model_dir = Path(args.out) / args.model
    model_dir.mkdir(parents=True, exist_ok=True)
    ids: list[int] = []
    for f in args.ids:
        ids += [int(x) for x in Path(f).read_text().split()]
    ids = list(dict.fromkeys(ids))
    have = done_ids(model_dir)
    todo = [i for i in ids if i not in have and preview_path(root, i).exists()]
    if args.limit:
        todo = todo[: args.limit]
    missing = sum(1 for i in ids if not preview_path(root, i).exists())
    print(f"{args.model}: {len(ids)} ids, {len(have)} done, {missing} without preview, {len(todo)} to embed",
          flush=True)
    if not todo:
        return
    emb = REGISTRY[args.model]()
    emb.load()
    sr = emb.sr

    failures: list[int] = []

    def decode(tid: int):
        try:
            return tid, load_clip(preview_path(root, tid), sr)
        except (OSError, LibsndfileError, ValueError) as exc:
            print(f"decode failed {tid}: {exc}", flush=True)
            return tid, None

    shard_idx = 1 + max((int(s.stem.split("-")[1]) for s in model_dir.glob("part-*.npz")), default=-1)
    buf: dict[str, list] = {"ids": []}
    t0, n_done = time.time(), 0
    shapes: dict[str, tuple[int, ...]] | None = None

    def flush() -> None:
        nonlocal shard_idx, buf
        if not buf["ids"]:
            return
        arrays = {k: (np.asarray(v, dtype=np.int64) if k == "ids" else np.concatenate(v)) for k, v in buf.items()}
        path = model_dir / f"part-{shard_idx:04d}.npz"
        temporary = path.with_suffix(".npz.tmp")
        try:
            with temporary.open("wb") as stream:
                np.savez(stream, **arrays)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        shard_idx += 1
        buf = {"ids": []}

    with ThreadPoolExecutor(args.workers) as ex:
        # Executor.map eagerly submits its entire input on Python <3.14.
        # Prefetch one batch, never the entire audio corpus.
        for decoded in decoded_batches(ex, decode, todo, args.batch):
            batch_ids, batch = [], []
            for tid, clip in decoded:
                if clip is None:
                    failures.append(tid)
                else:
                    batch_ids.append(tid)
                    batch.append(clip)
            if not batch:
                continue
            out = emb.embed(np.stack(batch))
            if not out or "ids" in out:
                raise ValueError(f"{args.model}: expected non-empty embedding arrays, without reserved key 'ids'")
            for k, v in out.items():
                if not isinstance(v, np.ndarray) or v.ndim < 2 or v.shape[0] != len(batch):
                    raise ValueError(f"{args.model}/{k}: invalid batch shape")
                if not v.size or not np.isfinite(v).all():
                    raise ValueError(f"{args.model}/{k}: empty or non-finite embeddings")
            current_shapes = {k: v.shape[1:] for k, v in out.items()}
            if shapes is not None and current_shapes != shapes:
                raise ValueError(f"{args.model}: embedding keys or shapes changed between batches")
            shapes = current_shapes
            buf["ids"].extend(batch_ids)
            for k, v in out.items():
                buf.setdefault(k, []).append(v)
            n_done += len(batch)
            del decoded, batch
            if len(buf["ids"]) >= args.shard:
                flush()
                rate = n_done / (time.time() - t0)
                print(f"{n_done}/{len(todo)} {rate:.1f} tracks/s eta {(len(todo) - n_done) / rate / 60:.1f} min",
                      flush=True)
    flush()
    if failures:
        raise RuntimeError(f"{len(failures)} previews failed decoding: {failures[:20]}; "
                           f"saved {n_done} successful embeddings; repair previews and resume")
    print(f"done {n_done} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    sys.exit(main())
