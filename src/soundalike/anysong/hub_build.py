"""CSLS hub penalty for a built two-stage index.

Some tracks (near-silent intros, generic loops) sit close to everything and crowd real matches out of
every result list. CSLS ranks by ``2 s(q, t) - r(q) - r(t)``; for one query that is ``s - 0.5 r(t)``, so
serving subtracts ``alpha * h(t)`` with ``alpha = 0.5``. ``h(t)`` is the mean of track ``t``'s ``k`` best
*song* scores with its own song left out, so a song's other releases do not count as neighbours, and it
is computed from stage-1 scores, which is affordable at catalog scale. Per channel this writes ``hub.npy``
(one fp32 value per track) and records ``hub`` in ``meta.json``; ``TwoStageIndex.load`` applies it. A track
far from everything (e.g. a broken embedding) gets a very low ``h`` but stays out of results: its score to
any query is close to ``h`` itself, so ``s - 0.5 h`` remains about ``0.5 h``.

    python -m soundalike.anysong.hub_build --index index_full --meta catalog_meta.npz

Select ``alpha`` on development splits (``eval_index --hub-alpha``); 0.5 is plain CSLS. The per-channel
defaults (``DEFAULT_ALPHA``) leave melody off: a family of songs sharing one melody is a dense *genuine*
cluster, so its members have high ``h`` and the penalty pushes them down (development positives in the
top hub decile lose AP; see docs/ANYSONG_EVAL.md). Melody's ``hub.npy`` is still written, so
``--hub-alpha`` can switch it on for experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .eval_index import SongKeys, stored_batch
from .serve import CHANNELS, TwoStageIndex, load_meta

DEFAULT_ALPHA = {"melody": 0.0, "samples": 0.5, "sonic": 0.5}

@torch.inference_mode()
def hub_scores(ix: TwoStageIndex, keys: SongKeys, k: int = 10, batch: int = 128, log=None) -> np.ndarray:
    dev = ix.id_tensor.device
    ids = ix.ids.tolist()
    rowkey = keys.rows(ix.ids, dev)
    nsongs = int(rowkey.max()) + 1
    k = min(k, nsongs - 1)
    if k < 1:
        raise ValueError("hub scores need at least two songs")
    h = np.empty(len(ids), dtype=np.float32)
    t0 = time.perf_counter()
    for s in range(0, len(ids), batch):
        b = ids[s:s + batch]
        sc = ix.stage1(stored_batch(ix, b))
        best = torch.full((len(b), nsongs), -torch.inf, device=dev)
        best.scatter_reduce_(1, rowkey.expand(len(b), -1), sc, reduce="amax")
        best[torch.arange(len(b), device=dev), rowkey[s:s + len(b)]] = -torch.inf
        h[s:s + len(b)] = best.topk(k, dim=1).values.mean(1).cpu().numpy()
        if log and (s // batch) % 200 == 0:
            done = s + len(b)
            log(f"{ix.channel.name}: {done}/{len(ids)} "
                f"({done / max(time.perf_counter() - t0, 1e-9):.0f} tracks/s)")
    if not np.isfinite(h).all():
        raise ValueError(f"{ix.channel.name}: non-finite hub scores")
    return h


def build(index_dir: Path, channel: str, meta_path: Path, k: int = 10, alpha: float = 0.5, batch: int = 128,
          device: str = "cuda", log=None) -> dict:
    path = Path(index_dir) / channel
    ix = TwoStageIndex.load(path, CHANNELS[channel], device, hub_alpha=0)
    t0 = time.time()
    h = hub_scores(ix, SongKeys(load_meta(meta_path)), k, batch, log)
    with open(path / "hub.npy.tmp", "wb") as f:
        np.save(f, h)
    (path / "hub.npy.tmp").replace(path / "hub.npy")
    meta = json.loads((path / "meta.json").read_text())
    meta["hub"] = {"alpha": alpha, "k": k, "level": "song", "scores": "stage1",
                   "meta_sha256": hashlib.sha256(Path(meta_path).read_bytes()).hexdigest(),
                   "min": round(float(h.min()), 6), "median": round(float(np.median(h)), 6),
                   "max": round(float(h.max()), 6),
                   "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "secs": round(time.time() - t0, 1)}
    tmp = path / "meta.json.tmp"
    tmp.write_text(json.dumps(meta, indent=1))
    tmp.replace(path / "meta.json")
    return meta["hub"]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="hub_build")
    p.add_argument("--index", required=True, type=Path, help="index_build output directory")
    p.add_argument("--meta", required=True, type=Path, help="catalog_meta.npz (song identity, as the API)")
    p.add_argument("--channels", nargs="+", default=list(CHANNELS), choices=list(CHANNELS))
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--alpha", type=float, default=None,
                   help=f"serving weight recorded in meta.json (0 = off; default per channel {DEFAULT_ALPHA})")
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)  # noqa: E731
    for name in args.channels:
        log(f"{name}: hub scores")
        alpha = DEFAULT_ALPHA[name] if args.alpha is None else args.alpha
        hub = build(args.index, name, args.meta, args.k, alpha, args.batch, args.device, log)
        log(f"{name}: {json.dumps(hub)}")


if __name__ == "__main__":
    main()
