"""Any-song query engine: embed one unseen clip on the GPU and rank it against reference embeddings.

The query never has to be in the index. Channels and their scoring were selected on development
benchmarks only (see ``evaluate``):

- ``melody``  CLEWS DVI, 3x20 s shingles, normalised squared-Euclidean best pairs (covers, same melody)
- ``samples`` SampleID, 8x6.125 s chunks, centred cosine MaxSim (shared samples)
- ``sonic``   MuQ-MuLan, 3x10 s windows, cosine MaxSim (general sound-alike; weakest evidence)

Optional channel (``ALL_CHANNELS``, needs a built index, see ``sounds``):

- ``sounds``  the MuQ-MuLan windows through a small projection head trained on related-artist pairs, cosine
  MaxSim (the "Sounds like" list)

    python -m soundalike.anysong.serve --index /opt/soundalike/index/research --audio song.mp3 \
        --meta /data/catalog_meta.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .audio import load_clips
from .embedders import REGISTRY, Embedder
from .evaluate import l2, load_keys, score_block, window_scores


@dataclass(frozen=True)
class Channel:
    name: str
    model: str
    prep: str  # none | raw | center, as in evaluate.prep_features; head = raw, then a projection head
    mode: str  # window scoring mode understood by evaluate.score_block


CHANNELS: dict[str, Channel] = {c.name: c for c in (
    Channel("melody", "clews_dvi", "none", "bpwr"),
    Channel("samples", "sampleid", "center", "maxsim"),
    Channel("sonic", "muq_mulan", "raw", "maxsim"),
)}
# Channels a server can add by name; they only exist as built indexes (``<index>/<name>/meta.json``).
ALL_CHANNELS: dict[str, Channel] = CHANNELS | {"sounds": Channel("sounds", "muq_mulan", "head", "maxsim")}


class Head:
    """Residual projection ``x + W2 gelu(W1 x + b1) + b2`` of prepared windows (the ``head`` prep).

    Weights load from ``head.npz`` (no pickles); ``sounds`` converts the trained head and documents its training.
    """

    KEYS = ("inp_w", "inp_b", "out_w", "out_b")

    def __init__(self, inp_w: torch.Tensor, inp_b: torch.Tensor, out_w: torch.Tensor, out_b: torch.Tensor):
        hidden, dim = inp_w.shape
        if tuple(inp_b.shape) != (hidden,) or tuple(out_w.shape) != (dim, hidden) or tuple(out_b.shape) != (dim,):
            raise ValueError("inconsistent projection head weights")
        self.inp_w, self.inp_b, self.out_w, self.out_b = (t.float() for t in (inp_w, inp_b, out_w, out_b))
        self.dim = dim

    @classmethod
    def load(cls, path: Path | str, device: str | torch.device = "cpu") -> Head:
        with np.load(path, allow_pickle=False) as z:
            return cls(*(torch.from_numpy(z[k].astype(np.float32)).to(device) for k in cls.KEYS))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        return x + torch.nn.functional.gelu(x @ self.inp_w.T + self.inp_b) @ self.out_w.T + self.out_b


def prep_windows(x: torch.Tensor, prep: str, mu: torch.Tensor | None = None, head: Head | None = None
                 ) -> torch.Tensor:
    if prep == "none":
        return x
    if prep == "raw":
        return l2(x)
    if prep == "center":
        if mu is None:
            raise ValueError("center prep needs the index mean")
        return l2(x - mu)
    if prep == "head":
        if head is None:
            raise ValueError("head prep needs the projection head")
        return l2(head(l2(x)))
    raise ValueError(prep)


class _Ranked:
    """Track-id bookkeeping plus deterministic top-k / rank queries over a score row."""

    channel: Channel

    def _set_ids(self, ids: np.ndarray, device: torch.device | str) -> None:
        ids = np.asarray(ids, dtype=np.int64)
        if len(np.unique(ids)) != len(ids):
            raise ValueError("index ids must be unique")
        self.ids = ids
        self.pos = {int(t): i for i, t in enumerate(ids)}
        self.id_tensor = torch.as_tensor(ids, device=device)

    def __len__(self) -> int:
        return len(self.ids)

    def top(self, scores: torch.Tensor, k: int, exclude: Iterable[int] = ()) -> list[tuple[int, float]]:
        """Best ``k`` (track_id, score), ties broken by track ID; ``exclude`` IDs are never returned."""
        row = scores.clone()
        drop = sorted({self.pos[int(t)] for t in exclude if int(t) in self.pos})
        if drop:
            row[torch.tensor(drop, device=row.device)] = -torch.inf
        k = min(k, len(self) - len(drop))
        if k <= 0:
            return []
        vals, idx = row.topk(k)
        # Widen to every candidate tied with the k-th score so ID tie-breaking is order independent.
        cand = torch.nonzero(row >= vals[-1]).flatten()
        order = sorted(zip((-row[cand]).tolist(), self.ids[cand.cpu().numpy()].tolist()))[:k]
        return [(int(t), -float(s)) for s, t in order]

    def rank_of(self, scores: torch.Tensor, positives: Iterable[int], exclude: Iterable[int] = ()) -> np.ndarray:
        """Sorted 1-based ranks of the positives present in the index (``evaluate`` tie rules)."""
        excluded = {int(t) for t in exclude}
        row = scores.clone()
        drop = sorted(self.pos[t] for t in excluded if t in self.pos)
        if drop:
            row[torch.tensor(drop, device=row.device)] = -torch.inf
        pi = torch.tensor(sorted({self.pos[int(p)] for p in positives if int(p) in self.pos and int(p) not in excluded}),
                          device=row.device, dtype=torch.long)
        if not len(pi):
            return np.array([], dtype=np.int64)
        pv = row[pi]
        ahead = (row[None, :] > pv[:, None]) | (
            (row[None, :] == pv[:, None]) & (self.id_tensor[None, :] < self.id_tensor[pi, None]))
        return (ahead.sum(1) + 1).sort().values.cpu().numpy()


class ChannelIndex(_Ranked):
    """Prepared window embeddings of one channel, stored fp16 and scored exactly in fp32 chunks."""

    def __init__(self, channel: Channel, ids: np.ndarray, feats: torch.Tensor, mu: torch.Tensor | None,
                 head: Head | None = None):
        if feats.dim() != 3 or len(ids) != len(feats):
            raise ValueError("expected [N, W, D] window features matching ids")
        if channel.prep == "head" and head is None:
            raise ValueError("head prep needs the projection head")
        self.channel, self.feats, self.mu, self.head = channel, feats, mu, head
        self._set_ids(ids, feats.device)

    @classmethod
    def from_windows(cls, channel: Channel, ids: np.ndarray, win: np.ndarray, device: str = "cpu",
                     chunk: int = 4096, head: Head | None = None) -> ChannelIndex:
        mu = None
        if channel.prep == "center":
            total = np.zeros(win.shape[-1], np.float64)
            for s in range(0, len(win), chunk):
                total += win[s:s + chunk].astype(np.float32).reshape(-1, win.shape[-1]).sum(0, dtype=np.float64)
            mu = torch.from_numpy((total / (len(win) * win.shape[1])).astype(np.float32)).to(device)
        feats = torch.empty(win.shape, dtype=torch.float16, device=device)
        for s in range(0, len(win), chunk):
            x = torch.from_numpy(np.ascontiguousarray(win[s:s + chunk])).to(device).float()
            feats[s:s + chunk] = prep_windows(x, channel.prep, mu, head).half()
        return cls(channel, ids, feats, mu, head)

    @classmethod
    def load(cls, emb_dir: Path, channel: Channel, device: str = "cpu") -> ChannelIndex:
        if channel.prep == "head":
            raise ValueError(f"the {channel.name} channel needs a built index (<index>/{channel.name}/meta.json)")
        st = load_keys(Path(emb_dir) / channel.model, ["win"])
        return cls.from_windows(channel, st["ids"], st["win"], device)

    def prep_query(self, win: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Raw query windows [W, D] or [B, W, D] -> prepared fp32 tensor on the index device."""
        x = torch.as_tensor(win).to(self.feats.device).float()
        return prep_windows(x, self.channel.prep, self.mu, self.head)

    def stored_query(self, track_id: int) -> torch.Tensor:
        return self.feats[self.pos[int(track_id)]].float()

    @torch.inference_mode()
    def scores(self, q: torch.Tensor, chunk: int = 2048) -> torch.Tensor:
        """Prepared queries [W, D] or [B, W, D] -> similarities [N] or [B, N] (higher is closer)."""
        single = q.dim() == 2
        q = q[None] if single else q
        out = torch.empty(len(q), len(self), device=self.feats.device)
        for s in range(0, len(self), chunk):
            out[:, s:s + chunk] = score_block(q, self.feats[s:s + chunk].float(), self.channel.mode)
        return out[0] if single else out


def _to_device(arr: np.ndarray, device: str, rows: int = 16384) -> torch.Tensor:
    """Copy a (possibly memory-mapped) array to ``device`` in row blocks to bound host memory."""
    out = torch.empty(arr.shape, dtype=getattr(torch, arr.dtype.name), device=device)
    for s in range(0, len(arr), rows):
        out[s:s + rows] = torch.from_numpy(np.array(arr[s:s + rows]))
    return out


class RowFile:
    """Random row reads from a C-order ``.npy`` array.

    Where ``os.preadv`` exists, rows are read by a thread pool: page faults on a memory map are served
    one at a time, and ZFS caches mmap pages twice. Elsewhere this falls back to a read-only memory map.
    """

    def __init__(self, path: Path | str, threads: int = 8):
        self.mm = np.load(path, mmap_mode="r")
        self.shape, self.dtype = self.mm.shape, self.mm.dtype
        self.row_bytes = int(np.prod(self.shape[1:], dtype=np.int64)) * self.dtype.itemsize
        self.threads = threads
        self.fd = os.open(path, os.O_RDONLY) if hasattr(os, "preadv") and threads > 1 else None
        if self.fd is not None:
            self.pool = ThreadPoolExecutor(threads, thread_name_prefix="rowfile")

    def __len__(self) -> int:
        return self.shape[0]

    def _read(self, rows: np.ndarray, flat: np.ndarray, part: np.ndarray) -> None:
        for i in part.tolist():
            buf, off, got = memoryview(flat[i]), self.mm.offset + int(rows[i]) * self.row_bytes, 0
            while got < self.row_bytes:
                n = os.preadv(self.fd, [buf[got:]], off + got)
                if n <= 0:
                    raise OSError(f"short read at row {int(rows[i])}")
                got += n

    def __getitem__(self, rows: np.ndarray) -> np.ndarray:
        rows = np.asarray(rows, dtype=np.int64)
        if self.fd is None or len(rows) < 2:
            return np.array(self.mm[rows])
        out = np.empty((len(rows), *self.shape[1:]), self.dtype)
        flat = out.reshape(len(rows), -1).view(np.uint8)
        parts = np.array_split(np.arange(len(rows)), min(self.threads, len(rows)))
        for f in [self.pool.submit(self._read, rows, flat, p) for p in parts]:
            f.result()
        return out


class TwoStageIndex(_Ranked):
    """Catalog-scale channel index written by ``index_build``.

    Stage 1 scores every track from low-dimensional window projections held on the device (distance
    modes use exact squared norms, so only the dot product is approximated). Stage 2 re-scores the best
    ``candidates`` exactly from memory-mapped full windows. Tracks outside the candidate set keep their
    stage-1 estimate shifted below every exact score, so ranks past ``candidates`` are approximate.

    With a hub penalty (``hub_build``), every score is lowered by ``alpha * h(track)`` before candidates are
    picked, where ``h`` is the track's mean top-k song similarity: CSLS, which demotes tracks close to everything.
    """

    def __init__(self, channel: Channel, ids: np.ndarray, proj: torch.Tensor, basis: torch.Tensor,
                 norms: torch.Tensor, full: np.ndarray, mu: torch.Tensor | None, candidates: int = 1000,
                 hub: torch.Tensor | None = None, hub_alpha: float = 0.0, head: Head | None = None):
        n, w = len(ids), full.shape[1]
        if proj.shape[:2] != (n, w) or tuple(norms.shape) != (n, w) or len(full) != n \
                or basis.shape != (full.shape[2], proj.shape[2]):
            raise ValueError("inconsistent two-stage index arrays")
        if channel.prep == "center" and mu is None:
            raise ValueError("center prep needs the index mean")
        if channel.prep == "head" and (head is None or head.dim != full.shape[2]):
            raise ValueError("head prep needs a projection head of the window dimension")
        if candidates < 1:
            raise ValueError("candidates must be positive")
        if hub_alpha and (hub is None or tuple(hub.shape) != (n,)):
            raise ValueError("a hub penalty needs one value per track")
        self.channel, self.proj, self.basis, self.norms, self.full, self.mu = channel, proj, basis, norms, full, mu
        self.head = head
        self.candidates, self.device = candidates, proj.device
        self.hub_alpha = float(hub_alpha)
        self.penalty = (self.hub_alpha * hub.to(proj.device).float()) if self.hub_alpha else None
        self._set_ids(ids, proj.device)

    @classmethod
    def load(cls, path: Path, channel: Channel, device: str = "cpu", candidates: int | None = None,
             read_threads: int = 8, hub_alpha: float | None = None) -> TwoStageIndex:
        """``candidates`` defaults to the depth recorded at build time and ``hub_alpha`` to the alpha recorded
        by ``hub_build`` (0 when absent; pass 0 to disable the penalty)."""
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        built = (meta["model"], meta["prep"], meta["mode"])
        if built != (channel.model, channel.prep, channel.mode):
            raise ValueError(f"{path} was built for {built}, not {channel}")
        mu = torch.from_numpy(np.load(path / "mu.npy")).to(device) if (path / "mu.npy").exists() else None
        alpha = float(meta.get("hub", {}).get("alpha", 0.0) if hub_alpha is None else hub_alpha)
        if alpha and not (path / "hub.npy").exists():
            raise FileNotFoundError(f"{path / 'hub.npy'} is missing; run hub_build or use hub_alpha=0")
        hub = torch.from_numpy(np.load(path / "hub.npy").astype(np.float32)) if alpha else None
        head = Head.load(path / "head.npz", device) if channel.prep == "head" else None
        return cls(channel, np.load(path / "ids.npy"), _to_device(np.load(path / "proj.npy", mmap_mode="r"), device),
                   torch.from_numpy(np.load(path / "basis.npy")).to(device),
                   torch.from_numpy(np.load(path / "norms.npy")).to(device),
                   RowFile(path / "full.npy", read_threads), mu, candidates or meta.get("candidates", 1000),
                   hub, alpha, head)

    def prep_query(self, win: np.ndarray | torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(win).to(self.device).float()
        return prep_windows(x, self.channel.prep, self.mu, self.head)

    def _full(self, rows: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.asarray(self.full[rows])).to(self.device).float()

    def stored_query(self, track_id: int) -> torch.Tensor:
        return self._full(np.array([self.pos[int(track_id)]]))[0]

    @torch.inference_mode()
    def stage1(self, q: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
        """Prepared queries [B, W, D] -> approximate scores [B, N]."""
        qp, qn, dim = q @ self.basis, (q ** 2).sum(-1), q.shape[-1]
        out = torch.empty(len(q), len(self), device=self.device)
        for s in range(0, len(self), chunk):
            sims = torch.einsum("bwd,nvd->bnwv", qp, self.proj[s:s + chunk].float())
            out[:, s:s + chunk] = window_scores(sims, self.channel.mode, qn, self.norms[s:s + chunk], dim)
        return out

    @torch.inference_mode()
    def scores(self, q: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
        """Prepared queries [W, D] or [B, W, D] -> scores [N] or [B, N]; the top ``candidates`` are exact."""
        single = q.dim() == 2
        q = q[None] if single else q
        out = self.stage1(q, chunk)
        if self.penalty is not None:
            out -= self.penalty
        c = min(self.candidates, len(self))
        for b, row in enumerate(out):
            # Sorted rows keep memory-mapped reads sequential.
            cand = row.topk(c).indices.sort().values
            rows = cand.cpu().numpy()
            exact = torch.cat([score_block(q[b:b + 1], self._full(rows[s:s + 2048]), self.channel.mode)[0]
                               for s in range(0, c, 2048)])
            if self.penalty is not None:
                exact -= self.penalty[cand]
            row -= (row.max() - exact.min()).clamp_min(0) + 1
            row[cand] = exact
        return out[0] if single else out


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


class Engine:
    """Lazy model loading plus per-channel indexes. Not thread-safe: serialise calls per process.

    ``index_dir/<channel>/meta.json`` (``index_build`` output) loads a ``TwoStageIndex``; otherwise the
    channel is brute-forced from ``index_dir/<model>/part-*.npz`` shards.
    """

    def __init__(self, index_dir: Path | str, channels: Iterable[str] = tuple(CHANNELS), device: str = "cuda",
                 index_device: str | None = None, drop_text_tower: bool = True, candidates: int | None = None,
                 hub_alpha: float | None = None):
        self.device = device
        self.drop_text_tower = drop_text_tower
        self.channels = [ALL_CHANNELS[c] for c in channels]
        self.load_secs: dict[str, float] = {}
        self.indexes: dict[str, ChannelIndex | TwoStageIndex] = {}
        for ch in self.channels:
            t = time.perf_counter()
            built = Path(index_dir) / ch.name
            if (built / "meta.json").exists():
                self.indexes[ch.name] = TwoStageIndex.load(built, ch, index_device or device, candidates,
                                                           hub_alpha=hub_alpha)
            else:
                self.indexes[ch.name] = ChannelIndex.load(Path(index_dir), ch, index_device or device)
            self.load_secs[f"index:{ch.name}"] = round(time.perf_counter() - t, 2)
        self.embedders: dict[str, Embedder] = {}

    def embedder(self, model: str) -> Embedder:
        if model not in self.embedders:
            t = time.perf_counter()
            emb = REGISTRY[model](self.device)
            emb.load()
            if model == "muq_mulan" and self.drop_text_tower:
                # Audio queries never use the multilingual text encoder (~280M params).
                emb.model.mulan_module.text = None
                if self.device.startswith("cuda"):
                    torch.cuda.empty_cache()
            self.embedders[model] = emb
            self.load_secs[f"model:{model}"] = round(time.perf_counter() - t, 2)
        return self.embedders[model]

    def warm(self) -> None:
        for ch in self.channels:
            self.embedder(ch.model)

    def decode(self, path: str | Path) -> tuple[dict[int, np.ndarray], dict[str, float]]:
        """Decode once for every sample rate the channels need (CPU only; safe in worker threads)."""
        t = time.perf_counter()
        clips = load_clips(path, sorted({REGISTRY[ch.model].sr for ch in self.channels}))
        return clips, {"decode": (time.perf_counter() - t) * 1000}

    def embed_clips(self, clips: dict[int, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        """Raw window embeddings per channel for decoded clips, plus GPU timings in ms. Channels that share a model
        (``sonic`` and ``sounds``) share one embedding."""
        timings: dict[str, float] = {}
        out: dict[str, np.ndarray] = {}
        by_model: dict[str, np.ndarray] = {}
        for ch in self.channels:
            if ch.model in by_model:
                out[ch.name] = by_model[ch.model]
                continue
            emb = self.embedder(ch.model)
            _sync(self.device)
            t = time.perf_counter()
            win = emb.embed(clips[emb.sr][None])["win"][0]
            _sync(self.device)
            timings[f"embed_{ch.model}"] = (time.perf_counter() - t) * 1000
            if not np.isfinite(win.astype(np.float32)).all():
                raise ValueError(f"{ch.model} produced non-finite embeddings")
            out[ch.name] = by_model[ch.model] = win
        return out, timings

    def embed(self, path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        clips, t_decode = self.decode(path)
        wins, t_embed = self.embed_clips(clips)
        return wins, t_decode | t_embed

    def search_windows(self, wins: dict[str, np.ndarray], k: int | Mapping[str, int] = 20,
                       exclude: Iterable[int] = ()) -> tuple[dict[str, list[tuple[int, float]]], dict[str, float]]:
        """``k`` is one depth for every channel or a depth per channel name."""
        results, timings = {}, {}
        exclude = list(exclude)
        for ch in self.channels:
            idx = self.indexes[ch.name]
            _sync(self.device)
            t = time.perf_counter()
            depth = k[ch.name] if isinstance(k, Mapping) else k
            results[ch.name] = idx.top(idx.scores(idx.prep_query(wins[ch.name])), depth, exclude)
            _sync(self.device)
            timings[f"search_{ch.name}"] = (time.perf_counter() - t) * 1000
        return results, timings

    def search(self, path: str | Path, k: int = 20, exclude: Iterable[int] = ()) -> dict:
        wins, timings = self.embed(path)
        results, t_search = self.search_windows(wins, k, exclude)
        timings |= t_search
        return {"results": results, "timings_ms": {k_: round(v, 2) for k_, v in timings.items()}}

    def indexed(self, track_id: int) -> bool:
        return all(int(track_id) in ix.pos for ix in self.indexes.values())

    def search_stored(self, track_id: int, k: int | Mapping[str, int] = 20, exclude: Iterable[int] = ()
                      ) -> tuple[dict[str, list[tuple[int, float]]], dict[str, float]]:
        """Rank an indexed track from its stored fp16 windows instead of re-embedding its preview (live
        embeddings match the stored ones up to fp16 rounding; see ``serve_check``). The track itself is
        excluded. ``k`` is one depth for every channel or a depth per channel name."""
        results, timings = {}, {}
        exclude = [int(track_id), *exclude]
        for ch in self.channels:
            idx = self.indexes[ch.name]
            _sync(self.device)
            t = time.perf_counter()
            depth = k[ch.name] if isinstance(k, Mapping) else k
            results[ch.name] = idx.top(idx.scores(idx.stored_query(track_id)), depth, exclude)
            _sync(self.device)
            timings[f"search_{ch.name}"] = (time.perf_counter() - t) * 1000
        return results, timings


def load_meta(path: Path | None) -> dict[int, tuple[str, str]]:
    if path is None:
        return {}
    z = np.load(path, allow_pickle=True)
    return {int(t): (str(a), str(n)) for t, a, n in zip(z["track_ids"], z["artists"], z["titles"])}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="serve")
    p.add_argument("--index", required=True,
                   help="index_build output (<channel>/meta.json) or a directory with <model>/part-*.npz shards")
    p.add_argument("--audio", nargs="+", required=True)
    p.add_argument("--channels", nargs="+", default=list(CHANNELS), choices=list(ALL_CHANNELS))
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--candidates", type=int, default=None,
                   help="exact re-rank depth for built indexes (default: per-channel value from meta.json)")
    p.add_argument("--exclude", nargs="*", type=int, default=[], help="track IDs to leave out of results")
    p.add_argument("--meta", type=Path, default=None, help="catalog_meta.npz for titles")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)
    meta = load_meta(args.meta)
    eng = Engine(args.index, args.channels, args.device, candidates=args.candidates)
    eng.warm()
    for audio in args.audio:
        r = eng.search(audio, args.k, args.exclude)
        if args.json:
            print(json.dumps({"audio": audio, **r}))
            continue
        print(f"\n{audio}  {r['timings_ms']}")
        for name, rows in r["results"].items():
            print(f"  [{name}]")
            for i, (tid, s) in enumerate(rows, 1):
                artist, title = meta.get(tid, ("?", "?"))
                print(f"    {i:2d}. {s:+.4f}  {tid:<11d} {artist} - {title}")


if __name__ == "__main__":
    sys.exit(main())
