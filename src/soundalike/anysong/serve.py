"""Any-song query engine: embed one unseen clip on the GPU and rank it against reference embeddings.

The query never has to be in the index. Channels and their scoring were selected on development
benchmarks only (see ``evaluate``):

- ``melody``  CLEWS DVI, 3x20 s shingles, normalised squared-Euclidean best pairs (covers, same melody)
- ``samples`` SampleID, 8x6.125 s chunks, centred cosine MaxSim (shared samples)
- ``sonic``   MuQ-MuLan, 3x10 s windows, cosine MaxSim (general sound-alike; weakest evidence). An index built
  by ``index_whiten`` scores it in the PCA-whitened space instead (prep ``whiten``): the index stores the
  transform and ``prep_query`` applies it to live windows, so callers pass raw windows either way.

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
from dataclasses import dataclass, replace
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
    prep: str  # none | raw | center as in evaluate.prep_features; whiten: see prep_windows
    mode: str  # window scoring mode understood by evaluate.score_block


CHANNELS: dict[str, Channel] = {c.name: c for c in (
    Channel("melody", "clews_dvi", "none", "bpwr"),
    Channel("samples", "sampleid", "center", "maxsim"),
    Channel("sonic", "muq_mulan", "raw", "maxsim"),
)}


def prep_windows(x: torch.Tensor, prep: str, mu: torch.Tensor | None = None,
                 whiten: torch.Tensor | None = None) -> torch.Tensor:
    """Raw windows [..., D] -> the vectors a channel scores. ``whiten``: unit windows less the index mean ``mu``,
    times the whitening matrix [D, D] (``index_whiten``), then unit length again."""
    if prep == "none":
        return x
    if prep == "raw":
        return l2(x)
    if prep == "center":
        if mu is None:
            raise ValueError("center prep needs the index mean")
        return l2(x - mu)
    if prep == "whiten":
        if mu is None or whiten is None:
            raise ValueError("whiten prep needs the index mean and whitening matrix")
        return l2((l2(x) - mu) @ whiten)
    raise ValueError(prep)


def built_channel(channel: Channel, meta: Mapping) -> Channel | None:
    """``channel`` as the index with this ``meta.json`` holds it: itself, or its whitened form when a ``raw``
    channel's index was whitened (same model, windows and scoring); None for an index of another channel."""
    if (meta["model"], meta["mode"]) != (channel.model, channel.mode):
        return None
    if meta["prep"] == channel.prep:
        return channel
    if (channel.prep, meta["prep"]) == ("raw", "whiten"):
        return replace(channel, prep="whiten")
    return None


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
        """Best ``k`` (track_id, score), ties broken by track ID; ``exclude`` IDs and rows scored -inf (outside an
        ``allow`` mask) are never returned."""
        row = scores.clone()
        drop = sorted({self.pos[int(t)] for t in exclude if int(t) in self.pos})
        if drop:
            row[torch.tensor(drop, device=row.device)] = -torch.inf
        k = min(k, int((row > -torch.inf).sum()))
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

    def __init__(self, channel: Channel, ids: np.ndarray, feats: torch.Tensor, mu: torch.Tensor | None):
        if feats.dim() != 3 or len(ids) != len(feats):
            raise ValueError("expected [N, W, D] window features matching ids")
        self.channel, self.feats, self.mu = channel, feats, mu
        self._set_ids(ids, feats.device)

    @classmethod
    def from_windows(cls, channel: Channel, ids: np.ndarray, win: np.ndarray, device: str = "cpu",
                     chunk: int = 4096) -> ChannelIndex:
        mu = None
        if channel.prep == "center":
            total = np.zeros(win.shape[-1], np.float64)
            for s in range(0, len(win), chunk):
                total += win[s:s + chunk].astype(np.float32).reshape(-1, win.shape[-1]).sum(0, dtype=np.float64)
            mu = torch.from_numpy((total / (len(win) * win.shape[1])).astype(np.float32)).to(device)
        feats = torch.empty(win.shape, dtype=torch.float16, device=device)
        for s in range(0, len(win), chunk):
            x = torch.from_numpy(np.ascontiguousarray(win[s:s + chunk])).to(device).float()
            feats[s:s + chunk] = prep_windows(x, channel.prep, mu).half()
        return cls(channel, ids, feats, mu)

    @classmethod
    def load(cls, emb_dir: Path, channel: Channel, device: str = "cpu") -> ChannelIndex:
        st = load_keys(Path(emb_dir) / channel.model, ["win"])
        return cls.from_windows(channel, st["ids"], st["win"], device)

    def prep_query(self, win: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Raw query windows [W, D] or [B, W, D] -> prepared fp32 tensor on the index device."""
        x = torch.as_tensor(win).to(self.feats.device).float()
        return prep_windows(x, self.channel.prep, self.mu)

    def stored_query(self, track_id: int) -> torch.Tensor:
        return self.feats[self.pos[int(track_id)]].float()

    @torch.inference_mode()
    def score_rows(self, q: torch.Tensor, rows: np.ndarray) -> torch.Tensor:
        """Exact scores of one prepared query [W, D] against the given rows."""
        if not len(rows):
            return torch.empty(0, device=self.feats.device)
        return score_block(q[None], self.feats[torch.as_tensor(rows, device=self.feats.device)].float(),
                           self.channel.mode)[0]

    @torch.inference_mode()
    def scores(self, q: torch.Tensor, chunk: int = 2048, allow: torch.Tensor | None = None) -> torch.Tensor:
        """Prepared queries [W, D] or [B, W, D] -> similarities [N] or [B, N] (higher is closer). Rows outside a
        boolean ``allow`` mask ([N] or [B, N]) score -inf."""
        return self.scores_with_base(q, chunk, allow, base=False)[0]

    @torch.inference_mode()
    def scores_with_base(self, q: torch.Tensor, chunk: int = 2048, allow: torch.Tensor | None = None,
                         base: bool = True) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``scores`` plus (with ``base``) every row's score before the ``allow`` mask."""
        single = q.dim() == 2
        q = q[None] if single else q
        out = torch.empty(len(q), len(self), device=self.feats.device)
        for s in range(0, len(self), chunk):
            out[:, s:s + chunk] = score_block(q, self.feats[s:s + chunk].float(), self.channel.mode)
        kept = None
        if base:
            kept = out.clone() if allow is not None else out
        if allow is not None:
            out.masked_fill_(~allow, -torch.inf)
        return (out[0], kept[0] if base else None) if single else (out, kept)

    def search(self, q: torch.Tensor, allow: torch.Tensor | None = None, base: bool = False
               ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """One prepared query [W, D] -> its ``scores_with_base`` pair and which rows are scored exactly (None: every
        row)."""
        return (*self.scores_with_base(q, allow=allow, base=base), None)

    def top_within(self, q: torch.Tensor, scores: torch.Tensor, exact: torch.Tensor | None, within: torch.Tensor,
                   k: int, exclude: Iterable[int] = ()) -> list[tuple[int, float]]:
        """Best ``k`` (track_id, score) among the rows of the boolean mask ``within``, from ``search``'s scores."""
        return self.top(scores.masked_fill(~within, -torch.inf), k, exclude)


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
    """Catalog-scale channel index written by ``index_build`` (or ``index_whiten``, whose stage 1 holds every
    dimension as 8-bit integers and whose ``whiten`` matrix and mean ``mu`` turn raw query windows into its space).

    Stage 1 scores every track from low-dimensional window projections held on the device (distance
    modes use exact squared norms, so only the dot product is approximated). Stage 2 re-scores the best
    ``candidates`` exactly from memory-mapped full windows. Tracks outside the candidate set keep their
    stage-1 estimate shifted below every exact score, so ranks past ``candidates`` are approximate.

    Projections are fp16, or integers with one scale per window (``proj_scale``): int8, or uint8 holding two
    4-bit values per byte (``index_build.quantize_windows``). Integers are exact in fp16, so with ``half_stage1``
    stage 1 multiplies them as an fp16 GEMM (tensor cores, half the bytes) against the query projection scaled to
    |x| <= 1, which keeps every window dot product inside fp16's range; fp16 projections ignore the option.

    With a hub penalty (``hub_build``), every score is lowered by ``alpha * h(track)`` before candidates are
    picked, where ``h`` is the track's mean top-k song similarity: CSLS, which demotes tracks close to everything.
    """

    def __init__(self, channel: Channel, ids: np.ndarray, proj: torch.Tensor, basis: torch.Tensor,
                 norms: torch.Tensor, full: np.ndarray, mu: torch.Tensor | None, candidates: int = 1000,
                 hub: torch.Tensor | None = None, hub_alpha: float = 0.0, proj_scale: torch.Tensor | None = None,
                 half_stage1: bool = False, whiten: torch.Tensor | None = None):
        n, w = len(ids), full.shape[1]
        self.proj_bits = {torch.int8: 8, torch.uint8: 4}.get(proj.dtype, 16)
        stored_dim = proj.shape[2] * 2 if self.proj_bits == 4 else proj.shape[2]
        if proj.shape[:2] != (n, w) or tuple(norms.shape) != (n, w) or len(full) != n \
                or basis.shape != (full.shape[2], stored_dim):
            raise ValueError("inconsistent two-stage index arrays")
        if (self.proj_bits < 16) != (proj_scale is not None) or \
                (proj_scale is not None and tuple(proj_scale.shape) != (n, w)):
            raise ValueError("quantized projections need one scale per window (and fp16 ones none)")
        if channel.prep in ("center", "whiten") and mu is None:
            raise ValueError(f"{channel.prep} prep needs the index mean")
        if (channel.prep == "whiten") != (whiten is not None) or \
                (whiten is not None and tuple(whiten.shape) != (full.shape[2], full.shape[2])):
            raise ValueError("a whitened index needs its [D, D] whitening matrix (and no other index has one)")
        if candidates < 1:
            raise ValueError("candidates must be positive")
        if hub_alpha and (hub is None or tuple(hub.shape) != (n,)):
            raise ValueError("a hub penalty needs one value per track")
        self.channel, self.proj, self.basis, self.norms, self.full, self.mu = channel, proj, basis, norms, full, mu
        self.whiten = whiten
        self.proj_scale = proj_scale.to(proj.device).float() if proj_scale is not None else None
        self.candidates, self.device = candidates, proj.device
        self.half_stage1 = bool(half_stage1) and self.proj_bits < 16
        self.hub_alpha = float(hub_alpha)
        self.penalty = (self.hub_alpha * hub.to(proj.device).float()) if self.hub_alpha else None
        self._set_ids(ids, proj.device)

    @classmethod
    def load(cls, path: Path, channel: Channel, device: str = "cpu", candidates: int | None = None,
             read_threads: int = 8, hub_alpha: float | None = None, half_stage1: bool = False) -> TwoStageIndex:
        """``candidates`` defaults to the depth recorded at build time and ``hub_alpha`` to the alpha recorded
        by ``hub_build`` (0 when absent; pass 0 to disable the penalty)."""
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        built = built_channel(channel, meta)
        if built is None:
            raise ValueError(f"{path} was built for {(meta['model'], meta['prep'], meta['mode'])}, not {channel}")
        channel = built
        mu = torch.from_numpy(np.load(path / "mu.npy")).to(device) if (path / "mu.npy").exists() else None
        alpha = float(meta.get("hub", {}).get("alpha", 0.0) if hub_alpha is None else hub_alpha)
        if alpha and not (path / "hub.npy").exists():
            raise FileNotFoundError(f"{path / 'hub.npy'} is missing; run hub_build or use hub_alpha=0")
        hub = torch.from_numpy(np.load(path / "hub.npy").astype(np.float32)) if alpha else None
        scale = torch.from_numpy(np.load(path / "proj_scale.npy")) if (path / "proj_scale.npy").exists() else None
        whiten = torch.from_numpy(np.load(path / "whiten.npy")).to(device) if channel.prep == "whiten" else None
        return cls(channel, np.load(path / "ids.npy"), _to_device(np.load(path / "proj.npy", mmap_mode="r"), device),
                   torch.from_numpy(np.load(path / "basis.npy")).to(device),
                   torch.from_numpy(np.load(path / "norms.npy")).to(device),
                   RowFile(path / "full.npy", read_threads), mu, candidates or meta.get("candidates", 1000),
                   hub, alpha, scale, half_stage1, whiten)

    def prep_query(self, win: np.ndarray | torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(win).to(self.device).float()
        return prep_windows(x, self.channel.prep, self.mu, self.whiten)

    def _full(self, rows: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.asarray(self.full[rows])).to(self.device).float()

    def stored_query(self, track_id: int) -> torch.Tensor:
        return self._full(np.array([self.pos[int(track_id)]]))[0]

    @torch.inference_mode()
    def score_rows(self, q: torch.Tensor, rows: np.ndarray) -> torch.Tensor:
        """Exact scores (hub penalty included, as ``scores`` gives its candidates) of one prepared query [W, D]
        against the given rows, sorted ascending for sequential reads."""
        exact = torch.cat([score_block(q[None], self._full(rows[s:s + 2048]), self.channel.mode)[0]
                           for s in range(0, len(rows), 2048)]) if len(rows) else torch.empty(0, device=self.device)
        if self.penalty is not None and len(rows):
            exact -= self.penalty[torch.as_tensor(rows, device=self.device)]
        return exact

    def _proj_rows(self, a: int, b: int) -> torch.Tensor:
        """Stage-1 projections of rows a..b as fp32, or fp16 with ``half_stage1`` (quantized integers before their
        per-window scale)."""
        p = self.proj[a:b]
        if self.proj_bits == 4:
            p = torch.stack(((p & 15).to(torch.int8) - 8, (p >> 4).to(torch.int8) - 8), -1).flatten(-2)
        return p.half() if self.half_stage1 else p.float()

    @torch.inference_mode()
    def stage1(self, q: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
        """Prepared queries [B, W, D] -> approximate scores [B, N]."""
        qp, qn, dim = q @ self.basis, (q ** 2).sum(-1), q.shape[-1]
        if self.half_stage1:
            qs = qp.abs().amax(dim=(1, 2)).clamp_min(1e-12)[:, None, None, None]
            qp = (qp / qs[..., 0]).half()
        out = torch.empty(len(q), len(self), device=self.device)
        for s in range(0, len(self), chunk):
            sims = torch.einsum("bwd,nvd->bnwv", qp, self._proj_rows(s, s + chunk))
            if self.half_stage1:
                sims = sims.float() * qs
            if self.proj_scale is not None:
                sims *= self.proj_scale[s:s + chunk][None, :, None, :]
            out[:, s:s + chunk] = window_scores(sims, self.channel.mode, qn, self.norms[s:s + chunk], dim)
        return out

    @torch.inference_mode()
    def scores(self, q: torch.Tensor, chunk: int = 8192, allow: torch.Tensor | None = None) -> torch.Tensor:
        """Prepared queries [W, D] or [B, W, D] -> scores [N] or [B, N]; the top ``candidates`` are exact.

        Rows outside a boolean ``allow`` mask ([N] or [B, N]) score -inf and are skipped when candidates are
        picked, so the exact re-scoring covers the best allowed rows."""
        return self.scores_with_base(q, chunk, allow, base=False)[0]

    @torch.inference_mode()
    def scores_with_base(self, q: torch.Tensor, chunk: int = 8192, allow: torch.Tensor | None = None,
                         base: bool = True) -> tuple[torch.Tensor, torch.Tensor | None]:
        """``scores`` plus (with ``base``) every row's stage-1 score less its hub penalty: before the ``allow`` mask
        and the exact re-scoring, which moves the other rows below the candidates, so on one scale for all rows."""
        single = q.dim() == 2
        out, kept, _ = self._scored(q[None] if single else q, chunk, allow, base)
        return (out[0], kept[0] if base else None) if single else (out, kept)

    def _scored(self, q: torch.Tensor, chunk: int, allow: torch.Tensor | None, base: bool
                ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        """``scores_with_base`` for queries [B, W, D], plus each query's exactly re-scored rows."""
        out = self.stage1(q, chunk)
        if self.penalty is not None:
            out -= self.penalty
        kept = out.clone() if base else None
        if allow is not None:
            out.masked_fill_(~allow, -torch.inf)
        c = min(self.candidates, len(self))
        cands = []
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
            cands.append(cand)
        if allow is not None:  # with fewer allowed rows than candidates, some re-scored rows were not allowed
            out.masked_fill_(~allow, -torch.inf)
        return out, kept, cands

    @torch.inference_mode()
    def search(self, q: torch.Tensor, allow: torch.Tensor | None = None, base: bool = False
               ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """One prepared query [W, D] -> its ``scores_with_base`` pair and a boolean mask of the rows that were
        re-scored exactly (the candidates)."""
        out, kept, cands = self._scored(q[None], 8192, allow, base)
        exact = torch.zeros(len(self), dtype=torch.bool, device=self.device)
        exact[cands[0]] = True
        return out[0], kept[0] if base else None, exact

    @torch.inference_mode()
    def top_within(self, q: torch.Tensor, scores: torch.Tensor, exact: torch.Tensor, within: torch.Tensor, k: int,
                   exclude: Iterable[int] = ()) -> list[tuple[int, float]]:
        """Best ``k`` (track_id, score) among the rows of the boolean mask ``within``, from ``search``'s scores and
        exact mask, without a second pass over the index: the best ``candidates`` of those rows (the search's own
        candidates first, then by stage-1 score) are the pool, the ones the search did not re-score are scored exactly
        now, and the pool is ranked by exact score."""
        row = scores.masked_fill(~within, -torch.inf)
        c = min(self.candidates, int((row > -torch.inf).sum()))
        if c <= 0:
            return []
        cand = row.topk(c).indices
        out = torch.full_like(row, -torch.inf)
        out[cand] = row[cand]
        new = cand[~exact[cand]].sort().values
        if len(new):
            out[new] = self.score_rows(q, new.cpu().numpy())
        return self.top(out, k, exclude)


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


class Engine:
    """Lazy model loading plus per-channel indexes. Not thread-safe: serialise calls per process.

    ``index_dir/<channel>/meta.json`` (``index_build`` output) loads a ``TwoStageIndex``; otherwise the
    channel is brute-forced from ``index_dir/<model>/part-*.npz`` shards.
    """

    # Optional ``blend.Blend`` over the sonic index: the searches then also return ``results["blend"]``.
    blend = None

    def __init__(self, index_dir: Path | str, channels: Iterable[str] = tuple(CHANNELS), device: str = "cuda",
                 index_device: str | None = None, drop_text_tower: bool = True, candidates: int | None = None,
                 hub_alpha: float | None = None, half_stage1: Iterable[str] = ()):
        self.device = device
        self.drop_text_tower = drop_text_tower
        self.channels = [CHANNELS[c] for c in channels]
        self.load_secs: dict[str, float] = {}
        self.indexes: dict[str, ChannelIndex | TwoStageIndex] = {}
        half = set(half_stage1)
        for ch in self.channels:
            t = time.perf_counter()
            built = Path(index_dir) / ch.name
            if (built / "meta.json").exists():
                self.indexes[ch.name] = TwoStageIndex.load(built, ch, index_device or device, candidates,
                                                           hub_alpha=hub_alpha, half_stage1=ch.name in half)
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
        """Raw window embeddings per channel for decoded clips, plus GPU timings in ms."""
        timings: dict[str, float] = {}
        out: dict[str, np.ndarray] = {}
        for ch in self.channels:
            emb = self.embedder(ch.model)
            _sync(self.device)
            t = time.perf_counter()
            win = emb.embed(clips[emb.sr][None])["win"][0]
            _sync(self.device)
            timings[f"embed_{ch.model}"] = (time.perf_counter() - t) * 1000
            if not np.isfinite(win.astype(np.float32)).all():
                raise ValueError(f"{ch.model} produced non-finite embeddings")
            out[ch.name] = win
        return out, timings

    def embed(self, path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        clips, t_decode = self.decode(path)
        wins, t_embed = self.embed_clips(clips)
        return wins, t_decode | t_embed

    def _search(self, ch: Channel, q: torch.Tensor, k: int, exclude: list[int], allow, row: int | None,
                windows: np.ndarray | None, results: dict, timings: dict[str, float], within=None) -> None:
        """One channel's top ``k`` for a prepared query. With a ``blend``, the sonic channel also gives
        ``results["blend"]``: the top ``k`` rows by blended score (``blend.Blend.scores``; the same ``allow`` mask
        and ``exclude``) in that order, each with its exact sonic score; and ``results["blendz"]``: the blended score
        of every song of those two lists, as (track id, score). ``row``: the seed's row in the sonic index (None for
        a live seed, whose raw sonic ``windows`` then give its head vectors).

        With ``within`` (channel -> boolean row mask), ``results["within"]`` holds the same lists among the rows of
        that mask alone (``top_within``): one per channel and ``"blend"``, each holding every song of its unmasked
        list that passes the mask (so at times a few more than ``k``); ``blendz`` covers their songs too. The scores
        of the search are reused; the time goes to ``timings["search_lang"]``."""
        idx = self.indexes[ch.name]
        mask = allow.get(ch.name) if allow else None
        inside = within.get(ch.name) if within else None
        blend = self.blend if ch.name == "sonic" else None
        _sync(self.device)
        t = time.perf_counter()
        scores, base, exact_rows = idx.search(q, allow=mask, base=blend is not None)
        results[ch.name] = idx.top(scores, k, exclude)
        _sync(self.device)
        timings[f"search_{ch.name}"] = (time.perf_counter() - t) * 1000
        if inside is not None:
            t = time.perf_counter()
            top = idx.top_within(q, scores, exact_rows, inside, k, exclude)
            if results[ch.name]:
                # Exact scores can place newly scored rows above songs of the unmasked list: those stay listed.
                have = {tid for tid, _ in top}
                at = torch.as_tensor([idx.pos[tid] for tid, _ in results[ch.name]], dtype=torch.long,
                                     device=inside.device)
                more = [pair for pair, ok in zip(results[ch.name], inside[at].tolist()) if ok and pair[0] not in have]
                if more:
                    top = sorted(top + more, key=lambda pair: (-pair[1], pair[0]))
            results.setdefault("within", {})[ch.name] = top
            _sync(self.device)
            timings["search_lang"] = timings.get("search_lang", 0.0) + (time.perf_counter() - t) * 1000
        if blend is None:
            return
        t = time.perf_counter()
        blended = blend.scores(base, blend.seed_head(row, windows), row)
        if mask is not None:
            blended = blended.masked_fill(~mask, -torch.inf)
        tids = [tid for tid, _ in idx.top(blended, k, exclude)]
        rows = np.array(sorted(idx.pos[tid] for tid in tids), dtype=np.int64)
        exact = dict(zip(idx.ids[rows].tolist(), idx.score_rows(q, rows).tolist()))
        results["blend"] = [(tid, float(exact[tid])) for tid in tids]
        both = list(dict.fromkeys([tid for tid, _ in results[ch.name]] + tids))
        at = torch.as_tensor([idx.pos[tid] for tid in both], dtype=torch.long, device=blended.device)
        results["blendz"] = list(zip(both, blended[at].tolist()))
        _sync(self.device)
        timings["search_blend"] = (time.perf_counter() - t) * 1000
        if inside is None:
            return
        t = time.perf_counter()
        # The masked blended ranking; exact sonic scores come from the lists that hold them, else are computed.
        tids = [tid for tid, _ in idx.top(blended.masked_fill(~inside, -torch.inf), k, exclude)]
        exact |= dict(results["within"][ch.name])
        rows = np.array(sorted(idx.pos[tid] for tid in tids if tid not in exact), dtype=np.int64)
        exact |= dict(zip(idx.ids[rows].tolist(), idx.score_rows(q, rows).tolist()))
        results["within"]["blend"] = [(tid, float(exact[tid])) for tid in tids]
        seen = set(both)
        more = [tid for tid in dict.fromkeys([tid for tid, _ in results["within"][ch.name]] + tids)
                if tid not in seen]
        if more:
            at = torch.as_tensor([idx.pos[tid] for tid in more], dtype=torch.long, device=blended.device)
            results["blendz"] += list(zip(more, blended[at].tolist()))
        _sync(self.device)
        timings["search_lang"] += (time.perf_counter() - t) * 1000

    def search_windows(self, wins: dict[str, np.ndarray], k: int = 20, exclude: Iterable[int] = (),
                       allow: Mapping[str, torch.Tensor] | None = None,
                       within: Mapping[str, torch.Tensor] | None = None
                       ) -> tuple[dict[str, list[tuple[int, float]]], dict[str, float]]:
        """Top ``k`` per channel for raw windows; ``allow`` maps channels to row masks (``TwoStageIndex.scores``).
        With a ``blend``, also ``results["blend"]``; with ``within`` (channel -> row mask), also
        ``results["within"]``: the lists among those rows alone (``_search``)."""
        results, timings = {}, {}
        exclude = list(exclude)
        for ch in self.channels:
            q = self.indexes[ch.name].prep_query(wins[ch.name])
            self._search(ch, q, k, exclude, allow, None, wins[ch.name], results, timings, within)
        return results, timings

    def search(self, path: str | Path, k: int = 20, exclude: Iterable[int] = ()) -> dict:
        wins, timings = self.embed(path)
        results, t_search = self.search_windows(wins, k, exclude)
        timings |= t_search
        return {"results": results, "timings_ms": {k_: round(v, 2) for k_, v in timings.items()}}

    def indexed(self, track_id: int) -> bool:
        return all(int(track_id) in ix.pos for ix in self.indexes.values())

    def score_tracks(self, channel: str, track_ids: Iterable[int], windows: np.ndarray | None = None,
                     query_track: int | None = None) -> dict[int, float]:
        """Exact scores of the indexed ``track_ids`` in one channel, against live ``windows`` (raw, as embedded) or
        the stored windows of ``query_track``; ids outside the index are left out."""
        idx = self.indexes[channel]
        q = idx.prep_query(windows) if windows is not None else idx.stored_query(int(query_track))
        rows = np.array(sorted({idx.pos[int(t)] for t in track_ids if int(t) in idx.pos}), dtype=np.int64)
        values = idx.score_rows(q, rows).tolist()
        return {int(idx.ids[r]): float(v) for r, v in zip(rows, values)}

    def search_stored(self, track_id: int, k: int = 20, exclude: Iterable[int] = (),
                      allow: Mapping[str, torch.Tensor] | None = None,
                      within: Mapping[str, torch.Tensor] | None = None
                      ) -> tuple[dict[str, list[tuple[int, float]]], dict[str, float]]:
        """Rank an indexed track from its stored fp16 windows instead of re-embedding its preview (live
        embeddings match the stored ones up to fp16 rounding; see ``serve_check``). The track itself is
        excluded; ``allow`` and ``within`` as in ``search_windows``."""
        results, timings = {}, {}
        exclude = [int(track_id), *exclude]
        for ch in self.channels:
            idx = self.indexes[ch.name]
            self._search(ch, idx.stored_query(track_id), k, exclude, allow, idx.pos[int(track_id)], None, results,
                         timings, within)
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
    p.add_argument("--channels", nargs="+", default=list(CHANNELS), choices=list(CHANNELS))
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
