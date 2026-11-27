"""Blended catalog score: one ranking of every indexed song from three views of the seed
(docs/ANYSONG_EVAL.md, "Blended main list"):

    blend = z(sonic) + wh * z(head) + wc * zcf + wp * zpop

- sonic: the sonic channel's score of every row before its candidates are re-scored exactly (stage 1 minus the hub
  penalty for a ``serve.TwoStageIndex``), so one scale covers the whole catalog;
- head: the best window-pair cosine between the seed's scene-head vectors (``scenehead``) and every row's;
- zcf: the cosine between playlist co-occurrence vectors, z-scored over the songs of the playlist graph; songs
  outside the graph get 0, and a seed outside it (every live seed) has no such term;
- zpop (optional, wp 0 by default): the song's popularity, log1p(Deezer rank) (``listener.Popularity``), z-scored
  over every row of the index once at load. It does not depend on the seed.

z(x) = (x - mean) / sd over every row of the index for that seed.

With the head table in host memory (``Blend.load`` with ``host_table``; ``serve.HostTable``) the head term reads it
block by block, with the same result.

Files: the scene head's ``<name>.npz`` and ``<name>.emb.npy``; ``<dir>/cooc.npz`` with ``rows`` (the graph songs'
sonic-index rows) and ``<dir>/cf128.npy`` (one vector per graph song).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .scenehead import SceneHead
from .serve import HostTable, _to_device

WEIGHTS = (2.0, 1.0)  # wh, wc; a third weight is wp (0 without it)
HEAD_CHUNK = 16384  # rows of the head table per product


def zscore(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean()) / x.std().clamp_min(1e-6)


def unit_half(x: torch.Tensor) -> torch.Tensor:
    """fp16 vectors [..., D] at unit length again, in fp16 (the consensus study's tables, ``consensus``)."""
    return x / x.float().norm(dim=-1, keepdim=True).clamp_min(1e-6).half()


def _unit_block(unit_seed: torch.Tensor, block: torch.Tensor) -> torch.Tensor:
    """``Blend.unit_scores`` for one block of table rows [n, W, dim] -> [n]."""
    rows = unit_half(block.half())
    n, w, d = rows.shape
    q = unit_seed.half()
    return (q @ rows.reshape(n * w, d).T).reshape(len(q), n, w).amax(dim=(0, 2)).float()


class Blend:
    """Scene-head vectors and playlist vectors on the index device; ``scores`` blends them with the sonic scores."""

    def __init__(self, head: SceneHead, table: torch.Tensor | HostTable, cf_rows: np.ndarray, cf: np.ndarray,
                 weights: tuple[float, ...] = WEIGHTS, pop: np.ndarray | None = None):
        n = len(table)
        cf_rows = np.asarray(cf_rows, dtype=np.int64)
        cf = np.asarray(cf, dtype=np.float32)
        if cf.ndim != 2 or len(cf) != len(cf_rows):
            raise ValueError(f"{len(cf_rows)} graph rows do not fit playlist vectors {cf.shape}")
        if len(cf_rows) and (cf_rows.min() < 0 or cf_rows.max() >= n or len(np.unique(cf_rows)) != len(cf_rows)):
            raise ValueError(f"graph rows must be distinct rows of the index ({n} songs)")
        self.head, self.table = head, table
        self.streamed = isinstance(table, HostTable)  # the head table is in host memory
        if len(weights) not in (2, 3):
            raise ValueError("blend weights are (wh, wc) or (wh, wc, wp)")
        self.wh, self.wc, self.wp = (float(w) for w in (*weights, 0.0)[:3])
        if self.wp and pop is None:
            raise ValueError("a popularity weight needs the popularity of every index row")
        if pop is not None and np.shape(pop) != (n,):
            raise ValueError(f"popularity {np.shape(pop)} does not fit the index ({n} songs)")
        device = table.device
        self.cf = torch.from_numpy(cf / np.maximum(np.linalg.norm(cf, axis=1, keepdims=True), 1e-6)).to(device)
        self.cf_rows = torch.from_numpy(cf_rows).to(device)
        # Row of the index -> row of ``cf``; -1 outside the playlist graph.
        self.graph_pos = torch.full((n,), -1, dtype=torch.long, device=device)
        self.graph_pos[self.cf_rows] = torch.arange(len(cf_rows), device=device)
        # The popularity term of every row; None without a popularity weight.
        self.zpop = zscore(torch.from_numpy(np.asarray(pop, dtype=np.float32)).to(device)) if self.wp else None

    @classmethod
    def load(cls, head: Path | str, cf_dir: Path | str, pos: dict[int, int], device: torch.device | str = "cpu",
             weights: tuple[float, ...] = WEIGHTS, pop: np.ndarray | None = None, host_table: bool = False
             ) -> Blend:
        """``head``: path prefix of the scene head files; ``pos``: the sonic index's track id -> row; ``pop``: the
        popularity of every index row, in row order (needed for a popularity weight); ``host_table``: the head
        table stays in host memory (``serve.HostTable``) instead of on the device."""
        scene_head = SceneHead(f"{head}.npz", f"{head}.emb.npy", pos)
        with np.load(Path(cf_dir) / "cooc.npz", allow_pickle=False) as z:
            rows = z["rows"]
        table = HostTable(f"{head}.emb.npy", device, HEAD_CHUNK) if host_table else \
            _to_device(scene_head.table, device)
        return cls(scene_head, table, rows, np.load(Path(cf_dir) / "cf128.npy"), weights, pop)

    def __len__(self) -> int:
        return len(self.table)

    def info(self) -> dict:
        return {"songs": len(self), "graph": len(self.cf_rows),
                "weights": [self.wh, self.wc, *([self.wp] if self.wp else [])],
                **({"head_table": "host"} if self.streamed else {})}

    def _blocks(self, chunk: int):
        """(first row, the next rows of the head table on the device) over the table."""
        if self.streamed:
            return self.table.blocks()
        return ((s, self.table[s:s + chunk]) for s in range(0, len(self), chunk))

    def seed_head(self, row: int | None, windows: np.ndarray | None = None, part=None) -> torch.Tensor:
        """The seed's head vectors [W, dim]: its stored ones (``part``: only those windows of them), or those of a
        live seed's raw sonic ``windows``."""
        if row is not None:
            head = self.table[row].float()
            return head if part is None else head[list(part)]
        return torch.from_numpy(self.head.embed(np.asarray(windows, dtype=np.float32))).to(self.table.device)

    @torch.inference_mode()
    def head_scores(self, seed: torch.Tensor, chunk: int = HEAD_CHUNK) -> torch.Tensor:
        """Best window-pair cosine between the seed's head vectors and every row's [N]."""
        return self.head_views(seed, None, chunk)[0]

    def unit_seed(self, row: int | None, seed: torch.Tensor, part=None) -> torch.Tensor:
        """The seed's head vectors for ``unit_scores``: a stored seed's at unit length again in fp16 (``part``: only
        those windows of them), a live seed's (``seed``, from ``seed_head``) as they are."""
        if row is None:
            return seed
        unit = unit_half(self.table[row].half())
        return unit if part is None else unit[list(part)]

    @torch.inference_mode()
    def head_views(self, seed: torch.Tensor, unit_seed: torch.Tensor | None, chunk: int = HEAD_CHUNK
                   ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """One pass over the head table for ``head_scores`` and, with ``unit_seed`` [W, dim], ``unit_scores`` of it."""
        device = self.table.device
        out = torch.empty(len(self), device=device)
        unit = torch.empty(len(self), device=device) if unit_seed is not None else None
        qt = seed.float().T.contiguous()
        for s, block in self._blocks(chunk):
            n = len(block)
            out[s:s + n] = (block.float().flatten(0, 1) @ qt).view(n, -1).amax(1)
            if unit is not None:
                unit[s:s + n] = _unit_block(unit_seed, block)
        return out, unit

    @torch.inference_mode()
    def unit_scores(self, unit_seed: torch.Tensor, chunk: int = HEAD_CHUNK) -> torch.Tensor:
        """The head signal of ``consensus``: best window-pair product of the seed's vectors [W, dim] with every row's
        [N], the rows at unit length again in fp16 and the product in fp16."""
        out = torch.empty(len(self), device=self.table.device)
        for s, block in self._blocks(chunk):
            out[s:s + len(block)] = _unit_block(unit_seed, block)
        return out

    @torch.inference_mode()
    def cf_scores(self, row: int | None) -> torch.Tensor | None:
        """The playlist term [N] for the seed at index ``row``; None for a seed outside the playlist graph."""
        g = int(self.graph_pos[row]) if row is not None else -1
        if g < 0:
            return None
        c = self.cf @ self.cf[g]
        mu = c.mean()
        sd = ((c - mu) ** 2).mean().sqrt().clamp_min(1e-6)
        out = torch.zeros(len(self), device=c.device)
        out[self.cf_rows] = (c - mu) / sd
        return out

    @torch.inference_mode()
    def scores(self, sonic: torch.Tensor, seed_head: torch.Tensor, row: int | None,
               head: torch.Tensor | None = None) -> torch.Tensor:
        """Blended score of every row [N] from the seed's ``sonic`` scores [N] (finite for every row), its head
        vectors and its row in the index (None for a live seed). ``head``: ``head_scores(seed_head)`` when the
        caller has them already."""
        out = zscore(sonic.float())
        if self.wh:
            out = out + self.wh * zscore(self.head_scores(seed_head) if head is None else head)
        if self.wc:
            cf = self.cf_scores(row)
            if cf is not None:
                out = out + self.wc * cf
        if self.wp:
            out = out + self.wp * self.zpop
        return out
