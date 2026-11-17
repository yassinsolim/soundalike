"""Blended catalog score: one ranking of every indexed song from three views of the seed
(docs/ANYSONG_EVAL.md, "Blended main list"):

    blend = z(sonic) + wh * z(head) + wc * zcf

- sonic: the sonic channel's score of every row before its candidates are re-scored exactly (stage 1 minus the hub
  penalty for a ``serve.TwoStageIndex``), so one scale covers the whole catalog;
- head: the best window-pair cosine between the seed's scene-head vectors (``scenehead``) and every row's;
- zcf: the cosine between playlist co-occurrence vectors, z-scored over the songs of the playlist graph; songs
  outside the graph get 0, and a seed outside it (every live seed) has no such term.

z(x) = (x - mean) / sd over every row of the index for that seed.

Files: the scene head's ``<name>.npz`` and ``<name>.emb.npy``; ``<dir>/cooc.npz`` with ``rows`` (the graph songs'
sonic-index rows) and ``<dir>/cf128.npy`` (one vector per graph song).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .scenehead import SceneHead
from .serve import _to_device

WEIGHTS = (2.0, 1.0)  # wh, wc


def zscore(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean()) / x.std().clamp_min(1e-6)


class Blend:
    """Scene-head vectors and playlist vectors on the index device; ``scores`` blends them with the sonic scores."""

    def __init__(self, head: SceneHead, table: torch.Tensor, cf_rows: np.ndarray, cf: np.ndarray,
                 weights: tuple[float, float] = WEIGHTS):
        n = len(table)
        cf_rows = np.asarray(cf_rows, dtype=np.int64)
        cf = np.asarray(cf, dtype=np.float32)
        if cf.ndim != 2 or len(cf) != len(cf_rows):
            raise ValueError(f"{len(cf_rows)} graph rows do not fit playlist vectors {cf.shape}")
        if len(cf_rows) and (cf_rows.min() < 0 or cf_rows.max() >= n or len(np.unique(cf_rows)) != len(cf_rows)):
            raise ValueError(f"graph rows must be distinct rows of the index ({n} songs)")
        self.head, self.table = head, table
        self.wh, self.wc = (float(w) for w in weights)
        device = table.device
        self.cf = torch.from_numpy(cf / np.maximum(np.linalg.norm(cf, axis=1, keepdims=True), 1e-6)).to(device)
        self.cf_rows = torch.from_numpy(cf_rows).to(device)
        # Row of the index -> row of ``cf``; -1 outside the playlist graph.
        self.graph_pos = torch.full((n,), -1, dtype=torch.long, device=device)
        self.graph_pos[self.cf_rows] = torch.arange(len(cf_rows), device=device)

    @classmethod
    def load(cls, head: Path | str, cf_dir: Path | str, pos: dict[int, int], device: torch.device | str = "cpu",
             weights: tuple[float, float] = WEIGHTS) -> Blend:
        """``head``: path prefix of the scene head files; ``pos``: the sonic index's track id -> row."""
        scene_head = SceneHead(f"{head}.npz", f"{head}.emb.npy", pos)
        with np.load(Path(cf_dir) / "cooc.npz", allow_pickle=False) as z:
            rows = z["rows"]
        return cls(scene_head, _to_device(scene_head.table, device), rows, np.load(Path(cf_dir) / "cf128.npy"),
                   weights)

    def __len__(self) -> int:
        return len(self.table)

    def info(self) -> dict:
        return {"songs": len(self), "graph": len(self.cf_rows), "weights": [self.wh, self.wc]}

    def seed_head(self, row: int | None, windows: np.ndarray | None = None) -> torch.Tensor:
        """The seed's head vectors [W, dim]: its stored ones, or those of a live seed's raw sonic ``windows``."""
        if row is not None:
            return self.table[row].float()
        return torch.from_numpy(self.head.embed(np.asarray(windows, dtype=np.float32))).to(self.table.device)

    @torch.inference_mode()
    def head_scores(self, seed: torch.Tensor, chunk: int = 16384) -> torch.Tensor:
        """Best window-pair cosine between the seed's head vectors and every row's [N]."""
        out = torch.empty(len(self), device=self.table.device)
        qt = seed.float().T.contiguous()
        for s in range(0, len(self), chunk):
            block = self.table[s:s + chunk].float()
            out[s:s + chunk] = (block.flatten(0, 1) @ qt).view(len(block), -1).amax(1)
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
    def scores(self, sonic: torch.Tensor, seed_head: torch.Tensor, row: int | None) -> torch.Tensor:
        """Blended score of every row [N] from the seed's ``sonic`` scores [N] (finite for every row), its head
        vectors and its row in the index (None for a live seed)."""
        out = zscore(sonic.float())
        if self.wh:
            out = out + self.wh * zscore(self.head_scores(seed_head))
        if self.wc:
            cf = self.cf_scores(row)
            if cf is not None:
                out = out + self.wc * cf
        return out
