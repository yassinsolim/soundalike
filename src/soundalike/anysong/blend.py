"""Blended catalog score: one ranking of every indexed song from three views of the seed
(docs/ANYSONG_EVAL.md, "Blended main list"):

    blend = z(sonic) + wh * z(head) + wc * zcf

- sonic: the sonic channel's score of every row before its candidates are re-scored exactly (stage 1 minus the hub
  penalty for a ``serve.TwoStageIndex``), so one scale covers the whole catalog;
- head: the best window-pair cosine between the seed's scene-head vectors (``scenehead``) and every row's;
- zcf: the cosine between playlist co-occurrence vectors, z-scored over the songs of the playlist graph; songs
  outside the graph get 0, and a seed outside it (every live seed) has no such term.

z(x) = (x - mean) / sd over every row of the index for that seed. An optional fourth term follows a listener's
taste (``Blend``).

Files: the scene head's ``<name>.npz`` and ``<name>.emb.npy``; ``<dir>/cooc.npz`` with ``rows`` (the graph songs'
sonic-index rows) and ``<dir>/cf128.npy`` (one vector per graph song).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from .scenehead import SceneHead
from .serve import _to_device

WEIGHTS = (2.0, 1.0)  # wh, wc
TASTE_GATE = 0.5


def zscore(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean()) / x.std().clamp_min(1e-6)


class Blend:
    """Scene-head vectors and playlist vectors on the index device; ``scores`` blends them with the sonic scores.

    Taste (optional): ``taste`` = (profiles [C, dim], unit rows in the head's space; their names). A song's pooled
    head vector is the unit mean of its windows' head vectors. A seed whose pooled vector has a cosine of at least
    ``taste_gate`` to its nearest profile gets one more term, ``taste_weight`` * z(cosine of every row's pooled
    vector to that profile); other seeds, and every seed at weight 0, get none.
    """

    def __init__(self, head: SceneHead, table: torch.Tensor, cf_rows: np.ndarray, cf: np.ndarray,
                 weights: tuple[float, float] = WEIGHTS, taste: tuple[np.ndarray, Sequence[str]] | None = None,
                 taste_weight: float = 0.0, taste_gate: float = TASTE_GATE):
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
        self.taste_weight, self.taste_gate = float(taste_weight), float(taste_gate)
        self.profiles = self.pooled = None
        self.taste_names: list[str] = []
        if taste is not None:
            profiles, names = np.asarray(taste[0], dtype=np.float32), [str(name) for name in taste[1]]
            if profiles.ndim != 2 or profiles.shape[1] != table.shape[2] or len(names) != len(profiles) \
                    or not len(names):
                raise ValueError(f"taste profiles {profiles.shape} with {len(names)} names do not fit the head "
                                 f"({table.shape[2]} dims)")
            self.taste_names = names
            self.profiles = torch.nn.functional.normalize(torch.from_numpy(profiles).to(device), dim=1)
            # Every row's pooled head vector [N, dim].
            self.pooled = torch.empty(n, table.shape[2], device=device)
            for s in range(0, n, 16384):
                self.pooled[s:s + 16384] = self._pool(table[s:s + 16384].float())

    @staticmethod
    def _pool(vectors: torch.Tensor) -> torch.Tensor:
        """Head vectors [..., W, dim] -> the unit mean over the windows [..., dim]."""
        return torch.nn.functional.normalize(vectors.mean(-2), dim=-1)

    @classmethod
    def load(cls, head: Path | str, cf_dir: Path | str, pos: dict[int, int], device: torch.device | str = "cpu",
             weights: tuple[float, float] = WEIGHTS, taste: Path | str | None = None, taste_weight: float = 0.0,
             taste_gate: float = TASTE_GATE) -> Blend:
        """``head``: path prefix of the scene head files; ``pos``: the sonic index's track id -> row; ``taste``: npz
        with ``profiles`` and ``names``."""
        scene_head = SceneHead(f"{head}.npz", f"{head}.emb.npy", pos)
        with np.load(Path(cf_dir) / "cooc.npz", allow_pickle=False) as z:
            rows = z["rows"]
        profiles = None
        if taste is not None:
            with np.load(taste, allow_pickle=False) as z:
                profiles = (z["profiles"], z["names"].tolist())
        return cls(scene_head, _to_device(scene_head.table, device), rows, np.load(Path(cf_dir) / "cf128.npy"),
                   weights, profiles, taste_weight, taste_gate)

    def __len__(self) -> int:
        return len(self.table)

    def info(self) -> dict:
        out = {"songs": len(self), "graph": len(self.cf_rows), "weights": [self.wh, self.wc]}
        if self.profiles is not None:
            out["taste"] = {"profiles": len(self.taste_names), "weight": self.taste_weight, "gate": self.taste_gate}
        return out

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
    def taste(self, seed_head: torch.Tensor) -> tuple[int, float] | None:
        """The seed's nearest taste profile and its cosine to it, when the taste term applies to the seed: profiles
        loaded, a weight, and that cosine at least ``taste_gate``."""
        if self.profiles is None or not self.taste_weight:
            return None
        cos, c = (self.profiles @ self._pool(seed_head.float())).max(0)
        return (int(c), float(cos)) if float(cos) >= self.taste_gate else None

    @torch.inference_mode()
    def scores(self, sonic: torch.Tensor, seed_head: torch.Tensor, row: int | None,
               profile: int | None = None) -> torch.Tensor:
        """Blended score of every row [N] from the seed's ``sonic`` scores [N] (finite for every row), its head
        vectors and its row in the index (None for a live seed); with ``profile`` (from ``taste``), plus the taste
        term for that profile."""
        out = zscore(sonic.float())
        if self.wh:
            out = out + self.wh * zscore(self.head_scores(seed_head))
        if self.wc:
            cf = self.cf_scores(row)
            if cf is not None:
                out = out + self.wc * cf
        if profile is not None:
            out = out + self.taste_weight * zscore(self.pooled @ self.profiles[profile])
        return out
