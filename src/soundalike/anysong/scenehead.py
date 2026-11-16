"""Scene head: a small MLP over the sonic channel's windows, trained on track-level playlist co-occurrence
(docs/ANYSONG_EVAL.md, "v8 ranker"). Its cosine between a seed and a candidate is one feature of the listener
ranker; it works for songs outside the catalog because it only needs the seed's own windows.

Files: ``<name>.npz`` with ``w1, b1, w2, b2`` (512 -> hidden -> dim, GELU) and ``<name>.emb.npy``, the head
vectors of every indexed song ([N, windows, dim] float16, rows aligned with the sonic index).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np


class SceneHead:
    def __init__(self, weights: Path | str, table: Path | str, pos: Mapping[int, int]):
        with np.load(weights, allow_pickle=False) as z:
            self.w1, self.b1, self.w2, self.b2 = (z[k].astype(np.float32) for k in ("w1", "b1", "w2", "b2"))
        self.table = np.load(table, mmap_mode="r")
        self.pos = pos
        if self.table.shape[0] != len(pos) or self.table.shape[2] != self.w2.shape[1]:
            raise ValueError(f"scene head table {self.table.shape} does not fit the index ({len(pos)} songs) "
                             f"or the head ({self.w2.shape[1]} dims)")

    def embed(self, windows: np.ndarray) -> np.ndarray:
        """Unit sonic windows [W, 512] -> unit head vectors [W, dim]."""
        h = np.asarray(windows, dtype=np.float32) @ self.w1 + self.b1
        h = 0.5 * h * (1.0 + np.vectorize(math.erf, otypes=[np.float32])(h / math.sqrt(2.0)))
        h = h @ self.w2 + self.b2
        return h / np.maximum(np.linalg.norm(h, axis=-1, keepdims=True), 1e-9)

    def seed(self, windows: np.ndarray | None, track_id: int | None) -> np.ndarray | None:
        """The seed's head vectors: from its live windows, else from the table; None for an unknown song."""
        if windows is not None:
            return self.embed(windows)
        row = self.pos.get(int(track_id)) if track_id is not None else None
        return None if row is None else np.asarray(self.table[row], dtype=np.float32)

    def cosines(self, seed: np.ndarray, track_ids: Iterable[int]) -> dict[int, float]:
        """Best window-pair cosine between the seed and each indexed track; tracks outside the index are left out."""
        tids = [int(t) for t in track_ids if int(t) in self.pos]
        if not tids:
            return {}
        rows = np.array([self.pos[t] for t in tids])
        order = np.argsort(rows)
        vecs = np.asarray(self.table[rows[order]], dtype=np.float32)
        best = np.einsum("wd,nvd->nwv", seed, vecs).max(axis=(1, 2))
        return {tids[i]: float(s) for i, s in zip(order, best)}
