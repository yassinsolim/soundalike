"""Catalog tiers: which indexed songs a seed may be recommended.

The core catalog is the set of songs the index held before ``anysong.crawl`` grew it. The crawl adds songs found
along Deezer's related-artist graph, mostly long-tail releases. In the grown catalog's evaluation
(docs/ANYSONG_EVAL.md, "Catalog growth"), such songs by artists outside the seed's scene took about 45 % of list
rows, yet the seed artist's listeners almost never played those artists (ListenBrainz hit rate 0.004, against 0.052
for core songs as far from the seed). Songs the crawl added by the seed's related artists did hit.

So a discovered (non-core) song is eligible only for seeds by its own primary artist or by an artist whose own
Deezer related list names that artist; core songs are always eligible. The lists are the ones stored in the scene
table, so the search never waits for Deezer; a seed whose artist has no list there gets core songs and its own
artist's (never fewer candidates than the core catalog alone). The gate is a boolean mask per channel index row,
applied before candidates are picked (``TwoStageIndex.scores``), so it removes songs without shortening lists.

    python -m soundalike.anysong.api ... --scene scene.npz --core-ids core_ids.npy
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import torch

from .scene import SceneTable


class TierGate:
    """Row masks per channel index: core rows plus the rows of the seed's eligible artists."""

    def __init__(self, core_ids: np.ndarray, scene: SceneTable, indexes: Mapping[str, object]):
        core = np.unique(np.asarray(core_ids, dtype=np.int64))
        self.scene = scene
        self.rows: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.sizes: dict[str, dict[str, int]] = {}
        for name, ix in indexes.items():
            ids = np.asarray(ix.ids, dtype=np.int64)
            artist = np.zeros(len(ids), dtype=np.int64)
            if len(scene.track_ids):
                at = np.searchsorted(scene.track_ids, ids).clip(max=len(scene.track_ids) - 1)
                found = scene.track_ids[at] == ids
                artist[found] = scene.track_artist[at[found]]
            is_core = np.isin(ids, core)
            device = ix.id_tensor.device
            self.rows[name] = (torch.as_tensor(is_core, device=device), torch.as_tensor(artist, device=device))
            self.sizes[name] = {"core": int(is_core.sum()), "discovered": int(len(ids) - is_core.sum())}
        if scene.rel_ptr is None and any(s["discovered"] for s in self.sizes.values()):
            raise ValueError("the scene table has no related lists; rebuild it with anysong.scene")

    @classmethod
    def load(cls, path: str | Path, scene: SceneTable, indexes: Mapping[str, object]) -> TierGate:
        return cls(np.load(path), scene, indexes)

    def artists(self, artist_id: int | None) -> list[int]:
        """Artists whose discovered songs a seed by ``artist_id`` may get: the artist and its own related list in the
        scene table. Empty for an unknown artist."""
        if not artist_id:
            return []
        return sorted({int(artist_id), *(int(a) for a in self.scene.related(artist_id) or () if a)})

    def mask(self, channel: str, artists: Iterable[int]) -> torch.Tensor:
        """True for the channel's core rows and the rows of ``artists``."""
        core, artist = self.rows[channel]
        wanted = sorted({int(a) for a in artists if a})
        if not wanted:
            return core
        return core | torch.isin(artist, torch.tensor(wanted, dtype=artist.dtype, device=artist.device))

    def masks(self, artists: Iterable[int]) -> dict[str, torch.Tensor]:
        artists = list(artists)
        return {name: self.mask(name, artists) for name in self.rows}
