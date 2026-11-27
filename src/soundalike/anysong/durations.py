"""Song lengths of the catalog, for saying where a preview sits in its song (``api.preview_span``).

The API knows a song's length from its Deezer track object. A song outside the catalog always comes with one; an
indexed song is ranked from its stored windows without asking Deezer, so its length is only known when a search
happened to return it in the last few minutes. This table (``durations.npz``: ``ids`` sorted int64, ``seconds``
uint16) holds the length Deezer gave when the catalog was collected, so that no request is needed:

    python -m soundalike.anysong.durations --ids index/sonic/ids.npy --out index/durations.npz \
        --tracks crawl/export/deezer_tracks.jsonl previews/tracks.jsonl

``--tracks``: JSON-lines files with one Deezer track per line, its id as ``track_id`` or ``id`` and its length in
seconds as ``duration``: what ``anysong.crawl export`` and ``anysong.previews --save-meta`` write. Earlier files
win; songs outside ``--ids`` and lengths that are missing or not positive are left out.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np

MAX_SECONDS = 65535  # uint16; longer tracks (18 hours) are stored as this


class DurationTable:
    """Length in seconds by track id (binary search over sorted ids: 10 bytes a song)."""

    def __init__(self, ids: np.ndarray, seconds: np.ndarray):
        ids, seconds = np.asarray(ids, dtype=np.int64), np.asarray(seconds, dtype=np.uint16)
        if ids.shape != seconds.shape or ids.ndim != 1:
            raise ValueError("duration table ids and seconds differ in length")
        if len(ids) > 1 and not (ids[1:] > ids[:-1]).all():
            raise ValueError("duration table ids must be sorted and distinct")
        self.ids, self.values = ids, seconds

    @classmethod
    def load(cls, path: str | Path) -> DurationTable:
        with np.load(path, allow_pickle=False) as z:
            return cls(z["ids"], z["seconds"])

    def __len__(self) -> int:
        return len(self.ids)

    def seconds(self, track_id: int) -> int | None:
        """The song's length, None when the table does not have it."""
        at = int(np.searchsorted(self.ids, int(track_id)))
        if at == len(self.ids) or int(self.ids[at]) != int(track_id) or not self.values[at]:
            return None
        return int(self.values[at])


def save(path: str | Path, seconds: Mapping[int, int | float]) -> int:
    """Write ``durations.npz`` from track id -> seconds (non-positive lengths left out); the number of songs."""
    known = sorted((int(tid), min(int(round(s)), MAX_SECONDS)) for tid, s in seconds.items() if s and s > 0)
    ids = np.array([tid for tid, _ in known], dtype=np.int64)
    np.savez(path, ids=ids, seconds=np.array([s for _, s in known], dtype=np.uint16))
    return len(ids)


def read_tracks(paths: Iterable[str | Path], wanted: set[int] | None = None) -> dict[int, int]:
    """Track id -> seconds from JSON-lines track files (see the module text); earlier files win."""
    seconds: dict[int, int] = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                tid, duration = row.get("track_id", row.get("id")), row.get("duration")
                if type(tid) is not int or tid in seconds or (wanted is not None and tid not in wanted):
                    continue
                if isinstance(duration, (int, float)) and not isinstance(duration, bool) and duration > 0:
                    seconds[tid] = int(round(duration))
    return seconds


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="durations", description=__doc__.split("\n")[0])
    p.add_argument("--tracks", type=Path, nargs="+", required=True, help="JSON-lines track files, best first")
    p.add_argument("--ids", type=Path, default=None, help="an index's ids.npy: keep only its songs")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    ids = None if args.ids is None else np.load(args.ids, allow_pickle=False)
    seconds = read_tracks(args.tracks, None if ids is None else set(ids.tolist()))
    count = save(args.out, seconds)
    print(json.dumps({"out": str(args.out), "songs": count, "indexed": None if ids is None else len(ids),
                      "share": None if ids is None or not len(ids) else round(count / len(ids), 4),
                      "under_90_s": sum(s < 90 for s in seconds.values())}))


if __name__ == "__main__":
    main()
