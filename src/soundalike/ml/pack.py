"""Consolidate per-track spectrogram .npy files into one compact array.

Random-reading 25k small files during training starved the GPU (especially on a
slow drive). This packs every spectrogram into a single fixed-size float16
tensor of shape (N, n_mels, frames) plus aligned metadata, so training loads the
whole dataset into RAM once (a few GB) and never touches disk per batch.

Each spectrogram is truncated/padded to `frames` columns (default 512), which
still leaves room to randomly crop a 256-frame window for augmentation.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from .collect import DatasetCollector
from .data import _spec_path


def _fix_width(spec: np.ndarray, frames: int) -> np.ndarray:
    n = spec.shape[1]
    if n == frames:
        return spec
    if n > frames:
        return spec[:, :frames]
    pad = frames - n
    return np.pad(spec, ((0, 0), (0, pad)), mode="constant", constant_values=spec.min())


def pack(
    manifest: Path,
    spec_dir: Path,
    out_path: Path,
    frames: int = 512,
    n_mels: int = 128,
    progress=print,
) -> Path:
    entries = DatasetCollector.load(manifest)
    have = [e for e in entries if _spec_path(Path(spec_dir), e.track_id).exists()]
    progress(f"Packing {len(have)} spectrograms -> {out_path}")

    X = np.empty((len(have), n_mels, frames), dtype=np.float16)
    track_ids, genres, titles, artists = [], [], [], []
    t0 = time.time()
    for i, e in enumerate(have):
        spec = np.load(_spec_path(Path(spec_dir), e.track_id))
        X[i] = _fix_width(spec, frames).astype(np.float16)
        track_ids.append(e.track_id); genres.append(e.genre)
        titles.append(e.title); artists.append(e.artist)
        if (i + 1) % 2000 == 0:
            rate = (i + 1) / (time.time() - t0)
            progress(f"  {i+1}/{len(have)}  ({rate:.0f}/s)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        X=X,
        track_ids=np.array(track_ids),
        genres=np.array(genres),
        titles=np.array(titles),
        artists=np.array(artists),
    )
    size_gb = out_path.stat().st_size / 1e9 if out_path.exists() else 0
    progress(f"Saved {X.shape} float16 ({size_gb:.1f} GB) in {(time.time()-t0)/60:.1f} min")
    return out_path


def main(argv: Optional[list] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Pack spectrograms into one array.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--spec-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--frames", type=int, default=512)
    args = parser.parse_args(argv)
    pack(Path(args.manifest), Path(args.spec_dir), Path(args.out), frames=args.frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
