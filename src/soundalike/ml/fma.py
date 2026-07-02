"""Loader for the Free Music Archive (FMA) dataset — the scale-up path.

Deezer's free API hard-caps us at a few hundred tracks per session, which is far
too few for self-supervised deep learning. FMA is the standard open benchmark for
exactly this problem:

  * FMA-small : 8,000 tracks, 8 balanced genres, 30s clips (~7.2 GB)
  * FMA-medium: 25,000 tracks, 16 genres (~22 GB)

Download (see https://github.com/mdeff/fma):
  * audio    : https://os.unil.cloud.switch.ch/fma/fma_small.zip
  * metadata : https://os.unil.cloud.switch.ch/fma/fma_metadata.zip

Unzip both, then point --audio-dir at the fma_small folder and --metadata at
fma_metadata/tracks.csv. This module turns that into the same manifest +
spectrogram-cache format the trainer already understands, so no other code
changes are needed to train at scale.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from .collect import TrackEntry


def _track_id_to_path(audio_dir: Path, track_id: int) -> Path:
    # FMA stores audio as <audio_dir>/<zero-padded id / 1000>/<zero-padded id>.mp3
    tid = f"{track_id:06d}"
    return audio_dir / tid[:3] / f"{tid}.mp3"


def load_fma_manifest(
    audio_dir: str | Path,
    tracks_csv: str | Path,
    subset: str = "small",
    max_tracks: Optional[int] = None,
    include_unlabeled: bool = False,
) -> List[TrackEntry]:
    """Build TrackEntry rows from FMA metadata, using local file paths as previews.

    The `preview_url` field is repurposed to hold the local mp3 path so the
    existing precompute step (which just needs a readable audio source) works
    unchanged — see `precompute_from_paths` below for the local variant.

    If `include_unlabeled` is True, tracks with no `genre_top` are kept with the
    genre "unlabeled" — useful for self-supervised training on FMA-large, where
    ~54% of tracks are unlabeled. Evaluation code filters these out.
    """
    import pandas as pd

    audio_dir = Path(audio_dir)
    tracks = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])
    subset_col = tracks[("set", "subset")]
    genre_col = tracks[("track", "genre_top")]
    title_col = tracks[("track", "title")]
    artist_col = tracks[("artist", "name")]

    order = {"small": 0, "medium": 1, "large": 2}
    keep = order.get(subset, 0)

    entries: List[TrackEntry] = []
    for track_id, subset_val in subset_col.items():
        if order.get(str(subset_val), 99) > keep:
            continue
        genre = genre_col.get(track_id)
        if not isinstance(genre, str) or not genre:
            if not include_unlabeled:
                continue
            genre = "unlabeled"
        path = _track_id_to_path(audio_dir, int(track_id))
        if not path.exists():
            continue
        entries.append(
            TrackEntry(
                track_id=int(track_id),
                title=str(title_col.get(track_id, "")),
                artist=str(artist_col.get(track_id, "")),
                genre=genre,
                preview_url=str(path),  # local path, consumed by precompute_from_paths
            )
        )
        if max_tracks and len(entries) >= max_tracks:
            break
    return entries


def precompute_from_paths(entries: List[TrackEntry], spec_dir: Path, progress=print) -> int:
    """Compute + cache spectrograms directly from local audio files (no download)."""
    import numpy as np

    from .data import _spec_path
    from .spectrogram import SpectrogramConfig, load_audio, log_mel_full

    cfg = SpectrogramConfig()
    spec_dir = Path(spec_dir)
    spec_dir.mkdir(parents=True, exist_ok=True)
    available = 0
    for i, entry in enumerate(entries):
        out = _spec_path(spec_dir, entry.track_id)
        if out.exists():
            available += 1
            continue
        try:
            spec = log_mel_full(load_audio(entry.preview_url, cfg.sample_rate), cfg)
            np.save(out, spec)
            available += 1
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            progress(f"  skip {entry.title}: {type(exc).__name__}")
        if (i + 1) % 250 == 0:
            progress(f"  prepared {i + 1}/{len(entries)} ({available} cached)")
    return available


def main(argv: Optional[list] = None) -> int:
    import argparse

    from .collect import DatasetCollector

    parser = argparse.ArgumentParser(description="Build a manifest from a local FMA dataset.")
    parser.add_argument("--audio-dir", required=True, help="Path to fma_small/ (or fma_medium/).")
    parser.add_argument("--metadata", required=True, help="Path to fma_metadata/tracks.csv.")
    parser.add_argument("--subset", default="small", choices=["small", "medium", "large"])
    parser.add_argument("--out", default="ml_data/fma_manifest.csv")
    parser.add_argument("--max-tracks", type=int, default=None)
    parser.add_argument("--include-unlabeled", action="store_true",
                        help="Keep tracks without a genre label (genre='unlabeled').")
    args = parser.parse_args(argv)

    entries = load_fma_manifest(args.audio_dir, args.metadata, args.subset, args.max_tracks,
                                include_unlabeled=args.include_unlabeled)
    DatasetCollector.save(entries, Path(args.out))
    genres = sorted({e.genre for e in entries})
    print(f"Wrote {len(entries)} FMA tracks to {args.out}")
    print(f"Genres ({len(genres)}): {', '.join(genres)}")
    print("Next: cache spectrograms with soundalike.ml.fma.precompute_from_paths, then train.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
