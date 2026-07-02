"""Harvest-once, re-embed-forever cache of song mel-spectrograms.

Building a deep-vibe library couples two expensive-but-independent steps:
downloading each preview from Deezer (rate-limited, slow) and embedding it with a
neural encoder (fast, but tied to a specific model). Every time we train a new
encoder we would otherwise have to re-download the whole library just to re-embed
it — wasteful and rate-limit bound.

This module splits those steps. It downloads each preview *once* and caches the
raw mel-spectrogram plus the hand-crafted vibe vector keyed by track id. After
that, embedding the whole library with any encoder is a local, offline,
seconds-long operation — so comparing two models on an identical song set (or
swapping in a freshly trained encoder) costs nothing extra.

    # 1) harvest audio once (rate-limited, resumable, checkpointed)
    python -m soundalike.ml.spec_cache harvest --cache ml_data/spec_cache.npz

    # 2) re-embed with any encoder, as often as you like (fast, offline)
    python -m soundalike.ml.spec_cache build \
        --cache ml_data/spec_cache.npz --model-dir ml_data/model_vibe \
        --out ml_data/deepvibe_vibeaware.npz
"""

from __future__ import annotations

import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable, List, Optional

import numpy as np
import requests

from ..audio.previews import DeezerClient
from ..audio.vibe import vibe_from_file
from .spectrogram import SpectrogramConfig, _fit_frames, load_audio, log_mel_full


class SpecCache:
    """Per-track mel-spectrogram + vibe vector, keyed by Deezer track id."""

    def __init__(self, track_ids=None, titles=None, artists=None, specs=None, vibe=None):
        track_ids = [] if track_ids is None else track_ids
        self.track_ids: List[int] = [int(t) for t in track_ids]
        self.titles: List[str] = list([] if titles is None else titles)
        self.artists: List[str] = list([] if artists is None else artists)
        # specs stored float16 to keep the cache small (128*512*2 ~= 131KB/track).
        self.specs: List[np.ndarray] = list([] if specs is None else specs)
        self.vibe: List[np.ndarray] = list([] if vibe is None else vibe)
        self._have = set(self.track_ids)

    def __len__(self) -> int:
        return len(self.track_ids)

    def has(self, track_id: int) -> bool:
        return int(track_id) in self._have

    def add(self, track_id: int, title: str, artist: str, spec: np.ndarray, vibe: np.ndarray) -> None:
        if self.has(track_id):
            return
        self.track_ids.append(int(track_id))
        self.titles.append(title)
        self.artists.append(artist)
        self.specs.append(spec.astype(np.float16))
        self.vibe.append(vibe.astype(np.float32))
        self._have.add(int(track_id))

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            track_ids=np.asarray(self.track_ids, dtype=np.int64),
            titles=np.asarray(self.titles, dtype=str),
            artists=np.asarray(self.artists, dtype=str),
            specs=np.asarray(self.specs, dtype=np.float16),
            vibe=np.asarray(self.vibe, dtype=np.float32),
        )

    @classmethod
    def load(cls, path: Path) -> "SpecCache":
        d = np.load(Path(path), allow_pickle=True)
        return cls(
            track_ids=d["track_ids"], titles=d["titles"], artists=d["artists"],
            specs=list(d["specs"]), vibe=list(d["vibe"]),
        )


def _artist_id(session: requests.Session, name: str) -> Optional[int]:
    try:
        r = session.get("https://api.deezer.com/search/artist",
                        params={"q": name, "limit": 1}, timeout=20).json()
        data = r.get("data") or []
        return int(data[0]["id"]) if data else None
    except Exception:  # noqa: BLE001
        return None


def harvest_to_cache(
    cache_path: Path,
    seed_artists: List[str],
    per_artist: int = 12,
    related_per_seed: int = 6,
    checkpoint_every: int = 50,
    progress: Callable[[str], None] = print,
) -> SpecCache:
    """Download previews for a curated artist set and cache spec + vibe per track.

    Resumable: reloads an existing cache and skips tracks already present, so a
    rate-limit interruption never loses work and never re-downloads.
    """
    client = DeezerClient()
    session = requests.Session()
    cfg = SpectrogramConfig()

    cache = SpecCache.load(cache_path) if Path(cache_path).exists() else SpecCache()
    if len(cache):
        progress(f"Loaded existing cache: {len(cache)} tracks")

    progress(f"Resolving {len(seed_artists)} seed artists...")
    artist_ids: set = set()
    for name in seed_artists:
        aid = _artist_id(session, name)
        if not aid:
            continue
        artist_ids.add(aid)
        for rid in client.related_artists(aid, related_per_seed):
            artist_ids.add(int(rid))
        time.sleep(0.1)
    progress(f"Expanded to {len(artist_ids)} artists (incl. related).")

    candidates = {}
    for aid in artist_ids:
        try:
            for t in client.artist_top_tracks(aid, per_artist):
                if t.has_preview and not cache.has(t.id) and t.id not in candidates:
                    candidates[t.id] = t
        except Exception:  # noqa: BLE001
            continue
    progress(f"{len(candidates)} new candidate tracks. Harvesting...")

    t0 = time.time()
    done = 0
    with TemporaryDirectory() as tmp:
        wd = Path(tmp)
        for i, track in enumerate(candidates.values(), 1):
            try:
                dest = wd / f"{track.id}.mp3"
                if client.download_preview(track, dest) is None:
                    continue
                y = load_audio(dest, cfg.sample_rate)
                spec = _fit_frames(log_mel_full(y, cfg), cfg.target_frames)
                vfeat = vibe_from_file(str(dest))
                dest.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                continue
            cache.add(track.id, track.title, track.artist, spec, vfeat.vector())
            done += 1
            if done % checkpoint_every == 0:
                cache.save(cache_path)
                progress(f"  {i}/{len(candidates)} ({done/(time.time()-t0):.1f}/s) "
                         f"[cache: {len(cache)} tracks]")
    cache.save(cache_path)
    progress(f"Done. Cache now {len(cache)} tracks -> {cache_path}")
    return cache


def build_index_from_cache(cache_path: Path, model_dir: Path,
                           progress: Callable[[str], None] = print):
    """Embed every cached spec with an encoder -> DeepVibeIndex (fast, offline)."""
    from .deepvibe import DeepVibeIndex
    from .encoder_infer import EncoderExtractor

    cache = SpecCache.load(cache_path)
    progress(f"Embedding {len(cache)} cached specs with {model_dir}...")
    extractor = EncoderExtractor(model_dir)
    specs = np.asarray(cache.specs, dtype=np.float32)
    neural = extractor.embed_specs(specs, batch=256)
    idx = DeepVibeIndex(
        cache.track_ids, cache.titles, cache.artists,
        neural, np.asarray(cache.vibe, dtype=np.float32),
    )
    progress(f"Built deep-vibe index: {len(idx)} tracks")
    return idx


def main(argv: Optional[list] = None) -> int:
    import argparse

    from .grow_deepvibe import SEED_ARTISTS

    parser = argparse.ArgumentParser(description="Harvest-once spec cache for deep-vibe.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ph = sub.add_parser("harvest", help="Download previews and cache spec+vibe.")
    ph.add_argument("--cache", default="ml_data/spec_cache.npz")
    ph.add_argument("--per-artist", type=int, default=12)

    pb = sub.add_parser("build", help="Embed cached specs with an encoder -> index.")
    pb.add_argument("--cache", default="ml_data/spec_cache.npz")
    pb.add_argument("--model-dir", default="ml_data/model_vibe")
    pb.add_argument("--out", default="ml_data/deepvibe_vibeaware.npz")

    args = parser.parse_args(argv)
    if args.cmd == "harvest":
        harvest_to_cache(Path(args.cache), SEED_ARTISTS, per_artist=args.per_artist)
    elif args.cmd == "build":
        idx = build_index_from_cache(Path(args.cache), Path(args.model_dir))
        idx.save(Path(args.out))
        print(f"Saved -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
