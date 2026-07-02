"""Grow a deep-vibe library from curated seed artists, with checkpointing.

Genre charts skew mainstream, so a niche seed song (hyperpop, underground
electronic) has no close neighbours in a genre-harvested library. This script
expands the library from a curated list of artists (plus their Deezer
"related" artists), computing both the neural embedding and the vibe features,
and saves a checkpoint every N tracks so rate-limit interruptions never lose
progress.

Usage:
    python -m soundalike.ml.grow_deepvibe --index ml_data/deepvibe_index.npz
"""

from __future__ import annotations

import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional

import numpy as np
import requests

from ..audio.previews import DeezerClient
from ..audio.vibe import vibe_from_file
from .deepvibe import DeepVibeIndex
from .encoder_infer import EncoderExtractor
from .spectrogram import SpectrogramConfig, _fit_frames, load_audio, log_mel_full

# Curated artists spanning hyperpop, underground/electronic, bass, rage and
# adjacent scenes — the sonic neighbourhoods genre charts miss.
SEED_ARTISTS: List[str] = [
    "ericdoa", "glaive", "midwxst", "aldn", "brakence", "quannnic", "d0llywood1",
    "jane remover", "underscores", "f1lthy", "100 gecs", "osquinn", "kmoe",
    "twikipedia", "5v", "vaeo", "angelus", "Sematary", "blackwinterwells",
    "Skrillex", "Flume", "Porter Robinson", "ODESZA", "Illenium", "Sub Urban",
    "Madeon", "San Holo", "Rezz", "What So Not", "Slushii", "Kotori",
    "Yeat", "Ken Carson", "Playboi Carti", "Destroy Lonely", "Juice WRLD",
    "The Kid LAROI", "iann dior", "Trippie Redd", "Lil Uzi Vert",
    "PinkPantheress", "Charli xcx", "SOPHIE", "A. G. Cook", "Kero Kero Bonito",
    "Nessa Barrett", "d4vd", "Clairo", "beabadoobee", "The Marías", "Cage The Elephant",
]


def _artist_id(session: requests.Session, name: str) -> Optional[int]:
    try:
        r = session.get("https://api.deezer.com/search/artist",
                        params={"q": name, "limit": 1}, timeout=20).json()
        data = r.get("data") or []
        return int(data[0]["id"]) if data else None
    except Exception:  # noqa: BLE001
        return None


def grow(
    index_path: Path,
    model_dir: Path,
    per_artist: int = 12,
    related_per_seed: int = 6,
    checkpoint_every: int = 100,
    progress=print,
) -> DeepVibeIndex:
    client = DeezerClient()
    session = requests.Session()
    extractor = EncoderExtractor(model_dir)
    cfg = SpectrogramConfig()

    # Start from the existing library if present.
    if index_path.exists():
        idx = DeepVibeIndex.load(index_path)
        ids = list(int(t) for t in idx.track_ids)
        titles = list(idx.titles); artists = list(idx.artists)
        neural = list(idx.neural); vibe = list(idx.vibe)
        progress(f"Loaded existing library: {len(ids)} tracks")
    else:
        ids, titles, artists, neural, vibe = [], [], [], [], []
    have = set(ids)

    # Gather candidate track ids from seed artists + their related artists.
    progress(f"Resolving {len(SEED_ARTISTS)} seed artists...")
    artist_ids: set = set()
    for name in SEED_ARTISTS:
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
                if t.has_preview and t.id not in have and t.id not in candidates:
                    candidates[t.id] = t
        except Exception:  # noqa: BLE001
            continue
    progress(f"{len(candidates)} new candidate tracks. Embedding...")

    def _save():
        DeepVibeIndex(ids, titles, artists, np.array(neural, np.float32),
                      np.array(vibe, np.float32)).save(index_path)

    t0 = time.time()
    with TemporaryDirectory() as tmp:
        wd = Path(tmp)
        for i, track in enumerate(candidates.values(), 1):
            try:
                dest = wd / f"{track.id}.mp3"
                if client.download_preview(track, dest) is None:
                    continue
                y = load_audio(dest, cfg.sample_rate)
                spec = _fit_frames(log_mel_full(y, cfg), cfg.target_frames)
                nvec = extractor.embed_spec(spec)
                vfeat = vibe_from_file(str(dest))
                dest.unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                continue
            ids.append(int(track.id)); titles.append(track.title); artists.append(track.artist)
            neural.append(nvec); vibe.append(vfeat.vector())
            if i % checkpoint_every == 0:
                _save()
                progress(f"  {i}/{len(candidates)} ({i/(time.time()-t0):.1f}/s) "
                         f"[checkpoint: {len(ids)} total]")
    _save()
    progress(f"Done. Library now {len(ids)} tracks -> {index_path}")
    return DeepVibeIndex(ids, titles, artists, np.array(neural, np.float32),
                         np.array(vibe, np.float32))


def main(argv: Optional[list] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Grow the deep-vibe library from seed artists.")
    parser.add_argument("--index", default="ml_data/deepvibe_index.npz")
    parser.add_argument("--model-dir", default="ml_data/model_fma_large")
    parser.add_argument("--per-artist", type=int, default=12)
    args = parser.parse_args(argv)
    grow(Path(args.index), Path(args.model_dir), per_artist=args.per_artist)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
