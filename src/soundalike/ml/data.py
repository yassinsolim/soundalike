"""Dataset preparation and a torch Dataset for contrastive training.

Downloading a preview and computing its mel-spectrogram is slow, so we do it
once and cache each spectrogram as a .npy file keyed by track id. Training then
reads those cached arrays quickly and applies random augmentations on the fly.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import requests

from .collect import TrackEntry
from .spectrogram import (
    SpectrogramConfig,
    augment,
    load_audio,
    log_mel_full,
    mel_spectrogram,
    random_crop,
)


def _spec_path(spec_dir: Path, track_id: int) -> Path:
    return spec_dir / f"{track_id}.npy"


def precompute_spectrograms(
    entries: List[TrackEntry],
    spec_dir: Path,
    cfg: Optional[SpectrogramConfig] = None,
    progress=print,
) -> int:
    """Download each preview and cache its mel-spectrogram. Returns #available."""
    cfg = cfg or SpectrogramConfig()
    spec_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    available = 0
    tmp = spec_dir / "_tmp.mp3"

    for i, entry in enumerate(entries):
        out = _spec_path(spec_dir, entry.track_id)
        if out.exists():
            available += 1
            continue
        try:
            resp = session.get(entry.preview_url, timeout=30)
            resp.raise_for_status()
            tmp.write_bytes(resp.content)
            spec = log_mel_full(load_audio(tmp, cfg.sample_rate), cfg)
            np.save(out, spec)
            available += 1
        except (requests.RequestException, ValueError, RuntimeError) as exc:
            progress(f"  skip {entry.title}: {type(exc).__name__}")
        if (i + 1) % 25 == 0:
            progress(f"  prepared {i + 1}/{len(entries)} ({available} cached)")
    tmp.unlink(missing_ok=True)
    return available


class SpectrogramDataset:
    """Yields two augmented views of each cached spectrogram (for contrastive loss).

    Implemented against the torch Dataset interface lazily so importing this
    module doesn't require torch until a dataset is actually constructed.
    """

    def __init__(self, entries: List[TrackEntry], spec_dir: Path, seed: int = 0,
                 target_frames: int = 256):
        self.spec_dir = Path(spec_dir)
        self.entries = [e for e in entries if _spec_path(self.spec_dir, e.track_id).exists()]
        if not self.entries:
            raise RuntimeError("No cached spectrograms found. Run precompute first.")
        self.labels = [e.genre for e in self.entries]
        self.target_frames = target_frames
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.entries)

    def load_spec(self, index: int) -> np.ndarray:
        return np.load(_spec_path(self.spec_dir, self.entries[index].track_id))

    def __getitem__(self, index: int):
        import torch

        spec = self.load_spec(index)
        # Two DIFFERENT time windows of the same song form the positive pair;
        # each is then independently masked. This forces the encoder to learn
        # what is consistent across the whole track (its sonic character).
        v1 = augment(random_crop(spec, self.target_frames, self._rng), self._rng)
        v2 = augment(random_crop(spec, self.target_frames, self._rng), self._rng)
        t1 = torch.from_numpy(v1).unsqueeze(0)
        t2 = torch.from_numpy(v2).unsqueeze(0)
        return t1, t2, index


def all_specs_matrix(dataset: "SpectrogramDataset"):
    """Stack a deterministic center-crop of every spectrogram for embedding."""
    import torch

    from .spectrogram import _fit_frames

    specs = [_fit_frames(dataset.load_spec(i), dataset.target_frames)
             for i in range(len(dataset))]
    arr = np.stack(specs)[:, None, :, :]  # (N, 1, n_mels, frames)
    return torch.from_numpy(arr)
