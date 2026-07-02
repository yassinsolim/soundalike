"""Compute a "vibe target" vector from a mel-spectrogram.

The learned encoder is trained (in train_vibe) to predict this vector from a
short crop, which forces its embedding to encode the qualities that define a
song's vibe — its frequency-band balance and its dynamics — rather than just
genre/timbre. Crucially this is computed straight from the packed FMA
mel-spectrograms, so a vibe-aware model can be trained on all 106k songs with no
re-downloading.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

# Frequency bands (Hz), matching the hand-crafted vibe engine.
_BANDS_HZ = [(20, 60), (60, 250), (250, 500), (500, 2000),
             (2000, 4000), (4000, 6000), (6000, 11025)]

# Target layout: 7 band fractions + dyn_std + dyn_range + centroid(kHz).
VIBE_TARGET_DIM = 10


@lru_cache(maxsize=4)
def _band_bins(n_mels: int, sr: int) -> tuple:
    import librosa

    mel_hz = librosa.mel_frequencies(n_mels=n_mels, fmin=0, fmax=sr / 2)
    bins = tuple(
        np.where((mel_hz >= lo) & (mel_hz < hi))[0] for lo, hi in _BANDS_HZ
    )
    return mel_hz, bins


def vibe_target_from_mel(spec: np.ndarray, sr: int = 22050) -> np.ndarray:
    """spec: (n_mels, frames) normalized log-mel -> (10,) vibe target."""
    spec = spec.astype(np.float32)
    n_mels = spec.shape[0]
    mel_hz, bins = _band_bins(n_mels, sr)

    energy = np.exp(spec)  # back toward relative linear energy
    band = np.array([energy[b, :].mean() if len(b) else 0.0 for b in bins], dtype=np.float32)
    band = band / (band.sum() + 1e-9)

    frame_level = spec.mean(axis=0)  # per-frame loudness proxy
    dyn_std = float(frame_level.std())
    dyn_range = float(np.percentile(frame_level, 95) - np.percentile(frame_level, 10))

    centroid = float((mel_hz[:, None] * energy).sum(0).mean() / (energy.sum(0).mean() + 1e-9))

    return np.concatenate([band, [dyn_std, dyn_range, centroid / 1000.0]]).astype(np.float32)


def vibe_targets_for_batch(specs: np.ndarray, sr: int = 22050) -> np.ndarray:
    """(N, n_mels, frames) -> (N, 10) vibe targets."""
    return np.stack([vibe_target_from_mel(s, sr) for s in specs]).astype(np.float32)
