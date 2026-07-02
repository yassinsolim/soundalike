"""Turn preview audio into mel-spectrograms and augmented views.

A mel-spectrogram is a time-frequency image of the audio: the x-axis is time,
the y-axis is (mel-scaled) frequency, and pixel intensity is energy. It's the
standard input representation for music CNNs — it exposes rhythm, timbre and
harmony in a form a convolutional network can learn from.

For self-supervised contrastive training we need two *different* views of the
same clip that should still map to nearby embeddings; the augmentations here
(time/frequency masking, small crops, gain jitter) provide that.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 1024
HOP = 512
CLIP_SECONDS = 30
# Fixed number of time frames per view (~6s at hop 512 / sr 22050).
TARGET_FRAMES = 256


@dataclass
class SpectrogramConfig:
    sample_rate: int = SAMPLE_RATE
    n_mels: int = N_MELS
    n_fft: int = N_FFT
    hop: int = HOP
    target_frames: int = TARGET_FRAMES


def load_audio(path: str | Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load a preview file to a mono waveform at the target sample rate."""
    import librosa

    y, _ = librosa.load(str(path), sr=sr, mono=True, duration=CLIP_SECONDS)
    return y.astype(np.float32)


def mel_spectrogram(y: np.ndarray, cfg: Optional[SpectrogramConfig] = None) -> np.ndarray:
    """Compute a log-mel-spectrogram, returned as a (n_mels, target_frames) array."""
    cfg = cfg or SpectrogramConfig()
    return _fit_frames(log_mel_full(y, cfg), cfg.target_frames)


def log_mel_full(y: np.ndarray, cfg: Optional[SpectrogramConfig] = None) -> np.ndarray:
    """Full-length normalized log-mel-spectrogram (no time cropping).

    Kept full so training can sample *different* time windows of the same song
    as contrastive views. Padded up to at least `target_frames`.
    """
    import librosa

    cfg = cfg or SpectrogramConfig()
    if y.size == 0:
        raise ValueError("Empty audio signal.")
    mel = librosa.feature.melspectrogram(
        y=y, sr=cfg.sample_rate, n_fft=cfg.n_fft, hop_length=cfg.hop, n_mels=cfg.n_mels
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-6)
    if log_mel.shape[1] < cfg.target_frames:
        pad = cfg.target_frames - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad)), mode="constant",
                         constant_values=log_mel.min())
    return log_mel


def random_crop(spec: np.ndarray, frames: int, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Take a random time window of `frames` columns from a spectrogram."""
    rng = rng or np.random.default_rng()
    total = spec.shape[1]
    if total <= frames:
        return _fit_frames(spec, frames)
    start = int(rng.integers(0, total - frames + 1))
    return spec[:, start : start + frames]


def _fit_frames(spec: np.ndarray, target: int) -> np.ndarray:
    frames = spec.shape[1]
    if frames == target:
        return spec
    if frames > target:
        start = (frames - target) // 2
        return spec[:, start : start + target]
    pad = target - frames
    return np.pad(spec, ((0, 0), (0, pad)), mode="constant", constant_values=spec.min())


def augment(spec: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Produce a randomly augmented view of a spectrogram (SpecAugment-style)."""
    rng = rng or np.random.default_rng()
    out = spec.copy()
    n_mels, frames = out.shape
    fill = out.min()

    # Frequency masking.
    if rng.random() < 0.8:
        f = int(rng.integers(1, max(2, n_mels // 8)))
        f0 = int(rng.integers(0, n_mels - f))
        out[f0 : f0 + f, :] = fill
    # Time masking.
    if rng.random() < 0.8:
        t = int(rng.integers(1, max(2, frames // 8)))
        t0 = int(rng.integers(0, frames - t))
        out[:, t0 : t0 + t] = fill
    # Small gain jitter.
    out = out * float(rng.uniform(0.9, 1.1))
    return out.astype(np.float32)


def two_views(spec: np.ndarray, rng: Optional[np.random.Generator] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Two independent augmented views for contrastive learning."""
    rng = rng or np.random.default_rng()
    return augment(spec, rng), augment(spec, rng)
