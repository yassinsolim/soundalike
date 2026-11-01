"""Audio decoding helpers for 30 s preview clips."""

from __future__ import annotations

from pathlib import Path

import numpy as np

CLIP_SECONDS = 30.0


def load_clip(path: str | Path, sr: int, seconds: float = CLIP_SECONDS) -> np.ndarray:
    """Decode an MP3/WAV to mono float32 at ``sr`` and repeat-pad/crop to ``seconds``."""
    return load_clips(path, (sr,), seconds)[sr]


def load_clips(path: str | Path, rates: tuple[int, ...] | list[int],
               seconds: float = CLIP_SECONDS) -> dict[int, np.ndarray]:
    """Decode once, then resample to each rate; each clip is identical to ``load_clip`` at that rate."""
    mono, native_sr = decode_mono(path)
    return fit_clips(mono, native_sr, rates, seconds)


def decode_mono(path: str | Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    data, native_sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if len(mono) == 0:
        raise ValueError(f"Empty audio: {path}")
    if not np.isfinite(mono).all():
        raise ValueError(f"Non-finite audio: {path}")
    return mono, native_sr


def fit_clips(mono: np.ndarray, native_sr: int, rates: tuple[int, ...] | list[int],
              seconds: float = CLIP_SECONDS) -> dict[int, np.ndarray]:
    """Resample mono audio to each rate and repeat-pad/crop it to ``seconds``."""
    import soxr

    if len(mono) == 0:
        raise ValueError("empty audio")
    out = {}
    for sr in rates:
        x = soxr.resample(mono, native_sr, sr, quality="HQ") if native_sr != sr else mono
        n = int(round(seconds * sr))
        if len(x) < n:
            x = np.tile(x, int(np.ceil(n / len(x))))
        out[sr] = np.ascontiguousarray(x[:n], dtype=np.float32)
    return out


def windows(clip: np.ndarray, sr: int, win_s: float, hop_s: float | None = None) -> np.ndarray:
    """Split a clip into fixed windows -> [W, win_samples]."""
    w = int(round(win_s * sr))
    h = int(round((hop_s or win_s) * sr))
    starts = list(range(0, max(len(clip) - w, 0) + 1, h)) or [0]
    return np.stack([clip[s:s + w] for s in starts])
