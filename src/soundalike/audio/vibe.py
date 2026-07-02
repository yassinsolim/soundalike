"""Rich "vibe" features that capture how a song actually sounds and feels.

The original acoustic engine averaged every feature over the whole clip, which
washes out exactly what makes a track's vibe: its bass profile and its dynamics.
A song with quiet verses and a heavy drop ends up looking "medium" everywhere.

This module fixes that by measuring two things the flat averages miss:

1. **Frequency-band balance** — how the energy splits across sub / bass / low-mid
   / mid / high-mid / presence / air. This is the literal "how much sub-bass,
   how much highs" of a track. A dubstep drop lives in the sub/bass bands; an
   airy acoustic song lives up top.

2. **Dynamics** — how much the loudness *moves* over time (standard deviation,
   dynamic range, and crest factor = peak / average). Steady soft songs barely
   move; drop-heavy songs spike hard. This is what separates a mellow track from
   one with a big drop even when their average loudness is identical.

Plus tempo, brightness and an MFCC timbre fingerprint for texture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

# Frequency bands in Hz. Names match how people talk about a mix.
BANDS_HZ: List[tuple] = [
    (20, 60),      # sub      — the felt-not-heard rumble / 808 sub
    (60, 250),     # bass     — bassline, kick body
    (250, 500),    # low-mid  — warmth, low vocals
    (500, 2000),   # mid      — vocals, most instruments
    (2000, 4000),  # high-mid — presence, attack
    (4000, 6000),  # presence — clarity, consonants
    (6000, 11000), # air      — cymbals, sparkle
]
BAND_NAMES = ["sub", "bass", "low_mid", "mid", "high_mid", "presence", "air"]

N_MFCC = 13
SAMPLE_RATE = 22050
CLIP_SECONDS = 30

# The ordered vector layout. Keep in sync with VibeFeatures.vector().
FEATURE_NAMES: List[str] = (
    ["tempo", "brightness", "rolloff", "onset_rate",
     "rms_mean", "rms_std", "dynamic_range", "crest", "low_end_ratio"]
    + [f"band_{b}" for b in BAND_NAMES]
    + [f"mfcc_{i}" for i in range(1, N_MFCC + 1)]
)

FEATURE_DESCRIPTIONS = {
    "tempo": "beats per minute",
    "brightness": "spectral centroid — dark vs bright",
    "rolloff": "high-frequency extent",
    "onset_rate": "how busy/rhythmic (onsets per second)",
    "rms_mean": "average loudness",
    "rms_std": "how much the loudness moves over time",
    "dynamic_range": "loud-section vs quiet-section gap (the 'drop' size)",
    "crest": "peak / average loudness (spikiness)",
    "low_end_ratio": "fraction of energy in sub + bass (bass heaviness)",
}


@dataclass
class VibeFeatures:
    tempo: float
    brightness: float
    rolloff: float
    onset_rate: float
    rms_mean: float
    rms_std: float
    dynamic_range: float
    crest: float
    low_end_ratio: float
    bands: List[float]          # 7 band energy fractions, summing to ~1
    mfcc: List[float]

    def vector(self) -> np.ndarray:
        return np.array(
            [self.tempo, self.brightness, self.rolloff, self.onset_rate,
             self.rms_mean, self.rms_std, self.dynamic_range, self.crest,
             self.low_end_ratio, *self.bands, *self.mfcc],
            dtype=float,
        )

    def to_dict(self) -> dict:
        return {
            "tempo": self.tempo, "brightness": self.brightness, "rolloff": self.rolloff,
            "onset_rate": self.onset_rate, "rms_mean": self.rms_mean, "rms_std": self.rms_std,
            "dynamic_range": self.dynamic_range, "crest": self.crest,
            "low_end_ratio": self.low_end_ratio, "bands": list(self.bands),
            "mfcc": list(self.mfcc),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "VibeFeatures":
        return cls(
            tempo=float(d["tempo"]), brightness=float(d["brightness"]),
            rolloff=float(d["rolloff"]), onset_rate=float(d["onset_rate"]),
            rms_mean=float(d["rms_mean"]), rms_std=float(d["rms_std"]),
            dynamic_range=float(d["dynamic_range"]), crest=float(d["crest"]),
            low_end_ratio=float(d["low_end_ratio"]), bands=[float(x) for x in d["bands"]],
            mfcc=[float(x) for x in d["mfcc"]],
        )

    def describe(self) -> Dict[str, str]:
        """A human-readable summary of the vibe (for the CLI)."""
        loud = "very dynamic (big drops)" if self.crest > 2.0 else (
            "dynamic" if self.crest > 1.7 else "steady/flat")
        bass = "bass-heavy" if self.low_end_ratio > 0.6 else (
            "balanced low-end" if self.low_end_ratio > 0.4 else "light low-end")
        bright = "bright" if self.brightness > 2500 else (
            "warm" if self.brightness > 1500 else "dark")
        return {"tempo": f"{self.tempo:.0f} BPM", "dynamics": loud, "low_end": bass,
                "tone": bright}


def vibe_from_signal(y: np.ndarray, sr: int) -> VibeFeatures:
    """Compute vibe features from a mono waveform (testable without files)."""
    import librosa

    if y.size == 0:
        raise ValueError("Empty audio signal.")

    tempo = float(np.atleast_1d(librosa.beat.beat_track(y=y, sr=sr)[0])[0])

    # Power spectrogram once, reused for bands + brightness.
    stft = np.abs(librosa.stft(y, n_fft=2048)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    band_energy = np.array(
        [stft[(freqs >= lo) & (freqs < hi)].mean() if np.any((freqs >= lo) & (freqs < hi))
         else 0.0 for lo, hi in BANDS_HZ]
    )
    total = band_energy.sum() + 1e-12
    bands = (band_energy / total).tolist()
    low_end_ratio = float(bands[0] + bands[1])  # sub + bass

    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))

    # Dynamics from the RMS energy envelope.
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = float(rms.mean())
    rms_std = float(rms.std())
    dynamic_range = float(np.percentile(rms, 95) - np.percentile(rms, 10))
    crest = float(rms.max() / (rms_mean + 1e-9))

    onset_env = librosa.onset.onset_detect(y=y, sr=sr, units="time")
    onset_rate = float(len(onset_env) / max(len(y) / sr, 1e-6))

    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC), axis=1)

    return VibeFeatures(
        tempo=tempo, brightness=centroid, rolloff=rolloff, onset_rate=onset_rate,
        rms_mean=rms_mean, rms_std=rms_std, dynamic_range=dynamic_range, crest=crest,
        low_end_ratio=low_end_ratio, bands=bands, mfcc=[float(x) for x in mfcc],
    )


def vibe_from_file(path: str) -> VibeFeatures:
    import librosa

    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=CLIP_SECONDS)
    return vibe_from_signal(y, sr)


# Default weights: emphasize the things that define "vibe" — the low-end balance
# and the dynamics — over raw timbre, so a bass-heavy drop track matches other
# bass-heavy drop tracks rather than anything that merely shares a timbre.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "low_end_ratio": 3.0,
    "crest": 2.5,
    "dynamic_range": 2.5,
    "rms_std": 2.0,
    "band_sub": 2.5,
    "band_bass": 2.0,
    "tempo": 1.5,
    "onset_rate": 1.5,
    "brightness": 1.5,
}


def weight_vector(weights: Dict[str, float]) -> np.ndarray:
    return np.array([float(weights.get(n, 1.0)) for n in FEATURE_NAMES], dtype=float)
