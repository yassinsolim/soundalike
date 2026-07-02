"""Acoustic feature extraction from raw audio (digital signal processing).

This is the "science" core: instead of trusting anyone's precomputed numbers
(Spotify, a dataset, a website), we compute features directly from the audio
waveform of a track using librosa. Two songs are "similar" when these measured
acoustic vectors are close.

Everything here operates on a short (~30s) audio clip, which is all a preview
gives us — plenty to characterize tempo, energy, brightness and timbre.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

# Number of MFCC coefficients we keep (timbre fingerprint). 13 is the classic
# choice in music information retrieval.
N_MFCC = 13

# The canonical, ordered names of every dimension in an acoustic vector.
# Keep this in sync with AcousticFeatures.vector().
FEATURE_NAMES: List[str] = (
    ["tempo", "rms_energy", "spectral_centroid", "spectral_rolloff",
     "spectral_bandwidth", "zero_crossing_rate", "spectral_contrast"]
    + [f"mfcc_{i}" for i in range(1, N_MFCC + 1)]
)

# Human-readable notes on what each non-MFCC feature captures.
FEATURE_DESCRIPTIONS = {
    "tempo": "beats per minute (speed)",
    "rms_energy": "loudness / intensity",
    "spectral_centroid": "brightness (where spectral energy is centered)",
    "spectral_rolloff": "high-frequency content",
    "spectral_bandwidth": "spread of frequencies",
    "zero_crossing_rate": "noisiness / percussiveness",
    "spectral_contrast": "difference between peaks and valleys in the spectrum",
}

SAMPLE_RATE = 22050
CLIP_SECONDS = 30


@dataclass
class AcousticFeatures:
    """A measured acoustic fingerprint of one track."""

    tempo: float
    rms_energy: float
    spectral_centroid: float
    spectral_rolloff: float
    spectral_bandwidth: float
    zero_crossing_rate: float
    spectral_contrast: float
    mfcc: List[float]

    def vector(self) -> np.ndarray:
        """Return the fixed-order feature vector (see FEATURE_NAMES)."""
        return np.array(
            [
                self.tempo,
                self.rms_energy,
                self.spectral_centroid,
                self.spectral_rolloff,
                self.spectral_bandwidth,
                self.zero_crossing_rate,
                self.spectral_contrast,
                *self.mfcc,
            ],
            dtype=float,
        )

    def to_dict(self) -> dict:
        return {
            "tempo": self.tempo,
            "rms_energy": self.rms_energy,
            "spectral_centroid": self.spectral_centroid,
            "spectral_rolloff": self.spectral_rolloff,
            "spectral_bandwidth": self.spectral_bandwidth,
            "zero_crossing_rate": self.zero_crossing_rate,
            "spectral_contrast": self.spectral_contrast,
            "mfcc": list(self.mfcc),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AcousticFeatures":
        return cls(
            tempo=float(data["tempo"]),
            rms_energy=float(data["rms_energy"]),
            spectral_centroid=float(data["spectral_centroid"]),
            spectral_rolloff=float(data["spectral_rolloff"]),
            spectral_bandwidth=float(data["spectral_bandwidth"]),
            zero_crossing_rate=float(data["zero_crossing_rate"]),
            spectral_contrast=float(data["spectral_contrast"]),
            mfcc=[float(x) for x in data["mfcc"]],
        )


def features_from_signal(y: np.ndarray, sr: int) -> AcousticFeatures:
    """Compute acoustic features from a mono audio signal.

    Separated from file loading so it can be unit-tested on synthetic signals
    without any audio files or network access.
    """
    import librosa  # imported lazily: heavy dependency, only needed here

    if y.size == 0:
        raise ValueError("Empty audio signal.")

    tempo = librosa.beat.beat_track(y=y, sr=sr)[0]
    tempo = float(np.atleast_1d(tempo)[0])

    rms = float(np.mean(librosa.feature.rms(y=y)))
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    contrast = float(np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)))
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC), axis=1)

    return AcousticFeatures(
        tempo=tempo,
        rms_energy=rms,
        spectral_centroid=centroid,
        spectral_rolloff=rolloff,
        spectral_bandwidth=bandwidth,
        zero_crossing_rate=zcr,
        spectral_contrast=contrast,
        mfcc=[float(x) for x in mfcc],
    )


def features_from_file(path: str) -> AcousticFeatures:
    """Load an audio file (e.g. an MP3 preview) and compute its features."""
    import librosa

    y, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=CLIP_SECONDS)
    return features_from_signal(y, sr)
