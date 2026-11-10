"""Stem dissection: six stems per song (htdemucs_6s), what each one does, and per-stem grades between two songs.

The catalog build (``python -m soundalike.anysong.stems build``) and the API's live dissection share this code, so a
stored dissection and a live one agree. Per song:

- ``vec``: MuQ-MuLan embedding of each stem (the sonic channel's embedder, on the stem instead of the mix);
  ``present`` marks the stems loud enough to grade (within ``PRESENT_DB`` of the mix). Quiet stems are mostly
  separation bleed and all embed alike, so they are never graded.
- measurements: tempo, beats and downbeats (Beat This!, as the tempo table), a 16-step kick/snare/hat pattern per
  bar, key and key-relative chroma, vocal and bass pitch statistics and interval habits, and the stem balance.
- lanes for the Spicetify view at ``LANE_FPS``: loudness of each stem and of the mix, kick/snare/hat activity, and
  vocal and bass pitch.

Stems themselves are never stored.

    python -m soundalike.anysong.stems build --ids ids.npy --previews previews --demucs ckpt/demucs \
        --beat-model final0.ckpt --out shard-000.npz
    python -m soundalike.anysong.stems pack --shards shards/*.npz --out stems/
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path

import numpy as np

SR = 44_100
SR22 = 22_050
SECONDS = 30.0
SAMPLES = int(SR * SECONDS)
STEMS = ("vocals", "drums", "bass", "guitar", "piano", "other")
VOCALS, DRUMS, BASS = 0, 1, 2
HARMONIC = (2, 3, 4, 5)  # bass, guitar, piano, other: the chroma and key source
HTDEMUCS_6S = "5c90dfd2"  # htdemucs_6s signature (demucs 4, MIT code); weights in a local repo folder
VERSION = 2  # 2: drum lanes unmixed by templates (band_onsets)
LANE_FPS = 10
LANE_FRAMES = int(SECONDS * LANE_FPS)
ONSET_FPS = 100
PITCH_FPS = 50
PRESENT_DB = -20.0
SILENT_DBFS = -60.0
MAX_BEATS = 128
STEPS = 16
# drum lane names and order; the ranges are the bands of the VERSION 1 lanes, kept as the lab's baseline
DRUM_BANDS = {"kick": (30.0, 120.0), "snare": (180.0, 1500.0), "hats": (7000.0, 16000.0)}
DB_RANGE = 60.0  # lane loudness: 0 = 60 dB below full scale or quieter, 255 = full scale
EPS = 1e-9

DESC = ("bpm", "beat_stability", "key_tonic", "key_minor", "key_strength", "drum_bars",
        "vocal_presence", "vocal_median", "vocal_range", "vocal_voiced", "vocal_tight", "vocal_rate",
        "bass_median", "bass_range", "bass_rate", "bass_sub", "bass_sustain",
        "kick_rate", "snare_rate", "hats_rate", "loud_range")
D = {name: i for i, name in enumerate(DESC)}

# Pair features (``pair_features``): one column per stem (MuQ cosine of centred stem vectors), then the measurements.
FACETS = (*STEMS, "groove", "tempo", "harmony", "key", "vocal_pitch", "vocal_range", "vocal_presence",
          "bass_pitch", "bass_motion", "balance")
F = {name: i for i, name in enumerate(FACETS)}
# Distances: smaller is more alike; the grade flips them.
DISTANCES = frozenset({"key", "vocal_pitch", "vocal_range", "vocal_presence", "bass_pitch", "balance"})

PITCHES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
KRUMHANSL_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KRUMHANSL_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

COLUMNS = {  # name: (shape after the song axis, dtype)
    "vec": ((len(STEMS), 512), np.float16),
    "present": ((len(STEMS),), np.bool_),
    "balance": ((len(STEMS),), np.float32),
    "desc": ((len(DESC),), np.float32),
    "pattern": ((3, STEPS), np.uint8),
    "chroma": ((12,), np.float16),
    "intervals": ((2, 12), np.float16),
    "lanes": ((len(STEMS) + 1, LANE_FRAMES), np.uint8),
    "hits": ((3, LANE_FRAMES), np.uint8),
    "pitch": ((2, LANE_FRAMES), np.uint8),
    "beats": ((MAX_BEATS,), np.uint16),
    "downbeat": ((MAX_BEATS,), np.bool_),
}
NO_BEAT = np.iinfo(np.uint16).max


# ---------------------------------------------------------------- audio and models
def decode_stereo(path: str | Path) -> np.ndarray:
    """Stereo float32 [2, SAMPLES] at 44.1 kHz, repeat-padded or cropped to 30 s like ``audio.fit_clips``."""
    import soundfile as sf
    import soxr

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if len(data) < sr // 2 or not np.isfinite(data).all():
        raise ValueError("clip too short or not finite")
    data = np.repeat(data, 2, axis=1) if data.shape[1] == 1 else data[:, :2]
    if sr != SR:
        data = soxr.resample(data, sr, SR, quality="HQ")
    if len(data) < SAMPLES:
        data = np.tile(data, (int(np.ceil(SAMPLES / len(data))), 1))
    return np.ascontiguousarray(data[:SAMPLES].T, dtype=np.float32)


class Separator:
    """htdemucs_6s: [B, 2, T] mixes at 44.1 kHz -> [B, 6, 2, T] stems in ``STEMS`` order.

    Each clip is normalised by its own mean and standard deviation as ``demucs.separate`` does; no shift averaging.
    """

    def __init__(self, repo: str | Path, device: str = "cuda", overlap: float = 0.25, fp16: bool = True,
                 batch: int = 4):
        import torch
        from demucs.pretrained import get_model

        self.torch = torch
        self.device = torch.device(device)
        self.model = get_model(HTDEMUCS_6S, repo=Path(repo)).to(self.device).eval()
        sources = list(self.model.sources)
        self.order = [sources.index(s) for s in STEMS]
        self.overlap, self.batch = overlap, batch
        self.fp16 = fp16 and self.device.type == "cuda"

    def __call__(self, mixes: np.ndarray) -> np.ndarray:
        from demucs.apply import apply_model

        torch = self.torch
        out = []
        with torch.inference_mode():
            for b0 in range(0, len(mixes), self.batch):
                x = torch.from_numpy(np.ascontiguousarray(mixes[b0:b0 + self.batch])).to(self.device)
                ref = x.mean(1)
                mean = ref.mean(-1)[:, None, None]
                std = ref.std(-1).clamp_min(1e-4)[:, None, None]
                with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.fp16):
                    y = apply_model(self.model, (x - mean) / std, shifts=0, split=True, overlap=self.overlap,
                                    progress=False, device=self.device, num_workers=0)
                y = y.float() * std[:, None] + mean[:, None]
                out.append(y[:, self.order].cpu().numpy())
        return np.concatenate(out)


def mix22(mix: np.ndarray) -> np.ndarray:
    """Mono 22,050 Hz clip of a stereo 44.1 kHz mix: the Beat This! input (as ``tempo.decode22``)."""
    import soxr

    return np.ascontiguousarray(soxr.resample(mix.mean(0), SR, SR22), dtype=np.float32)


def stem_clips(stems: np.ndarray, sr: int) -> np.ndarray:
    """[6, 2, T] stems -> [6, 30 s at ``sr``] mono clips, through ``audio.fit_clips`` as the served embeddings."""
    from .audio import fit_clips

    return np.stack([fit_clips(s.mean(0), SR, (sr,))[sr] for s in stems])


def embed_stems(embedder, clips: np.ndarray, batch: int = 6) -> np.ndarray:
    """[N, T] clips at ``embedder.sr`` -> [N, D] L2-normalised track embeddings, ``batch`` clips per call."""
    return np.concatenate([embedder.embed(clips[i:i + batch])["track"] for i in range(0, len(clips), batch)])


# ---------------------------------------------------------------- measurements
def _frames(y: np.ndarray, frame: int, hop: int) -> np.ndarray:
    import librosa

    pad = [(0, 0)] * (y.ndim - 1) + [(frame // 2, frame // 2)]
    return librosa.util.frame(np.pad(y, pad), frame_length=frame, hop_length=hop, axis=-1)


def power_db(y: np.ndarray, frame: int, hop: int) -> np.ndarray:
    return 10 * np.log10((_frames(y, frame, hop) ** 2).mean(-2) + 1e-10)


def _fit(x: np.ndarray, n: int) -> np.ndarray:
    return x[..., :n] if x.shape[-1] >= n else np.pad(x, [(0, 0)] * (x.ndim - 1) + [(0, n - x.shape[-1])], mode="edge")


# Kick/snare/hat attack-flux templates (columns; each sums to 1) over 81 log bands, 12 per octave from 30 Hz to 16 kHz
# (bands narrower than an FFT bin merged): median attack spectra of isolated hits in MDB Drums' MIREX-2017 training
# tracks after htdemucs_6s. Lane exponents and iterations were tuned on the same tracks (anysong-lab/stems/drums).
DRUM_TEMPLATES = np.array([
    [0.156057, 0.00388362, 0.000501097],  # 43 Hz
    [0.206148, 0.0049701, 0.000565328],  # 65 Hz
    [0.168806, 0.00551708, 0.000975446],  # 86 Hz
    [0.11114, 0.00526719, 0.000946413],  # 108 Hz
    [0.0721317, 0.00758275, 0.00111191],  # 129 Hz
    [0.041204, 0.0139863, 0.00114966],  # 151 Hz
    [0.0271278, 0.0314635, 0.00131954],  # 172 Hz
    [0.0161045, 0.0468915, 0.00105148],  # 194 Hz
    [0.0117387, 0.0377372, 0.00101226],  # 215 Hz
    [0.00649433, 0.0223748, 0.000889472],  # 237 Hz
    [0.00494329, 0.0139724, 0.000863945],  # 258 Hz
    [0.00417531, 0.0108537, 0.000904447],  # 280 Hz
    [0.00388825, 0.00934546, 0.000796752],  # 301 Hz
    [0.00365315, 0.00937889, 0.000773308],  # 323 Hz
    [0.00381885, 0.00842479, 0.000713537],  # 345 Hz
    [0.00452541, 0.00728805, 0.0008174],  # 366 Hz
    [0.00471725, 0.00591659, 0.000787204],  # 388 Hz
    [0.00464498, 0.00590091, 0.000784546],  # 409 Hz
    [0.00718246, 0.0108958, 0.00115383],  # 441 Hz
    [0.00270661, 0.00572318, 0.000685306],  # 474 Hz
    [0.00208555, 0.00476118, 0.000676238],  # 495 Hz
    [0.00346416, 0.00774197, 0.00142681],  # 528 Hz
    [0.00182735, 0.00424808, 0.000778364],  # 560 Hz
    [0.00303953, 0.0066128, 0.00157001],  # 592 Hz
    [0.0015046, 0.00349452, 0.000934339],  # 624 Hz
    [0.00290375, 0.00588939, 0.00132142],  # 657 Hz
    [0.00225836, 0.00563304, 0.00105295],  # 700 Hz
    [0.00260739, 0.00565774, 0.00117079],  # 743 Hz
    [0.00220059, 0.00474885, 0.00126721],  # 786 Hz
    [0.00232535, 0.00450893, 0.00114206],  # 829 Hz
    [0.00305979, 0.00568802, 0.00134626],  # 883 Hz
    [0.00214406, 0.00367206, 0.000972426],  # 937 Hz
    [0.00249465, 0.00447365, 0.00119834],  # 991 Hz
    [0.00233152, 0.00506642, 0.00119294],  # 1055 Hz
    [0.00210844, 0.00564469, 0.00134075],  # 1120 Hz
    [0.00187125, 0.00519059, 0.00117463],  # 1184 Hz
    [0.00209665, 0.00573794, 0.00143452],  # 1249 Hz
    [0.00219726, 0.00754443, 0.001986],  # 1324 Hz
    [0.00159526, 0.00531311, 0.00144804],  # 1400 Hz
    [0.00195704, 0.00691615, 0.0020375],  # 1475 Hz
    [0.00199072, 0.00781173, 0.00196193],  # 1561 Hz
    [0.0024414, 0.00969473, 0.00250071],  # 1658 Hz
    [0.00221896, 0.00824932, 0.00218075],  # 1766 Hz
    [0.00207627, 0.0085541, 0.00187244],  # 1873 Hz
    [0.0024548, 0.00666539, 0.00179316],  # 1981 Hz
    [0.00336577, 0.00879407, 0.00290512],  # 2099 Hz
    [0.00360266, 0.00973732, 0.00271596],  # 2229 Hz
    [0.00303138, 0.00942797, 0.0024956],  # 2358 Hz
    [0.003567, 0.0109596, 0.00385557],  # 2498 Hz
    [0.00309222, 0.0100716, 0.00406861],  # 2649 Hz
    [0.00288448, 0.0101432, 0.00375663],  # 2799 Hz
    [0.00337328, 0.012327, 0.00474864],  # 2961 Hz
    [0.00299049, 0.0127306, 0.00442614],  # 3133 Hz
    [0.00303543, 0.0137911, 0.00573317],  # 3316 Hz
    [0.00369741, 0.0153669, 0.00959832],  # 3521 Hz
    [0.00359544, 0.0171324, 0.0104025],  # 3736 Hz
    [0.00369823, 0.0170186, 0.0122919],  # 3951 Hz
    [0.00314747, 0.0199671, 0.0110224],  # 4188 Hz
    [0.00318208, 0.0226868, 0.0108981],  # 4447 Hz
    [0.00267575, 0.0231304, 0.0147422],  # 4705 Hz
    [0.00296861, 0.0247164, 0.0223624],  # 4985 Hz
    [0.0027608, 0.0245084, 0.0208786],  # 5286 Hz
    [0.00301061, 0.0266908, 0.0204109],  # 5599 Hz
    [0.00262971, 0.0242026, 0.0218379],  # 5932 Hz
    [0.0023198, 0.0241575, 0.0270396],  # 6277 Hz
    [0.0026844, 0.0229189, 0.0282556],  # 6643 Hz
    [0.00229353, 0.0229682, 0.0301778],  # 7041 Hz
    [0.00234935, 0.0211954, 0.0322785],  # 7461 Hz
    [0.0024147, 0.0188288, 0.0340233],  # 7903 Hz
    [0.0015719, 0.0179875, 0.0353531],  # 8376 Hz
    [0.00145594, 0.0189408, 0.0380557],  # 8882 Hz
    [0.00139932, 0.0186554, 0.0444634],  # 9410 Hz
    [0.00106359, 0.0169177, 0.0446115],  # 9970 Hz
    [0.000657966, 0.0162618, 0.0400449],  # 10562 Hz
    [0.000645732, 0.0145189, 0.0507563],  # 11186 Hz
    [0.000395391, 0.0136928, 0.0533177],  # 11854 Hz
    [0.000522616, 0.0122318, 0.0555033],  # 12554 Hz
    [0.000465813, 0.011179, 0.0714637],  # 13297 Hz
    [0.000380301, 0.00896038, 0.0643365],  # 14093 Hz
    [0.000338628, 0.00712331, 0.0630613],  # 14933 Hz
    [0.000269426, 0.00316645, 0.0445233],  # 15687 Hz
], np.float32)
DRUM_LANE_POWER = (1.5, 1.0, 0.75)
DRUM_NMF_ITERS = 20


def _drum_band_bins(per_oct: int = 12, lo: float = 30.0, hi: float = 16000.0) -> np.ndarray:
    fr = np.arange(1025) * SR / 2048
    edges = lo * 2.0 ** (np.arange(int(np.ceil(np.log2(hi / lo) * per_oct)) + 1) / per_oct)
    return np.unique(np.searchsorted(fr, np.minimum(edges, hi)))


DRUM_BAND_BINS = _drum_band_bins()


def band_onsets(drums44: np.ndarray) -> dict[str, np.ndarray]:
    """Kick/snare/hat onset envelopes of the drum stem at ``ONSET_FPS``: the rectified flux of 81 log bands, unmixed
    by KL-NMF with fixed kick/snare/hat attack templates (after Dittmar & Gaertner 2014). A band's own log flux fires
    as hard on another drum's spill as on its own drum (an 808 lit the snare lane, a clap the kick lane); unmixing lets
    each onset's spectral shape pick its lane. Deterministic: no random initialisation."""
    import librosa

    b = DRUM_BAND_BINS
    M = np.abs(librosa.stft(np.asarray(drums44, np.float32), n_fft=2048, hop_length=SR // ONSET_FPS))
    X = np.add.reduceat(M[b[0]:b[-1]], b[:-1] - b[0], axis=0)
    V = np.maximum(0.0, np.diff(X, axis=1, prepend=X[:, :1]))
    W = DRUM_TEMPLATES
    tiny = 1e-12 + 1e-9 * float(V.mean())
    V = V + tiny
    H = np.repeat(V.sum(0, keepdims=True) / W.shape[1], W.shape[1], axis=0)
    for _ in range(DRUM_NMF_ITERS):
        H *= (W.T @ (V / (W @ H))) / (W.sum(0)[:, None] + tiny)
    return {name: H[i] ** p for i, (name, p) in enumerate(zip(DRUM_BANDS, DRUM_LANE_POWER))}


def step_pattern(envs: Sequence[np.ndarray], beats: np.ndarray, downbeats: np.ndarray) -> tuple[np.ndarray, int]:
    """Onset envelopes averaged on a 16-step grid per bar (downbeat to downbeat; every 4th beat without downbeats)
    -> ([lanes, 16] scaled to each lane's maximum, bars used)."""
    bars = downbeats if len(downbeats) >= 3 else beats[::4]
    rows = []
    for a, b in zip(bars[:-1], bars[1:]):
        if not 1.0 <= b - a <= 6.0:
            continue
        idx = np.round((a + (b - a) * np.arange(STEPS) / STEPS) * ONSET_FPS).astype(int)
        rows.append([[env[max(i - 3, 0):i + 4].max() if i < len(env) else 0.0 for i in idx] for env in envs])
    if not rows:
        return np.zeros((len(envs), STEPS)), 0
    P = np.mean(rows, axis=0)
    return P / (P.max(axis=1, keepdims=True) + EPS), len(rows)


def pitch_track(y22: np.ndarray, fmin: float, fmax: float, frame: int, gate_db: float) -> np.ndarray:
    """pYIN F0 at ``PITCH_FPS`` on semitone bins (fast; the statistics and lanes are per note) -> MIDI, NaN where
    unvoiced or where the stem is more than ``gate_db`` below its own loud parts (pYIN tracks bleed otherwise)."""
    import librosa

    hop = SR22 // PITCH_FPS
    f0, voiced, _ = librosa.pyin(y22, fmin=fmin, fmax=fmax, sr=SR22, frame_length=frame, hop_length=hop,
                                 resolution=1.0, center=True)
    midi = librosa.hz_to_midi(f0)
    # pYIN's semitone bins only pick the note; YIN's interpolated F0 gives the cents where it agrees on that note
    fine = librosa.hz_to_midi(librosa.yin(y22, fmin=fmin, fmax=fmax, sr=SR22, frame_length=frame,
                                          hop_length=hop, center=True))[:len(midi)]
    agree = np.abs(fine - midi) <= 0.5
    midi[agree] = fine[agree]
    level = _fit(power_db(y22, frame, hop), len(midi))
    midi[~voiced | (level < np.percentile(level, 95) + gate_db)] = np.nan
    return midi


def notes_from_midi(midi: np.ndarray, fps: float = PITCH_FPS, min_s: float = 0.06) -> list[int]:
    q = np.round(midi)
    notes, run_val, run_len = [], None, 0
    for v in list(q) + [np.nan]:
        if not np.isnan(v) and v == run_val:
            run_len += 1
            continue
        if run_val is not None and run_len >= min_s * fps:
            notes.append(int(run_val))
        run_val, run_len = (None, 0) if np.isnan(v) else (v, 1)
    return notes


def interval_hist(notes: list[int]) -> np.ndarray:
    h = np.zeros(12)
    for a, b in zip(notes[:-1], notes[1:]):
        h[(b - a) % 12] += 1
    return h / (h.sum() + EPS)


def pitch_stats(midi: np.ndarray) -> tuple[float, float, float, float]:
    """(median MIDI, 5-95 % range in semitones, voiced share, share within 20 cents of the song's tuning)."""
    v = midi[np.isfinite(midi)]
    if len(v) < 20:
        return np.nan, np.nan, len(v) / max(len(midi), 1), np.nan
    tuning = np.median(((v - np.round(v)) + 0.5) % 1 - 0.5)
    dev = (v - tuning) - np.round(v - tuning)
    return (float(np.median(v)), float(np.percentile(v, 95) - np.percentile(v, 5)), len(v) / len(midi),
            float(np.mean(np.abs(dev) < 0.2)))


def key_of(chroma_mean: np.ndarray) -> tuple[int, bool, float]:
    """Krumhansl-Schmuckler key: (tonic pitch class, minor, correlation)."""
    best = (0, False, -2.0)
    for minor, prof in ((False, KRUMHANSL_MAJOR), (True, KRUMHANSL_MINOR)):
        for k in range(12):
            r = float(np.corrcoef(chroma_mean, np.roll(prof, k))[0, 1])
            if r > best[2]:
                best = (k, minor, r)
    return best


def _rate(env: np.ndarray, fps: float) -> float:
    import librosa

    if env.max() <= EPS:
        return 0.0
    peaks = librosa.util.peak_pick(env / env.max(), pre_max=3, post_max=3, pre_avg=5, post_avg=5, delta=0.15, wait=5)
    return len(peaks) / (len(env) / fps)


def _lane_u8(db: np.ndarray) -> np.ndarray:
    return np.clip(np.round((db + DB_RANGE) / DB_RANGE * 255), 0, 255).astype(np.uint8)


def _pool(x: np.ndarray, n: int, fn=np.max) -> np.ndarray:
    k = len(x) // n
    return fn(x[:n * k].reshape(n, k), axis=1)


def describe(stems: np.ndarray, mix: np.ndarray, beats: np.ndarray, downbeats: np.ndarray) -> dict[str, np.ndarray]:
    """Measurements and lanes of one song from its [6, 2, T] stems, [2, T] mix (44.1 kHz) and Beat This! beats.
    Every column of ``COLUMNS`` except ``vec``."""
    import librosa
    import soxr

    mono = stems.mean(1)
    y22 = np.ascontiguousarray(soxr.resample(np.vstack([mono, mix.mean(0)[None]]).T, SR, SR22).T)
    hop = SR22 // LANE_FPS
    lane_db = _fit(power_db(y22, 2 * hop, hop), LANE_FRAMES)  # [7, 300]
    power = (y22 ** 2).mean(-1)
    balance = 10 * np.log10(power[:6] / (power[6] + 1e-12) + 1e-12)
    present = (balance > PRESENT_DB) & (10 * np.log10(power[:6] + 1e-12) > SILENT_DBFS)
    desc = np.full(len(DESC), np.nan, np.float32)

    ibi = np.diff(beats)
    if len(ibi) >= 3 and np.median(ibi) > 0:
        med = float(np.median(ibi))
        # beats sit on Beat This!'s 20 ms frames, so fit the period across all beats (counting skipped ones)
        k = np.concatenate([[0.0], np.cumsum(np.maximum(1.0, np.round(ibi / med)))])
        period = float(np.polyfit(k, np.asarray(beats, float), 1)[0])
        desc[D["bpm"]] = 60.0 / (period if period > 0 else med)
        desc[D["beat_stability"]] = float(np.mean(np.abs(ibi / med - 1.0) <= 0.10))

    # drums: band hit lanes and the bar pattern
    hits = np.zeros((3, LANE_FRAMES), np.uint8)
    pattern = np.zeros((3, STEPS), np.uint8)
    if present[DRUMS]:
        env = band_onsets(mono[DRUMS])
        for i, (name, e) in enumerate(env.items()):
            e = _fit(e, int(SECONDS * ONSET_FPS))
            lane = _pool(e, LANE_FRAMES) / (np.percentile(e, 99.5) + EPS)
            hits[i] = np.clip(np.round(lane * 255), 0, 255)
            desc[D[f"{name}_rate"]] = _rate(e, ONSET_FPS)
        P, bars = step_pattern(list(env.values()), beats, downbeats)
        desc[D["drum_bars"]] = bars
        if bars:
            pattern = np.round(P * 255).astype(np.uint8)

    # vocals and bass: pitch lanes, statistics and interval habits
    pitch = np.zeros((2, LANE_FRAMES), np.uint8)
    intervals = np.zeros((2, 12), np.float16)
    # note-aligned ranges keep pYIN's semitone bins on the MIDI grid: C2-C6 for vocals, B0-E4 for bass
    for row, (s, fmin, fmax, frame, gate) in enumerate(((VOCALS, 65.406, 1046.502, 2048, -25.0),
                                                       (BASS, 30.868, 329.628, 4096, -24.0))):
        if not present[s]:
            continue
        midi = pitch_track(y22[s], fmin, fmax, frame, gate)
        midi = _fit(midi, int(SECONDS * PITCH_FPS))
        with np.errstate(all="ignore"):
            lane = np.array([np.nanmedian(c) if np.isfinite(c).any() else np.nan
                             for c in midi[:LANE_FRAMES * 5].reshape(LANE_FRAMES, 5)])
        pitch[row] = np.where(np.isfinite(lane), np.clip(np.round(lane * 2), 1, 255), 0)
        intervals[row] = interval_hist(notes_from_midi(midi))
        med, rng, voiced, tight = pitch_stats(midi)
        name = "vocal" if s == VOCALS else "bass"
        desc[D[f"{name}_median"]], desc[D[f"{name}_range"]] = med, rng
        if s == VOCALS:
            desc[D["vocal_voiced"]], desc[D["vocal_tight"]] = voiced, tight
            on = librosa.onset.onset_detect(y=y22[VOCALS], sr=SR22, hop_length=512)
            desc[D["vocal_rate"]] = len(on) / SECONDS
        else:
            on = librosa.onset.onset_detect(y=y22[BASS], sr=SR22, hop_length=512)
            desc[D["bass_rate"]] = len(on) / SECONDS
            S = np.abs(librosa.stft(y22[BASS], n_fft=8192, hop_length=2205)) ** 2
            fr = librosa.fft_frequencies(sr=SR22, n_fft=8192)
            desc[D["bass_sub"]] = float(S[fr < 60].sum() / (S[fr < 300].sum() + EPS))
            b = lane_db[BASS]
            desc[D["bass_sustain"]] = float(np.mean(b > np.percentile(b, 95) - 12))
    v = lane_db[VOCALS]
    desc[D["vocal_presence"]] = float(np.mean(v > np.percentile(v, 95) - 25)) if present[VOCALS] else 0.0

    # harmony: chroma of the pitched stems, key, key-relative profile
    harm = y22[list(HARMONIC)].sum(0)
    chroma = librosa.feature.chroma_stft(y=harm, sr=SR22, n_fft=8192, hop_length=2205).mean(axis=1)
    if chroma.sum() > EPS:
        k, minor, strength = key_of(chroma)
        desc[D["key_tonic"]], desc[D["key_minor"]], desc[D["key_strength"]] = k, float(minor), strength
    m = lane_db[6]
    desc[D["loud_range"]] = float(np.percentile(m, 95) - np.percentile(m, 10))

    beat_arr = np.full(MAX_BEATS, NO_BEAT, np.uint16)
    down = np.zeros(MAX_BEATS, np.bool_)
    b = np.asarray(beats, float)[:MAX_BEATS]
    beat_arr[:len(b)] = np.clip(np.round(b * 100), 0, NO_BEAT - 1)
    if len(b) and len(downbeats):
        down[:len(b)] = np.min(np.abs(b[:, None] - np.asarray(downbeats, float)[None]), axis=1) < 0.02
    return {"present": present, "balance": balance.astype(np.float32), "desc": desc, "pattern": pattern,
            "chroma": (chroma / (chroma.sum() + EPS)).astype(np.float16), "intervals": intervals,
            "lanes": _lane_u8(lane_db), "hits": hits, "pitch": pitch, "beats": beat_arr, "downbeat": down}


def warm_up() -> dict[str, np.ndarray]:
    """``describe`` a synthetic song with vocals, drums and bass, so the measurements' numba kernels (pYIN, onsets)
    compile before the first live dissection instead of during it (about 5 s on first use)."""
    t = np.arange(SAMPLES) / SR
    mono = np.zeros((len(STEMS), SAMPLES), np.float32)
    mono[VOCALS] = 0.2 * np.sin(2 * np.pi * 220.0 * t)
    mono[BASS] = 0.3 * np.sin(2 * np.pi * 55.0 * t)
    n = int(0.08 * SR)
    hit = 0.8 * np.sin(2 * np.pi * 60.0 * np.arange(n) / SR) * np.exp(-np.arange(n) / (0.015 * SR))
    beats = np.arange(0.1, SECONDS - 0.2, 0.6)
    for b in beats:
        i = int(b * SR)
        mono[DRUMS, i:i + n] += hit[:SAMPLES - i]
    stems = np.repeat(mono[:, None], 2, axis=1)
    return describe(stems, stems.sum(0), beats, beats[::4])


# ---------------------------------------------------------------- dissecting songs
class Dissector:
    """Separation, beats and stem embeddings on one device; measurements on a CPU thread pool."""

    def __init__(self, separator: Separator, tracker, embedder, workers: int = 4, embed_batch: int = 6):
        self.separator, self.tracker, self.embedder = separator, tracker, embedder
        self.embed_batch = embed_batch
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="stems-describe") if workers else None

    def gpu(self, mixes: np.ndarray) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
        """The device work for [B, 2, T] mixes: (stems [B, 6, 2, T], beats per song, vec [B, 6, D] float16)."""
        stems = self.separator(mixes)
        beats = self.tracker.track([mix22(m) for m in mixes])
        clips = np.concatenate([stem_clips(s, self.embedder.sr) for s in stems])
        vec = embed_stems(self.embedder, clips, self.embed_batch).reshape(len(mixes), len(STEMS), -1)
        return stems, beats, vec.astype(np.float16)

    def cpu(self, stems: np.ndarray, mixes: np.ndarray, beats, vec: np.ndarray) -> list[dict[str, np.ndarray]]:
        def one(i: int) -> dict[str, np.ndarray]:
            return describe(stems[i], mixes[i], *beats[i]) | {"vec": vec[i]}

        if self.pool is None:
            return [one(i) for i in range(len(mixes))]
        return list(self.pool.map(one, range(len(mixes))))

    def __call__(self, mixes: np.ndarray) -> list[dict[str, np.ndarray]]:
        stems, beats, vec = self.gpu(mixes)
        return self.cpu(stems, mixes, beats, vec)


def stack(records: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    out = {}
    for name, (shape, dtype) in COLUMNS.items():
        out[name] = np.stack([np.asarray(r[name], dtype) for r in records]) if records else np.zeros((0, *shape), dtype)
    return out


# ---------------------------------------------------------------- grading two songs
def _cos_rows(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    return (B @ a) / (np.linalg.norm(B, axis=1) * np.linalg.norm(a) + EPS)


def groove_sim(Pa: np.ndarray, Pb: np.ndarray) -> float:
    """Cosine of two 3x16 patterns, best over whole-beat rotations (the bar phase is uncertain)."""
    a = Pa.astype(np.float64).ravel()
    best = -1.0
    for shift in (0, 4, 8, 12):
        b = np.roll(Pb.astype(np.float64), shift, axis=1).ravel()
        best = max(best, float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + EPS)))
    return best


def tempo_sim(a: float, b: float) -> float:
    """1 for the same pulse (double and half time count as the same), falling off within a few per cent."""
    if not (np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0):
        return np.nan
    r = abs(np.log2(a / b))
    r = min(r % 1.0, 1.0 - r % 1.0)
    return float(np.exp(-r / 0.05))


def key_distance(ta: float, ma: float, tb: float, mb: float) -> float:
    """Steps around the circle of fifths between two keys (relative major/minor count as the same key), 0..6."""
    if not all(np.isfinite(v) for v in (ta, ma, tb, mb)):
        return np.nan
    ra = (int(ta) + 3 * int(ma)) % 12  # relative major
    rb = (int(tb) + 3 * int(mb)) % 12
    d = (7 * (ra - rb)) % 12
    return float(min(d, 12 - d))


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.std() < EPS or b.std() < EPS:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def pair_features(a: dict[str, np.ndarray], b: dict[str, np.ndarray], mu: np.ndarray | None = None) -> np.ndarray:
    """``FACETS`` for two songs (rows of ``COLUMNS``); NaN where a facet does not apply (a stem missing in either
    song, no drums, no vocals, unknown tempo or key). ``mu`` [6, D]: per-stem mean vectors to centre by."""
    f = np.full(len(FACETS), np.nan, np.float32)
    va, vb = a["vec"].astype(np.float32), b["vec"].astype(np.float32)
    if mu is not None:
        va, vb = va - mu, vb - mu
    both = np.asarray(a["present"], bool) & np.asarray(b["present"], bool)
    for s in range(len(STEMS)):
        if both[s]:
            f[s] = va[s] @ vb[s] / (np.linalg.norm(va[s]) * np.linalg.norm(vb[s]) + EPS)
    da, db = a["desc"], b["desc"]
    if both[DRUMS] and da[D["drum_bars"]] > 0 and db[D["drum_bars"]] > 0:
        f[F["groove"]] = groove_sim(a["pattern"], b["pattern"])
    f[F["tempo"]] = tempo_sim(float(da[D["bpm"]]), float(db[D["bpm"]]))
    if np.isfinite(da[D["key_tonic"]]) and np.isfinite(db[D["key_tonic"]]):
        ca = np.roll(a["chroma"].astype(np.float64), -int(da[D["key_tonic"]]))
        cb = np.roll(b["chroma"].astype(np.float64), -int(db[D["key_tonic"]]))
        same_mode = da[D["key_minor"]] == db[D["key_minor"]]
        c = _corr(ca, cb)
        f[F["harmony"]] = max(c, 0.0) * (1.0 if same_mode else 0.8) if np.isfinite(c) else np.nan
        f[F["key"]] = key_distance(da[D["key_tonic"]], da[D["key_minor"]], db[D["key_tonic"]], db[D["key_minor"]])
    if both[VOCALS]:
        f[F["vocal_pitch"]] = abs(da[D["vocal_median"]] - db[D["vocal_median"]])
        f[F["vocal_range"]] = abs(da[D["vocal_range"]] - db[D["vocal_range"]])
    f[F["vocal_presence"]] = abs(da[D["vocal_presence"]] - db[D["vocal_presence"]])
    if both[BASS]:
        f[F["bass_pitch"]] = abs(da[D["bass_median"]] - db[D["bass_median"]])
        f[F["bass_motion"]] = _corr(a["intervals"][1], b["intervals"][1])
    f[F["balance"]] = float(np.linalg.norm(np.maximum(a["balance"], -40) - np.maximum(b["balance"], -40)))
    return f


def grade(features: np.ndarray, calib: np.ndarray) -> np.ndarray:
    """Percentile (0..100) of each facet among calibration pairs (seeds and their candidates), higher = more alike;
    NaN stays NaN. ``calib`` [F, 101]: quantiles of each facet."""
    out = np.full(len(features), np.nan, np.float32)
    for i, v in enumerate(features):
        if not np.isfinite(v) or not np.isfinite(calib[i]).all():
            continue
        p = float(np.interp(v, calib[i], np.linspace(0, 100, calib.shape[1])))
        out[i] = 100 - p if FACETS[i] in DISTANCES else p
    return out


# ---------------------------------------------------------------- the table the API reads
class StemTable:
    """Dissections by track id: a directory of ``<column>.npy`` (memory-mapped) plus ``ids.npy`` and ``meta.json``
    (``mu``: per-stem mean vectors; ``calib``: facet quantiles)."""

    def __init__(self, ids: np.ndarray, cols: dict[str, np.ndarray], meta: dict):
        self.ids = np.asarray(ids, np.int64)
        # ``pack`` writes ids sorted; a binary search avoids a per-song dict (~100 MB at a million songs).
        self._order = None if bool(np.all(self.ids[1:] > self.ids[:-1])) else np.argsort(self.ids, kind="stable")
        self._sorted = self.ids if self._order is None else self.ids[self._order]
        self.cols, self.meta = cols, meta
        self.mu = np.asarray(meta["mu"], np.float32) if meta.get("mu") is not None else None
        self.calib = np.asarray(meta["calib"], np.float32) if meta.get("calib") is not None else None

    @classmethod
    def load(cls, path: str | Path) -> StemTable:
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        if meta.get("version") != VERSION:
            raise ValueError(f"stem table version {meta.get('version')} (this code reads {VERSION})")
        cols = {name: np.load(path / f"{name}.npy", mmap_mode="r") for name in COLUMNS}
        return cls(np.load(path / "ids.npy"), cols, meta)

    def __len__(self) -> int:
        return len(self.ids)

    def _index(self, track_id: int) -> int | None:
        track_id = int(track_id)
        j = int(np.searchsorted(self._sorted, track_id))
        if j < len(self._sorted) and int(self._sorted[j]) == track_id:
            return j if self._order is None else int(self._order[j])
        return None

    def __contains__(self, track_id: int) -> bool:
        return self._index(track_id) is not None

    def row(self, track_id: int) -> dict[str, np.ndarray] | None:
        i = self._index(track_id)
        return None if i is None else {name: np.array(col[i]) for name, col in self.cols.items()}


class StemStore:
    """Rows for the API by track id: the packed table, then dissections cached on disk, then a small LRU.
    ``mu`` and ``calib`` come from the table (an empty table can carry them for live-only serving)."""

    def __init__(self, table: StemTable | None = None, cache_dir: str | Path | None = None, size: int = 256):
        self.table, self.size = table, size
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.mu = table.mu if table is not None else None
        self.calib = table.calib if table is not None else None
        self._mem: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        self._lock = threading.Lock()

    def _path(self, track_id: int) -> Path:
        return self.cache_dir / f"{track_id % 1000:03d}" / f"{track_id}.npz"

    def _remember(self, track_id: int, row: dict[str, np.ndarray]) -> None:
        with self._lock:
            self._mem[track_id] = row
            self._mem.move_to_end(track_id)
            while len(self._mem) > self.size:
                self._mem.popitem(last=False)

    def get(self, track_id: int) -> tuple[dict[str, np.ndarray] | None, str | None]:
        """(row, "catalog" | "cache") or (None, None)."""
        track_id = int(track_id)
        if self.table is not None:
            row = self.table.row(track_id)
            if row is not None:
                return row, "catalog"
        with self._lock:
            row = self._mem.get(track_id)
            if row is not None:
                self._mem.move_to_end(track_id)
                return row, "cache"
        if self.cache_dir is not None and self._path(track_id).exists():
            try:
                with np.load(self._path(track_id), allow_pickle=False) as z:
                    row = {name: z[name] for name in COLUMNS} if int(z["version"]) == VERSION else None
            except Exception:  # noqa: BLE001 - a torn or foreign cache file is a miss
                row = None
            if row is not None:
                self._remember(track_id, row)
                return row, "cache"
        return None, None

    def put(self, track_id: int, row: dict[str, np.ndarray]) -> None:
        track_id = int(track_id)
        row = {name: np.asarray(row[name], dtype) for name, (_, dtype) in COLUMNS.items()}
        self._remember(track_id, row)
        if self.cache_dir is not None:
            path = self._path(track_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f".{track_id}.{threading.get_ident()}.tmp.npz")
            np.savez(tmp, version=np.int32(VERSION), **row)
            os.replace(tmp, path)


def pack(shards: Iterable[str | Path], out: str | Path, mu: np.ndarray | None = None,
         calib: np.ndarray | None = None, info: dict | None = None) -> dict:
    """Merge build shards into a ``StemTable`` directory (the last shard wins for a repeated id)."""
    parts = []
    for p in shards:
        with np.load(p, allow_pickle=False) as z:
            parts.append({k: z[k] for k in z.files})
    ids = np.concatenate([p["ids"] for p in parts]) if parts else np.zeros(0, np.int64)
    last = {int(t): i for i, t in enumerate(ids)}
    keep = np.array(sorted(last.values(), key=lambda i: ids[i]), np.int64)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "ids.npy", ids[keep].astype(np.int64))
    for name in COLUMNS:
        col = np.concatenate([p[name] for p in parts]) if parts else np.zeros((0, *COLUMNS[name][0]), COLUMNS[name][1])
        np.save(out / f"{name}.npy", np.ascontiguousarray(col[keep]))
    vec = np.concatenate([p["vec"] for p in parts])[keep] if parts else None
    present = np.concatenate([p["present"] for p in parts])[keep] if parts else None
    if mu is None and vec is not None and len(vec):
        mu = np.stack([vec[present[:, s], s].astype(np.float32).mean(0) if present[:, s].any()
                       else np.zeros(vec.shape[-1], np.float32) for s in range(len(STEMS))])
    meta = {"version": VERSION, "model": f"htdemucs_6s ({HTDEMUCS_6S}) + MuQ-MuLan per stem", "songs": len(keep),
            "stems": list(STEMS), "facets": list(FACETS), "desc": list(DESC), "present_db": PRESENT_DB,
            "lane_fps": LANE_FPS, "mu": None if mu is None else np.asarray(mu, np.float32).round(6).tolist(),
            "calib": None if calib is None else np.asarray(calib, np.float32).round(6).tolist()} | (info or {})
    (out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return {k: v for k, v in meta.items() if k not in ("mu", "calib")}


# ---------------------------------------------------------------- what the API sends
def _midi_name(m: float) -> str | None:
    if not np.isfinite(m):
        return None
    n = round(float(m))
    return f"{PITCHES[n % 12]}{n // 12 - 1}"


def _num(x, nd: int = 2):
    x = float(x)
    return round(x, nd) if np.isfinite(x) else None


def payload(track_id: int, row: dict[str, np.ndarray]) -> dict:
    """One song's dissection as JSON for the Spicetify view (lanes as base64 bytes to keep it small)."""
    import base64

    d = row["desc"]
    beats = row["beats"][row["beats"] != NO_BEAT]
    key = None
    if np.isfinite(d[D["key_tonic"]]):
        key = {"tonic": PITCHES[int(d[D["key_tonic"]])], "mode": "minor" if d[D["key_minor"]] else "major",
               "strength": _num(d[D["key_strength"]])}

    def b64(a: np.ndarray) -> str:
        return base64.b64encode(np.ascontiguousarray(a, np.uint8).tobytes()).decode("ascii")

    return {
        "id": int(track_id), "version": VERSION, "seconds": SECONDS, "fps": LANE_FPS, "frames": LANE_FRAMES,
        "bpm": _num(d[D["bpm"]], 1), "key": key,
        "beats": (beats / 100).round(2).tolist(),
        "downbeats": (beats[row["downbeat"][:len(beats)]] / 100).round(2).tolist(),
        "stems": [{"name": s, "present": bool(row["present"][i]), "balance_db": _num(row["balance"][i], 1),
                   "loudness": b64(row["lanes"][i])} for i, s in enumerate(STEMS)],
        "mix": b64(row["lanes"][len(STEMS)]),
        "db_range": DB_RANGE,
        "drums": {name: b64(row["hits"][i]) for i, name in enumerate(DRUM_BANDS)},
        "pattern": {name: row["pattern"][i].tolist() for i, name in enumerate(DRUM_BANDS)},
        "pitch": {"vocals": b64(row["pitch"][0]), "bass": b64(row["pitch"][1])},
        "vocals": {"median": _midi_name(d[D["vocal_median"]]), "range_st": _num(d[D["vocal_range"]], 1),
                   "presence": _num(d[D["vocal_presence"]]), "onsets_per_s": _num(d[D["vocal_rate"]], 1)},
        "bass": {"median": _midi_name(d[D["bass_median"]]), "range_st": _num(d[D["bass_range"]], 1),
                 "onsets_per_s": _num(d[D["bass_rate"]], 1), "sub_share": _num(d[D["bass_sub"]])},
        "drum_rates": {name: _num(d[D[f"{name}_rate"]], 1) for name in DRUM_BANDS},
    }


FACET_LABELS = {"vocals": "Vocals", "drums": "Drums", "bass": "Bass", "guitar": "Guitar", "piano": "Keys",
                "other": "Synths & other", "groove": "Groove", "tempo": "Tempo", "harmony": "Harmony",
                "key": "Key", "vocal_pitch": "Vocal register", "vocal_range": "Vocal range",
                "vocal_presence": "Vocal presence", "bass_pitch": "Bass register", "bass_motion": "Bassline moves",
                "balance": "Mix balance"}


def compare(a: dict[str, np.ndarray], b: dict[str, np.ndarray], mu: np.ndarray | None,
            calib: np.ndarray | None) -> dict:
    """Facet values and grades for the view, strongest matches first."""
    f = pair_features(a, b, mu)
    g = grade(f, calib) if calib is not None else np.full(len(f), np.nan, np.float32)
    facets = [{"name": n, "label": FACET_LABELS[n], "value": _num(f[i], 3), "grade": _num(g[i], 0)}
              for i, n in enumerate(FACETS) if np.isfinite(f[i])]
    graded = sorted((x for x in facets if x["grade"] is not None), key=lambda x: -x["grade"])
    return {"facets": facets, "graded": calib is not None,
            "matched_on": [x["name"] for x in graded if x["grade"] >= 85][:3]}


# ---------------------------------------------------------------- catalog build
def _prefetch(pool: ThreadPoolExecutor, fn: Callable, items: Iterable, ahead: int):
    """``pool.map`` in order with at most ``ahead`` results in flight (a decoded clip is ~10 MB)."""
    it = iter(items)
    queue = deque(pool.submit(fn, x) for x in islice(it, ahead))
    while queue:
        result = queue.popleft().result()
        for x in islice(it, 1):
            queue.append(pool.submit(fn, x))
        yield result


def build(ids: Sequence[int], previews: Path, dissector: Dissector, out: Path, batch: int = 8, workers: int = 8,
          log: Callable[[str], None] = print) -> dict:
    """Dissect ``previews/<id % 1000:03d>/<id>.mp3`` for ``ids`` into one shard (``ids`` + ``COLUMNS``); unreadable
    previews are logged as ``decode failed <id>: ...`` and skipped."""
    def load(t: int):
        try:
            return t, decode_stereo(previews / f"{t % 1000:03d}" / f"{t}.mp3")
        except Exception as exc:  # noqa: BLE001 - one bad preview must not stop a shard
            log(f"decode failed {t}: {exc}")
            return t, None

    done_ids, records, t0, gpu_s = [], [], time.time(), 0.0
    # The device works on batch n+1 while one collector thread measures batch n (fanning out to the dissector's pool).
    with ThreadPoolExecutor(max_workers=workers) as io, ThreadPoolExecutor(max_workers=1) as collector:
        loaded = _prefetch(io, load, ids, ahead=2 * batch + workers)
        pending = None
        buf: list[tuple[int, np.ndarray]] = []
        batches = 0

        def flush(items):
            nonlocal gpu_s
            t = time.perf_counter()
            mixes = np.stack([m for _, m in items])
            stems, beats, vec = dissector.gpu(mixes)
            gpu_s += time.perf_counter() - t
            return collector.submit(dissector.cpu, stems, mixes, beats, vec), [t for t, _ in items]

        def collect(job) -> None:
            nonlocal batches
            records.extend(job[0].result())
            done_ids.extend(job[1])
            batches += 1
            if batches % 25 == 0:
                log(f"{len(done_ids)}/{len(ids)} songs ({time.time() - t0:.0f} s, device {gpu_s:.0f} s)")

        for tid, mix in loaded:
            if mix is None:
                continue
            buf.append((tid, mix))
            if len(buf) == batch:
                job = flush(buf)
                buf = []
                if pending is not None:
                    collect(pending)
                pending = job
        if buf:
            job = flush(buf)
            if pending is not None:
                collect(pending)
            pending = job
        if pending is not None:
            collect(pending)
    cols = stack(records)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, ids=np.asarray(done_ids, np.int64), **cols)
    tmp.replace(out)
    secs = time.time() - t0
    info = {"songs": len(done_ids), "seconds": round(secs, 1), "device_s": round(gpu_s, 1),
            "per_song_ms": round(1000 * secs / max(len(done_ids), 1), 1)}
    log(f"wrote {out}: {info}")
    return info


def load_dissector(demucs: str | Path, beat_model: str | Path, device: str = "cuda", batch: int = 4,
                   workers: int = 4, fp16: bool = True, embedder=None) -> Dissector:
    from .embedders import MuQMuLanEmb
    from .tempo import Tracker

    if embedder is None:
        embedder = MuQMuLanEmb(device=device)
        embedder.load()
    return Dissector(Separator(demucs, device, fp16=fp16, batch=batch), Tracker(beat_model, device),
                     embedder, workers=workers)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="stems")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="dissect previews into one shard")
    b.add_argument("--ids", type=Path, required=True, help=".npy or text file of Deezer track ids")
    b.add_argument("--previews", type=Path, required=True)
    b.add_argument("--demucs", type=Path, required=True, help="folder holding the htdemucs_6s weights")
    b.add_argument("--beat-model", required=True, help="Beat This! checkpoint")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--device", default="cuda")
    b.add_argument("--batch", type=int, default=8)
    b.add_argument("--workers", type=int, default=8, help="CPU threads for the measurements")
    b.add_argument("--fp32", action="store_true", help="separate in float32 (default: float16 autocast)")
    k = sub.add_parser("pack", help="merge shards into a table directory")
    k.add_argument("--shards", nargs="+", type=Path, required=True)
    k.add_argument("--out", type=Path, required=True)
    k.add_argument("--calib", type=Path, default=None, help=".npy facet quantiles [F, 101]")
    a = p.parse_args(argv)
    if a.cmd == "build":
        ids = np.load(a.ids) if a.ids.suffix == ".npy" else np.array(a.ids.read_text().split(), np.int64)
        dis = load_dissector(a.demucs, a.beat_model, a.device, batch=a.batch, workers=a.workers, fp16=not a.fp32)
        build([int(t) for t in ids], a.previews, dis, a.out, batch=a.batch, workers=a.workers)
    else:
        calib = np.load(a.calib) if a.calib else None
        print(json.dumps(pack(a.shards, a.out, calib=calib), indent=1))


if __name__ == "__main__":
    main()
