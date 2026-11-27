"""Audio decoding helpers for 30 s preview clips."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

CLIP_SECONDS = 30.0
# Where each channel's windows sit in a 30 s preview, as its embedder cuts them: (hop, length) in seconds and how
# many, the first starting at 0 (``embedders``: MuQ-MuLan 3 x 10 s, CLEWS 20 s shingles every 5 s, SampleID 6.125 s
# chunks at half overlap).
WINDOW_LAYOUT: dict[str, tuple[float, float, int]] = {
    "melody": (5.0, 20.0, 3), "samples": (3.0625, 6.125, 8), "sonic": (10.0, 10.0, 3)}
PART_COVER = 0.7  # a window belongs to a part of the preview that covers at least this share of it


def part_windows(start: float, end: float, layout: Mapping[str, tuple[float, float, int]] = WINDOW_LAYOUT
                 ) -> dict[str, list[int]]:
    """The windows of each channel that stand for the span ``start`` to ``end`` (seconds) of a 30 s preview: those
    the span covers at least ``PART_COVER`` of; when it covers none that far, the one it covers most (ties: the one
    centred nearest the span's centre). Sound windows are 10 s and samples chunks 6 s, so they follow a span closely;
    melody shingles are 20 s, so a 10 s span is one shingle that also hears 5 s either side of it."""
    out = {}
    for channel, (hop, length, count) in layout.items():
        cover = [max(0.0, min(end, i * hop + length) - max(start, i * hop)) / length for i in range(count)]
        chosen = [i for i, c in enumerate(cover) if c >= PART_COVER]
        if not chosen:
            centre = (start + end) / 2
            chosen = [min(range(count), key=lambda i: (-round(cover[i], 6), abs(i * hop + length / 2 - centre), i))]
        out[channel] = chosen
    return out


def window_spans(part: Mapping[str, Sequence[int]], layout: Mapping[str, tuple[float, float, int]] = WINDOW_LAYOUT
                 ) -> dict[str, list[float]]:
    """The seconds of the preview each channel hears for ``part`` (``part_windows``): [first start, last end]."""
    return {channel: [round(min(rows) * layout[channel][0], 2),
                      round(max(rows) * layout[channel][0] + layout[channel][1], 2)]
            for channel, rows in part.items()}


def excerpt_clips(mono: np.ndarray, native_sr: int, rates: Mapping[str, int]) -> dict[str, np.ndarray]:
    """One clip per channel (``rates``: channel -> its model's sample rate) for an excerpt of any length, cut so that
    each model hears only as many windows as the excerpt fills, unlike a preview, which is repeated to 30 s: sound
    10 s windows spread evenly over the excerpt (one for up to 10 s, two up to 20 s, three beyond), melody 20 s
    shingles every 5 s (one for up to 20 s), samples 6.125 s chunks at half overlap. Audio shorter than one window is
    repeated to fill it. A 30 s excerpt gives the windows of a preview."""
    import soxr

    if len(mono) == 0:
        raise ValueError("empty audio")
    out, resampled = {}, {}
    for channel, sr in rates.items():
        if sr not in resampled:
            resampled[sr] = soxr.resample(mono, native_sr, sr, quality="HQ") if native_sr != sr else mono
        x = resampled[sr]
        hop, length, _ = WINDOW_LAYOUT[channel]
        w = int(round(length * sr))
        if len(x) < w:
            x = np.tile(x, int(np.ceil(w / len(x))))[:w]
        if channel == "sonic":
            n = min(3, int(np.ceil(len(x) / w - 1e-6)))
            starts = np.linspace(0, len(x) - w, n).round().astype(int) if n > 1 else [0]
            x = np.concatenate([x[s:s + w] for s in starts])
        else:
            h = int(round(hop * sr))
            x = x[:w + (len(x) - w) // h * h]
        out[channel] = np.ascontiguousarray(x, dtype=np.float32)
    return out


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


EXCERPT_FORMATS = ("wav", "flac", "ogg", "mp3")  # what ``decode_bytes`` can be told to take
MAX_CHANNELS = 2
MIN_RATE, MAX_RATE = 8000, 96_000


class UnsupportedAudio(ValueError):
    """The bytes are not audio in a format that is accepted (HTTP 415, as against 422 for a clip that is)."""


def sniff(data: bytes) -> str | None:
    """The container of a clip by its first bytes, whatever a client says it is: "wav", "flac", "ogg", "mp3" or
    None. MP3 is an ID3 tag or a frame sync (MPEG 1/2/2.5, layer III)."""
    head = bytes(data[:12])
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "wav"
    if head[:4] == b"fLaC":
        return "flac"
    if head[:4] == b"OggS":
        return "ogg"
    if head[:3] == b"ID3" or (len(head) >= 2 and head[0] == 0xFF and head[1] & 0xE6 == 0xE2):
        return "mp3"
    return None


def _checked(mono: np.ndarray, sr: int, max_seconds: float, min_seconds: float) -> tuple[np.ndarray, int]:
    if len(mono) > (max_seconds + 0.5) * sr:
        raise ValueError(f"the clip is longer than {max_seconds:g} s")
    if len(mono) < min_seconds * sr:
        raise ValueError(f"the clip is shorter than {min_seconds:g} s")
    if len(mono) == 0 or not np.isfinite(mono).all():
        raise ValueError("the clip holds no usable audio")
    return np.ascontiguousarray(mono[:int(max_seconds * sr)], dtype=np.float32), sr


def decode_wav(data: bytes, max_seconds: float, min_seconds: float = 0.0) -> tuple[np.ndarray, int]:
    """Mono float32 and the sample rate of a RIFF/WAVE clip held in memory, read here with numpy and no decoder
    library: uncompressed PCM of 8, 16, 24 or 32 bits or IEEE float of 32 or 64 bits (also as WAVE_FORMAT_EXTENSIBLE),
    ``MAX_CHANNELS`` channels at most, ``MIN_RATE`` to ``MAX_RATE`` Hz. Chunk sizes are never trusted beyond the
    bytes that are there; the length is checked before a sample is converted. Raises ValueError with a message fit
    for the client."""
    import struct

    view = memoryview(data)
    if len(view) < 12 or bytes(view[:4]) != b"RIFF" or bytes(view[8:12]) != b"WAVE":
        raise UnsupportedAudio("not a WAV file")
    fmt = body = None
    at = 12
    for _ in range(64):  # a few chunks precede the audio in any real file
        if at + 8 > len(view) or (fmt is not None and body is not None):
            break
        name, size = bytes(view[at:at + 4]), struct.unpack_from("<I", view, at + 4)[0]
        chunk = view[at + 8:at + 8 + size]  # a size past the end is cut to what is there
        if name == b"fmt " and fmt is None:
            fmt = bytes(chunk[:40])
        elif name == b"data" and body is None:
            body = chunk
        at += 8 + size + (size & 1)
    if fmt is None or body is None or len(fmt) < 16:
        raise ValueError("could not decode the audio (a WAV file needs a fmt and a data chunk)")
    tag, channels, sr, _, align, bits = struct.unpack_from("<HHIIHH", fmt)
    if tag == 0xFFFE and len(fmt) >= 26:  # WAVE_FORMAT_EXTENSIBLE: the real format leads the sub-format GUID
        tag = struct.unpack_from("<H", fmt, 24)[0]
    kinds = {(1, 8): "u1", (1, 16): "<i2", (1, 24): None, (1, 32): "<i4", (3, 32): "<f4", (3, 64): "<f8"}
    if (tag, bits) not in kinds:
        raise ValueError("unsupported WAV encoding (send 16-bit PCM)")
    if not MIN_RATE <= sr <= MAX_RATE or not 1 <= channels <= MAX_CHANNELS or align != channels * bits // 8:
        raise ValueError("unsupported sample rate or channel count")
    frames = len(body) // align
    if frames > (max_seconds + 0.5) * sr:
        raise ValueError(f"the clip is longer than {max_seconds:g} s")
    if frames < min_seconds * sr:
        raise ValueError(f"the clip is shorter than {min_seconds:g} s")
    raw = body[:frames * align]
    if bits == 24:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        x = ((b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)) << 8 >> 8).astype(np.float32) / 8388608.0
    else:
        x = np.frombuffer(raw, dtype=kinds[(tag, bits)])
        if tag == 1:
            x = (x.astype(np.float32) - 128.0) / 128.0 if bits == 8 else x.astype(np.float32) / float(2 ** (bits - 1))
    with np.errstate(over="ignore", invalid="ignore"):
        mono = x.reshape(-1, channels).astype(np.float32).mean(axis=1)
    return _checked(mono, sr, max_seconds, min_seconds)


def decode_bytes(data: bytes, max_seconds: float, min_seconds: float = 0.0,
                 formats: Sequence[str] = EXCERPT_FORMATS) -> tuple[np.ndarray, int]:
    """``decode_mono`` for an untrusted clip held in memory; nothing touches the disk and no other program is run.
    The container is told by its first bytes (``sniff``); one that is not audio, or not among ``formats``, raises
    ``UnsupportedAudio``. WAV is read by ``decode_wav`` (numpy only). FLAC, Ogg Vorbis/Opus and MP3 go to
    libsndfile, a C library: its header is read first, so a file that claims more than ``max_seconds``, an odd
    sample rate or many channels is refused before any audio is decoded, and at most ``max_seconds`` + 1 s is ever
    decoded. Raises ValueError with a message fit for the client."""
    kind = sniff(data)
    if kind is None or kind not in formats:
        raise UnsupportedAudio(f"not an audio file this service takes (send {', '.join(formats).upper()})")
    if kind == "wav":
        return decode_wav(data, max_seconds, min_seconds)
    import io

    import soundfile as sf

    try:
        with sf.SoundFile(io.BytesIO(data)) as f:
            if f.format not in {"flac": ("FLAC",), "ogg": ("OGG",), "mp3": ("MPEG", "MP3")}[kind]:
                raise UnsupportedAudio("the file is not what its first bytes say")
            if not MIN_RATE <= f.samplerate <= MAX_RATE or not 1 <= f.channels <= MAX_CHANNELS:
                raise ValueError("unsupported sample rate or channel count")
            # MP3 and Ogg headers may not know their length: read at most the limit plus one second and check after.
            if f.frames > (max_seconds + 1.0) * f.samplerate:
                raise ValueError(f"the clip is longer than {max_seconds:g} s")
            sr = f.samplerate
            block = f.read(int((max_seconds + 1.0) * sr) + 1, dtype="float32", always_2d=True)
    except sf.LibsndfileError as exc:
        raise ValueError("could not decode the audio") from exc
    return _checked(block.mean(axis=1), sr, max_seconds, min_seconds)


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
