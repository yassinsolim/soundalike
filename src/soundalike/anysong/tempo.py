"""Tempo for Same vibe: a catalog table plus Beat This! for songs outside it.

The catalog table (``tempo.npz``: ``ids``, ``bpm`` float32 with NaN for unknown, ``confidence``, ``meta``) holds
one tempo per indexed track. A song outside the catalog gets the same estimate live from its preview: decode with
soundfile (mono mean), resample to 22,050 Hz with soxr, log-mel spectrogram, Beat This! ``final0`` (Foscarin et al.,
ISMIR 2024; MIT licence, code and weights) on one 30-s chunk in fp16, "minimal" post-processing, then
``60 / median(inter-beat interval)``. Fewer than four beats leaves the tempo unknown.

    python -m soundalike.anysong.tempo --ids index/samples/ids.npy --previews previews --model final0.ckpt --out tempo.npz
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

SR = 22050
CHUNK, BORDER = 1500, 6  # frames of one Beat This! chunk (50 frames/s) and the border it ignores at each end


def decode22(path: str | Path) -> np.ndarray:
    """Mono float32 at 22,050 Hz, at most 31 s; raises on clips shorter than 0.5 s or with non-finite samples."""
    import soundfile as sf
    import soxr

    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(1)
    if len(x) < sr // 2 or not np.isfinite(x).all():
        raise ValueError("clip too short or not finite")
    if sr != SR:
        x = soxr.resample(x, sr, SR)
    return np.ascontiguousarray(x[:31 * SR], dtype=np.float32)


def tempo_of(beats: Sequence[float]) -> tuple[float, float]:
    """(bpm, stability): 60 / median inter-beat interval and the share of intervals within 10 % of the median."""
    beats = np.asarray(beats)
    if len(beats) < 4:
        return float("nan"), 0.0
    ibi = np.diff(beats)
    med = float(np.median(ibi))
    if med <= 0:
        return float("nan"), 0.0
    return 60.0 / med, float(np.mean(np.abs(ibi / med - 1.0) <= 0.10))


class TempoTable:
    """Catalog tempos by track id."""

    def __init__(self, ids: np.ndarray, bpm: np.ndarray):
        if len(ids) != len(bpm):
            raise ValueError("tempo table ids and bpm differ in length")
        self.by_id = {int(t): float(b) for t, b in zip(ids, bpm) if np.isfinite(b) and b > 0}
        self.size = len(ids)

    @classmethod
    def load(cls, path: str | Path) -> TempoTable:
        with np.load(path, allow_pickle=False) as z:
            return cls(z["ids"], z["bpm"])

    def bpm(self, track_id: int) -> float | None:
        return self.by_id.get(int(track_id))

    def __contains__(self, track_id: int) -> bool:
        return int(track_id) in self.by_id


class Tracker:
    """Beat This! beat tracker on the GPU; tempo for decoded 22,050 Hz clips."""

    def __init__(self, checkpoint: str | Path = "final0", device: str = "cuda", fp16: bool = True, batch: int = 64):
        import torch
        from beat_this.inference import load_model
        from beat_this.model.postprocessor import Postprocessor
        from beat_this.preprocessing import LogMelSpect

        self.torch = torch
        self.device = torch.device(device)
        self.model = load_model(str(checkpoint), self.device)
        self.spect = LogMelSpect(device=self.device)
        self.post = Postprocessor(type="minimal")
        self.fp16 = fp16 and self.device.type == "cuda"
        self.batch = batch

    def beats(self, clips: Sequence[np.ndarray]) -> list[np.ndarray]:
        """Beat times in seconds for each clip (only the first chunk: up to 29.9 s)."""
        return [b for b, _ in self.track(clips)]

    def track(self, clips: Sequence[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
        """(beat times, downbeat times) in seconds for each clip (only the first chunk: up to 29.9 s)."""
        torch = self.torch
        with torch.inference_mode():
            specs: list = [None] * len(clips)
            by_len: dict[int, list[int]] = {}
            for i, x in enumerate(clips):
                by_len.setdefault(len(x), []).append(i)
            for idx in by_len.values():
                xb = torch.from_numpy(np.stack([clips[i] for i in idx])).to(self.device)
                sb = torch.log1p(self.spect.log_multiplier * self.spect.spect_class(xb)).transpose(-1, -2)
                for i, s in zip(idx, sb):
                    specs[i] = s
            lens = [min(len(s), CHUNK - BORDER) for s in specs]
            x = torch.zeros((len(specs), CHUNK, specs[0].shape[1]), device=self.device)
            for i, s in enumerate(specs):
                x[i, BORDER:BORDER + lens[i]] = s[:lens[i]]
            beat, down = [], []
            for b0 in range(0, len(x), self.batch):
                with torch.autocast(self.device.type, enabled=self.fp16):
                    y = self.model(x[b0:b0 + self.batch])
                beat.append(y["beat"].float()[:, BORDER:CHUNK - BORDER])
                down.append(y["downbeat"].float()[:, BORDER:CHUNK - BORDER])
            beat, down = torch.cat(beat), torch.cat(down)
            mask = torch.arange(beat.shape[1], device=self.device)[None] < torch.tensor(lens, device=self.device)[:, None]
            beats, downs = self.post(beat.masked_fill(~mask, -1000.0), down.masked_fill(~mask, -1000.0), mask)
        return [(np.asarray(b, np.float32), np.asarray(d, np.float32)) for b, d in zip(beats, downs)]

    def tempo(self, clip: np.ndarray) -> tuple[float, float]:
        return tempo_of(self.beats([clip])[0])


def build(ids: np.ndarray, previews: Path, tracker: Tracker, out: Path, batch: int = 64, workers: int = 8,
          log=print) -> dict:
    """Tempo table for ``ids`` from ``previews/<id % 1000:03d>/<id>.mp3``; unreadable previews get NaN."""
    bpm = np.full(len(ids), np.nan, np.float32)
    confidence = np.zeros(len(ids), np.float32)
    failed = 0
    t0 = time.time()

    def load(t: int):
        try:
            return decode22(previews / f"{int(t) % 1000:03d}" / f"{int(t)}.mp3")
        except Exception:  # noqa: BLE001 - a missing or corrupt preview leaves the tempo unknown
            return None

    with ThreadPoolExecutor(workers) as pool:
        for s in range(0, len(ids), batch):
            clips = list(pool.map(load, ids[s:s + batch]))
            ok = [i for i, c in enumerate(clips) if c is not None]
            failed += len(clips) - len(ok)
            if ok:
                for i, beats in zip(ok, tracker.beats([clips[i] for i in ok])):
                    bpm[s + i], confidence[s + i] = tempo_of(beats)
            if (s // batch) % 100 == 0:
                log(f"{s + len(clips)}/{len(ids)} tracks, {time.time() - t0:.0f} s")
    meta = {"model": "beat_this", "tracks": int(len(ids)), "known": int(np.isfinite(bpm).sum()), "failed": failed,
            "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "secs": round(time.time() - t0, 1)}
    np.savez(out, ids=np.asarray(ids, np.int64), bpm=bpm, confidence=confidence, meta=json.dumps(meta))
    return meta


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="tempo")
    p.add_argument("--ids", type=Path, required=True, help="track ids (.npy), e.g. an index channel's ids.npy")
    p.add_argument("--previews", type=Path, required=True)
    p.add_argument("--model", default="final0", help="Beat This! checkpoint path or published model name")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch", type=int, default=64)
    args = p.parse_args(argv)
    meta = build(np.load(args.ids), args.previews, Tracker(args.model, args.device, batch=args.batch), args.out,
                 args.batch)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
