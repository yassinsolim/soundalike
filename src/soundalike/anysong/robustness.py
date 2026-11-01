"""Robustness of the live any-song path to everyday edits of a song.

Each sampled catalog track's preview is decoded once, edited (sped up, slowed, key change, tempo change,
noise, phone band, a 10 s excerpt), embedded live like an unseen query and ranked against the index with
nothing excluded. A channel is robust to an edit when the original song still ranks first in the API's
song view (any release of it counts). ``original`` re-embeds the unedited preview as the reference.

    python -m soundalike.anysong.robustness --index /opt/soundalike/index/current \
        --meta /opt/soundalike/index/catalog_meta.npz --previews /data/previews --n 200 \
        --out /data/logs/robustness.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from .audio import decode_mono, fit_clips
from .embed_run import preview_path
from .embedders import REGISTRY
from .eval_index import SongKeys, song_best, song_ranks
from .serve import CHANNELS, Engine, load_meta

Edit = Callable[[np.ndarray, int, np.random.Generator], np.ndarray]


def speed(factor: float) -> Edit:
    """Play faster/slower: tempo and pitch move together ("sped up" / "slowed" edits)."""
    def run(x, sr, rng):
        import soxr

        return soxr.resample(x, sr * factor, sr, quality="HQ").astype(np.float32)
    return run


def pitch(semitones: float) -> Edit:
    def run(x, sr, rng):
        import librosa

        return librosa.effects.pitch_shift(x, sr=sr, n_steps=semitones).astype(np.float32)
    return run


def tempo(rate: float) -> Edit:
    def run(x, sr, rng):
        import librosa

        return librosa.effects.time_stretch(x, rate=rate).astype(np.float32)
    return run


def noise(snr_db: float) -> Edit:
    def run(x, sr, rng):
        power = float(np.mean(x.astype(np.float64) ** 2)) or 1e-8
        return (x + rng.standard_normal(len(x)) * np.sqrt(power / 10 ** (snr_db / 10))).astype(np.float32)
    return run


def phone(x: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    """300-3400 Hz band through an 8 kHz round trip, like a call or a phone speaker recording."""
    import soxr
    from scipy.signal import butter, sosfiltfilt

    y = sosfiltfilt(butter(4, [300, 3400], btype="bandpass", fs=sr, output="sos"), x)
    return soxr.resample(soxr.resample(y, sr, 8000), 8000, sr).astype(np.float32)


def excerpt(start: float, seconds: float) -> Edit:
    def run(x, sr, rng):
        return x[int(start * sr):int((start + seconds) * sr)]
    return run


EDITS: dict[str, Edit] = {
    "original": lambda x, sr, rng: x,
    "sped_up_1.25": speed(1.25),
    "slowed_0.85": speed(0.85),
    "pitch_+2": pitch(2),
    "pitch_-2": pitch(-2),
    "tempo_1.10": tempo(1.10),
    "noise_10db": noise(10),
    "phone": phone,
    "clip_10s": excerpt(10, 10),
}


def summarize(rows: list[tuple[int, int]]) -> dict[str, float | int]:
    """rows: (song rank, raw self rank); 0 = not ranked."""
    song = np.array([r for r, _ in rows], dtype=np.float64)
    raw = np.array([r for _, r in rows], dtype=np.float64)
    found = np.where(song > 0, song, np.inf)
    return {"n": len(rows), "hr@1": round(float((found <= 1).mean()), 4), "hr@10": round(float((found <= 10).mean()), 4),
            "mrr": round(float(np.mean(1 / found)), 4), "median_rank": float(np.median(found)),
            "self_hr@1": round(float(((raw > 0) & (raw <= 1)).mean()), 4)}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="robustness")
    p.add_argument("--index", required=True)
    p.add_argument("--meta", type=Path, required=True)
    p.add_argument("--previews", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=200, help="random indexed tracks with a cached preview")
    p.add_argument("--ids", nargs="*", type=int, default=[], help="track IDs to include before the random sample")
    p.add_argument("--edits", nargs="+", default=list(EDITS), choices=list(EDITS))
    p.add_argument("--channels", nargs="+", default=list(CHANNELS), choices=list(CHANNELS))
    p.add_argument("--workers", type=int, default=3, help="CPU decode/edit threads")
    p.add_argument("--candidates", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)  # noqa: E731

    keys = SongKeys(load_meta(args.meta))
    eng = Engine(args.index, args.channels, args.device, candidates=args.candidates)
    eng.warm()
    rowkeys = {name: keys.rows(ix.ids, ix.id_tensor.device) for name, ix in eng.indexes.items()}
    common = sorted(set.intersection(*(set(ix.ids.tolist()) for ix in eng.indexes.values())))
    rng = random.Random(args.seed)
    pool_ids = [t for t in rng.sample(common, min(len(common), args.n * 3)) if preview_path(args.previews, t).exists()]
    sample = [t for t in args.ids if t in eng.indexes[args.channels[0]].pos] + pool_ids[:args.n]
    rates = sorted({REGISTRY[ch.model].sr for ch in eng.channels})
    log(f"index {[len(ix) for ix in eng.indexes.values()]} sample={len(sample)} edits={args.edits}")

    def prepare(tid: int):
        try:
            mono, sr = decode_mono(preview_path(args.previews, tid))
            g = np.random.default_rng(tid)
            return {e: fit_clips(EDITS[e](mono, sr, g), sr, rates) for e in args.edits}
        except Exception as exc:  # noqa: BLE001 - recorded in the report
            return exc

    rows: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    failures, misses = [], defaultdict(list)
    embed_ms: list[float] = []
    pool = ThreadPoolExecutor(max(1, args.workers))
    futs = [pool.submit(prepare, t) for t in sample[:args.workers * 2]]
    t0 = time.perf_counter()
    with torch.inference_mode():
        for i, tid in enumerate(sample):
            if i + args.workers * 2 < len(sample):
                futs.append(pool.submit(prepare, sample[i + args.workers * 2]))
            res = futs[i].result()
            futs[i] = None
            if isinstance(res, Exception):
                failures.append({"id": tid, "error": f"{type(res).__name__}: {res}"})
                continue
            target = keys.get(tid)
            for edit, clips in res.items():
                wins, te = eng.embed_clips(clips)
                embed_ms.append(sum(te.values()))
                for name, ix in eng.indexes.items():
                    sc = ix.scores(ix.prep_query(wins[name]))
                    sr_ = song_ranks(song_best(sc, rowkeys[name], None), {target})
                    raw = ix.rank_of(sc, [tid])
                    rank = int(sr_[0]) if len(sr_) else 0
                    rows[(edit, name)].append((rank, int(raw[0]) if len(raw) else 0))
                    if rank != 1 and len(misses[(edit, name)]) < 5:
                        misses[(edit, name)].append({"id": tid, "song": keys.names[target], "rank": rank})
            if (i + 1) % 20 == 0 or i + 1 == len(sample):
                rate = (i + 1) / (time.perf_counter() - t0)
                log(f"{i + 1}/{len(sample)} tracks {rate:.2f}/s")
    pool.shutdown()
    report = {"args": {k: str(v) for k, v in vars(args).items()}, "sample": len(sample),
              "index_sizes": {k: len(v) for k, v in eng.indexes.items()},
              "results": {f"{edit}|{name}": summarize(r) for (edit, name), r in rows.items()},
              "misses": {f"{edit}|{name}": m for (edit, name), m in misses.items()},
              "embed_ms_p50": round(float(np.median(embed_ms)), 1) if embed_ms else None,
              "failures": failures, "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{'edit':14s} " + " ".join(f"{c + ' HR@1/HR@10':>22s}" for c in args.channels))
    for edit in args.edits:
        cells = []
        for c in args.channels:
            m = report["results"].get(f"{edit}|{c}")
            cells.append(f"{m['hr@1']:>10.3f}/{m['hr@10']:<11.3f}" if m else f"{'-':>22s}")
        print(f"{edit:14s} " + " ".join(cells))
    log(f"wrote {args.out} failures={len(failures)}")


if __name__ == "__main__":
    sys.exit(main())
