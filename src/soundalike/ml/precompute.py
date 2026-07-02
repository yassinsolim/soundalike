"""Parallel spectrogram precomputation for large datasets (e.g. FMA-medium).

Turning 25k audio files into mel-spectrograms single-threaded takes ~2 hours;
librosa decode+mel is CPU-bound. This module fans the work out across all cores
with a process pool, is fully resumable (skips already-cached .npy files), and
tolerates the handful of corrupt/short clips FMA is known to contain.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .collect import TrackEntry
from .data import _spec_path


def _worker(args: Tuple[int, str, str]) -> Tuple[int, bool, str]:
    """Compute + save one spectrogram. Returns (track_id, ok, message)."""
    track_id, audio_path, out_path = args
    if os.path.exists(out_path):
        return track_id, True, "cached"
    try:
        # Imports inside the worker so each process initializes its own libs.
        import warnings

        warnings.filterwarnings("ignore")
        from .spectrogram import SpectrogramConfig, load_audio, log_mel_full

        cfg = SpectrogramConfig()
        y = load_audio(audio_path, cfg.sample_rate)
        spec = log_mel_full(y, cfg)
        # Atomic-ish write: save to temp then rename.
        tmp = out_path + ".tmp.npy"
        np.save(tmp, spec)
        os.replace(tmp, out_path)
        return track_id, True, "ok"
    except Exception as exc:  # noqa: BLE001 - report and skip bad files
        return track_id, False, f"{type(exc).__name__}: {exc}"


def precompute_parallel(
    entries: List[TrackEntry],
    spec_dir: Path,
    workers: Optional[int] = None,
    progress=print,
) -> Tuple[int, int]:
    """Compute spectrograms for every entry in parallel.

    Each entry's `preview_url` is a local audio path (FMA) or a URL already
    downloaded; here we assume local paths (see soundalike.ml.fma). Returns
    (n_ok, n_failed).
    """
    spec_dir = Path(spec_dir)
    spec_dir.mkdir(parents=True, exist_ok=True)
    workers = workers or max(1, (os.cpu_count() or 4) - 2)

    tasks = [
        (e.track_id, e.preview_url, str(_spec_path(spec_dir, e.track_id)))
        for e in entries
    ]
    total = len(tasks)
    ok = failed = done = 0
    progress(f"Precomputing {total} spectrograms across {workers} workers...")

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_worker, t) for t in tasks]
        for fut in as_completed(futures):
            _tid, success, _msg = fut.result()
            done += 1
            ok += int(success)
            failed += int(not success)
            if done % 500 == 0 or done == total:
                progress(f"  {done}/{total}  (ok={ok}, failed={failed})")
    progress(f"Done. {ok} cached, {failed} failed.")
    return ok, failed


def main(argv: Optional[list] = None) -> int:
    import argparse

    from .collect import DatasetCollector

    parser = argparse.ArgumentParser(description="Parallel spectrogram precompute.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--spec-dir", required=True)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args(argv)

    entries = DatasetCollector.load(Path(args.manifest))
    precompute_parallel(entries, Path(args.spec_dir), workers=args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
