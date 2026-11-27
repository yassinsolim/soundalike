"""Derive a whitened channel index from a built ``raw`` one, without re-embedding.

A ``raw`` channel scores unit windows by cosine. When a few directions carry most of the variance (MuQ-MuLan: about
20 of 512), they decide every cosine and fine differences between songs are lost. PCA whitening removes the window
mean and scales every principal direction to unit variance:

    y = l2((l2(x) - mean) @ M),   M = V / sd

with ``V`` the eigenvectors and ``sd`` the root eigenvalues of the window covariance, fitted on a seeded random
sample of the base index's windows (docs/ANYSONG_EVAL.md, "Whitened sound channel"). ``<out>/`` then holds, with
prep ``whiten`` in ``meta.json``:

- ``mu.npy`` fp32 [D] and ``whiten.npy`` fp32 [D, D]: the transform, applied to raw query windows by
  ``serve.TwoStageIndex.prep_query`` and to new tracks by ``index_append``;
- ``full.npy`` fp16 [N, W, D]: the whitened unit windows (exact re-scoring);
- ``proj.npy``, ``proj_scale.npy``, ``basis.npy``: stage 1 over EVERY dimension (identity basis), quantized as
  ``index_build.quantize_windows``. Whitening spreads the variance over all dimensions, so a low-dimensional
  projection would lose most of it; 8-bit integers keep the ranking almost exactly at a quarter of fp32's bytes;
- ``ids.npy``, ``norms.npy`` as ``index_build``; no hub penalty (run ``hub_build`` for one).

The base index is left untouched and still holds the raw windows.

    python -m soundalike.anysong.index_whiten --base index_v9/sonic --out index_v11/sonic
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .index_build import quantize_windows
from .serve import prep_windows

FIT_WINDOWS = 300_000


def fit_whitening(full: np.ndarray, windows: int = FIT_WINDOWS, seed: int = 0
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mean [D], matrix [D, D], eigenvalues [D], largest first) of the PCA whitening of ``windows`` unit windows
    drawn without replacement from ``full`` [N, W, D] by ``seed``. float64 on the CPU, so a rerun gives the same
    transform."""
    n, w, d = full.shape
    pick = np.sort(np.random.default_rng(seed).choice(n * w, min(windows, n * w), replace=False))
    if len(pick) <= d:
        raise ValueError(f"whitening {d} dimensions needs more than {d} windows, not {len(pick)}")
    x = np.asarray(full.reshape(n * w, d)[pick], dtype=np.float64)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)
    mean = x.mean(0)
    x -= mean
    evals, evecs = np.linalg.eigh(x.T @ x / (len(x) - 1))
    evals, evecs = evals[::-1], evecs[:, ::-1]
    if evals[-1] <= 1e-12 * evals[0]:
        raise ValueError("the window covariance is singular: these windows cannot be whitened in every dimension")
    return mean.astype(np.float32), (evecs / np.sqrt(evals)).astype(np.float32), evals


def build(base: Path, out: Path, windows: int = FIT_WINDOWS, seed: int = 0, bits: int = 8,
          candidates: int | None = None, device: str = "cpu", chunk: int = 16384) -> dict:
    t0 = time.time()
    base, out = Path(base), Path(out)
    meta = json.loads((base / "meta.json").read_text())
    if meta["prep"] != "raw":
        raise ValueError(f"{base} holds {meta['prep']} windows; whitening starts from a raw channel")
    if bits not in (4, 8, 16):
        raise ValueError("projection bits must be 4, 8 or 16")
    raw = np.load(base / "full.npy", mmap_mode="r")
    ids = np.load(base / "ids.npy")
    n, w, d = raw.shape
    if bits == 4 and d % 2:
        raise ValueError("4-bit projections need an even dimension")
    mean, matrix, evals = fit_whitening(raw, windows, seed)
    mu, m = torch.from_numpy(mean).to(device), torch.from_numpy(matrix).to(device)

    out.mkdir(parents=True, exist_ok=True)
    full = np.lib.format.open_memmap(out / "full.npy.tmp", "w+", np.float16, (n, w, d))
    proj_dtype, proj_dim = {16: (np.float16, d), 8: (np.int8, d), 4: (np.uint8, d // 2)}[bits]
    proj = np.lib.format.open_memmap(out / "proj.npy.tmp", "w+", proj_dtype, (n, w, proj_dim))
    scale = np.empty((n, w), np.float32) if bits < 16 else None
    norms = np.empty((n, w), np.float32)
    for s in range(0, n, chunk):
        x = torch.from_numpy(np.array(raw[s:s + chunk])).to(device).float()
        x16 = prep_windows(x, "whiten", mu, m).half()
        e = s + len(x16)
        full[s:e] = x16.cpu().numpy()
        # As index_build: stage 1 and the norms come from the stored fp16 values.
        p = x16.float()
        if bits == 16:
            proj[s:e] = x16.cpu().numpy()
        else:
            q, sc = quantize_windows(p, bits)
            proj[s:e] = q.cpu().numpy()
            scale[s:e] = sc.cpu().numpy()
        norms[s:e] = (p ** 2).sum(-1).cpu().numpy()
    full.flush()
    proj.flush()
    del full, proj
    (out / "full.npy.tmp").replace(out / "full.npy")
    (out / "proj.npy.tmp").replace(out / "proj.npy")
    np.save(out / "ids.npy", ids)
    np.save(out / "basis.npy", np.eye(d, dtype=np.float32))
    np.save(out / "norms.npy", norms)
    if scale is not None:
        np.save(out / "proj_scale.npy", scale)
    else:
        (out / "proj_scale.npy").unlink(missing_ok=True)
    np.save(out / "mu.npy", mean)
    np.save(out / "whiten.npy", matrix)
    (out / "hub.npy").unlink(missing_ok=True)
    kept = {k: meta[k] for k in ("channel", "model", "mode", "dropped") if k in meta}
    new = kept | {
        "prep": "whiten", "n": n, "windows": w, "dim": d, "proj_dim": d, "proj_bits": bits, "energy": 1.0,
        "candidates": int(candidates or meta.get("candidates", 1000)),
        "ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(),
        "whiten": {"base": str(base), "base_ids_sha256": meta.get("ids_sha256"),
                   "fit_windows": int(min(windows, n * w)), "seed": seed,
                   "eigenvalue_max": float(evals[0]), "eigenvalue_min": float(evals[-1]),
                   # Directions that hold 90% of the raw variance: how concentrated the raw space was.
                   "dims_for_90pct_variance": int(np.searchsorted(np.cumsum(evals) / evals.sum(), 0.9) + 1),
                   "mu_sha256": hashlib.sha256(mean.tobytes()).hexdigest(),
                   "matrix_sha256": hashlib.sha256(matrix.tobytes()).hexdigest()},
        "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "secs": round(time.time() - t0, 1)}
    (out / "meta.json").write_text(json.dumps(new, indent=1))
    return new


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="index_whiten")
    p.add_argument("--base", required=True, type=Path, help="a built raw channel directory (left untouched)")
    p.add_argument("--out", required=True, type=Path, help="the whitened channel directory to write")
    p.add_argument("--fit-windows", type=int, default=FIT_WINDOWS, help="random windows the transform is fitted on")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--proj-bits", type=int, default=8, choices=(4, 8, 16), help="bits of the full-dimension stage 1")
    p.add_argument("--candidates", type=int, default=None, help="exact re-rank depth (default: the base's)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    if args.out.resolve() == args.base.resolve():
        p.error("--out must differ from --base")
    print(json.dumps(build(args.base, args.out, args.fit_windows, args.seed, args.proj_bits, args.candidates,
                           args.device)), flush=True)


if __name__ == "__main__":
    main()
