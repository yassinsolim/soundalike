"""Build a compact two-stage serving index from window-embedding shards.

Stage 1 scores every track on the GPU using an uncentred PCA projection of the prepared windows
(the projection that best preserves inner products); stage 2 re-scores the top candidates exactly
from full-dimensional prepared windows that stay memory-mapped on disk. Per channel, ``<out>/<name>/``:

- ``ids.npy``    int64 [N]
- ``full.npy``   fp16 [N, W, D]  prepared windows (exact rerank)
- ``proj.npy``   stage-1 projection, loaded onto the GPU: fp16 [N, W, d]; with ``--proj-bits`` 8, int8 [N, W, d];
                 with 4, uint8 [N, W, d/2] holding two values per byte (low nibble first, each stored as value + 8)
- ``proj_scale.npy`` fp32 [N, W]  per-window scale of a quantized projection (value = integer * scale)
- ``basis.npy``  fp32 [D, d]
- ``norms.npy``  fp32 [N, W]     squared norms of the prepared windows (distance scoring modes)
- ``mu.npy``     fp32 [D]        window mean over the corpus (``center`` prep only)
- ``meta.json``  channel, shapes, retained energy and provenance

    python -m soundalike.anysong.index_build --emb emb_full --out index/full --ids catalog_ids.txt \
        --dims melody=256 samples=384 sonic=128 --candidates melody=1000 samples=2000 sonic=1000 \
        --meta catalog_meta.npz --max-song-tracks 100

Defaults come from stage-1 recall on the research corpus (development queries, candidate counts scaled
to the 271k catalog): these keep ~99% of the brute-force top-100 per channel. Quantized projections use one
symmetric scale per window: with a per-dimension scale, a few extreme melody tracks stretch the scale and coarsen
every other track (see docs/ANYSONG_EVAL.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .api import song_key
from .serve import CHANNELS, Channel, load_meta, prep_windows


def plan_rows(model_dir: Path, keep: set[int] | None, drop: frozenset[int] | set[int] = frozenset()
              ) -> tuple[list[tuple[Path, np.ndarray]], np.ndarray]:
    """Shard order with first occurrence of each id kept (``embed_run.load_store`` semantics)."""
    seen: set[int] = set()
    plan, ids = [], []
    for shard in sorted(model_dir.glob("part-*.npz")):
        with np.load(shard) as z:
            sid = z["ids"]
        rows = []
        for i, t in enumerate(sid.tolist()):
            if t not in seen:
                seen.add(t)
                if (keep is None or t in keep) and t not in drop:
                    rows.append(i)
        if rows:
            plan.append((shard, np.asarray(rows)))
            ids.append(sid[rows])
    if not plan:
        raise ValueError(f"no rows to index in {model_dir}")
    return plan, np.concatenate(ids).astype(np.int64)


def iter_windows(plan: list[tuple[Path, np.ndarray]]):
    for shard, rows in plan:
        with np.load(shard) as z:
            yield z["win"][rows]


def crowded_songs(meta: dict[int, tuple[str, str]], pool: set[int] | None, limit: int
                  ) -> tuple[set[int], dict[str, int]]:
    """Tracks whose song, grouped as the API groups releases, has more than ``limit`` tracks in ``pool``.

    The song view keeps each song's best-scoring track, so a "song" made of thousands of tracks gets thousands
    of chances to reach every top 10. In the 271k catalog the only such family is a radio show's spoken
    segments (2,031 "A State Of Trance (ASOT n) (Track Recap / Coming Up / Intro, ...)" tracks); the next
    largest song has 25 tracks.
    """
    families: dict[tuple[str, str], list[int]] = {}
    for t, (artist, title) in meta.items():
        if pool is None or t in pool:
            families.setdefault(song_key(artist, title), []).append(t)
    drop: set[int] = set()
    report: dict[str, int] = {}
    for key, tracks in families.items():
        if len(tracks) > limit:
            drop.update(tracks)
            report[" - ".join(key)] = len(tracks)
    return drop, report


def quantize_windows(x: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-window quantization of projections [..., d] -> (stored integers, scale [...]).

    8 bits store int8 values in -127..127; 4 bits pack values in -7..7 two per byte (low nibble first, + 8)."""
    if bits not in (4, 8):
        raise ValueError("projection bits must be 4, 8 or 16")
    levels = 2 ** (bits - 1) - 1
    scale = (x.abs().amax(-1) / levels).clamp_min(torch.finfo(torch.float32).tiny)
    q = (x / scale[..., None]).round().clamp(-levels, levels).to(torch.int8)
    if bits == 8:
        return q, scale
    if q.shape[-1] % 2:
        raise ValueError("4-bit projections need an even dimension")
    u = (q + 8).to(torch.uint8)
    return u[..., 0::2] | (u[..., 1::2] << 4), scale


def build_channel(channel: Channel, emb: Path, out: Path, dim: int, keep: set[int] | None, device: str,
                  candidates: int = 1000, drop: frozenset[int] | set[int] = frozenset(),
                  dropped: dict | None = None, bits: int = 16) -> dict:
    t0 = time.time()
    if bits not in (4, 8, 16):
        raise ValueError("projection bits must be 4, 8 or 16")
    if bits == 4 and dim % 2:
        raise ValueError("4-bit projections need an even dimension")
    plan, ids = plan_rows(emb / channel.model, keep, drop)
    first = next(iter_windows(plan[:1]))
    n, w, d_full = len(ids), first.shape[1], first.shape[2]
    if not 0 < dim <= d_full:
        raise ValueError(f"{channel.name}: projection dim {dim} must be in 1..{d_full}")
    mu = None
    if channel.prep == "center":
        total = np.zeros(d_full, np.float64)
        for win in iter_windows(plan):
            total += win.astype(np.float32).reshape(-1, d_full).sum(0, dtype=np.float64)
        mu = torch.from_numpy((total / (n * w)).astype(np.float32)).to(device)

    def prepared(win: np.ndarray) -> torch.Tensor:
        return prep_windows(torch.from_numpy(win).to(device).float(), channel.prep, mu)

    second = torch.zeros(d_full, d_full, dtype=torch.float64, device=device)
    for win in iter_windows(plan):
        x = prepared(win).reshape(-1, d_full)
        second += (x.T @ x).double()
    evals, evecs = torch.linalg.eigh(second)
    evals, evecs = evals.flip(0), evecs.flip(1)
    basis = evecs[:, :dim].float().contiguous()
    energy = float(evals[:dim].sum() / evals.sum())

    out.mkdir(parents=True, exist_ok=True)
    full = np.lib.format.open_memmap(out / "full.npy.tmp", "w+", np.float16, (n, w, d_full))
    proj_dtype, proj_dim = {16: (np.float16, dim), 8: (np.int8, dim), 4: (np.uint8, dim // 2)}[bits]
    proj = np.lib.format.open_memmap(out / "proj.npy.tmp", "w+", proj_dtype, (n, w, proj_dim))
    scale = np.empty((n, w), np.float32) if bits < 16 else None
    norms = np.empty((n, w), np.float32)
    s = 0
    for win in iter_windows(plan):
        x = prepared(win)
        x16 = x.half()
        full[s:s + len(x)] = x16.cpu().numpy()
        # Project and take norms from the stored fp16 values so both stages see the same vectors.
        p = x16.float() @ basis
        if bits == 16:
            proj[s:s + len(x)] = p.half().cpu().numpy()
        else:
            q, sc = quantize_windows(p, bits)
            proj[s:s + len(x)] = q.cpu().numpy()
            scale[s:s + len(x)] = sc.cpu().numpy()
        norms[s:s + len(x)] = (x16.float() ** 2).sum(-1).cpu().numpy()
        s += len(x)
    full.flush()
    proj.flush()
    del full, proj
    (out / "full.npy.tmp").replace(out / "full.npy")
    (out / "proj.npy.tmp").replace(out / "proj.npy")
    np.save(out / "ids.npy", ids)
    np.save(out / "basis.npy", basis.cpu().numpy())
    np.save(out / "norms.npy", norms)
    if scale is not None:
        np.save(out / "proj_scale.npy", scale)
    else:
        (out / "proj_scale.npy").unlink(missing_ok=True)
    if mu is not None:
        np.save(out / "mu.npy", mu.cpu().numpy())
    meta = {"channel": channel.name, "model": channel.model, "prep": channel.prep, "mode": channel.mode,
            "n": n, "windows": w, "dim": d_full, "proj_dim": dim, "proj_bits": bits, "energy": round(energy, 6),
            "candidates": candidates,
            "ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(), "shards": len(plan),
            "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "secs": round(time.time() - t0, 1)}
    if dropped:
        meta["dropped"] = dropped
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="index_build")
    p.add_argument("--emb", required=True, type=Path, help="directory with <model>/part-*.npz shards")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--ids", type=Path, default=None, help="optional whitespace-separated track ids to keep")
    p.add_argument("--dims", nargs="+", default=["melody=256", "samples=384", "sonic=128"],
                   help="channel=projection_dim; only listed channels are built")
    p.add_argument("--candidates", nargs="+", default=["melody=1000", "samples=2000", "sonic=1000"],
                   help="channel=default exact re-rank depth stored in meta.json (1000 if unlisted)")
    p.add_argument("--proj-bits", nargs="+", default=[],
                   help="channel=bits for the stage-1 projection: 16 (fp16, default), 8 or 4 (per-window scale)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--meta", type=Path, default=None, help="catalog_meta.npz (song identity, as the API)")
    p.add_argument("--max-song-tracks", type=int, default=0,
                   help="drop every track of a song with more tracks than this (needs --meta; 0 = keep all)")
    args = p.parse_args(argv)
    keep = {int(x) for x in args.ids.read_text().split()} if args.ids else None
    drop: set[int] = set()
    dropped = None
    if args.max_song_tracks:
        if args.meta is None:
            p.error("--max-song-tracks needs --meta")
        drop, songs = crowded_songs(load_meta(args.meta), keep, args.max_song_tracks)
        dropped = {"max_song_tracks": args.max_song_tracks, "tracks": len(drop), "songs": songs,
                   "meta_sha256": hashlib.sha256(args.meta.read_bytes()).hexdigest()}
        print(json.dumps({"dropped": dropped}), flush=True)
    cands = {k: int(v) for k, v in (s.split("=") for s in args.candidates)}
    bits = {k: int(v) for k, v in (s.split("=") for s in args.proj_bits)}
    if any(b not in (4, 8, 16) for b in bits.values()):
        p.error("--proj-bits values must be 4, 8 or 16")
    for spec in args.dims:
        name, dim = spec.split("=")
        meta = build_channel(CHANNELS[name], args.emb, args.out / name, int(dim), keep, args.device,
                             cands.get(name, 1000), drop, dropped, bits.get(name, 16))
        print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
