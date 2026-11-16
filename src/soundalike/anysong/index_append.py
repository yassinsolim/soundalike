"""Append new tracks to a built two-stage index without touching its existing rows.

``index_build`` fits the stage-1 basis (and, for ``center`` channels, the window mean) on every row it is given, so
rebuilding a grown catalog changes all stored rows a little and needs every old shard. A small addition does not
need that: this writes ``<out>/<channel>/`` as a copy of ``<base>/<channel>/`` followed by the new tracks, prepared
with the base's ``mu.npy`` and ``basis.npy`` and quantized like the base (``index_build.quantize_windows``). Old rows
keep their bytes, so every score between old tracks is unchanged; the basis keeps the energy it had on the base rows.

    python -m soundalike.anysong.index_append --base index_1m --emb emb_new --ids new_ids.txt --out index_v8
    python -m soundalike.anysong.hub_build --index index_v8 --meta catalog_meta.npz

``hub.npy`` is not copied: new tracks change their neighbours' hub scores, so run ``hub_build`` on the result (the
``hub`` settings of the base are kept in ``meta.json`` under ``appended.base_hub``). Refit with ``index_build`` once
additions are no longer small against the catalog.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from .index_build import iter_windows, plan_rows, quantize_windows
from .serve import CHANNELS, Channel, prep_windows

_COPIED = ("basis.npy", "mu.npy")
_BLOCK = 64 << 20


def _grow(src: Path, dst: Path, rows: int) -> np.memmap:
    """``dst`` = the array in ``src`` with ``rows`` more (unset) rows; returns it memory-mapped for writing."""
    base = np.load(src, mmap_mode="r")
    out = np.lib.format.open_memmap(dst, "w+", base.dtype, (len(base) + rows, *base.shape[1:]))
    offset, left = out.offset, base.nbytes
    del out
    with open(src, "rb") as fin, open(dst, "r+b") as fout:
        fin.seek(base.offset)
        fout.seek(offset)
        while left:
            block = fin.read(min(_BLOCK, left))
            if not block:
                raise OSError(f"{src} is shorter than its header says")
            fout.write(block)
            left -= len(block)
    return np.load(dst, mmap_mode="r+")


def append_channel(channel: Channel, base: Path, emb: Path, out: Path, keep: set[int] | None,
                   device: str = "cpu") -> dict:
    t0 = time.time()
    meta = json.loads((base / "meta.json").read_text())
    if (meta["model"], meta["prep"], meta["mode"]) != (channel.model, channel.prep, channel.mode):
        raise ValueError(f"{base} was not built for {channel}")
    bits, dim = int(meta.get("proj_bits", 16)), int(meta["proj_dim"])
    base_ids = np.load(base / "ids.npy")
    plan, ids = plan_rows(emb / channel.model, keep, set(base_ids.tolist()))
    basis = torch.from_numpy(np.load(base / "basis.npy")).to(device)
    mu = torch.from_numpy(np.load(base / "mu.npy")).to(device) if (base / "mu.npy").exists() else None
    if channel.prep == "center" and mu is None:
        raise ValueError(f"{base} has no mu.npy")
    first = next(iter_windows(plan[:1]))
    if first.shape[1:] != (meta["windows"], meta["dim"]) or tuple(basis.shape) != (meta["dim"], dim):
        raise ValueError(f"{channel.name}: new windows {first.shape[1:]} do not fit the base index")
    n0, m = len(base_ids), len(ids)
    out.mkdir(parents=True, exist_ok=True)
    full = _grow(base / "full.npy", out / "full.npy.tmp", m)
    proj = _grow(base / "proj.npy", out / "proj.npy.tmp", m)
    scale = np.empty((m, meta["windows"]), np.float32) if bits < 16 else None
    norms = np.empty((m, meta["windows"]), np.float32)
    s = 0
    for win in iter_windows(plan):
        x16 = prep_windows(torch.from_numpy(win).to(device).float(), channel.prep, mu).half()
        e = s + len(x16)
        full[n0 + s:n0 + e] = x16.cpu().numpy()
        # As index_build: project and take norms from the stored fp16 values.
        p = x16.float() @ basis
        if bits == 16:
            proj[n0 + s:n0 + e] = p.half().cpu().numpy()
        else:
            q, sc = quantize_windows(p, bits)
            proj[n0 + s:n0 + e] = q.cpu().numpy()
            scale[s:e] = sc.cpu().numpy()
        norms[s:e] = (x16.float() ** 2).sum(-1).cpu().numpy()
        s = e
    full.flush()
    proj.flush()
    del full, proj
    (out / "full.npy.tmp").replace(out / "full.npy")
    (out / "proj.npy.tmp").replace(out / "proj.npy")
    all_ids = np.concatenate([base_ids, ids]).astype(np.int64)
    np.save(out / "ids.npy", all_ids)
    np.save(out / "norms.npy", np.concatenate([np.load(base / "norms.npy"), norms]))
    if scale is not None:
        np.save(out / "proj_scale.npy", np.concatenate([np.load(base / "proj_scale.npy"), scale]))
    for name in _COPIED:
        if (base / name).exists():
            shutil.copyfile(base / name, out / name)
    (out / "hub.npy").unlink(missing_ok=True)
    new = dict(meta, n=int(n0 + m), ids_sha256=hashlib.sha256(all_ids.tobytes()).hexdigest())
    new["appended"] = {"base": str(base), "base_n": int(n0), "base_ids_sha256": meta["ids_sha256"], "rows": int(m),
                       "shards": len(plan), "base_hub": new.pop("hub", None),
                       "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "secs": round(time.time() - t0, 1)}
    (out / "meta.json").write_text(json.dumps(new, indent=1))
    return new


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="index_append")
    p.add_argument("--base", required=True, type=Path, help="index_build output to extend (left untouched)")
    p.add_argument("--emb", required=True, type=Path, help="directory with <model>/part-*.npz shards of new tracks")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--ids", type=Path, default=None, help="optional whitespace-separated track ids to add")
    p.add_argument("--channels", nargs="+", default=list(CHANNELS), choices=list(CHANNELS))
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)
    if args.out.resolve() == args.base.resolve():
        p.error("--out must differ from --base")
    keep = {int(x) for x in args.ids.read_text().split()} if args.ids else None
    for name in args.channels:
        meta = append_channel(CHANNELS[name], args.base / name, args.emb, args.out / name, keep, args.device)
        print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
