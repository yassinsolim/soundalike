"""Sounds like v2: MuQ-MuLan windows through a projection head trained on related-artist pairs, plus the scene.

The ``sounds`` channel scores the catalog by cosine MaxSim over projected MuQ-MuLan windows (``serve.Head``: a
residual MLP 512 -> 1024 -> 512 trained with InfoNCE on pairs of songs by Deezer-related artists, artists of every
evaluation seed and every benchmark track excluded). The served list re-ranks the channel's top ``POOL`` songs by
``z(score) + SCENE_WEIGHT * affinity`` (``scene.affinity`` with the shipped parameters), drops confidently
other-language songs in same-language mode and keeps one song per artist. Training, selection and results are in
docs/ANYSONG_EVAL.md#sounds-like-v2.

    python -m soundalike.anysong.sounds convert-head --pt e2_mlp_graph.pt --out head.npz
    python -m soundalike.anysong.sounds build --sonic index/full/sonic --head head.npz --out index/full/sounds
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from collections.abc import Hashable, Sequence
from pathlib import Path

import numpy as np

DEPTH = 1200  # tracks ranked before de-duplication, enough for POOL songs
POOL = 200  # songs re-ranked with the scene
SCENE_WEIGHT = 1.0  # b in z(score) + b * affinity
CANDIDATES = 1500  # exact stage-2 depth of the built index, so the whole DEPTH is scored exactly


def order(scores: Sequence[float], affinities: Sequence[float], scene_weight: float = SCENE_WEIGHT) -> list[int]:
    """Indices of a best-first song pool re-ranked by z(score) + scene_weight * affinity; ties keep pool order."""
    s = np.asarray(scores, dtype=np.float64)
    if not len(s):
        return []
    z = (s - s.mean()) / (s.std() + 1e-9)
    combined = z + scene_weight * np.asarray(affinities, dtype=np.float64)
    return np.lexsort((np.arange(len(s)), -combined)).tolist()


def cap(groups: Sequence[Hashable], k: int, per_group: int = 1) -> list[int]:
    """Indices of the first ``k`` items keeping at most ``per_group`` items per group (one song per artist)."""
    out, seen = [], {}
    for i, g in enumerate(groups):
        if seen.get(g, 0) >= per_group:
            continue
        seen[g] = seen.get(g, 0) + 1
        out.append(i)
        if len(out) >= k:
            break
    return out


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def convert_head(pt: Path, out: Path) -> str:
    """Trained ``Residual`` state dict (``inp``/``outp`` linear layers) -> ``head.npz``; returns its sha256."""
    import torch

    from .serve import Head

    state = torch.load(pt, map_location="cpu", weights_only=True)
    arrays = {"inp_w": state["inp.weight"], "inp_b": state["inp.bias"],
              "out_w": state["outp.weight"], "out_b": state["outp.bias"]}
    Head(*arrays.values())  # validates the shapes
    np.savez(out, **{k: v.float().numpy() for k, v in arrays.items()})
    return _sha256(out)


def build(sonic: Path, head_path: Path, out: Path, dim: int = 128, candidates: int = CANDIDATES,
          device: str = "cuda", chunk: int = 32768) -> dict:
    """Two-stage ``sounds`` index (``index_build`` layout) from a built ``sonic`` channel's prepared windows."""
    import torch

    from .evaluate import l2
    from .serve import ALL_CHANNELS, Head

    t0 = time.time()
    meta_s = json.loads((sonic / "meta.json").read_text())
    if (meta_s["model"], meta_s["prep"]) != ("muq_mulan", "raw"):
        raise ValueError(f"{sonic} is not a MuQ-MuLan sonic index")
    ids = np.load(sonic / "ids.npy")
    full_s = np.load(sonic / "full.npy", mmap_mode="r")
    n, w, d = full_s.shape
    head = Head.load(head_path, device)
    if head.dim != d or not 0 < dim <= d:
        raise ValueError("head or projection dimension does not match the sonic windows")

    def projected(s: int) -> torch.Tensor:
        with torch.inference_mode():
            x = torch.from_numpy(np.array(full_s[s:s + chunk], dtype=np.float32)).to(device)
            return l2(head(x)).half()

    second = torch.zeros(d, d, dtype=torch.float64, device=device)
    for s in range(0, n, chunk):
        x = projected(s).float().reshape(-1, d)
        second += (x.T @ x).double()
    evals, evecs = torch.linalg.eigh(second)
    evals, evecs = evals.flip(0), evecs.flip(1)
    basis = evecs[:, :dim].float().contiguous()
    energy = float(evals[:dim].sum() / evals.sum())

    out.mkdir(parents=True, exist_ok=True)
    full = np.lib.format.open_memmap(out / "full.npy.tmp", "w+", np.float16, (n, w, d))
    proj = np.lib.format.open_memmap(out / "proj.npy.tmp", "w+", np.float16, (n, w, dim))
    norms = np.empty((n, w), np.float32)
    for s in range(0, n, chunk):
        x16 = projected(s)
        full[s:s + len(x16)] = x16.cpu().numpy()
        proj[s:s + len(x16)] = (x16.float() @ basis).half().cpu().numpy()
        norms[s:s + len(x16)] = (x16.float() ** 2).sum(-1).cpu().numpy()
    full.flush()
    proj.flush()
    del full, proj
    (out / "full.npy.tmp").replace(out / "full.npy")
    (out / "proj.npy.tmp").replace(out / "proj.npy")
    np.save(out / "ids.npy", ids)
    np.save(out / "basis.npy", basis.cpu().numpy())
    np.save(out / "norms.npy", norms)
    if head_path.resolve() != (out / "head.npz").resolve():
        shutil.copyfile(head_path, out / "head.npz")
    ch = ALL_CHANNELS["sounds"]
    meta = {"channel": ch.name, "model": ch.model, "prep": ch.prep, "mode": ch.mode,
            "n": n, "windows": w, "dim": d, "proj_dim": dim, "energy": round(energy, 6), "candidates": candidates,
            "ids_sha256": hashlib.sha256(ids.tobytes()).hexdigest(), "head_sha256": _sha256(out / "head.npz"),
            "source": {"sonic_ids_sha256": meta_s.get("ids_sha256"), "sonic_built": meta_s.get("built")},
            "built": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "secs": round(time.time() - t0, 1)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="sounds")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert-head", help="trained head state dict (.pt) -> head.npz")
    c.add_argument("--pt", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("build", help="build the sounds channel from a built sonic channel")
    b.add_argument("--sonic", type=Path, required=True)
    b.add_argument("--head", type=Path, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--dim", type=int, default=128)
    b.add_argument("--candidates", type=int, default=CANDIDATES)
    b.add_argument("--device", default=None, help="default: cuda when available")
    args = p.parse_args(argv)
    if args.cmd == "convert-head":
        print(f"{args.out} sha256 {convert_head(args.pt, args.out)}")
    else:
        import torch

        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(json.dumps(build(args.sonic, args.head, args.out, args.dim, args.candidates, device), indent=1))


if __name__ == "__main__":
    main()
