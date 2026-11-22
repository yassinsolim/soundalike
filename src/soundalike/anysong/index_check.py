"""Is an index directory complete for the v12 service? A check to run on the serving host before the symlink swap; it
reads headers and small tables only, and never touches the GPU.

    python -m soundalike.anysong.index_check /opt/soundalike/index-store/NAME

The directory is everything the service reads that belongs to one index (docs/ANYSONG.md, "Deployment"):

    melody/ samples/ sonic/      index_build / index_whiten output: meta.json, ids.npy, proj.npy, basis.npy, norms.npy,
                                 full.npy, and as the index needs them proj_scale.npy, mu.npy, whiten.npy, hub.npy
    heads/<h2>.npz, .emb.npy     scene head of the listener ranker (--head)
    heads/<h4>.npz, .emb.npy     scene head of the blend and the consensus body (--blend-head)
    playlists/cooc.npz, cf128.npy                       playlist vectors (--blend-cf)
    catalog_meta.npz languages.npz scene.npz core_ids.npy tempo.npz      the catalog tables
    ranker.npz popularity.npz                           listener ranker and the popularity it reads
    match_quantiles.json overall.json                   sound-match and overall-match scales of this index
    consensus.npz consensus_quantiles.json              consensus model and the quantiles of its body score

Exit status 0 when every file is there and they fit each other (rows, dimensions, ids), else 1 with one line per
problem.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from . import listener

CHANNEL_NAMES = ("melody", "samples", "sonic")
TABLES = ("catalog_meta.npz", "languages.npz", "scene.npz", "core_ids.npy", "tempo.npz", "ranker.npz",
          "popularity.npz", "match_quantiles.json", "overall.json", "consensus.npz", "consensus_quantiles.json")


def _array(path: Path, problems: list[str]) -> np.ndarray | None:
    try:
        return np.load(path, mmap_mode="r", allow_pickle=False)
    except Exception as exc:  # noqa: BLE001 - any unreadable file is one problem line
        problems.append(f"{path}: {'missing' if not path.exists() else f'unreadable ({exc})'}")
        return None


def _channel(root: Path, name: str, problems: list[str]) -> tuple[np.ndarray | None, int | None]:
    """Check one channel's index; (its ids, its window dimension)."""
    from .serve import CHANNELS, built_channel

    path = root / name
    try:
        meta = json.loads((path / "meta.json").read_text())
    except Exception as exc:  # noqa: BLE001
        problems.append(f"{path / 'meta.json'}: {'missing' if not (path / 'meta.json').exists() else exc}")
        return None, None
    built = built_channel(CHANNELS[name], meta)
    if built is None:
        problems.append(f"{path}: built for {(meta.get('model'), meta.get('prep'), meta.get('mode'))}, "
                        f"not the {name} channel")
        return None, None
    n, w, d, bits = meta["n"], meta["windows"], meta["dim"], meta.get("proj_bits", 16)
    stored = meta["proj_dim"] // 2 if bits == 4 else meta["proj_dim"]
    want = {"ids.npy": ((n,), None), "full.npy": ((n, w, d), None), "norms.npy": ((n, w), None),
            "basis.npy": ((d, meta["proj_dim"]), None),
            "proj.npy": ((n, w, stored), {16: "float16", 8: "int8", 4: "uint8"}[bits])}
    if bits < 16:
        want["proj_scale.npy"] = ((n, w), None)
    if built.prep in ("center", "whiten"):
        want["mu.npy"] = ((d,), None)
    if built.prep == "whiten":
        want["whiten.npy"] = ((d, d), None)
    if meta.get("hub", {}).get("alpha"):
        want["hub.npy"] = ((n,), None)
    ids = None
    for file, (shape, dtype) in want.items():
        arr = _array(path / file, problems)
        if arr is None:
            continue
        if tuple(arr.shape) != shape or (dtype is not None and arr.dtype.name != dtype):
            problems.append(f"{path / file}: {arr.dtype.name} {tuple(arr.shape)}, meta.json says "
                            f"{dtype or 'any type'} {shape}")
        elif file == "ids.npy":
            ids = np.asarray(arr, dtype=np.int64)
    if ids is not None:
        if len(np.unique(ids)) != len(ids):
            problems.append(f"{path / 'ids.npy'}: track ids repeat")
        if "ids_sha256" in meta and hashlib.sha256(np.asarray(_array(path / "ids.npy", [])).tobytes()).hexdigest() \
                != meta["ids_sha256"]:
            problems.append(f"{path / 'ids.npy'}: not the ids meta.json was written for (ids_sha256)")
    return ids, d


def _head(root: Path, name: str, n: int | None, dim: int | None, problems: list[str]) -> None:
    table = _array(root / "heads" / f"{name}.emb.npy", problems)
    try:
        with np.load(root / "heads" / f"{name}.npz", allow_pickle=False) as z:
            w1, w2 = z["w1"], z["w2"]
            _ = z["b1"], z["b2"]
    except Exception as exc:  # noqa: BLE001
        problems.append(f"{root / 'heads' / f'{name}.npz'}: {exc}")
        return
    if dim is not None and w1.shape[0] != dim:
        problems.append(f"heads/{name}.npz: reads {w1.shape[0]}-dimensional windows, the sonic index has {dim}")
    if table is not None and (table.ndim != 3 or table.shape[2] != w2.shape[1] or (n is not None and len(table) != n)):
        problems.append(f"heads/{name}.emb.npy: {tuple(table.shape)} does not fit the sonic index ({n} songs) or "
                        f"the head ({w2.shape[1]} dimensions)")


def check(root: Path | str, head: str = "h2", blend_head: str = "h4") -> tuple[list[str], dict]:
    """(problems, summary) for the index directory ``root``; no problems = the v12 service can start on it."""
    from . import api, consensus
    from .languages import LanguageTable
    from .scene import SceneTable

    root = Path(root)
    problems: list[str] = []
    ids, dims = {}, {}
    for name in CHANNEL_NAMES:
        ids[name], dims[name] = _channel(root, name, problems)
    sonic = ids["sonic"]
    n = None if sonic is None else len(sonic)
    for name in ("melody", "samples"):
        if sonic is not None and ids[name] is not None and not np.array_equal(np.sort(ids[name]), np.sort(sonic)):
            problems.append(f"{name}/ids.npy: {len(ids[name])} songs that are not the sonic index's {n}")
    for name in dict.fromkeys((head, blend_head)):
        _head(root, name, n, dims["sonic"], problems)
    rows, cf = None, _array(root / "playlists" / "cf128.npy", problems)
    try:
        with np.load(root / "playlists" / "cooc.npz", allow_pickle=False) as z:
            rows = z["rows"]
    except Exception as exc:  # noqa: BLE001
        problems.append(f"{root / 'playlists' / 'cooc.npz'}: {exc}")
    if rows is not None and cf is not None and n is not None and (
            cf.ndim != 2 or len(cf) != len(rows) or len(np.unique(rows)) != len(rows) or rows.min() < 0
            or rows.max() >= n):
        problems.append(f"playlists: {len(rows)} graph rows and vectors {tuple(cf.shape)} do not fit {n} songs")

    def table(file: str, load):
        try:
            return load(root / file)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{root / file}: {'missing' if not (root / file).exists() else exc}")
            return None

    def catalog(path: Path) -> int:
        with np.load(path, allow_pickle=True) as z:
            tids, artists, titles = z["track_ids"], z["artists"], z["titles"]
        if not len(tids) == len(artists) == len(titles):
            raise ValueError("track_ids, artists and titles differ in length")
        return 0 if sonic is None else int(np.isin(sonic, tids).sum())

    def tempo(path: Path) -> int:
        with np.load(path, allow_pickle=False) as z:
            if z["ids"].shape != z["bpm"].shape:
                raise ValueError("ids and bpm differ in length")
            return 0 if sonic is None else int(np.isin(sonic, z["ids"]).sum())

    def quantiles(path: Path):
        return api.checked_match_quantiles(json.loads(path.read_text(encoding="utf-8")))

    named = table("catalog_meta.npz", catalog)
    if named is not None and n and named < n:
        problems.append(f"catalog_meta.npz: names for {named} of the index's {n} songs")
    table("languages.npz", LanguageTable.load)
    table("scene.npz", SceneTable.load)
    core = table("core_ids.npy", lambda path: np.load(path, allow_pickle=False))
    if core is not None and sonic is not None and not np.isin(core, sonic).any():
        problems.append("core_ids.npy: none of its songs is in the index")
    with_tempo = table("tempo.npz", tempo)
    ranker = table("ranker.npz", listener.Ranker.load)
    table("popularity.npz", listener.Popularity.load)
    table("match_quantiles.json", quantiles)
    table("consensus_quantiles.json", quantiles)
    table("overall.json", api._overall_scales)
    model = table("consensus.npz", listener.Regressor.load)
    if model is not None:
        unknown = [f for f in model.features if f not in consensus.FEATURES]
        if unknown:
            problems.append(f"consensus.npz: reads features this code does not compute: {unknown}")
    return problems, {"songs": n, "named": named, "with_tempo": with_tempo,
                      "graph": None if rows is None else len(rows),
                      "ranker": None if ranker is None else ranker.meta or "on",
                      "consensus": None if model is None else model.meta or "on"}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="index_check", description=__doc__.split("\n")[0])
    p.add_argument("index", type=Path, help="the index directory (what index/current will point at)")
    p.add_argument("--head", default="h2", help="name of the listener ranker's scene head under heads/")
    p.add_argument("--blend-head", default="h4", help="name of the blend's and the consensus body's scene head")
    args = p.parse_args(argv)
    problems, summary = check(args.index, args.head, args.blend_head)
    for line in problems:
        print(f"PROBLEM {line}")
    print(json.dumps({"ok": not problems, "index": str(args.index)} | summary))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
