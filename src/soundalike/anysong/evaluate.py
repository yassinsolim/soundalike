"""Offline retrieval evaluation of candidate any-song embeddings.

Corpus = every embedded track (random catalogue distractors + benchmark tracks).
Each benchmark is turned into queries with a set of positives; the query itself
is never retrievable. Metrics per (benchmark, split): mAP, MRR, HR@1/10/100,
P@10 and R@100. Model/config selection must use ``development`` splits only;
``held_out`` (and the Gen Alpha ``test`` group) are for reporting.

    python -m soundalike.anysong.evaluate --emb C:/anysong-lab/emb --bench C:/anysong-lab/bench \
        --catalog C:/anysong-lab/catalog_meta.npz --model muq_mulan --view track --prep raw center csls
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .deezer import normalize

KS = (1, 10, 100)


@dataclass(frozen=True)
class Query:
    bench: str
    key: str
    split: str
    q: int
    pos: frozenset[int]
    artists: frozenset[str] = field(default_factory=frozenset)


def _artist_keys(artist: str | list[str] | None) -> set[str]:
    from .bench_build import _split_credit

    names = artist if isinstance(artist, list) else _split_credit(artist or "")
    out = set()
    for n in names:
        out.update(normalize(x) for x in _split_credit(n))
    return {a for a in out if a}


def load_benchmarks(bench_dir: Path) -> tuple[list[Query], dict[int, set[str]]]:
    queries: list[Query] = []
    artists: dict[int, set[str]] = defaultdict(set)

    ga = bench_dir / "genalpha.json"
    if ga.exists():
        members = [m for m in json.loads(ga.read_text(encoding="utf-8"))["members"] if m.get("deezer_id")]
        ids = [int(m["deezer_id"]) for m in members]
        for m in members:
            artists[int(m["deezer_id"])] |= _artist_keys(m["artists"])
        for m, i in zip(members, ids):
            queries.append(Query("genalpha", str(i), "test", i, frozenset(set(ids) - {i}),
                                 frozenset(artists[i])))

    cov = bench_dir / "mb_cover_groups.json"
    if cov.exists():
        for g in json.loads(cov.read_text(encoding="utf-8"))["groups"]:
            ids = [int(m["deezer_id"]) for m in g["members"]]
            for m in g["members"]:
                artists[int(m["deezer_id"])] |= _artist_keys(m["artist"])
            for i in ids:
                queries.append(Query("mb_covers", f"{g['id']}:{i}", g["split"], i,
                                     frozenset(set(ids) - {i}), frozenset(artists[i])))

    for fname, bench in (("mb_samples_pairs.json", "mb_samples"), ("v4_pairs.json", "v4"),
                         ("sampleid_pairs.json", "sampleid")):
        f = bench_dir / fname
        if not f.exists():
            continue
        fwd: dict[tuple[int, str], set[int]] = defaultdict(set)
        rev: dict[tuple[int, str], set[int]] = defaultdict(set)
        categories: dict[tuple[str, int, str], set[int]] = defaultdict(set)
        for p in json.loads(f.read_text(encoding="utf-8"))["pairs"]:
            q, t = p["query"].get("deezer_id"), p["target"].get("deezer_id")
            if not q or not t or q == t:
                continue
            q, t = int(q), int(t)
            artists[q] |= _artist_keys(p["query"]["artist"])
            artists[t] |= _artist_keys(p["target"]["artist"])
            fwd[(q, p["split"])].add(t)
            rev[(t, p["split"])].add(q)
            if bench == "v4" and p.get("evidence_category"):
                categories[(p["evidence_category"], q, p["split"])].add(t)
        for (q, split), pos in fwd.items():
            queries.append(Query(bench, str(q), split, q, frozenset(pos), frozenset(artists[q])))
        for (category, q, split), pos in categories.items():
            queries.append(Query(f"v4_{category}", str(q), split, q, frozenset(pos),
                                 frozenset(artists[q])))
        if bench == "mb_samples":
            for (t, split), pos in rev.items():
                queries.append(Query("mb_samples_rev", str(t), split, t, frozenset(pos), frozenset(artists[t])))
    return queries, artists


def load_catalog_artists(path: Path, artists: dict[int, set[str]]) -> None:
    z = np.load(path, allow_pickle=True)
    for tid, a in zip(z["track_ids"], z["artists"]):
        artists[int(tid)] |= _artist_keys(str(a))


def load_keys(model_dir: Path, keys: list[str]) -> dict[str, np.ndarray]:
    parts: dict[str, list[np.ndarray]] = defaultdict(list)
    for s in sorted(model_dir.glob("part-*.npz")):
        with np.load(s) as z:
            parts["ids"].append(z["ids"])
            for k in keys:
                parts[k].append(z[k])
    if not parts:
        raise SystemExit(f"no shards in {model_dir}")
    out = {k: np.concatenate(v) for k, v in parts.items()}
    _, first = np.unique(out["ids"], return_index=True)
    first.sort()
    return {k: v[first] for k, v in out.items()}


# ---------------------------------------------------------------- preprocessing

def l2(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def prep_features(x: torch.Tensor, prep: str) -> torch.Tensor:
    """x: [N, D] or [N, W, D] float32 on device."""
    if prep == "none":
        return x
    if prep in ("raw", "csls"):
        return l2(x)
    flat = x.reshape(-1, x.shape[-1])
    mu = flat.mean(0)
    if prep in ("center", "center_csls"):
        return l2(x - mu)
    if prep.startswith("pcaw"):  # pcaw256 -> PCA whitening to 256 dims, eigenvalue-regularised
        k = int(prep[4:].split("_")[0] or 256)
        xc = flat - mu
        cov = (xc.T @ xc) / max(len(xc) - 1, 1)
        evals, evecs = torch.linalg.eigh(cov.double())
        evals, evecs = evals.flip(0)[:k].float(), evecs.flip(1)[:, :k].float()
        w = evecs / (evals + 0.1 * evals.mean()).sqrt()
        return l2((x - mu) @ w)
    raise ValueError(prep)


def hub_penalty(feats: torch.Tensor, k: int = 10, chunk: int = 2048,
                mode: str = "dot") -> torch.Tensor:
    """CSLS reference term: mean similarity of each item to its k nearest corpus items."""
    if k < 1 or chunk < 1:
        raise ValueError("CSLS k and chunk must be positive")
    if len(feats) < 2:
        return torch.zeros(len(feats), device=feats.device)
    k = min(k, len(feats) - 1)
    if feats.dim() == 3:
        chunk = min(chunk, 48)
    r = torch.empty(len(feats), device=feats.device)
    for s in range(0, len(feats), chunk):
        sim = score_block(feats[s:s + chunk], feats, mode)
        sim[torch.arange(sim.shape[0], device=sim.device),
            torch.arange(s, s + sim.shape[0], device=sim.device)] = -torch.inf
        r[s:s + chunk] = sim.topk(k, dim=1).values.mean(1)
    return r


# ---------------------------------------------------------------- scoring

def _bpwr(dist: torch.Tensor) -> torch.Tensor:
    """CLEWS "best pairs without replacement": greedy 1:1 shingle matching, mean matched distance.
    dist: [B, N, S1, S2] -> [B, N]."""
    if dist.shape[3] < dist.shape[2]:
        dist = dist.transpose(2, 3)
    big = torch.finfo(dist.dtype).max / 4
    d = dist.clone()
    total = torch.zeros(dist.shape[:2], device=dist.device)
    n = dist.shape[2]
    for _ in range(n):
        flat = d.flatten(2)
        mn, arg = flat.min(-1)
        total += mn
        r, c = arg // d.shape[3], arg % d.shape[3]
        d.scatter_(2, r[..., None, None].expand(-1, -1, 1, d.shape[3]), big)
        d.scatter_(3, c[..., None, None].expand(-1, -1, d.shape[2], 1), big)
    return total / n


DISTANCE_MODES = ("bpwr", "nsq_min")


def window_scores(sims: torch.Tensor, mode: str, qn: torch.Tensor | None = None, dn: torch.Tensor | None = None,
                  dim: int | None = None) -> torch.Tensor:
    """Window-pair dot products [B, N, W1, W2] -> track scores [B, N] (higher is closer).
    Distance modes also need squared window norms ``qn`` [B, W1], ``dn`` [N, W2] and the feature ``dim``."""
    if mode in DISTANCE_MODES:  # CLEWS: un-normalised shingle embeddings, nsqeuc distance
        dist = (qn[:, None, :, None] + dn[None, :, None, :] - 2 * sims).clamp_min(0) / dim
        return -(_bpwr(dist) if mode == "bpwr" else dist.amin(dim=(2, 3)))
    if mode == "maxsim":
        return sims.amax(dim=(2, 3))
    if mode.startswith("top"):  # sampleid "top-k retrieval": mean of the k best chunk-pair sims
        return sims.flatten(2).topk(int(mode[3:]), dim=-1).values.mean(-1)
    if mode == "chamfer":
        return 0.5 * (sims.amax(3).mean(2) + sims.amax(2).mean(2))
    raise ValueError(mode)


def score_block(qf: torch.Tensor, df: torch.Tensor, mode: str) -> torch.Tensor:
    if qf.dim() == 2:
        return qf @ df.T
    # window sets [B, W, D] x [N, W, D]
    sims = torch.einsum("bwd,nvd->bnwv", qf, df)
    if mode in DISTANCE_MODES:
        return window_scores(sims, mode, (qf ** 2).sum(-1), (df ** 2).sum(-1), qf.shape[-1])
    return window_scores(sims, mode)


def evaluate(feats: torch.Tensor, ids: np.ndarray, queries: list[Query], artists: dict[int, set[str]],
             mode: str = "dot", penalty: torch.Tensor | None = None, xartist: bool = False,
             batch: int = 256) -> dict[str, dict[str, float | int | None]]:
    """Returns {"bench/split": metrics}. ``xartist`` drops candidates sharing an artist with the query."""
    index = {int(t): i for i, t in enumerate(ids)}
    if len(index) != len(ids) or len(feats) != len(ids):
        raise ValueError("Expected unique track IDs matching embedding rows")
    if not torch.isfinite(feats).all():
        raise ValueError("Cannot evaluate non-finite embeddings")
    dev = feats.device
    track_ids = torch.as_tensor(ids, device=dev)
    coverage: dict[str, dict[str, int]] = {}
    usable = []
    for q in queries:
        key = f"{q.bench}/{q.split}"
        c = coverage.setdefault(key, {"n_expected": 0, "missing_query": 0, "no_positive": 0,
                                     "partial_positives": 0, "artist_filtered": 0})
        c["n_expected"] += 1
        if q.q not in index:
            c["missing_query"] += 1
            continue
        positives = q.pos - {q.q}
        present = positives & index.keys()
        if not present:
            c["no_positive"] += 1
            continue
        c["partial_positives"] += int(len(present) < len(positives))
        usable.append(q)
    art_ix: dict[str, list[int]] = defaultdict(list)
    if xartist:
        for t, i in index.items():
            for a in artists.get(t, ()):
                art_ix[a].append(i)
    per: dict[str, list[dict[str, float]]] = defaultdict(list)
    if mode != "dot":
        batch = min(batch, 48)
    for s in range(0, len(usable), batch):
        qs = usable[s:s + batch]
        qi = torch.tensor([index[q.q] for q in qs], device=dev)
        sc = score_block(feats[qi], feats, mode) if mode != "dot" else feats[qi] @ feats.T
        if penalty is not None:
            # CSLS = 2*cos - r(query) - r(candidate); the query term is constant.
            sc = 2 * sc - penalty[None, :]
        sc[torch.arange(len(qs), device=dev), qi] = -torch.inf
        for j, q in enumerate(qs):
            row = sc[j]
            pos = [index[p] for p in q.pos if p in index and p != q.q]
            if xartist and q.artists:
                drop = {i for a in q.artists for i in art_ix.get(a, ())} - {index[q.q]}
                if drop:
                    row = row.clone()
                    row[torch.tensor(sorted(drop), device=dev)] = -torch.inf
                    pos = [p for p in pos if p not in drop]
                if not pos:
                    coverage[f"{q.bench}/{q.split}"]["artist_filtered"] += 1
                    continue
            pi = torch.tensor(pos, device=dev)
            pv = row[pi]
            # Break score ties by Deezer ID, independent of shard order.
            ahead = (row[None, :] > pv[:, None]) | (
                (row[None, :] == pv[:, None]) & (track_ids[None, :] < track_ids[pi, None]))
            ranks = (ahead.sum(1) + 1).sort().values.cpu().numpy()
            n_pos = len(ranks)
            ap = float(np.mean(np.arange(1, n_pos + 1) / ranks))
            m = {"ap": ap, "rr": 1.0 / ranks[0], "p@10": float((ranks <= 10).sum()) / 10,
                 "r@100": float((ranks <= 100).mean())}
            for k in KS:
                m[f"hr@{k}"] = float(ranks[0] <= k)
            per[f"{q.bench}/{q.split}"].append(m)
    out = {}
    for key, counts in sorted(coverage.items()):
        rows = per[key]
        metrics = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} if rows else {
            k: None for k in ("ap", "rr", "p@10", "r@100", *(f"hr@{k}" for k in KS))}
        out[key] = metrics | counts | {"n": len(rows)}
    return out


# ---------------------------------------------------------------- views

def build_view(emb_dir: Path, model: str, view: str, device: str) -> tuple[np.ndarray, torch.Tensor, str]:
    """view: track | win:maxsim | win:chamfer | L<k> | L<a>-<b> (mean of layers) | LW<k>:maxsim"""
    d = emb_dir / model
    if view == "track":
        st = load_keys(d, ["track"])
        return st["ids"], torch.from_numpy(st["track"].astype(np.float32)).to(device), "dot"
    if view.startswith("win:"):
        st = load_keys(d, ["win"])
        return st["ids"], torch.from_numpy(st["win"].astype(np.float32)).to(device), view.split(":")[1]
    if view.startswith("LW"):
        layer, mode = view[2:].split(":")
        st = load_keys(d, ["layer_win"])
        x = st["layer_win"][:, int(layer)].astype(np.float32)
        return st["ids"], torch.from_numpy(x).to(device), mode
    if view.startswith("L"):
        st = load_keys(d, ["layers"])
        spec = view[1:]
        if "-" in spec:
            a, b = (int(v) for v in spec.split("-"))
            x = st["layers"][:, a:b + 1].astype(np.float32)
            x = x - x.mean(0, keepdims=True)
            x = x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)
            x = x.mean(1)
        else:
            x = st["layers"][:, int(spec)].astype(np.float32)
        return st["ids"], torch.from_numpy(x).to(device), "dot"
    raise ValueError(view)


def n_layers(emb_dir: Path, model: str) -> int:
    first = sorted((emb_dir / model).glob("part-*.npz"))[0]
    with np.load(first) as z:
        return int(z["layers"].shape[1]) if "layers" in z else 0


def run_config(emb_dir: Path, model: str, view: str, prep: str, queries, artists, device: str,
               corpus: set[int] | None, xartist: bool, require_complete: bool = False) -> dict:
    ids, x, mode = build_view(emb_dir, model, view, device)
    if corpus is not None:
        missing = corpus - set(map(int, ids))
        if require_complete and missing:
            raise ValueError(f"{model}: missing {len(missing)} required corpus embeddings")
        keep = np.array([int(t) in corpus for t in ids])
        ids, x = ids[keep], x[torch.from_numpy(keep).to(device)]
    f = prep_features(x, prep)
    pen = hub_penalty(f, mode=mode) if prep.endswith("csls") else None
    return {"model": model, "view": view, "prep": prep, "xartist": xartist, "corpus": len(ids),
            "results": evaluate(f, ids, queries, artists, mode=mode, penalty=pen, xartist=xartist)}


HEADLINE = {"genalpha": ("ap", "p@10"), "mb_covers": ("ap", "hr@10"), "mb_samples": ("rr", "hr@100"),
            "mb_samples_rev": ("rr", "hr@100"), "v4": ("rr", "hr@100"), "sampleid": ("rr", "hr@100")}


def fmt_row(r: dict, split_filter: tuple[str, ...]) -> str:
    cells = []
    for key, m in r["results"].items():
        bench, split = key.split("/")
        if split not in split_filter:
            continue
        a, b = HEADLINE.get(bench, ("ap", "hr@10"))
        label = f"{bench[:10]}/{split[:4]}({m['n']}/{m.get('n_expected', m['n'])})"
        cells.append(f"{label} {a}={m[a]:.3f} {b}={m[b]:.3f}" if m["n"] else f"{label} unavailable")
    return f"{r['model']:<10} {r['view']:<12} {r['prep']:<12} {'X' if r['xartist'] else ' '} | " + " | ".join(cells)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="evaluate")
    p.add_argument("--emb", required=True)
    p.add_argument("--bench", required=True)
    p.add_argument("--catalog", default=None, help="catalog_meta.npz for artist names of distractors")
    p.add_argument("--model", nargs="+", required=True)
    p.add_argument("--view", nargs="+", default=["auto"],
                   help="track, win:maxsim, win:chamfer, L<k>, L<a>-<b>, LW<k>:maxsim, alllayers, auto")
    p.add_argument("--prep", nargs="+", default=["center"])
    p.add_argument("--corpus", nargs="*", default=None, help="restrict corpus to ids in these files (+benchmarks)")
    p.add_argument("--require-complete", action="store_true", help="fail if any requested corpus ID lacks embeddings")
    p.add_argument("--xartist", action="store_true", help="also evaluate with same-artist candidates removed")
    p.add_argument("--splits", nargs="+", default=["development", "test", "held_out"])
    p.add_argument("--out", default=None, help="append JSON lines here")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)

    emb_dir, bench_dir = Path(args.emb), Path(args.bench)
    queries, artists = load_benchmarks(bench_dir)
    if args.catalog:
        load_catalog_artists(Path(args.catalog), artists)
    corpus = None
    if args.corpus:
        corpus = {q.q for q in queries} | {p_ for q in queries for p_ in q.pos}
        for f in args.corpus:
            corpus |= {int(x) for x in Path(f).read_text().split()}
    if args.require_complete and corpus is None:
        p.error("--require-complete needs --corpus")
    queries = [q for q in queries if q.split in args.splits]
    counts = defaultdict(int)
    for q in queries:
        counts[f"{q.bench}/{q.split}"] += 1
    print("queries:", dict(sorted(counts.items())), flush=True)

    out_fh = open(args.out, "a", encoding="utf-8") if args.out else None
    for model in args.model:
        views = args.view
        if views == ["auto"]:
            nl = n_layers(emb_dir, model)
            views = ["track", "win:maxsim"] if nl == 0 else [f"L{i}" for i in range(nl)]
        elif views == ["alllayers"]:
            views = [f"L{i}" for i in range(n_layers(emb_dir, model))]
        for view in views:
            for prep in args.prep:
                for xa in ([False, True] if args.xartist else [False]):
                    t0 = time.time()
                    try:
                        r = run_config(emb_dir, model, view, prep, queries, artists, args.device, corpus, xa,
                                       require_complete=args.require_complete)
                    except KeyError as exc:
                        print(f"skip {model} {view}: missing {exc}")
                        continue
                    r["secs"] = round(time.time() - t0, 1)
                    print(fmt_row(r, tuple(args.splits)), flush=True)
                    if out_fh:
                        out_fh.write(json.dumps(r) + "\n")
                        out_fh.flush()
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.exit(main())
