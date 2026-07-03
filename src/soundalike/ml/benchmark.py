"""Measure recommendation quality — and the library-size vs accuracy trade-off.

Everything so far has been eyeballed ("these look right"). This module puts
numbers on it, using two label-free metrics computed straight from the bundled
deep-vibe index (no downloads, no human labels):

* **same-artist recall@K** (a *precision* proxy) — for a song whose artist has
  another track in the library, how often is that same-artist track in the
  top-K neighbours? A good recommender ranks a song's obvious sonic siblings
  near the top; as the library grows, more distractors compete, so this falls.
  It measures "does the ranking stay sharp?".

* **nearest-neighbour cosine on a held-out probe set** (a *coverage* proxy) —
  for songs held out of the library, how close is their nearest neighbour? A
  bigger, broader library means any given song has a closer match, so this
  rises with size. It measures "is there something close to recommend?".

Precision falls and coverage rises with library size, so the "perfect balance"
is the knee: the smallest library that already captures most of the coverage,
beyond which you mostly pile on distractors that erode precision.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _fit_whiten(neural: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mean, W) for ZCA whitening, matching DeepVibeRecommender."""
    mean = neural.mean(axis=0)
    centered = neural - mean
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    W = evecs @ np.diag(1.0 / np.sqrt(np.clip(evals, 1e-5, None))) @ evecs.T
    return mean, W


def _whiten(vecs: np.ndarray, mean: np.ndarray, W: np.ndarray) -> np.ndarray:
    x = (vecs - mean) @ W
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def same_artist_recall(
    neural_w: np.ndarray,
    artists: np.ndarray,
    k: int = 10,
    n_queries: int = 1000,
    seed: int = 0,
) -> Dict[str, float]:
    """Fraction of query songs with a same-artist track in their top-K neighbours.

    `neural_w` must already be whitened + L2-normalized (so a dot is cosine).
    Only songs whose artist has >=2 tracks in the library are eligible queries.
    """
    artists = np.asarray([str(a).casefold() for a in artists])
    # Eligible = artists appearing >= 2 times.
    uniq, counts = np.unique(artists, return_counts=True)
    multi = set(uniq[counts >= 2].tolist())
    eligible = np.array([i for i, a in enumerate(artists) if a in multi])
    if len(eligible) == 0:
        return {"recall_at_k": 0.0, "mrr": 0.0, "n": 0}

    rng = np.random.default_rng(seed)
    q_idx = rng.choice(eligible, size=min(n_queries, len(eligible)), replace=False)

    hits, rr = 0, 0.0
    lib = neural_w
    for qi in q_idx:
        sims = lib @ lib[qi]
        sims[qi] = -np.inf  # exclude self
        top = np.argpartition(-sims, k)[:k]
        top = top[np.argsort(-sims[top])]
        same = artists[top] == artists[qi]
        if same.any():
            hits += 1
            rr += 1.0 / (int(np.argmax(same)) + 1)
    n = len(q_idx)
    return {"recall_at_k": hits / n, "mrr": rr / n, "n": n}


def coverage_score(probe_w: np.ndarray, lib_w: np.ndarray) -> float:
    """Mean nearest-neighbour cosine of each held-out probe against the library."""
    best = np.full(len(probe_w), -np.inf, dtype=np.float32)
    step = 4096
    for s in range(0, len(lib_w), step):
        chunk = lib_w[s : s + step]
        sims = probe_w @ chunk.T  # (P, chunk)
        best = np.maximum(best, sims.max(axis=1))
    return float(best.mean())


def same_artist_map(
    neural_w: np.ndarray,
    artists: np.ndarray,
    n_queries: int = 2000,
    max_rank: int = 100,
    seed: int = 0,
) -> float:
    """Mean average precision for same-artist retrieval — the headline metric.

    For each query (a song whose artist has >=2 tracks), rank all other songs by
    cosine and treat same-artist songs as the relevant set. Average precision
    rewards putting *all* of a song's siblings high, not just one — so it's far
    more sensitive to ranking quality than recall@K. `neural_w` must be whitened
    + L2-normalized. Averaged over a random query sample.
    """
    art = np.asarray([str(a).casefold() for a in artists])
    uniq, counts = np.unique(art, return_counts=True)
    multi = set(uniq[counts >= 2].tolist())
    eligible = np.array([i for i, a in enumerate(art) if a in multi])
    if len(eligible) == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    q_idx = rng.choice(eligible, size=min(n_queries, len(eligible)), replace=False)

    aps = []
    for qi in q_idx:
        sims = neural_w @ neural_w[qi]
        sims[qi] = -np.inf
        rel_total = int((art == art[qi]).sum()) - 1  # exclude self
        if rel_total <= 0:
            continue
        # Only need the top max_rank for a stable AP estimate on huge libraries.
        top = np.argpartition(-sims, max_rank)[:max_rank]
        top = top[np.argsort(-sims[top])]
        rel = (art[top] == art[qi]).astype(np.float32)
        if rel.sum() == 0:
            aps.append(0.0)
            continue
        cum = np.cumsum(rel)
        ranks = np.arange(1, len(rel) + 1)
        precision_at_hits = (cum / ranks) * rel
        aps.append(float(precision_at_hits.sum() / min(rel_total, max_rank)))
    return float(np.mean(aps)) if aps else 0.0


def score_embeddings(
    neural: np.ndarray,
    artists: np.ndarray,
    whiten: bool = True,
    k: int = 10,
    n_queries: int = 2000,
    n_probe: int = 1500,
    seed: int = 0,
) -> Dict[str, float]:
    """One-call quality report for any embedding matrix (the experiment scorer).

    Whitens (as production does), then reports same-artist mAP + recall@K
    (precision-side) and held-out nearest-neighbour cosine (coverage-side).
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(neural))
    probe_idx = perm[:n_probe]
    lib_idx = perm[n_probe:]

    if whiten:
        mean, W = _fit_whiten(neural[lib_idx])
        lib_w = _whiten(neural[lib_idx], mean, W)
        probe_w = _whiten(neural[probe_idx], mean, W)
    else:
        lib_w = neural[lib_idx] / (np.linalg.norm(neural[lib_idx], axis=1, keepdims=True) + 1e-9)
        probe_w = neural[probe_idx] / (np.linalg.norm(neural[probe_idx], axis=1, keepdims=True) + 1e-9)

    lib_artists = np.asarray(artists)[lib_idx]
    mapv = same_artist_map(lib_w, lib_artists, n_queries=n_queries, seed=seed)
    rec = same_artist_recall(lib_w, lib_artists, k=k, n_queries=n_queries, seed=seed)
    cov = coverage_score(probe_w, lib_w)
    return {"map": mapv, "recall_at_k": rec["recall_at_k"], "mrr": rec["mrr"],
            "coverage": cov, "dim": int(neural.shape[1]), "n_lib": int(len(lib_idx))}


def fixed_pair_precision(
    lib_w: np.ndarray,
    active: np.ndarray,
    pairs: List[Tuple[int, int]],
    k: int = 10,
) -> float:
    """Recall@K over FIXED (query, target) same-artist pairs.

    `lib_w` is the whitened full library; `active` is a boolean mask of which
    rows are in the current (nested) subsample. Each pair's query and target are
    guaranteed present in every subsample, so only the number of *distractors*
    changes with size — isolating whether a bigger pool pushes a song's true
    sibling out of the top-K.
    """
    idx_active = np.where(active)[0]
    sub = lib_w[idx_active]
    pos = {int(g): i for i, g in enumerate(idx_active)}  # global -> local row
    hits = 0
    for q, t in pairs:
        sims = sub @ lib_w[q]
        sims[pos[q]] = -np.inf  # exclude self
        top = np.argpartition(-sims, k)[:k]
        if pos[t] in set(top.tolist()):
            hits += 1
    return hits / len(pairs) if pairs else 0.0


def library_size_sweep(
    neural: np.ndarray,
    artists: np.ndarray,
    sizes: List[int],
    k: int = 10,
    n_pairs: int = 1500,
    n_probe: int = 800,
    seed: int = 0,
) -> List[Dict[str, float]]:
    """Nested-subsample sweep: fixed-pair precision + held-out coverage per size.

    Subsamples are nested (each size a superset of the smaller), so a query and
    its same-artist target present at the smallest size stay present throughout;
    growing the library only adds distractors. Whitening is re-fit per size to
    match production.
    """
    rng = np.random.default_rng(seed)
    n_total = len(neural)
    sizes = sorted(min(s, n_total - n_probe) for s in sizes)

    perm = rng.permutation(n_total)
    probe_idx = perm[:n_probe]
    pool = perm[n_probe:]  # ordered; nested prefixes are the subsamples

    art = np.array([str(a).casefold() for a in artists])

    # Fixed same-artist pairs drawn from the SMALLEST subsample (so both members
    # are in every subsample). One target per query to avoid target-count bias.
    core = pool[: sizes[0]]
    core_by_artist: Dict[str, List[int]] = {}
    for g in core:
        core_by_artist.setdefault(art[g], []).append(int(g))
    pairs: List[Tuple[int, int]] = []
    for a, members in core_by_artist.items():
        if len(members) >= 2:
            pairs.append((members[0], members[1]))
    rng.shuffle(pairs)
    pairs = pairs[:n_pairs]

    rows = []
    for size in sizes:
        sub = pool[:size]
        active = np.zeros(n_total, dtype=bool)
        active[sub] = True

        mean, W = _fit_whiten(neural[sub])
        lib_w_full = _whiten(neural, mean, W)  # whiten all (queries/targets live here)
        probe_w = _whiten(neural[probe_idx], mean, W)

        prec = fixed_pair_precision(lib_w_full, active, pairs, k=k)
        cov = coverage_score(probe_w, lib_w_full[sub])
        rows.append({
            "size": int(size),
            "recall_at_k": prec,
            "coverage": cov,
            "n_pairs": len(pairs),
        })
    return rows


def find_sweet_spot(rows: List[Dict[str, float]], coverage_frac: float = 0.95) -> int:
    """Smallest library size whose coverage reaches `coverage_frac` of the way
    up the swept coverage *range* (min-max normalized).

    Beyond this point coverage has largely saturated, so additional songs mostly
    add distractors that erode precision — the practical "balance" point.
    """
    cov = np.array([r["coverage"] for r in rows])
    sizes = np.array([r["size"] for r in rows])
    cov_norm = (cov - cov.min()) / (cov.max() - cov.min() + 1e-9)
    hit = np.where(cov_norm >= coverage_frac)[0]
    return int(sizes[hit[0]]) if len(hit) else int(sizes[-1])


def balance_point(rows: List[Dict[str, float]]) -> int:
    """Library size that best balances precision and coverage.

    Min-max normalizes both curves over the swept range and returns the size
    that maximizes their harmonic mean (F1) — the point where neither precision
    nor coverage is being sacrificed hard for the other.
    """
    prec = np.array([r["recall_at_k"] for r in rows])
    cov = np.array([r["coverage"] for r in rows])
    sizes = np.array([r["size"] for r in rows])
    pn = (prec - prec.min()) / (prec.max() - prec.min() + 1e-9)
    cn = (cov - cov.min()) / (cov.max() - cov.min() + 1e-9)
    f1 = 2 * pn * cn / (pn + cn + 1e-9)
    return int(sizes[int(np.argmax(f1))])


def main(argv: Optional[list] = None) -> int:
    import argparse
    import json

    from .deepvibe import DeepVibeIndex

    parser = argparse.ArgumentParser(description="Benchmark deep-vibe recommendation quality.")
    parser.add_argument("--index", default=None, help="Deep-vibe index (.npz). Default: bundled.")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--sizes", default="5000,10000,20000,40000,60000,87000")
    parser.add_argument("--figure", default="docs/library_size_sweep.png")
    args = parser.parse_args(argv)

    index_path = Path(args.index) if args.index else DeepVibeIndex.default_path()
    index = DeepVibeIndex.load(index_path)
    print(f"Loaded index: {len(index)} tracks, neural dim {index.neural.shape[1]}")

    sizes = [int(s) for s in args.sizes.split(",")]
    rows = library_size_sweep(index.neural, index.artists, sizes, k=args.k)

    print(f"\n{'size':>8} {'recall@'+str(args.k):>11} {'coverage':>10}   (fixed same-artist pairs)")
    for r in rows:
        print(f"{r['size']:>8} {r['recall_at_k']:>11.3f} {r['coverage']:>10.3f}")
    sweet = find_sweet_spot(rows)
    bal = balance_point(rows)
    print(f"\nCoverage-saturation (95%) library size: ~{sweet:,} tracks")
    print(f"Precision/coverage balance point (F1):  ~{bal:,} tracks")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        s = [r["size"] for r in rows]
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax2 = ax1.twinx()
        l1 = ax1.plot(s, [r["recall_at_k"] for r in rows], "o-", color="#1f77b4",
                      label=f"fixed-pair recall@{args.k} (precision)")
        l2 = ax2.plot(s, [r["coverage"] for r in rows], "s--", color="#d62728",
                      label="held-out NN cosine (coverage)")
        ax1.axvline(sweet, color="gray", ls=":", alpha=0.7)
        ax1.set_xlabel("library size (tracks)")
        ax1.set_ylabel(f"fixed-pair recall@{args.k}", color="#1f77b4")
        ax2.set_ylabel("held-out NN cosine", color="#d62728")
        ax1.tick_params(axis="y", labelcolor="#1f77b4")
        ax2.tick_params(axis="y", labelcolor="#d62728")
        ax1.set_title("Library size vs recommendation quality\n"
                      "(distractors held fixed; does a bigger pool bury true siblings?)")
        lns = l1 + l2
        ax1.legend(lns, [ln.get_label() for ln in lns], loc="center right")
        fig.tight_layout()
        Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.figure, dpi=130)
        print(f"Saved figure -> {args.figure}")
    except ImportError:
        pass

    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
