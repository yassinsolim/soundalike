"""Publication-style results figure for a trained embedding model.

Combines, in one image:
  * the training curve (contrastive loss + kNN genre-probe accuracy per epoch),
  * a 2D UMAP/PCA map of the embedding space colored by genre,
  * a per-genre nearest-neighbor "hit rate" bar chart.

This turns the raw numbers into something you can look at and immediately judge
whether the model learned musical structure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np


def _project_2d(embeddings: np.ndarray, seed: int = 0, max_points: int = 6000):
    """Project to 2D (UMAP if available, else PCA). Subsamples for speed/clarity."""
    n = len(embeddings)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)[: min(n, max_points)]
    sub = embeddings[idx]
    try:
        import umap  # type: ignore

        coords = umap.UMAP(
            n_neighbors=20, min_dist=0.15, metric="cosine", random_state=seed
        ).fit_transform(sub)
    except Exception:  # noqa: BLE001
        from sklearn.decomposition import PCA

        coords = PCA(n_components=2, random_state=seed).fit_transform(sub)
    return coords, idx


def _per_genre_hit_rate(embeddings: np.ndarray, labels: np.ndarray) -> dict:
    from .evaluate import nearest_neighbor_genre_match  # reuse chunked NN

    x = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
    genres = sorted(set(labels.tolist()))
    hits = {g: [0, 0] for g in genres}
    chunk = 2048
    n = len(x)
    for start in range(0, n, chunk):
        sims = x[start : start + chunk] @ x.T
        for r, row in enumerate(sims):
            i = start + r
            row[i] = -np.inf
            j = int(np.argmax(row))
            g = labels[i]
            hits[g][1] += 1
            hits[g][0] += int(labels[i] == labels[j])
    return {g: (h[0] / h[1] if h[1] else 0.0) for g, h in hits.items()}


def results_figure(model_dir: Path, out_path: Optional[Path] = None, seed: int = 0) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_dir = Path(model_dir)
    out_path = out_path or (model_dir / "results.png")
    data = np.load(model_dir / "embeddings.npz", allow_pickle=True)
    embeddings, labels = data["embeddings"].astype(np.float32), data["labels"]

    history = []
    hist_path = model_dir / "history.json"
    if hist_path.exists():
        history = json.loads(hist_path.read_text())

    fig = plt.figure(figsize=(18, 6))

    # Panel 1: training curve.
    ax1 = fig.add_subplot(1, 3, 1)
    if history:
        ep = [h["epoch"] for h in history]
        ax1.plot(ep, [h["loss"] for h in history], "o-", color="tab:red", label="contrastive loss")
        ax1.set_xlabel("epoch"); ax1.set_ylabel("loss", color="tab:red")
        ax2 = ax1.twinx()
        ax2.plot(ep, [h["knn_accuracy"] for h in history], "s-", color="tab:blue",
                 label="kNN genre acc")
        ax2.set_ylabel("kNN accuracy", color="tab:blue"); ax2.set_ylim(0, 1)
    ax1.set_title("Training: loss down, genre-probe up")

    # Panel 2: embedding map.
    ax3 = fig.add_subplot(1, 3, 2)
    coords, idx = _project_2d(embeddings, seed)
    sub_labels = labels[idx]
    genres = sorted(set(sub_labels.tolist()))
    cmap = plt.get_cmap("tab20")
    for i, g in enumerate(genres):
        m = sub_labels == g
        ax3.scatter(coords[m, 0], coords[m, 1], s=6, alpha=0.6, color=cmap(i % 20), label=g)
    ax3.set_title(f"Embedding space ({len(idx)} of {len(embeddings)} tracks)")
    ax3.legend(loc="best", fontsize=7, markerscale=2, ncol=2)
    ax3.set_xticks([]); ax3.set_yticks([])

    # Panel 3: per-genre nearest-neighbor hit rate.
    ax4 = fig.add_subplot(1, 3, 3)
    rates = _per_genre_hit_rate(embeddings, labels)
    order = sorted(rates, key=rates.get, reverse=True)
    ax4.barh(order, [rates[g] for g in order], color="tab:green")
    ax4.set_xlim(0, 1); ax4.invert_yaxis()
    ax4.set_xlabel("nearest-neighbor same-genre rate")
    ax4.set_title("Per-genre retrieval quality")

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build a results figure.")
    parser.add_argument("--model-dir", default="ml_data/model_fma")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    out = results_figure(Path(args.model_dir), Path(args.out) if args.out else None)
    print(f"Saved results figure -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
