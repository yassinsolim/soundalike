"""Project learned embeddings to 2D and plot them, colored by genre.

This is the visual sanity check: if the self-supervised model learned anything
musically meaningful, songs of the same genre should form clusters even though
the model never saw genre labels during training. Genres are used only to color
the points.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def _project_2d(embeddings: np.ndarray, seed: int = 0) -> np.ndarray:
    """Reduce embeddings to 2D, preferring UMAP, falling back to PCA."""
    try:
        import umap  # type: ignore

        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=seed)
        return reducer.fit_transform(embeddings)
    except Exception:  # noqa: BLE001 - UMAP optional / may not be installed
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=seed).fit_transform(embeddings)


def plot_embeddings(npz_path: Path, out_path: Path, seed: int = 0) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = np.load(npz_path, allow_pickle=True)
    embeddings = data["embeddings"]
    labels = data["labels"]

    coords = _project_2d(embeddings, seed)
    genres = sorted(set(labels.tolist()))
    cmap = plt.get_cmap("tab10")

    plt.figure(figsize=(10, 8))
    for i, genre in enumerate(genres):
        mask = labels == genre
        plt.scatter(
            coords[mask, 0], coords[mask, 1],
            s=18, alpha=0.75, color=cmap(i % 10), label=genre,
        )
    plt.title("Learned audio embedding space (colored by genre)")
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=130)
    plt.close()
    return out_path


def genre_separation_score(npz_path: Path) -> float:
    """A simple label-agnostic quality metric: silhouette score of genres in
    embedding space. Higher (max 1.0) => genres are better separated.
    """
    from sklearn.metrics import silhouette_score

    data = np.load(npz_path, allow_pickle=True)
    embeddings = data["embeddings"]
    labels = data["labels"]
    if len(set(labels.tolist())) < 2:
        return float("nan")
    return float(silhouette_score(embeddings, labels, metric="cosine"))


def main(argv: Optional[list] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Visualize the embedding space.")
    parser.add_argument("--embeddings", default="ml_data/model/embeddings.npz")
    parser.add_argument("--out", default="ml_data/model/embedding_map.png")
    args = parser.parse_args(argv)

    npz = Path(args.embeddings)
    out = plot_embeddings(npz, Path(args.out))
    score = genre_separation_score(npz)
    print(f"Saved plot -> {out}")
    print(f"Genre separation (silhouette, cosine): {score:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
