"""Evaluate learned embeddings: kNN genre probe, silhouette, retrieval.

For self-supervised models the standard quality measure is a *probe*: freeze the
embeddings and see how well a simple classifier (here kNN) recovers genre from
them. If similar-sounding songs really are neighbors, kNN accuracy climbs well
above chance. We also report the silhouette score and a music-retrieval metric
(how often a track's nearest neighbor shares its genre).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np


def load_embeddings(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    return data["embeddings"].astype(np.float32), data["labels"]


def knn_genre_probe(
    embeddings: np.ndarray, labels: np.ndarray, k: int = 10, val_frac: float = 0.2, seed: int = 0
) -> Dict[str, float]:
    """Train a kNN on a train split, report accuracy on a held-out split."""
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.neighbors import KNeighborsClassifier

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(labels))
    n_val = int(len(labels) * val_frac)
    val, train = idx[:n_val], idx[n_val:]

    clf = KNeighborsClassifier(n_neighbors=k, metric="cosine")
    clf.fit(embeddings[train], labels[train])
    pred = clf.predict(embeddings[val])
    return {
        "knn_accuracy": float(accuracy_score(labels[val], pred)),
        "knn_macro_f1": float(f1_score(labels[val], pred, average="macro")),
        "k": k,
        "n_train": len(train),
        "n_val": len(val),
    }


def nearest_neighbor_genre_match(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of tracks whose single nearest neighbor shares their genre.

    A direct measure of "does the embedding put same-genre songs closest?".
    """
    # Normalize for cosine; compute similarities in chunks to bound memory.
    x = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)
    n = len(x)
    matches = 0
    chunk = 2048
    for start in range(0, n, chunk):
        sims = x[start : start + chunk] @ x.T
        for r, row in enumerate(sims):
            i = start + r
            row[i] = -np.inf  # exclude self
            j = int(np.argmax(row))
            matches += int(labels[i] == labels[j])
    return matches / max(n, 1)


def chance_accuracy(labels: np.ndarray) -> float:
    _, counts = np.unique(labels, return_counts=True)
    return float(counts.max() / counts.sum())


def silhouette(embeddings: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import silhouette_score

    if len(set(labels.tolist())) < 2:
        return float("nan")
    return float(silhouette_score(embeddings, labels, metric="cosine"))


def full_report(npz_path: Path, k: int = 10) -> Dict[str, float]:
    embeddings, labels = load_embeddings(npz_path)
    report = {
        "n_tracks": int(len(labels)),
        "n_genres": int(len(set(labels.tolist()))),
        "chance_accuracy": chance_accuracy(labels),
        "silhouette": silhouette(embeddings, labels),
        "nn_genre_match": nearest_neighbor_genre_match(embeddings, labels),
    }
    report.update(knn_genre_probe(embeddings, labels, k=k))
    return report


def main(argv=None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Evaluate learned embeddings.")
    parser.add_argument("--embeddings", default="ml_data/model/embeddings.npz")
    parser.add_argument("-k", type=int, default=10)
    args = parser.parse_args(argv)

    report = full_report(Path(args.embeddings), k=args.k)
    print(json.dumps(report, indent=2))
    print("\nInterpretation:")
    lift = report["knn_accuracy"] - report["chance_accuracy"]
    print(f"  kNN accuracy {report['knn_accuracy']:.3f} vs chance {report['chance_accuracy']:.3f}"
          f"  (+{lift:.3f} lift)")
    print(f"  {report['nn_genre_match']:.1%} of tracks' nearest neighbor share their genre.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
