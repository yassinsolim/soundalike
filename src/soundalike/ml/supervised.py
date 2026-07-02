"""Supervised genre-embedding model.

We proved genres are separable from audio, but the self-supervised model needs
far more data than we have. A supervised classifier is the pragmatic bridge: it
trains the SAME CNN encoder to predict genre, and we take the penultimate layer
as the embedding. Because the objective directly rewards genre structure, the
embedding space clusters cleanly even on a small dataset — a useful, honest
counterpoint to the self-supervised result.

Evaluation uses a train/validation split so the reported accuracy reflects
generalization, not memorization.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from .collect import DatasetCollector
from .data import precompute_spectrograms
from .model import AudioEncoder


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class _LabelledSpecs:
    """Center-cropped spectrograms + integer genre labels for supervised training."""

    def __init__(self, entries, spec_dir: Path, target_frames: int = 256):
        from .data import _spec_path
        from .spectrogram import _fit_frames

        self.items = []
        self.genres = sorted({e.genre for e in entries})
        genre_to_idx = {g: i for i, g in enumerate(self.genres)}
        for e in entries:
            p = _spec_path(Path(spec_dir), e.track_id)
            if p.exists():
                self.items.append((p, genre_to_idx[e.genre], _fit_frames))
        self.target_frames = target_frames

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import torch

        from .spectrogram import augment, random_crop

        path, label, _fit = self.items[i]
        spec = np.load(path)
        # Light augmentation (random crop + masking) as regularization.
        view = augment(random_crop(spec, self.target_frames))
        return torch.from_numpy(view).unsqueeze(0), label, i


class GenreClassifier:
    """CNN encoder + linear genre head."""

    def __init__(self, embedding_dim: int = 128, n_classes: int = 8):
        import torch.nn as nn

        self.encoder = AudioEncoder(embedding_dim=embedding_dim)
        self.head = nn.Linear(embedding_dim, n_classes)


def train_supervised(
    manifest: Path,
    spec_dir: Path,
    out_dir: Path,
    epochs: int = 60,
    batch_size: int = 64,
    val_frac: float = 0.2,
    progress=print,
) -> float:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Subset

    entries = DatasetCollector.load(manifest)
    precompute_spectrograms(entries, spec_dir, progress=progress)
    ds = _LabelledSpecs(entries, spec_dir)
    progress(f"Labelled spectrograms: {len(ds)} across {len(ds.genres)} genres")

    rng = np.random.default_rng(0)
    idx = rng.permutation(len(ds))
    n_val = int(len(ds) * val_frac)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    train_loader = DataLoader(Subset(ds, train_idx.tolist()), batch_size=batch_size,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(Subset(ds, val_idx.tolist()), batch_size=batch_size)

    device = _device()
    mem = torch.channels_last if device == "cuda" else torch.contiguous_format
    model = GenreClassifier(n_classes=len(ds.genres))
    encoder = model.encoder.to(device, memory_format=mem)
    head = model.head.to(device)
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(head.parameters()),
                            lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        encoder.train(); head.train()
        for x, y, _ in train_loader:
            x = x.to(device, memory_format=mem, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                logits = head(encoder(x, normalize=False))
                loss = criterion(logits, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()

        # Validation accuracy.
        encoder.eval(); head.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y, _ in val_loader:
                x = x.to(device, memory_format=mem)
                logits = head(encoder(x.to(device), normalize=False))
                correct += int((logits.argmax(1).cpu() == y).sum())
                total += len(y)
        acc = correct / max(total, 1)
        best_acc = max(best_acc, acc)
        if epoch % 10 == 0 or epoch == 1:
            progress(f"epoch {epoch:3d}/{epochs}  val_acc {acc:.3f}  (best {best_acc:.3f})")

    # Save embeddings of all tracks for visualization.
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder.eval()
    from .spectrogram import _fit_frames

    specs, labels, titles, artists, ids = [], [], [], [], []
    from .data import _spec_path
    for e in entries:
        p = _spec_path(Path(spec_dir), e.track_id)
        if not p.exists():
            continue
        specs.append(_fit_frames(np.load(p), 256))
        labels.append(e.genre); titles.append(e.title)
        artists.append(e.artist); ids.append(e.track_id)
    arr = torch.from_numpy(np.stack(specs)[:, None]).to(device, memory_format=mem)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=device == "cuda"):
        emb = encoder(arr).float().cpu().numpy()
    np.savez(out_dir / "embeddings.npz", embeddings=emb, labels=np.array(labels),
             titles=np.array(titles), artists=np.array(artists), track_ids=np.array(ids))
    torch.save({"state_dict": encoder.state_dict(), "embedding_dim": 128}, out_dir / "encoder.pt")
    progress(f"Best validation accuracy: {best_acc:.3f}")
    return best_acc


def main(argv: Optional[list] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Train a supervised genre-embedding model.")
    parser.add_argument("--manifest", default="ml_data/manifest.csv")
    parser.add_argument("--spec-dir", default="ml_data/specs")
    parser.add_argument("--out-dir", default="ml_data/model_supervised")
    parser.add_argument("--epochs", type=int, default=60)
    args = parser.parse_args(argv)
    print("Device:", _device())
    train_supervised(Path(args.manifest), Path(args.spec_dir), Path(args.out_dir),
                     epochs=args.epochs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
