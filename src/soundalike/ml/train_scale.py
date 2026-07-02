"""Large-scale contrastive training (built for FMA-medium on the RTX 5080).

Differences from the small `train.py`:
  * ResNet-style encoder (bigger capacity for 25k tracks),
  * a held-out test set (stratified by genre) for honest evaluation,
  * large batch + cosine LR schedule with warmup,
  * periodic kNN genre-probe + silhouette during training so you can watch the
    representation actually improve,
  * AMP + channels-last so the 5080's Tensor Cores do the work.

The objective is still self-supervised NT-Xent (two augmented time-crops of the
same track are positives); genre labels are used ONLY for evaluation.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from .collect import DatasetCollector, TrackEntry
from .data import SpectrogramDataset, _spec_path
from .evaluate import knn_genre_probe, nearest_neighbor_genre_match, silhouette
from .model import ProjectionHead, ResNetAudioEncoder, nt_xent_loss
from .spectrogram import _fit_frames


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _stratified_split(entries: List[TrackEntry], val_frac=0.1, test_frac=0.1, seed=0):
    rng = np.random.default_rng(seed)
    by_genre = {}
    for e in entries:
        by_genre.setdefault(e.genre, []).append(e)
    train, val, test = [], [], []
    for _genre, items in by_genre.items():
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        n_test = int(n * test_frac)
        n_val = int(n * val_frac)
        test += items[:n_test]
        val += items[n_test : n_test + n_val]
        train += items[n_test + n_val :]
    return train, val, test


def _embed_all(encoder, entries, spec_dir, device, mem, batch=256):
    import torch

    specs, labels, keep = [], [], []
    for e in entries:
        p = _spec_path(Path(spec_dir), e.track_id)
        if p.exists():
            specs.append(_fit_frames(np.load(p), 256))
            labels.append(e.genre)
            keep.append(e)
    embs = []
    encoder.eval()
    with torch.no_grad():
        for start in range(0, len(specs), batch):
            arr = np.stack(specs[start : start + batch])[:, None]
            x = torch.from_numpy(arr).to(device, memory_format=mem)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                embs.append(encoder(x).float().cpu().numpy())
    return np.concatenate(embs), np.array(labels), keep


def _worker_init(worker_id: int) -> None:
    """Give each DataLoader worker an independent augmentation RNG.

    On Windows (spawn) every worker inherits an identical copy of the dataset's
    RNG, which would correlate augmentations. Reseed per worker here.
    """
    import torch

    info = torch.utils.data.get_worker_info()
    if info is not None:
        info.dataset._rng = np.random.default_rng(info.seed % (2**32))


def train_scale(
    manifest: Path,
    spec_dir: Path,
    out_dir: Path,
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 3e-3,
    width: int = 64,
    embedding_dim: int = 256,
    num_workers: int = 8,
    eval_every: int = 5,
    progress=print,
) -> dict:
    import torch
    from torch.utils.data import DataLoader

    entries = [
        e for e in DatasetCollector.load(manifest)
        if _spec_path(Path(spec_dir), e.track_id).exists()
    ]
    progress(f"Usable tracks with spectrograms: {len(entries)}")
    train_e, val_e, test_e = _stratified_split(entries)
    progress(f"Split -> train {len(train_e)}  val {len(val_e)}  test {len(test_e)}")

    train_ds = SpectrogramDataset(train_e, spec_dir)
    loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,
        worker_init_fn=_worker_init if num_workers > 0 else None,
    )

    device = _device()
    mem = torch.channels_last if device == "cuda" else torch.contiguous_format
    encoder = ResNetAudioEncoder(embedding_dim=embedding_dim, width=width).to(
        device, memory_format=mem
    )
    head = ProjectionHead(in_dim=embedding_dim, hidden=512, out_dim=128).to(device)
    params = list(encoder.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)

    steps_per_epoch = max(1, len(loader))
    total_steps = epochs * steps_per_epoch
    warmup = 5 * steps_per_epoch

    def lr_at(step):
        if step < warmup:
            return step / max(1, warmup)
        p = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))

    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_knn = 0.0
    step = 0
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        encoder.train(); head.train()
        epoch_loss = 0.0
        t0 = time.time()
        for v1, v2, _ in loader:
            for g in opt.param_groups:
                g["lr"] = lr * lr_at(step)
            v1 = v1.to(device, memory_format=mem, non_blocking=True)
            v2 = v2.to(device, memory_format=mem, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                z1 = head(encoder(v1, normalize=False))
                z2 = head(encoder(v2, normalize=False))
                loss = nt_xent_loss(z1, z2)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            epoch_loss += float(loss.item())
            step += 1

        avg = epoch_loss / steps_per_epoch
        msg = f"epoch {epoch:3d}/{epochs}  loss {avg:.4f}  ({time.time()-t0:.1f}s)"
        if epoch % eval_every == 0 or epoch == 1 or epoch == epochs:
            val_emb, val_lab, _ = _embed_all(encoder, val_e, spec_dir, device, mem)
            probe = knn_genre_probe(val_emb, val_lab, k=10)
            sil = silhouette(val_emb, val_lab)
            msg += f"  | val kNN {probe['knn_accuracy']:.3f}  silhouette {sil:+.3f}"
            history.append({"epoch": epoch, "loss": avg, **probe, "silhouette": sil})
            if probe["knn_accuracy"] > best_knn:
                best_knn = probe["knn_accuracy"]
                torch.save(
                    {"state_dict": encoder.state_dict(), "embedding_dim": embedding_dim,
                     "width": width, "arch": "resnet"},
                    out_dir / "encoder_best.pt",
                )
        progress(msg)

    # Final: embed EVERYTHING and evaluate on the held-out test set.
    all_emb, all_lab, kept = _embed_all(encoder, entries, spec_dir, device, mem)
    np.savez(
        out_dir / "embeddings.npz",
        embeddings=all_emb, labels=all_lab,
        titles=np.array([e.title for e in kept]),
        artists=np.array([e.artist for e in kept]),
        track_ids=np.array([e.track_id for e in kept]),
    )
    test_emb, test_lab, _ = _embed_all(encoder, test_e, spec_dir, device, mem)
    final = knn_genre_probe(test_emb, test_lab, k=10)
    final["silhouette"] = silhouette(test_emb, test_lab)
    final["nn_genre_match"] = nearest_neighbor_genre_match(test_emb, test_lab)
    final["best_val_knn"] = best_knn
    final["minutes"] = round((time.time() - t_start) / 60, 1)

    torch.save(
        {"state_dict": encoder.state_dict(), "embedding_dim": embedding_dim,
         "width": width, "arch": "resnet"},
        out_dir / "encoder.pt",
    )
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "test_report.json").write_text(json.dumps(final, indent=2))
    progress("\n=== FINAL TEST-SET REPORT ===")
    progress(json.dumps(final, indent=2))
    return final


def main(argv: Optional[list] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Large-scale contrastive training.")
    parser.add_argument("--manifest", default="Z:/fma/fma_manifest.csv")
    parser.add_argument("--spec-dir", default="Z:/fma/specs")
    parser.add_argument("--out-dir", default="ml_data/model_fma")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    print("Device:", _device())
    train_scale(
        Path(args.manifest), Path(args.spec_dir), Path(args.out_dir),
        epochs=args.epochs, batch_size=args.batch_size, width=args.width,
        num_workers=args.workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
