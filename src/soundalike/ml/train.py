"""Train the audio embedding model on the GPU (self-supervised contrastive).

End to end:
  1. load the manifest,
  2. cache mel-spectrograms for every preview (idempotent),
  3. train the encoder with NT-Xent using two augmented views per clip,
  4. save the encoder checkpoint and the learned embedding of every track.

Optimized for the RTX 5080: channels-last memory format + automatic mixed
precision so the convolutions run on Tensor Cores (see soundalike.ml.gpu for the
kernel-selection story).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from .collect import DatasetCollector
from .data import SpectrogramDataset, all_specs_matrix, precompute_spectrograms
from .model import AudioEncoder, ProjectionHead, nt_xent_loss


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def train(
    manifest: Path,
    spec_dir: Path,
    out_dir: Path,
    epochs: int = 40,
    batch_size: int = 128,
    lr: float = 1e-3,
    embedding_dim: int = 128,
    progress=print,
) -> Path:
    import torch
    from torch.utils.data import DataLoader

    entries = DatasetCollector.load(manifest)
    progress(f"Manifest: {len(entries)} tracks. Preparing spectrograms...")
    precompute_spectrograms(entries, spec_dir, progress=progress)

    dataset = SpectrogramDataset(entries, spec_dir)
    progress(f"Trainable spectrograms: {len(dataset)}")
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0
    )

    device = _device()
    memory_format = torch.channels_last if device == "cuda" else torch.contiguous_format
    encoder = AudioEncoder(embedding_dim=embedding_dim).to(device, memory_format=memory_format)
    head = ProjectionHead(in_dim=embedding_dim).to(device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()), lr=lr, weight_decay=1e-4
    )
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    encoder.train()
    head.train()
    for epoch in range(1, epochs + 1):
        epoch_loss, batches = 0.0, 0
        t0 = time.time()
        for v1, v2, _ in loader:
            v1 = v1.to(device, memory_format=memory_format, non_blocking=True)
            v2 = v2.to(device, memory_format=memory_format, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                z1 = head(encoder(v1, normalize=False))
                z2 = head(encoder(v2, normalize=False))
                loss = nt_xent_loss(z1, z2)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += float(loss.item())
            batches += 1
        if epoch % 5 == 0 or epoch == 1:
            progress(
                f"epoch {epoch:3d}/{epochs}  loss {epoch_loss / max(batches,1):.4f}"
                f"  ({time.time() - t0:.1f}s)"
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "encoder.pt"
    torch.save({"state_dict": encoder.state_dict(), "embedding_dim": embedding_dim}, ckpt)

    # Compute and store the embedding of every track (un-augmented).
    encoder.eval()
    specs = all_specs_matrix(dataset).to(device, memory_format=memory_format)
    with torch.no_grad(), torch.amp.autocast("cuda", enabled=use_amp):
        embeddings = encoder(specs).float().cpu().numpy()
    np.savez(
        out_dir / "embeddings.npz",
        embeddings=embeddings,
        labels=np.array(dataset.labels),
        track_ids=np.array([e.track_id for e in dataset.entries]),
        titles=np.array([e.title for e in dataset.entries]),
        artists=np.array([e.artist for e in dataset.entries]),
    )
    progress(f"Saved encoder -> {ckpt}")
    progress(f"Saved embeddings -> {out_dir / 'embeddings.npz'}")
    return ckpt


def main(argv: Optional[list] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Train the audio embedding model.")
    parser.add_argument("--manifest", default="ml_data/manifest.csv")
    parser.add_argument("--spec-dir", default="ml_data/specs")
    parser.add_argument("--out-dir", default="ml_data/model")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args(argv)

    print("Device:", _device())
    train(
        Path(args.manifest),
        Path(args.spec_dir),
        Path(args.out_dir),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
