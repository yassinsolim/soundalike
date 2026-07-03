"""Train a vibe-aware encoder (contrastive + vibe-target regression).

This is the dedicated "teach the model the vibe" training. On top of the
self-supervised contrastive objective (two crops of a song are positives), the
encoder must also predict the song's **vibe target** — its frequency-band
balance and dynamics (see vibe_target.py) — from each crop. That auxiliary task
forces the embedding to encode *how the song sounds and moves* (bass, drops),
not just its genre/timbre.

Trains on the packed FMA mel-spectrograms with no re-downloading. Reuses the
GPU/CPU-resident data handling and augmentation from train_fast.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .evaluate import knn_genre_probe, nearest_neighbor_genre_match, silhouette
from .model import ProjectionHead, ResNetAudioEncoder, nt_xent_loss
from .train_fast import _embed, _gpu_augment, _stratified_idx
from .vibe_target import VIBE_TARGET_DIM, vibe_targets_for_batch


def train_vibe(
    packed: Path,
    out_dir: Path,
    epochs: int = 45,
    batch_size: int = 512,
    lr: float = 3e-3,
    width: int = 64,
    embedding_dim: int = 256,
    crop: int = 256,
    vibe_lambda: float = 1.0,
    eval_every: int = 5,
    seed: int = 0,
    progress=print,
) -> dict:
    import torch
    import torch.nn as nn

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mem = torch.channels_last if device == "cuda" else torch.contiguous_format

    data = np.load(packed, allow_pickle=True)
    X = data["X"]
    genres = data["genres"]
    titles, artists, track_ids = data["titles"], data["artists"], data["track_ids"]
    progress(f"Loaded packed dataset {X.shape} ({X.nbytes/1e9:.1f} GB)")

    # Vibe targets from the FULL spectrogram (captures the whole song's dynamics).
    # Pass X as-is: vibe_target_from_mel casts each (n_mels, frames) slice to
    # float32 on the fly, so we avoid a transient float32 copy of the whole
    # (multi-GB) packed dataset that would otherwise double peak host RAM.
    progress("Computing vibe targets from mel-spectrograms...")
    t_targets = time.time()
    targets = vibe_targets_for_batch(X)
    # Standardize targets so the regression loss is well-scaled.
    tmean, tstd = targets.mean(0), targets.std(0) + 1e-6
    targets = ((targets - tmean) / tstd).astype(np.float32)
    progress(f"  vibe targets {targets.shape} in {time.time()-t_targets:.0f}s")

    train_idx, val_idx, test_idx = _stratified_idx(genres, seed=seed)
    progress(f"Split -> train {len(train_idx)}  val {len(val_idx)}  test {len(test_idx)}")

    # Data residency (GPU if it fits, else pinned CPU RAM), same policy as train_fast.
    dataset_gb = X.nbytes / 1e9
    gpu_resident = False
    if device == "cuda":
        free_b, total_b = torch.cuda.mem_get_info()
        gpu_resident = dataset_gb < (total_b / 1e9) - 7.0
    X_t = torch.from_numpy(X)
    tgt_t = torch.from_numpy(targets)
    if gpu_resident:
        X_res = X_t.to(device); tgt_res = tgt_t.to(device)
        progress(f"Dataset GPU-resident: {dataset_gb:.1f} GB")
        def gather(sel):
            s = torch.as_tensor(sel, device=device)
            return X_res[s], tgt_res[s]
    else:
        X_res = X_t.pin_memory() if device == "cuda" else X_t
        tgt_res = tgt_t.pin_memory() if device == "cuda" else tgt_t
        progress(f"Dataset CPU-resident ({dataset_gb:.1f} GB); streaming batches")
        def gather(sel):
            s = torch.as_tensor(sel).cpu()
            return X_res[s].to(device, non_blocking=True), tgt_res[s].to(device, non_blocking=True)

    encoder = ResNetAudioEncoder(embedding_dim=embedding_dim, width=width).to(
        device, memory_format=mem
    )
    head = ProjectionHead(in_dim=embedding_dim, hidden=512, out_dim=128).to(device)
    vibe_head = nn.Sequential(
        nn.Linear(embedding_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, VIBE_TARGET_DIM)
    ).to(device)
    params = list(encoder.parameters()) + list(head.parameters()) + list(vibe_head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    steps_per_epoch = max(1, len(train_idx) // batch_size)
    total_steps = epochs * steps_per_epoch
    warmup = 5 * steps_per_epoch
    rng = np.random.default_rng(seed)

    def lr_at(step):
        if step < warmup:
            return step / max(1, warmup)
        p = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))

    out_dir.mkdir(parents=True, exist_ok=True)
    history, best_knn, step = [], 0.0, 0
    t_start = time.time()
    train_base = np.asarray(train_idx)

    def embed_fn(idx):
        # For evaluation, embed by gathering specs only.
        return _embed(encoder, lambda s: gather(s)[0], idx, mem, crop)

    for epoch in range(1, epochs + 1):
        encoder.train(); head.train(); vibe_head.train()
        perm = train_base[np.random.default_rng(seed + epoch).permutation(len(train_base))]
        ep_c, ep_v, batches = 0.0, 0.0, 0
        t0 = time.time()
        for s in range(steps_per_epoch):
            sel = perm[s * batch_size : (s + 1) * batch_size]
            raw, tgt = gather(sel)
            raw = raw.float().unsqueeze(1).to(memory_format=mem)
            v1 = _gpu_augment(raw, crop, rng, training=True)
            v2 = _gpu_augment(raw, crop, rng, training=True)
            for g in opt.param_groups:
                g["lr"] = lr * lr_at(step)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                e1 = encoder(v1, normalize=False)
                e2 = encoder(v2, normalize=False)
                contrast = nt_xent_loss(head(e1), head(e2))
                # Each view predicts the song's vibe target.
                vibe_pred = vibe_head(torch.cat([e1, e2], 0))
                vibe_true = torch.cat([tgt, tgt], 0)
                vibe_loss = torch.nn.functional.mse_loss(vibe_pred, vibe_true)
                loss = contrast + vibe_lambda * vibe_loss
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            ep_c += float(contrast.item()); ep_v += float(vibe_loss.item()); batches += 1
            step += 1

        msg = (f"epoch {epoch:3d}/{epochs}  contrast {ep_c/batches:.3f}  "
               f"vibe {ep_v/batches:.3f}  ({time.time()-t0:.1f}s)")
        if epoch % eval_every == 0 or epoch == 1 or epoch == epochs:
            val_emb = embed_fn(val_idx)
            probe = knn_genre_probe(val_emb, genres[val_idx], k=10)
            sil = silhouette(val_emb, genres[val_idx])
            msg += f"  | val kNN {probe['knn_accuracy']:.3f}  sil {sil:+.3f}"
            history.append({"epoch": epoch, "contrast": ep_c/batches, "vibe": ep_v/batches,
                            **probe, "silhouette": sil})
            if probe["knn_accuracy"] > best_knn:
                best_knn = probe["knn_accuracy"]
                torch.save({"state_dict": encoder.state_dict(), "embedding_dim": embedding_dim,
                            "width": width, "arch": "resnet"}, out_dir / "encoder_best.pt")
        progress(msg)

    all_emb = embed_fn(np.arange(len(X)))
    np.savez(out_dir / "embeddings.npz", embeddings=all_emb, labels=genres,
             titles=titles, artists=artists, track_ids=track_ids)
    test_emb = embed_fn(test_idx)
    final = knn_genre_probe(test_emb, genres[test_idx], k=10)
    final["silhouette"] = silhouette(test_emb, genres[test_idx])
    final["nn_genre_match"] = nearest_neighbor_genre_match(test_emb, genres[test_idx])
    final["best_val_knn"] = best_knn
    final["minutes"] = round((time.time() - t_start) / 60, 1)
    torch.save({"state_dict": encoder.state_dict(), "embedding_dim": embedding_dim,
                "width": width, "arch": "resnet"}, out_dir / "encoder.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "test_report.json").write_text(json.dumps(final, indent=2))
    progress("\n=== FINAL TEST REPORT (vibe-aware encoder) ===")
    progress(json.dumps(final, indent=2))
    return final


def main(argv: Optional[list] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Train a vibe-aware encoder.")
    parser.add_argument("--packed", default="C:/fma_data/fma_large_packed.npz")
    parser.add_argument("--out-dir", default="ml_data/model_vibe")
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--vibe-lambda", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--log", default=None)
    args = parser.parse_args(argv)

    log_fh = open(args.log, "a", encoding="utf-8", buffering=1) if args.log else None

    def progress(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        if log_fh:
            log_fh.write(line + "\n")

    progress("Device: cuda")
    try:
        train_vibe(Path(args.packed), Path(args.out_dir), epochs=args.epochs,
                   batch_size=args.batch_size, vibe_lambda=args.vibe_lambda,
                   width=args.width, embedding_dim=args.embedding_dim, progress=progress)
    finally:
        if log_fh:
            log_fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
