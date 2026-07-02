"""Maximum-throughput contrastive training with the whole dataset on the GPU.

Once spectrograms are packed (see soundalike.ml.pack) the dataset is only a few
GB, so we upload it to the GPU once as float16 and never touch disk during
training. Batching, random cropping and SpecAugment-style masking all happen on
the GPU, which keeps the RTX 5080 near 100% utilization instead of starving on
data loading.

Objective: self-supervised NT-Xent (two augmented crops of the same track are
positives). Genre labels are used only for evaluation (kNN probe / silhouette).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .evaluate import knn_genre_probe, nearest_neighbor_genre_match, silhouette
from .model import ProjectionHead, ResNetAudioEncoder, nt_xent_loss


def _stratified_idx(genres: np.ndarray, val_frac=0.1, test_frac=0.1, seed=0):
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
def _stratified_idx(genres, val_frac=0.1, test_frac=0.1, seed=0):
    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for g in np.unique(genres):
        idx = np.where(genres == g)[0]
        rng.shuffle(idx)
        # Unlabeled tracks can't be evaluated -> use them all for training only.
        if g == "unlabeled":
            train += idx.tolist()
            continue
        n = len(idx)
        nt, nv = int(n * test_frac), int(n * val_frac)
        test += idx[:nt].tolist()
        val += idx[nt : nt + nv].tolist()
        train += idx[nt + nv :].tolist()
    return np.array(train), np.array(val), np.array(test)


def _gpu_augment(batch, crop: int, rng, training: bool = True):
    """Random-crop a time window and apply SpecAugment masks, all on GPU.

    batch: (B, 1, n_mels, frames) float. Returns (B, 1, n_mels, crop).
    """
    import torch

    b, _, n_mels, frames = batch.shape
    if frames > crop:
        if training:
            starts = torch.randint(0, frames - crop + 1, (1,), device=batch.device).item()
        else:
            starts = (frames - crop) // 2
        batch = batch[:, :, :, starts : starts + crop]
    if not training:
        return batch

    out = batch.clone()
    fill = out.amin(dim=(2, 3), keepdim=True)
    # Frequency mask.
    fw = int(rng.integers(1, max(2, n_mels // 8)))
    f0 = int(rng.integers(0, n_mels - fw))
    out[:, :, f0 : f0 + fw, :] = fill
    # Time mask.
    tw = int(rng.integers(1, max(2, crop // 8)))
    t0 = int(rng.integers(0, crop - tw))
    out[:, :, :, t0 : t0 + tw] = fill
    # Per-sample gain jitter.
    gain = torch.empty(b, 1, 1, 1, device=batch.device).uniform_(0.9, 1.1)
    return out * gain


def _embed(encoder, gather, idx, mem, crop=256, batch=512):
    import torch

    encoder.eval()
    embs = []
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for start in range(0, len(idx), batch):
            sl = idx[start : start + batch]
            xb = gather(sl).float().unsqueeze(1).to(memory_format=mem)
            xb = _gpu_augment(xb, crop, rng, training=False)
            with torch.amp.autocast("cuda"):
                embs.append(encoder(xb).float().cpu().numpy())
    return np.concatenate(embs)


def train_packed(
    packed: Path,
    out_dir: Path,
    epochs: int = 80,
    batch_size: int = 512,
    lr: float = 3e-3,
    width: int = 64,
    embedding_dim: int = 256,
    crop: int = 256,
    eval_every: int = 5,
    seed: int = 0,
    progress=print,
) -> dict:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mem = torch.channels_last if device == "cuda" else torch.contiguous_format

    data = np.load(packed, allow_pickle=True)
    X = data["X"]  # (N, n_mels, frames) float16
    genres = data["genres"]
    titles, artists, track_ids = data["titles"], data["artists"], data["track_ids"]
    progress(f"Loaded packed dataset {X.shape} ({X.nbytes/1e9:.1f} GB) ")

    train_idx, val_idx, test_idx = _stratified_idx(genres, seed=seed)
    progress(f"Split -> train {len(train_idx)}  val {len(val_idx)}  test {len(test_idx)}")

    # Decide data residency: keep the whole dataset on the GPU when it fits with
    # room to spare (fastest), otherwise keep it in pinned CPU RAM and stream
    # each batch over PCIe (still fast — a batch is only tens of MB).
    dataset_gb = X.nbytes / 1e9
    gpu_resident = False
    if device == "cuda":
        free_b, total_b = torch.cuda.mem_get_info()
        gpu_resident = dataset_gb < (total_b / 1e9) - 7.0  # leave ~7GB for training
    X_t = torch.from_numpy(X)
    if gpu_resident:
        X_res = X_t.to(device)
        progress(f"Dataset GPU-resident: {dataset_gb:.1f} GB in VRAM")
        def gather(sel):
            sel_t = torch.as_tensor(sel, device=device)
            return X_res[sel_t]
    else:
        X_res = X_t.pin_memory() if device == "cuda" else X_t
        progress(f"Dataset CPU-resident ({dataset_gb:.1f} GB); streaming batches to GPU")
        def gather(sel):
            sel_cpu = torch.as_tensor(sel).cpu()
            return X_res[sel_cpu].to(device, non_blocking=True)

    encoder = ResNetAudioEncoder(embedding_dim=embedding_dim, width=width).to(
        device, memory_format=mem
    )
    head = ProjectionHead(in_dim=embedding_dim, hidden=512, out_dim=128).to(device)
    params = list(encoder.parameters()) + list(head.parameters())
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

    for epoch in range(1, epochs + 1):
        encoder.train(); head.train()
        perm = train_base[np.random.default_rng(seed + epoch).permutation(len(train_base))]
        epoch_loss = 0.0
        t0 = time.time()
        for s in range(steps_per_epoch):
            sel = perm[s * batch_size : (s + 1) * batch_size]
            raw = gather(sel).float().unsqueeze(1).to(memory_format=mem)
            v1 = _gpu_augment(raw, crop, rng, training=True)
            v2 = _gpu_augment(raw, crop, rng, training=True)
            for g in opt.param_groups:
                g["lr"] = lr * lr_at(step)
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
            val_emb = _embed(encoder, gather, val_idx, mem, crop)
            probe = knn_genre_probe(val_emb, genres[val_idx], k=10)
            sil = silhouette(val_emb, genres[val_idx])
            msg += f"  | val kNN {probe['knn_accuracy']:.3f}  sil {sil:+.3f}"
            history.append({"epoch": epoch, "loss": avg, **probe, "silhouette": sil})
            if probe["knn_accuracy"] > best_knn:
                best_knn = probe["knn_accuracy"]
                torch.save({"state_dict": encoder.state_dict(), "embedding_dim": embedding_dim,
                            "width": width, "arch": "resnet"}, out_dir / "encoder_best.pt")
        progress(msg)

    # Final artifacts: embed everything, evaluate on held-out test set.
    all_emb = _embed(encoder, gather, np.arange(len(X)), mem, crop)
    np.savez(out_dir / "embeddings.npz", embeddings=all_emb, labels=genres,
             titles=titles, artists=artists, track_ids=track_ids)
    test_emb = _embed(encoder, gather, test_idx, mem, crop)
    final = knn_genre_probe(test_emb, genres[test_idx], k=10)
    final["silhouette"] = silhouette(test_emb, genres[test_idx])
    final["nn_genre_match"] = nearest_neighbor_genre_match(test_emb, genres[test_idx])
    final["best_val_knn"] = best_knn
    final["minutes"] = round((time.time() - t_start) / 60, 1)
    torch.save({"state_dict": encoder.state_dict(), "embedding_dim": embedding_dim,
                "width": width, "arch": "resnet"}, out_dir / "encoder.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "test_report.json").write_text(json.dumps(final, indent=2))
    progress("\n=== FINAL TEST REPORT ===")
    progress(json.dumps(final, indent=2))
    return final


def main(argv: Optional[list] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="GPU-resident contrastive training.")
    parser.add_argument("--packed", default="ml_data/fma_packed.npz")
    parser.add_argument("--out-dir", default="ml_data/model_fma")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--width", type=int, default=64)
    args = parser.parse_args(argv)
    print("Device:", "cuda")
    train_packed(Path(args.packed), Path(args.out_dir), epochs=args.epochs,
                 batch_size=args.batch_size, width=args.width)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
