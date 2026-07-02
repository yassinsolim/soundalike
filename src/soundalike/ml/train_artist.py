"""Fine-tune a *domain-matched, artist-aware* encoder on the real-song library.

The FMA-trained encoder learns audio texture, but FMA is mostly instrumental /
Creative-Commons music, so on real vocal music (pop, R&B, indie, hyperpop) it
confuses scenes — e.g. it places a hyperpop-with-vocals track next to smooth
R&B. Growing the library exposed this: the encoder is the precision ceiling.

This trainer fixes it with the strongest free style signal available on the
harvested library: **the artist**. Two songs by the same artist (or, thanks to
the related-artist harvest, by neighbouring artists) share a sonic identity, so
we fine-tune the encoder with a *supervised contrastive* objective that pulls
same-artist songs together and pushes everything else apart. That teaches "sounds
like the same kind of thing" directly, on exactly the domain users query.

We keep the vibe-target regression as an auxiliary so the embedding stays
grounded in bass/dynamics, and we start from the FMA vibe-aware encoder so the
low-level audio features transfer.

Data: the cached mel-spectrograms + artist labels (no re-downloading).
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from .model import ProjectionHead, ResNetAudioEncoder
from .train_fast import _embed, _gpu_augment
from .vibe_target import VIBE_TARGET_DIM, vibe_targets_for_batch


def _supcon_loss(z, labels, temperature: float = 0.1):
    """Supervised contrastive loss (Khosla et al. 2020).

    z: (B, d) L2-normalized embeddings. labels: (B,) int artist ids. Positives
    for an anchor are the other samples sharing its label (same artist, incl.
    the anchor's second augmented view).
    """
    import torch

    device = z.device
    sim = z @ z.t() / temperature
    # Numerical stability.
    sim = sim - sim.max(dim=1, keepdim=True)[0].detach()
    exp = torch.exp(sim)
    self_mask = torch.eye(len(z), device=device, dtype=torch.bool)
    exp = exp.masked_fill(self_mask, 0.0)
    denom = exp.sum(dim=1, keepdim=True) + 1e-12
    log_prob = sim - torch.log(denom)

    pos_mask = (labels[:, None] == labels[None, :]) & ~self_mask
    pos_counts = pos_mask.sum(dim=1)
    valid = pos_counts > 0
    # Mean log-prob over positives, averaged over anchors that have any.
    pos_log_prob = (log_prob * pos_mask).sum(dim=1)[valid] / pos_counts[valid]
    return -pos_log_prob.mean()


def _pk_batches(labels: np.ndarray, p_artists: int, k_songs: int, seed: int):
    """Yield PK-sampled index batches: P artists x K songs each."""
    rng = np.random.default_rng(seed)
    by_artist = defaultdict(list)
    for i, a in enumerate(labels):
        by_artist[int(a)].append(i)
    artists = [a for a, songs in by_artist.items() if len(songs) >= 2]
    rng.shuffle(artists)
    for start in range(0, len(artists) - p_artists + 1, p_artists):
        chosen = artists[start : start + p_artists]
        batch = []
        for a in chosen:
            songs = by_artist[a]
            pick = rng.choice(songs, size=k_songs, replace=len(songs) < k_songs)
            batch.extend(int(s) for s in pick)
        yield np.array(batch)


def train_artist(
    cache_path: Path,
    out_dir: Path,
    init_model: Optional[Path] = None,
    epochs: int = 40,
    p_artists: int = 128,
    k_songs: int = 4,
    lr: float = 1e-3,
    crop: int = 224,
    width: int = 64,
    embedding_dim: int = 256,
    vibe_lambda: float = 0.5,
    temperature: float = 0.1,
    seed: int = 0,
    progress=print,
) -> dict:
    import torch
    import torch.nn as nn

    from .spec_cache import SpecCache

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mem = torch.channels_last if device == "cuda" else torch.contiguous_format

    cache = SpecCache.load(cache_path)
    X = np.asarray(cache.specs, dtype=np.float16)
    artists = np.asarray(cache.artists, dtype=object)
    uniq = {a: i for i, a in enumerate(sorted(set(artists.tolist())))}
    labels = np.array([uniq[a] for a in artists], dtype=np.int64)
    progress(f"Loaded {len(X)} songs, {len(uniq)} artists, specs {X.shape}")

    # Vibe targets (standardized) from the same specs — grounds bass/dynamics.
    progress("Computing vibe targets...")
    targets = vibe_targets_for_batch(X)
    tmean, tstd = targets.mean(0), targets.std(0) + 1e-6
    targets = ((targets - tmean) / tstd).astype(np.float32)

    # Held-out songs (10%) for a same-artist nearest-neighbour probe.
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X))
    val_idx = perm[: len(X) // 10]
    train_idx = perm[len(X) // 10 :]

    X_res = torch.from_numpy(X).to(device)
    tgt_res = torch.from_numpy(targets).to(device)
    lab_res = torch.from_numpy(labels).to(device)

    def gather(sel):
        s = torch.as_tensor(sel, device=device)
        return X_res[s], tgt_res[s], lab_res[s]

    encoder = ResNetAudioEncoder(embedding_dim=embedding_dim, width=width).to(
        device, memory_format=mem
    )
    if init_model is not None:
        ck = torch.load(Path(init_model) / "encoder.pt", map_location="cpu")
        encoder.load_state_dict({k: v.float() for k, v in ck["state_dict"].items()})
        progress(f"Initialized encoder from {init_model}")
    head = ProjectionHead(in_dim=embedding_dim, hidden=512, out_dim=128).to(device)
    vibe_head = nn.Sequential(
        nn.Linear(embedding_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, VIBE_TARGET_DIM)
    ).to(device)
    params = list(encoder.parameters()) + list(head.parameters()) + list(vibe_head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    steps = sum(1 for _ in _pk_batches(labels[train_idx], p_artists, k_songs, 0))
    total = epochs * max(1, steps)
    warmup = max(1, steps)
    aug_rng = np.random.default_rng(seed)

    def lr_at(step):
        if step < warmup:
            return step / warmup
        p = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))

    out_dir.mkdir(parents=True, exist_ok=True)
    history, best_nn, step = [], 0.0, 0
    t_start = time.time()
    train_labels = labels[train_idx]

    for epoch in range(1, epochs + 1):
        encoder.train(); head.train(); vibe_head.train()
        ep_c, ep_v, nb = 0.0, 0.0, 0
        t0 = time.time()
        for local in _pk_batches(train_labels, p_artists, k_songs, seed + epoch):
            sel = train_idx[local]
            raw, tgt, lab = gather(sel)
            raw = raw.float().unsqueeze(1).to(memory_format=mem)
            v1 = _gpu_augment(raw, crop, aug_rng, training=True)
            v2 = _gpu_augment(raw, crop, aug_rng, training=True)
            for g in opt.param_groups:
                g["lr"] = lr * lr_at(step)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                e1 = encoder(v1, normalize=False)
                e2 = encoder(v2, normalize=False)
                z = nn.functional.normalize(torch.cat([head(e1), head(e2)], 0), dim=1)
                lab2 = torch.cat([lab, lab], 0)
                contrast = _supcon_loss(z, lab2, temperature)
                vibe_pred = vibe_head(torch.cat([e1, e2], 0))
                vibe_true = torch.cat([tgt, tgt], 0)
                vibe_loss = nn.functional.mse_loss(vibe_pred, vibe_true)
                loss = contrast + vibe_lambda * vibe_loss
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            ep_c += float(contrast.item()); ep_v += float(vibe_loss.item()); nb += 1
            step += 1

        msg = (f"epoch {epoch:3d}/{epochs}  supcon {ep_c/nb:.3f}  vibe {ep_v/nb:.3f}  "
               f"({time.time()-t0:.1f}s)")
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            nn_acc = _same_artist_nn(encoder, gather, train_idx, val_idx, labels, mem, crop)
            msg += f"  | val same-artist NN {nn_acc:.3f}"
            history.append({"epoch": epoch, "supcon": ep_c/nb, "vibe": ep_v/nb, "nn": nn_acc})
            if nn_acc > best_nn:
                best_nn = nn_acc
                torch.save({"state_dict": encoder.state_dict(), "embedding_dim": embedding_dim,
                            "width": width, "arch": "resnet"}, out_dir / "encoder_best.pt")
        progress(msg)

    torch.save({"state_dict": encoder.state_dict(), "embedding_dim": embedding_dim,
                "width": width, "arch": "resnet"}, out_dir / "encoder.pt")
    report = {"artists": len(uniq), "songs": len(X), "best_val_nn": best_nn,
              "minutes": round((time.time() - t_start) / 60, 1)}
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "test_report.json").write_text(json.dumps(report, indent=2))
    progress(f"\n=== ARTIST-AWARE ENCODER ===\n{json.dumps(report, indent=2)}")
    return report


def _same_artist_nn(encoder, gather, train_idx, val_idx, labels, mem, crop):
    """Fraction of val songs whose nearest train neighbour shares its artist."""
    tr = _embed(encoder, lambda s: gather(s)[0], train_idx, mem, crop)
    va = _embed(encoder, lambda s: gather(s)[0], val_idx, mem, crop)
    tr /= np.linalg.norm(tr, axis=1, keepdims=True) + 1e-9
    va /= np.linalg.norm(va, axis=1, keepdims=True) + 1e-9
    sims = va @ tr.T
    nn = sims.argmax(axis=1)
    return float(np.mean(labels[train_idx][nn] == labels[val_idx]))


def main(argv: Optional[list] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Train an artist-aware encoder on real songs.")
    parser.add_argument("--cache", default="ml_data/spec_cache.npz")
    parser.add_argument("--out-dir", default="ml_data/model_artist")
    parser.add_argument("--init-model", default="ml_data/model_vibe")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--log", default=None)
    args = parser.parse_args(argv)

    log_fh = open(args.log, "a", encoding="utf-8", buffering=1) if args.log else None

    def progress(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        if log_fh:
            log_fh.write(line + "\n")

    print("Device: cuda")
    try:
        train_artist(Path(args.cache), Path(args.out_dir),
                     init_model=Path(args.init_model) if args.init_model else None,
                     epochs=args.epochs, progress=progress)
    finally:
        if log_fh:
            log_fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
