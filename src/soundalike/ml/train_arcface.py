"""Experiment: train the artist-aware encoder with an ArcFace margin objective.

The shipped encoder uses supervised-contrastive (NT-Xent over same-artist
positives). ArcFace (additive angular margin, Deng et al. 2019) is the gold
standard for retrieval/verification: it learns a per-artist weight vector and
adds an angular margin to the true artist's logit, forcing embeddings to sit
tighter around their artist and further from others than plain contrastive does.

This trainer mirrors train_artist (same base init, PK sampling, augmentation,
vibe-target auxiliary, epochs) so the only variable is the objective, making the
benchmark comparison fair. Optionally keeps a SupCon term alongside ArcFace.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

from .model import ProjectionHead, ResNetAudioEncoder
from .train_artist import _pk_batches, _same_artist_nn, _supcon_loss
from .train_fast import _embed, _gpu_augment
from .vibe_target import VIBE_TARGET_DIM, vibe_targets_for_batch


def train_arcface(
    cache_path: Path,
    out_dir: Path,
    init_model: Optional[Path] = None,
    epochs: int = 55,
    p_artists: int = 128,
    k_songs: int = 4,
    lr: float = 1e-3,
    crop: int = 224,
    width: int = 64,
    embedding_dim: int = 384,
    pool_type: str = "avg",
    vibe_lambda: float = 0.5,
    arc_margin: float = 0.2,
    arc_scale: float = 24.0,
    supcon_lambda: float = 0.5,
    temperature: float = 0.1,
    seed: int = 0,
    progress=print,
) -> dict:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from .spec_cache import SpecCache

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mem = torch.channels_last if device == "cuda" else torch.contiguous_format

    cache = SpecCache.load(cache_path)
    X = np.asarray(cache.specs, dtype=np.float16)
    artists = np.asarray(cache.artists, dtype=object)
    uniq = {a: i for i, a in enumerate(sorted(set(artists.tolist())))}
    labels = np.array([uniq[a] for a in artists], dtype=np.int64)
    n_classes = len(uniq)
    progress(f"Loaded {len(X)} songs, {n_classes} artists, specs {X.shape}")

    progress("Computing vibe targets...")
    targets = vibe_targets_for_batch(X)
    tmean, tstd = targets.mean(0), targets.std(0) + 1e-6
    targets = ((targets - tmean) / tstd).astype(np.float32)

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

    encoder = ResNetAudioEncoder(
        embedding_dim=embedding_dim, width=width, pool_type=pool_type
    ).to(device, memory_format=mem)
    if init_model is not None:
        ck = torch.load(Path(init_model) / "encoder.pt", map_location="cpu")
        missing, unexpected = encoder.load_state_dict(
            {k: v.float() for k, v in ck["state_dict"].items()}, strict=(pool_type != "gem")
        )
        if pool_type == "gem":
            if set(missing) - {"pool.p"} or unexpected:
                raise RuntimeError(f"Unexpected warm-start mismatch: missing={missing} unexpected={unexpected}")
            with torch.no_grad():
                encoder.pool.p.fill_(1.0)  # start == avg pooling (see train_artist)
        progress(f"Initialized encoder from {init_model}")

    # ArcFace weight matrix: one L2-normalized prototype per artist.
    arc_W = nn.Parameter(torch.empty(n_classes, embedding_dim, device=device))
    nn.init.xavier_uniform_(arc_W)
    head = ProjectionHead(in_dim=embedding_dim, hidden=512, out_dim=128).to(device)
    vibe_head = nn.Sequential(
        nn.Linear(embedding_dim, 128), nn.ReLU(inplace=True), nn.Linear(128, VIBE_TARGET_DIM)
    ).to(device)
    params = (list(encoder.parameters()) + [arc_W] + list(head.parameters())
              + list(vibe_head.parameters()))
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    cos_m, sin_m = math.cos(arc_margin), math.sin(arc_margin)
    th = math.cos(math.pi - arc_margin)
    mm = math.sin(math.pi - arc_margin) * arc_margin

    def arcface_loss(emb, lab):
        # cos(theta) between each embedding and every artist prototype.
        cosine = F.linear(F.normalize(emb), F.normalize(arc_W)).clamp(-1 + 1e-6, 1 - 1e-6)
        sine = torch.sqrt(1.0 - cosine ** 2)
        phi = cosine * cos_m - sine * sin_m  # cos(theta + m)
        # keep monotonic where theta+m would exceed pi.
        phi = torch.where(cosine > th, phi, cosine - mm)
        one_hot = F.one_hot(lab, n_classes).float()
        logits = (one_hot * phi + (1.0 - one_hot) * cosine) * arc_scale
        return F.cross_entropy(logits, lab)

    eligible = sum(1 for c in Counter(labels[train_idx].tolist()).values() if c >= 2)
    if eligible < p_artists:
        p_artists = max(2, eligible)
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
        ep_a, ep_c, ep_v, nb = 0.0, 0.0, 0.0, 0
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
                arc = 0.5 * (arcface_loss(e1, lab) + arcface_loss(e2, lab))
                if supcon_lambda > 0:
                    z = F.normalize(torch.cat([head(e1), head(e2)], 0), dim=1)
                    contrast = _supcon_loss(z, torch.cat([lab, lab], 0), temperature)
                else:
                    contrast = torch.zeros((), device=device)
                vibe_pred = vibe_head(torch.cat([e1, e2], 0))
                vibe_loss = F.mse_loss(vibe_pred, torch.cat([tgt, tgt], 0))
                loss = arc + supcon_lambda * contrast + vibe_lambda * vibe_loss
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            ep_a += float(arc.item()); ep_c += float(contrast.item())
            ep_v += float(vibe_loss.item()); nb += 1
            step += 1

        nb = max(1, nb)
        msg = (f"epoch {epoch:3d}/{epochs}  arc {ep_a/nb:.3f}  supcon {ep_c/nb:.3f}  "
               f"vibe {ep_v/nb:.3f}  ({time.time()-t0:.1f}s)")
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            nn_acc = _same_artist_nn(encoder, gather, train_idx, val_idx, labels, mem, crop)
            msg += f"  | val same-artist NN {nn_acc:.3f}"
            history.append({"epoch": epoch, "arc": ep_a/nb, "supcon": ep_c/nb,
                            "vibe": ep_v/nb, "nn": nn_acc})
            if nn_acc > best_nn:
                best_nn = nn_acc
                torch.save({"state_dict": encoder.state_dict(), "embedding_dim": embedding_dim,
                            "width": width, "arch": "resnet", "pool_type": pool_type},
                           out_dir / "encoder_best.pt")
        progress(msg)

    torch.save({"state_dict": encoder.state_dict(), "embedding_dim": embedding_dim,
                "width": width, "arch": "resnet", "pool_type": pool_type}, out_dir / "encoder.pt")
    report = {"artists": n_classes, "songs": len(X), "best_val_nn": best_nn,
              "arc_margin": arc_margin, "arc_scale": arc_scale,
              "supcon_lambda": supcon_lambda, "minutes": round((time.time() - t_start) / 60, 1)}
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "test_report.json").write_text(json.dumps(report, indent=2))
    progress(f"\n=== ARCFACE ENCODER ===\n{json.dumps(report, indent=2)}")
    return report


def main(argv: Optional[list] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Train an ArcFace artist encoder.")
    parser.add_argument("--cache", default="ml_data/spec_cache_dedup.npz")
    parser.add_argument("--out-dir", default="ml_data/model_arcface")
    parser.add_argument("--init-model", default="ml_data/model_vibe384")
    parser.add_argument("--epochs", type=int, default=55)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=384)
    parser.add_argument("--pool-type", default="avg", choices=["avg", "gem"])
    parser.add_argument("--arc-margin", type=float, default=0.2)
    parser.add_argument("--arc-scale", type=float, default=24.0)
    parser.add_argument("--supcon-lambda", type=float, default=0.5)
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
        train_arcface(Path(args.cache), Path(args.out_dir),
                      init_model=Path(args.init_model) if args.init_model else None,
                      epochs=args.epochs, width=args.width, embedding_dim=args.embedding_dim,
                      pool_type=args.pool_type,
                      arc_margin=args.arc_margin, arc_scale=args.arc_scale,
                      supcon_lambda=args.supcon_lambda, progress=progress)
    finally:
        if log_fh:
            log_fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
