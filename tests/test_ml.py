"""CPU-only tests for the ML pipeline (no GPU or network required)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from soundalike.ml.model import AudioEncoder, ProjectionHead, ResNetAudioEncoder, nt_xent_loss
from soundalike.ml.spectrogram import (
    SpectrogramConfig,
    augment,
    mel_spectrogram,
    two_views,
)


def _tone(freq: float, sr: int = 22050, seconds: float = 5.0) -> np.ndarray:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ------------------------------------------------------------------ spectrogram
def test_mel_spectrogram_fixed_shape():
    cfg = SpectrogramConfig()
    spec = mel_spectrogram(_tone(440.0), cfg)
    assert spec.shape == (cfg.n_mels, cfg.target_frames)
    assert np.isfinite(spec).all()


def test_mel_spectrogram_pads_short_clips():
    cfg = SpectrogramConfig(target_frames=512)
    spec = mel_spectrogram(_tone(440.0, seconds=1.0), cfg)
    assert spec.shape == (cfg.n_mels, 512)


def test_augment_changes_but_preserves_shape():
    spec = mel_spectrogram(_tone(440.0), SpectrogramConfig())
    a = augment(spec, np.random.default_rng(0))
    assert a.shape == spec.shape
    assert not np.allclose(a, spec)


def test_two_views_differ():
    spec = mel_spectrogram(_tone(440.0), SpectrogramConfig())
    v1, v2 = two_views(spec, np.random.default_rng(1))
    assert v1.shape == v2.shape == spec.shape
    assert not np.allclose(v1, v2)


# ------------------------------------------------------------------------ model
def test_encoder_output_normalized():
    enc = AudioEncoder(embedding_dim=64).eval()
    x = torch.randn(4, 1, 128, 256)
    with torch.no_grad():
        z = enc(x)
    assert z.shape == (4, 64)
    norms = z.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_projection_head_shape():
    head = ProjectionHead(in_dim=64, out_dim=32)
    out = head(torch.randn(5, 64))
    assert out.shape == (5, 32)


def test_nt_xent_positive_pairs_lower_loss():
    torch.manual_seed(0)
    base = torch.nn.functional.normalize(torch.randn(16, 32), dim=1)

    # Good: the two views are nearly identical -> positives are clearly closest.
    good = nt_xent_loss(base, base + 0.01 * torch.randn_like(base))
    # Bad: the second view is unrelated random noise.
    bad = nt_xent_loss(base, torch.nn.functional.normalize(torch.randn(16, 32), dim=1))
    assert good < bad
    assert torch.isfinite(good) and good >= 0


def test_encoder_trains_one_step():
    enc = AudioEncoder(embedding_dim=32)
    head = ProjectionHead(in_dim=32, out_dim=16)
    opt = torch.optim.SGD(list(enc.parameters()) + list(head.parameters()), lr=0.1)
    x1 = torch.randn(8, 1, 128, 256)
    x2 = x1 + 0.01 * torch.randn_like(x1)
    loss0 = nt_xent_loss(head(enc(x1, normalize=False)), head(enc(x2, normalize=False)))
    loss0.backward()
    opt.step()
    assert torch.isfinite(loss0)


def test_resnet_encoder_output_normalized():
    enc = ResNetAudioEncoder(embedding_dim=64, width=16).eval()
    x = torch.randn(3, 1, 128, 256)
    with torch.no_grad():
        z = enc(x)
    assert z.shape == (3, 64)
    norms = z.norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)


def test_resnet_width_scales_params():
    small = sum(p.numel() for p in ResNetAudioEncoder(width=16).parameters())
    big = sum(p.numel() for p in ResNetAudioEncoder(width=64).parameters())
    assert big > small * 3  # wider network has many more params


def test_knn_probe_and_metrics_recover_clusters():
    from soundalike.ml.evaluate import (
        chance_accuracy,
        knn_genre_probe,
        nearest_neighbor_genre_match,
    )

    rng = np.random.default_rng(0)
    # Two well-separated Gaussian blobs => trivially separable "genres".
    a = rng.normal(loc=[5, 0, 0], scale=0.3, size=(60, 3))
    b = rng.normal(loc=[-5, 0, 0], scale=0.3, size=(60, 3))
    emb = np.vstack([a, b]).astype(np.float32)
    labels = np.array(["A"] * 60 + ["B"] * 60)

    probe = knn_genre_probe(emb, labels, k=5)
    assert probe["knn_accuracy"] > 0.95
    assert nearest_neighbor_genre_match(emb, labels) > 0.95
    assert abs(chance_accuracy(labels) - 0.5) < 1e-9


def test_pack_fix_width_truncate_and_pad():
    from soundalike.ml.pack import _fix_width

    spec = np.random.rand(128, 700).astype(np.float32)
    assert _fix_width(spec, 512).shape == (128, 512)
    short = np.random.rand(128, 300).astype(np.float32)
    padded = _fix_width(short, 512)
    assert padded.shape == (128, 512)
    # Original content preserved in the first 300 columns.
    assert np.allclose(padded[:, :300], short)


def test_stratified_split_is_disjoint_and_covers_genres():
    from soundalike.ml.train_fast import _stratified_idx

    genres = np.array(["rock"] * 50 + ["jazz"] * 30 + ["pop"] * 20)
    train, val, test = _stratified_idx(genres, val_frac=0.2, test_frac=0.2, seed=1)
    all_idx = np.concatenate([train, val, test])
    assert len(all_idx) == len(genres)
    assert len(set(all_idx.tolist())) == len(genres)  # disjoint, full coverage
    # Every split should contain all three genres (stratified).
    for split in (train, val, test):
        assert len(set(genres[split].tolist())) == 3


def test_stratified_split_routes_unlabeled_to_train_only():
    from soundalike.ml.train_fast import _stratified_idx

    genres = np.array(["rock"] * 20 + ["jazz"] * 20 + ["unlabeled"] * 40)
    train, val, test = _stratified_idx(genres, val_frac=0.2, test_frac=0.2, seed=0)
    # Unlabeled tracks must never appear in val/test (can't be evaluated).
    assert "unlabeled" not in set(genres[val].tolist())
    assert "unlabeled" not in set(genres[test].tolist())
    # But they must all be used for training.
    assert (genres[train] == "unlabeled").sum() == 40
