"""Audio embedding model and self-supervised contrastive loss.

The encoder is a small convolutional network that maps a mel-spectrogram to a
fixed-length embedding vector. Songs that *sound* alike should land close
together in this space.

We train it self-supervised with NT-Xent (the SimCLR loss): two augmented views
of the same clip are pulled together, while views of different clips are pushed
apart. This needs no similarity labels — exactly right, since nobody hands us
ground-truth "these two songs are similar" data.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class AudioEncoder(nn.Module):
    """Mel-spectrogram -> L2-normalized embedding."""

    def __init__(self, n_mels: int = 128, embedding_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(1, 32),
            _conv_block(32, 64),
            _conv_block(64, 128),
            _conv_block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embed = nn.Linear(256, embedding_dim)

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        # x: (batch, 1, n_mels, frames)
        h = self.features(x)
        h = self.pool(h).flatten(1)
        z = self.embed(h)
        return F.normalize(z, dim=1) if normalize else z


class GeMPool(nn.Module):
    """Generalized-mean pooling (Radenovic et al. 2018).

    Average pooling weights every time-frequency cell equally; max pooling keeps
    only the single loudest cell. GeM interpolates between them with a learnable
    exponent ``p``: ``(mean(x**p))**(1/p)``. p=1 is average pooling, p->inf
    approaches max. Letting the network learn ``p`` lets it decide how peaky the
    pooled summary should be, which is a well-known retrieval win because the
    discriminative evidence in a spectrogram is often concentrated (a drop, a
    vocal run) rather than spread evenly across the clip.
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * float(p))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, C, 1, 1), matching AdaptiveAvgPool2d(1).
        clamped = x.clamp(min=self.eps).pow(self.p)
        pooled = F.avg_pool2d(clamped, (x.size(-2), x.size(-1)))
        return pooled.pow(1.0 / self.p)


class ResidualBlock(nn.Module):
    """A pre-activation residual block with an optional downsample."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x)), inplace=True)
        h = self.bn2(self.conv2(h))
        return F.relu(h + self.skip(x), inplace=True)


class ResNetAudioEncoder(nn.Module):
    """A larger ResNet-style encoder for training on big datasets (e.g. FMA).

    `width` scales every stage's channel count, so the same code gives a small
    or large model. Produces an L2-normalized embedding.
    """

    def __init__(self, embedding_dim: int = 256, width: int = 64, pool_type: str = "avg"):
        super().__init__()
        w = width
        # Downsample early (like ResNet's stem): 7x7 stride-2 conv + maxpool takes
        # a 128x256 spectrogram down to 32x64 BEFORE the residual blocks, cutting
        # the expensive high-resolution compute ~16x with negligible quality loss.
        self.stem = nn.Sequential(
            nn.Conv2d(1, w, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(w),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layers = nn.Sequential(
            ResidualBlock(w, w),
            ResidualBlock(w, 2 * w, stride=2),
            ResidualBlock(2 * w, 2 * w),
            ResidualBlock(2 * w, 4 * w, stride=2),
            ResidualBlock(4 * w, 4 * w),
            ResidualBlock(4 * w, 8 * w, stride=2),
        )
        self.pool_type = pool_type
        self.pool = GeMPool() if pool_type == "gem" else nn.AdaptiveAvgPool2d(1)
        self.embed = nn.Linear(8 * w, embedding_dim)

    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        h = self.stem(x)
        h = self.layers(h)
        h = self.pool(h).flatten(1)
        z = self.embed(h)
        return F.normalize(z, dim=1) if normalize else z


class ProjectionHead(nn.Module):
    """Small MLP used only during contrastive training (discarded at inference)."""

    def __init__(self, in_dim: int = 128, hidden: int = 256, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """Normalized temperature-scaled cross-entropy (SimCLR) loss.

    z1, z2 are (batch, dim) L2-normalized projections of the two augmented
    views. For each sample, its positive is the matching view; all other 2N-2
    samples in the batch are negatives.
    """
    batch = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # (2N, dim)
    sim = z @ z.t() / temperature  # cosine similarities (z is normalized)

    # Mask out self-similarity on the diagonal.
    diag = torch.eye(2 * batch, dtype=torch.bool, device=z.device)
    sim.masked_fill_(diag, float("-inf"))

    # Positive index for row i is its partner view.
    targets = torch.arange(2 * batch, device=z.device)
    targets = (targets + batch) % (2 * batch)
    return F.cross_entropy(sim, targets)
