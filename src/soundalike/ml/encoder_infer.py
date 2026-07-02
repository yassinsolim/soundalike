"""Load a trained encoder as a feature extractor for arbitrary audio.

The FMA-trained encoder learned a rich *timbre/texture* representation. Here we
reuse it purely as a feature extractor: given any preview file, produce its
neural embedding. This is what lets the deep-vibe engine apply the learned model
to real, popular songs (not just the FMA catalog it was trained on).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


class EncoderExtractor:
    """Wraps a trained encoder checkpoint to embed mel-spectrograms."""

    def __init__(self, model_dir: Path, device: Optional[str] = None):
        import torch

        from .model import AudioEncoder, ResNetAudioEncoder
        from .spectrogram import SpectrogramConfig

        self.model_dir = Path(model_dir)
        ckpt = torch.load(self.model_dir / "encoder.pt", map_location="cpu")
        dim = int(ckpt["embedding_dim"])
        if ckpt.get("arch") == "resnet":
            self.encoder = ResNetAudioEncoder(embedding_dim=dim, width=int(ckpt.get("width", 64)))
        else:
            self.encoder = AudioEncoder(embedding_dim=dim)
        self.encoder.load_state_dict(ckpt["state_dict"])
        self.encoder.eval()
        self.embedding_dim = dim
        self.cfg = SpectrogramConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device)

    def embed_file(self, path: str | Path) -> np.ndarray:
        """Preview file -> L2-normalized neural embedding."""
        from .spectrogram import _fit_frames, load_audio, log_mel_full

        spec = _fit_frames(
            log_mel_full(load_audio(path, self.cfg.sample_rate), self.cfg),
            self.cfg.target_frames,
        )
        return self.embed_spec(spec)

    def embed_spec(self, spec: np.ndarray) -> np.ndarray:
        import torch

        x = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.encoder(x).float().cpu().numpy()[0]

    def embed_specs(self, specs: np.ndarray, batch: int = 256) -> np.ndarray:
        """Batch of (N, n_mels, frames) -> (N, dim) embeddings."""
        import torch

        out = []
        with torch.no_grad():
            for start in range(0, len(specs), batch):
                x = torch.from_numpy(specs[start : start + batch]).unsqueeze(1).to(self.device)
                out.append(self.encoder(x).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.embedding_dim), np.float32)
