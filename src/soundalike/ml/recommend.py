"""Use a trained encoder to embed songs and find acoustic neighbors.

This bridges the learned model back to real recommendations: embed a query
track (from its Deezer preview) and return the nearest tracks in the trained
embedding space. Also supports a human-in-the-loop rating loop so we can
actually measure whether the model's idea of "similar" matches yours.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


class TrainedRecommender:
    def __init__(self, model_dir: Path):
        import torch

        from .model import AudioEncoder, ResNetAudioEncoder

        self.model_dir = Path(model_dir)
        data = np.load(self.model_dir / "embeddings.npz", allow_pickle=True)
        self.embeddings = data["embeddings"].astype(np.float32)
        self.labels = data["labels"]
        self.titles = data["titles"]
        self.artists = data["artists"]
        self.track_ids = data["track_ids"]

        ckpt = torch.load(self.model_dir / "encoder.pt", map_location="cpu")
        dim = int(ckpt["embedding_dim"])
        if ckpt.get("arch") == "resnet":
            self.encoder = ResNetAudioEncoder(embedding_dim=dim, width=int(ckpt.get("width", 64)))
        else:
            self.encoder = AudioEncoder(embedding_dim=dim)
        self.encoder.load_state_dict(ckpt["state_dict"])
        self.encoder.eval()

    def _embed_spec(self, spec: np.ndarray) -> np.ndarray:
        import torch

        x = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            return self.encoder(x).cpu().numpy()[0]

    def neighbors_by_index(self, index: int, n: int = 10) -> List[Tuple[str, str, str, float]]:
        query = self.embeddings[index]
        sims = self.embeddings @ query
        order = np.argsort(sims)[::-1]
        out = []
        for i in order:
            if i == index:
                continue
            out.append((str(self.titles[i]), str(self.artists[i]), str(self.labels[i]),
                        float(sims[i])))
            if len(out) >= n:
                break
        return out

    def neighbors_for_song(
        self, title: str, artist: Optional[str] = None, n: int = 10
    ) -> List[Tuple[str, str, str, float]]:
        from ..audio.previews import DeezerClient
        from .spectrogram import SpectrogramConfig, load_audio, log_mel_full, _fit_frames
        from tempfile import TemporaryDirectory

        client = DeezerClient()
        track = client.search_track(title, artist)
        if track is None or not track.has_preview:
            raise LookupError(f"No previewable track for '{title}'.")
        cfg = SpectrogramConfig()
        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "q.mp3"
            client.download_preview(track, dest)
            spec = _fit_frames(log_mel_full(load_audio(dest, cfg.sample_rate), cfg),
                               cfg.target_frames)
        query = self._embed_spec(spec)
        sims = self.embeddings @ query
        order = np.argsort(sims)[::-1]
        out = []
        for i in order:
            out.append((str(self.titles[i]), str(self.artists[i]), str(self.labels[i]),
                        float(sims[i])))
            if len(out) >= n:
                break
        return out
