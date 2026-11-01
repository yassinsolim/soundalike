"""Candidate embedding models behind one interface.

Each embedder maps a batch of fixed-length clips at ``sr`` to a dict of arrays:
- ``track``  [B, D]       L2-normalised clip embedding
- ``win``    [B, W, D]    optional per-window embeddings (for MaxSim)
- ``layers`` [B, L, D]    optional per-layer time-mean features (SSL models)

Licences (weights): CLAP Apache/CC0, MuQ/MuQ-MuLan CC-BY-NC-4.0, MERT CC-BY-NC-4.0,
MAEST CC-BY-NC-SA-4.0, CLEWS MIT, sampleid MIT. Non-commercial use only is
acceptable for this project (free, non-monetised site).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

from .audio import windows


def l2n(x: np.ndarray, axis: int = -1) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), 1e-8)


def _patch_easydict_configs(model: torch.nn.Module) -> None:
    """MuQ builds HF conformer encoders from EasyDict configs; newer transformers
    expects ``_attn_implementation`` on them."""
    for mod in model.modules():
        cfg = getattr(mod, "config", None)
        if cfg is not None and type(cfg).__name__ == "EasyDict" and not hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = "eager"


class Embedder:
    name = "base"
    sr = 24_000

    def __init__(self, device: str = "cuda"):
        self.device = device

    def load(self) -> None:
        raise NotImplementedError

    @torch.inference_mode()
    def embed(self, clips: np.ndarray) -> dict[str, np.ndarray]:
        raise NotImplementedError


class Clap630k(Embedder):
    """Production-family baseline: LAION-CLAP HTSAT-tiny 630k+AudioSet, 3x10 s windows at 48 kHz."""

    name = "clap630k"
    sr = 48_000

    def load(self) -> None:
        import sys

        argv, sys.argv = sys.argv, sys.argv[:1]  # laion_clap parses argv on import
        try:
            import laion_clap
            from huggingface_hub import hf_hub_download
            from laion_clap.clap_module.factory import load_state_dict

            self.model = laion_clap.CLAP_Module(enable_fusion=False, device=self.device)
            path = hf_hub_download("lukewys/laion_clap", "630k-audioset-best.pt")  # == load_ckpt(model_id=1)
            # The official checkpoint pickles numpy scalars (epoch metadata); allow just those.
            from numpy.core.multiarray import scalar

            allow = [(scalar, "numpy.core.multiarray.scalar"), np.dtype,
                     np.dtypes.Float64DType, np.dtypes.Float32DType, np.dtypes.Int64DType]
            with torch.serialization.safe_globals(allow):
                sd = load_state_dict(path, skip_params=True)
            sd.pop("text_branch.embeddings.position_ids", None)  # non-persistent buffer in transformers>=4.31
            self.model.model.load_state_dict(sd)
        finally:
            sys.argv = argv
        self.model.model.eval()

    @torch.inference_mode()
    def embed(self, clips: np.ndarray) -> dict[str, np.ndarray]:
        from laion_clap.hook import float32_to_int16, int16_to_float32

        win = np.stack([windows(c, self.sr, 10.0) for c in clips])  # [B, 3, 480000]
        b, w, t = win.shape
        if t != 480_000 or self.model.model.audio_branch.enable_fusion:
            raise ValueError("CLAP waveform path requires fixed 10-second windows and fusion disabled")
        # The unfused model computes its own GPU mel features. The upstream hook
        # also builds CPU fusion mels that this model never reads.
        quantized = int16_to_float32(float32_to_int16(win.reshape(b * w, t)))
        waveform = torch.from_numpy(quantized)
        e = self.model.model.get_audio_embedding(
            [{"waveform": row} for row in waveform]
        ).float().cpu().numpy()
        e = l2n(e).reshape(b, w, -1)
        return {"track": l2n(e.mean(1)), "win": e.astype(np.float16)}


class MuQMuLanEmb(Embedder):
    name = "muq_mulan"
    sr = 24_000

    def load(self) -> None:
        from muq import MuQMuLan

        self.model = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large").to(self.device).eval()
        _patch_easydict_configs(self.model)

    @torch.inference_mode()
    def embed(self, clips: np.ndarray) -> dict[str, np.ndarray]:
        win = np.stack([windows(c, self.sr, 10.0) for c in clips])
        b, w, t = win.shape
        x = torch.from_numpy(win.reshape(b * w, t)).to(self.device)
        e = self.model.mulan_module.get_audio_latents(x).float().cpu().numpy()
        e = l2n(e).reshape(b, w, -1)
        return {"track": l2n(e.mean(1)), "win": e.astype(np.float16)}


class _SSLLayers(Embedder):
    """Time-mean pooled hidden states of every layer, plus 10 s window means of each layer."""

    amp = True  # fp16 autocast (2070S has no fast bf16)

    def _hidden(self, x: torch.Tensor) -> list[torch.Tensor]:
        raise NotImplementedError

    @torch.inference_mode()
    def embed(self, clips: np.ndarray) -> dict[str, np.ndarray]:
        x = torch.from_numpy(np.asarray(clips, dtype=np.float32)).to(self.device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            hs = self._hidden(x)
        layers = torch.stack([h.float().mean(1) for h in hs], 1)  # [B, L, D]
        # 3 equal time windows per layer, for MaxSim experiments
        thirds = torch.stack([torch.stack([c.float().mean(1) for c in h.chunk(3, dim=1)], 1) for h in hs], 1)
        return {
            "layers": layers.cpu().numpy().astype(np.float16),
            "layer_win": thirds.cpu().numpy().astype(np.float16),  # [B, L, 3, D]
        }


class MuQEmb(_SSLLayers):
    name = "muq"
    sr = 24_000
    amp = False  # fp16 overflows in MuQ's mel/conv front-end (NaN from layer 0)

    def load(self) -> None:
        from muq import MuQ

        self.model = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter").to(self.device).eval()
        _patch_easydict_configs(self.model)

    def _hidden(self, x: torch.Tensor) -> list[torch.Tensor]:
        return list(self.model(x, output_hidden_states=True).hidden_states)


class MertEmb(_SSLLayers):
    name = "mert330"
    sr = 24_000
    repo = "m-a-p/MERT-v1-330M"

    def load(self) -> None:
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(self.repo, trust_remote_code=True).to(self.device).eval()

    def _hidden(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = (x - x.mean(1, keepdim=True)) / torch.sqrt(x.var(1, keepdim=True) + 1e-7)
        return list(self.model(x, output_hidden_states=True).hidden_states)


class Mert95Emb(MertEmb):
    name = "mert95"
    repo = "m-a-p/MERT-v1-95M"


class MaestEmb(_SSLLayers):
    name = "maest"
    sr = 16_000
    repo = "mtg-upf/discogs-maest-30s-pw-129e-519l"

    def load(self) -> None:
        from transformers import AutoFeatureExtractor, AutoModel
        from transformers.audio_utils import mel_filter_bank, window_function

        self.fe = AutoFeatureExtractor.from_pretrained(self.repo, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.repo, trust_remote_code=True).to(self.device).eval()
        self.window = torch.as_tensor(window_function(self.fe.n_fft, "hann"), device=self.device)
        self.mel_filters = torch.as_tensor(mel_filter_bank(
            self.fe.n_fft // 2 + 1, self.fe.num_mel_bins, 0, self.sr / 2, self.sr,
            norm="slaney", mel_scale="slaney"), device=self.device)

    def _features(self, clips: np.ndarray) -> torch.Tensor:
        x = torch.as_tensor(clips, dtype=torch.float64, device=self.device)
        spectrum = torch.stft(x, self.fe.n_fft, hop_length=self.fe.hop_length,
                              window=self.window, center=True, pad_mode="constant", return_complex=True)
        # Match the upstream NumPy frontend's float64 FFT -> complex64 storage
        # -> float64 power/mel calculation before its final float32 conversion.
        power = spectrum.to(torch.complex64).to(torch.complex128).abs().square()
        mel = (power.transpose(1, 2) @ self.mel_filters).clamp_min(1e-30)
        if self.fe.log_compression == "logC":
            mel = torch.log10(1 + mel * 10000)
        elif self.fe.log_compression == "log":
            mel = torch.log(mel + torch.finfo(torch.float64).eps)
        elif self.fe.log_compression:
            raise ValueError(f"Unsupported MAEST log compression: {self.fe.log_compression}")
        mel = mel.float()
        if self.fe.max_length > 0:
            mel = mel[:, :self.fe.max_length]
            if mel.shape[1] < self.fe.max_length:
                mel = torch.nn.functional.pad(mel, (0, 0, 0, self.fe.max_length - mel.shape[1]))
        if self.fe.do_normalize:
            mel = (mel - self.fe.mean) / (self.fe.std * 2)
        return mel

    @torch.inference_mode()
    def embed(self, clips: np.ndarray) -> dict[str, np.ndarray]:
        feats = self._features(clips)
        with torch.autocast("cuda", dtype=torch.float16):
            hs = self.model(input_values=feats, output_hidden_states=True).hidden_states
        # MAEST: CLS + DIST tokens then patches; paper recommends block ~7 with CLS/DIST/mean concat.
        layers = torch.stack([torch.cat([h[:, 0], h[:, 1], h[:, 2:].mean(1)], -1).float() for h in hs], 1)
        return {"layers": layers.cpu().numpy().astype(np.float16)}


class SampleIDEmb(Embedder):
    """Sony sampleid (ICASSP 2026, MIT): 6.125 s chunks at 50 % overlap, as in its multiscale eval."""

    name = "sampleid"
    sr = 16_000
    chunk = 98_000
    hop = 49_000

    def load(self) -> None:
        from sampleid import SampleID

        ckpt = os.environ.get("SAMPLEID_CKPT")
        self.model = SampleID.load_checkpoint(ckpt, device=torch.device(self.device))

    @torch.inference_mode()
    def embed(self, clips: np.ndarray) -> dict[str, np.ndarray]:
        x = torch.from_numpy(np.asarray(clips, dtype=np.float32))
        starts = list(range(0, x.shape[1] - self.chunk + 1, self.hop))
        ch = torch.stack([x[:, s:s + self.chunk] for s in starts], 1)
        b, w, t = ch.shape
        spec = self.model.transform(ch.reshape(b * w, t).to(self.device))
        crop = self.model.crop_size
        spec = spec[:, crop:-crop]
        # Upstream LogNorm gain-normalises by the max over its whole input, and upstream evaluation
        # feeds one track per call; apply it per track so embeddings don't depend on batch-mates.
        enc = self.model.encoder
        h = torch.cat([enc.prenorm(spec[i * w:(i + 1) * w]) for i in range(b)])
        e = enc.embed(h)[0].reshape(b * w, -1).float().cpu().numpy()
        e = l2n(e).reshape(b, w, -1)
        return {"track": l2n(e.mean(1)), "win": e.astype(np.float16)}


class ClewsEmb(Embedder):
    """Sony CLEWS (ICML 2025, MIT) musical-version embeddings: 20 s CQT shingles, hop 5 s.

    Shingle embeddings are kept un-normalised; CLEWS compares them with normalised
    squared Euclidean distance and best-pairs-without-replacement reduction."""

    name = "clews_dvi"
    sr = 16_000
    variant = "dvi-clews"

    def load(self) -> None:
        import importlib

        from omegaconf import OmegaConf

        repo = Path(os.environ.get("SOUNDALIKE_EXT", "/opt/soundalike/ext")) / "clews"
        ckdir = Path(os.environ.get("CLEWS_CKPT", "/opt/soundalike/ckpt/clews")) / self.variant
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        conf = OmegaConf.load(ckdir / "configuration.yaml")
        module = importlib.import_module("models." + conf.model.name)
        self.model = module.Model(conf.model, sr=conf.data.samplerate)
        state = torch.load(ckdir / "checkpoint_best.ckpt", map_location="cpu", weights_only=False)
        sd = state["model"] if "model" in state else state
        self.model.load_state_dict(sd)
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def embed(self, clips: np.ndarray) -> dict[str, np.ndarray]:
        x = torch.from_numpy(np.asarray(clips, dtype=np.float32)).to(self.device)
        z = self.model(x, shingle_hop=5.0).float().cpu().numpy()  # [B, S, 1024]
        return {"track": l2n(z.mean(1)), "win": z.astype(np.float16)}


class ClewsShsEmb(ClewsEmb):
    name = "clews_shs"
    variant = "shs-clews"


REGISTRY: dict[str, type[Embedder]] = {
    cls.name: cls for cls in (Clap630k, MuQMuLanEmb, MuQEmb, MertEmb, Mert95Emb, MaestEmb, SampleIDEmb,
                              ClewsEmb, ClewsShsEmb)
}


def register(cls: type[Embedder]) -> type[Embedder]:
    REGISTRY[cls.name] = cls
    return cls
