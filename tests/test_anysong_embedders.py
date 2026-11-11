import numpy as np
import pytest

torch = pytest.importorskip("torch")

from soundalike.anysong.embedders import SampleIDEmb  # noqa: E402


class GlobalMaxNorm(torch.nn.Module):
    """Mirrors sampleid's LogNorm: gain-normalises by the max over its entire input."""

    def forward(self, x):
        x = x.pow_(2).sum(-1).sqrt_()
        x.div_(x.max())
        return x.add_(1e-7).log_().unsqueeze(-3)


class TinyEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.prenorm = GlobalMaxNorm()
        self.proj = torch.nn.Linear(6, 4)

    def embed(self, h):
        return self.proj(h.mean(-1).squeeze(-3)).unsqueeze(1), None

    def forward(self, x):
        return self.embed(self.prenorm(x))[0]


class TinySampleID(torch.nn.Module):
    """Same call structure as sampleid.SampleID: complex transform -> frequency crop -> encoder."""

    crop_size = 1

    def __init__(self):
        super().__init__()
        self.encoder = TinyEncoder()

    def transform(self, x):
        spec = x.reshape(x.shape[0], 8, 16)
        return torch.stack([spec, spec.roll(1, -1)], -1)

    def forward(self, x, audio=True):
        return self.encoder(self.transform(x)[:, self.crop_size:-self.crop_size])


def tiny_embedder():
    torch.manual_seed(0)
    e = SampleIDEmb("cpu")
    e.chunk, e.hop = 128, 64
    e.model = TinySampleID().eval()
    return e


def test_sampleid_embedding_is_independent_of_batch_mates():
    e = tiny_embedder()
    rng = np.random.default_rng(0)
    quiet = 0.01 * rng.standard_normal((1, 320)).astype(np.float32)
    loud = 10.0 * rng.standard_normal((1, 320)).astype(np.float32)
    alone = e.embed(quiet)["win"][0].astype(np.float32)
    batched = e.embed(np.concatenate([loud, quiet]))["win"][1].astype(np.float32)
    np.testing.assert_allclose(batched, alone, atol=2e-3)


def test_sampleid_matches_one_track_per_model_call():
    e = tiny_embedder()
    clips = np.random.default_rng(1).standard_normal((3, 320)).astype(np.float32) * [[0.1], [1.0], [5.0]]
    got = e.embed(clips.astype(np.float32))["win"].astype(np.float32)
    for i, clip in enumerate(clips):
        x = torch.from_numpy(clip.astype(np.float32))
        ch = torch.stack([x[s:s + e.chunk] for s in range(0, len(x) - e.chunk + 1, e.hop)])
        with torch.inference_mode():
            ref = e.model(ch).reshape(len(ch), -1).numpy()
        ref /= np.linalg.norm(ref, axis=-1, keepdims=True)
        np.testing.assert_allclose(got[i], ref, atol=2e-3)
