import sys
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from transformers.audio_utils import mel_filter_bank, spectrogram, window_function

from soundalike.anysong.embedders import Clap630k, MaestEmb


@pytest.mark.parametrize("compression", ["logC", "log", None])
@pytest.mark.parametrize("max_length", [20, 100])
def test_maest_features_match_upstream_numpy(compression, max_length):
    model = MaestEmb(device="cpu")
    model.fe = SimpleNamespace(n_fft=512, hop_length=256, num_mel_bins=96,
                               max_length=max_length, log_compression=compression,
                               do_normalize=True, mean=2.06755686098554, std=1.268292820667291)
    window = window_function(512, "hann")
    filters = mel_filter_bank(257, 96, 0, 8000, 16000, norm="slaney", mel_scale="slaney")
    model.window = torch.as_tensor(window)
    model.mel_filters = torch.as_tensor(filters)
    signal = np.random.default_rng(42).standard_normal(8192).astype(np.float32)
    impulse = np.zeros_like(signal)
    impulse[4096] = 1
    inputs = np.stack([signal, signal * 1e-6, np.zeros_like(signal), impulse])
    expected = []
    for waveform in inputs:
        mel = spectrogram(waveform, window, 512, 256, power=2, mel_filters=filters,
                          min_value=1e-30, mel_floor=1e-30, pad_mode="constant").T
        if compression == "logC":
            mel = np.log10(1 + mel * 10000)
        elif compression == "log":
            mel = np.log(mel + np.finfo(float).eps)
        mel = mel.astype(np.float32)[:max_length]
        mel = np.pad(mel, ((0, max(0, max_length - len(mel))), (0, 0)))
        expected.append((mel - model.fe.mean) / (model.fe.std * 2))
    torch.testing.assert_close(model._features(inputs), torch.from_numpy(np.stack(expected)),
                               rtol=1e-5, atol=1e-6)


def test_clap_preserves_quantization_without_unused_fusion_features(monkeypatch):
    seen = []

    def get_embedding(rows):
        assert all(set(row) == {"waveform"} for row in rows)
        seen.extend(row["waveform"].numpy() for row in rows)
        return torch.ones(len(rows), 4)

    monkeypatch.setitem(sys.modules, "laion_clap.hook", SimpleNamespace(
        float32_to_int16=lambda x: (np.clip(x, -1, 1) * 32767).astype(np.int16),
        int16_to_float32=lambda x: (x / 32767.0).astype(np.float32)))
    model = Clap630k(device="cpu")
    model.model = SimpleNamespace(model=SimpleNamespace(
        audio_branch=SimpleNamespace(enable_fusion=False), get_audio_embedding=get_embedding))
    audio = np.linspace(-1.5, 1.5, 1_440_000, dtype=np.float32)[None]
    output = model.embed(audio)
    expected = ((np.clip(audio, -1, 1) * 32767).astype(np.int16) / 32767).astype(np.float32)
    np.testing.assert_array_equal(np.concatenate(seen), expected[0])
    assert output["track"].shape == (1, 4)
    assert output["win"].shape == (1, 3, 4)
    model.model.model.audio_branch.enable_fusion = True
    with pytest.raises(ValueError, match="fusion disabled"):
        model.embed(audio)
