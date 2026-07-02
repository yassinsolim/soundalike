"""Tests for vibe-target extraction and the vibe-aware training pieces."""

from __future__ import annotations

import numpy as np
import pytest

from soundalike.ml.vibe_target import (
    VIBE_TARGET_DIM,
    vibe_target_from_mel,
    vibe_targets_for_batch,
)


def _mel_bass_heavy(n_mels=128, frames=512):
    """A spectrogram with energy concentrated in the low mel bins."""
    spec = np.full((n_mels, frames), -4.0, dtype=np.float32)
    spec[:8, :] = 2.0  # strong low-frequency energy
    return spec


def _mel_bright(n_mels=128, frames=512):
    spec = np.full((n_mels, frames), -4.0, dtype=np.float32)
    spec[100:, :] = 2.0  # strong high-frequency energy
    return spec


def _mel_dynamic(n_mels=128, frames=512):
    """Quiet first half, loud second half — high dynamics."""
    spec = np.full((n_mels, frames), -3.0, dtype=np.float32)
    spec[:, frames // 2:] = 3.0
    return spec


def test_target_dim():
    t = vibe_target_from_mel(_mel_bass_heavy())
    assert t.shape == (VIBE_TARGET_DIM,) == (10,)
    assert np.isfinite(t).all()


def test_bass_vs_bright_band_emphasis():
    bass = vibe_target_from_mel(_mel_bass_heavy())
    bright = vibe_target_from_mel(_mel_bright())
    # bands are the first 7 entries (sub..air); bass-heavy has more low-band energy.
    assert bass[0] + bass[1] > bright[0] + bright[1]
    # bright has more air/presence than bass-heavy
    assert bright[5] + bright[6] > bass[5] + bass[6]
    # centroid (last entry, kHz) is higher for the bright spectrum
    assert bright[9] > bass[9]


def test_dynamics_captured():
    steady = vibe_target_from_mel(np.full((128, 512), 1.0, dtype=np.float32))
    dynamic = vibe_target_from_mel(_mel_dynamic())
    # dyn_std is index 7, dyn_range index 8
    assert dynamic[7] > steady[7]
    assert dynamic[8] > steady[8]


def test_batch_shape():
    specs = np.stack([_mel_bass_heavy(), _mel_bright(), _mel_dynamic()])
    out = vibe_targets_for_batch(specs)
    assert out.shape == (3, 10)


def test_band_fractions_sum_to_one():
    t = vibe_target_from_mel(_mel_bass_heavy())
    assert t[:7].sum() == pytest.approx(1.0, abs=1e-4)
