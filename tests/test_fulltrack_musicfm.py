import hashlib
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
import torch

from soundalike.ml.fulltrack_extract import ExtractionConfig, FullTrackExtractionError
from soundalike.ml.fulltrack_musicfm import (
    CHECKPOINT_BYTES,
    CHECKPOINT_SHA256,
    CONFORMER_CONFIG_SHA256,
    EMBEDDING_DIM,
    HOP_SECONDS,
    MODEL_ID,
    SAMPLE_RATE,
    SOURCE_COMMIT,
    SOURCE_LICENSE_SHA256,
    STATS_SHA256,
    WINDOW_SECONDS,
    FrozenMusicFMAdapter,
    MusicFMCapability,
    _extraction_binding,
    _musicfm_source_import,
    _pool_hidden_states,
    build_parser,
    inspect_capability,
)


class _FakeTensor:
    def cuda(self):
        return self


class _FakeTorch:
    @staticmethod
    def from_numpy(_values):
        return _FakeTensor()

    @staticmethod
    def inference_mode():
        return nullcontext()


class _FakeModel:
    def __init__(self, output):
        self.output = output
        self.layer_index = None

    def get_latent(self, _tensor, *, layer_ix):
        self.layer_index = layer_ix
        return self.output


def _adapter_with_output(output) -> FrozenMusicFMAdapter:
    adapter = object.__new__(FrozenMusicFMAdapter)
    adapter._torch = _FakeTorch()
    adapter._model = _FakeModel(output)
    return adapter


def _capability() -> MusicFMCapability:
    return MusicFMCapability(
        available=True,
        model_id=MODEL_ID,
        asset_root="assets",
        source_commit=SOURCE_COMMIT,
        checkpoint_sha256=CHECKPOINT_SHA256,
        checkpoint_bytes=CHECKPOINT_BYTES,
        stats_sha256=STATS_SHA256,
        conformer_config_sha256=CONFORMER_CONFIG_SHA256,
        source_license_sha256=SOURCE_LICENSE_SHA256,
        package_versions={"torch": "test"},
        cuda_device="test",
        license="MIT",
        reasons=(),
    )


def test_musicfm_cli_uses_approved_window_and_bounded_batch():
    args = build_parser().parse_args(
        [
            "extract",
            "--metadata-root",
            "metadata",
            "--audio-root",
            "audio",
            "--state-root",
            "state",
            "--output",
            "store",
        ]
    )
    assert args.batch_size == 1
    assert args.shard_tracks == 64
    assert SAMPLE_RATE == 24_000
    assert WINDOW_SECONDS == 30.0
    assert HOP_SECONDS == 15.0
    assert args.track_plan is None


def test_musicfm_source_cleanup_does_not_mask_model_error(tmp_path):
    source_root = tmp_path / "musicfm"
    parent = str(tmp_path)
    assert parent not in sys.path
    with pytest.raises(RuntimeError, match="model failed"):
        with _musicfm_source_import(source_root):
            sys.path.remove(parent)
            raise RuntimeError("model failed")


def test_musicfm_hidden_state_pooling_is_time_mean_and_normalized():
    hidden = torch.zeros((2, 3, EMBEDDING_DIM), dtype=torch.float32)
    hidden[0, :, 0] = 1.0
    hidden[1, :, 1] = 2.0
    pooled = _pool_hidden_states(hidden)
    assert pooled.shape == (2, EMBEDDING_DIM)
    np.testing.assert_allclose(np.linalg.norm(pooled, axis=1), 1.0)
    np.testing.assert_array_equal(np.argmax(pooled, axis=1), [0, 1])
    with pytest.raises(FullTrackExtractionError, match="unexpected"):
        _pool_hidden_states(torch.zeros((2, EMBEDDING_DIM)))


def test_musicfm_adapter_embeds_exact_windows_at_the_pinned_layer():
    adapter = _adapter_with_output(torch.ones((2, 3, EMBEDDING_DIM)))
    windows = np.zeros((2, SAMPLE_RATE * int(WINDOW_SECONDS)), dtype=np.float32)
    embedded = adapter.embed_windows(windows)
    assert embedded.shape == (2, EMBEDDING_DIM)
    np.testing.assert_allclose(np.linalg.norm(embedded, axis=1), 1.0)
    assert adapter._model.layer_index == 7


@pytest.mark.parametrize(
    "windows, message",
    [
        (
            np.zeros((1, SAMPLE_RATE), dtype=np.float32),
            "exact 30-second",
        ),
        (
            np.zeros((3, SAMPLE_RATE * int(WINDOW_SECONDS)), dtype=np.float32),
            "batch exceeds",
        ),
    ],
)
def test_musicfm_adapter_rejects_invalid_window_batches(windows, message):
    adapter = _adapter_with_output(torch.ones((1, 3, EMBEDDING_DIM)))
    with pytest.raises(FullTrackExtractionError, match=message):
        adapter.embed_windows(windows)


def test_musicfm_binding_covers_every_model_asset_and_blocks_promotion():
    config = ExtractionConfig(
        sample_rate=SAMPLE_RATE,
        window_seconds=WINDOW_SECONDS,
        hop_seconds=HOP_SECONDS,
    )
    binding = _extraction_binding(config, _capability())
    assert binding["promotion_allowed"] is False
    assert binding["model"]["source_commit"] == SOURCE_COMMIT
    assert binding["model"]["checkpoint_sha256"] == CHECKPOINT_SHA256
    assert binding["model"]["stats_sha256"] == STATS_SHA256
    assert binding["model"]["conformer_config_sha256"] == CONFORMER_CONFIG_SHA256
    assert binding["model"]["source_license_sha256"] == SOURCE_LICENSE_SHA256


def test_musicfm_binding_includes_frozen_track_protocol():
    config = ExtractionConfig(
        sample_rate=SAMPLE_RATE,
        window_seconds=WINDOW_SECONDS,
        hop_seconds=HOP_SECONDS,
    )
    protocol = {
        "artifact_kind": "v3_artist_disjoint_semantic_head_scale_protocol",
        "payload_sha256": "1" * 64,
        "selection_sha256": "2" * 64,
        "track_limit": 8192,
    }
    binding = _extraction_binding(
        config, _capability(), track_protocol=protocol
    )
    assert binding["track_protocol"] == protocol
    assert binding["promotion_allowed"] is False


def test_capability_fails_closed_without_assets(tmp_path):
    capability = inspect_capability(tmp_path)
    assert capability.available is False
    assert capability.checkpoint_sha256 is None
    assert capability.source_commit is None
    assert any("checkpoint" in reason for reason in capability.reasons)
    assert any("source checkout" in reason for reason in capability.reasons)
