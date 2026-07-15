import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from soundalike.ml.fulltrack_extract import (
    ExtractionConfig,
    FullTrackExtractionError,
    PyAVAudioDecoder,
    _offline_model_environment,
    build_parser,
    extract_track,
    iter_overlapping_windows,
    length_aware_global_pool,
    select_repeated_section_indices,
    select_repeated_sections,
    select_salient_section_indices,
    select_salient_sections,
)
from soundalike.ml.jamendo_fulltrack import JamendoTrack, TrackLicense


class FakeDecoder:
    def __init__(self, samples: np.ndarray, chunk: int):
        self.samples = np.asarray(samples, dtype=np.float32)
        self.chunk = chunk

    def decode(self, path, *, sample_rate, chunk_samples):
        assert self.chunk <= chunk_samples
        for start in range(0, len(self.samples), self.chunk):
            yield self.samples[start : start + self.chunk]


class FakeEncoder:
    model_id = "fake"
    checkpoint_sha256 = hashlib.sha256(b"fake").hexdigest()
    embedding_dim = 4
    sample_rate = 10
    max_batch_size = 2

    def __init__(self):
        self.batch_sizes = []

    def embed_windows(self, windows):
        self.batch_sizes.append(len(windows))
        output = []
        for window in windows:
            value = np.asarray(
                [
                    float(np.mean(window)),
                    float(np.std(window)),
                    float(window[0]),
                    1.0,
                ],
                dtype=np.float32,
            )
            output.append(value / np.linalg.norm(value))
        return np.stack(output)


def fake_track(path: Path, payload: bytes, *, duration: float = 2.1) -> JamendoTrack:
    path.write_bytes(payload)
    relative = "00/1.mp3"
    return JamendoTrack(
        row_index=0,
        track_id=1,
        artist_id=2,
        album_id=3,
        relative_path=relative,
        audio_path=path,
        duration_seconds=duration,
        tags=("genre---test",),
        title="Synthetic",
        artist_name="Fixture",
        album_name="Fixture",
        release_date="2026-01-01",
        jamendo_url="http://www.jamendo.com/track/1",
        license=TrackLicense(
            path=relative,
            attribution="fixture",
            name="CC BY-NC-SA",
            url="http://creativecommons.org/licenses/by-nc-sa/3.0/",
            permits_commercial_use=False,
            permits_derivatives=True,
        ),
        expected_audio_sha256=hashlib.sha256(payload).hexdigest(),
        expected_audio_bytes=len(payload),
    )


def test_streaming_overlap_and_end_aligned_tail_are_exact():
    signal = np.arange(21, dtype=np.float32)
    chunks = (signal[:7], signal[7:14], signal[14:])
    windows = list(
        iter_overlapping_windows(
            chunks,
            window_samples=10,
            hop_samples=5,
            max_chunk_samples=7,
        )
    )
    assert [window.start_sample for window in windows] == [0, 5, 10, 11]
    np.testing.assert_array_equal(windows[-1].samples, signal[11:21])


def test_short_track_repeatpad_and_zero_pad_policies():
    repeated = list(
        iter_overlapping_windows(
            [np.asarray([1, 2, 3], dtype=np.float32)],
            window_samples=8,
            hop_samples=4,
            short_track_policy="repeatpad",
        )
    )[0].samples
    np.testing.assert_array_equal(repeated, [1, 2, 3, 1, 2, 3, 1, 2])
    zero = list(
        iter_overlapping_windows(
            [np.asarray([1, 2, 3], dtype=np.float32)],
            window_samples=8,
            hop_samples=4,
            short_track_policy="zero_pad",
        )
    )[0].samples
    np.testing.assert_array_equal(zero, [1, 2, 3, 0, 0, 0, 0, 0])
    with pytest.raises(FullTrackExtractionError, match="shorter"):
        list(
            iter_overlapping_windows(
                [np.asarray([1, 2, 3], dtype=np.float32)],
                window_samples=8,
                hop_samples=4,
                short_track_policy="reject",
            )
        )


def test_pooling_and_section_selection_are_deterministic():
    windows = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
    pooled = length_aware_global_pool(
        windows, [0, 5], decoded_samples=15, window_samples=10
    )
    np.testing.assert_allclose(pooled, [2**-0.5, 2**-0.5], atol=1e-6)

    values = np.asarray(
        [[1, 0], [0.9, 0.1], [0, 1], [1, 0], [0, 1]], dtype=np.float32
    )
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    repeat_indices = select_repeated_section_indices(
        values, budget=3, minimum_gap=2
    )
    salient_indices = select_salient_section_indices(
        values, pooled, budget=3, minimum_gap=2
    )
    assert len(set(repeat_indices.tolist())) == 3
    assert len(set(salient_indices.tolist())) == 3
    first_repeat = select_repeated_sections(values, budget=3, minimum_gap=2)
    first_salient = select_salient_sections(
        values, pooled, budget=3, minimum_gap=2
    )
    np.testing.assert_array_equal(first_repeat, values[repeat_indices])
    np.testing.assert_array_equal(first_salient, values[salient_indices])
    np.testing.assert_array_equal(
        first_repeat,
        select_repeated_sections(values, budget=3, minimum_gap=2),
    )
    np.testing.assert_array_equal(
        first_salient,
        select_salient_sections(values, pooled, budget=3, minimum_gap=2),
    )


def test_section_rankings_are_nested_for_8_16_32_budgets():
    rng = np.random.default_rng(20260715)
    values = rng.normal(size=(40, 8)).astype(np.float32)
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    global_embedding = np.mean(values, axis=0)
    global_embedding /= np.linalg.norm(global_embedding)

    repeated_32 = select_repeated_section_indices(values, budget=32)
    salient_32 = select_salient_section_indices(
        values, global_embedding, budget=32
    )
    np.testing.assert_array_equal(
        select_repeated_section_indices(values, budget=8), repeated_32[:8]
    )
    np.testing.assert_array_equal(
        select_repeated_section_indices(values, budget=16), repeated_32[:16]
    )
    np.testing.assert_array_equal(
        select_salient_section_indices(values, global_embedding, budget=8),
        salient_32[:8],
    )
    np.testing.assert_array_equal(
        select_salient_section_indices(values, global_embedding, budget=16),
        salient_32[:16],
    )


def test_production_section_defaults_and_cli_are_32():
    config = ExtractionConfig()
    assert config.repetition_sections == 32
    assert config.salient_sections == 32
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
    assert args.repetition_sections == 32
    assert args.salient_sections == 32


def test_extract_track_bounds_model_batches_and_creates_no_wav(tmp_path):
    payload = b"not-decoded-by-fake"
    track = fake_track(tmp_path / "fixture.mp3", payload)
    encoder = FakeEncoder()
    config = ExtractionConfig(
        sample_rate=10,
        window_seconds=1.0,
        hop_seconds=0.5,
        decoder_chunk_seconds=0.6,
        model_batch_size=2,
        max_windows_per_track=10,
        repetition_sections=2,
        salient_sections=2,
        section_min_gap_windows=1,
    )
    result = extract_track(
        track,
        decoder=FakeDecoder(np.linspace(-1, 1, 21), chunk=6),
        encoder=encoder,
        config=config,
    )
    assert result.window_starts.tolist() == [0, 5, 10, 11]
    assert max(encoder.batch_sizes) <= 2
    assert result.decoded_samples == 21
    assert len(np.unique(result.repeated_indices)) == len(result.repeated_indices)
    assert len(np.unique(result.salient_indices)) == len(result.salient_indices)
    np.testing.assert_array_equal(
        result.repeated_sections, result.window_embeddings[result.repeated_indices]
    )
    np.testing.assert_array_equal(
        result.salient_sections, result.window_embeddings[result.salient_indices]
    )
    assert not list(tmp_path.glob("*.wav"))


def test_decoder_and_window_resource_bounds_fail_closed():
    with pytest.raises(FullTrackExtractionError, match="chunk exceeds"):
        list(
            iter_overlapping_windows(
                [np.ones(11, dtype=np.float32)],
                window_samples=10,
                hop_samples=5,
                max_chunk_samples=10,
            )
        )
    with pytest.raises(FullTrackExtractionError, match="max_windows"):
        list(
            iter_overlapping_windows(
                [np.ones(30, dtype=np.float32)],
                window_samples=10,
                hop_samples=5,
                max_windows=2,
            )
        )


def test_model_environment_forces_offline_and_restores(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "previous")
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    with _offline_model_environment():
        assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["HF_HUB_OFFLINE"] == "previous"
    assert "TRANSFORMERS_OFFLINE" not in os.environ


def test_pinned_pyav_decodes_synthetic_mp3_without_wav(tmp_path):
    av = pytest.importorskip("av")
    path = tmp_path / "tone.mp3"
    container = av.open(str(path), mode="w")
    stream = container.add_stream("mp3", rate=48_000)
    stream.layout = "mono"
    samples = (
        0.1
        * np.sin(2 * np.pi * 440 * np.arange(12_000, dtype=np.float32) / 48_000)
    ).reshape(1, -1)
    frame = av.AudioFrame.from_ndarray(samples, format="fltp", layout="mono")
    frame.sample_rate = 48_000
    for packet in stream.encode(frame):
        container.mux(packet)
    for packet in stream.encode(None):
        container.mux(packet)
    container.close()

    chunks = list(
        PyAVAudioDecoder().decode(path, sample_rate=48_000, chunk_samples=4_000)
    )
    assert sum(map(len, chunks)) > 0
    assert max(map(len, chunks)) <= 4_000
    assert not list(tmp_path.glob("*.wav"))
