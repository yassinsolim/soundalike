import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("librosa")
pytest.importorskip("soxr")
from soundalike.anysong import audio, robustness as rb

SR = 22050


def tone(seconds=4.0, freq=440.0):
    t = np.arange(int(seconds * SR)) / SR
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def peak_hz(x):
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return np.fft.rfftfreq(len(x), 1 / SR)[spec.argmax()]


def test_speed_changes_length_and_pitch_together():
    y = rb.speed(1.25)(tone(), SR, None)
    assert len(y) == pytest.approx(len(tone()) / 1.25, rel=0.01)
    assert peak_hz(y) == pytest.approx(440 * 1.25, rel=0.01)


def test_pitch_keeps_length_and_tempo_keeps_pitch():
    rng = np.random.default_rng(0)
    up = rb.pitch(2)(tone(), SR, rng)
    assert len(up) == len(tone()) and peak_hz(up) == pytest.approx(440 * 2 ** (2 / 12), rel=0.01)
    fast = rb.tempo(1.1)(tone(), SR, rng)
    assert len(fast) == pytest.approx(len(tone()) / 1.1, rel=0.01)
    assert peak_hz(fast) == pytest.approx(440, rel=0.01)


def test_noise_hits_the_requested_snr_and_phone_band_limits():
    x = tone()
    noisy = rb.noise(10)(x, SR, np.random.default_rng(1))
    snr = 10 * np.log10(np.mean(x ** 2) / np.mean((noisy - x) ** 2))
    assert snr == pytest.approx(10, abs=0.2)
    high = rb.phone(tone(freq=6000), SR, None)
    assert np.sqrt(np.mean(high ** 2)) < 0.01
    assert len(rb.phone(x, SR, None)) == pytest.approx(len(x), abs=4)


def test_excerpt_is_repeat_padded_like_short_previews():
    x = np.arange(SR * 30, dtype=np.float32) / (SR * 30)
    clip = rb.excerpt(10, 10)(x, SR, None)
    assert len(clip) == 10 * SR and clip[0] == x[10 * SR]
    fitted = audio.fit_clips(clip, SR, [SR])[SR]
    assert len(fitted) == 30 * SR and np.array_equal(fitted[10 * SR:20 * SR], clip)


def test_excerpt_past_the_end_of_a_short_preview_is_a_clear_error():
    clip = rb.excerpt(10, 10)(tone(seconds=1.0), SR, None)
    assert len(clip) == 0
    with pytest.raises(ValueError, match="empty audio"):
        audio.fit_clips(clip, SR, [SR])


def test_summarize_counts_misses_as_not_found():
    m = rb.summarize([(1, 1), (3, 2), (0, 0), (12, 12)])
    assert m["n"] == 4 and m["hr@1"] == 0.25 and m["hr@10"] == 0.5 and m["self_hr@1"] == 0.25
    assert m["mrr"] == pytest.approx((1 + 1 / 3 + 0 + 1 / 12) / 4, abs=1e-4)
    assert m["median_rank"] == 7.5
