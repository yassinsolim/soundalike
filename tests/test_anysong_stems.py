import base64
import json

import numpy as np
import pytest
import soundfile as sf

from soundalike.anysong import stems as S

PERIOD = 0.613  # 97.9 BPM; Beat This! reports beats on 20 ms frames


def _tone(midi: float, seconds: float = S.SECONDS, sr: int = S.SR, amp: float = 0.2) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    f = 440.0 * 2 ** ((midi - 69) / 12)
    return amp * (np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t))


def _beats(period: float = PERIOD) -> tuple[np.ndarray, np.ndarray]:
    beats = np.round(np.arange(0.1, S.SECONDS - 0.2, period) / 0.02) * 0.02
    return beats, beats[::4]


def _drums(beats: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(1)
    y = np.zeros(S.SAMPLES)
    n = int(0.08 * S.SR)
    env = np.exp(-np.arange(n) / (0.015 * S.SR))
    kick = 0.8 * np.sin(2 * np.pi * 60 * np.arange(n) / S.SR) * env
    hat = 0.2 * rng.standard_normal(n) * env
    for i, b in enumerate(beats):
        i0 = int(b * S.SR)
        y[i0:i0 + n] += (kick if i % 2 == 0 else hat)[:len(y[i0:i0 + n])]
    return y


def _song(vocal: float = 57.0, bass: float = 33.0, chord=(57, 60, 64)) -> tuple[np.ndarray, np.ndarray]:
    """[6, 2, T] stems (no guitar or piano) and their mix."""
    beats, _ = _beats()
    mono = np.zeros((len(S.STEMS), S.SAMPLES))
    mono[S.VOCALS] = _tone(vocal)
    mono[S.DRUMS] = _drums(beats)
    mono[S.BASS] = _tone(bass, amp=0.3)
    mono[5] = sum(_tone(m, amp=0.08) for m in chord)
    stems = np.repeat(mono[:, None], 2, axis=1).astype(np.float32)
    return stems, stems.sum(0)


@pytest.fixture(scope="module")
def record():
    stems, mix = _song()
    return S.describe(stems, mix, *_beats()) | {"vec": np.ones((6, 512), np.float16)}


def test_describe_columns_and_measurements(record):
    cols = S.stack([record])
    for name, (shape, dtype) in S.COLUMNS.items():
        assert cols[name].shape == (1, *shape) and cols[name].dtype == dtype, name
    assert record["present"].tolist() == [True, True, True, False, False, True]
    d = record["desc"]
    # the beat grid is quantised to 20 ms; the fitted period recovers the tempo between the grid's steps
    assert d[S.D["bpm"]] == pytest.approx(60 / PERIOD, abs=0.3)
    assert d[S.D["vocal_median"]] == pytest.approx(57.0, abs=0.15)
    assert d[S.D["bass_median"]] == pytest.approx(33.0, abs=0.15)
    assert d[S.D["vocal_tight"]] > 0.9
    assert S.key_distance(d[S.D["key_tonic"]], d[S.D["key_minor"]], 9, 1) == 0  # A minor or its relative major
    beats, downbeats = _beats()
    stored = record["beats"][record["beats"] != S.NO_BEAT]
    assert len(stored) == len(beats) and record["downbeat"].sum() == len(downbeats)
    assert d[S.D["kick_rate"]] > 0 and d[S.D["drum_bars"]] > 0


def test_detuned_and_gliding_vocals():
    beats, downbeats = _beats()
    stems, mix = _song(vocal=57.4)
    d = S.describe(stems, mix, beats, downbeats)["desc"]
    assert d[S.D["vocal_median"]] == pytest.approx(57.4, abs=0.15)  # cents survive pYIN's semitone bins
    assert d[S.D["vocal_tight"]] > 0.9  # in tune with itself
    t = np.arange(S.SAMPLES) / S.SR
    f = 440.0 * 2 ** ((57 + 5 * t / S.SECONDS - 69) / 12)
    stems[S.VOCALS] = 0.2 * np.sin(2 * np.pi * np.cumsum(f) / S.SR)[None]
    d = S.describe(stems, stems.sum(0), beats, downbeats)["desc"]
    assert d[S.D["vocal_tight"]] < 0.7  # a slow glide sits between notes most of the time


def _rand_record(rng, present=None):
    rec = {name: np.zeros(shape, dtype) for name, (shape, dtype) in S.COLUMNS.items()}
    rec["vec"] = rng.standard_normal((6, 512)).astype(np.float16)
    rec["present"] = np.ones(6, bool) if present is None else np.asarray(present, bool)
    rec["balance"] = rng.uniform(-30, 0, 6).astype(np.float32)
    desc = rng.uniform(1, 60, len(S.DESC)).astype(np.float32)
    desc[S.D["key_tonic"]], desc[S.D["key_minor"]], desc[S.D["drum_bars"]] = rng.integers(12), rng.integers(2), 4
    rec["desc"] = desc
    rec["pattern"] = rng.integers(0, 256, (3, 16)).astype(np.uint8)
    rec["chroma"] = rng.dirichlet(np.ones(12)).astype(np.float16)
    rec["intervals"] = rng.dirichlet(np.ones(12), 2).astype(np.float16)
    return rec


def test_pair_features_symmetric_and_nan_where_not_applicable():
    rng = np.random.default_rng(0)
    a = _rand_record(rng)
    b = _rand_record(rng, present=[False, True, True, False, True, True])
    fab, fba = S.pair_features(a, b), S.pair_features(b, a)
    np.testing.assert_allclose(fab, fba, rtol=1e-5, equal_nan=True)
    for name in ("vocals", "guitar", "vocal_pitch", "vocal_range"):
        assert np.isnan(fab[S.F[name]]), name
    same = S.pair_features(a, a)
    np.testing.assert_allclose(same[:6], 1.0, atol=1e-3)
    assert same[S.F["tempo"]] == pytest.approx(1.0) and same[S.F["key"]] == 0
    for name in S.DISTANCES:
        assert same[S.F[name]] == pytest.approx(0.0, abs=1e-6), name
    mu = rng.standard_normal((6, 512)).astype(np.float32)
    assert not np.allclose(S.pair_features(a, b, mu)[:6], fab[:6], equal_nan=True)


def test_tempo_key_and_groove_helpers():
    assert S.tempo_sim(100, 200) == pytest.approx(1.0) and S.tempo_sim(140, 70) == pytest.approx(1.0)
    assert S.tempo_sim(100, 104) < 0.5 and np.isnan(S.tempo_sim(np.nan, 100))
    assert S.key_distance(0, 0, 9, 1) == 0  # C major, A minor
    assert S.key_distance(0, 0, 7, 0) == 1 and S.key_distance(0, 0, 6, 0) == 6
    P = np.random.default_rng(2).integers(0, 256, (3, 16)).astype(np.uint8)
    assert S.groove_sim(P, np.roll(P, 4, axis=1)) == pytest.approx(1.0)
    assert S.groove_sim(P, np.roll(P, 2, axis=1)) < 1.0


def test_grade_flips_distances_and_keeps_nan():
    calib = np.tile(np.linspace(0, 1, 101), (len(S.FACETS), 1))
    f = np.full(len(S.FACETS), 0.9, np.float32)
    f[S.F["tempo"]] = np.nan
    g = S.grade(f, calib)
    assert g[S.F["vocals"]] == pytest.approx(90) and g[S.F["key"]] == pytest.approx(10)
    assert np.isnan(g[S.F["tempo"]])


def test_pack_table_payload_and_compare_round_trip(tmp_path, record):
    rng = np.random.default_rng(3)
    other = _rand_record(rng)
    for i, (ids, recs) in enumerate((([7, 5], [record, other]), ([5], [other]))):
        np.savez(tmp_path / f"s{i}.npz", ids=np.array(ids, np.int64), **S.stack(recs))
    calib = np.tile(np.linspace(-1, 1, 101), (len(S.FACETS), 1))
    info = S.pack(sorted(tmp_path.glob("s*.npz")), tmp_path / "table", calib=calib)
    assert info["songs"] == 2
    table = S.StemTable.load(tmp_path / "table")
    assert table.ids.tolist() == [5, 7] and 7 in table and 9 not in table and table.row(9) is None
    assert table.mu.shape == (6, 512) and table.calib.shape == (len(S.FACETS), 101)
    np.testing.assert_array_equal(table.row(7)["lanes"], record["lanes"])
    p = json.loads(json.dumps(S.payload(7, table.row(7))))
    assert len(base64.b64decode(p["stems"][0]["loudness"])) == S.LANE_FRAMES
    assert p["key"]["mode"] in ("major", "minor") and len(p["downbeats"]) == record["downbeat"].sum()
    assert [s["present"] for s in p["stems"]] == record["present"].tolist()
    c = S.compare(table.row(7), table.row(7), table.mu, table.calib)
    assert c["matched_on"] and all(x["grade"] is not None for x in c["facets"])
    grades = {x["name"]: x["grade"] for x in c["facets"]}
    assert c["matched_on"] == sorted(c["matched_on"], key=lambda n: -grades[n])[:3]
    (tmp_path / "table" / "meta.json").write_text(json.dumps({"version": S.VERSION + 1}), encoding="utf-8")
    with pytest.raises(ValueError):
        S.StemTable.load(tmp_path / "table")


def test_table_finds_rows_when_ids_are_not_sorted():
    rng = np.random.default_rng(4)
    recs = [_rand_record(rng) for _ in range(3)]
    table = S.StemTable(np.array([30, 10, 20]), S.stack(recs), {})
    assert [t in table for t in (10, 20, 30, 15, 99, 0)] == [True, True, True, False, False, False]
    np.testing.assert_array_equal(table.row(10)["vec"], recs[1]["vec"])
    assert table.mu is None and table.calib is None


def test_store_reads_the_table_then_its_cache_and_keeps_live_rows_across_restarts(tmp_path):
    rng = np.random.default_rng(5)
    stored, live = _rand_record(rng), _rand_record(rng)
    table = S.StemTable(np.array([7]), S.stack([stored]), {"mu": np.zeros((6, 512)).tolist()})
    store = S.StemStore(table, tmp_path / "cache", size=1)
    assert store.get(7)[1] == "catalog" and store.mu.shape == (6, 512) and store.calib is None
    assert store.get(1234) == (None, None)
    store.put(1234, live)
    assert (tmp_path / "cache" / "234" / "1234.npz").exists() and not list((tmp_path / "cache").rglob("*.tmp*"))
    row, source = store.get(1234)
    assert source == "cache" and all(row[k].dtype == dt for k, (_, dt) in S.COLUMNS.items())
    store.put(99, live)  # evicts 1234 from memory; the disk copy remains
    again = S.StemStore(None, tmp_path / "cache")
    np.testing.assert_array_equal(again.get(1234)[0]["vec"], live["vec"])
    (tmp_path / "cache" / "234" / "1234.npz").write_bytes(b"torn")
    assert S.StemStore(None, tmp_path / "cache").get(1234) == (None, None)  # a broken file is a miss


class _FakeSeparator:
    def __call__(self, mixes):
        return np.repeat(mixes[:, None], 6, axis=1) / 6


class _FakeTracker:
    def track(self, clips):
        return [_beats() for _ in clips]


class _FakeEmbedder:
    sr = 24_000

    def embed(self, clips):
        return {"track": np.tile(np.abs(clips).mean(1, keepdims=True), (1, 512))}


def test_build_pipelines_batches_and_skips_bad_previews(tmp_path):
    ids = [1001, 2002, 3003, 4004]
    for i, t in enumerate(ids[:3]):
        folder = tmp_path / f"{t % 1000:03d}"
        folder.mkdir(exist_ok=True)
        y = _tone(50 + i, seconds=29.5, sr=48_000)  # short clips are repeat-padded, other rates resampled
        sf.write(folder / f"{t}.mp3", np.stack([y, y], 1), 48_000, format="WAV")
    logs = []
    dis = S.Dissector(_FakeSeparator(), _FakeTracker(), _FakeEmbedder(), workers=2)
    info = S.build(ids, tmp_path, dis, tmp_path / "out" / "shard.npz", batch=2, workers=2, log=logs.append)
    assert info["songs"] == 3 and any("decode failed 4004" in m for m in logs)
    with np.load(tmp_path / "out" / "shard.npz") as z:
        assert z["ids"].tolist() == ids[:3]
        assert z["vec"].shape == (3, 6, 512) and z["lanes"].shape == (3, 7, S.LANE_FRAMES)
    assert not list((tmp_path / "out").glob("*.tmp*"))
    direct = dis(np.stack([S.decode_stereo(tmp_path / "001" / "1001.mp3")]))
    np.testing.assert_array_equal(direct[0]["desc"], np.load(tmp_path / "out" / "shard.npz")["desc"][0])
