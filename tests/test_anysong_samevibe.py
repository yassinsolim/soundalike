import math

import numpy as np
import pytest

from soundalike.anysong import api, samevibe, tempo
from soundalike.anysong.samevibe import Candidate, SameVibeParams, same_vibe, tempo_compatible

P = SameVibeParams(pool=10, tau=0.5, size=5)


def test_tempo_gate_accepts_same_double_and_half_tempo_and_unknowns():
    assert tempo_compatible(120, 124, 0.06) and tempo_compatible(120, 60.5, 0.06) and tempo_compatible(120, 236, 0.06)
    assert not tempo_compatible(120, 100, 0.06) and not tempo_compatible(120, 180, 0.06)
    assert tempo_compatible(None, 90, 0.06) and tempo_compatible(120, float("nan"), 0.06) and tempo_compatible(0, 90, 0.06)


def test_gated_best_songs_first_then_the_sounds_list_then_the_rest():
    c = {
        "b1": Candidate(1, 121, 0.2, artist_id=11),   # tempo ok, in the sounds list -> gated
        "b2": Candidate(2, 90, 0.9, artist_id=12),    # tempo off -> only in the last fill
        "b3": Candidate(3, 60, 0.6, artist_id=13),    # half tempo, sounds alike by score -> gated
        "b4": Candidate(4, 120, 0.1, artist_id=14),   # tempo ok, sound off -> last fill
        "s1": Candidate(5, 119, 0.4, artist_id=15),   # sounds list, tempo ok -> first fill
        "s2": Candidate(6, 200, 0.4, artist_id=16),   # sounds list, tempo off -> second fill
        "s3": Candidate(7, 120, 0.4, artist_id=11),   # same artist as b1 -> never
    }
    best, sonic = ["b1", "b2", "b3", "b4"], ["s3", "b1", "s1", "s2"]
    assert same_vibe(best, sonic, c, 120, params=P) == ["b1", "b3", "s1", "s2", "b2"]
    assert same_vibe(best, sonic, c, 120, params=SameVibeParams(pool=10, tau=0.5, size=6)) == \
        ["b1", "b3", "s1", "s2", "b2", "b4"]
    assert same_vibe(best, sonic, c, None, params=P)[:3] == ["b1", "b2", "b3"]  # unknown seed tempo: sound gate only
    assert same_vibe(best, sonic, c, 120, params=SameVibeParams(pool=1, tau=0.5, size=3)) == ["b1", "s1", "s2"]


def test_one_song_per_artist_unknown_artists_count_alone_and_language_filter():
    c = {"a": Candidate(1, None, 1.0, 0, "en"), "b": Candidate(2, None, 1.0, 0, "fr"), "c": Candidate(3, None, 1.0, 7),
         "d": Candidate(4, None, 1.0, 7, "en")}
    assert same_vibe(["a", "b", "c", "d"], [], c, 100, "en", "same", P) == ["a", "c"]
    assert same_vibe(["a", "b", "c", "d"], [], c, 100, "en", "all", P) == ["a", "b", "c"]
    assert same_vibe(["a", "b", "c", "d"], [], c, 100, "unknown", "same", P) == ["a", "b", "c"]


def test_tempo_of_and_table():
    assert math.isnan(tempo.tempo_of([0.0, 0.5, 1.0])[0])
    bpm, stability = tempo.tempo_of(np.arange(10) * 0.5)
    assert bpm == pytest.approx(120.0) and stability == 1.0
    table = tempo.TempoTable(np.array([5, 6, 7]), np.array([120.0, np.nan, 0.0], np.float32))
    assert table.bpm(5) == 120.0 and table.bpm(6) is None and table.bpm(7) is None and table.bpm(8) is None
    assert 5 in table and 6 not in table and len(table.by_id) == 1


def test_decode22_resamples_to_mono_and_rejects_short_clips(tmp_path):
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("soxr")
    sf.write(tmp_path / "a.wav", np.zeros((44100, 2), np.float32), 44100)
    x = tempo.decode22(tmp_path / "a.wav")
    assert x.dtype == np.float32 and abs(len(x) - 22050) <= 1
    sf.write(tmp_path / "b.wav", np.zeros(1000, np.float32), 44100)
    with pytest.raises(ValueError):
        tempo.decode22(tmp_path / "b.wav")


META = {1: ("Seed", "Seed Song"), 2: ("A", "Alpha"), 3: ("B", "Beta"), 4: ("C", "Gamma"), 5: ("D", "Delta"),
        6: ("B", "Beta"), 7: ("E", "Epsilon")}


class Index:
    def __init__(self, ids):
        self.pos = {int(t): i for i, t in enumerate(ids)}

    def __len__(self):
        return len(self.pos)


class Engine:
    device = "cpu"

    def __init__(self, raw, muq):
        self.raw, self.muq = raw, muq
        self.indexes = {ch: Index(META) for ch in ("melody", "samples", "sonic")}
        self.scored = []

    def indexed(self, tid):
        return tid in META

    def search_stored(self, tid, k):
        return {ch: [r for r in rows if r[0] != tid] for ch, rows in self.raw.items()}, {}

    def score_tracks(self, channel, track_ids, windows=None, query_track=None):
        self.scored.append((channel, sorted(track_ids), query_track))
        return {t: self.muq[t] for t in track_ids}


def test_api_builds_the_same_vibe_list_from_representative_tracks():
    # Song "B - Beta" has two releases: track 3 (melody) and track 6 (sonic); the melody track represents it.
    raw = {"melody": [(2, -0.5), (3, -0.6), (4, -0.7)], "samples": [(5, 0.4)],
           "sonic": [(6, 0.9), (7, 0.45), (4, 0.42)]}
    muq = {2: 0.1, 3: 0.2, 5: 0.6}
    table = tempo.TempoTable(np.array([1, 2, 3, 4, 5, 6, 7]),
                             np.array([120, 120, 90, 121, 60, 120, 240], np.float32))
    svc = api.AnySong(Engine(raw, muq), None, META, tempo=table, vibe_params=SameVibeParams(tau=0.5))
    r = svc.similar(1, k=10)
    assert svc.engine.scored == [("sonic", [2, 3, 5], 1)]  # sonic rows keep their own scores
    assert r["query"]["bpm"] == 120.0
    # Gated Best songs in Best order: Delta (half tempo, MuQ 0.6) and Gamma (in the sounds list); Beta's melody
    # release (90 BPM) fails the tempo gate, so the sounds list adds Epsilon (double tempo) before it.
    assert [x["id"] for x in r["results"]["vibe"]] == [5, 4, 7, 3, 2]
    assert r["results"]["vibe"][0]["score"] == round(1 / (api.FUSE_K + 1), 5) and "ranks" not in r["results"]["vibe"][0]
    assert list(r["results"])[:2] == ["best", "vibe"]
    assert svc.health()["tempo"] == {"tracks": 7, "live": False}
    plain = api.AnySong(Engine(raw, muq), None, META)
    assert "vibe" not in plain.similar(1, k=10)["results"] and "bpm" not in plain.similar(1, k=10)["query"]


class LiveDeezer:
    def __init__(self, tmp_path):
        self.tmp_path = tmp_path

    def track(self, tid):
        return {"id": tid, "title": "New", "artist": {"name": "Someone"}, "album": {}, "preview": "u"}

    def fetch_preview(self, tid, meta=None):
        path = self.tmp_path / f"{tid}.mp3"
        path.write_bytes(b"mp3")
        return path


class LiveEngine(Engine):
    def decode(self, path):
        return {16000: np.zeros(4, np.float32)}, {"decode": 1.0}

    def embed_clips(self, clips):
        return {"sonic": np.ones((3, 4))}, {}

    def search_windows(self, wins, k, exclude=()):
        return {ch: list(rows) for ch, rows in self.raw.items()}, {}


class Tracker:
    def tempo(self, clip):
        return 60.0, 1.0


def test_unseen_seed_tempo_comes_from_the_tracker(tmp_path, monkeypatch):
    raw = {"melody": [(2, -0.5)], "samples": [(5, 0.4)], "sonic": [(7, 0.45)]}
    table = tempo.TempoTable(np.array([2, 5, 7]), np.array([120, 60, 240], np.float32))
    monkeypatch.setattr(api, "decode22", lambda path: np.zeros(22050, np.float32))
    svc = api.AnySong(LiveEngine(raw, {2: 0.1, 5: 0.6}), LiveDeezer(tmp_path), META, tempo=table, tracker=Tracker())
    r = svc.similar(99, k=10)
    assert r["query"]["source"] == "live" and r["query"]["bpm"] == 60.0 and "tempo" in r["timings_ms"]
    assert [x["id"] for x in r["results"]["vibe"]] == [5, 7, 2]
    monkeypatch.setattr(api, "decode22", lambda path: (_ for _ in ()).throw(RuntimeError("bad")))
    svc = api.AnySong(LiveEngine(raw, {2: 0.1, 5: 0.6}), LiveDeezer(tmp_path), META, tempo=table, tracker=Tracker())
    assert svc.similar(98, k=10)["query"]["bpm"] is None  # an undecodable preview leaves the tempo unknown


def test_engine_scores_tracks_exactly_as_its_ranking():
    torch = pytest.importorskip("torch")
    from soundalike.anysong import serve

    rng = np.random.default_rng(5)
    ids = np.arange(100, 160, dtype=np.int64)
    win = rng.standard_normal((len(ids), 3, 16)).astype(np.float16)
    eng = serve.Engine.__new__(serve.Engine)
    eng.device, eng.channels = "cpu", [serve.CHANNELS["sonic"]]
    eng.indexes = {"sonic": serve.ChannelIndex.from_windows(serve.CHANNELS["sonic"], ids, win)}
    idx = eng.indexes["sonic"]
    full = idx.scores(idx.stored_query(100))
    got = eng.score_tracks("sonic", [105, 130, 999], query_track=100)
    assert set(got) == {105, 130}
    assert got[105] == pytest.approx(float(full[5]), abs=1e-6) and got[130] == pytest.approx(float(full[30]), abs=1e-6)
    live = eng.score_tracks("sonic", [105], windows=win[0].astype(np.float32))
    assert live[105] == pytest.approx(got[105], abs=2e-3)
    assert eng.score_tracks("sonic", [], query_track=100) == {}
    assert torch.is_tensor(idx.score_rows(idx.stored_query(100), np.array([1, 2])))
