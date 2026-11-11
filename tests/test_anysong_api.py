import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from soundalike.anysong import api
from soundalike.anysong import stems as stemlib
from soundalike.anysong.languages import LanguageTable, decide, top_languages
from soundalike.anysong.scene import SceneParams
from soundalike.anysong.scene import build as build_scene
from soundalike.anysong.tempo import TempoTable

META = {
    1: ("Daft Punk", "One More Time"),
    2: ("Daft Punk", "One More Time (Radio Edit)"),
    3: ("Some Choir", "One More Time"),
    4: ("Daft Punk feat. Romanthony", "One More Time - 2021 Remaster"),
    5: ("Other Artist", "Different Song"),
    6: ("Other Artist", "Different Song [Live]"),
    7: ("Third", "Third Song"),
}


class FakeIndex:
    def __init__(self, ids):
        self.pos = {int(t): i for i, t in enumerate(ids)}

    def __len__(self):
        return len(self.pos)


class FakeEngine:
    device = "cpu"

    def __init__(self, ids=(1, 2, 3, 4, 5, 6, 7), rows=None, gate=None):
        self.indexes = {"melody": FakeIndex(ids), "samples": FakeIndex(ids)}
        self.rows = rows or [(2, 0.99), (3, 0.9), (4, 0.8), (5, 0.7), (6, 0.6), (7, 0.5)]
        self.gate = gate
        self.calls = []
        self.decoded = []

    def indexed(self, tid):
        return all(tid in ix.pos for ix in self.indexes.values())

    def _ranked(self, exclude):
        if self.gate is not None:
            self.gate.wait(5)
        return {ch: [r for r in self.rows if r[0] not in exclude] for ch in self.indexes}, {"search_melody": 1.0}

    def search_stored(self, tid, k):
        self.calls.append(("stored", tid, k))
        return self._ranked({tid})

    def decode(self, path):
        self.decoded.append(path)
        if path.read_bytes() == b"bad":
            raise RuntimeError("corrupt mp3")
        return {16000: np.zeros(4, np.float32)}, {"decode": 2.0}

    def embed_clips(self, clips):
        self.calls.append(("embed",))
        return {"melody": np.zeros((3, 4))}, {"embed_clews_dvi": 3.0}

    def search_windows(self, wins, k, exclude=()):
        self.calls.append(("windows", tuple(exclude), k))
        return self._ranked(set(exclude))


class FakeDeezer:
    def __init__(self, tmp_path, tracks=None, preview=b"mp3"):
        self.tmp_path, self.preview = tmp_path, preview
        self.tracks = tracks if tracks is not None else {
            99: {"id": 99, "title": "One More Time (Live)", "artist": {"name": "Daft Punk"},
                 "album": {"title": "Alive", "cover_medium": "http://c/99.jpg"}, "duration": 30, "preview": "u"},
        }
        self.fail = False
        self.searches = 0
        self.lookups = 0
        self.related = {}
        self.related_calls = []

    def track(self, tid):
        self.lookups += 1
        if self.fail:
            raise RuntimeError("down")
        return self.tracks.get(tid, {})

    def related_artists(self, artist_id):
        self.related_calls.append(artist_id)
        if self.fail:
            raise RuntimeError("down")
        return self.related.get(artist_id, [])

    def fetch_preview(self, tid, meta=None):
        if not (meta or {}).get("preview"):
            return None
        path = self.tmp_path / f"{tid}.mp3"
        path.write_bytes(self.preview)
        return path

    def search(self, q, limit):
        self.searches += 1
        if self.fail:
            raise RuntimeError("down")
        return [self.tracks[99], {"id": 5, "title": "Different Song", "artist": {"name": "Other Artist"},
                                  "album": {}, "preview": ""}, {"no": "id"}][:limit]


@pytest.fixture
def svc(tmp_path):
    return api.AnySong(FakeEngine(), FakeDeezer(tmp_path), META)


class FakeLanguages:
    def __init__(self, track_languages, artist_languages=None):
        self.track_languages = track_languages
        self.artist_languages = artist_languages or {}

    def language(self, track_id):
        return self.track_languages.get(track_id, "unknown")

    def artist_language(self, artist):
        return self.artist_languages.get(artist.casefold(), "unknown") if artist else "unknown"


def test_language_decision_and_table_lookup(tmp_path):
    probs = np.asarray([[0.90, 0.06, 0.04], [0.74, 0.20, 0.06]], dtype=np.float32)
    indices, values = top_languages(probs)
    assert indices.tolist() == [[0, 1, 2], [0, 1, 2]]
    assert decide(indices, values).tolist() == [0, -1]

    path = tmp_path / "languages.npz"
    np.savez(path, ids=np.asarray([7, 1, 9]), vocab=np.asarray(["en", "fr"]),
             top_idx=np.asarray([[0, 1], [0, 1], [1, 0]], dtype=np.uint8),
             top_prob=np.asarray([[0.95, 0.04], [0.96, 0.03], [0.94, 0.05]], dtype=np.float32),
             meta=np.asarray("{}"))
    table = LanguageTable.load(path, artist_names={1: "Artist A", 7: "Artist A", 9: "Artist B"})
    assert table.languages([1, 7, 9, 100]) == ["en", "en", "fr", "unknown"]
    assert table.artist_language("artist a") == "en"
    assert table.artist_language("Artist B") == "unknown"
    assert table.labelled(9) and not table.labelled(100)


@pytest.mark.parametrize("artist,title,expected", [
    ("Daft Punk", "One More Time (Radio Edit)", ("daft punk", "one more time")),
    ("Daft Punk feat. Romanthony", "One More Time - 2021 Remaster", ("daft punk", "one more time")),
    ("Lil Freaky", "Dripset (feat. Future)", ("lil freaky", "dripset")),
    ("Crosby, Stills, Nash & Young", "Looking Forward", ("crosby", "looking forward")),
    ("A", "(Intro)", ("a", "intro")),
])
def test_song_key_drops_version_tags(artist, title, expected):
    assert api.song_key(artist, title) == expected


def test_catalog_track_uses_stored_windows_and_hides_same_song_versions(svc):
    r = svc.similar(1, k=10)
    assert svc.engine.calls == [("stored", 1, api.FETCH)]
    assert r["query"] == {"id": 1, "artist": "Daft Punk", "title": "One More Time", "source": "catalog",
                          "language": "unknown", "language_mode": "same", "hidden": 0, "scene": "off"}
    # 2 and 4 are other releases of the query; 6 is a live version of 5. The cover by another artist stays.
    assert [x["id"] for x in r["results"]["melody"]] == [3, 5, 7]
    assert r["results"]["melody"][0] == {"id": 3, "score": 0.9, "artist": "Some Choir", "title": "One More Time"}
    assert r["cached"] is False and "total" in r["timings_ms"] and "queue" in r["timings_ms"]


def test_dedupe_can_be_disabled_and_k_truncates(svc):
    assert [x["id"] for x in svc.similar(1, k=3, dedupe=False)["results"]["samples"]] == [2, 3, 4]


def test_results_are_cached_across_k_and_dedupe(svc):
    svc.similar(1, k=2)
    r = svc.similar(1, k=5, dedupe=False)
    assert len(svc.engine.calls) == 1
    assert r["cached"] is True and len(r["results"]["melody"]) == 5
    assert svc.health()["served"] == {"catalog": 1, "live": 0, "cached": 1}


def test_fuse_sums_weighted_reciprocal_ranks():
    fused = api.fuse({"melody": ["a", "b", "c"], "samples": ["c", "d"], "sonic": ["e"]},
                     {"melody": 1.0, "samples": 1.0}, k=1)
    # c: 1/4 + 1/2; a: 1/2; b and d tie at 1/3 with the same best rank, so first appearance wins
    assert [key for key, _, _ in fused] == ["c", "a", "b", "d"]
    assert fused[0][1] == pytest.approx(0.75) and fused[0][2] == {"melody": 3, "samples": 1}
    assert api.fuse({"melody": ["a"], "samples": ["b"]}, {"melody": 1.0, "samples": 2.0}, k=1)[0][0] == "b"
    assert api.fuse({"sonic": ["e"]}) == []


def test_sound_match_places_a_sonic_score_among_the_songs_shown_for_typical_seeds():
    q = api.MATCH_QUANTILES
    assert len(q) == 21 and list(q) == sorted(q)
    assert api.sound_match(q[10]) == 50 and api.sound_match(q[19]) == 95
    assert api.sound_match(q[-1] + 1.0) == 100 and api.sound_match(-1.0) == 0
    assert api.sound_match((q[2] + q[3]) / 2) in (12, 13)
    values = [api.sound_match(s) for s in np.linspace(-0.6, 0.7, 200)]
    assert values == sorted(values) and all(isinstance(v, int) for v in values)


def test_best_list_fuses_the_song_views(svc):
    res = {"query": {"id": 1, "artist": "Daft Punk", "title": "One More Time", "source": "catalog"},
           "raw": {"melody": [(2, 0.99), (3, 0.9), (5, 0.8), (7, 0.7)],
                   "samples": [(7, 0.5), (6, 0.4), (3, 0.3)]},
           "timings_ms": {}}
    r = svc._view(res, k=10, dedupe=True, cached=False)
    assert list(r["results"]) == ["best", "melody", "samples"]
    best = r["results"]["best"]
    # 3 and 7 tie (1st and 3rd in each list) and beat 2nd + 2nd for any k; 6 is a live version of 5, so song 5
    # is 2nd in both lists
    assert [x["id"] for x in best] == [3, 7, 5]
    k = api.FUSE_K
    assert best[1] == {"id": 7, "score": round(1 / (k + 1) + 1 / (k + 3), 5), "artist": "Third",
                       "title": "Third Song", "ranks": {"melody": 3, "samples": 1}}
    assert best[2]["ranks"] == {"melody": 2, "samples": 2} and best[2]["title"] == "Different Song"
    assert [x["id"] for x in svc._view(res, k=2, dedupe=True, cached=False)["results"]["best"]] == [3, 7]
    assert [x["id"] for x in r["results"]["samples"]] == [7, 6, 3]


def test_language_gate_filters_only_best_and_keeps_unknown(svc):
    svc.languages = FakeLanguages({1: "en", 2: "fr", 3: "en", 5: "unknown", 6: "unknown", 7: "de"})
    res = {"query": {"id": 1, "artist": "Daft Punk", "title": "One More Time", "source": "catalog"},
           "raw": {"melody": [(2, 0.99), (3, 0.9), (5, 0.8), (7, 0.7)],
                   "samples": [(7, 0.5), (6, 0.4), (3, 0.3)]},
           "timings_ms": {}}

    filtered = svc._view(res, k=3, dedupe=True, cached=False)
    assert filtered["query"]["language"] == "en"
    assert filtered["query"]["language_mode"] == "same"
    assert filtered["query"]["hidden"] == 1
    assert [item["id"] for item in filtered["results"]["best"]] == [3, 6]
    assert [item["id"] for item in filtered["results"]["melody"]] == [3, 5, 7]
    assert [item["id"] for item in filtered["results"]["samples"]] == [7, 6, 3]

    all_languages = svc._view(res, k=3, dedupe=True, cached=False, lang_mode="all")
    assert [item["id"] for item in all_languages["results"]["best"]] == [3, 7, 5]
    assert all_languages["query"]["hidden"] == 1


def test_unknown_seed_language_fails_open(svc):
    svc.languages = FakeLanguages({1: "unknown", 3: "en", 7: "fr"})
    res = {"query": {"id": 1, "artist": "Unknown Artist", "title": "Seed", "source": "catalog"},
           "raw": {"melody": [(3, 0.9), (7, 0.8)]}, "timings_ms": {}}
    result = svc._view(res, k=10, dedupe=True, cached=False)
    assert result["query"]["language"] == "unknown"
    assert result["query"]["hidden"] == 0
    assert [item["id"] for item in result["results"]["best"]] == [3, 7]


def test_artist_language_fallback_filters_unseen_seed(svc):
    svc.languages = FakeLanguages({1: "unknown", 3: "en", 7: "fr"}, {"daft punk": "en"})
    res = {"query": {"id": 1, "artist": "Daft Punk", "title": "Seed", "source": "live"},
           "raw": {"melody": [(3, 0.9), (7, 0.8)]}, "timings_ms": {}}
    result = svc._view(res, k=10, dedupe=True, cached=False)
    assert result["query"]["language"] == "en"
    assert [item["id"] for item in result["results"]["best"]] == [3]


def test_artist_languages_hide_unlabelled_songs_by_artists_of_another_language(svc):
    svc.languages = FakeLanguages({1: "en", 3: "en"}, {"other artist": "fr", "third": "en"})
    res = {"query": {"id": 1, "artist": "Daft Punk", "title": "One More Time", "source": "catalog"},
           "raw": {"melody": [(3, 0.9), (5, 0.8), (7, 0.7)]}, "timings_ms": {}}
    svc.artist_languages = False
    assert [x["id"] for x in svc._view(res, k=10, dedupe=True, cached=False)["results"]["best"]] == [3, 5, 7]
    svc.artist_languages = True  # the default: 5 is by Other Artist, who sings French; Third (7) sings English
    view = svc._view(res, k=10, dedupe=True, cached=False)
    assert [x["id"] for x in view["results"]["best"]] == [3, 7] and view["query"]["hidden"] == 1


def test_other_versions_of_the_seed_song_are_limited(tmp_path):
    meta = {1: ("Brent Faiyaz", "Paper Soldier"), 2: ("d3puu", "Paper Soldier"), 3: ("Choir A", "Paper Soldier"),
            4: ("Choir B", "Paper Soldier (Live)"), 5: ("Choir C", "Paper Soldier"), 6: ("Cassie", "Addiction")}
    svc = api.AnySong(FakeEngine(ids=tuple(meta)), FakeDeezer(tmp_path), meta)
    res = {"query": {"id": 1, "artist": "Brent Faiyaz", "title": "Paper Soldier", "source": "catalog"},
           "raw": {"melody": [(2, 0.99), (3, 0.9), (6, 0.85), (4, 0.8), (5, 0.7)],
                   "samples": [(2, 0.9), (6, 0.5), (5, 0.4)]}, "timings_ms": {}}

    def best():
        return [x["id"] for x in svc._view(res, k=10, dedupe=True, cached=False)["results"]["best"]]

    svc.copy_score = svc.max_versions = None
    assert best() == [2, 6, 5, 3, 4]
    svc.copy_score = api.COPY_SAMPLES_SCORE  # 2 shares the seed's audio (samples 0.9): a copy of its recording
    assert best() == [6, 5, 3, 4]
    svc.max_versions = 2  # of the other versions, the two best placed in any channel stay
    assert best() == [6, 5, 3]
    view = svc._view(res, k=10, dedupe=True, cached=False)
    assert [x["id"] for x in view["results"]["melody"]] == [2, 3, 6, 4, 5]  # the channel lists stay whole
    svc.copy_score, svc.max_versions = None, 0
    assert best() == [6]


def test_sounds_boost_lifts_the_seed_artist_scene(tmp_path):
    svc = api.AnySong(FakeEngine(), FakeDeezer(tmp_path), META, scene=_scene_table())
    res = {"query": {"id": 1, "artist": "Daft Punk", "title": "One More Time", "source": "catalog", "artist_id": 100},
           "raw": {"sonic": [(5, 0.9), (3, 0.85), (7, 0.8)]}, "related": [], "timings_ms": {}}

    def sounds(**kw):
        return [x["id"] for x in svc._view(res, k=10, dedupe=True, cached=False, **kw)["results"]["sounds"]]

    svc.sounds_boost = 0.0
    assert sounds() == [5, 3, 7]
    svc.sounds_boost = 4.0  # Third (7) shares Daft Punk's related artists; Other Artist (5) is a US act
    assert sounds() == [7, 5, 3]
    assert sounds(scene=False) == [5, 3, 7]
    svc.sounds_boost = api.SOUNDS_BOOST  # the default
    assert sounds() == [7, 5, 3]


def test_no_best_list_without_fused_channels(svc):
    res = {"query": {"id": 1, "artist": "A", "title": "B", "source": "catalog"}, "raw": {"sonic": [(3, 0.9)]},
           "timings_ms": {}}
    assert list(svc._view(res, k=5, dedupe=True, cached=False)["results"]) == ["sounds", "sonic"]


def test_sounds_list_filters_language_keeps_one_song_per_artist_and_adds_tempo(tmp_path):
    meta = META | {8: ("Other Artist & Friend", "Another Song")}
    svc = api.AnySong(FakeEngine(), FakeDeezer(tmp_path), meta,
                      languages=FakeLanguages({1: "en", 3: "fr", 5: "en", 8: "en", 7: "unknown"}),
                      tempo=TempoTable(np.array([5, 7]), np.array([128.04, np.nan], np.float32)))
    res = {"query": {"id": 1, "artist": "Daft Punk", "title": "One More Time", "source": "catalog"},
           "raw": {"sonic": [(2, 0.99), (3, 0.9), (5, 0.8), (6, 0.75), (8, 0.7), (7, 0.6)]}, "timings_ms": {}}

    def sounds(**kw):
        lists = svc._view(res, dedupe=True, cached=False, scene=False, **kw)["results"]
        return [(x["id"], x["bpm"]) for x in lists["sounds"]]

    # 2 is the seed's radio edit and 6 a live take of 5, so both are the same songs; Some Choir (3) sings French.
    assert sounds(k=10) == [(5, 128.0), (8, None), (7, None)]
    assert sounds(k=10, lang_mode="all") == [(3, None), (5, 128.0), (8, None), (7, None)]
    assert sounds(k=2) == [(5, 128.0), (8, None)]
    # With the scene table, one Deezer artist id covers differently credited tracks.
    svc.scene = build_scene({3: 400, 5: 200, 7: 300, 8: 200}, {}, {}, {})
    assert sounds(k=10) == [(5, 128.0), (7, None)]


def _scene_table():
    # Daft Punk (100) and Third (300) list each other; Other Artist (200) is a US act; Some Choir (400) has no data.
    tracks = {1: 100, 2: 100, 3: 400, 4: 100, 5: 200, 6: 200, 7: 300}
    related = {100: [300, 500], 300: [100, 500], 200: [600, 601]}
    return build_scene(tracks, related, {100: ["FR"], 300: ["FR"], 500: ["FR"], 200: ["US"]}, {})


LISTENER_META = META | {8: ("Third", "Third Song Two"), 9: ("Fifth", "Fünfter Song"), 10: ("Daft Punk", "Around")}


def _listener_scene_table():
    # As _scene_table, plus a second Third song (8), a song by Fifth (500, related to both) and another Daft Punk song.
    tracks = {1: 100, 2: 100, 3: 400, 4: 100, 5: 200, 6: 200, 7: 300, 8: 300, 9: 500, 10: 100}
    related = {100: [300, 500], 300: [100, 500], 200: [600, 601]}
    return build_scene(tracks, related, {100: ["FR"], 300: ["FR"], 500: ["FR"], 200: ["US"]}, {})


def _stump(feature, left, right):
    from soundalike.anysong.listener import FEATURES, Ranker

    return Ranker(FEATURES, 0.0, np.array([0]), np.array([FEATURES.index(feature), 0, 0]), np.array([0.5, 0, 0]),
                  np.array([True, True, True]), np.array([1, 1, 2]), np.array([2, 1, 2]),
                  np.array([0.0, left, right]), depth=1, meta="stump")


def test_listener_lists_keep_the_audio_head_and_rank_the_rest_for_listeners(tmp_path):
    # best_head_standout=0: any melody score over the calibrated threshold extends the head (the standout rule is
    # checked at the end); one song is kept without evidence.
    svc = api.AnySong(FakeEngine(), FakeDeezer(tmp_path), LISTENER_META, scene=_listener_scene_table(),
                      copy_score=None, ranker=_stump("fwd", 0.0, 0.0), scene_params=SceneParams(head_min=1),
                      languages=FakeLanguages({1: "en", 9: "de"}), best_head_standout=0.0, best_head_bare=1,
                      best_head_evidence_only=False,
                      tempo=TempoTable(np.array([5, 7, 8, 10]), np.array([120, 120, 121, 120], np.float32)))
    p = {3: 0.2, 5: 0.1, 7: 0.9, 8: 0.8, 9: 0.95, 10: 0.99}
    res = {"query": {"id": 1, "artist": "Daft Punk", "title": "One More Time", "source": "catalog", "artist_id": 100},
           "raw": {"melody": [(5, -1.0), (3, -2.0)], "samples": [(3, 0.1)],
                   "sonic": [(10, 0.5), (5, 0.45), (7, 0.44)]},
           "related": [], "timings_ms": {}, "vibe": {"muq": {10: 0.5, 5: 0.45, 7: 0.44, 3: 0.2}, "bpm": 120.0},
           "listener": {"p": p, "son": {3: 0.2, 5: 0.45, 7: 0.44, 8: 0.40, 9: 0.43, 10: 0.5}, "scene": [8, 9]}}

    def lists(**kw):
        return svc._view(res, k=10, dedupe=True, cached=False, **kw)["results"]

    r = lists()
    # Head: the audio fusion down to the cover evidence (5's melody score), then the listener order; Fifth's German
    # song (9) is left out, and the seed artist's own song (10) is placed by sound alone.
    assert [x["id"] for x in r["best"]] == [3, 5, 7, 10, 8]
    assert [x["ranks"] for x in r["best"]] == [{"melody": 2, "samples": 1}, {"melody": 1, "sonic": 2}, {"sonic": 3},
                                               {"sonic": 1}, {}]
    assert [x["id"] for x in r["sounds"]] == [7, 10, 5]
    assert {x["id"] for x in r["vibe"]} <= {3, 5, 7, 8, 10}
    # The shipped head holds only the songs with evidence: 3 is above 5 in the fusion but is no likely cover or sample,
    # so it is placed for listeners.
    svc.best_head_evidence_only = api.BEST_HEAD_EVIDENCE_ONLY
    assert [x["id"] for x in lists()["best"]] == [5, 7, 10, 3, 8]
    svc.best_head_evidence_only = False
    p[10] = 0.01  # the seed artist's songs ignore their listener probability
    assert [x["id"] for x in lists()["best"]] == [3, 5, 7, 10, 8] and [x["id"] for x in lists()["sounds"]] == [7, 10, 5]
    assert 9 in [x["id"] for x in lists(lang_mode="all")["best"]]
    assert [x["id"] for x in lists(scene=False)["best"]] == [3, 5]  # no scene: the plain fusion, no ranker
    # The shipped standout rule: 5's melody score does not stand out in a two-song list, so it no longer extends the
    # head and is placed like the other candidates.
    svc.best_head_standout = api.BEST_HEAD_STANDOUT
    assert [x["id"] for x in lists()["best"]] == [3, 10, 7, 5, 8]
    # Every shown song carries its sound match: its sonic score against the seed on the calibrated 0..100 scale.
    son = res["listener"]["son"]
    for name in ("best", "vibe", "sounds"):
        assert [x["match"] for x in r[name]] == [api.sound_match(son[x["id"]]) for x in r[name]]
    assert r["sounds"][1]["match"] > r["sounds"][0]["match"]  # 10 (0.5) sounds more alike than 7 (0.44)
    svc.best_head_bare = 0  # no evidence, no head: Best is the listener order from its first place
    assert [x["id"] for x in lists()["best"]][0] != 3
    svc.best_head_bare = 1
    del res["listener"]
    assert [x["id"] for x in lists()["best"]][:2] == [3, 5] and 8 not in [x["id"] for x in lists()["best"]]


class SonicEngine(FakeEngine):
    def __init__(self):
        super().__init__(ids=tuple(range(1, 11)))
        self.indexes["sonic"] = FakeIndex(range(1, 11))
        self.scored = []

    def score_tracks(self, channel, track_ids, windows=None, query_track=None):
        tids = sorted(int(t) for t in track_ids)
        self.scored.append((channel, tids, query_track))
        return {t: 0.3 + t / 100 for t in tids if t in self.indexes[channel].pos}


def test_listener_inputs_add_scene_songs_once_per_seed(tmp_path):
    engine = SonicEngine()
    svc = api.AnySong(engine, FakeDeezer(tmp_path), LISTENER_META, scene=_listener_scene_table(), copy_score=None,
                      ranker=_stump("fwd", -1.0, 1.0))
    res = {"query": {"id": 1, "artist": "Daft Punk", "title": "One More Time", "artist_id": 100},
           "raw": {"melody": [(5, -1.0)], "sonic": [(7, 0.44)]}, "related": [], "vibe": None}
    timings = {}
    out = svc._listener(res, None, timings)
    # Pool songs without a sound score, then the neighbours' songs (Third: 7, 8; Fifth: 9), from the stored windows.
    assert engine.scored == [("sonic", [5, 7], 1), ("sonic", [7, 8, 9], 1)]
    assert out["scene"] == [7, 8, 9] and set(out["p"]) == {5, 7, 8, 9} and out["son"][9] == pytest.approx(0.39)
    assert out["p"][7] > out["p"][5]  # Daft Punk's related list names Third, not Other Artist
    assert "listener" in timings
    assert svc._listener({**res, "query": {"id": 1, "artist": "Daft Punk"}}, None, {}) is None  # no artist id

    r = svc.similar(1, k=10)
    assert {8, 9} <= {x["id"] for x in r["results"]["best"]}
    assert next(x for x in r["results"]["best"] if x["id"] == 9)["ranks"] == {}
    assert svc.health()["ranker"] == "stump"
    assert api.AnySong(FakeEngine(), FakeDeezer(tmp_path), META, ranker=_stump("fwd", 0, 1)).ranker is None


def test_scene_layer_lifts_related_artists_and_can_be_turned_off(tmp_path):
    # copy_score=None: the fake engine scores every row 0.9 on samples, which would read as copies of the seed
    svc = api.AnySong(FakeEngine(), FakeDeezer(tmp_path), META, scene=_scene_table(), copy_score=None,
                      scene_params=SceneParams(pattern=""))  # the audio continuation alone, without slots
    r = svc.similar(1, k=10)
    assert r["query"]["scene"] == "on" and r["query"]["artist_id"] == 100
    assert [x["id"] for x in r["results"]["best"]] == [7, 3, 5]  # audio order is 3, 5, 7
    assert [x["id"] for x in svc.similar(1, k=10, scene=False)["results"]["best"]] == [3, 5, 7]
    assert svc.similar(1, k=10, scene=False)["query"]["scene"] == "off"
    assert svc.deezer.related_calls == []  # the table has Daft Punk's own related list
    assert svc.health()["scene"] == {"tracks": 7, "artists": 7}
    svc.scene_params = SceneParams()  # shipped slots: a three-song audio head stays on top
    assert [x["id"] for x in svc.similar(1, k=10)["results"]["best"]] == [3, 5, 7]


def test_scene_layer_adds_sonic_matches_only_for_the_artist_scene(tmp_path):
    svc = api.AnySong(FakeEngine(), FakeDeezer(tmp_path), META, scene=_scene_table(),
                      scene_params=SceneParams(sonic=1.0, pattern=""))
    res = {"query": {"id": 1, "artist": "Daft Punk", "title": "One More Time", "source": "catalog", "artist_id": 100},
           "raw": {"melody": [(3, 0.9), (5, 0.8)], "samples": [(3, 0.5)], "sonic": [(5, 0.7), (7, 0.6)]},
           "related": [], "timings_ms": {}}
    best = svc._view(res, k=10, dedupe=True, cached=False)["results"]["best"]
    assert [x["id"] for x in best] == [3, 5, 7]
    assert best[2]["ranks"] == {"sonic": 2}  # Third's song is only in the sonic list; Other Artist adds nothing
    svc.scene_params = SceneParams(pattern="")  # no sonic term and no scene lane: audio songs only
    assert [x["id"] for x in svc._view(res, k=10, dedupe=True, cached=False)["results"]["best"]] == [3, 5]
    svc.scene_params = SceneParams()  # shipped: the scene lane admits the sonic-only scene neighbour
    best = svc._view(res, k=10, dedupe=True, cached=False)["results"]["best"]
    assert [x["id"] for x in best] == [3, 5, 7] and best[2]["ranks"] == {"sonic": 2}


def test_unseen_artist_uses_live_related_artists_once_and_fails_open(tmp_path):
    deezer = FakeDeezer(tmp_path)
    deezer.tracks[98] = {"id": 98, "title": "New Song", "artist": {"id": 999, "name": "New Artist"},
                         "album": {}, "duration": 30, "preview": "u"}
    deezer.related[999] = [300, 500]
    svc = api.AnySong(FakeEngine(), deezer, META, scene=_scene_table(), scene_params=SceneParams(pattern=""))
    r = svc.similar(98, k=10)
    assert r["query"]["artist_id"] == 999 and r["query"]["scene"] == "on" and "related" in r["timings_ms"]
    # New Artist's live related list (Third, 500) makes Daft Punk and Third scene neighbours; audio: 2, 3, 5, 7
    assert [x["id"] for x in r["results"]["best"]] == [2, 7, 3, 5]
    assert svc._related_jobs == {}  # the background lookup was consumed
    svc.results = api.LRU(8)
    svc.similar(98, k=10)
    assert deezer.related_calls == [999]  # cached per artist

    down = FakeDeezer(tmp_path)
    down.tracks[98] = deezer.tracks[98]
    svc = api.AnySong(FakeEngine(), down, META, scene=_scene_table())
    svc.known.put(98, down.tracks[98])  # the track lookup succeeds; only the related-artist call fails
    down.fail = True
    r = svc.similar(98, k=10)
    assert r["query"]["scene"] == "unknown_artist" and r["query"]["artist_id"] == 999
    assert [x["id"] for x in r["results"]["best"]] == [2, 3, 5, 7]


def test_unseen_track_is_fetched_embedded_and_excluded(svc, tmp_path):
    r = svc.similar(99, k=10)
    assert r["query"]["source"] == "live" and r["query"]["cover"] == "http://c/99.jpg"
    assert svc.engine.calls == [("embed",), ("windows", (99,), api.FETCH)]
    assert not (tmp_path / "99.mp3").exists()  # scratch preview removed after decoding
    assert [x["id"] for x in r["results"]["melody"]] == [3, 5, 7]  # query is a Daft Punk version too
    assert {"lookup", "fetch", "decode", "embed_clews_dvi", "search_melody", "total"} <= set(r["timings_ms"])
    assert svc.deezer.lookups == 1


def test_search_results_skip_the_track_lookup(svc):
    svc.search("one more time")
    r = svc.similar(99)
    assert r["query"]["source"] == "live" and svc.deezer.lookups == 0 and "lookup" not in r["timings_ms"]


def test_known_tracks_expire(tmp_path):
    now = [0.0]
    svc = api.AnySong(FakeEngine(), FakeDeezer(tmp_path), META, known_ttl=60)
    svc.known.clock = lambda: now[0]
    svc.search("one more time")
    now[0] = 61
    svc.similar(99)
    assert svc.deezer.lookups == 1


def test_keep_previews_option(tmp_path):
    svc = api.AnySong(FakeEngine(), FakeDeezer(tmp_path), META, keep_previews=True)
    svc.similar(99)
    assert (tmp_path / "99.mp3").exists()


def test_learn_queue_records_live_songs_once_without_request_details(tmp_path):
    deezer = FakeDeezer(tmp_path)
    deezer.tracks[98] = {"id": 98, "title": "New Song", "artist": {"id": 555, "name": "New Artist"},
                         "album": {}, "preview": "u"}
    queue = api.LearnQueue(tmp_path / "learn" / "queue.jsonl", clock=lambda: 1_794_000_000)
    (tmp_path / "learn").mkdir()
    svc = api.AnySong(FakeEngine(), deezer, META, learn=queue)
    svc.similar(1)  # catalog song: nothing to learn
    svc.similar(98)
    svc.results = api.LRU(16)  # a later request computes the same song again
    svc.similar(98)
    svc.similar(99)
    with pytest.raises(api.ApiError):
        svc.similar(42)  # unknown track: not answered, not queued
    lines = [json.loads(line) for line in (tmp_path / "learn" / "queue.jsonl").read_text().splitlines()]
    assert lines == [{"id": 98, "artist_id": 555, "day": "2026-11-06"}, {"id": 99, "artist_id": None, "day": "2026-11-06"}]


def test_learn_queue_caps_each_day_and_never_fails_a_request(tmp_path):
    now = [1_794_000_000.0]
    queue = api.LearnQueue(tmp_path / "queue.jsonl", per_day=2, clock=lambda: now[0])
    assert [queue.add(t, 1) for t in (1, 2, 3)] == [True, True, False]
    now[0] += 86_400
    assert queue.add(3, 1) and not queue.add(3, 1)
    assert [json.loads(x)["id"] for x in (tmp_path / "queue.jsonl").read_text().splitlines()] == [1, 2, 3]
    broken = api.LearnQueue(tmp_path / "missing-dir" / "queue.jsonl")
    assert broken.add(7, 1) is False
    svc = api.AnySong(FakeEngine(), FakeDeezer(tmp_path), META, learn=broken)
    assert svc.similar(99)["query"]["source"] == "live"


@pytest.mark.parametrize("setup,status", [
    (lambda s: None, 404),  # unknown track id
    (lambda s: s.deezer.tracks.__setitem__(42, {"id": 42, "title": "x", "preview": ""}), 404),
    (lambda s: setattr(s.deezer, "fail", True), 502),
    (lambda s: (s.deezer.tracks.__setitem__(42, {"id": 42, "title": "x", "preview": "u"}),
                setattr(s.deezer, "preview", b"bad")), 422),
])
def test_live_errors(svc, setup, status):
    setup(svc)
    with pytest.raises(api.ApiError) as err:
        svc.similar(42)
    assert err.value.status == status
    assert svc.results.get(42) is None and not svc._flight


def test_busy_when_pending_slots_are_full(tmp_path):
    svc = api.AnySong(FakeEngine(), FakeDeezer(tmp_path), META, max_pending=1)
    svc.pending.acquire()
    with pytest.raises(api.ApiError) as err:
        svc.similar(1)
    assert err.value.status == 503
    svc.pending.release()
    assert svc.similar(1)["ok"]


def test_concurrent_requests_for_one_track_compute_once(tmp_path):
    gate = threading.Event()
    svc = api.AnySong(FakeEngine(gate=gate), FakeDeezer(tmp_path), META)
    out = []
    threads = [threading.Thread(target=lambda: out.append(svc.similar(1, k=3))) for _ in range(4)]
    for t in threads:
        t.start()
    for _ in range(100):
        if svc._flight:
            break
        threading.Event().wait(0.01)
    gate.set()
    for t in threads:
        t.join(5)
    assert len(out) == 4 and len(svc.engine.calls) == 1
    assert sorted(r["cached"] for r in out) == [False, True, True, True]


def test_search_cards_flag_catalog_tracks_and_cache(svc):
    r = svc.search("  one   more time ", limit=5)
    assert r["query"] == "one more time"
    assert r["tracks"] == [
        {"id": 99, "title": "One More Time (Live)", "artist": "Daft Punk", "album": "Alive",
         "cover": "http://c/99.jpg", "duration": 30, "has_preview": True, "in_catalog": False},
        {"id": 5, "title": "Different Song", "artist": "Other Artist", "album": "", "cover": "",
         "duration": None, "has_preview": False, "in_catalog": True},
    ]
    svc.search("ONE MORE TIME", limit=5)
    assert svc.deezer.searches == 1


@pytest.mark.parametrize("q", ["", "   ", "x" * 201])
def test_search_rejects_bad_queries(svc, q):
    with pytest.raises(api.ApiError) as err:
        svc.search(q)
    assert err.value.status == 400


def test_search_upstream_failure(svc):
    svc.deezer.fail = True
    with pytest.raises(api.ApiError) as err:
        svc.search("abc")
    assert err.value.status == 502


NEW_SONG = {99: {"id": 99, "title": "Brand New Song (Live)", "artist": {"name": "New Artist"},
                 "album": {"title": "Fresh", "cover_medium": "http://c/99.jpg"}, "duration": 30, "preview": "u"}}
ORIGINAL = {99: {"id": 99, "title": "Mutations", "artist": {"name": "Nil\u00fcfer Yanya"}, "album": {}, "duration": 30,
                 "preview": "u"}}


def test_resolve_finds_the_requested_version_in_the_catalog_without_deezer(svc):
    calls = []
    assert svc.resolve("Daft Punk", "One More Time - Radio Edit", lambda: calls.append(1)) == 2
    assert svc.resolve("  daft  PUNK feat. Someone ", "One More Time - Remastered") == 1  # same recording
    assert svc.resolve("Daft Punk", "One More Time (feat. Romanthony)") == 1
    assert svc.resolve("Some Choir", "One More Time") == 3
    assert svc.resolve("Other Artist", "Different Song - Live") == 6
    assert calls == [] and svc.deezer.searches == 0


def test_resolve_prefers_indexed_releases_then_untagged_titles(tmp_path):
    svc = api.AnySong(FakeEngine(ids=(2, 3, 4, 5, 6, 7)), FakeDeezer(tmp_path), META)
    assert svc.resolve("Daft Punk", "One More Time") == 4  # 1 has metadata but no windows; 2 is a radio edit
    meta = {1: ("A", "Song (Unplugged)"), 2: ("A", "Song - Remastered"), 3: ("A", "Song"), 4: ("A feat. B", "Song")}
    assert api.AnySong(FakeEngine(), FakeDeezer(tmp_path), meta).resolve("A", "Song - Remastered 2009") == 3
    # "Song" is the remastered recording, not the unplugged performance (``_content_tags``)
    assert api.AnySong(FakeEngine(ids=(1, 2)), FakeDeezer(tmp_path), meta).resolve("A", "Song") == 2


def test_resolve_leaves_releases_without_windows_to_the_deezer_search(tmp_path):
    meta = {1: ("A", "Song"), 2: ("A", "Song (Club Remix)")}
    tracks = {99: {"id": 99, "title": "Song", "artist": {"name": "A"}, "album": {}, "duration": 30, "preview": "u"}}
    assert api.AnySong(FakeEngine(ids=(2,)), FakeDeezer(tmp_path, tracks=tracks), meta).resolve("A", "Song") == 99
    assert api.AnySong(FakeEngine(ids=(2,)), FakeDeezer(tmp_path), meta).resolve("A", "Song") == 2
    with pytest.raises(api.ApiError) as err:
        api.AnySong(FakeEngine(ids=()), FakeDeezer(tmp_path), meta).resolve("A", "Song")
    assert err.value.status == 404


def test_resolve_searches_deezer_for_a_version_the_catalog_lacks(tmp_path):
    svc = api.AnySong(FakeEngine(), FakeDeezer(tmp_path, tracks=ORIGINAL),
                      {7: ("Nil\u00fcfer Yanya", "Mutations (Empress Of Remix)")})
    calls = []
    assert svc.resolve("Nil\u00fcfer Yanya", "Mutations", lambda: calls.append(1)) == 99  # the original, not the remix
    assert svc.resolve("Nil\u00fcfer Yanya", "Mutations - Empress Of Remix", lambda: calls.append(1)) == 7
    assert calls == [1] and svc.deezer.searches == 1


def test_resolve_settles_for_another_version_of_the_song(tmp_path):
    svc = api.AnySong(FakeEngine(), FakeDeezer(tmp_path), META)  # Deezer's only hit is a live "One More Time"
    assert svc.resolve("Daft Punk", "One More Time - Live") == 99
    assert svc.resolve("Daft Punk", "One More Time - Acoustic") == 1  # neither has it: the catalog's best release

    def limited():
        raise api.ApiError(429, "too many requests")

    assert svc.resolve("Daft Punk", "One More Time (Live at Wembley)", limited) == 1
    svc.deezer.fail = True
    assert svc.resolve("Daft Punk", "One More Time (Live in Paris)") == 1
    assert svc.deezer.searches == 3


def test_resolve_matches_accents_in_either_unicode_form(tmp_path):
    svc = api.AnySong(FakeEngine(), FakeDeezer(tmp_path), {7: ("Rose\u0301", "APT.")})
    assert svc.resolve("ROS\u00c9", "APT.") == 7 and svc.deezer.searches == 0


def test_resolve_falls_back_to_a_deezer_search_hit_of_the_same_song(tmp_path):
    svc = api.AnySong(FakeEngine(), FakeDeezer(tmp_path, tracks=NEW_SONG), META)
    calls = []
    assert svc.resolve("New Artist feat. Guest", "Brand New Song", lambda: calls.append(1)) == 99
    assert calls == [1] and svc.deezer.searches == 1
    assert svc.similar(99)["query"]["source"] == "live" and svc.deezer.lookups == 0


@pytest.mark.parametrize("artist,title", [
    ("Daft Punk", "Unknown Song"),  # the hit is another song
    ("Other Artist", "Different Song"),  # the hit has no preview
])
def test_resolve_rejects_search_hits_of_other_songs_or_without_previews(tmp_path, artist, title):
    svc = api.AnySong(FakeEngine(), FakeDeezer(tmp_path), {})
    with pytest.raises(api.ApiError) as err:
        svc.resolve(artist, title)
    assert err.value.status == 404 and svc.deezer.searches == 1


@pytest.mark.parametrize("artist,title", [("", "x"), ("x", "   "), ("x" * 201, "y"), ("x", "y" * 201)])
def test_resolve_validates_names_before_any_lookup(svc, artist, title):
    with pytest.raises(api.ApiError) as err:
        svc.resolve(artist, title, lambda: pytest.fail("rate limit consulted"))
    assert err.value.status == 400 and svc.deezer.searches == 0


def test_resolve_search_limit_and_upstream_failure(tmp_path):
    svc = api.AnySong(FakeEngine(), FakeDeezer(tmp_path, tracks=NEW_SONG), META)

    def limited():
        raise api.ApiError(429, "too many requests")

    with pytest.raises(api.ApiError) as err:
        svc.resolve("New Artist", "Brand New Song", limited)
    assert err.value.status == 429 and svc.deezer.searches == 0
    svc.deezer.fail = True
    with pytest.raises(api.ApiError) as err:
        svc.resolve("New Artist", "Brand New Song")
    assert err.value.status == 502


@pytest.mark.parametrize("a,b", [
    (("The Black Eyed Peas", "The Time (Dirty Bit)"), ("Black Eyed Peas", "The Time (Dirty Bit)")),
    (("Beyonc\u00e9", "D\u00e9j\u00e0 Vu"), ("Beyonce", "Deja Vu")),
    (("\u271d\u271d\u271d (Crosses)", "Bitches Brew"), ("\u2020\u2020\u2020 (Crosses)", "Bitches Brew")),
    (("The Jackson 5", "ABC"), ("Jackson 5", "ABC")),
])
def test_lookup_key_ignores_credit_spellings(a, b):
    assert api._lookup_key(*a) == api._lookup_key(*b)


def test_lookup_key_keeps_marks_outside_latin_and_lone_words():
    assert api._lookup_key("A", "\u304b\u304d") != api._lookup_key("A", "\u304c\u304d")  # kaki and gaki differ
    assert api._lookup_key("The The", "Uncertain Smile")[0] == "the"
    assert api._lookup_key("The", "x")[0] == "the"


@pytest.mark.parametrize("title,tags", [
    ("Canon in D (Santa Clause is Coming to Town)", {"santa clause is coming to town"}),
    ("Song (feat. X) [Live] - 2011 Remaster", set()),
    ("The Shadow Of Your Smile (Love Theme From \"The Sandpiper\")", set()),
    ("Song (Original Motion Picture Soundtrack)", set()),
    ("Song (Prod. by X)", set()),
    ("YUKON (INTERLUDE)", {"interlude"}),
    ("Song", set()),
])
def test_content_tags_name_other_content_not_versions_or_credits(title, tags):
    assert api._content_tags(title) == tags


def test_content_tags_read_full_width_brackets():
    assert len(api._content_tags("\u60f3\u3044\uff08TV\u30a2\u30cb\u30e1\u633f\u5165\u6b4c\uff09")) == 1


def test_resolve_prefers_the_plain_song_over_a_catalog_medley(tmp_path):
    meta = {7: ("Johann Pachelbel", "Canon in D (Santa Clause is Coming to Town)")}
    plain = {99: {"id": 99, "title": "Canon in D", "artist": {"name": "Johann Pachelbel"}, "album": {},
                  "duration": 30, "preview": "u"}}
    svc = api.AnySong(FakeEngine(), FakeDeezer(tmp_path, tracks=plain), meta)
    assert svc.resolve("Johann Pachelbel", "Canon in D") == 99
    assert svc.resolve("Johann Pachelbel", "Canon in D (Santa Clause is Coming to Town)") == 7  # asked for it
    assert svc.deezer.searches == 1
    other = api.AnySong(FakeEngine(), FakeDeezer(tmp_path), meta)  # Deezer only finds other songs
    assert other.resolve("Johann Pachelbel", "Canon in D") == 7  # the catalog's medley stands in


def test_resolve_finds_catalog_songs_under_another_credit_spelling(tmp_path):
    meta = {1: ("The Black Eyed Peas", "The Time (Dirty Bit)"), 2: ("Beyonce", "Deja Vu")}
    svc = api.AnySong(FakeEngine(ids=(1, 2)), FakeDeezer(tmp_path), meta)
    assert svc.resolve("Black Eyed Peas", "The Time (Dirty Bit)") == 1
    assert svc.resolve("Beyonc\u00e9", "D\u00e9j\u00e0 Vu (feat. Jay-Z)") == 2
    assert svc.deezer.searches == 0


class QueryDeezer(FakeDeezer):
    """Search results that depend on the query text, plus Deezer's ISRC lookup."""

    def __init__(self, tmp_path, by_query=None, by_isrc=None, tracks=None):
        super().__init__(tmp_path, tracks=tracks)
        self.by_query, self.by_isrc = by_query, by_isrc or {}
        self.queries, self.isrc_lookups = [], 0

    def search(self, q, limit):
        if self.by_query is None:
            return super().search(q, limit)
        self.searches += 1
        self.queries.append(q)
        if self.fail:
            raise RuntimeError("down")
        return self.by_query.get(q, [])[:limit]

    def track_by_isrc(self, isrc):
        self.isrc_lookups += 1
        if self.fail:
            raise RuntimeError("down")
        return self.by_isrc.get(isrc, {})


CROSSES = {"id": 42, "title": "Bitches Brew", "artist": {"name": "\u2020\u2020\u2020 (Crosses)"}, "album": {},
           "duration": 30, "preview": "u"}
UNLICENSED = {"id": 77, "title": "But Beautiful", "artist": {"name": "Earl Klugh"}, "album": {}, "duration": 30,
              "preview": "u"}


def test_resolve_searches_again_with_the_plain_key_when_the_names_find_nothing(tmp_path):
    dz = QueryDeezer(tmp_path, by_query={"crosses bitches brew": [CROSSES]})
    calls = []
    assert api.AnySong(FakeEngine(), dz, META).resolve("\u271d\u271d\u271d (Crosses)", "Bitches Brew",
                                                       lambda: calls.append(1)) == 42
    assert dz.queries == ["\u271d\u271d\u271d (Crosses) Bitches Brew", "crosses bitches brew"] and calls == [1]


def test_resolve_isrc_asks_the_catalog_then_deezer_once(tmp_path):
    dz = QueryDeezer(tmp_path, by_isrc={"USHU21300155": UNLICENSED})
    svc = api.AnySong(FakeEngine(), dz, META)
    calls = []
    assert svc.resolve_isrc("ushu2130 0155", "Daft Punk", "One More Time", lambda: calls.append(1)) == 1
    assert calls == [] and dz.isrc_lookups == 0  # an indexed release of the named song needs no Deezer request
    for _ in range(2):
        assert svc.resolve_isrc("USHU21300155", "Earl Klugh", "But Beautiful", lambda: calls.append(1)) == 77
    assert calls == [1] and dz.isrc_lookups == 1 and dz.searches == 0
    assert svc.similar(77)["query"]["source"] == "live" and dz.lookups == 0  # the lookup's track is reused


def test_resolve_isrc_falls_back_to_names_and_rejects_bad_codes(tmp_path):
    no_preview = UNLICENSED | {"id": 78, "preview": ""}
    dz = QueryDeezer(tmp_path, by_isrc={"GBAYE0000009": no_preview}, tracks=NEW_SONG)
    svc = api.AnySong(FakeEngine(), dz, META)
    assert svc.resolve_isrc("GBAYE0000001", "New Artist", "Brand New Song") == 99  # no ISRC hit: name search
    assert svc.resolve_isrc("GBAYE0000009", "New Artist", "Brand New Song") == 99  # hit without preview
    for code, status in (("GBAYE0000002", 404), ("not-an-isrc", 400)):
        with pytest.raises(api.ApiError) as err:
            svc.resolve_isrc(code)
        assert err.value.status == status
    dz.fail = True
    assert svc.resolve_isrc("GBAYE0000003", "Third", "Third Song") == 7  # Deezer down: the catalog still answers
    assert svc.resolve_isrc("GBAYE0000005", "New Artist", "Brand New Song") == 99  # searched before: cached
    with pytest.raises(api.ApiError) as err:
        svc.resolve_isrc("GBAYE0000004", "Someone Else", "Uncached Song")
    assert err.value.status == 502


def test_lru_evicts_and_expires():
    now = [0.0]
    c = api.LRU(2, ttl=10, clock=lambda: now[0])
    c.put("a", 1)
    c.put("b", 2)
    c.get("a")
    c.put("c", 3)
    assert c.get("b") is None and c.get("a") == 1
    now[0] = 11
    assert c.get("a") is None and len(c) == 1


def test_token_bucket_bursts_refills_and_prunes():
    now = [0.0]
    b = api.TokenBucket(per_min=60, burst=2, clock=lambda: now[0], max_clients=3)
    assert [b.allow("x") for _ in range(3)] == [True, True, False]
    now[0] = 1.0
    assert b.allow("x") and not b.allow("x")
    assert b.allow("y") and b.allow("z")
    now[0] = 100.0
    assert b.allow("w")  # table was full: idle clients are forgotten
    assert set(b._buckets) == {"w"}


@pytest.mark.parametrize("raw,key", [
    ("203.0.113.5", "203.0.113.5"),
    (" 203.0.113.5 ", "203.0.113.5"),
    ("2001:db8:1:2:aaaa::1", "2001:db8:1:2::/64"),
    ("2001:db8:1:2:ffff:ffff:ffff:ffff", "2001:db8:1:2::/64"),
    ("::ffff:198.51.100.7", "198.51.100.7"),
    ("", None),
    ("not-an-ip", None),
    ("203.0.113.5, 10.0.0.1", None),
])
def test_client_key_normalises_addresses(raw, key):
    assert api.client_key(raw) == key


@pytest.fixture
def server(tmp_path):
    # copy_score=None: the fake engine scores every row 0.9 on samples, which would read as copies of the seed
    srv = api.Server(("127.0.0.1", 0), api.AnySong(FakeEngine(), FakeDeezer(tmp_path), META, copy_score=None),
                     search_per_min=600, similar_per_min=8, cors_origin="https://yassin.app",
                     client_header="CF-Connecting-IP")
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield srv
    srv.shutdown()
    srv.server_close()


def get(srv, path, headers=None):
    req = urllib.request.Request(f"http://127.0.0.1:{srv.server_address[1]}{path}", headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, dict(resp.headers), json.loads(resp.read())
    except urllib.error.HTTPError as err:
        return err.code, dict(err.headers), json.loads(err.read())


def test_http_routes(server):
    code, headers, body = get(server, "/healthz")
    assert code == 200 and body["channels"] == {"melody": 7, "samples": 7}
    assert headers["Cache-Control"] == "no-store" and headers["Access-Control-Allow-Origin"] == "https://yassin.app"

    code, headers, body = get(server, "/api/anysong/similar?id=1&k=2")
    assert code == 200 and [x["id"] for x in body["results"]["melody"]] == [3, 5]
    assert headers["Cache-Control"] == "public, max-age=3600"
    assert headers["Content-Type"] == "application/json; charset=utf-8"

    code, _, body = get(server, "/api/anysong/search?q=one%20more%20time&limit=1")
    assert code == 200 and [t["id"] for t in body["tracks"]] == [99]

    code, headers, body = get(server, "/api/anysong/similar?title=One%20More%20Time%20-%20Remastered"
                                      "&artist=Daft%20Punk&k=2")
    assert code == 200 and body["query"]["id"] == 1 and [x["id"] for x in body["results"]["best"]] == [3, 5]
    assert headers["Cache-Control"] == "public, max-age=3600"

    code, _, body = get(server, "/api/anysong/similar?id=7&title=One%20More%20Time&artist=Daft%20Punk")
    assert code == 200 and body["query"]["id"] == 7  # an explicit id wins over names

    code, _, body = get(server, "/api/anysong/similar?id=1&lang=all")
    assert code == 200 and body["query"]["language_mode"] == "all"
    code, _, body = get(server, "/api/anysong/similar?id=1&lang=invalid")
    assert code == 400 and body["error"] == "lang must be 'same' or 'all'"


@pytest.mark.parametrize("path,status", [
    ("/nope", 404),
    ("/api/anysong/similar", 400),
    ("/api/anysong/similar?id=abc", 400),
    ("/api/anysong/similar?id=0", 400),
    ("/api/anysong/similar?id=1&k=51", 400),
    ("/api/anysong/similar?id=1&dedupe=2", 400),
    ("/api/anysong/similar?id=12345", 404),
    ("/api/anysong/similar?title=One%20More%20Time", 400),
    ("/api/anysong/similar?artist=Daft%20Punk&title=", 400),
    ("/api/anysong/similar?title=One%20More%20Time&artist=Daft%20Punk&k=0", 400),
    ("/api/anysong/similar?title=Unknown%20Song&artist=Nobody", 404),
    ("/api/anysong/search?q=", 400),
    ("/api/anysong/search?q=a&limit=26", 400),
    ("/api/anysong/similar?" + "&".join(f"a{i}=1" for i in range(9)), 400),
])
def test_http_errors(server, path, status):
    code, headers, body = get(server, path)
    assert code == status and body["ok"] is False and body["error"]
    assert headers["Cache-Control"] == "no-store"


def test_http_rate_limit_is_per_client_and_skips_cached_results(server):
    ip = {"CF-Connecting-IP": "203.0.113.5"}
    codes = [get(server, f"/api/anysong/similar?id={i}", ip)[0] for i in (1, 3, 5)]
    assert codes == [200, 200, 429]  # similar_per_min=8 -> burst of 2
    assert [get(server, "/api/anysong/similar?id=1&k=3", ip)[0] for _ in range(4)] == [200] * 4
    assert get(server, "/api/anysong/similar?id=5", {"CF-Connecting-IP": "203.0.113.6"})[0] == 200


def test_http_rate_limit_groups_ipv6_by_64_and_falls_back_to_peer(server):
    codes = [get(server, f"/api/anysong/similar?id={i}", {"CF-Connecting-IP": f"2001:db8:7:7::{i}"})[0]
             for i in (1, 3, 5)]
    assert codes == [200, 200, 429]  # rotating addresses inside one /64 shares a bucket
    assert get(server, "/api/anysong/similar?id=5", {"CF-Connecting-IP": "2001:db8:7:8::1"})[0] == 200
    codes = [get(server, f"/api/anysong/similar?id={i}", {"CF-Connecting-IP": "garbage"})[0] for i in (2, 4, 6)]
    assert codes[-1] == 429  # unparseable header -> keyed by the peer address, not a fresh bucket per value


def test_http_name_lookups_spend_the_search_budget_only_on_catalog_misses(tmp_path):
    srv = api.Server(("127.0.0.1", 0), api.AnySong(FakeEngine(), FakeDeezer(tmp_path, tracks=NEW_SONG), META),
                     search_per_min=4, similar_per_min=600)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        miss = "/api/anysong/similar?title=Another%20Song&artist=Nobody"
        assert [get(srv, miss)[0] for _ in range(2)] == [404, 429]  # search_per_min=4 -> burst of 1
        assert get(srv, "/api/anysong/similar?title=Third%20Song&artist=Third")[0] == 200
        assert srv.service.deezer.searches == 1
    finally:
        srv.shutdown()
        srv.server_close()


def test_http_isrc_lookups_answer_songs_deezer_search_cannot_see(tmp_path):
    dz = QueryDeezer(tmp_path, by_query={}, by_isrc={"USHU21300155": UNLICENSED})
    srv = api.Server(("127.0.0.1", 0), api.AnySong(FakeEngine(), dz, META), search_per_min=600,
                     similar_per_min=600)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        names = "title=But%20Beautiful&artist=Earl%20Klugh"
        assert get(srv, f"/api/anysong/similar?{names}")[0] == 404  # Deezer's search does not list it
        code, _, body = get(srv, f"/api/anysong/similar?isrc=USHU21300155&{names}&k=2")
        assert code == 200 and body["query"]["id"] == 77 and body["query"]["source"] == "live"
        code, _, body = get(srv, f"/api/anysong/similar?isrc=bad&{names}")
        assert code == 400 and body["error"] == "isrc must be a 12-character ISRC"
    finally:
        srv.shutdown()
        srv.server_close()


def test_http_internal_errors_are_generic(server, capsys):
    server.service.engine.search_stored = lambda tid, k: 1 / 0
    code, _, body = get(server, "/api/anysong/similar?id=7")
    assert code == 500 and body == {"ok": False, "error": "internal error"}
    assert "ZeroDivisionError" in capsys.readouterr().err


def test_engine_search_stored_matches_live_search_on_same_windows():
    torch = pytest.importorskip("torch")
    from soundalike.anysong import serve

    rng = np.random.default_rng(3)
    ids = np.arange(500, 560, dtype=np.int64)
    eng = serve.Engine.__new__(serve.Engine)
    eng.device, eng.indexes, eng.channels = "cpu", {}, []
    raw = {}
    for name in ("melody", "samples", "sonic"):
        ch = serve.CHANNELS[name]
        raw[name] = rng.standard_normal((len(ids), 3, 16)).astype(np.float16)
        eng.channels.append(ch)
        eng.indexes[name] = serve.ChannelIndex.from_windows(ch, ids, raw[name])
    tid = int(ids[11])
    assert eng.indexed(tid) and not eng.indexed(1)
    stored, timings = eng.search_stored(tid, 10)
    live, _ = eng.search_windows({n: w[11] for n, w in raw.items()}, 10, [tid])
    assert set(timings) == {"search_melody", "search_samples", "search_sonic"}
    for name in raw:
        assert tid not in [t for t, _ in stored[name]]
        assert [t for t, _ in stored[name]] == [t for t, _ in live[name]]
        np.testing.assert_allclose([s for _, s in stored[name]], [s for _, s in live[name]], atol=2e-3)
    assert torch.is_tensor(eng.indexes["melody"].stored_query(tid))


# ---------------------------------------------------------------- stem dissection
def _stem_record(seed, present=None):
    rng = np.random.default_rng(seed)
    rec = {name: np.zeros(shape, dtype) for name, (shape, dtype) in stemlib.COLUMNS.items()}
    rec["vec"] = rng.standard_normal((6, 512)).astype(np.float16)
    rec["present"] = np.ones(6, bool) if present is None else np.asarray(present, bool)
    rec["balance"] = rng.uniform(-30, 0, 6).astype(np.float32)
    desc = rng.uniform(1, 60, len(stemlib.DESC)).astype(np.float32)
    desc[stemlib.D["key_tonic"]], desc[stemlib.D["key_minor"]], desc[stemlib.D["drum_bars"]] = seed % 12, 1, 4
    rec["desc"] = desc
    rec["chroma"] = rng.dirichlet(np.ones(12)).astype(np.float16)
    rec["intervals"] = rng.dirichlet(np.ones(12), 2).astype(np.float16)
    rec["beats"][:] = stemlib.NO_BEAT
    rec["beats"][:4], rec["downbeat"][0] = [50, 100, 150, 200], True
    return rec


def _stem_table(ids=(1, 3), calib=True):
    recs = [_stem_record(t) for t in ids]
    meta = {"mu": np.zeros((6, 512)).tolist(),
            "calib": np.tile(np.linspace(-1, 1, 101), (len(stemlib.FACETS), 1)).tolist() if calib else None}
    return stemlib.StemTable(np.array(ids), stemlib.stack(recs), meta)


class FakeDissector:
    def __init__(self, gate=None, fail=None):
        self.gate, self.fail = gate, fail
        self.gpu_calls, self.cpu_calls = [], 0

    def gpu(self, mixes):
        self.gpu_calls.append(mixes.shape)
        if self.gate is not None:
            self.gate.wait(5)
        if self.fail is not None:
            raise self.fail
        return np.zeros((len(mixes), 6, 2, 4), np.float32), [(np.zeros(0), np.zeros(0))] * len(mixes), \
            np.zeros((len(mixes), 6, 512), np.float16)

    def cpu(self, stems, mixes, beats, vec):
        self.cpu_calls += 1
        return [_stem_record(1000 + len(self.gpu_calls)) for _ in mixes]


@pytest.fixture
def stereo(monkeypatch):
    decoded = []

    def decode(path):
        decoded.append(path)
        if path.read_bytes() == b"bad":
            raise RuntimeError("corrupt mp3")
        return np.zeros((2, stemlib.SAMPLES), np.float32)

    monkeypatch.setattr(stemlib, "decode_stereo", decode)
    return decoded


def test_dissect_serves_stored_rows_and_grades_the_pair(tmp_path):
    svc = api.AnySong(FakeEngine(), FakeDeezer(tmp_path), META, stems=stemlib.StemStore(_stem_table()))
    out = json.loads(json.dumps(svc.dissect(1, 3)))
    assert [s["id"] for s in out["songs"]] == [1, 3] and {s["source"] for s in out["songs"]} == {"catalog"}
    assert all("ms" not in s for s in out["songs"]) and out["songs"][0]["beats"] == [0.5, 1.0, 1.5, 2.0]
    assert out["compare"]["graded"] and out["compare"]["facets"]
    assert all(x["grade"] is not None for x in out["compare"]["facets"])
    assert set(out["compare"]["matched_on"]) <= {x["name"] for x in out["compare"]["facets"]}
    assert "compare" not in svc.dissect(1) and "compare" not in svc.dissect(1, 1)
    with pytest.raises(api.ApiError) as err:
        svc.dissect(1, 5)
    assert err.value.status == 404 and err.value.message == "this song has not been dissected yet"
    assert not svc.stems_missing(5)  # without a dissector nothing is computed, so nothing is charged
    assert svc.health()["stems"] == {"songs": 2, "live": False, "graded": True,
                                     "dissected": {"stored": 5, "live": 0}}
    assert svc.health()["gpu_mib"] is None


def test_dissect_is_off_without_a_store(svc):
    assert svc.health()["stems"] is None and not svc.stems_missing(1)
    with pytest.raises(api.ApiError) as err:
        svc.dissect(1, 3)
    assert err.value.status == 404 and err.value.message == "song dissection is not enabled"


def test_dissect_live_once_then_from_the_cache_and_never_keeps_the_download(tmp_path, stereo):
    dz, dis = FakeDeezer(tmp_path), FakeDissector()
    store = stemlib.StemStore(_stem_table(calib=False), tmp_path / "cache")
    svc = api.AnySong(FakeEngine(), dz, META, stems=store, dissector=dis)
    assert svc.stems_missing(99) and not svc.stems_missing(1)
    out = svc.dissect(1, 99)
    assert [s["source"] for s in out["songs"]] == ["catalog", "live"] and "ms" in out["songs"][1]
    assert not out["compare"]["graded"] and out["compare"]["matched_on"] == []
    assert dis.gpu_calls == [(1, 2, stemlib.SAMPLES)] and dis.cpu_calls == 1
    assert dz.lookups == 1 and not (tmp_path / "99.mp3").exists()
    assert (tmp_path / "cache" / "099" / "99.npz").exists() and not svc.stems_missing(99)
    assert svc.dissect(99)["songs"][0]["source"] == "cache" and len(dis.gpu_calls) == 1
    restarted = api.AnySong(FakeEngine(), dz, META, stems=stemlib.StemStore(None, tmp_path / "cache"),
                            dissector=FakeDissector())
    assert restarted.dissect(99)["songs"][0]["source"] == "cache"
    assert svc.health()["stems"]["dissected"] == {"stored": 2, "live": 1}


def test_dissect_reads_catalog_previews_without_deezer(tmp_path, stereo):
    (tmp_path / "previews" / "005").mkdir(parents=True)
    (tmp_path / "previews" / "005" / "5.mp3").write_bytes(b"mp3")
    dz = FakeDeezer(tmp_path)
    svc = api.AnySong(FakeEngine(), dz, META, dissector=FakeDissector(), catalog_previews=tmp_path / "previews")
    assert svc.dissect(5)["songs"][0]["source"] == "live"
    assert dz.lookups == 0 and stereo == [tmp_path / "previews" / "005" / "5.mp3"]
    assert (tmp_path / "previews" / "005" / "5.mp3").exists()  # the catalog copy stays
    assert svc.health()["stems"] == {"songs": 0, "live": True, "graded": False, "dissected": {"stored": 0, "live": 1}}


@pytest.mark.parametrize("setup,status,message", [
    (lambda dz, dis: setattr(dz, "fail", True), 502, "track lookup failed"),
    (lambda dz, dis: dz.tracks[99].update(preview=""), 404, "no preview available for this track"),
    (lambda dz, dis: dz.tracks.clear(), 404, "unknown track"),
    (lambda dz, dis: setattr(dz, "preview", b"bad"), 422, "could not decode the preview"),
    (lambda dz, dis: setattr(dis, "fail", RuntimeError("CUDA out of memory. Tried to allocate")), 503,
     "the GPU is busy, retry shortly"),
])
def test_dissect_live_errors_free_their_slot_and_leave_nothing_cached(tmp_path, stereo, setup, status, message):
    dz, dis = FakeDeezer(tmp_path), FakeDissector()
    setup(dz, dis)
    svc = api.AnySong(FakeEngine(), dz, META, max_pending=1, dissector=dis)
    for _ in range(2):  # a leaked pending slot would turn the second try into 503 busy
        with pytest.raises(api.ApiError) as err:
            svc.dissect(99)
        assert (err.value.status, err.value.message) == (status, message)
    assert svc.stems_missing(99) and not svc._stem_flight and not (tmp_path / "99.mp3").exists()


def test_dissect_other_gpu_errors_propagate(tmp_path, stereo):
    svc = api.AnySong(FakeEngine(), FakeDeezer(tmp_path), META, dissector=FakeDissector(fail=ValueError("shape")))
    with pytest.raises(ValueError):
        svc.dissect(99)


def test_dissect_busy_and_concurrent_requests_for_one_song_dissect_once(tmp_path, stereo):
    gate = threading.Event()
    dis = FakeDissector(gate=gate)
    svc = api.AnySong(FakeEngine(), FakeDeezer(tmp_path), META, max_pending=1, dissector=dis)
    results = []
    threads = [threading.Thread(target=lambda: results.append(svc.dissect(99))) for _ in range(3)]
    for th in threads:
        th.start()
    for _ in range(100):
        if dis.gpu_calls:
            break
        threading.Event().wait(0.01)
    with pytest.raises(api.ApiError) as err:  # the one pending slot is taken by the running dissection
        svc.dissect(77)
    assert err.value.status == 503
    gate.set()
    for th in threads:
        th.join(5)
    assert len(results) == 3 and len(dis.gpu_calls) == 1
    assert sorted(r["songs"][0]["source"] for r in results) == ["live", "live", "live"]


def test_http_dissect_route_charges_only_live_dissections(tmp_path, stereo):
    store = stemlib.StemStore(_stem_table())
    srv = api.Server(("127.0.0.1", 0), api.AnySong(FakeEngine(), FakeDeezer(tmp_path), META, stems=store,
                                                   dissector=FakeDissector()),
                     search_per_min=600, similar_per_min=600, dissect_per_min=4)  # burst of 1
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        for _ in range(3):
            code, headers, body = get(srv, "/api/anysong/dissect?id=1&with=3")
            assert code == 200 and body["ok"] and len(body["songs"]) == 2 and "compare" in body
            assert headers["Cache-Control"] == "public, max-age=3600"
        code, _, body = get(srv, "/api/anysong/dissect?id=1&with=99")
        assert code == 200 and body["songs"][1]["source"] == "live"
        assert get(srv, "/api/anysong/dissect?id=99&with=1")[0] == 200  # now cached: free
        code, _, body = get(srv, "/api/anysong/dissect?id=1&with=77")
        assert code == 429 and body["error"] == "too many requests"
        for path in ("/api/anysong/dissect", "/api/anysong/dissect?id=x", "/api/anysong/dissect?id=1&with=0"):
            code, _, body = get(srv, path)
            assert code == 400 and not body["ok"]
        assert get(srv, "/healthz")[2]["stems"]["live"] is True
    finally:
        srv.shutdown()
        srv.server_close()
