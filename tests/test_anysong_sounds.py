import json

import numpy as np
import pytest

from soundalike.anysong import api, sounds
from soundalike.anysong.scene import build as build_scene

META = {
    1: ("Seed Artist", "Seed Song"),
    2: ("Seed Artist", "Second Song"),
    3: ("Scene Artist", "Scene Song"),
    4: ("Scene Artist", "Scene Song Two"),
    5: ("Far Artist", "Far Song"),
    6: ("Far Artist", "Far Song Two"),
    7: ("French Act", "Chanson"),
    8: ("Other", "Other Song"),
}
# Best-first sounds rows for seed 1. z-scores: 5: 1.21, 6: 1.13, 3: 0.42, 4: 0.34, 2: -0.37, 8: -1.17, 7: -1.56
SOUNDS_ROWS = [(5, 0.90), (6, 0.89), (3, 0.80), (4, 0.79), (2, 0.70), (8, 0.60), (7, 0.55)]
AUDIO_ROWS = {"melody": [(7, 0.9), (5, 0.8), (3, 0.7)], "samples": [(3, 0.5), (8, 0.4)]}


def scene_table():
    # Seed Artist (100) and Scene Artist (300) list each other; Far Artist (200) is a US act; 400 and 500 have no data.
    tracks = {1: 100, 2: 100, 3: 300, 4: 300, 5: 200, 6: 200, 7: 400, 8: 500}
    related = {100: [300, 600], 300: [100, 600], 200: [700, 701]}
    return build_scene(tracks, related, {100: ["GB"], 300: ["GB"], 600: ["GB"], 200: ["US"]}, {})


class Index:
    def __init__(self, ids):
        self.pos = {int(t): i for i, t in enumerate(ids)}

    def __len__(self):
        return len(self.pos)


class Engine:
    device = "cpu"

    def __init__(self, rows):
        self.rows = rows
        self.indexes = {channel: Index(META) for channel in rows}
        self.depths = []

    def indexed(self, tid):
        return all(tid in ix.pos for ix in self.indexes.values())

    def search_stored(self, tid, k):
        self.depths.append(k)
        return {ch: [r for r in rows if r[0] != tid] for ch, rows in self.rows.items()}, {}


class Deezer:
    def related_artists(self, artist_id):
        raise AssertionError("the scene table has the seed artist's related list")


class Languages:
    def __init__(self, labels):
        self.labels = labels

    def language(self, tid):
        return self.labels.get(tid, "unknown")

    def artist_language(self, artist):
        return "unknown"


def service(rows, **kw):
    return api.AnySong(Engine(rows), Deezer(), META, **kw)


def test_order_ranks_by_z_score_plus_scene_affinity_and_keeps_ties():
    assert sounds.order([], []) == []
    assert sounds.order([0.5, 0.5, 0.5], [0.0, 1.0, 0.0]) == [1, 0, 2]
    # z = +1, -1: an affinity of 1.5 is not enough to pass the first song, 2.5 is
    assert sounds.order([1.0, 0.0], [0.0, 1.5]) == [0, 1]
    assert sounds.order([1.0, 0.0], [0.0, 2.5]) == [1, 0]
    assert sounds.order([1.0, 0.0], [0.0, 2.5], scene_weight=0.5) == [0, 1]


def test_cap_keeps_the_first_songs_of_each_group():
    assert sounds.cap(["a", "a", "b", "c", "b"], k=10) == [0, 2, 3]
    assert sounds.cap(["a", "b", "c"], k=2) == [0, 1]
    assert sounds.cap(["a", "a", "a"], k=5, per_group=2) == [0, 1]


def test_sounds_list_lifts_the_scene_and_keeps_one_song_per_artist():
    svc = service({**AUDIO_ROWS, "sounds": SOUNDS_ROWS}, scene=scene_table())
    r = svc.similar(1, k=10)
    # z + affinity: Scene Artist (affinity 1) passes Far Artist; the seed's own artist counts 0.6
    assert [x["id"] for x in r["results"]["sounds"]] == [3, 5, 2, 8, 7]
    assert [x["score"] for x in r["results"]["sounds"]] == [round(1 / (api.FUSE_K + i), 5) for i in range(1, 6)]
    assert r["results"]["sounds"][0] == {"id": 3, "score": round(1 / (api.FUSE_K + 1), 5), "artist": "Scene Artist",
                                         "title": "Scene Song"}
    assert r["query"]["hidden_sounds"] == 0
    assert svc.engine.depths == [{"melody": api.FETCH, "samples": api.FETCH, "sounds": sounds.DEPTH}]
    # scene=0: the channel order, still one song per artist
    assert [x["id"] for x in svc.similar(1, k=10, scene=False)["results"]["sounds"]] == [5, 3, 2, 8, 7]
    assert [x["id"] for x in svc.similar(1, k=2)["results"]["sounds"]] == [3, 5]


def test_sounds_channel_leaves_best_and_the_channel_lists_unchanged():
    plain = service(AUDIO_ROWS, scene=scene_table())
    with_sounds = service({**AUDIO_ROWS, "sounds": SOUNDS_ROWS}, scene=scene_table())
    a, b = plain.similar(1, k=10), with_sounds.similar(1, k=10)
    assert b["results"]["best"] == a["results"]["best"]
    assert {ch: b["results"][ch] for ch in AUDIO_ROWS} == {ch: a["results"][ch] for ch in AUDIO_ROWS}
    assert "sounds" not in a["results"] and "hidden_sounds" not in a["query"]
    assert plain.engine.depths == [api.FETCH]


def test_sounds_list_hides_other_languages_in_same_language_mode():
    labels = {1: "en", 2: "en", 3: "en", 5: "unknown", 7: "fr", 8: "en"}
    svc = service({"sounds": SOUNDS_ROWS}, scene=scene_table(), languages=Languages(labels))
    same = svc.similar(1, k=10)
    assert [x["id"] for x in same["results"]["sounds"]] == [3, 5, 2, 8]
    assert same["query"]["hidden_sounds"] == 1 and same["query"]["language"] == "en"
    every = svc.similar(1, k=10, lang_mode="all")
    assert [x["id"] for x in every["results"]["sounds"]] == [3, 5, 2, 8, 7] and every["query"]["hidden_sounds"] == 1
    assert svc.similar(1, k=3)["query"]["hidden_sounds"] == 0  # French Act is not in the top 3


def test_sounds_list_without_a_scene_table_caps_by_artist_name():
    svc = service({"sounds": SOUNDS_ROWS})
    assert [x["id"] for x in svc.similar(1, k=10)["results"]["sounds"]] == [5, 3, 2, 8, 7]


def _head_arrays(rng, d, h, scale=0.3):
    return {"inp_w": (scale * rng.standard_normal((h, d))).astype(np.float32),
            "inp_b": (scale * rng.standard_normal(h)).astype(np.float32),
            "out_w": (scale * rng.standard_normal((d, h))).astype(np.float32),
            "out_b": (scale * rng.standard_normal(d)).astype(np.float32)}


def test_head_is_a_residual_gelu_mlp_loaded_without_pickles(tmp_path):
    torch = pytest.importorskip("torch")
    from soundalike.anysong import serve

    rng = np.random.default_rng(0)
    w = _head_arrays(rng, 8, 16)
    np.savez(tmp_path / "head.npz", **w)
    head = serve.Head.load(tmp_path / "head.npz")
    lin1, lin2 = torch.nn.Linear(8, 16), torch.nn.Linear(16, 8)
    with torch.no_grad():
        lin1.weight.copy_(torch.from_numpy(w["inp_w"]))
        lin1.bias.copy_(torch.from_numpy(w["inp_b"]))
        lin2.weight.copy_(torch.from_numpy(w["out_w"]))
        lin2.bias.copy_(torch.from_numpy(w["out_b"]))
        x = torch.randn(5, 3, 8)
        torch.testing.assert_close(head(x), x + lin2(torch.nn.functional.gelu(lin1(x))), rtol=1e-5, atol=1e-5)
    zero = serve.Head(torch.from_numpy(w["inp_w"]), torch.from_numpy(w["inp_b"]), torch.zeros(8, 16), torch.zeros(8))
    torch.testing.assert_close(zero(x), x)
    prepared = serve.prep_windows(3 * x, "head", head=head)
    torch.testing.assert_close(prepared.norm(dim=-1), torch.ones(5, 3))
    with pytest.raises(ValueError):
        serve.prep_windows(x, "head")
    with pytest.raises(ValueError):
        serve.Head(torch.zeros(16, 8), torch.zeros(16), torch.zeros(8, 17), torch.zeros(8))


def test_convert_head_reads_a_trained_state_dict(tmp_path):
    torch = pytest.importorskip("torch")
    from soundalike.anysong import serve

    class Residual(torch.nn.Module):  # the trained module's layout
        def __init__(self):
            super().__init__()
            self.inp, self.outp = torch.nn.Linear(8, 16), torch.nn.Linear(16, 8)

        def forward(self, x):
            return x + self.outp(torch.nn.functional.gelu(self.inp(x)))

    model = Residual()
    torch.save(model.state_dict(), tmp_path / "head.pt")
    sha = sounds.convert_head(tmp_path / "head.pt", tmp_path / "head.npz")
    assert len(sha) == 64
    x = torch.randn(4, 8)
    with torch.no_grad():
        torch.testing.assert_close(serve.Head.load(tmp_path / "head.npz")(x), model(x), rtol=1e-5, atol=1e-5)


def test_build_writes_a_two_stage_index_that_matches_brute_force(tmp_path):
    pytest.importorskip("torch")
    from soundalike.anysong import serve

    rng = np.random.default_rng(1)
    n, w, d = 120, 3, 8
    ids = np.arange(1000, 1000 + n, dtype=np.int64)
    raw = rng.standard_normal((n, w, d)).astype(np.float32)
    sonic = tmp_path / "sonic"
    sonic.mkdir()
    np.save(sonic / "ids.npy", ids)
    np.save(sonic / "full.npy", (raw / np.linalg.norm(raw, axis=-1, keepdims=True)).astype(np.float16))
    (sonic / "meta.json").write_text(json.dumps({"model": "muq_mulan", "prep": "raw", "ids_sha256": "s", "built": "t"}))
    np.savez(tmp_path / "head.npz", **_head_arrays(rng, d, 16))

    meta = sounds.build(sonic, tmp_path / "head.npz", tmp_path / "sounds", dim=4, candidates=n, device="cpu")
    assert (meta["channel"], meta["prep"], meta["n"], meta["candidates"]) == ("sounds", "head", n, n)
    assert meta["source"] == {"sonic_ids_sha256": "s", "sonic_built": "t"} and len(meta["head_sha256"]) == 64
    channel = serve.ALL_CHANNELS["sounds"]
    built = serve.TwoStageIndex.load(tmp_path / "sounds", channel)
    brute = serve.ChannelIndex.from_windows(channel, ids, raw, head=serve.Head.load(tmp_path / "head.npz"))
    for row in (5, 77):
        tid = int(ids[row])
        a = built.top(built.scores(built.prep_query(raw[row])), 6, [tid])
        b = brute.top(brute.scores(brute.prep_query(raw[row])), 6, [tid])
        assert [t for t, _ in a] == [t for t, _ in b]
        np.testing.assert_allclose([s for _, s in a], [s for _, s in b], atol=3e-3)
        stored = built.top(built.scores(built.stored_query(tid)), 6, [tid])
        assert [t for t, _ in stored] == [t for t, _ in a]
    with pytest.raises(ValueError):
        serve.ChannelIndex.load(tmp_path, channel)  # no shard fallback for a projected channel
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "meta.json").write_text(json.dumps({"model": "clews_dvi", "prep": "none"}))
    with pytest.raises(ValueError):
        sounds.build(bad, tmp_path / "head.npz", tmp_path / "x", device="cpu")


def test_engine_embeds_each_model_once_for_channels_that_share_it():
    pytest.importorskip("torch")
    from soundalike.anysong import serve

    class Embedder:
        sr = 24000
        calls = 0

        def embed(self, batch):
            Embedder.calls += 1
            return {"win": np.ones((1, 3, 4), np.float32)}

    eng = serve.Engine.__new__(serve.Engine)
    eng.device, eng.embedders = "cpu", {"muq_mulan": Embedder()}
    eng.channels = [serve.ALL_CHANNELS["sonic"], serve.ALL_CHANNELS["sounds"]]
    wins, timings = eng.embed_clips({24000: np.zeros(10, np.float32)})
    assert Embedder.calls == 1 and wins["sonic"] is wins["sounds"] and set(timings) == {"embed_muq_mulan"}
