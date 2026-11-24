import numpy as np
import pytest

torch = pytest.importorskip("torch")
from soundalike.anysong import consensus as cm
from soundalike.anysong import listener, serve
from soundalike.anysong.blend import Blend
from soundalike.anysong.scene import build as build_scene
from soundalike.anysong.scenehead import SceneHead

N, W, D = 60, 3, 16
IDS = np.arange(1000, 1000 + N, dtype=np.int64)
# Twenty artists with three songs each; every third song credits a guest after a comma, in another case.
ARTISTS = [f"Artist {i % 20}" if i % 3 else f"ARTIST {i % 20} , Guest {i}" for i in range(N)]


def _unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _constant(value=0.0, features=("lr_snd",)):
    """A model of one leaf: every song is predicted ``value``."""
    one = np.array([0])
    return listener.Regressor(features, value, one, one, np.zeros(1), np.array([True]), one, one, np.zeros(1), 0)


def _stump(feature, threshold, left, right):
    return listener.Regressor((feature,), 0.0, np.array([0]), np.array([0, 0, 0]), np.array([threshold, 0, 0]),
                              np.array([True, True, True]), np.array([1, 1, 2]), np.array([2, 1, 2]),
                              np.array([0.0, left, right]), depth=1)


class World:
    """A catalog of N songs with every table the consensus reads; the melody index has its own row order."""

    def __init__(self, tmp_path, model=None, **kw):
        tmp_path = tmp_path / f"world{len(list(tmp_path.iterdir()))}"  # an open memory map keeps its file on Windows
        tmp_path.mkdir()
        rng = np.random.default_rng(0)
        self.win = _unit(rng.standard_normal((N, W, D))).astype(np.float16)
        np.savez(tmp_path / "h.npz", w1=rng.standard_normal((D, 8)).astype(np.float32),
                 b1=rng.standard_normal(8).astype(np.float32), w2=rng.standard_normal((8, 6)).astype(np.float32),
                 b2=rng.standard_normal(6).astype(np.float32))
        np.save(tmp_path / "h.emb.npy", np.zeros((N, W, 6), np.float16))
        pos = {int(t): i for i, t in enumerate(IDS)}
        head = SceneHead(tmp_path / "h.npz", tmp_path / "h.emb.npy", pos)
        table = np.stack([head.embed(w.astype(np.float32)) for w in self.win]).astype(np.float16)
        del head  # releases the memory map, which Windows will not replace while it is open
        np.save(tmp_path / "h.emb.npy", table)
        self.graph = np.sort(rng.choice(N, 25, replace=False))
        self.cf = rng.standard_normal((25, 5)).astype(np.float32) * 3
        self.cf[3] *= 1e-8  # all but zero: unit length here, not in the blend
        np.savez(tmp_path / "cooc.npz", rows=self.graph)
        np.save(tmp_path / "cf128.npy", self.cf)
        self.blend = Blend.load(tmp_path / "h", tmp_path, pos)
        self.melody_ids = IDS[rng.permutation(N)]
        self.bases = {"melody": torch.from_numpy(rng.standard_normal(N).astype(np.float32)),
                      "samples": torch.from_numpy(rng.standard_normal(N).astype(np.float32))}
        self.codes = rng.integers(-1, 3, N).astype(np.int16)
        self.bpm = rng.uniform(60, 180, N).astype(np.float32)
        self.bpm[[5, 9]] = np.nan
        self.bpm[7] = 30.0  # not a tempo the features read
        args = dict(model=model or _constant(), ids=IDS, windows=self.win, blend=self.blend, cf=self.cf,
                    orders={"melody": self.melody_ids, "samples": IDS}, artists=ARTISTS, codes=self.codes,
                    bpm=self.bpm, prep=lambda w: torch.from_numpy(_unit(np.asarray(w, dtype=np.float32))))
        self.cons = cm.Consensus(**(args | kw))


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def _col(c, name):
    return c.X[:, cm.FEATURES.index(name)].astype(np.float64)


def test_regressor_export_matches_scikit_learn_with_missing_values(tmp_path):
    ensemble = pytest.importorskip("sklearn.ensemble")
    rng = np.random.default_rng(1)
    X = rng.standard_normal((3000, 6))
    X[rng.random(X.shape) < 0.15] = np.nan
    X[:, 4] = np.where(np.isnan(X[:, 4]), 0.0, X[:, 4])  # a column without missing values in training
    y = np.where(np.isnan(X[:, 0]), 1.5, X[:, 0]) + np.nan_to_num(X[:, 1]) * np.nan_to_num(X[:, 2]) \
        + 0.1 * rng.standard_normal(len(X))
    model = ensemble.HistGradientBoostingRegressor(max_iter=40, max_leaf_nodes=15, random_state=0).fit(X, y)
    names = tuple(f"any_{i}" for i in range(6))
    exported = listener.export_regressor(model, names, "note")
    fresh = rng.standard_normal((2000, 6)).astype(np.float32) * 2
    fresh[rng.random(fresh.shape) < 0.3] = np.nan  # also in the column that had none
    assert np.isnan(fresh[:, 4]).any() and np.array_equal(exported.predict(fresh), model.predict(fresh))
    exported.save(tmp_path / "m.npz")
    loaded = listener.Regressor.load(tmp_path / "m.npz")
    assert loaded.features == names and loaded.meta == "note"
    assert np.array_equal(loaded.predict(fresh), model.predict(fresh)) and loaded.predict(fresh[:0]).shape == (0,)
    with pytest.raises(ValueError, match="feature names"):
        listener.export_regressor(model, names[:5])
    with pytest.raises(ValueError, match="distinct"):
        listener.Regressor(("a", "a"), 0.0, *[np.zeros(1, int)] * 2, np.zeros(1), np.ones(1, bool),
                           *[np.zeros(1, int)] * 2, np.zeros(1), 0)
    # The classifier path is as it was: its own feature lists, probabilities, and no regressors.
    clf = ensemble.HistGradientBoostingClassifier(max_iter=20, random_state=0).fit(
        np.nan_to_num(X[:, :1]).repeat(len(listener.FEATURES), 1), y > 0)
    ranker = listener.export(clf)
    probe = rng.standard_normal((500, len(listener.FEATURES)))
    np.testing.assert_allclose(ranker.predict(probe), clf.predict_proba(probe)[:, 1], atol=1e-12)
    with pytest.raises(ValueError, match="classifiers"):
        listener.export(model)
    with pytest.raises(ValueError, match="regressors"):
        listener.export_regressor(clf, listener.FEATURES)
    with pytest.raises(ValueError, match="differ from this code's"):
        listener.Ranker(names, 0.0, *[np.zeros(1, int)] * 2, np.zeros(1), np.ones(1, bool),
                        *[np.zeros(1, int)] * 2, np.zeros(1), 0)


def test_signals_score_every_song_on_each_views_own_scale(world):
    cons, row = world.cons, int(world.graph[0])
    sig = cons.signals(world.bases, row)
    unit = _unit(world.win.astype(np.float64))
    np.testing.assert_allclose(sig["snd"].numpy(), np.einsum("wd,nvd->nwv", unit[row], unit).max(axis=(1, 2)),
                               atol=3e-3)
    table = _unit(world.blend.table.numpy().astype(np.float64))
    np.testing.assert_allclose(sig["head"].numpy(), np.einsum("wd,nvd->nwv", table[row], table).max(axis=(1, 2)),
                               atol=3e-3)
    # Playlist cosines for the graph songs alone, every vector at unit length (the all-but-zero one too).
    cf = _unit(world.cf.astype(np.float64))
    got = sig["cf"].numpy()
    np.testing.assert_allclose(got[world.graph], cf @ cf[0], atol=1e-5)
    assert np.isneginf(np.delete(got, world.graph)).all() and abs(got[world.graph[3]]) > 1e-3
    outside = next(i for i in range(N) if i not in world.graph)
    assert np.isneginf(cons.signals(world.bases, outside)["cf"].numpy()).all()
    # Melody scores arrive in the melody index's row order.
    at = {int(t): i for i, t in enumerate(world.melody_ids)}
    assert sig["mel"].tolist() == [float(world.bases["melody"][at[int(t)]]) for t in IDS]
    assert torch.equal(sig["smp"], world.bases["samples"])


def test_features_describe_each_candidate_under_every_signal(world):
    cons = world.cons
    row = int(world.graph[0])
    code = int(world.codes[row]) if world.codes[row] >= 0 else 1
    seed = cons.seed(code, row)
    sig = cons.signals(world.bases, row)
    allow = ((world.codes == code) | (world.codes < 0)) & (cons.artist_np != cons.artist_np[row])
    assert cons.allowed(seed).tolist() == allow.tolist() and not allow[row] and 5 < allow.sum() < N - 3
    c = cons.candidates(sig, seed)
    assert c.rows.tolist() == np.flatnonzero(allow).tolist()  # a small catalog: every allowed song is a candidate
    assert c.X.dtype == np.float32 and c.X.shape == (len(c.rows), len(cm.FEATURES))
    logs = []
    for k, key in enumerate(cm.SIGNALS):
        s = sig[key].numpy().astype(np.float64)
        pool = s[allow & np.isfinite(s)]
        v = s[c.rows]
        has = np.isfinite(v)
        rank = np.array([1 + (pool > x).sum() for x in v])
        assert c.ranks[:, k].tolist() == np.where(has, rank, 0).tolist()
        lr = np.where(has, np.log10(rank), cm.NO_RANK)
        logs.append(lr)
        top = np.sort(np.where(allow, s, -np.inf))[::-1]  # the tenth best may be a song without a score
        v = np.where(has, v, 0.0)
        for name, want in (("s", v), ("z", (v - pool.mean()) / pool.std()), ("g1", v - top[0]), ("g10", v - top[9])):
            got = _col(c, f"{name}_{key}")
            assert np.isnan(got).tolist() == (~has).tolist()
            np.testing.assert_allclose(got[has], want[has], atol=1e-5)
        np.testing.assert_allclose(_col(c, f"lr_{key}"), lr, atol=1e-6)
    assert (~np.isfinite(sig["cf"].numpy()[c.rows])).any()  # songs outside the playlist graph: no score, NO_RANK
    L = np.stack(logs, 1)
    for name, cut in (("n10", 10), ("n50", 50), ("n200", 200), ("n1000", 1000)):
        assert _col(c, name).tolist() == (L <= np.log10(cut) + 1e-9).sum(1).tolist()
    np.testing.assert_allclose(_col(c, "lr_min"), L.min(1), atol=1e-6)
    np.testing.assert_allclose(_col(c, "lr_2nd"), np.sort(L, 1)[:, 1], atol=1e-6)
    np.testing.assert_allclose(_col(c, "lr_mean"), L.mean(1), atol=1e-6)
    # Tempo: |log2 ratio|, folded over an octave; unknown and implausible tempos are missing.
    bpm = world.bpm[c.rows].astype(np.float64)
    known = bpm > 40
    d = np.abs(np.log2(bpm[known] / float(world.bpm[row])))
    np.testing.assert_allclose(_col(c, "tempo_strict")[known], d, atol=1e-5)
    np.testing.assert_allclose(_col(c, "tempo_fold")[known], np.minimum(d, np.abs(d - 1)), atol=1e-5)
    assert np.isnan(_col(c, "tempo_fold")[~known]).all() and (~known).any()
    assert _col(c, "lang_same").tolist() == (world.codes[c.rows] == code).tolist()
    assert _col(c, "lang_unknown").tolist() == (world.codes[c.rows] < 0).tolist()
    assert set(_col(c, "lang_q_known")) == {1.0} and set(_col(c, "cf_q")) == {1.0}
    assert _col(c, "cf_c").tolist() == np.isin(c.rows, world.graph).tolist()
    assert np.isnan(c.X[:, cm.FEATURES.index("graph"):cm.FEATURES.index("nb") + 1]).all()  # no scene table
    # A study's own rows replace the pool, the ranks stay those among the allowed songs.
    some = cons.candidates(sig, seed, rows=c.rows[[4, 1]])
    assert np.array_equal(some.X, c.X[[4, 1]], equal_nan=True) and np.array_equal(some.score, c.score[[4, 1]])
    # A seed outside the graph has no playlist view of any song; without a tempo, no tempo columns.
    outside = next(i for i in range(N) if i not in world.graph and not world.bpm[i] > 40)
    c = cons.candidates(cons.signals(world.bases, outside), cons.seed(-1, outside))
    assert np.isnan(_col(c, "s_cf")).all() and set(_col(c, "lr_cf")) == {np.float64(np.float32(cm.NO_RANK))}
    assert set(_col(c, "cf_q")) == {0.0} and np.isnan(_col(c, "tempo_strict")).all()
    assert set(_col(c, "lang_q_known")) == {0.0} and not _col(c, "lang_same").any()


def test_body_orders_by_score_one_song_per_artist_without_the_seeds_artist(world, tmp_path):
    cons, row = world.cons, 4
    sig = cons.signals(world.bases, row)
    seed = cons.seed(-1, row)
    own = {i for i in range(N) if i % 20 == row % 20}  # "Artist 4" and "ARTIST 4 , Guest .." are one artist
    assert len(own) == 3 and {int(cons.artist_np[i]) for i in own} == {seed.artist}
    snd = sig["snd"].numpy()
    by_sound = [i for i in np.argsort(-snd, kind="stable") if i not in own]
    rank = {i: k for k, i in enumerate(by_sound, 1)}
    # A constant prediction: the sound term alone orders the body; the best song of each of the 19 other artists.
    body = cons.body(sig, seed)
    want, seen = [], set()
    for i in by_sound:
        if i % 20 not in seen:
            seen.add(i % 20)
            want.append(i)
    assert [t for t, _, _ in body] == [int(IDS[i]) for i in want] and len(body) == 19
    for (tid, score, ranks), i in zip(body, want):
        assert ranks[0] == rank[i] and score == pytest.approx(-0.1 * np.log10(rank[i]), abs=1e-6)
        assert len(ranks) == len(cm.SIGNALS) and all(type(r) is int for r in ranks)
    assert [t for t, _, _ in cons.body(sig, seed, keep=4)] == [int(IDS[i]) for i in want[:4]]
    # Songs past the sound cap are in no body, whatever the model thinks of them.
    cons.sound_cap = 12
    capped = cons.body(sig, seed)
    assert [t for t, _, _ in capped] == [int(IDS[i]) for i in want if rank[i] <= 12] and 0 < len(capped) < 19
    # The model decides, the sound term breaks its ties: songs in the head signal's top 5 first.
    cons = World(tmp_path, model=_stump("lr_head", float(np.log10(5.5)), 2.0, 0.0), sound_weight=0.25).cons
    c = cons.candidates(sig, seed)
    top5 = set(c.rows[c.ranks[:, 1] <= 5].tolist())
    assert len(top5) == 5
    np.testing.assert_allclose(c.score, np.where(np.isin(c.rows, list(top5)), 2.0, 0.0)
                               - 0.25 * np.log10(c.ranks[:, 0]), atol=1e-6)
    first = [i for i in by_sound if i in top5]
    kept = [i for k, i in enumerate(first) if i % 20 not in {j % 20 for j in first[:k]}]
    assert [t for t, _, _ in cons.body(sig, seed)][:len(kept)] == [int(IDS[i]) for i in kept]


def test_scores_are_the_models_predictions_less_the_sound_term(tmp_path):
    ensemble = pytest.importorskip("sklearn.ensemble")
    names = ("s_cf", "lr_snd", "tempo_fold", "z_head", "lang_unknown")  # two of them with missing values
    rng = np.random.default_rng(2)
    X = rng.standard_normal((2000, len(names)))
    X[rng.random(X.shape) < 0.2] = np.nan
    model = ensemble.HistGradientBoostingRegressor(max_iter=30, random_state=0).fit(X, np.nansum(X, 1))
    world = World(tmp_path, model=listener.export_regressor(model, names), sound_cap=30)
    cons = world.cons
    c = cons.candidates(cons.signals(world.bases, 4), cons.seed(-1, 4))
    inputs = c.X[:, [cm.FEATURES.index(f) for f in names]]
    assert np.isnan(inputs[:, 0]).any() and np.isnan(inputs[:, 2]).any()
    inside = c.ranks[:, 0] <= 30
    want = model.predict(inputs) - 0.1 * np.minimum(c.X[:, cm.FEATURES.index("lr_snd")], 5.0)
    assert np.array_equal(c.score[inside], want[inside]) and np.isneginf(c.score[~inside]).all() and (~inside).any()
    assert np.array_equal(cons._predict(inputs), model.predict(inputs))  # the device descent is the model's own


def test_same_language_body_ranks_among_the_languages_songs(world):
    cons, row = world.cons, 4
    code = 2
    out = cons.bodies(world.bases, code, row=row)
    assert out["same"] and {int(world.codes[t - 1000]) for t, _, _ in out["same"]} <= {code, -1}
    assert {int(world.codes[t - 1000]) for t, _, _ in out["all"]} - {code, -1}  # other languages only here
    # Ranks count the allowed songs alone: the language's closest song is its sound rank 1.
    assert out["same"][0][2][0] == 1 and out["all"][0][2][0] == 1 and out["same"][0][0] != out["all"][0][0] or \
        world.codes[out["all"][0][0] - 1000] in (code, -1)
    assert out["all"] == cons.body(cons.signals(world.bases, row), cons.seed(-1, row))
    assert out["same"] == cons.body(cons.signals(world.bases, row), cons.seed(code, row))
    unknown = cons.bodies(world.bases, -1, row=row)
    assert unknown["same"] is unknown["all"] and unknown["all"] == out["all"]


def test_live_seed_uses_its_own_windows_and_has_no_playlist_view(tmp_path):
    tracks = {int(t): 100 + i % 20 for i, t in enumerate(IDS)}
    scene = build_scene(tracks, {104: [105, 106], 105: [104]}, {104: ["FR"], 105: ["FR"], 106: ["US"]}, {})
    world = World(tmp_path, scene=scene)
    cons, row = world.cons, 4
    live = _unit(world.win[row].astype(np.float32) + 0.02 * np.random.default_rng(3).standard_normal((W, D)))
    sig = cons.signals(world.bases, windows=live)
    stored = cons.signals(world.bases, row)
    assert int(sig["snd"].argmax()) == row and np.isneginf(sig["cf"].numpy()).all()
    np.testing.assert_allclose(sig["snd"].numpy(), stored["snd"].numpy(), atol=0.1)
    np.testing.assert_allclose(sig["head"].numpy(), stored["head"].numpy(), atol=0.2)
    # The credited artist is matched by primary name; the scene table knows the artist id; the tempo is the live one.
    seed = cons.seed(1, artist=" artist 4, Someone", artist_id=104, bpm=float(world.bpm[row]))
    assert seed.row is None and seed.artist == int(cons.artist_np[row])
    c = cons.candidates(sig, seed)
    assert not {i for i in range(N) if i % 20 == 4} & set(c.rows.tolist())
    assert np.isnan(_col(c, "s_cf")).all() and set(_col(c, "cf_q")) == {0.0} and np.isfinite(_col(c, "s_snd")).all()
    assert np.isfinite(_col(c, "tempo_fold")[world.bpm[c.rows] > 40]).all()
    by_artist = {int(i) % 20: j for j, i in enumerate(c.rows)}
    fwd, rev, nb, graph = (_col(c, name) for name in ("fwd", "rev", "nb", "graph"))
    assert fwd[by_artist[5]] == 1 and rev[by_artist[5]] == 1 and fwd[by_artist[6]] == 1 and rev[by_artist[6]] == 0
    assert nb[by_artist[5]] == 1 and np.isfinite(graph[by_artist[5]])
    assert graph[by_artist[7]] == 0 and fwd[by_artist[7]] == 0 and nb[by_artist[7]] == 0  # an unrelated artist
    # A stored seed reads the same columns through its own track.
    again = cons.candidates(stored, cons.seed(1, row))
    assert cons.seed(1, row).artist_id == 104 and np.array_equal(_col(again, "fwd"), fwd)
    # The columns are scene.components / affinity of each artist against the seed's, as the listener ranker has them.
    from soundalike.anysong.scene import SceneParams, affinity, components
    profile = scene.profile(104, scene.related(104))
    for a in (5, 6):
        want = (*components(profile, scene.stored(100 + a)), affinity(profile, scene.stored(100 + a), 100 + a,
                                                                       SceneParams()))
        got = c.X[by_artist[a], cm.FEATURES.index("graph"):cm.FEATURES.index("aff") + 1]
        np.testing.assert_allclose(got, want, atol=1e-6)
    # An artist the catalog and the scene table do not know, no tempo: nobody is left out, those columns are missing.
    seed = cons.seed(-1, artist="Nobody Known", artist_id=999)
    assert seed.artist is None and np.isnan(seed.bpm)
    c = cons.candidates(sig, seed)
    assert len(c.rows) == N and np.isnan(_col(c, "fwd")).all() and np.isnan(_col(c, "tempo_fold")).all()
    bodies = cons.bodies(world.bases, 1, windows=live, artist="Nobody Known")
    assert bodies["all"][0][0] == int(IDS[row]) and len(bodies["all"]) == 20


def test_consensus_rejects_tables_that_do_not_fit(tmp_path):
    world = World(tmp_path)
    assert world.cons.info() == {"model": "on", "features": 1, "sound_weight": 0.1, "sound_cap": 5000,
                                 "windows": "pinned"}
    with pytest.raises(ValueError, match="does not compute"):
        World(tmp_path, model=_constant(features=("pop",)))
    with pytest.raises(ValueError, match="does not hold"):
        World(tmp_path, orders={"melody": IDS + 1, "samples": IDS})
    with pytest.raises(ValueError, match="must cover"):
        World(tmp_path, artists=ARTISTS[:-1])
    with pytest.raises(ValueError, match="playlist vectors"):
        World(tmp_path, cf=world.cf[:-1])
    with pytest.raises(ValueError, match="sound cap"):
        World(tmp_path, sound_cap=0)


def test_engine_keeps_the_base_scores_only_when_asked(world):
    index = serve.ChannelIndex.from_windows(serve.CHANNELS["sonic"], IDS, world.win)
    eng = serve.Engine.__new__(serve.Engine)
    eng.device, eng.channels, eng.indexes = "cpu", [serve.CHANNELS["sonic"]], {"sonic": index}
    tid = int(IDS[4])
    plain, _ = eng.search_stored(tid, k=6)
    assert set(plain) == {"sonic"} and not serve.Engine.keep_base
    eng.keep_base = True
    allow = torch.ones(N, dtype=torch.bool)
    allow[:10] = False
    kept, _ = eng.search_stored(tid, k=6, allow={"sonic": allow})
    base = kept.pop("base")
    assert set(base) == {"sonic"} and torch.equal(base["sonic"], index.scores(index.stored_query(tid)))  # unmasked
    assert kept["sonic"] == index.top(index.scores(index.stored_query(tid), allow=allow), 6, [tid])
    again, _ = eng.search_stored(tid, k=6)
    assert again.pop("base") and again == plain


def test_windows_read_from_the_file_for_each_seed_give_the_same_bodies(tmp_path):
    import gc

    world = World(tmp_path)
    np.save(tmp_path / "full.npy", world.win)
    streamed = World(tmp_path, windows=serve.HostTable(tmp_path / "full.npy", "cpu", rows=16)).cons
    cons = world.cons
    assert streamed.streamed and not cons.streamed and streamed.info()["windows"] == "stream"
    live = _unit(world.win[7].astype(np.float32) + 0.02 * np.random.default_rng(3).standard_normal((W, D)))
    for kw in ({"row": 4}, {"row": 11}, {"windows": live}):
        a, b = cons.signals(world.bases, **kw), streamed.signals(world.bases, **kw)
        assert all(torch.equal(a[key], b[key]) for key in cm.SIGNALS)
    assert streamed.bodies(world.bases, 1, row=4) == cons.bodies(world.bases, 1, row=4)
    assert streamed.bodies(world.bases, 2, windows=live, artist="Artist 3") == \
        cons.bodies(world.bases, 2, windows=live, artist="Artist 3")
    # Head scores the engine computed in the blend's pass are taken as they are.
    given = torch.arange(N, dtype=torch.float32)
    assert cons.signals(world.bases | {"head": given}, 4)["head"] is given
    del streamed
    gc.collect()


def test_a_part_of_a_stored_seed_reads_only_its_windows(world):
    cons, row = world.cons, int(world.graph[0])
    whole, part = cons.signals(world.bases, row), cons.signals(world.bases, row, part=[1])
    unit = _unit(world.win.astype(np.float64))
    np.testing.assert_allclose(part["snd"].numpy(), np.einsum("d,nvd->nv", unit[row, 1], unit).max(axis=1), atol=3e-3)
    table = _unit(world.blend.table.numpy().astype(np.float64))
    np.testing.assert_allclose(part["head"].numpy(), np.einsum("d,nvd->nv", table[row, 1], table).max(axis=1),
                               atol=3e-3)
    assert (part["snd"] <= whole["snd"] + 1e-6).all() and not torch.equal(part["snd"], whole["snd"])
    # The playlist view is the song's, whichever part of it is asked.
    assert torch.equal(part["cf"], whole["cf"]) and torch.equal(part["mel"], whole["mel"])
    # Every window is the whole seed; the bodies take the part and their own sound weight.
    assert torch.equal(cons.signals(world.bases, row, part=[0, 1, 2])["snd"], whole["snd"])
    plain = cons.bodies(world.bases, -1, row=row)["all"]
    assert cons.bodies(world.bases, -1, row=row, part=[0, 1, 2])["all"] == plain
    heavy = cons.bodies(world.bases, -1, row=row, sound_weight=1.0)["all"]
    assert [t for t, _, _ in heavy] == [t for t, _, _ in plain]  # a constant model: the sound rank orders both
    assert heavy[-1][1] < plain[-1][1]


def test_a_clip_of_a_stored_song_brings_its_audio_and_keeps_the_songs_playlist_view(world):
    cons, row = world.cons, int(world.graph[0])
    other = (row + 7) % N
    clip = world.win[other].astype(np.float32)  # raw windows of other audio, as an excerpt is embedded
    own, live, mixed = cons.signals(world.bases, row), cons.signals(world.bases, windows=clip), \
        cons.signals(world.bases, row, audio=clip)
    # Sound and head are the clip's, the playlist signal the song's; melody and samples come with the searches.
    assert torch.equal(mixed["snd"], live["snd"]) and torch.equal(mixed["head"], live["head"])
    assert torch.equal(mixed["cf"], own["cf"]) and not torch.equal(mixed["snd"], own["snd"])
    assert np.isneginf(live["cf"].numpy()).all() and np.isfinite(mixed["cf"].numpy()).any()
    assert torch.equal(mixed["mel"], own["mel"])
    # The body is the stored seed's (its row, artist and tempo are left out and read as the song's) on those signals.
    body = cons.bodies(world.bases, -1, row=row, audio=clip)["all"]
    assert body == cons.body(mixed, cons.seed(-1, row))
    assert IDS[row] not in [t for t, _, _ in body]
