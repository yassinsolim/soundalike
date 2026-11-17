import numpy as np
import pytest

torch = pytest.importorskip("torch")
from soundalike.anysong import index_build, serve
from soundalike.anysong.blend import Blend
from soundalike.anysong.scenehead import SceneHead

N, W, D = 14, 3, 16
IDS = np.arange(100, 100 + N, dtype=np.int64)
GRAPH = np.array([9, 0, 4, 7, 2, 12], dtype=np.int64)  # sonic-index rows of the playlist graph, in cf order


def _unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _files(tmp_path, seed=0):
    """Sonic windows, a scene head with its table, and playlist vectors for the GRAPH rows."""
    rng = np.random.default_rng(seed)
    win = _unit(rng.standard_normal((N, W, D))).astype(np.float32)
    np.savez(tmp_path / "h.npz", w1=rng.standard_normal((D, 8)).astype(np.float32),
             b1=rng.standard_normal(8).astype(np.float32), w2=rng.standard_normal((8, 6)).astype(np.float32),
             b2=rng.standard_normal(6).astype(np.float32))
    np.save(tmp_path / "h.emb.npy", np.zeros((N, W, 6), np.float16))
    head = SceneHead(tmp_path / "h.npz", tmp_path / "h.emb.npy", {int(t): i for i, t in enumerate(IDS)})
    table = np.stack([head.embed(w) for w in win]).astype(np.float16)
    del head  # releases the memory map, which Windows will not replace while it is open
    np.save(tmp_path / "h.emb.npy", table)
    cf = rng.standard_normal((len(GRAPH), 5)).astype(np.float32) * 3  # not unit: the loader normalises
    np.savez(tmp_path / "cooc.npz", rows=GRAPH)
    np.save(tmp_path / "cf128.npy", cf)
    return win, table.astype(np.float32), cf


def _engine(index):
    eng = serve.Engine.__new__(serve.Engine)
    eng.device, eng.channels, eng.indexes = "cpu", [serve.CHANNELS["sonic"]], {"sonic": index}
    return eng


def _z(x):
    return (x - x.mean()) / x.std(ddof=1)


def _zcf(cf, seed_row):
    """Hand-computed playlist term: cosines z-scored over the graph songs, 0 elsewhere and for seeds outside."""
    out = np.zeros(N)
    if seed_row in GRAPH:
        unit = _unit(cf.astype(np.float64))
        c = unit @ unit[list(GRAPH).index(seed_row)]
        out[GRAPH] = (c - c.mean()) / c.std()
    return out


def _head_cos(table, seed_vecs):
    return np.einsum("wd,nvd->nwv", seed_vecs, table).max(axis=(1, 2))


@pytest.fixture
def setup(tmp_path):
    win, table, cf = _files(tmp_path)
    index = serve.ChannelIndex.from_windows(serve.CHANNELS["sonic"], IDS, win.astype(np.float16))
    eng = _engine(index)
    eng.blend = Blend.load(tmp_path / "h", tmp_path, index.pos, weights=(2.0, 1.0))
    return eng, index, win, table, cf


def _expected(index, sonic, table, seed_vecs, cf, seed_row, drop=(), wh=2.0, wc=1.0):
    score = _z(sonic) + wh * _z(_head_cos(table, seed_vecs)) + wc * _zcf(cf, seed_row)
    order = [int(i) for i in np.argsort(-score) if int(i) not in set(drop)]
    return [int(index.ids[i]) for i in order], score


def test_blend_order_is_the_z_sum_and_carries_exact_sonic_scores(setup):
    eng, index, _, table, cf = setup
    row = 4  # in the playlist graph
    sonic = index.scores(index.stored_query(int(IDS[row]))).numpy().astype(np.float64)
    want, score = _expected(index, sonic, table, table[row], cf, row, drop=[row])
    got = eng.blend.scores(torch.from_numpy(sonic).float(), eng.blend.seed_head(row), row).numpy()
    np.testing.assert_allclose(got, score, atol=1e-4)
    assert _zcf(cf, row).any() and want != [int(index.ids[i]) for i in np.argsort(-sonic) if i != row]

    results, timings = eng.search_stored(int(IDS[row]), k=6)
    assert [t for t, _ in results["blend"]] == want[:6] and "search_blend" in timings
    for tid, s in results["blend"]:  # the sonic channel's own exact score, not the blended one
        assert s == pytest.approx(sonic[index.pos[tid]], abs=1e-6)
    assert results["sonic"] == index.top(index.scores(index.stored_query(int(IDS[row]))), 6, [int(IDS[row])])
    # The blended score of every song of the two lists, for orders that mix them.
    blendz = dict(results["blendz"])
    assert set(blendz) == {t for t, _ in results["sonic"]} | {t for t, _ in results["blend"]}
    assert len(blendz) > 6 and all(z == pytest.approx(score[index.pos[t]], abs=1e-4) for t, z in blendz.items())
    # ``exclude`` and the sonic ``allow`` mask apply to the blended ranking as to the channel.
    allow = torch.ones(N, dtype=torch.bool)
    allow[[index.pos[want[0]], index.pos[want[2]]]] = False
    masked, _ = eng.search_stored(int(IDS[row]), k=N, exclude=[want[1]], allow={"sonic": allow})
    assert [t for t, _ in masked["blend"]] == want[3:]
    assert {t for t, _ in masked["sonic"]} == set(want[3:])
    # Other weights; without a blend the engine returns the channels alone.
    eng.blend.wh, eng.blend.wc = 0.5, 3.0
    assert [t for t, _ in eng.search_stored(int(IDS[row]), k=N)[0]["blend"]] == \
        _expected(index, sonic, table, table[row], cf, row, drop=[row], wh=0.5, wc=3.0)[0]
    eng.blend = None
    plain, timings = eng.search_stored(int(IDS[row]), k=6)
    assert set(plain) == {"sonic"} and plain["sonic"] == results["sonic"] and "search_blend" not in timings


def test_seed_outside_the_playlist_graph_and_live_seeds_have_no_playlist_term(setup):
    eng, index, win, table, cf = setup
    row = 5
    assert row not in GRAPH and eng.blend.cf_scores(row) is None and eng.blend.cf_scores(None) is None
    sonic = index.scores(index.stored_query(int(IDS[row]))).numpy().astype(np.float64)
    want, _ = _expected(index, sonic, table, table[row], cf, row, drop=[row], wc=0.0)
    assert [t for t, _ in eng.search_stored(int(IDS[row]), k=N)[0]["blend"]] == want
    in_graph = eng.blend.cf_scores(4).numpy()
    assert not in_graph[[i for i in range(N) if i not in GRAPH]].any() and in_graph[GRAPH].any()
    np.testing.assert_allclose(in_graph, _zcf(cf, 4), atol=1e-4)

    # A live seed: head vectors from its raw sonic windows, no playlist term even for a song of the graph.
    live = _unit(win[4] + 0.05 * np.random.default_rng(1).standard_normal((W, D))).astype(np.float32)
    sonic = index.scores(index.prep_query(live)).numpy().astype(np.float64)
    want, _ = _expected(index, sonic, table, eng.blend.head.embed(live), cf, None, wc=0.0)
    results, timings = eng.search_windows({"sonic": live}, k=5, exclude=[want[1]])
    assert [t for t, _ in results["blend"]] == [want[0], *want[2:6]] and "search_blend" in timings
    for tid, s in results["blend"]:
        assert s == pytest.approx(sonic[index.pos[tid]], abs=1e-6)


def test_two_stage_index_blends_stage_one_scores_and_attaches_exact_ones(tmp_path):
    win, table, cf = _files(tmp_path)
    ch = serve.CHANNELS["sonic"]
    (tmp_path / "emb" / ch.model).mkdir(parents=True)
    np.savez(tmp_path / "emb" / ch.model / "part-00000.npz", ids=IDS, win=win.astype(np.float16), track=win.mean(1))
    index_build.build_channel(ch, tmp_path / "emb", tmp_path / "ix" / "sonic", 6, None, "cpu")
    hub = torch.linspace(0.0, 0.3, N)
    two = serve.TwoStageIndex.load(tmp_path / "ix" / "sonic", ch, candidates=3)
    two.hub_alpha, two.penalty = 0.5, 0.5 * hub
    q = two.stored_query(int(IDS[4]))
    scores, base = two.scores_with_base(q)
    stage1 = two.stage1(q[None])[0] - 0.5 * hub
    assert torch.allclose(base, stage1, atol=1e-6) and torch.equal(scores, two.scores(q))
    exact = two.score_rows(q, np.arange(N))
    assert not torch.allclose(stage1, exact, atol=1e-3)  # the short projection only approximates

    eng = _engine(two)
    eng.blend = Blend.load(tmp_path / "h", tmp_path, two.pos)
    want, _ = _expected(two, stage1.numpy().astype(np.float64), table, table[4], cf, 4, drop=[4])
    results, _ = eng.search_stored(int(IDS[4]), k=8)
    assert [t for t, _ in results["blend"]] == want[:8]  # deeper than the three exact candidates
    for tid, s in results["blend"]:
        assert s == pytest.approx(float(exact[two.pos[tid]]), abs=1e-6)
    assert results["sonic"] == two.top(two.scores(q), 8, [int(IDS[4])])
    del two, eng  # releases the memory map of full.npy


def test_blend_rejects_files_that_do_not_fit(tmp_path):
    _, table, cf = _files(tmp_path)
    pos = {int(t): i for i, t in enumerate(IDS)}
    blend = Blend.load(tmp_path / "h", tmp_path, pos)
    assert blend.info() == {"songs": N, "graph": len(GRAPH), "weights": [2.0, 1.0]}
    assert torch.allclose(blend.cf.norm(dim=1), torch.ones(len(GRAPH)), atol=1e-5)
    with pytest.raises(ValueError, match="does not fit"):
        Blend.load(tmp_path / "h", tmp_path, {1: 0})
    with pytest.raises(ValueError, match="do not fit"):
        Blend(blend.head, blend.table, GRAPH[:3], cf)
    with pytest.raises(ValueError, match="distinct rows"):
        Blend(blend.head, blend.table, np.array([0, 0, 1, 2, 3, N]), cf)
