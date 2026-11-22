import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from soundalike.anysong import index_append, index_build, index_whiten, serve
from soundalike.anysong.blend import Blend, zscore

N, W, D = 60, 3, 16
IDS = np.arange(1000, 1000 + N, dtype=np.int64)
SONIC = serve.CHANNELS["sonic"]


def _unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _windows(n, seed):
    """Unit windows whose variance sits in a few directions around a common offset, as MuQ-MuLan's does."""
    rng = np.random.default_rng(seed)
    spread = np.geomspace(4.0, 0.05, D)
    return _unit(rng.standard_normal((n, W, D)) * spread + 3.0).astype(np.float32)


def _raw_index(tmp_path, win, ids=IDS, name="emb"):
    (tmp_path / name / SONIC.model).mkdir(parents=True)
    np.savez(tmp_path / name / SONIC.model / "part-00000.npz", ids=ids, win=win.astype(np.float16))
    return tmp_path / name


@pytest.fixture
def built(tmp_path):
    win = _windows(N, 0)
    index_build.build_channel(SONIC, _raw_index(tmp_path, win), tmp_path / "v9" / "sonic", 6, None, "cpu", bits=8)
    meta = index_whiten.build(tmp_path / "v9" / "sonic", tmp_path / "v11" / "sonic", seed=3)
    return tmp_path, win, meta


def _whitened(raw, path):
    mu, m = (torch.from_numpy(np.load(path / f)) for f in ("mu.npy", "whiten.npy"))
    return serve.prep_windows(torch.as_tensor(raw).float(), "whiten", mu, m)


def _maxsim(q, x):
    return torch.einsum("wd,nvd->nwv", q, x).amax(dim=(1, 2))


def test_fit_is_seeded_and_gives_unit_variance_in_every_direction():
    win = _windows(400, 1).astype(np.float16)
    mean, matrix, evals = index_whiten.fit_whitening(win, windows=900, seed=5)
    again = index_whiten.fit_whitening(win, windows=900, seed=5)
    assert np.array_equal(mean, again[0]) and np.array_equal(matrix, again[1])
    assert not np.array_equal(matrix, index_whiten.fit_whitening(win, windows=900, seed=6)[1])
    assert evals[0] > 100 * evals[-1] > 0 and (np.diff(evals) <= 0).all()
    # The sample it was fitted on: zero mean, identity covariance.
    pick = np.sort(np.random.default_rng(5).choice(400 * W, 900, replace=False))
    y = (_unit(win.reshape(-1, D)[pick].astype(np.float64)) - mean) @ matrix
    np.testing.assert_allclose(y.mean(0), 0, atol=1e-3)
    np.testing.assert_allclose(np.cov(y.T), np.eye(D), atol=2e-2)
    with pytest.raises(ValueError, match="more than"):
        index_whiten.fit_whitening(win[:5])
    flat = win.copy()
    flat[..., -1] = 0  # no variance in one direction
    with pytest.raises(ValueError, match="singular"):
        index_whiten.fit_whitening(flat)


def test_whitened_index_stores_the_transform_and_scores_in_its_space(built):
    tmp_path, win, meta = built
    path = tmp_path / "v11" / "sonic"
    assert meta["prep"] == "whiten" and meta["proj_dim"] == D and meta["proj_bits"] == 8 and "hub" not in meta
    assert meta["whiten"]["seed"] == 3 and meta["whiten"]["fit_windows"] == N * W
    assert meta == json.loads((path / "meta.json").read_text())
    assert np.load(path / "mu.npy").shape == (D,) and np.load(path / "whiten.npy").shape == (D, D)
    assert np.load(path / "proj.npy").dtype == np.int8 and not (path / "hub.npy").exists()
    assert np.array_equal(np.load(path / "ids.npy"), IDS)
    # The base still holds the raw windows; the new index holds their whitened form.
    raw = np.load(tmp_path / "v9" / "sonic" / "full.npy")
    assert np.array_equal(raw, _unit(win.astype(np.float16).astype(np.float32)).astype(np.float16))
    want = _whitened(raw, path)
    assert np.array_equal(np.load(path / "full.npy"), want.half().numpy())
    assert not np.allclose(want.numpy(), raw.astype(np.float32), atol=0.05)

    two = serve.TwoStageIndex.load(path, SONIC, candidates=N)
    assert two.channel == serve.Channel("sonic", "muq_mulan", "whiten", "maxsim") and two.penalty is None
    # A live query: raw windows in, whitened out; a stored one is whitened already.
    live = _unit(win[7] + 0.0005 * np.random.default_rng(2).standard_normal((W, D))).astype(np.float32)
    q = two.prep_query(live)
    assert torch.allclose(q, _whitened(live, path), atol=1e-6) and torch.allclose(q.norm(dim=-1), torch.ones(W))
    assert torch.allclose(two.prep_query(3.0 * live), q, atol=1e-5)  # the length of raw windows does not matter
    assert torch.equal(two.stored_query(int(IDS[7])), want[7].half().float())
    full = torch.from_numpy(np.load(path / "full.npy")).float()
    for query in (q, two.stored_query(int(IDS[7]))):
        assert torch.allclose(two.scores(query), _maxsim(query, full), atol=1e-5)
    assert two.top(two.scores(q), 1)[0][0] == int(IDS[7])
    # Stage 1 holds every dimension: 8 bits lose almost nothing, so a short candidate list finds the exact top.
    exact = _maxsim(q, full)
    assert torch.allclose(two.stage1(q[None])[0], exact, atol=5e-3)
    short = serve.TwoStageIndex.load(path, SONIC, candidates=12)
    assert [t for t, _ in short.top(short.scores(q), 5)] == [t for t, _ in two.top(exact, 5)]
    half = serve.TwoStageIndex.load(path, SONIC, candidates=12, half_stage1=True)
    assert [t for t, _ in half.top(half.scores(q), 5)] == [t for t, _ in two.top(exact, 5)]
    del two, short, half  # releases the memory maps of full.npy


def test_engine_feeds_raw_windows_to_the_head_and_whitened_ones_to_the_scores(built):
    tmp_path, win, _ = built
    rng = np.random.default_rng(4)
    np.savez(tmp_path / "h.npz", w1=rng.standard_normal((D, 8)).astype(np.float32),
             b1=rng.standard_normal(8).astype(np.float32), w2=rng.standard_normal((8, 6)).astype(np.float32),
             b2=rng.standard_normal(6).astype(np.float32))
    np.save(tmp_path / "h.emb.npy", _unit(rng.standard_normal((N, W, 6))).astype(np.float16))
    graph = np.array([9, 0, 4, 7, 2, 12], dtype=np.int64)
    np.savez(tmp_path / "cooc.npz", rows=graph)
    np.save(tmp_path / "cf128.npy", rng.standard_normal((len(graph), 5)).astype(np.float32))

    engines = {}
    for name in ("v9", "v11"):
        eng = serve.Engine(tmp_path / name, ["sonic"], "cpu", candidates=N)
        eng.blend = Blend.load(tmp_path / "h", tmp_path, eng.indexes["sonic"].pos)
        engines[name] = eng
    old, new = engines["v9"], engines["v11"]
    assert old.indexes["sonic"].channel.prep == "raw" and new.indexes["sonic"].channel.prep == "whiten"
    ix = new.indexes["sonic"]
    live = _unit(win[4] + 0.0005 * rng.standard_normal((W, D))).astype(np.float32)
    seen = []
    embed = new.blend.head.embed
    new.blend.head.embed = lambda windows: (seen.append(np.array(windows)), embed(windows))[1]

    # A live seed: scores from the whitened windows, head vectors from the raw ones (as on the raw index).
    results, _ = new.search_windows({"sonic": live}, k=5)
    assert len(seen) == 1 and np.array_equal(seen[0], live)
    q = ix.prep_query(live)
    assert results["sonic"] == ix.top(ix.scores(q), 5) and results["sonic"][0][0] == int(IDS[4])
    seed_head = new.blend.seed_head(None, live)
    assert torch.equal(seed_head, old.blend.seed_head(None, live))
    assert torch.equal(new.blend.head_scores(seed_head), old.blend.head_scores(seed_head))
    blended = zscore(ix.stage1(q[None])[0]) + 2.0 * zscore(new.blend.head_scores(seed_head))
    assert [t for t, _ in results["blend"]] == [t for t, _ in ix.top(blended, 5)]
    assert results["sonic"] != old.search_windows({"sonic": live}, k=5)[0]["sonic"]  # another space, other scores
    assert new.score_tracks("sonic", IDS[:9], windows=live) == pytest.approx(
        {int(t): float(s) for t, s in zip(IDS[:9], ix.scores(q)[:9])}, abs=1e-6)

    # A stored seed: its whitened windows and the head table's row; the head never sees stored windows.
    tid = int(IDS[4])
    results, _ = new.search_stored(tid, k=5)
    assert len(seen) == 2  # only the live seed_head call above
    sq = ix.stored_query(tid)
    assert results["sonic"] == ix.top(ix.scores(sq), 5, [tid])
    assert torch.equal(new.blend.seed_head(4), old.blend.seed_head(4))
    blended = zscore(ix.stage1(sq[None])[0]) + 2.0 * zscore(new.blend.head_scores(new.blend.seed_head(4))) \
        + new.blend.cf_scores(4)
    assert [t for t, _ in results["blend"]] == [t for t, _ in ix.top(blended, 5, [tid])]
    assert new.score_tracks("sonic", IDS[:9], query_track=tid) == pytest.approx(
        {int(t): float(s) for t, s in zip(IDS[:9], ix.scores(sq)[:9])}, abs=1e-6)
    del engines, old, new, ix


def test_append_whitens_new_tracks_with_the_stored_transform(built):
    tmp_path, _, _ = built
    new_ids = np.arange(5000, 5010, dtype=np.int64)
    new_win = _windows(len(new_ids), 9)
    base, out = tmp_path / "v11" / "sonic", tmp_path / "grown"
    meta = index_append.append_channel(SONIC, base, _raw_index(tmp_path, new_win, new_ids, "new"), out, None)
    assert meta["prep"] == "whiten" and meta["n"] == N + len(new_ids)
    for name in ("mu.npy", "whiten.npy", "basis.npy"):
        assert np.array_equal(np.load(out / name), np.load(base / name))
    full = np.load(out / "full.npy")
    assert np.array_equal(full[:N], np.load(base / "full.npy"))
    assert np.array_equal(full[N:], _whitened(new_win.astype(np.float16), base).half().numpy())
    two = serve.TwoStageIndex.load(out, SONIC, candidates=5)
    assert two.top(two.scores(two.prep_query(new_win[3])), 1)[0][0] == int(new_ids[3])
    del two


def test_only_raw_channels_are_whitened_and_the_matrix_must_fit(built, tmp_path):
    path = tmp_path / "v11" / "sonic"
    with pytest.raises(ValueError, match="starts from a raw channel"):
        index_whiten.build(path, tmp_path / "twice")
    meta = json.loads((path / "meta.json").read_text())
    assert serve.built_channel(SONIC, meta).prep == "whiten"
    assert serve.built_channel(serve.CHANNELS["samples"], meta) is None
    assert serve.built_channel(serve.Channel("sonic", "muq_mulan", "center", "maxsim"), meta) is None
    with pytest.raises(ValueError, match="built for"):
        serve.TwoStageIndex.load(path, serve.CHANNELS["samples"])
    two = serve.TwoStageIndex.load(path, SONIC)
    args = (two.ids, two.proj, two.basis, two.norms, two.full, two.mu, 5, None, 0.0, two.proj_scale)
    with pytest.raises(ValueError, match="whitening matrix"):
        serve.TwoStageIndex(two.channel, *args)
    with pytest.raises(ValueError, match="whitening matrix"):
        serve.TwoStageIndex(SONIC, *args, whiten=two.whiten)
    with pytest.raises(ValueError, match="whitening matrix"):
        serve.TwoStageIndex(two.channel, *args, whiten=two.whiten[:, :4])
    with pytest.raises(ValueError, match="whiten prep needs"):
        serve.prep_windows(torch.ones(2, D), "whiten", two.mu)
    del two
