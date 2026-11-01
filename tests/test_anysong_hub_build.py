import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from soundalike.anysong import hub_build, index_build, serve

from test_anysong_index_build import write_shards


def build(tmp_path, name="sonic", n=40, d=16):
    ch = serve.CHANNELS[name]
    ids, win = write_shards(tmp_path / "emb" / ch.model, n=n, d=d, dup=False)
    index_build.build_channel(ch, tmp_path / "emb", tmp_path / "ix" / name, d, None, "cpu")
    # Pairs of releases share a song (same artist and title, one with a parenthetical); the last track has no meta.
    artists = [f"artist{i // 2}" for i in range(n - 1)]
    titles = [f"song{i // 2}" + (" (Remastered)" if i % 2 else "") for i in range(n - 1)]
    np.savez(tmp_path / "meta.npz", track_ids=ids[:-1], artists=np.array(artists), titles=np.array(titles))
    return ch, ids, win


def reference_hub(two, ids, k):
    """Independent loop: mean of each track's k best other-song stage-1 scores."""
    song = {int(t): (i // 2 if i < len(ids) - 1 else ("solo", int(t))) for i, t in enumerate(ids)}
    out = []
    for t in ids.tolist():
        row = two.stage1(two.stored_query(t)[None])[0].numpy()
        best: dict = {}
        for j, u in enumerate(ids.tolist()):
            if song[u] != song[t]:
                best[song[u]] = max(best.get(song[u], -np.inf), float(row[j]))
        out.append(np.mean(sorted(best.values(), reverse=True)[:k]))
    return np.array(out, dtype=np.float32)


@pytest.mark.parametrize("name", ["melody", "samples", "sonic"])
def test_hub_scores_are_song_level_and_exclude_own_song(tmp_path, name):
    ch, ids, _ = build(tmp_path, name)
    hub = hub_build.build(tmp_path / "ix", name, tmp_path / "meta.npz", k=3, alpha=0.5, batch=7, device="cpu")
    meta = json.loads((tmp_path / "ix" / name / "meta.json").read_text())
    assert meta["hub"] == hub and hub["alpha"] == 0.5 and hub["k"] == 3 and hub["level"] == "song"
    h = np.load(tmp_path / "ix" / name / "hub.npy")
    two = serve.TwoStageIndex.load(tmp_path / "ix" / name, ch, hub_alpha=0)
    assert h.shape == (len(ids),) and h.dtype == np.float32
    np.testing.assert_allclose(h, reference_hub(two, ids, 3), rtol=1e-5, atol=1e-5)
    assert hub["min"] == pytest.approx(float(h.min()), abs=1e-5)


def test_far_outlier_stays_out_of_results(tmp_path):
    ch = serve.CHANNELS["melody"]
    rng = np.random.default_rng(2)
    n, d = 40, 16
    ids = np.arange(7000, 7000 + n, dtype=np.int64)
    win = rng.standard_normal((n, 3, d)).astype(np.float32)
    win[5] *= 300  # a broken embedding: far from everything, so its hub term is hugely negative
    (tmp_path / "emb" / ch.model).mkdir(parents=True)
    np.savez(tmp_path / "emb" / ch.model / "part-00000.npz", ids=ids, win=win.astype(np.float16), track=win.mean(1))
    index_build.build_channel(ch, tmp_path / "emb", tmp_path / "ix" / "melody", d, None, "cpu")
    np.savez(tmp_path / "meta.npz", track_ids=ids, artists=np.array([f"a{i}" for i in range(n)]),
             titles=np.array([f"t{i}" for i in range(n)]))
    hub_build.build(tmp_path / "ix", "melody", tmp_path / "meta.npz", k=5, alpha=0.5, device="cpu")
    assert int(np.argmin(np.load(tmp_path / "ix" / "melody" / "hub.npy"))) == 5
    served = serve.TwoStageIndex.load(tmp_path / "ix" / "melody", ch, candidates=8)
    for t in ids:
        if t != ids[5]:
            assert ids[5] not in {x for x, _ in served.top(served.scores(served.stored_query(int(t))), 10, [int(t)])}


def test_served_scores_are_csls_and_alpha_zero_disables(tmp_path):
    ch, ids, win = build(tmp_path)
    hub_build.build(tmp_path / "ix", "sonic", tmp_path / "meta.npz", k=3, alpha=0.5, batch=8, device="cpu")
    h = torch.from_numpy(np.load(tmp_path / "ix" / "sonic" / "hub.npy"))
    plain = serve.TwoStageIndex.load(tmp_path / "ix" / "sonic", ch, candidates=len(ids), hub_alpha=0)
    served = serve.TwoStageIndex.load(tmp_path / "ix" / "sonic", ch, candidates=len(ids))
    assert plain.penalty is None and served.hub_alpha == 0.5
    q = torch.stack([plain.stored_query(int(ids[3])), plain.prep_query(win[8].astype(np.float32) + 0.3)])
    assert torch.allclose(served.scores(q), plain.scores(q) - 0.5 * h, atol=1e-5)
    # With a short candidate list the penalty is applied before candidates are picked.
    short = serve.TwoStageIndex.load(tmp_path / "ix" / "sonic", ch, candidates=5)
    want = (plain.stage1(q) - 0.5 * h)[0].topk(5).indices
    assert set(short.scores(q)[0].topk(5).indices.tolist()) == set(want.tolist())
    assert serve.Engine(tmp_path / "ix", ["sonic"], "cpu", hub_alpha=0).indexes["sonic"].penalty is None
    assert serve.Engine(tmp_path / "ix", ["sonic"], "cpu").indexes["sonic"].hub_alpha == 0.5


def test_penalty_demotes_a_hub(tmp_path):
    ch = serve.CHANNELS["sonic"]
    rng = np.random.default_rng(1)
    n, d = 30, 64
    ids = np.arange(5000, 5000 + n, dtype=np.int64)
    win = rng.standard_normal((n, 3, d)).astype(np.float32)
    win /= np.linalg.norm(win, axis=-1, keepdims=True)
    # Track 0's first window is a direction u that every other track's first window leans towards (cos 0.8),
    # so track 0 is the nearest neighbour of everything: a hub.
    u = win[0, 0]
    rest = win[1:, 0] - np.outer(win[1:, 0] @ u, u)
    win[1:, 0] = 0.8 * u + 0.6 * rest / np.linalg.norm(rest, axis=1, keepdims=True)
    (tmp_path / "emb" / ch.model).mkdir(parents=True)
    np.savez(tmp_path / "emb" / ch.model / "part-00000.npz", ids=ids, win=win.astype(np.float16), track=win.mean(1))
    index_build.build_channel(ch, tmp_path / "emb", tmp_path / "ix" / "sonic", d, None, "cpu")
    np.savez(tmp_path / "meta.npz", track_ids=ids, artists=np.array([f"a{i}" for i in range(n)]),
             titles=np.array([f"t{i}" for i in range(n)]))
    hub_build.build(tmp_path / "ix", "sonic", tmp_path / "meta.npz", k=5, alpha=1.0, device="cpu")
    assert int(np.argmax(np.load(tmp_path / "ix" / "sonic" / "hub.npy"))) == 0

    def times_first(ix):
        return sum(ix.top(ix.scores(ix.stored_query(int(t))), 1, [int(t)])[0][0] == ids[0] for t in ids[1:])

    plain = serve.TwoStageIndex.load(tmp_path / "ix" / "sonic", ch, candidates=n, hub_alpha=0)
    served = serve.TwoStageIndex.load(tmp_path / "ix" / "sonic", ch, candidates=n)
    assert times_first(served) < times_first(plain)


def test_alpha_without_hub_file_is_an_error(tmp_path):
    ch, _, _ = build(tmp_path)
    with pytest.raises(FileNotFoundError):
        serve.TwoStageIndex.load(tmp_path / "ix" / "sonic", ch, hub_alpha=0.5)
    assert serve.TwoStageIndex.load(tmp_path / "ix" / "sonic", ch).penalty is None


def test_cli_defaults_leave_melody_off(tmp_path):
    for name in ("melody", "samples"):
        build(tmp_path, name)
    hub_build.main(["--index", str(tmp_path / "ix"), "--meta", str(tmp_path / "meta.npz"),
                    "--channels", "melody", "samples", "--k", "3", "--device", "cpu"])
    for name, alpha in (("melody", 0.0), ("samples", 0.5)):
        assert (tmp_path / "ix" / name / "hub.npy").exists()
        assert json.loads((tmp_path / "ix" / name / "meta.json").read_text())["hub"]["alpha"] == alpha
        ix = serve.TwoStageIndex.load(tmp_path / "ix" / name, serve.CHANNELS[name])
        assert ix.hub_alpha == alpha and (ix.penalty is None) == (alpha == 0)
    on = serve.TwoStageIndex.load(tmp_path / "ix" / "melody", serve.CHANNELS["melody"], hub_alpha=0.5)
    assert on.penalty is not None
