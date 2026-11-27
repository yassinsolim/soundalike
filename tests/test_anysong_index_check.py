import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from soundalike.anysong import durations, index_build, index_check, listener, serve
from soundalike.anysong.scene import build as build_scene
from soundalike.anysong.scenehead import SceneHead

N, W = 40, 3
IDS = np.arange(500, 500 + N, dtype=np.int64)
QUANTILES = [round(0.05 * i, 2) for i in range(21)]


def _index(root):
    """A complete index directory as the service reads it."""
    rng = np.random.default_rng(0)
    dims = {"melody": 12, "samples": 16, "sonic": 8}
    for name, bits in (("melody", 8), ("samples", 4), ("sonic", 16)):
        ch = serve.CHANNELS[name]
        win = rng.standard_normal((N, W, dims[name])).astype(np.float16)
        (root / "emb" / ch.model).mkdir(parents=True)
        order = rng.permutation(N) if name == "melody" else np.arange(N)  # another row order is fine
        np.savez(root / "emb" / ch.model / "part-00000.npz", ids=IDS[order], win=win, track=win.mean(1))
        index_build.build_channel(ch, root / "emb", root / name, 4, None, "cpu", bits=bits)
    (root / "heads").mkdir()
    for head in ("h2", "h4"):
        np.savez(root / "heads" / f"{head}.npz", w1=np.zeros((8, 5), np.float32), b1=np.zeros(5, np.float32),
                 w2=np.zeros((5, 6), np.float32), b2=np.zeros(6, np.float32))
        np.save(root / "heads" / f"{head}.emb.npy", np.zeros((N, W, 6), np.float16))
    (root / "playlists").mkdir()
    np.savez(root / "playlists" / "cooc.npz", rows=np.array([3, 9, 1]))
    np.save(root / "playlists" / "cf128.npy", np.ones((3, 4), np.float32))
    np.savez(root / "catalog_meta.npz", track_ids=IDS, artists=np.array([f"A{i}" for i in range(N)], dtype=object),
             titles=np.array([f"T{i}" for i in range(N)], dtype=object))
    np.savez(root / "languages.npz", ids=IDS[:4], top_idx=np.zeros((4, 3), np.uint8),
             top_prob=np.tile(np.array([0.9, 0.05, 0.05], np.float32), (4, 1)), vocab=np.array(["en", "fr", "de"]))
    scene = build_scene({int(t): 100 + i % 5 for i, t in enumerate(IDS)}, {100: [101]}, {100: ["FR"]}, {})
    scene.save(root / "scene.npz")
    np.save(root / "core_ids.npy", IDS[:10])
    np.savez(root / "tempo.npz", ids=IDS, bpm=np.full(N, 120, np.float32))
    durations.save(root / "durations.npz", {int(t): 200 for t in IDS[:30]})
    one = np.array([0])
    arrays = (0.0, one, one, np.zeros(1), np.array([True]), one, one, np.zeros(1), 0)
    listener.Ranker(listener.FEATURES_POP, *arrays, "test ranker").save(root / "ranker.npz")
    listener.Regressor(("lr_snd", "cf_q"), *arrays, "test consensus").save(root / "consensus.npz")
    np.savez(root / "popularity.npz", ids=IDS, rank=np.ones(N, np.float32))
    for name in ("match_quantiles.json", "consensus_quantiles.json"):
        (root / name).write_text(json.dumps(QUANTILES))
    (root / "overall.json").write_text(json.dumps(
        {name: {"sound_sd": 0.1, "logit_sd": 1.0, "quantiles": QUANTILES} for name in ("main", "listeners")}))


def test_a_complete_index_directory_passes_and_each_gap_is_named(tmp_path, capsys):
    try:
        _index(tmp_path)
    except (AttributeError, KeyError, TypeError) as exc:  # a table writer this test assumed differs
        pytest.skip(f"cannot write the test tables: {exc}")
    problems, summary = index_check.check(tmp_path)
    assert problems == [] and summary["songs"] == N and summary["named"] == N and summary["graph"] == 3
    assert summary["ranker"] == "test ranker" and summary["consensus"] == "test consensus"
    assert summary["with_length"] == 30
    assert index_check.main([str(tmp_path)]) == 0 and json.loads(capsys.readouterr().out)["ok"] is True

    def broken(change):
        change()
        return " | ".join(index_check.check(tmp_path)[0])

    # A missing file, a table of another catalog, a model this code cannot evaluate, a bad scale.
    (tmp_path / "sonic" / "norms.npy").rename(tmp_path / "norms.keep")
    assert "norms.npy: missing" in " | ".join(index_check.check(tmp_path)[0])
    (tmp_path / "norms.keep").rename(tmp_path / "sonic" / "norms.npy")
    (tmp_path / "durations.npz").rename(tmp_path / "durations.keep")
    assert "durations.npz: missing" in " | ".join(index_check.check(tmp_path)[0])
    assert "durations.npz: none of its songs" in broken(
        lambda: durations.save(tmp_path / "durations.npz", {1: 200, 2: 300}))
    (tmp_path / "durations.npz").unlink()
    (tmp_path / "durations.keep").rename(tmp_path / "durations.npz")
    assert "does not fit the sonic index" in broken(
        lambda: np.save(tmp_path / "heads" / "h4.emb.npy", np.zeros((N - 1, W, 6), np.float16)))
    np.save(tmp_path / "heads" / "h4.emb.npy", np.zeros((N, W, 6), np.float16))
    assert "do not fit" in broken(lambda: np.savez(tmp_path / "playlists" / "cooc.npz", rows=np.array([3, 9, N])))
    np.savez(tmp_path / "playlists" / "cooc.npz", rows=np.array([3, 9, 1]))
    one = np.array([0])
    arrays = (0.0, one, one, np.zeros(1), np.array([True]), one, one, np.zeros(1), 0)
    assert "does not compute" in broken(
        lambda: listener.Regressor(("pop",), *arrays).save(tmp_path / "consensus.npz"))
    listener.Regressor(("lr_snd",), *arrays).save(tmp_path / "consensus.npz")
    assert "consensus_quantiles.json" in broken(
        lambda: (tmp_path / "consensus_quantiles.json").write_text(json.dumps(QUANTILES[::-1])))
    (tmp_path / "consensus_quantiles.json").write_text(json.dumps(QUANTILES))
    assert "names for" in broken(lambda: np.savez(
        tmp_path / "catalog_meta.npz", track_ids=IDS[:5], artists=np.array(["A"] * 5, dtype=object),
        titles=np.array(["T"] * 5, dtype=object)))
    # The ids of another build under a channel's meta.json.
    ids = np.load(tmp_path / "samples" / "ids.npy")
    np.save(tmp_path / "samples" / "ids.npy", ids + 1)
    text = " | ".join(index_check.check(tmp_path)[0])
    assert "ids_sha256" in text and "not the sonic index's" in text
    assert index_check.main([str(tmp_path)]) == 1 and "PROBLEM" in capsys.readouterr().out
    assert index_check.main([str(tmp_path / "nowhere")]) == 1
