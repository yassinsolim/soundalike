import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from soundalike.anysong import index_append, index_build, serve


def shards(model_dir, ids, seed, d=16, w=3):
    rng = np.random.default_rng(seed)
    model_dir.mkdir(parents=True)
    win = rng.standard_normal((len(ids), w, d)).astype(np.float16)
    half = len(ids) // 2
    for i, rows in enumerate((slice(0, half), slice(half, None))):
        np.savez(model_dir / f"part-{i:04d}.npz", ids=np.asarray(ids[rows], np.int64), win=win[rows])
    return win


@pytest.mark.parametrize("prep,mode,bits", [("raw", "maxsim", 8), ("center", "maxsim", 4), ("none", "bpwr", 16)])
def test_append_keeps_old_rows_and_matches_a_build_with_the_same_basis(tmp_path, prep, mode, bits):
    ch = serve.Channel("test", "fake", prep, mode)
    old_ids, new_ids = list(range(1000, 1060)), list(range(5000, 5020))
    shards(tmp_path / "old" / "fake", old_ids, 0)
    # The new shards repeat an indexed id (skipped) and hold one id outside --ids (left out).
    new_win = shards(tmp_path / "new" / "fake", [*new_ids, old_ids[3], 7777], 1)
    base_meta = index_build.build_channel(ch, tmp_path / "old", tmp_path / "base", 8, None, "cpu", bits=bits)
    (tmp_path / "base" / "hub.npy").write_bytes(b"stale")
    meta = index_append.append_channel(ch, tmp_path / "base", tmp_path / "new", tmp_path / "out", set(new_ids))
    n0, m = len(old_ids), len(new_ids)
    assert meta["n"] == n0 + m and meta["appended"]["rows"] == m and meta["appended"]["base_n"] == n0
    assert meta["appended"]["base_ids_sha256"] == base_meta["ids_sha256"] and "hub" not in meta
    assert not (tmp_path / "out" / "hub.npy").exists()
    assert json.loads((tmp_path / "base" / "meta.json").read_text())["n"] == n0  # the base is left alone
    for name in ("full.npy", "proj.npy", "norms.npy", "ids.npy", *(["proj_scale.npy"] if bits < 16 else [])):
        old, new = np.load(tmp_path / "base" / name), np.load(tmp_path / "out" / name)
        assert new.dtype == old.dtype and len(new) == n0 + m and np.array_equal(new[:n0], old)
    assert np.load(tmp_path / "out" / "ids.npy")[n0:].tolist() == new_ids
    # New rows are what index_build stores for them given the base's mean and basis.
    mu = torch.from_numpy(np.load(tmp_path / "base" / "mu.npy")) if prep == "center" else None
    basis = torch.from_numpy(np.load(tmp_path / "base" / "basis.npy"))
    x16 = serve.prep_windows(torch.from_numpy(new_win[:m]).float(), prep, mu).half()
    assert np.array_equal(np.load(tmp_path / "out" / "full.npy")[n0:], x16.numpy())
    p = x16.float() @ basis
    stored = np.load(tmp_path / "out" / "proj.npy")[n0:]
    if bits == 16:
        assert np.array_equal(stored, p.half().numpy())
    else:
        q, scale = index_build.quantize_windows(p, bits)
        assert np.array_equal(stored, q.numpy())
        assert np.array_equal(np.load(tmp_path / "out" / "proj_scale.npy")[n0:], scale.numpy())
    # Served scores between old tracks are unchanged, and a new track finds itself.
    before = serve.TwoStageIndex.load(tmp_path / "base", ch, candidates=n0)
    after = serve.TwoStageIndex.load(tmp_path / "out", ch, candidates=n0 + m)
    q = before.stored_query(old_ids[7])
    assert torch.equal(after.scores(q)[:n0], before.scores(q))
    own = after.scores(after.stored_query(new_ids[5]))
    assert after.top(own, 1)[0][0] == new_ids[5]


def test_append_rejects_windows_of_another_shape(tmp_path):
    ch = serve.Channel("test", "fake", "raw", "maxsim")
    shards(tmp_path / "old" / "fake", list(range(40)), 0)
    shards(tmp_path / "new" / "fake", list(range(100, 110)), 1, d=12)
    index_build.build_channel(ch, tmp_path / "old", tmp_path / "base", 8, None, "cpu")
    with pytest.raises(ValueError, match="do not fit"):
        index_append.append_channel(ch, tmp_path / "base", tmp_path / "new", tmp_path / "out", None)
