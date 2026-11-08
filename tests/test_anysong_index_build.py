import gc
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from soundalike.anysong import index_build, serve


def channel(prep="raw", mode="maxsim", model="fake"):
    return serve.Channel("test", model, prep, mode)


def write_shards(model_dir, n=60, w=3, d=16, seed=0, dup=True, rank=None):
    """Two shards; the second repeats one ID from the first (with different windows) and itself.
    ``rank`` draws windows near a rank-``rank`` subspace so a short projection keeps most energy."""
    rng = np.random.default_rng(seed)
    model_dir.mkdir(parents=True)
    ids = np.arange(1000, 1000 + n, dtype=np.int64)
    if rank is None:
        win = rng.standard_normal((n, w, d)).astype(np.float16)
    else:
        win = (rng.standard_normal((n, w, rank)) @ rng.standard_normal((rank, d))
               + 0.05 * rng.standard_normal((n, w, d))).astype(np.float16)
    half = n // 2
    a_ids, a_win = ids[:half], win[:half]
    b_ids, b_win = ids[half:], win[half:]
    if dup:
        b_ids = np.concatenate([b_ids, [ids[0], ids[half]]])
        b_win = np.concatenate([b_win, rng.standard_normal((2, w, d)).astype(np.float16)])
    for i, (si, sw) in enumerate(((a_ids, a_win), (b_ids, b_win))):
        np.savez(model_dir / f"part-{i:05d}.npz", ids=si, win=sw, track=sw.astype(np.float32).mean(1))
    return ids, win


@pytest.mark.parametrize("prep,mode", [("raw", "maxsim"), ("center", "maxsim"), ("none", "bpwr")])
def test_two_stage_equals_brute_force_with_full_candidates(tmp_path, prep, mode):
    ch = channel(prep, mode)
    ids, win = write_shards(tmp_path / "emb" / "fake")
    meta = index_build.build_channel(ch, tmp_path / "emb", tmp_path / "ix", 16, None, "cpu")
    assert meta["n"] == len(ids) and meta["energy"] == pytest.approx(1.0, abs=1e-6)
    brute = serve.ChannelIndex.load(tmp_path / "emb", ch)
    two = serve.TwoStageIndex.load(tmp_path / "ix", ch, candidates=len(ids))
    assert np.array_equal(two.ids, brute.ids) and np.array_equal(two.ids, ids)
    if prep == "center":
        assert torch.allclose(two.mu, brute.mu, atol=1e-6)
    for t in (int(ids[0]), int(ids[37])):
        assert torch.equal(two.stored_query(t), brute.stored_query(t))
    q = torch.stack([brute.stored_query(int(ids[3])), brute.prep_query(win[8].astype(np.float32) + 0.3)])
    exact, staged = brute.scores(q), two.scores(q, chunk=7)
    assert torch.allclose(staged, exact, atol=1e-5)
    for b in range(2):
        got, want = two.top(staged[b], 10, [int(ids[3])]), brute.top(exact[b], 10, [int(ids[3])])
        assert [t for t, _ in got] == [t for t, _ in want]
        assert [s for _, s in got] == pytest.approx([s for _, s in want], abs=1e-5)


@pytest.mark.parametrize("prep,mode", [("raw", "maxsim"), ("none", "bpwr")])
def test_candidates_are_exact_and_rank_above_estimates(tmp_path, prep, mode):
    ch = channel(prep, mode)
    ids, _ = write_shards(tmp_path / "emb" / "fake", n=200, d=32, dup=False, rank=6)
    index_build.build_channel(ch, tmp_path / "emb", tmp_path / "ix", 6, None, "cpu")
    brute = serve.ChannelIndex.load(tmp_path / "emb", ch)
    two = serve.TwoStageIndex.load(tmp_path / "ix", ch, candidates=25)
    q = brute.stored_query(int(ids[11]))
    exact, staged = brute.scores(q), two.scores(q)
    cand = two.stage1(q[None])[0].topk(25).indices
    mask = torch.zeros(len(ids), dtype=torch.bool)
    mask[cand] = True
    assert torch.allclose(staged[mask], exact[mask], atol=1e-5)
    assert staged[~mask].max() < staged[mask].min()
    # The query's own track survives a 6-of-32 dim projection of near rank-6 data and is ranked first exactly.
    assert two.top(staged, 1)[0][0] == int(ids[11])
    assert two.rank_of(staged, [int(ids[11])])[0] == 1


@pytest.mark.parametrize("prep,mode", [("raw", "maxsim"), ("none", "bpwr")])
def test_allow_mask_picks_candidates_among_allowed_rows_only(tmp_path, prep, mode):
    ch = channel(prep, mode)
    ids, _ = write_shards(tmp_path / "emb" / "fake", n=200, d=32, dup=False, rank=6)
    index_build.build_channel(ch, tmp_path / "emb", tmp_path / "ix", 6, None, "cpu")
    brute = serve.ChannelIndex.load(tmp_path / "emb", ch)
    allow = torch.zeros(len(ids), dtype=torch.bool)
    allow[::3] = True
    q = brute.stored_query(int(ids[11]))
    exact = brute.scores(q, allow=allow)
    assert torch.isinf(exact[~allow]).all() and torch.equal(exact[allow], brute.scores(q)[allow])
    full = serve.TwoStageIndex.load(tmp_path / "ix", ch, candidates=len(ids)).scores(q, allow=allow)
    assert torch.isinf(full[~allow]).all() and torch.allclose(full[allow], exact[allow], atol=1e-5)
    two = serve.TwoStageIndex.load(tmp_path / "ix", ch, candidates=25)
    staged = two.scores(q, allow=allow)
    cand = two.stage1(q[None])[0].masked_fill(~allow, -torch.inf).topk(25).indices
    picked = torch.zeros(len(ids), dtype=torch.bool)
    picked[cand] = True
    assert bool(allow[cand].all())  # all 25 exact re-scores went to allowed rows
    assert torch.allclose(staged[picked], exact[picked], atol=1e-5)
    assert staged[allow & ~picked].max() < staged[picked].min() and torch.isinf(staged[~allow]).all()
    assert all(bool(allow[two.pos[t]]) for t, _ in two.top(staged, 50))
    # Fewer allowed rows than candidates or k: only those rows come back, exactly scored.
    few = torch.zeros(len(ids), dtype=torch.bool)
    few[[4, 40, 140]] = True
    got = two.top(two.scores(q, allow=few), 10)
    assert sorted(t for t, _ in got) == sorted(int(ids[i]) for i in (4, 40, 140))
    assert [s for _, s in got] == pytest.approx([s for _, s in brute.top(brute.scores(q, allow=few), 10)], abs=1e-5)
    # Batched queries take one mask per query.
    qs = torch.stack([q, brute.stored_query(int(ids[12]))])
    masks = torch.stack([allow, few])
    rows = two.scores(qs, allow=masks)
    assert torch.equal(torch.isfinite(rows), masks)


def test_keep_filter_and_first_occurrence(tmp_path):
    ids, win = write_shards(tmp_path / "emb" / "fake", n=10, d=8)
    keep = {int(ids[0]), int(ids[5]), int(ids[9])}
    index_build.build_channel(channel(), tmp_path / "emb", tmp_path / "ix", 4, keep, "cpu")
    two = serve.TwoStageIndex.load(tmp_path / "ix", channel())
    assert two.ids.tolist() == sorted(keep)
    # ids[0] repeats in the next shard and ids[5] within it; the first occurrence must win.
    expected = serve.prep_windows(torch.from_numpy(win[[0, 5, 9]]).float(), "raw").half().float()
    assert torch.equal(torch.from_numpy(np.load(tmp_path / "ix" / "full.npy")).float(), expected)


def test_engine_loads_built_index_and_rejects_mismatched_channel(tmp_path):
    samples = serve.CHANNELS["samples"]
    write_shards(tmp_path / "emb" / samples.model, n=12, d=8)
    index_build.build_channel(samples, tmp_path / "emb", tmp_path / "ix" / "samples", 4, None, "cpu", candidates=7)
    assert serve.Engine(tmp_path / "ix", ["samples"], "cpu").indexes["samples"].candidates == 7
    eng = serve.Engine(tmp_path / "ix", ["samples"], "cpu", candidates=5)
    assert isinstance(eng.indexes["samples"], serve.TwoStageIndex)
    assert eng.indexes["samples"].candidates == 5
    assert isinstance(serve.Engine(tmp_path / "emb", ["samples"], "cpu").indexes["samples"], serve.ChannelIndex)
    meta = json.loads((tmp_path / "ix" / "samples" / "meta.json").read_text())
    assert meta["proj_dim"] == 4 and meta["prep"] == "center"
    with pytest.raises(ValueError, match="built for"):
        serve.TwoStageIndex.load(tmp_path / "ix" / "samples", serve.CHANNELS["sonic"])


def test_build_rejects_bad_projection_dim(tmp_path):
    write_shards(tmp_path / "emb" / "fake", n=4, d=8)
    with pytest.raises(ValueError, match="projection dim"):
        index_build.build_channel(channel(), tmp_path / "emb", tmp_path / "ix", 9, None, "cpu")


@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("prep,mode", [("raw", "maxsim"), ("center", "maxsim"), ("none", "bpwr")])
def test_quantized_projection_decodes_within_half_a_step_and_reranks_exactly(tmp_path, prep, mode, bits):
    ch = channel(prep, mode)
    ids, win = write_shards(tmp_path / "emb" / "fake", n=120, d=16, dup=False, rank=6)
    meta = index_build.build_channel(ch, tmp_path / "emb", tmp_path / "ix", 8, None, "cpu", bits=bits)
    index_build.build_channel(ch, tmp_path / "emb", tmp_path / "ix16", 8, None, "cpu")
    assert meta["proj_bits"] == bits
    proj = np.load(tmp_path / "ix" / "proj.npy")
    assert proj.dtype == (np.int8 if bits == 8 else np.uint8) and proj.shape == (len(ids), 3, 8 if bits == 8 else 4)
    two = serve.TwoStageIndex.load(tmp_path / "ix", ch, candidates=len(ids))
    assert two.proj_bits == bits and tuple(two.proj_scale.shape) == (len(ids), 3)
    decoded = two._proj_rows(0, len(ids)) * two.proj_scale[..., None]
    assert decoded.abs().amax(-1).allclose(two.proj_scale * (2 ** (bits - 1) - 1))
    reference = torch.from_numpy(np.load(tmp_path / "ix16" / "proj.npy")).float()
    assert ((decoded - reference).abs() <= two.proj_scale[..., None] / 2 + 1e-3 * reference.abs() + 1e-6).all()
    brute = serve.ChannelIndex.load(tmp_path / "emb", ch)
    q = torch.stack([brute.stored_query(int(ids[3])), brute.prep_query(win[8].astype(np.float32) + 0.3)])
    plain = serve.TwoStageIndex(ch, two.ids, decoded, two.basis, two.norms, two.full, two.mu, len(ids))
    assert torch.allclose(two.stage1(q, chunk=7), plain.stage1(q), atol=1e-4)
    assert torch.allclose(two.scores(q), brute.scores(q), atol=1e-5)


def test_quantized_builds_reject_bad_settings_and_fp16_rebuilds_drop_the_scale(tmp_path):
    write_shards(tmp_path / "emb" / "fake", n=12, d=8)
    with pytest.raises(ValueError, match="even dimension"):
        index_build.build_channel(channel(), tmp_path / "emb", tmp_path / "ix", 3, None, "cpu", bits=4)
    with pytest.raises(ValueError, match="4, 8 or 16"):
        index_build.build_channel(channel(), tmp_path / "emb", tmp_path / "ix", 4, None, "cpu", bits=5)
    index_build.build_channel(channel(), tmp_path / "emb", tmp_path / "ix", 4, None, "cpu", bits=8)
    two = serve.TwoStageIndex.load(tmp_path / "ix", channel())
    with pytest.raises(ValueError, match="one scale per window"):
        serve.TwoStageIndex(two.channel, two.ids, two.proj, two.basis, two.norms, two.full, two.mu)
    del two  # releases the memory map of full.npy, which Windows will not replace while it is open
    gc.collect()
    index_build.build_channel(channel(), tmp_path / "emb", tmp_path / "ix", 4, None, "cpu")
    assert not (tmp_path / "ix" / "proj_scale.npy").exists()
    assert serve.TwoStageIndex.load(tmp_path / "ix", channel()).proj_bits == 16


def test_main_builds_quantized_channels_and_hub_scores_use_them(tmp_path):
    from soundalike.anysong import hub_build

    sonic = serve.CHANNELS["sonic"]
    ids, _ = write_shards(tmp_path / "emb" / sonic.model, n=40, d=8)
    np.savez(tmp_path / "meta.npz", track_ids=ids, artists=np.array([f"A{k % 7}" for k in range(40)]),
             titles=np.array([f"Song {k}" for k in range(40)]))
    index_build.main(["--emb", str(tmp_path / "emb"), "--out", str(tmp_path / "ix"), "--dims", "sonic=4",
                      "--proj-bits", "sonic=4", "--device", "cpu"])
    assert json.loads((tmp_path / "ix" / "sonic" / "meta.json").read_text())["proj_bits"] == 4
    hub = hub_build.build(tmp_path / "ix", "sonic", tmp_path / "meta.npz", k=3, device="cpu")
    assert np.isfinite(np.load(tmp_path / "ix" / "sonic" / "hub.npy")).all() and hub["alpha"] == 0.5
    assert serve.TwoStageIndex.load(tmp_path / "ix" / "sonic", sonic).penalty is not None
    with pytest.raises(SystemExit):
        index_build.main(["--emb", str(tmp_path / "emb"), "--out", str(tmp_path / "ix2"), "--dims", "sonic=4",
                          "--proj-bits", "sonic=3", "--device", "cpu"])


def test_max_song_tracks_drops_only_crowded_songs(tmp_path):
    sonic = serve.CHANNELS["sonic"]
    ids, _ = write_shards(tmp_path / "emb" / sonic.model, n=20, d=8)
    # Six radio-show segments that normalise to one song, a song with three releases, the rest unique;
    # ids[19] has no metadata and must be kept.
    titles = [f"Show ({k}) (Intro)" if k < 6 else f"Hit{' (Remastered)' * (k - 6)}" if k < 9 else f"Song {k}"
              for k in range(19)]
    np.savez(tmp_path / "meta.npz", track_ids=ids[:19],
             artists=np.array(["DJ"] * 6 + ["Band"] * 3 + ["X"] * 10), titles=np.array(titles))
    index_build.main(["--emb", str(tmp_path / "emb"), "--out", str(tmp_path / "ix"), "--dims", "sonic=4",
                      "--device", "cpu", "--meta", str(tmp_path / "meta.npz"), "--max-song-tracks", "3"])
    two = serve.TwoStageIndex.load(tmp_path / "ix" / "sonic", sonic)
    assert two.ids.tolist() == ids[6:].tolist()
    meta = json.loads((tmp_path / "ix" / "sonic" / "meta.json").read_text())
    assert meta["n"] == 14 and meta["dropped"]["tracks"] == 6 and meta["dropped"]["songs"] == {"dj - show": 6}
    with pytest.raises(SystemExit):
        index_build.main(["--emb", str(tmp_path / "emb"), "--out", str(tmp_path / "ix2"), "--max-song-tracks", "3"])


def test_row_file_parallel_reads_match_memory_map(tmp_path, monkeypatch):
    path = tmp_path / "full.npy"
    arr = np.random.default_rng(5).standard_normal((50, 3, 7)).astype(np.float16)
    np.save(path, arr)

    def short_preadv(fd, buffers, offset):  # at most 5 bytes per call exercises the retry loop
        with open(path, "rb") as f:
            f.seek(offset)
            return f.readinto(buffers[0][:5])

    monkeypatch.setattr(serve.os, "preadv", short_preadv, raising=False)
    rf = serve.RowFile(path, threads=4)
    assert rf.fd is not None and rf.shape == arr.shape and len(rf) == 50
    rows = np.array([0, 3, 4, 17, 18, 49, 20])
    assert np.array_equal(rf[rows], arr[rows])
    assert np.array_equal(rf[np.array([7])], arr[[7]])
    serve.os.close(rf.fd)
    assert np.array_equal(serve.RowFile(path, threads=1)[rows], arr[rows])
