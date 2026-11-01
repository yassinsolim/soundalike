import numpy as np
import pytest

torch = pytest.importorskip("torch")
from soundalike.anysong import evaluate as ev
from soundalike.anysong import serve


def channel(prep="raw", mode="maxsim"):
    return serve.Channel("test", "model", prep, mode)


def random_windows(n=40, w=3, d=8, seed=0):
    rng = np.random.default_rng(seed)
    return np.arange(100, 100 + n, dtype=np.int64), rng.standard_normal((n, w, d)).astype(np.float16)


@pytest.mark.parametrize("prep,mode", [("raw", "maxsim"), ("center", "maxsim"), ("none", "bpwr")])
def test_ranks_match_research_evaluator(prep, mode):
    ids, win = random_windows()
    ix = serve.ChannelIndex.from_windows(channel(prep, mode), ids, win)
    feats = ix.feats.float()
    q = int(ids[5])
    positives = {int(ids[1]), int(ids[17]), int(ids[30])}
    query = ev.Query("t", "k", "development", q, frozenset(positives))
    expected = ev.evaluate(feats, ids, [query], {}, mode=mode)["t/development"]
    ranks = ix.rank_of(ix.scores(ix.stored_query(q)), positives, exclude=[q])
    assert serve_metrics(ranks)["ap"] == pytest.approx(expected["ap"])
    assert serve_metrics(ranks)["hr@10"] == expected["hr@10"]


def serve_metrics(ranks):
    from soundalike.anysong.serve_check import rank_metrics

    return rank_metrics(ranks)


def test_center_prep_uses_index_mean_for_unseen_queries():
    ids, win = random_windows(seed=1)
    ix = serve.ChannelIndex.from_windows(channel("center"), ids, win)
    expected_mu = win.astype(np.float32).reshape(-1, win.shape[-1]).mean(0)
    assert ix.mu.numpy() == pytest.approx(expected_mu, abs=1e-5)
    q = ix.prep_query(np.ones((3, win.shape[-1]), np.float16))
    assert torch.linalg.vector_norm(q, dim=-1) == pytest.approx(torch.ones(3), abs=1e-5)
    assert torch.linalg.vector_norm(ix.feats.float(), dim=-1).numpy() == pytest.approx(1, abs=2e-3)


def test_top_excludes_ids_and_breaks_ties_by_id_independent_of_order():
    ids = np.array([9, 3, 7, 1], dtype=np.int64)
    win = np.ones((4, 2, 4), np.float16)
    ix = serve.ChannelIndex.from_windows(channel(), ids, win)
    scores = ix.scores(ix.prep_query(np.ones((2, 4), np.float16)))
    assert [t for t, _ in ix.top(scores, 3, exclude=[3])] == [1, 7, 9]
    order = np.array([2, 0, 3, 1])
    shuffled = serve.ChannelIndex.from_windows(channel(), ids[order], win[order])
    assert shuffled.top(shuffled.scores(shuffled.prep_query(np.ones((2, 4)))), 3, exclude=[3]) == \
        ix.top(scores, 3, exclude=[3])
    assert ix.top(scores, 10, exclude=ids) == []


def test_rank_of_ignores_excluded_and_missing_positives():
    ids, win = random_windows(n=10, seed=2)
    ix = serve.ChannelIndex.from_windows(channel(), ids, win)
    sc = ix.scores(ix.stored_query(int(ids[0])))
    assert len(ix.rank_of(sc, [int(ids[0]), 999_999], exclude=[int(ids[0])])) == 0
    assert ix.rank_of(sc, [int(ids[0])])[0] == 1


def test_batched_scores_equal_single_scores():
    ids, win = random_windows(n=12, seed=3)
    for prep, mode in (("raw", "maxsim"), ("none", "bpwr")):
        ix = serve.ChannelIndex.from_windows(channel(prep, mode), ids, win)
        q = torch.stack([ix.stored_query(int(ids[0])), ix.stored_query(int(ids[4]))])
        batched = ix.scores(q, chunk=5)
        assert torch.allclose(batched[1], ix.scores(q[1], chunk=7), atol=1e-6)


def test_index_rejects_duplicate_ids():
    ids, win = random_windows(n=3)
    ids[1] = ids[0]
    with pytest.raises(ValueError, match="unique"):
        serve.ChannelIndex.from_windows(channel(), ids, win)


def test_single_decode_matches_per_rate_decode(tmp_path):
    sf = pytest.importorskip("soundfile")
    pytest.importorskip("soxr")
    from soundalike.anysong.audio import load_clip, load_clips

    rng = np.random.default_rng(4)
    path = tmp_path / "clip.wav"
    sf.write(path, (0.1 * rng.standard_normal((44_100 * 3, 2))).astype(np.float32), 44_100)
    clips = load_clips(path, (16_000, 24_000, 44_100), seconds=5.0)
    for sr, clip in clips.items():
        assert clip.shape == (5 * sr,)
        assert np.array_equal(clip, load_clip(path, sr, seconds=5.0))
