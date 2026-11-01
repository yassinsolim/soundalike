import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from soundalike.anysong import evaluate as ev


def query(q=1, pos=(2, 3), split="development", artists=()):
    return ev.Query("example", str(q), split, q, frozenset(pos), frozenset(artists))


def test_ties_have_distinct_ranks_and_are_order_invariant():
    ids = np.array([4, 3, 1, 2])
    feats = torch.ones(4, 2)
    q = query(pos=(3, 4))
    result = ev.evaluate(feats, ids, [q], {})["example/development"]
    assert result["ap"] == pytest.approx((1 / 2 + 2 / 3) / 2)
    assert 0 <= result["ap"] <= 1
    order = np.array([3, 2, 1, 0])
    assert ev.evaluate(feats[order], ids[order], [q], {})["example/development"] == result


def test_missing_embeddings_are_explicit():
    ids = np.array([1, 2, 3])
    feats = torch.eye(3)
    queries = [query(pos=(2, 9)), query(q=8), query(pos=(9,))]
    result = ev.evaluate(feats, ids, queries, {})["example/development"]
    assert result["n_expected"] == 3
    assert result["n"] == 1
    assert result["missing_query"] == 1
    assert result["no_positive"] == 1
    assert result["partial_positives"] == 1


def test_v4_categories_are_not_conflated(tmp_path):
    pairs = []
    for tid, category in ((2, "pure_sonic"), (3, "cover_remix"), (4, "weak_unsubstantiated")):
        pairs.append({"split": "development", "evidence_category": category,
                      "query": {"deezer_id": 1, "artist": "Query"},
                      "target": {"deezer_id": tid, "artist": "Target"}})
    (tmp_path / "v4_pairs.json").write_text(json.dumps({"pairs": pairs}))
    queries, _ = ev.load_benchmarks(tmp_path)
    grouped = {q.bench: q.pos for q in queries}
    assert grouped["v4"] == frozenset({2, 3, 4})
    assert grouped["v4_pure_sonic"] == frozenset({2})
    assert grouped["v4_cover_remix"] == frozenset({3})
    assert grouped["v4_weak_unsubstantiated"] == frozenset({4})


def test_no_usable_queries_is_unavailable_not_perfect():
    result = ev.evaluate(torch.eye(2), np.array([1, 2]), [query(q=9)], {})
    assert result["example/development"]["ap"] is None
    assert result["example/development"]["n"] == 0


def test_same_artist_exclusions_are_counted():
    result = ev.evaluate(torch.eye(2), np.array([1, 2]),
                         [query(pos=(2,), artists=("a",))], {1: {"a"}, 2: {"a"}}, xartist=True)
    assert result["example/development"]["artist_filtered"] == 1
    assert result["example/development"]["ap"] is None


def test_query_cannot_be_its_own_positive():
    result = ev.evaluate(torch.eye(2), np.array([1, 2]), [query(pos=(1,))], {})
    assert result["example/development"]["no_positive"] == 1


def test_incomplete_corpus_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "build_view", lambda *args: (np.array([1]), torch.ones(1, 2), "dot"))
    with pytest.raises(ValueError, match="missing 1 required corpus"):
        ev.run_config(tmp_path, "model", "track", "raw", [], {}, "cpu", {1, 2}, False, True)


def test_csls_uses_twice_the_pair_similarity():
    feats = torch.tensor([[1.0, 0.0], [0.9, np.sqrt(0.19)], [0.8, 0.6]], dtype=torch.float32)
    result = ev.evaluate(feats, np.array([1, 2, 3]), [query(pos=(2,))], {},
                         penalty=torch.tensor([0.0, 0.15, 0.0]))
    assert result["example/development"]["hr@1"] == 1


@pytest.mark.parametrize("mode", ["dot", "maxsim"])
def test_csls_neighbors_use_actual_scorer_and_exclude_self(mode):
    feats = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    if mode == "maxsim":
        feats = torch.stack((feats, feats.flip(0)), dim=1)
    expected = ev.score_block(feats, feats, mode)
    expected.fill_diagonal_(-torch.inf)
    expected = expected.topk(2).values.mean(1)
    torch.testing.assert_close(ev.hub_penalty(feats, k=10, chunk=1, mode=mode), expected)


def test_split_filter_prevents_held_out_scoring(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "load_benchmarks", lambda _: (
        [query(split="development"), query(split="held_out"), query(split="test")], {}))
    seen = []

    def run(*args, **kwargs):
        seen.extend(q.split for q in args[4])
        return {"model": "fake", "view": "track", "prep": "raw", "xartist": False, "results": {}}

    monkeypatch.setattr(ev, "run_config", run)
    out = tmp_path / "results.jsonl"
    ev.main(["--emb", str(tmp_path), "--bench", str(tmp_path), "--model", "fake", "--view", "track",
             "--prep", "raw", "--splits", "development", "--out", str(out), "--device", "cpu"])
    assert seen == ["development"]
    assert len(out.read_text().splitlines()) == 1
    json.loads(out.read_text())
