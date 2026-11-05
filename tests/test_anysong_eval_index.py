import numpy as np
import pytest

torch = pytest.importorskip("torch")
from soundalike.anysong import eval_index as ei
from soundalike.anysong import evaluate as ev
from soundalike.anysong import index_build, serve


def channel(name="test", prep="raw", mode="maxsim", model="fake"):
    return serve.Channel(name, model, prep, mode)


def random_index(n=40, w=3, d=8, seed=0, name="test"):
    rng = np.random.default_rng(seed)
    ids = np.arange(100, 100 + n, dtype=np.int64)
    win = rng.standard_normal((n, w, d)).astype(np.float16)
    return ids, serve.ChannelIndex.from_windows(channel(name), ids, win)


def test_song_view_collapses_releases_and_drops_the_query_song():
    meta = {1: ("A", "Song"), 2: ("A", "Song (Remastered 2011)"), 3: ("B", "Hit"), 4: ("B feat. C", "Hit - Live"),
            5: ("D", "Other"), 6: ("E", "Else")}
    keys = ei.SongKeys(meta)
    ids = np.array([1, 2, 3, 4, 5, 6])
    rowkey = keys.rows(ids, "cpu")
    row = torch.tensor([-np.inf, 0.9, 0.8, 0.6, 0.7, 0.5])
    best = ei.song_best(row, rowkey, keys.get(1))
    assert ei.song_top(best, 10) == [keys.get(3), keys.get(5), keys.get(6)]
    # Track 4 alone ranks 4th among rows, but its song (B - Hit) is found first through track 3.
    assert ei.song_ranks(best, {keys.get(4)}).tolist() == [1]
    assert ei.song_ranks(best, {keys.get(2)}).tolist() == []
    assert keys.get(999) not in {keys.get(t) for t in meta}


def test_raw_and_song_views_match_research_evaluator_without_duplicates():
    ids, ix = random_index()
    q, positives = int(ids[5]), {int(ids[1]), int(ids[17]), int(ids[30])}
    query = ev.Query("t", "k", "development", q, frozenset(positives))
    expected = ev.evaluate(ix.feats.float(), ids, [query], {}, mode="maxsim")["t/development"]
    rep = ei.evaluate_indexes({"test": ix}, [query], ei.SongKeys({}), batch=4, log=lambda m: None)
    got = rep["channels"]["test"]["t/development"]
    for view in ("raw", "songs"):
        assert got[view]["ap"] == pytest.approx(expected["ap"], abs=1e-5)
        assert got[view]["hr@10"] == expected["hr@10"]
    assert rep["any_tab"]["t/development"]["hr@10"] == expected["hr@10"]


def test_missing_queries_and_same_song_positives_are_counted_not_scored():
    ids, ix = random_index()
    meta = {int(t): (f"artist{t}", f"title{t}") for t in ids}
    meta[int(ids[2])] = ("artist101", "title101 (Live)")  # same song as ids[1]
    queries = [ev.Query("t", "a", "test", int(ids[1]), frozenset({int(ids[2])})),
               ev.Query("t", "b", "test", 5, frozenset({int(ids[3])}))]
    rep = ei.evaluate_indexes({"test": ix}, queries, ei.SongKeys(meta), log=lambda m: None)
    assert rep["coverage"]["test"]["t/test"] == {"missing_query": 1, "same_song_only": 1}
    assert rep["channels"]["test"]["t/test"]["songs"] == {"n": 0}
    assert rep["channels"]["test"]["t/test"]["raw"]["n"] == 1


def test_any_tab_uses_the_best_channel_per_query():
    ids, a = random_index(seed=1, name="a")
    _, b = random_index(seed=2, name="b")
    q = ev.Query("t", "k", "test", int(ids[0]), frozenset(int(t) for t in ids[10:14]))
    rep = ei.evaluate_indexes({"a": a, "b": b}, [q], ei.SongKeys({}), log=lambda m: None)
    firsts = [rep["channels"][c]["t/test"]["songs"]["median_first"] for c in ("a", "b")]
    assert rep["any_tab"]["t/test"]["median_first"] == min(firsts)
    assert rep["any_tab"]["t/test"]["rr"] == pytest.approx(1 / min(firsts))


def test_best_list_is_evaluated_as_the_api_fuses_it(monkeypatch):
    ids, m = random_index(seed=1, name="melody")
    _, s = random_index(seed=2, name="samples")
    _, sonic = random_index(seed=3, name="sonic")
    keys = ei.SongKeys({})
    t = int(ids[0])
    orders = {}
    for name, ix in (("melody", m), ("samples", s)):
        row = ix.scores(ix.stored_query(t))
        orders[name] = [keys.get(x) for x, _ in ix.top(row, ei.FETCH, [t])]
    pos = [int(x) for x in ids[1:] if keys.get(int(x)) not in (orders["melody"][0], orders["samples"][0])][5:9]
    q = ev.Query("t", "k", "test", t, frozenset(pos))
    rep = ei.evaluate_indexes({"melody": m, "samples": s, "sonic": sonic}, [q], keys, log=lambda x: None)
    rank = {key: r for r, (key, _, _) in enumerate(ei.fuse(orders), 1)}
    want = sorted(rank[keys.get(p)] for p in pos)
    assert rep["best"]["t/test"]["median_first"] == want[0]
    assert rep["best"]["t/test"]["ap"] == pytest.approx(np.mean(np.arange(1, 5) / np.array(want)), abs=1e-5)
    # positives outside every served list rank after all of the catalog's songs
    monkeypatch.setattr(ei, "FETCH", 1)
    shallow = ei.evaluate_indexes({"melody": m, "samples": s}, [q], keys, log=lambda x: None)["best"]["t/test"]
    assert shallow["hr@20"] == 0.0 and shallow["median_first"] == len(ids)
    assert ei.evaluate_indexes({"sonic": sonic}, [q], keys, log=lambda x: None)["best"] == {}


@pytest.mark.parametrize("prep,mode", [("raw", "maxsim"), ("none", "bpwr")])
def test_exact_check_is_perfect_when_every_row_is_reranked(tmp_path, prep, mode):
    ch = channel(prep=prep, mode=mode)
    rng = np.random.default_rng(3)
    ids = np.arange(1000, 1060, dtype=np.int64)
    win = rng.standard_normal((60, 3, 16)).astype(np.float16)
    (tmp_path / "emb" / "fake").mkdir(parents=True)
    np.savez(tmp_path / "emb" / "fake" / "part-00000.npz", ids=ids, win=win, track=win.astype(np.float32).mean(1))
    index_build.build_channel(ch, tmp_path / "emb", tmp_path / "ix", 4, None, "cpu")
    two = serve.TwoStageIndex.load(tmp_path / "ix", ch, candidates=len(ids))
    rep = ei.exact_check(two, [int(t) for t in ids[:12]], ei.SongKeys({}), batch=5)
    assert rep["top10_overlap"] == rep["top50_overlap"] == rep["songs_top10_identical"] == 1.0
    brute = serve.ChannelIndex.load(tmp_path / "emb", ch)
    assert torch.allclose(ei.exact_scores(two, brute.stored_query(1003)[None], chunk=7),
                          brute.scores(brute.stored_query(1003))[None], atol=1e-5)


def test_exact_check_includes_the_hub_penalty(tmp_path):
    ch = channel()
    rng = np.random.default_rng(4)
    ids = np.arange(1000, 1060, dtype=np.int64)
    win = rng.standard_normal((60, 3, 16)).astype(np.float16)
    (tmp_path / "emb" / "fake").mkdir(parents=True)
    np.savez(tmp_path / "emb" / "fake" / "part-00000.npz", ids=ids, win=win, track=win.astype(np.float32).mean(1))
    index_build.build_channel(ch, tmp_path / "emb", tmp_path / "ix", 16, None, "cpu")
    hub = rng.uniform(0, 4, 60).astype(np.float32)
    np.save(tmp_path / "ix" / "hub.npy", hub)
    two = serve.TwoStageIndex.load(tmp_path / "ix", ch, candidates=len(ids), hub_alpha=0.5)
    brute = serve.ChannelIndex.load(tmp_path / "emb", ch)
    q = brute.stored_query(1003)[None]
    want = brute.scores(q[0])[None] - 0.5 * torch.from_numpy(hub)
    assert torch.allclose(ei.exact_scores(two, q, chunk=7), want, atol=1e-4)
    assert torch.allclose(two.scores(q), want, atol=1e-4)
    rep = ei.exact_check(two, [int(t) for t in ids[:12]], ei.SongKeys({}), batch=5)
    assert rep["top10_overlap"] == rep["top50_overlap"] == rep["songs_top10_identical"] == 1.0


def test_hubness_accounts_for_every_slot():
    ids, ix = random_index(n=30)
    sample = [int(t) for t in ids[:12]]
    rep = ei.hubness(ix, sample, ei.SongKeys({}), k=5, batch=5)
    assert sum(h["count"] for h in ei.hubness(ix, sample, ei.SongKeys({}), k=5, show=100)["hubs"]) == 12 * 5
    assert rep["songs"] == 30 and 0 < rep["coverage"] <= 1 and rep["top100_share"] == 1.0
    assert rep["hubs"][0]["count"] >= rep["hubs"][-1]["count"]


def test_language_gate_filters_each_channel_before_best_fusion():
    meta = {
        100: ("Seed Artist", "Seed"),
        101: ("English Artist", "English match"),
        102: ("French Artist", "French match"),
        103: ("Unknown Artist", "Unknown match"),
    }
    keys = ei.SongKeys(meta)
    song_ids = {track_id: keys.get(track_id) for track_id in meta}

    languages = ei.LanguageTable(
        np.array([100, 101, 102, 103]), np.array([0, 0, 1, -1], dtype=np.int16), ("en", "fr"), {}, {}
    )

    query = ev.Query("test", "language", "held_out", 100, frozenset({101, 102, 103}))
    orders = {"melody": [song_ids[102], song_ids[101], song_ids[103]],
              "samples": [song_ids[102], song_ids[101], song_ids[103]]}
    track_orders = {
        channel: [(song_ids[track_id], track_id) for track_id in (102, 101, 103)]
        for channel in orders
    }
    report, stats = ei.language_gate_report(
        [query], {100: orders}, {100: track_orders}, set(keys.of.values()), keys,
        languages, ["melody", "samples"],
    )

    assert report["test/held_out"]["ap"] == pytest.approx((1 + 1 + 0.75) / 3, abs=1e-5)
    assert stats["known_seed_queries"] == 1
    assert stats["total_other_language_in_top20"] == 1
    assert stats["mean_other_language_in_unfiltered_top20"] == 1.0


def test_evaluate_indexes_includes_language_filtered_best_metrics():
    ids, melody = random_index(seed=11, name="melody")
    _, samples = random_index(seed=12, name="samples")
    meta = {int(track_id): (f"Artist {track_id}", f"Song {track_id}") for track_id in ids}
    keys = ei.SongKeys(meta)
    query = ev.Query("test", "integration", "held_out", int(ids[0]), frozenset({int(ids[8]), int(ids[12])}))
    languages = ei.LanguageTable(ids.copy(), np.zeros(len(ids), dtype=np.int16), ("en",), {}, {})

    report = ei.evaluate_indexes(
        {"melody": melody, "samples": samples}, [query], keys, log=lambda _: None, languages=languages
    )

    assert report["language_gate"]["known_seed_queries"] == 1
    assert report["best_same_language"]["test/held_out"] == report["best"]["test/held_out"]


def test_scene_best_matches_plain_best_without_affinity_and_moves_with_it():
    from soundalike.anysong.scene import SceneParams, build

    ids, melody = random_index(seed=21, name="melody")
    _, samples = random_index(seed=22, name="samples")
    _, sonic = random_index(seed=23, name="sonic")
    meta = {int(t): (f"Artist {t}", f"Song {t}") for t in ids}
    keys = ei.SongKeys(meta)
    seed, pos = int(ids[0]), int(ids[30])
    query = ev.Query("test", "scene", "held_out", seed, frozenset({pos}))
    indexes = {"melody": melody, "samples": samples, "sonic": sonic}
    # Every track has its own artist; the seed's artist lists the positive's artist as related.
    table = build({int(t): int(t) for t in ids}, {seed: [pos]}, {}, {})
    off = SceneParams(boost=0.0, sonic=0.0, pattern="", max_same=10**6, max_other=10**6)
    report = ei.evaluate_indexes(indexes, [query], keys, log=lambda _: None, scene=table, scene_params=off)
    assert report["best_scene"] == report["best"]
    boosted = ei.evaluate_indexes(indexes, [query], keys, log=lambda _: None, scene=table,
                                  scene_params=SceneParams(boost=50.0, sonic=50.0, pattern=""))
    assert boosted["best_scene"]["test/held_out"]["median_first"] == 1
    assert report["best"]["test/held_out"]["median_first"] > 1
    slotted = ei.evaluate_indexes(indexes, [query], keys, log=lambda _: None, scene=table,
                                  scene_params=SceneParams(melody_evidence=9.0, samples_evidence=9.0,
                                                           rank_evidence=0))
    assert slotted["best_scene"]["test/held_out"]["median_first"] <= 6  # first slots after a three-song head
    # these synthetic cosines all clear the real melody threshold, so every top-10 song is head evidence
    everything = ei.evaluate_indexes(indexes, [query], keys, log=lambda _: None, scene=table)["best_scene"]
    assert everything["test/held_out"]["median_first"] == 11
