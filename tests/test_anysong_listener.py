import math

import numpy as np
import pytest

from soundalike.anysong import listener
from soundalike.anysong.listener import FEATURES, ArtistTracks, Ranker
from soundalike.anysong.scene import SceneParams
from soundalike.anysong.scene import build as build_scene


def stump(feature: str, left: float, right: float, threshold: float = 0.5) -> Ranker:
    """One split on ``feature``: values at most ``threshold`` (and missing ones) score ``left``, others ``right``."""
    return Ranker(FEATURES, 0.0, np.array([0]), np.array([FEATURES.index(feature), 0, 0]),
                  np.array([threshold, 0.0, 0.0]), np.array([True, True, True]), np.array([1, 1, 2]),
                  np.array([2, 1, 2]), np.array([0.0, left, right]), depth=1, meta="stump")


def test_exported_trees_match_scikit_learn_and_round_trip(tmp_path):
    sklearn = pytest.importorskip("sklearn.ensemble")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, len(FEATURES)))
    X[rng.random(X.shape) < 0.2] = np.nan
    y = ((np.nan_to_num(X[:, 0]) + 0.5 * np.nan_to_num(X[:, 11], nan=1.0) - np.isnan(X[:, 3])) > 0.2).astype(int)
    model = sklearn.HistGradientBoostingClassifier(max_iter=40, max_leaf_nodes=15, max_depth=6,
                                                   random_state=0).fit(X, y)
    ranker = listener.export(model, meta="test")
    Xt = rng.normal(size=(500, len(FEATURES)))
    Xt[rng.random(Xt.shape) < 0.3] = np.nan
    np.testing.assert_allclose(ranker.predict(Xt), model.predict_proba(Xt)[:, 1], rtol=0, atol=1e-12)
    ranker.save(tmp_path / "r.npz")
    loaded = Ranker.load(tmp_path / "r.npz")
    assert loaded.meta == "test" and loaded.depth == ranker.depth
    np.testing.assert_array_equal(loaded.predict(Xt), ranker.predict(Xt))


def test_ranker_checks_its_features_and_shapes():
    r = stump("fwd", -1.0, 1.0)
    p = r.predict(np.array([[0.0] * len(FEATURES), [1.0] * len(FEATURES)]))
    assert p[0] == pytest.approx(1 / (1 + math.e)) and p[1] == pytest.approx(1 / (1 + math.exp(-1)))
    assert r.predict(np.zeros((0, len(FEATURES)))).shape == (0,)
    with pytest.raises(ValueError):
        r.predict(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        Ranker(("son",), 0.0, np.array([0]), np.array([0]), np.array([0.0]), np.array([True]), np.array([0]),
               np.array([0]), np.array([0.0]), depth=0)
    with pytest.raises(ValueError):
        Ranker(FEATURES, 0.0, np.array([0]), np.array([0]), np.array([0.0]), np.array([True]), np.array([5]),
               np.array([0]), np.array([0.0]), depth=1)


@pytest.fixture()
def table():
    # 10 lists 11 and 12; 13 lists 10 (so it is a neighbour of 10 too); 14 is unrelated.
    tracks = {1: 10, 2: 11, 3: 11, 4: 11, 5: 11, 6: 12, 7: 13, 8: 14, 9: 10}
    related = {10: [11, 12], 11: [10], 12: [10], 13: [10]}
    return build_scene(tracks, related, {10: ["GB"], 11: ["GB"], 12: ["US"]}, {})


def test_scene_candidates_keep_each_neighbours_closest_songs(table):
    tracks = ArtistTracks(table)
    assert tracks(11).tolist() == [2, 3, 4, 5] and tracks(11, limit=2).tolist() == [2, 3]
    assert tracks(99).tolist() == [] and tracks(0).tolist() == []
    sound = {2: 0.1, 3: 0.5, 4: 0.4, 5: 0.3, 6: 0.2, 7: 0.6, 8: 0.9}
    asked = []

    def score(tids):
        asked.append(sorted(tids))
        return {t: sound[t] for t in tids if t != 4}  # 4 is not indexed

    profile = table.profile(10)
    out = listener.scene_candidates(profile, tracks, score, table.artist)
    assert asked == [[2, 3, 4, 5, 6, 7]]  # neighbours 11, 12 and 13; never the seed artist or 14
    assert out == {3: 0.5, 5: 0.3, 2: 0.1, 6: 0.2, 7: 0.6}
    out = listener.scene_candidates(profile, tracks, score, table.artist, allowed=lambda t: t != 7)
    assert 7 not in out and asked[-1] == [2, 3, 4, 5, 6]


def test_feature_rows_describe_pool_and_scene_songs(table):
    raw = {"melody": [(6, -1.2)], "samples": [(6, 0.4), (8, 0.3), (99, 0.2)], "sonic": [(8, 0.5), (1, 0.9)]}
    son = {6: 0.35, 8: 0.5, 7: 0.45, 99: 0.1}
    tids, X = listener.features(raw, son, [7], query_id=1, profile=table.profile(10), related=[11, 12],
                                table=table, params=SceneParams(), seed_bpm=120.0,
                                bpm_of={6: 60.0, 7: 125.0}.get, language_known=lambda t: t == 6)
    assert tids == [6, 7, 8, 99]  # the seed (1) is left out
    row = {t: dict(zip(FEATURES, x)) for t, x in zip(tids, X)}
    assert row[6]["r_melody"] == 1 and row[6]["s_melody"] == -1.2 and row[6]["r_samples"] == 1
    assert row[6]["in_pool"] == 1 and row[6]["from_scene"] == 0 and row[6]["fwd"] == 1 and row[6]["rev"] == 1
    assert row[6]["tempo_gap"] == pytest.approx(0.0) and row[6]["lang_known"] == 1
    assert math.isnan(row[7]["r_melody"]) and math.isnan(row[7]["r_sonic"]) and row[7]["in_pool"] == 0
    assert row[7]["from_scene"] == 1 and row[7]["fwd"] == 0 and row[7]["rev"] == 1 and row[7]["nb"] == 1
    assert row[7]["tempo_gap"] == pytest.approx(math.log(125 / 120))
    assert row[8]["r_sonic"] == 1 and row[8]["nb"] == 0 and row[8]["aff"] == 0 and math.isnan(row[8]["tempo_gap"])
    assert row[8]["son"] == 0.5 and row[8]["graph"] == 0  # artist 14 is known but unrelated
    assert math.isnan(row[99]["graph"]) and row[99]["nb"] == 0  # no artist for track 99


def test_blend_and_logit():
    np.testing.assert_allclose(listener.blend([1, 2, 3], [5, 5, 5], 2.0), [-1.224744871, 0, 1.224744871])
    np.testing.assert_allclose(listener.blend([1, 1], [0, 2], 1.0), [-1, 1])
    assert listener.logit(np.array([0.5]))[0] == 0 and np.isfinite(listener.logit(np.array([0.0, 1.0]))).all()
    assert listener.tempo_gap(120, 240) == pytest.approx(0) and math.isnan(listener.tempo_gap(None, 100))
