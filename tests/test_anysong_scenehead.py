import math

import numpy as np
import pytest

from soundalike.anysong import listener
from soundalike.anysong.scenehead import SceneHead


def _unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


@pytest.fixture
def head(tmp_path):
    rng = np.random.default_rng(0)
    w1, b1 = rng.standard_normal((512, 16)).astype(np.float32), rng.standard_normal(16).astype(np.float32)
    w2, b2 = rng.standard_normal((16, 8)).astype(np.float32), rng.standard_normal(8).astype(np.float32)
    np.savez(tmp_path / "h.npz", w1=w1, b1=b1, w2=w2, b2=b2)
    windows = _unit(rng.standard_normal((4, 3, 512))).astype(np.float32)
    built = SceneHead.__new__(SceneHead)
    built.w1, built.b1, built.w2, built.b2 = w1, b1, w2, b2
    table = np.stack([built.embed(w) for w in windows]).astype(np.float16)
    np.save(tmp_path / "h.emb.npy", table)
    pos = {10: 0, 11: 1, 12: 2, 13: 3}
    return SceneHead(tmp_path / "h.npz", tmp_path / "h.emb.npy", pos), windows, table


def test_embed_is_unit_gelu_mlp(head):
    h, windows, _ = head
    x = windows[0]
    hidden = x @ h.w1 + h.b1
    hidden = np.array([[0.5 * v * (1 + math.erf(v / math.sqrt(2))) for v in row] for row in hidden], dtype=np.float32)
    want = _unit(hidden @ h.w2 + h.b2)
    np.testing.assert_allclose(h.embed(x), want, rtol=1e-4, atol=1e-5)


def test_live_seed_matches_its_stored_vectors(head):
    h, windows, table = head
    np.testing.assert_allclose(h.seed(windows[2], None), table[2].astype(np.float32), atol=2e-3)
    np.testing.assert_array_equal(h.seed(None, 12), table[2].astype(np.float32))
    assert h.seed(None, 99) is None


def test_cosines_take_the_best_window_pair_and_skip_unknown_tracks(head):
    h, _, table = head
    seed = table[0].astype(np.float32)
    got = h.cosines(seed, [13, 99, 11])
    assert set(got) == {11, 13}
    for tid, row in ((11, 1), (13, 3)):
        assert got[tid] == pytest.approx(float((seed @ table[row].astype(np.float32).T).max()), abs=1e-6)
    assert h.cosines(seed, [10])[10] == pytest.approx(1.0, abs=2e-3)


def test_table_must_fit_the_index(tmp_path, head):
    h, _, _ = head
    with pytest.raises(ValueError, match="does not fit"):
        SceneHead(tmp_path / "h.npz", tmp_path / "h.emb.npy", {10: 0})


def test_ranker_accepts_the_head_feature_set_only_as_declared():
    arrays = (0.0, np.array([0]), np.array([len(listener.FEATURES)]), np.array([0.5]), np.array([True]),
              np.array([0]), np.array([0]), np.array([0.0]), 0)
    listener.Ranker(listener.FEATURES_HEAD, *arrays)  # splitting on the head column is valid here
    with pytest.raises(ValueError, match="outside the model"):
        listener.Ranker(listener.FEATURES, *arrays)
    with pytest.raises(ValueError, match="differ"):
        listener.Ranker(listener.FEATURES + ("other",), *arrays)
