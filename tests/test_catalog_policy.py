from dataclasses import fields

import numpy as np
import pytest

from soundalike.ml.catalog_policy import (
    CatalogPolicy,
    CatalogPolicyRanker,
    graph_score,
    policy_score,
)


class FakeRecommender:
    def __init__(self):
        self.titles = np.asarray(["Seed"] + [f"Track {i}" for i in range(1, 13)])
        self.artists = np.asarray(["seed"] + [f"artist {i}" for i in range(1, 13)])
        self.track_ids = np.arange(100, 113)
        self._sonic = np.ones((13, 2), np.float32)
        self._clap = np.ones((13, 2), np.float32)
        self._vscaled = np.zeros((13, 1), np.float32)
        self.alpha = 0.8
        self.static_popularity = np.arange(13, dtype=np.float64)
        self.production_rows = [12, 11, 10, 9, 8, 7, 6]

    def recommend(self, row, n=20, **kwargs):
        assert row == 0
        assert kwargs == {
            "alpha": 0.8,
            "diversity": 0.15,
            "max_per_artist": 1,
            "quality_filter": True,
            "genre_rerank": True,
        }
        return {
            "results": [
                {
                    "deezer_id": int(self.track_ids[candidate]),
                    "title": str(self.titles[candidate]),
                    "artist": str(self.artists[candidate]),
                }
                for candidate in self.production_rows[:n]
            ]
        }


class FakeGraph:
    track_artist_ids = np.arange(13, dtype=np.int32)
    track_rows = np.arange(13, dtype=np.int32)
    track_indptr = np.arange(14, dtype=np.int32)

    def __init__(self, *, music4all=True, shared=6):
        left = np.arange(1, 9, dtype=np.int32)
        right = np.concatenate(
            (np.arange(1, shared + 1), np.arange(9, 9 + 8 - shared))
        ).astype(np.int32)
        self.payload = {
            "lastfm": {
                "artist_ids": left,
                "weights": np.linspace(1.0, 0.65, 8, dtype=np.float32),
            },
            "music4all": {
                "artist_ids": right if music4all else np.empty(0, np.int32),
                "weights": (
                    np.linspace(1.0, 0.65, 8, dtype=np.float32)
                    if music4all
                    else np.empty(0, np.float32)
                ),
            },
            "union_artist_ids": np.asarray(
                sorted(set(left) | (set(right) if music4all else set())),
                dtype=np.int32,
            ),
            "source_coverage": {"lastfm": True, "music4all": music4all},
            "mode": "dual_source_union" if music4all else "dual_source_unavailable",
        }

    def dual_source_neighbors(self, artist):
        assert artist == "seed"
        return self.payload


class FakeStyles:
    def __init__(self, value):
        self.value = value

    def style_overlap(self, query, candidate):
        return self.value


def _ranker(policy, *, style=0.9, music4all=True, shared=6):
    return CatalogPolicyRanker(
        FakeRecommender(),
        FakeGraph(music4all=music4all, shared=shared),
        FakeStyles(style),
        policy,
    )


def test_policy_has_exactly_three_numeric_fields_and_fixed_formulas():
    policy = CatalogPolicy(0.4, 0.2, 0.3)
    assert [field.name for field in fields(policy)] == [
        "tau",
        "sigma",
        "audio_weight",
    ]
    assert all(isinstance(getattr(policy, field.name), float) for field in fields(policy))
    expected = 0.7 * 0.5 + 0.3 / np.log2(3)
    assert np.isclose(graph_score(0.5, 2), expected)
    assert policy_score(expected, 0.25, 0.0, policy) == policy_score(
        expected, 0.25, 1.0, policy
    )


@pytest.mark.parametrize(
    "values",
    [(-0.1, 0.5, 0.2), (1.1, 0.5, 0.2), (0.5, -0.1, 0.2),
     (0.5, 1.1, 0.2), (0.5, 0.5, -0.1), (0.5, 0.5, np.inf)],
)
def test_policy_validation(values):
    with pytest.raises(ValueError):
        CatalogPolicy(*values)


def test_missing_source_fails_closed_to_production_order():
    payload = _ranker(
        CatalogPolicy(0.0, 0.0, 0.2), music4all=False
    ).recommend(0, 5)
    assert payload["mode"] == "production_abstention"
    assert payload["gate"]["reason"] == "missing_independent_source"
    assert [item["row"] for item in payload["results"]] == [12, 11, 10, 9, 8]
    assert all(
        item["rationale"]["source"] == "production_abstention"
        for item in payload["results"]
    )
    assert all(item["rationale"]["A"] > 0 for item in payload["results"])


def test_strong_agreement_but_low_consistency_abstains():
    payload = _ranker(CatalogPolicy(0.7, 0.8, 0.2), style=0.1).recommend(0, 5)
    assert payload["gate"]["agreement"] >= 0.7
    assert payload["gate"]["consistency"] < 0.8
    assert payload["gate"]["reason"] == "consistency_below_sigma"
    assert payload["mode"] == "production_abstention"


def test_weak_agreement_but_high_consistency_abstains():
    payload = _ranker(CatalogPolicy(0.99, 0.8, 0.2), style=0.9).recommend(0, 5)
    assert payload["gate"]["agreement"] < 0.99
    assert payload["gate"]["consistency"] >= 0.8
    assert payload["gate"]["reason"] == "agreement_below_tau"
    assert payload["mode"] == "production_abstention"


def test_both_thresholds_pass_returns_graph_head_and_explicit_tail():
    payload = _ranker(CatalogPolicy(0.7, 0.8, 0.2)).recommend(0, 7)
    assert payload["gate"]["fired"] is True
    assert payload["mode"] == "dual_source_graph"
    assert all(
        item["rationale"]["source"] == "dual_source_graph"
        for item in payload["results"][:5]
    )
    assert all(
        item["rationale"]["source"] == "production_tail"
        for item in payload["results"][5:]
    )
    assert all(
        np.isclose(
            item["rationale"]["G"],
            0.5
            * (
                item["rationale"]["lastfm_G"]
                + item["rationale"]["music4all_G"]
            ),
        )
        for item in payload["results"][:5]
    )


def test_fewer_than_five_shared_neighbors_abstains():
    payload = _ranker(
        CatalogPolicy(0.0, 0.0, 0.2), shared=4
    ).recommend(0, 5)
    assert payload["gate"]["shared_count"] == 4
    assert payload["gate"]["reason"] == "fewer_than_five_shared_neighbors"


def test_static_popularity_has_no_effect():
    ranker = _ranker(CatalogPolicy(0.7, 0.8, 0.2))
    before = [item["row"] for item in ranker.recommend(0, 7)["results"]]
    ranker.rec.static_popularity[:] = ranker.rec.static_popularity[::-1]
    after = [item["row"] for item in ranker.recommend(0, 7)["results"]]
    assert after == before
