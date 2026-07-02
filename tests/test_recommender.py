"""Tests for the offline content-based recommender."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from soundalike import ContentBasedRecommender, FeatureConfig, load_bundled_dataset
from soundalike.dataset import Dataset
from soundalike.features import AUDIO_FEATURES, resolve_feature
from soundalike.profile import parse_seed_string, read_seed_file


def _toy_frame() -> pd.DataFrame:
    # Three "clusters": low-energy acoustic, high-energy dance, mid.
    rows = [
        ("Calm A", "Artist X", 70, 20, 20, 15, 90, 0, 10, 3),
        ("Calm B", "Artist Y", 72, 25, 25, 18, 88, 0, 12, 4),
        ("Hype A", "Artist Z", 170, 90, 80, 95, 2, 0, 20, 8),
        ("Hype B", "Artist W", 165, 88, 78, 92, 5, 0, 18, 7),
        ("Mid A", "Artist Q", 120, 55, 50, 55, 40, 0, 15, 5),
    ]
    cols = ["title", "artist", "bpm", "danceability", "valence", "energy",
            "acousticness", "instrumentalness", "liveness", "speechiness"]
    return pd.DataFrame(rows, columns=cols)


@pytest.fixture
def toy_dataset() -> Dataset:
    return Dataset._from_raw(_toy_frame())


@pytest.fixture(scope="module")
def bundled_recommender() -> ContentBasedRecommender:
    return ContentBasedRecommender().fit(load_bundled_dataset())


# ------------------------------------------------------------------- dataset
def test_bundled_dataset_loads():
    ds = load_bundled_dataset()
    assert len(ds) > 100
    for feat in AUDIO_FEATURES:
        assert feat in ds.frame.columns
    assert ds.frame[AUDIO_FEATURES].notna().all().all()


def test_canonical_columns_and_helpers(toy_dataset):
    assert "primary_artist" in toy_dataset.frame.columns
    idx = toy_dataset.find_one("calm a")  # case-insensitive
    assert idx == 0
    assert toy_dataset.find_one("does not exist") is None


def test_missing_feature_column_raises():
    bad = _toy_frame().drop(columns=["energy"])
    with pytest.raises(ValueError, match="missing required audio-feature"):
        Dataset._from_raw(bad)


# --------------------------------------------------------------- feature config
def test_feature_config_validation():
    with pytest.raises(ValueError):
        FeatureConfig(features=["not_a_feature"]).validate()
    with pytest.raises(ValueError):
        FeatureConfig(metric="manhattan").validate()
    with pytest.raises(ValueError):
        FeatureConfig(weights={"energy": -1}).validate()


def test_resolve_feature_aliases():
    assert resolve_feature("dance") == "danceability"
    assert resolve_feature("Tempo") == "bpm"
    with pytest.raises(ValueError):
        resolve_feature("loudness")


# ------------------------------------------------------------------ similar_to
def test_similar_excludes_seed_and_orders_by_cluster(toy_dataset):
    rec = ContentBasedRecommender().fit(toy_dataset)
    results = rec.similar_to("Calm A", n=4)
    titles = [r.title for r in results]
    assert "Calm A" not in titles  # seed excluded
    assert titles[0] == "Calm B"  # nearest cluster-mate ranks first
    # scores should be sorted descending
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_similar_unknown_title_raises(toy_dataset):
    rec = ContentBasedRecommender().fit(toy_dataset)
    with pytest.raises(LookupError):
        rec.similar_to("Nonexistent Song")


def test_exclude_same_artist(toy_dataset):
    # Duplicate an artist across clusters to test exclusion.
    frame = _toy_frame()
    frame.loc[len(frame)] = ("Calm C", "Artist Z", 71, 22, 22, 16, 89, 0, 11, 3)
    ds = Dataset._from_raw(frame)
    rec = ContentBasedRecommender().fit(ds)
    results = rec.similar_to("Hype A", n=10, exclude_same_artist=True)
    assert all(r.artist != "Artist Z" for r in results)


def test_fit_required():
    with pytest.raises(RuntimeError):
        ContentBasedRecommender().similar_to("anything")


# ---------------------------------------------------------- recommend_for_profile
def test_profile_recommends_and_reports_unmatched(toy_dataset):
    rec = ContentBasedRecommender().fit(toy_dataset)
    recs, unmatched = rec.recommend_for_profile(
        [("Calm A", None), ("Calm B", None), ("Ghost Song", None)], n=3
    )
    assert ("Ghost Song", None) in unmatched
    titles = [r.title for r in recs]
    assert "Calm A" not in titles and "Calm B" not in titles  # known excluded
    assert titles[0] == "Mid A"  # closest remaining to the calm centroid


def test_profile_all_unmatched_raises(toy_dataset):
    rec = ContentBasedRecommender().fit(toy_dataset)
    with pytest.raises(LookupError):
        rec.recommend_for_profile([("Ghost", None)])


def test_weights_change_ranking(bundled_recommender):
    base = ContentBasedRecommender(FeatureConfig()).fit(bundled_recommender.dataset)
    weighted = ContentBasedRecommender(
        FeatureConfig(weights={"acousticness": 6.0})
    ).fit(bundled_recommender.dataset)
    base_top = [r.title for r in base.similar_to("Blinding Lights", n=10)]
    weighted_top = [r.title for r in weighted.similar_to("Blinding Lights", n=10)]
    assert base_top != weighted_top


def test_cosine_metric_runs(toy_dataset):
    rec = ContentBasedRecommender(FeatureConfig(metric="cosine")).fit(toy_dataset)
    results = rec.similar_to("Hype A", n=2)
    assert len(results) == 2
    # Cosine similarity on standardized vectors ranges from -1 to 1.
    assert all(-1.0 - 1e-9 <= r.score <= 1.0 + 1e-9 for r in results)


# --------------------------------------------------------------------- profile io
def test_parse_seed_string():
    seeds = parse_seed_string("Blinding Lights - The Weeknd; One Dance\nBelieber | X")
    assert ("Blinding Lights", "The Weeknd") in seeds
    assert ("One Dance", None) in seeds
    assert ("Belieber", "X") in seeds


def test_read_seed_csv(tmp_path):
    csv = tmp_path / "seeds.csv"
    csv.write_text("title,artist\nBlinding Lights,The Weeknd\nOne Dance,Drake\n", encoding="utf-8")
    seeds = read_seed_file(csv)
    assert seeds == [("Blinding Lights", "The Weeknd"), ("One Dance", "Drake")]
