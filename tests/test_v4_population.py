"""Focused tests for the V4 artist-disjoint population freeze."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from soundalike.ml import v4_population as population


class Context:
    def __init__(self, tracks):
        self.tracks = tuple(tracks)
        self.source_fingerprint = "f" * 64

    @property
    def by_track_id(self):
        return MappingProxyType({track.track_id: track for track in self.tracks})


def track(track_id: int, artist_id: int, part: str | None):
    return SimpleNamespace(
        track_id=track_id,
        artist_id=artist_id,
        fold_parts=(part,),
    )


def fixture_context():
    return Context(
        [
            track(1, 10, "train"),
            track(2, 10, "train"),
            track(3, 20, "validation"),
            track(4, 30, "test"),
            track(5, 30, "test"),
            track(6, 40, "test"),
            track(7, 50, None),
        ]
    )


def test_population_excludes_whole_exposed_artist_and_is_deterministic(tmp_path):
    source = tmp_path / "pack.json"
    source.write_text(
        json.dumps({"seed_track_id": 2, "track_ids": [999, 6]}),
        encoding="utf-8",
    )
    context = fixture_context()
    first = population.build_population_manifest(context, [source])
    second = population.build_population_manifest(context, [source])
    assert first == second
    assert first["excluded"]["track_ids"] == [2, 6]
    assert first["excluded"]["artist_ids"] == [10, 40]
    assert first["development"]["track_ids"] == [3]
    assert first["human_reserve"]["track_ids"] == [4, 5]
    assert first["unassigned_track_ids"] == [7]
    population.validate_population_manifest(first, context)


def test_population_rejects_artist_overlap_even_after_rehash(tmp_path):
    source = tmp_path / "pack.json"
    source.write_text("{}", encoding="utf-8")
    context = fixture_context()
    document = copy.deepcopy(population.build_population_manifest(context, [source]))
    document["development"]["track_ids"].append(4)
    document["development"]["track_ids"].sort()
    document["development"]["artist_ids"].append(30)
    document["development"]["artist_ids"].sort()
    document["counts"]["development_tracks"] += 1
    document["counts"]["development_artists"] += 1
    document["content_sha256"] = population._content_sha256(document)
    with pytest.raises(population.V4PopulationError, match="overlap"):
        population.validate_population_manifest(document, context)


def test_population_rejects_tampering_and_invalid_sources(tmp_path):
    source = tmp_path / "pack.json"
    source.write_text("{}", encoding="utf-8")
    context = fixture_context()
    document = copy.deepcopy(population.build_population_manifest(context, [source]))
    document["counts"]["source_tracks"] += 1
    with pytest.raises(population.V4PopulationError, match="hash drift"):
        population.validate_population_manifest(document, context)
    invalid = tmp_path / "pack.txt"
    invalid.write_text("{}", encoding="utf-8")
    with pytest.raises(population.V4PopulationError, match="not JSON"):
        population.collect_exposed_tracks([invalid], set(context.by_track_id))
