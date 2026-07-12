import json
from pathlib import Path

import numpy as np

from soundalike.ml.catalog_style import (
    CatalogStyleIndex,
    SCENE_NAMES,
    audit_catalog_styles,
    build_catalog_style_asset,
    tags_to_scene_vector,
)


def _assets(tmp_path: Path):
    graph = tmp_path / "graph.npz"
    np.savez_compressed(
        graph,
        artist_names=np.asarray(["alpha", "beta", "blend", "cold"]),
        artist_audio=np.asarray(
            [[1, 0], [0, 1], [0.8, 0.2], [0.05, 0.95]], dtype=np.float16
        ),
    )
    cache = tmp_path / "musicbrainz.json"
    cache.write_text(
        json.dumps(
            {
                "alpha": ["indie rock", "shoegaze"],
                "beta": ["techno", "ambient"],
                "blend": ["folk metal", "electropop"],
            }
        ),
        encoding="utf-8",
    )
    return graph, cache


def test_broad_tags_are_multilabel_and_support_genre_blends():
    vector = tags_to_scene_vector(["folk metal", "electropop"])
    active = {SCENE_NAMES[row] for row in np.flatnonzero(vector)}
    assert active == {"folk_country", "metal", "electronic", "pop"}
    assert np.isclose(np.linalg.norm(vector), 1.0)
    assert not tags_to_scene_vector(["seen live", "favorite"]).any()


def test_build_is_independent_compact_deterministic_and_covers_catalog(tmp_path):
    graph, cache = _assets(tmp_path)
    first = tmp_path / "styles-1.npz"
    second = tmp_path / "styles-2.npz"
    metadata = build_catalog_style_asset(
        graph, cache, first, anchors=2, chunk_size=1, anchor_block_size=2
    )
    build_catalog_style_asset(
        graph, cache, second, anchors=2, chunk_size=3, anchor_block_size=1
    )
    assert metadata["source"]["provider"].startswith("MusicBrainz")
    assert metadata["source"]["graph_source_independent"] is True
    assert metadata["source"]["uses_lastfm"] is False
    assert metadata["source"]["uses_music4all"] is False
    assert metadata["coverage"]["covered_artists"] == 4
    with np.load(first, allow_pickle=False) as one, np.load(
        second, allow_pickle=False
    ) as two:
        assert one["style_vectors"].dtype == np.float16
        assert one["confidence"].dtype == np.float16
        np.testing.assert_array_equal(one["style_vectors"], two["style_vectors"])
        np.testing.assert_array_equal(one["confidence"], two["confidence"])
        direct = one["direct_mask"].astype(bool)
        expected = np.stack(
            [
                tags_to_scene_vector(["indie rock", "shoegaze"]),
                tags_to_scene_vector(["techno", "ambient"]),
                tags_to_scene_vector(["folk metal", "electropop"]),
            ]
        ).astype(np.float16)
        np.testing.assert_array_equal(one["style_vectors"][direct], expected)


def test_runtime_overlap_and_false_exclusion_audit(tmp_path):
    graph, cache = _assets(tmp_path)
    output = tmp_path / "styles.npz"
    build_catalog_style_asset(graph, cache, output, anchors=1)
    index = CatalogStyleIndex(output)
    assert index.style_overlap("BETA", "cold") > 0.99
    assert index.style_overlap("alpha", "missing") == 0.0
    np.testing.assert_array_equal(
        index.style_overlaps("beta", ["cold", "missing"]),
        np.asarray([index.style_overlap("beta", "cold"), 0], dtype=np.float32),
    )
    audit = audit_catalog_styles(
        index,
        [("beta", "cold"), ("alpha", "beta"), ("missing", "beta")],
        threshold=0.5,
    )
    assert audit["coverage"] == {
        "catalogue_artists": 4,
        "direct_tag_artists": 3,
        "direct_tag_fraction": 0.75,
        "propagated_artists": 1,
        "covered_artists": 4,
        "covered_fraction": 1.0,
    }
    assert audit["false_exclusions"]["evaluated_pairs"] == 2
    assert audit["false_exclusions"]["unresolved_pairs"] == 1
    assert audit["false_exclusions"]["count"] == 1
    assert audit["false_exclusions"]["rate"] == 0.5
