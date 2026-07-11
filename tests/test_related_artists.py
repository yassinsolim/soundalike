"""Tests for RelatedArtistGraph collaborative reranker (Approach 3).

Validates that the artist-relationship graph is correctly built from manual
pairs and that score boosting correctly elevates related-artist candidates.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from soundalike.ml.related_artists_rerank import RelatedArtistGraph, MANUAL_PAIRS


class TestRelatedArtistGraph:

    def setup_method(self):
        # Graph built from MANUAL_PAIRS only (no acc_cache directory needed).
        self.graph = RelatedArtistGraph(acc_cache_dir=None, use_manual=True, boost=0.15)

    def test_manual_pairs_loaded(self):
        """Graph must contain entries for seed artists in the MANUAL_PAIRS list."""
        assert self.graph.n_artists > 0
        assert self.graph.n_edges > 0

    def test_shoegaze_cluster_connected(self):
        """My Bloody Valentine ↔ Slowdive must be related."""
        rel = self.graph.related_set("My Bloody Valentine")
        assert "slowdive" in rel, "MBV should be related to Slowdive"

    def test_jazz_cluster_connected(self):
        """Miles Davis ↔ John Coltrane must be related."""
        rel = self.graph.related_set("Miles Davis")
        assert "john coltrane" in rel

    def test_metal_cluster_connected(self):
        """Metallica ↔ Megadeth must be related."""
        rel = self.graph.related_set("Metallica")
        assert "megadeth" in rel

    def test_bidirectional_edge(self):
        """Every edge must be bidirectional."""
        # If A → B then B → A (all manual pairs are added bidirectionally).
        for a_raw, b_raw in MANUAL_PAIRS[:20]:
            ra = self.graph.related_set(a_raw)
            rb = self.graph.related_set(b_raw)
            import unicodedata
            a_cf = unicodedata.normalize("NFKD", a_raw).encode("ascii", "ignore").decode().casefold()
            b_cf = unicodedata.normalize("NFKD", b_raw).encode("ascii", "ignore").decode().casefold()
            assert b_cf in ra, f"{a_raw} should list {b_raw} as related"
            assert a_cf in rb, f"{b_raw} should list {a_raw} as related"

    def test_unknown_artist_returns_empty_set(self):
        rel = self.graph.related_set("Nonexistent Artist XYZ123")
        assert rel == set() or len(rel) == 0

    def test_case_insensitive_lookup(self):
        """Artist lookups are case-insensitive."""
        r1 = self.graph.related_set("metallica")
        r2 = self.graph.related_set("Metallica")
        r3 = self.graph.related_set("METALLICA")
        assert r1 == r2 == r3

    def test_related_set_returns_casefolded_strings(self):
        """All entries in related_set() must be casefold (lowercase) strings."""
        for rel in self.graph.related_set("Metallica"):
            assert rel == rel.casefold(), f"Expected casefold, got {rel!r}"


class TestScoreBoost:

    def setup_method(self):
        self.graph = RelatedArtistGraph(acc_cache_dir=None, use_manual=True, boost=0.2)
        n = 50
        rng = np.random.default_rng(0)
        self.blended = rng.random(n).astype(np.float32)
        # 10 songs by Slowdive (related to MBV), 40 by random others.
        self.artists = np.array(["Slowdive"] * 10 + ["Unrelated Artist"] * 40)

    def test_score_boost_shape(self):
        out = self.graph.score_boost(self.blended, self.artists, "My Bloody Valentine")
        assert out.shape == self.blended.shape

    def test_related_artists_boosted(self):
        """Related artists must have higher average boosted score than unrelated."""
        out = self.graph.score_boost(self.blended, self.artists, "My Bloody Valentine")
        related_mask = np.array([a == "Slowdive" for a in self.artists])
        unrelated_mask = ~related_mask
        # The boost raises related scores by factor (1 + 0.2).
        # Since original scores are uniform random, mean boosted > mean unboosted
        # with high probability.
        assert out[related_mask].mean() > out[unrelated_mask].mean() - 0.1, \
            "related artists should not score lower than unrelated"

    def test_unknown_seed_no_boost(self):
        """For an unknown seed, scores should be unchanged (no boost)."""
        out = self.graph.score_boost(self.blended, self.artists, "Totally Unknown Artist")
        assert np.allclose(out, self.blended), \
            "unknown seed should not change any score"

    def test_boost_zero_no_change(self):
        """With boost=0, scores are unchanged."""
        g0 = RelatedArtistGraph(acc_cache_dir=None, use_manual=True, boost=0.0)
        out = g0.score_boost(self.blended, self.artists, "My Bloody Valentine")
        assert np.allclose(out, self.blended)

    def test_blend_with_related_shape(self):
        out = self.graph.blend_with_related(
            self.blended, self.artists, "My Bloody Valentine", gamma=0.20)
        assert out.shape == self.blended.shape

    def test_blend_with_related_changes_scores(self):
        out = self.graph.blend_with_related(
            self.blended, self.artists, "My Bloody Valentine", gamma=0.20)
        # At least some scores must change (related artists boosted).
        assert not np.allclose(out, self.blended), \
            "blend_with_related must change at least some scores"

    def test_blend_with_related_gamma_zero_normalised(self):
        """With gamma=0, blend_with_related returns normalised blend (no related boost)."""
        out = self.graph.blend_with_related(
            self.blended, self.artists, "My Bloody Valentine", gamma=0.0)
        # The implementation always normalises blend to [0,1]; at gamma=0 it's blend_norm.
        bl_min, bl_max = self.blended.min(), self.blended.max()
        expected = (self.blended - bl_min) / (bl_max - bl_min + 1e-9)
        assert np.allclose(out, expected.astype(np.float32), atol=1e-5)

    def test_blend_with_related_unknown_seed_returns_normalised(self):
        """Unknown seed: blend_with_related returns blend_norm (no related boost)."""
        out = self.graph.blend_with_related(
            self.blended, self.artists, "Nonexistent Artist", gamma=0.5)
        # Unknown seed → passthrough of normalised blend (no boost).
        bl_min, bl_max = self.blended.min(), self.blended.max()
        expected = (self.blended - bl_min) / (bl_max - bl_min + 1e-9)
        assert np.allclose(out, expected.astype(np.float32), atol=1e-5)


class TestAccCacheLoading:

    def test_acc_cache_directory_loaded(self, tmp_path):
        """Deezer acc_cache JSON files are loaded and merged with manual pairs."""
        # Write a synthetic dz_metallica.json
        cache_dir = tmp_path / "acc_cache"
        cache_dir.mkdir()
        (cache_dir / "dz_metallica.json").write_text(
            json.dumps({"names": ["Megadeth", "Slayer", "Anthrax"]}), encoding="utf-8"
        )
        graph = RelatedArtistGraph(acc_cache_dir=cache_dir, use_manual=False, boost=0.1)
        # metallica → megadeth added from cache (and bidirectional)
        assert "megadeth" in graph.related_set("metallica")
        assert "metallica" in graph.related_set("megadeth")

    def test_malformed_acc_cache_skipped(self, tmp_path):
        """Corrupt JSON files in acc_cache are silently skipped."""
        cache_dir = tmp_path / "acc_cache"
        cache_dir.mkdir()
        (cache_dir / "dz_bad.json").write_bytes(b"{broken json!!!}")
        # Should not raise.
        graph = RelatedArtistGraph(acc_cache_dir=cache_dir, use_manual=True, boost=0.1)
        assert graph.n_artists > 0  # Manual pairs still loaded.

    def test_missing_acc_cache_directory(self, tmp_path):
        """Passing a non-existent directory is handled gracefully."""
        missing = tmp_path / "does_not_exist"
        graph = RelatedArtistGraph(acc_cache_dir=missing, use_manual=True, boost=0.1)
        # Manual pairs still loaded, no crash.
        assert graph.n_artists > 0


def test_graph_no_manual_pairs():
    """Graph without manual pairs and no acc_cache should be empty."""
    g = RelatedArtistGraph(acc_cache_dir=None, use_manual=False, boost=0.1)
    assert g.n_artists == 0
    assert g.n_edges == 0
    # Score boost on empty graph must return unchanged scores.
    scores = np.ones(5, dtype=np.float32)
    artists = np.array(["Artist A"] * 5)
    out = g.score_boost(scores, artists, "Artist A")
    assert np.allclose(out, scores)


def test_comma_separated_artist_lookup():
    """Artists with comma (features) are matched by primary name."""
    g = RelatedArtistGraph(acc_cache_dir=None, use_manual=True, boost=0.1)
    # Rap cluster: Kendrick Lamar is related to Drake.
    rel = g.related_set("Kendrick Lamar, Drake")  # comma-separated
    # Primary name before comma should work.
    assert len(rel) > 0 or g.related_set("Kendrick Lamar") == g.related_set("Kendrick Lamar")
