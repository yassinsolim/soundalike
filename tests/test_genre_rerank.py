"""Tests for ArtistCentroidIndex genre reranker (Approach 2).

Validates that per-artist centroids are correctly computed and that the
genre-coherence blend correctly boosts same-scene artists.
"""

from __future__ import annotations

import numpy as np
import pytest

from soundalike.ml.genre_rerank import ArtistCentroidIndex, build_centroid_index


def _make_clustered_index(n_clusters=4, per_cluster=30, dim=32, seed=0):
    """Return (neural_w, artists) with tight clusters in embedding space.

    Each cluster represents one 'artist', so songs in the same cluster share
    an artist. The clusters are well-separated so same-artist centroids
    clearly dominate cross-artist distances.
    """
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim)).astype(np.float32)
    centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-9

    neural, artists = [], []
    for c_idx in range(n_clusters):
        for _ in range(per_cluster):
            v = centers[c_idx] + 0.05 * rng.standard_normal(dim).astype(np.float32)
            v /= np.linalg.norm(v) + 1e-9
            neural.append(v)
            artists.append(f"artist_{c_idx}")
    return np.array(neural, dtype=np.float32), np.array(artists)


class TestArtistCentroidIndex:

    def setup_method(self):
        self.neural, self.artists = _make_clustered_index()
        self.idx = ArtistCentroidIndex(self.neural, self.artists, min_songs=2)

    def test_centroid_count(self):
        # 4 clusters → 4 centroids
        assert self.idx.n_centroids == 4

    def test_centroid_is_unit_norm(self):
        for c in self.idx._centroids.values():
            assert abs(np.linalg.norm(c) - 1.0) < 1e-5, "centroids must be unit-norm"

    def test_seed_artist_centroid_returned(self):
        c = self.idx.seed_artist_centroid("artist_0")
        assert c is not None
        assert c.shape == (self.neural.shape[1],)

    def test_unknown_artist_centroid_is_none(self):
        assert self.idx.seed_artist_centroid("nobody") is None

    def test_genre_similarity_shape(self):
        sim = self.idx.genre_similarity("artist_0")
        assert sim.shape == (len(self.neural),)

    def test_genre_similarity_same_cluster_highest(self):
        """Songs from the seed artist's cluster should have higher similarity."""
        sim = self.idx.genre_similarity("artist_0")
        # Mean sim for same-artist songs
        same_mask = np.array([a == "artist_0" for a in self.artists])
        diff_mask = ~same_mask
        assert sim[same_mask].mean() > sim[diff_mask].mean() + 0.1, \
            "same-artist songs should score higher than cross-artist songs"

    def test_genre_similarity_unknown_artist_fallback(self):
        """When seed artist is unknown, neural fallback is used."""
        seed_emb = self.neural[0]  # use a specific embedding as fallback
        sim = self.idx.genre_similarity("unknown_artist", seed_neural_w=seed_emb)
        assert sim.shape == (len(self.neural),)
        # Should be non-zero (the fallback computes real cosines)
        assert sim.max() > 0.0

    def test_genre_similarity_all_zeros_no_fallback(self):
        """Unknown artist with no fallback → all-zero similarity."""
        sim = self.idx.genre_similarity("unknown_artist")
        assert np.allclose(sim, 0.0)

    def test_blend_with_genre_boosts_same_scene(self):
        """blend_with_genre must increase scores for same-artist candidates."""
        blended = np.random.default_rng(42).random(len(self.neural)).astype(np.float32)
        new_blend = self.idx.blend_with_genre(blended, "artist_0", gamma=0.4)
        assert new_blend.shape == blended.shape

        # Same-artist songs should rank higher on average after blending.
        same_mask = np.array([a == "artist_0" for a in self.artists])
        diff_mask = ~same_mask
        assert new_blend[same_mask].mean() > new_blend[diff_mask].mean(), \
            "genre blend must boost same-artist songs"

    def test_blend_gamma_zero_unchanged(self):
        """With gamma=0, blend_with_genre output equals normalised original blend."""
        blended = np.arange(len(self.neural), dtype=np.float32)
        result = self.idx.blend_with_genre(blended, "artist_0", gamma=0.0)
        # At gamma=0 it returns blend_norm (blend normalised to [0,1])
        bl_min, bl_max = blended.min(), blended.max()
        expected = (blended - bl_min) / (bl_max - bl_min + 1e-9)
        assert np.allclose(result, expected, atol=1e-5)

    def test_blend_gamma_one_pure_genre(self):
        """With gamma=1, result is entirely genre_norm."""
        blended = np.zeros(len(self.neural), dtype=np.float32)
        result = self.idx.blend_with_genre(blended, "artist_0", gamma=1.0)
        genre_sim = self.idx.genre_similarity("artist_0")
        gs_min, gs_max = genre_sim.min(), genre_sim.max()
        expected = (genre_sim - gs_min) / (gs_max - gs_min + 1e-9)
        assert np.allclose(result, expected.astype(np.float32), atol=1e-5)

    def test_min_songs_threshold(self):
        """Artists with fewer than min_songs tracks get no centroid."""
        # Build index with 1 track per artist → min_songs=2 should exclude them.
        n = 20
        rng = np.random.default_rng(99)
        neural_solo = rng.standard_normal((n, 16)).astype(np.float32)
        neural_solo /= np.linalg.norm(neural_solo, axis=1, keepdims=True)
        artists_solo = [f"solo_{i}" for i in range(n)]
        idx_strict = ArtistCentroidIndex(neural_solo, artists_solo, min_songs=2)
        # All artists have 1 song → no centroids
        assert idx_strict.n_centroids == 0

    def test_casefold_lookup(self):
        """Artist lookup is case-insensitive."""
        c1 = self.idx.seed_artist_centroid("Artist_0")
        c2 = self.idx.seed_artist_centroid("ARTIST_0")
        c3 = self.idx.seed_artist_centroid("artist_0")
        assert c1 is not None and c2 is not None and c3 is not None
        assert np.allclose(c1, c3) and np.allclose(c2, c3)


def test_build_centroid_index_convenience():
    """build_centroid_index() is a transparent wrapper for ArtistCentroidIndex()."""
    rng = np.random.default_rng(7)
    neural = rng.standard_normal((20, 16)).astype(np.float32)
    neural /= np.linalg.norm(neural, axis=1, keepdims=True) + 1e-9
    artists = ["a"] * 10 + ["b"] * 10
    idx = build_centroid_index(neural, artists)
    assert idx.n_centroids == 2


def test_single_song_artist_fallback():
    """An artist with a single song uses the song's own embedding as centroid fallback."""
    rng = np.random.default_rng(11)
    neural = rng.standard_normal((10, 8)).astype(np.float32)
    neural /= np.linalg.norm(neural, axis=1, keepdims=True) + 1e-9
    # Artist "loner" has only 1 song; min_songs=2 → no centroid stored.
    # genre_similarity should still return a non-zero array (fallback path).
    artists = ["group"] * 9 + ["loner"]
    idx = ArtistCentroidIndex(neural, artists, min_songs=2)
    sim = idx.genre_similarity("loner")
    assert sim.shape == (10,)
    # loner has no centroid → returns zeros (neutral)
    assert np.allclose(sim, 0.0)
    # but with seed_neural_w fallback it returns something non-trivial
    sim_fb = idx.genre_similarity("loner", seed_neural_w=neural[9])
    assert sim_fb.max() > 0.0


def test_float32_output():
    """Output arrays are always float32 (for memory efficiency at 272k songs)."""
    neural, artists = _make_clustered_index()
    idx = ArtistCentroidIndex(neural, artists)
    sim = idx.genre_similarity("artist_0")
    assert sim.dtype == np.float32
    blended = np.zeros(len(neural), dtype=np.float32)
    out = idx.blend_with_genre(blended, "artist_0")
    assert out.dtype == np.float32
