"""Tests for the TitleQualityFilter junk-track removal (Approach 1).

Validates that slowed/reverb/karaoke/tribute derivatives are correctly
identified and that real tracks are never suppressed.
"""

from __future__ import annotations

import numpy as np
import pytest

from soundalike.ml.quality_filter import TitleQualityFilter, keep_mask, default_filter


class TestTitleQualityFilter:
    """Unit tests for TitleQualityFilter.is_junk() and .keep_mask()."""

    def setup_method(self):
        self.f = TitleQualityFilter()

    # ── Positive (should be kept) ─────────────────────────────────────────────

    def test_normal_pop_track_is_kept(self):
        assert not self.f.is_junk("Blinding Lights", "The Weeknd")

    def test_metal_track_is_kept(self):
        assert not self.f.is_junk("Master of Puppets", "Metallica")

    def test_jazz_track_is_kept(self):
        assert not self.f.is_junk("Take Five", "Dave Brubeck Quartet")

    def test_track_with_remaster_note_is_kept(self):
        # "Remastered" in parentheses is a version tag, not a junk type.
        assert not self.f.is_junk("Master of Puppets (Remastered 2021)", "Metallica")

    def test_city_pop_with_stay_in_title_is_kept(self):
        assert not self.f.is_junk("Mayonaka no Door / Stay With Me", "Miki Matsubara")

    def test_hyperpop_track_is_kept(self):
        assert not self.f.is_junk("Ring Ring", "Charli XCX")

    def test_lofi_hip_hop_is_kept(self):
        # Generic "lofi hip hop" genre name — only lofi *version/remix/cover* is junk.
        assert not self.f.is_junk("lofi hip hop study beats", "Chillhop Music")

    def test_piano_solo_is_kept(self):
        # "piano" alone is fine — only "piano version" is junk.
        assert not self.f.is_junk("Nocturne Op. 9 No. 2", "Chopin")

    def test_real_cover_artist_with_cover_in_title_is_not_blocked(self):
        # A song titled "Cover Me" is fine — "cover version" is what we block.
        assert not self.f.is_junk("Cover Me", "Bruce Springsteen")

    # ── Negative (should be filtered) ────────────────────────────────────────

    def test_slowed_is_junk(self):
        assert self.f.is_junk("Bad Guy (Slowed)", "Billie Eilish")

    def test_slowed_no_parens_is_junk(self):
        assert self.f.is_junk("Levitating slowed + reverb", "Dua Lipa")

    def test_reverb_only_is_junk(self):
        assert self.f.is_junk("Heat Waves reverb", "Glass Animals")

    def test_nightcore_is_junk(self):
        assert self.f.is_junk("Nightcore - Blinding Lights", "Nightcore")

    def test_karaoke_in_title_is_junk(self):
        assert self.f.is_junk("Bohemian Rhapsody (Karaoke Version)", "Queen")

    def test_tribute_in_title_is_junk(self):
        assert self.f.is_junk("Tribute to The Weeknd", "Various")

    def test_karaoke_artist_is_junk(self):
        assert self.f.is_junk("Any Track", "Karaoke Universe")

    def test_tribute_artist_is_junk(self):
        assert self.f.is_junk("Shape of You", "Tribute to Ed Sheeran")

    def test_cover_version_is_junk(self):
        assert self.f.is_junk("Lose Yourself (Cover Version)", "Generic Artist")

    def test_piano_version_is_junk(self):
        assert self.f.is_junk("Starboy - Piano Version", "Piano Covers")

    def test_sped_up_is_junk(self):
        assert self.f.is_junk("Industry Baby Sped Up", "Lil Nas X")

    def test_mashup_triple_x_pattern_is_junk(self):
        # "A x B x C" indicates a mashup of three songs.
        assert self.f.is_junk("Money Trees x Blinding Lights x Levitating", "Mashup")

    def test_medley_is_junk(self):
        assert self.f.is_junk("80s Pop Medley", "Hits Artist")

    def test_sing_along_is_junk(self):
        assert self.f.is_junk("Sweet Caroline (Singalong Version)", "Boston Pop")

    def test_lofi_version_is_junk(self):
        assert self.f.is_junk("Sunflower - Lofi Version", "lofi artist")

    def test_nightcore_artist_is_junk(self):
        assert self.f.is_junk("Circles", "Nightcore Remixes")

    # ── seed_title_in_result ──────────────────────────────────────────────────

    def test_seed_title_in_result_mashup_detected(self):
        # Seed "Money Trees" — result "Money Trees x Blinding Lights"
        assert self.f.seed_title_in_result("Money Trees", "Money Trees x Blinding Lights")

    def test_seed_title_in_result_exact_match_flagged(self):
        # Same-title covers/alternate originals are prohibited recommendations.
        assert self.f.seed_title_in_result("Money Trees", "Money Trees")

    def test_seed_title_typo_variant_flagged(self):
        assert self.f.seed_title_in_result("Ornithology", "Orinthology")

    def test_legitimate_tribute_title_is_not_globally_suppressed(self):
        assert not self.f.is_junk("A Tribute To Someone", "Herbie Hancock")

    def test_seed_title_not_in_unrelated_result(self):
        assert not self.f.seed_title_in_result("Money Trees", "Alright")

    def test_seed_title_in_result_partial_overlap(self):
        # "Blinding Lights" seed — "The Blinding Lights Tribute" is a mashup/tribute.
        assert self.f.seed_title_in_result("Blinding Lights", "The Blinding Lights Tribute")

    # ── keep_mask ────────────────────────────────────────────────────────────

    def test_keep_mask_correct_shape_and_type(self):
        titles = ["Normal Song", "Slowed Remix", "Nightcore Edit", "Real Track"]
        artists = ["ArtistA", "ArtistB", "ArtistC", "ArtistD"]
        mask = self.f.keep_mask(titles, artists)
        assert mask.dtype == bool
        assert mask.shape == (4,)
        assert mask[0] and mask[3]     # normal tracks kept
        assert not mask[1] or not mask[2]  # at least one junk track caught

    def test_keep_mask_without_artists(self):
        titles = ["Song", "Song slowed reverb", "Real Song"]
        mask = self.f.keep_mask(titles)
        assert mask[0] and mask[2]
        assert not mask[1]

    def test_keep_mask_all_kept(self):
        titles = ["Master of Puppets", "Blinding Lights", "Take Five"]
        assert self.f.keep_mask(titles).all()

    def test_keep_mask_all_filtered(self):
        titles = ["Song slowed", "Nightcore version", "Karaoke Hits"]
        artists = ["ArtistA", "ArtistB", "Karaoke Universe"]
        assert not self.f.keep_mask(titles, artists).any()


def test_module_level_keep_mask():
    """Module-level keep_mask convenience function works."""
    titles = ["Real Song", "Slowed Version"]
    mask = keep_mask(titles)
    assert mask[0] and not mask[1]


def test_default_filter_singleton():
    """default_filter() returns the same instance on repeated calls."""
    f1 = default_filter()
    f2 = default_filter()
    assert f1 is f2


def test_extra_patterns():
    """Custom extra patterns are added on top of the defaults."""
    f = TitleQualityFilter(extra_title_patterns=[r"\bdemo\b"])
    assert f.is_junk("Blinding Lights (Demo)", "The Weeknd")
    assert not f.is_junk("Blinding Lights", "The Weeknd")


def test_quality_filter_integration_with_numpy_index():
    """keep_mask on a numpy array of titles (as strings) works correctly."""
    titles_arr = np.array(["Normal Song", "slowed + reverb", "Real Track"], dtype=object)
    artists_arr = np.array(["A", "B", "C"], dtype=object)
    mask = keep_mask(list(titles_arr), list(artists_arr))
    assert mask[0] and mask[2]
    assert not mask[1]
