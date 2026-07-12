"""Tests for the human-aligned evaluation suite (eval_suite.py).

Validates scene coherence scoring, junk detection, seed-title mashup detection,
the scene-inference logic, and the frozen baseline comparison helper.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np
import pytest

from soundalike.ml.eval_suite import (
    EVAL_SEEDS,
    HELD_OUT_SEEDS,
    EvalReport,
    EvalResult,
    _ARTIST_SCENE,
    _SCENE_RELATIVES,
    _aggregate,
    compare_reports,
    infer_scene,
    is_junk,
    is_seed_title_in_result,
    load_report,
    print_report,
    run_eval,
    save_report,
)


# ─────────────────────────────────────────────────────────────────────────────
# Seed catalogue validation
# ─────────────────────────────────────────────────────────────────────────────

class TestSeedCatalogue:

    def test_at_least_50_seeds(self):
        assert len(EVAL_SEEDS) >= 50, "Need ≥50 seeds for the acceptance criterion"

    def test_at_least_12_scenes(self):
        scenes = {s for _, _, s in EVAL_SEEDS}
        assert len(scenes) >= 12, f"Need ≥12 distinct scenes, found: {scenes}"

    def test_required_scenes_present(self):
        scenes = {s for _, _, s in EVAL_SEEDS}
        required = {"rap", "rnb", "indie", "shoegaze", "hyperpop", "electronic",
                    "metal", "jazz", "city_pop", "kpop", "latin", "afrobeats", "difficult"}
        missing = required - scenes
        assert not missing, f"Missing required scenes: {missing}"

    def test_all_seeds_have_three_fields(self):
        for seed in EVAL_SEEDS:
            assert len(seed) == 3, f"Seed must be (title, artist, scene): {seed}"
            title, artist, scene = seed
            assert title and artist and scene, f"Empty field in seed: {seed}"

    def test_seed_scenes_in_relatives_dict(self):
        scenes = {s for _, _, s in EVAL_SEEDS}
        # Every scene must have a relatives entry (even if it's just itself).
        for s in scenes:
            assert s in _SCENE_RELATIVES, f"Scene '{s}' missing from _SCENE_RELATIVES"


class TestHeldOutCatalogue:

    def test_held_out_exactly_20_seeds(self):
        assert len(HELD_OUT_SEEDS) == 20, \
            f"Held-out set must be exactly 20 seeds for the AC (got {len(HELD_OUT_SEEDS)})"

    def test_held_out_no_overlap_with_main_suite(self):
        """HELD_OUT_SEEDS must be fully disjoint from EVAL_SEEDS to avoid leakage."""
        main_keys = {(t.casefold(), a.casefold()) for t, a, _ in EVAL_SEEDS}
        held_keys = {(t.casefold(), a.casefold()) for t, a, _ in HELD_OUT_SEEDS}
        overlap = main_keys & held_keys
        assert not overlap, f"Seeds appear in both eval and held-out sets: {overlap}"

    def test_held_out_all_have_three_fields(self):
        for seed in HELD_OUT_SEEDS:
            assert len(seed) == 3
            title, artist, scene = seed
            assert title and artist and scene

    def test_held_out_scenes_in_relatives_dict(self):
        for _, _, scene in HELD_OUT_SEEDS:
            assert scene in _SCENE_RELATIVES, \
                f"Held-out scene '{scene}' not in _SCENE_RELATIVES"


# ─────────────────────────────────────────────────────────────────────────────
# Junk detection
# ─────────────────────────────────────────────────────────────────────────────

class TestIsJunk:

    def test_normal_track_not_junk(self):
        assert not is_junk("Blinding Lights", "The Weeknd")

    def test_slowed_reverb_is_junk(self):
        assert is_junk("Blinding Lights slowed reverb", "")

    def test_karaoke_artist_is_junk(self):
        assert is_junk("Shape of You", "Karaoke Universe")

    def test_tribute_in_title_is_junk(self):
        assert is_junk("Tribute to Metallica: Master of Puppets", "")

    def test_nightcore_is_junk(self):
        assert is_junk("Nightcore - Take Me to Church", "Nightcore")

    def test_remaster_not_junk(self):
        # "(Remastered 2015)" is a legitimate version, not junk.
        assert not is_junk("Money Trees (Remastered 2015)", "Kendrick Lamar")

    def test_medley_is_junk(self):
        assert is_junk("80s Hits Medley", "Studio Singers")

    def test_mashup_is_junk(self):
        assert is_junk("80s Pop Mashup", "Various Artists")


class TestIsSeedTitleInResult:

    def test_seed_title_mashup_detected(self):
        assert is_seed_title_in_result("Money Trees", "Money Trees x Blinding Lights")

    def test_seed_title_same_as_result_is_flagged(self):
        # A result with the exact same title as the seed IS flagged: it may be
        # a cover, tribute, or version from a different artist (all undesirable).
        assert is_seed_title_in_result("Money Trees", "Money Trees")

    def test_unrelated_result_not_flagged(self):
        assert not is_seed_title_in_result("Money Trees", "Alright")

    def test_empty_inputs_not_flagged(self):
        assert not is_seed_title_in_result("", "Something")
        assert not is_seed_title_in_result("Something", "")

    def test_partial_overlap_flagged(self):
        assert is_seed_title_in_result("Blinding Lights", "Blinding Lights x Levels")


# ─────────────────────────────────────────────────────────────────────────────
# Scene inference
# ─────────────────────────────────────────────────────────────────────────────

class TestInferScene:

    def test_known_artist_returns_scene(self):
        assert infer_scene("Blinding Lights", "The Weeknd") == "rnb"

    def test_known_rap_artist(self):
        assert infer_scene("HUMBLE.", "Kendrick Lamar") == "rap"

    def test_known_metal_artist(self):
        assert infer_scene("Master of Puppets", "Metallica") == "metal"

    def test_known_jazz_artist(self):
        assert infer_scene("Take Five", "Dave Brubeck Quartet") == "jazz"

    def test_known_kpop_artist(self):
        assert infer_scene("ETA", "NewJeans") == "kpop"

    def test_known_city_pop_artist(self):
        assert infer_scene("Plastic Love", "Mariya Takeuchi") == "city_pop"

    def test_unknown_returns_none(self):
        assert infer_scene("Track Title", "Completely Unknown Artist XYZ") is None

    def test_keyword_fallback_rap(self):
        # Even without a known artist, "hip-hop" keyword → rap.
        result = infer_scene("hip-hop vibes", "Unknown Artist")
        assert result == "rap"

    def test_keyword_fallback_metal(self):
        result = infer_scene("black metal riffs", "Unknown Band")
        assert result == "metal"

    def test_keyword_fallback_jazz(self):
        result = infer_scene("jazz improvisation", "Unknown Quartet")
        assert result == "jazz"

    def test_artist_scene_mapping_not_empty(self):
        assert len(_ARTIST_SCENE) >= 100, "Artist→scene mapping should be comprehensive"

    def test_comma_in_artist_name_handled(self):
        # "Kendrick Lamar, J. Cole" → should match on "kendrick lamar"
        # (real-world data sometimes has featuring in artist field)
        scene = infer_scene("Track", "Kendrick Lamar, J. Cole")
        assert scene == "rap"


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation runner with a mock recommender
# ─────────────────────────────────────────────────────────────────────────────

class MockRecommender:
    """Minimal recommender stub for testing run_eval().

    Simulates an 87k-size library index with a handful of songs. It always
    returns the specified ``results`` for any seed, regardless of the actual
    row requested.
    """

    def __init__(self, results: List[Dict[str, Any]], library_titles, library_artists):
        self.titles = np.array(library_titles, dtype=object)
        self.artists = np.array(library_artists, dtype=object)
        self._results = results

    def find_row(self, title: str, artist: str = "") -> int:
        # Find the first matching row in our mini library.
        for i, (t, a) in enumerate(zip(self.titles, self.artists)):
            if title.lower() in t.lower() and (not artist or artist.lower() in a.lower()):
                return i
        return None

    def recommend(self, row: int, n: int = 5, **kwargs) -> Dict:
        return {"ok": True, "results": self._results[:n], "library_size": len(self.titles)}


def _scene_coherent_recs(scene_artist_pairs):
    """Build a rec list with known-scene artists."""
    return [{"title": f"Track {i}", "artist": a, "score": 1.0 - 0.1 * i,
             "deezer_id": i + 1000}
            for i, (_, a) in enumerate(scene_artist_pairs)]


class TestRunEval:

    def setup_method(self):
        # Library has 10 seeds from different scenes.
        lib_titles = [title for title, _, _ in EVAL_SEEDS[:10]]
        lib_artists = [artist for _, artist, _ in EVAL_SEEDS[:10]]
        # Recommendations: all from known same-scene artists.
        rnb_recs = _scene_coherent_recs([
            ("rnb", "The Weeknd"), ("rnb", "SZA"), ("rnb", "Frank Ocean"),
            ("rnb", "H.E.R."), ("rnb", "Bryson Tiller"),
        ])
        self.mock = MockRecommender(rnb_recs, lib_titles, lib_artists)

    def test_run_eval_returns_report(self):
        seeds = [("Money Trees", "Kendrick Lamar", "rap"),
                 ("Blinding Lights", "The Weeknd", "rnb")]
        # Only rnb seed can be found in the mock library.
        report = run_eval(self.mock, seeds=seeds, method_name="test")
        assert isinstance(report, EvalReport)
        assert report.method_name == "test"
        assert report.n_seeds == 2

    def test_run_eval_not_found_counted(self):
        seeds = [("A Song Not In Library", "Unknown Artist", "indie")]
        report = run_eval(self.mock, seeds=seeds)
        assert report.n_found == 0
        assert report.n_seeds == 1

    def test_run_eval_coherence_for_known_scene(self):
        """Recommendations by known same-scene artists should score coherently."""
        seeds = [("Blinding Lights", "The Weeknd", "rnb")]
        report = run_eval(self.mock, seeds=seeds)
        if report.n_found == 1:
            # R&B recs → scene_coherent should be high (all from R&B artists).
            assert report.primary_score > 0.5, \
                f"Expected high coherence for R&B recs, got {report.primary_score:.3f}"

    def test_run_eval_per_scene_breakdown(self):
        seeds = list(EVAL_SEEDS[:5])
        report = run_eval(self.mock, seeds=seeds)
        # per_scene should be populated for found seeds.
        for scene, stats in report.per_scene.items():
            assert "coherence" in stats
            assert "junk_rate" in stats
            assert "n_seeds" in stats


# ─────────────────────────────────────────────────────────────────────────────
# Aggregate + compare
# ─────────────────────────────────────────────────────────────────────────────

def _make_result(scene, coherent=True, junk=False, same_artist=False):
    rec = {"title": "Some Track", "artist": "Some Artist", "score": 0.8}
    return EvalResult(
        seed_title="Seed", seed_artist="Seed Artist", seed_scene=scene,
        found_in_index=True, recs=[rec] * 5,
        junk_flags=[junk] * 5,
        seed_mashup_flags=[False] * 5,
        same_artist_flags=[same_artist] * 5,
        scene_coherent_flags=[coherent] * 5,
        scene_tags=[scene] * 5,
    )


class TestAggregate:

    def test_primary_score_all_coherent(self):
        results = [_make_result("rap", coherent=True) for _ in range(5)]
        report = _aggregate(results, "test")
        assert abs(report.primary_score - 1.0) < 1e-5

    def test_primary_score_none_coherent(self):
        results = [_make_result("rap", coherent=False) for _ in range(5)]
        report = _aggregate(results, "test")
        assert abs(report.primary_score - 0.0) < 1e-5

    def test_junk_rate_all_junk(self):
        results = [_make_result("rap", junk=True) for _ in range(3)]
        report = _aggregate(results, "test")
        assert abs(report.junk_rate - 1.0) < 1e-5

    def test_junk_rate_no_junk(self):
        results = [_make_result("rap", junk=False) for _ in range(3)]
        report = _aggregate(results, "test")
        assert abs(report.junk_rate - 0.0) < 1e-5

    def test_not_found_seeds_excluded(self):
        found = [_make_result("rap", coherent=True)]
        not_found = [EvalResult("T", "A", "rap", found_in_index=False)]
        results = found + not_found
        report = _aggregate(results, "test")
        assert report.n_found == 1
        assert report.n_seeds == 2
        assert abs(report.primary_score - 1.0) < 1e-5

    def test_per_scene_populated(self):
        results = (
            [_make_result("rap", coherent=True)] * 3 +
            [_make_result("metal", coherent=False)] * 2
        )
        report = _aggregate(results, "test")
        assert "rap" in report.per_scene
        assert "metal" in report.per_scene
        assert abs(report.per_scene["rap"]["coherence"] - 1.0) < 1e-5
        assert abs(report.per_scene["metal"]["coherence"] - 0.0) < 1e-5

    def test_zero_found_seeds(self):
        results = [EvalResult("T", "A", "rap", found_in_index=False)]
        report = _aggregate(results, "empty")
        assert report.primary_score == 0.0
        assert report.n_found == 0


class TestCompareReports:

    def test_positive_gain(self):
        baseline = {"primary_score": 0.5, "per_scene": {}, "junk_rate": 0.1}
        challenger = EvalReport(
            n_seeds=10, n_found=10, primary_score=0.65,
            top1_coherent=0.7, junk_rate=0.05, mashup_rate=0.0,
            same_artist_rate=0.0, per_scene={}, method_name="enh"
        )
        result = compare_reports(baseline, challenger)
        assert result["primary_relative_gain"] > 0.0
        assert result["baseline_primary"] == 0.5
        assert result["challenger_primary"] == 0.65

    def test_twenty_percent_gain_detected(self):
        """compare_reports must correctly compute a 20% relative gain."""
        baseline = {"primary_score": 0.5, "per_scene": {}, "junk_rate": 0.1}
        challenger = EvalReport(
            n_seeds=10, n_found=10, primary_score=0.60,  # +20% from 0.5
            top1_coherent=0.7, junk_rate=0.05, mashup_rate=0.0,
            same_artist_rate=0.0, per_scene={}, method_name="enh"
        )
        result = compare_reports(baseline, challenger)
        assert abs(result["primary_relative_gain"] - 0.20) < 0.01

    def test_no_regression(self):
        """A positive primary gain with no per-scene regressions is valid."""
        baseline = {
            "primary_score": 0.5,
            "per_scene": {"rap": {"coherence": 0.6}, "metal": {"coherence": 0.4}},
            "junk_rate": 0.1,
        }
        challenger = EvalReport(
            n_seeds=10, n_found=10, primary_score=0.62,
            top1_coherent=0.7, junk_rate=0.04, mashup_rate=0.0,
            same_artist_rate=0.0,
            per_scene={"rap": {"coherence": 0.65, "junk_rate": 0.0, "n_seeds": 3},
                       "metal": {"coherence": 0.44, "junk_rate": 0.0, "n_seeds": 2}},
            method_name="enh"
        )
        result = compare_reports(baseline, challenger)
        # Both scenes should be ≥0 (no regression > 10%).
        for scene, delta in result["per_scene_relative_delta"].items():
            assert delta >= -0.10, f"Scene {scene} regressed by {delta:.2%}"


# ─────────────────────────────────────────────────────────────────────────────
# Save / load JSON
# ─────────────────────────────────────────────────────────────────────────────

def test_save_and_load_report(tmp_path):
    results = [_make_result("rap", coherent=True) for _ in range(3)]
    report = _aggregate(results, "saved_test")
    path = tmp_path / "test_report.json"
    save_report(report, path)
    loaded = load_report(path)
    assert loaded["method_name"] == "saved_test"
    assert abs(loaded["primary_score"] - 1.0) < 1e-5


def test_save_report_creates_parent_directory(tmp_path):
    results = [_make_result("jazz", coherent=False)]
    report = _aggregate(results, "jazz_test")
    path = tmp_path / "subdir" / "report.json"
    save_report(report, path)
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["n_found"] == 1


def test_report_to_dict_serialisable():
    """EvalReport.to_dict() must produce a JSON-serialisable dict."""
    results = [_make_result("electronic", coherent=True)]
    report = _aggregate(results, "serialise_test")
    d = report.to_dict()
    json.dumps(d)  # must not raise


def test_print_report_does_not_crash():
    results = [_make_result("shoegaze", coherent=True)]
    report = _aggregate(results, "print_test")
    # Should write to stdout without raising.
    import io, sys
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        print_report(report)
    finally:
        sys.stdout = old_stdout
