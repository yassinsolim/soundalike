"""Tests for the Vercel web recommender (webapp/api/_reco.py).

The hosted library-mode recommender is a torch-free numpy reimplementation of
DeepVibeRecommender. These tests pin it to the canonical recommender so the two
can never silently diverge, and cover the query parser.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_API = Path(__file__).resolve().parents[1] / "webapp" / "api"
sys.path.insert(0, str(_API))


def _synthetic_index(
    tmp_path, n_artists=60, per=5, dim=48, seed=0,
    sonic=False, dual=False,
):
    """Build + save a small DeepVibeIndex so both recommenders read the same data."""
    from soundalike.ml.deepvibe import DeepVibeIndex

    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_artists, dim))
    neural, vibe, titles, artists, tids = [], [], [], [], []
    k = 0
    for a in range(n_artists):
        for j in range(per):
            neural.append(centers[a] + 0.2 * rng.standard_normal(dim))
            vibe_row = rng.standard_normal(29)
            vibe_row[0] = 80 + (k % 80)
            vibe.append(vibe_row)
            titles.append(f"song {k}")
            artists.append(f"artist {a}")
            tids.append(1000 + k)
            k += 1
    sonic_matrix = (
        rng.standard_normal((len(tids), 64)).astype(np.float16)
        if sonic or dual else None
    )
    clap_matrix = (
        rng.standard_normal((len(tids), 64)).astype(np.float16) if dual else None
    )
    wiki = rng.integers(0, 6, len(tids)).astype(np.float16) if dual else None
    wiki_specific = rng.integers(0, 2, len(tids)).astype(np.uint8) if dual else None
    idx = DeepVibeIndex(
        np.array(tids), np.array(titles, object), np.array(artists, object),
        np.asarray(neural, np.float32), np.asarray(vibe, np.float32),
        sonic_matrix, clap_matrix, wiki, wiki_specific,
    )
    p = tmp_path / "idx.npz"
    idx.save(p)
    return p, idx


def test_index_checksum_helper(tmp_path):
    from _reco import _sha256

    path = tmp_path / "index.npz"
    path.write_bytes(b"soundalike")
    assert _sha256(str(path)) == (
        "8ef7e84df18a9be28b16191183e83db57606492021a2f2faf4604a1670475d90"
    )


def test_old_and_sonic_index_roundtrip_compatibility(tmp_path):
    from soundalike.ml.deepvibe import DeepVibeIndex

    old_path, old = _synthetic_index(tmp_path, seed=3)
    assert DeepVibeIndex.load(old_path).sonic is None
    old.sonic = np.arange(len(old) * 64, dtype=np.float16).reshape(len(old), 64)
    new_path = tmp_path / "new.npz"
    old.save(new_path, half=True)
    loaded = DeepVibeIndex.load(new_path)
    assert loaded.sonic.dtype == np.float16
    assert np.array_equal(loaded.sonic, old.sonic)


def test_web_recommender_matches_canonical(tmp_path):
    """Baseline (no enhancements) must exactly match the canonical numpy recommender."""
    import os
    os.environ["SOUNDALIKE_INDEX_PATH"] = ""  # force explicit path use
    from _reco import WebRecommender
    from soundalike.ml.deepvibe import DeepVibeIndex, DeepVibeRecommender
    from soundalike.audio.vibe import VibeFeatures

    path, idx = _synthetic_index(tmp_path)
    # enhance=False → plain neural+vibe blend on both sides, must be identical
    web = WebRecommender(str(path), enhance=False)
    canon = DeepVibeRecommender(DeepVibeIndex.load(path), alpha=0.8, whiten=True,
                                enhance=False)

    for row in (0, 37, 111, 200, 250):
        w = web.recommend(row, n=15, alpha=0.8, diversity=0.15, max_per_artist=1)
        c = canon.recommend(
            np.asarray(idx.neural[row], np.float32),
            VibeFeatures.from_vector(np.asarray(idx.vibe[row], np.float32)),
            n=15, exclude_ids={int(idx.track_ids[row])},
            exclude_artist=None, diversity=0.15, max_per_artist=1,
            quality_filter=False, genre_rerank=False, related_boost=False)
        assert [(x["title"], x["artist"]) for x in w["results"]] == \
               [(r.title, r.artist) for r in c], f"mismatch at row {row}"


def test_web_recommender_can_return_same_artist_candidates(tmp_path):
    from _reco import WebRecommender

    path, _ = _synthetic_index(tmp_path, n_artists=4, per=4, seed=11)
    recommender = WebRecommender(str(path), enhance=True)
    result = recommender.recommend(0, n=5, diversity=0.0, max_per_artist=1)
    artists = [item["artist"] for item in result["results"]]
    assert "artist 0" in artists


def test_web_recommender_penalizes_versioned_variants(tmp_path):
    from _reco import WebRecommender
    from soundalike.ml.deepvibe import DeepVibeIndex

    neural = np.array([
        [0.0, 0.0, 1.0, 0.0],
        [0.2, 0.0, 0.98, 0.0],
        [0.2, 0.0, 0.98, 0.0],
    ], dtype=np.float32)
    vibe = np.zeros((3, 29), dtype=np.float32)
    vibe[:, 0] = [120, 121, 121]
    idx = DeepVibeIndex(
        np.array([1, 2, 3]),
        np.array(["Seed Song", "Echoes", "Echoes (Remix)"], dtype=object),
        np.array(["Seed Artist", "Artist A", "Artist B"], dtype=object),
        neural,
        vibe,
    )
    path = tmp_path / "versioned.npz"
    idx.save(path)
    recommender = WebRecommender(str(path), enhance=True)
    result = recommender.recommend(0, n=2, diversity=0.0, max_per_artist=1)

    assert [item["title"] for item in result["results"]][:2] == [
        "Echoes", "Echoes (Remix)"
    ]


def test_version_penalty_uses_title_metadata_not_artist_names():
    from _reco import _version_penalty as hosted_penalty
    from soundalike.ml.deepvibe import _version_penalty as canonical_penalty

    cases = [
        ("Live Forever", "Oasis", 0.0),
        ("Normal Song", "Little Mix", 0.0),
        ("Normal Song", "Live", 0.0),
        ("Hard to Live in the City", "Albert Hammond Jr.", 0.0),
        ("The Folks Who Live on the Hill", "Peggy Lee", 0.0),
        ("Free (You Got To Live)", "Ultra Naté", 0.0),
        ("Echoes (Acoustic)", "Artist", 0.10),
        ("Echoes - Live", "Artist", 0.10),
        ("Echoes (Club Remix)", "Artist", 0.30),
    ]
    for title, artist, expected in cases:
        assert canonical_penalty(title, artist) == expected
        assert hosted_penalty(title, artist) == expected


def test_version_penalty_is_disabled_for_unmodified_baseline(tmp_path):
    from _reco import WebRecommender

    path, _ = _synthetic_index(tmp_path, seed=12)
    baseline = WebRecommender(str(path), enhance=False)
    enhanced = WebRecommender(str(path), enhance=True)

    assert not np.any(baseline._version_penalty)
    assert np.array_equal(
        enhanced._version_penalty, enhanced._version_penalty_policy
    )



def test_model_quality_family_collapse_and_audio_penalties_match_hosted(tmp_path):
    from _reco import WebRecommender
    from soundalike.ml.deepvibe import DeepVibeIndex, DeepVibeRecommender

    rng = np.random.default_rng(91)
    titles = np.array([
        "Seed", "Echoes (Live)", "Echoes", "Echoes (Acoustic)",
        "Live Forever", "Different Song", "Echoes (Remix)",
    ], dtype=object)
    artists = np.array([
        "Seed Artist", "Artist A", "Artist A", "Artist A",
        "Artist A", "Artist A", "Artist B",
    ], dtype=object)
    vibe = rng.normal(size=(len(titles), 29)).astype(np.float32)
    vibe[:, 0] = [120, 60, 120, 205, 124, 128, 132]
    vibe[:, 5:8] = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    index = DeepVibeIndex(
        np.arange(1, len(titles) + 1), titles, artists,
        rng.normal(size=(len(titles), 12)).astype(np.float32), vibe,
    )
    path = tmp_path / "model-quality.npz"
    index.save(path)
    canonical = DeepVibeRecommender(index, enhance=True)
    hosted = WebRecommender(str(path), enhance=True)

    rows = [1, 3, 2, 4, 5, 6]
    expected = [2, 4, 5, 6]
    assert canonical._collapse_recording_families(rows, n=10) == expected
    assert hosted._collapse_recording_families(rows, n=10) == expected
    assert canonical._collapse_recording_families([3, 1], n=10) == [3]

    penalty_rows = np.array([1, 2, 3])
    canonical_penalties = canonical._audio_compatibility_penalty(
        index.vibe[0], penalty_rows
    )
    hosted_penalties = hosted._audio_compatibility_penalty(0, penalty_rows)
    assert np.allclose(canonical_penalties, hosted_penalties)
    assert canonical_penalties[0] == pytest.approx(0.0, abs=1e-7)
    assert canonical_penalties[1] == pytest.approx(0.0, abs=1e-7)
    assert canonical_penalties[2] > canonical_penalties[1]
    assert np.all(canonical_penalties <= 0.070001)

    index.vibe[3, :] = np.nan
    canonical._vscaled[3, :] = np.nan
    hosted._vscaled[3, :] = np.nan
    assert canonical._audio_compatibility_penalty(index.vibe[0], [3])[0] == 0
    assert hosted._audio_compatibility_penalty(0, [3])[0] == 0


def test_model_quality_guardrail_breaks_audio_ties_toward_compatible_pacing(
    tmp_path,
):
    from _reco import WebRecommender
    from soundalike.ml.deepvibe import DeepVibeIndex, DeepVibeRecommender

    compact = np.zeros((4, 64), dtype=np.float32)
    compact[0, 0] = 1.0
    compact[1:3, 0] = 0.9
    compact[1:3, 1] = np.sqrt(1.0 - 0.9 ** 2)
    compact[3, 0] = 0.5
    compact[3, 1] = np.sqrt(1.0 - 0.5 ** 2)
    vibe = np.zeros((4, 29), dtype=np.float32)
    vibe[:, 0] = [120, 120, 205, 120]
    index = DeepVibeIndex(
        np.arange(4),
        ["Seed", "Compatible", "Extreme Tempo", "Other"],
        ["Seed Artist", "Artist A", "Artist B", "Artist C"],
        np.eye(4, 12, dtype=np.float32),
        vibe,
        compact,
        compact,
        np.zeros(4, dtype=np.float16),
        np.zeros(4, dtype=np.uint8),
    )
    path = tmp_path / "pacing-guardrail.npz"
    index.save(path)
    canonical = DeepVibeRecommender(index, enhance=True)
    hosted = WebRecommender(str(path), enhance=True)

    canonical_tail = canonical._recommend_dual_tail(
        index.sonic[0],
        index.clap[0],
        index.vibe[0],
        n=3,
        exclude_ids={0},
        exclude_artist=None,
        seed_title="Seed",
    )
    hosted_tail = hosted._recommend_dual_tail(0, n=3)

    assert canonical_tail[0].title == "Compatible"
    assert [item.track_id for item in canonical_tail] == [
        item["deezer_id"] for item in hosted_tail
    ]


def test_spicetify_results_include_measured_bpm(tmp_path):
    from _reco import WebRecommender
    from spicetify_recommend import _enrich_result_tempos, _tempo_bpm

    path, index = _synthetic_index(tmp_path)
    recommender = WebRecommender(str(path), enhance=False)
    output = _enrich_result_tempos(recommender, recommender.recommend(0, n=5))

    assert _tempo_bpm(recommender, 0) == round(float(index.vibe[0][0]))
    assert all(item["bpm"] is not None for item in output["results"])
    for item in output["results"]:
        row = int(np.flatnonzero(recommender.track_ids == item["deezer_id"])[0])
        assert item["bpm"] == round(float(index.vibe[row][0]))


def test_spicetify_bpm_uses_track_id_for_duplicate_titles():
    from spicetify_recommend import _enrich_result_tempos

    class Recommender:
        feature_names = ["tempo"]
        track_ids = np.asarray([101, 202])
        _vscaled = np.asarray([[90.0], [130.0]])
        _w = np.asarray([1.0])
        _vstd = np.asarray([1.0])
        _vmean = np.asarray([0.0])

        @staticmethod
        def find_row(_title, _artist):
            return 0

    payload = {
        "results": [{
            "deezer_id": 202,
            "title": "Same Song",
            "artist": "Same Artist",
        }]
    }

    assert _enrich_result_tempos(Recommender(), payload)["results"][0]["bpm"] == 130


def test_spicetify_query_canonicalization_uses_decoded_values():
    from spicetify_recommend import (
        _language_policy_supported,
        _needs_canonical_redirect,
    )

    params = {
        "query": ["Blinding Lights — The Weeknd"],
        "n": ["20"],
        "diversity": ["0.15"],
        "v": ["2"],
    }

    assert not _needs_canonical_redirect(
        params, "Blinding Lights — The Weeknd", 20, 0.15
    )
    params["n"] = ["020"]
    assert _needs_canonical_redirect(
        params, "Blinding Lights — The Weeknd", 20, 0.15
    )

    params = {
        "query": ["Blinding Lights — The Weeknd"],
        "n": ["40"],
        "diversity": ["0.15"],
        "v": ["3"],
        "language_policy": ["spotify-lyrics-v1"],
    }
    assert not _needs_canonical_redirect(
        params,
        "Blinding Lights — The Weeknd",
        40,
        0.15,
        "3",
        "spotify-lyrics-v1",
    )
    params["warm"] = ["1"]
    assert not _needs_canonical_redirect(
        params,
        "Blinding Lights — The Weeknd",
        40,
        0.15,
        "3",
        "spotify-lyrics-v1",
        True,
    )

    params = {
        "query": ["Blinding Lights — The Weeknd"],
        "n": ["50"],
        "diversity": ["0.15"],
        "v": ["4"],
        "language_policy": ["spotify-lyrics-strict-v2"],
    }
    assert not _needs_canonical_redirect(
        params,
        "Blinding Lights — The Weeknd",
        50,
        0.15,
        "4",
        "spotify-lyrics-strict-v2",
    )
    params["ranking_policy"] = ["model-quality-v1"]
    assert not _needs_canonical_redirect(
        params,
        "Blinding Lights — The Weeknd",
        50,
        0.15,
        "4",
        "spotify-lyrics-strict-v2",
        False,
        "model-quality-v1",
    )
    assert _needs_canonical_redirect(
        {key: value for key, value in params.items() if key != "ranking_policy"},
        "Blinding Lights — The Weeknd",
        50,
        0.15,
        "4",
        "spotify-lyrics-strict-v2",
        False,
        "model-quality-v1",
    )
    assert _language_policy_supported("2", None)
    assert _language_policy_supported("3", "spotify-lyrics-v1")
    assert _language_policy_supported("4", "spotify-lyrics-strict-v2")
    assert not _language_policy_supported("4", "spotify-lyrics-v1")


def test_spicetify_v4_endpoint_accepts_only_the_strict_policy(monkeypatch):
    import spicetify_recommend

    class Recommender:
        feature_names = ["tempo"]
        track_ids = np.asarray([101, 202])
        _vscaled = np.asarray([[100.0], [120.0]])
        _w = np.asarray([1.0])
        _vstd = np.asarray([1.0])
        _vmean = np.asarray([0.0])

        @staticmethod
        def find_row(_title, _artist):
            return 0

        @staticmethod
        def recommend(_row, **_kwargs):
            return {
                "ok": True,
                "results": [{
                    "deezer_id": 202,
                    "title": "Candidate",
                    "artist": "Artist",
                }],
                "vibe": {},
            }

    monkeypatch.setattr(spicetify_recommend, "get_recommender", Recommender)
    request = spicetify_recommend.handler.__new__(spicetify_recommend.handler)
    sent = []
    request._send = lambda code, body, cacheable=True: sent.append(
        (code, body, cacheable)
    )
    request._redirect = lambda _location: pytest.fail("unexpected redirect")
    request.path = (
        "/api/spicetify_recommend?query=Seed&n=1&diversity=0.15&v=4"
        "&language_policy=spotify-lyrics-strict-v2&ranking_policy=model-quality-v1"
    )

    request.do_GET()

    assert sent[0][0] == 200
    assert sent[0][1]["language_policy"] == "spotify-lyrics-strict-v2"
    assert sent[0][1]["ranking_policy"] == "model-quality-v1"
    assert sent[0][1]["results"][0]["bpm"] == 120

    sent.clear()
    request.path = (
        "/api/spicetify_recommend?query=Seed&n=1&diversity=0.15&v=4"
        "&language_policy=spotify-lyrics-v1"
    )
    request.do_GET()
    assert sent == [
        (400, {"ok": False, "error": "unsupported language policy"}, True)
    ]


def test_enhanced_recommender_differs_from_baseline(tmp_path):
    """Enhanced mode must produce different (scene-improved) results from baseline."""
    from _reco import WebRecommender

    path, _ = _synthetic_index(tmp_path, n_artists=60, per=5, dim=48)
    web_base = WebRecommender(str(path), enhance=False)
    web_enh = WebRecommender(str(path), enhance=True)

    # With clustering in synthetic data, enhancements should shift the ranking.
    # At minimum the recommender runs without error.
    for row in (0, 100, 200):
        base_out = web_base.recommend(row, n=10)
        enh_out = web_enh.recommend(row, n=10)
        assert base_out["ok"] and enh_out["ok"]
        assert len(base_out["results"]) > 0 and len(enh_out["results"]) > 0
        assert base_out["retrieval_mode"] == "legacy_no_sonic_seed"


def test_enhanced_web_recommender_matches_canonical(tmp_path):
    """The shipped guarded winner must be identical on desktop and hosted paths."""
    from _reco import WebRecommender
    from soundalike.ml.deepvibe import DeepVibeIndex, DeepVibeRecommender
    from soundalike.audio.vibe import VibeFeatures

    path, idx = _synthetic_index(tmp_path, n_artists=60, per=5, dim=48, seed=9)
    web = WebRecommender(str(path), enhance=True)
    canon = DeepVibeRecommender(DeepVibeIndex.load(path), alpha=0.8, whiten=True,
                                enhance=True)
    for row in (0, 37, 111, 200):
        hosted = web.recommend(row, n=15, alpha=0.8, diversity=0.15,
                               max_per_artist=1)
        desktop = canon.recommend(
            np.asarray(idx.neural[row], np.float32),
            VibeFeatures.from_vector(np.asarray(idx.vibe[row], np.float32)),
            n=15,
            exclude_ids={int(idx.track_ids[row])},
            exclude_artist=None,
            seed_title=str(idx.titles[row]),
            seed_artist=str(idx.artists[row]),
            diversity=0.15,
            max_per_artist=1,
        )
        assert [(item["title"], item["artist"]) for item in hosted["results"]] == [
            (item.title, item.artist) for item in desktop
        ], f"enhanced mismatch at row {row}"
        assert hosted["ranking_policy"] == "model-quality-v1"


def test_sonic_hosted_matches_canonical_and_reports_diagnostics(tmp_path):
    from _reco import WebRecommender
    from soundalike.ml.deepvibe import DeepVibeIndex, DeepVibeRecommender
    from soundalike.audio.vibe import VibeFeatures

    path, idx = _synthetic_index(
        tmp_path, n_artists=60, per=5, dim=48, seed=41, sonic=True
    )
    hosted = WebRecommender(str(path), enhance=True)
    desktop = DeepVibeRecommender(DeepVibeIndex.load(path), enhance=True)
    for row in (0, 111):
        web_result = hosted.recommend(row, n=20)
        canonical = desktop.recommend(
            idx.neural[row], VibeFeatures.from_vector(idx.vibe[row]), n=20,
            exclude_ids={int(idx.track_ids[row])},
            exclude_artist=None, seed_title=str(idx.titles[row]),
            diversity=.15, max_per_artist=1, seed_row=row,
        )
        assert [(item["title"], item["artist"]) for item in web_result["results"]] == [
            (item.title, item.artist) for item in canonical
        ]
        assert web_result["retrieval_mode"] == "sonic64_stable_head"
        assert web_result["method"] == "sonic64_stable_head"
        assert web_result["index_version"] == "2026.07.11-dual-sonic64"


def test_dual_sonic_hosted_matches_canonical_and_preserves_guardrails(tmp_path):
    from _reco import WebRecommender
    from soundalike.ml.deepvibe import DeepVibeIndex, DeepVibeRecommender
    from soundalike.audio.vibe import VibeFeatures

    path, idx = _synthetic_index(
        tmp_path, n_artists=60, per=5, dim=48, seed=44, dual=True
    )
    hosted = WebRecommender(str(path), enhance=True)
    desktop = DeepVibeRecommender(DeepVibeIndex.load(path), enhance=True)
    legacy = _synthetic_index(
        tmp_path, n_artists=60, per=5, dim=48, seed=44
    )[0]
    legacy_head = WebRecommender(str(legacy), enhance=True).recommend(
        0, n=5
    )["results"]
    web_result = hosted.recommend(0, n=20)
    canonical = desktop.recommend(
        idx.neural[0], VibeFeatures.from_vector(idx.vibe[0]), n=20,
        exclude_ids={int(idx.track_ids[0])},
        exclude_artist=None, seed_title=str(idx.titles[0]),
        diversity=.15, max_per_artist=1, seed_row=0,
    )
    assert [(item["title"], item["artist"]) for item in web_result["results"]] == [
        (item.title, item.artist) for item in canonical
    ]
    assert web_result["results"][:5] == legacy_head
    assert web_result["method"] == "dual_sonic64_guardrail"
    assert web_result["index_version"] == "2026.07.11-dual-sonic64"


def test_dual_tail_priors_cannot_bypass_audio_candidate_gate(tmp_path):
    from _reco import WebRecommender
    from soundalike.ml.deepvibe import DeepVibeIndex, DeepVibeRecommender

    rng = np.random.default_rng(45)
    count = 1_002
    cosine = np.linspace(0.99, 0.10, count - 2, dtype=np.float32)
    compact = np.zeros((count, 64), dtype=np.float32)
    compact[0, 0] = 1.0
    compact[1:-1, 0] = cosine
    compact[1:-1, 1] = np.sqrt(1.0 - cosine * cosine)
    compact[-1, 1] = 1.0
    titles = [f"song {i}" for i in range(count)]
    titles[999] = "song 999 (Remix)"
    idx = DeepVibeIndex(
        np.arange(10_000, 10_000 + count),
        titles,
        [f"artist {i}" for i in range(count)],
        rng.normal(size=(count, 12)).astype(np.float32),
        rng.normal(size=(count, 29)).astype(np.float32),
        compact.astype(np.float16),
        compact.astype(np.float16),
        np.zeros(count, dtype=np.float16),
        np.zeros(count, dtype=np.uint8),
    )
    path = tmp_path / "hub-gate.npz"
    idx.save(path)
    canonical = DeepVibeRecommender(idx, enhance=True)
    hosted = WebRecommender(str(path), enhance=True)

    promoted_row = 900
    hub_row = count - 1
    for recommender in (canonical, hosted):
        recommender._wiki[promoted_row] = 100.0
        recommender._wiki[hub_row] = 200.0

    old_ungated_score = (
        0.25 * canonical._zscore(
            canonical._compact_cosine(canonical._sonic, idx.sonic[0])
        )
        + 0.75 * canonical._zscore(
            canonical._compact_cosine(canonical._clap, idx.clap[0])
        )
        + 0.20 * canonical._wiki
        + 0.10 * canonical._wiki_specific
        - canonical._version_penalty
    )
    assert int(np.argmax(old_ungated_score)) == hub_row

    canonical_tail = canonical._recommend_dual_tail(
        idx.sonic[0],
        idx.clap[0],
        idx.vibe[0],
        n=20,
        exclude_ids={int(idx.track_ids[0])},
        exclude_artist=None,
        seed_title=str(idx.titles[0]),
    )
    hosted_tail = hosted._recommend_dual_tail(0, n=20)
    canonical_ids = [item.track_id for item in canonical_tail]
    hosted_ids = [item["deezer_id"] for item in hosted_tail]

    assert canonical_ids == hosted_ids
    assert int(idx.track_ids[promoted_row]) == canonical_ids[0]
    assert int(idx.track_ids[hub_row]) not in canonical_ids
    oversized = hosted._recommend_dual_tail(0, n=count)
    oversized_ids = {item["deezer_id"] for item in oversized}
    assert len(oversized) == 999
    assert int(idx.track_ids[999]) in oversized_ids
    assert int(idx.track_ids[1000]) not in oversized_ids


def test_sonic_stable_head_is_exact_and_tail_changes(tmp_path):
    from _reco import WebRecommender

    old_path, _ = _synthetic_index(tmp_path, seed=42)
    legacy = WebRecommender(str(old_path), enhance=True).recommend(0, n=20)
    sonic_path, _ = _synthetic_index(tmp_path, seed=42, sonic=True)
    sonic = WebRecommender(str(sonic_path), enhance=True).recommend(0, n=20)
    legacy_ids = [item["deezer_id"] for item in legacy["results"]]
    sonic_ids = [item["deezer_id"] for item in sonic["results"]]
    assert sonic_ids[:5] == legacy_ids[:5]
    assert sonic_ids[5:] != legacy_ids[5:]


def test_stable_sonic_benchmark_method_uses_serving_ranker(tmp_path):
    from _reco import WebRecommender
    from soundalike.ml.real_benchmark import ProductionRanker

    path, _ = _synthetic_index(tmp_path, seed=43, sonic=True)
    recommender = WebRecommender(str(path), enhance=True)
    expected = recommender.recommend(0, n=20)["results"]
    ranked = ProductionRanker(recommender, heldout=set()).rank(
        0, "stable_sonic", n=20
    )
    assert [int(recommender.track_ids[row]) for row in ranked] == [
        item["deezer_id"] for item in expected
    ]


def test_web_recommender_search_and_findrow(tmp_path):
    from _reco import WebRecommender

    path, _ = _synthetic_index(tmp_path)
    web = WebRecommender(str(path))
    assert web.find_row("song 0", "artist 0") == 0
    assert web.find_row("song 7") == 7  # unambiguous title
    hits = web.search("song 1", limit=5)
    assert hits and all("title" in h and "row" in h for h in hits)


def test_lightweight_search_matches_recommender_without_model_init(tmp_path):
    from _reco import WebRecommender
    from _search import SearchCatalog

    path, _ = _synthetic_index(tmp_path)
    catalog = SearchCatalog.from_npz(str(path))
    recommender = WebRecommender(str(path))

    for query in ("song 1", "artist 7", "song 12 artist 2"):
        assert catalog.search(query, 8) == recommender.search(query, 8)
    assert catalog.find_row("song 7") == recommender.find_row("song 7")


def test_lightweight_search_reuses_normalized_query_cache(tmp_path):
    from _search import SearchCatalog

    path, _ = _synthetic_index(tmp_path)
    catalog = SearchCatalog.from_npz(str(path))
    first = catalog.search("Song 1", 8)
    before = catalog.cache_info()
    second = catalog.search("  song   1  ", 8)
    after = catalog.cache_info()

    assert second == first
    assert after.hits == before.hits + 1


def test_search_catalog_writer_is_deterministic_and_loadable(tmp_path):
    from _search import SearchCatalog, _sha256, write_search_catalog

    path, _ = _synthetic_index(tmp_path)
    first = tmp_path / "first.json.gz"
    second = tmp_path / "second.json.gz"
    first_result = write_search_catalog(path, first)
    second_result = write_search_catalog(path, second)

    assert first.read_bytes() == second.read_bytes()
    assert first_result == second_result
    catalog = SearchCatalog.from_gzip_json(first, _sha256(first))
    assert len(catalog) == 300
    assert catalog.search("artist 4", 3)


def test_packaged_search_catalog_is_bound_to_production_index():
    import json

    from _search import (
        _PACKAGED_CATALOG_PATH,
        _PACKAGED_CATALOG_SHA256,
        _PRODUCTION_LIBRARY_SIZE,
        _sha256,
    )

    assert _PACKAGED_CATALOG_PATH.stat().st_size == 3_961_198
    assert _sha256(_PACKAGED_CATALOG_PATH) == _PACKAGED_CATALOG_SHA256
    assert _PRODUCTION_LIBRARY_SIZE == 272_853
    config = json.loads(
        (_PACKAGED_CATALOG_PATH.parents[1] / "vercel.json").read_text()
    )
    assert (
        config["functions"]["api/search.py"]["includeFiles"]
        == "api/search_catalog.json.gz"
    )


def test_search_endpoint_uses_lightweight_catalog_and_clamps_limit(monkeypatch):
    import search

    calls = []

    class FakeCatalog:
        def search(self, query, limit):
            calls.append((query, limit))
            return [{"row": 1, "title": "Song", "artist": "Artist"}]

    request = search.handler.__new__(search.handler)
    request.path = "/api/search?q=Song&limit=999"
    sent = []
    request._send = lambda code, body: sent.append((code, body))
    monkeypatch.setattr(search, "get_search_catalog", lambda: FakeCatalog())
    request.do_GET()

    assert not hasattr(search, "get_recommender")
    assert calls == [("Song", 20)]
    assert sent[0][0] == 200
    assert sent[0][1]["results"][0]["row"] == 1


def test_autocomplete_client_aborts_stale_requests_and_reuses_cache():
    root = Path(__file__).resolve().parents[1] / "webapp"
    html = (root / "index.html").read_text(encoding="utf-8")
    script = (root / "search.js").read_text(encoding="utf-8")

    assert '<script src="/search.js"></script>' in html
    assert "AbortController" in script
    assert "requestSequence" in script
    assert "cachedPrefix" in script
    assert "requestIdleCallback" in script
    assert 'primaryRecommendationServer = "https://soundalike-api.yassin.app"' in script
    assert 'fallbackRecommendationServer = "https://soundalike.yassin.app"' in script
    assert "primaryRecommendationTimeoutMs = 5000" in script
    assert "/api/spicetify_recommend?" in script


def test_results_include_deezer_id_for_previews(tmp_path):
    # The preview feature needs each result to carry its Deezer track id so the
    # frontend can fetch a 30s preview by id.
    from _reco import WebRecommender

    path, idx = _synthetic_index(tmp_path)
    web = WebRecommender(str(path))
    out = web.recommend(0, n=8)
    assert out["results"], "expected some results"
    for r in out["results"]:
        assert "deezer_id" in r and isinstance(r["deezer_id"], int)


def test_split_query_parsing():
    import recommend as rec
    assert rec._split("Plastic Love — Mariya Takeuchi") == ("Plastic Love", "Mariya Takeuchi")
    assert rec._split("Redbone by Childish Gambino") == ("Redbone", "Childish Gambino")
    assert rec._split("Windowlicker") == ("Windowlicker", "")


def test_norm_keeps_with_and_strips_credits():
    from _reco import _norm
    # 'with' is a normal word — must NOT be stripped (the old bug collapsed it).
    assert _norm("Stay With Me") == "stay with me"
    # parenthetical credits / version suffixes are stripped for matching.
    assert _norm("Master of Puppets (Remastered)") == "master of puppets"
    assert _norm("Idol (From The Idol Vol. 1)") == "idol"
    assert _norm("Song - 2011 Remaster") == "song"
    assert _norm("Track (feat. Someone)") == "track"


def test_search_ranks_and_finds_titles_with_with(tmp_path):
    from _reco import WebRecommender
    from soundalike.ml.deepvibe import DeepVibeIndex
    import numpy as np

    # Tiny hand-made index including a 'with' title and a decoy.
    titles = np.array(["Mayonaka no Door / Stay With Me", "Stay Awake",
                       "Dancing", "Money Machine"], dtype=object)
    artists = np.array(["Miki Matsubara", "Decoy", "Decoy", "100 gecs"], dtype=object)
    idx = DeepVibeIndex(np.array([1, 2, 3, 4]), titles, artists,
                        np.random.default_rng(0).standard_normal((4, 16)).astype("float32"),
                        np.random.default_rng(1).standard_normal((4, 29)).astype("float32"))
    p = tmp_path / "mini.npz"; idx.save(p)
    rec = WebRecommender(str(p))
    # find_row locates the 'with' title (old bug returned None / wrong row).
    assert rec.find_row("Stay With Me", "Miki Matsubara") == 0
    # token search: 'miki stay' surfaces the right song.
    hits = rec.search("miki stay", 3)
    assert hits and hits[0]["artist"] == "Miki Matsubara"
    # a query that is an exact title ranks that title first.
    hits2 = rec.search("money machine", 3)
    assert hits2[0]["title"] == "Money Machine"


def test_search_prioritizes_an_exact_artist_over_title_mentions():
    from _search import SearchCatalog

    catalog = SearchCatalog(
        ["The Weeknd's Dark Secret", "Gone", "Blinding Lights"],
        ["American Dad! Cast", "The Weeknd", "The Weeknd"],
    )
    hits = catalog.search("The Weeknd", 3)

    assert hits[0]["artist"] == "The Weeknd"
    assert hits[1]["artist"] == "The Weeknd"
    assert hits[2]["artist"] == "American Dad! Cast"
    assert catalog.search("weeknd", 1)[0]["artist"] == "The Weeknd"


def test_find_row_prefers_original_over_remix(tmp_path):
    from _reco import WebRecommender
    from soundalike.ml.deepvibe import DeepVibeIndex

    titles = np.array(
        ["Treasure (Sharam Club Remix)", "Treasure", "Other Song"], dtype=object
    )
    artists = np.array(["Bruno Mars", "Bruno Mars", "Other"], dtype=object)
    rng = np.random.default_rng(20)
    index = DeepVibeIndex(
        np.array([1, 2, 3]), titles, artists,
        rng.standard_normal((3, 16)).astype("float32"),
        rng.standard_normal((3, 29)).astype("float32"),
    )
    path = tmp_path / "versions.npz"
    index.save(path)
    recommender = WebRecommender(str(path), enhance=False)
    assert recommender.find_row("Treasure", "Bruno Mars") == 1


def test_hosted_quality_rules_match_desktop_edge_cases():
    from _reco import _TitleQualityFilter
    from soundalike.ml.quality_filter import TitleQualityFilter

    hosted = _TitleQualityFilter()
    desktop = TitleQualityFilter()
    cases = [
        ("Sing Along Version", "Publisher"),
        ("One x Two x Three", "Mashup Artist"),
        ("Tribute Version", "Publisher"),
        ("A Tribute To Someone", "Herbie Hancock"),
        ("Cover Me", "Bruce Springsteen"),
        ("Mashup", "A Legitimate Artist"),
        ("Originally", "The Performers"),
        ("A x B", "Mathematics"),
        ("Song (Cover of Hit)", "Cover Publisher"),
        ("Song", "In the Style of Adele"),
        ("Song - Originally Performed by Adele", "Publisher"),
        ("First Song x Second Song", "DJ"),
        ("Love X Love", "George Benson"),
        ("Pola (The Geek x VRV Remix)", "Jabberwocky"),
    ]
    hosted_mask = hosted.keep_mask(
        [title for title, _ in cases], [artist for _, artist in cases]
    )
    desktop_mask = desktop.keep_mask(
        [title for title, _ in cases], [artist for _, artist in cases]
    )
    assert hosted_mask.tolist() == desktop_mask.tolist() == [
        False, False, False, True, True, False, True, True,
        False, False, False, False, True, True,
    ]


def test_guarded_reranker_can_promote_beyond_requested_n(tmp_path):
    """n=5/diversity=0 must still collect the full guarded top-20 window."""
    from _reco import WebRecommender

    path, _ = _synthetic_index(tmp_path, n_artists=30, per=3, dim=24, seed=21)
    recommender = WebRecommender(str(path), enhance=True)
    baseline = recommender.recommend(
        0, n=20, diversity=0, max_per_artist=0, genre_rerank=False
    )
    target_id = baseline["results"][10]["deezer_id"]
    target_row = int(np.where(recommender.track_ids == target_id)[0][0])

    class PromoteTarget:
        def blend_with_genre(self, blended, *args, **kwargs):
            scores = np.zeros_like(blended)
            scores[target_row] = 1.0
            return scores

    recommender._centroid_idx = PromoteTarget()
    guarded = recommender.recommend(
        0, n=5, diversity=0, max_per_artist=0, genre_rerank=True
    )
    assert guarded["results"][0]["deezer_id"] == target_id
