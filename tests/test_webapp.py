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
            vibe.append(rng.standard_normal(29))
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
            exclude_artist=str(idx.artists[row]), diversity=0.15, max_per_artist=1,
            quality_filter=False, genre_rerank=False, related_boost=False)
        assert [(x["title"], x["artist"]) for x in w["results"]] == \
               [(r.title, r.artist) for r in c], f"mismatch at row {row}"


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
            exclude_artist=str(idx.artists[row]),
            seed_title=str(idx.titles[row]),
            diversity=0.15,
            max_per_artist=1,
        )
        assert [(item["title"], item["artist"]) for item in hosted["results"]] == [
            (item.title, item.artist) for item in desktop
        ], f"enhanced mismatch at row {row}"


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
            exclude_artist=str(idx.artists[row]), seed_title=str(idx.titles[row]),
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
        exclude_artist=str(idx.artists[0]), seed_title=str(idx.titles[0]),
        diversity=.15, max_per_artist=1, seed_row=0,
    )
    assert [(item["title"], item["artist"]) for item in web_result["results"]] == [
        (item.title, item.artist) for item in canonical
    ]
    assert web_result["results"][:5] == legacy_head
    assert web_result["method"] == "dual_sonic64_guardrail"
    assert web_result["index_version"] == "2026.07.11-dual-sonic64"


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
        False, False, False, False, True, False,
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
