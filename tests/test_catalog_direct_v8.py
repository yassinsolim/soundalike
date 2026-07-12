import hashlib
import json

import numpy as np
import pytest

from soundalike.ml.catalog_direct_v8 import (
    DirectListError,
    LOCKED_SEEDS,
    _parser,
    run_direct_lists,
    validate_judgments,
    write_locked_seed_manifest,
)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _toy_run(tmp_path):
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(
            {"tau": 0.55, "sigma": 0.6, "audio_weight": 0.35}
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "seeds.json"
    write_locked_seed_manifest(manifest_path, _sha(policy_path))
    for name in ("index.npz", "graph.npz", "style.npz"):
        (tmp_path / name).write_bytes(name.encode())

    seed_titles = [seed["title"] for seed in LOCKED_SEEDS]
    seed_artists = [seed["artist"] for seed in LOCKED_SEEDS]

    class Rec:
        titles = np.array(
            seed_titles
            + ["Candidate - Karaoke", "Two", "Three", "Four", "Five"]
            + ["Six", "Seven", "Eight", "Nine", "Ten"]
            + ["Safe Candidate"]
        )
        artists = np.array(
            seed_artists
            + ["Junk Artist", "B", "C", "D", "E"]
            + ["F", "G", "H", "I", "J"]
            + ["Safe Artist"]
        )
        track_ids = np.arange(100, 131)
        last_retrieval_mode = "toy-dual-sonic"

    class Styles:
        scene_names = ("rock", "pop")
        direct_mask = np.array([False] * 31)
        confidence = np.array([0.7] * 31)

        def artist_id(self, artist):
            matches = np.flatnonzero(Rec.artists == artist)
            return int(matches[0]) if len(matches) else None

        def artist_vector(self, _artist):
            return np.array([0.8, 0.2], np.float32)

        def style_overlap(self, _left, _right):
            return 0.5

    class Ranker:
        def __init__(self, rec, _graph, styles, policy, quality):
            self.rec, self.styles, self.policy, self.quality = rec, styles, policy, quality

        def audio_scores(self, _row):
            return np.linspace(0.0, 1.0, len(self.rec.titles))

        def recommend(self, query_row, n=5):
            rows = list(range(20, 25))
            fired = query_row % 2 == 0
            if not fired:
                rows = list(range(25, 30))
            elif query_row:
                rows[0] = 30
            return {
                "gate": {
                    "fired": fired,
                    "reason": "both_gates_passed" if fired else "agreement_below_tau",
                    "agreement": 0.75 if fired else 0.4,
                    "consistency": 0.8,
                    "thresholds": {"tau": self.policy.tau, "sigma": self.policy.sigma},
                    "shared_count": 7,
                    "source_coverage": {
                        "lastfm": True,
                        "music4all": True,
                        "lastfm_candidates": 8,
                        "music4all_candidates": 8,
                    },
                },
                "results": [
                    {
                        "position": position,
                        "row": row,
                        "title": str(self.rec.titles[row]),
                        "artist": str(self.rec.artists[row]),
                        "track_id": int(self.rec.track_ids[row]),
                        "rationale": {
                            "G": 0.4,
                            "A": 0.6,
                            "S": 0.5,
                            "lastfm_G": 0.5,
                            "music4all_G": 0.3,
                            "source": (
                                "dual_source_graph" if fired
                                else "production_abstention"
                            ),
                            "query_mode": "toy-graph",
                        },
                    }
                    for position, row in enumerate(rows[:n], 1)
                ]
            }

    class Production:
        def __init__(self, _rec, _heldout):
            pass

        def rank(self, _row, method, n=5):
            assert method == "dual_sonic"
            return list(range(25, 25 + n))

    def preview(track_id, _title, _artist):
        return {
            "url": "https://preview/%s" % track_id if track_id % 2 else "",
            "status": "available" if track_id % 2 else "missing",
        }

    report = run_direct_lists(
        manifest_path,
        _sha(manifest_path),
        policy_path,
        tmp_path / "index.npz",
        tmp_path / "graph.npz",
        tmp_path / "style.npz",
        recommender_factory=lambda _path: Rec(),
        graph_factory=lambda _path: object(),
        style_factory=lambda _path: Styles(),
        ranker_factory=Ranker,
        production_factory=Production,
        preview_lookup=preview,
    )
    return report


def test_locked_seed_set_has_exact_distinct_count_and_broad_scenes(tmp_path):
    expected = [
        ("Pixies", "Where Is My Mind?"),
        ("Anri", "Last Summer Whisper"),
        ("Miki Matsubara", "Mayonaka no Door / Stay With Me"),
        ("Kali Uchis", "telepatía"),
        ("Bad Bunny", "Tití Me Preguntó"),
        ("100 gecs", "money machine"),
        ("brakence", "rosier/punk2"),
        ("glaive", "astrid"),
        ("Daft Punk", "Digital Love"),
        ("Gorillaz", "Clint Eastwood"),
        ("Massive Attack", "Teardrop"),
        ("my bloody valentine", "Sometimes"),
        ("Deftones", "Be Quiet and Drive (Far Away)"),
        ("A Tribe Called Quest", "Electric Relaxation"),
        ("Frank Ocean", "Nights"),
        ("Metallica", "Orion (Remastered)"),
        ("Miles Davis", "So What (Album Version)"),
        ("Burna Boy", "Ye"),
        ("NewJeans", "Super Shy"),
        ("FKA twigs", "cellophane"),
    ]
    assert len(LOCKED_SEEDS) == 20
    assert [(seed["artist"], seed["title"]) for seed in LOCKED_SEEDS] == expected
    assert len({seed["scene"] for seed in LOCKED_SEEDS}) >= 12
    classes = [seed["failure_class"] for seed in LOCKED_SEEDS]
    assert classes.count("city_pop") >= 2
    assert classes.count("latin") >= 2
    assert classes.count("hyperpop_digicore") >= 3
    assert {
        "daft_punk", "gorillaz", "pixies_to_trip_hop", "rap", "rnb",
        "shoegaze", "metal", "jazz", "afrobeats", "k_pop",
        "art_pop",
    } <= set(classes)
    scenes = {seed["scene"] for seed in LOCKED_SEEDS}
    assert {
        "electronic", "jazz_rap", "alternative_rnb", "shoegaze",
        "alternative_metal", "thrash_metal", "modal_jazz", "afrobeats",
        "k_pop", "art_pop",
    } <= scenes
    manifest = write_locked_seed_manifest(tmp_path / "lock.json", "a" * 64)
    assert manifest["seed_count"] == 20
    assert manifest["results_inspected"] is False
    assert manifest["target_labels_included"] is False
    assert manifest["fresh_final_identities_included"] is False
    assert manifest["result_outputs_at_lock"] == "unreviewed"
    assert "not FINAL labels" in manifest["output_status"]
    assert manifest["policy_manifest_sha256"] == "a" * 64
    assert manifest["inspection_rules"]["required_seed_passes"] == 16
    assert manifest["inspection_rules"]["positions_1_to_3"] == "no unrelated result"
    assert manifest["inspection_rules"]["coherent_results_required_per_list"] == 4
    assert len(manifest["inspection_rules"]["automatic_seed_failure"]) == 3
    assert manifest["content_sha256"]


def test_lists_requires_lock_and_exact_manifest_and_policy_hash(tmp_path):
    with pytest.raises(DirectListError, match="lock-seeds"):
        run_direct_lists(
            tmp_path / "missing.json", "0" * 64, {}, "i", "g", "s"
        )

    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"tau": 0.5, "sigma": 0.5}))
    manifest = tmp_path / "manifest.json"
    write_locked_seed_manifest(manifest, _sha(policy))
    with pytest.raises(DirectListError, match="three policy"):
        run_direct_lists(manifest, _sha(manifest), policy, "i", "g", "s")

    policy.write_text(
        json.dumps(
            {
                "tau": 0.5,
                "sigma": 0.5,
                "audio_weight": 0.3,
                "fourth": 4,
            }
        )
    )
    with pytest.raises(DirectListError, match="policy manifest hash"):
        run_direct_lists(manifest, _sha(manifest), policy, "i", "g", "s")


def test_toy_lists_have_rationales_flags_styles_and_preview_status(tmp_path):
    report = _toy_run(tmp_path)
    assert len(report["records"]) == 20
    assert report["target_blind_disclosure"]["human_judgments_included"] is False
    record = report["records"][0]
    assert record["resolution"]["resolver"] == "PairResolver"
    result = record["lists"]["catalog_policy"][0]
    assert {
        "G", "A", "S", "lastfm_G", "music4all_G", "source", "query_mode"
    } <= set(result["rationale"])
    assert result["rationale"]["A_definition"].startswith("audio-derived")
    assert result["style"]["source"] == "audio_propagated"
    assert set(result["flags"]) == {"junk", "duplicate", "seed_variant", "same_artist"}
    assert result["flags"]["junk"]
    statuses = {
        item["preview_status"]
        for values in record["lists"].values()
        for item in values
    }
    assert statuses == {"available", "missing"}
    assert report["method"]["formula"]
    assert report["method"]["parameters"] == ["tau", "sigma", "audio_weight"]
    method_text = report["method"]["formula"] + report["provenance"]["policy"]
    assert "Production-default abstention" in method_text
    assert "independent-source" in method_text
    assert "consistency" in method_text
    assert "frozen equal source-graph blend" in method_text
    assert "single audio tie-break" in method_text
    assert "style_weight" not in method_text
    assert "guard" not in method_text
    assert report["provenance"]["assets"]["index"]["sha256"]
    assert record["gate"] == {
        "fired": True,
        "abstained": False,
        "reason": "both_gates_passed",
        "agreement": 0.75,
        "consistency": 0.8,
        "thresholds": {"tau": 0.55, "sigma": 0.6},
        "shared_count": 7,
        "source_coverage": {
            "lastfm": True,
            "music4all": True,
            "lastfm_candidates": 8,
            "music4all_candidates": 8,
        },
    }
    abstained = report["records"][1]
    assert abstained["gate"]["abstained"]
    assert [item["row"] for item in abstained["lists"]["catalog_policy"]] == [
        item["row"] for item in abstained["lists"]["current_production_dual_sonic"]
    ]
    assert [
        (item["title"], item["artist"])
        for item in abstained["lists"]["catalog_policy"]
    ] == [
        (item["title"], item["artist"])
        for item in abstained["lists"]["current_production_dual_sonic"]
    ]


def _judgments(report):
    values = []
    for record in report["records"]:
        positions = []
        for list_name, results in record["lists"].items():
            positions.extend(
                {
                    "list": list_name,
                    "position": item["position"],
                    "title": item["title"],
                    "artist": item["artist"],
                    "rationale": "Listened and recorded a human position assessment.",
                    "junk": False,
                    "junk_evidence": "No karaoke, tribute, duplicate, or seed variant evidence.",
                }
                for item in results
            )
        values.append(
            {
                "id": record["id"],
                "challenger_pass": True,
                "production_pass": True,
                "positions": positions,
            }
        )
    return {"lists_sha256": report["content_sha256"], "judgments": values}


def test_judgment_gate_uses_human_bool_and_automatic_junk_failure(tmp_path):
    report = _toy_run(tmp_path)
    judgments = _judgments(report)
    result = validate_judgments(report, judgments)
    assert result["challenger_effective_passes"] == 19
    assert result["production_effective_passes"] == 20
    assert result["gate_met"]
    assert result["per_seed"][0]["automatic_failure"]["challenger"]
    assert not result["per_seed"][0]["automatic_failure"]["production"]
    assert not result["coherence_inferred"]
    assert result["review_evidence_disclosure"]["position_rationales_included"]
    assert result["review_evidence_disclosure"]["position_junk_evidence_included"]
    assert sum(
        counts["fired"] + counts["abstained"]
        for counts in result["gate_summary_by_failure_class"].values()
    ) == 20

    for judgment in judgments["judgments"][1:5]:
        judgment["challenger_pass"] = False
    judgments["judgments"][5]["production_pass"] = False
    result = validate_judgments(report, judgments)
    assert result["challenger_effective_passes"] == 15
    assert result["production_effective_passes"] == 19
    assert not result["gate_met"]

    judgments["judgments"][5]["positions"][0]["artist"] = "not inspected"
    with pytest.raises(DirectListError, match="names"):
        validate_judgments(report, judgments)


@pytest.mark.parametrize("flag", ["junk", "duplicate", "seed_variant"])
def test_each_automatic_flag_fails_only_its_method(tmp_path, flag):
    report = _toy_run(tmp_path)
    report["records"][1]["lists"]["current_production_dual_sonic"][0]["flags"][
        flag
    ] = True
    unsigned = dict(report)
    unsigned.pop("content_sha256")
    report["content_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()
    judgments = _judgments(report)
    result = validate_judgments(report, judgments)
    assert result["challenger_effective_passes"] == 19
    assert result["production_effective_passes"] == 19
    assert result["per_seed"][1]["automatic_failure"]["production"]


def test_judgments_require_both_passes_and_position_evidence(tmp_path):
    report = _toy_run(tmp_path)
    judgments = _judgments(report)
    del judgments["judgments"][0]["production_pass"]
    with pytest.raises(DirectListError, match="production_pass"):
        validate_judgments(report, judgments)

    judgments = _judgments(report)
    del judgments["judgments"][0]["positions"][0]["junk_evidence"]
    with pytest.raises(DirectListError, match="incomplete"):
        validate_judgments(report, judgments)


def test_cli_exposes_only_requested_operations():
    action = next(item for item in _parser()._actions if item.dest == "command")
    assert set(action.choices) == {"lock-seeds", "lists", "validate"}
